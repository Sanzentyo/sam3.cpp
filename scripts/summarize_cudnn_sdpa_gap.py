# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FATTN_RE = re.compile(
    r"GGML_CUDA_PROFILE_FATTN "
    r"(?:name=(?P<name>\S+) )?"
    r"Q_convert_ms=(?P<q_convert>[0-9.]+) "
    r"K_convert_ms=(?P<k_convert>[0-9.]+) "
    r"V_convert_ms=(?P<v_convert>[0-9.]+) "
    r"kernel_ms=(?P<kernel>[0-9.]+) "
    r"fixup_ms=(?P<fixup>[0-9.]+) "
    r"compute_ms=(?P<compute>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+).* "
    r"Q=\[(?P<head_dim>[0-9]+),(?P<seq>[0-9]+),(?P<heads>[0-9]+),(?P<batch>[0-9]+)\] .* "
    r"types=(?P<types>[a-z0-9/]+)"
)

CUDNN_RE = re.compile(
    r"backend=cudnn-sdpa(?:-bf16| "
    r"io_dtype=(?P<io_dtype>[a-z0-9]+) "
    r"out_dtype=(?P<out_dtype>[a-z0-9]+)) "
    r"batch=(?P<batch>[0-9]+) "
    r"heads=(?P<heads>[0-9]+) "
    r"seq=(?P<seq>[0-9]+) "
    r"head_dim=(?P<head_dim>[0-9]+) "
    r"(?:q_seq=(?P<q_seq>[0-9]+) )?"
    r"(?:kv_seq=(?P<kv_seq>[0-9]+) )?"
    r"stats=(?P<stats>[01]) "
    r"(?:convert_f32_inputs=(?P<convert_f32_inputs>[01]) )?"
    r"mean_ms=(?P<mean>[0-9.]+) "
    r"workspace_bytes=(?P<workspace>[0-9]+)"
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((len(ordered) - 1) * pct)), len(ordered) - 1)
    return ordered[idx]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def shape_key(batch: int, heads: int, seq: int, head_dim: int) -> str:
    return f"batch={batch} heads={heads} seq={seq} head_dim={head_dim}"


def parse_fattn(paths: list[Path], drop_first_per_shape: bool) -> dict[str, Any]:
    rows_by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = FATTN_RE.search(line)
            if not match:
                continue
            groups = match.groupdict()
            key = shape_key(
                batch=int(groups["batch"]),
                heads=int(groups["heads"]),
                seq=int(groups["seq"]),
                head_dim=int(groups["head_dim"]),
            )
            rows_by_shape[key].append(
                {
                    "q_convert_ms": float(groups["q_convert"]),
                    "k_convert_ms": float(groups["k_convert"]),
                    "v_convert_ms": float(groups["v_convert"]),
                    "kernel_ms": float(groups["kernel"]),
                    "fixup_ms": float(groups["fixup"]),
                    "compute_ms": float(groups["compute"]),
                    "total_ms": float(groups["total"]),
                    "types": groups["types"],
                    "name": groups.get("name") or "",
                    "source": str(path),
                }
            )

    result: dict[str, Any] = {}
    for key, rows in rows_by_shape.items():
        effective = rows[1:] if drop_first_per_shape and len(rows) > 1 else rows
        result[key] = {
            "count": len(rows),
            "effective_count": len(effective),
            "drop_first_per_shape": drop_first_per_shape,
            "total_ms": stats([row["total_ms"] for row in effective]),
            "kernel_ms": stats([row["kernel_ms"] for row in effective]),
            "convert_ms": stats(
                [
                    row["q_convert_ms"] + row["k_convert_ms"] + row["v_convert_ms"]
                    for row in effective
                ]
            ),
            "types": sorted({row["types"] for row in effective}),
        }
    return result


def parse_cudnn(paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = CUDNN_RE.search(line)
            if not match:
                continue
            groups = match.groupdict()
            key = shape_key(
                batch=int(groups["batch"]),
                heads=int(groups["heads"]),
                seq=int(groups["seq"]),
                head_dim=int(groups["head_dim"]),
            )
            result[key] = {
                "mean_ms": float(groups["mean"]),
                "workspace_bytes": int(groups["workspace"]),
                "stats": groups["stats"] == "1",
                "convert_f32_inputs": groups.get("convert_f32_inputs") == "1",
                "io_dtype": groups.get("io_dtype") or "bf16",
                "out_dtype": groups.get("out_dtype") or "bf16",
                "q_seq": int(groups["q_seq"] or groups["seq"]),
                "kv_seq": int(groups["kv_seq"] or groups["seq"]),
                "source": str(path),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ggml FATTN profile logs with cuDNN SDPA head64 bench logs."
    )
    parser.add_argument("--fattn", type=Path, nargs="+", required=True)
    parser.add_argument("--cudnn", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--drop-first-per-shape", action="store_true")
    parser.add_argument("--global-calls", type=int, default=12)
    parser.add_argument("--window-calls", type=int, default=84)
    args = parser.parse_args()

    fattn = parse_fattn(args.fattn, args.drop_first_per_shape)
    cudnn = parse_cudnn(args.cudnn)

    comparisons: list[dict[str, Any]] = []
    call_count_by_seq_batch = {
        (5184, 1): args.global_calls,
        (576, 9): args.window_calls,
    }
    for key, cudnn_row in cudnn.items():
        shape_match = re.search(
            r"batch=([0-9]+) heads=([0-9]+) seq=([0-9]+) head_dim=([0-9]+)", key
        )
        if not shape_match:
            continue
        batch = int(shape_match.group(1))
        seq = int(shape_match.group(3))
        calls = call_count_by_seq_batch.get((seq, batch), 0)
        fattn_row = fattn.get(key)
        effective_calls = (
            int(fattn_row["total_ms"]["count"])
            if fattn_row and isinstance(fattn_row.get("total_ms"), dict)
            else 0
        )
        comparisons.append(
            {
                "shape": key,
                "sam3_vit_calls": calls,
                "profile_effective_calls": effective_calls,
                "ggml_fattn_total_ms": fattn_row["total_ms"]["sum"]
                if fattn_row
                else None,
                "ggml_fattn_mean_ms": fattn_row["total_ms"]["mean"]
                if fattn_row
                else None,
                "cudnn_mean_ms": cudnn_row["mean_ms"],
                "cudnn_projected_total_ms": cudnn_row["mean_ms"] * calls
                if calls
                else None,
                "cudnn_projected_effective_total_ms": cudnn_row["mean_ms"]
                * effective_calls
                if effective_calls
                else None,
                "projected_delta_ms": (
                    cudnn_row["mean_ms"] * calls - fattn_row["total_ms"]["sum"]
                    if calls and fattn_row
                    else None
                ),
                "projected_effective_delta_ms": (
                    cudnn_row["mean_ms"] * effective_calls
                    - fattn_row["total_ms"]["sum"]
                    if effective_calls and fattn_row
                    else None
                ),
            }
        )

    result = {
        "fattn_sources": [str(path) for path in args.fattn],
        "cudnn_sources": [str(path) for path in args.cudnn],
        "fattn": fattn,
        "cudnn": cudnn,
        "comparisons": comparisons,
    }

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
