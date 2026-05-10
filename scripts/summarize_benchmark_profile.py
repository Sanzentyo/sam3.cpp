# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


TABLE_ROW_RE = re.compile(
    r"^\s*\d+\s*\|\s*(?P<model>\S+)\s+\|[^\n]*?\|\s*(?P<backend>CUDA|CPU)\s*\|"
    r"\s*(?P<load>[0-9.]+)\s*\|\s*(?P<init>[0-9.]+)\s*\|"
    r"\s*(?P<track>[0-9.]+)\s*\|\s*(?P<p50>[0-9.]+)\s*\|"
    r"\s*(?P<p95>[0-9.]+)\s*\|\s*(?P<total>[0-9.]+)",
    re.M,
)
ENCODE_RE = re.compile(r"sam2_encode_image_hiera: SAM2 image encoded in ([0-9.]+) ms")
COMPUTE_RE = re.compile(r"SAM3_PROFILE compute backend=(\S+) nodes=(\d+) ms=([0-9.]+)")
TENSOR_SET_RE = re.compile(r"SAM3_PROFILE tensor_set name=\S+ bytes=\d+ offset=\d+ ms=([0-9.]+)")
TENSOR_GET_RE = re.compile(r"SAM3_PROFILE tensor_get name=\S+ bytes=\d+ offset=\d+ ms=([0-9.]+)")
CPU_SPAN_RE = re.compile(r"SAM3_PROFILE cpu name=(\S+) ms=([0-9.]+)")
HIERA_CUT_RE = re.compile(
    r"SAM3_PROFILE_HIERA_CUT label=(\S+) nodes=(\d+) mean_ms=([0-9.]+) warmup=(\d+) iter=(\d+)"
)
HIERA_CUT_OPS_RE = re.compile(
    r"SAM3_PROFILE_HIERA_CUT_OPS label=(\S+) op=(\S+) count=(\d+) out_elements=(\d+)"
)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    row = TABLE_ROW_RE.search(text)
    encodes = [float(value) for value in ENCODE_RE.findall(text)]
    computes = [(backend, int(nodes), float(ms)) for backend, nodes, ms in COMPUTE_RE.findall(text)]
    tensor_sets = [float(value) for value in TENSOR_SET_RE.findall(text)]
    tensor_gets = [float(value) for value in TENSOR_GET_RE.findall(text)]
    cpu_spans: dict[str, list[float]] = {}
    for name, ms in CPU_SPAN_RE.findall(text):
        cpu_spans.setdefault(name, []).append(float(ms))
    hiera_cuts: dict[str, list[dict[str, Any]]] = {}
    for label, nodes, ms, warmup, iters in HIERA_CUT_RE.findall(text):
        hiera_cuts.setdefault(label, []).append(
            {
                "nodes": int(nodes),
                "mean_ms": float(ms),
                "warmup": int(warmup),
                "iter": int(iters),
            }
        )
    hiera_cut_ops: dict[str, dict[str, dict[str, int | float]]] = {}
    for label, op, count, elements in HIERA_CUT_OPS_RE.findall(text):
        op_stats = hiera_cut_ops.setdefault(label, {}).setdefault(
            op, {"samples": 0, "count": 0, "out_elements": 0}
        )
        op_stats["samples"] += 1
        op_stats["count"] += int(count)
        op_stats["out_elements"] += int(elements)
    for ops in hiera_cut_ops.values():
        for op_stats in ops.values():
            samples = int(op_stats["samples"])
            op_stats["mean_count"] = op_stats["count"] / samples
            op_stats["mean_out_elements"] = op_stats["out_elements"] / samples

    result: dict[str, Any] = {
        "path": str(path),
        "encode_ms": encodes,
        "encode_mean_ms": mean(encodes),
        "profile_compute_count": len(computes),
        "profile_compute_sum_ms": sum(ms for _, _, ms in computes),
        "profile_compute_top": [
            {"backend": backend, "nodes": nodes, "ms": ms}
            for backend, nodes, ms in sorted(computes, key=lambda item: item[2], reverse=True)[:5]
        ],
        "tensor_set_sum_ms": sum(tensor_sets),
        "tensor_get_sum_ms": sum(tensor_gets),
        "tensor_get_count": len(tensor_gets),
        "cpu_spans": {
            name: {
                "count": len(values),
                "sum_ms": sum(values),
                "mean_ms": mean(values),
                "values_ms": values,
            }
            for name, values in sorted(cpu_spans.items())
        },
        "hiera_cuts": {
            label: {
                "count": len(values),
                "mean_ms": mean([row["mean_ms"] for row in values]),
                "values": values,
            }
            for label, values in sorted(hiera_cuts.items())
        },
        "hiera_cut_ops": {
            label: {
                op: stats
                for op, stats in sorted(ops.items(), key=lambda item: item[0])
            }
            for label, ops in sorted(hiera_cut_ops.items(), key=lambda item: item[0])
        },
    }
    if row:
        result.update(
            {
                "model": row.group("model"),
                "backend": row.group("backend"),
                "load_ms": float(row.group("load")),
                "init_ms": float(row.group("init")),
                "track_ms": float(row.group("track")),
                "p50_ms": float(row.group("p50")),
                "p95_ms": float(row.group("p95")),
                "total_ms": float(row.group("total")),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = {"logs": [summarize(path) for path in args.logs]}
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
