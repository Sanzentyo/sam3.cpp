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


PROFILE_RE = re.compile(
    r"GGML_CUDA_PROFILE_FATTN56 "
    r"pack_ms=(?P<pack>[0-9.]+) "
    r"mma_ms=(?P<mma>[0-9.]+) "
    r"slice_ms=(?P<slice>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+) "
    r"Q=\[(?P<q>[0-9,]+)\] .* "
    r"K=\[(?P<k>[0-9,]+)\] .* "
    r"V=\[(?P<v>[0-9,]+)\] .* "
    r"qkv_contiguous=(?P<contiguous>[01]) "
    r"dst_contiguous=(?P<dst_contiguous>[01]) "
    r"combined=(?P<combined>[01])"
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((len(ordered) - 1) * pct)), len(ordered) - 1)
    return ordered[idx]


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "sum": sum(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def shape_key(row: dict[str, Any]) -> str:
    return (
        f"Q[{row['q']}] K[{row['k']}] V[{row['v']}] "
        f"combined={row['combined']} qkv_contiguous={row['qkv_contiguous']}"
    )


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROFILE_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        rows.append(
            {
                "path": str(path),
                "pack_ms": float(groups["pack"]),
                "mma_ms": float(groups["mma"]),
                "slice_ms": float(groups["slice"]),
                "total_ms": float(groups["total"]),
                "q": groups["q"],
                "k": groups["k"],
                "v": groups["v"],
                "qkv_contiguous": groups["contiguous"] == "1",
                "dst_contiguous": groups["dst_contiguous"] == "1",
                "combined": groups["combined"] == "1",
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], drop_first_per_shape: bool) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[shape_key(row)].append(row)

    by_shape: list[dict[str, Any]] = []
    effective_all_rows: list[dict[str, Any]] = []
    for key, shape_rows in groups.items():
        effective_rows = shape_rows[1:] if drop_first_per_shape and len(shape_rows) > 1 else shape_rows
        effective_all_rows.extend(effective_rows)
        pack = [float(row["pack_ms"]) for row in effective_rows]
        mma = [float(row["mma_ms"]) for row in effective_rows]
        slice_ = [float(row["slice_ms"]) for row in effective_rows]
        total = [float(row["total_ms"]) for row in effective_rows]
        by_shape.append(
            {
                "shape": key,
                "count": len(shape_rows),
                "effective_count": len(effective_rows),
                "pack_ms": summarize_values(pack),
                "mma_ms": summarize_values(mma),
                "slice_ms": summarize_values(slice_),
                "total_ms": summarize_values(total),
            }
        )

    by_shape.sort(key=lambda row: row["total_ms"]["sum"], reverse=True)
    return {
        "rows": len(rows),
        "effective_rows": len(effective_all_rows),
        "drop_first_per_shape": drop_first_per_shape,
        "totals": {
            "pack_ms": summarize_values([float(row["pack_ms"]) for row in effective_all_rows]),
            "mma_ms": summarize_values([float(row["mma_ms"]) for row in effective_all_rows]),
            "slice_ms": summarize_values([float(row["slice_ms"]) for row in effective_all_rows]),
            "total_ms": summarize_values([float(row["total_ms"]) for row in effective_all_rows]),
        },
        "by_shape": by_shape,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--drop-first-per-shape", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.logs:
        rows.extend(parse_log(path))

    result = {
        "sources": [str(path) for path in args.logs],
        "note": "FATTN56 profiling synchronizes with CUDA events and should be used for attribution, not normal runtime.",
        **summarize(rows, args.drop_first_per_shape),
    }

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
