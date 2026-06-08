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


MMQ_RE = re.compile(
    r"GGML_CUDA_PROFILE_MMQ "
    r"(?:(?P<cached>cached_src1=1) )?"
    r"quant_ms=(?P<quant>[0-9.]+) "
    r"mmq_ms=(?P<mmq>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+) "
    r"stream_k=(?P<stream_k>[01]) "
    r"native_fp4=(?P<native_fp4>[01]) "
    r"src0_type=(?P<src0_type>\S+) "
    r"src0_ne=(?P<src0_ne>[0-9,]+) "
    r"src1_ne=(?P<src1_ne>[0-9,]+) "
    r"dst_ne=(?P<dst_ne>[0-9,]+) "
    r"src0_name=(?P<src0_name>.*?) "
    r"src1_name=(?P<src1_name>.*?) "
    r"src1_ptr=(?P<src1_ptr>\S+) "
    r"name=(?P<name>.*)$"
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((len(ordered) - 1) * pct)), len(ordered) - 1)
    return ordered[idx]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "sum": sum(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def parse_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MMQ_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        rows.append(
            {
                "source": str(path),
                "cached": groups["cached"] is not None,
                "quant_ms": float(groups["quant"]),
                "mmq_ms": float(groups["mmq"]),
                "total_ms": float(groups["total"]),
                "stream_k": groups["stream_k"] == "1",
                "native_fp4": groups["native_fp4"] == "1",
                "src0_type": groups["src0_type"],
                "src0_ne": groups["src0_ne"],
                "src1_ne": groups["src1_ne"],
                "dst_ne": groups["dst_ne"],
                "src0_name": groups["src0_name"].strip(),
                "src1_name": groups["src1_name"].strip(),
                "name": groups["name"].strip(),
            }
        )
    return rows


def summarize(logs: list[Path]) -> dict[str, Any]:
    rows = [row for path in logs for row in parse_rows(path)]
    mlp_fc2 = [
        row
        for row in rows
        if row["src0_type"] == "q4_1" and ".mlp.fc2.weight" in str(row["src0_name"])
    ]
    cached = [row for row in mlp_fc2 if row["cached"]]
    uncached = [row for row in mlp_fc2 if not row["cached"]]

    by_shape: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"cached": [], "uncached": []}
    )
    for row in mlp_fc2:
        bucket = "cached" if row["cached"] else "uncached"
        by_shape[(row["src0_ne"], row["src1_ne"], row["dst_ne"])][bucket].append(row)

    shape_rows: list[dict[str, Any]] = []
    for (src0_ne, src1_ne, dst_ne), grouped in by_shape.items():
        shape_rows.append(
            {
                "src0_ne": src0_ne,
                "src1_ne": src1_ne,
                "dst_ne": dst_ne,
                "cached_count": len(grouped["cached"]),
                "uncached_count": len(grouped["uncached"]),
                "cached_quant_ms": stats([float(row["quant_ms"]) for row in grouped["cached"]]),
                "uncached_quant_ms": stats([float(row["quant_ms"]) for row in grouped["uncached"]]),
                "cached_total_ms": stats([float(row["total_ms"]) for row in grouped["cached"]]),
                "uncached_total_ms": stats([float(row["total_ms"]) for row in grouped["uncached"]]),
            }
        )
    shape_rows.sort(
        key=lambda row: max(row["cached_quant_ms"]["sum"], row["uncached_quant_ms"]["sum"]),
        reverse=True,
    )

    return {
        "sources": [str(path) for path in logs],
        "note": (
            "Summarizes q4_1 Hiera MLP fc2 MMQ rows. cached_quant_ms near zero "
            "confirms consumer-side quantization is skipped by the experimental "
            "prequant cache; this does not include the producer-side standalone "
            "quantization cost added to populate the cache."
        ),
        "fc2_count": len(mlp_fc2),
        "cached_count": len(cached),
        "uncached_count": len(uncached),
        "cached_quant_ms": stats([float(row["quant_ms"]) for row in cached]),
        "uncached_quant_ms": stats([float(row["quant_ms"]) for row in uncached]),
        "cached_total_ms": stats([float(row["total_ms"]) for row in cached]),
        "uncached_total_ms": stats([float(row["total_ms"]) for row in uncached]),
        "by_shape": shape_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = summarize(args.logs)
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
