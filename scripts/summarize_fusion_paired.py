#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


MEAN_RE = re.compile(r"\bmean_ms=(?P<mean>[0-9.]+)")
MEDIAN_RE = re.compile(r"\bmedian_ms=(?P<median>[0-9.]+)")


def parse_value(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    mean = MEAN_RE.search(text)
    median = MEDIAN_RE.search(text)
    if mean is None or median is None:
        return None
    return {
        "path": str(path),
        "mean_ms": float(mean.group("mean")),
        "median_ms": float(median.group("median")),
    }


def summarize_rows(rows: list[dict[str, Any]], drop_first: bool) -> dict[str, Any]:
    used = rows[1:] if drop_first and len(rows) > 1 else rows
    means = [row["mean_ms"] for row in used]
    medians = [row["median_ms"] for row in used]
    return {
        "rows": rows,
        "used_rows": used,
        "drop_first": drop_first and len(rows) > 1,
        "count": len(used),
        "mean_of_means_ms": statistics.fmean(means) if means else None,
        "median_of_means_ms": statistics.median(means) if means else None,
        "mean_of_medians_ms": statistics.fmean(medians) if medians else None,
        "median_of_medians_ms": statistics.median(medians) if medians else None,
    }


def collect(directory: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"{prefix}_*.log")):
        row = parse_value(path)
        if row is not None:
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize paired fusion-kernel microbench logs.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--drop-first", action="store_true")
    args = parser.parse_args()

    default = collect(args.directory, "default")
    prequant = collect(args.directory, "prequant")
    summary = {
        "directory": str(args.directory),
        "default": summarize_rows(default, args.drop_first),
        "prequant": summarize_rows(prequant, args.drop_first),
    }
    default_mean = summary["default"]["mean_of_means_ms"]
    prequant_mean = summary["prequant"]["mean_of_means_ms"]
    if default_mean is not None and prequant_mean is not None:
        summary["prequant_minus_default_mean_ms"] = prequant_mean - default_mean
        summary["prequant_speedup_vs_default_pct"] = (default_mean / prequant_mean - 1.0) * 100.0

    text = json.dumps(summary, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
