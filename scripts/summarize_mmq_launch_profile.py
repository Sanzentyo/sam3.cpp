#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LINE_RE = re.compile(r"GGML_CUDA_PROFILE_MMQ_LAUNCH (?P<body>.*)$")


def parse_kv_body(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in body.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def to_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "p50": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("type", ""),
            row.get("ncols_x", ""),
            row.get("nrows_x", ""),
            row.get("ncols_dst", ""),
            row.get("ncols_max", ""),
            row.get("nchannels", ""),
            row.get("nsamples", ""),
            row.get("mmq_x", ""),
            row.get("mmq_y", ""),
            row.get("stream_k", ""),
            row.get("fixup", ""),
        )
        groups[key].append(row)

    by_shape: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        (
            typ,
            ncols_x,
            nrows_x,
            ncols_dst,
            ncols_max,
            nchannels,
            nsamples,
            mmq_x,
            mmq_y,
            stream_k,
            fixup,
        ) = key
        efficiencies = [to_int(row, "efficiency") for row in group_rows if "efficiency" in row]
        stream_blocks = [to_int(row, "stream_blocks") for row in group_rows if "stream_blocks" in row]
        ntiles_dst = [to_int(row, "ntiles_dst") for row in group_rows if "ntiles_dst" in row]
        by_shape.append(
            {
                "shape": (
                    f"type={typ} src0=[{ncols_x},{nrows_x}] dst=[{nrows_x},{ncols_dst}] "
                    f"ncols_max={ncols_max} nchannels={nchannels} nsamples={nsamples}"
                ),
                "count": len(group_rows),
                "mmq_x": int(mmq_x),
                "mmq_y": int(mmq_y),
                "stream_k": int(stream_k),
                "fixup": int(fixup),
                "shared_bytes": int(group_rows[0].get("shared", "0")),
                "ntiles_dst": stats(ntiles_dst),
                "stream_blocks": stats(stream_blocks),
                "efficiency": stats(efficiencies),
            }
        )

    by_shape.sort(key=lambda item: (-item["count"], item["shape"]))
    return {
        "rows": len(rows),
        "by_shape": by_shape,
        "fixup_rows": sum(1 for row in rows if row.get("fixup") == "1"),
        "stream_k_rows": sum(1 for row in rows if row.get("stream_k") == "1"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GGML_CUDA_PROFILE_MMQ_LAUNCH logs.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for log_path in args.logs:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = LINE_RE.search(line)
            if match:
                rows.append(parse_kv_body(match.group("body")))

    summary = summarize(rows)
    summary["sources"] = [str(path) for path in args.logs]

    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
