#!/usr/bin/env python3
"""Summarize ggml CUDA cuBLASLt bias-fusion timing logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TIMING_RE = re.compile(
    r"GGML_CUDA_CUBLASLT_BIAS_TIMING "
    r"dst=(?P<dst>\S+) .*?dst_type=(?P<dst_type>\S+) "
    r"ne=\[(?P<ne0>\d+),(?P<ne1>\d+),(?P<ne2>\d+),(?P<ne3>\d+)\] "
    r"convert_ms=(?P<convert>[0-9.]+) "
    r"matmul_ms=(?P<matmul>[0-9.]+) "
    r"dst_convert_ms=(?P<dst_convert>[0-9.]+) "
    r"algo_index=(?P<algo>\d+) returned=(?P<returned>\d+)"
)
SUCCESS_RE = re.compile(
    r"GGML_CUDA_CUBLASLT_BIAS_FUSION success "
    r"dst=(?P<dst>\S+) src0=(?P<src0>.*?) src1=.*? "
    r"bias=(?P<bias>\S+) .*?src0_type=(?P<src0_type>\S+) "
    r"src1_type=(?P<src1_type>\S+) .*?gelu=(?P<gelu>[01])"
)


def classify(src0: str) -> str:
    if src0.startswith("vit.blocks."):
        if ".attn.qkv." in src0:
            return "vit_qkv"
        if ".attn.proj." in src0:
            return "vit_attn_proj"
        if ".mlp.lin1." in src0:
            return "vit_mlp_fc1"
        if ".mlp.lin2." in src0:
            return "vit_mlp_fc2"
        return "vit_other"
    if src0.startswith("text.blocks."):
        return "text"
    if src0.startswith("fenc."):
        return "fenc"
    if src0.startswith("ddec."):
        return "ddec"
    if src0.startswith("mem_attn."):
        return "mem_attn"
    if src0.startswith("mem_enc."):
        return "mem_enc"
    if src0.startswith("sam_dec."):
        return "sam_dec"
    if src0.startswith("seg."):
        return "seg"
    return src0.split(".", 1)[0] if src0 else "unknown"


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matmul = [float(row["matmul_ms"]) for row in rows]
    convert = [float(row["convert_ms"]) for row in rows]
    dst_convert = [float(row["dst_convert_ms"]) for row in rows]
    return {
        "n": len(rows),
        "matmul_sum_ms": sum(matmul),
        "matmul_mean_ms": statistics.mean(matmul),
        "matmul_median_ms": statistics.median(matmul),
        "convert_sum_ms": sum(convert),
        "dst_convert_sum_ms": sum(dst_convert),
        "algo_indices": sorted({row["algo_index"] for row in rows}),
        "returned_counts": sorted({row["returned"] for row in rows}),
        "src0_types": sorted({row["src0_type"] for row in rows}),
        "src1_types": sorted({row["src1_type"] for row in rows}),
        "dst_types": sorted({row["dst_type"] for row in rows}),
        "gelu_values": sorted({row["gelu"] for row in rows}),
    }


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if match := TIMING_RE.search(line):
                pending = {
                    "dst": match.group("dst"),
                    "dst_type": match.group("dst_type"),
                    "ne": [int(match.group(f"ne{i}")) for i in range(4)],
                    "convert_ms": float(match.group("convert")),
                    "matmul_ms": float(match.group("matmul")),
                    "dst_convert_ms": float(match.group("dst_convert")),
                    "algo_index": int(match.group("algo")),
                    "returned": int(match.group("returned")),
                }
                continue
            if pending is None:
                continue
            if match := SUCCESS_RE.search(line):
                src0 = match.group("src0")
                rows.append(
                    pending
                    | {
                        "src0": src0,
                        "bias": match.group("bias"),
                        "src0_type": match.group("src0_type"),
                        "src1_type": match.group("src1_type"),
                        "gelu": int(match.group("gelu")),
                        "group": classify(src0),
                    }
                )
                pending = None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--group-by", choices=("group", "src0", "shape"), default="group")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = parse_log(args.log)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if args.group_by == "group":
            key = (row["group"],)
        elif args.group_by == "src0":
            key = (row["src0"],)
        else:
            key = (row["group"], *row["ne"])
        grouped[key].append(row)

    summary = []
    for key, items in grouped.items():
        entry = summarize_group(items)
        entry["key"] = list(key)
        summary.append(entry)
    summary.sort(key=lambda item: item["matmul_sum_ms"], reverse=True)

    print(
        json.dumps(
            {
                "source": str(args.log),
                "rows": len(rows),
                "group_by": args.group_by,
                "summary": summary[: args.top],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
