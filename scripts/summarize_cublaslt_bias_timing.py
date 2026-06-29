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
    r"(?:gemm=\[(?P<gemm_m>\d+),(?P<gemm_n>\d+),(?P<gemm_k>\d+),(?P<gemm_batch>\d+)\] )?"
    r"dst_convert_ms=(?P<dst_convert>[0-9.]+) "
    r"(?:host_setup_ms=(?P<host_setup>[0-9.]+) )?"
    r"(?:host_total_ms=(?P<host_total>[0-9.]+) )?"
    r"(?:compute_type=(?P<compute_type>-?\d+) )?"
    r"algo_index=(?P<algo>\d+) "
    r"(?:selected_algo_index=(?P<selected_algo>\d+) )?"
    r"returned=(?P<returned>\d+)"
    r"(?: algo_cache_hit=(?P<algo_cache_hit>[01]))?"
    r"(?: plan_cache_hit=(?P<plan_cache_hit>[01]))?"
    r"(?: autotuned=(?P<autotuned>[01]))?"
    r"(?: autotune_ms=(?P<autotune_ms>[0-9.]+))?"
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
    host_total = [float(row.get("host_total_ms") or 0.0) for row in rows]
    host_setup = [float(row.get("host_setup_ms") or 0.0) for row in rows]
    algo_cache_hits = [int(row.get("algo_cache_hit") or 0) for row in rows]
    plan_cache_hits = [int(row.get("plan_cache_hit") or 0) for row in rows]
    autotuned = [int(row.get("autotuned") or 0) for row in rows]
    return {
        "n": len(rows),
        "matmul_sum_ms": sum(matmul),
        "matmul_mean_ms": statistics.mean(matmul),
        "matmul_median_ms": statistics.median(matmul),
        "convert_sum_ms": sum(convert),
        "dst_convert_sum_ms": sum(dst_convert),
        "host_total_sum_ms": sum(host_total),
        "host_setup_sum_ms": sum(host_setup),
        "algo_cache_hit_count": sum(algo_cache_hits),
        "algo_cache_hit_rate": sum(algo_cache_hits) / len(algo_cache_hits)
        if algo_cache_hits
        else 0.0,
        "plan_cache_hit_count": sum(plan_cache_hits),
        "plan_cache_hit_rate": sum(plan_cache_hits) / len(plan_cache_hits)
        if plan_cache_hits
        else 0.0,
        "algo_indices": sorted({row["algo_index"] for row in rows}),
        "selected_algo_indices": sorted({row["selected_algo_index"] for row in rows}),
        "compute_types": sorted({row["compute_type"] for row in rows}),
        "autotuned_count": sum(autotuned),
        "returned_counts": sorted({row["returned"] for row in rows}),
        "src0_types": sorted({row["src0_type"] for row in rows}),
        "src1_types": sorted({row["src1_type"] for row in rows}),
        "dst_types": sorted({row["dst_type"] for row in rows}),
        "gelu_values": sorted({row["gelu"] for row in rows}),
        "gemm_shapes": sorted(
            {tuple(row["gemm"]) for row in rows if row.get("gemm") is not None}
        ),
    }


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def append_timing_only() -> None:
        nonlocal pending
        if pending is None:
            return
        rows.append(
            pending
            | {
                "src0": "",
                "bias": "",
                "src0_type": "unknown",
                "src1_type": "unknown",
                "gelu": -1,
                "group": "timing_only",
            }
        )
        pending = None

    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if match := TIMING_RE.search(line):
                append_timing_only()
                pending = {
                    "dst": match.group("dst"),
                    "dst_type": match.group("dst_type"),
                    "ne": [int(match.group(f"ne{i}")) for i in range(4)],
                    "convert_ms": float(match.group("convert")),
                    "matmul_ms": float(match.group("matmul")),
                    "dst_convert_ms": float(match.group("dst_convert")),
                    "host_total_ms": float(match.group("host_total") or 0.0),
                    "host_setup_ms": float(match.group("host_setup") or 0.0),
                    "gemm": (
                        [
                            int(match.group("gemm_m")),
                            int(match.group("gemm_n")),
                            int(match.group("gemm_k")),
                            int(match.group("gemm_batch")),
                        ]
                        if match.group("gemm_m") is not None
                        else None
                    ),
                    "compute_type": int(match.group("compute_type") or -1),
                    "algo_index": int(match.group("algo")),
                    "selected_algo_index": int(
                        match.group("selected_algo") or match.group("algo")
                    ),
                    "returned": int(match.group("returned")),
                    "algo_cache_hit": int(match.group("algo_cache_hit") or 0),
                    "plan_cache_hit": int(match.group("plan_cache_hit") or 0),
                    "autotuned": int(match.group("autotuned") or 0),
                    "autotune_ms": float(match.group("autotune_ms") or 0.0),
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
    append_timing_only()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--group-by", choices=("group", "src0", "shape"), default="group"
    )
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
            key = (row["group"], *(row.get("gemm") or row["ne"]))
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
