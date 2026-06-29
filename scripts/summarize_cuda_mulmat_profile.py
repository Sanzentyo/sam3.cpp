# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MULMAT_RE = re.compile(
    r"GGML_CUDA_PROFILE_MUL_MAT ms=(?P<ms>[0-9.]+) path=(?P<path>\S+) "
    r"src0=(?P<src0>.*?) src1=(?P<src1>.*?) dst=(?P<dst>.*?) "
    r"types=(?P<src0_type>\S+)/(?P<src1_type>\S+)->(?P<dst_type>\S+) "
    r"src0_ne=\[(?P<src0_ne>[^\]]+)\] src1_ne=\[(?P<src1_ne>[^\]]+)\] "
    r"dst_ne=\[(?P<dst_ne>[^\]]+)\]"
)


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(","))


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum_ms": 0.0, "mean_ms": 0.0}
    out: dict[str, Any] = {
        "count": len(values),
        "sum_ms": sum(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }
    if len(values) > 1:
        drop_max = sum(values) - max(values)
        out["sum_ms_drop_max"] = drop_max
        out["mean_ms_drop_max"] = drop_max / (len(values) - 1)
    return out


def classify(row: dict[str, Any]) -> str:
    src0 = str(row["src0"])
    src1 = str(row["src1"])
    dst = str(row["dst"])

    for text in (dst, src0, src1):
        if "sam3_vit_block_" in text:
            if "_qkv" in text:
                return "sam3-vit:qkv"
            if "_proj" in text:
                return "sam3-vit:proj"
            if "_mlp_fc1" in text:
                return "sam3-vit:mlp-fc1"
            if "_mlp_fc2" in text:
                return "sam3-vit:mlp-fc2"
            return "sam3-vit:other"

    if src0.startswith("vit.blocks."):
        if ".attn.qkv." in src0:
            return "sam3-vit:qkv"
        if ".attn.proj." in src0:
            return "sam3-vit:proj"
        if ".mlp.lin1." in src0:
            return "sam3-vit:mlp-fc1"
        if ".mlp.lin2." in src0:
            return "sam3-vit:mlp-fc2"
        return "sam3-vit:other"
    if src1.startswith("vit.patch_embed.") or "vit.patch_embed" in src0:
        return "sam3-vit:patch-embed"
    if src0.startswith("neck.") or src1.startswith("neck."):
        return "sam3-neck"
    if src0.startswith("text.blocks.") or src1.startswith("text.blocks."):
        return "sam3-text"
    if src0.startswith("geom.") or src1.startswith("geom."):
        return "sam3-geom"
    if src0.startswith("fenc.") or src1.startswith("fenc."):
        return "sam3-fenc"
    if src0.startswith("ddec.") or src1.startswith("ddec."):
        return "sam3-ddec"
    if src0.startswith("mem_") or src1.startswith("mem_"):
        return "sam3-memory"
    if src0.startswith("seg.") or src1.startswith("seg."):
        return "sam3-seg"
    return "other"


def shape_key(row: dict[str, Any]) -> str:
    return (
        f"{row['src0_type']}[{','.join(map(str, row['src0_ne']))}] x "
        f"{row['src1_type']}[{','.join(map(str, row['src1_ne']))}] -> "
        f"{row['dst_type']}[{','.join(map(str, row['dst_ne']))}]"
    )


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MULMAT_RE.search(line)
        if not match:
            continue
        row: dict[str, Any] = match.groupdict()
        row["ms"] = float(row["ms"])
        row["src0_ne"] = parse_shape(row["src0_ne"])
        row["src1_ne"] = parse_shape(row["src1_ne"])
        row["dst_ne"] = parse_shape(row["dst_ne"])
        row["group"] = classify(row)
        row["shape"] = shape_key(row)
        rows.append(row)
    return rows


def summarize_rows(
    rows: list[dict[str, Any]], key_parts: tuple[str, ...], top: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    examples: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[part] for part in key_parts)
        grouped[key].append(float(row["ms"]))
        examples.setdefault(key, row)

    out = []
    for key, values in grouped.items():
        item = stats(values)
        item["key"] = list(key)
        sample = examples[key]
        item["example"] = {
            "src0": sample["src0"],
            "src1": sample["src1"],
            "dst": sample["dst"],
            "shape": sample["shape"],
        }
        out.append(item)
    out.sort(key=lambda item: float(item["sum_ms"]), reverse=True)
    return out[:top]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = parse_log(args.log)
    result = {
        "source": str(args.log),
        "note": (
            "GGML_CUDA_PROFILE_MUL_MAT synchronizes each profiled GEMM. Use it for "
            "path and shape attribution, not for normal wall-clock totals."
        ),
        "rows": len(rows),
        "total": stats([float(row["ms"]) for row in rows]),
        "by_group": summarize_rows(rows, ("group",), args.top),
        "by_group_path": summarize_rows(rows, ("group", "path"), args.top),
        "by_group_shape": summarize_rows(rows, ("group", "shape"), args.top),
        "by_path": summarize_rows(rows, ("path",), args.top),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
