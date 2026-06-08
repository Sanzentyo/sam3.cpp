# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sort_key(item: tuple[str, dict[str, Any]]) -> float:
    row = item[1]
    return float(row.get("sum_ms_drop_max", row.get("sum_ms", 0.0)))


def compact_signature(key: str) -> str:
    return key.removeprefix("op=")


def hotspot_rows(summary: dict[str, Any], label: str | None, limit: int) -> list[dict[str, Any]]:
    log = summary["logs"][0]
    if label:
        groups = log.get("cuda_profile_nodes_by_label_signature", {}).get(label, {})
    else:
        groups = log.get("cuda_profile_nodes_by_signature", {})

    rows: list[dict[str, Any]] = []
    for key, row in sorted(groups.items(), key=sort_key, reverse=True)[:limit]:
        sum_ms = float(row.get("sum_ms", 0.0))
        drop_max = float(row.get("sum_ms_drop_max", sum_ms))
        count = int(row.get("count", 0))
        adjusted_count = max(count - 1, 1) if "sum_ms_drop_max" in row else max(count, 1)
        rows.append(
            {
                "op": row.get("op"),
                "count": count,
                "sum_ms": sum_ms,
                "sum_ms_drop_max": drop_max,
                "mean_ms_drop_max": drop_max / adjusted_count,
                "signature": compact_signature(key),
            }
        )
    return rows


def classify(row: dict[str, Any]) -> str:
    signature = row["signature"]
    op = row["op"]
    if op == "FLASH_ATTN_EXT" and (
        "f32[56,8,4096,1]" in signature or "f32[56,8,1024,1]" in signature
    ):
        return "primary: head_dim=56 global FlashAttention"
    if op == "FLASH_ATTN_EXT" and (
        "f32[56,8,196,25]" in signature or "f32[56,8,196,9]" in signature
    ):
        return "primary: head_dim=56 window FlashAttention"
    if op == "FLASH_ATTN_EXT" and (
        "f32[56,2,64,1024]" in signature
        or "f32[56,4,16,1024]" in signature
        or "f32[56,8,4,1024]" in signature
        or "f32[56,16,49,25]" in signature
    ):
        return "primary: head_dim=56 q-pool FlashAttention"
    if op == "FLASH_ATTN_EXT" and "f32[256,1,4096,1]" in signature:
        return "primary: propagation head_dim=256 FlashAttention"
    if op == "MUL_MAT" and ("q4_0[1792,448" in signature or "q8_0[1792,448" in signature):
        return "secondary: stage-2 quantized MLP projection"
    if op == "MUL_MAT" and ("dst=f32[448,4096" in signature or "dst=f32[448,1024" in signature):
        return "secondary: stage-2 quantized MLP projection"
    if op == "MUL_MAT" and ("q4_0[448,1792" in signature or "q8_0[448,1792" in signature):
        return "secondary: stage-2 quantized MLP expansion"
    if op == "MUL_MAT" and ("dst=f32[1792,4096" in signature or "dst=f32[1792,1024" in signature):
        return "secondary: stage-2 quantized MLP expansion"
    if op == "MUL_MAT" and ("q4_0[448,1344" in signature or "q8_0[448,1344" in signature):
        return "secondary: window qkv quantized projection"
    if op == "MUL_MAT" and ("dst=f32[1344,196,25" in signature or "dst=f32[1344,196,9" in signature):
        return "secondary: window qkv quantized projection"
    if op in {"ADD", "UNARY", "NORM", "CONT", "PAD"}:
        return "tertiary: graph/layout or elementwise overhead"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--label", default="hiera_encode")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = hotspot_rows(load_json(args.summary), args.label or None, args.limit)
    for row in rows:
        row["priority"] = classify(row)

    result = {
        "source": str(args.summary),
        "label": args.label,
        "note": (
            "GGML_CUDA_PROFILE_NODES synchronizes per node and disables CUDA graphs; "
            "use drop-max sums for hotspot ranking, not as normal runtime totals."
        ),
        "hotspots": rows,
    }

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
