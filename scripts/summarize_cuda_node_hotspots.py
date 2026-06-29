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


def select_profile_log(summary: dict[str, Any], label: str | None) -> dict[str, Any]:
    logs = summary.get("logs", [])
    if label:
        for log in logs:
            groups_by_label = log.get("cuda_profile_nodes_by_label_signature", {})
            if groups_by_label.get(label):
                return log

    for log in logs:
        if log.get("cuda_profile_nodes_by_signature"):
            return log

    return logs[0] if logs else {}


def hotspot_rows(summary: dict[str, Any], label: str | None, limit: int) -> list[dict[str, Any]]:
    log = select_profile_log(summary, label)
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


def selected_log_path(summary: dict[str, Any], label: str | None) -> str | None:
    log = select_profile_log(summary, label)
    path = log.get("path")
    return str(path) if path is not None else None


def classify(row: dict[str, Any]) -> str:
    signature = row["signature"]
    op = row["op"]
    if op == "MUL_MAT" and "dst=f32[3072,5184" in signature:
        return "sam3-vit: qkv cuBLASLt"
    if op == "MUL_MAT" and "dst=f32[4736,5184" in signature:
        return "sam3-vit: mlp fc1 cuBLASLt"
    if op == "MUL_MAT" and "dst=f32[1024,5184" in signature:
        return "sam3-vit: mlp fc2 / attn proj cuBLASLt"
    if op == "FLASH_ATTN_EXT" and "dst=f32[64,16,576,9]" in signature:
        return "sam3-vit: window FlashAttention head64"
    if op == "FLASH_ATTN_EXT" and "dst=f32[64,16,5184,1]" in signature:
        return "sam3-vit: global FlashAttention head64"
    if op == "CONT" and "dst=f32[1024,576,9,3]" in signature:
        return "sam3-vit: qkv layout materialization"
    if op == "CONT" and (
        "dst=f32[64,576,16,9]" in signature or "dst=f32[64,5184,16,1]" in signature
    ):
        return "sam3-vit: q/k contiguous materialization before RoPE"
    if op == "CPY" and (
        "dst=bf16[1024,24,24,9]" in signature
        or "dst=bf16[1024,72,72,1]" in signature
        or "dst=bf16[64,16,576,9]" in signature
        or "dst=bf16[64,16,5184,1]" in signature
    ):
        return "sam3-vit: BF16 cast bridge"
    if op == "ADD" and "dst=f32[1024,72,72,1]" in signature:
        return "sam3-vit: residual / position add"
    if op in {"NORM", "MUL"} and "dst=f32[1024,72,72,1]" in signature:
        return "sam3-vit: layernorm affine"
    if op == "UNARY" and "dst=f32[4736,72,72,1]" in signature:
        return "sam3-vit: MLP GELU"
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
    parser.add_argument(
        "--label",
        default=None,
        help=(
            "Optional profile label to summarize. When omitted, all CUDA-profiled nodes are "
            "grouped together; this works for both legacy Hiera and SAM3 profile labels."
        ),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = load_json(args.summary)
    rows = hotspot_rows(summary, args.label, args.limit)
    for row in rows:
        row["priority"] = classify(row)

    result = {
        "source": str(args.summary),
        "selected_log_path": selected_log_path(summary, args.label),
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
