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
from typing import Any


CUDA_NODE_RE = re.compile(
    r"GGML_CUDA_PROFILE_NODE fused=(?P<fused>\d+) skipped=(?P<skipped>\d+) "
    r"op=(?P<op>\S+) name=(?P<name>.*?) ms=(?P<ms>[0-9.]+) "
    r"dst_type=(?P<dst_type>\S+) dst_ne=(?P<dst_ne>[0-9,]+)"
    r"(?: src0_type=(?P<src0_type>\S+) src0_ne=(?P<src0_ne>[0-9,]+) "
    r"src1_type=(?P<src1_type>\S+) src1_ne=(?P<src1_ne>[0-9,]+))?"
)
COMPUTE_RE = re.compile(
    r"SAM3_PROFILE compute (?:label=(?P<label>\S+) )?backend=(?P<backend>\S+) "
    r"nodes=(?P<nodes>\d+) ms=(?P<ms>[0-9.]+)"
)
SAM3_VIT_BLOCK_RE = re.compile(r"sam3_vit_block_(?P<block>\d{2})_(?P<tail>.+)")


def empty_stats() -> dict[str, Any]:
    return {"count": 0, "sum_ms": 0.0, "min_ms": None, "max_ms": 0.0}


def add_sample(stats: dict[str, Any], value: float) -> None:
    stats["count"] += 1
    stats["sum_ms"] += value
    stats["min_ms"] = value if stats["min_ms"] is None else min(stats["min_ms"], value)
    stats["max_ms"] = max(stats["max_ms"], value)


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats["count"])
    sum_ms = float(stats["sum_ms"])
    out = dict(stats)
    out["mean_ms"] = sum_ms / count if count else 0.0
    if count > 1:
        out["sum_ms_drop_max"] = sum_ms - float(stats["max_ms"])
        out["mean_ms_drop_max"] = out["sum_ms_drop_max"] / (count - 1)
        out["max_fraction_of_sum"] = float(stats["max_ms"]) / sum_ms if sum_ms else 0.0
    return out


def classify_stage(name: str, op: str, dst_type: str, dst_ne: str) -> str:
    if name.startswith("vit_patch_embed") or name.startswith("vit_after_pos"):
        return "sam3-vit:prefix"
    if name == "vit_output":
        return "sam3-vit:output"
    if op == "IM2COL" and dst_ne.startswith("588,72,72"):
        return "sam3-vit:prefix"
    if op == "MUL_MAT" and dst_type == "f32" and dst_ne.startswith("5184,1024"):
        return "sam3-vit:prefix-matmul"
    if op == "REPEAT" and dst_ne.startswith("1024,72,72"):
        return "sam3-vit:prefix-pos"
    if op == "NORM" and dst_ne.startswith("1024,72,72"):
        return "sam3-vit:norm"
    if op == "CONT" and dst_ne == "72,72,1024,1":
        return "sam3-vit:output-layout"

    if name.startswith("neck.") or name.startswith("neck.trk.") or name.startswith("neck.det."):
        if op == "CPY" and "weight (copy)" in name:
            return "sam3-neck:weight-cast"
        return "sam3-neck:other"

    if name.startswith("neck_det_") or name.startswith("neck_trk_"):
        if op == "CONT":
            return "sam3-neck:layout"
        return "sam3-neck:other"

    if name.startswith("sam3_neck_") or name.startswith("sam3_vit_batch_neck_"):
        if "_deconv" in name and op == "CONV_TRANSPOSE_2D":
            return "sam3-neck:deconv"
        if "_deconv" in name and "bias" in name:
            return "sam3-neck:deconv-bias"
        if name.endswith("_gelu"):
            return "sam3-neck:gelu"
        if "_conv1x1" in name:
            return "sam3-neck:conv1x1"
        if "_conv3x3" in name and op == "CONV_2D":
            return "sam3-neck:conv3x3"
        if "_conv3x3" in name and "bias" in name:
            return "sam3-neck:conv3x3-bias"
        if name.endswith("_out") or "input_whcb" in name:
            return "sam3-neck:layout"
        if name.endswith("_pool"):
            return "sam3-neck:pool"
        return "sam3-neck:other"

    if op == "CONV_TRANSPOSE_2D" and dst_type == "f32":
        if dst_ne.startswith(("144,144,", "288,288,")):
            return "sam3-neck:deconv"
    if op in {"CONV_2D", "IM2COL"} and dst_type in {"f32", "bf16", "f16"}:
        if dst_ne.startswith(("36,36,", "72,72,", "144,144,", "288,288,")):
            return "sam3-neck:conv-or-im2col"
        if op == "IM2COL" and dst_ne.startswith(
            ("256,288,288,", "512,144,144,", "1024,72,72,", "1024,36,36,")
        ):
            return "sam3-neck:conv-or-im2col"
    if (
        op == "MUL_MAT"
        and dst_type == "f32"
        and dst_ne.startswith(("82944,256,", "20736,256,", "5184,256,", "1296,256,"))
    ):
        return "sam3-neck:conv1x1-matmul"

    if op == "MUL_MAT":
        if name.endswith("_qkv_matmul"):
            return "sam3-vit:qkv-matmul"
        if name.endswith("_proj_matmul"):
            return "sam3-vit:proj-matmul"
        if "_mlp_fc1" in name and name.endswith("_matmul"):
            return "sam3-vit:mlp-fc1-matmul"
        if name.endswith("_mlp_fc2_matmul"):
            return "sam3-vit:mlp-fc2-matmul"

    match = SAM3_VIT_BLOCK_RE.search(name)
    if not match:
        if op == "MUL_MAT" and dst_type == "f32" and dst_ne.startswith("3072,5184"):
            return "sam3-vit:qkv-matmul"
        if op == "MUL_MAT" and dst_type == "f32" and dst_ne.startswith("4736,5184"):
            return "sam3-vit:mlp-fc1-matmul"
        if op == "MUL_MAT" and dst_type == "f32" and dst_ne.startswith("1024,5184"):
            return "sam3-vit:proj-or-mlp-fc2-matmul"
        return "other"

    tail = match.group("tail")
    if "window_qkv" in tail or "global_qkv" in tail:
        return "sam3-vit:qk-layout-rope"
    if "window_attn_bf16" in tail or "global_attn_bf16" in tail:
        return "sam3-vit:attn-bf16-copy"
    if "window_attn" in tail:
        return "sam3-vit:window-attn"
    if "global_attn" in tail:
        return "sam3-vit:global-attn"
    if "window_part (copy)" in tail:
        return "sam3-vit:window-part-bf16-copy"
    if "window_part" in tail:
        return "sam3-vit:window-part"
    if "window_unpart" in tail:
        return "sam3-vit:window-unpart"
    if "after_attn_residual" in tail:
        return "sam3-vit:after-attn-residual-norm-copy"
    if tail.startswith("norm") and "(copy)" in tail:
        return "sam3-vit:norm-copy"
    if "mlp_gelu" in tail:
        return "sam3-vit:mlp-gelu"
    if tail == "out":
        return "sam3-vit:block-out-residual-norm-copy"
    return "sam3-vit:other"


def node_from_match(match: re.Match[str]) -> dict[str, Any]:
    groups = match.groupdict()
    return {
        "fused": bool(int(groups["fused"])),
        "skipped": int(groups["skipped"]),
        "op": groups["op"],
        "name": groups["name"],
        "ms": float(groups["ms"]),
        "dst_type": groups["dst_type"],
        "dst_ne": groups["dst_ne"],
        "src0_type": groups.get("src0_type") or "none",
        "src0_ne": groups.get("src0_ne") or "0,0,0,0",
        "src1_type": groups.get("src1_type") or "none",
        "src1_ne": groups.get("src1_ne") or "0,0,0,0",
    }


def parse_labeled_nodes(path: Path) -> dict[str, list[dict[str, Any]]]:
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending: list[dict[str, Any]] = []
    saw_compute = False

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        node_match = CUDA_NODE_RE.search(line)
        if node_match:
            pending.append(node_from_match(node_match))
            continue
        compute_match = COMPUTE_RE.search(line)
        if compute_match:
            saw_compute = True
            label = compute_match.group("label") or "unlabeled"
            labels[label].extend(pending)
            pending = []

    if pending:
        labels["unlabeled_tail"].extend(pending)
    if not saw_compute:
        labels["all"].extend(labels.pop("unlabeled_tail", []))
    return labels


def parse_labeled_node_occurrences(path: Path) -> dict[str, list[dict[str, Any]]]:
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending: list[dict[str, Any]] = []
    saw_compute = False
    counts: dict[str, int] = defaultdict(int)

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        node_match = CUDA_NODE_RE.search(line)
        if node_match:
            pending.append(node_from_match(node_match))
            continue
        compute_match = COMPUTE_RE.search(line)
        if compute_match:
            saw_compute = True
            label = compute_match.group("label") or "unlabeled"
            counts[label] += 1
            occurrence = f"{label}#{counts[label]:02d}"
            labels[occurrence].extend(pending)
            pending = []

    if pending:
        labels["unlabeled_tail#01"].extend(pending)
    if not saw_compute:
        labels["all#01"].extend(labels.pop("unlabeled_tail#01", []))
    return labels


def summarize_nodes(
    nodes: list[dict[str, Any]],
    *,
    name_regex: re.Pattern[str] | None,
    op_filter: set[str] | None,
    top: int,
) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    by_stage: dict[str, dict[str, Any]] = {}
    selected_count = 0

    for node in nodes:
        if op_filter is not None and node["op"] not in op_filter:
            continue
        if name_regex is not None and not name_regex.search(node["name"]):
            continue
        selected_count += 1
        name_key = f"op={node['op']} name={node['name']} dst={node['dst_type']}[{node['dst_ne']}]"
        name_stats = by_name.setdefault(
            name_key,
            {
                **empty_stats(),
                "op": node["op"],
                "name": node["name"],
                "dst_type": node["dst_type"],
                "dst_ne": node["dst_ne"],
                "src0_type": node["src0_type"],
                "src0_ne": node["src0_ne"],
                "src1_type": node["src1_type"],
                "src1_ne": node["src1_ne"],
            },
        )
        add_sample(name_stats, float(node["ms"]))

        stage = classify_stage(
            node["name"], node["op"], node["dst_type"], node["dst_ne"]
        )
        stage_stats = by_stage.setdefault(stage, empty_stats())
        add_sample(stage_stats, float(node["ms"]))

    finalized_names = {
        key: finalize_stats(stats)
        for key, stats in sorted(
            by_name.items(),
            key=lambda item: (
                -float(
                    item[1]["sum_ms"] - item[1]["max_ms"]
                    if item[1]["count"] > 1
                    else item[1]["sum_ms"]
                ),
                item[0],
            ),
        )[:top]
    }
    finalized_stages = {
        key: finalize_stats(stats)
        for key, stats in sorted(
            by_stage.items(),
            key=lambda item: (
                -float(
                    item[1]["sum_ms"] - item[1]["max_ms"]
                    if item[1]["count"] > 1
                    else item[1]["sum_ms"]
                ),
                item[0],
            ),
        )
    }
    return {
        "selected_node_count": selected_count,
        "top_by_name": finalized_names,
        "by_stage": finalized_stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--name-regex", default=None)
    parser.add_argument("--op", action="append", default=None)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--split-computes", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    labels = (
        parse_labeled_node_occurrences(args.log)
        if args.split_computes
        else parse_labeled_nodes(args.log)
    )
    name_regex = re.compile(args.name_regex) if args.name_regex else None
    op_filter = set(args.op) if args.op else None
    selected_labels = (
        [
            label
            for label in sorted(labels)
            if label == args.label or label.startswith(f"{args.label}#")
        ]
        if args.label and args.split_computes
        else ([args.label] if args.label else sorted(labels))
    )

    result = {
        "source": str(args.log),
        "label": args.label,
        "split_computes": args.split_computes,
        "name_regex": args.name_regex,
        "op_filter": sorted(op_filter) if op_filter else None,
        "note": (
            "GGML_CUDA_PROFILE_NODES synchronizes per node; use drop-max sums for ranking, "
            "not as normal wall-clock totals."
        ),
        "labels": {
            label: summarize_nodes(
                labels.get(label, []),
                name_regex=name_regex,
                op_filter=op_filter,
                top=args.top,
            )
            for label in selected_labels
        },
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
