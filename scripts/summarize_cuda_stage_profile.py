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


CUDA_NODE_RE = re.compile(
    r"GGML_CUDA_PROFILE_NODE fused=(\d+) skipped=(\d+) op=(\S+) name=(.*?) ms=([0-9.]+) "
    r"dst_type=(\S+) dst_ne=([0-9,]+)"
    r"(?: src0_type=(\S+) src0_ne=([0-9,]+) src1_type=(\S+) src1_ne=([0-9,]+))?"
)
COMPUTE_RE = re.compile(
    r"SAM3_PROFILE compute (?:label=(\S+) )?backend=(\S+) nodes=(\d+) ms=([0-9.]+)"
)
SAM3_VIT_BLOCK_RE = re.compile(r"sam3_vit_block_(\d{2})")


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stats(values: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(values),
        "sum_ms": sum(values),
        "mean_ms": mean(values),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "values_ms": values,
    }
    if len(values) > 1:
        drop_max = sum(values) - max(values)
        result["sum_ms_drop_max"] = drop_max
        result["mean_ms_drop_max"] = drop_max / (len(values) - 1)
    return result


def make_cuda_node(match: re.Match[str]) -> dict[str, Any]:
    (
        fused,
        skipped,
        op,
        name,
        ms,
        dst_type,
        dst_ne,
        src0_type,
        src0_ne,
        src1_type,
        src1_ne,
    ) = match.groups()
    return {
        "fused": bool(int(fused)),
        "skipped": int(skipped),
        "op": op,
        "name": name,
        "ms": float(ms),
        "dst_type": dst_type,
        "dst_ne": dst_ne,
        "src0_type": src0_type or "none",
        "src0_ne": src0_ne or "0,0,0,0",
        "src1_type": src1_type or "none",
        "src1_ne": src1_ne or "0,0,0,0",
    }


def is_sam3_neck_node(name: str) -> bool:
    return name.startswith(
        (
            "sam3_neck_",
            "sam31_neck_",
            "sam31_prop_raw_",
            "neck_",
            "neck.",
        )
    )


def mode_for_call(label: str, nodes: list[dict[str, Any]]) -> str:
    if label not in {"sam3_encode", "sam31_encode"}:
        return label
    has_detector_neck = any(
        node["name"].startswith(
            (
                "sam3_neck_det_",
                "sam31_neck_det_",
                "neck_det_",
                "neck.det.",
            )
        )
        for node in nodes
    )
    return "frame0" if has_detector_neck else "tracking"


def stage_for_node(node: dict[str, Any], current_stage: str) -> tuple[str, str]:
    name = node["name"]
    op = node["op"]
    dst_ne = node["dst_ne"]

    if name.startswith("neck.") and name.endswith("(copy)"):
        return "constant_cast", current_stage

    if name.startswith("vit_patch_embed") or name.startswith("vit_after_pos"):
        return "vit_prefix", "vit_prefix"
    if op == "IM2COL" and dst_ne.startswith("588,72,72"):
        return "vit_prefix", "vit_prefix"
    if op == "MUL_MAT" and dst_ne.startswith("5184,1024"):
        return "vit_prefix", "vit_prefix"

    if is_sam3_neck_node(name):
        return "neck", "neck"

    block_match = SAM3_VIT_BLOCK_RE.search(name)
    if block_match:
        stage = f"vit_block_{int(block_match.group(1)):02d}"
        return stage, stage

    if name == "vit_output":
        return current_stage, current_stage
    if name.startswith("vit_output "):
        return "neck", "neck"
    if current_stage == "neck":
        return "neck", "neck"
    if current_stage.startswith("vit_block_"):
        return current_stage, current_stage
    return "vit_prefix", current_stage


def summarize_call(
    label: str, call_index: int, compute: dict[str, Any], nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    stage_ms: defaultdict[str, float] = defaultdict(float)
    stage_ops: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    current_stage = "vit_prefix"
    for node in nodes:
        stage, current_stage = stage_for_node(node, current_stage)
        stage_ms[stage] += node["ms"]
        stage_ops[stage][node["op"]] += node["ms"]

    stages = {
        stage: {
            "sum_ms": ms,
            "ops": {
                op: op_ms
                for op, op_ms in sorted(
                    ops.items(), key=lambda item: item[1], reverse=True
                )
            },
            "top_ops": [
                {"op": op, "sum_ms": op_ms}
                for op, op_ms in sorted(
                    ops.items(), key=lambda item: item[1], reverse=True
                )[:8]
            ],
        }
        for stage, ms in sorted(
            stage_ms.items(), key=lambda item: item[1], reverse=True
        )
        for ops in [stage_ops[stage]]
    }
    return {
        "label": label,
        "call_index": call_index,
        "mode": mode_for_call(label, nodes),
        "backend": compute["backend"],
        "compute_nodes": compute["nodes"],
        "compute_ms": compute["ms"],
        "profile_node_count": len(nodes),
        "profile_node_sum_ms": sum(node["ms"] for node in nodes),
        "stages": stages,
    }


def summarize(path: Path, label_filter: str | None) -> dict[str, Any]:
    pending_nodes: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    call_counts: defaultdict[str, int] = defaultdict(int)

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if node_match := CUDA_NODE_RE.search(line):
            pending_nodes.append(make_cuda_node(node_match))
            continue
        if compute_match := COMPUTE_RE.search(line):
            label = compute_match.group(1) or "unlabeled"
            compute = {
                "backend": compute_match.group(2),
                "nodes": int(compute_match.group(3)),
                "ms": float(compute_match.group(4)),
            }
            call_counts[label] += 1
            if pending_nodes and (label_filter is None or label == label_filter):
                calls.append(
                    summarize_call(label, call_counts[label], compute, pending_nodes)
                )
            pending_nodes = []

    by_mode_stage_values: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_mode_stage_ops: defaultdict[str, defaultdict[str, defaultdict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for call in calls:
        mode = call["mode"]
        for stage, stage_row in call["stages"].items():
            by_mode_stage_values[mode][stage].append(float(stage_row["sum_ms"]))
            for op, op_ms in stage_row.get("ops", {}).items():
                by_mode_stage_ops[mode][stage][op] += float(op_ms)

    by_mode_stage = {
        mode: {
            stage: (
                row_stats
                | {
                    "top_ops": [
                        {
                            "op": op,
                            "sum_ms": op_ms,
                            "mean_ms": op_ms / max(int(row_stats["count"]), 1),
                        }
                        for op, op_ms in sorted(
                            by_mode_stage_ops[mode][stage].items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:8]
                    ]
                }
            )
            for stage, row_stats in sorted(
                ((stage, stats(values)) for stage, values in stages.items()),
                key=lambda item: float(item[1]["sum_ms"]),
                reverse=True,
            )
        }
        for mode, stages in sorted(by_mode_stage_values.items())
    }

    return {
        "source": str(path),
        "label_filter": label_filter,
        "note": (
            "This uses full-graph GGML_CUDA_PROFILE_NODE output. It does not execute partial "
            "graphs, so the measured node sequence matches the profiled E2E graph. Node profiling "
            "synchronizes per node and should be used for hotspot ranking rather than normal "
            "runtime totals."
        ),
        "calls": calls,
        "by_mode_stage": by_mode_stage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = summarize(args.log, args.label)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
