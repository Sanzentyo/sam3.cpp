#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Summarize SAM3.1 mask-init E2E process split and variant decisions."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


ACCEPTANCE_SIGNAL_THRESHOLD = -2.0
QUALITY_IOU_THRESHOLD = 0.999999


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def metric(item: dict[str, Any], name: str) -> dict[str, Any]:
    value = item.get(name)
    return value if isinstance(value, dict) else {}


def metric_mean(item: dict[str, Any], name: str) -> float | None:
    return number(metric(item, name).get("mean"))


def metric_min_prefer(item: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = number(metric(item, name).get("min"))
        if value is not None:
            return value
    return None


def metric_max_prefer(item: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = number(metric(item, name).get("max"))
        if value is not None:
            return value
    return None


def metric_mean_any(item: dict[str, Any], names: str | tuple[str, ...]) -> float | None:
    if isinstance(names, str):
        return metric_mean(item, names)
    for name in names:
        mean = metric_mean(item, name)
        if mean is not None:
            return mean
    return None


def metric_stdev(item: dict[str, Any], name: str) -> float | None:
    return number(metric(item, name).get("stdev"))


def metric_count(item: dict[str, Any], name: str) -> int | None:
    value = metric(item, name).get("count")
    return int(value) if isinstance(value, int | float) else None


def metric_list_means(item: dict[str, Any], name: str) -> list[float | None]:
    value = item.get(name)
    if not isinstance(value, list):
        return []
    means: list[float | None] = []
    for entry in value:
        means.append(number(entry.get("mean")) if isinstance(entry, dict) else None)
    return means


def pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator * 100.0


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def delta_signal(
    baseline_sd: float | None,
    baseline_n: int | None,
    candidate_sd: float | None,
    candidate_n: int | None,
    delta_ms: float | None,
) -> float | None:
    if (
        delta_ms is None
        or baseline_sd is None
        or candidate_sd is None
        or baseline_n is None
        or candidate_n is None
        or baseline_n <= 0
        or candidate_n <= 0
    ):
        return None
    stderr = math.sqrt(
        (baseline_sd * baseline_sd / baseline_n)
        + (candidate_sd * candidate_sd / candidate_n)
    )
    if stderr <= 0.0:
        return None
    return delta_ms / stderr


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}x"


def fmt_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def fmt_float_list(values: Any) -> str:
    if not isinstance(values, list):
        return "-"
    formatted = [fmt_float(number(value)) for value in values]
    return "[" + ", ".join(formatted) + "]"


def fmt_signal(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def fmt_count(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def fmt_ops(value: Any) -> str:
    if not isinstance(value, list):
        return "-"
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        op = item.get("op")
        ms = number(item.get("mean_ms"))
        if ms is None:
            ms = number(item.get("sum_ms"))
        parts.append(f"{op}:{fmt_ms(ms)}")
    return ", ".join(parts) if parts else "-"


def fmt_stage_list(value: Any) -> str:
    if not isinstance(value, list):
        return "-"
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        stage = item.get("stage")
        ms = number(item.get("profile_ms"))
        parts.append(f"{stage}:{fmt_ms(ms)}")
    return ", ".join(parts) if parts else "-"


def variant_quality_status(item: dict[str, Any]) -> tuple[bool, dict[str, float | None]]:
    min_iou = metric_min_prefer(item, ("default_sequence_iou", "default_iou"))
    max_xor = metric_max_prefer(item, ("default_sequence_xor_pixels", "default_xor_pixels"))
    same_as_default = (
        min_iou is not None
        and min_iou >= QUALITY_IOU_THRESHOLD
        and max_xor is not None
        and max_xor == 0.0
    )
    return same_as_default, {"min_default_iou": min_iou, "max_default_xor_pixels": max_xor}


def decision_for_variant(
    *,
    variant: str,
    same_as_default: bool,
    full_delta_ms: float | None,
    full_delta_signal: float | None,
) -> str:
    if variant == "default":
        return "baseline"
    if not same_as_default:
        return "reject_quality_diff"
    if (
        full_delta_ms is not None
        and full_delta_ms < 0.0
        and full_delta_signal is not None
        and full_delta_signal <= ACCEPTANCE_SIGNAL_THRESHOLD
    ):
        return "candidate_win"
    if full_delta_ms is not None and full_delta_ms < 0.0:
        return "noise_win"
    return "reject_no_e2e_win"


def process_split_rows(variant_item: dict[str, Any]) -> list[dict[str, Any]]:
    denominator = metric_mean(variant_item, "full_frame_step_ms")
    components = [
        ("frame0.total", "frame0_ms", "frame0 encode plus detection-mask init"),
        ("frame0.image_encode", "encode_frame0_ms", "frame0 image encoder"),
        ("frame0.add_detection", "add_detection_ms", "mask prompt and tracker initialization"),
        ("tail.total", "propagate_ms", "frame1 encode plus encoded propagation"),
        ("tail.image_encode", "encode_frame1_ms", "frame1 image encoder"),
        (
            "tail.image_encode_total",
            "tail_image_encode_total_ms",
            "all propagated-frame image encodes",
        ),
        ("tail.propagate_encoded", "propagate_encoded_ms", "propagation after encoded frame"),
        ("tail.remainder", "propagate_remainder_ms", "frame construction and unaccounted tail work"),
        ("encode.total", "two_frame_encode_ms", "frame0 + frame1 image encoding"),
    ]
    rows = [
        {
            "process": "full_frame_step",
            "metric": "full_frame_step_ms",
            "mean_ms": denominator,
            "full_frame_step_pct": 100.0 if denominator is not None else None,
            "note": "E2E denominator for this mask-init sequence contract",
        }
    ]
    for process, key, note in components:
        mean = metric_mean(variant_item, key)
        rows.append(
            {
                "process": process,
                "metric": key,
                "mean_ms": mean,
                "full_frame_step_pct": pct(mean, denominator),
                "note": note,
            }
        )
    return rows


def required_e2e_split_rows(variant_item: dict[str, Any]) -> list[dict[str, Any]]:
    denominator = metric_mean(variant_item, "run_required_e2e_ms")
    if denominator is None:
        return []
    components = [
        ("required.session_setup", "required_session_setup_ms", "state and tracker setup"),
        ("required.input_prepare", "required_input_prepare_ms", "synthetic frame and prompt construction"),
        ("required.model_execute", "required_model_execute_ms", "frame0 mask-init plus tail propagation"),
        ("required.image_encode", "required_image_encode_ms", "all required image encoder calls"),
        (
            "required.image_encode.graph_compute",
            "required_image_encode_graph_compute_ms",
            "required image encoder graph compute",
        ),
        (
            "required.image_encode.non_compute",
            "required_image_encode_non_compute_ms",
            "required image encoder wall time outside graph compute",
        ),
        ("required.mask_init", "required_mask_init_ms", "mask prompt initialization model work"),
        (
            "required.propagate_encoded",
            "required_propagate_encoded_ms",
            "propagation work after frames are encoded",
        ),
        (
            "required.model_accounted",
            "required_model_accounted_ms",
            "image encode + mask init + encoded propagation",
        ),
        (
            "required.core_compute",
            "required_core_compute_ms",
            "image-encoder graph plus measured mask/propgate model subgraphs",
        ),
        (
            "required.non_image_model",
            "required_non_image_model_ms",
            "mask initialization plus encoded propagation outside image encode",
        ),
        (
            "required.non_graph_or_setup",
            "required_non_graph_or_setup_ms",
            "required E2E outside measured model graph/subgraph compute",
        ),
        (
            "required.model_remainder",
            "required_model_remainder_ms",
            "model execution accounting delta",
        ),
        ("state.create", "state_create_ms", "C++ state allocation for this sequence"),
        ("tracker.create", "tracker_create_ms", "visual tracker creation and warm caches"),
        ("frame.construct_total", "frame_construct_total_ms", "synthetic input frame construction"),
        ("detection.construct", "detection_construct_ms", "synthetic mask prompt construction"),
        ("frame0.total", "frame0_ms", "frame0 image encode plus mask-init model work"),
        ("frame0.image_encode", "encode_frame0_ms", "frame0 image encoder"),
        (
            "frame0.image_encode.graph_compute",
            "frame0_encode_graph_compute_ms",
            "frame0 image encoder graph compute",
        ),
        ("frame0.add_detection", "add_detection_ms", "mask prompt and tracker initialization"),
        ("tail.total", "propagate_ms", "frame1 encode plus encoded propagation"),
        ("tail.image_encode", "tail_image_encode_total_ms", "all required tail image encodes"),
        (
            "tail.image_encode.graph_compute",
            "tail_encode_graph_compute_total_ms",
            "all required tail image encoder graph compute",
        ),
        (
            "tail.image_encode.non_compute",
            "tail_encode_non_compute_total_ms",
            "all required tail image encoder wall time outside graph compute",
        ),
        ("tail.frame1.image_encode", "encode_frame1_ms", "first tail frame image encoder"),
        ("tail.propagate_encoded", "propagate_encoded_ms", "propagation after encoded frame"),
        ("tail.remainder", "propagate_remainder_ms", "tail wall time not in encode/propagate"),
        ("required.accounted", "required_e2e_accounted_ms", "sum of explicit required-E2E parts"),
        ("required.remainder", "required_e2e_remainder_ms", "required-E2E accounting delta"),
    ]
    rows = [
        {
            "process": "required_e2e",
            "metric": "run_required_e2e_ms",
            "mean_ms": denominator,
            "e2e_pct": 100.0,
            "note": "steady-state required E2E for this mask-init sequence contract",
        }
    ]
    for process, key, note in components:
        mean = metric_mean_any(variant_item, key)
        rows.append(
            {
                "process": process,
                "metric": key,
                "mean_ms": mean,
                "e2e_pct": pct(mean, denominator),
                "note": note,
            }
        )
    return rows


def default_internal_timing_rows(variant_item: dict[str, Any]) -> list[dict[str, Any]]:
    components = [
        (
            "frame0.image_encode",
            [
                ("total", "frame0_encode_timing_total_ms"),
                ("preprocess", "frame0_encode_preprocess_ms"),
                ("graph_build", "frame0_encode_graph_build_ms"),
                ("graph_alloc", "frame0_encode_graph_alloc_ms"),
                ("input_upload", "frame0_encode_input_upload_ms"),
                ("graph_compute", "frame0_encode_graph_compute_ms"),
                ("state_update", "frame0_encode_state_update_ms"),
                ("pe_build", "frame0_encode_pe_build_ms"),
            ],
        ),
        (
            "frame1.image_encode",
            [
                ("total", "frame1_encode_timing_total_ms"),
                ("preprocess", "frame1_encode_preprocess_ms"),
                ("graph_build", "frame1_encode_graph_build_ms"),
                ("graph_alloc", "frame1_encode_graph_alloc_ms"),
                ("input_upload", "frame1_encode_input_upload_ms"),
                ("graph_compute", "frame1_encode_graph_compute_ms"),
                ("state_update", "frame1_encode_state_update_ms"),
                ("pe_build", "frame1_encode_pe_build_ms"),
            ],
        ),
        (
            "tail.propagate_encoded",
            [
                ("total", "propagate_timing_total_ms"),
                ("graph_build", "propagate_graph_build_ms"),
                ("graph_alloc", "propagate_graph_alloc_ms"),
                ("input_upload", "propagate_input_upload_ms"),
                ("graph_compute", "propagate_graph_compute_ms"),
                ("output_read", "propagate_output_read_ms"),
                ("memory_prepare", "propagate_memory_prepare_ms"),
                ("result_build", "propagate_result_build_ms"),
            ],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for process, metrics in components:
        row = {"process": process}
        for label, key in metrics:
            row[label] = metric_mean(variant_item, key)
        rows.append(row)
    return rows


def python_required_e2e_split_rows(python_timing: dict[str, Any]) -> list[dict[str, Any]]:
    denominator = number(python_timing.get("required_e2e_ms"))
    if denominator is None:
        return []
    components = [
        ("python.session_setup", "required_session_setup_ms", "init_state for this sequence"),
        (
            "python.input_prepare",
            "required_input_prepare_ms",
            "synthetic frame construction and mask tensor preparation",
        ),
        (
            "python.model_execute",
            "required_model_execute_ms",
            "backbone/cache, mask init, and cached propagation",
        ),
        (
            "python.backbone_or_image_encode",
            "required_image_encode_or_backbone_ms",
            "official backbone/cache work for required frames",
        ),
        ("python.mask_init", "required_mask_init_ms", "official mask prompt initialization"),
        (
            "python.cached_propagate",
            "required_propagate_encoded_ms",
            "official propagation after cached frame features",
        ),
        (
            "python.accounted",
            "required_accounted_ms",
            "sum of explicit official Python required-E2E parts",
        ),
        ("python.remainder", "required_remainder_ms", "official Python accounting delta"),
    ]
    rows = [
        {
            "process": "python.required_e2e",
            "metric": "required_e2e_ms",
            "mean_ms": denominator,
            "e2e_pct": 100.0,
            "note": "official Python required E2E for the same mask-init sequence contract",
        }
    ]
    for process, key, note in components:
        mean = number(python_timing.get(key))
        rows.append(
            {
                "process": process,
                "metric": key,
                "mean_ms": mean,
                "e2e_pct": pct(mean, denominator),
                "note": note,
            }
        )
    return rows


def cpp_profile_stage_rows(case_item: dict[str, Any]) -> list[dict[str, Any]]:
    profile = case_item.get("cpp_profile")
    if not isinstance(profile, dict):
        return []
    rows = profile.get("top_stage_rows")
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "mode": row.get("mode"),
                "stage": row.get("stage"),
                "count": number(row.get("count")),
                "mean_ms": number(row.get("mean_ms")),
                "mean_ms_drop_max": number(row.get("mean_ms_drop_max")),
                "top_ops": row.get("top_ops"),
            }
        )
    return result


def load_optional_json(path_value: Any) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def profile_stage_sum(stage: dict[str, Any]) -> float:
    value = number(stage.get("sum_ms"))
    if value is not None:
        return value
    count = number(stage.get("count"))
    mean = number(stage.get("mean_ms"))
    if count is None or mean is None:
        return 0.0
    return count * mean


def profile_stage_count(stage: dict[str, Any]) -> int:
    value = number(stage.get("count"))
    return int(value) if value is not None else 0


def infer_encode_profile_mode(item: dict[str, Any]) -> str:
    stages = item.get("by_stage")
    if not isinstance(stages, dict):
        return "image_encoder"
    neck_deconv = stages.get("sam3-neck:deconv")
    neck_conv = stages.get("sam3-neck:conv-or-im2col")
    deconv_count = profile_stage_count(neck_deconv) if isinstance(neck_deconv, dict) else 0
    conv_count = profile_stage_count(neck_conv) if isinstance(neck_conv, dict) else 0
    if max(deconv_count, conv_count) >= 6:
        return "frame0"
    if max(deconv_count, conv_count) > 0:
        return "tracking"
    return "image_encoder"


def encode_profile_index(label: str) -> int:
    match = re.search(r"#(\d+)$", label)
    return int(match.group(1)) if match else 0


def image_component_for_stage(stage: str) -> str:
    if stage.startswith("sam3-neck:"):
        return "image.neck"
    if stage in {"sam3-vit:mlp-fc1-matmul", "sam3-vit:mlp-fc2-matmul"}:
        return "image.vit.mlp_matmul"
    if stage == "sam3-vit:qkv-matmul":
        return "image.vit.qkv_matmul"
    if stage in {"sam3-vit:window-attn", "sam3-vit:global-attn"}:
        return "image.vit.attention"
    if stage == "sam3-vit:proj-matmul":
        return "image.vit.proj_matmul"
    if stage in {
        "sam3-vit:after-attn-residual-norm-copy",
        "sam3-vit:attn-bf16-copy",
        "sam3-vit:block-out-residual-norm-copy",
        "sam3-vit:qk-layout-rope",
        "sam3-vit:window-part",
        "sam3-vit:window-part-bf16-copy",
        "sam3-vit:window-unpart",
    }:
        return "image.vit.layout_rope_copy"
    if stage == "sam3-vit:mlp-gelu":
        return "image.vit.mlp_gelu"
    if stage.startswith("sam3-vit:"):
        return "image.vit.other"
    return "image.other"


def cpp_profile_image_component_rows(
    case_item: dict[str, Any], default_variant: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cpp_profile = case_item.get("cpp_profile")
    if not isinstance(cpp_profile, dict):
        return [], {}
    node_profile = load_optional_json(cpp_profile.get("node_names_split"))
    labels = node_profile.get("labels")
    if not isinstance(labels, dict):
        return [], {}

    latest_by_mode: dict[str, tuple[int, str, dict[str, Any]]] = {}
    for label, item in labels.items():
        if not isinstance(label, str) or not isinstance(item, dict):
            continue
        if not (label.startswith("sam31_encode#") or label.startswith("sam3_encode#")):
            continue
        mode = infer_encode_profile_mode(item)
        index = encode_profile_index(label)
        if mode not in latest_by_mode or index > latest_by_mode[mode][0]:
            latest_by_mode[mode] = (index, label, item)

    selected = [
        latest_by_mode[mode]
        for mode in ("frame0", "tracking", "image_encoder")
        if mode in latest_by_mode
    ]
    if not selected:
        return [], {}

    raw_stage_sums: dict[str, float] = {}
    component_sums: dict[str, float] = {}
    component_stages: dict[str, dict[str, float]] = {}
    selected_labels: list[str] = []
    for _index, label, item in selected:
        selected_labels.append(label)
        stages = item.get("by_stage")
        if not isinstance(stages, dict):
            continue
        for stage, stage_item in stages.items():
            if not isinstance(stage, str) or not isinstance(stage_item, dict):
                continue
            stage_ms = profile_stage_sum(stage_item)
            if stage_ms <= 0.0:
                continue
            component = image_component_for_stage(stage)
            raw_stage_sums[stage] = raw_stage_sums.get(stage, 0.0) + stage_ms
            component_sums[component] = component_sums.get(component, 0.0) + stage_ms
            component_stage = component_stages.setdefault(component, {})
            component_stage[stage] = component_stage.get(stage, 0.0) + stage_ms

    profile_total_ms = sum(component_sums.values())
    normal_graph_ms = metric_mean(default_variant, "required_image_encode_graph_compute_ms")
    normal_required_ms = metric_mean(default_variant, "run_required_e2e_ms")
    rows: list[dict[str, Any]] = []
    for component, profile_ms in sorted(
        component_sums.items(), key=lambda item: item[1], reverse=True
    ):
        graph_share = profile_ms / profile_total_ms if profile_total_ms > 0.0 else None
        projected_ms = (
            graph_share * normal_graph_ms
            if graph_share is not None and normal_graph_ms is not None
            else None
        )
        rows.append(
            {
                "component": component,
                "profile_ms": profile_ms,
                "profile_image_graph_pct": (
                    graph_share * 100.0 if graph_share is not None else None
                ),
                "projected_normal_ms": projected_ms,
                "required_e2e_pct": pct(projected_ms, normal_required_ms),
                "top_profile_stages": [
                    {"stage": stage, "profile_ms": stage_ms}
                    for stage, stage_ms in sorted(
                        component_stages.get(component, {}).items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:4]
                ],
            }
        )

    meta = {
        "node_names_split": cpp_profile.get("node_names_split"),
        "selected_labels": selected_labels,
        "profile_total_ms": profile_total_ms,
        "normal_image_graph_ms": normal_graph_ms,
        "normal_required_e2e_ms": normal_required_ms,
        "note": (
            "Component shares come from synchronized CUDA node profiling. Projected normal ms "
            "applies those shares to the non-profiled required_image_encode_graph_compute_ms."
        ),
        "top_raw_stages": [
            {"stage": stage, "profile_ms": stage_ms}
            for stage, stage_ms in sorted(
                raw_stage_sums.items(), key=lambda item: item[1], reverse=True
            )[:16]
        ],
    }
    return rows, meta


def summarize_case(
    *,
    size_label: str,
    case_label: str,
    case_item: dict[str, Any],
) -> dict[str, Any]:
    variants = case_item.get("variants")
    if not isinstance(variants, dict) or "default" not in variants:
        return {
            "size": size_label,
            "case": case_label,
            "status": "missing_default_variant",
            "variants": [],
        }
    default = variants["default"]
    python_timing = case_item.get("python_timing")
    python_timing = python_timing if isinstance(python_timing, dict) else {}
    python_full = number(python_timing.get("full_cache_step_ms"))
    python_cached_propagate = number(python_timing.get("cached_propagate_ms"))
    python_required = number(python_timing.get("required_e2e_ms"))
    python_backbone = number(python_timing.get("required_image_encode_or_backbone_ms"))
    baseline_full = metric_mean(default, "full_frame_step_ms")
    baseline_full_sd = metric_stdev(default, "full_frame_step_ms")
    baseline_full_n = metric_count(default, "full_frame_step_ms")
    decision_metric = (
        "run_required_e2e_ms"
        if metric_mean(default, "run_required_e2e_ms") is not None
        else "full_frame_step_ms"
    )
    baseline_decision = metric_mean(default, decision_metric)
    baseline_decision_sd = metric_stdev(default, decision_metric)
    baseline_decision_n = metric_count(default, decision_metric)
    baseline_encode = metric_mean(default, "required_image_encode_ms")
    if baseline_encode is None:
        baseline_encode = metric_mean(default, "two_frame_encode_ms")
    baseline_encode_graph = metric_mean(default, "required_image_encode_graph_compute_ms")
    baseline_tail_encode = metric_mean(default, "tail_image_encode_total_ms")
    baseline_tail_encode_graph = metric_mean(default, "tail_encode_graph_compute_total_ms")
    baseline_prop_encoded = metric_mean(default, "propagate_encoded_ms")
    variant_rows: list[dict[str, Any]] = []
    for variant_name, raw_item in variants.items():
        if not isinstance(raw_item, dict):
            continue
        full = metric_mean(raw_item, "full_frame_step_ms")
        encode = metric_mean(raw_item, "required_image_encode_ms")
        if encode is None:
            encode = metric_mean(raw_item, "two_frame_encode_ms")
        encode_graph = metric_mean(raw_item, "required_image_encode_graph_compute_ms")
        tail_encode = metric_mean(raw_item, "tail_image_encode_total_ms")
        tail_encode_graph = metric_mean(raw_item, "tail_encode_graph_compute_total_ms")
        prop_encoded = metric_mean(raw_item, "propagate_encoded_ms")
        full_delta = None if full is None or baseline_full is None else full - baseline_full
        encode_delta = None if encode is None or baseline_encode is None else encode - baseline_encode
        encode_graph_delta = (
            None
            if encode_graph is None or baseline_encode_graph is None
            else encode_graph - baseline_encode_graph
        )
        tail_encode_delta = (
            None
            if tail_encode is None or baseline_tail_encode is None
            else tail_encode - baseline_tail_encode
        )
        tail_encode_graph_delta = (
            None
            if tail_encode_graph is None or baseline_tail_encode_graph is None
            else tail_encode_graph - baseline_tail_encode_graph
        )
        prop_delta = (
            None
            if prop_encoded is None or baseline_prop_encoded is None
            else prop_encoded - baseline_prop_encoded
        )
        decision_value = metric_mean(raw_item, decision_metric)
        decision_delta = (
            None
            if decision_value is None or baseline_decision is None
            else decision_value - baseline_decision
        )
        signal = delta_signal(
            baseline_decision_sd,
            baseline_decision_n,
            metric_stdev(raw_item, decision_metric),
            metric_count(raw_item, decision_metric),
            decision_delta,
        )
        same_as_default, quality = variant_quality_status(raw_item)
        if variant_name == "default":
            same_as_default = True
        variant_rows.append(
            {
                "variant": variant_name,
                "decision": decision_for_variant(
                    variant=variant_name,
                    same_as_default=same_as_default,
                    full_delta_ms=decision_delta,
                    full_delta_signal=signal,
                ),
                "decision_metric": decision_metric,
                "decision_metric_ms": decision_value,
                "decision_metric_delta_ms": decision_delta,
                "decision_metric_delta_signal": signal,
                "run_required_e2e_ms": metric_mean(raw_item, "run_required_e2e_ms"),
                "run_required_e2e_delta_ms": (
                    None
                    if metric_mean(raw_item, "run_required_e2e_ms") is None
                    or metric_mean(default, "run_required_e2e_ms") is None
                    else metric_mean(raw_item, "run_required_e2e_ms")
                    - metric_mean(default, "run_required_e2e_ms")
                ),
                "full_frame_step_ms": full,
                "full_frame_step_delta_ms": full_delta,
                "full_frame_step_delta_signal": delta_signal(
                    baseline_full_sd,
                    baseline_full_n,
                    metric_stdev(raw_item, "full_frame_step_ms"),
                    metric_count(raw_item, "full_frame_step_ms"),
                    full_delta,
                ),
                "image_encode_ms": encode,
                "image_encode_delta_ms": encode_delta,
                "image_encode_graph_compute_ms": encode_graph,
                "image_encode_graph_compute_delta_ms": encode_graph_delta,
                "image_encode_graph_compute_delta_signal": delta_signal(
                    metric_stdev(default, "required_image_encode_graph_compute_ms"),
                    metric_count(default, "required_image_encode_graph_compute_ms"),
                    metric_stdev(raw_item, "required_image_encode_graph_compute_ms"),
                    metric_count(raw_item, "required_image_encode_graph_compute_ms"),
                    encode_graph_delta,
                ),
                "tail_image_encode_ms": tail_encode,
                "tail_image_encode_delta_ms": tail_encode_delta,
                "tail_image_encode_graph_compute_ms": tail_encode_graph,
                "tail_image_encode_graph_compute_delta_ms": tail_encode_graph_delta,
                "tail_image_encode_graph_compute_delta_signal": delta_signal(
                    metric_stdev(default, "tail_encode_graph_compute_total_ms"),
                    metric_count(default, "tail_encode_graph_compute_total_ms"),
                    metric_stdev(raw_item, "tail_encode_graph_compute_total_ms"),
                    metric_count(raw_item, "tail_encode_graph_compute_total_ms"),
                    tail_encode_graph_delta,
                ),
                "two_frame_encode_ms": encode,
                "two_frame_encode_delta_ms": encode_delta,
                "required_session_setup_ms": metric_mean(
                    raw_item, "required_session_setup_ms"
                ),
                "required_input_prepare_ms": metric_mean(raw_item, "required_input_prepare_ms"),
                "required_model_execute_ms": metric_mean(
                    raw_item, "required_model_execute_ms"
                ),
                "required_image_encode_ms": metric_mean(raw_item, "required_image_encode_ms"),
                "required_mask_init_ms": metric_mean(raw_item, "required_mask_init_ms"),
                "required_propagate_encoded_ms": metric_mean(
                    raw_item, "required_propagate_encoded_ms"
                ),
                "required_model_accounted_ms": metric_mean(
                    raw_item, "required_model_accounted_ms"
                ),
                "required_model_remainder_ms": metric_mean(
                    raw_item, "required_model_remainder_ms"
                ),
                "required_core_compute_ms": metric_mean(raw_item, "required_core_compute_ms"),
                "required_non_image_model_ms": metric_mean(
                    raw_item, "required_non_image_model_ms"
                ),
                "required_non_graph_or_setup_ms": metric_mean(
                    raw_item, "required_non_graph_or_setup_ms"
                ),
                "mask_foreground_pixels": metric_mean(raw_item, "mask_foreground_pixels"),
                "selected_mask_index": metric_mean(raw_item, "selected_mask_index"),
                "decoder_iou_scores": metric_list_means(raw_item, "decoder_iou_scores"),
                "propagate_encoded_ms": prop_encoded,
                "propagate_encoded_delta_ms": prop_delta,
                "python_full_cache_over_cpp_full_ratio": ratio(python_full, full),
                "python_full_cache_over_cpp_required_e2e_ratio": ratio(
                    python_full,
                    metric_mean(raw_item, "run_required_e2e_ms"),
                ),
                "python_required_e2e_over_cpp_required_e2e_ratio": ratio(
                    python_required,
                    metric_mean(raw_item, "run_required_e2e_ms"),
                ),
                "python_backbone_over_cpp_image_encode_ratio": ratio(
                    python_backbone,
                    encode,
                ),
                "python_backbone_over_cpp_image_graph_ratio": ratio(
                    python_backbone,
                    encode_graph,
                ),
                "python_cached_propagate_over_cpp_encoded_ratio": ratio(
                    python_cached_propagate,
                    prop_encoded,
                ),
                "python_mask_iou_min": metric_min_prefer(
                    raw_item,
                    ("python_sequence_iou", "python_iou"),
                ),
                "python_mask_xor_max": metric_max_prefer(
                    raw_item,
                    ("python_sequence_xor_pixels", "python_xor_pixels"),
                ),
                "python_frame1_mask_iou_min": number(metric(raw_item, "python_iou").get("min")),
                "python_frame1_mask_xor_max": number(
                    metric(raw_item, "python_xor_pixels").get("max")
                ),
                "python_sequence_mask_iou_min": number(
                    metric(raw_item, "python_sequence_iou").get("min")
                ),
                "python_sequence_mask_xor_max": number(
                    metric(raw_item, "python_sequence_xor_pixels").get("max")
                ),
                "sequence_frame_count": number(raw_item.get("python_sequence_frame_count")),
                "same_as_default": same_as_default,
                **quality,
            }
        )
    variant_rows.sort(
        key=lambda row: (
            0 if row["variant"] == "default" else 1,
            row["decision_metric_delta_ms"]
            if row["decision_metric_delta_ms"] is not None
            else float("inf"),
        )
    )
    default_process_split = process_split_rows(default)
    default_required_split = required_e2e_split_rows(default)
    default_internal_split = default_internal_timing_rows(default)
    python_required_split = python_required_e2e_split_rows(python_timing)
    profile_stage_rows = cpp_profile_stage_rows(case_item)
    image_component_rows, image_component_meta = cpp_profile_image_component_rows(
        case_item, default
    )
    encode_total = metric_mean(default, "required_image_encode_ms")
    if encode_total is None:
        encode_total = metric_mean(default, "two_frame_encode_ms")
    encode_graph_total = metric_mean(default, "required_image_encode_graph_compute_ms")
    full = metric_mean(default, "full_frame_step_ms")
    required = metric_mean(default, "run_required_e2e_ms")
    return {
        "size": size_label,
        "case": case_label,
        "status": "ok",
        "python_timing": python_timing,
        "python_required_e2e_split": python_required_split,
        "default_profile_stage_attribution": profile_stage_rows,
        "default_image_component_attribution": image_component_rows,
        "default_image_component_attribution_meta": image_component_meta,
        "cpp_profile": case_item.get("cpp_profile") if isinstance(case_item.get("cpp_profile"), dict) else None,
        "decision_metric": decision_metric,
        "default_required_e2e_split": default_required_split,
        "default_internal_timing_split": default_internal_split,
        "default_process_split": default_process_split,
        "default_encode_share_pct": pct(encode_total, full),
        "default_required_encode_share_pct": pct(encode_total, required),
        "default_required_encode_graph_share_pct": pct(encode_graph_total, required),
        "variants": variant_rows,
    }


def summarize_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    case_summaries: list[dict[str, Any]] = []
    sizes = matrix.get("sizes")
    if isinstance(sizes, dict):
        for size_label, size_item in sizes.items():
            if not isinstance(size_item, dict):
                continue
            cases = size_item.get("cases")
            if not isinstance(cases, dict):
                continue
            for case_label, case_item in cases.items():
                if isinstance(case_item, dict):
                    case_summaries.append(
                        summarize_case(
                            size_label=str(size_label),
                            case_label=str(case_label),
                            case_item=case_item,
                        )
                    )
    candidate_wins = [
        row
        for case in case_summaries
        for row in case.get("variants", [])
        if isinstance(row, dict) and row.get("decision") == "candidate_win"
    ]
    return {
        "status": "ok" if case_summaries else "missing_cases",
        "matrix_status": matrix.get("status"),
        "contract": {
            "cases": matrix.get("cases"),
            "sizes": list(sizes.keys()) if isinstance(sizes, dict) else [],
            "variants": matrix.get("variants"),
            "repeats": matrix.get("repeats"),
            "python": matrix.get("python"),
            "cpp": matrix.get("cpp"),
        },
        "case_summaries": case_summaries,
        "candidate_wins": candidate_wins,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SAM3.1 Mask-Init E2E Optimization Targets",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Matrix status: `{result.get('matrix_status')}`",
        f"- Repeats: `{result.get('contract', {}).get('repeats')}`",
        f"- Variants: `{', '.join(result.get('contract', {}).get('variants') or [])}`",
        "",
    ]
    for case in result.get("case_summaries", []):
        if not isinstance(case, dict):
            continue
        lines.extend(
            [
                f"## {case.get('size')} {case.get('case')}",
                "",
                f"- Decision metric: `{case.get('decision_metric')}`",
                f"- Default image-encode share: `{fmt_pct(case.get('default_encode_share_pct'))}`",
                f"- Default required-E2E image-encode share: "
                f"`{fmt_pct(case.get('default_required_encode_share_pct'))}`",
                f"- Default required-E2E image-encode graph share: "
                f"`{fmt_pct(case.get('default_required_encode_graph_share_pct'))}`",
                "",
                "### Official Python Required E2E Split",
                "",
                "| process | mean ms | Python required E2E % | note |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        python_split = case.get("python_required_e2e_split", [])
        if python_split:
            for row in python_split:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| `{row.get('process')}` | {fmt_ms(row.get('mean_ms'))} | "
                    f"{fmt_pct(row.get('e2e_pct'))} | {row.get('note')} |"
                )
        else:
            lines.append("| `python.required_e2e` | - | - | unavailable in this matrix |")
        lines.extend(
            [
                "",
                "### Default Required E2E Split",
                "",
                "| process | mean ms | required E2E % | note |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for row in case.get("default_required_e2e_split", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| `{row.get('process')}` | {fmt_ms(row.get('mean_ms'))} | "
                f"{fmt_pct(row.get('e2e_pct'))} | {row.get('note')} |"
            )
        internal_split = case.get("default_internal_timing_split", [])
        if internal_split:
            lines.extend(
                [
                    "",
                    "### Default Internal Timing Split",
                    "",
                    (
                        "| process | total | preprocess | graph build | graph alloc | "
                        "input upload | graph compute | state/PE/output |"
                    ),
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in internal_split:
                if not isinstance(row, dict):
                    continue
                state_pe_output = (
                    number(row.get("state_update")) or 0.0
                ) + (number(row.get("pe_build")) or 0.0) + (number(row.get("output_read")) or 0.0)
                lines.append(
                    f"| `{row.get('process')}` | {fmt_ms(row.get('total'))} | "
                    f"{fmt_ms(row.get('preprocess'))} | "
                    f"{fmt_ms(row.get('graph_build'))} | "
                    f"{fmt_ms(row.get('graph_alloc'))} | "
                    f"{fmt_ms(row.get('input_upload'))} | "
                    f"{fmt_ms(row.get('graph_compute'))} | "
                    f"{fmt_ms(state_pe_output)} |"
                )
        lines.extend(
            [
                "",
                "### Default CUDA Stage Attribution",
                "",
            ]
        )
        profile_rows = case.get("default_profile_stage_attribution", [])
        if profile_rows:
            lines.extend(
                [
                    (
                        "This profiling pass uses synchronized CUDA-node timing for attribution; "
                        "normal variant rows remain the E2E speed source."
                    ),
                    "",
                    "| mode | stage | mean ms | drop-max mean ms | top ops |",
                    "| --- | --- | ---: | ---: | --- |",
                ]
            )
            for row in profile_rows:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| `{row.get('mode')}` | `{row.get('stage')}` | "
                    f"{fmt_ms(row.get('mean_ms'))} | "
                    f"{fmt_ms(row.get('mean_ms_drop_max'))} | "
                    f"{fmt_ops(row.get('top_ops'))} |"
                )
        else:
            lines.append(
                "No synchronized CUDA stage profile was attached. Re-run the matrix with "
                "`--profile-default` to add attribution."
            )
        image_component_rows = case.get("default_image_component_attribution", [])
        if image_component_rows:
            meta = case.get("default_image_component_attribution_meta", {})
            meta = meta if isinstance(meta, dict) else {}
            lines.extend(
                [
                    "",
                    "### Default Image Encoder Component Attribution",
                    "",
                    (
                        "The profile rows are synchronized and are used only for ratios. "
                        "Projected ms applies those ratios to the normal, non-profiled "
                        "`required_image_encode_graph_compute_ms` denominator."
                    ),
                    "",
                    f"- Profile labels: `{', '.join(meta.get('selected_labels') or [])}`",
                    f"- Normal image graph denominator: "
                    f"`{fmt_ms(meta.get('normal_image_graph_ms'))} ms`",
                    "",
                    "| component | profile raw ms | image graph % | projected normal ms | required E2E % | top profiled stages |",
                    "| --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in image_component_rows:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| `{row.get('component')}` | "
                    f"{fmt_ms(row.get('profile_ms'))} | "
                    f"{fmt_pct(row.get('profile_image_graph_pct'))} | "
                    f"{fmt_ms(row.get('projected_normal_ms'))} | "
                    f"{fmt_pct(row.get('required_e2e_pct'))} | "
                    f"{fmt_stage_list(row.get('top_profile_stages'))} |"
                )
        lines.extend(
            [
                "",
                "### Default Model Step Split",
                "",
                "| process | mean ms | E2E % | note |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for row in case.get("default_process_split", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| `{row.get('process')}` | {fmt_ms(row.get('mean_ms'))} | "
                f"{fmt_pct(row.get('full_frame_step_pct'))} | {row.get('note')} |"
            )
        lines.extend(
            [
                "",
                "### Variant Decisions",
                "",
                (
                    "| variant | decision | required ms | delta ms | signal | image encode ms | "
                    "image graph delta | tail encode ms | tail graph delta | prop encoded ms | "
                    "sel idx | fg px | decoder scores | Py req/C++ req | Py bb/C++ graph | "
                    "Py mask IoU min | same default |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in case.get("variants", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| `{row.get('variant')}` | `{row.get('decision')}` | "
                f"{fmt_ms(row.get('run_required_e2e_ms'))} | "
                f"{fmt_ms(row.get('decision_metric_delta_ms'))} | "
                f"{fmt_signal(row.get('decision_metric_delta_signal'))} | "
                f"{fmt_ms(row.get('image_encode_ms'))} | "
                f"{fmt_ms(row.get('image_encode_graph_compute_delta_ms'))} | "
                f"{fmt_ms(row.get('tail_image_encode_ms'))} | "
                f"{fmt_ms(row.get('tail_image_encode_graph_compute_delta_ms'))} | "
                f"{fmt_ms(row.get('propagate_encoded_ms'))} | "
                f"{fmt_count(row.get('selected_mask_index'))} | "
                f"{fmt_count(row.get('mask_foreground_pixels'))} | "
                f"`{fmt_float_list(row.get('decoder_iou_scores'))}` | "
                f"{fmt_ratio(row.get('python_required_e2e_over_cpp_required_e2e_ratio'))} | "
                f"{fmt_ratio(row.get('python_backbone_over_cpp_image_graph_ratio'))} | "
                f"{fmt_float(row.get('python_mask_iou_min'))} | "
                f"`{row.get('same_as_default')}` |"
            )
        lines.append("")
    if result.get("candidate_wins"):
        lines.extend(["## Accepted Candidates", ""])
        for row in result["candidate_wins"]:
            lines.append(
                f"- `{row.get('variant')}`: `{row.get('decision_metric')}` delta "
                f"{fmt_ms(row.get('decision_metric_delta_ms'))}, signal "
                f"{fmt_signal(row.get('decision_metric_delta_signal'))}"
            )
    else:
        lines.extend(
            [
                "## Accepted Candidates",
                "",
                "No variant met the same-default-mask and statistically meaningful E2E win gates.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    result = summarize_matrix(load_json(args.matrix))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result), encoding="utf-8")
    print(
        "sam31_mask_init_audit_status="
        f"{result['status']} cases={len(result['case_summaries'])} "
        f"candidate_wins={len(result['candidate_wins'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
