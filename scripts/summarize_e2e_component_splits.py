# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOP_LEVEL_COMPONENTS = (
    "required_e2e_ms",
    "model_e2e_ms",
    "input_load_ms",
    "input_directory_scan_ms",
    "input_sort_ms",
    "input_frame_decode_ms",
    "input_frame_store_ms",
    "input_remainder_ms",
    "session_setup_ms",
    "frame0_encode_ms",
    "frame0_prompt_model_ms",
    "frame0_segment_ms",
    "frame0_tracker_mask_prepare_ms",
    "frame0_tracker_memory_encode_ms",
    "frame0_tracker_obj_ptr_ms",
    "frame0_tracker_store_ms",
    "frame0_prompt_unattributed_ms",
    "tail_runtime_wall_ms",
    "tail_visible_model_ms",
    "tail_preencode_required_model_ms",
    "tail_required_model_ms",
    "tail_encode_required_total_ms",
    "required_tail_state_create_ms",
    "required_tail_propagate_ms",
    "required_tail_unattributed_ms",
    "model_remainder_ms",
    "tail_inline_encode_ms",
    "tail_preencode_encode_ms",
    "tail_inline_propagate_ms",
    "tail_preencoded_propagate_ms",
    "frame0_encode_graph_compute_ms",
    "frame0_encode_preprocess_ms",
    "frame0_encode_graph_admin_ms",
    "frame0_encode_graph_build_ms",
    "frame0_encode_graph_alloc_ms",
    "frame0_encode_input_upload_ms",
    "frame0_encode_state_update_ms",
    "frame0_encode_pe_build_ms",
    "frame0_encode_remainder_ms",
    "tail_inline_encode_graph_compute_ms",
    "tail_preencode_encode_graph_compute_ms",
    "tail_encode_graph_compute_ms",
    "tail_encode_non_compute_ms",
    "tail_encode_preprocess_ms",
    "tail_encode_graph_admin_ms",
    "tail_encode_input_upload_ms",
    "tail_encode_state_update_ms",
    "tail_encode_pe_build_ms",
    "tail_encode_remainder_ms",
    "tail_inline_encode_graph_admin_ms",
    "tail_inline_encode_non_compute_ms",
    "tail_inline_encode_preprocess_ms",
    "tail_inline_encode_input_upload_ms",
    "tail_inline_encode_graph_build_ms",
    "tail_inline_encode_graph_alloc_ms",
    "tail_inline_encode_state_update_ms",
    "tail_inline_encode_pe_build_ms",
    "tail_inline_encode_remainder_ms",
    "tail_preencode_encode_graph_admin_ms",
    "tail_preencode_encode_non_compute_ms",
    "tail_preencode_encode_preprocess_ms",
    "tail_preencode_encode_input_upload_ms",
    "tail_preencode_encode_graph_build_ms",
    "tail_preencode_encode_graph_alloc_ms",
    "tail_preencode_encode_state_update_ms",
    "tail_preencode_encode_pe_build_ms",
    "tail_preencode_encode_remainder_ms",
    "required_encode_graph_admin_ms",
    "required_encode_preprocess_ms",
    "required_encode_state_update_ms",
    "required_encode_pe_build_ms",
    "required_encode_remainder_ms",
    "required_encode_non_compute_ms",
    "tail_propagate_graph_admin_ms",
    "tail_propagate_graph_compute_ms",
    "tail_propagate_cache_compute_ms",
    "tail_propagate_input_upload_ms",
    "tail_propagate_input_prompt_upload_ms",
    "tail_propagate_input_rope_upload_ms",
    "tail_propagate_input_memory_upload_ms",
    "tail_propagate_input_feature_upload_ms",
    "tail_propagate_input_constant_upload_ms",
    "tail_propagate_input_sparse_upload_ms",
    "required_core_compute_ms",
    "required_non_core_compute_ms",
)

FRAME_TIMING_COMPONENTS = (
    ("input_total_ms", "input_ms", "total_ms"),
    ("input_directory_scan_ms", "input_ms", "directory_scan_ms"),
    ("input_sort_ms", "input_ms", "sort_ms"),
    ("input_frame_decode_ms", "input_ms", "frame_decode_ms"),
    ("input_frame_store_ms", "input_ms", "frame_store_ms"),
    ("input_accounted_ms", "input_ms", "accounted_ms"),
    ("input_remainder_ms", "input_ms", "remainder_ms"),
    ("frame0_graph_compute_ms", "frame0_ms", "encode_graph_compute_ms"),
    ("frame0_encode_preprocess_ms", "frame0_ms", "encode_preprocess_ms"),
    ("frame0_graph_build_ms", "frame0_ms", "encode_graph_build_ms"),
    ("frame0_graph_alloc_ms", "frame0_ms", "encode_graph_alloc_ms"),
    ("frame0_input_upload_ms", "frame0_ms", "encode_input_upload_ms"),
    ("frame0_encode_state_update_ms", "frame0_ms", "encode_state_update_ms"),
    ("frame0_encode_pe_build_ms", "frame0_ms", "encode_pe_build_ms"),
    ("frame0_encode_remainder_ms", "frame0_ms", "encode_remainder_ms"),
    ("tail_graph_compute_ms", "tail_all_required_ms", "encode_graph_compute_ms"),
    ("tail_encode_preprocess_ms", "tail_all_required_ms", "encode_preprocess_ms"),
    ("tail_graph_build_ms", "tail_all_required_ms", "encode_graph_build_ms"),
    ("tail_graph_alloc_ms", "tail_all_required_ms", "encode_graph_alloc_ms"),
    ("tail_input_upload_ms", "tail_all_required_ms", "encode_input_upload_ms"),
    ("tail_encode_state_update_ms", "tail_all_required_ms", "encode_state_update_ms"),
    ("tail_encode_pe_build_ms", "tail_all_required_ms", "encode_pe_build_ms"),
    ("tail_encode_remainder_ms", "tail_all_required_ms", "encode_remainder_ms"),
    ("tail_inline_required_wall_ms", "tail_inline_required_ms", "required_wall_ms"),
    ("tail_inline_visible_model_ms", "tail_inline_required_ms", "visible_model_ms"),
    ("tail_inline_encode_ms", "tail_inline_required_ms", "inline_encode_ms"),
    (
        "tail_inline_encode_graph_compute_ms",
        "tail_inline_required_ms",
        "encode_graph_compute_ms",
    ),
    (
        "tail_inline_encode_preprocess_ms",
        "tail_inline_required_ms",
        "encode_preprocess_ms",
    ),
    (
        "tail_inline_encode_input_upload_ms",
        "tail_inline_required_ms",
        "encode_input_upload_ms",
    ),
    (
        "tail_inline_encode_graph_build_ms",
        "tail_inline_required_ms",
        "encode_graph_build_ms",
    ),
    (
        "tail_inline_encode_graph_alloc_ms",
        "tail_inline_required_ms",
        "encode_graph_alloc_ms",
    ),
    (
        "tail_inline_encode_state_update_ms",
        "tail_inline_required_ms",
        "encode_state_update_ms",
    ),
    (
        "tail_inline_encode_pe_build_ms",
        "tail_inline_required_ms",
        "encode_pe_build_ms",
    ),
    (
        "tail_inline_encode_remainder_ms",
        "tail_inline_required_ms",
        "encode_remainder_ms",
    ),
    ("tail_inline_propagate_ms", "tail_inline_required_ms", "propagate_ms"),
    (
        "tail_preencoded_runtime_wall_ms",
        "tail_preencoded_required_ms",
        "runtime_wall_ms",
    ),
    (
        "tail_preencoded_visible_model_ms",
        "tail_preencoded_required_ms",
        "visible_model_ms",
    ),
    (
        "tail_preencode_state_create_ms",
        "tail_preencoded_required_ms",
        "preencode_state_create_ms",
    ),
    ("tail_preencode_encode_ms", "tail_preencoded_required_ms", "preencode_ms"),
    (
        "tail_preencode_encode_graph_compute_ms",
        "tail_preencoded_required_ms",
        "encode_graph_compute_ms",
    ),
    (
        "tail_preencode_encode_preprocess_ms",
        "tail_preencoded_required_ms",
        "encode_preprocess_ms",
    ),
    (
        "tail_preencode_encode_input_upload_ms",
        "tail_preencoded_required_ms",
        "encode_input_upload_ms",
    ),
    (
        "tail_preencode_encode_graph_build_ms",
        "tail_preencoded_required_ms",
        "encode_graph_build_ms",
    ),
    (
        "tail_preencode_encode_graph_alloc_ms",
        "tail_preencoded_required_ms",
        "encode_graph_alloc_ms",
    ),
    (
        "tail_preencode_encode_state_update_ms",
        "tail_preencoded_required_ms",
        "encode_state_update_ms",
    ),
    (
        "tail_preencode_encode_pe_build_ms",
        "tail_preencoded_required_ms",
        "encode_pe_build_ms",
    ),
    (
        "tail_preencode_encode_remainder_ms",
        "tail_preencoded_required_ms",
        "encode_remainder_ms",
    ),
    (
        "tail_preencoded_propagate_ms",
        "tail_preencoded_required_ms",
        "propagate_ms",
    ),
    ("tail_runtime_wall_sum_ms", "tail_all_required_ms", "runtime_wall_ms"),
    ("tail_visible_model_sum_ms", "tail_all_required_ms", "visible_model_ms"),
    (
        "tail_preencode_required_model_sum_ms",
        "tail_all_required_ms",
        "preencode_required_model_ms",
    ),
    ("tail_required_model_sum_ms", "tail_all_required_ms", "required_model_ms"),
    ("tail_required_wall_sum_ms", "tail_all_required_ms", "required_wall_ms"),
    (
        "tail_propagate_graph_compute_ms",
        "tail_all_required_ms",
        "propagate_graph_compute_ms",
    ),
    (
        "tail_propagate_graph_build_ms",
        "tail_all_required_ms",
        "propagate_graph_build_ms",
    ),
    (
        "tail_propagate_graph_alloc_ms",
        "tail_all_required_ms",
        "propagate_graph_alloc_ms",
    ),
    (
        "tail_propagate_input_upload_ms",
        "tail_all_required_ms",
        "propagate_input_upload_ms",
    ),
    (
        "tail_propagate_input_prompt_upload_ms",
        "tail_all_required_ms",
        "propagate_input_prompt_upload_ms",
    ),
    (
        "tail_propagate_input_rope_upload_ms",
        "tail_all_required_ms",
        "propagate_input_rope_upload_ms",
    ),
    (
        "tail_propagate_input_memory_upload_ms",
        "tail_all_required_ms",
        "propagate_input_memory_upload_ms",
    ),
    (
        "tail_propagate_input_feature_upload_ms",
        "tail_all_required_ms",
        "propagate_input_feature_upload_ms",
    ),
    (
        "tail_propagate_input_constant_upload_ms",
        "tail_all_required_ms",
        "propagate_input_constant_upload_ms",
    ),
    (
        "tail_propagate_input_sparse_upload_ms",
        "tail_all_required_ms",
        "propagate_input_sparse_upload_ms",
    ),
    (
        "tail_prop_alias_feature_count",
        "tail_all_required_ms",
        "propagate_aliased_feature_input_count",
    ),
    (
        "tail_prop_upload_feature_count",
        "tail_all_required_ms",
        "propagate_uploaded_feature_input_count",
    ),
    (
        "tail_prop_alias_constant_count",
        "tail_all_required_ms",
        "propagate_aliased_constant_input_count",
    ),
    (
        "tail_prop_upload_constant_count",
        "tail_all_required_ms",
        "propagate_uploaded_constant_input_count",
    ),
    (
        "tail_propagate_prompt_build_ms",
        "tail_all_required_ms",
        "propagate_prompt_build_ms",
    ),
    (
        "tail_propagate_rope_cache_ms",
        "tail_all_required_ms",
        "propagate_rope_cache_ms",
    ),
    (
        "tail_propagate_rope_k_build_ms",
        "tail_all_required_ms",
        "propagate_rope_k_build_ms",
    ),
    (
        "tail_propagate_output_read_ms",
        "tail_all_required_ms",
        "propagate_output_read_ms",
    ),
    (
        "tail_propagate_bbox_from_logits_ms",
        "tail_all_required_ms",
        "propagate_bbox_from_logits_ms",
    ),
    (
        "tail_propagate_cache_compute_ms",
        "tail_all_required_ms",
        "propagate_graph_cache_compute_ms",
    ),
    (
        "tail_propagate_cache_input_upload_ms",
        "tail_all_required_ms",
        "propagate_graph_cache_input_upload_ms",
    ),
    (
        "tail_propagate_memory_update_ms",
        "tail_all_required_ms",
        "propagate_memory_update_ms",
    ),
    ("tail_propagate_accounted_ms", "tail_all_required_ms", "propagate_accounted_ms"),
    ("tail_propagate_remainder_ms", "tail_all_required_ms", "propagate_remainder_ms"),
)

TABLE_COLUMNS = (
    ("required_e2e_ms", "ms"),
    ("model_e2e_ms", "ms"),
    ("input_load_ms", "ms"),
    ("input_frame_decode_ms", "ms"),
    ("frame0_encode_ms", "ms"),
    ("frame0_prompt_model_ms", "ms"),
    ("frame0_segment_ms", "ms"),
    ("tail_runtime_wall_ms", "ms"),
    ("tail_visible_model_ms", "ms"),
    ("tail_preencode_required_model_ms", "ms"),
    ("tail_required_model_ms", "ms"),
    ("tail_encode_required_total_ms", "ms"),
    ("tail_graph_compute_ms", "ms"),
    ("tail_inline_encode_ms", "ms"),
    ("tail_inline_encode_graph_compute_ms", "ms"),
    ("tail_preencode_encode_ms", "ms"),
    ("tail_preencode_encode_graph_compute_ms", "ms"),
    ("tail_preencode_state_create_ms", "ms"),
    ("tail_encode_required_accounting_delta_ms", "ms"),
    ("tail_encode_graph_compute_accounting_delta_ms", "ms"),
    ("tail_encode_non_compute_ms", "ms"),
    ("tail_encode_preprocess_ms", "ms"),
    ("tail_encode_graph_admin_ms", "ms"),
    ("tail_encode_input_upload_ms", "ms"),
    ("tail_encode_state_update_ms", "ms"),
    ("tail_encode_pe_build_ms", "ms"),
    ("tail_encode_remainder_ms", "ms"),
    ("tail_inline_encode_non_compute_ms", "ms"),
    ("tail_preencode_encode_non_compute_ms", "ms"),
    ("required_encode_graph_admin_ms", "ms"),
    ("required_encode_non_compute_ms", "ms"),
    ("required_tail_propagate_ms", "ms"),
    ("tail_propagate_graph_compute_ms", "ms"),
    ("tail_propagate_graph_admin_ms", "ms"),
    ("tail_propagate_input_upload_ms", "ms"),
    ("tail_propagate_input_prompt_upload_ms", "ms"),
    ("tail_propagate_input_rope_upload_ms", "ms"),
    ("tail_propagate_input_memory_upload_ms", "ms"),
    ("tail_propagate_input_feature_upload_ms", "ms"),
    ("tail_propagate_input_constant_upload_ms", "ms"),
    ("tail_propagate_input_sparse_upload_ms", "ms"),
    ("tail_required_model_accounting_delta_ms", "ms"),
    ("required_graph_admin_ms", "ms"),
    ("tail_prop_alias_feature_count", "count"),
    ("tail_prop_upload_feature_count", "count"),
    ("tail_prop_alias_constant_count", "count"),
    ("tail_prop_upload_constant_count", "count"),
    ("required_core_compute_ms", "ms"),
    ("required_core_compute_pct", "pct"),
    ("tail_encode_graph_compute_pct", "pct"),
)

PROCESS_PAIR_KEYS = (
    "comparable_session_e2e_rows",
    "same_contract_session_e2e_diagnostic_rows",
    "comparable_model_e2e_rows",
    "same_contract_model_e2e_diagnostic_rows",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def metric_with_stats(row: dict[str, Any], key: str) -> dict[str, Any]:
    mean = numeric(row.get(key))
    stats = row.get(f"{key}_stats")
    if not isinstance(stats, dict):
        return {"mean_ms": mean}
    return {
        "mean_ms": numeric(stats.get("mean", mean)),
        "median_ms": numeric(stats.get("median")),
        "sd_ms": numeric(stats.get("sd")),
        "min_ms": numeric(stats.get("min")),
        "max_ms": numeric(stats.get("max")),
        "n": int(stats["n"]) if isinstance(stats.get("n"), int | float) else None,
    }


def frame_metric_with_stats(
    frame_timing: dict[str, Any], output_key: str, scope: str, metric: str
) -> dict[str, Any]:
    mean = numeric(nested_get(frame_timing, ("mean", scope, metric)))
    stats = nested_get(frame_timing, ("stats", scope, metric))
    if not isinstance(stats, dict):
        return {"mean_ms": mean}
    return {
        "mean_ms": numeric(stats.get("mean", mean)),
        "median_ms": numeric(stats.get("median")),
        "sd_ms": numeric(stats.get("sd")),
        "min_ms": numeric(stats.get("min")),
        "max_ms": numeric(stats.get("max")),
        "n": int(stats["n"]) if isinstance(stats.get("n"), int | float) else None,
        "source_metric": f"{scope}.{metric}",
        "component": output_key,
    }


def component_mean(components: dict[str, dict[str, Any]], key: str) -> float:
    value = components.get(key, {}).get("mean_ms")
    return float(value) if isinstance(value, int | float) else 0.0


def component_number(components: dict[str, dict[str, Any]], key: str) -> float | None:
    value = components.get(key, {}).get("mean_ms")
    return float(value) if isinstance(value, int | float) else None


def component_stat(
    components: dict[str, dict[str, Any]], key: str, stat: str
) -> float | None:
    value = components.get(key, {}).get(stat)
    return float(value) if isinstance(value, int | float) else None


def component_sum_if_any(
    components: dict[str, dict[str, Any]], keys: tuple[str, ...]
) -> float | None:
    values = [component_number(components, key) for key in keys]
    if not any(value is not None for value in values):
        return None
    return sum(value for value in values if value is not None)


def set_mean_component(
    components: dict[str, dict[str, Any]], key: str, value: float | None
) -> None:
    components[key] = {"mean_ms": value}


def alias_component(
    components: dict[str, dict[str, Any]], target: str, source: str
) -> None:
    if component_number(components, target) is not None:
        return
    if component_number(components, source) is None:
        return
    components[target] = components[source] | {
        "component": target,
        "source_component": source,
    }


def add_derived_components(components: dict[str, dict[str, Any]]) -> None:
    alias_component(
        components, "frame0_graph_compute_ms", "frame0_encode_graph_compute_ms"
    )
    alias_component(components, "frame0_graph_build_ms", "frame0_encode_graph_build_ms")
    alias_component(components, "frame0_graph_alloc_ms", "frame0_encode_graph_alloc_ms")
    alias_component(
        components, "frame0_input_upload_ms", "frame0_encode_input_upload_ms"
    )
    alias_component(components, "tail_graph_compute_ms", "tail_encode_graph_compute_ms")
    alias_component(components, "tail_graph_build_ms", "tail_encode_graph_build_ms")
    alias_component(components, "tail_graph_alloc_ms", "tail_encode_graph_alloc_ms")
    alias_component(components, "tail_input_upload_ms", "tail_encode_input_upload_ms")
    alias_component(components, "tail_encode_input_upload_ms", "tail_input_upload_ms")
    required_e2e = component_mean(components, "required_e2e_ms")
    required_e2e_value = component_number(components, "required_e2e_ms")
    tail_encode_required = component_mean(components, "tail_encode_required_total_ms")
    tail_encode_required_value = component_number(
        components, "tail_encode_required_total_ms"
    )
    core_component_keys = (
        "frame0_graph_compute_ms",
        "tail_graph_compute_ms",
        "tail_propagate_graph_compute_ms",
        "tail_propagate_cache_compute_ms",
    )
    computed_required_core = sum(
        component_mean(components, key) for key in core_component_keys
    )
    required_core_compute = (
        computed_required_core
        if any(
            component_number(components, key) is not None for key in core_component_keys
        )
        else component_mean(components, "required_core_compute_ms")
    )
    required_graph_admin = component_sum_if_any(
        components,
        (
            "frame0_graph_build_ms",
            "frame0_graph_alloc_ms",
            "frame0_input_upload_ms",
            "tail_graph_build_ms",
            "tail_graph_alloc_ms",
            "tail_input_upload_ms",
            "tail_propagate_graph_build_ms",
            "tail_propagate_graph_alloc_ms",
            "tail_propagate_input_upload_ms",
            "tail_propagate_output_read_ms",
        ),
    )
    frame0_encode_graph_admin = component_sum_if_any(
        components,
        (
            "frame0_graph_build_ms",
            "frame0_graph_alloc_ms",
            "frame0_input_upload_ms",
        ),
    )
    tail_encode_graph_admin = component_sum_if_any(
        components,
        (
            "tail_graph_build_ms",
            "tail_graph_alloc_ms",
            "tail_input_upload_ms",
        ),
    )
    tail_inline_encode_graph_admin = component_sum_if_any(
        components,
        (
            "tail_inline_encode_graph_build_ms",
            "tail_inline_encode_graph_alloc_ms",
            "tail_inline_encode_input_upload_ms",
        ),
    )
    tail_preencode_encode_graph_admin = component_sum_if_any(
        components,
        (
            "tail_preencode_encode_graph_build_ms",
            "tail_preencode_encode_graph_alloc_ms",
            "tail_preencode_encode_input_upload_ms",
        ),
    )
    set_mean_component(components, "frame0_encode_graph_admin_ms", frame0_encode_graph_admin)
    set_mean_component(components, "tail_encode_graph_admin_ms", tail_encode_graph_admin)
    set_mean_component(
        components, "tail_inline_encode_graph_admin_ms", tail_inline_encode_graph_admin
    )
    set_mean_component(
        components,
        "tail_preencode_encode_graph_admin_ms",
        tail_preencode_encode_graph_admin,
    )
    tail_encode_non_compute = component_sum_if_any(
        components,
        (
            "tail_encode_preprocess_ms",
            "tail_encode_graph_admin_ms",
            "tail_encode_state_update_ms",
            "tail_encode_pe_build_ms",
            "tail_encode_remainder_ms",
        ),
    )
    tail_inline_encode_non_compute = component_sum_if_any(
        components,
        (
            "tail_inline_encode_preprocess_ms",
            "tail_inline_encode_graph_admin_ms",
            "tail_inline_encode_state_update_ms",
            "tail_inline_encode_pe_build_ms",
            "tail_inline_encode_remainder_ms",
        ),
    )
    tail_preencode_encode_non_compute = component_sum_if_any(
        components,
        (
            "tail_preencode_encode_preprocess_ms",
            "tail_preencode_encode_graph_admin_ms",
            "tail_preencode_encode_state_update_ms",
            "tail_preencode_encode_pe_build_ms",
            "tail_preencode_encode_remainder_ms",
        ),
    )
    tail_propagate_graph_admin = component_sum_if_any(
        components,
        (
            "tail_propagate_graph_build_ms",
            "tail_propagate_graph_alloc_ms",
            "tail_propagate_input_upload_ms",
            "tail_propagate_output_read_ms",
        ),
    )
    required_encode_graph_admin = component_sum_if_any(
        {"x": {"mean_ms": frame0_encode_graph_admin}, "y": {"mean_ms": tail_encode_graph_admin}},
        ("x", "y"),
    )
    required_encode_preprocess = component_sum_if_any(
        components,
        ("frame0_encode_preprocess_ms", "tail_encode_preprocess_ms"),
    )
    required_encode_state_update = component_sum_if_any(
        components,
        ("frame0_encode_state_update_ms", "tail_encode_state_update_ms"),
    )
    required_encode_pe_build = component_sum_if_any(
        components,
        ("frame0_encode_pe_build_ms", "tail_encode_pe_build_ms"),
    )
    required_encode_remainder = component_sum_if_any(
        components,
        ("frame0_encode_remainder_ms", "tail_encode_remainder_ms"),
    )
    required_encode_non_compute = component_sum_if_any(
        {
            "preprocess": {"mean_ms": required_encode_preprocess},
            "graph_admin": {"mean_ms": required_encode_graph_admin},
            "state_update": {"mean_ms": required_encode_state_update},
            "pe_build": {"mean_ms": required_encode_pe_build},
            "remainder": {"mean_ms": required_encode_remainder},
        },
        ("preprocess", "graph_admin", "state_update", "pe_build", "remainder"),
    )
    components["required_core_compute_ms"] = {"mean_ms": required_core_compute}
    components["required_non_core_compute_ms"] = {
        "mean_ms": required_e2e - required_core_compute if required_e2e else None
    }
    set_mean_component(components, "required_graph_admin_ms", required_graph_admin)
    set_mean_component(components, "tail_encode_non_compute_ms", tail_encode_non_compute)
    set_mean_component(
        components, "tail_inline_encode_non_compute_ms", tail_inline_encode_non_compute
    )
    set_mean_component(
        components,
        "tail_preencode_encode_non_compute_ms",
        tail_preencode_encode_non_compute,
    )
    set_mean_component(components, "tail_propagate_graph_admin_ms", tail_propagate_graph_admin)
    set_mean_component(components, "required_encode_graph_admin_ms", required_encode_graph_admin)
    set_mean_component(components, "required_encode_preprocess_ms", required_encode_preprocess)
    set_mean_component(components, "required_encode_state_update_ms", required_encode_state_update)
    set_mean_component(components, "required_encode_pe_build_ms", required_encode_pe_build)
    set_mean_component(components, "required_encode_remainder_ms", required_encode_remainder)
    set_mean_component(components, "required_encode_non_compute_ms", required_encode_non_compute)
    components["required_core_compute_pct"] = {
        "mean_pct": required_core_compute / required_e2e_value * 100.0
        if required_e2e_value
        else None
    }
    components["required_input_load_pct"] = {
        "mean_pct": component_mean(components, "input_load_ms") / required_e2e_value * 100.0
        if required_e2e_value
        else None
    }
    components["tail_encode_graph_compute_pct"] = {
        "mean_pct": component_number(components, "tail_graph_compute_ms")
        / tail_encode_required_value
        * 100.0
        if component_number(components, "tail_graph_compute_ms") is not None
        and tail_encode_required_value
        else None
    }
    tail_required_model = component_number(components, "tail_required_model_ms")
    tail_required_model_sum = component_number(components, "tail_required_model_sum_ms")
    components["tail_required_model_accounting_delta_ms"] = {
        "mean_ms": tail_required_model - tail_required_model_sum
        if tail_required_model is not None and tail_required_model_sum is not None
        else None
    }
    tail_encode_split_sum = component_sum_if_any(
        components, ("tail_inline_encode_ms", "tail_preencode_encode_ms")
    )
    components["tail_encode_required_accounting_delta_ms"] = {
        "mean_ms": tail_encode_required - tail_encode_split_sum
        if tail_encode_required_value is not None and tail_encode_split_sum is not None
        else None
    }
    tail_encode_graph_compute = component_number(components, "tail_graph_compute_ms")
    tail_encode_graph_compute_split_sum = component_sum_if_any(
        components,
        (
            "tail_inline_encode_graph_compute_ms",
            "tail_preencode_encode_graph_compute_ms",
        ),
    )
    components["tail_encode_graph_compute_accounting_delta_ms"] = {
        "mean_ms": tail_encode_graph_compute - tail_encode_graph_compute_split_sum
        if tail_encode_graph_compute is not None
        and tail_encode_graph_compute_split_sum is not None
        else None
    }


def summarize_cpp_row(
    path: Path, summary: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    components = {key: metric_with_stats(row, key) for key in TOP_LEVEL_COMPONENTS}
    frame_timing = row.get("cpp_frame_timing_aggregate_ms")
    if isinstance(frame_timing, dict):
        for output_key, scope, metric in FRAME_TIMING_COMPONENTS:
            components[output_key] = frame_metric_with_stats(
                frame_timing, output_key, scope, metric
            )
    add_derived_components(components)

    required_component_sum_ms = sum(
        component_mean(components, key)
        for key in (
            "input_load_ms",
            "session_setup_ms",
            "frame0_encode_ms",
            "frame0_prompt_model_ms",
            "required_tail_state_create_ms",
            "tail_encode_required_total_ms",
            "required_tail_propagate_ms",
            "required_tail_unattributed_ms",
            "model_remainder_ms",
        )
    )
    required_e2e = component_mean(components, "required_e2e_ms")
    model_e2e = component_mean(components, "model_e2e_ms")
    model_component_sum_ms = required_component_sum_ms - component_mean(
        components, "input_load_ms"
    )

    benchmark_context = summary.get("benchmark_context", {})
    return {
        "source": str(path),
        "model": row.get("model"),
        "precision": row.get("precision"),
        "backend": row.get("backend"),
        "cpp_variant": row.get("cpp_variant", "default"),
        "cpp_env_overrides": row.get("cpp_env_overrides", {}),
        "frames": benchmark_context.get("frames"),
        "timed_start_frame": benchmark_context.get("timed_start_frame"),
        "frame_format": benchmark_context.get("frame_format"),
        "python_dtype": benchmark_context.get("python_dtype"),
        "tf32_policy": benchmark_context.get("tf32_policy"),
        "python_sam3_version": benchmark_context.get("python_sam3_version"),
        "cpp_text_init_selected_only": benchmark_context.get(
            "cpp_text_init_selected_only"
        ),
        "encode_img_size": row.get("requested_encode_img_size"),
        "effective_encode_img_size": row.get("effective_encode_img_size"),
        "cpp_output_artifacts": row.get("cpp_output_artifacts"),
        "preencoded_track_frames": row.get("preencoded_track_frames"),
        "preencoded_cached_tail_frames": row.get("preencoded_cached_tail_frames"),
        "components": components,
        "accounting": {
            "required_component_sum_ms": required_component_sum_ms,
            "required_delta_ms": required_e2e - required_component_sum_ms,
            "model_component_sum_ms": model_component_sum_ms,
            "model_delta_ms": model_e2e - model_component_sum_ms,
        },
    }


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        summary = load_json(path)
        cpp_rows = summary.get("cpp", [])
        if not isinstance(cpp_rows, list):
            continue
        for row in cpp_rows:
            if isinstance(row, dict):
                rows.append(summarize_cpp_row(path, summary, row))
    return rows


def row_label(row: dict[str, Any]) -> str:
    variant = row.get("cpp_variant") or "default"
    model = row.get("model") or "unknown"
    return f"{model}:{variant}"


def add_deltas(rows: list[dict[str, Any]], baseline_label: str | None) -> None:
    if not rows:
        return
    baseline = rows[0]
    if baseline_label:
        for row in rows:
            if (
                row_label(row) == baseline_label
                or str(row.get("source")) == baseline_label
            ):
                baseline = row
                break
    baseline_components = baseline["components"]
    for row in rows:
        deltas: dict[str, dict[str, float | None]] = {}
        for key, metric in row["components"].items():
            value = numeric(metric.get("mean_ms"))
            base = numeric(baseline_components.get(key, {}).get("mean_ms"))
            if value is None or base is None:
                deltas[key] = {"delta_ms": None, "delta_pct": None}
                continue
            deltas[key] = {
                "delta_ms": value - base,
                "delta_pct": ((value / base) - 1.0) * 100.0 if base else None,
            }
        row["delta_from_baseline"] = deltas
    for row in rows:
        row["baseline"] = row is baseline


def required_process_percent(
    cpp_ms: float | None, required_e2e: float | None
) -> float | None:
    if cpp_ms is None or required_e2e is None or required_e2e <= 0.0:
        return None
    return cpp_ms / required_e2e * 100.0


def required_process_metric(
    components: dict[str, dict[str, Any]], key: str
) -> dict[str, Any]:
    metric = components.get(key, {})
    return {
        "cpp_ms": numeric(metric.get("mean_ms")),
        "sd_ms": numeric(metric.get("sd_ms")),
        "n": int(metric["n"]) if isinstance(metric.get("n"), int | float) else None,
        "source_component": key,
    }


def required_process_value(
    components: dict[str, dict[str, Any]], key: str
) -> float | None:
    return numeric(components.get(key, {}).get("mean_ms"))


def required_process_sum(
    components: dict[str, dict[str, Any]], *keys: str
) -> float | None:
    values = [required_process_value(components, key) for key in keys]
    if not any(value is not None for value in values):
        return None
    return sum(value for value in values if value is not None)


def required_process_diff(total: float | None, *parts: float | None) -> float | None:
    if total is None:
        return None
    return total - sum(part for part in parts if part is not None)


def make_required_process_row(
    row: dict[str, Any],
    *,
    table: str,
    process: str,
    bucket: str,
    role: str,
    cpp_ms: float | None,
    required_e2e_ms: float | None,
    basis: str,
    source_component: str | None = None,
    sd_ms: float | None = None,
    n: int | None = None,
    parent_process: str | None = None,
) -> dict[str, Any]:
    acceptance_target = role in {"denominator", "exclusive_required", "exclusive_detail"}
    additive_required_work = role in {"exclusive_required", "exclusive_detail"}
    measurement_class = {
        "denominator": "formal_required_denominator",
        "exclusive_required": "formal_required_rollup",
        "exclusive_detail": "formal_required_detail",
        "placement_detail": "placement_diagnostic",
    }.get(role, "diagnostic")
    return {
        "source": row.get("source"),
        "model": row.get("model"),
        "precision": row.get("precision"),
        "backend": row.get("backend"),
        "cpp_variant": row.get("cpp_variant", "default"),
        "table": table,
        "bucket": bucket,
        "process": process,
        "parent_process": parent_process,
        "role": role,
        "cpp_ms": cpp_ms,
        "required_e2e_pct": required_process_percent(cpp_ms, required_e2e_ms),
        "sd_ms": sd_ms,
        "n": n,
        "source_component": source_component,
        "measurement_class": measurement_class,
        "included_in_required_e2e": True,
        "acceptance_target": acceptance_target,
        "additive_required_work": additive_required_work,
        "placement_only": role == "placement_detail",
        "basis": basis,
    }


def required_process_rollup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    process_rows: list[dict[str, Any]] = []
    definitions = (
        (
            "input.prepare",
            "input",
            "exclusive_required",
            "input_load_ms",
            "Host frame discovery/decode/storage required before model session timing.",
        ),
        (
            "model.session_setup",
            "setup",
            "exclusive_required",
            "session_setup_ms",
            "C++ model session state/tracker setup after frames are available.",
        ),
        (
            "frame0.image_encode",
            "frame0",
            "exclusive_required",
            "frame0_encode_ms",
            "Frame-0 image encode including detector neck.",
        ),
        (
            "frame0.prompt_and_tracker_init",
            "frame0",
            "exclusive_required",
            "frame0_prompt_model_ms",
            "Frame-0 text prompt, segmentation, and tracker initialization.",
        ),
        (
            "tail.image_encode",
            "tail_encode",
            "exclusive_required",
            "tail_encode_required_total_ms",
            "Required image encode for timed tail frames, including cached-tail preencode.",
        ),
        (
            "tail.preencode_state_create",
            "tail_encode",
            "exclusive_required",
            "required_tail_state_create_ms",
            "State creation for cached-tail preencoded frames; additive required work before tail propagation.",
        ),
        (
            "tail.propagate",
            "tail_propagate",
            "exclusive_required",
            "required_tail_propagate_ms",
            "Required tracker propagation for timed tail frames.",
        ),
        (
            "tail.unattributed",
            "accounting",
            "exclusive_required",
            "required_tail_unattributed_ms",
            "Tail wall time not attributed to encode, propagate, or artifacts.",
        ),
        (
            "model.remainder",
            "accounting",
            "exclusive_required",
            "model_remainder_ms",
            "Model-session wall time not attributed to explicit required components.",
        ),
    )
    for row in rows:
        components = row.get("components", {})
        if not isinstance(components, dict):
            continue
        required_e2e = required_process_value(components, "required_e2e_ms")
        process_rows.append(
            make_required_process_row(
                row,
                table="rollup",
                process="required_e2e.total",
                bucket="total",
                role="denominator",
                cpp_ms=required_e2e,
                required_e2e_ms=required_e2e,
                basis="Formal C++ required E2E denominator: input_load_ms + model_e2e_ms.",
                source_component="required_e2e_ms",
                sd_ms=component_stat(components, "required_e2e_ms", "sd_ms"),
                n=int(components["required_e2e_ms"]["n"])
                if isinstance(
                    components.get("required_e2e_ms", {}).get("n"), int | float
                )
                else None,
            )
        )
        for process, bucket, role, component, basis in definitions:
            metric = required_process_metric(components, component)
            process_rows.append(
                make_required_process_row(
                    row,
                    table="rollup",
                    process=process,
                    bucket=bucket,
                    role=role,
                    cpp_ms=metric["cpp_ms"],
                    required_e2e_ms=required_e2e,
                    basis=basis,
                    source_component=component,
                    sd_ms=metric["sd_ms"],
                    n=metric["n"],
                )
            )
    return process_rows


def required_process_detail_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        components = row.get("components", {})
        if not isinstance(components, dict):
            continue
        required_e2e = required_process_value(components, "required_e2e_ms")

        input_load = required_process_value(components, "input_load_ms")
        input_decode = required_process_value(components, "input_frame_decode_ms")
        input_other = required_process_diff(input_load, input_decode)
        frame0_encode = required_process_value(components, "frame0_encode_ms")
        frame0_encode_graph = required_process_value(
            components, "frame0_encode_graph_compute_ms"
        )
        frame0_encode_other = required_process_diff(frame0_encode, frame0_encode_graph)
        frame0_prompt = required_process_value(components, "frame0_prompt_model_ms")
        frame0_prompt_parts = {
            "segment": required_process_value(components, "frame0_segment_ms"),
            "tracker_mask_prepare": required_process_value(
                components, "frame0_tracker_mask_prepare_ms"
            ),
            "tracker_memory_encode": required_process_value(
                components, "frame0_tracker_memory_encode_ms"
            ),
            "tracker_obj_ptr": required_process_value(
                components, "frame0_tracker_obj_ptr_ms"
            ),
            "tracker_store": required_process_value(
                components, "frame0_tracker_store_ms"
            ),
            "prompt_unattributed": required_process_value(
                components, "frame0_prompt_unattributed_ms"
            ),
        }
        frame0_prompt_other = required_process_diff(
            frame0_prompt, *frame0_prompt_parts.values()
        )
        tail_encode = required_process_value(
            components, "tail_encode_required_total_ms"
        )
        tail_preencode_state_create = required_process_value(
            components, "required_tail_state_create_ms"
        )
        tail_encode_graph = first_number(
            required_process_value(components, "tail_graph_compute_ms"),
            required_process_value(components, "tail_encode_graph_compute_ms"),
        )
        tail_encode_parts = (
            (
                "preprocess",
                required_process_value(components, "tail_encode_preprocess_ms"),
                "tail_encode_preprocess_ms",
            ),
            (
                "graph_admin",
                required_process_value(components, "tail_encode_graph_admin_ms"),
                "tail_encode_graph_admin_ms",
            ),
            (
                "state_update",
                required_process_value(components, "tail_encode_state_update_ms"),
                "tail_encode_state_update_ms",
            ),
            (
                "pe_build",
                required_process_value(components, "tail_encode_pe_build_ms"),
                "tail_encode_pe_build_ms",
            ),
            (
                "remainder",
                required_process_value(components, "tail_encode_remainder_ms"),
                "tail_encode_remainder_ms",
            ),
        )
        tail_encode_other = first_number(
            required_process_value(components, "tail_encode_non_compute_ms"),
            required_process_diff(tail_encode, tail_encode_graph),
        )
        tail_inline_encode_parts = (
            (
                "total",
                required_process_value(components, "tail_inline_encode_ms"),
                "tail_inline_encode_ms",
            ),
            (
                "graph_compute",
                required_process_value(
                    components, "tail_inline_encode_graph_compute_ms"
                ),
                "tail_inline_encode_graph_compute_ms",
            ),
            (
                "non_compute",
                required_process_value(components, "tail_inline_encode_non_compute_ms"),
                "tail_inline_encode_non_compute_ms",
            ),
            (
                "preprocess",
                required_process_value(components, "tail_inline_encode_preprocess_ms"),
                "tail_inline_encode_preprocess_ms",
            ),
            (
                "graph_admin",
                required_process_value(components, "tail_inline_encode_graph_admin_ms"),
                "tail_inline_encode_graph_admin_ms",
            ),
            (
                "state_update",
                required_process_value(
                    components, "tail_inline_encode_state_update_ms"
                ),
                "tail_inline_encode_state_update_ms",
            ),
            (
                "pe_build",
                required_process_value(components, "tail_inline_encode_pe_build_ms"),
                "tail_inline_encode_pe_build_ms",
            ),
            (
                "remainder",
                required_process_value(components, "tail_inline_encode_remainder_ms"),
                "tail_inline_encode_remainder_ms",
            ),
        )
        tail_preencode_encode_parts = (
            (
                "total",
                required_process_value(components, "tail_preencode_encode_ms"),
                "tail_preencode_encode_ms",
            ),
            (
                "graph_compute",
                required_process_value(
                    components, "tail_preencode_encode_graph_compute_ms"
                ),
                "tail_preencode_encode_graph_compute_ms",
            ),
            (
                "non_compute",
                required_process_value(
                    components, "tail_preencode_encode_non_compute_ms"
                ),
                "tail_preencode_encode_non_compute_ms",
            ),
            (
                "preprocess",
                required_process_value(
                    components, "tail_preencode_encode_preprocess_ms"
                ),
                "tail_preencode_encode_preprocess_ms",
            ),
            (
                "graph_admin",
                required_process_value(
                    components, "tail_preencode_encode_graph_admin_ms"
                ),
                "tail_preencode_encode_graph_admin_ms",
            ),
            (
                "state_update",
                required_process_value(
                    components, "tail_preencode_encode_state_update_ms"
                ),
                "tail_preencode_encode_state_update_ms",
            ),
            (
                "pe_build",
                required_process_value(components, "tail_preencode_encode_pe_build_ms"),
                "tail_preencode_encode_pe_build_ms",
            ),
            (
                "remainder",
                required_process_value(
                    components, "tail_preencode_encode_remainder_ms"
                ),
                "tail_preencode_encode_remainder_ms",
            ),
        )
        tail_propagate = required_process_value(
            components, "required_tail_propagate_ms"
        )
        tail_propagate_graph = required_process_value(
            components, "tail_propagate_graph_compute_ms"
        )
        tail_rope_setup = required_process_sum(
            components, "tail_propagate_rope_cache_ms", "tail_propagate_rope_k_build_ms"
        )
        tail_propagate_parts = {
            "graph_compute": tail_propagate_graph,
            "prompt_build": required_process_value(
                components, "tail_propagate_prompt_build_ms"
            ),
            "rope_setup": tail_rope_setup,
            "input_upload": required_process_value(
                components, "tail_propagate_input_upload_ms"
            ),
            "output_read": required_process_value(
                components, "tail_propagate_output_read_ms"
            ),
            "bbox_from_logits": required_process_value(
                components, "tail_propagate_bbox_from_logits_ms"
            ),
            "cache_compute": required_process_value(
                components, "tail_propagate_cache_compute_ms"
            ),
            "memory_update": required_process_value(
                components, "tail_propagate_memory_update_ms"
            ),
            "remainder": required_process_value(
                components, "tail_propagate_remainder_ms"
            ),
        }
        tail_propagate_other = required_process_diff(
            tail_propagate, *tail_propagate_parts.values()
        )
        details = (
            (
                "input.prepare.decode",
                "input.prepare",
                "input",
                "exclusive_detail",
                input_decode,
                "input_frame_decode_ms",
                "Frame decode/PNG load from the prepared frame directory.",
            ),
            (
                "input.prepare.other",
                "input.prepare",
                "input",
                "exclusive_detail",
                input_other,
                None,
                "Directory scan, sort, store, and input accounting remainder.",
            ),
            (
                "frame0.image_encode.graph_compute",
                "frame0.image_encode",
                "frame0",
                "exclusive_detail",
                frame0_encode_graph,
                "frame0_encode_graph_compute_ms",
                "ggml graph compute for frame-0 image encode.",
            ),
            (
                "frame0.image_encode.non_compute",
                "frame0.image_encode",
                "frame0",
                "exclusive_detail",
                frame0_encode_other,
                None,
                "Frame-0 encode preprocessing, graph build/allocation, upload, PE/state, and remainder.",
            ),
            *(
                (
                    f"frame0.prompt_and_tracker_init.{name}",
                    "frame0.prompt_and_tracker_init",
                    "frame0",
                    "exclusive_detail",
                    value,
                    None,
                    f"Frame-0 prompt/tracker init detail: {name}.",
                )
                for name, value in frame0_prompt_parts.items()
            ),
            (
                "frame0.prompt_and_tracker_init.other",
                "frame0.prompt_and_tracker_init",
                "frame0",
                "exclusive_detail",
                frame0_prompt_other,
                None,
                "Frame-0 prompt/tracker init residual.",
            ),
            (
                "tail.image_encode.graph_compute",
                "tail.image_encode",
                "tail_encode",
                "exclusive_detail",
                tail_encode_graph,
                "tail_graph_compute_ms",
                "ggml graph compute for required tail image encode.",
            ),
            (
                "tail.image_encode.non_compute",
                "tail.image_encode",
                "tail_encode",
                "exclusive_detail",
                tail_encode_other,
                None,
                "Tail encode preprocessing, graph build/allocation, upload, PE/state, and remainder.",
            ),
            *(
                (
                    f"tail.image_encode.non_compute.{name}",
                    "tail.image_encode.non_compute",
                    "tail_encode",
                    "exclusive_detail",
                    value,
                    source_component,
                    f"Tail encode non-compute detail: {name}.",
                )
                for name, value, source_component in tail_encode_parts
            ),
            *(
                (
                    f"tail.image_encode.inline.{name}",
                    "tail.image_encode",
                    "tail_encode",
                    "placement_detail",
                    value,
                    source_component,
                    f"Tail inline encode placement detail: {name}.",
                )
                for name, value, source_component in tail_inline_encode_parts
            ),
            *(
                (
                    f"tail.image_encode.preencode.{name}",
                    "tail.image_encode",
                    "tail_encode",
                    "placement_detail",
                    value,
                    source_component,
                    f"Cached-tail preencode placement detail: {name}.",
                )
                for name, value, source_component in tail_preencode_encode_parts
            ),
            (
                "tail.preencode_state_create",
                "tail.preencode_state_create",
                "tail_encode",
                "exclusive_detail",
                tail_preencode_state_create,
                "required_tail_state_create_ms",
                "Cached-tail state allocation/initialization that is added back to required E2E.",
            ),
            *(
                (
                    f"tail.propagate.{name}",
                    "tail.propagate",
                    "tail_propagate",
                    "exclusive_detail",
                    value,
                    None,
                    f"Tail propagation detail: {name}.",
                )
                for name, value in tail_propagate_parts.items()
            ),
            (
                "tail.propagate.other",
                "tail.propagate",
                "tail_propagate",
                "exclusive_detail",
                tail_propagate_other,
                None,
                "Tail propagation residual outside named substeps.",
            ),
        )
        for process, parent, bucket, role, cpp_ms, source_component, basis in details:
            detail_rows.append(
                make_required_process_row(
                    row,
                    table="detail",
                    process=process,
                    parent_process=parent,
                    bucket=bucket,
                    role=role,
                    cpp_ms=cpp_ms,
                    required_e2e_ms=required_e2e,
                    basis=basis,
                    source_component=source_component,
                )
            )
    return detail_rows


def python_tail_hook_contracts(
    process_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    process_to_key = {
        "tail_image_encode_or_backbone": "timed_tail_backbone_detection_ms",
        "tail_propagate_or_tracker": "timed_tail_tracker_propagation_ms",
    }
    for process_row in process_rows:
        if not isinstance(process_row, dict):
            continue
        process = process_row.get("process")
        output_key = process_to_key.get(str(process))
        if output_key is None:
            continue
        pair = str(process_row.get("pair", ""))
        variant = pair.split(" vs python:", 1)[0]
        if not variant:
            continue
        contract = contracts.setdefault(
            variant,
            {
                "timed_tail_backbone_detection_ms": None,
                "timed_tail_tracker_propagation_ms": None,
                "exact_family_match": bool(process_row.get("exact_family_match")),
                "basis": (
                    "official Python timed-tail stage hooks for the same frame range; "
                    "diagnostic hooks, not a separate Python denominator"
                ),
            },
        )
        value = numeric(process_row.get("python_ms"))
        if value is not None:
            contract[output_key] = value
        contract["exact_family_match"] = bool(
            contract.get("exact_family_match")
        ) and bool(process_row.get("exact_family_match"))

    for contract in contracts.values():
        backbone = numeric(contract.get("timed_tail_backbone_detection_ms"))
        tracker = numeric(contract.get("timed_tail_tracker_propagation_ms"))
        contract["timed_tail_hooks_available"] = (
            backbone is not None and tracker is not None
        )
        contract["timed_tail_hook_sum_ms"] = (
            backbone + tracker
            if backbone is not None and tracker is not None
            else None
        )
    return contracts


def required_process_summary(
    rows: list[dict[str, Any]], process_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    rollup_rows = required_process_rollup_rows(rows)
    detail_rows = required_process_detail_rows(rows)
    python_contracts = python_tail_hook_contracts(process_rows or [])
    contracts = [
        required_e2e_contract(
            row, rollup_rows, detail_rows, python_contracts.get(row_label(row))
        )
        for row in rows
    ]
    return {
        "note": (
            "rollup rows are exclusive required-E2E components and should add up to "
            "required_e2e.total within the accounting remainder. detail rows split those "
            "components for optimization; placement_detail rows explain cached-tail placement "
            "and are not additional E2E work. measurement_class marks whether a row is the "
            "formal denominator, additive required work, a required substep, or placement-only "
            "diagnostic accounting."
        ),
        "contracts": contracts,
        "rollup_rows": rollup_rows,
        "detail_rows": detail_rows,
    }


def required_e2e_contract(
    row: dict[str, Any],
    rollup_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    python_tail_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components = row.get("components", {})
    if not isinstance(components, dict):
        components = {}
    label = row_label(row)
    required_e2e = required_process_value(components, "required_e2e_ms")
    model_e2e = required_process_value(components, "model_e2e_ms")
    input_load = required_process_value(components, "input_load_ms")
    accounting_required_delta = (
        row.get("accounting", {}).get("required_delta_ms")
        if isinstance(row.get("accounting"), dict)
        else None
    )
    exclusive_sum = sum(
        float(item["cpp_ms"])
        for item in rollup_rows
        if item.get("table") == "rollup"
        and item.get("role") == "exclusive_required"
        and f"{item.get('model')}:{item.get('cpp_variant', 'default')}" == label
        and isinstance(item.get("cpp_ms"), int | float)
    )
    placement_sum = sum(
        float(item["cpp_ms"])
        for item in detail_rows
        if item.get("role") == "placement_detail"
        and f"{item.get('model')}:{item.get('cpp_variant', 'default')}" == label
        and isinstance(item.get("cpp_ms"), int | float)
        and str(item.get("process", "")).endswith(".total")
    )
    tail_inline = required_process_value(components, "tail_inline_encode_ms")
    tail_preencode = required_process_value(components, "tail_preencode_encode_ms")
    tail_encode_total = required_process_value(components, "tail_encode_required_total_ms")
    tail_encode_graph = required_process_value(components, "tail_encode_graph_compute_ms")
    tail_encode_non_compute = required_process_value(components, "tail_encode_non_compute_ms")
    preencoded_cached_tail = bool(row.get("preencoded_cached_tail_frames"))
    preencoded_track_frames = bool(row.get("preencoded_track_frames"))
    tail_encode_placement_delta = required_process_diff(
        tail_encode_total, tail_inline, tail_preencode
    )
    python_tail_contract = python_tail_contract or {}
    python_timed_tail_hooks_available = bool(
        python_tail_contract.get("timed_tail_hooks_available")
    )
    return {
        "variant": label,
        "denominator": "required_e2e_ms",
        "denominator_measurement_class": "formal_required_denominator",
        "required_e2e_ms": required_e2e,
        "model_e2e_ms": model_e2e,
        "input_load_ms": input_load,
        "required_e2e_formula": "input_load_ms + model_e2e_ms",
        "exclusive_rollup_sum_ms": exclusive_sum,
        "exclusive_rollup_delta_ms": required_process_diff(required_e2e, exclusive_sum),
        "accounting_required_delta_ms": numeric(accounting_required_delta),
        "cached_tail": {
            "enabled": preencoded_cached_tail,
            "preencoded_track_frames": preencoded_track_frames,
            "work_is_added_back_to_required_e2e": True,
            "inline_encode_ms": tail_inline,
            "preencode_encode_ms": tail_preencode,
            "tail_encode_required_total_ms": tail_encode_total,
            "tail_encode_placement_delta_ms": tail_encode_placement_delta,
            "interpretation": (
                "cached-tail changes placement only; preencode work stays in the "
                "required-E2E denominator and must not be treated as a Python-equivalent omission"
            ),
        },
        "tail_encode_core_split": {
            "graph_compute_ms": tail_encode_graph,
            "non_compute_ms": tail_encode_non_compute,
            "graph_compute_pct_of_tail_encode": required_process_percent(
                tail_encode_graph, tail_encode_total
            ),
        },
        "placement_detail_total_ms": placement_sum,
        "python_comparison_contract": {
            "formal_wall_rows": (
                "required_session_e2e",
                "model_e2e",
                "input_prepare_or_start_session",
                "frame0_prompt_or_detection",
            ),
            "diagnostic_only_rows": (
                "tail_image_encode_or_backbone",
                "tail_image_encode.graph_compute",
                "tail_propagate_or_tracker",
                "tail_propagate.graph_compute",
            ),
            "cached_tail_python_equivalent": (
                python_timed_tail_hooks_available and preencoded_cached_tail
            ),
            "timed_tail_hooks_available": python_timed_tail_hooks_available,
            "timed_tail_backbone_detection_ms": python_tail_contract.get(
                "timed_tail_backbone_detection_ms"
            ),
            "timed_tail_tracker_propagation_ms": python_tail_contract.get(
                "timed_tail_tracker_propagation_ms"
            ),
            "timed_tail_hook_sum_ms": python_tail_contract.get(
                "timed_tail_hook_sum_ms"
            ),
            "basis": python_tail_contract.get(
                "basis",
                "official Python timed-tail hooks were not available in this summary",
            ),
        },
        "checks": {
            "exclusive_rollup_near_denominator": abs(
                required_process_diff(required_e2e, exclusive_sum) or 0.0
            )
            <= 1.0,
            "cached_tail_not_hidden": (
                not preencoded_cached_tail
                or abs(tail_encode_placement_delta or 0.0) <= 1.0
            ),
            "tail_encode_is_compute_bound": (
                (required_process_percent(tail_encode_graph, tail_encode_total) or 0.0)
                >= 90.0
            ),
        },
    }


def ratio(numerator: Any, denominator: Any) -> float | None:
    lhs = numeric(numerator)
    rhs = numeric(denominator)
    if lhs is None or rhs is None or rhs == 0.0:
        return None
    return lhs / rhs


def first_number(*values: Any) -> float | None:
    for value in values:
        number = numeric(value)
        if number is not None:
            return number
    return None


def pair_frame_mean(pair: dict[str, Any], scope: str, metric: str) -> float | None:
    return numeric(
        nested_get(pair, ("cpp_frame_timing_aggregate_ms", "mean", scope, metric))
    )


def sum_numbers(*values: Any) -> float | None:
    total = 0.0
    any_value = False
    for value in values:
        number = numeric(value)
        if number is None:
            continue
        total += number
        any_value = True
    return total if any_value else None


def legacy_component_split(pair: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the newer component split from older comparison summaries."""
    return {
        "cpp": {
            "required_e2e_ms": pair.get("cpp_required_e2e_ms"),
            "input_load_ms": pair.get("cpp_input_load_ms"),
            "input_frame_decode_ms": nested_get(
                pair, ("cpp_input_breakdown_ms", "frame_decode_ms")
            ),
            "input_remainder_ms": nested_get(
                pair, ("cpp_input_breakdown_ms", "remainder_ms")
            ),
            "session_setup_ms": nested_get(
                pair, ("cpp_model_e2e_breakdown_ms", "session_setup_ms")
            ),
            "frame0_encode_ms": pair.get("cpp_frame0_encode_ms"),
            "frame0_prompt_model_ms": pair.get("cpp_frame0_prompt_model_ms"),
            "frame0_segment_ms": pair.get("cpp_frame0_segment_ms"),
            "frame0_tracker_mask_prepare_ms": pair.get(
                "cpp_frame0_tracker_mask_prepare_ms"
            ),
            "frame0_tracker_memory_encode_ms": pair.get(
                "cpp_frame0_tracker_memory_encode_ms"
            ),
            "frame0_tracker_obj_ptr_ms": pair.get("cpp_frame0_tracker_obj_ptr_ms"),
            "tail_state_create_ms": pair.get("cpp_required_tail_state_create_ms"),
            "tail_runtime_wall_ms": pair.get("cpp_tail_runtime_wall_ms"),
            "tail_visible_model_ms": pair.get("cpp_tail_visible_model_ms"),
            "tail_preencode_required_model_ms": pair.get(
                "cpp_tail_preencode_required_model_ms"
            ),
            "tail_required_model_ms": pair.get("cpp_tail_required_model_ms"),
            "tail_encode_required_ms": pair.get("cpp_tail_encode_required_total_ms"),
            "tail_inline_encode_ms": pair.get("cpp_tail_inline_encode_ms"),
            "tail_preencode_state_create_ms": pair.get(
                "cpp_tail_preencode_state_create_ms"
            ),
            "tail_preencode_encode_ms": pair.get("cpp_tail_preencode_encode_ms"),
            "tail_preencoded_propagate_ms": pair.get(
                "cpp_tail_preencoded_propagate_ms"
            ),
            "tail_propagate_ms": pair.get("cpp_required_tail_propagate_ms"),
            "tail_propagate_graph_compute_ms": pair.get(
                "cpp_tail_propagate_graph_compute_ms"
            ),
            "tail_propagate_input_upload_ms": nested_get(
                pair,
                (
                    "cpp_required_e2e_breakdown_ms",
                    "tail_ms",
                    "propagate_input_upload_ms",
                ),
            ),
            "core_compute_ms": pair.get("cpp_required_core_compute_ms"),
            "non_core_compute_ms": pair.get("cpp_required_non_core_compute_ms"),
            "required_remainder_ms": pair.get("cpp_required_remainder_ms"),
            "model_remainder_ms": pair.get("cpp_model_remainder_ms"),
        },
        "python": {
            "session_e2e_ms": pair.get("python_session_e2e_ms"),
            "start_session_ms": pair.get("python_start_session_ms"),
            "model_e2e_ms": pair.get("python_model_e2e_ms"),
            "add_prompt_ms": pair.get("python_add_prompt_ms"),
            "propagate_wall_ms": pair.get("python_propagate_wall_ms"),
        },
        "python_profile_diagnostic": {
            "start_session_load_video_frames_ms": nested_get(
                pair,
                (
                    "python_required_e2e_breakdown_ms",
                    "key_stage_mean_ms",
                    "start_session_load_video_frames",
                ),
            ),
            "start_session_construct_initial_input_batch_ms": nested_get(
                pair,
                (
                    "python_required_e2e_breakdown_ms",
                    "key_stage_mean_ms",
                    "start_session_construct_initial_input_batch",
                ),
            ),
            "add_prompt_backbone_detection_ms": nested_get(
                pair,
                (
                    "python_required_e2e_breakdown_ms",
                    "key_stage_mean_ms",
                    "add_prompt_backbone_detection",
                ),
            ),
            "timed_tail_backbone_detection_ms": nested_get(
                pair,
                (
                    "python_required_e2e_breakdown_ms",
                    "key_stage_mean_ms",
                    "timed_tail_backbone_detection",
                ),
            ),
            "timed_tail_tracker_propagation_ms": nested_get(
                pair,
                (
                    "python_required_e2e_breakdown_ms",
                    "key_stage_mean_ms",
                    "timed_tail_tracker_propagation",
                ),
            ),
        },
    }


def cpp_python_process_rows(
    path: Path, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    process_pair_keys = (
        ("comparisons",)
        if isinstance(summary.get("comparisons"), list) and summary["comparisons"]
        else PROCESS_PAIR_KEYS
    )
    for pair_key in process_pair_keys:
        pair_rows = summary.get(pair_key, [])
        if not isinstance(pair_rows, list):
            continue
        for pair in pair_rows:
            if not isinstance(pair, dict):
                continue
            split = pair.get("required_e2e_component_split_ms")
            if not isinstance(split, dict):
                split = legacy_component_split(pair)
            cpp = split.get("cpp", {})
            python = split.get("python", {})
            python_diag = split.get("python_profile_diagnostic", {})
            if not isinstance(cpp, dict) or not isinstance(python, dict):
                continue
            if not isinstance(python_diag, dict):
                python_diag = {}

            pair_label = (
                f"{pair.get('model', 'unknown')}:{pair.get('cpp_variant', 'default')}"
                f" vs python:{pair.get('python_family', 'unknown')}"
            )
            dedupe_key = (
                str(path),
                pair_key,
                pair_label,
                str(pair.get("precision", "")),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            definitions = (
                {
                    "process": "required_session_e2e",
                    "cpp_ms": cpp.get("required_e2e_ms"),
                    "python_ms": python.get("session_e2e_ms"),
                    "bucket": "total",
                    "role": "formal_wall",
                    "cpp_required_e2e": True,
                    "python_session_e2e": True,
                    "basis": "C++ input_load+model_e2e vs Python start_session+model_e2e wall time",
                },
                {
                    "process": "model_e2e",
                    "cpp_ms": pair.get("cpp_model_e2e_ms"),
                    "python_ms": python.get("model_e2e_ms"),
                    "bucket": "total",
                    "role": "formal_wall",
                    "cpp_required_e2e": True,
                    "python_session_e2e": True,
                    "basis": "frame0 prompt/init plus full propagation wall time; input/start_session excluded",
                },
                {
                    "process": "input_prepare_or_start_session",
                    "cpp_ms": cpp.get("input_load_ms"),
                    "python_ms": python.get("start_session_ms"),
                    "python_diagnostic_ms": first_number(
                        python_diag.get("start_session_load_video_frames_ms"),
                        python_diag.get(
                            "start_session_construct_initial_input_batch_ms"
                        ),
                    ),
                    "bucket": "input",
                    "role": "formal_wall_with_diagnostic_hooks",
                    "cpp_required_e2e": True,
                    "python_session_e2e": True,
                    "basis": "C++ host frame load/decode; Python start_session includes load/resize/normalize/staging",
                },
                {
                    "process": "frame0_encode",
                    "cpp_ms": cpp.get("frame0_encode_ms"),
                    "python_ms": None,
                    "bucket": "frame0_encode",
                    "role": "cpp_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ frame-0 image encode graph; Python has no exact public hook",
                },
                {
                    "process": "frame0_prompt_or_detection",
                    "cpp_ms": cpp.get("frame0_prompt_model_ms"),
                    "python_ms": python.get("add_prompt_ms"),
                    "python_diagnostic_ms": python_diag.get(
                        "add_prompt_backbone_detection_ms"
                    ),
                    "bucket": "frame0_prompt",
                    "role": "formal_wall_with_diagnostic_hook",
                    "cpp_required_e2e": True,
                    "python_session_e2e": True,
                    "basis": "C++ text prompt/detector/tracker init; Python add_prompt wall, plus backbone_detection hook",
                },
                {
                    "process": "frame0_prompt.segment",
                    "cpp_ms": pair.get("cpp_frame0_segment_ms"),
                    "python_ms": None,
                    "python_diagnostic_ms": python_diag.get(
                        "add_prompt_backbone_detection_ms"
                    ),
                    "bucket": "frame0_prompt",
                    "role": "diagnostic_subcomponent",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ frame-0 segment substep; Python add_prompt backbone_detection is diagnostic only, not an exact substep match",
                },
                {
                    "process": "frame0_prompt.tracker_mask_prepare",
                    "cpp_ms": pair.get("cpp_frame0_tracker_mask_prepare_ms"),
                    "python_ms": None,
                    "bucket": "frame0_prompt",
                    "role": "diagnostic_subcomponent",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tracker mask preparation for frame-0 prompt",
                },
                {
                    "process": "frame0_prompt.tracker_memory_encode",
                    "cpp_ms": pair.get("cpp_frame0_tracker_memory_encode_ms"),
                    "python_ms": None,
                    "bucket": "frame0_prompt",
                    "role": "diagnostic_subcomponent",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tracker memory encode for frame-0 prompt",
                },
                {
                    "process": "frame0_prompt.tracker_obj_ptr",
                    "cpp_ms": pair.get("cpp_frame0_tracker_obj_ptr_ms"),
                    "python_ms": None,
                    "bucket": "frame0_prompt",
                    "role": "diagnostic_subcomponent",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ object-pointer update for frame-0 tracker init",
                },
                {
                    "process": "tail_required_model",
                    "cpp_ms": first_number(
                        cpp.get("tail_required_model_ms"), cpp.get("tail_model_ms")
                    ),
                    "python_ms": python_diag.get("timed_tail_total_ms"),
                    "bucket": "tail",
                    "role": "cpp_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": True,
                    "basis": "C++ required tail model work with cached/preencoded work added back; Python timed-tail total if profiled",
                },
                {
                    "process": "tail_image_encode_or_backbone",
                    "cpp_ms": cpp.get("tail_encode_required_ms"),
                    "python_ms": python_diag.get("timed_tail_backbone_detection_ms"),
                    "bucket": "tail_encode",
                    "role": "cpp_required_component_with_python_diagnostic_hook",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ required tail encode total, including cached-tail preencode; Python timed-tail backbone_detection hook",
                },
                {
                    "process": "tail_image_encode.graph_compute",
                    "cpp_ms": cpp.get("tail_encode_graph_compute_ms"),
                    "python_ms": python_diag.get("timed_tail_backbone_detection_ms"),
                    "bucket": "tail_encode",
                    "role": "core_graph_compute",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail image encode graph-compute core vs Python timed-tail backbone_detection hook",
                },
                {
                    "process": "tail_image_encode.non_compute_required",
                    "cpp_ms": first_number(
                        cpp.get("required_encode_non_compute_ms"),
                        sum_numbers(
                            cpp.get("required_encode_preprocess_ms"),
                            cpp.get("required_encode_graph_admin_ms"),
                            cpp.get("required_encode_state_update_ms"),
                            cpp.get("required_encode_pe_build_ms"),
                            cpp.get("required_encode_remainder_ms"),
                        ),
                        sum_numbers(
                            pair_frame_mean(pair, "frame0_ms", "encode_preprocess_ms"),
                            pair_frame_mean(
                                pair, "frame0_ms", "encode_input_upload_ms"
                            ),
                            pair_frame_mean(pair, "frame0_ms", "encode_graph_build_ms"),
                            pair_frame_mean(pair, "frame0_ms", "encode_graph_alloc_ms"),
                            pair_frame_mean(
                                pair, "frame0_ms", "encode_state_update_ms"
                            ),
                            pair_frame_mean(pair, "frame0_ms", "encode_pe_build_ms"),
                            pair_frame_mean(pair, "frame0_ms", "encode_remainder_ms"),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_preprocess_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_input_upload_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_graph_build_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_graph_alloc_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_state_update_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_pe_build_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_remainder_ms"
                            ),
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ required frame0+tail image encode work outside graph compute: preprocess, graph admin, state/PE update, and remainder",
                },
                {
                    "process": "tail_image_encode.graph_admin",
                    "cpp_ms": first_number(
                        cpp.get("tail_encode_graph_admin_ms"),
                        sum_numbers(
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_input_upload_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_graph_build_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_graph_alloc_ms"
                            ),
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail image encode input upload plus graph build/allocation outside graph compute",
                },
                {
                    "process": "tail_image_encode.preprocess",
                    "cpp_ms": first_number(
                        cpp.get("tail_encode_preprocess_ms"),
                        pair_frame_mean(
                            pair, "tail_all_required_ms", "encode_preprocess_ms"
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ CPU/GPU image preprocessing before tail image encode graph execution",
                },
                {
                    "process": "tail_image_encode.state_update",
                    "cpp_ms": first_number(
                        cpp.get("tail_encode_state_update_ms"),
                        pair_frame_mean(
                            pair, "tail_all_required_ms", "encode_state_update_ms"
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail image encode state/context ownership update after graph compute",
                },
                {
                    "process": "tail_image_encode.pe_build",
                    "cpp_ms": first_number(
                        cpp.get("tail_encode_pe_build_ms"),
                        pair_frame_mean(
                            pair, "tail_all_required_ms", "encode_pe_build_ms"
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail image encode neck positional-encoding lookup/build work",
                },
                {
                    "process": "tail_image_encode.remainder",
                    "cpp_ms": first_number(
                        cpp.get("tail_encode_remainder_ms"),
                        pair_frame_mean(
                            pair, "tail_all_required_ms", "encode_remainder_ms"
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "accounting_check",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail image encode wall time not accounted by explicit encode substeps",
                },
                {
                    "process": "tail_image_encode.inline",
                    "cpp_ms": cpp.get("tail_inline_encode_ms"),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "runtime_visible_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ required tail image encode executed inside the visible tail loop",
                },
                {
                    "process": "tail_image_encode.inline.graph_compute",
                    "cpp_ms": cpp.get("tail_inline_encode_graph_compute_ms"),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "runtime_visible_core_graph_compute",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "Graph-compute portion of C++ visible inline tail image encode",
                },
                {
                    "process": "tail_image_encode.inline.non_compute",
                    "cpp_ms": cpp.get("tail_inline_encode_non_compute_ms"),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "runtime_visible_non_core_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "Non-compute portion of C++ visible inline tail image encode",
                },
                {
                    "process": "tail_image_encode.preencode",
                    "cpp_ms": cpp.get("tail_preencode_encode_ms"),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "cached_tail_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ required tail image encode moved before the visible tail loop by cached-tail mode",
                },
                {
                    "process": "tail_image_encode.preencode.graph_compute",
                    "cpp_ms": cpp.get("tail_preencode_encode_graph_compute_ms"),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "cached_tail_core_graph_compute",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "Graph-compute portion of C++ cached-tail preencode image encode",
                },
                {
                    "process": "tail_image_encode.preencode.non_compute",
                    "cpp_ms": cpp.get("tail_preencode_encode_non_compute_ms"),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "cached_tail_non_core_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "Non-compute portion of C++ cached-tail preencode image encode",
                },
                {
                    "process": "tail_image_encode.admin_upload_build_alloc",
                    "cpp_ms": first_number(
                        cpp.get("tail_encode_graph_admin_ms"),
                        sum_numbers(
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_input_upload_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_graph_build_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "encode_graph_alloc_ms"
                            ),
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_encode",
                    "role": "deprecated_alias",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "Deprecated alias: C++ tail image encode graph admin/upload work outside graph compute",
                },
                {
                    "process": "tail_propagate_or_tracker",
                    "cpp_ms": cpp.get("tail_propagate_ms"),
                    "python_ms": python_diag.get("timed_tail_tracker_propagation_ms"),
                    "bucket": "tail_propagate",
                    "role": "cpp_required_component_with_python_diagnostic_hook",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail propagation wall split; Python timed-tail tracker_propagation hook",
                },
                {
                    "process": "tail_propagate.graph_compute",
                    "cpp_ms": cpp.get("tail_propagate_graph_compute_ms"),
                    "python_ms": python_diag.get("timed_tail_tracker_propagation_ms"),
                    "bucket": "tail_propagate",
                    "role": "core_graph_compute",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail propagation graph-compute core vs Python timed-tail tracker_propagation hook",
                },
                {
                    "process": "tail_propagate.input_upload",
                    "cpp_ms": pair_frame_mean(
                        pair, "tail_all_required_ms", "propagate_input_upload_ms"
                    ),
                    "python_ms": None,
                    "bucket": "tail_propagate",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ host-to-device updates required before tail propagation graph execution",
                },
                {
                    "process": "tail_propagate.rope_setup",
                    "cpp_ms": sum_numbers(
                        pair_frame_mean(
                            pair, "tail_all_required_ms", "propagate_rope_k_build_ms"
                        ),
                        pair_frame_mean(
                            pair, "tail_all_required_ms", "propagate_rope_cache_ms"
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_propagate",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail propagation RoPE key/cache preparation outside graph compute",
                },
                {
                    "process": "tail_propagate.graph_admin",
                    "cpp_ms": first_number(
                        cpp.get("tail_propagate_graph_admin_ms"),
                        sum_numbers(
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "propagate_graph_build_ms"
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "propagate_graph_alloc_ms"
                            ),
                            pair_frame_mean(
                                pair,
                                "tail_all_required_ms",
                                "propagate_input_upload_ms",
                            ),
                            pair_frame_mean(
                                pair, "tail_all_required_ms", "propagate_output_read_ms"
                            ),
                        ),
                    ),
                    "python_ms": None,
                    "bucket": "tail_propagate",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ tail propagation graph build/allocation plus required input/output transfers outside graph compute",
                },
                {
                    "process": "tail_propagate.output_read",
                    "cpp_ms": pair_frame_mean(
                        pair, "tail_all_required_ms", "propagate_output_read_ms"
                    ),
                    "python_ms": None,
                    "bucket": "tail_propagate",
                    "role": "non_core_required_component",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ device-to-host output read needed for bbox/result materialization",
                },
                {
                    "process": "cpp_required_core_graph_compute",
                    "cpp_ms": cpp.get("core_compute_ms"),
                    "python_ms": None,
                    "bucket": "core_compute",
                    "role": "optimization_denominator_subtotal",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ graph-compute only: frame0 encode, tail encode, tail propagate, cache compute",
                },
                {
                    "process": "cpp_required_non_core",
                    "cpp_ms": cpp.get("non_core_compute_ms"),
                    "python_ms": None,
                    "bucket": "non_core",
                    "role": "optimization_denominator_subtotal",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ required E2E minus graph-compute core",
                },
                {
                    "process": "cpp_required_remainder",
                    "cpp_ms": cpp.get("required_remainder_ms"),
                    "python_ms": None,
                    "bucket": "accounting",
                    "role": "accounting_check",
                    "cpp_required_e2e": True,
                    "python_session_e2e": None,
                    "basis": "C++ required accounting delta; should stay near zero",
                },
            )
            for definition in definitions:
                cpp_ms = numeric(definition.get("cpp_ms"))
                python_ms = numeric(definition.get("python_ms"))
                rows.append(
                    {
                        "source": str(path),
                        "pair_set": pair_key,
                        "pair": pair_label,
                        "process": definition["process"],
                        "bucket": definition["bucket"],
                        "role": definition["role"],
                        "cpp_ms": cpp_ms,
                        "python_ms": python_ms,
                        "python_diagnostic_ms": numeric(
                            definition.get("python_diagnostic_ms")
                        ),
                        "python_over_cpp_ratio": ratio(python_ms, cpp_ms),
                        "cpp_in_required_e2e": definition["cpp_required_e2e"],
                        "python_in_session_e2e": definition["python_session_e2e"],
                        "basis": definition["basis"],
                        "exact_family_match": pair.get("exact_family_match"),
                        "comparable_session_e2e_claim": pair.get(
                            "comparable_session_e2e_claim"
                        ),
                        "same_contract_session_e2e_diagnostic": pair.get(
                            "same_contract_session_e2e_diagnostic"
                        ),
                    }
                )
    return rows


def load_process_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(cpp_python_process_rows(path, load_json(path)))
    return rows


def format_metric(row: dict[str, Any], key: str, unit: str) -> str:
    metric = row["components"].get(key, {})
    if unit == "pct":
        mean = numeric(metric.get("mean_pct"))
        return "-" if mean is None else f"{mean:.1f}%"
    if unit == "count":
        mean = numeric(metric.get("mean_ms"))
        if mean is None:
            return "-"
        delta = numeric(row.get("delta_from_baseline", {}).get(key, {}).get("delta_ms"))
        text = f"{mean:.1f}"
        if delta is not None and not row.get("baseline"):
            text += f" ({delta:+.1f})"
        return text

    mean = numeric(metric.get("mean_ms"))
    if mean is None:
        return "-"
    sd = numeric(metric.get("sd_ms"))
    delta = numeric(row.get("delta_from_baseline", {}).get(key, {}).get("delta_ms"))
    text = f"{mean:.3f}"
    if sd is not None:
        text += f" +/- {sd:.3f}"
    if delta is not None and not row.get("baseline"):
        text += f" ({delta:+.3f})"
    return text


def format_optional_ms(value: Any) -> str:
    number = numeric(value)
    return "-" if number is None else f"{number:.3f}"


def format_optional_ratio(value: Any) -> str:
    number = numeric(value)
    return "-" if number is None else f"{number:.3f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = ["variant", *(key for key, _unit in TABLE_COLUMNS), "acct_delta"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        acct_delta = numeric(row.get("accounting", {}).get("required_delta_ms"))
        values = [
            row_label(row),
            *(format_metric(row, key, unit) for key, unit in TABLE_COLUMNS),
            "-" if acct_delta is None else f"{acct_delta:+.3f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def process_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = [
        "pair_set",
        "pair",
        "bucket",
        "process",
        "role",
        "cpp_ms",
        "cpp_req",
        "python_ms",
        "py_sess",
        "py/cpp",
        "python_diag_ms",
        "basis",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("pair_set", "-")),
            str(row.get("pair", "-")),
            str(row.get("bucket", "-")),
            str(row.get("process", "-")),
            str(row.get("role", "-")),
            format_optional_ms(row.get("cpp_ms")),
            str(row.get("cpp_in_required_e2e")),
            format_optional_ms(row.get("python_ms")),
            str(row.get("python_in_session_e2e")),
            format_optional_ratio(row.get("python_over_cpp_ratio")),
            format_optional_ms(row.get("python_diagnostic_ms")),
            str(row.get("basis", "-")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def required_process_markdown_table(summary: dict[str, Any]) -> str:
    contracts = summary.get("contracts", [])
    rollup = summary.get("rollup_rows", [])
    details = summary.get("detail_rows", [])
    lines = [
        "# Required E2E Process Split",
        "",
        summary.get("note", ""),
        "",
        "## Measurement Contract",
        "",
        "| variant | denominator | required ms | exclusive delta | cached tail | cached work hidden | tail encode graph % | Python tail rows |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        cached_tail = contract.get("cached_tail")
        checks = contract.get("checks")
        core_split = contract.get("tail_encode_core_split")
        python_contract = contract.get("python_comparison_contract")
        hidden = None
        graph_pct = None
        if isinstance(checks, dict):
            hidden = not bool(checks.get("cached_tail_not_hidden"))
        if isinstance(core_split, dict):
            graph_pct = numeric(core_split.get("graph_compute_pct_of_tail_encode"))
        tail_rows = "unknown"
        if isinstance(python_contract, dict):
            if python_contract.get("cached_tail_python_equivalent"):
                tail_rows = "timed_tail_hooks"
            elif python_contract.get("timed_tail_hooks_available"):
                tail_rows = "timed_tail_hooks_diagnostic"
            else:
                tail_rows = "diagnostic_only"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(contract.get("variant", "-")),
                    str(contract.get("denominator", "-")),
                    format_optional_ms(contract.get("required_e2e_ms")),
                    format_optional_ms(contract.get("exclusive_rollup_delta_ms")),
                    str(cached_tail.get("enabled"))
                    if isinstance(cached_tail, dict)
                    else "-",
                    str(hidden) if hidden is not None else "-",
                    "-" if graph_pct is None else f"{graph_pct:.1f}%",
                    tail_rows,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Exclusive Rollup",
            "",
        ]
    )
    rollup_header = [
        "variant",
        "bucket",
        "process",
        "class",
        "cpp_ms",
        "e2e_pct",
        "sd",
        "n",
        "basis",
    ]
    lines.extend(
        [
            "| " + " | ".join(rollup_header) + " |",
            "| " + " | ".join("---" for _ in rollup_header) + " |",
        ]
    )
    for row in rollup:
        if not isinstance(row, dict):
            continue
        values = [
            f"{row.get('model', 'unknown')}:{row.get('cpp_variant', 'default')}",
            str(row.get("bucket", "-")),
            str(row.get("process", "-")),
            str(row.get("measurement_class", "-")),
            format_optional_ms(row.get("cpp_ms")),
            "-"
            if numeric(row.get("required_e2e_pct")) is None
            else f"{numeric(row.get('required_e2e_pct')):.1f}%",
            format_optional_ms(row.get("sd_ms")),
            "-" if row.get("n") is None else str(row.get("n")),
            str(row.get("basis", "-")),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Optimization Detail", ""])
    detail_header = [
        "variant",
        "parent",
        "process",
        "role",
        "class",
        "additive",
        "cpp_ms",
        "e2e_pct",
        "basis",
    ]
    lines.extend(
        [
            "| " + " | ".join(detail_header) + " |",
            "| " + " | ".join("---" for _ in detail_header) + " |",
        ]
    )
    for row in details:
        if not isinstance(row, dict):
            continue
        values = [
            f"{row.get('model', 'unknown')}:{row.get('cpp_variant', 'default')}",
            str(row.get("parent_process", "-")),
            str(row.get("process", "-")),
            str(row.get("role", "-")),
            str(row.get("measurement_class", "-")),
            "yes" if row.get("additive_required_work") else "no",
            format_optional_ms(row.get("cpp_ms")),
            "-"
            if numeric(row.get("required_e2e_pct")) is None
            else f"{numeric(row.get('required_e2e_pct')):.1f}%",
            str(row.get("basis", "-")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument(
        "--baseline", help="Baseline source path or model:variant label"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a compact markdown table instead of JSON.",
    )
    parser.add_argument(
        "--process-markdown",
        action="store_true",
        help=(
            "Print a C++/Python required-E2E process table from comparison rows. "
            "This keeps Python hook-based diagnostics separate from formal wall-time columns."
        ),
    )
    parser.add_argument(
        "--required-processes",
        action="store_true",
        help="Print a required-E2E-only process split JSON.",
    )
    parser.add_argument(
        "--required-process-markdown",
        action="store_true",
        help="Print a compact required-E2E-only process split markdown report.",
    )
    args = parser.parse_args()

    rows = load_rows(args.summaries)
    add_deltas(rows, args.baseline)
    process_rows = load_process_rows(args.summaries)
    required_processes = required_process_summary(rows, process_rows)
    result = {
        "sources": [str(path) for path in args.summaries],
        "note": (
            "required_e2e_ms is the optimization denominator for C++ E2E. "
            "The accounting delta should stay near zero; large deltas mean the benchmark "
            "is missing a required component split. required_core_compute_ms is derived from "
            "frame timing JSONL graph-compute fields and separates model kernel compute from "
            "input, graph setup, uploads, artifacts, and cached-tail bookkeeping. Tail rows "
            "separate runtime wall time from required model time so preencoded/cached work "
            "cannot be hidden outside the E2E denominator."
        ),
        "rows": rows,
        "process_rows": process_rows,
        "required_processes": required_processes,
    }

    if args.required_process_markdown:
        text = required_process_markdown_table(required_processes)
    elif args.required_processes:
        text = (
            json.dumps(
                {
                    "sources": [str(path) for path in args.summaries],
                    "required_processes": required_processes,
                },
                indent=2,
            )
            + "\n"
        )
    elif args.process_markdown:
        text = process_markdown_table(process_rows) + "\n"
    elif args.markdown:
        text = markdown_table(rows) + "\n"
    else:
        text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
