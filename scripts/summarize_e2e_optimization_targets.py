# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DELTA_COMPONENTS = (
    "required_e2e_ms",
    "model_e2e_ms",
    "required_core_compute_ms",
    "tail_encode_required_total_ms",
    "tail_encode_graph_compute_ms",
    "tail_encode_non_compute_ms",
    "tail_encode_graph_admin_ms",
    "tail_encode_preprocess_ms",
    "tail_encode_state_update_ms",
    "tail_encode_pe_build_ms",
    "tail_encode_remainder_ms",
    "tail_inline_encode_non_compute_ms",
    "tail_preencode_encode_non_compute_ms",
    "required_tail_propagate_ms",
    "tail_propagate_graph_compute_ms",
    "required_non_core_compute_ms",
    "required_graph_admin_ms",
)

ACCEPTANCE_SIGNAL_THRESHOLD = -2.0

STAGE_BUDGET_GROUPS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("image.vit.qkv_matmul",),
    "qkv_layout": ("image.vit.layout_rope_copy",),
    "qkv_rope": ("image.vit.layout_rope_copy",),
    "attn_core": ("image.vit.attention",),
    "attn_proj": ("image.vit.proj_matmul",),
    "window_part": ("image.vit.layout_rope_copy",),
    "window_unpart": ("image.vit.layout_rope_copy",),
    "mlp_fc1": ("image.vit.mlp_matmul",),
    "mlp_gelu": ("image.vit.mlp_gelu",),
    "mlp_fc2": ("image.vit.mlp_matmul",),
    "mlp": ("image.vit.mlp_matmul", "image.vit.mlp_gelu"),
    "mlp_fc1_gelu": ("image.vit.mlp_matmul", "image.vit.mlp_gelu"),
    "norm1": ("image.vit.norm_residual_copy",),
    "norm2": ("image.vit.norm_residual_copy",),
}

STAGE_BUDGET_STAGES: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("sam3-vit:qkv-matmul",),
    "qkv_layout": ("sam3-vit:qk-layout-rope",),
    "qkv_rope": ("sam3-vit:qk-layout-rope",),
    "attn_core": ("sam3-vit:window-attn", "sam3-vit:global-attn"),
    "attn_proj": ("sam3-vit:proj-matmul",),
    "window_part": ("sam3-vit:window-part",),
    "window_unpart": ("sam3-vit:window-unpart",),
    "mlp_fc1": ("sam3-vit:mlp-fc1-matmul",),
    "mlp_gelu": ("sam3-vit:mlp-gelu",),
    "mlp_fc2": ("sam3-vit:mlp-fc2-matmul",),
    "mlp": (
        "sam3-vit:mlp-fc1-matmul",
        "sam3-vit:mlp-gelu",
        "sam3-vit:mlp-fc2-matmul",
    ),
    "mlp_fc1_gelu": ("sam3-vit:mlp-fc1-matmul", "sam3-vit:mlp-gelu"),
    "norm1": ("sam3-vit:norm",),
    "norm2": ("sam3-vit:norm",),
}


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def metric_mean(metric: dict[str, Any] | None) -> float | None:
    if not isinstance(metric, dict):
        return None
    return number(metric.get("mean_ms"))


def metric_sd(metric: dict[str, Any] | None) -> float | None:
    if not isinstance(metric, dict):
        return None
    return number(metric.get("sd_ms"))


def metric_n(metric: dict[str, Any] | None) -> int | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("n")
    return int(value) if isinstance(value, int | float) else None


def metric(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    components = row.get("components")
    if not isinstance(components, dict):
        return None
    value = components.get(key)
    return value if isinstance(value, dict) else None


def pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator * 100.0


def speed_ratio(python_ms: float | None, cpp_ms: float | None) -> float | None:
    if python_ms is None or cpp_ms is None or cpp_ms <= 0.0:
        return None
    return python_ms / cpp_ms


def first_row(component_split: dict[str, Any]) -> dict[str, Any]:
    rows = component_split.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("component split JSON has no rows")
    row = rows[0]
    if not isinstance(row, dict):
        raise SystemExit("component split row is not an object")
    return row


def first_contract(component_split: dict[str, Any]) -> dict[str, Any]:
    required_processes = component_split.get("required_processes")
    if not isinstance(required_processes, dict):
        return {}
    contracts = required_processes.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        return {}
    contract = contracts[0]
    return contract if isinstance(contract, dict) else {}


def contract_signature(component_split: dict[str, Any]) -> dict[str, Any]:
    row = first_row(component_split)
    contract = first_contract(component_split)
    cached_tail = contract.get("cached_tail") if isinstance(contract, dict) else None
    return {
        "model": row.get("model"),
        "precision": row.get("precision"),
        "backend": row.get("backend"),
        "python_dtype": row.get("python_dtype"),
        "tf32_policy": row.get("tf32_policy"),
        "python_sam3_version": row.get("python_sam3_version"),
        "frames": row.get("frames"),
        "timed_start_frame": row.get("timed_start_frame"),
        "frame_format": row.get("frame_format"),
        "encode_img_size": row.get("encode_img_size"),
        "effective_encode_img_size": row.get("effective_encode_img_size"),
        "cpp_text_init_selected_only": row.get("cpp_text_init_selected_only"),
        "cpp_output_artifacts": row.get("cpp_output_artifacts"),
        "preencoded_track_frames": row.get("preencoded_track_frames"),
        "preencoded_cached_tail_frames": row.get("preencoded_cached_tail_frames"),
        "cached_tail_work_added_back": cached_tail.get("work_is_added_back_to_required_e2e")
        if isinstance(cached_tail, dict)
        else None,
    }


def contract_compatibility_rows(
    baseline_split: dict[str, Any],
    candidates: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    baseline_signature = contract_signature(baseline_split)
    rows: list[dict[str, Any]] = []
    for path, candidate_split in candidates:
        candidate_signature = contract_signature(candidate_split)
        mismatches = {
            key: {
                "baseline": baseline_signature.get(key),
                "candidate": candidate_signature.get(key),
            }
            for key in baseline_signature
            if baseline_signature.get(key) != candidate_signature.get(key)
        }
        rows.append(
            {
                "candidate": row_label(first_row(candidate_split), path),
                "candidate_component_split": str(path),
                "compatible": not mismatches,
                "mismatches": mismatches,
                "baseline_signature": baseline_signature,
                "candidate_signature": candidate_signature,
            }
        )
    return rows


def row_label(row: dict[str, Any], path: Path | None = None) -> str:
    variant = row.get("cpp_variant")
    model = row.get("model")
    if isinstance(variant, str) and variant:
        return variant
    if isinstance(model, str) and model:
        return model
    return path.stem if path is not None else "candidate"


def parse_named_path(value: str, *, option_name: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise SystemExit(f"{option_name} must use NAME=PATH")
    return name, Path(raw_path)


def load_named_json(
    values: list[str], *, option_name: str
) -> dict[str, dict[str, Any]]:
    named_paths = {
        name: path
        for name, path in (
            parse_named_path(value, option_name=option_name) for value in values
        )
    }
    return {name: load_json(path) for name, path in named_paths.items()}


def delta_signal(
    baseline_sd: float | None,
    baseline_n: int | None,
    candidate_sd: float | None,
    candidate_n: int | None,
    delta_ms: float,
) -> float | None:
    if (
        baseline_sd is None
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


def component_delta_rows(
    baseline: dict[str, Any], candidates: list[tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, candidate in candidates:
        for key in DELTA_COMPONENTS:
            baseline_metric = metric(baseline, key)
            candidate_metric = metric(candidate, key)
            baseline_ms = metric_mean(baseline_metric)
            candidate_ms = metric_mean(candidate_metric)
            if baseline_ms is None or candidate_ms is None:
                continue
            delta_ms = candidate_ms - baseline_ms
            rows.append(
                {
                    "candidate": row_label(candidate, path),
                    "candidate_component_split": str(path),
                    "component": key,
                    "baseline_ms": baseline_ms,
                    "candidate_ms": candidate_ms,
                    "delta_ms": delta_ms,
                    "delta_pct": pct(delta_ms, baseline_ms),
                    "baseline_sd_ms": metric_sd(baseline_metric),
                    "candidate_sd_ms": metric_sd(candidate_metric),
                    "baseline_n": metric_n(baseline_metric),
                    "candidate_n": metric_n(candidate_metric),
                    "delta_standard_error_score": delta_signal(
                        metric_sd(baseline_metric),
                        metric_n(baseline_metric),
                        metric_sd(candidate_metric),
                        metric_n(candidate_metric),
                        delta_ms,
                    ),
                }
            )
    return rows


def process_delta_rows(
    baseline_split: dict[str, Any],
    candidates: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    baseline_by_process = {
        str(row.get("process")): row
        for row in baseline_split.get("process_rows", [])
        if isinstance(row, dict) and row.get("process")
    }
    rows: list[dict[str, Any]] = []
    for path, candidate_split in candidates:
        candidate_label = row_label(first_row(candidate_split), path)
        for row in candidate_split.get("process_rows", []):
            if not isinstance(row, dict):
                continue
            process = str(row.get("process") or "")
            baseline = baseline_by_process.get(process)
            if not process or baseline is None:
                continue
            baseline_cpp_ms = number(baseline.get("cpp_ms"))
            candidate_cpp_ms = number(row.get("cpp_ms"))
            if baseline_cpp_ms is None or candidate_cpp_ms is None:
                continue
            delta_ms = candidate_cpp_ms - baseline_cpp_ms
            rows.append(
                {
                    "candidate": candidate_label,
                    "candidate_component_split": str(path),
                    "bucket": row.get("bucket"),
                    "process": process,
                    "role": row.get("role"),
                    "baseline_cpp_ms": baseline_cpp_ms,
                    "candidate_cpp_ms": candidate_cpp_ms,
                    "cpp_delta_ms": delta_ms,
                    "cpp_delta_pct": pct(delta_ms, baseline_cpp_ms),
                    "baseline_python_ms": number(baseline.get("python_ms")),
                    "candidate_python_ms": number(row.get("python_ms")),
                    "basis": row.get("basis"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row.get("bucket") != "total",
            -abs(float(row.get("cpp_delta_ms") or 0.0)),
            str(row.get("process") or ""),
        ),
    )


def required_process_rows(component_split: dict[str, Any]) -> list[dict[str, Any]]:
    required_processes = component_split.get("required_processes")
    if not isinstance(required_processes, dict):
        return []
    rows: list[dict[str, Any]] = []
    for table_key in ("rollup_rows", "detail_rows"):
        process_rows = required_processes.get(table_key)
        if not isinstance(process_rows, list):
            continue
        for row in process_rows:
            if not isinstance(row, dict):
                continue
            cpp_ms = number(row.get("cpp_ms"))
            if cpp_ms is None:
                continue
            role = str(row.get("role") or "")
            rows.append(
                {
                    "table": row.get("table"),
                    "bucket": row.get("bucket"),
                    "process": row.get("process"),
                    "parent_process": row.get("parent_process"),
                    "role": row.get("role"),
                    "measurement_class": row.get("measurement_class"),
                    "cpp_ms": cpp_ms,
                    "required_e2e_pct": number(row.get("required_e2e_pct")),
                    "sd_ms": number(row.get("sd_ms")),
                    "n": int(row["n"])
                    if isinstance(row.get("n"), int | float)
                    else None,
                    "acceptance_target": bool(
                        row.get(
                            "acceptance_target",
                            role in {"denominator", "exclusive_required", "exclusive_detail"},
                        )
                    ),
                    "included_in_required_e2e": bool(
                        row.get("included_in_required_e2e", True)
                    ),
                    "additive_required_work": bool(
                        row.get(
                            "additive_required_work",
                            role in {"exclusive_required", "exclusive_detail"},
                        )
                    ),
                    "placement_only": bool(
                        row.get("placement_only", role == "placement_detail")
                    ),
                    "basis": row.get("basis"),
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            item.get("table") != "rollup",
            -float(item.get("cpp_ms") or 0.0),
            str(item.get("process") or ""),
        ),
    )


def acceptance_required_process_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("acceptance_target"))
        and not bool(row.get("placement_only"))
    ]


def placement_process_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("placement_only")) or row.get("role") == "placement_detail"
    ]


def required_process_delta_rows(
    baseline_split: dict[str, Any],
    candidates: list[tuple[Path, dict[str, Any]]],
    *,
    acceptance_only: bool = True,
) -> list[dict[str, Any]]:
    baseline_rows = required_process_rows(baseline_split)
    if acceptance_only:
        baseline_rows = acceptance_required_process_rows(baseline_rows)
    baseline_by_process = {
        (str(row.get("table") or ""), str(row.get("process") or "")): row
        for row in baseline_rows
        if row.get("process")
    }
    rows: list[dict[str, Any]] = []
    for path, candidate_split in candidates:
        candidate_label = row_label(first_row(candidate_split), path)
        candidate_rows = required_process_rows(candidate_split)
        if acceptance_only:
            candidate_rows = acceptance_required_process_rows(candidate_rows)
        for row in candidate_rows:
            process_key = (str(row.get("table") or ""), str(row.get("process") or ""))
            baseline = baseline_by_process.get(process_key)
            if baseline is None:
                continue
            baseline_cpp_ms = number(baseline.get("cpp_ms"))
            candidate_cpp_ms = number(row.get("cpp_ms"))
            if baseline_cpp_ms is None or candidate_cpp_ms is None:
                continue
            delta_ms = candidate_cpp_ms - baseline_cpp_ms
            rows.append(
                {
                    "candidate": candidate_label,
                    "candidate_component_split": str(path),
                    "table": row.get("table"),
                    "bucket": row.get("bucket"),
                    "parent_process": row.get("parent_process"),
                    "process": row.get("process"),
                    "role": row.get("role"),
                    "measurement_class": row.get("measurement_class"),
                    "baseline_cpp_ms": baseline_cpp_ms,
                    "candidate_cpp_ms": candidate_cpp_ms,
                    "cpp_delta_ms": delta_ms,
                    "cpp_delta_pct": pct(delta_ms, baseline_cpp_ms),
                    "baseline_required_e2e_pct": number(
                        baseline.get("required_e2e_pct")
                    ),
                    "candidate_required_e2e_pct": number(row.get("required_e2e_pct")),
                    "basis": row.get("basis"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row.get("table") != "rollup",
            -abs(float(row.get("cpp_delta_ms") or 0.0)),
            str(row.get("process") or ""),
        ),
    )


def parity_status(parity: dict[str, Any] | None) -> str:
    if parity is None:
        return "missing"
    if (
        int(parity.get("diff_rows", -1)) == 0
        and int(parity.get("tolerance_diff_rows", -1)) == 0
    ):
        return "exact"
    return "failed"


def python_quality_parity_status(parity: dict[str, Any] | None) -> str:
    if parity is None:
        return "missing"
    if int(parity.get("tolerance_diff_rows", -1)) != 0:
        return "failed"
    if int(parity.get("diff_rows", -1)) == 0:
        return "exact"
    return "quality_pass"


def parity_summary(parity: dict[str, Any] | None) -> dict[str, Any] | None:
    if parity is None:
        return None
    summary = parity.get("summary")
    return {
        "diff_rows": parity.get("diff_rows"),
        "tolerance_diff_rows": parity.get("tolerance_diff_rows"),
        "lhs_rows": parity.get("lhs_rows"),
        "rhs_rows": parity.get("rhs_rows"),
        "min_bbox_iou": summary.get("min_bbox_iou")
        if isinstance(summary, dict)
        else None,
        "max_bbox_delta_px": summary.get("max_bbox_delta_px")
        if isinstance(summary, dict)
        else None,
        "max_score_abs_delta": summary.get("max_score_abs_delta")
        if isinstance(summary, dict)
        else None,
        "max_abs_mask_area_rel_delta": summary.get("max_abs_mask_area_rel_delta")
        if isinstance(summary, dict)
        else None,
        "min_mask_pixel_iou": summary.get("min_mask_pixel_iou")
        if isinstance(summary, dict)
        else None,
        "max_mask_pixel_xor": summary.get("max_mask_pixel_xor")
        if isinstance(summary, dict)
        else None,
        "mask_hash_equal_rows": summary.get("mask_hash_equal_rows")
        if isinstance(summary, dict)
        else None,
    }


def python_quality_parity_summary(parity: dict[str, Any] | None) -> dict[str, Any] | None:
    summary = parity_summary(parity)
    if summary is None:
        return None
    summary["status"] = python_quality_parity_status(parity)
    summary["lhs"] = parity.get("lhs")
    summary["rhs"] = parity.get("rhs")
    summary["tolerance"] = parity.get("tolerance")
    return summary


def candidate_decision_rows(
    deltas: list[dict[str, Any]],
    parity_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in deltas:
        candidate = str(row["candidate"])
        component = str(row["component"])
        by_candidate.setdefault(candidate, {})[component] = row

    decisions: list[dict[str, Any]] = []
    for candidate, components in by_candidate.items():
        e2e = components.get("required_e2e_ms")
        tail_graph = components.get("tail_encode_graph_compute_ms")
        core = components.get("required_core_compute_ms")
        e2e_delta = e2e.get("delta_ms") if e2e else None
        tail_delta = tail_graph.get("delta_ms") if tail_graph else None
        e2e_signal = e2e.get("delta_standard_error_score") if e2e else None
        tail_signal = (
            tail_graph.get("delta_standard_error_score") if tail_graph else None
        )
        core_delta = core.get("delta_ms") if core else None
        improves_required_e2e = isinstance(e2e_delta, int | float) and e2e_delta < 0.0
        improves_tail_graph = isinstance(tail_delta, int | float) and tail_delta < 0.0
        has_e2e_signal = isinstance(e2e_signal, int | float)
        has_tail_signal = isinstance(tail_signal, int | float)
        parity = (parity_by_candidate or {}).get(candidate)
        current_parity_status = parity_status(parity)
        signal_ok = (
            has_e2e_signal
            and has_tail_signal
            and e2e_signal <= ACCEPTANCE_SIGNAL_THRESHOLD
            and tail_signal <= ACCEPTANCE_SIGNAL_THRESHOLD
        )
        if e2e_delta is None or tail_delta is None:
            decision = "incomplete_measurement"
            basis = "requires required_e2e_ms and tail_encode_graph_compute_ms"
        elif not improves_required_e2e or not improves_tail_graph:
            decision = "reject_or_keep_diagnostic"
            basis = "requires required_e2e_ms and tail_encode_graph_compute_ms to both improve"
        elif not has_e2e_signal or not has_tail_signal:
            decision = "needs_more_repeats"
            basis = "mean improved, but standard-error signal is unavailable"
        elif not signal_ok:
            decision = "needs_more_repeats"
            basis = (
                "both means improve, but standard-error signal is below the acceptance threshold "
                f"of z <= {ACCEPTANCE_SIGNAL_THRESHOLD:.1f}"
            )
        elif current_parity_status == "exact":
            decision = "candidate_speedup"
            basis = (
                "required_e2e_ms and tail_encode_graph_compute_ms both improve with "
                f"z <= {ACCEPTANCE_SIGNAL_THRESHOLD:.1f}, and parity is exact"
            )
        elif current_parity_status == "missing":
            decision = "needs_parity_check"
            basis = "speed signal passed, but artifact-producing JSONL/mask parity was not supplied"
        else:
            decision = "reject_parity"
            basis = "speed signal passed, but artifact-producing JSONL/mask parity did not match"
        decisions.append(
            {
                "candidate": candidate,
                "required_e2e_delta_ms": e2e_delta,
                "tail_encode_graph_compute_delta_ms": tail_delta,
                "required_core_compute_delta_ms": core_delta,
                "required_e2e_signal": e2e_signal,
                "tail_encode_graph_compute_signal": tail_signal,
                "decision": decision,
                "basis": basis,
                "parity_status": current_parity_status,
                "parity": parity_summary(parity),
            }
        )
    return sorted(decisions, key=lambda row: float(row["required_e2e_delta_ms"] or 0.0))


def component_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    components = row.get("components", {})
    if not isinstance(components, dict):
        return []
    required_e2e = metric_mean(components.get("required_e2e_ms"))
    interesting = (
        "required_e2e_ms",
        "model_e2e_ms",
        "input_load_ms",
        "session_setup_ms",
        "frame0_encode_ms",
        "frame0_encode_graph_compute_ms",
        "frame0_prompt_model_ms",
        "tail_encode_required_total_ms",
        "tail_encode_graph_compute_ms",
        "tail_encode_non_compute_ms",
        "tail_encode_graph_admin_ms",
        "tail_encode_preprocess_ms",
        "tail_encode_state_update_ms",
        "tail_encode_pe_build_ms",
        "tail_encode_remainder_ms",
        "tail_inline_encode_non_compute_ms",
        "tail_preencode_encode_non_compute_ms",
        "required_tail_propagate_ms",
        "tail_propagate_graph_compute_ms",
        "tail_propagate_graph_admin_ms",
        "required_core_compute_ms",
        "required_non_core_compute_ms",
        "required_graph_admin_ms",
    )
    rows: list[dict[str, Any]] = []
    for key in interesting:
        metric = components.get(key)
        mean = metric_mean(metric)
        if mean is None:
            continue
        rows.append(
            {
                "component": key,
                "mean_ms": mean,
                "sd_ms": metric_sd(metric),
                "n": metric_n(metric),
                "required_e2e_pct": pct(mean, required_e2e),
            }
        )
    return sorted(rows, key=lambda item: float(item["mean_ms"]), reverse=True)


def process_rows(component_split: dict[str, Any]) -> list[dict[str, Any]]:
    rows = component_split.get("process_rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cpp_ms = number(row.get("cpp_ms"))
        python_ms = number(row.get("python_ms"))
        ratio = speed_ratio(python_ms, cpp_ms)
        out.append(
            {
                "bucket": row.get("bucket"),
                "process": row.get("process"),
                "role": row.get("role"),
                "cpp_ms": cpp_ms,
                "python_ms": python_ms,
                "python_diagnostic_ms": number(row.get("python_diagnostic_ms")),
                "python_over_cpp_ratio": ratio,
                "cpp_slower_ms": cpp_ms - python_ms
                if cpp_ms is not None and python_ms is not None and cpp_ms > python_ms
                else None,
                "cpp_in_required_e2e": row.get("cpp_in_required_e2e"),
                "python_in_session_e2e": row.get("python_in_session_e2e"),
                "exact_family_match": row.get("exact_family_match"),
                "comparable_session_e2e_claim": row.get(
                    "comparable_session_e2e_claim"
                ),
                "same_contract_session_e2e_diagnostic": row.get(
                    "same_contract_session_e2e_diagnostic"
                ),
                "basis": row.get("basis"),
            }
        )
    return sorted(
        out,
        key=lambda item: (
            item["cpp_slower_ms"] is None,
            -float(item["cpp_slower_ms"] or 0.0),
            -float(item["cpp_ms"] or 0.0),
        ),
    )


def node_stage_rows(node_names: dict[str, Any], label: str) -> list[dict[str, Any]]:
    if not node_names:
        return []
    labels = node_names.get("labels")
    if not isinstance(labels, dict):
        return []
    selected = labels.get(label)
    if not isinstance(selected, dict):
        candidates = [key for key in labels if str(key).startswith(label)]
        selected = labels.get(candidates[-1]) if candidates else None
    if not isinstance(selected, dict):
        return []
    stages = selected.get("by_stage")
    if not isinstance(stages, dict):
        return []
    total = sum(
        number(stage.get("sum_ms")) or 0.0
        for stage in stages.values()
        if isinstance(stage, dict)
    )
    rows: list[dict[str, Any]] = []
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        sum_ms = number(stage.get("sum_ms"))
        if sum_ms is None:
            continue
        rows.append(
            {
                "stage": stage_name,
                "count": int(stage.get("count", 0)),
                "sum_ms": sum_ms,
                "mean_ms": number(stage.get("mean_ms")),
                "profile_pct": pct(sum_ms, total),
            }
        )
    return sorted(rows, key=lambda item: float(item["sum_ms"]), reverse=True)


def tail_graph_stage_budget_rows(
    stages: list[dict[str, Any]],
    *,
    required_e2e_ms: float | None,
    tail_graph_compute_ms: float | None,
    mode: str = "tail",
    basis: str | None = None,
) -> list[dict[str, Any]]:
    profile_total_ms = sum(
        number(stage.get("sum_ms")) or 0.0
        for stage in stages
        if isinstance(stage, dict)
    )
    if (
        not stages
        or profile_total_ms <= 0.0
        or tail_graph_compute_ms is None
        or tail_graph_compute_ms <= 0.0
    ):
        return []

    rows: list[dict[str, Any]] = []
    for stage in stages:
        profile_sum_ms = number(stage.get("sum_ms"))
        if profile_sum_ms is None:
            continue
        profile_share = profile_sum_ms / profile_total_ms
        required_budget_ms = tail_graph_compute_ms * profile_share
        rows.append(
            {
                "stage": stage.get("stage"),
                "mode": mode,
                "count": stage.get("count"),
                "profile_sum_ms": profile_sum_ms,
                "profile_share": profile_share,
                "profile_pct": profile_share * 100.0,
                "required_budget_ms": required_budget_ms,
                "required_e2e_pct": pct(required_budget_ms, required_e2e_ms),
                "mean_ms": stage.get("mean_ms"),
                "basis": basis
                or (
                    "GGML_CUDA_PROFILE_NODES stage share normalized to the normal "
                    "required tail image-encode graph_compute_ms."
                ),
            }
        )
    return sorted(rows, key=lambda row: float(row["required_budget_ms"]), reverse=True)


def occurrence_label(label: str, occurrence: int) -> str:
    prefix, separator, suffix = label.partition("#")
    if separator and suffix.isdigit():
        return f"{prefix}#{occurrence:02d}"
    return label


def tail_graph_budget_group_name(stage: Any) -> str:
    name = str(stage or "")
    if name in {"sam3-vit:prefix", "sam3-vit:prefix-matmul", "sam3-vit:prefix-pos"}:
        return "image.vit.prefix"
    if name in {"sam3-vit:mlp-fc1-matmul", "sam3-vit:mlp-fc2-matmul"}:
        return "image.vit.mlp_matmul"
    if name == "sam3-vit:qkv-matmul":
        return "image.vit.qkv_matmul"
    if name == "sam3-vit:proj-matmul":
        return "image.vit.proj_matmul"
    if name in {"sam3-vit:window-attn", "sam3-vit:global-attn"}:
        return "image.vit.attention"
    if name == "sam3-vit:mlp-gelu":
        return "image.vit.mlp_gelu"
    if name in {
        "sam3-vit:norm",
        "sam3-vit:norm-copy",
        "sam3-vit:output",
        "sam3-vit:output-layout",
        "sam3-vit:block-out-residual-norm-copy",
        "sam3-vit:after-attn-residual-norm-copy",
    }:
        return "image.vit.norm_residual_copy"
    if name.startswith("sam3-neck:"):
        return "image.neck"
    if name in {
        "sam3-vit:qk-layout-rope",
        "sam3-vit:window-unpart",
        "sam3-vit:window-part",
        "sam3-vit:attn-bf16-copy",
        "sam3-vit:window-part-bf16-copy",
    }:
        return "image.vit.layout_rope_copy"
    if name.startswith("sam3-vit:"):
        return "image.vit.other"
    return "image.other"


def tail_graph_budget_group_rows(
    budget_rows: list[dict[str, Any]], *, required_e2e_ms: float | None
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in budget_rows:
        group_name = tail_graph_budget_group_name(row.get("stage"))
        group = groups.setdefault(
            group_name,
            {
                "group": group_name,
                "required_budget_ms": 0.0,
                "profile_sum_ms": 0.0,
                "profile_share": 0.0,
                "count": 0,
                "stages": [],
                "modes": [],
            },
        )
        group["required_budget_ms"] += float(row.get("required_budget_ms") or 0.0)
        group["profile_sum_ms"] += float(row.get("profile_sum_ms") or 0.0)
        group["profile_share"] += float(row.get("profile_share") or 0.0)
        group["count"] += int(row.get("count") or 0)
        group["stages"].append(row.get("stage"))
        if row.get("mode") not in group["modes"]:
            group["modes"].append(row.get("mode"))

    out = []
    for group in groups.values():
        out.append(
            {
                **group,
                "profile_pct": float(group["profile_share"]) * 100.0,
                "required_e2e_pct": pct(group["required_budget_ms"], required_e2e_ms),
            }
        )
    return sorted(out, key=lambda row: float(row["required_budget_ms"]), reverse=True)


def required_image_graph_stage_budget_rows(
    node_names: dict[str, Any],
    *,
    node_label: str,
    required_e2e_ms: float | None,
    frame0_graph_compute_ms: float | None,
    tail_graph_compute_ms: float | None,
) -> list[dict[str, Any]]:
    frame0_label = occurrence_label(node_label, 1)
    frame0_stages = node_stage_rows(node_names, frame0_label)
    tail_stages = node_stage_rows(node_names, node_label)
    budget_rows = tail_graph_stage_budget_rows(
        frame0_stages,
        required_e2e_ms=required_e2e_ms,
        tail_graph_compute_ms=frame0_graph_compute_ms,
        mode="frame0",
        basis=(
            "GGML_CUDA_PROFILE_NODES stage share normalized to the normal "
            "required frame0 image-encode graph_compute_ms."
        ),
    ) + tail_graph_stage_budget_rows(
        tail_stages,
        required_e2e_ms=required_e2e_ms,
        tail_graph_compute_ms=tail_graph_compute_ms,
        mode="tail",
        basis=(
            "GGML_CUDA_PROFILE_NODES stage share normalized to the normal "
            "required tail image-encode graph_compute_ms."
        ),
    )
    return sorted(budget_rows, key=lambda row: float(row["required_budget_ms"]), reverse=True)


def required_image_graph_group_budget_rows(
    budget_rows: list[dict[str, Any]], *, required_e2e_ms: float | None
) -> list[dict[str, Any]]:
    return tail_graph_budget_group_rows(budget_rows, required_e2e_ms=required_e2e_ms)


def stage_profile_rows(stage_profile: dict[str, Any]) -> list[dict[str, Any]]:
    by_mode_stage = stage_profile.get("by_mode_stage")
    if not isinstance(by_mode_stage, dict):
        return []
    rows: list[dict[str, Any]] = []
    for mode, stages in by_mode_stage.items():
        if not isinstance(stages, dict):
            continue
        total = sum(
            number(stage.get("sum_ms")) or 0.0
            for stage in stages.values()
            if isinstance(stage, dict)
        )
        for stage_name, stage in stages.items():
            if not isinstance(stage, dict):
                continue
            sum_ms = number(stage.get("sum_ms"))
            if sum_ms is None:
                continue
            top_ops = stage.get("top_ops")
            rows.append(
                {
                    "mode": mode,
                    "stage": stage_name,
                    "count": int(stage.get("count", 0)),
                    "sum_ms": sum_ms,
                    "mean_ms": number(stage.get("mean_ms")),
                    "profile_pct": pct(sum_ms, total),
                    "top_ops": top_ops if isinstance(top_ops, list) else [],
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            {"tracking": 0, "frame0": 1}.get(str(item.get("mode") or ""), 2),
            -float(item.get("sum_ms") or 0.0),
            str(item.get("stage") or ""),
        ),
    )


def cublaslt_rows(cublaslt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = cublaslt.get("summary")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("key")
        if isinstance(key, list):
            group = "/".join(str(item) for item in key)
        else:
            group = str(key)
        matmul_sum_ms = number(row.get("matmul_sum_ms"))
        if matmul_sum_ms is None:
            continue
        out.append(
            {
                "group": group,
                "n": int(row.get("n", 0)),
                "matmul_sum_ms": matmul_sum_ms,
                "matmul_mean_ms": number(row.get("matmul_mean_ms")),
                "host_total_sum_ms": number(row.get("host_total_sum_ms")),
                "selected_algo_indices": row.get(
                    "selected_algo_indices", row.get("algo_indices")
                ),
                "src0_types": row.get("src0_types"),
                "src1_types": row.get("src1_types"),
                "dst_types": row.get("dst_types"),
            }
        )
    return sorted(out, key=lambda item: float(item["matmul_sum_ms"]), reverse=True)


FATTN_PROFILE_GROUP_STAGE = {
    "sam3-vit-window-head64": "sam3-vit:window-attn",
    "sam3-vit-global-head64": "sam3-vit:global-attn",
}


def profile_metric_steady_mean(group: dict[str, Any], metric: str) -> float | None:
    metrics = group.get("metrics")
    if not isinstance(metrics, dict):
        return None
    row = metrics.get(metric)
    if not isinstance(row, dict):
        return None
    return number(row.get("steady_mean"))


def optional_sum(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def budget_sum_for_stage(rows: list[dict[str, Any]], stage_name: str) -> float | None:
    matched = [
        number(row.get("required_budget_ms"))
        for row in rows
        if row.get("stage") == stage_name
    ]
    return optional_sum(matched)


def fmt_counter_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{key}:{count}" for key, count in sorted(value.items()))


def fattn_profile_rows(
    profiles: dict[str, dict[str, Any]],
    *,
    tail_graph_stage_budget: list[dict[str, Any]],
    required_image_graph_stage_budget: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, profile in profiles.items():
        groups = profile.get("groups")
        if not isinstance(groups, dict):
            continue
        for group_name, group in groups.items():
            if not isinstance(group, dict):
                continue
            q_convert = profile_metric_steady_mean(group, "Q_convert_ms")
            k_convert = profile_metric_steady_mean(group, "K_convert_ms")
            v_convert = profile_metric_steady_mean(group, "V_convert_ms")
            qkv_convert = optional_sum([q_convert, k_convert, v_convert])
            total = profile_metric_steady_mean(group, "total_ms")
            stage_name = FATTN_PROFILE_GROUP_STAGE.get(str(group_name))
            rows.append(
                {
                    "profile": label,
                    "group": group_name,
                    "stage": stage_name,
                    "count": group.get("count"),
                    "types": group.get("types"),
                    "ncols1": group.get("ncols1"),
                    "schedule": group.get("schedule"),
                    "total_steady_mean_ms": total,
                    "kernel_steady_mean_ms": profile_metric_steady_mean(
                        group, "kernel_ms"
                    ),
                    "q_convert_steady_mean_ms": q_convert,
                    "k_convert_steady_mean_ms": k_convert,
                    "v_convert_steady_mean_ms": v_convert,
                    "qkv_convert_steady_mean_ms": qkv_convert,
                    "fixup_steady_mean_ms": profile_metric_steady_mean(
                        group, "fixup_ms"
                    ),
                    "qkv_convert_pct_of_total": pct(qkv_convert, total),
                    "kernel_pct_of_total": pct(
                        profile_metric_steady_mean(group, "kernel_ms"), total
                    ),
                    "tail_budget_ms": budget_sum_for_stage(
                        tail_graph_stage_budget, stage_name
                    )
                    if stage_name
                    else None,
                    "required_image_budget_ms": budget_sum_for_stage(
                        required_image_graph_stage_budget, stage_name
                    )
                    if stage_name
                    else None,
                    "source": profile.get("source"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("profile") or ""),
            -float(row.get("total_steady_mean_ms") or 0.0),
            str(row.get("group") or ""),
        ),
    )


def isolated_bench_rows(benches: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    def sample_stats(values: Any, divisor: float) -> dict[str, float | int | None]:
        if not isinstance(values, list):
            return {
                "n": None,
                "mean": None,
                "sd": None,
                "se": None,
                "ci95_half_width": None,
            }
        samples = [float(value) / divisor for value in values if isinstance(value, int | float)]
        if not samples:
            return {
                "n": 0,
                "mean": None,
                "sd": None,
                "se": None,
                "ci95_half_width": None,
            }
        mean = sum(samples) / len(samples)
        sd = None
        se = None
        ci95 = None
        if len(samples) >= 2:
            variance = sum((value - mean) ** 2 for value in samples) / (len(samples) - 1)
            sd = math.sqrt(variance)
            se = sd / math.sqrt(len(samples))
            ci95 = 1.96 * se
        return {
            "n": len(samples),
            "mean": mean,
            "sd": sd,
            "se": se,
            "ci95_half_width": ci95,
        }

    def group_name(name: str) -> str:
        group, separator, _ = name.partition("/")
        return group if separator else ""

    def leaf_name(name: str) -> str:
        return name.rsplit("/", maxsplit=1)[-1]

    rows: list[dict[str, Any]] = []
    baseline = benches.get("baseline")
    default_baseline_mean = (
        number(baseline.get("mean_ms_per_frame"))
        if isinstance(baseline, dict)
        else None
    )
    if default_baseline_mean is None:
        default_baseline_mean = next(
            (
                number(bench.get("mean_ms_per_frame"))
                for name, bench in benches.items()
                if name.lower().startswith("baseline") and isinstance(bench, dict)
            ),
            None,
        )
    baseline_by_group: dict[str, dict[str, float | int | None]] = {}
    for name, bench in benches.items():
        if not isinstance(bench, dict):
            continue
        if leaf_name(name).lower() != "baseline":
            continue
        mean = number(bench.get("mean_ms_per_frame"))
        if mean is not None:
            batch_size = bench.get("batch_size")
            per_frame_divisor = (
                float(batch_size) if isinstance(batch_size, int | float) and batch_size else 1.0
            )
            baseline_by_group[group_name(name)] = {
                **sample_stats(bench.get("samples_ms"), per_frame_divisor),
                "mean": mean,
            }
    for name, bench in benches.items():
        if not isinstance(bench, dict):
            continue
        mean_per_frame = number(bench.get("mean_ms_per_frame"))
        median = number(bench.get("median_ms"))
        if mean_per_frame is None and median is None:
            continue
        batch_size = bench.get("batch_size")
        batch_size_number = (
            int(batch_size) if isinstance(batch_size, int | float) else None
        )
        per_frame_divisor = float(batch_size_number or 1)
        min_ms = number(bench.get("min_ms"))
        max_ms = number(bench.get("max_ms"))
        samples = sample_stats(bench.get("samples_ms"), per_frame_divisor)
        group_baseline = baseline_by_group.get(group_name(name))
        group_baseline_mean = (
            number(group_baseline.get("mean")) if isinstance(group_baseline, dict) else None
        )
        if group_baseline_mean is None:
            group_baseline_mean = default_baseline_mean
        baseline_se = (
            number(group_baseline.get("se")) if isinstance(group_baseline, dict) else None
        )
        sample_se = number(samples.get("se"))
        delta = (
            mean_per_frame - group_baseline_mean
            if mean_per_frame is not None and group_baseline_mean is not None
            else None
        )
        combined_se = (
            math.sqrt(sample_se**2 + baseline_se**2)
            if sample_se is not None and baseline_se is not None
            else None
        )
        rows.append(
            {
                "name": name,
                "model": bench.get("model"),
                "backend": bench.get("backend"),
                "batch_size": batch_size_number,
                "include_tracker_neck": bench.get("include_tracker_neck"),
                "mean_ms": number(bench.get("mean_ms")),
                "median_ms": median,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "mean_ms_per_frame": mean_per_frame,
                "sample_n": samples.get("n"),
                "sample_sd_ms_per_frame": samples.get("sd"),
                "sample_ci95_half_width_ms_per_frame": samples.get("ci95_half_width"),
                "mean_ms_per_frame_delta_vs_baseline": delta,
                "mean_ms_per_frame_delta_pct_vs_baseline": pct(
                    mean_per_frame - group_baseline_mean, group_baseline_mean
                )
                if mean_per_frame is not None and group_baseline_mean is not None
                else None,
                "delta_vs_baseline_se_signal": delta / combined_se
                if delta is not None and combined_se not in (None, 0.0)
                else None,
                "median_ms_per_frame": median / per_frame_divisor
                if median is not None
                else None,
                "min_ms_per_frame": min_ms / per_frame_divisor
                if min_ms is not None
                else None,
                "max_ms_per_frame": max_ms / per_frame_divisor
                if max_ms is not None
                else None,
                "lane_max_abs_diff": number(bench.get("lane_max_abs_diff")),
            }
        )
    return sorted(rows, key=lambda row: str(row["name"]))


def fmt_shape(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    dims = [str(int(dim)) for dim in value if isinstance(dim, int | float)]
    return "x".join(dims) if dims else None


def rows_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, str):
            out.setdefault(value, []).append(row)
    return out


def combined_budget(
    matched: list[dict[str, Any]],
    *,
    basis: str,
    names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "budget_basis": basis,
        "budget_names": [str(name) for name in names],
        "budget_ms": sum(number(row.get("required_budget_ms")) or 0.0 for row in matched),
        "budget_required_e2e_pct": sum(
            number(row.get("required_e2e_pct")) or 0.0 for row in matched
        ),
        "budget_profile_sum_ms": sum(number(row.get("profile_sum_ms")) or 0.0 for row in matched),
        "budget_count": sum(int(row.get("count") or 0) for row in matched),
    }


def stage_budget_projection(
    stage_name: str,
    *,
    stage_budget_by_stage: dict[str, list[dict[str, Any]]],
    group_budget_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    stages = STAGE_BUDGET_STAGES.get(stage_name)
    if stages:
        matched_stages = [
            row
            for stage in stages
            for row in stage_budget_by_stage.get(stage, [])
        ]
        if all(stage in stage_budget_by_stage for stage in stages):
            return combined_budget(matched_stages, basis="stage", names=stages)

    groups = STAGE_BUDGET_GROUPS.get(stage_name)
    if not groups:
        return {
            "budget_basis": None,
            "budget_names": [],
            "budget_ms": None,
            "budget_required_e2e_pct": None,
            "budget_projection_reason": "no direct isolated-stage mapping",
        }

    matched_groups = [
        row
        for group in groups
        for row in group_budget_by_group.get(group, [])
    ]
    if all(group in group_budget_by_group for group in groups):
        return combined_budget(matched_groups, basis="group", names=groups)

    return {
        "budget_basis": None,
        "budget_names": list(stages or groups),
        "budget_ms": None,
        "budget_required_e2e_pct": None,
        "budget_projection_reason": "mapped budget stage/group missing from audit",
    }


def add_isolated_stage_budget_projection(
    row: dict[str, Any],
    *,
    tail_stage_budget_by_stage: dict[str, list[dict[str, Any]]],
    tail_group_budget_by_group: dict[str, list[dict[str, Any]]],
    required_stage_budget_by_stage: dict[str, list[dict[str, Any]]],
    required_group_budget_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    stage_name = str(row.get("stage_name") or "")
    effective_ms = number(row.get("local_estimate_ms"))
    measurement_kind = "derived" if effective_ms is not None else "direct"
    if effective_ms is None:
        effective_ms = number(row.get("mean_ms"))

    tail_budget = stage_budget_projection(
        stage_name,
        stage_budget_by_stage=tail_stage_budget_by_stage,
        group_budget_by_group=tail_group_budget_by_group,
    )
    required_budget = stage_budget_projection(
        stage_name,
        stage_budget_by_stage=required_stage_budget_by_stage,
        group_budget_by_group=required_group_budget_by_group,
    )

    def projected_scale(budget: dict[str, Any]) -> float | None:
        budget_ms = number(budget.get("budget_ms"))
        if budget_ms is None or effective_ms is None or effective_ms <= 0.0:
            return None
        return budget_ms / effective_ms

    return {
        **row,
        "measurement_kind": measurement_kind,
        "effective_stage_ms": effective_ms,
        "tail_budget_basis": tail_budget.get("budget_basis"),
        "tail_budget_names": tail_budget.get("budget_names"),
        "tail_budget_ms": tail_budget.get("budget_ms"),
        "tail_budget_required_e2e_pct": tail_budget.get("budget_required_e2e_pct"),
        "tail_budget_profile_sum_ms": tail_budget.get("budget_profile_sum_ms"),
        "tail_budget_count": tail_budget.get("budget_count"),
        "tail_budget_scale_per_stage_ms": projected_scale(tail_budget),
        "required_image_budget_basis": required_budget.get("budget_basis"),
        "required_image_budget_names": required_budget.get("budget_names"),
        "required_image_budget_ms": required_budget.get("budget_ms"),
        "required_image_budget_required_e2e_pct": required_budget.get(
            "budget_required_e2e_pct"
        ),
        "required_image_budget_profile_sum_ms": required_budget.get("budget_profile_sum_ms"),
        "required_image_budget_count": required_budget.get("budget_count"),
        "required_image_budget_scale_per_stage_ms": projected_scale(required_budget),
    }


def isolated_stage_rows(
    benches: dict[str, dict[str, Any]],
    *,
    tail_graph_stage_budget: list[dict[str, Any]],
    tail_graph_group_budget: list[dict[str, Any]],
    required_image_graph_stage_budget: list[dict[str, Any]],
    required_image_graph_group_budget: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tail_stage_budget_by_stage = rows_by_key(tail_graph_stage_budget, "stage")
    tail_group_budget_by_group = rows_by_key(tail_graph_group_budget, "group")
    required_stage_budget_by_stage = rows_by_key(required_image_graph_stage_budget, "stage")
    required_group_budget_by_group = rows_by_key(required_image_graph_group_budget, "group")
    rows: list[dict[str, Any]] = []
    for name, bench in benches.items():
        if not isinstance(bench, dict):
            continue
        stage_rows = bench.get("rows")
        if not isinstance(stage_rows, list):
            continue
        for row in stage_rows:
            if not isinstance(row, dict):
                continue
            bench_mode = row.get("bench_mode")
            if bench_mode != "isolated_stage":
                continue
            stage = row.get("stage")
            stage_name = row.get("stage_name")
            mean_ms = number(row.get("mean_ms_per_frame"))
            if mean_ms is None:
                mean_ms = number(row.get("mean_ms"))
            if not isinstance(stage, int | float) or not isinstance(stage_name, str):
                continue
            local_estimate_ms = None
            local_estimate_basis = None
            if stage_name == "qkv_rope":
                local_estimate_ms = number(row.get("derived_rope_mean_ms"))
                local_estimate_basis = "qkv_rope - qkv_layout"
            elif stage_name == "attn_core":
                local_estimate_ms = number(row.get("derived_attn_core_mean_ms"))
                local_estimate_basis = "attn_core - qkv_rope"
            elif stage_name == "mlp_fc1_gelu":
                local_estimate_ms = number(row.get("delta_from_split_mean_ms"))
                local_estimate_basis = "direct fc1+gelu - split fc1+gelu"
            base_row = {
                "bench": name,
                "block": row.get("block_stage_index", row.get("block")),
                "stage": int(stage),
                "stage_name": stage_name,
                "mean_ms": mean_ms,
                "median_ms": number(row.get("median_ms_per_frame"))
                or number(row.get("median_ms")),
                "sd_ms": number(row.get("sd_ms_per_frame")) or number(row.get("sd_ms")),
                "ci95_ms": number(row.get("ci95_ms_per_frame")) or number(row.get("ci95_ms")),
                "input_ne": fmt_shape(row.get("input_ne")),
                "output_ne": fmt_shape(row.get("output_ne")),
                "local_estimate_ms": local_estimate_ms,
                "local_estimate_basis": local_estimate_basis,
                "source": row.get("source"),
                "optimization_target": stage_name
                in {
                    "mlp",
                    "mlp_fc2",
                    "mlp_fc1",
                    "qkv_proj",
                    "attn_core",
                    "block",
                },
            }
            rows.append(
                add_isolated_stage_budget_projection(
                    base_row,
                    tail_stage_budget_by_stage=tail_stage_budget_by_stage,
                    tail_group_budget_by_group=tail_group_budget_by_group,
                    required_stage_budget_by_stage=required_stage_budget_by_stage,
                    required_group_budget_by_group=required_group_budget_by_group,
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            not bool(row.get("optimization_target")),
            -float(row.get("required_image_budget_ms") or row.get("tail_budget_ms") or 0.0),
            -float(row.get("effective_stage_ms") or 0.0),
            str(row.get("bench") or ""),
            int(row.get("stage") or 0),
        ),
    )


def required_work_gate(process: str, role: str | None) -> str:
    if process == "required_e2e.total":
        return "primary acceptance: reduce required_e2e_ms and preserve official Python parity"
    if role == "placement_detail":
        return "placement only: cannot be accepted unless required_e2e_ms also falls"
    if process == "tail.image_encode":
        return "reduce tail_encode_required_total_ms and required_e2e_ms"
    if process == "tail.image_encode.graph_compute":
        return "reduce tail_encode_graph_compute_ms, normalized tail graph budget, and required_e2e_ms"
    if process.startswith("tail.image_encode.non_compute"):
        return "reduce tail_encode_non_compute_ms and show the delta in required_e2e_ms"
    if process == "tail.propagate":
        return "reduce required_tail_propagate_ms and required_e2e_ms"
    if process.startswith("tail.propagate."):
        return "reduce required_tail_propagate_ms and show the delta in required_e2e_ms"
    if process == "frame0.image_encode":
        return "reduce frame0_encode_ms and required_e2e_ms"
    if process == "frame0.image_encode.graph_compute":
        return "reduce frame0_encode_graph_compute_ms and required_e2e_ms"
    if process == "frame0.prompt_and_tracker_init":
        return "reduce frame0_prompt_model_ms and required_e2e_ms"
    if process == "input.prepare":
        return "reduce input_load_ms without changing the input contract"
    return "reduce the parent required work and required_e2e_ms"


E2E_PROCESS_MAP_ORDER = {
    "required_session_e2e": 0,
    "model_e2e": 1,
    "input_prepare_or_start_session": 2,
    "frame0_encode": 3,
    "frame0_prompt_or_detection": 4,
    "tail_required_model": 5,
    "tail_image_encode_or_backbone": 6,
    "tail_image_encode.graph_compute": 7,
    "tail_image_encode.non_compute_required": 8,
    "tail_image_encode.inline": 9,
    "tail_image_encode.inline.graph_compute": 10,
    "tail_image_encode.preencode": 11,
    "tail_image_encode.preencode.graph_compute": 12,
    "tail_propagate_or_tracker": 13,
    "tail_propagate.graph_compute": 14,
    "tail_propagate.input_upload": 15,
    "tail_propagate.rope_setup": 16,
    "tail_propagate.graph_admin": 17,
    "cpp_required_core_graph_compute": 18,
    "cpp_required_non_core": 19,
    "cpp_required_remainder": 20,
}


def e2e_process_measurement_gate(process: str, role: str | None) -> str:
    if process == "required_session_e2e":
        return "formal acceptance denominator: C++ input+model E2E must fall while Python quality passes"
    if process == "model_e2e":
        return "formal model wall subtotal; useful only when required_session_e2e also improves"
    if process == "input_prepare_or_start_session":
        return "formal but usually not the bottleneck; keep the same input contract"
    if process == "frame0_encode":
        return "C++ required image encode; optimize with the same image graph contract"
    if process == "frame0_prompt_or_detection":
        return "formal wall row with Python hook context; preserve selected target parity"
    if process == "tail_required_model":
        return "C++ required tail rollup; cached-tail work is included here"
    if process in {"tail_image_encode_or_backbone", "tail_image_encode.graph_compute"}:
        return "primary root target: reduce image graph compute and required_session_e2e"
    if process == "tail_image_encode.non_compute_required":
        return "secondary target only if it moves required_session_e2e"
    if process.startswith("tail_image_encode.inline") or process.startswith(
        "tail_image_encode.preencode"
    ):
        return "placement diagnostic: do not accept unless the required tail encode total falls"
    if process in {"tail_propagate_or_tracker", "tail_propagate.graph_compute"}:
        return "optimize only after image encode, unless this row becomes the largest gap"
    if process.startswith("tail_propagate."):
        return "propagation detail; show a parent tail_propagate and required_session_e2e win"
    if process == "cpp_required_core_graph_compute":
        return "optimization subtotal: frame0 encode + tail encode + propagate graph compute"
    if process == "cpp_required_non_core":
        return "non-core subtotal: optimize only if no longer dominated by graph compute"
    if process == "cpp_required_remainder":
        return "accounting check; should stay near zero"
    return required_work_gate(process, role)


def e2e_process_measurement_rows(
    process_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in process_targets:
        process = str(row.get("process") or "")
        if process not in E2E_PROCESS_MAP_ORDER:
            continue
        cpp_ms = number(row.get("cpp_ms"))
        python_ms = number(row.get("python_ms"))
        rows.append(
            {
                "process": process,
                "bucket": row.get("bucket"),
                "role": row.get("role"),
                "cpp_ms": cpp_ms,
                "python_ms": python_ms,
                "python_diagnostic_ms": number(row.get("python_diagnostic_ms")),
                "python_over_cpp_ratio": row.get("python_over_cpp_ratio"),
                "cpp_minus_python_ms": cpp_ms - python_ms
                if cpp_ms is not None and python_ms is not None
                else None,
                "cpp_in_required_e2e": row.get("cpp_in_required_e2e"),
                "python_in_session_e2e": row.get("python_in_session_e2e"),
                "exact_family_match": row.get("exact_family_match"),
                "comparable_session_e2e_claim": row.get(
                    "comparable_session_e2e_claim"
                ),
                "same_contract_session_e2e_diagnostic": row.get(
                    "same_contract_session_e2e_diagnostic"
                ),
                "basis": row.get("basis"),
                "gate": e2e_process_measurement_gate(process, row.get("role")),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            E2E_PROCESS_MAP_ORDER.get(str(row.get("process") or ""), 999),
            str(row.get("bucket") or ""),
            str(row.get("role") or ""),
        ),
    )


def tail_group_isolated_stage_names(group: str) -> tuple[str, ...]:
    if group == "image.vit.prefix":
        return ()
    if group == "image.vit.mlp_matmul":
        return ("mlp", "mlp_fc1", "mlp_fc2", "mlp_fc1_gelu")
    if group == "image.vit.qkv_matmul":
        return ("qkv_proj",)
    if group == "image.vit.attention":
        return ("attn_core",)
    if group == "image.vit.proj_matmul":
        return ("attn_proj",)
    if group == "image.vit.mlp_gelu":
        return ("mlp_gelu", "mlp_fc1_gelu")
    if group == "image.vit.norm_residual_copy":
        return ("norm1", "norm2")
    if group == "image.vit.layout_rope_copy":
        return (
            "window_part",
            "qkv_layout",
            "qkv_rope",
            "window_unpart",
        )
    if group == "image.neck":
        return ()
    return ()


def isolated_stage_hints_for_group(
    group: str, isolated_stages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    stage_names = set(tail_group_isolated_stage_names(group))
    if not stage_names:
        return []
    hints: list[dict[str, Any]] = []
    for row in isolated_stages:
        stage_name = row.get("stage_name")
        if stage_name not in stage_names:
            continue
        local_basis = row.get("local_estimate_basis")
        local_estimate = row.get("local_estimate_ms")
        hint_ms = (
            local_estimate
            if local_basis in {"qkv_rope - qkv_layout", "attn_core - qkv_rope"}
            else row.get("effective_stage_ms", row.get("mean_ms"))
        )
        hints.append(
            {
                "bench": row.get("bench"),
                "block": row.get("block"),
                "stage_name": stage_name,
                "mean_ms": row.get("mean_ms"),
                "local_estimate_ms": row.get("local_estimate_ms"),
                "measurement_kind": row.get("measurement_kind"),
                "hint_ms": hint_ms,
                "source": row.get("source"),
            }
        )
    return sorted(
        hints,
        key=lambda row: -float(row.get("hint_ms") or 0.0),
    )


def required_work_measurement_rows(
    required_process_targets: list[dict[str, Any]],
    required_image_graph_group_budget: list[dict[str, Any]],
    tail_graph_group_budget: list[dict[str, Any]],
    isolated_stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    important_detail_processes = {
        "input.prepare.decode",
        "input.prepare.other",
        "frame0.image_encode.graph_compute",
        "frame0.image_encode.non_compute",
        "tail.image_encode.graph_compute",
        "tail.image_encode.non_compute",
        "tail.propagate.graph_compute",
        "tail.propagate.cache_compute",
        "tail.propagate.input_upload",
        "tail.propagate.output_read",
        "tail.propagate.remainder",
    }
    rows: list[dict[str, Any]] = []
    for row in required_process_targets:
        process = str(row.get("process") or "")
        table = str(row.get("table") or "")
        role = str(row.get("role") or "")
        include = table == "rollup" or process in important_detail_processes
        if not include:
            continue
        rows.append(
            {
                "scope": "formal_required_e2e" if table == "rollup" else "required_detail",
                "process": process,
                "parent_process": row.get("parent_process"),
                "role": role,
                "measurement_class": row.get("measurement_class"),
                "included_in_required_e2e": row.get("included_in_required_e2e"),
                "additive_required_work": row.get("additive_required_work"),
                "placement_only": row.get("placement_only"),
                "profile_only": False,
                "cpp_ms": row.get("cpp_ms"),
                "required_e2e_pct": row.get("required_e2e_pct"),
                "source": row.get("source_component") or row.get("basis"),
                "measurement": "component_split",
                "gate": required_work_gate(process, role),
            }
        )

    for group in required_image_graph_group_budget:
        group_name = str(group.get("group") or "")
        rows.append(
            {
                "scope": "required_image_graph_budget",
                "process": f"required.image_encode.graph_compute/{group_name}",
                "parent_process": "required.image_encode.graph_compute",
                "role": "implementation_budget",
                "measurement_class": "implementation_budget",
                "included_in_required_e2e": False,
                "additive_required_work": False,
                "placement_only": False,
                "profile_only": True,
                "cpp_ms": group.get("required_budget_ms"),
                "required_e2e_pct": group.get("required_e2e_pct"),
                "source": (
                    "frame0 and tail GGML_CUDA_PROFILE_NODES normalized to their "
                    "required graph_compute_ms"
                ),
                "measurement": "node_profile_budget",
                "gate": (
                    "pre-gate in isolated ViT/stage bench, then require lower "
                    "frame0/tail image-encode graph_compute_ms and required_e2e_ms"
                ),
                "profile_pct": group.get("profile_pct"),
                "stages": group.get("stages"),
                "modes": group.get("modes"),
                "isolated_stage_hints": isolated_stage_hints_for_group(
                    group_name, isolated_stages
                ),
            }
        )

    for group in tail_graph_group_budget:
        group_name = str(group.get("group") or "")
        rows.append(
            {
                "scope": "tail_graph_budget",
                "process": f"tail.image_encode.graph_compute/{group_name}",
                "parent_process": "tail.image_encode.graph_compute",
                "role": "implementation_budget",
                "measurement_class": "implementation_budget",
                "included_in_required_e2e": False,
                "additive_required_work": False,
                "placement_only": False,
                "profile_only": True,
                "cpp_ms": group.get("required_budget_ms"),
                "required_e2e_pct": group.get("required_e2e_pct"),
                "source": "GGML_CUDA_PROFILE_NODES normalized to tail_encode_graph_compute_ms",
                "measurement": "node_profile_budget",
                "gate": (
                    "pre-gate in isolated ViT/stage bench, then require lower "
                    "tail_encode_graph_compute_ms and required_e2e_ms"
                ),
                "profile_pct": group.get("profile_pct"),
                "stages": group.get("stages"),
                "isolated_stage_hints": isolated_stage_hints_for_group(
                    group_name, isolated_stages
                ),
            }
        )

    scope_order = {
        "formal_required_e2e": 0,
        "required_detail": 1,
        "required_image_graph_budget": 2,
        "tail_graph_budget": 3,
    }
    role_order = {
        "denominator": 0,
        "exclusive_required": 1,
        "exclusive_detail": 2,
        "implementation_budget": 3,
        "placement_detail": 4,
    }
    return sorted(
        rows,
        key=lambda row: (
            scope_order.get(str(row.get("scope")), 9),
            role_order.get(str(row.get("role")), 9),
            -float(row.get("cpp_ms") or 0.0),
            str(row.get("process") or ""),
        ),
    )


def focus_rows(
    processes: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    tail_graph_budget: list[dict[str, Any]],
    tail_graph_group_budget: list[dict[str, Any]],
    stage_profile_stages: list[dict[str, Any]],
    isolated_stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    focus: list[dict[str, Any]] = []
    for row in processes:
        if row.get("cpp_slower_ms") is None:
            continue
        focus.append(
            {
                "target": row.get("process"),
                "reason": "C++ is slower than the Python diagnostic/formal process",
                "cpp_ms": row.get("cpp_ms"),
                "python_ms": row.get("python_ms"),
                "gap_ms": row.get("cpp_slower_ms"),
                "python_over_cpp_ratio": row.get("python_over_cpp_ratio"),
            }
        )
    for stage in stages[:5]:
        focus.append(
            {
                "target": stage.get("stage"),
                "reason": "Top CUDA-node stage inside the profiled tail image encoder",
                "profile_sum_ms": stage.get("sum_ms"),
                "profile_pct": stage.get("profile_pct"),
            }
        )
    for stage in tail_graph_budget[:5]:
        focus.append(
            {
                "target": stage.get("stage"),
                "reason": "Top required-E2E tail graph budget after normalizing the node profile",
                "required_budget_ms": stage.get("required_budget_ms"),
                "required_e2e_pct": stage.get("required_e2e_pct"),
                "profile_pct": stage.get("profile_pct"),
            }
        )
    for group in tail_graph_group_budget[:5]:
        focus.append(
            {
                "target": group.get("group"),
                "reason": "Top required-E2E tail graph budget group for implementation-level work",
                "required_budget_ms": group.get("required_budget_ms"),
                "required_e2e_pct": group.get("required_e2e_pct"),
                "profile_pct": group.get("profile_pct"),
                "stages": group.get("stages"),
            }
        )
    tracking_stage_count = 0
    for stage in stage_profile_stages:
        if stage.get("mode") != "tracking":
            continue
        focus.append(
            {
                "target": f"tracking.{stage.get('stage')}",
                "reason": "Top synchronized CUDA stage inside required tail image encode",
                "profile_sum_ms": stage.get("sum_ms"),
                "profile_pct": stage.get("profile_pct"),
                "top_ops": stage.get("top_ops"),
            }
        )
        tracking_stage_count += 1
        if tracking_stage_count >= 5:
            break
    isolated_stage_count = 0
    for stage in isolated_stages:
        if not stage.get("optimization_target"):
            continue
        focus.append(
            {
                "target": f"{stage.get('bench')}.{stage.get('stage_name')}",
                "reason": "Top isolated ViT block-local stage captured from the required E2E tensor",
                "block": stage.get("block"),
                "mean_ms": stage.get("mean_ms"),
                "ci95_ms": stage.get("ci95_ms"),
                "local_estimate_ms": stage.get("local_estimate_ms"),
                "input_ne": stage.get("input_ne"),
                "output_ne": stage.get("output_ne"),
            }
        )
        isolated_stage_count += 1
        if isolated_stage_count >= 5:
            break
    return focus


def fmt_ms(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def fmt_pct(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f}%"


def fmt_ratio(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def fmt_bool(value: Any) -> str:
    if value is None:
        return "-"
    return "yes" if bool(value) else "no"


def fmt_top_ops(value: Any) -> str:
    if not isinstance(value, list):
        return "-"
    parts: list[str] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        op = item.get("op")
        sum_ms = number(item.get("sum_ms"))
        if op is None or sum_ms is None:
            continue
        parts.append(f"{op}:{sum_ms:.3f}")
    return ", ".join(parts) if parts else "-"


def fmt_isolated_hints(value: Any) -> str:
    if not isinstance(value, list):
        return "-"
    parts: list[str] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        stage_name = item.get("stage_name")
        mean_ms = number(item.get("hint_ms"))
        if mean_ms is None:
            mean_ms = number(item.get("mean_ms"))
        if stage_name is None or mean_ms is None:
            continue
        block = item.get("block")
        block_prefix = (
            f"b{int(block)}:"
            if isinstance(block, int | float)
            else ""
        )
        parts.append(f"{block_prefix}{stage_name}:{mean_ms:.3f}")
    return ", ".join(parts) if parts else "-"


def stage_profile_markdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for mode in ("tracking", "frame0"):
        selected.extend(row for row in rows if row.get("mode") == mode)
        selected = selected[:12] if mode == "tracking" else selected[:24]
    if len(selected) >= 24:
        return selected
    selected_ids = {(row.get("mode"), row.get("stage")) for row in selected}
    for row in rows:
        row_id = (row.get("mode"), row.get("stage"))
        if row_id in selected_ids:
            continue
        selected.append(row)
        if len(selected) >= 24:
            break
    return selected


def markdown_table(result: dict[str, Any]) -> str:
    lines = ["# Required E2E Optimization Targets", ""]
    summary = result["summary"]
    lines.extend(
        [
            f"- Model: `{summary.get('model')}`",
            f"- Variant: `{summary.get('cpp_variant')}`",
            f"- Python dtype / TF32: `{summary.get('python_dtype') or '-'}` / "
            f"`{summary.get('tf32_policy') or '-'}`",
            f"- Required E2E: `{fmt_ms(summary.get('required_e2e_ms'))} ms`",
            f"- Required core graph compute: `{fmt_ms(summary.get('required_core_compute_ms'))} ms` "
            f"({fmt_pct(summary.get('required_core_compute_pct'))})",
            "",
        ]
    )
    contract = result.get("measurement_contract")
    if isinstance(contract, dict) and contract:
        cached_tail = contract.get("cached_tail")
        core_split = contract.get("tail_encode_core_split")
        checks = contract.get("checks")
        python_contract = contract.get("python_comparison_contract")
        lines.extend(
            [
                "## Measurement Contract",
                "",
                "| item | value |",
                "| --- | --- |",
                f"| denominator | `{contract.get('denominator')}` |",
                f"| exclusive rollup delta | {fmt_ms(contract.get('exclusive_rollup_delta_ms'))} ms |",
                f"| cached tail enabled | `{cached_tail.get('enabled') if isinstance(cached_tail, dict) else '-'}` |",
                f"| cached work added back | `{cached_tail.get('work_is_added_back_to_required_e2e') if isinstance(cached_tail, dict) else '-'}` |",
                f"| Python cached-tail equivalent | `{python_contract.get('cached_tail_python_equivalent') if isinstance(python_contract, dict) else '-'}` |",
                f"| tail encode graph compute | {fmt_pct(core_split.get('graph_compute_pct_of_tail_encode') if isinstance(core_split, dict) else None)} |",
                f"| cached work hidden check | `{checks.get('cached_tail_not_hidden') if isinstance(checks, dict) else '-'}` |",
                "",
            ]
        )
    if result.get("required_work_measurements"):
        lines.extend(
            [
                "## Required E2E Work Measurement Gates",
                "",
                "These rows separate the work that is inside the required E2E denominator "
                "from diagnostic or placement-only measurements. Implementation-budget rows "
                "normalize the profiled tail image-encode graph to the same denominator.",
                "",
                "| scope | process | class | role | ms | required E2E % | measurement | gate | isolated hint |",
                "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in result["required_work_measurements"][:40]:
            gate = str(row.get("gate") or "-")
            if len(gate) > 96:
                gate = gate[:93] + "..."
            lines.append(
                f"| `{row.get('scope')}` | `{row.get('process')}` | "
                f"`{row.get('measurement_class', '-')}` | "
                f"`{row.get('role')}` | {fmt_ms(row.get('cpp_ms'))} | "
                f"{fmt_pct(row.get('required_e2e_pct'))} | "
                f"`{row.get('measurement')}` | {gate} | "
                f"`{fmt_isolated_hints(row.get('isolated_stage_hints'))}` |"
            )
        lines.append("")
    if result.get("e2e_process_measurements"):
        lines.extend(
            [
                "## Required E2E Process Map",
                "",
                "This table keeps formal required-E2E work separate from Python hook "
                "diagnostics. Python hook rows are useful for locating the gap, but "
                "candidate acceptance still requires a lower C++ required denominator.",
                "",
                "| process | role | C++ ms | C++ req | Python ms | Python sess | py/cpp | C++ - Python | Python diag | gate |",
                "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in result["e2e_process_measurements"][:28]:
            gate = str(row.get("gate") or "-")
            if len(gate) > 96:
                gate = gate[:93] + "..."
            lines.append(
                f"| `{row.get('process')}` | `{row.get('role')}` | "
                f"{fmt_ms(row.get('cpp_ms'))} | "
                f"{fmt_bool(row.get('cpp_in_required_e2e'))} | "
                f"{fmt_ms(row.get('python_ms'))} | "
                f"{fmt_bool(row.get('python_in_session_e2e'))} | "
                f"{fmt_ratio(row.get('python_over_cpp_ratio'))} | "
                f"{fmt_ms(row.get('cpp_minus_python_ms'))} | "
                f"{fmt_ms(row.get('python_diagnostic_ms'))} | {gate} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Process Gaps",
            "",
            "| process | cpp ms | python ms | py/cpp | cpp slower ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    process_gap_rows = [
        row for row in result["process_targets"] if row.get("python_ms") is not None
    ]
    if not process_gap_rows:
        process_gap_rows = result["process_targets"]
    for row in process_gap_rows[:12]:
        lines.append(
            f"| `{row.get('process')}` | {fmt_ms(row.get('cpp_ms'))} | "
            f"{fmt_ms(row.get('python_ms'))} | {fmt_ratio(row.get('python_over_cpp_ratio'))} | "
            f"{fmt_ms(row.get('cpp_slower_ms'))} |"
        )
    if result["acceptance_required_process_targets"]:
        lines.extend(
            [
                "",
                "## Acceptance Required Process Split",
                "",
                "These rows are additive required work or the formal denominator. "
                "They are the process-level rows that can be used for speed acceptance.",
                "",
                "| table | bucket | process | parent | class | cpp ms | required E2E % | role |",
                "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in result["acceptance_required_process_targets"][:24]:
            lines.append(
                f"| `{row.get('table')}` | `{row.get('bucket')}` | `{row.get('process')}` | "
                f"`{row.get('parent_process') or '-'}` | "
                f"`{row.get('measurement_class', '-')}` | {fmt_ms(row.get('cpp_ms'))} | "
                f"{fmt_pct(row.get('required_e2e_pct'))} | `{row.get('role')}` |"
            )
    if result.get("placement_process_targets"):
        lines.extend(
            [
                "",
                "## Required Placement Details",
                "",
                "These rows explain where cached-tail work was executed. They are not "
                "additional work to add on top of `tail.image_encode`.",
                "",
                "| bucket | process | parent | class | cpp ms | required E2E % |",
                "| --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in result["placement_process_targets"][:16]:
            lines.append(
                f"| `{row.get('bucket')}` | `{row.get('process')}` | "
                f"`{row.get('parent_process') or '-'}` | "
                f"`{row.get('measurement_class', '-')}` | {fmt_ms(row.get('cpp_ms'))} | "
                f"{fmt_pct(row.get('required_e2e_pct'))} |"
            )
    if result.get("python_quality_parity"):
        lines.extend(
            [
                "",
                "## Official Python Quality Parity",
                "",
                "| comparison | status | rows | diff rows | tolerance diff rows | min bbox IoU | max bbox delta | max score delta | min mask IoU | max mask xor | mask hash equal rows |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, parity in result["python_quality_parity"].items():
            if not isinstance(parity, dict):
                continue
            lhs_rows = parity.get("lhs_rows")
            rhs_rows = parity.get("rhs_rows")
            diff_rows = parity.get("diff_rows")
            tolerance_diff_rows = parity.get("tolerance_diff_rows")
            max_mask_pixel_xor = parity.get("max_mask_pixel_xor")
            mask_hash_equal_rows = parity.get("mask_hash_equal_rows")
            lines.append(
                f"| `{name}` | `{parity.get('status')}` | "
                f"{lhs_rows if lhs_rows is not None else '-'} / "
                f"{rhs_rows if rhs_rows is not None else '-'} | "
                f"{diff_rows if diff_rows is not None else '-'} | "
                f"{tolerance_diff_rows if tolerance_diff_rows is not None else '-'} | "
                f"{fmt_ratio(parity.get('min_bbox_iou'))} | "
                f"{fmt_ms(parity.get('max_bbox_delta_px'))} | "
                f"{fmt_ratio(parity.get('max_score_abs_delta'))} | "
                f"{fmt_ratio(parity.get('min_mask_pixel_iou'))} | "
                f"{max_mask_pixel_xor if max_mask_pixel_xor is not None else '-'} | "
                f"{mask_hash_equal_rows if mask_hash_equal_rows is not None else '-'} |"
            )
    lines.extend(
        [
            "",
            "## Required Components",
            "",
            "| component | mean ms | required E2E % | sd | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["component_targets"][:16]:
        lines.append(
            f"| `{row['component']}` | {fmt_ms(row.get('mean_ms'))} | "
            f"{fmt_pct(row.get('required_e2e_pct'))} | {fmt_ms(row.get('sd_ms'))} | "
            f"{row.get('n') or '-'} |"
        )
    if result["node_stage_targets"]:
        lines.extend(
            [
                "",
                "## Tail Encode Node Stages",
                "",
                "| stage | calls | profile sum ms | profile % | mean ms |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["node_stage_targets"][:16]:
            lines.append(
                f"| `{row['stage']}` | {row.get('count', '-')} | {fmt_ms(row.get('sum_ms'))} | "
                f"{fmt_pct(row.get('profile_pct'))} | {fmt_ms(row.get('mean_ms'))} |"
            )
    if result.get("fattn_profile_targets"):
        lines.extend(
            [
                "",
                "## FATTN Event Split",
                "",
                "These rows are synchronized CUDA event profiles for isolated attention "
                "stages. They explain where `attn_core` time goes, but normal speed "
                "acceptance still uses the required E2E denominator.",
                "",
                "| profile | group | stage | calls | total steady ms | kernel steady ms | QKV convert steady ms | V convert steady ms | convert % | tail budget | required image budget | ncols1 | schedule |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in result["fattn_profile_targets"]:
            lines.append(
                f"| `{row.get('profile')}` | `{row.get('group')}` | "
                f"`{row.get('stage') or '-'}` | {row.get('count', '-')} | "
                f"{fmt_ms(row.get('total_steady_mean_ms'))} | "
                f"{fmt_ms(row.get('kernel_steady_mean_ms'))} | "
                f"{fmt_ms(row.get('qkv_convert_steady_mean_ms'))} | "
                f"{fmt_ms(row.get('v_convert_steady_mean_ms'))} | "
                f"{fmt_pct(row.get('qkv_convert_pct_of_total'))} | "
                f"{fmt_ms(row.get('tail_budget_ms'))} | "
                f"{fmt_ms(row.get('required_image_budget_ms'))} | "
                f"`{fmt_counter_text(row.get('ncols1'))}` | "
                f"`{fmt_counter_text(row.get('schedule'))}` |"
            )
    if result.get("required_image_graph_group_budget"):
        lines.extend(
            [
                "",
                "## Required Image Encode Graph Budget Groups",
                "",
                "These rows combine `frame0.image_encode.graph_compute` and "
                "`tail.image_encode.graph_compute`. Use them to estimate the full "
                "E2E value of ViT/neck changes that affect every required image encode.",
                "",
                "| group | required budget ms | required E2E % | profile sum ms | calls | modes | stages |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in result["required_image_graph_group_budget"][:12]:
            stages = ", ".join(f"`{stage}`" for stage in row.get("stages") or [])
            modes = ", ".join(f"`{mode}`" for mode in row.get("modes") or [])
            lines.append(
                f"| `{row['group']}` | {fmt_ms(row.get('required_budget_ms'))} | "
                f"{fmt_pct(row.get('required_e2e_pct'))} | "
                f"{fmt_ms(row.get('profile_sum_ms'))} | {row.get('count', '-')} | "
                f"{modes} | {stages} |"
            )
    if result.get("required_image_graph_stage_budget"):
        lines.extend(
            [
                "",
                "## Required Image Encode Graph Budget",
                "",
                "These rows keep `frame0` and `tail` image-encode graph stages separate, "
                "then normalize each stage to its required graph-compute wall time. Use "
                "this for stage-specific E2E projection when a candidate targets one "
                "atomic ViT/neck stage.",
                "",
                "| mode | stage | required budget ms | required E2E % | profile share | profile sum ms | calls |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["required_image_graph_stage_budget"][:20]:
            lines.append(
                f"| `{row.get('mode')}` | `{row.get('stage')}` | "
                f"{fmt_ms(row.get('required_budget_ms'))} | "
                f"{fmt_pct(row.get('required_e2e_pct'))} | "
                f"{fmt_pct(row.get('profile_pct'))} | "
                f"{fmt_ms(row.get('profile_sum_ms'))} | {row.get('count', '-')} |"
            )
    if result.get("tail_graph_group_budget"):
        lines.extend(
            [
                "",
                "## Tail Required Graph Budget Groups",
                "",
                "These rows group the normalized tail graph budget into implementation-level "
                "targets. A speed candidate should move one of these groups and then reduce "
                "the normal required E2E denominator.",
                "",
                "| group | required budget ms | required E2E % | profile share | profile sum ms | calls | stages |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in result["tail_graph_group_budget"][:12]:
            stages = ", ".join(f"`{stage}`" for stage in row.get("stages") or [])
            lines.append(
                f"| `{row['group']}` | {fmt_ms(row.get('required_budget_ms'))} | "
                f"{fmt_pct(row.get('required_e2e_pct'))} | "
                f"{fmt_pct(row.get('profile_pct'))} | "
                f"{fmt_ms(row.get('profile_sum_ms'))} | {row.get('count', '-')} | "
                f"{stages} |"
            )
    if result.get("tail_graph_stage_budget"):
        lines.extend(
            [
                "",
                "## Tail Required Graph Budget",
                "",
                "This normalizes synchronized node-profile shares to the normal required "
                "`tail_encode_graph_compute_ms`; use it for E2E optimization budget ranking, "
                "not as an additional wall-clock sum.",
                "",
                "| stage | required budget ms | required E2E % | profile share | profile sum ms | calls |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["tail_graph_stage_budget"][:16]:
            lines.append(
                f"| `{row['stage']}` | {fmt_ms(row.get('required_budget_ms'))} | "
                f"{fmt_pct(row.get('required_e2e_pct'))} | "
                f"{fmt_pct(row.get('profile_pct'))} | "
                f"{fmt_ms(row.get('profile_sum_ms'))} | {row.get('count', '-')} |"
            )
    if result.get("stage_profile_targets"):
        lines.extend(
            [
                "",
                "## Required Encode CUDA Stage Split",
                "",
                "| mode | stage | calls | profile sum ms | profile % | mean ms | top ops |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in stage_profile_markdown_rows(result["stage_profile_targets"]):
            lines.append(
                f"| `{row.get('mode')}` | `{row.get('stage')}` | "
                f"{row.get('count', '-')} | {fmt_ms(row.get('sum_ms'))} | "
                f"{fmt_pct(row.get('profile_pct'))} | {fmt_ms(row.get('mean_ms'))} | "
                f"`{fmt_top_ops(row.get('top_ops'))}` |"
            )
    if result["cublaslt_targets"]:
        lines.extend(
            [
                "",
                "## cuBLASLt Bias GEMM Groups",
                "",
                "| group | calls | matmul sum ms | mean ms | selected algo | dst |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in result["cublaslt_targets"][:12]:
            lines.append(
                f"| `{row['group']}` | {row.get('n', '-')} | {fmt_ms(row.get('matmul_sum_ms'))} | "
                f"{fmt_ms(row.get('matmul_mean_ms'))} | `{row.get('selected_algo_indices')}` | "
                f"`{row.get('dst_types')}` |"
            )
    if result.get("isolated_benches"):
        lines.extend(
            [
                "",
                "## Isolated Required-Process Benches",
                "",
                "| name | batch | tracker neck | n | mean/frame ms | sd | ci95 | delta vs baseline | signal | median/frame ms | min/frame ms | max/frame ms | lane max diff |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["isolated_benches"]:
            lines.append(
                f"| `{row['name']}` | {row.get('batch_size') or '-'} | "
                f"`{row.get('include_tracker_neck')}` | "
                f"{row.get('sample_n') if row.get('sample_n') is not None else '-'} | "
                f"{fmt_ms(row.get('mean_ms_per_frame'))} | "
                f"{fmt_ms(row.get('sample_sd_ms_per_frame'))} | "
                f"{fmt_ms(row.get('sample_ci95_half_width_ms_per_frame'))} | "
                f"{fmt_ms(row.get('mean_ms_per_frame_delta_vs_baseline'))} | "
                f"{fmt_ratio(row.get('delta_vs_baseline_se_signal'))} | "
                f"{fmt_ms(row.get('median_ms_per_frame'))} | "
                f"{fmt_ms(row.get('min_ms_per_frame'))} | "
                f"{fmt_ms(row.get('max_ms_per_frame'))} | "
                f"{fmt_ms(row.get('lane_max_abs_diff'))} |"
            )
    if result.get("isolated_stage_targets"):
        lines.extend(
            [
                "",
                "## Isolated ViT Stage Split",
                "",
                "| bench | block | stage | kind | effective ms | mean ms | req image budget | tail budget | budget basis | local basis | input | output |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for row in result["isolated_stage_targets"][:28]:
            budget_basis = row.get("required_image_budget_basis") or row.get("tail_budget_basis")
            lines.append(
                f"| `{row.get('bench')}` | {row.get('block', '-')} | "
                f"`{row.get('stage_name')}` | `{row.get('measurement_kind')}` | "
                f"{fmt_ms(row.get('effective_stage_ms'))} | "
                f"{fmt_ms(row.get('mean_ms'))} | "
                f"{fmt_ms(row.get('required_image_budget_ms'))} | "
                f"{fmt_ms(row.get('tail_budget_ms'))} | "
                f"`{budget_basis or '-'}` | "
                f"`{row.get('local_estimate_basis') or '-'}` | "
                f"`{row.get('input_ne') or '-'}` | `{row.get('output_ne') or '-'}` |"
            )
    if result.get("candidate_decisions"):
        lines.extend(
            [
                "",
                "## Candidate Decisions",
                "",
                "| candidate | required E2E delta | tail graph delta | core compute delta | e2e z | tail z | parity | diff rows | decision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
            ]
        )
        for row in result["candidate_decisions"]:
            parity = row.get("parity")
            diff_rows = parity.get("diff_rows") if isinstance(parity, dict) else None
            lines.append(
                f"| `{row['candidate']}` | {fmt_ms(row.get('required_e2e_delta_ms'))} | "
                f"{fmt_ms(row.get('tail_encode_graph_compute_delta_ms'))} | "
                f"{fmt_ms(row.get('required_core_compute_delta_ms'))} | "
                f"{fmt_ratio(row.get('required_e2e_signal'))} | "
                f"{fmt_ratio(row.get('tail_encode_graph_compute_signal'))} | "
                f"`{row.get('parity_status')}` | "
                f"{diff_rows if diff_rows is not None else '-'} | "
                f"`{row.get('decision')}` |"
            )
    if result.get("candidate_contract_compatibility"):
        lines.extend(
            [
                "",
                "## Candidate Contract Compatibility",
                "",
                "| candidate | compatible | mismatches |",
                "| --- | --- | --- |",
            ]
        )
        for row in result["candidate_contract_compatibility"]:
            mismatches = row.get("mismatches")
            mismatch_keys = (
                ", ".join(sorted(str(key) for key in mismatches))
                if isinstance(mismatches, dict) and mismatches
                else "-"
            )
            lines.append(
                f"| `{row.get('candidate')}` | `{row.get('compatible')}` | `{mismatch_keys}` |"
            )
    if result.get("candidate_required_process_deltas"):
        lines.extend(
            [
                "",
                "## Candidate Required Process Deltas",
                "",
                "Only acceptance-target required work is listed here; cached-tail placement "
                "details are reported separately.",
                "",
                "| candidate | table | bucket | process | parent | class | baseline ms | candidate ms | delta ms | delta % |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["candidate_required_process_deltas"][:24]:
            lines.append(
                f"| `{row['candidate']}` | `{row.get('table')}` | `{row.get('bucket')}` | "
                f"`{row['process']}` | `{row.get('parent_process') or '-'}` | "
                f"`{row.get('measurement_class', '-')}` | "
                f"{fmt_ms(row.get('baseline_cpp_ms'))} | "
                f"{fmt_ms(row.get('candidate_cpp_ms'))} | "
                f"{fmt_ms(row.get('cpp_delta_ms'))} | "
                f"{fmt_pct(row.get('cpp_delta_pct'))} |"
            )
    if result.get("candidate_placement_process_deltas"):
        lines.extend(
            [
                "",
                "## Candidate Placement Deltas",
                "",
                "Placement rows explain where required tail image encode ran; they are "
                "diagnostic only and cannot accept a speed candidate by themselves.",
                "",
                "| candidate | process | class | baseline ms | candidate ms | delta ms |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in result["candidate_placement_process_deltas"][:16]:
            lines.append(
                f"| `{row['candidate']}` | `{row['process']}` | "
                f"`{row.get('measurement_class', '-')}` | "
                f"{fmt_ms(row.get('baseline_cpp_ms'))} | "
                f"{fmt_ms(row.get('candidate_cpp_ms'))} | "
                f"{fmt_ms(row.get('cpp_delta_ms'))} |"
            )
    if result.get("candidate_component_deltas"):
        lines.extend(
            [
                "",
                "## Candidate Component Deltas",
                "",
                "| candidate | component | baseline ms | candidate ms | delta ms | delta % | baseline sd | candidate sd | n |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["candidate_component_deltas"]:
            lines.append(
                f"| `{row['candidate']}` | `{row['component']}` | "
                f"{fmt_ms(row.get('baseline_ms'))} | {fmt_ms(row.get('candidate_ms'))} | "
                f"{fmt_ms(row.get('delta_ms'))} | {fmt_pct(row.get('delta_pct'))} | "
                f"{fmt_ms(row.get('baseline_sd_ms'))} | "
                f"{fmt_ms(row.get('candidate_sd_ms'))} | "
                f"{row.get('candidate_n') or '-'} |"
            )
    if result.get("candidate_process_deltas"):
        lines.extend(
            [
                "",
                "## Candidate Process Deltas",
                "",
                "| candidate | bucket | process | baseline cpp ms | candidate cpp ms | delta ms | delta % | python ms |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["candidate_process_deltas"]:
            lines.append(
                f"| `{row['candidate']}` | `{row.get('bucket')}` | `{row['process']}` | "
                f"{fmt_ms(row.get('baseline_cpp_ms'))} | "
                f"{fmt_ms(row.get('candidate_cpp_ms'))} | "
                f"{fmt_ms(row.get('cpp_delta_ms'))} | "
                f"{fmt_pct(row.get('cpp_delta_pct'))} | "
                f"{fmt_ms(row.get('candidate_python_ms'))} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Optimize only changes that reduce `tail_image_encode.graph_compute`, the normalized "
            "tail required graph budget, or the top tail encode node stages under the same "
            "required-E2E split. Cached-tail placement, graph admin, and input timing are "
            "accounting context unless the required E2E denominator also improves.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-split", type=Path, required=True)
    parser.add_argument("--node-names", type=Path)
    parser.add_argument("--node-label", default="sam3_encode#02")
    parser.add_argument("--stage-profile", type=Path)
    parser.add_argument("--cublaslt", type=Path)
    parser.add_argument(
        "--candidate-component-split",
        action="append",
        default=[],
        type=Path,
        help="Candidate component_split.json to compare against --component-split.",
    )
    parser.add_argument(
        "--candidate-parity",
        action="append",
        default=[],
        help=(
            "Candidate JSONL/mask compare result as NAME=PATH. NAME must match "
            "the candidate component split label."
        ),
    )
    parser.add_argument(
        "--python-parity",
        action="append",
        default=[],
        help=(
            "Official Python vs C++ quality parity result as NAME=PATH. This is "
            "reported separately from exact C++ baseline/candidate parity."
        ),
    )
    parser.add_argument(
        "--isolated-bench",
        action="append",
        default=[],
        help="Named isolated required-process bench JSON as NAME=PATH.",
    )
    parser.add_argument(
        "--fattn-profile",
        action="append",
        default=[],
        help="Named summarize_fattn_profile.py JSON as NAME=PATH.",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    component_split = load_json(args.component_split)
    row = first_row(component_split)
    candidate_splits = [
        (path, load_json(path)) for path in args.candidate_component_split
    ]
    candidate_rows = [(path, first_row(split)) for path, split in candidate_splits]
    candidate_parity_paths = {
        name: path
        for name, path in (
            parse_named_path(value, option_name="--candidate-parity")
            for value in args.candidate_parity
        )
    }
    candidate_parity = {
        name: load_json(path) for name, path in candidate_parity_paths.items()
    }
    python_parity_paths = {
        name: path
        for name, path in (
            parse_named_path(value, option_name="--python-parity")
            for value in args.python_parity
        )
    }
    python_parity = {
        name: load_json(path) for name, path in python_parity_paths.items()
    }
    isolated_bench_paths = {
        name: path
        for name, path in (
            parse_named_path(value, option_name="--isolated-bench")
            for value in args.isolated_bench
        )
    }
    isolated_benches = {
        name: load_json(path) for name, path in isolated_bench_paths.items()
    }
    fattn_profile_paths = {
        name: path
        for name, path in (
            parse_named_path(value, option_name="--fattn-profile")
            for value in args.fattn_profile
        )
    }
    fattn_profiles = {
        name: load_json(path) for name, path in fattn_profile_paths.items()
    }
    components = row.get("components", {})
    if not isinstance(components, dict):
        components = {}
    required_e2e = metric_mean(components.get("required_e2e_ms"))
    required_core = metric_mean(components.get("required_core_compute_ms"))
    processes = process_rows(component_split)
    node_names = load_json(args.node_names) if args.node_names else {}
    stage_profile = load_json(args.stage_profile) if args.stage_profile else {}
    cublaslt = load_json(args.cublaslt) if args.cublaslt else {}
    stages = node_stage_rows(node_names, args.node_label)
    tail_graph_compute_ms = metric_mean(components.get("tail_encode_graph_compute_ms"))
    tail_graph_budget = tail_graph_stage_budget_rows(
        stages,
        required_e2e_ms=required_e2e,
        tail_graph_compute_ms=tail_graph_compute_ms,
    )
    tail_graph_group_budget = tail_graph_budget_group_rows(
        tail_graph_budget, required_e2e_ms=required_e2e
    )
    required_image_graph_stage_budget = required_image_graph_stage_budget_rows(
        node_names,
        node_label=args.node_label,
        required_e2e_ms=required_e2e,
        frame0_graph_compute_ms=metric_mean(components.get("frame0_encode_graph_compute_ms")),
        tail_graph_compute_ms=tail_graph_compute_ms,
    )
    required_image_graph_group_budget = required_image_graph_group_budget_rows(
        required_image_graph_stage_budget, required_e2e_ms=required_e2e
    )
    stage_profile_stages = stage_profile_rows(stage_profile)
    cublaslt_targets = cublaslt_rows(cublaslt)
    fattn_profile_targets = fattn_profile_rows(
        fattn_profiles,
        tail_graph_stage_budget=tail_graph_budget,
        required_image_graph_stage_budget=required_image_graph_stage_budget,
    )
    isolated_stage_targets = isolated_stage_rows(
        isolated_benches,
        tail_graph_stage_budget=tail_graph_budget,
        tail_graph_group_budget=tail_graph_group_budget,
        required_image_graph_stage_budget=required_image_graph_stage_budget,
        required_image_graph_group_budget=required_image_graph_group_budget,
    )
    required_process_targets = required_process_rows(component_split)
    e2e_process_measurements = e2e_process_measurement_rows(processes)
    acceptance_required_process_targets = acceptance_required_process_rows(
        required_process_targets
    )
    placement_process_targets = placement_process_rows(required_process_targets)
    required_work_measurements = required_work_measurement_rows(
        required_process_targets,
        required_image_graph_group_budget,
        tail_graph_group_budget,
        isolated_stage_targets,
    )
    candidate_deltas = component_delta_rows(row, candidate_rows)
    candidate_process_deltas = process_delta_rows(component_split, candidate_splits)
    candidate_required_process_deltas = required_process_delta_rows(
        component_split, candidate_splits
    )
    candidate_placement_process_deltas = [
        row
        for row in required_process_delta_rows(
            component_split, candidate_splits, acceptance_only=False
        )
        if row.get("role") == "placement_detail"
    ]
    candidate_contract_compatibility = contract_compatibility_rows(
        component_split, candidate_splits
    )
    result = {
        "inputs": {
            "component_split": str(args.component_split),
            "node_names": str(args.node_names) if args.node_names else None,
            "node_label": args.node_label,
            "stage_profile": str(args.stage_profile) if args.stage_profile else None,
            "cublaslt": str(args.cublaslt) if args.cublaslt else None,
            "candidate_component_splits": [
                str(path) for path in args.candidate_component_split
            ],
            "candidate_parity": {
                name: str(path) for name, path in candidate_parity_paths.items()
            },
            "python_parity": {
                name: str(path) for name, path in python_parity_paths.items()
            },
            "isolated_benches": {
                name: str(path) for name, path in isolated_bench_paths.items()
            },
            "fattn_profiles": {
                name: str(path) for name, path in fattn_profile_paths.items()
            },
        },
        "summary": {
            "model": row.get("model"),
            "precision": row.get("precision"),
            "backend": row.get("backend"),
            "cpp_variant": row.get("cpp_variant"),
            "python_dtype": row.get("python_dtype"),
            "tf32_policy": row.get("tf32_policy"),
            "required_e2e_ms": required_e2e,
            "required_core_compute_ms": required_core,
            "required_core_compute_pct": pct(required_core, required_e2e),
        },
        "measurement_contract": first_contract(component_split),
        "contract_signature": contract_signature(component_split),
        "focus": focus_rows(
            processes,
            stages,
            tail_graph_budget,
            tail_graph_group_budget,
            stage_profile_stages,
            isolated_stage_targets,
        ),
        "process_targets": processes,
        "e2e_process_measurements": e2e_process_measurements,
        "required_process_targets": required_process_targets,
        "acceptance_required_process_targets": acceptance_required_process_targets,
        "placement_process_targets": placement_process_targets,
        "required_work_measurements": required_work_measurements,
        "component_targets": component_rows(row),
        "node_stage_targets": stages,
        "tail_graph_group_budget": tail_graph_group_budget,
        "tail_graph_stage_budget": tail_graph_budget,
        "required_image_graph_stage_budget": required_image_graph_stage_budget,
        "required_image_graph_group_budget": required_image_graph_group_budget,
        "stage_profile_targets": stage_profile_stages,
        "cublaslt_targets": cublaslt_targets,
        "fattn_profile_targets": fattn_profile_targets,
        "isolated_benches": isolated_bench_rows(isolated_benches),
        "isolated_stage_targets": isolated_stage_targets,
        "candidate_parity": {
            name: parity_summary(parity) for name, parity in candidate_parity.items()
        },
        "python_quality_parity": {
            name: python_quality_parity_summary(parity)
            for name, parity in python_parity.items()
        },
        "candidate_decisions": candidate_decision_rows(
            candidate_deltas, candidate_parity
        ),
        "candidate_component_deltas": candidate_deltas,
        "candidate_process_deltas": candidate_process_deltas,
        "candidate_required_process_deltas": candidate_required_process_deltas,
        "candidate_placement_process_deltas": candidate_placement_process_deltas,
        "candidate_contract_compatibility": candidate_contract_compatibility,
    }

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_table(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
