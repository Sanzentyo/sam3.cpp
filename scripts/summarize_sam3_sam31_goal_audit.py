#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Summarize current SAM3/SAM3.1 goal status from concrete local evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_env_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    return None


def read_text_tail(path: Path, max_lines: int = 20) -> str | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def criterion(name: str, status: str, evidence: Any) -> dict[str, Any]:
    return {"criterion": name, "status": status, "evidence": evidence}


def with_next_action(item: dict[str, Any], next_action: str) -> dict[str, Any]:
    return {**item, "next_action": next_action}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def run_values(row: dict[str, Any], key: str) -> list[float]:
    runs = row.get("runs")
    if not isinstance(runs, list):
        return []
    values: list[float] = []
    for run in runs:
        if isinstance(run, dict) and isinstance(run.get(key), int | float):
            values.append(float(run[key]))
    return values


def summarize_single_sam3_perf(perf_summary: Any) -> tuple[str, Any]:
    if not isinstance(perf_summary, dict):
        return (
            "missing",
            (
                "No current same-contract C++ vs official Python SAM3 performance result is "
                "attached to this audit. CUDA is available; rerun model-matrix for the target "
                "precision and TF32 policy."
            ),
        )

    rows = [
        row
        for row in perf_summary.get("comparable_speed_rows", [])
        if isinstance(row, dict) and row.get("family") == "sam3"
    ]
    if not rows:
        return (
            "missing",
            {
                "summary": "No SAM3 comparable_speed_rows were found in the supplied model-matrix summary.",
                "comparison_contract": perf_summary.get("comparison_contract"),
            },
        )

    row = max(
        rows,
        key=lambda item: float(item.get("python_over_cpp_track_ratio_if_comparable") or 0.0),
    )
    ratio = float(row.get("python_over_cpp_track_ratio_if_comparable") or 0.0)

    cpp_rows = perf_summary.get("cpp", [])
    python_rows = perf_summary.get("python", [])
    cpp_row = next(
        (
            item
            for item in cpp_rows
            if isinstance(item, dict) and item.get("model") == row.get("model")
        ),
        {},
    )
    python_row = python_rows[0] if python_rows and isinstance(python_rows[0], dict) else {}
    cpp_track_values = run_values(cpp_row, "track_ms") if isinstance(cpp_row, dict) else []
    python_track_values = (
        run_values(python_row, "track_ms") if isinstance(python_row, dict) else []
    )
    evidence = {
        "status": "cpp_faster" if ratio > 1.0 else "cpp_not_faster",
        "model": row.get("model"),
        "precision": row.get("precision"),
        "python_dtype": row.get("python_dtype"),
        "tf32_policy": row.get("tf32_policy"),
        "decoded_source_width": row.get("decoded_source_width"),
        "decoded_source_height": row.get("decoded_source_height"),
        "frames": row.get("frames"),
        "same_input_encode_size": row.get("same_input_encode_size"),
        "same_measured_tracking_scope": row.get("same_measured_tracking_scope"),
        "cpp_track_scope": row.get("cpp_track_scope"),
        "python_track_scope": row.get("python_track_scope"),
        "same_warmup_policy": row.get("same_warmup_policy"),
        "cpp_warmup_runs": row.get("cpp_warmup_runs"),
        "python_warmup_runs": row.get("python_warmup_runs"),
        "cpp_track_ms": row.get("cpp_track_ms"),
        "python_track_ms": row.get("python_track_ms"),
        "python_over_cpp_track_ratio": ratio,
        "speedup_percent": (ratio - 1.0) * 100.0,
        "cpp_track_ms_stats": {
            "n": len(cpp_track_values),
            "mean": mean(cpp_track_values),
            "sample_sd": sample_sd(cpp_track_values),
            "min": min(cpp_track_values, default=None),
            "max": max(cpp_track_values, default=None),
        },
        "python_track_ms_stats": {
            "n": len(python_track_values),
            "mean": mean(python_track_values),
            "sample_sd": sample_sd(python_track_values),
            "min": min(python_track_values, default=None),
            "max": max(python_track_values, default=None),
        },
        "comparison_contract": perf_summary.get("comparison_contract"),
    }
    return ("met" if ratio > 1.0 else "missing", evidence)


def process_row(process_targets: Any, process: str) -> dict[str, Any]:
    if not isinstance(process_targets, list):
        return {}
    for row in process_targets:
        if isinstance(row, dict) and row.get("process") == process:
            return row
    return {}


def summarize_single_sam3_e2e_audit(e2e_audit: Any) -> tuple[str, Any]:
    if not isinstance(e2e_audit, dict):
        return "missing", "No SAM3 required-E2E audit JSON was supplied."
    summary = e2e_audit.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    required_session = process_row(
        e2e_audit.get("process_targets"), "required_session_e2e"
    )
    model_e2e = process_row(e2e_audit.get("process_targets"), "model_e2e")
    python_quality = e2e_audit.get("python_quality_parity")
    quality_rows = (
        [
            row
            for row in python_quality.values()
            if isinstance(row, dict)
        ]
        if isinstance(python_quality, dict)
        else []
    )
    quality_pass = bool(quality_rows) and all(
        row.get("status") == "quality_pass" for row in quality_rows
    )
    required_ratio = (
        float(required_session["python_over_cpp_ratio"])
        if isinstance(required_session.get("python_over_cpp_ratio"), int | float)
        else None
    )
    model_ratio = (
        float(model_e2e["python_over_cpp_ratio"])
        if isinstance(model_e2e.get("python_over_cpp_ratio"), int | float)
        else None
    )
    evidence = {
        "status": (
            "cpp_faster_quality_pass"
            if required_ratio is not None
            and required_ratio > 1.0
            and model_ratio is not None
            and model_ratio > 1.0
            and quality_pass
            else "missing_required_e2e_win_or_quality"
        ),
        "model": summary.get("model"),
        "precision": summary.get("precision"),
        "backend": summary.get("backend"),
        "cpp_variant": summary.get("cpp_variant"),
        "python_dtype": summary.get("python_dtype"),
        "tf32_policy": summary.get("tf32_policy"),
        "required_e2e_ms": summary.get("required_e2e_ms"),
        "required_session_python_ms": required_session.get("python_ms"),
        "required_session_python_over_cpp_ratio": required_ratio,
        "model_e2e_ms": model_e2e.get("cpp_ms"),
        "model_e2e_python_ms": model_e2e.get("python_ms"),
        "model_e2e_python_over_cpp_ratio": model_ratio,
        "contract_signature": e2e_audit.get("contract_signature"),
        "measurement_contract": e2e_audit.get("measurement_contract"),
        "python_quality_parity": python_quality,
    }
    if required_ratio and required_ratio > 1.0 and model_ratio and model_ratio > 1.0 and quality_pass:
        return "met", evidence
    return "missing", evidence


def summarize_sam3_perf(
    perf_summaries: list[Any], e2e_audits: list[Any]
) -> tuple[str, Any]:
    e2e_rows = [
        summarize_single_sam3_e2e_audit(audit)
        for audit in e2e_audits
        if isinstance(audit, dict)
    ]
    if e2e_rows:
        statuses = [status for status, _evidence in e2e_rows]
        supplemental = [
            summarize_single_sam3_perf(summary)
            for summary in perf_summaries
            if isinstance(summary, dict)
        ]
        evidence = {
            "required_e2e_audits": [evidence for _status, evidence in e2e_rows],
            "supplemental_model_matrix": [
                evidence for _status, evidence in supplemental
            ],
            "met_count": sum(1 for status in statuses if status == "met"),
            "missing_count": sum(1 for status in statuses if status != "met"),
        }
        return ("met" if all(status == "met" for status in statuses) else "missing", evidence)

    rows = [
        summarize_single_sam3_perf(summary)
        for summary in perf_summaries
        if isinstance(summary, dict)
    ]
    if not rows:
        return summarize_single_sam3_perf(None)
    statuses = [status for status, _evidence in rows]
    evidence = {
        "summaries": [evidence for _status, evidence in rows],
        "met_count": sum(1 for status in statuses if status == "met"),
        "missing_count": sum(1 for status in statuses if status != "met"),
    }
    if all(status == "met" for status in statuses):
        return "met", evidence
    if any(status == "blocked" for status in statuses):
        return "blocked", evidence
    return "missing", evidence


def summarize_sam31_mask_init_audit(mask_init_audit: Any) -> tuple[str, Any]:
    if not isinstance(mask_init_audit, dict):
        return (
            "missing",
            (
                "No SAM3.1 mask-init E2E split audit is attached. Run "
                "sam31-mask-init-audit to measure the Python/C++ two-frame "
                "mask-init contract and variant decisions."
            ),
        )
    case_summaries = [
        case
        for case in mask_init_audit.get("case_summaries", [])
        if isinstance(case, dict) and case.get("status") == "ok"
    ]
    default_rows = []
    for case in case_summaries:
        variants = case.get("variants")
        if not isinstance(variants, list):
            continue
        default = next(
            (row for row in variants if isinstance(row, dict) and row.get("variant") == "default"),
            None,
        )
        if isinstance(default, dict):
            default_rows.append(
                {
                    "size": case.get("size"),
                    "case": case.get("case"),
                    "default_encode_share_pct": case.get("default_encode_share_pct"),
                    "full_frame_step_ms": default.get("full_frame_step_ms"),
                    "python_full_cache_over_cpp_full_ratio": default.get(
                        "python_full_cache_over_cpp_full_ratio"
                    ),
                    "python_mask_iou_min": default.get("python_mask_iou_min"),
                    "python_mask_xor_max": default.get("python_mask_xor_max"),
                }
            )
    faster_rows = [
        row
        for row in default_rows
        if isinstance(row.get("python_full_cache_over_cpp_full_ratio"), int | float)
        and float(row["python_full_cache_over_cpp_full_ratio"]) > 1.0
    ]
    evidence = {
        "status": mask_init_audit.get("status"),
        "matrix_status": mask_init_audit.get("matrix_status"),
        "contract": mask_init_audit.get("contract"),
        "default_rows": default_rows,
        "candidate_wins": mask_init_audit.get("candidate_wins"),
    }
    if case_summaries and len(faster_rows) == len(default_rows):
        return "met", evidence
    return "missing", evidence


def summarize_sam31_sequence_quality(mask_sequence_python: Any) -> dict[str, Any]:
    thresholds = {"min_mask_iou": 0.95, "max_mask_xor_pixels": 1000}
    if not isinstance(mask_sequence_python, dict):
        return {"status": "missing", "thresholds": thresholds}
    comparison = mask_sequence_python.get("comparison")
    if not isinstance(comparison, dict):
        return {
            "status": "missing",
            "thresholds": thresholds,
            "summary": "No C++ mask comparison was attached to the official Python sequence run.",
        }
    mask_iou = comparison.get("mask_iou")
    mask_iou = mask_iou if isinstance(mask_iou, dict) else {}
    iou = mask_iou.get("iou")
    xor_pixels = mask_iou.get("xor_pixels")
    cpp_sha = comparison.get("cpp_sha256")
    python_sha = comparison.get("python_sha256")
    exact = isinstance(cpp_sha, str) and cpp_sha == python_sha
    quality_pass = (
        isinstance(iou, int | float)
        and isinstance(xor_pixels, int | float)
        and float(iou) >= thresholds["min_mask_iou"]
        and int(xor_pixels) <= thresholds["max_mask_xor_pixels"]
    )
    return {
        "status": "exact" if exact else "quality_pass" if quality_pass else "quality_fail",
        "exact_mask_hash_equal": exact,
        "thresholds": thresholds,
        "mask_iou": iou,
        "xor_pixels": xor_pixels,
        "cpp_sha256": cpp_sha,
        "python_sha256": python_sha,
        "cpp_mask": comparison.get("cpp_mask"),
        "python_mask": comparison.get("python_mask"),
        "python_result": mask_sequence_python.get("result"),
        "precision": mask_sequence_python.get("precision"),
        "tf32_policy": mask_sequence_python.get("tf32_policy"),
        "width": mask_sequence_python.get("width"),
        "height": mask_sequence_python.get("height"),
        "mask_case": mask_sequence_python.get("mask_case"),
        "frame1_offset": mask_sequence_python.get("frame1_offset"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-health", type=Path, default=Path("outputs/cuda-health/summary.env"))
    parser.add_argument(
        "--sam3-perf-summary",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--sam3-e2e-audit",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--sam31-coverage",
        type=Path,
        default=Path("outputs/sam31-checkpoint-contract/sam3-vs-sam31-renamed-coverage.json"),
    )
    parser.add_argument(
        "--sam31-multiplex",
        type=Path,
        default=Path("outputs/sam31-multiplex-state/summary.json"),
    )
    parser.add_argument(
        "--sam31-mux-mask-decoder-case",
        type=Path,
        default=Path("outputs/sam31-mux-mask-decoder-case/summary.json"),
    )
    parser.add_argument(
        "--sam31-mux-mask-decoder-compare",
        type=Path,
        default=Path("outputs/sam31-mux-mask-decoder-compare/summary.json"),
    )
    parser.add_argument(
        "--sam31-memory-backbone-case",
        type=Path,
        default=Path("outputs/sam31-memory-backbone-case/summary.json"),
    )
    parser.add_argument(
        "--sam31-memory-backbone-compare",
        type=Path,
        default=Path("outputs/sam31-memory-backbone-compare/summary.json"),
    )
    parser.add_argument(
        "--sam31-propagation-features-compare",
        type=Path,
        default=Path("outputs/sam31-propagation-features-compare/summary.json"),
    )
    parser.add_argument(
        "--sam31-memory-attention-compare",
        type=Path,
        default=Path("outputs/sam31-memory-attention-compare/summary.json"),
    )
    parser.add_argument(
        "--sam31-memory-attention-decoder-compare",
        type=Path,
        default=Path("outputs/sam31-memory-attention-decoder-compare/summary.json"),
    )
    parser.add_argument(
        "--sam31-tracking-state",
        type=Path,
        default=Path("outputs/sam31-tracking-state-inspect/summary.json"),
    )
    parser.add_argument(
        "--sam31-tracking-mask-init-smoke-log",
        type=Path,
        default=Path("outputs/sam31-tracking-mask-init-smoke/summary.log"),
    )
    parser.add_argument(
        "--sam31-mask-sequence-python",
        type=Path,
        default=Path("outputs/sam31-mask-sequence-python/summary.json"),
    )
    parser.add_argument(
        "--sam31-mask-init-audit",
        type=Path,
        default=Path("outputs/sam31-mask-init-audit/summary.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cuda_health = read_env_value(args.cuda_health, "cuda_health")
    sam3_perf_summaries = [read_json(path) for path in args.sam3_perf_summary]
    sam3_e2e_audits = [read_json(path) for path in args.sam3_e2e_audit]
    sam31_coverage = read_json(args.sam31_coverage)
    sam31_multiplex = read_json(args.sam31_multiplex)
    sam31_mux_mask_decoder_case = read_json(args.sam31_mux_mask_decoder_case)
    sam31_mux_mask_decoder_compare = read_json(args.sam31_mux_mask_decoder_compare)
    sam31_memory_backbone_case = read_json(args.sam31_memory_backbone_case)
    sam31_memory_backbone_compare = read_json(args.sam31_memory_backbone_compare)
    sam31_propagation_features_compare = read_json(args.sam31_propagation_features_compare)
    sam31_memory_attention_compare = read_json(args.sam31_memory_attention_compare)
    sam31_memory_attention_decoder_compare = read_json(
        args.sam31_memory_attention_decoder_compare
    )
    sam31_tracking_state = read_json(args.sam31_tracking_state)
    sam31_mask_sequence_python = read_json(args.sam31_mask_sequence_python)
    sam31_mask_init_audit = read_json(args.sam31_mask_init_audit)
    sam31_tracking_mask_init_smoke_log = read_text_tail(
        args.sam31_tracking_mask_init_smoke_log
    )
    sam31_tracking_mask_init_smoke = {
        "path": str(args.sam31_tracking_mask_init_smoke_log),
        "status": (
            "ok"
            if sam31_tracking_mask_init_smoke_log
            and "sam31_tracking_mask_init_smoke ok:" in sam31_tracking_mask_init_smoke_log
            else "missing"
        ),
        "tail": sam31_tracking_mask_init_smoke_log,
    }
    sam31_sequence_quality = summarize_sam31_sequence_quality(sam31_mask_sequence_python)
    manifest_checks = (
        sam31_coverage.get("sam31_implementation_manifest_checks", {})
        if isinstance(sam31_coverage, dict)
        else {}
    )
    sam3_perf_status = "missing"
    sam3_perf_evidence: Any = (
        "No current same-contract C++ vs official Python SAM3 performance result is attached to "
        "this audit. CUDA is available; rerun model-matrix for the target precision and TF32 policy."
    )
    if cuda_health != "ok":
        sam3_perf_status = "blocked"
        sam3_perf_evidence = (
            "Same-contract SAM3 performance measurement is blocked while CUDA is unavailable."
        )
    else:
        sam3_perf_status, sam3_perf_evidence = summarize_sam3_perf(
            sam3_perf_summaries, sam3_e2e_audits
        )
    sam31_mask_init_status, sam31_mask_init_evidence = summarize_sam31_mask_init_audit(
        sam31_mask_init_audit
    )

    criteria = [
        with_next_action(
            criterion(
                "CUDA backend is available for same-device SAM3/SAM3.1 performance measurement",
                "met" if cuda_health == "ok" else "blocked",
                {"summary": str(args.cuda_health), "cuda_health": cuda_health},
            ),
            "Restore CUDA device visibility, then rerun same-contract SAM3/SAM3.1 parity and performance.",
        ),
        with_next_action(
            criterion(
                "SAM3 C++ beats official Python under same input, precision, and TF32 contract",
                sam3_perf_status,
                sam3_perf_evidence,
            ),
            "Resume CUDA kernel-level optimization and repeat paired C++ vs official Python measurements once CUDA is healthy.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 C++ tracking implementation exists",
                "met" if sam31_sequence_quality.get("status") == "exact" else "partial",
                {
                    "summary": (
                        "SAM3.1 C++ has checkpoint contract, multiplex helper parity, v4 file-format support, "
                        "native tensor-manifest registration, legacy tracker tensor separation, a compiled "
                        "multiplex mask-decoder graph builder, and official Python parity for that decoder "
                        "slice. The multiplex memory-backbone feature, position-encoding, and tracker "
                        "memory-slot storage, memory-attention prompt construction, and a standalone "
                        "SAM3.1 multiplex memory-attention encoder slice are also bound and "
                        "parity-checked against official Python. The memory-attention output is now "
                        "also fed into the multiplex propagation mask decoder in one strict F32 bridge "
                        "case. The runtime tracker now has a SAM3.1 branch that encodes multiplex "
                        "memory slots, saves previous-frame image features/positions, dispatches to "
                        "the SAM3.1 memory-attention plus multiplex mask decoder graph, and selects "
                        "the slot-0 multimask output by IoU. The SAM3.1 propagation feature-adapter "
                        "path is now wired and parity-checked as a slice. Detection-mask prompt-state "
                        "initialization is wired through the memory encoder for SAM3.1, including "
                        "raw low-resolution logits when available. A full-model C++ smoke now "
                        "loads the SAM3.1 F16 checkpoint, initializes tracking from a mask, and "
                        "propagates one frame. The same synthetic mask-init sequence now also "
                        "runs through official Python. Current C++ versus Python end-to-end "
                        f"mask quality status is {sam31_sequence_quality.get('status')} "
                        f"(IoU={sam31_sequence_quality.get('mask_iou')}, "
                        f"xor={sam31_sequence_quality.get('xor_pixels')}); exact sequence "
                        "hash parity is required before this criterion is fully met."
                    ),
                    "official_mask_sequence_quality": sam31_sequence_quality,
                    "official_mux_mask_decoder_case": sam31_mux_mask_decoder_case,
                    "official_mux_mask_decoder_compare": sam31_mux_mask_decoder_compare,
                    "official_memory_backbone_case": sam31_memory_backbone_case,
                    "official_memory_backbone_compare": sam31_memory_backbone_compare,
                    "official_propagation_features_compare": (
                        sam31_propagation_features_compare
                    ),
                    "official_memory_attention_compare": sam31_memory_attention_compare,
                    "official_memory_attention_decoder_compare": (
                        sam31_memory_attention_decoder_compare
                    ),
                    "official_tracking_state": sam31_tracking_state,
                    "cpp_tracking_mask_init_smoke": sam31_tracking_mask_init_smoke,
                    "official_mask_sequence_comparison": sam31_mask_sequence_python,
                },
            ),
            (
                "Fix the SAM3.1 detection-mask prompt-state and propagation contract until the "
                "synthetic mask-init sequence matches official Python, then add point/box prompt "
                "parity and same-contract performance measurements."
            ),
        ),
        with_next_action(
            criterion(
                (
                    "SAM3.1 mask-init E2E split is measured against official Python "
                    "under the same synthetic input contract"
                ),
                sam31_mask_init_status,
                sam31_mask_init_evidence,
            ),
            (
                "Use the mask-init E2E split as the acceptance gate for propagation and "
                "image-encoder optimization candidates; promote only candidates that keep "
                "the C++ mask unchanged and improve the full-frame denominator."
            ),
        ),
        with_next_action(
            criterion(
                "SAM3.1 official prompt and multiplex tracking state contract is captured",
                "met"
                if isinstance(sam31_tracking_state, dict)
                and (
                    sam31_tracking_state.get("snapshots", {})
                    .get("after_add_prompt", {})
                    .get("facts", {})
                    .get("sam2_inference_states_len")
                    or 0
                )
                >= 1
                else "missing",
                sam31_tracking_state,
            ),
            "Use this captured contract to implement and compare the C++ SAM3.1 prompt/frame0 state before adding full propagation.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 Object Multiplex mask-decoder slice matches official Python",
                "met"
                if isinstance(sam31_mux_mask_decoder_compare, dict)
                and sam31_mux_mask_decoder_compare.get("status") == "ok"
                else "missing",
                sam31_mux_mask_decoder_compare,
            ),
            "Keep this F32 CPU parity gate green while adding CUDA execution and larger feature sizes.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 Object Multiplex memory-backbone, storage, and memory-prompt slice matches official Python",
                "met"
                if isinstance(sam31_memory_backbone_compare, dict)
                and sam31_memory_backbone_compare.get("status") == "ok"
                else "missing",
                {
                    "case": sam31_memory_backbone_case,
                    "compare": sam31_memory_backbone_compare,
                },
            ),
            (
                "Use this memory prompt as the C++ memory-attention encoder input, preserving "
                "mem_out_dim=256 and the 32-channel multiplex mask input."
            ),
        ),
        with_next_action(
            criterion(
                "SAM3.1 propagation feature-adapter slice matches official Python weights",
                "met"
                if isinstance(sam31_propagation_features_compare, dict)
                and sam31_propagation_features_compare.get("status") == "ok"
                else "missing",
                sam31_propagation_features_compare,
            ),
            "Use this adapter output in full SAM3.1 image encoding and validate prompt/propagation sequence parity.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 Object Multiplex memory-attention encoder slice matches official Python",
                "met"
                if isinstance(sam31_memory_attention_compare, dict)
                and sam31_memory_attention_compare.get("status") == "ok"
                else "missing",
                sam31_memory_attention_compare,
            ),
            (
                "Wire this parity-checked encoder into the C++ propagation path using the "
                "stored memory prompt and current-frame tracker features."
            ),
        ),
        with_next_action(
            criterion(
                (
                    "SAM3.1 Object Multiplex memory-attention output feeds the mask decoder "
                    "with official Python parity"
                ),
                "met"
                if isinstance(sam31_memory_attention_decoder_compare, dict)
                and sam31_memory_attention_decoder_compare.get("status") == "ok"
                else "missing",
                sam31_memory_attention_decoder_compare,
            ),
            (
                "Promote this strict bridge into the streaming propagation implementation and "
                "measure the same frame sequence against official Python."
            ),
        ),
        with_next_action(
            criterion(
                "SAM3.1 checkpoint contract is loader-ready",
                "met" if manifest_checks.get("diff_rows") == 0 else "missing",
                manifest_checks,
            ),
            "Keep this gate green while adding each SAM3.1 loader/graph slice.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 Object Multiplex helper matches official Python",
                "met"
                if isinstance(sam31_multiplex, dict) and sam31_multiplex.get("diff_rows") == 0
                else "missing",
                sam31_multiplex,
            ),
            "Use the helper as the demux/mux reference when wiring SAM3.1 propagation outputs.",
        ),
    ]
    complete = all(item["status"] == "met" for item in criteria)
    missing_criteria = [
        item
        for item in criteria
        if item["status"] != "met"
    ]
    out = {
        "objective": (
            "SAM3 and SAM3.1 C++ beat official Python under the same precision, TF32 policy, "
            "and input contract, with parity/perf evidence."
        ),
        "complete": complete,
        "missing_criteria": missing_criteria,
        "criteria": criteria,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sam3_sam31_goal_complete={str(complete).lower()} criteria={len(criteria)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
