# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub",
#   "pillow",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any
import statistics

from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image


CPP_ROW_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s+\|\s+"
    r"(?P<model>[^|]+?)\s+\|\s+"
    r"(?P<size>[^|]+?)\s+\|\s+"
    r"(?P<backend>[^|]+?)\s+\|\s+"
    r"(?P<load>[-0-9.]+)\s+\|\s+"
    r"(?P<init>[-0-9.]+)\s+\|\s+"
    r"(?P<track>[-0-9.]+)\s+\|\s+"
    r"(?P<p50>[-0-9.]+)\s+\|\s+"
    r"(?P<p95>[-0-9.]+)\s+\|\s+"
    r"(?P<total>[-0-9.]+)\s+\|\s+"
    r"(?P<rss>[-0-9.]+)\s+\|\s+"
    r"(?P<det>\d+)\s+\|\s+OK\s*$"
)

CPP_FAIL_ROW_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s+\|\s+"
    r"(?P<model>[^|]+?)\s+\|\s+"
    r"(?P<size>[^|]+?)\s+\|\s+"
    r"(?P<backend>[^|]+?)\s+\|\s+"
    r"(?P<load>[-0-9.]+|-)\s+\|\s+"
    r"(?P<init>[-0-9.]+|-)\s+\|\s+"
    r"(?P<track>[-0-9.]+|-)\s+\|\s+"
    r"(?P<p50>[-0-9.]+|-)\s+\|\s+"
    r"(?P<p95>[-0-9.]+|-)\s+\|\s+"
    r"(?P<total>[-0-9.]+|-)\s+\|\s+"
    r"(?P<rss>[-0-9.]+|-)\s+\|\s+"
    r"(?P<det>\d+|-)\s+\|\s+FAIL:\s+(?P<error>.+?)\s*$"
)
CPP_RESULT_JSON_PREFIX = "SAM3_BENCH_RESULT_JSON "

SAM2_UV_DEPS = [
    "numpy",
    "torch==2.8.0",
    "torchvision==0.23.0",
    "tqdm",
    "hydra-core",
    "iopath",
    "opencv-python-headless",
    "pillow",
]

SAM3_UV_DEPS = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "timm",
    "numpy>=1.26,<2",
    "tqdm",
    "ftfy==6.1.1",
    "regex",
    "iopath",
    "einops",
    "typing_extensions",
    "huggingface_hub",
    "pycocotools",
    "psutil",
    "setuptools<81",
]

SAM3_PYTHON_NONE = "none"


def uv_run_python(
    project: Path, code: str, argv: list[str], deps: list[str]
) -> list[str]:
    _ = project
    cmd = ["uv", "run", "--no-project"]
    for dep in deps:
        cmd.extend(["--with", dep])
    return cmd + ["python", "-c", code, *argv]


def uv_run_python_script(
    project: Path, script: Path, argv: list[str], deps: list[str]
) -> list[str]:
    _ = project
    cmd = ["uv", "run", "--no-project"]
    for dep in deps:
        cmd.extend(["--with", dep])
    return cmd + ["python", str(script), *argv]


def run(
    cmd: list[str],
    cwd: Path,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(cmd)}")
    return proc.stdout


def model_key(name: str) -> str:
    stem = name.removesuffix(".ggml")
    for suffix in (
        "_f32",
        "_f16",
        "_q8_0",
        "_q4_1",
        "_q4_0",
        "_bf16",
        "_mxfp4",
        "_nvfp4",
        "-f32",
        "-bf16",
        "-f16",
        "-q8_0",
        "-q4_1",
        "-q4_0",
        "-mxfp4",
        "-nvfp4",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def comparison_contract_key(family: str) -> str:
    if family in {"sam3", "sam3.1"} or family.startswith("sam3-visual"):
        return "sam3-official"
    return family


def precision(name: str) -> str:
    stem = name.removesuffix(".ggml")
    for value in ("f32", "bf16", "f16", "q8_0", "q4_1", "q4_0", "mxfp4", "nvfp4"):
        if stem.endswith(value):
            return value
    return "unknown"


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def nested_mean(mapping: dict[str, Any] | None, *keys: str) -> float | None:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if not isinstance(cur, dict):
        return None
    value = cur.get("mean")
    return float(value) if isinstance(value, int | float) else None


def nested_number(mapping: dict[str, Any] | None, *keys: str) -> float | None:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return float(cur) if isinstance(cur, int | float) else None


def optional_sum(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def optional_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def phase_stage_means(mapping: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for phase, stages in mapping.items():
        if not isinstance(stages, dict):
            continue
        stage_means = {
            str(stage): float(stats["mean"])
            for stage, stats in stages.items()
            if isinstance(stats, dict) and isinstance(stats.get("mean"), int | float)
        }
        if stage_means:
            out[str(phase)] = stage_means
    return out


CPP_INPUT_TIMING_KEYS = (
    "total_ms",
    "directory_scan_ms",
    "sort_ms",
    "frame_decode_ms",
    "frame_store_ms",
    "accounted_ms",
    "remainder_ms",
    "input_frames",
)

CPP_FRAME0_TIMING_KEYS = (
    "total_ms",
    "model_ms",
    "encode_ms",
    "prompt_ms",
    "segment_ms",
    "tracker_add_ms",
    "tracker_mask_prepare_ms",
    "tracker_memory_encode_ms",
    "tracker_obj_ptr_ms",
    "tracker_store_ms",
    "tracker_unattributed_ms",
    "prompt_unattributed_ms",
    "artifact_ms",
    "encode_timing_total_ms",
    "encode_preprocess_ms",
    "encode_graph_build_ms",
    "encode_graph_alloc_ms",
    "encode_input_upload_ms",
    "encode_graph_compute_ms",
    "encode_state_update_ms",
    "encode_pe_build_ms",
    "encode_accounted_ms",
    "encode_remainder_ms",
)

CPP_TAIL_TIMING_KEYS = (
    "total_ms",
    "model_ms",
    "runtime_wall_ms",
    "visible_model_ms",
    "preencode_required_model_ms",
    "required_model_ms",
    "required_wall_ms",
    "inline_encode_ms",
    "preencode_state_create_ms",
    "preencode_ms",
    "required_encode_ms",
    "propagate_ms",
    "artifact_ms",
    "unattributed_ms",
    "encode_timing_total_ms",
    "encode_preprocess_ms",
    "encode_graph_build_ms",
    "encode_graph_alloc_ms",
    "encode_input_upload_ms",
    "encode_graph_compute_ms",
    "encode_state_update_ms",
    "encode_pe_build_ms",
    "encode_accounted_ms",
    "encode_remainder_ms",
    "propagate_timing_total_ms",
    "propagate_prepare_caches_ms",
    "propagate_memory_slot_read_ms",
    "propagate_obj_ptr_read_ms",
    "propagate_prompt_build_ms",
    "propagate_memory_prepare_ms",
    "propagate_ptr_prepare_ms",
    "propagate_rope_cache_ms",
    "propagate_rope_k_build_ms",
    "propagate_graph_build_ms",
    "propagate_graph_alloc_ms",
    "propagate_input_upload_ms",
    "propagate_input_prompt_upload_ms",
    "propagate_input_rope_upload_ms",
    "propagate_input_memory_upload_ms",
    "propagate_input_feature_upload_ms",
    "propagate_input_constant_upload_ms",
    "propagate_input_sparse_upload_ms",
    "propagate_graph_compute_ms",
    "propagate_output_read_ms",
    "propagate_graph_cache_build_ms",
    "propagate_graph_cache_input_upload_ms",
    "propagate_graph_cache_compute_ms",
    "propagate_graph_cache_output_read_ms",
    "propagate_bbox_from_logits_ms",
    "propagate_fullmask_active_resize_bbox_ms",
    "propagate_fullmask_pending_resize_bbox_ms",
    "propagate_memory_update_ms",
    "propagate_tracker_update_ms",
    "propagate_result_build_ms",
    "propagate_accounted_ms",
    "propagate_remainder_ms",
    "propagate_single_calls",
    "propagate_active_masklets",
    "propagate_pending_masklets",
    "propagate_aliased_feature_input_count",
    "propagate_aliased_constant_input_count",
    "propagate_uploaded_feature_input_count",
    "propagate_uploaded_constant_input_count",
)


def numeric_value(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def sum_timing_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "n": len(rows),
        **{key: sum(numeric_value(row, key) for row in rows) for key in keys},
    }


def resolve_output_path(out_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else out_dir / path


def aggregate_cpp_frame_timing_file(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    input_rows = [row for row in rows if row.get("stage") == "input"]
    frame0_rows = [row for row in rows if row.get("stage") == "frame0"]
    tail_rows = [row for row in rows if row.get("stage") == "tail"]
    required_tail_rows = [
        row for row in tail_rows if bool(row.get("included_in_required_e2e", False))
    ]
    timed_tail_rows = [row for row in required_tail_rows if bool(row.get("timed_tail"))]
    inline_tail_rows = [
        row for row in required_tail_rows if not bool(row.get("preencoded", False))
    ]
    preencoded_tail_rows = [
        row for row in required_tail_rows if bool(row.get("preencoded", False))
    ]
    return {
        "path": path.name,
        "input_ms": sum_timing_rows(input_rows, CPP_INPUT_TIMING_KEYS),
        "frame0_ms": sum_timing_rows(frame0_rows, CPP_FRAME0_TIMING_KEYS),
        "tail_all_required_ms": sum_timing_rows(
            required_tail_rows, CPP_TAIL_TIMING_KEYS
        ),
        "tail_timed_ms": sum_timing_rows(timed_tail_rows, CPP_TAIL_TIMING_KEYS),
        "tail_inline_required_ms": sum_timing_rows(
            inline_tail_rows, CPP_TAIL_TIMING_KEYS
        ),
        "tail_preencoded_required_ms": sum_timing_rows(
            preencoded_tail_rows, CPP_TAIL_TIMING_KEYS
        ),
    }


def aggregate_cpp_frame_timing_runs(
    out_dir: Path, path_values: list[str]
) -> dict[str, Any] | None:
    run_aggregates: list[dict[str, Any]] = []
    for path_value in path_values:
        path = resolve_output_path(out_dir, path_value)
        if path.exists():
            run_aggregates.append(aggregate_cpp_frame_timing_file(path))
    if not run_aggregates:
        return None

    scopes = (
        "input_ms",
        "frame0_ms",
        "tail_all_required_ms",
        "tail_timed_ms",
        "tail_inline_required_ms",
        "tail_preencoded_required_ms",
    )
    mean: dict[str, dict[str, float]] = {}
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for scope in scopes:
        keys = set().union(
            *[
                set(run.get(scope, {}).keys()) - {"n"}
                for run in run_aggregates
                if isinstance(run.get(scope), dict)
            ]
        )
        mean[scope] = {}
        stats[scope] = {}
        for key in sorted(keys):
            values = [
                float(run[scope][key])
                for run in run_aggregates
                if isinstance(run.get(scope), dict)
                and isinstance(run[scope].get(key), int | float)
            ]
            if values:
                mean[scope][key] = statistics.mean(values)
                stats[scope][key] = summarize_values(values)
        counts = [
            float(run[scope]["n"])
            for run in run_aggregates
            if isinstance(run.get(scope), dict)
            and isinstance(run[scope].get("n"), int | float)
        ]
        if counts:
            mean[scope]["n"] = statistics.mean(counts)

    return {"runs": run_aggregates, "mean": mean, "stats": stats}


def add_cpp_compute_split(row: dict[str, Any]) -> None:
    frame_timing = row.get("cpp_frame_timing_aggregate_ms")
    has_frame_timing = isinstance(frame_timing, dict)

    def frame_timing_or_row(row_key: str, *path: str) -> float | None:
        if has_frame_timing:
            value = nested_number(frame_timing, *path)
            if value is not None:
                return value
        row_value = row.get(row_key)
        return float(row_value) if isinstance(row_value, int | float) else None

    frame0_graph_compute_ms = frame_timing_or_row(
        "frame0_encode_graph_compute_ms",
        "mean",
        "frame0_ms",
        "encode_graph_compute_ms",
    )
    frame0_encode_preprocess_ms = frame_timing_or_row(
        "frame0_encode_preprocess_ms",
        "mean",
        "frame0_ms",
        "encode_preprocess_ms",
    )
    frame0_encode_graph_build_ms = frame_timing_or_row(
        "frame0_encode_graph_build_ms",
        "mean",
        "frame0_ms",
        "encode_graph_build_ms",
    )
    frame0_encode_graph_alloc_ms = frame_timing_or_row(
        "frame0_encode_graph_alloc_ms",
        "mean",
        "frame0_ms",
        "encode_graph_alloc_ms",
    )
    frame0_encode_input_upload_ms = frame_timing_or_row(
        "frame0_encode_input_upload_ms",
        "mean",
        "frame0_ms",
        "encode_input_upload_ms",
    )
    frame0_encode_state_update_ms = frame_timing_or_row(
        "frame0_encode_state_update_ms",
        "mean",
        "frame0_ms",
        "encode_state_update_ms",
    )
    frame0_encode_pe_build_ms = frame_timing_or_row(
        "frame0_encode_pe_build_ms",
        "mean",
        "frame0_ms",
        "encode_pe_build_ms",
    )
    frame0_encode_remainder_ms = frame_timing_or_row(
        "frame0_encode_remainder_ms",
        "mean",
        "frame0_ms",
        "encode_remainder_ms",
    )
    tail_encode_graph_compute_ms = frame_timing_or_row(
        "tail_encode_graph_compute_ms",
        "mean",
        "tail_all_required_ms",
        "encode_graph_compute_ms",
    )
    tail_encode_preprocess_ms = frame_timing_or_row(
        "tail_encode_preprocess_ms",
        "mean",
        "tail_all_required_ms",
        "encode_preprocess_ms",
    )
    tail_encode_graph_build_ms = frame_timing_or_row(
        "tail_encode_graph_build_ms",
        "mean",
        "tail_all_required_ms",
        "encode_graph_build_ms",
    )
    tail_encode_graph_alloc_ms = frame_timing_or_row(
        "tail_encode_graph_alloc_ms",
        "mean",
        "tail_all_required_ms",
        "encode_graph_alloc_ms",
    )
    tail_encode_input_upload_ms = frame_timing_or_row(
        "tail_encode_input_upload_ms",
        "mean",
        "tail_all_required_ms",
        "encode_input_upload_ms",
    )
    tail_encode_state_update_ms = frame_timing_or_row(
        "tail_encode_state_update_ms",
        "mean",
        "tail_all_required_ms",
        "encode_state_update_ms",
    )
    tail_encode_pe_build_ms = frame_timing_or_row(
        "tail_encode_pe_build_ms",
        "mean",
        "tail_all_required_ms",
        "encode_pe_build_ms",
    )
    tail_encode_remainder_ms = frame_timing_or_row(
        "tail_encode_remainder_ms",
        "mean",
        "tail_all_required_ms",
        "encode_remainder_ms",
    )
    tail_inline_encode_ms = frame_timing_or_row(
        "tail_inline_encode_ms",
        "mean",
        "tail_inline_required_ms",
        "inline_encode_ms",
    )
    tail_inline_encode_graph_compute_ms = frame_timing_or_row(
        "tail_inline_encode_graph_compute_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_graph_compute_ms",
    )
    tail_inline_encode_preprocess_ms = frame_timing_or_row(
        "tail_inline_encode_preprocess_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_preprocess_ms",
    )
    tail_inline_encode_graph_build_ms = frame_timing_or_row(
        "tail_inline_encode_graph_build_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_graph_build_ms",
    )
    tail_inline_encode_graph_alloc_ms = frame_timing_or_row(
        "tail_inline_encode_graph_alloc_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_graph_alloc_ms",
    )
    tail_inline_encode_input_upload_ms = frame_timing_or_row(
        "tail_inline_encode_input_upload_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_input_upload_ms",
    )
    tail_inline_encode_state_update_ms = frame_timing_or_row(
        "tail_inline_encode_state_update_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_state_update_ms",
    )
    tail_inline_encode_pe_build_ms = frame_timing_or_row(
        "tail_inline_encode_pe_build_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_pe_build_ms",
    )
    tail_inline_encode_remainder_ms = frame_timing_or_row(
        "tail_inline_encode_remainder_ms",
        "mean",
        "tail_inline_required_ms",
        "encode_remainder_ms",
    )
    tail_preencode_state_create_ms = frame_timing_or_row(
        "tail_preencode_state_create_ms",
        "mean",
        "tail_preencoded_required_ms",
        "preencode_state_create_ms",
    )
    tail_preencode_encode_ms = frame_timing_or_row(
        "tail_preencode_encode_ms",
        "mean",
        "tail_preencoded_required_ms",
        "preencode_ms",
    )
    tail_preencode_encode_graph_compute_ms = frame_timing_or_row(
        "tail_preencode_encode_graph_compute_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_graph_compute_ms",
    )
    tail_preencode_encode_preprocess_ms = frame_timing_or_row(
        "tail_preencode_encode_preprocess_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_preprocess_ms",
    )
    tail_preencode_encode_graph_build_ms = frame_timing_or_row(
        "tail_preencode_encode_graph_build_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_graph_build_ms",
    )
    tail_preencode_encode_graph_alloc_ms = frame_timing_or_row(
        "tail_preencode_encode_graph_alloc_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_graph_alloc_ms",
    )
    tail_preencode_encode_input_upload_ms = frame_timing_or_row(
        "tail_preencode_encode_input_upload_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_input_upload_ms",
    )
    tail_preencode_encode_state_update_ms = frame_timing_or_row(
        "tail_preencode_encode_state_update_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_state_update_ms",
    )
    tail_preencode_encode_pe_build_ms = frame_timing_or_row(
        "tail_preencode_encode_pe_build_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_pe_build_ms",
    )
    tail_preencode_encode_remainder_ms = frame_timing_or_row(
        "tail_preencode_encode_remainder_ms",
        "mean",
        "tail_preencoded_required_ms",
        "encode_remainder_ms",
    )
    tail_propagate_graph_compute_ms = frame_timing_or_row(
        "tail_propagate_graph_compute_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_graph_compute_ms",
    )
    tail_propagate_graph_build_ms = frame_timing_or_row(
        "tail_propagate_graph_build_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_graph_build_ms",
    )
    tail_propagate_graph_alloc_ms = frame_timing_or_row(
        "tail_propagate_graph_alloc_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_graph_alloc_ms",
    )
    tail_propagate_input_upload_ms = frame_timing_or_row(
        "tail_propagate_input_upload_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_input_upload_ms",
    )
    tail_propagate_output_read_ms = frame_timing_or_row(
        "tail_propagate_output_read_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_output_read_ms",
    )
    tail_propagate_cache_compute_ms = frame_timing_or_row(
        "tail_propagate_cache_compute_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_graph_cache_compute_ms",
    )
    tail_propagate_input_prompt_upload_ms = frame_timing_or_row(
        "tail_propagate_input_prompt_upload_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_input_prompt_upload_ms",
    )
    tail_propagate_input_rope_upload_ms = frame_timing_or_row(
        "tail_propagate_input_rope_upload_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_input_rope_upload_ms",
    )
    tail_propagate_input_memory_upload_ms = frame_timing_or_row(
        "tail_propagate_input_memory_upload_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_input_memory_upload_ms",
    )
    tail_propagate_input_feature_upload_ms = frame_timing_or_row(
        "tail_propagate_input_feature_upload_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_input_feature_upload_ms",
    )
    tail_propagate_input_constant_upload_ms = frame_timing_or_row(
        "tail_propagate_input_constant_upload_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_input_constant_upload_ms",
    )
    tail_propagate_input_sparse_upload_ms = frame_timing_or_row(
        "tail_propagate_input_sparse_upload_ms",
        "mean",
        "tail_all_required_ms",
        "propagate_input_sparse_upload_ms",
    )
    frame0_encode_graph_admin_ms = optional_sum(
        frame0_encode_graph_build_ms,
        frame0_encode_graph_alloc_ms,
        frame0_encode_input_upload_ms,
    )
    tail_encode_graph_admin_ms = optional_sum(
        tail_encode_graph_build_ms,
        tail_encode_graph_alloc_ms,
        tail_encode_input_upload_ms,
    )
    tail_inline_encode_graph_admin_ms = optional_sum(
        tail_inline_encode_graph_build_ms,
        tail_inline_encode_graph_alloc_ms,
        tail_inline_encode_input_upload_ms,
    )
    tail_inline_encode_non_compute_ms = optional_sum(
        tail_inline_encode_preprocess_ms,
        tail_inline_encode_graph_admin_ms,
        tail_inline_encode_state_update_ms,
        tail_inline_encode_pe_build_ms,
        tail_inline_encode_remainder_ms,
    )
    tail_preencode_encode_graph_admin_ms = optional_sum(
        tail_preencode_encode_graph_build_ms,
        tail_preencode_encode_graph_alloc_ms,
        tail_preencode_encode_input_upload_ms,
    )
    tail_preencode_encode_non_compute_ms = optional_sum(
        tail_preencode_encode_preprocess_ms,
        tail_preencode_encode_graph_admin_ms,
        tail_preencode_encode_state_update_ms,
        tail_preencode_encode_pe_build_ms,
        tail_preencode_encode_remainder_ms,
    )
    required_encode_graph_admin_ms = optional_sum(
        frame0_encode_graph_admin_ms,
        tail_encode_graph_admin_ms,
    )
    required_encode_preprocess_ms = optional_sum(
        frame0_encode_preprocess_ms,
        tail_encode_preprocess_ms,
    )
    required_encode_state_update_ms = optional_sum(
        frame0_encode_state_update_ms,
        tail_encode_state_update_ms,
    )
    required_encode_pe_build_ms = optional_sum(
        frame0_encode_pe_build_ms,
        tail_encode_pe_build_ms,
    )
    required_encode_remainder_ms = optional_sum(
        frame0_encode_remainder_ms,
        tail_encode_remainder_ms,
    )
    required_encode_non_compute_ms = optional_sum(
        required_encode_preprocess_ms,
        required_encode_graph_admin_ms,
        required_encode_state_update_ms,
        required_encode_pe_build_ms,
        required_encode_remainder_ms,
    )
    tail_propagate_graph_admin_ms = optional_sum(
        tail_propagate_graph_build_ms,
        tail_propagate_graph_alloc_ms,
        tail_propagate_input_upload_ms,
        tail_propagate_output_read_ms,
    )
    required_core_compute_ms = optional_sum(
        frame0_graph_compute_ms,
        tail_encode_graph_compute_ms,
        tail_propagate_graph_compute_ms,
        tail_propagate_cache_compute_ms,
    )
    required_e2e_ms = row.get("required_e2e_ms")
    tail_encode_required_total_ms = row.get("tail_encode_required_total_ms")
    required_non_core_compute_ms = (
        required_e2e_ms - required_core_compute_ms
        if isinstance(required_e2e_ms, int | float)
        and required_core_compute_ms is not None
        else None
    )
    row["required_core_compute_ms"] = required_core_compute_ms
    row["required_non_core_compute_ms"] = required_non_core_compute_ms
    row["required_core_compute_fraction"] = optional_ratio(
        required_core_compute_ms, required_e2e_ms
    )
    row["frame0_encode_preprocess_ms"] = frame0_encode_preprocess_ms
    row["frame0_encode_graph_build_ms"] = frame0_encode_graph_build_ms
    row["frame0_encode_graph_alloc_ms"] = frame0_encode_graph_alloc_ms
    row["frame0_encode_input_upload_ms"] = frame0_encode_input_upload_ms
    row["frame0_encode_state_update_ms"] = frame0_encode_state_update_ms
    row["frame0_encode_pe_build_ms"] = frame0_encode_pe_build_ms
    row["frame0_encode_remainder_ms"] = frame0_encode_remainder_ms
    row["frame0_encode_graph_admin_ms"] = frame0_encode_graph_admin_ms
    row["frame0_encode_graph_compute_ms"] = frame0_graph_compute_ms
    row["tail_encode_preprocess_ms"] = tail_encode_preprocess_ms
    row["tail_encode_graph_build_ms"] = tail_encode_graph_build_ms
    row["tail_encode_graph_alloc_ms"] = tail_encode_graph_alloc_ms
    row["tail_encode_input_upload_ms"] = tail_encode_input_upload_ms
    row["tail_encode_state_update_ms"] = tail_encode_state_update_ms
    row["tail_encode_pe_build_ms"] = tail_encode_pe_build_ms
    row["tail_encode_remainder_ms"] = tail_encode_remainder_ms
    row["tail_encode_graph_admin_ms"] = tail_encode_graph_admin_ms
    row["required_encode_graph_admin_ms"] = required_encode_graph_admin_ms
    row["required_encode_preprocess_ms"] = required_encode_preprocess_ms
    row["required_encode_state_update_ms"] = required_encode_state_update_ms
    row["required_encode_pe_build_ms"] = required_encode_pe_build_ms
    row["required_encode_remainder_ms"] = required_encode_remainder_ms
    row["required_encode_non_compute_ms"] = required_encode_non_compute_ms
    row["tail_encode_graph_compute_ms"] = tail_encode_graph_compute_ms
    row["tail_encode_graph_compute_fraction"] = optional_ratio(
        tail_encode_graph_compute_ms, tail_encode_required_total_ms
    )
    row["tail_inline_encode_ms"] = tail_inline_encode_ms
    row["tail_inline_encode_graph_compute_ms"] = tail_inline_encode_graph_compute_ms
    row["tail_inline_encode_preprocess_ms"] = tail_inline_encode_preprocess_ms
    row["tail_inline_encode_graph_admin_ms"] = tail_inline_encode_graph_admin_ms
    row["tail_inline_encode_graph_build_ms"] = tail_inline_encode_graph_build_ms
    row["tail_inline_encode_graph_alloc_ms"] = tail_inline_encode_graph_alloc_ms
    row["tail_inline_encode_input_upload_ms"] = tail_inline_encode_input_upload_ms
    row["tail_inline_encode_state_update_ms"] = tail_inline_encode_state_update_ms
    row["tail_inline_encode_pe_build_ms"] = tail_inline_encode_pe_build_ms
    row["tail_inline_encode_remainder_ms"] = tail_inline_encode_remainder_ms
    row["tail_inline_encode_non_compute_ms"] = tail_inline_encode_non_compute_ms
    row["tail_preencode_state_create_ms"] = tail_preencode_state_create_ms
    row["tail_preencode_encode_ms"] = tail_preencode_encode_ms
    row["tail_preencode_encode_graph_compute_ms"] = (
        tail_preencode_encode_graph_compute_ms
    )
    row["tail_preencode_encode_preprocess_ms"] = tail_preencode_encode_preprocess_ms
    row["tail_preencode_encode_graph_admin_ms"] = (
        tail_preencode_encode_graph_admin_ms
    )
    row["tail_preencode_encode_graph_build_ms"] = tail_preencode_encode_graph_build_ms
    row["tail_preencode_encode_graph_alloc_ms"] = tail_preencode_encode_graph_alloc_ms
    row["tail_preencode_encode_input_upload_ms"] = tail_preencode_encode_input_upload_ms
    row["tail_preencode_encode_state_update_ms"] = tail_preencode_encode_state_update_ms
    row["tail_preencode_encode_pe_build_ms"] = tail_preencode_encode_pe_build_ms
    row["tail_preencode_encode_remainder_ms"] = tail_preencode_encode_remainder_ms
    row["tail_preencode_encode_non_compute_ms"] = tail_preencode_encode_non_compute_ms
    row["tail_propagate_graph_compute_ms"] = tail_propagate_graph_compute_ms
    row["tail_propagate_graph_build_ms"] = tail_propagate_graph_build_ms
    row["tail_propagate_graph_alloc_ms"] = tail_propagate_graph_alloc_ms
    row["tail_propagate_input_upload_ms"] = tail_propagate_input_upload_ms
    row["tail_propagate_output_read_ms"] = tail_propagate_output_read_ms
    row["tail_propagate_graph_admin_ms"] = tail_propagate_graph_admin_ms
    row["tail_propagate_cache_compute_ms"] = tail_propagate_cache_compute_ms
    row["tail_propagate_input_prompt_upload_ms"] = tail_propagate_input_prompt_upload_ms
    row["tail_propagate_input_rope_upload_ms"] = tail_propagate_input_rope_upload_ms
    row["tail_propagate_input_memory_upload_ms"] = tail_propagate_input_memory_upload_ms
    row["tail_propagate_input_feature_upload_ms"] = (
        tail_propagate_input_feature_upload_ms
    )
    row["tail_propagate_input_constant_upload_ms"] = (
        tail_propagate_input_constant_upload_ms
    )
    row["tail_propagate_input_sparse_upload_ms"] = tail_propagate_input_sparse_upload_ms


def parse_key_value_overrides(values: list[str], *, option_name: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        try:
            tokens = shlex.split(value)
        except ValueError as exc:
            raise SystemExit(f"{option_name} could not parse {value!r}: {exc}") from exc
        for token in tokens:
            if "=" not in token:
                raise SystemExit(f"{option_name} expects KEY=VALUE, got: {token}")
            key, item_value = token.split("=", 1)
            if not key:
                raise SystemExit(f"{option_name} expects a non-empty KEY, got: {token}")
            overrides[key] = item_value
    return overrides


def cached_track_scope(timed_start_frame: int) -> str:
    return f"frame{timed_start_frame}_plus_cached_propagation_only"


def cpp_track_scope(args: argparse.Namespace) -> str:
    if args.cpp_preencode_track_frames:
        if args.timed_start_frame > 1:
            return cached_track_scope(args.timed_start_frame)
        return "frame1_plus_preencoded_tracker_features_propagate_only"
    if args.cpp_preencode_cached_tail_frames:
        if args.timed_start_frame > 1:
            return cached_track_scope(args.timed_start_frame)
        return python_sam3_track_scope(args.timed_start_frame)
    if args.cpp_text_init_selected_only:
        return f"frame{args.timed_start_frame}_plus_cpp_selected_text_detection_only"
    return f"frame{args.timed_start_frame}_plus_cpp_preprocess_encode_propagate_prompt_excluded"


def python_sam3_track_scope(timed_start_frame: int) -> str:
    if timed_start_frame > 1:
        return cached_track_scope(timed_start_frame)
    return "frame1_plus_python_start_session_cached_encode_propagate_prompt_excluded"


def download_ggml(models_dir: Path, filters: list[str]) -> list[Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files("PABannier/sam3.cpp") if f.endswith(".ggml")]
    if filters:
        files = [f for f in files if any(token in f for token in filters)]
    out: list[Path] = []
    for filename in files:
        target = models_dir / Path(filename).name
        if target.exists() and target.stat().st_size > 0:
            out.append(target)
            continue
        downloaded = hf_hub_download(
            repo_id="PABannier/sam3.cpp",
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
        out.append(Path(downloaded))
    return out


def parse_cpp_table(text: str) -> list[dict[str, Any]]:
    json_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith(CPP_RESULT_JSON_PREFIX):
            payload = json.loads(line.removeprefix(CPP_RESULT_JSON_PREFIX))
            model = str(payload["model"]).strip()
            row = {
                "model": model,
                "family": model_key(model),
                "precision": precision(model),
                "size": str(payload["size"]).strip(),
                "backend": str(payload["backend"]).strip(),
                "ok": bool(payload.get("ok", False)),
            }
            if row["ok"]:
                row.update(
                    {
                        "load_ms": float(payload["load_ms"]),
                        "model_load_ms": float(
                            payload.get("model_load_ms", payload["load_ms"])
                        ),
                        "init_ms": float(payload["frame0_ms"]),
                        "frame0_encode_ms": float(payload.get("frame0_encode_ms", 0.0)),
                        "frame0_prompt_ms": float(payload.get("frame0_prompt_ms", 0.0)),
                        "frame0_segment_ms": float(
                            payload.get("frame0_segment_ms", 0.0)
                        ),
                        "frame0_tracker_add_ms": float(
                            payload.get("frame0_tracker_add_ms", 0.0)
                        ),
                        "frame0_tracker_mask_prepare_ms": float(
                            payload.get("frame0_tracker_mask_prepare_ms", 0.0)
                        ),
                        "frame0_tracker_memory_encode_ms": float(
                            payload.get("frame0_tracker_memory_encode_ms", 0.0)
                        ),
                        "frame0_tracker_obj_ptr_ms": float(
                            payload.get("frame0_tracker_obj_ptr_ms", 0.0)
                        ),
                        "frame0_tracker_store_ms": float(
                            payload.get("frame0_tracker_store_ms", 0.0)
                        ),
                        "frame0_tracker_unattributed_ms": float(
                            payload.get("frame0_tracker_unattributed_ms", 0.0)
                        ),
                        "frame0_prompt_unattributed_ms": float(
                            payload.get("frame0_prompt_unattributed_ms", 0.0)
                        ),
                        "track_ms": float(payload["track_ms"]),
                        "p50_ms": float(payload["p50_ms"]),
                        "p95_ms": float(payload["p95_ms"]),
                        "total_ms": float(payload["total_ms"]),
                        "session_ms": float(
                            payload.get("session_ms", payload["total_ms"])
                        ),
                        "has_input_load_ms": "input_load_ms" in payload,
                        "input_load_ms": float(payload.get("input_load_ms", 0.0)),
                        "input_directory_scan_ms": float(
                            payload.get("input_directory_scan_ms", 0.0)
                        ),
                        "input_sort_ms": float(payload.get("input_sort_ms", 0.0)),
                        "input_frame_decode_ms": float(
                            payload.get("input_frame_decode_ms", 0.0)
                        ),
                        "input_frame_store_ms": float(
                            payload.get("input_frame_store_ms", 0.0)
                        ),
                        "input_remainder_ms": float(
                            payload.get("input_remainder_ms", 0.0)
                        ),
                        "input_frames": int(payload.get("input_frames", 0)),
                        "input_source": str(payload.get("input_source", "")),
                        "model_e2e_ms": float(
                            payload.get(
                                "model_e2e_ms",
                                payload.get("session_ms", payload["total_ms"]),
                            )
                        ),
                        "required_e2e_ms": float(
                            payload.get(
                                "required_e2e_ms",
                                payload.get(
                                    "model_e2e_ms",
                                    payload.get("session_ms", payload["total_ms"]),
                                )
                                + payload.get("input_load_ms", 0.0),
                            )
                        ),
                        "cold_required_e2e_ms": float(
                            payload.get(
                                "cold_required_e2e_ms",
                                payload.get(
                                    "required_e2e_ms",
                                    payload.get(
                                        "model_e2e_ms",
                                        payload.get("session_ms", payload["total_ms"]),
                                    )
                                    + payload.get("input_load_ms", 0.0),
                                )
                                + payload.get("model_load_ms", payload["load_ms"]),
                            )
                        ),
                        "input_plus_session_ms": float(
                            payload.get(
                                "input_plus_session_ms",
                                payload.get("session_ms", payload["total_ms"])
                                + payload.get("input_load_ms", 0.0),
                            )
                        ),
                        "required_accounted_ms": (
                            float(payload["required_accounted_ms"])
                            if "required_accounted_ms" in payload
                            else None
                        ),
                        "required_remainder_ms": float(
                            payload.get(
                                "required_remainder_ms",
                                payload.get("model_remainder_ms", 0.0),
                            )
                        ),
                        "required_accounting_delta_ms": float(
                            payload.get(
                                "required_accounting_delta_ms",
                                payload.get(
                                    "required_remainder_ms",
                                    payload.get("model_remainder_ms", 0.0),
                                ),
                            )
                        ),
                        "model_accounting_delta_ms": float(
                            payload.get(
                                "model_accounting_delta_ms",
                                payload.get("model_remainder_ms", 0.0),
                            )
                        ),
                        "model_accounted_ms": (
                            float(payload["model_accounted_ms"])
                            if "model_accounted_ms" in payload
                            else None
                        ),
                        "session_overhead_ms": float(
                            payload.get("session_overhead_ms", 0.0)
                        ),
                        "model_remainder_ms": float(
                            payload.get("model_remainder_ms", 0.0)
                        ),
                        "state_create_ms": float(payload.get("state_create_ms", 0.0)),
                        "tracker_create_ms": float(
                            payload.get("tracker_create_ms", 0.0)
                        ),
                        "session_setup_ms": float(payload.get("session_setup_ms", 0.0)),
                        "artifact_ms": float(payload.get("artifact_ms", 0.0)),
                        "frame0_artifact_ms": float(
                            payload.get("frame0_artifact_ms", 0.0)
                        ),
                        "frame0_model_ms": float(payload.get("frame0_model_ms", 0.0)),
                        "frame0_prompt_model_ms": float(
                            payload.get("frame0_prompt_model_ms", 0.0)
                        ),
                        "track_all_ms": float(payload.get("track_all_ms", 0.0)),
                        "track_all_model_ms": float(
                            payload.get("track_all_model_ms", 0.0)
                        ),
                        "tail_model_ms": float(payload.get("tail_model_ms", 0.0)),
                        "tail_runtime_wall_ms": float(
                            payload.get(
                                "tail_runtime_wall_ms",
                                payload.get("track_all_ms", 0.0),
                            )
                        ),
                        "tail_visible_model_ms": float(
                            payload.get(
                                "tail_visible_model_ms",
                                payload.get("track_all_model_ms", 0.0),
                            )
                        ),
                        "tail_preencode_required_model_ms": float(
                            payload.get(
                                "tail_preencode_required_model_ms",
                                payload.get("preencode_state_create_total_ms", 0.0)
                                + payload.get("preencode_total_ms", 0.0),
                            )
                        ),
                        "tail_required_model_ms": float(
                            payload.get(
                                "tail_required_model_ms",
                                payload.get("tail_model_ms", 0.0),
                            )
                        ),
                        "tail_encode_required_total_ms": float(
                            payload.get(
                                "tail_encode_required_total_ms",
                                payload.get("track_encode_total_ms", 0.0)
                                + payload.get("preencode_total_ms", 0.0),
                            )
                        ),
                        "required_tail_state_create_ms": float(
                            payload.get(
                                "required_tail_state_create_ms",
                                payload.get("preencode_state_create_total_ms", 0.0),
                            )
                        ),
                        "required_tail_encode_ms": float(
                            payload.get(
                                "required_tail_encode_ms",
                                payload.get(
                                    "tail_encode_required_total_ms",
                                    payload.get("track_encode_total_ms", 0.0)
                                    + payload.get("preencode_total_ms", 0.0),
                                ),
                            )
                        ),
                        "required_tail_propagate_ms": float(
                            payload.get(
                                "required_tail_propagate_ms",
                                payload.get("track_propagate_total_ms", 0.0),
                            )
                        ),
                        "required_tail_unattributed_ms": float(
                            payload.get(
                                "required_tail_unattributed_ms",
                                payload.get("track_unattributed_total_ms", 0.0),
                            )
                        ),
                        "required_output_artifact_ms": float(
                            payload.get(
                                "required_output_artifact_ms",
                                payload.get("artifact_ms", 0.0),
                            )
                        ),
                        "tail_inline_encode_ms": float(
                            payload.get(
                                "tail_inline_encode_ms",
                                payload.get("track_encode_total_ms", 0.0),
                            )
                        ),
                        "tail_preencode_encode_ms": float(
                            payload.get(
                                "tail_preencode_encode_ms",
                                payload.get("preencode_total_ms", 0.0),
                            )
                        ),
                        "tail_inline_propagate_ms": float(
                            payload.get(
                                "tail_inline_propagate_ms",
                                payload.get("track_propagate_total_ms", 0.0),
                            )
                        ),
                        "tail_preencoded_propagate_ms": float(
                            payload.get("tail_preencoded_propagate_ms", 0.0)
                        ),
                        "frame0_encode_graph_compute_ms": float(
                            payload.get("frame0_encode_graph_compute_ms", 0.0)
                        ),
                        "tail_inline_encode_graph_compute_ms": float(
                            payload.get("tail_inline_encode_graph_compute_ms", 0.0)
                        ),
                        "tail_preencode_encode_graph_compute_ms": float(
                            payload.get("tail_preencode_encode_graph_compute_ms", 0.0)
                        ),
                        "tail_encode_graph_compute_ms": float(
                            payload.get("tail_encode_graph_compute_ms", 0.0)
                        ),
                        "tail_propagate_graph_compute_ms": float(
                            payload.get("tail_propagate_graph_compute_ms", 0.0)
                        ),
                        "tail_propagate_cache_compute_ms": float(
                            payload.get("tail_propagate_cache_compute_ms", 0.0)
                        ),
                        "tail_propagate_input_prompt_upload_ms": float(
                            payload.get("tail_propagate_input_prompt_upload_ms", 0.0)
                        ),
                        "tail_propagate_input_rope_upload_ms": float(
                            payload.get("tail_propagate_input_rope_upload_ms", 0.0)
                        ),
                        "tail_propagate_input_memory_upload_ms": float(
                            payload.get("tail_propagate_input_memory_upload_ms", 0.0)
                        ),
                        "tail_propagate_input_feature_upload_ms": float(
                            payload.get("tail_propagate_input_feature_upload_ms", 0.0)
                        ),
                        "tail_propagate_input_constant_upload_ms": float(
                            payload.get("tail_propagate_input_constant_upload_ms", 0.0)
                        ),
                        "tail_propagate_input_sparse_upload_ms": float(
                            payload.get("tail_propagate_input_sparse_upload_ms", 0.0)
                        ),
                        "required_core_compute_ms": float(
                            payload.get("required_core_compute_ms", 0.0)
                        ),
                        "required_non_core_compute_ms": float(
                            payload.get("required_non_core_compute_ms", 0.0)
                        ),
                        "required_core_compute_fraction": float(
                            payload.get("required_core_compute_fraction", 0.0)
                        ),
                        "tail_encode_graph_compute_fraction": float(
                            payload.get("tail_encode_graph_compute_fraction", 0.0)
                        ),
                        "track_encode_ms": float(payload.get("track_encode_ms", 0.0)),
                        "track_encode_total_ms": float(
                            payload.get("track_encode_total_ms", 0.0)
                        ),
                        "track_propagate_ms": float(
                            payload.get("track_propagate_ms", 0.0)
                        ),
                        "track_propagate_total_ms": float(
                            payload.get("track_propagate_total_ms", 0.0)
                        ),
                        "track_unattributed_ms": float(
                            payload.get("track_unattributed_ms", 0.0)
                        ),
                        "track_unattributed_total_ms": float(
                            payload.get("track_unattributed_total_ms", 0.0)
                        ),
                        "track_artifact_ms": float(
                            payload.get("track_artifact_ms", 0.0)
                        ),
                        "track_artifact_total_ms": float(
                            payload.get("track_artifact_total_ms", 0.0)
                        ),
                        "preencode_avg_ms": float(payload.get("preencode_avg_ms", 0.0)),
                        "preencode_total_ms": float(
                            payload.get("preencode_total_ms", 0.0)
                        ),
                        "preencode_state_create_avg_ms": float(
                            payload.get("preencode_state_create_avg_ms", 0.0)
                        ),
                        "preencode_state_create_total_ms": float(
                            payload.get("preencode_state_create_total_ms", 0.0)
                        ),
                        "track_frames": int(payload.get("track_frames", 0)),
                        "track_all_frames": int(payload.get("track_all_frames", 0)),
                        "track_split_frames": int(payload.get("track_split_frames", 0)),
                        "preencoded_frames": int(payload.get("preencoded_frames", 0)),
                        "frame0_candidates": int(payload.get("frame0_candidates", 0)),
                        "frame0_added_instances": int(
                            payload.get("frame0_added_instances", 0)
                        ),
                        "text_init_selected_only": bool(
                            payload.get("text_init_selected_only", False)
                        ),
                        "rss_mib": float(payload["rss_mib"]),
                        "detections": int(payload["detections"]),
                    }
                )
            else:
                row["error"] = str(payload.get("error", "benchmark failed"))
            json_rows.append(row)
            continue

        match = CPP_ROW_RE.match(line)
        if match:
            row = match.groupdict()
            rows.append(
                {
                    "model": row["model"].strip(),
                    "family": model_key(row["model"].strip()),
                    "precision": precision(row["model"].strip()),
                    "size": row["size"].strip(),
                    "backend": row["backend"].strip(),
                    "ok": True,
                    "load_ms": float(row["load"]),
                    "init_ms": float(row["init"]),
                    "track_ms": float(row["track"]),
                    "p50_ms": float(row["p50"]),
                    "p95_ms": float(row["p95"]),
                    "total_ms": float(row["total"]),
                    "rss_mib": float(row["rss"]),
                    "detections": int(row["det"]),
                }
            )
            continue

        fail_match = CPP_FAIL_ROW_RE.match(line)
        if fail_match:
            row = fail_match.groupdict()
            model = row["model"].strip()
            rows.append(
                {
                    "model": model,
                    "family": model_key(model),
                    "precision": precision(model),
                    "size": row["size"].strip(),
                    "backend": row["backend"].strip(),
                    "ok": False,
                    "error": row["error"].strip(),
                }
            )
    return json_rows if json_rows else rows


def run_cpp(
    args: argparse.Namespace, out_dir: Path, frame_dir: Path
) -> list[dict[str, Any]]:
    run(["just", "build"], cwd=args.repo, log_path=out_dir / "cpp-build.log")
    benchmark = args.repo / "build/xmake-release-cuda/examples/sam3_benchmark"
    cpp_frame_timing_dir: Path | None = args.cpp_frame_timing_dir
    if cpp_frame_timing_dir is not None:
        if not cpp_frame_timing_dir.is_absolute():
            cpp_frame_timing_dir = out_dir / cpp_frame_timing_dir
        cpp_frame_timing_dir.mkdir(parents=True, exist_ok=True)
    filter_tokens = [token for token in args.filter if token]
    benchmark_filter = ""
    if filter_tokens:
        model_names = [path.name for path in args.models_dir.glob("*.ggml")]
        token_counts = [
            (sum(1 for name in model_names if token in name), index, token)
            for index, token in enumerate(filter_tokens)
        ]
        matching_counts = [item for item in token_counts if item[0] > 0]
        benchmark_filter = min(matching_counts or token_counts)[2]
    cmd = [
        str(benchmark),
        "--models-dir",
        str(args.models_dir),
        "--video",
        str(args.video),
        "--frame-dir",
        str(frame_dir),
        "--gpu-only",
        "--quiet",
        "--n-frames",
        str(args.frames),
        "--point-x",
        str(args.point_x),
        "--point-y",
        str(args.point_y),
        "--text-prompt",
        args.text_prompt,
        "--timed-start-frame",
        str(args.timed_start_frame),
    ]
    if not args.cpp_mask_output:
        cmd.append("--bbox-only")
    if args.cpp_warmup_runs > 0:
        cmd.extend(["--warmup-runs", str(args.cpp_warmup_runs)])
    if args.cpp_preencode_track_frames:
        cmd.append("--preencode-track-frames")
    if args.cpp_preencode_cached_tail_frames:
        cmd.append("--preencode-cached-tail-frames")
    if args.cpp_text_init_selected_only:
        cmd.append("--text-init-selected-only")
    if args.cpp_no_output_artifacts:
        cmd.append("--no-output-artifacts")
    if args.multimask:
        cmd.append("--multimask")
    if args.encode_img_size > 0:
        cmd.extend(["--encode-img-size", str(args.encode_img_size)])
    if benchmark_filter:
        cmd.extend(["--filter", benchmark_filter])
    cpp_env = os.environ.copy()
    cpp_env.pop("SAM3_BENCH_PREENCODE_TRACK_FRAMES", None)
    cpp_env.pop("SAM3_BENCH_PREENCODE_CACHED_TAIL_FRAMES", None)
    cpp_env_overrides = parse_key_value_overrides(args.cpp_env, option_name="--cpp-env")
    cpp_env.update(cpp_env_overrides)
    track_scope = cpp_track_scope(args)
    repeated_rows: list[dict[str, Any]] = []
    for run_index in range(1, args.repeats + 1):
        run_cmd = list(cmd)
        frame_timing_path = None
        if cpp_frame_timing_dir is not None:
            frame_timing_path = (
                cpp_frame_timing_dir / f"cpp-frame-timing-{run_index:02d}.jsonl"
            )
            run_cmd.extend(["--output-frame-timing-jsonl", str(frame_timing_path)])
        text = run(
            run_cmd,
            cwd=args.repo,
            log_path=out_dir / f"cpp-benchmark-{run_index:02d}.log",
            env=cpp_env,
        )
        rows = parse_cpp_table(text)
        for row in rows:
            row["run"] = run_index
            row["track_scope"] = track_scope
            row["preencoded_track_frames"] = args.cpp_preencode_track_frames
            row["preencoded_cached_tail_frames"] = args.cpp_preencode_cached_tail_frames
            row["text_init_selected_only"] = args.cpp_text_init_selected_only
            row["timed_start_frame"] = args.timed_start_frame
            row["warmup_runs"] = args.cpp_warmup_runs
            row["cpp_variant"] = args.cpp_variant
            row["cpp_env_overrides"] = cpp_env_overrides
            row["cpp_output_artifacts"] = not args.cpp_no_output_artifacts
            if frame_timing_path is not None:
                try:
                    row["cpp_frame_timing_jsonl"] = str(
                        frame_timing_path.relative_to(out_dir)
                    )
                except ValueError:
                    row["cpp_frame_timing_jsonl"] = str(frame_timing_path)
            repeated_rows.append(row)
    rows = repeated_rows
    if len(filter_tokens) > 1:
        rows = [
            row for row in rows if all(token in row["model"] for token in filter_tokens)
        ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["family"], row["precision"]), []).append(
            row
        )
    rows = []
    for (model, family, model_precision), group in grouped.items():
        ok_rows = [row for row in group if row.get("ok", True)]
        if not ok_rows:
            row = dict(group[-1])
            row["runs"] = group
            rows.append(row)
            continue
        track = [float(row["track_ms"]) for row in ok_rows]
        row = dict(ok_rows[-1])
        row["track_ms"] = statistics.mean(track)
        for key in (
            "model_load_ms",
            "p50_ms",
            "p95_ms",
            "total_ms",
            "session_ms",
            "input_load_ms",
            "input_directory_scan_ms",
            "input_sort_ms",
            "input_frame_decode_ms",
            "input_frame_store_ms",
            "input_remainder_ms",
            "model_e2e_ms",
            "required_e2e_ms",
            "cold_required_e2e_ms",
            "input_plus_session_ms",
            "required_accounted_ms",
            "required_remainder_ms",
            "required_accounting_delta_ms",
            "model_accounting_delta_ms",
            "model_accounted_ms",
            "session_overhead_ms",
            "model_remainder_ms",
            "state_create_ms",
            "tracker_create_ms",
            "session_setup_ms",
            "artifact_ms",
            "frame0_artifact_ms",
            "frame0_model_ms",
            "frame0_prompt_model_ms",
            "frame0_encode_ms",
            "frame0_prompt_ms",
            "frame0_segment_ms",
            "frame0_tracker_add_ms",
            "frame0_tracker_mask_prepare_ms",
            "frame0_tracker_memory_encode_ms",
            "frame0_tracker_obj_ptr_ms",
            "frame0_tracker_store_ms",
            "frame0_tracker_unattributed_ms",
            "frame0_prompt_unattributed_ms",
            "track_all_ms",
            "track_all_model_ms",
            "tail_model_ms",
            "tail_runtime_wall_ms",
            "tail_visible_model_ms",
            "tail_preencode_required_model_ms",
            "tail_required_model_ms",
            "tail_encode_required_total_ms",
            "required_tail_state_create_ms",
            "required_tail_encode_ms",
            "required_tail_propagate_ms",
            "required_tail_unattributed_ms",
            "required_output_artifact_ms",
            "tail_inline_encode_ms",
            "tail_preencode_encode_ms",
            "tail_inline_propagate_ms",
            "tail_preencoded_propagate_ms",
            "frame0_encode_graph_compute_ms",
            "tail_inline_encode_graph_compute_ms",
            "tail_preencode_encode_graph_compute_ms",
            "tail_encode_graph_compute_ms",
            "tail_propagate_graph_compute_ms",
            "tail_propagate_cache_compute_ms",
            "tail_propagate_input_prompt_upload_ms",
            "tail_propagate_input_rope_upload_ms",
            "tail_propagate_input_memory_upload_ms",
            "tail_propagate_input_feature_upload_ms",
            "tail_propagate_input_constant_upload_ms",
            "tail_propagate_input_sparse_upload_ms",
            "required_core_compute_ms",
            "required_non_core_compute_ms",
            "required_core_compute_fraction",
            "tail_encode_graph_compute_fraction",
            "track_encode_ms",
            "track_encode_total_ms",
            "track_propagate_ms",
            "track_propagate_total_ms",
            "track_unattributed_ms",
            "track_unattributed_total_ms",
            "track_artifact_ms",
            "track_artifact_total_ms",
            "preencode_avg_ms",
            "preencode_total_ms",
            "preencode_state_create_avg_ms",
            "preencode_state_create_total_ms",
        ):
            values = [float(item[key]) for item in ok_rows if key in item]
            if values:
                row[key] = statistics.mean(values)
        row["run"] = None
        row["model"] = model
        row["family"] = family
        row["precision"] = model_precision
        row["has_input_load_ms"] = all(
            bool(item.get("has_input_load_ms")) for item in ok_rows
        )
        row["track_ms_stats"] = summarize_values(track)
        for key in (
            "model_load_ms",
            "session_ms",
            "input_load_ms",
            "input_directory_scan_ms",
            "input_sort_ms",
            "input_frame_decode_ms",
            "input_frame_store_ms",
            "input_remainder_ms",
            "model_e2e_ms",
            "required_e2e_ms",
            "cold_required_e2e_ms",
            "input_plus_session_ms",
            "required_accounted_ms",
            "required_remainder_ms",
            "required_accounting_delta_ms",
            "model_accounting_delta_ms",
            "model_accounted_ms",
            "session_overhead_ms",
            "model_remainder_ms",
            "state_create_ms",
            "tracker_create_ms",
            "session_setup_ms",
            "artifact_ms",
            "frame0_encode_ms",
            "frame0_prompt_ms",
            "frame0_model_ms",
            "frame0_prompt_model_ms",
            "frame0_segment_ms",
            "frame0_tracker_add_ms",
            "frame0_tracker_mask_prepare_ms",
            "frame0_tracker_memory_encode_ms",
            "frame0_tracker_obj_ptr_ms",
            "frame0_tracker_store_ms",
            "frame0_tracker_unattributed_ms",
            "frame0_prompt_unattributed_ms",
            "frame0_artifact_ms",
            "track_encode_ms",
            "track_encode_total_ms",
            "track_propagate_ms",
            "track_propagate_total_ms",
            "track_unattributed_ms",
            "track_unattributed_total_ms",
            "track_artifact_ms",
            "track_all_model_ms",
            "tail_model_ms",
            "tail_runtime_wall_ms",
            "tail_visible_model_ms",
            "tail_preencode_required_model_ms",
            "tail_required_model_ms",
            "tail_encode_required_total_ms",
            "required_tail_state_create_ms",
            "required_tail_encode_ms",
            "required_tail_propagate_ms",
            "required_tail_unattributed_ms",
            "required_output_artifact_ms",
            "tail_inline_encode_ms",
            "tail_preencode_encode_ms",
            "tail_inline_propagate_ms",
            "tail_preencoded_propagate_ms",
            "frame0_encode_graph_compute_ms",
            "tail_inline_encode_graph_compute_ms",
            "tail_preencode_encode_graph_compute_ms",
            "tail_encode_graph_compute_ms",
            "tail_propagate_graph_compute_ms",
            "tail_propagate_cache_compute_ms",
            "tail_propagate_input_prompt_upload_ms",
            "tail_propagate_input_rope_upload_ms",
            "tail_propagate_input_memory_upload_ms",
            "tail_propagate_input_feature_upload_ms",
            "tail_propagate_input_constant_upload_ms",
            "tail_propagate_input_sparse_upload_ms",
            "required_core_compute_ms",
            "required_non_core_compute_ms",
            "required_core_compute_fraction",
            "tail_encode_graph_compute_fraction",
            "preencode_avg_ms",
            "preencode_state_create_avg_ms",
        ):
            values = [float(item[key]) for item in ok_rows if key in item]
            if values:
                row[f"{key}_stats"] = summarize_values(values)
        row["runs"] = group
        row["track_scope"] = track_scope
        row["preencoded_track_frames"] = args.cpp_preencode_track_frames
        row["preencoded_cached_tail_frames"] = args.cpp_preencode_cached_tail_frames
        row["text_init_selected_only"] = args.cpp_text_init_selected_only
        row["timed_start_frame"] = args.timed_start_frame
        row["warmup_runs"] = args.cpp_warmup_runs
        row["cpp_variant"] = args.cpp_variant
        row["cpp_env_overrides"] = cpp_env_overrides
        row["cpp_output_artifacts"] = not args.cpp_no_output_artifacts
        frame_timing_paths = [
            str(item["cpp_frame_timing_jsonl"])
            for item in ok_rows
            if item.get("cpp_frame_timing_jsonl")
        ]
        if frame_timing_paths:
            row["cpp_frame_timing_jsonl"] = frame_timing_paths[-1]
            row["cpp_frame_timing_jsonl_runs"] = frame_timing_paths
            frame_timing_aggregate = aggregate_cpp_frame_timing_runs(
                out_dir, frame_timing_paths
            )
            if frame_timing_aggregate is not None:
                row["cpp_frame_timing_aggregate_ms"] = frame_timing_aggregate
        add_cpp_compute_split(row)
        rows.append(row)
    (out_dir / "cpp-results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


def extract_frames(
    video: Path, frame_dir: Path, frames: int, frame_format: str
) -> None:
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-frames:v",
        str(frames),
    ]
    if frame_format == "jpg":
        cmd.extend(["-q:v", "2"])
    elif frame_format == "png":
        cmd.extend(["-pix_fmt", "rgb24"])
    else:
        raise ValueError(f"unsupported frame format: {frame_format}")
    cmd.append(str(frame_dir / f"%05d.{frame_format}"))
    run(cmd, cwd=frame_dir.parent)


def decoded_frame_size(frame_dir: Path, frame_format: str) -> tuple[int, int]:
    first = frame_dir / f"00001.{frame_format}"
    with Image.open(first) as img:
        return img.size


def cpp_effective_encode_size(
    row: dict[str, Any], requested_encode_size: int
) -> int | None:
    if requested_encode_size > 0:
        return requested_encode_size
    if row["family"].startswith("sam2.1_hiera_"):
        return 1024
    if row["family"] in {"sam3", "sam3.1"} or row["family"].startswith("sam3-visual"):
        return 1008
    return None


def validate_reused_python_results(
    py_rows: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    requested_versions = {
        version
        for version in (args.python_sam3_version or [])
        if version != SAM3_PYTHON_NONE
    }
    mismatches: list[str] = []
    for index, row in enumerate(py_rows, start=1):
        label = row.get("family") or row.get("sam3_version") or f"row {index}"
        expected_fields = {
            "frames": args.frames,
            "timed_start_frame": args.timed_start_frame,
            "python_dtype": args.python_dtype,
            "tf32_policy": args.tf32_policy,
        }
        if args.encode_img_size > 0:
            expected_fields["image_size"] = args.encode_img_size
        for key, expected in expected_fields.items():
            actual = row.get(key)
            if actual != expected:
                mismatches.append(f"{label}: {key}={actual!r}, expected {expected!r}")
        sam3_version = row.get("sam3_version")
        if (
            requested_versions
            and sam3_version
            and sam3_version not in requested_versions
        ):
            mismatches.append(
                f"{label}: sam3_version={sam3_version!r}, expected one of "
                f"{sorted(requested_versions)!r}"
            )
    if mismatches:
        preview = "\n  ".join(mismatches[:8])
        extra = "" if len(mismatches) <= 8 else f"\n  ... {len(mismatches) - 8} more"
        raise SystemExit(
            "--python-results contract mismatch; regenerate Python results or match the "
            f"current benchmark arguments:\n  {preview}{extra}"
        )


def run_python_sam3(
    args: argparse.Namespace, out_dir: Path, frame_dir: Path
) -> list[dict[str, Any]]:
    sam3_repo = Path(args.sam3_repo)
    if not sam3_repo.exists():
        return []
    requested_versions = args.python_sam3_version or ["sam3.1"]
    versions = [
        version for version in requested_versions if version != SAM3_PYTHON_NONE
    ]
    if not versions:
        return []
    script = args.repo / "scripts" / "run_sam3_python_baseline.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam3_repo)
    rows: list[dict[str, Any]] = []
    for version in versions:
        try:
            text = run(
                uv_run_python_script(
                    sam3_repo,
                    script,
                    [
                        str(sam3_repo),
                        str(frame_dir),
                        args.text_prompt,
                        str(args.frames),
                        args.python_dtype,
                        args.tf32_policy,
                        str(args.repeats),
                        version,
                        "1" if args.python_sam3_use_fa3 else "0",
                        "1" if args.python_sam3_compile else "0",
                        str(args.timed_start_frame),
                        python_sam3_track_scope(args.timed_start_frame),
                    ],
                    SAM3_UV_DEPS,
                ),
                cwd=args.repo,
                log_path=out_dir / f"python-{version}.log",
                env=env,
            )
        except Exception as exc:
            rows.append(
                {
                    "family": version,
                    "backend": f"PyTorch CUDA {args.python_dtype}",
                    "error": str(exc),
                }
            )
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                rows.append(json.loads(line))
                break
        else:
            rows.append(
                {
                    "family": version,
                    "backend": f"PyTorch CUDA {args.python_dtype}",
                    "error": f"could not parse Python {version} result",
                }
            )
    return rows


def run_python_sam2(
    args: argparse.Namespace, out_dir: Path, frame_dir: Path
) -> list[dict[str, Any]]:
    sam2_repo = Path(args.sam2_repo)
    checkpoints = {
        "sam2.1_hiera_tiny": (
            args.sam2_tiny_checkpoint,
            "configs/sam2.1/sam2.1_hiera_t.yaml",
        ),
        "sam2.1_hiera_small": (
            args.sam2_small_checkpoint,
            "configs/sam2.1/sam2.1_hiera_s.yaml",
        ),
        "sam2.1_hiera_base_plus": (
            args.sam2_base_plus_checkpoint,
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
        ),
        "sam2.1_hiera_large": (
            args.sam2_large_checkpoint,
            "configs/sam2.1/sam2.1_hiera_l.yaml",
        ),
    }
    rows: list[dict[str, Any]] = []
    if not sam2_repo.exists():
        return rows
    code = r"""
import json, sys, time
import numpy as np
import torch

sam2_repo, checkpoint, cfg, video_dir, point_x, point_y, image_size, frames, python_dtype, tf32_policy, repeats = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9], sys.argv[10], int(sys.argv[11])
sys.path.insert(0, sam2_repo)
if python_dtype == "bf16":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
elif python_dtype == "fp16":
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
allow_tf32 = tf32_policy == "on"
def apply_tf32_policy():
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
apply_tf32_policy()
from sam2.build_sam import build_sam2_video_predictor

overrides = []
if image_size > 0:
    overrides.append(f"model.image_size={image_size}")

predictor = build_sam2_video_predictor(
    cfg,
    checkpoint,
    device="cuda",
    vos_optimized=False,
    hydra_overrides_extra=overrides,
)
apply_tf32_policy()

def run_once(measure=False):
    state = predictor.init_state(video_path=video_dir)
    points = np.array([[point_x, point_y]], dtype=np.float32)
    labels = np.array([1], np.int32)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, points=points, labels=labels)
    count = 0
    measured_ms = 0.0
    prev = time.perf_counter()
    for out_frame_idx, _, _ in predictor.propagate_in_video(state):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if 0 < out_frame_idx < frames:
            measured_ms += (now - prev) * 1000.0
            count += 1
        prev = now
        if out_frame_idx + 1 >= frames:
            break
    torch.cuda.synchronize()
    if measure:
        return count, measured_ms
    return count, 0.0

for _ in range(2):
    run_once()

torch.cuda.reset_peak_memory_stats()
runs = []
for index in range(repeats):
    count, measured_ms = run_once(measure=True)
    runs.append({"run": index + 1, "track_frames": count, "track_ms": measured_ms / max(count, 1), "total_ms": measured_ms})
track_values = [row["track_ms"] for row in runs]
total_values = [row["total_ms"] for row in runs]
print(json.dumps({
    "backend": f"PyTorch CUDA {python_dtype}",
    "python_dtype": python_dtype,
    "tf32_policy": tf32_policy,
    "track_scope": "frame1_plus_encode_propagate_prompt_excluded",
    "warmup_runs": 2,
    "torch_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    "torch_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    "torch_float32_matmul_precision": torch.get_float32_matmul_precision() if hasattr(torch, "get_float32_matmul_precision") else None,
    "frames": frames,
    "track_frames": runs[-1]["track_frames"],
    "track_ms": sum(track_values) / len(track_values),
    "total_ms": sum(total_values) / len(total_values),
    "track_ms_stats": {
        "n": len(track_values),
        "mean": sum(track_values) / len(track_values),
        "median": sorted(track_values)[len(track_values) // 2],
        "min": min(track_values),
        "max": max(track_values),
    },
    "runs": runs,
    "rss_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
    "image_size": image_size if image_size > 0 else 1024,
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam2_repo)
    for family, (checkpoint, cfg) in checkpoints.items():
        if args.python_family and family not in args.python_family:
            continue
        if checkpoint is None or not checkpoint.exists():
            continue
        try:
            text = run(
                uv_run_python(
                    sam2_repo,
                    code,
                    [
                        str(sam2_repo),
                        str(checkpoint),
                        cfg,
                        str(frame_dir),
                        str(args.point_x),
                        str(args.point_y),
                        str(args.encode_img_size),
                        str(args.frames),
                        args.python_dtype,
                        args.tf32_policy,
                        str(args.repeats),
                    ],
                    SAM2_UV_DEPS,
                ),
                cwd=args.repo,
                log_path=out_dir / f"python-{family}.log",
                env=env,
            )
        except Exception as exc:
            rows.append(
                {"family": family, "backend": "PyTorch CUDA bf16", "error": str(exc)}
            )
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                row = json.loads(line)
                row["family"] = family
                rows.append(row)
                break
    return rows


def should_run_python_sam2(args: argparse.Namespace) -> bool:
    if args.skip_python_sam2:
        return False
    if args.python_family:
        return True
    filter_tokens = [token.lower() for token in args.filter if token]
    if not filter_tokens:
        return True
    sam2_tokens = ("sam2", "sam21", "hiera", "edgetam", "edge_tam")
    if any(any(marker in token for marker in sam2_tokens) for token in filter_tokens):
        return True
    return not all("sam3" in token for token in filter_tokens)


def summarize(
    cpp_rows: list[dict[str, Any]],
    py_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    decoded_width: int,
    decoded_height: int,
    frames: int,
    prompt: str,
    point_x: float,
    point_y: float,
    requested_encode_size: int,
    multimask: bool,
    frame_format: str,
) -> dict[str, Any]:
    py_by_family = {row["family"]: row for row in py_rows}
    py_by_contract: dict[str, dict[str, Any]] = {}
    for row in py_rows:
        family = str(row.get("family", ""))
        py_by_contract.setdefault(comparison_contract_key(family), row)
    comparisons: list[dict[str, Any]] = []
    for row in cpp_rows:
        if not row.get("ok", True):
            continue
        cpp_comparison_contract = comparison_contract_key(str(row["family"]))
        py = py_by_family.get(row["family"])
        exact_family_match = py is not None
        if not py:
            py = py_by_contract.get(cpp_comparison_contract)
        if not py:
            # SAM2 and SAM2.1 naming is already aligned; SAM3 visual has no official
            # separate Python baseline.
            continue
        python_family = str(py.get("family", ""))
        python_comparison_contract = comparison_contract_key(python_family)
        same_comparison_contract_family = (
            cpp_comparison_contract == python_comparison_contract
        )
        if "error" in py:
            comparisons.append(
                {
                    "model": row["model"],
                    "family": row["family"],
                    "python_family": python_family,
                    "exact_family_match": exact_family_match,
                    "comparison_contract_family": cpp_comparison_contract,
                    "same_comparison_contract_family": (
                        same_comparison_contract_family
                    ),
                    "precision": row["precision"],
                    "cpp_track_ms": row["track_ms"],
                    "python_error": py["error"],
                }
            )
            continue
        cpp_encode_size = cpp_effective_encode_size(row, requested_encode_size)
        python_encode_size = py.get("image_size")
        same_encode_size = (
            cpp_encode_size is not None
            and python_encode_size is not None
            and cpp_encode_size == python_encode_size
        )
        same_decoded_source_resolution = True
        same_prompt = True
        same_frame_range = py.get("frames") == frames
        cpp_track_scope = row.get("track_scope", "per_frame_encode_detect_propagate")
        python_track_scope = py.get("track_scope", "per_frame_encode_detect_propagate")
        same_measured_tracking_scope = cpp_track_scope == python_track_scope
        same_timed_start_frame = row.get("timed_start_frame") == py.get(
            "timed_start_frame"
        )
        same_text_init_cardinality = not bool(row.get("text_init_selected_only", False))
        cpp_warmup_runs = int(row.get("warmup_runs", 0))
        python_warmup_runs = int(py.get("warmup_runs", 2))
        same_warmup_policy = cpp_warmup_runs == python_warmup_runs
        comparable_speed_claim = (
            exact_family_match
            and same_decoded_source_resolution
            and same_frame_range
            and same_prompt
            and same_encode_size
            and same_measured_tracking_scope
            and same_timed_start_frame
            and same_warmup_policy
        )
        same_contract_speed_diagnostic = (
            same_comparison_contract_family
            and same_decoded_source_resolution
            and same_frame_range
            and same_prompt
            and same_encode_size
            and same_measured_tracking_scope
            and same_timed_start_frame
            and same_warmup_policy
        )
        track_ratio = py["track_ms"] / row["track_ms"] if row["track_ms"] > 0 else None
        cpp_model_load_ms = row.get("model_load_ms", row.get("load_ms"))
        cpp_session_with_artifacts_ms = row.get("session_ms")
        cpp_has_input_load_ms = bool(row.get("has_input_load_ms", False))
        cpp_input_load_ms = row.get("input_load_ms")
        cpp_input_breakdown_ms = {
            "source": row.get("input_source"),
            "frames": row.get("input_frames"),
            "directory_scan_ms": row.get("input_directory_scan_ms"),
            "sort_ms": row.get("input_sort_ms"),
            "frame_decode_ms": row.get("input_frame_decode_ms"),
            "frame_store_ms": row.get("input_frame_store_ms"),
            "remainder_ms": row.get("input_remainder_ms"),
        }
        cpp_model_e2e_ms = row.get("model_e2e_ms", cpp_session_with_artifacts_ms)
        cpp_required_e2e_ms = row.get("required_e2e_ms")
        if not isinstance(cpp_required_e2e_ms, int | float):
            cpp_required_e2e_ms = (
                cpp_input_load_ms + cpp_model_e2e_ms
                if isinstance(cpp_input_load_ms, int | float)
                and isinstance(cpp_model_e2e_ms, int | float)
                else None
            )
        cpp_cold_required_e2e_ms = row.get("cold_required_e2e_ms")
        if not isinstance(cpp_cold_required_e2e_ms, int | float):
            cpp_cold_required_e2e_ms = (
                cpp_required_e2e_ms + cpp_model_load_ms
                if isinstance(cpp_required_e2e_ms, int | float)
                and isinstance(cpp_model_load_ms, int | float)
                else None
            )
        cpp_input_plus_session_ms = row.get("input_plus_session_ms")
        cpp_required_accounted_ms = row.get("required_accounted_ms")
        cpp_required_remainder_ms = row.get("required_remainder_ms")
        cpp_required_accounting_delta_ms = row.get("required_accounting_delta_ms")
        cpp_model_accounting_delta_ms = row.get("model_accounting_delta_ms")
        cpp_model_accounted_ms = row.get("model_accounted_ms")
        cpp_session_overhead_ms = row.get("session_overhead_ms")
        cpp_model_remainder_ms = row.get("model_remainder_ms")
        python_model_e2e_ms = py.get("model_e2e_ms")
        python_session_e2e_ms = py.get("session_e2e_ms")
        e2e_ratio = (
            python_model_e2e_ms / cpp_model_e2e_ms
            if isinstance(cpp_model_e2e_ms, int | float)
            and isinstance(python_model_e2e_ms, int | float)
            and cpp_model_e2e_ms > 0
            else None
        )
        session_e2e_ratio = (
            python_session_e2e_ms / cpp_required_e2e_ms
            if isinstance(cpp_required_e2e_ms, int | float)
            and isinstance(python_session_e2e_ms, int | float)
            and cpp_required_e2e_ms > 0
            else None
        )
        cpp_artifact_ms = row.get("artifact_ms", 0.0)
        cpp_frame0_artifact_ms = row.get("frame0_artifact_ms", 0.0)
        cpp_track_artifact_total_ms = row.get("track_artifact_total_ms", 0.0)
        cpp_frame0_prompt_model_ms = (
            max(0.0, row.get("frame0_prompt_ms", 0.0) - cpp_frame0_artifact_ms)
            if isinstance(cpp_frame0_artifact_ms, int | float)
            else row.get("frame0_prompt_ms")
        )
        if isinstance(row.get("frame0_prompt_model_ms"), int | float):
            cpp_frame0_prompt_model_ms = row.get("frame0_prompt_model_ms")
        cpp_frame0_model_ms = row.get("frame0_model_ms")
        if not isinstance(cpp_frame0_model_ms, int | float):
            cpp_frame0_model_ms = (
                row.get("frame0_encode_ms", 0.0) + cpp_frame0_prompt_model_ms
                if isinstance(row.get("frame0_encode_ms"), int | float)
                and isinstance(cpp_frame0_prompt_model_ms, int | float)
                else None
            )
        cpp_track_all_model_ms = (
            max(0.0, row.get("track_all_ms", 0.0) - cpp_track_artifact_total_ms)
            if isinstance(cpp_track_artifact_total_ms, int | float)
            else row.get("track_all_ms")
        )
        if isinstance(row.get("track_all_model_ms"), int | float):
            cpp_track_all_model_ms = row.get("track_all_model_ms")
        cpp_tail_model_ms = row.get("tail_model_ms")
        cpp_tail_runtime_wall_ms = row.get(
            "tail_runtime_wall_ms", row.get("track_all_ms")
        )
        cpp_tail_visible_model_ms = row.get(
            "tail_visible_model_ms", cpp_track_all_model_ms
        )
        cpp_tail_preencode_required_model_ms = row.get(
            "tail_preencode_required_model_ms",
            (
                row.get("preencode_state_create_total_ms", 0.0)
                + row.get("preencode_total_ms", 0.0)
            ),
        )
        cpp_tail_required_model_ms = row.get(
            "tail_required_model_ms", cpp_tail_model_ms
        )
        cpp_tail_encode_required_total_ms = row.get("tail_encode_required_total_ms")
        cpp_frame_timing_aggregate_ms = row.get("cpp_frame_timing_aggregate_ms")

        def frame_metric_or_row(row_key: str, *path: str) -> float | None:
            value = nested_number(cpp_frame_timing_aggregate_ms, *path)
            if value is not None:
                return value
            row_value = row.get(row_key)
            return float(row_value) if isinstance(row_value, int | float) else None

        cpp_frame0_graph_compute_ms = frame_metric_or_row(
            "frame0_encode_graph_compute_ms",
            "mean",
            "frame0_ms",
            "encode_graph_compute_ms",
        )
        cpp_frame0_encode_preprocess_ms = frame_metric_or_row(
            "frame0_encode_preprocess_ms",
            "mean",
            "frame0_ms",
            "encode_preprocess_ms",
        )
        cpp_frame0_encode_graph_build_ms = frame_metric_or_row(
            "frame0_encode_graph_build_ms",
            "mean",
            "frame0_ms",
            "encode_graph_build_ms",
        )
        cpp_frame0_encode_graph_alloc_ms = frame_metric_or_row(
            "frame0_encode_graph_alloc_ms",
            "mean",
            "frame0_ms",
            "encode_graph_alloc_ms",
        )
        cpp_frame0_encode_input_upload_ms = frame_metric_or_row(
            "frame0_encode_input_upload_ms",
            "mean",
            "frame0_ms",
            "encode_input_upload_ms",
        )
        cpp_frame0_encode_state_update_ms = frame_metric_or_row(
            "frame0_encode_state_update_ms",
            "mean",
            "frame0_ms",
            "encode_state_update_ms",
        )
        cpp_frame0_encode_pe_build_ms = frame_metric_or_row(
            "frame0_encode_pe_build_ms",
            "mean",
            "frame0_ms",
            "encode_pe_build_ms",
        )
        cpp_frame0_encode_remainder_ms = frame_metric_or_row(
            "frame0_encode_remainder_ms",
            "mean",
            "frame0_ms",
            "encode_remainder_ms",
        )
        cpp_tail_encode_graph_compute_ms = frame_metric_or_row(
            "tail_encode_graph_compute_ms",
            "mean",
            "tail_all_required_ms",
            "encode_graph_compute_ms",
        )
        cpp_tail_encode_preprocess_ms = frame_metric_or_row(
            "tail_encode_preprocess_ms",
            "mean",
            "tail_all_required_ms",
            "encode_preprocess_ms",
        )
        cpp_tail_encode_graph_build_ms = frame_metric_or_row(
            "tail_encode_graph_build_ms",
            "mean",
            "tail_all_required_ms",
            "encode_graph_build_ms",
        )
        cpp_tail_encode_graph_alloc_ms = frame_metric_or_row(
            "tail_encode_graph_alloc_ms",
            "mean",
            "tail_all_required_ms",
            "encode_graph_alloc_ms",
        )
        cpp_tail_encode_input_upload_ms = frame_metric_or_row(
            "tail_encode_input_upload_ms",
            "mean",
            "tail_all_required_ms",
            "encode_input_upload_ms",
        )
        cpp_tail_encode_state_update_ms = frame_metric_or_row(
            "tail_encode_state_update_ms",
            "mean",
            "tail_all_required_ms",
            "encode_state_update_ms",
        )
        cpp_tail_encode_pe_build_ms = frame_metric_or_row(
            "tail_encode_pe_build_ms",
            "mean",
            "tail_all_required_ms",
            "encode_pe_build_ms",
        )
        cpp_tail_encode_remainder_ms = frame_metric_or_row(
            "tail_encode_remainder_ms",
            "mean",
            "tail_all_required_ms",
            "encode_remainder_ms",
        )
        cpp_tail_inline_encode_ms = frame_metric_or_row(
            "tail_inline_encode_ms",
            "mean",
            "tail_inline_required_ms",
            "inline_encode_ms",
        )
        cpp_tail_inline_encode_graph_compute_ms = frame_metric_or_row(
            "tail_inline_encode_graph_compute_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_graph_compute_ms",
        )
        cpp_tail_inline_encode_preprocess_ms = frame_metric_or_row(
            "tail_inline_encode_preprocess_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_preprocess_ms",
        )
        cpp_tail_inline_encode_graph_build_ms = frame_metric_or_row(
            "tail_inline_encode_graph_build_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_graph_build_ms",
        )
        cpp_tail_inline_encode_graph_alloc_ms = frame_metric_or_row(
            "tail_inline_encode_graph_alloc_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_graph_alloc_ms",
        )
        cpp_tail_inline_encode_input_upload_ms = frame_metric_or_row(
            "tail_inline_encode_input_upload_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_input_upload_ms",
        )
        cpp_tail_inline_encode_state_update_ms = frame_metric_or_row(
            "tail_inline_encode_state_update_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_state_update_ms",
        )
        cpp_tail_inline_encode_pe_build_ms = frame_metric_or_row(
            "tail_inline_encode_pe_build_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_pe_build_ms",
        )
        cpp_tail_inline_encode_remainder_ms = frame_metric_or_row(
            "tail_inline_encode_remainder_ms",
            "mean",
            "tail_inline_required_ms",
            "encode_remainder_ms",
        )
        cpp_tail_inline_propagate_ms = frame_metric_or_row(
            "tail_inline_propagate_ms",
            "mean",
            "tail_inline_required_ms",
            "propagate_ms",
        )
        cpp_tail_preencode_state_create_ms = frame_metric_or_row(
            "tail_preencode_state_create_ms",
            "mean",
            "tail_preencoded_required_ms",
            "preencode_state_create_ms",
        )
        cpp_tail_preencode_encode_ms = frame_metric_or_row(
            "tail_preencode_encode_ms",
            "mean",
            "tail_preencoded_required_ms",
            "preencode_ms",
        )
        cpp_tail_preencode_encode_graph_compute_ms = frame_metric_or_row(
            "tail_preencode_encode_graph_compute_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_graph_compute_ms",
        )
        cpp_tail_preencode_encode_preprocess_ms = frame_metric_or_row(
            "tail_preencode_encode_preprocess_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_preprocess_ms",
        )
        cpp_tail_preencode_encode_graph_build_ms = frame_metric_or_row(
            "tail_preencode_encode_graph_build_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_graph_build_ms",
        )
        cpp_tail_preencode_encode_graph_alloc_ms = frame_metric_or_row(
            "tail_preencode_encode_graph_alloc_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_graph_alloc_ms",
        )
        cpp_tail_preencode_encode_input_upload_ms = frame_metric_or_row(
            "tail_preencode_encode_input_upload_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_input_upload_ms",
        )
        cpp_tail_preencode_encode_state_update_ms = frame_metric_or_row(
            "tail_preencode_encode_state_update_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_state_update_ms",
        )
        cpp_tail_preencode_encode_pe_build_ms = frame_metric_or_row(
            "tail_preencode_encode_pe_build_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_pe_build_ms",
        )
        cpp_tail_preencode_encode_remainder_ms = frame_metric_or_row(
            "tail_preencode_encode_remainder_ms",
            "mean",
            "tail_preencoded_required_ms",
            "encode_remainder_ms",
        )
        cpp_tail_preencoded_propagate_ms = frame_metric_or_row(
            "tail_preencoded_propagate_ms",
            "mean",
            "tail_preencoded_required_ms",
            "propagate_ms",
        )
        cpp_tail_propagate_graph_compute_ms = frame_metric_or_row(
            "tail_propagate_graph_compute_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_graph_compute_ms",
        )
        cpp_tail_propagate_graph_build_ms = frame_metric_or_row(
            "tail_propagate_graph_build_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_graph_build_ms",
        )
        cpp_tail_propagate_graph_alloc_ms = frame_metric_or_row(
            "tail_propagate_graph_alloc_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_graph_alloc_ms",
        )
        cpp_tail_propagate_input_upload_ms = frame_metric_or_row(
            "tail_propagate_input_upload_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_input_upload_ms",
        )
        cpp_tail_propagate_output_read_ms = frame_metric_or_row(
            "tail_propagate_output_read_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_output_read_ms",
        )
        cpp_tail_propagate_cache_compute_ms = frame_metric_or_row(
            "tail_propagate_cache_compute_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_graph_cache_compute_ms",
        )
        cpp_tail_propagate_input_prompt_upload_ms = frame_metric_or_row(
            "tail_propagate_input_prompt_upload_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_input_prompt_upload_ms",
        )
        cpp_tail_propagate_input_rope_upload_ms = frame_metric_or_row(
            "tail_propagate_input_rope_upload_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_input_rope_upload_ms",
        )
        cpp_tail_propagate_input_memory_upload_ms = frame_metric_or_row(
            "tail_propagate_input_memory_upload_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_input_memory_upload_ms",
        )
        cpp_tail_propagate_input_feature_upload_ms = frame_metric_or_row(
            "tail_propagate_input_feature_upload_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_input_feature_upload_ms",
        )
        cpp_tail_propagate_input_constant_upload_ms = frame_metric_or_row(
            "tail_propagate_input_constant_upload_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_input_constant_upload_ms",
        )
        cpp_tail_propagate_input_sparse_upload_ms = frame_metric_or_row(
            "tail_propagate_input_sparse_upload_ms",
            "mean",
            "tail_all_required_ms",
            "propagate_input_sparse_upload_ms",
        )
        cpp_frame0_encode_graph_admin_ms = optional_sum(
            cpp_frame0_encode_graph_build_ms,
            cpp_frame0_encode_graph_alloc_ms,
            cpp_frame0_encode_input_upload_ms,
        )
        cpp_tail_encode_graph_admin_ms = optional_sum(
            cpp_tail_encode_graph_build_ms,
            cpp_tail_encode_graph_alloc_ms,
            cpp_tail_encode_input_upload_ms,
        )
        cpp_tail_inline_encode_graph_admin_ms = optional_sum(
            cpp_tail_inline_encode_graph_build_ms,
            cpp_tail_inline_encode_graph_alloc_ms,
            cpp_tail_inline_encode_input_upload_ms,
        )
        cpp_tail_inline_encode_non_compute_ms = optional_sum(
            cpp_tail_inline_encode_preprocess_ms,
            cpp_tail_inline_encode_graph_admin_ms,
            cpp_tail_inline_encode_state_update_ms,
            cpp_tail_inline_encode_pe_build_ms,
            cpp_tail_inline_encode_remainder_ms,
        )
        cpp_tail_preencode_encode_graph_admin_ms = optional_sum(
            cpp_tail_preencode_encode_graph_build_ms,
            cpp_tail_preencode_encode_graph_alloc_ms,
            cpp_tail_preencode_encode_input_upload_ms,
        )
        cpp_tail_preencode_encode_non_compute_ms = optional_sum(
            cpp_tail_preencode_encode_preprocess_ms,
            cpp_tail_preencode_encode_graph_admin_ms,
            cpp_tail_preencode_encode_state_update_ms,
            cpp_tail_preencode_encode_pe_build_ms,
            cpp_tail_preencode_encode_remainder_ms,
        )
        cpp_required_encode_graph_admin_ms = optional_sum(
            cpp_frame0_encode_graph_admin_ms,
            cpp_tail_encode_graph_admin_ms,
        )
        cpp_required_encode_preprocess_ms = optional_sum(
            cpp_frame0_encode_preprocess_ms,
            cpp_tail_encode_preprocess_ms,
        )
        cpp_required_encode_state_update_ms = optional_sum(
            cpp_frame0_encode_state_update_ms,
            cpp_tail_encode_state_update_ms,
        )
        cpp_required_encode_pe_build_ms = optional_sum(
            cpp_frame0_encode_pe_build_ms,
            cpp_tail_encode_pe_build_ms,
        )
        cpp_required_encode_remainder_ms = optional_sum(
            cpp_frame0_encode_remainder_ms,
            cpp_tail_encode_remainder_ms,
        )
        cpp_required_encode_non_compute_ms = optional_sum(
            cpp_required_encode_preprocess_ms,
            cpp_required_encode_graph_admin_ms,
            cpp_required_encode_state_update_ms,
            cpp_required_encode_pe_build_ms,
            cpp_required_encode_remainder_ms,
        )
        cpp_tail_propagate_graph_admin_ms = optional_sum(
            cpp_tail_propagate_graph_build_ms,
            cpp_tail_propagate_graph_alloc_ms,
            cpp_tail_propagate_input_upload_ms,
            cpp_tail_propagate_output_read_ms,
        )
        cpp_required_core_compute_ms = optional_sum(
            cpp_frame0_graph_compute_ms,
            cpp_tail_encode_graph_compute_ms,
            cpp_tail_propagate_graph_compute_ms,
            cpp_tail_propagate_cache_compute_ms,
        )
        cpp_required_non_core_compute_ms = (
            cpp_required_e2e_ms - cpp_required_core_compute_ms
            if isinstance(cpp_required_e2e_ms, int | float)
            and cpp_required_core_compute_ms is not None
            else None
        )
        cpp_required_core_compute_fraction = optional_ratio(
            cpp_required_core_compute_ms, cpp_required_e2e_ms
        )
        cpp_tail_encode_graph_compute_fraction = optional_ratio(
            cpp_tail_encode_graph_compute_ms, cpp_tail_encode_required_total_ms
        )
        cpp_required_compute_split_ms = {
            "core_compute_ms": cpp_required_core_compute_ms,
            "non_core_compute_ms": cpp_required_non_core_compute_ms,
            "core_compute_fraction": cpp_required_core_compute_fraction,
            "frame0_encode_graph_compute_ms": cpp_frame0_graph_compute_ms,
            "frame0_encode_preprocess_ms": cpp_frame0_encode_preprocess_ms,
            "frame0_encode_graph_admin_ms": cpp_frame0_encode_graph_admin_ms,
            "frame0_encode_graph_build_ms": cpp_frame0_encode_graph_build_ms,
            "frame0_encode_graph_alloc_ms": cpp_frame0_encode_graph_alloc_ms,
            "frame0_encode_input_upload_ms": cpp_frame0_encode_input_upload_ms,
            "frame0_encode_state_update_ms": cpp_frame0_encode_state_update_ms,
            "frame0_encode_pe_build_ms": cpp_frame0_encode_pe_build_ms,
            "frame0_encode_remainder_ms": cpp_frame0_encode_remainder_ms,
            "tail_encode_graph_compute_ms": cpp_tail_encode_graph_compute_ms,
            "tail_encode_preprocess_ms": cpp_tail_encode_preprocess_ms,
            "tail_encode_graph_admin_ms": cpp_tail_encode_graph_admin_ms,
            "tail_encode_graph_build_ms": cpp_tail_encode_graph_build_ms,
            "tail_encode_graph_alloc_ms": cpp_tail_encode_graph_alloc_ms,
            "tail_encode_input_upload_ms": cpp_tail_encode_input_upload_ms,
            "tail_encode_state_update_ms": cpp_tail_encode_state_update_ms,
            "tail_encode_pe_build_ms": cpp_tail_encode_pe_build_ms,
            "tail_encode_remainder_ms": cpp_tail_encode_remainder_ms,
            "required_encode_graph_admin_ms": cpp_required_encode_graph_admin_ms,
            "required_encode_preprocess_ms": cpp_required_encode_preprocess_ms,
            "required_encode_state_update_ms": cpp_required_encode_state_update_ms,
            "required_encode_pe_build_ms": cpp_required_encode_pe_build_ms,
            "required_encode_remainder_ms": cpp_required_encode_remainder_ms,
            "required_encode_non_compute_ms": cpp_required_encode_non_compute_ms,
            "tail_inline_encode_ms": cpp_tail_inline_encode_ms,
            "tail_inline_encode_graph_compute_ms": (
                cpp_tail_inline_encode_graph_compute_ms
            ),
            "tail_inline_encode_preprocess_ms": cpp_tail_inline_encode_preprocess_ms,
            "tail_inline_encode_graph_admin_ms": (
                cpp_tail_inline_encode_graph_admin_ms
            ),
            "tail_inline_encode_graph_build_ms": (
                cpp_tail_inline_encode_graph_build_ms
            ),
            "tail_inline_encode_graph_alloc_ms": (
                cpp_tail_inline_encode_graph_alloc_ms
            ),
            "tail_inline_encode_input_upload_ms": (
                cpp_tail_inline_encode_input_upload_ms
            ),
            "tail_inline_encode_state_update_ms": (
                cpp_tail_inline_encode_state_update_ms
            ),
            "tail_inline_encode_pe_build_ms": cpp_tail_inline_encode_pe_build_ms,
            "tail_inline_encode_remainder_ms": cpp_tail_inline_encode_remainder_ms,
            "tail_inline_encode_non_compute_ms": cpp_tail_inline_encode_non_compute_ms,
            "tail_preencode_state_create_ms": cpp_tail_preencode_state_create_ms,
            "tail_preencode_encode_ms": cpp_tail_preencode_encode_ms,
            "tail_preencode_encode_graph_compute_ms": (
                cpp_tail_preencode_encode_graph_compute_ms
            ),
            "tail_preencode_encode_preprocess_ms": (
                cpp_tail_preencode_encode_preprocess_ms
            ),
            "tail_preencode_encode_graph_admin_ms": (
                cpp_tail_preencode_encode_graph_admin_ms
            ),
            "tail_preencode_encode_graph_build_ms": (
                cpp_tail_preencode_encode_graph_build_ms
            ),
            "tail_preencode_encode_graph_alloc_ms": (
                cpp_tail_preencode_encode_graph_alloc_ms
            ),
            "tail_preencode_encode_input_upload_ms": (
                cpp_tail_preencode_encode_input_upload_ms
            ),
            "tail_preencode_encode_state_update_ms": (
                cpp_tail_preencode_encode_state_update_ms
            ),
            "tail_preencode_encode_pe_build_ms": (
                cpp_tail_preencode_encode_pe_build_ms
            ),
            "tail_preencode_encode_remainder_ms": (
                cpp_tail_preencode_encode_remainder_ms
            ),
            "tail_preencode_encode_non_compute_ms": (
                cpp_tail_preencode_encode_non_compute_ms
            ),
            "tail_propagate_graph_compute_ms": cpp_tail_propagate_graph_compute_ms,
            "tail_propagate_graph_build_ms": cpp_tail_propagate_graph_build_ms,
            "tail_propagate_graph_alloc_ms": cpp_tail_propagate_graph_alloc_ms,
            "tail_propagate_input_upload_ms": cpp_tail_propagate_input_upload_ms,
            "tail_propagate_output_read_ms": cpp_tail_propagate_output_read_ms,
            "tail_propagate_graph_admin_ms": cpp_tail_propagate_graph_admin_ms,
            "tail_propagate_cache_compute_ms": cpp_tail_propagate_cache_compute_ms,
            "tail_propagate_input_prompt_upload_ms": (
                cpp_tail_propagate_input_prompt_upload_ms
            ),
            "tail_propagate_input_rope_upload_ms": (
                cpp_tail_propagate_input_rope_upload_ms
            ),
            "tail_propagate_input_memory_upload_ms": (
                cpp_tail_propagate_input_memory_upload_ms
            ),
            "tail_propagate_input_feature_upload_ms": (
                cpp_tail_propagate_input_feature_upload_ms
            ),
            "tail_propagate_input_constant_upload_ms": (
                cpp_tail_propagate_input_constant_upload_ms
            ),
            "tail_propagate_input_sparse_upload_ms": (
                cpp_tail_propagate_input_sparse_upload_ms
            ),
            "tail_encode_graph_compute_fraction": cpp_tail_encode_graph_compute_fraction,
        }
        cpp_model_e2e_breakdown = {
            "model_load_ms": cpp_model_load_ms,
            "frame0_encode_ms": row.get("frame0_encode_ms"),
            "frame0_model_ms": cpp_frame0_model_ms,
            "frame0_prompt_ms": row.get("frame0_prompt_ms"),
            "frame0_prompt_model_ms": cpp_frame0_prompt_model_ms,
            "frame0_segment_ms": row.get("frame0_segment_ms"),
            "frame0_tracker_add_ms": row.get("frame0_tracker_add_ms"),
            "frame0_prompt_unattributed_ms": row.get("frame0_prompt_unattributed_ms"),
            "frame0_artifact_ms": cpp_frame0_artifact_ms,
            "track_all_ms": row.get("track_all_ms"),
            "track_all_model_ms": cpp_track_all_model_ms,
            "tail_model_ms": cpp_tail_model_ms,
            "tail_runtime_wall_ms": cpp_tail_runtime_wall_ms,
            "tail_visible_model_ms": cpp_tail_visible_model_ms,
            "tail_preencode_required_model_ms": cpp_tail_preencode_required_model_ms,
            "tail_required_model_ms": cpp_tail_required_model_ms,
            "tail_encode_required_total_ms": cpp_tail_encode_required_total_ms,
            "required_tail_state_create_ms": row.get("required_tail_state_create_ms"),
            "required_tail_encode_ms": row.get("required_tail_encode_ms"),
            "required_tail_propagate_ms": row.get("required_tail_propagate_ms"),
            "required_tail_unattributed_ms": row.get("required_tail_unattributed_ms"),
            "required_output_artifact_ms": row.get("required_output_artifact_ms"),
            "track_encode_avg_ms": row.get("track_encode_ms"),
            "track_encode_total_ms": row.get("track_encode_total_ms"),
            "track_propagate_avg_ms": row.get("track_propagate_ms"),
            "track_propagate_total_ms": row.get("track_propagate_total_ms"),
            "track_unattributed_avg_ms": row.get("track_unattributed_ms"),
            "track_unattributed_total_ms": row.get("track_unattributed_total_ms"),
            "track_artifact_avg_ms": row.get("track_artifact_ms"),
            "track_artifact_total_ms": cpp_track_artifact_total_ms,
            "preencode_total_ms": row.get("preencode_total_ms"),
            "preencode_avg_ms": row.get("preencode_avg_ms"),
            "preencode_state_create_total_ms": row.get(
                "preencode_state_create_total_ms"
            ),
            "preencode_state_create_avg_ms": row.get("preencode_state_create_avg_ms"),
            "input_load_ms": cpp_input_load_ms,
            "input_breakdown_ms": cpp_input_breakdown_ms,
            "artifact_ms": cpp_artifact_ms,
            "required_accounted_ms": cpp_required_accounted_ms,
            "required_remainder_ms": cpp_required_remainder_ms,
            "required_accounting_delta_ms": cpp_required_accounting_delta_ms,
            "model_accounting_delta_ms": cpp_model_accounting_delta_ms,
            "model_accounted_ms": cpp_model_accounted_ms,
            "model_e2e_ms": cpp_model_e2e_ms,
            "required_e2e_ms": cpp_required_e2e_ms,
            "required_compute_split_ms": cpp_required_compute_split_ms,
            "cold_required_e2e_ms": cpp_cold_required_e2e_ms,
            "session_with_artifacts_ms": cpp_session_with_artifacts_ms,
            "input_plus_session_ms": cpp_input_plus_session_ms,
            "session_overhead_ms": cpp_session_overhead_ms,
            "model_remainder_ms": cpp_model_remainder_ms,
            "state_create_ms": row.get("state_create_ms"),
            "tracker_create_ms": row.get("tracker_create_ms"),
            "session_setup_ms": row.get("session_setup_ms"),
            "frame0_candidates": row.get("frame0_candidates"),
            "frame0_added_instances": row.get("frame0_added_instances"),
            "text_init_selected_only": row.get("text_init_selected_only"),
            "frame_timing_aggregate_ms": cpp_frame_timing_aggregate_ms,
        }
        python_model_e2e_breakdown = {
            "start_session_ms": py.get("start_session_ms"),
            "add_prompt_ms": py.get("add_prompt_ms"),
            "propagate_wall_ms": py.get("propagate_wall_ms"),
            "profile_stage_total_ms_stats": py.get("profile_stage_total_ms_stats"),
            "profile_timed_tail_stage_total_ms_stats": py.get(
                "profile_timed_tail_stage_total_ms_stats"
            ),
            "profile_stage_phase_total_ms_stats": py.get(
                "profile_stage_phase_total_ms_stats"
            ),
            "profile_timed_tail_stage_phase_total_ms_stats": py.get(
                "profile_timed_tail_stage_phase_total_ms_stats"
            ),
            "profile_stage_event_counts": py.get("profile_stage_event_counts"),
            "profile_timed_tail_stage_event_counts": py.get(
                "profile_timed_tail_stage_event_counts"
            ),
            "timed_tail_backbone_detection_frame_ms": py.get(
                "timed_tail_backbone_detection_frame_ms"
            ),
            "timed_tail_backbone_detection_heavy_frames": py.get(
                "timed_tail_backbone_detection_heavy_frames"
            ),
            "timed_tail_backbone_detection_heavy_threshold_ms": py.get(
                "timed_tail_backbone_detection_heavy_threshold_ms"
            ),
            "model_e2e_ms": python_model_e2e_ms,
            "session_e2e_ms": py.get("session_e2e_ms"),
        }
        python_phase_stats = py.get("profile_stage_phase_total_ms_stats")
        python_timed_tail_phase_stats = py.get(
            "profile_timed_tail_stage_phase_total_ms_stats"
        )
        cpp_required_e2e_breakdown = {
            "required_e2e_ms": cpp_required_e2e_ms,
            "input_load_ms": cpp_input_load_ms,
            "input_breakdown_ms": cpp_input_breakdown_ms,
            "model_e2e_ms": cpp_model_e2e_ms,
            "accounted_ms": cpp_required_accounted_ms,
            "remainder_ms": cpp_required_remainder_ms,
            "accounting_delta_ms": cpp_required_accounting_delta_ms,
            "model_components_ms": {
                "session_setup_ms": row.get("session_setup_ms"),
                "state_create_ms": row.get("state_create_ms"),
                "tracker_create_ms": row.get("tracker_create_ms"),
                "frame0_model_ms": cpp_frame0_model_ms,
                "tail_model_ms": cpp_tail_model_ms,
                "preencode_state_create_total_ms": row.get(
                    "preencode_state_create_total_ms"
                ),
                "preencode_total_ms": row.get("preencode_total_ms"),
                "model_remainder_ms": cpp_model_remainder_ms,
                "model_accounted_ms": cpp_model_accounted_ms,
                "model_accounting_delta_ms": cpp_model_accounting_delta_ms,
            },
            "frame0_ms": {
                "encode_ms": row.get("frame0_encode_ms"),
                "prompt_model_ms": cpp_frame0_prompt_model_ms,
                "segment_ms": row.get("frame0_segment_ms"),
                "tracker_add_ms": row.get("frame0_tracker_add_ms"),
                "tracker_mask_prepare_ms": row.get("frame0_tracker_mask_prepare_ms"),
                "tracker_memory_encode_ms": row.get("frame0_tracker_memory_encode_ms"),
                "tracker_obj_ptr_ms": row.get("frame0_tracker_obj_ptr_ms"),
                "tracker_store_ms": row.get("frame0_tracker_store_ms"),
                "tracker_unattributed_ms": row.get("frame0_tracker_unattributed_ms"),
                "prompt_unattributed_ms": row.get("frame0_prompt_unattributed_ms"),
                "artifact_ms": cpp_frame0_artifact_ms,
            },
            "tail_ms": {
                "required_total_ms": cpp_tail_model_ms,
                "runtime_wall_ms": cpp_tail_runtime_wall_ms,
                "visible_model_ms": cpp_tail_visible_model_ms,
                "preencode_required_model_ms": cpp_tail_preencode_required_model_ms,
                "required_model_ms": cpp_tail_required_model_ms,
                "encode_required_total_ms": cpp_tail_encode_required_total_ms,
                "encode_graph_compute_ms": cpp_tail_encode_graph_compute_ms,
                "encode_graph_compute_fraction": cpp_tail_encode_graph_compute_fraction,
                "state_create_ms": row.get("required_tail_state_create_ms"),
                "propagate_ms": row.get("required_tail_propagate_ms"),
                "propagate_graph_compute_ms": cpp_tail_propagate_graph_compute_ms,
                "propagate_cache_compute_ms": cpp_tail_propagate_cache_compute_ms,
                "propagate_input_prompt_upload_ms": (
                    cpp_tail_propagate_input_prompt_upload_ms
                ),
                "propagate_input_rope_upload_ms": (
                    cpp_tail_propagate_input_rope_upload_ms
                ),
                "propagate_input_memory_upload_ms": (
                    cpp_tail_propagate_input_memory_upload_ms
                ),
                "propagate_input_feature_upload_ms": (
                    cpp_tail_propagate_input_feature_upload_ms
                ),
                "propagate_input_constant_upload_ms": (
                    cpp_tail_propagate_input_constant_upload_ms
                ),
                "propagate_input_sparse_upload_ms": (
                    cpp_tail_propagate_input_sparse_upload_ms
                ),
                "inline_encode_ms": cpp_tail_inline_encode_ms,
                "inline_encode_graph_compute_ms": cpp_tail_inline_encode_graph_compute_ms,
                "inline_propagate_ms": cpp_tail_inline_propagate_ms,
                "preencode_state_create_ms": cpp_tail_preencode_state_create_ms,
                "preencode_encode_ms": cpp_tail_preencode_encode_ms,
                "preencode_encode_graph_compute_ms": (
                    cpp_tail_preencode_encode_graph_compute_ms
                ),
                "preencoded_propagate_ms": cpp_tail_preencoded_propagate_ms,
                "unattributed_ms": row.get("required_tail_unattributed_ms"),
                "output_artifact_ms": row.get("required_output_artifact_ms"),
                "inline_encode_total_ms": row.get("track_encode_total_ms"),
                "inline_propagate_total_ms": row.get("track_propagate_total_ms"),
                "preencode_total_ms": row.get("preencode_total_ms"),
                "preencode_state_create_total_ms": row.get(
                    "preencode_state_create_total_ms"
                ),
            },
            "non_required_diagnostics_ms": {
                "model_load_ms": cpp_model_load_ms,
                "session_with_artifacts_ms": cpp_session_with_artifacts_ms,
                "artifact_ms": cpp_artifact_ms,
                "track_artifact_total_ms": cpp_track_artifact_total_ms,
                "input_plus_session_ms": cpp_input_plus_session_ms,
                "cold_required_e2e_ms": cpp_cold_required_e2e_ms,
            },
            "frame_timing_aggregate_ms": cpp_frame_timing_aggregate_ms,
        }
        python_required_e2e_breakdown = {
            "session_e2e_ms": python_session_e2e_ms,
            "model_e2e_ms": python_model_e2e_ms,
            "start_session_ms": py.get("start_session_ms"),
            "add_prompt_ms": py.get("add_prompt_ms"),
            "propagate_wall_ms": py.get("propagate_wall_ms"),
            "phase_stage_mean_ms": phase_stage_means(python_phase_stats),
            "timed_tail_phase_stage_mean_ms": phase_stage_means(
                python_timed_tail_phase_stats
            ),
            "key_stage_mean_ms": {
                "start_session_load_video_frames": nested_mean(
                    python_phase_stats, "start_session", "load_video_frames"
                ),
                "start_session_construct_initial_input_batch": nested_mean(
                    python_phase_stats,
                    "start_session",
                    "construct_initial_input_batch",
                ),
                "add_prompt_backbone_detection": nested_mean(
                    python_phase_stats, "add_prompt", "backbone_detection"
                ),
                "propagate_backbone_detection": nested_mean(
                    python_phase_stats, "propagate", "backbone_detection"
                ),
                "propagate_tracker_propagation": nested_mean(
                    python_phase_stats, "propagate", "tracker_propagation"
                ),
                "timed_tail_backbone_detection": nested_mean(
                    python_timed_tail_phase_stats, "propagate", "backbone_detection"
                ),
                "timed_tail_tracker_propagation": nested_mean(
                    python_timed_tail_phase_stats,
                    "propagate",
                    "tracker_propagation",
                ),
            },
        }
        required_e2e_component_split_ms = {
            "cpp": {
                "required_e2e_ms": cpp_required_e2e_ms,
                "input_load_ms": cpp_input_load_ms,
                "input_frame_decode_ms": row.get("input_frame_decode_ms"),
                "input_remainder_ms": row.get("input_remainder_ms"),
                "session_setup_ms": row.get("session_setup_ms"),
                "frame0_encode_ms": row.get("frame0_encode_ms"),
                "frame0_prompt_model_ms": cpp_frame0_prompt_model_ms,
                "tail_state_create_ms": row.get("required_tail_state_create_ms"),
                "tail_runtime_wall_ms": cpp_tail_runtime_wall_ms,
                "tail_visible_model_ms": cpp_tail_visible_model_ms,
                "tail_preencode_required_model_ms": cpp_tail_preencode_required_model_ms,
                "tail_required_model_ms": cpp_tail_required_model_ms,
                "tail_encode_required_ms": cpp_tail_encode_required_total_ms,
                "core_compute_ms": cpp_required_core_compute_ms,
                "non_core_compute_ms": cpp_required_non_core_compute_ms,
                "core_compute_fraction": cpp_required_core_compute_fraction,
                "frame0_encode_graph_compute_ms": cpp_frame0_graph_compute_ms,
                "frame0_encode_preprocess_ms": cpp_frame0_encode_preprocess_ms,
                "frame0_encode_graph_admin_ms": cpp_frame0_encode_graph_admin_ms,
                "frame0_encode_graph_build_ms": cpp_frame0_encode_graph_build_ms,
                "frame0_encode_graph_alloc_ms": cpp_frame0_encode_graph_alloc_ms,
                "frame0_encode_input_upload_ms": cpp_frame0_encode_input_upload_ms,
                "frame0_encode_state_update_ms": cpp_frame0_encode_state_update_ms,
                "frame0_encode_pe_build_ms": cpp_frame0_encode_pe_build_ms,
                "frame0_encode_remainder_ms": cpp_frame0_encode_remainder_ms,
                "tail_encode_graph_compute_ms": cpp_tail_encode_graph_compute_ms,
                "tail_encode_preprocess_ms": cpp_tail_encode_preprocess_ms,
                "tail_encode_graph_admin_ms": cpp_tail_encode_graph_admin_ms,
                "tail_encode_graph_build_ms": cpp_tail_encode_graph_build_ms,
                "tail_encode_graph_alloc_ms": cpp_tail_encode_graph_alloc_ms,
                "tail_encode_input_upload_ms": cpp_tail_encode_input_upload_ms,
                "tail_encode_state_update_ms": cpp_tail_encode_state_update_ms,
                "tail_encode_pe_build_ms": cpp_tail_encode_pe_build_ms,
                "tail_encode_remainder_ms": cpp_tail_encode_remainder_ms,
                "required_encode_graph_admin_ms": cpp_required_encode_graph_admin_ms,
                "required_encode_preprocess_ms": cpp_required_encode_preprocess_ms,
                "required_encode_state_update_ms": cpp_required_encode_state_update_ms,
                "required_encode_pe_build_ms": cpp_required_encode_pe_build_ms,
                "required_encode_remainder_ms": cpp_required_encode_remainder_ms,
                "required_encode_non_compute_ms": cpp_required_encode_non_compute_ms,
                "tail_encode_graph_compute_fraction": cpp_tail_encode_graph_compute_fraction,
                "tail_inline_encode_ms": cpp_tail_inline_encode_ms,
                "tail_inline_encode_graph_compute_ms": (
                    cpp_tail_inline_encode_graph_compute_ms
                ),
                "tail_inline_encode_preprocess_ms": cpp_tail_inline_encode_preprocess_ms,
                "tail_inline_encode_graph_admin_ms": (
                    cpp_tail_inline_encode_graph_admin_ms
                ),
                "tail_inline_encode_graph_build_ms": (
                    cpp_tail_inline_encode_graph_build_ms
                ),
                "tail_inline_encode_graph_alloc_ms": (
                    cpp_tail_inline_encode_graph_alloc_ms
                ),
                "tail_inline_encode_input_upload_ms": (
                    cpp_tail_inline_encode_input_upload_ms
                ),
                "tail_inline_encode_state_update_ms": (
                    cpp_tail_inline_encode_state_update_ms
                ),
                "tail_inline_encode_pe_build_ms": cpp_tail_inline_encode_pe_build_ms,
                "tail_inline_encode_remainder_ms": cpp_tail_inline_encode_remainder_ms,
                "tail_inline_encode_non_compute_ms": (
                    cpp_tail_inline_encode_non_compute_ms
                ),
                "tail_inline_propagate_ms": cpp_tail_inline_propagate_ms,
                "tail_preencode_state_create_ms": cpp_tail_preencode_state_create_ms,
                "tail_preencode_encode_ms": cpp_tail_preencode_encode_ms,
                "tail_preencode_encode_graph_compute_ms": (
                    cpp_tail_preencode_encode_graph_compute_ms
                ),
                "tail_preencode_encode_preprocess_ms": (
                    cpp_tail_preencode_encode_preprocess_ms
                ),
                "tail_preencode_encode_graph_admin_ms": (
                    cpp_tail_preencode_encode_graph_admin_ms
                ),
                "tail_preencode_encode_graph_build_ms": (
                    cpp_tail_preencode_encode_graph_build_ms
                ),
                "tail_preencode_encode_graph_alloc_ms": (
                    cpp_tail_preencode_encode_graph_alloc_ms
                ),
                "tail_preencode_encode_input_upload_ms": (
                    cpp_tail_preencode_encode_input_upload_ms
                ),
                "tail_preencode_encode_state_update_ms": (
                    cpp_tail_preencode_encode_state_update_ms
                ),
                "tail_preencode_encode_pe_build_ms": (
                    cpp_tail_preencode_encode_pe_build_ms
                ),
                "tail_preencode_encode_remainder_ms": (
                    cpp_tail_preencode_encode_remainder_ms
                ),
                "tail_preencode_encode_non_compute_ms": (
                    cpp_tail_preencode_encode_non_compute_ms
                ),
                "tail_preencoded_propagate_ms": cpp_tail_preencoded_propagate_ms,
                "tail_propagate_graph_compute_ms": cpp_tail_propagate_graph_compute_ms,
                "tail_propagate_graph_build_ms": cpp_tail_propagate_graph_build_ms,
                "tail_propagate_graph_alloc_ms": cpp_tail_propagate_graph_alloc_ms,
                "tail_propagate_input_upload_ms": cpp_tail_propagate_input_upload_ms,
                "tail_propagate_output_read_ms": cpp_tail_propagate_output_read_ms,
                "tail_propagate_graph_admin_ms": cpp_tail_propagate_graph_admin_ms,
                "tail_propagate_cache_compute_ms": cpp_tail_propagate_cache_compute_ms,
                "tail_propagate_input_prompt_upload_ms": (
                    cpp_tail_propagate_input_prompt_upload_ms
                ),
                "tail_propagate_input_rope_upload_ms": (
                    cpp_tail_propagate_input_rope_upload_ms
                ),
                "tail_propagate_input_memory_upload_ms": (
                    cpp_tail_propagate_input_memory_upload_ms
                ),
                "tail_propagate_input_feature_upload_ms": (
                    cpp_tail_propagate_input_feature_upload_ms
                ),
                "tail_propagate_input_constant_upload_ms": (
                    cpp_tail_propagate_input_constant_upload_ms
                ),
                "tail_propagate_input_sparse_upload_ms": (
                    cpp_tail_propagate_input_sparse_upload_ms
                ),
                "tail_propagate_ms": row.get("required_tail_propagate_ms"),
                "tail_unattributed_ms": row.get("required_tail_unattributed_ms"),
                "model_remainder_ms": cpp_model_remainder_ms,
                "required_remainder_ms": cpp_required_remainder_ms,
            },
            "python": {
                "session_e2e_ms": python_session_e2e_ms,
                "start_session_ms": py.get("start_session_ms"),
                "model_e2e_ms": python_model_e2e_ms,
                "add_prompt_ms": py.get("add_prompt_ms"),
                "propagate_wall_ms": py.get("propagate_wall_ms"),
            },
            "python_profile_diagnostic": {
                "start_session_load_video_frames_ms": nested_mean(
                    python_phase_stats, "start_session", "load_video_frames"
                ),
                "start_session_construct_initial_input_batch_ms": nested_mean(
                    python_phase_stats,
                    "start_session",
                    "construct_initial_input_batch",
                ),
                "add_prompt_backbone_detection_ms": nested_mean(
                    python_phase_stats, "add_prompt", "backbone_detection"
                ),
                "timed_tail_backbone_detection_ms": nested_mean(
                    python_timed_tail_phase_stats, "propagate", "backbone_detection"
                ),
                "timed_tail_tracker_propagation_ms": nested_mean(
                    python_timed_tail_phase_stats,
                    "propagate",
                    "tracker_propagation",
                ),
            },
        }
        cpp_track_encode_ms = row.get("track_encode_ms")
        cpp_track_propagate_ms = row.get("track_propagate_ms")
        cpp_track_stage_sum = (
            cpp_track_encode_ms + cpp_track_propagate_ms
            if isinstance(cpp_track_encode_ms, int | float)
            and isinstance(cpp_track_propagate_ms, int | float)
            else None
        )
        comparable_model_e2e_claim = (
            exact_family_match
            and same_decoded_source_resolution
            and same_frame_range
            and same_prompt
            and same_encode_size
            and same_warmup_policy
            and same_text_init_cardinality
            and e2e_ratio is not None
        )
        comparable_session_e2e_claim = (
            exact_family_match
            and same_decoded_source_resolution
            and same_frame_range
            and same_prompt
            and same_encode_size
            and same_warmup_policy
            and same_text_init_cardinality
            and cpp_has_input_load_ms
            and session_e2e_ratio is not None
        )
        same_contract_model_e2e_diagnostic = (
            same_comparison_contract_family
            and same_decoded_source_resolution
            and same_frame_range
            and same_prompt
            and same_encode_size
            and same_warmup_policy
            and same_text_init_cardinality
            and e2e_ratio is not None
        )
        same_contract_session_e2e_diagnostic = (
            same_comparison_contract_family
            and same_decoded_source_resolution
            and same_frame_range
            and same_prompt
            and same_encode_size
            and same_warmup_policy
            and same_text_init_cardinality
            and cpp_has_input_load_ms
            and session_e2e_ratio is not None
        )
        comparisons.append(
            {
                "model": row["model"],
                "family": row["family"],
                "python_family": python_family,
                "exact_family_match": exact_family_match,
                "comparison_contract_family": cpp_comparison_contract,
                "same_comparison_contract_family": same_comparison_contract_family,
                "precision": row["precision"],
                "decoded_source_width": decoded_width,
                "decoded_source_height": decoded_height,
                "frames": frames,
                "point_prompt": {"x": point_x, "y": point_y},
                "text_prompt": prompt,
                "cpp_encode_img_size": cpp_encode_size,
                "python_image_size": python_encode_size,
                "same_decoded_source_resolution": same_decoded_source_resolution,
                "same_frame_range": same_frame_range,
                "same_prompt": same_prompt,
                "same_input_encode_size": same_encode_size,
                "same_measured_tracking_scope": same_measured_tracking_scope,
                "same_timed_start_frame": same_timed_start_frame,
                "cpp_track_scope": cpp_track_scope,
                "python_track_scope": python_track_scope,
                "cpp_timed_start_frame": row.get("timed_start_frame"),
                "python_timed_start_frame": py.get("timed_start_frame"),
                "same_warmup_policy": same_warmup_policy,
                "same_text_init_cardinality": same_text_init_cardinality,
                "cpp_warmup_runs": cpp_warmup_runs,
                "python_warmup_runs": python_warmup_runs,
                "comparable_speed_claim": comparable_speed_claim,
                "comparable_model_e2e_claim": comparable_model_e2e_claim,
                "comparable_session_e2e_claim": comparable_session_e2e_claim,
                "same_contract_speed_diagnostic": same_contract_speed_diagnostic,
                "same_contract_model_e2e_diagnostic": (
                    same_contract_model_e2e_diagnostic
                ),
                "same_contract_session_e2e_diagnostic": (
                    same_contract_session_e2e_diagnostic
                ),
                "cpp_track_ms": row["track_ms"],
                "python_track_ms": py["track_ms"],
                "cpp_model_load_ms": cpp_model_load_ms,
                "cpp_has_input_load_ms": cpp_has_input_load_ms,
                "cpp_input_load_ms": cpp_input_load_ms,
                "cpp_input_breakdown_ms": cpp_input_breakdown_ms,
                "cpp_model_e2e_ms": cpp_model_e2e_ms,
                "cpp_required_e2e_ms": cpp_required_e2e_ms,
                "cpp_cold_required_e2e_ms": cpp_cold_required_e2e_ms,
                "cpp_session_with_artifacts_ms": cpp_session_with_artifacts_ms,
                "cpp_input_plus_session_ms": cpp_input_plus_session_ms,
                "cpp_required_accounted_ms": cpp_required_accounted_ms,
                "cpp_required_remainder_ms": cpp_required_remainder_ms,
                "cpp_required_accounting_delta_ms": cpp_required_accounting_delta_ms,
                "cpp_model_accounting_delta_ms": cpp_model_accounting_delta_ms,
                "cpp_model_accounted_ms": cpp_model_accounted_ms,
                "cpp_session_overhead_ms": cpp_session_overhead_ms,
                "cpp_model_remainder_ms": cpp_model_remainder_ms,
                "cpp_required_core_compute_ms": cpp_required_core_compute_ms,
                "cpp_required_non_core_compute_ms": cpp_required_non_core_compute_ms,
                "cpp_required_core_compute_fraction": cpp_required_core_compute_fraction,
                "cpp_tail_encode_graph_compute_ms": cpp_tail_encode_graph_compute_ms,
                "cpp_tail_encode_graph_compute_fraction": (
                    cpp_tail_encode_graph_compute_fraction
                ),
                "cpp_tail_propagate_graph_compute_ms": (
                    cpp_tail_propagate_graph_compute_ms
                ),
                "cpp_frame_timing_aggregate_ms": cpp_frame_timing_aggregate_ms,
                "cpp_artifact_ms": cpp_artifact_ms,
                "python_model_e2e_ms": python_model_e2e_ms,
                "python_session_e2e_ms": python_session_e2e_ms,
                "python_over_cpp_model_e2e_ratio": e2e_ratio,
                "python_over_cpp_model_e2e_ratio_if_comparable": (
                    e2e_ratio if comparable_model_e2e_claim else None
                ),
                "python_over_cpp_session_e2e_ratio": session_e2e_ratio,
                "python_over_cpp_session_e2e_ratio_if_comparable": (
                    session_e2e_ratio if comparable_session_e2e_claim else None
                ),
                "cpp_frame0_encode_ms": row.get("frame0_encode_ms"),
                "cpp_frame0_prompt_ms": row.get("frame0_prompt_ms"),
                "cpp_frame0_model_ms": cpp_frame0_model_ms,
                "cpp_frame0_prompt_model_ms": cpp_frame0_prompt_model_ms,
                "cpp_frame0_segment_ms": row.get("frame0_segment_ms"),
                "cpp_frame0_tracker_add_ms": row.get("frame0_tracker_add_ms"),
                "cpp_frame0_tracker_mask_prepare_ms": row.get(
                    "frame0_tracker_mask_prepare_ms"
                ),
                "cpp_frame0_tracker_memory_encode_ms": row.get(
                    "frame0_tracker_memory_encode_ms"
                ),
                "cpp_frame0_tracker_obj_ptr_ms": row.get("frame0_tracker_obj_ptr_ms"),
                "cpp_frame0_tracker_store_ms": row.get("frame0_tracker_store_ms"),
                "cpp_frame0_tracker_unattributed_ms": row.get(
                    "frame0_tracker_unattributed_ms"
                ),
                "cpp_frame0_prompt_unattributed_ms": row.get(
                    "frame0_prompt_unattributed_ms"
                ),
                "cpp_frame0_candidates": row.get("frame0_candidates"),
                "cpp_frame0_added_instances": row.get("frame0_added_instances"),
                "cpp_frame0_artifact_ms": cpp_frame0_artifact_ms,
                "cpp_track_encode_ms": row.get("track_encode_ms"),
                "cpp_tail_encode_required_total_ms": cpp_tail_encode_required_total_ms,
                "cpp_tail_inline_encode_ms": cpp_tail_inline_encode_ms,
                "cpp_tail_inline_encode_graph_compute_ms": (
                    cpp_tail_inline_encode_graph_compute_ms
                ),
                "cpp_tail_inline_encode_preprocess_ms": (
                    cpp_tail_inline_encode_preprocess_ms
                ),
                "cpp_tail_inline_encode_graph_admin_ms": (
                    cpp_tail_inline_encode_graph_admin_ms
                ),
                "cpp_tail_inline_encode_graph_build_ms": (
                    cpp_tail_inline_encode_graph_build_ms
                ),
                "cpp_tail_inline_encode_graph_alloc_ms": (
                    cpp_tail_inline_encode_graph_alloc_ms
                ),
                "cpp_tail_inline_encode_input_upload_ms": (
                    cpp_tail_inline_encode_input_upload_ms
                ),
                "cpp_tail_inline_encode_state_update_ms": (
                    cpp_tail_inline_encode_state_update_ms
                ),
                "cpp_tail_inline_encode_pe_build_ms": (
                    cpp_tail_inline_encode_pe_build_ms
                ),
                "cpp_tail_inline_encode_remainder_ms": (
                    cpp_tail_inline_encode_remainder_ms
                ),
                "cpp_tail_inline_encode_non_compute_ms": (
                    cpp_tail_inline_encode_non_compute_ms
                ),
                "cpp_tail_inline_propagate_ms": cpp_tail_inline_propagate_ms,
                "cpp_tail_preencode_state_create_ms": (
                    cpp_tail_preencode_state_create_ms
                ),
                "cpp_tail_preencode_encode_ms": cpp_tail_preencode_encode_ms,
                "cpp_tail_preencode_encode_graph_compute_ms": (
                    cpp_tail_preencode_encode_graph_compute_ms
                ),
                "cpp_tail_preencode_encode_preprocess_ms": (
                    cpp_tail_preencode_encode_preprocess_ms
                ),
                "cpp_tail_preencode_encode_graph_admin_ms": (
                    cpp_tail_preencode_encode_graph_admin_ms
                ),
                "cpp_tail_preencode_encode_graph_build_ms": (
                    cpp_tail_preencode_encode_graph_build_ms
                ),
                "cpp_tail_preencode_encode_graph_alloc_ms": (
                    cpp_tail_preencode_encode_graph_alloc_ms
                ),
                "cpp_tail_preencode_encode_input_upload_ms": (
                    cpp_tail_preencode_encode_input_upload_ms
                ),
                "cpp_tail_preencode_encode_state_update_ms": (
                    cpp_tail_preencode_encode_state_update_ms
                ),
                "cpp_tail_preencode_encode_pe_build_ms": (
                    cpp_tail_preencode_encode_pe_build_ms
                ),
                "cpp_tail_preencode_encode_remainder_ms": (
                    cpp_tail_preencode_encode_remainder_ms
                ),
                "cpp_tail_preencode_encode_non_compute_ms": (
                    cpp_tail_preencode_encode_non_compute_ms
                ),
                "cpp_tail_preencoded_propagate_ms": cpp_tail_preencoded_propagate_ms,
                "cpp_tail_model_ms": cpp_tail_model_ms,
                "cpp_tail_runtime_wall_ms": cpp_tail_runtime_wall_ms,
                "cpp_tail_visible_model_ms": cpp_tail_visible_model_ms,
                "cpp_tail_preencode_required_model_ms": (
                    cpp_tail_preencode_required_model_ms
                ),
                "cpp_tail_required_model_ms": cpp_tail_required_model_ms,
                "cpp_required_tail_state_create_ms": row.get(
                    "required_tail_state_create_ms"
                ),
                "cpp_required_tail_encode_ms": row.get("required_tail_encode_ms"),
                "cpp_required_tail_propagate_ms": row.get("required_tail_propagate_ms"),
                "cpp_required_tail_unattributed_ms": row.get(
                    "required_tail_unattributed_ms"
                ),
                "cpp_required_output_artifact_ms": row.get(
                    "required_output_artifact_ms"
                ),
                "cpp_track_propagate_ms": row.get("track_propagate_ms"),
                "cpp_track_artifact_ms": row.get("track_artifact_ms"),
                "cpp_track_artifact_total_ms": cpp_track_artifact_total_ms,
                "cpp_preencode_avg_ms": row.get("preencode_avg_ms"),
                "python_start_session_ms": py.get("start_session_ms"),
                "python_add_prompt_ms": py.get("add_prompt_ms"),
                "python_propagate_wall_ms": py.get("propagate_wall_ms"),
                "python_stage_ms_stats": py.get("stage_ms_stats"),
                "python_timed_tail_stage_ms_stats": py.get("timed_tail_stage_ms_stats"),
                "python_profile_stage_total_ms_stats": py.get(
                    "profile_stage_total_ms_stats"
                ),
                "python_profile_timed_tail_stage_total_ms_stats": py.get(
                    "profile_timed_tail_stage_total_ms_stats"
                ),
                "python_profile_stage_phase_total_ms_stats": py.get(
                    "profile_stage_phase_total_ms_stats"
                ),
                "python_profile_timed_tail_stage_phase_total_ms_stats": py.get(
                    "profile_timed_tail_stage_phase_total_ms_stats"
                ),
                "python_timed_tail_backbone_detection_frame_ms": py.get(
                    "timed_tail_backbone_detection_frame_ms"
                ),
                "python_timed_tail_backbone_detection_heavy_frames": py.get(
                    "timed_tail_backbone_detection_heavy_frames"
                ),
                "python_track_frames": py.get("track_frames"),
                "python_dtype": py.get("python_dtype"),
                "tf32_policy": py.get("tf32_policy"),
                "python_over_cpp_track_ratio": track_ratio,
                "python_over_cpp_track_ratio_if_comparable": track_ratio
                if comparable_speed_claim
                else None,
                "cpp_variant": row.get("cpp_variant", "default"),
                "cpp_env_overrides": row.get("cpp_env_overrides", {}),
                "cpp_output_artifacts": row.get("cpp_output_artifacts", True),
                "cpp_frame_timing_jsonl": row.get("cpp_frame_timing_jsonl"),
                "cpp_frame_timing_jsonl_runs": row.get("cpp_frame_timing_jsonl_runs"),
                "cpp_text_init_selected_only": row.get(
                    "text_init_selected_only", False
                ),
                "cpp_model_e2e_breakdown_ms": cpp_model_e2e_breakdown,
                "python_model_e2e_breakdown_ms": python_model_e2e_breakdown,
                "cpp_required_e2e_breakdown_ms": cpp_required_e2e_breakdown,
                "python_required_e2e_breakdown_ms": python_required_e2e_breakdown,
                "required_e2e_component_split_ms": required_e2e_component_split_ms,
                "cpp_track_encode_fraction": (
                    cpp_track_encode_ms / cpp_track_stage_sum
                    if cpp_track_stage_sum and cpp_track_stage_sum > 0
                    else None
                ),
                "cpp_track_propagate_fraction": (
                    cpp_track_propagate_ms / cpp_track_stage_sum
                    if cpp_track_stage_sum and cpp_track_stage_sum > 0
                    else None
                ),
                "cpp_rss_mib": row["rss_mib"],
                "python_cuda_alloc_mib": py["rss_mib"],
            }
        )
    comparable = [row for row in comparisons if row.get("comparable_speed_claim")]
    comparable_model_e2e = [
        row for row in comparisons if row.get("comparable_model_e2e_claim")
    ]
    comparable_session_e2e = [
        row for row in comparisons if row.get("comparable_session_e2e_claim")
    ]
    same_contract_speed_diagnostic = [
        row for row in comparisons if row.get("same_contract_speed_diagnostic")
    ]
    same_contract_model_e2e_diagnostic = [
        row for row in comparisons if row.get("same_contract_model_e2e_diagnostic")
    ]
    same_contract_session_e2e_diagnostic = [
        row for row in comparisons if row.get("same_contract_session_e2e_diagnostic")
    ]
    not_comparable = [
        row for row in comparisons if not row.get("comparable_speed_claim")
    ]
    summary = {
        "comparison_contract": {
            "speed_and_quality_claims_require": [
                "same decoded source-frame resolution",
                "same frame range",
                "same timed-start frame",
                "same measured tracking range (frame 0 add-instance excluded)",
                "same measured tracking scope (frame 1..N-1 encode+propagate; frame 0 prompt/add excluded)",
                "same text-init target cardinality (all detections versus selected target only)",
                "separate model_e2e timing for prompt/add plus full propagation wall time",
                "separate C++ input-load timing and include it in cpp_required_e2e_ms "
                "for session-e2e claims",
                "exclude C++ benchmark artifacts (JSONL/mask/logits/debug-only candidate passes) "
                "from cpp_model_e2e_ms or report them separately as cpp_artifact_ms",
                "separate C++ frame encode and cached propagation timing when the API exposes both stages",
                "separate official Python session/prompt/propagate wall timing plus available internal stage hooks",
                "reported preprocessing/input-staging timing policy",
                "same benchmark warmup policy",
                "same prompt",
                "same SAM model input resolution",
            ],
            "note": (
                "Rows with comparable_speed_claim=false are scaling or coverage rows. "
                "Do not use their C++/Python ratios as speed win/loss evidence. "
                "Use comparable_model_e2e_claim for prompt/add plus full propagation wall-time "
                "comparisons that exclude official Python start_session. "
                "Use comparable_session_e2e_claim when including official Python start_session "
                "as part of the required session work; for that claim, C++ uses "
                "cpp_required_e2e_ms = cpp_input_load_ms + cpp_model_e2e_ms and requires a real "
                "cpp_input_load_ms measurement. Rows using C++ selected-only text init are "
                "reported as application-contract measurements, not official all-detections "
                "Python parity claims. Use comparable_speed_claim for the narrower "
                "timed Track/fr scope. "
                "For C++, cpp_model_e2e_ms excludes optional benchmark artifact generation when "
                "the benchmark reports model_e2e_ms; cpp_session_with_artifacts_ms keeps the "
                "artifact-inclusive wall time. Cached/preencoded C++ tail work is added back "
                "through tail_required_model_ms and cpp_model_e2e_ms before speed claims. "
                "For SAM3 official Python, start_session preloads/resizes/normalizes frames and "
                "is reported separately from model_e2e_ms but included in session_e2e_ms. "
                "same_contract_*_diagnostic_rows pair SAM3-family C++ rows with the configured "
                "official SAM3/SAM3.1 Python baseline for bottleneck analysis, but they are not "
                "formal parity claims unless exact_family_match is also true."
            ),
        },
        "benchmark_context": {
            "decoded_source_width": decoded_width,
            "decoded_source_height": decoded_height,
            "decoded_frame_format": frame_format,
            "frames": frames,
            "timed_start_frame": max(
                (
                    int(row.get("timed_start_frame", 1))
                    for row in cpp_rows
                    if isinstance(row, dict)
                ),
                default=1,
            ),
            "point_prompt": {"x": point_x, "y": point_y},
            "text_prompt": prompt,
            "requested_encode_img_size": requested_encode_size,
            "multimask": multimask,
            "cpp_preencode_track_frames": any(
                bool(row.get("preencoded_track_frames"))
                for row in cpp_rows
                if isinstance(row, dict)
            ),
            "cpp_preencode_cached_tail_frames": any(
                bool(row.get("preencoded_cached_tail_frames"))
                for row in cpp_rows
                if isinstance(row, dict)
            ),
            "cpp_text_init_selected_only": any(
                bool(row.get("text_init_selected_only"))
                for row in cpp_rows
                if isinstance(row, dict)
            ),
            "cpp_warmup_runs": max(
                (
                    int(row.get("warmup_runs", 0))
                    for row in cpp_rows
                    if isinstance(row, dict)
                ),
                default=0,
            ),
            "python_warmup_runs": 2 if py_rows else None,
            "cpp_frame_timing_jsonl": sorted(
                {
                    str(path)
                    for row in cpp_rows
                    if isinstance(row, dict)
                    for path in row.get(
                        "cpp_frame_timing_jsonl_runs",
                        [row.get("cpp_frame_timing_jsonl")],
                    )
                    if path
                }
            ),
            "preprocessing_timing_policy": {
                "cpp": (
                    "input_load_ms covers benchmark frame loading/decoding into host images. "
                    "When the benchmark emits the newer fields, input_directory_scan_ms, "
                    "input_sort_ms, input_frame_decode_ms, input_frame_store_ms, and "
                    "input_remainder_ms split that required input preparation. "
                    "model_e2e_ms covers the measured model session after frames are available. "
                    "cpp_required_e2e_ms is input_load_ms + model_e2e_ms and is the denominator "
                    "for Python session_e2e comparisons. Track/fr includes per-frame image "
                    "preprocess/upload unless --cpp-preencode-track-frames or "
                    "--cpp-preencode-cached-tail-frames is used. The cached-tail mode keeps "
                    "frame 1 as normal encode+propagate and pre-encodes frames 2..N-1; "
                    "preencode_total_ms and preencode_state_create_total_ms stay inside "
                    "model_e2e_ms and tail_required_model_ms, so cached-tail does not remove "
                    "work from e2e claims. "
                    "state_create_ms, tracker_create_ms, session_setup_ms, "
                    "track_encode_total_ms, track_propagate_total_ms, and "
                    "track_unattributed_total_ms provide the finer model-session split. "
                    "If --cpp-frame-timing-dir is used, per-frame C++ timing JSONL rows "
                    "separate frame0, runtime wall time, visible model time, preencode work, "
                    "required model time, propagation, artifacts, and unattributed time after "
                    "the measured session; comparison rows also derive "
                    "required core graph-compute time from those JSONL fields so kernel/dataflow "
                    "optimizations can be separated from input, graph setup, upload, artifact, "
                    "and cached-tail accounting effects. "
                    "Optional output artifacts are either disabled with --cpp-no-output-artifacts "
                    "or reported separately from model_e2e_ms."
                ),
                "python_sam3": (
                    "start_session loads, resizes, normalizes, and usually moves frames before "
                    "the timed propagate_in_video loop; the short-video event profile shows "
                    "frame 1 as the main encode step and later frames as cached propagation. "
                    "profile_stage_phase_total_ms_stats splits available official Python hooks "
                    "by start_session/add_prompt/propagate phase; start_session includes "
                    "load_resource_frames and construct_initial_input_batch when those hooks "
                    "are present in the installed SAM3 version."
                ),
            },
            "cpp_track_scopes": sorted(
                {
                    str(row.get("track_scope", "per_frame_encode_detect_propagate"))
                    for row in cpp_rows
                    if isinstance(row, dict)
                }
            ),
            "python_track_scopes": sorted(
                {
                    str(row.get("track_scope", "per_frame_encode_detect_propagate"))
                    for row in py_rows
                    if isinstance(row, dict)
                }
            ),
            "python_dtype": py_rows[0].get("python_dtype") if py_rows else None,
            "tf32_policy": py_rows[0].get("tf32_policy") if py_rows else None,
            "cpp_variants": sorted(
                {
                    str(row.get("cpp_variant", "default"))
                    for row in cpp_rows
                    if isinstance(row, dict)
                }
            ),
            "cpp_env_overrides": sorted(
                {
                    json.dumps(row.get("cpp_env_overrides", {}), sort_keys=True)
                    for row in cpp_rows
                    if isinstance(row, dict)
                }
            ),
            "cpp_output_artifacts": sorted(
                {
                    bool(row.get("cpp_output_artifacts", True))
                    for row in cpp_rows
                    if isinstance(row, dict)
                }
            ),
        },
        "cpp": cpp_rows,
        "python": py_rows,
        "comparisons": comparisons,
        "comparable_speed_rows": comparable,
        "comparable_model_e2e_rows": comparable_model_e2e,
        "comparable_session_e2e_rows": comparable_session_e2e,
        "same_contract_speed_diagnostic_rows": same_contract_speed_diagnostic,
        "same_contract_model_e2e_diagnostic_rows": (same_contract_model_e2e_diagnostic),
        "same_contract_session_e2e_diagnostic_rows": (
            same_contract_session_e2e_diagnostic
        ),
        "non_comparable_speed_rows": not_comparable,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/model-matrix-compare")
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument(
        "--frame-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Frame image format extracted from the source video for both C++ and Python.",
    )
    parser.add_argument(
        "--timed-start-frame",
        type=int,
        default=1,
        help=(
            "Start Track/fr aggregation at this zero-based frame index. Earlier propagated "
            "frames still execute so tracker/cache state is identical."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each C++ and official Python measurement this many times and report aggregate statistics.",
    )
    parser.add_argument("--point-x", type=float, default=315.0)
    parser.add_argument("--point-y", type=float, default=250.0)
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Pass --multimask to the C++ benchmark for SAM2 point-prompt quality-contract runs.",
    )
    parser.add_argument(
        "--cpp-mask-output",
        action="store_true",
        help=(
            "Do not pass --bbox-only to the C++ benchmark. Use this when comparing "
            "against official Python runs that emit masks."
        ),
    )
    parser.add_argument(
        "--encode-img-size",
        type=int,
        default=0,
        help=(
            "Override the SAM input encode size for comparable C++/SAM2 Python "
            "runs. 0 keeps each model default."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=(
            "Filter C++ model filenames. Repeat to require multiple substrings; "
            "the benchmark receives the first token and this script post-filters all tokens."
        ),
    )
    parser.add_argument("--download-ggml", action="store_true")
    parser.add_argument("--download-filter", action="append", default=[])
    parser.add_argument("--skip-cpp", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument(
        "--cpp-warmup-runs",
        type=int,
        default=2,
        help=(
            "Run this many untimed C++ same-model sessions before the measured session. "
            "The official Python runner warms up with two run_once() calls."
        ),
    )
    parser.add_argument(
        "--cpp-variant",
        default="default",
        help="Label attached to C++ result rows, useful when measuring an opt-in optimization.",
    )
    parser.add_argument(
        "--cpp-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override passed only to the C++ benchmark. May be repeated.",
    )
    parser.add_argument(
        "--cpp-preencode-track-frames",
        action="store_true",
        help=(
            "Measure C++ tracking after pre-encoding tracker features for frames 1..N-1. "
            "Use with --timed-start-frame 2 or later for the cached-propagation contract."
        ),
    )
    parser.add_argument(
        "--cpp-preencode-cached-tail-frames",
        action="store_true",
        help=(
            "Measure C++ with frame 1 encoded inside Track/fr and frames 2..N-1 pre-encoded. "
            "This mirrors the observed official SAM3 short-video session-cache timing contract."
        ),
    )
    parser.add_argument(
        "--cpp-text-init-selected-only",
        action="store_true",
        help=(
            "Pass --text-init-selected-only to C++. This measures selected-target tracking "
            "rather than the official SAM3 all-text-detections contract."
        ),
    )
    parser.add_argument(
        "--cpp-no-output-artifacts",
        action="store_true",
        help=(
            "Skip C++ benchmark JSONL/mask/logits artifact generation in the measured session. "
            "Use this for model-e2e performance comparisons; run artifact-producing parity "
            "checks separately."
        ),
    )
    parser.add_argument(
        "--cpp-frame-timing-dir",
        type=Path,
        help=(
            "Directory for per-frame C++ timing JSONL files. Relative paths are created under "
            "--out-dir. These rows are written after the measured session and are not included "
            "in cpp_model_e2e_ms."
        ),
    )
    parser.add_argument(
        "--skip-python-sam2",
        action="store_true",
        help="Skip official SAM2 Python baselines; useful for SAM3/SAM3.1-only comparisons.",
    )
    parser.add_argument(
        "--python-results",
        type=Path,
        help="Reuse a previous python-results.json instead of running official Python baselines.",
    )
    parser.add_argument(
        "--python-family",
        action="append",
        default=[],
        help="Limit official Python SAM2 runs to a family such as sam2.1_hiera_base_plus. May be repeated.",
    )
    parser.add_argument(
        "--python-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Use bf16/fp16 autocast or strict fp32 for official Python baselines.",
    )
    parser.add_argument(
        "--tf32-policy",
        choices=("on", "off"),
        default="on",
        help="Enable or disable PyTorch TF32 matmul/cuDNN. Use off with --python-dtype fp32 for Strict FP32.",
    )
    parser.add_argument(
        "--python-sam3-version",
        action="append",
        choices=("sam3", "sam3.1", SAM3_PYTHON_NONE),
        default=None,
        help=(
            "Official SAM3 Python baseline version to run. Repeat for both sam3 and sam3.1, "
            "or pass none to skip SAM3 Python."
        ),
    )
    parser.add_argument(
        "--python-sam3-use-fa3",
        action="store_true",
        help="Enable official SAM3/SAM3.1 FlashAttention 3 path when the environment provides it.",
    )
    parser.add_argument(
        "--python-sam3-compile",
        action="store_true",
        help="Enable torch.compile for official SAM3/SAM3.1. This can add a long warm-up.",
    )
    parser.add_argument(
        "--sam2-repo",
        type=Path,
        default=Path(os.environ.get("SAM2_REPO", "external/sam2")),
    )
    parser.add_argument(
        "--sam3-repo",
        type=Path,
        default=Path(os.environ.get("SAM3_REPO", "external/sam3")),
    )
    default_sam2_checkpoint_dir = Path(
        os.environ.get("SAM2_CHECKPOINT_DIR", "external/sam2/checkpoints")
    )
    parser.add_argument(
        "--sam2-tiny-checkpoint",
        type=Path,
        default=Path(os.environ["SAM2_TINY_CHECKPOINT"])
        if "SAM2_TINY_CHECKPOINT" in os.environ
        else default_sam2_checkpoint_dir / "sam2.1_hiera_tiny.pt",
    )
    parser.add_argument(
        "--sam2-small-checkpoint",
        type=Path,
        default=Path(os.environ["SAM2_SMALL_CHECKPOINT"])
        if "SAM2_SMALL_CHECKPOINT" in os.environ
        else default_sam2_checkpoint_dir / "sam2.1_hiera_small.pt",
    )
    parser.add_argument(
        "--sam2-base-plus-checkpoint",
        type=Path,
        default=Path(os.environ["SAM2_BASE_PLUS_CHECKPOINT"])
        if "SAM2_BASE_PLUS_CHECKPOINT" in os.environ
        else default_sam2_checkpoint_dir / "sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument(
        "--sam2-large-checkpoint",
        type=Path,
        default=Path(os.environ["SAM2_LARGE_CHECKPOINT"])
        if "SAM2_LARGE_CHECKPOINT" in os.environ
        else default_sam2_checkpoint_dir / "sam2.1_hiera_large.pt",
    )
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.models_dir = args.models_dir.resolve()
    args.video = args.video.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.frames < 2:
        raise SystemExit("--frames must be >= 2")
    if args.timed_start_frame < 1 or args.timed_start_frame >= args.frames:
        raise SystemExit("--timed-start-frame must satisfy 1 <= value < --frames")
    if args.cpp_preencode_track_frames and args.cpp_preencode_cached_tail_frames:
        raise SystemExit(
            "--cpp-preencode-track-frames and --cpp-preencode-cached-tail-frames are mutually exclusive"
        )

    if args.download_ggml:
        download_ggml(args.models_dir, args.download_filter)

    frame_dir = args.out_dir / "frames"
    extract_frames(args.video, frame_dir, args.frames, args.frame_format)
    decoded_width, decoded_height = decoded_frame_size(frame_dir, args.frame_format)

    cpp_rows: list[dict[str, Any]] = []
    if not args.skip_cpp:
        cpp_rows = run_cpp(args, args.out_dir, frame_dir)

    py_rows: list[dict[str, Any]] = []
    if args.python_results is not None:
        py_rows = json.loads(args.python_results.resolve().read_text(encoding="utf-8"))
        validate_reused_python_results(py_rows, args)
        (args.out_dir / "python-results.json").write_text(
            json.dumps(py_rows, indent=2), encoding="utf-8"
        )
    elif not args.skip_python:
        if should_run_python_sam2(args):
            py_rows.extend(run_python_sam2(args, args.out_dir, frame_dir))
        py_rows.extend(run_python_sam3(args, args.out_dir, frame_dir))
        (args.out_dir / "python-results.json").write_text(
            json.dumps(py_rows, indent=2), encoding="utf-8"
        )

    summary = summarize(
        cpp_rows,
        py_rows,
        args.out_dir,
        decoded_width=decoded_width,
        decoded_height=decoded_height,
        frames=args.frames,
        prompt=args.text_prompt,
        point_x=args.point_x,
        point_y=args.point_y,
        requested_encode_size=args.encode_img_size,
        multimask=args.multimask,
        frame_format=args.frame_format,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
