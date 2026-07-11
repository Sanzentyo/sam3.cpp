# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(r"^SAM3_BENCH_RESULT_JSON (\{.*\})$", re.MULTILINE)
DEFAULT_METRICS = (
    "required_e2e_ms",
    "tail_encode_graph_compute_ms",
    "required_core_compute_ms",
)
MEMORY_METRICS = (
    "host_rss_current_after_warmups_bytes",
    "host_rss_current_final_bytes",
    "host_rss_peak_after_warmups_bytes",
    "host_rss_peak_final_bytes",
    "backend_device_used_after_model_load_bytes",
    "backend_device_used_after_warmups_bytes",
    "backend_device_used_final_bytes",
    "cuda_pool_current_reserved_bytes",
    "cuda_pool_peak_reserved_bytes",
    "cuda_pool_current_used_bytes",
    "cuda_pool_peak_used_bytes",
    "cuda_pool_largest_request_bytes",
    "cuda_pool_allocation_count",
    "cuda_pool_reuse_count",
)
DEFAULT_MEMORY_ACCEPT_METRICS = (
    "host_rss_peak_final_bytes",
    "backend_device_used_final_bytes",
    "cuda_pool_peak_reserved_bytes",
    "cuda_pool_peak_used_bytes",
)
EXTERNAL_MEMORY_METRIC = "nvidia_smi_peak_process_used_gpu_memory_bytes"
SAM3_TUNING_PREFIXES = (
    "SAM3_BENCH_",
    "SAM3_BF16_",
    "SAM3_CAPTURE_",
    "SAM3_CONV_",
    "SAM3_CUDA_",
    "SAM3_DEBUG_",
    "SAM3_DISABLE_",
    "SAM3_DUMP_",
    "SAM3_ENCODE_",
    "SAM3_ENABLE_",
    "SAM3_F16_",
    "SAM3_FAST_",
    "SAM3_FORCE_",
    "SAM3_HIERA_",
    "SAM3_LOG_",
    "SAM3_MEM_",
    "SAM3_PCS_",
    "SAM3_PROFILE",
    "SAM3_PROP_",
    "SAM3_USE_",
    "SAM3_VIT_",
)
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.1447866879,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_env(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise SystemExit(f"invalid environment override: {value!r}")
        result[key] = item
    return result


def parse_result(stdout: str, label: str) -> dict[str, Any]:
    matches = RESULT_RE.findall(stdout)
    if len(matches) != 1:
        raise RuntimeError(
            f"{label}: expected one benchmark result, found {len(matches)}"
        )
    result = json.loads(matches[0])
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(f"{label}: benchmark did not report ok=true")
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_repo_artifacts(benchmark: Path, repo: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["ldd", str(benchmark)], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to inspect benchmark dependencies: {completed.stderr}"
        )

    artifacts = {benchmark.resolve()}
    for line in completed.stdout.splitlines():
        field = line.split("=>", 1)[1] if "=>" in line else line
        token = field.strip().split(maxsplit=1)[0] if field.strip() else ""
        if not token.startswith("/"):
            continue
        path = Path(token).resolve()
        if path.is_file() and path.is_relative_to(repo):
            artifacts.add(path)
    return tuple(sorted(artifacts))


def artifact_stats(paths: tuple[Path, ...]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for path in paths:
        stat = path.stat()
        result[str(path)] = {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def artifact_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def require_unchanged_artifacts(
    paths: tuple[Path, ...], expected: dict[str, dict[str, int]], label: str
) -> None:
    current = artifact_stats(paths)
    if current != expected:
        raise RuntimeError(f"{label}: benchmark artifacts changed during the A/B run")


def wait_for_gpu_compute_idle(timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    reported_active = False
    while True:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to query active GPU compute processes: "
                f"{completed.stderr.strip()}"
            )
        active = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
        if not active:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU remained busy with compute processes: {active}")
        if not reported_active:
            print(f"waiting for GPU compute idle: {active}", flush=True)
            reported_active = True
        time.sleep(0.25)


def is_tuning_environment(key: str) -> bool:
    return key.startswith("GGML_CUDA_") or key.startswith(SAM3_TUNING_PREFIXES)


def is_recorded_environment(key: str) -> bool:
    return key.startswith(("GGML_CUDA_", "SAM3_"))


def metric_unit(metric: str) -> str:
    if metric.endswith("_ms"):
        return "ms"
    if metric.endswith("_bytes"):
        return "bytes"
    if metric.endswith("_mib"):
        return "MiB"
    if metric.endswith("_count"):
        return "count"
    if metric.endswith("_fraction"):
        return "ratio"
    return "number"


def metric_summary(
    baseline: list[float], candidate: list[float], t_critical: float, unit: str
) -> dict[str, Any]:
    deltas = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
    ]
    delta_mean = statistics.fmean(deltas)
    delta_sd = statistics.stdev(deltas)
    delta_se = delta_sd / math.sqrt(len(deltas))
    ci_half = t_critical * delta_se
    baseline_mean = statistics.fmean(baseline)
    candidate_mean = statistics.fmean(candidate)
    result = {
        "unit": unit,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "paired_delta_mean": delta_mean,
        "paired_delta_sd": delta_sd,
        "paired_delta_se": delta_se,
        "paired_t": delta_mean / delta_se if delta_se else None,
        "ci95_low": delta_mean - ci_half,
        "ci95_high": delta_mean + ci_half,
        "candidate_change_pct": (
            (candidate_mean / baseline_mean - 1.0) * 100.0 if baseline_mean else None
        ),
        "baseline_samples": baseline,
        "candidate_samples": candidate,
        "paired_deltas": deltas,
    }
    if unit == "ms":
        result.update(
            {
                "baseline_mean_ms": baseline_mean,
                "candidate_mean_ms": candidate_mean,
                "paired_delta_mean_ms": delta_mean,
                "paired_delta_sd_ms": delta_sd,
                "paired_delta_se_ms": delta_se,
                "ci95_low_ms": delta_mean - ci_half,
                "ci95_high_ms": delta_mean + ci_half,
                "speedup_pct": (
                    (baseline_mean / candidate_mean - 1.0) * 100.0
                    if candidate_mean
                    else None
                ),
                "baseline_samples_ms": baseline,
                "candidate_samples_ms": candidate,
                "paired_deltas_ms": deltas,
            }
        )
    return result


def descendant_pids(root_pid: int) -> set[int]:
    descendants = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = children_path.read_text(encoding="utf-8").split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for value in children:
            try:
                child = int(value)
            except ValueError:
                continue
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def sample_nvidia_smi(root_pid: int, started_at: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except subprocess.TimeoutExpired:
        return {
            "elapsed_ms": (time.monotonic() - started_at) * 1000.0,
            "ok": False,
            "error": "nvidia-smi timed out",
        }
    if completed.returncode != 0:
        return {
            "elapsed_ms": (time.monotonic() - started_at) * 1000.0,
            "ok": False,
            "error": completed.stderr.strip() or f"exit {completed.returncode}",
        }

    benchmark_pids = descendant_pids(root_pid)
    processes: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            used_mib = int(fields[1])
        except ValueError:
            continue
        if pid in benchmark_pids:
            processes.append({"pid": pid, "used_gpu_memory_mib": used_mib})
    return {
        "elapsed_ms": (time.monotonic() - started_at) * 1000.0,
        "ok": True,
        "benchmark_pids": sorted(benchmark_pids),
        "processes": processes,
        "total_used_gpu_memory_mib": sum(
            process["used_gpu_memory_mib"] for process in processes
        ),
    }


def run_benchmark_process(
    command: list[str],
    repo: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    nvidia_smi_sample_interval_ms: float | None,
) -> tuple[int, str, str, dict[str, Any] | None]:
    with (
        stdout_path.open("w+", encoding="utf-8") as stdout_handle,
        stderr_path.open("w+", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        sampler_result: dict[str, Any] | None = None
        if nvidia_smi_sample_interval_ms is None:
            returncode = process.wait()
        else:
            started_at = time.monotonic()
            samples: list[dict[str, Any]] = []
            interval_seconds = nvidia_smi_sample_interval_ms / 1000.0
            next_sample = started_at
            while process.poll() is None:
                now = time.monotonic()
                if now < next_sample:
                    time.sleep(next_sample - now)
                samples.append(sample_nvidia_smi(process.pid, started_at))
                next_sample = max(next_sample + interval_seconds, time.monotonic())
            returncode = process.wait()
            valid_samples = [sample for sample in samples if sample.get("ok")]
            sampler_result = {
                "interval_ms": nvidia_smi_sample_interval_ms,
                "sample_count": len(samples),
                "valid_sample_count": len(valid_samples),
                "peak_process_used_gpu_memory_mib": max(
                    (sample["total_used_gpu_memory_mib"] for sample in valid_samples),
                    default=0,
                ),
                "samples": samples,
            }
        stdout_handle.flush()
        stderr_handle.flush()
        stdout_handle.seek(0)
        stderr_handle.seek(0)
        return returncode, stdout_handle.read(), stderr_handle.read(), sampler_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run counterbalanced SAM3 benchmark pairs and report paired confidence intervals."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--model-filter", default="sam3-f16")
    parser.add_argument("--point-x", type=float, default=315.0)
    parser.add_argument("--point-y", type=float, default=250.0)
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--common-env", action="append", default=[])
    parser.add_argument("--baseline-env", action="append", default=[])
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--accept-metric", action="append", default=[])
    parser.add_argument(
        "--track-memory",
        action="store_true",
        help="collect internal RSS, backend-memory, and CUDA-pool statistics",
    )
    parser.add_argument(
        "--memory-accept-metric",
        action="append",
        default=[],
        help="byte metric subject to the optional memory noninferiority budget",
    )
    parser.add_argument(
        "--memory-noninferiority-mib",
        "--memory-noninferiority-budget-mib",
        type=float,
        default=None,
        help="maximum allowed upper 95%% CI for candidate minus baseline memory",
    )
    parser.add_argument(
        "--nvidia-smi-sample-interval-ms",
        type=float,
        default=None,
        help="opt-in external per-process GPU-memory sampling (perturbs timing)",
    )
    parser.add_argument(
        "--gpu-lock-file",
        type=Path,
        default=Path("/tmp/sam3-cuda-benchmark-gpu0.lock"),
    )
    parser.add_argument("--gpu-idle-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")
    if args.frames < 2:
        raise SystemExit("--frames must be at least 2")
    if args.gpu_idle_timeout_seconds < 0:
        raise SystemExit("--gpu-idle-timeout-seconds must be non-negative")
    if (
        args.memory_noninferiority_mib is not None
        and args.memory_noninferiority_mib < 0
    ):
        raise SystemExit("--memory-noninferiority-mib must be non-negative")
    if args.memory_accept_metric and not args.track_memory:
        raise SystemExit("--memory-accept-metric requires --track-memory")
    if args.memory_noninferiority_mib is not None and not args.track_memory:
        raise SystemExit("--memory-noninferiority-mib requires --track-memory")
    if args.nvidia_smi_sample_interval_ms is not None and not (
        10.0 <= args.nvidia_smi_sample_interval_ms <= 1000.0
    ):
        raise SystemExit("--nvidia-smi-sample-interval-ms must be between 10 and 1000")

    repo = Path.cwd().resolve()
    benchmark = args.benchmark.resolve()
    models_dir = args.models_dir.resolve()
    frame_dir = args.frame_dir.resolve()
    if not benchmark.is_file():
        raise SystemExit(f"benchmark does not exist: {benchmark}")
    if not models_dir.is_dir():
        raise SystemExit(f"models directory does not exist: {models_dir}")
    if not frame_dir.is_dir():
        raise SystemExit(f"frame directory does not exist: {frame_dir}")

    performance_metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS
    metrics = tuple(
        dict.fromkeys(
            [
                *performance_metrics,
                *(MEMORY_METRICS if args.track_memory else ()),
                *(
                    (EXTERNAL_MEMORY_METRIC,)
                    if args.nvidia_smi_sample_interval_ms is not None
                    else ()
                ),
            ]
        )
    )
    accept_metrics = (
        tuple(args.accept_metric) if args.accept_metric else performance_metrics[:2]
    )
    unknown_accept_metrics = set(accept_metrics) - set(metrics)
    if unknown_accept_metrics:
        raise SystemExit(
            f"accept metrics are not collected: {sorted(unknown_accept_metrics)}"
        )
    non_time_accept_metrics = [
        metric for metric in accept_metrics if metric_unit(metric) != "ms"
    ]
    if non_time_accept_metrics:
        raise SystemExit(
            "performance accept metrics must use milliseconds: "
            f"{non_time_accept_metrics}"
        )

    memory_accept_metrics = (
        tuple(args.memory_accept_metric)
        if args.memory_accept_metric
        else (
            *DEFAULT_MEMORY_ACCEPT_METRICS,
            *((EXTERNAL_MEMORY_METRIC,) if args.nvidia_smi_sample_interval_ms else ()),
        )
    )
    supported_memory_accept_metrics = {
        *MEMORY_METRICS,
        *(
            (EXTERNAL_MEMORY_METRIC,)
            if args.nvidia_smi_sample_interval_ms is not None
            else ()
        ),
    }
    unknown_memory_accept_metrics = (
        set(memory_accept_metrics) - supported_memory_accept_metrics
    )
    if unknown_memory_accept_metrics:
        raise SystemExit(
            f"unknown memory accept metrics: {sorted(unknown_memory_accept_metrics)}"
        )

    variants = {
        "baseline": parse_env(args.baseline_env),
        "candidate": parse_env(args.candidate_env),
    }
    common_env = parse_env(args.common_env)
    if args.track_memory:
        if common_env.get("GGML_CUDA_TRACK_POOL_STATS", "1") != "1":
            raise SystemExit("--track-memory requires GGML_CUDA_TRACK_POOL_STATS=1")
        common_env["GGML_CUDA_TRACK_POOL_STATS"] = "1"
        for variant, overrides in variants.items():
            if overrides.get("GGML_CUDA_TRACK_POOL_STATS", "1") != "1":
                raise SystemExit(
                    f"{variant} disables GGML_CUDA_TRACK_POOL_STATS while "
                    "--track-memory is active"
                )

    lock_path = args.gpu_lock_file.expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    print(f"waiting for GPU benchmark lock: {lock_path}", flush=True)
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    print(f"acquired GPU benchmark lock: {lock_path}", flush=True)
    wait_for_gpu_compute_idle(args.gpu_idle_timeout_seconds)

    artifact_paths = discover_repo_artifacts(benchmark, repo)
    initial_artifact_stats = artifact_stats(artifact_paths)
    initial_artifact_hashes = artifact_hashes(artifact_paths)
    args.out.mkdir(parents=True, exist_ok=False)

    command = [
        str(benchmark),
        "--models-dir",
        str(models_dir),
        "--frame-dir",
        str(frame_dir),
        "--gpu-only",
        "--bbox-only",
        "--quiet",
        "--n-frames",
        str(args.frames),
        "--point-x",
        str(args.point_x),
        "--point-y",
        str(args.point_y),
        "--text-prompt",
        args.text_prompt,
        "--filter",
        args.model_filter,
        "--warmup-runs",
        str(args.warmup_runs),
        "--timed-start-frame",
        "1",
        "--preencode-cached-tail-frames",
        "--text-init-selected-only",
        "--no-output-artifacts",
    ]

    rows: dict[str, dict[str, dict[str, Any]]] = {
        "baseline": {},
        "candidate": {},
    }
    order: list[dict[str, Any]] = []
    external_memory_samples: dict[str, dict[str, Any]] = {}
    base_env = os.environ.copy()
    cleared_inherited_environment = {
        key: base_env[key] for key in sorted(base_env) if is_tuning_environment(key)
    }
    inherited_sam3_environment = {
        key: base_env[key] for key in sorted(base_env) if key.startswith("SAM3_")
    }
    for key in tuple(base_env):
        if is_tuning_environment(key):
            base_env.pop(key)
    base_env.update(common_env)
    effective_variant_environment = {
        variant: {
            key: value
            for key, value in sorted((base_env | overrides).items())
            if is_recorded_environment(key)
        }
        for variant, overrides in variants.items()
    }

    for pair in range(1, args.repeats + 1):
        pair_id = f"{pair:02d}"
        pair_order = (
            ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")
        )
        order.append({"pair": pair, "order": list(pair_order)})
        for variant in pair_order:
            label = f"{variant}_{pair_id}"
            environment = base_env | variants[variant]
            wait_for_gpu_compute_idle(args.gpu_idle_timeout_seconds)
            require_unchanged_artifacts(artifact_paths, initial_artifact_stats, label)
            stdout_path = args.out / f"{label}.stdout"
            stderr_path = args.out / f"{label}.stderr"
            returncode, stdout, stderr, sampler_result = run_benchmark_process(
                command,
                repo,
                environment,
                stdout_path,
                stderr_path,
                args.nvidia_smi_sample_interval_ms,
            )
            if sampler_result is not None:
                atomic_write_json(args.out / f"{label}.nvidia-smi.json", sampler_result)
                external_memory_samples[label] = {
                    key: value
                    for key, value in sampler_result.items()
                    if key != "samples"
                }
            require_unchanged_artifacts(artifact_paths, initial_artifact_stats, label)
            if returncode != 0 or "SUMMARY: 1 runs, 1 OK, 0 FAIL" not in stdout:
                raise RuntimeError(
                    f"{label}: benchmark failed with exit {returncode}; "
                    f"stderr: {stderr[-1000:]}"
                )
            result = parse_result(stdout, label)
            if sampler_result is not None:
                peak_mib = sampler_result["peak_process_used_gpu_memory_mib"]
                if sampler_result["valid_sample_count"] == 0 or peak_mib <= 0:
                    raise RuntimeError(
                        f"{label}: nvidia-smi sampler captured no process memory"
                    )
                result[EXTERNAL_MEMORY_METRIC] = peak_mib * 1024 * 1024
            for metric in metrics:
                if isinstance(result.get(metric), bool) or not isinstance(
                    result.get(metric), int | float
                ):
                    raise RuntimeError(f"{label}: missing numeric metric {metric!r}")
            if (
                args.track_memory
                and result.get("cuda_pool_tracking_enabled") is not True
            ):
                raise RuntimeError(
                    f"{label}: CUDA pool memory tracking was not enabled"
                )
            rows[variant][pair_id] = result
            print(f"[{pair:02d}/{args.repeats:02d}] {variant} ok", flush=True)

    pair_ids = sorted(set(rows["baseline"]) & set(rows["candidate"]))
    if len(pair_ids) != args.repeats:
        raise RuntimeError(
            f"expected {args.repeats} complete pairs, found {len(pair_ids)}"
        )

    t_critical = T_CRIT_95.get(args.repeats - 1, 1.96)
    metric_results: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        baseline_samples = [float(rows["baseline"][pair][metric]) for pair in pair_ids]
        candidate_samples = [
            float(rows["candidate"][pair][metric]) for pair in pair_ids
        ]
        metric_results[metric] = metric_summary(
            baseline_samples, candidate_samples, t_critical, metric_unit(metric)
        )

    timing_perturbed = (
        args.track_memory or args.nvidia_smi_sample_interval_ms is not None
    )
    performance_accept = (
        None
        if timing_perturbed
        else all(metric_results[metric]["ci95_high"] < 0.0 for metric in accept_metrics)
    )
    memory_budget_bytes = (
        args.memory_noninferiority_mib * 1024.0 * 1024.0
        if args.memory_noninferiority_mib is not None
        else None
    )
    memory_accept = (
        all(
            metric_results[metric]["ci95_high"] <= memory_budget_bytes
            for metric in memory_accept_metrics
        )
        if memory_budget_bytes is not None
        else None
    )
    accepted = (
        None
        if performance_accept is None and memory_accept is None
        else (performance_accept is not False) and (memory_accept is not False)
    )
    final_artifact_hashes = artifact_hashes(artifact_paths)
    if final_artifact_hashes != initial_artifact_hashes:
        raise RuntimeError("benchmark artifact content changed during the A/B run")
    summary = {
        "schema_version": 3,
        "n": args.repeats,
        "order": order,
        "t_critical_95": t_critical,
        "benchmark": str(benchmark),
        "models_dir": str(models_dir),
        "frame_dir": str(frame_dir),
        "gpu_lock_file": str(lock_path),
        "gpu_idle_timeout_seconds": args.gpu_idle_timeout_seconds,
        "command": command,
        "common_environment": common_env,
        "variants": variants,
        "effective_variant_environment": effective_variant_environment,
        "cleared_inherited_environment": cleared_inherited_environment,
        "inherited_sam3_environment": inherited_sam3_environment,
        "artifacts": {
            path: {
                **initial_artifact_stats[path],
                "sha256": initial_artifact_hashes[path],
            }
            for path in initial_artifact_stats
        },
        "metrics": metric_results,
        "metric_units": {metric: metric_unit(metric) for metric in metrics},
        "accept_metrics": list(accept_metrics),
        "performance_accept_metrics": list(accept_metrics),
        "performance_acceptance_evaluated": not timing_perturbed,
        "performance_accept": performance_accept,
        "track_memory": args.track_memory,
        "memory": {
            "metrics": [
                *(MEMORY_METRICS if args.track_memory else ()),
                *((EXTERNAL_MEMORY_METRIC,) if timing_perturbed else ()),
            ],
            "accept_metrics": (
                list(memory_accept_metrics) if memory_budget_bytes is not None else []
            ),
            "noninferiority_budget_mib": args.memory_noninferiority_mib,
            "noninferiority_budget_bytes": memory_budget_bytes,
            "accept": memory_accept,
            "cuda_pool_kind_samples": {
                variant: [
                    rows[variant][pair].get("cuda_pool_kind") for pair in pair_ids
                ]
                for variant in ("baseline", "candidate")
            }
            if args.track_memory
            else {},
            "cuda_pool_tracking_enabled_samples": {
                variant: [
                    rows[variant][pair].get("cuda_pool_tracking_enabled")
                    for pair in pair_ids
                ]
                for variant in ("baseline", "candidate")
            }
            if args.track_memory
            else {},
        },
        "timing_perturbed": timing_perturbed,
        "external_nvidia_smi": {
            "enabled": args.nvidia_smi_sample_interval_ms is not None,
            "interval_ms": args.nvidia_smi_sample_interval_ms,
            "timing_acceptance_disabled": args.nvidia_smi_sample_interval_ms
            is not None,
            "runs": external_memory_samples,
        },
        "accept": accepted,
    }
    atomic_write_json(args.out / "paired_summary.json", summary)
    print(json.dumps(summary, indent=2))
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    return 0 if accepted is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
