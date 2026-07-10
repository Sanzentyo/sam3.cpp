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


def metric_summary(
    baseline: list[float], candidate: list[float], t_critical: float
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
    return {
        "baseline_mean_ms": baseline_mean,
        "candidate_mean_ms": candidate_mean,
        "paired_delta_mean_ms": delta_mean,
        "paired_delta_sd_ms": delta_sd,
        "paired_delta_se_ms": delta_se,
        "paired_t": delta_mean / delta_se if delta_se else None,
        "ci95_low_ms": delta_mean - ci_half,
        "ci95_high_ms": delta_mean + ci_half,
        "speedup_pct": (baseline_mean / candidate_mean - 1.0) * 100.0,
        "baseline_samples_ms": baseline,
        "candidate_samples_ms": candidate,
        "paired_deltas_ms": deltas,
    }


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

    metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS
    accept_metrics = tuple(args.accept_metric) if args.accept_metric else metrics[:2]
    unknown_accept_metrics = set(accept_metrics) - set(metrics)
    if unknown_accept_metrics:
        raise SystemExit(
            f"accept metrics are not collected: {sorted(unknown_accept_metrics)}"
        )

    variants = {
        "baseline": parse_env(args.baseline_env),
        "candidate": parse_env(args.candidate_env),
    }
    common_env = parse_env(args.common_env)

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
    base_env = os.environ.copy()
    cleared_inherited_environment = {
        key: base_env[key] for key in sorted(base_env) if key.startswith("GGML_CUDA_")
    }
    for key in tuple(base_env):
        if key.startswith("GGML_CUDA_"):
            base_env.pop(key)
    base_env.update(common_env)
    effective_variant_environment = {
        variant: {
            key: value
            for key, value in sorted((base_env | overrides).items())
            if key.startswith("GGML_CUDA_")
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
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            (args.out / f"{label}.stdout").write_text(
                completed.stdout, encoding="utf-8"
            )
            (args.out / f"{label}.stderr").write_text(
                completed.stderr, encoding="utf-8"
            )
            require_unchanged_artifacts(artifact_paths, initial_artifact_stats, label)
            if (
                completed.returncode != 0
                or "SUMMARY: 1 runs, 1 OK, 0 FAIL" not in completed.stdout
            ):
                raise RuntimeError(
                    f"{label}: benchmark failed with exit {completed.returncode}"
                )
            result = parse_result(completed.stdout, label)
            for metric in metrics:
                if not isinstance(result.get(metric), int | float):
                    raise RuntimeError(f"{label}: missing numeric metric {metric!r}")
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
            baseline_samples, candidate_samples, t_critical
        )

    accepted = all(
        metric_results[metric]["ci95_high_ms"] < 0.0 for metric in accept_metrics
    )
    final_artifact_hashes = artifact_hashes(artifact_paths)
    if final_artifact_hashes != initial_artifact_hashes:
        raise RuntimeError("benchmark artifact content changed during the A/B run")
    summary = {
        "schema_version": 2,
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
        "artifacts": {
            path: {
                **initial_artifact_stats[path],
                "sha256": initial_artifact_hashes[path],
            }
            for path in initial_artifact_stats
        },
        "metrics": metric_results,
        "accept_metrics": list(accept_metrics),
        "accept": accepted,
    }
    atomic_write_json(args.out / "paired_summary.json", summary)
    print(json.dumps(summary, indent=2))
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
