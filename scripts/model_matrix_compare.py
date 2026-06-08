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
import shutil
import subprocess
import sys
import time
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


def uv_run_python(project: Path, code: str, argv: list[str], deps: list[str]) -> list[str]:
    _ = project
    cmd = ["uv", "run", "--no-project"]
    for dep in deps:
        cmd.extend(["--with", dep])
    return cmd + ["python", "-c", code, *argv]


def run(cmd: list[str], cwd: Path, log_path: Path | None = None, env: dict[str, str] | None = None) -> str:
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
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
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
    return rows


def run_cpp(args: argparse.Namespace, out_dir: Path, frame_dir: Path) -> list[dict[str, Any]]:
    run(["just", "build"], cwd=args.repo, log_path=out_dir / "cpp-build.log")
    benchmark = args.repo / "build/xmake-release-cuda/examples/sam3_benchmark"
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
    ]
    if args.multimask:
        cmd.append("--multimask")
    if args.encode_img_size > 0:
        cmd.extend(["--encode-img-size", str(args.encode_img_size)])
    if benchmark_filter:
        cmd.extend(["--filter", benchmark_filter])
    repeated_rows: list[dict[str, Any]] = []
    for run_index in range(1, args.repeats + 1):
        text = run(cmd, cwd=args.repo, log_path=out_dir / f"cpp-benchmark-{run_index:02d}.log")
        rows = parse_cpp_table(text)
        for row in rows:
            row["run"] = run_index
            repeated_rows.append(row)
    rows = repeated_rows
    if len(filter_tokens) > 1:
        rows = [row for row in rows if all(token in row["model"] for token in filter_tokens)]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["family"], row["precision"]), []).append(row)
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
        row["p50_ms"] = statistics.mean(float(item["p50_ms"]) for item in ok_rows)
        row["p95_ms"] = statistics.mean(float(item["p95_ms"]) for item in ok_rows)
        row["total_ms"] = statistics.mean(float(item["total_ms"]) for item in ok_rows)
        row["run"] = None
        row["model"] = model
        row["family"] = family
        row["precision"] = model_precision
        row["track_ms_stats"] = summarize_values(track)
        row["runs"] = group
        rows.append(row)
    (out_dir / "cpp-results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def extract_frames(video: Path, frame_dir: Path, frames: int) -> None:
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-frames:v",
            str(frames),
            "-q:v",
            "2",
            str(frame_dir / "%05d.jpg"),
        ],
        cwd=frame_dir.parent,
    )


def decoded_frame_size(frame_dir: Path) -> tuple[int, int]:
    first = frame_dir / "00001.jpg"
    with Image.open(first) as img:
        return img.size


def cpp_effective_encode_size(row: dict[str, Any], requested_encode_size: int) -> int | None:
    if requested_encode_size > 0:
        return requested_encode_size
    if row["family"].startswith("sam2.1_hiera_"):
        return 1024
    if row["family"] in {"sam3", "sam3.1"} or row["family"].startswith("sam3-visual"):
        return 1008
    return None


def run_python_sam3(args: argparse.Namespace, out_dir: Path, frame_dir: Path) -> list[dict[str, Any]]:
    sam3_repo = Path(args.sam3_repo)
    if not sam3_repo.exists():
        return []
    requested_versions = args.python_sam3_version or ["sam3.1"]
    versions = [version for version in requested_versions if version != SAM3_PYTHON_NONE]
    if not versions:
        return []
    code = r'''
import inspect, json, os, sys, time, types, uuid
import torch

sam3_repo, video_dir, prompt, frames, python_dtype, tf32_policy, repeats, version, use_fa3, compile_model = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7]), sys.argv[8], sys.argv[9] == "1", sys.argv[10] == "1"
sys.path.insert(0, sam3_repo)
if python_dtype == "bf16":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
allow_tf32 = tf32_policy == "on"
torch.backends.cuda.matmul.allow_tf32 = allow_tf32
torch.backends.cudnn.allow_tf32 = allow_tf32
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
from sam3.model_builder import build_sam3_predictor

predictor = build_sam3_predictor(
    version=version,
    compile=compile_model,
    use_fa3=use_fa3,
    async_loading_frames=False,
)
if "offload_state_to_cpu" not in inspect.signature(predictor.model.init_state).parameters:
    def start_session_without_state_offload(self, resource_path, session_id=None, offload_video_to_cpu=False, offload_state_to_cpu=False):
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        inference_state = self.model.init_state(**init_kwargs)
        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}
    predictor.start_session = types.MethodType(start_session_without_state_offload, predictor)

def run_once(measure=False):
    response = predictor.handle_request({"type": "start_session", "resource_path": video_dir})
    session_id = response["session_id"]
    predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt})
    stream_count = 0
    t0 = time.perf_counter()
    for _response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
        stream_count += 1
        if stream_count >= frames:
            break
    torch.cuda.synchronize()
    measured_ms = (time.perf_counter() - t0) * 1000.0
    predictor.handle_request({"type": "reset_session", "session_id": session_id})
    track_count = max(min(stream_count, frames) - 1, 1)
    if measure:
        return track_count, measured_ms
    return track_count, 0.0

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
    "family": version,
    "backend": f"PyTorch CUDA {python_dtype}",
    "python_dtype": python_dtype,
    "tf32_policy": tf32_policy,
    "sam3_version": version,
    "sam3_use_fa3": use_fa3,
    "sam3_compile": compile_model,
    "torch_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    "torch_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    "frames": frames,
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
    "image_size": 1008,
}))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam3_repo)
    rows: list[dict[str, Any]] = []
    for version in versions:
        try:
            text = run(
                uv_run_python(
                    sam3_repo,
                    code,
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
                    ],
                    SAM3_UV_DEPS,
                ),
                cwd=args.repo,
                log_path=out_dir / f"python-{version}.log",
                env=env,
            )
        except Exception as exc:
            rows.append({"family": version, "backend": f"PyTorch CUDA {args.python_dtype}", "error": str(exc)})
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


def run_python_sam2(args: argparse.Namespace, out_dir: Path, frame_dir: Path) -> list[dict[str, Any]]:
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
    code = r'''
import json, sys, time
import numpy as np
import torch

sam2_repo, checkpoint, cfg, video_dir, point_x, point_y, image_size, frames, python_dtype, tf32_policy, repeats = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9], sys.argv[10], int(sys.argv[11])
sys.path.insert(0, sam2_repo)
if python_dtype == "bf16":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
allow_tf32 = tf32_policy == "on"
torch.backends.cuda.matmul.allow_tf32 = allow_tf32
torch.backends.cudnn.allow_tf32 = allow_tf32
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
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
    "torch_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    "torch_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
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
'''
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
            rows.append({"family": family, "backend": "PyTorch CUDA bf16", "error": str(exc)})
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                row = json.loads(line)
                row["family"] = family
                rows.append(row)
                break
    return rows


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
) -> dict[str, Any]:
    py_by_family = {row["family"]: row for row in py_rows}
    comparisons: list[dict[str, Any]] = []
    for row in cpp_rows:
        if not row.get("ok", True):
            continue
        py = py_by_family.get(row["family"])
        if not py:
            # SAM2 and SAM2.1 naming is already aligned; SAM3 visual has no official
            # separate Python baseline.
            continue
        if "error" in py:
            comparisons.append(
                {
                    "model": row["model"],
                    "family": row["family"],
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
        comparable_speed_claim = (
            same_decoded_source_resolution
            and same_frame_range
            and same_prompt
            and same_encode_size
        )
        track_ratio = py["track_ms"] / row["track_ms"] if row["track_ms"] > 0 else None
        comparisons.append(
            {
                "model": row["model"],
                "family": row["family"],
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
                "comparable_speed_claim": comparable_speed_claim,
                "cpp_track_ms": row["track_ms"],
                "python_track_ms": py["track_ms"],
                "python_track_frames": py.get("track_frames"),
                "python_dtype": py.get("python_dtype"),
                "tf32_policy": py.get("tf32_policy"),
                "python_over_cpp_track_ratio": track_ratio,
                "python_over_cpp_track_ratio_if_comparable": track_ratio if comparable_speed_claim else None,
                "cpp_rss_mib": row["rss_mib"],
                "python_cuda_alloc_mib": py["rss_mib"],
            }
        )
    comparable = [row for row in comparisons if row.get("comparable_speed_claim")]
    not_comparable = [row for row in comparisons if not row.get("comparable_speed_claim")]
    summary = {
        "comparison_contract": {
            "speed_and_quality_claims_require": [
                "same decoded source-frame resolution",
                "same frame range",
                "same measured tracking range (frame 0 add-instance excluded)",
                "same prompt",
                "same SAM model input resolution",
            ],
            "note": (
                "Rows with comparable_speed_claim=false are scaling or coverage rows. "
                "Do not use their C++/Python ratios as speed win/loss evidence."
            ),
        },
        "benchmark_context": {
            "decoded_source_width": decoded_width,
            "decoded_source_height": decoded_height,
            "frames": frames,
            "point_prompt": {"x": point_x, "y": point_y},
            "text_prompt": prompt,
            "requested_encode_img_size": requested_encode_size,
            "multimask": multimask,
            "python_dtype": py_rows[0].get("python_dtype") if py_rows else None,
            "tf32_policy": py_rows[0].get("tf32_policy") if py_rows else None,
        },
        "cpp": cpp_rows,
        "python": py_rows,
        "comparisons": comparisons,
        "comparable_speed_rows": comparable,
        "non_comparable_speed_rows": not_comparable,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/model-matrix-compare"))
    parser.add_argument("--frames", type=int, default=10)
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
        choices=("bf16", "fp32"),
        default="bf16",
        help="Use bf16 autocast or strict fp32 for official Python baselines.",
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
    parser.add_argument("--sam2-repo", type=Path, default=Path(os.environ.get("SAM2_REPO", "external/sam2")))
    parser.add_argument("--sam3-repo", type=Path, default=Path(os.environ.get("SAM3_REPO", "external/sam3")))
    default_sam2_checkpoint_dir = Path(os.environ.get("SAM2_CHECKPOINT_DIR", "external/sam2/checkpoints"))
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

    if args.download_ggml:
        download_ggml(args.models_dir, args.download_filter)

    frame_dir = args.out_dir / "frames"
    extract_frames(args.video, frame_dir, args.frames)
    decoded_width, decoded_height = decoded_frame_size(frame_dir)

    cpp_rows: list[dict[str, Any]] = []
    if not args.skip_cpp:
        cpp_rows = run_cpp(args, args.out_dir, frame_dir)

    py_rows: list[dict[str, Any]] = []
    if args.python_results is not None:
        py_rows = json.loads(args.python_results.resolve().read_text(encoding="utf-8"))
        (args.out_dir / "python-results.json").write_text(json.dumps(py_rows, indent=2), encoding="utf-8")
    elif not args.skip_python:
        if not args.skip_python_sam2:
            py_rows.extend(run_python_sam2(args, args.out_dir, frame_dir))
        py_rows.extend(run_python_sam3(args, args.out_dir, frame_dir))
        (args.out_dir / "python-results.json").write_text(json.dumps(py_rows, indent=2), encoding="utf-8")

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
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
