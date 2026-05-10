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
    "torch",
    "torchvision",
    "tqdm",
    "hydra-core",
    "iopath",
    "pillow",
]

SAM3_UV_DEPS = [
    "torch",
    "torchvision",
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
]


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
    for suffix in ("_f32", "_f16", "_q8_0", "_q4_1", "_q4_0", "-f32", "-f16", "-q8_0", "-q4_1", "-q4_0"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def precision(name: str) -> str:
    stem = name.removesuffix(".ggml")
    for value in ("f32", "f16", "q8_0", "q4_1", "q4_0"):
        if stem.endswith(value):
            return value
    return "unknown"


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


def run_cpp(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    run(["just", "build"], cwd=args.repo, log_path=out_dir / "cpp-build.log")
    benchmark = args.repo / "build/xmake-release-cuda/examples/sam3_benchmark"
    cmd = [
        str(benchmark),
        "--models-dir",
        str(args.models_dir),
        "--video",
        str(args.video),
        "--gpu-only",
        "--bbox-only",
        "--quiet",
        "--n-frames",
        str(args.frames),
        "--point-x",
        str(args.point_x),
        "--point-y",
        str(args.point_y),
    ]
    if args.encode_img_size > 0:
        cmd.extend(["--encode-img-size", str(args.encode_img_size)])
    if args.filter:
        cmd.extend(["--filter", args.filter])
    text = run(cmd, cwd=args.repo, log_path=out_dir / "cpp-benchmark.log")
    rows = parse_cpp_table(text)
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
    return None


def run_python_sam3(args: argparse.Namespace, out_dir: Path, frame_dir: Path) -> dict[str, Any] | None:
    sam3_repo = Path(args.sam3_repo)
    if not sam3_repo.exists():
        return None
    code = r'''
import json, os, sys, time
import torch

sam3_repo, video_dir, prompt, frames = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, sam3_repo)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
from sam3 import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3", compile=False, async_loading_frames=False)

def run_once():
    response = predictor.handle_request({"type": "start_session", "resource_path": video_dir})
    session_id = response["session_id"]
    predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt})
    count = 0
    for _response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
        count += 1
    torch.cuda.synchronize()
    predictor.handle_request({"type": "reset_session", "session_id": session_id})
    return count

for _ in range(2):
    run_once()

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
count = run_once()
torch.cuda.synchronize()
t1 = time.perf_counter()
elapsed = t1 - t0
print(json.dumps({
    "family": "sam3",
    "backend": "PyTorch CUDA bf16",
    "frames": count,
    "track_ms": elapsed * 1000.0 / max(count, 1),
    "total_ms": elapsed * 1000.0,
    "rss_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
}))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam3_repo)
    try:
        text = run(
            uv_run_python(
                sam3_repo,
                code,
                [str(sam3_repo), str(frame_dir), args.text_prompt, str(args.frames)],
                SAM3_UV_DEPS,
            ),
            cwd=args.repo,
            log_path=out_dir / "python-sam3.log",
            env=env,
        )
    except Exception as exc:
        return {"family": "sam3", "backend": "PyTorch CUDA bf16", "error": str(exc)}
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {"family": "sam3", "backend": "PyTorch CUDA bf16", "error": "could not parse Python SAM3 result"}


def run_python_sam2(args: argparse.Namespace, out_dir: Path, frame_dir: Path) -> list[dict[str, Any]]:
    sam2_repo = Path(args.sam2_repo)
    checkpoints = {
        "sam2.1_hiera_tiny": (
            args.sam2_tiny_checkpoint,
            "configs/sam2.1/sam2.1_hiera_t.yaml",
        ),
        "sam2.1_hiera_base_plus": (
            args.sam2_base_plus_checkpoint,
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
        ),
    }
    rows: list[dict[str, Any]] = []
    if not sam2_repo.exists():
        return rows
    code = r'''
import json, sys, time
import numpy as np
import torch

sam2_repo, checkpoint, cfg, video_dir, point_x, point_y, image_size = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
sys.path.insert(0, sam2_repo)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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

def run_once():
    state = predictor.init_state(video_path=video_dir)
    points = np.array([[point_x, point_y]], dtype=np.float32)
    labels = np.array([1], np.int32)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, points=points, labels=labels)
    count = 0
    for _ in predictor.propagate_in_video(state):
        count += 1
    torch.cuda.synchronize()
    return count

for _ in range(2):
    run_once()

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
count = run_once()
torch.cuda.synchronize()
t1 = time.perf_counter()
elapsed = t1 - t0
print(json.dumps({
    "backend": "PyTorch CUDA bf16",
    "frames": count,
    "track_ms": elapsed * 1000.0 / max(count, 1),
    "total_ms": elapsed * 1000.0,
    "rss_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
    "image_size": image_size if image_size > 0 else 1024,
}))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam2_repo)
    for family, (checkpoint, cfg) in checkpoints.items():
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
    parser.add_argument("--point-x", type=float, default=315.0)
    parser.add_argument("--point-y", type=float, default=250.0)
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument(
        "--encode-img-size",
        type=int,
        default=0,
        help=(
            "Override the SAM input encode size for comparable C++/SAM2 Python "
            "runs. 0 keeps each model default."
        ),
    )
    parser.add_argument("--filter", default="")
    parser.add_argument("--download-ggml", action="store_true")
    parser.add_argument("--download-filter", action="append", default=[])
    parser.add_argument("--skip-cpp", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--sam2-repo", type=Path, default=Path(os.environ.get("SAM2_REPO", "external/sam2")))
    parser.add_argument("--sam3-repo", type=Path, default=Path(os.environ.get("SAM3_REPO", "external/sam3")))
    parser.add_argument(
        "--sam2-tiny-checkpoint",
        type=Path,
        default=Path(os.environ["SAM2_TINY_CHECKPOINT"]) if "SAM2_TINY_CHECKPOINT" in os.environ else None,
    )
    parser.add_argument(
        "--sam2-base-plus-checkpoint",
        type=Path,
        default=Path(os.environ["SAM2_BASE_PLUS_CHECKPOINT"]) if "SAM2_BASE_PLUS_CHECKPOINT" in os.environ else None,
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
        cpp_rows = run_cpp(args, args.out_dir)

    py_rows: list[dict[str, Any]] = []
    if not args.skip_python:
        py_rows.extend(run_python_sam2(args, args.out_dir, frame_dir))
        py_sam3 = run_python_sam3(args, args.out_dir, frame_dir)
        if py_sam3:
            py_rows.append(py_sam3)
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
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
