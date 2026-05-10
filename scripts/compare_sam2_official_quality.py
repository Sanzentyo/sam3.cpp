# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from PIL import Image


SAM2_UV_DEPS = [
    "numpy",
    "torch",
    "torchvision",
    "tqdm",
    "hydra-core",
    "iopath",
    "pillow",
]


def run(cmd: list[str], cwd: Path, log_path: Path | None = None, env: dict[str, str] | None = None) -> str:
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
        raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def extract_frames(video: Path, frame_dir: Path, frames: int) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    if list(frame_dir.glob("*.jpg")):
        return
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def split_cpp_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    metadata = next((row for row in rows if row.get("source") == "sam3cpp-meta"), None)
    detections = [row for row in rows if "offset" in row and row.get("source") != "sam3cpp-meta"]
    return metadata, detections


def frame_size(frame_dir: Path) -> tuple[int, int]:
    first = sorted(frame_dir.glob("*.jpg"))[0]
    with Image.open(first) as img:
        return img.size


def validate_cpp_metadata(
    metadata: dict[str, Any] | None,
    args: argparse.Namespace,
    frame_dir: Path,
    python_summary: dict[str, Any],
) -> dict[str, Any]:
    expected_width, expected_height = frame_size(frame_dir)
    result: dict[str, Any] = {
        "present": metadata is not None,
        "expected_decoded_width": expected_width,
        "expected_decoded_height": expected_height,
        "expected_frames": args.frames,
        "expected_point_x": args.point_x,
        "expected_point_y": args.point_y,
        "expected_encode_img_size": python_summary.get("image_size"),
    }
    if metadata is None:
        result["status"] = "missing"
        result["message"] = "cpp JSONL does not contain a sam3cpp-meta row; condition parity was not checked"
        return result

    checks = {
        "decoded_width": metadata.get("decoded_width") == expected_width,
        "decoded_height": metadata.get("decoded_height") == expected_height,
        "frames": metadata.get("frames") == args.frames,
        "point_x": abs(float(metadata.get("point_x", float("nan"))) - args.point_x) < 1e-4,
        "point_y": abs(float(metadata.get("point_y", float("nan"))) - args.point_y) < 1e-4,
        "encode_img_size": metadata.get("encode_img_size_effective") == python_summary.get("image_size"),
    }
    result["metadata"] = metadata
    result["checks"] = checks
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        result["status"] = "failed"
        result["failed_checks"] = failed
        raise ValueError(f"C++/Python comparison condition mismatch: {', '.join(failed)}")
    result["status"] = "passed"
    return result


def mask_array(path: Path) -> list[int]:
    img = Image.open(path).convert("L")
    return [1 if value > 127 else 0 for value in img.tobytes()]


def mask_iou(lhs_path: Path, rhs_path: Path) -> float:
    lhs = mask_array(lhs_path)
    rhs = mask_array(rhs_path)
    if len(lhs) != len(rhs):
        raise ValueError(f"mask size mismatch: {lhs_path} vs {rhs_path}")
    inter = sum(1 for a, b in zip(lhs, rhs, strict=True) if a and b)
    union = sum(1 for a, b in zip(lhs, rhs, strict=True) if a or b)
    return inter / union if union else 1.0


def bbox_from_mask(path: Path) -> list[float] | None:
    img = Image.open(path).convert("L")
    width, height = img.size
    xs: list[int] = []
    ys: list[int] = []
    data = img.tobytes()
    for index, value in enumerate(data):
        if value > 127:
            xs.append(index % width)
            ys.append(index // width)
    if not xs:
        return None
    return [float(min(xs)), float(min(ys)), float(max(xs) + 1), float(max(ys) + 1)]


def bbox_iou(lhs: list[float] | None, rhs: list[float] | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    ax0, ay0, ax1, ay1 = lhs
    bx0, by0, bx1, by1 = rhs
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    lhs_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    rhs_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = lhs_area + rhs_area - inter
    return inter / denom if denom > 0.0 else 1.0


def run_python_sam2(args: argparse.Namespace, frame_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    code = r'''
import json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sam2_repo = Path(sys.argv[1])
checkpoint = sys.argv[2]
cfg = sys.argv[3]
frame_dir = Path(sys.argv[4])
out_dir = Path(sys.argv[5])
point_x = float(sys.argv[6])
point_y = float(sys.argv[7])
frames = int(sys.argv[8])
image_size = int(sys.argv[9])

sys.path.insert(0, str(sam2_repo))
from sam2.build_sam import build_sam2_video_predictor

out_dir.mkdir(parents=True, exist_ok=True)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
state = predictor.init_state(video_path=str(frame_dir))
points = np.array([[point_x, point_y]], dtype=np.float32)
labels = np.array([1], np.int32)

def save_row(frame_idx, obj_ids, mask_logits, elapsed_ms=None):
    if frame_idx >= frames:
        return None
    obj_index = 0
    if len(obj_ids) > 0 and 1 in obj_ids:
        obj_index = list(obj_ids).index(1)
    mask = (mask_logits[obj_index] > 0.0).detach().cpu().numpy()
    mask = np.squeeze(mask).astype(np.uint8) * 255
    path = out_dir / f"frame_{frame_idx:05d}.png"
    Image.fromarray(mask).save(path)
    area = int((mask > 127).sum())
    return {
        "offset": int(frame_idx),
        "expected_frame_index": int(frame_idx),
        "mask_path": str(path),
        "mask_area": area,
        "elapsed_ms": elapsed_ms,
        "source": "official-sam2",
    }

rows = {}
with torch.inference_mode():
    frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )
    row = save_row(frame_idx, obj_ids, mask_logits)
    if row is not None:
        rows[int(frame_idx)] = row
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        row = save_row(out_frame_idx, out_obj_ids, out_mask_logits, elapsed_ms)
        if row is not None:
            rows[int(out_frame_idx)] = row
        if out_frame_idx + 1 >= frames:
            break
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0

ordered = [rows[i] for i in sorted(rows) if i < frames]
for row in ordered:
    print(json.dumps(row), flush=True)
print(json.dumps({
    "summary": {
        "frames": len(ordered),
        "propagate_total_ms": total_ms,
        "propagate_ms_per_frame": total_ms / max(len(ordered), 1),
        "cuda_alloc_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "image_size": image_size if image_size > 0 else 1024,
    }
}), flush=True)
'''
    cmd = ["uv", "run", "--no-project"]
    for dep in SAM2_UV_DEPS:
        cmd.extend(["--with", dep])
    cmd.extend(
        [
            "python",
            "-c",
            code,
            str(args.sam2_repo),
            str(args.checkpoint),
            args.config,
            str(frame_dir),
            str(out_dir / "python_masks"),
            str(args.point_x),
            str(args.point_y),
            str(args.frames),
            str(args.image_size),
        ]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.sam2_repo)
    text = run(cmd, cwd=args.repo, log_path=out_dir / "python-official.log", env=env)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "summary" in row:
            summary = row["summary"]
        else:
            rows.append(row)
    (out_dir / "python.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (out_dir / "python-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cpp-jsonl", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, default=Path("../sam2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Official SAM2 image_size override. 0 keeps the config default.",
    )
    parser.add_argument("--point-x", type=float, default=315.0)
    parser.add_argument("--point-y", type=float, default=250.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sam2-official-quality"))
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.video = args.video.resolve()
    args.cpp_jsonl = args.cpp_jsonl.resolve()
    args.sam2_repo = args.sam2_repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = args.out_dir / "frames"
    extract_frames(args.video, frame_dir, args.frames)
    cpp_metadata, cpp_rows = split_cpp_rows(read_jsonl(args.cpp_jsonl))
    py_rows = run_python_sam2(args, frame_dir, args.out_dir)
    py_by_offset = {int(row["offset"]): row for row in py_rows}
    python_summary = json.loads((args.out_dir / "python-summary.json").read_text(encoding="utf-8"))
    metadata_validation = validate_cpp_metadata(cpp_metadata, args, frame_dir, python_summary)

    comparisons: list[dict[str, Any]] = []
    for cpp in cpp_rows:
        offset = int(cpp["offset"])
        py = py_by_offset.get(offset)
        if py is None or cpp.get("mask_path") is None:
            continue
        cpp_mask = (args.repo / cpp["mask_path"]).resolve()
        py_mask = Path(py["mask_path"]).resolve()
        cpp_bbox = cpp.get("bbox_xyxy")
        py_bbox = bbox_from_mask(py_mask)
        comparisons.append(
            {
                "offset": offset,
                "mask_iou": mask_iou(cpp_mask, py_mask),
                "bbox_iou": bbox_iou(cpp_bbox, py_bbox),
                "cpp_mask_area": cpp.get("mask_area"),
                "python_mask_area": py.get("mask_area"),
                "cpp_bbox_xyxy": cpp_bbox,
                "python_bbox_xyxy": py_bbox,
            }
        )

    summary = {
        "frames_compared": len(comparisons),
        "min_mask_iou": min((row["mask_iou"] for row in comparisons), default=None),
        "mean_mask_iou": sum(row["mask_iou"] for row in comparisons) / max(len(comparisons), 1),
        "min_bbox_iou": min((row["bbox_iou"] for row in comparisons if row["bbox_iou"] is not None), default=None),
        "comparisons": comparisons,
        "python_summary": python_summary,
        "metadata_validation": metadata_validation,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
