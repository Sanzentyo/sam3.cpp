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
    "decord",
    "numpy",
    "torch==2.8.0",
    "torchvision==0.23.0",
    "tqdm",
    "hydra-core",
    "iopath",
    "opencv-python-headless",
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


def selected_candidate(rows: list[dict[str, Any]]) -> int | None:
    selected = [int(row["candidate_index"]) for row in rows if row.get("selected")]
    if not selected:
        return None
    if len(selected) > 1:
        raise ValueError(f"multiple selected candidates: {selected}")
    return selected[0]


def cpp_candidate_space_index(row: dict[str, Any]) -> int | None:
    selected = row.get("selected_mask_index")
    if selected is None:
        return None
    selected = int(selected)
    if selected < 0:
        return None
    scores = row.get("decoder_iou_scores") or []
    if len(scores) == 4 and selected >= 1:
        return selected - 1
    return selected


def summarize_initial_candidates(
    cpp_candidates_path: Path | None,
    official_candidates_path: Path,
) -> dict[str, Any]:
    official_rows = [
        row
        for row in read_jsonl(official_candidates_path)
        if row.get("source") == "official-sam2-initial-candidate"
    ]
    result: dict[str, Any] = {
        "official_candidates": len(official_rows),
        "official_selected_index": selected_candidate(official_rows),
        "official_path": str(official_candidates_path),
    }
    if cpp_candidates_path is None:
        result["status"] = "not_checked"
        result["message"] = "pass --cpp-initial-candidates-jsonl to compare initial multimask selection"
        return result

    cpp_rows = [
        row
        for row in read_jsonl(cpp_candidates_path)
        if row.get("source") == "sam3cpp-initial-candidate"
    ]
    cpp_by_index = {int(row["candidate_index"]): row for row in cpp_rows}
    official_by_index = {int(row["candidate_index"]): row for row in official_rows}
    common = sorted(set(cpp_by_index) & set(official_by_index))
    per_candidate = []
    for idx in common:
        cpp = cpp_by_index[idx]
        official = official_by_index[idx]
        per_candidate.append(
            {
                "candidate_index": idx,
                "cpp_iou_score": cpp.get("iou_score"),
                "official_iou_score": official.get("iou_score"),
                "cpp_mask_area": cpp.get("mask_area"),
                "official_mask_area": official.get("mask_area"),
                "bbox_iou": bbox_iou(cpp.get("bbox_xyxy"), official.get("bbox_xyxy")),
            }
        )

    cpp_selected = selected_candidate(cpp_rows)
    result.update(
        {
            "status": "checked",
            "cpp_path": str(cpp_candidates_path),
            "cpp_candidates": len(cpp_rows),
            "cpp_selected_index": cpp_selected,
            "candidate_count_match": len(cpp_rows) == len(official_rows),
            "selected_index_match": cpp_selected == result["official_selected_index"],
            "per_candidate": per_candidate,
        }
    )
    return result


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
    cxx_input_source = metadata.get("input_source", "unknown")
    result["input_source"] = {
        "cpp": cxx_input_source,
        "python": args.python_video_source,
        "same_kind": cxx_input_source == args.python_video_source
        or (cxx_input_source == "frame_dir" and args.python_video_source == "frames"),
    }
    if args.python_video_source == "frames" and cxx_input_source != "frame_dir":
        result["input_source"]["warning"] = (
            "Python used extracted frame images while C++ metadata does not report frame_dir input; "
            "pixel decode/resize differences may affect mask IoU."
        )
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
import torch.nn.functional as F

sam2_repo = Path(sys.argv[1])
checkpoint = sys.argv[2]
cfg = sys.argv[3]
video_source = sys.argv[4]
out_dir = Path(sys.argv[5])
point_x = float(sys.argv[6])
point_y = float(sys.argv[7])
frames = int(sys.argv[8])
image_size = int(sys.argv[9])
python_dtype = sys.argv[10]
save_python_logits = bool(int(sys.argv[11]))
extra_overrides = json.loads(sys.argv[12])
python_fill_hole_area = int(sys.argv[13])
python_binarize_cond_mem = sys.argv[14]
save_python_mem_mask = bool(int(sys.argv[15]))
tf32_policy = sys.argv[16]

sys.path.insert(0, str(sam2_repo))
from sam2.build_sam import build_sam2_video_predictor

out_dir.mkdir(parents=True, exist_ok=True)
if python_dtype == "bf16":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
allow_tf32 = tf32_policy == "on"
torch.backends.cuda.matmul.allow_tf32 = allow_tf32
torch.backends.cudnn.allow_tf32 = allow_tf32
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

overrides = []
if image_size > 0:
    overrides.append(f"model.image_size={image_size}")
overrides.extend(extra_overrides)

predictor = build_sam2_video_predictor(
    cfg,
    checkpoint,
    device="cuda",
    vos_optimized=False,
    hydra_overrides_extra=overrides,
)
if python_fill_hole_area >= 0:
    predictor.fill_hole_area = python_fill_hole_area
if python_binarize_cond_mem != "default":
    predictor.binarize_mask_from_pts_for_mem_enc = python_binarize_cond_mem == "true"
mem_mask_counter = 0
original_run_memory_encoder = predictor._run_memory_encoder

def save_mem_mask(frame_idx, mask_for_mem, is_mask_from_pts):
    global mem_mask_counter
    if not save_python_mem_mask:
        return
    mem_dir = out_dir / "python_mem_masks"
    mem_dir.mkdir(parents=True, exist_ok=True)
    arr = mask_for_mem.detach().float().cpu().contiguous().numpy()
    stem = mem_dir / (
        f"frame_{int(frame_idx):05d}_call_{mem_mask_counter:03d}_"
        f"frompts{int(bool(is_mask_from_pts))}"
    )
    arr.tofile(stem.with_suffix(".bin"))
    stem.with_suffix(".shape").write_text(
        " ".join(str(dim) for dim in arr.shape) + "\n",
        encoding="utf-8",
    )
    mem_mask_counter += 1

def wrapped_run_memory_encoder(*m_args, **m_kwargs):
    frame_idx = m_kwargs.get("frame_idx")
    high_res_masks = m_kwargs.get("high_res_masks")
    is_mask_from_pts = m_kwargs.get("is_mask_from_pts", False)
    if high_res_masks is None and len(m_args) >= 4:
        frame_idx = m_args[1]
        high_res_masks = m_args[3]
        is_mask_from_pts = m_args[5] if len(m_args) >= 6 else False
    if save_python_mem_mask and high_res_masks is not None and frame_idx is not None:
        with torch.no_grad():
            mask_for_mem = high_res_masks.detach()
            if predictor.non_overlap_masks_for_mem_enc and not predictor.training:
                mask_for_mem = predictor._apply_non_overlapping_constraints(mask_for_mem)
            if (
                predictor.binarize_mask_from_pts_for_mem_enc
                and is_mask_from_pts
                and not predictor.training
            ):
                mask_for_mem = (mask_for_mem > 0).float()
            else:
                mask_for_mem = torch.sigmoid(mask_for_mem)
            if predictor.sigmoid_scale_for_mem_enc != 1.0:
                mask_for_mem = mask_for_mem * predictor.sigmoid_scale_for_mem_enc
            if predictor.sigmoid_bias_for_mem_enc != 0.0:
                mask_for_mem = mask_for_mem + predictor.sigmoid_bias_for_mem_enc
            save_mem_mask(frame_idx, mask_for_mem, is_mask_from_pts)
    return original_run_memory_encoder(*m_args, **m_kwargs)

predictor._run_memory_encoder = wrapped_run_memory_encoder
state = predictor.init_state(video_path=video_source)
points = np.array([[point_x, point_y]], dtype=np.float32)
labels = np.array([1], np.int32)
captures = []
original_forward = predictor._forward_sam_heads

def bbox_from_bool_mask(mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

def tensor_floats(tensor):
    return [float(v) for v in tensor.flatten().tolist()]

def tensor_mask_areas(tensor):
    return [int((mask > 0.0).sum().item()) for mask in tensor]

def wrapped_forward(*f_args, **f_kwargs):
    out = original_forward(*f_args, **f_kwargs)
    low_res_multimasks, high_res_multimasks, ious, *_ = out
    ious_cpu = ious.detach().float().cpu()
    low_res_cpu = low_res_multimasks.detach().float().cpu()
    captures.append({
        "multimask_output": bool(f_kwargs.get("multimask_output", False)),
        "has_point_inputs": f_kwargs.get("point_inputs") is not None,
        "low_res_candidate_areas": tensor_mask_areas(low_res_cpu[0]),
        "high_res_multimasks": high_res_multimasks.detach().float().cpu(),
        "ious": ious_cpu,
        "selected_mask_index": int(torch.argmax(ious_cpu[0]).item()) if ious_cpu.numel() else -1,
        "iou_scores": tensor_floats(ious_cpu[0]),
    })
    return out

predictor._forward_sam_heads = wrapped_forward

def save_row(frame_idx, obj_ids, mask_logits, elapsed_ms=None, capture=None):
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
    row = {
        "offset": int(frame_idx),
        "expected_frame_index": int(frame_idx),
        "mask_path": str(path),
        "mask_area": area,
        "elapsed_ms": elapsed_ms,
        "source": "official-sam2",
    }
    if save_python_logits:
        logits_dir = out_dir / "python_logits"
        logits_dir.mkdir(parents=True, exist_ok=True)
        logits = mask_logits[obj_index].detach().float().cpu().squeeze().contiguous().numpy()
        logits_path = logits_dir / f"frame_{frame_idx:05d}.bin"
        logits_shape_path = logits_dir / f"frame_{frame_idx:05d}.shape"
        logits.tofile(logits_path)
        logits_shape_path.write_text(" ".join(str(dim) for dim in logits.shape) + "\n", encoding="utf-8")
        row["logits_path"] = str(logits_path)
        row["logits_shape"] = [int(dim) for dim in logits.shape]
    if capture is not None:
        row.update({
            "official_multimask_output": bool(capture["multimask_output"]),
            "official_has_point_inputs": bool(capture["has_point_inputs"]),
            "official_selected_mask_index": int(capture["selected_mask_index"]),
            "official_iou_scores": capture["iou_scores"],
            "official_lowres_mask_areas": capture["low_res_candidate_areas"],
        })
    return row

rows = {}
with torch.inference_mode():
    frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )
    initial_capture = captures[-1] if captures else None
    capture_cursor = len(captures)
    row = save_row(frame_idx, obj_ids, mask_logits, capture=initial_capture)
    if row is not None:
        rows[int(frame_idx)] = row

    if initial_capture is not None:
        capture = initial_capture
        ious = capture["ious"][0]
        selected_index = int(torch.argmax(ious).item())
        frame_h, frame_w = mask_logits.shape[-2:]
        resized = F.interpolate(
            capture["high_res_multimasks"],
            size=(frame_h, frame_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        candidate_rows = []
        candidate_dir = out_dir / "official_initial_candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for candidate_index in range(resized.shape[0]):
            mask = (resized[candidate_index].numpy() > 0.0)
            path = candidate_dir / f"candidate_{candidate_index:02d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            candidate_rows.append({
                "source": "official-sam2-initial-candidate",
                "frame_index": int(frame_idx),
                "candidate_index": int(candidate_index),
                "selected_index": selected_index,
                "selected": int(candidate_index) == selected_index,
                "multimask_output": capture["multimask_output"],
                "iou_score": float(ious[candidate_index].item()),
                "mask_area": int(mask.sum()),
                "bbox_xyxy": bbox_from_bool_mask(mask),
                "mask_path": str(path),
                "frame_width": int(frame_w),
                "frame_height": int(frame_h),
                "image_size": image_size if image_size > 0 else 1024,
            })
        (out_dir / "official-initial-candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in candidate_rows),
            encoding="utf-8",
        )
    torch.cuda.synchronize()
    prev = time.perf_counter()
    track_total_ms = 0.0
    track_frames = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        capture = None
        if len(captures) > capture_cursor:
            capture = captures[capture_cursor]
            capture_cursor += 1
        row = save_row(out_frame_idx, out_obj_ids, out_mask_logits, None, capture)
        torch.cuda.synchronize()
        now = time.perf_counter()
        interval_ms = (now - prev) * 1000.0
        if 0 < out_frame_idx < frames:
            track_total_ms += interval_ms
            track_frames += 1
            if row is not None:
                row["elapsed_ms"] = track_total_ms
                row["track_interval_ms"] = interval_ms
        if row is not None and out_frame_idx != 0:
            rows[int(out_frame_idx)] = row
        prev = now
        if out_frame_idx + 1 >= frames:
            break

ordered = [rows[i] for i in sorted(rows) if i < frames]
for row in ordered:
    print(json.dumps(row), flush=True)
print(json.dumps({
    "summary": {
        "frames": len(ordered),
        "track_frames": track_frames,
        "propagate_total_ms": track_total_ms,
        "propagate_ms_per_frame": track_total_ms / max(track_frames, 1),
        "timed_frame_range": "1..frames-1",
        "cuda_alloc_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "image_size": image_size if image_size > 0 else 1024,
        "python_dtype": python_dtype,
        "tf32_policy": tf32_policy,
        "torch_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
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
            str(args.video if args.python_video_source == "video" else frame_dir),
            str(out_dir / "python_masks"),
            str(args.point_x),
            str(args.point_y),
            str(args.frames),
            str(args.image_size),
            args.python_dtype,
            "1" if args.save_python_logits else "0",
            json.dumps(args.hydra_override),
            str(args.python_fill_hole_area),
            args.python_binarize_cond_mem,
            "1" if args.save_python_mem_mask else "0",
            args.tf32_policy,
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
    parser.add_argument("--cpp-initial-candidates-jsonl", type=Path)
    parser.add_argument("--sam2-repo", type=Path, default=Path("../sam2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument(
        "--python-video-source",
        choices=("video", "frames"),
        default="video",
        help="Feed official SAM2 either the original video file or extracted JPEG frames.",
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Official SAM2 image_size override. 0 keeps the config default.",
    )
    parser.add_argument(
        "--python-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Autocast dtype for official PyTorch SAM2. fp32 disables CUDA autocast.",
    )
    parser.add_argument(
        "--tf32-policy",
        choices=("on", "off"),
        default="on",
        help="Enable or disable PyTorch TF32 matmul/cuDNN. Use off for Strict FP32.",
    )
    parser.add_argument(
        "--save-python-logits",
        action="store_true",
        help="Save official PyTorch selected mask logits next to python masks for parity diagnostics.",
    )
    parser.add_argument(
        "--save-python-mem-mask",
        action="store_true",
        help="Save official PyTorch memory-encoder input masks for parity diagnostics.",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="Extra Hydra override passed to official SAM2 builder. May be specified more than once.",
    )
    parser.add_argument(
        "--python-fill-hole-area",
        type=int,
        default=-1,
        help="Diagnostic override for predictor.fill_hole_area after official SAM2 construction. -1 keeps builder default.",
    )
    parser.add_argument(
        "--python-binarize-cond-mem",
        choices=("default", "true", "false"),
        default="default",
        help="Diagnostic override for official predictor.binarize_mask_from_pts_for_mem_enc after construction.",
    )
    parser.add_argument("--point-x", type=float, default=315.0)
    parser.add_argument("--point-y", type=float, default=250.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sam2-official-quality"))
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.video = args.video.resolve()
    args.cpp_jsonl = args.cpp_jsonl.resolve()
    if args.cpp_initial_candidates_jsonl is not None:
        args.cpp_initial_candidates_jsonl = args.cpp_initial_candidates_jsonl.resolve()
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
    source_width, source_height = frame_size(frame_dir)
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
        cpp_selected = cpp_candidate_space_index(cpp)
        py_selected = py.get("official_selected_mask_index")
        comparisons.append(
            {
                "offset": offset,
                "mask_iou": mask_iou(cpp_mask, py_mask),
                "bbox_iou": bbox_iou(cpp_bbox, py_bbox),
                "cpp_mask_area": cpp.get("mask_area"),
                "python_mask_area": py.get("mask_area"),
                "cpp_bbox_xyxy": cpp_bbox,
                "python_bbox_xyxy": py_bbox,
                "cpp_selected_mask_index": cpp.get("selected_mask_index"),
                "cpp_selected_candidate_index": cpp_selected,
                "cpp_decoder_iou_scores": cpp.get("decoder_iou_scores"),
                "cpp_lowres_mask_areas": cpp.get("decoder_lowres_mask_areas"),
                "python_selected_candidate_index": py_selected,
                "python_decoder_iou_scores": py.get("official_iou_scores"),
                "python_lowres_mask_areas": py.get("official_lowres_mask_areas"),
                "selected_index_match": (
                    cpp_selected == py_selected if cpp_selected is not None and py_selected is not None else None
                ),
            }
        )

    selection_rows = [
        row
        for row in comparisons
        if row["offset"] > 0 and row.get("selected_index_match") is not None
    ]
    summary = {
        "frames_compared": len(comparisons),
        "min_mask_iou": min((row["mask_iou"] for row in comparisons), default=None),
        "mean_mask_iou": sum(row["mask_iou"] for row in comparisons) / max(len(comparisons), 1),
        "min_bbox_iou": min((row["bbox_iou"] for row in comparisons if row["bbox_iou"] is not None), default=None),
        "propagation_candidate_selection": {
            "frames_compared": len(selection_rows),
            "matching_frames": sum(1 for row in selection_rows if row["selected_index_match"]),
            "mismatches": [
                {
                    "offset": row["offset"],
                    "cpp_selected_candidate_index": row["cpp_selected_candidate_index"],
                    "python_selected_candidate_index": row["python_selected_candidate_index"],
                    "cpp_decoder_iou_scores": row["cpp_decoder_iou_scores"],
                    "python_decoder_iou_scores": row["python_decoder_iou_scores"],
                    "cpp_lowres_mask_areas": row["cpp_lowres_mask_areas"],
                    "python_lowres_mask_areas": row["python_lowres_mask_areas"],
                    "mask_iou": row["mask_iou"],
                }
                for row in selection_rows
                if not row["selected_index_match"]
            ],
        },
        "comparisons": comparisons,
        "python_summary": python_summary,
        "metadata_validation": metadata_validation,
        "initial_candidate_comparison": summarize_initial_candidates(
            args.cpp_initial_candidates_jsonl,
            args.out_dir / "python_masks" / "official-initial-candidates.jsonl",
        ),
        "comparison_contract": {
            "same_decoded_source_resolution": True,
            "decoded_source_width": source_width,
            "decoded_source_height": source_height,
            "same_frame_count": True,
            "same_point_prompt": True,
            "same_encode_img_size": True,
            "python_video_source": args.python_video_source,
            "note": (
                "C++/official PyTorch speed or quality rows are directly comparable "
                "only when the decoded source-frame resolution and SAM model input "
                "resolution match. Other resolutions are scaling studies."
            ),
        },
        "official_python_overrides": {
            "hydra_override": args.hydra_override,
            "python_fill_hole_area": args.python_fill_hole_area,
            "python_binarize_cond_mem": args.python_binarize_cond_mem,
            "python_dtype": args.python_dtype,
            "tf32_policy": args.tf32_policy,
            "save_python_logits": args.save_python_logits,
            "save_python_mem_mask": args.save_python_mem_mask,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
