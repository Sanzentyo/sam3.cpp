# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "hydra-core",
#   "iopath",
#   "numpy",
#   "pillow",
#   "torch",
#   "torchvision",
#   "tqdm",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


def run(cmd: list[str], cwd: Path, log_path: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
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
    if len(list(frame_dir.glob("*.jpg"))) >= frames:
        return
    for old in frame_dir.glob("*.jpg"):
        old.unlink()
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


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def resize_logits_to_frame(logits: torch.Tensor, width: int, height: int) -> torch.Tensor:
    logits = logits.detach().float().cpu()
    if logits.ndim == 2:
        logits = logits[None, None, :, :]
    elif logits.ndim == 3:
        logits = logits[:, None, :, :]
    elif logits.ndim != 4:
        raise ValueError(f"unexpected mask logits shape: {tuple(logits.shape)}")
    return torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def save_candidate_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def first_frame_size(frame_dir: Path) -> tuple[int, int]:
    with Image.open(sorted(frame_dir.glob("*.jpg"))[0]) as img:
        return img.size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--sam2-repo", type=Path, default=Path("../sam2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--point-x", type=float, default=315.0)
    parser.add_argument("--point-y", type=float, default=250.0)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sam2-initial-mask-diagnostics"))
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.sam2_repo = args.sam2_repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.video = args.video.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = args.out_dir / "frames"
    extract_frames(args.video, frame_dir, args.frames)
    frame_width, frame_height = first_frame_size(frame_dir)

    sys.path.insert(0, str(args.sam2_repo))
    os.environ["PYTHONPATH"] = str(args.sam2_repo)
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    overrides = []
    if args.image_size > 0:
        overrides.append(f"model.image_size={args.image_size}")

    predictor = build_sam2_video_predictor(
        args.config,
        str(args.checkpoint),
        device="cuda",
        vos_optimized=False,
        hydra_overrides_extra=overrides,
    )

    captures: list[dict[str, Any]] = []
    original_forward = predictor._forward_sam_heads

    def wrapped_forward(*f_args: Any, **f_kwargs: Any) -> Any:
        started = time.perf_counter()
        out = original_forward(*f_args, **f_kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = out
        captures.append(
            {
                "elapsed_ms": elapsed_ms,
                "multimask_output": bool(f_kwargs.get("multimask_output", False)),
                "low_res_multimasks": low_res_multimasks.detach().float().cpu(),
                "high_res_multimasks": high_res_multimasks.detach().float().cpu(),
                "ious": ious.detach().float().cpu(),
                "low_res_masks": low_res_masks.detach().float().cpu(),
                "high_res_masks": high_res_masks.detach().float().cpu(),
                "object_score_logits": object_score_logits.detach().float().cpu()
                if object_score_logits is not None
                else None,
                "obj_ptr_norm": float(obj_ptr.detach().float().norm().cpu()),
            }
        )
        return out

    predictor._forward_sam_heads = wrapped_forward  # type: ignore[method-assign]

    state = predictor.init_state(video_path=str(frame_dir))
    points = np.array([[args.point_x, args.point_y]], dtype=np.float32)
    labels = np.array([1], np.int32)

    with torch.inference_mode():
        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )
        torch.cuda.synchronize()

    if not captures:
        raise RuntimeError("no _forward_sam_heads capture was recorded")

    capture = captures[-1]
    ious = capture["ious"][0].numpy()
    selected_index = int(np.argmax(ious))
    rows: list[dict[str, Any]] = []

    high_res = capture["high_res_multimasks"][0]
    frame_logits = resize_logits_to_frame(high_res, frame_width, frame_height)
    for candidate_index in range(frame_logits.shape[0]):
        mask = (frame_logits[candidate_index].numpy() > 0.0)
        mask_path = args.out_dir / "candidate_masks" / f"candidate_{candidate_index:02d}.png"
        save_candidate_mask(mask, mask_path)
        object_score = capture["object_score_logits"]
        rows.append(
            {
                "source": "official-sam2-initial-candidate",
                "frame_index": int(frame_idx),
                "obj_ids": [int(x) for x in obj_ids],
                "candidate_index": candidate_index,
                "selected_index": selected_index,
                "selected": candidate_index == selected_index,
                "multimask_output": capture["multimask_output"],
                "iou_score": float(ious[candidate_index]),
                "object_score_logit": float(object_score.reshape(-1)[0]) if object_score is not None else None,
                "obj_ptr_norm": capture["obj_ptr_norm"],
                "mask_area": int(mask.sum()),
                "bbox_xyxy": bbox_from_mask(mask),
                "mask_path": str(mask_path.relative_to(args.repo)) if mask_path.is_relative_to(args.repo) else str(mask_path),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "image_size": args.image_size,
                "point_x": args.point_x,
                "point_y": args.point_y,
                "elapsed_ms": capture["elapsed_ms"],
            }
        )

    selected_mask = (mask_logits[0] > 0.0).detach().cpu().numpy().squeeze()
    selected_path = args.out_dir / "selected_from_predictor.png"
    save_candidate_mask(selected_mask, selected_path)
    rows.append(
        {
            "source": "official-sam2-initial-selected-output",
            "frame_index": int(frame_idx),
            "selected_index": selected_index,
            "mask_area": int(selected_mask.sum()),
            "bbox_xyxy": bbox_from_mask(selected_mask),
            "mask_path": str(selected_path.relative_to(args.repo))
            if selected_path.is_relative_to(args.repo)
            else str(selected_path),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "image_size": args.image_size,
        }
    )

    jsonl = args.out_dir / "initial_candidates.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "rows": len(rows),
        "candidate_rows": len(rows) - 1,
        "selected_index": selected_index,
        "selected_iou": float(ious[selected_index]),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "image_size": args.image_size,
        "jsonl": str(jsonl.relative_to(args.repo)) if jsonl.is_relative_to(args.repo) else str(jsonl),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
