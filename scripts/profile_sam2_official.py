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
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch


def cuda_ms(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000.0


def timed_wrapper(stats: dict[str, list[float]], name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result, elapsed_ms = cuda_ms(fn, *args, **kwargs)
        stats.setdefault(name, []).append(elapsed_ms)
        return result

    return wrapper


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "sum_ms": sum(values),
        "mean_ms": sum(values) / len(values),
        "min_ms": ordered[0],
        "p50_ms": ordered[len(ordered) // 2],
        "max_ms": ordered[-1],
        "values_ms": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-repo", type=Path, default=Path("../sam2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--point-x", type=float, default=315.0)
    parser.add_argument("--point-y", type=float, default=250.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

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

    stats: dict[str, list[float]] = {}
    predictor.forward_image = timed_wrapper(stats, "forward_image", predictor.forward_image)
    predictor.track_step = timed_wrapper(stats, "track_step", predictor.track_step)
    predictor._run_memory_encoder = timed_wrapper(
        stats, "_run_memory_encoder", predictor._run_memory_encoder
    )

    def run_once(measure: bool = False) -> dict[str, Any]:
        stats.clear()
        state = predictor.init_state(video_path=str(args.video_dir))
        points = np.array([[args.point_x, args.point_y]], dtype=np.float32)
        labels = np.array([1], np.int32)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )
        count = 0
        measured_ms = 0.0
        per_frame: list[dict[str, Any]] = []
        prev = time.perf_counter()
        for out_frame_idx, _, _ in predictor.propagate_in_video(state):
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed_ms = (now - prev) * 1000.0
            if 0 < out_frame_idx < args.frames:
                measured_ms += elapsed_ms
                count += 1
                if measure:
                    per_frame.append({"frame": int(out_frame_idx), "elapsed_ms": elapsed_ms})
            prev = now
            if out_frame_idx + 1 >= args.frames:
                break
        torch.cuda.synchronize()
        return {
            "track_frames": count,
            "track_total_ms": measured_ms,
            "track_ms": measured_ms / max(count, 1),
            "per_frame": per_frame,
            "stats": {name: summarize_values(values) for name, values in stats.items()},
        }

    for _ in range(2):
        run_once()

    torch.cuda.reset_peak_memory_stats()
    result = run_once(measure=True)
    result.update(
        {
            "backend": "PyTorch CUDA bf16",
            "frames": args.frames,
            "image_size": args.image_size if args.image_size > 0 else 1024,
            "cuda_alloc_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
