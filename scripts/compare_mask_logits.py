# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pillow",
#   "torch==2.8.0",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def read_logits(base: Path, frame: int) -> np.ndarray:
    shape_path = base / f"frame_{frame:05d}.shape"
    data_path = base / f"frame_{frame:05d}.bin"
    shape = tuple(int(value) for value in shape_path.read_text(encoding="utf-8").split())
    data = np.fromfile(data_path, dtype=np.float32)
    return data.reshape(shape)


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "mean_abs_delta": float(abs_values.mean()),
        "p95_abs_delta": float(np.percentile(abs_values, 95)),
        "min_delta": float(values.min()),
        "max_delta": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-logits-dir", type=Path, required=True)
    parser.add_argument("--python-logits-dir", type=Path, required=True)
    parser.add_argument("--cpp-mask-dir", type=Path, required=True)
    parser.add_argument("--python-mask-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boundary-logit-band", type=float, default=0.5)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for comparison in summary["comparisons"]:
        frame = int(comparison["offset"])
        cpp = read_logits(args.cpp_logits_dir, frame)
        python = read_logits(args.python_logits_dir, frame)
        cpp_up = (
            F.interpolate(
                torch.from_numpy(cpp).float()[None, None],
                size=python.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .numpy()
        )
        delta = cpp_up - python

        cpp_mask = read_mask(args.cpp_mask_dir / f"frame_{frame:05d}.png")
        python_mask = read_mask(args.python_mask_dir / f"frame_{frame:05d}.png")
        xor = cpp_mask ^ python_mask
        cpp_only = cpp_mask & ~python_mask
        python_only = python_mask & ~cpp_mask
        near_boundary = (np.abs(cpp_up) < args.boundary_logit_band) | (
            np.abs(python) < args.boundary_logit_band
        ) | xor

        rows.append(
            {
                "offset": frame,
                "mask_iou": comparison["mask_iou"],
                "cpp_area": int(cpp_mask.sum()),
                "python_area": int(python_mask.sum()),
                "xor_pixels": int(xor.sum()),
                "cpp_only_pixels": int(cpp_only.sum()),
                "python_only_pixels": int(python_only.sum()),
                "all": stats(delta.reshape(-1)),
                "near_boundary": stats(delta[near_boundary]),
                "xor": stats(delta[xor]),
                "cpp_only": stats(delta[cpp_only]),
                "python_only": stats(delta[python_only]),
                "cpp_up_positive_python_nonpositive": int(((cpp_up > 0) & (python <= 0)).sum()),
                "cpp_nonpositive_python_positive": int(((cpp_up <= 0) & (python > 0)).sum()),
            }
        )

    output = {
        "source": "cpp selected logits resized with torch bilinear align_corners=False vs official PyTorch selected logits",
        "cpp_logits_dir": str(args.cpp_logits_dir),
        "python_logits_dir": str(args.python_logits_dir),
        "boundary_logit_band": args.boundary_logit_band,
        "frames": rows,
        "worst_by_iou": sorted(rows, key=lambda row: row["mask_iou"])[:3],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["worst_by_iou"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
