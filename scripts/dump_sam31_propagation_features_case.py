#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.8.0",
#   "numpy>=1.24",
# ]
# ///
"""Dump an official-weight SAM3.1 propagation feature-adapter parity case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def write_raw_tensor(base: Path, name: str, values: np.ndarray) -> None:
    values = values.astype(np.float32, copy=False)
    (base / f"{name}.shape").write_text(
        ",".join(str(dim) for dim in values.shape) + "\n",
        encoding="utf-8",
    )
    values.ravel(order="F").tofile(base / f"{name}.bin")


def nchw_to_ggml_cwhb(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().transpose(1, 3, 2, 0)


def conv2d(state: dict[str, torch.Tensor], prefix: str, x: torch.Tensor, padding: int = 0) -> torch.Tensor:
    return F.conv2d(x, state[f"{prefix}.weight"].float(), state[f"{prefix}.bias"].float(), padding=padding)


def conv_transpose2d(state: dict[str, torch.Tensor], prefix: str, x: torch.Tensor) -> torch.Tensor:
    return F.conv_transpose2d(
        x,
        state[f"{prefix}.weight"].float(),
        state[f"{prefix}.bias"].float(),
        stride=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feat-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3105)
    args = parser.parse_args()

    if args.feat_size <= 0:
        raise SystemExit("--feat-size must be positive")

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    vit_out = torch.randn(1, 1024, args.feat_size, args.feat_size, dtype=torch.float32)
    p = "detector.backbone.vision_backbone.propagation_convs"

    raw_s0 = conv_transpose2d(ckpt, f"{p}.0.dconv_2x2_0", vit_out)
    raw_s0 = F.gelu(raw_s0, approximate="none")
    raw_s0 = conv_transpose2d(ckpt, f"{p}.0.dconv_2x2_1", raw_s0)
    raw_s0 = conv2d(ckpt, f"{p}.0.conv_1x1", raw_s0)
    raw_s0 = conv2d(ckpt, f"{p}.0.conv_3x3", raw_s0, padding=1)

    raw_s1 = conv_transpose2d(ckpt, f"{p}.1.dconv_2x2", vit_out)
    raw_s1 = conv2d(ckpt, f"{p}.1.conv_1x1", raw_s1)
    raw_s1 = conv2d(ckpt, f"{p}.1.conv_3x3", raw_s1, padding=1)

    image_features = conv2d(ckpt, f"{p}.2.conv_1x1", vit_out)
    image_features = conv2d(ckpt, f"{p}.2.conv_3x3", image_features, padding=1)

    d = "tracker.model.sam_mask_decoder"
    projected_s0 = conv2d(ckpt, f"{d}.conv_s0", raw_s0)
    projected_s1 = conv2d(ckpt, f"{d}.conv_s1", raw_s1)

    args.out.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.out / "cpp_inputs"
    expected_dir = args.out / "expected_cpp_layout"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    write_raw_tensor(inputs_dir, "vit_out", nchw_to_ggml_cwhb(vit_out))
    for name, tensor in {
        "projected_s0": projected_s0,
        "projected_s1": projected_s1,
        "image_features": image_features,
    }.items():
        write_raw_tensor(expected_dir, name, nchw_to_ggml_cwhb(tensor))

    npz_path = args.out / "official_propagation_features_case.npz"
    np.savez(
        npz_path,
        vit_out=vit_out.detach().cpu().numpy(),
        projected_s0=projected_s0.detach().cpu().numpy(),
        projected_s1=projected_s1.detach().cpu().numpy(),
        image_features=image_features.detach().cpu().numpy(),
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "npz": str(npz_path),
        "cpp_inputs": str(inputs_dir),
        "expected_cpp_layout": str(expected_dir),
        "seed": args.seed,
        "feat_size": args.feat_size,
        "shapes": {
            "vit_out": list(vit_out.shape),
            "projected_s0": list(projected_s0.shape),
            "projected_s1": list(projected_s1.shape),
            "image_features": list(image_features.shape),
        },
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {npz_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
