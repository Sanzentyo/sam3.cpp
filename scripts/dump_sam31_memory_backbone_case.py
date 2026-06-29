#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.8.0",
#   "numpy>=1.24",
#   "timm>=1.0",
#   "iopath>=0.1.10",
#   "einops>=0.7",
#   "pycocotools>=2.0.11",
#   "psutil>=7.0",
#   "ftfy>=6.3",
#   "regex>=2026.5.9",
#   "setuptools<81",
# ]
# ///
"""Dump an official Python SAM3.1 multiplex memory-backbone parity case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def write_raw_tensor(base: Path, name: str, values: np.ndarray) -> None:
    values = values.astype(np.float32, copy=False)
    (base / f"{name}.shape").write_text(
        ",".join(str(dim) for dim in values.shape) + "\n",
        encoding="utf-8",
    )
    values.ravel(order="F").tofile(base / f"{name}.bin")


def nchw_to_ggml_dwhb(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().transpose(1, 3, 2, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feat-size", type=int, default=72)
    parser.add_argument("--mask-size", type=int, default=1152)
    parser.add_argument("--buckets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()

    if args.feat_size <= 0:
        raise SystemExit("--feat-size must be positive")
    if args.mask_size <= 0:
        raise SystemExit("--mask-size must be positive")
    if args.buckets != 1:
        raise SystemExit("C++ memory backbone slice currently supports --buckets 1")

    sys.path.insert(0, str(args.repo.resolve()))
    from sam3.model_builder import _create_multiplex_maskmem_backbone

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    model = _create_multiplex_maskmem_backbone(multiplex_count=16).eval()
    state = {
        key.removeprefix("tracker.model.maskmem_backbone."): value
        for key, value in ckpt.items()
        if key.startswith("tracker.model.maskmem_backbone.")
    }
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"memory backbone state mismatch: missing={missing}, unexpected={unexpected}")

    b = args.buckets
    c = 256
    h = args.feat_size
    mask_ch = 32
    mask_size = args.mask_size
    pix_feat = torch.randn(b, c, h, h, dtype=torch.float32)
    masks = torch.randn(b, mask_ch, mask_size, mask_size, dtype=torch.float32)

    mask_downsampled = model.mask_downsampler(masks)
    pix_feat_proj = model.pix_feat_proj(pix_feat)
    fused_input = pix_feat_proj + mask_downsampled
    fuser0 = model.fuser.layers[0](fused_input)
    vision_features = model.fuser.layers[1](fuser0)
    vision_pos_enc = model.position_encoding(vision_features).to(vision_features.dtype)
    t_pos = 1
    tpos_enc = ckpt["tracker.model.maskmem_tpos_enc"][7 - t_pos - 1].to(
        device=vision_pos_enc.device,
        dtype=torch.float32,
    )
    memory_prompt = vision_features.flatten(2).permute(1, 2, 0)
    memory_prompt_pos = vision_pos_enc.flatten(2).permute(1, 2, 0) + tpos_enc.view(c, 1, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.out / "cpp_inputs"
    expected_dir = args.out / "expected_cpp_layout"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / "official_memory_backbone_case.npz"
    np.savez(
        npz_path,
        pix_feat=pix_feat.detach().cpu().numpy(),
        masks=masks.detach().cpu().numpy(),
        mask_downsampled=mask_downsampled.detach().cpu().numpy(),
        pix_feat_proj=pix_feat_proj.detach().cpu().numpy(),
        fused_input=fused_input.detach().cpu().numpy(),
        fuser0=fuser0.detach().cpu().numpy(),
        vision_features=vision_features.detach().cpu().numpy(),
        vision_pos_enc=vision_pos_enc.detach().cpu().numpy(),
        memory_prompt=memory_prompt.detach().cpu().numpy(),
        memory_prompt_pos=memory_prompt_pos.detach().cpu().numpy(),
    )

    write_raw_tensor(inputs_dir, "pix_feat", nchw_to_ggml_dwhb(pix_feat))
    write_raw_tensor(inputs_dir, "masks", masks.detach().cpu().numpy().transpose(3, 2, 1, 0))

    for name, tensor in {
        "mask_downsampled": mask_downsampled,
        "pix_feat_proj": pix_feat_proj,
        "fused_input": fused_input,
        "fuser0": fuser0,
        "vision_features": vision_features,
        "vision_pos_enc": vision_pos_enc,
        "stored_spatial_feats": vision_features,
        "stored_spatial_pe": vision_pos_enc,
    }.items():
        write_raw_tensor(expected_dir, name, nchw_to_ggml_dwhb(tensor))
    write_raw_tensor(expected_dir, "memory_prompt", memory_prompt.detach().cpu().numpy())
    write_raw_tensor(expected_dir, "memory_prompt_pos", memory_prompt_pos.detach().cpu().numpy())

    summary = {
        "checkpoint": str(args.checkpoint),
        "npz": str(npz_path),
        "cpp_inputs": str(inputs_dir),
        "expected_cpp_layout": str(expected_dir),
        "seed": args.seed,
        "buckets": b,
        "feat_size": h,
        "mask_size": mask_size,
        "mask_channels": mask_ch,
        "skip_mask_sigmoid": True,
        "shapes": {
            name: list(tensor.shape)
            for name, tensor in {
                "pix_feat": pix_feat,
                "masks": masks,
                "mask_downsampled": mask_downsampled,
                "pix_feat_proj": pix_feat_proj,
                "fused_input": fused_input,
                "fuser0": fuser0,
                "vision_features": vision_features,
                "vision_pos_enc": vision_pos_enc,
                "memory_prompt": memory_prompt,
                "memory_prompt_pos": memory_prompt_pos,
            }.items()
        },
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {npz_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
