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
"""Dump an official Python SAM3.1 multiplex mask-decoder parity case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def write_raw_tensor(base: Path, name: str, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values.astype(np.float32, copy=False))
    (base / f"{name}.shape").write_text(
        ",".join(str(dim) for dim in values.shape) + "\n",
        encoding="utf-8",
    )
    values.tofile(base / f"{name}.bin")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feat-size", type=int, default=4)
    parser.add_argument("--buckets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    if args.feat_size <= 0:
        raise SystemExit("--feat-size must be positive")
    if args.buckets <= 0:
        raise SystemExit("--buckets must be positive")

    sys.path.insert(0, str(args.repo.resolve()))
    from sam3.model.multiplex_mask_decoder import MultiplexMaskDecoder
    from sam3.sam.transformer import TwoWayTransformer

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    dec = MultiplexMaskDecoder(
        multiplex_count=16,
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=True,
        iou_prediction_use_sigmoid=False,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_outputs_only=True,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
    ).eval()
    dec_state = {
        key.removeprefix("tracker.model.sam_mask_decoder."): value
        for key, value in ckpt.items()
        if key.startswith("tracker.model.sam_mask_decoder.")
    }
    missing, unexpected = dec.load_state_dict(dec_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"decoder state mismatch: missing={missing}, unexpected={unexpected}")
    output_valid_embed = ckpt["tracker.model.output_valid_embed"].to(dtype=torch.float32)

    b = args.buckets
    c = dec.transformer_dim
    h = args.feat_size
    m = dec.multiplex_count
    k = dec.num_mask_output_per_object

    image_embeddings = torch.randn(b, c, h, h, dtype=torch.float32)
    image_pe = torch.randn(1, c, h, h, dtype=torch.float32)
    projected_feat_s0 = torch.randn(b, c // 8, h * 4, h * 4, dtype=torch.float32)
    projected_feat_s1 = torch.randn(b, c // 4, h * 2, h * 2, dtype=torch.float32)
    extra_per_object_embeddings = output_valid_embed.view(1, m, c).expand(b, m, c).contiguous()

    out = dec.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        high_res_features=[projected_feat_s0, projected_feat_s1],
        extra_per_object_embeddings=extra_per_object_embeddings,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.out / "cpp_inputs"
    expected_dir = args.out / "expected_cpp_layout"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / "official_mux_mask_decoder_case.npz"
    np.savez(
        npz_path,
        image_embeddings=image_embeddings.numpy(),
        image_pe=image_pe.numpy(),
        projected_feat_s0=projected_feat_s0.numpy(),
        projected_feat_s1=projected_feat_s1.numpy(),
        extra_per_object_embeddings=extra_per_object_embeddings.numpy(),
        masks=out["masks"].numpy(),
        iou_pred=out["iou_pred"].numpy(),
        mask_tokens_out=out["mask_tokens_out"].numpy(),
        object_score_logits=out["object_score_logits"].numpy(),
    )

    image_embeddings_np = image_embeddings.numpy()
    image_pe_np = image_pe.numpy()
    projected_feat_s0_np = projected_feat_s0.numpy()
    projected_feat_s1_np = projected_feat_s1.numpy()
    extra_np = extra_per_object_embeddings.numpy()
    masks_np = out["masks"].numpy()
    iou_np = out["iou_pred"].numpy()
    mask_tokens_np = out["mask_tokens_out"].numpy()
    obj_np = out["object_score_logits"].numpy()

    write_raw_tensor(inputs_dir, "image_feats", image_embeddings_np.transpose(1, 3, 2, 0))
    write_raw_tensor(inputs_dir, "image_pe", image_pe_np.transpose(1, 3, 2, 0))
    write_raw_tensor(inputs_dir, "projected_feat_s0", projected_feat_s0_np.transpose(1, 3, 2, 0))
    write_raw_tensor(inputs_dir, "projected_feat_s1", projected_feat_s1_np.transpose(1, 3, 2, 0))
    write_raw_tensor(inputs_dir, "extra_per_object_embeddings", extra_np.transpose(2, 1, 0))

    write_raw_tensor(expected_dir, "masks", masks_np.reshape(b, m * k, h * 4 * h * 4).transpose(2, 1, 0))
    write_raw_tensor(expected_dir, "iou_pred", iou_np.transpose(2, 1, 0))
    write_raw_tensor(expected_dir, "mask_tokens_out", mask_tokens_np.reshape(b, m * k, c).transpose(2, 1, 0))
    write_raw_tensor(expected_dir, "object_score_logits", obj_np.transpose(2, 1, 0))

    summary = {
        "checkpoint": str(args.checkpoint),
        "npz": str(npz_path),
        "cpp_inputs": str(inputs_dir),
        "expected_cpp_layout": str(expected_dir),
        "seed": args.seed,
        "buckets": b,
        "feat_size": h,
        "multiplex_count": m,
        "masks_per_object": k,
        "shapes": {
            name: list(value.shape)
            for name, value in {
                "image_embeddings": image_embeddings,
                "image_pe": image_pe,
                "projected_feat_s0": projected_feat_s0,
                "projected_feat_s1": projected_feat_s1,
                "extra_per_object_embeddings": extra_per_object_embeddings,
                **out,
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
