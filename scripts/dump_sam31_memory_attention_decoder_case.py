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
"""Dump an official SAM3.1 memory-attention -> multiplex decoder parity case."""

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


def seq_b_c_to_ggml(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().transpose(2, 0, 1)


def nchw_to_ggml_dwhb(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().transpose(1, 3, 2, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feat-size", type=int, default=4)
    parser.add_argument("--buckets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["auto", "f32", "f16", "bf16"], default="f32")
    args = parser.parse_args()

    if args.feat_size <= 0:
        raise SystemExit("--feat-size must be positive")
    if args.buckets != 1:
        raise SystemExit("C++ SAM3.1 combined slice currently supports --buckets 1")

    sys.path.insert(0, str(args.repo.resolve()))
    from sam3.model.multiplex_mask_decoder import MultiplexMaskDecoder
    from sam3.model_builder import _create_multiplex_transformer
    from sam3.sam.transformer import TwoWayTransformer

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    dtype_name = "f16" if args.dtype == "auto" and device.type == "cuda" else args.dtype
    if dtype_name == "auto":
        dtype_name = "f32"
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    encoder = _create_multiplex_transformer(use_fa3=False, use_rope_real=True).encoder.eval().to(
        device=device,
        dtype=dtype,
    )
    enc_state = {
        key.removeprefix("tracker.model.transformer.encoder."): value
        for key, value in ckpt.items()
        if key.startswith("tracker.model.transformer.encoder.")
    }
    missing, unexpected = encoder.load_state_dict(enc_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"memory-attention state mismatch: missing={missing}, unexpected={unexpected}")

    decoder = MultiplexMaskDecoder(
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
    ).eval().to(device=device, dtype=dtype)
    dec_state = {
        key.removeprefix("tracker.model.sam_mask_decoder."): value
        for key, value in ckpt.items()
        if key.startswith("tracker.model.sam_mask_decoder.")
    }
    missing, unexpected = decoder.load_state_dict(dec_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"decoder state mismatch: missing={missing}, unexpected={unexpected}")

    b = args.buckets
    c = 256
    h = args.feat_size
    n = h * h
    m = decoder.multiplex_count
    k = decoder.num_mask_output_per_object

    image = torch.randn(n, 1, c, dtype=dtype, device=device)
    src = torch.randn(n, b, c, dtype=dtype, device=device)
    memory_image = torch.randn(n, 1, c, dtype=dtype, device=device)
    memory = torch.randn(n, b, c, dtype=dtype, device=device)
    src_pos = torch.randn(n, b, c, dtype=dtype, device=device)
    memory_image_pos = torch.randn(n, 1, c, dtype=dtype, device=device)
    decoder_image_pe = torch.randn(1, c, h, h, dtype=dtype, device=device)
    projected_feat_s0 = torch.randn(b, c // 8, h * 4, h * 4, dtype=dtype, device=device)
    projected_feat_s1 = torch.randn(b, c // 4, h * 2, h * 2, dtype=dtype, device=device)
    output_valid_embed = ckpt["tracker.model.output_valid_embed"].to(device=device, dtype=dtype)
    extra_per_object_embeddings = output_valid_embed.view(1, m, c).expand(b, m, c).contiguous()

    output = src + 0.1 * src_pos
    image_b = image.transpose(0, 1)
    output_b = output.transpose(0, 1)
    memory_image_b = memory_image.transpose(0, 1)
    memory_b = memory.transpose(0, 1)
    src_pos_b = src_pos.transpose(0, 1)
    memory_image_pos_b = memory_image_pos.transpose(0, 1)

    for layer in encoder.layers:
        output_b = layer._forward_sa(output_b, src_pos_b)
        output_b = layer._forward_ca(
            image=image_b,
            tgt=output_b,
            memory_image=memory_image_b,
            memory=memory_b,
            query_pos=src_pos_b,
            memory_image_pos=memory_image_pos_b,
            num_k_exclude_rope=0,
        )
        ffn = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(output_b)))))
        output_b = output_b + layer.dropout3(ffn)

    memory_out = encoder.norm(output_b).transpose(0, 1)
    image_embeddings = memory_out.permute(1, 2, 0).view(b, c, h, h)
    decoder_out = decoder.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=decoder_image_pe,
        high_res_features=[projected_feat_s0, projected_feat_s1],
        extra_per_object_embeddings=extra_per_object_embeddings,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.out / "cpp_inputs"
    expected_dir = args.out / "expected_cpp_layout"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    for name, tensor in {
        "image": image,
        "src": src,
        "memory_image": memory_image,
        "memory": memory,
        "src_pos": src_pos,
        "memory_image_pos": memory_image_pos,
    }.items():
        write_raw_tensor(inputs_dir, name, seq_b_c_to_ggml(tensor))
    write_raw_tensor(inputs_dir, "image_pe", nchw_to_ggml_dwhb(decoder_image_pe))
    write_raw_tensor(inputs_dir, "projected_feat_s0", nchw_to_ggml_dwhb(projected_feat_s0))
    write_raw_tensor(inputs_dir, "projected_feat_s1", nchw_to_ggml_dwhb(projected_feat_s1))
    write_raw_tensor(
        inputs_dir,
        "extra_per_object_embeddings",
        extra_per_object_embeddings.detach().cpu().numpy().transpose(2, 1, 0),
    )

    masks = decoder_out["masks"]
    iou = decoder_out["iou_pred"]
    mask_tokens = decoder_out["mask_tokens_out"]
    obj = decoder_out["object_score_logits"]
    write_raw_tensor(expected_dir, "sam31_mem_attn_output", seq_b_c_to_ggml(memory_out))
    write_raw_tensor(expected_dir, "masks", masks.detach().cpu().numpy().reshape(b, m * k, h * 4 * h * 4).transpose(2, 1, 0))
    write_raw_tensor(expected_dir, "iou_pred", iou.detach().cpu().numpy().transpose(2, 1, 0))
    write_raw_tensor(
        expected_dir,
        "mask_tokens_out",
        mask_tokens.detach().cpu().numpy().reshape(b, m * k, c).transpose(2, 1, 0),
    )
    write_raw_tensor(expected_dir, "object_score_logits", obj.detach().cpu().numpy().transpose(2, 1, 0))

    npz_path = args.out / "official_memory_attention_decoder_case.npz"
    np.savez(
        npz_path,
        image=image.detach().cpu().numpy(),
        src=src.detach().cpu().numpy(),
        memory_image=memory_image.detach().cpu().numpy(),
        memory=memory.detach().cpu().numpy(),
        src_pos=src_pos.detach().cpu().numpy(),
        memory_image_pos=memory_image_pos.detach().cpu().numpy(),
        decoder_image_pe=decoder_image_pe.detach().cpu().numpy(),
        projected_feat_s0=projected_feat_s0.detach().cpu().numpy(),
        projected_feat_s1=projected_feat_s1.detach().cpu().numpy(),
        extra_per_object_embeddings=extra_per_object_embeddings.detach().cpu().numpy(),
        memory_out=memory_out.detach().cpu().numpy(),
        masks=masks.detach().cpu().numpy(),
        iou_pred=iou.detach().cpu().numpy(),
        mask_tokens_out=mask_tokens.detach().cpu().numpy(),
        object_score_logits=obj.detach().cpu().numpy(),
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "npz": str(npz_path),
        "cpp_inputs": str(inputs_dir),
        "expected_cpp_layout": str(expected_dir),
        "seed": args.seed,
        "device": str(device),
        "dtype": dtype_name,
        "buckets": b,
        "feat_size": h,
        "multiplex_count": m,
        "masks_per_object": k,
        "num_obj_ptr_tokens": 0,
        "shapes": {
            "memory_out": list(memory_out.shape),
            "image_embeddings": list(image_embeddings.shape),
            "masks": list(masks.shape),
            "iou_pred": list(iou.shape),
            "mask_tokens_out": list(mask_tokens.shape),
            "object_score_logits": list(obj.shape),
        },
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {npz_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
