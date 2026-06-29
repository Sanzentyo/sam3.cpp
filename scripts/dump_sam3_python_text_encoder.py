#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "einops",
#   "ftfy==6.1.1",
#   "huggingface_hub",
#   "iopath",
#   "numpy>=1.26,<2",
#   "pillow",
#   "psutil",
#   "pycocotools",
#   "regex",
#   "setuptools<81",
#   "timm",
#   "torch==2.8.0",
#   "torchvision==0.23.0",
#   "tqdm",
#   "typing_extensions",
# ]
# ///
"""Dump official Python SAM3 text encoder intermediates in ggml raw layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def write_f32_tensor(prefix: Path, values: np.ndarray) -> None:
    values = values.astype(np.float32, copy=False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".shape").write_text(
        ",".join(str(dim) for dim in values.shape) + "\n",
        encoding="utf-8",
    )
    values.ravel(order="F").tofile(prefix.with_suffix(".bin"))


def write_i32_tensor(prefix: Path, values: np.ndarray) -> None:
    values = values.astype(np.int32, copy=False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".shape").write_text(
        ",".join(str(dim) for dim in values.shape) + "\n",
        encoding="utf-8",
    )
    values.ravel(order="F").tofile(prefix.with_suffix(".bin"))


def dump(prefix: Path, tensor: torch.Tensor) -> None:
    write_f32_tensor(prefix, tensor.detach().float().cpu().numpy())


def tensor_e_l(tensor_b_l_e: torch.Tensor) -> torch.Tensor:
    return tensor_b_l_e[0].transpose(0, 1).contiguous()


def qkv_3e_l(block: torch.nn.Module, tensor_b_l_e: torch.Tensor) -> torch.Tensor:
    qkv = F.linear(
        tensor_b_l_e,
        block.attn.in_proj_weight,
        block.attn.in_proj_bias,
    )
    return qkv[0].transpose(0, 1).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tf32", choices=["on", "off"], default="on")
    parser.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.sam3_repo))

    allow_tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16"
        else torch.autocast(device_type="cuda", dtype=torch.float16)
        if args.dtype == "fp16"
        else torch.autocast(device_type="cuda", enabled=False)
    )

    from sam3.model_builder import build_sam3_predictor

    with autocast, torch.inference_mode():
        predictor = build_sam3_predictor(
            version=args.version,
            compile=False,
            use_fa3=False,
            async_loading_frames=False,
        )
        language_backbone = predictor.model.detector.backbone.language_backbone
        encoder = language_backbone.encoder
        tokenized = language_backbone.tokenizer(
            [args.prompt], context_length=language_backbone.context_length
        ).to("cuda")
        write_i32_tensor(args.out_dir / "token_ids", tokenized.detach().cpu().numpy()[0])

        x = encoder.token_embedding(tokenized)
        dump(args.out_dir / "text_token_embed", tensor_e_l(x))

        seq_len = x.shape[1]
        attn_mask = encoder.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]
        x = x + encoder.positional_embedding[:seq_len]
        dump(args.out_dir / "text_after_pos_embed", tensor_e_l(x))

        for block_idx, block in enumerate(encoder.transformer.resblocks):
            y = block.ln_1(x)
            dump(args.out_dir / f"text_block_{block_idx:02d}_after_ln1", tensor_e_l(y))
            dump(args.out_dir / f"text_block_{block_idx:02d}_qkv", qkv_3e_l(block, y))

            attn_out = block.ls_1(block.attention(q_x=y, attn_mask=attn_mask))
            dump(args.out_dir / f"text_block_{block_idx:02d}_attn_out", tensor_e_l(attn_out))
            x = x + attn_out
            dump(
                args.out_dir / f"text_block_{block_idx:02d}_after_attn_residual",
                tensor_e_l(x),
            )

            y = block.ln_2(x)
            dump(args.out_dir / f"text_block_{block_idx:02d}_after_ln2", tensor_e_l(y))
            mlp_fc1 = block.mlp.c_fc(y)
            dump(args.out_dir / f"text_block_{block_idx:02d}_mlp_fc1", tensor_e_l(mlp_fc1))
            mlp_gelu = block.mlp.gelu(mlp_fc1)
            dump(args.out_dir / f"text_block_{block_idx:02d}_mlp_gelu", tensor_e_l(mlp_gelu))
            mlp_out = block.ls_2(block.mlp.c_proj(mlp_gelu))
            dump(args.out_dir / f"text_block_{block_idx:02d}_mlp_out", tensor_e_l(mlp_out))
            x = x + mlp_out
            dump(args.out_dir / f"text_block_{block_idx:02d}_out", tensor_e_l(x))

        x = encoder.ln_final(x)
        dump(args.out_dir / "text_final_ln", tensor_e_l(x))
        x = language_backbone.resizer(x.transpose(0, 1))
        dump(args.out_dir / "text_features_2d", x[:, 0, :].transpose(0, 1).contiguous())
        torch.cuda.synchronize()

    print({"out_dir": str(args.out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
