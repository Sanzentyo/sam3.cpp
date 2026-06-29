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
"""Dump an official Python SAM3.1 multiplex memory-attention parity case."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feat-size", type=int, default=72)
    parser.add_argument("--buckets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3102)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "f32", "f16", "bf16"], default="auto")
    args = parser.parse_args()

    if args.feat_size <= 0:
        raise SystemExit("--feat-size must be positive")
    if args.buckets != 1:
        raise SystemExit("C++ SAM3.1 memory-attention slice currently supports --buckets 1")

    sys.path.insert(0, str(args.repo.resolve()))
    from sam3.model_builder import _create_multiplex_transformer

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

    model = _create_multiplex_transformer(use_fa3=False, use_rope_real=True).eval().to(device=device, dtype=dtype)
    state = {
        key.removeprefix("tracker.model.transformer.encoder."): value
        for key, value in ckpt.items()
        if key.startswith("tracker.model.transformer.encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"memory-attention state mismatch: missing={missing}, unexpected={unexpected}")

    b = args.buckets
    c = 256
    n = args.feat_size * args.feat_size
    image = torch.randn(n, 1, c, dtype=dtype, device=device)
    src = torch.randn(n, b, c, dtype=dtype, device=device)
    memory_image = torch.randn(n, 1, c, dtype=dtype, device=device)
    memory = torch.randn(n, b, c, dtype=dtype, device=device)
    image_pos = torch.randn(n, 1, c, dtype=dtype, device=device)
    src_pos = torch.randn(n, b, c, dtype=dtype, device=device)
    memory_image_pos = torch.randn(n, 1, c, dtype=dtype, device=device)
    memory_pos = torch.randn(n, b, c, dtype=dtype, device=device)

    # The encoder has batch_first=True, so it transposes seq-first inputs internally.
    output = src + 0.1 * src_pos
    image_b = image.transpose(0, 1)
    output_b = output.transpose(0, 1)
    memory_image_b = memory_image.transpose(0, 1)
    memory_b = memory.transpose(0, 1)
    src_pos_b = src_pos.transpose(0, 1)
    memory_image_pos_b = memory_image_pos.transpose(0, 1)

    expected: dict[str, torch.Tensor] = {
        "sam31_mem_attn_input": output,
    }
    for i, layer in enumerate(model.encoder.layers):
        output_b = layer._forward_sa(output_b, src_pos_b)
        expected[f"sam31_mem_attn_layer{i}_after_sa"] = output_b.transpose(0, 1)
        output_b = layer._forward_ca(
            image=image_b,
            tgt=output_b,
            memory_image=memory_image_b,
            memory=memory_b,
            query_pos=src_pos_b,
            memory_image_pos=memory_image_pos_b,
            num_k_exclude_rope=0,
        )
        expected[f"sam31_mem_attn_layer{i}_after_ca"] = output_b.transpose(0, 1)
        ffn = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(output_b)))))
        output_b = output_b + layer.dropout3(ffn)
        expected[f"sam31_mem_attn_layer{i}_after_ffn"] = output_b.transpose(0, 1)

    memory_out = model.encoder.norm(output_b).transpose(0, 1)
    expected["sam31_mem_attn_output"] = memory_out

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
        "image_pos": image_pos,
        "src_pos": src_pos,
        "memory_image_pos": memory_image_pos,
        "memory_pos": memory_pos,
    }.items():
        write_raw_tensor(inputs_dir, name, seq_b_c_to_ggml(tensor))

    for name, tensor in expected.items():
        write_raw_tensor(expected_dir, name, seq_b_c_to_ggml(tensor))

    npz_path = args.out / "official_memory_attention_case.npz"
    np.savez(
        npz_path,
        image=image.detach().cpu().numpy(),
        src=src.detach().cpu().numpy(),
        memory_image=memory_image.detach().cpu().numpy(),
        memory=memory.detach().cpu().numpy(),
        image_pos=image_pos.detach().cpu().numpy(),
        src_pos=src_pos.detach().cpu().numpy(),
        memory_image_pos=memory_image_pos.detach().cpu().numpy(),
        memory_pos=memory_pos.detach().cpu().numpy(),
        **{name: tensor.detach().cpu().numpy() for name, tensor in expected.items()},
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
        "feat_size": args.feat_size,
        "num_obj_ptr_tokens": 0,
        "shapes": {
            name: list(tensor.shape)
            for name, tensor in {
                "image": image,
                "src": src,
                "memory_image": memory_image,
                "memory": memory,
                "image_pos": image_pos,
                "src_pos": src_pos,
                "memory_image_pos": memory_image_pos,
                "memory_pos": memory_pos,
                **expected,
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
