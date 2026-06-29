#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.8.0",
#   "torchvision==0.23.0",
#   "numpy>=1.26,<2",
#   "pillow>=11.0",
#   "timm>=1.0",
#   "iopath>=0.1.10",
#   "einops>=0.7",
#   "pycocotools>=2.0.11",
#   "psutil>=7.0",
#   "ftfy>=6.3",
#   "regex>=2026.5.9",
#   "huggingface_hub",
#   "setuptools<81",
# ]
# ///
"""Dump official Python SAM3.1 state after the synthetic mask-init frame."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def make_frame(width: int, height: int, offset: int) -> Image.Image:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (xx + offset) % 256
    rgb[..., 1] = (yy * 2 + offset) % 256
    rgb[..., 2] = (xx + yy + offset) % 256
    return Image.fromarray(rgb, mode="RGB")


def scale_dim(size: int, per_mille: int) -> int:
    return min(max(size * per_mille // 1000, 0), size)


def mask_case_rect(width: int, height: int, mask_case: str) -> tuple[int, int, int, int]:
    if mask_case == "center":
        return (
            scale_dim(width, 250),
            scale_dim(height, 250),
            scale_dim(width, 750),
            scale_dim(height, 750),
        )
    if mask_case == "small-center":
        return (
            scale_dim(width, 375),
            scale_dim(height, 375),
            scale_dim(width, 625),
            scale_dim(height, 625),
        )
    if mask_case == "left-wide":
        return (
            scale_dim(width, 100),
            scale_dim(height, 200),
            scale_dim(width, 550),
            scale_dim(height, 800),
        )
    if mask_case == "bottom-band":
        return (
            scale_dim(width, 200),
            scale_dim(height, 550),
            scale_dim(width, 800),
            scale_dim(height, 900),
        )
    raise ValueError(f"unknown mask case {mask_case}")


def make_rect_mask(width: int, height: int, mask_case: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = mask_case_rect(width, height, mask_case)
    mask[y0:y1, x0:x1] = 255
    return mask


def write_raw_tensor(base: Path, name: str, values: np.ndarray) -> None:
    values = values.astype(np.float32, copy=False)
    (base / f"{name}.shape").write_text(
        " ".join(str(dim) for dim in values.shape) + "\n",
        encoding="utf-8",
    )
    values.ravel(order="F").tofile(base / f"{name}.bin")


def nchw_to_ggml_dwhb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().transpose(1, 3, 2, 0)


def nchw_to_ggml_whcb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().transpose(3, 2, 1, 0)


def nhwc_flat_to_ggml_dwhb(tensor: torch.Tensor, height: int, width: int) -> np.ndarray:
    # Official stores image features as (HW), B, C with H-major flattening.
    values = tensor.detach().float().cpu().numpy()
    batch = values.shape[1]
    channels = values.shape[2]
    return values.reshape(height, width, batch, channels).transpose(3, 1, 0, 2)


def nhwc_to_ggml_dwhb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().transpose(3, 2, 1, 0)


def seq_b_c_to_ggml_dnb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().transpose(2, 0, 1)


def run_backbone_cache(model: Any, state: dict[str, Any], frame_idx: int) -> float:
    per_frame_prompt = state.get("per_frame_geometric_prompt", [])
    geometric_prompt = (
        per_frame_prompt[frame_idx]
        if frame_idx < len(per_frame_prompt) and per_frame_prompt[frame_idx] is not None
        else state["constants"]["empty_geometric_prompt"]
    )
    start = time.perf_counter()
    model.run_backbone_and_detection(
        frame_idx=frame_idx,
        num_frames=state["num_frames"],
        input_batch=state["input_batch"],
        geometric_prompt=geometric_prompt,
        feature_cache=state["feature_cache"],
        reverse=False,
    )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "min": float(data.min().cpu().item()),
        "max": float(data.max().cpu().item()),
        "mean": float(data.mean().cpu().item()),
    }


def parse_block_selector(value: str | None, depth: int, fallback: set[int]) -> set[int]:
    if value is None:
        return set(fallback)
    value = value.strip()
    if value == "":
        return set()
    if value == "all":
        return set(range(depth))

    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
        else:
            start = end = int(token)
        if start < 0 or end < start:
            raise ValueError(f"invalid block selector token: {token!r}")
        selected.update(i for i in range(start, end + 1) if i < depth)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sam31-mask-init-state-python"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--mask-case",
        choices=("center", "small-center", "left-wide", "bottom-band"),
        default="center",
    )
    parser.add_argument("--frame1-offset", type=int, default=3)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument(
        "--dump-propagation-attn",
        action="store_true",
        help="Also run frame-1 propagation and dump memory-attention self-attention tensors.",
    )
    parser.add_argument(
        "--propagation-attn-layers",
        default="0",
        help="Comma/range selector for memory-attention layers to dump when --dump-propagation-attn is set.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for official SAM3.1 mask-init state dump")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    sys.path.insert(0, str(args.sam3_repo.resolve()))
    torch.set_grad_enabled(False)
    allow_tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    autocast_enabled = args.dtype != "fp32"

    def precision_context() -> contextlib.AbstractContextManager[None]:
        if autocast_enabled:
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    from sam3 import build_sam3_predictor
    from sam3.sam.rope import apply_rotary_enc, apply_rotary_enc_real
    from sam3.model.vitdet import window_partition, window_unpartition

    captures: dict[str, torch.Tensor] = {}
    frames = [make_frame(args.width, args.height, 0), make_frame(args.width, args.height, args.frame1_offset)]
    input_mask_np = make_rect_mask(args.width, args.height, args.mask_case)

    predictor = build_sam3_predictor(
        checkpoint_path=str(args.checkpoint),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=1,
        use_fa3=False,
        use_rope_real=True,
        async_loading_frames=False,
    )
    model = predictor.model
    original_trunk_forward = model.detector.backbone.vision_backbone.trunk.forward
    original_patch_embed_forward = model.detector.backbone.vision_backbone.trunk.patch_embed.forward
    original_ln_pre_forward = model.detector.backbone.vision_backbone.trunk.ln_pre.forward
    original_forward = model.tracker.maskmem_backbone.forward
    dump_all_vit_blocks = os.environ.get("SAM31_MASK_INIT_DUMP_ALL_VIT_BLOCKS", "0") not in {
        "",
        "0",
    }
    vit_depth = len(model.detector.backbone.vision_backbone.trunk.blocks)
    block_capture_indices = (
        set(range(vit_depth))
        if dump_all_vit_blocks
        else {0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 31}
    )
    stage_capture_blocks = parse_block_selector(
        os.environ.get("SAM31_MASK_INIT_DUMP_VIT_STAGE_BLOCKS"),
        vit_depth,
        {0, 12, 13, 14, 15, 23, 31},
    )
    original_block_forwards = {
        i: model.detector.backbone.vision_backbone.trunk.blocks[i].forward
        for i in block_capture_indices | stage_capture_blocks
    }

    def capture_once(name: str, tensor: torch.Tensor) -> None:
        if name not in captures:
            captures[name] = tensor.detach().clone()

    def rope_qkv_for_dump(layer: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rope = layer.self_attention_rope
        batch, n_tokens, channels = q.shape
        head_dim = channels // rope.num_heads
        qh = q.reshape(batch, n_tokens, rope.num_heads, head_dim).transpose(1, 2)
        kh = k.reshape(batch, n_tokens, rope.num_heads, head_dim).transpose(1, 2)
        vh = v.reshape(batch, n_tokens, rope.num_heads, head_dim).transpose(1, 2)

        width = height = int(math.sqrt(n_tokens))
        rope.freqs_cis = rope.freqs_cis.to(q.device)
        if rope.freqs_cis.shape[0] != n_tokens:
            rope.freqs_cis = rope.compute_cis(end_x=width, end_y=height, device=q.device)
            if rope.use_rope_real:
                rope.freqs_cis_real = rope.freqs_cis.real
                rope.freqs_cis_imag = rope.freqs_cis.imag
        if rope.use_rope_real:
            qh, kh = apply_rotary_enc_real(
                qh,
                kh,
                freqs_cis_real=rope.freqs_cis_real,
                freqs_cis_imag=rope.freqs_cis_imag,
                repeat_freqs_k=rope.rope_k_repeat,
            )
        else:
            qh, kh = apply_rotary_enc(qh, kh, rope.freqs_cis, repeat_freqs_k=rope.rope_k_repeat)
        return qh, kh, vh

    def bhnd_to_ggml_dnh(tensor: torch.Tensor) -> np.ndarray:
        return tensor[0].detach().float().cpu().numpy().transpose(2, 1, 0)

    def bnc_to_ggml_dhn(tensor: torch.Tensor, num_heads: int) -> np.ndarray:
        batch, n_tokens, channels = tensor.shape
        head_dim = channels // num_heads
        return (
            tensor.reshape(batch, n_tokens, num_heads, head_dim)[0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .transpose(2, 1, 0)
        )

    propagation_attn_layers = parse_block_selector(
        args.propagation_attn_layers,
        len(model.tracker.transformer.encoder.layers),
        {0},
    )
    original_sa_forwards: dict[int, Any] = {}
    capture_propagation_attn = False

    def make_sa_capture(layer_idx: int, layer: Any) -> Any:
        original = layer._forward_sa

        def capture_forward_sa(self: Any, tgt: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
            if not capture_propagation_attn:
                return original(tgt, query_pos)
            tgt2 = self.norm1(tgt)
            q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
            q = self.self_attn_q_proj(q)
            k = self.self_attn_k_proj(k)
            v = self.self_attn_v_proj(tgt2)
            qh, kh, vh = rope_qkv_for_dump(self, q, k, v)
            out = self.self_attention_rope(q, k, v)
            prefix = f"prop_mem_attn_layer{layer_idx}_sa"
            capture_once(f"{prefix}_q", qh)
            capture_once(f"{prefix}_k", kh)
            capture_once(f"{prefix}_v", vh)
            capture_once(f"{prefix}_out", out)
            tgt2 = self.self_attn_out_proj(out)
            return tgt + self.dropout1(tgt2)

        original_sa_forwards[layer_idx] = original
        return MethodType(capture_forward_sa, layer)

    def attention_with_captures(block: Any, x: torch.Tensor, prefix: str) -> torch.Tensor:
        attn = block.attn
        s = 1 if attn.cls_token else 0
        if x.ndim == 4:
            batch, height, width, _ = x.shape
            assert s == 0
            length = height * width
        else:
            batch, length, _ = x.shape
            height = width = int(length**0.5)

        qkv_linear = attn.qkv(x)
        capture_once(f"{prefix}_qkv", qkv_linear)
        qkv = qkv_linear.reshape(batch, length, 3, attn.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attn._apply_rope(q, k)

        if attn.use_rel_pos:
            raise RuntimeError("block0 diagnostic does not support rel_pos yet")
        attn_type_name = getattr(attn.attn_type, "name", str(attn.attn_type))
        if attn_type_name != "Vanilla":
            raise RuntimeError(f"unsupported attention type: {attn.attn_type}")

        if attn.use_fa3:
            from sam3.perflib.fa3 import flash_attn_func

            attn_out = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        if x.ndim == 4:
            attn_out = (
                attn_out.view(batch, attn.num_heads, height, width, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(batch, height, width, -1)
            )
        else:
            attn_out = (
                attn_out.view(batch, attn.num_heads, length, -1)
                .permute(0, 2, 1, 3)
                .reshape(batch, length, -1)
            )

        capture_once(f"{prefix}_attn_core", attn_out)
        proj = attn.proj(attn_out)
        capture_once(f"{prefix}_attn_proj", proj)
        return proj

    def block_forward_with_captures(block_idx: int, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        block = model.detector.backbone.vision_backbone.trunk.blocks[block_idx]
        prefix = f"vit_block_{block_idx:02d}"
        capture_once(f"{prefix}_input", x)
        shortcut = x

        x = block.norm1(x)
        capture_once(f"{prefix}_norm1", x)

        pad_hw: tuple[int, int] | None = None
        original_hw: tuple[int, int] | None = None
        if block.window_size > 0:
            original_hw = (x.shape[1], x.shape[2])
            x, pad_hw = window_partition(x, block.window_size)
            capture_once(f"{prefix}_window_part", x)

        x = block.ls1(attention_with_captures(block, x, prefix))

        if block.window_size > 0:
            assert pad_hw is not None and original_hw is not None
            x = window_unpartition(x, block.window_size, pad_hw, original_hw)
            capture_once(f"{prefix}_window_unpart", x)

        x = shortcut + block.dropout(block.drop_path(x))
        capture_once(f"{prefix}_after_attn_residual", x)

        norm2 = block.norm2(x)
        capture_once(f"{prefix}_norm2", norm2)
        mlp_fc1 = block.mlp.fc1(norm2)
        capture_once(f"{prefix}_mlp_fc1", mlp_fc1)
        mlp_gelu = block.mlp.act(mlp_fc1)
        capture_once(f"{prefix}_mlp_gelu", mlp_gelu)
        mlp_fc2 = block.mlp.fc2(mlp_gelu)
        capture_once(f"{prefix}_mlp_fc2", mlp_fc2)
        mlp_out = mlp_fc2
        capture_once(f"{prefix}_mlp", mlp_out)

        out = x + block.dropout(block.drop_path(block.ls2(mlp_out)))
        capture_once(f"{prefix}_out", out)
        return out

    def block0_forward_with_captures(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return block_forward_with_captures(0, x, *args, **kwargs)

    def capture_trunk_forward(*args: Any, **kwargs: Any) -> Any:
        out = original_trunk_forward(*args, **kwargs)
        last = out[-1]
        capture_once("vit_output", getattr(last, "tensors", last))
        return out

    def capture_patch_embed_forward(*args: Any, **kwargs: Any) -> Any:
        capture_once("vit_patch_embed_input", args[0])
        out = original_patch_embed_forward(*args, **kwargs)
        capture_once("vit_patch_embed", out)
        return out

    def capture_ln_pre_forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        capture_once("vit_after_pos", x)
        out = original_ln_pre_forward(x, *args, **kwargs)
        capture_once("vit_block_00_input", out)
        return out

    def make_block_capture(block_idx: int) -> Any:
        original = original_block_forwards[block_idx]

        def capture_block_forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if block_idx in stage_capture_blocks:
                return block_forward_with_captures(block_idx, x, *args, **kwargs)
            out = original(x, *args, **kwargs)
            capture_once(f"vit_block_{block_idx:02d}_out", out)
            return out

        return capture_block_forward

    def capture_memory_forward(pix_feat: torch.Tensor, masks: torch.Tensor, *rest: Any, **kwargs: Any) -> Any:
        captures["memory_pix_feat"] = pix_feat.detach().clone()
        captures["memory_mux_mask_input"] = masks.detach().clone()
        out = original_forward(pix_feat, masks, *rest, **kwargs)
        captures["memory_vision_features"] = out["vision_features"].detach().clone()
        captures["memory_vision_pos_enc"] = out["vision_pos_enc"][-1].detach().clone()
        return out

    model.detector.backbone.vision_backbone.trunk.forward = capture_trunk_forward
    model.detector.backbone.vision_backbone.trunk.patch_embed.forward = capture_patch_embed_forward
    model.detector.backbone.vision_backbone.trunk.ln_pre.forward = capture_ln_pre_forward
    for block_idx in block_capture_indices:
        model.detector.backbone.vision_backbone.trunk.blocks[block_idx].forward = make_block_capture(block_idx)
    model.tracker.maskmem_backbone.forward = capture_memory_forward
    if args.dump_propagation_attn:
        for layer_idx in propagation_attn_layers:
            layer = model.tracker.transformer.encoder.layers[layer_idx]
            layer._forward_sa = make_sa_capture(layer_idx, layer)

    state = model.init_state(resource_path=frames, offload_video_to_cpu=False, async_loading_frames=False)
    input_image_preprocessed = state["input_batch"].img_batch.tensors[:1].detach().clone()
    mask_t = torch.from_numpy(input_mask_np > 127).to(device="cuda", dtype=torch.float32).unsqueeze(0)

    with precision_context():
        frame0_cache_ms = run_backbone_cache(model, state, 0)

        start = time.perf_counter()
        tracker_states = model._tracker_add_new_objects(
            frame_idx=0,
            num_frames=state["num_frames"],
            new_obj_ids=[1],
            new_obj_masks=mask_t,
            tracker_states_local=state["sam2_inference_states"],
            orig_vid_height=state["orig_height"],
            orig_vid_width=state["orig_width"],
            feature_cache=state["feature_cache"],
        )
        torch.cuda.synchronize()
        add_mask_ms = (time.perf_counter() - start) * 1000.0

        frame1_cache_ms = None
        propagate_ms = None
        if args.dump_propagation_attn:
            frame1_cache_ms = run_backbone_cache(model, state, 1)
            start = time.perf_counter()
            capture_propagation_attn = True
            for frame_idx, *_rest in model.tracker.propagate_in_video(
                tracker_states[0],
                start_frame_idx=0,
                max_frame_num_to_track=1,
                reverse=False,
                tqdm_disable=True,
                run_mem_encoder=True,
            ):
                if frame_idx == 1:
                    break
            capture_propagation_attn = False
            torch.cuda.synchronize()
            propagate_ms = (time.perf_counter() - start) * 1000.0

    tracker_state = tracker_states[0]
    cond = tracker_state["output_dict"]["cond_frame_outputs"][0]
    h = int(cond["maskmem_features"].shape[-2])
    w = int(cond["maskmem_features"].shape[-1])

    out_dir = args.out
    expected_dir = out_dir / "expected_cpp_layout"
    expected_dir.mkdir(parents=True, exist_ok=True)

    write_raw_tensor(expected_dir, "input_image_preprocessed", nchw_to_ggml_whcb(input_image_preprocessed))
    write_raw_tensor(expected_dir, "vit_patch_embed_input", nchw_to_ggml_whcb(captures["vit_patch_embed_input"]))
    write_raw_tensor(expected_dir, "vit_output", nchw_to_ggml_dwhb(captures["vit_output"]))
    write_raw_tensor(expected_dir, "vit_patch_embed", nhwc_to_ggml_dwhb(captures["vit_patch_embed"]))
    write_raw_tensor(expected_dir, "vit_after_pos", nhwc_to_ggml_dwhb(captures["vit_after_pos"]))
    write_raw_tensor(expected_dir, "vit_block_00_input", nhwc_to_ggml_dwhb(captures["vit_block_00_input"]))
    stage_names = (
        "input",
        "norm1",
        "window_part",
        "qkv",
        "attn_core",
        "attn_proj",
        "window_unpart",
        "after_attn_residual",
        "norm2",
        "mlp_fc1",
        "mlp_gelu",
        "mlp_fc2",
        "mlp",
    )
    for block_idx in sorted(stage_capture_blocks):
        for stage_name in stage_names:
            name = f"vit_block_{block_idx:02d}_{stage_name}"
            if name in captures:
                write_raw_tensor(expected_dir, name, nhwc_to_ggml_dwhb(captures[name]))
    for block_idx in sorted(block_capture_indices):
        name = f"vit_block_{block_idx:02d}_out"
        write_raw_tensor(expected_dir, name, nhwc_to_ggml_dwhb(captures[name]))
    memory_mux_mask_input = captures["memory_mux_mask_input"]
    write_raw_tensor(expected_dir, "memory_mux_mask_input", nchw_to_ggml_whcb(memory_mux_mask_input))
    interpol_size = getattr(model.tracker.maskmem_backbone.mask_downsampler, "interpol_size", None)
    if interpol_size is not None and list(memory_mux_mask_input.shape[-2:]) != list(interpol_size):
        memory_mux_mask_interpolated = F.interpolate(
            memory_mux_mask_input.float(),
            size=interpol_size,
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
    else:
        memory_mux_mask_interpolated = memory_mux_mask_input.float()
    write_raw_tensor(
        expected_dir,
        "memory_mux_mask_interpolated",
        nchw_to_ggml_whcb(memory_mux_mask_interpolated),
    )
    write_raw_tensor(expected_dir, "memory_pix_feat", nchw_to_ggml_dwhb(captures["memory_pix_feat"]))
    write_raw_tensor(expected_dir, "stored_spatial_feats", nchw_to_ggml_dwhb(cond["maskmem_features"]))
    write_raw_tensor(expected_dir, "stored_spatial_pe", nchw_to_ggml_dwhb(cond["maskmem_pos_enc"][-1]))
    write_raw_tensor(expected_dir, "image_features", nhwc_flat_to_ggml_dwhb(cond["image_features"], h, w))
    write_raw_tensor(expected_dir, "image_pos_enc", nhwc_flat_to_ggml_dwhb(cond["image_pos_enc"], h, w))
    write_raw_tensor(expected_dir, "obj_ptr", seq_b_c_to_ggml_dnb(cond["obj_ptr"]))
    write_raw_tensor(expected_dir, "object_score_logits", cond["object_score_logits"].detach().float().cpu().numpy())
    for layer_idx in sorted(propagation_attn_layers):
        prefix = f"prop_mem_attn_layer{layer_idx}_sa"
        if f"{prefix}_q" in captures:
            write_raw_tensor(expected_dir, f"{prefix}_q", bhnd_to_ggml_dnh(captures[f"{prefix}_q"]))
            write_raw_tensor(expected_dir, f"{prefix}_k", bhnd_to_ggml_dnh(captures[f"{prefix}_k"]))
            write_raw_tensor(expected_dir, f"{prefix}_v", bhnd_to_ggml_dnh(captures[f"{prefix}_v"]))
            layer = model.tracker.transformer.encoder.layers[layer_idx]
            write_raw_tensor(
                expected_dir,
                f"{prefix}_out",
                bnc_to_ggml_dhn(captures[f"{prefix}_out"], layer.self_attention_rope.num_heads),
            )

    summary = {
        "status": "ok",
        "version": "sam3.1",
        "checkpoint": str(args.checkpoint),
        "width": args.width,
        "height": args.height,
        "mask_case": args.mask_case,
        "frame1_offset": args.frame1_offset,
        "precision": args.dtype,
        "tf32_policy": args.tf32,
        "frame0_cache_ms": frame0_cache_ms,
        "add_mask_ms": add_mask_ms,
        "frame1_cache_ms": frame1_cache_ms,
        "propagate_ms": propagate_ms,
        "dump_propagation_attn": args.dump_propagation_attn,
        "propagation_attn_layers": sorted(propagation_attn_layers),
        "expected_cpp_layout": str(expected_dir),
        "conditioning_objects": sorted(int(x) for x in cond["conditioning_objects"]),
        "torch": {
            "version": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "tensors": {
            name: tensor_summary(tensor)
            for name, tensor in {
                "memory_mux_mask_input": memory_mux_mask_input,
                "input_image_preprocessed": input_image_preprocessed,
                "vit_patch_embed_input": captures["vit_patch_embed_input"],
                "vit_output": captures["vit_output"],
                "vit_patch_embed": captures["vit_patch_embed"],
                "vit_after_pos": captures["vit_after_pos"],
                "vit_block_00_input": captures["vit_block_00_input"],
                **{
                    name: captures[name]
                    for block_idx in sorted(stage_capture_blocks)
                    for stage_name in stage_names
                    for name in (f"vit_block_{block_idx:02d}_{stage_name}",)
                    if name in captures
                },
                **{
                    f"vit_block_{block_idx:02d}_out": captures[f"vit_block_{block_idx:02d}_out"]
                    for block_idx in sorted(block_capture_indices)
                },
                "memory_mux_mask_interpolated": memory_mux_mask_interpolated,
                "memory_pix_feat": captures["memory_pix_feat"],
                "stored_spatial_feats": cond["maskmem_features"],
                "stored_spatial_pe": cond["maskmem_pos_enc"][-1],
                "image_features": cond["image_features"],
                "image_pos_enc": cond["image_pos_enc"],
                "obj_ptr": cond["obj_ptr"],
                "object_score_logits": cond["object_score_logits"],
            }.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
