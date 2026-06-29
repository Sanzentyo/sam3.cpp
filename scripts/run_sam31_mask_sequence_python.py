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
"""Run official Python SAM3.1 on the same synthetic mask-init sequence as C++."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
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


def write_pgm(path: Path, mask: np.ndarray) -> None:
    path.write_bytes(
        f"P5\n{mask.shape[1]} {mask.shape[0]}\n255\n".encode("ascii") + mask.astype(np.uint8).tobytes()
    )


def frame_mask_stem(frame_idx: int) -> str:
    return f"frame{frame_idx:04d}_mask"


def write_raw_tensor(base: Path, name: str, values: np.ndarray) -> None:
    values = values.astype(np.float32, copy=False)
    (base / f"{name}.shape").write_text(
        ",".join(str(dim) for dim in values.shape) + "\n",
        encoding="utf-8",
    )
    values.ravel(order="F").tofile(base / f"{name}.bin")


def bchw_to_seq_b_c(values: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = values.shape
    return values.reshape(batch, channels, height * width).permute(2, 0, 1)


def bchw_to_ggml_image(values: torch.Tensor) -> np.ndarray:
    return values.detach().float().cpu().numpy().transpose(1, 3, 2, 0)


def nhwc_to_ggml_image(values: torch.Tensor) -> np.ndarray:
    return values.detach().float().cpu().numpy().transpose(3, 2, 1, 0)


def seq_b_c_to_ggml(values: torch.Tensor) -> np.ndarray:
    return values.detach().float().cpu().numpy().transpose(2, 0, 1)


def batch_seq_c_to_ggml(values: torch.Tensor) -> np.ndarray:
    return values.detach().float().cpu().numpy().transpose(2, 1, 0)


def mux_masks_to_ggml(values: torch.Tensor) -> np.ndarray:
    batch, multiplex_count, mask_count, height, width = values.shape
    return (
        values.detach()
        .float()
        .cpu()
        .numpy()
        .reshape(batch, multiplex_count * mask_count, height * width)
        .transpose(2, 1, 0)
    )


def mux_iou_to_ggml(values: torch.Tensor) -> np.ndarray:
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    return values.detach().float().cpu().numpy().transpose(2, 1, 0)


def mux_tokens_to_ggml(values: torch.Tensor) -> np.ndarray:
    batch, multiplex_count, mask_count, channels = values.shape
    return (
        values.detach()
        .float()
        .cpu()
        .numpy()
        .reshape(batch, multiplex_count * mask_count, channels)
        .transpose(2, 1, 0)
    )


def install_propagation_dump_hooks(model: Any, dump_dir: Path) -> Callable[[], None]:
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_counts: dict[str, int] = {}

    def dump_tensor(name: str, values: np.ndarray) -> None:
        write_raw_tensor(dump_dir, name, values)
        dump_counts[name] = dump_counts.get(name, 0) + 1
        (dump_dir / "dump_counts.json").write_text(
            json.dumps(dump_counts, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    tracker = model.tracker
    original_prepare = tracker._prepare_memory_conditioned_features
    original_encoder_forward = tracker.transformer.encoder.forward
    original_predict_masks = tracker.sam_mask_decoder.predict_masks
    original_decoder_forward = tracker.sam_mask_decoder.forward
    original_sam_transformer_forward = tracker.sam_mask_decoder.transformer.forward
    original_layer_methods: list[tuple[Any, Any, Any, Any]] = []
    original_decoder_layer_methods: list[tuple[Any, Any]] = []
    forward_hooks: list[Any] = []

    for layer_idx, layer in enumerate(tracker.transformer.encoder.layers):
        original_sa = layer._forward_sa
        original_ca = layer._forward_ca
        original_forward = layer.forward
        original_layer_methods.append((layer, original_sa, original_ca, original_forward))

        def sa_wrapper(
            tgt: torch.Tensor,
            query_pos: torch.Tensor,
            *,
            _idx: int = layer_idx,
            _layer: Any = layer,
        ) -> torch.Tensor:
            tgt2 = _layer.norm1(tgt)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_sa_norm", batch_seq_c_to_ggml(tgt2))
            q = k = tgt2 + query_pos if _layer.pos_enc_at_attn else tgt2
            q = _layer.self_attn_q_proj(q)
            k = _layer.self_attn_k_proj(k)
            v = _layer.self_attn_v_proj(tgt2)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_sa_q", batch_seq_c_to_ggml(q))
            dump_tensor(f"sam31_mem_attn_layer{_idx}_sa_k", batch_seq_c_to_ggml(k))
            dump_tensor(f"sam31_mem_attn_layer{_idx}_sa_v", batch_seq_c_to_ggml(v))
            attn = _layer.self_attention_rope(q, k, v)
            tgt2 = _layer.self_attn_out_proj(attn)
            out = tgt + _layer.dropout1(tgt2)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_after_sa", batch_seq_c_to_ggml(out))
            return out

        def ca_wrapper(*args: Any, _idx: int = layer_idx, _layer: Any = layer, **kwargs: Any) -> torch.Tensor:
            num_k_exclude_rope = kwargs.get("num_k_exclude_rope", 0)
            rope_kwargs = {"num_k_exclude_rope": num_k_exclude_rope} if num_k_exclude_rope > 0 else {}
            image = kwargs["image"]
            tgt = kwargs["tgt"]
            memory_image = kwargs["memory_image"]
            memory = kwargs["memory"]
            query_pos = kwargs["query_pos"]
            memory_image_pos = kwargs["memory_image_pos"]

            tgt2 = _layer.norm2(tgt)
            q = _layer.image_cross_attn_q_proj(image) + _layer.cross_attn_q_proj(tgt2)
            if _layer.pos_enc_at_cross_attn_queries:
                q = q + query_pos
            k = _layer.image_cross_attn_k_proj(memory_image) + _layer.cross_attn_k_proj(memory)
            if _layer.pos_enc_at_cross_attn_keys:
                k = k + memory_image_pos
            v = _layer.cross_attn_v_proj(memory)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_ca_q", batch_seq_c_to_ggml(q))
            dump_tensor(f"sam31_mem_attn_layer{_idx}_ca_k", batch_seq_c_to_ggml(k))
            dump_tensor(f"sam31_mem_attn_layer{_idx}_ca_v", batch_seq_c_to_ggml(v))
            attn = _layer.cross_attention_rope(q, k, v, **rope_kwargs)
            tgt2 = _layer.cross_attn_out_proj(attn)
            out = tgt + _layer.dropout2(tgt2)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_after_ca", batch_seq_c_to_ggml(out))
            return out

        def layer_forward_wrapper(
            *args: Any,
            _idx: int = layer_idx,
            _layer: Any = layer,
            _original: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            required = {
                "image",
                "tgt",
                "memory_image",
                "memory",
                "query_pos",
                "memory_image_pos",
            }
            if not required.issubset(kwargs):
                out = _original(*args, **kwargs)
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], torch.Tensor):
                    dump_tensor(f"sam31_mem_attn_layer{_idx}_after_ffn", batch_seq_c_to_ggml(out[1]))
                return out

            image = kwargs["image"]
            tgt = kwargs["tgt"]
            num_k_exclude_rope = kwargs.get("num_k_exclude_rope", 0)

            def run_ca(current: torch.Tensor) -> torch.Tensor:
                return _layer._forward_ca(
                    image=image,
                    tgt=current,
                    memory_image=kwargs["memory_image"],
                    memory=kwargs["memory"],
                    query_pos=kwargs["query_pos"],
                    memory_image_pos=kwargs["memory_image_pos"],
                    num_k_exclude_rope=num_k_exclude_rope,
                )

            if _layer.cross_attention_first:
                tgt = run_ca(tgt)
                tgt = _layer._forward_sa(tgt, kwargs["query_pos"])
            else:
                tgt = _layer._forward_sa(tgt, kwargs["query_pos"])
                tgt = run_ca(tgt)

            ffn_norm = _layer.norm3(tgt)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_ffn_norm", batch_seq_c_to_ggml(ffn_norm))
            ffn_fc1 = _layer.linear1(ffn_norm)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_ffn_fc1", batch_seq_c_to_ggml(ffn_fc1))
            ffn_gelu = _layer.activation(ffn_fc1)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_ffn_gelu", batch_seq_c_to_ggml(ffn_gelu))
            ffn_fc2 = _layer.linear2(_layer.dropout(ffn_gelu))
            dump_tensor(f"sam31_mem_attn_layer{_idx}_ffn_fc2", batch_seq_c_to_ggml(ffn_fc2))
            tgt = tgt + _layer.dropout3(ffn_fc2)
            dump_tensor(f"sam31_mem_attn_layer{_idx}_after_ffn", batch_seq_c_to_ggml(tgt))
            return image, tgt

        layer._forward_sa = sa_wrapper
        layer._forward_ca = ca_wrapper
        layer.forward = layer_forward_wrapper

    for layer_idx, layer in enumerate(tracker.sam_mask_decoder.transformer.layers):
        original_forward = layer.forward
        original_decoder_layer_methods.append((layer, original_forward))

        def decoder_layer_forward_wrapper(
            *,
            queries: torch.Tensor,
            keys: torch.Tensor,
            query_pe: torch.Tensor,
            key_pe: torch.Tensor,
            _idx: int = layer_idx,
            _layer: Any = layer,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if _layer.skip_first_layer_pe:
                queries = _layer.self_attn(q=queries, k=queries, v=queries)
            else:
                q = queries + query_pe
                attn_out = _layer.self_attn(q=q, k=q, v=queries)
                queries = queries + attn_out
            queries = _layer.norm1(queries)
            dump_tensor(f"sam31_mux_block{_idx}_after_sa", batch_seq_c_to_ggml(queries))

            q = queries + query_pe
            k = keys + key_pe
            attn_out = _layer.cross_attn_token_to_image(q=q, k=k, v=keys)
            queries = _layer.norm2(queries + attn_out)
            dump_tensor(
                f"sam31_mux_block{_idx}_after_token_to_image",
                batch_seq_c_to_ggml(queries),
            )

            queries = _layer.norm3(queries + _layer.mlp(queries))
            dump_tensor(f"sam31_mux_block{_idx}_after_mlp", batch_seq_c_to_ggml(queries))

            q = queries + query_pe
            k = keys + key_pe
            img2tok_attn = _layer.cross_attn_image_to_token
            q_proj = img2tok_attn.q_proj(k)
            k_proj = img2tok_attn.k_proj(q)
            v_proj = img2tok_attn.v_proj(queries)
            dump_tensor(
                f"sam31_mux_block{_idx}_img2tok_attn_q_proj",
                batch_seq_c_to_ggml(q_proj),
            )
            dump_tensor(
                f"sam31_mux_block{_idx}_img2tok_attn_k_proj",
                batch_seq_c_to_ggml(k_proj),
            )
            dump_tensor(
                f"sam31_mux_block{_idx}_img2tok_attn_v_proj",
                batch_seq_c_to_ggml(v_proj),
            )
            q_heads = img2tok_attn._separate_heads(q_proj, img2tok_attn.num_heads)
            k_heads = img2tok_attn._separate_heads(k_proj, img2tok_attn.num_heads)
            v_heads = img2tok_attn._separate_heads(v_proj, img2tok_attn.num_heads)
            attn_core = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0)
            attn_core = img2tok_attn._recombine_heads(attn_core)
            dump_tensor(
                f"sam31_mux_block{_idx}_img2tok_attn_core",
                batch_seq_c_to_ggml(attn_core),
            )
            attn_out = img2tok_attn.out_proj(attn_core)
            keys = _layer.norm4(keys + attn_out)
            dump_tensor(
                f"sam31_mux_block{_idx}_after_image_to_token",
                batch_seq_c_to_ggml(keys),
            )
            dump_tensor(f"sam31_mux_block{_idx}_queries", batch_seq_c_to_ggml(queries))
            dump_tensor(f"sam31_mux_block{_idx}_keys", batch_seq_c_to_ggml(keys))
            return queries, keys

        layer.forward = decoder_layer_forward_wrapper

    def prepare_wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = original_prepare(*args, **kwargs)
        if isinstance(out, torch.Tensor) and out.ndim == 4:
            dump_tensor("sam31_mem_attn_output", seq_b_c_to_ggml(bchw_to_seq_b_c(out)))
        return out

    def encoder_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        src = kwargs.get("src")
        src_pos = kwargs.get("src_pos")
        if isinstance(src, torch.Tensor) and isinstance(src_pos, torch.Tensor):
            dump_tensor("sam31_mem_attn_src", seq_b_c_to_ggml(src))
            dump_tensor("sam31_mem_attn_src_pos", seq_b_c_to_ggml(src_pos))
            dump_tensor("sam31_mem_attn_input", seq_b_c_to_ggml(src + 0.1 * src_pos))
        memory = kwargs.get("memory")
        memory_pos = kwargs.get("memory_pos")
        if isinstance(memory, torch.Tensor):
            dump_tensor("sam31_mem_attn_prompt", seq_b_c_to_ggml(memory))
        if isinstance(memory_pos, torch.Tensor):
            dump_tensor("sam31_mem_attn_prompt_pos", seq_b_c_to_ggml(memory_pos))
        memory_image = kwargs.get("memory_image")
        memory_image_pos = kwargs.get("memory_image_pos")
        if isinstance(memory_image, torch.Tensor):
            dump_tensor("sam31_mem_attn_memory_image", seq_b_c_to_ggml(memory_image))
        if isinstance(memory_image_pos, torch.Tensor):
            dump_tensor("sam31_mem_attn_memory_image_pos", seq_b_c_to_ggml(memory_image_pos))
        out = original_encoder_forward(*args, **kwargs)
        if isinstance(out, dict) and isinstance(out.get("memory"), torch.Tensor):
            dump_tensor("sam31_mem_attn_output", seq_b_c_to_ggml(out["memory"]))
        return out

    def sam_transformer_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        image_embedding = args[0] if len(args) > 0 else kwargs["image_embedding"]
        image_pe = args[1] if len(args) > 1 else kwargs["image_pe"]
        point_embedding = args[2] if len(args) > 2 else kwargs["point_embedding"]

        batch_size, channels, height, width = image_embedding.shape
        del batch_size, channels
        if isinstance(image_pe, torch.Tensor) and image_pe.ndim == 4:
            dump_tensor("sam31_mux_image_pe", bchw_to_ggml_image(image_pe))
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding
        for layer in tracker.sam_mask_decoder.transformer.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        del height, width
        q = queries + point_embedding
        k = keys + image_pe
        dump_tensor("sam31_mux_final_q", batch_seq_c_to_ggml(q))
        dump_tensor("sam31_mux_final_k", batch_seq_c_to_ggml(k))
        final_attn = tracker.sam_mask_decoder.transformer.final_attn_token_to_image
        q_proj = final_attn.q_proj(q)
        k_proj = final_attn.k_proj(k)
        v_proj = final_attn.v_proj(keys)
        dump_tensor("sam31_mux_final_attn_q_proj", batch_seq_c_to_ggml(q_proj))
        dump_tensor("sam31_mux_final_attn_k_proj", batch_seq_c_to_ggml(k_proj))
        dump_tensor("sam31_mux_final_attn_v_proj", batch_seq_c_to_ggml(v_proj))
        q_heads = final_attn._separate_heads(q_proj, final_attn.num_heads)
        k_heads = final_attn._separate_heads(k_proj, final_attn.num_heads)
        v_heads = final_attn._separate_heads(v_proj, final_attn.num_heads)
        attn_core = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0)
        attn_core = final_attn._recombine_heads(attn_core)
        dump_tensor("sam31_mux_final_attn_core", batch_seq_c_to_ggml(attn_core))
        attn_out = final_attn.out_proj(attn_core)
        dump_tensor("sam31_mux_final_attn_out", batch_seq_c_to_ggml(attn_out))
        queries = queries + attn_out
        dump_tensor("sam31_mux_final_pre_norm", batch_seq_c_to_ggml(queries))
        queries = tracker.sam_mask_decoder.transformer.norm_final_attn(queries)
        dump_tensor("sam31_mux_final_queries", batch_seq_c_to_ggml(queries))
        dump_tensor("sam31_mux_final_keys", batch_seq_c_to_ggml(keys))
        return queries, keys

    def predict_masks_wrapper(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        image_embeddings = kwargs.get("image_embeddings")
        extra_per_object_embeddings = kwargs.get("extra_per_object_embeddings")
        if isinstance(image_embeddings, torch.Tensor):
            batch = image_embeddings.shape[0]
            decoder = tracker.sam_mask_decoder
            token_list = []
            if decoder.pred_obj_scores and not decoder.decode_mask_attribute_with_shared_tokens:
                token_list.append(decoder.obj_score_token.weight)
            if not decoder.decode_mask_attribute_with_shared_tokens:
                token_list.append(decoder.iou_token.weight)
            tokens = torch.cat(token_list, dim=0)
            tokens = tokens.unsqueeze(0).expand(batch, -1, -1)
            if isinstance(extra_per_object_embeddings, torch.Tensor):
                mask_tokens = decoder.mask_tokens.weight.view(
                    1,
                    decoder.multiplex_count,
                    decoder.num_mask_output_per_object,
                    -1,
                ).expand(batch, -1, -1, -1)
                mask_tokens = mask_tokens + extra_per_object_embeddings.unsqueeze(2)
                mask_tokens = mask_tokens.flatten(1, 2)
            else:
                mask_tokens = decoder.mask_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
            tokens = torch.cat([tokens, mask_tokens], dim=1)
            dump_tensor("sam31_mux_tokens_initial", batch_seq_c_to_ggml(tokens))
        out = original_predict_masks(*args, **kwargs)
        if isinstance(out.get("mask_tokens_out"), torch.Tensor):
            mask_tokens_out = out["mask_tokens_out"]
            dump_tensor("sam31_mux_mask_tokens_all", mux_tokens_to_ggml(mask_tokens_out))
            if (
                mask_tokens_out.ndim == 4
                and hasattr(tracker.sam_mask_decoder, "output_hypernetworks_mlps")
            ):
                decoder = tracker.sam_mask_decoder
                for hyper_idx, hyper_mlp in enumerate(decoder.output_hypernetworks_mlps):
                    token_idx = 0 if decoder.decode_mask_with_shared_tokens else hyper_idx
                    if token_idx >= mask_tokens_out.shape[2]:
                        continue
                    hyper = hyper_mlp(mask_tokens_out[:, :, token_idx, :])
                    dump_tensor(f"sam31_mux_hyper_{hyper_idx}", batch_seq_c_to_ggml(hyper))
        return out

    def decoder_forward_wrapper(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        out = original_decoder_forward(*args, **kwargs)
        if isinstance(out, dict) and "masks" in out and "iou_pred" in out:
            dump_tensor("sam31_mux_masks", mux_masks_to_ggml(out["masks"]))
            dump_tensor("sam31_mux_iou", mux_iou_to_ggml(out["iou_pred"]))
            if "object_score_logits" in out:
                dump_tensor("sam31_mux_obj_score", mux_iou_to_ggml(out["object_score_logits"]))
            if "sam_tokens_out" in out:
                dump_tensor("sam31_mux_mask_tokens", mux_tokens_to_ggml(out["sam_tokens_out"]))
        return out

    tracker._prepare_memory_conditioned_features = prepare_wrapper
    tracker.transformer.encoder.forward = encoder_forward_wrapper
    tracker.sam_mask_decoder.transformer.forward = sam_transformer_forward_wrapper
    tracker.sam_mask_decoder.predict_masks = predict_masks_wrapper
    tracker.sam_mask_decoder.forward = decoder_forward_wrapper
    output_upscaling = getattr(tracker.sam_mask_decoder, "output_upscaling", None)
    if output_upscaling is not None and len(output_upscaling) >= 5:
        def upscaled_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                dump_tensor("sam31_mux_upscaled", bchw_to_ggml_image(output))

        forward_hooks.append(output_upscaling[4].register_forward_hook(upscaled_hook))

    def restore() -> None:
        tracker._prepare_memory_conditioned_features = original_prepare
        tracker.transformer.encoder.forward = original_encoder_forward
        tracker.sam_mask_decoder.transformer.forward = original_sam_transformer_forward
        tracker.sam_mask_decoder.predict_masks = original_predict_masks
        tracker.sam_mask_decoder.forward = original_decoder_forward
        for layer, original_sa, original_ca, original_forward in original_layer_methods:
            layer._forward_sa = original_sa
            layer._forward_ca = original_ca
            layer.forward = original_forward
        for layer, original_forward in original_decoder_layer_methods:
            layer.forward = original_forward
        for hook in forward_hooks:
            hook.remove()

    return restore


def install_backbone_output_dump_hooks(model: Any, dump_dir: Path) -> Callable[[], None]:
    dump_dir.mkdir(parents=True, exist_ok=True)
    detector = model.detector
    original_forward_video_grounding = detector.forward_video_grounding_multigpu
    original_forward_video_grounding_batched = detector.forward_video_grounding_batched_multigpu
    active_frame_idx: int | None = None
    forward_hooks: list[Any] = []

    def dump_trunk_output(_module: Any, _inputs: Any, output: Any) -> None:
        if active_frame_idx != 1 or not isinstance(output, (list, tuple)) or not output:
            return
        tensor = getattr(output[-1], "tensors", output[-1])
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            write_raw_tensor(dump_dir, "tracking_vit_output", bchw_to_ggml_image(tensor))

    trunk = getattr(getattr(detector.backbone, "vision_backbone", None), "trunk", None)
    if trunk is not None:
        forward_hooks.append(trunk.register_forward_hook(dump_trunk_output))

    def dump_sam3_image_out(frame_idx: int | None, sam3_image_out: Any) -> None:
        if frame_idx != 1 or not isinstance(sam3_image_out, dict):
            return
        names = (
            ("sam2_backbone_fpn_0", "tracking_sam31_prop_raw_s0"),
            ("sam2_backbone_fpn_1", "tracking_sam31_prop_raw_s1"),
            ("sam2_backbone_fpn_2", "tracking_sam31_prop_raw_s2"),
        )
        for source_name, dump_name in names:
            tensor = sam3_image_out.get(source_name)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                write_raw_tensor(dump_dir, dump_name, bchw_to_ggml_image(tensor))

    def frame_idx_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
        if "frame_idx" in kwargs:
            return int(kwargs["frame_idx"])
        return None

    def forward_video_grounding_wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal active_frame_idx
        active_frame_idx = frame_idx_from_call(args, kwargs)
        try:
            out = original_forward_video_grounding(*args, **kwargs)
        finally:
            active_frame_idx = None
        if isinstance(out, tuple) and out:
            dump_sam3_image_out(frame_idx_from_call(args, kwargs), out[0])
        return out

    def forward_video_grounding_batched_wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal active_frame_idx
        active_frame_idx = frame_idx_from_call(args, kwargs)
        try:
            out = original_forward_video_grounding_batched(*args, **kwargs)
        finally:
            active_frame_idx = None
        if isinstance(out, tuple) and out:
            dump_sam3_image_out(frame_idx_from_call(args, kwargs), out[0])
        return out

    detector.forward_video_grounding_multigpu = forward_video_grounding_wrapper
    detector.forward_video_grounding_batched_multigpu = forward_video_grounding_batched_wrapper

    def restore() -> None:
        detector.forward_video_grounding_multigpu = original_forward_video_grounding
        detector.forward_video_grounding_batched_multigpu = original_forward_video_grounding_batched
        for hook in forward_hooks:
            hook.remove()

    return restore


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_match": False, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    af = a > 127
    bf = b > 127
    inter = int(np.logical_and(af, bf).sum())
    union = int(np.logical_or(af, bf).sum())
    xor = int(np.logical_xor(af, bf).sum())
    return {
        "shape_match": True,
        "intersection": inter,
        "union": union,
        "xor_pixels": xor,
        "iou": (inter / union) if union else 1.0,
    }


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


def dump_feature_cache_frame(state: dict[str, Any], frame_idx: int, dump_dir: Path) -> None:
    cached = state.get("feature_cache", {}).get(frame_idx)
    if not isinstance(cached, tuple) or len(cached) < 2:
        return
    backbone_cache = cached[1]
    if not isinstance(backbone_cache, dict):
        return
    sam2_backbone = backbone_cache.get("sam2_backbone_out")
    if not isinstance(sam2_backbone, dict):
        return
    backbone_fpn = sam2_backbone.get("backbone_fpn")
    if isinstance(backbone_fpn, list) and len(backbone_fpn) >= 3:
        names = (
            "sam31_prop_projected_s0",
            "sam31_prop_projected_s1",
            "sam31_prop_image_raw",
        )
        for name, nested in zip(names, backbone_fpn[:3], strict=False):
            tensor = getattr(nested, "tensors", nested)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                write_raw_tensor(dump_dir, name, bchw_to_ggml_image(tensor))
    vision_pos_enc = sam2_backbone.get("vision_pos_enc")
    if isinstance(vision_pos_enc, list) and vision_pos_enc:
        tensor = vision_pos_enc[-1]
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            write_raw_tensor(dump_dir, "sam31_prop_image_pe", bchw_to_ggml_image(tensor))


def dump_direct_backbone_frame(model: Any, state: dict[str, Any], frame_idx: int, dump_dir: Path) -> None:
    input_batch = state.get("input_batch")
    img_batch = getattr(input_batch, "img_batch", None)
    tensors = getattr(img_batch, "tensors", None)
    if not isinstance(tensors, torch.Tensor) or tensors.ndim != 4:
        return
    if frame_idx < 0 or frame_idx >= tensors.shape[0]:
        return

    detector = getattr(model, "detector", None)
    backbone = getattr(detector, "backbone", None)
    if backbone is None or not hasattr(backbone, "forward_image"):
        return

    frame_tensor = tensors[frame_idx : frame_idx + 1]
    trunk = getattr(getattr(backbone, "vision_backbone", None), "trunk", None)
    direct_hooks: list[Any] = []

    def dump_trunk_output(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = getattr(output[-1], "tensors", output[-1]) if isinstance(output, (list, tuple)) and output else output
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            write_raw_tensor(dump_dir, "tracking_direct_vit_output", bchw_to_ggml_image(tensor))

    if trunk is not None:
        direct_hooks.append(trunk.register_forward_hook(dump_trunk_output))
        block_indices = {0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 31}
        blocks = getattr(trunk, "blocks", [])
        for block_idx in sorted(i for i in block_indices if i < len(blocks)):
            def dump_block_output(_module: Any, _inputs: Any, output: Any, *, _idx: int = block_idx) -> None:
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    write_raw_tensor(
                        dump_dir,
                        f"tracking_direct_vit_block_{_idx:02d}_out",
                        nhwc_to_ggml_image(output),
                    )

            direct_hooks.append(blocks[block_idx].register_forward_hook(dump_block_output))

    try:
        out = backbone.forward_image(
            frame_tensor,
            need_sam3_out=False,
            need_interactive_out=False,
            need_propagation_out=True,
        )
    finally:
        for hook in direct_hooks:
            hook.remove()

    if not isinstance(out, dict):
        return
    sam2_backbone = out.get("sam2_backbone_out")
    if not isinstance(sam2_backbone, dict):
        return

    backbone_fpn = sam2_backbone.get("backbone_fpn")
    if isinstance(backbone_fpn, list) and len(backbone_fpn) >= 3:
        raw_names = (
            "tracking_direct_sam31_prop_raw_s0",
            "tracking_direct_sam31_prop_raw_s1",
            "tracking_direct_sam31_prop_raw_s2",
        )
        for name, nested in zip(raw_names, backbone_fpn[:3], strict=False):
            tensor = getattr(nested, "tensors", nested)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                write_raw_tensor(dump_dir, name, bchw_to_ggml_image(tensor))

        sam_mask_decoder = getattr(getattr(model, "tracker", None), "sam_mask_decoder", None)
        conv_s0 = getattr(sam_mask_decoder, "conv_s0", None)
        conv_s1 = getattr(sam_mask_decoder, "conv_s1", None)
        if conv_s0 is not None and conv_s1 is not None:
            tensor0 = getattr(backbone_fpn[0], "tensors", backbone_fpn[0])
            tensor1 = getattr(backbone_fpn[1], "tensors", backbone_fpn[1])
            tensor2 = getattr(backbone_fpn[2], "tensors", backbone_fpn[2])
            if (
                isinstance(tensor0, torch.Tensor)
                and tensor0.ndim == 4
                and isinstance(tensor1, torch.Tensor)
                and tensor1.ndim == 4
                and isinstance(tensor2, torch.Tensor)
                and tensor2.ndim == 4
            ):
                write_raw_tensor(
                    dump_dir,
                    "tracking_direct_sam31_prop_projected_s0",
                    bchw_to_ggml_image(conv_s0(tensor0)),
                )
                write_raw_tensor(
                    dump_dir,
                    "tracking_direct_sam31_prop_projected_s1",
                    bchw_to_ggml_image(conv_s1(tensor1)),
                )
                write_raw_tensor(
                    dump_dir,
                    "tracking_direct_sam31_prop_image_raw",
                    bchw_to_ggml_image(tensor2),
                )

    vision_pos_enc = sam2_backbone.get("vision_pos_enc")
    if isinstance(vision_pos_enc, list) and vision_pos_enc:
        tensor = vision_pos_enc[-1]
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            write_raw_tensor(dump_dir, "tracking_direct_sam31_prop_image_pe", bchw_to_ggml_image(tensor))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sam31-mask-sequence-python"))
    parser.add_argument("--cpp-mask", type=Path)
    parser.add_argument("--cpp-mask-dir", type=Path)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--mask-case",
        choices=("center", "small-center", "left-wide", "bottom-band"),
        default="center",
    )
    parser.add_argument("--frame1-offset", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--dump-propagation-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for official SAM3.1 mask sequence")
    if args.num_frames < 2:
        raise SystemExit("--num-frames must be >= 2")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    sys.path.insert(0, str(args.sam3_repo.resolve()))
    torch.set_grad_enabled(False)
    allow_tf32 = args.tf32 == "on"

    def apply_tf32_policy() -> None:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    apply_tf32_policy()

    from sam3 import build_sam3_predictor

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
    apply_tf32_policy()
    model = predictor.model

    def make_run_inputs() -> tuple[list[Image.Image], np.ndarray, float]:
        start = time.perf_counter()
        frames = [
            make_frame(args.width, args.height, args.frame1_offset * frame_idx)
            for frame_idx in range(args.num_frames)
        ]
        input_mask_np = make_rect_mask(args.width, args.height, args.mask_case)
        return frames, input_mask_np, (time.perf_counter() - start) * 1000.0

    def run_once(write_outputs: bool) -> dict[str, Any]:
        run_start = time.perf_counter()
        frames, input_mask_np, input_prepare_ms = make_run_inputs()
        start = time.perf_counter()
        state = model.init_state(resource_path=frames, offload_video_to_cpu=False, async_loading_frames=False)
        torch.cuda.synchronize()
        init_state_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        mask_t = torch.from_numpy(input_mask_np > 127).to(device="cuda", dtype=torch.float32).unsqueeze(0)
        torch.cuda.synchronize()
        mask_tensor_prepare_ms = (time.perf_counter() - start) * 1000.0

        with precision_context():
            frame0_cache_ms = run_backbone_cache(model, state, 0)
            persistent_feature_cache = dict(state["feature_cache"])

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
            for tracker_state in tracker_states:
                tracker_state["cached_features"] = persistent_feature_cache

            frame1_mask = None
            frame1_obj_score = None
            tail_cache_ms_by_frame: dict[int, float] = {}
            propagate_ms_by_frame: dict[int, float] = {}
            frame_masks: dict[int, np.ndarray] = {}
            frame_obj_scores: dict[int, float] = {}
            for target_frame_idx in range(1, args.num_frames):
                tail_cache_ms_by_frame[target_frame_idx] = run_backbone_cache(
                    model,
                    state,
                    target_frame_idx,
                )
                persistent_feature_cache.update(state["feature_cache"])
                for tracker_state in tracker_states:
                    tracker_state["cached_features"] = persistent_feature_cache
                if (
                    target_frame_idx == 1
                    and write_outputs
                    and args.dump_propagation_dir is not None
                ):
                    dump_feature_cache_frame(state, 1, args.dump_propagation_dir)
                    dump_direct_backbone_frame(model, state, 1, args.dump_propagation_dir)

                start = time.perf_counter()
                found_target = False
                for frame_idx, _obj_ids, _low_res_masks, video_res_masks, obj_scores in model.tracker.propagate_in_video(
                    tracker_states[0],
                    start_frame_idx=target_frame_idx - 1,
                    max_frame_num_to_track=1,
                    reverse=False,
                    tqdm_disable=True,
                    run_mem_encoder=True,
                ):
                    if frame_idx != target_frame_idx:
                        continue
                    raw_mask = (video_res_masks[0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
                    frame_mask = np.squeeze(raw_mask)
                    if frame_mask.ndim != 2:
                        raise RuntimeError(
                            f"unexpected official frame {frame_idx} mask shape: {list(raw_mask.shape)}"
                        )
                    frame_masks[int(frame_idx)] = frame_mask
                    frame_obj_scores[int(frame_idx)] = float(obj_scores[0].detach().float().cpu().item())
                    if frame_idx == 1:
                        frame1_mask = frame_mask
                        frame1_obj_score = frame_obj_scores[int(frame_idx)]
                    found_target = True
                    break
                torch.cuda.synchronize()
                propagate_ms_by_frame[target_frame_idx] = (time.perf_counter() - start) * 1000.0
                if not found_target:
                    raise RuntimeError(
                        f"official Python propagation did not produce frame {target_frame_idx}"
                    )
        propagate_ms = sum(propagate_ms_by_frame.values())
        run_required_e2e_ms = (time.perf_counter() - run_start) * 1000.0
        if frame1_mask is None:
            raise RuntimeError("official Python propagation did not produce frame 1")
        missing_frames = [frame_idx for frame_idx in range(1, args.num_frames) if frame_idx not in frame_masks]
        if missing_frames:
            raise RuntimeError(f"official Python propagation did not produce frames {missing_frames}")

        required_session_setup_ms = init_state_ms
        required_input_prepare_ms = input_prepare_ms + mask_tensor_prepare_ms
        tail_cache_total_ms = sum(tail_cache_ms_by_frame.values())
        tail_cache_avg_ms = tail_cache_total_ms / max(len(tail_cache_ms_by_frame), 1)
        frame1_cache_ms = tail_cache_ms_by_frame[1]
        required_model_execute_ms = frame0_cache_ms + add_mask_ms + tail_cache_total_ms + propagate_ms
        required_image_encode_or_backbone_ms = frame0_cache_ms + tail_cache_total_ms
        required_mask_init_ms = add_mask_ms
        required_propagate_encoded_ms = propagate_ms
        required_accounted_ms = (
            required_session_setup_ms + required_input_prepare_ms + required_model_execute_ms
        )

        if write_outputs:
            args.out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(input_mask_np, mode="L").save(args.out / "input_mask.png")
            Image.fromarray(frame1_mask, mode="L").save(args.out / "frame1_mask.png")
            write_pgm(args.out / "input_mask.pgm", input_mask_np)
            write_pgm(args.out / "frame1_mask.pgm", frame1_mask)
            for frame_idx, frame_mask in sorted(frame_masks.items()):
                stem = frame_mask_stem(frame_idx)
                Image.fromarray(frame_mask, mode="L").save(args.out / f"{stem}.png")
                write_pgm(args.out / f"{stem}.pgm", frame_mask)

        per_frame_results = [
            {
                "frame_index": frame_idx,
                "cache_ms": tail_cache_ms_by_frame[frame_idx],
                "propagate_ms": propagate_ms_by_frame[frame_idx],
                "mask_width": int(frame_masks[frame_idx].shape[1]),
                "mask_height": int(frame_masks[frame_idx].shape[0]),
                "mask_foreground_pixels": int((frame_masks[frame_idx] > 127).sum()),
                "mask_sha256": mask_sha256(frame_masks[frame_idx]),
                "obj_score_logit": frame_obj_scores[frame_idx],
            }
            for frame_idx in sorted(frame_masks)
        ]

        return {
            "input_prepare_ms": input_prepare_ms,
            "init_state_ms": init_state_ms,
            "mask_tensor_prepare_ms": mask_tensor_prepare_ms,
            "frame0_cache_ms": frame0_cache_ms,
            "add_mask_ms": add_mask_ms,
            "frame1_cache_ms": frame1_cache_ms,
            "tail_cache_total_ms": tail_cache_total_ms,
            "tail_cache_avg_ms": tail_cache_avg_ms,
            "propagate_ms": propagate_ms,
            "required_session_setup_ms": required_session_setup_ms,
            "required_input_prepare_ms": required_input_prepare_ms,
            "required_model_execute_ms": required_model_execute_ms,
            "required_image_encode_or_backbone_ms": required_image_encode_or_backbone_ms,
            "required_mask_init_ms": required_mask_init_ms,
            "required_propagate_encoded_ms": required_propagate_encoded_ms,
            "required_accounted_ms": required_accounted_ms,
            "required_e2e_ms": run_required_e2e_ms,
            "required_remainder_ms": run_required_e2e_ms - required_accounted_ms,
            "num_frames": args.num_frames,
            "propagated_frames": len(frame_masks),
            "per_frame_results": per_frame_results,
            "frame1_obj_score_logit": frame1_obj_score,
            "mask_width": int(frame1_mask.shape[1]),
            "mask_height": int(frame1_mask.shape[0]),
            "mask_foreground_pixels": int((frame1_mask > 127).sum()),
            "mask_sha256": mask_sha256(frame1_mask),
            "mask": frame1_mask if write_outputs else None,
            "frame_masks": frame_masks if write_outputs else None,
        }

    for _ in range(args.warmup_runs):
        _ = run_once(write_outputs=False)

    restore_dump_hooks: list[Callable[[], None]] = []
    if args.dump_propagation_dir is not None:
        restore_dump_hooks.append(install_backbone_output_dump_hooks(model, args.dump_propagation_dir))
        restore_dump_hooks.append(install_propagation_dump_hooks(model, args.dump_propagation_dir))
    try:
        result = run_once(write_outputs=True)
    finally:
        for restore in reversed(restore_dump_hooks):
            restore()
    mask = result.pop("mask")
    frame_masks = result.pop("frame_masks")
    comparison = None
    if args.cpp_mask:
        cpp = read_mask(args.cpp_mask)
        comparison = {
            "cpp_mask": str(args.cpp_mask),
            "python_mask": str(args.out / "frame1_mask.png"),
            "cpp_sha256": mask_sha256(cpp),
            "python_sha256": mask_sha256(mask),
            "mask_iou": mask_iou(cpp, mask),
        }
    if args.cpp_mask_dir:
        cpp_frame_comparisons = []
        for frame_idx in range(1, args.num_frames):
            cpp_path = args.cpp_mask_dir / f"{frame_mask_stem(frame_idx)}.png"
            if frame_idx == 1 and not cpp_path.exists():
                cpp_path = args.cpp_mask_dir / "frame1_mask.png"
            if not cpp_path.exists():
                cpp_frame_comparisons.append(
                    {
                        "frame_index": frame_idx,
                        "status": "missing_cpp_mask",
                        "cpp_mask": str(cpp_path),
                    }
                )
                continue
            cpp = read_mask(cpp_path)
            py_mask = frame_masks[frame_idx]
            item = {
                "frame_index": frame_idx,
                "status": "ok",
                "cpp_mask": str(cpp_path),
                "python_mask": str(args.out / f"{frame_mask_stem(frame_idx)}.png"),
                "cpp_sha256": mask_sha256(cpp),
                "python_sha256": mask_sha256(py_mask),
                "mask_iou": mask_iou(cpp, py_mask),
            }
            cpp_frame_comparisons.append(item)
        ok_items = [item for item in cpp_frame_comparisons if item.get("status") == "ok"]
        ious = [
            float(item["mask_iou"]["iou"])
            for item in ok_items
            if isinstance(item.get("mask_iou"), dict) and item["mask_iou"].get("shape_match")
        ]
        xors = [
            int(item["mask_iou"]["xor_pixels"])
            for item in ok_items
            if isinstance(item.get("mask_iou"), dict) and item["mask_iou"].get("shape_match")
        ]
        exact_all = bool(ok_items) and len(ok_items) == args.num_frames - 1 and all(
            item.get("cpp_sha256") == item.get("python_sha256") for item in ok_items
        )
        sequence = {
            "num_frames": args.num_frames,
            "propagated_frames": args.num_frames - 1,
            "compared_frames": len(ok_items),
            "missing_frames": [
                item["frame_index"]
                for item in cpp_frame_comparisons
                if item.get("status") != "ok"
            ],
            "min_iou": min(ious) if ious else None,
            "max_xor_pixels": max(xors) if xors else None,
            "exact_all_mask_hash_equal": exact_all,
            "frames": cpp_frame_comparisons,
        }
        if comparison is None:
            frame1 = next(
                (item for item in cpp_frame_comparisons if item.get("frame_index") == 1),
                None,
            )
            if isinstance(frame1, dict) and frame1.get("status") == "ok":
                comparison = {
                    "cpp_mask": frame1.get("cpp_mask"),
                    "python_mask": frame1.get("python_mask"),
                    "cpp_sha256": frame1.get("cpp_sha256"),
                    "python_sha256": frame1.get("python_sha256"),
                    "mask_iou": frame1.get("mask_iou"),
                }
        if comparison is None:
            comparison = {}
        comparison["sequence"] = sequence

    summary = {
        "status": "ok",
        "version": "sam3.1",
        "precision": args.dtype,
        "tf32_policy": args.tf32,
        "width": args.width,
        "height": args.height,
        "mask_case": args.mask_case,
        "frame1_offset": args.frame1_offset,
        "num_frames": args.num_frames,
        "warmup_runs": args.warmup_runs,
        "torch": {
            "version": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None,
        },
        "result": result,
        "comparison": comparison,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
