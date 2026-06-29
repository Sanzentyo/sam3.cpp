#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26,<2",
# ]
# ///
"""Compare C++ and official Python SAM3 text-detector boundary dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_shape(prefix: Path) -> tuple[int, ...]:
    return tuple(
        int(part)
        for part in prefix.with_suffix(".shape").read_text(encoding="utf-8").strip().split(",")
        if part
    )


def read_f32(prefix: Path) -> np.ndarray:
    shape = read_shape(prefix)
    values = np.fromfile(prefix.with_suffix(".bin"), dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{prefix}: expected {shape}, got {values.size} values")
    return values.reshape(shape, order="F")


def read_i32(prefix: Path) -> np.ndarray:
    shape = read_shape(prefix)
    values = np.fromfile(prefix.with_suffix(".bin"), dtype=np.int32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{prefix}: expected {shape}, got {values.size} values")
    return values.reshape(shape, order="F")


def diff_stats(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    while lhs.ndim > rhs.ndim and lhs.shape[-1] == 1:
        lhs = lhs[..., 0]
    while rhs.ndim > lhs.ndim and rhs.shape[-1] == 1:
        rhs = rhs[..., 0]
    if lhs.shape != rhs.shape:
        return {
            "lhs_shape": list(lhs.shape),
            "rhs_shape": list(rhs.shape),
            "shape_mismatch": True,
        }
    diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    return {
        "shape": list(lhs.shape),
        "total": int(diff.size),
        "equal": int(np.count_nonzero(lhs == rhs)),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "p95_abs": float(np.percentile(diff, 95)) if diff.size else 0.0,
        "p99_abs": float(np.percentile(diff, 99)) if diff.size else 0.0,
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def top_values(values: np.ndarray, limit: int) -> list[dict[str, Any]]:
    flat = values.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:limit]
    return [{"idx": int(idx), "score": float(flat[idx])} for idx in order]


def maybe_read_f32(root: Path, name: str) -> np.ndarray | None:
    prefix = root / name
    if not prefix.with_suffix(".bin").exists() or not prefix.with_suffix(".shape").exists():
        return None
    return read_f32(prefix)


def maybe_add(
    result: dict[str, Any], key: str, lhs: np.ndarray | None, rhs: np.ndarray | None
) -> None:
    if lhs is not None and rhs is not None:
        result[key] = diff_stats(lhs, rhs)


def python_nbd_to_cpp_dn(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[1] != 1:
        raise ValueError(f"expected [N,1,D], got {raw.shape}")
    return raw[:, 0, :].T


def python_nbd_to_cpp_hdnh(raw: np.ndarray, heads: int) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[1] != 1:
        raise ValueError(f"expected [N,1,D], got {raw.shape}")
    n = raw.shape[0]
    d = raw.shape[2]
    if d % heads != 0:
        raise ValueError(f"D={d} is not divisible by heads={heads}")
    head_dim = d // heads
    return raw[:, 0, :].reshape(n, heads, head_dim).transpose(2, 0, 1)


def python_nbd_to_cpp_dhw1(raw: np.ndarray) -> np.ndarray:
    dn = python_nbd_to_cpp_dn(raw)
    d, n = dn.shape
    side = int(round(n**0.5))
    if side * side != n:
        raise ValueError(f"cannot infer square spatial shape from {raw.shape}")
    return dn.reshape(d, side, side, 1, order="F")


def prompt_mask_to_cpp_bias(mask: np.ndarray) -> np.ndarray:
    flat = mask.reshape(-1).astype(bool)
    return np.where(flat, -1.0e9, 0.0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--python-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    cpp_token_ids = read_i32(args.cpp_dir / "token_ids").reshape(-1)
    py_token_ids = read_i32(args.python_dir / "token_ids").reshape(-1)
    cpp_scores = read_f32(args.cpp_dir / "ddec_class_scores").reshape(-1)
    cpp_presence_logit = float(read_f32(args.cpp_dir / "ddec_presence_logit").reshape(-1)[0])
    cpp_presence_prob = float(sigmoid(np.array([cpp_presence_logit]))[0])

    py_post_logits = maybe_read_f32(args.python_dir, "pred_logits")
    py_pre_nms_probs = maybe_read_f32(args.python_dir, "pre_nms_pred_probs")
    py_pre_nms_keep = maybe_read_f32(args.python_dir, "pre_nms_keep")

    cpp_class_probs = sigmoid(cpp_scores)
    cpp_joint_probs = cpp_class_probs * cpp_presence_prob
    result: dict[str, Any] = {
        "notes": [
            "Python pred_logits are captured after NMS suppression.",
            "Python pre_nms_pred_probs are the joint detection probabilities passed into NMS.",
            "C++ ddec_class_scores are raw class logits; cpp_joint_probs multiply by ddec_presence_logit sigmoid.",
        ],
        "token_ids_equal": bool(np.array_equal(cpp_token_ids, py_token_ids)),
        "token_ids_cpp": [int(v) for v in cpp_token_ids.tolist()],
        "token_ids_py": [int(v) for v in py_token_ids.tolist()],
        "cpp_presence_logit": cpp_presence_logit,
        "cpp_presence_prob": cpp_presence_prob,
        "cpp_top_class_probs": top_values(cpp_class_probs, args.top_k),
        "cpp_top_joint_probs": top_values(cpp_joint_probs, args.top_k),
    }

    result["text_features"] = diff_stats(
        read_f32(args.cpp_dir / "text_features"),
        read_f32(args.python_dir / "text_features"),
    )
    maybe_add(
        result,
        "input_image_preprocessed_vs_python",
        maybe_read_f32(args.cpp_dir, "input_image_preprocessed"),
        maybe_read_f32(args.python_dir, "python_input_image_preprocessed"),
    )
    py_prompt_before = maybe_read_f32(args.python_dir, "python_prompt_before_enc_raw")
    py_prompt_after = maybe_read_f32(args.python_dir, "python_prompt_after_enc_raw")
    py_prompt_mask = maybe_read_f32(args.python_dir, "python_prompt_mask_raw")
    py_encoder_hidden = maybe_read_f32(args.python_dir, "python_encoder_hidden_states_raw")
    py_encoder_pos = maybe_read_f32(args.python_dir, "python_encoder_pos_embed_raw")

    maybe_add(
        result,
        "combined_prompt_vs_python_prompt_before_enc",
        maybe_read_f32(args.cpp_dir, "combined_prompt"),
        python_nbd_to_cpp_dn(py_prompt_before) if py_prompt_before is not None else None,
    )
    maybe_add(
        result,
        "combined_prompt_vs_python_prompt_after_enc",
        maybe_read_f32(args.cpp_dir, "combined_prompt"),
        python_nbd_to_cpp_dn(py_prompt_after) if py_prompt_after is not None else None,
    )
    maybe_add(
        result,
        "combined_bias_vs_python_prompt_mask_bias",
        maybe_read_f32(args.cpp_dir, "combined_bias"),
        prompt_mask_to_cpp_bias(py_prompt_mask) if py_prompt_mask is not None else None,
    )
    maybe_add(
        result,
        "fenc_output_vs_python_encoder_hidden_states",
        maybe_read_f32(args.cpp_dir, "fenc_output"),
        python_nbd_to_cpp_dhw1(py_encoder_hidden) if py_encoder_hidden is not None else None,
    )
    maybe_add(
        result,
        "image_pos_embed_vs_python_encoder_pos_embed",
        maybe_read_f32(args.cpp_dir, "image_pos_embed"),
        python_nbd_to_cpp_dhw1(py_encoder_pos) if py_encoder_pos is not None else None,
    )
    for layer_index in range(6):
        for suffix in (
            "sa_q_proj",
            "sa_k_proj",
            "sa_v_proj",
            "sa_out",
            "ca_q_proj",
            "ca_k_proj",
            "ca_v_proj",
            "ca_out",
            "ffn_out",
        ):
            py_stage = maybe_read_f32(
                args.python_dir, f"python_fenc_layer{layer_index}_{suffix}_raw"
            )
            maybe_add(
                result,
                f"fenc_layer{layer_index}_{suffix}_vs_python",
                maybe_read_f32(args.cpp_dir, f"fenc_layer{layer_index}_{suffix}"),
                python_nbd_to_cpp_dn(py_stage) if py_stage is not None else None,
            )
        for source_suffix, packed_suffix in (
            ("sa_q_proj", "sa_q_packed"),
            ("sa_k_proj", "sa_k_packed"),
            ("sa_v_proj", "sa_v_packed"),
        ):
            py_stage = maybe_read_f32(
                args.python_dir, f"python_fenc_layer{layer_index}_{source_suffix}_raw"
            )
            maybe_add(
                result,
                f"fenc_layer{layer_index}_{packed_suffix}_vs_python",
                maybe_read_f32(args.cpp_dir, f"fenc_layer{layer_index}_{packed_suffix}"),
                python_nbd_to_cpp_hdnh(py_stage, 8) if py_stage is not None else None,
            )
        py_layer = maybe_read_f32(args.python_dir, f"python_fenc_layer{layer_index}_out_raw")
        maybe_add(
            result,
            f"fenc_layer{layer_index}_out_vs_python",
            maybe_read_f32(args.cpp_dir, f"fenc_layer{layer_index}_out"),
            python_nbd_to_cpp_dn(py_layer) if py_layer is not None else None,
        )

    cpp_ddec_queries = maybe_read_f32(args.cpp_dir, "ddec_queries")
    py_seg_obj_queries = maybe_read_f32(args.python_dir, "python_seg_obj_queries")
    if cpp_ddec_queries is not None and cpp_ddec_queries.ndim == 2 and cpp_ddec_queries.shape[1] > 1:
        maybe_add(
            result,
            "ddec_obj_queries_vs_python_seg_obj_queries",
            cpp_ddec_queries[:, 1:, None],
            py_seg_obj_queries,
        )
    for name in (
        "ddec_query_embed",
        "ddec_presence_token",
        "ddec_reference_points",
        "ddec_layer0_sa_in_proj_weight",
        "ddec_layer0_sa_in_proj_bias",
        "ddec_layer0_norm2_weight",
        "ddec_layer0_norm2_bias",
    ):
        maybe_add(
            result,
            f"{name}_vs_python",
            maybe_read_f32(args.cpp_dir, name),
            maybe_read_f32(args.python_dir, f"python_{name}"),
        )
    maybe_add(
        result,
        "ddec_normed_output_vs_python_seg_obj_queries",
        maybe_read_f32(args.cpp_dir, "ddec_normed_output"),
        py_seg_obj_queries,
    )
    for layer_index in range(6):
        py_rpb_mask = maybe_read_f32(args.python_dir, f"python_ddec_rpb_mask_{layer_index}")
        if py_rpb_mask is not None and py_rpb_mask.ndim >= 2 and py_rpb_mask.shape[1] > 1:
            py_rpb_mask = py_rpb_mask[:, 1:, ...]
        maybe_add(
            result,
            f"ddec_rpb_mask_obj_{layer_index}_vs_python",
            maybe_read_f32(args.cpp_dir, f"ddec_rpb_mask_obj_{layer_index}"),
            py_rpb_mask,
        )
        for suffix in (
            "sa_q_in",
            "sa_q_proj",
            "sa_k_proj",
            "sa_v_proj",
            "sa_out",
            "sa_pre_norm",
            "img_ca_q_in",
            "img_ca_k_in",
            "img_ca_v_in",
            "img_ca_q_proj",
            "img_ca_k_proj",
            "img_ca_v_proj",
            "img_ca_out",
            "img_ca_pre_norm",
        ):
            maybe_add(
                result,
                f"ddec_layer{layer_index}_{suffix}_vs_python",
                maybe_read_f32(args.cpp_dir, f"ddec_layer{layer_index}_{suffix}"),
                maybe_read_f32(args.python_dir, f"python_ddec_layer{layer_index}_{suffix}"),
            )
        for suffix in ("after_sa", "after_text_ca", "after_img_ca", "full_out"):
            maybe_add(
                result,
                f"ddec_layer{layer_index}_{suffix}_vs_python",
                maybe_read_f32(args.cpp_dir, f"ddec_layer{layer_index}_{suffix}"),
                maybe_read_f32(args.python_dir, f"python_ddec_layer{layer_index}_{suffix}"),
            )
    for name in (
        "seg_ca_norm",
        "seg_ca_out",
        "seg_enc_after_ca",
        "seg_pixel_decoder_out",
        "seg_instance_embed",
        "seg_mask_embed",
    ):
        maybe_add(
            result,
            f"{name}_vs_python",
            maybe_read_f32(args.cpp_dir, name),
            maybe_read_f32(args.python_dir, f"python_{name}"),
        )

    if py_pre_nms_probs is not None:
        py_pre = py_pre_nms_probs.reshape(-1)
        result["python_top_pre_nms_probs"] = top_values(py_pre, args.top_k)
        result["cpp_joint_probs_vs_python_pre_nms_probs"] = diff_stats(cpp_joint_probs, py_pre)
    if py_pre_nms_keep is not None:
        keep = py_pre_nms_keep.reshape(-1).astype(bool)
        result["python_pre_nms_keep_indices"] = [int(i) for i in np.flatnonzero(keep).tolist()]
    if py_post_logits is not None:
        py_post_probs = sigmoid(py_post_logits.reshape(-1))
        result["python_top_post_nms_probs"] = top_values(py_post_probs, args.top_k)
        result["cpp_joint_probs_vs_python_post_nms_probs"] = diff_stats(
            cpp_joint_probs, py_post_probs
        )

    py_boxes = maybe_read_f32(args.python_dir, "pred_boxes")
    if py_boxes is not None:
        result["ddec_pred_boxes_vs_python_pred_boxes"] = diff_stats(
            read_f32(args.cpp_dir / "ddec_pred_boxes"),
            np.squeeze(py_boxes, axis=0).T if py_boxes.shape[:1] == (1,) else py_boxes,
        )

    py_masks = maybe_read_f32(args.python_dir, "pred_masks")
    if py_masks is not None:
        result["seg_mask_logits_vs_python_pred_masks"] = diff_stats(
            read_f32(args.cpp_dir / "seg_mask_logits"),
            np.squeeze(py_masks, axis=0).transpose(2, 1, 0)
            if py_masks.ndim == 4 and py_masks.shape[0] == 1
            else py_masks,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
