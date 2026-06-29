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
"""Dump official SAM3 Python tracking rows in sam3_benchmark-compatible JSONL."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


FNV1A64_OFFSET = 1469598103934665603
FNV1A64_PRIME = 1099511628211


def patch_start_session(predictor: Any) -> None:
    if (
        "offload_state_to_cpu"
        in inspect.signature(predictor.model.init_state).parameters
    ):
        return

    def start_session_without_state_offload(
        self: Any,
        resource_path: str,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        del offload_state_to_cpu
        init_kwargs: dict[str, Any] = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        state = self.model.init_state(**init_kwargs)
        session_id = session_id or str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = types.MethodType(
        start_session_without_state_offload, predictor
    )


def fnv1a64(values: np.ndarray) -> str:
    h = FNV1A64_OFFSET
    for value in values.astype(np.uint8, copy=False).ravel(order="C"):
        h ^= int(value)
        h = (h * FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


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


def nchw_to_ggml_dwhb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().transpose(1, 3, 2, 0)


def nchw_to_ggml_whcb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().transpose(3, 2, 1, 0)


def selected_state_object_index(
    state_record: dict[str, Any], selected_obj_id: int | None
) -> int:
    obj_ids = list(state_record.get("obj_ids", []))
    if selected_obj_id is not None:
        for index, obj_id in enumerate(obj_ids):
            try:
                if int(obj_id) == int(selected_obj_id):
                    return index
            except (TypeError, ValueError):
                if obj_id == selected_obj_id:
                    return index
    return 0


def find_tracker_cond_output(
    state_record: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    candidates: list[tuple[dict[str, Any], str]] = [(state_record, "root")]
    for state_key in ("sam2_inference_states", "tracker_inference_states"):
        for index, sub_state in enumerate(state_record.get(state_key, [])):
            if isinstance(sub_state, dict):
                candidates.append((sub_state, f"{state_key}[{index}]"))
    for candidate, label in candidates:
        output_dict = candidate.get("output_dict", {})
        if not isinstance(output_dict, dict):
            continue
        cond_outputs = output_dict.get("cond_frame_outputs", {})
        if not isinstance(cond_outputs, dict):
            continue
        cond = cond_outputs.get(0)
        if isinstance(cond, dict):
            return candidate, cond, label
    return None, None, "missing"


def tensor_shape(value: Any) -> list[int] | None:
    if torch.is_tensor(value):
        return [int(dim) for dim in value.shape]
    return None


def jsonable_obj_ids(value: Any) -> list[Any]:
    result: list[Any] = []
    for item in value:
        if isinstance(item, np.generic):
            result.append(item.item())
        else:
            result.append(item)
    return result


def write_tracker_state_meta(
    tensor_dir: Path,
    *,
    root_state: dict[str, Any],
    source: str,
    state: dict[str, Any] | None,
    cond: dict[str, Any] | None,
    selected_obj_id: int | None,
    obj_idx: int,
) -> None:
    meta = {
        "source": source,
        "selected_obj_id": selected_obj_id,
        "selected_obj_idx": obj_idx,
        "root_state_keys": sorted(root_state.keys()),
        "state_keys": sorted(state.keys()) if state is not None else [],
        "state_obj_ids": jsonable_obj_ids(state.get("obj_ids", []))
        if state is not None
        else [],
        "cond_keys": sorted(cond.keys()) if cond is not None else [],
        "cond_tensor_shapes": {
            key: shape
            for key, value in (cond or {}).items()
            if (shape := tensor_shape(value)) is not None
        },
    }
    if cond is not None:
        pos = cond.get("maskmem_pos_enc")
        if isinstance(pos, (list, tuple)):
            meta["maskmem_pos_enc_shapes"] = [
                tensor_shape(item) for item in pos if torch.is_tensor(item)
            ]
    tensor_dir.mkdir(parents=True, exist_ok=True)
    (tensor_dir / "tracker_state_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_nchw_object(tensor: torch.Tensor, obj_idx: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] <= obj_idx:
        obj_idx = 0
    return tensor[obj_idx : obj_idx + 1]


def dump_tracker_state_tensors(
    tensor_dir: Path,
    model: Any,
    state_record: dict[str, Any],
    selected_obj_id: int | None,
) -> None:
    state, cond, source = find_tracker_cond_output(state_record)
    obj_idx = selected_state_object_index(state or state_record, selected_obj_id)
    write_tracker_state_meta(
        tensor_dir,
        root_state=state_record,
        source=source,
        state=state,
        cond=cond,
        selected_obj_id=selected_obj_id,
        obj_idx=obj_idx,
    )
    if state is None or cond is None:
        return

    obj_ptr = cond.get("obj_ptr")
    if torch.is_tensor(obj_ptr) and obj_ptr.ndim == 2 and obj_ptr.shape[0] > 0:
        ptr_idx = min(obj_idx, int(obj_ptr.shape[0]) - 1)
        write_f32_tensor(
            tensor_dir / "obj_ptr",
            obj_ptr[ptr_idx : ptr_idx + 1].detach().float().cpu().numpy().T,
        )

    object_score_logits = cond.get("object_score_logits")
    if (
        torch.is_tensor(object_score_logits)
        and object_score_logits.ndim >= 1
        and object_score_logits.shape[0] > 0
    ):
        score_idx = min(obj_idx, int(object_score_logits.shape[0]) - 1)
        write_f32_tensor(
            tensor_dir / "object_score_logits",
            object_score_logits[score_idx : score_idx + 1]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(1, -1),
        )

    maskmem_features = cond.get("maskmem_features")
    if torch.is_tensor(maskmem_features) and maskmem_features.ndim == 4:
        write_f32_tensor(
            tensor_dir / "stored_spatial_feats",
            nchw_to_ggml_dwhb(select_nchw_object(maskmem_features, obj_idx)),
        )
    maskmem_pos_enc = cond.get("maskmem_pos_enc")
    if isinstance(maskmem_pos_enc, (list, tuple)) and maskmem_pos_enc:
        pos = maskmem_pos_enc[-1]
        if torch.is_tensor(pos) and pos.ndim == 4:
            write_f32_tensor(
                tensor_dir / "stored_spatial_pe",
                nchw_to_ggml_dwhb(select_nchw_object(pos, obj_idx)),
            )

    captures = detector_captures(model)
    memory_mask_input = captures.get("memory_mask_input")
    if torch.is_tensor(memory_mask_input) and memory_mask_input.ndim == 4:
        write_f32_tensor(
            tensor_dir / "memory_mask_input",
            nchw_to_ggml_whcb(select_nchw_object(memory_mask_input, obj_idx)),
        )

    memory_pix_feat = captures.get("memory_pix_feat")
    if torch.is_tensor(memory_pix_feat) and memory_pix_feat.ndim == 4:
        write_f32_tensor(
            tensor_dir / "image_features",
            nchw_to_ggml_dwhb(select_nchw_object(memory_pix_feat, obj_idx)),
        )


def dump_detector_tensors(
    tensor_dir: Path, model: Any, feature_cache: dict[str, Any]
) -> None:
    text_cache = feature_cache.get("text", {})
    if text_cache:
        _key, text_outputs = next(iter(text_cache.items()))
        text_memory = text_outputs.get("language_features")
        if text_memory is not None:
            # Official shape is [L, B, D]; C++ dump uses [D, L].
            text = text_memory[:, 0, :].detach().float().cpu().numpy().T
            write_f32_tensor(tensor_dir / "text_features", text)
        text_attention_mask = text_outputs.get("language_mask")
        if text_attention_mask is not None:
            write_f32_tensor(
                tensor_dir / "text_attention_mask",
                text_attention_mask.detach().float().cpu().numpy().reshape(-1),
            )

    captures = getattr(model, "_codex_sam3_detector_captures", {})
    for name, tensor in captures.items():
        write_f32_tensor(tensor_dir / name, tensor.detach().float().cpu().numpy())


def detector_captures(model: Any) -> dict[str, torch.Tensor]:
    captures = getattr(model, "_codex_sam3_detector_captures", None)
    if captures is None:
        captures = {}
        model._codex_sam3_detector_captures = captures
    return captures


def read_image_size(frame_dir: Path) -> tuple[int, int]:
    first = next(iter(sorted(frame_dir.glob("*.jpg"))), None)
    if first is None:
        first = next(iter(sorted(frame_dir.glob("*.png"))), None)
    if first is None:
        raise RuntimeError(f"no frames found in {frame_dir}")
    with Image.open(first) as image:
        return image.size


def xywh_to_xyxy(box: np.ndarray, width: int, height: int) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    x0 = x * width
    y0 = y * height
    x1 = (x + w) * width
    y1 = (y + h) * height
    return [x0, y0, x1, y1]


def selected_index(
    outputs: dict[str, Any], initial_obj_id: int | None = None
) -> int | None:
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    if obj_ids.size == 0:
        return None
    if initial_obj_id is not None:
        matches = np.flatnonzero(obj_ids == initial_obj_id)
        if matches.size:
            return int(matches[0])
    probs = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
    if probs.size == obj_ids.size:
        return int(np.argmax(probs))
    return 0


def row_from_outputs(
    outputs: dict[str, Any],
    *,
    offset: int,
    frame_index: int,
    width: int,
    height: int,
    mask_dir: Path | None,
    initial_obj_id: int | None = None,
) -> tuple[dict[str, Any], int | None]:
    idx = selected_index(outputs, initial_obj_id)
    if idx is None:
        return (
            {
                "offset": offset,
                "expected_frame_index": frame_index,
                "bbox_xyxy": None,
                "score": 0,
                "mask_area": 0,
                "mask_fnv1a64": "0000000000000000",
                "mask_path": None,
                "source": "official-python-sam3-missing",
            },
            initial_obj_id,
        )

    obj_ids = np.asarray(outputs["out_obj_ids"])
    probs = np.asarray(outputs["out_probs"], dtype=np.float32)
    boxes = np.asarray(outputs["out_boxes_xywh"], dtype=np.float32)
    masks = np.asarray(outputs["out_binary_masks"])
    mask_u8 = np.where(masks[idx], 255, 0).astype(np.uint8)
    mask_path = None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        path = mask_dir / f"frame_{offset:05d}.png"
        Image.fromarray(mask_u8).save(path)
        mask_path = str(path)
    row = {
        "offset": offset,
        "expected_frame_index": frame_index,
        "bbox_xyxy": [round(v, 3) for v in xywh_to_xyxy(boxes[idx], width, height)],
        "score": round(float(probs[idx]), 6),
        "selected_mask_index": int(idx),
        "decoder_iou_scores": [round(float(v), 6) for v in probs.tolist()],
        "decoder_lowres_mask_areas": [],
        "mask_area": int(np.count_nonzero(mask_u8 > 127)),
        "mask_fnv1a64": fnv1a64(mask_u8),
        "mask_path": mask_path,
        "source": "official-python-sam3",
    }
    return row, int(obj_ids[idx])


def candidate_rows_from_outputs(
    outputs: dict[str, Any],
    *,
    width: int,
    height: int,
    selected_obj_id: int | None,
) -> list[dict[str, Any]]:
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    probs = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
    boxes = np.asarray(outputs.get("out_boxes_xywh", []), dtype=np.float32)
    masks = np.asarray(outputs.get("out_binary_masks", []))
    rows: list[dict[str, Any]] = []
    for idx in range(int(obj_ids.size)):
        mask_u8 = np.where(masks[idx], 255, 0).astype(np.uint8)
        rows.append(
            {
                "source": "official-python-sam3-initial-candidate",
                "frame_index": 0,
                "candidate_index": idx,
                "selected": selected_obj_id is not None
                and int(obj_ids[idx]) == selected_obj_id,
                "instance_id": int(obj_ids[idx]),
                "bbox_xyxy": [
                    round(v, 3) for v in xywh_to_xyxy(boxes[idx], width, height)
                ],
                "score": round(float(probs[idx]), 6),
                "mask_area": int(np.count_nonzero(mask_u8 > 127)),
                "mask_fnv1a64": fnv1a64(mask_u8),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-candidates-jsonl", type=Path)
    parser.add_argument("--out-mask-dir", type=Path)
    parser.add_argument("--out-tensor-dir", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tf32", choices=["on", "off"], default="on")
    parser.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3")
    args = parser.parse_args()

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

    width, height = read_image_size(args.frame_dir)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with autocast, torch.inference_mode():
        predictor = build_sam3_predictor(
            version=args.version,
            compile=False,
            use_fa3=False,
            async_loading_frames=False,
        )
        patch_start_session(predictor)
        if args.out_tensor_dir is not None:
            try:
                tokenized = (
                    predictor.model.detector.backbone.language_backbone.tokenizer(
                        [args.prompt], context_length=32
                    )
                )
                write_i32_tensor(
                    args.out_tensor_dir / "token_ids", tokenized.cpu().numpy()[0]
                )
            except Exception:
                pass

        if args.out_tensor_dir is not None:
            import sam3.model.sam3_image as sam3_image_module

            original_nms_masks = sam3_image_module.nms_masks
            original_forward_image = predictor.model.detector.backbone.forward_image
            original_forward_grounding = predictor.model.detector.forward_grounding

            def capture_bchw_list(prefix: str, values: Any) -> None:
                captures = detector_captures(predictor.model)
                if not isinstance(values, (list, tuple)):
                    return
                for index, value in enumerate(values):
                    if (
                        torch.is_tensor(value)
                        and value.ndim == 4
                        and f"{prefix}_{index}" not in captures
                    ):
                        captures[f"{prefix}_{index}"] = (
                            value[0].detach().float().unsqueeze(-1).clone()
                        )

            def forward_image_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
                result = original_forward_image(*call_args, **call_kwargs)
                if isinstance(result, dict):
                    capture_bchw_list("sam3_backbone_fpn", result.get("backbone_fpn"))
                    capture_bchw_list("sam3_vision_pos_enc", result.get("vision_pos_enc"))
                return result

            predictor.model.detector.backbone.forward_image = forward_image_wrapper

            hook_handles = []
            encoder_layers = getattr(
                getattr(predictor.model.detector.transformer, "encoder", None),
                "layers",
                [],
            )

            def make_encoder_layer_hook(index: int):
                def encoder_layer_hook(_module: Any, _inputs: Any, output: Any) -> None:
                    captures = detector_captures(predictor.model)
                    key = f"python_fenc_layer{index}_out_raw"
                    if torch.is_tensor(output) and output.ndim == 3 and key not in captures:
                        if output.shape[0] == 1:
                            captures[key] = output[0].detach().float().unsqueeze(1).clone()
                        elif output.shape[1] == 1:
                            captures[key] = output.detach().float().clone()

                return encoder_layer_hook

            def capture_encoder_bnd(index: int, suffix: str, output: Any) -> None:
                captures = detector_captures(predictor.model)
                key = f"python_fenc_layer{index}_{suffix}_raw"
                tensor = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(tensor) and tensor.ndim == 3 and key not in captures:
                    if tensor.shape[0] == 1:
                        captures[key] = tensor[0].detach().float().unsqueeze(1).clone()
                    elif tensor.shape[1] == 1:
                        captures[key] = tensor.detach().float().clone()

            def make_encoder_module_hook(index: int, suffix: str):
                def encoder_module_hook(_module: Any, _inputs: Any, output: Any) -> None:
                    capture_encoder_bnd(index, suffix, output)

                return encoder_module_hook

            def capture_encoder_nbd_tensor(index: int, suffix: str, tensor: torch.Tensor) -> None:
                captures = detector_captures(predictor.model)
                key = f"python_fenc_layer{index}_{suffix}_raw"
                if key in captures or tensor.ndim != 3:
                    return
                if tensor.shape[0] == 1:
                    captures[key] = tensor[0].detach().float().unsqueeze(1).clone()
                elif tensor.shape[1] == 1:
                    captures[key] = tensor.detach().float().clone()

            def make_mha_projection_hook(index: int, prefix: str):
                def mha_projection_hook(
                    module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]
                ) -> None:
                    query = inputs[0] if len(inputs) > 0 else kwargs.get("query")
                    key = inputs[1] if len(inputs) > 1 else kwargs.get("key")
                    value = inputs[2] if len(inputs) > 2 else kwargs.get("value")
                    if not (
                        torch.is_tensor(query)
                        and torch.is_tensor(key)
                        and torch.is_tensor(value)
                    ):
                        return

                    bias = getattr(module, "in_proj_bias", None)
                    embed_dim = int(getattr(module, "embed_dim", query.shape[-1]))
                    in_proj_weight = getattr(module, "in_proj_weight", None)
                    if torch.is_tensor(in_proj_weight):
                        weight = in_proj_weight.detach().float()
                        bias_f = bias.detach().float() if torch.is_tensor(bias) else None
                        q_bias = bias_f[:embed_dim] if bias_f is not None else None
                        k_bias = bias_f[embed_dim : 2 * embed_dim] if bias_f is not None else None
                        v_bias = bias_f[2 * embed_dim :] if bias_f is not None else None
                        q_proj = torch.nn.functional.linear(
                            query.detach().float(), weight[:embed_dim], q_bias
                        )
                        k_proj = torch.nn.functional.linear(
                            key.detach().float(), weight[embed_dim : 2 * embed_dim], k_bias
                        )
                        v_proj = torch.nn.functional.linear(
                            value.detach().float(), weight[2 * embed_dim :], v_bias
                        )
                    else:
                        q_weight = getattr(module, "q_proj_weight", None)
                        k_weight = getattr(module, "k_proj_weight", None)
                        v_weight = getattr(module, "v_proj_weight", None)
                        if not (
                            torch.is_tensor(q_weight)
                            and torch.is_tensor(k_weight)
                            and torch.is_tensor(v_weight)
                        ):
                            return
                        bias_f = bias.detach().float() if torch.is_tensor(bias) else None
                        q_bias = bias_f[:embed_dim] if bias_f is not None else None
                        k_bias = bias_f[embed_dim : 2 * embed_dim] if bias_f is not None else None
                        v_bias = bias_f[2 * embed_dim :] if bias_f is not None else None
                        q_proj = torch.nn.functional.linear(
                            query.detach().float(), q_weight.detach().float(), q_bias
                        )
                        k_proj = torch.nn.functional.linear(
                            key.detach().float(), k_weight.detach().float(), k_bias
                        )
                        v_proj = torch.nn.functional.linear(
                            value.detach().float(), v_weight.detach().float(), v_bias
                        )

                    capture_encoder_nbd_tensor(index, f"{prefix}_q_proj", q_proj)
                    capture_encoder_nbd_tensor(index, f"{prefix}_k_proj", k_proj)
                    capture_encoder_nbd_tensor(index, f"{prefix}_v_proj", v_proj)

                return mha_projection_hook

            for layer_index, layer in enumerate(encoder_layers):
                captures = detector_captures(predictor.model)
                out_proj = getattr(layer.self_attn, "out_proj", None)
                if out_proj is not None:
                    weight = getattr(out_proj, "weight", None)
                    bias = getattr(out_proj, "bias", None)
                    if torch.is_tensor(weight):
                        captures[f"python_fenc_layer{layer_index}_sa_out_proj_weight_raw"] = (
                            weight.detach().float().clone()
                        )
                    if torch.is_tensor(bias):
                        captures[f"python_fenc_layer{layer_index}_sa_out_proj_bias_raw"] = (
                            bias.detach().float().clone()
                        )
                hook_handles.append(layer.register_forward_hook(make_encoder_layer_hook(layer_index)))
                hook_handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        make_mha_projection_hook(layer_index, "sa"), with_kwargs=True
                    )
                )
                hook_handles.append(
                    layer.cross_attn_image.register_forward_pre_hook(
                        make_mha_projection_hook(layer_index, "ca"), with_kwargs=True
                    )
                )
                hook_handles.append(
                    layer.self_attn.register_forward_hook(
                        make_encoder_module_hook(layer_index, "sa_out")
                    )
                )
                hook_handles.append(
                    layer.cross_attn_image.register_forward_hook(
                        make_encoder_module_hook(layer_index, "ca_out")
                    )
                )
                hook_handles.append(
                    layer.linear2.register_forward_hook(
                        make_encoder_module_hook(layer_index, "ffn_out")
                    )
                )

            def capture_once(name: str, tensor: torch.Tensor) -> None:
                captures = detector_captures(predictor.model)
                if name not in captures:
                    captures[name] = tensor.detach().float().contiguous().clone()

            def capture_nbd_as_dnb(name: str, tensor: Any) -> None:
                if torch.is_tensor(tensor) and tensor.ndim == 3:
                    capture_once(name, tensor.permute(2, 0, 1))

            def capture_bqd_as_dqb(name: str, tensor: Any) -> None:
                if torch.is_tensor(tensor) and tensor.ndim == 3:
                    capture_once(name, tensor.permute(2, 1, 0))

            def capture_bchw_as_dwhb(name: str, tensor: Any) -> None:
                if torch.is_tensor(tensor) and tensor.ndim == 4:
                    capture_once(name, tensor.permute(1, 3, 2, 0))

            def capture_attn_mask_as_kqhb(name: str, tensor: Any, heads: int) -> None:
                if not torch.is_tensor(tensor) or tensor.ndim != 3 or heads <= 0:
                    return
                if tensor.shape[0] % heads != 0:
                    return
                batch = tensor.shape[0] // heads
                mask = tensor.detach().float().reshape(
                    batch, heads, tensor.shape[1], tensor.shape[2]
                )
                capture_once(name, mask.permute(3, 2, 1, 0))

            decoder = getattr(predictor.model.detector.transformer, "decoder", None)
            decoder_layers = getattr(decoder, "layers", [])
            if decoder is not None:
                query_embed = getattr(decoder, "query_embed", None)
                query_weight = getattr(query_embed, "weight", None)
                if torch.is_tensor(query_weight):
                    capture_once("python_ddec_query_embed", query_weight.T)
                presence_token = getattr(decoder, "presence_token", None)
                presence_weight = getattr(presence_token, "weight", None)
                if torch.is_tensor(presence_weight):
                    capture_once("python_ddec_presence_token", presence_weight.T)
                reference_points = getattr(decoder, "reference_points", None)
                reference_weight = getattr(reference_points, "weight", None)
                if torch.is_tensor(reference_weight):
                    capture_once("python_ddec_reference_points", reference_weight.T)
                if len(decoder_layers) > 0:
                    layer0 = decoder_layers[0]
                    self_attn = getattr(layer0, "self_attn", None)
                    in_proj_weight = getattr(self_attn, "in_proj_weight", None)
                    in_proj_bias = getattr(self_attn, "in_proj_bias", None)
                    if torch.is_tensor(in_proj_weight):
                        capture_once(
                            "python_ddec_layer0_sa_in_proj_weight",
                            in_proj_weight.T,
                        )
                    if torch.is_tensor(in_proj_bias):
                        capture_once("python_ddec_layer0_sa_in_proj_bias", in_proj_bias)
                    norm2 = getattr(layer0, "norm2", None)
                    norm2_weight = getattr(norm2, "weight", None)
                    norm2_bias = getattr(norm2, "bias", None)
                    if torch.is_tensor(norm2_weight):
                        capture_once("python_ddec_layer0_norm2_weight", norm2_weight)
                    if torch.is_tensor(norm2_bias):
                        capture_once("python_ddec_layer0_norm2_bias", norm2_bias)

            def make_decoder_mha_pre_hook(index: int, prefix: str):
                def decoder_mha_pre_hook(
                    module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]
                ) -> None:
                    query = inputs[0] if len(inputs) > 0 else kwargs.get("query")
                    key = inputs[1] if len(inputs) > 1 else kwargs.get("key")
                    value = inputs[2] if len(inputs) > 2 else kwargs.get("value")
                    if not (
                        torch.is_tensor(query)
                        and torch.is_tensor(key)
                        and torch.is_tensor(value)
                    ):
                        return
                    if prefix == "img_ca":
                        capture_attn_mask_as_kqhb(
                            f"python_ddec_rpb_mask_{index}",
                            kwargs.get("attn_mask"),
                            int(getattr(module, "num_heads", 0)),
                        )
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{prefix}_q_in", query)
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{prefix}_k_in", key)
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{prefix}_v_in", value)

                    weight = getattr(module, "in_proj_weight", None)
                    bias = getattr(module, "in_proj_bias", None)
                    if not torch.is_tensor(weight):
                        return
                    embed_dim = int(getattr(module, "embed_dim", query.shape[-1]))
                    bias_f = bias.detach().float() if torch.is_tensor(bias) else None
                    q_bias = bias_f[:embed_dim] if bias_f is not None else None
                    k_bias = bias_f[embed_dim : 2 * embed_dim] if bias_f is not None else None
                    v_bias = bias_f[2 * embed_dim :] if bias_f is not None else None
                    weight_f = weight.detach().float()
                    q_proj = torch.nn.functional.linear(
                        query.detach().float(), weight_f[:embed_dim], q_bias
                    )
                    k_proj = torch.nn.functional.linear(
                        key.detach().float(),
                        weight_f[embed_dim : 2 * embed_dim],
                        k_bias,
                    )
                    v_proj = torch.nn.functional.linear(
                        value.detach().float(),
                        weight_f[2 * embed_dim :],
                        v_bias,
                    )
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{prefix}_q_proj", q_proj)
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{prefix}_k_proj", k_proj)
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{prefix}_v_proj", v_proj)

                return decoder_mha_pre_hook

            def make_decoder_mha_out_hook(index: int, prefix: str):
                def decoder_mha_out_hook(_module: Any, _inputs: Any, output: Any) -> None:
                    attn_out = output[0] if isinstance(output, tuple) else output
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{prefix}_out", attn_out)

                return decoder_mha_out_hook

            def make_decoder_norm_pre_hook(index: int, suffix: str):
                def decoder_norm_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    tensor = inputs[0] if inputs else None
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{suffix}", tensor)

                return decoder_norm_pre_hook

            def make_decoder_norm_hook(index: int, suffix: str):
                def decoder_norm_hook(
                    _module: Any, _inputs: Any, output: Any
                ) -> None:
                    capture_nbd_as_dnb(f"python_ddec_layer{index}_{suffix}", output)

                return decoder_norm_hook

            for layer_index, layer in enumerate(decoder_layers):
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None:
                    hook_handles.append(
                        self_attn.register_forward_pre_hook(
                            make_decoder_mha_pre_hook(layer_index, "sa"), with_kwargs=True
                        )
                    )
                    hook_handles.append(
                        self_attn.register_forward_hook(
                            make_decoder_mha_out_hook(layer_index, "sa")
                        )
                    )
                norm2 = getattr(layer, "norm2", None)
                if norm2 is not None:
                    hook_handles.append(
                        norm2.register_forward_pre_hook(
                            make_decoder_norm_pre_hook(layer_index, "sa_pre_norm")
                        )
                    )
                cross_attn = getattr(layer, "cross_attn", None)
                if cross_attn is not None:
                    hook_handles.append(
                        cross_attn.register_forward_pre_hook(
                            make_decoder_mha_pre_hook(layer_index, "img_ca"),
                            with_kwargs=True,
                        )
                    )
                    hook_handles.append(
                        cross_attn.register_forward_hook(
                            make_decoder_mha_out_hook(layer_index, "img_ca")
                        )
                    )
                norm1 = getattr(layer, "norm1", None)
                if norm1 is not None:
                    hook_handles.append(
                        norm1.register_forward_pre_hook(
                            make_decoder_norm_pre_hook(layer_index, "img_ca_pre_norm")
                        )
                    )
                for attr, suffix in (
                    ("norm2", "after_sa"),
                    ("catext_norm", "after_text_ca"),
                    ("norm1", "after_img_ca"),
                    ("norm3", "full_out"),
                ):
                    module = getattr(layer, attr, None)
                    if module is not None:
                        hook_handles.append(
                            module.register_forward_hook(
                                make_decoder_norm_hook(layer_index, suffix)
                            )
                        )

            seg_encoder_hidden_for_residual: torch.Tensor | None = None
            seg_head = getattr(predictor.model.detector, "segmentation_head", None)
            if seg_head is not None:

                def seg_head_pre_hook(
                    _module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]
                ) -> None:
                    nonlocal seg_encoder_hidden_for_residual
                    obj_queries = kwargs.get(
                        "obj_queries", inputs[1] if len(inputs) > 1 else None
                    )
                    encoder_hidden_states = kwargs.get(
                        "encoder_hidden_states",
                        inputs[3] if len(inputs) > 3 else None,
                    )
                    if torch.is_tensor(encoder_hidden_states):
                        seg_encoder_hidden_for_residual = (
                            encoder_hidden_states.detach().float().clone()
                        )
                        capture_nbd_as_dnb(
                            "python_seg_encoder_hidden_input",
                            seg_encoder_hidden_for_residual,
                        )
                    if torch.is_tensor(obj_queries):
                        obj_queries_last = (
                            obj_queries[-1] if obj_queries.ndim == 4 else obj_queries
                        )
                        capture_bqd_as_dqb("python_seg_obj_queries", obj_queries_last)

                hook_handles.append(
                    seg_head.register_forward_pre_hook(
                        seg_head_pre_hook, with_kwargs=True
                    )
                )

                cross_attn_norm = getattr(seg_head, "cross_attn_norm", None)
                if cross_attn_norm is not None:

                    def seg_ca_norm_hook(
                        _module: Any, _inputs: Any, output: Any
                    ) -> None:
                        capture_nbd_as_dnb("python_seg_ca_norm", output)

                    hook_handles.append(
                        cross_attn_norm.register_forward_hook(seg_ca_norm_hook)
                    )

                cross_attend_prompt = getattr(seg_head, "cross_attend_prompt", None)
                if cross_attend_prompt is not None:

                    def seg_ca_hook(
                        _module: Any,
                        _inputs: tuple[Any, ...],
                        _kwargs: dict[str, Any],
                        output: Any,
                    ) -> None:
                        nonlocal seg_encoder_hidden_for_residual
                        attn_out = output[0] if isinstance(output, tuple) else output
                        if not torch.is_tensor(attn_out):
                            return
                        capture_nbd_as_dnb("python_seg_ca_out", attn_out)
                        if (
                            seg_encoder_hidden_for_residual is not None
                            and seg_encoder_hidden_for_residual.shape == attn_out.shape
                        ):
                            capture_nbd_as_dnb(
                                "python_seg_enc_after_ca",
                                seg_encoder_hidden_for_residual + attn_out.detach().float(),
                            )

                    hook_handles.append(
                        cross_attend_prompt.register_forward_hook(
                            seg_ca_hook, with_kwargs=True
                        )
                    )

                pixel_decoder = getattr(seg_head, "pixel_decoder", None)
                if pixel_decoder is not None:

                    def seg_pixel_decoder_hook(
                        _module: Any, _inputs: Any, output: Any
                    ) -> None:
                        capture_bchw_as_dwhb("python_seg_pixel_decoder_out", output)

                    hook_handles.append(
                        pixel_decoder.register_forward_hook(seg_pixel_decoder_hook)
                    )

                instance_seg_head = getattr(seg_head, "instance_seg_head", None)
                if instance_seg_head is not None:

                    def seg_instance_hook(
                        _module: Any, _inputs: Any, output: Any
                    ) -> None:
                        capture_bchw_as_dwhb("python_seg_instance_embed", output)

                    hook_handles.append(
                        instance_seg_head.register_forward_hook(seg_instance_hook)
                    )

                mask_predictor = getattr(seg_head, "mask_predictor", None)
                mask_embed = getattr(mask_predictor, "mask_embed", None)
                if mask_embed is not None:

                    def seg_mask_embed_hook(
                        _module: Any, _inputs: Any, output: Any
                    ) -> None:
                        capture_bqd_as_dqb("python_seg_mask_embed", output)

                    hook_handles.append(
                        mask_embed.register_forward_hook(seg_mask_embed_hook)
                    )

            def forward_grounding_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
                result = original_forward_grounding(*call_args, **call_kwargs)
                if isinstance(result, dict):
                    captures = detector_captures(predictor.model)
                    encoder_out = result.get("prev_encoder_out", {}).get("encoder_out", {})
                    for key, value in (
                        ("python_encoder_hidden_states_raw", encoder_out.get("encoder_hidden_states")),
                        ("python_encoder_pos_embed_raw", encoder_out.get("pos_embed")),
                        ("python_prompt_before_enc_raw", encoder_out.get("prompt_before_enc")),
                        ("python_prompt_after_enc_raw", encoder_out.get("prompt_after_enc")),
                        ("python_prompt_mask_raw", encoder_out.get("prompt_mask")),
                    ):
                        if torch.is_tensor(value) and key not in captures:
                            captures[key] = value.detach().float().clone()
                return result

            predictor.model.detector.forward_grounding = forward_grounding_wrapper

            def nms_masks_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
                pred_probs = call_kwargs.get(
                    "pred_probs", call_args[0] if call_args else None
                )
                pred_masks = call_kwargs.get(
                    "pred_masks", call_args[1] if len(call_args) > 1 else None
                )
                captures = detector_captures(predictor.model)
                if torch.is_tensor(pred_probs) and "pre_nms_pred_probs" not in captures:
                    captures["pre_nms_pred_probs"] = pred_probs.detach().clone()
                if torch.is_tensor(pred_masks) and "pre_nms_pred_masks" not in captures:
                    captures["pre_nms_pred_masks"] = pred_masks.detach().clone()
                keep = original_nms_masks(*call_args, **call_kwargs)
                if torch.is_tensor(keep) and "pre_nms_keep" not in captures:
                    captures["pre_nms_keep"] = keep.detach().clone()
                return keep

            sam3_image_module.nms_masks = nms_masks_wrapper

            tracker_model = getattr(predictor.model, "tracker", None)
            maskmem_backbone = getattr(tracker_model, "maskmem_backbone", None)
            if maskmem_backbone is not None:
                original_maskmem_forward = maskmem_backbone.forward

                def maskmem_forward_wrapper(
                    *call_args: Any, **call_kwargs: Any
                ) -> Any:
                    if call_args and torch.is_tensor(call_args[0]):
                        captures = detector_captures(predictor.model)
                        if "memory_pix_feat" not in captures:
                            captures["memory_pix_feat"] = (
                                call_args[0].detach().float().clone()
                            )
                    return original_maskmem_forward(*call_args, **call_kwargs)

                maskmem_backbone.forward = maskmem_forward_wrapper

                mask_downsampler = getattr(maskmem_backbone, "mask_downsampler", None)
                if mask_downsampler is not None:
                    original_mask_downsampler_forward = mask_downsampler.forward

                    def mask_downsampler_forward_wrapper(
                        x: torch.Tensor, *call_args: Any, **call_kwargs: Any
                    ) -> Any:
                        captures = detector_captures(predictor.model)
                        if "memory_mask_input" not in captures and torch.is_tensor(x):
                            memory_mask_input = x.detach().float()
                            interpol_size = getattr(
                                mask_downsampler, "interpol_size", None
                            )
                            if (
                                interpol_size is not None
                                and list(memory_mask_input.shape[-2:])
                                != list(interpol_size)
                            ):
                                memory_mask_input = F.interpolate(
                                    memory_mask_input,
                                    size=interpol_size,
                                    align_corners=False,
                                    mode="bilinear",
                                    antialias=True,
                                )
                            captures["memory_mask_input"] = memory_mask_input.clone()
                        return original_mask_downsampler_forward(
                            x, *call_args, **call_kwargs
                        )

                    mask_downsampler.forward = mask_downsampler_forward_wrapper

        original_forward_video_grounding = (
            predictor.model.detector.forward_video_grounding_multigpu
        )

        def forward_video_grounding_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
            result = original_forward_video_grounding(*call_args, **call_kwargs)
            image_out = result[0] if isinstance(result, tuple) else result
            if (
                args.out_tensor_dir is not None
                and isinstance(image_out, dict)
            ):
                captures = detector_captures(predictor.model)
                for key in (
                    "pred_logits",
                    "pred_boxes",
                    "pred_boxes_xyxy",
                    "pred_masks",
                    "tracker_backbone_fpn_0",
                    "tracker_backbone_fpn_1",
                    "tracker_backbone_fpn_2",
                    "tracker_backbone_pos_enc",
                ):
                    value = image_out.get(key)
                    if torch.is_tensor(value) and key not in captures:
                        captures[key] = value.detach().clone()
            return result

        predictor.model.detector.forward_video_grounding_multigpu = (
            forward_video_grounding_wrapper
        )
        response = predictor.handle_request(
            {"type": "start_session", "resource_path": str(args.frame_dir)}
        )
        session_id = response["session_id"]
        add_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": args.prompt,
            }
        )
        rows = []
        row, selected_obj_id = row_from_outputs(
            add_response["outputs"],
            offset=0,
            frame_index=0,
            width=width,
            height=height,
            mask_dir=args.out_mask_dir,
        )
        rows.append(row)
        candidate_rows = candidate_rows_from_outputs(
            add_response["outputs"],
            width=width,
            height=height,
            selected_obj_id=selected_obj_id,
        )
        for response in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_index = int(response["frame_index"])
            if frame_index == 0:
                continue
            if frame_index >= args.frames:
                break
            row, selected_obj_id = row_from_outputs(
                response["outputs"],
                offset=frame_index,
                frame_index=frame_index,
                width=width,
                height=height,
                mask_dir=args.out_mask_dir,
                initial_obj_id=selected_obj_id,
            )
            rows.append(row)
            if len(rows) >= args.frames:
                break
        torch.cuda.synchronize()
        if args.out_tensor_dir is not None:
            state_record = predictor._all_inference_states[session_id]["state"]
            input_batch = state_record.get("input_batch")
            img_batch = getattr(input_batch, "img_batch", None)
            img_tensors = (
                img_batch
                if torch.is_tensor(img_batch)
                else getattr(img_batch, "tensors", None)
            )
            if torch.is_tensor(img_tensors) and img_tensors.ndim == 4:
                write_f32_tensor(
                    args.out_tensor_dir / "python_input_image_preprocessed",
                    nchw_to_ggml_whcb(img_tensors[:1]),
                )
            dump_detector_tensors(
                args.out_tensor_dir,
                predictor.model,
                state_record.get("feature_cache", {}),
            )
            dump_tracker_state_tensors(
                args.out_tensor_dir,
                predictor.model,
                state_record,
                selected_obj_id,
            )
        predictor.handle_request({"type": "reset_session", "session_id": session_id})

    meta = {
        "source": "official-python-sam3-meta",
        "sam3_version": args.version,
        "frame_dir": str(args.frame_dir),
        "decoded_width": width,
        "decoded_height": height,
        "frames": args.frames,
        "text_prompt": args.prompt,
        "python_dtype": args.dtype,
        "tf32_policy": args.tf32,
        "image_size": 1008,
    }
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        out.write(json.dumps(meta, sort_keys=True) + "\n")
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    if args.out_candidates_jsonl is not None:
        args.out_candidates_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_candidates_jsonl.open("w", encoding="utf-8") as out:
            for row in candidate_rows:
                out.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"out_jsonl": str(args.out_jsonl), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
