#!/usr/bin/env python3
"""Summarize SAM3.1 checkpoint compatibility with the current SAM3 converter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

import convert_sam3_to_ggml


def load_state_dict(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    return ckpt


def tensor_meta(key: str, value: Any) -> dict[str, Any]:
    return {
        "source_key": key,
        "shape": list(value.shape) if hasattr(value, "shape") else None,
        "dtype": str(value.dtype) if hasattr(value, "dtype") else type(value).__name__,
        "numel": int(value.numel()) if hasattr(value, "numel") else None,
    }


def renamed_tensor_map(ckpt: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    renamed: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for key, value in ckpt.items():
        name = convert_sam3_to_ggml.rename_key(key)
        if name is None:
            skipped.append(key)
            continue
        renamed[name] = tensor_meta(key, value)
    return renamed, skipped


def prefix_counts(names: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in sorted(names):
        prefix = name.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def samples(names: set[str], mapping: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in sorted(names)[:limit]:
        out.append({"name": name, **mapping[name]})
    return out


def classify_sam31_only(name: str) -> str:
    if name.startswith("det.backbone.vision_backbone.interactive_convs."):
        return "detector_interactive_convs"
    if name.startswith("det.backbone.vision_backbone.propagation_convs."):
        return "detector_propagation_convs"
    if name.startswith("trk.model.transformer.encoder.layers."):
        if ".self_attn_" in name:
            return "tracker_transformer_self_attention"
        if ".image_cross_attn_" in name:
            return "tracker_transformer_image_cross_attention"
        if ".cross_attn_" in name:
            return "tracker_transformer_object_cross_attention"
        if ".linear" in name or ".norm" in name:
            return "tracker_transformer_ffn_norm"
        return "tracker_transformer_other"
    if (
        name.startswith("trk.model.sam_mask_decoder.")
        or name.startswith("trk.model.interactive_sam_mask_decoder.")
    ):
        return "tracker_multiplex_mask_decoder"
    if name.startswith("trk.model.interactive_sam_prompt_encoder."):
        return "tracker_multiplex_prompt_encoder"
    if name.startswith("trk.model.maskmem_backbone."):
        return "tracker_multiplex_maskmem_backbone"
    if name.startswith("trk.model.interactive_mask_downsample."):
        return "tracker_multiplex_mask_downsample"
    if name.startswith("trk.model.memory_encoder.") or name.startswith("trk.model.memory_attention."):
        return "tracker_multiplex_memory"
    if (
        name.startswith("trk.model.obj_ptr_proj.")
        or name.startswith("trk.model.interactive_obj_ptr_proj.")
        or name.startswith("trk.model.obj_ptr_tpos_proj.")
    ):
        return "tracker_object_pointer"
    if name.startswith("trk.model."):
        return "tracker_multiplex_core"
    return "other"


def classify_sam3_only(name: str) -> str:
    if name.startswith("mem_attn."):
        return "legacy_tracker_memory_attention"
    if name.startswith("mem_enc."):
        return "legacy_tracker_memory_encoder"
    if name.startswith("sam_dec."):
        return "legacy_sam_mask_decoder"
    if name.startswith("sam_pe."):
        return "legacy_sam_prompt_encoder"
    if name.startswith("neck."):
        return "legacy_tracker_neck"
    if name.startswith("obj_ptr") or name.startswith("no_obj") or name.startswith("no_mem"):
        return "legacy_object_pointer"
    if name.startswith("trk_mask_ds."):
        return "legacy_mask_downsample"
    return "other"


def category_summary(
    names: set[str],
    mapping: dict[str, dict[str, Any]],
    classifier,
    sample_limit: int,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        category = classifier(name)
        item = summary.setdefault(
            category, {"count": 0, "numel": 0, "names": [], "tensor_specs": [], "samples": []}
        )
        item["count"] += 1
        item["numel"] += int(mapping[name].get("numel") or 0)
        item["names"].append(name)
        item["tensor_specs"].append({"name": name, **mapping[name]})
        if len(item["samples"]) < sample_limit:
            item["samples"].append({"name": name, **mapping[name]})
    return dict(sorted(summary.items(), key=lambda item: (-item[1]["count"], item[0])))


def _shape_for(tensor_specs: list[dict[str, Any]], name: str) -> list[int] | None:
    for spec in tensor_specs:
        if spec["name"] == name:
            shape = spec.get("shape")
            return list(shape) if isinstance(shape, list) else None
    return None


def _unique_index_count(tensor_names: list[str], pattern: str) -> int:
    compiled = re.compile(pattern)
    return len({int(match.group(1)) for name in tensor_names if (match := compiled.search(name))})


def _mask_decoder_component_contract(
    tensor_names: list[str],
    tensor_specs: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    iou_shape = _shape_for(tensor_specs, f"{prefix}.iou_token.weight") or []
    mask_shape = _shape_for(tensor_specs, f"{prefix}.mask_tokens.weight") or []
    obj_shape = _shape_for(tensor_specs, f"{prefix}.obj_score_token.weight") or []
    iou_head_out = _shape_for(tensor_specs, f"{prefix}.iou_prediction_head.layers.2.weight") or []
    has_obj_score = bool(obj_shape)

    multiplex_count = int(iou_shape[0]) if len(iou_shape) >= 2 else 0
    embedding_dim = int(iou_shape[1]) if len(iou_shape) >= 2 else 0
    mask_token_count = int(mask_shape[0]) if len(mask_shape) >= 2 else 0
    masks_per_object = int(iou_head_out[0]) if len(iou_head_out) >= 2 else 0
    output_token_count = mask_token_count + multiplex_count + (multiplex_count if has_obj_score else 0)

    return {
        "prefix": prefix,
        "embedding_dim": embedding_dim,
        "multiplex_count": multiplex_count,
        "has_obj_score_token": has_obj_score,
        "iou_token_count": multiplex_count,
        "obj_score_token_count": int(obj_shape[0]) if has_obj_score else 0,
        "mask_token_count": mask_token_count,
        "masks_per_object": masks_per_object,
        "output_token_count_before_sparse_prompts": output_token_count,
        "hyper_mlp_count": _unique_index_count(
            tensor_names, rf"^{re.escape(prefix)}\.output_hypernetworks_mlps\.(\d+)\."
        ),
        "twoway_depth": _unique_index_count(
            tensor_names, rf"^{re.escape(prefix)}\.transformer\.layers\.(\d+)\."
        ),
        "uses_high_res_features": _shape_for(tensor_specs, f"{prefix}.conv_s0.weight") is not None
        and _shape_for(tensor_specs, f"{prefix}.conv_s1.weight") is not None,
    }


def _mask_decoder_contract(tensor_names: list[str], tensor_specs: list[dict[str, Any]]) -> dict[str, Any]:
    interactive_prefix = "trk.model.interactive_sam_mask_decoder"
    propagation_prefix = "trk.model.sam_mask_decoder"
    propagation = _mask_decoder_component_contract(tensor_names, tensor_specs, propagation_prefix)
    interactive = _mask_decoder_component_contract(tensor_names, tensor_specs, interactive_prefix)
    return {
        "interactive_decoder": interactive,
        "propagation_decoder": propagation,
        "implementation_notes": [
            "The propagation decoder is SAM3.1's multiplex path.",
            "The propagation decoder uses multimask_outputs_only=True in this checkpoint.",
            "C++ graph output shape before demux is [num_buckets, multiplex_count, masks_per_object, H, W].",
            "Object, IoU, and mask output tokens are concatenated before sparse prompt tokens.",
        ],
    }


def slice_architecture_contract(
    slice_name: str,
    tensor_names: list[str],
    tensor_specs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if slice_name == "multiplex_mask_decoder":
        return _mask_decoder_contract(tensor_names, tensor_specs)
    return None


def implementation_manifest(candidate_categories: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    order = [
        (
            "multiplex_mask_decoder",
            ["tracker_multiplex_mask_decoder", "tracker_multiplex_prompt_encoder"],
            "Load and run SAM3.1 multiplex/interactive SAM decoder tensors.",
            "C++ decoder tensor dumps match official Python for one prompt frame.",
        ),
        (
            "multiplex_memory_backbone",
            ["tracker_multiplex_maskmem_backbone", "tracker_multiplex_mask_downsample"],
            "Load and run per-object multiplex memory feature encoding.",
            "C++ mask memory tensors match official Python before temporal attention.",
        ),
        (
            "multiplex_tracker_transformer",
            [
                "tracker_transformer_self_attention",
                "tracker_transformer_object_cross_attention",
                "tracker_transformer_image_cross_attention",
                "tracker_transformer_ffn_norm",
                "tracker_object_pointer",
                "tracker_multiplex_core",
            ],
            "Load and run the decoupled multiplex tracker transformer and object pointer path.",
            "C++ propagated mask logits match official Python on the same frame range.",
        ),
        (
            "multiplex_detector_feature_adapters",
            ["detector_interactive_convs", "detector_propagation_convs"],
            "Load and run detector-side interactive/propagation feature adapters.",
            "C++ text/point prompt detection inputs match official Python before tracking.",
        ),
    ]
    manifest = []
    for name, categories, purpose, acceptance in order:
        count = sum(int(candidate_categories.get(category, {}).get("count", 0)) for category in categories)
        numel = sum(int(candidate_categories.get(category, {}).get("numel", 0)) for category in categories)
        tensor_names = [
            tensor_name
            for category in categories
            for tensor_name in candidate_categories.get(category, {}).get("names", [])
        ]
        tensor_specs = [
            tensor_spec
            for category in categories
            for tensor_spec in candidate_categories.get(category, {}).get("tensor_specs", [])
        ]
        architecture_contract = slice_architecture_contract(name, tensor_names, tensor_specs)
        manifest.append(
            {
                "slice": name,
                "categories": categories,
                "tensor_count": count,
                "numel": numel,
                "tensor_names": tensor_names,
                "tensor_specs": tensor_specs,
                "purpose": purpose,
                "acceptance": acceptance,
                "architecture_contract": architecture_contract,
            }
        )
    return manifest


def manifest_checks(manifest: list[dict[str, Any]], candidate_only_count: int) -> dict[str, Any]:
    tensor_names = [name for item in manifest for name in item["tensor_names"]]
    tensor_specs = [spec for item in manifest for spec in item["tensor_specs"]]
    duplicate_names = sorted({name for name in tensor_names if tensor_names.count(name) > 1})
    slice_count_mismatches = [
        item["slice"] for item in manifest if item["tensor_count"] != len(item["tensor_names"])
    ]
    missing_source_keys = sorted(spec["name"] for spec in tensor_specs if not spec.get("source_key"))
    missing_shapes = sorted(spec["name"] for spec in tensor_specs if spec.get("shape") is None)
    missing_dtypes = sorted(spec["name"] for spec in tensor_specs if not spec.get("dtype"))
    missing_numel = sorted(spec["name"] for spec in tensor_specs if spec.get("numel") is None)
    mask_decoder = next((item for item in manifest if item["slice"] == "multiplex_mask_decoder"), {})
    propagation = (
        mask_decoder.get("architecture_contract", {})
        .get("propagation_decoder", {})
    )
    mask_decoder_contract_ok = (
        propagation.get("multiplex_count") == 16
        and propagation.get("mask_token_count") == 48
        and propagation.get("masks_per_object") == 3
        and propagation.get("output_token_count_before_sparse_prompts") == 80
    )
    return {
        "manifest_tensor_count": len(tensor_names),
        "candidate_only_tensors": candidate_only_count,
        "diff_rows": int(
            len(tensor_names) != candidate_only_count
            or bool(duplicate_names)
            or bool(slice_count_mismatches)
            or bool(missing_source_keys)
            or bool(missing_shapes)
            or bool(missing_dtypes)
            or bool(missing_numel)
            or not mask_decoder_contract_ok
        ),
        "duplicate_tensor_names": duplicate_names,
        "slice_count_mismatches": slice_count_mismatches,
        "missing_source_keys": missing_source_keys,
        "missing_shapes": missing_shapes,
        "missing_dtypes": missing_dtypes,
        "missing_numel": missing_numel,
        "mask_decoder_contract_ok": mask_decoder_contract_ok,
    }


def compare_checkpoints(reference_path: Path, candidate_path: Path, sample_limit: int) -> dict[str, Any]:
    reference = load_state_dict(reference_path)
    candidate = load_state_dict(candidate_path)

    reference_renamed, reference_skipped = renamed_tensor_map(reference)
    candidate_renamed, candidate_skipped = renamed_tensor_map(candidate)

    reference_names = set(reference_renamed)
    candidate_names = set(candidate_renamed)
    common = reference_names & candidate_names

    shape_diffs = []
    same_shape_common = 0
    for name in sorted(common):
        lhs = reference_renamed[name]
        rhs = candidate_renamed[name]
        if lhs["shape"] == rhs["shape"]:
            same_shape_common += 1
            continue
        shape_diffs.append(
            {
                "name": name,
                "reference": lhs,
                "candidate": rhs,
            }
        )

    reference_only = reference_names - candidate_names
    candidate_only = candidate_names - reference_names
    candidate_inventory = convert_sam3_to_ggml.checkpoint_inventory(candidate)
    reference_categories = category_summary(
        reference_only, reference_renamed, classify_sam3_only, sample_limit
    )
    candidate_categories = category_summary(
        candidate_only, candidate_renamed, classify_sam31_only, sample_limit
    )
    manifest = implementation_manifest(candidate_categories)

    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "candidate_sam31_detected": candidate_inventory["sam31_detected"],
        "candidate_sam31_markers": candidate_inventory["sam31_markers"],
        "reference_renamed_tensors": len(reference_renamed),
        "candidate_renamed_tensors": len(candidate_renamed),
        "reference_skipped_tensors": len(reference_skipped),
        "candidate_skipped_tensors": len(candidate_skipped),
        "common_renamed_tensors": len(common),
        "same_shape_common_tensors": same_shape_common,
        "shape_diff_tensors": len(shape_diffs),
        "reference_only_tensors": len(reference_only),
        "candidate_only_tensors": len(candidate_only),
        "reference_only_prefix_counts": prefix_counts(reference_only),
        "candidate_only_prefix_counts": prefix_counts(candidate_only),
        "reference_only_categories": reference_categories,
        "candidate_only_categories": candidate_categories,
        "sam31_implementation_manifest": manifest,
        "sam31_implementation_manifest_checks": manifest_checks(manifest, len(candidate_only)),
        "shape_diffs": shape_diffs[:sample_limit],
        "reference_only_samples": samples(reference_only, reference_renamed, sample_limit),
        "candidate_only_samples": samples(candidate_only, candidate_renamed, sample_limit),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3", type=Path, default=Path("models/sam3/sam3.pt"))
    parser.add_argument("--sam31", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--inventory-out", type=Path, required=True)
    parser.add_argument("--coverage-out", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    sam31 = load_state_dict(args.sam31)
    inventory = convert_sam3_to_ggml.checkpoint_inventory(sam31)
    coverage = compare_checkpoints(args.sam3, args.sam31, args.sample_limit)

    args.inventory_out.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.coverage_out.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "sam31_detected={detected} common={common} same_shape={same_shape} "
        "reference_only={reference_only} candidate_only={candidate_only}".format(
            detected=coverage["candidate_sam31_detected"],
            common=coverage["common_renamed_tensors"],
            same_shape=coverage["same_shape_common_tensors"],
            reference_only=coverage["reference_only_tensors"],
            candidate_only=coverage["candidate_only_tensors"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
