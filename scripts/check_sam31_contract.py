#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fail if the SAM3.1 checkpoint contract is not loader-ready."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _parse_cxx_int_constants(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group("name"): int(match.group("value"))
        for match in re.finditer(
            r"inline constexpr int (?P<name>[a-zA-Z_][a-zA-Z0-9_]*) = (?P<value>\d+);",
            text,
        )
    }


def _parse_python_int_constants(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group("name"): int(match.group("value"))
        for match in re.finditer(
            r"^(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>\d+)$",
            text,
            re.M,
        )
    }


def _parse_cxx_mask_decoder_contract(path: Path, name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"inline constexpr MaskDecoderContract {re.escape(name)} = \{{(?P<body>.*?)\n\}};",
        text,
        re.S,
    )
    if match is None:
        return {}
    fields: dict[str, Any] = {}
    for field_match in re.finditer(
        r"\.(?P<name>[a-zA-Z_][a-zA-Z0-9_]*) = (?P<value>true|false|\d+|[a-zA-Z_][a-zA-Z0-9_]*),",
        match.group("body"),
    ):
        value = field_match.group("value")
        if value == "true":
            parsed: Any = True
        elif value == "false":
            parsed = False
        elif value.isdigit():
            parsed = int(value)
        else:
            parsed = value
        fields[field_match.group("name")] = parsed
    return fields


def _check_cxx_header(path: Path, expected_contract: dict[str, Any], errors: list[str]) -> None:
    constants = _parse_cxx_int_constants(path)
    for key, value in {
        "image_size": 1008,
        "backbone_stride": 14,
        "multiplex_count": 16,
        "eval_multiplex_count": 16,
        "max_num_objects": 16,
        "num_maskmem": 7,
        "max_obj_ptrs_in_encoder": 16,
        "tracker_mem_dim": 256,
    }.items():
        _require(constants.get(key) == value, f"C++ {key} changed: {constants.get(key)!r}", errors)

    propagation = _parse_cxx_mask_decoder_contract(path, "propagation_mask_decoder")
    for key, value in expected_contract.items():
        parsed = propagation.get(key)
        if parsed == "multiplex_count":
            parsed = constants.get("multiplex_count")
        _require(parsed == value, f"C++ propagation {key} changed: {parsed!r} != {value!r}", errors)


def _check_cxx_registration(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    required_fragments = [
        "const bool use_sam31_tracker = hp.is_sam3_1()",
        "if (!use_sam31_tracker)",
        'register_neck(model.neck_trk, "neck.trk.")',
        "register_sam31_manifest_tensor",
        "sam31_bind_multiplex_mask_decoder(model)",
        "sam31_build_mux_mask_dec_graph",
        "sam31_mux_tokens_initial",
        "sam31_mux_masks",
        "sam31_mux_iou",
        "sam31_mux_obj_score",
        'const std::string p = "trk.model.sam_mask_decoder"',
        "dec.twoway_blocks.resize(contract.twoway_depth)",
        "bind_attn(dec.final_attn, p + \".transformer.final_attn_token_to_image\")",
        "model.file_tensor_types.find(tensor_name)",
        "ggml_new_tensor(ctx, file_type, shape.rank, ne.data())",
        "if (use_sam31_tracker)",
        "sam3::sam31::tensor_manifest",
        "return;",
    ]
    for fragment in required_fragments:
        _require(fragment in text, f"C++ SAM3.1 registration fragment missing: {fragment}", errors)


def _check_cxx_tensor_manifest(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"std::array<TensorSpec, (?P<count>\d+)> tensor_manifest", text)
    count = int(match.group("count")) if match else None
    _require(count == 493, f"C++ SAM3.1 tensor manifest count changed: {count!r}", errors)
    _require("static_assert(tensor_manifest.size() == 493)" in text, "C++ tensor manifest static_assert missing", errors)
    for fragment in [
        '"trk.model.sam_mask_decoder.iou_token.weight"',
        '"trk.model.interactive_sam_mask_decoder.iou_token.weight"',
        '"trk.model.maskmem_backbone.mask_downsampler.encoder.0.weight"',
        '"trk.model.transformer.encoder.layers.0.self_attn_q_proj.weight"',
        '"det.backbone.vision_backbone.propagation_convs.0.conv_3x3.weight"',
    ]:
        _require(fragment in text, f"C++ SAM3.1 tensor manifest missing: {fragment}", errors)


def _check_file_format_versions(
    converter: Path,
    cxx_source: Path,
    cxx_api: Path,
    errors: list[str],
) -> None:
    py_constants = _parse_python_int_constants(converter)
    cxx_text = cxx_source.read_text(encoding="utf-8")
    cxx_api_text = cxx_api.read_text(encoding="utf-8")
    current_match = re.search(r"static constexpr int SAM3_FILE_VERSION = (?P<value>\d+);", cxx_text)
    min_match = re.search(r"static constexpr int SAM3_FILE_VERSION_MIN = (?P<value>\d+);", cxx_text)
    current = int(current_match.group("value")) if current_match else None
    minimum = int(min_match.group("value")) if min_match else None
    _require(py_constants.get("VERSION") == 4, f"converter VERSION changed: {py_constants.get('VERSION')!r}", errors)
    _require(
        py_constants.get("MODEL_TYPE_SAM31") == 4,
        f"converter MODEL_TYPE_SAM31 changed: {py_constants.get('MODEL_TYPE_SAM31')!r}",
        errors,
    )
    converter_text = converter.read_text(encoding="utf-8")
    _require(
        "SAM3.1 Object Multiplex conversion enabled" in converter_text,
        "converter still lacks SAM3.1 conversion enablement",
        errors,
    )
    _require(
        "model_type=MODEL_TYPE_SAM31 if convert_as_sam31 else MODEL_TYPE_SAM3" in converter_text,
        "converter does not write SAM3.1 model_type",
        errors,
    )
    _require(current == 4, f"C++ SAM3_FILE_VERSION changed: {current!r}", errors)
    _require(minimum == 3, f"C++ SAM3_FILE_VERSION_MIN changed: {minimum!r}", errors)
    _require("sam3_1 = 4" in cxx_api_text, "C++ sam3_model_type::sam3_1 is not pinned to 4", errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cxx-header", type=Path, default=Path("sam3_sam31.h"))
    parser.add_argument("--cxx-tensor-manifest", type=Path, default=Path("sam3_sam31_tensor_manifest.h"))
    parser.add_argument("--cxx-source", type=Path, default=Path("sam3.cpp"))
    parser.add_argument("--cxx-api", type=Path, default=Path("sam3.h"))
    parser.add_argument("--converter", type=Path, default=Path("convert_sam3_to_ggml.py"))
    args = parser.parse_args()

    coverage = _load_json(args.coverage)
    inventory = _load_json(args.inventory)
    checks = coverage.get("sam31_implementation_manifest_checks", {})
    manifest = coverage.get("sam31_implementation_manifest", [])
    errors: list[str] = []

    _require(inventory.get("sam31_detected") is True, "inventory did not detect SAM3.1", errors)
    sam31_contract = inventory.get("sam31_contract") or {}
    for key, value in {
        "model_type": "sam3.1",
        "image_size": 1008,
        "backbone_stride": 14,
        "multiplex_count": 16,
        "eval_multiplex_count": 16,
        "max_num_objects": 16,
        "num_maskmem": 7,
        "max_obj_ptrs_in_encoder": 16,
        "tracker_mem_dim": 256,
        "num_multimask_outputs": 3,
        "multimask_output_in_sam": True,
        "propagation_masks_per_object": 3,
        "propagation_mask_token_count": 48,
        "propagation_output_token_count": 80,
    }.items():
        _require(
            sam31_contract.get(key) == value,
            f"inventory sam31_contract {key} changed: {sam31_contract.get(key)!r} != {value!r}",
            errors,
        )
    _require(checks.get("diff_rows") == 0, f"manifest checks failed: {checks}", errors)
    _require(checks.get("manifest_tensor_count") == 493, "unexpected manifest tensor count", errors)
    _require(checks.get("candidate_only_tensors") == 493, "unexpected SAM3.1-only tensor count", errors)
    _require(checks.get("mask_decoder_contract_ok") is True, "mask decoder contract is not valid", errors)

    required_slices = {
        "multiplex_mask_decoder": 273,
        "multiplex_memory_backbone": 40,
        "multiplex_tracker_transformer": 144,
        "multiplex_detector_feature_adapters": 36,
    }
    manifest_by_slice = {item.get("slice"): item for item in manifest}
    for slice_name, tensor_count in required_slices.items():
        item = manifest_by_slice.get(slice_name)
        _require(item is not None, f"missing manifest slice {slice_name}", errors)
        if item is not None:
            _require(
                item.get("tensor_count") == tensor_count,
                f"slice {slice_name} tensor count changed: {item.get('tensor_count')} != {tensor_count}",
                errors,
            )

    mask_decoder = manifest_by_slice.get("multiplex_mask_decoder", {})
    propagation = (
        mask_decoder.get("architecture_contract", {})
        .get("propagation_decoder", {})
    )
    expected_contract = {
        "embedding_dim": 256,
        "multiplex_count": 16,
        "iou_token_count": 16,
        "obj_score_token_count": 16,
        "mask_token_count": 48,
        "masks_per_object": 3,
        "output_token_count_before_sparse_prompts": 80,
        "hyper_mlp_count": 3,
        "twoway_depth": 2,
        "has_obj_score_token": True,
        "uses_high_res_features": True,
    }
    for key, value in expected_contract.items():
        _require(
            propagation.get(key) == value,
            f"propagation decoder {key} changed: {propagation.get(key)!r} != {value!r}",
            errors,
        )
    _check_cxx_header(args.cxx_header, expected_contract, errors)
    _check_cxx_tensor_manifest(args.cxx_tensor_manifest, errors)
    _check_cxx_registration(args.cxx_source, errors)
    _check_file_format_versions(args.converter, args.cxx_source, args.cxx_api, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        "sam31_contract_check=ok "
        f"manifest_tensors={checks['manifest_tensor_count']} "
        f"slices={len(required_slices)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
