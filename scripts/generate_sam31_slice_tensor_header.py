#!/usr/bin/env python3
"""Generate a C++ tensor-name table from the SAM3.1 implementation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cxx_string_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def cxx_shape(shape: list[int] | None) -> str:
    dims = list(shape or [])
    rank = len(dims)
    padded = (dims + [0, 0, 0, 0])[:4]
    return "{{{{{}}}, {}}}".format(", ".join(str(dim) for dim in padded), rank)


def append_mask_decoder_contract(lines: list[str], contract: dict) -> None:
    propagation = contract["propagation_decoder"]
    interactive = contract["interactive_decoder"]
    lines.extend(
        [
            "",
            "struct Sam31MaskDecoderContract {",
            "    int embedding_dim;",
            "    int multiplex_count;",
            "    int iou_token_count;",
            "    int obj_score_token_count;",
            "    int mask_token_count;",
            "    int masks_per_object;",
            "    int output_token_count_before_sparse_prompts;",
            "    int hyper_mlp_count;",
            "    int twoway_depth;",
            "    bool has_obj_score_token;",
            "    bool uses_high_res_features;",
            "};",
            "",
            "inline constexpr Sam31MaskDecoderContract multiplex_mask_decoder_propagation_contract = {",
            f"    {int(propagation['embedding_dim'])},",
            f"    {int(propagation['multiplex_count'])},",
            f"    {int(propagation['iou_token_count'])},",
            f"    {int(propagation['obj_score_token_count'])},",
            f"    {int(propagation['mask_token_count'])},",
            f"    {int(propagation['masks_per_object'])},",
            f"    {int(propagation['output_token_count_before_sparse_prompts'])},",
            f"    {int(propagation['hyper_mlp_count'])},",
            f"    {int(propagation['twoway_depth'])},",
            f"    {str(bool(propagation['has_obj_score_token'])).lower()},",
            f"    {str(bool(propagation['uses_high_res_features'])).lower()},",
            "};",
            "",
            "inline constexpr Sam31MaskDecoderContract multiplex_mask_decoder_interactive_contract = {",
            f"    {int(interactive['embedding_dim'])},",
            f"    {int(interactive['multiplex_count'])},",
            f"    {int(interactive['iou_token_count'])},",
            f"    {int(interactive['obj_score_token_count'])},",
            f"    {int(interactive['mask_token_count'])},",
            f"    {int(interactive['masks_per_object'])},",
            f"    {int(interactive['output_token_count_before_sparse_prompts'])},",
            f"    {int(interactive['hyper_mlp_count'])},",
            f"    {int(interactive['twoway_depth'])},",
            f"    {str(bool(interactive['has_obj_score_token'])).lower()},",
            f"    {str(bool(interactive['uses_high_res_features'])).lower()},",
            "};",
            "",
            "static_assert(multiplex_mask_decoder_propagation_contract.multiplex_count == 16);",
            "static_assert(multiplex_mask_decoder_propagation_contract.mask_token_count == 48);",
            "static_assert(multiplex_mask_decoder_propagation_contract.masks_per_object == 3);",
            "static_assert(multiplex_mask_decoder_propagation_contract.output_token_count_before_sparse_prompts == 80);",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--slice", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    manifest = coverage["sam31_implementation_manifest"]
    entry = next((item for item in manifest if item["slice"] == args.slice), None)
    if entry is None:
        raise SystemExit(f"slice not found: {args.slice}")
    tensor_names = entry["tensor_names"]
    tensor_specs = entry["tensor_specs"]
    if entry["tensor_count"] != len(tensor_names) or entry["tensor_count"] != len(tensor_specs):
        raise SystemExit(
            f"manifest count mismatch for {args.slice}: "
            f"{entry['tensor_count']} != names {len(tensor_names)} / specs {len(tensor_specs)}"
        )

    guard_name = "SAM3_GENERATED_SAM31_" + args.slice.upper() + "_TENSORS_H"
    lines = [
        "#include <array>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        f"#ifndef {guard_name}",
        f"#define {guard_name}",
        "",
        "namespace sam3::generated {",
        "",
        "struct Sam31TensorShape {",
        "    std::array<int64_t, 4> dims;",
        "    int rank;",
        "};",
        "",
        "struct Sam31TensorSpec {",
        "    std::string_view name;",
        "    std::string_view source_key;",
        "    Sam31TensorShape shape;",
        "    std::string_view dtype;",
        "    int64_t numel;",
        "};",
        "",
        f"inline constexpr std::array<std::string_view, {len(tensor_names)}> {args.slice}_tensor_names = {{",
    ]
    lines.extend(f"    {cxx_string_literal(name)}," for name in tensor_names)
    lines.extend(
        [
            "};",
            "",
            f"inline constexpr std::array<Sam31TensorSpec, {len(tensor_specs)}> {args.slice}_tensor_specs = {{ {{",
        ]
    )
    lines.extend(
        "    {"
        + f"{cxx_string_literal(spec['name'])}, "
        + f"{cxx_string_literal(spec.get('source_key') or '')}, "
        + f"{cxx_shape(spec.get('shape'))}, "
        + f"{cxx_string_literal(spec.get('dtype') or '')}, "
        + f"{int(spec.get('numel') or 0)}"
        + "},"
        for spec in tensor_specs
    )
    lines.extend(
        [
            "} };",
            "",
            f"static_assert({args.slice}_tensor_names.size() == {args.slice}_tensor_specs.size());",
            f"static_assert({args.slice}_tensor_specs.size() == {len(tensor_specs)});",
        ]
    )
    architecture_contract = entry.get("architecture_contract")
    if args.slice == "multiplex_mask_decoder" and architecture_contract is not None:
        append_mask_decoder_contract(lines, architecture_contract)
    lines.extend(
        [
            "",
            "}  // namespace sam3::generated",
            "",
            f"#endif  // {guard_name}",
            "",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(tensor_names)} tensor names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
