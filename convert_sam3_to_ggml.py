#!/usr/bin/env python3
"""Convert SAM 3 PyTorch checkpoint to ggml binary format.

Usage:
    python convert_sam3_to_ggml.py --model sam3.pt --output sam3.ggml [--ftype 1] [--tokenizer <dir>]

ftype: 0 = float32, 1 = float16 (default)
The tokenizer (vocab.json + merges.txt) is embedded in the output file.
"""

import argparse
import json
import struct
import sys
import os
import re
from collections import Counter
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

MAGIC   = 0x73616D33   # "sam3"
VERSION = 4
FTYPE_F32 = 0
FTYPE_F16 = 1
FTYPE_BF16 = 24
GGML_TYPE_BF16 = 30
MODEL_TYPE_SAM3 = 0
MODEL_TYPE_SAM3_VISUAL = 1
MODEL_TYPE_SAM31 = 4

SAM31_CONTRACT = {
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
    "use_obj_ptrs_in_encoder": True,
    "use_high_res_features_in_sam": True,
    "is_dynamic_multiplex": True,
}


def detect_sam31_checkpoint(keys) -> bool:
    """Return True for SAM3.1 Object Multiplex checkpoints."""
    key_set = set(keys)
    return (
        any(k.startswith("tracker.model.") for k in key_set)
        or any(".interactive_convs." in k for k in key_set)
        or any(".propagation_convs." in k for k in key_set)
        or "tracker.model.output_valid_embed" in key_set
    )


def print_checkpoint_summary(ckpt: dict, *, max_keys: int = 40) -> None:
    """Print a compact checkpoint inventory for converter bring-up."""
    keys = list(ckpt)
    print(f"Checkpoint has {len(keys)} tensors")
    print(f"SAM3.1 Object Multiplex detected: {detect_sam31_checkpoint(keys)}")
    print("Top-level prefixes:")
    for prefix, count in Counter(k.split(".", 1)[0] for k in keys).most_common(20):
        print(f"  {prefix}: {count}")
    print("First tensors:")
    for key in keys[:max_keys]:
        value = ckpt[key]
        shape = list(value.shape) if hasattr(value, "shape") else type(value).__name__
        print(f"  {key}  {shape}")


def checkpoint_inventory(ckpt: dict) -> dict:
    """Build a machine-readable checkpoint inventory for converter bring-up."""
    keys = list(ckpt)
    renamed = []
    skipped = []
    tensors = []
    for key in keys:
        value = ckpt[key]
        new_name = rename_key(key)
        if new_name is None:
            skipped.append(key)
        else:
            renamed.append(new_name)
        shape = list(value.shape) if hasattr(value, "shape") else None
        dtype = str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
        numel = int(value.numel()) if hasattr(value, "numel") else None
        tensors.append({
            "key": key,
            "renamed": new_name,
            "shape": shape,
            "dtype": dtype,
            "numel": numel,
        })

    top_prefixes = Counter(k.split(".", 1)[0] for k in keys)
    second_prefixes = Counter(".".join(k.split(".")[:2]) for k in keys if "." in k)
    sam31_markers = {
        "output_valid_embed": "tracker.model.output_valid_embed" in keys,
        "output_invalid_embed": "tracker.model.output_invalid_embed" in keys,
        "interactive_convs": any(".interactive_convs." in k for k in keys),
        "propagation_convs": any(".propagation_convs." in k for k in keys),
        "tracker_model": any(k.startswith("tracker.model.") for k in keys),
    }

    return {
        "num_tensors": len(keys),
        "sam31_detected": detect_sam31_checkpoint(keys),
        "sam31_contract": SAM31_CONTRACT if detect_sam31_checkpoint(keys) else None,
        "top_prefixes": dict(top_prefixes.most_common()),
        "second_prefixes": dict(second_prefixes.most_common()),
        "renamed_tensors": len(renamed),
        "skipped_tensors": len(skipped),
        "skipped_keys": skipped,
        "sam31_markers": sam31_markers,
        "tensors": tensors,
    }

# ── Hyperparameter defaults ───────────────────────────────────────────────────

HPARAMS_FIELDS = [
    ("img_size",              1008),
    ("patch_size",              14),
    ("vit_embed_dim",         1024),
    ("vit_depth",               32),
    ("vit_num_heads",           16),
    ("vit_mlp_ratio_x1000",  4625),
    ("vit_window_size",         24),
    ("n_global_attn_blocks",     4),
    ("global_attn_idx_0",        7),
    ("global_attn_idx_1",       15),
    ("global_attn_idx_2",       23),
    ("global_attn_idx_3",       31),
    ("text_width",            1024),
    ("text_heads",              16),
    ("text_layers",             24),
    ("text_ctx_len",            32),
    ("text_vocab_size",      49408),
    ("text_out_dim",           256),
    ("neck_dim",               256),
    ("fenc_layers",              6),
    ("fenc_heads",               8),
    ("fenc_ffn_dim",          2048),
    ("ddec_layers",              6),
    ("ddec_heads",               8),
    ("ddec_ffn_dim",          2048),
    ("ddec_num_queries",       200),
    ("geom_layers",              3),
    ("n_presence_tokens",        1),
    ("n_geom_queries",           4),
    ("sam_embed_dim",          256),
    ("sam_dec_depth",            2),
    ("sam_n_multimask",          3),
    ("sam_iou_head_depth",       3),
    ("mem_out_dim",             64),
    ("mem_attn_layers",          4),
    ("num_maskmem",              7),
    ("max_obj_ptrs",            16),
    ("n_amb_experts",            2),
    ("visual_only",              0),
]

# Tensor prefixes that belong exclusively to the detector path.
# When --visual-only is set, tensors whose renamed key starts with any of
# these prefixes are stripped from the output file.
VISUAL_ONLY_STRIP_PREFIXES = (
    "text.", "fenc.", "ddec.", "seg.", "geom.", "scoring.", "neck.det.",
)

# Tensor prefixes that MUST be present in a visual-only model.
VISUAL_ONLY_REQUIRED_PREFIXES = (
    "vit.", "neck.trk.", "sam_pe.", "sam_dec.", "mem_enc.", "mem_attn.", "obj_ptr_proj.",
)


# ── Key renaming ──────────────────────────────────────────────────────────────

def rename_key(k: str) -> str | None:
    """Map a PyTorch state_dict key to the flat ggml name.

    Returns None if the tensor should be skipped.
    """

    # ── Skip rules ────────────────────────────────────────────────────────
    # Only skip tensors that are genuinely training-only
    skip_patterns = [
        "attn_mask",                       # causal mask (deterministic, recomputed)
        ".dac_",                           # DAC dual supervision (training)
        "_dn_",                            # denoising queries (training)
        "text_projection",                 # unused in SAM3 inference (pooled output discarded by VETextEncoder)
    ]
    for pat in skip_patterns:
        if pat in k:
            return None

    # ── Detector path ─────────────────────────────────────────────────────
    # ViT backbone
    k = k.replace("detector.backbone.vision_backbone.trunk.", "vit.")
    # ViT MLP uses fc1/fc2 in timm
    k = k.replace(".mlp.fc1.", ".mlp.lin1.")
    k = k.replace(".mlp.fc2.", ".mlp.lin2.")
    # Attention
    k = k.replace(".attn.qkv.", ".attn.qkv.")
    k = k.replace(".attn.proj.", ".attn.proj.")

    # Detector neck
    k = k.replace("detector.backbone.vision_backbone.convs.", "neck.det.")
    k = k.replace("detector.backbone.vision_backbone.sam2_convs.", "neck.trk.")

    # Text encoder
    k = k.replace("detector.backbone.language_backbone.encoder.transformer.resblocks.",
                   "text.blocks.")
    k = k.replace("detector.backbone.language_backbone.encoder.token_embedding.",
                   "text.token_embed.")
    k = k.replace("detector.backbone.language_backbone.encoder.positional_embedding",
                   "text.pos_embed")
    k = k.replace("detector.backbone.language_backbone.encoder.ln_final.",
                   "text.ln_final.")
    k = k.replace("detector.backbone.language_backbone.resizer.",
                   "text.resizer.")
    # Text block sub-keys
    k = k.replace(".attn.in_proj_weight", ".attn.in_proj.weight")
    k = k.replace(".attn.in_proj_bias",   ".attn.in_proj.bias")
    k = k.replace(".mlp.c_fc.",  ".mlp.fc1.")
    k = k.replace(".mlp.c_proj.", ".mlp.fc2.")

    # Fusion encoder
    k = k.replace("detector.transformer.encoder.layers.", "fenc.layers.")
    k = k.replace(".cross_attn_image.", ".ca.")

    # DETR decoder
    k = k.replace("detector.transformer.decoder.layers.", "ddec.layers.")
    k = k.replace("detector.transformer.decoder.", "ddec.")
    k = k.replace(".cross_attn_image.", ".ca.")
    k = k.replace(".cross_attn.", ".ca.")
    k = k.replace(".self_attn.",  ".sa.")
    k = k.replace(".ca_text.",    ".ca_text.")
    k = k.replace(".catext_norm.", ".norm_ca_text.")

    # Geometry encoder
    k = k.replace("detector.geometry_encoder.", "geom.")
    k = k.replace("geom.encode.", "geom.layers.")

    # Segmentation head
    k = k.replace("detector.segmentation_head.", "seg.")

    # DotProductScoring
    k = k.replace("detector.dot_prod_scoring.", "scoring.")

    # ── Tracker path ──────────────────────────────────────────────────────
    # Memory attention transformer
    k = k.replace("tracker.transformer.encoder.layers.", "mem_attn.layers.")
    k = k.replace("tracker.transformer.encoder.norm.", "mem_attn.norm.")
    # RoPE attention: already uses q_proj/k_proj/v_proj/out_proj

    # Memory encoder (maskmem_backbone)
    k = k.replace("tracker.maskmem_backbone.", "mem_enc.")
    k = k.replace("mem_enc.fuser.layers.", "mem_enc.fuser.")
    k = k.replace("mem_enc.mask_downsampler.encoder.", "mem_enc.ds.")

    # SAM prompt encoder
    k = k.replace("tracker.sam_prompt_encoder.", "sam_pe.")
    k = k.replace("sam_pe.pe_layer.positional_encoding_gaussian_matrix",
                   "sam_pe.pe_gaussian")
    k = k.replace("sam_pe.mask_downscaling.", "sam_pe.mask_ds.")

    # SAM mask decoder
    k = k.replace("tracker.sam_mask_decoder.", "sam_dec.")
    k = k.replace("sam_dec.transformer.layers.", "sam_dec.twoway.")
    k = k.replace("sam_dec.transformer.final_attn_token_to_image.",
                   "sam_dec.final_attn.")
    k = k.replace("sam_dec.transformer.norm_final_attn.",
                   "sam_dec.final_norm.")
    k = k.replace("sam_dec.output_upscaling.", "sam_dec.upscale.")
    k = k.replace("sam_dec.output_hypernetworks_mlps.", "sam_dec.hyper.")

    # Object pointer projection
    k = k.replace("tracker.obj_ptr_proj.", "obj_ptr_proj.")
    k = k.replace("tracker.obj_ptr_tpos_proj.", "obj_ptr_tpos_proj.")
    k = k.replace("tracker.no_obj_ptr", "no_obj_ptr")
    k = k.replace("tracker.no_mem_embed", "no_mem_embed")
    k = k.replace("tracker.no_mem_pos_enc", "no_mem_pos_enc")
    k = k.replace("tracker.no_obj_embed_spatial", "no_obj_embed_spatial")
    k = k.replace("tracker.maskmem_tpos_enc", "mem_enc.tpos_enc")
    k = k.replace("tracker.mask_downsample.", "trk_mask_ds.")

    # ── Catch-all: remove any remaining prefixes ──────────────────────────
    k = k.replace("detector.", "det.")
    k = k.replace("tracker.", "trk.")

    return k


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_header(
    fout,
    ftype: int,
    n_tensors: int,
    visual_only: bool = False,
    model_type: int = MODEL_TYPE_SAM3,
):
    """Write file header: magic, version, ftype, n_tensors, hparams."""
    fout.write(struct.pack("<I", MAGIC))
    fout.write(struct.pack("<i", VERSION))
    fout.write(struct.pack("<i", ftype))
    fout.write(struct.pack("<i", n_tensors))
    if visual_only and model_type != MODEL_TYPE_SAM31:
        model_type = MODEL_TYPE_SAM3_VISUAL
    for name, val in HPARAMS_FIELDS:
        if name == "visual_only" and visual_only:
            val = 1
        fout.write(struct.pack("<i", val))
    fout.write(struct.pack("<i", model_type))


def fp32_to_bf16_bytes(data: np.ndarray) -> bytes:
    """Return round-to-nearest-even BF16 storage bytes for float32 data."""
    f32 = data.astype(np.float32, copy=False)
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    rounded = u32 + np.uint32(0x7FFF) + lsb.astype(np.uint32)
    return (rounded >> 16).astype(np.uint16).tobytes()


def write_tensor(fout, name: str, data: np.ndarray, ftype: int):
    """Write one tensor record with 32-byte aligned data."""
    n_dims = len(data.shape)
    name_bytes = name.encode("utf-8")

    # Determine storage dtype
    # 1D tensors, embeddings, and positions → always f32
    use_lowp = (ftype in (FTYPE_F16, FTYPE_BF16) and n_dims >= 2
                and "embed" not in name
                and "pos_embed" not in name
                and "tpos" not in name
                and "pe_gaussian" not in name
                and "freqs_cis" not in name
                and "token" not in name
                and "no_obj" not in name
                and "no_mem" not in name
                and "gamma" not in name)

    if use_lowp and ftype == FTYPE_F16:
        dtype_id = FTYPE_F16
        data = data.astype(np.float16)
        payload = data.tobytes()
    elif use_lowp and ftype == FTYPE_BF16:
        dtype_id = GGML_TYPE_BF16
        data = data.astype(np.float32)
        payload = fp32_to_bf16_bytes(data)
    else:
        dtype_id = FTYPE_F32
        data = data.astype(np.float32)
        payload = data.tobytes()

    # Write: n_dims, name_len, dtype, shape (reversed), name, padding, data
    fout.write(struct.pack("<i", n_dims))
    fout.write(struct.pack("<i", len(name_bytes)))
    fout.write(struct.pack("<i", dtype_id))

    # ggml expects dimensions in reverse order (column-major)
    for dim in reversed(data.shape):
        fout.write(struct.pack("<i", dim))

    fout.write(name_bytes)

    # Pad to 32-byte alignment
    pos = fout.tell()
    pad = (32 - pos % 32) % 32
    fout.write(b"\x00" * pad)

    fout.write(payload)


# ── Tokenizer embedding ──────────────────────────────────────────────────────

TOK_MAGIC = 0x746F6B00   # "tok\0"

def write_tokenizer(fout, tokenizer_dir: str):
    """Embed BPE tokenizer (vocab + merges) into the model file."""
    import json

    vocab_path  = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.txt")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

    # Sentinel
    fout.write(struct.pack("<I", TOK_MAGIC))

    # Vocab (sorted by token_id for determinism)
    fout.write(struct.pack("<i", len(vocab)))
    for token_str, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        token_bytes = token_str.encode("utf-8")
        fout.write(struct.pack("<i", len(token_bytes)))
        fout.write(token_bytes)
        fout.write(struct.pack("<i", token_id))

    # Merges
    fout.write(struct.pack("<i", len(merges)))
    for a, b in merges:
        a_bytes = a.encode("utf-8")
        b_bytes = b.encode("utf-8")
        fout.write(struct.pack("<i", len(a_bytes)))
        fout.write(a_bytes)
        fout.write(struct.pack("<i", len(b_bytes)))
        fout.write(b_bytes)

    print(f"Embedded tokenizer: {len(vocab)} vocab entries, {len(merges)} merges")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert SAM3 checkpoint to ggml format")
    parser.add_argument("--model",  required=True, help="Path to sam3.pt")
    parser.add_argument("--output", help="Output .ggml path")
    parser.add_argument("--ftype",  type=int, default=1, choices=[FTYPE_F32, FTYPE_F16, FTYPE_BF16],
                        help="0=f32, 1=f16 (default), 24=bf16")
    parser.add_argument("--visual-only", action="store_true",
                        help="Strip text encoder and detector-only components")
    parser.add_argument("--sam31", action="store_true",
                        help="Treat input as SAM3.1 Object Multiplex. Conversion is not implemented yet.")
    parser.add_argument("--sam31-mask-decoder-only", action="store_true",
                        help="For SAM3.1, write only tracker.model.sam_mask_decoder tensors.")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Print checkpoint summary and exit without writing a ggml file")
    parser.add_argument("--inspect-json",
                        help="Write a machine-readable checkpoint inventory JSON")
    parser.add_argument("--tokenizer", default=None,
                        help="Directory containing vocab.json + merges.txt "
                             "(default: same directory as --model)")
    args = parser.parse_args()
    if not args.inspect_only and not args.inspect_json and not args.output:
        parser.error("--output is required unless --inspect-only or --inspect-json is used")

    import torch

    print(f"Loading checkpoint: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)

    # Handle nested {"model": {...}} format
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    print_checkpoint_summary(ckpt)
    is_sam31 = detect_sam31_checkpoint(ckpt.keys())
    if args.inspect_json:
        with open(args.inspect_json, "w", encoding="utf-8") as fout:
            json.dump(checkpoint_inventory(ckpt), fout, indent=2, sort_keys=True)
            fout.write("\n")
        print(f"Wrote checkpoint inventory JSON: {args.inspect_json}")
    if args.inspect_only:
        return
    convert_as_sam31 = args.sam31 or is_sam31
    if convert_as_sam31:
        print(
            "SAM3.1 Object Multiplex conversion enabled: writing v4 model_type=sam3.1 "
            "with native tracker tensor names. Full C++ graph execution is still being added.",
            file=sys.stderr,
        )

    # ── First pass: rename keys, skip unwanted tensors ────────────────────
    renamed = {}
    skipped = []
    for k, v in ckpt.items():
        new_name = rename_key(k)
        if new_name is None:
            skipped.append(k)
            continue
        if isinstance(v, torch.Tensor):
            # Complex tensors (e.g., freqs_cis): convert to real pairs via view_as_real
            # [N, D] complex64 → [N, D, 2] float32 (re, im interleaved in last dim)
            if v.is_complex():
                v = torch.view_as_real(v).contiguous()
            data = v.numpy()
        else:
            data = v
        # vit.pos_embed: checkpoint stores [1, 577, 1024] (576 spatial + 1 cls token).
        # The C++ loader expects [24, 24, 1024] (spatial grid only, no cls token).
        # Strip the cls token and reshape to the pretrained spatial grid.
        if new_name == "vit.pos_embed" and isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[1] == 577:
                grid = int(np.sqrt(data.shape[1] - 1))
                assert grid * grid == data.shape[1] - 1, (
                    f"pos_embed spatial tokens ({data.shape[1]-1}) is not a perfect square"
                )
                data = data[:, 1:, :]             # [1, 576, 1024]
                data = data.reshape(grid, grid, -1)  # [24, 24, 1024]
                print(f"  vit.pos_embed: stripped cls token, reshaped to {list(data.shape)}")

        renamed[new_name] = data

    print(f"Kept:    {len(renamed)} tensors")
    print(f"Skipped: {len(skipped)} tensors")
    if skipped:
        print("  First 10 skipped:")
        for s in skipped[:10]:
            print(f"    {s}")

    if args.sam31_mask_decoder_only:
        if not convert_as_sam31:
            parser.error("--sam31-mask-decoder-only requires --sam31 or a SAM3.1 checkpoint")
        full_count = len(renamed)
        renamed = {
            name: data
            for name, data in renamed.items()
            if name.startswith("trk.model.sam_mask_decoder.")
        }
        print(
            f"\n--sam31-mask-decoder-only: kept {len(renamed)} of {full_count} "
            "SAM3.1 tensors"
        )

    # ── Visual-only filtering ─────────────────────────────────────────────
    if args.visual_only:
        full_count = len(renamed)
        stripped = {k: v for k, v in renamed.items()
                    if k.startswith(VISUAL_ONLY_STRIP_PREFIXES)}
        renamed = {k: v for k, v in renamed.items()
                   if not k.startswith(VISUAL_ONLY_STRIP_PREFIXES)}
        print(f"\n--visual-only: stripped {len(stripped)} detector tensors "
              f"({full_count} → {len(renamed)})")
        if stripped:
            print("  First 10 stripped:")
            for s in list(stripped.keys())[:10]:
                print(f"    {s}")

        # Validate required tracker prefixes
        for pfx in VISUAL_ONLY_REQUIRED_PREFIXES:
            if not any(k.startswith(pfx) for k in renamed):
                print(f"  WARNING: no tensors with required prefix '{pfx}'")

    # ── Write ─────────────────────────────────────────────────────────────
    print(f"\nWriting {args.output} (ftype={args.ftype}) ...")

    with open(args.output, "wb") as fout:
        write_header(
            fout,
            args.ftype,
            len(renamed),
            visual_only=args.visual_only or convert_as_sam31,
            model_type=MODEL_TYPE_SAM31 if convert_as_sam31 else MODEL_TYPE_SAM3,
        )

        for i, (name, data) in enumerate(renamed.items()):
            write_tensor(fout, name, data, args.ftype)
            if (i + 1) % 100 == 0 or i == len(renamed) - 1:
                print(f"  [{i+1}/{len(renamed)}] {name}  {list(data.shape)}")

        # Embed tokenizer for non-visual-only SAM3 models.
        if not args.visual_only and not convert_as_sam31:
            tok_dir = args.tokenizer if args.tokenizer else os.path.dirname(os.path.abspath(args.model))
            write_tokenizer(fout, tok_dir)

    file_size = os.path.getsize(args.output)
    print(f"\nDone. {len(renamed)} tensors, {file_size / 1e9:.2f} GB")


# ── Listing mode (no conversion, just prints keys) ───────────────────────────

def list_keys():
    """Quick utility: python convert_sam3_to_ggml.py --list --model sam3.pt"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not args.list:
        return False

    import torch
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    for k, v in sorted(ckpt.items()):
        shape = list(v.shape) if hasattr(v, "shape") else "?"
        new = rename_key(k)
        tag = "SKIP" if new is None else new
        print(f"{k:100s}  {str(shape):30s}  → {tag}")
    return True


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_keys()
    else:
        main()
