#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
#   "pillow>=11.0",
# ]
# ///
"""Run SAM3.1 mask-init smoke across CUDA kernel variants and resolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SAM31_VIT_WINDOW_LATE_GLOBAL_FATTN_MATCH = (
    "_window_attn,"
    "sam3_vit_block_15_global_attn,"
    "sam3_vit_block_23_global_attn,"
    "sam3_vit_block_31_global_attn"
)


FIXED_VARIANTS: dict[str, dict[str, str | None]] = {
    "default": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_DISABLE_MATERIALIZED_FUSED_MEM_ATTN_SA_QKV": "0",
    },
    "legacy-default": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_DISABLE_MATERIALIZED_FUSED_MEM_ATTN_SA_QKV": "1",
    },
    "streamk": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": "1",
    },
    "no-l0sa-tile": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_SAM31_MEM_ATTN_L0_SA_TILE": "1",
    },
    "no-l0sa-tile-ncols8": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_SAM31_MEM_ATTN_L0_SA_TILE": "1",
        "GGML_CUDA_FATTN32_MMA_NCOLS1": "8",
        "GGML_CUDA_FATTN32_MMA_NCOLS1_MATCH": "sam31_mem_attn_layer0_sa_fattn",
    },
    "no-l0sa-tile-ncols16": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_SAM31_MEM_ATTN_L0_SA_TILE": "1",
        "GGML_CUDA_FATTN32_MMA_NCOLS1": "16",
        "GGML_CUDA_FATTN32_MMA_NCOLS1_MATCH": "sam31_mem_attn_layer0_sa_fattn",
    },
    "no-l0sa-tile-ncols32": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_SAM31_MEM_ATTN_L0_SA_TILE": "1",
        "GGML_CUDA_FATTN32_MMA_NCOLS1": "32",
        "GGML_CUDA_FATTN32_MMA_NCOLS1_MATCH": "sam31_mem_attn_layer0_sa_fattn",
    },
    "all": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": "all",
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
    },
    "mux-self-mma": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_FORCE_FATTN_MMA_MATCH": (
            "sam31_mux_block0_self_attn_fattn,"
            "sam31_mux_block1_self_attn_fattn"
        ),
    },
    "mux-self0-mma": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_FORCE_FATTN_MMA_MATCH": "sam31_mux_block0_self_attn_fattn",
    },
    "mux-self1-mma": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_FORCE_FATTN_MMA_MATCH": "sam31_mux_block1_self_attn_fattn",
    },
    "sa123": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": (
            "sam31-ca+matches:"
            "sam31_mem_attn_layer1_sa_fattn,"
            "sam31_mem_attn_layer2_sa_fattn,"
            "sam31_mem_attn_layer3_sa_fattn"
        ),
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
    },
    "sa123-streamk": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": (
            "sam31-ca+matches:"
            "sam31_mem_attn_layer1_sa_fattn,"
            "sam31_mem_attn_layer2_sa_fattn,"
            "sam31_mem_attn_layer3_sa_fattn"
        ),
        "GGML_CUDA_FORCE_FATTN_STREAM_K": "1",
    },
    "static-kv": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_EXPERIMENTAL_STATIC_MEM_CA_KV_CACHE": "1",
    },
    "no-kv-cache": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_DISABLE_MEM_CA_KV_CACHE": "1",
    },
    "mem-attn-input-boundary": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_BF16_MEM_ATTN_INPUT_BOUNDARY": "1",
    },
    "f16-mem-attn-activation": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_F16_MEM_ATTN_ACTIVATION": "1",
    },
    "f16-mem-attn-attention-activation": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_F16_MEM_ATTN_ATTENTION_ACTIVATION": "1",
    },
    "f16-attn-cudnn-head32-contig-v": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_F16_MEM_ATTN_ATTENTION_ACTIVATION": "1",
        "SAM31_MEM_ATTN_CONTIGUOUS_V": "1",
        "GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32": "1",
        "GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32_UNSAFE_RUN": "1",
    },
    "f16-mem-attn-ffn-activation": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_F16_MEM_ATTN_FFN_ACTIVATION": "1",
    },
    "f16-mem-attn-activation-contig-v": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_F16_MEM_ATTN_ACTIVATION": "1",
        "SAM31_MEM_ATTN_CONTIGUOUS_V": "1",
    },
    "fused-sa-qkv": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_FUSED_MEM_ATTN_SA_QKV": "1",
    },
    "direct-fused-sa-qkv": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_DIRECT_FUSED_MEM_ATTN_SA_QKV": "1",
    },
    "materialized-fused-sa-qkv": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_DISABLE_MATERIALIZED_FUSED_MEM_ATTN_SA_QKV": "0",
        "SAM31_ENABLE_MATERIALIZED_FUSED_MEM_ATTN_SA_QKV": "1",
    },
    "prop-feature-boundary": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_BF16_PROPAGATION_FEATURE_BOUNDARY": "1",
    },
    "prop-alias-feature-inputs": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_PROP_ALIAS_FEATURE_INPUTS": "1",
    },
    "prop-graph-cache": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_PROP_GRAPH_CACHE": "1",
        "SAM3_CUDA_ENABLE_GRAPHS": "1",
    },
    "prop-graph-cache-no-cuda-graphs": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_PROP_GRAPH_CACHE": "1",
    },
    "cuda-graphs": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_CUDA_ENABLE_GRAPHS": "1",
    },
    "unary-cpy-packed": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_UNARY_CPY_VEC4_PACKED": "1",
    },
    "no-unary-cpy-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_UNARY_CPY_FUSION": "1",
    },
    "no-add-norm-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_ADD_NORM_FUSION": "1",
    },
    "no-add-norm-preserve-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_ADD_NORM_PRESERVE_FUSION": "1",
    },
    "mixed-add-norm-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_MIXED_ADD_NORM_FUSION": "1",
    },
    "norm-affine-cpy-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_NORM_AFFINE_CPY_FUSION": "1",
    },
    "no-norm-affine-cpy-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_NORM_AFFINE_CPY_FUSION": "1",
    },
    "direct-mem-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_PROP_DIRECT_MEM_OUTPUT": "1",
    },
    "direct-final-queries": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_MUX_DIRECT_FINAL_QUERIES": "1",
    },
    "direct-copy-prune": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_PROP_DIRECT_MEM_OUTPUT": "1",
        "SAM31_ENABLE_MUX_DIRECT_FINAL_QUERIES": "1",
    },
    "backend-memory-slot": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_BACKEND_MEMORY_SLOT": "1",
    },
    "cpu-memory-slot": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_DISABLE_BACKEND_MEMORY_SLOT": "1",
    },
    "neck-1x1-mulmat": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_ENABLE_NECK_1X1_MULMAT": "1",
    },
    "conv-transpose-bf16": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_CONV_TRANSPOSE_KEEP_BF16": "1",
    },
    "no-direct-neck-3x3": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_DISABLE_DIRECT_NECK_3X3_CONV": "1",
    },
    "cublaslt-autotune": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_CUBLASLT_BIAS_AUTOTUNE": "1",
    },
    "no-cudnn-conv2d": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_CUDNN_CONV2D": "1",
    },
    "cudnn-conv2d-f32-lowp": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_CUDNN_CONV2D_F32_LOWP": "1",
    },
    "no-conv-transpose-k2s2-gemm": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_CONV_TRANSPOSE_K2S2_GEMM": "0",
    },
    "no-conv-transpose-k2s2": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_CONV_TRANSPOSE_K2S2": "0",
    },
    "vit-linear-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_LINEAR_OUTPUT": "1",
    },
    "f16-vit-linear-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_F16_VIT_LINEAR_OUTPUT": "1",
    },
    "f16-vit-attn-linear-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_F16_VIT_ATTENTION_LINEAR_OUTPUT": "1",
    },
    "f16-vit-mlp-linear-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_F16_VIT_MLP_LINEAR_OUTPUT": "1",
    },
    "f16-vit-mlp-fc1-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_F16_VIT_MLP_FC1_OUTPUT": "1",
    },
    "f16-vit-mlp-fc2-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_F16_VIT_MLP_FC2_OUTPUT": "1",
    },
    "no-vit-linear-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_DISABLE_BF16_VIT_LINEAR_OUTPUT": "1",
    },
    "no-vit-linear-output-b9-31": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_DISABLE_BF16_VIT_LINEAR_OUTPUT_BLOCKS": "9-31",
    },
    "no-vit-linear-output-b12-31": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_DISABLE_BF16_VIT_LINEAR_OUTPUT_BLOCKS": "12-31",
    },
    "no-vit-fc2-input": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_DISABLE_BF16_VIT_FC2_INPUT": "1",
    },
    "no-vit-linear-inputs": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_DISABLE_BF16_VIT_LINEAR_INPUTS": "1",
    },
    "vit-qkv-chain": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN": "1",
    },
    "vit-mlp-chain": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_MLP_CHAIN": "1",
    },
    "vit-mlp-flat-chain": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_ENABLE_VIT_MLP_FLAT_CHAIN": "1",
    },
    "vit-direct-qkv-views": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_ENABLE_VIT_DIRECT_QKV_VIEWS": "1",
    },
    "no-vit-direct-qkv-views": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_DISABLE_VIT_DIRECT_QKV_VIEWS": "1",
    },
    "vit-contiguous-attention-v": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_ENABLE_VIT_CONTIGUOUS_ATTENTION_V": "1",
    },
    "bf16-vit-win-part-input": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_WIN_PART_INPUT": "1",
    },
    "vit-qkv-mlp-chain": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN": "1",
        "SAM3_BF16_VIT_MLP_CHAIN": "1",
    },
    "vit-qkv-chain-b0": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN_BLOCKS": "0",
    },
    "vit-qkv-chain-b0-qkv-bf16-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN_BLOCKS": "0",
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV_BF16_DST": "1",
    },
    "vit-qkv-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV": "1",
    },
    "vit-attn-proj-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_ATTN_PROJ": "1",
    },
    "vit-mlp-fc1-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1": "1",
    },
    "vit-mlp-fc2-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2": "1",
    },
    "cublaslt-gelu-erf": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION": "1",
    },
    "cublaslt-bias-residual": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_CUBLASLT_BIAS_RESIDUAL_FUSION": "1",
    },
    "cublaslt-gelu-erf-residual": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION": "1",
        "GGML_CUDA_ENABLE_CUBLASLT_BIAS_RESIDUAL_FUSION": "1",
    },
    "cudnn-mlp-bf16": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16": "1",
        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16_UNSAFE_RUN": "1",
    },
    "cudnn-mlp-f16-f32": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_F32": "1",
        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_F32_UNSAFE_RUN": "1",
    },
    "vit-mlp-chain-b0": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_MLP_CHAIN_BLOCKS": "0",
    },
    "vit-mlp-chain-b0-fc1-bf16-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_MLP_CHAIN_BLOCKS": "0",
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1_BF16_DST": "1",
    },
    "vit-qkv-mlp-chain-b0": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN_BLOCKS": "0",
        "SAM3_BF16_VIT_MLP_CHAIN_BLOCKS": "0",
    },
    "vit-qkv-chain-b0-7": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN_BLOCKS": "0-7",
    },
    "vit-qkv-chain-b0-7-qkv-bf16-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN_BLOCKS": "0-7",
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV_BF16_DST": "1",
    },
    "vit-mlp-chain-b0-7": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_MLP_CHAIN_BLOCKS": "0-7",
    },
    "vit-mlp-chain-b0-7-fc1-bf16-algo1": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_MLP_CHAIN_BLOCKS": "0-7",
        "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1_BF16_DST": "1",
    },
    "vit-qkv-mlp-chain-b0-7": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_QKV_CHAIN_BLOCKS": "0-7",
        "SAM3_BF16_VIT_MLP_CHAIN_BLOCKS": "0-7",
    },
    "vit-attn-proj-output-global": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_ATTN_PROJ_OUTPUT": "global",
    },
    "vit-attn-proj-output-all": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_BF16_VIT_ATTN_PROJ_OUTPUT": "all",
    },
    "vit-pos-cache": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM3_ENABLE_VIT_POS_CACHE": "1",
    },
    "win-part-cpy-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION": "1",
    },
    "win-unpart-add-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_WIN_UNPART_ADD_FUSION": "1",
    },
    "win-window-fusions": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION": "1",
        "GGML_CUDA_ENABLE_WIN_UNPART_ADD_FUSION": "1",
    },
    "no-win-part-cpy-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_WIN_PART_CPY_FUSION": "1",
    },
    "no-win-unpart-add-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_WIN_UNPART_ADD_FUSION": "1",
    },
    "no-win-window-fusions": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_WIN_PART_CPY_FUSION": "1",
        "GGML_CUDA_DISABLE_WIN_UNPART_ADD_FUSION": "1",
    },
    "cuda-bicubic-preprocess": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_ENABLE_CUDA_BICUBIC_PREPROCESS": "1",
    },
    "no-cuda-bicubic-preprocess": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "SAM31_DISABLE_CUDA_BICUBIC_PREPROCESS": "1",
    },
    "vit-window-late-global-tile": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_FORCE_FATTN_TILE_MATCH": SAM31_VIT_WINDOW_LATE_GLOBAL_FATTN_MATCH,
    },
    "vit-window-late-global-tile-linear-output": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_FORCE_FATTN_TILE_MATCH": SAM31_VIT_WINDOW_LATE_GLOBAL_FATTN_MATCH,
        "SAM3_BF16_VIT_LINEAR_OUTPUT": "1",
    },
    "no-fattn64-nomask-fast": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_FATTN64_NOMASK_FAST": "1",
    },
    "no-rope-pair-exact-inplace": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_ROPE_PAIR_EXACT_INPLACE": "1",
    },
    "no-rope-pair-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_ROPE_PAIR_FUSION": "1",
    },
    "no-cont-rope-pair-fusion": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_CONT_ROPE_PAIR_FUSION": "1",
    },
    "no-unary-cpy-vec2": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_UNARY_CPY_VEC2": "1",
    },
    "no-rope-pair-f32-vec2": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_ROPE_PAIR_F32_VEC2": "1",
    },
    "no-cpy-rows64-row4": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_CPY_ROWS64_VEC4_ROW4": "1",
    },
    "no-cpy-row-group": {
        "GGML_CUDA_ENABLE_FATTN32_MMA": None,
        "GGML_CUDA_FORCE_FATTN_STREAM_K": None,
        "GGML_CUDA_DISABLE_CPY_VEC4_ROW_GROUP": "1",
    },
}

VARIANT_ENV_KEYS = frozenset(key for spec in FIXED_VARIANTS.values() for key in spec)


def layer_variant_env(name: str) -> dict[str, str] | None:
    suffix = ""
    base = name
    if name.endswith("-streamk"):
        suffix = "-streamk"
        base = name.removesuffix("-streamk")
    if not base.startswith("sa"):
        return None
    layers = base.removeprefix("sa")
    if not layers or any(ch not in "0123" for ch in layers) or len(set(layers)) != len(layers):
        return None
    matches = ",".join(f"sam31_mem_attn_layer{layer}_sa_fattn" for layer in layers)
    env = {"GGML_CUDA_ENABLE_FATTN32_MMA": f"sam31-ca+matches:{matches}"}
    if suffix:
        env["GGML_CUDA_FORCE_FATTN_STREAM_K"] = "1"
    return env


def validate_variant(name: str) -> str:
    if name in FIXED_VARIANTS or layer_variant_env(name) is not None:
        return name
    raise argparse.ArgumentTypeError(
        f"unknown variant {name}; use default, legacy-default, all, static-kv, no-kv-cache, "
        "streamk, no-l0sa-tile, no-l0sa-tile-ncols8, no-l0sa-tile-ncols16, "
        "no-l0sa-tile-ncols32, mem-attn-input-boundary, f16-mem-attn-activation, "
        "mux-self-mma, mux-self0-mma, mux-self1-mma, "
        "f16-mem-attn-attention-activation, f16-attn-cudnn-head32-contig-v, "
        "f16-mem-attn-ffn-activation, "
        "f16-mem-attn-activation-contig-v, fused-sa-qkv, direct-fused-sa-qkv, "
        "materialized-fused-sa-qkv, prop-feature-boundary, "
        "prop-alias-feature-inputs, "
        "direct-mem-output, direct-final-queries, direct-copy-prune, backend-memory-slot, "
        "neck-1x1-mulmat, conv-transpose-bf16, no-direct-neck-3x3, "
        "cublaslt-autotune, no-cudnn-conv2d, cudnn-conv2d-f32-lowp, "
        "no-conv-transpose-k2s2-gemm, no-conv-transpose-k2s2, "
        "vit-linear-output, f16-vit-linear-output, f16-vit-attn-linear-output, "
        "f16-vit-mlp-linear-output, f16-vit-mlp-fc1-output, f16-vit-mlp-fc2-output, "
        "no-vit-linear-output, no-vit-linear-output-b9-31, "
        "no-vit-linear-output-b12-31, no-vit-fc2-input, no-vit-linear-inputs, "
        "vit-qkv-chain, vit-mlp-chain, "
        "vit-mlp-flat-chain, vit-direct-qkv-views, no-vit-direct-qkv-views, "
        "vit-contiguous-attention-v, bf16-vit-win-part-input, "
        "vit-qkv-mlp-chain, "
        "vit-qkv-algo1, vit-attn-proj-algo1, vit-mlp-fc1-algo1, vit-mlp-fc2-algo1, "
        "cublaslt-gelu-erf, cublaslt-bias-residual, cublaslt-gelu-erf-residual, "
        "cudnn-mlp-bf16, cudnn-mlp-f16-f32, "
        "vit-attn-proj-output-global, vit-attn-proj-output-all, "
        "vit-pos-cache, "
        "win-part-cpy-fusion, win-unpart-add-fusion, win-window-fusions, "
        "no-win-part-cpy-fusion, no-win-unpart-add-fusion, no-win-window-fusions, "
        "cuda-bicubic-preprocess, no-cuda-bicubic-preprocess, "
        "vit-window-late-global-tile, "
        "vit-window-late-global-tile-linear-output, "
        "no-fattn64-nomask-fast, "
        "no-rope-pair-exact-inplace, no-rope-pair-fusion, "
        "no-cont-rope-pair-fusion, "
        "unary-cpy-packed, no-unary-cpy-fusion, no-unary-cpy-vec2, "
        "no-add-norm-fusion, no-add-norm-preserve-fusion, mixed-add-norm-fusion, "
        "norm-affine-cpy-fusion, no-norm-affine-cpy-fusion, "
        "no-rope-pair-f32-vec2, "
        "no-cpy-rows64-row4, "
        "no-cpy-row-group, "
        "sa<layers>, or sa<layers>-streamk"
    )


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class Case:
    mask_case: str
    frame1_offset: int

    @property
    def label(self) -> str:
        return f"{self.mask_case}-off{self.frame1_offset}"


def parse_size(text: str) -> Size:
    if "x" not in text:
        raise argparse.ArgumentTypeError(f"size must be WIDTHxHEIGHT: {text}")
    width_text, height_text = text.split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"size must be WIDTHxHEIGHT: {text}") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive: {text}")
    return Size(width, height)


def parse_case(text: str) -> Case:
    if "@" in text:
        mask_case, offset_text = text.split("@", 1)
    else:
        mask_case, offset_text = text, "3"
    if mask_case not in {"center", "small-center", "left-wide", "bottom-band"}:
        raise argparse.ArgumentTypeError(f"unknown mask case: {mask_case}")
    try:
        frame1_offset = int(offset_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid frame1 offset: {text}") from exc
    if frame1_offset < 0:
        raise argparse.ArgumentTypeError(f"frame1 offset must be non-negative: {text}")
    return Case(mask_case=mask_case, frame1_offset=frame1_offset)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()


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


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "stdev": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def nested_number(item: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = item
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, int | float) else None


def summary_values(runs: list[dict[str, Any]], *path: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        summary = run.get("summary")
        if not isinstance(summary, dict):
            continue
        value = nested_number(summary, tuple(path))
        if value is not None:
            values.append(value)
    return values


def summary_array_values(runs: list[dict[str, Any]], *path: str) -> list[list[float]]:
    values: list[list[float]] = []
    for run in runs:
        summary = run.get("summary")
        value: Any = summary
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if not isinstance(value, list):
            continue
        row = [float(item) for item in value if isinstance(item, int | float)]
        values.append(row)
    return values


def numeric_summary_by_index(rows: list[list[float]]) -> list[dict[str, float | int | None]]:
    max_len = max((len(row) for row in rows), default=0)
    return [
        numeric_summary([row[index] for row in rows if index < len(row)])
        for index in range(max_len)
    ]


def sum_components(rows: list[tuple[float, ...]]) -> list[float]:
    return [sum(row) for row in rows if all(value is not None for value in row)]


def python_timing_contract(summary: dict[str, Any]) -> dict[str, float | str | None]:
    result = summary.get("result", {})
    result = result if isinstance(result, dict) else {}

    def result_float(name: str) -> float | None:
        value = result.get(name)
        return float(value) if isinstance(value, int | float) else None

    input_prepare = result_float("input_prepare_ms")
    init_state = result_float("init_state_ms")
    mask_tensor_prepare = result_float("mask_tensor_prepare_ms")
    frame0_cache = result_float("frame0_cache_ms")
    add_mask = result_float("add_mask_ms")
    frame1_cache = result_float("frame1_cache_ms")
    propagate = result_float("propagate_ms")
    required_e2e = result_float("required_e2e_ms")
    required_session_setup = result_float("required_session_setup_ms")
    required_input_prepare = result_float("required_input_prepare_ms")
    required_model_execute = result_float("required_model_execute_ms")
    required_backbone = result_float("required_image_encode_or_backbone_ms")
    required_mask_init = result_float("required_mask_init_ms")
    required_propagate = result_float("required_propagate_encoded_ms")
    required_accounted = result_float("required_accounted_ms")
    required_remainder = result_float("required_remainder_ms")
    full_inputs = [frame0_cache, add_mask, frame1_cache, propagate]
    full_cache_step = sum(value for value in full_inputs if value is not None)
    if any(value is None for value in full_inputs):
        full_cache_step = None
    if required_backbone is None and frame0_cache is not None and frame1_cache is not None:
        required_backbone = frame0_cache + frame1_cache
    if required_model_execute is None and full_cache_step is not None:
        required_model_execute = full_cache_step
    if required_session_setup is None:
        required_session_setup = init_state
    if required_input_prepare is None:
        input_parts = [input_prepare, mask_tensor_prepare]
        required_input_prepare = (
            sum(value for value in input_parts if value is not None)
            if all(value is not None for value in input_parts)
            else None
        )
    if required_mask_init is None:
        required_mask_init = add_mask
    if required_propagate is None:
        required_propagate = propagate
    if (
        required_accounted is None
        and required_session_setup is not None
        and required_input_prepare is not None
        and required_model_execute is not None
    ):
        required_accounted = (
            required_session_setup + required_input_prepare + required_model_execute
        )
    if required_e2e is None:
        required_e2e = required_accounted
    if required_remainder is None and required_e2e is not None and required_accounted is not None:
        required_remainder = required_e2e - required_accounted
    return {
        "input_prepare_ms": input_prepare,
        "init_state_ms": init_state,
        "mask_tensor_prepare_ms": mask_tensor_prepare,
        "frame0_cache_ms": frame0_cache,
        "add_mask_ms": add_mask,
        "frame1_cache_ms": frame1_cache,
        "cached_propagate_ms": propagate,
        "full_cache_step_ms": full_cache_step,
        "required_session_setup_ms": required_session_setup,
        "required_input_prepare_ms": required_input_prepare,
        "required_model_execute_ms": required_model_execute,
        "required_image_encode_or_backbone_ms": required_backbone,
        "required_mask_init_ms": required_mask_init,
        "required_propagate_encoded_ms": required_propagate,
        "required_accounted_ms": required_accounted,
        "required_e2e_ms": required_e2e,
        "required_remainder_ms": required_remainder,
        "note": (
            "Official Python prefetches the next frame in the frame0 cache step on the "
            "two-frame, single-GPU contract, so compare this full_cache_step_ms with "
            "C++ full_frame_step_ms. For required E2E, compare required_e2e_ms with "
            "C++ run_required_e2e_ms; it includes Python init_state, input construction, "
            "mask tensor preparation, backbone/cache, mask init, and cached propagation."
        ),
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str | None] | None,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        for key, value in env.items():
            if value is None:
                merged_env.pop(key, None)
            else:
                merged_env[key] = value
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(command)}")


def variant_env(name: str) -> dict[str, str | None]:
    env: dict[str, str | None] = {key: None for key in VARIANT_ENV_KEYS}
    dynamic = layer_variant_env(name)
    if dynamic is not None:
        env.update(dynamic)
        return env
    spec = FIXED_VARIANTS[name]
    env.update(spec)
    return env


def cpp_tf32_policy_env(tf32: str) -> dict[str, str | None]:
    if tf32 == "off":
        return {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1"}
    return {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": None}


def run_python_reference(
    args: argparse.Namespace, size: Size, case: Case, root: Path, case_dir: Path
) -> dict[str, Any]:
    py_dir = case_dir / "python"
    command = [
        "uv",
        "run",
        "scripts/run_sam31_mask_sequence_python.py",
        "--repo-root",
        ".",
        "--sam3-repo",
        str(args.sam3_repo),
        "--checkpoint",
        str(args.checkpoint),
        "--out",
        str(py_dir),
        "--width",
        str(size.width),
        "--height",
        str(size.height),
        "--mask-case",
        case.mask_case,
        "--frame1-offset",
        str(case.frame1_offset),
        "--dtype",
        args.python_dtype,
        "--tf32",
        args.tf32,
        "--warmup-runs",
        str(args.python_warmup_runs),
    ]
    run_command(command, cwd=root, env=None, log_path=py_dir / "run.log")
    summary = read_json(py_dir / "summary.json")
    mask = read_mask(py_dir / "frame1_mask.png")
    return {
        "dir": rel(py_dir, root),
        "summary": summary,
        "mask_sha256": mask_sha256(mask),
        "mask_foreground_pixels": int((mask > 127).sum()),
    }


def run_cpp_variant(
    args: argparse.Namespace,
    size: Size,
    case: Case,
    root: Path,
    case_dir: Path,
    variant: str,
    repeat: int,
) -> dict[str, Any]:
    run_dir = case_dir / variant / f"r{repeat:02d}"
    env = variant_env(variant)
    env.update(cpp_tf32_policy_env(args.tf32))
    command = [
        str(args.cpp_exe),
        "--model",
        str(args.model),
        "--threads",
        str(args.threads),
        "--gpu" if args.device == "gpu" else "--cpu",
        "--width",
        str(size.width),
        "--height",
        str(size.height),
        "--mask-case",
        case.mask_case,
        "--frame1-offset",
        str(case.frame1_offset),
        "--warmup-runs",
        str(args.cpp_warmup_runs),
        "--out",
        str(run_dir),
    ]
    run_command(command, cwd=root, env=env, log_path=run_dir / "run.log")
    summary = read_json(run_dir / "summary.json")
    mask = read_mask(run_dir / "frame1_mask.png")
    return {
        "dir": rel(run_dir, root),
        "summary": summary,
        "mask_sha256": mask_sha256(mask),
        "mask_foreground_pixels": int((mask > 127).sum()),
    }


def profile_stage_rows(stage_profile: dict[str, Any], *, top: int) -> list[dict[str, Any]]:
    by_mode_stage = stage_profile.get("by_mode_stage")
    if not isinstance(by_mode_stage, dict):
        return []
    rows: list[dict[str, Any]] = []
    for mode, stages in by_mode_stage.items():
        if not isinstance(stages, dict):
            continue
        for stage, item in stages.items():
            if not isinstance(item, dict):
                continue
            top_ops = item.get("top_ops")
            top_ops = top_ops if isinstance(top_ops, list) else []
            rows.append(
                {
                    "mode": str(mode),
                    "stage": str(stage),
                    "count": item.get("count"),
                    "sum_ms": item.get("sum_ms"),
                    "mean_ms": item.get("mean_ms"),
                    "mean_ms_drop_max": item.get("mean_ms_drop_max"),
                    "top_ops": [
                        {
                            "op": op.get("op"),
                            "sum_ms": op.get("sum_ms"),
                            "mean_ms": op.get("mean_ms"),
                        }
                        for op in top_ops[:3]
                        if isinstance(op, dict)
                    ],
                }
            )
    return sorted(
        rows,
        key=lambda row: float(row.get("mean_ms_drop_max") or row.get("mean_ms") or 0.0),
        reverse=True,
    )[:top]


def run_cpp_profile_default(
    args: argparse.Namespace,
    size: Size,
    case: Case,
    root: Path,
    case_dir: Path,
) -> dict[str, Any]:
    profile_dir = case_dir / "profile" / "default"
    run_dir = profile_dir / "run"
    env = variant_env("default")
    env.update(cpp_tf32_policy_env(args.tf32))
    env.update({"SAM3_PROFILE": "1", "GGML_CUDA_PROFILE_NODES": "1"})
    command = [
        str(args.cpp_exe),
        "--model",
        str(args.model),
        "--threads",
        str(args.threads),
        "--gpu" if args.device == "gpu" else "--cpu",
        "--width",
        str(size.width),
        "--height",
        str(size.height),
        "--mask-case",
        case.mask_case,
        "--frame1-offset",
        str(case.frame1_offset),
        "--warmup-runs",
        str(args.profile_warmup_runs),
        "--out",
        str(run_dir),
    ]
    profile_log = profile_dir / "profile.log"
    run_command(command, cwd=root, env=env, log_path=profile_log)

    stage_profile_path = profile_dir / "stage_profile.json"
    node_names_path = profile_dir / "node_names_split.json"
    run_command(
        [
            "uv",
            "run",
            "--no-project",
            "scripts/summarize_cuda_stage_profile.py",
            str(profile_log),
            "--out",
            str(stage_profile_path),
        ],
        cwd=root,
        env=None,
        log_path=profile_dir / "stage_profile.stdout",
    )
    run_command(
        [
            "uv",
            "run",
            "--no-project",
            "scripts/summarize_cuda_node_names.py",
            str(profile_log),
            "--split-computes",
            "--top",
            str(args.profile_node_top),
            "--out",
            str(node_names_path),
        ],
        cwd=root,
        env=None,
        log_path=profile_dir / "node_names_split.stdout",
    )

    stage_profile = read_json(stage_profile_path)
    return {
        "dir": rel(profile_dir, root),
        "run_dir": rel(run_dir, root),
        "profile_log": rel(profile_log, root),
        "stage_profile": rel(stage_profile_path, root),
        "node_names_split": rel(node_names_path, root),
        "warmup_runs": args.profile_warmup_runs,
        "note": (
            "GGML_CUDA_PROFILE_NODES synchronizes every CUDA node. Use this profile for "
            "stage attribution only; use normal variant runs for E2E speed decisions."
        ),
        "top_stage_rows": profile_stage_rows(stage_profile, top=args.profile_stage_top),
    }


def summarize_variant(
    runs: list[dict[str, Any]],
    *,
    python_mask: np.ndarray,
    default_mask: np.ndarray | None,
    python_timing: dict[str, float | str | None],
    root: Path,
) -> dict[str, Any]:
    propagate_encoded = [float(run["summary"]["propagate_encoded_ms"]) for run in runs]
    propagate_total = [float(run["summary"]["propagate_ms"]) for run in runs]
    encode_frame0 = [float(run["summary"]["encode_frame0_ms"]) for run in runs]
    encode_frame1 = [float(run["summary"]["encode_frame1_ms"]) for run in runs]
    add_detection = [float(run["summary"]["add_detection_ms"]) for run in runs]
    frame0_total = [float(run["summary"]["frame0_ms"]) for run in runs]
    track_step_total = summary_values(runs, "track_step_total_ms")
    track_step_avg = [float(run["summary"]["track_step_avg_ms"]) for run in runs]
    full_frame_step = [frame0 + tail for frame0, tail in zip(frame0_total, track_step_total)]
    tail_image_encode_total = summary_values(runs, "tail_image_encode_total_ms")
    if not tail_image_encode_total:
        tail_image_encode_total = encode_frame1
    tail_image_encode_avg = summary_values(runs, "tail_image_encode_avg_ms")
    if not tail_image_encode_avg:
        tail_image_encode_avg = encode_frame1
    two_frame_encode = [frame0 + frame1 for frame0, frame1 in zip(encode_frame0, encode_frame1)]
    run_required_e2e = summary_values(runs, "run_required_e2e_ms")
    one_shot_required_e2e = summary_values(runs, "one_shot_required_e2e_ms")
    model_load = summary_values(runs, "model_load_ms")
    state_create = summary_values(runs, "state_create_ms")
    tracker_create = summary_values(runs, "tracker_create_ms")
    frame_construct_total = summary_values(runs, "frame_construct_total_ms")
    frame0_construct = summary_values(runs, "frame0_construct_ms")
    frame1_construct = summary_values(runs, "frame1_construct_ms")
    detection_construct = summary_values(runs, "detection_construct_ms")
    output_write = summary_values(runs, "output_write_ms")
    required_session_setup = summary_values(runs, "required_session_setup_ms")
    if not required_session_setup:
        required_session_setup = [
            state + tracker for state, tracker in zip(state_create, tracker_create)
        ]
    required_input_prepare = summary_values(runs, "required_input_prepare_ms")
    if not required_input_prepare:
        required_input_prepare = [
            frames + detection
            for frames, detection in zip(frame_construct_total, detection_construct)
        ]
    required_model_execute = summary_values(runs, "required_model_execute_ms")
    if not required_model_execute:
        required_model_execute = [frame0 + tail for frame0, tail in zip(frame0_total, track_step_total)]
    required_image_encode = summary_values(runs, "required_image_encode_ms")
    if not required_image_encode:
        required_image_encode = [
            frame0 + tail for frame0, tail in zip(encode_frame0, tail_image_encode_total)
        ]
    tail_encode_timing_total = summary_values(runs, "tail_encode_timing_total", "total_ms")
    if not tail_encode_timing_total:
        tail_encode_timing_total = tail_image_encode_total
    tail_encode_graph_compute = summary_values(
        runs, "tail_encode_timing_total", "graph_compute_ms"
    )
    if not tail_encode_graph_compute:
        tail_encode_graph_compute = summary_values(
            runs, "frame1_encode_timing", "graph_compute_ms"
        )
    tail_encode_non_compute = [
        total - graph
        for total, graph in zip(tail_image_encode_total, tail_encode_graph_compute)
    ]
    frame0_encode_graph_compute = summary_values(
        runs, "frame0_encode_timing", "graph_compute_ms"
    )
    required_image_encode_graph_compute = [
        frame0 + tail
        for frame0, tail in zip(frame0_encode_graph_compute, tail_encode_graph_compute)
    ]
    required_image_encode_non_compute = [
        image_encode - graph_compute
        for image_encode, graph_compute in zip(
            required_image_encode,
            required_image_encode_graph_compute,
        )
    ]
    required_mask_init = summary_values(runs, "required_mask_init_ms")
    if not required_mask_init:
        required_mask_init = add_detection
    required_propagate_encoded = summary_values(runs, "required_propagate_encoded_ms")
    if not required_propagate_encoded:
        required_propagate_encoded = summary_values(runs, "propagate_encoded_total_ms")
    required_model_accounted = summary_values(runs, "required_model_accounted_ms")
    if not required_model_accounted:
        required_model_accounted = [
            image_encode + mask_init + prop_encoded
            for image_encode, mask_init, prop_encoded in zip(
                required_image_encode,
                required_mask_init,
                required_propagate_encoded,
            )
        ]
    required_model_remainder = summary_values(runs, "required_model_remainder_ms")
    if not required_model_remainder:
        required_model_remainder = [
            model - accounted
            for model, accounted in zip(required_model_execute, required_model_accounted)
        ]
    propagate_remainder = [
        max(0.0, total - encode - encoded)
        for total, encode, encoded in zip(propagate_total, encode_frame1, propagate_encoded)
    ]
    required_accounted = summary_values(runs, "required_accounted_ms")
    if not required_accounted:
        required_accounted = [
            session + input_prepare + model
            for session, input_prepare, model in zip(
                required_session_setup,
                required_input_prepare,
                required_model_execute,
            )
        ]
    required_remainder = summary_values(runs, "required_remainder_ms")
    if not required_remainder:
        required_remainder = [
            required - accounted
            for required, accounted in zip(run_required_e2e, required_accounted)
        ]
    add_memory_encode_values = summary_values(runs, "add_detection_timing", "memory_encode_ms")
    propagate_graph_compute_values = summary_values(runs, "propagate_timing", "graph_compute_ms")
    propagate_graph_cache_compute_values = summary_values(
        runs, "propagate_timing", "graph_cache_compute_ms"
    )
    required_core_compute = [
        image_graph + add_memory + propagate_graph + propagate_cache_graph
        for image_graph, add_memory, propagate_graph, propagate_cache_graph in zip(
            required_image_encode_graph_compute,
            add_memory_encode_values,
            propagate_graph_compute_values,
            propagate_graph_cache_compute_values,
        )
    ]
    required_non_image_model = [
        mask_init + prop_encoded
        for mask_init, prop_encoded in zip(required_mask_init, required_propagate_encoded)
    ]
    required_non_graph_or_setup = [
        required - core for required, core in zip(run_required_e2e, required_core_compute)
    ]
    add_accounted = [
        mask_prepare + memory_encode + obj_ptr + store
        for mask_prepare, memory_encode, obj_ptr, store in zip(
            summary_values(runs, "add_detection_timing", "mask_prepare_ms"),
            add_memory_encode_values,
            summary_values(runs, "add_detection_timing", "obj_ptr_ms"),
            summary_values(runs, "add_detection_timing", "store_ms"),
        )
    ]
    add_remainder = [
        total - accounted for total, accounted in zip(add_detection, add_accounted)
    ]
    propagate_accounted = [
        sum(parts)
        for parts in zip(
            summary_values(runs, "propagate_timing", "prepare_caches_ms"),
            summary_values(runs, "propagate_timing", "memory_slot_read_ms"),
            summary_values(runs, "propagate_timing", "obj_ptr_read_ms"),
            summary_values(runs, "propagate_timing", "prompt_build_ms"),
            summary_values(runs, "propagate_timing", "memory_prepare_ms"),
            summary_values(runs, "propagate_timing", "ptr_prepare_ms"),
            summary_values(runs, "propagate_timing", "rope_cache_ms"),
            summary_values(runs, "propagate_timing", "rope_k_build_ms"),
            summary_values(runs, "propagate_timing", "graph_build_ms"),
            summary_values(runs, "propagate_timing", "graph_alloc_ms"),
            summary_values(runs, "propagate_timing", "input_upload_ms"),
            summary_values(runs, "propagate_timing", "graph_compute_ms"),
            summary_values(runs, "propagate_timing", "output_read_ms"),
            summary_values(runs, "propagate_timing", "graph_cache_build_ms"),
            summary_values(runs, "propagate_timing", "graph_cache_input_upload_ms"),
            summary_values(runs, "propagate_timing", "graph_cache_compute_ms"),
            summary_values(runs, "propagate_timing", "graph_cache_output_read_ms"),
            summary_values(runs, "propagate_timing", "bbox_from_logits_ms"),
            summary_values(runs, "propagate_timing", "fullmask_active_resize_bbox_ms"),
            summary_values(runs, "propagate_timing", "fullmask_pending_resize_bbox_ms"),
            summary_values(runs, "propagate_timing", "memory_update_ms"),
            summary_values(runs, "propagate_timing", "tracker_update_ms"),
            summary_values(runs, "propagate_timing", "result_build_ms"),
        )
    ]
    propagate_internal_remainder = [
        total - accounted
        for total, accounted in zip(
            summary_values(runs, "propagate_timing", "total_ms"),
            propagate_accounted,
        )
    ]
    scores = [float(run["summary"]["score"]) for run in runs]
    obj_logits = [float(run["summary"]["obj_score_logit"]) for run in runs]
    mask_foreground_pixels = [
        float(run.get("mask_foreground_pixels"))
        for run in runs
        if isinstance(run.get("mask_foreground_pixels"), int | float)
    ]
    selected_mask_indices = summary_values(runs, "selected_mask_index")
    decoder_iou_score_rows = summary_array_values(runs, "decoder_iou_scores")
    full_summary = numeric_summary(full_frame_step)
    encoded_prop_summary = numeric_summary(propagate_encoded)
    python_full = python_timing.get("full_cache_step_ms")
    python_cached_prop = python_timing.get("cached_propagate_ms")
    python_required = python_timing.get("required_e2e_ms")
    python_backbone = python_timing.get("required_image_encode_or_backbone_ms")
    python_comp = []
    default_comp = []
    for run in runs:
        mask = read_mask(root / run["dir"] / "frame1_mask.png")
        python_comp.append(mask_iou(mask, python_mask))
        if default_mask is not None:
            default_comp.append(mask_iou(mask, default_mask))
    return {
        "runs": runs,
        "model_load_ms": numeric_summary(model_load),
        "state_create_ms": numeric_summary(state_create),
        "tracker_create_ms": numeric_summary(tracker_create),
        "frame0_construct_ms": numeric_summary(frame0_construct),
        "frame1_construct_ms": numeric_summary(frame1_construct),
        "frame_construct_total_ms": numeric_summary(frame_construct_total),
        "detection_construct_ms": numeric_summary(detection_construct),
        "encode_frame0_ms": numeric_summary(encode_frame0),
        "encode_frame1_ms": numeric_summary(encode_frame1),
        "tail_image_encode_total_ms": numeric_summary(tail_image_encode_total),
        "tail_image_encode_avg_ms": numeric_summary(tail_image_encode_avg),
        "tail_encode_timing_total_ms": numeric_summary(tail_encode_timing_total),
        "tail_encode_graph_compute_total_ms": numeric_summary(tail_encode_graph_compute),
        "tail_encode_non_compute_total_ms": numeric_summary(tail_encode_non_compute),
        "add_detection_ms": numeric_summary(add_detection),
        "frame0_ms": numeric_summary(frame0_total),
        "two_frame_encode_ms": numeric_summary(two_frame_encode),
        "propagate_encoded_ms": numeric_summary(propagate_encoded),
        "propagate_ms": numeric_summary(propagate_total),
        "propagate_remainder_ms": numeric_summary(propagate_remainder),
        "track_step_total_ms": numeric_summary(track_step_total),
        "track_step_ms": numeric_summary(track_step_avg),
        "full_frame_step_ms": full_summary,
        "run_required_e2e_ms": numeric_summary(run_required_e2e),
        "one_shot_required_e2e_ms": numeric_summary(one_shot_required_e2e),
        "required_session_setup_ms": numeric_summary(required_session_setup),
        "required_input_prepare_ms": numeric_summary(required_input_prepare),
        "required_model_execute_ms": numeric_summary(required_model_execute),
        "required_image_encode_ms": numeric_summary(required_image_encode),
        "required_image_encode_graph_compute_ms": numeric_summary(
            required_image_encode_graph_compute
        ),
        "required_image_encode_non_compute_ms": numeric_summary(
            required_image_encode_non_compute
        ),
        "required_mask_init_ms": numeric_summary(required_mask_init),
        "required_propagate_encoded_ms": numeric_summary(required_propagate_encoded),
        "required_model_accounted_ms": numeric_summary(required_model_accounted),
        "required_model_remainder_ms": numeric_summary(required_model_remainder),
        "required_core_compute_ms": numeric_summary(required_core_compute),
        "required_non_image_model_ms": numeric_summary(required_non_image_model),
        "required_non_graph_or_setup_ms": numeric_summary(required_non_graph_or_setup),
        "required_accounted_ms": numeric_summary(required_accounted),
        "required_remainder_ms": numeric_summary(required_remainder),
        "required_e2e_accounted_ms": numeric_summary(required_accounted),
        "required_e2e_remainder_ms": numeric_summary(required_remainder),
        "output_write_ms": numeric_summary(output_write),
        "python_full_cache_over_cpp_full_ratio": ratio(
            python_full if isinstance(python_full, float) else None,
            full_summary["mean"] if isinstance(full_summary["mean"], float) else None,
        ),
        "python_full_cache_over_cpp_required_e2e_ratio": ratio(
            python_full if isinstance(python_full, float) else None,
            statistics.fmean(run_required_e2e) if run_required_e2e else None,
        ),
        "python_required_e2e_over_cpp_required_e2e_ratio": ratio(
            python_required if isinstance(python_required, float) else None,
            statistics.fmean(run_required_e2e) if run_required_e2e else None,
        ),
        "python_backbone_over_cpp_image_encode_ratio": ratio(
            python_backbone if isinstance(python_backbone, float) else None,
            statistics.fmean(required_image_encode) if required_image_encode else None,
        ),
        "python_backbone_over_cpp_image_graph_ratio": ratio(
            python_backbone if isinstance(python_backbone, float) else None,
            statistics.fmean(required_image_encode_graph_compute)
            if required_image_encode_graph_compute
            else None,
        ),
        "python_cached_propagate_over_cpp_encoded_ratio": ratio(
            python_cached_prop if isinstance(python_cached_prop, float) else None,
            encoded_prop_summary["mean"] if isinstance(encoded_prop_summary["mean"], float) else None,
        ),
        "frame0_encode_timing_total_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "total_ms")
        ),
        "frame0_encode_preprocess_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "preprocess_ms")
        ),
        "frame0_encode_graph_build_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "graph_build_ms")
        ),
        "frame0_encode_graph_alloc_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "graph_alloc_ms")
        ),
        "frame0_encode_input_upload_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "input_upload_ms")
        ),
        "frame0_encode_graph_compute_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "graph_compute_ms")
        ),
        "frame0_encode_state_update_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "state_update_ms")
        ),
        "frame0_encode_pe_build_ms": numeric_summary(
            summary_values(runs, "frame0_encode_timing", "pe_build_ms")
        ),
        "frame1_encode_timing_total_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "total_ms")
        ),
        "frame1_encode_preprocess_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "preprocess_ms")
        ),
        "frame1_encode_graph_build_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "graph_build_ms")
        ),
        "frame1_encode_graph_alloc_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "graph_alloc_ms")
        ),
        "frame1_encode_input_upload_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "input_upload_ms")
        ),
        "frame1_encode_graph_compute_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "graph_compute_ms")
        ),
        "frame1_encode_state_update_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "state_update_ms")
        ),
        "frame1_encode_pe_build_ms": numeric_summary(
            summary_values(runs, "frame1_encode_timing", "pe_build_ms")
        ),
        "add_mask_prepare_ms": numeric_summary(
            summary_values(runs, "add_detection_timing", "mask_prepare_ms")
        ),
        "add_memory_encode_ms": numeric_summary(
            summary_values(runs, "add_detection_timing", "memory_encode_ms")
        ),
        "add_obj_ptr_ms": numeric_summary(summary_values(runs, "add_detection_timing", "obj_ptr_ms")),
        "add_store_ms": numeric_summary(summary_values(runs, "add_detection_timing", "store_ms")),
        "add_accounted_ms": numeric_summary(add_accounted),
        "add_remainder_ms": numeric_summary(add_remainder),
        "propagate_timing_total_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "total_ms")
        ),
        "propagate_prepare_caches_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "prepare_caches_ms")
        ),
        "propagate_memory_slot_read_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "memory_slot_read_ms")
        ),
        "propagate_obj_ptr_read_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "obj_ptr_read_ms")
        ),
        "propagate_prompt_build_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "prompt_build_ms")
        ),
        "propagate_memory_prepare_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "memory_prepare_ms")
        ),
        "propagate_ptr_prepare_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "ptr_prepare_ms")
        ),
        "propagate_rope_cache_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "rope_cache_ms")
        ),
        "propagate_rope_k_build_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "rope_k_build_ms")
        ),
        "propagate_graph_build_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "graph_build_ms")
        ),
        "propagate_graph_alloc_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "graph_alloc_ms")
        ),
        "propagate_input_upload_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "input_upload_ms")
        ),
        "propagate_graph_compute_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "graph_compute_ms")
        ),
        "propagate_output_read_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "output_read_ms")
        ),
        "propagate_graph_cache_build_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "graph_cache_build_ms")
        ),
        "propagate_graph_cache_input_upload_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "graph_cache_input_upload_ms")
        ),
        "propagate_graph_cache_compute_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "graph_cache_compute_ms")
        ),
        "propagate_graph_cache_output_read_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "graph_cache_output_read_ms")
        ),
        "propagate_bbox_from_logits_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "bbox_from_logits_ms")
        ),
        "propagate_fullmask_active_resize_bbox_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "fullmask_active_resize_bbox_ms")
        ),
        "propagate_fullmask_pending_resize_bbox_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "fullmask_pending_resize_bbox_ms")
        ),
        "propagate_memory_update_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "memory_update_ms")
        ),
        "propagate_tracker_update_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "tracker_update_ms")
        ),
        "propagate_result_build_ms": numeric_summary(
            summary_values(runs, "propagate_timing", "result_build_ms")
        ),
        "propagate_accounted_ms": numeric_summary(propagate_accounted),
        "propagate_internal_remainder_ms": numeric_summary(propagate_internal_remainder),
        "score": numeric_summary(scores),
        "obj_score_logit": numeric_summary(obj_logits),
        "mask_foreground_pixels": numeric_summary(mask_foreground_pixels),
        "selected_mask_index": numeric_summary(selected_mask_indices),
        "decoder_iou_scores": numeric_summary_by_index(decoder_iou_score_rows),
        "python_iou": numeric_summary([float(item["iou"]) for item in python_comp if item["shape_match"]]),
        "python_xor_pixels": numeric_summary(
            [float(item["xor_pixels"]) for item in python_comp if item["shape_match"]]
        ),
        "default_iou": numeric_summary([float(item["iou"]) for item in default_comp if item["shape_match"]]),
        "default_xor_pixels": numeric_summary(
            [float(item["xor_pixels"]) for item in default_comp if item["shape_match"]]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--cpp-exe",
        type=Path,
        default=Path("build/xmake-release-cuda/examples/sam31_tracking_mask_init_smoke"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/sam3.1/sam3.1_multiplex-bf16.ggml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sam31-mask-init-matrix"))
    parser.add_argument("--size", action="append", type=parse_size, required=True)
    parser.add_argument("--case", action="append", type=parse_case, default=None)
    parser.add_argument("--variant", action="append", type=validate_variant, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--python-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--python-warmup-runs", type=int, default=1)
    parser.add_argument("--cpp-warmup-runs", type=int, default=1)
    parser.add_argument(
        "--profile-default",
        action="store_true",
        help=(
            "Run one default C++ pass with SAM3_PROFILE=1 and GGML_CUDA_PROFILE_NODES=1 "
            "per case for synchronized CUDA stage attribution."
        ),
    )
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=None,
        help="Warmup runs for --profile-default; defaults to --cpp-warmup-runs.",
    )
    parser.add_argument("--profile-stage-top", type=int, default=16)
    parser.add_argument("--profile-node-top", type=int, default=80)
    parser.add_argument(
        "--interleave-variants",
        action="store_true",
        help="Run repeat 1 for every variant before repeat 2 to reduce thermal/order bias.",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.cpp_warmup_runs < 0:
        raise SystemExit("--cpp-warmup-runs must be non-negative")
    if args.profile_warmup_runs is None:
        args.profile_warmup_runs = args.cpp_warmup_runs
    if args.profile_warmup_runs < 0:
        raise SystemExit("--profile-warmup-runs must be non-negative")
    if args.profile_stage_top <= 0:
        raise SystemExit("--profile-stage-top must be positive")
    if args.profile_node_top <= 0:
        raise SystemExit("--profile-node-top must be positive")
    if not args.case:
        args.case = [Case(mask_case="center", frame1_offset=3)]
    return args


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "status": "ok",
        "sizes": {},
        "cases": [case.label for case in args.case],
        "variants": args.variant,
        "repeats": args.repeats,
        "python": {"dtype": args.python_dtype, "tf32": args.tf32, "warmup_runs": args.python_warmup_runs},
        "cpp": {
            "warmup_runs": args.cpp_warmup_runs,
            "tf32": args.tf32,
            "tf32_policy_env": cpp_tf32_policy_env(args.tf32),
            "profile_default": args.profile_default,
            "profile_warmup_runs": args.profile_warmup_runs,
        },
    }
    for size in args.size:
        size_item: dict[str, Any] = {"cases": {}}
        for case in args.case:
            case_dir = out / size.label / case.label
            python_ref = run_python_reference(args, size, case, root, case_dir)
            python_timing = python_timing_contract(python_ref["summary"])
            python_mask = read_mask(case_dir / "python" / "frame1_mask.png")

            variant_runs: dict[str, list[dict[str, Any]]] = {variant: [] for variant in args.variant}
            default_mask: np.ndarray | None = None
            if args.interleave_variants:
                for repeat in range(args.repeats):
                    for variant in args.variant:
                        run = run_cpp_variant(args, size, case, root, case_dir, variant, repeat + 1)
                        variant_runs[variant].append(run)
                        if variant == "default":
                            default_mask = read_mask(root / run["dir"] / "frame1_mask.png")
            else:
                for variant in args.variant:
                    runs = [
                        run_cpp_variant(args, size, case, root, case_dir, variant, repeat + 1)
                        for repeat in range(args.repeats)
                    ]
                    variant_runs[variant] = runs
                    if variant == "default":
                        default_mask = read_mask(root / runs[-1]["dir"] / "frame1_mask.png")

            case_item = {
                "python": python_ref,
                "python_timing": python_timing,
                "variants": {},
            }
            if args.profile_default:
                case_item["cpp_profile"] = run_cpp_profile_default(
                    args, size, case, root, case_dir
                )
            for variant, runs in variant_runs.items():
                case_item["variants"][variant] = summarize_variant(
                    runs,
                    python_mask=python_mask,
                    default_mask=default_mask,
                    python_timing=python_timing,
                    root=root,
                )
            size_item["cases"][case.label] = case_item
        report["sizes"][size.label] = size_item

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
