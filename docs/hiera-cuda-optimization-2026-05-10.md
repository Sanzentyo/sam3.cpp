# Hiera CUDA Optimization Status

Date: 2026-05-10

This note tracks the current Hiera encode optimization pass for SAM 2.1 Base+ on
CUDA. It also records the official PyTorch quality comparison needed before this
path can be treated as a Rust-wrapper baseline.

## Changes

- Added detailed `SAM3_PROFILE` stages inside `sam2_encode_image_hiera`.
- Cached SAM2 Hiera input positional embeddings per state/img size.
- Cached SAM2 neck positional embeddings on CPU.
- Reused SAM2 neck PE backend tensors when the FPN shapes are unchanged.
- Added `sam3_benchmark --output-mask-dir` so C++ masks can be compared with
  official PyTorch masks.
- Added `scripts/compare_sam2_official_quality.py` for official SAM2 mask IoU.
- Added a `sam3cpp-meta` JSONL row from `sam3_benchmark --output-jsonl` and
  validation in the official SAM2 comparison script so decoded input
  resolution, frame count, prompt, and effective encode size are checked before
  interpreting C++ vs PyTorch quality results.
- Cached the Hiera positional embedding as a backend tensor. Steady-state
  frames now copy the cached device tensor into the graph input instead of
  uploading the same 4 MiB CPU buffer every frame.
- Added opt-in `SAM3_PROFILE_HIERA_CUTS=1` cumulative cut profiling for SAM2
  Hiera. It times Hiera stage outputs and all FPN outputs separately from the
  normal graph so the dominant region can be identified.
- Added optional `SAM3_PROFILE_HIERA_CUT_OPS=1` op-count logging for Hiera cuts.
- Removed avoidable Q/K/V layout copies in non-q-stride Hiera attention blocks
  by viewing the fused QKV projection as head-split tensors directly.
- Removed explicit Hiera LayerNorm affine `REPEAT` nodes and rely on ggml
  broadcasted `MUL`/`ADD` instead.

## Performance

Model: `sam2.1_hiera_base_plus_q4_0`, 10 frames, CUDA, bbox-only tracking.

Comparison rule: C++ vs official PyTorch speed/quality claims are valid only
when both runs use the same decoded source-frame resolution, the same frame
range, the same prompt, and the same SAM2 input encode size
(`--encode-img-size` in C++, `model.image_size` in official PyTorch). The
effective input resolution must match before interpreting speed or quality
differences as implementation differences. Running multiple source resolutions
or encode sizes is still useful, but those rows are a scaling study rather than
a cross-implementation win/loss comparison.

| State | Track ms/frame | P50 ms | P95 ms | Note |
| --- | ---: | ---: | ---: | --- |
| Before PE caching | 547.3 | 544.8 | 559.4 | Base+ fallback only |
| After CPU PE caching | 136.1 | 133.3 | 148.9 | removed repeated Hiera/neck PE generation |
| After backend neck PE reuse | 129.8 | 127.3 | 139.7 | avoids repeated neck PE backend upload |
| After `head_dim=56` FA tile | 120.1 | 119.0 | 127.1 | avoids fallback attention for Base+ |
| After fused preprocess | 117.4 | 115.4 | 128.3 | resize and normalize in one pass |
| After gating debug tensor outputs | 116.6 | 115.8 | 126.2 | removes nonessential debug graph outputs unless requested |
| After Hiera PE backend cache | 113.6 | 113.8 | 121.6 | avoids repeated CPU upload of Hiera positional embedding |
| After direct QKV head views | 110.1 | 109.1 | 117.9 | removes extra Q/K/V `cont` and reshape nodes in non-q-stride Hiera blocks |
| After LayerNorm broadcast affine | 107.2 | 105.5 | 115.4 | removes explicit Hiera norm weight/bias `REPEAT` nodes |

Encode-size sweep after PE caching. These rows use the same decoded source
video frames and vary only the SAM2 input encode size. This is a scaling sweep,
not an apples-to-apples comparison against the official PyTorch baseline unless
the official run uses the same decoded source-frame resolution and the same
input encode size.

| Encode size | Track ms/frame | P50 ms | P95 ms | RSS MiB | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 129.8 | 127.3 | 139.7 | 647.4 | default, comparable to official default |
| 768 | 72.3 | 71.3 | 80.6 | 574.2 | lower-resolution scaling point |
| 640 | 48.3 | 47.1 | 53.1 | 552.2 | lower-resolution scaling point |
| 512 | 34.4 | 33.4 | 39.5 | 572.8 | lower-resolution scaling point |

Base+ precision sweep after CPU PE caching:

| Model | Track ms/frame | P50 ms | P95 ms | RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| `sam2.1_hiera_base_plus_f32` | 149.8 | 149.7 | 159.5 | 678.8 |
| `sam2.1_hiera_base_plus_f16` | 146.9 | 145.5 | 156.2 | 705.1 |
| `sam2.1_hiera_base_plus_q8_0` | 134.6 | 133.6 | 141.8 | 646.6 |
| `sam2.1_hiera_base_plus_q4_1` | 136.0 | 134.4 | 143.7 | 649.6 |
| `sam2.1_hiera_base_plus_q4_0` | 134.5 | 133.6 | 142.1 | 647.8 |

Official PyTorch SAM2.1 Base+ on the same 10-frame clip and default 1024 encode
size measured `43.4 ms/frame` for propagation with `855.2 MiB` CUDA allocation.
The current C++ CUDA path is therefore still about 3x slower than official
PyTorch for this prompt/video at the default 1024 encode size, despite the PE
cache improvement. The lower encode-size rows above are useful for understanding
scaling, but they must not be used to claim a speed win over the official 1024
PyTorch baseline. Any lower-resolution comparison needs an official PyTorch run
configured to the same input encode size.

## Detailed Profile

After backend neck PE reuse, Base+ q4_0 still spends most of steady-state frame
time in the Hiera encode graph:

| Stage | Mean ms | Count |
| --- | ---: | ---: |
| `hiera_encode_graph_compute_total` | 89.9 | 10 |
| `hiera_encode_preprocess` | 16.4 | 10 |
| `hiera_encode_pos_embed_upload` | 2.0 | 10 |
| `hiera_encode_fpn_copy_to_state` | 0.3 | 10 |

The first frame still pays one-time cache construction:

| Stage | First-run cost |
| --- | ---: |
| `hiera_encode_pos_embed_compute` | 218.1 ms |
| `hiera_encode_pe_compute_upload` | 108.3 ms |

## Official Quality Comparison

Compared C++ `sam2.1_hiera_base_plus_q4_0` against official PyTorch SAM2.1
Base+ using the same video frames, same point prompt, same input encode size,
and 10 frames.

| Metric | Value |
| --- | ---: |
| Frames compared | 10 |
| Mean mask IoU | 0.0197 |
| Minimum mask IoU | 0.0000 |
| Minimum bbox IoU | 0.0169 |
| C++ track ms/frame | 139.6 |
| PyTorch propagate ms/frame | 43.4 |

This is not quality parity. The official SAM2 mask selected by the same point is
much larger than the current C++ mask. PyTorch-level speed is not sufficient by
itself until this semantic mismatch is understood.

After enabling ggml CUDA FlashAttention tile kernels for `head_dim=56`, the
same q4_0 run improves substantially:

| Encode size | C++ track ms/frame | PyTorch ms/frame | Mean mask IoU | Min mask IoU | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 120.2 | 43.4 | 0.4745 | 0.0000 | metadata-validated same decoded resolution, frame range, prompt, and encode size; frame 8 official output is empty |
| 512 | 36.8 | 15.4 | 0.7798 | 0.7743 | metadata-validated same decoded resolution, frame range, prompt, and encode size |

This indicates the previous no-mask attention fallback was both slower and
semantically weak for SAM2.1 Base+. The implementation is still slower than
official PyTorch at the same input encode size, but the quality gap is much
smaller with the CUDA FA tile path.

Fusing resize and normalization in preprocessing preserves the measured quality
numbers and reduces CPU preprocessing from about `15.7 ms/frame` to
`12.9 ms/frame` at 1024. The 1024 bbox-only tracking path improves to
`117.4 ms/frame`; the 512 bbox-only path improves to `32.0 ms/frame`. Gating
debug-only Hiera/FPN graph outputs behind `SAM3_DEBUG_TENSORS` or
`SAM2_DUMP_DIR` gives only a small additional 1024 bbox-only change to
`116.6 ms/frame`, so it should be treated as graph cleanup rather than a major
speedup.

Caching the Hiera positional embedding as a backend tensor removes the repeated
CPU-to-device upload from steady-state frames. In the profile run,
`hiera_encode_pos_embed_backend_copy` is about `0.09 ms/frame`, replacing the
previous roughly `1.9 ms/frame` upload. The 1024 bbox-only tracking run improves
to `113.6 ms/frame`. Full-mask parity against the previous output is exact on
the 10-frame sample: `mask_hash_equal_rows=10`, `min_bbox_iou=1.0`, and
`max_score_abs_delta=0.0`.

The cumulative Hiera cut profiler shows the dominant region is stage 2 of the
Hiera backbone, not FPN:

| Cut | Cumulative ms | Incremental ms | Nodes |
| --- | ---: | ---: | ---: |
| `hiera_stage_0` | 12.6 | 12.6 | 134 |
| `hiera_stage_1` | 27.5 | 14.9 | 333 |
| `hiera_stage_2` | 69.9 | 42.4 | 1331 |
| `hiera_stage_3` | 72.0 | 2.2 | 1539 |
| `fpn_all` | 75.1 | 3.1 | 1601 |

This means the next meaningful CUDA-side optimization should target the many
windowed Hiera blocks in stage 2, especially their QKV/projection/MLP `mul_mat`,
window partition/unpartition, and normalization/add chains. FPN work is no
longer a first-priority target for Base+ 1024 latency.

Adding stage-2 block-group cuts for Base+ narrows the largest region further.
The model's actual stage 2 covers block 5 through block 20. The profiler records
cumulative cut points after blocks 10, 16, and 20:

| Cut | Cumulative ms | Incremental ms | Nodes |
| --- | ---: | ---: | ---: |
| `hiera_stage_1` | 26.4 | 14.0 | 333 |
| `hiera_stage_2_block_10` | 40.7 | 14.2 | 730 |
| `hiera_stage_2_block_16` | 60.9 | 20.2 | 1088 |
| `hiera_stage_2_block_20` | 70.8 | 9.9 | 1331 |

The heaviest block group is therefore the middle of stage 2, roughly blocks
11-16. This region contains both windowed blocks and global-attention blocks
12 and 16 at the 64x64 feature resolution, so the next optimization candidate
is reducing the per-block overhead of repeated window partition/attention
projection/MLP sequences and the global-attention blocks rather than tuning
FPN.

A focused cut run around stage-2 blocks 11-16 gives this prioritization signal:

| Cut | Cumulative ms | Incremental ms | Nodes |
| --- | ---: | ---: | ---: |
| `hiera_stage_1` | 26.4 | 26.4 | 333 |
| `hiera_stage_2_focus_block_11` | 43.7 | 17.3 | 796 |
| `hiera_stage_2_focus_block_12` | 52.1 | 8.4 | 870 |
| `hiera_stage_2_focus_block_13` | 51.3 | -0.8 | 928 |
| `hiera_stage_2_focus_block_14` | 51.8 | 0.5 | 986 |
| `hiera_stage_2_focus_block_15` | 54.1 | 2.3 | 1029 |
| `hiera_stage_2_focus_block_16` | 58.7 | 4.6 | 1088 |
| `hiera_stage_2_block_20` | 69.1 | 10.4 | 1331 |
| `hiera_stage_3` | 74.2 | 5.1 | 1539 |
| `fpn_all` | 76.7 | 2.5 | 1601 |

Because this focus run uses one profiling iteration per cut, the per-block
incremental values are noisy and should not be read as exact block costs. The
stable conclusion is that stage 2 dominates, with blocks 11-16 and the global
blocks 12/16 worth kernel-level profiling next.

The op-count version of the same focus profile confirms that the middle stage-2
cuts are dominated by repeated layout and projection work. From `hiera_stage_1`
to `hiera_stage_2_focus_block_11`, the cumulative graph adds 172 `RESHAPE`, 59
`CONT`, 57 `ADD`, 39 `PERMUTE`, and 29 `MUL_MAT` nodes per frame. This made
QKV/head layout handling a lower-risk target than changing attention math.

The direct-QKV-head-view change preserves the previous C++ output exactly on the
10-frame 1024 full-mask sample: `mask_hash_equal_rows=10`,
`min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, and `max_score_abs_delta=0.0`.
The normal Hiera graph shrinks from 1601 to 1391 nodes. The measured 1024
bbox-only run improves from `113.6 ms/frame` to `110.1 ms/frame`; the
full-mask run with JSONL output measures `111.6 ms/frame`. This is still slower
than official PyTorch at the same 1024 encode size, so the next target remains
Hiera stage 2, especially the global-attention blocks and projection/MLP
matmuls.

Removing explicit Hiera LayerNorm affine `REPEAT` nodes keeps the same full-mask
output versus the direct-QKV-head-view run:
`mask_hash_equal_rows=10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, and
`max_score_abs_delta=0.0`. The normal Hiera graph shrinks again from 1391 to
1295 nodes. The 1024 bbox-only run improves to `107.2 ms/frame`; the full-mask
JSONL run measures `108.9 ms/frame`.

Precision does not explain the quality gap. Re-running the same default 1024
comparison across Base+ precisions gives nearly identical low IoU:

| Model | C++ track ms/frame | Mean mask IoU | Min mask IoU | PyTorch ms/frame |
| --- | ---: | ---: | ---: | ---: |
| `sam2.1_hiera_base_plus_f32` | 147 | 0.0185 | 0.0000 | 43.3 |
| `sam2.1_hiera_base_plus_f16` | 148 | 0.0185 | 0.0000 | 43.4 |
| `sam2.1_hiera_base_plus_q8_0` | 135 | 0.0179 | 0.0000 | 43.4 |
| `sam2.1_hiera_base_plus_q4_0` | 135 | 0.0197 | 0.0000 | 43.6 |

An experimental C++ `--multimask` initial prompt path was also tested. It
selects the highest predicted IoU mask among SAM mask tokens 1..3 and uses the
matching mask token for the tracker object pointer. This did not fix 1024
parity:

| Mode | Encode size | C++ track ms/frame | Mean mask IoU | Min mask IoU |
| --- | ---: | ---: | ---: | ---: |
| single-mask prompt | 1024 | 135-140 | 0.0197 | 0.0000 |
| multimask prompt | 1024 | 134 | 0.0321 | 0.0000 |
| multimask prompt | 512 | 39 | 0.1822 | 0.1394 |

At `--encode-img-size 512`, the earlier bbox-only scaling run reached
`34.4 ms/frame`, but that initial comparison was against the default official
PyTorch 1024 run and was therefore not a fair speed baseline. The current
metadata-validated full-mask quality run measures C++ at `36.8 ms/frame` and
official PyTorch with `model.image_size=512` at `15.4 ms/frame`, so C++ is still
about 2.4x slower at the same input encode size for the full-mask quality path:

| Metric | Value |
| --- | ---: |
| Frames compared | 10 |
| Mean mask IoU vs PyTorch 1024 | 0.0521 |
| Mean mask IoU vs PyTorch 512 | 0.1496 |
| Minimum mask IoU vs PyTorch 512 | 0.1416 |
| Minimum bbox IoU vs PyTorch 512 | 0.0968 |
| C++ 512 track ms/frame | 36.8 |
| PyTorch 512 propagate ms/frame | 15.4 |

## Priority

1. Fix or explain the SAM2 point-prompt quality gap against official PyTorch.
   The current C++ mask is not a faithful substitute for the official model on
   this sample. The f32/f16/q8/q4 sweep shows this is not primarily a
   quantization issue, and the multimask prompt experiment does not close the
   gap at 1024.
2. Reduce Hiera encode graph compute. After caching fixed PE, enabling
   `head_dim=56` tile FlashAttention, and fusing preprocessing, the remaining
   dominant steady-state cost is still the Hiera CUDA graph compute, now roughly
   65-70 ms after warmup after direct QKV head views and Hiera LayerNorm
   broadcast affine cleanup.
3. Decide whether lower encode sizes are acceptable for the Rust wrapper. This
   requires same-resolution measurements: C++ 512 must be compared with
   official PyTorch 512, C++ 640 with official PyTorch 640, and so on, with the
   same decoded source-frame resolution in both runs.
4. Continue reducing CPU preprocessing cost or move it to GPU. Fused
   resize/normalize brings it down to roughly 10-15 ms/frame at 1024, but that
   is still large relative to the PyTorch baseline.
5. Treat `head_dim=56` FlashAttention support as a secondary optimization. The
   tile path is now enabled and gives a modest 1024 speedup plus a large quality
   improvement. MMA support for 56 did not compile and should be treated as a
   separate ggml kernel-design task, not a simple allowlist change.

## Artifacts

```text
outputs/hiera-pe-backend-cache/base_plus_q4_0_noprofile.log
outputs/hiera-pe-backend-cache/base_plus_summary.json
outputs/hiera-pe-cache-profile2/base_plus_all.log
outputs/sam2-official-quality-10/summary.json
outputs/sam2-official-quality-10/cpp.log
outputs/sam2-official-quality-10/python-official.log
outputs/hiera-encode-size-sweep/base_plus_q4_0_512.log
outputs/sam2-official-quality-512/summary.json
outputs/sam2-official-quality-512-same-res/summary.json
outputs/sam2-official-quality-10-precision-f32/summary.json
outputs/sam2-official-quality-10-precision-f16/summary.json
outputs/sam2-official-quality-10-precision-q8_0/summary.json
outputs/sam2-official-quality-10-precision-q4_0/summary.json
outputs/sam2-official-quality-10-multimask-fullmask/summary.json
outputs/sam2-official-quality-512-multimask-fullmask-same-res/summary.json
outputs/hiera-head56-tile/base_plus_q4_0_noprofile.log
outputs/hiera-head56-tile/summary.json
outputs/sam2-official-quality-10-head56-tile-q4_0/summary.json
outputs/sam2-official-quality-512-head56-tile-q4_0/summary.json
outputs/preprocess-fused-head56/summary.json
outputs/sam2-official-quality-10-preprocess-fused-q4_0/summary.json
outputs/sam2-official-quality-512-preprocess-fused-q4_0/summary.json
outputs/debug-output-gated/summary.json
outputs/metadata-jsonl-check/cpp.jsonl
outputs/sam2-official-quality-10-metadata-q4_0/summary.json
outputs/sam2-official-quality-512-metadata-q4_0/summary.json
outputs/hiera-pos-backend-cache/summary.json
outputs/hiera-pos-backend-cache/parity.json
outputs/hiera-cut-profile/base_plus_q4_0_1024_cuts_fpn_all_summary.json
outputs/hiera-cut-profile/base_plus_q4_0_1024_stage2_blocks_summary.json
outputs/hiera-cut-profile/base_plus_q4_0_1024_stage2_focus_summary.json
outputs/hiera-cut-profile/base_plus_q4_0_1024_stage2_focus_ops_summary.json
outputs/hiera-qkv-view-opt/bbox_only_summary.json
outputs/hiera-qkv-view-opt/fullmask_summary.json
outputs/hiera-qkv-view-opt/profile_summary.json
outputs/hiera-qkv-view-opt/stage2_focus_summary.json
outputs/hiera-qkv-view-opt/parity.json
outputs/hiera-norm-broadcast/bbox_only_summary.json
outputs/hiera-norm-broadcast/fullmask_summary.json
outputs/hiera-norm-broadcast/profile_summary.json
outputs/hiera-norm-broadcast/parity.json
```
