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
- Added optional `SAM3_PROFILE_HIERA_CUT_MATMULS=1` Hiera cut matmul shape
  logging.
- Removed avoidable Q/K/V layout copies in non-q-stride Hiera attention blocks
  by viewing the fused QKV projection as head-split tensors directly.
- Removed explicit Hiera LayerNorm affine `REPEAT` nodes and rely on ggml
  broadcasted `MUL`/`ADD` instead.
- Parallelized CPU resize/normalize preprocessing across `n_threads` while
  preserving per-pixel arithmetic and output parity.

## Performance

Model: `sam2.1_hiera_base_plus_q4_0`, 10 frames, CUDA, bbox-only tracking.

Comparison rule: C++ vs official PyTorch speed/quality claims are valid only
when both runs use the same decoded source-frame resolution, the same frame
range, the same prompt, and the same SAM2 input encode size
(`--encode-img-size` in C++, `model.image_size` in official PyTorch). The
primary comparison must keep both resolution layers fixed: the decoded source
frame size before preprocessing, and the model input encode size after
preprocessing. Running multiple source resolutions or encode sizes is still
useful, but those rows are a scaling study; they must not be used as a
cross-implementation win/loss comparison unless the matching PyTorch run uses
the same decoded source resolution and encode size. In short: use matched
same-input rows for C++/PyTorch conclusions, and use unmatched resolution rows
only to describe scaling.

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
| After threaded preprocess | 102.2 | 100.0 | 114.5 | same-size matrix run with `--n-threads 4`; output parity versus `--n-threads 1` |
| After CUDA conv-transpose k2s2 kernel | 93.9 | 90.5 | 106.0 | five paired 1024 runs; parity versus generic conv-transpose path |

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

Precision sweep after the Hiera graph cleanups and before the CUDA
conv-transpose k2s2 specialization. The C++ rows below use the same decoded
`1008x568` source frames as the official PyTorch comparison for the matching
encode-size row; the 512 and 1024 rows should be read as separate same-input
comparisons, not as cross-resolution wins or losses:

| Encode size | Model | C++ track ms/frame | P50 ms | P95 ms | PyTorch bf16 ms/frame |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024 | `sam2.1_hiera_base_plus_f32` | 121.0 | 120.8 | 128.9 | 57.0 |
| 1024 | `sam2.1_hiera_base_plus_f16` | 119.2 | 115.8 | 129.8 | 57.0 |
| 1024 | `sam2.1_hiera_base_plus_q8_0` | 105.9 | 102.8 | 115.9 | 57.0 |
| 1024 | `sam2.1_hiera_base_plus_q4_1` | 106.2 | 103.7 | 115.3 | 57.0 |
| 1024 | `sam2.1_hiera_base_plus_q4_0` | 105.7 | 103.3 | 115.1 | 57.0 |
| 512 | `sam2.1_hiera_base_plus_f32` | 32.5 | 31.7 | 37.0 | 21.0 |
| 512 | `sam2.1_hiera_base_plus_f16` | 30.2 | 29.1 | 34.9 | 21.0 |
| 512 | `sam2.1_hiera_base_plus_q8_0` | 29.9 | 28.9 | 34.6 | 21.0 |
| 512 | `sam2.1_hiera_base_plus_q4_1` | 30.2 | 29.2 | 34.4 | 21.0 |
| 512 | `sam2.1_hiera_base_plus_q4_0` | 29.9 | 29.1 | 34.2 | 21.0 |

The quantized rows are close to each other at both sizes. Precision selection
alone is therefore not enough to overtake official PyTorch; the remaining work
needs to reduce Hiera graph compute or improve the CUDA kernels used by the
stage-2 projection/MLP matmuls and layout operations.

The current same-script C++/official-PyTorch comparison uses the same decoded
source-frame resolution, frame range, point prompt, model family, and explicit
SAM2 input encode size:

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 track ms/frame | PyTorch/C++ ratio | Note |
| --- | ---: | ---: | ---: | --- |
| 1024 | 93.9 | 56.9 | 0.606 | same decoded 1008x568 source frames, `--encode-img-size`, and `model.image_size` |
| 512 | 23.6 | 20.7 | 0.877 | same decoded 1008x568 source frames, `--encode-img-size`, and `model.image_size` |

Values below `1.0` in the ratio column mean official PyTorch is faster. The
current C++ CUDA path is therefore still about `1.65x` slower than official
PyTorch at 1024 and about `1.14x` slower at 512 for this prompt/video. The
lower encode-size rows above are still useful for understanding scaling, but
they must not be used to claim a speed win over the official 1024 PyTorch
baseline.

Re-running the same comparison after the input-resolution contract update gives
the same conclusion. The 2026-05-10 rerun used the same decoded `1008x568`
source frames, 10-frame range, point prompt, model family, and explicit encode
size in both implementations:

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 track ms/frame | PyTorch/C++ ratio |
| --- | ---: | ---: | ---: |
| 1024 | 93.6 | 57.65 | 0.616 |
| 512 | 25.0 | 20.08 | 0.803 |

The 512 gap is now about `4.9 ms/frame`, but it is still not a win. The next
accepted optimization must either remove the short-run Hiera encode outliers or
reduce steady Hiera kernel time enough to make the comparable 512 row exceed
the PyTorch baseline.

Advancing `tracker.frame_index` after the initial point-prompt instance is
added fixes the internal frame numbering for the first propagated frame and
avoids an immediate duplicate non-conditioning memory encode. This improves the
comparable rows but still does not overtake PyTorch:

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 track ms/frame | PyTorch/C++ ratio | Quality note |
| --- | ---: | ---: | ---: | --- |
| 1024 | 88.5 | 56.61 | 0.640 | mean mask IoU `0.4742`, min `0.0` |
| 512 | 23.8 | 19.71 | 0.828 | mean mask IoU `0.7761`, min `0.7693` |

The profile confirms the expected mechanism: the 512 run now has one
`memory_encode` call instead of two (`1.205 ms` total), while Hiera encode
remains the dominant cost. The 512 Hiera compute values are still
`130.252, 17.613, 17.281, 17.474, 17.044, 34.509, 34.187, 15.280, 15.089,
14.996 ms`, so the remaining gap is not solved by tracker bookkeeping.

The CUDA conv-transpose k2s2 specialization targets the SAM decoder upsampling
shape `kernel=2,stride=2,padding=0`. The generic CUDA kernel checked every
kernel position for every output element even though this shape has exactly one
contributing input pixel per output pixel. The specialized path preserves the
same arithmetic and can be disabled with `GGML_CUDA_CONV_TRANSPOSE_K2S2=0` for
A/B checks.

| Encode size | Generic mean | Specialized mean | Saved mean | Mean speedup | Parity |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 101.56 | 93.92 | 7.64 | 8.13% | `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, `max_score_abs_delta=0` |
| 512 | 25.02 | 23.60 | 1.42 | 6.02% | `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, `max_score_abs_delta=0` |

Threaded preprocessing is parity-preserving on the 10-frame 1024 full-mask
sample: comparing `--n-threads 1` with `--n-threads 4` gives 10 rows on both
sides and `diff_rows=0` for frame index, bbox, score, mask area, and mask hash.
Five paired bbox-only runs show the expected benefit is modest but stable:

| Encode size | Threads=1 mean | Threads=4 mean | Saved | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 105.38 | 101.38 | 4.00 | 1.039x |
| 512 | 26.24 | 25.08 | 1.16 | 1.046x |

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

After the graph cleanups, a current q8_0 1024 quality spot-check against
official PyTorch gives `105.9 ms/frame` bbox-only, `111.1 ms/frame` for the
full-mask JSONL run, mean mask IoU `0.4495`, and min mask IoU `0.0` with the
same metadata checks passing. This is not enough to replace q4_0 as the quality
baseline even though q8_0 is slightly faster in the bbox-only spot-check.

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

The matmul-shape profile after these graph cleanups shows that stage 2 is now
mostly repeated quantized projection/MLP work. From `hiera_stage_1` to
`hiera_stage_2`, the largest added matmul signatures per frame are:

| Mean count | Signature |
| ---: | --- |
| 16 | `q4_0[448,1792] x f32[448,4096] -> f32[1792,4096]` |
| 16 | `q4_0[1792,448] x f32[1792,4096] -> f32[448,4096]` |
| 12 | `q4_0[448,1344] x f32[448,196,25] -> f32[1344,196,25]` |
| 12 | `q4_0[448,448] x f32[448,196,25] -> f32[448,196,25]` |

The first two rows are stage-2 MLP expansion/projection at 64x64; the latter
two rows are window-attention QKV/projection over 25 windows of 14x14 tokens.
This makes the next optimization target the quantized CUDA matmul path for
medium-width, many-column Hiera shapes, not FPN.

A separate `GGML_CUDA_FORCE_MMQ=ON` build does not improve these Base+ shapes:

| Model | Default track ms/frame | Force MMQ track ms/frame | Result |
| --- | ---: | ---: | --- |
| `sam2.1_hiera_base_plus_q4_0` | 107.6 | 109.5 | slower |
| `sam2.1_hiera_base_plus_q8_0` | 105.9 | 108.9 | slower |

So the immediate path is not to force MMQ globally. The remaining speed work
should either reduce the number of stage-2 matmuls/layout transitions or improve
the dequantize-plus-GEMM path for these medium-width Hiera shapes.

A separate `GGML_CUDA_FORCE_CUBLAS=ON` build is also worse after threaded
preprocessing:

| Encode size | Default track ms/frame | Force cuBLAS track ms/frame | Result |
| --- | ---: | ---: | --- |
| 1024 | 102.2 | 182.9 | much slower, higher RSS |
| 512 | 25.0 | 46.9 | much slower, higher RSS |

This rules out global dispatch forcing as the next path. The remaining CUDA
work needs shape-specific improvement for the existing quantized matmul path or
graph-level reductions that remove repeated stage-2 layout/matmul work.

Flattening non-q-stride window-attention QKV/projection matmuls from
`[C, N, B_win]` into `[C, N * B_win]` was tested and rejected. It did not
improve the 1024 full-mask run (`109.2 ms/frame` versus the current
`108.9 ms/frame` full-mask reference), and it changed C++ output:
`mask_hash_equal_rows=0`, `min_bbox_iou=0.9647`, `max_bbox_delta_px=3.0`, and
`max_score_abs_delta=0.0691`. Because this path changes floating-point grouping
and does not buy speed, it should not be used as a parity-preserving
optimization.

Making the ggml CUDA graph cache shape-keyed by default is parity-preserving
and gives a measurable Base+ improvement. The legacy pointer-key behavior is
kept behind `GGML_CUDA_GRAPH_PTR_KEY=1` for debugging. Five paired runs comparing
legacy pointer-key against shape-key default produced:

| Encode size | Legacy mean ms/frame | Shape-key mean ms/frame | Mean saved | Speedup | Parity |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 107.62 | 105.86 | 1.76 | 1.66% | 0 diff rows |
| 512 | 30.00 | 26.22 | 3.78 | 14.42% | 0 diff rows |

The larger 512 gain indicates graph recapture/launch overhead is more visible
when the Hiera compute itself is smaller. This does not close the 1024 PyTorch
gap, but it is a low-risk default improvement and removes the need for callers
to remember `GGML_CUDA_GRAPH_SHAPE_KEY=1`.

Adding labels to `SAM3_PROFILE compute` makes the 512 post-shape-key track path
clearer. On a 10-frame q4_0 512 profile, steady Hiera graph compute is about
`16 ms/frame`, CPU preprocess is about `2.6 ms/frame`, and
`propagate_single` settles at about `5 ms/frame` after its first graph warmup
frames. The remaining 512 gap to official PyTorch is therefore no longer only
Hiera encode; it also includes the propagation graph and small CPU overheads.
For 1024, Hiera graph compute still dominates at roughly `65 ms` steady-state.

The generic graph structure profile (`SAM3_PROFILE_GRAPH_OPS=1` and
`SAM3_PROFILE_GRAPH_MATMULS=1`) shows `propagate_single` at 512 is also dominated
by small/medium matmul and layout work: about 96 `MUL_MAT`, 127 `RESHAPE`, 80
`VIEW`, 77 `CONT`, and 53 `PERMUTE` nodes per propagation graph. The most common
propagation matmul is
`q4_0[256,256] x f32[256,1024] -> f32[256,1024]` with mean count 24, followed by
single-token/object-pointer shapes such as `q4_0[256,256] x f32[256,1]`. This
means the 512 path cannot be finished by Hiera-only work; the propagation graph
needs the same kind of layout reduction or quantized small-GEMM improvement.

The opt-in ggml CUDA node profiler (`GGML_CUDA_PROFILE_NODES=1`) disables CUDA
graphs and synchronizes after every node, so its totals are not comparable to
normal benchmark rows. It is useful for ranking kernels and shapes. The summary
script now records `sum_ms_drop_max` and `mean_ms_drop_max` for node groups so
one-time first-use outliers do not hide steady-state hotspots. On the current
3-frame 1024 q4_0 run after the conv-transpose k2s2 kernel, the Hiera encode
profile shows:

| Region | Node/signature | Count | Sum ms | Drop-max sum ms | Mean drop-max ms | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Hiera | `FLASH_ATTN_EXT`, `f32[56,8,4096,1]` output | 9 | 36.57 | 32.39 | 4.05 | steady global-attention hotspot |
| Hiera | `FLASH_ATTN_EXT`, `f32[56,8,196,25]` output | 36 | 18.22 | 17.70 | 0.51 | window-attention hotspot |
| Hiera | q4_0 MLP `448 -> 1792` matmul | 48 | 6.34 | 6.20 | 0.13 | repeated stage-2 MLP shape |
| Hiera | q4_0 MLP `1792 -> 448` matmul | 48 | 6.42 | 6.28 | 0.13 | repeated stage-2 MLP shape |
| Propagate | `CONV_TRANSPOSE_2D` | 6 | 1.98 | 1.42 | 0.28 | no longer a primary bottleneck after k2s2 specialization |

The largest raw `MUL_MAT` signature,
`f32[147,65536] x f32[147,112] -> f32[65536,112]`, is almost entirely a first-use
outlier (`69.82 ms` raw sum, `0.38 ms` after dropping the max sample). It should
not be treated as the steady-state target. The steady 1024 target is now the
global `FLASH_ATTN_EXT` path for head_dim 56 and 4096 tokens, followed by the
window-attention and quantized MLP matmul shapes. A simple experiment that cast
Hiera Q/K/V to f16 before `ggml_flash_attn_ext` was rejected: the current CUDA
FlashAttention tile asserts `Q->type == GGML_TYPE_F32` for this path, so f16/bf16
attention would require a real kernel/backend change rather than a graph-level
cast.

The head_dim 56 tile column width was also tested. Forcing smaller tile widths
was slower on the same 1024 q4_0 benchmark: 16 cols measured `112.5 ms/frame`,
8 cols `135.3 ms/frame`, and 4 cols `141.3 ms/frame` against the default
`101.7 ms/frame`. A diagnostic 64-col variant was slightly faster over five
paired runs (`101.78 -> 100.58 ms/frame`, mean `1.20 ms/frame` saved), but it
changed full-mask output (`mask_hash_equal_rows=0/10`, `min_bbox_iou=0.9645`,
`max_bbox_delta_px=3.0`, `max_score_abs_delta=0.0847`). This is not acceptable
as a parity-preserving optimization, so the default 32-col tile path remains in
place.

Changing the NVIDIA FP32 tile config for `head_dim=56,ncols=32` from
`nbatch_fa=32` to `64` compiled and preserved full-mask parity on the 10-frame
1024 q4_0 sample (`mask_hash_equal_rows=10/10`), but the measured speed change
was not meaningful: bbox-only was `94.4 -> 93.7 ms/frame` in a single A/B pair
and full-mask stayed `94.1 -> 94.1 ms/frame`. This was reverted and is not a
priority unless a larger paired run shows a real effect.

Changing CPU resize weights and bilinear math from `double` to `float` also
preserved full-mask parity (`mask_hash_equal_rows=10/10`), but did not speed up
preprocessing on the same 5-frame 1024 profile. `hiera_encode_preprocess` was
`8.60 ms` with the default double path and `8.74 ms` with the float path, so this
was reverted. The next preprocessing optimization should be a real GPU
resize/normalize/upload path, not scalar type tuning inside the CPU loop.

Precomputing a 3x256 uint8 normalization lookup table also preserved full-mask
parity (`mask_hash_equal_rows=10/10`), but did not improve the same 5-frame 1024
profile. `hiera_encode_preprocess` moved from `8.599 ms` to `8.759 ms`, while
total tracking stayed at `98.8 ms/frame`. This was reverted; the CPU loop is not
currently limited by the per-pixel normalize arithmetic enough for this
micro-optimization to matter.

Changing Hiera qkv/proj/MLP/patch-embed broadcast bias adds to
`ggml_add_inplace` preserved full-mask parity at both 1024 and 512
(`mask_hash_equal_rows=10/10` for each), but did not improve speed. The paired
current-binary comparison was `94 -> 95 ms/frame` at 1024 and effectively
unchanged at 512 (`34 -> 34 ms/frame`), so this graph cleanup was reverted.
The remaining Hiera target is still kernel time, especially head_dim 56
FlashAttention, not add-node allocation style.

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
2. Reduce Hiera encode graph compute. After PE caching, `head_dim=56` tile
   FlashAttention, graph cleanups, threaded preprocessing, and the SAM decoder
   conv-transpose k2s2 kernel, the 1024 path is still about 1.56x slower than
   official PyTorch. The next speed target is Hiera global/window
   FlashAttention and stage-2 quantized MLP matmul shapes rather than the SAM
   decoder upsampling kernel.
3. Decide whether lower encode sizes are acceptable for the Rust wrapper. This
   requires same-resolution measurements: C++ 512 must be compared with
   official PyTorch 512, C++ 640 with official PyTorch 640, and so on, with the
   same decoded source-frame resolution in both runs.
4. Continue reducing CPU preprocessing cost or move it to GPU. Threaded fused
   resize/normalize is parity-preserving and helps, but the remaining CPU cost
   is still visible relative to the PyTorch baseline.
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
outputs/hiera-matmul-shapes/base_plus_q4_0_1024_two_frame_summary.json
outputs/hiera-current-precision/summary.json
outputs/sam2-official-quality-10-current-q8_0/summary.json
outputs/hiera-force-mmq/summary.json
outputs/hiera-window-matmul-flatten/parity.json
outputs/model-matrix-sam2-base-plus-current-1024/summary.json
outputs/model-matrix-sam2-base-plus-current-512/summary.json
outputs/parity-sam2-base-plus-q4-1024-shape-key-default/summary.json
outputs/parity-sam2-base-plus-q4-512-shape-key-default/summary.json
outputs/hiera-shape-key-profile-current/q4_0_512_labeled_profile_summary.json
outputs/propagate-profile-current/q4_0_512_graph_ops_summary.json
outputs/model-matrix-sam2-base-plus-q4-quiet-1024/summary.json
outputs/model-matrix-sam2-base-plus-q4-quiet-512/summary.json
outputs/preprocess-threaded/q4_0_1024_profile_summary.json
outputs/preprocess-threaded/paired_threads_stats/summary.json
outputs/model-matrix-sam2-base-plus-q4-threaded-1024/summary.json
outputs/model-matrix-sam2-base-plus-q4-threaded-512/summary.json
outputs/preprocess-threaded/q4_0_1024_force_cublas.log
outputs/preprocess-threaded/q4_0_512_force_cublas.log
outputs/conv-transpose-k2s2/paired_summary.json
outputs/conv-transpose-k2s2/q4_0_1024_bbox_parity.json
outputs/conv-transpose-k2s2/q4_0_512_bbox_parity.json
outputs/cuda-node-profile-current/q4_0_1024_nodes_summary.json
outputs/fattn56-nbatch64/fullmask_parity.json
outputs/preprocess-float-resize/fullmask_parity.json
outputs/preprocess-float-resize/default_profile_summary.json
outputs/preprocess-float-resize/float_profile_summary.json
outputs/preprocess-norm-lut/fullmask_parity.json
outputs/preprocess-norm-lut/lut_profile_summary.json
outputs/hiera-inplace-bias/q4_0_1024_parity.json
outputs/hiera-inplace-bias/q4_0_512_parity.json
outputs/model-matrix-current-rerun-1024/summary.json
outputs/model-matrix-current-rerun-512/summary.json
outputs/model-matrix-frame-index-1024/summary.json
outputs/model-matrix-frame-index-512/summary.json
outputs/tracker-frame-index/q4_0_1024_profile_summary.json
outputs/tracker-frame-index/q4_0_512_profile_summary.json
outputs/sam2-official-quality-frame-index-1024/summary.json
outputs/sam2-official-quality-frame-index-512/summary.json
```
