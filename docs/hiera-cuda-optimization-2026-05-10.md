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
when both runs use the same input-resolution contract:

- the same decoded source-frame resolution before preprocessing,
- the same frame range,
- the same prompt,
- the same SAM2 model input resolution after preprocessing
  (`--encode-img-size` in C++, `model.image_size` in official PyTorch).

The primary comparison must keep both resolution layers fixed. Running multiple
source resolutions or encode sizes is still useful, but those rows are a
scaling study; they must not be used as a cross-implementation win/loss
comparison unless the matching PyTorch run uses the same decoded source
resolution and the same model input resolution. In short: use matched
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

Releasing the previous frame's Hiera encoder output buffer before allocating
the next Hiera graph removes the mid-run 512 Hiera compute outliers in the
profile path while preserving the neck PE cache. Full-mask parity against the
pre-change artifact is exact (`mask_hash_equal_rows=10/10`). The same-input
speed comparison improves the 512 row modestly, but it still does not overtake
official PyTorch:

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 track ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| 1024 | 89.6 | 57.36 | 0.640 | `outputs/model-matrix-free-prev-state-1024/summary.json` |
| 512 | 23.1 | 20.10 | 0.870 | `outputs/model-matrix-free-prev-state-512/summary.json` |

The accepted effect is therefore stability and a small 512 gain, not completion
of the PyTorch-speed target. The remaining gap still points at steady Hiera
FlashAttention / quantized MLP kernels rather than CPU graph build or output
copy overhead.

The Python comparison script now times the same tracking range as the C++
benchmark: frame 0 point-prompt/add-instance is excluded, and only frames
`1..N-1` are averaged. With that stricter contract, official PyTorch is
substantially faster than the earlier all-yield average suggested:

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 track ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| 1024 | 89.2 | 43.76 | 0.491 | `outputs/model-matrix-free-prev-state-1024-trackrange/summary.json` |
| 512 | 24.7 | 12.56 | 0.508 | `outputs/model-matrix-free-prev-state-512-trackrange/summary.json` |

This makes the remaining target larger: matching PyTorch requires roughly a
2x reduction in the steady C++ tracking path, not just removing the previous
short-run outliers.

Official PyTorch was also instrumented with synchronized wrappers around
`forward_image`, `track_step`, and `_run_memory_encoder`. These wrapper timings
add synchronization overhead and are not the speed baseline, but they identify
where the official implementation spends time:

| Encode size | Instrumented PyTorch track ms/frame | `forward_image` mean | `track_step` mean | Memory encoder | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 47.62 | 30.41 | 15.35 | 1.75 | `outputs/python-official-profile/base_plus_1024.json` |
| 512 | 23.22 | 15.92 | 6.60 | 3.03 | `outputs/python-official-profile/base_plus_512.json` |

The comparable C++ synchronized profile still spends much more time in Hiera:
latest 1024 `hiera_encode` steady values are about `65-69 ms/frame`, with
`propagate_single` around `11-12 ms/frame`; latest 512 `hiera_encode` steady
values are about `16.7-17.0 ms/frame`, with `propagate_single` around
`3.7-3.9 ms/frame`. The priority remains Hiera encode kernels, especially
`head_dim=56` FlashAttention and q4 MLP matmul shapes. Propagation and memory
encoding are secondary unless Hiera is first brought closer to the PyTorch
`forward_image` range.

`outputs/hiera-gap-priority/summary.json` records the stage-gap summary derived
from those synchronized profiles. The 512 row is close at the Hiera/forward
stage alone (`16.87 ms` C++ steady Hiera versus `15.92 ms` PyTorch
`forward_image`), while the 1024 row still has a large Hiera gap (`66.46 ms`
versus `30.41 ms`). This makes 1024 Hiera encode the first-priority target; the
512 end-to-end gap also needs non-Hiera overhead work, but that should not pull
attention away from the larger 1024 encoder gap.

Rejected follow-up experiments:

- `GGML_CUDA_FORCE_CUBLAS=1` for q4_0 matmuls did not improve the remaining
  gap. The 512 row regressed to `31.7 ms/frame` with mid-run encode outliers,
  and 1024 stayed at `90.4 ms/frame`.
- Forcing Hiera FlashAttention V tensors through `ggml_cont` before
  `ggml_flash_attn_ext` regressed the 512 row to `25.2 ms/frame`; the added
  copy cost is larger than any stride-handling benefit.
- Switching Base+ from q4_0 to f16 does not recover the PyTorch gap on the
  current CUDA path. Latest f16 measurements are `37.2 ms/frame` at 512 and
  `110.0 ms/frame` at 1024, both slower than q4_0. Latest q8_0 checks were also
  slower (`33.6 ms/frame` at 512, `92.1 ms/frame` at 1024).
- Changing the NVIDIA FP32 `head_dim=56` FlashAttention tile config from
  `nbatch_K=56` to `28` compiled, but did not improve speed. The measured rows
  were `24.8 ms/frame` at 512 and `89.1 ms/frame` at 1024, effectively equal to
  the current baseline.
- Padding Hiera `head_dim=56` Q/K/V tensors to 64 only around
  `ggml_flash_attn_ext` gave a small single-run 1024 bbox-only speed improvement
  (`89.3 -> 84.7 ms/frame`) but broke full-mask parity against the default path:
  `mask_hash_equal_rows=0/10`, `min_bbox_iou=0.0111`, and
  `max_score_abs_delta=0.4663`. Splitting the experiment into global-only and
  window-only attention did not recover parity either (`mask_hash_equal_rows=0/10`
  for both). This is not an acceptable optimization.
- Forcing only the repeated Hiera stage-2 q4 MLP shapes
  `q4_0[448,1792] x f32[448,4096]` and
  `q4_0[1792,448] x f32[1792,4096]` away from MMQ to the cuBLAS fallback also
  regressed the 1024 bbox-only row (`89.2 -> 92.5 ms/frame`) and increased RSS
  (`609.9 -> 658.1 MiB`). This confirms the previous global
  `GGML_CUDA_FORCE_CUBLAS=1` rejection is not just caused by unrelated small
  matmuls.
- Switching the two Hiera block residual adds to `ggml_add_inplace` preserved
  512 full-mask parity against the current artifact (`mask_hash_equal_rows=10/10`),
  but did not improve the 1024 bbox-only row in a single-run check
  (`89.2 -> 90.0 ms/frame`). It was reverted because it does not move the
  PyTorch-speed target.

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

`scripts/summarize_quality_gap.py` summarizes the latest frame-index-fixed
quality artifacts. These rows pass the same-input contract: decoded source
frames are `1008x568`, frame count/prompt match, and each row uses the same
SAM2 input encode size on the C++ and official PyTorch sides. The gap is not a
small numeric/hash drift: at 512, C++ masks average about `0.784x` the official
Python mask area and mean mask IoU is `0.7761`; at 1024, C++ masks average
about `0.567x` the official Python mask area, mean mask IoU is `0.4742`, and
official Python has an empty mask at offset 8 while C++ still tracks a non-empty
mask. The current quality task is therefore mask-selection / semantic parity,
not just floating-point tolerance.

`scripts/diagnose_sam2_initial_masks.py` now captures the official SAM2 initial
point-prompt candidates by wrapping `_forward_sam_heads`. On the 1024 Base+
case, official PyTorch uses `multimask_output=true` and selects candidate 0
(`iou=0.1796875`, area `31954`, bbox `[195,87,597,565]`). The C++ benchmark can
now emit its own initial candidates with `--output-initial-candidates-jsonl`.
For q4_0 with `--multimask`, C++ assigns the highest IoU to candidate 2
(`iou=0.515116`, area `15980`, bbox `[214,248,393,460]`), while candidate 0 is
larger but lower-scored (`iou=0.433599`, area `23378`, bbox
`[194,169,506,546]`). A f32 C++ spot-check flips the selection back to candidate
0 (`iou=0.399207`, area `30251`, bbox `[194,96,584,558]`), much closer to the
official selected mask. This narrows the Base+ q4_0 quality gap to multimask
candidate scoring/selection under quantization rather than a frame-order issue.

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

`scripts/summarize_cuda_node_hotspots.py` extracts this ranking from the
profile summary and tags the primary/secondary kernel classes. The generated
`outputs/cuda-node-profile-current/q4_0_1024_hiera_hotspots.json` should be
used after future ggml kernel experiments to confirm that a change actually
reduces the intended hotspot instead of moving noise between unrelated nodes.

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

The source-level kernel selection check confirms this is a ggml kernel-design
task. In `ggml/src/ggml-cuda/fattn.cu`, NVIDIA tensor-core MMA is intentionally
skipped for head sizes 40, 56, and 72, so SAM2 Base+ head_dim 56 is routed to
the tile kernel. In `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, the instantiated MMA
cases cover 64, 80, 96, 112, 128, 256, and a few large GQA shapes, but no
`DKQ=56,DV=56` cases. Therefore the remaining 1024 Hiera gap cannot be closed
by another allowlist or shape-key tweak; it needs either a real 56-aware
MMA/WMMA path or a redesigned tile kernel for the Base+ global/window shapes.

A prototype 56-aware MMA path was tried by instantiating `DKQ=56,DV=56` and
zero-padding the internal half2 MMA tiles to the next 8-half2 boundary while
keeping the external tensor shape at 56. It compiled and improved the rough
1024 q4_0 bbox timing from about `89 ms/frame` to `82 ms/frame`, but it failed
the parity gate: bbox output collapsed after frame 0 (`mask_hash_equal_rows=0/10`,
`min_bbox_iou=0.0`, `max_score_abs_delta=0.971792`). Restricting the experiment
to only the global `4096x4096` attention or only the window attention still
failed bbox parity. This confirms that the speed target needs a correctness
first kernel implementation, not just adding 56 instantiations to the existing
f16 MMA path. The restored tile build was rechecked against the reference JSONL
with exact bbox/mask parity (`mask_hash_equal_rows=10/10`).

The same conclusion was reconfirmed with an env-gated rerun that routed
`head_dim=56` through the existing generic f16 MMA dispatch while keeping the
external shape at 56. It improved 1024 q4_0 timing (`93.9 -> 81.1 ms/frame` with
the default MMA column selection, `93.9 -> 85.5 ms/frame` when capped at
`ncols=32`), but both variants produced empty masks from the first frame
(`bbox=[1008,568,0,0]`, `mask_area=0`) and then stayed empty. This is a broken
kernel path, not acceptable numerical drift. The experiment was reverted; only
the generator/extern cleanup needed to reproduce the existing tile
instantiations was kept.

To keep future kernel work from reaching this late failure mode again,
`sam3_fattn_parity` now checks `ggml_flash_attn_ext` directly against a scalar
CPU reference. It matches the CUDA path's fp16 K/V conversion before comparing,
so it is a kernel-output gate rather than a full-model proxy. Current valid
tile-path checks pass for the Hiera window shape
(`D=56,N=196,heads=8,batch=25`, `max_abs=0.0321010835`) and sampled global shape
(`D=56,N=4096,heads=8,batch=1,sampled_q=9`, `max_abs=0.00512025505`) under the
default `0.04` absolute tolerance.

The tile FlashAttention launch was also tested with `stream_k=true` to check
whether the long 4096-token global attention was limited by the current
parallel-block combine path. It produced a large rough 1024 bbox speedup
(`91.3 -> 72.6 ms/frame`) but invalid output immediately (`det=0`; frame 0 bbox
collapsed to `[1008,568,0,0]`). This is not a valid switch: the existing tile
kernel indexes work by `blockIdx.x/y/z`, while `stream_k` uses a continuous block
mapping that only kernels written for that contract can interpret. The result is
useful only as direction: a correct head_dim 56 stream-k/no-mask kernel could be
worth pursuing, but the generic tile kernel cannot be enabled with the
`launch_fattn` flag alone.

For the non-stream-k tile path, forcing the number of KV parallel blocks was
also tested as a scheduling-level alternative to kernel rewriting. Fixed values
`1,2,4,8,16` all preserved successful execution, but did not improve the 1024
bbox timing (`90.1-92.4 ms/frame`, comparable to the current baseline) and
changed the initial bbox/hash through different reduction order. This means the
existing auto-selected parallel-block schedule is not the main remaining
bottleneck; the speedup seen with invalid stream-k still points to a kernel that
is explicitly written for the stream-k contract.

## Kernel-Level Optimization Direction

Further work should not be limited to changing the current tile kernel
parameters. The remaining gap is large enough that the next branch should treat
`head_dim=56` as a kernel-design problem:

1. Add a SAM2 Base+ specific CUDA FlashAttention path for the observed Hiera
   shapes, initially limited to `Q/K/V=[56,4096,8,1]` and
   `Q/K/V=[56,196,8,25]`.
2. Keep the external ggml tensor contract exactly `[56,N,heads,batch]`; any
   padding to tensor-core friendly widths must be internal to the kernel and
   must zero masked lanes before both `KQ` and `VKQ`.
3. Do not route this through the existing generic MMA templates unless the
   remainder lanes are proven correct. The previous prototype showed that
   "instantiate 56 and pad locally" is not enough.
4. Consider a dedicated no-mask, no-sink, no-GQA kernel first. Hiera image
   encode calls `ggml_flash_attn_ext` without an attention mask, so the fastest
   path can skip mask loading, GQA handling, sink handling, and stream-k fixup
   branches that exist for LLM KV-cache workloads.
5. If the dedicated kernel is still too slow, split the problem further:
   specialize global attention (`4096x4096`) and window attention
   (`196x196 x 25 windows`) separately. These have different occupancy and
   memory-reuse constraints and should not be forced through one generic
   schedule.
6. Treat QKV projection and output projection as the next kernel-fusion target
   only after attention parity is stable. Fusing projection with attention has
   more surface area because Hiera uses q4 model weights and ggml graph
   allocation/layout assumptions; it should not be the first correctness risk.

### Acceptance Criteria For The Next Kernel Branch

- Correctness:
  - Full-mask parity against the current tile path for 10 same frames at
    `encode-img-size=512` and `1024`.
  - `mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, and
    `max_score_abs_delta=0.0` for the tile-vs-new-kernel comparison.
  - Separate global-only and window-only toggles must both pass before enabling
    the combined path by default.
- Performance:
  - Report paired runs, not single rows, for `q4_0` Base+ at `512` and `1024`.
  - The 1024 `track_ms` mean must beat the current tile baseline by at least
    15% before the kernel is considered worth merging.
  - The target remains official PyTorch parity or better on the matched
    decode/source-resolution/input-resolution protocol.
- Scope:
  - Gate the new path by exact shape and CUDA capability first.
  - Fall back to the current tile path for all unsupported shapes or any future
    model whose head dimension/layout differs.
  - Keep documentation and generated benchmark artifacts free of local absolute
    user paths.

`scripts/summarize_goal_audit.py` combines the matched C++/official-Python
speed rows, official-Python quality checks, Hiera stage gap, and CUDA node
hotspots into a single completion audit. The current audit artifact is
`outputs/goal-audit/sam2-base-plus-cuda-vs-python.json`; it marks the profiling
and prioritization work as complete, but keeps the overall objective
`not_complete` because C++ is still slower than official PyTorch on matched
512/1024 rows and Base+ mask quality is not at parity.

The head_dim 56 tile column width was also tested. Forcing smaller tile widths
was slower on the same 1024 q4_0 benchmark: 16 cols measured `112.5 ms/frame`,
8 cols `135.3 ms/frame`, and 4 cols `141.3 ms/frame` against the default
`101.7 ms/frame`. A diagnostic 64-col variant was slightly faster over five
paired runs (`101.78 -> 100.58 ms/frame`, mean `1.20 ms/frame` saved), but it
changed full-mask output (`mask_hash_equal_rows=0/10`, `min_bbox_iou=0.9645`,
`max_bbox_delta_px=3.0`, `max_score_abs_delta=0.0847`). This is not acceptable
as a parity-preserving optimization, so the default 32-col tile path remains in
place.

Specializing the `head_dim=56` tile path for Hiera's mask-free
`ggml_flash_attn_ext` calls removes the inner mask-add branch only when
`DKQ=56,DV=56` and `mask == nullptr`; other head sizes keep the previous
runtime path. The direct kernel parity checks still pass with the same error
range as the baseline:

| Shape | Max abs | Mean abs | Bad values |
| --- | ---: | ---: | ---: |
| `D=56,N=196,heads=8,batch=25` | `0.0321010835` | `0.00512901675` | `0/2195200` |
| `D=56,N=4096,heads=8,batch=1,sampled_q=9` | `0.00512025505` | `0.00105904906` | `0/4032` |

A short same-input A/B run showed only a small 1024 improvement candidate and
no 512 improvement: restored baseline measured `90.8 ms/frame` at 1024 and
`23.2 ms/frame` at 512; the mask-free specialization measured `88.8 ms/frame`
at 1024 and `23.2 ms/frame` at 512. This is useful as a low-risk cleanup, but
it is not a major speed fix and does not change the conclusion that PyTorch is
still much faster on the matched comparison.

The next root-level `head_dim=56` attempt moved the D56-to-D64 padding into
ggml's CUDA FlashAttention op instead of adding SAM graph pad/view nodes. For
mask-free f32 Q/K/V on tensor-core GPUs, ggml now zero-pads temporary Q/K/V
buffers to D64, runs the existing D64 f16 MMA FlashAttention path, and slices
the first 56 output lanes back to the original destination. The old tile path
can be restored with `GGML_CUDA_DISABLE_FATTN56_PAD_MMA=1`.

This is not bit-identical to the tile path because the f16 MMA accumulation is a
different numeric path, but the direct attention parity checks stay within the
same tolerance used for the existing CUDA FlashAttention checks:

| Shape | Max abs | Mean abs | Bad values | Non-finite |
| --- | ---: | ---: | ---: | ---: |
| `D=56,N=196,heads=8,batch=25` | `0.0320834816` | `0.00512893543` | `0/2195200` | `0` |
| `D=56,N=4096,heads=8,batch=1,sampled_q=9` | `0.00513070542` | `0.00105931065` | `0/4032` | `0` |

On the 10-frame Base+ q4_0 1024 bbox-only run, the paired measurements were:

| Path | Track ms/frame runs | Mean | Stdev | P50 mean | P95 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tile baseline | `90.0, 90.0, 91.0` | `90.33` | `0.58` | `88.33` | `99.67` |
| ggml D56->D64 MMA | `81.0, 81.0, 81.0` | `81.00` | `0.00` | `79.00` | `88.67` |

That is a `10.33%` track-ms improvement on this sample. Full-mask output stays
semantically close but not hash-identical to the tile path:
`mask_hash_equal_rows=0/10`, `min_bbox_iou=0.9957537155`,
`max_bbox_delta_px=4.0`, `max_score_abs_delta=0.009795`, and
`max_abs_mask_area_rel_delta=0.0208719`. Treat this as a fast numeric path, not
as a bit-exact replacement.

The follow-up optimization kept Q as padded f32, but packed K/V directly into
padded f16 buffers before calling the existing D64 MMA path. This removes the
generic f32-contiguous K/V pad followed by the generic f32-to-f16 conversion in
`launch_fattn`. With the same 10-frame Base+ q4_0 1024 bbox-only setup:

| Path | Track ms/frame runs | Mean | Stdev | P50 mean | P95 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tile baseline | `89.0, 89.0, 90.0` | `89.33` | `0.58` | `87.00` | `99.67` |
| ggml D56->D64 MMA, direct K/V f16 pack | `78.0, 78.0, 78.0` | `78.00` | `0.00` | `76.00` | `85.00` |

That raises the measured improvement to `12.69%` on this sample. The direct
attention parity checks are unchanged from the previous D56-to-D64 MMA path, and
the full-mask semantic delta is also unchanged:
`mask_hash_equal_rows=0/10`, `min_bbox_iou=0.9957537155`,
`max_bbox_delta_px=4.0`, `max_score_abs_delta=0.009795`, and
`max_abs_mask_area_rel_delta=0.0208719`. The important acceptance point is that
this is a faster non-bit-exact numeric path; official PyTorch quality and speed
remain the primary gate for keeping it enabled by default.

A smaller follow-up fused the normal same-shape Q/K/V D56-to-D64 pack into one
CUDA launch while preserving the separate-pack fallback behind
`GGML_CUDA_DISABLE_FATTN56_COMBINED_PACK=1`. This does not change the numeric
path. Same-binary 1024 bbox-only paired checks showed:

| Path | Track ms/frame runs | Mean | Stdev | P50 runs | P95 runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Separate Q/K/V pack | `79.9, 79.9, 78.5` | `79.43` | `0.81` | `77.6, 77.8, 76.8` | `90.2, 89.3, 86.1` |
| Combined Q/K/V pack | `78.3, 78.2, 78.7` | `78.40` | `0.26` | `76.5, 76.3, 77.3` | `86.0, 86.0, 85.9` |

That is a small `1.30%` improvement over the direct K/V f16 pack path on this
sample. Full-mask JSONL output is identical between the separate and combined
pack paths (`diff_rows=0/10`), so this is a launch-count optimization, not a new
numeric approximation.

Metal already fuses `NORM + MUL (+ ADD)` for affine layer normalization. CUDA
had the equivalent RMSNorm fusion but not plain Norm fusion, even though Hiera
uses plain `ggml_norm` before applying norm weights and biases. The CUDA backend
now mirrors the Metal optimization for F32 Norm and keeps an A/B gate through
`GGML_CUDA_DISABLE_NORM_FUSION=1`.

Commit-history audit found that this was not only a missing kernel but also a
missing Metal-side graph behavior: Metal can fuse across non-compute
`RESHAPE`/`VIEW` nodes, while CUDA's generic `ggml_can_fuse` only sees adjacent
cgraph nodes. The CUDA Norm affine path now skips those non-compute nodes for
this exact pattern, so Hiera's weight/bias reshapes no longer block the fusion.
The fused affine step uses explicit round-to-nearest multiply and add operations
to preserve the old two-kernel rounding order.

Same-binary 1024 bbox-only paired checks showed a measurable improvement:

| Path | Track ms/frame mean | 95% CI | Median | Stdev |
| --- | ---: | ---: | ---: | ---: |
| Norm fusion disabled | `77.64` | `76.83..78.45` | `77.70` | `0.65` |
| Norm fusion enabled | `75.48` | `74.83..76.13` | `75.20` | `0.53` |

The paired mean delta is `2.16 ms/frame` (`2.87%`), with a 95% CI of
`1.13..3.19 ms/frame`. Full-mask JSONL parity is bit-identical
(`diff_rows=0/10`). The node profiler confirms Hiera Norm affine chains now
show `fused=1 skipped=4`, which means the CUDA path is actually crossing the
weight/bias reshape/view nodes instead of only handling already-adjacent chains.

The next lower-risk CUDA pass targeted the Hiera MLP `ADD(bias) + GELU` pattern
instead of changing the q4_0 matmul algorithm itself. Profiling showed that the
q4_0 MLP matmuls already use the MMQ path on the current NVIDIA target, while
the following bias add and GELU still launched separately for each MLP block.
The CUDA backend now fuses F32 `ADD + GELU`, `ADD + GELU_ERF`, and
`ADD + GELU_QUICK` through the existing broadcast add launcher, writing directly
to the unary node output. The fallback gate is
`GGML_CUDA_DISABLE_ADD_UNARY_FUSION=1`.

Same-binary 1024 q4_0 bbox-only paired checks showed:

| Path | Track ms/frame mean | Median | Stdev | Min..Max |
| --- | ---: | ---: | ---: | ---: |
| Add+GELU fusion disabled | `75.98` | `76.10` | `0.45` | `75.5..76.4` |
| Add+GELU fusion enabled | `74.50` | `74.50` | `0.82` | `73.7..75.7` |

The paired mean delta is `1.48 ms/frame` (`1.99%`). Full-mask JSONL parity is
bit-identical against the disabled path on the same 10-frame sample:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, and
`max_score_abs_delta=0.0`. Node profiling confirms the graph change: fused ADD
nodes with `skipped=1` appear 74 times, and standalone UNARY executions drop
from 137 to 63 in the 3-frame profile.

Metal also specializes common binary broadcast layouts to avoid the fully
generic broadcast kernel. CUDA still used the generic unraveling path for Hiera
F32 axis-broadcast bias adds. A CUDA fast path now handles the contiguous case
where `src0` and `dst` have the same shape and `src1` broadcasts along one
axis, including the already-fused `ADD + GELU` path. The fallback gate is
`GGML_CUDA_DISABLE_BIN_BCAST_AXIS_FAST=1`.

Same-binary 1024 q4_0 bbox-only paired checks showed a small improvement:

| Path | Track ms/frame mean | Median | Stdev | Min..Max |
| --- | ---: | ---: | ---: | ---: |
| Axis broadcast fast path disabled | `114.90` | `115.30` | `1.85` | `112.9..117.1` |
| Axis broadcast fast path enabled | `113.64` | `113.50` | `1.09` | `112.2..115.2` |

The paired mean delta is `1.26 ms/frame` (`1.11%`). This is below the larger
Norm and Add+GELU fusions but still parity-preserving on the same 10-frame
full-mask sample: `mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0.0`, `max_score_abs_delta=0.0`, and
`max_abs_mask_area_rel_delta=0.0`.

The same history pass checked other recent upstream or Metal-only candidates:
`CONV_TRANSPOSE_1D` is not in the SAM2/Base+ graph, `OUT_PROD` and `SNAKE` are
also absent, `DKQ=192,DV=128` FlashAttention support does not match the Hiera
`head_dim=56` shapes, and the older Metal direct `CONV_2D` /
`CONV_TRANSPOSE_2D` work already has CUDA direct kernels or the SAM-specific
`k2s2` specialization in this branch. The remaining active CUDA targets are
therefore still the profiled Hiera FlashAttention and q4_0 MLP matmul shapes,
not another obvious Metal-only op gap.

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

The head_dim 56 FlashAttention padded-MMA path was also tightened for the
common contiguous Hiera layout. The existing path already combines Q/K/V packing
and converts K/V directly to f16 before running the D64 MMA kernel. The new
contiguous pack/slice kernels avoid the generic stride decomposition for this
layout while preserving the generic fallback through
`GGML_CUDA_DISABLE_FATTN56_CONTIGUOUS_PACK=1`.

Standalone D56 checks still pass for both profiled shapes:
`N=196, heads=8, batch=25` and `N=4096, heads=8, batch=1` both report
`bad=0`. Full-mask JSONL parity against the generic pack/slice path is
bit-identical on the 10-frame 1024 q4_0 sample:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0.0`, and `max_score_abs_delta=0.0`.

Same-binary bbox-only paired checks show a small but non-negative improvement:

| Path | Track ms/frame mean | Median | Stdev | Min..Max |
| --- | ---: | ---: | ---: | ---: |
| Generic head56 pack/slice | `115.00` | `115.00` | `1.22` | `114.0..117.0` |
| Contiguous head56 pack/slice | `114.20` | `114.00` | `0.45` | `114.0..115.0` |

The paired mean delta is `0.80 ms/frame` (`0.70%`). The larger comparison is
still the padded-MMA path versus the original generic tile path: current A/B
shows `131.0 -> 115.0 ms/frame`, or about `13.9%`, so the padded-MMA strategy
remains the correct baseline while further work should target the D56
attention kernel itself.

An env-gated D56 stage profiler was added to ggml CUDA as
`GGML_CUDA_PROFILE_FATTN56=1`. It disables CUDA graph capture while active,
records CUDA events around pack, padded D64 MMA, and slice, and prints Q/K/V
shape and stride metadata. The profiler is diagnostic only; normal runs do not
create events or synchronize. The 1024 Base+ q4_0 stage profile shows that the
remaining head56 cost is mostly the MMA body, not the padding kernels:

| Shape | Combined pack | Mean pack ms | Mean MMA ms | Mean slice ms | Mean total ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q=[56,4096,8,1]` | yes | `0.0557` | `1.1161` | `0.0176` | `1.1894` |
| `Q=[56,196,8,25]` | yes | `0.0776` | `0.1987` | `0.0217` | `0.2980` |
| `Q=[56,64,2,1024]` | yes | `0.2962` | `0.3495` | `0.1020` | `0.7478` |

Two follow-up CUDA experiments were rejected from this evidence. A
Q/K-contiguous plus V-transposed-view pack kernel reduced pack time on some
small shapes, but normal five-pair 1024 runs regressed from `112.4` to
`113.2 ms/frame` on average. Retuning the padded D64 MMA `ncols=64` config for
Blackwell also regressed: `256` threads raised global D56 MMA to about
`1.89 ms`, and `nbatch_fa=128` raised it to about `1.33 ms`. The Ampere-style
`128` thread, `nbatch_fa=64` config remains the best measured setting on the
RTX 5070 Ti Laptop GPU so far.

The CUDA binary broadcast path now has a same-shape contiguous fast path for
ordinary binary ops. This is the CUDA analogue of avoiding unnecessary index
unraveling for non-broadcast elementwise work. It is gated by
`GGML_CUDA_DISABLE_BIN_CONTIGUOUS_FAST=1`. Full-mask parity against the generic
binary path is exact on the 10-frame 1024 q4_0 sample
(`mask_hash_equal_rows=10/10`). Node profiling shows the intended ADD reduction:
`ADD sum_ms_drop_max` moved from `41.78` to `37.31 ms` on the 3-frame 1024
profile. The normal five-pair bbox timing is a small improvement,
`113.4 -> 113.0 ms/frame` (`0.35%`), so this is useful hygiene but not a
PyTorch-closing change.

MMQ tile-size retuning was also checked for the q4 Hiera MLP hotspot. Changing
NVIDIA `mmq_y` from `128` to `64` is not viable as a global setting: existing
MMA MMQ template instantiations assert that the writeback tile height matches
the selected warp/tile shape. Capping `mmq_x` at runtime was buildable, but not
faster on the 1024 Base+ q4_0 bbox run: default `128` averaged
`114.0 ms/frame`, cap `96` averaged `114.3 ms/frame`, and cap `64` averaged
`116.3 ms/frame`. The existing MMQ x/y selection remains the best measured
setting for these shapes.

An attempted MMQ `MUL_MAT + ADD(bias)` fusion was rejected. Switching the
fusion matcher from CUDA same-shape fusion to generic adjacency made the
intended Base+ q4_0 MMQ nodes fire in the CUDA node profile, including the
`112x65536`, `1792x4096`, and `448x4096` Hiera shapes. However, adding bias
inside MMQ broke full-mask parity badly because the NVIDIA MMQ path uses
stream-k partial sums. Moving the bias to a single post-MMQ row-bias kernel
fixed the gross failure but still changed masks by about `1%` area on the
10-frame 1024 q4_0 sample (`mask_hash_equal_rows=0/10`,
`min_bbox_iou=0.998745`, `max_bbox_delta_px=1.0`). A control run comparing two
disabled-fusion executions was bit-identical (`mask_hash_equal_rows=10/10`), so
this was not run-to-run nondeterminism. The branch was reverted. Future MMQ
bias work should either preserve the original MUL_MAT output buffer and prove
bit-identical ADD replacement, or avoid this route and target MMQ arithmetic
throughput directly.

Precision alone does not explain the quality gap, but precision affects the
multimask candidate scoring. Re-running the same default 1024 comparison across
Base+ precisions without fixing candidate selection gave similarly poor
official-Python IoU:

| Model | C++ track ms/frame | Mean mask IoU | Min mask IoU | PyTorch ms/frame |
| --- | ---: | ---: | ---: | ---: |
| `sam2.1_hiera_base_plus_f32` | 147 | 0.0185 | 0.0000 | 43.3 |
| `sam2.1_hiera_base_plus_f16` | 148 | 0.0185 | 0.0000 | 43.4 |
| `sam2.1_hiera_base_plus_q8_0` | 135 | 0.0179 | 0.0000 | 43.4 |
| `sam2.1_hiera_base_plus_q4_0` | 135 | 0.0197 | 0.0000 | 43.6 |

An experimental C++ `--multimask` initial prompt path was also tested. It
selects the highest predicted IoU mask among SAM mask tokens 1..3 and uses the
matching mask token for the tracker object pointer. On the current build, this
shows that q4_0 is the outlier: q4_0 still selects the small candidate and stays
near the single-mask quality level, while f16/q8_0 select the same candidate
index as official PyTorch and improve the 1024 quality substantially. It is
still not parity and remains slower than official PyTorch.

| Model/mode | Encode size | C++ track ms/frame | Mean mask IoU | Frame-0 mask IoU | Min bbox IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4_0 single-mask prompt | 1024 | 89-96 | 0.4742 | 0.5020 | 0.2500 |
| q4_0 multimask prompt | 1024 | 96.2 | 0.4731 | 0.4959 | 0.1975 |
| q4_0 multimask, forced candidate 0 | 1024 | 96.4 | 0.6025 | 0.6375 | 0.3807 |
| q4_0 + f16 SAM decoder/prompt/obj_ptr | 1024 | 97.8 | 0.4740 | 0.5007 | 0.1379 |
| q4_0 + q8_0 Hiera/FPN | 1024 | 96.9 | 0.6747 | 0.7769 | 0.6290 |
| q4_0 + f16 Hiera/FPN | 1024 | 106.4 | 0.6253 | 0.7793 | 0.5314 |
| q4_0 + q4_1 Hiera/FPN | 1024 | 97.1 | 0.6007 | 0.6503 | 0.5985 |
| q4_1 multimask prompt | 1024 | 96.7 | 0.5835 | 0.6539 | 0.6093 |
| q8_0 multimask prompt | 1024 | 96.7 | 0.6999 | 0.8591 | 0.6594 |
| f16 multimask prompt | 1024 | 109.3 | 0.6995 | 0.8397 | 0.6439 |

The official initial candidate diagnostic for the same prompt selects candidate
0. C++ f32/f16/q8_0/q4_1 also score candidate 0 highest, while q4_0 scores
candidate 2 highest. A diagnostic `--initial-candidate-index 0` run confirms
that candidate selection is a real part of the q4_0 quality gap: mean mask IoU
improves from `0.4731` to `0.6025`, and frame-0 bbox IoU improves from `0.1975`
to `0.6090`. It still does not reach parity, and later frames remain vertically
shifted relative to official PyTorch. The remaining gap therefore cannot be
explained by q4_0 IoU ranking alone; the selected mask/object-pointer/memory
propagation path needs a deeper comparison before Base+ q4_0 is used as a
Rust-wrapper quality baseline.

A mixed precision diagnostic then kept Hiera/FPN/memory tensors from the q4_0
model and replaced only `sam_pe.*`, `sam_dec.*`, and `obj_ptr_proj.*` with f16
tensors. This required loader support for registering tensors from their file
dtype rather than assuming one dtype from the model-level `ftype`. The model
loaded and ran, but quality did not improve: mean mask IoU stayed at `0.4740`,
and the initial candidate dump still scored candidate 2 highest. That points
away from SAM decoder quantization as the main cause and toward q4_0 Hiera/FPN
features or memory propagation state as the next quality target.

The inverse mixed precision diagnostic confirms that direction. Keeping q4_0
for the rest of the model while replacing only `hiera.*` and `fpn.*` with q8_0
restores candidate-0 selection and raises mean mask IoU to `0.6747`, close to
the full q8_0 row. Replacing the same tensors with f16 also restores candidate-0
selection but reaches only `0.6253` and is slower. q4_1 Hiera/FPN reaches
`0.6007` at roughly the same measured track time as q8_0 Hiera/FPN, so the next
practical mixed-precision candidate is q8_0 Hiera/FPN rather than q4_1
Hiera/FPN. The quality target is therefore not "make the SAM decoder f16"; it is
to avoid q4_0 degradation in Hiera/FPN features, or to use a more faithful Hiera
quantization format.

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
   this sample. Mixed precision diagnostics now point to q4_0 Hiera/FPN feature
   degradation as the primary quality issue; q8_0 Hiera/FPN recovers most of
   the gap while f16 SAM decoder/prompt does not.
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
5. Keep `head_dim=56` FlashAttention as a ggml kernel-design task. The current
   D56-to-D64 MMA path gives a real speedup, but it is not bit-exact against the
   tile path. A dedicated D56 MMA kernel would need to handle padded lanes
   internally rather than simply allowlisting `DKQ=56,DV=56`.

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
outputs/cuda-node-profile-current/q4_0_1024_hiera_hotspots.json
outputs/goal-audit/sam2-base-plus-cuda-vs-python.json
outputs/quality-gap/sam2-base-plus-frame-index.json
outputs/sam2-initial-mask-diagnostics-1024/initial_candidates.jsonl
outputs/sam2-initial-mask-diagnostics-1024/cpp_initial_candidates.jsonl
outputs/sam2-initial-mask-diagnostics-1024/cpp_initial_candidates_f32.jsonl
outputs/sam2-initial-mask-diagnostics-1024/cpp_initial_candidates_f16.jsonl
outputs/sam2-initial-mask-diagnostics-1024/cpp_initial_candidates_q8_0.jsonl
outputs/sam2-initial-mask-diagnostics-1024/cpp_initial_candidates_q4_1.jsonl
outputs/sam2-official-quality-1024-f16-multimask-diagnostic/summary.json
outputs/sam2-official-quality-1024-q8_0-multimask-diagnostic/summary.json
outputs/sam2-official-quality-1024-q4_1-multimask-diagnostic/summary.json
outputs/sam2-official-quality-1024-q4_0-multimask-diagnostic/summary.json
outputs/fattn56-mma/q4_0_1024_bbox.log
outputs/fattn56-mma/q4_0_1024_global_only_bbox.log
outputs/fattn56-mma/q4_0_1024_window_only_bbox.log
outputs/fattn56-mma/default_vs_mma_bbox.json
outputs/fattn56-mma/default_vs_restored_bbox.json
outputs/fattn56-mma-gated/default_1024.log
outputs/fattn56-mma-gated/mma_1024.log
outputs/fattn56-mma-gated/mma_ncols32_1024.log
build/xmake-release-cuda/examples/sam3_fattn_parity --cuda --d 56 --n 196 --heads 8 --batch 25
build/xmake-release-cuda/examples/sam3_fattn_parity --cuda --d 56 --n 4096 --heads 8 --batch 1 --sample-queries 9
outputs/fattn56-nbatch64/fullmask_parity.json
outputs/fattn56-pad-mma/final/bbox_stats.json
outputs/fattn56-pad-mma/final/fullmask_parity_summary.json
outputs/fattn56-pad-mma/kv-f16/bbox_stats.json
outputs/fattn56-pad-mma/kv-f16/fullmask_summary.json
outputs/root-perf-next/fattn56-combined-pack/bbox_stats.json
outputs/root-perf-next/fattn56-combined-pack/fullmask/summary.json
outputs/root-perf-next/cuda-norm-fusion/bbox_stats.json
outputs/root-perf-next/cuda-norm-fusion/fullmask/summary.json
outputs/root-perf-next/hiera-norm-affine-fusion/bbox_stats.json
outputs/root-perf-next/hiera-norm-affine-fusion/fullmask/rn_summary.json
outputs/root-perf-next/hiera-norm-affine-fusion/profile/q4_0_1024_inplace_nonseq_profile.log
outputs/root-perf-next/add-gelu-fusion/bbox_stats.json
outputs/root-perf-next/add-gelu-fusion/fullmask/summary.json
outputs/root-perf-next/add-gelu-fusion/profile/fused_profile_summary.json
outputs/root-perf-next/add-gelu-fusion/profile/no_fusion_profile_summary.json
outputs/root-perf-next/bin-bcast-axis/bbox_stats.json
outputs/root-perf-next/bin-bcast-axis/fullmask/summary.json
outputs/root-perf-next/metal-gap-refresh/q4_0_1024_hotspots.json
outputs/root-perf-next/metal-gap-refresh/q4_0_1024_profile_summary.json
outputs/root-perf-next/fattn56-contiguous-pack/bbox_stats.json
outputs/root-perf-next/fattn56-contiguous-pack/fullmask/summary.json
outputs/root-perf-next/fattn56-current-ab/
outputs/root-perf-next/mmq-bias-fusion/profile-verify/fusion_summary.json
outputs/root-perf-next/mmq-bias-fusion/fullmask-clean/summary.json
outputs/root-perf-next/mmq-bias-fusion/fullmask-postadd/summary.json
outputs/root-perf-next/mmq-bias-fusion/fullmask-skip-unary/summary.json
outputs/root-perf-next/mmq-bias-fusion/nondet-check/summary.json
outputs/root-perf-next/mmq-bias-fusion/rejected-postadd-wip.patch
outputs/root-perf-next/fattn56-stage-profile/q4_0_1024_nb.log
outputs/root-perf-next/fattn56-qk-v-pack/
outputs/root-perf-next/fattn64-config-256t/
outputs/root-perf-next/fattn64-config-nbatch128/
outputs/root-perf-next/bin-contiguous-fast/fullmask/summary.json
outputs/root-perf-next/bin-contiguous-fast/bbox/stats.json
outputs/root-perf-next/bin-contiguous-fast/profile/enabled_summary.json
outputs/root-perf-next/bin-contiguous-fast/profile/disabled_summary.json
outputs/root-perf-next/mmq-x-cap/bbox/stats.json
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
outputs/sam2-official-quality-1024-q4_0-multimask-candidate0/summary.json
outputs/sam2-official-quality-1024-q4_0-multimask-candidate0/cpp_initial_candidates.jsonl
outputs/sam2-official-quality-1024-q4_0-samdec-f16-multimask/summary.json
outputs/sam2-official-quality-1024-q4_0-samdec-f16-multimask/cpp_initial_candidates.jsonl
outputs/sam2-mixed-q4-samdec-f16-summary.json
outputs/sam2-official-quality-1024-q4_0-hiera-q8_0-multimask/summary.json
outputs/sam2-official-quality-1024-q4_0-hiera-q8_0-multimask/cpp_initial_candidates.jsonl
outputs/sam2-official-quality-1024-q4_0-hiera-f16-multimask/summary.json
outputs/sam2-official-quality-1024-q4_0-hiera-f16-multimask/cpp_initial_candidates.jsonl
outputs/sam2-official-quality-1024-q4_0-hiera-q4_1-multimask/summary.json
outputs/sam2-official-quality-1024-q4_0-hiera-q4_1-multimask/cpp_initial_candidates.jsonl
outputs/sam2-mixed-q4-hiera-q8-summary.json
outputs/sam2-mixed-q4-hiera-f16-summary.json
outputs/sam2-mixed-q4-hiera-q4_1-summary.json
outputs/hiera-free-prev-state/q4_0_512_fixed_summary.json
outputs/hiera-free-prev-state/q4_0_512_fullmask_summary.json
outputs/hiera-free-prev-state/q4_0_512_force_cublas_summary.json
outputs/hiera-free-prev-state/q4_0_1024_force_cublas_summary.json
outputs/hiera-v-cont/q4_0_512_summary.json
outputs/hiera-free-prev-state/f16_512_summary.json
outputs/hiera-free-prev-state/f16_1024_summary.json
outputs/hiera-free-prev-state/q8_0_512_summary.json
outputs/hiera-free-prev-state/q8_0_1024_summary.json
outputs/fattn56-nbatchK28/q4_0_512_summary.json
outputs/fattn56-nbatchK28/q4_0_1024_summary.json
outputs/hiera-pad-head64/pad64_1024_parity.json
outputs/hiera-pad-head64/global_1024_parity.json
outputs/hiera-pad-head64/window_1024_parity.json
outputs/hiera-mlp448-cublas/default_1024.log
outputs/hiera-mlp448-cublas/mlp448_1024.log
outputs/hiera-residual-inplace/q4_0_512_parity.json
outputs/hiera-residual-inplace/q4_0_1024_bbox.log
outputs/model-matrix-free-prev-state-1024/summary.json
outputs/model-matrix-free-prev-state-512/summary.json
outputs/model-matrix-free-prev-state-1024-trackrange/summary.json
outputs/model-matrix-free-prev-state-512-trackrange/summary.json
outputs/python-official-profile/base_plus_1024.json
outputs/python-official-profile/base_plus_512.json
outputs/hiera-gap-priority/summary.json
outputs/hiera-free-prev-state/q4_0_1024_profile_summary.json
```
