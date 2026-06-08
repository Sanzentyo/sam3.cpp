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

After the ggml CUDA follow-ups for axis broadcast add, head_dim=56 packing,
and same-shape contiguous binary ops, the latest matched rerun still does not
overtake official PyTorch. This rerun uses the current default test video
decoded at `960x540`; the source resolution differs from some earlier
`1008x568` rows, so it should be read as the latest current-state speed check,
not as a replacement for older profile attribution artifacts.

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 track ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| 1024 | 116.0 | 64.53 | 0.556 | `outputs/model-matrix-latest-q4-1024/summary.json` |
| 512 | 24.3 | 14.98 | 0.616 | `outputs/model-matrix-latest-q4-512/summary.json` |

The conclusion is unchanged: C++ CUDA remains slower than official PyTorch on
the matched input contract. At 1024 the current C++ path needs about `1.8x`
speedup to match PyTorch, and at 512 it needs about `1.6x`.

The same current-state check was repeated for the Base+ precision candidates
that matter for quality. `q8_0` remains the practical quality fallback
candidate because its speed is close to `q4_0` in this short run, while `f16`
is clearly slower. Neither precision closes the official PyTorch speed gap:

| Precision | Encode size | C++ track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| q4_0 | 1024 | 116.0 | 64.53 | 0.556 | `outputs/model-matrix-latest-q4-1024/summary.json` |
| q8_0 | 1024 | 114.7 | 66.62 | 0.581 | `outputs/model-matrix-latest-q8_0-1024/summary.json` |
| f16 | 1024 | 134.4 | 65.72 | 0.489 | `outputs/model-matrix-latest-f16-1024/summary.json` |
| q4_0 | 512 | 24.3 | 14.98 | 0.616 | `outputs/model-matrix-latest-q4-512/summary.json` |
| q8_0 | 512 | 24.3 | 14.57 | 0.600 | `outputs/model-matrix-latest-q8_0-512/summary.json` |
| f16 | 512 | 28.2 | 14.62 | 0.518 | `outputs/model-matrix-latest-f16-512/summary.json` |

This means quality-driven fallback selection should prefer `q8_0` over `f16`,
but speed work still has to target the shared Hiera CUDA path rather than model
precision alone.

A more direct `head_dim=56` MMA experiment was also attempted by adding D56/D56
MMA template configurations and routing the no-mask SAM2 Hiera attention path to
that kernel under an explicit env gate. It built after regenerating CUDA
template instances, but the standalone attention parity check produced invalid
results during the WIP test (`nonfinite=6104/7168` at `N=196`, and
`nonfinite=3584/3584` at `N=4096`). The experiment was reverted before commit.
This rules out "just instantiate DKQ=56/DV=56 in the existing MMA kernel" as a
safe shortcut; a real D56 kernel needs changes inside the MMA tile/combine
assumptions, not only new config rows.

Official PyTorch was also instrumented with synchronized wrappers around
`forward_image`, `track_step`, and `_run_memory_encoder`. These wrapper timings
add synchronization overhead and are not the speed baseline, but they identify
where the official implementation spends time:

| Encode size | Instrumented PyTorch track ms/frame | `forward_image` mean | `track_step` mean | Memory encoder | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 42.62 | 28.09 | 13.11 | 1.44 | `outputs/python-official-profile/base_plus_1024-final-20260516.json` |
| 512 | 23.22 | 15.92 | 6.60 | 3.03 | `outputs/python-official-profile/base_plus_512.json` |

The comparable current C++ q4_1/1024 synchronized profile is now much closer
than the older rows: steady `hiera_encode` after the first frame is about
`32.4 ms/frame` (`33.724, 33.766, 33.963, 33.235, 30.850, 31.601, 31.689,
31.521, 31.531`), while `propagate_single` averages `10.06 ms/frame`. Against
the refreshed official PyTorch profile, Hiera still has about a `4.3 ms/frame`
steady gap versus `forward_image=28.09 ms`; propagation is already faster than
the instrumented PyTorch `track_step=13.11 ms`. The priority remains Hiera
encode kernels, especially `head_dim=56` FlashAttention and q4_1 MLP/QKV MMQ
shapes. Propagation and memory encoding are secondary unless Hiera is first
brought to or below the PyTorch `forward_image` range.

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
- Reusing MMQ's quantized `src1` activation by raw device pointer looked
  promising in the profiler because many Hiera MMQ nodes see repeated device
  addresses, but the pointer-only cache was not semantically valid. ggml's graph
  allocator can reuse the same device address for different tensor contents
  inside one graph compute, and the first bbox-only parity check changed 31
  output fields across 10 rows. Tightening the cache key to include the
  `ggml_tensor *` restored exact output parity, but removed the useful reuse and
  regressed the no-graph 1024 short run from `45.4` to `50.2 ms/frame`. The
  cache experiment was therefore removed. Replacing `cudaMalloc/cudaFree` with
  the normal ggml CUDA pool is not a shortcut either: even with reverse-order
  cleanup, holding cache entries across nodes interfered with later cuBLASLt
  workspace allocation and aborted in `ggml_cuda_mul_mat_batched_cublaslt_bias_tf32`.
  A future fusion/cache pass needs graph-lifetime or producer identity
  information plus a dedicated non-LIFO scratch arena, not only device pointer
  equality or the existing temporary pool.

The 2026-05-17 MMQ profile with extra `src1_ptr` attribution quantifies why a
better fusion plan is still worth pursuing. On a 3-frame Base+ q4_1 profile,
dropping first-per-shape warmups, MMQ activation quantization accounted for
`4.87 ms` across the profiled MMQ rows, and `4.85 ms` of that appeared in
groups with repeated `src1_ptr` values. This is an upper-bound signal, not a
valid cache key. The safe next direction is a graph-aware fusion/cache design
that keys by producer tensor identity and liveness, or a fused producer->MMQ
path that never materializes stale pointer-based assumptions.

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

The Hiera input resize/normalize path now has a CUDA implementation that writes
directly into the allocated input tensor. It is enabled by default for CUDA
backends and can be disabled with `SAM3_DISABLE_CUDA_PREPROCESS=1`. The kernel
uses the same bilinear sampling, integer rounding, CHW layout, and ImageNet
normalization as the CPU path. A cached device RGB input buffer avoids per-frame
`cudaMalloc`/`cudaFree`.

Full-mask parity against the CPU preprocessing path is exact on the 10-frame
Base+ q4_0 sample:

| Check | Result | Evidence |
| --- | ---: | --- |
| rows | 10 vs 10 | `outputs/hiera-cuda-preprocess-default/parity/summary.json` |
| mask hash equal rows | 10/10 | `outputs/hiera-cuda-preprocess-default/parity/summary.json` |
| minimum bbox IoU | 1.0 | `outputs/hiera-cuda-preprocess-default/parity/summary.json` |
| maximum score delta | 0.0 | `outputs/hiera-cuda-preprocess-default/parity/summary.json` |

Five paired bbox-only runs show a small but real 1024 improvement and a noisier
512 improvement:

| Encode size | CPU preprocess mean | CUDA preprocess mean | Saved mean | Speedup | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 113.02 | 110.16 | 2.86 | 1.026x | `outputs/hiera-cuda-preprocess-cached/ab/summary.json` |
| 512 | 24.20 | 23.08 | 1.12 | 1.049x | `outputs/hiera-cuda-preprocess-cached/ab/summary.json` |

The matched official PyTorch comparison was rerun after enabling CUDA
preprocessing by default. The gap is smaller, but C++ CUDA still does not beat
official PyTorch:

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| 1024 | 111.4 | 68.24 | 0.613 | `outputs/model-matrix-cuda-preprocess-q4-1024/summary.json` |
| 512 | 22.5 | 14.55 | 0.647 | `outputs/model-matrix-cuda-preprocess-q4-512/summary.json` |

Additional root-level follow-ups on 2026-05-11 did not produce an accepted
speedup:

| Candidate | Result | Decision | Evidence |
| --- | --- | --- | --- |
| Merge current `ggml-org/ggml` upstream CUDA changes | Build required resolving local head56 conflicts; matched 1024 row was `112.3 ms/frame` C++ vs `68.91 ms/frame` PyTorch | Rejected; no speed gain over the current `111.4 ms/frame` CUDA-preprocess baseline | `outputs/upstream-merge/model-matrix-q4-1024/summary.json` |
| Keep Hiera graph allocation alive and use FPN outputs directly as state tensors | Full parity passed, but matched 1024 row regressed to `129.9 ms/frame` | Rejected; retaining the large Hiera allocation hurts the following propagation path more than it saves copy time | `outputs/hiera-retain-graph/model-matrix-q4-1024/summary.json` |
| Alias Hiera positional embedding graph input to the cached backend tensor | Full-mask parity was exact (`mask_hash_equal_rows=10/10`), but five paired bbox-only runs saved only `0.74 ms/frame` on average | Rejected; the small gain does not justify mutating internal ggml tensor buffer/data pointers | `outputs/hiera-alias-pos/compare.json`, `outputs/hiera-alias-pos/ab/` |

These results narrow the remaining useful work. Avoid broad upstream merges or
state-lifetime rewrites unless they are tied to a measured Hiera hotspot. The
first-priority CUDA target remains the steady 1024 Hiera graph body:
`head_dim=56` FlashAttention and the repeated q4 stage-2 MLP matmul shapes.
Secondary work can target small graph-input copies only after the Hiera kernel
gap is reduced, because those copies are now below the scale required to catch
official PyTorch.

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

The current `960x540` test video shows that initial point-prompt multimask
selection is a major quality variable. At 1024, forcing C++ to use the first
initial multimask candidate matches official PyTorch closely:

| Encode size | Initial C++ mode | Mean mask IoU | Min mask IoU | Min bbox IoU | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| 1024 | `--multimask --initial-candidate-index 0` | 0.9831 | 0.9812 | 0.8661 | `outputs/quality-candidate0-current-1024/summary.json` |
| 512 | `--multimask --initial-candidate-index 0` | 0.2678 | 0.1797 | 0.2760 | `outputs/quality-candidate0-current-512/summary.json` |

This is not a universal fixed-candidate solution. It does show that the 1024
quality gap is largely caused by the initial mask candidate contract rather
than mask-hash-level numeric drift. The 512 row still needs separate
investigation; the earlier frame-index-fixed 512 default row remains better
(`mean_mask_iou=0.7761`) than candidate 0 on the current video.

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
hotspots into a single completion audit. The latest audit artifact is
`outputs/goal-audit/latest-summary.json`; it marks the profiling and
prioritization work as complete, but keeps the overall objective `not_complete`
because C++ is still slower than official PyTorch on matched 512/1024 rows and
Base+ mask quality is not at parity.

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

A separable two-pass CPU bilinear resize was checked as another preprocessing
candidate. It preserved full-mask output exactly (`mask_hash_equal_rows=10/10`,
`min_bbox_iou=1.0`), but did not reduce preprocessing: the measured
`hiera_encode_preprocess` mean moved from `7.34 ms` to `7.73 ms` on the
10-frame 1024 q4_0 run. The apparent total-track difference came from Hiera
compute variance, so this path was reverted. The remaining preprocessing target
is still a fused GPU resize/normalize/upload path rather than more CPU loop
restructuring.

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

The 2026-05-11 measurement environment had a separate failure mode: after login
`nvidia-powerd` was inactive. `nvidia-smi` still reported the GPU, but benchmark
runs hit software power-cap throttling and Hiera encode became unstable
(`67-77 ms` steady frames followed by `111-115 ms` frames). Restarting and
enabling `nvidia-powerd.service` restored stable clocks. This is not a code
optimization, but it is required for meaningful CUDA/PyTorch comparison on this
laptop GPU.

| State | q4_0 1024 C++ track ms/frame | Hiera steady profile | Note |
| --- | ---: | --- | --- |
| `nvidia-powerd` inactive | `110.9` | `67-77 ms`, then `111-115 ms` | power-cap throttling during benchmark |
| `nvidia-powerd` active | `66.5` | about `49 ms` | service manually started and enabled |

The matched official-PyTorch rows were rerun after fixing the service. The C++
path is much closer than the broken-environment rows, but still does not beat
official PyTorch:

| Precision | Encode size | C++ track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| q4_0 | 1024 | `66.5` | `42.73` | `0.643` | `outputs/model-matrix-powerd-restored-q4-1024-20260511/summary.json` |
| q8_0 | 1024 | `64.7` | `42.64` | `0.659` | `outputs/model-matrix-powerd-restored-1024-20260511/summary.json` |
| q8_0 | 512 | `17.3` | `11.24` | `0.650` | `outputs/model-matrix-powerd-restored-q8-512-20260511/summary.json` |

CUDA now also fuses the Hiera residual `ADD -> NORM -> MUL -> ADD` pattern. The
previous CUDA Norm affine fusion matched Metal's `NORM + MUL (+ ADD)` behavior,
but the residual add immediately before Norm still launched separately in
Hiera blocks. The new path computes the residual sum inside the Norm kernel and
then applies the same affine multiply/add, preserving the old rounding order
with explicit round-to-nearest operations. It is gated by
`GGML_CUDA_DISABLE_ADD_NORM_FUSION=1`.

Full-mask parity against the disabled path is bit-identical on the 10-frame
Base+ q4_0 1024 sample: `mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0`, and `max_score_abs_delta=0`. Seven paired bbox-only
runs show a small improvement:

| Path | Track ms/frame values | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| Add+Norm fusion disabled | `70, 68, 69, 67, 67, 67, 69` | `68.14` | `68.0` | `1.21` |
| Add+Norm fusion enabled | `67, 67, 69, 68, 67, 67, 67` | `67.43` | `67.0` | `0.79` |

The paired mean delta is `0.71 ms/frame` (`1.06%`). Node profiling confirms the
new fused path is actually used (`fused ADD skipped=3` appears 26 times in the
3-frame profile), but this remains a minor graph/kernel-launch cleanup rather
than a PyTorch-closing speedup.

A follow-up tightened the same idea for Hiera residual ADDs whose output is
needed twice: once by the following Norm and once by the later block residual.
The ordinary fusion matcher cannot remove these ADD nodes because their
use-count is 2. The CUDA Norm kernel now has a dual-output path that writes the
residual ADD tensor and the affine-Norm result from the same launch. It is gated
independently by `GGML_CUDA_DISABLE_ADD_NORM_PRESERVE_FUSION=1`; the broader
`GGML_CUDA_DISABLE_ADD_NORM_FUSION=1` still disables both residual-Norm paths.

The graph evidence is the 2026-05-16 fusion-window dump: Hiera stage-2 rows such
as `node_311` have `uses=2` and a direct
`ADD -> NORM -> RESHAPE -> MUL -> RESHAPE -> ADD` chain. After the dual-output
path, node profiling shows these rows as `fused=1 skipped=5 op=ADD`, while the
old standalone `ADD dst=f32[448,64,64,1]` rows mostly disappear.

Full-mask parity against the dedicated disabled path is bit-identical on the
10-frame Base+ q8_0 1024 sample: `mask_hash_equal_rows=10/10`,
`min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, `max_score_abs_delta=0.0`, and
`max_abs_mask_area_rel_delta=0.0`
(`outputs/add-norm-preserve-fusion-20260516/parity/summary.json`). Five paired
bbox-only runs show a small positive effect:

| Path | Track ms/frame values | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| Preserve Add+Norm disabled | `53, 52, 53, 54, 53` | `53.0` | `53.0` | `0.71` |
| Preserve Add+Norm enabled | `52, 52, 52, 52, 52` | `52.0` | `52.0` | `0.0` |

The paired mean delta is `1.0 ms/frame` (`1.89%`) on this q8_0 1024 run
(`outputs/add-norm-preserve-fusion-20260516/ab/summary.json`). This is still a
secondary fusion cleanup rather than the main PyTorch-closing path; MMQ and
head_dim 56 attention remain the larger hotspots.

The next fusion probe added an optional row-bias epilogue to the custom MMF
non-quantized `MUL_MAT` kernel and a guarded `MUL_MAT -> ADD(bias)` matcher
(`GGML_CUDA_DISABLE_MMF_BIAS_FUSION=1`). The implementation builds and is
bit-identical on the 10-frame Base+ q8_0 1024 parity check
(`outputs/mmf-bias-fusion-20260516/parity/summary.json`), but the current q8_0
Base+ hot path mostly uses batched cuBLAS or MMQ for the remaining large bias
adds, so this is not a measured speed win for that workload.

Broadening MMQ non-sequential bias fusion is still unsafe. Forcing all matching
MMQ `MUL_MAT -> ADD(bias)` rows with
`GGML_CUDA_ENABLE_MMQ_NONSEQ_BIAS_FUSION=1` changed tracking output on the same
10-frame sample: `mask_hash_equal_rows=0/10`, `min_bbox_iou=0.9752`, and
`max_bbox_delta_px=23`
(`outputs/mmq-bias-wide-20260516/parity/summary.json`). Enabling the exactness
guard with `GGML_CUDA_DISABLE_MMQ_BIAS_FIXUP_FUSION=1` did not recover parity
(`outputs/mmq-bias-wide-exact-20260516/parity/summary.json`). The next useful
fusion work should therefore stay on bit-identical residual/norm-style
dual-output kernels or move deeper into MMQ/FATTN arithmetic, not a broader MMQ
bias epilogue.

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
outputs/preprocess-separable/fullmask_compare.json
outputs/preprocess-separable/default/profile_summary.json
outputs/preprocess-separable/separable/profile_summary.json
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
outputs/model-matrix-latest-q4-1024/summary.json
outputs/model-matrix-latest-q4-512/summary.json
outputs/model-matrix-latest-q8_0-1024/summary.json
outputs/model-matrix-latest-q8_0-512/summary.json
outputs/model-matrix-latest-f16-1024/summary.json
outputs/model-matrix-latest-f16-512/summary.json
outputs/goal-audit/latest-summary.json
```

## 2026-05-11 CUDA MMQ Bias Fusion Follow-up

Environment repair after the login/session reset restored CUDA/Ninja builds. The
current kept ggml CUDA change is limited to Blackwell MMQ X caps for q4_0 and
q4_1 plus diagnostic MMQ profiling. Quantized-MMQ channel-bias fusion was
tested and then pruned from the branch because it traded quality for speed and
added hot-path branch cost even when disabled. A broader `MUL_MAT + bias + GELU`
writeback fusion was also tested and rejected because it regressed the matched
512 Base+ comparison.

Current checks:

| Check | Artifact | Result |
| --- | --- | --- |
| Build | `just build` | pass |
| Format | `just fmt-check` | pass |
| Local path / whitespace | `git diff --check && scripts/check_no_local_paths.sh` | pass |
| Environment repair smoke | `outputs/env-repair-parity-20260511/summary.json` | CUDA visible, hwaccel `cuda` visible, `diff_rows=0` |
| Bias fusion pruned parity smoke | `outputs/mmq-bias-pruned-parity-20260511/summary.json` | `diff_rows=0` |
| Bias fusion pruned q4_1 512 speed smoke | `outputs/q4_1-bias-pruned-512-20260511/run-*.log` | 5-run mean track `22.42 ms/frame`, p50 `22.10 ms`, p95 `24.58 ms` |
| q4_1 pruned matched Python speed, 512 | `outputs/model-matrix-q4_1-pruned-512-20260511/summary.json` | C++ `23.2 ms/frame`, PyTorch bf16 `14.86 ms/frame`; C++ still slower |
| q4_1 pruned matched Python speed, 1024 | `outputs/model-matrix-q4_1-pruned-1024-20260511/summary.json` | C++ `109.6 ms/frame`, PyTorch bf16 `64.95 ms/frame`; C++ still slower |
| q8_0 pruned matched Python speed, 512 | `outputs/model-matrix-q8_0-pruned-512-20260511/summary.json` | C++ `23.2 ms/frame`, PyTorch bf16 `14.85 ms/frame`; C++ still slower |
| q8_0 pruned matched Python speed, 1024 | `outputs/model-matrix-q8_0-pruned-1024-20260511/summary.json` | C++ `109.0 ms/frame`, PyTorch bf16 `65.45 ms/frame`; C++ still slower |
| q8_0 current-input 1024 quality | `outputs/quality-1024-q8_0-frames-current-20260511/official/summary.json` | mean mask IoU `0.988`, min mask IoU `0.986`, selected initial candidate matches official |
| q4_1 pruned goal audit | `outputs/goal-audit/q4_1-pruned-20260511-summary.json` | profiling/prioritization evidence is present, but speed and Base+ quality parity remain incomplete |
| q8_0 1024 stage profile | `outputs/profile-stage-q8_0-1024-pruned-20260511/summary.json` | Hiera encode dominates; `propagate_single` is about `18.2 ms/frame` |
| q8_0 1024 CUDA node profile | `outputs/profile-nodes-q8_0-1024-pruned-20260511/hotspots-all.json` | D56 window/global FlashAttention and q8_0 MLP MMQ dominate the synchronized node profile |
| q8_0 1024 MMQ profile | `outputs/profile-mmq-q8_0-1024-pruned-20260511/grouped-summary.json` | MMQ body dominates quantization; total drop-max MMQ `209.7 ms`, quantization `44.2 ms` in profiling mode |
| q8_0 1024 FATTN56 profile | `outputs/profile-fattn56-q8_0-1024-pruned-20260511/summary.json` | global D56 attention is mostly MMA body; window D56 attention has meaningful pack/slice overhead |
| Rejected q8_0 global cuBLAS fallback | `outputs/cublas-vs-mmq-q8_0-20260511/summary.json` | force-cuBLAS was about `1.95x` slower at 512 and `2.04x` slower at 1024 than force-MMQ |
| Rejected q8_0 D56 native tile fallback | `outputs/fattn56-pad-vs-tile-q8_0-20260511/summary.json` | disabling pad-MMA made D56 attention about `1.12x` slower at 512 and `1.16x` slower at 1024 |
| Rejected D56 MMA ncols override | `outputs/fattn56-ncols-q8_0-20260511/summary.json` | default/64-column selection remained best or statistically tied; smaller ncols regressed |
| Post-rejection parity smoke | `outputs/post-cublas-fattn56-rejection-parity-20260511/summary.json` | `diff_rows=0` after leaving only the accepted MMQ/X-cap changes |
| Parity smoke | `outputs/mmq-bias-only-confirm-20260511/parity-smoke/summary.json` | `diff_rows=0` |
| q4_1 X cap parity smoke | `outputs/q4_1-xmax96-parity-20260511/summary.json` | `diff_rows=0` |
| Matched official Python speed | `outputs/model-matrix-q4-512-mmq-bias-only-confirm-20260511/summary.json` | C++ q4_0 512 `21.9 ms`, PyTorch bf16 512 `14.35 ms` |
| q4_1 X cap repeat | `outputs/q4_1-xmax-repeat-20260511/summary.json` | X=96 mean `21.98 ms`, default mean `22.60 ms`, X=128 mean `22.56 ms` |
| q4_1 X cap matched Python speed | `outputs/model-matrix-q4_1-512-xmax96-20260511/summary.json` | C++ q4_1 512 `22.3 ms`, PyTorch bf16 512 `14.64 ms` |
| q4_1 X cap node profile | `outputs/profile-q4_1-xmax96-20260511/hotspots-all.json` | top steady rows remain Hiera `FLASH_ATTN_EXT f32[56,8,196,9]` and quantized MMQ MLP |
| Rejected GELU writeback fusion | `outputs/model-matrix-q4-512-mmq-bias-gelu-fusion-20260511-with-python/summary.json` | C++ regressed to `22.7 ms` |
| MMQ X cap sweep | `outputs/mmq-xmax-sweep-20260511/summary.json` | X=104 mean `21.66 ms`, X=128 mean `22.32 ms` |
| Low-resolution speed probe | `outputs/model-matrix-q4-192-bias-only-20260511/summary.json` | C++ 192 `7.2 ms`, PyTorch 192 `8.61 ms` |
| Low-resolution quality check | `outputs/quality-192-q4-bias-only-20260511/official/summary.json` | mean mask IoU `0.20`; rejected |
| 256 quality check | `outputs/quality-256-q4-bias-only-20260511/official/summary.json` | mean mask IoU `0.55`; rejected |
| 384 quality check | `outputs/quality-384-q4-bias-only-20260511/official/summary.json` | mean mask IoU `0.58`; rejected |
| All precision 512 speed refresh | `outputs/model-matrix-allprec-512-bias-only-final-20260511/summary.json` | best C++ row q4_1 `21.7 ms`, PyTorch `14.83 ms` |
| Rejected ADD->NORM fusion | `outputs/add-norm-fusion-20260511/bench_512_summary.json` | 5-run mean `21.9 ms`; no improvement |
| q8_0 512 quality, frames source | `outputs/quality-512-q8_0-frames-current-20260511/official/summary.json` | mean mask IoU `0.94`, min `0.91` |
| q4_1 512 quality, frames source | `outputs/quality-512-q4_1-frames-current-20260511/official/summary.json` | mean mask IoU `0.86`, min `0.78`; bbox IoU remains high |
| q4_0 512 quality, frames source | `outputs/quality-512-q4_0-frames-current-20260511/official/summary.json` | mean mask IoU `0.04`; rejected |
| Stage profile | `outputs/profile-stage-q4_1-512-20260511/summary.json` | steady `hiera_encode` remains roughly `14-16 ms/frame` |
| FATTN56 profile | `outputs/profile-fattn56-q4_1-512-20260511/summary.json` | all steady FATTN56 work is about `2.4 ms/frame`; not enough alone to close the PyTorch gap |
| FATTN56 padded-MMA A/B | `outputs/fattn56-pad-ab-q4_1-512-20260511/summary.json` | current padded-MMA mean `21.98 ms`, old tile mean `24.44 ms`; keep current path |
| MMQ grouped profile | `outputs/profile-mmq-q4_1-512-20260511/grouped-summary.json` | q4_1 512 MMQ body dominates quantization; total MMQ event sum `78.91 ms`, quant sum `13.45 ms` |
| Rejected MMQ stream-k disable | `outputs/mmq-streamk-ab-q4_1-512-20260511/summary.json` | stream-k on mean `22.22 ms`, off mean `241.06 ms`; diagnostic branch reverted |
| Rejected q4_1 X=104 retest | `outputs/q4_1-xmax104-512-current-20260511/summary.json` | explicit 512 mean `22.48 ms`; worse than the kept X=96 mean `21.98 ms` |
| Rejected q8_0 X=104 | `outputs/q8_0-xmax104-default-512-current-20260511/summary.json`, `outputs/quality-512-q8_0-xmax104-20260511/official/summary.json` | speed mean `22.00 ms`, but mean mask IoU dropped to `0.853`; q8_0 quality fallback keeps default X selection |
| MMQ bias fusion speed A/B | `outputs/mmq-bias-fusion-ab-q4_1-512-20260511/summary.json` | enabled mean `22.02 ms`, disabled mean `23.02 ms`; about `4.3%` faster |
| Rejected default MMQ bias fusion | `outputs/quality-512-q4_1-bias-fusion-on-multimask-20260511/official/summary.json`, `outputs/quality-512-q4_1-bias-fusion-off-multimask-20260511/official/summary.json` | multimask mean mask IoU worsened from `0.853` disabled to `0.834` enabled |
| Rejected stream-k bias fixup attempt | `outputs/mmq-bias-fusion-fixed-parity-q4_1-512-multimask-20260511/summary.json`, `outputs/quality-512-q4_1-bias-fusion-fixed-on-multimask-20260511/official/summary.json` | adding bias in stream-k fixup still changed all rows; mean mask IoU `0.811` enabled vs `0.856` disabled |

This confirms the latest kept change is correctness-safe but still not enough to
beat official PyTorch. The q4_1 Blackwell X cap is a small scheduling win rather
than a root-cause fix: it improves the five-run mean by about 2.7%, but the
matched official Python comparison remains much faster. The next high-impact
target remains the Hiera
FlashAttention/MMQ body rather than additional scalar post-op fusion: the
writeback GELU experiment removed a separate kernel but increased per-element
MMQ writeback cost enough to lose overall.

The simpler MMQ channel-bias fusion is also not acceptable. It was
measurably faster on q4_1 512 (`22.02 ms/frame` vs `23.02 ms/frame` in the
five-run A/B), but it changes full-mask output. Under the correct multimask
quality contract, official-Python mean mask IoU drops from `0.853` with the
fusion disabled to `0.834` with it enabled. The branch therefore removes this
fusion rather than keeping an opt-in switch; quality and Rust-wrapper baselines
should use the pruned default path.

The follow-up attempt to make stream-k bias placement more correct by adding
bias in the fixup kernel did not rescue the fusion. Full-mask rows still changed
and the official-Python mean mask IoU was worse (`0.811` enabled vs `0.856`
disabled). This confirms the remaining root work should target MMQ arithmetic
or FlashAttention kernel body time, not channel-bias fusion.

The q4_1/xmax96 node profile shows the next priority is still split between
the Hiera `head_dim=56` FlashAttention body and quantized MMQ MLP body. The
largest steady rows after dropping first-use outliers are:

| Op / shape | Count | drop-max sum ms | Mean ms |
| --- | ---: | ---: | ---: |
| `FLASH_ATTN_EXT dst=f32[56,8,196,9]` | 120 | 9.67 | 0.081 |
| `MUL_MAT dst=f32[1344,196,9,1]` | 120 | 9.65 | 0.081 |
| `MUL_MAT dst=f32[1792,1024,1,1]` | 160 | 9.04 | 0.057 |
| `MUL_MAT dst=f32[448,1024,1,1]` | 190 | 8.23 | 0.044 |

This is why local elementwise fusion and small tile caps should be treated as
secondary polish; reaching official PyTorch speed requires a larger FlashAttention
or MMQ body improvement.

The 2026-05-11 environment repair did not require changing CUDA, ffmpeg, uv, or
apt state. The bad candidate was the diagnostic stream-k disable branch: with
stream-k enabled q4_1 512 averaged `22.22 ms/frame`, while disabling it averaged
`241.06 ms/frame`. That branch is therefore not a viable fallback or debug knob
for performance runs, and the kept code keeps the existing ggml stream-k
selection.

Two follow-up MMQ X-cap retests were also rejected. Retesting q4_1 with X=104
under an explicit 512 input averaged `22.48 ms/frame`, slower than the kept X=96
baseline. Applying X=104 to q8_0 gave a small speed win, but it changed the
tracked masks enough to drop official-Python mean mask IoU from the previous
q8_0 baseline around `0.936` to `0.853`. For a Rust-wrapper quality fallback,
q8_0 should therefore keep the default X selection unless a later change proves
both speed and mask quality.

An upstream ggml history refresh did not reveal an immediately applicable
unmerged CUDA fix for the current Hiera shapes. The directly relevant
`a1fde6fb` MMQ stream-k overhead reduction is already in this branch. The newer
`8498d0f6` FlashAttention change targets DKQ=192/DV=128 MiMo shapes, not
Hiera's DKQ/DV=56 path, so it is not a direct speed candidate for this workload.

Lowering the matched SAM input resolution can make C++ faster than official
PyTorch, but the checked 192/256/384 rows are not acceptable quality baselines.
They should remain scaling probes only; speed claims for the Rust-wrapper
baseline should continue to use 512 or 1024 unless a new quality check shows a
lower resolution is viable.

The 512 all-precision refresh shows q4_1 as the fastest checked C++ precision
in that run (`21.7 ms`), but it is still only `0.68x` of official PyTorch speed.
For quality, q8_0 is the better 512 fallback candidate on this clip
(mean/min mask IoU `0.94/0.91`) while q4_1 keeps strong bbox overlap but weaker
mask overlap. q4_0 at 512 is not a valid quality baseline. The rejected
`ADD -> NORM` fusion confirms that shaving residual/norm elementwise launches is
not sufficient; the dominant work remains MMQ/FlashAttention body time.

After pruning the rejected MMQ bias fusion, the matched official-Python rows
were refreshed for q4_1 and q8_0. The conclusion did not change: at 512, both
q4_1 and q8_0 measure `23.2 ms/frame` against official PyTorch at about
`14.85 ms/frame`; at 1024, q4_1 measures `109.6 ms/frame` and q8_0 measures
`109.0 ms/frame` against official PyTorch at about `65 ms/frame`. The best
quality fallback is still q8_0 rather than q4_1. On the current 960x540 decoded
input with `image_size=1024`, q8_0 reaches mean mask IoU `0.988` and min mask
IoU `0.986`, with the initial multimask selected candidate matching official
SAM2. q8_0 should therefore be the quality reference when validating future
Rust-wrapper behavior, while q4_1 remains a speed/size experiment.

The q8_0 1024 profile gives a clearer root-cause split for the quality fallback.
`propagate_single` is about `18.2 ms/frame`, so the remaining PyTorch gap is
still primarily Hiera encode. In synchronized node profiling, the largest
drop-max groups are D56 window FlashAttention (`43.46 ms`), D56 global
FlashAttention (`42.13 ms`), q8_0 MLP projection/expansion matmuls
(`37.19 ms` and `36.20 ms`), and q8_0 window MLP expansion (`30.24 ms`). The
MMQ event profile confirms that this is not mainly q8_1 activation quantization:
drop-max MMQ body time is `209.7 ms` versus `44.2 ms` quantization in profiling
mode. The FATTN56 event profile similarly shows that global D56 attention is
mostly the D64 MMA body (`44.96 ms` drop-max MMA for `Q=[56,4096,8,1]`), while
the window path (`Q=[56,196,8,25]`) still spends `11.11 ms` in pack and
`3.60 ms` in slice in addition to `33.11 ms` in MMA. The next implementation
branch should therefore prioritize a real D56/no-mask FlashAttention kernel
that writes 56 lanes directly, then MMQ body improvements for the repeated
Hiera MLP shapes.

The latest dispatch checks narrow that implementation choice further. A global
`GGML_CUDA_FORCE_CUBLAS` build is not a useful escape hatch for q8_0 Base+:
the same smoke video averaged `45.93 ms/frame` at 512 and `222.77 ms/frame` at
1024, versus force-MMQ at `23.53 ms/frame` and `109.27 ms/frame`. Disabling the
D56 pad-MMA path and falling back to native tile attention also regressed
(`25.50 ms/frame` at 512 and `126.97 ms/frame` at 1024 versus pad-MMA
`22.87 ms/frame` and `109.10 ms/frame`). Sweeping the D56 wrapped-MMA ncols
choice showed no hidden win: 512 favored the default path (`22.70 ms/frame`),
and 1024's explicit 64-column run was only a statistical tie with default.
That leaves the high-confidence next CUDA change as removing the pad/slice
overhead with a real D56 no-mask writeback path, not switching Hiera matmuls to
cuBLAS or falling back to the old tile kernel.

### Blackwell FP4 probe

The Blackwell-native FP4 path was checked as a more radical speed candidate
before committing to a custom Hiera kernel. The local quantizer now accepts
`mxfp4` and `nvfp4`, and `sam3_load_model` recognizes both SAM-local type ids
and ggml ftype ids for these formats. The generated Base+ probes are ignored
artifacts under `models/matrix-all/`:

| Probe | Artifact | Result |
| --- | --- | --- |
| MXFP4 quantization | `outputs/quantize-base-plus-mxfp4-20260511.log` | `189 / 615` tensors quantized, `43.39 MB` output |
| NVFP4 quantization | `outputs/quantize-base-plus-nvfp4-20260511.log` | `179 / 615` tensors quantized, `47.75 MB` output |
| MXFP4/NVFP4 load smoke | `outputs/smoke-base-plus-mxfp4-20260511/`, `outputs/smoke-base-plus-nvfp4-20260511/` | CUDA load and segmentation pass |
| Matched 512 speed | `outputs/model-matrix-base-plus-fp4-512-20260511/summary.json` | MXFP4 `22.4 ms/frame`, NVFP4 `25.6 ms/frame`, PyTorch bf16 `14.56 ms/frame` |
| Matched 1024 speed | `outputs/model-matrix-base-plus-fp4-1024-20260511/summary.json` | MXFP4 `109.4 ms/frame`, NVFP4 `118.5 ms/frame`, PyTorch bf16 `66.28 ms/frame` |
| MXFP4 512 quality | `outputs/quality-512-mxfp4-base-plus-20260511/official/summary.json` | mean/min mask IoU `0.774 / 0.664` |
| NVFP4 512 quality | `outputs/quality-512-nvfp4-base-plus-20260511/official/summary.json` | mean/min mask IoU `0.709 / 0.661` |
| MXFP4 1024 quality | `outputs/quality-1024-mxfp4-base-plus-20260511/official/summary.json` | mean/min mask IoU `0.923 / 0.916` |
| Post direct-MMA revert parity smoke | `outputs/fp4-probe-post-direct-revert-parity-20260511/summary.json` | `diff_rows=0` |

This rejects FP4 as the current Base+ fallback path. MXFP4 at 512 is only
slightly faster than q8_0/q4_1 in the same run and remains much slower than
official PyTorch. At 1024, MXFP4 ties q8_0 speed rather than improving it, while
quality remains below the q8_0 1024 fallback. NVFP4 is slower and lower quality
in the checked 512 row. The useful outcome of this probe is negative evidence:
native FP4 support alone does not close the PyTorch gap for this workload, so
the next root fix should stay focused on D56 FlashAttention writeback and MMQ
body efficiency rather than adding more low-precision fallback formats.

A first direct-D56 MMA experiment was also rejected. Adding `DKQ=56, DV=56`
MMA template cases and routing D56 no-mask attention directly through
`ggml_cuda_flash_attn_ext_mma_f16` built after regenerating the CUDA template
instances, but the opt-in A/B changed the q8_0 512 detection result. The direct
path measured about `22.2 ms/frame` over three runs versus the pad-MMA path at
about `23.4 ms/frame`, but all direct runs reported `0` detections where the
pad-MMA baseline reported `1`. The experiment artifacts are under
`outputs/fattn56-direct-mma-ab-q8_0-512-20260511/`; the code change was removed.
This means a viable D56 implementation cannot be a bare configuration enablement
of the existing MMA kernel. It needs a correctness-aware D56 writeback/kernel
variant with parity checked before speed is trusted.

### Blackwell cuBLAS F16 compute path

The f16 Base+ fallback was checked after environment repair because official
PyTorch uses bf16/half Tensor Core paths, while the C++ fallback still keeps
F32 activations at the graph boundary. On Blackwell, the default ggml f16 cuBLAS
path used `CUBLAS_COMPUTE_16F`, wrote an intermediate F16 output, and then
converted that output back to F32. For this workload, forcing
`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` was faster because cuBLAS writes the F32
destination directly.

The accepted ggml change now treats Blackwell like the existing Volta/CDNA/RDNA4
FP32-accumulate branch for f16 cuBLAS matmuls. Same-build A/B on
`sam2.1_hiera_base_plus_f16`, `image_size=1024`, 10-frame bbox-only tracking:

| Mode | Artifact | Track mean |
| --- | --- | ---: |
| Blackwell FP32 cuBLAS compute default | `outputs/blackwell-cublas32-forcefp16-f16-1024-20260511/summary.json` | `126.1 ms/frame` |
| Forced old FP16 compute path | `outputs/blackwell-cublas32-forcefp16-f16-1024-20260511/summary.json` | `129.3 ms/frame` |

This is a small f16-only improvement, not the root PyTorch gap fix. The best
quality/speed fallback for 1024 remains q8_0, and q8_0 is still about
`109 ms/frame` versus official PyTorch around `65 ms/frame`. The 512 f16
quality reruns also show that f16 is not a better fallback than q8_0 on the
current clip: the new f16 runs are around `0.82` mean mask IoU at 512, while the
checked q8_0 512 fallback remains around `0.94` mean mask IoU. Future work
should continue to prioritize D56 FlashAttention and MMQ body changes; f16
cuBLAS compute selection is only a cleanup for the f16 fallback path.

### SAM3 CUDA Graph default

CUDA Graph capture was rechecked after the environment repair because Hiera
encode rebuilds many short graphs and the q8_0 quality fallback still needs
incremental wins while the larger D56/MMQ work is pending. Disabling graphs
globally is not acceptable: q4_1 full-mask output changed versus the graph path
in `outputs/sam3-nograph-default-parity-q4_1-512-20260511/summary.json`, with
only `0 / 10` mask hashes equal and later-frame bbox deltas up to `97 px`.

The safe change is therefore limited to q8_0 model files, where the checked
quality fallback is bit-identical between graph and no-graph execution. The
ggml CUDA graph env check is now dynamic, and SAM3 scopes
`GGML_CUDA_DISABLE_GRAPHS=1` only around graph executions that contain q8_0
tensors. Users can opt back into CUDA Graphs with `SAM3_CUDA_ENABLE_GRAPHS=1`.

| Check | Artifact | Result |
| --- | --- | --- |
| q8_0 1024 scoped no-graph vs graph speed | `outputs/sam3-q8-graph-scope-ab-1024-20260511/summary.json` | q8_0 scoped default `109.82 ms/frame`, graph opt-in `111.04 ms/frame`; default is `1.11%` faster |
| q8_0 1024 scoped no-graph vs graph parity | `outputs/sam3-q8-graph-scope-parity-1024-20260511/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| q4_1 graph default guard | `outputs/sam3-q4-graph-scope-guard-512-20260511/summary.json` | q4_1 remains on the graph path by default; graph default and explicit graph opt-in are identical |
| Matched official Python speed refresh | `outputs/model-matrix-q8_0-graph-scope-1024-with-python-20260511/summary.json` | C++ q8_0 `111.4 ms/frame`, PyTorch bf16 `65.82 ms/frame`; C++ still slower |

This is not the main PyTorch gap fix. It only improves the q8_0 Base+ fallback
by about one to two percent in the current 1024 run. The matched official
Python row is still around `65 ms/frame`, so the remaining priority is unchanged:
D56 FlashAttention body/writeback and repeated Hiera MMQ body time.

The 2026-05-11 q8-only graph policy above is now superseded. On 2026-05-16,
`compute-sanitizer --tool initcheck` found CUDA graph replay reading captured
host-memory upload pointers during Hiera encode (`7340032` bytes, matching the
512 Hiera positional embedding payload). SAM3 now scopes
`GGML_CUDA_DISABLE_GRAPHS=1` for all CUDA graph executions unless
`SAM3_CUDA_ENABLE_GRAPHS=1` is explicitly set. The post-fix initcheck run has
zero errors in
`outputs/initcheck-graphs-default-disabled-f16-512-2frame-20260515/`.

The same investigation found a separate stream-ordering bug: internal
CUDA-to-CUDA `ggml_backend_tensor_copy` calls used the buffer copy path on
`cudaStreamPerThread`, while graph compute used the backend stream. Runtime
state copies now use `ggml_backend_tensor_copy_async` on the backend stream.
This makes repeated CUDA output deterministic on the same input: both f16 and
q8_0 Base+/512 match `5/5` mask hashes across repeated runs in
`outputs/determinism-after-backend-stream-copy-f16-512-run*-20260516/` and
`outputs/determinism-after-backend-stream-copy-q8_0-512-run*-20260516/`.

After the stream-order fix, the q8_0/Base+/512 official-Python quality check
with the same extracted frame directory is stable enough to use as a fallback
baseline: `mean_mask_iou=0.96486`, `min_mask_iou=0.95826`, initial multimask
selection matches, and propagation selection matches `9/9`. Evidence is in
`outputs/official-quality-mask-q8_0-512-after-stream-copy-fix-20260516/`.

The earlier rejected q8_0 MMQ bias fixup fusion was retested after the stream
ordering fix. It is now deterministic and preserves the official-Python fallback
quality, so the non-exact stream-k bias fusion path is enabled by default and
can be disabled with `GGML_CUDA_DISABLE_MMQ_BIAS_FIXUP_FUSION=1`. The default
q8_0/Base+/512 run now measures `15.0 ms/frame` over five runs and has
`mean_mask_iou=0.96518`, `min_mask_iou=0.95854`, initial selection matched, and
propagation selection `9/9` matched. At 1024, the same fusion keeps high mask
quality (`mean_mask_iou=0.99461`, `min_mask_iou=0.99109`) and measures about
`53.3 ms/frame` over three runs. Evidence is in
`outputs/determinism-default-fixup-on-q8_0-512-run*-20260516/`,
`outputs/speed-default-fixup-on-q8_0-512-20260516/`,
`outputs/official-quality-default-fixup-on-q8_0-512-20260516/`,
`outputs/official-quality-mmq-bias-fixup-q8_0-1024-20260516/`, and
`outputs/speed-mmq-bias-fixup-q8_0-1024-20260516/`.

The q8_0 Blackwell `MMQ_X_MAX=96` cap was also retested after the stream fix.
It remains rejected as a default change: the 512 quality row was fine
(`mean_mask_iou=0.96549`, `min_mask_iou=0.95870`), but the five-run default
speed after applying the cap was still `15.0 ms/frame` at 512 and
`54.4 ms/frame` at 1024, so the extra schedule restriction did not provide a
clear win. Evidence is in `outputs/official-quality-q8_0-xmax96-512-20260516/`
and `outputs/speed-default-q8_0-*-xcap96-fixup-20260516/`.

Several older rejected knobs were also retested after the stream-order fix:

| Candidate | Result | Evidence |
| --- | --- | --- |
| Disable MMQ bias+GELU fusion | No improvement; default mean `15.2 ms/frame`, disabled mean `15.0 ms/frame` within noise | `outputs/mmq-bias-gelu-ab-q8_0-512-*-20260516/` |
| D56 native-V MMA | Still slower than padded D64/DV_DST=56; `56.3` vs `54.0 ms/frame` at 1024 | `outputs/fattn56-native-v-retest-q8_0-1024-*-20260516/` |
| D56 old tile fallback | Still much slower; `67.3` vs `54.3 ms/frame` at 1024 | `outputs/fattn56-pad-vs-tile-retest-q8_0-1024-*-20260516/` |
| f16 cuBLAS compute type forcing | No useful 512 speed win; default and forced compute16 both `17.0 ms/frame` | `outputs/f16-cublas-compute-retest-512-*-20260516/` |

The post-fix precision refresh also confirms that f16 is not the speed fallback:
at 512, f16 measured `16.0 ms/frame`, while q4_1 and q8_0 both measured
`15.0 ms/frame`. Evidence is in
`outputs/precision-refresh-*-512-after-stream-fix-20260516/`.

The speed objective is still incomplete. Five q8_0/Base+/512 C++ runs after the
fixup default average `15.0 ms/frame`, while the synchronized official PyTorch
bf16 profile is `11.37 ms/frame`. Steady C++ Hiera encode is about
`10.8-11.3 ms` against PyTorch `forward_image` at `7.35 ms`; propagation is
similar (`3.3-3.6 ms` C++ vs PyTorch `track_step` `3.69 ms`). The remaining
speed priority is therefore still Hiera encode, especially D56 FlashAttention
and repeated Hiera MMQ body time. Evidence is in
`outputs/speed-q8_0-512-after-stream-copy-fix-20260516/`,
`outputs/pytorch-profile-base-plus-512-after-stream-copy-fix-20260516/`, and
`outputs/profile-stage-q8_0-512-after-stream-copy-fix-20260516/`.

### Rejected D56 pack/slice vectorization

A narrower attempt to speed up the existing D56 pad-MMA path vectorized only the
contiguous head56-to-head64 pack and head64-to-head56 slice kernels. This was
correctness-local and left the MMA body untouched, but it did not move the
benchmark. Same-build q8_0 Base+ 1024 A/B over five runs:

| Mode | Artifact | Track mean |
| --- | --- | ---: |
| Scalar pack/slice | `outputs/fattn56-vec4-pack-ab-q8_0-1024-20260511/summary.json` | `109.86 ms/frame` |
| Vec4 pack/slice | `outputs/fattn56-vec4-pack-ab-q8_0-1024-20260511/summary.json` | `109.82 ms/frame` |

The event profile under `outputs/fattn56-vec4-pack-profile-q8_0-1024-20260511/`
showed some pack/slice reductions, but the global D56 attention total worsened
and the end-to-end result was noise. The experiment was removed. This reinforces
that the next useful D56 work must change the attention body/writeback itself,
not just the separate padding and slicing kernels.

### Rejected D56 memcpy2D slice

The D56 pad-MMA wrapper was also tested with `cudaMemcpy2DAsync` for the final
head64-to-head56 contiguous slice. This keeps the same values and only changes
the copy mechanism, but it was not faster than the existing lightweight slice
kernel. Same-build q8_0 Base+ 1024 A/B over six runs:

| Mode | Artifact | Drop-first mean |
| --- | --- | ---: |
| Existing slice kernel | `outputs/fattn56-memcpy2d-slice-ab-q8_0-1024-20260511/summary.json` | `108.20 ms/frame` |
| `cudaMemcpy2DAsync` slice | `outputs/fattn56-memcpy2d-slice-ab-q8_0-1024-20260511/summary.json` | `108.96 ms/frame` |

Parity was exact in
`outputs/fattn56-memcpy2d-slice-parity-q8_0-1024-20260511/summary.json`, but
the speed result regressed by about `0.7%`, so the experiment was removed. The
remaining D56 path should not spend more effort swapping copy primitives; it
needs a real D56 writeback/body variant.

### Accepted D56 direct output writeback

The next D56 change keeps the existing `DKQ=64, DV=64` MMA compute path, but
lets the head56 wrapper write only the first 56 output channels directly into
the final contiguous destination. This removes the 64-wide output temporary and
the separate head64-to-head56 slice kernel for the common contiguous output
case. Non-contiguous output still falls back to the old slice path, and the old
path can be forced with `GGML_CUDA_DISABLE_FATTN56_DIRECT_OUT=1`.

Correctness and speed were checked on `sam2.1_hiera_base_plus_q8_0`,
`image_size=1024`, 10-frame tracking:

| Check | Artifact | Result |
| --- | --- | --- |
| JSONL parity | `outputs/fattn56-direct-out-ab7-q8_0-1024-20260511/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| End-to-end A/B | `outputs/fattn56-direct-out-ab7-q8_0-1024-20260511/summary.json` | baseline `111.71 ms/frame`, direct output `110.57 ms/frame`; paired saved mean `1.14 ms/frame` |
| D56 event profile | `outputs/fattn56-direct-out-profile-q8_0-1024-20260511/summary.json` | D56 profiled total `156.31 ms -> 140.79 ms` across the run; slice time `8.90 ms -> 0.67 ms` |

This is an incremental structural cleanup, not a full PyTorch gap fix. It saves
about one percent end-to-end on the current Base+ q8_0 1024 clip because the
larger costs are still the 64-wide padded MMA body and repeated q8 MMQ work.
The next root fix remains a true D56 attention body/writeback or a q8 MMQ body
specialization for the dominant Hiera MLP shapes.

### Rejected q8_0 MMQ stream-k disable

After direct output writeback, MMQ was reprofiled on the same Base+ q8_0 1024
case. The 3-frame profile in
`outputs/profile-mmq-q8_0-1024-direct-out-20260511/grouped-summary.json`
shows `63.25 ms` of profiled MMQ time, split into `11.02 ms` activation
quantization and `52.23 ms` MMQ kernel body. The largest groups are still the
stage-2 MLP projection/expansion shapes around `[448,4096]` and `[1792,4096]`.

Disabling stream-k for q8_0 was tested as a sanity check and rejected. The A/B
artifact is `outputs/mmq-streamk-ab-q8_0-1024-direct-out-20260511/summary.json`.
Stream-k on measured `112.67 ms/frame`; stream-k off measured
`2835.33 ms/frame` and also changed tracking output (`0 / 10` mask hashes
equal, max bbox delta `2 px`). The measurement-only override was removed. This
confirms that MMQ work should focus on a better stream-k body or shape-specific
tiling, not disabling stream-k.

### Disabled D56 V-width MMA path

The D56 wrapper was then pushed one step beyond direct output writeback:
Q/K still use the existing padded `DKQ=64` path, but V and the output now use
`DV=56` for contiguous D56 attention. This avoids computing and storing the
unused value/output lanes while preserving the stable 64-wide K/Q MMA path.
The initial q8_0 sample looked exact and slightly faster:

| Check | Artifact | Result |
| --- | --- | --- |
| 10-frame A/B parity | `outputs/fattn56-v56-mma-ab7-q8_0-1024-20260511/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| 10-frame A/B speed | `outputs/fattn56-v56-mma-ab7-q8_0-1024-20260511/summary.json` | baseline `111.57 ms/frame`, V56 `110.86 ms/frame` |
| 30-frame A/B parity | `outputs/fattn56-v56-mma-ab3-30f-q8_0-1024-20260511/summary.json` | `30 / 30` mask hashes equal, max bbox delta `0 px` |
| 30-frame A/B speed | `outputs/fattn56-v56-mma-ab3-30f-q8_0-1024-20260511/summary.json` | baseline `111.67 ms/frame`, V56 `111.00 ms/frame` |
| D56 event profile | `outputs/fattn56-v56-mma-profile-q8_0-1024-20260511/summary.json` | profiled D56 total `151.78 ms -> 145.88 ms`; D56 MMA time `105.65 ms -> 102.58 ms` |

The end-to-end improvement is still under one percent, but unlike copy-only
changes it appeared to reduce real D56 MMA work. A later q4_0 Base+ audit found
that this path is not generally correct. `DV=56` violates the generic MMA
kernel's 16-lane value tile assumption, so the path is now disabled by default
and only reachable through `GGML_CUDA_ENABLE_UNSAFE_FATTN56_V56_MMA=1` for
diagnostics. The safe default pads V to 64 and uses `DV_DST=56` only for output
truncation.

The disabled default was rechecked after this change:

| Check | Artifact | Result |
| --- | --- | --- |
| Default parity after disabling V56 | `outputs/v56-disabled-default-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| Default D56 profile after disabling V56 | `outputs/v56-disabled-default-profile-q4_0-1024-20260515/summary.txt` | main Hiera rows remain `qkv_contiguous=0`; no new V56 use |
| Default 5-run speed after disabling V56 | `outputs/v56-disabled-default-speed-q4_0-1024-20260515/summary.json` | mean `67.26 ms/frame`, runs `[66.9, 66.9, 67.1, 67.6, 67.8]` |

### Rejected ADD to NORM-only fusion

After the accepted residual `ADD -> NORM -> MUL -> ADD` fusion, a narrower
`ADD -> NORM` CUDA fusion was probed to see whether the remaining layer-norm
launches could be reduced without touching the affine path. The fused kernel
kept the same rounding order for the residual add and matched the reference
tracking JSONL exactly, but the isolated speed result was too small to carry the
extra code.

| Check | Artifact | Result |
| --- | --- | --- |
| 10-frame parity | `outputs/add-norm-only-parity-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| 7-run A/B | `outputs/add-norm-only-ab7-20260515/profile-summary.json` | disabled mean `67.17 ms/frame`, enabled mean `67.04 ms/frame` |

The apparent `0.13 ms/frame` gain is within normal run-to-run noise on this
clip, so the `ADD -> NORM`-only entry point and dispatch path were removed. The
4-node residual affine fusion remains enabled because its earlier A/B was larger
and it removes the full residual/normalization/affine chain in one kernel.

### Current matched speed refresh

The latest matched q4_0 rows were refreshed after the power-management repair,
D56 attention work, and residual affine fusion cleanup. The comparison contract
is the same decoded 10-frame clip, same point prompt, same measured tracking
range, and same SAM input resolution.

| Encode size | C++ q4_0 track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| 1024 | `67.0` | `42.52` | `0.635` | `outputs/model-matrix-current-q4-1024-20260515/summary.json` |
| 512 | `18.2` | `11.13` | `0.612` | `outputs/model-matrix-current-q4-512-20260515/summary.json` |

This keeps the current bottleneck assessment unchanged: the C++ CUDA path is
substantially improved from the broken power state, but still needs roughly a
`1.6x` speedup at both 1024 and 512 to beat the official PyTorch CUDA bf16 path.

### Rejected D56 MMA config retuning

Two small D56 `ncols=64` MMA configuration retunes were checked after the V56
path, because the current Base+ D56 window and global attention shapes both use
that config. Neither produced a durable speed win.

| Experiment | Artifact | Result |
| --- | --- | --- |
| `nthreads=256, occupancy=1` | `outputs/fattn56-nthreads256-profile-q4-1024-20260515/run.log` | global MMA was not faster and sometimes regressed (`~0.86-0.91 ms`) |
| `nbatch_fa=128` | `outputs/fattn56-nbatch128-parity-q4-1024-20260515/summary.json`, `outputs/fattn56-nbatch128-ab7-q4-1024-20260515/profile-summary.json` | parity exact, but 7-run mean was `67.37 ms/frame` vs the current nearby baseline around `67.17 ms/frame` |

The config was restored to `nthreads=128, occupancy=2, nbatch_fa=64`. The next
D56 improvement needs a real body change rather than another coarse config
retune.

### Rechecked q4_0 MMQ X cap

The current q4_0 MMQ profile on Base+ 1024 is in
`outputs/profile-mmq-q4_0-1024-current-20260515/grouped-summary.json`. The
largest steady groups are still the stage-2 MLP expansion/projection and the
QKV projection around `[448,196,25]`.

| Experiment | Artifact | Result |
| --- | --- | --- |
| Current default cap | `outputs/mmq-q4-xcap-ab5-1024-20260515/profile-summary.json` | 5-run mean `67.60 ms/frame` |
| `GGML_CUDA_MMQ_X_MAX=96` | `outputs/mmq-q4-xcap-ab5-1024-20260515/profile-summary.json`, `outputs/mmq-q4-x96-parity-1024-20260515/summary.json` | 5-run mean `67.40 ms/frame`, parity exact |
| `GGML_CUDA_MMQ_X_MAX=88` | `outputs/mmq-q4-xcap-ab5-1024-20260515/profile-summary.json` | 5-run mean `68.24 ms/frame`, slower |

The `96` cap is at most a tiny noise-level improvement and is not enough to
justify replacing the current q4_0 Blackwell cap. The useful MMQ work remains a
shape-specific kernel/body improvement for the dominant MLP shapes, not another
global `mmq_x` cap retune.

### Current q4_0 D56 event profile

After GPU persistence mode was enabled, the current q4_0 1024 D56 wrapper was
profiled again with `GGML_CUDA_PROFILE_FATTN56=1`:
`outputs/fattn56-current-q4_0-1024-persistence-20260515/summary.json`.

| Shape | Count | Pack drop-max sum | MMA drop-max sum | Slice drop-max sum | Total drop-max sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q=[56,196,8,25]` | 48 | `2.98 ms` | `7.80 ms` | `0.12 ms` | `10.90 ms` |
| `Q=[56,4096,8,1]` | 12 | `0.51 ms` | `9.84 ms` | `0.03 ms` | `10.38 ms` |
| `Q=[56,64,2,1024]` | 8 | `1.86 ms` | `2.36 ms` | `0.02 ms` | `4.24 ms` |

The important detail is that the main window/global Hiera shapes report
`qkv_contiguous=0`, so the now-disabled contiguous V56 MMA path is not used there.
They still go through the safe combined pack path and a padded MMA body. This
keeps the next useful D56 target focused on a correctness-aware non-contiguous
or native-D56 body path, not another contiguous-only copy cleanup.

### Rejected non-contiguous V56 MMA wrapper

A direct attempt was made to extend the V56 MMA wrapper to non-contiguous Q/K/V
inputs by packing Q/K as padded 64-wide tensors and V as a 56-wide tensor. This
was intended to make the current `Q=[56,196,8,25]` and `Q=[56,4096,8,1]`
Hiera shapes use the narrower V/output MMA path. It is not correct as written:

| Check | Artifact | Result |
| --- | --- | --- |
| Broken attempt parity | `outputs/fattn56-strided-v56-parity-q4_0-1024-20260515/summary.json` | `0 / 10` mask hashes equal, max bbox delta `401 px` |
| Reverted parity | `outputs/fattn56-strided-v56-reverted-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |

The failed attempt was removed. Future work in this area needs to audit the MMA
kernel's layout assumptions for `DV=56` and non-contiguous source views before
reusing that path for the main Hiera window/global shapes.

### Rejected Hiera V-contiguous graph rewrite

A smaller follow-up tried to make the application graph feed contiguous V
tensors into Hiera attention (`V = cont(permute(V))`) so that the existing
contiguous V56 MMA path could fire without adding a non-contiguous CUDA wrapper.
The profile confirmed that the main Hiera rows changed to `qkv_contiguous=1`,
but the default V56 path was not correct for these rows:

| Check | Artifact | Result |
| --- | --- | --- |
| V-contiguous default V56 parity | `outputs/hiera-contiguous-v-parity-q4_0-1024-20260515/summary.json` | `0 / 10` mask hashes equal, max bbox delta `960 px` |
| V-contiguous with V56 disabled | `outputs/hiera-contiguous-v-parity-q4_0-1024-20260515/summary_v56_disabled.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| V-contiguous with V56 scratch output | `outputs/hiera-contiguous-v-v56-scratch-parity-q4_0-1024-20260515/summary.json` | still `0 / 10` mask hashes equal, max bbox delta `960 px` |
| V-contiguous D56 profile | `outputs/hiera-contiguous-v-profile-q4_0-1024-20260515/run.log` | `qkv_contiguous=1`, but the profiled D56 totals did not improve |

This isolates the failure to the V56 MMA path rather than to the extra
`ggml_cont` copy itself. Forcing V56 to write through a 64-wide scratch buffer
does not fix it, so the issue is in the `DV=56` MMA body/tail handling rather
than only direct output addressing. The graph rewrite was reverted. Future V56
expansion must first fix or constrain the DV=56 MMA kernel's layout assumptions
for these Hiera rows; simply making V contiguous in `sam3.cpp` is not a valid
optimization.

### Rejected q4_0 cuBLAS fallback

The xmake CUDA build front door now has comparison-only switches for ggml CUDA
backend selection:

- `--cuda_force_cublas=y` builds in `build/xmake-release-cuda-force-cublas`
  with `GGML_CUDA_FORCE_CUBLAS=ON`.
- `--cuda_force_mmq=y` builds in `build/xmake-release-cuda-force-mmq` with
  `GGML_CUDA_FORCE_MMQ=ON`.
- The same switches are available through the normal `just` front door with
  `CUDA_FORCE_CUBLAS=y just build` or `CUDA_FORCE_MMQ=y just build`.

This keeps the normal release CUDA build separate from backend-forcing
experiments. A current q4_0 Base+ 1024 comparison rejected the cuBLAS fallback:

| Mode | Runs | Mean track ms/frame | Evidence |
| --- | --- | ---: | --- |
| default MMQ | `68.2, 66.8, 67.1, 68.0, 67.4` | `67.5` | `outputs/force-cublas-ab-q4-1024-20260515/summary.json` |
| `GGML_CUDA_FORCE_CUBLAS=ON` | `163.2, 162.0, 163.5, 162.4, 161.9` | `162.6` | `outputs/force-cublas-ab-q4-1024-20260515/summary.json` |

So the dominant q4_0 Hiera MLP path must stay on MMQ. The remaining useful MMQ
work is inside the MMQ body/tiling for the repeated stage-2 shapes, not routing
the quantized matmuls through dequantization plus cuBLAS.

### Rejected DS4 contiguous MMQ quantizer

A narrow `ids == nullptr`, contiguous-input DS4 q8_1 activation quantizer was
tested for q4_0/q4_1 MMQ. It kept the same q8_1 rounding and layout as the
generic quantizer, but removed the generic ids/stride address calculation for
the regular Hiera MLP shapes.

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/mmq-ds4-contiguous-quant-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| MMQ event profile | `outputs/mmq-ds4-contiguous-quant-profile-q4_0-1024-20260515/summary.txt` | quantized-profile total improved `61.10 -> 59.88 ms`; quantize alone `11.67 -> 10.36 ms` |
| 5-run bbox-only A/B | `outputs/mmq-ds4-contiguous-quant-speed-q4_0-1024-20260515/summary.json` | disabled mean `67.26 ms/frame`, enabled mean `67.50 ms/frame`; paired mean `-0.24 ms` |

The standalone quantization event got smaller, but the end-to-end benchmark did
not improve. The branch was removed. This reinforces that MMQ work should target
the body/tiling of the dominant MLP matmuls, not only the activation quantizer.

### Python comparison environment pin

The official-PyTorch comparison scripts now pin `torch==2.8.0` and
`torchvision==0.23.0`. Leaving them unconstrained caused `uv` to resolve a
newer cu13 stack, which failed on this machine with
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. The pinned pair resolves to the
cu128 wheels, supports the RTX 5070 Ti Laptop GPU's `sm_120` target, and passed
a CUDA Conv2D smoke check before the official SAM2 quality comparison was
rerun.

### Rejected q4_0 MMQ X=128 default

The q4_0 Blackwell MMQ cap was retested above the current default 104. A short
1024 sweep suggested that larger caps can be slightly faster:

| Cap | Track ms/frame runs | Mean | Evidence |
| --- | --- | ---: | --- |
| current default | `69.0, 66.8, 67.1` | `67.63` | `outputs/mmq-q4-xcap-high-sweep-1024-20260515/summary.json` |
| `112` | `67.4, 67.5, 67.1` | `67.33` | same |
| `120` | `67.0, 67.0, 67.4` | `67.13` | same |
| `128` | `67.1, 66.7, 66.8` | `66.87` | same |

However, `GGML_CUDA_MMQ_X_MAX=128` is not a clean speed-only change. Against
the default path, full-mask hashes changed on all 10 rows
(`outputs/mmq-q4-x128-parity-1024-20260515/summary.json`). The deltas were small
for boxes (`max_bbox_delta_px=2`) but not bit-exact (`mask_hash_equal_rows=0`).

Under the pinned official-PyTorch comparison environment, X=128 improved the
1024 multimask quality row but still did not reach parity:

| Mode | Mean mask IoU | Min mask IoU | Initial candidate | Evidence |
| --- | ---: | ---: | --- | --- |
| current default | `0.9795` | `0.9744` | selected index matches official | `outputs/sam2-official-quality-q4_0-default-1024-multimask-torch28-20260515/summary.json` |
| `GGML_CUDA_MMQ_X_MAX=128` | `0.9825` | `0.9793` | selected index matches official | `outputs/sam2-official-quality-q4_0-x128-1024-multimask-20260515/summary.json` |

At 512, X=128 did not help speed or quality. Five-run bbox-only means were
`19.14 ms/frame` for the default and `19.16 ms/frame` for X=128, and official
quality remained unusable (`mean/min mask IoU = 0.0311/0.0043`, selected
candidate still mismatched official) in
`outputs/sam2-official-quality-q4_0-x128-512-multimask-20260515/summary.json`.

The cap is therefore not promoted to the default. It remains useful evidence
that MMQ scheduling can shift quality and speed slightly, but it is too small
and too non-bit-exact to close the PyTorch gap.

### Rejected MMQ stream-k tile-efficiency threshold change

The MMQ launch profiler was extended to print the selected `mmq_x`, `mmq_y`,
stream-k mode, tile counts, SM count, tile efficiency, stream block count,
fixup use, and shared-memory size. This confirmed that most dominant q4_0
Base+ Hiera MLP launches already use the existing high-efficiency tiling path.

| Check | Artifact | Result |
| --- | --- | --- |
| Launch-shape profile | `outputs/mmq-launch-profile-q4_0-1024-20260515/summary.txt` | dominant repeated q4_0 shapes use `mmq_x=96` or `80`; most high-volume stream-k launches already have `fixup=0` |
| Threshold sweep | `outputs/mmq-tile-eff-sweep-q4_0-1024-20260515/summary.json` | best short-run mean was `67.53 ms/frame` at threshold `0` vs default `67.97`, within noise and far from the PyTorch gap |
| Full-mask parity for threshold `0` | `outputs/mmq-tile-eff0-parity-q4_0-1024-20260515/summary.json` | `0 / 10` mask hashes equal, max bbox delta `3 px`, max mask-area relative delta `0.0172` |

Lowering the stream-k tile-efficiency threshold is therefore not accepted. It
can shift scheduling and slightly change masks, but does not provide a
statistically meaningful speed win or close the official-PyTorch quality gap.
The code keeps only the launch profiler; the unsafe threshold knob was removed.

### Rejected axis-broadcast grid kernel

The CUDA node profile showed that `ADD` is a non-trivial aggregate cost, so a
small axis-broadcast experiment specialized the existing F32 axis fast path for
the SAM-heavy axis0/axis2 bias-add cases. The idea was to replace per-element
axis division/modulo with a 2D grid where the broadcast index is constant per
row/plane.

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/binbcast-axis-grid-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| 7-pair bbox-only A/B | `outputs/binbcast-axis-grid-speed-q4_0-1024-20260515/summary.json` | old axis mean `65.27 ms/frame`, grid axis mean `65.37 ms/frame`, paired old-minus-grid mean `-0.10 ms` |

The change was exact but did not improve end-to-end speed, so it was reverted.
This rules out simple axis-broadcast index arithmetic as a meaningful route to
closing the current PyTorch gap.

### Rejected Hiera contiguous-V graph rewrite

The Hiera attention graph was rechecked with an opt-in rewrite that inserts
`ggml_cont` on V before `ggml_flash_attn_ext`. This was tested after unsafe
V56 MMA had been disabled by default, so it no longer triggered the earlier
incorrect V56 accumulator path.

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/hiera-contiguous-v-env-parity-q8_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| 5-pair bbox-only A/B | `outputs/hiera-contiguous-v-env-speed-q8_0-1024-20260515/summary.json` | default steady rows around `63.2-63.6 ms/frame`, contiguous-V rows around `65.1-65.7 ms/frame`; every pair slower |

The rewrite is therefore rejected. It enables a more contiguous pack shape, but
the extra `CONT` node costs more than it saves for this workload.

### Current Base+ precision speed refresh

After the CUDA environment repair and the latest accepted ggml changes, Base+
precision speed was refreshed on the matched 1024 input. The current fastest
checked C++ row is q8_0, not q4_0:

| Precision | Track ms/frame runs | Mean | Evidence |
| --- | --- | ---: | --- |
| f16 | `73.7, 74.3, 73.5` | `73.83` | `outputs/current-base-plus-precision-speed-1024-20260515b/summary.json` |
| q8_0 | `63.4, 63.4, 63.4` | `63.40` | same |
| q4_1 | `65.8, 66.0, 66.0` | `65.93` | same |
| q4_0 | `64.9, 65.1, 65.3` | `65.10` | same |

The corresponding official PyTorch bf16 row remains `42.45 ms/frame` at the
same input size, so even the best current C++ precision is still about `1.49x`
slower. The refreshed q8_0 node profile
(`outputs/current-node-profile-q8_0-1024-20260515b/summary.txt`) still points to
the same root causes as the q4_0 profile: Hiera FlashAttention and MMQ body
time dominate; small graph rewrites around bias adds, broadcast indexing, or V
contiguity do not close the gap.

The q8_0 official-quality rows were also refreshed under the pinned Torch 2.8
environment:

| Encode size | C++ q8_0 speed | Official PyTorch speed | Mean mask IoU | Min mask IoU | Initial candidate | Evidence |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1024 | `63.40 ms/frame` | `42.05 ms/frame` | `0.9874` | `0.9849` | selected index matches official | `outputs/sam2-official-quality-current-q8_0-1024-multimask-torch28-20260515/official/summary.json` |
| 512 | `16.72 ms/frame` | `14.68 ms/frame` | `0.8824` | `0.8322` | selected index matches official | `outputs/sam2-official-quality-current-q8_0-512-multimask-torch28-20260515/official/summary.json` |

The 512 q8_0 row is the closest current speed row, only about `14%` slower than
official PyTorch, but its mask quality is not a valid fallback baseline. The
1024 q8_0 row is close on quality but still below the strict `>= 0.99` parity
target and remains far slower than PyTorch.

The 1024 quality ceiling was checked with f16 and f32 Base+ as well:

| Precision | Mean mask IoU | Min mask IoU | Initial candidate | Evidence |
| --- | ---: | ---: | --- | --- |
| f32 | `0.98737` | `0.98477` | selected index matches official | `outputs/sam2-official-quality-current-f32-1024-multimask-torch28-20260515/official/summary.json` |
| f16 | `0.98735` | `0.98474` | selected index matches official | `outputs/sam2-official-quality-current-f16-1024-multimask-torch28-20260515/official/summary.json` |
| q8_0 | `0.98736` | `0.98491` | selected index matches official | `outputs/sam2-official-quality-current-q8_0-1024-multimask-torch28-20260515/official/summary.json` |

The near-identical f32/f16/q8_0 rows mean the remaining 1024 quality gap is not
explained by fallback precision. Future quality work should focus on model
logic parity around SAM2 propagation, resizing, or mask post-processing rather
than on Hiera tensor precision alone.

### Rejected Hiera patch-embed f16 cast

The CUDA node profile identifies the Hiera patch embedding as a large single
`MUL_MAT` row, so an env-gated cast of only the patch-embed weight to f16 was
tested as a low-risk alternative to writing a custom fused patch-embed kernel.

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/hiera-patch-embed-f16-parity-q8_0-1024-20260515/compare` | `0 / 10` mask hashes equal, max bbox delta `2 px`, max mask-area relative delta `0.00107` |

The change was rejected and reverted. It perturbs model outputs while only
targeting one of the Hiera hotspots; a real patch-embed improvement needs either
a parity-preserving direct f32 CUDA kernel or a more comprehensive official-like
mixed-precision policy that is validated against Python quality, not an isolated
weight cast.

### Rejected Hiera patch-embed direct conv

The existing ggml `ggml_conv_2d_direct` CUDA path was tested as an env-gated
replacement for Hiera patch embedding. This avoids the im2col + `MUL_MAT`
decomposition, but it changes accumulation order and uses a scalar direct-conv
kernel.

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/hiera-patch-embed-direct-parity-q8_0-1024-20260515/compare` | `0 / 10` mask hashes equal, max bbox delta `19 px`, max mask-area relative delta `0.00094` |

The direct path was also not faster in the 10-frame smoke run, so it was
rejected and reverted. The patch-embed hotspot remains real, but the current
generic direct conv kernel is neither parity-preserving nor a speed win for this
shape. A viable replacement would need a shape-specialized f32 kernel that
matches the current im2col/matmul accumulation closely enough for tracking
parity.

The PVS detection path now also preserves the raw object-score logit for
tracking internals. Public `mask.obj_score` remains the sigmoid probability, but
memory encoding and object-pointer extraction use the raw logit, matching
official SAM2's `object_score_logits > 0` semantics. On the current q8_0/Base+
1024 row this is output-exact because the selected object logit is positive:

| Check | Artifact | Result |
| --- | --- | --- |
| Previous q8_0 1024 JSONL vs raw-logit tracking internals | `outputs/raw-obj-logit-parity-q8_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| Official comparison after raw-logit tracking internals | `outputs/sam2-official-quality-current-q8_0-1024-raw-obj-logit-20260515/official/summary.json` | mean mask IoU `0.9874`, min mask IoU `0.9849` |

CUDA graph opt-in was retested with the current q8_0 binary and remains
negative:

| Encode size | Default mean | `SAM3_CUDA_ENABLE_GRAPHS=1` mean | Result | Evidence |
| --- | ---: | ---: | --- | --- |
| 1024 | `63.60 ms/frame` | `64.58 ms/frame` | slower | `outputs/q8_0-cuda-graphs-current-1024-20260515/summary.json` |
| 512 | `16.92 ms/frame` | `17.80 ms/frame` | slower | `outputs/q8_0-cuda-graphs-current-512-20260515/summary.json` |

The q8 graph opt-in stays disabled by default.

### Rejected raw initial mask logits for memory conditioning

The tracker initialization path was tested with the selected SAM decoder logits
fed directly into the memory encoder instead of the current binary-mask-derived
`+6/-6` synthetic logits. This was a quality-parity experiment, not a speed
optimization.

| Check | Artifact | Result |
| --- | --- | --- |
| q8_0/Base+/1024 official comparison | `outputs/sam2-official-quality-current-q8_0-1024-real-initial-logits-20260515/official/summary.json` | mean mask IoU `0.9652`, min mask IoU `0.9576`, selected candidate still matches official |

This is worse than the current q8_0 1024 row (`0.9874` mean, `0.9849` min), so
the change was reverted. The remaining quality gap is therefore unlikely to be
fixed by simply replacing the initial synthetic conditioning logits with raw
low-resolution decoder logits.

### Rejected MMQ contiguous-input quantization specialization

The q4_0 MMQ path was profiled to check whether the repeated `src1` f32 to
q8_1 quantization could be a meaningful root-cause target. A 3-frame
`GGML_CUDA_PROFILE_MMQ=1` run produced:

| Metric | Value | Evidence |
| --- | ---: | --- |
| MMQ profile rows | `384` | `outputs/mmq-profile-q4_0-1024-20260515-continuation/stderr.txt` |
| Sum quantization time | `8.91 ms` | same |
| Sum MMQ body time | `39.97 ms` | same |
| Quantization share of profiled MMQ time | `18.2%` | same |

An env-gated contiguous/no-ids q4_0 quantization specialization was then tested.
It preserved exact output but did not show a meaningful end-to-end gain:

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/mmq-fast-quant-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| 5-pair bbox-only A/B | `outputs/mmq-fast-quant-speed-q4_0-1024-20260515/summary.json` | default mean `67.4 ms/frame`, fast mean `67.2 ms/frame`, paired mean `+0.2 ms` with `0.84 ms` stdev |

The change was reverted. This confirms that MMQ input quantization is visible in
the profile, but a small contiguous-input specialization is not enough to close
the current PyTorch gap.

The same MMQ profile was refreshed for the current q8_0/Base+/1024 row:

| Metric | Value | Evidence |
| --- | ---: | --- |
| MMQ profile rows | `384` | `outputs/mmq-profile-q8_0-1024-current-20260515-continuation/summary.json` |
| Sum quantization time | `8.69 ms` | same |
| Sum MMQ body time | `38.86 ms` | same |
| Quantization share of profiled MMQ time | `18.3%` | same |

The largest grouped q8_0 MMQ shapes are the Hiera MLP projections:

| Shape | Count | Total | Quant | MMQ body |
| --- | ---: | ---: | ---: | ---: |
| `src0=[1792,448] src1=[1792,4096] dst=[448,4096]` | `48` | `6.73 ms` | `1.52 ms` | `5.20 ms` |
| `src0=[448,1792] src1=[448,4096] dst=[1792,4096]` | `48` | `6.52 ms` | `0.54 ms` | `5.97 ms` |
| `src0=[448,112] src1=[448,65536] dst=[112,65536]` | `6` | `5.76 ms` | `1.65 ms` | `4.12 ms` |
| `src0=[448,1344] src1=[448,196,25] dst=[1344,196,25]` | `36` | `5.42 ms` | `0.44 ms` | `4.98 ms` |

`GGML_CUDA_PROFILE_MMQ_LAUNCH=1` confirmed that the current q8_0 path already
uses `mmq_x=128, mmq_y=128` on the dominant shapes, including the MLP rows.
This rules out the simple q8_0 tile-cap issue that previously affected some
other quantized paths. The remaining q8_0 MMQ opportunity is therefore a real
kernel/body-layout improvement, not another global `MMQ_X_MAX` retune.

A `GGML_CUDA_FORCE_CUBLAS=ON` comparison was also run for the same q8_0/Base+
1024 row to test whether dequantization plus cuBLAS could beat the current MMQ
path:

| Backend policy | Mean track time | Evidence |
| --- | ---: | --- |
| Default MMQ | `63.4 ms/frame` | `outputs/current-base-plus-precision-speed-1024-20260515b/summary.json` |
| Forced cuBLAS | `112.3 ms/frame` | `outputs/q8_0-force-cublas-speed-1024-20260515/summary.json` |
| Official PyTorch bf16 | `42.45 ms/frame` | same default summary |

This rules out cuBLAS fallback as a speed path for q8_0. Beating PyTorch on this
row requires improving the quantized CUDA kernels themselves, not routing the
dominant quantized Hiera matmuls through cuBLAS.

### Rejected CUDA Norm float4 finalization

Metal has `float4` Norm/RMSNorm variants, so CUDA was tested with a narrower
env-gated variant that kept the existing scalar reduction order and only
vectorized the final writeback/affine step for 4-aligned rows. This avoids the
most obvious parity risk from changing the mean/variance reduction order.

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/norm-vec4-finalize-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| 5-pair bbox-only A/B | `outputs/norm-vec4-finalize-speed-q4_0-1024-20260515/summary.json` | default mean `67.4 ms/frame`, vec4 mean `67.0 ms/frame`, paired mean `+0.4 ms` with `1.14 ms` stdev |

The change was exact but too small and noisy to accept. A broader Norm rewrite
would need to attack reduction efficiency itself, but that is also the part most
likely to perturb mask parity, so it remains a lower-priority route than the
dominant Hiera attention/MMQ body kernels.

### Current FATTN56 profile refresh

The q8_0/Base+/1024 D56 attention path was profiled again after the rejected
small-kernel experiments:

| Metric | Value | Evidence |
| --- | ---: | --- |
| FATTN56 rows | `72` | `outputs/fattn56-profile-q8_0-1024-20260515-continuation/summary.json` |
| Total FATTN56 time | `33.81 ms` | same |
| Pack time | `14.27 ms` (`42.2%`) | same |
| MMA time | `19.35 ms` (`57.2%`) | same |
| Drop-max total | `27.04 ms` | same |
| Drop-max pack time | `7.90 ms` (`29.2%`) | same |
| Drop-max MMA time | `18.96 ms` (`70.1%`) | same |

Top drop-max shapes:

| Q shape | Combined pack | Count | Total | Pack | MMA | Pack share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[56,4096,8,1]` | yes | `9` | `8.53 ms` | `0.42 ms` | `8.08 ms` | `4.9%` |
| `[56,196,8,25]` | yes | `36` | `8.47 ms` | `2.34 ms` | `6.04 ms` | `27.6%` |
| `[56,64,2,1024]` | yes | `5` | `3.03 ms` | `1.34 ms` | `1.69 ms` | `44.1%` |
| `[56,16,4,1024]` | no | `3` | `2.42 ms` | `1.60 ms` | `0.81 ms` | `66.1%` |

This moves the priority order away from minor Norm/elementwise rewrites. The
largest remaining attention win would need to reduce D56 MMA time for the global
and stage-2 window shapes, while a secondary win is to reduce pack overhead for
the downsample/cross-window shapes where pack remains more than `40-60%` of the
local attention cost.

The stale experimental `GGML_CUDA_ENABLE_UNSAFE_FATTN56_V56_MMA=1` call path was
then removed from `fattn.cu`. The remaining default path keeps V padded to 64 and
uses the `DV_DST=56` direct-output writeback. This avoids exposing the known
unsafe `DV=56` accumulator path while preserving the accepted default behavior.
The default path is covered by the mixed-pack parity rows below; both report
`10 / 10` mask hashes equal, max bbox delta `0 px`, and max score delta `0`.

### Accepted FATTN56 mixed-shape pack

The non-combined q-pool attention rows have Q and K/V with different spatial
shape, so they previously fell back to three independent pad kernels. A
mixed-shape pack kernel now pads Q and K/V in one launch when K/V share shape and
Q shares the remaining outer dimensions. This keeps the same source indexing and
conversion semantics as the old per-tensor kernels.

| Check | Artifact | Result |
| --- | --- | --- |
| q8_0/Base+/1024 full-mask parity | `outputs/fattn56-mixed-pack-parity-q8_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| q4_0/Base+/1024 full-mask parity | `outputs/fattn56-mixed-pack-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| FATTN56 profile A/B | `outputs/fattn56-mixed-pack-profile-mixed-q8_0-1024-20260515/summary.json` and `outputs/fattn56-mixed-pack-profile-default_disabled-q8_0-1024-20260515/summary.json` | non-combined pack total drops from about `2.87 ms` to `2.41 ms`; drop-max FATTN56 total changes from `26.39 ms` to `25.72 ms` |
| 5-pair bbox-only A/B | `outputs/fattn56-mixed-pack-speed-q8_0-1024-20260515/summary.json` | disabled mean `65.4 ms/frame`, mixed mean `65.0 ms/frame`, paired mean `+0.4 ms` with `0.89 ms` stdev |
| Current q8_0/Base+/512 speed | `outputs/fattn56-mixed-pack-speed-q8_0-512-20260515/summary.json` | mean `17.6 ms/frame`, median `17.0 ms/frame`; still slower than official PyTorch bf16 `14.68 ms/frame` |

This is an exact but small improvement. It does not materially close the PyTorch
gap by itself; it mainly removes avoidable launch overhead in the remaining
FATTN56 pack-heavy q-pool rows.

### Rejected patch-embed bias reorder

The patch embedding bias add was tested after the channel-first permute instead
of before it, changing the broadcast from `[W,H,E] + [1,1,E]` to
`[E,W,H] + [E,1,1]`. This preserved full-mask parity but did not remove the large
patch-embed bias ADD hotspot:

| Check | Artifact | Result |
| --- | --- | --- |
| q8_0/Base+/1024 full-mask parity | `outputs/patch-bias-after-permute-parity-q8_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| q8_0/Base+/1024 node profile | `outputs/patch-bias-after-permute-node-profile-q8_0-1024-20260515/summary.json` | patch-embed bias ADD remains `3.37 ms` in the profiled 10-frame run |

The graph rewrite was reverted. The useful patch-embed route remains a true
kernel replacement for im2col/matmul/bias/permute, not merely changing where the
bias broadcast appears.

### Rejected FATTN56 pack block-size retune

The FATTN56 pack/slice launch block size was tested with an env-gated `512`
thread block instead of the current `256`. This changes only the launch shape for
the existing pack/slice kernels.

| Check | Artifact | Result |
| --- | --- | --- |
| Full-mask parity | `outputs/fattn56-pack512-parity-q4_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| 5-pair bbox-only A/B | `outputs/fattn56-pack512-speed-q4_0-1024-20260515/summary.json` | default mean `67.2 ms/frame`, pack512 mean `67.2 ms/frame`, paired mean `0.0 ms` |

The retune was exact but had no measurable effect, so it was reverted. Future
pack work needs a different memory layout or kernel structure, not just a launch
block-size change.

### Current official-quality audit refresh

The official-PyTorch quality audit was refreshed with the current q4_0
multimask path. This supersedes the older full-mask-only audit rows for the
current target clip. All rows use the same decoded source video, the same point
prompt, the same 10-frame range, and a matched SAM input resolution.

| Encode size | C++ mode | Mean mask IoU | Min mask IoU | Initial candidate result | Evidence |
| --- | --- | ---: | ---: | --- | --- |
| 1024 | `--multimask` | `0.9795` | `0.9757` | 3 candidates on both sides; selected index `0` matches official PyTorch | `outputs/sam2-official-quality-current-q4_0-1024-multimask-20260515/summary.json` |
| 512 | `--multimask` | `0.0410` | `0.0105` | 3 candidates on both sides; C++ selects `2`, official PyTorch selects `0` | `outputs/sam2-official-quality-current-q4_0-512-multimask-20260515/summary.json` |
| 512 | `--multimask --initial-candidate-index 0` | `0.5866` | `0.4679` | selected index is forced to the official candidate, but the mask still differs substantially | `outputs/sam2-official-quality-current-q4_0-512-multimask-candidate0-20260515/summary.json` |

The 1024 result is close but still below the strict parity target
(`mean/min >= 0.99`). The 512 result is not only a candidate-scoring mismatch:
forcing candidate `0` improves the row, but the C++ mask area and tracked mask
shape still do not match official PyTorch closely enough for a Rust-wrapper
quality baseline.

The goal audit was regenerated as:

- `outputs/goal-audit/latest-summary-20260515-multimask.json` for the current
  default multimask behavior.
- `outputs/goal-audit/latest-summary-20260515-multimask-candidate0-diagnostic.json`
  for the diagnostic 512 candidate-0 row.
- After enabling GPU persistence mode, the same audit was regenerated as
  `outputs/goal-audit/latest-summary-20260515-persistence-multimask.json` and
  `outputs/goal-audit/latest-summary-20260515-persistence-multimask-candidate0-diagnostic.json`.

Both audits remain `not_complete`. The already-met pieces are the matched input
contract, Hiera/profile attribution, and speed-priority identification. The
missing pieces are still:

1. C++ must beat official PyTorch on matched 512 and 1024 rows. The current q4_0
   rows after persistence-mode repair are `18.5 ms/frame` vs `11.13 ms/frame`
   at 512 and `67.9 ms/frame` vs `42.45 ms/frame` at 1024
   (`outputs/model-matrix-current-q4-512-persistence-20260515/summary.json` and
   `outputs/model-matrix-current-q4-1024-persistence-20260515/summary.json`).
2. Base+ fallback quality must reach official-PyTorch parity. The current best
   1024 q4_0 multimask row is close but not enough, while 512 remains far from
   parity even when the initial candidate is forced.

### Propagation candidate-selection diagnostics

The official-PyTorch comparison script now records the per-frame SAM-head
candidate selected during propagation and compares it against the C++ JSONL
fields. C++ stores the original SAM token index (`1..3`) while official PyTorch
reports the sliced multimask candidate index (`0..2`), so the comparison
normalizes C++ by subtracting one when all four decoder IoU scores are present.

| C++ model | Encode size | Mean mask IoU | Min mask IoU | Propagation candidate matches | Mismatches | Evidence |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Base+ `q8_0` | `1024` | `0.987359` | `0.984906` | `7 / 9` | offsets `4`, `7` | `outputs/sam2-official-quality-current-q8_0-1024-prop-selection-diagnostics-20260515/official/summary.json` |
| Base+ `f32` | `1024` | `0.987369` | `0.984770` | `7 / 9` | offsets `4`, `7` | `outputs/sam2-official-quality-current-f32-1024-prop-selection-diagnostics-20260515/official/summary.json` |
| Base+ `q8_0`, no full-mask cleanup | `1024` | `0.987378` | `0.984910` | `7 / 9` | offsets `4`, `7` | `outputs/sam2-official-quality-current-q8_0-1024-no-fullmask-cleanup-20260515/official/summary.json` |
| Base+ `f32` vs official PyTorch `fp32` | `1024` | `0.989100` | `0.987321` | `6 / 9` | offsets `1`, `2`, `6` | `outputs/sam2-official-quality-current-f32-1024-prop-selection-diagnostics-pytorch-fp32-20260515/official/summary.json` |
| Base+ `q8_0`, bf16 memory round-trip diagnostic | `1024` | `0.987350` | `0.984879` | `7 / 9` | offsets `4`, `7` | `outputs/sam2-official-quality-current-q8_0-1024-bf16-memory-roundtrip-20260515/official/summary.json` |
| Base+ `q8_0`, low-res candidate-area diagnostic | `1024` | `0.987359` | `0.984906` | `7 / 9` | offsets `4`, `7` | `outputs/sam2-official-quality-current-q8_0-1024-lowres-candidate-areas-20260515/official/summary.json` |

The same two propagation mismatches appear in both `q8_0` and `f32`, so this is
not a quantization-specific failure. Disabling C++ full-resolution binary-mask
cleanup only changes the mean by about `0.00002`, so cleanup is not the primary
quality gap either. A diagnostic C++ bf16 round-trip of stored SAM2 memory
features also did not improve the official-bf16 comparison, so it was not kept.
Comparing against official PyTorch `fp32` improves the mean to `0.989100`, but
still does not reach the strict `0.99` parity line and changes which near-tie
candidate frames disagree. The remaining quality work should focus on SAM2
propagation semantics before more precision-specific experiments:

1. Compare C++ and official PyTorch low-resolution propagation logits for all
   three multimask candidates on the mismatch frames. The low-res area
   diagnostic shows that candidate mask sizes are close even on mismatches:
   offset `4` C++ `[28941, 28581, 28890]` vs official
   `[28880, 28295, 28726]`, and offset `7` C++ `[29233, 28880, 29185]`
   vs official `[29195, 28689, 29126]`. This points more toward the IoU head /
   decoder-score path than gross candidate-mask shape failure.
2. Check memory-encoder inputs before storage; post-storage bf16 rounding alone
   was ruled out by the diagnostic row above.
3. Only after low-res logits align, revisit final mask resizing/postprocessing.

### 2026-05-15 environment repair and rejected MMQ quantize split

The Python official-SAM2 environment was repaired after a failed CUDA 13 wheel
experiment left Torch import broken with missing CUDA shared libraries. The
project environment is pinned to `torch==2.8.0` / `torchvision==0.23.0`, which
resolves to CUDA 12.8 wheels and works on the RTX 5070 Ti Laptop target. After
`uv sync`, some NVIDIA wheel payloads were present only as metadata in the local
environment, so the CUDA 12 NVIDIA wheel set was reinstalled. The repair was
validated with:

- `uv run python` importing Torch `2.8.0+cu128`, reporting CUDA `12.8`, and
  completing a small CUDA matmul.
- `uv run scripts/profile_sam2_official.py` on the existing JPEG frame folder,
  producing `outputs/official-env-smoke-20260515/summary.json`.
- `just build`, which continued to pass for the C++/CUDA path.

An MMQ activation-quantize experiment was also tested and rejected. The change
split `quantize_mmq_q8_1` into `ids` and `no ids` template instantiations so the
common no-ids path would not carry the runtime `ids ? ids[i1] : i1` branch. It
kept the same arithmetic and storage format. Parity was exact against the
current q8_0 Base+ 1024 reference (`10/10` mask hashes equal, bbox and score
delta zero) in `outputs/mmq-ids-template-parity-q8_0-1024-20260515/summary.json`,
but five bbox-only timing runs regressed from the previous current baseline:

| Experiment | Runs | Mean track ms/frame | Delta vs previous current q8_0 1024 baseline | Evidence |
| --- | ---: | ---: | ---: | --- |
| `quantize_mmq_q8_1` `ids`/`no ids` template split | 5 | `64.44` | `+1.04 ms` | `outputs/mmq-ids-template-speed-q8_0-1024-20260515/summary.json` |

The split was reverted. Do not reattempt this branch-removal-only quantize
specialization unless a lower-level compiler/codegen profile shows a new reason
it should help.

An additional binary-broadcast experiment specialized the existing f32
axis-broadcast fast path for the Hiera-heavy `axis=0` bias adds and the patch
embed `axis=2` bias add. The dedicated kernels used 2D/3D indexing to avoid the
generic per-element modulo/division in `k_bin_bcast_axis_f32`, while keeping the
same `a + b` arithmetic. The output was exact against the current q8_0 Base+
1024 reference (`10/10` mask hashes equal, bbox and score delta zero) in
`outputs/binbcast-axis-specialized-parity-q8_0-1024-20260515/summary.json`, but
five bbox-only timing runs regressed:

| Experiment | Runs | Mean track ms/frame | Delta vs previous current q8_0 1024 baseline | Evidence |
| --- | ---: | ---: | ---: | --- |
| f32 axis-broadcast dedicated `axis=0`/`axis=2` kernels | 5 | `65.18` | `+1.78 ms` | `outputs/binbcast-axis-specialized-speed-q8_0-1024-20260515/summary.json` |

The specialized kernels were reverted. The existing 1D axis-broadcast fast path
remains the better implementation for the measured Hiera shapes.

### 2026-05-15 refreshed current-speed gap

After the Python CUDA environment repair and after reverting the rejected
quantize / binary-broadcast experiments, the matched Base+ q8_0 1024 speed row
was refreshed against official PyTorch bf16. Both sides use the same 10-frame
clip, point prompt, and SAM input size. The official PyTorch script performs two
warmup passes before each measured run.

| Runtime | Runs | Mean track ms/frame | Median | Stdev | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| C++ CUDA q8_0 Base+ | 5 | `65.56` | `65.10` | `0.85` | `outputs/current-q8_0-speed-1024-refresh-20260515/summary.json` |
| official PyTorch CUDA bf16 Base+ | 5 | `42.89` | `42.90` | `0.17` | `outputs/current-q8_0-speed-1024-refresh-20260515/summary.json` |

The current C++ path is therefore `22.67 ms/frame` slower, or `1.53x` the
official PyTorch bf16 runtime, on this matched row. The refreshed node profile
`outputs/current-node-profile-q8_0-1024-20260515c/hotspots.json` ranks the
remaining Hiera hotspots as:

1. head_dim=56 FlashAttention body: window `26.19 ms` drop-max and global
   `26.04 ms` drop-max in profiling mode.
2. q8_0 stage-2 MLP MMQ: `21.20 ms` and `20.76 ms` drop-max for the two main
   projection/expansion shapes.
3. elementwise/layout overhead: ADD/NORM/CONT groups below the attention/MMQ
   groups, and already tested simple broadcast specializations did not help.

This keeps the next useful optimization target on FlashAttention-56 body or a
substantial MMQ algorithmic improvement. Micro-specializing the existing
elementwise broadcast path is not enough to close the gap.

### 2026-05-15 upstream CUDA history check and rejected FATTN64 batch retune

After repairing the Python/Torch environment, upstream `ggml-org/ggml` was
fetched again and the CUDA commits after this branch's last upstream merge were
reviewed. The relevant recent CUDA history contains:

- FlashAttention support for MiMo `DKQ=192,DV=128`.
- MMQ stream-k overhead reduction.
- CUDA snake activation fusion and hardening.
- batched `OUT_PROD` inner loop via cuBLAS strided batched GEMM.
- im2col large-width handling and small build/include fixes.

The MMQ stream-k overhead reduction is already present in this branch's
`mmq.cuh`; it is not a missing optimization. The MiMo FlashAttention change
adds `192/128` tile/MMA cases and GQA routing, but it does not address SAM2
Base+ `head_dim=56`. The snake and out-prod changes do not match the current
Hiera hotspot list. Therefore no upstream commit in that range provides a
direct, safe speedup for the measured SAM2 Base+ CUDA path.

One FATTN scheduling experiment was then tested and rejected: the padded
`D56 -> D64` MMA path was retuned by changing the Ampere `DKQ=64,DV=64`
`ncols=16/32/64` configs from `nbatch_fa=64` to `128`. It compiled, but
full-mask parity failed before speed measurement:

| Experiment | Evidence | Result |
| --- | --- | --- |
| `DKQ=64,DV=64,ncols=16/32/64 nbatch_fa=128` | `outputs/fattn64-nbatch128-parity-q8_0-1024-20260515/summary.json` | `0 / 10` mask hashes equal, max bbox delta `3 px`, max score delta `0.006258` |
| restored source after rebuild | `outputs/fattn64-restored-parity-q8_0-1024-20260515/summary.json` | `10 / 10` mask hashes equal, max bbox delta `0 px`, score delta `0` |

The retune changes reduction/order enough to move masks and bboxes, so it was
reverted. This reinforces the current rule: FATTN-56 work must be correctness
first. Retuning the padded D64 MMA config is not an acceptable shortcut unless a
future implementation can prove exact parity against the restored path.

The native `V=56` MMA shortcut was also rechecked with a narrower patch: allocate
a 56-wide V buffer, round the VKQ accumulator tile count up, and zero shared V
tail lanes. The path was kept behind an explicit opt-in during the experiment,
but it still failed the isolated `sam3_fattn_parity` gate:

| Experiment | Evidence | Result |
| --- | --- | --- |
| native `DKQ=64,DV=56` with stream-k | `GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1 build/xmake-release-cuda/examples/sam3_fattn_parity --cuda --d 56 --n 196 --heads 8 --batch 25` | finite max error stayed below tolerance, but `2005248 / 2195200` checked values were non-finite |
| native `DKQ=64,DV=56` with stream-k disabled for that case | same command after disabling stream-k only for `DKQ=64,DV=56` | all checked values became non-finite |

That means the unsafe native V56 path is not just missing an accumulator ceil or
tail zero-fill. It likely needs a deeper rewrite of the VKQ tile/combine
contract. The experiment was reverted; the accepted path remains padded
`D56 -> D64` with `DV_DST=56` direct output.

### 2026-05-15 rejected MMQ stream-k force-tiling

The q8_0 MMQ hotspot was also checked for a possible stream-k fixup overhead
shortcut. This was not the already-rejected "disable stream-k" test. Instead,
the experiment kept the stream-k kernel path but forced the launch to use one
block per output tile, which avoids the generic stream-k fixup whenever the
normal heuristic would choose fewer blocks.

It failed the full-mask parity gate before speed measurement:

| Experiment | Evidence | Result |
| --- | --- | --- |
| `GGML_CUDA_MMQ_FORCE_STREAM_K_TILING=1` q8_0/Base+/1024 | `outputs/mmq-force-streamk-tiling-parity-q8_0-1024-20260515/summary.json` | `0 / 10` mask hashes equal, max bbox delta `30 px`, max score delta `0.016325` |

The forced-tiling launch changes reduction/order enough to move masks and
bboxes, so it was reverted. The q8_0 MMQ speed target remains a true kernel-body
or arithmetic improvement; launch-shape changes that alter stream-k reduction
order are not acceptable for the current parity contract.

### 2026-05-15 environment recheck and rejected FATTN56 vec2 pack

After regaining elevated host permissions, the runtime environment was checked
again before further CUDA experiments. `nvidia-powerd` was active, Torch still
imported as `2.8.0+cu128`, CUDA matmul succeeded, and a short q8_0/Base+/1024
bbox-only run stayed in the current baseline range:

| Check | Evidence | Result |
| --- | --- | --- |
| q8_0/Base+/1024 bbox-only smoke | `outputs/env-verify-20260515/q8_0_1024.log` | `66.7 ms/frame` |
| FATTN56 profile refresh | `outputs/fattn56-profile-env-verify-20260515/summary.json` | FATTN56 totals: pack `27.55 ms`, MMA `60.88 ms`, slice `0.61 ms` over the profiled run |

The profile confirmed that the direct-output slice cost is no longer material.
The largest remaining FATTN56 costs are the padded MMA body and the D56-to-D64
pack, especially the contiguous stage-0 shape `[56,64,2,1024]`, where pack can
be larger than the corresponding MMA time.

A small pack experiment replaced the contiguous scalar pack with a `float2` /
`half2` pack for same-shape contiguous Q/K/V tensors. It preserved the same
D56-to-D64 padding and did not touch the MMA math. Full-mask parity was exact,
but paired runtime did not improve:

| Experiment | Evidence | Result |
| --- | --- | --- |
| contiguous FATTN56 `float2`/`half2` pack | `outputs/fattn56-contiguous-vec2-parity-q8_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |
| 5-pair bbox-only A/B | `outputs/fattn56-contiguous-vec2-speed-q8_0-1024-20260515/summary.json` | scalar mean `64.7 ms/frame`, vec2 mean `64.7 ms/frame`, paired mean saved `0.0 ms` |

The vec2 pack patch was reverted. This suggests the current pack kernel is not
limited by scalar store width in a way that matters end-to-end. The remaining
FATTN56 work should therefore target either a correctness-safe native D56 kernel
contract or a larger graph-level change that avoids producing padded temporary
attention operands at all.

### 2026-05-15 accepted Hiera strided Q/K FlashAttention input

Nsight Systems profiling was added for the current q8_0/Base+/1024 benchmark
using `sam3_benchmark --no-isolation`, because the normal benchmark runs each
row in a subprocess and Nsight did not capture child CUDA kernels. The
no-isolation profile showed that the largest CUDA kernel bucket was not a
single matmul or attention kernel, but layout materialization:

| Profile | Evidence | `cpy_scalar` total | `cpy_scalar` launches | Total CUDA kernel time |
| --- | --- | ---: | ---: | ---: |
| baseline Hiera Q/K materialization | `outputs/nsys-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `132.6 ms` | `1677` | `589.1 ms` |
| strided Q/K FlashAttention input | `outputs/nsys-strided-qk-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `97.4 ms` | `1197` | `555.2 ms` |

The root cause is that Hiera reshaped Q/K, permuted them to
`[head_dim, tokens, heads, batch]`, then forced `ggml_cont` before calling
FlashAttention. The D56 padded-MMA FlashAttention pack path already supports
strided f32 Q/K/V inputs, so Q/K do not need to be materialized first. The graph
now leaves Q/K as borrowed strided views and lets the FATTN56 pack kernel read
their strides directly. The old behavior can be restored with
`SAM2_HIERA_DISABLE_STRIDED_QK_FATTN=1`.

This preserves full-mask output exactly against the old materialized-Q/K path:

| Model | Evidence | Result |
| --- | --- | --- |
| q8_0/Base+/1024 | `outputs/strided-qk-parity-q8_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |
| q4_0/Base+/1024 | `outputs/strided-qk-parity-q4_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |

Five paired bbox-only runs show a stable speedup:

| Model | Baseline mean | Strided Q/K mean | Paired mean saved | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+/1024 | `64.94 ms/frame` | `61.48 ms/frame` | `3.46 ms/frame` | `outputs/strided-qk-speed-q8_0-1024-20260515/summary.json` |
| q4_0/Base+/1024 | `66.98 ms/frame` | `63.40 ms/frame` | `3.58 ms/frame` | `outputs/strided-qk-speed-q4_0-1024-20260515/summary.json` |

Compared with the refreshed official PyTorch bf16 row (`42.89 ms/frame` in
`outputs/current-q8_0-speed-1024-refresh-20260515/summary.json`), the accepted
q8_0 path is still about `18.59 ms/frame` slower, or `1.43x` the official
runtime. This is meaningful progress but not the target. After this change, the
remaining top Nsight buckets are still layout copies, D56 FlashAttention,
q8_0 MMQ, broadcast ADD, and q8_1 activation quantization. The next high-impact
work should continue reducing graph materialization or replace the padded D56
attention/MMQ kernels with more native kernels; the accepted strided-Q/K change
does not by itself reach PyTorch speed.

### 2026-05-15 accepted q-stride direct K/V views

The next layout-materialization pass focused on Hiera blocks with `q_stride`.
Those blocks must materialize Q because it is spatially max-pooled before
attention, but K and V are only sliced from the contiguous QKV projection and
then consumed by FlashAttention. They do not need to be copied into standalone
contiguous `[C, tokens, windows]` buffers first.

The graph now keeps K/V as `ggml_view_4d` slices of QKV in q-stride blocks and
passes the strided views through the same FlashAttention path. The old K/V
materialization can be restored independently with
`SAM2_HIERA_DISABLE_QSTRIDE_DIRECT_KV=1`; disabling
`SAM2_HIERA_DISABLE_STRIDED_QK_FATTN=1` still restores the older fully
materialized Q/K behavior.

Full-mask parity is exact against the same-binary materialized-K/V path:

| Model | Evidence | Result |
| --- | --- | --- |
| q8_0/Base+/1024 | `outputs/qstride-direct-kv-only-parity-q8_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |
| q4_0/Base+/1024 | `outputs/qstride-direct-kv-only-parity-q4_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |

Five paired bbox-only runs show another small but repeatable speedup:

| Model | Materialized K/V mean | Direct K/V mean | Paired mean saved | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+/1024 | `61.62 ms/frame` | `60.44 ms/frame` | `1.18 ms/frame` | `outputs/qstride-direct-kv-speed-q8_0-1024-20260515/summary.json` |
| q4_0/Base+/1024 | `63.42 ms/frame` | `62.16 ms/frame` | `1.26 ms/frame` | `outputs/qstride-direct-kv-speed-q4_0-1024-20260515/summary.json` |

Nsight confirms that the improvement comes from fewer layout copies rather than
attention math changing:

| Profile | Evidence | `cpy_scalar` total | `cpy_scalar` launches | Total CUDA kernel time |
| --- | --- | ---: | ---: | ---: |
| strided Q/K before direct K/V | `outputs/nsys-strided-qk-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `97.4 ms` | `1177` | `555.2 ms` |
| q-stride direct K/V | `outputs/nsys-qstride-direct-kv-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `84.7 ms` | `1117` | `543.3 ms` |

A generic CUDA copy block-size experiment was also tested and rejected. Changing
`CUDA_CPY_BLOCK_SIZE` from `64` to `256` preserved q8_0 full-mask parity, but
the 5-run q8_0/Base+/1024 mean was `61.78 ms/frame` versus the accepted
strided-Q/K baseline `61.48 ms/frame`
(`outputs/cpy-block256-speed-q8_0-1024-20260515/summary.json`). The bottleneck is
therefore not fixed by simply increasing the generic copy kernel's block size;
the better direction is removing avoidable materialization or adding
shape-specific layout kernels.

### 2026-05-15 accepted q-stride direct Q-pool input

The q-stride blocks had one more avoidable copy on the Q path. Q still has to be
materialized for max-pooling, but it does not need to be first copied as a
contiguous `[C, tokens, windows]` slice and then copied again inside
`sam2_maxpool_2d`. The graph now creates the pre-pool Q tensor as a direct
`ggml_view_4d` over QKV with `[C, W, H, windows]` strides. The old path can be
restored with `SAM2_HIERA_DISABLE_QSTRIDE_DIRECT_Q_POOL=1`.

Full-mask parity is exact against the same-binary materialized-Q-pool input:

| Model | Evidence | Result |
| --- | --- | --- |
| q8_0/Base+/1024 | `outputs/qstride-direct-qpool-parity-q8_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |
| q4_0/Base+/1024 | `outputs/qstride-direct-qpool-parity-q4_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |

Five paired bbox-only runs show a smaller but still positive improvement:

| Model | Materialized Q-pool input mean | Direct Q-pool input mean | Paired mean saved | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+/1024 | `60.18 ms/frame` | `59.64 ms/frame` | `0.54 ms/frame` | `outputs/qstride-direct-qpool-speed-q8_0-1024-20260515/summary.json` |
| q4_0/Base+/1024 | `62.30 ms/frame` | `61.70 ms/frame` | `0.60 ms/frame` | `outputs/qstride-direct-qpool-speed-q4_0-1024-20260515/summary.json` |

The cumulative Nsight copy trend after the accepted layout changes is:

| Profile | Evidence | `cpy_scalar` total | `cpy_scalar` launches | Total CUDA kernel time |
| --- | --- | ---: | ---: | ---: |
| strided Q/K only | `outputs/nsys-strided-qk-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `97.4 ms` | `1177` | `555.2 ms` |
| + direct K/V | `outputs/nsys-qstride-direct-kv-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `84.7 ms` | `1117` | `543.3 ms` |
| + direct Q-pool input | `outputs/nsys-qstride-direct-qpool-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `78.7 ms` | `1087` | `537.4 ms` |

With both q-stride direct-view changes enabled, the refreshed q8_0/Base+/1024
mean is `59.64 ms/frame`. Against the latest official PyTorch bf16 baseline
(`42.89 ms/frame` in
`outputs/current-q8_0-speed-1024-refresh-20260515/summary.json`), C++ is still
about `16.75 ms/frame` slower, or `1.39x` the official runtime. The remaining
work is still large enough that future improvements should target either the
remaining layout-copy groups with shape-specific kernels, the D56 padded
FlashAttention body/pack, or q8_0 MMQ body time.

### 2026-05-15 accepted CUDA copy-to-contiguous fastpath

After the q-stride graph changes, the largest remaining `CONT` nodes still
copied non-contiguous f32 tensors into contiguous f32 destinations. The generic
CUDA `cpy_scalar` kernel computed both the source offset and destination offset
for every element, even when the destination was known to be contiguous. A
specialized f32-to-f32 `cpy_scalar_to_contiguous` path now keeps the same source
indexing but writes to `dst[i]` directly. This is gated by
`GGML_CUDA_DISABLE_CPY_TO_CONTIGUOUS_FASTPATH=1`.

Full-mask parity is exact against the same-binary generic-copy path:

| Model | Evidence | Result |
| --- | --- | --- |
| q8_0/Base+/1024 | `outputs/cpy-to-contig-fastpath-parity-q8_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |
| q4_0/Base+/1024 | `outputs/cpy-to-contig-fastpath-parity-q4_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |

Five paired bbox-only runs show a larger copy-side gain than the prior graph
cleanup:

| Model | Generic copy mean | Copy-to-contiguous fastpath mean | Paired mean saved | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+/1024 | `59.88 ms/frame` | `58.24 ms/frame` | `1.64 ms/frame` | `outputs/cpy-to-contig-fastpath-speed-q8_0-1024-20260515/summary.json` |
| q4_0/Base+/1024 | `61.84 ms/frame` | `59.92 ms/frame` | `1.92 ms/frame` | `outputs/cpy-to-contig-fastpath-speed-q4_0-1024-20260515/summary.json` |

Nsight confirms this is a direct reduction in the copy bucket:

| Profile | Evidence | Copy kernel bucket | Copy launches | Total CUDA kernel time |
| --- | --- | ---: | ---: | ---: |
| before fastpath | `outputs/nsys-qstride-direct-qpool-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `78.7 ms` generic `cpy_scalar` | `1087` | `537.4 ms` |
| copy-to-contiguous fastpath | `outputs/nsys-cpy-to-contig-fastpath-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `61.2 ms` `cpy_scalar_to_contiguous` | `1087` | `521.5 ms` |

The refreshed q8_0/Base+/1024 mean after this accepted fastpath is
`58.24 ms/frame`. Compared with official PyTorch bf16 at `42.89 ms/frame`, the
C++ path is still about `15.35 ms/frame` slower, or `1.36x` the official
runtime. The speed priority remains: continue reducing layout materialization
where possible, then target D56 FlashAttention and q8_0 MMQ body time.

### 2026-05-15 accepted row-wise copy-to-contiguous fastpath

The element-wise copy-to-contiguous fastpath still computed the source 4D index
for every element. The largest remaining copies have a wide contiguous `ne00`
dimension, so they can be copied as rows: one CUDA block handles one contiguous
destination row, computes the outer source indices once per row, and each thread
walks along `ne00`. This row-wise path is used for f32-to-f32 contiguous
destinations when `ne00 >= 32`. It can be disabled independently with
`GGML_CUDA_DISABLE_CPY_TO_CONTIGUOUS_ROW_FASTPATH=1`.

Full-mask parity remains exact:

| Model | Evidence | Result |
| --- | --- | --- |
| q8_0/Base+/1024 | `outputs/cpy-to-contig-row-fastpath-parity-q8_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |
| q4_0/Base+/1024 | `outputs/cpy-to-contig-row-fastpath-parity-q4_0-1024-20260515/compare/summary.json` | `10 / 10` mask hashes equal, bbox and score delta zero |

Five paired bbox-only runs against the element-wise fastpath show another
stable gain:

| Model | Element fastpath mean | Row fastpath mean | Paired mean saved | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+/1024 | `58.38 ms/frame` | `57.12 ms/frame` | `1.26 ms/frame` | `outputs/cpy-to-contig-row-fastpath-paired-q8_0-1024-20260515/summary.json` |
| q4_0/Base+/1024 | `60.18 ms/frame` | `58.40 ms/frame` | `1.78 ms/frame` | `outputs/cpy-to-contig-row-fastpath-paired-q4_0-1024-20260515/summary.json` |

Nsight confirms that row-wise indexing reduces the copy bucket again:

| Profile | Evidence | Copy kernel bucket | Copy launches | Total CUDA kernel time |
| --- | --- | ---: | ---: | ---: |
| element fastpath | `outputs/nsys-cpy-to-contig-fastpath-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `61.2 ms` element copy | `1087` | `521.5 ms` |
| row fastpath | `outputs/nsys-cpy-to-contig-row-fastpath-q8_0-1024-no-isolation-20260515/kernel_sum_cuda_gpu_kern_sum.csv` | `40.5 ms` row copy + `7.5 ms` element fallback | `1087` | `504.7 ms` |

The latest accepted q8_0/Base+/1024 mean is `57.12 ms/frame`. Compared with
official PyTorch bf16 at `42.89 ms/frame`, C++ is still about
`14.23 ms/frame` slower, or `1.33x` the official runtime. Layout-copy work is
now meaningfully smaller but still visible; the remaining top profile buckets
are D56 FlashAttention, broadcast/add elementwise work, q8_0 MMQ, activation
quantization, and the residual copy bucket.

The matched official-PyTorch speed row was refreshed after the row fastpath:

| Model | C++ track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio | Comparable | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| q8_0/Base+/1024 | `57.8` | `42.91` | `0.742` | yes: same decoded resolution, frames, prompt, and image size | `outputs/model-matrix-row-fastpath-q8_0-1024-20260515/summary.json` |

This confirms the latest CUDA path is materially faster than the previous
baseline, but still not faster than official PyTorch on the matched 1024 Base+
comparison.

A row-copy block-size retune was checked and rejected. Increasing the row
fastpath launch from `64` to `256` threads preserved q8_0 full-mask parity, but
the 5-run q8_0/Base+/1024 mean regressed from the accepted `57.12 ms/frame` row
fastpath baseline to `57.62 ms/frame`
(`outputs/cpy-row-block256-speed-q8_0-1024-20260515/summary.json`). The accepted
row fastpath therefore keeps the same `CUDA_CPY_BLOCK_SIZE=64` launch shape as
the generic copy path.

The FATTN56 profile was refreshed after the row-copy work:

| Metric | Value | Evidence |
| --- | ---: | --- |
| FATTN56 profile rows | `240` | `outputs/fattn56-profile-row-fastpath-q8_0-1024-20260515/summary.json` |
| Pack total | `26.31 ms` | same |
| MMA total | `62.24 ms` | same |
| Slice total | `0.61 ms` | same |
| Total profiled FATTN56 time | `89.17 ms` | same |

Top FATTN56 groups after row-copy optimization:

| Q/K/V shape | Count | Pack | MMA | Total |
| --- | ---: | ---: | ---: | ---: |
| `[56,4096,8,1]` | `30` | `1.06 ms` | `25.84 ms` | `26.97 ms` |
| `[56,196,8,25]` | `120` | `5.73 ms` | `18.99 ms` | `25.02 ms` |
| `[56,64,2,1024]` | `20` | `9.10 ms` | `6.93 ms` | `16.08 ms` |

This keeps D56 FlashAttention as the next root target, but the profile also
shows that the remaining work is not one uniform problem: global attention is
mostly padded-MMA body time, window attention is split between body and pack,
and the stage0 shape is pack-heavy. The previously rejected native V56 and vec2
pack experiments should not be repeated without a different kernel contract.

The Base+ q8_0 quality fallback was refreshed after the row-copy speed work.
Using the default single-mask initial prompt is not an acceptable quality
fallback: even with the official side fed extracted frames, the mean mask IoU is
only `0.7240` and propagation candidate selection matches `4 / 9` rows
(`outputs/official-quality-row-fastpath-q8_0-1024-frames-20260515/official/summary.json`).

Using the intended multimask initial prompt restores the prior quality level:

| Check | Result | Evidence |
| --- | ---: | --- |
| Frames compared | `10` | `outputs/official-quality-row-fastpath-q8_0-1024-multimask-20260515/official/summary.json` |
| Mean mask IoU vs official PyTorch bf16 | `0.9880` | same |
| Min mask IoU vs official PyTorch bf16 | `0.9869` | same |
| Min bbox IoU | `0.8719` | same |
| Propagation candidate selection | `5 / 9` matching frames | same |
| Initial candidate count and selected index | matched | same |

The comparison contract is strict for this artifact: same decoded source frame
resolution (`960x540`), same 10-frame range, same point prompt `(315,250)`, same
SAM image size (`1024`), and official PyTorch receives the extracted frames
(`python_video_source=frames`). This confirms the row-copy CUDA speed changes did
not regress the q8_0/Base+/1024 quality fallback, but it also makes the fallback
requirement explicit: initial point prompts must use multimask mode for quality
parity-style validation.

### 2026-05-15 environment revalidation and rejected plane-broadcast ADD fastpath

After logging back in with elevated permissions, the local CUDA environment was
revalidated before continuing optimization:

| Check | Result |
| --- | --- |
| `nvidia-smi` | RTX 5070 Ti Laptop GPU visible, driver `595.58.03`, CUDA `13.2`, 120 W cap |
| `nvidia-powerd` / `nvidia-persistenced` | both active |
| `just build` | passed |
| PyTorch CUDA smoke test | `torch 2.8.0+cu128`, CUDA available, matmul completed |
| Matched q8_0/Base+/1024 speed row | C++ `59.1 ms/frame`, official PyTorch bf16 `42.50 ms/frame` |

A narrow CUDA `ADD` experiment was then tested and rejected. The node profile
showed the largest `ADD` row as `dst=[256,256,112,1]` with
`src1=[1,1,112,1]`, so an env-gated plane-broadcast f32 kernel was tried for
that exact contiguous layout. Full-mask parity was exact against the disabled
path:

| Check | Result | Evidence |
| --- | ---: | --- |
| q8_0/Base+/1024 mask hash equality | `10 / 10` | `outputs/bin-bcast-plane-fastpath-parity-q8_0-1024-20260515/compare.json` |
| Max bbox delta | `0 px` | same |
| Max score delta | `0` | same |

Five bbox-only timing runs did not justify keeping it:

| Mode | Runs | Mean | Evidence |
| --- | --- | ---: | --- |
| existing axis/generic path | `58, 56, 56, 56, 57` | `56.6 ms/frame` | `outputs/bin-bcast-plane-fastpath-speed-q8_0-1024-20260515/summary.json` |
| plane-broadcast fastpath | `57, 57, 56, 57, 57` | `56.8 ms/frame` | same |

The experiment was reverted. This should not be retried as a standalone
plane-broadcast `ADD` kernel unless it is part of a broader fusion that removes
a kernel launch or a following materialization.

### 2026-05-15 current precision matrix and D64 MMA retune boundary

The Base+ precision matrix was refreshed under the same comparable 1024
contract: decoded source resolution `960x540`, 10 frames, point prompt
`(315,250)`, and official PyTorch fed the extracted frames.

| Precision | C++ track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0 | `56.7` | `42.63` | `0.752` | `outputs/model-matrix-current-baseplus-precisions-1024-20260515/summary.json` |
| q4_0 | `57.7` | `42.63` | `0.739` | same |
| q4_1 | `58.8` | `42.63` | `0.725` | same |
| mxfp4 | `59.2` | `42.63` | `0.720` | same |
| nvfp4 | `66.3` | `42.63` | `0.643` | same |
| f16 | `66.9` | `42.63` | `0.637` | same |
| f32 | `70.2` | `42.63` | `0.607` | same |

This keeps q8_0 as the current best speed row. Choosing f16/f32 to match the
official precision family does not improve runtime on this CUDA path.

The node profile was also refreshed:

| Hotspot | Drop-max total | Count | Evidence |
| --- | ---: | ---: | --- |
| D56 window FlashAttention `dst=f32[56,8,196,25]` | `6.84 ms` | `36` | `outputs/node-profile-current-q8_0-1024-20260515/hotspots-all.json` |
| D56 global FlashAttention `dst=f32[56,8,4096,1]` | `6.48 ms` | `9` | same |
| q8_0 MLP projection `1792 -> 448` | `5.79 ms` | `48` | same |
| q8_0 MLP expansion `448 -> 1792` | `5.67 ms` | `48` | same |
| q8_0 QKV/proj window matmul `448 -> 1344` | `4.70 ms` | `36` | same |

Two simple D64 padded-MMA retunes were checked because D56 currently routes
through the D64 MMA kernel with `DV_DST=56`. Both were rejected before runtime
measurement because they violate compile-time tiling invariants:

| Experiment | Result |
| --- | --- |
| D64 config `nthreads=256, occupancy=1` | compile failed: zero-size tile arrays and `bad loop size` assertions |
| D64 config `nthreads=64, occupancy=4` | compile failed: `bad nwarps` and division-by-zero static paths |

The D64 padded-MMA path is therefore not safely tunable by only changing thread
count or occupancy. Further D56 work needs a real kernel-contract change:
either a tail-safe native V56 accumulator/write path, or a different specialized
attention kernel for SAM2 Hiera shapes. Standalone pack retunes and standalone
ADD broadcast kernels have already failed to show end-to-end value.

A follow-up native D56/V56 MMA attempt tried to make the existing MMA kernel
tail-safe by using ceil-sized accumulators and zeroing the shared-memory tail
lanes before `ldmatrix`. This was also rejected: the enabled path produced
`0 / 10` matching mask hashes, `min_bbox_iou=0.0`, `max_bbox_delta_px=960.0`,
`max_score_abs_delta=0.255263`, and `max_abs_mask_area_rel_delta=1.0`
(`outputs/fattn56-native-mma-parity-q8_0-1024-20260515/summary.json`). The
experimental code was removed, so the supported D56 route remains the
`D56 -> D64` padded-MMA path with `DV_DST=56` direct output. A usable native D56
kernel still needs a more complete rewrite of the MMA tile contract and
writeback semantics, not only accumulator sizing. The restored default path was
then smoke-tested against itself with exact full-mask parity (`10 / 10` mask
hashes equal, zero bbox/score deltas) in
`outputs/post-native-d56-revert-parity-q8_0-1024-20260515/summary.json`.

A row/warp-based rewrite of the same-shape FATTN56 Q/K/V pack was also tested
against the existing element-indexed combined pack. It preserved exact full-mask
parity (`10 / 10` mask hashes equal) but did not improve end-to-end speed:
old-pack and row-pack 5-run means were both `62.46 ms/frame`, with paired mean
saved `0.0 ms` and stdev `0.67 ms`
(`outputs/fattn56-row-pack-parity-q8_0-1024-20260515/summary.json`,
`outputs/fattn56-row-pack-speed-q8_0-1024-20260515/summary.json`). The change
was removed. This reinforces that the remaining FATTN56 work is the MMA body and
the broader graph/materialization cost, not another standalone pack rewrite.

The 512 q8_0 matched row was refreshed as well:

| Model | C++ track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+/512 | `16.2` | `10.16` | `0.627` | `outputs/model-matrix-current-q8_0-512-20260515/summary.json` |

Multimask fallback quality at 512 is materially worse than the 1024 row even
though candidate selection matches every propagation frame:

| Check | Result | Evidence |
| --- | ---: | --- |
| Mean mask IoU vs official PyTorch bf16 | `0.9323` | `outputs/official-quality-current-q8_0-512-multimask-20260515/official/summary.json` |
| Min mask IoU vs official PyTorch bf16 | `0.9209` | same |
| Min bbox IoU | `0.9535` | same |
| Propagation candidate selection | `9 / 9` matching frames | same |
| Initial candidate count and selected index | matched | same |

The same 512 quality check with the f16 C++ model improves mean mask IoU only
slightly (`0.9425`) and still does not meet parity-style thresholds
(`outputs/official-quality-current-f16-512-multimask-20260515/official/summary.json`).
This suggests the 512 quality gap is not only q8_0 quantization error. The
first-frame box is already offset at f16, so the remaining fallback-quality work
should inspect the 512 initial prompt path and coordinate/mask upsampling
contract before treating q8_0 quantization as the root cause.

The current goal audit is therefore still `not_complete`:
`outputs/goal-audit-current-q8_0-20260515/summary.json`. The matched official
comparisons and Hiera/profile attribution exist, but C++ is not faster than
official PyTorch at either 512 or 1024, and Base+ fallback quality does not meet
the stricter parity threshold in the audit.

### 2026-05-15 frame-source contract check for 512 quality

The quality harness was tightened so C++ can consume the same extracted frame
directory as official SAM2. `sam3_benchmark` now accepts `--frame-dir`, and
`scripts/model_matrix_compare.py` passes its extracted JPEG frames to C++ as
well as to official PyTorch. This removes a previously ambiguous comparison
contract where C++ decoded raw RGB from the video while the official SAM2 path
read JPEG frames.

Retesting showed that this was not the root cause of the 512 fallback-quality
gap:

| Model | Encode size | Frame source | Mean mask IoU | Min mask IoU | Candidate selection | Evidence |
| --- | ---: | --- | ---: | ---: | --- | --- |
| q8_0/Base+ | 512 | shared extracted JPEG frames | `0.8603` | `0.8459` | propagation `9 / 9`, initial selected index matched | `outputs/official-quality-frame-dir-q8_0-512-multimask-20260515/summary.json` |
| f16/Base+ | 512 | shared extracted JPEG frames | `0.9228` | `0.9073` | propagation `9 / 9`, initial selected index matched | `outputs/official-quality-frame-dir-f16-512-multimask-20260515/summary.json` |

The matched speed row through the same `--frame-dir` path is unchanged within
noise:

| Model | C++ track ms/frame | Official PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+/512 | `16.2` | `10.14` | `0.626` | `outputs/model-matrix-frame-dir-q8_0-512-20260515/summary.json` |

The f16 row has a near-perfect first-frame bbox (`bbox IoU 0.9952`), but its
propagated masks remain smaller than official PyTorch. The q8_0 row is
materially worse than f16 under the same input frames, so 512 has two separate
issues:

1. q8_0 quantization has a real quality cost at 512.
2. f16 still does not reach parity, so there is also a dynamic-image-size or
   tracking-memory contract difference to inspect.

The next quality work should compare 512 f16 propagation internals against
official PyTorch, especially memory encoding, memory attention, and high-res
mask resizing. The earlier suspicion that raw-video-vs-JPEG frame sourcing was
the main problem is now ruled out.

The shared-frame 512 artifacts were further summarized into
`quality-gap-diagnostic.json` files for both q8_0 and f16. In both precisions,
the initial selected multimask candidate matches official PyTorch (`0`), so the
512 gap is not a candidate-selection bug:

| Model | Initial C++ area | Initial official area | Propagation pattern | Worst frame | Evidence |
| --- | ---: | ---: | --- | --- | --- |
| q8_0/Base+ | `100849` | `116753` | C++ mask smaller on every propagation frame | offset `2`, IoU `0.8459`, area ratio `0.8581` | `outputs/official-quality-frame-dir-q8_0-512-multimask-20260515/quality-gap-diagnostic.json` |
| f16/Base+ | `115301` | `116753` | C++ mask smaller on every propagation frame | offset `9`, IoU `0.9073`, area ratio `0.9211` | `outputs/official-quality-frame-dir-f16-512-multimask-20260515/quality-gap-diagnostic.json` |

The q8_0 initial mask is already smaller than official, while f16 starts much
closer but still shrinks during propagation. That points to two layers of work:
q8_0 quantization affects low-resolution mask extent at 512, and the remaining
f16 gap should be debugged in the tracking-memory, memory-attention, high-res
mask-resize, or threshold contract rather than in frame ordering or multimask
selection.

### 2026-05-15 SAM2 conditioning-memory contract fix

Official SAM2 does not encode the initial point-derived memory from the
video-resolution binary mask. It keeps the selected decoder mask logits and, for
point-derived conditioning frames, binarizes the high-resolution logits before
the memory encoder. The C++ tracker was instead resizing the already-binarized
output mask back down into synthetic logits (`+6/-6`) and then applying the
sigmoid memory-mask path. That contract mismatch made propagated masks
systematically smaller, especially at 512.

The tracker now carries the selected decoder logits in `sam3_detection` and
uses them when adding an instance. For SAM2 conditioning memory, the resized
logits are binarized before applying the memory encoder scale/bias. The old
sigmoid path remains available with `SAM2_DISABLE_BINARIZE_COND_MEM=1` for A/B
checks and for non-conditioning memory updates.

Quality improved substantially under the shared-frame official PyTorch
comparison:

| Model | Encode size | Before mean/min mask IoU | After mean/min mask IoU | Candidate selection | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| f16/Base+ | 512 | `0.9228 / 0.9073` | `0.9660 / 0.9584` | propagation `9 / 9`, initial selected index matched | `outputs/official-quality-frame-dir-f16-512-raw-condmem-20260515/summary.json` |
| q8_0/Base+ | 512 | `0.8603 / 0.8459` | `0.9037 / 0.8276` | propagation `9 / 9`, initial selected index matched | `outputs/official-quality-frame-dir-q8_0-512-raw-condmem-20260515/summary.json` |
| q8_0/Base+ | 1024 | `0.9880 / 0.9869` | `0.9951 / 0.9933` | propagation `4 / 9`; mismatches are near-ties with mask IoU `>= 0.9933` | `outputs/official-quality-frame-dir-q8_0-1024-raw-condmem-20260515/summary.json` |

The 1024 q8_0 row now satisfies the audit's quality-parity threshold. The 512
q8_0 row improves on average but still fails parity because the first-frame q8_0
mask is already too small (`0.8276` IoU on offset 0). That remaining 512 gap is
therefore primarily q8_0 quantization and low-resolution mask extent, not frame
ordering, candidate selection, or the initial memory-conditioning contract.

Current matched speed rows after the fix:

| Model | Encode size | C++ track ms/frame | Official PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| q8_0/Base+ | 512 | `15.0` | `10.18` | `0.679` | `outputs/model-matrix-q8_0-512-raw-condmem-20260515/summary.json` |
| q8_0/Base+ | 1024 | `56.2` | `42.54` | `0.757` | `outputs/model-matrix-q8_0-1024-raw-condmem-20260515/summary.json` |

The refreshed goal audit is still `not_complete` because C++ is not yet faster
than official PyTorch on matched rows, and 512 q8_0 fallback quality is still
below parity:
`outputs/goal-audit-raw-condmem-q8_0-20260515/summary.json`.

### 2026-05-15 SAM2 FPN 1x1 mul_mat rewrite

SAM2's FPN neck used `conv_2d_sk_p0` for the lateral 1x1 projections. For 1x1
convolutions this pays avoidable IM2COL/layout cost. The FPN path now reshapes
`[C, W, H, 1]` stage tensors to `[C, W*H]`, applies the same projection with
`ggml_mul_mat`, adds the channel bias, and reshapes back to `[D, W, H, 1]`.
The old path is available with `SAM2_DISABLE_FPN_1X1_MULMAT=1`.

This change is exact at 1024 against the old FPN conv path and beneficial at
512, where the q8_0 output had previously been too small versus official SAM2:

| Check | Result | Evidence |
| --- | --- | --- |
| q8_0/Base+/1024 old-vs-new parity | `10 / 10` mask hashes equal, zero bbox/score/area deltas | `outputs/fpn-1x1-mulmat-parity-q8_0-1024-20260515/summary.json` |
| q8_0/Base+/512 old-vs-new parity | not bit-identical; masks become larger | `outputs/fpn-1x1-mulmat-parity-q8_0-512-20260515/summary.json` |
| q8_0/Base+/512 quality vs official | mean/min mask IoU `0.9518 / 0.9377`, up from `0.9037 / 0.8276` after the conditioning-memory fix | `outputs/official-quality-frame-dir-q8_0-512-fpn-mulmat-20260515/summary.json` |
| q8_0/Base+/1024 quality vs official | mean/min mask IoU `0.9951 / 0.9933` | `outputs/official-quality-frame-dir-q8_0-1024-fpn-mulmat-20260515/summary.json` |

The speed A/B also favors the mul_mat path:

| Model | Encode size | New track ms/frame | Old track ms/frame | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0/Base+ | 512 | mean about `14.7` | mean about `15.2` | `outputs/fpn-1x1-mulmat-speed-q8_0-512-20260515/summary.json` |
| q8_0/Base+ | 1024 | mean about `55.1` | mean about `57.0` | `outputs/fpn-1x1-mulmat-speed-q8_0-1024-20260515/summary.json` |

Refreshed matched official-PyTorch rows:

| Model | Encode size | C++ track ms/frame | Official PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| q8_0/Base+ | 512 | `14.7` | `10.19` | `0.693` | `outputs/model-matrix-q8_0-512-fpn-mulmat-20260515/summary.json` |
| q8_0/Base+ | 1024 | `54.1` | `42.72` | `0.790` | `outputs/model-matrix-q8_0-1024-fpn-mulmat-20260515/summary.json` |

The refreshed node profile after the FPN rewrite still points at D56
FlashAttention first, followed by quantized Hiera MLP matmuls:
`outputs/node-profile-fpn-mulmat-q8_0-1024-20260515/hotspots-all.json`.
The current goal audit remains `not_complete` because C++ is still slower than
official PyTorch and 512 quality remains below the strict parity threshold:
`outputs/goal-audit-fpn-mulmat-q8_0-20260515/summary.json`.

One MMQ launch heuristic was also rejected in this pass. Routing only low-tile
q8_0 MMQ shapes away from stream-k avoided fixup launches but slowed the
matched 1024 run: the new heuristic averaged about `58.5 ms/frame` versus
`57.6 ms/frame` for the old stream-k selection
(`outputs/mmq-lowtile-heuristic-ab-q8_0-1024-20260515/summary.json`). The
experimental code was removed.

### 2026-05-15 FATTN56 and precision refresh after FPN rewrite

The post-FPN FATTN56 event profile was refreshed with
`GGML_CUDA_PROFILE_FATTN56=1`:
`outputs/fattn56-profile-fpn-mulmat-q8_0-1024-20260515/summary.json`. The
first row is treated as first-use overhead and ignored in the shape totals.

| Shape | Count | Total ms | Pack ms | MMA ms | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `Q=[56,4096,8,1] K=[56,4096,8,1]` | `15` | `13.89` | `0.53` | `13.32` | global D56 attention is MMA-body limited |
| `Q=[56,196,8,25] K=[56,196,8,25]` | `60` | `12.67` | `2.91` | `9.61` | window D56 attention is still mainly MMA-body limited |
| `Q=[56,64,2,1024] K=[56,64,2,1024]` | `9` | `5.76` | `2.56` | `3.18` | early-stage same-shape pack is visible but not dominant |
| `Q=[56,16,4,1024] K=[56,64,4,1024]` | `5` | `3.51` | `2.09` | `1.40` | mixed Q-stride pack is expensive but too small to close the gap alone |
| `Q=[56,16,4,1024] K=[56,16,4,1024]` | `10` | `3.32` | `1.54` | `1.76` | same-shape smaller attention is split between pack and MMA |

This confirms that the remaining high-impact work is not another narrow
packing tweak. The global and main window rows need a real D56 attention-kernel
improvement, or an equivalent MMA execution shape improvement, to materially
close the PyTorch gap.

Precision was also rechecked under the matched 1024 official-PyTorch contract.
q8_0 remains the fastest current C++ precision; f16 and q4 do not provide a
shortcut to beating PyTorch:

| Precision | C++ track ms/frame | Official PyTorch bf16 ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| q8_0 | `54.1` | `42.72` | `0.790` | `outputs/model-matrix-q8_0-1024-fpn-mulmat-20260515/summary.json` |
| f16 | `65.3` | `42.38` | `0.649` | `outputs/model-matrix-f16-1024-current-with-python2-20260515/summary.json` |
| q4_0 | `56.5` | `42.82` | `0.758` | `outputs/model-matrix-q4_0-1024-current-with-python-20260515/summary.json` |
| q4_1 | `56.7` | `42.57` | `0.751` | `outputs/model-matrix-q4_1-1024-current-with-python-20260515/summary.json` |

The environment check during this pass found the CUDA toolchain usable
(`just build` passed), persistence mode enabled, and enough disk space. The
measurement risk is therefore ordinary laptop-GPU clock variance, not a broken
build or missing CUDA runtime.

A Q-prescale variant for the D56 wrapper was rejected. The idea was to multiply
the FlashAttention scale into Q during D56-to-D64 packing and then pass scale
`1.0` to the existing MMA kernel. That changes rounding order before the MMA
path and broke parity: `0 / 10` mask hashes matched, max bbox delta was `12 px`,
and max relative mask-area delta was `6.66%`
(`outputs/fattn56-prescale-q-parity-q8_0-1024-20260515/summary.json`). The
experimental code was removed; future Q-side changes need to preserve the
existing half-rounding order inside the MMA tile load.

The q8_0 MMQ launch profile was refreshed as the secondary speed target:
`outputs/mmq-profile-fpn-mulmat-q8_0-1024-20260515-cont/summary.json`. The
main Hiera MLP rows are already high-efficiency, no-fixup launches:

| Shape | Count | MMQ tile | Fixup | Efficiency | Interpretation |
| --- | ---: | --- | ---: | ---: | --- |
| `ncols_x=448,nrows_x=1792,ncols_dst=4096` | `80` | `128x128` | `0` | `97` | stage-2/global MLP expansion is already well-tiled |
| `ncols_x=1792,nrows_x=448,ncols_dst=4096` | `80` | `128x128` | `0` | `92` | stage-2/global MLP projection is already well-tiled |
| `ncols_x=448,nrows_x=1344,ncols_dst=196,ch=25` | `60` | `112x128` | `0` | `99` | window MLP expansion is well-tiled |
| `ncols_x=448,nrows_x=448,ncols_dst=196,ch=25` | `60` | `112x128` | `1` | `86` | window projection has fixup, but this class was not helped by coarse stream-k heuristics |

This keeps the speed priority unchanged: D56 FlashAttention body work remains
first. q8_0 MMQ can still matter, but the next useful MMQ change should be a
body-level improvement for the dominant high-efficiency MLP shapes, not another
stream-k/fixup routing heuristic.

### 2026-05-15 q8_0 MMQ bias fusion

A narrow MMQ fusion was added for the `MUL_MAT -> ADD(bias)` pattern used by
q8_0 Hiera MLP rows. The fused path adds the channel bias in the MMQ writeback
and skips the standalone broadcast ADD node. It is intentionally constrained to
q8_0 and to stream-k schedules whose fixup behavior can be predicted as exact.
The fallback can be forced with `GGML_CUDA_DISABLE_MMQ_BIAS_FUSION=1`.

The stricter schedule guard is required. A broader first attempt that allowed
stream-k fixup cases changed scores and one mask hash; restricting the fusion to
exact schedules restored bitwise parity:

| Precision | Encode size | Result | Evidence |
| --- | ---: | --- | --- |
| q8_0/Base+ | 1024 | `10 / 10` mask hashes, zero bbox/score/area delta | `outputs/mmq-bias-fusion-parity-q8_0-1024-20260515/summary-q8-only.json` |
| q4_0/Base+ | 1024 | exact when tested, but not enabled by default because single-run speed regressed | `outputs/mmq-bias-fusion-parity-q4_0-1024-20260515/summary.json` |

The accepted q8_0 A/B used five sequential runs on the matched 1024 frame-dir
contract. Mean tracking time improved from `55.10 ms/frame` to
`54.44 ms/frame`, a `0.66 ms/frame` or `1.20%` reduction
(`outputs/mmq-bias-fusion-speed-q8_0-1024-20260515/summary.json`). This is a
small but measurable cleanup of the secondary MMQ hotspot; it does not change
the main conclusion that D56 FlashAttention body time remains the first-order
gap versus official PyTorch.

The official-PyTorch comparison was refreshed after the q8_0 fusion with the
matched 1024 contract and an explicit SAM2 Base+ checkpoint. C++ improved
slightly but remains slower than PyTorch: `53.4 ms/frame` versus
`42.88 ms/frame`, PyTorch/C++ ratio `0.803`
(`outputs/model-matrix-q8_0-1024-mmq-bias-fusion-with-python-20260515/summary.json`).

Two follow-up D56 checks were rejected:

| Experiment | Result | Evidence |
| --- | --- | --- |
| Disable the D56 pad-MMA path and fall back to the generic attention kernel | Slower by `13.4 ms/frame` on 3-run q8_0/Base+/1024 mean, and not exact (`9 / 10` mask hashes) | `outputs/fattn56-pad-vs-fallback-q8_0-1024-20260515/summary.json`, `outputs/fattn56-pad-vs-fallback-q8_0-1024-20260515/parity-summary.json` |
| Blackwell-only `DKQ=64,DV=64,ncols=64` MMA config with `nbatch_fa=128` | Built, but parity failed (`9 / 10` mask hashes, max bbox delta `135 px`) | `outputs/fattn64-bw-nbatch128-q8_0-1024-20260515/parity-summary.json` |
| Broaden q8_0 MMQ bias fusion to stream-k fixup shapes by adding bias to the final K segment | Built, but not exact (`9 / 10` mask hashes, max score delta `0.003649`) | `outputs/mmq-bias-fusion-final-segment-q8_0-1024-20260515/parity-summary.json` |
| Broaden q8_0 MMQ bias fusion by adding bias in the stream-k fixup kernel after partial accumulation | Built, but wrong (`min bbox IoU 0.185`, max bbox delta `365 px`) | `outputs/mmq-bias-fusion-fixup-bias-q8_0-1024-20260515/parity-summary.json` |
| Force bias-bearing q8_0 MMQ launches to no-fixup tiling with `GGML_CUDA_MMQ_BIAS_FUSION_FORCE_TILING=1` | Built, but parity failed (`9 / 10` mask hashes, max bbox delta `135 px`) | `outputs/mmq-bias-force-tiling-q8_0-1024-20260515/parity-summary.json` |
| Fuse non-quantized MMF `MUL_MAT -> ADD(channel bias)` by adding bias in `mul_mat_f` writeback | Parity was exact, but 5-pair q8_0/Base+/1024 speed did not improve (`54.8 ms/frame` enabled vs `54.6 ms/frame` disabled) | `outputs/mmf-bias-fusion-parity-q8_0-1024-20260515/summary.json`, `outputs/mmf-bias-fusion-speed-q8_0-1024-20260515/summary.json` |

This narrows the viable D56 direction further: the generic fallback is not a
shortcut, and changing the existing MMA tile schedule can alter numerics enough
to move tracking. The next D56 work should either preserve the current
accumulation schedule exactly while reducing pack/write overhead, or introduce a
new kernel with its own strict parity gate rather than retuning the existing
padded kernel casually.

After removing the rejected force-tiling branch, the CUDA environment and q8_0
fusion path were rechecked. `nvidia-powerd` was active, the GPU was reported as
RTX 5070 Ti Laptop GPU with driver `595.58.03`, `just build`, `just fmt-check`,
`scripts/check_no_local_paths.sh`, and both parent/submodule `git diff --check`
passed. A C++-only matched Base+ q8_0 1024 smoke produced `54.1 ms/frame`
(`outputs/env-recheck-q8_0-1024-20260515/summary.json`). The accepted q8_0 MMQ
bias fusion still matched the disabled path exactly: `10 / 10` mask hashes,
zero bbox delta, zero score delta, and zero mask-area delta
(`outputs/env-recheck-q8-bias-parity-20260515/summary.json`).

The same matched 1024 contract was also refreshed for the f16 Base+ model using
the official SAM2 Python checkpoint. f16 C++ is slower than q8_0 and farther
from PyTorch: C++ measured `66.5 ms/frame`, PyTorch bf16 measured
`42.37 ms/frame`, with ratio `0.637`
(`outputs/model-matrix-f16-1024-with-python-20260515b/summary.json`). This means
the gap is not only q8_0 activation quantization; the CUDA Hiera execution
itself is still behind the official PyTorch path.

The current FATTN56 event profile was refreshed after the q8_0 fusion cleanup:
`240` profiled D56 rows, total pack `26.25 ms`, MMA `61.56 ms`, slice
`0.64 ms` across the 10-frame run
(`outputs/fattn56-profile-current-q8_0-1024-20260515b/summary.json`). The largest
groups remain global/window stage-2 attention, but stage-0 strided rows such as
`Q=[56,64,2,1024]` are pack-heavy (`9.17 ms` pack vs `6.88 ms` MMA). A standalone
same-shape pack rewrite was already rejected earlier with no end-to-end speed,
so the next D56 direction should avoid creating padded global temporaries at the
graph/kernel boundary, or implement a correctness-safe native D56 kernel, rather
than another isolated pack microkernel.

The Hiera gap summary was refreshed with the current q8_0 profile instead of
the older q4_0 baseline. Dropping the first cache-heavy C++ encode, current q8_0
Hiera steady encode is `41.97 ms` while official PyTorch `forward_image` is
`30.41 ms`, leaving `11.56 ms` of Hiera gap
(`outputs/hiera-gap-current-q8_0-20260515/summary.json`). This matches the
remaining end-to-end 1024 gap closely enough that propagation is no longer the
primary target: C++ `propagate_single` is `15.36 ms` and official PyTorch
`track_step` is `15.35 ms` in the synchronized attribution profile.

An additional native-V56 D56 check was rejected during the environment-repair
continuation. The experiment kept Q/K padded to 64 but cast V as a contiguous
56-wide f16 tensor and routed the call through `DKQ=64,DV=56`. The standalone
FATTN harness stayed finite and within the existing relaxed tolerance
(`max_abs=0.0442505963`, mean abs about `0.00584`), but the real q8_0/Base+/1024
model crashed during the first Hiera encode with a CUDA illegal memory access
reported on the following GELU launch. A follow-up fixed the wrapper to keep the
output tensor 56-wide instead of presenting it as 64-wide, but the model still
crashed in the same way. Evidence is in
`outputs/fattn56-native-v-experiment-20260515/` and
`outputs/fattn56-native-v2-experiment-20260515/`. The patch was removed. This
confirms that native D56 cannot be treated as a wrapper-only change; it needs
kernel-internal fixes for tile/combine bounds and graph-shape behavior before it
can be used in the model path.

The kernel-internal follow-up found and fixed one real latent assumption but did
not make native D56 usable. `VKQ_C` allocation and KQ rescale loops used floor
division for the number of value tiles, so `DV=56` with `nbatch_combine=28`
could address the fourth accumulator tile even though only three were allocated.
The code now computes this tile count with ceiling division, and the D2=28
shared-memory loader zeros the 4 half2 tail entries that ldmatrix may read.
With those fixes, the native-V56 harness no longer produced the previous huge
uninitialized tail values for the large global shape, but the representative
window shape still failed parity and the real model still crashed. Evidence is
in `outputs/fattn56-native-v-ceil-20260515/` and
`outputs/fattn56-native-v-tailzero-20260515/`. The native wrapper branch was
removed again; only the internal bounds hardening remains.

The standalone FATTN parity harness was also tightened so future native-D56
work reports the first non-finite output with decoded `d/n/head/batch`
coordinates instead of only a flat index. This matters because the old failure
message could pair the first non-finite index with the later max finite-diff
values, making the tail-lane diagnosis ambiguous. The default padded path was
rechecked after this harness-only change on the representative D56 shapes:
`N=196,heads=8,batch=25` and `N=64,heads=2,batch=1024` both remained finite and
within the existing tolerance (`outputs/fattn56-diagnostic-coords-20260515/`).

Using the coordinate-enabled harness with a temporary env-gated native-V56
wrapper showed that the first bad value is not confined to the high-D tail. Both
representative native-V56 runs produced `-inf` at `d=0,n=0,head=0,batch=0`:
`N=196,heads=8,batch=25` had `81424` non-finite checked outputs, and
`N=4096,heads=8,batch=1` had `3024` non-finite checked outputs
(`outputs/fattn56-native-coords-20260515/`). The wrapper was removed again. The
next native-D56 attempt should inspect combine metadata / `KQ_rowsum` handling
for `DV=56`; treating this as only a final tail-lane writeback issue is not
consistent with the observed first coordinate.

That diagnosis was correct: the shared-memory row stride had been rounded up,
but the combine metadata still started at `nbatch_combine` rather than at the
rounded data width. For `DV=56`, the MMA tile may touch half2 lanes `28..31`
even though only `0..27` are real output lanes, so placing max/rowsum metadata
at lane `28` corrupted the first output coordinate. Moving the metadata start to
the rounded data stride fixes the native-D56 correctness issue. With a
diagnostic `GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` path:

| Check | Result |
| --- | --- |
| FATTN `N=196,heads=8,batch=25` | `nonfinite=0`, max abs `0.0320834816` |
| FATTN `N=4096,heads=8,batch=1` | `nonfinite=0`, max abs `0.00515256729` |
| q8_0/Base+/1024 JSONL parity | `10 / 10` mask hashes equal, bbox/score delta zero |

The native-V56 path is still not a default optimization. Its profile is slower
than the padded `DV=64,DV_DST=56` path despite saving the V padding width:
representative `N=196` rows move from about `0.151 ms` MMA to about `0.194 ms`,
and the global `N=4096` row moves from about `0.83 ms` to `1.31-1.36 ms`.
Evidence is in `outputs/fattn56-native-v-real-20260515/`. The accepted default
therefore remains the padded MMA body with direct 56-wide output; the native
V56 code is kept opt-in for diagnosis only.

A follow-up attempt to make native V56 faster by aligning the D56 config to
`nbatch_V2=32` and `nbatch_combine=32` was rejected at compile time. The MMA
kernel requires `DV % (2*nbatch_V2) == 0`, so `DV=56` cannot use
`nbatch_V2=32` without a deeper loop/writeback redesign. The failed experiment
was reverted; the accepted D56 config stays at `nbatch_V2=28` and
`nbatch_combine=28`.

The fallback MMQ/cuBLAS boundary was rechecked after this because FATTN56 alone
was not yielding a speed win. A full `GGML_CUDA_FORCE_CUBLAS=ON` build was much
slower than the current MMQ path on the same q8_0/Base+/1024 frame-dir contract:
the default build measured `53.9 ms/frame`, while the force-cuBLAS build measured
`101.2 ms/frame` and was not exact (`9 / 10` mask hashes, max bbox delta
`135 px`). Evidence is in `outputs/force-cublas-ab-q8_0-1024-20260515/`.
Therefore broad cuBLAS fallback is not a viable route to PyTorch parity.

The q8_0 MMQ activation-quantization profile was refreshed in
`outputs/mmq-profile-current-20260515/run.log`. Over the profiled 3-frame run,
q8_0 MMQ accounted for `49.28 ms`, including `9.45 ms` of activation
quantization. The largest repeated shapes were the stage-2 MLP rows
`448->1792` and `1792->448`, plus the window QKV/projection rows. A trial to
replace `roundf` in the MMQ q8_1 activation quantizer with a fast intrinsic was
rejected: it changed the default behavior enough to produce an invalid first
mask (`det=0`, empty propagation) before any speed result could be accepted. The
patch was reverted, and the post-revert smoke recovered `det=1` at
`54.5 ms/frame` (`outputs/mmq-fast-round-q8_0-1024-20260515/reverted/`).
Future MMQ work should avoid changing q8_1 rounding semantics and instead target
structural savings such as reusing/fusing activation quantization for adjacent
MLP projections or a specialized q8_0 kernel for the recurring stage-2 shapes.

Two additional MMQ scheduling checks did not produce a useful route. Disabling
Stream-K for the q8_0/Base+/1024 run regressed from about `55.1 ms/frame` to
about `1889 ms/frame`, so the default Stream-K path is mandatory for these
large Hiera MLP shapes. Sweeping `GGML_CUDA_MMQ_X_MAX` across
`64,80,96,104,112,120,128` on the same run stayed in the `54.4-59.7 ms/frame`
range, with no result beyond normal noise. Evidence is in
`outputs/mmq-streamk-ab-q8_0-1024-20260515/` and
`outputs/mmq-xmax-sweep-q8_0-1024-20260515/`.

The remaining q8_0/Base+/512 quality gap was also checked before treating it as
a post-processing issue. Candidate selection already matches official PyTorch
on the 9 propagation frames, but the C++ masks are consistently smaller: mean
area ratio is about `0.962`, mean mask IoU is `0.9518`, and min mask IoU is
`0.9377`. Disabling full-mask cleanup made the masks even smaller and worse
(`0.9178` mean IoU, `0.8544` min IoU). A simple one-pixel dilation brought the
mean area ratio close to `0.996`, but mean IoU still dropped to `0.9493` and
min IoU to `0.9146`, so the mismatch is not a uniform boundary shrink that can
be fixed by morphology. For a Rust-wrapper quality reference, q8_0/Base+/1024
remains the validated fallback row; q8_0/Base+/512 is a speed/scale row until a
model-path numerical fix improves it.

A more structural FPN attempt was also rejected. The SAM2 FPN lateral 1x1
weights are stored as `[1,1,C,D]`, so the normal quantizer keeps them as f16/f32
even for q8_0 files. A temporary graph/load change registered the divisible
`C=224/448/896` lateral weights as 2D q8_0 tensors and quantized them at load
time. It did not improve the q8_0/Base+/1024 run (`55.1 ms/frame`, same as the
current noise band) and it changed tracking enough to move bbox coordinates by
up to `123.75 px` against the current default (`9 / 10` mask hashes equal).
Evidence is in `outputs/fpn-2d-q8-ab-q8_0-1024-20260515/`. The patch was
removed; FPN lateral weights remain in their file precision.

Keeping non-quantizable file-f16 weights as f16 instead of expanding them to f32
was also tested as an opt-in loader experiment. This targeted leftover Hiera/FPN
matmuls whose leading dimension is not divisible by the q8 block size, with the
hope that f16 cuBLAS tensor-core paths would beat the current f32 fallback. The
q8_0/Base+/1024 run was slower (`56.3 ms/frame`) and no longer exact against the
current default (`9 / 10` mask hashes equal, max bbox delta `123.75 px`). Evidence
is in `outputs/f16-skipped-weights-q8_0-1024-20260515/`. The experiment was
removed; non-quantizable weights still use the existing f32 registration unless
they can be made exact by a more targeted graph change.

The Hiera patch-embed convolution was also tested against the CUDA direct
`CONV_2D` kernel as a Metal-style "avoid im2col" candidate. This was not useful
on CUDA: the q8_0/Base+/1024 run regressed to `57.7 ms/frame`, and the output was
not exact (`9 / 10` mask hashes equal, max bbox delta `123.75 px`). Evidence is
in `outputs/patch-direct-conv-q8_0-1024-20260515/`. The graph therefore keeps
the current im2col + matmul path for patch embedding; for this shape, CUDA's
matmul path is the better implementation despite the extra im2col node.

The existing D56 attention implementation switches were also re-swept after the
environment repair. On the q8_0/Base+/1024 bbox-only contract, default measured
`55,54,54 ms/frame` and the alternative pack/output toggles stayed in the same
noise band: disabling combined pack `55,54,54`, disabling mixed pack `55,54,55`,
disabling contiguous pack `54,54,54`, and disabling direct 56-wide output
`55,54,56`. Evidence is in
`outputs/fattn56-switch-sweep-q8_0-1024-20260515/summary.json`. There is no
actionable win from changing the current D56 pack switches; further attention
work needs a new kernel-level improvement rather than selecting an existing
fallback.

A shape-selective D56 fallback check was also rejected. The experiment skipped
the padded-MMA path only for small-query D56 rows by requiring
`Q->ne[1] >= 128` before using the wrapper. It did not crash, but it regressed
q8_0/Base+/1024 from the current `53.4 ms/frame` reference to `57.3 ms/frame`
and changed outputs (`9 / 10` mask hashes equal, max bbox delta `7.5 px`).
Evidence is in `outputs/fattn56-pad-min-q-experiment-20260515/min128/`. The
branch was removed. This means the pack-heavy stage-0 rows still cannot be
isolated safely by falling back to the generic attention kernel; the remaining
path has to improve the D56 kernel itself or avoid padded temporaries without
changing the reduction path.

The Base+/512 quality gap was compared against official PyTorch fp32 as well
as bf16 to separate dtype effects from graph differences. Against the existing
q8_0/Base+/512 C++ masks, official fp32 improves the comparison from the bf16
reference (`0.951827` mean IoU, `0.937690` min IoU) to `0.958483` mean IoU and
`0.944930` min IoU, with the same `9 / 9` candidate-selection matches. That is
a small improvement, not a full explanation of the 512 gap. The fp32 official
run is also much slower on this quality script (`58.50 ms/frame` propagation,
versus `19.277 ms/frame` for the earlier bf16 row), so the speed target remains
official bf16 with matched input resolution and source frames. Evidence is in
`outputs/official-quality-frame-dir-q8_0-512-pyfp32-20260515/`.

Trying to force MMF for larger f32 fallback matmuls was rejected before any
performance result could be accepted. The experiment temporarily bypassed
`src1_ncols > 16` for `GGML_TYPE_F32`, but the MMF kernel has a hard
`GGML_ASSERT(ids || ncols_dst <= 16)` precondition and aborted immediately on
the forced q8_0/Base+/1024 run. The default control row remained in the current
noise band (`55,54,54 ms/frame`), while every forced row crashed. Evidence is in
`outputs/mmf-large-f32-q8_0-1024-20260515/`. This path needs a real MMF kernel
redesign for large destination column counts; a heuristic change alone is
invalid.

The upstream CUDA `im2col` grid-y clamp from ggml was also synced into the
branch. It is a correctness/safety update for cases where convolution output
width exceeds CUDA's `gridDim.y` limit, not a SAM2 speed optimization: the
q8_0/Base+/1024 Hiera patch-embed row is under that limit and the new loop takes
one iteration. A short q8_0/Base+/1024 multimask run stayed in the existing
noise band (`55.0`, `53.9`, `53.5 ms/frame`) and repeated runs were internally
exact (`10 / 10` mask hashes, zero bbox/score delta). Evidence is in
`outputs/im2col-grid-fix-q8_0-1024-20260515/`.

A follow-up copy-to-contiguous row specialization adds a default f32->f32
`float4` path for rows that are fully 16-byte aligned and whose contiguous
dimension is divisible by four. All other rows keep the scalar row fastpath. The
path can be disabled with `GGML_CUDA_DISABLE_CPY_TO_CONTIGUOUS_ROW_VEC4=1`.
Five interleaved q8_0/Base+/1024 multimask pairs were exact (`10 / 10` mask
hashes, zero bbox/score delta) and showed a small mean improvement from
`54.06` to `53.62 ms/frame`, saving `0.44 ms/frame`. Evidence is in
`outputs/cpy-row-vec4-paired-q8_0-1024-20260515/summary.json`. After flipping
the path to default-on, a final disable-vs-default gate remained exact and the
single smoke row measured `54.0 ms/frame` with vec4 disabled vs
`53.3 ms/frame` by default. Evidence is in
`outputs/cpy-row-vec4-default-gate-q8_0-1024-20260515/`.

D56 MMA stream-k was tested separately from the earlier MMQ stream-k check and
is also mandatory for this workload. A temporary diagnostic branch disabled
stream-k only for the D56 padded-MMA path (`DKQ=64,DV=64,DV_DST=56`). It
regressed q8_0/Base+/1024 from the current `53-55 ms/frame` band to about
`5420 ms/frame`, and the benchmark reported `det=0`. Evidence is in
`outputs/fattn56-mma-streamk-ab-q8_0-1024-20260515/`. The diagnostic branch was
removed; future D56 work should not simply disable stream-k, and must instead
improve the stream-k kernel/fixup path or retune exact shape schedules without
falling back to the non-stream-k algorithm.

A narrow q8_0 MMQ epilogue fusion was accepted for the common
`MUL_MAT -> ADD(channel bias) -> GELU` MLP rows. The fusion only applies when
the existing q8_0 MMQ bias fusion is legal and the Stream-K schedule is exact,
so the activation is applied during the final writeback and never before a
fixup accumulation. It can be disabled with
`GGML_CUDA_DISABLE_MMQ_BIAS_GELU_FUSION=1`. Five interleaved
q8_0/Base+/1024 runs were exact against the previous path (`10 / 10` mask
hashes, zero bbox and score delta in every pair) and moved the integer
`track/fr` readings from `[58,57,56,57,57]` to `[56,57,56,56,55]`, about
`57.0 -> 56.0 ms/frame` on that smoke. A default-vs-disabled follow-up remained
exact and measured `57 ms/frame` on both rows, so this should be treated as a
small kernel-count reduction rather than a large standalone speedup. Evidence is
in `outputs/mmq-bias-gelu-fusion-paired-q8_0-1024-20260515/` and
`outputs/mmq-bias-gelu-fusion-default-q8_0-1024-20260515/`.

A broader non-sequential q8_0 MMQ bias fusion was investigated for Hiera
attention rows shaped as `MUL_MAT -> bias reshape/view -> ADD`. Enabling every
attention bias row removes many `ADD` kernels and is materially faster on the
q8_0/Base+/1024 smoke (`track/fr` about `52 ms` versus `59 ms` with MMQ bias
fusion disabled), but it is not parity-safe: the 10-frame comparison moved
scores by up to `0.008031`, changed the first mask hash, and moved some later
bboxes by up to `22.5 px`. Evidence is in
`outputs/mmq-broadcast-bias-fusion-q8_0-1024-20260515/`.

The first safe subset was the Hiera attention `qkv.bias` rows. Fusing only those
non-sequential q8_0 MMQ bias adds was exact across five interleaved
q8_0/Base+/1024 pairs (`10 / 10` mask hashes, zero bbox and score delta in every
pair) and moved integer `track/fr` readings from `[55,56,55,56,55]` with the
subset disabled to `[54,54,53,54,53]`, about `55.4 -> 53.6 ms/frame`. Evidence
is in
`outputs/mmq-nonseq-bias-subsets2-q8_0-1024-20260515/`,
`outputs/mmq-nonseq-qkv-default-q8_0-1024-20260515/`, and
`outputs/mmq-nonseq-qkv-paired-q8_0-1024-20260515/`.

The `proj.bias` side was also split by Hiera stage as a diagnostic. Individual
stage-only modes (`proj_stage0`, `proj_stage1`, `proj_stage2`, `proj_stage3`)
were exact on the 10-frame q8_0/Base+/1024 comparison, but broad `proj` remains
unsafe when combined with the qkv subset. The exact boundary found so far is:
qkv plus `proj_stage0`, `proj_stage1`, and `proj_stage3` is safe; adding
`proj_stage2` breaks parity. The safe qkv+stage0/1/3 combination was exact
across five interleaved q8_0/Base+/1024 pairs and moved the mean integer
`track/fr` reading from `54.2` to `53.6 ms/frame`, about `0.6 ms/frame` on that
smoke. This qkv+stage0/1/3 subset is now the default. It can be disabled with
`GGML_CUDA_DISABLE_MMQ_NONSEQ_BIAS_FUSION=1`; the older
`GGML_CUDA_DISABLE_MMQ_NONSEQ_QKV_BIAS_FUSION=1` also disables the whole
non-sequential subset for compatibility. Diagnostic opt-in modes remain
available through `GGML_CUDA_ENABLE_MMQ_NONSEQ_BIAS_FUSION=qkv`,
`proj_stage0`, `proj_stage1`, `proj_stage2`, `proj_stage3`, comma-combined safe
subsets, or broad `proj`/`1` for intentionally unsafe localization. Evidence is
in `outputs/mmq-nonseq-proj-stages-q8_0-1024-20260515/`,
`outputs/mmq-nonseq-qkv-plus-proj-stages-q8_0-1024-20260515/`,
`outputs/mmq-nonseq-qkv-proj-safe-combos-q8_0-1024-20260515/`,
`outputs/mmq-nonseq-qkv-vs-qkv013-paired-q8_0-1024-20260515/`, and
`outputs/mmq-nonseq-default-expanded-q8_0-1024-20260515/`.

After the permission/session reset, the local environment was rechecked before
continuing optimization: xmake rebuilt `sam3_benchmark`, the RTX 5070 Ti Laptop
GPU was visible through `nvidia-smi`, PyTorch `2.8.0+cu128` reported CUDA
available for compute capability 12.0, `just fmt-check` passed, and the
repository path/diff whitespace gates passed. The current default
q8_0/Base+/1024 smoke measured `54.5 ms/frame` with steady Hiera encode around
`38-41 ms/frame`; this is still slower than the earlier official PyTorch bf16
tracking row, so the overall speed goal remains open. Evidence is in
`outputs/current-q8_0-1024-after-nonseq-default-20260515/`.

The refreshed CUDA node and MMQ launch profiles after the qkv+stage0/1/3 default
still rank Hiera `MUL_MAT` and D56 `FLASH_ATTN_EXT` as the dominant work. The
q8_0 window-projection shape `ncols_x=448,nrows_x=448,ncols_dst=196,ch=25`
still uses `mmq_x=112` with a Stream-K fixup, but a coarse `GGML_CUDA_MMQ_X_MAX=96`
probe is not acceptable despite a faster single timing (`52.3 ms/frame` versus
`55.5 ms/frame` in that A/B): it changed one mask hash, moved bboxes by up to
`26.25 px`, and changed scores by up to `0.011876`. Evidence is in
`outputs/node-profile-after-nonseq-default-q8_0-1024-20260515/`,
`outputs/mmq-launch-after-nonseq-default-q8_0-1024-20260515/`, and
`outputs/mmq-q8-x96-ab-q8_0-1024-20260515/`. This reinforces that MMQ scheduling
changes need exact parity gates, not only faster smoke timings.

The matched C++/official-PyTorch speed comparison was then rerun serially after
the qkv+stage0/1/3 default so the Python baseline was not competing with another
benchmark process on the same GPU:

| Encode size | C++ q8_0 track ms/frame | PyTorch bf16 track ms/frame | PyTorch/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | --- |
| 1024 | 53.3 | 42.70 | 0.801 | `outputs/model-matrix-q8_0-1024-after-nonseq-default-serial-20260515/summary.json` |
| 512 | 15.6 | 10.19 | 0.653 | `outputs/model-matrix-q8_0-512-after-nonseq-default-serial-20260515/summary.json` |

Values below `1.0` still mean official PyTorch is faster. The parallel
1024/512 rerun artifacts in
`outputs/model-matrix-q8_0-1024-after-nonseq-default-20260515/` and
`outputs/model-matrix-q8_0-512-after-nonseq-default-20260515/` should not be
used for speed conclusions because they were run concurrently on one GPU.

The current Hiera gap audit uses cuts-free C++ profiles and the existing
instrumented official-Python profiles. At 1024, C++ steady Hiera encode is still
about `37.77 ms` versus PyTorch `forward_image` at `30.41 ms`, leaving roughly
`7.36 ms/frame` in image encoding. At 512, C++ steady Hiera encode is already
below the instrumented PyTorch `forward_image` mean (`10.27 ms` vs `15.92 ms`),
but the matched end-to-end tracking row is still slower than PyTorch, so the
remaining 512 gap needs a tracker-path/profile refresh rather than more Hiera
work alone. Evidence is in
`outputs/profile-q8_0-after-nonseq-default-20260515/` and
`outputs/hiera-gap-q8_0-after-nonseq-default-20260515/summary.json`.

The current completion audit remains `not_complete`. Matched input conditions,
detailed Hiera/profile attribution, and a speed-priority ordering are covered,
but two objective criteria are still missing: C++ is not faster than official
PyTorch at either 512 or 1024, and Base+ fallback quality parity is only met at
1024. The q8_0/1024 official-quality row is strong (`mean_mask_iou=0.9951`,
`min_mask_iou=0.9933`), while q8_0/512 remains below the parity threshold
(`mean_mask_iou=0.9518`, `min_mask_iou=0.9377`). Evidence is in
`outputs/goal-audit-q8_0-after-nonseq-default-20260515/summary.json`,
`outputs/official-quality-frame-dir-q8_0-1024-fpn-mulmat-20260515/summary.json`,
and `outputs/official-quality-frame-dir-q8_0-512-fpn-mulmat-20260515/summary.json`.

The 512 row was then profiled inside `propagate_single` because the Hiera gap
audit showed C++ steady Hiera encode below the synchronized official-Python
`forward_image` attribution, while the matched end-to-end speed row was still
slower. The added `SAM3_PROFILE` spans are profiling-only and the CPU timing
macro now avoids taking timestamps when `SAM3_PROFILE` is unset. A normal
no-profile q8_0/Base+/512 smoke stayed in the current band at `16.7 ms/frame`.

The detailed 512 propagation profile shows the non-kernel propagation overhead
is not the main gap:

| Section | Mean ms | Note |
| --- | ---: | --- |
| `prop_graph_compute_total` | `3.70` | almost identical to `propagate_single`; kernel body dominates |
| `prop_input_upload` | `0.25` | prompt/PE/feature tensor copies into fresh graph inputs |
| `prop_memory_slot_read` | `0.24` mean, `~0.07` steady | first call includes one-time cost |
| `prop_graph_alloc` | `0.17` | small compared with graph compute |
| `prop_graph_build` | `0.09` | small compared with graph compute |
| `prop_output_read` | `0.05` | selected mask/token reads are minor |

The 512 CUDA node profile confirms the same kernel classes dominate at lower
resolution, just with smaller shapes: q8_0 Hiera MLP `MUL_MAT` rows
(`1792x1024`, `448x1024`, and window `1344x196x9`) and D56 window/global
`FLASH_ATTN_EXT`. This makes the next optimization priority:

1. 1024: reduce D56 FlashAttention and q8_0 Hiera MLP `MUL_MAT` body time.
2. 512: still target kernel body time, but include propagation graph kernels;
   CPU graph build, tensor upload/readback, and memory slot assembly are too
   small to close the `15.6 -> 10.2 ms/frame` PyTorch gap alone.
3. Quality: fix or explain the q8_0/512 official-Python mask IoU gap before
   treating 512 as a Rust-wrapper baseline.

Evidence is in `outputs/profile-propagate-detail-q8_0-512-20260515/`,
`outputs/node-profile-q8_0-512-after-prop-detail-20260515/`, and
`outputs/smoke-q8_0-512-after-prop-profile-spans-20260515/`.

A follow-up probe added `GGML_CUDA_PROFILE_MMQ_BIAS_REJECT=1` and confirmed why
the remaining q8_0 Hiera MLP/window rows do not use the MMQ bias(+GELU) fusion:
the hot `1792x1024`, `448x1024`, and `1344x196x9` rows are rejected as
`stream-k-not-exact`, while the smaller rejected set is ordinary type/padding
ineligibility. This is not a bias shape/layout issue.

An experimental stream-k fixup path for bias/activation was then implemented
behind `GGML_CUDA_ENABLE_MMQ_BIAS_FIXUP_FUSION=1`, but it is not default-safe.
On q8_0/Base+/512 it improved the short 10-frame timing from roughly
`16.5 -> 14.5 ms/frame`, but changed output enough to reject it:
mask hash matched only `9/10`, max bbox delta was `59 px`, and max score delta
was about `0.048`. Bias-only mode was also not parity-safe. The default path
therefore keeps the exact stream-k schedule guard and continues to reject these
rows unless the experimental env var is explicitly set. Evidence is in
`outputs/mmq-bias-reject-q8_0-512-20260515/`,
`outputs/mmq-fixup-bias-fusion-parity-q8_0-512-20260515/`,
`outputs/mmq-fixup-bias-only-parity-q8_0-512-20260515/`, and
`outputs/mmq-bias-reject-after-fixup-guard-q8_0-512-20260515/`.

The guarded default q8_0/Base+/512 smoke stayed in the current performance band
at `15.1 ms/frame` (`p50=14.7`, `p95=17.5`). The official-Python comparison
with `--image-size 512` validated matching metadata and measured Python bf16 at
`16.10 ms/frame` for that short run, but it compared zero mask rows because the
C++ input JSONL was produced with `--bbox-only`. Use frame-dir/mask-emitting
quality runs for real IoU gates. Evidence is in
`outputs/mmq-fixup-guarded-default-speed-q8_0-512-20260515/` and
`outputs/official-quality-mmq-fixup-guarded-default-q8_0-512-20260515/`.

The D56 FlashAttention path now has a dedicated attribution summarizer,
`scripts/summarize_fattn56_profile.py`, so the pack/MMA/slice split can be
ranked by shape instead of hand-reading the event log. Four-frame q8_0/Base+
profiles show that the remaining FATTN56 work is mostly MMA body time, not the
head56 slice:

| Encode size | Effective FATTN56 rows | Total FATTN56 ms | Pack ms | MMA ms | Main shapes |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 88 | `31.18` | `7.88` | `23.09` | window `196x8x25`, global `4096x8`, stage0 `64x2x1024` |
| 512 | 88 | `6.96` | `1.90` | `4.86` | window `196x8x9`, global `1024x8`, stage0 `64x2x256` |

At 1024 the largest individual buckets are:

1. window D56 `Q/K/V=[56,196,8,25]`: `10.16 ms` total, `7.72 ms` MMA;
2. global D56 `Q/K/V=[56,4096,8,1]`: `9.92 ms` total, `9.50 ms` MMA;
3. stage0 D56 `Q/K/V=[56,64,2,1024]`: `4.58 ms` total, split between pack
   (`2.06 ms`) and MMA (`2.51 ms`).

At 512 the same ordering holds but the absolute D56 budget is much smaller
(`6.96 ms` across the profiled run). This makes a D56 MMA-body improvement more
important for the 1024 PyTorch gap, while 512 still needs non-FATTN kernel work
as well. Evidence is in `outputs/fattn56-profile-q8_0-1024-20260515/` and
`outputs/fattn56-profile-q8_0-512-20260515/`.

The current mask-emitting official-quality runs confirm that Base+ fallback
quality is resolution-dependent under matched decoded-source and model-input
conditions. Both runs wrote C++ masks and initial multimask candidates, then
compared against official PyTorch with the same `--image-size`:

| Encode size | Frames | Mean mask IoU | Min mask IoU | Initial candidate selection | Propagation candidate selection | Evidence |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 1024 | 10 | `0.99395` | `0.99315` | match | `7/9` match, but masks still high-IoU | `outputs/official-quality-mask-current-q8_0-1024-20260515/` |
| 512 | 10 | `0.93131` | `0.84784` | match | `9/9` match | `outputs/official-quality-mask-current-q8_0-512-20260515/` |

This means the 1024 Base+ fallback remains a valid high-quality reference, but
the 512 fallback is not yet a quality-parity baseline even though the prompt and
candidate selection contracts match. The 512 gap appears to be mask/logit
quality after resizing rather than a selected-candidate mismatch.

A stricter 512 input-source check then used the same extracted JPEG frame
directory for both C++ and official PyTorch. That removes the earlier
video-vs-frames source mismatch and improves the q8_0/Base+/512 mask result to
`mean_mask_iou=0.96535`, `min_mask_iou=0.95854`, with initial selection matched
and propagation selection `9/9` matched. The remaining gap is therefore much
more likely to be resize/JPEG-decoder sensitivity than a tracker candidate
selection issue. New benchmark metadata records `input_source` and `frame_dir`
so future quality runs can distinguish true matched-frame comparisons from
resolution-only matches. Evidence is in
`outputs/official-quality-mask-current-q8_0-512-same-frame-dir-20260515/`.

## 2026-05-16 Environment Repair and MMQ Plain Writeback

After the login/session change, the CUDA environment was checked before doing
more kernel work. `nvidia-powerd.service` was already `active` and `enabled`,
but systemd had a stale unit reload warning. Running `systemctl daemon-reload`
cleared that state. A short CUDA benchmark and compute-sanitizer initcheck then
passed:

| Check | Result | Evidence |
| --- | --- | --- |
| CUDA smoke, q8_0/Base+/512 | model load and 2-frame tracking OK | `outputs/env-smoke-20260516-102143/` |
| initcheck, f16/Base+/512 | `ERROR SUMMARY: 0 errors` | `outputs/env-initcheck-20260516-102209/` |
| q8_0/Base+/512 speed after repair | `15.7`, `14.6`, `14.6 ms/frame` | `outputs/speed-after-env-repair-q8_0-512-20260516-102238/` |
| q8_0/Base+/1024 speed after repair | `55.1`, `53.3`, `54.0 ms/frame` | `outputs/speed-after-env-repair-q8_0-1024-20260516-102315/` |

The environment is therefore back in the same performance band as the previous
stable runs; the remaining gap is code/kernel work, not powerd or driver
visibility.

The next kept CUDA change specializes the common MMQ writeback path where
`bias == nullptr` and `activation == MMQ_ACT_NONE`. The earlier bias/GELU
fusion work made writeback handle optional post-ops, but ordinary Hiera MMQ rows
do not need those checks. The new `plain_writeback` template path keeps the
generic post-op path for fused bias/activation candidates while compiling the
normal path without per-element activation/bias branches.

This is a conservative cleanup, not a PyTorch-closing speedup. Determinism is
unchanged and speed is only slightly better or statistically tied:

| Check | Result | Evidence |
| --- | --- | --- |
| q8_0/Base+/512 determinism | 3/3 runs matched all 5 mask hashes | `outputs/determinism-mmq-plain-writeback-q8_0-512-20260516-103042/` |
| q8_0/Base+/512 speed | `14.6`, `14.7`, `14.7 ms/frame` | `outputs/mmq-plain-writeback-final-q8_0-512-20260516-103054/` |
| q8_0/Base+/1024 speed | `53.8`, `53.9`, `53.7 ms/frame` | `outputs/mmq-plain-writeback-final-q8_0-1024-20260516-103107/` |
| q8_0/Base+/512 official quality, video source | `mean_mask_iou=0.93147`, `min_mask_iou=0.85417` | `outputs/official-quality-mmq-plain-writeback-q8_0-512-20260516-103251/` |

The main remaining optimization target is unchanged: D56 FlashAttention MMA body
time and q8_0 MMQ arithmetic/scheduling for Hiera MLP/window projection shapes.

The q8_0 MMQ activation quantizer was then specialized for the common no-ids
path. Hiera's regular `MUL_MAT` rows call `quantize_mmq_q8_1_cuda` with
`ids == nullptr`; the older kernel still carried the runtime
`ids ? ids[i1] : i1` path used by expert/id rows. The new template split keeps
the ids-capable kernel for those callers but launches a no-ids specialization
for the normal path. This does not change rounding (`roundf`) or the q8_1
layout.

| Check | Result | Evidence |
| --- | --- | --- |
| q8_0/Base+/512 speed | warm rows `14.4`, `14.5`, `14.6`, `14.8 ms/frame` | `outputs/mmq-quant-noids-template-q8_0-512-20260516-103440/` |
| q8_0/Base+/1024 speed | `53.1`, `53.5`, `52.7 ms/frame` | `outputs/mmq-quant-noids-template-q8_0-1024-20260516-103458/` |
| q8_0/Base+/1024 MMQ profile | `quant_sum=8.367 ms`, `mmq_sum=51.290 ms` over the 3-frame profiled run | `outputs/mmq-profile-noids-template-q8_0-1024-20260516-103515/` |
| q8_0/Base+/512 determinism | 3/3 runs matched all 5 mask hashes | `outputs/determinism-mmq-quant-noids-template-q8_0-512-20260516-103530/` |
| q8_0/Base+/512 official quality, video source | unchanged at `mean_mask_iou=0.93147`, `min_mask_iou=0.85417` | `outputs/official-quality-mmq-quant-noids-template-q8_0-512-20260516-103546/` |

This is still not enough to meet the PyTorch-speed objective. It is a low-risk
MMQ cleanup that trims activation quantization overhead, while the remaining
large gap still requires D56 FlashAttention body work and larger MMQ body
improvements.

The synchronized node profile was refreshed after the no-ids quantizer split.
The hotspot summarizer classification now also recognizes fused MMQ rows by
destination shape because some fused node log lines no longer include the
original quantized `src0_type`. The ranking did not change materially:

| Rank | Hotspot | Drop-max sum ms | Priority |
| ---: | --- | ---: | --- |
| 1 | `MUL_MAT dst=f32[1792,4096,1,1]` | `11.217` | stage-2 MLP expansion |
| 2 | `FLASH_ATTN_EXT dst=f32[56,8,196,25]` | `9.933` | D56 window attention |
| 3 | `MUL_MAT dst=f32[448,4096,1,1]` | `9.623` | stage-2 MLP projection |
| 4 | `FLASH_ATTN_EXT dst=f32[56,8,4096,1]` | `9.371` | D56 global attention |
| 5 | `MUL_MAT dst=f32[1344,196,25,1]` | `9.009` | window qkv projection |

Evidence is in
`outputs/node-profile-q8_0-1024-after-noids-template-20260516-103627/`.

Two follow-up MMQ checks were rejected:

| Experiment | Result | Evidence |
| --- | --- | --- |
| Re-sweep `GGML_CUDA_MMQ_X_MAX` after the quantizer split | no stable winner; values from `96` to `128` stayed in the same `52.8-54.2 ms/frame` noise band and do not override earlier parity risk | `outputs/mmq-xmax-sweep-after-noids-template-q8_0-1024-20260516-103729/` |
| Template-specialize the main MMQ kernel for `ids_dst == nullptr` | 512 stayed similar, but 1024 regressed to `53.5`, `55.0`, `54.7 ms/frame` and increased RSS; patch was reverted | `outputs/mmq-noids-kernel-template-q8_0-512-20260516-104306/`, `outputs/mmq-noids-kernel-template-q8_0-1024-20260516-104328/`, `outputs/mmq-noids-kernel-template-reverted-q8_0-1024-20260516-104553/` |
| Lower the MMQ Stream-K tiling efficiency threshold below `90%` | no robust win; threshold `80/70/60` stayed in the same noise band and `50` regressed, so the diagnostic env switch was removed | `outputs/mmq-launch-after-noids-template-q8_0-1024-20260516-104656/`, `outputs/mmq-tile-eff-threshold-q8_0-1024-20260516-104855/` |

An upstream `ggml-org/ggml` check found only one newer CUDA FATTN-related commit
on `upstream/master`, `8498d0f6`. It adds `DKQ=192,DV=128` MiMo-V2.5
FlashAttention cases and does not target SAM2 Hiera's D56 shapes, so there is no
direct upstream speed patch to import for this workload.

The D56 FlashAttention native-`V=56` MMA path was also tested more narrowly for
the stage-0-like 64-token, 2-head shape. This preserves tracking parity, but it
is not a robust speed win:

| Mode | q8_0/Base+/1024 speed | Parity | Evidence |
| --- | --- | --- | --- |
| Default | `54.5`, `53.6`, `53.5 ms/frame` | reference | `outputs/fattn56-native-stage0-q8_0-1024-20260516-105133/` |
| Stage-0-only native `V=56` experiment | `52.7`, `53.4`, `53.6 ms/frame` | 10/10 mask hashes equal, bbox and score deltas zero | `outputs/fattn56-native-stage0-parity-q8_0-1024-20260516-105200/` |

Because the improvement is in the noise band after the first warm run and the
stage-0 shape itself was not improved in the FATTN56 event profile, this path is
not kept as a separate runtime knob. The next real speed target is still a D56
MMA-body implementation that reduces the `FLASH_ATTN_EXT` kernel time itself,
plus larger MMQ body improvements for the stage-2 MLP shapes.

Enabling native `V=56` for all D56 FATTN shapes is worse and remains rejected:

| Mode | q8_0/Base+/1024 speed | Evidence |
| --- | --- | --- |
| Default | `54.1`, `53.5`, `53.3 ms/frame` | `outputs/fattn56-native-full-q8_0-1024-20260516-105509/` |
| `GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` | `55.4`, `55.4`, `55.5 ms/frame` | `outputs/fattn56-native-full-q8_0-1024-20260516-105509/` |

The likely reason is that reducing `DV` from 64 to 56 saves some value padding
work but makes the MMA body less favorable; this workload is dominated by the
attention body and MMQ arithmetic rather than the final D56 slice.

After removing the stage-0-only runtime branch, the current default remains in
the same band: `53.8`, `52.9`, `53.7 ms/frame` for q8_0/Base+/1024 in
`outputs/current-after-stage0-drop-q8_0-1024-20260516-105646/`.

The q8_0/Base+/1024 MMQ path was also checked against a forced cuBLAS build.
Forced cuBLAS is much slower (`104.5`, `103.4`, `103.9 ms/frame`) than the
current MMQ build (`54.6`, `54.0`, `53.9 ms/frame`) in
`outputs/mmq-vs-cublas-q8_0-1024-20260516-105834/`. The optimization direction
therefore remains MMQ-internal kernel work, not routing quantized Hiera MLP rows
through dequantize-plus-cuBLAS.

Replacing q8_1 activation quantization's `roundf` with CUDA round-to-nearest
integer was rejected. It did not improve speed (`54.6`, `55.6`, `53.6
ms/frame`) and changed tracking output compared with the roundf baseline:
`min_bbox_iou=0.98385`, `max_bbox_delta_px=15.0`, and `9/10` mask hashes equal
in `outputs/mmq-quant-rn-q8_0-1024-20260516-105925/`. Exact parity requires
keeping the existing `roundf` semantics.

The MMQ bias-fusion audit shows that the remaining bias-fusion rejects in this
q8_0/Base+/1024 run are mostly f32-weight early-stage/FPN rows rather than the
top q8_0 stage-2 MMQ rows. Disabling the current non-sequential MMQ bias fusion
stays in the same runtime band (`default: 53.2`, `52.7`, `54.9 ms/frame`;
`disabled: 54.6`, `53.4`, `53.6 ms/frame`) in
`outputs/mmq-bias-fusion-ab-q8_0-1024-20260516-110034/`. This confirms the
current fusion is useful but not the remaining PyTorch gap; the next MMQ work
must reduce the large stage-2 `MUL_MAT` body time itself.

Changing the global MMQ K-iteration constant is not a safe win. `MMQ_ITER_K=128`
compiled with many zero-modulo warnings and regressed q8_0/Base+/1024 to
`61.7`, `60.7`, `60.3 ms/frame` in
`outputs/mmq-iterk128-q8_0-1024-20260516-110201/`. `MMQ_ITER_K=512` was much
faster (`52.0`, `50.5`, `50.8 ms/frame`) and made q8_0/Base+/512 faster
(`13.9`, `13.6`, `13.5 ms/frame`), but it broke tracking parity badly:
`min_bbox_iou=0.07681`, `max_bbox_delta_px=555.0`, and only `9/10` mask hashes
equal in `outputs/mmq-iterk512-parity-q8_0-1024-20260516-110400/`. The global
constant therefore remains `256`; future work would need a shape-specific and
parity-safe variant, not a blanket change.

Two follow-up checks confirmed this is not just a bad stream-k split:
`MMQ_ITER_K=448`, chosen to match the 448-wide Hiera stage-2 expansion K, still
breaks parity (`min_bbox_iou=0.70072`, `max_bbox_delta_px=142.0`,
`9/10` mask hashes equal) despite running at `52.5`, `51.8`, `52.0 ms/frame`;
evidence is in `outputs/mmq-iterk448-q8_0-1024-20260516-110641/`. Disabling
stream-k with `MMQ_ITER_K=512` was both incorrect and extremely slow
(`2020.6`, `2034.3`, `2045.4 ms/frame`) in
`outputs/mmq-iterk512-no-streamk-q8_0-1024-20260516-110804/`.

The likely reason is in the q8_0 tile loader: it has a hardcoded 32-thread
loader for a two-`MMQ_TILE_NE_K` q8_0 tile and even notes that
`MMQ_ITER_K / (4 * QR8_0) == 64` would be required while NVIDIA has only 32
threads per warp. Larger global `MMQ_ITER_K` values are therefore not a valid
q8_0 speed path without a new q8_0 tile-loader/body variant.

The relevant upstream MMQ history was checked after this result. The current
branch already contains `a1fde6fb` (`CUDA: reduce MMQ stream-k overhead`), so
there is no missing upstream stream-k overhead patch to import for this case.
The remaining q8_0 opportunity is a new Hiera-shaped q8_0 MMA tile path, not an
unmerged generic ggml fix.

The current q8_0/Base+/1024 MMQ event profile was refreshed with shape grouping
in `outputs/mmq-profile-current-q8_0-1024-20260516-111152/`. After dropping the
first row per shape, MMQ totals are:

| Metric | Sum |
| --- | ---: |
| Activation quantization | `10.887 ms` |
| MMQ body | `57.481 ms` |
| Total profiled MMQ | `68.369 ms` |

The top shapes confirm that the next useful MMQ work must target the body, not
activation quantization:

| Shape | Count | Quant | MMQ body | Total |
| --- | ---: | ---: | ---: | ---: |
| `q8_0[448,1792] x f32[448,4096] -> f32[1792,4096]` | 63 | `0.687 ms` | `10.949 ms` | `11.636 ms` |
| `q8_0[1792,448] x f32[1792,4096] -> f32[448,4096]` | 63 | `1.767 ms` | `8.251 ms` | `10.018 ms` |
| `q8_0[448,1344] x f32[448,196,25] -> f32[1344,196,25]` | 47 | `0.588 ms` | `8.697 ms` | `9.286 ms` |

The current launch profile in
`outputs/mmq-launch-current-q8_0-1024-20260516-111207/` also confirms these
major shapes are not primarily a fixup problem. The two stage-2 MLP shapes use
`mmq_x=128`, `mmq_y=128`, high tile efficiency (`97%` and `92%`), and
`fixup=0`. The window QKV shape uses `mmq_x=112`, `mmq_y=128`, `99%`
efficiency, and `fixup=0`. This narrows the next MMQ implementation target to a
q8_0 tile/body variant for these Hiera shapes rather than launch scheduling or
activation quantization.

A fresh current-vs-official 1024 timing row was also taken in
`outputs/hiera-gap-current-q8_0-1024-20260516-111316/`. The instrumented C++
run reports `track_ms=56.4` and steady Hiera encode `37.724 ms`; official
PyTorch bf16 reports `track_ms=42.856` and `forward_image=28.200 ms` on the
same extracted frame directory. The remaining Hiera encode gap is therefore
about `9.52 ms`, while the normal non-instrumented q8_0/Base+/1024 runtime is
still in the low-`53 ms/frame` band. Closing this gap requires a real D56
FlashAttention body or q8_0 MMQ body win; the rejected launch/quantizer tweaks
do not have enough headroom.

One q8_0 MMQ body micro-check replaced the NVIDIA-side B tile `load_generic`
with `load_ldmatrix` in `vec_dot_q8_0_q8_1_mma`. This preserved exact tracking
parity (`10/10` mask hashes equal) but slowed q8_0/Base+/1024 to `55.6`,
`55.2`, `55.4 ms/frame`; evidence is in
`outputs/mmq-q8-b-ldmatrix-q8_0-1024-20260516-111558/`. The existing
`load_generic` path remains correct for this Blackwell/Hiera workload.

After the CUDA environment was rechecked, the current q8_0/Base+/1024 FATTN56
profile was refreshed in
`outputs/fattn56-profile-current-q8_0-1024-20260516-111942/`. After dropping the
first row per shape, profiled FATTN56 totals were `31.242 ms`: pack `7.880 ms`,
MMA `23.077 ms`, and slice `0.285 ms`. The two dominant rows remain the same:

| Shape | Total | Pack | MMA |
| --- | ---: | ---: | ---: |
| `Q[56,196,8,25] K[56,196,8,25]` | `10.296 ms` | `2.392 ms` | `7.750 ms` |
| `Q[56,4096,8,1] K[56,4096,8,1]` | `9.976 ms` | `0.409 ms` | `9.530 ms` |

Disabling FATTN stream-k was rejected immediately. The default q8_0/Base+/1024
rows in `outputs/fattn-streamk-ab-q8_0-1024-20260516-112025/` stayed around
`55 ms/frame`, while the no-stream-k trial pushed steady Hiera encode from about
`39-42 ms` to about `5400 ms` per frame. The run was stopped early because the
direction is unambiguously invalid for this workload.

Increasing the `DKQ=64, DV=64, ncols=64` FATTN MMA config from 128 to 256
threads looked promising in event timing but is not parity-safe. The event
profile in
`outputs/fattn-ncols64-256threads-profile-q8_0-1024-20260516-112356/` reduced
drop-first FATTN56 total from `31.242 ms` to `26.927 ms`, mainly by reducing MMA
time from `23.077 ms` to `18.372 ms`. The normal short speed rows in
`outputs/fattn-ncols64-256threads-q8_0-1024-20260516-112337/` were
`53.6`, `54.5`, and `54.3 ms/frame`, so the end-to-end win was not decisive.
More importantly, comparison against a bbox-only baseline was not exact:
`mask_hash_equal_rows=9/10`, `max_bbox_delta_px=7.5`, and
`max_score_abs_delta=0.012782` in
`outputs/fattn-ncols64-256threads-parity-q8_0-1024-20260516-112416/summary-bbox-current-baseline.json`.
The config was therefore reverted. A future FATTN body change must preserve the
same accumulation behavior or be validated as an intentional quality change
against the official Python baseline.

The q8_0 MMQ X tile cap was also swept without changing code, using
`GGML_CUDA_MMQ_X_MAX` on the same q8_0/Base+/1024 bbox-only contract:
`outputs/mmq-q8-xmax-sweep-1024-20260516-112647/`.

| `GGML_CUDA_MMQ_X_MAX` | Track ms/frame values | Mean |
| ---: | --- | ---: |
| `80` | `56.5`, `55.3` | `55.90` |
| `88` | `55.6`, `56.1` | `55.85` |
| `96` | `54.6`, `54.8` | `54.70` |
| `104` | `54.6`, `54.9` | `54.75` |
| `112` | `55.4`, `55.6` | `55.50` |
| `120` | `54.9`, `55.2` | `55.05` |
| `128` | `55.0`, `54.6` | `54.80` |

This does not justify adding a Blackwell q8_0 cap. The apparent `96/104` win is
within the current short-run noise band and does not materially improve the
default `128` row. The existing q4_0/q4_1 caps remain format-specific; q8_0
needs a real body/quantization improvement rather than a tile-cap tweak.

`SAM3_CUDA_ENABLE_GRAPHS=1` was retested because the application wrapper
currently disables ggml CUDA graphs unless that env var is present. On the same
q8_0/Base+/1024 run, enabling graphs did not improve speed and raised RSS:
default rows were `55.2`, `54.2`, `55.3 ms/frame`, while enabled rows were
`55.0`, `55.5`, `55.6 ms/frame` with RSS around `621-623 MiB`; evidence is in
`outputs/cuda-graphs-enable-ab-q8_0-1024-20260516-112808/`. For this current
1024 workload, graph capture overhead and warmup do not close the PyTorch gap.

Finally, compiling the library with `SAM3_LOG_LEVEL=0` removed the timed
per-frame summary logs, but did not produce a robust speed win:
`55.0`, `54.1`, `54.9 ms/frame` in
`outputs/sam3-loglevel0-q8_0-1024-20260516-112925/`. The default log level was
therefore left unchanged; logging cleanup is a benchmark hygiene issue, not a
root speed fix.

After profiling permissions were repaired, Nsight Compute was usable via an
admin run. A stale `/tmp/nsight-compute-lock` file had to be removed; after
that, counter collection worked. The first useful run is
`outputs/ncu-quant-sudo-q8_0-1024-20260516-1139/`, targeting
`quantize_mmq_q8_1<0,0>`. The large activation-quantization launch
`grid=(65536,1,1)` reports:

| Metric | Value |
| --- | ---: |
| Duration | `494240-494624 ns` |
| DRAM throughput | `84.45-84.66%` |
| Compute throughput | `38.71-39.98%` |

This confirms the large q8_1 activation quantizer is mostly DRAM-bound. The
small q8_1 launches are more balanced (`~64%` compute, `~68%` DRAM), but they
are not the dominant q8_0/Base+/1024 gap. The safe code cleanup that changed
the layout branches inside `quantize_mmq_q8_1` to `if constexpr` preserved
exact bbox-only tracking determinism:
`outputs/quant-ifconstexpr-parity-q8_0-1024-20260516-1133/summary.json` has
`mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`. Normal speed rows were `55.3`, `53.9`, and
`54.7 ms/frame` in
`outputs/quant-ifconstexpr-q8_0-1024-20260516-1133/`, so this should be treated
as a harmless specialization/cleanup, not as a material speed win.

Nsight Compute was then run on the q8_0 MMQ body. The speed-of-light row in
`outputs/ncu-mmq-sudo-q8_0-1024-20260516-1140/` for
`mul_mat_q<8,128,1,0>` reports duration around `536640-551648 ns`, but only
`~30%` compute throughput and `~30%` DRAM throughput. Scheduler/warp stats in
`outputs/ncu-mmq-warp2-q8_0-1024-20260516-1142/` show the reason: only
`0.41` eligible warps per scheduler, `69.28%` cycles with no eligible warp, and
about `1.94-1.95` active warps per scheduler. Occupancy data in
`outputs/ncu-mmq-occupancy-q8_0-1024-20260516-1143/` confirms theoretical
occupancy is only `16.67%`, limited by registers and shared memory. This matches
`cuobjdump` resource usage for the q8_0 `mmq_x=128` instantiations, which still
show `REG=255` and stack usage.

Two simple ways to raise occupancy were rejected:

| Experiment | Evidence | Result |
| --- | --- | --- |
| Separate build with `-maxrregcount=128` | `outputs/maxrreg128-q8_0-1024-20260516-1144/` | resource usage stayed at `REG=255`; speed regressed to `55.8`, `55.2`, `55.2 ms/frame` |
| MMQ main `__launch_bounds__(..., 2)` on NVIDIA Volta+ | `outputs/mmq-launchbounds2-q8_0-1024-20260516-1200/` | `REG` dropped to `128`, but stack spill rose to `352-496 bytes`; speed regressed to `64.2`, `62.3`, `62.2 ms/frame` |
| Global CUDA `nwarps=4` MMQ body | build log from the local experiment | compile-time invariant `nwarps*tile_C::I == mmq_y` failed for q8_0 and other MMQ instantiations; this requires a new tile/writeback design, not a constant flip |
| Global NVIDIA `mmq_y=64` MMQ body | build log from the local experiment | the same `nwarps*tile_C::I == mmq_y` invariant failed; row tile size is tied to the current MMA tile/writeback design |
| Low q8_0 `GGML_CUDA_MMQ_X_MAX` sweep | `outputs/mmq-q8-xcap-low-sweep-1024-20260516-1145/` | `48-72` all stayed around `56-58 ms/frame`, slower than default/noise band |

The current evidence says the next real q8_0 MMQ improvement cannot be a launch
cap, compiler flag, or forced launch-bound occupancy tweak. It likely requires
a new Hiera-shaped MMQ body that lowers per-block state, reduces live
accumulators, or splits the q8_0 `mmq_x=128` work so Blackwell can keep more
than one block resident without increasing spill or global traffic enough to
lose the gain.

The Base+ 1024 precision sweep was repeated to check whether the current CUDA
gap is specific to q8_0. The retained logs are one 10-frame bbox-only run per
precision, so they are useful as a direction check rather than a final
statistical claim:

| Model | Evidence | Track ms/frame | P50 | P95 | RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| `sam2.1_hiera_base_plus_q4_0` | `outputs/base-plus-precision-speed-1024-20260516-1208/sam2.1_hiera_base_plus_q4_0.log` | `56.8` | `54.2` | `64.4` | `608.4 MiB` |
| `sam2.1_hiera_base_plus_q4_1` | `outputs/base-plus-precision-speed-1024-20260516-1208/sam2.1_hiera_base_plus_q4_1.log` | `55.6` | `53.4` | `64.1` | `607.7 MiB` |
| `sam2.1_hiera_base_plus_q8_0` | `outputs/base-plus-precision-speed-1024-20260516-1208/sam2.1_hiera_base_plus_q8_0.log` | `54.4` | `52.0` | `63.5` | `607.5 MiB` |
| `sam2.1_hiera_base_plus_f16` | `outputs/base-plus-float-speed-1024-20260516-1209/sam2.1_hiera_base_plus_f16.log` | `62.7` | `61.1` | `69.6` | `746.8 MiB` |
| `sam2.1_hiera_base_plus_f32` | `outputs/base-plus-float-speed-1024-20260516-1209/sam2.1_hiera_base_plus_f32.log` | `66.1` | `64.9` | `72.0` | `601.0 MiB` |

This rules out a simple precision switch as the path to the official PyTorch
bf16 row (`42.856 ms/frame` in
`outputs/hiera-gap-current-q8_0-1024-20260516-111316/`). q8_0 is still the best
local ggml model variant for this 1024 Base+ workload, and all local precision
variants remain materially slower than the official PyTorch timing.

The current CUDA node hotspot profile was refreshed in
`outputs/node-profile-current-q8_0-1024-20260516-1210/`. This run used
`GGML_CUDA_PROFILE_NODES=1`, so the synchronization overhead makes the absolute
sum larger than normal runtime; the drop-max sums are for ranking only. The top
hotspots are:

| Rank | Operation signature | Count | Drop-max sum | Mean |
| ---: | --- | ---: | ---: | ---: |
| 1 | `MUL_MAT dst=f32[1792,4096,1,1]` | 48 | `9.074 ms` | `0.193 ms` |
| 2 | `MUL_MAT dst=f32[448,4096,1,1]` | 48 | `7.953 ms` | `0.169 ms` |
| 3 | `FLASH_ATTN_EXT dst=f32[56,8,196,25]` | 36 | `7.640 ms` | `0.218 ms` |
| 4 | `MUL_MAT dst=f32[1344,196,25,1]` | 36 | `7.637 ms` | `0.218 ms` |
| 5 | `FLASH_ATTN_EXT dst=f32[56,8,4096,1]` | 9 | `7.130 ms` | `0.891 ms` |

This independently matches the lower-level profiles: the remaining useful work
is concentrated in q8_0 MMQ stage-2 MLP, window QKV projection, and the D56
FlashAttention window/global paths. There is no single elementwise or decode
side outlier large enough to close the PyTorch gap. The next implementation
attempt should therefore be one of:

1. Build a q8_0-only Hiera-shaped MMQ body/tile variant that reduces the
   current `REG=255`/single-resident-block pressure without increasing spill.
2. Implement a parity-safe D56 FlashAttention body improvement, preserving the
   accumulation behavior that the rejected 256-thread ncols64 trial changed.
3. Recheck official-quality parity after any speed win, because bbox-only
   parity can hide small score/mask differences in attention and quantization.

Two narrow follow-up investigations were run from this state.

First, the q8_0 MMQ register-pressure hypothesis was checked with a local
live-range reduction experiment in the NVIDIA branch of
`vec_dot_q8_0_q8_1_mma`. The experiment moved the q8_0 `A` and `dA` fragment
storage closer to the `j0/k01` use site while keeping the tile geometry and
writeback contract unchanged. The first attempt accidentally compared a
full-mask run against a bbox-only baseline, so that comparison is invalid. The
correct bbox-only comparison is
`outputs/mmq-q8-live-range-jouter-q8_0-1024-20260516-1245/parity-bbox-vs-quant-ifconstexpr.json`,
which is exact: `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`.

The experiment still does not solve the profiler issue. Resource usage for the
q8_0 `mmq_x=128` main kernel remains at `REG=255` with stack use, and the
short speed rows are `56` and `55 ms/frame` after the cold row. The MMQ event
summary in
`outputs/mmq-q8-live-range-jouter-q8_0-1024-20260516-1245/profile-summary.json`
also stays in the same band as the current baseline. This means the compiler
does not free the occupancy-limiting state with simple C++ live-range
restructuring. The rejected experiment was reverted. A real q8_0 MMQ fix still
needs a new tile/body split that reduces the per-thread accumulator footprint
or changes the per-block work shape, not just moving local declarations.

Second, the CUDA D56 FlashAttention path was compared against the relevant
Metal-side staging direction. Metal has a non-padded FA KV staging path and a
direct-output fast path for the single-workgroup case. CUDA's D56 path still
funnels through `ggml_cuda_flash_attn_ext_head56_pad_mma_f16`: Q/K/V are
staged through 56-to-64 padding, then the MMA kernel writes either direct D56
or a temporary D64/D56 output depending on contiguity. The next parity-safe
FATTN experiment should therefore be staging-only: port a Metal-style
`has_kvpad`/direct-staging decision into the CUDA D56 wrapper while keeping the
existing 64-padded MMA math and accumulation order. Thread-count or ncols
tuning should remain secondary because the prior 256-thread ncols64 trial
changed tracking output.

A smaller D56 MMA-body config experiment changed only the Ampere/Blackwell
`DKQ=64, DV=64, ncols=64` config from `Q_in_reg=true` to `Q_in_reg=false`.
This does not change the thread count, ncols, or output layout. The bbox-only
tracking parity check was exact:
`outputs/fattn-qinreg-off-q8_0-1024-20260516-1320/parity-bbox-vs-baseline.json`
has `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`.

The experiment is still not worth keeping. The FATTN56 profile in
`outputs/fattn-qinreg-off-profile-q8_0-1024-20260516-1320/summary.json` shows
the main window/global MMA means improving (`0.1556 ms` for
`Q[56,196,8,25]` and `0.8196 ms` for `Q[56,4096,8,1]`), but total profiled
FATTN56 time rises to `32.626 ms` because pack-side outliers appear in the same
run. Normal bbox-only speed stays in the same band after the cold row:
`54` and `55 ms/frame` in
`outputs/fattn-qinreg-off-speed-q8_0-1024-20260516-1320/`. Since this does not
produce an end-to-end win and has only bbox-only parity evidence, the config was
reverted. The result is useful only as a hint that Q-register pressure is part
of the D56 MMA body cost; it is not a standalone optimization.

While preparing the next MMQ subtile experiment, the q8_0 B-tile load path was
audited and the previously rejected `load_ldmatrix` variant was found in the
working diff. That variant had already been measured as exact but slower, so it
was restored to `load_generic`. The smoke evidence is
`outputs/mmq-restore-bload-generic-q8_0-1024-20260516-1340/`: bbox-only parity
against the current baseline is exact (`mask_hash_equal_rows=10/10`,
`max_bbox_delta_px=0`, `max_score_abs_delta=0`), and warm speed rows are back in
the current band (`53`, `55 ms/frame`). This is not a new optimization claim; it
keeps the branch from carrying a known-regressed q8_0 micro-change before
testing deeper MMQ body work.

The deeper q8_0 MMQ live-range idea was then tested more directly by changing
the NVIDIA `vec_dot_q8_0_q8_1_mma` body to process one `k01` fragment at a time
instead of keeping both q8_0 `A` fragments and their `dA` scales live across the
outer `j0` loop. The experiment built and was bbox-only exact against the
current q8_0 Base+ 1024 baseline:
`outputs/mmq-q8-stream-k-q8_0-1024-20260516-1223/parity-bbox-vs-baseline.json`
has `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`.

It is still rejected. The resource artifact
`outputs/mmq-q8-stream-k-resource-20260516-1223/q8_0-resource.txt` shows the
dominant `mmq_x=128` main kernels still capped at `REG=255`, with stack use
increasing from the current artifact
`outputs/mmq-resource-current-q8_0-20260516-1223/q8_0-resource.txt` on some
instantiations. The short speed rows,
`outputs/mmq-q8-stream-k-speed-q8_0-1024-20260516-1223/`, were only `53.1` and
`53.5 ms/frame`, which is inside the current noise band rather than a win. This
means q8_0 register pressure is not solved by C++ declaration lifetime alone.
The next credible MMQ attempt needs to reduce the accumulator footprint or use a
different q8_0 output-tile body, not only stream the existing two K fragments
through the same MMA/update structure.

A smaller accepted D56 FlashAttention staging improvement was added for the
same-shape, non-contiguous head56 pack path. The previous combined pack path
used one CUDA thread per padded head element. The new `row4` pack kernel handles
one logical head row with 16 lanes, copies the 56 real head values in four-value
groups, and writes the eight padded values from one lane. It is guarded by
`GGML_CUDA_DISABLE_FATTN56_ROW4_PACK=1` for same-build A/B checks.

The bbox-only tracking comparison is exact against the current q8_0 Base+ 1024
baseline:
`outputs/fattn56-row4-pack-q8_0-1024-20260516-1240/parity-bbox-vs-baseline.json`
has `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`. The FATTN56 event profile improved from
`22.8957 ms` total in
`outputs/fattn56-current-profile-q8_0-1024-20260516-1240/summary.json` to
`21.5554 ms` total in
`outputs/fattn56-row4-pack-profile-q8_0-1024-20260516-1240/summary.json`.
The largest direct pack wins were the common combined shapes:

| Shape | Old pack mean | New pack mean | Old total mean | New total mean |
| --- | ---: | ---: | ---: | ---: |
| `Q[56,196,8,25]` | `0.0501 ms` | `0.0411 ms` | `0.2195 ms` | `0.2015 ms` |
| `Q[56,4096,8,1]` | `0.0363 ms` | `0.0268 ms` | `0.9332 ms` | `0.8172 ms` |
| `Q[56,49,16,25]` | `0.0229 ms` | `0.0172 ms` | `0.0684 ms` | `0.0596 ms` |

Normal same-build A/B over five alternating runs is a small but consistent
directional win:

| Mode | Evidence | Mean track ms/frame | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| row4 disabled | `outputs/fattn56-row4-pack-ab-q8_0-1024-20260516-1240/off*.log` | `53.58` | `53.5` | `0.432` |
| row4 enabled | `outputs/fattn56-row4-pack-ab-q8_0-1024-20260516-1240/on*.log` | `53.24` | `53.2` | `0.288` |

The official PyTorch quality contract was also rechecked for the same q8_0
Base+ 1024 multimask path in
`outputs/official-quality-fattn56-row4-pack-q8_0-1024-20260516/official/summary.json`.
Metadata validation passed, `mean_mask_iou=0.993873`, and
`min_mask_iou=0.992896`. The full-mask C++ row in
`outputs/official-quality-fattn56-row4-pack-q8_0-1024-20260516/cpp.log`
measured `63.2 ms/frame`; official PyTorch propagation measured
`63.0307 ms/frame` in this full-mask quality harness. This confirms row4 pack
does not create a quality regression, but it does not yet satisfy the broader
PyTorch-speed goal for the quality path.

The same row-wise pack idea was extended to the D56 mixed Q/KV shape path where
Q has fewer rows than K/V but shares the same batch dimensions. The new
`mixed_row4` kernel packs Q rows and K/V rows independently in one launch and
can be disabled with `GGML_CUDA_DISABLE_FATTN56_MIXED_ROW4_PACK=1` for A/B. The
bbox-only tracking comparison stayed exact:
`outputs/fattn56-mixed-row4-pack-q8_0-1024-20260516-1250/parity-bbox-vs-baseline.json`
has `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`.

The FATTN56 event profile improved again, from the previous row4 total
`21.5554 ms` to `21.3643 ms` in
`outputs/fattn56-mixed-row4-pack-profile-q8_0-1024-20260516-1250/summary.json`.
The mixed-shape pack wins were:

| Shape | Previous pack mean | Mixed row4 pack mean | Previous total mean | Mixed row4 total mean |
| --- | ---: | ---: | ---: | ---: |
| `Q[56,16,4,1024] K[56,64,4,1024]` | `0.4372 ms` | `0.4209 ms` | `0.7214 ms` | `0.7120 ms` |
| `Q[56,4,8,1024] K[56,16,8,1024]` | `0.2053 ms` | `0.1968 ms` | `0.4290 ms` | `0.4204 ms` |
| `Q[56,49,16,25] K[56,196,16,25]` | `0.1283 ms` | `0.1126 ms` | `0.2489 ms` | `0.2295 ms` |

Normal same-build A/B over five alternating runs is only a small directional
win because the mixed path is not the dominant D56 case:

| Mode | Evidence | Mean track ms/frame | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| mixed row4 disabled | `outputs/fattn56-mixed-row4-pack-ab-q8_0-1024-20260516-1250/off*.log` | `53.12` | `52.8` | `0.642` |
| mixed row4 enabled | `outputs/fattn56-mixed-row4-pack-ab-q8_0-1024-20260516-1250/on*.log` | `53.02` | `53.0` | `0.432` |

The official quality contract was rechecked after enabling both row4 pack paths
in
`outputs/official-quality-fattn56-mixed-row4-pack-q8_0-1024-20260516/official/summary.json`.
Metadata validation passed and the quality row stayed unchanged:
`mean_mask_iou=0.993873`, `min_mask_iou=0.992896`. The C++ full-mask row
measured `63.4 ms/frame`; the official PyTorch propagation row for that run was
`52.4452 ms/frame`, so this remains a validated quality-preserving CUDA
micro-optimization, not completion of the PyTorch-speed objective.

The next measured full-mask bottleneck was not a CUDA kernel. With profiling
enabled on the q8_0 Base+ 1024 multimask path,
`outputs/fullmask-profile-current-q8_0-1024-20260516-1300/summary.json` showed
`prop_fullmask_cleanup` taking `10.545 ms` total over two propagated frames
(`5.2725 ms` mean). That step performs connected-component hole filling and
sprinkle removal on full-resolution masks. Disabling it with
`SAM3_DISABLE_FULLMASK_CLEANUP=1` kept the official quality contract stable:
`outputs/fullmask-no-cleanup-quality-q8_0-1024-20260516/official/summary.json`
reported `mean_mask_iou=0.9938955` and `min_mask_iou=0.9929261`, which is
slightly higher than the cleanup-enabled row in this sample.

The implementation was therefore changed so full-resolution overlap resolution
still runs by default, but the expensive connected-component cleanup is opt-in
via `sam3_video_params::cleanup_full_masks`,
`sam3_visual_track_params::cleanup_full_masks`, or
`SAM3_ENABLE_FULLMASK_CLEANUP=1`. `SAM3_DISABLE_FULLMASK_CLEANUP=1` still wins
as an explicit disable switch. This is a post-processing default change, not a
CUDA kernel optimization.

The default-off validation is
`outputs/fullmask-cleanup-default-off-q8_0-1024-20260516/`. The C++ full-mask
row measured `57.6 ms/frame`, while official PyTorch propagation in the same
quality harness measured `57.7101 ms/frame`; metadata validation passed with
`mean_mask_iou=0.9938955` and `min_mask_iou=0.9929261`. A profiling smoke run at
`outputs/fullmask-cleanup-default-profile-q8_0-1024-20260516/` no longer emits
`prop_fullmask_cleanup`, while the opt-in smoke at
`outputs/fullmask-cleanup-optin-profile-q8_0-1024-20260516/` emits the cleanup
span again at about `5.3-5.6 ms` per propagated frame. The bbox-only contract
also stayed exact:
`outputs/fullmask-cleanup-bbox-parity-q8_0-1024-20260516/parity-bbox-vs-baseline.json`
has `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`.

This narrows the full-mask gap to noise-level parity for this one measured run,
but it is not yet enough evidence to claim a robust PyTorch-speed win. The next
step should be paired repeated runs of C++ default-off and official PyTorch on
the same decoded source and encode size, followed by more Hiera encode work if
the median still favors PyTorch.

An initial three-pair repeat was then run in
`outputs/fullmask-cleanup-default-off-paired-q8_0-1024-20260516/` with the same
decoded source and 1024 encode size:

| Run | C++ full-mask track ms/frame | Official PyTorch propagate ms/frame | Mean mask IoU |
| --- | ---: | ---: | ---: |
| 1 | `58.0` | `61.8253` | `0.9938955` |
| 2 | `57.3` | `57.5375` | `0.9938955` |
| 3 | `59.5` | `62.8804` | `0.9938955` |

The three-run C++ mean/median/stdev is `58.27 / 58.0 / 1.12 ms`; the official
PyTorch mean/median/stdev is `60.75 / 61.83 / 2.83 ms`. This is the first
matched full-mask set where C++ is directionally faster on average, but the
sample count is still small and PyTorch variance is high. Treat it as a
promising accepted default change, not as final proof that the broader
PyTorch-speed objective is complete across models, precisions, or videos.

The current Hiera attribution was refreshed after these changes in
`outputs/hiera-current-profile-q8_0-1024-20260516-continued/`. In a 3-frame
node-profile run, steady Hiera encode was `43.829` and `44.389 ms` after the
cold row. The synchronized hotspot ranking still points to the same two root
targets:

| Hotspot | Drop-max sum | Mean after drop-max | Priority |
| --- | ---: | ---: | --- |
| q8_0 MLP expansion `MUL_MAT dst=f32[1792,4096,1,1]` | `8.6413 ms` | `0.1839 ms` | primary MMQ body |
| q8_0 MLP projection `MUL_MAT dst=f32[448,4096,1,1]` | `7.4071 ms` | `0.1576 ms` | primary MMQ body |
| D56 window `FLASH_ATTN_EXT dst=f32[56,8,196,25]` | `7.2263 ms` | `0.2065 ms` | primary FATTN body |
| D56 global `FLASH_ATTN_EXT dst=f32[56,8,4096,1]` | `7.0415 ms` | `0.8802 ms` | primary FATTN body |
| q8_0 window QKV projection `MUL_MAT dst=f32[1344,196,25,1]` | `6.9482 ms` | `0.1985 ms` | primary MMQ body |

The FATTN56 event profile in
`outputs/hiera-current-profile-q8_0-1024-20260516-continued/fattn56/summary.json`
attributes `23.3698 ms` effective D56 work as `5.3549 ms` pack,
`17.8255 ms` MMA, and `0.1894 ms` slice. The MMQ event profile in
`outputs/hiera-current-profile-q8_0-1024-20260516-continued/mmq/summary.json`
attributes `48.5684 ms` effective q8_0 MMQ work as `7.7901 ms` activation
quantization and `40.7783 ms` MMQ body. This makes the next fusion-kernel work
more specific: small elementwise fusion is secondary; the worthwhile candidates
are q8_0 MMQ epilogue/fixup fusion that preserves parity, or a D56 attention
kernel that removes padded-MMA body overhead rather than just improving pack.

The Base+ 1024 full-mask quality matrix was also refreshed against official
PyTorch in `outputs/base-plus-quality-matrix-1024-20260516/summary.json`:

| Precision | Mean mask IoU | Min mask IoU | Metadata |
| --- | ---: | ---: | --- |
| f16 | `0.9941393` | `0.9931828` | passed |
| f32 | `0.9939986` | `0.9932173` | passed |
| q8_0 | `0.9938955` | `0.9929261` | passed |
| q4_1 | `0.9906275` | `0.9870660` | passed |
| q4_0 | `0.9838896` | `0.9802606` | passed |
| mxfp4 | `0.9191294` | `0.9162061` | passed |
| nvfp4 | `0.9633121` | `0.9582759` | passed |

For a Rust-wrapper fallback, this keeps q8_0 as the best size/speed/quality
candidate among quantized models. q4_1 is close on mean IoU but fails a strict
`min_mask_iou >= 0.99` parity bar on this clip; q4_0 and FP4 remain quality
experiments rather than fallbacks.

The fusion-kernel follow-up focused on q8_0 MMQ Stream-K bias/fixup fusion. The
important correctness rule is that a split Stream-K tile must not apply the
bias or activation in the main kernel before the later fixup has accumulated all
partial sums. The MMQ epilogue condition now only applies main-kernel
bias/activation when the output tile is complete in that block:
`kb0_start == 0 && kb0_stop == blocks_per_ne00`. This is a safety-side
correction and stayed exact against the previous default output on the q8_0
Base+ 1024 bbox-only sample:
`outputs/mmq-fixup-epilogue-fixed-q8_0-1024-20260516/parity-before-patch-baseline-vs-default.json`
has `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`.

The broader non-exact fixup fusion is still not parity-complete. Comparing the
default fusion path against `GGML_CUDA_DISABLE_MMQ_BIAS_FIXUP_FUSION=1` in
`outputs/mmq-fixup-epilogue-fixed-q8_0-1024-20260516/parity-fixup-guard-vs-default.json`
still gives `mask_hash_equal_rows=9/10` and `max_score_abs_delta=0.005139`.
The official quality row after the safety correction remains high:
`outputs/mmq-fixup-epilogue-fixed-quality-q8_0-1024-20260516/official/summary.json`
reports `mean_mask_iou=0.9938955`, `min_mask_iou=0.9929261`, and metadata
validation passed. The next fusion-kernel step should therefore be a
fixup-specific parity investigation, not broadening the fusion set.

That parity investigation found the first unsafe non-exact fixup subset. Launch
diffs in `outputs/mmq-fixup-launch-diff-q8_0-1024-20260516/` showed that the
default path added non-exact Stream-K fusion for small mask-decoder rows with
`ncols_dst=7/8`. These rows are not Hiera encode hotspots, but they affect the
initial candidate score/mask selection: the guard-vs-default comparison above
had only the first mask hash changed and a `0.005139` score delta.

The default predicate now rejects non-exact Stream-K q8_0 MMQ bias fusion when
`dst->ne[1] <= 8`, while leaving larger Hiera rows eligible. The corrected
guard-vs-default comparison is exact:
`outputs/mmq-small-fixup-guard2-q8_0-1024-20260516/parity-fixup-guard-vs-default.json`
has `mask_hash_equal_rows=10/10`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0`. The same run measured `56.9 ms/frame` with
`GGML_CUDA_DISABLE_MMQ_BIAS_FIXUP_FUSION=1` versus `52.2 ms/frame` by default,
so the Hiera-relevant fusion speed remains while the small mask-decoder rows are
kept on the exact path.

The official quality row after this small-output guard is
`outputs/mmq-small-fixup-guard2-quality-q8_0-1024-20260516/official/summary.json`:
metadata validation passed, `mean_mask_iou=0.9939710`, and
`min_mask_iou=0.9930974`. The C++ full-mask row in that run measured
`56.7 ms/frame`; official PyTorch propagation measured `62.3476 ms/frame`.
This is another matched 1024 Base+ q8_0 run where C++ is faster than official
PyTorch, but the goal still needs broader statistical coverage across videos and
model/precision fallbacks before calling the overall objective complete.

The fixed default was then measured with five paired full-mask q8_0 Base+ 1024
runs on the original `data/test_video.mp4` sample. The summary is in
`outputs/q8_0-1024-current-paired-fullmask-20260516/summary.json`.

| Metric | C++ | Official PyTorch |
| --- | ---: | ---: |
| Mean track ms/frame | `57.54` | `61.77` |
| Median track ms/frame | `57.5` | `61.92` |
| Stdev | `0.54` | `4.38` |
| Faster C++ runs | `4 / 5` | - |
| Mean mask IoU | `0.9939710` | reference |
| Min mask IoU | `0.9930974` | reference |

This is the strongest current evidence that q8_0/Base+/1024 can beat official
PyTorch on the bedroom sample under matched decoded source and SAM input size:
C++ is `4.23 ms/frame` faster on average and all metadata checks passed.

Before correcting the official PyTorch timing denominator, the same default did
not appear to generalize to a second video. The pre-correction 3-pair run on the
official SAM2 `05_default_juggle.mp4` sample with point prompt
`(710, 315)` is in
`outputs/q8_0-1024-juggle-paired-fullmask-20260516/summary.json`:

| Metric | C++ | Official PyTorch |
| --- | ---: | ---: |
| Mean track ms/frame | `61.83` | `52.91` |
| Median track ms/frame | `62.1` | `53.16` |
| Stdev | `0.74` | `4.90` |
| Faster C++ runs | `0 / 3` | - |
| Mean mask IoU | `0.9918656` | reference |
| Min mask IoU | `0.9876141` | reference |

This table is retained for history, but its official PyTorch timing column is no
longer the strict comparison contract. The corrected timing rows below supersede
it. The quality concern remains valid: the min-mask-IoU row is below a strict
`0.99` threshold on the dynamic-video sample.

The first juggle profile is
`outputs/profile-juggle-q8_0-1024-20260516/summary.json`. It shows that the
juggle miss is not simply a slower Hiera encode path: after the cold row, Hiera
encode is `39.613` and `37.916 ms`, which is the same current band as the
bedroom sample. `propagate_single` is `13.304` and `11.369 ms`, while
`prop_fullmask_active_resize_bbox` is about `0.67 ms/frame`. This means the next
cross-video work should not only chase Hiera kernels. Hiera D56/MMQ work remains
the main model-wide kernel target, but the dynamic-video gap also needs
mask-decoder candidate/selection and official-PyTorch timing comparisons before
claiming general PyTorch-speed parity.

### Official PyTorch Timing Contract Correction

`scripts/compare_sam2_official_quality.py` previously divided official
`propagate_total_ms` by all saved rows, including frame 0. That was not the same
contract as `sam3_benchmark`, where frame 0 point-prompt/add-instance is
reported separately and `Track/fr` covers only frames `1..N-1`. The script now
records `track_frames`, `timed_frame_range="1..frames-1"`, and computes
`propagate_ms_per_frame = propagate_total_ms / track_frames`.

The corrected one-run checks are:

| Sample | C++ row used | Corrected official PyTorch |
| --- | ---: | ---: |
| `data/test_video.mp4` | existing run1 C++ JSONL | `61.911 ms/frame`, `track_frames=9` |
| `05_default_juggle.mp4` | existing run1 C++ JSONL | `56.846 ms/frame`, `track_frames=9` |

Artifacts:

- `outputs/q8_0-1024-current-paired-fullmask-corrected-20260516/run1/official/summary.json`
- `outputs/q8_0-1024-juggle-paired-fullmask-corrected-20260516/run1/official/summary.json`

This does not close the goal by itself. It makes the speed comparison stricter
and removes the short-video denominator bias before more fusion-kernel work is
judged.

After rerunning official PyTorch for every existing paired C++ JSONL, the
corrected paired summaries are:

| Sample | Runs | C++ mean / median / stdev | PyTorch mean / median / stdev | Faster C++ runs | Mean mask IoU | Min mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/test_video.mp4` | 5 | `57.54 / 57.50 / 0.54` | `61.32 / 61.91 / 3.14` | `4 / 5` | `0.9939710` | `0.9930974` |
| `05_default_juggle.mp4` | 3 | `61.83 / 62.10 / 0.74` | `66.17 / 67.96 / 8.57` | `2 / 3` | `0.9918656` | `0.9876141` |

Artifacts:

- `outputs/q8_0-1024-current-paired-fullmask-corrected-20260516/summary.json`
- `outputs/q8_0-1024-juggle-paired-fullmask-corrected-20260516/summary.json`

This changes the speed conclusion: under the corrected frame-range contract,
q8_0/Base+/1024 is now faster than official PyTorch on both checked paired
samples on mean, though the juggle sample has high PyTorch variance. It still
does not satisfy the full fallback goal because juggle min mask IoU remains
`0.9876141`, below the strict `0.99` quality bar.

### Fusion-Kernel Direction Check

D56 FlashAttention already has useful fusion/staging pieces in this branch:
combined Q/K/V pack and direct 56-wide output. A small A/B on the 1024 Base+
q8_0 path showed:

- `GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` is not worth enabling by default. It
  avoids V padding, but the D56 MMA path gets slower; global D56 attention
  moved from roughly `0.88 ms` to `1.29 ms` in the profiled row.
- `GGML_CUDA_DISABLE_FATTN56_DIRECT_OUT=1` is also not a clear improvement.
  Five normal bbox runs were about `53.5 ms/frame`, while the current default
  was about `52.8 ms/frame` under the same short benchmark. The direct-output
  default stays enabled.

The next viable fusion-kernel work is therefore not toggling these D56 staging
flags. It should be a real CUDA kernel change in one of two areas:

1. D56 FlashAttention MMA body: reduce the padded-64 math cost while preserving
   the current combined-pack/direct-output contract.
2. q8_0 MMQ body: reduce register pressure or split output subtile accumulation
   without changing Stream-K fixup semantics.

A q8_0 MMQ live-range experiment was tried by streaming one K fragment at a
time in the NVIDIA q8_0 MMA path. It preserved bbox parity against
`outputs/mmq-small-fixup-guard2-q8_0-1024-20260516/default/run.jsonl`
(`mask_hash_equal_rows=10/10`, zero bbox and score deltas), but it regressed
the short 1024 bbox benchmark from about `52.8` to `53.5 ms/frame`. That patch
was reverted; it is useful evidence that live-range reduction alone is not the
right first fusion change for q8_0 on this GPU.

### Rejected D56 Native-QK Fusion Probe

A more aggressive D56 experiment tried to avoid Q/K padding by routing the
head-56 path through a temporary `DKQ=56, DV=64, DV_DST=56` MMA variant under
`GGML_CUDA_ENABLE_FATTN56_NATIVE_QK=1`. This compiled, but it was rejected and
reverted because it broke bbox parity immediately. Evidence is in
`outputs/fattn56-native-qk-20260516/parity-default-vs-native-qk.json`: the
4-frame smoke comparison had `mask_hash_equal_rows=0/4`, frame 0 bbox IoU `0`,
and subsequent rows were missing detections. The profile also did not justify
repairing this route first: window D56 rows were slightly faster in MMA body,
but global D56 regressed to about `1.19 ms` versus the default band around
`0.85-0.88 ms`, and separate Q/K/V packing roughly doubled the window pack cost.

This narrows the viable fusion-kernel direction: do not simply instantiate the
existing MMA template at `DKQ=56`. A correct D56 kernel needs a dedicated KQ/VKQ
layout and writeback design, or it should keep the current padded-64 math and
focus on reducing pack/slice overhead around the proven parity-safe path.

### Current Fusion Bundle A/B

The current CUDA fusion bundle was rechecked as a normal same-binary A/B on
`sam2.1_hiera_base_plus_q8_0`, 1024 input, 10-frame bbox-only tracking, using
the same `../sam2/notebooks/videos/bedroom.mp4` source. The disabled side set:

- `GGML_CUDA_DISABLE_NORM_FUSION=1`
- `GGML_CUDA_DISABLE_ADD_NORM_FUSION=1`
- `GGML_CUDA_DISABLE_ADD_UNARY_FUSION=1`
- `GGML_CUDA_DISABLE_BIN_BCAST_AXIS_FAST=1`
- `GGML_CUDA_DISABLE_BIN_CONTIGUOUS_FAST=1`
- `GGML_CUDA_DISABLE_MMQ_BIAS_FUSION=1`
- `GGML_CUDA_DISABLE_MMQ_BIAS_GELU_FUSION=1`

The measured bbox-only result is:

| Path | Track ms/frame runs | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| Fusion enabled | `54.0, 53.0, 54.0, 53.0, 53.0` | `53.4` | `53.0` | `0.55` |
| Fusion disabled | `58.0, 59.0, 58.0, 58.0, 58.0` | `58.2` | `58.0` | `0.45` |

The paired disabled-minus-enabled delta is
`4.0, 6.0, 4.0, 5.0, 5.0 ms/frame`, mean `4.8 ms/frame`. This is about an
`8.25%` reduction versus the disabled-fusion path for this specific q8_0 Base+
run. Evidence is in
`outputs/fusion-kernel-current-20260516/ab/summary.json`.

Full-mask parity against the disabled-fusion path remains bit-identical on the
same 10-frame sample:

| Metric | Value |
| --- | ---: |
| Rows | `10 vs 10` |
| Mask hash equal rows | `10/10` |
| Minimum bbox IoU | `1.0` |
| Maximum bbox delta | `0.0 px` |
| Maximum score delta | `0.0` |
| Maximum mask-area relative delta | `0.0` |

Evidence is in `outputs/fusion-kernel-current-20260516/parity/summary.json`.

`GGML_CUDA_PROFILE_NODES=1` confirms that the bundle is not only theoretically
matched: fused `NORM` nodes, fused `ADD -> NORM -> MUL -> ADD` nodes, fused
`ADD + GELU` nodes, and fused `MUL_MAT + ADD`/activation MMQ rows all appear in
the q8_0 profile. The profile artifact is
`outputs/fusion-kernel-current-20260516/profile-enabled.log`. Because the node
profiler synchronizes per node and disables CUDA graphs, these profile timings
are only for attribution; the normal A/B above is the speed evidence.

This keeps the next fusion-kernel direction focused on deeper kernel work:
the safe graph-level fusions are already buying roughly `5 ms/frame` on this
sample, so remaining gains should come from the D56 FlashAttention body or MMQ
arithmetic/tiling rather than adding more launch-only fusions.

### Upstream / Metal Fusion Candidate Triage

Two small-scope audits were run against the current dirty CUDA tree:

- ggml history candidate `8498d0f6` adds FlashAttention MMA/tile support for
  `DKQ=192,DV=128` MiMo-V2.5 shapes. It is not applied here because SAM2/SAM3
  Hiera Base+ uses head dimension `56`, and the current active path is the
  dedicated head56 pad-to-64 MMA route. Pulling the MiMo templates into this
  branch would increase compile surface without touching the profiled Hiera
  hotspot.
- Metal still has explicit `flash_attn_ext_pad` and `flash_attn_ext_blk`
  staging helpers. CUDA's head56 path already has bespoke Q/K/V pad/pack and
  direct-output logic, so the next experiment should not replace the whole
  attention route. The first safe test is to port only the staging helper
  behavior behind the existing `GGML_OP_FLASH_ATTN_EXT` dispatch, keep the MMA
  body unchanged, and require full-mask hash parity against current CUDA.
- Metal's vector reduce helper is a lower-priority candidate because changing
  reduction order can legitimately perturb softmax results. It should be tried
  only behind an env gate and judged by direct attention tests before model
  benchmarks.

As a safety-only follow-up from upstream commit `1b6112bf`, CUDA
`supports_op` now rejects `ADD/SUB/MUL/DIV` unless both inputs and the output
are F32 or F16. This mirrors the actual CUDA binary broadcast dispatch surface
and prevents broader fusion/support checks from routing unsupported integer or
other typed binary ops to CUDA. It is not a speed change. After this hardening,
`just build-target sam3_benchmark` passed and the same q8_0 Base+ full-mask
fusion parity check remained exact:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, and
`max_score_abs_delta=0.0`. Evidence:
`outputs/fusion-kernel-current-20260516/support-op-hardening-parity/summary.json`.

### Dynamic-Video Mask Gap Diagnostics

The remaining Base+ fallback blocker is quality, not speed. On the dynamic
`05_default_juggle.mp4` sample, even C++ f32 compared against official PyTorch
fp32 has low rows:

- mean mask IoU: `0.9927882`
- min mask IoU: `0.9871384`
- lowest rows: offsets `7` and `8`
- selected mask index matches official on those lowest rows

The saved-mask diagnostics show this is not frame order drift and not a simple
spatial offset. Searching C++ mask shifts in `[-4,+4]` pixels against official
fp32 masks kept the best shift at `(dx=0, dy=0)` for every frame. The low rows
are boundary-only differences: all XOR pixels are within one pixel of the
combined mask boundary. At the worst rows:

| Offset | IoU | C++ only px | Official only px | XOR px | Best shift |
| ---: | ---: | ---: | ---: | ---: | --- |
| `7` | `0.9875697` | `73` | `217` | `290` | `(0, 0)` |
| `8` | `0.9871384` | `57` | `248` | `305` | `(0, 0)` |
| `9` | `0.9902140` | `33` | `202` | `235` | `(0, 0)` |

Artifacts:

- `outputs/juggle-quality-f32-1024-fp32official-20260516/shift_diagnostics.json`
- `outputs/juggle-quality-f32-1024-fp32official-20260516/boundary_diagnostics.json`

To test whether this is only a threshold convention mismatch, the benchmark now
has a diagnostic-only `--output-logits-dir` option that writes the selected
low-res logits as `.bin/.shape`. Propagation logits are copied into
`sam3_detection` only when this option enables `SAM3_CAPTURE_PROP_LOGITS`, so
normal benchmark timing is not affected.

Using those logits on the same C++ f32 run, resizing with PyTorch
`interpolate(..., align_corners=False)` and sweeping the threshold showed that
threshold tuning alone does not close the gap:

| Threshold | Mean IoU vs official fp32 | Min IoU vs official fp32 |
| ---: | ---: | ---: |
| `-0.20` | `0.9929357` | `0.9884586` |
| `-0.15` | `0.9929818` | `0.9883651` |
| `-0.10` | `0.9929556` | `0.9881436` |
| `0.00` | `0.9927882` | `0.9871384` |

Artifact:
`outputs/juggle-logit-diagnostics-f32-1024-20260516b/threshold_sweep_vs_official_fp32.json`.

This narrows the quality work: the mismatch is a small boundary-level logits
difference accumulated by propagation/memory, not a decoded-frame ordering
problem, not a full-mask resize coordinate bug, and not fixable by a global
threshold offset. The next useful quality check is to dump official high-res
logits for the same frames and compare signed logit deltas near the boundary,
then trace the first frame where C++ and official logits diverge materially.

Two official-Python fallback toggles were checked to make sure this is not a
missing high-level SAM2 inference option:

- Setting `predictor.fill_hole_area = 0` after construction did not change the
  f32/Base+/1024 juggle comparison. The result stayed at mean mask IoU
  `0.9927882` and min mask IoU `0.9871384`.
- Setting `predictor.binarize_mask_from_pts_for_mem_enc = False` made the
  comparison worse: mean mask IoU `0.9907328`, min mask IoU `0.9845485`, and
  propagation candidate selection dropped to `6/9` matching frames.

Artifacts:

- `outputs/juggle-quality-f32-1024-fp32official-fill0-direct-20260516/summary.json`
- `outputs/juggle-quality-f32-1024-fp32official-binarizefalse-20260516/summary.json`

This confirms that C++'s conditional-frame memory binarization matches the
official default direction. The remaining quality gap is not explained by
official low-res hole filling or by disabling point-frame memory binarization.

The official PyTorch comparison script now has `--save-python-logits`, which
stores the selected official mask logits next to the saved official masks. The
reusable diagnostic script `scripts/compare_mask_logits.py` compares those
official logits with C++ `--output-logits-dir` dumps after resizing the C++
logits with PyTorch bilinear interpolation and `align_corners=False`.

For the same f32/Base+/1024 juggle run, the signed logit comparison confirms
that the low rows are not explained by threshold convention alone. The worst
frames have many more official-only boundary pixels than C++-only pixels, and
the official-only pixels are consistently negative in `cpp - official`:

| Offset | IoU | XOR px | C++ only px | Official only px | XOR mean delta | Official-only mean delta | Boundary p95 abs delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `0.9871384` | `305` | `57` | `248` | `-0.9167` | `-1.3482` | `2.8437` |
| `7` | `0.9875697` | `290` | `73` | `217` | `-0.6621` | `-1.2925` | `2.8098` |
| `9` | `0.9902140` | `235` | `33` | `202` | `-1.2997` | `-1.7434` | `2.7706` |

Artifact:
`outputs/juggle-logit-diagnostics-f32-1024-20260516b/logit_delta_vs_official_fp32.json`.

This also changes how fusion-kernel work should be accepted. A deeper CUDA
fusion is allowed only if it preserves the existing full-mask parity against the
current C++ default and does not worsen this official-logit boundary diagnostic.
The priority remains:

1. Keep accepted graph-level CUDA fusions enabled; they save about
   `4.8 ms/frame` on the q8_0/Base+/1024 A/B and are exact against the disabled
   fusion baseline.
2. For new kernel fusion, target D56 FlashAttention staging/body or q8_0 MMQ
   arithmetic where the profiler shows real time, not extra launch-only fusions.
3. Gate every experimental kernel behind an environment switch until both
   current-C++ parity and official-logit boundary diagnostics are clean.

### 2026-05-16 Current q8_0 Kernel Profile Refresh

The current q8_0/Base+/1024 profile was refreshed with all CUDA attribution
probes enabled:

```bash
GGML_CUDA_PROFILE_NODES=1 \
GGML_CUDA_PROFILE_FATTN56=1 \
GGML_CUDA_PROFILE_MMQ=1 \
build/xmake-release-cuda/examples/sam3_benchmark \
  --models-dir models/matrix-all \
  --video data/test_video.mp4 \
  --gpu-only \
  --filter sam2.1_hiera_base_plus_q8_0 \
  --point-x 315 \
  --point-y 250 \
  --n-frames 10 \
  --bbox-only \
  --multimask \
  --no-isolation \
  --quiet
```

The profiler synchronizes per node and disables CUDA graphs, so the benchmark
row itself (`track/fr=62.6 ms`) is only an attribution run. The relevant
artifacts are:

- `outputs/current-profile-q8_0-1024-20260516/profile.log`
- `outputs/current-profile-q8_0-1024-20260516/benchmark-summary.json`
- `outputs/current-profile-q8_0-1024-20260516/hotspots-all.json`
- `outputs/current-profile-q8_0-1024-20260516/fattn56-summary.json`
- `outputs/current-profile-q8_0-1024-20260516/mmq-summary.json`

The refreshed node hotspot order is still split between q8_0 MMQ and D56
FlashAttention:

| Rank | Class | Shape / op | Count | Drop-max sum ms |
| ---: | --- | --- | ---: | ---: |
| 1 | MMQ MLP expansion | `MUL_MAT dst=f32[1792,4096,1,1]` | `160` | `30.31` |
| 2 | MMQ MLP projection | `MUL_MAT dst=f32[448,4096,1,1]` | `160` | `26.36` |
| 3 | D56 global attention | `FLASH_ATTN_EXT dst=f32[56,8,4096,1]` | `30` | `25.15` |
| 4 | D56 window attention | `FLASH_ATTN_EXT dst=f32[56,8,196,25]` | `120` | `25.00` |
| 5 | MMQ window QKV | `MUL_MAT dst=f32[1344,196,25,1]` | `120` | `24.05` |

The dedicated FATTN56 event profile, after dropping the first row per shape,
shows total D56 attention attribution of `76.68 ms` across the profiled run:

| D56 shape | Effective rows | Pack sum | MMA sum | Slice sum | Total sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| global `4096 x 8` | `29` | `0.78` | `23.94` | `0.08` | `24.80` |
| window `196 x 25 x 8` | `119` | `4.61` | `18.56` | `0.35` | `23.52` |
| early `64 x 2 x 1024` | `19` | `5.02` | `6.23` | `0.05` | `11.30` |

The MMQ event profile, after the same first-row drop, shows total MMQ
attribution of `171.50 ms`, with `145.26 ms` in the MMQ body and `26.24 ms` in
src1 quantization. Top MMQ shapes are:

| MMQ shape | Effective rows | Quant sum | MMQ sum | Total sum |
| --- | ---: | ---: | ---: | ---: |
| `q8_0[448,1792] x f32[448,4096] -> f32[1792,4096]` | `159` | `1.71` | `26.72` | `28.42` |
| `q8_0[1792,448] x f32[1792,4096] -> f32[448,4096]` | `159` | `4.34` | `20.17` | `24.52` |
| `q8_0[448,1344] x f32[448,196,25] -> f32[1344,196,25]` | `119` | `1.45` | `21.22` | `22.66` |

This refresh changes the immediate kernel-fusion priority:

1. q8_0 MMQ body/quantization is the larger aggregate target than D56 FATTN in
   the current Base+ 1024 run.
2. D56 FATTN remains worth improving, but global attention is already almost
   entirely MMA body time and slice is negligible. Window attention still has
   measurable pack overhead, but its aggregate is similar to one MMQ MLP block
   class.
3. Additional graph-level bias/GELU launch fusion is not the main route. The
   hot `src0=none` MUL_MAT rows are already fused-profile rows, and prior
   channel-bias experiments showed that stream-k exactness and mask quality must
   be guarded carefully.

The next concrete kernel experiment should therefore start from q8_0 MMQ:
profile and reduce src1 quantization or MMQ body work for the three hot shapes
above, behind an environment gate, then run current-C++ full-mask parity and the
official-logit boundary diagnostic before considering it as a default.

A first stream-k scheduling probe tried lowering the MMQ tiled-launch
efficiency threshold from `90%` to `0%`, forcing more low-efficiency shapes to
launch one block per tile and avoid the fixup kernel. This was rejected:

| Path | Track ms/frame runs | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| Default `90%` threshold | `54.0, 55.0, 54.0, 53.0, 53.0` | `53.8` | `54.0` | `0.84` |
| Forced tiled launch | `54.0, 53.0, 54.0, 53.0, 54.0` | `53.6` | `54.0` | `0.55` |

The apparent `0.2 ms/frame` mean difference is noise, and full-mask parity
against the default path failed (`mask_hash_equal_rows=0/10`,
`max_bbox_delta_px=26`, `max_score_abs_delta=0.016325`). The temporary env-gated
code was removed. Evidence:

- `outputs/mmq-stream-k-threshold-q8_0-1024-20260516/summary.json`
- `outputs/mmq-stream-k-threshold-q8_0-1024-20260516/parity/summary.json`

This confirms that stream-k scheduling is part of the numeric contract for this
model. Future MMQ work should preserve the existing stream-k schedule unless it
also proves exact current-C++ mask parity and official-logit boundary stability.

The existing `GGML_CUDA_MMQ_X_MAX` retuning hook was also checked for q8_0
Base+/1024. Lowering the maximum tile width from the default `128` to `112` or
`96` did not produce a meaningful normal-runtime speedup:

| Path | Track ms/frame runs | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| Default | `54.0, 52.0, 53.0, 54.0, 53.0` | `53.2` | `53.0` | `0.84` |
| `GGML_CUDA_MMQ_X_MAX=112` | `53.0, 53.0, 53.0, 53.0, 53.0` | `53.0` | `53.0` | `0.00` |
| `GGML_CUDA_MMQ_X_MAX=96` | `53.0, 53.0, 53.0, 53.0, 53.0` | `53.0` | `53.0` | `0.00` |

The representative `x96` full-mask parity check also failed against the default
path (`mask_hash_equal_rows=0/10`, `max_bbox_delta_px=25`,
`max_score_abs_delta=0.011876`). This is not a viable optimization route.
Evidence:

- `outputs/mmq-xmax-q8_0-1024-20260516/summary.json`
- `outputs/mmq-xmax-q8_0-1024-20260516/parity/summary-x96.json`

## 2026-05-16 Fusion Kernel Probe

An MMF-side `MUL_MAT + row bias + GELU` epilogue was implemented as an
experiment for non-quantized f16/f32 paths. It is not enabled by default; use
`GGML_CUDA_ENABLE_MMF_BIAS_GELU_FUSION=1` to test it. The q8_0 route already
uses the MMQ-side epilogue machinery, so this probe is only expected to matter
for non-quantized models if their hot matmuls actually select the custom MMF
kernel instead of cuBLAS.

The Base+ f16 1024 smoke is exact but not faster:

| Path | Track ms/frame runs | Mean | Result |
| --- | --- | ---: | --- |
| Opt-in MMF bias+GELU | `63.0, 62.4, 61.9, 62.7, 61.8` | `62.36` | no speed win |
| Default path | `61.9, 62.3, 62.2, 62.7, 62.2` | `62.26` | baseline |

The opt-in-vs-default parity check was exact across 10 frames
(`mask_hash_equal_rows=10`, `min_bbox_iou=1.0`, zero bbox/score deltas).
Because the measured delta is effectively noise and slightly negative, this
fusion remains opt-in. The result reinforces the current priority: pursue
cuBLAS/cuBLASLt epilogues for large f16/f32 matmuls, or continue with q8_0 MMQ
body/quantization kernels, rather than adding more small graph-level epilogues.
Evidence:

- `outputs/mmf-bias-gelu-fusion-20260516/ab/summary.json`
- `outputs/mmf-bias-gelu-fusion-20260516/optin/parity-summary.json`

Two q8_0 MMQ body probes were then checked and rejected.

The first probe capped the Base+ window-attention shape
`q8_0[448,1344] x f32[448,196,25] -> f32[1344,196,25]` to `mmq_x=64` behind
`GGML_CUDA_ENABLE_MMQ_Q8_0_196_X64=1`. It reduced the single 10-frame smoke
from `54.7 ms/frame` to `53.4 ms/frame`, but full-mask parity against the
default path failed:

| Probe | Track ms/frame | P50 | P95 | Parity result |
| --- | ---: | ---: | ---: | --- |
| Default | `54.7` | `53.4` | `64.1` | baseline |
| `GGML_CUDA_ENABLE_MMQ_Q8_0_196_X64=1` | `53.4` | `51.4` | `62.1` | `0 / 10` mask hashes, max bbox delta `1 px`, max score delta `0.004305` |

This remains an opt-in diagnostic only. It is not defaultable because the
project's acceptance gate requires current-C++ full-mask parity before a kernel
scheduling change can be treated as a speedup.

The second probe tried to reduce the NVIDIA q8_0 MMA live range in two forms:
fully streaming A/dA by `k01`, and a smaller dA-only variant. Both built and
were exact when compared against a matched `--bbox-only --multimask` default
run (`10 / 10` mask hashes, zero bbox/score/area deltas), but neither showed a
useful single-run speed signal. The full streaming form measured
`53.1 ms/frame` and the dA-only form measured `54.0 ms/frame`, while the matched
default after reverting measured `52.2 ms/frame`. The code was reverted because
this is at best noise and likely a regression, despite preserving parity.
Evidence:

- `outputs/mmq-q8-196-x64-20260516/default.log`
- `outputs/mmq-q8-196-x64-20260516/enabled.log`
- `outputs/mmq-q8-196-x64-20260516/parity-summary.json`
- `outputs/mmq-q8-live-range-20260516/parity-vs-matched-default.json`
- `outputs/mmq-q8-live-range-da-only-20260516/parity-vs-matched-default.json`
- `outputs/mmq-q8-live-range-bbox-multimask-default-20260516/default.log`

The practical conclusion is that q8_0 MMQ is still the right aggregate target,
but simple tile-width and live-range rewrites are not sufficient. The next q8_0
MMQ work should start from a standalone microbenchmark or diagnostic kernel
that can separate register pressure, occupancy, and memory traffic for one hot
shape before it is wired into the full graph.

A standalone MMQ microbenchmark was added as `sam3_mmq_bench` to make that
next step measurable outside the full Hiera graph. The default shape matches
the largest Base+ q8_0 MLP expansion row:
`src0=q8_0[448,1792,1,1]`, `src1=f32[448,4096,1,1]`,
`dst=f32[1792,4096,1,1]`. It can also build the epilogue pattern
`MUL_MAT + bias + GELU`, which lets future fusion-kernel changes be accepted
or rejected on one hot shape before running the full video benchmark.

Example commands:

```bash
just build-target sam3_mmq_bench
build/xmake-release-cuda/examples/sam3_mmq_bench \
  --k 448 --rows 1792 --cols 4096 --iters 50 --warmup 5 --bias --gelu
GGML_CUDA_DISABLE_MMQ_BIAS_GELU_FUSION=1 \
  build/xmake-release-cuda/examples/sam3_mmq_bench \
  --k 448 --rows 1792 --cols 4096 --iters 50 --warmup 5 --bias --gelu
```

On the current RTX 5070 Ti Laptop GPU run, the isolated q8_0 hot shape measured
`0.176812 ms` with the existing MMQ bias/GELU epilogue fusion and
`0.218860 ms` with that fusion disabled. That is a `19.21%` isolated reduction
for this single node shape. The MMQ event profile, dropping the first row per
shape, reports `0.163988 ms` fused vs `0.168381 ms` disabled for the MMQ node
itself; the larger wall-clock difference comes from removing the separate ADD
and GELU graph nodes.

Evidence:

- `examples/mmq_bench.cpp`
- `outputs/mmq-microbench-fusion-q8_0-20260516/default-wall.log`
- `outputs/mmq-microbench-fusion-q8_0-20260516/disabled-wall.log`
- `outputs/mmq-microbench-fusion-q8_0-20260516/default-summary.json`
- `outputs/mmq-microbench-fusion-q8_0-20260516/disabled-summary.json`

An attempted follow-up to template-specialize the MMQ writeback activation was
not kept. The idea was to remove the per-element runtime activation switch for
the fused `bias + GELU` path, but it expanded CUDA MMQ template instantiations
enough that the rebuild did not complete in a practical time window. That makes
it a poor default direction unless it is narrowed to only the few Hiera
q8_0/q4_* hot specializations instead of all MMQ types and tile widths.

The refreshed Base+ q8_0 1024 comparison is still not a completed PyTorch
speed win for the full tracker. The C++ run measured `52.3 ms/frame` over the
same 10-frame source segment and 1024 encode size, while the official PyTorch
row measured `42.6667 ms/frame` over the 9 propagated frames. The comparable
ratio is `0.8158` Python/C++, so the full C++ path remains slower even though
the Hiera encoder itself is now ahead in the steady-state profile.

Current stage profile, with the first Hiera frame kept visible:

| Stage | Values / summary |
| --- | --- |
| `hiera_encode` | `[124.399, 37.172, 37.396, 37.602, 36.384, 35.734, 36.166, 36.114, 36.016, 35.806] ms`; steady mean excluding first: `36.49 ms` |
| `propagate_single` | mean `10.83 ms`, values `[12.303, 10.644, 10.595, 10.479, 10.683, 10.690, 10.724, 10.607, 10.753] ms` |
| `memory_encode` | `2.134 ms` |
| `unlabeled` | `14.827 ms` |

Evidence:

- `outputs/model-matrix-current-q8_0-1024-refresh-20260516/summary.json`
- `outputs/stage-profile-current-q8_0-only-1024-20260516/summary.json`

The propagate node profile points away from MMQ epilogues and toward D256
attention/memory-tracker work as the next full-tracker bottleneck. With node
profiling enabled, the largest drop-max propagate rows are:

| Pattern | Drop-max sum |
| --- | ---: |
| `FLASH_ATTN_EXT dst=f32[256,1,4096,1] src1=f32[256,4100,1,1]` | `4.852672 ms` |
| `FLASH_ATTN_EXT dst=f32[256,1,4096,1] src1=f32[256,4096,1,1]` | `4.833792 ms` |
| `MUL_MAT dst=f32[256,4096,1,1]` | `3.948192 ms` |
| `MUL dst=f32[1,128,4096,1]` | `1.827904 ms` |

A D256 FlashAttention config probe that changed the Ampere case from
`nthreads=128, ncols=2` to `nthreads=256, ncols=1` was rejected. It passed the
standalone D256 parity check
(`max_abs=8.94954428e-06`, `mean_abs=1.3194111e-06`, `0/2048` bad samples),
but made the profiled propagate stage worse: `15.6783 ms` baseline mean vs
`16.2193 ms` with the probe, and the two D256 FlashAttention drop-max sums
grew from `4.852672 / 4.833792 ms` to `6.159456 / 6.130304 ms`.

Evidence:

- `outputs/prop-node-profile-current-q8_0-1024-20260516/prop-hotspots.json`
- `outputs/fattn256-nthreads256-q8_0-1024-20260516/summary.json`
- `outputs/fattn256-nthreads256-q8_0-1024-20260516/prop-hotspots.json`

The fusion-kernel direction remains useful, but the next fusion should not be a
blind MMQ specialization. The already accepted MMQ `bias + GELU` epilogue
removes separate `ADD`/`GELU` nodes for the quantized MLP hot shape and is worth
keeping. The remaining high-value fusion candidate is the cuBLAS path's
unfused F32 `MUL_MAT + broadcast bias` rows in Hiera, for example
`f32[112,336] x f32[112,64,1024] -> f32[336,64,1024]` followed by
`bias=f32[336,1,1,1]`. These rows are not covered by the current MMF/MMQ
fusion path because they go through the batched cuBLAS implementation. A
cuBLASLt bias epilogue or a narrow custom F32 batched-matmul+bias kernel should
therefore be evaluated before more MMQ body rewrites.

A first cuBLASLt `EPILOGUE_BIAS` probe was implemented behind
`GGML_CUDA_ENABLE_CUBLASLT_BIAS_FUSION=1` and then removed. The first version
failed at the cuBLASLt `BATCH_COUNT` layout attribute because the API expects an
`int32_t` value. After fixing that, the opt-in path ran but did not preserve
full-mask parity against the default CUDA path on a 5-frame Base+ q8_0 1024
smoke:

| Metric | Result |
| --- | ---: |
| mask-hash equal rows | `4 / 5` |
| min bbox IoU | `0.8813666561` |
| max bbox delta | `93.75 px` |
| max score delta | `0.002084` |
| max mask-area relative delta | `0.0075829` |

This likely reflects changed GEMM/bias rounding or algorithm choice becoming
large enough to perturb the initial selected mask. Since the acceptance gate
requires full-mask parity for kernel changes, the cuBLASLt epilogue code was
not kept. A future retry needs a dedicated numeric parity harness for the exact
batched F32 shapes before wiring it into graph fusion, and should compare
strict-FP32, TF32, and cuBLAS pointer-array baselines directly.

Evidence:

- `outputs/cublaslt-bias-fusion-q8_0-1024-20260516/default.log`
- `outputs/cublaslt-bias-fusion-q8_0-1024-20260516/enabled.log`
- `outputs/cublaslt-bias-fusion-q8_0-1024-20260516/parity.json`

That numeric harness now exists as `sam3_f32_batched_bench`. It isolates the
Hiera-style F32 pattern
`src0=f32[k,rows]`, `src1=f32[k,cols,batches]`, optional
`bias=f32[rows]`, and `dst=f32[rows,cols,batches]`. The small CUDA check
passes exactly against the sampled C++ reference:

```bash
build/xmake-release-cuda/examples/sam3_f32_batched_bench \
  --k 16 --rows 32 --cols 8 --batches 4 --iters 3 --warmup 1 \
  --check --check-samples 1024
```

Result: `max_abs=0`, `mean_abs=0`, `bad=0/1024`.

The current baseline measurements show that the unfused broadcast bias is a
real cost center:

| Shape | No bias | With bias | Delta |
| --- | ---: | ---: | ---: |
| `k=112 rows=112 cols=64 batches=1024` | `0.126434 ms` | `0.303927 ms` | `+0.177493 ms` |
| `k=112 rows=336 cols=64 batches=1024` | `0.356670 ms` | `0.803976 ms` | `+0.447306 ms` |
| `k=112 rows=672 cols=64 batches=1024` | `0.554498 ms` | `1.273863 ms` | `+0.719365 ms` |
| `k=112 rows=256 cols=65536 batches=1` | `0.208288 ms` | `0.482477 ms` | `+0.274189 ms` |
| `k=448 rows=1792 cols=4096 batches=1` | `0.240610 ms` | `0.300449 ms` | `+0.059839 ms` |

This changes the next fusion-kernel requirement: the target is not merely
"call cuBLASLt with an epilogue". The accepted implementation must match the
current cuBLAS pointer-array plus broadcast-add numerics closely enough to keep
full-mask parity. If cuBLASLt cannot do that under strict settings, the next
candidate should be a narrow custom F32 batched-matmul+bias path or a post-GEMM
bias kernel that preserves the current GEMM output and reduces only the
broadcast-add overhead.

A post-GEMM in-place bias probe was also tried behind
`GGML_CUDA_ENABLE_CUBLAS_BIAS_INPLACE_FUSION=1` and then removed. This variant
kept the current cuBLAS GEMM path, wrote the GEMM output directly into the
final ADD tensor, then applied a narrow in-place axis-0 F32 bias kernel. It
preserved full tracker parity on the Base+ q8_0 1024 10-frame smoke
(`10 / 10` mask hashes, zero bbox/score/area deltas), but the isolated
microbenchmark did not improve the hot shape: the fused profile row for
`k=112 rows=336 cols=64 batches=1024` still measured about `0.804 ms`, matching
the existing `MUL_MAT + ADD` graph. The full tracker A/B runs also stayed within
noise (`52..53 ms/frame` for both default and opt-in). Since this does not move
the performance target, it was not kept.

Evidence:

- `examples/f32_batched_bench.cpp`
- `outputs/cublas-bias-inplace-fusion-q8_0-1024-20260516/parity.json`
- `outputs/cublas-bias-inplace-fusion-q8_0-1024-20260516/default-1.log`
- `outputs/cublas-bias-inplace-fusion-q8_0-1024-20260516/default-2.log`
- `outputs/cublas-bias-inplace-fusion-q8_0-1024-20260516/default-3.log`
- `outputs/cublas-bias-inplace-fusion-q8_0-1024-20260516/enabled-1.log`
- `outputs/cublas-bias-inplace-fusion-q8_0-1024-20260516/enabled-2.log`
- `outputs/cublas-bias-inplace-fusion-q8_0-1024-20260516/enabled-3.log`
- `outputs/f32-batched-bench-20260516/no-bias.log`
- `outputs/f32-batched-bench-20260516/bias.log`
- `outputs/f32-batched-bench-20260516/inplace-profile.log`
- `outputs/f32-batched-bench-20260516/k112_r112_c64_b1024-no-bias.log`
- `outputs/f32-batched-bench-20260516/k112_r112_c64_b1024-bias.log`
- `outputs/f32-batched-bench-20260516/k112_r672_c64_b1024-no-bias.log`
- `outputs/f32-batched-bench-20260516/k112_r672_c64_b1024-bias.log`
- `outputs/f32-batched-bench-20260516/k112_r256_c65536_b1-no-bias.log`
- `outputs/f32-batched-bench-20260516/k112_r256_c65536_b1-bias.log`
- `outputs/f32-batched-bench-20260516/k448_r1792_c4096_b1-no-bias.log`
- `outputs/f32-batched-bench-20260516/k448_r1792_c4096_b1-bias.log`

A direct custom CUDA fused F32 batched-matmul+bias probe was added to
`sam3_f32_batched_bench` behind `--custom-cuda`. This path does not touch the
main ggml graph; it exists to test whether a narrow Hiera-style custom kernel is
even a plausible replacement for the current cuBLAS-plus-ADD route before
wiring a new ggml kernel into production. The small correctness check is exact
against the sampled C++ reference:

```bash
build/xmake-release-cuda/examples/sam3_f32_batched_bench \
  --custom-cuda --k 16 --rows 32 --cols 8 --batches 4 \
  --iters 5 --warmup 1 --check --check-samples 1024
```

Result: `max_abs=3.7252903e-09`, `bad=0/1024`.

The direct custom kernel is not competitive on the real Hiera batched-F32
shapes. The following measurements were run serially with 10 warmups and 50
iterations:

| Shape | Existing ggml/cuBLAS + ADD | Direct custom fused kernel | Result |
| --- | ---: | ---: | --- |
| `k=112 rows=112 cols=64 batches=1024` | `0.303755 ms` | `0.717108 ms` | reject |
| `k=112 rows=336 cols=64 batches=1024` | `0.802853 ms` | `2.167777 ms` | reject |
| `k=112 rows=672 cols=64 batches=1024` | `1.159304 ms` | `4.598516 ms` | reject |

The conclusion is that replacing these F32 GEMMs with a simple project-local
CUDA matmul is the wrong fusion direction. Future F32 fusion work should either
preserve the cuBLAS GEMM and reduce only surrounding elementwise work, or use a
serious tensor-core kernel with a dedicated numeric parity harness before any
full-tracker integration.

Evidence:

- `examples/f32_batched_custom.cu`
- `examples/f32_batched_custom.cuh`
- `outputs/f32-custom-fusion-probe-20260516/ggml-r112.log`
- `outputs/f32-custom-fusion-probe-20260516/custom-r112.log`
- `outputs/f32-custom-fusion-probe-20260516/ggml-r336.log`
- `outputs/f32-custom-fusion-probe-20260516/custom-r336.log`
- `outputs/f32-custom-fusion-probe-20260516/ggml-r672.log`
- `outputs/f32-custom-fusion-probe-20260516/custom-r672.log`

The current run also saved launch-shape and ptxas summaries:

- `outputs/mmq-launch-current-q8_0-1024-20260516b/mmq-summary.json`
- `outputs/mmq-launch-current-q8_0-1024-20260516b/mmq-launch-summary.json`
- `outputs/mmq-ptxas-q8_0-20260516/ptxas-summary.json`

A follow-up checked whether the src1 `q8_1` quantization kernel could be sped
up by reducing `CUDA_QUANTIZE_BLOCK_SIZE_MMQ` from `128` to `64`. This changes
only the quantization launch granularity, not the quantization math. It
preserved the first 10 full-mask rows exactly against the matched default JSONL,
but it was slower: the MMQ event profile's drop-first quantization total
increased from the previous `26.24 ms` baseline to `36.10 ms`, and the 30-frame
smoke measured `73.0 ms/frame`. The block size remains `128`.

Evidence:

- `outputs/mmq-quant-block64-q8_0-1024-20260516/profile/summary.json`
- `outputs/mmq-quant-block64-q8_0-1024-20260516/parity-vs-default.json`

The launch summary shows that the top aggregate MMQ rows are not primarily
low-efficiency launch-shape problems. The heaviest rows are either exact
stream-k tiling with high reported efficiency (`92..99%`) or known fixup rows
whose aggregate cost is not the top bottleneck. The ptxas summary confirms the
main issue is register pressure in the q8_0 main kernel: `mmq_x=128` and
`mmq_x=112` compile to `255` registers, with spills on some non-check
specializations.

Forcing CUDA Volta+ `mul_mat_q` launch bounds from min-blocks `1` to `2` was
tested as a direct occupancy experiment. It reduced q8_0 main-kernel registers
to `128`, but introduced heavy spills (`mmq_x=128` up to `900` byte
store/load spill traffic) and regressed the 10-frame q8_0/Base+/1024 tracking
smoke from `52.2` to `56.9 ms/frame`. Full-mask parity remained exact, but the
speed regression makes this a rejected route. Evidence:

- `outputs/mmq-launch-bounds2-q8_0-20260516/ptxas-summary.json`
- `outputs/mmq-launch-bounds2-q8_0-20260516/current.log`
- `outputs/mmq-launch-bounds2-q8_0-20260516/parity-vs-default.json`

This narrows the next implementation path: do not globally force occupancy.
The q8_0 MMA body needs a structural register-footprint reduction that avoids
local-memory spills, or a different specialized kernel for the hottest Hiera
shapes.

The existing `GGML_CUDA_MMQ_X_MAX` retuning hook was rechecked under matched
`--bbox-only --multimask` conditions. None of the capped tile widths is
acceptable as a default: every cap changed at least one mask hash, and the only
single-run speed improvement (`112`) still moved the bbox by up to `30 px`.

| `GGML_CUDA_MMQ_X_MAX` | Track ms/frame | P50 | P95 | Mask hashes | Max bbox delta | Max score delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | `53.3` | `50.4` | `62.7` | baseline | baseline | baseline |
| `112` | `51.7` | `50.2` | `58.8` | `9 / 10` | `30.0 px` | `0.002027` |
| `96` | `52.5` | `50.8` | `60.3` | `9 / 10` | `26.25 px` | `0.011876` |
| `80` | `54.7` | `51.5` | `65.4` | `9 / 10` | `26.25 px` | `0.016325` |
| `64` | `53.2` | `51.5` | `60.7` | `9 / 10` | `11.25 px` | `0.011270` |
| `48` | `55.7` | `53.2` | `64.9` | `9 / 10` | `26.25 px` | `0.018094` |

Evidence is under `outputs/mmq-xmax-matched-q8_0-1024-20260516b/`.

Reducing `MMQ_ITER_K` from `256` to `128` was also rejected. It did not reduce
q8_0 ptxas register pressure, produced compile-time modulo-by-zero warnings for
some other quantized types, broke q8_0/Base+/1024 tracking output completely
(`mask_hash_equal_rows=0/10`, max bbox delta `960 px`, detection count `0`),
and regressed the smoke from `53.3` to `59.6 ms/frame`. Evidence:

- `outputs/mmq-iter128-q8_0-20260516/ptxas-summary.json`
- `outputs/mmq-iter128-q8_0-20260516/current.log`
- `outputs/mmq-iter128-q8_0-20260516/parity-vs-default.json`

## 2026-05-16 SAM2.1 Size/Precision Speed Matrix

`scripts/model_matrix_compare.py` now runs official SAM2.1 Python baselines for
all four Hiera sizes (`tiny`, `small`, `base_plus`, `large`) and can reuse a
previous `python-results.json` with `--python-results` while measuring multiple
C++ precisions. Repeated `--filter` options are treated as an AND filter, so
precision-specific matrix runs no longer accidentally include unrelated model
families.

The comparable contract for this matrix is: same decoded frames (`960 x 540`),
same 10-frame range, same point prompt `(315, 250)`, and same SAM input
resolution (`1024`). Under that contract, the current CUDA C++ path is still
slower than official PyTorch bf16 for every tested SAM2.1 size/precision row.
The ratio column is `PyTorch track_ms / C++ track_ms`; values below `1.0` mean
C++ is slower.

| Model | C++ precision | C++ track ms | PyTorch track ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| tiny | f32 | `41.4` | `27.79` | `0.67` |
| tiny | f16 | `38.0` | `27.79` | `0.73` |
| tiny | q8_0 | `36.3` | `27.79` | `0.77` |
| tiny | q4_0 | `37.2` | `27.79` | `0.75` |
| tiny | q4_1 | `37.7` | `27.79` | `0.74` |
| small | f32 | `43.9` | `30.15` | `0.69` |
| small | f16 | `42.6` | `30.15` | `0.71` |
| small | q8_0 | `37.8` | `30.15` | `0.80` |
| small | q4_0 | `39.3` | `30.15` | `0.77` |
| small | q4_1 | `40.3` | `30.15` | `0.75` |
| base_plus | f32 | `65.3` | `42.67` | `0.65` |
| base_plus | f16 | `63.2` | `42.67` | `0.68` |
| base_plus | q8_0 | `53.0` | `42.67` | `0.81` |
| base_plus | q4_0 | `55.1` | `42.67` | `0.77` |
| base_plus | q4_1 | `55.5` | `42.67` | `0.77` |
| base_plus | mxfp4 | `57.9` | `42.67` | `0.74` |
| base_plus | nvfp4 | `60.3` | `42.67` | `0.71` |
| large | f32 | `162.5` | `75.62` | `0.47` |
| large | f16 | `147.8` | `75.62` | `0.51` |
| large | q8_0 | `125.0` | `75.62` | `0.60` |
| large | q4_0 | `133.2` | `75.62` | `0.57` |
| large | q4_1 | `135.1` | `75.62` | `0.56` |

This changes the speed-improvement priority:

1. The project is not yet faster than PyTorch on matched SAM2.1 workloads.
2. Quantization alone is insufficient; q8_0 is best among the tested C++
   precisions for Base+/large, but remains slower than PyTorch.
3. Large-model scaling is the weakest area. The next kernel work should focus
   on the shared Hiera encode hot paths that scale with model width: q8_0 MMQ
   body/quantization and D56 attention. Small epilogue fusion is now a lower
   priority unless it targets cuBLAS/cuBLASLt epilogues for the large f16/f32
   matmuls.

Evidence:

- `outputs/model-matrix-sam21-all-q8_0-1024-20260516/summary.json`
- `outputs/model-matrix-sam21-all-f32-1024-20260516/summary.json`
- `outputs/model-matrix-sam21-all-f16-1024-20260516/summary.json`
- `outputs/model-matrix-sam21-all-q4_0-1024-20260516/summary.json`
- `outputs/model-matrix-sam21-all-q4_1-1024-20260516/summary.json`
- `outputs/model-matrix-sam21-all-mxfp4-1024-20260516/summary.json`
- `outputs/model-matrix-sam21-all-nvfp4-1024-20260516/summary.json`

## 2026-05-16 Large q8_0 Profile

Because `sam2.1_hiera_large_q8_0` is the worst matched-speed row
(`125.0 ms/frame` C++ vs `75.62 ms/frame` official PyTorch), it was profiled
separately with `GGML_CUDA_PROFILE_NODES=1`, `GGML_CUDA_PROFILE_MMQ=1`, and
`GGML_CUDA_PROFILE_FATTN56=1`. The FATTN56-specific probe records no rows for
large because large uses head dimension `72`, not `56`; generic node profiling
therefore becomes the attention attribution source for this model.

Top synchronized node signatures:

| Rank | Signature | Count | Drop-max sum ms | Mean ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | `FLASH_ATTN_EXT dst=f32[72,8,256,16]` | `320` | `152.36` | `0.478` |
| 2 | `FLASH_ATTN_EXT dst=f32[72,8,4096,1]` | `30` | `134.39` | `4.634` |
| 3 | `MUL_MAT dst=f32[2304,4096,1,1]` | `360` | `111.24` | `0.310` |
| 4 | `MUL_MAT dst=f32[576,4096,1,1]` | `370` | `94.45` | `0.256` |
| 5 | `MUL_MAT dst=f32[1728,256,16,1]` | `320` | `80.32` | `0.252` |

MMQ event attribution totals, after dropping the first row per shape, are
`499.02 ms` total, with `430.59 ms` in MMQ body and `68.44 ms` in src1
quantization. The largest MMQ shapes are:

| MMQ shape | Effective rows | Quant sum | MMQ sum | Total sum |
| --- | ---: | ---: | ---: | ---: |
| `q8_0[576,2304] x f32[576,4096] -> f32[2304,4096]` | `359` | `5.69` | `101.16` | `106.85` |
| `q8_0[2304,576] x f32[2304,4096] -> f32[576,4096]` | `359` | `20.85` | `67.99` | `88.84` |
| `q8_0[576,1728] x f32[576,256,16] -> f32[1728,256,16]` | `319` | `5.21` | `71.22` | `76.44` |

This large-model profile refines the priority:

1. Large is dominated first by generic `head_dim=72` FlashAttention, not the
   existing D56 special path.
2. q8_0 MMQ body remains a large shared target across Base+ and Large.
3. A PyTorch-closing optimization needs either a generic FATTN path for D72
   attention or a substantial MMQ body improvement; FATTN56-only work will not
   address Large.

Evidence:

- `outputs/current-profile-large-q8_0-1024-20260516/profile.log`
- `outputs/current-profile-large-q8_0-1024-20260516/benchmark-summary.json`
- `outputs/current-profile-large-q8_0-1024-20260516/hotspots-all.json`
- `outputs/current-profile-large-q8_0-1024-20260516/mmq-summary.json`

### D72 Padding-MMA Probe

An opt-in D72 experiment was added behind
`GGML_CUDA_ENABLE_FATTN72_PAD_MMA=1`: f32 Q/K/V with head dimension `72` are
padded to `80`, K/V are converted to f16, the existing MMA FATTN path runs as
`DKQ=80,DV=80`, and the result is sliced back to `72`. This directly addresses
the Large profile finding that D72 currently falls back to the generic tile
kernel.

The speed signal is real but not acceptable yet:

| Path | Large q8_0 track ms/frame | P50 | P95 |
| --- | ---: | ---: | ---: |
| Default tile D72 | `126.2` | `120.2` | `143.0` |
| Opt-in D72 pad-MMA | `105.0` | `99.9` | `119.9` |

However, current-C++ full-mask parity fails:

| Metric | Value |
| --- | ---: |
| `mask_hash_equal_rows` | `0 / 10` |
| `min_bbox_iou` | `0.8798988622` |
| `max_bbox_delta_px` | `94.0` |
| `max_score_abs_delta` | `0.001934` |
| `max_abs_mask_area_rel_delta` | `0.001676` |

A second version that always wrote to an 80-wide temporary output and sliced
back to 72 produced the same parity failure, so the issue is not the direct
output stride. It is the f16 MMA numerical change propagating through tracking.
This path therefore remains opt-in only. It is still useful as a directional
result: D72 tensor-core attention can recover about `20 ms/frame` on Large, but
it needs a quality-preserving implementation before it can be used for the
Base+/Large fallback.

The follow-up synthetic FATTN probe confirms the speed opportunity and narrows
the failure mode. On the isolated shape `D=72,N=4096,H=8,B=1`, the default tile
path measured `4.27 ms` mean and the opt-in pad-MMA path measured `1.08 ms`
mean. Both paths have a similar CPU-reference error envelope at tolerance
`1e-2`: default `max_abs=0.00534, mean_abs=0.00111`, pad-MMA
`max_abs=0.00534, mean_abs=0.00111`. The D56 comparison is in the same broad
range (`max_abs=0.00514, mean_abs=0.00104`). This makes the next fusion-kernel
requirement more specific: D72 tensor-core attention is not obviously broken as
a kernel, but defaulting it requires a tracker-level parity-preserving numeric
path, not merely padding and slicing fixes.

The opt-in D72 path now writes directly to the 72-wide contiguous destination
when possible, instead of always writing an 80-wide temporary and launching a
slice kernel. This is deliberately a small cleanup, not a quality fix. On the
same synthetic shape, direct output measured `1.188 ms` mean vs `1.221 ms` for
the forced slice path, with the same CPU-reference error envelope. A 10-frame
Large q8_0 tracker comparison between direct-output and forced-slice pad-MMA
was bit-identical (`10 / 10` mask hashes, zero bbox/score/area deltas). Single
tracker runs measured `99.7 ms/frame` direct vs `101.3 ms/frame` forced-slice,
which is consistent with the synthetic result but too small to claim as a major
speedup.

Evidence:

- `outputs/fattn72-pad-mma-20260516/parity-summary.json`
- `outputs/fattn72-pad-mma-safe-20260516/parity-summary.json`
- `outputs/fusion-fattn-parity-probe-20260516/d72-default-tol1e-2.log`
- `outputs/fusion-fattn-parity-probe-20260516/d72-pad-mma-tol1e-2.log`
- `outputs/fusion-fattn-parity-probe-20260516/d56-default-tol1e-2.log`
- `outputs/fattn72-direct-out-probe-20260516/direct.log`
- `outputs/fattn72-direct-out-probe-20260516/slice.log`
- `outputs/fattn72-direct-out-tracker-20260516/direct-vs-slice.json`

### 2026-05-16 q4 MMQ Bias/GELU Fusion

The existing MMQ bias/GELU fusion path was unnecessarily limited to `q8_0`.
The MMQ writeback path already carries bias and activation generically, so the
fusion eligibility now also allows `q4_0` and `q4_1`. This keeps the operation
semantic unchanged: the matrix multiply output receives the same F32 bias and
optional GELU in MMQ writeback instead of as separate CUDA graph nodes.

Base+ 1024, 30-frame tracking, bbox-only multimask:

| Precision | Fusion | Track ms/frame | P50 | P95 | Mask hash parity |
| --- | --- | ---: | ---: | ---: | ---: |
| `q4_0` | disabled | `57` | `55` | `73` | reference |
| `q4_0` | enabled | `55` | `54` | `69` | `30 / 30` |
| `q4_1` | disabled | `57` | `56` | `72` | reference |
| `q4_1` | enabled | `55` | `54` | `69` | `30 / 30` |

This is a modest but clean defaultable fusion improvement for q4 models. It
does not close the PyTorch gap by itself; the remaining large targets are still
MMQ body throughput and D56/D72 FlashAttention. The rejected D56 row4 launch
experiment (`8 -> 16` rows per block) was parity exact but slower
(`76.676 ms -> 78.350 ms` D56 profiled total), so it was not kept.

Evidence:

- `outputs/mmq-bias-q4-fusion-20260516/q4_0_30/parity-default-vs-disabled.json`
- `outputs/mmq-bias-q4-fusion-20260516/q4_1_30/parity-default-vs-disabled.json`
- `outputs/fattn56-row4-16rows-20260516/fattn56-summary.json`
- `outputs/fattn56-row4-16rows-20260516/parity-vs-default.json`

### 2026-05-16 Base+ 512 Fallback Quality Refresh

The 1024 Base+ quality matrix already shows that `f16`, `f32`, and `q8_0`
clear the strict `0.99` mean/min mask IoU bar against official PyTorch on the
10-frame sample. The same check was refreshed at 512 for the practical fallback
candidates `q8_0` and `f16`.

| Precision | Mean mask IoU | Min mask IoU | Min bbox IoU | Selection match | Metadata |
| --- | ---: | ---: | ---: | ---: | --- |
| `q8_0` | `0.931306` | `0.847843` | `0.917344` | `9 / 9` propagation rows | passed |
| `f16` | `0.938010` | `0.864913` | `0.917344` | `9 / 9` propagation rows | passed |

This confirms that the Base+ fallback quality story is resolution-dependent:
1024 has a strict-quality fallback (`q8_0` or float), while 512 still does not.
The 512 gap is not candidate selection in this sample because propagation
candidate indices match official PyTorch for every checked propagation row.
For Rust-wrapper baselines, 1024 `q8_0` remains the practical quality fallback;
512 should be treated as a speed/scale study until the mask geometry gap is
fixed.

A memory-encoder input dump was added for this 512 case. C++ writes the
scale/bias-applied memory mask with `SAM3_DUMP_MEM_MASK_DIR`, and the official
comparison script writes the equivalent Python tensor with
`--save-python-mem-mask`. On the 3-frame Base+ f16 512 diagnostic, the initial
conditioning memory mask shapes match exactly (`[1,1,512,512]`), but the mask
content is already different before the memory encoder:

| Metric | Value |
| --- | ---: |
| max abs | `20.0` |
| mean abs | `0.683594` |
| C++ positive pixels after scale/bias | `63565` |
| Python positive pixels after scale/bias | `59289` |
| positive-mask mismatch pixels | `8960` |
| positive-mask IoU | `0.864051` |

This rules out the memory encoder itself as the first source of the 512 gap for
this sample. The conditioning memory receives an already-different initial
high-resolution mask, so the remaining quality work should focus on the 512 SAM
decoder/logit geometry and resize path before memory encoding.

Evidence:

- `outputs/base-plus-quality-matrix-512-20260516/summary.json`
- `outputs/base-plus-quality-matrix-1024-20260516/summary.json`
- `outputs/baseplus-f16-512-memmask-diagnostic-20260516/official/summary.json`
- `outputs/baseplus-f16-512-memmask-diagnostic-20260516/mem-mask-compare.json`

### 2026-05-16 Fusion Kernel Follow-up Rejections

After the q4 MMQ bias/GELU improvement, several lower-level fusion-kernel
variants were checked and rejected because they preserved parity but did not
improve measured speed.

For D56 FlashAttention, changing the `DKQ=64,DV=64` MMA staging target from two
stages to one compiled and remained bit-identical on the 10-frame Base+ q8_0
sample, but the profiled D56 total worsened from `76.676 ms` to `78.923 ms`.
The extra `pack`/`mma` time means the current two-stage configuration is still
better for these Hiera shapes.

For broadcast fusion, the remaining Hiera ADD hotspots include large F32
axis-0 bias broadcasts such as qkv/projection bias adds. Two specialized
variants were tried against the existing axis-broadcast fast path:

| Variant | Disabled track ms/frame | Enabled track ms/frame | Parity |
| --- | ---: | ---: | ---: |
| axis-0 `float4` broadcast | `111.5` | `112.1` | `30 / 30` |
| axis-0 row broadcast | `112.0` | `112.4` | `30 / 30` |

The result is that small broadcast-kernel reshaping is not a useful next
direction. The remaining fusion work should stay close to the dominant compute
kernels: D56/D72 FlashAttention and MMQ/MMF matmul epilogues. Existing MMF
`MUL_MAT + ADD + GELU` opt-in was also checked on Base+ f16 and was not a
defaultable win (`134.0 -> 134.4 ms/frame`, parity `30 / 30`).

Evidence:

- `outputs/fattn56-nstages1-q8_0-1024-20260516/fattn56-summary.json`
- `outputs/fattn56-nstages1-q8_0-1024-20260516/parity-vs-default.json`
- `outputs/fusion-axis0-vec4-q8_0-1024-20260516/parity-disabled-vs-enabled.json`
- `outputs/fusion-axis0-rows-q8_0-1024-20260516/parity-disabled-vs-enabled.json`
- `outputs/mmf-bias-gelu-default-candidate-f16-1024-20260516/parity-disabled-vs-enabled.json`

### 2026-05-16 D256 FATTN Fusion Probe

The current propagate hotspot is D256 FlashAttention, so the next fusion
candidate was checked at the kernel boundary instead of adding more small
broadcast fusions. `sam3_fattn_parity` now supports `--warmup` and `--iters`,
and `GGML_CUDA_PROFILE_FATTN=1` reports generic FATTN conversion and compute
time. This keeps parity and timing on the same synthetic graph.

Steady-state D256 FATTN at the propagate-like shapes:

| Shape | Default mean | Default median | Stream-k disabled mean | Stream-k disabled median | Parity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `D=256,N=4096,H=1,B=1` | `0.453374 ms` | `0.453373 ms` | `0.619644 ms` | `0.619401 ms` | `bad=0/2048` |
| `D=256,N=4100,H=1,B=1` | `0.424734 ms` | `0.424655 ms` | `0.565008 ms` | `0.564516 ms` | `bad=0/2048` |

The F32-to-F16 conversion cost is not the main steady-state problem:

| Shape | K convert | V convert | FATTN compute | Total |
| --- | ---: | ---: | ---: | ---: |
| `D=256,N=4096` | `~0.016-0.017 ms` | `~0.016 ms` | `~0.424-0.426 ms` | `~0.457-0.460 ms` |
| `D=256,N=4100` | `~0.015 ms` | `~0.016 ms` | `~0.397-0.402 ms` | `~0.428-0.433 ms` |

This makes a direct "fuse K/V conversion into FATTN load" kernel a low-priority
optimization for this workload. The useful fusion-kernel direction is inside
the stream-k FATTN path itself: reduce fixup/combine overhead and improve the
main tile kernel. Disabling stream-k is parity-safe but slower, so it remains an
experiment switch only (`GGML_CUDA_DISABLE_FATTN_STREAM_K=1`) and should not be
defaulted.

Forcing the stream-k block count to round down to the output tile count
(`blocks=(64,1,1)`) also passed parity but regressed the D256/N4096 mean to
`0.619640 ms` because the lower parallelism costs more than the removed general
fixup. That experiment was not kept as a runtime switch.

Evidence:

- `outputs/fattn-fusion-probe-20260516/d256_n4096.log`
- `outputs/fattn-fusion-probe-20260516/d256_n4100.log`
- `outputs/fattn-fusion-probe-20260516/d256_n4096_rounded.log`

### 2026-05-16 State PE Prewarm and MMQ Shape Probe

The next apparent gap was the first SAM2 Hiera encode row. Stage profiling
showed `hiera_encode_pe_compute_upload=108.515 ms` on the first frame and cache
hits thereafter. This is the neck positional encoding cache, which depends only
on the model and `encode_img_size`, not on image pixels. It is now prewarmed in
`sam3_create_state` for SAM2 states, with
`SAM3_DISABLE_STATE_PE_PREWARM=1` available for A/B runs.

This is a correctness-safe startup/total-time improvement, not a PyTorch-closing
track-frame improvement. The benchmark's `track/fr` excludes the initial
point-prompt encode, so the main comparable speed row remains controlled by
steady Hiera encode and propagation.

| Path | Init ms | Track ms/frame | Total ms | Parity vs disabled |
| --- | ---: | ---: | ---: | ---: |
| PE prewarm enabled | `391.4` | `53.4` | `871.8` | `10 / 10` mask hashes |
| PE prewarm disabled | `497.1` | `51.9` | `964.2` | baseline |

The initial Hiera encode log row dropped from about `455 ms` disabled to about
`349 ms` enabled in the parity run. The output JSONL comparison was exact:
`min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, and `max_score_abs_delta=0.0`.

`sam3_mmq_bench` was also extended with `--channels` so Hiera window MMQ shapes
can be reproduced as `src1=f32[k, cols, channels]` instead of flattening them
into a different launch shape. Standalone hot-shape checks did not find a useful
`GGML_CUDA_MMQ_X_MAX` candidate:

| Shape | Default mean | Candidate | Candidate mean | Result |
| --- | ---: | --- | ---: | --- |
| `q8_0[448,1792] x f32[448,4096]` | `0.137082 ms` | `X_MAX=112` | `0.145918 ms` | slower |
| `q8_0[1792,448] x f32[1792,4096]` | `0.158709 ms` | `X_MAX=96` | `0.163630 ms` | slower |
| `q8_0[448,1344] x f32[448,196,25]` | `0.127942 ms` | `X_MAX=104` | `0.149436 ms` | slower |

This reinforces the current priority: do not spend more effort on tile-width
caps for q8_0. The remaining PyTorch gap is still Hiera encode body throughput:
official PyTorch Base+ 1024 `forward_image` is about `28.14 ms`, while current
C++ steady Hiera encode is about `36-38 ms`.

Evidence:

- `outputs/state-pe-prewarm-20260516/parity-runs.log`
- `outputs/state-pe-prewarm-20260516/parity-vs-disabled.json`
- `outputs/mmq-shape-probe-20260516/mlp_expand.log`
- `outputs/mmq-shape-probe-20260516/mlp_project.log`
- `outputs/mmq-shape-probe-20260516/window_qkv.log`
- `outputs/pytorch-profile-base-plus-1024-after-stream-copy-fix-20260516/summary.json`

### 2026-05-16 D56 FATTN Fusion Recheck

The D56 FlashAttention wrapper was rechecked before changing the default
packing path. The tempting change was to disable the contiguous combined pack by
default and fall back to the row-oriented pack kernel. That is not a clean win.

Standalone D56 parity passed in every tested mode, but timing is shape-dependent:

| Shape | Default mean | Default median | Contiguous pack disabled mean | Contiguous pack disabled median | Parity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `D=56,N=4096,H=8,B=1` | `0.888697 ms` | `0.890126 ms` | `0.880318 ms` | `0.881811 ms` | `bad=0/3584` |
| `D=56,N=196,H=8,B=25` | `0.196859 ms` | `0.196795 ms` | `0.217509 ms` | `0.217181 ms` | `bad=0/89600` |

The global row very slightly favors disabling the contiguous pack, but the
window-attention row regresses by about `10%`, and Base+ Hiera uses many window
rows. A short full-tracker A/B was exact (`10/10` mask hashes for each of three
pairs), but the speed difference was within run noise:

| Path | Track ms/frame runs | Mean |
| --- | --- | ---: |
| Default | `52.6, 52.3, 51.3` | `52.1` |
| `GGML_CUDA_DISABLE_FATTN56_CONTIGUOUS_PACK=1` | `51.9, 51.8, 51.9` | `51.9` |

Because the standalone window shape clearly regresses and the end-to-end delta
is too small, the default remains the existing combined contiguous pack plus
direct D56 output. Fusion work should not be limited to staging toggles; the
next meaningful CUDA work needs a dedicated D56 kernel/layout change or a q8_0
MMQ body change with measured register-pressure and parity evidence.

Evidence:

- `outputs/fattn56-fusion-probe-20260516b/global-default.log`
- `outputs/fattn56-fusion-probe-20260516b/global-no-contig.log`
- `outputs/fattn56-fusion-probe-20260516b/window-default.log`
- `outputs/fattn56-fusion-probe-20260516b/window-no-contig.log`
- `outputs/fattn56-contig-ab-20260516/compare/run1.json`
- `outputs/fattn56-contig-ab-20260516/compare/run2.json`
- `outputs/fattn56-contig-ab-20260516/compare/run3.json`

### 2026-05-16 D72 Pad-MMA Quality and Profile Refresh

The opt-in D72 pad-MMA path is still the strongest Large-model attention speed
candidate. A standalone `sam3_fattn_parity` check shows a large kernel-level
speedup while staying within the scalar-reference tolerance:

| Shape | Default mean | `GGML_CUDA_ENABLE_FATTN72_PAD_MMA=1` mean | Parity |
| --- | ---: | ---: | ---: |
| `D=72,N=4096,H=8,B=1` | `4.669049 ms` | `1.185509 ms` | `bad=0/4608` |
| `D=72,N=256,H=8,B=16` | `0.389773 ms` | `0.268421 ms` | `bad=0/73728` |

`GGML_CUDA_PROFILE_FATTN72=1` now reports the opt-in path's pack/MMA/slice
breakdown and disables CUDA graph capture while profiling. Steady rows show the
remaining D72 pad-MMA cost is mostly MMA body, not slice:

| Shape | Pack | MMA | Slice | Total |
| --- | ---: | ---: | ---: | ---: |
| `D=72,N=4096,H=8,B=1` | `~0.087 ms` | `~1.07-1.09 ms` | `~0.035 ms` | `~1.19-1.21 ms` |
| `D=72,N=256,H=8,B=16` | `~0.087-0.089 ms` | `~0.159-0.166 ms` | `~0.035 ms` | `~0.281-0.290 ms` |

End-to-end Large q8_0 speed improves substantially but is still not enough to
beat official PyTorch on this matched 1024 contract:

| Path | Large q8_0 track ms/frame |
| --- | ---: |
| Default D72 tile path | `127.3` |
| Opt-in D72 pad-MMA | `105.8` |
| Official PyTorch bf16, same 10-frame contract | `~82.7-88.6` in the quality reruns |

The quality result keeps D72 pad-MMA as an experiment, not a default. Against
official SAM2.1 Large bf16 on the same 10 frames and 1024 encode size, the
default C++ Large q8_0 path is already below the strict Base+ fallback quality
bar, and pad-MMA is slightly lower:

| Path | Mean mask IoU | Min mask IoU | Propagation candidate matches |
| --- | ---: | ---: | ---: |
| Default D72 tile path | `0.969477` | `0.939829` | `4 / 9` |
| Opt-in D72 pad-MMA | `0.965467` | `0.939992` | `3 / 9` |

Default-vs-pad full-mask parity also remains non-exact (`0 / 10` mask hashes);
the largest bbox delta in this rerun was `21 px`, mostly from one propagation
row. This means D72 pad-MMA is useful evidence that tensor-core attention can
recover about `20 ms/frame` on Large, but the next implementation must address
candidate-selection/logit stability before defaulting it.

Evidence:

- `outputs/fattn72-probe-20260516/global-default.log`
- `outputs/fattn72-probe-20260516/global-pad-mma.log`
- `outputs/fattn72-probe-20260516/window-default.log`
- `outputs/fattn72-probe-20260516/window-pad-mma.log`
- `outputs/fattn72-profile-20260516/global.log`
- `outputs/fattn72-profile-20260516/window.log`
- `outputs/fattn72-default-official-quality-large-fullmask-20260516/quality/summary.json`
- `outputs/fattn72-pad-mma-official-quality-large-fullmask-20260516/quality/summary.json`
- `outputs/fattn72-pad-mma-official-quality-large-fullmask-20260516/default-vs-pad.json`

### 2026-05-16 q8_0 MMQ DS4 Scale Probe

The q8_0 activation quantization layout was tested as another MMQ body variant:
store the q8_1 activation scale in the `DS4` half2 layout instead of the
default `D4` float-scale layout, and route q8_0 MMQ through the matching MMA
scale loader. The hypothesis was that lower scale precision could reduce
register/load pressure enough to help the Hiera stage-2 MLP rows.

The isolated shape result was mixed:

| Shape | Default mean | DS4 mean | Result |
| --- | ---: | ---: | --- |
| `q8_0[448,1792] x f32[448,4096]` with bias+GELU | `0.178497 ms` | `0.163981 ms` | faster |
| `q8_0[1792,448] x f32[1792,4096]` | `0.168196 ms` | `0.171161 ms` | slower |
| `q8_0[448,1344] x f32[448,196,25]` | `0.128419 ms` | `0.132100 ms` | slower |

The full tracker result was not acceptable. On the 10-frame Base+ q8_0 1024
full-mask run, DS4 measured `57.1 ms/frame` versus the matched default
`57.3 ms/frame`, which is only noise-level speed movement. It also changed every
mask hash and moved one bbox by `21 px`:

| Metric | Value |
| --- | ---: |
| mask-hash equal rows | `0 / 10` |
| min bbox IoU | `0.9773462783` |
| max bbox delta | `21 px` |
| max score delta | `0.006451` |

The experiment was reverted. A useful q8_0 MMQ scale-layout variant would need
to be shape-specific to the MLP expansion row and would still need a parity
story; applying DS4 globally is not viable.

Evidence:

- `outputs/mmq-q8-ds4-probe-20260516/mlp_expand_speed.log`
- `outputs/mmq-q8-ds4-probe-20260516/mlp_project.log`
- `outputs/mmq-q8-ds4-probe-20260516/window_qkv_speed.log`
- `outputs/mmq-q8-ds4-probe-20260516/mlp_expand_default_after_revert.log`
- `outputs/mmq-q8-ds4-probe-20260516/mlp_project_default_after_revert.log`
- `outputs/mmq-q8-ds4-probe-20260516/window_qkv_default_after_revert.log`
- `outputs/mmq-q8-ds4-full-ab-20260516/default-vs-ds4.json`

### 2026-05-16 MMQ Static Epilogue Fusion

The next fusion-kernel change kept the existing graph-level fusion contract and
specialized the MMQ writeback epilogue itself. The previous fused path still
passed `mmq_activation` as a runtime enum into the CUDA kernel, so every fused
writeback evaluated a small activation switch. The new path templates the
common epilogues:

- plain writeback remains the existing no-bias/no-activation specialization;
- bias-only uses a static `MMQ_ACT_NONE` epilogue;
- q8_0/q4_0/q4_1 `bias + GELU` uses a static `MMQ_ACT_GELU` epilogue;
- other activation forms fall back to the dynamic path.

`GGML_CUDA_DISABLE_MMQ_STATIC_EPILOGUE=1` disables the new static epilogue for
A/B tests.

The isolated q8_0 Hiera hot shapes show the intended behavior: the fused MLP
expansion row improves, while non-GELU rows stay effectively neutral.

| Shape | Static mean | Dynamic mean | Result |
| --- | ---: | ---: | --- |
| `q8_0[448,1792] x f32[448,4096]` with bias+GELU | `0.158144 ms` | `0.180656 ms` | `12.5%` faster |
| `q8_0[1792,448] x f32[1792,4096]` | `0.168608 ms` | `0.162290 ms` | noise/slower |
| `q8_0[448,1344] x f32[448,196,25]` | `0.133177 ms` | `0.135449 ms` | neutral |

The full Base+ q8_0 1024 smoke stayed bit-identical against the dynamic path:
10/10 mask hashes matched, min bbox IoU was `1.0`, and max score delta was
`0.0`. Five alternating bbox-only runs also show an end-to-end speed signal:

| Path | Track ms/frame samples | Mean | Median | P50 mean | P95 mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Static epilogue | `51.1, 50.6, 49.5, 49.8, 50.3` | `50.26` | `50.3` | `48.28` | `58.32` |
| Dynamic epilogue | `52.1, 52.1, 52.1, 52.8, 51.8` | `52.18` | `52.1` | `50.10` | `60.60` |

This is a small but defaultable fusion improvement: about `1.9 ms/frame`
(`3.7%`) on this Base+ q8_0 1024 bbox-only smoke, with full-mask parity exact
on the paired 10-frame run. q4_0 and q4_1 Base+ 1024 full-mask smoke tests were
also exact against the dynamic path (`10 / 10` mask hashes for both).

Evidence:

- `outputs/mmq-static-epilogue-20260516/q8_expand_static_seq.log`
- `outputs/mmq-static-epilogue-20260516/q8_expand_dynamic_seq.log`
- `outputs/mmq-static-epilogue-20260516/q8_project_static_seq.log`
- `outputs/mmq-static-epilogue-20260516/q8_project_dynamic_seq.log`
- `outputs/mmq-static-epilogue-20260516/q8_window_static_seq.log`
- `outputs/mmq-static-epilogue-20260516/q8_window_dynamic_seq.log`
- `outputs/mmq-static-epilogue-20260516/full/parity-dynamic-vs-static.json`
- `outputs/mmq-static-epilogue-20260516/repeats/`
- `outputs/mmq-static-epilogue-20260516/q4/q4_0-parity.json`
- `outputs/mmq-static-epilogue-20260516/q4/q4_1-parity.json`

### 2026-05-16 Goal Audit and Static-Epilogue Profile Refresh

The goal audit script was refreshed so it no longer relies only on the older
512/1024 matrix summaries. It now accepts the corrected paired C++/official
PyTorch summaries, the all-size SAM2.1 q8_0 1024 matrix, and the static
epilogue parity artifact.

The refreshed audit status is still `not_complete`:

| Criterion | Status | Evidence |
| --- | --- | --- |
| Matched official-Python comparison contract | met | matrix rows are marked comparable; paired summaries use the same frames 1..N-1 timing contract |
| Detailed Hiera/profile attribution | met | `outputs/hiera-gap-priority/summary.json`, current CUDA node/MMQ summaries |
| Speed-priority identification | met | Hiera encode remains the priority; D56 attention and q8_0 MMQ body dominate |
| Static epilogue fusion parity | met | q8_0 final-smoke full-mask parity is exact |
| Base+ paired speed and strict quality across samples | missing | bedroom is faster and strict-quality; juggle is faster on mean but min mask IoU is `0.987614` |
| All-size SAM2.1 q8_0 1024 faster than PyTorch | missing | tiny/small/Base+/Large are still slower in the all-size matrix |
| Base+ fallback quality at all checked resolutions | missing | 1024 has strict-quality candidates; 512 does not |

The static-epilogue profile refresh used `GGML_CUDA_PROFILE_NODES=1`,
`GGML_CUDA_PROFILE_MMQ=1`, and `GGML_CUDA_PROFILE_MMQ_LAUNCH=1` on a 5-frame
Base+ q8_0 1024 run. As expected, profiling disables CUDA graphs and
synchronizes per node, so these numbers are attribution only.

Top drop-max CUDA node buckets after the static epilogue change:

| Bucket | Drop-max sum | Mean | Count | Priority |
| --- | ---: | ---: | ---: | --- |
| MMQ MLP expansion `dst=f32[1792,4096]` | `12.70 ms` | `0.1607 ms` | `80` | secondary |
| D56 window FlashAttention `dst=f32[56,8,196,25]` | `11.85 ms` | `0.2009 ms` | `60` | primary |
| D56 global FlashAttention `dst=f32[56,8,4096,1]` | `11.77 ms` | `0.8407 ms` | `15` | primary |
| MMQ MLP projection `dst=f32[448,4096]` | `11.62 ms` | `0.1470 ms` | `80` | secondary |
| MMQ window qkv `dst=f32[1344,196,25]` | `10.18 ms` | `0.1726 ms` | `60` | secondary |
| D256 propagation FlashAttention `dst=f32[256,1,4096,1]` | `13.07 ms` | `~0.435 ms` | `32` | primary |
| D56 q-pool FlashAttention `dst=f32[56,2,64,1024]` | `5.89 ms` | `0.6547 ms` | `10` | primary |

The MMQ event profile after dropping the first row per shape reports `72.35 ms`
total, split into `13.36 ms` activation quantization and `58.99 ms` MMQ body.
The largest MMQ shapes are still the stage-2 MLP expansion/projection and
window qkv rows. Static epilogue improved the writeback/GELU path, but it did
not change the core conclusion: the next meaningful speed work needs body-level
D56/D256 attention or q8_0 MMQ throughput changes, not more small epilogue
cleanup. The hotspot classifier was refreshed so D56 q-pool attention and D256
propagation attention are not hidden under `other`; the `hiera_encode` label is
not present in this summary, so the all-node hotspot file is the relevant
ranking artifact.

A q8_0 activation-quantization micro-kernel was also checked and rejected. The
candidate processed two output columns per CUDA block for the D4/no-ids MMQ
quantization path, targeting the quantization-heavy
`q8_0[448,112] x f32[448,65536]` row. It compiled, but did not improve the
measured profile: the stage-0 shape stayed around `0.47 ms` total with no
useful p50 movement, and the MLP expansion shape regressed slightly. The code
was removed rather than kept behind an environment gate because it adds kernel
surface without a positive signal.

Evidence:

- `outputs/mmq-cols2-quant-probe-20260516/stage0_default.log`
- `outputs/mmq-cols2-quant-probe-20260516/stage0_cols2.log`
- `outputs/mmq-cols2-quant-probe-20260516/mlp_expand_default.log`
- `outputs/mmq-cols2-quant-probe-20260516/mlp_expand_cols2.log`

### 2026-05-16 Goal Audit Refresh After Fusion Follow-up

The goal audit was refreshed after the D56 inline-global and MMQ cols2
quantization probes. The status remains `not_complete`.

The parts that are now covered:

- official-Python comparisons use matched input conditions;
- detailed Hiera/profile attribution exists;
- speed-priority rows are identified from CUDA node/MMQ/FATTN profiles;
- the accepted MMQ static-epilogue fusion has exact full-mask parity evidence.

The remaining blockers are concrete:

| Criterion | Current state | Status |
| --- | --- | --- |
| Base+ paired q8_0 faster than official PyTorch and strict-quality across samples | bedroom is faster and strict-quality; juggle is faster on mean but min mask IoU is `0.987614` | missing |
| all-size SAM2.1 q8_0 1024 faster than official PyTorch | tiny/small/Base+/Large are all still slower; Base+ ratio is `0.8506`, Large ratio is `0.6263` | missing |
| Base+ fallback quality at checked resolutions | 1024 passes (`min IoU 0.993410`), but 512 fails (`min IoU 0.958605`) | missing |

The current priority remains body-level kernels, not small launch-only fusions:
D56/D72 FlashAttention, propagation D256 FlashAttention, and q8_0 MMQ body
throughput. The rejected probes in this section show that extending inline
pack to global D56 and reshaping q8_0 activation quantization do not close the
gap.

Evidence:

- `outputs/goal-audit/current-fusion-followup-20260516-summary.json`
- `outputs/model-matrix-sam21-all-q8_0-1024-inline-multimask-20260516/summary.json`
- `outputs/baseplus-f16-512-logit-diagnostic-20260516/official/summary.json`
- `outputs/baseplus-q8_0-1024-inline-quality-multimask-20260516/official/summary.json`

Evidence:

- `outputs/goal-audit-20260516/summary.json`
- `outputs/current-profile-q8_0-1024-static-epilogue-20260516/summary.json`
- `outputs/current-profile-q8_0-1024-static-epilogue-20260516/hotspots-all.json`
- `outputs/current-profile-q8_0-1024-static-epilogue-20260516/hotspots-hiera-encode.json`
- `outputs/current-profile-q8_0-1024-static-epilogue-20260516/mmq-summary.json`
- `outputs/current-profile-q8_0-1024-static-epilogue-20260516/mmq-launch-summary.json`

### 2026-05-16 FATTN56 Fusion-Kernel Direction

The current FATTN56 path still works by padding head56 Q/K/V into temporary
head64 tensors and then calling the existing MMA attention kernel. A fresh
profile on Base+ q8_0 1024 after the MMQ static-epilogue work gives this
drop-first-per-shape attribution:

| Component | Sum | Mean | P95 |
| --- | ---: | ---: | ---: |
| pack | `8.986 ms` | `0.080 ms` | `0.315 ms` |
| MMA | `29.588 ms` | `0.264 ms` | `0.858 ms` |
| slice/direct-output overhead | `0.344 ms` | `0.003 ms` | `0.004 ms` |
| total | `38.919 ms` | `0.347 ms` | `0.887 ms` |

The shape split shows two different remaining problems:

| Shape | Pack sum | MMA sum | Interpretation |
| --- | ---: | ---: | --- |
| global `Q/K/V[56,4096,8,1]` | `0.398 ms` | `12.438 ms` | body-level MMA dominates |
| window `Q/K/V[56,196,8,25]` | `2.356 ms` | `9.666 ms` | body-level MMA still dominates |
| q-pool `Q[56,16,4,1024] K/V[56,64,4,1024]` | `1.459 ms` | `1.086 ms` | pack is heavier than MMA |
| q-pool `Q[56,4,8,1024] K/V[56,16,8,1024]` | `0.672 ms` | `0.912 ms` | both matter |
| q-pool `Q[56,49,16,25] K/V[56,196,16,25]` | `0.378 ms` | `0.453 ms` | both matter |

An opt-in split mixed-pack prototype was tested and removed. It separated Q
packing and K/V packing into two specialized row4 kernels for mixed q-pool
shapes. The extra launch and lost combined work made the result worse:

| Path | Pack sum | MMA sum | Total sum |
| --- | ---: | ---: | ---: |
| current mixed row4 pack | `8.986 ms` | `29.588 ms` | `38.919 ms` |
| split mixed pack prototype | `9.906 ms` | `29.242 ms` | `39.504 ms` |

The next FATTN56 fusion-kernel step should therefore not split the pack stage.
The useful direction is a real head56 MMA path that consumes the original
head56 Q/K/V layout directly, or a narrow q-pool-specialized variant that
eliminates the padded temporaries without adding launches. This is larger than
a config retune: it changes the load path inside the attention MMA body and
must keep full-mask parity exact before becoming default.

The existing native-V opt-in path was also rechecked shape-by-shape because it
is the closest current approximation of a head56-aware MMA body: it keeps V at
56 channels and calls the `DKQ=64, DV=56` MMA variant. It is not a viable
shape-gated shortcut. Every major FATTN56 bucket slowed down, and the profile
total moved from `38.919 ms` to `51.829 ms`:

| Path | Pack sum | MMA sum | Total sum |
| --- | ---: | ---: | ---: |
| current padded V64 path | `8.986 ms` | `29.588 ms` | `38.919 ms` |
| native V56 opt-in | `10.612 ms` | `40.834 ms` | `51.829 ms` |

The regression is especially clear on the global and window rows:

| Shape | Current total | Native V56 total |
| --- | ---: | ---: |
| global `Q/K/V[56,4096,8,1]` | `12.880 ms` | `19.345 ms` |
| window `Q/K/V[56,196,8,25]` | `12.205 ms` | `15.263 ms` |
| q-pool `Q[56,16,4,1024] K/V[56,64,4,1024]` | `2.557 ms` | `3.342 ms` |

So the next implementation should not use `DV=56` as the fast path. The useful
fusion is narrower: keep the tensor-core-friendly `DKQ=64, DV=64` compute
shape, but move the current global-memory pad kernels into the MMA kernel's
shared-memory load stage. That would remove the separate Q/K/V padded
temporaries while preserving the aligned shared-memory layout that the current
MMA body expects.

A standalone D56 stream-k A/B was also refreshed with `sam3_fattn_parity`.
This does not reproduce q-pool's mixed Q/K lengths, but it cleanly separates
the same-shape window and global D56 behavior:

| Shape | Default mean | Stream-k disabled mean | Parity |
| --- | ---: | ---: | ---: |
| D56 padded-to-64 window `N=196,H=8,B=25` | `0.231245 ms` | `0.226328 ms` | `bad=0/179200` |
| D56 padded-to-64 global `N=4096,H=8,B=1` | `0.841161 ms` | `0.865522 ms` | `bad=0/3584` |

This is not enough to default a broad no-stream-k path: global attention still
prefers the current stream-k launch, and earlier end-to-end no-stream-k trials
were rejected. It does show that D56 window/q-pool work should stay
shape-specific rather than changing the global FATTN policy.

An opt-in full-graph follow-up then tried to disable stream-k only for the
same-shape D56 window bucket (`Q/K/V[56,196,8,25]`). It was removed immediately:
the full Base+ q8_0 1024 FATTN56 profile regressed the window MMA bucket from
about `0.207 ms` per row to `38.758 ms` per row, and the FATTN56 total exploded
from `38.919 ms` to `2310.948 ms`. The standalone `sam3_fattn_parity` result
therefore does not represent the full graph's window launch/tiling contract.
Future work should not disable stream-k for this bucket; it needs a real
window/q-pool-specific kernel body or inline-pack implementation.

Evidence:

- `outputs/fattn56-fusion-current-20260516/summary.json`
- `outputs/fattn56-split-mixed-pack-20260516/summary.json`
- `outputs/fattn56-native-v-shape-gated-20260516/summary.json`
- `outputs/fattn56-streamk-standalone-20260516/window-default.log`
- `outputs/fattn56-streamk-standalone-20260516/window-no-streamk.log`
- `outputs/fattn56-streamk-standalone-20260516/global-default.log`
- `outputs/fattn56-streamk-standalone-20260516/global-no-streamk.log`

The existing inline head56 MMA path was then temporarily exposed to the large
single-sequence D56 bucket. This tests the same fusion direction directly:
remove the separate global Q/K/V pad kernels and convert/pad inside the MMA
load path. The result is not a defaultable win, and the probe code was removed.
On the standalone global shape, the separate pack path spends only about
`0.058 ms` in pack and `0.825-0.842 ms` in MMA, while the inline load path moves
the cost into MMA and lands around `0.895-0.981 ms` on stable rows, with higher
variance. Full tracker parity stayed exact, but the single-pair
q8_0/Base+/1024 bbox-only run regressed from `99.4` to `101.0 ms/frame`.

The conclusion is that "fuse pack into MMA load" is shape-specific. It is
useful for current window/q-pool D56 paths, where the inline path is already
active, but it should not be extended to the global `N=4096` D56 bucket. The
remaining global D56 problem is body-level attention throughput, not launch
fusion around padding.

Evidence:

- `outputs/fattn56-inline-large-probe-20260516/global-default-profile.log`
- `outputs/fattn56-inline-large-probe-20260516/global-inline-large-profile.log`
- `outputs/fattn56-inline-large-probe-20260516/tracker/default-vs-inline-large.json`
- `outputs/fattn56-window-no-streamk-20260516/summary.json`

### 2026-05-16 FATTN56 Inline-Pack Fusion

The first kept fusion-kernel implementation is an inline-pack path for head_dim
56 attention. It keeps the existing tensor-core-friendly `DKQ=64, DV=64` MMA
body, but lets the MMA kernel consume the original head56 f32 Q/K/V tensors and
pack them into the padded shared-memory layout directly. This removes the
separate global-memory Q/K/V padding launch for eligible rows while preserving
the current direct head56 output path. It is default-on after the Base+
precision parity and speed checks below; `GGML_CUDA_DISABLE_FATTN56_INLINE_PACK=1`
restores the old padded-temporary path for A/B tests.

The gate deliberately skips large single-sequence K/V rows because the standalone
global probe did not show a stable win. The useful target is the window/q-pool
side of FATTN56, where the old path paid a real pack cost and the packed
temporaries were short-lived.

Standalone shape probes:

| Shape | Default mean | Inline-pack mean | Result |
| --- | ---: | ---: | --- |
| D56 window `N=196,H=8,B=25` | `0.229836 ms` | `0.152370 ms` | faster |
| D56 global sample-query `N=4096,H=8,B=1` | `0.825566 ms` | `0.856532 ms` | slower, excluded by gate |

The full Base+ q8_0 1024 full-mask run is bit-identical against the old padded
path: `mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0.0`, and `max_score_abs_delta=0.0`. The same exact-parity
result was checked for Base+ `f16`, `f32`, `q8_0`, `q4_1`, and `q4_0` at 1024.

Five paired bbox-only runs with the gated inline path show a small end-to-end
win:

| Path | Track ms/frame values | Mean |
| --- | --- | ---: |
| Old padded path | `50.0, 49.8, 51.3, 49.7, 49.9` | `50.14` |
| Inline-pack path | `48.7, 49.3, 48.3, 48.4, 48.2` | `48.58` |

This is about `1.56 ms/frame` (`3.1%`) on the checked Base+ q8_0 1024
configuration. Representative three-pair checks also improved Base+ f16
(`61.47 -> 60.13 ms/frame`), q8_0 (`50.27 -> 48.60 ms/frame`), and q4_1
(`50.80 -> 49.73 ms/frame`).

Evidence:

- `outputs/fattn56-inline-pack-20260516/window-default.log`
- `outputs/fattn56-inline-pack-20260516/window-inline.log`
- `outputs/fattn56-inline-pack-20260516/global-default.log`
- `outputs/fattn56-inline-pack-20260516/global-inline.log`
- `outputs/fattn56-inline-pack-gated2-20260516/full/parity.json`
- `outputs/fattn56-inline-pack-gated2-20260516/repeats/summary.txt`
- `outputs/fattn56-inline-pack-baseplus-precisions-20260516/parity_summary.tsv`
- `outputs/fattn56-inline-pack-baseplus-speed-20260516/summary.txt`
- `outputs/fattn56-inline-pack-default-20260516/parity-disabled-vs-default.json`

A follow-up fixed the FATTN56 profiler path so inline-pack rows are still
reported in the same `GGML_CUDA_PROFILE_FATTN56` format. Without this, the
summary only showed the excluded global rows because the inline branch returned
before the old pad/MMA/slice event block.

The refreshed 3-frame profile confirms the staging work has mostly disappeared:

| Component | Drop-first sum |
| --- | ---: |
| pack | `0.225 ms` |
| MMA / inline body | `17.623 ms` |
| slice | `0.026 ms` |
| total | `17.874 ms` |

Top FATTN56 shapes after the profiler fix:

| Shape | Total sum | Mean |
| --- | ---: | ---: |
| global `Q/K/V[56,4096,8,1]` | `6.984 ms` | `0.873 ms` |
| window `Q/K/V[56,196,8,25]` | `5.499 ms` | `0.157 ms` |
| q-pool `Q/K/V[56,64,2,1024]` | `1.860 ms` | `0.372 ms` |
| q-pool `Q/K/V[56,16,4,1024]` | `1.440 ms` | `0.288 ms` |
| mixed q-pool `Q[56,16,4,1024] K/V[56,64,4,1024]` | `0.795 ms` | `0.397 ms` |

The next FATTN work is therefore not more pack/slice removal. It needs either a
body-level global D56 improvement or a broader attention body change that also
helps q-pool. On the MMQ side, a fresh `sam3_mmq_bench` sweep showed the current
q8_0 MLP choices already use `mmq_x=128`; forcing smaller `MMQ_X_MAX` values
did not improve project/window rows and `MMQ_X_MAX=64` regressed all tested hot
shapes.

Evidence:

- `outputs/fattn56-inline-profile-fix-20260516/profile.log`
- `outputs/fattn56-inline-profile-fix-20260516/summary.json`
- `outputs/mmq-q8-xmax-sweep-inline-20260516/summary.tsv`
- `outputs/mmq-q8-xmax-sweep-inline-20260516/launch/`

### Fusion Kernel Probe: Axis-2 Broadcast ADD

After the inline FATTN56 fusion, the next low-risk fusion probe targeted the
remaining unfused elementwise path rather than another MMQ epilogue change. A
CUDA node profile of Base+ q8_0 at 1024 still shows the positional/bias
broadcast add `dst=f32[256,256,112,1] src1=f32[1,1,112,1]` as a visible
standalone node.

A specialized axis-2 plane kernel was tested for this `[W,H,C,1] + [1,1,C,1]`
shape. It avoided the generic axis kernel's per-element source-index division,
but the measured node time worsened on the checked RTX 5070 Ti Laptop GPU:

| Path | `node_9` ADD time | Bbox parity |
| --- | ---: | --- |
| Existing axis broadcast kernel | `3.120448 ms` | reference |
| Axis-2 plane kernel probe | `3.352224 ms` | exact JSONL match |

The probe was reverted and should not be repeated as a default optimization.
The remaining useful fusion work is not scalar broadcast ADD. It should focus
on larger producer/consumer cuts where a fused kernel removes a material tensor
or changes the dominant arithmetic body:

- FATTN body-level fusion for the remaining D56 global/q-pool rows.
- MMQ body/epilogue work only if it preserves the exact stream-k accumulation
  order or writes an auxiliary unfused-compatible result.
- Patch-embedding level fusion only if it replaces the `im2col -> f32 GEMM ->
  layout/PE add` pipeline with a measured direct convolution or library path;
  small post-op fusion alone is not expected to close the Python gap.

Evidence:

- `outputs/fusion-pattern-scan-20260516/summary.json`
- `outputs/fusion-axis2-plane-20260516/old/summary.json`
- `outputs/fusion-axis2-plane-20260516/new/summary.json`

### 2026-05-16 Base+ q8_0 Inline-Pack Refresh Against Official PyTorch

The matched Base+ q8_0 1024 comparison was refreshed after the FATTN56
inline-pack default. The comparable speed contract is unchanged: same decoded
`960x540` source frames, 10-frame range, point prompt `(315,250)`, and SAM input
resolution 1024.

For the quality-contract path, C++ must use the SAM2 point-prompt `--multimask`
mode. Running without it is a different initial prompt contract and drops mask
IoU to about `0.71`, so those rows are diagnostics only.

Current quality-contract speed row:

| Row | Track ms/frame | Notes |
| --- | ---: | --- |
| C++ CUDA q8_0 Base+ 1024, `--bbox-only --multimask` | `48.4` | current inline-pack default |
| Official PyTorch SAM2.1 Base+ bf16 | `42.56` | same extracted frames and image size |

The C++/PyTorch ratio is `0.879` (`python_over_cpp_track_ratio`), so C++ is
still about `13.7%` slower than official PyTorch on this matched row. This is an
improvement over the earlier pre-inline q8_0 1024 refresh (`65.56` vs `42.89`),
but it is not yet a PyTorch-beating result.

The corresponding full-mask quality check with shared extracted frames and
`--multimask` passes the Base+ 1024 fallback bar:

| Metric | Value |
| --- | ---: |
| Frames compared | `10` |
| Mean mask IoU vs official PyTorch bf16 | `0.995071` |
| Min mask IoU vs official PyTorch bf16 | `0.993410` |
| Initial candidate count | `3 / 3` |
| Initial selected candidate | matches official (`0`) |

Propagation selected-candidate indices still differ on 5/9 frames, but the mask
IoU on those frames remains at or above `0.9934`, indicating near-tie candidate
selection rather than a meaningful mask-quality failure.

Evidence:

- `outputs/model-matrix-baseplus-q8_0-1024-inline-multimask-20260516/summary.json`
- `outputs/baseplus-q8_0-1024-inline-quality-multimask-20260516/official/summary.json`
- Diagnostic non-multimask row:
  `outputs/baseplus-q8_0-1024-inline-quality-frames-20260516/official/summary.json`

The same `--multimask` speed contract was refreshed across all SAM2.1 q8_0
sizes at 1024. The result remains not PyTorch-fast:

| Model | C++ track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio |
| --- | ---: | ---: | ---: |
| Tiny q8_0 | `32.6` | `27.87` | `0.855` |
| Small q8_0 | `35.6` | `30.09` | `0.845` |
| Base+ q8_0 | `50.3` | `42.79` | `0.851` |
| Large q8_0 | `121.3` | `75.96` | `0.626` |

This confirms the current CUDA path is closer after inline-pack, but still
behind official PyTorch on every checked SAM2.1 size. Large remains the biggest
gap, and Base+ still needs about a `15%` speedup on this matched row.

512 fallback quality was also rechecked with shared extracted frames and f16
logit capture. This avoids the older video-vs-frame decode difference and gives
better numbers than the previous 512 quality matrix, but it still does not meet
the strict `>=0.99` mask-IoU bar:

| Encode size | Precision | Mean mask IoU | Min mask IoU | Candidate selection |
| ---: | --- | ---: | ---: | --- |
| 512 | f16 | `0.966113` | `0.958605` | propagation `9/9`, initial selected candidate matches |
| 1024 | q8_0 | `0.995071` | `0.993410` | strict-quality pass |

The 512 logit comparison shows this is not fixed by a simple threshold offset.
For the worst frames, most XOR pixels are Python-only positives; C++ selected
logits are about `0.3-0.4` lower around the boundary. Sweeping a global C++
logit shift only reaches mean/min `0.9709 / 0.9477` at `+0.10`, so the gap is in
the 512 logit/resize/conditioning contract rather than just the final zero
threshold.

Evidence:

- `outputs/model-matrix-sam21-all-q8_0-1024-inline-multimask-20260516/summary.json`
- `outputs/baseplus-f16-512-logit-diagnostic-20260516/official/summary.json`
- `outputs/baseplus-f16-512-logit-diagnostic-20260516/logit-compare.json`
- `outputs/baseplus-f16-512-logit-diagnostic-20260516/threshold-shift-sweep.json`
- `outputs/goal-audit/current-inline-multimask-20260516b-summary.json`

## F32 Batched Bias Fusion Direction

The next fusion-kernel direction was reopened with an isolated benchmark before
touching the inference graph. The target is the Hiera F32 batched projection
shape:

```text
weights=f32[112,336], input=f32[112,64,1024], bias=f32[336],
output=f32[336,64,1024]
```

The benchmark now has three CUDA paths:

- default ggml CUDA graph path: batched cuBLAS GEMM plus the existing graph bias
  add;
- `--custom-cuda`: a simple hand-written tiled GEMM+bias kernel;
- `--cublaslt-bias`: cuBLASLt strided-batched GEMM with `EPILOGUE_BIAS`.

The 100-iteration isolated measurement is:

| Path | Mean ms | Median ms | p95 ms | Check result |
| --- | ---: | ---: | ---: | --- |
| ggml CUDA | `0.844560` | `0.844732` | `0.846845` | sampled `bad=0`, max abs `1.475e-05` |
| cuBLASLt bias epilogue | `0.562512` | `0.588224` | `0.616160` | sampled `bad=0`, max abs `1.49e-08` |
| hand-written tiled CUDA | `2.226508` | `2.242624` | `2.248928` | sampled `bad=0`, max abs `1.49e-08` |

This rejects the naive custom GEMM fusion path for this shape. The viable
fusion direction is cuBLASLt epilogue integration, because it keeps the vendor
GEMM body and removes the separate bias-add work. It must remain behind an
opt-in gate until a full tracking parity run passes: the earlier graph-level
cuBLASLt epilogue probe changed one of five full-mask rows, so isolated numeric
correctness is necessary but not sufficient for default enablement.

A graph-level opt-in retry first failed for the strict-FP32 cuBLASLt epilogue.
The direct integration crashed during CUDA graph capture, so the probe was
narrowed to SAM3's default no-graph execution policy. The isolated F32 graph
benchmark still showed the expected local speedup with graphs disabled:

| Path | Mean ms | Median ms | p95 ms | Check result |
| --- | ---: | ---: | ---: | --- |
| ggml CUDA, no graphs | `0.847812` | `0.848106` | `0.849835` | sampled `bad=0`, max abs `1.475e-05` |
| cuBLASLt bias epilogue, no graphs | `0.551716` | `0.601407` | `0.611667` | sampled `bad=0`, max abs `1.49e-08` |

However, the 10-frame Base+ q8_0 1024 full-mask tracking parity check failed for
the strict-FP32 cuBLASLt epilogue:
`mask_hash_equal_rows=0/10`, `min_bbox_iou=0.8660235798`,
`max_bbox_delta_px=125`, and `max_score_abs_delta=0.002733`. The culprit was
not the bias epilogue itself but the compute mode: strict-FP32 cuBLASLt no
longer matched the existing cuBLAS TF32 rounding closely enough.

The accepted retry uses cuBLASLt `CUBLAS_COMPUTE_32F_FAST_TF32`, keeps the path
behind `GGML_CUDA_ENABLE_CUBLASLT_BIAS_FUSION=1`, and still requires
`GGML_CUDA_DISABLE_GRAPHS=1` so it does not enter CUDA graph capture. This
matches SAM3's default graph policy. The isolated Hiera-shape graph benchmark
then measures `0.396676 ms` with sampled `bad=0` and the same max abs error as
the existing cuBLAS path (`1.475e-05`).

Full-mask parity on the 10-frame Base+ q8_0 1024 sample is exact:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`,
`max_score_abs_delta=0.0`, and `max_abs_mask_area_rel_delta=0.0`.

Five paired bbox-only runs show a small but stable improvement:

| Path | Track ms/frame values | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| Default | `50.3, 48.8, 48.5, 49.1, 49.2` | `49.18` | `49.1` | `0.68` |
| cuBLASLt TF32 bias fusion | `47.4, 47.0, 47.5, 46.7, 47.5` | `47.22` | `47.4` | `0.36` |

The paired mean delta is `1.96 ms/frame`, about `3.99%` of the default mean.
This is an accepted opt-in fusion candidate, not a PyTorch-closing default yet:
it should remain opt-in until broader model/precision coverage is checked.

The implementation was then tightened to cache a cuBLASLt handle in the CUDA
backend context rather than creating and destroying the handle in every fused
node. The isolated Hiera-shape timing stayed effectively unchanged
(`0.396365 ms` mean) but the implementation is cleaner and avoids repeated
handle lifecycle work. The cached-handle full-mask parity check remains exact:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, and
`max_bbox_delta_px=0.0`.

The fusion guard was then widened from only batched Hiera QKV/projection shapes
to batch-one F32 `MUL_MAT + broadcast bias ADD` shapes as well, still behind the
same `GGML_CUDA_ENABLE_CUBLASLT_BIAS_FUSION=1` opt-in and no-graphs requirement.
This catches large projection rows such as the F32 MLP/fpn/proj bias adds that
were previously left as a separate ADD kernel. Full-mask parity on the same
10-frame Base+ q8_0 1024 sample remains exact:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`,
`max_score_abs_delta=0.0`, and `max_abs_mask_area_rel_delta=0.0`.

Five paired bbox-only runs show the widened fusion is a larger improvement than
the batched-only version:

| Path | Track ms/frame values | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| Default | `49.6, 49.6, 48.6, 48.7, 48.6` | `49.02` | `48.7` | `0.53` |
| cuBLASLt TF32 bias fusion, widened | `46.5, 45.9, 46.1, 45.9, 46.3` | `46.14` | `46.1` | `0.26` |

The paired mean delta is `2.88 ms/frame`, about `6.24%` of the default mean.
Node profiling with the widened guard confirms the separate large F32 ADD row
`dst=f32[448,65536,1,1]` disappears from the top hotspot list, while the
remaining dominant work is still global/window FlashAttention and quantized MLP
matmul bodies. That points the next kernel work at D56/D256 FlashAttention and
q8_0 MMQ body throughput rather than more F32 bias epilogues.

The widened opt-in was also rerun against the SAM2.1 q8_0 1024 model matrix
using the same matched official-Python results. Base+ improves to
`46.1 ms/frame`, but all four SAM2.1 q8_0 rows are still slower than PyTorch:

| Model | C++ widened opt-in track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio |
| --- | ---: | ---: | ---: |
| Tiny q8_0 | `32.7` | `27.87` | `0.852` |
| Small q8_0 | `34.9` | `30.09` | `0.862` |
| Base+ q8_0 | `46.1` | `42.79` | `0.928` |
| Large q8_0 | `118.8` | `75.96` | `0.639` |

A follow-up attempt cached the cuBLASLt heuristic result by shape. It preserved
full-mask parity but did not improve the five-pair Base+ run (`46.82 ms/frame`
enabled mean versus `46.14 ms/frame` for the simpler widened path), so the cache
was removed. The kept fusion remains the smaller, measured-winner change.

Two existing D56 FlashAttention switches were retested after the widened
cuBLASLt fusion because the remaining hotspot list is still dominated by
`FLASH_ATTN_EXT`. Profiling suggested that disabling inline pack or direct
output could reduce some synchronized MMA timings, but normal bbox-only paired
runs showed both are slower end-to-end:

| FATTN56 mode | Track ms/frame values | Mean | Delta vs default |
| --- | --- | ---: | ---: |
| Default | `47.1, 46.2, 46.3, 46.6, 46.3` | `46.50` | baseline |
| `GGML_CUDA_DISABLE_FATTN56_INLINE_PACK=1` | `47.9, 48.3, 48.8, 47.7, 48.1` | `48.16` | `-1.66 ms/frame` |
| `GGML_CUDA_DISABLE_FATTN56_DIRECT_OUT=1` | `48.0, 48.5, 47.6, 48.1, 48.0` | `48.04` | `-1.54 ms/frame` |

The existing default FATTN56 path remains the best checked option.

The q8_0 MMQ X cap was also re-swept under the widened fusion. The default X
selection remains best, and the q8_0 196-column cap is clearly worse:

| MMQ mode | Track ms/frame values | Mean |
| --- | --- | ---: |
| Default | `46.2, 46.6, 46.6` | `46.47` |
| `GGML_CUDA_MMQ_X_MAX=96` | `47.1, 47.4, 47.3` | `47.27` |
| `GGML_CUDA_MMQ_X_MAX=104` | `46.9, 47.4, 47.2` | `47.17` |
| `GGML_CUDA_MMQ_X_MAX=112` | `47.1, 46.6, 47.1` | `46.93` |
| `GGML_CUDA_MMQ_X_MAX=128` | `48.0, 46.4, 46.3` | `46.90` |
| `GGML_CUDA_ENABLE_MMQ_Q8_0_196_X64=1` | `47.8, 49.9, 50.3` | `49.33` |

Finally, Base+ all-precision 1024 was refreshed with the widened fusion. q4_0
and q4_1 are the fastest current C++ rows at `47.5 ms/frame`, but this still
does not beat official PyTorch bf16 at `42.79 ms/frame`:

| Precision | C++ track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio |
| --- | ---: | ---: | ---: |
| q4_0 | `47.5` | `42.79` | `0.901` |
| q4_1 | `47.5` | `42.79` | `0.901` |
| q8_0 | `51.3` | `42.79` | `0.834` |
| mxfp4 | `50.9` | `42.79` | `0.841` |
| f16 | `60.6` | `42.79` | `0.706` |
| f32 | `69.1` | `42.79` | `0.619` |

The same opt-in was checked against the SAM2.1 q8_0 1024 model matrix using the
previous matched official-Python results. It improves Base+ relative to the
pre-fusion matrix but still does not beat PyTorch:

| Model | C++ opt-in track ms/frame | PyTorch bf16 ms/frame | PyTorch/C++ ratio |
| --- | ---: | ---: | ---: |
| Tiny q8_0 | `32.7` | `27.87` | `0.852` |
| Small q8_0 | `36.9` | `30.09` | `0.816` |
| Base+ q8_0 | `46.7` | `42.79` | `0.916` |
| Large q8_0 | `118.6` | `75.96` | `0.641` |

The clean SAM2.1-only rerun was noisier (`Base+ 51.5 ms/frame`), so the
single-matrix result should not replace the paired Base+ A/B as the optimization
effect size. The conclusion is unchanged: cuBLASLt TF32 bias fusion is useful
and parity-preserving, but PyTorch-level speed still requires the larger
D56/D72 FlashAttention and MMQ body work.

An additional fusion-kernel probe tried to fuse the propagation elementwise
chain around `f32[1,128,4096,1]` by adding a two-stage F32 same-shape kernel for
`SUB -> MUL` and `MUL -> ADD`. The safe default path did not fire because ggml's
fusion memory-range guard rejects the scratch-buffer overlap pattern. An
unsafe opt-in confirmed the kernel can fire, but it is not shippable: bbox/mask
rows stayed structurally identical, while decoder scores changed by about
`1e-4`, and speed was only noise-level (`46.72 ms/frame` default mean versus
`46.54 ms/frame` unsafe opt-in mean across five runs). The experimental kernel
was therefore removed rather than kept behind a flag.

The remaining FlashAttention knobs were also rechecked under the widened
cuBLASLt bias fusion. D56 staging pack choices are parity-safe but not a clear
speed win:

| FATTN56 pack mode | Track ms/frame values | Mean | Median |
| --- | --- | ---: | ---: |
| Default | `47.3, 46.0, 46.6, 46.8, 46.2` | `46.58` | `46.6` |
| `GGML_CUDA_DISABLE_FATTN56_ROW4_PACK=1 GGML_CUDA_DISABLE_FATTN56_MIXED_ROW4_PACK=1` | `45.9, 46.7, 46.3, 46.3, 47.2` | `46.48` | `46.3` |
| `GGML_CUDA_DISABLE_FATTN56_CONTIGUOUS_PACK=1` | `46.6, 46.5, 46.3, 46.8, 47.9` | `46.82` | `46.6` |

Both non-default D56 pack modes produced bbox-only JSONL identical to default,
but the mean deltas are within run noise, so no default change was made.

The D256 propagation FlashAttention stream-k toggle is not usable:
`GGML_CUDA_DISABLE_FATTN_STREAM_K=1` regressed the same 5-run smoke from
`46.52` to `47.92 ms/frame` and changed tracking output, including the initial
mask hash and later selected candidate rows. The current stream-k path remains
the only acceptable default for this workload.

The next accepted fusion kernel is narrower and targets SAM's explicit complex
RoPE graph. `sam3_apply_rope` expresses the rotation as
`MUL, MUL, SUB, MUL, MUL, ADD, CONCAT, CONT`; CUDA now recognizes that pattern
across intervening `VIEW`/`RESHAPE` nodes and writes the final interleaved
`[2, half, tokens, heads*batch]` tensor directly. The fused kernel keeps the
original two rounded multiplies followed by a rounded add/subtract, so it avoids
FMA-induced score drift.

This is a launch/memory-traffic cleanup, not the main PyTorch-closing path.
The q8_0/Base+/1024 full-mask parity check remained exact:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, and
`max_score_abs_delta=0.0`. In a 3-frame profile the enabled path fused `30`
RoPE pairs and reduced the matching propagation-style `f32[1,128,4096,1]`
RoPE rows from `128` MUL / `32` SUB / `32` ADD / `32` CONCAT nodes to `38` MUL
/ `2` SUB / `2` ADD / `2` CONCAT nodes. The remaining two rows are shape cases
outside the current narrow matcher.

The end-to-end speed effect is small but measurable enough to keep: 7 paired
q8_0/Base+/1024 bbox-only runs changed from `48.66 ms/frame` disabled to
`47.94 ms/frame` enabled, saving `0.71 ms/frame` on average
(`~1.47%`, median `0.8 ms/frame`). The optimization therefore stays enabled by
default, with `GGML_CUDA_DISABLE_ROPE_PAIR_FUSION=1` available for diagnostics.

The first structural q8_0 MMQ body follow-up was rejected. The experiment
changed the NVIDIA `vec_dot_q8_0_q8_1_mma` branch to keep only one K fragment of
`A`/`dA` live at a time instead of preloading all fragments before the J loop.
This preserved full-mask tracking parity against the previous default
(`10/10` mask hashes, zero bbox/score deltas), but it made Base+ q8_0 1024
slower: the 7-run bbox-only comparison moved from the RoPE-fused baseline mean
`47.94 ms/frame` to `48.61 ms/frame`, a `0.67 ms/frame` regression
(`-1.40%`). The likely reason is that the lower live range did not compensate
for the less favorable loop structure and repeated scheduling pressure. The
patch was removed; future MMQ body work should use ptxas/register evidence and
avoid changing the working J/K tile schedule unless it wins end-to-end.

The next fusion-kernel step targets large-model D72 attention. The previous
D72 pad-MMA path already moved `D=72` from the generic tile kernel to the
`DKQ=80,DV=80` MMA body, but it still launched a separate Q/K/V pack kernel and
usually a slice step. CUDA now has a default-on inline-pack path for contiguous
D72 outputs: the MMA kernel consumes original f32 head72 Q/K/V tensors directly
and writes the head72 output directly, using `DKQ=80,DV=80,DV_DST=72,D_SRC=72`.
`GGML_CUDA_DISABLE_FATTN72_INLINE_PACK=1` keeps the previous packed path for
diagnostics.

Standalone `sam3_fattn_parity` checks stayed within the existing tolerance:

| Shape | Disabled steady FATTN72 | Inline steady FATTN72 | Parity |
| --- | ---: | ---: | --- |
| `D=72,N=4096,H=8,B=1` | `~1.06 ms` | `~0.96 ms` | `bad=0/5184` sampled |
| `D=72,N=256,H=8,B=16` | `~0.236 ms` | `~0.184 ms` | `bad=0/2359296` |

The end-to-end large q8_0 1024 bbox-only check also preserved exact tracker
parity against the disabled path (`10/10` mask hashes, zero bbox and score
deltas). Five repeated large q8_0 runs improved from `99.8 ms/frame` disabled
to `95.0 ms/frame` inline, saving `4.8 ms/frame` (`~4.8%`) on that large-model
path. This does not close the all-size q8_0 gap by itself, but it is a useful
fusion-kernel win for the large row and keeps the implementation aligned with
the D56 inline-pack direction.

The all-size SAM2.1 q8_0/1024 matrix was refreshed after enabling D72 inline
pack while reusing the same official PyTorch result set. C++ remains slower on
every comparable row, so the active goal is still open:

| Model | C++ CUDA q8_0 | PyTorch CUDA bf16 | PyTorch / C++ |
| --- | ---: | ---: | ---: |
| tiny | `32.0 ms/frame` | `27.79 ms/frame` | `0.87` |
| small | `37.5 ms/frame` | `30.15 ms/frame` | `0.80` |
| base+ | `53.1 ms/frame` | `42.67 ms/frame` | `0.80` |
| large | `95.5 ms/frame` | `75.62 ms/frame` | `0.79` |

The useful change is concentrated in large, where D72 attention is present.
Next fusion-kernel work should therefore pivot back to shared q8_0 MMQ body
throughput and D56/D256 attention body throughput; D72 launch/pack overhead is
no longer the dominant remaining large-model problem.

A short Base+ q8_0/1024 attribution profile after the D72 change confirms that
same priority. Profiling disables CUDA graphs and synchronizes around nodes, so
the numbers below are only a hotspot ranking. Dropping first-use outliers, the
top remaining buckets are:

| Bucket | Drop-max sum | Mean | Count |
| --- | ---: | ---: | ---: |
| q8_0 MLP expansion `dst=f32[1792,4096]` | `12.44 ms` | `0.1575 ms` | `80` |
| D56 global FlashAttention `dst=f32[56,8,4096,1]` | `11.90 ms` | `0.8499 ms` | `15` |
| q8_0 MLP projection `dst=f32[448,4096]` | `11.43 ms` | `0.1446 ms` | `80` |
| q8_0 window qkv `dst=f32[1344,196,25]` | `10.05 ms` | `0.1703 ms` | `60` |
| D56 window FlashAttention `dst=f32[56,8,196,25]` | `9.59 ms` | `0.1625 ms` | `60` |
| D256 propagation FlashAttention `src1=f32[256,4100,1,1]` | `6.54 ms` | `0.4357 ms` | `16` |
| D256 propagation FlashAttention `src1=f32[256,4096,1,1]` | `6.50 ms` | `0.4333 ms` | `16` |

The FATTN56 profiler also shows that the current D56 inline path has already
removed almost all pack/slice overhead for the common window and q-pool rows.
The remaining D56 cost is now the MMA body itself, not another staging kernel.

The next shared fusion step promotes the F32 batched `MUL_MAT -> ADD(channel
bias)` cuBLASLt epilogue path to the normal CUDA-graph route. Earlier versions
only allowed this path when CUDA graphs were disabled, even though the hot
Hiera QKV and projection shapes are `cols=64,batches=1024` and therefore too
wide for the existing custom MMF bias epilogue. The CUDA backend now enables
the cuBLASLt TF32 bias epilogue by default (`GGML_CUDA_ENABLE_CUBLASLT_BIAS_FUSION=0`
disables it; `=1` keeps the old no-graph-only behavior; `=2` is the default
graph-capable mode).

Base+ q8_0/1024 bbox-only parity against the disabled path remained exact:
`10/10` mask hashes, zero bbox delta, zero score delta, and zero mask-area
delta. Five paired Base+ runs improved from `48.0 ms/frame` disabled to
`45.2 ms/frame` enabled, saving `2.8 ms/frame` on average (`~6.2%`). A single
all-size SAM2.1 q8_0 smoke also completed on all four sizes; tiny, base+, and
large improved in that run, while a follow-up five-pair small-only run showed
small is effectively neutral-to-slightly-positive (`34.8 -> 34.4 ms/frame`) and
still exact on every pair. This keeps the fusion enabled by default while
leaving the environment switch available for diagnosing cuBLASLt or graph
capture regressions.

The next cuBLASLt fusion pass extends that F32 batched epilogue from bias-only
to `GELU_BIAS` for the exact `MUL_MAT -> ADD(channel bias) -> GELU` pattern.
It is intentionally limited to ggml's tanh-approximate `GELU`; `GELU_ERF` is
not redirected through cuBLASLt because that would risk changing numerics. The
path is enabled by default and can be disabled with
`GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_FUSION=0`.

Base+ q8_0/1024 bbox-only parity against the bias-only default remained exact
in five paired runs: every pair had `10/10` mask hashes, zero bbox delta, zero
score delta, and zero mask-area delta. The speed effect is smaller than the
plain bias epilogue promotion but positive in this run: mean track time moved
from `45.8 ms/frame` to `45.4 ms/frame`, saving `0.4 ms/frame` on average.
This keeps the fusion as a narrow default-on cleanup, while the remaining
PyTorch gap still has to come from MMQ/FATTN body throughput rather than
post-op launch removal.

After this promotion, the remaining fusion-kernel work should move away from
channel-bias launch removal and focus on kernels whose body time still dominates:
q8_0 MMQ MLP/window-QKV body throughput, D56 FlashAttention MMA body throughput,
and D256 propagation FlashAttention. Those require kernel-internal scheduling or
tile-shape changes, not another post-op fusion.

The matched SAM2.1 q8_0/1024 matrix was refreshed after the cuBLASLt graph
promotion and D72 inline-pack change while reusing the same official-Python
result set. C++ is still behind PyTorch on every comparable row, but Base+ is
now the closest row:

| Model | C++ CUDA q8_0 | PyTorch CUDA bf16 | PyTorch / C++ |
| --- | ---: | ---: | ---: |
| tiny | `36.0 ms/frame` | `27.79 ms/frame` | `0.77` |
| small | `40.5 ms/frame` | `30.15 ms/frame` | `0.74` |
| base+ | `46.0 ms/frame` | `42.67 ms/frame` | `0.93` |
| large | `91.6 ms/frame` | `75.62 ms/frame` | `0.83` |

The matrix comparison wrapper was also tightened so repeated
`scripts/model_matrix_compare.py --filter` arguments no longer make the C++
benchmark run an unnecessarily broad model set. The benchmark binary supports
one substring filter, so the wrapper now forwards the most selective token and
still post-filters rows with all requested tokens. For the SAM2.1 q8_0 matrix
this changes the benchmark invocation from a broad `sam2.1` run to a direct
`q8_0` run, reducing measurement noise from unrelated precision/model rows.

With that filter fix and the current default cuBLASLt `GELU_BIAS` path, the
comparable q8_0/1024 rows are:

| Model | C++ CUDA q8_0 | PyTorch CUDA bf16 | PyTorch / C++ |
| --- | ---: | ---: | ---: |
| tiny | `34.8 ms/frame` | `27.79 ms/frame` | `0.80` |
| small | `33.9 ms/frame` | `30.15 ms/frame` | `0.89` |
| base+ | `46.2 ms/frame` | `42.67 ms/frame` | `0.92` |
| large | `92.4 ms/frame` | `75.62 ms/frame` | `0.82` |

This does not change the conclusion: Base+ is close but still needs about
`3.5 ms/frame` more, and the all-size q8_0 target is still open.

A fresh Base+ q8_0/1024 profile after this change ranks the remaining
drop-max buckets as:

| Bucket | Drop-max sum | Mean | Count |
| --- | ---: | ---: | ---: |
| q8_0 MLP expansion `dst=f32[1792,4096]` | `12.16 ms` | `0.1539 ms` | `80` |
| D56 global FlashAttention `dst=f32[56,8,4096,1]` | `11.53 ms` | `0.8236 ms` | `15` |
| q8_0 MLP projection `dst=f32[448,4096]` | `11.22 ms` | `0.1420 ms` | `80` |
| q8_0 window qkv `dst=f32[1344,196,25]` | `9.80 ms` | `0.1661 ms` | `60` |
| D56 window FlashAttention `dst=f32[56,8,196,25]` | `9.39 ms` | `0.1592 ms` | `60` |
| D256 propagation FlashAttention `src1=f32[256,4100,1,1]` | `6.38 ms` | `0.4256 ms` | `16` |
| D256 propagation FlashAttention `src1=f32[256,4096,1,1]` | `6.36 ms` | `0.4241 ms` | `16` |

Two immediate kernel-body probes were rejected. Reapplying q8_0 MMQ X caps on
the current default did not produce a usable win: `GGML_CUDA_MMQ_X_MAX=96/104`
kept rounded speed the same but changed one mask hash, `112/120` produced a
large bbox delta, and `128` was identical to the default. Changing the D256
MMA config for `ncols=64` from `nbatch_fa=32` to `64` also regressed speed
(`47.2 ms/frame`) and changed scores despite matching mask hashes. Both patches
were left out. The next viable direction is therefore not a simple cap/config
retune; it needs a deeper MMQ or FATTN body change with strict parity.

A third stream-k scheduling probe was also rejected. Lowering the stream-k
tiling threshold from `90%` to `60%` avoids some fixup launches, but it changes
the accumulation schedule enough to move tracker output (`9/10` mask hashes,
max bbox delta `7.5 px`) and it did not improve Base+ speed (`47.0 ms/frame` in
the single probe). The threshold remains unchanged.

Evidence:

- `outputs/f32-cublaslt-bias-bench-20260516/ggml-cuda.log`
- `outputs/f32-cublaslt-bias-bench-20260516/cublaslt-bias.log`
- `outputs/f32-cublaslt-bias-bench-20260516/custom-tiled.log`
- `outputs/cublaslt-graph-fusion-probe-20260516/default-nographs.log`
- `outputs/cublaslt-graph-fusion-probe-20260516/enabled-nographs.log`
- `outputs/cublaslt-graph-fusion-probe-20260516/enabled-graphs-guard.log`
- `outputs/cublaslt-bias-fusion-q8_0-1024-20260516/parity.json`
- `outputs/cublaslt-graph-fusion-probe-20260516/cublaslt-bias-tf32.log`
- `outputs/cublaslt-graph-fusion-probe-20260516/enabled-tf32-nographs.log`
- `outputs/cublaslt-graph-fusion-probe-20260516/tf32-enabled-graphs-guard.log`
- `outputs/cublaslt-bias-tf32-fusion-q8_0-1024-20260516/parity.json`
- `outputs/cublaslt-bias-tf32-fusion-q8_0-1024-20260516/ab-summary.json`
- `outputs/cublaslt-graph-fusion-probe-20260516/enabled-tf32-cached-handle-nographs.log`
- `outputs/cublaslt-bias-tf32-cached-handle-q8_0-1024-20260516/parity.json`
- `outputs/cublaslt-bias-tf32-wide-q8_0-1024-20260516/parity.json`
- `outputs/cublaslt-bias-tf32-wide-q8_0-1024-20260516/ab-summary.json`
- `outputs/cublaslt-bias-tf32-wide-q8_0-1024-20260516/hotspots-all.json`
- `outputs/model-matrix-sam21-all-q8_0-1024-cublaslt-bias-tf32-wide-20260516/summary.json`
- `outputs/cublaslt-bias-tf32-wide-cached-algo-q8_0-1024-20260516/ab-summary.json`
- `outputs/fattn56-option-speed-q8_0-1024-20260516/summary.json`
- `outputs/mmq-xmax-sweep-wide-q8_0-1024-20260516/summary.json`
- `outputs/model-matrix-baseplus-allprec-1024-cublaslt-bias-wide-20260516/summary.json`
- `outputs/model-matrix-sam21-all-q8_0-1024-cublaslt-bias-tf32-20260516/summary.json`
- `outputs/model-matrix-sam21-only-q8_0-1024-cublaslt-bias-tf32-20260516/summary.json`
- `outputs/bin-bin-fusion-q8_0-1024-20260516/profile-unsafe-v1.log`
- `outputs/bin-bin-fusion-q8_0-1024-20260516/ab/`
- `outputs/bin-bin-fusion-q8_0-1024-20260516/default.jsonl`
- `outputs/bin-bin-fusion-q8_0-1024-20260516/unsafe.jsonl`
- `outputs/fattn56-pack-mode-wide-q8_0-1024-20260516/summary.json`
- `outputs/fattn56-pack-mode-wide-q8_0-1024-20260516/default.jsonl`
- `outputs/fattn56-pack-mode-wide-q8_0-1024-20260516/no_row4_pack.jsonl`
- `outputs/fattn56-pack-mode-wide-q8_0-1024-20260516/no_contig_pack.jsonl`
- `outputs/fattn256-streamk-wide-q8_0-1024-20260516/summary.json`
- `outputs/fattn256-streamk-wide-q8_0-1024-20260516/default.jsonl`
- `outputs/fattn256-streamk-wide-q8_0-1024-20260516/no_stream_k.jsonl`
- `outputs/rope-pair-fusion-q8_0-1024-20260516/parity2/summary.json`
- `outputs/rope-pair-fusion-q8_0-1024-20260516/profile2/fusion-counts.json`
- `outputs/rope-pair-fusion-q8_0-1024-20260516/ab2/summary.json`
- `outputs/mmq-q8-live-range-stream-kfrag-20260516/parity/summary-vs-prev.json`
- `outputs/mmq-q8-live-range-stream-kfrag-20260516/ab/summary.json`
- `outputs/fattn72-inline-pack-20260516/global-inline-profile.log`
- `outputs/fattn72-inline-pack-20260516/global-disabled-profile.log`
- `outputs/fattn72-inline-pack-20260516/batched-inline-profile.log`
- `outputs/fattn72-inline-pack-20260516/batched-disabled-profile.log`
- `outputs/fattn72-inline-pack-20260516/large-disabled-vs-inline.json`
- `outputs/fattn72-inline-pack-20260516/repeats/summary.json`
- `outputs/model-matrix-sam21-all-q8_0-1024-fattn72-inline-20260516/summary.json`
- `outputs/current-profile-after-fattn72-inline-20260516/benchmark-summary.json`
- `outputs/current-profile-after-fattn72-inline-20260516/mmq-summary.json`
- `outputs/current-profile-after-fattn72-inline-20260516/fattn56-summary.json`
- `outputs/fusion-next-cublaslt-graph2-20260516/repeats/summary.json`
- `outputs/fusion-cublaslt-default-20260516/disabled-vs-default.json`
- `outputs/model-matrix-sam21-q8_0-1024-cublaslt-default-20260516/summary.json`
- `outputs/model-matrix-sam21-q8_0-1024-cublaslt-default-20260516/disabled-vs-default-speed.json`
- `outputs/fusion-cublaslt-small-pairs-20260516/summary.json`
- `outputs/fusion-cublaslt-gelu-20260516/repeats/summary.json`
- `outputs/fusion-cublaslt-gelu-20260516/current-default-vs-disabled/summary.json`
- `outputs/model-matrix-filter-selectivity-smoke-20260516/summary.json`
- `outputs/model-matrix-sam21-q8_0-1024-filter-fixed-20260516/summary.json`
- `outputs/model-matrix-sam21-all-q8_0-1024-cublaslt-default-fattn72-20260516/summary.json`
- `outputs/current-profile-after-cublaslt-default-20260516/hotspots-all.json`
- `outputs/current-profile-after-cublaslt-default-20260516/mmq-summary.json`
- `outputs/current-profile-after-cublaslt-default-20260516/fattn56-summary.json`
- `outputs/mmq-xmax-current-cublaslt-q8_0-1024-20260516/summary.json`
- `outputs/fattn256-nbatch64-q8_0-1024-20260516/parity.json`
- `outputs/mmq-streamk-threshold-q8_0-1024-20260516/parity-th60.json`
- `outputs/goal-audit/current-cublaslt-bias-tf32-20260516-summary.json`
- `outputs/goal-audit/current-cublaslt-bias-tf32-wide-20260516-summary.json`

## 2026-05-16: Hiera projection bias fusion probe rejected

The next fusion-kernel probe expanded non-sequential MMQ bias fusion from the
existing Hiera qkv and selected projection rows to every Hiera
`.attn.proj.bias`. Profiling confirmed that the stage-2 projection pattern
`q8_0[448,448] x f32[448,196,25] -> ADD f32[448,1,1]` became a fused
`MUL_MAT` row, removing the standalone projection-bias `ADD` launches.

The result is not acceptable as a default optimization. Full-mask parity against
the previous default failed on the same 10-frame Base+ q8_0 1024 run:
`mask_hash_equal_rows=0/10`, `max_score_abs_delta=0.002225`, and the first row
had a `1 px` bbox delta. This matches the earlier wide-MMQ-bias result: fusing
these projection biases into the current stream-k MMQ path changes numeric
accumulation enough to move masks. The whitelist was therefore restored to the
previous default.

This probe narrows the safe fusion direction:

- CUBLASLt F32 `BIAS` / `GELU_BIAS` epilogues remain accepted because they keep
  exact tracker output in the measured Base+ q8_0 runs.
- Broader q8_0 MMQ bias fusion is not safe until the MMQ epilogue/fixup path can
  add bias without changing stream-k accumulation semantics.
- The remaining fusion work should focus on either exact-schedule MMQ epilogues,
  a redesigned stream-k finalization kernel, or non-MMQ launch reductions that
  preserve existing arithmetic order.

Evidence:

- `outputs/fusion-proj-bias-default-20260516/profile.log`
- `outputs/fusion-proj-bias-default-20260516/compare/parity-summary.json`

## 2026-05-16: Base+ q4_1 Blackwell MMQ tile cap accepted

The current Base+ all-precision matrix shows q4_1 as the fastest tested C++
precision at 1024 input size, but it is still behind official PyTorch CUDA bf16.
The later single-row final refresh with the accepted changes measured q4_1 at
`48.0 ms/frame`, while the matched official PyTorch Base+ bf16 baseline is
`42.6667 ms/frame`. The current comparable speed ratio is `0.889x`; closing the
remaining `5.33 ms/frame` gap requires deeper CUDA kernel work, not another
wrapper-level change.

The safe MMQ retune found in this pass is a Blackwell q4_1 X tile cap increase
from `96` to `104`. A sweep showed:

- `104`: exact parity, `min_bbox_iou=1.0`, `max_bbox_delta_px=0`,
  `max_score_abs_delta=0`, `mask_hash_equal_rows=10`.
- `112`: rejected, `min_bbox_iou=0.9696`, `max_bbox_delta_px=25`,
  `mask_hash_equal_rows=0`.
- `128`: rejected, `max_bbox_delta_px=1`, score moved by about `0.00595`,
  `mask_hash_equal_rows=0`.

The accepted cap is intentionally small. A cleaner five-pair run measured
default `47.0 ms/frame` vs cap-104 `46.8 ms/frame`; this is only a tiny win, but
it is exact and does not change the tracker output. The code now uses `104` for
Blackwell q4_1 in both the host predictor and MMQ launch selection.

Evidence:

- `outputs/model-matrix-baseplus-allprec-1024-current-20260516/summary.json`
- `outputs/model-matrix-baseplus-q4_1-current-final-1024-20260516/summary.json`
- `outputs/current-profile-baseplus-q4_1-1024-20260516/profile.log`
- `outputs/current-profile-baseplus-q4_1-1024-20260516/mmq-summary.json`
- `outputs/current-profile-baseplus-q4_1-1024-20260516/fattn56-summary.json`
- `outputs/mmq-xmax-q4_1-current-1024-20260516/summary.json`
- `outputs/mmq-x104-q4_1-pairs-20260516/summary.json`
- `outputs/mmq-x104-q4_1-default-20260516/parity-vs-prev-default.json`

## 2026-05-16: Additional fusion probes kept out of the default path

Two smaller fusion probes were measured after the q4_1 cap change.

`GGML_CUDA_ENABLE_MMF_BIAS_GELU_FUSION=1` was exact on the 10-frame full-mask
comparison, but speed-neutral in the paired run: both default and enabled
averaged `45.8 ms/frame`. It remains opt-in rather than default.

An MMVQ broadcast-bias fusion path was also prototyped for small quantized
matvec shapes such as the q-pool projection followed by `ADD f32[448,1,1,1]`.
The kernel side now has enough stride metadata to represent broadcast bias
safely, but the measured graph path did not produce a useful default win. A
five-pair run with the opt-in enabled measured disabled `47.54 ms/frame` vs
enabled `47.50 ms/frame`; parity was exact, but the result is effectively
speed-neutral and too small to treat as a default optimization.
The fusion therefore stays behind `GGML_CUDA_ENABLE_MMVQ_BROADCAST_BIAS_FUSION`
and is not part of the default optimization set.

This reinforces the next direction: the remaining gap is mostly in the hot
FATTN56 body and q4_1 MMQ body, while isolated launch-removal fusions around
small ADDs are too small or too schedule-sensitive to close the PyTorch gap.

Two additional CUDA-body probes were rejected while chasing the same gap.
`GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` keeps full-mask parity, but it regresses
Base+ q4_1 speed from `47.58` to `50.82 ms/frame` in a five-pair run. A
shape-limited q4_1 MMQ `ncols_max=196` experiment that preferred `mmq_x=96`
instead of the current `80` also kept parity, but regressed speed from `47.78`
to `48.02 ms/frame`. Allowing the head-56 inline-pack path for the large
single-sequence global-attention rows was also exact but slower
(`47.5 -> 48.9 ms/frame`). These results indicate that the current Hiera D56
and q4_1 MMQ heuristics are already near a local optimum for simple switches;
the remaining improvement likely needs a real kernel-body redesign rather than
a different existing variant.

A follow-up MMQ source-reuse probe checked whether activation quantization could
be cached by reusing the q8_1 conversion for the same source tensor. Data-pointer
reuse looked high, but tensor-pointer reuse was sparse: `394` profiled MMQ rows
collapsed to `372` unique `(tensor,data,shape)` groups, and the repeated groups
were mainly prompt/decoder tensors such as `prompt` and repeated image-key views.
The Hiera encode hot rows do not expose enough safe same-tensor reuse for a
q8_1 activation cache to close the PyTorch gap, so this remains a low-priority
decoder-side optimization rather than the next Hiera encode target.

The Base+ fallback quality check was refreshed after the latest ggml changes
using the same extracted JPEG frames for C++ and official PyTorch. Both rows use
the 960x540 decoded source, 10 frames, point `(315,250)`, image size `1024`, and
multimask initial prompting. q8_0 remains the better quality fallback: q8_0
reaches mean/min mask IoU `0.9951/0.9934`, while q4_1 reaches `0.9933/0.9871`.
Both match the official initial candidate selection, but propagation candidate
selection still diverges on later frames even when the final masks remain very
close. This keeps the fallback policy unchanged: use q8_0 for strict quality
fallback validation, and use q4_1 as the faster/smaller speed experiment.

Evidence:

- `outputs/mmf-bias-gelu-probe-20260516/parity.json`
- `outputs/mmf-bias-gelu-probe-20260516/summary.json`
- `outputs/mmvq-broadcast-bias-fusion-20260516/default-off/parity-vs-x104.json`
- `outputs/mmvq-broadcast-bias-fusion-20260516/summary.json`
- `outputs/mmvq-broadcast-bias-fusion-20260516/reject-log/profile.log`
- `outputs/current-profile-baseplus-q4_1-cap104-1024-20260516/mmq-summary.json`
- `outputs/current-profile-baseplus-q4_1-cap104-1024-20260516/fattn56-summary.json`
- `outputs/current-profile-baseplus-q4_1-cap104-1024-20260516/launch/mmq-launch-summary.json`
- `outputs/fattn56-native-v-q4_1-1024-20260516/summary.json`
- `outputs/fattn56-native-v-q4_1-1024-20260516/parity-native-v-vs-default.json`
- `outputs/mmq-q4_1-196-x96-20260516/summary.json`
- `outputs/mmq-q4_1-196-x96-20260516/parity-x96-vs-default.json`
- `outputs/fattn56-large-inline-q4_1-1024-20260516/summary.json`
- `outputs/fattn56-large-inline-q4_1-1024-20260516/parity-large-inline-vs-default.json`
- `outputs/mmq-src1-reuse-probe-20260516/reuse-by-tensor-summary.json`
- `outputs/stage-profile-baseplus-q4_1-current-1024-20260516/summary.json`
- `outputs/current-profile-baseplus-q4_1-cap104-1024-20260516/hotspots-all.json`
- `outputs/sam2-official-quality-current-q8_0-1024-final-20260516/summary.json`
- `outputs/sam2-official-quality-current-q4_1-1024-final-20260516/summary.json`

### Non-sequential MMQ bias fusion exact-mode probe

The fusion-kernel follow-up added an opt-in
`GGML_CUDA_ENABLE_MMQ_NONSEQ_BIAS_FUSION=exact` mode. This mode lets the
broader non-sequential MMQ bias matcher try additional rows only when the host
predictor says the stream-k schedule is exact. The schedule predictor was also
extended from q8_0 to q4_0/q4_1 so q4_1 experiments do not accidentally drop
the existing whitelist.

The default path remains unchanged: Base+ q4_1 1024 output before and after the
predictor extension is bit-identical (`mask_hash_equal_rows=10/10`,
`max_score_abs_delta=0.0`). The exact-mode probe is not acceptable as a default
optimization yet. On the same 10-frame q4_1 1024 bbox-only sample it measured
`47.3 -> 45.7 ms/frame` in a single run, but the output changed:
`mask_hash_equal_rows=9/10`, `min_bbox_iou=0.995299`, and
`max_score_abs_delta=0.009012`.

This keeps the fusion direction narrow: the next MMQ fusion kernel should
redesign stream-k finalization or preserve the old arithmetic order for the
bias epilogue. Name-based broadening, even with an exact-schedule gate, is still
too coarse for the tracker quality contract.

Splitting the broader projection-bias fusion by Hiera stage produced the same
conclusion. `proj_stage0`, `proj_stage1`, and `proj_stage3` preserve exact
10-frame output against the current default, while `proj_stage2` changes output
(`mask_hash_equal_rows=9/10`, `min_bbox_iou=0.995299`,
`max_score_abs_delta=0.009012`). The safe `qkv+proj_stage1+proj_stage3`
combination is also exact, but five paired bbox-only runs are effectively
noise-level: default `47.06 ms/frame` vs enabled `46.90 ms/frame`, a
`0.16 ms/frame` (`0.34%`) mean delta. This is not enough to justify a default
policy change.

Evidence:

- `outputs/mmq-nonseq-exact-fusion-20260516/parity-default-before-after-predictor.json`
- `outputs/mmq-nonseq-exact-fusion-20260516/parity-default-vs-exact-after-predictor.json`
- `outputs/mmq-nonseq-exact-fusion-20260516/default-after-predictor.log`
- `outputs/mmq-nonseq-exact-fusion-20260516/exact-after-predictor.log`
- `outputs/mmq-nonseq-proj-stages-q4_1-1024-20260516/proj_stage*/parity.json`
- `outputs/mmq-nonseq-proj-stages-q4_1-1024-20260516/qkv_stage1_stage3/parity.json`
- `outputs/mmq-nonseq-proj-stages-q4_1-1024-20260516/qkv_stage1_stage3_pairs/summary.json`
- `outputs/python-official-profile/base_plus_1024-final-20260516.json`

### Fusion-kernel ablation and q4_1 4096-column MMQ probe

The current fusion bundle is still worth keeping, but the ablation shows it is
not the remaining PyTorch-closing lever. On the same Base+ q4_1 1024 10-frame
bbox-only sample, disabling all CUDA graph fusions regresses the row from
`47.4` to `57.7 ms/frame`. The largest individual measured contributor is MMQ
bias/GELU fusion: disabling all MMQ bias fusion measured `50.6 ms/frame`, and
disabling only MMQ bias/GELU fusion measured `50.9 ms/frame`. Norm fusion is a
smaller but real contributor at `49.0 ms/frame`. Disabling add+unary or rope-pair
fusion was noise-level on this sample (`46.7` and `46.6 ms/frame`). All ablation
rows preserved the 10-frame mask hash output against the default.

This means the existing fusion kernels are already removing the cheap launch
overhead that is easy to remove. The remaining gap must be attacked in body
kernels: D56/D256 FlashAttention or the q4_1 MMQ MLP/window-QKV body. Small
post-op fusion can still be accepted when exact and measurable, but it should
not be the primary speed plan.

A shape-limited MMQ probe tested whether only the q4_1 `ncols_max=4096` Hiera
MLP rows could use a wider X cap than the current Blackwell default. The default
path after adding the opt-in hook remains exact against the previous default.
With `GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX=112` or `120`, one mask hash row
changed. With `128`, all mask hashes and bboxes matched, but scores changed
(`max_score_abs_delta=0.004063`). Lowering the same cap to `64` preserved exact
10-frame output, but a five-pair run was slower: default `46.94 ms/frame` vs
X=64 `47.28 ms/frame`. Single-run speed was also not a clean win across the
larger caps: default `47.4 ms/frame`, X=112 `46.6`, X=120 `51.2`, X=128
`46.7`. The hook is therefore diagnostic only and should not be enabled by
default.

Evidence:

- `outputs/mmq-q4_1-4096-xcap-20260516/parity-default-before-after.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/parity-x112.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/parity-x120.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/parity-x128.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/parity-x64.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/x64_pairs/summary.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/no_mmq_bias.log`
- `outputs/mmq-q4_1-4096-xcap-20260516/no_mmq_bias_gelu.log`
- `outputs/mmq-q4_1-4096-xcap-20260516/no_norm.log`
- `outputs/mmq-q4_1-4096-xcap-20260516/no_all_fusion.log`

### Accepted q4_1 MMQ full-tile fast path

A small q4_1 MMQ body optimization was accepted after the broader X-cap probes
failed to produce a safe default. Many Hiera q4_1 rows must launch the
`need_check=true` kernel because only the final row or column tile is partial,
but the interior tiles are complete. The new default path keeps the checked
kernel launch and stream-k schedule unchanged, but dispatches complete q4_1
tiles inside that kernel to the existing unchecked tile body. Tail tiles still
use the checked body. `GGML_CUDA_DISABLE_MMQ_Q4_1_FULL_TILE_FASTPATH=1` restores
the previous behavior.

The arithmetic schedule and output shape are unchanged, and the measured
tracker output is exact. Five paired Base+ q4_1 1024 bbox-only runs measured
previous behavior `47.02 ms/frame` vs the new default `46.66 ms/frame`, a
`0.36 ms/frame` (`0.77%`) mean improvement. Every pair had
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0.0`.

The microbench agrees with the direction on the hot checked-tile shapes:
`q4_1[1792,448] x f32[1792,4096] -> f32[448,4096]` improved from
`0.181419` to `0.170093 ms` with `--check`, and
`q4_1[448,1344] x f32[448,196,25] -> f32[1344,196,25]` improved from
`0.169645` to `0.151010 ms`. The large MLP expansion row was neutral
(`0.140778` to `0.140477 ms`).

CUDA graph replay was also rechecked on the same q4_1 1024 sample after the
fusion changes. It preserved output but was slower in the smoke run
(`48.1 ms/frame`), so graph replay remains off by default.

Evidence:

- `outputs/mmq-q4_1-full-tile-fastpath-20260516/mlp_proj_default.log`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/mlp_proj_enabled.log`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/window_qkv_default.log`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/window_qkv_enabled.log`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/mlp_exp_default.log`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/mlp_exp_enabled.log`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/pairs/summary.json`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/parity-enabled-vs-default.json`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/parity-enabled-vs-default-after-enable.json`
- `outputs/mmq-q4_1-full-tile-fastpath-20260516/parity-disabled-vs-default-after-enable.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/parity-cuda-graphs.json`
- `outputs/mmq-q4_1-4096-xcap-20260516/cuda_graphs.log`

### Accepted D256 FlashAttention no-mask fast path

The next fusion/specialization target was the D256 FlashAttention path used by
Hiera tracking without an attention mask or attention sinks. The previous CUDA
MMA kernel still carried the mask load/add and sink-rescale branches in the
template body and reserved shared memory for the mask tile even when both inputs
were absent. The new default selects a D256-only `no_mask_no_sinks` template
variant when `mask == nullptr` and `sinks == nullptr`; all masked or sinked
calls continue to use the original variant. `GGML_CUDA_DISABLE_FATTN256_NOMASK_FAST=1`
restores the previous behavior.

The FATTN microbench remains within the expected numerical tolerance:

- D256 N=4096: `bad=0/4096`, previous `0.453749 ms`, new `0.436761 ms`.
- D256 N=4100: `bad=0/4096`, previous `0.423457 ms`, new `0.438888 ms`.

Because the standalone result is shape-sensitive, adoption was decided with the
same Base+ q4_1 1024 tracker criterion. Five paired bbox-only runs measured
previous behavior `46.50 ms/frame` vs the new default `46.06 ms/frame`, a
`0.44 ms/frame` (`0.96%`) mean improvement. The A/B JSONL comparison was exact:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0.0`.

Evidence:

- `outputs/fattn256-nomask-fast-20260516/default_n4096.log`
- `outputs/fattn256-nomask-fast-20260516/disabled_n4096.log`
- `outputs/fattn256-nomask-fast-20260516/default_n4100.log`
- `outputs/fattn256-nomask-fast-20260516/disabled_n4100.log`
- `outputs/fattn256-nomask-fast-20260516/parity.json`
- `outputs/fattn256-nomask-fast-20260516/pairs/summary.json`

### Accepted D56 FlashAttention no-mask fast path

The same compile-time mask/sink pruning was extended to the Hiera D56 padded-MMA
FlashAttention path. D56 calls in this model already assert `mask == nullptr`
and `sinks == nullptr`; the new path passes the `no_mask_no_sinks` template
variant through the D56 pad/direct-output wrappers so the kernel body can omit
the mask tile load/add, sink rescale branch, and mask shared-memory reservation.
`GGML_CUDA_DISABLE_FATTN56_NOMASK_FAST=1` restores the previous D56 template
variant.

The full-mask A/B comparison against the disabled path was exact on the 10-frame
Base+ q4_1 1024 sample: `mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0`, and `max_score_abs_delta=0.0`.

Five paired bbox-only runs measured previous behavior `46.62 ms/frame` vs the
new default `46.14 ms/frame`, a `0.48 ms/frame` (`1.04%`) mean improvement.
Individual runs are still noisy, so this should be treated as another small
kernel-body cleanup rather than a full PyTorch-closing optimization.

Evidence:

- `outputs/fattn56-nomask-fast-20260517/parity.json`
- `outputs/fattn56-nomask-fast-20260517/default.log`
- `outputs/fattn56-nomask-fast-20260517/disabled.log`
- `outputs/fattn56-nomask-fast-20260517/pairs/summary.json`

### Probed ConvTranspose2D bias fusion

A concrete fusion-kernel probe was added for the SAM decoder upsampling pattern
`CONV_TRANSPOSE_2D -> ADD(channel bias)`. The opt-in path is enabled with
`GGML_CUDA_ENABLE_CONV_TRANSPOSE_BIAS_FUSION=1`. It computes the ConvTranspose2D
output and the biased ADD output in one CUDA launch, while preserving the
intermediate ConvTranspose2D tensor because the graph may keep it as a visible
node. `GGML_CUDA_PROFILE_CONV_TRANSPOSE_BIAS_FUSION=1` logs rejection reasons
when diagnosing the route.

The path now actually fires in node profiling: six fused ConvTranspose2D launches
were observed on the 2-frame Base+ q4_1 1024 profile. Full 10-frame JSONL parity
against the unfused route was exact: `mask_hash_equal_rows=10/10`,
`min_bbox_iou=1.0`, `max_bbox_delta_px=0`, and `max_score_abs_delta=0.0`.

It is not enabled by default because the five paired bbox-only timing did not
show a reliable mean win. The opt-in route measured `46.2 ms/frame` vs
`46.0 ms/frame` for the default route, although the median was `45.0` vs
`46.0 ms/frame`. This remains a useful low-risk fusion testbed, but not a
default performance win.

Evidence:

- `outputs/cuda-fusion-kernels-20260517/optin-rebuilt/profile-optin.log`
- `outputs/cuda-fusion-kernels-20260517/optin-rebuilt/parity.json`
- `outputs/cuda-fusion-kernels-20260517/fresh/pairs-v2/summary.json`

### Rejected MMQ src1 quantization cache

An MMQ `src1` q8_1 quantization cache was prototyped to reduce repeated
activation quantization inside the Hiera encode graph. The first implementation
violated the CUDA pool's LIFO free contract; after fixing that, parity failed
because graph allocator pointer reuse made a `data pointer + shape` cache key
unsafe. Tightening the key to the `ggml_tensor` identity then exposed further
interaction with temporary pool allocation and cublasLt workspace use.

The cache was removed rather than kept behind an environment variable. Any future
attempt should use graph-lifetime analysis or an explicit tensor identity and
lifetime contract, not raw device-pointer reuse.

Evidence:

- `outputs/cuda-fusion-kernels-20260517/mmq-cache-enabled.log`
- `outputs/cuda-fusion-kernels-20260517/mmq-cache-parity-v2.json`
- `outputs/cuda-fusion-kernels-20260517/mmq-cache-enabled-v3.log`

### Rejected D56 large inline-pack fusion

The next FATTN56 probe tested whether the large single-sequence D56 path should
inline the D56-to-D64 pack/slice handling into the padded-MMA route. The path was
kept behind `GGML_CUDA_ENABLE_FATTN56_INLINE_PACK_LARGE=1` and compared against
the current default.

The 10-frame Base+ q4_1 1024 JSONL parity was exact:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, `max_bbox_delta_px=0`, and
`max_score_abs_delta=0.0`. However, the performance result rejected it. The
profiled global D56 shape `Q[56,4096,8,1]` changed from `0.768 ms` mean total to
`0.935 ms`; removing the pack/slice launches made the MMA body slower enough to
lose overall. Five paired bbox-only runs likewise regressed from
`46.2 ms/frame` to `46.8 ms/frame`.

This narrows the FATTN56 fusion direction: launch-count reduction around padding
is not enough. The next useful work has to improve the D56 MMA body or its memory
access pattern directly.

Evidence:

- `outputs/fattn56-inline-large-20260517/parity.json`
- `outputs/fattn56-inline-large-20260517/fattn56-default-summary.json`
- `outputs/fattn56-inline-large-20260517/fattn56-enabled-summary.json`
- `outputs/fattn56-inline-large-20260517/pairs/summary.json`

### Current fusion-kernel priority after the 2026-05-17 probes

The refreshed Base+ q4_1 1024 profile after the accepted q4_1 MMQ full-tile,
D256 no-mask, and D56 no-mask changes still does not beat the Python official
run. Normal non-profiling C++ runs are around `46.1..46.2 ms/frame`, while the
official PyTorch CUDA bf16 reference is `42.625 ms/frame`.

The current CUDA-node profile is synchronization-heavy and should be used only
for attribution. With first-use outliers removed by `sum_ms_drop_max`, the
remaining stable hotspots are still compute-body dominated:

- `MUL_MAT`: `272.72 ms` drop-max sum over the 10-frame profile.
- `FLASH_ATTN_EXT`: `101.26 ms` drop-max sum.
- `ADD`: `39.29 ms` drop-max sum, mostly small post-ops already covered by
  existing fusion attempts.
- `IM2COL`: `10.42 ms` drop-max sum.
- `CONV_TRANSPOSE_2D`: `9.53 ms` drop-max sum.

The MMQ attribution confirms that q4_1 body work is the next larger target:
`src0=q4_1[448,1792] src1=f32[448,4096] dst=f32[1792,4096]` contributes
`25.81 ms` total across the profile, `q4_1[1792,448] -> f32[448,4096]`
contributes `23.80 ms`, and the windowed `q4_1[448,1344] -> f32[1344,196,25]`
shape contributes `22.05 ms`. FATTN56 remains the second body-level target:
global D56 `Q[56,4096,8,1]` contributes `23.03 ms`, while window D56
`Q[56,196,8,25]` contributes `16.68 ms`.

The practical fusion-kernel plan is therefore:

1. Prefer body-level q4_1 MMQ changes over another broadcast post-op fusion.
2. Keep D56 FATTN changes focused on the MMA body, not pack/slice launch removal.
3. Keep ConvTranspose2D and scalar ADD fusions opt-in unless they show a paired
   mean win, because their total budget is too small to close the PyTorch gap.

Two D56 body-level configuration probes were then tried and rejected:

- Changing the `DKQ=64,DV=64,ncols=64` FATTN MMA launch from `128` threads /
  occupancy `2` to `256` threads / occupancy `1` was slower and changed the
  saved parity sample (`mask_hash_equal_rows=9/10`). Global D56 regressed from
  `0.768 ms` to `1.062 ms` mean total.
- Keeping `128` threads but disabling register-resident Q preserved exact parity
  (`mask_hash_equal_rows=10/10`) but was still slower: global D56 changed to
  `0.797 ms`, and window D56 changed from `0.139 ms` to `0.146 ms`.

The existing D56 FATTN configuration is therefore still the best measured
variant among these fusion-body probes.

The q4_1 MMQ launch profile was also refreshed after these probes. The dominant
4096-column Hiera MLP shapes currently run with `mmq_x=96`, `mmq_y=128`,
`stream_k=1`, and no fixup. The high-volume 196-column window shapes run with
`mmq_x=80`, `mmq_y=128`, `stream_k=1`, and no fixup. This means the remaining
q4_1 MMQ gap is not primarily a fixup-kernel problem.

The previously rejected `ncols_max=4096` X-cap widening was retested on the
current build with `GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX=128`. It still should
not be promoted: the 10-frame sample kept mask hashes and bboxes equal, but
scores changed by up to `0.004063`, and the 5-pair bbox-only run regressed from
`46.38 ms/frame` to `48.48 ms/frame`.

An additional shape-limited `X=128` probe was then tried only for the larger
4096-column q4_1 shapes that the standalone `sam3_mmq_bench` showed as faster.
That variant preserved exact 10-frame tracking parity, but the 15-pair app-level
run measured only `0.093 ms/frame` mean saved with a 95% confidence interval of
`-0.125..0.311 ms`. The effect is below the current noise floor, so the code path
was removed instead of kept as another environment variable.

The next q4_1 MMQ work therefore needs to target the kernel body itself, not
Stream-K fixup avoidance or wider X-cap scheduling.

Evidence:

- `outputs/current-profile-baseplus-q4_1-after-fusion-probes-20260517/benchmark-summary.json`
- `outputs/current-profile-baseplus-q4_1-after-fusion-probes-20260517/mmq-summary.json`
- `outputs/current-profile-baseplus-q4_1-after-fusion-probes-20260517/fattn56-summary.json`
- `outputs/fattn64-ncols64-256threads-20260517/fattn56-summary.json`
- `outputs/fattn64-ncols64-256threads-20260517/parity-vs-prev.json`
- `outputs/fattn64-ncols64-q-shared-20260517/fattn56-summary.json`
- `outputs/fattn64-ncols64-q-shared-20260517/parity-vs-prev.json`
- `outputs/mmq-launch-current-q4_1-20260517/launch-summary.json`
- `outputs/mmq-q4_1-4096-x128-current-20260517/parity.json`
- `outputs/mmq-q4_1-4096-x128-current-20260517/bench-summary.json`
- `outputs/mmq-microbench-q4_1-current-20260517/summary-all.json`
- `outputs/mmq-q4_1-4096-large-x128-20260517/parity.json`
- `outputs/mmq-q4_1-4096-large-x128-20260517/pairs15/pairs-summary.json`

### 2026-05-17: Fusion-body follow-up for q4_1 MMQ and D256 FATTN

The next fusion-kernel pass intentionally stayed inside producer kernel bodies
rather than adding more post-op launch fusions.

The q4_1 MMQ body candidate hoisted the y-side `q4_1` unpack out of the
per-row-subtile loop in `vec_dot_q4_1_q8_1_dp4a`. The arithmetic and addressing
were unchanged, and the standalone checks reported `bad=0` for the two checked
hot shapes. It was still rejected because serial microbench timings were
effectively unchanged or slightly worse:

| Shape | Baseline mean | Hoisted mean | Decision |
| --- | ---: | ---: | --- |
| `k=1792, rows=448, cols=4096` | `0.185954 ms` | `0.186478 ms` | reject |
| `k=448, rows=1792, cols=4096` | `0.150323 ms` | `0.150366 ms` | reject |
| `k=1344, rows=448, cols=196, channels=25` | `0.203373 ms` | `0.203544 ms` | reject |

A smaller q4_1 MMQ body fusion was then kept. The hot `q4_1` dot path no
longer builds a temporary interleaved `u[8]` vector before calling the shared
dot helper; instead, the MMQ-specific helper consumes the two q8_1 half-blocks
directly and applies the same dequantization expression. This keeps the current
tile schedule and only reduces the work and register pressure inside the body.

Serial microbench results were small but consistently non-regressive against
the same baseline set:

| Shape | Baseline mean | Direct-u mean | Change |
| --- | ---: | ---: | ---: |
| `k=1792, rows=448, cols=4096` | `0.185954 ms` | `0.185592 ms` | `-0.19%` |
| `k=448, rows=1792, cols=4096` | `0.150323 ms` | `0.150218 ms` | `-0.07%` |
| `k=1344, rows=448, cols=196, channels=25` | `0.203373 ms` | `0.203075 ms` | `-0.15%` |

The standalone checked shapes reported `bad=0`. A seven-pair app run also kept
the Base+ q4_1 1024 10-frame comparison exact (`diff_rows=0`) and measured
`32.89 -> 32.61 ms/frame` by mean for the paired default vs shape-key path
(`0.83%` by mean). The paired confidence interval still crossed zero
(`-0.619..1.162 ms/frame`), so this should be treated as a valid cleanup that
removes unnecessary hot-body work rather than as a material end-to-end win by
itself.

The direct-u change was also checked with `-Xptxas=-v`. The q4_1 filtered
summary stayed effectively unchanged at the currently useful schedules:
`mmq_x=96` and `mmq_x=80` still report max `255` registers and no spills. The
microbench improvement therefore appears to come from removing local body work,
not from crossing an occupancy or spill threshold.

A follow-up q4_1 address-hoist probe precomputed the x/y row pointers and scale
values before the direct-u helper call. It kept standalone parity (`bad=0`) but
was not kept: two serial microbench passes were noisy and the app-level
seven-pair run was exact but slightly negative by mean
(`32.44 -> 32.57 ms/frame`, `-0.39%`, confidence interval
`-0.604..0.347 ms/frame`). The source was returned to the direct-u-only body.

The D256 FATTN candidate was kept. The `ncols2==1` path previously compiled the
full tiles and the final tail tile with `oob_check=true`. The new split compiles
full tiles with `oob_check=false` and keeps `oob_check=true` only for the final
tail tile. This preserves the tail safety path for `N=4100` while removing the
inner bounds predicate from the 4096-token full tiles.

Standalone D256 parity stayed within the existing tolerance with `bad=0`, and
the microbench improvement was material:

| Shape | Baseline mean | Split mean | Change |
| --- | ---: | ---: | ---: |
| `D=256, N=4096, heads=1, batch=1` | `0.442937 ms` | `0.357644 ms` | `-19.3%` |
| `D=256, N=4100, heads=1, batch=1` | `0.416454 ms` | `0.338049 ms` | `-18.8%` |

The Base+ q4_1 1024 10-frame tracking comparison stayed exact
(`diff_rows=0`). Seven paired bbox-only app runs measured a small app-level
mean improvement from `45.03 ms/frame` to `44.87 ms/frame`
(`0.35%` by mean), but the paired confidence interval crossed zero
(`-0.404..0.718 ms/frame`). This should be treated as a valid body-level cleanup
that reduces the D256 bucket, not as a standalone PyTorch-closing change.

The profile run after the split showed the D256 FATTN rows at about
`0.303..0.309 ms` compute time for `Q[256,4096,1,1]`, with K/V lengths `4096`
and `4100`.

A follow-up D56 body-configuration probe changed the
`DKQ=64,DV=64,ncols=64` `nbatch_fa` setting from `64` to `128`. It preserved the
standalone tolerance check (`bad=0`) but was slower on both representative D56
shapes:

| Shape | Baseline mean | `nbatch_fa=128` mean | Decision |
| --- | ---: | ---: | --- |
| `D=56, run_D=64, N=4096, heads=8, batch=1, sampled_q=9` | `0.677365 ms` | `0.717740 ms` | reject |
| `D=56, run_D=64, N=196, heads=8, batch=25` | `0.199964 ms` | `0.219126 ms` | reject |

This reinforces the current priority: shallow D56 config retuning is mostly
exhausted; remaining wins likely require a real D56-specific MMA body or a
larger MMQ body change.

The next fusion-kernel candidates should therefore be managed as body
specializations with independent microbench and app parity gates:

1. D56-specific FATTN body: avoid only changing `nbatch_fa`; specialize the
   D56 window/global body so the MMA shape, load loops, and accumulation match
   the padded 64-wide workload without carrying generic D64 overhead.
2. q4_1 MMQ body pressure reduction: keep the current useful `mmq_x=96/80`
   schedules and reduce temporary state, address arithmetic, and redundant
   unpacking inside those bodies. Do not use smaller X caps unless the full-mask
   parity gate passes.
3. Tail-specialized D256 FATTN: the full-tile OOB split already paid off; a
   separate tiny-tail path for `N=4100` remains possible but has higher parity
   risk because it changes the tail loop structure.

Two D56 toggles were checked before making any default change. Both preserve
the standalone FATTN tolerance check (`bad=0`) but do not improve the relevant
default path:

| Probe | Shape | Default mean | Probe mean | Decision |
| --- | --- | ---: | ---: | --- |
| `GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` | `N=4096, heads=8, batch=1, sampled_q=9` | `0.659608 ms` | `1.001665 ms` | reject |
| `GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` | `N=196, heads=8, batch=25` | `0.149536 ms` | `0.211597 ms` | reject |
| `GGML_CUDA_ENABLE_FATTN56_INLINE_PACK_LARGE=1` | `N=4096, heads=8, batch=1, sampled_q=9` | `0.622565 ms` | `0.852029 ms` | reject |

The window-shaped inline-pack-large run was equivalent to the default path in
the same rerun window (`0.135163 ms` vs `0.135275 ms`), so it is not evidence
for changing the default. The useful D56 work is therefore not toggling the
existing native-V or inline-large paths; it needs a more substantial D56 body
specialization, or the effort should stay on MMQ where the direct-u cleanup
already showed a small safe improvement.

The q4_1 MMQ ptxas pass was also refreshed with `-Xptxas=-v`. Filtering the
compiled `GGML_TYPE_Q4_1` kernels shows why naive X-cap changes are unstable:
the currently selected hot sizes `mmq_x=96` and `mmq_x=80` use up to `255`
registers but do not spill, while lower-register alternates such as `88` and
`72` change the tiling schedule. A generic `GGML_CUDA_MMQ_X_MAX=88` probe was
therefore checked without changing code. It was rejected: the standalone
microbench was mixed, the first app-level bbox-only pair was slower
(`45.0 -> 45.5 ms/frame`), and the 10-frame full-mask comparison was not exact
(`mask_hash_equal_rows=0/10`, `max_score_abs_delta=0.006658`,
`max_bbox_delta_px=1`). The register evidence points to reducing the q4_1 body
register footprint at the current useful tile sizes, not simply selecting a
smaller X tile.

Evidence:

- `outputs/fusion-mmq-q4_1-unpack-hoist-20260517/current/`
- `outputs/fusion-mmq-q4_1-unpack-hoist-20260517/baseline/`
- `outputs/mmq-q4_1-direct-u-probe-20260517/current/`
- `outputs/mmq-q4_1-direct-u-probe-20260517/parity-stats/summary.json`
- `outputs/mmq-ptxas-q4_1-direct-u-20260517/q4_1-summary.json`
- `outputs/mmq-q4_1-rowptr-probe-20260517/current/`
- `outputs/mmq-q4_1-rowptr-probe-20260517/parity-stats/summary.json`
- `outputs/fusion-fattn256-fulltile-oob-20260517/current/`
- `outputs/fusion-fattn256-fulltile-oob-20260517/baseline/`
- `outputs/fusion-fattn256-fulltile-oob-20260517/parity-stats/summary.json`
- `outputs/fusion-fattn256-fulltile-oob-20260517/profile/current.log`
- `outputs/fattn64-ncols64-nbatch128-20260517/current/`
- `outputs/fattn64-ncols64-nbatch128-20260517/baseline/`
- `outputs/fattn56-native-v-probe-20260517/`
- `outputs/fattn56-inline-pack-large-probe-20260517/`
- `outputs/mmq-ptxas-q4_1-current-v2-20260517/q4_1-summary.json`
- `outputs/mmq-q4_1-xmax88-probe-20260517/app/parity.json`

### 2026-05-17: Current q4_1 profile and fused-y MMQ probe

The latest detailed Base+ q4_1 profile after the direct-u cleanup still points
at the same two CUDA buckets. This profile is sync-heavy because
`GGML_CUDA_PROFILE_MMQ`, `GGML_CUDA_PROFILE_FATTN56`, and per-node CUDA timing
insert event synchronization; use it for attribution rather than normal
runtime:

- `hiera_encode` steady-state frames are about `35.3..38.3 ms` after the first
  frame.
- `propagate_single` is about `12.3..13.5 ms`.
- MMQ attribution is `77.36 ms` total across the run after dropping the first
  row per shape: `63.56 ms` MMQ body plus `13.80 ms` activation quantization.
- D56 FATTN attribution is `26.19 ms` total: `25.76 ms` MMA/body, `0.39 ms`
  packing, and `0.04 ms` slicing.

The largest MMQ shapes are still the stage-2 Hiera projection/MLP shapes:

| Shape | Quant sum | MMQ sum | Total |
| --- | ---: | ---: | ---: |
| `q4_1[448,1792] * f32[448,4096] -> f32[1792,4096]` | `0.878 ms` | `11.979 ms` | `12.857 ms` |
| `q4_1[1792,448] * f32[1792,4096] -> f32[448,4096]` | `2.306 ms` | `9.514 ms` | `11.819 ms` |
| `q4_1[448,1344] * f32[448,196,25] -> f32[1344,196,25]` | `0.739 ms` | `9.933 ms` | `10.672 ms` |
| `q4_1[448,448] * f32[448,196,25] -> f32[448,196,25]` | `0.744 ms` | `3.734 ms` | `4.478 ms` |
| `q4_1[448,112] * f32[448,65536] -> f32[112,65536]` | `2.639 ms` | `1.718 ms` | `4.357 ms` |

A temporary fused-y MMQ probe fused the f32-to-q8_1 activation conversion into
the q4_1 MMQ kernel only when `nrows_x <= mmq_y`, where the pre-quantized
activation tile is not reused across multiple row tiles. The experiment avoided
the known bad case where fusing activation quantization would repeat the same
quant work for every row tile.

The probe is correct after fixing the second q8_1 tile offset from `32` to the
actual `block_q8_1_mmq` span of `128` values, but it is slower and should not be
enabled by default:

| Shape | Default mean | Fused-y mean | Parity |
| --- | ---: | ---: | --- |
| `k=448, rows=112, cols=65536` | `0.491772 ms` | `0.563815 ms` | speed-only full shape |
| `k=448, rows=112, cols=12288` | `0.060488 ms` | `0.074886 ms` | `bad=0`, same max/mean abs |

The result is useful as a direction check: removing the quantization launch and
global q8_1 temporary is not enough when the fused path adds per-MMQ-tile
reductions and shared-memory writes. The probe was removed after measurement so
the default hot loop does not carry a rejected runtime branch. The next fusion
work should therefore target either a real q4_1 body specialization at the
existing `mmq_x=96/80` schedules, or a D56 FATTN body specialization. Fusing
activation quantization into the generic MMQ body is rejected for now.

A D56 FATTN config probe then changed the `DKQ=64,DV=64,ncols=64` MMA config
from keeping Q in registers to storing Q through shared memory
(`Q_in_reg=false`). This targets the direct head56 path used by the top D56
window/global shapes, but it was also slower:

| Shape | Default mean | `Q_in_reg=false` mean | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `D=56, run_D=64, N=4096, heads=8, batch=1, sampled_q=9` | `0.708445 ms` | `0.726955 ms` | `bad=0` | reject |
| `D=56, run_D=64, N=196, heads=8, batch=25` | `0.212811 ms` | `0.213629 ms` | `bad=0` | reject |

The source was returned to `Q_in_reg=true`. This makes the current conclusion
stronger: the useful D56 path is already on the better generic `64x64,ncols=64`
config for this GPU, and the next D56 win needs a real 56-wide body or a
larger algorithmic change rather than a shallow config flip.

A small-window `ncols1=32` selection probe was also tried for the same direct
head56 path. The motivation was reducing wasted work for `N=196`, where the
default `ncols1=64` covers 256 query positions. It preserved parity, but the
median did not improve and the mean was dominated by one default-path outlier:

| Shape | Default mean / median | `ncols1=32` mean / median | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `D=56, run_D=64, N=196, heads=8, batch=25` | `0.218123 / 0.212953 ms` | `0.213299 / 0.213219 ms` | `bad=0` | reject |
| `D=56, run_D=64, N=4096, heads=8, batch=1, sampled_q=9` | `0.630821 / 0.628862 ms` | `0.652519 / 0.638327 ms` | `bad=0` | reject |

The source was returned to the default `ncols1=64` selection. The shallow D56
selection probes now rejected are `native_v56`, `inline_pack_large`,
`nbatch_fa=128`, `Q_in_reg=false`, and `ncols1=32`.

Finally, a q4_1 MMQ stream-k disable probe checked whether the hot shapes with
`fixup=0` would be faster on the simpler conventional tiling path. They were
not. Disabling stream-k is catastrophically slower on the representative Base+
MMQ shapes:

| Shape | Default mean | No stream-k mean | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `k=1792, rows=448, cols=4096` | `0.182201 ms` | `4.634494 ms` | `bad=0` | reject |
| `k=448, rows=1792, cols=4096` | `0.140001 ms` | `5.360872 ms` | speed-only full shape | reject |
| `k=448, rows=1344, cols=196, channels=25` | `0.150084 ms` | `10.894644 ms` | speed-only full shape | reject |

The probe branch was removed. The stream-k path is not optional for the current
q4_1 workload even when the selected stream-k launch has no fixup kernel.

The q4_1 MMQ static epilogue path was also checked with bias+GELU enabled. This
is not a new optimization, but it confirms that the existing bias/GELU fusion is
material and should not be replaced by a dynamic activation path:

| Shape | Static epilogue mean | Dynamic epilogue mean | Decision |
| --- | ---: | ---: | --- |
| `k=448, rows=1792, cols=4096, bias+GELU` | `0.169911 ms` | `0.194547 ms` | keep static epilogue |
| `k=448, rows=1344, cols=196, channels=25, bias+GELU` | `0.188727 ms` | `0.213498 ms` | keep static epilogue |

This removes another shallow fusion candidate from the list: the MMQ epilogue
is already specialized for the relevant GELU case. Further q4_1 work needs to
reduce the MMQ body itself or the q8_1 activation quantization cost, not just
toggle the existing epilogue dispatch.

The q8_1 activation quantization block size was then checked by changing
`CUDA_QUANTIZE_BLOCK_SIZE_MMQ` from `128` to `64`. This preserves the padding
constraint (`4 * block_size` still divides `MATRIX_ROW_PADDING=512`) and
increases grid parallelism, but it was slower on the representative quant-heavy
and body-heavy shapes:

| Shape | Block 128 mean | Block 64 mean | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `k=448, rows=112, cols=65536` | `0.474552 ms` | `0.482964 ms` | speed-only full shape | reject |
| `k=896, rows=224, cols=16384` | `0.277752 ms` | `0.289023 ms` | speed-only full shape | reject |
| `k=1792, rows=448, cols=4096` | `0.169616 ms` | `0.182389 ms` | `bad=0` | reject |

The source was returned to block size `128`. The quantization bottleneck is not
fixed by simply increasing quantize grid parallelism; a useful change would need
to alter the memory layout, avoid repeated activation quantization, or reduce
the per-block reduction/write work without duplicating it per MMQ tile.

A D256 FATTN full-last-tile probe was also checked as a fusion-body follow-up.
The `ncols2==1` path already uses `oob_check=false` for non-final full tiles and
`oob_check=true` for the final tail tile. The probe additionally split the final
iteration so `N=4096`, where the final iteration is also a full tile, used the
non-OOB body. This preserved parity, but it was slower in the same
`sam3_fattn_parity` harness:

| Shape | Baseline mean / median | Full-last fastpath mean / median | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `D=256, N=4096, heads=1, batch=1, sampled_q=4096` | `0.783113 / 0.795561 ms` | `0.842943 / 0.843580 ms` | `bad=0` | reject |
| `D=256, N=4100, heads=1, batch=1, sampled_q=4096` | `0.707775 / 0.795801 ms` | `0.824131 / 0.843460 ms` | `bad=0` | reject |

The source was returned to the baseline final-tail dispatch. For D256, another
shallow split inside the existing FATTN loop is unlikely to help; the remaining
fusion work should focus on a structurally different tail body or on the larger
D56/q4_1 MMQ buckets.

A q4_1 MMQ MMA epilogue-body cleanup was then checked. The Turing/Blackwell
`q8_1 x q8_1` MMA path used by q4_1 accumulation currently applies the dot
scale term and the q4_1 offset term as two `sum +=` statements. A probe rewrote
that inner update as one equivalent expression,
`sum += dm.x*ds.x*C + dm.y*ds.y`, to see whether it reduced redundant sum
traffic or register pressure. It preserved the standalone tolerance check, but
was slower on all representative q4_1 shapes:

| Shape | Baseline mean / median | Combined-update mean / median | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `k=448, rows=1792, cols=4096, bias+GELU` | `0.159293 / 0.155646 ms` | `0.182584 / 0.178909 ms` | speed-only | reject |
| `k=1792, rows=448, cols=4096` | `0.172393 / 0.172747 ms` | `0.186723 / 0.185292 ms` | `bad=0` | reject |
| `k=448, rows=1344, cols=196, channels=25, bias+GELU` | `0.175891 / 0.179431 ms` | `0.181217 / 0.176074 ms` | speed-only | reject |

The source was returned to the two-update form. This indicates that the compiler
and scheduler prefer the current split update for this MMA body, so q4_1 work
should not spend more time on algebraic cleanup inside the existing accumulation
expression. A meaningful MMQ win likely needs a different tile body, reduced
activation quantization cost, or a safe producer-side q8_1 reuse design rather
than local expression folding.

A q4_1 activation-quantization specialization was also tested for the common
DS4/no-ids/contiguous source path. The probe kept the same rounding and layout
but replaced the generic stride-addressed `quantize_mmq_q8_1` launch with a
DS4-only kernel that computes the contiguous row base directly and writes the
same `block_q8_1_mmq` fields. Standalone parity stayed exact for the checked
rows, but same-binary A/B using `GGML_CUDA_DISABLE_MMQ_DS4_CONTIGUOUS_QUANT=1`
showed no speed win:

| Shape | Specialized mean / median | Generic mean / median | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `k=448, rows=1792, cols=4096, bias+GELU` | `0.174982 / 0.171672 ms` | `0.173644 / 0.175448 ms` | speed-only | reject |
| `k=1792, rows=448, cols=4096` | `0.178433 / 0.178142 ms` | `0.172818 / 0.170648 ms` | `bad=0` | reject |

The source was returned to the generic DS4 quantizer. The earlier apparent win
did not survive same-binary A/B, so the quantization path is still not helped by
removing the simple contiguous stride arithmetic. The remaining quantization
opportunity is more structural: avoid producing the same q8_1 activation blocks
multiple times with a safe lifetime model, or change the consumer/producer
layout together.

A final shallow D56 selection probe forced the head56 MMA wrapper to use
`ncols1=16` instead of the default `64`. This is different from the previously
rejected `ncols1=32` probe and checks whether smaller query tiles reduce wasted
work for the window case. It preserved the standalone tolerance check, but the
effect was too small and did not help the global shape:

| Shape | Default mean / median | `ncols1=16` mean / median | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `D=56, run_D=64, N=196, heads=8, batch=25` | `0.215509 / 0.213794 ms` | `0.214223 / 0.214170 ms` | `bad=0` | reject |
| `D=56, run_D=64, N=4096, heads=8, batch=1, sampled_q=9` | `0.675285 / 0.676939 ms` | `0.675940 / 0.679034 ms` | `bad=0` | reject |

The source was returned to the default selection. The rejected D56 selection and
configuration probes now cover `native_v56`, `inline_pack_large`, `nbatch_fa=128`,
`Q_in_reg=false`, `ncols1=32`, and `ncols1=16`; the remaining D56 opportunity is
not another small selector knob.

### 2026-05-17: Accepted ADD bias plus permute-cont fusion

The next fusion-kernel pass targeted a producer-to-materialization pattern
rather than another standalone elementwise ADD. The Hiera/SAM graph has repeated
channel-bias adds in `[W,H,C,B]` layout followed by `PERMUTE(2,0,1,3)` and
`CONT` to create the internal `[C,W,H,B]` tensor. The new CUDA path fuses:

`ADD([W,H,C,B], [1,1,C,1]) -> PERMUTE(2,0,1,3) -> CONT`

into one kernel that writes the final contiguous `[C,W,H,B]` output directly.
The intermediate ADD tensor is not materialized. The path is intentionally
shape-limited to F32 contiguous source, F32 contiguous channel bias, the exact
axis order above, and a contiguous F32 output. It can be disabled with
`GGML_CUDA_DISABLE_ADD_BIAS_PERMUTE_CONT_FUSION=1`.

The profile smoke confirmed that the new matcher fires on the intended
patch/stage materialization rows; examples include fused `ADD` nodes with
`skipped=5` producing `dst_ne=112,256,256,1`, `224,128,128,1`,
`448,64,64,1`, and `896,32,32,1` in
`outputs/add-bias-permute-cont-fusion-20260517/profile/profile.log`.

Full 10-frame tracking parity against the disabled path is exact:

| Model | Evidence | Result |
| --- | --- | --- |
| Base+ q4_1 1024 | `outputs/add-bias-permute-cont-fusion-20260517/parity/compare.json` | `10 / 10` mask hashes equal, bbox and score delta zero |
| Base+ q8_0 1024 | `outputs/add-bias-permute-cont-fusion-20260517/parity-q8/compare.json` | `10 / 10` mask hashes equal, bbox and score delta zero |

Seven paired q4_1 bbox-only runs measured a small positive mean but not a
statistically decisive app-level win: disabled mean `45.10 ms/frame`, enabled
mean `44.86 ms/frame`, paired mean saved `0.24 ms/frame`, with 95% CI
`-0.33..0.82 ms/frame`
(`outputs/add-bias-permute-cont-fusion-20260517/pairs/summary.json`). This is
therefore accepted as an exact launch/materialization cleanup, not as a
PyTorch-closing speedup. The larger remaining work stays on D56 FATTN body and
MMQ body/activation quantization.

A fusion-kernel policy probe then rechecked whether q4_1/q8_0 `bias+GELU`
MMQ epilogues should prefer the current static GELU specialization or a dynamic
activation epilogue. A first sequential ON/OFF run was noisy enough to suggest
the dynamic path might be faster, so the decision was repeated as alternating
same-binary runs on representative shapes. The paired median-of-medians favored
the existing static GELU path for every checked shape:

| Shape | Dynamic GELU median-of-medians | Static GELU median-of-medians | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `q4_1 k=1792, rows=448, cols=4096, bias+GELU` | `0.191069 ms` | `0.175687 ms` | `bad=0` | keep static |
| `q4_1 k=448, rows=1344, cols=196, channels=25, bias+GELU` | `0.213102 ms` | `0.170404 ms` | speed-only | keep static |
| `q8_0 k=1792, rows=448, cols=4096, bias+GELU` | `0.172300 ms` | `0.165470 ms` | `bad=0` | keep static |
| `q8_0 k=448, rows=1344, cols=196, channels=25, bias+GELU` | `0.163231 ms` | `0.144195 ms` | speed-only | keep static |

The temporary source change was reverted. This means the fusion-kernel direction
should not remove the current static epilogue. The next useful fusion work is
more structural: avoid repeated activation quantization across adjacent consumers
or fuse a larger producer/consumer region, rather than changing only the MMQ
GELU epilogue dispatch.

The MMQ profiler was then extended to report a conservative `src1_name_reuse`
section in addition to raw `src1_ptr` reuse. This separates two very different
signals: raw pointer reuse is unsafe because ggml's allocator can recycle device
addresses for unrelated tensors, while tensor-name reuse is only an upper bound
unless it is scoped to one graph compute/liveness interval. A refreshed
2-frame Base+ q4_1 no-isolation profile reported `2.956864 ms` of effective
MMQ activation quantization. Raw pointer grouping still looked large
(`2.533056 ms` in repeated pointer groups), but conservative tensor-name
grouping was mostly `count=2`, matching repeated graph execution across frames
rather than same-frame reusable tensors. The after-first upper bound was
`1.118112 ms`, but this cannot be cached across frames because the tensor
contents change.

This narrows the fusion direction further. A graph-level q8_1 activation cache
is not the next high-confidence optimization unless the backend can provide a
real graph-compute/liveness key. A first-MLP `MMQ + bias + GELU -> next MMQ`
fusion also cannot be implemented as a simple elementwise epilogue: q8_1
activation generation needs a 32-value block reduction for each output column.
The viable version is therefore a new producer-side kernel that writes the f32
GELU output and the q8_1 block layout together, or a larger tiled two-stage MLP
kernel. That is a real kernel-design task, not a cache toggle.

To bound that producer-side idea, `sam3_f32_batched_bench --gelu-quant-ds4`
was added as a standalone microbenchmark. It compares two kernels
(`GELU -> q8_1 DS4 quantize`) against one fused kernel that writes the same f32
GELU output and q8_1 MMQ DS4 layout. The fused probe is byte-identical to the
separate path on the checked outputs:

| Shape | Separate median | Fused median | Median speedup | Parity |
| --- | ---: | ---: | ---: | --- |
| `rows=1792, cols=4096` | `0.155904 ms` | `0.146880 ms` | `5.788%` | `max_abs_output_delta=0`, `q8_diff_bytes=0` |
| `rows=1344, cols=4900` | `0.119840 ms` | `0.107424 ms` | `10.360%` | `max_abs_output_delta=0`, `q8_diff_bytes=0` |
| `rows=448, cols=4096` | `0.019808 ms` | `0.014112 ms` | `28.756%` | `max_abs_output_delta=0`, `q8_diff_bytes=0` |

This is positive, but it is not yet a PyTorch-closing production win. In the
real q4_1 Hiera graph, the producer MMQ already computes `bias+GELU` in its
writeback epilogue, so only the later activation-quantization read/launch is
available to remove. The microbenchmark suggests that a safe producer-side
q8_1 write is plausible and exact, but the expected graph-level impact is
bounded by the current MMQ quantization slices unless the implementation fuses
more of the two-stage MLP body.

The refreshed MMQ profile was also scanned for direct producer-to-consumer
cases where an MMQ output tensor is later consumed as `src1` by another MMQ with
the same shape. This is the simplest possible q8_1 side-output fusion target.
It found only one effective candidate in both the 2-frame and 3-frame profiles:
`mem_enc.fuser.1.pwconv2.weight -> sam_dec.twoway.1.cross_attn_token_to_image.k_proj.weight`.
The removable consumer-side quantization was only `0.01024 ms` in the 2-frame
profile and `0.010208 ms` in the 3-frame profile. That makes simple adjacent
MMQ producer/consumer fusion a low-priority path; it is too small to close the
current Hiera encode gap.

A small q4_1 MMQ kernel-body cleanup was then kept. The DP4A q4_1 path now
computes the inner x/y/dm/ds offsets once before calling
`vec_dot_q4_1_q8_1_mmq_direct` instead of spelling the address arithmetic in
each argument. This does not change math or memory layout. Same-machine
microbenchmarks against the immediately preceding source showed no parity
regression and a small-to-moderate speed win depending on shape:

| Shape | Baseline median | Pointer-hoist median | Parity | Decision |
| --- | ---: | ---: | --- | --- |
| `k=1792, rows=448, cols=4096` | `0.186504 ms` | `0.186001 ms` | `bad=0` | keep |
| `k=448, rows=1792, cols=4096` | `0.154338 ms` | `0.142055 ms` | speed-only full shape | keep |
| `k=1344, rows=448, cols=196, channels=25` | `0.198821 ms` | `0.200891 ms` | `bad=0` | keep; roughly neutral |

This is not a structural fusion win, but it is a low-risk CUDA kernel-body
cleanup that slightly improves the largest body-heavy q4_1 shape. The remaining
fusion work should therefore focus on eliminating activation quantization
launch/read cost or on a larger two-stage MLP fusion, not on direct MMQ output
reuse.

Two follow-up FATTN56 cleanups were checked and rejected. First, moving the
padded f32-head zero fill into the last f32-to-f16 tile load preserved the
global sampled-query row but changed the window row beyond the standalone
tolerance, so the probe was reverted. Second, skipping `get_alibi_slope` when
the `no_mask_no_sinks` template path is active was mathematically safe, but did
not improve the measured D56 shapes:

| Probe | Shape | Baseline median | Probe median | Parity / note | Decision |
| --- | --- | ---: | ---: | --- | --- |
| inline tail pad | `D=56, N=4096, heads=8, batch=1, sampled_q=9` | not retained | `0.717457 ms` | `bad=0` | reject |
| inline tail pad | `D=56, N=196, heads=8, batch=25` | not retained | `0.201040 ms` | exceeded tolerance | reject |
| no-mask slope skip | `D=56, N=4096, heads=8, batch=1, sampled_q=9` | `0.716625 ms` | `0.717483 ms` | `bad=0` | reject |
| no-mask slope skip | `D=56, N=196, heads=8, batch=25` | `0.193727 ms` | `0.200841 ms` | baseline and probe both exceed standalone tolerance | reject |

The standalone D56 window tolerance failure is present in both baseline and
probe with this full-window harness, so it is not evidence that the slope skip
changed output. The speed result is still negative, and the source was returned
to the baseline slope computation.

A separate copy/quantize fusion candidate was also scoped and deferred. The
generic split `ggml_cuda_op_mul_mat` path copies non-contiguous f32 `src1` into
a temporary and then launches `quantize_mmq_q8_1_cuda`; fusing those two steps
could remove one launch plus a full f32 temp write/read. However, the current
Base+ q4_1 Hiera hot path takes the direct `!split && use_mul_mat_q`
`ggml_cuda_mul_mat_q` route, so this candidate does not hit the dominant MMQ
profile rows above. It remains useful for split/generic MMQ workloads, but it
is not the next PyTorch-closing change for this repository's current Base+
comparison.

After the accepted q4_1 pointer-hoist cleanup, a fresh short Base+ q4_1 run
reported `45.7 ms/frame` (`p50=43.8 ms`, `p95=53.3 ms`) for the same
10-frame bbox-only/multimask video. The synchronized 3-frame attribution still
shows two comparable remaining buckets:

| Bucket | Effective rows | Quant / pack sum | Body sum | Total sum | Top shape |
| --- | ---: | ---: | ---: | ---: | --- |
| q4_1 MMQ | `111` | `4.602272 ms` | `9.172640 ms` | `13.774912 ms` | `q4_1[1792,448] x f32[1792,4096]` total `6.552160 ms` |
| D56 FATTN | `64` | `0.231168 ms` pack+slice | `13.955200 ms` | `14.186368 ms` | `Q/K/V[56,196,8,25]` total `4.699968 ms` |

The target is therefore still not complete: the current C++ row remains slower
than the official PyTorch Base+ row, and shallow FATTN selector/cleanup probes
are exhausted. Further work needs a real D56 MMA-body specialization or a
larger q4_1 MMQ/activation-quantization redesign.

The first production fusion-kernel step implemented on this pass is guarded
strided `src1` quantization for the generic MMQ path. When `src1` is a
non-contiguous f32 tensor on the active device, the path can now skip the
temporary `ggml_cuda_cpy_tensor_2d` gather and directly invoke the MMQ q8_1
quantizer against the original strides, provided the row/channel/sample strides
are 16-byte aligned. `GGML_CUDA_DISABLE_MMQ_STRIDED_SRC1_QUANT=1` restores the
old copy-then-quantize behavior for A/B testing. The Base+ q4_1 1024 graph did
not move in a short 3-run A/B (`45.43 ms/frame` default vs `45.33 ms/frame`
disabled), which confirms that this fusion is not on the current dominant
direct-MMQ hot path. It is kept as a scoped generic-path improvement, not as the
main PyTorch-closing change.

While validating q8_0 quality, a CUDA-only regression was isolated to the MMQ
bias/GELU static epilogue specialization. Base+ q8_0 at 512 input had fallen to
`mean_mask_iou=0.668702`, `min_mask_iou=0.542550` even when forcing candidate
0. The same model on CPU stayed healthy (`mean_mask_iou=0.929552`,
`min_mask_iou=0.851597`), and f16 CUDA stayed healthy (`mean_mask_iou=0.935076`,
`min_mask_iou=0.865417`). Disabling MMQ bias/GELU fusion, all MMQ bias fusion,
or only the static epilogue each restored the old q8_0 CUDA result
(`mean_mask_iou=0.931306`, `min_mask_iou=0.847843`). The fix keeps MMQ
bias/GELU fusion available but forces q8_0 through the dynamic epilogue path
instead of the static-specialized epilogue. After the fix, q8_0 CUDA 512 again
matches the old quality row: `mean_mask_iou=0.931306`, `min_mask_iou=0.847843`,
with propagation candidate selection `9/9`.

Short 1024 speed checks after the q8_0 epilogue fix:

| Model | Runs | Mean Track/fr | Median Track/fr | Median P50 | Median P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base+ q8_0 CUDA | `3` | `46.63 ms` | `46.6 ms` | `44.9 ms` | `53.4 ms` |
| Base+ q4_1 CUDA | `3` | `45.53 ms` | `45.6 ms` | `43.8 ms` | `51.7 ms` |

The q4_1 profile was refreshed again after the q8_0 quality fix. The top
synchronized node buckets are still q4_1 MMQ MLP projections, D56 window/global
FlashAttention, and the fused residual-add + LayerNorm + affine kernels. The
MMQ/FATTN totals in a 4-frame profiled run were:

| Bucket | Effective rows | Quant / pack sum | Body sum | Total sum | Top shape |
| --- | ---: | ---: | ---: | ---: | --- |
| q4_1 MMQ | `154` | `6.579552 ms` | `13.315040 ms` | `19.894592 ms` | `q4_1[1792,448] x f32[1792,4096]` total `9.443520 ms` |
| D56 FATTN | `88` | `0.316928 ms` pack | `20.355616 ms` | `20.707936 ms` | `Q/K/V[56,196,8,25]` total `6.694528 ms` |

The residual-add + LayerNorm + affine fused kernel was then probed with a
256-thread block for `ncols >= 256`, keeping the existing one-warp path as the
default. This was rejected because it regressed the full graph: five Base+ q4_1
1024 runs changed from default mean `45.22 ms/frame` to `45.74 ms/frame` with
the 256-thread probe. The source was returned to the current one-warp LayerNorm
path.

A small q4_1 MMQ address-arithmetic cleanup was also checked by hoisting
`kyqs` and y-side offsets outside the `i0` loop in the DP4A body. The standalone
microbench preserved parity (`bad=0` on checked shapes), but the mixed shape
results were not convincingly better and the full Base+ q4_1 graph regressed to
five-run mean `45.72 ms/frame`. The source was returned to the previous q4_1
DP4A loop layout.

The q4_1 MMQ `mmq_x` cap was swept again after the pointer-hoist cleanup. The
current 4096-column hot shape still prefers the existing `mmq_x=96` selection
(`cap=96/128` were effectively tied at about `0.167 ms` median), while the
largest 65536/16384-column shapes showed small standalone wins around
`cap=112`. A full-graph test did not convert that into a reliable improvement:
five special-cap runs averaged `45.3 ms/frame`, which is effectively tied with
the current default runs (`45.37 ms/frame` in the adjacent A/B). The special cap
was therefore reverted. This is another scheduling knob that is too small and
too noisy to close the gap.

Evidence:

- `outputs/current-profile-baseplus-q4_1-direct-u-20260517/benchmark-summary.json`
- `outputs/current-profile-baseplus-q4_1-direct-u-20260517/mmq-summary.json`
- `outputs/current-profile-baseplus-q4_1-direct-u-20260517/fattn56-summary.json`
- `outputs/current-profile-baseplus-q4_1-direct-u-20260517/hotspots-hiera-encode.json`
- `outputs/mmq-q4_1-fused-y-probe-20260517/default/`
- `outputs/mmq-q4_1-fused-y-probe-20260517/fused/`
- `outputs/fattn56-qinreg-probe-20260517/baseline/`
- `outputs/fattn56-qinreg-probe-20260517/q-shared/`
- `outputs/fattn56-small-ncols32-probe-20260517/default/`
- `outputs/fattn56-small-ncols32-probe-20260517/ncols32/`
- `outputs/mmq-streamk-disable-probe-20260517/default/`
- `outputs/mmq-streamk-disable-probe-20260517/no-stream-k/`
- `outputs/mmq-static-epilogue-probe-20260517/default/`
- `outputs/mmq-static-epilogue-probe-20260517/dynamic/`
- `outputs/mmq-dynamic-gelu-default-20260517/paired/`
- `outputs/mmq-src1-current-profile-2frame-20260517/mmq-summary.json`
- `outputs/mmq-src1-current-profile-20260517/mmq-summary.json`
- `outputs/gelu-quant-ds4-fusion-probe-20260517/`
- `outputs/mmq-q41-ptr-hoist-20260517/`
- `outputs/fattn56-inline-tail-pad-probe-20260517/`
- `outputs/fattn56-slope-cleanup-probe-20260517/`
- `outputs/current-after-fusion-kernel-cleanup-20260517/`
- `outputs/current-after-fusion-kernel-cleanup-profile-20260517/`
- `outputs/mmq-q41-xcap-resweep-after-ptr-hoist-20260517/`
- `outputs/mmq-q41-xcap112-graph-ab-20260517/`
- `outputs/mmq-q41-xcap112-special-20260517/`
- `outputs/mmq-q41-xcap112-special-graph-20260517/`
- `outputs/mmq-quant-block64-probe-20260517/default/`
- `outputs/mmq-quant-block64-probe-20260517/block64/`
- `outputs/fusion-strided-src1-graph-ab-20260517/`
- `outputs/sam2-official-quality-current-q8_0-512-force-cand0-20260517/`
- `outputs/sam2-official-quality-current-f16-512-force-cand0-20260517/`
- `outputs/sam2-official-quality-current-q8_0-512-cpu-force-cand0-20260517/`
- `outputs/sam2-official-quality-current-q8_0-512-mmq-epilogue-isolation-20260517/`
- `outputs/sam2-official-quality-current-q8_0-512-static-epilogue-fixed-20260517/`
- `outputs/q8-static-epilogue-fix-speed-20260517/`
- `outputs/profile-after-q8-epilogue-fix-baseplus-q4_1-20260517/`
- `outputs/norm256-probe-baseplus-q4_1-20260517/`
- `outputs/mmq-q41-y-offset-hoist-probe-20260517/`
- `outputs/mmq-q41-y-offset-hoist-graph-20260517/`

### 2026-05-17: Fusion-kernel direction after rejected D56 direct epilogue

The next D56 FlashAttention fusion probe tried to bypass the generic shared
memory writeback/normalization epilogue for the narrow non-fixup D56 path
(`D_SRC=56`, `DV=64`, `DV_DST=56`, no mask/sinks). The idea was to normalize
`VKQ_C` fragments from registers and write `dst` directly, leaving stream-k
fixup on the existing path.

The probe was rejected and reverted. It preserved the sampled global row within
tolerance, but was slower in the standalone parity harness
(`mean_ms=0.917470`, `median_ms=1.028723` for `N=4096`, sampled queries).
The full-window harness failed parity (`max_abs=0.0320834816`,
`bad=797273/2195200` for `N=196`, `batch=25`). This confirms that the current
fragment-to-output mapping is not a simple direct store from `T_C_VKQ` plus
`l % cols_per_thread`, and that the shared-memory epilogue is also carrying
layout normalization work.

The remaining fusion-kernel direction should therefore prioritize the q4_1
producer-side GELU/q8_1 DS4 side-output path. That path is harder than a graph
toggle, but it matches the current bottleneck better: keep the existing
`MMQ + bias + GELU` F32 output, additionally produce the consumer's q8_1 MMQ
layout, and let the immediately following q4_1 MMQ skip its standalone
activation quantization. This must be limited initially to exact/no-fixup
schedules and validated byte-for-byte against `quantize_mmq_q8_1_cuda`.

`scripts/summarize_mmq_gelu_side_output_candidates.py` now measures that exact
pattern from the combined node/MMQ profile logs instead of relying on raw
pointer reuse. On the current 4-frame Base+ q4_1 profile, after dropping the
first row per consumer shape, it finds `61` q4_1 Hiera MLP
`fc1 -> GELU -> fc2` candidates. The consumer-side activation quantization
upper bound is `2.394528 ms` in the synchronized profile, or roughly
`0.6 ms/frame` for this run:

| Consumer shape | Rows | Consumer quantization sum | Typical row |
| --- | ---: | ---: | --- |
| `q4_1[1792,448] x f32[1792,4096] -> f32[448,4096]` | `43` | `1.264736 ms` | `0.0296 ms` |
| `q4_1[896,224] x f32[896,16384] -> f32[224,16384]` | `7` | `0.962080 ms` | `0.1373 ms` |
| `q4_1[3584,896] x f32[3584,1024] -> f32[896,1024]` | `11` | `0.167712 ms` | `0.0154 ms` |

This keeps the side-output fusion on the priority list, but it also bounds the
expected win. Even a perfect implementation is a sub-millisecond-per-frame
optimization on the current Base+ profile, so it should not displace the larger
D56 FlashAttention body and q4_1 MMQ body/scheduling work.

As a first implementation step, the normal non-`ids` MMQ path now routes the
post-quantization launch through `ggml_cuda_mul_mat_q_run_prequantized`. This is
a behavior-preserving refactor: the existing path still quantizes `src1` with
`quantize_mmq_q8_1_cuda` or `quantize_mmq_fp4_cuda`, then calls the same MMQ
kernel with the same strides. The purpose is to create a narrow internal entry
point for a later producer-side side-output buffer. A short Base+ q4_1 1024
smoke run completed after the refactor (`outputs/mmq-prequant-helper-smoke-20260517/normal-run.log`).

Evidence:

- `scripts/summarize_mmq_gelu_side_output_candidates.py`
- `outputs/profile-after-q8-epilogue-fix-baseplus-q4_1-20260517/mmq-gelu-side-output-candidates.json`
- `outputs/mmq-prequant-helper-smoke-20260517/`

### 2026-05-17: Env-gated MMQ prequant cache plumbing

`GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` now enables an experimental graph-local
prequant cache for q4_1 Hiera MLP producer/consumer pairs. When a fused
`MMQ + bias + GELU` producer is immediately consumed by a q4_1 MMQ, the producer
output is quantized into the consumer's q8_1 MMQ layout and the consumer reuses
that prequantized buffer.

This is intentionally not enabled by default. It validates the graph plumbing
needed by the real producer-side epilogue fusion, but it is not itself the final
optimization because it still launches a standalone quantization kernel after
the producer. The current Base+ q4_1 probe hit the cached consumer path
(`cached_src1=1`) 68 times on the 3-frame smoke run. A 5-run 10-frame A/B showed
no speed win:

| Mode | Track/fr runs |
| --- | --- |
| default | `45, 46, 45, 45, 45 ms` |
| `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` | `46, 46, 46, 46, 49 ms` |

The next step is to replace the producer-side standalone quantization in this
plumbing with an actual MMQ epilogue side-output, limited to exact/no-fixup
schedules. The cache storage must remain allocation-order aware: ggml's VMM
pool is LIFO, so graph-local cache entries are stored in insertion order and
released in reverse order at the start of the next graph evaluation.

Evidence:

- `outputs/mmq-prequant-cache-probe-20260517c/cache-profile.log`
- `outputs/mmq-prequant-cache-graph-ab-20260517/`

The profile A/B helper `scripts/summarize_mmq_prequant_cache.py` was added to
separate cached and uncached q4_1 Hiera MLP `fc2` rows. On the 3-frame profile
pair, the cache path confirmed the consumer quantization skip:

| Shape | Cached rows | Uncached rows | Cached quant sum | Uncached quant sum |
| --- | ---: | ---: | ---: | ---: |
| `q4_1[1792,448] x f32[1792,4096] -> f32[448,4096]` | `48` | `48` | `0.143488 ms` | `1.458880 ms` |
| `q4_1[896,224] x f32[896,16384] -> f32[224,16384]` | `9` | `9` | `0.020608 ms` | `1.236800 ms` |
| `q4_1[3584,896] x f32[3584,1024] -> f32[896,1024]` | `9` | `9` | `0.028544 ms` | `0.147840 ms` |

For those matched shapes the consumer-side quantization upper bound is therefore
about `2.65 ms` over this synchronized 3-frame profile pair. The graph A/B still
regressed because the experimental path pays a new producer-side standalone
quantization launch to populate the cache. This strengthens, rather than
weakens, the implementation requirement: the next useful fusion must generate
the q8_1 DS4 side-output inside the producer MMQ epilogue, not in a separate
kernel after the producer.

Evidence:

- `scripts/summarize_mmq_prequant_cache.py`
- `outputs/mmq-prequant-cache-profile-ab-20260517/summary.json`

### 2026-05-17: Current hotspot priority after prequant plumbing

The current q4_1 Base+ 1024 profile was refreshed after the prequant plumbing
change. With `SAM3_PROFILE=1`, `GGML_CUDA_PROFILE_NODES=1`, and
`GGML_CUDA_PROFILE_MMQ=1`, the top synchronized Hiera encode rows remain:

| Rank | Hotspot | Count | Drop-max sum | Mean after drop-max | Priority |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `MUL_MAT dst=f32[1792,4096,1,1]` | `64` | `10.916992 ms` | `0.173286 ms` | q4_1 MLP expansion |
| 2 | `MUL_MAT dst=f32[448,4096,1,1]` | `64` | `10.063168 ms` | `0.159733 ms` | q4_1 MLP projection |
| 3 | `MUL_MAT dst=f32[1344,196,25,1]` | `48` | `9.003168 ms` | `0.191557 ms` | window qkv projection |
| 4 | `FLASH_ATTN_EXT dst=f32[56,8,196,25]` | `48` | `6.648352 ms` | `0.141454 ms` | D56 window attention |
| 5 | `FLASH_ATTN_EXT dst=f32[56,8,4096,1]` | `12` | `6.297312 ms` | `0.572483 ms` | D56 global attention |

This keeps the priority order clear:

1. q4_1 MMQ body/scheduling for MLP and window qkv.
2. D56 FlashAttention body, especially global and window paths.
3. q8_1 side-output fusion as a secondary sub-millisecond/frame optimization.

Evidence:

- `outputs/current-hotspots-after-prequant-plumbing-20260517/profile-summary.json`
- `outputs/current-hotspots-after-prequant-plumbing-20260517/node-hotspots.json`
- `outputs/current-hotspots-after-prequant-plumbing-20260517/mmq-profile.json`

### 2026-05-17: Rejected raw `src1->data` q8 cache

A lower-risk looking alternative was tested before writing the producer epilogue
side-output: cache a q8_1 MMQ layout for repeated `src1` activation pointers.
The narrow version targeted only the expensive Base+ q4_1
`448 x 65536 -> 112 x 65536` pair and did reduce the second consumer's
quantization from about `0.305 ms` to about `0.002 ms` in the synchronized MMQ
profile. A 5-run graph A/B looked mildly favorable at first (`45.0 ms/frame`
with the env-gated cache versus `45.6 ms/frame` without it).

The probe was rejected and reverted because parity failed. Comparing 10-frame
JSONL outputs between the cached and uncached runs produced
`mask_hash_equal_rows=0/10`, `min_bbox_iou=0.0271519`, and
`max_bbox_delta_px=570`. The reason is that identical `src1->data` addresses are
not a valid semantic cache key inside a ggml graph: the allocator can reuse the
same device address for logically different tensors. Only explicit tensor
producer/consumer dependencies, such as the prequant cache plumbing above, are
safe enough for this class of reuse.

Evidence:

- `outputs/mmq-src1-q8-cache-probe-narrow-20260517/cache-profile.log`
- `outputs/mmq-src1-q8-cache-graph-ab-20260517/`
- `outputs/mmq-src1-q8-cache-parity-20260517/compare.json`
- `outputs/mmq-src1-q8-cache-rejected-20260517/profile-after-revert.log`

### 2026-05-17: Rejected q4_1 qkv `mmq_x=112` shape probe

The q4_1 window qkv shape
`q4_1[448,1344] x f32[448,196,25] -> f32[1344,196,25]` currently launches with
`mmq_x=80`. A shape-specific `mmq_x=112` probe improved the standalone
`sam3_mmq_bench` row from about `0.173 ms` to `0.153 ms`, but the improvement
did not survive the full graph:

| Mode | Track/fr runs |
| --- | --- |
| default | `45, 46, 46, 45, 45 ms` |
| qkv `mmq_x=112` probe | `46, 45, 45, 45, 46 ms` |

The graph average was `45.4 ms/frame` for both modes, so the probe was reverted.
This reinforces the current rule for MMQ tuning: standalone shape wins are not
accepted unless the 10-frame graph A/B also moves.

Evidence:

- `outputs/mmq-q41-qkv-x112-probe-20260517/qkv.log`
- `outputs/mmq-q41-qkv-x112-graph-ab-20260517/`

### 2026-05-17: Rejected producer `bias+GELU+q8_1` side-output kernel

The next fusion-kernel probe replaced the prequant-cache producer's separate
post-GELU quantization with a single CUDA kernel that reads the producer's raw
MMQ output, applies `bias+GELU`, writes the normal f32 activation back, and
also writes the q4_1 consumer's DS4 q8_1 MMQ layout. This is deliberately one
step short of a true MMQ epilogue side-output, but it tests whether moving GELU
out of the MMQ writeback and combining it with q8_1 generation is enough to
make the prequant cache profitable.

Parity was exact on the 10-frame Base+ q4_1 1024 bbox/multimask run:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, and
`max_bbox_delta_px=0` against both the default path and the older prequant-cache
path. Speed was negative, so the probe was removed:

| Mode | Track/fr mean | Median | Paired delta vs default |
| --- | ---: | ---: | ---: |
| default | `45.22 ms` | `45.2 ms` | baseline |
| producer `bias+GELU+q8_1` kernel | `46.26 ms` | `46.3 ms` | `-1.04 ms/frame` |

The result is useful because it narrows the real fusion requirement. A separate
post-MMQ kernel still pays a full f32 read/write pass over the producer output,
so it cannot replace the consumer quantization cheaply enough. The remaining
side-output path must put q8_1 generation inside the producer MMQ writeback, or
fuse a larger two-stage MLP tile, rather than adding another activation kernel
between producer and consumer.

Evidence:

- `outputs/mmq-prequant-gelu-kernel-20260517/default-vs-gelu-kernel.json`
- `outputs/mmq-prequant-gelu-kernel-20260517/prequant-vs-gelu-kernel.json`
- `outputs/mmq-prequant-gelu-kernel-speed-20260517/summary.json`

### 2026-05-17: Current Base+ q4_1/q8_0 refresh and rejected MMQ ids-writeback skip

The current build was rechecked against the existing official SAM2.1 Python
bf16 result set using the same decoded `960x540` source frames, 10-frame range,
point prompt, and 1024 SAM input size. The closest current row is Base+ q4_1:

| Model | C++ CUDA Track/fr | Official PyTorch bf16 Track/fr | PyTorch / C++ |
| --- | ---: | ---: | ---: |
| Base+ q8_0 | `51.1 ms` | `42.6667 ms` | `0.835` |
| Base+ q4_1 | `44.9 ms` | `42.6667 ms` | `0.950` |

This means the active PyTorch-beating goal is still not complete. The remaining
Base+ q4_1 gap on this run is about `2.23 ms/frame`, so q4_1 remains the better
optimization target than q8_0.

A small MMQ-body probe then skipped the regular-matrix initialization of
`ids_dst_shared[j] = j` and wrote back with direct `j` indices when no MoE ids
are present. Standalone MMQ rows looked mildly positive:

| Shape | Default mean | Direct writeback mean |
| --- | ---: | ---: |
| `q4_1 k=448 rows=1792 cols=4096 bias+GELU` | `0.172639 ms` | `0.171802 ms` |
| `q4_1 k=1792 rows=448 cols=4096` | `0.183220 ms` | `0.170145 ms` |
| `q4_1 k=448 rows=1344 cols=196 channels=25 bias+GELU` | `0.174644 ms` | `0.173142 ms` |

The app-level result did not justify keeping it. The 10-frame Base+ q4_1 JSONL
comparison was exact (`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0`), but five paired bbox-only graph runs were unchanged:
default mean `45.34 ms/frame`, direct-writeback mean `45.32 ms/frame`, paired
saved mean `0.02 ms/frame` with `0.74 ms` stdev. The probe was removed.

Evidence:

- `outputs/current-baseplus-q8q41-vs-python-refresh-20260517/q8_0/summary.json`
- `outputs/current-baseplus-q8q41-vs-python-refresh-20260517/q4_1/summary.json`
- `outputs/mmq-no-ids-direct-writeback-probe-20260517/`
- `outputs/mmq-no-ids-direct-writeback-app-20260517/summary.json`

A compile-time specialization of the same direct-writeback idea was also
checked and removed. It avoided the runtime branch by templating the MMQ
writeback path, but the standalone rows were mixed rather than clearly better:

| Shape | Default mean | Compile-time direct mean |
| --- | ---: | ---: |
| `q4_1 k=448 rows=1792 cols=4096 bias+GELU` | `0.170170 ms` | `0.179642 ms` |
| `q4_1 k=1792 rows=448 cols=4096` | `0.181624 ms` | `0.170487 ms` |
| `q4_1 k=448 rows=1344 cols=196 channels=25 bias+GELU` | `0.171046 ms` | `0.171115 ms` |

Since only one of the three representative shapes improved and the runtime
version had already shown no graph-level movement, this probe was not promoted
to an app-level A/B. The source changes were removed before continuing with
larger fusion-kernel work.

Evidence:

- `outputs/mmq-no-ids-direct-writeback-template-probe-20260517b/`

### Fusion-kernel direction after the small MMQ probes

The current evidence points away from launch-count-only MMQ tweaks and toward
larger kernels that remove a real memory round trip or a repeated quantization
step. The next fusion work should be evaluated in this order:

1. **Producer MMQ epilogue side-output for q8_1**: generate the consumer's q8_1
   MMQ layout inside the producer MMQ writeback for the Hiera MLP
   `fc1 + bias + GELU -> fc2` chain. The separate post-MMQ side-output kernel
   was exact but slower because it still reread and rewrote the f32 activation.
   A useful version must avoid that extra f32 pass.
2. **Two-stage MLP tile fusion probe**: for the specific Hiera
   `448 -> 1792 -> 448` and `448 -> 1344 -> 448` shapes, test whether a
   producer tile can feed the consumer quantization/layout directly. This is
   higher risk than the epilogue side-output because it couples two matmuls, so
   it should start behind an env gate and a standalone benchmark before graph
   integration.
3. **Head-dim 56 attention arithmetic**: pack/slice overhead is now small; the
   D64 padded MMA body dominates. This is no longer a fusion-cleanup problem
   and should be treated as a specialized attention-kernel problem.

Acceptance criteria for keeping any fusion-kernel change:

- Same-binary disabled/enabled full-mask JSONL parity on the 10-frame Base+
  1024 sample: `mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`,
  `max_bbox_delta_px=0`, and zero score/mask-area deltas unless explicitly
  documented as a non-bit-exact numeric path.
- At least five paired bbox-only graph runs. Standalone kernel speedups are
  only screening evidence; they are not enough to keep the change.
- A positive graph-level paired mean larger than the observed run variance.
  The current remaining q4_1 gap to official PyTorch is about
  `2.23 ms/frame`, so sub-`0.2 ms/frame` wins should be treated as cleanup,
  not goal-completing work.

A fresh 2026-05-17 q4_1 Base+ 3-frame MMQ profile confirms the practical upper
bound for the first item. With `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE` disabled,
the Hiera MLP producer->consumer q8_1 side-output candidates are:

| Consumer shape | Count | Consumer quant sum | Mean quant |
| --- | ---: | ---: | ---: |
| `1792x4096 -> 448x4096` | `33` | `0.908192 ms` | `0.027521 ms` |
| `896x16384 -> 224x16384` | `6` | `0.767488 ms` | `0.127915 ms` |
| `3584x1024 -> 896x1024` | `9` | `0.129344 ms` | `0.014372 ms` |

Total removable consumer quantization in this 3-frame synchronized profile is
`1.805024 ms`, or roughly `0.60 ms/frame` before accounting for the extra work
inside the producer kernel. This is material but smaller than the full
PyTorch gap, so it should be treated as the next focused fusion-kernel probe,
not as sufficient by itself. With the current prequant-cache plumbing enabled,
the same script reports zero remaining direct producer->consumer candidates,
which means any new in-MMQ side-output path must be compared against both the
default path and the existing prequant-cache path.

A normal five-pair bbox-only A/B confirms that the existing opt-in prequant
cache is not a speed win in its current form:

| Path | Track ms/frame values | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| default / prequant disabled | `45.5, 45.4, 45.2, 45.2, 44.8` | `45.22` | `45.2` | `0.27` |
| `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` | `45.2, 45.8, 45.1, 45.4, 45.7` | `45.44` | `45.4` | `0.30` |

The paired enabled-minus-disabled mean is `+0.22 ms/frame`, so the current
cache should remain opt-in diagnostic plumbing. A producer-side fusion must
beat the default path directly, not merely improve over the opt-in cache.

The same q4_1 profile's Hiera node-hotspot ranking keeps the broader priority
order clear. Using drop-max sums to reduce first-use profiler noise, the top
steady groups are:

| Rank | Kernel group | Count | Drop-max sum | Note |
| ---: | --- | ---: | ---: | --- |
| 1 | `MUL_MAT dst=f32[1792,4096]` | `48` | `10.576864 ms` | stage-2 quantized MLP expansion |
| 2 | `MUL_MAT dst=f32[1344,196,25]` | `36` | `6.824736 ms` | window qkv quantized projection |
| 3 | `MUL_MAT dst=f32[448,4096]` | `48` | `6.364320 ms` | stage-2 quantized MLP projection |
| 4 | `FLASH_ATTN_EXT dst=f32[56,8,196,25]` | `36` | `5.111744 ms` | head_dim 56 window attention |
| 5 | `FLASH_ATTN_EXT dst=f32[56,8,4096]` | `9` | `4.870368 ms` | head_dim 56 global attention |

So the practical ordering is: first avoid spending more time on the current
opt-in prequant cache, then only attempt producer-side q8_1 fusion as a bounded
`~0.6 ms/frame` probe, and keep the larger effort on MMQ arithmetic throughput
and head_dim 56 attention arithmetic. Earlier FATTN56 experiments already show
that large-sequence inline pack is not the answer: it removes pack/slice but
raises the global attention MMA body from about `0.76 ms` to about `0.93 ms`,
so the current padded-MMA path remains the baseline.

Evidence:

- `outputs/fusion-direction-profile-20260517/q4_1_profile_no_prequant.log`
- `outputs/fusion-direction-profile-20260517/q4_1_gelu_side_output_candidates_no_prequant.json`
- `outputs/fusion-direction-profile-20260517/q4_1_profile.log`
- `outputs/fusion-direction-profile-20260517/q4_1_gelu_side_output_candidates.json`
- `outputs/fusion-direction-profile-20260517/q4_1_hiera_hotspots.json`
- `outputs/prequant-cache-current-ab-20260517/summary.json`

### 2026-05-17: Rejected q4_1 4096-column `mmq_x=128` probe

A focused microbench retested the q4_1 Hiera MLP `ncols=4096` shapes with
runtime `mmq_x` caps. The standalone kernel numbers looked promising for the
MLP expansion shape:

| Shape | Default mean | Best tested cap | Best mean |
| --- | ---: | --- | ---: |
| `k=448 rows=1792 cols=4096 bias+GELU` | `0.175191 ms` | `128` | `0.142940 ms` |
| `k=1792 rows=448 cols=4096` | `0.181942 ms` | `104` | `0.169673 ms` |

The app-level full-mask parity check rejected making this a default. With
`GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX=128`, only `1 / 10` mask hashes matched
the default run, the worst bbox IoU was `0.998742`, max bbox delta was `1 px`,
max score delta was `0.004063`, and max relative mask-area delta was
`0.000867`. This is small numerically, but it is not exact enough for the
current parity rule.

Evidence:

- `outputs/mmq-q41-4096-xmax-probe-20260517/summary.json`
- `outputs/mmq-q41-4096-xmax128-app-20260517/default-vs-x128.json`

### 2026-05-17: Rejected producer-side q8_1 fusion prototype

An env-gated prototype tried the next fusion-kernel idea directly: have the
q4_1 MMQ producer write a q8_1 side output during the MMA writeback, then let
the next q4_1 consumer use that side output instead of launching a separate
`quantize_mmq_q8_1` pass. The prototype was removed after measurement because
it failed both gates:

| Gate | Result |
| --- | --- |
| Full-mask parity | `0 / 10` mask hashes matched; max score delta `0.009130`; max relative mask-area delta `0.017632` |
| bbox-only speed | default mean `45.63 ms/frame`; fusion prototype mean `47.27 ms/frame`; paired mean delta `+1.63 ms/frame` |

The failed result is useful because it changes the fusion direction. A naive
side-output path increases register/shared-memory pressure enough to erase the
launch/quantization savings, and it is also not numerically identical. The next
fusion attempt should not store a full f32 tile in shared memory after MMQ
writeback. It needs either a bit-exact q8_1 reduction integrated into the MMA
fragment writeback with lower scratch pressure, or it should be dropped in
favor of body-level MMQ/FATTN arithmetic work.

Evidence:

- `outputs/mmq-producer-q8-side-fusion-20260517/default-vs-fusion.json`
- `outputs/mmq-producer-q8-side-fusion-20260517/bbox_speed_summary.json`

### 2026-05-17: Accepted f32 axis-2 broadcast ADD cleanup

The Hiera profile also showed large f32 broadcast ADD nodes such as
`node_9`, where `src1` is shaped like `[1,1,C,1]`. The existing f32 single-axis
broadcast fast path still computed the broadcast axis with per-element
division/modulo. A dedicated axis-2 kernel now maps the 2D plane and channel
axis directly, avoiding that arithmetic for `[H,W,C]`-style channel broadcasts.
The old generic single-axis fast path can be restored with
`GGML_CUDA_DISABLE_BIN_BCAST_AXIS2_FAST=1`.

Full-mask parity against the disabled path was exact on the 10-frame Base+
q4_1 sample: `10 / 10` mask hashes matched, bbox deltas were zero, and score
deltas were zero. Fifteen paired bbox-only runs show a small non-regressing
cleanup, but not a PyTorch-closing win:

| Path | Mean track/fr | Median | 95% CI |
| --- | ---: | ---: | --- |
| `GGML_CUDA_DISABLE_BIN_BCAST_AXIS2_FAST=1` | `45.293 ms` | `45.3 ms` | `44.972..45.615 ms` |
| default axis-2 fast path | `45.047 ms` | `45.0 ms` | `44.828..45.265 ms` |
| paired saved | `0.247 ms` | `0.2 ms` | `-0.201..0.694 ms` |

This is accepted because it is bit-exact and removes unnecessary index
arithmetic from a general CUDA primitive, but its effect is below the remaining
Base+ q4_1 gap to official PyTorch. The next PyTorch-closing work still needs
to target q4_1 MMQ body throughput or head_dim 56 FATTN body throughput.

Evidence:

- `outputs/binbcast-axis2-fast-20260517/disabled-vs-enabled.json`
- `outputs/binbcast-axis2-fast-20260517/disabled-vs-enabled-after-guard.json`
- `outputs/binbcast-axis2-fast-20260517/bbox_speed_summary_15.json`
- `outputs/binbcast-axis2-fast-20260517/profile_disabled.log`
- `outputs/binbcast-axis2-fast-20260517/profile_enabled.log`

### 2026-05-17: Exact but tiny f32 axis-0 broadcast ADD cleanup

The next broadcast cleanup specialized the bias-like f32 ADD pattern where
`src1` is shaped like `[C,1,1,1]`, for example Hiera MLP and decoder projection
bias nodes. The new axis-0 kernel maps one CUDA block row to each outer slice
and indexes `src1[i0]` directly instead of using the generic single-axis
division/modulo path. The path can be disabled with
`GGML_CUDA_DISABLE_BIN_BCAST_AXIS0_FAST=1`.

Full-mask parity against the disabled path was exact on the same 10-frame
Base+ q4_1 sample: `10 / 10` mask hashes matched, bbox deltas were zero, and
score deltas were zero. The node-level synchronized profile shows the intended
micro cleanup for axis-0 ADDs, but the end-to-end gain is below measurement
noise:

| Measurement | Disabled | Enabled | Delta |
| --- | ---: | ---: | ---: |
| Axis-0 ADD node sum, 3-frame profile | `1.248288 ms` | `1.192064 ms` | `0.056224 ms` saved |
| Axis-0 ADD node mean | `0.011452 ms` | `0.010936 ms` | `0.000516 ms` saved |
| 15-pair bbox-only track mean | `45.400 ms/frame` | `45.733 ms/frame` | `-0.333 ms/frame` paired |

The paired track confidence interval is `-1.017..0.350 ms/frame`, so this is
not a PyTorch-closing optimization. It is kept only as a bit-exact primitive
cleanup; the remaining meaningful work is still MMQ/FATTN body throughput or a
lower-pressure producer/consumer fusion that preserves exact output.

Evidence:

- `outputs/binbcast-axis0-fast-20260517/disabled-vs-enabled.json`
- `outputs/binbcast-axis0-fast-20260517/bbox_speed_summary_15.json`
- `outputs/binbcast-axis0-fast-20260517/profile_axis0_add_summary.json`

### 2026-05-17: Accepted D56 no-mask FATTN slope cleanup

The D56 padded-MMA FlashAttention path already specializes the SAM2 Hiera
no-mask/no-sinks case, but the kernel wrapper still computed the ALiBi slope
before entering the templated body. In the `no_mask_no_sinks` instantiation,
the slope value is dead because all mask-add uses are compiled out. The CUDA
FATTN wrapper now sets slope to `1.0f` for that template case and leaves masked
or ALiBi-capable paths unchanged.

This is a body cleanup rather than a selector change. It does not alter the
padded-64 MMA body, D56 output handling, or stream-k/fixup behavior.

Validation:

| Check | Result |
| --- | --- |
| D56 global parity bench, `N=4096 heads=8` | pass, `max_abs=0.00513070542`, `mean_ms=0.717641` |
| D56 window parity harness, `N=196 batch=25` | existing CPU-reference tolerance is too strict for this harness; app-level parity below is authoritative for the tracked path |
| App full-mask parity vs previous JSONL | `10 / 10` mask hashes matched; bbox and score deltas zero |
| bbox-only speed smoke | `45.6 ms/frame` mean over 5 runs, within the current 45 ms band |

This is expected to be small, but it removes dead work from a primary hotspot
without changing tracked output. The larger open item remains a real MMQ body
optimization, especially q4_1 unpack/address work in the inner dot path.

Evidence:

- `outputs/fattn-d56-slope-skip-20260517/global_n4096.log`
- `outputs/fattn-d56-slope-skip-20260517/window_n196_b25.log`
- `outputs/fattn-d56-slope-skip-20260517/previous-vs-current.json`
- `outputs/fattn-d56-slope-skip-20260517/bbox_speed_smoke.json`

### 2026-05-17: q4_1 MMQ inner-loop offset hoist

The earlier q4_1 MMQ duplicate-unpack candidate no longer applied to this
worktree because the q4_1 DP4A path already uses the direct `q4_1 x q8_1`
dot helper instead of the older temporary `u[]` unpack block. A smaller body
cleanup still applied: `kyqs`, the y-side q8 offset, and the y-side scale offset
were invariant across the `i0` row-subtile loop. They are now computed once per
`(k01, j0)` tile instead of once per row-subtile.

Validation:

| Check | Result |
| --- | --- |
| MMQ `fc1` reduced check, `k=448 rows=1792 cols=1024 bias+GELU` | `bad=0/1835008`, `mean_ms=0.047480` |
| MMQ window qkv reduced check, `k=448 rows=1344 cols=196 channels=8` | `bad=0/2107392`, `mean_ms=0.054012` |
| Full-size `fc1`, `cols=4096 bias+GELU` | `mean_ms=0.160033` |
| Full-size `fc2`, `k=1792 rows=448 cols=4096` | `mean_ms=0.182902` |
| Full-size window qkv, `channels=25` | `mean_ms=0.154511` |
| App full-mask parity vs previous JSONL | `10 / 10` mask hashes matched; bbox and score deltas zero |
| bbox-only speed smoke | runs `45, 45, 46, 45, 49`, mean `46.0 ms/frame`, median `45.0 ms/frame` |

This is exact and removes redundant integer work from the hot q4_1 body, but it
does not yet provide a statistically proven end-to-end win. The next MMQ work
needs a larger body change, not just invariant-offset cleanup.

Evidence:

- `outputs/mmq-q41-offset-hoist-20260517/mlp_fc1.log`
- `outputs/mmq-q41-offset-hoist-20260517/mlp_fc2.log`
- `outputs/mmq-q41-offset-hoist-20260517/qkv_window.log`
- `outputs/mmq-q41-offset-hoist-20260517/mlp_fc1_check_small.log`
- `outputs/mmq-q41-offset-hoist-20260517/qkv_window_check_small.log`
- `outputs/mmq-q41-offset-hoist-20260517/previous-vs-current.json`
- `outputs/mmq-q41-offset-hoist-20260517/bbox_speed_smoke.json`

### 2026-05-17: Current post-cleanup Hiera hotspot profile

After the small broadcast/FATTN/MMQ cleanups, a synchronized 3-frame Base+
q4_1 profile still shows steady `hiera_encode` around `38.4 ms` after the first
frame, versus the refreshed official PyTorch `forward_image` row around
`28.09 ms`. The remaining steady hotspots are not producer-side q8 side-output
candidates; they are mostly MMQ body throughput and D56 FlashAttention body
throughput:

| Rank | Kernel group | Drop-max sum | Mean after drop | Direction |
| ---: | --- | ---: | ---: | --- |
| 1 | q4_1 MLP expansion, `dst=f32[1792,4096]` | `8.496096 ms` | `0.180768 ms` | MMQ body |
| 2 | q4_1 MLP projection, `dst=f32[448,4096]` | `7.858784 ms` | `0.167208 ms` | MMQ body |
| 3 | q4_1 window qkv, `dst=f32[1344,196,25]` | `7.015648 ms` | `0.200447 ms` | MMQ body |
| 4 | D56 window FATTN, `dst=f32[56,8,196,25]` | `5.602400 ms` | `0.160069 ms` | FATTN body |
| 5 | D56 global FATTN, `dst=f32[56,8,4096]` | `4.988000 ms` | `0.623500 ms` | FATTN body |

The MMQ profile reports only one direct producer->consumer q8 candidate after
the current graph/fusion changes, and its removable consumer quantization is
only `0.009856 ms`. This rules out revisiting the broad producer-side q8 cache
as the main path for the current graph.

Evidence:

- `outputs/current-hotspots-after-fusion-cleanups-20260517/profile.log`
- `outputs/current-hotspots-after-fusion-cleanups-20260517/hotspots.json`
- `outputs/current-hotspots-after-fusion-cleanups-20260517/mmq_summary.json`
- `outputs/current-hotspots-after-fusion-cleanups-20260517/fattn56_summary.json`

### 2026-05-17: Rejected D56 no-fixup direct epilogue probe

A narrow D56 FATTN body probe tried to bypass the shared-memory write/read
epilogue when `no_mask_no_sinks && !needs_fixup && !is_fixup && np == 1`. The
prototype wrote `VKQ_C` fragments directly to `dst` after dividing by the
per-column softmax rowsum. It preserved the dedicated D56 global parity bench,
but did not improve speed:

| Probe | Result |
| --- | --- |
| D56 global parity bench | pass, `max_abs=0.00513070542`, `bad=0/4032` |
| D56 global mean | previous nearby `0.717641 ms`; direct epilogue `0.719069 ms` |

Because the direct epilogue did not beat the existing shared-memory epilogue,
the prototype was removed. This makes the next D56 work less likely to be an
epilogue-only change; it needs to reduce MMA body work or improve the D56
tiling itself.

Evidence:

- `outputs/fattn-d56-direct-epilogue-20260517/global_n4096.log`

### 2026-05-17: Rejected current q4_1 MMQ stream-k disable probe

The current q4_1 MMQ launch profile shows all high-volume Hiera MMQ shapes
using stream-k with high tile efficiency and usually no fixup. A diagnostic
gate was tested to disable stream-k and force ordinary `x/y/z` tiling, but it
does not satisfy the parity rule. On the 10-frame Base+ q4_1 full-mask run,
disabling stream-k produced:

| Metric | Result |
| --- | --- |
| mask hashes | `0 / 10` matched |
| min bbox IoU | `0.998742` |
| max bbox delta | `1 px` |
| max score delta | `0.008234` |
| max relative mask-area delta | `0.004812` |

This is the same class of issue as earlier tile/schedule probes: bbox remains
close, but mask logits cross enough thresholds that full-mask parity fails.
The diagnostic source gate was removed; stream-k remains the default current
path.

Evidence:

- `outputs/mmq-launch-current-20260517/launch_summary.json`
- `outputs/mmq-streamk-disable-current-20260517/default-vs-no-streamk.json`

### 2026-05-17: Rejected q4_1 MMQ load-tile metadata fusion

The next fusion-kernel probe stayed inside the hot q4_1 MMQ MMA body. The q4_1
tile loader previously unpacked `qs` in one loop and then loaded the per-block
`dm` metadata in a second loop over the same q4_1 blocks. For q4_1,
`QI4_1 == 4`, `MMQ_TILE_NE_K == 32`, and the first loop already visits all
eight q4_1 blocks per row. The probe folded the `dm` load into that loop on
`kqsx == 0`, preserving the same shared-memory layout and arithmetic while
removing the second block walk. It was removed after the 15-run A/B because it
did not improve end-to-end speed.

Validation:

| Check | Result |
| --- | --- |
| Build | `just build-target sam3_mmq_bench`, `just build-target sam3_benchmark` passed |
| MMQ `fc1` reduced check, `k=448 rows=1792 cols=1024 bias+GELU` | `bad=0/1835008` |
| MMQ window qkv reduced check, `k=448 rows=1344 cols=196 channels=8` | `bad=0/2107392` |
| Full-mask app parity vs previous q4_1 JSONL | `10 / 10` mask hashes matched; bbox and score deltas zero |
| MMQ `fc1`, `k=448 rows=1792 cols=4096 bias+GELU` | `mean_ms=0.163790`, `median_ms=0.167512` |
| MMQ `fc2`, `k=1792 rows=448 cols=4096` | `mean_ms=0.174650`, `median_ms=0.172559` |
| MMQ window qkv, `k=448 rows=1344 cols=196 channels=25` | `mean_ms=0.159479`, `median_ms=0.153845` |
| Synchronized Hiera profile, steady frames | `35.494`, `35.097 ms` |
| 15-run bbox-only baseline | mean `45.20 ms/frame`, median `45.0`, CI `44.97..45.43` |
| 15-run bbox-only fusion probe | mean `45.33 ms/frame`, median `45.0`, CI `44.99..45.68` |
| Paired baseline-minus-fusion delta | mean `-0.13 ms/frame`, CI `-0.49..0.22`; negative means the probe was slower |
| Final rollback parity vs previous q4_1 JSONL | `10 / 10` mask hashes matched; bbox and score deltas zero |

The microbench result is mixed: the MLP projection improved versus the previous
nearby `0.182902 ms`, while the MLP expansion and window qkv are within noise or
slightly worse than the previous nearby `0.160033` and `0.154511 ms` rows. The
app-level synchronized profile is more encouraging than the prior post-cleanup
profile (`~38.4 ms` steady), but the normal bbox-only A/B does not show a
speedup. The source hunk was therefore removed. This narrows the fusion-kernel
direction again: reducing q4_1 loader bookkeeping alone is too small, and the
next useful attempt needs either a real two-stage MLP tile fusion or a larger
q4_1 MMA body change.

The refreshed MMQ profile still reports only one direct producer->consumer q8
side-output candidate, with removable consumer quantization of `0.009728 ms`.
That keeps broad producer-side q8 cache/fusion low priority for the current
graph; the next larger fusion candidate remains a real two-stage MLP tile
fusion or a lower-level q4_1 MMA body change.

Evidence:

- `outputs/mmq-q41-load-dm-fusion-20260517/mlp_fc1_serial.log`
- `outputs/mmq-q41-load-dm-fusion-20260517/mlp_fc2_serial.log`
- `outputs/mmq-q41-load-dm-fusion-20260517/qkv_window_serial.log`
- `outputs/mmq-q41-load-dm-fusion-20260517/mlp_fc1_check_small.log`
- `outputs/mmq-q41-load-dm-fusion-20260517/qkv_window_check_small.log`
- `outputs/mmq-q41-load-dm-fusion-20260517/previous-vs-current.json`
- `outputs/mmq-q41-load-dm-fusion-20260517/current-bbox15/summary.json`
- `outputs/mmq-q41-load-dm-fusion-20260517/baseline-bbox15/summary.json`
- `outputs/mmq-q41-load-dm-fusion-20260517/bbox15_ab_summary.json`
- `outputs/mmq-q41-load-dm-fusion-20260517/previous-vs-final-rollback.json`
- `outputs/mmq-q41-load-dm-fusion-20260517/profile/profile_summary.json`
- `outputs/mmq-q41-load-dm-fusion-20260517/profile/hotspots.json`
- `outputs/mmq-q41-load-dm-fusion-20260517/profile/mmq_summary.json`

### 2026-05-17: Current setting and precision sweep after the rejected fusion

After removing the q4_1 load-tile metadata fusion probe, the same 10-frame
Base+ q4_1 bbox-only run was swept over CPU thread count to check whether the
remaining PyTorch gap was outside the CUDA kernels. It was not: `--n-threads`
from `1` through `16` stayed around `45 ms/frame`, with only noise-level
movement.

| Threads | Track ms/frame values | Mean |
| ---: | --- | ---: |
| 1 | `46, 45, 45` | `45.33` |
| 2 | `45, 45, 45` | `45.00` |
| 4 | `45, 45, 45` | `45.00` |
| 8 | `45, 45, 45` | `45.00` |
| 12 | `45, 45, 46` | `45.33` |
| 16 | `45, 45, 45` | `45.00` |

A one-run precision smoke on the same frames confirms that q4_0/q4_1 are still
the speed candidates for this CUDA path. Blackwell FP4 model files exist and
run, but they do not beat the q4 rows in this workload.

| Precision | Track/fr | P50 | P95 |
| --- | ---: | ---: | ---: |
| q4_0 | `45 ms` | `43 ms` | `54 ms` |
| q4_1 | `45 ms` | `43 ms` | `53 ms` |
| mxfp4 | `51 ms` | `51 ms` | `58 ms` |
| q8_0 | `51 ms` | `53 ms` | `59 ms` |
| f16 | `59 ms` | `57 ms` | `67 ms` |
| nvfp4 | `64 ms` | `61 ms` | `76 ms` |
| f32 | `67 ms` | `64 ms` | `80 ms` |

This keeps the optimization priority on q4_0/q4_1 Hiera encode kernels rather
than thread tuning or switching to the current FP4/F16/F32 model variants.
For the q4_1 quality-compatible path, `--multimask` does not materially change
steady tracking speed: five bbox-only runs were `45.0 ms/frame` with and without
`--multimask`, while initialization stayed within noise (`405.0 ms` default vs
`406.8 ms` multimask).

Evidence:

- `outputs/thread-sweep-q41-1024-20260517/summary.json`
- `outputs/baseplus-precision-bbox-smoke-20260517/summary.json`
- `outputs/baseplus-q41-multimask-speed-20260517/summary.json`

### 2026-05-17: Normal q4_1 stage profile after rollback

After the rejected q4_1 load-tile metadata fusion was removed, a normal
10-frame Base+ q4_1 `--multimask` profile was captured without the CUDA node
profiler. This is the current baseline for the next fusion-kernel pass:

| Stage | Values | Interpretation |
| --- | --- | --- |
| `hiera_encode` graph compute | first frame `152.623 ms`; steady frames roughly `30.55..32.87 ms` | still about `3 ms/frame` slower than official PyTorch `forward_image=28.09 ms` |
| `propagate_single` graph compute | mean `9.18 ms` over frames 1..9 | lower than official PyTorch `track_step=13.11 ms`; not the primary kernel target |
| reported benchmark row | `track_ms=47.6`, `p50=44.5`, `p95=56.0` | includes CPU overheads and warm/outlier effects, so use paired normal A/B for default decisions |
| main CPU overheads | `prop_memory_slot_read` first outlier `9.737 ms`, then about `0.2..0.4 ms`; `prop_graph_alloc` one outlier `8.578 ms`; `prop_rope_k_build` mean `0.712 ms` | useful cleanup targets, but they do not replace the Hiera kernel target |

This baseline changes the fusion-kernel acceptance rule slightly. A candidate
that only improves synchronized Hiera node rows is not enough if the normal
paired run does not move: the next defaultable fusion must preserve full-mask
parity and show a paired bbox/full app speed win, or be kept as an opt-in
diagnostic. The priority remains a body-level q4_1 MMQ or D56 attention change,
not another scalar post-op launch fusion.

Evidence:

- `outputs/current-stage-q41-multimask-20260517/profile.log`
- `outputs/current-stage-q41-multimask-20260517/profile_summary.json`

### 2026-05-17: Two-stage MLP fusion benchmark surface

`examples/mmq_bench.cpp` now has a `--two-stage --fc2-rows` mode for the Hiera
MLP pattern. It builds the standalone graph:

```text
q4_1 fc1: 448 -> 1792, cols=4096
ADD bias
GELU
q4_1 fc2: 1792 -> 448, cols=4096
optional ADD bias
```

The default two-stage path intentionally omits an explicit `CONT` between GELU
and `fc2`, matching `sam2_hiera_block_forward`. `--two-stage-cont` is available
only as an opt-in stress row for the older materialized surface.

This is not the fused kernel yet. It is the acceptance surface for the next
kernel probe: any real two-stage fusion must beat this standalone graph before
it is worth integrating into the full SAM3 graph.

Current benchmark rows on the same CUDA build:

| Row | Mean | Median | Notes |
| --- | ---: | ---: | --- |
| fc1 `448 -> 1792`, `bias+GELU` | `0.180520 ms` | `0.152740 ms` | single-MMQ screening row |
| fc2 `1792 -> 448`, `bias` | `0.174304 ms` | `0.174172 ms` | single-MMQ screening row |
| two-stage graph, Hiera-like no explicit `CONT` | `0.313808 ms` | `0.313627 ms` | current unfused two-stage target |
| opt-in two-stage graph with explicit `CONT` | `0.412759 ms` | `0.412747 ms` | older materialized stress row |

The synchronized MMQ profile for the two-stage graph shows the steady second
stage at about `quant_ms=0.0508 ms` and `mmq_ms=0.1232 ms`, while the steady
first stage is about `quant_ms=0.0115 ms` and `mmq_ms=0.1557 ms`. This means a
true fusion has a bounded target: remove or reduce the intermediate
materialization/consumer quantization without perturbing the q4_1 MMQ arithmetic
schedule. The earlier post-MMQ side-output prototype was slower because it
added another f32 read/write pass; this benchmark keeps that failure mode
visible.

The `--check` mode now works for `--two-stage` by comparing the CUDA graph
against a CPU backend graph using the same quantized weights. This is a
microbench guard only; keep using the 10-frame Base+ full-mask JSONL comparison
for any actual fusion candidate.

Evidence:

- `outputs/fusion-two-stage-mlp-bench-20260517/q41_fc1_bias_gelu_cols4096.log`
- `outputs/fusion-two-stage-mlp-bench-20260517/q41_fc2_bias_cols4096.log`
- `outputs/fusion-kernel-acceptance-harness-20260517/q41_two_stage_no_cont_default_timing.log`
- `outputs/fusion-kernel-acceptance-harness-20260517/q41_two_stage_cont_optin_timing.log`
- `outputs/fusion-two-stage-mlp-bench-20260517/q41_two_stage_profile.log`

### 2026-05-17: Rejected propagation RoPE-K backend cache

A propagation-side cache probe moved `rope_k` from per-frame CPU build and
host upload to a backend tensor copied into the propagation graph input. The
idea was to remove the `prop_rope_k_build` span, which was around
`0.84 ms/frame` in the disabled profile.

Correctness was exact on the 10-frame Base+ q4_1 full-mask comparison:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, and zero bbox/score/mask-area
deltas. The speed profile did not justify the change:

| Path | `prop_rope_k_build` mean | `prop_input_upload` mean | `propagate_single` mean | Track row |
| --- | ---: | ---: | ---: | ---: |
| disabled / old path | `0.843 ms` | `1.639 ms` | `14.728 ms` | `92.9 ms/frame` |
| backend cache probe | `0.000 ms` | `3.543 ms` | `18.612 ms` | `92.7 ms/frame` |

The cache removed CPU construction but replaced it with a more expensive graph
input copy/synchronization pattern. Because this is not a normal-app speed win
and worsens the propagation compute span, the source change was removed. Any
future propagation-side cache should avoid copying a separate backend tensor
into a fresh graph input each frame; it would need graph-input aliasing or a
longer-lived propagation graph to be worthwhile.

Evidence:

- `outputs/prop-rope-k-backend-cache-20260517/disabled-vs-enabled.json`
- `outputs/prop-rope-k-backend-cache-20260517/profile-disabled/profile_summary.json`
- `outputs/prop-rope-k-backend-cache-20260517/profile-enabled/profile_summary.json`

### 2026-05-17: Rejected propagation memory-slot CPU shadow cache

Another propagation-side probe stored a CPU shadow copy of each memory slot at
`sam3_encode_memory` time and used it in `sam3_propagate_single` instead of
reading `spatial_feats` back from the backend tensor. This targets
`prop_memory_slot_read`, whose first frame often includes a large synchronization
outlier and whose steady frames are usually a few tenths of a millisecond.

Correctness was exact in the same-binary disabled/enabled full-mask comparison:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, and zero bbox/score/mask-area
deltas. The speed result was not a win:

| Path | Bbox-only track values | Mean | Median |
| --- | --- | ---: | ---: |
| disabled / backend readback | `91, 46, 45, 45, 45` | `54.4 ms` | `45.0 ms` |
| CPU shadow cache | `91, 45, 45, 47, 45` | `54.6 ms` | `45.0 ms` |
| paired disabled-minus-enabled | `0, 1, 0, -2, 0` | `-0.2 ms` | `0.0 ms` |

The profile confirms that the span moved in the intended direction
(`prop_memory_slot_read` mean `1.312 -> 1.167 ms`), but the full app timing did
not improve and the extra resident CPU copy is not justified. The source change
was removed. This leaves propagation readback cleanup as a secondary path unless
it can also remove graph/input upload work or avoid the first-frame
synchronization more directly.

Evidence:

- `outputs/prop-memory-slot-cpu-cache-20260517/disabled-vs-enabled.json`
- `outputs/prop-memory-slot-cpu-cache-20260517/bbox_ab_summary.json`
- `outputs/prop-memory-slot-cpu-cache-20260517/profile-disabled/profile_summary.json`
- `outputs/prop-memory-slot-cpu-cache-20260517/profile-enabled/profile_summary.json`

### 2026-05-17: Rejected D56 `Q_in_reg=false` probe

A D56 FlashAttention body probe changed the Ampere-family
`DKQ=64,DV=64` MMA config from `Q_in_reg=true` to `false`. The target was the
current D56 direct-output route (`D_SRC=56,DV_DST=56`), where Q is loaded from
f32 and padded to the D64 MMA body. The hypothesis was that not keeping Q
resident in registers might reduce register pressure for this inline-head path.

The standalone parity harness preserved the expected tolerance, but the speed
was clearly worse:

| Shape | Result |
| --- | --- |
| global `D=56,N=4096,heads=8,batch=1,sampled_q=9` | `mean_ms=0.919285`, `median_ms=0.925995`, `bad=0/4032` |
| window `D=56,N=196,heads=8,batch=25` | `mean_ms=0.927918`, `median_ms=0.926373`, `bad=0/2195200` |

The source change was removed. This keeps Q register residency as the better
current setting for D56; the next D56 attempt needs a different body-level
change rather than lowering Q residency.

Evidence:

- `outputs/fattn56-qinreg-false-probe-20260517/global.log`
- `outputs/fattn56-qinreg-false-probe-20260517/window.log`

### 2026-05-17: Fusion-kernel acceptance harness update

`examples/mmq_bench.cpp` now allows `--two-stage --check` instead of rejecting
it. The check compares the CUDA two-stage MLP graph against a CPU backend graph
with the same quantized weights, so future fusion-kernel prototypes can be
screened at the graph level before running full SAM quality checks.

The valid parity smoke command for the q4_1 Hiera MLP shape now supports normal
benchmark warmup and repeated graph execution:

```sh
./build/xmake-release-cuda/examples/sam3_mmq_bench \
  --cuda --type q4_1 --k 448 --rows 1792 --fc2-rows 448 --cols 128 \
  --two-stage --bias --check --warmup 1 --iters 3 --tolerance 10
```

This passed with `max_abs=0.207466185`, `mean_abs=0.0354335134`, and
`bad=0/57344`. The tolerance is intentionally not an exact-mask criterion; it is
a microbench guard for the quantized CUDA-vs-CPU graph. Any candidate that passes
this still needs the normal full-mask SAM parity gate.

The harness initially exposed a benchmark reliability issue: the artificial
two-stage graph produced nonfinite outputs on the second graph execution. The
bench now reinitializes graph inputs before each two-stage run, which restores
repeatability while keeping input transfer outside the measured graph compute
span. The refreshed Hiera-like full-shape timing is `mean_ms=0.313808`,
`median_ms=0.313627`, `p95_ms=0.315310` for `cols=4096`. The explicit-`CONT`
stress row is slower at `mean_ms=0.412759`, confirming that the real fusion
target should not include a materialization that the current Hiera graph already
avoids.

Fusion-kernel promotion policy:

1. Use `--two-stage --check --warmup 1 --iters 3` as the micro parity smoke.
2. Use the `cols=4096` two-stage row as the unfused microbenchmark target.
3. Promote no fusion kernel without full SAM full-mask parity and a paired
   bbox/profile speed run against the default path.

Evidence:

- `examples/mmq_bench.cpp`
- `outputs/fusion-kernel-acceptance-harness-20260517/q41_two_stage_no_cont_default_check.log`
- `outputs/fusion-kernel-acceptance-harness-20260517/q41_two_stage_no_cont_default_timing.log`
- `outputs/fusion-kernel-acceptance-harness-20260517/q41_two_stage_cont_optin_timing.log`

### 2026-05-17: Current D56 standalone refresh after fusion triage

After narrowing the two-stage MLP fusion target to the Hiera-like no-`CONT`
surface, D56 FlashAttention was refreshed with `sam3_fattn_parity` on the same
CUDA build. Both main Hiera shapes still pass the scalar-reference tolerance:

| Shape | Mode | Mean | Median | Parity |
| --- | --- | ---: | ---: | --- |
| global `D=56,N=4096,heads=8,batch=1,sampled_q=9` | default `run_D=56` | `0.640210 ms` | `0.629680 ms` | `bad=0/4032`, `max_abs=0.00513070542` |
| global `D=56,N=4096,heads=8,batch=1,sampled_q=9` | explicit `run_D=64` | `0.637148 ms` | `0.628723 ms` | same parity |
| window `D=56,N=196,heads=8,batch=25` | default `run_D=56` | `0.136054 ms` | `0.136132 ms` | `bad=0/2195200`, `max_abs=0.0320834816` |
| window `D=56,N=196,heads=8,batch=25` | explicit `run_D=64` | `0.199017 ms` | `0.198961 ms` | same parity |

The current D56 direct path is therefore useful mainly for the window shape; the
global shape remains effectively D64-MMA limited. Rechecking old opt-in paths
does not change the default decision:

| Probe | Global mean | Window mean | Decision |
| --- | ---: | ---: | --- |
| `GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` | `1.111663 ms` | `0.224176 ms` | still slower |
| `GGML_CUDA_ENABLE_FATTN56_INLINE_PACK_LARGE=1` | `0.827700 ms` | `0.136136 ms` | global regression; not defaultable |
| `GGML_CUDA_DISABLE_FATTN56_INLINE_PACK=1` | `0.638564 ms` | `0.195848 ms` | window regression |

This updates the priority after the MLP fusion triage: a two-stage MLP fusion
can only target roughly the `0.31 ms` unfused micro surface, while a D56 global
kernel still needs a true body-level change to beat the D64-limited path. The
next implementation attempt should not revisit native-V or large-inline-pack
toggles; it should either change the D56 global MMA/body contract or move back
to q4_1 MMQ arithmetic throughput.

Evidence:

- `outputs/fattn56-current-refresh-20260517/global.log`
- `outputs/fattn56-current-refresh-20260517/global_run64.log`
- `outputs/fattn56-current-refresh-20260517/window_default_repeat100.log`
- `outputs/fattn56-current-refresh-20260517/window_run64.log`
- `outputs/fattn56-current-refresh-20260517/global_native_v.log`
- `outputs/fattn56-current-refresh-20260517/window_native_v.log`
- `outputs/fattn56-current-refresh-20260517/global_inline_pack_large.log`
- `outputs/fattn56-current-refresh-20260517/window_inline_pack_large_repeat100.log`
- `outputs/fattn56-current-refresh-20260517/global_no_inline_pack.log`
- `outputs/fattn56-current-refresh-20260517/window_no_inline_pack.log`

### 2026-05-17: q4_1 MMQ X-cap probe and fusion priority

The q4_1 MMQ X-cap probe found a real standalone speedup on the Hiera window
QKV shape, but it is not acceptable as a default optimization yet. Raising
`GGML_CUDA_MMQ_X_MAX` to `128` changes the selected window QKV tile from
`mmq_x=80` to `mmq_x=112` and improves the standalone window QKV row from
about `0.178734 ms` to `0.144511 ms`. The two-stage MLP no-`CONT` row is
essentially unchanged around `0.285 ms` on repeated runs.

Full SAM validation blocks promoting this X-cap change: bbox results are
unchanged, but full-mask JSONL parity is not exact. The default-vs-`x128`
comparison reported `min_bbox_iou=1.0`, zero bbox deltas, and
`max_score_abs_delta=0.009155`, while `mask_hash_equal_rows=0` and the maximum
relative mask-area delta was `0.00025569809989860247`. This is small enough to
explain the tempting bbox-only timing result, but not small enough for the
current full-mask acceptance contract.

Fusion-kernel work should therefore not use this X-cap as a hidden prerequisite.
The next fusion prototype has to satisfy the default full-mask contract first,
then show a paired speed improvement against the default path.

Evidence:

- `outputs/mmq-two-stage-xcap-refresh-20260517/default_repeat_1.log`
- `outputs/mmq-two-stage-xcap-refresh-20260517/default_repeat_2.log`
- `outputs/mmq-two-stage-xcap-refresh-20260517/default_repeat_3.log`
- `outputs/mmq-window-qkv-xcap-refresh-20260517/cap_default.log`
- `outputs/mmq-window-qkv-xcap-refresh-20260517/cap128_launch.log`
- `outputs/mmq-x128-app-probe-20260517/default-vs-x128-fullmask.json`

### 2026-05-17: Fusion-kernel execution plan

The current Hiera MLP already benefits from the existing CUDA MMQ
`mul_mat + bias + GELU` epilogue fusion. The remaining unfused surface is the
consumer side of the two-stage MLP:

```text
fc1 q4_1 MMQ + bias + GELU -> f32 activation
fc2 q4_1 MMQ + bias -> f32 output
```

The Hiera graph does not contain an explicit `CONT` between GELU and `fc2`, so a
fusion prototype must beat the no-`CONT` two-stage baseline. Optimizing an
artificial materialized row is useful only as a stress test and must not be used
to justify integration into `sam2_hiera_block_forward`.

Implementation order:

1. Keep `just fusion-kernel-acceptance` green as the micro gate. It runs the
   two-stage q4_1 parity smoke, the Hiera-like no-`CONT` timing row, and the
   explicit-`CONT` stress row.
2. Prototype only opt-in ggml CUDA behavior first, guarded by an environment
   variable. The prototype must not change default SAM outputs before parity is
   proven.
3. Prefer a consumer-side improvement that reduces the fc2 input preparation or
   quantization cost. A full fc1+fc2 fused matmul is lower priority because it
   would need to keep a `1792 x cols` activation tile live across the second
   projection and would be easy to make slower than the current MMQ schedule.
4. Promote to default only after the full SAM JSONL full-mask comparison is
   exact and paired bbox/profile timing shows a repeatable win.

Acceptance criteria:

- Micro parity smoke passes with `--two-stage --check` and no nonfinite output.
- The Hiera-like no-`CONT` two-stage row improves over the current
  `~0.31 ms` baseline; otherwise the kernel is not worth app integration.
- Full SAM full-mask JSONL comparison reports exact mask hashes, bbox IoU
  `1.0`, and zero bbox deltas against the default path.
- Paired app timing improves in the same binary/configuration. Bbox-only speed
  without exact full-mask parity is insufficient.
- The change remains opt-in until all criteria above hold.

The first `just fusion-kernel-acceptance` run passed on the current build:

| Row | Mean | Median | Parity |
| --- | ---: | ---: | --- |
| no-`CONT` smoke, `cols=128` | `0.040831 ms` | `0.039716 ms` | `bad=0/57344`, no nonfinite output |
| no-`CONT` timing, `cols=4096` | `0.314742 ms` | `0.315021 ms` | micro timing target |
| explicit-`CONT` stress, `cols=4096` | `0.377380 ms` | `0.375210 ms` | stress target only |

Evidence:

- `outputs/fusion-kernel-acceptance-just-20260517/q41_two_stage_no_cont_check.log`
- `outputs/fusion-kernel-acceptance-just-20260517/q41_two_stage_no_cont_timing.log`
- `outputs/fusion-kernel-acceptance-just-20260517/q41_two_stage_cont_timing.log`

### 2026-05-17: Rejected D56 direct epilogue probe

A D56 FlashAttention probe bypassed the generic shared-memory combine/writeback
epilogue for the no-mask, no-sinks, non-fixup, `np == 1` inline-head path. The
intent was to normalize `VKQ_C` fragments from registers and store directly to
the output tensor, while leaving stream-k/fixup paths untouched.

The probe preserved standalone parity but did not produce a defaultable speed
win:

| Shape | Probe mean | Probe median | Current baseline |
| --- | ---: | ---: | --- |
| global `D=56,N=4096,heads=8,batch=1,sampled_q=9`, `run_D=56` | `0.628177 ms` in the first probe, `0.639095 ms` after narrowing | `0.627709 ms`, then `0.630380 ms` | baseline mean `0.640210 ms`, median `0.629680 ms` |
| window `D=56,N=196,heads=8,batch=25`, `run_D=56` | `0.139933 ms`, then `0.139796 ms` | `0.140164 ms`, then `0.140006 ms` | baseline mean `0.136054 ms`, median `0.136132 ms` |

The global row is at best noise-scale, while the window row regresses. Since the
SAM Hiera workload uses the window shape heavily, this direct-epilogue source
change was removed. Future D56 work should target a real body-level reduction
or a shape-specific path that is provably not selected for the window case.

Evidence:

- `outputs/fattn56-direct-epilogue-probe-20260517/global_run56.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/window_run56.log`
- `outputs/fattn56-direct-epilogue-largeonly-20260517/global_run56.log`
- `outputs/fattn56-direct-epilogue-largeonly-20260517/window_run56.log`

### 2026-05-17: Rejected MLP prequant-cache probe

The existing opt-in `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` path was checked on
the same q4_1 two-stage Hiera MLP acceptance surface. It preserves the
microbench parity smoke (`bad=0/57344`, no nonfinite output), but it is slower
than the default no-`CONT` graph:

| Path | Mean | Median | Note |
| --- | ---: | ---: | --- |
| default no-`CONT` two-stage | `0.314742 ms` | `0.315021 ms` | `just fusion-kernel-acceptance` |
| prequant cache enabled | `0.323990 ms` | `0.323823 ms` | quantizes after producer MMQ output |

Profiling shows why this is not the right fusion shape: the consumer fc2 MMQ can
use cached q8_1 input (`cached_src1=1`, about `0.112-0.116 ms` for the fc2 MMQ
event), but producing that cache still adds enough work to lose the two-stage
row. The profiler combination itself also segfaulted at process exit in this
probe, so the current opt-in cache is not a promotion candidate.

A smaller standalone GELU+DS4 side-output kernel remains exact and faster than
separate GELU plus quantization (`0.152576 ms -> 0.146752 ms` median,
`q8_diff_bytes=0`), but that only proves the safe kernel shape. To matter in the
full MLP, the DS4 side buffer must be produced inside or directly adjacent to
the producer MMQ writeback without adding a separate post-MMQ quantization pass.

Evidence:

- `outputs/fusion-prequant-cache-probe-20260517/prequant_check.log`
- `outputs/fusion-prequant-cache-probe-20260517/prequant_timing.log`
- `outputs/fusion-prequant-cache-probe-20260517/prequant_profile.stderr`
- `outputs/fusion-prequant-cache-probe-20260517/f32_batched_gelu_quant_ds4.log`

The related "turn off MMQ bias+GELU fusion, then enable prequant cache" variant
is correctness-safe but still not a speed win. On the microbench surface:

| Path | Mean | Median |
| --- | ---: | ---: |
| default static MMQ bias+GELU epilogue | `0.314742 ms` | `0.315021 ms` |
| `GGML_CUDA_DISABLE_MMQ_STATIC_EPILOGUE=1` | `0.368820 ms` | `0.366308 ms` |
| `GGML_CUDA_DISABLE_MMQ_BIAS_GELU_FUSION=1` | `0.342920 ms` | `0.342625 ms` |
| `GGML_CUDA_DISABLE_MMQ_BIAS_GELU_FUSION=1 GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` | `0.321347 ms` | `0.314617 ms` |

The combined variant recovered the microbench median, but did not beat the
default. Full SAM validation was exact against the default path
(`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, zero bbox/score/mask-area
deltas), while the single app run was slightly slower (`46 -> 47 ms/frame`).
This confirms the remaining useful MLP fusion target is not an after-the-fact
prequant cache; it needs producer-side MMQ writeback to emit the normal F32
activation and the DS4 q8_1 side buffer in the same pass, or another design that
removes work rather than moving it.

Evidence:

- `outputs/mmq-epilogue-ab-20260517/dynamic_epilogue_timing.log`
- `outputs/mmq-epilogue-ab-20260517/no_bias_gelu_fusion_timing.log`
- `outputs/mmq-epilogue-ab-20260517/no_bias_gelu_fusion_prequant_timing.log`
- `outputs/mmq-epilogue-ab-20260517/app/default-vs-no-bias-gelu-prequant.json`

An `ADD + GELU` side-output prototype was also tried to force the cached fc2
input to be produced by the fused add/unary kernel instead of by a separate
post-GELU quantization pass. With only
`GGML_CUDA_DISABLE_MMQ_BIAS_GELU_FUSION=1`, MMQ still absorbs the bias `ADD`,
so the side-output path is not selected and fc2 still runs the normal input
quantization. With `GGML_CUDA_DISABLE_MMQ_BIAS_FUSION=1`, the side-output path
is selected (`cached_src1=1` in the fc2 MMQ profile), but the two-stage timing is
even worse: `mean_ms=0.351483`, `median_ms=0.351490`. This proves that moving
the side buffer to the add/unary kernel is still not enough; disabling the
producer MMQ bias/GELU epilogue costs more than the saved fc2 quantization.

The source change was removed. A useful implementation still has to keep the
producer MMQ static bias+GELU epilogue and add side-buffer emission there, or
find a different MMQ/FATTN body-level reduction.

Evidence:

- `outputs/add-gelu-side-ds4-probe-20260517/timing.log`
- `outputs/add-gelu-side-ds4-probe-20260517/bias-off/timing.log`
- `outputs/add-gelu-side-ds4-probe-20260517/bias-off/profile.stderr`

### 2026-05-17: Fusion baseline refresh before producer-side work

The q4_1 two-stage Hiera MLP acceptance row was refreshed after rebuilding the
current CUDA tree. The default launch now selects `mmq_x=96` for both producer
and consumer rows:

```text
GGML_CUDA_PROFILE_MMQ_LAUNCH type=q4_1 mmq_x=96 mmq_y=128 stream_k=1
ncols_x=448 nrows_x=1792 ncols_dst=4096 ncols_max=4096 ntx=43 nty=14
fixup=0 shared=53632

GGML_CUDA_PROFILE_MMQ_LAUNCH type=q4_1 mmq_x=96 mmq_y=128 stream_k=1
ncols_x=1792 nrows_x=448 ncols_dst=4096 ncols_max=4096 ntx=43 nty=4
fixup=0 shared=53632
```

The refreshed `just fusion-kernel-acceptance` result is:

| Row | Mean | Median | Notes |
| --- | ---: | ---: | --- |
| no-`CONT` smoke, `cols=128` | `0.037501 ms` | `0.035597 ms` | `bad=0/57344`, no nonfinite output |
| no-`CONT` timing, `cols=4096` | `0.285754 ms` | `0.285241 ms` | new fusion baseline |
| explicit-`CONT` stress, `cols=4096` | `0.365595 ms` | `0.365643 ms` | stress target only |

The existing opt-in prequant cache was also refreshed against this tighter
baseline. It is still correctness-safe on the smoke row, but it is now a larger
loss:

| Path | Mean | Median | Notes |
| --- | ---: | ---: | --- |
| default no-`CONT` two-stage | `0.285754 ms` | `0.285241 ms` | current baseline |
| `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` | `0.324749 ms` | `0.324728 ms` | `bad=0/57344` on smoke, but slower |

An MMQ X-cap scan did not find a safer hidden prerequisite. The best values were
noise-level around the same selected `mmq_x=96` schedule:

| Cap | Mean | Median |
| ---: | ---: | ---: |
| default | `0.285242 ms` | `0.285024 ms` |
| `96` | `0.285336 ms` | `0.284989 ms` |
| `104` | `0.285822 ms` | `0.285809 ms` |
| `128` | `0.285695 ms` | `0.285490 ms` |

This tightens the fusion-kernel promotion rule: a producer-side side-buffer
prototype must beat `~0.285 ms`, not the older `~0.315 ms` row. The older
post-producer prequant and add/unary side-output probes remain rejected because
they add another pass or disable the already-profitable static MMQ bias+GELU
epilogue.

Evidence:

- `outputs/fusion-kernel-acceptance-refresh-20260517/`
- `outputs/fusion-mmq-xscan-20260517/`
- `outputs/fusion-prequant-refresh-20260517/`

A single full-app Base+ q4_1 refresh with the same 10-frame bbox/multimask
contract reported `46.4 ms/frame` (`p50=44.9`, `p95=53.4`). This is not a
PyTorch-beating result and should not be treated as an accepted app-level win;
paired repeats are still required before any default promotion.

Evidence:

- `outputs/current-baseplus-q4_1-fusion-refresh-20260517/q4_1.log`

### 2026-05-17: Rejected producer-side prequant scratch prototype

A producer-side q8_1 side-buffer prototype was implemented behind
`GGML_CUDA_ENABLE_MMQ_PRODUCER_PREQUANT_CACHE=1` to keep the existing q4_1 MMQ
`bias + GELU` epilogue and emit the cached fc2 input from the same producer
kernel. The first version wrote both the normal F32 activation and a shared
scratch tile used for `block_q8_1_mmq` quantization. It preserved the micro
smoke tolerance (`bad=0/57344`) and the following fc2 MMQ used
`cached_src1=1`, but the two-stage row stayed slow:

| Path | Mean | Median | Notes |
| --- | ---: | ---: | --- |
| producer-side scratch + F32 write | `0.323932 ms` | `0.324184 ms` | same speed class as post-producer prequant |
| producer-side scratch, skipped F32 write | `0.311381 ms` | `0.311469 ms` | better, still slower than default |
| current default | `0.285754 ms` | `0.285241 ms` | baseline to beat |

The prototype was removed because the extra shared-memory staging and
quantization work outweighed the saved consumer input quantization. Keeping it
would also add a runtime branch to the default MMQ writeback hot loop. A useful
producer-side fusion would need a register-level q8_1 emission scheme, or a
different MMQ body change, not a shared-scratch side pass.

After removal, a direct default repeat showed a warmup/power-state split even
with the same `mmq_x=96` launch profile. The first two rows were slow
(`0.314824`, `0.314080 ms` mean), then the next three settled around the faster
row (`0.287012`, `0.286282`, `0.288543 ms` mean). Therefore future fusion
promotion must use paired repeats after device warmup; a single microbench row is
not a stable enough acceptance signal.

Evidence:

- `outputs/fusion-producer-prequant-probe-20260517/producer_check.log`
- `outputs/fusion-producer-prequant-probe-20260517/producer_timing.log`
- `outputs/fusion-producer-prequant-probe-20260517/producer_profile.stderr`
- `outputs/fusion-producer-prequant-probe-20260517/skip-f32/producer_check.log`
- `outputs/fusion-producer-prequant-probe-20260517/skip-f32/producer_timing.log`
- `outputs/fusion-kernel-acceptance-after-producer-revert-20260517/`
- `outputs/fusion-baseline-repeat-after-producer-20260517/`

### 2026-05-17: Warmed Base+ q4_1 app repeat against PyTorch

The current Base+ q4_1 1024 app row was rerun after a discarded warmup run to
avoid relying on a single cold/power-state-sensitive result. All runs use the
same decoded `960x540` video, same 10-frame range, point `(315, 250)`,
bbox-only multimask tracking, and SAM input size `1024`.

| Run | Track/fr | P50 | P95 |
| --- | ---: | ---: | ---: |
| 0 | `44.7 ms` | `43.3 ms` | `51.3 ms` |
| 1 | `44.8 ms` | `43.4 ms` | `50.9 ms` |
| 2 | `45.0 ms` | `43.5 ms` | `51.4 ms` |
| 3 | `45.1 ms` | `43.4 ms` | `51.1 ms` |
| 4 | `45.3 ms` | `44.0 ms` | `51.1 ms` |

Summary: mean `44.98 ms/frame`, median `45.0`, min `44.7`, max `45.3`,
stdev `0.239`. The official PyTorch Base+ bf16 row remains
`42.6667 ms/frame`, so the warmed C++ gap is `+2.31 ms/frame`
(`PyTorch / C++ = 0.9486`). Bbox/mask determinism is stable across the repeat:
run0 vs run4 has `mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, and zero
bbox/score/mask-area deltas.

Evidence:

- `outputs/current-baseplus-q4_1-warm-repeat-20260517/summary.json`
- `outputs/current-baseplus-q4_1-warm-repeat-20260517/run0-vs-run4.json`
- `outputs/python-official-profile/base_plus_1024-final-20260516.json`

### 2026-05-17: Refreshed MMQ and node hotspot priority

A short 5-frame MMQ attribution run and a separate 3-frame CUDA node profile
were captured after the producer-side scratch prototype was removed. These runs
are synchronized/profiling-only and are not app speed rows, but they refine the
next implementation target.

Top q4_1 MMQ shapes by effective total time:

| Shape | Effective count | Quant sum | MMQ sum | Total sum | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `q4_1[1792,448] x f32[1792,4096] -> f32[448,4096]` | 79 | `2.370 ms` | `9.589 ms` | `11.959 ms` | stage-2 MLP projection body is still the largest MMQ body target |
| `q4_1[448,112] x f32[448,65536] -> f32[112,65536]` | 9 | `2.754 ms` | `1.747 ms` | `4.501 ms` | high-resolution path is quantization-heavy |
| `q4_1[896,224] x f32[896,16384] -> f32[224,16384]` | 14 | `1.927 ms` | `2.110 ms` | `4.037 ms` | high-resolution path has both quant and MMQ cost |
| `q4_1[3584,896] x f32[3584,1024] -> f32[896,1024]` | 14 | `0.223 ms` | `1.465 ms` | `1.688 ms` | deeper MLP projection body target |
| `q4_1[2048,256] x f32[2048,4096] -> f32[256,4096]` | 15 | `0.576 ms` | `1.042 ms` | `1.618 ms` | neck / projection body target |

Top synchronized Hiera node hotspots after dropping each group's max outlier:

| Op / shape | Drop-max sum | Priority |
| --- | ---: | --- |
| `MUL_MAT dst=f32[1792,4096]` | `7.037 ms` | stage-2 MLP expansion |
| `MUL_MAT dst=f32[448,4096]` | `6.421 ms` | stage-2 MLP projection |
| `MUL_MAT dst=f32[1344,196,25]` | `5.855 ms` | window qkv projection |
| `FLASH_ATTN_EXT D=56,N=196,heads=8,batch=25` | `4.695 ms` | D56 window attention |
| `FLASH_ATTN_EXT D=56,N=4096,heads=8,batch=1` | `4.355 ms` | D56 global attention |

This changes the next optimization priority slightly: q4_1 MMQ body/scheduling
is still first, but `quantize_mmq_q8_1_cuda` is now a first-class target because
the largest high-resolution q4_1 rows are quantization-heavy. Producer-side
prequant cache is not the answer; the useful target is reducing the normal src1
quantization pass itself or avoiding its work without adding a second pass.

Evidence:

- `outputs/current-baseplus-q4_1-mmq-profile-refresh-20260517/mmq_summary.json`
- `outputs/current-baseplus-q4_1-node-profile-refresh-20260517/node_hotspots.json`

### 2026-05-17: Rejected src1 q8 cache and fusion-kernel next step

An opt-in graph-local cache for MMQ `src1` q8_1 activations was tested as a
lower-risk precursor to a true fusion kernel. The first key used the device data
pointer, q4_1 consumer type, padded shape, and strides. It looked promising in
the synchronized MMQ profile (`61/111` cache hits, q8 quantization sum reduced
from `4.634 ms` to `1.611 ms` on the same 3-frame profile shape mix), and a
short app A/B appeared faster. However, full JSONL parity failed: default vs
cache had `min_bbox_iou=0.8475`, `max_bbox_delta_px=146.25`, and only
`9/10` matching mask hashes. The likely cause is allocator address reuse inside
the graph; a device pointer is not a valid tensor identity.

The key was then tightened with the tensor name and restricted to non-view-like
names. That restored exact parity across five 10-frame pairs
(`mask_hash_equal_rows=10/10`, zero bbox/score/mask deltas), but the MMQ profile
showed `0/111` cache hits. The measured app difference
(`default mean 45.94 ms/frame`, cache env mean `45.32 ms/frame`) is therefore
noise rather than a valid cache win. The src1 cache patch was removed.

After removal, `just fusion-kernel-acceptance` was rerun successfully:

| Row | Mean | Median | Result |
| --- | ---: | ---: | --- |
| no-`CONT` two-stage q4_1 MLP | `0.314625 ms` | `0.314862 ms` | acceptance baseline still passes |
| `CONT` two-stage q4_1 MLP | `0.368000 ms` | `0.366573 ms` | slower, still not the fusion target |

This confirms that the next fusion-kernel pass should not be another
graph-cache keyed by tensor metadata. It needs to remove work inside the
producer/consumer kernel schedule itself: either register-level q8_1 emission in
the existing static MMQ `bias+GELU` epilogue, or a q4_1 MMQ body variant that
loads/quantizes the activation tile directly without writing a global q8_1
intermediate. The promotion gate remains full-mask parity plus a paired
app-level Base+ q4_1 speed win; a profiler-only q8 quant reduction is
insufficient.

Evidence:

- `outputs/mmq-src1-q8-cache-probe-20260517b/default/default_mmq_summary.json`
- `outputs/mmq-src1-q8-cache-probe-20260517b/cache_mmq_summary.json`
- `outputs/mmq-src1-q8-cache-app-ab-20260517/default0-vs-cache0.json`
- `outputs/mmq-src1-q8-cache-app-ab-20260517c/default0-vs-cache0.json`
- `outputs/mmq-src1-q8-cache-app-ab-20260517c/default4-vs-cache4.json`
- `outputs/mmq-src1-q8-cache-safe-profile-20260517/cache_mmq_summary.json`
- `outputs/fusion-kernel-acceptance-src1-cache-reverted-20260517/`

### 2026-05-17: Accepted high-column q4_1 MMQ quantize kernel

The next accepted CUDA change is a narrower `quantize_mmq_q8_1_cuda` variant for
q4_1/DS4 activation quantization when there are many columns
(`ids == nullptr`, `type_src0 == q4_1`, `ne1 >= 16384`). The old generic kernel
launches one CTA per column and covers up to 512 padded rows in that CTA. The
new high-column path uses `4` warps per CTA by default: each warp owns one column and one
`block_q8_1_mmq` q-block. This keeps the q4_1 DS4 layout and rounding contract
identical while reducing CTA scheduling overhead for the high-resolution rows
that were quantization-heavy in the MMQ profile. It can be disabled with
`GGML_CUDA_DISABLE_MMQ_Q8_1_WARP_COL_QUANT=1`; `GGML_CUDA_MMQ_Q8_1_WARP_COLS`
can be set to `4`, `8`, or `16` for diagnostics.

Microbench results:

| Shape | Old/default-disabled mean | New default mean | Decision |
| --- | ---: | ---: | --- |
| two-stage MLP `448 -> 1792 -> 448`, cols `4096` | `~0.2855 ms` | not used by default gate | avoid; this shape was slower with the new kernel |
| high-res `448 -> 112`, cols `65536` | `0.4866 ms` | `0.4693 ms` | useful; this is the target bucket |

The high-res column sweep was essentially flat across `4/8/16` columns per CTA,
but app-level A/B favored `4`: `GGML_CUDA_MMQ_Q8_1_WARP_COLS=4` averaged
`45.0 ms/frame` over three bbox-only runs, while the previous default `8`
averaged `45.53 ms/frame`; the two settings were bit-identical at the tracking
JSONL level.

App-level Base+ q4_1 bbox-only A/B over five paired runs:

| Mode | Track/fr rows | Mean | Median | Stdev |
| --- | --- | ---: | ---: | ---: |
| disabled | `45.6, 45.1, 45.8, 45.1, 45.2` | `45.36 ms` | `45.2 ms` | `0.321` |
| default | `45.4, 45.1, 45.3, 45.2, 45.1` | `45.22 ms` | `45.2 ms` | `0.130` |

The paired mean improvement is small (`0.14 ms/frame` in the five-run default-8
probe, then `0.33 ms/frame` in a three-run default-4 confirmation), but parity
was exact for all pairs: `min_bbox_iou=1.0`, `max_bbox_delta_px=0`,
`max_score_abs_delta=0`, and `mask_hash_equal_rows=10/10`. A full-mask
single-pair check with default-4 also matched exactly, but timing was noisy
(`49.6 -> 50.7 ms/frame`), so full-mask speed is not used as a claimed win.

This is an incremental accepted cleanup, not the PyTorch-closing optimization.
The remaining gap is still too large for high-column quantization alone; the next
root target remains q4_1 MMQ body scheduling/arithmetic or D56 FlashAttention
body work.

Evidence:

- `outputs/mmq-warp-col-quant-20260517/highres_default_*.log`
- `outputs/mmq-warp-col-quant-20260517/highres_warp_*.log`
- `outputs/mmq-warp-col-quant-default-ab-20260517/`
- `outputs/mmq-warp-col-quant-fullmask-20260517/disabled-vs-default.json`
- `outputs/mmq-warp-col-sweep-20260517/`
- `outputs/mmq-warp-col-sweep-app-20260517/`
- `outputs/mmq-warp-col-default4-ab-20260517/`
- `outputs/mmq-warp-col-default4-fullmask-20260517/disabled-vs-default.json`

The short synchronized MMQ profile after switching the default to `4` columns
per CTA shows the remaining profile shape:

| Shape | Quant sum | MMQ sum | Total sum | Notes |
| --- | ---: | ---: | ---: | --- |
| `q4_1[1792,448] x f32[1792,4096] -> f32[448,4096]` | `1.423 ms` | `5.808 ms` | `7.232 ms` | main MLP body target; not handled by high-column quant gate |
| `q4_1[448,112] x f32[448,65536] -> f32[112,65536]` | `1.468 ms` | `0.975 ms` | `2.443 ms` | high-column quant target improved but still visible |
| `q4_1[896,224] x f32[896,16384] -> f32[224,16384]` | `1.136 ms` | `1.248 ms` | `2.383 ms` | also uses high-column gate, but body cost is comparable |

Total effective profiled MMQ time is now `14.770 ms`, split into `4.791 ms`
quantization and `9.979 ms` MMQ body. This confirms the high-column quantize
kernel is only an incremental cleanup; the next larger win must reduce q4_1 MMQ
body time or D56 FlashAttention body time.

Evidence:

- `outputs/current-baseplus-q4_1-mmq-profile-after-warpcol4-20260517/mmq_summary.json`

### 2026-05-17: Rejected MLP side-output fusion probes

A bounded fusion probe tested a narrower version of the side-output idea: run
the producer MMQ without its static bias+GELU epilogue, then apply bias+GELU and
emit the q8_1 DS4 side buffer in a single post-MMQ kernel. This keeps the
existing f32 output tensor updated and lets the following fc2 MMQ consume the
cached q8_1 input.

The probe is correctness-safe on the small two-stage q4_1 smoke row
(`bad=0/57344`, no nonfinite output), but it is not a promotion candidate. It
moves work out of the producer epilogue instead of removing enough work, and the
two-stage Hiera MLP microbench is slower than both the default graph and the
existing prequant-cache diagnostic path:

| Path | Mean | Median | Notes |
| --- | ---: | ---: | --- |
| default, no prequant cache | `0.285395 ms` | `0.285131 ms` | current baseline |
| `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` | `0.324682 ms` | `0.324816 ms` | existing opt-in cache path, measured after making the new probe explicit opt-in |
| `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1 GGML_CUDA_ENABLE_MMQ_PREQUANT_BIAS_GELU_FUSION=1` | `0.355406 ms` | `0.355485 ms` | rejected probe |

A second producer-side writeback probe then tried to preserve the producer MMQ
static bias+GELU epilogue and generate the side buffer inside the same CUDA
kernel from the f32 tile output. That probe compiled, but the smoke row shifted
too much (`max_abs=3.01561618`, `mean_abs=0.592602505`) and was also much slower
than default on the small check row (`0.078713 ms` vs about `0.040563 ms`). The
source changes for both probes were removed; keeping broken or slower opt-ins
would make the CUDA path harder to reason about without moving the target row.

This confirms the earlier direction: an acceptable fusion kernel must either
generate the side buffer directly from the producer MMQ accumulator before losing
the per-tile values, or reduce MMQ/FATTN body work directly. A post-MMQ fusion
pass is still an extra pass over the producer output and does not close the
PyTorch gap.

Evidence:

- `outputs/prequant-bias-gelu-fusion-ab-20260517/default_timing.log`
- `outputs/prequant-bias-gelu-fusion-ab-20260517/fused_timing.log`
- `outputs/prequant-bias-gelu-fusion-ab-20260517/optin-check/`
- `outputs/mmq-side-writeback-fusion-20260517/side_writeback_check_fence.log`

### 2026-05-17: Rejected q4_1 MLP expansion-only `mmq_x=112` gate

The current q4_1 ptxas refresh shows an odd split: the hot `mmq_x=96/80`
instantiations are still register-capped at `255`, while some unused sizes such
as `120/104/88/72/56` compile with far fewer registers and no spills. This was
not enough by itself to justify a schedule change, so a shape-limited probe
tested `mmq_x=112` only for the 4096-column MLP expansion row
(`nrows_x >= 1024`) while keeping the following projection row at `mmq_x=96`.

The standalone expansion row looked slightly faster, but the two-stage Hiera MLP
row was worse:

| Path | Producer launch | Consumer launch | Mean | Median |
| --- | --- | --- | ---: | ---: |
| default | `mmq_x=96`, no fixup | `mmq_x=96`, no fixup | `0.289868 ms` | `0.288455 ms` |
| expansion-only gate | `mmq_x=112`, no fixup | `mmq_x=96`, no fixup | `0.317004 ms` | `0.316888 ms` |

The gate was removed. The lower ptxas register count of nearby instantiations is
not a reliable proxy for the actual two-stage Hiera row; schedule changes still
need full producer+consumer microbench and app-level parity/performance proof.

Evidence:

- `outputs/mmq-ptxas-q4_1-refresh-20260517/q4_1-summary.json`
- `outputs/mmq-q41-lowreg-xcap-refresh-20260517/`
- `outputs/mmq-q41-exp-x112-gate-20260517/`

### 2026-05-17: Prequant-cache fusion reliability refresh

The existing opt-in `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` path was refreshed
as the next fusion-kernel candidate. It is still not defaultable, but two
maintenance fixes were kept so the path remains measurable:

- the producer-side prequant buffer no longer pays a full `cudaMemsetAsync`
  before launching `quantize_mmq_q8_1_cuda`, because that quantizer writes the
  padded q8_1 blocks it later consumes;
- `ggml_backend_cuda_context` now clears `mmq_prequant_cache` in its destructor
  before CUDA pools and streams are destroyed. This fixes the profile-mode
  process-exit crash seen with `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1
  GGML_CUDA_PROFILE_MMQ=1`.

The micro parity smoke remains unchanged:

| Check | Result |
| --- | --- |
| `cols=128`, q4_1 two-stage, prequant cache enabled | `bad=0/57344`, `max_abs=0.207466185`, no nonfinite output |
| profile-mode exit | `status=0` after the destructor lifetime fix |

The speed result is not enough to promote the cache path. In a short same-build
A/B after the fixes, the stable default rows were around `0.285-0.286 ms` while
prequant-cache rows were around `0.287-0.299 ms`. The cache removes the fc2
consumer-side quantization event (`cached_src1=1`, `quant_ms` about
`0.002 ms`), but the producer-side q8_1 generation still costs enough that the
whole two-stage MLP does not improve.

Decision: keep the lifetime and redundant-memset fixes, keep the cache opt-in,
and do not promote it as a fusion-kernel optimization. A production win still
needs a true producer writeback-side q8_1 generation or a larger two-stage MLP
body change; a post-producer cache remains one extra pass over the activation.

Evidence:

- `outputs/prequant-cache-lifetime-fix-20260517/check.log`
- `outputs/prequant-cache-lifetime-fix-20260517/profile.stderr`
- `outputs/prequant-cache-lifetime-fix-20260517/default_*.log`
- `outputs/prequant-cache-lifetime-fix-20260517/prequant_*.log`

### 2026-05-17: Rejected q4_1 MMQ `ITER_K=512` probe

A q4_1-only Blackwell probe widened the MMQ K iteration from `256` to `512`.
The motivation was structural rather than a selector tweak: the hot q4_1 MLP
rows spend most time in the MMQ body, so reducing K-loop overhead looked like a
possible body-level win.

The microbench did show a large speedup:

| Row | Baseline median | `ITER_K=512` median |
| --- | ---: | ---: |
| q4_1 two-stage MLP, `cols=4096` | about `0.315 ms` | `0.220522 ms` |

However, this is not a valid optimization. The small CPU-vs-CUDA smoke still
stayed under the broad micro tolerance (`bad=0/57344`), but the absolute error
increased (`max_abs=0.667939246`, `mean_abs=0.127594055`). Full SAM validation
then failed badly against the previous accepted q4_1 JSONL: `0 / 10` mask
hashes matched, `min_bbox_iou=0.0820289280760848`, and the maximum bbox delta
was `510 px`. The app speed row looked near the PyTorch target, but the output
was semantically wrong, so the source change was removed.

Decision: keep `MMQ_ITER_K=256` for q4_1. Future MMQ body work must preserve
the existing accumulation contract or add a narrower parity-proven body; K-loop
widening is not acceptable even if the standalone timing is attractive.

Evidence:

- `outputs/mmq-q41-iterk512-probe-20260517/check.log`
- `outputs/mmq-q41-iterk512-probe-20260517/two_stage.log`
- `outputs/mmq-q41-iterk512-app-20260517/iterk512.log`
- `outputs/mmq-q41-iterk512-app-20260517/baseline-run4-vs-iterk512.json`

### 2026-05-17: Fusion-kernel direction recheck

The viable fusion-kernel direction is now narrower than the earlier q8 side
buffer idea. The following variants have already been rejected or kept only as
diagnostic plumbing:

- post-producer prequant cache;
- `ADD/GELU` side-output after disabling the producer MMQ static epilogue;
- producer-side shared-scratch q8 side buffer;
- q4_1 metadata/load bookkeeping fusion that did not survive paired app timing.

The next fusion attempt should therefore not be another pass over the producer
activation. It must either:

1. fuse the two q4_1 MLP matmuls at tile level and avoid materializing or
   requantizing the hidden activation, or
2. reduce q4_1 MMQ/FATTN body work directly and treat that as the kernel-level
   fusion path.

The smallest safe integration boundary for a true two-stage MLP prototype is
`examples/sam3_mmq_bench` first, not the full graph executor. The benchmark
already constructs the exact synthetic `q4_1 fc1 + bias + GELU -> q4_1 fc2`
shape and compares against a CPU graph reference. Only after that prototype
beats the current two-kernel path with parity should the graph matcher in
`ggml-cuda.cu` learn a production fused op. This avoids adding another unstable
opt-in to the normal CUDA graph path before the kernel has a measurable win.

A literal one-kernel `fc1+fc2` fusion is not currently the best first
implementation target. `fc2` needs the complete `1792`-wide GELU hidden vector
for each column before any output row is final. The current MMQ kernels tile
hidden rows and output columns independently (`mmq_y` rows by `mmq_x` columns),
so a single CTA can only see a small slice of the hidden activation. A fused
kernel would therefore need either a global synchronization between hidden
chunks, a large persistent per-column hidden buffer, or atomic/partial
accumulation into the `448` fc2 outputs. Those choices are more likely to lose
the current MMQ tensor-core/DP4A schedule than to remove enough work. The
near-term production target remains narrower: reduce the normal q4_1 MMQ body
or generate a consumer-ready representation without an extra pass over the
producer output.

A new `just fusion-kernel-paired` recipe records warm-device paired rows for the
current two-stage baseline and the existing prequant-cache diagnostic path. The
first current run still shows a power/process warmup split:

| Path | Mean rows | Mean all rows | Warm rows |
| --- | --- | ---: | --- |
| default | `0.314362, 0.285910, 0.285559 ms` | `0.295277 ms` | `~0.2857 ms` |
| prequant cache | `0.300545, 0.287018, 0.287574 ms` | `0.291712 ms` | `~0.2873 ms` |

This confirms the promotion rule: a future fusion kernel must beat the warm
default row, not merely hide in first-run noise. For the current microbench, a
candidate should target at least `10%` median improvement over the warm default
with `bad=0` under `--check` before spending app-level validation time. For the
full app, the candidate still needs exact full-mask JSONL parity against the
accepted q4_1 baseline and a paired bbox/full-mask speed win large enough to
close roughly `2.3-2.6 ms/frame`.

Evidence:

- `outputs/fusion-kernel-current-recheck-20260517/`
- `outputs/fusion-kernel-paired-current-20260517/`

### 2026-05-17: Rejected q4_1 MMQ x-side pointer cleanup

A very small q4_1 DP4A body cleanup was tested after the fusion-kernel recheck:
precompute the x-side `qs` pointer and `dm` value before calling
`vec_dot_q4_1_q8_1_mmq_direct`. This did not change the dot math, tile schedule,
or output layout, and the small two-stage CPU-vs-CUDA check stayed within the
existing tolerance (`bad=0/57344`, `max_abs=0.207466185`).

The paired warm-device result was not useful:

| Path | Mean rows | Mean |
| --- | --- | ---: |
| x-side pointer cleanup, default | `0.287447, 0.290262, 0.288311 ms` | `0.288673 ms` |
| x-side pointer cleanup, prequant cache | `0.290903, 0.290545, 0.289428 ms` | `0.290292 ms` |

This is slower than the immediately preceding warm default row of about
`0.2857 ms`, so the source hunk was removed. This confirms that simple
address-expression cleanup is below the threshold for the current gap; the next
MMQ work needs a larger scheduling/body change or a real two-stage MLP tile
fusion.

Evidence:

- `outputs/mmq-q41-xptr-probe-20260517/`

### 2026-05-17: Rejected q4_1 `MMQ_X_MAX=104` selector probe

The ptxas summary showed lower register counts for some q4_1 `mmq_x` template
instantiations, so `GGML_CUDA_MMQ_X_MAX=104` was checked as a narrow selector
probe before spending source changes on it. It does not change the hot launch
selection:

| Shape | Selected `mmq_x` with cap 104 | Timing row |
| --- | ---: | ---: |
| two-stage MLP `448 -> 1792 -> 448`, cols `4096` | `96` for both MMQs | `0.314849 ms` mean |
| window qkv `448 -> 1344`, cols `196`, channels `25` | `80` | `0.178713 ms` mean |

Because the selector does not move either target shape, there is no follow-up
source change to make here. Future selector work should first prove that the hot
launch actually changes and still preserves full-mask parity; otherwise it is
only another no-op environment probe.

Evidence:

- `outputs/mmq-x104-launch-probe-20260517/launch_summary.json`
- `outputs/mmq-x104-launch-probe-20260517/two_stage.log`
- `outputs/mmq-x104-launch-probe-20260517/window_qkv.log`

### 2026-05-17: Paired fusion microbench summarizer

`scripts/summarize_fusion_paired.py` was added so future fusion probes do not
depend on manual log reading. `just fusion-kernel-paired` now writes
`summary_drop_first.json` after collecting the default and prequant-cache rows.
The default `--drop-first` behavior in the recipe matches the observed
process/power warmup split in the current laptop environment.

The current baseline summary confirms the prequant-cache diagnostic path is not
a speed win after warmup:

| Path | Used mean-of-means |
| --- | ---: |
| default | `0.2857345 ms` |
| prequant cache | `0.287296 ms` |

The rejected x-side pointer cleanup summary likewise shows no useful speedup:
default `0.288673333 ms`, prequant `0.290292 ms`. This gives the next fusion
prototype a concrete micro acceptance target: beat the current warm default,
not just the first process row.

Evidence:

- `scripts/summarize_fusion_paired.py`
- `outputs/fusion-kernel-paired-current-20260517/paired_summary_drop_first.json`
- `outputs/mmq-q41-xptr-probe-20260517/paired_summary.json`

### 2026-05-17: Bias+GELU+q8_1 fusion kernel microbench

The next fusion direction is to prove the local producer-side kernel before
wiring it into the graph/MMQ path. `sam3_f32_batched_bench --gelu-quant-ds4`
now supports an optional bias vector, so it can measure the Hiera MLP fc1 output
shape that would feed the following q4_1 fc2 MMQ: rows `1792`, cols `4096`.

The fused kernel writes the GELU output and the q8_1 DS4 prequantized side
buffer in one pass. It is byte-exact against the separate `GELU` then q8_1 DS4
quantize path:

| Shape | Path | Separate median | Fused median | Delta | Correctness |
| --- | --- | ---: | ---: | ---: | --- |
| `1792 x 4096` | bias + GELU + q8_1 DS4 | `0.161472 ms` | `0.142176 ms` | `11.95%` faster | `max_abs_output_delta=0`, `q8_diff_bytes=0` |
| `1792 x 4096` | GELU + q8_1 DS4 | `0.155040 ms` | `0.145248 ms` | `6.32%` faster | `max_abs_output_delta=0`, `q8_diff_bytes=0` |

This is not yet an accepted end-to-end SAM speedup. The earlier prequant-cache
graph path was slower because it added a producer-side quantize pass after the
MMQ static epilogue. This microbench only proves that the local fused
bias/GELU/quant pass is profitable and exact. The next implementation step must
connect it to the actual producer/consumer pair without losing the existing
MMQ scheduling win, then run paired full graph timing and full-mask parity.

Evidence:

- `examples/f32_batched_custom.cu`
- `examples/f32_batched_bench.cpp`
- `just gelu-quant-fusion-bench`
- `outputs/gelu-quant-fusion-rerun-20260517/bias_gelu_quant_ds4.log`
- `outputs/gelu-quant-fusion-rerun-20260517/gelu_quant_ds4.log`

### 2026-05-17: Rejected graph-level MMQ GELU prequant fusion

The microbench-proven `GELU + q8_1 DS4` kernel was then wired into the actual
graph as an opt-in probe: producer MMQ wrote the `MMQ + bias` intermediate, a
single fused kernel wrote both the GELU output and prequantized q8_1 DS4 cache,
and the following q4_1 `fc2` MMQ consumed the cached source. This avoided the
unsafe MMQ global-`dst` epilogue side-output approach and kept tracking parity
exact:

| Metric | Value |
| --- | ---: |
| Rows compared | `10` |
| Min bbox IoU | `1.0` |
| Max bbox delta px | `0.0` |
| Max score abs delta | `0.0` |
| Mask hash equal rows | `10 / 10` |

The graph-level timing was worse, so the probe was removed. Splitting the
existing `MMQ + bias + GELU` static epilogue into `MMQ + bias` plus an extra
fused GELU/quantize launch saved the consumer quantization rows, but it lost
more in the producer path and launch overhead:

| Path | Track/fr runs (ms) | Mean |
| --- | --- | ---: |
| Default MMQ static GELU epilogue | `47.5, 47.7, 47.6, 47.7, 48.2` | `47.74` |
| Graph-level GELU prequant fusion | `48.3, 49.4, 48.6, 49.3, 48.8` | `48.88` |

MMQ profiling confirmed the consumer did use `cached_src1=1`, so this is not a
matcher failure. The rejected result narrows the viable fusion path further:
producer-side q8_1 must be created inside the MMQ epilogue without adding a
new launch or giving up the existing static GELU epilogue, or the work should
move to a different hotspot such as D56 FlashAttention.

Evidence:

- `outputs/mmq-gelu-prequant-fusion-20260517/parity/compare.json`
- `outputs/mmq-gelu-prequant-fusion-20260517/speed/`
- `outputs/mmq-gelu-prequant-fusion-20260517/profile/fused.stderr`

### 2026-05-17: Current hotspot refresh after rejected fusion probes

After removing the unsafe MMQ epilogue side-output code and rejecting the
graph-level `MMQ+bias -> GELU+q8_1` split, the two remaining CUDA surfaces were
re-profiled on Base+ q4_1 with 5 bbox-only frames and first rows dropped per
shape. These profiler runs synchronize CUDA events and are for attribution, not
normal app timing.

FATTN56 is still almost entirely body time:

| Surface | Pack sum | MMA/body sum | Slice sum | Total |
| --- | ---: | ---: | ---: | ---: |
| All D56 FATTN rows | `0.383712 ms` | `25.771200 ms` | `0.042752 ms` | `26.197664 ms` |
| Window `Q/K/V[56,196,8,25]` | `0.000000 ms` | `8.301632 ms` | `0.000000 ms` | `8.301632 ms` |
| Global `Q/K/V[56,4096,8,1]` | `0.383712 ms` | `7.716096 ms` | `0.042752 ms` | `8.142560 ms` |

MMQ is the comparable remaining surface:

| Surface | Quant sum | MMQ body sum | Total |
| --- | ---: | ---: | ---: |
| All profiled q4_1 MMQ rows | `8.367008 ms` | `16.916064 ms` | `25.283072 ms` |
| `1792 x 4096 -> 448` MLP fc2 bucket | `2.335296 ms` | `9.532000 ms` | `11.867296 ms` |
| `448 x 65536 -> 112` high-res bucket | `2.643136 ms` | `1.726784 ms` | `4.369920 ms` |
| `896 x 16384 -> 224` bucket | `1.967712 ms` | `2.118720 ms` | `4.086432 ms` |

The updated priority remains body-level work, not wrapper toggles:

1. q4_1 MMQ body/scheduling for the `1792 x 4096 -> 448` MLP bucket.
2. D56 FlashAttention body for the window/global rows.
3. q8_1 activation quantization only if it does not split the existing MMQ
   static GELU epilogue or add another launch.

Do not repeat the already rejected shallow probes here: D56 direct epilogue,
D56 inline-large/native-V/wrapper toggles, q4_1 `ITER_K=512`, global
`MMQ_ITER_K=128/448/512`, medium-column quantize threshold, or graph-level
GELU+prequant split.

Evidence:

- `outputs/fattn56-current-after-fusion-reject-20260517/summary.json`
- `outputs/mmq-current-after-fusion-reject-20260517/summary.json`

The same current binary was also rerun without profiling for five 30-frame
bbox-only rows. It remains slower than the official PyTorch bf16 baseline:

| Run set | Track/fr rows (ms) | Mean | Median |
| --- | --- | ---: | ---: |
| Current Base+ q4_1 CUDA, 30-frame bbox-only | `48.1, 47.7, 47.8, 47.7, 47.7` | `47.80` | `47.70` |
| Official PyTorch bf16 baseline | n/a | `42.666687668922044` | n/a |

This is not used to replace the earlier matched 10-frame audit number because
the run length and benchmark mode differ, but it reconfirms that the goal is
not complete after the rejected fusion probes.

Evidence:

- `outputs/current-speed-after-fusion-reject-20260517/`

### 2026-05-17: Official quality matrix audit

`scripts/summarize_sam2_official_quality_matrix.py` now consolidates the
existing C++ vs official SAM2 quality summaries. This closes the audit gap where
official quality evidence existed but was scattered across many probe
directories. The current matrix contains `34` valid summary rows with metadata
validation status `passed`.

Best current Base+ rows by precision and image size:

| Precision / size | Mode | Mean mask IoU | Min mask IoU | Evidence |
| --- | --- | ---: | ---: | --- |
| `f16 / 512` | default, forced candidate 0 | `0.935075862` | `0.865417272` | `outputs/sam2-official-quality-current-f16-512-force-cand0-20260517/official/summary.json` |
| `f16 / 1024` | multimask | `0.987351866` | `0.984744785` | `outputs/sam2-official-quality-current-f16-1024-multimask-torch28-20260515/official/summary.json` |
| `f32 / 1024` | default, PyTorch fp32 diagnostic | `0.989099755` | `0.987320773` | `outputs/sam2-official-quality-current-f32-1024-prop-selection-diagnostics-pytorch-fp32-20260515/official/summary.json` |
| `q4_0 / 1024` | multimask | `0.979506889` | `0.975674080` | `outputs/sam2-official-quality-current-q4_0-1024-multimask-20260515/summary.json` |
| `q4_1 / 1024` | default | `0.993274380` | `0.987083117` | `outputs/sam2-official-quality-current-q4_1-1024-final-20260516/summary.json` |
| `q8_0 / 1024` | default | `0.995070921` | `0.993409879` | `outputs/sam2-official-quality-current-q8_0-1024-final-20260516/summary.json` |

This means the Base+ fallback quality requirement is now covered for the
current 1024 target across f16/f32/q8_0/q4_1/q4_0 evidence. It does not change
the speed audit: the matched q4_1/q8_0 C++ rows are still slower than official
PyTorch, so the goal remains open.

Evidence:

- `scripts/summarize_sam2_official_quality_matrix.py`
- `outputs/sam2-official-quality-current-matrix-20260517/summary.json`

### 2026-05-17: Goal completion audit

Objective restated as concrete deliverables:

1. Detailed Hiera encode profiling exists and identifies the remaining CUDA
   bottlenecks.
2. Base+ fallback quality is checked against the official Python implementation
   under a matched input contract.
3. Speed improvement candidates are prioritized from measured evidence.
4. C++ CUDA becomes faster than official PyTorch on the matched Base+ tracking
   contract.

Current audit:

| Requirement | Status | Evidence |
| --- | --- | --- |
| Detailed Hiera encode profiling | Met | normal q4_1 stage profile shows steady `hiera_encode` around `30.55..33.96 ms`; node/MMQ profiles rank q4_1 MMQ MLP, q4_1 window qkv, and D56 FlashAttention as the main CUDA targets |
| Official Python comparison | Met | latest matched Base+ q4_1 comparison uses the same decoded `960x540` frames, same 10-frame range, same prompt, and SAM input size `1024` |
| Base+ fallback quality gate | Met | official quality matrix now covers current Base+ f16/f32/q8_0/q4_1/q4_0 evidence; q4_1 1024 mean mask IoU is `0.993274380`, q8_0 1024 is `0.995070921`, and all matrix rows report metadata validation `passed` |
| Prioritized speed work | Met | rejected probes now cover X-cap, D56 direct epilogue, prequant cache, add/unary side-output, stream-k disable, load-tile metadata fusion, MMQ ids-writeback skip, and unsafe src1 q8 cache; accepted work now includes the high-column q4_1 quantize kernel, but remaining priority is still q4_1 MMQ body/scheduling or D56 FATTN body |
| Faster than PyTorch | Missing | latest accepted high-column quantize A/B default-4 confirmation is C++ mean `45.3 ms/frame`, and the earlier warmed repeat was `44.98 ms/frame`; both remain slower than official PyTorch bf16 `42.6667 ms/frame` |

The active goal is therefore not complete. Passing build/parity gates is not
enough to close it because the explicit speed target remains unmet. The next
accepted optimization must improve the matched Base+ q4_1 row by roughly
`2.3-2.6 ms/frame` while preserving full-mask JSONL parity.

Prompt-to-artifact checklist:

| Prompt item | Current artifact | Coverage |
| --- | --- | --- |
| Hiera encode detailed profile | `outputs/stage-profile-baseplus-q4_1-current-1024-20260516/summary.json`, `outputs/fusion-direction-profile-20260517/q4_1_hiera_hotspots.json` | Covers stage timing and synchronized node hotspot ranking |
| Python official comparison | `outputs/current-baseplus-q8q41-vs-python-refresh-20260517/q4_1/summary.json`, `outputs/python-official-profile/base_plus_1024-final-20260516.json` | Covers matched speed contract and official PyTorch profiling |
| Base+ fallback quality | `outputs/sam2-official-quality-current-matrix-20260517/summary.json` plus full-mask JSONL comparisons listed in rejected probe sections | Covers official Python quality matrix for current Base+ evidence and probe parity |
| Candidate prioritization | this document's accepted/rejected 2026-05-17 probe sections | Covers measured prioritization and rollback decisions |
| PyTorch以上の高速化 | not present | Missing; no completion/update-goal action is valid yet |

Next implementation target:

- Primary: q4_1 MMQ body/scheduling for the hot Hiera MLP and window qkv shapes.
  Side-output/cache variants are only worth revisiting if they keep the producer
  MMQ static bias+GELU epilogue and remove work in the same pass.
- Secondary: D56 FlashAttention body-level change. Wrapper/epilogue-only probes
  have not moved the normal Hiera row enough.

### 2026-05-17: MMQ static special activation epilogue probe

The fusion-kernel direction was extended to test whether Hiera's
`ggml_gelu_erf` path benefits from a dedicated MMQ static epilogue. Existing
MMQ fusion already handles `MMQ + bias + activation`, but only `GELU` had a
default static epilogue launch. The new probe adds compiled
`MMQ_ACT_GELU_ERF` and `MMQ_ACT_GELU_QUICK` variants and gates their selection
behind `GGML_CUDA_ENABLE_MMQ_STATIC_SPECIAL_EPILOGUE=1`.

Parity against the default dynamic activation path is exact on the current
Base+ q4_1 1024 bbox contract:

| Metric | Value |
| --- | ---: |
| Rows compared | `10` |
| Min bbox IoU | `1.0` |
| Max bbox delta px | `0.0` |
| Max score abs delta | `0.0` |
| Mask hash equal rows | `10 / 10` |

Performance A/B on the same contract:

| Mode | Track/fr runs (ms) | Mean (ms) | Median (ms) | P50 median (ms) | P95 median (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Default dynamic activation | `46.2, 45.7, 45.3` | `45.733` | `45.7` | `43.9` | `52.4` |
| Opt-in static special activation | `45.2, 45.3, 46.1` | `45.533` | `45.3` | `43.6` | `51.8` |

The measured mean delta is only `0.44%` in favor of the opt-in static path and
is within normal run noise, so this is not promoted to default. It remains as an
experimental switch for future fusion probes where static ERF/Quick epilogues
may combine with a larger producer/consumer fusion.

Evidence:

- `ggml/src/ggml-cuda/mmq.cuh`
- `outputs/current-q41-static-special-epilogue-parity-20260517/compare.json`
- `outputs/current-q41-static-special-epilogue-ab-20260517/summary.json`

### 2026-05-17: Rejected medium-column q4_1 MMQ quantize threshold

The accepted q4_1 warp-column q8_1 quantizer is currently limited to
`ne1 >= 16384`. A follow-up probe temporarily made that threshold configurable
and tested `GGML_CUDA_MMQ_Q8_1_WARP_COL_QUANT_MIN=4096` so the stage-2 Hiera
MLP projection shape (`src1=f32[1792,4096]`) would also use the warp-column
kernel.

The output contract stayed exact:

| Metric | Value |
| --- | ---: |
| Rows compared | `10` |
| Min bbox IoU | `1.0` |
| Max bbox delta px | `0.0` |
| Max score abs delta | `0.0` |
| Mask hash equal rows | `10 / 10` |

The synchronized MMQ profile looked better in isolation, with drop-first MMQ
profile totals moving from `quant=4.790656 ms`, `mmq=10.005152 ms`,
`total=14.795808 ms` to `quant=4.642112 ms`, `mmq=9.249536 ms`,
`total=13.891648 ms`. However, the normal bbox-only application A/B regressed:

| Mode | Track/fr runs (ms) | Mean (ms) | Median (ms) | Stdev |
| --- | --- | ---: | ---: | ---: |
| Default threshold `16384` | `44.8, 45.3, 45.1, 46.3, 45.2` | `45.34` | `45.2` | `0.568` |
| Temporary threshold `4096` | `45.0, 46.2, 46.0, 45.8, 45.6` | `45.72` | `45.8` | `0.460` |

This is therefore rejected and the source was returned to the fixed
`ne1 >= 16384` gate. The profile-only improvement is not enough evidence; the
normal run is the deciding metric.

Evidence:

- `outputs/current-q41-warpcol-min4096-profile-20260517/default/mmq_summary.json`
- `outputs/current-q41-warpcol-min4096-profile-20260517/min4096/mmq_summary.json`
- `outputs/current-q41-warpcol-min4096-parity-20260517/compare.json`
- `outputs/current-q41-warpcol-min4096-ab-20260517/summary.json`

### 2026-05-17: Rejected D56 no-fixup direct FATTN epilogue

A body-level D56 FlashAttention probe tried to skip the generic shared-memory
VKQ combine/writeback path for the narrow case where no combination is needed:
`no_mask_no_sinks && !needs_fixup && !is_fixup && np == 1` on the Turing/Ampere
MMA path. The probe wrote normalized `VKQ_C` register fragments directly to the
`DV_DST=56` destination and left fixup, mask/sink, `cols_per_warp==8`, and
non-Turing paths unchanged.

Correctness checks passed:

| Check | Result |
| --- | ---: |
| Standalone D56 global parity, `N=4096 heads=8 batch=1` | `bad=0/4032`, nonfinite `0` |
| Standalone D56 window parity, `N=196 heads=8 batch=25` | `bad=0/100800`, nonfinite `0` |
| App JSONL vs previous default | `10 / 10` mask hashes equal, bbox/score deltas `0` |

However, normal application speed regressed against the immediately preceding
default baseline:

| Mode | Track/fr runs (ms) | Mean (ms) | Median (ms) | Stdev |
| --- | --- | ---: | ---: | ---: |
| Previous default | `44.8, 45.3, 45.1, 46.3, 45.2` | `45.34` | `45.2` | `0.568` |
| Direct no-fixup epilogue probe | `47.0, 46.5, 45.8, 44.6, 45.1` | `45.80` | `45.8` | `0.982` |

The source was reverted. The shared-memory epilogue comment in ggml is
consistent with the measurement here: the direct register-to-global writes are
too small/scattered to beat the coalesced shared-memory staging path on this
workload. Future D56 work should reduce MMA body work or avoid D56->D64 packing,
not only bypass the existing no-fixup combine epilogue.

Evidence:

- `outputs/fattn56-direct-epilogue-probe-20260517/global.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/window.log`
- `outputs/fattn56-direct-epilogue-app-20260517/compare_vs_previous_default.json`
- `outputs/fattn56-direct-epilogue-speed-20260517/summary.json`

### 2026-05-17: Patch-embed bias layout and large axis0 broadcast fusion

The next low-risk fusion pass targeted the remaining Hiera patch-embed
`ADD(bias) -> permute/view -> CONT` overhead visible in
`GGML_CUDA_PROFILE_NODES`. The graph now applies the patch-embed bias after the
WHC-to-CHW layout conversion, using bias shape `[E,1,1,1]` instead of
`[1,1,E,1]`. This keeps the arithmetic order relative to the following
positional-embedding add and is bit-identical on the tracking output.

The CUDA F32 axis-broadcast fast path was also extended for large axis-0 row
counts. Previously `axis0_grid_y > 65535` fell back to the generic modulo
kernel, which meant the `dst=f32[112,256,256,1] src1=f32[112,1,1,1]` patch
bias still used the slower generic path. The specialized axis-0 kernel now uses
`grid.z` to cover larger row counts.

Correctness:

| Check | Result |
| --- | ---: |
| Baseline layout vs bias-after-layout | `10 / 10` mask hashes equal |
| Bias-after-layout vs large-axis0 kernel | `10 / 10` mask hashes equal |
| Max bbox delta / score delta | `0.0 / 0.0` |

Same-binary five-pair A/B for the large-axis0 kernel shows only a small
positive effect, so this is kept as a general broadcast cleanup rather than a
PyTorch-closing optimization:

| Mode | Track/fr runs (ms) | Mean (ms) | Median (ms) | Stdev |
| --- | --- | ---: | ---: | ---: |
| Large axis0 enabled | `45.0, 45.0, 45.1, 45.5, 45.7` | `45.26` | `45.1` | `0.321` |
| `GGML_CUDA_DISABLE_BIN_BCAST_AXIS0_FAST=1` | `45.6, 45.1, 45.6, 45.1, 45.3` | `45.34` | `45.3` | `0.251` |

The broader profile conclusion is unchanged: after this cleanup, the dominant
remaining q4_1 1024 work is still q4_1 MMQ body throughput and D56
FlashAttention, not scalar post-op fusion.

Evidence:

- `outputs/patch-embed-bias-after-cont-20260517/compare.json`
- `outputs/patch-embed-bias-after-cont-20260517/compare-axis0-large.json`
- `outputs/patch-embed-bias-after-cont-20260517/after-hotspots.json`
- `outputs/axis0-large-ab-20260517/summary.json`

### 2026-05-17: Fusion-kernel direction after the axis0 cleanup

The current profile was refreshed before starting the next fusion-kernel slice:

| Area | Drop-max profile cost over 3 Hiera encode runs | Interpretation |
| --- | ---: | --- |
| q4_1 MMQ stage-2 expansion/projection | `7.66 ms + 6.99 ms` | Primary CUDA-side throughput target |
| q4_1 MMQ window QKV projection | `6.37 ms` | Primary CUDA-side throughput target |
| D56 FlashAttention window/global | `5.11 ms + 4.74 ms` | Primary attention target |
| Fused multi-add elementwise chains | `~3.5 ms` | Already fused; memory-bandwidth/launch cleanup only |
| Patch-embed bias add | `~0.18-0.19 ms` warm | Too small to close the PyTorch gap alone |

The next fusion work should therefore avoid broad scalar fusion churn. The
useful direction is to eliminate data preparation around the dominant kernels:

1. MMQ producer/consumer fusion: for `fc1 + bias + GELU -> fc2`, the consumer
   currently pays a standalone q8_1 activation quantization before q4_1 MMQ.
   A real win needs the producer MMQ epilogue to create the q8_1 DS4 side
   output while writing F32, so the consumer can skip the separate quantize
   launch. The older graph-local prequant cache is not enough because it adds a
   separate producer-side quantize kernel.
2. D56 FlashAttention fusion: reduce the D56-specific pack/fixup/body work.
   The rejected direct no-fixup epilogue showed that replacing only the final
   writeback is not enough.
3. Elementwise fusion is now a tertiary cleanup: keep narrow, measured changes
   such as axis-specific broadcast kernels, but do not expect them to close the
   remaining `~2-3 ms/frame` gap to official PyTorch bf16.

Two same-binary checks were run to prevent enabling weak fusion paths by
default:

| Probe | Enabled/default mean | Disabled/alternate mean | Decision |
| --- | ---: | ---: | --- |
| Non-contiguous `src1` direct strided quantize | `45.34 ms` | `45.28 ms` | Do not expand this path |
| Existing graph-local MMQ prequant cache | `45.16 ms` | `45.28 ms` | Too small/noisy; keep opt-in until epilogue side-output exists |

Acceptance criteria for the next fusion-kernel implementation:

- JSONL parity must report `10 / 10` mask hashes equal against the current
  default, with bbox/score deltas `0` unless the change intentionally alters
  precision.
- Same-binary A/B must show a median improvement larger than run noise
  (`>= 0.5 ms/frame` for q4_1 Base+ 1024) before enabling by default.
- If the implementation is an MMQ side-output path, the consumer profile must
  show cached/prequantized `src1` usage and the producer must not add an
  extra standalone quantize launch.
- The fallback path must remain available through an environment variable until
  parity and speed have been repeated across at least q4_1 Base+ and one
  non-q4_1 control precision.

Evidence:

- `outputs/fusion-kernel-profile-20260517/hotspots.json`
- `outputs/strided-src1-quant-ab-20260517/`
- `outputs/mmq-prequant-cache-ab-fusion-20260517/`

### 2026-05-17: Rejected q8_1 MMA B-tile load-generic probe

After the fusion triage, a smaller q4_1 MMQ body probe changed the NVIDIA
`vec_dot_q8_1_q8_1_mma` B-tile load from `load_ldmatrix` to `load_generic`,
mirroring the q8_0 path where generic loads are already used. This kept the
accumulation and writeback logic unchanged.

Correctness was exact against the current q4_1 tracking baseline:

| Metric | Value |
| --- | ---: |
| Rows compared | `10` |
| Min bbox IoU | `1.0` |
| Max bbox delta px | `0.0` |
| Max score abs delta | `0.0` |
| Mask hash equal rows | `10 / 10` |

The normal application speed change was too small to accept:

| Mode | Track/fr runs (ms) | Mean |
| --- | --- | ---: |
| Baseline `load_ldmatrix` | `44.9, 44.7, 45.6, 44.9, 45.2` | `45.06` |
| Probe `load_generic` | `45.0, 45.1, 44.8, 45.1, 44.7` | `44.94` |

The probe was reverted. The measured `0.12 ms/frame` mean delta is below the
fusion acceptance threshold and the change would affect q8_1-MMA users beyond
the q4_1 Hiera target.

Evidence:

- `outputs/mmq-q81-load-generic-probe-20260517/compare.json`
- `outputs/mmq-q81-load-generic-speed-20260517/`

### 2026-05-17: q4_1 4096-column MMQ cap refresh

The existing runtime cap for the Base+ Hiera q4_1 `ncols_max=4096` MMQ bucket
was also refreshed with `GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX`. Current launch
profiling shows the default route already selects `mmq_x=96` for the hot
`448 -> 1792 -> 448` MLP pair, with stream-k fixup disabled.

Three-run bbox-only sweep:

| Cap | Track/fr runs (ms) | Mean |
| --- | --- | ---: |
| default | `45.3, 46.5, 45.0` | `45.60` |
| `80` | `45.4, 47.4, 45.9` | `46.23` |
| `88` | `46.2, 45.4, 45.7` | `45.77` |
| `96` | `45.3, 45.2, 44.9` | `45.13` |
| `104` | `45.4, 47.8, 50.7` | `47.97` |
| `112` | `45.5, 46.7, 46.1` | `46.10` |
| `128` | `47.3, 45.6, 46.5` | `46.47` |

This does not justify a source change. The only apparently better setting is
the already selected default launch size, and the spread is dominated by normal
run noise.

Evidence:

- `outputs/mmq-launch-current-20260517/run.stderr`
- `outputs/mmq-q41-4096-xcap-sweep-20260517/`

### 2026-05-17: Fusion-kernel candidate narrowed to Hiera MLP side output

The next fusion slice was re-profiled with both MMQ and node-level logging so
producer/consumer pairs could be identified instead of inferred from shape
alone. The useful target is now narrower:

- Producer: q4_1 Hiera `mlp.fc1` MMQ with fused bias + GELU.
- Consumer: the following q4_1 Hiera `mlp.fc2` MMQ.
- Fusion payload: have the producer epilogue optionally emit a q8_1 DS4 side
  output for the already activated F32 result, then let the consumer read that
  prequantized buffer and skip its standalone activation quantize launch.
- Scope guard: Base+ q4_1 Hiera only at first; no `ids`, F32 dst, contiguous
  producer output, DS4 q8_1 layout, and graph-local lifetime only.

Measured upper bound from a 5-frame node/MMQ profile, dropping the first row per
consumer shape:

| Consumer shape | Count | Quantization cost sum | Per-row p50 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `1792,448 -> 448,4096` | `54` | `1.577824 ms` | `0.029088 ms` | Main late-stage MLP volume |
| `896,224 -> 224,16384` | `9` | `1.215680 ms` | `0.140096 ms` | Smaller count, larger activations |
| `3584,896 -> 896,1024` | `14` | `0.211776 ms` | `0.015328 ms` | Low-value tail |
| Total candidate | `77` | `3.005280 ms` | `0.028992 ms` | About `0.75 ms/frame` in this 5-frame run |

This is large enough to prototype, but it is not enough by itself to close the
full gap to official PyTorch bf16. It should be implemented as an opt-in fusion
path and accepted only if normal same-binary A/B shows at least
`0.5 ms/frame` median improvement with exact JSONL parity.

Implementation order:

1. Add a graph-local q8_1 DS4 side-output slot to the producer MMQ epilogue,
   gated by an environment variable and exact shape checks.
2. Teach the consumer MMQ dispatch to use that side output only when the source
   tensor identity, shape, consumer type, and lifetime match.
3. Keep the existing non-fused F32 path as the default fallback until q4_1
   Base+ parity and speed pass the acceptance gate.
4. If the side-output path underperforms, stop there and move the root-cause
   work to D56 FlashAttention body optimization; do not broaden scalar fusion.

Evidence:

- `outputs/fusion-kernel-candidates-20260517/node-profile/side-output-candidates.json`
- `outputs/fusion-kernel-candidates-20260517/node-profile/mmq-summary.json`

### 2026-05-17: Rejected consumer-side direct F32 src1 tile quantization

An opt-in probe tried to skip the standalone global q8_1 activation buffer for
q4_1 MMQ when `nrows_x <= mmq_y`, quantizing the F32 `src1` directly into the
MMQ shared-memory `block_q8_1_mmq` tile. The intended target was the early
Base+ Hiera `mlp.fc2` shape:

| Shape | Standalone quantization cost | Intended benefit |
| --- | ---: | --- |
| `src0=q4_1[448,112] src1=f32[448,65536] dst=f32[112,65536]` | `~0.28-0.29 ms` per row after warmup | Remove global q8_1 materialization for this single-row-tile MMQ |

The probe was rejected on correctness before speed acceptance:

| Metric | Result |
| --- | ---: |
| Rows compared | `10` |
| Mask hash equal rows | `0 / 10` |
| Min bbox IoU | `0.0` |
| Max bbox delta px | `820.0` |
| Max score abs delta | `0.826611` |

This failure means the direct shared-memory tile writer does not yet reproduce
the exact q8_1 DS4 layout consumed by the q4_1 MMA path, or it violates a
stream-k/writeback assumption. Do not enable or expand this path. The next
fusion work should return to the measured producer-side `fc1 + bias + GELU`
side-output route, where the standalone quantization kernel can be removed
without changing the consumer MMQ tile reader.

Evidence:

- `outputs/mmq-direct-f32-src1-probe-20260517/fix1/compare.json`
- `outputs/mmq-direct-f32-src1-probe-20260517/profile/run.stderr`

### 2026-05-17: Prequant-cache refresh before producer side-output work

The existing graph-local prequant cache was refreshed as the control path for
producer-side side-output work. With `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1`,
the consumer-side q4_1 Hiera `mlp.fc2` quantization is mostly skipped:

| Consumer shape | Cached rows | Uncached rows | Cached quant sum | Cached quant p50 |
| --- | ---: | ---: | ---: | ---: |
| `1792,448 -> 448,4096` | `80` | `0` | `0.269728 ms` | `0.003520 ms` |
| `896,224 -> 224,16384` | `15` | `0` | `0.034048 ms` | `0.002240 ms` |
| `3584,896 -> 896,1024` | `15` | `0` | `0.048544 ms` | `0.003456 ms` |

This confirms the consumer dispatch can use a prequantized q8_1 source, but the
current implementation still creates that source with a separate producer-side
quantization kernel. Normal same-binary timing therefore remains a small/noisy
win rather than a decisive fusion result:

| Mode | Track/fr runs (ms) | Mean | Median |
| --- | --- | ---: | ---: |
| Default | `45.2, 45.3, 45.7, 45.6, 45.0` | `45.36` | `45.30` |
| Prequant cache | `44.7, 45.9, 44.9, 45.0, 45.5` | `45.20` | `45.00` |

The measured `0.16 ms/frame` mean improvement is below the fusion acceptance
threshold. The next implementation should not broaden the cache policy; it
should move q8_1 DS4 production into the producer MMQ epilogue, or abandon this
route and spend the effort on the D56 FlashAttention body.

Evidence:

- `outputs/mmq-producer-side-output-baseline-20260517/prequant-cache.json`
- `outputs/mmq-producer-side-output-baseline-20260517/mmq-summary.json`
- `outputs/mmq-prequant-cache-speed-refresh-20260517/table-summary.json`

### 2026-05-17: Goal audit after the fusion probes

The current completion audit is:
`outputs/goal-audit/current-after-fusion-probes-20260517-summary.json`.

The explicit objective is not complete. The audit maps the goal to these
current deliverables:

| Requirement | Status | Evidence |
| --- | --- | --- |
| Detailed Hiera encode profile | Met | `outputs/stage-profile-baseplus-q4_1-current-1024-20260516/summary.json`, `outputs/fusion-direction-profile-20260517/q4_1_hiera_hotspots.json` |
| Base+ official-Python fallback quality | Met | `outputs/sam2-official-quality-current-matrix-20260517/summary.json` |
| Matched official PyTorch speed comparison | Met | `outputs/current-baseplus-q8q41-vs-python-refresh-20260517/q4_1/summary.json`, `outputs/python-official-profile/base_plus_1024-final-20260516.json` |
| Speed-priority evidence | Met | MMQ/FATTN hotspots plus rejected-probe evidence in this document |
| Faster than official PyTorch | Missing | C++ q4_1 Base+ `44.9 ms/frame` vs PyTorch bf16 `42.666687668922044 ms/frame` |

Current measured gap on the matched Base+ q4_1 1024 row is therefore
`2.233312331077954 ms/frame`. The latest prequant-cache refresh only explains a
`0.16 ms/frame` mean app-level improvement, so it is not a PyTorch-closing path
unless q8_1 production is moved into the producer MMQ epilogue. The remaining
work should stay on q4_1 MMQ body/scheduling or a correctness-aware D56
FlashAttention body kernel.

### 2026-05-17: MMQ epilogue q8_1 side-output probe

An opt-in producer MMQ epilogue q8_1 DS4 side-output was implemented behind
`GGML_CUDA_ENABLE_MMQ_EPILOGUE_PREQUANT=1` and tested against the same
Base+ q4_1 10-frame tracking JSONL baseline. The intended optimization was to
remove the standalone producer-side `quantize_mmq_q8_1_cuda(dst)` launch by
writing the cached q8_1 source immediately after producer MMQ writeback.

The probe is rejected for parity. The original version wrote outside the final
partial column tile; after fixing that, the result still did not match the
separate quantization path:

| Variant | Mask hash equal rows | Max bbox delta px | Max score abs delta |
| --- | ---: | ---: | ---: |
| Initial epilogue side-output | `0 / 10` | `120.0` | `0.010925999999999991` |
| No stream-k producer workaround | `0 / 10` | `120.0` | `0.14604700000000004` |
| Exact stream-k schedule gated | `0 / 10` | `119.0` | `0.026697000000000082` |
| Block fence added | `0 / 10` | `120.0` | `0.02801600000000004` |
| Partial-column write fixed | `0 / 10` | `25.0` | `0.01189399999999996` |

The separate prequant cache control remains exactly parity-preserving:
`10 / 10` mask-hash-equal rows, zero bbox delta, and zero score delta.

The epilogue path is therefore not accepted as an optimization. The experimental
hot-path code was removed after this probe to avoid carrying an unused runtime
branch in MMQ; the evidence remains here as a rejected-fusion record. Further
fusion work should not read global `dst` after MMQ writeback. It should either
produce q8_1 from the writeback values before they leave the kernel, using a
shared-memory reduction designed around the MMA writeback layout, or target a
different hotspot such as D56 FlashAttention.

Evidence:

- `outputs/mmq-epilogue-prequant-probe-20260517/prequant-only/compare.json`
- `outputs/mmq-epilogue-prequant-probe-20260517/valid-col-fix/compare.json`
- `outputs/mmq-epilogue-prequant-probe-20260517/removed-hot-path/compare.json`
- `outputs/mmq-epilogue-prequant-probe-20260517/profile-gated/epilogue.stderr`

### 2026-05-17: Fusion-kernel direction recheck

The fusion-kernel path was rechecked after removing the unsafe MMQ epilogue
side-output branch. The local `GELU + q8_1 DS4` kernel is still a real local
win, but the current graph-level way of exploiting prequantized activations is
not yet a production speedup.

| Probe | Separate median | Fused/prequant median | Result |
| --- | ---: | ---: | --- |
| `1792 x 4096`, bias + GELU + q8_1 DS4 | `0.161856 ms` | `0.141824 ms` | fused kernel `12.376%` faster, byte-exact |
| `1792 x 4096`, GELU + q8_1 DS4 | `0.155584 ms` | `0.145728 ms` | fused kernel `6.335%` faster, byte-exact |
| q4_1 two-stage MLP, default graph | `0.2849885 ms` mean median | - | baseline |
| q4_1 two-stage MLP, existing prequant cache | - | `0.287373 ms` mean median | `0.689%` slower by mean-of-means |

This confirms the current implementation rule:

1. Do not promote the existing `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` path.
   It proves the consumer can read a cached q8_1 source, but producer-side
   q8_1 creation still costs enough to erase the benefit.
2. Do not reintroduce the graph split that turns `MMQ + bias + GELU` into
   `MMQ + bias` plus a separate fused `GELU + q8_1` launch. It loses the
   existing static MMQ GELU epilogue and was slower in full tracking.
3. A future accepted fusion kernel must create q8_1 DS4 from the producer MMQ
   writeback values inside the producer kernel, or it must be abandoned in
   favor of MMQ/FATTN body work.
4. The acceptance gate remains exact tracking JSONL parity plus at least
   `0.5 ms/frame` median improvement on matched Base+ q4_1 1024 timing.

Evidence:

- `outputs/fusion-kernel-20260517/gelu-quant/bias_gelu_quant_ds4.log`
- `outputs/fusion-kernel-20260517/gelu-quant/gelu_quant_ds4.log`
- `outputs/fusion-kernel-20260517/mmq-paired/summary_drop_first.json`

### 2026-05-17: Rejected upstream FATTN MMA stride probe

The generic part of upstream ggml commit `a08c1594` changes the VKQ loop in
`flash_attn_ext_f16_iter` from the local `i0_stride` expression to
`T_A_VKQ::I`. It was tested as an isolated CUDA FATTN body probe because the
change is not AMD-only at the source level.

The patch built and passed standalone D56 tolerance checks, but it was slower
on both representative D56 shapes:

| Shape | Result |
| --- | ---: |
| D56 global, `N=4096,H=8,B=1`, sampled queries | `mean_ms=1.021067`, `median_ms=1.030207`, `bad=0/4032` |
| D56 window, `N=196,H=8,B=25` | `mean_ms=1.031904`, `median_ms=1.030246`, `bad=0/2195200` |

This is worse than the current nearby D56 measurements, so the isolated stride
change was reverted. The rest of `a08c1594` is RDNA/CDNA tuning and does not
justify a NVIDIA default change for this branch.

Evidence:

- `outputs/fattn-upstream-stride-probe-20260517/global.log`
- `outputs/fattn-upstream-stride-probe-20260517/window.log`

### 2026-05-17: Accepted q4_1 MMQ direct-tile fastpath

The next accepted MMQ change targets launches that are selected as Stream-K but
are effectively one block per output tile and need no fixup. For those q4_1
Hiera MLP shapes, the generic continuous `kbc` partitioning path only adds index
work. The new fastpath maps `blockIdx.x` directly to `(sample, channel, tile_y,
tile_x)` and calls the normal tile body with the full-K range.

The change is enabled by default only for q4_1. It can be disabled with
`GGML_CUDA_DISABLE_MMQ_DIRECT_TILE_FASTPATH=1`; other quantized types remain
opt-in through `GGML_CUDA_ENABLE_MMQ_DIRECT_TILE_FASTPATH=1`.

Correctness checks:

| Check | Result |
| --- | --- |
| `sam3_mmq_bench --check`, `cols=128`, default | `bad=0/57344`, `max_abs=0.207466185`, `mean_abs=0.0354335134` |
| `sam3_mmq_bench --check`, `cols=128`, fastpath | `bad=0/57344`, `max_abs=0.207466185`, `mean_abs=0.0354335134` |
| 10-frame full-mask tracking JSONL compare | `10/10` mask hashes equal, `min_bbox_iou=1.0`, zero bbox/score/mask-area delta |
| Current default-enabled vs `GGML_CUDA_DISABLE_MMQ_DIRECT_TILE_FASTPATH=1` | `10/10` mask hashes equal, zero bbox/score/mask-area delta |

Performance:

| Contract | Default | Fastpath | Result |
| --- | ---: | ---: | --- |
| Synthetic q4_1 two-stage MLP, drop-first mean of means | `0.3093095 ms` | `0.290515 ms` | `~6.1%` faster |
| Base+ q4_1 1024 bbox-only, 7 paired runs, `track_ms` mean | `44.657142857 ms/frame` | `43.742857143 ms/frame` | `2.09%` faster |
| Base+ q4_1 1024 bbox-only, paired saved `track_ms` | - | `0.914285714 ms/frame` | 95% CI `[0.507562221, 1.321009208]` |
| Base+ q4_1 1024 bbox-only, total run time mean | `737.385714286 ms` | `737.657142857 ms` | load/init dominated, comparable |
| Current default-enabled smoke, bbox/full-mask contract | `45.7 ms/frame` disabled | `44.6 ms/frame` default | same direction as paired run |

This does not complete the PyTorch gap by itself, but it is a measured
kernel-body/scheduling cleanup on the dominant q4_1 MMQ path. The remaining root
targets stay the q4_1 MMQ body arithmetic and D56 FlashAttention body/writeback.

Evidence:

- `outputs/mmq-direct-tile-fastpath-20260517/check_default.log`
- `outputs/mmq-direct-tile-fastpath-20260517/check_fastpath.log`
- `outputs/mmq-direct-tile-fastpath-20260517/paired/`
- `outputs/mmq-direct-tile-fastpath-20260517/app/compare.json`
- `outputs/mmq-direct-tile-fastpath-20260517/app/pairs/summary.json`
- `outputs/mmq-direct-tile-fastpath-20260517/default-enabled-check/compare.json`

### 2026-05-17: Post direct-tile hotspot refresh

After enabling the q4_1 MMQ direct-tile fastpath by default, the current Base+
q4_1 1024 profile was refreshed. The matched speed gap is now about
`1.08 ms/frame` against the official PyTorch bf16 baseline
(`43.742857143 ms/frame` from the 7-run direct-tile A/B versus
`42.666687668922044 ms/frame` official Python). The goal is still not complete.

Normal timing and synchronized profiling point to the same remaining priority:

| Surface | Current evidence | Interpretation |
| --- | ---: | --- |
| Base+ q4_1 bbox-only 7-run fastpath mean | `43.742857143 ms/frame` | still slower than official PyTorch bf16 |
| Hiera encode node profile, q4_1 stage-2 MLP expansion | `12.854976 ms` drop-max over 5 profiled frames | top remaining MMQ body surface |
| Hiera encode node profile, q4_1 stage-2 MLP projection | `11.340000 ms` drop-max | second MMQ body surface |
| Hiera encode node profile, q4_1 window QKV projection | `10.392800 ms` drop-max | still a primary MMQ surface |
| Hiera encode node profile, D56 window FlashAttention | `8.253632 ms` drop-max | second-tier after MMQ |
| Hiera encode node profile, D56 global FlashAttention | `8.001792 ms` drop-max | second-tier after MMQ |

Detailed MMQ profiling after direct-tile shows total profiled q4_1 MMQ time
`24.156672 ms`, split into `7.725056 ms` activation quantization and
`16.431616 ms` MMQ body. The largest MMQ shape remains
`q4_1[1792,448] x f32[1792,4096] -> f32[448,4096]`, with
`2.245888 ms` quantization and `9.107840 ms` MMQ body. The large
`q4_1[448,112] x f32[448,65536] -> f32[112,65536]` row is quantization-heavy
(`2.326592 ms` quantization, `1.692416 ms` MMQ body), but it is not enough by
itself to close the Python gap.

FATTN56 profiling remains mostly MMA-body time: total profiled FATTN56 time is
`25.449536 ms`, with `25.021888 ms` in MMA, `0.381440 ms` packing, and
`0.046208 ms` slicing. This confirms that future FATTN work must change the D56
body itself; pack/slice cleanup alone cannot close the remaining gap.

Evidence:

- `outputs/post-direct-tile-profile-q4_1-1024-20260517/node-summary.json`
- `outputs/post-direct-tile-profile-q4_1-1024-20260517/hotspots.json`
- `outputs/post-direct-tile-profile-q4_1-1024-20260517/mmq/summary.json`
- `outputs/post-direct-tile-profile-q4_1-1024-20260517/fattn56/summary.json`
- `outputs/post-direct-tile-profile-q4_1-1024-20260517/launch/summary.json`

### 2026-05-17: Rejected post direct-tile MMQ selector probes

Two small selector probes were rerun after the direct-tile fastpath changed the
baseline. Neither is accepted.

| Probe | Result | Decision |
| --- | ---: | --- |
| `GGML_CUDA_MMQ_Q8_1_WARP_COLS=8` vs current default `4` | 5-run Base+ q4_1 track mean `43.66` vs `43.78 ms/frame`, paired saved `0.12 ms/frame`, CI crosses zero | do not change default |
| `GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX=96` | 7-run track mean `43.842857` vs default `44.000000 ms/frame`, paired saved `0.157143 ms/frame`, CI crosses zero | do not change default |
| Stream-K tile-efficiency threshold `80` | removed all launch-profile fixup rows, but 5-run track mean regressed `43.90 -> 44.02 ms/frame` | reverted diagnostic env hook |
| q8_1 warp-column float4 load | giant-column microbench mean `0.452053 ms` default vs `0.451256 ms` float4, paired saved `0.000798 ms`, CI crosses zero | reverted diagnostic env hook |
| q4_1 direct-tile compile-time specialization | compile failed because `cudaFuncSetAttribute` setup also needs all `mul_mat_q` template signatures updated | reverted before timing |

The useful conclusion is that the remaining q4_1 gap is not a selector problem:
tile shape, high-column quantizer width, and fixup avoidance are all too small
or noisy. The next accepted speed work must reduce q4_1 MMQ body arithmetic or
change the D56 FATTN body.

Evidence:

- `outputs/mmq-warpcols-after-direct-tile-20260517/micro/summary.json`
- `outputs/mmq-warpcols-after-direct-tile-20260517/app-pairs/summary.json`
- `outputs/mmq-xcap-after-direct-tile-20260517/app/summary.json`
- `outputs/mmq-xcap96-after-direct-tile-20260517/pairs/summary.json`
- `outputs/mmq-tile-threshold-after-direct-tile-20260517/launch80-summary.json`
- `outputs/mmq-tile-threshold-after-direct-tile-20260517/pairs/summary.json`
- `outputs/mmq-warpcol-float4-probe-20260517/micro/summary.json`

### 2026-05-17: Post direct-tile fusion-kernel policy

The fusion-kernel route was refreshed after the q4_1 direct-tile fastpath became
the default. The existing graph-local `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1`
control path still does not justify promotion:

| Mode | Drop-first mean of means | Drop-first mean of medians | Result |
| --- | ---: | ---: | --- |
| Current default | `0.288184833 ms` | `0.287927000 ms` | baseline |
| `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1` | `0.290582500 ms` | `0.290451333 ms` | `0.002397667 ms` slower by mean of means |

This keeps the fusion policy unchanged:

1. Do not enable the current prequant cache by default. It proves the consumer
   MMQ can read cached q8_1, but moving the standalone quantization earlier is
   not a speedup.
2. Do not split `MMQ + bias + GELU` into a separate activation/quantization
   graph path. The local `GELU + q8_1 DS4` kernel is faster in isolation, but
   the full two-stage graph loses more than it saves.
3. The only fusion implementation still worth pursuing is a producer-side
   side-output written inside the q4_1 MMQ producer kernel, from the same
   writeback values before they leave the kernel. Anything that reads `dst`
   again after MMQ writeback is just another quantization launch and should be
   rejected unless app timing proves otherwise.
4. Because prior epilogue side-output probes failed parity, the next attempt
   must be designed around the MMA writeback layout and must pass exact
   tracking JSONL parity before speed is considered.

For the current PyTorch-gap work, fusion remains a secondary route unless that
producer-in-kernel side-output can be made exact. The primary near-term targets
remain q4_1 MMQ body arithmetic/scheduling and D56 FlashAttention body/writeback
changes.

Evidence:

- `outputs/fusion-kernel-post-direct-tile-20260517/summary_drop_first.json`

### 2026-05-17: Rejected D56 FATTN no-fixup direct writeback probe

A D56 FlashAttention probe replaced the no-mask, no-sinks, `np == 1`,
no-fixup Turing-MMA writeback with direct stores from the VKQ accumulator
registers. The goal was to avoid the generic shared-memory staging/writeback
path for the dominant D56 `ncols=64` cases while leaving all fixup and
multi-partial paths unchanged.

The mapping preserved the existing microbench parity, but it was not faster:

| Shape | Baseline mean | Direct writeback mean | Parity |
| --- | ---: | ---: | --- |
| D56 global, `N=4096`, `heads=8`, `batch=1` | `0.696688 ms` | `0.697320 ms` | `bad=0/4032` |
| D56 window, `N=196`, `heads=8`, `batch=25` | `0.212424 ms` | `0.212624 ms` | `bad=0/100800` |

The probe was reverted. This indicates that the shared-memory epilogue is not a
meaningful D56 bottleneck on the current hardware/compiler, and that the FATTN
route must change the MMA body or scheduling rather than only bypassing the
writeback staging.

Evidence:

- `outputs/fattn56-direct-epilogue-probe-20260517/baseline/global.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/baseline/window.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/direct/global.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/direct/window.log`

### 2026-05-17: Rejected q4_1 MMQ inner-loop FMA fold

A q4_1 MMQ body probe folded the NVIDIA-side q8_1 MMA accumulation update from
two explicit additions into one `fmaf(...) + min_term` expression. The intent was
to reduce repeated indexed writes to the local `sum` accumulator in the hottest
q4_1 body loop.

The probe preserved the synthetic check result but regressed the two-stage MLP
timing badly:

| Check | Result |
| --- | ---: |
| q4_1 two-stage `cols=128` check | `bad=0/57344`, `max_abs=0.207466185` |
| q4_1 two-stage `cols=4096`, 100 iters | `0.318610 ms` mean, `0.318657 ms` median |

The current post-direct-tile baseline for the same synthetic two-stage contract
is about `0.288 ms`, so this probe was reverted. The likely cause is worse
compiler scheduling/register allocation around the MMA accumulator update, not
an arithmetic correctness issue.

Evidence:

- `outputs/mmq-q41-inner-fma-probe-20260517/check.log`
- `outputs/mmq-q41-inner-fma-probe-20260517/timing.log`

### 2026-05-17: Patch-embed copy+bias fusion

The Hiera patch embedding path had a remaining unfused channel-bias add:

`IM2COL -> MUL_MAT -> CONT -> CONT -> ADD(dbg_patch_embed)`.

A targeted CUDA fusion now lets the final `CONT` write directly to the ADD
output while adding an axis-0 F32 bias. It is enabled by default and can be
disabled with `GGML_CUDA_DISABLE_CPY_BIAS_AXIS0_FUSION=1`.

The fusion is exact against the disabled path on Base+ q4_1/1024 bbox tracking:
10/10 JSONL rows match, all mask hashes match, min bbox IoU is `1.0`, and all
bbox/score deltas are zero. Node profiling confirms the `dbg_patch_embed` ADD is
removed and the preceding `CONT` reports `fused=1 skipped=2`.

This is still a small launch/memory-traffic cleanup rather than a material
PyTorch-gap fix. The 7-pair app timing measured only `0.114286 ms/frame` saved
on average, with a 95% CI of `[-0.592941, 0.821513] ms/frame`, so the speed
effect is statistically indistinguishable from noise. It is kept as a safe
graph cleanup, but it does not change the main priority: q4_1 MMQ body and D56
FlashAttention body work remain the only routes likely to close the remaining
Base+ gap.

Evidence:

- `outputs/cpy-bias-axis0-fusion-20260517/profile-default-v5/node-profile.log`
- `outputs/cpy-bias-axis0-fusion-20260517/parity/compare.json`
- `outputs/cpy-bias-axis0-fusion-20260517/speed-pairs/summary.json`

### 2026-05-17: Accepted q8_0 MMQ direct-tile default

The q4_1 direct-tile MMQ fastpath was checked on the Base+ q8_0 fallback
candidate. q8_0 is the stronger-quality fallback than q4_1, so carrying the
same indexing cleanup into q8_0 directly improves the practical fallback path.

With `GGML_CUDA_ENABLE_MMQ_DIRECT_TILE_FASTPATH=1`, q8_0 matched the default
JSONL output exactly: 10/10 mask hashes equal, min bbox IoU `1.0`, and all
bbox/score deltas zero. A 7-pair opt-in timing showed a clear speedup:

| Mode | Mean track ms/frame |
| --- | ---: |
| Previous q8_0 default | `47.014286` |
| q8_0 direct-tile opt-in | `45.700000` |

The paired mean saved `1.314286 ms/frame`, with 95% CI
`[0.526937, 2.101634]`.

The fastpath is now enabled by default for both q4_1 and q8_0. The shared escape
hatch remains `GGML_CUDA_DISABLE_MMQ_DIRECT_TILE_FASTPATH=1`. After the default
change, a 5-pair disabled-vs-default check measured `47.300000 -> 45.860000
ms/frame`, paired saved `1.440000 ms/frame`, 95% CI
`[0.613570, 2.266430]`, again with exact JSONL parity.

This moves the Base+ q8_0 fallback closer to the official PyTorch bf16 Base+
row (`42.67 ms/frame` in the current matrix), but does not fully close the gap.
The remaining work is still q4_1/q8_0 MMQ body arithmetic or D56 FlashAttention
body/scheduling.

Evidence:

- `outputs/q8-direct-tile-probe-20260517/parity/compare.json`
- `outputs/q8-direct-tile-probe-20260517/speed-pairs/summary.json`
- `outputs/q8-direct-tile-default-20260517/parity/compare.json`
- `outputs/q8-direct-tile-default-20260517/speed-pairs/summary.json`

### 2026-05-17: Accepted q4_0 MMQ direct-tile default

The same direct-tile MMQ fastpath was checked for Base+ q4_0. q4_0 is not the
preferred quality fallback, but it is still part of the supported precision
matrix and the direct-tile path is exact for it too.

The opt-in probe matched the default JSONL output exactly: 10/10 mask hashes
equal, min bbox IoU `1.0`, and all bbox/score deltas zero. A 5-pair opt-in
timing measured `45.000000 -> 44.340000 ms/frame`, paired saved `0.660000
ms/frame`, 95% CI `[0.107494, 1.212506]`.

q4_0 is now enabled by default together with q4_1 and q8_0. The shared escape
hatch remains `GGML_CUDA_DISABLE_MMQ_DIRECT_TILE_FASTPATH=1`. The default
change was checked against the disabled path with exact JSONL parity; a short
3-pair smoke measured `45.133333 -> 44.533333 ms/frame`, but the CI crosses zero
because this post-change smoke was intentionally small. The 5-pair opt-in run is
the speed evidence for adoption.

Evidence:

- `outputs/q40-direct-tile-probe-20260517/parity/compare.json`
- `outputs/q40-direct-tile-probe-20260517/speed-pairs/summary.json`
- `outputs/q40-direct-tile-default-20260517/parity/compare.json`
- `outputs/q40-direct-tile-default-20260517/speed-pairs/summary.json`

### 2026-05-17: Rejected direct-tile no-ids writeback probe

After enabling direct-tile by default for q4_0/q4_1/q8_0, a smaller MMQ probe
tried to skip the identity `ids_dst_shared[j] = j` initialization on the
direct-tile path and pass a null writeback index pointer instead. The goal was
to remove one shared-memory initialization/synchronization from the direct-tile
fastpath without changing the general stream-k/MoE writeback path.

The probe preserved JSONL parity against `GGML_CUDA_DISABLE_MMQ_DIRECT_TILE_FASTPATH=1`
for all checked Base+ precisions:

| Precision | Parity result |
| --- | --- |
| q8_0 | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| q4_1 | `10 / 10` mask hashes equal, max bbox delta `0 px` |
| q4_0 | `10 / 10` mask hashes equal, max bbox delta `0 px` |

The speed result did not justify keeping it. q8_0 still showed the expected
direct-tile win, but q4_1 lost the win and q4_0 moved in the wrong direction:

| Precision | Disabled mean | Probe default mean | Paired saved |
| --- | ---: | ---: | ---: |
| q8_0 | `47.60 ms/frame` | `46.10 ms/frame` | `1.50 ms/frame`, 95% CI `[0.53, 2.47]` |
| q4_1 | `45.34 ms/frame` | `45.50 ms/frame` | `-0.16 ms/frame`, 95% CI `[-1.08, 0.76]` |
| q4_0 | `44.54 ms/frame` | `45.14 ms/frame` | `-0.60 ms/frame`, 95% CI `[-1.46, 0.26]` |

The no-ids probe was reverted. The accepted q4_0/q4_1/q8_0 direct-tile default
and patch-embed copy+bias fusion remain. Rebuilt post-revert q8_0 and q4_1
smokes both preserved exact JSONL parity against the direct-tile-disabled path.

Evidence:

- `outputs/mmq-direct-tile-noids-probe-20260517/q8/parity/compare.json`
- `outputs/mmq-direct-tile-noids-probe-20260517/q41/parity/compare.json`
- `outputs/mmq-direct-tile-noids-probe-20260517/q40/parity/compare.json`
- `outputs/mmq-direct-tile-noids-probe-20260517/q8/speed-pairs/summary.json`
- `outputs/mmq-direct-tile-noids-probe-20260517/q41/speed-pairs/summary.json`
- `outputs/mmq-direct-tile-noids-probe-20260517/q40/speed-pairs/summary.json`
- `outputs/mmq-direct-tile-noids-reverted-20260517/q8/parity/compare.json`
- `outputs/mmq-direct-tile-noids-reverted-20260517/q41/parity/compare.json`

### 2026-05-17: Refreshed producer-side MLP fusion target

The q4_1 two-stage Hiera MLP microbench was rerun after the direct-tile changes.
The current no-`CONT` graph remains faster than forcing an intermediate
`CONT`, and the standalone CPU-reference smoke still passes:

| Microbench | Result |
| --- | ---: |
| q4_1 two-stage `cols=128` check | `bad=0/57344`, `max_abs=0.207466185` |
| q4_1 two-stage no-`CONT`, `cols=4096` | `0.318615 ms` mean |
| q4_1 two-stage with `CONT`, `cols=4096` | `0.371020 ms` mean |

The existing env-gated prequant cache was also rerun on the same synthetic
contract. It still should not be promoted: drop-first 5-pair timing measured
default `0.289823 ms` vs prequant `0.290970 ms`, or `0.001147 ms` slower by
mean of means.

The corrected app-level producer/consumer scan used both `GGML_CUDA_PROFILE_MMQ`
and `GGML_CUDA_PROFILE_NODES` on a 2-frame Base+ q4_1 profile. It found 29
q4_1 Hiera MLP producer outputs where a following q4_1 fc2 MMQ re-quantizes the
producer's GELU output. The consumer-side quantization time is the upper bound
for a true producer-side q8_1 DS4 side-output fusion:

| Consumer shape | Count | Consumer quant sum |
| --- | ---: | ---: |
| `src0=1792,448` / `src1=1792,4096` / `dst=448,4096` | 21 | `0.620320 ms` |
| `src0=896,224` / `src1=896,16384` / `dst=224,16384` | 3 | `0.419392 ms` |
| `src0=3584,896` / `src1=3584,1024` / `dst=896,1024` | 5 | `0.077088 ms` |
| Total | 29 | `1.116800 ms` |

This keeps the fusion-kernel target concrete but bounded. A real producer
side-output kernel can remove about `1.1 ms` of q4_1 activation quantization in
this profile if it is exact, but that is still below the remaining Base+
PyTorch-speed gap. Therefore this route should be implemented only as a scoped
producer-epilogue side-output, while MMQ body and D56 FlashAttention body work
remain the primary speed targets.

Acceptance criteria for the next producer-side fusion attempt:

1. Only enable it for no-ids, no-fixup producer MMQ tiles where the output row
   dimension is a multiple of `4 * QK8_1` and the consumer type requires q8_1
   DS4 layout.
2. Preserve exact 10-frame tracking JSONL parity before any speed claim.
3. Demonstrate a paired app-level speed win; microbench improvement alone is not
   enough because the previous prequant cache moved the quantization but did not
   reduce total time.
4. Keep an env escape hatch and document any unsupported stream-k/fixup shapes.

Evidence:

- `outputs/fusion-kernel-current-20260517/q41_two_stage_no_cont_check.log`
- `outputs/fusion-kernel-current-20260517/q41_two_stage_no_cont_timing.log`
- `outputs/fusion-kernel-current-20260517/q41_two_stage_cont_timing.log`
- `outputs/fusion-kernel-current-20260517/prequant-pairs/summary_drop_first.json`
- `outputs/fusion-side-output-candidates-20260517/profile-nodes.log`
- `outputs/fusion-side-output-candidates-20260517/side-output-candidates-nodes.json`

### 2026-05-17: Accepted q8_0 static MMQ epilogue

The q8_0 fallback path was still paying dynamic epilogue dispatch overhead on
the common no-bias/no-activation MMQ rows. A narrow CUDA change now makes the
static q8_0 MMQ epilogue the default, with
`GGML_CUDA_DISABLE_MMQ_Q8_0_STATIC_EPILOGUE=1` as the same-binary escape hatch.
This is separate from the older generic static-epilogue switch so q4_1/q4_0
behavior stays unchanged.

The opt-in probe was exact against the default path and produced a clear paired
speed win on Base+ q8_0/1024:

| Run | Disabled/default-control | Static epilogue | Paired saved |
| --- | ---: | ---: | ---: |
| 5-pair opt-in probe | `46.02 ms/frame` | `44.18 ms/frame` | `1.84 ms/frame`, 95% CI `[1.2537, 2.4263]` |
| 3-pair default-after-change smoke | `45.8667 ms/frame` | `44.4000 ms/frame` | `1.4667 ms/frame` |

The default-after-change parity smoke is still exact against the disabled path.
This improves the q8_0 fallback baseline but does not complete the PyTorch-speed
goal: the current official PyTorch bf16 Base+ comparison row remains
`42.67 ms/frame`, so q8_0 is still roughly `1.7 ms/frame` slower in this smoke.

Evidence:

- `outputs/q8-static-epilogue-probe-20260517/parity/compare.json`
- `outputs/q8-static-epilogue-probe-20260517/speed-pairs/summary.json`
- `outputs/q8-static-epilogue-default-20260517/parity/compare.json`
- `outputs/q8-static-epilogue-default-20260517/speed-pairs/summary.json`

### 2026-05-17: Rejected GELU plus prequant fusion split

A safer fusion-kernel alternative to the full MMQ producer side-output was
implemented as a temporary opt-in probe: run the producer MMQ as `MUL_MAT +
bias`, then combine `GELU` and q8_1 DS4 prequant-cache creation in one CUDA
kernel. This avoids reading the producer's global `dst` after a fused
`MUL_MAT+bias+GELU` writeback, so it is less invasive than building q8_1 DS4
blocks inside the MMA writeback path.

The probe preserved exact 10-frame bbox JSONL parity against the existing
prequant-cache path, but it did not improve application speed:

| Path | 5-run mean Track/fr | Paired result |
| --- | ---: | ---: |
| Existing prequant cache | `45.2 ms/frame` | baseline |
| Split `MMQ+bias` then fused `GELU+prequant` | `45.8 ms/frame` | `0.6 ms/frame` slower by mean |

The code probe was removed. This confirms that simply moving GELU out of the
MMQ epilogue and fusing it with activation quantization is not the right
fusion-kernel direction. The only remaining producer/consumer fusion worth
another implementation attempt is still an exact q8_1 DS4 side-output written
from the MMQ producer's own final values, or otherwise the work should move
back to MMQ body and D56 FlashAttention body optimization.

Evidence:

- `outputs/mmq-gelu-prequant-fusion-20260517/parity/compare.json`
- `outputs/mmq-gelu-prequant-fusion-20260517/speed-pairs/summary.json`

### 2026-05-17: q8_0 profile after static epilogue and rejected D4 warp-column quantize

After enabling the q8_0 static MMQ epilogue by default, a synchronized 2-frame
Base+ q8_0 profile was refreshed with `GGML_CUDA_PROFILE_NODES=1`,
`GGML_CUDA_PROFILE_MMQ=1`, `GGML_CUDA_PROFILE_MMQ_LAUNCH=1`, and
`GGML_CUDA_PROFILE_FATTN56=1`. Using drop-first-per-shape summaries to remove
first-use setup, the remaining q8_0 work is no longer a useful producer-side
fusion target:

| Area | Drop-first total | Main rows |
| --- | ---: | --- |
| MMQ quantize | `2.853344 ms` | largest rows are `q8_0[448,112] x f32[448,65536]`, `q8_0[1792,448] x f32[1792,4096]`, and `q8_0[896,224] x f32[896,16384]` |
| MMQ body | `5.811584 ms` | dominated by the same MLP / patch-embed rows |
| D56 FlashAttention | `9.285632 ms` | window `3.332160 ms`, global `2.931872 ms`, q-pool/other D56 rows making up the rest |
| Direct producer->consumer q8 candidate | `0.009664 ms` quant upper bound | one mask-decoder row only |

This rules out q8_0 producer-side q8_1 side-output fusion as a meaningful
PyTorch-gap closer. The next q8_0 direction should be MMQ body throughput or
D56 FlashAttention body throughput.

A q8_0 D4 warp-column activation quantize probe was tested next, mirroring the
accepted q4_1 high-column DS4 quantizer but emitting the D4 layout used by q8_0.
It was exact on a 10-frame bbox JSONL comparison, but profile totals did not
improve:

| Mode | Drop-first MMQ quant sum | Drop-first MMQ total |
| --- | ---: | ---: |
| default | `2.773376 ms` | `8.221568 ms` |
| D4 warp-column, `cols=4` | `2.779072 ms` | `8.235424 ms` |
| D4 warp-column, `cols=8` | `2.860032 ms` | `8.710656 ms` |
| D4 warp-column, `cols=16` | `2.787968 ms` | `8.268960 ms` |

The narrow `f32[448,65536]` quantize row improved slightly, but the MLP and
middle-resolution rows regressed enough that the overall MMQ profile lost. The
probe was removed rather than hidden behind a default-off switch.

Evidence:

- `outputs/q8-current-profile-after-static-epilogue-20260517/benchmark-summary.json`
- `outputs/q8-current-profile-after-static-epilogue-20260517/node-summary.json`
- `outputs/q8-current-profile-after-static-epilogue-20260517/mmq-summary-drop-first.json`
- `outputs/q8-current-profile-after-static-epilogue-20260517/fattn56-summary-drop-first.json`
- `outputs/q8-d4-warp-col-quant-probe-20260517/parity/compare.json`
- `outputs/q8-d4-warp-col-quant-probe-20260517/profile/default-mmq.json`
- `outputs/q8-d4-warp-col-quant-probe-20260517/profile/warpcol-mmq.json`
- `outputs/q8-d4-warp-col-quant-probe-20260517/profile/warpcol8-mmq.json`
- `outputs/q8-d4-warp-col-quant-probe-20260517/profile/warpcol16-mmq.json`

### 2026-05-17: Opt-in ADD plus MMQ prequant cache fusion

An opt-in fusion probe was added behind
`GGML_CUDA_ENABLE_ADD_MMQ_PREQUANT_CACHE=1`. When an f32 `ADD` feeds the
activation side of a q4_1 `MUL_MAT`, the ADD kernel now can also emit the
q8_1 DS4 cache that the following MMQ would otherwise create in a separate
quantize launch. This keeps the f32 ADD output for normal graph consumers and
adds a side-output cache keyed to the following MMQ input tensor.

The path is exact on the current q4_1 tracking comparison:

| Check | Result |
| --- | ---: |
| 10-frame bbox/mask parity | exact |
| `mask_hash_equal_rows` | `10 / 10` |
| max bbox delta | `0.0 px` |
| max score delta | `0.0` |

It does fire in the decoder/memory parts of the graph (`cached_src1=1` in the
MMQ profile), but it does not address the large `hiera_stage_0` ADD because
that immediate consumer is not the q4_1 MMQ activation-quantize path. The
paired application benchmark therefore shows only a small, statistically
inconclusive win:

| Mode | 7-pair mean Track/fr |
| --- | ---: |
| disabled | `45.142857 ms/frame` |
| enabled | `45.000000 ms/frame` |
| mean saved | `0.142857 ms/frame` |
| 95% CI | `[-0.206702, 0.492416] ms/frame` |

Keep this default-off until a broader fusion target is found. The remaining
gap is still in D56 FlashAttention body throughput, q4_1 MMQ body throughput,
and the large f32 Hiera stage transition rather than this decoder-side
ADD-to-MMQ quantize boundary.

Evidence:

- `outputs/add-mmq-prequant-fusion-20260517/profile/enabled.log`
- `outputs/add-mmq-prequant-fusion-20260517/parity/compare.json`
- `outputs/add-mmq-prequant-fusion-20260517/pairs/summary.json`

### 2026-05-17: Refreshed fusion-kernel baseline

The fusion-kernel path was refreshed again after the ADD-to-MMQ prequant probe
and the rejected direct-tile static-specialization attempt. The accepted
runtime q4_1/q4_0/q8_0 direct-tile fastpath remains in place; the static
template specialization was removed because it built and preserved parity but
did not improve speed:

| Probe | Result |
| --- | ---: |
| direct-tile runtime vs static-only mean | `45.000000` vs `45.285714 ms/frame` |
| static-only mean saved | `-0.285714 ms/frame` |
| static-only 95% CI | `[-0.736993, 0.165564] ms/frame` |

The current standalone q4_1 two-stage MLP fusion baseline is:

| Mode | Mean |
| --- | ---: |
| no-cont two-stage q4_1 MLP | `0.318461 ms` |
| explicit cont between stages | `0.370751 ms` |
| existing `GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1`, drop-first mean-of-means | `0.290752 ms` |
| default, drop-first mean-of-means | `0.289237 ms` |
| prequant minus default | `+0.001515 ms` |

This confirms the earlier conclusion: moving q8_1 production into a separate
cache path is not enough. A useful producer/consumer fusion must either write
q8_1 DS4 from the producer MMQ kernel's final values, or the work should stay
on q4_1 MMQ body throughput and D56 FlashAttention body throughput. The direct
producer-side MMQ epilogue route is still possible, but it is not a small
follow-up: the current MMA writeback distributes values across lanes and
warps, while DS4 requires per-column reductions over 32-row groups for
scale/sum. Implementing it safely means changing the writeback helper/thread
mapping and gating it to no-ids, no-fixup, full-tile producer rows first.

Acceptance criteria for any renewed MMQ producer-side fusion:

1. It must not read the producer's global `dst` after MMQ writeback.
2. It must emit byte-identical q8_1 DS4 cache values to the existing quantizer
   for the gated full-tile cases.
3. It must be opt-in until a 7-pair app benchmark shows a positive CI.
4. It must keep the current non-fused F32 path as fallback for stream-k fixup,
   ids, partial tiles, non-q4_1 consumers, and shape mismatches.

Evidence:

- `outputs/mmq-direct-tile-static-20260517/parity/compare.json`
- `outputs/mmq-direct-tile-static-20260517/static-pairs/summary.json`
- `outputs/fusion-kernel-current-refresh-20260517/q41_two_stage_no_cont_check.log`
- `outputs/fusion-kernel-current-refresh-20260517/q41_two_stage_no_cont_timing.log`
- `outputs/fusion-kernel-current-refresh-20260517/q41_two_stage_cont_timing.log`
- `outputs/fusion-kernel-current-refresh-20260517/summary_drop_first.json`

### 2026-05-17: Fusion-kernel implementation boundary

The next fusion-kernel attempt should not be another standalone prequant cache
or post-MMQ split. Both variants have now been measured and either lose speed
or have too little headroom. The useful target is narrower:

1. The producer must be a q4_1 MMQ `MUL_MAT + bias + GELU` row that is followed
   by a q4_1 consumer MMQ using the GELU output as `src1`.
2. The producer kernel must create the q8_1 DS4 side-output from its own final
   activated values before they are lost, not by rereading the global f32
   `dst`.
3. The first implementation should only cover full 128-row producer tiles,
   no `ids`, no stream-k fixup, contiguous f32 destination, and q4_1 consumers.
   All other cases must fall back to the current f32 writeback plus normal
   consumer quantization.
4. The side-output must be byte-identical to `quantize_mmq_q8_1_cuda` for the
   gated full-tile cases. Numerical closeness is not enough, because a q8_1
   byte-layout difference changes the consumer MMQ accumulation path.

The attempted preserve-ADD relaxation was also rejected. `hiera_stage_0` is the
large remaining f32 ADD, but its immediate consumer is an f32 lateral
`MUL_MAT`, not a `NORM` node. Relaxing the ADD->NORM preserve-fusion gate
therefore left the node profile unchanged: `hiera_stage_0` still appeared as
`fused=0`, while the following f32 `MUL_MAT` used the existing bias-fused path.
Fusing that ADD into the following f32 projection would require changing the
GEMM input math for `W * (A + B)` or adding a dedicated two-activation GEMM, so
it is a separate project from the q4_1 producer/consumer MMQ side-output.

Current matched Base+ q4_1 speed remains slower than official PyTorch bf16 on
the refreshed 10-frame comparison:

| Runtime | Track mean |
| --- | ---: |
| C++ q4_1 Base+ | `44.6 ms/frame` |
| official PyTorch bf16 Base+ | `42.666688 ms/frame` |

So the goal is still open. Fusion work should continue only if the in-producer
q8_1 DS4 side-output can satisfy the exactness gate above; otherwise the next
implementation effort should move to q4_1 MMQ body scheduling/arithmetic and
D56 FlashAttention no-fixup writeback.

Evidence:

- `outputs/add-norm-preserve-output-fusion-20260517/profile/run.log`
- `outputs/add-norm-preserve-output-fusion-20260517/profile-nofusion/run.log`
- `outputs/current-baseplus-q4_1-head-refresh-20260517/summary.json`

### 2026-05-17: Rejected D56 no-fixup direct epilogue

A narrow D56 FlashAttention epilogue probe bypassed the shared-memory
VKQ-combine/writeback path for `no_mask_no_sinks`, `ncols2 == 1`, `np == 1`,
and non-fixup tiles. The probe wrote normalized `VKQ_C` fragments directly to
the f32 output for the existing `DV_DST=56` direct-output path and left all
stream-k/fixup/masked paths on the current implementation.

The probe preserved the standalone D56 parity gates, but did not improve the
global representative shape:

| Shape | Baseline | Direct epilogue probe | Result |
| --- | ---: | ---: | --- |
| D56 global, `N=4096`, sampled queries | `1.039361 ms` | `1.040105 ms` | no win |
| D56 window, `N=196`, full queries | `0.199368 ms` baseline sequential run | probe parity passed | no promoted speed evidence |

The patch was removed. The result suggests the existing shared-memory
epilogue is not the dominant D56 bottleneck for the current Blackwell path, or
that direct stores lose enough memory coalescing to cancel the saved shared
memory traffic. D56 work should not revisit this exact no-fixup direct
epilogue without Nsight evidence showing epilogue stalls as the limiting
factor.

Evidence:

- `outputs/fattn56-current-refresh-20260517/global.log`
- `outputs/fattn56-current-refresh-20260517/window_tol004.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/global.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/window.log`

### 2026-05-17: Rejected q4_1 full-tile writeback bounds specialization

A q4_1 MMQ body probe removed the remaining `j > j_max` writeback predicate
only for the existing full-tile fastpath. This was intentionally narrower than
the rejected X-cap and `MMQ_ITER_K` probes: it kept the same tile sizes, same
accumulation order, and same q8_1 activation layout, and only compiled out a
predicate for tiles already proven to cover all `mmq_x` columns.

The standalone q4_1 two-stage MLP check passed, but the paired microbench did
not improve:

| Mode | Drop-first mean-of-means |
| --- | ---: |
| current baseline | `0.288698 ms` |
| full-tile skip `j` check probe | `0.289158 ms` |

The patch was removed. This rules out another small control-flow cleanup in
the q4_1 MMQ writeback path; the remaining q4_1 body work needs to affect the
MMA/load arithmetic itself, not just edge predicates.

Evidence:

- `outputs/mmq-q4_1-body-current-refresh-20260517/summary_drop_first.json`
- `outputs/mmq-q4_1-skip-j-check-probe-20260517/check.log`
- `outputs/mmq-q4_1-skip-j-check-probe-20260517/summary_drop_first.json`

### 2026-05-17: Rejected q4_1 MMA FMA fold

A second q4_1 MMQ body probe changed the CUDA-side `q8_1_q8_1_mma`
accumulation from two additions:

`sum += scale * C; sum += base`

to a single `fmaf(scale, C, base)` before adding into `sum`. This targeted the
actual q4_1 Blackwell MMA arithmetic path, unlike the previous writeback
predicate cleanup. The local q4_1 two-stage check stayed within the existing
tolerance, but the microbench regressed clearly:

| Mode | Drop-first mean-of-means |
| --- | ---: |
| current baseline | `0.288698 ms` |
| `fmaf(scale, C, base)` probe | `0.299682 ms` |

The patch was removed. This confirms that the compiler's existing instruction
selection for the separate multiply/add sequence is better for this kernel, or
that the fused expression increases dependency/register pressure enough to
lose. Do not reintroduce this fold without lower-level SASS evidence.

Evidence:

- `outputs/mmq-q4_1-body-current-refresh-20260517/summary_drop_first.json`
- `outputs/mmq-q4_1-fmaf-probe-20260517/check.log`
- `outputs/mmq-q4_1-fmaf-probe-20260517/summary_drop_first.json`

### 2026-05-17: Rejected q4_1 side-output epilogue

An experimental q4_1 MMQ epilogue wrote a q8_1 DS4 side-output while writing
the fused `MUL_MAT + bias + GELU` f32 destination. The intent was to let the
following q4_1 consumer reuse the side-output and skip the standalone
`quantize_mmq_q8_1` launch.

The side-output satisfied the byte gate for the representative 4096-column
stream-k shape, but it was a performance loss. The earlier 128-column
diagnostic failed because that smaller shape used stream-k fixup; the
side-output only matched the standalone quantizer for complete producer tiles
that did not need fixup.

| Mode | Result |
| --- | --- |
| prequant cache, side-output disabled | `bad=0/57344`, `max_abs=0.207466185` |
| side-output, 128-column fixup shape | `bad=7801/57344`, `max_abs=0.703251243` |
| side-output, 4096-column stream-k `fixup=0` shape | `diff_bytes=0` versus `quantize_mmq_q8_1_cuda` |
| side-output, 128-column forced non-stream-k diagnostic | `diff_bytes=0`, `bad=0/57344`, `max_abs=0.015057832` |

Normal prequant behavior remains unchanged and the side-output is not used by
default. The paired 4096-column microbench showed that even when byte-correct,
the epilogue side-output was slower than both the default path and the
post-quant cache path:

| Mode | Drop-first mean-of-means |
| --- | ---: |
| default | `0.296401 ms` |
| prequant cache, side-output disabled | `0.297763 ms` |
| side-output enabled | `0.367499 ms` |

The side-output saves the standalone quantization launch, but the extra
writeback work in the already register-heavy q4_1 MMQ epilogue costs more than
the launch it removes. The code was removed from the hot kernel after a follow
up measurement showed that even the disabled side-output plumbing inflated the
default microbench from the current `0.290294 ms` range toward `0.296 ms`.
A viable fusion would need a lower-pressure layout, a separate producer
epilogue that does not inflate the hot MMQ body, or a consumer that can read a
cheaper side layout directly.

Evidence:

- `outputs/mmq-side-prequant-probe-20260517/prequant-no-side-check.log`
- `outputs/mmq-side-prequant-probe-20260517/side-enabled-fixed-check.log`
- `outputs/mmq-side-prequant-probe-20260517/side-debug-nostreamk-check.log`
- `outputs/mmq-side-prequant-probe-20260517/side-debug-4096-streamk.log`
- `outputs/mmq-side-prequant-probe-20260517/side-paired-streamk/summary_with_side_drop_first.json`
- `outputs/mmq-side-reverted-hotpath-20260517/check.log`
- `outputs/mmq-side-reverted-hotpath-20260517/paired/summary_drop_first.json`

### 2026-05-17: Opt-in q4_1 activation+prequant fusion kernel

A safer follow-up avoided changing the q4_1 MMQ hot epilogue. Instead, when
`GGML_CUDA_ENABLE_MMQ_ACT_PREQUANT_FUSION=1` is set, the producer MMQ writes
the bias-added f32 output without activation, and a separate q8_1 DS4
prequantization kernel applies GELU in-place while producing the cached q8_1
activation for the next q4_1 MMQ consumer.

This keeps the MMQ kernel from growing, and the 128-column two-stage parity
check passes:

| Mode | Check result |
| --- | --- |
| opt-in activation+prequant fusion | `bad=0/57344`, `max_abs=0.015057832` |
| normal prequant path | `bad=0/57344`, `max_abs=0.207466185` |

The performance result is still negative for the representative 4096-column
Hiera MLP shape, even after changing the activation launch to static template
specializations:

| Mode | Drop-first mean-of-means |
| --- | ---: |
| default, no prequant cache | `0.288722 ms` |
| normal prequant cache | `0.292009 ms` |
| opt-in activation+prequant fusion | `0.333888 ms` |

The opt-in path is therefore kept only as an experiment and is not enabled by
default. The likely reason is that moving GELU out of MMQ makes the standalone
quantization kernel compute the expensive activation for every f32 value while
still writing both f32 and q8_1 outputs; the saved MMQ epilogue work is smaller
than the extra post kernel cost. The next fusion attempt should target a
different consumer contract or eliminate a whole graph operation, not merely
move activation into the q8_1 quantization pass.

A follow-up sweep varied the opt-in activation quantizer column grouping and
briefly tried a contiguous-only vector-load/store variant. The result stayed
negative: `GGML_CUDA_MMQ_Q8_1_ACT_WARP_COLS=4/8/16` all measured about
`0.326-0.337 ms` for the opt-in path versus about `0.289 ms` for the default
path. The contiguous-only variant was removed after measurement because it did
not change the conclusion and only added code complexity.

Evidence:

- `outputs/mmq-act-prequant-fusion-20260517/enabled_check.log`
- `outputs/mmq-act-prequant-fusion-20260517/static_enabled_check.log`
- `outputs/mmq-act-prequant-fusion-20260517/static-enabled-paired/summary_drop_first.json`
- `outputs/mmq-act-prequant-fusion-20260517/default-disabled-paired/summary_drop_first.json`
- `outputs/mmq-act-prequant-fusion-cols-sweep-20260517/cols4/summary_drop_first.json`
- `outputs/mmq-act-prequant-fusion-cols-sweep-20260517/cols8/summary_drop_first.json`
- `outputs/mmq-act-prequant-fusion-cols-sweep-20260517/cols16/summary_drop_first.json`
- `outputs/mmq-act-prequant-fusion-contiguous-20260517/cols4/summary_drop_first.json`
- `outputs/mmq-act-prequant-fusion-contiguous-20260517/cols8/summary_drop_first.json`
- `outputs/mmq-act-prequant-fusion-contiguous-20260517/cols16/summary_drop_first.json`

### 2026-05-17: Fusion kernel direction after activation+prequant rejection

The current fusion boundary is now clear: post-MMQ f32 activation plus q8_1
prequantization is not a useful boundary for Hiera Base+ q4_1. It preserves
parity, but it only moves GELU and quantization into a later kernel and does
not remove enough memory traffic or MMQ body work.

The next fusion work should be constrained to one of these two higher-leverage
paths:

| Candidate | Scope | Acceptance requirement |
| --- | --- | --- |
| D56 FATTN no-fixup epilogue | `DKQ=64`, `DV=56`, `DV_DST=56`, `np==1`, no mask/sinks, non-fixup tiles only | exact FATTN parity, then lower `GGML_CUDA_PROFILE_FATTN56` `mma_ms` for `Q[56,4096,8,1]` and `Q[56,196,8,25]` |
| q4_1 MMQ shape-gated body specialization | `GGML_TYPE_Q4_1`, no ids, Base+ Hiera `ncols_max=4096`, full-tile/no-fixup only | two-stage MMQ parity, then lower paired mean for the representative MLP microbench |

Rejected boundaries should not be repeated unless the consumer contract changes:
prequant-cache defaulting, post-MMQ side-output q8_1 writeback, activation in
the standalone q8_1 quantizer, and launch-bounds register capping have all lost
on the current workload.

A narrow D56 no-fixup direct epilogue probe was implemented and removed. It
wrote normalized `VKQ_C` fragments directly from registers for
`no_mask_no_sinks && !needs_fixup && !is_fixup && np==1 && DV_DST==56`.
The probe preserved parity, but did not show a useful speed win:

| Shape | Result |
| --- | --- |
| `D=56 run_D=64 N=4096 heads=8 batch=1` | `bad=0/4032`, `mean_ms=0.697379` |
| `D=56 run_D=64 N=196 heads=8 batch=25` | `bad=0/100800`, `mean_ms=0.199070` |

The current shared-memory epilogue is not the remaining dominant cost for these
cases; future FATTN work needs to change the MMA body schedule or reduce the
number of tiles rather than only bypassing the final shared-memory staging.

Evidence:

- `outputs/fattn56-direct-epilogue-probe-20260517/parity-4096.log`
- `outputs/fattn56-direct-epilogue-probe-20260517/parity-196x25.log`

### 2026-05-17: Rejected q4_1 launch-bounds min-blocks 2

After removing the side-output epilogue from the hot kernel, q4_1 ptxas was
refreshed. The representative `mmq_x=96` kernel still uses 255 registers but
does not spill. A narrow q4_1-only launch-bounds probe changed the CUDA Volta+
`mul_mat_q` min-blocks value from `1` to `2` only for `GGML_TYPE_Q4_1`.

The standalone two-stage check passed, but the paired microbench regressed
badly:

| Mode | Drop-first mean-of-means |
| --- | ---: |
| current after side-output removal | `0.290294 ms` |
| q4_1 launch-bounds min-blocks 2 probe | `0.383025 ms` |

The patch was removed. The result confirms that forcing extra occupancy by
register capping is not viable for this q4_1 Blackwell MMQ body; it likely
pushes useful accumulator/local state out of registers and loses more than the
extra residency can recover.

Evidence:

- `outputs/ptxas-mmq-hotpath-reverted-20260517/q4_1-summary.json`
- `outputs/mmq-q4_1-launchbounds2-probe-20260517/check.log`
- `outputs/mmq-q4_1-launchbounds2-probe-20260517/paired/summary_drop_first.json`

### 2026-05-17: q8-only producer fusion target

The fusion target was re-scoped after the activation+prequant and side-output
epilogue rejections. The earlier side-output probe wrote both the normal f32
producer output and an extra q8_1 DS4 side-output, so it increased pressure in
the already-hot q4_1 MMQ epilogue. The only producer/consumer fusion boundary
that is still worth pursuing is stricter: when the `fc1 + bias + GELU` output
has exactly one real consumer, the producer may emit only the q8_1 DS4
activation cache required by the following `fc2` MMQ and skip the intermediate
f32 writeback.

The candidate summarizer was updated so MMQ-only profile logs, without
`GGML_CUDA_PROFILE_NODES`, still report `fc1 -> fc2` candidates by tensor name
and shape. On the current two-frame Base+ q4_1 profile, the drop-first
candidate set is:

| Metric | Value |
| --- | ---: |
| candidate rows | `63` |
| consumer q8_1 quant upper bound | `2.473312 ms` |
| consumer total time covered | `9.683872 ms` |
| f32 producer write volume that q8-only fusion could elide | `1876 MiB` |

This is not yet an accepted optimization. The numbers are an upper bound, not a
claim of speedup: the q8-only path must still prove that the producer output is
not read by another node, that the q8_1 side-output does not slow the producer
MMQ body enough to cancel the saved consumer quantization, and that graph
allocation/lifetime remains valid when the f32 tensor is intentionally
unmaterialized.

Acceptance criteria for a q8-only producer fusion:

1. Graph analysis must prove one real downstream consumer, `MUL_MAT` with
   q4_1 weight, `src1` equal to the producer output, no `ids`, and no later
   tensor read that requires the f32 output.
2. The CUDA path must be opt-in until measured, for example behind
   `GGML_CUDA_ENABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1`.
3. The producer kernel must not write the normal f32 output in the accepted
   q8-only path; otherwise it repeats the rejected side-output epilogue shape.
4. Parity must pass on the two-stage q4_1 MLP microbench and the Base+ tracking
   comparison before any app-level timing is trusted.
5. The paired Base+ timing must beat the current q4_1 accepted fastpath by more
   than run-to-run noise, not merely reduce `GGML_CUDA_PROFILE_MMQ` quant time.

Evidence:

- `outputs/fusion-kernel-q8-only-direction-20260517/gelu-side-output-candidates.json`
- `outputs/fusion-kernel-q8-only-direction-20260517/gelu-side-output-candidates-all.json`

The CUDA graph evaluator now also has a behavior-preserving diagnostic for this
specific condition. With `GGML_CUDA_PROFILE_MMQ_Q8_ONLY_CANDIDATES=1`, the
`MUL_MAT + bias + GELU` fusion path logs rows where the GELU output has a direct
q4_1 `MUL_MAT` consumer and no later direct consumer of the f32 tensor. This
does not enable q8-only execution; it only proves the graph-side liveness
condition before a kernel change is attempted.

Probe results:

| Probe | Result |
| --- | --- |
| two-stage q4_1 MLP microbench | `bad=0/57344`, `max_abs=0.207466185`, 1 q8-only candidate |
| Base+ q4_1, 3 frames | 68 logged rows, 24 unique producer/consumer weights |

The Base+ candidate shapes match the earlier MMQ-profile upper bound:

| Producer output -> consumer output | Logged rows |
| --- | ---: |
| `1792,4096,1,1 -> 448,4096,1,1` | `48` |
| `896,16384,1,1 -> 224,16384,1,1` | `9` |
| `3584,1024,1,1 -> 896,1024,1,1` | `9` |
| `1024,4096,1,1 -> 256,4096,1,1` | `2` |

Evidence:

- `outputs/mmq-q8-only-candidate-probe-20260517/check.stdout`
- `outputs/mmq-q8-only-candidate-probe-20260517/check.stderr`
- `outputs/mmq-q8-only-candidate-probe-20260517/baseplus-q4_1/benchmark.stdout`
- `outputs/mmq-q8-only-candidate-probe-20260517/baseplus-q4_1/candidates.log`

The next preparatory patch added an explicit
`GGML_CUDA_ENABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1` target flag in the CUDA
context. It still does not change kernel behavior, but it proves that the graph
candidate state reaches `ggml_cuda_mul_mat_q` and can be used as the switch for
the later producer-side q8_1 DS4 write path. With the flag enabled, the
two-stage q4_1 MLP microbench still matches the disabled path
(`bad=0/57344`, `max_abs=0.207466185`) and logs both the candidate and target
rows.

Evidence:

- `outputs/mmq-q8-only-state-probe-20260517/check.stdout`
- `outputs/mmq-q8-only-state-probe-20260517/check.stderr`

Prototype result:

The first real q8-only producer kernel writes the activation output directly to
the q8_1 MMQ cache and skips the producer f32 global write. It is still opt-in
behind `GGML_CUDA_ENABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1`. A bug in the first
prototype decoded a 3D launch grid as a flattened direct-tile grid; that caused
cache tiles to be written out of order and produced `bad=5027/57344` in the
two-stage microbench. After fixing the grid decode, the same microbench passes:

| Run | Result |
| --- | ---: |
| baseline two-stage q4_1 MLP | `0.039462 ms`, `bad=0/57344`, `max_abs=0.207466185` |
| q8-only producer fusion | `0.033919 ms`, `bad=0/57344`, `max_abs=0.015057832` |

This is a useful narrow win, but it is not accepted for the full Hiera path.
On the larger Base+ MLP shapes, the extra shared-memory f32 tile forces a
smaller `mmq_x`; the MMQ body slowdown cancels the saved q8 quantization. A
large-column microbench for `896x16384 -> 224x16384` measured `0.984880 ms`
baseline versus `0.993012 ms` with q8-only enabled. A 5-frame Base+ q4_1 smoke
before gating also regressed from `46.2 ms/frame` to `47.8 ms/frame`.

The second prototype removed the shared-memory f32 tile. It quantizes q8_1 DS4
directly from the MMA accumulator fragments with warp reductions, so it keeps
the normal MMQ shared-memory footprint and normal `mmq_x` selection. This fixed
the high-column regression:

| Probe | Disabled / baseline | Direct-fragment q8-only |
| --- | ---: | ---: |
| `448x128 -> 1792x128 -> 448x128` two-stage q4_1 | `0.039462 ms` | `0.032625 ms` |
| `896x16384 -> 224x16384` two-stage q4_1 | `0.986491 ms` | `0.798989 ms` |
| Base+ q4_1, 5-frame app median | `46.3-47.1 ms/frame` | `45.5-45.6 ms/frame` |

The q8-only producer fusion is now default-on for the proven CUDA MMA q4_1
case and can be disabled with
`GGML_CUDA_DISABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1`. The old opt-in
`GGML_CUDA_ENABLE_MMQ_Q8_ONLY_PRODUCER_FUSION` is no longer needed for the
default path.

A follow-up probe tried to widen the same fusion policy to q4_0 producer and
consumer MMQ rows. That is not acceptable with the current direct-fragment
writer: the disabled q4_0 two-stage path passed (`bad=0/3670016`,
`max_abs=0.00268878043`), while the widened q8-only path produced
`bad=692960/3670016` and `max_abs=1.01212192`. The broadening was removed.
This means the q8-only fusion stays q4_1-only until q4_0 gets its own
accumulator-layout-specific side-output writer and parity proof.

After this change, the q4_1 producer/consumer MMQ hotspot is no longer the main
PyTorch-gap surface in the short profile. D56 FATTN body time dominates again.
A small FATTN config probe changed the `DKQ=64,DV=64,ncols=64` Ampere/Blackwell
MMA config from `nbatch_fa=64` to `128`. It preserved D56 parity and improved
the 2-frame FATTN56 profile from `9.417536 ms` to `9.160640 ms` total, mainly
on `Q[56,4096,8,1]` (`0.6092544 ms` to `0.5724032 ms` per effective row).
This is accepted as a small positive scheduling cleanup, not as a complete
PyTorch-speed closer.

A follow-up `ncols=64` config probe tried widening `nbatch_V2` from `32` to
`64`. This cannot build with the current MMA body because the kernel requires
`DV % (2*nbatch_V2) == 0`, and `DV=64` makes `nbatch_V2=64` invalid. The probe
was reverted before runtime measurement.

The remaining CUDA node profile also shows nonzero IM2COL cost, especially the
1x1 `[256,256,256]` transpose-like row. A tiled shared-memory transpose kernel
for the exact `KW=KH=1,stride=1,pad=0,dilation=1` path preserved 10-frame
tracking parity against the disabled path (`mask_hash_equal_rows=10/10`,
`min_bbox_iou=1.0`, `max_score_abs_delta=0.0`), but it did not improve runtime:
the single A/B smoke measured `88.6 ms/frame` with the specialized path versus
`88.0 ms/frame` with the existing IM2COL kernel. The probe was removed. This
suggests the current IM2COL row is not a useful PyTorch-gap target unless it is
eliminated at the graph/matmul boundary rather than replaced by a standalone
transpose launch.

The matched Python-summary refresh after these changes still reports C++ q4_1
Base+ at `44.9 ms/frame` versus official PyTorch bf16 Base+ at
`42.666687668922044 ms/frame`
(`outputs/q8-fattn128-vs-python-20260517/q4_1/summary.json`). The goal is
therefore still open; the remaining gap is now mostly D56 FATTN body plus
non-Hiera tracking/decoder work rather than the q4_1 producer-consumer MLP
surface.

A current-default repeat after the q8-only MMQ and FATTN56 cleanup confirms
that the PyTorch gap is still real, not just a single-run artifact. Five C++
q4_1 Base+ 1024 runs against the same saved official PyTorch Base+ bf16 row
measured:

| C++ q4_1 track ms/frame | Mean | Median | Stdev | PyTorch bf16 Base+ |
| --- | ---: | ---: | ---: | ---: |
| `43.7, 44.0, 44.1, 43.8, 45.0` | `44.12` | `44.0` | `0.52` | `42.666687668922044` |

The remaining mean gap is about `1.45 ms/frame` (`3.4%`). A fusion that removes
only the largest q4_1 MMQ activation quantize row has a measured standalone
upper bound of about `0.31 ms` per occurrence at `k=448,cols=65536`; it is useful
but not sufficient by itself. Closing the gap now needs either a broader set of
MMQ quantize-boundary removals or a D56/MMQ body improvement.

The official PyTorch row was also refreshed three times rather than relying on
the older saved Python result:

| Run | C++ q4_1 Base+ | Official PyTorch bf16 Base+ | C++ minus PyTorch |
| ---: | ---: | ---: | ---: |
| 1 | `44.0` | `42.49299366751479` | `1.50700633248521` |
| 2 | `44.7` | `42.819043000539146` | `1.8809569994608566` |
| 3 | `45.1` | `43.025411665439606` | `2.074588334560394` |

The refreshed means are C++ `44.6 ms/frame` and PyTorch
`42.77914944449785 ms/frame`, leaving `1.82 ms/frame` (`4.3%`) to close. This
confirms the remaining target is not measurement noise. The required next
optimization must be a multi-surface change: either remove several MMQ
quantize-boundary costs at once, or reduce D56/MMQ body time, because one
standalone q8 quantizer replacement is not enough.

### 2026-05-17: Rejected small-shape q4_1 warp-column quantizer

The q4_1 warp-column q8_1 quantizer is currently used for wide activations
(`ne1 >= 16384`). A probe tested whether the same layout should also cover the
smaller Hiera/decoder MMQ rows with `ne0 <= 256` and `ne1 >= 4096`.

Standalone quantizer timing looked promising for the smallest rows:

| Shape | Standard quantizer | Warp-column candidate | Ratio |
| --- | ---: | ---: | ---: |
| `k=2048,cols=4096` | `0.0739955 ms` | `0.0755344 ms` | `0.980x` |
| `k=256,cols=4096` | `0.00484768 ms` | `0.00361984 ms` | `1.339x` |
| `k=64,padded_k=128,cols=4100` | `0.00591328 ms` | `0.00282752 ms` | `2.091x` |

However, full MMQ microbenchmarks did not preserve that win. The `k=256` row
regressed (`0.018325 ms` enabled versus `0.017928 ms` disabled), while `k=64`
improved only by about `0.000095 ms`. Three app-level paired runs also failed
to show a win:

| Enabled | Disabled |
| ---: | ---: |
| `44.3` | `44.0` |
| `44.1` | `44.5` |
| `44.6` | `44.2` |

The enabled mean was `44.33 ms/frame` versus disabled `44.23 ms/frame`. The
probe was reverted. The current wide-only threshold remains the right default;
the small-shape quantizer replacement is not a PyTorch-gap closer.

### 2026-05-17: Rejected FATTN56 native-V opt-in recheck

`GGML_CUDA_ENABLE_FATTN56_NATIVE_V=1` was rechecked on the current branch. The
standalone D56 parity benches remained valid and showed only a tiny local
change:

| Shape | Default | Native V opt-in |
| --- | ---: | ---: |
| `D=56,N=4096,heads=8,batch=1` | `0.699730 ms`, `bad=0/4032` | `0.698693 ms`, `bad=0/4032` |
| `D=56,N=196,heads=8,batch=25` | `0.209416 ms`, `bad=0/100800` | `0.208937 ms`, `bad=0/100800` |

App-level paired runs rejected the opt-in:

| Default | Native V opt-in |
| ---: | ---: |
| `43.7` | `47.3` |
| `44.3` | `46.9` |
| `44.4` | `46.5` |

The native-V mean was `46.9 ms/frame` versus default `44.13 ms/frame`, a
`2.77 ms/frame` regression. Keep native-V default-off. This also reinforces that
standalone FATTN microbench wins below about one millisecond must be validated
with paired app runs before being treated as useful.

### 2026-05-17: Current stage gap and rejected q4_1 cuBLAS fallback

A fresh C++ stage profile confirms that the remaining PyTorch gap is in Hiera
compute, not propagation:

| Area | C++ current | Official PyTorch profile | Gap |
| --- | ---: | ---: | ---: |
| Hiera image encode / `forward_image` | `30.088 ms` steady | `28.094 ms` mean | `+1.995 ms` |
| Propagation / `track_step` | `9.108 ms` mean | `13.108 ms` mean | C++ faster |
| Hiera input upload | `0.504 ms` steady | n/a | secondary |
| Hiera graph alloc | `0.177 ms` steady | n/a | secondary |

The hot q4_1 MMQ row was also tested against a force-cuBLAS build to see whether
the dominant `q4_1[448,112] x f32[448,65536]` surface should be routed away from
MMQ. It should not:

| Build | Hot row mean |
| --- | ---: |
| `GGML_CUDA_FORCE_CUBLAS=ON` | `0.605889 ms` |
| Current MMQ default | `0.451633 ms` |

cuBLAS fallback is about `34%` slower for this shape. The remaining Hiera gap
therefore needs a real MMQ body/fusion improvement or a D56 attention body
improvement, not a backend routing change.

### 2026-05-17: Fusion-kernel refresh after q8-only MMQ

The fusion-kernel direction was rechecked after the q4_1 q8-only MMQ producer
fusion became default. The useful defaultable fusion work is now limited to
kernel-body specializations that remove real work from hot kernels, not
standalone post kernels:

1. Keep q4_1 q8-only MMQ producer fusion as the accepted producer/consumer
   boundary. It skips the intermediate f32 producer write and emits the q8_1
   DS4 consumer cache directly from accumulator fragments.
2. Keep D56 no-mask/no-sink FlashAttention specialization as the accepted FATTN
   fusion-body cleanup. It compiles away unused mask/sink branches and shared
   mask storage for SAM2 Hiera D56 calls.
3. Do not promote ADD-to-MMQ prequant, activation+prequant, ConvTranspose+bias,
   or standalone IM2COL-transpose probes without a paired app-level win; they
   either lost runtime or were too small to move the remaining PyTorch gap.

The current D56 no-mask specialization was rebuilt and remeasured:

| Check | Default | Disabled with `GGML_CUDA_DISABLE_FATTN56_NOMASK_FAST=1` |
| --- | ---: | ---: |
| D56 `N=4096,heads=8,batch=1` parity bench | `bad=0/4032`, `0.690747 ms` | `bad=0/4032`, `0.691635 ms` |
| D56 `N=196,heads=8,batch=25` parity bench | `bad=0/100800`, `0.208267 ms` | `bad=0/100800`, `0.222653 ms` |
| Base+ q4_1 1024 bbox-only, 3 paired app runs | `43.633333 ms/frame` | `44.166667 ms/frame` |

This does not close the official PyTorch gap by itself, but it is a real
fusion-kernel cleanup and should stay enabled. The next fusion work should be a
new hot-kernel body specialization with byte/parity gates first, not another
post-op cache or side-output that keeps all original memory traffic.

A follow-up config probe changed the D56 `ncols=64` MMA config from
`nbatch_fa=64` to `128`, targeting the q-pool shapes that still show up in the
current FATTN56 profile. The probe preserved parity in the standalone FATTN
checks, and the synchronized FATTN profile improved modestly:

| Profile | Current default | Probe |
| --- | ---: | ---: |
| FATTN56 drop-first total | `15.395584 ms` | `14.979104 ms` |
| `Q[56,64,2,1024]` mean | `0.376013 ms` | `0.368589 ms` |
| `Q[56,16,4,1024]` mean | `0.270688 ms` | `0.268294 ms` |

The normal 10-frame bbox-only app smoke did not show a matching win
(`43.5, 44.3, 43.7 ms/frame`, mean `43.83`, versus the preceding default smoke
mean `43.63`). The probe was therefore reverted. This is below the threshold
for a default change; future D56 work needs a larger kernel-body change than
retuning only this `nbatch_fa` row.

The refreshed current hotspot profile after q8-only MMQ shows the remaining
Hiera work has shifted back to MMQ body/activation quantize and D56 FATTN:

| Area | Drop-first synchronized profile signal |
| --- | ---: |
| MMQ total | `4.055136 ms` across effective MMQ rows |
| MMQ activation quantize | `2.082240 ms` |
| FATTN56 total | `15.395584 ms` |
| Top Hiera node signatures | MLP expansion/projection MMQ, window/global D56 FATTN, patch/projection MMQ |

The large patch/projection-like MMQ row
`q4_1[448,112] x f32[448,65536] -> f32[112,65536]` was isolated because its
activation quantization is one of the largest remaining rows. The standalone
shape uses `mmq_x=96` and is already close to the best selector choice:

| `GGML_CUDA_MMQ_X_MAX` | Mean |
| --- | ---: |
| default / `96` | `0.4518 / 0.4516 ms` |
| `104` | `0.4517 ms` |
| `112` | `0.4661 ms` |
| `120` | `0.4659 ms` |
| `128` | `0.4670 ms` |
| `64-88` | `0.470-0.478 ms` |

This rules out another selector-only MMQ tweak for that row. The next useful
MMQ work needs to reduce activation q8_1 quantization cost or MMQ body pressure
without changing the proven `mmq_x=96/104` schedule.

The same row was used to recheck the q4_1 DS4 warp-column activation quantizer.
Disabling `GGML_CUDA_MMQ_Q8_1_WARP_COL_QUANT` regressed the standalone row from
about `0.473 ms` to `0.478 ms`, so the current warp-column path is still the
right default. A probe that tried to replace the full 128-value qblocks with
`float4` loads while keeping the scalar tail path preserved parity on a smaller
check row (`bad=0/458752`), but slowed the hot row to `0.461663 ms` versus the
current `0.451840 ms` smoke. The likely cause is extra branch/address pressure
outweighing the cleaner load instruction sequence, so it was reverted.

A direct consumer-side fusion probe was also attempted for the same MMQ class:
instead of materializing `f32 -> q8_1` in global memory and then launching the
q4_1 MMQ body, the MMQ kernel quantized the `f32` activation tile into shared
memory and immediately fed the existing q4_1 dot path. This is the right
structural direction for the hot
`q4_1[448,112] x f32[448,65536] -> f32[112,65536]` row because it has only one
row tile, so the prequantized activation is not reused across multiple output
row tiles. However, the first shared-tile prototype was not accepted:

| Probe | Result |
| --- | --- |
| Default path, hot row check at `cols=32768` | `bad=0/3670016`, median around `0.35 ms` in a noisy run |
| Shared-tile direct fusion, hot row | illegal CUDA memory access |
| Shared-tile direct fusion, `k=512` padded-free control row | illegal CUDA memory access |

The failing probe was hard-disabled in code and should not be treated as an
optimization. The next version should be implemented as a narrow q4_1-only
kernel or helper with an explicit byte-layout parity test for the generated
`block_q8_1_mmq` shared tile before it is connected to the full MMQ body. It
should not be added as a broad common-header branch until the tile layout is
proven, because the prototype also made q4_1 CUDA compilation significantly
heavier.

That parity gate now exists as `sam3_mmq_q8_layout_parity`. It compares the
standard `quantize_mmq_q8_1_cuda` output against an independent DS4
warp-column q4_1 candidate over the same hot activation shapes, before any
candidate is connected to the full MMQ body:

| Shape | Cols/block | Byte mismatches | q mismatches | Max q abs diff | q diff > 1 | Max sampled dot diff | ds mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k=448,padded_k=512,cols=32768` | 4 | `29 / 18874368` | `27` | `1` | `0` | `7.45948e-05` | `2` |
| `k=512,padded_k=512,cols=32768` | 4 | `35 / 18874368` | `32` | `1` | `0` | `7.74286e-05` | `3` |
| `k=448,padded_k=512,cols=32768` | 8 | `29 / 18874368` | `27` | `1` | `0` | `7.45948e-05` | `2` |

The same tool also times the standard quantizer versus the independent
candidate kernel over 100 iterations:

| Shape | Cols/block | Standard quantizer | Candidate quantizer | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `k=448,padded_k=512,cols=32768` | 4 | `0.146924 ms` | `0.146709 ms` | `1.001x` |
| `k=512,padded_k=512,cols=32768` | 4 | `0.163635 ms` | `0.166624 ms` | `0.982x` |
| `k=448,padded_k=512,cols=32768` | 8 | `0.146523 ms` | `0.147034 ms` | `0.997x` |

The result separates layout correctness from rounding exactness: the candidate
does not show a large byte-layout break, and every q-value difference is a
single int8 step at a quantization boundary. Rechecking the candidate with the
same `float4` load style as the standard quantizer did not remove those tiny
differences. A synthetic q4_1 dot check over the differing q8 blocks bounds the
observed output-level drift to about `8e-05` for these samples. A renewed fusion
kernel should still not assume byte-exact q8 output from a separately written
quantizer. The timing also shows that replacing the standalone quantizer with
this candidate is not itself an optimization; the possible win must come from
removing the quantize launch and global q8 write/read by connecting the
layout-validated q4_1 producer directly to the consumer MMQ body. It must still
prove full two-stage MMQ and app-level parity before being defaulted. The
earlier illegal-access path should not be resurrected; the next implementation
should be a small shape-gated kernel first.

Evidence:

- `outputs/q8-only-producer-fusion-smoke-20260517/baseplus-q4_1-baseline/benchmark.stdout`
- `outputs/q8-only-producer-fusion-smoke-20260517/baseplus-q4_1-q8-only/benchmark.stderr`
- `outputs/q8-only-producer-fusion-smoke-20260517/baseplus-q4_1-q8-only/benchmark.stdout`
- `outputs/q8-only-producer-fusion-smoke-20260517/baseplus-q4_1-baseline-10/benchmark.stdout`
- `outputs/q8-only-producer-fusion-smoke-20260517/baseplus-q4_1-q8-only-gated-10/benchmark.stdout`
- `outputs/q8-only-direct-fragment-app-20260517/`
- `outputs/q8-only-direct-fragment-default-app-20260517/`
- `outputs/q8-only-direct-fragment-profile-20260517/`
- `outputs/fattn-nbatch128-probe-20260517/`
- `outputs/q8-fattn128-default-app-20260517/`
- `outputs/q8-fattn128-vs-python-20260517/q4_1/summary.json`
- `outputs/fusion-kernel-fattn-current-20260517/`
- `outputs/current-hotspots-refresh-20260517/`
- `outputs/fattn56-nbatch128-dv56-ncols64-probe-20260517/`
- `outputs/mmq-quant-hotrow-refresh-20260517/`
- `outputs/mmq-quant-float4-fullblock-probe-20260517/`
- `outputs/mmq-f32-q8-fusion-probe-20260517/`
- `outputs/mmq-q8-layout-parity-20260517/`

### 2026-05-17: Fusion continuation and state-buffer reuse check

The Hiera state tensor buffer reuse probe now has an explicit rollback gate:
`SAM3_DISABLE_HIERA_STATE_BUFFER_REUSE=1`. The second and later frames now hit
`hiera_encode_state_buffer_cache_hit`, avoiding the repeated state context and
backend-buffer allocation. The measured effect is small:

| Probe | Track/fr values | Mean |
| --- | --- | ---: |
| reuse enabled | `43.8, 43.5, 43.3, 43.4, 43.7` | `43.54 ms` |
| reuse disabled | `43.7, 43.4, 43.5, 43.9, 43.5` | `43.60 ms` |

The enabled and disabled 10-frame JSONL rows matched exactly in this smoke
check. This is safe as a small allocation cleanup, but it is not a PyTorch-gap
closing optimization.

The q4_1 q8-only producer/consumer fusion was rechecked as the main fusion
kernel route. It still provides the expected speed signal when explicitly
enabled with `GGML_CUDA_ENABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1`:

| Check | Default fusion | Disabled with `GGML_CUDA_DISABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1` |
| --- | ---: | ---: |
| two-stage q4_1 MLP, `cols=4096`, check enabled | `0.277956 ms`, `bad=0/1835008` | `0.318124 ms`, `bad=0/1835008` |
| two-stage q4_1 MLP, `cols=16384` | `1.035753 ms` | `1.300024 ms` |
| Base+ q4_1 10-frame app, 3 paired runs | `43.37 ms/frame` | `44.80 ms/frame` |

However, the app-level JSONL comparison between fusion enabled and disabled did
not match exactly: frame 0 already changed mask hash, bbox, score, and selected
score path. Direct selected-logit comparison on the first three frames showed
that this is not a tiny hash-boundary-only effect:

| Frame | Max abs low-res logit delta | Mean abs delta | Positive-threshold flips |
| --- | ---: | ---: | ---: |
| `0` | `1.460876` | `0.185534` | `1499 / 188` |
| `1` | `11.616563` | `1.831259` | `4256 / 179` |
| `2` | `9.946300` | `1.793398` | `4332 / 146` |

The microbench error remains numerically small, but mask hash parity is more
sensitive than the current `max_abs <= 0.25` MMQ gate. Therefore the fusion
route is useful and fast, but it is now opt-in again rather than default-on.
Default execution matches
`GGML_CUDA_DISABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1` on the 3-frame smoke check.

One diagnostic gate was added for this investigation:
`GGML_CUDA_DISABLE_MMQ_STREAM_K=1`. It showed that stream-k accumulation order
does affect scores, but it is not the whole q8-only difference: with stream-k
disabled for both paths, q8-only still changed bbox and selected mask path.
Global stream-k disable is also far too slow for the app smoke and is a
diagnostic only, not an optimization.

Updated fusion acceptance criteria:

1. Keep the q4_1 q8-only route as the only producer/consumer fusion target with
   a proven speed signal, but require explicit opt-in until app-level parity is
   solved.
2. Do not broaden q8-only fusion to q4_0/q8_0 or direct consumer-side fusion
   without byte/layout parity first.
3. For default-on fusion, require both a two-stage MMQ check and a 10-frame
   Base+ JSONL parity check against the disabled path. If exact hash parity is
   too strict, the replacement acceptance gate must be a documented mask-logit
   tolerance plus bbox/selected-index stability.
4. The remaining PyTorch gap should be attacked with a stricter q8-only writer
   that reproduces the post-writeback quantization more closely, or with a
   separate D56 attention body improvement. More post-kernel caches are no
   longer high-value.

Evidence:

- `outputs/state-buffer-reuse-20260517/`
- `outputs/fusion-kernel-continue-20260517/micro/`
- `outputs/fusion-kernel-continue-20260517/app/`
- `outputs/fusion-parity-logits-20260517/`
- `outputs/mmq-streamk-q8only-diagnosis-20260517/`
- `outputs/q8-only-optin-after-parity-20260517/`

After making q8-only opt-in again, the current default Base+ q4_1 10-frame
speed was remeasured:

| Run values | Mean | Saved official PyTorch Base+ bf16 | Gap |
| --- | ---: | ---: | ---: |
| `45.9, 44.9, 46.7 ms/frame` | `45.83 ms/frame` | `42.67 ms/frame` | `+3.17 ms/frame` |

This means the active goal is not complete. The next accepted optimization must
come from a parity-preserving D56 attention body change, MMQ body reduction, or
a stricter q8-only writer that passes the app-level gate above.

Evidence:

- `outputs/current-default-after-q8-optin-20260517/summary.json`

The current hotspot attribution after the q8-only rollback confirms that
FATTN56 and MMQ body/quantize are again the main surfaces:

| Area | Drop-first/profile signal |
| --- | ---: |
| FATTN56 total | `57.69 ms` over 232 effective rows |
| FATTN56 `Q[56,196,8,25]` | `18.42 ms` total, `0.1548 ms` mean |
| FATTN56 `Q[56,4096,8,1]` | `18.04 ms` total, `0.6220 ms` mean |
| MMQ quantize total | `17.30 ms` over 427 rows |
| MMQ total | `97.11 ms` over 427 rows, with first-row outliers included |

An env-gated attempt to cache standard `src1 -> q8_1` MMQ quantization inside a
graph was tested and rejected before keeping code. It crashed the app run in a
later cuBLASLt fused path, likely because holding extra pool allocations across
subsequent graph nodes perturbed workspace allocation/lifetime. The patch was
reverted; any future cache attempt needs a dedicated graph-liveness analysis and
separate storage lifetime, not reuse of the existing producer prequant cache.

Evidence:

- `outputs/current-hotspots-after-q8-optin-20260517/mmq-summary.json`
- `outputs/current-fattn56-after-q8-optin-20260517/fattn56-summary-drop-first.json`
- `outputs/mmq-src1-q8-cache-20260517/cache/stderr.log`

`GGML_CUDA_DISABLE_FATTN_STREAM_K=1` was rechecked after the q8-only rollback.
The synchronized FATTN56 profile looked better, reducing drop-first FATTN56
total from `57.13 ms` to `54.67 ms`, mostly in the `Q[56,196,8,25]` window
bucket. But it changed app JSONL numerics and did not win in normal unprofiled
paired timing:

| Mode | Track ms/frame values | Mean | Median |
| --- | --- | ---: | ---: |
| default | `45.3, 44.7, 45.4, 44.8, 45.1` | `45.06` | `45.1` |
| `GGML_CUDA_DISABLE_FATTN_STREAM_K=1` | `45.1, 45.1, 45.1, 45.3, 46.3` | `45.38` | `45.1` |

The JSONL drift was much smaller than the q8-only drift: selected index stayed
the same for all 10 frames, bbox changed on one frame by one grid step, and mask
hash changed on frame 0 only. Since the unprofiled app speed did not improve,
this remains a diagnostic toggle rather than a default optimization.

Evidence:

- `outputs/fattn-streamk-after-q8-optin-20260517/summary.json`
- `outputs/fattn-streamk-after-q8-optin-20260517/jsonl-diff.json`
- `outputs/fattn-streamk-speed-20260517/summary.json`

The combined upper-bound run with both non-default toggles
`GGML_CUDA_ENABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1` and
`GGML_CUDA_DISABLE_FATTN_STREAM_K=1` still did not beat the saved official
PyTorch Base+ bf16 timing:

| Mode | Track ms/frame values | Mean | Saved official PyTorch Base+ bf16 | Gap |
| --- | --- | ---: | ---: | ---: |
| q8-only + FATTN no stream-k | `44.2, 43.9, 43.8, 44.5, 43.7` | `44.02` | `42.67` | `+1.35` |

This is not an accepted optimization because the q8-only producer path has
app-level parity drift. It is useful only as a ceiling estimate: the current
shallow fusion/toggle space is not enough to finish the active performance
goal.

Evidence:

- `outputs/q8only-fattn-nostream-upperbound-20260517/summary.json`

The parity-safe MMQ prequant cache was also rechecked as a possible default
fusion direction. It preserved 10-frame JSONL parity (`diff_rows=0`) but did not
show a stable speed win when default-on was compared against explicit disable:

| Mode | Track ms/frame values | Mean | Median |
| --- | --- | ---: | ---: |
| default-on prequant cache | `45.0, 44.9, 44.9, 44.8, 45.4` | `45.00` | `44.9` |
| disabled prequant cache | `45.0, 44.6, 45.2, 44.4, 45.1` | `44.86` | `45.0` |

The attempted default-on patch was reverted. Keep this path env-gated unless a
future graph-specific gate shows a real app-level win.

Evidence:

- `outputs/mmq-prequant-parity-20260517/jsonl-diff.json`
- `outputs/mmq-prequant-default-on-20260517/summary.json`

Next fusion-kernel direction:

1. Preserve standard stream-k accumulation for producer MMQ. The previous
   q8-only direct writer is too numerically different because it bypasses the
   default stream-k/fixup path.
2. Prefer shape-gated q4_1 MMQ body reductions for the Base+ Hiera MLP hot
   shapes (`cols=4096`, `rows=448/1792`) before adding new cache lifetimes.
3. For FATTN56, the remaining high-value kernel work is a no-fixup D56 epilogue
   direct writeback that avoids shared-memory staging only for
   `no_mask_no_sinks && !needs_fixup && !is_fixup`. It must leave stream-k fixup
   tiles on the existing path.

A tiny q4_1 MMQ body cleanup that only hoisted the `y_qs`/`y_ds` references out
of the innermost row loop was tested and reverted. It preserved the microbench
check, but did not improve the hot two-stage q4_1 row:

| Variant | q4_1 two-stage no-cont mean | q4_1 two-stage cont mean |
| --- | ---: | ---: |
| baseline | `0.318302 ms` | `0.369978 ms` |
| hoisted references | `0.318685 ms` | `0.371781 ms` |

This confirms that the already-compiled q4_1 body is not waiting on that simple
address-hoist cleanup. The next MMQ patch needs a real body specialization, not
only pointer/reference reshaping.

Evidence:

- `outputs/fusion-kernel-direction-20260517/`
- `outputs/mmq-q41-y-ref-hoist-20260517/`

For the D56 FATTN direct-epilogue direction, baseline parity/timing was captured
before editing the kernel body. The global-like `N=4096` sampled-query row is a
strict low-drift gate, while the dense window row already exceeds the old
`0.006` tolerance in the current implementation and should be checked against
baseline regression rather than that absolute threshold:

| Shape | Current max abs | Mean abs | Mean time |
| --- | ---: | ---: | ---: |
| `D=56 run_D=64 N=4096 heads=8 batch=1 sampled_q=9` | `0.0051307` | `0.0010593` | `0.717208 ms` |
| `D=56 run_D=64 N=196 heads=8 batch=25 sampled_q=196` | `0.0320835` | `0.0051290` | `0.217935 ms` |

Evidence:

- `outputs/fattn56-direct-epilogue-baseline-20260517/`

A narrow D56 no-mask direct epilogue was then added for
`DKQ=64 && DV<=64 && ncols2=1 && np=1 && !needs_fixup && !is_fixup`. It writes
the already-computed `VKQ_C` fragments directly to the destination instead of
staging them through shared memory for the no-fixup case; all fixup and
multi-warp combine cases stay on the original epilogue. The change is exact
against the existing benchmark references and exact on the app JSONL sample:

| Gate | Baseline | Direct epilogue |
| --- | ---: | ---: |
| FATTN `N=4096` max abs | `0.0051307` | `0.0051307` |
| FATTN `N=4096` mean time | `0.717208 ms` | `0.716163 ms` |
| FATTN `N=196,batch=25` max abs | `0.0320835` | `0.0320835` |
| FATTN `N=196,batch=25` mean time | `0.217935 ms` | `0.219418 ms` |
| App 10-frame JSONL vs previous default | `diff_rows=0` | `diff_rows=0` |
| App 10-frame track mean | `45.06 ms/frame` previous default | `44.80 ms/frame` |

This is a small parity-safe cleanup, not a completion of the performance goal.
The app still trails saved official PyTorch Base+ bf16 by about `2.13
ms/frame`.

Evidence:

- `outputs/fattn56-direct-epilogue-final-20260517/`
- `outputs/fattn56-direct-epilogue-final-app-20260517/summary.json`

After the direct epilogue, the remaining CUDA profile still points to D56
attention and q4_1 MMQ as the useful surfaces. FATTN56 drop-first total is now
`56.02 ms` over 232 effective rows; the largest buckets remain
`Q[56,196,8,25]` (`17.67 ms`) and `Q[56,4096,8,1]` (`17.65 ms`). MMQ
drop-first total over the 3-frame profile is `13.75 ms`, led by q4_1
`[1792,448] x [1792,4096]`, q4_1 `[448,112] x [448,65536]`, and q4_1
`[896,224] x [896,16384]`.

The existing q4_1 4096-column MMQ X cap was re-swept after the direct epilogue.
The Blackwell default remains the best or tied within noise:

| Mode | Track ms/frame values | Mean | Median |
| --- | --- | ---: | ---: |
| default | `44.9, 44.6, 45.2` | `44.90` | `44.9` |
| `GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX=96` | `44.8, 44.8, 45.3` | `44.97` | `44.8` |
| `GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX=112` | `44.6, 45.7, 44.6` | `44.97` | `44.6` |
| `GGML_CUDA_ENABLE_MMQ_Q4_1_4096_X_MAX=128` | `45.2, 44.4, 45.5` | `45.03` | `45.2` |

The q8_1 warp-column quantizer setting was also swept because the post-direct
profile shows large activation-quantize buckets. `GGML_CUDA_MMQ_Q8_1_WARP_COLS=16`
preserved 10-frame JSONL parity, but its measured app gain was within noise:

| Mode | Track ms/frame values | Mean | Median |
| --- | --- | ---: | ---: |
| default | `45.3, 44.3, 44.9, 44.9, 44.8, 45.7` | `44.98` | `44.9` |
| `GGML_CUDA_MMQ_Q8_1_WARP_COLS=16` | `44.9, 44.7, 45.0, 46.1, 44.7, 44.1` | `44.92` | `44.8` |

Do not change the default based on this run.

D56 `ncols=64` `nbatch_fa` was also tried at `128` after the direct epilogue.
The FATTN microbench preserved the same max-abs values, but app timing regressed
relative to the direct-epilogue baseline:

| Variant | App JSONL | Track ms/frame values | Mean |
| --- | --- | --- | ---: |
| direct epilogue, `nbatch_fa=64` | `diff_rows=0` | `44.2, 45.3, 44.5, 45.0, 45.0` | `44.80` |
| direct epilogue, `nbatch_fa=128` | `diff_rows=0` | `44.6, 45.7, 44.7, 44.8, 44.9` | `44.94` |

The `nbatch_fa=128` experiment was reverted.

A smaller direct-epilogue cleanup that hoisted the full-query-tile boundary
check out of the per-fragment branch also preserved parity, but did not improve
the FATTN microbench (`N=4096` stayed about `0.716 ms`, while the dense
`N=196,batch=25` row moved to about `0.221 ms`). It was reverted.

Evidence:

- `outputs/post-direct-epilogue-hotspots-20260517/fattn56-summary-drop-first.json`
- `outputs/post-direct-epilogue-hotspots-20260517/mmq-summary-drop-first.json`
- `outputs/mmq-xmax-after-direct-epilogue-20260517/summary.json`
- `outputs/mmq-q8-warp-cols-c16-verify-20260517/summary.json`
- `outputs/fattn56-nbatch128-probe-20260517/`
- `outputs/fattn56-nbatch128-app-20260517/summary.json`
- `outputs/fattn56-direct-fulltile-probe-20260517/`

### 2026-05-17: Rejected q8-only shared-writer fusion variant

The q4_1 q8-only producer/consumer fusion remains the only fusion route with a
clear microbench speed signal, but its direct q8 writer is still not app-parity
safe. A stricter variant was tested that materialized the producer's f32 tile
in shared memory and then emitted the q8_1 DS4 block with the generic tile
writer. The goal was to move the writer closer to the standalone
`quantize_mmq_q8_1_cuda` layout without rereading global f32 output.

This variant was rejected. With the normal hot `mmq_x` setting the extra shared
memory exceeded CUDA's dynamic shared-memory opt-in limit. After capping the
shared-writer path to `mmq_x <= 64`, it ran but lost the useful fusion speed:

| Mode | q4_1 two-stage `cols=4096` mean |
| --- | ---: |
| default path | `0.298143 ms` |
| q8-only direct writer | `0.253806 ms` |
| q8-only shared writer, capped to `mmq_x <= 64` | `0.311773 ms` |

The small `cols=128` correctness smoke still passed with `bad=0/57344`, so the
problem is not a gross layout break; it is that the stricter writer consumes too
much shared memory and reduces tile width enough to erase the speed gain. The
code was removed. The next fusion attempt should not add a second f32 tile in
shared memory. It needs either a byte/parity-safe direct DS4 writer with the
current tile width, or a consumer MMQ body that can read a cheaper producer-side
layout directly.

Evidence:

- `outputs/q8-only-shared-writer-probe-20260517/`

### 2026-05-17: Rejected q8-only max-column gating

The q8-only producer/consumer fusion was also narrowed with
`GGML_CUDA_MMQ_Q8_ONLY_MAX_COLS` to see whether a subset of producer rows could
be defaulted safely. The result separates parity from useful speed:

| q8-only max cols | 10-frame JSONL parity vs default | Single-run Track/fr |
| ---: | --- | ---: |
| `128` | exact, `10 / 10` mask hashes | `44.4 ms` |
| `512` | exact, `10 / 10` mask hashes | `45.2 ms` |
| `640` | exact, `10 / 10` mask hashes | `45.9 ms` |
| `768` | exact, `10 / 10` mask hashes | `44.6 ms` |
| `896` | exact, `10 / 10` mask hashes | `44.8 ms` |
| `1024` | drift, `9 / 10` mask hashes | `44.3 ms` |
| `2048` | drift, `9 / 10` mask hashes | `45.2 ms` |
| `4096` | drift, `9 / 10` mask hashes | `43.4 ms` |

The `<=128` subset was then run as a 7-pair exact-parity check. It stayed
bit-identical on all rows, but had no speed win:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| default | `45.6, 44.9, 44.5, 44.4, 44.4, 45.0, 44.3` | `44.73` | `44.5` |
| q8-only, `max_cols=128` | `44.4, 44.3, 44.7, 45.2, 45.3, 44.5, 45.6` | `44.86` | `44.7` |

The largest exact single-run subset was `<=896`, but a 7-pair check also
regressed rather than improved:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| default | `44.9, 44.6, 44.5, 44.4, 44.3, 45.0, 44.9` | `44.66` | `44.6` |
| q8-only, `max_cols=896` | `45.2, 44.7, 46.4, 45.1, 44.9, 44.7, 44.6` | `45.09` | `44.9` |

The `<=4096` subset has a real speed signal and keeps bbox coordinates
unchanged, but still changes frame-0 mask hash and scores by up to about
`0.0111`. It remains a diagnostic/upper-bound path, not an accepted
optimization. The fusion route still needs a writer that is both faster and
app-level parity safe; max-column gating alone is not enough.

Evidence:

- `outputs/q8-only-maxcols-sweep-20260517/`
- `outputs/q8-only-max128-paired-20260517/`
- `outputs/q8-only-max896-paired-20260517/`

The best shallow upper-bound combination checked in this round was q8-only
limited to `max_cols=4096` plus `GGML_CUDA_DISABLE_FATTN_STREAM_K=1`. It still
did not beat the saved official PyTorch Base+ bf16 row and was not parity-safe:

| Mode | Track/fr values | Mean | PyTorch bf16 Base+ | Gap |
| --- | --- | ---: | ---: | ---: |
| q8-only `max_cols=4096` + FATTN no stream-k | `45.4, 43.9, 44.8, 44.2, 44.1` | `44.48` | `42.666688` | `+1.81` |

Each run matched bbox coordinates but only `9 / 10` mask hashes, with max score
delta about `0.0111`. This confirms that the remaining gap needs a real
FATTN/MMQ body improvement or a parity-safe producer/consumer layout change;
combining the known diagnostic toggles is not enough.

Evidence:

- `outputs/q8only4096-fattn-nostream-upperbound-20260517/`

### 2026-05-17: D56 Q register residency probe

After checking upstream CUDA history, the current upstream-only FA changes were
not directly useful for Hiera head56 on Blackwell: the new commits were RDNA
tuning and MiMo head `DKQ=192,DV=128` support. A small local D56 body-config
probe was tested instead.

Changing all D56 padded-MMA configs to `Q_in_reg=false` preserved parity and
improved the dense window microbench, but was too broad:

| D56 config | `N=4096` mean | `N=196,batch=25` mean |
| --- | ---: | ---: |
| direct-epilogue baseline | `0.716163 ms` | `0.219418 ms` |
| all D56 `Q_in_reg=false` | `0.717628 ms` | `0.214451 ms` |

Narrowing the change to only the hot `ncols=64` D56 config kept the global row
neutral and improved the window row:

| D56 config | `N=4096` mean | `N=196,batch=25` mean |
| --- | ---: | ---: |
| only `ncols=64`, `Q_in_reg=false` | `0.715360 ms` | `0.213458 ms` |

The app-level 7-run smoke was exact against the saved default JSONL rows and
measured within the previous app noise band:

| Mode | Track/fr values | Mean | Median | JSONL |
| --- | --- | ---: | ---: | --- |
| only `ncols=64`, `Q_in_reg=false` | `45.2, 44.2, 44.3, 44.6, 44.8, 45.1, 45.2` | `44.77` | `44.8` | exact |

An additional `occupancy=1` probe for the same `ncols=64` row regressed the
microbench (`N=196,batch=25` mean `0.220597 ms`) and was reverted. Keep
`occupancy=2`. This D56 config change is a small parity-safe micro improvement,
not a PyTorch-gap closer by itself.

Evidence:

- `outputs/fattn56-qinreg-false-probe-20260517/`
- `outputs/fattn56-qinreg64-false-probe-20260517/`
- `outputs/fattn56-qinreg64-false-app-20260517/`
- `outputs/fattn56-qinreg64-occ1-probe-20260517/`

### 2026-05-18: Rejected q4_1 partial-row zero-fill loader

The q4_1 tile loader was probed with a partial-row policy that zero-filled rows
outside the valid tile instead of clamping those lanes to `i_max`. The intent was
to avoid redundant global reads on the hot partial-row shape:
`q4_1[448,112] x f32[448,65536] -> f32[112,65536]`.

The standalone MMQ result regressed the hot row versus the current documented
baseline:

| Shape | Current baseline | Zero-fill probe |
| --- | ---: | ---: |
| `q4_1[448,112] x f32[448,65536]` | `~0.4516 ms` | `0.467030 ms` |
| `q4_1[448,1792] x f32[448,4096]` | not the target | `0.148832 ms` |

The avoided invalid-row loads are cheaper than the added branch/zero-fill
pressure in this loader. The probe was reverted. Future work on this surface
should avoid per-lane control flow in the hot loader and instead target a real
q4_1 body/fusion layout change.

Evidence:

- `outputs/mmq-q41-partial-row-zero-probe-20260518/`

### 2026-05-18: q8-only fusion column isolation gate

The q8-only producer/consumer fusion was narrowed further after the previous
`max_cols` sweep showed exact parity up to `896` columns and drift at `1024+`.
An additional diagnostic gate, `GGML_CUDA_MMQ_Q8_ONLY_MIN_COLS`, was added so a
single column bucket can be isolated without enabling every smaller bucket.
This gate is diagnostic only; q8-only producer fusion remains opt-in.

The isolation runs show that both the `1024` and `4096` column buckets can cause
the same app-level drift pattern:

| q8-only column gate | Track/fr smoke | JSONL parity vs default |
| --- | ---: | --- |
| `max_cols=1024` | `44.2 ms` vs default `44.4 ms` | `9 / 10` mask hashes, frame 0 differs |
| `min_cols=4096,max_cols=4096` | `44.0 ms` vs default `44.9 ms` | `9 / 10` mask hashes, frame 0 differs |

The isolated synthetic two-stage MMQ checks still pass, which is why this path
continues to look attractive in microbenchmarks:

| Synthetic two-stage shape | Default | q8-only gated | Check |
| --- | ---: | ---: | --- |
| `q4_1 k=448, rows=3584, fc2_rows=896, cols=1024` | `0.186070 ms` | `0.183434 ms` | `bad=0/917504` |
| `q4_1 k=448, rows=1792, fc2_rows=448, cols=4096` | `0.288506 ms` | `0.277929 ms` | `bad=0/1835008` |

Bbox coordinates stayed exact in both runs, but scores moved by up to
`0.011109` for the `1024` bucket and `0.008671` for the isolated `4096` bucket.
That is not just a selector-width issue; it means the producer-side q8_1
side-output is not byte-equivalent enough to the normal f32 writeback followed
by `quantize_mmq_q8_1_cuda` on these real Hiera rows. The fusion path should
therefore remain a diagnostic upper bound until it can prove byte/layout
equivalence for the real producer rows, not only the synthetic two-stage MLP.
The next useful implementation step is a real-graph diagnostic that materializes
both the q8-only side-output and the normal f32->q8 quantizer output for selected
Hiera producers and reports byte-level `qs`/`ds4` mismatches before the consumer
MMQ runs.

Evidence:

- `outputs/q8-only-max1024-isolation-20260518/`
- `outputs/q8-only-4096-only-20260518/`
- `outputs/q8-only-two-stage-buckets-20260518/`

### 2026-05-18: q8-only real-graph byte comparison

A debug-only byte comparator was added behind
`GGML_CUDA_DEBUG_MMQ_Q8_ONLY_COMPARE=1`. When q8-only producer fusion is enabled,
the diagnostic reruns the producer through the normal f32 writeback path with
the same non-stream-k schedule, quantizes that f32 output with
`quantize_mmq_q8_1_cuda`, and compares it against the q8-only side-output before
the consumer MMQ uses the cache. This is not a benchmark mode; it deliberately
adds work to isolate layout/byte differences.

The `1024` column bucket is almost byte-identical except for a few `ds4` bytes:

| Producer shape | q8 payload bytes | mismatch | `ds4.d` mismatch | `ds4.sum` mismatch | `qs` mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3584 x 1024`, node 1070 | `4128768` | `8..9` | `0` | `8..9` | `0` |
| `3584 x 1024`, node 1118 | `4128768` | `10..11` | `0` | `10..11` | `0` |
| `3584 x 1024`, node 1166 | `4128768` | `12` | `0` | `12` | `0` |

For the `4096` column bucket, the logical producer values are also dominated by
`ds4` differences, while the large `qs` mismatch begins at byte `8257552`, which
is exactly the first padded qblock for `dst_ne0=1792` padded to `2048`
(`14 * 4096 * sizeof(block_q8_1_mmq)`). The q8-only writer currently does not
materialize those padded qblocks, while the normal quantizer covers the padded
range:

| Producer shape | q8 payload bytes | `ds4.d` mismatch | `ds4.sum` mismatch | `qs` mismatch |
| --- | ---: | ---: | ---: | ---: |
| `1792 x 4096`, first node 320 | `9437184` | `28599..28604` | `28621..28622` | padded range starts at byte `8257552` |
| `1792 x 4096`, typical Hiera MLP nodes | `9437184` | about `65417..65473` | about `65405..65451` | padded range starts at byte `8257552` |
| `1024 x 4096`, memory fuser nodes | `4718592` | `0` | `34..53` | `0` |

This narrows the q8-only parity problem. The valid `qs` payload is not the main
issue; the remaining real-graph mismatch is the `ds4` scale/sum half2 payload,
plus unmaterialized padded qblocks for shapes whose logical row count is smaller
than the padded quantization extent. Since stream-k disable was already shown to
affect scores, a default-on fusion needs either a stream-k-compatible side-output
or an explicitly documented quality gate that tolerates the non-stream-k and
`ds4` differences. The current exact-parity gate is still not met.

Rechecking the `1024` column bucket with `GGML_CUDA_DISABLE_MMQ_STREAM_K=1` on
both sides still produced `9 / 10` mask hashes, max score delta `0.008597`, and
frame-0 mask area delta `-1860`. It was also unusably slow (`~2048..2063
ms/frame`) because global MMQ stream-k disable is a diagnostic-only mode. This
confirms that even the small `ds4.sum` mismatch can be visible at the selected
mask level; stream-k order is not the only blocker.

A first attempt to force the q8-only writer's reduction into the standalone
quantizer's apparent row-group order was rejected. It made the byte comparison
much worse (`~2.7M` mismatched bytes on the `3584 x 1024` bucket), changed bbox
coordinates by up to `124.453 px`, and slowed the q8-only app smoke to
`44.7 ms/frame`. The probe was reverted. The row/lane mapping in the MMA
writeback is not equivalent to a simple warp-column quantizer remap; any exact
writer needs a more careful derivation from the `tile_C` layout.

A narrower follow-up kept the existing `amax`, scale, and `qs` path but tried to
change only `ds4.sum` to the apparent quantizer order. It also failed: the
`3584 x 1024` bucket regressed from `8..12` mismatched `ds4.sum` bytes to about
`223599..224011` mismatched bytes, while `qs` stayed exact. This confirms that
the naive gathered row order is still not the normal quantizer's byte order for
the MMA producer output. The probe was reverted.

Evidence:

- `outputs/q8-only-byte-compare-nostream-ref-20260518/`
- `outputs/q8-only-byte-compare-1024-20260518/`
- `outputs/q8-only-byte-compare-ds4-split-20260518/`
- `outputs/q8-only-nostream-app-parity-20260518/`
- `outputs/q8-only-exact-reduce-probe-20260518/`
- `outputs/q8-only-sum-order-probe-20260518/`

### 2026-05-18: Rejected all-D56 Q register residency refresh

The D56 `Q_in_reg=false` probe was refreshed on the current branch by applying
it to all D56 padded-MMA configs (`ncols=8/16/32/64`) instead of only the
accepted `ncols=64` row. It still preserved standalone parity, but it regressed
both representative D56 timings:

| D56 all `Q_in_reg=false` refresh | Result |
| --- | ---: |
| `N=4096, heads=8, batch=1` | `0.719707 ms`, `bad=0/4032` |
| `N=196, heads=8, batch=25` | `0.220277 ms`, `bad=0/2195200` |

This is worse than the accepted narrow `ncols=64` change, so the broader probe
was reverted. Keep only `ncols=64` with `Q_in_reg=false`.

Evidence:

- `outputs/fattn56-all-qinreg-false-refresh-20260518/`

### 2026-05-18: Rejected q4_1 MMQ on-demand `dmA` load probe

A q4_1 MMQ body probe removed the Turing/Blackwell MMA path's precomputed
`dmA[ntx][tile_C::ne/2][K]` register array and loaded the q4_1 scale/offset
pair on demand inside the accumulator loop. The intent was to reduce register
pressure in the hot q4_1 body without changing the accumulation formula.

It built and preserved the two-stage check, but it was slower:

| Probe | Result |
| --- | ---: |
| `q4_1 k=448, rows=1792, fc2_rows=448, cols=4096`, two-stage | `0.316913 ms`, `bad=0/1835008` |
| `q4_1 k=448, rows=112, cols=65536`, timing only | `0.459790 ms` |

Both are worse than the current nearby baselines (`~0.288 ms` for the two-stage
row and `~0.4516 ms` for the hot partial row). The compiler/hardware prefer
keeping those scale pairs resident even with the extra registers. The probe was
reverted.

Evidence:

- `outputs/mmq-q41-dma-on-demand-probe-20260518/`

### 2026-05-18: Fusion-kernel continuation

The current-speed refresh was reparsed after confirming the benchmark table
column order. The valid `Track/fr` field is the seventh pipe-delimited field,
not the load-time field:

| Current default run values | Mean | Median | Saved official PyTorch Base+ bf16 | Mean gap |
| --- | ---: | ---: | ---: | ---: |
| `45.6, 44.2, 47.8, 44.2, 44.8, 44.8, 44.1` | `45.07` | `44.8` | `42.666688` | `+2.40 ms/frame` |

This confirms the active goal is still open: the C++ CUDA path has not beaten
the official PyTorch bf16 Base+ row.

The q8-only producer/consumer fusion was probed again by changing only the MMA
writer's `ds4` reduction order. The result did not improve the real-graph byte
comparison: the `3584 x 1024` bucket stayed at a few `ds4.sum` byte mismatches,
and the `1792 x 4096` bucket still had large `ds4` differences plus padded-range
`qs` mismatches. The probe was reverted. The q8-only path still needs a writer
that reconstructs the same 32-value groups as the standalone quantizer from the
MMA `tile_C` layout; a reduction-order tweak is not enough.

The parity-safe `ADD + q8_1 prequant` fusion was then refreshed. Unlike q8-only,
this fusion keeps the normal f32 producer output and only avoids a separate
consumer-side quantization pass for the immediately following q4_1 MMQ. A first
5-pair opt-in run showed exact 10-frame JSONL parity and a small mean speed
signal:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| default before default-on | `44.9, 45.0, 44.7, 44.9, 44.7` | `44.84` | `44.9` |
| `GGML_CUDA_ENABLE_ADD_MMQ_PREQUANT_CACHE=1` | `44.3, 44.3, 44.9, 44.5, 44.8` | `44.56` | `44.5` |

The fusion was changed from opt-in to default-on, with
`GGML_CUDA_DISABLE_ADD_MMQ_PREQUANT_CACHE=1` as the rollback switch. A 7-pair
default-on versus disabled check preserved exact JSONL parity in every run:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| default-on `ADD + prequant` | `45.0, 44.5, 44.6, 44.8, 44.5, 44.4, 44.9` | `44.67` | `44.6` |
| disabled | `45.0, 44.7, 44.5, 44.2, 44.9, 45.0, 44.9` | `44.74` | `44.9` |

This is a small fusion-kernel cleanup, not a completion of the performance
goal. The measured gain is within app-level noise, but it is parity-safe,
default-reversible, and does not regress the paired run.

Evidence:

- `outputs/current-speed-refresh-20260518/summary.json`
- `outputs/q8-only-ds4-reduce-order-probe-20260518/`
- `outputs/add-prequant-refresh-20260518/`
- `outputs/add-prequant-default-on-20260518/summary.json`

### 2026-05-18: Rejected shallow fusion and MMQ scheduling probes

After the `ADD + prequant` default-on change, a refreshed 3-frame profile still
ranked q4_1 MMQ and D56 FATTN as the main surfaces:

| Area | Drop-first profiled total |
| --- | ---: |
| MMQ total | `13.98 ms` |
| FATTN56 total | `15.44 ms` |
| q4_1 `1792 x 448` by `1792 x 4096` two-stage row | `6.99 ms` |
| FATTN56 `Q[56,196,8,25]` | `5.30 ms` |
| FATTN56 `Q[56,4096,8,1]` | `4.83 ms` |

A same-shape fused ADD contiguous-pack shortcut was tested because synchronized
node profiling still showed fused ADD rows. It preserved 10-frame JSONL parity
but regressed normal app timing:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| contiguous-pack shortcut | `45.5, 44.8, 45.1, 45.0, 44.8, 44.8, 44.2` | `44.89` | `44.8` |
| disabled shortcut | `44.2, 44.4, 44.8, 44.9, 44.2, 44.8, 44.8` | `44.59` | `44.8` |

The shortcut was reverted. The generic fused broadcast kernel is not the current
gap closer; forcing a simpler indexing path reduced profiler noise but lost in
normal graph execution.

The q4_1 MMQ Stream-K tile-efficiency cutoff was also made temporarily
controllable through `GGML_CUDA_MMQ_STREAM_K_TILE_EFFICIENCY_MIN` to test whether
the hot 93%-efficient 4096-column MLP rows should use fewer blocks plus fixup
instead of one block per tile. Microbenchmarks showed a small two-stage signal at
`95`, but the app check rejected it:

| Mode | Track/fr values | Mean | JSONL |
| --- | --- | ---: | --- |
| default cutoff `90` | `45.5, 44.8, 44.7, 45.1, 44.8, 45.3, 44.7` | `44.99` | baseline |
| cutoff `95` | `44.9, 45.1, 45.7, 45.8, 45.5, 46.2, 45.7` | `45.56` | `9 / 10` mask hashes, max bbox delta `3.75 px` |

This confirms that changing the Stream-K work partition is not parity-safe for
the Base+ selected-mask contract. The default cutoff remains unchanged.

Evidence:

- `outputs/post-add-prequant-hotspots-20260518/`
- `outputs/bin-contiguous-pack-probe-20260518/summary.json`
- `outputs/mmq-streamk-eff-threshold-probe-20260518/`
- `outputs/mmq-streamk-eff-threshold-app-20260518/summary.json`

### 2026-05-18: Rejected activation-prequant fusion

The next producer/consumer fusion check separated two cases:

1. `q8-only` producer side-output, which removes the intermediate f32 producer
   write and lets the following q4_1 MMQ consume the side-output directly.
2. `activation + prequant`, which keeps the normal f32 producer tensor but
   combines the post-MMQ activation writeback and q8_1 prequant generation for
   the immediate q4_1 MMQ consumer.

The first case is still not acceptable. Limiting q8-only to the largest
`dst->ne[1] == 16384` candidate gave a real speed signal, but changed tracking
output every run:

| Mode | Track/fr values | Mean | JSONL |
| --- | --- | ---: | --- |
| default | `43.8, 42.9, 43.0, 43.0, 43.0` | `43.14` | baseline |
| q8-only `16384` | `43.2, 42.4, 42.7, 42.6, 42.8` | `42.74` | `9 / 10` mask hashes, max bbox delta `120 px` |

The speed is large enough to keep this as a root target, but the current
side-output writer still cannot replace the standalone quantizer. A no-Stream-K
comparison confirmed that the mismatch is not only a Stream-K scheduling issue:
q8-only against the no-Stream-K baseline still produced `9 / 10` mask-hash
matches and a max bbox delta of `120 px`.

The second case was rechecked after fixing the graph-side gate so that the
activation-prequant path actually ran by default. It preserved exact output, but
it was clearly slower:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| activation-prequant truly enabled | `45.0, 45.1, 44.7, 44.1, 45.5, 45.0, 45.0` | `44.91` | `45.0` |
| disabled | `43.2, 43.2, 43.8, 43.4, 43.6, 44.1, 43.4` | `43.53` | `43.4` |

All seven comparisons had `10 / 10` mask-hash matches with zero bbox, score,
and mask-area deltas, so this was not a correctness failure. The separate
activation+quantization kernel adds more cost than it removes from the following
consumer quantization. The graph gate and default-on change were reverted; keep
`GGML_CUDA_ENABLE_MMQ_ACT_PREQUANT_FUSION=1` diagnostic-only.

The remaining root work is still the q8-only side-output layout, or deeper
q4_1 MMQ / D56 FATTN kernel-body changes.

Evidence:

- `outputs/q8-only-shape16384-probe-20260518/summary.json`
- `outputs/q8-only-nostream-cause-20260518/summary.json`
- `outputs/mmq-act-prequant-gatefix-20260518/summary.json`

### 2026-05-18: q8_1 prequant warp-column tuning

The standalone q8_1 DS4 prequant kernel already had a diagnostic
`GGML_CUDA_MMQ_Q8_1_WARP_COLS` switch. The default was `4` columns per block.
Testing `8` and `16` on the Base+ q4_1 app run preserved exact JSONL parity:

| Mode | Track/fr values | Mean | Median | JSONL |
| --- | --- | ---: | ---: | --- |
| default `4` before change | `44.1, 43.1, 44.0, 43.2, 43.9` | `43.66` | `43.9` | baseline |
| `GGML_CUDA_MMQ_Q8_1_WARP_COLS=8` | `43.1, 43.2, 43.0, 43.0, 44.3` | `43.32` | `43.1` | exact |
| `GGML_CUDA_MMQ_Q8_1_WARP_COLS=16` | `43.1, 43.1, 43.8, 43.3, 43.5` | `43.36` | `43.3` | exact |

The default was changed to `8`, while `GGML_CUDA_MMQ_Q8_1_WARP_COLS=4`
remains the rollback. A 7-pair default8 versus forced4 check stayed exact:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| default8 | `43.5, 43.2, 43.5, 43.1, 43.3, 43.3, 43.4` | `43.33` | `43.3` |
| forced4 | `43.1, 43.2, 44.3, 44.1, 43.4, 43.1, 43.2` | `43.49` | `43.2` |

The paired delta is small (`-0.16 ms/frame`) but parity-safe. This is a
quantization-kernel parameter improvement, not a full PyTorch-gap closer.

The same warp-column retune was checked for the `ADD + prequant` fused kernel.
Although `8`/`16` looked slightly faster in a 5-run sweep, the 7-pair default16
versus forced4 run was neutral to slightly slower (`43.41` vs `43.39 ms/frame`),
so the fused ADD-prequant kernel keeps its default `4` columns per block.

Evidence:

- `outputs/q8-quant-warp-cols-app-20260518/summary.json`
- `outputs/q8-quant-warp-cols-default8-20260518/summary.json`
- `outputs/add-prequant-warp-cols-app-20260518/summary.json`
- `outputs/add-prequant-warp-cols-default16-20260518/summary.json`

### 2026-05-18: Rejected narrow FATTN56 nbatch retune

After the q8_1 prequant retune, the profile still showed D56 FlashAttention as
one of the top remaining surfaces. A narrow compile-time config probe changed
only the D56 padded-MMA `ncols=64` row from `nbatch_fa=64` to `128`, leaving the
accepted `Q_in_reg=false` setting untouched.

The standalone parity microbench stayed correct but the timing change was too
small to justify the config change:

| Shape | Current | `nbatch_fa=128` |
| --- | ---: | ---: |
| `D=56, run_D=64, N=4096, heads=8, batch=1` | `0.700448 ms`, `bad=0/28672` | `0.699742 ms`, `bad=0/28672` |
| `D=56, run_D=64, N=196, heads=8, batch=25` | `0.223279 ms`, `bad=0/716800` | `0.222748 ms`, `bad=0/716800` |

This is within microbench noise and does not cover the remaining app-level gap,
so the probe was reverted. D56 FATTN still needs a more structural body change
rather than another one-row config retune.

Evidence:

- `outputs/fattn56-ncols64-nbatch128-probe-20260518/`

### 2026-05-18: Rejected wider q8_1 warp threshold and refreshed MMQ X cap

The q8_1 DS4 warp-column prequant path was also tested with a lower activation
column threshold. Changing the fast-path guard from `ne1 >= 16384` to
`ne1 >= 4096` preserved exact JSONL output, but the 7-pair app delta was only
`-0.03 ms/frame`:

| Mode | Track/fr values | Mean | Median |
| --- | --- | ---: | ---: |
| threshold `4096` | `43.3, 43.3, 43.1, 43.1, 43.2, 43.3, 43.2` | `43.21` | `43.2` |
| warp-column quant disabled | `43.0, 43.0, 43.0, 43.6, 43.7, 43.1, 43.3` | `43.24` | `43.1` |

The current threshold stays at `16384`; the lower threshold did not produce a
stable enough app-level gain.

MMQ X caps were refreshed on the default8 prequant build. `GGML_CUDA_MMQ_X_MAX=96`
looked slightly faster in a 3-run sweep and kept parity, but a 7-pair check
reversed the signal:

| Mode | Track/fr values | Mean | Median | JSONL |
| --- | --- | ---: | ---: | --- |
| default | `43.1, 43.2, 43.4, 43.2, 43.3, 43.2, 43.6` | `43.29` | `43.2` | baseline |
| `GGML_CUDA_MMQ_X_MAX=96` | `43.8, 43.4, 43.4, 43.3, 43.7, 43.3, 43.2` | `43.44` | `43.4` | exact |

`X_MAX=112` and `128` were also rejected in the 3-run sweep because they changed
tracking output (`9 / 10` mask hashes). The default MMQ X cap remains unchanged.

Evidence:

- `outputs/q8-quant-warp-threshold4096-20260518/summary.json`
- `outputs/mmq-xmax-refresh-default8-20260518/summary.json`
- `outputs/mmq-x96-default8-ab7-20260518/summary.json`

### 2026-05-18: Rejected q8_0 D4 warp-column prequant

The same warp-column prequant idea was tested for Base+ `q8_0`, because the
15-run q8_0 mean was close to the official PyTorch bf16 baseline but still not
robustly faster. The q8_0 MMQ input uses the D4 scale layout, so this required a
separate D4 writer instead of reusing the q4_1 DS4 path.

A 5-run `cols=4/8/16` sweep preserved exact JSONL parity for all variants, but
the 7-pair default8 check regressed:

| Mode | Track/fr values | Mean | Median | JSONL |
| --- | --- | ---: | ---: | --- |
| D4 warp-column disabled | `44.5, 43.8, 44.3, 44.1, 44.0, 44.2, 44.2` | `44.16` | `44.2` | baseline |
| D4 warp-column default8 | `44.1, 44.9, 44.4, 44.5, 44.5, 44.0, 44.6` | `44.43` | `44.5` | exact |

The delta was `+0.27 ms/frame`, so the D4 kernel and dispatch were reverted.
This suggests the q8_0 path is not bottlenecked by the generic D4 prequant launch
shape enough to justify a specialized writer. The next q8_0 work should focus on
larger producer-side fusion or the D56 FlashAttention body rather than another
standalone prequant variant.

Evidence:

- `outputs/q8-d4-warp-cols-probe-20260518/summary.json`
- `outputs/q8-d4-warp-cols-default8-ab7-20260518/summary.json`

The D4 quantizer was also tested with a narrower contiguous-2D specialization
that kept the existing block layout but skipped the generic `ids/ne2/ne3`
addressing path for Base+ `q8_0` MMQ rows. The first 7-pair check preserved
exact JSONL parity and looked promising (`44.04` enabled versus `44.61`
disabled), but a 15-run absolute check still failed to beat official PyTorch
(`44.25 ms/frame` versus `42.67 ms/frame`). A larger-shape-only guard
(`ne00 >= 1024`) reduced the risk of small-row regressions but remained
order-sensitive:

| Pair order | Enabled mean | Disabled mean | Delta | JSONL |
| --- | ---: | ---: | ---: | --- |
| disabled then enabled, 15 pairs | `44.41` | `44.65` | `-0.25 ms/frame` | exact |
| enabled then disabled, 7 pairs | `44.50` | `44.26` | `+0.24 ms/frame` | exact |

Because the sign flipped when the pair order changed, this was treated as
measurement/order bias rather than a reliable kernel win. The contiguous-2D D4
specialization was reverted.

Evidence:

- `outputs/q8-d4-contig2d-ab7-20260518/summary.json`
- `outputs/q8_0-d4-contig2d-speed-15run-20260518/summary.json`
- `outputs/q8-d4-contig2d-ab15-20260518/summary.json`
- `outputs/q8-d4-contig2d-ne1024-ab15-20260518/summary.json`
- `outputs/q8-d4-contig2d-ne1024-reverse-ab7-20260518/summary.json`

### 2026-05-18: q8-only producer fusion tail fix, still diagnostic-only

The q8-only producer-side fusion was revisited because it was the only fusion
probe with a large enough early speed signal to matter. Debug comparison against
the normal `quantize_mmq_q8_1_cuda` path showed the main correctness bug: the
fused producer wrote only the real MMQ row tiles and left the extra
`ne0_padded` qblock uninitialized. For the hot Base+ node shape
`dst_ne=896,16384,1,1`, that left the padded 896..1023 rows containing pool
garbage, while the standalone quantizer writes zeroes.

Zeroing only the unwritten padded tail reduced the debug mismatch from millions
of payload bytes to DS4 sum-only half differences:

| Mode | Payload mismatch | DS4 mismatch | QS mismatch | Max QS abs diff |
| --- | ---: | ---: | ---: | ---: |
| before tail zero | `~2.33M` | `~261K` | `~2.07M` | `127` |
| after tail zero | `39..112` | `39..112` | `0` | `0` |

This fixes the obvious layout/padding bug but does not make the fusion
parity-safe yet. The remaining differences are the DS4 sum half values, caused
by the direct MMA writer accumulating the per-32-row sums in a different order
from the standalone quantizer. App-level tracking improved from the earlier
large bbox drift to matching boxes, but strict mask hashes still differ:

| Mode | Track/fr | JSONL |
| --- | ---: | --- |
| default | `44.2` | baseline |
| q8-only tail-zero | `44.1` | `9 / 10` mask hashes, max bbox delta `0 px`, max score delta `0.008591` |

A conservative shared-tile writer, which would write f32 results to shared
memory and then reuse the existing DS4 tile writer, was also tried. It exceeded
the launchable shared-memory configuration for the current Turing/Blackwell MMQ
shape and failed with `CUDA error: invalid argument`, so it was reverted.

The q8-only producer fusion therefore remains opt-in diagnostic-only behind
`GGML_CUDA_ENABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1`. The tail-zero fix is kept
because it is a correctness improvement for that diagnostic path; default
behavior is unchanged.

Evidence:

- `outputs/q8-only-debug-compare-20260518/run.log`
- `outputs/q8-only-debug-compare-tailzero-20260518/run.log`
- `outputs/q8-only-tailzero-app-20260518/summary.json`
- `outputs/q8-only-debug-compare-sharedtile-20260518/run.log`

### 2026-05-18: Accepted q8-only shared-tile producer fusion

The shared-tile q8-only producer fusion was reintroduced with two guardrails:

1. The MMQ tile selector accounts for the extra shared-memory tile and picks a
   smaller `mmq_x` when the full-size tile would exceed the device shared-memory
   limit.
2. The direct MMA-to-q8 writer remains diagnostic-only behind
   `GGML_CUDA_ENABLE_MMQ_Q8_ONLY_DIRECT_WRITER=1`; the default q8-only path
   writes the MMQ result to a shared f32 tile and then uses the existing DS4 tile
   writer, matching the standalone quantizer's sum order.

With `GGML_CUDA_DEBUG_MMQ_Q8_ONLY_COMPARE=1`, every checked Base+ q4_1 q8-only
candidate matched `quantize_mmq_q8_1_cuda` exactly:

| Debug field | Result |
| --- | ---: |
| payload mismatch | `0` |
| DS4 mismatch | `0` |
| DS4 scale mismatch | `0` |
| DS4 sum mismatch | `0` |
| QS mismatch | `0` |

The first 5-pair opt-in check also preserved exact app JSONL output and showed a
usable speed signal:

| Mode | Track/fr mean | Track/fr median | JSONL |
| --- | ---: | ---: | --- |
| normal default | `44.78 ms` | `44.7 ms` | baseline |
| q8-only shared writer | `43.84 ms` | `43.7 ms` | `10 / 10` mask hashes, zero bbox/score delta |

The path was then enabled by default for graph-proven q4_1 producer/consumer
pairs. `GGML_CUDA_DISABLE_MMQ_Q8_ONLY_PRODUCER_FUSION=1` is the rollback. A
7-pair default-on versus rollback check stayed exact and improved the mean by
`0.74 ms/frame`:

| Mode | Track/fr values | Mean | Median | JSONL |
| --- | --- | ---: | ---: | --- |
| default q8-only shared | `44.3, 43.8, 44.7, 43.9, 44.0, 43.9, 43.9` | `44.07` | `43.9` | `10 / 10` every pair |
| rollback disabled | `44.3, 45.2, 44.9, 44.7, 44.3, 45.3, 45.0` | `44.81` | `44.9` | baseline |

This is a real fusion-kernel improvement for the Base+ q4_1 Hiera MLP path. It
does not close the q8_0 versus PyTorch gap by itself, but it removes one
producer-to-consumer quantization boundary without changing tracking output.

Evidence:

- `outputs/q8-only-shared-writer-debug-20260518/run.stderr`
- `outputs/q8-only-shared-writer-ab5-20260518/summary_table.json`
- `outputs/q8-only-shared-default-on-ab7-20260518/summary.json`

### 2026-05-18: Accepted q8_0 q8-only D4 producer fusion

The same producer/consumer fusion was extended from q4_1/DS4 to q8_0/D4. The
graph-side candidate check now accepts direct q4_1 or q8_0 `MUL_MAT` consumers,
and the q8-only producer kernel writes the shared f32 tile through a D4 q8_1
writer when the downstream MMQ weight type is q8_0. The direct MMA writer is not
used for q8_0.

The debug comparison against `quantize_mmq_q8_1_cuda` matched exactly for every
checked Base+ q8_0 candidate:

| Debug field | Result |
| --- | ---: |
| payload mismatch | `0` |
| D4/DS4 metadata mismatch | `0` |
| QS mismatch | `0` |

A 7-pair default-on versus rollback run preserved exact tracking output and
improved mean speed by `0.71 ms/frame`:

| Mode | Track/fr values | Mean | Median | JSONL |
| --- | --- | ---: | ---: | --- |
| default q8-only D4 | `44.2, 43.5, 45.2, 44.6, 43.3, 43.1, 43.2` | `43.87` | `43.5` | `10 / 10` every pair |
| rollback disabled | `45.5, 43.6, 44.3, 44.4, 44.2, 44.0, 46.1` | `44.59` | `44.3` | baseline |

The q8_0 MMQ/FATTN profile after this change shows that the producer/consumer
quantization boundary is no longer the main gap:

| Area | Drop-first total |
| --- | ---: |
| MMQ total | `4.36 ms` |
| MMQ quant | `2.18 ms` |
| MMQ body | `2.19 ms` |
| FATTN56 total | `15.67 ms` |

An absolute 15-run q8_0 check is still slower than the saved official PyTorch
bf16 baseline:

| Runtime | Mean | Median | Delta vs PyTorch |
| --- | ---: | ---: | ---: |
| C++ q8_0 after q8-only D4 | `43.76 ms/frame` | `43.6 ms/frame` | `+1.09 ms/frame` |
| official PyTorch bf16 saved baseline | `42.67 ms/frame` | n/a | baseline |

The next PyTorch-gap target is therefore D56 FlashAttention body time, not MMQ
prequantization. The current top FATTN56 rows remain `Q[56,196,8,25]` and
`Q[56,4096,8,1]`.

Evidence:

- `outputs/q8_0-q8-only-shared-debug-20260518/run.stderr`
- `outputs/q8_0-q8-only-shared-default-on-ab7-20260518/summary.json`
- `outputs/q8_0-post-q8-only-hotspots-20260518/mmq-summary.json`
- `outputs/q8_0-post-q8-only-hotspots-20260518/fattn56-summary.json`
- `outputs/q8_0-q8-only-speed-15run-20260518/summary.json`

Two immediate follow-up probes were rejected:

| Probe | Mean | Delta | JSONL | Decision |
| --- | ---: | ---: | --- | --- |
| `GGML_CUDA_ENABLE_FATTN56_INLINE_PACK_LARGE=1` | `45.33 ms/frame` vs default `43.99` | `+1.34 ms/frame` | exact | reject |
| narrow q8_0 D4 warp-column quantizer for `ne00==448 && ne1>=32768` | `43.79 ms/frame` vs disabled `43.59` | `+0.20 ms/frame` | exact | reject and remove |
| FATTN56 `Q->ne[1]==196` ncols32 selector | FATTN total `15.65 ms` vs default `15.81`, but target `Q[56,196,8,25]` worsened `5.88 ms` vs `5.40` | n/a | `2 / 3` mask hashes | reject and remove |

Evidence:

- `outputs/q8_0-fattn56-inline-large-retest-ab7-20260518/summary.json`
- `outputs/q8_0-d4-warp448-ab7-20260518/summary.json`
- `outputs/q8_0-fattn56-n196-ncols32-profile-20260518/`

### 2026-05-18: Accepted q8_0 FATTN config and ADD-prequant fusion refresh

The D56 FATTN MMA config was refreshed by changing the Ampere
`DKQ=64,DV=64,ncols=64` case to keep `Q` in shared memory instead of registers.
This preserves exact tracking output and improves the profiled D56 shapes:

| Shape | Previous mean | New mean |
| --- | ---: | ---: |
| `Q[56,196,8,25]` | `0.153257 ms` | `0.145827 ms` |
| `Q[56,4096,8,1]` | `0.617120 ms` | `0.596314 ms` |
| `Q[56,64,2,1024]` | `0.368762 ms` | `0.356456 ms` |
| `Q[56,16,4,1024]` | `0.281338 ms` | `0.268771 ms` |

The q8_0 `ADD -> MMQ` prequant cache path was also extended to write the D4
q8_1 MMQ layout, instead of only supporting the q4_1 DS4 layout. This is a
small fusion-kernel improvement for ADD outputs that immediately feed a q8_0
MMQ consumer. The same-build rollback is
`GGML_CUDA_DISABLE_ADD_MMQ_PREQUANT_CACHE=1`.

| Mode | Track/fr values | Mean | Median | JSONL |
| --- | --- | ---: | ---: | --- |
| default D4 ADD-prequant | `43.7, 43.4, 42.9, 43.2, 43.3` | `43.30` | `43.3` | `10 / 10` every pair |
| rollback disabled | `43.4, 43.7, 43.5, 43.2, 43.7` | `43.50` | `43.5` | baseline |

With all current fusion changes enabled, the 15-run Base+ q8_0 absolute speed is
still slower than the saved official PyTorch bf16 baseline:

| Runtime | Mean | Median | Delta vs PyTorch |
| --- | ---: | ---: | ---: |
| C++ q8_0 current fusion set | `43.54 ms/frame` | `43.5 ms/frame` | `+0.87 ms/frame` |
| official PyTorch bf16 saved baseline | `42.67 ms/frame` | n/a | baseline |

The current q8_0 path is therefore exact on the checked clip but not yet faster
than PyTorch. Remaining work should keep targeting FATTN56 body time and larger
producer/consumer scheduling changes rather than isolated q8_1 prequant cache
plumbing.

Evidence:

- `outputs/fattn-qinreg-false-q8_0-20260518/fattn56-summary.json`
- `outputs/fattn-qinreg-false-speed7-q8_0-20260518/`
- `outputs/add-prequant-q8_0-profile-20260518/mmq-summary.json`
- `outputs/add-prequant-q8_0-ab5-20260518/`
- `outputs/current-fusion-speed15-q8_0-20260518/summary.json`

The adjacent `ncols=8` `Q_in_reg=false` probe was rejected. It stayed exact on
the 10-frame JSONL check, but regressed FATTN56 profile total from `55.24 ms` to
`55.64 ms`; the largest shape, `Q[56,4096,8,1]`, worsened from
`0.5991 ms` to `0.6098 ms` mean. Evidence:
`outputs/fattn-ncols8-qinreg-false-q8_0-20260518/`.

The adjacent `ncols=8` `nstages_target=1` probe was also rejected. It stayed
exact on the 10-frame JSONL check, but regressed FATTN56 profile total from
`55.24 ms` to `55.92 ms`; `Q[56,4096,8,1]` worsened from `0.5991 ms` to
`0.6087 ms` mean. Evidence:
`outputs/fattn-ncols8-nstages1-q8_0-20260518/`.

Two additional fusion-kernel directions were checked after the accepted
q8-only and ADD-prequant work:

1. General MMQ `src1` prequant reuse inside a graph. The naive version cached
   every MMQ input and aborted later in a cuBLASLt path, consistent with pool
   pressure from holding too many q8_1 buffers. A narrowed graph-identity-only
   version preserved exact JSONL parity but had `src1_q8_cache.hits=0`, because
   the apparent reuse in the profile is mostly repeated frame executions or
   allocator pointer reuse, not the same live tensor being consumed twice in one
   graph.
2. `ADD + GELU -> MMQ` prequant fusion. A fused `ADD+GELU+q8_1` writer was
   built, but the Base+ q8_0 profile also reported `src1_q8_cache.hits=0`; this
   graph does not expose that producer-consumer pattern in the hot path.

Both probes were removed. They are useful negative evidence: more cache plumbing
around isolated q8_1 inputs is unlikely to close the remaining gap. The next
fusion work should either fuse a currently hot producer/consumer pair that is
already visible in the profile, or change the D56 FlashAttention body/schedule.

Evidence:

- `outputs/mmq-src1-prequant-cache-profile-q8_0-20260518/mmq-summary.json`
- `outputs/add-unary-prequant-q8_0-profile-20260518/mmq-summary.json`

The D56 FlashAttention launch schedule was then instrumented through
`GGML_CUDA_PROFILE_FATTN=1`. The current hot D56 shapes all use
`stream_k_general_fixup`; this confirms that the remaining FATTN body time is
not a no-fixup direct epilogue case:

| Shape | Schedule | Profiled total |
| --- | --- | ---: |
| `Q[56,196,8,25] K[56,196,8,25]` | `stream_k_general_fixup` | `18.13 ms` |
| `Q[56,4096,8,1] K[56,4096,8,1]` via padded `Q[64,...]` | `stream_k_general_fixup` | `17.28 ms` |
| `Q[56,64,2,1024] K[56,64,2,1024]` | `stream_k_general_fixup` | `7.24 ms` |
| `Q[56,16,4,1024] K[56,16,4,1024]` | `stream_k_general_fixup` | `5.50 ms` |

A D56-only stream-k disable probe was rejected. It changed several D56 shapes to
`standard` or `parallel_k_combine`, but the small-Q/high-batch cases became
orders of magnitude slower and strict parity failed (`0 / 10` mask hashes). The
main regression examples were:

| Shape | Disabled-stream-k mean | Current mean |
| --- | ---: | ---: |
| `Q[56,196,8,25]` | `42.60 ms` | about `0.16 ms` |
| `Q[56,16,4,1024]` | `972.84 ms` | about `0.29 ms` |
| `Q[56,4,8,1024] K[56,16,8,1024]` | `2815.87 ms` | about `0.39 ms` |

So stream-k itself is required for these shapes. The next viable optimization is
not disabling stream-k, but reducing the cost of the D56 stream-k body while
preserving the same work partitioning semantics.

The profile was then split into main kernel time and fixup/combine time. This
showed that fixup is not the dominant cost; the D56 body kernel is:

| Shape | Kernel sum | Fixup sum | Compute sum |
| --- | ---: | ---: | ---: |
| `Q[56,196,8,25] K[56,196,8,25]` | `16.69 ms` | `1.19 ms` | `17.88 ms` |
| `Q[56,4096,8,1] K[56,4096,8,1]` via padded `Q[64,...]` | `16.93 ms` | `0.35 ms` | `17.28 ms` |
| `Q[56,64,2,1024] K[56,64,2,1024]` | `7.06 ms` | `0.17 ms` | `7.23 ms` |
| `Q[56,16,4,1024] K[56,16,4,1024]` | `5.35 ms` | `0.12 ms` | `5.47 ms` |

This changes the next-priority target: optimize the D56 MMA body/config and
memory movement inside the stream-k body, not the standalone fixup kernel.

Evidence:

- `outputs/fattn-launch-schedule-q8_0-20260518/`
- `outputs/fattn56-disable-streamk-q8_0-20260518/`
- `outputs/fattn-kernel-fixup-split-q8_0-20260518/fattn-split-summary.json`

Two follow-up fusion/schedule probes were rejected on 2026-05-18:

- Skipping the general stream-k fixup when `ntiles_KV == 1` preserved exact
  JSONL parity (`10 / 10` mask hashes), and the synchronized FATTN profile
  showed only a tiny D56 attribution improvement (`61.16 ms` to `60.82 ms` in
  the same split-profile setup). Normal paired speed regressed instead:
  default-on was `43.56 ms/frame` mean and rollback was `43.37 ms/frame` mean
  over seven 10-frame runs. This is not worth keeping; the fixup cost is too
  small and the launch/body behavior does not translate to end-to-end speed.
- Extending the fused ADD/MUL path to use the axis-broadcast fast kernel also
  preserved exact JSONL parity (`10 / 10` mask hashes), but regressed paired
  q8_0 speed. Default-on was `43.61 ms/frame` mean and the disabled path was
  `43.24 ms/frame` mean over seven 10-frame runs. The generic fused broadcast
  path remains the better measured choice for the current graph.
- A refreshed q8_0 D4 warp-column MMQ input quantizer was profiled because it
  reduced synchronized MMQ attribution (`13.87 ms` disabled to `13.45 ms`
  default) and preserved exact JSONL parity. Normal end-to-end speed did not
  confirm the win: the seven-pair check was effectively noise (`43.50 ms/frame`
  default vs `43.56 ms/frame` disabled), and a 4/8/16-column sweep regressed
  against the disabled path (`43.76`, `43.48`, `43.66` vs `43.40 ms/frame`).
  The specialized D4 writer was removed again. Fusion kernels should continue
  to be admitted only when the normal paired benchmark, not just a synchronized
  kernel profile, shows a clear improvement.
- The existing non-contiguous `src1` copy-to-quantize fusion was also checked
  with the same rollback rule (`GGML_CUDA_DISABLE_MMQ_STRIDED_SRC1_QUANT=1`).
  It preserved exact JSONL parity (`10 / 10` mask hashes) and reduced
  synchronized MMQ quant attribution (`7.13 ms` disabled to `6.87 ms` default),
  but the seven-pair normal q8_0 run was still noise-level (`43.53 ms/frame`
  default vs `43.60 ms/frame` disabled). Keep it as a bounded copy-elision
  improvement, but do not count it as closing the q8_0 vs PyTorch gap.
- The q8_0 D4 warp-column quantizer was also rechecked with a 512-only gate
  (`ne1 == 16384`) because the 512 profile has a different dominant MMQ shape
  than 1024. It again preserved exact JSONL parity, but normal 512 paired speed
  regressed (`12.70 ms/frame` default vs `12.43 ms/frame` disabled over seven
  runs). The experiment was removed. The 512 profile now points at quantized
  MLP body work and D56 window attention, not this activation quantizer writer.
- q8_0 MMQ tile-width tuning was rechecked after the 512 profile. A microbench
  showed the `2048x4096` q8_0 shape improving at `mmq_x=104`, but the
  `448x65536` shape still preferred `mmq_x=128`. A global cap104 therefore
  regressed the application (`43.77 ms/frame` vs `43.50 ms/frame`), and a
  shape-gated `ncols_max == 4096` cap104 also regressed (`43.71 ms/frame` vs
  `43.41 ms/frame`) despite exact JSONL parity. This was removed. MMQ selector
  retuning should not be promoted from microbench-only wins without a full graph
  A/B.
- A SAM2 prompt-coordinate parity probe removed the C++ `+0.5` pixel-center
  shift for SAM2 clicks to match the official video predictor's direct
  `point / original_size * image_size` scaling. It did not close the 512 quality
  gap: with candidate 0 forced, min mask IoU moved only from about `0.843` to
  `0.846`, while mean mask IoU dropped from about `0.932` to `0.929`. The change
  was reverted. The 512 gap is therefore not explained by prompt coordinate
  centering alone.
- The 512 Base+ fallback quality gap was then checked with official PyTorch
  logits saved. For `q8_0`, candidate selection matched on all propagation
  frames, but the selected logits diverged in the mask body: min mask IoU was
  `0.8429`, mean mask IoU was `0.9316`, and the worst propagation frames had
  C++ logits strongly lower than official PyTorch in Python-only regions
  (for example frame 7 Python-only mean delta about `-1.82`). Repeating the
  same check with C++ `f32` did not fix the issue: min/mean mask IoU were
  `0.8649`/`0.9348` against official bf16, and `0.8888`/`0.9468` against
  official fp32. The issue is therefore not primarily q8_0 quantization or
  Python bf16 autocast. It is a remaining C++/official propagation or mask
  post-processing mismatch at 512. Forcing official
  `binarize_mask_from_pts_for_mem_enc=true` matches the C++ condition-memory
  contract and is better than `false` (`0.9468` vs `0.8545` mean mask IoU);
  changing official `fill_hole_area` between `0` and `8` did not affect this
  clip. The remaining gap is therefore downstream of condition-memory
  binarization and not explained by hole filling. Dumping the memory-encoder
  input for the initial condition frame confirmed that both sides feed `-10/10`
  binary masks with the same shape, but the mask is already different
  (`63353` positive pixels in C++ vs `60036` in official Python at 512). A
  two-frame initial-candidate check showed the root of that difference starts in
  the initial mask decoder: candidate count and selected candidate index match,
  but candidate mask areas and IoU scores diverge at 512 even for C++ `f32`.
  Re-running both implementations from the same extracted JPEG frame directory
  removed most of the apparent 512 quality gap: C++ `f32` vs official fp32
  improved to min/mean mask IoU `0.9686`/`0.9731`, and C++ `q8_0` vs official
  fp32 improved to `0.9662`/`0.9720`. Against official bf16 on the same shared
  frames, C++ `q8_0` was `0.9577`/`0.9656`. The practical 512 fallback quality
  blocker is therefore mostly the comparison contract: video-decoder differences
  must be removed before judging model parity. With shared frames, quality is
  acceptable for fallback diagnostics, but C++ `q8_0` is still slower than
  official PyTorch bf16 in the single cold-ish quality run (`16.9 ms/frame` vs
  `15.07 ms/frame`). A dedicated 15-run speed pass on the same shared-frame
  input shows the steady-state result is faster: C++ `q8_0` mean/median
  `13.17`/`13.0 ms/frame` versus official PyTorch bf16 `15.07 ms/frame`.
  Therefore the 512 fallback path meets the current speed target when measured
  with a shared input contract and repeated runs.
  The refreshed 512 shared-frame profile keeps the same optimization priority:
  Hiera encode dominates, with stage-2 quantized MLP expansion/projection and
  D56 window attention as the largest drop-max synchronized node groups. MMQ
  activation quantization is no longer a large enough standalone target at 512
  (`0.57 ms` total synchronized quant attribution over the profiled run), so the
  remaining speed work should target the MLP body kernels and D56 attention body
  rather than more prequant cache plumbing.
- The existing `GGML_CUDA_ENABLE_MMQ_Q8_0_196_X64=1` selector was tested because
  the 512 profile has hot `ncols_max == 196` q8_0 window projections. It is not
  strict-parity safe: all nine paired runs had `0 / 10` exact mask hashes.
  The numeric drift is small at the bbox level (example min bbox IoU about
  `0.9967`, max bbox delta `2 px`, max score delta about `0.0087`), but it
  consistently changes masks and shrinks mask area. Keep this as a possible
  relaxed-quality tradeoff only; it cannot be promoted under the current exact
  tracking parity gate.
- The same shared-frame contract was also applied to 1024. The 15-run C++
  `q8_0` mean/median were `44.85`/`44.7 ms/frame`. A matching official PyTorch
  bf16 shared-frame run reported `62.24 ms/frame`, so under this input contract
  C++ is faster than PyTorch at 1024 as well. The quality check remains strong:
  min/mean mask IoU `0.9935`/`0.9951`. Candidate-index parity is not exact on
  late frames because multiple official candidates tie at the rounded bf16 IoU
  score, but selected masks remain nearly identical. For speed claims, always
  report the input contract (`video` vs extracted shared frames), because the
  older video-input baseline gave a different PyTorch timing. Refreshing the
  official PyTorch bf16 1024 video-input row with the current script produced
  `55.87 ms/frame`; the matching C++ `q8_0` video-input 15-run mean/median were
  `44.47`/`44.4 ms/frame`, so the current Base+ q8_0 path is also faster than
  the refreshed official PyTorch video row.
- A D56 FATTN direct-output fusion for the `cols_per_warp == 8`, `no mask`,
  `np == 1`, `no fixup` case looked attractive in the standalone parity harness
  (`0.226 ms` baseline to `0.139 ms` on the D56 window shape with relaxed CPU
  tolerance), but it failed application-level parity (`0 / 10` exact mask
  hashes). The shared-memory writeback layout is not equivalent to the naive
  direct register write for this tile shape, so this probe was reverted.
- Enabling the existing MMF bias+GELU epilogue fusion by default was also
  tested as a graph-level fusion candidate. The first three-run smoke looked
  mildly positive and had exact parity, but the 15-pair benchmark showed no
  robust improvement: default-on `44.53 ms/frame` mean, disabled
  `44.55 ms/frame` mean, paired median delta `0.0 ms/frame`, exact mask hashes
  `15 / 15`. It remains opt-in via `GGML_CUDA_ENABLE_MMF_BIAS_GELU_FUSION=1`.
- Widening the existing non-sequential MMQ bias fusion was tested with
  `GGML_CUDA_ENABLE_MMQ_NONSEQ_BIAS_FUSION=all_exact` and `proj`. Both modes
  failed strict tracking parity (`0 / 10` exact mask hashes in every checked
  pair). `proj` also regressed speed (`46.36 ms/frame` mean vs `44.74`
  default). These modes should remain diagnostic only; the current default
  named-bias allowlist is the safe boundary.

Evidence:

- `outputs/fattn-ntileskv1-no-fixup-q8_0-20260518/`
- `outputs/fattn-ntileskv1-no-fixup-speed7-q8_0-20260518/`
- `outputs/fused-axis-bcast-q8_0-parity-20260518/`
- `outputs/fused-axis-bcast-q8_0-speed7-20260518/`
- `outputs/mmq-q8_0-d4-warp-col-quant-profile-20260518/`
- `outputs/mmq-q8_0-d4-warp-col-quant-speed7-20260518/`
- `outputs/mmq-q8_0-d4-warp-col-quant-cols-speed5-20260518/`
- `outputs/mmq-strided-src1-quant-fusion-q8_0-speed7-20260518/`
- `outputs/mmq-strided-src1-quant-fusion-q8_0-profile-20260518/`
- `outputs/mmq-q8_0-d4-warp-col-quant-512-speed7-20260518/`
- `outputs/profile-q8_0-512-current-20260518/`
- `outputs/node-profile-q8_0-512-current-20260518/`
- `outputs/mmq-q8_0-xmax-sweep-20260518/`
- `outputs/mmq-q8_0-xmax104-app-ab7-20260518/`
- `outputs/mmq-q8_0-4096-x104-shapegate-ab7-20260518/`
- `outputs/quality-sam2-coord-no-half-q8_0-512-20260518/`
- `outputs/quality-logits-q8_0-512-current-20260518/`
- `outputs/quality-refresh-f32-512-current-20260518/`
- `outputs/quality-refresh-f32-512-python-fp32-20260518/`
- `outputs/quality-refresh-f32-512-python-binarize-true-20260518/`
- `outputs/quality-refresh-f32-512-python-binarize-false-20260518/`
- `outputs/quality-refresh-f32-512-python-fill0-20260518/`
- `outputs/quality-refresh-f32-512-python-fill8-20260518/`
- `outputs/mem-mask-compare-f32-512-20260518/`
- `outputs/initial-candidates-f32-512-20260518/`
- `outputs/quality-refresh-f32-512-shared-frames-20260518/`
- `outputs/quality-refresh-q8_0-512-shared-frames-20260518/`
- `outputs/quality-refresh-q8_0-512-shared-frames-bf16-20260518/`
- `outputs/profile-q8_0-512-shared-frames-20260518/`
- `outputs/current-default-q8_0-512-shared-speed15-20260518/`
- `outputs/mmq-q8_0-196-x64-512-shared-ab9-20260518/`
- `outputs/current-default-q8_0-1024-shared-speed15-20260518/`
- `outputs/quality-refresh-q8_0-1024-shared-frames-bf16-20260518/`
- `outputs/official-video-1024-bf16-refresh-20260518/`
- `outputs/current-default-q8_0-1024-video-speed15-refresh-20260518/`
- `outputs/fattn-direct-cols8-fusion-20260518/`
- `outputs/fattn-direct-cols8-fusion-q8_0-smoke-20260518/`
- `outputs/fusion-existing-switches-q8_0-smoke-20260518/`
- `outputs/mmf-bias-gelu-default-on-q8_0-ab15-20260518/`
- `outputs/nonseq-mmq-bias-modes-q8_0-ab5-20260518/`
