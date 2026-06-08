# SAM3 and SAM3.1 Status

Date: 2026-05-22

This note records the current SAM3/SAM3.1 coverage and the benchmark contract
used before treating this repository as a Rust-wrapper baseline.

## Coverage

- SAM2.1 tracking is the mature path. It has same-input C++ versus official
  Python comparisons, strict FP32 controls, repeat statistics, and mask-quality
  checks.
- SAM3 full models load and run in C++ for the existing `sam3-*.ggml` files.
  The current C++ path is not yet faster than official Python on the same input
  contract.
- SAM3 visual-only models load separately through the visual tracker path. They
  are useful for point-prompt tracker coverage but are not a substitute for the
  full text-prompt SAM3 detector/tracker path.
- SAM3.1 official Python runs through the current upstream `sam3` repository
  with `build_sam3_predictor(version="sam3.1")`.
- SAM3.1 C++ is not implemented yet. The repository does not currently contain
  a SAM3.1 GGML checkpoint or a multiplex-specific loader/graph path.

## Benchmark Contract

Speed and quality claims are valid only when the compared rows use:

- the same decoded source-frame resolution,
- the same frame range,
- the same prompt,
- the same model input resolution,
- the same precision and TF32 policy when comparing strict numerical behavior.

Rows that intentionally vary source resolution, encode size, precision, or
prompt are scaling or coverage rows. They must not be used as direct C++ versus
official Python win/loss evidence.

## Current Evidence

Recent checks confirmed the execution paths:

| Path | Input | Result | Evidence |
| --- | --- | ---: | --- |
| C++ `sam3-f16` | 3 frames, 1008 model input, text prompt | 755.8 ms/frame | `outputs/sam3-textprompt-pe-cache-r3-20260522/summary.json` |
| Official Python SAM3 | same 3 frames, 1008 model input, text prompt | 213.3 ms/frame | `outputs/sam3-textprompt-pe-cache-r3-20260522/summary.json` |
| Official Python SAM3.1 | 2-frame smoke | 468.2 ms/frame | `outputs/sam31-official-smoke-20260522-r4/summary.json` |
| Official Python SAM3.1 | 3 frames, 1008 model input, text prompt | 373.76 ms/frame | `outputs/sam3-sam31-current-contract-r3-20260522/summary.json` |

The earlier C++ SAM3 smoke rows that did not pass a text prompt into
`sam3_video_params` are invalid for C++ versus official SAM3 text-prompt speed
claims. The benchmark now accepts `--text-prompt` and the matrix script forwards
it to the C++ benchmark.

The first accepted SAM3 optimization in this pass caches SAM3 neck positional
embeddings in the state. In the profile path, the second encode's PE build drops
from roughly `139 ms` to effectively zero. Overall text-prompt tracking improves
from about `922.8 ms/frame` to `792.1 ms/frame` in the 2-frame profile smoke,
and the 3-repeat matrix row now averages `755.8 ms/frame`.

Later same-contract SAM3 work changed the benchmark to initialize the tracker
from the frame-0 text prompt and measure propagation frames, matching the
official Python `add_prompt` then `propagate_in_video` contract. The accepted
CUDA changes include a 2x2 stride-2 transposed-convolution GEMM path and a
Blackwell cuBLAS path that writes f16-weight batched GEMMs directly to f32
outputs. The current evidence is:

| Path | Input | Result | Evidence |
| --- | --- | ---: | --- |
| C++ `sam3-f16` | 3 frames, 1008 model input, text prompt, CUDA | 325.67 ms/frame | `outputs/sam3-blackwell-compute32-default-r3-20260522/summary.json` |
| Official Python SAM3 | same 3 frames, bf16 autocast, TF32 on | 118.98 ms/frame | `outputs/sam3-blackwell-compute32-default-r3-20260522/summary.json` |
| C++ `sam3-q8_0` | same 3 frames, CUDA | 236.7 ms/frame | `outputs/sam3-quant-cublaslt-fallback-r1-20260522/summary.json` |
| C++ `sam3-q4_1` | same 3 frames, CUDA | 234.4 ms/frame | `outputs/sam3-quant-cublaslt-fallback-r1-20260522/summary.json` |
| C++ `sam3-q4_0` | same 3 frames, CUDA | 223.0 ms/frame | `outputs/sam3-quant-cublaslt-fallback-r1-20260522/summary.json` |

The current C++ SAM3 path is therefore still slower than official Python even
for existing quantized models. The next bottleneck is the ViT image encoder:
profiled propagation is dominated by the MLP and QKV/projection GEMMs inside the
32 ViT blocks, not by tracker memory propagation.

The next CUDA pass added one accepted f16-weight ViT change and one
non-accepted low-precision attention experiment:

- cuBLASLt broadcast-bias fusion now accepts f16 weights with f32 activations
  and supports flattened `ne[2] * ne[3]` window batches. This removes the
  separate QKV/projection bias ADD kernels and falls back to the old path if
  cuBLASLt has no valid algorithm.
- `SAM3_FAST_F16_VIT_ATTENTION=1` exists in the test-stage helper path, but it
  was not wired into the production `sam3_vit_block_forward()` graph used by
  `sam3_benchmark`. A production experiment that kept Q in f32 and K/V in the
  low-precision weight type did run, but it regressed BF16 tracking to
  `328.33 ms/frame` and changed mask hashes (`min_bbox_iou=0.9994216035820871`,
  `max_bbox_delta_px=0.027`, `max_score_abs_delta=0.001222`). It is not
  retained in the production graph. The real root item remains a lower-overhead
  ggml CUDA FlashAttention/RoPE path, not extra application-level cast nodes.

Latest same-contract timing:

| Path | Input | Result | Evidence |
| --- | --- | ---: | --- |
| C++ `sam3-f16`, default after cuBLASLt 4D bias fusion | 3 frames, 1008 model input, text prompt, CUDA | 320.9 ms/frame | `outputs/sam3-cublaslt-f16-bias-4d-r3-20260522/summary.json` |
| C++ `sam3-f16`, env-labelled run | same 3 frames | 316.2 ms/frame | `outputs/sam3-fast-f16-attn-r3-20260522/summary.json` |
| C++ `sam3-f16`, env-labelled refreshed run | same 3 frames | 315.97 ms/frame | `outputs/sam3-sam31-current-contract-r3-20260522/summary.json` |
| C++ `sam3-bf16` | same 3 frames, BF16 GGML weights, CUDA | 314.0 ms/frame | `outputs/sam3-bf16-cublaslt-bias-r3-20260522/summary.json` |
| C++ `sam3-f16`, regenerated local model | same 3 frames, CUDA, Python skipped | 316.97 ms/frame | `outputs/sam3-f16-bf16-current-r3-20260522/summary.json` |
| C++ `sam3-bf16`, current local model | same 3 frames, CUDA, Python skipped | 313.2 ms/frame | `outputs/sam3-f16-bf16-current-r3-20260522/summary.json` |
| Official Python SAM3 | same 3 frames, bf16 autocast, TF32 on | 118.67 ms/frame | `outputs/sam3-fast-f16-attn-r3-20260522/summary.json` |
| Official Python SAM3 | same 3 frames, refreshed same-contract run | 118.54 ms/frame | `outputs/sam3-sam31-current-contract-r3-20260522/summary.json` |
| Official Python SAM3 | same 3 frames, BF16+TF32 comparison run | 119.16 ms/frame | `outputs/sam3-bf16-cublaslt-bias-r3-20260522/summary.json` |
| Official Python SAM3.1 | same 3 frames, bf16 autocast, TF32 on | 373.76 ms/frame | `outputs/sam3-sam31-current-contract-r3-20260522/summary.json` |
| C++ `sam3-bf16`, propagate-only tracker-neck contract | same 3 frames, CUDA | 298.37 ms/frame | `outputs/sam3-bf16-propagate-trkneck-r3-20260522/summary.json` |
| C++ `sam3-bf16`, window-linear flattened GEMM | same 3 frames, CUDA | 266.07 ms/frame | `outputs/sam3-bf16-window-linear-flat-r3-20260522/summary.json` |
| C++ `sam3-bf16`, matrix rerun after window-linear flatten | same 3 frames, CUDA | 265.83 ms/frame | `outputs/sam3-bf16-window-linear-flat-matrix-20260522/summary.json` |
| Official Python SAM3, matrix rerun | same 3 frames, bf16 autocast, TF32 on | 119.34 ms/frame | `outputs/sam3-bf16-window-linear-flat-matrix-20260522/summary.json` |
| C++ `sam3-bf16`, all ViT linear inputs flattened | same 3 frames, CUDA | 222.53 ms/frame | `outputs/sam3-bf16-all-vit-linear-flat-r3-20260522/summary.json` |
| C++ `sam3-bf16`, matrix rerun after all-ViT flatten | same 3 frames, CUDA | 221.4 ms/frame | `outputs/sam3-bf16-all-vit-linear-flat-matrix-20260522/summary.json` |
| Official Python SAM3, matrix rerun | same 3 frames, bf16 autocast, TF32 on | 119.71 ms/frame | `outputs/sam3-bf16-all-vit-linear-flat-matrix-20260522/summary.json` |
| C++ `sam3-bf16`, experimental 256-thread ncols=1024 norm | same 3 frames, CUDA | 214.27 ms/frame | `outputs/sam3-bf16-norm1024-256-r3-20260522/summary.json` |
| C++ `sam3-bf16`, same norm experiment matrix rerun | same 3 frames, CUDA | 216.33 ms/frame | `outputs/sam3-bf16-norm1024-256-matrix-20260522/summary.json` |
| Official Python SAM3, same matrix rerun | same 3 frames, bf16 autocast, TF32 on | 118.92 ms/frame | `outputs/sam3-bf16-norm1024-256-matrix-20260522/summary.json` |
| C++ `sam3-bf16`, default BF16 linear/fc2 inputs | same 3 frames, CUDA | 218.53 ms/frame | `outputs/sam3-bf16-default-linear-fc2-input-matrix-20260522/summary.json` |
| Official Python SAM3, same matrix rerun | same 3 frames, bf16 autocast, TF32 on | 119.48 ms/frame | `outputs/sam3-bf16-default-linear-fc2-input-matrix-20260522/summary.json` |
| C++ `sam3-bf16`, GELU->CPY fusion plus default BF16 `attn.proj` input | same 3 frames, CUDA | 207.0 ms/frame | `outputs/sam3-bf16-attn-proj-input-default-20260522/summary.json` |

This is an incremental improvement, not a win. Relative to the earlier
Blackwell f32-output GEMM baseline, default C++ moved from `325.67` to
`320.9 ms/frame`. The refreshed C++ f16 row is `315.97 ms/frame`, while
official Python SAM3 is `118.54 ms/frame`; Python is still about `2.67x`
faster on this contract. The BF16 GGML path now loads and runs, and cuBLASLt
bias fusion handles BF16 weights with f32 activations, but the measured BF16 row
is `314.0 ms/frame` versus official Python SAM3 BF16+TF32 at
`119.16 ms/frame`; Python is still about `2.64x` faster. Combining
`SAM3_FAST_F16_VIT_ATTENTION=1` with `SAM3_FAST_F16_VIT_MLP=1` was not accepted
because the 3-repeat mean regressed to `323.1 ms/frame`.
The later production low-precision K/V attention experiment also regressed to
`328.33 ms/frame` (`outputs/sam3-bf16-lowp-kv-r3-20260522/summary.json`), so it
is recorded only as negative evidence.
Changing the existing MLP activation-cast experiment to cast BF16 model
activations to BF16 instead of f16 also regressed to `327.6 ms/frame`
(`outputs/sam3-bf16-fast-mlp-r3-20260522/summary.json`) and changed mask hashes
against default BF16 (`outputs/sam3-bf16-fast-mlp-parity-20260522/compare.json`);
that code path is not retained.
Enabling CUDA graph capture with `SAM3_CUDA_ENABLE_GRAPHS=1` also did not help
this SAM3 BF16 contract: the 3-repeat mean was `318.43 ms/frame`
(`outputs/sam3-bf16-cuda-graphs-r3-20260522/summary.json`), slower than the
current default BF16 row.

The latest accepted CUDA/SAM3 benchmark-contract changes are larger but still
not enough to beat Python. The benchmark now uses `sam3_propagate_frame()` for
SAM3 text-prompt runs after the frame-0 text detection, matching the official
Python `add_prompt` then `propagate_in_video` measurement range. That path also
uses tracker-neck-only image encoding during propagation, so it no longer builds
the detector neck after initialization. This moved BF16 from the prior
`314.37 ms/frame` local default evidence to `298.37 ms/frame`. The following
accepted graph-level ViT change flattens window-batch linear inputs from
`[C,24,24,9]` into `[C,5184]` before QKV/projection GEMMs, matching PyTorch
linear's large-GEMM shape instead of launching nine smaller batched GEMMs. That
reduced BF16 to `266.07 ms/frame` and preserved exact JSONL parity against the
propagate-only baseline (`min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`,
`max_score_abs_delta=0.0`, `mask_hash_equal_rows=3`). The next accepted graph
change applies the same flattening to all ViT linear inputs, including global
blocks and MLP `fc1`/`fc2`, so `[C,72,72,1]` is also sent to cuBLASLt as
`[C,5184]`. That reduced BF16 further to `222.53 ms/frame` and preserved exact
JSONL parity against the window-linear baseline (`min_bbox_iou=1.0`,
`max_bbox_delta_px=0.0`, `max_score_abs_delta=0.0`,
`mask_hash_equal_rows=3`). The same matrix contract now reports C++
`221.4 ms/frame` versus official Python SAM3 BF16+TF32 `119.71 ms/frame`;
Python is still about `1.85x` faster.

An experimental ggml CUDA LayerNorm scheduler changes the ncols=1024 norm
kernel from 1024 threads per row to 256 threads per row when
`GGML_CUDA_NORM_1024_MODE=2` is set. This targets the SAM3 ViT residual
add+norm+affine fusion rather than an application-level toggle. The 3-repeat
BF16 row improved to `214.27 ms/frame`, and the same matrix contract reported
C++ `216.33 ms/frame` versus official Python `118.92 ms/frame`; Python is
still about `1.82x` faster. Profiling confirms the targeted node group dropped:
`ADD dst=f32[1024,72,72,1]` under `sam3_encode` moved from drop-max
`46.58 ms` to `24.10 ms`
(`outputs/sam3-bf16-norm1024-256-profile-20260522/sam3-encode-hotspots.json`).
This is not a strict same-hash C++ parity change: compared with the all-ViT
flatten baseline, frame 0 has `min_bbox_iou=0.9984537526132371`,
`max_bbox_delta_px=0.066`, `max_score_abs_delta=0.001487`, and
`mask_hash_equal_rows=2`
(`outputs/sam3-bf16-norm1024-256-r3-20260522/compare-vs-all-flat.json`).
Combining this norm scheduler with `SAM3_BF16_VIT_MLP_CHAIN=1` improved speed
further to `212.7 ms/frame`, but it increased the frame-0 bbox/score delta
against the C++ baseline (`min_bbox_iou=0.9978199759765972`,
`max_bbox_delta_px=0.122`, `max_score_abs_delta=0.002429`), so it remains an
experiment rather than a default path.

Additional post-flatten work removed most of the hidden cuBLASLt input
conversion cost, but did not close the Python gap:

- BF16 ViT linear inputs and MLP `fc2` inputs now default to explicit BF16
  activation tensors before the cuBLASLt fused-bias matmul. This removes a
  large part of the hidden F32-to-BF16 input conversion from the cuBLASLt path
  while preserving exact JSONL parity against the all-ViT flatten baseline
  (`min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, `max_score_abs_delta=0.0`,
  `mask_hash_equal_rows=3`). The repeat matrix row is now C++
  `218.53 ms/frame` versus official Python `119.48 ms/frame`, so Python is
  still about `1.83x` faster. A timing-instrumented run showed why this is the
  right remaining target: before the `fc2` default, the fused cuBLASLt path
  still spent about `45.85 ms` converting `dst=f32[1024,5184,1,1]` inputs from
  F32 to BF16 across the measured calls
  (`outputs/sam3-bf16-cublaslt-bias-timing-20260522/summary.json`).
- BF16 `attn.proj` input now defaults on. It explicitly casts FlashAttention
  output to BF16 before `attn.proj`, matching the Python autocast direction and
  avoiding cuBLASLt's internal F32-to-BF16 input conversion for that projection.
  The old behavior can be restored with
  `SAM3_DISABLE_BF16_VIT_ATTN_PROJ_INPUT=1`. The full-mask A/B check against
  the old behavior is exact on the checked 3-frame clip (`min_bbox_iou=1.0`,
  `max_bbox_delta_px=0.0`, `max_score_abs_delta=0.0`,
  `mask_hash_equal_rows=3`), and the paired drop-first timing moved from
  `207.6` to `207.0 ms/frame`
  (`outputs/sam3-bf16-attn-proj-input-default-20260522/summary.json`). This is
  a small conversion-cleanup win, not a root-level speedup by itself.
- `SAM3_ENABLE_VIT_POS_CACHE=1` replaces the ViT pos-embed `REPEAT` node with a
  state-owned backend tensor. It preserved exact JSONL parity against the
  no-cache path, but the 3-repeat mean was `223.1 ms/frame`, so the extra cached
  tensor does not currently pay for itself.
- `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX` can select a nonzero cuBLASLt heuristic
  candidate for the fused bias matmul path. A coarse single-run sweep over
  indices `0..5` under the norm256 experiment found no large win; index 4 was
  only `214.5 ms/frame` versus index 0 at `215.4 ms/frame`.

Two post-flatten toggles were rejected. `SAM3_CUDA_ENABLE_GRAPHS=1` preserved
parity but regressed the 3-repeat mean to `226.63 ms/frame`
(`outputs/sam3-bf16-all-flat-cuda-graphs-r3-20260522/summary.json`).
`GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION=1` was only marginally faster
at `221.43 ms/frame`, but it changed one checked mask hash and produced small
bbox/score deltas (`min_bbox_iou=0.9993716774944309`,
`max_bbox_delta_px=0.042`, `max_score_abs_delta=0.000242`), so it is not an
accepted default path.

An experimental ggml/SAM3 BF16 linear-output path was also tested with
`SAM3_BF16_VIT_LINEAR_OUTPUT=1`. It adds a `ggml_mul_mat_cast` graph builder,
lets CUDA `mul_mat` write BF16 destinations, and enables BF16/F32 broadcast add
for the limited ViT `attn.proj` global-block and `mlp.fc2` output slice. This
is a correctness-oriented foundation for lower-precision activation work, but
it is not an accepted speed optimization: the 3-repeat SAM3 BF16 row regressed
to `349.37 ms/frame` (`outputs/sam3-bf16-linear-output-r3-20260522/direct-summary.json`).
The first failure mode was cuBLASLt refusing direct `D=BF16` bias-epilogue
heuristics for those `mlp.fc2` and global `attn.proj` shapes. A follow-up
fallback makes that path use a fused cuBLASLt F32 temporary followed by one
F32-to-BF16 conversion; it removes the large regression but still does not win:
the 3-repeat row is `317.57 ms/frame`
(`outputs/sam3-bf16-linear-output-f32temp-r3-20260522/direct-summary.json`).
The default BF16 path after this implementation remains in the prior range at
`314.73 ms/frame`
(`outputs/sam3-bf16-default-after-linear-output-r3-20260522/direct-summary.json`).
The result confirms that merely changing selected linear outputs to BF16 is not
enough; the next useful slice needs to fuse a longer BF16 activation chain or
remove the remaining conversion and standalone add costs.

A follow-up fixed the experimental flag split so
`SAM3_BF16_VIT_QKV_CHAIN=1` actually makes the ViT QKV projection write BF16.
This is not an accepted optimization. The single-run same-contract smoke moved
from `314.1 ms/frame` for default BF16 to `328.5 ms/frame` with the QKV-chain
flag (`outputs/sam3-bf16-qkv-chain-real-20260522/`). The bbox output stayed
close (`min_bbox_iou=0.9990246487582046`, `max_bbox_delta_px=0.037`,
`max_score_abs_delta=0.00371`), but the checked mask hashes did not match
default BF16. A profile run with split FlashAttention conversion timing shows
the immediate cause: for BF16/BF16/BF16 attention calls, the compatibility path
currently converts Q to F32 and K/V to F16 before the existing MMA kernel,
averaging about `0.083 ms`, `0.081 ms`, and `0.080 ms` per call respectively
over `96` BF16 attention calls. This validates the root target but not the
implementation: to win, the CUDA FlashAttention path must consume the SAM3 BF16
Q/K/V contract directly, or QKV/RoPE/attention must be fused so these format
round trips disappear.

Another BF16 activation-chain experiment keeps ViT MLP `fc1 -> bias ->
GELU_ERF` in BF16 and lets `fc2` consume BF16 activations via cuBLASLt
(`SAM3_BF16_VIT_MLP_CHAIN=1`). It is also not an accepted optimization. On the
same build, default BF16 was `314.37 ms/frame`
(`outputs/sam3-bf16-default-after-mlp-chain-r3-20260522/summary.json`). A
single MLP-chain smoke improved from `312.5` to `309.4 ms/frame`, but the
3-repeat mean was `314.97 ms/frame`
(`outputs/sam3-bf16-mlp-chain-r3-20260522/summary.json`), statistically no
better than the current default BF16 evidence. The output stayed close but not
same-hash against default BF16 (`min_bbox_iou=0.9952524247836044`,
`max_bbox_delta_px=0.226`, `max_score_abs_delta=0.00181`,
`mask_hash_equal_rows=0`). Profiling confirmed that `fc2` can consume BF16
activations through cuBLASLt, but `fc1` BF16 output still uses an F32 temporary
plus F32-to-BF16 conversion. Forcing direct cuBLASLt `D=BF16` with
`GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1` failed the heuristic for all `96`
`[4736,72,72,1]` MLP `fc1` shapes and regressed a smoke run to
`321.7 ms/frame`
(`outputs/sam3-bf16-mlp-chain-direct-bf16-20260522/`). This again points to a
custom SAM3 ViT fusion/kernel rather than another graph-level cast toggle.
Disabling cuBLASLt bias fusion entirely while keeping the MLP-chain path also
regressed to `338.6 ms/frame`
(`outputs/sam3-bf16-mlp-chain-no-cublaslt-r1-20260522/`), so the current
cuBLASLt fused-bias path is still the better fallback even though it is not good
enough to beat Python.

The FlashAttention scheduling toggle is also not the missing win. Disabling
stream-k with `GGML_CUDA_DISABLE_FATTN_STREAM_K=1` regressed the current BF16
contract to `327.93 ms/frame`
(`outputs/sam3-bf16-disable-fattn-streamk-r3-20260522/summary.json`). This
leaves the same root target: reduce materialization/conversion around SAM3 ViT
linear, RoPE, and FlashAttention kernels instead of selecting a different
existing scheduler.

The latest BF16 profile confirms the remaining root bottleneck is still the
SAM3 ViT image encoder. The largest drop-max CUDA node groups are the ViT
`MUL_MAT` shapes `f32[1024,72,72,1]`, `f32[4736,72,72,1]`, and
`f32[3072,24,24,9]`, followed by window/global FlashAttention and neck
upsampling kernels (`outputs/sam3-bf16-current-profile-20260522/`).

The next accepted CUDA-side optimization fuses `UNARY(GELU_ERF)` followed by
`CPY(F32->BF16/F16)` in the ggml CUDA evaluator. This preserves the graph
contract used by the current SAM3 BF16 path: compute exact F32 `GELU_ERF`, then
round the result to the lower-precision activation consumed by the next linear.
It is enabled by default and can be disabled with
`GGML_CUDA_DISABLE_UNARY_CPY_FUSION=1`. The same-contract SAM3 BF16 matrix now
reports C++ `207.30 ms/frame` vs official Python BF16+TF32
`118.36 ms/frame`
(`outputs/sam3-bf16-gelu-cpy-fusion-matrix-20260522/summary.json`). The
full-mask A/B check against the disabled path is exact on the checked 3-frame
clip: `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`,
`max_score_abs_delta=0.0`, and `mask_hash_equal_rows=3`
(`outputs/sam3-bf16-gelu-cpy-fusion-fullmask-parity-20260522/compare.json`).
The current gap to Python is therefore smaller but still large: Python remains
about `1.75x` faster on the comparable BF16+TF32 row.

After the GELU->CPY fusion, a cuBLASLt bias timing run shows that the dominant
strict-default cost is no longer input conversion. The largest BF16-input ViT
GEMMs spend about `90.02 ms`, `89.13 ms`, and `58.65 ms` in matmul work across
the measured calls, while the remaining F32-input conversion rows are much
smaller (`7.41 ms` for `dst=f32[1024,5184,1,1]` and `4.82 ms` for
`dst=f32[256,5184,1,1]` in
`outputs/sam3-bf16-gelu-cpy-cublaslt-bias-timing-20260522/summary.json`). The
next strict-parity target is therefore the ViT GEMM/FlashAttention/norm kernels
themselves, not another application-level cast toggle.

A matching official Python torch-profiler run confirms why this is the right
comparison point. In `outputs/sam3-python-profiler-20260522-r2/summary.json`,
the SAM3 BF16+TF32 propagate section uses PyTorch/CUTLASS fused BF16 kernels:
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16...` accounts for about
`56.46 ms` across 104 calls, `_addmm_activation` / `gemm_gelu_bf16` accounts
for about `28.94 ms` across 32 MLP `fc1+GELU` calls, and PyTorch flash-attention
kernels account for the main attention work. C++ still spends most of its time
in separate ggml ViT GEMM and FlashAttention paths, so closing the gap requires
a real CUDA kernel path comparable to PyTorch's BF16 fused linear/activation and
flash-attention implementations.

`GGML_CUDA_NORM_1024_MODE=1` and `GGML_CUDA_NORM_1024_MODE=2` remain
performance experiments, not strict defaults. With the GELU->CPY fusion already
enabled, both modes reach about `200 ms/frame`, but they change the initial
frame by a small amount (`max_bbox_delta_px<=0.066`,
`max_score_abs_delta<=0.001487`, `mask_hash_equal_rows=2/3` in
`outputs/sam3-bf16-gelu-cpy-plus-norm1024-sweep-20260522/`). This is probably
acceptable as low-precision numerical drift, but it is not same-hash parity.

Two additional CUDA-side checks were rejected after the GELU->CPY fusion. First,
cuBLASLt `BIAS+GELU_ERF` epilogue still changes the checked mask hashes
(`mask_hash_equal_rows=0/3`) even though the bbox/score drift is small
(`outputs/sam3-bf16-gelu-cpy-plus-cublaslt-gelu-erf-20260522/compare.json`).
Second, caching the cuBLASLt bias-fusion heuristic algorithm preserved exact
output parity but did not improve paired timing (`206.5 ms/frame` disabled vs
`206.7 ms/frame` enabled in
`outputs/sam3-bf16-cublaslt-algo-cache-r6-20260522/summary.json`), so it was
not kept as production code.
Third, forcing BF16 Q/K/V FlashAttention onto the vector kernel was tested with
the BF16 QKV chain and `attn.proj` BF16 input. It preserved exact output parity
against the same QKV-chain path but did not improve speed (`223 ms/frame` for
both paths in
`outputs/sam3-bf16-qkv-chain-force-bf16-fattn-vec-20260522/`), so the
experiment switch was removed.
Fourth, an `ADD -> NORM -> MUL -> ADD -> CPY(BF16/F16)` fusion was prototyped
to make the ViT residual LayerNorm path write the lower-precision consumer
tensor directly. It preserved exact full-mask parity, but paired timing was
unchanged (`206.8 ms/frame` disabled and enabled in
`outputs/sam3-bf16-add-norm-cpy-fusion-20260522/summary.json`), so the code was
not kept.
Fifth, a `MUL_MAT -> ADD -> GELU_ERF -> CPY(BF16/F16)` cuBLASLt fusion was
prototyped to mirror Python's fused BF16 MLP `fc1+GELU` shape more closely. It
did not win: paired timing regressed from `206.8` to `207.8 ms/frame`, and the
full-mask A/B check changed all checked mask hashes
(`mask_hash_equal_rows=0/3`,
`outputs/sam3-bf16-cublaslt-gelu-cpy-fusion-20260522/summary.json`). The code
was removed.
Sixth, increasing the cuBLASLt bias-fusion workspace did not expose a faster
heuristic for the dominant ViT GEMM shapes. A diagnostic
`GGML_CUDA_CUBLASLT_BIAS_WORKSPACE_MB` knob now exists for repeatable sweeps,
but `8`, `32`, and `256` MiB returned the same top heuristic counts and nearly
identical shape-level matmul times
(`outputs/sam3-bf16-cublaslt-workspace-timing-20260522/`). Sweeping global
cuBLASLt heuristic index `0..2` likewise showed index `0` remains best for the
dominant shapes
(`outputs/sam3-bf16-cublaslt-algo-timing-20260522b/`). The remaining gap is
therefore not a simple workspace or heuristic-index setting.

Additional root-path probes on the current `~207 ms/frame` BF16 baseline also
failed to produce an accepted win:

- For FlashAttention, forcing the BF16 QKV chain made total profiled attention
  work worse: default FATTN summed to `123.794 ms` across the captured rows,
  while the BF16 QKV-chain path summed to `129.940 ms`
  (`outputs/sam3-bf16-fattn-profile-default-vs-qkv-20260522/fattn-summary.json`).
  The native BF16 attention direction is still valid, but this graph-level
  chain adds more conversion cost than it removes.
- Direct cuBLASLt BF16 destination output was rechecked and did not help:
  `206.03 ms/frame` baseline versus `206.70 ms/frame` with direct BF16 dst
  (`outputs/sam3-bf16-direct-bf16-dst-sweep-20260522/summary.json`). Combining
  direct BF16 dst with fast BF16 compute mode likewise stayed in the same range
  (`outputs/sam3-bf16-direct-bf16-fast16bf-20260522/summary.json`).
- A cuBLASLt bias-algorithm heuristic cache preserved exact JSONL parity but
  did not improve mean timing (`206.283` disabled versus `206.333 ms/frame`
  enabled in
  `outputs/sam3-bf16-cublaslt-algo-cache-20260522/summary.json`), so it was
  removed rather than kept as production code.
- Porting the upstream CUDA Programmatic Dependent Launch subset for FATTN
  preserved parity but regressed timing (`207.383` disabled versus
  `207.883 ms/frame` enabled in
  `outputs/sam3-bf16-fattn-pdl-20260522/summary.json`), so it was reverted.
- Enabling cuBLASLt `BIAS+GELU_ERF` across the reshaped ViT MLP `fc1` output
  now reaches the intended nodes, but it is still not acceptable: the 3-repeat
  mean regressed from `207.6` to `218.6 ms/frame` and frame-0 mask hash changed
  (`outputs/sam3-bf16-cublaslt-gelu-erf-env-20260522/summary.json`).
- `CUBLAS_COMPUTE_32F_FAST_16BF` for the cuBLASLt BF16-input bias path
  preserved exact JSONL parity but did not improve timing (`206.88` baseline
  versus `207.22 ms/frame` fast mode in
  `outputs/sam3-bf16-cublaslt-fast16bf-20260522/summary.json`), so it was not
  retained.
- The existing BF16 `fc2` input path is confirmed necessary: disabling it
  regressed `206.4` to `220.4 ms/frame` with exact output parity
  (`outputs/sam3-bf16-mlp-chain-sweep-20260522/summary.json`). Enabling the
  broader `SAM3_BF16_VIT_MLP_CHAIN=1` remained slower and changed frame-0 mask
  hash in the same sweep.
- A full single-run cuBLASLt bias heuristic-index sweep over `0..15` found no
  usable win. The fastest single row was index `5` at `206.2 ms/frame`, but it
  changed frame-0 mask hash; exact-parity indices such as `7`, `8`, `10`, and
  later were effectively the same speed as index `0`
  (`outputs/sam3-bf16-cublaslt-algo-index-sweep-0-15-20260522/summary.json`).
- A fused `GELU_ERF -> BF16` approximation probe that substituted approximate
  `GELU` inside only the fused cpy kernel was also rejected. It did not produce
  a reliable speed win and changed frame-0 mask hashes
  (`outputs/sam3-bf16-fast-gelu-erf-cpy-approx-20260522/summary.json`).
- Replacing the ViT MLP graph activation itself with approximate `GELU` was
  rechecked after the latest flattened/BF16/fused-cpy changes. It still did not
  win: both normal and cuBLASLt-GELU-disabled variants were around
  `216.5 ms/frame` and changed frame-0 mask hashes
  (`outputs/sam3-bf16-current-approx-gelu-20260522/summary.json`).
- A narrow ViT attention probe that cast Q/K/V to BF16 immediately before
  `ggml_flash_attn_ext` also regressed, from `206.17` to `219.23 ms/frame`,
  and changed frame-0 mask hash and score
  (`outputs/sam3-bf16-fattn-inputs-bf16-20260522/summary.json`). This confirms
  the attention gap needs a native BF16/RoPE/FATTN kernel path, not extra graph
  casts around the existing F32 attention graph.

After removing those non-winning probes, the cleaned baseline rebuild reports
`208.23 ms/frame` over three repeats with repeat JSONL equality
(`outputs/sam3-bf16-post-cleanup-baseline-20260522/summary.json`). The active
goal is therefore still open. The remaining performance gap is not a tuning
knob; it requires a real BF16 ViT linear/activation or FlashAttention kernel
path that avoids materializing PyTorch-autocast BF16 work as multiple ggml F32
nodes.

A new accepted CUDA default promotes the ncols=1024 LayerNorm scheduler from an
opt-in experiment to the default path. `GGML_CUDA_NORM_1024_MODE=2` is now the
implicit mode when no override is set; setting `GGML_CUDA_NORM_1024_MODE=0`
restores the previous 1024-thread scheduler. On the same 3-frame SAM3 BF16
matrix contract, this reports C++ `198.97 ms/frame` versus official Python
BF16+TF32 `118.21 ms/frame`
(`outputs/sam3-bf16-norm1024-default-20260522/summary.json`). A direct full-mask
A/B check against the previous scheduler shows small low-precision drift, not
same-hash parity: `min_bbox_iou=0.9989639387064913`,
`max_bbox_delta_px=0.05000000000001137`,
`max_score_abs_delta=0.0008589999999999987`,
`max_abs_mask_area_rel_delta=0.0006079951360389116`, and
`mask_hash_equal_rows=0`
(`outputs/sam3-bf16-norm1024-default-parity-20260522/compare.json`). This is a
real root-kernel speedup, but the active goal is still open because Python
remains about `1.68x` faster on the comparable row.

The synchronized profile after this default change confirms the remaining
surface has moved back to ViT GEMM and attention, not LayerNorm scheduler
tuning. The largest drop-max groups are `MUL_MAT dst=f32[1024,5184,1,1]`
at `113.30 ms`, `MUL_MAT dst=f32[4736,5184,1,1]` at `88.92 ms`,
`MUL_MAT dst=f32[3072,5184,1,1]` at `58.34 ms`, window FlashAttention at
`42.76 ms`, and global FlashAttention at `38.40 ms`
(`outputs/sam3-bf16-norm1024-default-profile-20260522/all-hotspots.json`).
The residual LayerNorm/affine group is now lower (`ADD
dst=f32[1024,72,72,1]` at `25.88 ms`). The next meaningful root target remains
direct BF16 ViT linear outputs and BF16-aware RoPE/FlashAttention, not another
LayerNorm-only change.

Two follow-up checks after the default scheduler change did not produce a
better configuration. `SAM3_ENABLE_VIT_POS_CACHE=1` reported
`200.6 ms/frame`, slightly worse than the default
(`outputs/sam3-bf16-norm1024-pos-cache-20260522/summary.json`).
`GGML_CUDA_NORM_1024_MODE=1` reported `208.97 ms/frame` with a large outlier,
also worse than the mode-2 default
(`outputs/sam3-bf16-norm1024-mode1-20260522/summary.json`). Neither setting is
promoted.

The existing BF16 linear-output and activation-chain switches were also
rechecked after the scheduler default changed, and they still do not win.
`SAM3_BF16_VIT_LINEAR_OUTPUT=1` reports `201.17 ms/frame`
(`outputs/sam3-bf16-norm1024-linear-output-20260522/summary.json`),
`SAM3_BF16_VIT_MLP_CHAIN=1` reports `214.03 ms/frame`
(`outputs/sam3-bf16-norm1024-mlp-chain-20260522/summary.json`), and
`SAM3_BF16_VIT_QKV_CHAIN=1` reports `216.87 ms/frame`
(`outputs/sam3-bf16-norm1024-qkv-chain-20260522/summary.json`). This rules out
the current graph-level BF16 toggles under the new default; a useful next step
must be a narrower CUDA kernel path rather than enabling those flags.
Combining `SAM3_BF16_VIT_LINEAR_OUTPUT=1` with direct cuBLASLt BF16 output
(`GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1`) was also worse at
`205.87 ms/frame`
(`outputs/sam3-bf16-norm1024-linear-output-direct-bf16dst-20260522/summary.json`).
The existing cuBLASLt output-type switch therefore is not enough to reproduce
PyTorch autocast's advantage.
The same direct-output check scoped to the QKV chain was also a regression:
`SAM3_BF16_VIT_QKV_CHAIN=1 GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1` reported
`212.17 ms/frame`
(`outputs/sam3-bf16-norm1024-qkv-chain-direct-bf16dst-20260522/summary.json`).

An attempted FlashAttention scheduler override did not expose a better existing
kernel choice. Forcing the tile kernel regressed to `285.93 ms/frame`
(`outputs/sam3-bf16-norm1024-fattn-force-tile-20260522/summary.json`), while
forcing the vector kernel reported `199.27 ms/frame`
(`outputs/sam3-bf16-norm1024-fattn-force-vec-20260522/summary.json`), effectively
the same as the `198.97 ms/frame` default but not a win. Forcing the MMA path is
not a valid global override for this graph because unsupported head shapes abort
inside the existing switch
(`outputs/sam3-bf16-norm1024-fattn-force-mma-20260522/summary.json`). The
selector therefore should stay on its default path; improving attention now
requires a SAM3-specific BF16/RoPE/FATTN implementation rather than reusing a
different existing ggml FATTN scheduler.

A named cuBLASLt timing run was added for attribution. It confirms the ViT
matmul path is already using BF16 inputs with cuBLASLt bias fusion and has no
input conversion time in the dominant block-linear ops. In the 2-frame
diagnostic run, summed ViT matmul time is led by `vit_mlp_fc2` at `60.73 ms`,
`vit_mlp_fc1` at `60.10 ms`, `vit_qkv` at `39.71 ms`, and `vit_attn_proj` at
`14.57 ms`
(`outputs/sam3-bf16-cublaslt-named-profile-20260522/vit-matmul-summary.json`).
The summary can now be reproduced with
`scripts/summarize_cublaslt_bias_timing.py`; a regenerated shape-grouped
summary is stored at
`outputs/sam3-bf16-cublaslt-named-profile-20260522/vit-matmul-summary-v2.json`.
This run is attribution-only because cuBLASLt timing synchronizes each op.
Global cuBLASLt heuristic changes did not win: `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX=1`
reported `203.3 ms/frame`, index `2` reported `200.97 ms/frame`, and
`GGML_CUDA_CUBLASLT_BIAS_WORKSPACE_MB=128` reported `200.63 ms/frame`
(`outputs/sam3-bf16-norm1024-cublaslt-algo1-20260522/summary.json`,
`outputs/sam3-bf16-norm1024-cublaslt-algo2-20260522/summary.json`,
`outputs/sam3-bf16-norm1024-cublaslt-ws128-20260522/summary.json`). The default
after adding ViT node names remains in the same band at `200.23 ms/frame`
(`outputs/sam3-bf16-norm1024-named-default-20260522/summary.json`).
Enabling ggml CUDA graphs from the SAM3 wrapper also did not help:
`SAM3_CUDA_ENABLE_GRAPHS=1` reported `204.67 ms/frame`
(`outputs/sam3-bf16-norm1024-cuda-graphs-20260522/summary.json`), so the current
wrapper-side graph disable is not the performance gap.

A refreshed synchronized node profile after cleanup reports the same priority
surface: ViT `MUL_MAT` shapes dominate (`dst=f32[1024,5184,1,1]`,
`dst=f32[4736,5184,1,1]`, `dst=f32[3072,5184,1,1]`), followed by
FlashAttention and the fused residual LayerNorm/affine path
(`outputs/sam3-bf16-profile-refresh-20260522/all-hotspots.json`). The profile
uses `GGML_CUDA_PROFILE_NODES=1`, so it is for attribution only, not normal
runtime timing.

The official Python SAM3 and SAM3.1 BF16+TF32 runs keep the dominant ViT linear
outputs in BF16 under CUDA autocast. The reproducible dtype hooks are recorded
in `outputs/sam3-python-dtypes-20260522/sam3-dtypes.json` and
`outputs/sam3-python-dtypes-20260522/sam31-dtypes.json`: block-0 `attn.qkv`
receives `torch.float32` and outputs `torch.bfloat16`, block-0 `attn.proj`
receives and outputs `torch.bfloat16`, block-0 `mlp.fc2` receives and outputs
`torch.bfloat16`, and block-7 `attn.qkv` outputs `torch.bfloat16` for both
versions. In contrast, the current ggml graph creates `GGML_TYPE_F32` results
for `ggml_mul_mat` and `ggml_mul_mat_id`, so the C++ SAM3 ViT path expands
linear outputs back to f32 even when the checkpoint is BF16. This is the current
root precision-contract gap, and it explains why app-level cast experiments did
not close the speed gap: they add conversion work after f32 outputs have already
been materialized. The next accepted optimization target should therefore be a
ggml/SAM3 low-precision linear output path, for example a new op or CUDA fusion
that can write BF16 linear-plus-bias outputs directly for the SAM3 ViT shapes
while preserving measured mask parity.

BF16 parity against the regenerated f16 GGML path is close but not bit-exact:
`outputs/sam3-f16-bf16-current-parity-20260522/compare.json` reports
`min_bbox_iou=0.9975836178358802`, `max_bbox_delta_px=0.102`,
`max_score_abs_delta=0.005622`, and `max_abs_mask_area_rel_delta=0.0008967`,
with mask hashes different on the three checked rows. This is expected for a
different low-precision format and should not be treated as same-hash parity.

An approximate-GELU experiment was also rejected. Replacing ViT MLP
`GELU_ERF` with approximate `GELU` would make the graph eligible for the
existing cuBLASLt `BIAS+GELU` epilogue, but the measured row regressed to
`323.07 ms/frame` even though the 3-frame mask hashes stayed equal
(`outputs/sam3-approx-gelu-r3-20260522/summary.json` and
`outputs/sam3-approx-gelu-parity-20260522/compare.json`).

For SAM2.1 Base+ strict FP32, the latest same-input rows show the current split:

| Encode size | C++ track ms/frame | Official Python ms/frame | Interpretation |
| --- | ---: | ---: | --- |
| 512 | 28.15 | 27.55 | C++ is roughly parity but still slightly slower |
| 1024 | 103.87 | 149.14 | C++ is faster on this strict FP32 contract |

## Next Implementation Targets

1. Promote `scripts/model_matrix_compare.py` and `just model-matrix` as the
   repeatable benchmark entry point for SAM2.1, SAM3, and official SAM3.1.
2. Profile C++ SAM3 full tracking and isolate detector, text encoder, tracker,
   memory update, CUDA graph, and ggml kernel costs.
3. Optimize the dominant C++ SAM3 kernels or graph boundaries before claiming
   parity with official Python. The next root target is true low-precision ViT
   activation support inside ggml CUDA: direct BF16 linear-plus-bias outputs for
   the SAM3 ViT GEMM shapes first, then BF16-aware RoPE and FlashAttention so
   Q/K/V can stay bf16 without forcing Q to f32 or K/V to f16 compatibility
   buffers.
4. Add SAM3.1 checkpoint conversion metadata, loader support, and a
   multiplex-specific C++ execution path.

## SAM3.1 Conversion Bring-up

The SAM3.1 checkpoint is structurally different from the existing SAM3
checkpoint. A local inventory of `sam3.1_multiplex.pt` reports `1623` tensors:

| Prefix | Count |
| --- | ---: |
| `detector` | 1166 |
| `tracker` | 457 |

The checkpoint contains Object Multiplex-specific tensors such as
`tracker.model.output_valid_embed`, `tracker.model.output_invalid_embed`, and
interactive/propagation convolution stacks under
`detector.backbone.vision_backbone`. The existing SAM3 converter now detects
that structure, supports `--inspect-only`, and can write a v4 GGML file with
`model_type=sam3.1` while preserving native SAM3.1 tracker tensor names. This
does not mean full C++ graph execution is complete; it means the converter no
longer has to mislabel SAM3.1 as an old SAM3 tracker file.

The current reproducible inventory command is:

```bash
just sam31-contract
```

This writes `outputs/sam31-checkpoint-contract/inventory.json` and
`outputs/sam31-checkpoint-contract/sam3-vs-sam31-renamed-coverage.json`. The
JSON inventory records tensor names, renamed candidate names, shapes, dtypes,
prefix counts, skipped keys, SAM3.1 marker presence, and a `sam31_contract`
block. The recipe also runs `scripts/check_sam31_contract.py`, which fails if
SAM3.1 detection, slice counts, loader metadata, the multiplex decoder shape
contract, the checked-in `sam3_sam31.h` constants, or the `sam3.cpp`
SAM3.1 registration block drift. The same check also pins the SAM3 GGML file
format bridge and model-type ABI: converter output is v4, the C++ loader accepts
v3..v4, both Python/C++ keep SAM3.1 as model type `4`, the converter writes
`MODEL_TYPE_SAM31` for SAM3.1 checkpoints, and the converter's
`sam31_contract` fields match the manifest-derived decoder contract. The current
contract records the official Python defaults that the C++ loader must preserve:
`image_size=1008`, `backbone_stride=14`,
`multiplex_count=16`, `eval_multiplex_count=16`, `max_num_objects=16`,
`num_maskmem=7`, `max_obj_ptrs_in_encoder=16`, and `tracker_mem_dim=256`. This
should be the source of truth for the next loader slice. The first required
slice is not a speed optimization: it is adding explicit SAM3.1 model metadata
and a dedicated multiplex loader path so SAM3.1 cannot silently share the old
SAM3 tensor contract.

A renamed-key coverage check against the SAM3 checkpoint is also recorded in
`outputs/sam31-checkpoint-contract/sam3-vs-sam31-renamed-coverage.json`. It
shows `1129` common renamed tensors and `0` common shape differences, so the
detector/ViT backbone is structurally close. The blocker is the tracker contract:
SAM3.1 lacks `335` old SAM3 tracker-side tensors after renaming and instead adds
`493` renamed-only tensors: `trk` (`457`) plus Object Multiplex detector convs
(`36`). A correct C++ SAM3.1 implementation therefore needs a dedicated
multiplex tracker model; loading SAM3.1 into the old SAM3 tracker graph would be
a false parity result.

The first C++ Object Multiplex helper is now checked against the official Python
`MultiplexController(random=False)` contract:

```bash
just sam31-multiplex-parity
```

The current summary in `outputs/sam31-multiplex-state/summary.json` reports
`diff_rows=0` for `1`, `16`, `17`, and `33` objects with
`multiplex_count=16`, including assignments, `mux`, and `demux` round-trips.
This only validates the bucket/data-space mapping; it does not mean the
SAM3.1 tracker graph itself is implemented yet.

The coverage JSON now also groups SAM3.1-only tensors by implementation area.
The largest C++ loader/graph slices are:

| SAM3.1-only category | Tensors | Elements |
| --- | ---: | ---: |
| `tracker_multiplex_mask_decoder` | 256 | 8,309,097 |
| `tracker_transformer_ffn_norm` | 40 | 4,209,664 |
| `tracker_multiplex_maskmem_backbone` | 38 | 3,931,120 |
| `tracker_transformer_self_attention` | 32 | 1,052,672 |
| `tracker_transformer_object_cross_attention` | 32 | 1,052,672 |
| `detector_interactive_convs` | 18 | 6,949,632 |
| `detector_propagation_convs` | 18 | 6,949,632 |

This makes the next implementation order concrete: load the multiplex SAM mask
decoder and memory backbone first, then the decoupled tracker transformer, and
only then the detector-side interactive/propagation feature adapters.
The same order is emitted in
`outputs/sam31-checkpoint-contract/sam3-vs-sam31-renamed-coverage.json` as
`sam31_implementation_manifest`:

| Slice | Tensors | Elements | Acceptance |
| --- | ---: | ---: | --- |
| `multiplex_mask_decoder` | 273 | 8,315,573 | C++ decoder tensor dumps match official Python for one prompt frame |
| `multiplex_memory_backbone` | 40 | 3,931,137 | C++ mask memory tensors match official Python before temporal attention |
| `multiplex_tracker_transformer` | 144 | 7,382,784 | C++ propagated mask logits match official Python on the same frame range |
| `multiplex_detector_feature_adapters` | 36 | 13,899,264 | C++ text/point prompt detection inputs match official Python before tracking |

Each manifest entry also includes the full `tensor_names` list, so the C++
loader work can be implemented slice-by-slice without re-deriving names from
the raw checkpoint inventory. The manifest self-check currently reports
`diff_rows=0`, with `493` manifest tensors matching all `493` SAM3.1-only
tensors, no duplicate tensor names, no missing source keys/shapes/dtypes/numel,
and `mask_decoder_contract_ok=true`.

The `multiplex_mask_decoder` manifest also records the decoder architecture
derived from checkpoint tensor shapes. The interactive path remains the normal
SAM-style decoder: one object slot, four mask tokens, and six output tokens
before sparse prompts. The propagation path is the SAM3.1 multiplex path:
`multiplex_count=16`, `mask_token_count=48`, `masks_per_object=3`, and `80`
output tokens before sparse prompts (`16` object-score tokens, `16` IoU tokens,
and `48` mask tokens). This confirms the C++ graph must implement the
`multimask_outputs_only=True` contract from official Python, not the older
four-mask SAM decoder contract.

The same stable constants are available to C++ code in `sam3_sam31.h`. The
`sam31_multiplex_state` helper uses `sam3::sam31::multiplex_count` rather than
hard-coding the slot count, so contract drift is visible in both the Python
manifest gate and the C++ helper build. The SAM3.1-only tensor manifest is now
checked in as `sam3_sam31_tensor_manifest.h` with all `493` native SAM3.1
tensors and checkpoint shapes. `sam3_register_tensors()` now separates the old
SAM3 tracker path from SAM3.1: SAM3.1 skips `neck.trk`, `sam_pe`, `sam_dec`,
`mem_enc`, `mem_attn`, object-pointer, and old standalone tracker tensors, then
registers the manifest tensors using the file-scanned tensor storage types. This
removes the known tensor-count mismatch class for SAM3.1 loader bring-up.

The propagation mask-decoder tensor handles are now bound from the native
manifest registration, and an internal `sam31_build_mux_mask_dec_graph()` graph
builder exists for the official propagation token contract:
`obj_score(16) + iou(16) + mask(48) = 80` tokens, producing multiplexed
`masks`, `iou`, `object_score`, and `mask_tokens` outputs before demux. This is
still not a complete SAM3.1 tracker implementation. The next missing piece is
evidence: add official Python tensor dumps for the same slice, run the C++ graph
against those inputs, and compare the outputs before moving to the memory
backbone.

The first official Python dump command for that evidence now exists:

```bash
just sam31-mux-mask-decoder-case
```

The current small decoder-only case uses the real `sam3.1_multiplex.pt`
propagation decoder weights with deterministic random inputs at `feat_size=4`.
It writes `outputs/sam31-mux-mask-decoder-case/official_mux_mask_decoder_case.npz`,
`cpp_inputs/*.bin`, `expected_cpp_layout/*.bin`, and a summary JSON. The raw
C++ input layout is already in ggml dimension order, for example
`image_feats=[256,4,4,1]`, `projected_feat_s0=[32,16,16,1]`,
`projected_feat_s1=[64,8,8,1]`, and
`extra_per_object_embeddings=[256,16,1]`. The expected C++ output layout is
`masks=[256,48,1]`, `iou_pred=[3,16,1]`,
`mask_tokens_out=[256,48,1]`, and `object_score_logits=[1,16,1]`. This is
Python-side evidence only; the C++ dump runner and numeric comparison are still
the next required step.

The SAM3 GGML header now supports a backward-compatible v4 hparams extension:
v3 files still load, while v4 files append an explicit `model_type` integer
after the existing hparams fields. This is required before a SAM3.1 GGML can
select `SAM3_MODEL_SAM3_1` without overloading `visual_only`.

The active goal status can be summarized without relying on memory of previous
runs:

```bash
just sam3-sam31-goal-audit
```

This writes `outputs/sam3-sam31-goal-audit/summary.json`. It is intentionally a
negative/continuation audit today. The recipe regenerates `cuda-health`,
`sam31-contract`, and `sam31-multiplex-parity` first, then records that CUDA is
blocked, SAM3 C++ has not beaten official Python on the same contract, and
SAM3.1 still lacks the full C++ loader/graph path.

For C++ bring-up, a slice-specific tensor table can be generated directly from
the manifest:

```bash
just sam31-slice-header
```

By default this writes
`outputs/sam31-checkpoint-contract/multiplex_mask_decoder_tensor_names.h`, a
header containing both `std::array<std::string_view, 273>` tensor names and
`std::array<Sam31TensorSpec, 273>` source-key/name/shape/dtype/numel specs for
the first implementation slice. For `multiplex_mask_decoder`, the generated
header also contains `Sam31MaskDecoderContract` constants and static asserts for
the propagation decoder's `16` slots, `48` mask tokens, `3` masks per object,
and `80` pre-prompt output tokens. Set `SAM31_SLICE` to generate another slice.
The recipe also runs a C++23 syntax check on the generated header, including
static asserts that the name/spec table sizes match. To generate and
syntax-check every slice:

```bash
just sam31-slice-headers-all
```

## 2026-05-22 Goal Loop Update

The active goal remains open: C++ SAM3 BF16 still does not beat official Python
SAM3 BF16+TF32 on the same decoded frames, prompt, frame range, and 1008 model
input contract. A SAM3-only rerun is recorded in
`outputs/sam3-goal-current-rerun-sam3only-20260522/summary.json`:

| Row | Mean track ms/frame | Repeats | Notes |
| --- | ---: | ---: | --- |
| C++ `sam3-bf16` | `199.77` | 3 | CUDA, BF16 GGML checkpoint |
| Official Python `sam3` | `119.30` | 3 | BF16 autocast, TF32 enabled |

The comparable speed ratio is `0.597` Python/C++, meaning Python is still about
`1.67x` faster. This confirms the next optimization should stay focused on the
dominant ViT CUDA path, not on benchmark noise or mismatched input resolution.

The benchmark harness now supports `--skip-python-sam2`, exposed through
`MODEL_MATRIX_SKIP_PYTHON_SAM2=1 just model-matrix`, so SAM3/SAM3.1-only
acceptance loops do not spend time measuring unrelated SAM2 Python baselines.

The BF16 ViT linear path was isolated with
`examples/sam3_f32_batched_bench --bf16-cublaslt` on the same dominant 5184-token
shapes. Results are recorded in
`outputs/sam3-bf16-vit-linear-microbench-20260522/`:

| Shape / role | Mean ms | Interpretation |
| --- | ---: | --- |
| `mlp_fc1`, `k=1024 rows=4736 cols=5184` | `0.990` | BF16 inputs, F32 dst, bias epilogue |
| `mlp_fc2`, `k=4736 rows=1024 cols=5184` | `0.895` | BF16 inputs, F32 dst, bias epilogue |
| `qkv`, `k=1024 rows=3072 cols=5184` | `0.588` | BF16 inputs, F32 dst, bias epilogue |
| `attn/proj`, `k=1024 rows=1024 cols=5184` | `0.213` | BF16 inputs, F32 dst, bias epilogue |
| `mlp_fc1` + `CUBLAS_COMPUTE_32F_FAST_16BF` | `0.999` | no win |
| `mlp_fc1` + cuBLASLt `GELU_BIAS` | `1.380` | slower and still not exact `GELU_ERF` parity |
| `mlp_fc1` + direct BF16 dst | failed | no cuBLASLt heuristic for this configuration |

This explains why the remaining gap is not solved by another cuBLASLt mode
switch: the standalone BF16 GEMMs are already close to the per-node timings seen
inside the full C++ profile. The next root optimization should reduce the
non-GEMM boundary cost around ViT attention and activation chains: QKV
reshape/permute/RoPE/FATTN, residual/norm/affine, and exact
`GELU_ERF -> BF16` dataflow.

Two further probes were rejected in this goal loop:

| Probe | Result | Reason |
| --- | ---: | --- |
| Explicit SAM3 ViT K/V F16 input before FATTN | `200.53 ms/frame` | FATTN profile improved, but end-to-end did not |
| `GGML_CUDA_ENABLE_MMF_BIAS_GELU_FUSION=1` | `206.80 ms/frame` | high variance and slower than default |
| FATTN64 inline F32 tile conversion | `215.80 ms/frame`, `0` detections | reduced FATTN-local conversion cost, but broke the accepted end-to-end path |
| Upstream CUDA FATTN PDL launch path | `201.8 ms/frame` vs `200.4 ms/frame` with PDL off | parity OK, but no speed win on the SAM3 RTX 50-series path |

The K/V F16 probe is useful attribution but not an accepted optimization. It
changed FATTN input types from `f32/f32/f32` to `f32/f16/f16` and reduced the
filtered FATTN profile sum from `128.07 ms` to `108.88 ms`, but the node profile
then showed a new explicit `CPY f32->f16` cost of about `16.07 ms`. This means
the real fix is not moving the conversion boundary around; it is eliminating the
boundary with a fused producer or native FATTN input path.

The FATTN64 inline F32 tile-conversion probe reached the same conclusion from
the opposite direction. It reduced the filtered FATTN profile sum from
`128.07 ms` to `111.20 ms` and cut K/V conversion sums from about `17.37 ms` to
about `2.44 ms`, but the attention kernel itself became slower and the
end-to-end benchmark returned no detections. That experiment was reverted; a
usable fix needs a parity-safe native head64 FATTN path rather than an
environment-gated tile-load variant.

The upstream PDL probe was also rejected for this model path. It preserved
`det=1`, but a three-run local comparison recorded in
`outputs/sam3-bf16-pdl-r3-20260522/` showed no improvement: PDL default-on was
about `201.8 ms/frame`, while `GGML_CUDA_PDL=0` was about `200.4 ms/frame`.
PDL may still be useful for other ggml workloads, but it does not close the
SAM3 ViT attention gap here.

The official Python SAM3 profile confirms the same direction. PyTorch is using
BF16 CUTLASS kernels with fused bias/activation for the dominant ViT MLP shapes
(`aten::_addmm_activation` / BF16 CUTLASS GELU). The C++ path needs a ggml CUDA
kernel or fused matmul path that produces the exact downstream low-precision
layout without extra F32 materialization. Until that exists, cuBLASLt mode
switches and standalone casts are not enough to close the remaining gap.

A targeted official Python profiler run is recorded in
`outputs/sam3-python-profiler-20260522/profile.log`. It used the same SAM3
BF16+TF32 setting, same extracted three-frame folder, and `compile=False`.
The relevant CUDA totals were:

| Python op group | Calls | CUDA total |
| --- | ---: | ---: |
| ViT `mlp_fc1` fused `aten::_addmm_activation` | 96 | `88.03 ms` |
| ViT `mlp_fc2` `aten::addmm` | 96 | `89.57 ms` |
| ViT `qkv` `aten::addmm` | 96 | `56.60 ms` |
| ViT global flash attention, `5184 x head64` | 12 | `24.52 ms` |
| ViT window flash attention, `576 x head64` | 84 | `22.95 ms` |

Compared with the C++ CUDA node profile, GEMM timing is already close, while
FATTN is not: C++ filtered FATTN was about `128.07 ms` for the same run shape.
The next accepted optimization therefore needs to replace or substantially
specialize the head64 CUDA FlashAttention path, not just retune cuBLASLt.

A standalone cuDNN SDPA feasibility bench was added as
`examples/cudnn_sdpa_bench.cpp` and exposed as the optional
`sam3_cudnn_sdpa_bench` build target when cuDNN Frontend and cuDNN are
available. It is intended to benchmark the two SAM3 ViT attention shapes before
wiring anything into ggml:

```bash
just cudnn-sdpa-head64
```

That recipe writes the raw cuDNN logs plus `summary.json`, comparing the cuDNN
per-call timings against a fresh `GGML_CUDA_PROFILE_FATTN=1` SAM3 run with the
SAM3 ViT `global=12` and `window=84` call counts.

The bench could not be timed in this loop because the local NVIDIA device fell
out of CUDA visibility after a PCIe AER recovery failure. `nvidia-smi` reported
that it could not determine the GPU handle, and both cuDNN and `sam3_benchmark`
then failed to create a CUDA device. A PCI function reset reached `reset done`
in the kernel log but did not restore CUDA visibility, so the next measurement
step requires a clean GPU driver/device state. The benchmark path now treats a
requested GPU row as strict: if ggml reports no usable CUDA device, the row
fails immediately instead of silently falling back to CPU and then producing an
invalid CUDA-labeled timing.

The current reproducible readiness check is:

```bash
just cuda-health
```

It writes `outputs/cuda-health/nvidia-smi.log`,
`outputs/cuda-health/sam3-bf16-gpu-smoke.log`, and
`outputs/cuda-health/summary.env`. The latest run reports
`cuda_health=blocked`: `nvidia-smi` cannot determine the GPU0 device handle, and
the strict SAM3 BF16 GPU smoke fails before inference with `GPU backend is
required but unavailable`.
