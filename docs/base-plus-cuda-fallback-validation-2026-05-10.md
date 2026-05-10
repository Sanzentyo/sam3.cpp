# Base+ CUDA Fallback Validation

Date: 2026-05-10

This records the first validation pass after adding the Hiera attention fallback
for CUDA FlashAttention head dimensions that ggml does not currently support.

## Commands

Use a local test video through `SAM3_VIDEO`; do not commit local absolute paths.

```bash
build/xmake-release-cuda/examples/sam3_benchmark \
  --models-dir models/matrix-all \
  --video "$SAM3_VIDEO" \
  --gpu-only \
  --filter sam2.1_hiera_base_plus_q4_0 \
  --n-frames 3 \
  --output-jsonl outputs/base-plus-parity/cuda.jsonl

build/xmake-release-cuda/examples/sam3_benchmark \
  --models-dir models/matrix-all \
  --video "$SAM3_VIDEO" \
  --cpu-only \
  --filter sam2.1_hiera_base_plus_q4_0 \
  --n-frames 3 \
  --output-jsonl outputs/base-plus-parity/cpu.jsonl

uv run scripts/compare_tracking_jsonl.py \
  outputs/base-plus-parity/cpu.jsonl \
  outputs/base-plus-parity/cuda.jsonl \
  --lhs-label cpu \
  --rhs-label cuda \
  --out outputs/base-plus-parity/cpu-vs-cuda-summary.json
```

For profiler capture, `sam3_benchmark` now has `--no-isolation`. The normal
isolated POSIX mode forks a child process, which prevents `nsys profile` from
capturing CUDA kernel data from the benchmark body.

```bash
nsys profile \
  --force-overwrite=true \
  --trace=cuda,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true \
  --output=outputs/hiera-cuda-profile/nsys/base_plus_q4_0_no_isolation \
  build/xmake-release-cuda/examples/sam3_benchmark \
    --models-dir models/matrix-all \
    --video "$SAM3_VIDEO" \
    --gpu-only \
    --bbox-only \
    --filter sam2.1_hiera_base_plus_q4_0 \
    --n-frames 3 \
    --no-isolation
```

## Parity

Compared `sam2.1_hiera_base_plus_q4_0` CPU vs CUDA on 3 frames.

| Metric | Value |
| --- | ---: |
| Rows compared | 3 |
| Minimum bbox IoU | 0.9362 |
| Maximum bbox delta | 2 px |
| Maximum score absolute delta | 0.0366 |
| Maximum absolute mask-area relative delta | 5.40% |
| Exact mask-hash matches | 0 / 3 |

Interpretation:

- The fallback no longer crashes and preserves the selected object at bbox
  level for this short clip.
- The mask hashes are not expected to match exactly across CPU and CUDA because
  small floating-point differences change thresholded mask pixels. This run
  shows small bbox drift but not bitwise mask parity.
- The next quality gate should compare against official Python on the same
  prompt/video, preferably with mask IoU from actual mask images rather than
  hash equality only.

## CUDA Profile

Short `SAM3_PROFILE=1` runs for the best representative SAM 2.1 Hiera rows:

| Model | Track ms | Encode mean ms | Profiled compute sum ms | Tensor set sum ms |
| --- | ---: | ---: | ---: | ---: |
| `sam2.1_hiera_tiny_q4_0` | 482.6 | 485.7 | 54.1 | 27.9 |
| `sam2.1_hiera_base_plus_q4_0` | 563.5 | 561.0 | 55.0 | 31.1 |
| `sam2.1_hiera_small_q4_1` | 486.3 | 487.7 | 53.2 | 28.8 |
| `sam2.1_hiera_large_q8_0` | 682.6 | 682.3 | 53.9 | 44.5 |

The wall-time bottleneck is the Hiera image encode path. The backend compute
sum reported by `SAM3_PROFILE` is much smaller than the encode wall time, so the
current profiler is not yet decomposing graph construction, allocation,
synchronization, and transfer overhead well enough to identify the exact kernel
or graph operation from `SAM3_PROFILE` alone.

The `nsys` Base+ no-isolation run captured CUDA kernel data. The largest kernel
groups by total GPU time were:

| Kernel group | Total GPU time |
| --- | ---: |
| scalar copy kernels | 62.7 ms |
| binary broadcast add kernels | 29.8 ms |
| CUTLASS GEMM kernels | 27.3 ms |
| `scale_f32` | 22.5 ms |
| transpose convolution | 18.5 ms |
| `soft_max_f32<4096,1024>` | 17.8 ms |

## Next Work

The immediate speed target is not the fallback branch alone. Base+ is slower
than Tiny/Small, but the measured profile suggests the broader Hiera encode
graph has high overhead across SAM 2.x models. Next changes should:

- Add explicit stage labels around encode graph build/compute/copy sections so
  `SAM3_PROFILE` explains the 500+ ms encode wall time.
- Investigate graph reuse or graph-capture opportunities for Hiera encode,
  because per-frame encode dominates tracking.
- Keep the `head_dim=56` fallback as a correctness guard, then evaluate a
  ggml CUDA FlashAttention kernel for `head_dim=56` only after the broader
  encode profile is decomposed.
