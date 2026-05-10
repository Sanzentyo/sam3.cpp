# Model Matrix Results

Date: 2026-05-10

This is the first full CUDA model/precision matrix for the Rust-wrapper
baseline work. The C++ matrix was run against every `.ggml` file downloaded from
`PABannier/sam3.cpp`.

## Inputs

```bash
uv run scripts/model_matrix_compare.py \
  --download-ggml \
  --skip-python \
  --models-dir models/matrix-all \
  --video "$SAM3_VIDEO" \
  --frames 10 \
  --point-x 315 \
  --point-y 250 \
  --text-prompt person \
  --out-dir outputs/model-matrix-compare-all-cpp
```

Python baselines were run separately for locally available official checkpoints:

- SAM 2.1 Tiny
- SAM 2.1 Base+
- SAM 3

## Coverage

| Matrix | Count |
| --- | ---: |
| GGML files downloaded | 52 |
| C++ CUDA benchmark OK | 42 |
| C++ CUDA benchmark failed | 10 |

The 10 failures are all SAM 2 / SAM 2.1 Base+ variants. They crash in the CUDA
flash-attention path during Hiera image encoding:

```text
ggml-cuda/fattn.cu:560 fatal error
```

| Model | Backend | Status |
| --- | --- | --- |
| `sam2_hiera_base_plus_f32` | CUDA | crashed, signal 6 |
| `sam2_hiera_base_plus_f16` | CUDA | crashed, signal 6 |
| `sam2_hiera_base_plus_q8_0` | CUDA | crashed, signal 6 |
| `sam2_hiera_base_plus_q4_1` | CUDA | crashed, signal 6 |
| `sam2_hiera_base_plus_q4_0` | CUDA | crashed, signal 6 |
| `sam2.1_hiera_base_plus_f32` | CUDA | crashed, signal 6 |
| `sam2.1_hiera_base_plus_f16` | CUDA | crashed, signal 6 |
| `sam2.1_hiera_base_plus_q8_0` | CUDA | crashed, signal 6 |
| `sam2.1_hiera_base_plus_q4_1` | CUDA | crashed, signal 6 |
| `sam2.1_hiera_base_plus_q4_0` | CUDA | crashed, signal 6 |

## Best C++ Row By Family

| Family | Best model | Precision | Track ms | P50 ms | P95 ms | RSS MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| EdgeTAM | `edgetam_q4_0` | `q4_0` | 52.1 | 50.5 | 61.1 | 549.1 |
| SAM 3 | `sam3-q8_0` | `q8_0` | 1717.9 | 1718.3 | 1752.2 | 632.5 |
| SAM 3 visual | `sam3-visual-q4_1` | `q4_1` | 1058.8 | 1058.5 | 1070.4 | 754.1 |
| SAM 2 tiny | `sam2_hiera_tiny_q4_0` | `q4_0` | 486.1 | 483.8 | 496.5 | 611.2 |
| SAM 2 small | `sam2_hiera_small_q4_1` | `q4_1` | 489.1 | 486.6 | 500.7 | 621.3 |
| SAM 2 large | `sam2_hiera_large_q4_0` | `q4_0` | 686.0 | 684.6 | 697.7 | 677.2 |
| SAM 2.1 tiny | `sam2.1_hiera_tiny_q4_0` | `q4_0` | 479.9 | 478.6 | 490.3 | 611.9 |
| SAM 2.1 small | `sam2.1_hiera_small_q4_1` | `q4_1` | 484.2 | 481.3 | 496.2 | 597.9 |
| SAM 2.1 large | `sam2.1_hiera_large_q8_0` | `q8_0` | 687.7 | 688.2 | 699.5 | 677.1 |

## Python Baselines

Python uses official PyTorch CUDA bf16 paths and does not have GGML precision
equivalents. These are family-level references, not precision-equivalent rows.

| Family | Backend | Track ms | CUDA alloc MiB |
| --- | --- | ---: | ---: |
| SAM 2.1 Tiny | PyTorch CUDA bf16 | 42.1 | 674.5 |
| SAM 2.1 Base+ | PyTorch CUDA bf16 | 58.2 | 854.0 |
| SAM 3 | PyTorch CUDA bf16 | 254.1 | 4695.1 |

## C++ vs Python

`python_over_cpp_track_ratio = python_track_ms / cpp_track_ms`.

Values below `1.0` mean Python is faster for the measured path. On this CUDA
machine, the current C++ path is not speed-competitive with official Python for
the comparable baselines, although C++ uses much less memory for SAM 3.

| C++ model | C++ track ms | Python track ms | Python/C++ ratio |
| --- | ---: | ---: | ---: |
| `sam3-f32` | 2017.7 | 254.1 | 0.126 |
| `sam3-f16` | 1811.8 | 254.1 | 0.140 |
| `sam3-q8_0` | 1717.9 | 254.1 | 0.148 |
| `sam3-q4_1` | 1718.3 | 254.1 | 0.148 |
| `sam3-q4_0` | 1718.0 | 254.1 | 0.148 |
| `sam2.1_hiera_tiny_f32` | 485.6 | 42.1 | 0.087 |
| `sam2.1_hiera_tiny_f16` | 485.9 | 42.1 | 0.087 |
| `sam2.1_hiera_tiny_q8_0` | 483.9 | 42.1 | 0.087 |
| `sam2.1_hiera_tiny_q4_1` | 482.8 | 42.1 | 0.087 |
| `sam2.1_hiera_tiny_q4_0` | 479.9 | 42.1 | 0.088 |

## Interpretation

- The Rust wrapper should treat current C++ CUDA performance as a correctness
  and memory baseline, not as a speed baseline.
- EdgeTAM is the only currently fast C++ tracking path on CUDA at roughly
  52 ms/frame.
- SAM 3 full remains dominated by image encoding/tracking graph cost at roughly
  1.7 s/frame even after quantization.
- Quantization reduces model size and load time, but it does not materially
  reduce per-frame CUDA tracking latency for SAM 3 or SAM 2 Hiera in the current
  implementation.
- SAM 2 / SAM 2.1 Base+ CUDA support is incomplete because all Base+ precision
  variants crash in `ggml-cuda/fattn.cu`.

## Artifacts

```text
outputs/model-matrix-compare-all-cpp/summary.json
outputs/model-matrix-compare-all-cpp/cpp-results.json
outputs/model-matrix-compare-all-cpp/cpp-benchmark.log
outputs/model-matrix-compare-local/summary.json
```
