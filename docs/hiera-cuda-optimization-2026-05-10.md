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

## Performance

Model: `sam2.1_hiera_base_plus_q4_0`, 10 frames, CUDA, bbox-only tracking.

| State | Track ms/frame | P50 ms | P95 ms | Note |
| --- | ---: | ---: | ---: | --- |
| Before PE caching | 547.3 | 544.8 | 559.4 | Base+ fallback only |
| After CPU PE caching | 136.1 | 133.3 | 148.9 | removed repeated Hiera/neck PE generation |
| After backend neck PE reuse | 129.8 | 127.3 | 139.7 | avoids repeated neck PE backend upload |

Encode-size sweep after PE caching:

| Encode size | Track ms/frame | P50 ms | P95 ms | RSS MiB | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 129.8 | 127.3 | 139.7 | 647.4 | default size |
| 768 | 72.3 | 71.3 | 80.6 | 574.2 | faster, lower resolution |
| 640 | 48.3 | 47.1 | 53.1 | 552.2 | near PyTorch speed |
| 512 | 34.4 | 33.4 | 39.5 | 572.8 | faster than PyTorch speed |

Base+ precision sweep after CPU PE caching:

| Model | Track ms/frame | P50 ms | P95 ms | RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| `sam2.1_hiera_base_plus_f32` | 149.8 | 149.7 | 159.5 | 678.8 |
| `sam2.1_hiera_base_plus_f16` | 146.9 | 145.5 | 156.2 | 705.1 |
| `sam2.1_hiera_base_plus_q8_0` | 134.6 | 133.6 | 141.8 | 646.6 |
| `sam2.1_hiera_base_plus_q4_1` | 136.0 | 134.4 | 143.7 | 649.6 |
| `sam2.1_hiera_base_plus_q4_0` | 134.5 | 133.6 | 142.1 | 647.8 |

Official PyTorch SAM2.1 Base+ on the same 10-frame clip measured
`43.4 ms/frame` for propagation with `855.2 MiB` CUDA allocation. The current
C++ CUDA path is therefore still about 3x slower than official PyTorch for this
prompt/video at the default 1024 encode size, despite the PE cache improvement.
At `--encode-img-size 512`, C++ is faster than PyTorch on this clip, but that is
a speed/quality tradeoff and not a parity-preserving replacement.

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

Compared C++ `sam2.1_hiera_base_plus_q4_0` against official PyTorch
SAM2.1 Base+ using the same video, same point prompt, and 10 frames.

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

At `--encode-img-size 512`, C++ reaches `34.4 ms/frame`, but quality remains far
from official PyTorch:

| Metric | Value |
| --- | ---: |
| Frames compared | 10 |
| Mean mask IoU | 0.0521 |
| Minimum mask IoU | 0.0000 |
| Minimum bbox IoU | 0.0310 |

## Priority

1. Fix or explain the SAM2 point-prompt quality gap against official PyTorch.
   The current C++ mask is not a faithful substitute for the official model on
   this sample.
2. Reduce Hiera encode graph compute. After caching fixed PE, the remaining
   dominant steady-state cost is the 80-90 ms ggml CUDA graph compute.
3. Decide whether lower encode sizes are acceptable for the Rust wrapper. A
   512 encode size beats PyTorch speed on this sample, but current mask IoU is
   too low to treat it as a quality-preserving optimization.
4. Reduce CPU preprocessing cost. Resize/normalize still costs roughly
   14-18 ms/frame.
5. Treat `head_dim=56` FlashAttention support as a secondary optimization. The
   fallback is no longer the main measured cost after PE caching.

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
```
