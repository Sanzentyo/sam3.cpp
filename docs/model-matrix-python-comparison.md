# Model Matrix And Python Comparison

Date: 2026-05-10

This document defines the benchmark used before adding a Rust wrapper. The goal
is to keep a reproducible model/precision matrix for `sam3.cpp` and compare it
against the official Python implementations where an equivalent baseline exists.

## Scope

`sam3.cpp` is benchmarked for every available `.ggml` model file:

- SAM 3: `f32`, `f16`, `q8_0`, `q4_1`, `q4_0`
- SAM 3 visual-only: `f16`, `q8_0`, `q4_1`, `q4_0`
- SAM 2 / SAM 2.1: tiny, small, base+, large across available precisions
- EdgeTAM: `f16`, `q8_0`, `q4_0`

The Python baseline is different by design:

- official Python models do not expose the same GGML quantized precisions;
- Python is therefore measured as a family-level non-quantized baseline;
- C++ precision rows are compared against the corresponding family baseline
  when that official Python model is available locally.

This avoids pretending that `q4_0` or `q8_0` has a direct PyTorch equivalent.

Speed and quality conclusions require matching input conditions. For C++ vs
official Python, compare only rows with the same decoded source-frame
resolution, frame range, prompt, model family, and SAM input encode size. Lower
encode-size or lower source-resolution rows are useful scaling data, but they
are not evidence that one implementation is faster than another unless the
official Python row uses the same effective input size.

The `python_over_cpp_track_ratio` field in `summary.json` is:

```text
python_track_ms / cpp_track_ms
```

Values above `1.0` mean the Python baseline is slower than the C++ row. Values
below `1.0` mean the Python baseline is faster for that benchmark path.

For SAM 3 specifically, the current C++ benchmark initializes the tracker from a
visual point prompt and then tracks/propagates, while the official Python video
baseline uses a text prompt. Treat that comparison as a family-level latency
reference until the C++ benchmark grows a text-prompt tracking mode and the
Python path grows a matching visual-prompt mode.

## Benchmark Command

```bash
export SAM3_MODELS_DIR=models
export SAM3_VIDEO=data/test_video.mp4

uv run scripts/model_matrix_compare.py \
  --models-dir "$SAM3_MODELS_DIR" \
  --video "$SAM3_VIDEO" \
  --frames 10 \
  --point-x 315 \
  --point-y 250 \
  --text-prompt person \
  --out-dir outputs/model-matrix-compare
```

To download the complete GGML model matrix first:

```bash
export SAM3_MODELS_DIR_ALL=models-all
export SAM3_VIDEO=data/test_video.mp4

uv run scripts/model_matrix_compare.py \
  --download-ggml \
  --models-dir "$SAM3_MODELS_DIR_ALL" \
  --video "$SAM3_VIDEO" \
  --frames 10 \
  --point-x 315 \
  --point-y 250 \
  --text-prompt person \
  --out-dir outputs/model-matrix-compare-all
```

The script writes:

```text
cpp-results.json
python-results.json
summary.json
cpp-benchmark.log
python-*.log
```

For Python baselines outside this repository, pass paths explicitly or set:

```bash
export SAM2_REPO=external/sam2
export SAM3_REPO=external/sam3
export SAM2_TINY_CHECKPOINT=checkpoints/sam2.1_hiera_tiny.pt
export SAM2_BASE_PLUS_CHECKPOINT=checkpoints/sam2.1_hiera_base_plus.pt
```

Python baselines use fresh session/state timing after warmup so cached repeated
streaming does not under-report latency.

The all-family matrix should normally keep each model's default input size.
Use `--encode-img-size` only for a targeted same-size comparison where the C++
row and the official Python row are both configured to that size.

For SAM2 encode-size sweeps, run one matrix per encode size and keep the C++
and official Python override identical:

```bash
uv run scripts/model_matrix_compare.py \
  --models-dir "$SAM3_MODELS_DIR_ALL" \
  --video "$SAM3_VIDEO" \
  --filter sam2.1_hiera_base_plus \
  --encode-img-size 512 \
  --frames 10 \
  --point-x 315 \
  --point-y 250 \
  --out-dir outputs/model-matrix-sam2-base-plus-512
```

## Acceptance Criteria

- The C++ benchmark covers every `.ggml` file in the selected models directory.
- The Python benchmark records every available official baseline that can run on
  the local machine.
- C++ vs Python speed/quality claims use the same decoded source-frame
  resolution, frame range, prompt, model family, and SAM input encode size.
- `summary.json` records the decoded source-frame size, requested encode size,
  and a `comparable_speed_claim` flag for each C++/Python comparison row.
- Quantized C++ rows are compared only against a family-level Python baseline,
  not treated as precision-equivalent PyTorch rows.
- The output includes track latency, total latency, and memory columns.
- The summary is machine-readable so the later Rust wrapper can reuse the same
  threshold and regression checks.

## Current Local Caveat

At the time this document was added, the local model directory used for the
first smoke comparison contained only:

```text
edgetam_q4_0.ggml
sam2.1_hiera_tiny_q4_0.ggml
sam3-q4_0.ggml
```

The complete matrix requires downloading the remaining `.ggml` files from
`PABannier/sam3.cpp`.
