# Final Fork Validation

Date: 2026-05-10

This records the validation after recreating both GitHub forks and pushing the
current SAM3/ggml integration branches.

## Repository State

| Repository | Branch | Remote state |
| --- | --- | --- |
| `<owner>/sam3.cpp` | `sam3-cuda/ggml-perf-integration` | public fork of `PABannier/sam3.cpp` |
| `<owner>/ggml` | `sam3-cuda/model-exec-perf` | public fork of `ggml-org/ggml` |

Current `ggml` submodule commit:

```text
0185f0effafcaea7b97a281e906bff97de05ab29 cuda: add win partition kernels
```

The previous private mirror was removed after the branch was pushed to the new
public fork.

## Validation Commands

```bash
export SAM3_MODELS_DIR=models
export SAM3_VIDEO=data/test_video.mp4

just fmt-check
git diff --check
just build-target sam3_smoke
timeout 240s build/xmake-release-cuda/examples/sam3_smoke \
  --model "$SAM3_MODELS_DIR/sam3-q4_0.ggml" \
  --video "$SAM3_VIDEO" \
  --frame 0 \
  --point-x 315 \
  --point-y 250 \
  --text person \
  --output-dir outputs/sam3-smoke-final-fork
PARITY_RUNS=15 PARITY_OUT=outputs/parity-final-fork just parity-stats
```

## Build And Smoke Result

The final fork state builds with CUDA enabled:

- `just fmt-check`: passed
- `git diff --check`: passed
- `just build-target sam3_smoke`: passed
- `ggml commit` reported by configure: `0185f0ef`

Real-model smoke used:

- model: `$SAM3_MODELS_DIR/sam3-q4_0.ggml`
- video: `$SAM3_VIDEO`
- frame: `0`
- prompt: point `(315, 250)` and text `person`

Smoke detections:

| Path | Detections | Notes |
| --- | ---: | --- |
| PVS | 1 | score `0.459547`, box `[271.0,246.0,491.0,460.0]` |
| PCS | 2 | top scores `0.621358`, `0.610469` |

Generated local artifacts:

```text
outputs/sam3-smoke-final-fork/pvs_00.png
outputs/sam3-smoke-final-fork/pcs_00.png
outputs/sam3-smoke-final-fork/pcs_01.png
```

## Parity And Performance

The parity check compares default CUDA graph behavior against the shape-keyed
CUDA graph cache path.

Parity result:

| Metric | Value |
| --- | ---: |
| default rows | 10 |
| shape-key rows | 10 |
| diff rows | 0 |

Track-time statistics, 15 paired runs:

| Variant | Mean ms | Median ms | SD ms | Min ms | Max ms | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default | 51.727 | 51.600 | 0.371 | 51.200 | 52.400 | `[51.521, 51.932]` |
| shape-key | 50.300 | 50.100 | 0.619 | 49.700 | 51.700 | `[49.957, 50.643]` |
| paired saved | 1.427 | 1.500 | 0.689 | -0.300 | 2.400 | `[1.045, 1.808]` |

Mean track-time speedup:

```text
2.8363154406892077%
```

The full local benchmark output is in:

```text
outputs/parity-final-fork/summary.json
outputs/parity-final-fork/default_*.log
outputs/parity-final-fork/shape_*.log
outputs/parity-final-fork/default_fullmask.jsonl
outputs/parity-final-fork/shape_fullmask.jsonl
```

## Acceptance Criteria

- `sam3.cpp` and `ggml` are both GitHub forks under the configured repository owner.
- The `ggml` fork is public and keeps the GitHub fork relationship to
  `ggml-org/ggml`.
- The current `ggml` optimization branch is present on the new fork.
- The `sam3.cpp` submodule points at a commit available from the new `ggml`
  fork.
- Formatting, whitespace checks, and CUDA smoke target build pass.
- Real-model smoke produces PVS and PCS outputs.
- Parity between default and shape-keyed CUDA graph behavior has zero row
  differences.
- Performance reporting includes paired-run statistics, not a single run.
