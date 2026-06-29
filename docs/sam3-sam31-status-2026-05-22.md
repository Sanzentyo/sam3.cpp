# SAM3 and SAM3.1 Status

Date: 2026-05-22

This note records the current SAM3/SAM3.1 coverage and the benchmark contract
used before treating this repository as a Rust-wrapper baseline.

## Coverage

- SAM2.1 tracking is the mature path. It has same-input C++ versus official
  Python comparisons, strict FP32 controls, repeat statistics, and mask-quality
  checks.
- SAM3 full models load and run in C++ for the existing `sam3-*.ggml` files.
  The current selected-target BF16 required-E2E contract is faster than official
  Python at the session/model wall level, but the comparable tail image
  encode/backbone diagnostic is still slower. The remaining root-performance
  gap is concentrated in the ViT projection/attention/MLP dataflow.
  All-model/all-precision validation and official-output quality parity remain
  separate acceptance items.
- SAM3 visual-only models load separately through the visual tracker path. They
  are useful for point-prompt tracker coverage but are not a substitute for the
  full text-prompt SAM3 detector/tracker path.
- SAM3.1 official Python runs through the current upstream `sam3` repository
  with `build_sam3_predictor(version="sam3.1")`.
- SAM3.1 C++ is partial. The repository now has SAM3.1 GGML v4 loading,
  multiplex tensor registration, memory-backbone, propagation-feature,
  memory-attention, memory-attention-to-decoder slice parity, and a full-model
  mask-init smoke. End-to-end C++ versus official Python tracking parity is not
  met yet.

## Benchmark Contract

Speed and quality claims are valid only when the compared rows use:

- the same decoded source-frame resolution,
- the same frame range,
- the same measured tracking interval,
- the same benchmark warmup policy,
- the same prompt,
- the same model input resolution,
- the same precision and TF32 policy when comparing strict numerical behavior.

Rows that intentionally vary source resolution, encode size, precision, or
prompt are scaling or coverage rows. They must not be used as direct C++ versus
official Python win/loss evidence.

## SAM3 Required Work Classes 2026-06-29

The current SAM3 BF16 required-E2E audit now marks each process row with
`measurement_class`, so the optimizer can distinguish the formal denominator
from additive required work, required substeps, placement-only diagnostics, and
implementation budgets. This prevents cached-tail placement or profile-derived
budgets from being mistaken for skipped work or direct E2E speedups.

Fresh regenerated evidence:

- split:
  `outputs/e2e-required-split-sam3-bf16-required-work-classes-20260629c/required_process_split.md`
- audit:
  `outputs/e2e-required-audit-required-work-classes-20260629c/optimization_targets.md`

The same source run was regenerated with the Python timed-tail hook contract
made explicit:

- split:
  `outputs/e2e-required-split-sam3-bf16-python-tail-contract-20260629d/required_process_split.md`
- audit:
  `outputs/e2e-required-audit-python-tail-contract-20260629d/optimization_targets.md`

That newer audit marks `Python cached-tail equivalent` as `True` because the
official Python run exposes timed-tail diagnostic hooks for the same frame
range: backbone/detection `167.175 ms`, tracker propagation `42.661 ms`, hook
sum `209.836 ms`. These hooks are used only for gap attribution; they are not a
separate Python denominator and do not change the C++ required-E2E denominator.

The required denominator and additive process split are unchanged:

| Row | Class | C++ ms | Required E2E share |
| --- | --- | ---: | ---: |
| `required_e2e.total` | `formal_required_denominator` | `592.421` | `100.0%` |
| `tail.image_encode` | `formal_required_rollup` | `284.144` | `48.0%` |
| `frame0.image_encode` | `formal_required_rollup` | `159.658` | `27.0%` |
| `frame0.prompt_and_tracker_init` | `formal_required_rollup` | `106.802` | `18.0%` |
| `tail.propagate` | `formal_required_rollup` | `25.356` | `4.3%` |

The required detail rows confirm that the bottleneck is still graph compute, not
decode, output artifacts, graph admin, or cached-tail placement:

| Row | Class | C++ ms | Required E2E share |
| --- | --- | ---: | ---: |
| `tail.image_encode.graph_compute` | `formal_required_detail` | `282.341` | `47.7%` |
| `frame0.image_encode.graph_compute` | `formal_required_detail` | `158.714` | `26.8%` |
| `tail.propagate.graph_compute` | `formal_required_detail` | `20.595` | `3.5%` |
| `input.prepare.decode` | `formal_required_detail` | `7.998` | `1.4%` |
| `tail.image_encode.non_compute` | `formal_required_detail` | `1.803` | `0.3%` |

`tail.image_encode.inline.*` and `tail.image_encode.preencode.*` remain
`placement_diagnostic` rows. They explain where cached-tail work ran, but they
are not accepted as speedups unless `tail.image_encode`,
`tail.image_encode.graph_compute`, and `required_e2e.total` fall under the same
contract.

The implementation budget rows are now explicitly marked
`implementation_budget`. They are not additive E2E rows; they are the profiled
CUDA node budget projected onto the required denominator. The first root
optimization targets remain:

| Budget row | Class | Projected ms | Required E2E share |
| --- | --- | ---: | ---: |
| `required.image_encode.graph_compute/image.vit.mlp_matmul` | `implementation_budget` | `141.117` | `23.8%` |
| `required.image_encode.graph_compute/image.neck` | `implementation_budget` | `65.668` | `11.1%` |
| `required.image_encode.graph_compute/image.vit.qkv_matmul` | `implementation_budget` | `60.256` | `10.2%` |
| `required.image_encode.graph_compute/image.vit.attention` | `implementation_budget` | `58.557` | `9.9%` |

Optimization implication: the next candidate should be promoted only if it
reduces a formal required row, preferably `tail.image_encode.graph_compute` and
`frame0.image_encode.graph_compute`, and keeps exact C++ parity plus the
official-Python quality gate. A local kernel or graph change that only improves
an `implementation_budget` row is a pre-gate until it moves the required-E2E
denominator.

## SAM3 Required E2E Stage/FATTN Split 2026-06-30

The latest SAM3 BF16 required-E2E audit keeps the formal E2E denominator,
required process split, node-profile budget, isolated ViT stage timings, and
FATTN CUDA-event attribution in one report:
`outputs/e2e-required-audit-stage-budget-fattn-current-20260630b/optimization_targets.md`.

Current same-contract smoke values:

| Row | C++ ms | Official Python ms | Role |
| --- | ---: | ---: | --- |
| `required_session_e2e` | `592.421` | `635.364` | formal acceptance denominator |
| `model_e2e` | `584.315` | `584.124` | subtotal; not enough by itself |
| `tail_image_encode.graph_compute` | `282.341` | `167.175` | primary remaining root target |
| `frame0.image_encode.graph_compute` | `158.714` | - | also required E2E work |

The normalized tail graph budget now ranks the implementation work as:

| Tail group | Required budget ms | Required E2E share |
| --- | ---: | ---: |
| `image.vit.mlp_matmul` | `119.776` | `20.2%` |
| `image.vit.qkv_matmul` | `38.958` | `6.6%` |
| `image.vit.attention` | `38.057` | `6.4%` |
| `image.neck` | `21.910` | `3.7%` |
| `image.vit.layout_rope_copy` | `19.564` | `3.3%` |

The combined required image-encode graph budget, covering frame 0 plus tail,
puts the same ViT MLP matmuls at `141.117 ms` (`23.8%` of required E2E), QKV
at `60.256 ms` (`10.2%`), and attention at `58.557 ms` (`9.9%`). This is the
right budget for changes that affect every required image encode.

The isolated stage split is now tied back to those budgets. The largest pure
local stages are `mlp_fc2` (`1.148-1.168 ms`), `mlp_fc1` (`0.928-0.937 ms`),
and `qkv_proj` (`0.630-0.632 ms`). The attention-side derived local cost is
`0.325 ms` for the representative window block and `1.617 ms` for the global
block. This keeps block-local experiments from being accepted unless they map
onto the required E2E budget.

`scripts/summarize_fattn_profile.py` now also emits Markdown, and
`scripts/summarize_e2e_optimization_targets.py` accepts
`--fattn-profile NAME=PATH`. `just e2e-required-audit` forwards those through
`E2E_REQUIRED_AUDIT_FATTN_PROFILES`. The latest FATTN split shows:

| Profile | FATTN group | total steady ms | kernel steady ms | QKV convert steady ms | Tail budget |
| --- | --- | ---: | ---: | ---: | ---: |
| block0 `attn_core` | `sam3-vit-window-head64` | `0.442` | `0.350` | `0.090` | `23.536` |
| block7 `attn_core` | `sam3-vit-global-head64` | `1.744` | `1.651` | `0.091` | `14.522` |

This says window attention still has a visible conversion/boundary component,
while global attention is dominated by the FATTN kernel body. It does not move
the top target: exact-preserving MLP/QKV matmul dataflow remains the highest
payoff path before another attention-only experiment.

Two local candidate checks were rejected under this separated stage budget:

| Candidate | Local result | Projected required-E2E result | Decision |
| --- | --- | --- | --- |
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` | block0 `mlp` `-0.012 ms`, block7 `mlp` `+0.045 ms` | window-only signal is not robust; global regresses by about `+2.984 ms` projected | reject |
| `SAM3_ENABLE_VIT_CONTIGUOUS_ATTENTION_V=1` | block0 `attn_core` `+1.400 ms`, block7 `+0.121 ms` | projected tail regression `+80.391 ms` / `+2.348 ms` | reject |
| `GGML_CUDA_ENABLE_FATTN_F32_TO_F16_NC_DIM0_VEC4=1` | block0 `attn_core` `+0.028 ms`, block7 `+0.015 ms` | projected tail regression `+1.603 ms` / `+0.296 ms` | reject |
| `SAM3_ENABLE_VIT_CONTIGUOUS_ATTENTION_V=1 GGML_CUDA_ENABLE_FATTN_F32_TO_F16_PAIR_VEC2=1` | block0 `attn_core` `+0.126 ms`, block7 `+0.106 ms` | projected tail regression `+7.255 ms` / `+2.060 ms` | reject |
| local-only FATTN64 `ncols1=128` instance probe | block0 `attn_core` `+0.071 ms`, block7 `+0.371 ms` | projected tail regression `+4.067 ms` / `+7.202 ms` | reject |

The flat-MLP path now has diagnostic block selectors:
`SAM3_ENABLE_VIT_MLP_FLAT_CHAIN_BLOCKS` and
`SAM3_DISABLE_VIT_MLP_FLAT_CHAIN_BLOCKS`, using the same syntax as the existing
ViT block selectors (`1-7,12` or `all`). This is not enabled by default. A
window-only check with `0-6,8-14,16-22,24-30` measured the full ViT+neck
pre-gate at baseline `151.656 ms` versus window-flat `151.445 ms`, but the
signal was only `z=-0.108` and the median moved by only `-0.132 ms`, so it is
not promoted to full required-E2E A/B. Evidence:
`outputs/e2e-required-vit-bench-baseline-r30-20260630b/`,
`outputs/e2e-required-vit-bench-mlp-flat-window-20260630b/`,
`outputs/e2e-required-vit-stage-ab-mlp-flat-window-b0-20260630b/`, and
`outputs/e2e-required-vit-stage-ab-mlp-flat-window-b7-20260630b/`.

The `NC_DIM0_VEC4` FATTN conversion probe was rerun serially after discarding
contended parallel timings. Evidence:
`outputs/e2e-required-vit-stage-ab-nc-dim0-vec4-b0-20260629d/stage_ab.md` and
`outputs/e2e-required-vit-stage-ab-nc-dim0-vec4-b7-20260629d/stage_ab.md`.

The V-contiguous plus K/V pair-conversion probe recovers much of the earlier
V-contiguous regression on the window representative, but still remains slower
than the default non-contiguous V path on both representatives. Evidence:
`outputs/e2e-required-vit-stage-ab-contig-v-pair-vec2-b0-20260629e/stage_ab.md`
and
`outputs/e2e-required-vit-stage-ab-contig-v-pair-vec2-b7-20260629e/stage_ab.md`.

The head64 `ncols1=128` MMA instance probe required a temporary local instance
and selector path, then was removed after measurement because it slowed both
representative attention stages. Evidence:
`outputs/e2e-required-vit-stage-ab-fattn64-ncols128-b0-20260629f/stage_ab.md`
and
`outputs/e2e-required-vit-stage-ab-fattn64-ncols128-b7-20260629f/stage_ab.md`.

Conclusion: the current measurement stack can now separate required E2E work,
tail placement, isolated stage cost, and FATTN sub-cost. The next optimization
should target real graph compute, especially the BF16 MLP FC1/FC2 and QKV
matmul dataflow. A candidate is not accepted unless it lowers
`required_e2e_ms` and `tail_image_encode.graph_compute` under the same contract,
then passes exact C++ parity and the official-Python quality gate.

## SAM3 cuDNN MLP Candidate 2026-06-29

The first MLP dataflow candidate that passes the required-E2E A/B gate combines
the flat ViT MLP chain with BF16 MLP tensors and a cuDNN FC1+GELU path:

```text
SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1
SAM3_BF16_VIT_MLP_CHAIN=1
GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16=1
GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16_UNSAFE_RUN=1
```

Evidence:

- required E2E A/B:
  `outputs/e2e-required-ab-sam3-bf16-cudnn-mlp-flat-bf16-r3-20260629a/optimization_targets.md`
- C++ candidate parity:
  `outputs/e2e-required-ab-sam3-bf16-cudnn-mlp-flat-bf16-r3-20260629a/parity/compare.json`
- block0 stage A/B:
  `outputs/e2e-required-vit-stage-ab-cudnn-mlp-flat-bf16-b0-20260629a/stage_ab.md`
- block7 stage A/B:
  `outputs/e2e-required-vit-stage-ab-cudnn-mlp-flat-bf16-b7-20260629a/stage_ab.md`

The required-E2E A/B promotes the candidate:

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `required_e2e_ms` | `938.987` | `924.919` | `-14.068` |
| `model_e2e_ms` | `926.231` | `912.171` | `-14.060` |
| `required_core_compute_ms` | `786.218` | `774.750` | `-11.468` |
| `tail_encode_graph_compute_ms` | `583.159` | `577.255` | `-5.904` |
| `frame0_encode_graph_compute_ms` | `161.027` | `155.794` | `-5.233` |

The statistical signals are also in the right direction: required E2E `z=-6.723`
and tail graph `z=-2.953`. Candidate parity is exact against the C++ baseline
for the compared JSONL rows (`diff_rows=0`), so this optimization does not
change the selected-target outputs covered by that contract.

The isolated MLP stage checks agree with the E2E direction:

| Representative block | Baseline steady ms | Candidate steady ms | Delta | Projected tail E2E delta |
| --- | ---: | ---: | ---: | ---: |
| block0 window | `2.110` | `1.954` | `-0.156` | `-9.926` |
| block7 global | `2.109` | `1.984` | `-0.125` | `-7.966` |

This should be treated as an accepted candidate, not the final state. The
Python tail image-encode diagnostic is still faster than C++ for the same
contract (`524.812 ms` Python diagnostic versus `577.255 ms` candidate C++
tail graph), so the next root targets remain FC2/QKV dataflow and reducing the
remaining image-encode graph gap.

A separate no-flat BF16+cuDNN probe now reaches the cuDNN FC1+GELU path through
reshape/view look-through, but its full MLP stage is still slower than the
current baseline (`2.169 ms` versus about `2.109 ms`). That path is useful
implementation groundwork, but it is not an accepted speed candidate without
the flat-chain dataflow.

## SAM3 Required E2E Deep Split 2026-06-29

The SAM3 BF16 required-E2E bundle now separates the optimization denominator
from attribution-only profiling:

- `required_e2e_ms` remains the formal wall denominator.
- `tail.image_encode.graph_compute` is the primary root-performance target.
- synchronized CUDA node/profile data is normalized back onto the normal
  `tail_encode_graph_compute_ms` before it is used as an E2E budget.
- isolated ViT stage measurements are captured from the actual E2E frame input
  and attached to the audit, so block-local experiments are not inferred only
  from cumulative stop points.

The current same-contract SAM3 BF16 run is:

| Item | C++ ms | Official Python ms | Ratio / gap |
| --- | ---: | ---: | ---: |
| `required_session_e2e` | `935.069` | `1072.842` | C++ `1.147x` faster |
| `model_e2e` | `922.026` | `1020.333` | C++ `1.107x` faster |
| `tail_image_encode_or_backbone` | `585.925` | `525.789` | C++ slower by `60.135 ms` |
| `tail_image_encode.graph_compute` | `581.868` | `525.789` | C++ slower by `56.079 ms` |

Evidence:
`outputs/e2e-required-measure-sam3-bf16-current-20260629goal-a/audit/optimization_targets.md`.
Official Python quality parity passes the tolerance gate on the same five-frame
contract: bbox IoU minimum `0.979`, mask IoU minimum `0.985`, max mask xor
`483 px`. Exact mask hashes are not expected for this official-Python
comparison.

The audit now emits `Required E2E Work Measurement Gates`, which separates
formal required work from implementation-budget attribution. The formal rows
show that `tail.image_encode` is `585.925 ms` (`62.7%` of required E2E),
`frame0.image_encode` is `160.333 ms` (`17.1%`), and `tail.propagate` is
`49.942 ms` (`5.3%`). The detail rows show `tail.image_encode.graph_compute`
as `581.868 ms` (`62.2%`), so kernel/dataflow candidates must first move this
number and then pass the full `required_e2e_ms` and official-Python parity
gates. Placement-only cached-tail rows are diagnostic and are not acceptance
evidence unless the formal denominator improves.

Because frame 0 image encoding is also required E2E work, the audit now also
reports `Required Image Encode Graph Budget Groups` for `frame0 + tail` graph
compute. This is the broader E2E budget for image-encoder changes:

| Required image graph group | Projected normal ms | Required E2E share |
| --- | ---: | ---: |
| `image.vit.mlp_matmul` | `268.737` | `28.7%` |
| `image.vit.qkv_matmul` | `102.183` | `10.9%` |
| `image.vit.attention` | `98.393` | `10.5%` |
| `image.neck` | `88.970` | `9.5%` |
| `image.vit.layout_rope_copy` | `44.535` | `4.8%` |
| `image.vit.prefix` | `40.155` | `4.3%` |
| `image.vit.mlp_gelu` | `34.017` | `3.6%` |
| `image.vit.proj_matmul` | `32.557` | `3.5%` |
| `image.vit.norm_residual_copy` | `31.434` | `3.4%` |

The tail-only table below remains the comparison target for the current Python
tail-backbone diagnostic gap, while this combined table is the right budget for
estimating total E2E payoff when the same kernel/dataflow change also affects
frame 0.

The normalized tail required graph budget is now the preferred way to decide
where a CUDA optimization must land:

| Tail graph group | Projected normal ms | Required E2E share | Interpretation |
| --- | ---: | ---: | --- |
| `image.vit.mlp_matmul` | `247.196` | `26.4%` | primary root target; both FC1 and FC2 must improve under the normal graph-compute gate |
| `image.vit.qkv_matmul` | `80.495` | `8.6%` | secondary GEMM target |
| `image.vit.attention` | `78.157` | `8.4%` | attention kernel/dataflow target |
| `image.neck` | `45.336` | `4.8%` | conv/deconv target, but current direct conv paths are already required |
| `image.vit.layout_rope_copy` | `40.067` | `4.3%` | existing RoPE/window fusions already help; further work must move the required graph slice |
| `image.vit.mlp_gelu` | `30.175` | `3.2%` | activation/copy fusion is already present; standalone toggles have not won |
| `image.vit.proj_matmul` | `29.937` | `3.2%` | attention projection GEMM target |
| `image.vit.norm_residual_copy` | `28.411` | `3.0%` | layernorm/residual/copy target; existing norm/copy toggles have not won |

| Tail graph component | Projected normal ms | Required E2E share |
| --- | ---: | ---: |
| `sam3-vit:mlp-fc2-matmul` | `124.200` | `13.3%` |
| `sam3-vit:mlp-fc1-matmul` | `122.996` | `13.2%` |
| `sam3-vit:qkv-matmul` | `80.495` | `8.6%` |
| `sam3-vit:window-attn` | `48.485` | `5.2%` |
| `sam3-vit:mlp-gelu` | `30.175` | `3.2%` |
| `sam3-vit:proj-matmul` | `29.937` | `3.2%` |
| `sam3-vit:global-attn` | `29.672` | `3.2%` |
| `sam3-vit:qk-layout-rope` | `19.409` | `2.1%` |
| `sam3-vit:block-out-residual-norm-copy` | `16.166` | `1.7%` |
| `sam3-neck:conv3x3` | `14.817` | `1.6%` |
| `sam3-neck:deconv` | `14.391` | `1.5%` |

The attached isolated ViT stage sweeps now cover both the representative window
block 0 and the first global-attention block 7 from real E2E block inputs. This
keeps the local pre-gate aligned with the required image-encode denominator
instead of relying on one block shape:

| Isolated stage | Block 0 mean | Block 7 mean | Local note |
| --- | ---: | ---: | --- |
| `block` | `3.668 ms` | `5.166 ms` | global block is slower overall |
| `mlp` | `2.102 ms` | `2.044 ms` | MLP cost is similar across window/global |
| `mlp_fc2` | `1.168 ms` | `1.164 ms` | largest pure MLP op |
| `mlp_fc1` | `0.937 ms` | `0.937 ms` | second largest pure MLP op |
| `qkv_proj` | `0.632 ms` | `0.632 ms` | QKV GEMM is shape-stable |
| `attn_core - qkv_rope` | `0.375 ms` | `1.620 ms` | global attention is the attention-side root target |

The required tail image-encode path was also checked for batch-scaling as a
possible root optimization. This computes the same image encoder work rather
than skipping tail frames, but it is not a win on the current CUDA path:

| Batch size | Mean ms | Mean ms/frame | Lane max diff | Decision |
| ---: | ---: | ---: | ---: | --- |
| `1` | `149.634` | `149.634` | - | reference |
| `2` | `310.313` | `155.156` | `0` | reject |
| `4` | `613.548` | `153.387` | `0` | reject |

Evidence: `outputs/e2e-required-vit-bench-batch-scaling-20260629a/`.
`sam3_vit_batch_bench` now also reports `median_ms_per_frame`,
`min_ms_per_frame`, and `max_ms_per_frame`, so batch and isolated measurements
can be compared without post-processing each sample list.

`just e2e-required-measure` now includes isolated stage sweeps by default for
`E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCKS`, defaulting to `0 7`. It writes
`vit-isolated-stage-sweep-block-<block>/` directories and passes them into
`e2e-required-audit` as `tail-stage-block<block>`. Set
`E2E_REQUIRED_MEASURE_ISOLATED_STAGE_SWEEP=` for a faster accounting-only run,
or override `E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCKS` when a candidate only
targets a different required block family.

The audit's required-work gate table also includes block-labelled isolated
hints, for example `b7:attn_core:1.620` versus `b0:attn_core:0.375`, so the
window/global split is visible in the same report as the formal E2E
denominator. Evidence:
`outputs/e2e-required-audit-sam3-bf16-required-work-multiblock-20260630o/` and
the smoke route check in
`outputs/e2e-required-measure-sam3-bf16-multiblock-smoke-20260630p/`.

The process split now makes the E2E acceptance boundary explicit in both JSON
and Markdown. `required_processes` rows carry `acceptance_target`,
`additive_required_work`, and `placement_only`; the optimization audit emits
`acceptance_required_process_targets` separately from
`placement_process_targets`. The Markdown uses `Acceptance Required Process
Split` for the rows that can justify a speed candidate, and `Required Placement
Details` only to explain where cached-tail encode work ran. This keeps
`tail.image_encode.inline.*` and `tail.image_encode.preencode.*` out of
candidate acceptance deltas unless the formal `required_e2e_ms` and required
tail graph metrics also move.

No new speed candidate is accepted by this measurement change alone. The next
root optimization must reduce the normal `tail_encode_graph_compute_ms` budget,
not only synchronized profile time. The largest viable targets are
exact-preserving BF16 MLP/QKV matmul dataflow, attention kernels, and then neck
conv/deconv work. A candidate must improve both `required_e2e_ms` and
`tail_encode_graph_compute_ms` under the same contract before exact C++ parity
and official-Python quality parity are considered.

Isolated stage candidates now have a small A/B summarizer:
`scripts/summarize_vit_stage_variant_ab.py`. It reports raw mean, steady mean
after dropping the configured initial timed sample, median, CI, and a
Welch-style signal against the chosen baseline. This is intentionally a
pre-gate only: a candidate must first show a local stage win with matching
input/output shape, then move the required image-encode graph slice, and only
then be promoted to a full required-E2E run.

The first use of this A/B route rechecked the SAM3 BF16 block-7 global-attention
`FATTN64` `ncols1` selector from a real E2E captured stage input. Forced
`ncols1=16/32/8` are slower. Forced `ncols1=64` sometimes looks faster than a
slow default run, but reverse-order repeats show the same small-stage timing is
state/order sensitive. It is not promoted as a real algorithmic win without an
E2E graph-compute movement:

| Variant | Raw mean | Steady mean | Steady median | Delta vs stable default | Signal | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `default_b` | `1.979 ms` | `1.970 ms` | `1.970 ms` | `0.000 ms` | `0.00` | baseline |
| `n64_b` | `1.968 ms` | `1.968 ms` | `1.970 ms` | `-0.002 ms` | `-1.05` | reject/no signal |
| `n32` | `2.338 ms` | - | - | slower | - | reject |
| `n16` | `2.708 ms` | - | - | slower | - | reject |
| `n8` | `7.653 ms` | - | - | slower | - | reject |

Evidence:
`outputs/e2e-required-vit-isolated-fattn64-ncols1-b7-recheck-20260630q/variant_ab.md`
,
`outputs/e2e-required-vit-isolated-fattn64-ncols1-b7-reverse-20260630q/variant_ab.md`,
and the initial selector sweep in
`outputs/e2e-required-vit-isolated-fattn64-ncols1-b7-20260630q/variant_ab.md`.

The first same-contract root-candidate pre-gate under this measurement rejected
the existing graph-level MLP toggles again:

| Candidate | Mean ms/frame | Median ms/frame | Delta vs baseline | Signal | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | `150.565` | `148.201` | `0.000` | `0.000` | reference |
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` | `150.877` | `148.121` | `+0.312` | `0.108` | reject/no signal |
| `SAM3_BF16_VIT_MLP_CHAIN=1` | `168.496` | `167.640` | `+17.931` | `6.171` | reject/slower |

Evidence:
`outputs/e2e-required-vit-bench-sam3-bf16-root-candidates-20260629a/` and the
`Isolated Required-Process Benches` section in
`outputs/e2e-required-measure-sam3-bf16-current-20260629goal-a/audit/optimization_targets.md`.

The same E2E image-encode isolated contract also rejected the currently
available neck and generic CUDA toggles. These are pre-gates only; none should
be promoted to full required-E2E A/B without a new implementation change:

| Candidate | Mean ms/frame | Delta vs local baseline | Signal | Decision |
| --- | ---: | ---: | ---: | --- |
| `neck/baseline` | `150.880` | `0.000` | `0.000` | reference |
| `SAM3_ENABLE_NECK_1X1_MULMAT=1` | `151.583` | `+0.703` | `0.253` | reject/no signal |
| `SAM3_DISABLE_DIRECT_NECK_3X3_CONV=1` | `158.366` | `+7.485` | `2.392` | reject/slower |
| `existing-cuda/baseline` | `151.333` | `0.000` | `0.000` | reference |
| `GGML_CUDA_CUBLASLT_BIAS_AUTOTUNE=1` | `152.511` | `+1.178` | `0.438` | reject |
| `GGML_CUDA_ENABLE_MIXED_ADD_NORM_FUSION=1` | `152.896` | `+1.562` | `0.573` | reject |
| `GGML_CUDA_ENABLE_NORM_AFFINE_CPY_FUSION=1` | `153.024` | `+1.690` | `0.632` | reject |
| `GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION=1 + GGML_CUDA_ENABLE_WIN_UNPART_ADD_FUSION=1` | `154.011` | `+2.678` | `0.966` | reject |
| `GGML_CUDA_DISABLE_CUDNN_CONV2D=1` | `461.829` | `+310.496` | `112.605` | reject/slower |

Evidence:
`outputs/e2e-required-vit-bench-neck-candidates-20260629a/`,
`outputs/e2e-required-vit-bench-existing-cuda-candidates-20260629a/`, and the
`Isolated Required-Process Benches` section in
`outputs/e2e-required-measure-sam3-bf16-current-20260629goal-a/audit/optimization_targets.md`.

The standalone BF16 low-level shape bench explains why the next candidate
should not be another broad output-chain toggle. Current cuBLASLt single-GEMM
times are already close to the isolated block costs: FC1 `0.883 ms`, FC2
`0.888 ms`, QKV `0.577 ms`; the best existing full BF16 MLP-chain microbench
row was `2.037 ms`, only a small local win before graph/dataflow overhead and
parity risk. Evidence:
`outputs/e2e-required-lowlevel-mlp-shape-20260629a/`.

Two cuBLASLt BF16 pre-gates were checked against the same isolated tail image
encode input. Neither is worth a full required-E2E A/B:

| Candidate | Mean ms/frame | Delta vs baseline | Signal | Decision |
| --- | ---: | ---: | ---: | --- |
| baseline | `151.904` | `0.000` | `0.000` | reference |
| `GGML_CUDA_CUBLASLT_BIAS_AUTOTUNE=1` | `152.812` | `+0.909` | `0.211` | reject |
| `GGML_CUDA_CUBLASLT_BIAS_FAST_16BF_FOR_BF16=1` | `151.968` | `+0.064` | `0.015` | reject/no signal |

Evidence:
`outputs/e2e-required-vit-bench-cublaslt-autotune-20260629a/audit/optimization_targets.md`
and
`outputs/e2e-required-vit-bench-cublaslt-fast16bf-20260629a/audit/optimization_targets.md`.

## SAM3.1 Propagation Update 2026-06-11

### 2026-06-29 Required E2E Split and Propagation Dense-PE Fix

The SAM3.1 mask-init E2E contract is now measured as an explicitly separated
required path rather than as one opaque frame step. For the same-input
`320x240`, `center@3`, BF16/TF32-on contract, the measured C++ pieces are:

| Process | C++ mean ms | Required E2E share | Official Python mean ms |
| --- | ---: | ---: | ---: |
| Required E2E | `354.525` | `100.0%` | `443.439` |
| Session/setup | `13.810` | `3.9%` | `17.507` |
| Input/prompt construction | `0.121` | `0.0%` | `1.681` |
| Required model execute | `340.603` | `96.1%` | `424.209` |
| Image encode / official backbone-cache | `302.337` | `85.3%` | `397.570` |
| Image encode graph compute | `294.730` | `83.1%` | n/a |
| Mask init | `23.099` | `6.5%` | `9.816` |
| Encoded propagation | `15.167` | `4.3%` | `16.822` |

Evidence:
`outputs/sam31-prop-dense-pe-fix-matrix-r5-20260629x/summary.json` and
`outputs/sam31-prop-dense-pe-fix-audit-r5-20260629x/optimization_targets.md`.
This keeps the C++ required E2E about `1.251x` faster than official Python on
this local contract, but also shows that the remaining optimization target is
not the encoded propagation tail. Image encoding is about `85%` of required
E2E, and image-encoder graph compute alone is about `83%`.

The required-E2E harness also preserves the sub-split inside each required
encode, and the audit Markdown now emits this table for every matrix summary.
On the current `center@3` coverage run, graph compute is the only large encode
component; graph build/allocation/upload are small, and frame-0 PE construction
is visible but secondary:

| Process | Total | Graph compute | Graph build | Graph alloc | Input upload | PE/state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frame0.image_encode` | `159.770 ms` | `153.269 ms` | `0.557 ms` | `0.635 ms` | `0.142 ms` | PE `5.151 ms` |
| `frame1.image_encode` | `143.416 ms` | `142.309 ms` | `0.239 ms` | `0.646 ms` | `0.099 ms` | state `0.107 ms` |
| `tail.propagate_encoded` | `14.968 ms` | `13.868 ms` | `0.338 ms` | `0.219 ms` | `0.031 ms` | output `0.087 ms` |

Evidence:
`outputs/sam31-conv-transpose-bf16-parity-coverage-r3-20260629x/summary.json`.
This is the current measurement boundary for E2E optimization: an accepted
candidate should reduce `required_image_encode_graph_compute_ms` and not merely
move time into graph build, allocation, PE construction, or setup.

The same gate was used to screen the CUDA window partition/unpartition fusion
switches. All rows preserved the default C++ mask, but none reduced the
required E2E denominator. Disabling the default window fusions regressed E2E by
`+5.431 ms`, which confirms that the default unpartition/add fusion is useful.
Forcing the opt-in partition/copy fusion regressed E2E by `+2.660 ms`, and
forcing both window fusions regressed by `+6.801 ms`.

| Variant | Required E2E delta | Image graph delta | Tail graph delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `win-unpart-add-fusion` | `+1.157 ms`, signal `0.84` | `+0.383 ms` | `+0.062 ms` | reject/no win |
| `win-part-cpy-fusion` | `+2.660 ms`, signal `3.42` | `+2.578 ms` | `+1.270 ms` | reject |
| `no-win-window-fusions` | `+5.431 ms`, signal `4.09` | `+3.617 ms` | `+1.594 ms` | reject, confirms default fusion helps |
| `win-window-fusions` | `+6.801 ms`, signal `2.44` | `+2.825 ms` | `+1.430 ms` | reject |

Evidence:
`outputs/sam31-e2e-window-fusion-audit-r3-20260629y/optimization_targets.md`.

The SAM3.1 mask-init matrix runner now supports an explicit synchronized
profile side channel with `--profile-default`. This records a normal
required-E2E matrix for speed decisions, then runs one default C++ profile pass
with `SAM3_PROFILE=1` and `GGML_CUDA_PROFILE_NODES=1` only for stage
attribution. The summary Markdown keeps those values in a separate
`Default CUDA Stage Attribution` section so synchronized per-node timing is not
mistaken for normal wall-clock speed.

The first r5 use of that split rechecked RoPE fusion controls. The default
mask was unchanged for both disable rows, so the deltas are valid speed
evidence for the existing fusion:

| Variant | Required E2E delta | Image graph delta | Tail graph delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `no-cont-rope-pair-fusion` | `+5.681 ms`, signal `7.34` | `+5.776 ms` | `+2.915 ms` | reject; default cont+RoPE fusion is useful |
| `no-rope-pair-fusion` | `+65.434 ms`, signal `47.43` | `+64.611 ms` | `+33.260 ms` | reject; default RoPE pair fusion is required |

The attached profile attribution ranks the warm steady-state targets as
`frame0` neck `23.712 ms` after cold-drop, `tracking` neck `12.759 ms`,
`sam31_propagate_single` `14.808 ms`, and the global ViT blocks at about
`5.15-5.28 ms` each. Evidence:
`outputs/sam31-e2e-separated-profile-rope-ab-r5-20260629a/optimization_targets.md`.

The neck-specific r5 screen still has no promotable toggle. `conv-transpose-bf16`
was a noise-sized `-0.163 ms` E2E movement but changed the default mask,
`neck-1x1-mulmat` was `+0.263 ms` and changed the mask, and
`no-direct-neck-3x3` regressed E2E by `+17.182 ms` while also changing the
mask. Evidence:
`outputs/sam31-e2e-neck-variant-r5-20260629a/optimization_targets.md`.

This run also fixes a SAM3.1 implementation mismatch in the propagation mask
decoder. Official Python passes `tracker.model.image_pe_layer` dense PE into
the multiplex mask decoder. The C++ path had been reusing the memory-attention
vision positional encoding from `cached_src_pos_tensor`. C++ now binds
`trk.model.image_pe_layer.positional_encoding_gaussian_matrix`, builds a
model-level backend dense-PE cache, and passes that tensor as
`sam31_mux_image_pe` while keeping `sam31_prop_image_pe` for the memory-attention
PE. The graph-cache key includes this tensor so cached propagation graphs cannot
silently reuse the wrong decoder PE.

The focused dump check on the same case shows the decoder-PE issue is closed to
the current BF16-GGML versus `.pt` precision level:

| Tensor | Max abs | Mean abs | RMSE |
| --- | ---: | ---: | ---: |
| `sam31_mux_image_pe` | `0.174735` | `0.005945` | `0.010214` |
| `sam31_mux_final_k` | `1.409976` | `0.059059` | `0.077549` |
| `sam31_mux_final_attn_k_proj` | `0.842113` | `0.052927` | `0.071826` |
| `sam31_mux_masks` | `0.686936` | `0.047976` | `0.065462` |

Evidence:
`outputs/sam31-prop-dense-pe-fix-20260629x/mux-summary.json`. The older
`final_k` mismatch was mean-large because it added a different PE, not because
of a simple layout permutation. After this fix, the remaining error is in the
same scale as the upstream BF16 model/official `.pt` precision boundary and the
image-feature path.

The current mask comparison on this contract is still not exact:
Python foreground pixels are `14592`, C++ foreground pixels are `14313`, and
mask IoU is `0.975870`. The selected mask index remains `0`; C++ decoder IoU
scores are `[0.880722, 0.508892, -1.344622]`.

Optimization variants were rechecked under the same required-E2E split. No
candidate met both the same-default-mask gate and a statistically meaningful
E2E win:

| Candidate group | Result | Evidence |
| --- | --- | --- |
| Propagation graph/cache and direct-copy pruning | no required-E2E win; quality-preserving rows were slower or noise | `outputs/sam31-e2e-split-variants-audit-r5-20260629x/optimization_targets.md` |
| ViT output precision and late-global FATTN tile variants | no accepted win; some rows changed the mask and were slower | same |
| cuDNN MLP, ViT qkv/mlp chain, direct qkv views, cuBLASLt algo probes | r3 screen found no accepted win; several chain/cuDNN rows changed output and slowed E2E | `outputs/sam31-e2e-kernel-screen-audit-r3-20260629x/optimization_targets.md` |
| RoPE pair fusion disable controls | default cont+RoPE saves `5.681 ms`; default full RoPE pair fusion saves `65.434 ms`; both preserve the default mask | `outputs/sam31-e2e-separated-profile-rope-ab-r5-20260629a/optimization_targets.md` |
| Neck 1x1 mulmat, BF16 conv-transpose, direct-neck-3x3 disable | no accepted win; every non-default row changed the default mask, and the speed signal was noise or slower | `outputs/sam31-e2e-neck-variant-r5-20260629a/optimization_targets.md` |
| Window partition/copy and unpartition/add fusion toggles | no accepted win; default already benefits from the unpartition/add fusion, while opt-in partition/copy fusion regresses E2E | `outputs/sam31-e2e-window-fusion-audit-r3-20260629y/optimization_targets.md` |

The CUDA-node profile summarizer now splits the normal E2E encode calls into
`frame0` and `tracking` buckets instead of lumping both under `sam31_encode`.
Node profiling synchronizes per node, so the values are for hotspot ranking,
not normal runtime totals. After dropping the cold max call, the warm hotspots
are:

| Mode | Stage | Mean profiled ms after cold drop | Main ops |
| --- | --- | ---: | --- |
| `frame0` | `neck` | `23.711` | `CONV_2D`, `CONV_TRANSPOSE_2D`, `CONT` |
| `frame0` | global ViT blocks `07/15/23/31` | about `5.18-5.31` each | `MUL_MAT`, `FLASH_ATTN_EXT` |
| `tracking` | `neck` | `12.819` | `CONV_TRANSPOSE_2D`, `CONV_2D`, `CONT` |
| `tracking` | global ViT blocks `07/15/23/31` | about `5.19-5.25` each | `MUL_MAT`, `FLASH_ATTN_EXT` |
| `sam31_propagate_single` | full propagation graph | `14.753` | `FLASH_ATTN_EXT`, `MUL_MAT`, `CONT` |

Evidence:
`outputs/sam31-e2e-split-profile-20260629x/stage_profile_split.json`.

The BF16 conv-transpose path was then checked as a parity experiment across
four `320x240` masks, not just the center prompt. It is not a default
optimization candidate: all rows differ from the C++ default mask, no row has
a meaningful E2E speed signal, and Python-IoU movement is mixed.

| Case | Default required E2E | BF16 conv-transpose delta | Default Python IoU / XOR | BF16 conv-transpose Python IoU / XOR | Default-mask delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bottom-band@3` | `358.234 ms` | `-0.464 ms`, signal `-1.05` | `0.999210` / `10` | `0.999210` / `10` | IoU `0.999842`, XOR `2` |
| `center@3` | `355.689 ms` | `+0.458 ms`, signal `0.74` | `0.975870` / `353` | `0.976208` / `348` | IoU `0.999511`, XOR `7` |
| `left-wide@3` | `357.895 ms` | `-0.719 ms`, signal `-1.29` | `0.970333` / `482` | `0.970331` / `482` | IoU `0.999747`, XOR `4` |
| `small-center@3` | `361.535 ms` | `-3.757 ms`, signal `-0.92` | `0.970887` / `129` | `0.970661` / `130` | IoU `0.999772`, XOR `1` |

Evidence:
`outputs/sam31-conv-transpose-bf16-parity-coverage-audit-r3-20260629x/optimization_targets.md`.

Next optimization acceptance criteria:

- Use the same required-E2E contract unless the row is explicitly marked as a
  scaling row: `320x240`, `center@3`, BF16 GGML, official Python BF16 autocast,
  TF32 on, same prompt, same frame offset, no tensor-dump mode.
- Report the split fields, not just total time: required E2E, image encode,
  image encode graph compute, mask init, encoded propagation, setup, and
  remainder.
- A default candidate must preserve the C++ default mask exactly
  (`default_iou >= 0.999999`, XOR `0`) or be labeled as a precision/parity
  experiment rather than an optimization default.
- A speed candidate needs at least r5 interleaved statistics and a meaningful
  negative delta signal (`<= -2.0`) on `run_required_e2e_ms`; improving only the
  encoded propagation tail is insufficient unless the total E2E win is visible.
- Any change touching PE, ViT dataflow, memory attention, or the multiplex
  decoder must also publish tensor evidence for the affected boundary, at
  minimum `sam31_mux_image_pe`, `sam31_mux_final_k`, final attention K/V, masks,
  and scores.

### 2026-06-12 Current Precision Check

The current SAM3.1 default was rechecked on the small same-input mask-init
contract (`320x240`, `center@7`, three repeats, one Python and C++ warmup). This
supersedes the older small-case IoU rows below for the current branch state.

| Precision | Python slice | C++ encoded propagation | C++ full frame step | Mask vs Python | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 / TF32 on | `16.5615 ms` propagation | `15.9965 ms` mean, `16.0382 ms` median | `369.891 ms` mean | IoU `0.980441`, XOR `282 px` | `outputs/sam31-mask-init-matrix-bf16-b12-31-default-20260612c/summary.json` |
| F16 / TF32 on | `17.0433 ms` propagation | `17.7513 ms` mean, `17.7983 ms` median | `377.190 ms` mean | IoU `0.980905`, XOR `276 px` | `outputs/sam31-mask-init-matrix-current-default-f16-20260612b/summary.json` |
| F32 / TF32 on | `16.6904 ms` propagation | `16.1927 ms` mean, `16.3141 ms` median | `1462.576 ms` mean | IoU `0.978824`, XOR `306 px` | `outputs/sam31-mask-init-matrix-current-default-f32-20260612b/summary.json` |

This shows the current split clearly. BF16 and F32 beat the official Python
propagation slice at the same precision setting, while F16 still trails on that
narrow slice. BF16/F16 full frame steps are faster than the Python smoke on this
small synthetic case because C++ image encoding is faster there, but the mask is
not bit/semantic parity yet. F32 is not a viable full-frame default because the
C++ F32 image encode dominates (`two_frame_encode_ms` about `1406.8 ms`).

The next acceptance item is therefore not another propagation-only cache. It is
to close the remaining mask drift while preserving the BF16/F32 propagation win
and to move F16 encoded propagation below the Python row. The highest-priority
bottleneck for full-frame F32 remains ViT image encoding; for BF16/F16, the
remaining blocker is accuracy/parity rather than raw propagation speed.

The BF16 default now keeps late SAM3.1 ViT linear outputs in F32 from block 12
onward instead of forcing every block output to BF16. This is a parity-first
dataflow adjustment, not a cache or kernel change. The old all-BF16-output path
is still available with `SAM3_BF16_VIT_LINEAR_OUTPUT=1` or the
`vit-linear-output` matrix variant. On `center@7`, the new default improves the
mask from IoU `0.980247` / XOR `285 px` to IoU `0.980441` / XOR `282 px` while
remaining faster than the Python propagation slice. In the direct new-default
versus old-output rerun, `vit-linear-output` reproduced the older mask hash and
the new default produced the improved mask hash
(`outputs/sam31-mask-init-matrix-bf16-b12-31-default-20260612c/summary.json`).

The same block-12 cutoff was checked across four `320x240` synthetic masks
before making it the default. It improved Python XOR in all four cases with no
clear encoded-propagation regression:

| Case | Old all-BF16-output IoU / XOR | New default IoU / XOR | New encoded propagation |
| --- | ---: | ---: | ---: |
| `center@7` | `0.980247` / `285` | `0.980441` / `282` | `16.134 ms` |
| `small-center@7` | `0.870598` / `632` | `0.871616` / `626` | `16.238 ms` |
| `left-wide@7` | `0.968188` / `513` | `0.968316` / `511` | `16.029 ms` |
| `bottom-band@7` | `0.994731` / `66` | `0.995048` / `62` | `16.209 ms` |

Evidence is in
`outputs/sam31-mask-init-matrix-bf16-b12-31-coverage-20260612c/summary.json`.

The old layer-0 self-attention FATTN heuristic remains faster but is not the
current same-accuracy default. With
`GGML_CUDA_DISABLE_SAM31_MEM_ATTN_L0_SA_TILE=1`, BF16 encoded propagation drops
from `16.10 ms` to `12.83 ms` mean and F16 drops from `17.97 ms` to `14.05 ms`
mean on the same small case. However, the C++ mask changes from the default
(`116` BF16 xor pixels and `110` F16 xor pixels), and Python IoU is slightly
lower in both checks. Evidence is in
`outputs/sam31-mask-init-matrix-current-bf16-l0sa-variant-20260612b/summary.json`
and
`outputs/sam31-mask-init-matrix-current-f16-l0sa-variant-20260612b/summary.json`.
This keeps the faster heuristic as an opt-in diagnostic path; making it the
default would trade away the parity-first contract rather than solving it.

The current BF16 tensor dump still localizes the remaining drift to the ViT
image-feature path. A fresh Python/C++ state dump on the same `320x240`,
`center@7` case shows exact preprocessing, exact interpolated multiplex mask,
close patch embedding/position add, and close block-0 through block-8 outputs.
Max outliers start crossing the broad `0.5` threshold at block 9, then amplify
through the block 12-15 MLP/residual path. Representative values:

| Boundary | Max abs | Mean abs | Evidence |
| --- | ---: | ---: | --- |
| `vit_block_08_out` | `0.2717` | `0.00791` | `outputs/sam31-mask-init-state-compare-current-bf16-20260612b/vit-stage-12-15-summary.json` |
| `vit_block_09_out` | `0.5415` | `0.00781` | same |
| `vit_block_12_mlp_fc2` | `1.1563` | `0.00560` | same |
| `vit_block_14_out` | `12.3398` | `0.01073` | same |
| `vit_block_15_out` | `25.1367` | `0.01298` | same |
| `vit_output` | `97.9742` | `0.02301` | `outputs/sam31-mask-init-state-compare-current-bf16-20260612b/summary.json` |
| `stored_spatial_feats` | `0.8347` | `0.00744` | same |
| `image_features` | `0.5618` | `0.00928` | same |

This keeps the next root fix focused on the ViT BF16/F32 projection and MLP
dataflow, especially how rare channel/spatial outliers accumulate through later
blocks. Preprocessing, multiplex mask interpolation, positional encodings,
memory K/V cache reuse, and final propagation-only staging are not the current
primary parity blockers.

The SAM3.1 mask-init propagation path now uses a validated static spatial
memory cross-attention K/V cache by default. The cache stores the spatial K/V
projections without temporal-position encoding and adds the per-frame
temporal-position vector at propagation time. This fixes the earlier cache
contract bug where the cached K path used a different `maskmem_tpos_enc` row
than the normal path. Set `SAM31_DISABLE_MEM_CA_KV_CACHE=1` to force the old
per-propagation K/V projection path.

Small same-input evidence, `320x240`, `small-center@7`, BF16 GGML model,
official Python BF16 autocast with TF32 on:

| Path | Propagation Contract | Result | Mask vs Python | Evidence |
| --- | --- | ---: | ---: | --- |
| Official Python SAM3.1 | cached frame propagation | 16.526 ms | reference | `outputs/sam31-mask-init-matrix-bf16-autocast-small-20260611q/summary.json` |
| C++ default, K/V cache on | encoded-frame propagation | 16.365 ms mean | IoU 0.837168, XOR 821 px | `outputs/sam31-mask-init-matrix-bf16-autocast-small-20260611q/summary.json` |
| C++ `no-kv-cache` | encoded-frame propagation | 19.068 ms mean | IoU 0.837334, XOR 820 px | `outputs/sam31-mask-init-matrix-bf16-autocast-small-20260611q/summary.json` |
| C++ current default, layer0 SA TILE | encoded-frame propagation | 16.260 ms mean, 16.279 ms median | IoU 0.870598, XOR 632 px | `outputs/sam31-mask-init-matrix-l0sa-variant-20260612/summary.json` |
| C++ `no-l0sa-tile`, old FATTN heuristic | encoded-frame propagation | 13.224 ms mean, 13.128 ms median | IoU 0.848996, XOR 752 px | `outputs/sam31-mask-init-matrix-l0sa-variant-20260612/summary.json` |

The fixed static cache is about `14.2%` faster than the old C++ no-cache mean
on this propagation slice and about `1.0%` faster than the official Python
BF16 propagation timing measured by the same local runner. The cache does not yet
make the whole frame step faster than Python because C++ still includes the
separate frame encode path in `propagate_ms`; the accepted comparison point here
is the already-encoded propagation slice.

The current CUDA FATTN heuristic now selects the TILE path for
`sam31_mem_attn_layer0_sa_fattn` by default. This is a parity-first tradeoff:
three same-input repeats improved mask IoU from `0.848996` to `0.870598` while
keeping encoded propagation at `16.260 ms` mean / `16.279 ms` median, still
slightly below the local official Python BF16 propagation row of `16.689 ms`.
The `scripts/run_sam31_mask_init_matrix.py` runner now has a `no-l0sa-tile`
variant to measure the old heuristic under the same harness. Set
`GGML_CUDA_DISABLE_SAM31_MEM_ATTN_L0_SA_TILE=1` to recover the old faster but
less Python-like heuristic.

Raw tensor checks show that the fixed cache is numerically equivalent to the
normal C++ path at the cached K/V boundary: layer-0 cross-attention K has
`max_abs=9.54e-7`, V is exact, final memory output has `mean_abs=0.00155`, and
mask logits have `mean_abs=0.00153`
(`outputs/sam31-prop-dump-cpp-default-vs-static-kv-encidx-small-20260611p/summary.json`).
Against official Python, the remaining end-to-end mask-init mismatch is still
present (`IoU ~= 0.837` on this small synthetic case), so this is a propagation
speed win, not a full SAM3.1 parity completion.

The Python SAM3.1 runner now applies the requested precision with CUDA autocast.
Earlier BF16/F16 labels from this harness were not reliable because the dtype
argument was parsed but not applied around the official Python calls.

F16 propagation also has an accepted ggml CUDA fix for pathological tiny
cuBLASLt bias matmuls in the SAM3.1 mask head. A new F16-weight/F32-activation
tiny bias kernel covers non-GELU `dst.ne[0] <= 4`, `dst.ne[1] <= 32` shapes and
can be disabled with `GGML_CUDA_DISABLE_TINY_F16_F32_BIAS_KERNEL=1`.

| Path | Propagation Contract | Result | Mask vs Python | Evidence |
| --- | --- | ---: | ---: | --- |
| Official Python SAM3.1 | fp16 autocast, TF32 on | 16.669 ms | reference | `outputs/sam31-mask-init-matrix-f16-tiny-small-20260611r/summary.json` |
| C++ default after F16 tiny kernel | F16 GGML, encoded-frame propagation | 14.551 ms mean, 14.469 ms median | IoU 0.827634, XOR 867 px | `outputs/sam31-mask-init-matrix-f16-tiny-small-20260611r/summary.json` |
| C++ current tiny kernel disabled | F16 GGML, one direct smoke | 41.162 ms | same C++ mask hash as enabled | `outputs/sam31-f16-tiny-compare-disabled-small-20260611r/disabled/summary.json` |

The node profile shows the root cause and the fix: before the F16 tiny kernel,
the final object-score `dst=f32[1,16,1,1]` fused matmul took `21.185 ms` and
the IoU-head `dst=f32[3,16,1,1]` took `2.560 ms`
(`outputs/sam31-cuda-nodes-f16-small-20260611r/profile-summary.json`). After
the fix these are `0.0066 ms` and `0.159 ms`
(`outputs/sam31-cuda-nodes-f16-tiny-small-20260611r/profile-summary.json`).
The enabled/disabled A/B smoke produced the same C++ mask hash and only
sub-`1e-6` score/logit differences, but official Python parity remains open.
The enabled side is recorded in
`outputs/sam31-f16-tiny-compare-disabled-small-20260611r/current/summary.json`.

The current SAM3.1 parity triage uses propagation tensor dumps from both the
official Python runner and the C++ full-model smoke. The Python mask-init state
dump now also applies the requested CUDA autocast precision; before this fix,
state dumps labeled BF16/FP16 still executed the official calls outside autocast.

Small `320x240`, `small-center@7`, BF16+TF32 findings:

| Boundary | Result | Evidence |
| --- | ---: | --- |
| Mask-init `obj_ptr` | `max_abs=0.0883`, `mean_abs=0.00157` | `outputs/sam31-mask-init-state-compare-bf16-autocast-20260611s/obj-summary.json` |
| Mask-init `object_score_logits` | exact | `outputs/sam31-mask-init-state-compare-bf16-autocast-20260611s/obj-summary.json` |
| Propagation memory-attention input | `max_abs=0.4687`, `mean_abs=0.00847` | `outputs/sam31-prop-dump-compare-bf16-20260611s/mem-attn-summary.json` |
| Propagation memory-attention output | `max_abs=1.543`, `mean_abs=0.0304` | `outputs/sam31-prop-dump-compare-bf16-20260611s/mem-attn-summary.json` |
| Propagation decoder `upscaled` | `max_abs=0.205`, `mean_abs=0.00696` | `outputs/sam31-prop-dump-compare-bf16-20260611s/hyper-summary.json` |
| Propagation decoder final pre-norm queries | `max_abs=0.381`, `mean_abs=0.0354` | `outputs/sam31-prop-dump-compare-bf16-20260611s/prenorm-summary.json` |
| Propagation decoder final post-norm queries | `max_abs=0.795`, `mean_abs=0.0650` | `outputs/sam31-prop-dump-compare-bf16-20260611s/prenorm-summary.json` |
| Propagation decoder `hyper_0` | `max_abs=0.710`, `mean_abs=0.0838` | `outputs/sam31-prop-dump-compare-bf16-20260611s/hyper-summary.json` |
| Propagation mask logits | `max_abs=2.423`, `mean_abs=0.0987` | `outputs/sam31-prop-dump-compare-bf16-20260611s/hyper-summary.json` |

The selected mask index is the same on both sides for this case: slot 0,
mask 0. The low-resolution selected mask already has IoU about `0.838` before
PNG resizing, so the remaining mismatch is inside the propagation mask decoder,
not final postprocessing. `upscaled` features are close; the larger error is in
`mask_tokens`/hypernetwork outputs after the two-way decoder.

Additional propagation dumps narrowed this further:

| Boundary | Result | Evidence |
| --- | ---: | --- |
| Manual final LayerNorm on Python pre-norm | reproduces Python post-norm within about `2e-6` max | `outputs/sam31-prop-dump-compare-bf16-20260611s/final-ln-manual-summary.json` |
| Manual final LayerNorm on C++ pre-norm | reproduces C++ post-norm within about `2e-6` max | `outputs/sam31-prop-dump-compare-bf16-20260611s/final-ln-manual-summary.json` |
| F32 propagation pre-norm path | mismatch remains, `final_queries mean_abs=0.0660` | `outputs/sam31-prop-dump-compare-bf16-20260611s/f32-prenorm-summary.json` |
| No static memory K/V cache | does not fix the mismatch | `outputs/sam31-prop-dump-compare-bf16-20260611s/f32-nokv-summary.json` |
| Final attention Q projection | close, `mean_abs=0.0122` | `outputs/sam31-prop-dump-compare-bf16-20260611s/finalattn-inner-summary.json` |
| Final attention K projection | divergent, `mean_abs=0.306` | `outputs/sam31-prop-dump-compare-bf16-20260611s/finalattn-inner-summary.json` |
| Final attention V projection | divergent, `mean_abs=0.0797` | `outputs/sam31-prop-dump-compare-bf16-20260611s/finalattn-inner-summary.json` |
| Block0 image-to-token token-side K/V | close, `mean_abs=0.0124/0.0110` | `outputs/sam31-prop-dump-compare-img2tok-bf16-20260611u/img2tok-summary.json` |
| Block0 image-to-token image-side Q | divergent, `mean_abs=0.276` | `outputs/sam31-prop-dump-compare-img2tok-bf16-20260611u/img2tok-summary.json` |
| Memory-attention layer0 FFN norm | input difference amplified but still mean-small, `mean_abs=0.0338` | `outputs/sam31-prop-dump-compare-ffn-snap-bf16-20260611u/ffn-summary.json` |
| Memory-attention layer0 FFN fc2 | close, `mean_abs=0.0090` | `outputs/sam31-prop-dump-compare-ffn-snap-bf16-20260611u/ffn-summary.json` |
| Memory-attention layer0 SA/CA projections | mean-small, but max outliers remain | `outputs/sam31-prop-dump-compare-memproj-bf16-20260611u/layer0-summary.json` |

The final LayerNorm is therefore not the implementation bug; it faithfully
amplifies the query differences it receives. The final decoder attention query
side is also close. The currently strongest mismatch path is the image-feature
side: memory-attention output differences are projected into the two-way
decoder image-to-token Q, and later into final K/V. Forcing ggml CUDA FATTN
globally to TILE/VEC improves this small-case mask IoU, but the narrower
accepted change is the layer0 self-attention TILE heuristic above. Remaining
mask drift still follows the image-feature side after memory-attention, not
object selection, mask resize, final LayerNorm, or the SAM mask head MLP.

The BF16 memory-attention activation diagnostic remains rejected as a default.
With dump disabled, five paired C++ smokes on the same case measured default
encoded propagation at `15.60 ms mean`, `15.02 ms median`; enabling
`SAM31_BF16_MEM_ATTN_ACTIVATION=1` measured `15.87 ms mean`, `15.71 ms median`
and changed the C++ mask hash
(`outputs/sam31-bf16-memact-speed-20260611s`). It is useful for diagnosis but
not a speed or parity improvement.

BF16/F16 speed now beats the official Python propagation slice on this small
contract, but SAM3.1 is still not accepted as same-accuracy complete. The
current accepted statement is: C++ SAM3.1 encoded propagation is faster on the
checked small synthetic propagation slice, while end-to-end official Python
mask parity remains open.

## SAM3.1 ViT State Triage 2026-06-11

The current BF16 same-contract rerun confirms the speed/parity split. On
`320x240`, `small-center@7`, BF16 GGML versus official Python BF16 autocast
with TF32 on, official Python propagation measured `16.675897 ms`; C++ encoded
propagation measured `14.590394 ms` mean, but the mask comparison is still
`IoU=0.8489959839`, `xor=752 px`
(`outputs/sam31-mask-init-matrix-current-bf16-default-20260611d/summary.json`).

The mask-init state dump path now snapshots requested ViT debug tensors before
dumping them. This avoids graph allocator reuse producing misleading ranges or
NaNs in intermediate dumps; the normal inference graph is unchanged unless
`SAM31_MASK_INIT_STATE_DUMP_DIR` is set. With the snapshot dump, preprocessing,
mask interpolation, stored position encoding, patch embedding, and ViT
positional addition all remain close or exact. The largest remaining ViT error
is localized and channel-skewed rather than a broad tensor mismatch:

| Boundary | Max abs | Mean abs | RMSE | Evidence |
| --- | ---: | ---: | ---: | --- |
| block12 `mlp_fc2` | `1.15625` | `0.00559977` | `0.00816763` | `outputs/sam31-mask-init-state-compare-bf16-block12-14-20260611d/vit-boundary-12-14-summary.json` |
| block12 output | `1.964745` | `0.00818019` | `0.01238343` | same |
| block13 `mlp_fc2` | `2.492188` | `0.00634849` | `0.00979281` | same |
| block13 output | `4.302525` | `0.00943358` | `0.01536084` | same |
| block14 `mlp_fc2` | `8.21875` | `0.00724143` | `0.01278367` | same |
| block14 output | `12.339802` | `0.01072884` | `0.01984636` | same |

The worst coordinates concentrate on channel `679`, especially around spatial
coordinate `(64,61)`, and then propagate into block15/23/31 outputs. A later
isolated-stage rerun narrowed this further: when the Python tensor is used as
the direct stage input, the window and global ViT stages are close to the Python
dump. The remaining mask drift is therefore an accumulated BF16/F32 dataflow
effect across the full 32-block ViT path, not a single broken preprocessing,
window partition, MLP GELU, final mask resize, object selection, memory K/V
cache, or SAM3.1 memory-attention terminal norm.

Three immediate A/B candidates were rejected:

| Candidate | Result | Evidence |
| --- | --- | --- |
| `SAM3_DISABLE_BF16_VIT_FC2_INPUT=1` | same mask and IoU, slower encode | `outputs/sam31-mask-init-matrix-fc2-input-ab-20260611d/summary.json` |
| `SAM3_BF16_VIT_MLP_CHAIN=1` | worse, `IoU=0.8383316783`, `xor=814 px` | `outputs/sam31-mask-init-matrix-mlp-chain-current-20260611d/summary.json` |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2=0..4` | same mask and IoU for all checked indices | `outputs/sam31-mask-init-matrix-fc2-algo{0..4}-20260611d/summary.json` |

The isolated stage bench itself had one reliability bug: it uploaded the input
tensor once and then reused the same graph for warmup/repeat iterations. Some
ggml execution paths can reuse or overwrite allocation storage, so repeated
stage runs could dump the result of a mutated input. The bench now restores the
input tensor before every warmup and timed compute, without including the
restore in the measured interval. With that fix, clean isolated runs show:

| Stage group | Representative result | Evidence |
| --- | --- | --- |
| window blocks 12-14 MLP | `mlp` max abs `0.0625-0.125`, mean abs about `0.00094`; per-block MLP mean `2.12-2.25 ms` | `outputs/sam31-vit-mlp-stage-isolated-reset-input-20260611a/compare.json` |
| window blocks 12-14 attention/layout | window partition/unpartition exact; QKV mean abs about `0.00117-0.00122`; attention core mean abs about `0.00052-0.00058` | `outputs/sam31-vit-block-stage-isolated-reset-input-20260611a/compare.json` |
| global blocks 15/23/31 | global attention core mean abs `0.00045-0.00122`; MLP FC1 is the largest isolated drift but still small (`0.00196-0.00389` mean abs) | `outputs/sam31-vit-global-stage-isolated-reset-input-20260611a/compare.json` |

A refreshed precision A/B on the same `320x240 small-center@7` BF16/TF32
contract keeps the default as the best accepted row. `no-vit-linear-output`,
`vit-qkv-chain`, `vit-mlp-chain`, and `vit-attn-proj-output-all` were all slower
or worse against the Python mask. `GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1`
reduced frame encode by about `1.5 ms`, but changed the mask and worsened the
Python comparison (`IoU=0.8419271352`, `xor=794 px`), so it is not accepted for
the same-parity goal
(`outputs/sam31-mask-init-matrix-precision-ab-reset-input-20260611a/summary.json`,
`outputs/sam31-mask-init-matrix-direct-bf16-dst-ab-20260611a/summary.json`).

The next root target is therefore the full ViT projection/dataflow path rather
than an isolated block-stage fix. The current node profile still puts SAM3.1
tracking encode hotspots on BF16/F32 projection matmuls: `MUL_MAT
dst=bf16[1024,5184]`, `MUL_MAT dst=f32[4736,5184]`, `MUL_MAT
dst=f32[3072,5184]`, window head64 attention, and global head64 attention
(`outputs/sam31-profile-nodes-current-reset-input-20260611a/encode-hotspots.json`).
The encoded propagation slice remains faster than official Python on this small
row; the full-frame gap is now primarily frame encoding.

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
| C++ `sam3-f16`, vec4 window partition accepted row | same 3 frames, CUDA | 205.43 ms/frame | `outputs/model-matrix-sam3-f16-winpart-vec4-r3/summary.json` |
| Official Python SAM3, same accepted F16 row | same 3 frames, fp16/TF32 | 206.98 ms/frame | `outputs/model-matrix-sam3-f16-winpart-vec4-r3/summary.json` |
| C++ `sam3-f16`, current row-group copy default | same 3 frames, fp16-weight GGML, CUDA, 2 warmups | 193.0 ms/frame mean, 185.3 ms median | `outputs/sam3-final-f16-rowgroup-matrix-20260611c/summary.json` |
| Official Python SAM3, current same-contract row | same 3 frames, fp16 autocast, TF32 on, 2 warmups | 120.86 ms/frame mean, 120.92 ms median | `outputs/sam3-final-f16-rowgroup-matrix-20260611c/summary.json` |
| C++ `sam3-f16`, cuDNN neck 3x3 conv opt-in | same 3 frames, `SAM3_ENABLE_DIRECT_NECK_3X3_CONV=1`, C++ only A/B, 2 warmups | 184.68 ms/frame mean, 180.9 ms median | `outputs/sam3-f16-default-vs-cudnn-neck-cpponly-20260611d/cudnn-neck/summary.json` |
| C++ `sam3-f16`, default row-group copy A/B control | same 3 frames, C++ only A/B, 2 warmups | 195.74 ms/frame mean, 186.0 ms median | `outputs/sam3-f16-default-vs-cudnn-neck-cpponly-20260611d/default/summary.json` |
| C++ `sam3-f16`, cuDNN neck 3x3 conv default-on | same 3 frames, C++ only A/B, 2 warmups | 190.18 ms/frame mean, 181.4 ms median | `outputs/sam3-f16-direct-neck-default-ab-20260611e/default/summary.json` |
| C++ `sam3-f16`, direct neck disabled control | same 3 frames, `SAM3_DISABLE_DIRECT_NECK_3X3_CONV=1`, C++ only A/B, 2 warmups | 190.86 ms/frame mean, 185.7 ms median | `outputs/sam3-f16-direct-neck-default-ab-20260611e/disabled/summary.json` |
| C++ `sam3-bf16`, current BF16 cuDNN conv2d rerun | same 3 frames, BF16 GGML, CUDA, 2 warmups | 175.2 ms/frame | `outputs/sam3-bf16-current-rerun-20260611g/summary-cudnn-bf16.json` |
| Official Python SAM3, current BF16+TF32 rerun | same 3 frames, bf16 autocast, TF32 on, 2 warmups | 120.62 ms/frame mean | `outputs/sam3-bf16-python-current-20260611g/summary.json` |

This remains an incremental improvement, not a current Python win. The latest
same-contract Python row has official Python at `120.86 ms/frame` mean. The
checked default C++ rows are still in the `181-190 ms/frame` range depending on
median versus mean, so Python remains materially faster. The cuDNN neck 3x3
direct-conv path is now the CUDA default; set
`SAM3_DISABLE_DIRECT_NECK_3X3_CONV=1` to force the old `ggml_conv_2d_s1_ph`
neck. The opt-in probe moved the C++ only A/B mean from `195.74` to
`184.68 ms/frame` and median from `186.0` to `180.9 ms/frame`, while preserving
mask hashes against the default C++ row (`mask_hash_equal_rows=3`,
`min_bbox_iou=0.9998142510967348`, `max_score_abs_delta=4.9e-5`;
`outputs/sam3-f16-cudnn-neck-parity-20260611d/compare.json`). After default-on,
the 5-run A/B reported default median `181.4 ms/frame` versus disabled median
`185.7 ms/frame`, with the same parity envelope
(`outputs/sam3-f16-direct-neck-default-ab-20260611e/default/summary.json`,
`outputs/sam3-f16-direct-neck-default-ab-20260611e/disabled/summary.json`,
`outputs/sam3-f16-direct-neck-default-parity-20260611e/compare.json`). Its cuDNN
algorithm probe chose `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`
(`algo=1`) as the fastest supported path; `algo=0` and `algo=7` were slower,
and `algo=3/6` fell back to the old naive direct CUDA conv
(`outputs/sam3-f16-cudnn-neck-algo-probe-20260611d/summary.json`). Relative to the
earlier Blackwell f32-output GEMM baseline, default C++ moved from `325.67` to
`320.9 ms/frame`, and later work brought the checked F16 path down into the
`185-193 ms/frame` range depending on whether median or mean is used. The BF16
GGML path loads and runs. The current CUDA conv2d path now accepts BF16 neck
3x3 kernels through cuDNN by converting the F32 activation input to BF16,
running BF16 cuDNN convolution, and converting BF16 output back to F32 when a
direct F32-output cuDNN plan is unavailable. This fixes the BF16 abort introduced
by the direct-neck default and restores the current BF16 row to
`175.2 ms/frame`. It is deterministic across repeated default JSONL checks
(`mask_hash_equal_rows=3` in
`outputs/sam3-bf16-cudnn-conv-determinism-20260611g/compare.json`) but it is not
bit-identical to the scalar CUDA BF16 conv fallback (`min_bbox_iou=0.9988396561`,
`max_bbox_delta_px=0.059`, `max_score_abs_delta=0.00088` in
`outputs/sam3-bf16-cudnn-conv-parity-20260611g/compare.json`). Therefore the
remaining same-accuracy comparison should be made against official Python BF16,
not by treating the slow scalar fallback as a golden performance path. Official
Python SAM3 BF16+TF32 currently measures `120.62 ms/frame`, so Python is still
about `1.45x` faster than the current C++ BF16 row. Combining
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

A 2026-06-11 root-profile refresh on the current BF16 row keeps the same
conclusion with newer numbers. The normal C++ row is `175.2 ms/frame`
(`outputs/sam3-bf16-current-rerun-20260611g/summary-cudnn-bf16.json`), while the
official Python BF16+TF32 row is `120.62 ms/frame`
(`outputs/sam3-bf16-python-current-20260611g/summary.json`). The steady C++
tracking frame is approximately `158 ms` for `sam3_encode` plus `12 ms` for
`propagate_single`
(`outputs/sam3-bf16-fattn-profile-current-20260611h/benchmark.log`). In that
profile, head64 FlashAttention is no longer the dominant standalone gap:
window calls are mostly `0.43-0.47 ms` each and global calls are about
`1.82-1.83 ms`, with K/V conversion only around `0.02/0.05 ms` per ViT block.

The refreshed cuBLASLt attribution also does not show an easy heuristic win.
The dominant ViT groups remain `vit_mlp_fc2` (`58.92 ms`), `vit_mlp_fc1`
(`58.25 ms`), `vit_qkv` (`38.56 ms`), and `vit_attn_proj` (`14.20 ms`) across
the profiled encode calls
(`outputs/sam3-bf16-default-mulmat-profile-20260611h/cublaslt-groups.json`).
Sweeping `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX` over the tested alternatives did
not improve the row: index `0` reported `175.9 ms/frame`, index `1` regressed to
`192.3 ms/frame`, and index `2` reported `176.8 ms/frame`
(`outputs/sam3-bf16-cublaslt-algo-sweep-20260611h/summary.json`). Enabling
cuBLASLt `BIAS+GELU_ERF` also regressed (`180.9 -> 197.1 ms/frame`) and changed
checked outputs enough to remain rejected
(`outputs/sam3-bf16-cublaslt-gelu-erf-ab-20260611h/summary.json`,
`outputs/sam3-bf16-cublaslt-gelu-erf-ab-20260611h/compare.json`).

The same cuBLASLt direction was rechecked with the current post-revert BF16
row. A direct algorithm sweep over the tested indices still selected the default
index `0` as the best row: `175 ms/frame` for index `0`, `177-180 ms/frame` for
the closest alternatives, and `193-197 ms/frame` for the slower choices
(`outputs/sam3-bf16-cublaslt-algo-sweep-20260611j/results-parsed.tsv`). Turning
cuBLASLt bias fusion off entirely regressed to `217 ms/frame`
(`outputs/sam3-bf16-disable-cublaslt-bias-20260611j/benchmark.log`), confirming
that the existing bias fusion is still required. A timing run also showed that
descriptor/setup overhead is not the steady-state gap: after the first
descriptor build, the dominant `[3072/4736/1024,5184]` BF16 GEMM shapes had
`host_setup_ms` medians around `0.005 ms`, while the matmul medians remained
about `0.612 ms` for QKV, `0.934 ms` for `mlp.fc1`, and a split
`0.238/0.9 ms` distribution for the two `[1024,5184]` projection/fc2 shapes
(`outputs/sam3-bf16-cublaslt-timing-current-20260611j/summary-shape.json`).
Another `BIAS+GELU_ERF` rerun fused all 32 ViT `mlp.fc1` nodes, but still
regressed to `188` and `186 ms/frame` and changed the first checked mask hash
against the current row
(`outputs/sam3-bf16-cublaslt-gelu-erf-20260611j/gelu-erf-rerun1.log`,
`outputs/sam3-bf16-cublaslt-gelu-erf-20260611j/gelu-erf-rerun2.log`,
`outputs/sam3-bf16-cublaslt-gelu-erf-20260611j/compare-current-rerun2.json`).

The existing graph-level BF16 output-chain toggles were rechecked on the same
current code after the neck-pruning revert. They still do not win:
`SAM3_BF16_VIT_LINEAR_OUTPUT=1` measured `177.2 ms/frame`,
`SAM3_BF16_VIT_MLP_CHAIN=1` measured `191.8 ms/frame`, and
`SAM3_BF16_VIT_QKV_CHAIN=1` measured `196.0 ms/frame`, versus the paired default
rerun at `174.7 ms/frame`
(`outputs/sam3-bf16-current-toggle-rerun-20260611i/summary.json`). All three
also changed the first checked mask hash against the default row, with only
`2/3` equal mask hashes in the JSONL comparison files under the same output
directory. This reinforces that isolated graph-level casts are the wrong level
of abstraction for the remaining gap.

The CUDA node-profile parser was fixed to handle `GGML_CUDA_PROFILE_NODE` names
that contain spaces, such as `"(reshaped) (permuted)"`. Before that fix, many
`CONT` rows were silently missing from the hotspot summary. The regenerated
profile now shows an additional root surface: SAM3 ViT QKV layout
materialization costs `13.20 ms` drop-max, and Q/K contiguous materialization
before RoPE costs another `8.52 ms` for window attention plus `1.17 ms` for
global attention
(`outputs/sam3-bf16-default-mulmat-profile-20260611h/profile-summary-v2.json`,
`outputs/sam3-bf16-default-mulmat-profile-20260611h/hotspots-sam3-encode-v2.json`).
This makes the next CUDA target more specific: the remaining non-GEMM path is
not generic graph overhead, but the QKV projection output being reshaped,
permuted, materialized, RoPE-adjusted, and then fed to FlashAttention as several
separate f32/contiguous intermediates. A useful root optimization should fuse or
specialize that SAM3 head64 QKV/RoPE/FATTN dataflow, rather than adding another
post-hoc BF16 output toggle.

A narrow graph-pruning probe that built only the three SAM3 neck outputs used by
the tracker was parity-safe against itself (`min_bbox_iou=1`,
`max_bbox_delta_px=0`, `mask_hash_equal_rows=6`), but it did not improve speed:
the measured row was `175.9 ms/frame`
(`outputs/sam3-bf16-neck3-skip-current-20260611h/summary.json`,
`outputs/sam3-bf16-neck3-skip-ab-20260611h/compare-repeat.json`). It is therefore
recorded as negative evidence, not kept as an accepted default. After reverting
that probe, a repeat run returned to the expected current band at
`174.7 ms/frame`, with exact JSONL agreement against the preceding run
(`mask_hash_equal_rows=3`, `min_bbox_iou=1`,
`outputs/sam3-bf16-current-after-neck-revert-20260611i/summary-rerun.json`,
`outputs/sam3-bf16-current-after-neck-revert-20260611i/compare-run1-run2.json`).
The next useful optimization should be a root kernel/dataflow change that avoids
materializing SAM3 ViT low-precision linear, activation, and attention
intermediates as f32 in ggml, rather than another small wrapper-level graph
toggle.

Additional current-row toggles keep pointing to the same conclusion. Disabling
CUDA graphs measured `177 ms/frame`, forcing cublas compute16 measured
`175.8 ms/frame`, and enabling the ViT positional cache measured
`199 ms/frame`; none is an accepted optimization
(`outputs/sam3-bf16-disable-cuda-graphs-20260611j/benchmark.log`,
`outputs/sam3-bf16-force-compute16-20260611j/benchmark.log`,
`outputs/sam3-bf16-vit-pos-cache-20260611j/benchmark.log`). Removing the
default BF16 `attn.proj` input cast kept exact output parity but stayed flat at
`176 ms/frame`; removing all ViT linear input casts or only the `fc2` input cast
regressed to `207.8` and `205.4 ms/frame`
(`outputs/sam3-bf16-disable-attn-proj-input-cast-20260611j/benchmark.log`,
`outputs/sam3-bf16-disable-linear-input-casts-20260611j/benchmark.log`,
`outputs/sam3-bf16-disable-fc2-input-cast-20260611j/benchmark.log`). A more
radical 4D batched-linear probe kept the window/global dimensions instead of
flattening to `[C,5184]`, and cuBLASLt fusion did succeed on shapes such as
`[3072,24,24,9]` and `[4736,72,72,1]`, but end-to-end tracking regressed to
`254.5 ms/frame`
(`outputs/sam3-bf16-experiment-batched-linear-20260611j/profile-summary.json`,
`outputs/sam3-bf16-experiment-batched-linear-20260611j/fusion.stderr.log`).
Those probes were removed. The remaining viable root path is therefore a
dedicated QKV/RoPE/FATTN dataflow or custom CUDA op that avoids the current
materialized f32 `cont(permute)` and Q/K pack sequence while preserving raw
tensor parity, not another global backend flag.

The next accepted ggml CUDA root-path change fuses the SAM3 ViT
`CONT -> RoPE pair` pattern. The old graph first materialized the Q/K packed
contiguous tensor, then built real/imag views, multiplied by cosine/sine, and
materialized the final RoPE output. The new path recognizes that exact sequence
and writes the existing RoPE output layout directly from the original strided
Q/K tensor. Set `GGML_CUDA_DISABLE_CONT_ROPE_PAIR_FUSION=1` to force the old
path. On a short SAM3 BF16 probe, the fusion fired `132` times and preserved
exact JSONL parity against the disabled path. Three paired bbox-only repeats
measured disabled `177/177/192 ms/frame` versus enabled `174/172/175 ms/frame`;
the mean delta was `8.33 ms/frame` in favor of the fused path, but with high
run-to-run noise from one disabled outlier. Full-mask and bbox-only A/B checks
both kept `min_bbox_iou=1.0`, zero bbox/score deltas, and all checked
`mask_hash` rows equal
(`outputs/sam3-bf16-cont-rope-fusion-probe-20260611/repeat/`,
`outputs/sam3-bf16-cont-rope-fusion-probe-20260611/compare-disabled-enabled.json`,
`outputs/sam3-bf16-cont-rope-fusion-probe-20260611/compare-disabled-enabled-bbox.json`).

This is accepted as a small dataflow cleanup, not as the Python-closing
optimization. The node profile with `GGML_CUDA_PROFILE_NODES=1` moved total
profiled node time only from about `886.6 ms` disabled to `878.8 ms` enabled on
the two-frame probe, while skipping `797` additional nodes. The remaining
SAM3 BF16 gap is still the larger ViT path: QKV/projection/MLP GEMMs plus the
FlashAttention data contract around those tensors. A larger win still needs a
native head64 QKV/RoPE/FATTN path or a fused BF16 ViT block dataflow comparable
to the official Python/CUTLASS execution.

The fused `CONT -> RoPE pair` kernel now stores the adjacent real/imag output
pair with `float2`, `half2`, or `bfloat162` when the output stride is contiguous.
This does not change the graph contract or arithmetic order; it only replaces
two scalar stores in the accepted fusion. A fresh 3-pair check against
`GGML_CUDA_DISABLE_CONT_ROPE_PAIR_FUSION=1` stayed exact on all checked rows:
`min_bbox_iou=1.0`, zero bbox/score deltas, and `mask_hash_equal_rows=3` in
each pair. Timing remained noisy but positive on this short bbox-only probe:
disabled `175/200/177 ms/frame` versus enabled `174/177/174 ms/frame`,
mean delta `9.0 ms/frame`, median delta `3.0 ms/frame`
(`outputs/sam3-bf16-cont-rope-vecstore-20260611/repeat/`).

A cuDNN SDPA head64 bridge was also added as an unsafe opt-in diagnostic, not a
default optimization. It is gated behind both
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD64=1` and
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD64_UNSAFE_RUN=1`; F32 inputs require
`GGML_CUDA_CUDNN_SDPA_ALLOW_F32_TO_BF16=1`. By default this only allows the
SAM3 global `5184 x head64` shape; window `576 x head64` additionally requires
`GGML_CUDA_CUDNN_SDPA_HEAD64_ALLOW_WINDOW=1`. The global-only probe routed the
8 global attention calls but was slower and changed outputs: default
`173/173/175 ms/frame` versus cuDNN `174/175/175 ms/frame`, with
`min_bbox_iou=0.9983367626`, `max_bbox_delta_px=0.072`, and only `2/3`
mask hashes equal in each pair. Allowing window attention routed all 64 SAM3
ViT head64 calls but regressed to about `217.8 ms/frame` on the short probe and
also changed mask hashes. The bridge is therefore useful only for follow-up
kernel comparison, not for accepted same-parity performance
(`outputs/sam3-bf16-cudnn-head64-probe-20260611/`).

Another head64 FATTN conversion probe tried to combine the existing separate
F32-to-F16 K and V conversion launches into one `half2` pair-conversion kernel.
It is now available only as an explicit diagnostic with
`GGML_CUDA_ENABLE_FATTN_F32_TO_F16_PAIR_VEC2=1`; the default path keeps the
previous separate vec2 converters. The opt-in path preserved exact JSONL parity
against default on the two checked rows, but it was slower inside FATTN:
window head64 K/V conversion sum changed from `4.718208 ms` default to
`5.293312 ms` opt-in across the captured calls, and total window FATTN time
changed from `25.581984 ms` to `26.645952 ms`. Global head64 was roughly flat
(`14.528512 ms` default versus `14.465408 ms` opt-in), so the aggregate effect
does not justify a default. The result confirms that the next FATTN work should
target the kernel body or avoid the conversion boundary entirely, not merely
merge the existing K/V conversion launches
(`outputs/sam3-bf16-fattn-kv-pair-20260611/`).

The fused `UNARY(GELU_ERF) -> CPY(F32->BF16/F16)` kernel now has a wider
aligned `float4` path. When the source and destination are contiguous, 16-byte
source aligned, 8-byte destination aligned, and the element count is divisible
by four, one thread loads four F32 values and writes two packed BF16/F16 pairs.
Set `GGML_CUDA_DISABLE_UNARY_CPY_VEC4=1` to force the previous `float2` fused
path. This keeps the same scalar `GELU_ERF` operation per element and preserves
the same output contract. Three paired bbox-only SAM3 BF16 runs against the
previous vec2 path were exact (`min_bbox_iou=1.0`, zero bbox/score deltas, all
checked mask hashes equal) and measured vec2 `172/193/173 ms/frame` versus
vec4 `173/174/173 ms/frame`; the median end-to-end delta is effectively flat
because of one vec2 outlier. A node-profile A/B is the safer local signal:
fused `mlp_gelu` UNARY nodes moved from `18.186688 ms` to `17.987552 ms`
across the captured profile, with exact JSONL parity
(`outputs/sam3-bf16-unary-cpy-vec4-20260611/`).

The earlier short same-contract SAM3 BF16+TF32 speed row that appeared to beat
official Python is now superseded by the stricter target-stable harness below.
The benchmark uses
`sam3_propagate_frame()` for SAM3 text-prompt runs after frame-0 text detection,
matching the official Python `add_prompt` then `propagate_in_video` measurement
range. It also uses tracker-neck-only image encoding during propagation, so it
no longer builds the detector neck after initialization. Subsequent graph-level
ViT changes flatten window-batch and global linear inputs into `[C,5184]` before
QKV/projection/MLP GEMMs, matching PyTorch linear's large-GEMM shape instead of
launching smaller batched GEMMs, and preserved exact JSONL parity across their
local A/B checks.

The previous matrix evidence that reported C++ `221.4 ms/frame` versus official
Python `119.71 ms/frame` is superseded by the current BF16 CUDA path. The latest
6-frame, 2-repeat, same decoded source resolution, same 1008 encode size,
same prompt, same measured tracking scope, and same 2-warmup policy run reports
C++ `177.25 ms/frame` versus official Python SAM3 BF16+TF32
`182.88246260490268 ms/frame`, so Python/C++ is `1.0317769399430334x`
and the C++ row is about `3.18%` faster
(`outputs/model-matrix-sam3-bf16-current-20260611-cpy3/summary.json`). This is
only a SAM3 BF16+TF32 short-run speed result; SAM3.1 coverage and full
C++/official-output quality parity remain separate acceptance items.

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

## 2026-06-10 F16 Attention Refresh

The earlier `SAM3_FAST_F16_VIT_ATTENTION=1` row in this document is now stale.
After the tracker PE cache, prompt reference, and F16 MLP fast-path changes, the
same flag was re-tested on the 10-frame bedroom contract with two warmup runs.
It is now accepted for the SAM3 F16 path and enabled by default because it is
parity-safe and not slower, but it is not enough to claim a Python speed win.

Current evidence:

| Path | Result | Evidence |
| --- | ---: | --- |
| C++ `sam3-f16`, default before attention default-on | `210.5 ms/frame` | `outputs/model-matrix-sam3-f16-normal-warmup2-pe-prewarm/summary.json` |
| C++ `sam3-f16`, attention F16 input default-on | `210.13 ms/frame` | `outputs/model-matrix-sam3-f16-fast-attn-default-r3/summary.json` |
| C++ `sam3-f16`, attention F16 input disabled | about `211 ms/frame` | `outputs/sam3-f16-disable-fast-attn-r3/run-*.log` |
| Official Python SAM3, fp16/TF32 | `205.85 ms/frame` | `outputs/model-matrix-sam3-f16-fast-attn-default-r3/summary.json` |

Parity against the previous default is bit-exact for the checked tracking
surface: `outputs/sam3-f16-fast-attn-current/default-vs-attn-summary.json`
reports 10/10 equal mask hashes, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0.0`, and `max_score_abs_delta=0.0`.

Operationally, the fast path is now default-on for F16 QKV weights. Set
`SAM3_DISABLE_FAST_F16_VIT_ATTENTION=1` to force the previous graph for A/B
checks. `SAM3_FAST_F16_VIT_ATTENTION=0` remains accepted as a local override
when the disable variable is not set.

The F16 linear-output route remains rejected. A ggml CUDA safety fix now allows
`ggml_mul_mat_cast(..., GGML_TYPE_F16)` to fall back through the normal cublas
path instead of asserting when cublasLt bias fusion is disabled, but the measured
path is still slower (`233.3 ms/frame` with MLP F16 output and cublasLt bias
fusion disabled; `outputs/sam3-f16-linear-output-no-cublaslt/speed-after-fallback-fix.log`).
Splitting MLP F16 output into fc1-only and fc2-only was also slower, so these
remain diagnostic toggles rather than accepted defaults.

Splitting the attention linear-output toggle showed the same issue more
directly. `SAM3_F16_VIT_QKV_OUTPUT=1 SAM3_F16_VIT_ATTN_PROJ_OUTPUT=0` makes the
SAM3 ViT head64 FATTN calls use `types=f16/f16/f16` and improves the profiled
window-attention per-call timing, but the end-to-end row regresses to
`223 ms/frame` (`outputs/sam3-f16-qkv-output-current/qkv-speed.log`) versus
`210 ms/frame` for the default in the paired run. Mask hashes stayed equal for
10/10 frames, but bbox/score had tiny numeric deltas
(`outputs/sam3-f16-qkv-output-current/default-vs-qkv-summary.json`). Enabling
`GGML_CUDA_CUBLASLT_DIRECT_F16_DST=1` with qkv-only output was still slower at
`220 ms/frame` (`outputs/sam3-f16-qkv-output-direct-current/speed.log`). The
root cause is visible in `ggml/src/ggml-cuda/fattn-common.cuh`: the common FATTN
launcher still expands Q to f32 before the kernel, so this path removes only
part of the conversion cost while adding low-precision output/cast overhead.
The native head64 F16-Q loader now exists behind this diagnostic path. With
`SAM3_F16_VIT_QKV_OUTPUT=1 SAM3_F16_VIT_ATTN_PROJ_OUTPUT=0`, the native loader
removes the Q expansion and improves the qkv-output A/B row from
`221.3 ms/frame` with `GGML_CUDA_DISABLE_FATTN_F16_Q=1` to `217.3 ms/frame`;
`outputs/sam3-f16-native-f16-q-smoke/disabled-vs-native.json` reports 10/10
equal mask hashes. The split FATTN profile confirms the intended local effect:
`outputs/sam3-f16-fattn-profile-qkv-native-current/summary.json` reports total
FATTN Q conversion `1.05 ms` versus `17.11 ms` before the native loader in
`outputs/sam3-f16-fattn-profile-qkv-output-current/summary.json`.

This is still not an accepted end-to-end default. In the same smoke contract,
default F16 reports `209.3 ms/frame`, qkv-output plus native F16 Q reports
`220.2 ms/frame`, and adding `SAM3_FAST_F16_VIT_ATTN_PROJ_INPUT=1` reports
`219.6 ms/frame` (`outputs/sam3-f16-native-f16-q-combos/summary.txt`). The root
cause moved from Q expansion to the graph-level cost of materializing F16 QKV
outputs and downstream casts. The next useful implementation is a fused or
lower-overhead low-precision Q/K/V dataflow, or a cuDNN SDPA integration that is
fed by that parity-safe graph instead of adding isolated conversions.

The cuDNN SDPA feasibility bench is now runnable with the locally available F16
SAM3 model by default. Override the model row with `CUDNN_SDPA_MODEL_FILTER` if
a BF16 GGML checkpoint is present. The latest F16 feasibility run is
`outputs/cudnn-sdpa-head64-f16-current/summary.json`: pure F16 cuDNN IO is much
faster for the two head64 attention shapes, but F16-from-F32 conversion makes
the global shape slower and the window shape only modestly faster. A production
cuDNN integration should therefore be paired with a parity-safe low-precision
Q/K/V path, not inserted as an isolated F32-to-F16 conversion wrapper.

### FATTN F32 K/V Conversion Vec2 Refresh

The latest accepted ggml CUDA head64 FATTN cleanup is narrower than the rejected
graph-level low-precision Q/K/V paths. When the existing launcher must convert
contiguous F32 K/V tensors to F16 for the MMA kernel, it now uses a FATTN-local
`float2 -> half2` converter instead of the generic scalar converter. The tensor
contract and downstream FATTN kernel are unchanged, and
`GGML_CUDA_DISABLE_FATTN_F32_TO_F16_VEC2=1` restores the generic converter for
A/B checks.

The SAM3.1 ViT block attention-core slice was checked on blocks 15, 23, and 31
with the same saved Q/K/V inputs. All three blocks were bit-exact against the
old converter (`max_abs=0`, `nonzero=0`) and finite
(`outputs/sam31-fattn64-vec2-convert-probe-20260611a/summary.json`).

| Block | Stage mean delta | FATTN `total_ms` delta | K convert delta | V convert delta |
| --- | ---: | ---: | ---: | ---: |
| 15 | `-1.42%` | `-6.16%` | `-75.32%` | `-22.39%` |
| 23 | `-3.15%` | `-5.57%` | `-77.69%` | `-32.00%` |
| 31 | `+0.22%` first run, `-3.72%` reverse-order recheck | `-5.49%` first run, `-5.32%` recheck | about `-75%` | `-25%` to `-40%` |

This is a small but real kernel-side cleanup for the current head64 fallback
path. It does not close the Python gap by itself because the dominant remaining
work is still ViT GEMM and the FATTN kernel body, but it removes measurable
conversion overhead without changing raw outputs.

A wider `float4 -> half2x2` variant was also tested and rejected. It was
bit-exact against the accepted vec2 converter on the same blocks, but the
attention-core stage mean regressed on all three blocks (`+1.90%`, `+2.69%`,
and `+0.30%` for blocks 15, 23, and 31). Although the profiled FATTN
`total_ms` was slightly lower, the slice-level timing is the acceptance signal
for this path, so the vec4 code was not kept
(`outputs/sam31-fattn64-vec4-convert-probe-20260611a/summary.json`).

### Stream-K Heuristic Refresh

A smaller CUDA-side heuristic was accepted after the qkv-only rejection. The
previous ggml CUDA FATTN launcher used stream-k on Ada/Blackwell whenever the
kernel supported it, which made the SAM3 ViT head64 shapes choose
`stream_k_general_fixup` even though their output tile count already fills the
GPU. The heuristic now keeps stream-k for underfilled tile grids, AMD WMMA, or
explicit `GGML_CUDA_FORCE_FATTN_STREAM_K=1`, but otherwise uses
`one_block_per_tile` when `ntiles_dst >= max_blocks`.

Evidence:

| Path | Result | Evidence |
| --- | ---: | --- |
| C++ `sam3-f16`, old stream-k default evidence | `210.13 ms/frame` | `outputs/model-matrix-sam3-f16-fast-attn-default-r3/summary.json` |
| C++ `sam3-f16`, refreshed stream-k heuristic | `208.83 ms/frame` | `outputs/model-matrix-sam3-f16-streamk-heuristic-r3/summary.json` |
| Official Python SAM3, fp16/TF32 in refreshed run | `206.97 ms/frame` | `outputs/model-matrix-sam3-f16-streamk-heuristic-r3/summary.json` |

The refreshed same-contract F16 row is still not a Python speed win:
`python_over_cpp_track_ratio=0.9911`, so C++ remains about `0.89%` slower.
The paired stream-k parity check in
`outputs/sam3-f16-streamk-parity-current/summary.json` reports 10/10 equal mask
hashes, `min_bbox_iou=0.9999309509`, `max_bbox_delta_px=0.003`, and
`max_score_abs_delta=4.4e-05`.

For diagnosis, ggml CUDA now also accepts `GGML_CUDA_FORCE_FATTN_TILE=1`,
`GGML_CUDA_FORCE_FATTN_VEC=1`, and `GGML_CUDA_FORCE_FATTN_MMA=1`. These are not
accepted speed defaults; they are for controlled kernel-selection A/B tests.
The TILE force path was clearly slower in the SAM3 F16 smoke run
(`281 ms/frame`, `outputs/sam3-f16-fattn-force-current-v2/tile.log`).

### CUDA Window Partition Vec4 Refresh

The next accepted F16 win is in ggml CUDA `WIN_PART` / `WIN_UNPART`. The SAM3
ViT window-attention path repeatedly copies contiguous channel-major F32 windows
such as `[1024,24,24,9]`. The previous CUDA kernel used one thread per scalar
element; the new default path copies four F32 channels per thread when the
channel dimension and strides are compatible. Set
`GGML_CUDA_DISABLE_WIN_PART_VEC4=1` to force the old scalar path for A/B checks.

Evidence:

| Path | Result | Evidence |
| --- | ---: | --- |
| C++ `sam3-f16`, scalar window partition | `209.80 ms/frame` | `outputs/sam3-f16-winpart-vec4-r3/summary.json` |
| C++ `sam3-f16`, vec4 window partition | `205.73 ms/frame` | `outputs/sam3-f16-winpart-vec4-r3/summary.json` |
| C++ `sam3-f16`, vec4 same-contract model matrix | `205.43 ms/frame` | `outputs/model-matrix-sam3-f16-winpart-vec4-r3/summary.json` |
| Official Python SAM3, fp16/TF32, same contract | `206.98 ms/frame` | `outputs/model-matrix-sam3-f16-winpart-vec4-r3/summary.json` |

The paired scalar/vec4 C++ run reports a `1.98%` speedup with 10/10 equal mask
hashes, `min_bbox_iou=1.0`, `max_bbox_delta_px=0.0`, and
`max_score_abs_delta=0.0`
(`outputs/sam3-f16-winpart-vec4-r3/disabled-vs-enabled.json`).

The same-contract Python comparison is now a narrow but valid F16 speed win:
`python_over_cpp_track_ratio=1.0075`, with the comparison contract marked
`comparable_speed_claim=true`. This row uses the same decoded 960x540 source
frames, the same 10-frame range, the same text prompt, the same 1008 model input
resolution, and the same two warmup runs. It does not close the broader goal by
itself because SAM3.1 full C++ propagation is still not wired, but it does move
SAM3 F16 from "slower than Python" to "slightly faster than Python" under the
accepted contract.

Two related probes were rejected:

| Probe | Result | Reason |
| --- | ---: | --- |
| `ggml_conv_2d_direct` for SAM3 neck 3x3 convs | `514.5 ms/frame` vs `208.4 ms/frame` default | parity-safe but much slower than im2col+GEMM |
| cuBLASLt bias-fusion gap check | 659 successes, 0 rejects | no large remaining GEMM bias-fusion miss to fix |

### SAM3.1 Multiplex Memory Backbone Slice

SAM3.1 tracking now has a second parity-checked slice beyond the multiplex mask
decoder. The C++ loader binds native
`trk.model.maskmem_backbone.*` tensors into the existing memory-encoder field
layout, and a dedicated runner executes the official
`SimpleMaskEncoder.forward(pix_feat, masks, skip_mask_sigmoid=True)` feature
path:

`mask_downsampler -> pix_feat_proj -> add -> fuser.layers[0..1] -> vision_features`

The same runner also dumps the deterministic `vision_pos_enc` produced by
`PositionEmbeddingSine` for the memory feature grid, then stores both tensors
through the C++ tracker memory slot path. The stored aliases correspond to the
official Python prompt state fields `maskmem_features` and `maskmem_pos_enc`.
It also builds the flattened memory-attention `prompt` and `prompt_pos` tensors
from the stored slot for a frame-1 propagation lookup of the frame-0
conditioning memory (`t_pos=1`).

The parity case uses the official SAM3.1 checkpoint, `pix_feat` shape
`[1,256,72,72]`, multiplex mask shape `[1,32,1152,1152]`, and a F32 ggml slice
model generated with `--sam31-memory-backbone-only`. SAM3.1 conversion now
overrides the v4 header `mem_out_dim` to `256`, matching the official
`maskmem_features [1,256,72,72]` contract. The memory-backbone parity recipe
rebuilds this small F32 slice every run so stale `mem_out_dim=64` headers are
not reused.

Evidence:

| Tensor | max_abs | mean_abs | Evidence |
| --- | ---: | ---: | --- |
| `mask_downsampled` | `0.0042800` | `0.0001799` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `pix_feat_proj` | `0.0011970` | `0.0000870` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `fused_input` | `0.0043629` | `0.0002095` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `fuser0` | `0.0041886` | `0.0002564` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `vision_features` | `0.0042401` | `0.0002713` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `vision_pos_enc` | `0.0000000596` | `0.00000000246` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `stored_spatial_feats` | `0.0042401` | `0.0002713` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `stored_spatial_pe` | `0.0000000596` | `0.00000000246` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `memory_prompt` | `0.0042401` | `0.0002713` | `outputs/sam31-memory-backbone-compare/summary.json` |
| `memory_prompt_pos` | `0.000000119` | `0.00000000235` | `outputs/sam31-memory-backbone-compare/summary.json` |

The compare status is `ok` under `max_abs <= 1e-2` and `mean_abs <= 1e-3`.

### SAM3.1 Propagation Feature-Adapter Slice

SAM3.1 image encoding now has a dedicated propagation feature path. The C++
loader binds `det.backbone.vision_backbone.propagation_convs.*` into the tracker
neck fields and the SAM3.1 encode path produces the official propagation
features expected by the multiplex tracker:

`propagation_convs[0] -> sam_mask_decoder.conv_s0 -> projected_s0 [32,4H,4H]`

`propagation_convs[1] -> sam_mask_decoder.conv_s1 -> projected_s1 [64,2H,2H]`

`propagation_convs[2] -> image_features [256,H,H]`

A new parity case feeds deterministic random `vit_out` tokens into the official
Python checkpoint weights and the C++ graph. This verifies the feature-adapter
operation order and tensor layouts without requiring a full ViT image pass.

| Tensor | max_abs | mean_abs | Evidence |
| --- | ---: | ---: | --- |
| `projected_s0` | `0.0003115` | `0.0000421` | `outputs/sam31-propagation-features-compare/summary.json` |
| `projected_s1` | `0.0001800` | `0.0000213` | `outputs/sam31-propagation-features-compare/summary.json` |
| `image_features` | `0.000000387` | `0.0000000585` | `outputs/sam31-propagation-features-compare/summary.json` |

The compare status is `ok` under `max_abs <= 1e-2` and `mean_abs <= 1e-3`.
The runtime `sam3_encode_image_impl` now dispatches SAM3.1 models to this
propagation image path, so the existing SAM3.1 propagation branch can receive
the projected high-resolution decoder features it requires.

### SAM3.1 Multiplex Memory-Attention Slice

The next boundary is now also isolated. The C++ loader binds native
`trk.model.transformer.encoder.*` tensors, including SAM3.1-specific
`image_cross_attn_q_proj` and `image_cross_attn_k_proj`, into the tracker
memory-attention structure. A dedicated runner executes the official SAM3.1
decoupled encoder order:

`src + 0.1 * src_pos -> 4x(self-attn, image/object cross-attn, GELU FFN) -> final norm`

Two parity checks are available:

| Case | Precision | Shape | max_abs | mean_abs | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| strict structure check | F32 Python CPU vs F32 GGML CUDA | `[256,16]` | `0.0027169` | `0.0005007` | `outputs/sam31-memory-attention-compare-fs4-f32/summary.json` |
| production token length | FP16 Python CUDA FlashAttention vs FP16 GGML CUDA | `[256,5184]` | `0.2609522` | `0.0150387` | `outputs/sam31-memory-attention-compare/summary.json` |

The full token-length case uses looser thresholds (`max_abs <= 0.45`,
`mean_abs <= 0.02`) because it compares two different FP16 attention kernels
over four residual layers. The strict F32 small-shape case remains the guard
against shape/order mistakes.

### SAM3.1 Memory-Attention To Mask-Decoder Bridge

The next bridge is now isolated as a single graph: official Python produces
random memory-attention inputs, runs the SAM3.1 multiplex encoder, reshapes the
encoder output as propagation `image_embeddings`, then calls
`MultiplexMaskDecoder.predict_masks`. The C++ runner executes the same
`memory-attention -> multiplex mask decoder` sequence from the dumped inputs.

The strict default parity gate is F32 Python CPU vs F32 GGML CPU at
`feat-size=4`. It uses a small shape so that this remains a fast structural
guard for the propagation boundary.

| Tensor | max_abs | mean_abs | Evidence |
| --- | ---: | ---: | --- |
| `sam31_mem_attn_output` | `0.00000381` | `0.000000676` | `outputs/sam31-memory-attention-decoder-compare/summary.json` |
| `masks` | `0.0008130` | `0.0001167` | `outputs/sam31-memory-attention-decoder-compare/summary.json` |
| `iou_pred` | `0.0000381` | `0.0000158` | `outputs/sam31-memory-attention-decoder-compare/summary.json` |
| `mask_tokens_out` | `0.00000572` | `0.000000632` | `outputs/sam31-memory-attention-decoder-compare/summary.json` |
| `object_score_logits` | `0.00000191` | `0.000000998` | `outputs/sam31-memory-attention-decoder-compare/summary.json` |

A CUDA F32 run of the same bridge also keeps the structural outputs close, but
`iou_pred` observed `max_abs=0.0502625` and `mean_abs=0.0132246`, so it is kept
as backend numerical evidence rather than the strict parity gate.

The parity-checked bridge has now been promoted into the runtime tracking
branch for the first supported shape: one valid object in multiplex slot 0. The
SAM3.1 `sam3_encode_memory` branch uses the 32-channel multiplex memory-mask
contract, keeps the SimpleMaskEncoder identity `out_proj`, stores
`maskmem_features` and `maskmem_pos_enc`, and also saves the previous-frame
`image_features` / `image_pos_enc` needed by SAM3.1 decoupled cross-attention.
`sam3_propagate_single` now dispatches SAM3.1 models to the
`memory-attention -> multiplex mask decoder` graph and selects the best of the
three mask tokens for slot 0 by IoU.

This is still not enough for a full SAM3.1 speed claim. The remaining
implementation boundary is now narrower: detection-mask prompt-state
initialization is wired through the SAM3.1 memory encoder and accepts raw
low-resolution mask logits when the caller already has decoder logits. The
full-model smoke recipe `just sam31-tracking-mask-init-smoke` now loads the
F16 SAM3.1 GGML checkpoint, initializes one instance from a synthetic mask, and
propagates one frame through image encode, memory encode, memory-attention, and
the multiplex decoder. The latest smoke log is
`outputs/sam31-tracking-mask-init-smoke/summary.log`.

Point/box prompt parity and an official Python sequence comparison are still
required. That comparison must exercise the full image encode, memory encode,
memory-attention, multiplex decoder, and mask selection path before the
same-contract C++ vs Python timing claim.

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
4. Finish SAM3.1 prompt-state parity beyond detection-mask initialization, then
   run a full official Python sequence comparison that covers the runtime
   multiplex tracking branch.

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
incomplete, SAM3 C++ had not yet beaten official Python on the same contract,
and SAM3.1 still lacked the full C++ loader/graph path at that point in the
goal loop.

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

## 2026-05-22 Historical Goal Loop Update

At this point in the goal loop, C++ SAM3 BF16 still did not beat official Python
SAM3 BF16+TF32 on the same decoded frames, prompt, frame range, and 1008 model
input contract. This is historical evidence; the current accepted SAM3 F16 row
above is the active same-contract win evidence. The SAM3-only rerun is recorded in
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
`cuda_health=ok`.

## Current SAM3.1 End-to-End Evidence

`just sam3-sam31-goal-audit` now includes a same synthetic two-frame SAM3.1
mask-init sequence through both C++ and official Python. The C++ full-model
smoke loads `models/sam3.1/sam3.1_multiplex-f16.ggml`, adds a rectangle mask on
frame 0, propagates to frame 1, and writes masks under
`outputs/sam31-tracking-mask-init-smoke/`. The official Python comparison uses
`models/sam3.1/sam3.1_multiplex.pt`, BF16 autocast, and TF32 enabled, then
writes `outputs/sam31-mask-sequence-python/summary.json`.

Latest measured result:

| Path | Input | Result | Evidence |
| --- | --- | ---: | --- |
| C++ SAM3.1 mask-init smoke | 2 synthetic frames, 320x240, F16 GGML, CUDA | frame0 `578.399 ms`, propagate `306.536 ms` | `outputs/sam31-tracking-mask-init-smoke/summary.log` |
| Official Python SAM3.1 mask sequence | same frames/mask, BF16+TF32 | add-mask `38.6533 ms`, propagate `26.0039 ms` | `outputs/sam31-mask-sequence-python/summary.json` |
| C++ vs Python frame-1 mask | same output size | IoU `0.8830955355381682`, xor `1778` pixels | `outputs/sam31-mask-sequence-python/summary.json` |

This is not accepted parity. The slice-level parity gates are useful, but the
streaming prompt-state and propagation contract still diverges from official
Python. The next SAM3.1 work should compare the detection-mask prompt state
immediately after frame 0, then narrow whether the divergence enters at
low-resolution logits, memory encoding, object-score handling, or memory-bank
construction.

The current mask-init state comparison has narrowed the divergence:

| Tensor | Status | Max abs | Mean abs | Evidence |
| --- | --- | ---: | ---: | --- |
| `memory_mux_mask_interpolated` | failed, boundary-only mismatch | `2.0` | `0.0001872520387908559` | `outputs/sam31-mask-init-state-compare/summary.json` |
| `stored_spatial_pe` | ok | `5.960464477539063e-08` | `2.4575099486254555e-09` | same |
| `image_pos_enc` | ok | `0.00195235013961792` | `0.0003452987854483185` | same |
| `image_features` | failed | `1.9427968263626099` | `0.07511368198000842` | same |
| `stored_spatial_feats` | failed | `2.8374671936035156` | `0.18116487516893784` | same |
| `obj_ptr` | ok | `0.12690982222557068` | `0.0024847209606377874` | `outputs/sam31-mask-init-state-compare/obj_ptr-summary.json` |
| `object_score_logits` | ok | `0.0` | `0.0` | same |

The SAM3.1 propagation graph now supports non-zero object-pointer tail tokens
and excludes those tail keys from cross-attention RoPE. The C++ loader now also
binds the SAM3.1 interactive prompt encoder, interactive SAM mask decoder,
interactive feature adapter, normal `obj_ptr_proj`, and
`interactive_obj_ptr_proj` tensors into internal C++ structures. The synthetic
mask-init path now runs the SAM3.1 interactive mask-input head when the caller
adds a plain external mask without a SAM decoder token, stores the resulting
object pointer, and emits mask-present `object_score_logits=10` for the state
comparison. That fixes the previous zero-pointer shortcut: object pointer parity
is now within tolerance.

The latest mask-preprocessing fix also changes SAM3.1 synthetic detection-mask
memory input from `1/-1` logits to the official `1/0` mask contract, then
binarizes at the model input resolution before the memory-backbone interpolation
step. This improved the synthetic frame-1 C++ versus Python mask IoU from
`0.8804943219772879` to `0.8830955355381682`, but it does not close parity.

This is still not an accepted SAM3.1 speed or parity result. The interactive
head currently stages encoded image features through host F32 buffers to avoid a
separate CUDA graph-buffer ownership issue, so its frame0 timing is not a final
performance datapoint. The remaining correctness gap is now concentrated in the
memory path: the anti-aliased mask interpolation boundary mismatch remains, and
`image_features` / `stored_spatial_feats` still exceed tolerance before frame-1
propagation.

## 2026-06-10 SAM3.1 Parity Recheck

The SAM3.1 mask-init dump had a measurement bug: ViT captures were overwritten
by a later frame, while `input_image_preprocessed` was still frame 0. The Python
dump now records `vit_patch_embed_input` and keeps the first ViT capture for the
sequence. After this fix, `input_image_preprocessed` and `vit_patch_embed_input`
match exactly in `outputs/sam31-mask-init-state-python-fixed/expected_cpp_layout`.

A F32 visual-only GGML was generated at
`models/sam3.1/sam3.1_multiplex-f32.ggml` to compare against official Python's
FP32 weights. With the fixed Python dump and the F32 GGML, patch embedding now
matches within tolerance:

| Tensor | Max abs | Mean abs | Evidence |
| --- | ---: | ---: | --- |
| `vit_patch_embed` | `0.03544139862060547` | `0.0008829233773609681` | `outputs/sam31-vit-prefix-patch-fixed-f32/summary.json` |

The new diagnostic `sam31_vit_block_case` compares ViT block sub-stages using
official Python tensors as inputs. Block 0 is within tolerance for norm,
window partition/unpartition, qkv, attention, projection, and MLP stages; the
stage summary is in `outputs/sam31-vit-block-stage-current/summary.json`.

Full mask-init state with F32 GGML is improved at the ViT prefix but still not
accepted end-to-end:

| Tensor | Status | Max abs | Mean abs | Evidence |
| --- | --- | ---: | ---: | --- |
| `vit_patch_embed` | ok | `0.1168169379234314` | `0.0024976116195042177` | `outputs/sam31-mask-init-state-compare-fixed-f32/summary.json` |
| `vit_after_pos` | ok | `0.1168169379234314` | `0.002497611621191395` | same |
| `vit_block_00_out` | ok | `0.4802582114934921` | `0.022943686665165` | same |
| `vit_output` | failed | `171.47027206420898` | `0.1792834728932645` | same |
| `image_features` | failed | `1.9671211242675781` | `0.07070565307816541` | same |
| `stored_spatial_feats` | failed | `2.8495171070098877` | `0.1798270657365592` | same |
| `obj_ptr` | ok | `0.14266278129070997` | `0.0026768686412879106` | same |

The current same-sequence speed comparison is:

| Path | Runs | Mean frame0/add-mask path | Mean propagate path | Evidence |
| --- | ---: | ---: | ---: | --- |
| C++ SAM3.1 BF16 GGML smoke | 3 | `676.385 ms` frame0+mask init | `296.224 ms` | `outputs/sam31-speed-smoke-bf16-20260610/summary.json` |
| C++ SAM3.1 F32 GGML smoke | 3 | `615.827 ms` frame0+mask init | `366.117 ms` | `outputs/sam31-speed-smoke-f32-20260610/summary.json` |
| Official Python BF16+TF32 | 3 | `616.703 ms` frame0 cache + `42.537 ms` add-mask | `29.074 ms` propagate | `outputs/sam31-speed-python-bf16-20260610/summary.json` |

This means C++ has not yet beaten official Python for SAM3.1 tracking. The
largest speed gap is propagation, not initial ViT patch embedding. The next
optimization target should be the SAM3.1 propagation/memory-attention path and
the host-staged interactive feature/mask path, after adding per-stage timing to
the C++ smoke so its frame0+mask-init total is split like the Python summary.

## 2026-06-10 SAM3.1 Tracking Timing Split

The C++ SAM3.1 tracking smoke now reports timings on the same comparison axes
as the official Python sequence script:

- `encode_frame0_ms`: frame 0 visual encode.
- `add_detection_ms`: memory encode plus interactive mask/object-pointer init.
- `encode_frame1_ms`: explicit tracker-neck encode for frame 1.
- `propagate_encoded_ms`: propagation on an already encoded frame.
- `propagate_ms`: `encode_frame1_ms + propagate_encoded_ms`, kept for the old
  total.

This matters because the old C++ `propagate_ms` included frame 1 encoding,
while Python reports `frame1_cache_ms` and `propagate_ms` separately. The C++
smoke now calls `sam3_encode_image_for_tracking()` and then
`sam3_propagate_encoded_frame()` directly.

Visual tracker construction also prewarms the tracker PE/RoPE CPU cache. This
removes the first-propagation `sam31_prop_prepare_caches` cost from the measured
encoded propagate path: it was about `14.6 ms` in the BF16 profile and is now
about `0.001 ms`.

Current post-split timings:

| Path | Runs | `encode_frame0_ms` | `add_detection_ms` | `encode_frame1_ms` | `propagate_encoded_ms` | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C++ SAM3.1 BF16 GGML, prewarm | 3 | `431.528 ms` | `244.533 ms` | `212.308 ms` | `65.135 ms` | `outputs/sam31-speed-smoke-bf16-prewarm-20260610/summary.json` |
| C++ SAM3.1 F32 GGML, prewarm | 3 | `432.636 ms` | `181.060 ms` | `295.637 ms` | `51.963 ms` | `outputs/sam31-speed-smoke-f32-prewarm-20260610/summary.json` |
| Official Python BF16+TF32 | 3 | `616.703 ms` frame0 cache | `42.537 ms` | `0.448 ms` frame1 cache | `29.074 ms` | `outputs/sam31-speed-python-bf16-20260610/summary.json` |

The C++ propagate comparison point is now `propagate_encoded_ms`, not the older
aggregate `propagate_ms`. C++ still does not beat official Python at the same
BF16+TF32 contract: the encoded propagate path is about `65.1 ms` versus
Python's `29.1 ms`. CUDA node profiling shows the remaining propagation cost is
dominated by SAM3.1 multiplex memory attention, especially the four self/cross
`FLASH_ATTN_EXT` pairs. The frame0 add-mask path is dominated by CPU mask
preparation and the interactive mask head, so it is a separate optimization
track.

An attempted BF16 activation path exposed missing ggml CUDA BF16 coverage:
`CONCAT` had no BF16 branch and fused BF16 add still aborts. `CONCAT` now has a
BF16 CUDA implementation, but the SAM3.1 runtime path stays on the previous F32
activation behavior until the remaining fused BF16 operations are implemented
and proven faster with parity.

## 2026-06-10 SAM3.1 Root Optimization Follow-Up

The missing ggml CUDA BF16 coverage for the SAM3.1 memory-attention experiment
has been extended beyond `CONCAT`: fused BF16 binary broadcast now supports the
SAM3.1 `BF16+BF16 -> BF16` and `BF16+F32 -> BF16` cases. This allows
`SAM31_BF16_MEM_ATTN_ACTIVATION=1` to run without disabling CUDA fusion.

The measured result is negative for the default path. Against the default F32
memory-attention activation row, BF16 activation made encoded propagation
slower:

| Path | Runs | `encode_frame0_ms` | `add_detection_ms` | `encode_frame1_ms` | `propagate_encoded_ms` | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C++ SAM3.1 BF16 GGML, default mem-attn activation | 3 | `427.606 ms` | `243.577 ms` | `212.214 ms` | `64.186 ms` | `outputs/sam31-speed-smoke-bf16-default-memattn-act-r3-20260610/summary.json` |
| C++ SAM3.1 BF16 GGML, BF16 mem-attn activation | 3 | `425.061 ms` | `245.098 ms` | `212.698 ms` | `67.621 ms` | `outputs/sam31-speed-smoke-bf16-bf16act-memattn-act-r3-20260610/summary.json` |

BF16 memory-attention activation therefore remains an opt-in diagnostic, not a
default optimization.

The propagation graph now names the SAM3.1 memory-attention internal nodes.
With `SAM3_PROFILE=1` and `GGML_CUDA_PROFILE_NODES=1`, the eight named
self/cross memory-attention `FLASH_ATTN_EXT` nodes total about `29 ms` in the
single-run node profile, with each layer's self/cross attention around
`3.5-3.9 ms`. This is already roughly the same scale as the entire official
Python encoded propagation row, so the next propagation work should target
ggml CUDA FlashAttention/RoPE layout and launch overhead instead of another
application-level precision toggle.

The mask-init path had a separate, larger host-side issue. The previous C++
SAM3.1 memory-encode input path built a dense 32-channel mux mask on the CPU:
for the current 320x240 smoke this is a `1152 x 1152 x 32` F32 tensor, about
`169 MB`, even though only the valid object plane and condition plane are
non-zero. Fine-grained profiling showed:

| Stage | Before | After | Evidence |
| --- | ---: | ---: | --- |
| `sam31_mem_mask_prepare` | `67.074 ms` | `5.662 ms` | `outputs/sam31-speed-smoke-bf16-mask-detail-20260610/profile-summary.json`, `outputs/sam31-speed-smoke-bf16-mask-plane-upload-20260610/profile-summary.json` |
| Dense CPU mux fill | `61.358 ms` | eliminated in normal path | same |
| Mask tensor upload path | `12.209 ms` full upload | `2.660 ms` GPU zero + `2.353 ms` two-plane upload | same |

The optimized path zeroes the backend tensor with `ggml_backend_tensor_memset`
and uploads only the valid mask plane plus the condition plane. Dense mux
construction is preserved only for debug dump environments. The saved smoke
outputs are unchanged: `input_mask.pgm` and `frame1_mask.pgm` SHA-256 hashes
match the previous default run, and `score` / `obj_score_logit` are identical.

Profile-free repeat timing shows the practical add-mask improvement:

| Path | Runs | `encode_frame0_ms` | `add_detection_ms` | `frame0_ms` | `encode_frame1_ms` | `propagate_encoded_ms` | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C++ SAM3.1 BF16 GGML, previous default | 3 | `427.606 ms` | `243.577 ms` | `671.183 ms` | `212.214 ms` | `64.186 ms` | `outputs/sam31-speed-smoke-bf16-default-memattn-act-r3-20260610/summary.json` |
| C++ SAM3.1 BF16 GGML, two-plane mask upload | 5 | `429.843 ms` | `172.302 ms` | `602.144 ms` | `214.306 ms` | `66.315 ms` | `outputs/sam31-speed-smoke-bf16-plane-upload-r5-20260610/summary.json` |

This is an accepted local speedup for the SAM3.1 mask-init smoke: add-detection
drops by about `29.3%` while the checked smoke output remains unchanged. It
does not solve the official Python gap by itself. The remaining large gaps are:

- encoded propagation: C++ BF16 is still about `64-66 ms` versus official Python
  BF16+TF32 at about `29 ms`;
- add-mask: after removing dense mux construction, C++ still spends about
  `80 ms` in the interactive mask head and still stages some intermediate
  features through host memory;
- frame-1 caching: C++ still performs an explicit tracker-neck encode for the
  second frame, while the Python smoke's measured frame1 cache path is much
  smaller on this synthetic sequence.

## 2026-06-10 SAM3.1 Head32 SDPA Feasibility

The next propagation bottleneck is now specifically the SAM3.1 memory-attention
head32 SDPA shape. A direct attempt to route head32 through the existing ggml
CUDA MMA FATTN template was rejected before measurement: the current
`fattn-mma-f16.cuh` template assumes tile sizes that make the head32
instantiations compile to zero-sized internal arrays. The build was restored to
the existing TILE path, and this is not an accepted change.

The cuDNN SDPA feasibility bench now accepts separate `--q-seq` and `--kv-seq`
arguments so it can measure the actual SAM3.1 memory-attention shapes:

| Shape | cuDNN dtype | Mean | Evidence |
| --- | --- | ---: | --- |
| self attention, `q=5184`, `kv=5184`, `heads=8`, `head_dim=32` | BF16 | `1.001428 ms` | `outputs/cudnn-sdpa-sam31-head32-20260610/summary.json` |
| cross attention, `q=5184`, `kv=5200`, `heads=8`, `head_dim=32` | BF16 | `1.153695 ms` | same |
| self attention, F32 input converted to BF16 in the bench | BF16 | `1.019860 ms` | same |
| cross attention, F32 input converted to BF16 in the bench | BF16 | `1.174260 ms` | same |
| self attention | FP16 | `1.017486 ms` | same |
| cross attention | FP16 | `1.154189 ms` | same |

The equivalent ggml CUDA node profile for `sam31_propagate_single` reports the
eight named memory-attention `FLASH_ATTN_EXT` nodes at about `29.9 ms` total,
or roughly `3.5-4.0 ms` per self/cross call. Replacing those calls with cuDNN
BF16 SDPA therefore has a realistic local target around `8.6-9.4 ms` for the
eight calls, before graph integration overhead. That is large enough to move
encoded propagation materially toward the official Python `29 ms` row.

The fp32 cuDNN bench rows are not usable yet: both self and cross fp32 runs
failed with an illegal memory access in this local cuDNN path. That does not
block the same-contract Python comparison, because the official SAM3.1 Python
row uses BF16 autocast with TF32 enabled. The next implementation step should
be an opt-in ggml CUDA cuDNN SDPA path for unmasked head32 BF16/FP16 Q/K/V with
the existing ggml output layout `[head_dim, heads, q_seq, batch]`, followed by
SAM3.1 propagation parity and repeat timing.

## 2026-06-10 CUDA Cold-Path Reduction

Two CUDA integration experiments were run after the head32 SDPA feasibility
work.

First, ggml CUDA now has an opt-in cuDNN head32 SDPA bridge guarded by
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32=1`. For correctness, F32 inputs are not
converted to BF16 unless the additional diagnostic flag
`GGML_CUDA_CUDNN_SDPA_ALLOW_F32_TO_BF16=1` is set. The F32 cuDNN path still
fails locally with CUDA illegal-memory-access errors, so the default path keeps
F32 attention on the existing ggml CUDA implementation. With only
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32=1`, the SAM3.1 smoke score and
`obj_score_logit` match the default run exactly.

The BF16 memory-attention cuDNN experiment is rejected for now. Enabling
`SAM31_BF16_MEM_ATTN_ACTIVATION=1`,
`SAM31_MEM_ATTN_CONTIGUOUS_V=1`, and
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32=1` routes the eight large SAM3.1 memory
attention calls through cuDNN, but this local graph is slower and changes the
numeric output:

| Path | `propagate_encoded_ms` | `score` | `obj_score_logit` | Evidence |
| --- | ---: | ---: | ---: | --- |
| Default F32 memory attention, tiny kernel enabled | `49.219 ms` mean, 5 runs | `1.079934359` | `1.685746431` | `outputs/sam31-tiny-kernel-ab-20260610/tiny-*/summary.json` |
| BF16 mem-attn + contiguous V + cuDNN | `97.845 ms`, 1 profiled run | `1.176482677` | `0.681378901` | `outputs/sam31-profile-current-20260610/bf16-mem-cudnn-vcont/summary.json` |

Second, the accepted optimization is a tiny fused BF16-weight/F32-activation
bias kernel for the final SAM3.1 mask-score heads. The previous cublasLt bias
fusion path spent almost all of its time in first-use host setup for very small
outputs: `sam31_mux_iou` was about `3.17 ms`, and `sam31_mux_obj_score` was
about `17.36 ms` in the profiled cold path. The tiny kernel preserves the same
BF16 input rounding contract and avoids the cublasLt heuristic for
`ne0 <= 4`, `cols <= 32`, `src0=BF16`, `src1=F32`, `dst=F32`, non-GELU bias
matmuls. It can be disabled with
`GGML_CUDA_DISABLE_TINY_BF16_F32_BIAS_KERNEL=1`.

Cold-process A/B timing on the same 320x240 SAM3.1 tracking smoke:

| Path | Runs | `propagate_encoded_ms` mean | `propagate_ms` mean | `score` / `obj_score_logit` |
| --- | ---: | ---: | ---: | --- |
| Tiny kernel disabled | 5 | `65.721 ms` | `280.242 ms` | `1.079934359` / `1.685746431` |
| Tiny kernel enabled | 5 | `49.219 ms` | `263.619 ms` | `1.079934359` / `1.685746431` |

This is an accepted propagation cold-path speedup: encoded propagation improves
by about `25.1%` (`1.335x`) with the checked smoke output unchanged. It does
not close the full official Python gap. The next high-value work remains either
same-contract BF16/TF32 memory-attention integration with parity, or a
shape-aware warmup/cache strategy for the remaining cublasLt cold shapes such
as the interactive mask head's first `128 x 7` projection.

That interactive cold shape has now been reduced without changing the smoke
output. `sam3_create_visual_tracker()` runs a CUDA-only SAM3.1 warmup for the
first token-to-image cross-attention `128 x 7` projection and the object-score
MLP. This moves the cublasLt first-use cost out of `add_detection`, matching the
warm execution contract used by the official Python comparison. The warmup can
be disabled with `SAM31_DISABLE_INTERACTIVE_WARMUP=1`.

The same pass also removed a real transfer bottleneck in
`sam31_compute_interactive_mask_obj_ptr`: tracker detector-neck features were
being staged GPU -> CPU -> GPU before the interactive mask head. F32-to-F32
staging now uses backend tensor copy, with CPU conversion kept only for
non-F32 fallback sources. In a profiled smoke this replaced three transfers of
about `5.3 MiB`, `84.9 MiB`, and `21.2 MiB` with backend copies reported as
`0.013 ms`, `0.003 ms`, and `0.003 ms`. The interactive mask-head graph itself
now profiles at about `7 ms`; the remaining add/init time is mostly mask-memory
preparation/encoding and small CPU-side object-pointer extraction.

Cold-process A/B timing on the same 320x240 SAM3.1 tracking smoke after the
backend-copy change:

| Path | Runs | `add_detection_ms` mean | `propagate_encoded_ms` mean | `score` / `obj_score_logit` |
| --- | ---: | ---: | ---: | --- |
| Backend copy, warmup disabled | 5 | `127.764 ms` | `49.143 ms` | `1.079934` / `1.685746` |
| Backend copy, interactive warmup enabled | 5 | `46.834 ms` | `49.021 ms` | `1.079934` / `1.685746` |

Evidence is in `outputs/sam31-warmup-copy-ab-20260610.json` and
`outputs/sam31-profile-nodes-warmup-copy-20260610.log`. Relative to the earlier
pre-warmup/pre-copy baseline (`175.399 ms` mean from
`outputs/sam31-warmup-min-ab-20260610.json` before the backend-copy patch),
`add_detection_ms` improved by about `73.3%` (`3.75x`). Relative to backend copy
without the interactive warmup, the warmup path is about `2.73x` faster. The
checked score and object logit are unchanged across all ten final A/B runs.

The same synthetic mask-sequence official Python comparison was regenerated
after this change. Python BF16+TF32 with zero warmup reports
`propagate_ms=23.076 ms` and `add_mask_ms=38.262 ms`
(`outputs/sam31-mask-sequence-python-tiny-20260610/summary.json`). The best
current C++ BF16 smoke row remains about `49.2 ms` for encoded propagation, so
C++ is still about `2.1x` slower on the propagation part of this SAM3.1
contract even after the accepted tiny-kernel fix.

The just default for `SAM31_TRACKING_MASK_INIT_MODEL` has since been corrected
to `models/sam3.1/sam3.1_multiplex-bf16.ggml`, matching the Python comparison
default `SAM31_MASK_SEQUENCE_PY_DTYPE=bf16` and TF32-on policy. The regenerated
same-precision comparison is:

| Path | `encode_frame0_ms` | `add/init ms` | `encode_frame1_ms` | `propagate ms` | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| C++ BF16 GGML, warmup + backend copy | `423.179` mean | `46.834` add-detection mean | `214.780` mean | `49.021` encoded mean, `263.802` total mean | `score=1.079934`, `obj_score_logit=1.685746` |
| Official Python BF16+TF32 | `610.682` frame0 cache | `38.838` add-mask | `0.450` frame1 cache | `23.378` | `obj_score_logit=2.890625` |

Evidence is in
`outputs/sam31-mask-sequence-python-bf16-default-20260610/summary.json` and
`outputs/sam31-warmup-copy-ab-20260610.json`. This is the current comparison
baseline: C++ is still about `2.10x` slower for encoded propagation and about
`1.21x` slower for add-mask/add-detection. End-to-end parity is still not
acceptable (`mask_iou=0.904224`, `xor_pixels=1483` from the last state compare),
so the next root-cause work should prioritize the Python/C++ mask-init state
divergence before treating the remaining CUDA gaps as pure performance work.

Two follow-up probes were rejected:

| Probe | Result | Reason |
| --- | --- | --- |
| cublasLt bias algo cache | 353 cache hits / 23 misses in a profiled smoke, but profile-free A/B was noise-level (`48.44 ms` disabled vs `48.64 ms` enabled for `propagate_encoded_ms`) | no measurable speedup for the current graph |
| F32/TF32 tiny bias kernel for interactive mask-head small projections | first `128 x 7` projection moved from cublasLt to a tiny kernel, but the cublasLt cold setup shifted to the next `128 x 5184` projection and score/logit changed | not parity-safe and not end-to-end faster |

## 2026-06-10 SAM3.1 Input and Memory Parity Fix

The previous SAM3.1 mask-init state comparison still mixed two separate
problems: the C++ image resize path did not reproduce the official Python PIL
input contract, and the SAM3.1 multiplex memory state did not add the empty
`no_obj_embed_spatial` slot embeddings for every absent multiplex slot.

Both have now been corrected in the C++ path:

- SAM3/SAM3.1 preprocessing now follows the official list-PIL input contract:
  RGB image, default PIL bicubic resize, uint8 two-pass rounding, then float16
  storage and float16 normalization before the model input tensor is materialized.
- SAM2 preprocessing remains on the previous ImageNet-normalized path and does
  not use the SAM3/SAM3.1 float16 storage emulation.
- SAM3.1 memory encoding now fills only the active object and condition planes
  on the backend tensor in the normal path, but adds `no_obj_embed_spatial` for
  all inactive multiplex slots after the memory backbone output is read. Dense
  mux-mask construction is still available for dump/debug environments.

An intermediate experiment with an empirical bicubic coefficient `a=-0.3`
produced a slightly higher frame-1 IoU (`0.9650476318261103`), but it did not
match the official PIL resize contract. It is therefore rejected as a parity
foundation. The accepted path uses the Pillow bicubic coefficient `a=-0.5` and
the same 8-bit two-pass rounding behavior as PIL.

The latest state comparison after this change:

| Tensor | Status | Max abs | Mean abs | Evidence |
| --- | --- | ---: | ---: | --- |
| `input_image_preprocessed` | ok | `0.0` | `0.0` | `outputs/sam31-mask-init-state-compare-pillowresize-20260610/summary.json` |
| `vit_block_00_input` | ok | `0.03360795974731445` | `5.797452676768133e-07` | same |
| `image_features` | ok | `0.4659088235348463` | `0.008727441895395842` | same |
| `memory_mux_mask_input` | ok | `0.0` | `0.0` | `outputs/sam31-mask-init-state-compare-pillowresize-20260610/key_summary.json` |
| `memory_mux_mask_interpolated` | ok | `0.0` | `0.0` | same |
| `object_score_logits` | ok | `0.0` | `0.0` | same |
| `obj_ptr` | ok under relaxed state tolerance | `0.12571450509130955` | `0.0024483503716510313` | same |
| `stored_spatial_feats` | failed max-only | `0.7977794408798218` | `0.0070697918587510015` | `outputs/sam31-mask-init-state-compare-pillowresize-20260610/summary.json` |

This is a large correction from the previous accepted comparison: image input is
now exact, `image_features` mean absolute error dropped from about `0.06034` to
`0.00873`, and `stored_spatial_feats` mean absolute error dropped from about
`0.04471` to `0.00707`. Full state parity is still not accepted because
`stored_spatial_feats` has a max outlier above the current `0.5` tolerance, and
late ViT block dumps still show large local max differences despite low means.

The regenerated same-sequence SAM3.1 comparison is:

| Path | Result | Evidence |
| --- | ---: | --- |
| C++ BF16 GGML | `encode_frame0_ms=443.748762`, `add_detection_ms=53.968958`, `encode_frame1_ms=226.314179`, `propagate_encoded_ms=48.471639` | `outputs/sam31-pillowresize-sequence-cpp-20260610/summary.json` |
| Official Python BF16+TF32 | `frame0_cache_ms=602.4797380669042`, `add_mask_ms=38.80041802767664`, `frame1_cache_ms=0.4503269447013736`, `propagate_ms=23.984011029824615` | `outputs/sam31-pillowresize-sequence-python-20260610/summary.json` |
| C++ vs Python frame-1 mask | IoU `0.9629355077835434`, xor `550` pixels | `outputs/sam31-pillowresize-sequence-python-20260610/summary.json` |

The speed conclusion is unchanged: C++ is faster than official Python for the
first visual encode on this smoke, but still slower for add-mask and encoded
propagation. The next correctness target is no longer preprocessing; it is the
remaining SAM3.1 memory/object-pointer path and the late-block numerical drift
that feeds the mask decoder. The next performance target remains the encoded
propagation path, especially memory-attention/FlashAttention layout and launch
overhead.

## 2026-06-10 SAM3.1 ViT Residual and FATTN32 MMA Update

The late ViT state divergence was narrowed further with per-block and staged
dumps. When individual C++ block stages are fed Python tensors, the tested
block 15/23/31 stages remain close: all sampled stage outputs stayed below a
`0.139` max absolute difference and had no `>0.5` outliers. In the full C++
path, the first `>0.5` block-output outliers appear at block 9 and then amplify
through later ViT blocks. Disabling BF16 linear inputs did not change the final
state diffs. Current interpretation: the remaining large local outliers are
accumulated precision/layout-kernel drift, not a single wrong stage or the
preprocess path.

The current CUDA profile after the input/memory fixes shows the SAM3.1 encoded
propagation bottleneck is the head-dim 32 memory-attention path:

| Item | Measurement | Evidence |
| --- | ---: | --- |
| `sam31_propagate_single` graph compute | `42.306 ms` | `outputs/sam31-profile-current-20260610c/profile-summary.json` |
| 8 memory-attention `FLASH_ATTN_EXT` nodes | about `29.2 ms` total | `outputs/sam31-profile-nodes-current-20260610c/summary.log` |
| default head32 FATTN node | about `3.6-3.7 ms` per long node | `outputs/sam31-profile-fattn-current-20260610c/summary.log` |

A D32 MMA FATTN path was added to ggml CUDA and made selectable with
`GGML_CUDA_ENABLE_FATTN32_MMA`. The first accepted policy used D32 MMA for
SAM3.1 memory cross-attention nodes (`*_ca_fattn`) and only one
self-attention node, `sam31_mem_attn_layer3_sa_fattn`. That was the best
measured default on the initial 320x240 mask-init smoke, but it has since been
superseded by the tensor-level analysis later in this document: the current
default uses D32 MMA for named SAM3.1 memory-attention self-attention and
cross-attention nodes. Set `GGML_CUDA_ENABLE_FATTN32_MMA=0` to force the old
D32 tile path for A/B checks.

| Mode | `propagate_encoded_ms` | Python mask IoU | xor pixels | Status |
| --- | ---: | ---: | ---: | --- |
| Previous default tile FATTN | `48.836` | `0.9629355077835434` | `550` | superseded |
| CA-only D32 MMA | `36.067` | `0.9635026647777103` | `541` | superseded |
| New default, CA + layer3 SA D32 MMA | `33.063` | `0.9655777537796977` | `510` | accepted current default |
| `GGML_CUDA_ENABLE_FATTN32_MMA=all` | `23.705` | `0.9586346738763095` | `612` | opt-in speed experiment |

Evidence for the accepted default is in
`outputs/sam31-fattn32-ca-sa3-default-compare-20260610d.json`. The layer probe
that selected it is in `outputs/sam31-fattn32-layer-compare-20260610d.json`:

| Mode | `propagate_encoded_ms` | Python mask IoU | xor pixels | Score |
| --- | ---: | ---: | ---: | ---: |
| CA-only default | `36.724` | `0.9635026647777103` | `541` | `0.961734` |
| CA + layer0 SA | `32.425` | `0.9595871002563756` | `599` | `0.937699` |
| CA + layer1 SA | `33.606` | `0.963235791038724` | `544` | `0.953417` |
| CA + layer2 SA | `32.458` | `0.9609390811576604` | `579` | `0.957366` |
| CA + layer3 SA | `33.045` | `0.9655777537796977` | `510` | `0.969601` |
| All D32 MMA | `23.649` | `0.9586346738763095` | `612` | `0.933125` |

Repeated measurements show the current state more clearly:

| Path | Runs | Median encoded propagation | Mean encoded propagation | Mean Python-mask IoU |
| --- | ---: | ---: | ---: | ---: |
| C++ new default, CA + layer3 SA | `7` | `33.169 ms` | `32.846 ms` | `0.9655777537796977` |
| C++ opt-in all D32 MMA | `7` | `23.374 ms` | `23.530 ms` | `0.9586346738763095` |
| Official Python BF16+TF32, warmup 1 | `5` | `16.616 ms` | `16.615 ms` | reference |

Evidence is in `outputs/sam31-fattn32-repeat-cpp-20260610d/summary.json` and
`outputs/sam31-fattn32-repeat-python-20260610d/summary.json`. The accepted
default is therefore a partial root optimization, not completion of the goal:
encoded propagation is about `32.7%` faster than the previous tile default on
this smoke while mask comparison is slightly better, but even the lower-parity
`all` mode is still slower than warm official Python for encoded propagation.
The next root optimization needs to reduce memory-attention launch/layout
overhead beyond per-node D32 MMA dispatch, likely by fusing the SAM3.1 memory
attention subgraph or by matching the official PyTorch/SDPA execution shape more
closely.

The D32 MMA implementation currently increases a cold `fattn.cu` rebuild by
about five minutes on this machine. That is acceptable for proving the kernel
direction, but it is not ideal as a long-term integration shape. If the
self-attention MMA path is pursued further, the next implementation task should
also reduce CUDA build cost, for example by narrowing D32 instantiations to the
actually used SAM3.1 shapes or splitting the D32 dispatch away from the large
`fattn.cu` translation unit.

## 2026-06-10 SAM3.1 Mask Hypernetwork Batching Update

The multiplex propagation mask decoder was still running the mask hypernetwork
one token at a time: 48 separate hypernetwork paths and 48 separate
`sam31_mux_upscaled @ hyper` matmuls for the 16 multiplex slots and 3 masks per
slot. This was replaced with 3 grouped paths, one per mask index, each operating
on all 16 slots at once. The output is reshaped back to the original
`[pixels, mask_index + 3 * slot, bucket]` order, so the accepted default mask is
bit-identical to the previous default output on the smoke case.

| Path | Runs | Median encoded propagation | Mean encoded propagation | Python mask IoU | Mask SHA status |
| --- | ---: | ---: | ---: | ---: | --- |
| Previous accepted default, CA + layer3 SA | `7` | `33.169 ms` | `32.846 ms` | `0.9655777537796977` | stable |
| Batched hypernetwork default | `7` | `30.406 ms` | `30.606 ms` | `0.9655777537796977` | unchanged |
| Previous opt-in all D32 MMA | `7` | `23.374 ms` | `23.530 ms` | `0.9586346738763095` | lower parity |
| Batched hypernetwork + opt-in all D32 MMA | `7` | `21.116 ms` | `21.109 ms` | `0.9585079064738479` | lower parity |
| Official Python BF16+TF32, warmup 1 | `5` | `16.616 ms` | `16.615 ms` | reference | reference |

Evidence is in `outputs/sam31-mask-hyper-batched-repeat-20260610d/summary.json`
and `outputs/sam31-fattn32-repeat-python-20260610d/summary.json`. Relative to
the original tile FATTN baseline, the accepted default is now about `37.3%`
faster for encoded propagation on this smoke (`48.836 ms` to `30.606 ms`) while
keeping the best measured mask IoU and the same mask hash as the pre-batching
default.

The new profile also changes the remaining bottleneck picture. In opt-in `all`
mode, the propagation graph compute is now `16.407 ms`, roughly the same range
as the official Python warm propagation timing, but end-to-end C++
`propagate_encoded_ms` is still about `21 ms`. The extra time is mostly around
the graph boundary: memory preparation (`1.843 ms`) and input upload
(`1.819 ms`) are the largest non-compute spans in
`outputs/sam31-mask-hyper-batched-profile-20260610d/cpu_spans.json`.

Two attempted knobs are rejected for the current default:

| Probe | Result | Decision |
| --- | --- | --- |
| `SAM3_CUDA_ENABLE_GRAPHS=1` | One-frame propagation became slower because graph capture cost is paid in the current per-call graph build path | reject until graph reuse is implemented |
| `SAM31_BF16_MEM_ATTN_ACTIVATION=1` | Default parity worsened (`0.9649679379007763` IoU vs `0.9655777537796977`) and speed did not improve; `all` also lost parity | reject |

The next root optimization is therefore not another mask-head launch reduction.
For same-parity default speed, the hard remaining work is the self-attention D32
MMA parity problem: layers 0-2 are fast but move the mask too much, while layer
3 is safe on this smoke. For raw speed, C++ is now close inside the captured
graph compute range, so the next end-to-end task is to reuse or preallocate the
SAM3.1 propagation graph inputs and reduce the memory-prepare/upload boundary.

## 2026-06-11 SAM3.1 FATTN32 Search and RoPE Cache Update

The D32 MMA self-attention search was extended beyond single-layer probes. The
default policy remains `CA + layer3 SA` because it is still the best observed
same-smoke mask parity point. A faster Pareto point exists with
`CA + layer1 SA + layer3 SA`, but it loses mask IoU and is therefore not a
default candidate under the current same-parity goal. The selection helper now
also accepts
`GGML_CUDA_ENABLE_FATTN32_MMA=sam31-ca+matches:<comma-separated-substrings>` so
specific SAM3.1 FATTN nodes can be tested without adding a new hard-coded mode.

Single-run search evidence is in
`outputs/sam31-fattn32-sa-combos-20260611a/summary.json`:

| Mode | Encoded propagation | Python mask IoU | xor pixels | Score |
| --- | ---: | ---: | ---: | ---: |
| `CA + layer3 SA` | `29.676 ms` | `0.9655777537796977` | `510` | `0.969601` |
| `CA + layer1 SA + layer3 SA` | `26.992 ms` | `0.9647034958415038` | `522` | `0.963196` |
| `CA only` | `34.616 ms` | `0.9635026647777103` | `541` | `0.961734` |
| `CA + layer1 SA` | `31.029 ms` | `0.9633083316455301` | `543` | `0.953417` |
| `all` | `21.164 ms` | `0.9584347119298753` | `615` | `0.933125` |

A direct tile-versus-MMA memory-attention slice check did not show a single
obviously broken self-attention layer. On a small f16 H=16 case, enabling any
single self-attention layer with CA already on changed the final
memory-attention output by about `0.0074-0.0082` max absolute and about
`0.0007` mean absolute against the tile baseline; `all` was about `0.0093` max
absolute and `0.0008` mean absolute. Current interpretation: the lower mask IoU
from layers 0-2 is accumulated Tensor Core precision sensitivity in the
propagation stack, not an obvious kernel-ordering or shape bug. Evidence is in
`outputs/sam31-memattn-fattn32-diff-20260611a/tile_vs_modes_summary.json`.

The C++ SAM3.1 propagation path also stopped recomputing and re-uploading fixed
PE/RoPE tensors for every propagation call. `sam3_tracker` now caches the
`[2, head_dim/2, H*H]` RoPE layout used by SAM3.1 memory attention, separate
from the older `[2, D/2, H*H]` tracker cache. It also pre-allocates backend
cache tensors for the fixed source position and per-head Q RoPE tensors so the
propagation graph can reference them directly. This preserves the accepted mask
hash, removes the `sam31_prop_rope_q_build` span from the profile, and cuts the
profiled propagation input upload from `1.915 ms` to `0.862 ms`.
The remaining propagation input uploads now use the existing async tensor-set
wrapper; this is a small repeat-measured improvement, not a structural fix for
the memory boundary.

Repeated 320x240 smoke timing after the RoPE/backend PE cache and async input
set:

| Mode | Runs | Mean encoded propagation | Median | Python mask IoU | xor pixels | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default `CA + layer3 SA` | `7` | `29.597 ms` | `29.099 ms` | `0.9655777537796977` | `510` | accepted current default |
| `CA + layer1 SA + layer3 SA` | `7` | `25.707 ms` | `25.456 ms` | `0.9647034958415038` | `522` | faster lower-parity option |
| `all` | `7` | `19.376 ms` | `19.347 ms` | `0.9585079064738479` | `614` | opt-in speed experiment |
| Official Python BF16+TF32 | `5` | `16.615 ms` | `16.616 ms` | reference | reference | reference |

Evidence is in `outputs/sam31-prop-async-set-repeat-20260611a/summary.json`,
`outputs/sam31-prop-async-set-ca-sa1-sa3-repeat-20260611a/summary.json`,
`outputs/sam31-rope-cache-python-compare-20260611a/summary.json`, and
`outputs/sam31-fattn32-repeat-python-20260610d/summary.json`. Relative to the
pre-cache batched default, the accepted default is roughly unchanged by mean
because one repeat was an outlier (`30.606 ms` to `29.597 ms`) but improves by
median (`30.406 ms` to `29.099 ms`). The lower-parity `all` mode moved from
`21.109 ms` to `19.376 ms` mean encoded propagation.

The current profile in `all` mode shows `sam31_propagate_single` graph compute
at about `15 ms`, but encoded propagation is still about `19-20 ms`. The
largest remaining non-compute cost is still the memory/input boundary, and the
profile numbers are noisy at this sub-millisecond scale
(`outputs/sam31-prop-async-set-profile-20260611a/summary.json`). A rejected
follow-up that precomputed condition-frame temporal position at slot-storage
time worsened `sam31_prop_memory_prepare`, so the next real boundary fix should
not add another CPU-side copy; it should keep prepared memory inputs resident on
the GPU or precompute the cross-attention memory K/V projections.
This means the raw CUDA graph compute can be close to official Python only in
the lower-parity mode; the accepted same-parity default still needs either a
higher-parity D32 MMA self-attention implementation or a more structural change
that keeps SAM3.1 memory inputs resident on the GPU and reuses the propagation
graph boundary.

## 2026-06-11 SAM3.1 Memory K/V Cache Update

The next accepted structural propagation change moves the spatial-memory
cross-attention K/V projections out of `sam31_propagate_single`. For a
condition-frame memory slot, C++ now precomputes each SAM3.1 memory-attention
layer's spatial cross-attention K and V tensors immediately after memory encode,
without baking in temporal-position encoding. Propagation then adds the
per-frame temporal-position vector to cached K, concatenates only the dynamic
object-pointer tail, and reuses the resident spatial K/V tensors. Set
`SAM31_DISABLE_MEM_CA_KV_CACHE=1` to force the old per-propagation K/V
projection path.

The first version of this cache baked the condition-frame temporal position into
the stored K tensor. That was superseded because it could not represent the
official SAM3.1 v2 `maskmem_tpos_enc[num_maskmem - t_pos - 1]` contract for a
later propagation frame. The current implementation stores tpos-free K/V and
adds the official row at propagation time.

The precompute cost is paid during add-detection/memory-encode, not during
encoded propagation; the profiled precompute graph was about `0.55 ms` on the
first smoke. The table below is retained as historical evidence for the earlier
kernel-selection state; the 2026-06-11 update near the top of this file is the
current default-K/V-cache evidence.

Repeated timing:

| Mode | Runs | Mean encoded propagation | Median | Python mask IoU | xor pixels | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default `CA + layer3 SA`, K/V cache on | `7` | `25.424 ms` | `25.427 ms` | `0.9655777537796977` | `510` | historical default |
| Default `CA + layer3 SA`, K/V cache disabled | `7` | `29.214 ms` | `29.117 ms` | `0.9655777537796977` | `510` | old path |
| `CA + layer1 SA + layer3 SA`, K/V cache on | `7` | `21.825 ms` | `21.901 ms` | `0.9647034958415038` | `522` | faster lower-parity option |
| `all`, K/V cache on | `7` | `15.089 ms` | `15.253 ms` | `0.9585079064738479` | `614` | faster than Python, lower parity |
| Official Python BF16+TF32 | `5` | `16.615 ms` | `16.616 ms` | reference | reference | reference |

Evidence is in `outputs/sam31-memkv-cache-repeat-20260611c/summary.json` and
`outputs/sam31-fattn32-repeat-python-20260610d/summary.json`. The cache removes
the previous graph-boundary bottleneck: in the `all` profile,
`sam31_prop_memory_prepare` is `0.051 ms` and `sam31_prop_input_upload` is
`0.033 ms`, while `sam31_propagate_single` graph compute is `14.078 ms`
(`outputs/sam31-memkv-cache-profile-all-20260611c/summary.json`). In the
same-parity default profile, the propagation graph compute is still
`23.945 ms` (`outputs/sam31-memkv-cache-profile-default-20260611c/summary.json`).

The result changed the remaining root problem at that point: C++ could beat
official Python on raw encoded propagation only in the lower-parity `all` D32
MMA mode. The later tpos-free K/V cache fix and default enablement supersede
that conclusion for the small same-input propagation slice; the remaining root
problem is now full SAM3.1 mask parity and broader multi-case validation, not
whether the encoded propagation slice can beat Python.

## 2026-06-11 D32 Attention Kernel Experiments After K/V Cache

Several post-cache CUDA kernel alternatives were tested and rejected as default
paths:

| Candidate | Encoded propagation | Python mask IoU | xor pixels | Result |
| --- | ---: | ---: | ---: | --- |
| cuDNN head32, F32->BF16 I/O | `26.887 ms` | `0.0` | `14592` | rejected; empty mask |
| D32 tile `cols_per_block=64` | `24.408 ms` | `0.9615644420426912` | `569` | rejected; lower parity |
| D32 tile forced parallel-K `4` | `27.224 ms` | `0.9604809835844086` | `585` | rejected; slower/lower parity |
| D32 tile forced parallel-K `8` | `25.366 ms` | `0.9591781562584483` | `604` | rejected; lower parity |
| D32 tile forced parallel-K `16` | `26.217 ms` | `0.9580826178081265` | `620` | rejected; slower/lower parity |
| D32 tile full-tail OOB-check skip | `24.569 ms` | `0.9654380990954502` | `512` | rejected; tiny parity loss |
| D32 MMA all + float rescale | `14.776 ms` | `0.958375565916616` | `616` | rejected; no parity improvement |
| D32 tile `nbatch_fa=128` | `23.955 ms` | `0.9653077753779697` | `514` | rejected; small parity loss |
| D32 no-mask delayed mask-index calculation | profile-only | `0.9655777537796977` | `510` | rejected; kernel profile regressed |
| cuDNN head32 reject diagnostics | `24.413 ms` | `0.9655777537796977` | `510` | accepted diagnostic; long nodes reject on non-contiguous V |
| cuDNN head32 V pack + BF16 I/O | `81.385 ms` | `0.0` | `14592` | rejected; empty mask and slower |
| cuDNN head32 V pack + BF16 output fix | `82.128 ms` | `0.9583614979045559` | `616` | rejected; non-empty but slower/lower parity |
| cuDNN head32 safety gate | `24.734 ms` | `0.9655777537796977` | `510` | accepted; unsafe cuDNN run requires an extra env gate |
| Default after experiments | `24.687 ms` | `0.9655777537796977` | `510` | accepted baseline restored |

Evidence is in `outputs/sam31-cudnn-head32-smoke-20260611a`,
`outputs/sam31-fattn32-tile64-smoke-20260611d`,
`outputs/sam31-fattn-parallel-blocks-smoke-20260611d`,
`outputs/sam31-tile-tail-fullchunk-smoke-20260611e`,
`outputs/sam31-fattn32-all-float-rescale-smoke-20260611e`, and
`outputs/sam31-tile-d32-nbatch128-smoke-20260611f`. The baseline was restored
again after the `nbatch_fa=128` probe in
`outputs/sam31-default-restored-after-nbatch128-reject-20260611f`, with
bit-identical mask SHA
`0998b8836a71debafbaa4a3f531b96104d9f9d5d9fc1303785ba8a3eebf41f30`.
The delayed mask-index probe is in
`outputs/sam31-profile-tile-d32-nomask-mask-index-20260611g`; it kept the same
mask but regressed the three D32 self-attention profile rows from kernel sum
`10.7940 ms` to `11.4794 ms`, so it was reverted.

The cuDNN head32 diagnostic run in
`outputs/sam31-cudnn-head32-longdiag-20260611h` confirmed that the long
SAM3.1 memory-attention calls are excluded because V is non-contiguous:
Q/K/dst are contiguous for `Q=[32,5184,8,1]`, but V is not. A V-pack experiment
in `outputs/sam31-cudnn-head32-vpack-20260611h` routed all eight long head32
memory-attention calls through cuDNN, but it produced an empty mask
(`sha=e2cc2a1fa6131cf4d86faa3baf78851f35a36853e2467c257b3df9d89e85cce5`,
`sum=0`) and slowed encoded propagation to `81.385 ms`. The immediate empty-mask
bug was the cuDNN output buffer contract: the graph-level I/O dtype is BF16/F16,
so the bridge must receive BF16/F16 output and explicitly convert it back to
F32 while transposing to ggml layout. After that fix,
`outputs/sam31-cudnn-head32-outputdtype-vpack-20260611h` produced a non-empty
mask (`sha=b604d6d867ae22716e133d2ccf885e256c21a6c0d44e6051b53ab09105445fd4`,
`sum=14380`), but it was still slower (`82.128 ms`) and lower parity
(Python-mask IoU `0.9583614979045559`, `616` xor pixels), so the cuDNN path
remains rejected as a default.

The implementation now keeps this path behind
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32_UNSAFE_RUN` in addition to the normal cuDNN
opt-in env. With only the normal opt-in enabled,
`outputs/sam31-cudnn-head32-safe-disabled-20260611h` is bit-identical to the
current default C++ mask
(`sha=0998b8836a71debafbaa4a3f531b96104d9f9d5d9fc1303785ba8a3eebf41f30`,
`sum=14530`).

The important implementation finding is that the existing `launch_fattn`
stream-K scheduler cannot be directly enabled for the tile kernel. It reduced
the D32 self-attention kernel time sharply, but the tile kernel interprets
`blockIdx.x/y` as query tile and KV partition, while the stream-K scheduler uses
a different work mapping and fixup contract. The observed mask IoU dropped to
`0.5425195828104038`, so this is a correctness issue rather than an acceptable
precision tradeoff.

The next root optimization should therefore be one of:

- implement a tile-native stream-K/fixup mapping for no-mask D32
  self-attention;
- implement a dedicated no-mask D32 self-attention kernel with the current tile
  precision contract;
- make the D32 MMA path numerically closer to the accepted tile path for
  self-attention layers 0-2.

One very small no-risk cleanup was accepted after these rejected experiments:
the tile kernel no longer computes the ALiBi slope in compile-time no-mask
variants. The default 320x240 mask-init smoke remains bit-identical to the
accepted baseline (`0998b8836a71debafbaa4a3f531b96104d9f9d5d9fc1303785ba8a3eebf41f30`,
Python-mask IoU `0.9655777537796977`, `510` xor pixels). The D32 tile
self-attention profile rows changed from mean `3.7648 ms` / median `3.7604 ms`
to mean `3.7520 ms` / median `3.7961 ms` for the three
`Q=[32,5184,8,1]` nodes. This is not a material speed win; it is accepted as a
parity-safe cleanup that avoids unused work in the remaining bottleneck path.
Evidence is in `outputs/sam31-tile-nomask-slope-gate-smoke-20260611f` and
`outputs/sam31-profile-tile-slope-gate-20260611f`.

The D32 tile path now also instantiates the existing no-mask template when
`DKQ==32 && DV==32` and the FATTN node has no mask. This keeps the same tile
accumulation order and only removes the compile-time mask branch from SAM3.1
no-mask self-attention. It is parity-safe on the same smoke: mask SHA remains
`0998b8836a71debafbaa4a3f531b96104d9f9d5d9fc1303785ba8a3eebf41f30`, with
Python-mask IoU `0.9655777537796977` and `510` xor pixels
(`outputs/sam31-tile-d32-nomask-specialized-smoke-20260611f`).

The local kernel profile improved for the three
`Q=[32,5184,8,1] K=[32,5184,8,1] V=[32,5184,8,1]` D32 self-attention rows:
kernel sum `11.2560 ms -> 10.7940 ms` and total sum
`11.5426 ms -> 11.0681 ms`
(`outputs/sam31-profile-tile-d32-nomask-specialized-20260611f`). End-to-end
repeat timing is still noise-bound rather than a clear speed win: the 7-run
encoded propagation median moved `25.427 ms -> 25.286 ms`, while the mean moved
`25.424 ms -> 25.726 ms` because two runs were outliers
(`outputs/sam31-d32-nomask-specialized-repeat-20260611f/summary.json`).

## 2026-06-11 D32 MMA Parity/Speed Triage

The post-cache bottleneck is now specifically the D32 no-mask self-attention
inside SAM3.1 memory attention. The useful Pareto result is that enabling the
D32 MMA path for all four self-attention layers is fast enough to beat the
official Python warm propagation baseline for this smoke input, but the single
mask comparison is lower than the accepted default C++ mask. Keeping only layer
3 on MMA preserves the accepted mask best, but leaves the C++ path slower than
Python.

| D32 self-attn layers on MMA | Encoded propagation | Python mask IoU | xor pixels | C++ default IoU | C++ xor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0123` | `14.749 ms` | `0.9584347120843472` | `615` | `0.9885152327900419` | `167` |
| `123` | `18.103 ms` | `0.9622845555931058` | `558` | `0.9929829389102917` | `102` |
| `023` | `18.284 ms` | `0.9592870164067248` | `603` | `0.9918145549594167` | `119` |
| `012` | `18.463 ms` | `0.9562191743801095` | `648` | `0.9874802228795487` | `182` |
| `013` | `18.534 ms` | `0.9601892531260561` | `589` | `0.9896184255757993` | `151` |
| `13` | `21.237 ms` | `0.964703` | `522` | `0.994908` | `74` |
| `3` | `25.880 ms` | `0.965578` | `510` | `1.0` | `0` |
| none | `27.399 ms` | `0.963503` | `541` | `0.995945` | `59` |

Evidence is in `outputs/sam31-d32-combo-summary-20260611j.json`. The layer-0
choice is the key speed cliff. In the `123` profile, layer 0 remains on the
tile path and one `Q=[32,5184,8,1] K=[32,5184,8,1] V=[32,5184,8,1]` row takes
about `3.81 ms` total, while the same shape on MMA is roughly sub-millisecond
in the all-MMA profile. Any same-parity win over Python therefore needs either
a numerically acceptable fast layer-0 path or acceptance evidence that all-MMA
is valid across more than one mask.

Two scheduler probes did not change that conclusion. Forcing stream-K on the
`123` configuration improved encoded propagation to `17.949 ms`, with Python
mask IoU `0.9624298939117508` and `556` xor pixels
(`outputs/sam31-d32-sa123-force-streamk-20260611j`). The accepted default with
forced stream-K was still `25.075 ms`, with Python mask IoU
`0.9652402807775378` and `515` xor pixels
(`outputs/sam31-default-force-streamk-20260611j`). This is a small scheduling
improvement, not the root fix.

Two deeper probes were rejected and reverted:

- D32 tile `ncols=64` for the `123` configuration measured `18.628 ms`, with
  Python mask IoU `0.9604355471391857` and `585` xor pixels
  (`outputs/sam31-d32-sa123-tile64-20260611j`). It was both slower than forced
  stream-K and lower parity.
- The upstream CUDA PDL launch pattern was tested on the all-MMA path after a
  full CUDA rebuild. It did not improve runtime: three all-MMA runs were
  roughly `16.158`, `15.013`, `14.861 ms` with PDL enabled versus `14.699`,
  `14.981`, `15.679 ms` with PDL disabled. Since the medians are effectively
  equal and the first enabled run also changed the score slightly, the PDL
  change was reverted.

The important nuance is that the single final mask is not enough to reject
all-MMA by itself. On the isolated official memory-attention raw tensor case,
all-MMA is closer to Python than the current accepted default:

| Mode | Tensor | max abs | mean abs | RMSE |
| --- | --- | ---: | ---: | ---: |
| default | output | `0.1490163803100586` | `0.007960733697439371` | `0.010996000951159966` |
| all-MMA | output | `0.024274706840515137` | `0.002237422127891238` | `0.003095443386337613` |
| default | layer3 after SA | `0.1435546875` | `0.0073561223058883725` | `0.009918824300767898` |
| all-MMA | layer3 after SA | `0.02734375` | `0.0017610307444960603` | `0.002418253366364975` |

Evidence is in
`outputs/sam31-memory-attention-compare-default-20260611j/summary.json` and
`outputs/sam31-memory-attention-compare-all-20260611j/summary.json`. That means
the next acceptance gate should not rely only on one tracking smoke mask hash.
The next root optimization phase should collect multi-case mask statistics,
raw-tensor drift, and timing under the same input resolution/precision contract
before choosing between:

- promote all-MMA if multi-case parity is statistically acceptable and remains
  faster than official Python;
- implement a layer-0-specific fast path that preserves the accepted tile
  numeric behavior;
- implement a tile-native stream-K/fixup mapping, rather than reusing the MMA
  scheduler contract on the tile kernel.

## 2026-06-11 Mask-Init Matrix Harness

A reproducible matrix runner was added so the all-MMA decision can be made from
repeatable statistics rather than a single smoke run:

```bash
SAM31_MASK_INIT_MATRIX_OUT=outputs/sam31-mask-init-matrix-r3-20260611k \
SAM31_MASK_INIT_MATRIX_SIZES=320x240,640x360 \
SAM31_MASK_INIT_MATRIX_REPEATS=3 \
SAM31_MASK_INIT_MATRIX_VARIANTS=default,all,sa123,sa123-streamk \
SAM31_MASK_INIT_MATRIX_PY_WARMUP_RUNS=1 \
just sam31-mask-init-matrix
```

The runner writes per-run C++ summaries and masks, one official Python reference
per resolution, and a combined summary at
`outputs/sam31-mask-init-matrix-r3-20260611k/summary.json`. Paths recorded in the
summary are repository-relative.

The first two-resolution result confirms that all-MMA is the only tested mode
that beats the official Python encoded propagation step under the same
bf16/TF32 mask-init contract:

| Resolution | Python propagate | Variant | C++ encoded median | Speed vs Python | Python mask IoU median | xor median |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `320x240` | `16.682 ms` | default | `25.013 ms` | `0.67x` | `0.9655777537796977` | `510` |
| `320x240` | `16.682 ms` | all-MMA | `15.041 ms` | `1.11x` | `0.9585079064738479` | `614` |
| `320x240` | `16.682 ms` | `123` | `19.854 ms` | `0.84x` | `0.9622845555931058` | `558` |
| `320x240` | `16.682 ms` | `123` + stream-K | `18.243 ms` | `0.91x` | `0.9624298939117508` | `556` |
| `640x360` | `16.704 ms` | default | `25.620 ms` | `0.65x` | `0.9821185057140114` | `931` |
| `640x360` | `16.704 ms` | all-MMA | `15.031 ms` | `1.11x` | `0.9810235767682576` | `990` |
| `640x360` | `16.704 ms` | `123` | `18.416 ms` | `0.91x` | `0.9816140485558008` | `958` |
| `640x360` | `16.704 ms` | `123` + stream-K | `18.876 ms` | `0.89x` | `0.9816901138130242` | `954` |

This does not yet prove all-MMA should become the default. It does prove that
the old one-mask rejection was too narrow: at `640x360`, all-MMA loses only
about `0.0011` IoU versus the default C++ mask-to-Python comparison while still
being about `1.7 ms` faster than Python propagation. The next acceptance run
should expand the same harness to more prompt/mask shapes and real-video frames
before either promoting all-MMA or investing in a layer-0-specific tile-precision
fast path.

The harness was then extended with explicit mask cases and frame-1 synthetic
motion offsets. C++ `sam31_tracking_mask_init_smoke`, the official Python
reference runner, and the state-dump parity runner all accept the same
`--mask-case` and `--frame1-offset` contract. Matrix variants can also be given
as `sa<layers>` or `sa<layers>-streamk` for focused self-attention layer
selection, for example `sa023`.

The four-case acceptance probe used:

```bash
SAM31_MASK_INIT_MATRIX_OUT=outputs/sam31-mask-init-matrix-cases-r3-20260611l \
SAM31_MASK_INIT_MATRIX_SIZES=320x240,640x360 \
SAM31_MASK_INIT_MATRIX_CASES=center@3,small-center@7,left-wide@11,bottom-band@17 \
SAM31_MASK_INIT_MATRIX_REPEATS=3 \
SAM31_MASK_INIT_MATRIX_VARIANTS=default,all,sa123 \
SAM31_MASK_INIT_MATRIX_PY_WARMUP_RUNS=1 \
just sam31-mask-init-matrix
```

Summary, comparing all-MMA to the current default C++ mask and official Python:

| Resolution/case | Python propagate | default C++ | all-MMA | all speed vs Python | default Python IoU | all Python IoU | IoU delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `320x240 center@3` | `16.984 ms` | `24.939 ms` | `15.014 ms` | `1.13x` | `0.9655777537796977` | `0.9585079064738479` | `-0.00707` |
| `320x240 small-center@7` | `16.638 ms` | `24.624 ms` | `15.168 ms` | `1.10x` | `0.8769874476987448` | `0.8540175833162952` | `-0.02297` |
| `320x240 left-wide@11` | `16.792 ms` | `24.723 ms` | `15.554 ms` | `1.08x` | `0.9444048928871918` | `0.9479268372062748` | `+0.00352` |
| `320x240 bottom-band@17` | `16.881 ms` | `25.136 ms` | `15.026 ms` | `1.12x` | `0.9809415337889142` | `0.980402582605393` | `-0.00054` |
| `640x360 center@3` | `16.851 ms` | `24.965 ms` | `15.198 ms` | `1.11x` | `0.9821185057140114` | `0.9810235767682576` | `-0.00109` |
| `640x360 small-center@7` | `16.713 ms` | `25.181 ms` | `15.653 ms` | `1.07x` | `0.8001598721023181` | `0.7903780068728522` | `-0.00978` |
| `640x360 left-wide@11` | `16.749 ms` | `24.786 ms` | `15.619 ms` | `1.07x` | `0.9588018005288549` | `0.9510114159823754` | `-0.00779` |
| `640x360 bottom-band@17` | `16.750 ms` | `24.663 ms` | `15.041 ms` | `1.11x` | `0.9812107033191987` | `0.981721137676391` | `+0.00051` |

This is a clear speed signal but still not a defaultable parity result. all-MMA
beats Python propagation in every tested case, but the worst mask IoU delta is
`-0.02297` on `320x240 small-center@7`, and `-0.00978` on the same case at
`640x360`. That is too large to call "same precision" for the goal.

A focused layer-combination probe on the weak `small-center@7` case also failed
to find a same-parity speed win:

| Resolution | Variant | C++ encoded | Python IoU | C++ default IoU |
| --- | --- | ---: | ---: | ---: |
| `320x240` | all-MMA | `17.702 ms` | `0.8540175833162952` | `0.9721413721413722` |
| `320x240` | `sa023` | `18.086 ms` | `0.8612943116240726` | `0.981753355704698` |
| `320x240` | `sa123` | `21.476 ms` | `0.8741666666666666` | `0.9902748414376321` |
| `640x360` | all-MMA | `15.581 ms` | `0.7903780068728522` | `0.9849706380013935` |
| `640x360` | `sa023` | `18.134 ms` | `0.7966914409014625` | `0.9914461905709171` |
| `640x360` | `sa123` | `19.059 ms` | `0.7929216265878405` | `0.9894464356829948` |

Evidence is in
`outputs/sam31-mask-init-matrix-layer-combos-smoke-20260611l/summary.json`.
The layer-combo result points back to a kernel-level fix: the closer-to-default
variants keep layer 0 on the tile path or reduce all-MMA drift, but they are not
fast enough to beat Python. The next optimization should therefore target a
numerically aligned fast D32 self-attention path for layer 0, or a tile-native
stream-K/fixup implementation, rather than defaulting all-MMA globally.

## 2026-06-11 D32 Self-Attention Root Triage

The actual SAM3.1 propagation graph now has an env-gated raw tensor dump through
`SAM31_PROPAGATION_DUMP_DIR`. Comparing default against all-MMA on
`320x240 small-center@7` shows that the inputs to memory attention are identical
and the drift starts immediately after layer-0 self-attention:

| Tensor | Max abs | Mean abs | RMSE |
| --- | ---: | ---: | ---: |
| `sam31_mem_attn_input` | `0.0` | `0.0` | `0.0` |
| `sam31_mem_attn_layer0_after_sa` | `0.050495386` | `0.00680253` | `0.00848031` |
| `sam31_mem_attn_layer3_after_ffn` | `0.510280` | `0.024393` | `0.031315` |
| `sam31_mux_masks` | `0.392370` | `0.018761` | `0.028012` |
| `sam31_mux_iou` | `0.321182` | `0.080726` | `0.119145` |

Evidence:
`outputs/sam31-prop-dump-all-vs-default-small-center-20260611n/summary.json`.
This makes the next kernel target narrower: the decoder is amplifying the
attention drift, but the first material divergence is the D32 self-attention
kernel itself.

The broader `sa23` probe keeps layers 0 and 1 on the accepted tile path and uses
MMA for layers 2 and 3. It is consistently faster than the current default but
still does not beat official Python, and one case drops below a comfortable
default-parity threshold:

| Resolution/case | Python propagate | default C++ | `sa23` C++ | default IoU | Python IoU default | Python IoU `sa23` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `320x240 bottom-band@17` | `16.570 ms` | `24.826 ms` | `23.080 ms` | `0.999229` | `0.980942` | `0.980334` |
| `320x240 center@3` | `16.541 ms` | `25.153 ms` | `21.574 ms` | `0.995600` | `0.965578` | `0.963150` |
| `320x240 left-wide@11` | `16.637 ms` | `24.759 ms` | `21.674 ms` | `0.992875` | `0.944405` | `0.946809` |
| `320x240 small-center@7` | `16.937 ms` | `24.875 ms` | `22.855 ms` | `0.994474` | `0.876987` | `0.880453` |
| `640x360 bottom-band@17` | `16.786 ms` | `25.132 ms` | `22.096 ms` | `0.998490` | `0.981211` | `0.981533` |
| `640x360 center@3` | `16.647 ms` | `25.406 ms` | `22.089 ms` | `0.996602` | `0.982119` | `0.982085` |
| `640x360 left-wide@11` | `17.429 ms` | `24.871 ms` | `22.273 ms` | `0.994689` | `0.958802` | `0.957545` |
| `640x360 small-center@7` | `17.191 ms` | `24.985 ms` | `22.184 ms` | `0.995721` | `0.800160` | `0.799185` |

Evidence:
`outputs/sam31-mask-init-matrix-sa23-cases-20260611n/summary.json`.
`sa23` is therefore useful negative/triage evidence but is not promoted as a
default: it is only `1.08x-1.17x` faster than default and remains about
`0.72x-0.78x` of official Python speed.

Two tile-side schedule probes were also rejected:

- Enabling tile stream-K for D32 self-attention kept the layer-0/1 tile kernel
  on the same arithmetic path but changed the schedule to one block per tile.
  The D32 SA row stayed around `3.76-3.77 ms` and the smoke regressed to
  `31.118 ms` encoded propagation.
- Increasing the D32 tile `nbatch_fa` from `64` to `128` reduced a single D32
  SA row from about `3.66-3.77 ms` to about `3.47-3.53 ms`, but the smoke did
  not improve (`25.394 ms` versus `25.280 ms` in the paired profile runs) and
  the mask changed by `11` pixels against the current default
  (`default IoU=0.9976615646258503`).

The cuDNN head32 path was rechecked as well. BF16 cuDNN remains unsuitable as a
default because it is slower in the full smoke and follows the lower-parity
MMA-like output. fp32 cuDNN SDPA is not usable in the current local bridge: the
standalone `sam3_cudnn_sdpa_bench --io-dtype fp32 --out-dtype fp32` run failed
with an illegal memory access. The cuDNN path should stay behind the unsafe
experimental gates until the dtype/output contract is fixed.

Current conclusion: the speed target is real but still open. C++ can beat the
official Python propagation timing only by using lower-parity all-MMA. To beat
Python at the accepted precision/parity level, the next implementation needs a
new D32 self-attention kernel that is closer to the tile numerical contract
while approaching the MMA/cuDNN runtime, instead of another selector or scheduler
toggle.

## 2026-06-11 D32 MMA Real-QKV Microbench

The FATTN triage path now dumps env-gated layer-0 self-attention snapshots:
`sam31_mem_attn_layer0_sa_fattn_{q,k,v}_dump` and
`sam31_mem_attn_layer0_sa_fattn_dump`. The snapshots are materialized with
`ggml_cont` only when `SAM31_PROPAGATION_DUMP_DIR` is set, avoiding allocator
aliasing from view tensors. Comparing default and all-MMA on
`320x240 small-center@7` confirms:

| Tensor | Max abs | Mean abs | RMSE |
| --- | ---: | ---: | ---: |
| FATTN Q dump | `0.0` | `0.0` | `0.0` |
| FATTN K dump | `0.0` | `0.0` | `0.0` |
| FATTN V dump | `0.0` | `0.0` | `0.0` |
| FATTN output dump | `0.16220212` | `0.00730673` | `0.01375103` |

Evidence:
`outputs/sam31-prop-d32-real-snap-20260611/snapshot_default_vs_mma_summary.json`.

`sam3_fattn_parity` now accepts real Q/K/V dumps through `--input-q`,
`--input-k`, and `--input-v`, and can emit both JSON timing and raw output
dumps. Using the real SAM3.1 layer-0 Q/K/V reproduces the model FATTN drift
outside the full graph:

| Variant | Mean ms | Max abs vs tile | Mean abs | RMSE |
| --- | ---: | ---: | ---: | ---: |
| default tile | `4.320852` | reference | reference | reference |
| original all-MMA | `2.501089` | `0.16220212` | `0.00730673` | `0.01375103` |
| all-MMA with D32/ncols8 `nbatch_fa=64` | `0.516760` | `0.16220212` | `0.00730673` | `0.01375103` |

Evidence:
`outputs/fattn-d32-real-qkv-20260611/default_vs_mma_summary.json` and
`outputs/fattn-d32-real-qkv-nbatch64-20260611/default_vs_mma_nbatch64_summary.json`.
The `nbatch_fa=64` change improves the real-QKV D32/ncols8 MMA microbench by
about `4.8x` versus the previous all-MMA timing and `8.36x` versus the tile
baseline, without changing the all-MMA output tensor. It is therefore a real
speed optimization but not a parity fix.

The focused mask matrix after this change:

| Variant | Encoded propagation mean | Python mean | Python speed ratio | default IoU | Python IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | `25.041748 ms` | `16.670802 ms` | `0.67x` | `1.0` | `0.87698745` |
| `sa23` | `21.924709 ms` | `16.670802 ms` | `0.76x` | `0.99447396` | `0.88045302` |
| `sa123` | `19.650702 ms` | `16.670802 ms` | `0.85x` | `0.99027484` | `0.87416667` |
| all-MMA | `15.126118 ms` | `16.670802 ms` | `1.10x` | `0.97214137` | `0.85401758` |

Evidence:
`outputs/sam31-mask-init-matrix-nbatch64-layers-small-20260611/summary.json`.
This moves the lower-parity all-MMA path comfortably ahead of official Python
on the weak `small-center@7` case, but same-parity remains open: layer-0 tile is
still required for near-default masks, and the layer-preserving variants remain
slower than Python. The next root optimization should reduce the MMA output
drift itself, likely by changing the D32 self-attention accumulation/softmax/VKQ
contract rather than selector policy.

### 2026-06-12 D32 Attention Python Boundary Recheck

The official Python dump script now has an opt-in propagation attention dump:
`scripts/dump_sam31_mask_init_state.py --dump-propagation-attn`. It runs frame-1
propagation and saves the selected memory-attention self-attention Q/K/V/output
tensors in ggml layout. This is a diagnostic path only; the default mask-init
state dump remains unchanged unless the flag is passed.

The first same-case check used `320x240 small-center@7`, BF16 autocast with TF32
on, C++ `sam3.1_multiplex-f16.ggml`, and current C++ dumps with and without the
layer-0 TILE path. The Python/C++ Q/K/V inputs are still not close enough to use
as a strict low-level parity gate, which means the upstream feature gap remains
visible before the attention kernel. At the attention output only, Python is
closer to the C++ MMA output than to the C++ TILE output:

| Tensor | Python vs C++ default TILE mean_abs | Python vs C++ layer0-MMA mean_abs | C++ TILE vs MMA mean_abs | Evidence |
| --- | ---: | ---: | ---: | --- |
| layer0 SA output | `0.00840589` | `0.00246659` | `0.00730287` | `outputs/sam31-prop-d32-real-snap-current-20260612a/python_vs_cpp_layer0_sa_summary.json` |

This does not justify switching the default to layer0-MMA by itself, because
mask-level behavior is still the acceptance criterion. A fresh two-case matrix
keeps that tradeoff clear:

| Case | Official Python propagation | C++ default encoded propagation | C++ layer0-MMA ncols8 | Default Python IoU/XOR | layer0-MMA Python IoU/XOR | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `320x240 center@3` | `17.336 ms` | `16.864 ms` | `14.588 ms` | `0.983081 / 249` | `0.977701 / 328` | `outputs/sam31-mask-init-matrix-current-l0mma-2case-20260612a/summary.json` |
| `320x240 small-center@7` | `16.831 ms` | `17.061 ms` | `14.505 ms` | `0.856450 / 710` | `0.831984 / 851` | `outputs/sam31-mask-init-matrix-current-l0mma-2case-20260612a/summary.json` |

Conclusion: layer0-MMA remains a real speed lever (`~14.5 ms` encoded
propagation on these two cases), but it is still a quality regression at the
mask boundary. The next CUDA work should not simply promote the existing MMA
selector. It should either make the fast D32 path preserve the accepted mask
behavior, or close the upstream Python/C++ feature gap enough that the MMA-like
attention contract can be judged against official Python at the full mask level.

### 2026-06-28 Clean-Env BF16 Matrix and Mux FATTN Guard

The mask-init matrix runner now treats `None` variant env entries as explicit
unsets. This prevents stale shell variables such as
`GGML_CUDA_FORCE_FATTN_MMA_MATCH` or `GGML_CUDA_ENABLE_FATTN32_MMA` from leaking
into a later `default` run. The same change adds mux self-attention experiment
variants, but those remain measurement-only.

Current clean-env BF16/TF32-on results with the `sam3.1_multiplex-bf16.ggml`
model keep the default C++ propagation ahead of official Python on all four
representative `320x240` cases:

| Case | Python propagation | C++ default encoded propagation mean/median | Mean speed ratio | Python IoU/XOR |
| --- | ---: | ---: | ---: | ---: |
| `bottom-band@17` | `17.200175 ms` | `15.608009 / 15.572871 ms` | `1.10x` | `0.994162687 / 77` |
| `center@3` | `16.682727 ms` | `15.498383 / 15.514636 ms` | `1.08x` | `0.981627654 / 270` |
| `left-wide@11` | `17.275114 ms` | `15.529897 / 15.530062 ms` | `1.11x` | `0.959958398 / 616` |
| `small-center@7` | `17.375352 ms` | `16.177796 / 15.544934 ms` | `1.07x` | `0.871616079 / 626` |

Evidence:
`outputs/sam31-mask-init-matrix-current-bf16-mux-self-cleanenv-4case-5run-20260628a/summary.json`.

The named mux FATTN profile showed that `GGML_CUDA_ENABLE_FATTN32_MMA=all`
only changed the small `sam31_mux_block{0,1}_self_attn_fattn` rows among the mux
nodes. A clean-env matrix does not justify adopting that as a default:

| Case | Default encoded mean | `mux-self-mma` encoded mean | Default XOR from mux variant |
| --- | ---: | ---: | ---: |
| `bottom-band@17` | `15.608009 ms` | `15.625995 ms` | `0` |
| `center@3` | `15.498383 ms` | `16.911525 ms` | `0` |
| `left-wide@11` | `15.529897 ms` | `15.575404 ms` | `5` |
| `small-center@7` | `16.177796 ms` | `15.582756 ms` | `2` |

The `small-center@7` mean improvement comes with a default-mask delta and does
not hold on median (`15.544934 ms` default vs `15.579805 ms` mux self-MMA). The
right default remains the existing D32 policy: SAM3.1 memory-attention D32 MMA
where it already preserves accepted masks, plus the layer-0 TILE guard.

The same sweep found a ggml robustness issue: forcing MMA on the mux head-16
token-to-image/final/image-to-token rows could select `BEST_FATTN_KERNEL_MMA_F16`
and then abort inside the MMA switch. The dispatch guard now enumerates the head
dimensions actually implemented by the MMA path, so unsupported head-16 rows
fall back to TILE instead of aborting. Verification:
`outputs/sam31-fattn-profile-mux-force-mma-fallback-bf16-center-20260628a/`.
Those logs show the forced head-16 rows completing with
`schedule=parallel_k_combine`, while the process exits successfully.

## 2026-06-11 Named FATTN Profile and D32 Tile Selector Sweep

`GGML_CUDA_PROFILE_FATTN=1` now includes the destination tensor name, so the
long D32 rows can be attributed directly instead of inferred from shape. A
focused `sa23` profile on `320x240 small-center@7` confirmed the remaining
same-parity bottleneck:

| Tensor | Compute ms | Total ms | ncols | nbatch_fa | Schedule |
| --- | ---: | ---: | ---: | ---: | --- |
| `sam31_mem_attn_layer0_sa_fattn` | `3.764608` | `3.809248` | `32` | `64` | `parallel_k_combine` |
| `sam31_mem_attn_layer1_sa_fattn` | `3.542080` | `3.584064` | `32` | `64` | `parallel_k_combine` |
| `sam31_mem_attn_layer2_sa_fattn` | `0.516224` | `0.558304` | `64` | `128` | `one_block_per_tile` |
| `sam31_mem_attn_layer3_sa_fattn` | `0.516352` | `0.558304` | `64` | `128` | `one_block_per_tile` |

Evidence:
`outputs/sam31-profile-named-sa23-nbatch64-20260611/fattn_named_summary.json`.

The existing D32 tile implementation was then swept across K-split and ncols
settings. These were negative experiments:

| Experiment | Layer-0 SA compute | Layer-1 SA compute | Result |
| --- | ---: | ---: | --- |
| `parallel_blocks=1` | `3.752160 ms` | `3.732832 ms` | no material win |
| `parallel_blocks=2` | `3.725312 ms` | `3.705728 ms` | current selector behavior |
| `parallel_blocks=4` | `3.781632 ms` | `3.764512 ms` | slower |
| `parallel_blocks=8` | `3.885312 ms` | `3.867584 ms` | slower |
| forced `ncols=64` | `4.677216 ms` | `4.640640 ms` | slower |
| forced `ncols=16` | `4.414464 ms` | `4.400128 ms` | slower |
| forced `ncols=8` | `6.958144 ms` | `6.936000 ms` | much slower |
| forced `ncols=4` | `7.442560 ms` | `7.380000 ms` | much slower |
| forced `ncols=2` | `8.838240 ms` | `8.845440 ms` | much slower |

Evidence:
`outputs/sam31-profile-named-sa23-pb1-20260611/summary.log`,
`outputs/sam31-profile-named-sa23-pb2-20260611/summary.log`,
`outputs/sam31-profile-named-sa23-pb4-20260611/summary.log`,
`outputs/sam31-profile-named-sa23-pb8-20260611/summary.log`,
`outputs/sam31-profile-named-sa23-d32ncols64-20260611/summary.log`, and
`outputs/sam31-profile-named-sa23-d32ncols{16,8,4,2}-20260611/summary.log`.

Conclusion: existing D32 tile selector tuning is exhausted for this shape.
The current `ncols=32`, `nbatch_fa=64`, `parallel_blocks=2` path is the best of
the tested tile configurations, but it is still about `7x` slower than the D32
MMA self-attention rows. The next optimization should therefore be a new
numerically aligned D32 kernel or an MMA output-contract fix, not another tile
selector tweak.

## 2026-06-11 D32 MMA Numeric Contract Recheck

The real-Q/K/V layer-0 snapshot was then compared against simple NumPy SDPA
numeric modes. This changes the interpretation of the tile-vs-MMA drift:
standard attention math is much closer to the D32 MMA output than to the old
D32 tile output.

| Comparison | Max abs | Mean abs | RMSE | Evidence |
| --- | ---: | ---: | ---: | --- |
| tile vs MMA | `0.1622021198` | `0.0073067276` | `0.0137510322` | `outputs/fattn-d32-numeric-modes-20260611/summary.json` |
| f32 SDPA vs tile | `0.1533054113` | `0.0073257908` | `0.0138011410` | same |
| f32 SDPA vs MMA | `0.0194953680` | `0.0005115899` | `0.0008641898` | same |
| q/k/v rounded to f16, f32 math vs tile | `0.1533403397` | `0.0073261855` | `0.0138017731` | same |
| q/k/v rounded to f16, f32 math vs MMA | `0.0195435286` | `0.0005113165` | `0.0008643024` | same |

This means the old "make MMA more tile-like" framing is not the right target
for the broader SAM3.1 goal. The tile path preserved an earlier C++ mask
baseline, but the MMA path is the one that better matches the official SDPA-like
attention contract. The selector was therefore promoted: with
`GGML_CUDA_ENABLE_FATTN32_MMA` unset, named SAM3.1 memory-attention
self-attention and cross-attention nodes now use D32 MMA. Use
`GGML_CUDA_ENABLE_FATTN32_MMA=0` only for the old tile A/B baseline.

The production memory-attention slice was re-run after this selector change.
The result is the same high-parity level previously seen in the all-MMA
diagnostic row, and is much closer to official Python than the old mixed/tile
default:

| Case | Tensor | Max abs | Mean abs | RMSE | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| new default D32 MMA selector | `sam31_mem_attn_output` | `0.0242747068` | `0.0022374221` | `0.0030954434` | `outputs/sam31-memory-attention-compare-default-mma-20260611/summary.json` |
| old default evidence | `sam31_mem_attn_output` | `0.1490163803` | `0.0079607337` | `0.0109960010` | `outputs/sam31-memory-attention-compare-default-20260611j/summary.json` |

The same 320x240 `small-center@7` mask-init matrix now gives the new default a
small propagation timing win over official Python on this narrow contract:

| Row | Propagation timing | Mask comparison | Evidence |
| --- | ---: | --- | --- |
| Official Python SAM3.1 bf16/TF32 | `16.56408899 ms` | mask reference | `outputs/sam31-mask-init-matrix-default-mma-small-20260611/summary.json` |
| C++ new default D32 MMA selector | mean `15.53550767 ms`, median `15.838695 ms` | Python IoU `0.8549212196`, xor `709` | same |
| C++ `GGML_CUDA_ENABLE_FATTN32_MMA=all` | mean `15.07612067 ms`, median `15.084465 ms` | Python IoU `0.8540175833`, xor `714`; default-vs-all xor `5` | same |

This is not a full goal pass. It proves that the memory-attention D32 MMA
selector is the right speed/parity direction at the tensor level, and that the
remaining SAM3.1 mask divergence is no longer explained by choosing tile over
MMA inside the memory-attention slice. The next root-cause target is therefore
outside this standalone attention comparison: propagation state, mask-selection
contract, or non-memory-attention decoder/input differences on the full
tracking path.

The BF16/autocast-correct runner was then re-used to isolate the remaining
full-path divergence. ggml CUDA now has diagnostic name-match overrides for
FATTN scheduler selection:

- `GGML_CUDA_FORCE_FATTN_TILE_MATCH`
- `GGML_CUDA_FORCE_FATTN_VEC_MATCH`
- `GGML_CUDA_FORCE_FATTN_MMA_MATCH`

These do not change the default. They are intended to test individual SAM3/SAM3.1
attention sites without globally changing every FATTN node. On the current
`320x240`, `small-center@7`, BF16 GGML vs official Python BF16+TF32 contract,
the useful diagnostic selector is the ViT image encoder, not SAM3.1 memory
attention:

| Variant | Python IoU | XOR pixels | Encoded propagation mean | Total propagation mean | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| C++ default | `0.8371677906` | `821` | `15.890099 ms` | `243.391624 ms` | `outputs/sam31-mask-init-matrix-targeted-vit-best-20260611w/summary.json` |
| `SAM3_BF16_VIT_LINEAR_OUTPUT=1` only | `0.8483146067` | `756` | `15.387980 ms` | `243.940042 ms` | `outputs/sam31-mask-init-matrix-vit-linear-output-20260611w/summary.json` |
| ViT window attention + global blocks 15/23/31 forced to TILE | `0.8554387384` | `715` | `15.358660 ms` | `304.690049 ms` | same |
| same TILE selector + `SAM3_BF16_VIT_LINEAR_OUTPUT=1` | `0.8587948874` | `696` | `16.094237 ms` | `304.718666 ms` | same |
| Official Python SAM3.1 BF16+TF32 | reference | reference | reference | `16.588219 ms` | same |

This is a repeatable parity movement, but it is not a full parity pass.
The linear-output-only row is now default-on for SAM3.1 BF16 models only; set
`SAM3_DISABLE_BF16_VIT_LINEAR_OUTPUT=1` to force the previous behavior. It is
not enabled by default for native SAM3 BF16, where the older SAM3 timing rows in
this document showed no clear win.

The default-on check was repeated after the code change with `default` versus
`no-vit-linear-output` over `320x240` and `640x360`, `center@3` and
`small-center@7`, three repeats each. New default improved mean Python IoU by
`+0.003650`, reduced xor by `30.5 px` on average, and changed total propagation
by `+0.426%` on average (`-0.475%` best, `+1.652%` worst):
`outputs/sam31-mask-init-matrix-vit-linear-output-default-on-20260611x/summary.json`.
One case still regressed slightly (`320x240 center@3`, IoU `-0.000551`), so this
is accepted as a small SAM3.1 BF16 default improvement, not as an accuracy
completion.

The TILE-forced rows move the mask further but come with a large
total-propagation regression because the frame-1 ViT encode falls back to slower
TILE attention. Tensor dumps also show that the TILE improvement is not a simple
monotonic reduction of all intermediate errors: the best diagnostic row reduces
late ViT max/RMSE outliers but makes `image_features` mean/RMSE slightly worse
(`mean_abs 0.010535` vs default `0.008715`, `rmse 0.016605` vs default `0.014171`;
`outputs/sam31-mask-init-state-compare-vitbest-20260611w/summary.json` and
`outputs/sam31-mask-init-state-compare-default-20260611w/summary.json`). The
current interpretation is that late ViT attention accumulation/order changes
move local logits across the final mask threshold without solving the underlying
same-precision parity gap.

The next implementation target is therefore still a real CUDA path for the ViT
attention/activation contract: keep the faster default scheduler, but remove the
numerical mismatch by matching the official BF16 autocast dataflow around ViT
QKV, RoPE/FATTN, attention projection, and MLP output boundaries. The current
name-match FATTN overrides should stay diagnostic until that path beats Python
without the full-frame TILE slowdown.

The graph-level BF16 activation-chain toggles were rechecked under the new
SAM3.1 BF16 default. They remain rejected. On `320x240` and `640x360`,
`center@3` and `small-center@7`, three repeats each:

| Variant | Mean IoU delta vs default | Mean XOR delta | Mean total propagation delta | Evidence |
| --- | ---: | ---: | ---: | --- |
| `SAM3_BF16_VIT_QKV_CHAIN=1` | `-0.000742` | `+13.75 px` | `+6.735%` | `outputs/sam31-mask-init-matrix-vit-chain-breakdown-20260611y/summary.json` |
| `SAM3_BF16_VIT_MLP_CHAIN=1` | `-0.004124` | `+42.75 px` | `+6.221%` | same |
| QKV + MLP chain | `-0.003768` | `+41.75 px` | `+12.681%` | same |

This confirms the old SAM3 BF16 conclusion also holds for the current SAM3.1
path: isolated graph-level BF16 QKV/fc1 materialization adds conversion and
layout cost and does not reliably move the mask toward Python. The next useful
work is not to enable more existing casts, but to fuse or replace the producer
path so low-precision data is produced and consumed without extra graph nodes.

A narrower attention-projection experiment also remains diagnostic-only. ggml
now permits `WIN_UNPART` on BF16 tensors in both CPU and CUDA backends, which
allows SAM3 window-attention projection output to be tested in BF16. However,
`SAM3_BF16_VIT_ATTN_PROJ_OUTPUT=all` is not accepted: over the same four-case
matrix it had mean IoU delta `-0.001460`, mean XOR delta `+3.75 px`, and mean
total propagation delta `+0.778%`
(`outputs/sam31-mask-init-matrix-vit-attn-proj-output-all-20260611y3/summary.json`).
One case improved (`640x360 small-center@7`, IoU `+0.003829`, XOR `-50 px`),
but `320x240 small-center@7` regressed sharply (`-0.008823`, `+52 px`). The
current default therefore stays global-attention `attn.proj` plus all MLP `fc2`
linear outputs for SAM3.1 BF16; window `attn.proj` BF16 output remains an
explicit diagnostic only.

The SAM3.1 tracker neck positional encoding is now cached in a persistent
backend tensor. This does not change the mask: the before/after `320x240`
`center@3` default smoke produced the same frame-1 mask hash
`132087bcf41866e37c93895539537aea62f63f5c3a5b431e7a2e271fa003d79b`. In the
node-profile smoke, the second `sam31_encode_pe_build` span dropped from about
`7.8 ms` to `0.007 ms`
(`outputs/sam31-profile-pe-cache-20260611z/profile-summary.json`). In the
profile-free three-repeat matrix, `encode_frame1_ms` improved from `229.57 ms`
to `221.17 ms` and total `propagate_ms` improved from `245.85 ms` to
`238.38 ms`, with the same Python IoU `0.976106` and XOR `351 px`
(`outputs/sam31-mask-init-matrix-pe-cache-default-20260611z/summary.json`).
The remaining frame step is still dominated by ViT/neck encode work; the next
root target is therefore the encode graph itself, especially the ViT GEMM/FATTN
path and the remaining CPU image preprocessing span, not additional
application-level BF16 cast toggles.

The SAM3.1 `sam3_encode_image_for_tracking` path now also respects its tracker
only contract. Before this fix, the SAM3.1 dispatcher ignored the
`include_detector_neck=false` argument and still built the interactive neck on
propagation frames. The frame-0 `sam3_encode_image` path still builds both
feature adapters because adding a detection can need the interactive head; the
frame-1 tracking encode now builds only the propagation adapter. This preserves
the same frame-1 mask hash
`132087bcf41866e37c93895539537aea62f63f5c3a5b431e7a2e271fa003d79b`, score, and
object logit. On the same `320x240 center@3` BF16+TF32 matrix,
`encode_frame1_ms` improved from `221.17 ms` to about `202.40 ms`, and total
C++ `propagate_ms` improved from `238.38 ms` to `219.83 ms`
(`outputs/sam31-mask-init-matrix-tracking-only-encode-20260611z/summary.json`).
The node profile confirms that the second `sam31_encode` graph dropped from
`2794` to `2710` nodes and from `196.9 ms` to `179.6 ms` under synchronized
node profiling
(`outputs/sam31-profile-tracking-only-encode-20260611z/profile-summary.json`).
This is an accepted local speedup, but not the final same-contract Python win:
official Python still reports the frame-1 cache and propagation as separate
small timings on this harness, while C++ still performs a full frame encode.

The older ViT position backend-cache switch was also rechecked after the SAM3.1
neck PE cache. `SAM3_ENABLE_VIT_POS_CACHE=1` remains diagnostic-only for
SAM3.1: it kept the same mask hash and scores, but the three-repeat
`320x240 center@3` mean was not better (`propagate_ms 238.91 ms`,
`encode_frame1_ms 221.44 ms`) than the current default PE-cache row
(`propagate_ms 238.38 ms`, `encode_frame1_ms 221.17 ms`;
`outputs/sam31-mask-init-matrix-vit-pos-cache-optin-20260611z/summary.json`).
The matrix runner now exposes this as the `vit-pos-cache` variant for future
A/B checks.

For the final goal, the key remaining SAM3.1 comparison issue is input-contract
scope. The official Python runner reports `frame1_cache_ms` separately from
`propagate_ms`; the C++ `propagate_ms` currently includes frame-1 encode plus
encoded propagation. C++ already wins the encoded propagation slice on some
small rows, but it does not yet win the same full frame-step contract while
preserving the current Python mask comparison. Further work must reduce the
frame encode path, not only the encoded propagation graph.

## 2026-06-11 SAM3.1 CUDA BICUBIC Preprocess Default

The remaining CPU preprocessing span in the SAM3.1 CUDA path has been replaced
with a CUDA implementation of the same input contract: Pillow-compatible
BICUBIC resize, 8-bit intermediate clipping after each separable pass, and the
same fp16-storage normalization used by the CPU path. This is enabled by
default for SAM3.1 when the model backend is CUDA. Set
`SAM31_DISABLE_CUDA_BICUBIC_PREPROCESS=1` to force the old CPU preprocessing
baseline.

The input tensor contract was checked directly before accepting the default.
On the `320x240 center@3` mask-init smoke, the dumped
`input_image_preprocessed` tensor from CPU preprocessing and the CUDA BICUBIC
path matched exactly (`max_abs 0.0`, `mean_abs 0.0`,
shape `[1008, 1008, 3, 1]`):
`outputs/sam31-cuda-bicubic-preprocess-state-compare-20260611z`.

The default-on three-repeat matrix preserves the same C++ frame-1 mask hash
`132087bcf41866e37c93895539537aea62f63f5c3a5b431e7a2e271fa003d79b`, score
`0.944625199`, object logit `3.015126944`, and Python comparison
(`IoU 0.9761061947`, XOR `351 px`). It reduces the full C++ frame-step timing
for the same `320x240 center@3` BF16+TF32 contract:

| Variant | `encode_frame1_ms` | `propagate_encoded_ms` | Full C++ `propagate_ms` | Evidence |
| --- | ---: | ---: | ---: | --- |
| default CUDA BICUBIC preprocess | mean about `172.79 ms` | mean `17.238604 ms` | mean `190.026316 ms` | `outputs/sam31-mask-init-matrix-cuda-bicubic-default-20260611z/summary.json` |
| `SAM31_DISABLE_CUDA_BICUBIC_PREPROCESS=1` | mean about `203.31 ms` | mean `16.571866 ms` | mean `219.883080 ms` | same |

This is an accepted frame-encode speedup: roughly `30.5 ms` less frame-1 encode
time and `29.9 ms` less full C++ frame-step time on this harness, with exact
input parity and unchanged mask output. It still does not complete the broader
same-contract Python goal, because official Python's reported frame-1 cache is
not the same full-frame encode workload that C++ currently performs.

The benchmark harness now also supports C++ warmup via
`--warmup-runs` on `sam31_tracking_mask_init_smoke` and
`--cpp-warmup-runs` in `scripts/run_sam31_mask_init_matrix.py`. This matters:
the official Python matrix was already using one warmup run, while the earlier
C++ rows were cold-process timings. With Python and C++ both using one warmup
run, the same `320x240 center@3` default row is nearly tied but still not a
clear C++ win:

| Path | Two-frame/cache timing | Add-mask / propagation timing | Full comparable step | Evidence |
| --- | ---: | ---: | ---: | --- |
| Official Python BF16+TF32, warmup 1 | `frame0_cache 403.831594 ms`, `frame1_cache 0.388644 ms` | `add_mask 9.998117 ms`, `propagate 16.791600 ms` | `431.009955 ms` | `outputs/sam31-mask-init-matrix-cuda-bicubic-warm-20260611z/summary.json` |
| C++ default, warmup 1 | mean two-frame encode `377.387672 ms` | mean `frame0 add+setup 38.9 ms`, encoded propagation `14.788595 ms` | mean `431.362455 ms` | same |
| C++ with CPU preprocess forced | mean two-frame encode `425.720270 ms` | encoded propagation `13.829808 ms` | mean `477.287497 ms` | same |

This confirms two separate points. The CUDA BICUBIC preprocess is necessary for
a fair warmed comparison, reducing the warmed full step by about `45.9 ms`
against the old CPU-preprocess path. But the default row is still only within
noise of official Python, not statistically ahead. `SAM3_ENABLE_VIT_POS_CACHE=1`
was rechecked under the warmed contract and remains rejected: it preserved the
mask but raised the mean full step to `440.451318 ms`
(`outputs/sam31-mask-init-matrix-warm-vit-pos-cache-20260611z/summary.json`).

## 2026-06-11 SAM3.1 Encode Cut Profiling

The SAM3.1 encoder now has a targeted cut profiler for the remaining frame
encode bottleneck. Set `SAM31_PROFILE_ENCODE_CUTS=1` to profile the tracking
encode once per process. By default it records representative cumulative cuts;
set `SAM31_PROFILE_ENCODE_CUTS_ALL=1` to record every ViT block. Optional
controls are `SAM31_PROFILE_ENCODE_CUTS_WARMUP` and
`SAM31_PROFILE_ENCODE_CUTS_ITER`. The profiler intentionally runs at most once
for the tracking encode and once for frame-0 encode, because ggml tensors share
allocation state inside a graph context.

Representative tracking-encode cuts on the `320x240 center@3` BF16 model show
that the remaining cost is dominated by the ViT encoder, not preprocessing:

| Cut | Cumulative nodes | Cumulative compute | Evidence |
| --- | ---: | ---: | --- |
| `tracking:vit_prefix` | `14` | `0.495 ms` | `outputs/sam31-encode-cuts-default-20260611z2/profile.log` |
| `tracking:vit_block_07` | `660` | `37.879 ms` | same |
| `tracking:vit_block_15` | `1306` | `76.982 ms` | same |
| `tracking:vit_block_23` | `1952` | `113.884 ms` | same |
| `tracking:vit_block_31` | `2598` | `152.412 ms` | same |
| `tracking:encode_outputs` | `2710` | `178.557 ms` | same |

The all-block single-iteration cut run is noisy because each cut is a separately
allocated graph, but it confirms the broad shape: most of the remaining frame
encode is the 32-block ViT path, with the propagation neck adding a smaller
tail. No individual early block eliminates the gap by itself:
`outputs/sam31-encode-cuts-allblocks-20260611z2/profile.log`.

The existing `GGML_CUDA_ENABLE_FATTN32_MMA=all` diagnostic was also rechecked
under the warmed comparison contract and remains rejected. It slightly improves
the Python mask comparison (`IoU 0.9767204411`, XOR `342 px`) but changes the
C++ mask hash, score, and object logit, and it is slower than default on the
full warmed step:

| Variant | Full warmed C++ step | Python comparison | Default comparison | Evidence |
| --- | ---: | --- | --- | --- |
| default | mean `435.236065 ms` | IoU `0.9761061947`, XOR `351 px` | reference | `outputs/sam31-mask-init-matrix-warm-default-all-20260611z2/summary.json` |
| `GGML_CUDA_ENABLE_FATTN32_MMA=all` | mean `439.079425 ms` | IoU `0.9767204411`, XOR `342 px` | IoU `0.9989619377`, XOR `15 px` | same |

The next optimization target is therefore not more preprocess/cache work or a
global FATTN selector flip. It should be a root change inside the SAM3.1 ViT
block execution path: fused or lower-overhead QKV/FATTN/projection/MLP dataflow,
or a CUDA/ggml kernel path that preserves the accepted BF16/TF32 contract while
removing graph-level conversion and launch overhead across repeated ViT blocks.

## 2026-06-11 SAM3.1 ViT Stage Compute Profile

`examples/sam31_vit_block_case` now supports compute-only repeated timing with
`--warmup-runs` and `--repeats`. The helper builds and allocates the stage graph
once, uploads the input once, then times repeated `sam3_graph_compute` calls.
This avoids the earlier misleading stage timings that included graph setup and
allocation per repeat. The stage builder was also aligned with the main ViT
graph's BF16/F16 input/output policy so QKV, projection, and MLP stage profiles
measure the same precision contract as the full SAM3.1 encode path.

Representative BF16 CUDA stage timings for blocks 15, 23, and 31, with
warmup 2 and repeats 10, show the useful priority order:

| Stage | Mean compute per block | Evidence |
| --- | ---: | --- |
| QKV projection | `0.653 ms` | `outputs/sam31-vit-stage-profile-valid-20260611z/stage_timings.jsonl` |
| attention core | `2.866 ms` | same |
| attention projection | `0.339 ms` | same |
| MLP fc1 | `0.960 ms` | same |
| MLP GELU | `0.405 ms` | same |
| MLP fc2 | `1.275 ms` | same |
| full MLP stage | `2.234 ms` | same |

The subtotal for QKV + attention core + attention projection + full MLP is
about `6.09 ms/block`, matching the earlier cumulative encode-cut conclusion:
the remaining encode cost is dominated by FATTN and MLP, not by preprocessing
or QKV projection alone.

Two precision/output shortcuts were rechecked and remain rejected:

| Candidate | Result | Evidence |
| --- | --- | --- |
| `GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1` | Stage MLP improved slightly (`2.23 -> 2.19 ms` on block 15), but the C++ mask hash changed and Python IoU moved from `0.9761061947` to `0.9758306100`; not a same-parity optimization. | `outputs/sam31-mask-init-matrix-direct-bf16dst-20260611z/summary.json` |
| `GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION=1` | Slower (`457.824983 ms` full warmed C++ step) and changed the C++ mask hash; rejected. | `outputs/sam31-mask-init-matrix-cublaslt-geluerf-20260611z/summary.json` |

`GGML_CUDA_PROFILE_FATTN=1` on block 15 attention core shows the remaining
attention cost is mostly the FATTN kernel itself: steady-state `kernel_ms` is
about `1.75 ms`, while K/V conversion is about `0.15 ms/block`
(`outputs/sam31-vit-attn-core-fattn-profile-20260611z/stderr.log`). The next
root optimization should therefore target QKV split/RoPE/FATTN packing and
kernel execution together, rather than another global cublasLt heuristic or
standalone GELU/output-cast toggle.

A later MLP refresh confirmed the post-attention MLP split remains stable on
blocks 15, 23, and 31: `mlp_fc1` averages `0.975 ms`, standalone `mlp_gelu`
averages `0.405 ms`, `mlp_fc2` averages `1.279 ms`, and the fused full MLP
stage averages `2.237 ms`
(`outputs/sam31-vit-mlp-stage-refresh-20260611a/stage_summary.json`). The
accepted MLP-side CUDA cleanup is a vectorized fused
`UNARY(GELU_ERF) -> CPY(F32->BF16/F16)` path in ggml CUDA. When the source and
destination are contiguous, aligned, and have an even element count, the fused
kernel now loads two F32 values with `float2` and stores one packed BF16/F16
pair. Set `GGML_CUDA_DISABLE_UNARY_CPY_VEC2=1` to force the previous scalar
fused kernel.

The SAM3.1 full-MLP stage is bit-exact against the scalar fused kernel on
blocks 15, 23, and 31 (`max_abs=0`, `nonzero=0`) and improved stage mean by
`-2.18%`, `-0.58%`, and `-0.47%`
(`outputs/sam31-mlp-unary-cpy-vec2-probe-20260611a/summary.json`). A reverse
order block-15 recheck remained bit-exact and measured `-1.59%` mean-of-means
(`outputs/sam31-mlp-unary-cpy-vec2-probe-20260611a/block15_recheck/summary.json`).
The full `320x240 center@3` SAM3.1 mask-init matrix also preserved the C++
mask hash, score, and object logit exactly against
`no-unary-cpy-vec2`; full-frame timing stayed within run noise while frame0
encode moved from `194.33 ms` to `192.97 ms`
(`outputs/sam31-mask-init-matrix-unary-cpy-vec2-20260611a/summary.json`).

An attempted Q-side RoPE pair fusion was investigated and rejected as a default.
The existing ggml CUDA `rope_pair_fusion` already fuses the K-side RoPE in this
SAM3.1 global-attention stage, but Q-side fusion is rejected by the generic
fusion memory-range guard because the allocator reuses the Q input buffer for
the final interleaved RoPE output:
`outputs/sam31-vit-rope-reverted-check-20260611z/profile_stderr.log`. A naive
in-place exception made the stage faster (`~2.59 ms` attention core on block
15), but raw tensor comparison showed it was numerically wrong
(`max_abs=422.1975`, `mean_abs=2.5145` against the non-in-place stage output;
`outputs/sam31-vit-rope-inplace-norestrict-20260611z`). The likely hazard is
not the arithmetic pattern itself but allowing an aliasing output through a
kernel and fusion path designed for non-overlapping operands. The in-place
exception was removed.

The first accepted code change from this probe was diagnostic:
`GGML_CUDA_PROFILE_ROPE_PAIR_FUSION=1` reports why a rope-pair candidate was
rejected, and `GGML_CUDA_PROFILE_FUSION_MEMORY=1` can print the overlapping
source/destination that caused a fusion memory rejection. The follow-up
optimization below implements the alias-safe path under a stricter exact-layout
condition instead of allowing arbitrary overlapping outputs.

## 2026-06-11 SAM3.1 Head64 No-Mask FATTN Fast Path

SAM3.1 ViT global attention uses head64 FATTN with no mask and no attention
sinks. The CUDA MMA FATTN kernel already had a `no_mask_no_sinks` specialization,
but the head64 selector was not using it. The selector now takes that existing
specialization by default when `mask == nullptr` and `sinks == nullptr`; set
`GGML_CUDA_DISABLE_FATTN64_NOMASK_FAST=1` to force the previous head64 path.

The local stage evidence is bit-exact and faster on the three SAM3.1 global ViT
blocks:

| Block | New default attention core mean | Old path mean | Raw diff | Evidence |
| ---: | ---: | ---: | --- | --- |
| 15 | `2.862155 ms` | `2.922861 ms` | `max_abs=0`, `mean_abs=0` | `outputs/sam31-fattn64-nomask-default-20260611g/` |
| 23 | `2.900634 ms` | `2.927669 ms` | `max_abs=0`, `mean_abs=0` | same |
| 31 | `2.887991 ms` | `3.008453 ms` | `max_abs=0`, `mean_abs=0` | same |

The full mask-init matrix also preserved the accepted C++ output exactly:
both paths produced mask hash
`132087bcf41866e37c93895539537aea62f63f5c3a5b431e7a2e271fa003d79b`, score
`0.944625199`, and object logit `3.015126944`. On five warmed repeats,
the new default reported `450.263682 ms` mean full-frame step versus
`457.378557 ms` with `GGML_CUDA_DISABLE_FATTN64_NOMASK_FAST=1`
(`outputs/sam31-mask-init-matrix-fattn64-nomask-default-20260611g/summary.json`).

Two additional Q RoPE allocator experiments were rejected while looking for a
larger fusion win. Materializing Q before RoPE and preserving the split QKV
buffer with `ggml_set_output` both failed raw output comparison by hundreds of
absolute units and did not remove the Q-side fusion memory rejection
(`outputs/sam31-vit-qrope-materialize-20260611c/` and
`outputs/sam31-vit-qkv-preserve-rope-20260611d/`). A K-before-Q graph reorder
also preserved raw output but did not remove the Q-side memory rejection
(`outputs/sam31-vit-rope-k-first-20260611h/`).

## 2026-06-11 SAM3.1 Exact-In-Place Q RoPE Fusion

The Q-side RoPE fusion is now enabled only when the final interleaved RoPE
output is an exact in-place alias of the Q real/imaginary views: same backend
buffer, `dst.data == x_re.data`, `x_im.data == dst.data + dst.nb[0]`, and
matching strides for dims 1..3. This avoids the earlier incorrect broad
overlap exception while still allowing the safe SAM3.1 ViT Q layout to use a
dedicated alias-safe kernel. Set
`GGML_CUDA_DISABLE_ROPE_PAIR_EXACT_INPLACE=1` to force the previous memory
rejection path.

Stage-level evidence is bit-exact and faster on the three SAM3.1 global ViT
blocks:

| Block | New default attention core mean | Disabled mean | Raw diff | Evidence |
| ---: | ---: | ---: | --- | --- |
| 15 | `2.539945 ms` | `2.863771 ms` | `max_abs=0`, `mean_abs=0` | `outputs/sam31-rope-exact-inplace-default-20260611k/summary.json` |
| 23 | `2.540849 ms` | `2.874246 ms` | `max_abs=0`, `mean_abs=0` | same |
| 31 | `2.558911 ms` | `3.028948 ms` | `max_abs=0`, `mean_abs=0` | same |

The full mask-init matrix also preserved the accepted C++ output exactly:
both paths produced mask hash
`132087bcf41866e37c93895539537aea62f63f5c3a5b431e7a2e271fa003d79b`, score
`0.944625199`, and object logit `3.015126944`. On three warmed repeats, the
new default reported `429.896402 ms` mean full-frame step versus
`432.501658 ms` with `GGML_CUDA_DISABLE_ROPE_PAIR_EXACT_INPLACE=1`
(`outputs/sam31-mask-init-matrix-rope-exact-inplace-default-20260611k/summary.json`).

The current follow-up narrows the exact in-place RoPE kernel further for the
common SAM3.1 ViT F32 pair layout. When `x_re/x_im`, `cos/sin`, and the output
are adjacent pair-contiguous, the CUDA path now loads/stores the pair as
`float2`; set `GGML_CUDA_DISABLE_ROPE_PAIR_F32_VEC2=1` to force the previous
scalar exact-in-place kernel. The stage refresh before this change measured the
global-attention split at about `0.659 ms` for QKV projection, `2.491 ms` for
attention core, and `0.335 ms` for attention projection on blocks 15/23/31
(`outputs/sam31-vit-attn-stage-refresh-20260611b/summary.json`), keeping
attention core as the relevant local target.

Stage A/B against `GGML_CUDA_DISABLE_ROPE_PAIR_F32_VEC2=1` remained bit-exact
on blocks 15, 23, and 31 (`max_abs=0`, `nonzero=0`). The first 40-repeat pass
was noisy (`-2.41%`, `-8.34%`, `+1.69%` mean deltas), so a reverse-order
three-pair recheck is the acceptance signal: mean-of-means moved by `-0.39%`,
`-0.01%`, and `-2.47%`, with every pair still raw-equal
(`outputs/sam31-rope-pair-f32-vec2-probe-20260611b/recheck/summary.json`).
The full `320x240 center@3` mask-init A/B also preserved the C++ mask hash,
score, object logit, Python IoU, and default IoU exactly; full-frame timing was
within run noise (`423.91 ms` default versus `423.35 ms` with
`no-rope-pair-f32-vec2`), while `encode_frame0_ms` was effectively unchanged
(`192.53 ms` versus `192.63 ms`;
`outputs/sam31-mask-init-matrix-rope-pair-f32-vec2-20260611b/summary.json`).

Two follow-up probes were rejected after this change. A direct QKV split that
skipped the initial full-QKV `cont(permute)` made block-15/23/31 attention core
about `4.4-6.6%` faster, but raw output diverged heavily
(`max_abs` up to `218.069244` and `mean_abs` up to `1.841763`;
`outputs/sam31-direct-qkv-split-20260611l2/summary.json`). This indicates the
visible `[3E, W, H, B]` QKV tensor shape is not enough to replace the existing
ggml reshape/permute layout contract with simple direct views. A head64
`ncols2` sweep was also rejected: forced `ncols2=1` was bit-exact but slightly
slower than the selector, while `ncols2=2/4/8` was much slower and numerically
wrong (`outputs/sam31-fattn64-ncols2-sweep-20260611l/summary.json`). The next
viable head64 optimization needs either a dedicated QKV pack kernel that
reproduces the current layout exactly, or a real head64 attention kernel/SDPA
path, not a simple view rewrite or ncols2 retune.

A cuDNN SDPA head64 opt-in probe was also rejected. The path did route on the
SAM3.1 global block-15 attention shape
`Q/K/V=[64,5184,16,1]`, but it was slower than the native CUDA FATTN path and
not parity-safe: default FATTN measured `2.385279 ms`, while the cuDNN path
measured `2.836438 ms`, with raw output diff `max_abs=426.056152` and
`mean_abs=2.979593` (`outputs/sam31-cudnn-head64-20260611m/`). The head64 cuDNN
entry point is therefore not exposed; the viable work remains a layout-exact
QKV pack kernel or a native head64 SDPA/FATTN implementation.

Additional head64 FATTN selector/config probes did not produce a safe win.
Forcing TILE or VEC on block 15 was about `11.94 ms` and numerically wrong
(`max_abs=429.382202`, `mean_abs=3.148819`), while forcing MMA was bit-exact
and within run noise of the default selector
(`outputs/sam31-fattn-kernel-modes-20260611m/summary.json`). Changing the
head64/ncols=64 MMA config to keep Q in registers was also rejected: it changed
block-15 raw output by `max_abs=427.995300`, `mean_abs=2.788460`. The config was
restored and verified bit-exact against the prior default
(`outputs/sam31-fattn64-restored-20260611m/summary.json`).

`GGML_CUDA_PROFILE_CPY=1` now reports CUDA copy/contiguous timings so QKV
layout work can be measured directly. On block 15 attention core, the warmed
copy profile shows the initial full-QKV `cont(permute)` at about `0.258 ms`,
the Q and K contiguous packs at about `0.102 ms` and `0.096 ms`, and the final
stage output copy at about `0.049 ms`
(`outputs/sam31-cpy-profile-20260611m/block15.stderr`). A stricter direct
`view_4d` split probe that explicitly preserved the source strides was still
rejected: it improved block-15 attention core to about `2.25-2.27 ms`, but raw
output diverged by `max_abs=223.970627`, `mean_abs=2.551306`
(`outputs/sam31-direct-qkv-views-20260611m/`). This confirms the next QKV
optimization should be a dedicated, parity-tested pack/split kernel rather than
another graph-level view rewrite.

A row-grouped variant of the existing F32 contiguous-row vec4 copy was also
rejected. It reduced the profiled full-QKV copy in block 15 from about
`0.256-0.260 ms` to about `0.210-0.213 ms`, but the attention-core stage did
not improve consistently across blocks 15/23/31, and the 320x240 mask-init
smoke was slightly slower overall (`+0.156%` full-frame mean) while remaining
bit-identical (`outputs/sam31-cpy-vec4-row4x64-20260611n/summary.json`,
`outputs/sam31-cpy-vec4-row4x64-smoke-20260611n/summary.json`). The row-grouped
kernel was removed; only the copy profiler remains.

A Q/K head-pack fastpath that read from the full-QKV `cont(permute)` source
instead of the packed tensor was also rejected. It was intentionally scoped to
the f32 head-pack `CONT` nodes and kept the graph dependencies unchanged, but
the raw attention-core outputs diverged in the same way as the earlier direct
QKV view probes: block 15 changed by `max_abs=218.104086799`, `mean_abs=1.540207171`,
and blocks 23/31 changed by `max_abs=150.485580485`, `mean_abs≈1.26274`
(`outputs/sam31-qkv-head-fastpath-20260611o/summary.json`). This confirms that
the visible source strides are still not a sufficient contract for bypassing
the materialized QKV pack. The safe route remains either a dedicated fused
producer that exactly writes the current packed Q/K/V tensors, with raw parity
checked at each output, or a native head64 attention path that consumes the
current materialized tensors.

The current head64 FATTN profile for SAM3.1 ViT global attention is:
`Q/K/V=[64,5184,16,1]`, `types=f32/f32/f32`, `ncols=64`, `nbatch_fa=128`,
`ncols2=1`, and `schedule=one_block_per_tile`. Warmed block-15 timings show
about `1.80 ms` FATTN total, with about `1.65 ms` in the kernel and about
`0.15 ms` in K/V f16 conversion
(`outputs/sam31-head64-fattn-profile-20260611o/block15.stderr`). Setting
`GGML_CUDA_DISABLE_FATTN_STREAM_K=1` did not change the actual schedule for
this shape (`outputs/sam31-head64-streamk-ab-20260611o/summary.json`), so the
next head64 work should not be another stream-k toggle. It should target the
MMA kernel body or a parity-safe native f32/f32/f32 head64 path that avoids the
current conversion and kernel cost.

An env-gated probe that reused the existing MMA inline f32-to-f16 tile loader
for head64 K/V was parity-safe but slower. On block 15 it removed the external
K/V conversion cost (`K/V_convert_ms≈0.003 ms`) and kept raw output
bit-identical (`max_abs=0`, `mean_abs=0`), but the FATTN kernel grew to about
`2.318 ms` and the attention-core stage regressed by `+16.145%`
(`outputs/sam31-fattn64-inline-kv-probe-20260611o/summary.json`). The probe was
removed. This narrows the next viable path further: keep preconverted K/V or
write a new head64 kernel that avoids the current inline-loader register/shared
memory tradeoff, rather than simply toggling `D_SRC != DKQ` on the existing MMA
template.

Combining the separate K and V f32-to-f16 conversion kernels into one head64
conversion kernel was also rejected. It preserved raw parity exactly on block
15 (`max_abs=0`, `mean_abs=0`), but the attention-core stage regressed by
`+1.707%` (`outputs/sam31-fattn64-combined-kv-convert-20260611o/summary.json`).
The conversion subtotal was not the limiting cost once launch overhead and the
MMA kernel runtime were included, so the combined conversion probe was removed.

A graph-side K/V F16 cast probe was also rejected. Casting both K and V before
`ggml_flash_attn_ext` made the block 15/23/31 attention-core timings look faster
(`-3.48%`, `-6.64%`, `-2.73%`), but every output value became NaN
(`5,308,416` non-finite mismatches per block) while the default outputs had no
non-finite values. Splitting the probe into K-only or V-only on block 15 did not
produce a usable intermediate option: both variants selected no CUDA FATTN
kernel and aborted at the existing `BEST_FATTN_KERNEL_NONE` path
(`ggml/src/ggml-cuda/fattn.cu:1949`). The supporting run data is in
`outputs/sam31-vit-fattn-kv-f16-probe-20260611p/stage_summary.json` and
`outputs/sam31-vit-fattn-kv-f16-split-probe-20260611q/kv_f16_rejection_summary.json`.

A head64 MMA config probe that changed the `DKQ=64`, `DV=64`, `ncols=64`
Ampere `nbatch_fa` from `128` to `64` was also rejected. It was not just slower:
raw attention-core output diverged heavily on the same saved SAM3.1 Q/K/V
inputs (`max_abs=424.24` on block 15 and `289.90` on blocks 23/31). Stage mean
regressed by `+0.11%`, `+0.90%`, and `+1.50%`, with FATTN `kernel_ms` also
worse by about `+1.0%` to `+1.5%`
(`outputs/sam31-fattn64-nbatch64-probe-20260611a/summary.json`). The config was
restored to `nbatch_fa=128`. This confirms the head64 MMA config is part of the
measured numerical contract for this parity target; future config sweeps must
use raw tensor checks, not timing alone.

A narrower MMA-kernel probe then split the existing f32-to-f16 inline loader so
only K, only V, or both K and V could be converted inside the head64 no-mask
kernel while preserving the external f32/f32/f32 tensor contract. All three
variants were raw bit-exact on block 15 (`max_abs=0`, `mean_abs=0`, no
non-finite mismatches), but all were slower than the default preconversion path:
K-only `+11.06%`, V-only `+9.19%`, and K+V `+14.36%`
(`outputs/sam31-fattn64-inline-kv-split-probe-20260611q/block15_summary.json`).
The split-inline code was removed, and a restored baseline recheck again showed
`need_f16_K=1`, `need_f16_V=1` with K/V conversion around `0.076/0.086 ms` and
kernel time around `1.78 ms`
(`outputs/sam31-fattn64-inline-kv-split-restored-20260611q/summary.json`).

The next accepted copy-side cleanup targets the Q/K head-pack copies that feed
the head64 global-attention FATTN input. These copies are already parity-safe
because they read from the materialized full-QKV `cont(permute)` tensor; the
change only replaces the generic one-row-per-block contiguous-row vec4 copy
with a `ne00==64` four-rows-per-block vec4 kernel. Set
`GGML_CUDA_DISABLE_CPY_ROWS64_VEC4_ROW4=1` to force the previous row kernel.
This deliberately avoids the earlier rejected direct-QKV-view and source-bypass
probes.

Stage A/B on blocks 15, 23, and 31 remained raw bit-exact (`max_abs=0`,
`nonzero=0`) and improved attention-core means by `-1.12%`, `-2.36%`, and
`-5.13%` in the first 40-repeat pass
(`outputs/sam31-cpy-rows64-row4-probe-20260611c/summary.json`). A reverse-order
three-pair recheck also stayed bit-exact and measured mean-of-means deltas of
`-2.95%`, `-3.03%`, and `-1.96%`
(`outputs/sam31-cpy-rows64-row4-probe-20260611c/recheck/summary.json`). The copy
profile shows the `[64,5184,16,1]` Q/K pack copies moving from about
`0.100 ms` to `0.079 ms` average while leaving the full-QKV materialization and
final output copy essentially unchanged
(`outputs/sam31-cpy-rows64-row4-profile-20260611c/`).

The full `320x240 center@3` mask-init A/B preserved the C++ mask hash
`132087bcf41866e37c93895539537aea62f63f5c3a5b431e7a2e271fa003d79b`, score
`0.944625199`, object logit `3.015126944`, Python IoU `0.9761061947`, and
default IoU `1.0`. Default full-frame mean was `412.416000 ms` versus
`422.712958 ms` with `no-cpy-rows64-row4`, and encoded propagation was
`14.056079 ms` versus `15.536444 ms`
(`outputs/sam31-mask-init-matrix-cpy-rows64-row4-20260611c/summary.json`). A
separate current-default seven-repeat run measured C++ full-frame
`420.864222 ms` against the same-run official Python BF16+TF32
`frame0_cache + add_mask + propagate = 429.083586 ms`, i.e. `-1.92%` on this
narrow smoke contract, with encoded propagation `-12.90%` versus Python
propagation (`outputs/sam31-current-default-matrix-20260611c/summary.json`).

The accepted copy fastpath has now been generalized conservatively beyond the
`ne00==64` Q/K pack case to the same F32 contiguous-row vec4 pattern at
`ne00==256` and `ne00==1024`. The broader disable switch is
`GGML_CUDA_DISABLE_CPY_VEC4_ROW_GROUP=1`; the legacy
`GGML_CUDA_DISABLE_CPY_ROWS64_VEC4_ROW4=1` still disables only the `ne00==64`
part. The SAM3 benchmark output JSONL remained semantically identical after
removing path-only differences, and the three emitted mask PNG SHA256 values
were unchanged:
`9c3a49ab90a7cd1049fdd7a3c9c5b3dec5559c0e242300fc55fc2cc1f0b9ea7d`,
`4970955d8ecdd278d053af765923da2bf218be28a0e4aaf16bb58dff220da5d6`, and
`98f4b073d5c897e8b33634d1ebd79538be98bd976e5678416a3b02b03355d79b`
(`outputs/sam3-f16-cpy-rowgroup-parity-20260611c/`).

On SAM3 f16, the copy profile showed the Hiera window QKV materialization
`[1024,576,9,3]` moving from `0.252057 ms` to `0.221643 ms` mean, the global
QKV materialization `[1024,5184,1,3]` from `0.248816 ms` to `0.220493 ms`, and
the Q/K pack `[64,576,16,9]` from `0.106151 ms` to `0.071500 ms`
(`outputs/sam3-f16-cpy-rowgroup-profile-20260611c/`). End-to-end A/B on
SAM3 f16 was small but positive when judged by the stable statistic:
default median `185.3 ms` versus disable median `187.2 ms` (`-1.0%`). The mean
delta was `-3.60%`, but the disabled side included one `212.3 ms` outlier, so
the median is the safer claim
(`outputs/sam3-f16-cpy-rowgroup-ab-20260611c/summary.json`). A wider attempt to
also cover `ne00==16` and `ne00==32` was rejected despite output parity because
the end-to-end result was not robust: enabled median `186.6 ms` versus disabled
median `187.7 ms`, but enabled runs also produced `207.0 ms` and `245.4 ms`
outliers (`outputs/sam3-f16-cpy-rowgroup-small-ab-20260611c/summary.json`).

The same full row-group disable comparison was added to the SAM3.1 matrix
runner as `no-cpy-row-group`. On `320x240 center@3`, default and disabled
preserved the same mask hash, score, object logit, Python IoU, and default IoU.
Default full-frame mean was `414.807992 ms` versus `421.483388 ms` with
`no-cpy-row-group` (`-1.58%`), and encoded propagation was `14.393617 ms`
versus `14.887963 ms` (`-3.32%`)
(`outputs/sam31-cpy-rowgroup-matrix-20260611c/summary.json`).

This still does not make SAM3 f16 faster than official Python at the same
input contract. The current comparable run uses decoded `960x540`, encode size
`1008`, prompt `person`, three frames, two warmups, and
`per_frame_encode_detect_propagate` scope on both sides. C++ f16 measured
`193.0 ms` mean and `185.3 ms` median; official Python SAM3 fp16+TF32 measured
`120.859593 ms` mean and `120.924765 ms` median. By mean, Python is still
`0.626x` the C++ time, i.e. C++ remains about `1.60x` slower
(`outputs/sam3-final-f16-rowgroup-matrix-20260611c/summary.json`). The next
root optimization should therefore move away from copy-only work and target the
dense encoder matmul / activation dtype contract: the current C++ graph keeps
many F16-weight matmuls and attention intermediates in F32 outputs, while the
official fp16+TF32 Python path is likely getting more favorable mixed-precision
kernel selection from PyTorch.

Two additional SAM3 f16 probes were rejected. `SAM3_CUDA_ENABLE_GRAPHS=1`
measured `198.84 ms` mean versus `199.04 ms` default over five pairs, only
`-0.10%`, so it is noise rather than a default change
(`outputs/sam3-f16-cuda-graphs-ab-20260611c/summary.json`). A refreshed
post-cuDNN-neck node profile shows the remaining steady-state bottlenecks:
`sam3_encode` is `165.010 ms`, dominated by `MUL_MAT` (`91.166 ms`) and
`FLASH_ATTN_EXT` (`19.648 ms`), while the repeated `pcs_fusion_encoder` call is
`31.191 ms`, dominated by six head_dim=32 global `FLASH_ATTN_EXT` nodes
(`21.228 ms`; `outputs/sam3-f16-cudnn-default-node-profile-20260611f/`).

Two head_dim=32 attention substitutions improve speed but fail same-output
parity, so neither is a default path. `GGML_CUDA_ENABLE_FATTN32_MMA=all` moves
the repeated `pcs_fusion_encoder` profile from `31.191` to `12.887 ms`, and
cuts its head_dim=32 attention total from `21.228` to `3.201 ms`
(`outputs/sam3-f16-fattn32-mma-probe-20260611f/`). Full-mask parity against the
default path is not exact: `mask_hash_equal_rows=0`, `min_bbox_iou=0.9977279532`,
`max_bbox_delta_px=0.152`, and `max_score_abs_delta=0.036469`
(`outputs/sam3-f16-fattn32-mma-parity-20260611f/compare.json`). The cuDNN SDPA
head32 route is similar: with
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32=1`,
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32_UNSAFE_RUN=1`, and
`GGML_CUDA_CUDNN_SDPA_ALLOW_F32_TO_BF16=1`, the 10-pair warmed A/B improved
frame-0 `init_ms` from `854.6` to `821.3 ms` and total from `1227.1` to
`1193.5 ms`, but full-mask parity was also non-exact
(`mask_hash_equal_rows=0`, `min_bbox_iou=0.9976745911`,
`max_score_abs_delta=0.038622`;
`outputs/sam3-f16-cudnn-sdpa-head32-ab-20260611f/summary.json`,
`outputs/sam3-f16-cudnn-sdpa-head32-parity-20260611f/compare.json`). These are
valid low-precision/unsafe experiments, not evidence that the same-precision
goal has been met.

### 2026-06-11 BF16 ADD-to-CPY Side-Output Probe

The BF16 CUDA profile was refreshed after the RoPE-pair, cuDNN SDPA head64,
FATTN K/V conversion, and unary-copy probes. With
`GGML_CUDA_PROFILE_MUL_MAT=1` and
`GGML_CUDA_PROFILE_CUBLASLT_BIAS_TIMING=1`, the warmed BF16 run still shows the
same root shape: ViT cuBLASLt matmuls dominate, followed by head64
FlashAttention, residual ADD, GELU, and layout/cast bridges
(`outputs/sam3-bf16-mulmat-warm-20260611/`). A synchronized node profile ranks
`sam3_encode` by drop-max time as:
`MUL_MAT [1024,5184]` `222.83 ms`, `MUL_MAT [4736,5184]` `176.38 ms`,
`MUL_MAT [3072,5184]` `115.79 ms`, window head64 FATTN `72.48 ms`,
residual ADD `[1024,72,72]` `52.39 ms`, and MLP GELU `46.57 ms`
(`outputs/sam3-bf16-node-profile-current-20260611/`).

An ADD-to-CPY side-output CUDA probe was added for cases where a fused ADD keeps
its required F32 output and can also write the following BF16/F16 CPY destination
from the same kernel. The disable switch is
`GGML_CUDA_DISABLE_FUSED_ADD_CPY_FUSION=1`, and
`GGML_CUDA_PROFILE_FUSED_ADD_CPY_FUSION=1` reports candidate diagnostics. This
path is parity-safe on the checked BF16 tracking output: all three 6-frame A/B
pairs had `min_bbox_iou=1`, `max_bbox_delta_px=0`, `max_score_abs_delta=0`, and
all mask hashes equal
(`outputs/sam3-bf16-add-cpy-side-fusion-20260611/seq-r2/compare-*.json`).

The performance signal is neutral rather than a root win. Over three sequential
6-frame A/B pairs, disabled measured `187.6`, `174.9`, and `188.1 ms/frame`
while enabled measured `182.5`, `182.9`, and `187.5 ms/frame`; medians are
`187.6` versus `182.9 ms/frame`, but means are effectively tied at `183.53`
versus `184.30 ms/frame`
(`outputs/sam3-bf16-add-cpy-side-fusion-20260611/seq-r2/`). The profile also
confirms that the hottest `norm2 (copy)` rows are not eliminated by this probe;
they are produced around a different ADD/NORM/affine layout. The next
CUDA-side root attempt should target that ADD/NORM/CPY sequence directly or move
upstream to the ViT matmul/activation dtype contract, not spend more effort on
generic copy-only kernels.

### 2026-06-11 BF16 ADD/NORM/CPY Side-Output Fusion

The next CUDA-side root attempt targeted that ADD/NORM/affine/CPY sequence
directly. The fused ADD/NORM kernel now keeps its existing F32 outputs and can
also write the following BF16/F16 CPY destination as a side output when the CPY
source is the fused ADD/NORM result and the destination has the same shape and
contiguous layout. The disable switch is
`GGML_CUDA_DISABLE_ADD_NORM_CPY_FUSION=1`; candidate diagnostics are available
with `GGML_CUDA_PROFILE_ADD_NORM_CPY_FUSION=1`.

The BF16 profile confirms that this hits the intended hot path. The diagnostic
run reported `72` successful ADD/NORM/CPY fusions and no rejects, including
`sam3_vit_block_00_norm2 -> sam3_vit_block_00_norm2 (copy)`. The previous hot
`norm2 (copy)` rows are no longer emitted as standalone CPY nodes for those
blocks (`outputs/sam3-bf16-add-norm-cpy-fusion-20260611/profile/run-r4.stderr.log`).

Parity is exact against the disabled path on all three checked 6-frame BF16
tracking pairs: each pair has `lhs_rows=6`, `rhs_rows=6`, `min_bbox_iou=1`,
`max_bbox_delta_px=0`, `max_score_abs_delta=0`,
`max_abs_mask_area_rel_delta=0`, and all mask hashes equal
(`outputs/sam3-bf16-add-norm-cpy-fusion-20260611/seq/compare-*.json`).

The warmed sequential A/B result is a small but real default-path improvement
within the noise floor of the full benchmark. With the fusion disabled,
`Track/fr` was `182.0`, `178.6`, and `184.2 ms/frame` (mean `181.6`,
median `182.0`). With the fusion enabled, `Track/fr` was `176.4`, `182.8`, and
`179.3 ms/frame` (mean `179.5`, median `179.3`). That is a mean improvement of
about `1.16%` and a median improvement of about `1.48%`; total runtime improved
from mean `1790.4 ms` to `1775.6 ms`
(`outputs/sam3-bf16-add-norm-cpy-fusion-20260611/seq/`). The optimization is
therefore accepted as a parity-safe root cleanup, but it is not enough by
itself to satisfy the larger same-precision goal against official Python.

The post-fusion hotspot summary still puts ViT matmuls first. On the profiled
BF16 `sam3_encode` run, drop-max synchronized node sums are: `1024x5184`
MUL_MAT `73.154 ms`, `4736x5184` MLP fc1 MUL_MAT `57.880 ms`, `3072x5184`
QKV MUL_MAT `38.349 ms`, window head64 FlashAttention `23.936 ms`, residual
ADD `19.701 ms`, and MLP `GELU_ERF` `15.594 ms`
(`outputs/sam3-bf16-add-norm-cpy-fusion-20260611/profile-summary/hotspots-sam3-encode-r4.json`).
The cuBLASLt timing probe shows the large QKV host setup outlier is the first
call only (`48.187 ms` setup on the first `3072x5184` row, then about
`0.005 ms`), so descriptor/heuristic caching is not the next primary steady-state
win (`outputs/sam3-bf16-cublaslt-timing-20260611/run.stderr.log`).

The existing BF16 graph-level output-chain switches were rechecked after this
fusion and remain rejected for the same-accuracy path. Over two sequential
6-frame A/B pairs, default measured `177.85 ms/frame` mean. With
`SAM3_BF16_VIT_LINEAR_OUTPUT=1`, mean `Track/fr` was `183.5 ms/frame`,
`min_bbox_iou=0.9989772543`, `max_bbox_delta_px=0.032`, and mask hashes changed.
With `SAM3_BF16_VIT_MLP_CHAIN=1`, mean `Track/fr` was `200.3 ms/frame`,
`min_bbox_iou=0.9987126317`, `max_bbox_delta_px=0.048`, and mask hashes changed
(`outputs/sam3-bf16-chain-rerun-20260611/`). These toggles are still useful for
unsafe precision experiments, but they do not advance the official-Python
same-precision win condition.

The cuBLASLt global algorithm-index probe is also rejected. With
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX=1`, mean `Track/fr` regressed from
`178.9` to `185.5 ms/frame` and mask hashes changed. Index `2` also regressed
to `184.25 ms/frame` and changed hashes
(`outputs/sam3-bf16-cublaslt-algo-ab-20260611/`). Shape-specific algorithm
selection is still possible, but a global index override is not a valid root
optimization.

A shape-specific cuBLASLt algorithm-index probe was then added as an opt-in
diagnostic for the four dominant SAM3 BF16 ViT projection groups:
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV`,
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_ATTN_PROJ`,
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1`, and
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2`. A 6-frame same-input sweep
kept JSONL parity exact for every tested candidate, but did not produce a
winner: default was `173.3 ms/frame`, while qkv index `1` was `178.9`, fc1
index `1` was `175.2`, fc2 index `1` was `176.1`, fc2 index `2` was `176.2`,
and attn-proj index `1` was `179.5 ms/frame`
(`outputs/sam3-bf16-cublaslt-shape-algo-sweep-20260611/`). This rejects
heuristic-index selection as the next default optimization; the remaining
projection gap needs a different matmul/dataflow change rather than another
cuBLASLt heuristic pick.

The accepted follow-up is a generic ggml CUDA CPY fast path for contiguous
`F32 -> BF16/F16`. The previous path used one scalar element per thread. The new
path reads `float4` and writes two `nv_bfloat162` or `half2` values when the
source/destination are aligned and the element count is divisible by four. Set
`GGML_CUDA_DISABLE_CPY_F32_LOWP_VEC4=1` to force the old scalar path.

The synchronized CPY profile shows the intended SAM3 BF16 bridge improvement:
`f32->bf16 [64,16,576,9]` moved from `0.06281 ms` mean to `0.02873 ms`, and
`f32->bf16 [1024,24,24,9]` moved from `0.06158 ms` mean to `0.02202 ms`
(`outputs/sam3-bf16-cpy-f32-lowp-vec4-20260611/profile/`). In the normal
6-frame A/B, disabling the fast path measured `Track/fr` `176.8`, `182.3`, and
`178.2 ms/frame` (mean `179.1`, median `178.2`); enabling it measured `179.0`,
`175.0`, and `175.1 ms/frame` (mean `176.37`, median `175.1`). All three pairs
were exact against the scalar path: `min_bbox_iou=1`, `max_bbox_delta_px=0`,
`max_score_abs_delta=0`, `max_abs_mask_area_rel_delta=0`, and all mask hashes
equal (`outputs/sam3-bf16-cpy-f32-lowp-vec4-20260611/seq/`). This is accepted as
a parity-safe root cleanup, but like the ADD/NORM/CPY fusion it is still too
small to close the official-Python gap by itself.

After this cleanup, a fresh synchronized node profile still ranks the dominant
same-precision SAM3 BF16 work as ViT projection matmuls and head64 attention:
`MUL_MAT dst=f32[1024,5184]` at `111.26 ms` drop-max across the 3-frame profile,
`MUL_MAT dst=f32[4736,5184]` at `88.17 ms`,
`MUL_MAT dst=f32[3072,5184]` at `57.85 ms`, window head64
`FLASH_ATTN_EXT` at `36.16 ms`, and global head64 `FLASH_ATTN_EXT` at
`19.69 ms`. The two CPY bridge rows dropped to `2.38 ms` and `1.84 ms`
drop-max respectively
(`outputs/sam3-bf16-cpy-f32-lowp-vec4-20260611/node-profile-current/`). The next
root change should therefore target the ViT projection implementation or a
parity-safe low-precision Q/K/V dataflow, not CPY launch overhead.

### 2026-06-11 SAM3.1 ViT MLP Flat-Chain Probe

The current SAM3.1 `320x240 center@3` BF16/TF32 mask-init contract now has C++ faster than official Python on the small full-frame step, but it is not a full same-output win. The five-repeat current default measured C++ full-frame `383.713361 ms` mean / `381.972120 ms` median, while official Python measured `frame0_cache + add_mask + propagate = 428.110443 ms`. Encoded propagation was `13.075029 ms` versus Python propagation `16.791128 ms`. The mask still differs from the Python mask: C++ hash `6b7661f74121a76a2d3c7cb40727e8eb601dc2ac433a4b975708dd79c619af33`, Python hash `603a3a3fe9550cf9d6b440413e1c79681ca239c7f030ab2fda4146fade8b8516`, IoU `0.9767916695`, and `341` xor pixels (`outputs/sam31-current-default-matrix-20260611d/summary.json`).

The precision/dataflow recheck on the same case did not find a better default. `no-vit-linear-output` was faster in that short run (`379.902199 ms` full-frame mean) but changed the C++ mask and slightly worsened the Python IoU to `0.9764513714`. `vit-qkv-chain`, `vit-mlp-chain`, `vit-qkv-mlp-chain`, and `vit-attn-proj-output-all` were slower or had worse Python IoU (`outputs/sam31-mask-init-matrix-precision-center3-20260611d/summary.json`). The opt-in `GGML_CUDA_ENABLE_CONV_TRANSPOSE_BIAS_FUSION=1` also preserved the same C++ mask and Python IoU, but measured `383.157723 ms` full-frame mean, which is not a robust default-path improvement over the current run (`outputs/sam31-conv-transpose-bias-fusion-20260611d/summary.json`).

An opt-in SAM3 ViT MLP flat-chain probe was added behind `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1`. It keeps FC1 -> GELU -> FC2 in the flattened `[C, tokens]` shape and reshapes back to `[C, W, H, B]` only after FC2. This does not change arithmetic or tensor order. On an isolated SAM3.1 block-15 MLP stage it was raw bit-exact against the default (`max_abs=0`, `mean_abs=0`, `nonzero=0`) and improved the 40-repeat stage mean from `2.198299 ms` to `2.113480 ms` (`outputs/sam31-vit-mlp-flat-chain-stage-20260611d/`). However, the full mask-init A/B did not show a robust encode win: default encode-frame0 mean `174.567042 ms`, flat-chain `174.609415 ms`; default full-frame mean `385.584232 ms`, flat-chain `382.592255 ms`, with the full-frame delta dominated by detection/add-mask noise. The masks and Python IoU stayed identical to default (`outputs/sam31-vit-mlp-flat-chain-ab-20260611d/summary.json`).

The synchronized node profile confirms why this remains an opt-in diagnostic, not a default root win. The dominant SAM3.1 sums are essentially unchanged: default `MUL_MAT dst=bf16[1024,5184]` `133.130240 ms`, `MUL_MAT dst=f32[4736,5184]` `118.011808 ms`, and QKV `MUL_MAT dst=f32[3072,5184]` `77.401472 ms`; flat-chain measured `133.326848 ms`, `118.277920 ms`, and `77.526304 ms` respectively (`outputs/sam31-vit-mlp-flat-chain-node-profile-20260611d/`). The remaining root path is still a projection-matmul or head64-attention implementation change, not reshape-only graph cleanup.

### 2026-06-11 SAM3.1 Direct QKV Views

SAM3.1 ViT attention now uses direct Q/K/V views from the `[3E, tokens, B]` QKV
projection output by default. This removes the previous intermediate
`[E, tokens, B, 3]` materialization before splitting Q, K, and V. The old path
is still available with `SAM3_DISABLE_VIT_DIRECT_QKV_VIEWS=1`; non-SAM3.1
models keep the old default unless `SAM3_ENABLE_VIT_DIRECT_QKV_VIEWS=1` is set.

The accepted five-repeat A/B on `320x240 center@3`, BF16 GGML, official Python
BF16 autocast with TF32 on, is exact against the old C++ path. Both default and
`no-vit-direct-qkv-views` produced mask hash
`6b7661f74121a76a2d3c7cb40727e8eb601dc2ac433a4b975708dd79c619af33`,
`default_iou=1.0`, `default_xor_pixels=0`, and the same Python comparison
(`IoU=0.9767916695`, `xor=341`).

The speed change is large enough to accept as a default SAM3.1 cleanup:

| Path | Full Frame Mean | Full Frame Median | Two-Frame Encode Mean | Evidence |
| --- | ---: | ---: | ---: | --- |
| Default direct QKV views | `371.524964 ms` | `371.380321 ms` | `319.653820 ms` | `outputs/sam31-direct-qkv-default-ab-20260611e/summary.json` |
| Direct QKV disabled | `382.047333 ms` | `381.114367 ms` | `329.877183 ms` | same |

Relative to the disabled path, this is about `2.75%` faster on full-frame mean
and about `3.10%` faster on the two-frame encode mean. Relative to the earlier
pre-change default row (`383.713361 ms` full-frame mean,
`330.435749 ms` two-frame encode mean from
`outputs/sam31-current-default-matrix-20260611d/summary.json`), the current
default is about `3.18%` faster full-frame and about `3.26%` faster on the
two-frame encode mean.

The node profile confirms the intended structural effect: the previous
`CONT dst=f32[1024,576,9,3]` QKV layout materialization hotspot is gone. The
remaining top SAM3.1 work is still projection matmul and head64 attention:
`MUL_MAT dst=bf16[1024,5184]` `133.382848 ms` drop-max,
`MUL_MAT dst=f32[4736,5184]` `118.216864 ms`,
`MUL_MAT dst=f32[3072,5184]` `77.594848 ms`, window head64 attention
`54.319840 ms`, and global head64 attention `27.711552 ms`
(`outputs/sam31-vit-direct-qkv-views-node-profile-20260611e/hotspots.json`).

Two related candidates remain rejected. Removing Q/K contiguous materialization
after permutation is not valid in the current graph because `ggml_reshape_3d`
requires a contiguous tensor at the RoPE boundary. cuDNN SDPA head64 does enter
the SAM3.1 window/global attention shapes when forced, but it converts F32
inputs through BF16, changes the C++ mask hash, worsens the Python mask IoU
slightly, and was slower in the short probe
(`outputs/sam31-cudnn-head64-probe-20260611e/summary.json`). The next root
target remains the projection matmul path or a parity-safe head64 attention
implementation, not this eliminated QKV layout copy.

### 2026-06-11 SAM3 BF16 Direct QKV Views

The same direct Q/K/V view split was rechecked on regular SAM3. It is now the
default for SAM3 BF16 checkpoints and remains opt-in for SAM3 F16. The override
flags are unchanged: `SAM3_DISABLE_VIT_DIRECT_QKV_VIEWS=1` forces the old
materialized split, and `SAM3_ENABLE_VIT_DIRECT_QKV_VIEWS=1` forces the direct
view path.

The accepted A/B used `sam3_benchmark` on the normal `data/test_video.mp4`
contract, `--gpu-only --bbox-only --n-frames 6 --warmup-runs 2`, repeated five
times. BF16 was exact against the old C++ path in every repeat:
`lhs_rows=rhs_rows=6`, `mask_hash_equal_rows=6`, `min_bbox_iou=1.0`,
`max_bbox_delta_px=0`, `max_score_abs_delta=0`, and
`max_abs_mask_area_rel_delta=0`
(`outputs/sam3-direct-qkv-views-paired6f-20260611k/summary.json`).

| Model | Old Default Track Mean | Direct QKV Track Mean | Paired Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `sam3-bf16` | `179.74 ms/frame` | `172.70 ms/frame` | `-7.04 ms` / `-3.91%` | default-on |
| `sam3-f16` | `185.06 ms/frame` | `183.12 ms/frame` | `-1.94 ms` / `-1.02%` mean, but `+1.10 ms` median | keep opt-in |

F16 also remained bit-exact in this A/B, but the paired median was not an
improvement, so defaulting it would add noise rather than a reliable win.

A same-contract Python comparison after the default change keeps SAM3 BF16 on
the C++-faster side for this short row. With the same 6 decoded frames, 1008
model input, text prompt, 2 C++/Python warmups, and BF16+TF32 policy, the
current C++ row measured `174.10 ms/frame` over three repeats. Official Python
SAM3 measured `182.835635 ms/frame`, so Python/C++ is `1.05018x`
(`outputs/model-matrix-sam3-bf16-direct-qkv-default-python-20260611k/summary.json`).
This is a SAM3 BF16 speed result only; all-precision SAM3 and SAM3.1
same-output parity remain open.

### 2026-06-11 SAM3 F16 Same-Contract Recheck

SAM3 F16 is still not an accepted Python win on the same 6-frame contract. With
the same decoded frames, 1008 model input, text prompt, 2 warmups, and
FP16+TF32 policy, C++ measured `183.833333 ms/frame` over three repeats while
official Python measured `182.550495 ms/frame`
(`outputs/model-matrix-sam3-f16-current-python-20260611k/summary.json`). The
comparison is valid (`comparable_speed_claim=true`) but Python/C++ is
`0.99302x`, so the C++ row is about `0.7%` slower by mean. The C++ median was
close (`182.3 ms/frame`) but that is not enough to claim a mean-speed win.

The current profile shows the remaining F16 issue is not a single broken
postprocessing path. Steady propagation is about `12 ms`, while `sam3_encode`
dominates at roughly `155-158 ms` on normal track frames and periodically jumps
to about `190-192 ms` on the fifth tracked frame. Repeating the exact same image
six times preserves the same last-frame jump, so the effect is call-order or
execution scheduling related rather than frame-content dependent
(`outputs/sam3-f16-repeat-frame-profile-20260611k/profile-summary.json`).
Node-level attribution points at the same root shapes as BF16: ViT MLP FC1,
QKV, FC2/attn projection GEMMs and head64 attention
(`outputs/sam3-f16-repeat-frame-node-profile-20260611k/encode-call2-vs-call6.json`).

Several F16 candidates were rechecked on this exact frame-dir contract and are
not defaults:

| Candidate | Result | Decision |
| --- | ---: | --- |
| `SAM3_ENABLE_VIT_DIRECT_QKV_VIEWS=1` | exact JSONL parity, but `+0.96 ms` mean / `+2.4 ms` median versus default | reject |
| F16 linear-output toggles | slower; some also introduce small bbox/score deltas | reject |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1=1` | exact parity, initially promising, but post-change A/B only `-0.96 ms` mean / `-1.3 ms` median versus old index | keep diagnostic |
| `FC1=1` plus `ATTN_PROJ=1` | exact parity, but paired A/B regressed by `+2.52 ms` mean / `+3.5 ms` median | reject |

The ggml CUDA shape-specific cuBLASLt override path now also recognizes F16
SAM3 ViT shapes, so the same `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_*`
diagnostic variables work for F16 and BF16. No F16 shape-specific heuristic is
defaulted yet because the paired data is too noisy and does not reliably close
the Python gap.

A longer same-image 12-frame run confirms that the periodic slowdown is not
caused by input content. With `SAM3_PROFILE=1`, `sam3_encode` calls after the
first initialization measured roughly `154-158 ms` on normal calls, but calls 5
and 10 rose to about `199.6 ms` and `188.6 ms`
(`outputs/sam3-f16-repeat12-profile-20260611l/benchmark.log`). A combined
`SAM3_PROFILE=1 GGML_CUDA_PROFILE_CUBLASLT_BIAS_TIMING=1` run shows the slow
calls are not localized to one kernel: QKV, MLP FC1, MLP FC2, and attention
projection cuBLASLt matmuls all slow down together
(`outputs/sam3-f16-repeat12-profile-cublaslt-20260611l/benchmark.log`).
Concurrent `nvidia-smi` sampling during a warmup run showed SM clocks moving
between about `1972` and `2445 MHz` while power draw was around `110-121 W`
(`outputs/sam3-f16-repeat12-clock-monitor-20260611l/nvidia-smi-samples.txt`).
That makes this outlier pattern more likely to be GPU power/clock behavior than
a single SAM3 kernel bug. It still counts for wall-clock benchmarking, but
cuBLASLt heuristic changes should be judged with paired runs rather than single
outlier runs.

The latest warmup-controlled 12-frame paired run keeps the current F16 default:

| Variant | Track Mean | Track Median | JSONL Parity vs Default | Decision |
| --- | ---: | ---: | ---: | --- |
| Default | `201.133 ms/frame` | `201.1 ms/frame` | reference | keep |
| Forced direct QKV views | `201.700 ms/frame` | `201.7 ms/frame` | exact, `36/36` hash rows | reject |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1=1` | `203.367 ms/frame` | `203.0 ms/frame` | exact, `36/36` hash rows | reject |

Evidence: `outputs/sam3-f16-repeat12-warmup2-paired-env-20260611l`.

### 2026-06-11 SAM3 PCS FPN GPU Copy

The SAM3 text-prompted segmentation head no longer stages detector-neck FPN
features through CPU buffers before launching the PCS segmentation head. The
default path now copies `neck_det_0` and `neck_det_1` directly on the backend
into `seg_fpn0` and `seg_fpn1`; `neck_det_2` is still guarded because the
current graph may leave that input unallocated. Set
`SAM3_DISABLE_PCS_FPN_GPU_COPY=1` to force the old CPU staging path for A/B
checks.

The profile evidence confirms the targeted copy disappeared: the old
`neck_det_0` GPU-to-CPU read plus `seg_fpn0` upload was replaced by a backend
copy of about `0.008 ms`, and `neck_det_1` by about `0.002 ms`
(`outputs/sam3-f16-pcs-fpn-gpu-copy-profile-20260611m/benchmark.log`).

Warmup-controlled 12-frame F16 A/B on the same repeated-frame contract:

| Variant | Track Mean | Track Median | Total Mean | JSONL Parity |
| --- | ---: | ---: | ---: | ---: |
| Default GPU copy | `201.667 ms/frame` | `201.8 ms/frame` | `2647.0 ms` | reference |
| CPU staging disabled-control | `202.167 ms/frame` | `202.1 ms/frame` | `2696.8 ms` | exact, `36/36` hash rows |

Evidence: `outputs/sam3-f16-pcs-fpn-gpu-copy-paired-20260611m`. The track
mean gain is small relative to the observed GPU clock noise, but the total row
improves consistently because the initial PCS segmentation step also avoids the
large FPN CPU round trip. Same-contract official Python comparisons after this
change show:

| Model | C++ Track Mean | Python Track Mean | Ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| `sam3-bf16` | `175.333 ms/frame` | `183.226 ms/frame` | Python/C++ `1.0450x` | C++ still wins |
| `sam3-f16` | `185.067 ms/frame` | `182.592 ms/frame` | Python/C++ `0.9866x` | F16 still open |

Evidence:
`outputs/model-matrix-sam3-bf16-pcs-fpn-gpu-copy-python-20260611m/summary.json`
and
`outputs/model-matrix-sam3-f16-pcs-fpn-gpu-copy-python-20260611m/summary.json`.

### 2026-06-11 SAM3 PCS Image/PE GPU Copy

The PCS text-prompt path also stopped staging the detector `neck_det_2` image
feature and `pe_2` positional encoding through CPU buffers for the geometry,
fusion-encoder, and DETR-decoder inputs. The default path now uses backend
tensor copies into shape-compatible raw 4D graph inputs and reshapes them inside
the graph. Set `SAM3_DISABLE_PCS_IMAGE_GPU_COPY=1` to force the old CPU staging
path.

The profile evidence confirms the targeted `neck_det_2` and `pe_2` reads and
uploads disappeared from the hot path. They were replaced by backend copies of
about `0.002-0.008 ms` per PCS graph input
(`outputs/sam3-f16-pcs-image-gpu-copy-profile-20260611n/benchmark.log`).

Warmup-controlled 12-frame F16 A/B on the same repeated-frame contract:

| Variant | Track Mean | Track Median | Total Mean | JSONL Parity |
| --- | ---: | ---: | ---: | ---: |
| Default GPU copy | `201.633 ms/frame` | `202.0 ms/frame` | `2646.067 ms` | reference |
| CPU staging disabled-control | `202.033 ms/frame` | `202.0 ms/frame` | `2654.433 ms` | exact, `36/36` hash rows |

Evidence: `outputs/sam3-f16-pcs-image-gpu-copy-paired-20260611n`. The track
mean gain is again small compared with run-to-run GPU clock noise, but parity is
exact and the CPU round trip is removed from the profiled PCS path.

Same-contract official Python comparisons after this change:

| Model | C++ Track Mean | Python Track Mean | Ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| `sam3-bf16` | `173.667 ms/frame` | `183.360 ms/frame` | Python/C++ `1.0558x` | C++ still wins |
| `sam3-f16` | `184.933 ms/frame` | `183.432 ms/frame` | Python/C++ `0.9919x` | F16 still open |

Evidence:
`outputs/model-matrix-sam3-bf16-pcs-image-gpu-copy-python-20260611n/summary.json`
and
`outputs/model-matrix-sam3-f16-pcs-image-gpu-copy-python-20260611n/summary.json`.

### 2026-06-11 SAM3 PCS Fusion-Output Copy Probe

The next PCS staging candidate was the fusion-encoder output. The old path reads
`fenc_layer5_out` to CPU and uploads the same tensor to both `ddec_enc` and
`seg_enc`. A direct backend-copy implementation was tested in two forms:

| Variant | Profile Effect | Parity | Same-Contract Result | Decision |
| --- | --- | --- | --- | --- |
| DDEC + segmentation backend copy | removes both `ddec_enc` and `seg_enc` CPU uploads, but keeps the fusion output buffer alive longer | exact, `6/6` hash rows | `189.867 ms/frame` C++ vs `182.994 ms/frame` Python | reject as default |
| DDEC-only backend copy | removes only the `ddec_enc` CPU upload, then reads once for `seg_enc` and releases the fusion graph | exact, `6/6` hash rows | `188.300 ms/frame` C++ vs `183.217 ms/frame` Python | reject as default |
| Default CPU staging | reference | reference | `186.667 ms/frame` C++ vs `182.719 ms/frame` Python | keep |

Evidence:
`outputs/sam3-f16-pcs-fenc-ddec-gpu-copy-profile-20260611q/benchmark.log`,
`outputs/sam3-f16-pcs-fenc-optin-parity-20260611q`,
`outputs/model-matrix-sam3-f16-pcs-fenc-gpu-copy-python-20260611p/summary.json`,
`outputs/model-matrix-sam3-f16-pcs-fenc-ddec-default-python-20260611q/summary.json`,
and
`outputs/model-matrix-sam3-f16-pcs-fenc-default-python-20260611p/summary.json`.

The code keeps the two fusion-output copy paths as opt-in diagnostics:
`SAM3_ENABLE_PCS_FENC_DDEC_GPU_COPY=1` for DDEC-only and
`SAM3_ENABLE_PCS_FENC_SEG_GPU_COPY=1` for the segmentation-side copy. They are
not accepted defaults because same-contract official Python comparison did not
improve, and F16 remains open.

### 2026-06-11 SAM3 F16 Direct-QKV + FATTN MMA Default

The previous F16-open rows are superseded for the current short same-contract
SAM3 comparison. Two changes are now defaulted together:

- ggml CUDA chooses the MMA F16 FlashAttention kernel for named
  `sam3_vit_block` head64 attention. Set
  `GGML_CUDA_DISABLE_SAM3_VIT_FATTN_MMA=1` to force the old dispatch.
- SAM3 F16 checkpoints now use direct Q/K/V views from the `[3E, tokens, B]`
  QKV projection output, matching the existing SAM3.1 and SAM3 BF16 cleanup.
  Set `SAM3_DISABLE_VIT_DIRECT_QKV_VIEWS=1` to force the old materialized split.

The FATTN dispatch change alone is parity-safe and improves C++ relative to its
disable-control, but does not robustly beat official Python by itself:

| Variant | C++ Track Mean | Python Track Mean | Result |
| --- | ---: | ---: | --- |
| MMA default, old materialized QKV | `184.10 ms/frame` | `182.687582 ms/frame` | Python still faster |
| `GGML_CUDA_DISABLE_SAM3_VIT_FATTN_MMA=1` | `188.32 ms/frame` | `183.456901 ms/frame` | C++ slower |

The paired JSONL comparison for the MMA dispatch path is exact: `6/6` mask
hash rows, `max_bbox_delta_px=0`, `max_score_abs_delta=0`, and no mask-area
delta (`outputs/sam3-f16-fattn-mma-default-ab-20260611q/compare.json`).

With direct QKV views also defaulted, the same decoded `960x540` frames, 1008
model input, text prompt `person`, FP16 Python autocast, TF32 on, 2 warmups,
and `per_frame_encode_detect_propagate` scope now have C++ slightly ahead:

| Model | C++ Track Mean | Python Track Mean | Ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| `sam3-f16` | `181.52 ms/frame` | `183.373529 ms/frame` | Python/C++ `1.0102x` | C++ wins on this contract |

Evidence:
`outputs/model-matrix-sam3-f16-default-direct-qkv-mma-20260611r/summary.json`.
The direct-QKV default remained exact against
`SAM3_DISABLE_VIT_DIRECT_QKV_VIEWS=1`: `6/6` mask hash rows,
`min_bbox_iou=1`, `max_bbox_delta_px=0`, `max_score_abs_delta=0`, and no
mask-area delta
(`outputs/sam3-f16-direct-qkv-default-ab-20260611r/compare.json`).

A cublasLt GELU-ERF epilogue probe remains rejected for same-output work:
`GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION=1` changed one of six mask
hashes and moved frame-0 bbox/score slightly (`max_bbox_delta_px=0.015`,
`max_score_abs_delta=0.001165`), so it is not part of the default path
(`outputs/sam3-f16-fattn-mma-default-ab-20260611q/compare-cublaslt-gelu-erf.json`).

### 2026-06-11 SAM3.1 Block15 FATTN Triage

The SAM3.1 `small-center@7` parity gap was rechecked after the SAM3 F16 default
changes. The encoded propagation slice remains faster than official Python, but
the end-to-end mask still does not match:

| Scope | C++ | Python | Result |
| --- | ---: | ---: | --- |
| encoded propagation | `13.024503 ms` | `16.702685 ms` | C++ faster |
| full-frame C++ mask-init | `372.285129 ms` | n/a | mask hash differs |

Evidence: `outputs/sam31-mask-init-matrix-current-after-sam3-f16-default-20260611r/summary.json`.
The current mask has Python IoU `0.8489959839` and `752` XOR pixels, so this is
not yet parity-complete.

After the block-test helper fix, the production smoke was repeated and remained
in the same state: C++ encoded propagation mean `12.725128 ms`, Python
propagation `17.583287 ms`, Python IoU `0.8489959839`, and `752` XOR pixels
(`outputs/sam31-mask-init-matrix-after-helperfix-20260611s/summary.json`).

State comparison shows the large visible drift is accumulated before and around
the first late global block rather than caused by a single broken block15
kernel. Selected current C++ vs official Python state errors:

| Tensor | Max abs | Mean abs | RMSE |
| --- | ---: | ---: | ---: |
| `vit_block_12_out` | `1.964745` | `0.008180` | `0.012383` |
| `vit_block_13_out` | `4.302525` | `0.009434` | `0.015361` |
| `vit_block_14_out` | `12.339802` | `0.010729` | `0.019846` |
| `vit_block_15_out` | `25.136677` | `0.012983` | `0.027389` |
| `vit_output` | `97.974163` | `0.023008` | `0.088335` |

Evidence:
`outputs/sam31-block00-15-current-compare-20260611s/summary.json`,
`outputs/sam31-block12-14-stage-current-compare-20260611s/summary.json`, and
`outputs/sam31-mask-init-state-compare-after-sam3-f16-default-20260611r/summary.json`.

The `sam31_vit_block_case` attention-core helper was corrected to follow the
production Q/K/V split path and to name the FlashAttention tensor before graph
execution. This makes targeted selector probes such as
`GGML_CUDA_FORCE_FATTN_VEC_MATCH=sam3_vit_block_15_global_attn` valid for block
unit tests.

With the corrected helper and Python's own block15 QKV tensor as input, block15
attention itself favors the default MMA path:

| Variant | Median | Max abs vs Python | Mean abs vs Python | Decision |
| --- | ---: | ---: | ---: | --- |
| default MMA | `4.568296 ms` | `0.052458` | `0.001222` | keep |
| VEC forced by match | `23.169163 ms` | `0.193381` | `0.006627` | reject |
| TILE forced by match | `23.401499 ms` | `0.193381` | `0.006627` | reject |

Evidence:
`outputs/sam31-vit-block15-attncore-default-after-helperfix-20260611s/compare-python.json`,
`outputs/sam31-vit-block15-attncore-vec-match-after-helperfix-20260611s/compare-python.json`,
and
`outputs/sam31-vit-block15-attncore-tile-match-after-helperfix-20260611s/compare-python.json`.
Earlier full-mask runs where block15 VEC made one case slightly closer are
therefore treated as downstream compensation, not a root parity fix.

The next SAM3.1 work should focus on the accumulated BF16/F32 linear-output
contract across window blocks 12-14 and the residual/MLP amplification path.
When Python stage inputs are injected directly, block14 operators are already
close to the Python stage outputs (`attn_core mean_abs=0.000567`,
`attn_proj mean_abs=0.000431`, `mlp_fc2 mean_abs=0.000137`), so broad FATTN
fallback is not an accepted optimization path
(`outputs/sam31-block14-stage-isolated-20260611s/compare-python.json` and
`outputs/sam31-block14-stage-isolated-20260611s/compare-qkv-python.json`).

### 2026-06-11 SAM3.1 Full-Block Error-Curve Triage

The block diagnostic runner now accepts `--stage block`, which runs the full
production ViT block from an injected tensor instead of only one internal stage.
The mask-init state dump path also accepts
`SAM31_MASK_INIT_DUMP_ALL_VIT_BLOCKS=1` to dump every `vit_block_##_out` tensor
from both official Python and C++.

Running full C++ blocks from Python block inputs shows the individual block
operator contract is much closer than the production chained state:

| Block | Isolated mean abs | Isolated max abs | Production mean abs | Production max abs |
| --- | ---: | ---: | ---: | ---: |
| 00 | `0.001944` | `0.099248` | `0.001946` | `0.099176` |
| 01 | `0.001351` | `0.080784` | `0.003136` | `0.248378` |
| 02 | `0.001117` | `0.085377` | `0.003945` | `0.257683` |
| 08 | `0.001840` | `0.061771` | `0.007914` | `0.271697` |
| 12 | `0.001648` | `0.077366` | `0.008180` | `1.964745` |
| 14 | `0.001678` | `0.157061` | `0.010729` | `12.339802` |

Evidence:
`outputs/sam31-block-full-from-python-20260611t/compare-python-00-14.json`
and `outputs/sam31-mask-init-state-allblocks-compare-20260611t/vit-blocks.json`.
This means the remaining parity gap is mostly a chained numerical-contract
accumulation problem rather than a single broken block implementation.

The all-block production curve confirms the drift grows gradually through the
first window-block group, then max error is amplified by later residual/MLP and
global blocks:

| Block | Mean abs | Max abs |
| --- | ---: | ---: |
| 00 | `0.001946` | `0.099176` |
| 03 | `0.005016` | `0.251515` |
| 07 | `0.007702` | `0.195433` |
| 12 | `0.008180` | `1.964745` |
| 15 | `0.012983` | `25.136677` |
| 23 | `0.020114` | `128.466442` |
| 31 | `0.023008` | `97.974163` |

Two broad precision-policy probes were rejected as defaults:

| Variant | Effect on all-block curve | Mask-init side effect | Decision |
| --- | --- | --- | --- |
| `SAM3_BF16_VIT_QKV_CHAIN=1` | Similar mean error; lowers some late max errors, e.g. block31 max `97.974 -> 84.320` | score changed to `0.189252`; previous matrix showed no robust IoU win | reject as broad default |
| `SAM3_DISABLE_BF16_VIT_LINEAR_OUTPUT=1` | Lowers mean and late max, e.g. block31 mean `0.023008 -> 0.021557`, max `97.974 -> 88.682` | score changed to `0.164998`; previous matrix showed worse small-center IoU | reject as broad default |

Evidence:
`outputs/sam31-mask-init-state-allblocks-compare-20260611t/vit-blocks-qkv-chain.json`
and
`outputs/sam31-mask-init-state-allblocks-compare-20260611t/vit-blocks-no-linear-output.json`.
The next implementation target should be narrower: match PyTorch autocast
boundaries inside the first window-block group without globally forcing all ViT
linear outputs to F32 or BF16.

### 2026-06-11 SAM3.1 Targeted ViT Block Precision Probe

Added diagnostic-only block selectors so precision/dataflow probes can be scoped
without changing the default path:

- `SAM3_BF16_VIT_QKV_CHAIN_BLOCKS=<selector>` enables QKV BF16 output for selected ViT blocks.
- `SAM3_BF16_VIT_MLP_CHAIN_BLOCKS=<selector>` enables MLP FC1 BF16 output for selected ViT blocks.
- `SAM3_BF16_VIT_LINEAR_OUTPUT_BLOCKS=<selector>` and
  `SAM3_DISABLE_BF16_VIT_LINEAR_OUTPUT_BLOCKS=<selector>` override the ViT linear-output policy for
  selected blocks.
- `SAM3_BF16_VIT_ATTN_PROJ_OUTPUT_BLOCKS=<selector>` and
  `SAM3_DISABLE_BF16_VIT_ATTN_PROJ_OUTPUT_BLOCKS=<selector>` override attention projection output
  policy for selected blocks.
- `SAM31_MASK_INIT_DUMP_VIT_STAGE_BLOCKS=<selector>` dumps internal ViT block stage tensors only for
  selected blocks. This is intentionally separate from `SAM31_MASK_INIT_DUMP_ALL_VIT_BLOCKS=1`,
  because dumping every stage for every block can exceed GPU memory in official Python.

The selector syntax is comma-separated block indices/ranges, for example `0`, `1-7`, `0,12,31`, or
`all`. Empty means no selected blocks for stage dump. The earlier parser treated `0` as disabled;
that was wrong for range selectors and has been fixed so block 0 can be probed directly.

Stage evidence on `320x240 small-center@7`, BF16 autocast, TF32 on:

| Boundary | Mean abs | Max abs | Evidence |
| --- | ---: | ---: | --- |
| block0 `norm1` | `0.000000217` | `0.018629` | `outputs/sam31-mask-init-state-stage-compare-20260611u/vit-stage-b00-07.json` |
| block0 `qkv` | `0.001644586` | `0.060065` | same |
| block0 `mlp_fc1` | `0.001610066` | `0.069773` | same |
| block0 `out` | `0.001945849` | `0.099176` | same |
| block1 `out` | `0.003140264` | `0.248378` | same |
| block7 `out` | `0.007724363` | `0.255455` | same |

The first material gap therefore appears immediately after the almost-exact block0 norm, at the
BF16-weight linear boundaries (`qkv` and MLP). Forcing block0 QKV/MLP outputs closer to Python's
BF16 autocast contract reduces stage mean error, but it does not improve final mask parity:

| Variant | block0 qkv mean | block0 fc1 mean | block0 out mean | block7 out mean | Python IoU | XOR px | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default | `0.001644586` | `0.001610066` | `0.001945849` | `0.007724363` | `0.848995984` | `752` | baseline |
| `SAM3_BF16_VIT_QKV_CHAIN_BLOCKS=0` | `0.000999916` | `0.001594593` | `0.001910974` | `0.007685706` | `0.846785500` | `765` | reject |
| `SAM3_BF16_VIT_MLP_CHAIN_BLOCKS=0` | `0.001644586` | `0.001226951` | `0.001911412` | `0.007697469` | `0.833860759` | `840` | reject |
| `SAM3_BF16_VIT_QKV_CHAIN_BLOCKS=0 SAM3_BF16_VIT_MLP_CHAIN_BLOCKS=0` | `0.000999916` | `0.001207279` | `0.001875453` | `0.007633174` | `0.847233360` | `762` | reject |
| `SAM3_BF16_VIT_QKV_CHAIN_BLOCKS=0-7` | not stage-rerun | not stage-rerun | not stage-rerun | not stage-rerun | `0.848825537` | `753` | reject |
| `SAM3_BF16_VIT_QKV_CHAIN_BLOCKS=0-7 SAM3_BF16_VIT_MLP_CHAIN_BLOCKS=0-7` | not stage-rerun | not stage-rerun | not stage-rerun | not stage-rerun | `0.848424011` | `755` | reject |

Evidence:
`outputs/sam31-mask-init-state-stage-compare-20260611v/` and
`outputs/sam31-mask-init-matrix-block-chain-b0-b07-20260611v/summary.json`.
The result is important because it shows that local tensor closeness at early ViT boundaries is not a
sufficient acceptance criterion. Any further precision-policy change must be accepted by the same
Python-mask IoU/XOR matrix, not by per-stage mean error alone.

### 2026-06-11 SAM3.1 BF16-dst cuBLASLt Algorithm Probe

The cuBLASLt ViT shape-specific algorithm selectors previously only matched `dst=F32`, so they did
not isolate the BF16-destination probes introduced for block-scoped QKV/MLP chain tests. The CUDA
backend now also checks low-precision destination tensors and supports destination-specific override
names first:

- `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV_BF16_DST`
- `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_ATTN_PROJ_BF16_DST`
- `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1_BF16_DST`
- `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2_BF16_DST`

The generic `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_*` variables remain the fallback. The same
suffix mechanism is also available for `_F16_DST`.

A profile check confirmed the intended isolation: with
`SAM3_BF16_VIT_QKV_CHAIN_BLOCKS=0` and
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV_BF16_DST=1`, block0 QKV reported
`dst_type=bf16`, `ne=[3072,5184,1,1]`, `algo_index=1`, while the following F32 QKV shapes kept
`algo_index=0` (`outputs/sam31-cublaslt-bf16dst-qkv-algo-probe-20260611w/run.log`).

The mask-level acceptance sweep on `320x240 small-center@7`, BF16 autocast, TF32 on, 1 repeat:

| Variant | Python IoU | XOR px | Full step ms | Decision |
| --- | ---: | ---: | ---: | --- |
| default | `0.848995984` | `752` | `695.626850` | baseline |
| `vit-qkv-chain-b0` | `0.846785500` | `765` | `703.497326` | reject |
| `vit-qkv-chain-b0-qkv-bf16-algo1` | `0.846785500` | `765` | `684.149017` | same mask, not parity win |
| `vit-qkv-chain-b0-7` | `0.848825537` | `753` | `693.319065` | reject |
| `vit-qkv-chain-b0-7-qkv-bf16-algo1` | `0.848825537` | `753` | `698.735438` | same mask, not parity win |
| `vit-mlp-chain-b0` | `0.833860759` | `840` | `712.097561` | reject |
| `vit-mlp-chain-b0-fc1-bf16-algo1` | `0.833860759` | `840` | `693.095103` | same mask, not parity win |
| `vit-mlp-chain-b0-7` | `0.837070847` | `821` | `711.637797` | reject |
| `vit-mlp-chain-b0-7-fc1-bf16-algo1` | `0.837070847` | `821` | `716.411196` | same mask, not parity win |

Evidence: `outputs/sam31-mask-init-matrix-bf16dst-algo-block-chain-20260611w/summary.json`.
The low-precision-destination algorithm override is useful for diagnosis and future perf sweeps, but
it does not solve SAM3.1 parity and should not become a default-path change.

### 2026-06-11 SAM3.1 Memory-Attention Input Boundary Probe

Fresh propagation boundary dumps on `320x240 center@3`, BF16 autocast, TF32 on confirmed that mask
selection itself is not the remaining mismatch. `sam31_mux_iou` argmax matched between official
Python and C++ for every multiplex slot, including the output slot 0. The selected low-res logit
slice for mux slot 0 still differed by `mean_abs=0.133729` and `max_abs=1.066491`, so the mismatch is
already in the logits before thresholding, not in final mask selection.

The same dump shows the first propagation-stage gap is already present at `sam31_mem_attn_input`:
default C++ versus official Python measured `mean_abs=0.009097`, `max_abs=0.496086`. The error then
propagates through memory attention and the two-way decoder to `sam31_mux_masks` with
`mean_abs=0.058980`, `max_abs=1.066491`
(`outputs/sam31-prop-boundary-20260611x/decoder-boundary-summary.json`).

An opt-in diagnostic, `SAM31_BF16_MEM_ATTN_INPUT_BOUNDARY=1`, was added to round only the external
memory-attention inputs and the initial `src + 0.1 * src_pos` boundary to BF16 while keeping the
existing F32 internal memory-attention path. This slightly reduced tensor-level differences
(`sam31_mem_attn_input mean_abs 0.009097 -> 0.008977`, `sam31_mux_masks mean_abs 0.058980 ->
0.058646`) but did not improve the accepted mask metric. In a 3-repeat matrix on the same case,
default C++ had Python IoU `0.9767916695`, XOR `341`, and encoded propagation mean `13.623238 ms`;
the input-boundary variant had Python IoU `0.9754901961`, XOR `360`, and encoded propagation mean
`14.485634 ms`
(`outputs/sam31-mask-init-matrix-memattn-input-boundary-20260611x/summary.json`).

Decision: keep `SAM31_BF16_MEM_ATTN_INPUT_BOUNDARY=1` as a diagnostic variant only. It is not a
same-parity speedup and should not become the default.

### 2026-06-11 SAM3.1 Propagation Feature Boundary Probe

The propagation input split dump on `320x240 center@3`, BF16/TF32, narrowed the first visible
tracking mismatch to the frame-1 propagation feature cache rather than positional encoding or mask
selection. Official Python versus C++ default:

| Boundary | Mean abs | Max abs | RMSE | Evidence |
| --- | ---: | ---: | ---: | --- |
| `sam31_prop_image_raw` | `0.00905967` | `0.494523` | `0.0142451` | `outputs/sam31-prop-feature-split-20260611y/feature-split-summary.json` |
| `sam31_mem_attn_src` | `0.00905967` | `0.494523` | `0.0142451` | same |
| `sam31_prop_image_pe` | `0.000345299` | `0.00195235` | `0.000626146` | same |
| `sam31_prop_projected_s0` | `0.00167697` | `0.114613` | `0.00272205` | same |
| `sam31_prop_projected_s1` | `0.00326365` | `0.152733` | `0.00517668` | same |

This means the large `sam31_mem_attn_input` gap is dominated by `src`/frame-1 image features, while
the PE term is much smaller. A Python `--dtype fp32 --tf32 on` rerun produced the same relevant
feature dump and mask as the BF16 run on this case, so the mismatch is not explained by Python
autocast labeling alone
(`outputs/sam31-prop-feature-fp32-python-20260611y/cpp-default-vs-python-fp32-summary.json`).

An opt-in diagnostic, `SAM31_BF16_PROPAGATION_FEATURE_BOUNDARY=1`, was added to round SAM3.1
propagation neck outputs through `F32 -> BF16 -> F32` before storing them in the C++ tracking state.
It only slightly changed tensor-level differences:

| Boundary | Default mean abs | Feature-boundary mean abs | Result |
| --- | ---: | ---: | --- |
| `sam31_prop_image_raw` | `0.00905967` | `0.00897435` | tiny improvement |
| `sam31_mem_attn_input` | `0.00909734` | `0.00906879` | tiny improvement |
| `sam31_mem_attn_output` | `0.04587837` | `0.04586857` | negligible |
| `sam31_mux_masks` | `0.05898019` | `0.05874650` | tiny improvement |

The mask/perf matrix rejects it as a default: default C++ had Python IoU `0.9767916695`, XOR `341`,
encoded propagation mean `13.253836 ms`, and two-frame encode mean `318.868408 ms`; the
feature-boundary variant had Python IoU `0.9761078211`, XOR `351`, encoded propagation mean
`12.963993 ms`, and two-frame encode mean `321.725149 ms`
(`outputs/sam31-mask-init-matrix-prop-feature-boundary-20260611y/summary.json`). The correct next
root target remains the SAM3.1 ViT/propagation-neck CUDA dataflow, not a state-cache boundary cast.

The follow-up frame-1 encode boundary dump added `frame0_`/`tracking_` prefixed C++ encode dumps and
Python dumps for the raw propagation FPN outputs before `sam_mask_decoder.conv_s0/conv_s1`. On the
same `320x240 center@3`, BF16/TF32 contract:

| Boundary | Mean abs | Max abs | RMSE | Evidence |
| --- | ---: | ---: | ---: | --- |
| `tracking_sam31_prop_raw_s0` | `0.000610337` | `0.0752438` | `0.00105043` | `outputs/sam31-frame1-encode-boundary-20260611z/raw-neck-boundary-summary.json` |
| `tracking_sam31_prop_raw_s1` | `0.00295697` | `0.175960` | `0.00460401` | same |
| `tracking_sam31_prop_raw_s2` | `0.00905967` | `0.494523` | `0.0142451` | same |
| `sam31_prop_projected_s0` | `0.00167697` | `0.114613` | `0.00272205` | `outputs/sam31-frame1-encode-boundary-20260611z/prop-boundary-summary.json` |
| `sam31_prop_projected_s1` | `0.00326365` | `0.152733` | `0.00517668` | same |
| `sam31_prop_image_raw` | `0.00905967` | `0.494523` | `0.0142451` | same |

This rules out the later high-resolution projection convs as the main source. `raw_s2` is already
identical to the later `sam31_prop_image_raw`/`sam31_mem_attn_src` gap, while `raw_s0` is much
closer. The next implementation target should therefore be the SAM3.1 top-level ViT-to-propagation
neck path, especially the scale-2 `conv_1x1/conv_3x3` path and the ViT output feeding it. A useful
next A/B is to isolate scale-2 propagation neck from a fixed Python ViT tensor before adding more
memory-attention or decoder changes.

The isolated propagation-neck case confirmed that the neck itself is not the dominant mismatch. With
a fixed random ViT tensor at feature size 72, Python checkpoint weights versus the BF16 GGML model
measured:

| Tensor | Mean abs | Max abs | RMSE | Evidence |
| --- | ---: | ---: | ---: | --- |
| `image_features` | `0.000753167` | `0.00634766` | `0.000953859` | `outputs/sam31-prop-features-isolated-20260611aa/compare.json` |
| `projected_s0` | `0.000311527` | `0.00450957` | `0.000413705` | same |
| `projected_s1` | `0.000248108` | `0.00398576` | `0.000341635` | same |

These are more than 10x smaller than the live frame-1 `raw_s2` mean gap (`0.00905967`), so the
remaining root target is the ViT trunk output/dataflow feeding the propagation head, not
`conv_s0`/`conv_s1` or the feature cache.

The Python runner now also dumps a direct-backbone diagnostic path from the exact preprocessed
frame-1 tensor. That direct path exactly matched the official Python feature cache for raw
propagation features:

| Boundary | Mean abs | Max abs | Evidence |
| --- | ---: | ---: | --- |
| Python cache vs direct `raw_s0` | `0.0` | `0.0` | `outputs/sam31-frame1-encode-boundary-20260611ab/direct-boundary-summary.json` |
| Python cache vs direct `raw_s1` | `0.0` | `0.0` | same |
| Python cache vs direct `raw_s2` | `0.0` | `0.0` | same |

C++ debug dumps now include `frame0_`/`tracking_` prefixes for ViT block outputs as well as the
top-level ViT output, avoiding accidental frame0/frame1 overwrite during diagnostics. On frame 1,
C++ versus Python direct backbone showed smooth ViT trunk error accumulation rather than a single
cache boundary jump:

| Boundary | Mean abs | Max abs | RMSE | Evidence |
| --- | ---: | ---: | ---: | --- |
| `vit_block_00_out` | `0.00193468` | `0.0803165` | `0.00305041` | `outputs/sam31-frame1-encode-boundary-20260611ad/frame1-vit-prop-boundary-summary.json` |
| `vit_block_07_out` | `0.00764832` | `0.172585` | `0.0101415` | same |
| `vit_block_15_out` | `0.0130657` | `10.0930` | `0.0245573` | same |
| `vit_block_23_out` | `0.0197216` | `56.4216` | `0.0601760` | same |
| `vit_block_31_out` / `vit_output` | `0.0225625` | `40.4163` | `0.0544875` | same |
| `tracking_sam31_prop_raw_s2` | `0.00905967` | `0.494523` | `0.0142451` | same |

Performance on the same `320x240 center@3`, BF16/TF32, one warmup, no debug dump:

| Implementation | Frame0/cache ms | Add mask ms | Frame1/cache ms | Propagate ms | Comparable total |
| --- | ---: | ---: | ---: | ---: | ---: |
| C++ GGML/CUDA | `168.367513` | `45.067933` | `150.341697` | `15.608528` | `379.385671` |
| Official Python | `400.627730` | `9.958516` | `0.408099` | `16.921483` | `427.915828` |

The Python frame-1 cache number is low because the official distributed path prefetches/buffers the
next frame during the frame-0 cache step. Even with that different scheduling, C++ is about `1.13x`
faster for the comparable two-frame mask-init + frame-1 propagation workflow
(`outputs/sam31-perf-profile-20260611ae`). The remaining performance target is still ViT encode:
node profiling ranks ViT `MUL_MAT` kernels first, followed by window/global FlashAttention and GELU
(`outputs/sam31-perf-profile-20260611ae/sam31-encode-hotspots.json`).

An A/B sweep of BF16 ViT output-policy toggles did not find a safe default speedup. Disabling all
BF16 ViT linear outputs was a few milliseconds faster in one run but worsened mask parity
(`IoU 0.9767916695 -> 0.9764513714`, XOR `341 -> 346`), so the default remains unchanged.

### 2026-06-11 cuBLASLt Algo Cache Probe

The SAM3.1 BF16 ViT hotspot is still dominated by `MUL_MAT + bias` shapes. A shape-specific
cuBLASLt heuristic-index sweep on `320x240 center@3`, warmup 5, kept exact default mask hashes but
did not beat the default heuristic. The checked non-defaults (`qkv=1`, `fc1=1`, `fc2=1/2`,
`attn_proj=1`, and two small combos) all regressed the comparable total by roughly `+3.0 ms` to
`+12.7 ms` versus default (`outputs/sam31-cublaslt-shape-ab-20260611ah`). Keep those environment
knobs diagnostic-only.

A smaller accepted change caches the selected cuBLASLt `MatmulAlgo` by device, dtype, epilogue,
workspace, batch/stride, and matrix shape. It is output-preserving and can be disabled with
`GGML_CUDA_DISABLE_CUBLASLT_BIAS_ALGO_CACHE=1`. In a 5-run A/B:

| Variant | Comparable total mean | Median | Mask hash | Evidence |
| --- | ---: | ---: | --- | --- |
| Algo cache enabled | `390.976839 ms` | `390.996214 ms` | same | `outputs/sam31-cublaslt-cache-ab-20260611ai` |
| Algo cache disabled | `392.810113 ms` | `393.144495 ms` | same | same |

The effect is intentionally small. Profiling showed cache hits on `1207/1230` cuBLASLt bias calls
and reduced aggregate host setup from `118.488010 ms` to `109.062260 ms` in the profiled run
(`outputs/sam31-cublaslt-cache-profile-20260611aj`). It does not change the main bottleneck:
ViT matmul kernel time and tensor layout remain the next root targets.

After the cache change, the same Python-reference matrix remained stable. For `320x240 center@3`,
BF16/TF32, C++ warmup 5 and Python warmup 1:

| Implementation | Mean comparable total | Python IoU / XOR | Evidence |
| --- | ---: | ---: | --- |
| C++ GGML/CUDA | `390.131173 ms` | `0.9767916695 / 341` | `outputs/sam31-cache-python-parity-20260611ak/summary.json` |
| Official Python | `431.045208 ms` | reference | same |

This single SAM3.1 row is still faster than official Python under the measured contract, but the
broader goal is not complete until the remaining model/precision matrix and stronger ViT parity
checks are finished.

### 2026-06-11 SAM3 Same-Contract Refresh

The current goal is still blocked on SAM3, not SAM3.1. With the same decoded
`960x540` source frames, same 3-frame range, same prompt (`person` at
`315,250`), same `1008` model input, same frame 1..N-1 encode+propagate
tracking scope with frame-0 prompt/add excluded, and two warmups, official
Python remains faster:

| Row | C++ track mean | C++ median | Python track mean | Python median | Python/C++ ratio | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SAM3 F16 / Python FP16+TF32 | `177.733333 ms` | `171.0 ms` | `120.544647 ms` | `120.238318 ms` | `0.678233` | `outputs/model-matrix-sam3-f16-current-20260611al/summary.json` |
| SAM3 BF16 / Python BF16+TF32 | `178.0 ms` | `183.9 ms` | `120.729806 ms` | `120.454088 ms` | `0.678257` | `outputs/model-matrix-sam3-bf16-current-20260611am/summary.json` |
| SAM3 BF16 refreshed 5-run default | `169.08 ms` | `166.0 ms` | `120.758296 ms` | `120.043270 ms` | `0.714208` | `outputs/model-matrix-sam3-bf16-pair-vec2-20260611at/summary.json` |
| SAM3 BF16 contract refresh with preprocessing policy | `165.266667 ms` | `164.9 ms` | `119.999425 ms` | `119.627957 ms` | `0.726096` | `outputs/model-matrix-sam3-bf16-contract-refresh-20260611/summary.json` |
| SAM3 F16 contract refresh with preprocessing policy | `169.466667 ms` | `169.5 ms` | `120.635821 ms` | `121.052602 ms` | `0.711856` | `outputs/model-matrix-sam3-f16-contract-refresh-20260611/summary.json` |

Note: older JSON rows use the label `per_frame_encode_detect_propagate`, but
the actual measured interval on both sides is frame 1..N-1 propagation after
the frame-0 prompt/add step. The comparison script now labels new SAM3 rows as
`frame1_plus_encode_propagate_prompt_excluded`.

The refreshed default run shows that normal benchmark variance is large enough
to move the C++ mean by about 9 ms, but not enough to change the conclusion:
C++ is still roughly `1.40x` slower than official Python on the comparable SAM3
BF16 row.

The current goal-loop rerun on the same 3-frame SAM3 BF16 contract keeps that
conclusion. With `models/sam3`, 960x540 decoded frames, 1008 model input,
prompt `person`, frame-0 prompt/add excluded, two C++ and two Python warmups,
and official Python SAM3 BF16 autocast with TF32 enabled, C++ measured
`170.0 ms/frame` mean / `163.3 ms/frame` median while official Python measured
`120.379702 ms/frame` mean / `120.252197 ms/frame` median. The comparison row is
marked `comparable_speed_claim=true` and reports Python/C++ `0.708116x`, so C++
is still about `1.41x` slower on this strict contract
(`outputs/model-matrix-sam3-bf16-current-20260612-goal-py/summary.json`). A
preceding failed attempt used the default `models/matrix-all` directory, which
does not contain `sam3-bf16`; the valid rerun explicitly set `SAM3_MODELS_DIR`
to `models/sam3`.

The current BF16 CUDA profile still points at the same root work. In the
synchronized node profile, `sam3_encode` dominates tracking and the top
drop-max hotspots are ViT projection matmuls, head64 attention, residual adds,
and GELU: `MUL_MAT dst=f32[1024,5184]` `336.251104 ms`,
`MUL_MAT dst=f32[4736,5184]` `267.849024 ms`,
`MUL_MAT dst=f32[3072,5184]` `175.126560 ms`, window head64 attention
`122.152960 ms`, residual ADD `[1024,72,72]` `83.626912 ms`, GELU
`67.359648 ms`, and global head64 attention `65.396096 ms` across the profiled
encode calls (`outputs/sam3-bf16-current-profile-20260611an/encode-hotspots.json`).

The 2026-06-12 profile refresh fixes the hotspot summarizer to select the log
that actually contains CUDA node-profile groups instead of assuming the first
log in a combined summary. The regenerated `sam3_encode` summary now records
the selected source log and reports the same bottleneck order on the current
build: `MUL_MAT dst=f32[1024,5184]` `109.980064 ms` drop-max,
`MUL_MAT dst=f32[4736,5184]` `87.426560 ms`,
`MUL_MAT dst=f32[3072,5184]` `57.196992 ms`, window head64 attention
`40.546176 ms`, residual ADD `[1024,72,72]` `28.990592 ms`, GELU
`23.370720 ms`, and global head64 attention `20.349824 ms`
(`outputs/sam3-bf16-current-profile-20260612a/hotspots-sam3-encode.json`).
These are synchronized profiling sums, not normal runtime totals, but they keep
the root conclusion unchanged: the remaining SAM3 gap is still projection/GEMM
and head64 attention dataflow, not decode, preprocessing, graph launch, or a
single missing copy kernel.

A 2026-06-12 cuBLASLt fusion check also rules out a missing default-path
projection-bias fusion. With `GGML_CUDA_PROFILE_CUBLASLT_BIAS_FUSION=1`, the
current SAM3 BF16 run reported successful bias fusion for the ViT QKV,
attention projection, MLP FC1, and MLP FC2 shapes; there were no reject rows in
the short current-default log
(`outputs/sam3-bf16-cublaslt-fusion-profile-20260612a/fusion.log`). Remaining
GEMM work is therefore kernel/algorithm/dataflow quality, not simply wiring the
existing bias epilogue to more default nodes.

The FATTN profile confirms that attention alone cannot close the gap. For the
profiled run, SAM3 ViT window attention totalled `126.075 ms` across 252 calls
and global attention totalled `68.518 ms` across 36 calls. That is meaningful,
but the remaining same-contract SAM3 gap is larger than a single attention
toggle can cover (`outputs/sam3-bf16-fattn-profile-20260611ar/benchmark.log`).

The 2026-06-12 FATTN refresh adds `scripts/summarize_fattn_profile.py` so the
CUDA event profile can be grouped consistently. On the current BF16 default,
SAM3 ViT head64 FATTN still receives F32 Q/K/V: window calls total
`12.435712 ms` Q/K/V conversion and `29.674880 ms` kernel time, while global
calls total `1.788224 ms` conversion and `20.627168 ms` kernel time
(`outputs/sam3-bf16-fattn-profile-20260612a/fattn-summary.json`). Forcing
`SAM3_BF16_VIT_QKV_CHAIN=1` changes the SAM3 ViT FATTN inputs to BF16, but the
current MMA path converts Q to F32 and K/V to F16, so window Q/K/V conversion
increases to `23.267008 ms`; the normal short benchmark also worsens from
`165.7` to `175.3 ms/frame`
(`outputs/sam3-bf16-qkv-chain-fattn-profile-20260612a/`). This makes a native
BF16 head64 FATTN path a valid root experiment, but not a complete answer by
itself: even removing the extra BF16 conversion would mostly recover the QKV
chain regression, while the Python gap still requires a faster projection/GEMM
or larger ViT block dataflow change.

Three current probes were rejected as defaults:

| Probe | C++ track mean | Reason | Evidence |
| --- | ---: | --- | --- |
| `GGML_CUDA_ENABLE_FATTN_F32_TO_F16_PAIR_VEC2=1` | `177.62 ms` | slower than refreshed default `169.08 ms` | `outputs/model-matrix-sam3-bf16-pair-vec2-enabled-20260611au/summary.json` |
| `GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION=1` | `176.08 ms` | slower than refreshed default and still far behind Python | `outputs/model-matrix-sam3-bf16-cublaslt-geluerf-20260611aw/summary.json` |
| Force SAM3 ViT FATTN tile/vector kernels | `235.33 ms` / `235.67 ms` | much slower than MMA default; disabling the SAM3 ViT MMA selector was not better either | `outputs/sam3-bf16-fattn-kernel-select-ab-20260611ax` |
| Existing SAM3 BF16 precision/toggle sweeps | no win | direct QKV, CUDA graphs, MLP flat-chain, QKV/MLP chain, output-policy, and cuDNN head64 probes did not produce a safe speedup | `outputs/sam3-bf16-existing-toggles-ab-20260611ao`, `outputs/sam3-bf16-precision-policy-ab-20260611aq`, `outputs/sam3-bf16-cudnn-head64-ab-20260611ap` |

The next root target should therefore be a real CUDA dataflow change rather
than another environment toggle: either a parity-safe low-precision ViT block
path that avoids repeated F32 materialization around projection/GELU/residual,
or a native head64 attention path closer to PyTorch flash-attention for SAM3
window blocks. cuBLASLt heuristic retuning and standalone copy/convert kernels
are now too small to make SAM3 beat official Python under the same contract.

The 2026-06-12 node-name profile pass adds
`scripts/summarize_cuda_node_names.py`, which parses raw
`GGML_CUDA_PROFILE_NODE` logs directly and groups synchronized CUDA node time by
profile label, tensor name, and SAM3 ViT stage. On the current `sam3_encode`
node-profile log, the stage ranking is:

| Stage | Drop-max synchronized sum | Count | Mean after drop-max | Evidence |
| --- | ---: | ---: | ---: | --- |
| SAM3 ViT window head64 FATTN | `67.823072 ms` | `140` | `0.487936 ms` | `outputs/sam3-bf16-current-node-profile-20260612b/names-sam3-encode.json` |
| SAM3 ViT MLP GELU | `37.989824 ms` | `160` | `0.238930 ms` | same |
| SAM3 ViT global head64 FATTN | `35.239392 ms` | `20` | `1.854705 ms` | same |
| after-attention residual/norm/copy | `26.549728 ms` | `160` | `0.166979 ms` | same |
| Q/K layout and RoPE materialization | `24.670048 ms` | `320` | `0.077336 ms` | same |
| block-output residual/norm/copy | `20.039680 ms` | `155` | `0.130128 ms` | same |

This stage view makes the remaining attention target more concrete. Isolated
PyTorch shape probes showed that BF16 SDPA for the SAM3 window shape can be
around `0.270 ms/call`, while the current ggml head64 window FATTN is around
`0.488 ms/call` in the synchronized node profile. The global shape is much less
attractive because current ggml is already close to the PyTorch SDPA timing.
The next attention-side work should therefore prioritize SAM3 window head64
FATTN, not another global-head64 switch.

The same pass rejected a tempting cuDNN shortcut. The standalone cuDNN SDPA
bench accepts BF16 input with FP32 output, but it is not faster for the current
window target: `sam3_cudnn_sdpa_bench --io-dtype bf16 --out-dtype fp32
--q-seq 576 --kv-seq 576 --batch 9 --heads 16 --head-dim 64` measured
`0.903324 ms/call`, and the `5184` global shape measured `2.455730 ms/call`.
Both are slower than the current ggml FATTN stage timings, so direct FP32
cuDNN output is not the root path for beating Python on this hardware.

A profile-free C++ default recheck after the measurement-tooling changes
measured `sam3-bf16` at `163.4 ms/frame` on the same 5-frame short video
contract (`outputs/sam3-bf16-current-recheck-20260612c.log`). This is within
the latest `~163-169 ms/frame` default range and does not show a performance
regression from the profiling-script/doc updates.

The first head64-window MMA scheduler probe did not produce a win. ggml CUDA now
has a diagnostic-only `GGML_CUDA_FATTN64_MMA_NCOLS1` override, optionally scoped
with `GGML_CUDA_FATTN64_MMA_NCOLS1_MATCH`, so the SAM3 window-attention MMA
`ncols1` can be measured without changing the default. Restricting the override
to `window_attn` and running three profile-free repeats gave:

| Variant | Track values | Mean | Decision |
| --- | --- | ---: | --- |
| default `ncols1=64` | `163.6, 164.3, 163.9 ms` | `163.933 ms` | keep default |
| forced `ncols1=32` | `167.2, 168.5, 166.9 ms` | `167.533 ms` | reject |
| forced `ncols1=16` | `170.0, 169.8, 169.5 ms` | `169.767 ms` | reject |
| forced `ncols1=8` | `185.3, 187.5, 185.1 ms` | `185.967 ms` | reject |

Evidence is in `outputs/sam3-bf16-fattn64-ncols1-ab-20260612a/summary.json`.
This rules out a smaller-MMA-tile fix for SAM3 window head64 on the current RTX
50-series path. A direct attempt to enable the existing multi-stage `cp.async`
pipeline for head64 MHA `ncols1=64,ncols2=1` was also rejected before runtime:
the current MMA template asserts that the required out-of-bounds checks are
incompatible with the multi-stage pipeline. Making that path viable would
require a real kernel specialization that removes or proves away the OOB cases,
not a scheduler flag.

Two follow-up dataflow probes were also rejected as defaults on 2026-06-12.
First, ggml CUDA now has a diagnostic-only
`GGML_CUDA_DISABLE_CUBLASLT_BIAS_FUSION_MATCH=<substring>` switch so a specific
bias tensor can avoid the cuBLASLt bias epilogue and fall through to later graph
fusion. Using `mlp.fc1.bias` to test whether `ADD+GELU` fusion beats the
current cuBLASLt-bias-plus-standalone-GELU path did not win: default measured
`163.3, 175.0, 165.1, 163.6, 163.7 ms` (`166.14 ms` mean), while the
no-fc1-bias-epilogue probe measured `164.0, 166.2, 165.0, 175.6, 164.3 ms`
(`167.02 ms` mean). Evidence is in
`outputs/sam3-bf16-mlp-fc1-add-gelu-ab-20260612a/summary.json`.

Second, `SAM3_ENABLE_VIT_CONTIGUOUS_ATTENTION_V=1` explicitly materializes the
SAM3 ViT attention `V` tensor before FATTN. This confirms the local bottleneck
but not a useful graph-level optimization: the FATTN profile improved
`sam3-vit-window-head64` from `42.444384 ms` to `39.410176 ms` across `84`
profiled calls by reducing `V_convert_ms` from `10.252224 ms` to
`4.007264 ms`, but the normal benchmark regressed from `164.3 ms/frame` to
`165.6 ms/frame`. Evidence is in
`outputs/sam3-bf16-v-cont-profile-20260612a/` and
`outputs/sam3-bf16-v-cont-ab-20260612a/summary.json`. The useful lesson is that
V conversion should be eliminated inside the FATTN/kernel dataflow; adding a
separate `cont` node is not enough.

### 2026-06-11 SAM3 BF16 cuDNN Pack-V and QKV Bias Follow-up

The cuDNN SDPA head64 diagnostic was extended so non-contiguous BF16 `V` tensors
can be packed into contiguous BF16 before calling cuDNN. This fixes the previous
route blocker where SAM3 BF16 Q/K were contiguous but `V` was not. With
`SAM3_BF16_VIT_QKV_CHAIN=1`, `GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1`,
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD64=1`, and
`GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD64_UNSAFE_RUN=1`, global-only cuDNN now routes;
adding `GGML_CUDA_CUDNN_SDPA_HEAD64_ALLOW_WINDOW=1` routes the window calls too.

This is still not an accepted same-parity optimization:

| Probe | Timing result | Parity result | Decision |
| --- | ---: | --- | --- |
| cuDNN global-only, isolated against the same QKV/direct baseline | `171.0 ms` mean vs `170.33 ms` baseline | `mask_hash_equal_rows=0/3`, `min_bbox_iou=0.9990748579786064` | reject |
| cuDNN global+window, isolated against the same QKV/direct baseline | `167.67 ms` mean vs `170.33 ms` baseline | `mask_hash_equal_rows=0/3`, `min_bbox_iou=0.9989087808087571` | reject |

Evidence:
`outputs/sam3-bf16-cudnn-isolated-packv-20260611be/summary.json` and
`outputs/sam3-bf16-cudnn-isolated-parity-20260611bf/`.

The same profile refresh showed that QKV direct BF16 output moves the SAM3 ViT
QKV projection into `dst=bf16[3072,5184]`, but the bias add remains a separate
`ADD dst=bf16[3072,5184] src1=f32[3072]` node. The synchronized node profile
ranked that broadcast add at `25.806912 ms` drop-max across the profiled calls.
cuBLASLt bias epilogue would be the cleaner fix, but every checked BF16-dst QKV
heuristic index `0..7` returned no heuristic for this shape, so the fallback
remains `MUL_MAT dst=bf16` plus a separate bias add
(`outputs/sam3-bf16-qkv-direct-profile-20260611bg/hotspots.json`,
`outputs/sam3-bf16-qkv-bf16dst-algo-probe-20260611bi/`).

A CUDA axis0 fast path for `bf16 + f32 bias -> bf16` was implemented as an
opt-in diagnostic behind `GGML_CUDA_ENABLE_BIN_BCAST_AXIS0_BF16_F32_FAST=1`.
It is default-off because the A/B was slower and noisier: disabled
`170.4 ms` mean / `170.0 ms` median versus enabled `179.4 ms` mean /
`173.0 ms` median on the same QKV/direct SAM3 BF16 contract
(`outputs/sam3-bf16-axis0-bf16-add-fast-ab-20260611bj/summary.json`).

Current conclusion: cuDNN route coverage and a standalone BF16 bias-add kernel
are not enough. The remaining SAM3 gap still needs a larger ViT dataflow change,
most likely a fused or lower-overhead QKV/RoPE/FATTN/projection path that keeps
the Python-like low-precision contract without extra materialization or a
custom GEMM+bias path for BF16 output shapes that cuBLASLt does not support.

### 2026-06-12 SAM3 BF16 FATTN Non-Contiguous V Pack Default

The SAM3 BF16 attention profile showed that the head64 MMA path still receives
F32 Q/K/V tensors and converts K/V to F16 before the kernel. The V tensor is the
direct QKV view after `[0,2,1,3]` permutation, so it uses the generic
non-contiguous conversion path. The specialized F32-to-F16 `half2` pack for
non-contiguous tensors whose dim0 is contiguous and even is now default-on. Set
`GGML_CUDA_DISABLE_FATTN_F32_TO_F16_NC_DIM0_VEC2=1` to force the previous
generic conversion path.

The refreshed local profile confirms the intended effect. With the optimized
path default-on, `sam3-vit-window-head64` V conversion is `5.370784 ms` across
`84` profiled calls versus `10.240096 ms` with the path disabled; total window
attention drops from `42.331840 ms` to `37.497600 ms`. Global head64 attention
also improves, with V conversion `1.465152 -> 0.754048 ms` and total
`23.059008 -> 21.902912 ms`
(`outputs/sam3-bf16-fattn-nc-dim0-vec2-defaulton-20260612a/`).

Parity against the previous conversion path is exact on the checked 10-frame
JSONL: `mask_hash_equal_rows=10/10`, `min_bbox_iou=1`,
`max_bbox_delta_px=0`, `max_score_abs_delta=0`, and no mask-area delta
(`outputs/sam3-bf16-fattn-nc-dim0-vec2-defaulton-20260612a/compare.json`).

The normal benchmark now also supports enabling it by default. In a 10-pair
same-input A/B before flipping the default, the optimized path saved
`1.37 ms/frame` mean and `1.5 ms/frame` median versus the generic path, with
all 10 masks exactly equal
(`outputs/sam3-bf16-fattn-nc-dim0-vec2-paired-20260612a/`). After flipping the
default, a 5-pair default-on versus disabled check measured `165.06 ms/frame`
default-on and `170.70 ms/frame` disabled, saving `5.64 ms/frame` mean and
`2.9 ms/frame` median
(`outputs/sam3-bf16-fattn-nc-dim0-vec2-defaulton-perf-20260612a/summary.json`).

Decision: keep this path default-on. It is a modest but parity-safe CUDA
conversion win, not the full SAM3 goal. The next root attempt should still
avoid the conversion altogether by changing the QKV/FATTN dataflow, or move the
conversion into a larger fused attention path where launch and memory traffic
are amortized.

### 2026-06-11 SAM3 Root Optimization Target Refresh

The current same-contract SAM3 blocker is no longer frame-interval ambiguity.
Both C++ and official Python measure frame 1..N-1 after the frame-0 prompt/add
step. The script now labels new rows as
`frame1_plus_encode_propagate_prompt_excluded`; older JSON rows with
`per_frame_encode_detect_propagate` should be read as the same measured
interval, not as including frame-0 detection.

The preprocessing/input-staging contract is now reported explicitly because it
is not identical at the timer boundary. Official Python `start_session` loads,
resizes, normalizes, and usually moves frames before the timed
`propagate_in_video` loop; C++ `Track/fr` normally includes the per-frame
preprocess/upload path before `sam3_encode`. This does not explain the current
SAM3 gap when CUDA preprocessing is enabled. A local 3-frame BF16 profile on the
same extracted frames measured the default CUDA preprocess path at
`164.2 ms/frame`; forcing CPU preprocessing with
`SAM3_DISABLE_CUDA_PREPROCESS=1` regressed to `206.5 ms/frame`. The tracked-frame
profile showed CUDA preprocess as `0.000 ms`, input upload about
`0.49-0.65 ms`, `sam3_encode` compute about `145-147 ms`, and
`propagate_single` about `11-12 ms` on the comparable default frames
(`outputs/sam3-bf16-preprocess-ab-20260611/benchmark.log`,
`outputs/sam3-bf16-preprocess-ab-20260611/profile-summary.json`). The practical
conclusion is unchanged: CUDA preprocessing must stay on, and beating the
official Python `~120.8 ms/frame` SAM3 BF16 row requires a roughly `30-40 ms`
reduction inside the ViT encode path, not another preprocessing tweak.

The next accepted optimization must target the ViT projection/attention
boundary, not another standalone conversion kernel. The strongest candidate is
a SAM3-specific CUDA dataflow that combines some or all of:

- QKV projection output bias handling,
- Q/K contiguous layout materialization,
- RoPE application,
- V packing or direct FATTN consumption,
- head64 window/global FATTN input preparation.

The reason is visible in the latest profiles: QKV/projection/MLP matmuls and
head64 FATTN dominate, while isolated V-pack, cuDNN SDPA routing, cuBLASLt
GELU, tile/vector FATTN forcing, direct BF16 QKV output, and standalone
BF16-bias add probes were either slower, too small, or not same-output safe.

Acceptance criteria for the next root implementation:

- It must be opt-in until proven, preferably with one environment variable.
- It must preserve default-path mask hashes on the 3-frame SAM3 BF16 JSONL
  parity row before any speed claim.
- It must be measured with at least a paired A/B run against the current default
  and a same-contract official Python row using the corrected tracking-scope
  label.
- It must show improvement in unsynchronized `Track/fr`, not only synchronized
  node-profile subtotals.
- It should reduce at least one of the current top synchronized hotspots:
  QKV/projection materialization, FATTN input conversion, or FATTN kernel body.

Rejected follow-up class: adding more single-purpose copy/convert kernels is
not enough unless the normal unsynchronized paired A/B also improves. The
non-contiguous V-pack probe is the reference failure mode: it reduced local
conversion time but made the normal 10-pair run worse.

A more structural FATTN attempt was also rejected. The experiment routed SAM3
ViT head64 MMA attention through an opt-in inline F32 K/V tile conversion path
instead of the common pre-FATTN K/V F16 pack. It preserved same-output JSONL
parity against the default path on the 3-frame BF16 check
(`mask_hash_equal_rows=3/3`, `min_bbox_iou=1`, `max_bbox_delta_px=0`,
`max_score_abs_delta=0`; `outputs/sam3-bf16-inline-f32-kv-20260611a/compare.json`),
and the FATTN profile confirmed that the external K/V conversion was removed
(`need_f16_K=0`, `need_f16_V=0`). However, normal unsynchronized `Track/fr`
regressed from `165.0 ms/frame` to `182.9 ms/frame`
(`outputs/sam3-bf16-inline-f32-kv-20260611a/default/benchmark.log`,
`outputs/sam3-bf16-inline-f32-kv-20260611a/inline/benchmark.log`). The attempted
implementation was removed; the evidence says moving conversion into the
current MMA body is slower than the existing prepack path. A viable FATTN change
needs a different kernel schedule or a producer-side QKV layout change, not just
inline tile conversion.

The F16 producer-side output knobs were refreshed on the current direct-QKV/MMA
baseline. `SAM3_F16_VIT_QKV_OUTPUT=1 SAM3_F16_VIT_ATTN_PROJ_OUTPUT=0`
preserved the checked mask hashes across five paired 3-frame runs, with only
small bbox/score drift (`min_bbox_iou=0.9997575160`,
`max_bbox_delta_px=0.01`, `max_score_abs_delta=0.000533`), but it did not
produce a robust speed win: default `Track/fr` values were
`[170.0, 171.0, 198.0, 172.0, 171.0]`, qkv-output values were
`[171.0, 192.0, 171.0, 173.0, 172.0]`, and qkv-output was slower by
`+1.0 ms` median / `+0.667 ms` trimmed mean
(`outputs/sam3-f16-qkv-producer-refresh-20260611b/paired5/summary.json`).
Adding `GGML_CUDA_CUBLASLT_DIRECT_F16_DST=1` was worse in the single-run
screen: `187.5 ms/frame` and only `2/3` equal mask hashes
(`outputs/sam3-f16-qkv-producer-refresh-20260611b/compare-qkv-direct.json`).

`SAM3_F16_VIT_ATTENTION_LINEAR_OUTPUT=1` was also rechecked because it can
reduce one F16 bridge before the attention projection. It was same-mask on the
checked rows (`min_bbox_iou=0.9999196408`, `max_bbox_delta_px=0.005`,
`max_score_abs_delta=0.000164`, `mask_hash_equal_rows=3/3`), but the 5-pair
normal timing did not improve: default values were
`[193.2, 170.7, 171.4, 172.1, 196.8]`, attention-output values were
`[173.5, 192.1, 202.0, 174.0, 175.9]`, and attention-output was slower by
`+1.9 ms` median / `+1.2 ms` trimmed mean
(`outputs/sam3-f16-attn-linear-output-paired5-20260611/summary.json`).
`SAM3_F16_VIT_MLP_LINEAR_OUTPUT=1` was slower in the single-run screen
(`187.7 ms/frame`), and enabling both attention and MLP linear outputs changed a
mask hash (`2/3` equal), so neither is a default candidate
(`outputs/sam3-f16-linear-output-refresh-20260611`).

F16 linear-output block selectors were added for narrower diagnostics:
`SAM3_F16_VIT_QKV_OUTPUT_BLOCKS`,
`SAM3_DISABLE_F16_VIT_QKV_OUTPUT_BLOCKS`,
`SAM3_F16_VIT_ATTN_PROJ_OUTPUT_BLOCKS`,
`SAM3_DISABLE_F16_VIT_ATTN_PROJ_OUTPUT_BLOCKS`,
`SAM3_F16_VIT_MLP_FC1_OUTPUT_BLOCKS`,
`SAM3_DISABLE_F16_VIT_MLP_FC1_OUTPUT_BLOCKS`,
`SAM3_F16_VIT_MLP_FC2_OUTPUT_BLOCKS`, and
`SAM3_DISABLE_F16_VIT_MLP_FC2_OUTPUT_BLOCKS`. They follow the existing
selector syntax such as `7,15,23,31` or `1-7,12`, and do not change the default
path. A global-block screen for `7,15,23,31` remained same-mask but did not show
a clear single-run speed win: default `170.5 ms/frame`, qkv-global
`170.3 ms/frame`, attn-proj-global `171.2 ms/frame`, qkv+attn-global
`171.1 ms/frame`, with all three comparisons at `mask_hash_equal_rows=3/3`
(`outputs/sam3-f16-block-selector-screen-20260611`). The selector-only smoke
confirmed `SAM3_F16_VIT_QKV_OUTPUT_BLOCKS=7,15,23,31` works without the global
enable env, but measured `171.9 ms/frame`; this is a diagnostic knob, not an
accepted default optimization
(`outputs/sam3-f16-block-selector-smoke-20260611/qkv-global-blocks-only/benchmark.log`).

The latest F16 fusion diagnostics also rule out a missing RoPE-pair pass as the
main cause. `GGML_CUDA_PROFILE_ROPE_PAIR_FUSION=1` reported successful
`GGML_CUDA_CONT_ROPE_PAIR_FUSION` for every SAM3 ViT block Q/K path in the
profile run (`outputs/sam3-f16-rope-fusion-profile-20260611b/benchmark.log`).
The remaining F16 bridge work is mostly around layernorm output copies:
`GGML_CUDA_PROFILE_ADD_NORM_CPY_FUSION=1` showed some successful
`ADD_NORM_CPY` cases, but most `sam3_vit_block_*_norm2 (copy)` candidates were
rejected with `reason=memory`
(`outputs/sam3-f16-add-cpy-profile-20260611/benchmark.log`). This cannot be
fixed by simply ignoring the guard because the low-precision side-output buffer
can overlap inputs that the fused layernorm still needs to read. The next
implementation attempt should either change allocation/lifetime so the side
output no longer overlaps live inputs, or move to a larger FATTN/projection
schedule where the copy is avoided rather than side-written.

### 2026-06-11 SAM3 BF16 Root-Path Recheck

The same fusion question was rechecked on the current SAM3 BF16 tracking path.
Unlike the F16 bridge case above, `GGML_CUDA_PROFILE_ADD_NORM_CPY_FUSION=1`
shows the normal SAM3 ViT residual/layernorm/copy candidates succeeding. The
current blocker is therefore not missing `ADD_NORM_CPY`; the synchronized
hotspot profile is still dominated by ViT `MUL_MAT`, head64
`FLASH_ATTN_EXT`, MLP `GELU`, and Q/K materialization
(`outputs/sam3-bf16-fusion-profile-20260611b/hotspots.json`).

The ncols=1024 LayerNorm scheduler was also refreshed on the current build. In
a short interleaved run, default/mode-2 stayed exact against the default JSONL,
while `GGML_CUDA_NORM_1024_MODE=1` and `=3` introduced small low-precision
drift and did not show a speed win. The measured medians were default
`165.0 ms/frame`, mode-1 `165.0`, mode-2 `181.0` with one slow outlier, and
mode-3 `172.0`; mode-2 is the default path and exact, so this does not identify
a new optimization surface
(`outputs/sam3-bf16-norm-mode-refresh-20260611c/`).

An opt-in FATTN conversion probe that packed contiguous K and non-contiguous
dim0-contiguous V into F16 in one kernel was implemented and then rejected. It
was exact against the default 3-frame JSONL (`min_bbox_iou=1.0`, zero
bbox/score/mask-area deltas, `mask_hash_equal_rows=3/3`), but the 10-pair
normal benchmark did not beat default: default measured mean `175.1`,
median `175.5`, trimmed mean `175.0 ms/frame`; the K+V fused probe measured
mean `175.8`, median `183.0`, trimmed mean `176.25 ms/frame`
(`outputs/sam3-bf16-fattn-kv-contig-v-nc-pack-20260611/ab10/summary.json`).
The probe was removed rather than kept as another diagnostic knob. The existing
V-only non-contiguous dim0 vec2 path remains the better localized FATTN
conversion diagnostic, but it is still not robust enough to enable by default.

CUDA graph capture was refreshed on the same current BF16 contract. It remains
parity-safe but slower: default measured mean `165.33`, median `165.0`,
trimmed mean `165.0 ms/frame`; `SAM3_CUDA_ENABLE_GRAPHS=1` measured
`173.33`, `168.0`, `168.0`; and adding `GGML_CUDA_GRAPH_OPT=1` measured
`167.67`, `168.0`, `168.0`
(`outputs/sam3-bf16-cuda-graphs-refresh-20260612/summary.json`). All checked
graph JSONLs were exact against default (`min_bbox_iou=1.0`, zero bbox/score
deltas, `mask_hash_equal_rows=3/3`). The profile-label run shows why this does
not close the gap: repeated `sam3_encode` and `propagate_single` graph computes
are not shorter than default, while capture/warmup adds memory pressure and
startup work (`outputs/sam3-bf16-cuda-graphs-refresh-20260612/profile-labels/`).

A follow-up cuBLASLt heuristic sweep found a measurement trap rather than a real
speedup. `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1=6` initially looked
faster in normal A/B, but timing logs showed cuBLASLt returned only two
heuristics for the `bf16 x bf16 -> f32 [4736,5184]` FC1 shape. Index 6 therefore
fell back to algorithm 0, while the cache key and timing report still carried
the requested index. The cache key is now normalized after clamping the
heuristic index to the actually usable range. After the fix, the same invalid
index reports `algo_index=0 returned=2`, stays exact against default, and
measures in the same speed band: default mean `166.25`, median `165.5`; invalid
index mean `165.5`, median `165.0`
(`outputs/sam3-bf16-cublaslt-invalid-index-normalized-20260612/`). This removes
a false-positive optimization path; valid FC1 index 1 was slower in the sweep,
so no cuBLASLt FC1 default change is accepted.

### 2026-06-12 SAM3 BF16 ConvTranspose Repeat-Bias Fusion Probe

The CUDA ConvTranspose2D bias-fusion matcher was extended to cover the SAM3
neck form `CONV_TRANSPOSE_2D -> REPEAT(channel bias) -> ADD`. The existing
opt-in `GGML_CUDA_ENABLE_CONV_TRANSPOSE_BIAS_FUSION=1` now reaches the large
neck deconvs as well as the smaller SAM decoder deconvs. A second diagnostic
env, `GGML_CUDA_ENABLE_CONV_TRANSPOSE_BIAS_FUSION_ADD_ONLY=1`, also enables
the fusion and skips the intermediate conv-output store when the conv output is
not externally needed.

The node profile confirms the intended root-path change: SAM3 neck deconv nodes
such as `node_2442` and `node_2447` now report `fused=1 skipped=3` under the
opt-in path (`outputs/sam3-bf16-conv-transpose-repeat-bias-profile2-20260612a/`).
JSONL parity against default was exact for both fused variants: `10/10` mask
hash rows, `min_bbox_iou=1.0`, and zero bbox/score/mask-area deltas
(`outputs/sam3-bf16-conv-transpose-repeat-bias-fusion-20260612a/compare.json`,
`outputs/sam3-bf16-conv-transpose-repeat-bias-writeconv-20260612a/compare.json`).

The performance result is too small and noisy to default-enable. The add-only
variant was exact but slower in the single run (`170.2 -> 172.8 ms/frame`).
The write-conv variant was exact and slightly positive in a six-pair run:
default mean `173.43`, median `174.45`; fused mean `172.68`, median `172.6`;
paired saved mean `0.75 ms/frame`, median `0.55`, with one regressing pair at
`-0.9 ms/frame`
(`outputs/sam3-bf16-conv-transpose-repeat-bias-writeconv-paired-20260612a/paired-summary.json`).
Keep this as an opt-in diagnostic rather than a default optimization.

### 2026-06-12 SAM3 BF16 FATTN Head64 Q-in-register Probe

The Ampere CUDA FATTN MMA config for the SAM3 head64, `ncols1=64/ncols2=1`
path now keeps Q in registers by default. This is a very small kernel-scheduler
change, not a graph-level fusion. The target was the current BF16 profile where
`sam3-vit-window-head64` and `sam3-vit-global-head64` were the remaining FATTN
hot spots after conversion-only probes stopped paying off.

The change is parity-safe on the checked tracking contract. A default-vs-probe
JSONL comparison over 3 frames reported `min_bbox_iou=1.0`, zero
bbox/score/mask-area deltas, and `mask_hash_equal_rows=3/3`
(`outputs/sam3-bf16-fattn-qinreg64-20260612a/compare-default.json`).

The synchronized FATTN event profile improved the affected groups modestly:
window head64 total `37.400 -> 37.032 ms` and global head64 total
`22.270 -> 21.717 ms`
(`outputs/sam3-bf16-fattn-default64-20260612a/fattn-summary.json`,
`outputs/sam3-bf16-fattn-qinreg64-20260612a/fattn-summary.json`). Normal
profile-free tracking is mostly noise-limited: 6-run median changed
`163.3 -> 163.1 ms/frame`, and drop-max mean changed
`163.28 -> 163.02 ms/frame`
(`outputs/sam3-bf16-fattn-qinreg64-paired-20260612a/paired-summary.json`).
This is accepted because it is exact and slightly faster in the targeted kernel,
but it does not materially close the C++ versus official Python gap by itself.

The cuBLASLt `BIAS+GELU_ERF` epilogue was also refreshed after this change
because the current profile still shows large MLP `UNARY(GELU_ERF)` time. It is
still not acceptable: `GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION=1`
regressed the short run from `161.8` to `172.8 ms/frame` and changed frame-0
mask output (`min_bbox_iou=0.995628`, `mask_hash_equal_rows=2/3`)
(`outputs/sam3-bf16-cublaslt-gelu-erf-refresh-20260612a/summary.json`,
`outputs/sam3-bf16-cublaslt-gelu-erf-refresh-20260612a/compare.json`).

A narrower `UNARY(GELU_ERF) -> CPY(BF16)` store probe was also rejected as a
default. `GGML_CUDA_ENABLE_UNARY_CPY_VEC4_PACKED=1` keeps the same scalar
`erff` GELU formula and rounds with the existing BF16 conversion, but writes
four BF16 results as one packed 64-bit store. It stayed exact in six paired
full-mask comparisons (`mask_hash_equal_rows=3/3`, zero bbox/score/area deltas;
`outputs/sam3-bf16-unary-cpy-packed-20260612a/summary.json`). The normal
benchmark was too small to justify defaulting it: opt-in measured `163.45 ms`
mean / `163.2 ms` median, default measured `163.83 ms` mean / `163.7 ms`
median. The synchronized node profile did not confirm a local win; the
dominant ViT MLP GELU signature moved from `22.884480 ms` default drop-max sum
to `23.302432 ms` with packed stores
(`outputs/sam3-bf16-unary-cpy-packed-20260612a/profile-default/profile-summary.json`,
`outputs/sam3-bf16-unary-cpy-packed-20260612a/profile-enabled/profile-summary.json`).
This points back to the `erff` work and the `fc1 -> GELU -> fc2` dataflow, not
to the two-store BF16 write pattern.

Two existing ViT GEMM layout/heuristic knobs were refreshed on the same
post-FATTN build and remain rejected as defaults. First,
`SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` stayed exact against default
(`mask_hash_equal_rows=3/3`) but did not show a stable speed win in the
six-pair warm benchmark: default median `164.65 ms/frame`, flat-chain median
`173.1`; drop-max mean `167.30` versus `171.04`
(`outputs/sam3-bf16-mlp-flat-paired-20260612a/paired-summary.json`). Second,
the current cuBLASLt timing run still reports two heuristics for the main
SAM3 ViT shapes, but forcing index 1 is not faster. Single same-contract
screens measured default `163.3 ms/frame`, QKV index 1 `163.4`, attention
projection index 1 `165.8`, MLP fc1 index 1 `165.3`, MLP fc2 index 1 `164.3`,
and all four index 1 `167.9`; all stayed exact against default
(`outputs/sam3-bf16-cublaslt-algo-refresh-20260612a/`). The remaining viable
GEMM direction is therefore not another heuristic toggle; it needs a materially
different BF16 ViT block dataflow or a custom kernel that beats cuBLASLt on the
`[3072/4736/1024, 5184]` shapes without changing the output contract.

The current CUDA node-profile tooling now separates repeated compute labels
with `--split-computes`, so `sam3_encode#01` can be treated as frame-0/cache
initialization and `sam3_encode#02/#03` as steady tracking encode passes
(`outputs/sam3-bf16-current-node-profile-20260612c/names-sam3-encode-split.json`).
This matters because the first encode has large one-time outliers: `other`
reported `243.643 ms` total with a `103.930 ms` max node, `qkv-matmul`
reported `65.828 ms` with a `47.462 ms` max node, and `window-attn` reported
`52.565 ms` with a `40.377 ms` max node. Those numbers are useful for startup
diagnostics but should not drive steady-track optimization choices.

On the steady encodes, the dominant cost is still the ViT GEMM path. For
`sam3_encode#02`, drop-max stage totals were `36.273 ms` for
`sam3-vit:proj-or-mlp-fc2-matmul`, `28.746 ms` for
`sam3-vit:mlp-fc1-matmul`, `18.769 ms` for `sam3-vit:qkv-matmul`,
`11.333 ms` for `sam3-vit:window-attn`, `10.993 ms` for `other`, and
`5.423 ms` for `sam3-vit:global-attn`. `sam3_encode#03` repeated the same
ranking: `36.447`, `28.864`, `18.897`, `11.335`, `10.985`, and `5.362 ms`.
The next root optimization should therefore target the BF16 ViT block GEMM
dataflow first. Neck/deconv and attention probes are still worth keeping, but
their remaining steady-track upside is secondary unless they combine with a
larger ViT fusion.

A short control run also confirmed that simply leaving the cuBLASLt
bias-epilogue path is not viable. With
`GGML_CUDA_ENABLE_CUBLASLT_BIAS_FUSION=0`, the same 3-frame SAM3 BF16 smoke
went from `162.6` to `205.2 ms/frame`
(`outputs/sam3-bf16-measurement-tooling-smoke-20260612/benchmark.log`,
`outputs/sam3-bf16-cublaslt-off-smoke-20260612/benchmark.log`). The JSONL
comparison was not exact either: `min_bbox_iou=0.9960337511`,
`max_score_abs_delta=0.000685`, `max_abs_mask_area_rel_delta=0.002548`, and
`mask_hash_equal_rows=0/3`
(`outputs/sam3-bf16-cublaslt-off-smoke-20260612/compare.json`). The accepted
baseline therefore remains cuBLASLt bias fusion; the next speed path has to
beat that baseline, not replace it with regular cuBLAS plus separate bias/add
nodes.

The standalone BF16 cuBLASLt microbench was also aligned with the SAM3 ViT
encode shapes (`k=1024`, `cols=5184`, `rows=1024/3072/4736`) so the next kernel
target has a lower-level baseline
(`outputs/sam3-bf16-vit-gemm-micro-20260612a/summary.json`). The plain bias
epilogue medians were `0.213120 ms` for rows `1024`, `0.577920 ms` for rows
`3072`, and `0.933312 ms` for rows `4736`, matching the application-level
cuBLASLt timing scale. `CUBLAS_COMPUTE_32F_FAST_16BF` did not provide a stable
win (`+0.001792`, `+0.014400`, `-0.017664 ms` median delta), cuBLASLt GELU
epilogue was much slower (`+9.85%`, `+42.05%`, `+35.44%`), and direct BF16 dst
failed for all three shapes in this standalone path.

An explicit microbench heuristic-index sweep confirmed the same conclusion as
the full tracking run: cuBLASLt algorithm index `0` is still the best available
choice for these shapes
(`outputs/sam3-bf16-vit-gemm-algo-index-20260612a/summary.json`). Index `1`
was slower by `+5.03%` for rows `1024`, `+12.26%` for rows `3072`, and
`+3.76%` for rows `4736`; index `2` was not returned by cuBLASLt. The remaining
root path is therefore a SAM3-specific BF16 GEMM/fusion that beats the current
cuBLASLt bias-epilogue baseline on these three shapes.

A workspace-size sweep was added to the standalone BF16 cuBLASLt microbench so
the current `32 MiB` default could be checked rather than assumed. On the three
SAM3 ViT GEMM shapes, `0/32/64/128/256 MiB` did not produce a single winner
(`outputs/sam3-bf16-vit-gemm-workspace-sweep-20260612a/summary.json`). The
most interesting isolated row was `128 MiB` on rows `3072`, where the median
fell from `0.596544` to `0.581344 ms` (`-2.55%`), but rows `1024` and `4736`
were effectively neutral or slightly worse.

That candidate did not survive the application-level check. Six paired
`sam3_benchmark` runs on SAM3 BF16 CUDA, 3 frames, in-process, measured the
same median and mean `track/fr` for default workspace and
`GGML_CUDA_CUBLASLT_BIAS_WORKSPACE_MB=128`: both medians were `164.5 ms` and
both means were `167.1667 ms`
(`outputs/sam3-bf16-workspace128-app-20260612a/summary.json`). All six JSONL
comparisons were exact (`bbox_iou=1.0`, zero bbox/score/area delta, and
`mask_hash_equal_rows=3/3`; see
`outputs/sam3-bf16-workspace128-app-20260612a/parity-summary.json`). The
workspace knob is therefore not accepted as a default performance change; it
remains an attribution control for future shape-specific experiments.

The next fusion target was the repeated `WIN_PART(F32) -> CPY(BF16)` before
window QKV. `ggml_win_part` was generalized to preserve the input dtype, and a
CUDA probe for `WIN_PART -> CPY` was added with a preserve-output variant that
would write both the original F32 window tensor and the BF16/F16 copy in one
kernel. The first profile found the candidate `168` times, but every candidate
was rejected by `ggml_cuda_check_fusion_memory_ranges`
(`outputs/sam3-bf16-winpart-cpy-fusion-profile-20260612f/summary.json`). A
follow-up memory-reject log showed the specific cause: `GGML_OP_CPY` carries a
self-reference in `src[1]` for its destination placeholder, and the generic
memory guard treated that self-reference as a real overlapping input. The guard
now ignores this `CPY` self-reference, which lets the fusion fire under
`GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION=1`.

The opt-in fusion is exact but not accepted as a default speed change. A
profiled 3-frame run fired `84/84` candidates
(`outputs/sam3-bf16-winpart-cpy-fusion-enabled-20260612a/profile.log`). Six
paired application runs measured enabled at `167.48 ms` mean /
`163.55 ms` median `track/fr`, versus disabled at `166.02 ms` mean /
`163.20 ms` median
(`outputs/sam3-bf16-winpart-cpy-fusion-ab-20260612a/summary.json`). The
enabled/disabled full-mask JSONL comparison stayed exact
(`mask_hash_equal_rows=3/3`;
`outputs/sam3-bf16-winpart-cpy-fusion-ab-20260612a/compare/fullmask-compare.json`).
Because the measured speed did not improve, this path remains opt-in only.

A simpler application-side opt-in,
`SAM3_BF16_VIT_WIN_PART_INPUT=1`, pre-casts the window-partition input to BF16
and uses the generalized low-precision `ggml_win_part` path. Six paired
application runs stayed exact, but did not improve speed: default measured
`164.0 ms` median / `167.0 ms` mean `track/fr`, while the opt-in measured
`164.5 ms` median / `167.5 ms` mean
(`outputs/sam3-bf16-winpart-bf16-input-20260612a/summary.json`). This path is
therefore left opt-in and is not accepted as a default performance change.

An attempted cuBLASLt residual epilogue fusion is also rejected. The opt-in
`GGML_CUDA_ENABLE_CUBLASLT_BIAS_RESIDUAL_FUSION=1` passes a same-shape residual
tensor as cuBLASLt C with `beta=1`, so eligible `MUL_MAT + bias -> ADD`
patterns can be computed as one cuBLASLt call. The first successful probe fired
`68` residual fusions in the feature encoder / decoder side, but not in the
dominant SAM3 ViT `mlp.lin2` path
(`outputs/sam3-bf16-cublaslt-residual-fusion-profile-20260612b/profile.log`).
It is not same-precision safe for the current goal: the enabled/disabled
full-mask comparison changed frame 0 (`bbox_iou=0.9988456949`,
`max_bbox_delta_px=0.042`, `max_score_abs_delta=0.001514`,
`mask_hash_equal_rows=2/3`;
`outputs/sam3-bf16-cublaslt-residual-fusion-ab-20260612a/compare/fullmask-compare.json`).
The six-pair timing was noisy and not enough to justify the numerical change:
enabled measured `165.80 ms` mean / `162.95 ms` median `track/fr`, disabled
measured `169.18 ms` mean / `163.60 ms` median
(`outputs/sam3-bf16-cublaslt-residual-fusion-ab-20260612a/summary.json`). Keep
this as a rejected diagnostic; the next residual-fusion attempt must target the
ViT pattern directly and prove full-mask parity.

The ViT-specific follow-up confirmed that the dominant block-output pattern is
visible but still not acceptable as a same-precision fusion. A reshape/view-aware
matcher found all `32` SAM3 ViT `mlp.lin2` residual candidates per frame. With
the normal fusion memory guard enabled, these candidates are rejected because
the block output and residual input overlap through ggml's in-place buffer
reuse. A diagnostic run that bypassed that guard fired `96/96` ViT
`mlp.lin2` residual fusions on the 3-frame smoke
(`outputs/sam3-bf16-vit-mlp-residual-fusion-profile-20260612b/profile.log`),
but the full-mask comparison still changed frame 0:
`bbox_iou=0.9968194164`, `max_bbox_delta_px=0.120`,
`max_score_abs_delta=0.001620`, and `mask_hash_equal_rows=2/3`
(`outputs/sam3-bf16-vit-mlp-residual-fusion-profile-20260612b/fullmask-compare.json`).
The guard bypass has therefore been removed. A future residual optimization
must preserve the original operation order `(GEMM + bias) + residual` or prove a
separate relaxed-precision contract; cuBLASLt `beta*C` epilogue fusion is not a
valid default for the current same-precision goal. The guarded implementation
now falls back to the existing bias-only fusion when residual fusion is rejected:
the guarded opt-in run reported `0` residual successes, `96` memory rejects
with bias-only retry, `96` ViT `mlp.lin2` bias-only successes, and exact
full-mask parity against default (`mask_hash_equal_rows=3/3`;
`outputs/sam3-bf16-vit-mlp-residual-fusion-guarded-20260612b/fullmask-compare.json`).

The next same-order residual/LayerNorm kernel specialization was also measured
and rejected as a default. `GGML_CUDA_ENABLE_NORM_1024_AFFINE_AXIS0=1` selects a
specialized `ncols=1024` residual LayerNorm affine kernel when the weight and
bias tensors are the normal `[1024,1,1,1]` axis-0 shape. It preserves the same
reduction thread count selected by `GGML_CUDA_NORM_1024_MODE` and only removes
the generic affine `fastmodulo()` path, so it stayed exact against default:
six paired full-mask comparisons all had `mask_hash_equal_rows=3/3` and zero
bbox/score/area deltas
(`outputs/sam3-bf16-norm1024-axis0-20260612a/summary.json`). It did not improve
application timing: opt-in measured `169.65 ms` mean / `163.4 ms` median
`track/fr`, while default measured `163.28 ms` mean / `163.25 ms` median. A
refreshed opt-in/default smoke after leaving the path opt-in-only also stayed
exact (`outputs/sam3-bf16-norm1024-axis0-20260612b/compare/fullmask-compare.json`).
This confirms that the remaining `ADD [1024,72,72]` profile cost is not mainly
the generic affine indexing overhead; a useful exact optimization would need to
reduce memory traffic or improve the reduction/writeback structure itself.

A first custom BF16 WMMA prototype was added to the standalone GEMM bench as
`--bf16-wmma`. It uses one warp per `16x16` output tile, accumulates with
WMMA BF16->F32, and applies the row bias before writing the column-major output.
The small-layout check passed against the F32 reference with
`max_abs=0.000296995044` at `k=1024, rows=1024, cols=64`
(`outputs/sam3-bf16-vit-gemm-wmma-proto-20260612a/`). It is not a performance
candidate yet: medians on the three SAM3 ViT shapes were `0.717728`,
`2.188320`, and `3.349632 ms`, versus cuBLASLt medians `0.213792`,
`0.598976`, and `0.917568 ms`. That is `3.36x`, `3.65x`, and `3.65x` slower.
The prototype is useful only as a measured lower bound for naive custom WMMA.
The next custom-kernel attempt needs a materially different design: larger MMA
tiling without excessive occupancy loss, better B/input reuse across output
rows, and an epilogue that does not round-trip through shared memory just to add
bias. A 4-warp-per-CTA column-tile variant was also measured and rejected: it
kept the small-layout check close but worsened the three-shape ratios to
`4.57x`, `5.28x`, and `5.29x` slower than cuBLASLt
(`outputs/sam3-bf16-vit-gemm-wmma4-proto-20260612a/summary.json`). The issue is
therefore not just CTA count; the custom path needs a deeper tile/data-reuse
design rather than a wider version of the same kernel.

The standalone GEMM bench now also has an explicit SAM3 ViT BF16 MLP-chain
mode, `--bf16-mlp-chain`, so future fused kernels can be compared against the
actual `fc1 -> GELU_ERF -> BF16 cast -> fc2` dataflow instead of two isolated
GEMMs. The baseline uses cuBLASLt bias epilogues for both BF16 GEMMs, computes
exact F32 `GELU_ERF`, rounds the activation to BF16, and then runs the second
cuBLASLt GEMM. On the SAM3 ViT shape `input_dim=1024`,
`hidden_dim=4736`, `output_dim=1024`, `cols=5184`, the 32 MiB workspace row
measured `2.134071 ms` mean / `2.124448 ms` median per MLP block
(`outputs/sam3-bf16-vit-mlp-chain-micro-20260612a/mlp-chain-ws32-rerun.log`).
This scales to roughly `68 ms` across the 32 ViT blocks, matching the current
steady-encode profile order of magnitude.

Two immediate cuBLASLt policy checks did not materially improve that chain
baseline. Increasing workspace to 128 MiB measured `2.120205 ms` mean /
`2.116768 ms` median, only about `0.36%` faster by median
(`outputs/sam3-bf16-vit-mlp-chain-micro-20260612a/mlp-chain-ws128-rerun.log`).
Using `CUBLAS_COMPUTE_32F_FAST_16BF` measured `2.129262 ms` mean /
`2.125376 ms` median at 32 MiB and `2.132922 ms` mean / `2.125248 ms` median
at 128 MiB
(`outputs/sam3-bf16-vit-mlp-chain-micro-20260612a/mlp-chain-fast16bf-ws32.log`,
`outputs/sam3-bf16-vit-mlp-chain-micro-20260612a/mlp-chain-fast16bf-ws128.log`).
The next acceptable MLP optimization must beat this measured `~2.12 ms/block`
chain baseline by enough to matter at application scale while preserving the
same `GELU_ERF` and BF16 activation rounding contract. Merely changing
workspace or cuBLASLt compute mode is not enough.

The chain overhead has now been split with an explicit standalone
`--bf16-gelu-cast` microbench. Using the same SAM3 ViT activation shape
`hidden_dim=4736`, `cols=5184`, `batches=1`, exact F32 `GELU_ERF`, and BF16
activation rounding, the standalone kernel measured `0.314103 ms` mean /
`0.313760 ms` median
(`outputs/sam3-bf16-vit-mlp-chain-micro-20260612a/gelu-cast-rerun.log`). The
previous single-GEMM medians were `0.884768 ms` for fc1 and `0.898176 ms` for
fc2, so the 32 MiB chain median `2.124448 ms` leaves `0.341504 ms/block`
outside the two GEMMs. Of that, `0.313760 ms/block` is the measured standalone
GELU/cast cost and only about `0.027744 ms/block` remains as extra sequencing
overhead. Across 32 ViT blocks this gives an absolute best case of roughly
`10.04 ms/frame` from deleting the standalone GELU/cast pass entirely, which is
only about `20%` of the current same-contract SAM3 BF16 gap to official Python
(`170.0 ms/frame` C++ mean versus `120.379702 ms/frame` Python mean). A useful
root optimization therefore cannot stop at launch fusion: it needs either a
CUTLASS-like fc1 epilogue that removes the GELU/cast memory round trip while
staying exact, plus additional GEMM-side wins, or a larger fused MLP kernel that
beats the current cuBLASLt GEMM baseline itself.

The same MLP shape was also checked against PyTorch BF16 to avoid chasing the
wrong kernel. A local CUDA-event microbench with `cols=5184`,
`input_dim=1024`, `hidden_dim=4736`, `output_dim=1024`, TF32 enabled, and
official-style autocast BF16 measured `2.083350 ms` mean / `2.089888 ms`
median per MLP block
(`outputs/sam3-bf16-vit-mlp-chain-micro-20260612a/pytorch-mlp-chain.log`).
That is only `0.034560 ms/block` faster than the C++ 32 MiB chain median. The
manual PyTorch path that forces `matmul -> F32 bias -> exact GELU -> BF16
activation -> matmul -> F32 bias` was slower (`3.052864 ms` median), so the
official Python advantage is not explained by a large same-shape MLP GEMM
advantage alone. The remaining full-app gap should be searched in the broader
ViT attention/layout path, decoder/neck work, or graph/dataflow differences.

The official SAM3 Python MLP contract is more specific than a plain
`Linear -> GELU` graph. `external/sam3/sam3/perflib/fused.py` implements
`addmm_act()` with `torch.ops.aten._addmm_activation`; it explicitly casts
`linear.bias`, `mat1`, and `linear.weight` to BF16, flattens the token dimension,
and runs fused addmm+GELU. The dtype hooks already confirm that official BF16
autocast then feeds `mlp.fc2` with BF16 and receives BF16 output
(`outputs/sam3-python-dtypes-20260522/sam3-dtypes.json`). Therefore, exact C++
F32 `GELU_ERF` parity against the current default is not the same as matching
official Python's performance contract.

The existing C++ fusion switches were rechecked from that official-contract
angle rather than as default-parity candidates. They still do not win. On a
three-pair short run, current default measured `169.4 ms` mean / `163.5 ms`
median; `GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION=1` measured
`183.3 ms` mean / `174.1 ms` median and changed all three checked mask hashes.
`SAM3_BF16_VIT_MLP_CHAIN=1` measured `180.3 ms` mean / `180.2 ms` median and
also changed all checked hashes, while combining MLP chain with cuBLASLt GELU
measured `183.733 ms` mean / `183.0 ms` median. Evidence:
`outputs/sam3-bf16-official-mlp-contract-probes-20260612a/summary.json`.

Direct BF16 cuBLASLt destination was also rechecked for this contract. With
`SAM3_BF16_VIT_MLP_CHAIN=1 GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1`, timing was
still slower (`175.3 ms` mean / `175.4 ms` median) and all checked mask hashes
changed. With broad `SAM3_BF16_VIT_LINEAR_OUTPUT=1` plus direct BF16 dst, median
was roughly neutral (`164.5 ms` versus default `164.6 ms`), but only `1/3`
mask rows matched per pair and the numerical output changed. The profile log
shows why this is not yet a root fix: cuBLASLt still reports `fail-heuristic`
for the SAM3 ViT MLP `fc1` shape `bf16 x bf16 -> bf16[4736,5184]` on every
block, exactly the shape that needs to match Python's fused
`_addmm_activation`. The next serious MLP optimization should therefore be a
CUTLASS-like or custom CUDA `fc1+bias+GELU -> BF16` kernel for the SAM3 ViT
shape, not another wrapper around the current cuBLASLt heuristic path
(`outputs/sam3-bf16-direct-bf16-mlp-contract-probes-20260612a/summary.json`).

A follow-up cuBLASLt probe tried to make that direct BF16 destination path
legal by separating the `C` and `D` matrix descriptors. It did not help:
direct BF16 destination with `BIAS+GELU` still failed heuristic selection, and
direct BF16 destination with bias-only also failed. The only legal direct BF16
case on the SAM3 ViT `fc1` shape was no-bias GEMM. A standalone MLP-chain
variant therefore tested `fc1` as no-bias BF16 output followed by a separate
BF16 in-place exact `bias+GELU_ERF` kernel before `fc2`. This improved the
microbench chain median from `2.127392 ms` to `2.072384 ms` per block
(`fast_16bf`: `2.113536 ms` to `2.069248 ms`), but it is not an acceptable app
optimization. The app-level probe hit all `64` main ViT `mlp.lin1` rows, yet
changed every checked mask hash (`mask_hash_equal_rows=0/3`,
`min_bbox_iou=0.9950488351966887`, `max_bbox_delta_px=0.222`,
`max_score_abs_delta=0.002938`) and remained slower than default in a
profile-free five-pair run: default measured `167.4 ms` mean / `164.0 ms`
median, the previous MLP-chain direct path measured `174.8 ms` mean /
`174.0 ms` median, and the split direct path measured `168.6 ms` mean /
`168.0 ms` median
(`outputs/sam3-bf16-mlp-direct-fc1-probe-20260612a/app-ab/perf/summary.json`).
The app split implementation was therefore rejected; the standalone bench keeps
the probe knobs only as evidence that cuBLASLt cannot supply the required fused
direct-BF16 epilogue for this shape.

An additional official-contract probe checks the GELU formula itself. PyTorch's
`torch.ops.aten._addmm_activation(..., use_gelu=True)` is not bit-identical to
either `F.gelu(approximate="none")` or `F.gelu(approximate="tanh")` on BF16
inputs because the fused kernel controls the accumulation and epilogue, but the
tanh approximation was slightly closer on the checked `4736 x 128` and
`4736 x 5184` shapes. An opt-in C++ switch,
`SAM3_VIT_MLP_APPROX_GELU=1`, therefore routes SAM3 ViT MLP `fc1` through
`ggml_gelu` instead of `ggml_gelu_erf`; block selectors are available through
`SAM3_VIT_MLP_APPROX_GELU_BLOCKS` and
`SAM3_DISABLE_VIT_MLP_APPROX_GELU_BLOCKS`. This is intentionally not a default:
a three-pair SAM3 BF16 A/B measured default at `169.833 ms` mean /
`163.8 ms` median, approximate GELU at `174.8 ms` mean / `174.5 ms` median,
approximate GELU plus MLP chain at `188.767 ms` mean / `183.2 ms` median, and
approximate GELU plus MLP chain plus direct BF16 destination at `174.267 ms`
mean / `174.4 ms` median. None matched default mask hashes; the closest
bounding-box IoU remained high (`>=0.99937`), but this is still a changed-output
relaxed-precision experiment rather than a same-parity optimization
(`outputs/sam3-bf16-approx-gelu-ab-20260612a/summary.json`).

The approximate-GELU profile is useful because it proves the current ggml
cuBLASLt path can fuse SAM3 ViT `mlp.lin1` as `BIAS+GELU` when the destination
is F32: the profile saw `66/66` MLP `fc1` calls succeed, with `64` GELU-enabled
main ViT rows. The Python-like direct BF16 target is the part that remains
unsupported by cuBLASLt here. With
`SAM3_VIT_MLP_APPROX_GELU=1`, `SAM3_BF16_VIT_MLP_CHAIN=1`, and
`GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1`, all `64` main ViT `mlp.lin1` calls hit
`fail-heuristic` for `dst_type=bf16`, while the non-main rows with F32
destination still succeeded
(`outputs/sam3-bf16-approx-gelu-probe-20260612a/`,
`outputs/sam3-bf16-approx-gelu-ab-20260612a/approx-gelu-profile-summary.json`).
A saved standalone `--bf16-wmma` check on the largest `rows=4736`,
`cols=5184`, `k=1024` shape measured `3.335808 ms` median for plain WMMA and
`3.355424 ms` median for the new `--bf16-wmma --gelu-epilogue` path that fuses
bias, tanh GELU, and BF16 store. By contrast, the same standalone harness
measured cuBLASLt BF16 bias GEMM at `0.885568 ms` median, cuBLASLt
`BIAS+GELU` at `1.291424 ms`, the full cuBLASLt MLP chain at `2.108800 ms`,
and standalone exact `GELU_ERF -> BF16` cast at `0.313024 ms`
(`outputs/sam3-bf16-wmma-fused-epilogue-20260612a/`). The next implementation
target is therefore not the existing naive WMMA prototype. It needs a
CUTLASS-grade or ggml-CUDA MMA-based tiled kernel that keeps the BF16 output
contract, fuses bias and GELU, and beats the existing cuBLASLt GEMM baseline
before it is wired into `sam3_benchmark`.

The cuDNN SDPA head64 path was rechecked as a relaxed-precision attention
candidate because the standalone cuDNN bench is fast (`0.309542 ms` mean for
the `batch=9, heads=16, seq=576, head_dim=64` window shape and `1.905622 ms`
for the `batch=1, heads=16, seq=5184` global shape). In the full app, however,
the ggml integration has to pack non-contiguous V and transpose the cuDNN BHSD
output back to ggml's `D,H,S,B` F32 layout. With all head64 calls enabled
(`F32->BF16`, window, and unsafe-run gates), a 3-pair profile-free smoke
measured default at `169.0667 ms` mean / `166.4 ms` median and cuDNN at
`170.0333 ms` mean / `169.7 ms` median
(`outputs/sam3-bf16-cudnn-sdpa-head64-smoke-20260612a/pair-summary.json`).
Full-mask parity also changed (`mask_hash_equal_rows=0/3`,
`max_bbox_delta_px=0.015`, `max_score_abs_delta=0.005001`;
`outputs/sam3-bf16-cudnn-sdpa-head64-smoke-20260612a/compare.json`). Global-only
cuDNN was worse (`184.8 ms/frame`) and still changed masks
(`mask_hash_equal_rows=1/3`). This path is therefore rejected as a default; a
future attention win must avoid the extra layout round trip or produce the
next consumer's layout directly. Combining cuDNN with
`SAM3_BF16_VIT_QKV_CHAIN=1` avoids the F32-to-BF16 Q/K conversion but still did
not win in a smoke run (`171.7 ms/frame`) and changed masks
(`mask_hash_equal_rows=0/3`;
`outputs/sam3-bf16-cudnn-sdpa-head64-smoke-20260612a/compare-qkv-chain.json`).
The standalone cuDNN F32-output variant is also slower than BF16 output
(`0.330934 ms` window and `2.161277 ms` global), so changing cuDNN's output
dtype is not the missing root optimization.

The cuDNN SDPA integration now has timing breakdowns under
`GGML_CUDA_PROFILE_CUDNN_SDPA_HEAD64=1`. A short 2-frame profile confirms the
reason it loses in the full app
(`outputs/sam3-bf16-cudnn-sdpa-head64-smoke-20260612a/cudnn-timing-summary.json`):
for window attention the steady median is `0.678176 ms` total, split into
`0.083360 ms` Q cast, `0.080128 ms` K cast, `0.119232 ms` V pack/cast,
`0.323008 ms` cuDNN execute, and `0.065216 ms` output layout conversion. For
global attention the steady median is `2.336352 ms` total, split into
`0.082592 ms` Q cast, `0.079872 ms` K cast, `0.119360 ms` V pack/cast,
`1.985952 ms` cuDNN execute, and `0.069952 ms` output conversion. The current
default ggml head64 profile is about `0.45 ms` for window and `1.78 ms` for
global on the same shapes, so the root issue is not just the SDPA kernel; it is
the surrounding layout and dtype contract. The next viable attention
optimization should either keep Q/K/V in the cuDNN-native low-precision layout
before the call and feed the projection without converting back, or specialize
the existing ggml head64 path rather than wrapping cuDNN behind extra copies.

The FATTN64 scheduler/config sweep was also refreshed with the synchronized
FATTN profile parser. On the current SAM3 BF16 head64 shapes, the existing
`ncols1=64,ncols2=1` MMA path is still the best local kernel configuration.
For the steady-state profile rows, forced `ncols1=64` measured `0.436160 ms`
median for window and `1.832256 ms` for global attention. Forced `ncols1=32`,
`16`, and `8` all regressed (`0.542080/2.164512 ms`,
`0.612896/2.600032 ms`, and `1.191072/6.662528 ms` for
window/global respectively), and forcing TILE or VEC fell back to the much
slower `parallel_k_combine` schedule (`~1.5 ms` window and `~11.5 ms`
global). Evidence is in
`outputs/sam3-bf16-fattn64-config-sweep-20260612a/summary.json`.

The BF16 QKV-chain path was also checked with a diagnostic BF16-Q-to-F16 MMA
FATTN variant. It changed only the opt-in QKV-chain path and left the default
precision contract untouched. The synchronized 2-frame profile showed a small
FATTN-local reduction: all FATTN rows went from `104.501376 ms` to
`102.762496 ms`; SAM3 ViT window head64 went from `35.385888 ms` to
`34.303328 ms`, while global head64 was flat-to-worse (`15.901088 ms` to
`16.021920 ms`). Profile-free 5-pair timing did not justify keeping the code:
normal BF16 measured `170.8 ms` mean / `164.0 ms` median, QKV-chain measured
`182.8 ms` mean / `176.0 ms` median, and QKV-chain plus BF16-Q-to-F16 measured
`181.6 ms` mean / `175.0 ms` median
(`outputs/sam3-bf16-qkv-chain-bf16q-to-f16-20260612a/`). This rejects BF16-Q
conversion as a root optimization; the next attention win must remove or fuse
the Q/K/V materialization boundary rather than change only the Q conversion
dtype.

The same sweep rechecked the opt-in combined contiguous K/V F32-to-F16
conversion kernel, `GGML_CUDA_ENABLE_FATTN_F32_TO_F16_PAIR_VEC2=1`. In the
synchronized FATTN profile it was mixed: global attention improved slightly
from `1.832256 ms` to `1.815104 ms` steady median, while window attention
regressed from `0.436160 ms` to `0.440032 ms`. A profile-free six-pair
application A/B then preserved exact JSONL parity for all pairs
(`mask_hash_equal_rows=3/3`, zero bbox/score/mask-area deltas), but timing did
not justify a default: current default was `163.583 ms` mean / `163.6 ms`
median, while pair-vec2 was `169.45 ms` mean / `164.45 ms` median. The
per-pair deltas were `+1.5`, `+1.0`, `-0.7`, `+15.6`, `-0.1`, and
`+17.9 ms`, so this remains diagnostic-only
(`outputs/sam3-bf16-fattn-pair-vec2-ab-20260612a/paired-summary.json`). The
root attention work should not spend more effort combining standalone K/V
conversion launches; it needs to remove the conversion/materialization boundary
or change the head64 MMA body itself.

A wider non-contiguous V conversion probe was also added as diagnostic-only:
`GGML_CUDA_ENABLE_FATTN_F32_TO_F16_NC_DIM0_VEC4=1` uses a `float4` load and two
RN `half2` stores for F32 tensors whose dim0 is contiguous, four-wide, and
stride-aligned. It preserves exact JSONL parity against the default checked
path (`mask_hash_equal_rows=3/3`, zero bbox/score/mask-area deltas), but it is
not a default optimization. The synchronized 2-frame profile slightly reduced
local V conversion versus the existing default vec2 path (`3.696512 ms` to
`3.534656 ms` across `56` window head64 calls, `0.515808 ms` to `0.487328 ms`
across `8` global calls), while the 10-frame application A/B was slower:
vec4 opt-in measured `174.333 ms` mean / `174.0 ms` median and the existing
vec2 default measured `172.333 ms` mean / `172.0 ms` median
(`outputs/sam3-bf16-fattn-nc-dim0-vec4-probe-20260612a/`). This reinforces
the current conclusion: the next useful attention change should not optimize
the standalone V conversion launch in isolation; it needs to remove the
materialization boundary or feed FATTN/projection with a layout that avoids the
conversion.

The next root-optimization probe therefore moved from standalone FATTN
conversion kernels to ViT MLP fusion. A new diagnostic target,
`sam3_cudnn_mlp_bench`, builds a cuDNN frontend graph for the dominant SAM3 ViT
shape `fc1: [1024,4736] x [1024,5184] -> [4736,5184]` plus bias and exact GELU.
This is not wired into inference yet. It exists to answer whether cuDNN can
beat the current cuBLASLt + separate activation/cast boundary under the same
BF16-input contract.

On the current RTX 5070 Ti laptop run (`warmup=5,iters=30`), the standalone
numbers are:

| path | mean ms | median ms | note |
|---|---:|---:|---|
| cuDNN `fc1+bias+GELU`, BF16 output | `0.846342` | `0.844880` | exact GELU, zero workspace |
| cuDNN `fc1+bias+GELU`, FP32 output | `1.291130` | `1.284736` | needs `33554688` bytes workspace |
| cuBLASLt BF16 `fc1+bias` | `0.890309` | `0.888512` | current matmul/bias baseline |
| cuBLASLt BF16 `fc1+bias+GELU` epilogue | `1.301130` | `1.290080` | slower than BF16-output cuDNN |
| cuBLASLt full BF16 MLP chain | `2.118207` | `2.128192` | `fc1+GELU+fc2` diagnostic chain |

Small-shape checks were added to the same target. For
`input=32,hidden=64,cols=17`, BF16-output exact GELU matched the local CPU
reference after BF16 rounding with `check_bad=0`, `check_max_abs=0`, and
`check_max_rel=0`; the FP32-output variant had `check_bad=0`,
`check_max_abs=7.91102648e-05`, and `check_max_rel=0.000466041849`. This makes
the BF16-output cuDNN fc1+GELU path a plausible next integration candidate,
provided the full SAM3 JSONL mask hashes and Python boundary tensors are
rechecked after wiring it into ggml.

The cuDNN full-MLP graph is not currently a viable replacement. Both col-major
and row-major attempts failed to build an execution plan for
`fc1+GELU -> fc2`; the col-major failure reports cuDNN's epilogue layout
restriction, and the row-major attempt still reports no valid execution plans.
The actionable next step is therefore narrower: add an opt-in SAM3 ViT
`fc1+bias+exact-GELU -> BF16` cuDNN graph path, feed its BF16 output directly to
the existing `fc2`, and accept it only if full JSONL parity is exact and
unsynchronized Track/fr improves against the current default.

That opt-in ggml CUDA path now exists as a diagnostic implementation. It is
guarded by both `GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16=1` and
`GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16_UNSAFE_RUN=1`, and it only intercepts
SAM3 ViT `mlp.lin1` BF16 `mul_mat + bias + GELU_ERF` when the output is BF16.
The first wiring attempt treated the ggml weight memory as `[hidden,input]` and
was numerically wrong; the fixed path views the ggml `[input,hidden]` weight as
a transposed tensor via cuDNN strides. With profiling enabled on a 2-frame smoke,
the fixed path fired for the ViT fc1 rows (`count=65`) and the cuDNN execute
time averaged `0.891696 ms`, with steady rows around `0.85 ms`
(`outputs/sam3-bf16-cudnn-mlp-fc1-probe-20260612a/`).

It is still not accepted. On a 3-frame smoke, current default measured
`163.2 ms/fr`, `SAM3_BF16_VIT_MLP_CHAIN=1` with cuBLASLt GELU_ERF fusion
measured `181.6 ms/fr`, and the fixed cuDNN fc1 path measured `160.9 ms/fr`.
However, default versus fixed cuDNN was not exact: `mask_hash_equal_rows=2/3`,
`min_bbox_iou=0.99906097739232`, `max_bbox_delta_px=0.031000000000062755`,
and `max_score_abs_delta=0.0014759999999999773`
(`outputs/sam3-bf16-cudnn-mlp-fc1-probe-20260612a/ab/default-vs-cudnn-fixed.json`).
The speed signal is real enough to keep the opt-in probe, but the acceptance
criterion remains unchanged: it cannot become default, or be counted as a
same-precision Python-beating path, until full JSONL/mask parity is exact.

A follow-up verified the cuDNN graph's ggml-transposed weight interpretation in
`sam3_cudnn_mlp_bench --ggml-weight-layout`. Small-shape CPU-reference checks
passed for BF16 output with `check_bad=0`, `check_max_abs=0`, and
`check_max_rel=0`; the full SAM3 ViT shape still measured `0.843324 ms` mean /
`0.843232 ms` median. The remaining app-level mismatch is therefore not a
weight-stride bug.

Block-scoped probes also did not find an exact-parity subset. With
`SAM3_BF16_VIT_MLP_CHAIN_BLOCKS=<selector>` plus the cuDNN fc1 probe, 3-frame
SAM3 BF16 results against a stable default baseline were:

| blocks | Track/fr | mask hashes | max bbox delta | max score delta |
|---|---:|---:|---:|---:|
| `1-31` | `159.4 ms` | `0/3` | `0.027 px` | `0.000670` |
| `8-31` | `162.6 ms` | `1/3` | `0.068 px` | `0.001735` |
| `15-31` | `165.8 ms` | `0/3` | `0.043 px` | `0.003306` |
| `23-31` | `169.9 ms` | `0/3` | `0.114 px` | `0.002541` |

The same default rerun compared against itself with exact parity
(`mask_hash_equal_rows=3/3`, zero bbox/score/area deltas), so these failures are
not measurement nondeterminism. The cuDNN fc1 BF16-output path remains useful
as an upper-bound speed probe, but the next accepted optimization needs to
either preserve the current default's F32 `fc1+bias+GELU` boundary exactly or
prove against official Python that the BF16 boundary is the intended same-input
contract.

A F32-output cuDNN fc1 probe was added to check the other side of that
acceptance condition. It uses the same cuDNN frontend graph but writes F32,
is guarded separately by `GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_F32=1` and
`GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_F32_UNSAFE_RUN=1`, and only runs when the
existing `GELU_ERF` fusion scan has already matched the SAM3 ViT `mlp.lin1`
pattern. On a 3-frame SAM3 BF16 smoke it fired `96` times and measured
`1.299578 ms` mean / `1.272928 ms` median per cuDNN execute
(`outputs/sam3-bf16-cudnn-mlp-f32-probe-20260612a/`). That is slower than the
BF16-output cuDNN upper-bound and slower than the current default path. It also
does not restore default parity: default versus F32 cuDNN had
`mask_hash_equal_rows=0/3`, `min_bbox_iou=0.9956282237191301`,
`max_bbox_delta_px=0.2339999999999236`, and
`max_score_abs_delta=0.0026269999999999905`. However, cuBLASLt
`BIAS+GELU_ERF` fusion versus F32 cuDNN was exact
(`mask_hash_equal_rows=3/3`, zero bbox/score/area deltas). This means the F32
cuDNN mismatch is not a cuDNN-specific layout bug; it follows the same numeric
contract as the already-rejected fused ERF path. The F32 cuDNN probe therefore
stays diagnostic-only, and the viable same-parity MLP work remains either
speeding the current separated `GELU_ERF -> BF16` dataflow without changing its
outputs, or replacing the whole Python-contract MLP with an exact
official-contract comparison rather than comparing only against current C++
default.

The standalone cuDNN MLP bench now also has `--dump-output` so small synthetic
cases can be compared directly with PyTorch's fused op. For
`input_dim=32`, `hidden_dim=64`, `cols=17`, `--ggml-weight-layout`, and BF16
inputs, `torch.ops.aten._addmm_activation(..., use_gelu=True)` returned BF16.
cuDNN BF16 output matched `1077/1088` elements bit-for-bit
(`98.98897058823529%`), and all mismatches were only `1` BF16 ULP
(`max_abs=0.001953125`, `mean_abs=1.1668485967675224e-05`). cuDNN F32 output
versus the same PyTorch fused BF16 result had `max_abs=0.0015467405319213867`
and `mean_abs=0.00015792169142514467`
(`outputs/sam3-bf16-cudnn-mlp-f32-probe-20260612a/pytorch-contract/torch-vs-cudnn-summary.json`).
This is close enough to justify a future official-boundary dump on real SAM3
ViT activations, but it is not bit-exact evidence. The next acceptance test for
any Python-contract MLP replacement should compare real block tensors from
official Python and C++ before using mask-level IoU as the final gate.

That real-activation boundary check now exists for SAM3. The diagnostic script
`scripts/dump_sam3_vit_mlp_contract.py` patches official Python's
`vitdet.addmm_act` during a normal `build_sam3_predictor` video run and dumps
the selected ViT `mlp.fc1` input plus fused `_addmm_activation(...,
use_gelu=True)` output in ggml's `[C,W,H,B]` raw-F32 dump layout. The companion
C++ stage runner `sam31_vit_block_case` now accepts both SAM3 and SAM3.1
models, so the same dumped block input can be fed through the C++ `mlp_fc1` and
`mlp_gelu` stages.

On a 3-frame SAM3 BF16/TF32 run, block 0 produced the expected real shapes:
Python MLP input `[1024,72,72,1]` and fused output `[4736,72,72,1]`
(`outputs/sam3-real-vit-mlp-contract-20260612b/python/summary.json`). The C++
stage timings on the same real input were `mlp_fc1` mean `0.957361 ms` /
median `0.961668 ms`, then `mlp_gelu` mean `0.463530 ms` / median
`0.456794 ms`. Comparing official Python fused BF16 output against C++ default
GELU output gave `max_abs=0.01845455`, `mean_abs=0.000275933`, `p95=0.00069043`.
After rounding the C++ GELU output to BF16, the comparison improved in mean
error and exact-value count but still was not bit-exact:
`equal_values=15065257/24551424`, `max_abs=0.03125`,
`mean_abs=0.000200785`, `p95=0.0009765625`, `p99=0.0009765625`
(`outputs/sam3-real-vit-mlp-contract-20260612b/mlp-fused-vs-cpp-summary.json`).

This narrows the MLP conclusion. The current C++ default is numerically close to
official Python's fused BF16 MLP boundary on real activations, but not exact.
The previously measured cuDNN/cublasLt fused ERF app paths changed mask hashes,
so they cannot be accepted merely because their local error is small. A serious
Python-contract MLP replacement must either match the official fused boundary
more closely on these real dumps or prove at full tracking level that the
official Python contract, not the current C++ default hash, is the acceptance
target.

A follow-up added a single graph stage, `mlp_fc1_gelu`, to
`sam31_vit_block_case`. This feeds the official Python block-0
`mlp.fc1` input through C++ `fc1+bias+GELU` in one ggml graph, making the
comparison boundary match Python's fused `_addmm_activation(...,
use_gelu=True)` boundary rather than comparing separate stage dumps.

On the same real SAM3 BF16/TF32 block-0 input, with `warmup=5` and
`repeats=31`, the candidate timings were:

| path | mean ms | median ms | cuDNN fires | Python fused BF16 comparison |
|---|---:|---:|---:|---|
| default C++ graph | `1.336887` | `1.336693` | `0` | raw `mean_abs=0.000275933`, `p95=0.000690430`; BF16-rounded `15065257/24551424` exact |
| cuBLASLt ERF fusion flag | `1.346755` | `1.346545` | `0` | raw `mean_abs=0.000252345`, `p95=0.000654697`; BF16-rounded `16069527/24551424` exact |
| cuDNN F32 fused output | `1.355182` | `1.354976` | `36` | same numeric contract as cuBLASLt ERF; slower than default |
| cuDNN BF16 fused output | `0.911168` | `0.910762` | `36` | same numeric contract as default BF16-rounded output |

The full summary is in
`outputs/sam3-real-vit-mlp-contract-20260612d/summary.json`, produced by
`scripts/compare_sam3_vit_mlp_contract.py`.

The current root-optimization result is therefore more precise:
cuDNN BF16 `fc1+bias+GELU` is a real local speed win for the dominant ViT MLP
boundary (`1.336887 / 0.911168 = 1.47x` faster than default at this stage),
but it is still a diagnostic path. It changes the MLP numeric contract relative
to the current C++ default, and the earlier full tracking smoke showed mask
hash changes. Before promoting it to a normal SAM3 path, the acceptance target
must be switched explicitly to official Python parity and verified end-to-end,
or a same-output optimization must be found for the current default contract.

The end-to-end SAM3 BF16/TF32 comparison was refreshed under the same
`model_matrix_compare.py` contract: decoded `960x540`, 3 frames, prompt
`person`, SAM input size `1008`, C++ and Python warmup both `2`, and
`repeats=5`. Current default C++ measured `162.6 ms/fr`, while official Python
measured `120.68043659674004 ms/fr`
(`outputs/sam3-python-contract-default-20260612a/summary.json`). Enabling the
diagnostic cuDNN BF16 MLP path measured `159.74 ms/fr`
(`outputs/sam3-python-contract-cudnn-bf16-mlp-20260612a/summary.json`). That is
only a `1.8%` full-tracking improvement and is still slower than official
Python by about `32.4%`.

Output parity was also made stricter. The new
`scripts/dump_sam3_python_tracking_jsonl.py` emits official Python SAM3 rows in
the same JSONL shape used by `sam3_benchmark`, including bbox, score, mask area,
and FNV-1a mask hash. On the same 3-frame BF16/TF32 input, C++ default versus
cuDNN BF16 matched `2/3` mask hashes, with `min_bbox_iou=0.9990793776706114`
and `max_bbox_delta_px=0.029999999999972715`
(`outputs/sam3-cudnn-bf16-mlp-jsonl-20260612a/default-vs-cudnn-bf16.json`).
Against official Python, however, both C++ default and cuDNN BF16 matched `0/3`
mask hashes and had only about `0.066` minimum bbox IoU:

| comparison | mask hashes | min bbox IoU | max bbox delta px | max score delta |
|---|---:|---:|---:|---:|
| official Python vs C++ default | `0/3` | `0.06572757114856556` | `321.685` | `0.473575` |
| official Python vs C++ cuDNN BF16 MLP | `0/3` | `0.06572757114856556` | `321.669` | `0.475150` |

This makes the next SAM3 root target clearer: the current blocker is not just
ViT MLP throughput. The C++ SAM3 text-init detector/target output is not
aligned with official Python's `add_prompt` output on the same frames, prompt,
precision, and TF32 policy. Until that detector/selection boundary is aligned,
MLP fusion can only be treated as a local performance probe, not as an accepted
same-accuracy Python-beating path.

The benchmark JSONL path now writes SAM3 text-init candidate rows when
`--output-initial-candidates-jsonl` is provided, and the official Python dump
script can emit matching candidate JSONL via `--out-candidates-jsonl`. On the
same frame-0 input, C++ produced one candidate:

`bbox=[434.159,321.685,537.192,401.718]`, `score=0.483414`,
`mask_area=6575`.

Official Python produced two candidates:

| candidate | selected | bbox | score | mask area |
|---:|---|---|---:|---:|
| 0 | false | `[146.0,135.0,291.0,404.0]` | `0.945206` | `19854` |
| 1 | true | `[312.0,0.0,514.0,394.0]` | `0.956989` | `31451` |

Evidence:
`outputs/sam3-text-init-candidates-20260612a/cpp/initial-candidates.jsonl`,
`outputs/sam3-text-init-candidates-20260612a/python/initial-candidates.jsonl`,
and
`outputs/sam3-text-init-candidates-20260612a/python-vs-cpp-default.json`.
The next useful SAM3 optimization pass should therefore start at the
text-init detector boundary: text tokenization/text encoder parity, detector
neck/projection logits, score thresholding/NMS, and mask decoder selection. A
faster MLP does not address the current official-output mismatch.

That boundary is now reproducible. `sam3_benchmark` can dump C++ PCS tensors via
`SAM3_PCS_DUMP_DIR`, and `scripts/dump_sam3_python_tracking_jsonl.py` can dump
official Python detector tensors via `--out-tensor-dir`. The comparison script
`scripts/compare_sam3_text_detector_boundary.py` reads both raw tensor
directories and writes a stable JSON summary. The refreshed run is stored under
`outputs/sam3-text-detector-boundary-20260612b/`.

The result rules out two weaker explanations:

| check | result |
|---|---:|
| token IDs | exact match |
| Python BF16 vs Python FP32 text features | exact match |
| Python BF16 vs Python FP32 pre-NMS probabilities | exact match |
| C++ text features vs official Python | `max_abs=0.10382214188575745`, `mean_abs=0.01623663346023818`, `p95=0.042732954025268555` |
| C++ joint probabilities vs Python pre-NMS probabilities | `max_abs=0.9452025394653766`, `mean_abs=0.019601439290966645`, `p95=0.02460785040164868` |
| C++ boxes vs Python boxes | `max_abs=0.8590332865715027`, `mean_abs=0.1659159291163087`, `p95=0.5289584502577779` |

The query ranking is already different before Python NMS. C++ top joint
probabilities are query `78=0.48341383415886513`, `171=0.42277046275238356`,
`157=0.010039615054118338`; official Python pre-NMS probabilities are query
`171=0.9569892287254333`, `104=0.9452055096626282`,
`92=0.0596877820789814`, with NMS keeping `[104,171]`. Therefore the mismatch
is not caused by Python's NMS suppression alone, and removing C++'s explicit
presence multiplication would still not select the same candidate.

The text encoder was then isolated with a new C++ diagnostic executable,
`sam3_text_encoder_case`, and a matching official Python dump script,
`scripts/dump_sam3_python_text_encoder.py`. The C++/Python run is stored in
`outputs/sam3-text-encoder-boundary-20260612a/`. Token embeddings and positional
embedding addition match exactly (`32768/32768` values equal for both
`text_token_embed` and `text_after_pos_embed`), so tokenizer and those converted
weights are correct. The final text projection remains different:

| comparison | tensor | max abs | mean abs | p95 abs |
|---|---|---:|---:|---:|
| C++ CUDA vs Python | `text_features_2d` | `0.10242116451263428` | `0.016151069417475128` | `0.04240560382604599` |
| C++ CPU vs Python | `text_features_2d` | `0.09437582734972239` | `0.016508427741172227` | `0.04321830868721008` |
| C++ CUDA vs CPU | `text_features_2d` | `0.04697674512863159` | `0.008098508802845572` | `0.021348363161087035` |

Some named intermediate LayerNorm dumps are not reliable acceptance boundaries
because the ggml graph uses in-place/fused nodes and the immediately downstream
`text_block_00_qkv` is much closer than the `after_ln1` dump suggests. The first
actionable boundary should therefore be a consumed tensor such as
`text_block_00_qkv` or the final `text_features_2d`, not an in-place alias-only
debug name. For block 0, `text_block_00_qkv` differs by
`max_abs=0.08606529235839844`, `mean_abs=0.0016446561383448948`, and
`p95=0.0055216670036315856`.

Acceptance criteria for the next parity/performance pass:

1. Add a PCS diagnostic mode that can replace only C++ `text_features` with an
   official Python `text_features_2d` raw dump while keeping the same C++ image
   features and detector graph.
2. If candidate ranking/boxes align under that replacement, fix the C++ text
   encoder numerics first; otherwise continue into fusion encoder and DETR
   decoder with the replacement still active.
3. A parity fix is accepted only when C++ and official Python match the selected
   frame-0 candidate query set before NMS, candidate JSONL, and 3-frame tracking
   JSONL under the same decoded frames, prompt, image size, precision, and TF32
   policy.
4. A performance optimization is accepted only after the parity gate passes and
   paired C++/official-Python timings show C++ faster under the same precision
   contract. The previous local MLP speedup remains diagnostic until this gate
   passes.

The follow-up PCS replacement pass added env-gated raw-F32 diagnostic inputs for
`text_features`, level-2 `image_features`, level-2 `image_pos_embed`, and
`fenc_output`. The loader validates both `.shape` and exact `.bin` byte count so
partial tensor dumps cannot be accepted silently. Official Python tracking
dumps now also capture detector backbone features, detector positional encodings,
pre-NMS probabilities/masks/keep indices, and the fusion-encoder input/output
boundary.

Replacing only `text_features` with the official tracking-path Python dump
proved that text encoder drift is not the primary detector-selection blocker.
The replacement was exact (`8192/8192` equal values), but the selected C++
candidate still stayed on query `78` with the wrong bbox:
`[434.184,321.703,537.192,401.738]` and score `0.482723`
(`outputs/sam3-tracking-text-features-ref-20260612a/boundary-compare.json`).

The image and fusion boundaries are much larger. On the same frame-0 input,
C++ level-2 image features versus official Python `sam3_backbone_fpn_2` differ
by `max_abs=6.237871170043945`, `mean_abs=0.7550213590939392`, and
`p95=1.9482688903808594`. C++ level-2 image positional encodings versus
official Python `sam3_vision_pos_enc_2` differ by `max_abs=2.0`,
`mean_abs=0.13033673016355174`, and `p95=0.7739618808030995`
(`outputs/sam3-image-boundary-20260612a/image-boundary-compare.json`). The C++
fusion output versus official Python `encoder_hidden_states` in the best tested
layout differs by `max_abs=18.35350775718689`,
`mean_abs=1.4780580753027728`, and `p95=4.084059000015257`
(`outputs/sam3-fenc-boundary-20260612a/fenc-output-layout-compare.json`).

The `DotProductScoring` pooling scale was then fixed. C++ had encoded the valid
token mask as `T / num_valid`; Python computes
`(prompt * is_valid).sum(dim=0) / num_valid`, so the correct mask weight is
`1 / num_valid`. This was not a cosmetic change: with official Python
`fenc_output` and `text_features` injected, the previous C++ scores saturated
near-equal for the two true queries. After the scale fix, C++ top joint
probabilities are query `171=0.9558451363041102` and
`104=0.9435245307976087`, matching official Python's top pre-NMS queries
`171=0.9569892287254333` and `104=0.9452055096626282`. Candidate JSONL also
selects the same target-side candidate first:

| path | candidate | selected | bbox | score | mask area |
|---|---:|---|---|---:|---:|
| C++ with Python fenc/text | 0 | true | `[310.646,0.0,514.971,396.256]` | `0.955845` | `30178` |
| C++ with Python fenc/text | 1 | false | `[146.582,135.079,290.329,405.66]` | `0.943525` | `19313` |
| official Python | 0 | false | `[146.0,135.0,291.0,404.0]` | `0.945206` | `19854` |
| official Python | 1 | true | `[312.0,0.0,514.0,394.0]` | `0.956989` | `31451` |

The remaining fenc-injected deltas are no longer detector ranking blockers:
joint probability mean abs is `0.003967022512436211` and bbox mean abs is
`0.049757527163019406` versus official Python
(`outputs/sam3-fenc-ref-scorefix-20260612a/boundary-compare.json`). The mask
logit delta is still large (`mean_abs=5.572314002253336`), so segmentation/FPN0
and FPN1 remain separate parity work.

The default C++ path after the scale fix still does not pass parity. It selects
query `78` with bbox `[434.159,321.685,537.192,401.718]` and score `0.306288`,
while official Python keeps `[104,171]`. Default C++ joint probabilities versus
official Python pre-NMS probabilities still have `max_abs=0.8673180067765892`,
`mean_abs=0.05523515485246906`, and `p95=0.1575302299773757`
(`outputs/sam3-scorefix-default-20260612a/boundary-compare.json`). Therefore the
next root target is no longer DotProductScoring; it is the detector
backbone/neck positional encoding and fusion-encoder dataflow that produces the
wrong `fenc_output` under the normal C++ path.

Updated acceptance criteria for promoting any SAM3 speed path:

1. Default C++ must reproduce official Python's frame-0 pre-NMS top query set
   and selected candidate under the same decoded frame, prompt, input size,
   precision, and TF32 policy.
2. The normal C++ path must meet the same gate without `SAM3_PCS_*_REF`
   replacement. The replacement hooks are diagnostic only.
3. `seg_mask_logits` and candidate mask hashes must be reconciled after the
   detector/fenc query ranking is fixed; fenc-only parity is not sufficient for
   full tracking parity.
4. Only after these parity gates pass should CUDA fusion kernels, cuDNN MLP
   paths, or ggml backend changes be considered accepted performance
   optimizations.

The next diagnostic pass narrowed the fusion-encoder problem further. C++ can
now dump `fenc_layer{i}` intermediate tensors, and the official Python dump
hooks capture the matching encoder layer outputs plus self-attention,
cross-attention, and FFN module outputs. `SAM3_PCS_COMBINED_PROMPT_REF` was
added so the whole prompt entering C++ fenc can be replaced with official
Python `prompt_before_enc`, not only the text portion.

With official Python `image_features`, `image_pos_embed`, and
`combined_prompt` injected, the input boundary is exact:

| Tensor | Result |
|---|---:|
| C++ `image_features` vs transposed Python level-2 backbone feature | `1327104/1327104` exact |
| C++ `image_pos_embed` vs Python encoder `pos_embed` layout | `1327104/1327104` exact |
| C++ `combined_prompt` vs Python `prompt_before_enc` | `8448/8448` exact |
| C++ `combined_bias` vs Python `prompt_mask` converted to additive bias | `33/33` exact |

Despite exact fenc inputs, C++ still selects the wrong query. The first layer
already diverges:

| Boundary | Max abs | Mean abs | P95 abs | Evidence |
|---|---:|---:|---:|---|
| layer0 self-attn output | `19.732791900634766` | `1.8894682616402894` | `4.833091092109677` | `outputs/sam3-fenc-inner-cpp-transposed-20260612a/inner-boundary-compare.json` |
| layer0 cross-attn output | `6.7992448806762695` | `0.5967763846531843` | `1.6249462485313415` | same |
| layer0 final output | `16.983402252197266` | `1.7196411157939049` | `4.477246212959286` | same |
| final fenc output | `18.326122760772705` | `1.4707100605759749` | `4.07356305122375` | same |

This rules out prompt mask polarity, text/geo prompt assembly, and the level-2
positional embedding layout as the primary cause. Disabling the CUDA head64
no-mask fast FATTN path did not change the numbers, and making fenc V
contiguous before FATTN also produced identical output. The next actionable root
target is therefore the fenc layer self-attention contract itself: Q/K/V
projection weight contents, QKV split direction, head packing, and
flash-attention merge layout versus official `MultiheadAttention`.

2026-06-12 follow-up: the self-attention diagnosis above was partly affected by
debug tensor lifetime. Internal fenc tensors that only had a name but were not
marked as graph outputs could be backed by reused gallocr memory after graph
execution. The debug path now marks fenc stage tensors as outputs when
`SAM3_PCS_DUMP_DIR` is set, so stage dumps are stable.

With exact Python image features, position embeddings, and combined prompt still
injected, fenc layer0 self-attention is no longer the primary blocker:

| Boundary | Max abs | Mean abs | P95 abs | Evidence |
|---|---:|---:|---:|---|
| layer0 self-attn Q projection | `0.030271530151367188` | `0.0011250184122205316` | `0.0035049915313720703` | `outputs/sam3-fenc-cross-cpp-transposed-20260612a/inner-boundary-compare.json` |
| layer0 self-attn output projection | `0.010304033756256104` | `0.0006712738189298297` | `0.0018169581890106201` | same |
| layer0 cross-attn Q projection | `0.024658203125` | `0.002056853204817883` | `0.005590903759002619` | same |
| layer0 cross-attn K projection | `0.020456790924072266` | `0.0010371710055055287` | `0.0033054620027542034` | same |
| layer0 cross-attn V projection | `0.012243509292602539` | `0.0007019825513563057` | `0.0023696273565292337` | same |
| layer0 cross-attn output | `1.6543241441249847` | `0.18202441726590585` | `0.4960167974233626` | same |
| final fenc output | `18.326122760772705` | `1.4707100605759749` | `4.07356305122375` | same |

The masked cross-attention fallback was then fixed. The root issue was the
`ggml_permute` contract: its arguments place each original axis into a result
axis, rather than listing source axes in result order. The fallback
`ggml_mul_mat(probs, V^T)` output is laid out as `[Nq, HD, NH, B]`; returning it
with `permute(1,2,0,3)` produced `[NH, Nq, HD, B]`, and the caller then reshaped
that as if it were flash-attention's `[HD, NH, Nq, B]`. The corrected fallback
uses `permute(2,0,1,3)`, producing `[HD, NH, Nq, B]` before the normal
`[D, Nq, B]` reshape.

With exact Python image features, position embeddings, and combined prompt
injected, the fixed layer0 cross-attention core now matches a PyTorch
scaled-dot-product attention reconstruction from the dumped C++ Q/K/V and prompt
mask:

| Boundary | Max abs | Mean abs | P95 abs | Evidence |
|---|---:|---:|---:|---|
| layer0 cross-attn raw/core vs PyTorch reconstruction | `8.423205200003281e-7` | `1.7110526553409775e-8` | `5.25279925861488e-8` | `outputs/sam3-fenc-ca-layoutfix-cpp-transposed-20260612a/inner-boundary-compare.json` |
| layer0 cross-attn output vs official Python hook | `0.007286` | `0.000457357` | `0.001273` | same |
| layer0 final output vs official Python layer output | `0.03758` | `0.00199595` | `0.005634` | same |
| final fenc output vs official Python encoder hidden states | `0.6105175018310547` | `0.03596408077362447` | `0.09430313110351562` | same |

The normal C++ path no longer needs `SAM3_PCS_*_REF` replacement to select the
same initial text-detected person. On the same three decoded frames and prompt,
C++ top joint probabilities are query `171=0.9662050651509954` and
`104=0.954115994020684`; official Python pre-NMS probabilities are
`171=0.9569892287254333` and `104=0.9452055096626282`
(`outputs/sam3-default-after-ca-layoutfix-20260612a/boundary-compare.json`).
The full 200-query joint-probability comparison is now
`mean_abs=0.0013871634506561696`, `max_abs=0.020951068871815984`, and
`p95_abs=0.005276159704153231`.

This is a parity improvement, not a speed win yet. The same 3-frame C++ row
without dump measured `track/fr=180.2 ms`, `p50=180.2 ms`, `p95=184.5 ms`
(`outputs/sam3-default-after-ca-layoutfix-speed-20260612a/run.log`), still slower
than the previously measured official Python row around `120 ms/frame` on this
contract. The next performance target is therefore the now-correct but slow
fenc masked cross-attention path: replace the fallback graph with a CUDA kernel
or supported ggml CUDA path specialized for `Nq=5184, Nkv=33, D=256, NH=8`,
while preserving the parity numbers above.

An A/B that folded `scale + mask add + softmax` into `ggml_soft_max_ext` kept
the same candidate ordering but did not materially improve the 3-frame timing:
`track/fr=181.2 ms`, `p50=181.2 ms`, `p95=186.5 ms`
(`outputs/sam3-default-softmax-ext-speed-20260612a/run.log`). This supports
treating the next speed step as a real CUDA/ggml kernel change, not another
minor graph cleanup.

2026-06-12 later profiling found one real fenc masked-attention graph problem
that could be fixed without changing the attention math. The prompt/token bias
is shared across heads, but the C++ graph expanded it to `[Nkv, Nq, 8, B]`.
ggml CUDA's `FLASH_ATTN_EXT` path only accepts masked attention when the mask
head dimension is broadcastable (`ne[2] == 1`), so this unnecessary head repeat
forced the slow `mul_mat + softmax + mul_mat` fallback. Keeping the mask as
`[Nkv, Nq, 1, B]` lets the direct flash-attention path run.

The profile-only fusion-encoder timing improved substantially: before the
change, `pcs_fusion_encoder` measured `96.201 ms`, dominated by
`MUL_MAT dst=f32[33,5184,8,1]` at `64.799683 ms`
(`outputs/sam3-profile-ca-layoutfix-20260612b/profile-summary.json` and
`outputs/sam3-profile-ca-layoutfix-20260612b/node-summary-split.json`). After
the broadcast-mask change, `pcs_fusion_encoder` measured `29.493 ms`, with the
dominant nodes becoming the six expected `FLASH_ATTN_EXT` calls around
`3.4 ms` each
(`outputs/sam3-fenc-mask-broadcast-profile-20260612a/profile-summary.json`).
This is a local graph improvement, but not yet an end-to-end win: the normal
3-frame row measured `track/fr=180.3 ms`, `p50=180.3 ms`, `p95=184.7 ms`
(`outputs/sam3-fenc-mask-broadcast-speed-20260612a/run.log`). Candidate order
and boxes stayed aligned with the fixed-default run; the top mask hashes changed
to `6bc61d72bd0ba90c` and `2d9aede37a776aa5`, so this needs to be treated as a
numeric-path change rather than a byte-identical output.

Several CUDA fusion A/Bs were rejected:

| Variant | Result | Decision |
|---|---:|---|
| cuDNN F32 MLP FC1/GELU probe | `track/fr=181.3 ms` | no speed win |
| cuBLASLt exact GELU-ERF epilogue | `track/fr=191.9 ms` | slower and changed mask hashes |
| cuBLASLt bias+residual epilogue | `track/fr=181.0 ms` | no win; residual path rejected by memory-range safety checks |
| BF16 ViT MLP chain | `track/fr=197.5 ms` | slower |
| flat ViT MLP chain | `track/fr=180.8 ms` | noise-level change only |
| approximate GELU + cuBLASLt GELU epilogue | `track/fr=191.6 ms` | slower and not the same-precision contract |

The remaining root target is therefore `sam3_encode`: later profile samples
still put the image encoder around `149 ms/frame` after first-frame positional
encoding setup, with the largest buckets in ViT projection/MLP matmuls,
window/global attention, GELU, and QK layout/rope. To beat official Python under
the same input contract, the next accepted change should be a real CUDA/ggml
kernel or fusion for this ViT path, not another environment-toggle experiment.

2026-06-12 strict-harness refresh: after the broadcast-mask change, current C++
default was remeasured with the same 3-frame source, same 1008 SAM input size,
same text prompt, and warmup policy as the official Python row. The comparable
row is still not a win: C++ measured `176.72 ms/frame` mean over five runs
(`min=175.2`, `max=177.3`) while official Python BF16/TF32-on remains
`120.68043659674004 ms/frame`
(`outputs/sam3-current-default-vs-python-20260612a/summary.json`). A
preencoded C++ tracking-only row measured `31.46 ms/frame`, but that is not a
comparable speed claim because Python still includes image encoding in its timed
scope (`outputs/sam3-current-default-preencoded-vs-python-20260612a/summary.json`).
This isolates the remaining gap to image encoding: roughly
`176.72 - 31.46 = 145.26 ms/frame` for C++ encode-side work, versus Python's
`120.68 ms/frame` for encode plus propagation.

The SAM3 benchmark JSONL writer was then fixed to keep the selected text-init
target stable across frames. Before this, `sam3_benchmark` added all text-init
detections to the tracker, but later JSONL/mask/logit output fell back to the
highest-score detection in each propagated frame. In multi-person clips this
could make the report drift away from the frame-0 selected person even when the
tracker itself still had the correct instance. The writer now records the
frame-0 selected detection's `instance_id` and uses it for subsequent output;
frame-0 mask/logit/JSONL output also uses the same selected candidate index as
the initial-candidate JSONL.

With that reporting fix, the same 3-frame BF16/TF32 contract aligns to the
official Python target instead of jumping to the other person. C++ measured
`177.8 ms/frame`, still slower than the official Python row above. The
official-Python versus C++ tracking comparison is now close in boxes but not
mask-exact: `min_bbox_iou=0.979151009380692`,
`max_bbox_delta_px=2.6129999999999995`,
`max_score_abs_delta=0.009195000000000064`,
`max_abs_mask_area_rel_delta=0.011196466892669425`, and
`mask_hash_equal_rows=0/3`
(`outputs/sam3-current-python-parity-20260612c/python-vs-cpp-track.json`).
Frame 1 and frame 2 are within `1 px` max bbox delta
(`IoU=0.9924954489727681` and `0.9923852183650615` respectively); the remaining
frame-0 delta is the detector/mask numerical gap, not the previous output-target
selection bug.

Additional ViT-path A/Bs did not produce an accepted speed win:

| Variant | Result | Decision |
|---|---:|---|
| Python-like BF16 qkv/proj/fc2 output contract | `track/fr=212.6 ms` | slower |
| CUDA Graphs enabled | `track/fr=186.9 ms` | slower |
| cuDNN FC1+GELU F32 after fusion-order fix | `track/fr=191.6 ms` | slower and changed mask hashes |
| cuBLASLt ViT shape algo index 1 for all qkv/proj/fc1/fc2 | `track/fr=180.6 ms` | slower |
| cuBLASLt workspace 256 MiB | `track/fr=177.2 ms` | noise-level only |
| cuBLASLt bias fusion disabled | `track/fr=223.3 ms` | much slower |
| Python-like BF16 output with direct BF16 dst | `track/fr=198.2 ms` | slower and changed mask hashes |
| QKV BF16 chain only | `track/fr=190.8 ms` | slower and changed mask hashes |

The cuDNN FC1+GELU ordering fix is intentionally not enabled by default. It
only makes the explicit `GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_*` path reachable
for diagnosis; the measured SAM3 path remains slower than the existing
cuBLASLt-bias-plus-standalone-GELU path. Independent microbenchmarks also show
that the BF16 ViT FC1 GEMM itself is not the main deficit: cuBLASLt measures
about `0.91 ms` for `[5184,1024] x [4736,1024]`, while a matching PyTorch
autocast linear measures about `0.95 ms`. The next root fix should therefore
target whole-block layout and kernel count: window partition/unpartition,
Q/K rope layout, attention input/output format, residual+norm staging, and
possibly a fused ViT block path that avoids repeatedly materializing Python's
NHWC-style flow as many ggml `[E,N]` intermediate tensors.

The cuDNN BF16 `fc1+bias+GELU` probe was rechecked after the target-stable JSONL
writer fix against the official Python row, not only against an older C++-only
video smoke. With
`GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16=1` and
`GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16_UNSAFE_RUN=1`, the same 3-frame
contract measured `188.3 ms/frame`, slower than the current C++ default
`177.8 ms/frame` and still slower than official Python BF16/TF32-on. The Python
comparison stayed in the same non-exact mask envelope:
`min_bbox_iou=0.9789630505585574`,
`max_bbox_delta_px=2.603999999999985`,
`max_score_abs_delta=0.009062000000000014`,
`max_abs_mask_area_rel_delta=0.010605542251111872`, and
`mask_hash_equal_rows=0/3`
(`outputs/sam3-python-parity-cudnn-mlp-fc1-20260612a/python-vs-cpp-cudnn.json`).
Against the current C++ default it also changed every checked mask hash while
only moving boxes by at most `0.056 px`
(`outputs/sam3-python-parity-cudnn-mlp-fc1-20260612a/default-vs-cudnn.json`).
Decision: keep this path diagnostic-only. The older `160.9 ms/fr` C++-only
signal used different measurement conditions and is not valid evidence for the
official-Python speed goal.

The ggml upstream CUDA history was also rechecked from the current fork branch.
The main upstream candidate is Programmatic Dependent Launch (PDL), but this
repository already has SAM3 evidence rejecting that direction for this path:
previous PDL probes preserved parity but measured slower or flat. The current
upstream PDL patch is broad and touches many generic CUDA launches; it does not
address the current top SAM3 profile buckets, which are ViT projection/MLP
GEMMs plus QKV/RoPE/FATTN materialization. Decision: do not merge or cherry-pick
PDL for this goal unless a new paired A/B on the corrected target-stable harness
shows a real `Track/fr` win. The next implementation target remains a
SAM3-specific ViT dataflow/kernel change, not a global launch-overlap toggle.

An opt-in CUDA fusion for the SAM3 window path now covers
`WIN_UNPART -> residual ADD` with `GGML_CUDA_ENABLE_WIN_UNPART_ADD_FUSION=1`.
The fused kernel writes the residual-added unpartitioned tensor directly and is
restricted to F32, contiguous, same-shape tensors. Its memory guard rejects
overlap with the window source and only allows residual/output aliasing when
the alias is exact per-element in-place. The profile probe fired the fusion
`168` times on the 3-frame harness, matching the window-block count across the
warmup/measured graph executions
(`outputs/sam3-win-unpart-add-fusion-20260612a/profile2/run.err`).

This is not accepted as a default speed optimization. Six paired 3-frame
BF16 runs stayed exact against default (`mask_hash_equal_rows=3/3`,
zero bbox/score/mask-area deltas in every pair), but normal unsynchronized
timing was noise-level: default `178.3167 ms/frame` mean /
`178.2 ms/frame` median, opt-in `178.2 ms/frame` mean /
`177.55 ms/frame` median, with pair deltas from `-2.2` to `+2.8 ms`
(`outputs/sam3-win-unpart-add-fusion-20260612a/ab/summary.json`). Decision:
keep this as a diagnostic dataflow fusion only. It confirms the window
unpartition/residual boundary is not a large enough standalone target; the next
accepted root change still needs to attack the larger ViT GEMM/attention block
dataflow.

The target-stable harness was also used to recheck two attention-path assumptions
that affect the next optimization target. First, forcing the SAM3 ViT head64
MMA attention tile width away from the current default `ncols1=64` is not useful.
On the same 3-frame BF16 text-prompt contract, forced `ncols1=8`, `16`, and `32`
all regressed versus default: default measured `178.07 ms/frame` mean /
`178.1 ms/frame` median, while `8` measured `217.63 / 218.1`, `16` measured
`187.80 / 186.0`, and `32` measured `181.67 / 181.8`. These forced variants
also changed one of the three checked mask hashes in each run, so `ncols1=64`
remains the correct same-output baseline
(`outputs/sam3-fattn64-ncols-restable-20260612a/summary.json`).

Second, the current code intentionally defaults SAM3 F16/BF16 to direct QKV
views. Disabling that path with `SAM3_DISABLE_VIT_DIRECT_QKV_VIEWS=1` preserved
exact JSONL parity in all six pairs, but slowed the corrected 3-frame BF16
contract from `177.95 ms/frame` mean / `177.65 ms/frame` median to
`185.2167 ms/frame` mean / `184.65 ms/frame` median. Pair deltas were all
regressions, from `+3.2` to `+11.2 ms/frame`
(`outputs/sam3-direct-qkv-current-ab-20260612a/summary.json`). This supersedes
older notes that rejected direct QKV views under earlier measurement conditions:
direct QKV views should stay default-on for current SAM3 F16/BF16 CUDA work.

A cuBLASLt shape-autotune prototype was also tested for the dominant SAM3 ViT
BF16/F16 bias matmul shapes. The prototype benchmarked returned cuBLASLt
heuristics during warmup and cached the fastest candidate for the same shape.
It selected heuristic index `1` for the QKV shape and left the main
projection/MLP shapes effectively on index `0`, but the application A/B did not
win. Six paired 3-frame BF16 runs were exact (`mask_hash_equal_rows=3/3`, zero
bbox/score/mask-area deltas in every pair), yet autotune regressed from
`177.5667 ms/frame` mean / `177.4 ms/frame` median to `178.8833 ms/frame`
mean / `178.6 ms/frame` median. Pair deltas were `[+3.3, +2.6, +0.2, +1.5,
+0.0, +0.3] ms/frame`
(`outputs/sam3-cublaslt-autotune-ab-20260612a/summary.json`). Decision: do not
keep the autotune implementation. The current cuBLASLt heuristic path is already
near the best available candidate for these shapes; the remaining speed gap
requires a larger dataflow change than selecting among returned GEMM algorithms.

The follow-up shape-specific sweep reached the same conclusion without the
prototype autotune machinery. Each major ViT shape was forced independently on
the corrected 3-frame BF16 contract: `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV=1`,
`..._VIT_ATTN_PROJ=1`, `..._VIT_MLP_FC1=1`, `..._VIT_MLP_FC2=1`, and
`..._VIT_MLP_FC2=2`. All variants preserved exact JSONL parity against default
for all three pairs, but none produced an accepted speed win. Mean/median deltas
versus default were: QKV `+1.93 / +0.30 ms`, attn-proj `-0.03 / +1.0 ms`,
FC1 `+1.43 / +1.30 ms`, FC2 index 1 `+1.53 / +1.0 ms`, and FC2 index 2
`+2.47 / +2.30 ms`
(`outputs/sam3-cublaslt-shape-single-ab-20260612b/summary.json`). Decision:
leave all SAM3 ViT BF16 GEMM heuristic defaults unchanged.

The next accepted CUDA change is smaller but robust: the existing
`CONV_TRANSPOSE_2D -> REPEAT(channel bias) -> ADD` fusion now defaults to the
add-output-only path. This removes the standalone bias broadcast/add after SAM3
neck transposed convolutions when the raw convolution output is not otherwise
needed. Set `GGML_CUDA_DISABLE_CONV_TRANSPOSE_BIAS_FUSION=1` to force the old
path; the full write-both-outputs mode remains opt-in with
`GGML_CUDA_ENABLE_CONV_TRANSPOSE_BIAS_FUSION=1`.

The first non-interleaved probe was too noisy, so the acceptance run used six
interleaved disabled/default pairs on the corrected 3-frame SAM3 BF16
text-prompt contract. The new default preserved exact JSONL parity against the
disabled control (`min_bbox_iou=1.0`, zero bbox/score/mask-area deltas, and all
three checked mask hashes equal). Track timing improved in all six pairs:
disabled mean/median `177.7667 / 177.95 ms/frame`, default mean/median
`176.7 / 176.9 ms/frame`, paired mean delta `-1.0667 ms/frame`, paired median
delta `-1.15 ms/frame`
(`outputs/sam3-conv-transpose-bias-defaulted-paired-20260612a/summary.json`,
`outputs/sam3-conv-transpose-bias-defaulted-paired-20260612a/compare-disabled-default-run1.json`).
This does not close the Python gap by itself, but it removes a real neck-side
kernel and memory pass without changing outputs, so it is accepted as a default
cleanup while the larger ViT dataflow work remains open.

The current official-Python comparison after this default change still does not
meet the goal. On the same decoded frames, same 1008 encode size, same text
prompt, same frame 1+ encode+propagate scope, BF16 Python autocast, TF32 enabled,
and two warmup sessions, the 3-repeat matrix reports C++ `sam3-bf16` at
`177.1667 ms/frame` mean / `177.0 ms/frame` median, while official Python SAM3
measures `121.0811 ms/frame` mean / `121.2405 ms/frame` median. The comparable
speed ratio is Python/C++ `0.68343x`, so C++ remains about `1.46x` slower
(`outputs/model-matrix-sam3-bf16-conv-transpose-defaulted-20260612a/summary.json`).
The remaining root work must therefore target the ViT block dataflow, not
another small neck cleanup.

Two nearby alternatives were rechecked and rejected on the same current branch.
First, CUDA conv2d is already using cuDNN for the SAM3 neck 3x3 convolutions.
Disabling it regressed the short BF16 contract from default `176.9 ms/frame`
mean / `177.1 ms/frame` median to `503.3667 / 504.0 ms/frame`, and also moved
one frame-0 mask hash with only subpixel bbox/score deltas
(`outputs/sam3-cudnn-conv2d-check-20260612a/summary.json`,
`outputs/sam3-cudnn-conv2d-check-20260612a/compare-default-disabled-run1.json`).
This confirms the neck 3x3 path is already on the necessary cuDNN backend; it is
not where the Python gap will be closed.

Second, cuBLASLt direct BF16 destination remains a bad tradeoff. Enabling broad
`SAM3_BF16_VIT_LINEAR_OUTPUT=1` with `GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1`
regressed from default `176.3667 / 176.3 ms/frame` to
`176.8333 / 176.7 ms/frame` and changed one checked mask hash. Restricting the
BF16 destination to global attention projection preserved exact JSONL parity,
but still regressed to `178.6 / 177.2 ms/frame`
(`outputs/sam3-bf16-direct-dst-current-smoke-20260612a/summary.json`,
`outputs/sam3-bf16-direct-dst-current-smoke-20260612a/compare-default-direct-all-run1.json`,
`outputs/sam3-bf16-direct-dst-current-smoke-20260612a/compare-default-direct-global-proj-run1.json`).
The accepted path is still to change the ViT layout/kernel flow itself rather
than forcing low-precision materialization at the existing ggml tensor
boundaries.

### 2026-06-12 SAM3 Neck 1x1/Scale3 Probes

A fresh synchronized node profile was taken on the current target-stable SAM3
BF16 harness (`outputs/sam3-current-profile-20260612-codex-continue-b/`). For
steady `sam3_encode`, the ranking is unchanged: `proj/fc2` matmuls are about
`36.5 ms`, MLP `fc1` about `28.9 ms`, QKV about `18.9 ms`, window head64
attention about `11.4 ms`, and MLP GELU about `7.1 ms` per encode. The
remaining gap to official Python is therefore still the ViT block dataflow, not
one isolated neck or elementwise operation.

Two neck-side structural probes were checked because they are exact-dataflow
candidates outside the already-rejected ViT toggles. First,
`SAM3_ENABLE_NECK_1X1_MULMAT=1` routes SAM3/SAM3.1 neck `1x1` convs through a
flat matmul diagnostic path. It is not accepted. Six paired SAM3 BF16 runs
measured old/default `175.9833 ms/frame` mean / `175.9` median versus opt-in
`176.1167` mean / `176.15` median, and the opt-in changed the frame-0 mask hash
in every pair (`mask_hash_equal_rows=2/3`, `max_bbox_delta_px=0.025`,
`max_score_abs_delta=0.000071`). Evidence:
`outputs/sam3-neck-1x1-mulmat-paired-20260612a/summary.json`.

Second, `SAM3_DISABLE_NECK_SCALE3=1` skips SAM3 neck scale 3. The normal
tracking, PCS, and segmentation paths use scales 0, 1, and 2, so this stayed
byte-identical in the checked JSONL rows. It is still not a speed default:
six paired runs measured scale3-enabled `176.55 ms/frame` mean / `176.35`
median versus scale3-skipped `177.4833` mean / `176.55` median. Pair deltas
were noisy and included a `+5.8 ms` regression, so the default remains to build
scale 3; the disable env is diagnostic only
(`outputs/sam3-neck-scale3-skip-paired-20260612a/summary.json`).

These probes reinforce the current prioritization. Neck cleanups are now too
small or too noisy to close the goal. The next accepted optimization needs to
remove materialization or kernel boundaries inside the SAM3 ViT block itself:
QKV/RoPE/FATTN/projection layout, MLP `fc1 -> GELU -> fc2` dataflow, or a
larger fused block path that preserves the target-stable tracking JSONL.

### 2026-06-12 Encode Overhead And Python Cache Contract

A profile pass was added to separate graph construction, allocation, upload,
and CUDA compute on the current SAM3 BF16 target-stable harness
(`outputs/sam3-profile-overhead-20260612a/`). This ruled out graph setup as the
main cause of the Python gap. `sam3_encode` compute was still the dominant cost:
9 samples averaged `180.206 ms` with a steady-state floor around `142-156 ms`.
By contrast, encode graph build averaged `0.281 ms`, graph allocation
`0.370 ms`, and image upload `0.710 ms`. Propagation had the same shape:
`propagate_single` compute averaged `11.617 ms`, while graph build/allocation
were only `0.107 ms` and `0.224 ms`. The next speedup should therefore not focus
on ggml graph lifetime reuse alone; it must reduce the CUDA work and tensor
materialization inside the video/image encoder path.

Three nearby low-risk CUDA toggles were rechecked and rejected:

| Probe | Result | Decision |
|---|---:|---|
| Disable BF16 bridge before ViT attention projection | `176.9 ms/frame` vs default `176.3` | no win |
| Disable BF16 bridge before ViT `fc2` | `190.4 ms/frame` | slower |
| Disable all BF16 ViT linear input bridges | `181.1 ms/frame` | slower |
| `GGML_CUDA_ENABLE_NORM_1024_AFFINE_AXIS0=1` | exact JSONL, `176.7` vs default `175.3 ms/frame` | slower |
| `GGML_CUDA_NORM_1024_MODE=1` | `195.1 ms/frame`, `mask_hash_equal_rows=2/3` | slower and not exact |
| `GGML_CUDA_NORM_1024_MODE=3` | `184.1 ms/frame`, `mask_hash_equal_rows=2/3` | slower and not exact |

The attempted RoPE post-`CONT` removal was also rejected. An opt-in prototype
returned the `CONCAT` result directly from `sam3_apply_rope` and tried to make
the CUDA RoPE-pair fuser write that destination. The graph then stopped hitting
the existing `CONT_ROPE_PAIR_FUSION` path and produced only `subgraph`/`memory`
rejections under `GGML_CUDA_PROFILE_ROPE_PAIR_FUSION=1`; the profiled 2-frame
run measured `237.4 ms/frame`
(`outputs/sam3-rope-no-post-cont-profile-20260612a/`). The prototype was not
kept. The current `CONT` is part of the exact fusion contract, so removing it is
not a speedup unless the fuser is redesigned together with ggml alias/memory
range handling.

The more important discovery is that the current official-Python comparison is
not a same-execution-contract per-frame encode comparison. A stage profile of
official SAM3 BF16/TF32 on the same extracted frames showed a strong frame
asymmetry (`outputs/sam3-python-detector-stage-profile-20260612a/profile.json`):

| Python stage | Frame | Mean |
|---|---:|---:|
| `detector.forward_video_grounding_multigpu` | 1 | `168.371 ms` |
| `model.run_backbone_and_detection` | 1 | `168.625 ms` |
| `tracker.track_step` | 1 | `18.086 ms` |
| `model._run_single_frame_inference` | 1 | `192.015 ms` |
| `detector.forward_video_grounding_multigpu` | 2 | `0.013 ms` |
| `model.run_backbone_and_detection` | 2 | `0.264 ms` |
| `tracker.track_step` | 2 | `23.056 ms` |
| `model._run_single_frame_inference` | 2 | `28.406 ms` |

This explains why official Python can report about `121 ms/frame` over the
3-frame harness while C++ reports about `176-177 ms/frame`: Python performs the
expensive detector/backbone work for frame 1 and caches or prepares later frame
features, making frame 2 nearly free on the detector side. The current C++
benchmark instead encodes each propagated frame independently. To make the goal
meaningful and then win it, the next root implementation target is not another
single-op CUDA toggle; it is C++ parity with Python's multi-frame video grounding
and feature-cache execution contract. Acceptable next work is one of:

- implement SAM3 video detector/backbone lookahead or batched-frame encoding in
  C++ and reuse the resulting feature cache during propagation;
- or split the benchmark into explicit contracts: single-frame encode latency,
  cached propagation latency, and multi-frame video inference latency, then
  compare Python and C++ only within the same contract.

Until that contract is aligned, a raw `Track/fr` average is useful as a local
C++ regression metric, but it is not sufficient evidence that C++ is slower or
faster than official Python at the same algorithmic work.

### 2026-06-12 SAM Decoder 1x1 Projection Default

The first accepted propagation-side CUDA dataflow change after the Python cache
contract check is in the SAM mask decoder high-resolution feature projection.
The old SAM3 path handled `conv_s0` and `conv_s1` as ggml 1x1 convolutions by
materializing `[W,H,256]` feature tensors before `IM2COL`. In the propagation
profile this made `prop_trk_s0 (permuted) (cont)` cost about `0.72 ms` per
propagation and fed a second large `IM2COL[256,288,288]` pass before the actual
projection.

CUDA now defaults SAM decoder `conv_s0`/`conv_s1` to a direct 1x1 `mul_mat`
projection from the existing `[C,W,H]` tracker features. This keeps the large
input in channel-first layout, applies the `[Cout,Cin]` 1x1 weight as a matrix
multiply, and only materializes the smaller `[W,H,Cout]` result needed by the
upscale-add path. Set `SAM3_DISABLE_SAM_DEC_1X1_MULMAT=1` to force the old
conv/im2col route; `SAM3_ENABLE_SAM_DEC_1X1_MULMAT=1` remains as an explicit
opt-in for non-default experiments.

Acceptance on the preencoded/cached propagation contract was exact and large
enough to matter. Six paired SAM3 BF16 runs with
`SAM3_BENCH_PREENCODE_TRACK_FRAMES=1` were byte-identical at the tracking JSONL
level (`mask_hash_equal_rows=3/3`, zero bbox/score/mask-area deltas in every
pair). Timing improved from disabled mean/median `31.1 / 30.9 ms/frame` to
new-default mean/median `28.6667 / 28.35 ms/frame`, with paired mean delta
`-2.4333 ms/frame`
(`outputs/sam3-samdec-1x1-mulmat-paired-preencoded-20260612a/summary.json`).
That puts C++ cached propagation in the same range as the official Python
profiled cached frame (`model._run_single_frame_inference` frame 2:
`28.406 ms` mean).

The encode-included 3-frame contract also improved exactly. Four paired SAM3
BF16 runs preserved exact JSONL parity and reduced `Track/fr` from disabled
mean/median `177.125 / 177.0 ms/frame` to new-default `174.2 / 173.65 ms/frame`,
paired mean delta `-2.925 ms/frame`
(`outputs/sam3-samdec-1x1-mulmat-paired-full-20260612a/summary.json`). A
post-default smoke confirmed the disable switch still recovers the old path:
preencoded disabled/default measured `32.2 -> 28.9 ms/frame`, and full
disabled/default measured `176.0 -> 173.4 ms/frame`, both with exact JSONL
parity
(`outputs/sam3-samdec-1x1-mulmat-defaulted-smoke-20260612a/`).

The node profile after defaulting shows the intended shape change:
`prop_trk_s0 (permuted) (cont)` and the corresponding 1x1 `IM2COL` no longer
appear as propagation hotspots. The replacement work is a `MUL_MAT` to
`[32,82944]` plus a much smaller `[288,288,32]` materialization
(`outputs/sam3-samdec-1x1-mulmat-defaulted-profile-20260612a/propagate-hotspots.json`).
The remaining cached-propagation gap is now dominated by memory-attention
`MUL_MAT`/`FLASH_ATTN_EXT` blocks and mask hypernetwork matmuls, not by the old
high-resolution 1x1 im2col path.

### 2026-06-12 Propagation Encoded-Feature Alias Probe

After the SAM decoder 1x1 projection change, the next propagation-side probe
looked at GPU-to-GPU copies into the per-frame propagation graph. The old graph
must not use `state.neck_trk[*]` directly with their original metadata because
that would pull in the image-encoder dependency tree. An opt-in alias helper now
creates fresh leaf metadata that points at already-encoded tracker feature
buffers, avoiding the copy without importing the original encoder graph. The
safe diagnostic envs are:

- `SAM3_ENABLE_PROP_ALIAS_FEATURE_INPUTS=1` for encoded tracker features only;
- `SAM3_ENABLE_PROP_ALIAS_CONSTANT_INPUTS=1` for constant PE/RoPE inputs only.

The combined feature+constant alias path was explicitly rejected. When both
groups were aliased together, the short bbox-only JSONL changed scores and boxes
even though the empty bbox-only mask hashes were unchanged. The implementation
therefore exposes no broad "alias everything" switch, and constant aliasing is
disabled whenever feature aliasing is active.

Feature aliasing is parity-safe in the checked cached-propagation contract.
Six paired SAM3 BF16 runs with `SAM3_BENCH_PREENCODE_TRACK_FRAMES=1` were exact
against default (`mask_hash_equal_rows=3/3`, zero bbox/score/mask-area deltas in
every pair). The median improved from `28.0 ms/frame` to `27.05 ms/frame`, with
median delta `-0.95 ms/frame`; one noisy outlier made the mean delta only
`-0.233 ms/frame`
(`outputs/sam3-prop-feature-alias-paired-20260612a/summary.json`). A profile run
with the feature alias showed the same direction in synchronized compute spans:
`prop_graph_compute_total` averaged `9.648 ms`, compared with the preceding
post-1x1 profile around `12.721 ms`
(`outputs/sam3-prop-feature-alias-profile-20260612a/profile-summary.json`).

The encode-included 3-frame contract was exact but not a clear speed win:
four paired runs measured default mean/median `174.125 / 174.15 ms/frame` versus
feature-alias `174.225 / 174.4 ms/frame`
(`outputs/sam3-prop-feature-alias-full-paired-20260612a/summary.json`). Decision:
keep feature aliasing diagnostic for the Python-cached propagation comparison
and for future Rust wrapper experiments, but do not default it yet. A default
change needs either a larger paired run showing no full-contract regression or a
public API/benchmark mode whose contract is explicitly "already-encoded cached
propagation".

Superseded note: the 2026-06-28 required-E2E split below retested this under the
current artifact-free cached-tail contract with full mask artifact parity. SAM3
CUDA now defaults both feature and constant propagation aliases, with explicit
disable switches for the old upload path.

### 2026-06-12 Cached Propagation Contract Measurement

The benchmark now has an explicit cached-propagation measurement contract. The
C++ CLI accepts `--preencode-track-frames` to pre-encode frames `1..N-1` before
the timed propagation loop, and `--timed-start-frame <n>` to execute earlier
propagated frames while excluding them from `Track/fr` aggregation. Metadata
JSONL rows include both `preencoded_track_frames` and `timed_start_frame`.
`scripts/model_matrix_compare.py` now passes these as CLI options instead of
using `SAM3_BENCH_PREENCODE_TRACK_FRAMES`, records the same contract fields in
`cpp-results.json`, and only marks a speed row comparable when C++ and Python
share the same track scope, timed-start frame, warmup policy, prompt, frame
range, and model input size.

The official SAM3 Python runner cannot be measured per frame from
`handle_stream_request` yield intervals: the first yield may already include
work for multiple frames, and later yields can be output-delivery overhead only.
The Python side of the matrix therefore wraps
`predictor.model._run_single_frame_inference` and records synchronized CUDA time
for each real frame inference. This matches the earlier stage-profile
interpretation and avoids the invalid sub-millisecond frame-2 timing produced by
yield-interval measurement.

Current same-contract SAM3 BF16/TF32-on cached frame-2 result on
`data/test_video.mp4` (`data/test_video.mp4` is the local bedroom-video symlink),
3 repeats, 3 frames, `--timed-start-frame 2`, and 2 warmup sessions:

| Implementation | Contract | Mean | Median / stdev |
|---|---|---:|---:|
| C++ CUDA `sam3-bf16` | preencoded frame 2 cached propagation | `28.1667 ms` | `27.7 / 0.9866 ms` |
| Official Python SAM3 BF16, TF32 on | internal `_run_single_frame_inference` frame 2 | `28.6745 ms` | `28.7791 / n/a in summary` |

The comparable ratio is Python/C++ `1.0180x`, so C++ is currently about `1.8%`
faster on this narrow cached-propagation contract
(`outputs/model-matrix-sam3-bf16-cached-frame2-internal-20260612a/summary.json`).
The Python internal events also confirm the contract split: measured runs showed
frame 1 around `190-196 ms` and frame 2 around `28-29 ms`, while C++ frame-2
cached propagation stayed around `27.5-29.3 ms`.

This is not yet the full goal. It proves that the accepted SAM decoder 1x1
change brought cached propagation to Python parity/slight win, but the larger
multi-frame contract still needs C++ feature-cache/video-grounding parity so the
expensive frame-1 detector/backbone work is not repeated as independent
per-frame encoding.

### 2026-06-12 SAM3 Static Memory K/V Cache

SAM3 now mirrors the SAM3.1 static spatial memory K/V cache for the safe common
case used by cached propagation: one selected non-perceiver conditioning memory
slot. At memory-slot creation, C++ precomputes each memory-attention layer's
spatial `ca.k_proj(prompt + prompt_pos + tpos)` and `ca.v_proj(prompt)` outputs
on the backend. During propagation, the graph reuses those spatial K/V tensors
and computes only the dynamic pointer-token tail before concatenation.

The cache is enabled by default for native SAM3 condition slots and can be
disabled with `SAM3_DISABLE_MEM_CA_KV_CACHE=1` for A/B checks. The default path
is intentionally narrow: it caches the selected condition slot at temporal
position `0`, which is the stable cached-propagation contract measured below.
The fallback path remains the previous full prompt projection path.

An experimental broader cache exists behind
`SAM3_ENABLE_MULTI_SLOT_MEM_CA_KV_CACHE=1`. It precomputes spatial K for every
temporal-position row and can cover more selected memory slots, but it is not an
accepted default optimization yet. On the 5-frame C++-only cached-propagation
repeat it preserved exact JSONL parity but measured slower than the disabled
path (`29.96 ms` mean versus `28.78 ms`; evidence:
`outputs/model-matrix-sam3-bf16-cached-5f-static-kv-multislot-20260612/summary.json`
and
`outputs/model-matrix-sam3-bf16-cached-5f-static-kv-disabled-20260612/summary.json`).
It should be treated as a diagnostic path until the precompute placement and
multi-slot reuse contract are redesigned.

Acceptance checks on the same 3-frame SAM3 BF16 cached-propagation contract:

| Check | Result | Evidence |
|---|---:|---|
| Cache disabled vs enabled JSONL parity | exact, `mask_hash_equal_rows=3/3`, zero bbox/score/mask-area deltas | `outputs/sam3-static-mem-kv-cache-20260612d/disabled-vs-enabled.json` |
| Single A/B screen | `29.9 ms` disabled vs `26.8 ms` enabled | `outputs/sam3-static-mem-kv-cache-20260612d/` |
| Gated default same-contract C++ mean, 5 runs | `26.74 ms`, median `26.8`, sd `0.4219` | `outputs/model-matrix-sam3-bf16-cached-frame2-static-kv-gated-20260612/summary.json` |
| Official Python SAM3 BF16/TF32-on mean, 5 runs | `28.9194 ms`, median `28.6612` | same summary |
| Comparable Python/C++ ratio | `1.0815x` | same summary |

This is the first refreshed cached-propagation row where C++ has a statistically
cleaner same-contract lead over official Python SAM3 BF16/TF32. The scope is
still the frame-2 cached propagation contract with preencoded tracking frames;
it does not by itself complete SAM3.1 parity or the full encode-included video
contract.

Longer 5-frame rows are currently diagnostic rather than accepted broad speed
claims. They are useful for exposing cache-coverage and staging costs, but the
Python event distribution includes large frame-1/frame-3 work that is not the
same narrow cached-frame-2 slice. Direct C++ versus Python claims should
therefore use the gated 3-frame same-contract row above until the full
multi-frame feature-cache contract is made identical.

### 2026-06-12 SAM3 Batched ViT/Neck Probe

A test-only batched SAM3 ViT benchmark was added as
`sam3_vit_batch_bench`. It repeats one decoded image into a synthetic batch and
can measure the ViT alone or ViT plus the tracker neck. The benchmark also reads
back the final tensor after timing and compares repeated-image batch lanes. For
windowed intermediate tensors it compares logical batch groups, not adjacent
window indices.

This exposed a correctness blocker in `ggml_win_part`: the CUDA kernel launch
covered the full destination, but the kernel-local `total` omitted the batch
factor. For B>1, windows after the first batch were left uncomputed. The fix
makes `ggml_win_part`/`ggml_win_unpart` shape and CPU/CUDA kernels batch-aware,
with `dst->ne[3] = windows_per_image * batch` during partition and restored
batch at unpartition. A stage probe confirmed the failure disappeared at the
first windowed block:

| Probe | B | Lane max abs diff | Lane mean abs diff |
|---|---:|---:|---:|
| prefix only | 2 | `0` | `0` |
| block 0 `window_part` after fix | 2 | `0` | `0` |
| block 0 `window_unpart` after fix | 2 | `0` | `0` |
| full ViT after fix | 2 | `0` | `0` |

The second blocker was performance in tracker-neck transposed convolution. The
existing k=2/stride=2 CUDA fast path used GEMM only for `batches == 1`; B>1 fell
back to the slow direct kernel. The fast path now uses
`cublasGemmStridedBatchedEx` for B>1 with the same packed 2x2 sub-kernels and
the existing batch-aware scatter. This reduced ViT+neck B=2 from the earlier
`476.675 ms` total (`238.337 ms/frame`) to `321.887 ms` total
(`160.944 ms/frame`) while preserving repeated-lane parity
(`lane_max_abs_diff=0`).

Current measurements on the same SAM3 BF16 model and sample frame:

| Probe | B | Mean total | Mean per frame | Lane parity |
|---|---:|---:|---:|---:|
| ViT only | 1 | `144.152 ms` | `144.152 ms` | n/a |
| ViT only | 2 | `300.803 ms` | `150.402 ms` | exact |
| ViT + tracker neck | 1 | `155.492 ms` | `155.492 ms` | n/a |
| ViT + tracker neck | 2 | `321.887 ms` | `160.944 ms` | exact |

Decision: the batch path is now correctness-safe for repeated-image SAM3 ViT
and neck probes, but it is not yet a speed win per frame. The next optimization
target is not "batching by itself"; it is the remaining per-token ViT work
(`MUL_MAT`, `FLASH_ATTN_EXT`, elementwise residual/GELU) and the actual
Python-equivalent feature-cache/video-grounding contract that avoids redundant
frame-1 style encoding.

### 2026-06-12 SAM3.1 Propagation Optimization Probes

SAM3.1 F16 mask-init propagation was remeasured against official Python fp16
with TF32 enabled on the same synthetic `320x240`, `center@7` contract. The
strict acceptance rule remains: C++ variants must preserve the default C++ mask
and should not worsen the official-Python mask comparison before being promoted
from diagnostic flags.

The F16 memory-attention activation split is currently diagnostic only:

| Variant | C++ default XOR | Python XOR | C++ `propagate_encoded_ms` mean / median | Decision |
|---|---:|---:|---:|---|
| default | `0` | `276` | `17.784 / 17.552 ms` | baseline |
| `SAM31_F16_MEM_ATTN_ATTENTION_ACTIVATION=1` | `0` | `276` | `17.738 / 17.792 ms` | parity-safe, not a clear speed win |
| `SAM31_F16_MEM_ATTN_ACTIVATION=1` | `4` | `278` | `17.475 / 17.582 ms` | not accepted: mask drift and no stable Python win |

Evidence:
`outputs/sam31-mask-init-matrix-f16-attn-split-15run-20260612e/summary.json`.
Earlier probes also rejected CUDA FATTN Stream-K forcing and contiguous-V for
this contract because they either worsened parity or did not improve runtime
(`outputs/sam31-mask-init-matrix-f16-streamk-20260612d/summary.json`,
`outputs/sam31-mask-init-matrix-f16-memact-contigv-20260612d/summary.json`).

Feature-input aliasing was also measured and kept diagnostic only. It preserved
the default C++ mask exactly but was slower in the single-frame mask-init
contract:
`outputs/sam31-mask-init-matrix-f16-prop-alias-20260612d/summary.json`.

An environment-gated SAM3.1 propagation graph cache was added for sequence
experiments:

- `SAM31_ENABLE_PROP_GRAPH_CACHE=1` keeps the SAM3.1 propagation
  `ggml_context`, graph, allocator, and input/output tensor handles on the
  tracker when shape, activation flags, cached PE tensors, and precomputed
  spatial K/V tensor pointers are unchanged.
- The cache is disabled automatically for propagation dump mode and does not
  combine with feature-input aliasing.
- It is not enabled by default.

Single-frame mask-init does not measure a useful graph-cache hit because the
smoke benchmark creates a fresh state/tracker and propagates only one frame per
measured run. In that contract, graph-cache variants were exact-mask parity but
slower:

| Variant | C++ default XOR | C++ `propagate_encoded_ms` mean / median |
|---|---:|---:|
| default | `0` | `17.633 / 17.785 ms` |
| graph cache + CUDA graphs | `0` | `18.751 / 18.905 ms` |
| graph cache without CUDA graphs | `0` | `17.962 / 18.042 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-prop-graph-cache-20260612a/summary.json` and
`outputs/sam31-mask-init-matrix-prop-graph-cache-nograph-20260612a/summary.json`.

The smoke benchmark now supports `--num-frames` so sequence-oriented changes can
be measured on one tracker. On the same synthetic setup, graph cache preserved
the frame-1 mask byte-for-byte and gave only a small sequence-side improvement:

| Frames | Variant | Propagated frames | Frame-1 mask | `propagate_encoded_avg_ms` |
|---:|---|---:|---|---:|
| 8 | default | `7` | same SHA | `18.303 ms` |
| 8 | graph cache | `7` | same SHA | `18.087 ms` |
| 20 | default | `19` | same SHA | `21.657 ms` |
| 20 | graph cache | `19` | same SHA | `21.337 ms` |

Evidence:
`outputs/sam31-seq8-default-20260612a/summary.json`,
`outputs/sam31-seq8-prop-graph-cache-20260612a/summary.json`,
`outputs/sam31-seq20-default-20260612a/summary.json`, and
`outputs/sam31-seq20-prop-graph-cache-20260612a/summary.json`.

Decision: graph caching is exact-parity and useful as a sequence diagnostic, but
the measured gain is only about `1-2%`, so it is not the root optimization needed
to beat official Python with margin. The next high-value work should return to
the compute hotspots: D32 self/cross FATTN, 256x5184 memory-attention
`MUL_MAT`, and possible CUDA fusion around attention projections/residuals/FFN
where Metal has hand-tuned coverage or where ggml currently dispatches many
small kernels.

### 2026-06-12 SAM3.1 CUDA FATTN/QKV Follow-up

The next CUDA pass focused on root memory-attention hotspots rather than tiling
flags alone. The node profile still points at SAM3.1 propagation
`FLASH_ATTN_EXT` and many `MUL_MAT dst=f32[256,5184]` calls. Three candidate
families were tested:

1. fuse SAM3.1 memory self-attention Q/K/V projection at the graph level;
2. tune D32 MMA FATTN for layer-0 self-attention;
3. route head-dim-32 memory attention through the existing cuDNN SDPA path.

Q/K/V fusion is exact-mask parity but not a speed win. The split-view version
kept the default mask hash but was slower because the fused output has to be
split and made contiguous before the existing RoPE/attention path. A B=1
direct-view version avoided the extra Q/K/V `cont` nodes, but the 15-run result
still did not beat default:

| Variant | Default XOR | Python XOR | `propagate_encoded_ms` mean / median / sd |
|---|---:|---:|---:|
| default | `0` | `276` | `17.455 / 17.464 / 0.593 ms` |
| direct fused SA QKV | `0` | `276` | `17.499 / 17.688 / 0.594 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-direct-fused-sa-qkv-15run-20260612a/summary.json`.
Decision: graph-level direct fusion stays diagnostic; materialized fused weights
were tested next.

Materialized memory self-attention QKV weights are now the accepted default
SAM3.1 propagation optimization. At model load, SAM3.1 builds one backend tensor
for each memory-attention layer's concatenated SA Q/K/V weight and bias. This
removes the graph-level Q/K/V concat path while preserving the downstream
RoPE/FATTN contract. It adds a small one-time model-load copy, but the hot
propagation graph is unchanged numerically.

15-run comparison on the existing F16 SAM3.1 model:

| Variant | Default XOR | Python XOR | Python IoU | `propagate_encoded_ms` mean / median / sd |
|---|---:|---:|---:|---:|
| legacy default | `0` | `276` | `0.9809049` | `17.935 / 17.875 / 0.565 ms` |
| direct fused SA QKV | `0` | `276` | `0.9809049` | `17.558 / 17.471 / 0.577 ms` |
| materialized fused SA QKV | `0` | `276` | `0.9809049` | `17.437 / 17.416 / 0.822 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-materialized-fused-sa-qkv-15run-20260612a/summary.json`.

After promotion, a 5-run default-vs-legacy smoke confirmed that the normal
`default` variant now uses the materialized path and preserves the exact mask
hash:

| Variant | Default XOR | Python XOR | `propagate_encoded_ms` mean / median |
|---|---:|---:|---:|
| legacy default | `0` | `276` | `17.472 / 17.572 ms` |
| default | `0` | `276` | `16.779 / 16.687 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-default-materialized-legacy-5run-20260612a/summary.json`.
The path is enabled by default for SAM3.1 and can be disabled with
`SAM31_DISABLE_MATERIALIZED_FUSED_MEM_ATTN_SA_QKV=1` for A/B comparisons.
`scripts/run_sam31_mask_init_matrix.py` keeps `legacy-default` as that baseline.

A related converter hypothesis was rejected. The F16 converter's broad
`"token"` exclusion leaves `cross_attn_token_to_image` and
`cross_attn_image_to_token` projection weights in F32. Re-converting those
projection weights to F16 did make the graph profile show F16 weights, but it
worsened the Python mask comparison and did not improve the 15-run runtime:

| Model/variant | Python XOR | Python IoU | `propagate_encoded_ms` mean / median / sd |
|---|---:|---:|---:|
| existing F16 + materialized QKV | `276` | `0.9809049` | `17.437 / 17.416 / 0.822 ms` |
| re-converted token-projection-lowp + materialized QKV | `278` | `0.9807666` | `17.881 / 17.985 / 1.068 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-token-proj-lowp-15run-20260612a/summary.json`.
Decision: do not change converter precision rules for this optimization pass.

Two graph copy-prune candidates were also tested after materialized QKV became
the default. They preserve the exact C++ mask hash but did not produce a stable
runtime win:

| Variant | Default XOR | Python XOR | Python IoU | `propagate_encoded_ms` mean / median / sd |
|---|---:|---:|---:|---:|
| default | `0` | `276` | `0.9809049` | `17.0787 / 16.9130 / 1.0094 ms` |
| direct memory-output + direct final-query copy prune | `0` | `276` | `0.9809049` | `17.3939 / 17.2850 / 1.0059 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-direct-copy-prune-15run-20260612a/summary.json`.
The individual `direct-mem-output` 15-run was also exact-mask parity but mixed
mean/median (`17.184 / 17.077 / 0.547 ms`) against its paired default
(`17.126 / 17.337 / 0.495 ms`), recorded in
`outputs/sam31-mask-init-matrix-direct-mem-output-15run-20260612a/summary.json`.
Decision: keep these as diagnostic variants only; do not promote either copy
prune path without a larger paired win.

Layer-0 self-attention D32 MMA is the fastest candidate, but it is not an
accepted same-precision optimization because the mask changes and the official
Python comparison also worsens. A D32 `ncols1` override was added for diagnosis,
matching the existing D64 override style, and all tested `ncols1` values kept the
same approximate drift:

| Variant | Default XOR | Python XOR | Python IoU | `propagate_encoded_ms` mean / median |
|---|---:|---:|---:|---:|
| default | `0` | `276` | `0.9809049` | `17.430 / 17.468 ms` |
| layer0 SA MMA default ncols | `110` | `284` | `0.9803038` | `14.048 / 13.798 ms` |
| layer0 SA MMA ncols8 | `111` | `285` | `0.9802344` | `14.973 / 14.699 ms` |
| layer0 SA MMA ncols16 | `110` | `286` | `0.9801678` | `14.273 / 14.112 ms` |
| layer0 SA MMA ncols32 | `110` | `284` | `0.9803038` | `14.828 / 14.197 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-l0-ncols-20260612a/summary.json`.
Decision: the speed is real, but this is a precision-contract problem, not an
`ncols` scheduling problem. The next viable path is a parity-preserving D32
kernel or a controlled acceptance criterion based on official-Python quality,
not enabling layer-0 MMA as-is.

The same layer-0 MMA path was rechecked against official-Python quality across
four synthetic masks to see whether the C++ default-hash drift could still be
accepted under a Python-quality contract. The answer is no: it is fast, and it
even improves the bottom-band mask, but it worsens three of four Python
comparisons.

| Case | Variant | Python IoU / XOR | Default XOR | `propagate_encoded_ms` mean / median |
|---|---|---:|---:|---:|
| `bottom-band@7` | default | `0.9950739` / `62` | `0` | `17.226 / 17.136 ms` |
| `bottom-band@7` | layer0 SA MMA | `0.9956308` / `55` | `21` | `13.926 / 13.891 ms` |
| `center@7` | default | `0.9809049` / `276` | `0` | `18.336 / 17.088 ms` |
| `center@7` | layer0 SA MMA | `0.9803038` / `284` | `110` | `13.332 / 13.187 ms` |
| `left-wide@7` | default | `0.9670609` / `532` | `0` | `16.793 / 16.734 ms` |
| `left-wide@7` | layer0 SA MMA | `0.9639913` / `581` | `85` | `14.931 / 13.608 ms` |
| `small-center@7` | default | `0.8484787` / `747` | `0` | `16.842 / 16.736 ms` |
| `small-center@7` | layer0 SA MMA | `0.8279634` / `865` | `122` | `14.045 / 13.167 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-l0sa-quality-coverage-5run-20260612a/summary.json`.
Decision: do not use layer-0 MMA as the default, even under a relaxed
official-Python quality criterion. It remains a useful upper-bound speed
diagnostic for a future parity-preserving D32 kernel.

The parity-preserving D32 TILE scheduling path was checked next because the
node profile shows `sam31_mem_attn_layer0_sa_fattn` as the propagation hotspot.
The current default node-profile row is about `4.23 ms` for that one FATTN node
(`outputs/sam31-node-profile-materialized-qkv-20260612b/profile-summary.json`).
Changing only the NVIDIA FP16 D32/ncols=8 TILE config from `256 threads,
64-row tile` to `128 threads, 128-row tile` reduced the node-profile sample to
about `4.13 ms`, but the normal smoke did not show a reliable wall-clock win.
Expanding to a `256-row tile` regressed the node-profile sample to about
`4.22 ms`
(`outputs/sam31-node-profile-fattn-tile-d32n8-128x128-20260612a/profile-summary.json`,
`outputs/sam31-node-profile-fattn-tile-d32n8-128x256-20260612a/profile-summary.json`).
Decision: do not change the global D32/ncols=8 TILE config in this pass. A
meaningful fix needs a dedicated SAM3.1 D32 parity-preserving kernel, not only
a generic TILE launch-parameter tweak.

Extending the existing FATTN VEC path to D32 was also rejected. A temporary
opt-in dispatch for `sam31_mem_attn_layer0_sa_fattn` compiled, but the smoke was
slower (`propagate_encoded_ms=20.390 ms`) and changed the model output
substantially (`obj_score_logit` changed from about `3.199` to `0.681`). Evidence:
`outputs/sam31-d32-vec-smoke-20260612a/summary.json`. Decision: do not broaden
the generic VEC kernel to D32; a viable kernel must preserve the TILE/F32
numerical path rather than converting K/V to F16 like the existing VEC helper.

A naive F32-preserving D32 self-attention kernel was also rejected. It used one
CUDA block per `(head, query)` and kept Q/K/V in F32, but the two-pass
dot/softmax/value implementation was far too slow and still changed the output
because its accumulation order did not match the accepted TILE path closely
enough. The smoke measured `propagate_encoded_ms=200.216 ms`, `score=1.022005`,
and `obj_score_logit=3.141527`, versus the default smoke's normal
`~16-17 ms`, `score=1.062546`, `obj_score_logit=3.199229`. Evidence:
`outputs/sam31-d32-f32-self-attn-smoke-20260612a/summary.json`. Decision: the
next viable D32 work must be tile-structured like the existing FATTN TILE kernel
or fuse into that kernel, not a per-query scalar softmax kernel.

cuDNN SDPA head32 can execute SAM3.1 memory attention only after F16 attention
activation and contiguous V are enabled, but that path is both slower and less
accurate in the current graph:

| Variant | Default XOR | Python XOR | `propagate_encoded_ms` mean / median |
|---|---:|---:|---:|
| default | `0` | `276` | `18.514 / 17.556 ms` |
| F16 attention activation | `5` | `277` | `16.946 / 17.117 ms` |
| F16 attention + cuDNN head32 + contiguous V | `110` | `288` | `19.026 / 18.942 ms` |

Evidence:
`outputs/sam31-mask-init-matrix-cudnn-head32-20260612a/summary.json`.
Decision: do not promote cuDNN head32 for this path. It is useful as a benchmark
reference, but conversion/layout costs plus numerical drift make it worse than
the current ggml FATTN path.

Current status: materialized SAM3.1 memory-attention SA QKV is accepted and
default-on. It is an exact-parity improvement, but it is not enough by itself to
beat official Python with margin under the same input/precision contract. The
remaining work should focus on a parity-preserving D32 FATTN kernel, reducing
the 256x5184 projection/FFN kernel count, and fusing residual/bias/activation
work without introducing graph-level concat/copy overhead or worsening the
official-Python mask comparison.

### 2026-06-28 SAM3 Session-Cache Timing Contract Refresh

The SAM3 BF16/TF32 comparison was refreshed on the current C++23/xmake branch
with the same decoded `960x540` three-frame clip, text prompt `person`, SAM
input size `1008`, and two C++ warmup sessions. The plain C++ encode-included
row is still not a valid direct speed claim against official Python because the
measured tracking scopes differ:

| Row | Track/fr mean | Track/fr median | Scope | Comparable speed claim |
|---|---:|---:|---|---|
| C++ default | `169.7667 ms` | `169.9 ms` | per-frame C++ preprocess/encode/propagate | no |
| official Python BF16/TF32 | `109.4520 ms` | `109.2920 ms` | Python `start_session` cache + propagate events | no |

Evidence:
`outputs/model-matrix-sam3-bf16-current-encode-20260628a/summary.json`.

The official Python event profile for the same short-video contract shows frame
1 as the expensive encode step and frame 2 as cached propagation. The existing
C++ `--preencode-track-frames` mode was too narrow for this comparison because
it pre-encoded frames `1..N-1`. A new benchmark mode,
`--preencode-cached-tail-frames`, now keeps frame 1 as normal
encode+propagate timing and pre-encodes only frames `2..N-1`. The matrix harness
exposes the same mode as `--cpp-preencode-cached-tail-frames` and marks it as
the SAM3 short-video session-cache scope.

Five C++ cached-tail runs versus the reused official Python BF16/TF32 baseline:

| Row | Track/fr mean | Track/fr median | sd | Speed ratio |
|---|---:|---:|---:|---:|
| C++ cached-tail | `99.860 ms` | `99.8 ms` | `0.680 ms` | `1.096x` Python/C++ |
| official Python BF16/TF32 | `109.452 ms` | `109.292 ms` | n/a from reused 3-run baseline | baseline |

This is a comparable speed row in the matrix output:
`outputs/model-matrix-sam3-bf16-cached-tail-20260628a/summary.json`.

After the DDEC/LayerNorm diagnostic cleanup, the same cached-tail contract was
remeasured with five C++ repeats and the same five-repeat official Python
BF16/TF32 run. C++ remains faster under the comparable cached-tail scope:

| Row | Track/fr mean | Track/fr median | sd/range | Speed ratio |
|---|---:|---:|---:|---:|
| C++ cached-tail | `97.940 ms` | `97.6 ms` | sd `0.754 ms` | `1.1198x` Python/C++ |
| official Python BF16/TF32 | `109.6705 ms` | `108.8984 ms` | range `108.0667-111.7828 ms` | baseline |

Evidence:
`outputs/sam3-bf16-lnfix-model-matrix-cached-tail-20260628a/summary.json`.
The encode-included row remains non-comparable against Python's session-cache
scope (`166.760 ms` C++ vs `109.6705 ms` Python; same decoded `960x540` source,
same `1008` model input, same prompt) and is recorded in
`outputs/sam3-bf16-lnfix-model-matrix-20260628a/summary.json`.

The cached-tail mode does not change C++ tracking output. Full JSONL comparison
against the normal C++ path was exact: `min_bbox_iou=1.0`, zero bbox and score
deltas, and `mask_hash_equal_rows=3/3`
(`outputs/sam3-python-quality-current-cached-tail-20260628a/cpp-default-vs-cached-tail.json`).
It also preserves the current official-Python quality envelope rather than
fixing it: Python versus C++ cached-tail still has
`min_bbox_iou=0.979151009380692`, `max_bbox_delta_px=2.613`,
`max_score_abs_delta=0.009195`, and `mask_hash_equal_rows=0/3`
(`outputs/sam3-python-quality-current-cached-tail-20260628a/python-vs-cpp-cached-tail.json`).

Two adjacent optimization probes were refreshed and rejected as promotion
candidates:

| Probe | Result | Decision |
|---|---|---|
| synthetic SAM3 ViT batch path | B=2..4 preserved lane parity, but per-frame time stayed slower than B=1 for both ViT-only and ViT+tracker-neck | keep as correctness/perf diagnostic, not a speed path |
| cuDNN BF16 `mlp.fc1+bias+GELU` with BF16 MLP chain | C++ mean `168.04 ms/fr`, slightly faster than default encode-included, but Python quality did not improve and C++ default parity changed (`mask_hash_equal_rows=2/3`) | keep diagnostic-only |

Batch evidence:
`outputs/sam3-vit-batch-current-20260628a/summary.json`. cuDNN evidence:
`outputs/model-matrix-sam3-bf16-cudnn-fc1-chain-20260628a/summary.json` and
`outputs/sam3-python-quality-current-cudnn-fc1-20260628a/`.

Current SAM3 conclusion: under the observed official-Python short-video
session-cache timing contract, C++ BF16/TF32 now beats Python by about `9.6%`
while preserving exact C++ JSONL parity. This is a measurement-contract fix, not
a new CUDA kernel speedup. The remaining SAM3 work is still the model-output
quality gap against official Python and the full non-cached encode path, whose
hotspots remain the ViT `MUL_MAT`, head64 `FLASH_ATTN_EXT`, residual adds, and
MLP GELU/cast boundary.

### 2026-06-28 SAM3 Mask-Output Contract Correction

The model-matrix harness was found to always pass `--bbox-only` to the C++
benchmark. That mode is not the same output contract as official Python SAM3
when comparing masks: C++ legitimately omits full propagated masks in bbox-only
mode, while official Python still emits `out_binary_masks`. The harness now has
`--cpp-mask-output`, which suppresses `--bbox-only` and should be used for any
SAM3 mask-quality or same-output speed claim.

With mask output enabled, C++ still wins the same short-video BF16/TF32
cached-tail speed comparison:

| Row | Track/fr mean | Track/fr median | sd | Speed ratio |
|---|---:|---:|---:|---:|
| C++ cached-tail, mask output | `102.24 ms` | `101.6 ms` | `1.060 ms` | `1.0705x` Python/C++ |
| official Python BF16/TF32 | `109.452 ms` | `109.292 ms` | n/a from reused 3-run baseline | baseline |

Evidence:
`outputs/model-matrix-sam3-bf16-cached-tail-mask-restored-20260628a/summary.json`.

A follow-up boundary pass added comparable Python/C++ dumps for the SAM3
text-init memory slot: `memory_mask_input`, `object_score_logits`, `obj_ptr`,
`stored_spatial_feats`, `stored_spatial_pe`, and `image_features`. It then kept
two official-contract fixes:

- C++ now emits the initial mask-present `object_score_logits=10`, matching
  Python `_use_mask_as_output`.
- When a text detection does not carry a SAM token, C++ now computes the initial
  object pointer through the official mask-input path: 1152 mask input,
  `tracker.mask_downsample` to 288, SAM prompt encoder dense mask embedding, SAM
  mask decoder, then `obj_ptr_proj`. The previous fallback used `no_obj_ptr`.

The first boundary dump for this change was later found to be contaminated by
the benchmark initialization path: the benchmark adds every text-init detection
to the tracker, so the single `SAM3_DUMP_MEM_SLOT_DIR` files were overwritten by
the last added instance rather than the selected target. The dump path now
accepts `SAM3_DUMP_MEM_SLOT_INST_ID`, and the target was remeasured with
`inst_id=1`.

Filtered target boundary result after this change:

| Tensor | Status | max_abs | mean_abs | Evidence |
|---|---|---:|---:|---|
| `object_score_logits` | ok | `0.0` | `0.0` | `outputs/sam3-memory-boundary-20260628i/memory-boundary-summary.json` |
| `stored_spatial_pe` | ok | `0.001948` | `0.000355` | same |
| `obj_ptr` | improved, still failed | `0.101815` | `0.023734` | same |
| `memory_mask_input` | boundary drift | `19.999092` | `0.023489` | same |
| `stored_spatial_feats` | failed | `2.682757` | `0.064292` | same |
| `image_features` | failed | `1.769424` | `0.032442` | same |

The earlier `memory_mask_input mean_abs=1.976486` row was therefore not valid
selected-object evidence. A follow-up changed the SAM3 initial mask prompt from
detector raw-logit thresholding to the official-style video-resolution binary
mask resized to input-mask resolution before `20*x-10`. This improved the
three-frame final mask result even though strict raw tensor parity is still not
closed. A no-sigmoid direct-memory variant (`20260628j`) made the raw
`memory_mask_input` mean slightly closer but regressed final mask IoU, so it was
not kept as the default.

With the target fixed, the remaining `memory_mask_input` drift is concentrated
at the selected-mask boundary; the 1152 mask-input binary IoU is `0.983091` on
the retained default. The object-pointer gap is no longer dominated by the
previous `no_obj_ptr` fallback, but strict parity remains blocked by
detector/mask-input boundary drift and the image-feature path. On the same
three-frame mask-output probe, mask IoU also improved slightly versus the
raw-logits-only fix:

| Frame | Raw-logits-only mask IoU | With mask-input obj_ptr + fractional init mask |
|---:|---:|---:|
| 0 | `0.997966` | `0.997966` |
| 1 | `0.981875` | `0.983801` |
| 2 | `0.986759` | `0.988070` |

Evidence:
`outputs/sam3-memory-boundary-20260628i/mask-iou.json` and
`outputs/sam3-memory-boundary-20260628i/python-vs-cpp-jsonl.json`.

The same mask-output cached-tail speed comparison remained in C++'s favor after
adding the official obj_ptr path:

| Row | Track/fr mean | Track/fr median | sd | Speed ratio |
|---|---:|---:|---:|---:|
| C++ cached-tail, mask output, fractional init mask | `99.12 ms` | `98.8 ms` | `0.779 ms` | `1.1042x` Python/C++ |
| official Python BF16/TF32 | `109.452 ms` | `109.292 ms` | n/a from reused 3-run baseline | baseline |

Evidence:
`outputs/model-matrix-sam3-bf16-cached-tail-mask-fractional-init-20260628a/summary.json`.

#### 2026-06-28 SAM3 Text-Detector Mask-Logit Axis Correction

The text-detector boundary comparison had a mask-axis bug in the diagnostic
script. Official Python dumps `pred_masks` as `[B,Q,H,W]`, while the C++ dump is
serialized as `[W,H,Q]`. The previous comparison used `[H,W,Q]`, which
overstated the mask-logit drift by effectively transposing the mask image. The
comparison now uses `B,Q,H,W -> W,H,Q`.

With the corrected axis on the same BF16/TF32 frame-0 boundary probe:

| Probe | `fenc_output` mean_abs | seg query mean_abs | `seg_mask_embed` mean_abs | `seg_mask_logits` mean_abs | `seg_mask_logits` p99 |
|---|---:|---:|---:|---:|---:|
| normal C++ | `0.118545` | `0.099623` | `0.083296` | `0.774092` | `7.511035` |
| Python `fenc_output` injected | `0.0` | `0.050997` | `0.042528` | `0.445347` | `4.420220` |
| Python seg query injected | `0.118545` | `0.099623` | `0.000780` | `0.174033` | `1.263338` |

Evidence:
`outputs/sam3-text-detector-boundary-20260628c/boundary-compare.json`,
`outputs/sam3-text-detector-boundary-20260628d/boundary-compare.json`, and
`outputs/sam3-text-detector-boundary-20260628e/boundary-compare.json`.

This changes the current quality diagnosis. The mask head is not grossly
miswired; its final einsum matches the dumped C++ tensors (`mean_abs=0.000824`
against a NumPy reconstruction). The remaining detector-mask drift is a
combination of fenc output drift, DDEC query drift, and smaller instance-embed
drift. Frame-0 binary masks are already very close despite hash mismatch:

| Probe | mask IoU | XOR pixels | area delta |
|---|---:|---:|---:|
| normal C++ | `0.997966` | `64` | `-30` |
| Python `fenc_output` injected | `0.998156` | `58` | `-44` |
| Python seg query injected | `0.997839` | `68` | `-40` |

The next parity work should therefore target the remaining fenc/DDEC query
numeric drift and propagated memory-boundary effects, not a wholesale rewrite
of the segmentation-head einsum.

#### 2026-06-28 SAM3 DDEC Boundary Triage

The DDEC boundary probe was tightened after adding same-name C++/Python dumps
for decoder post-norm stages and self/image-attention internals. The initial
large `ddec_layer0_after_sa` diagnostic mismatch was not an attention mismatch:
`sa_q_in`, `sa_q/k/v_proj`, `sa_out`, and `sa_pre_norm` were already close. A
manual NumPy LayerNorm over the dumped `sa_pre_norm` reproduced official Python,
so `sam3_layer_norm()` now uses non-inplace affine `ggml_mul`/`ggml_add` instead
of the previous in-place affine path. This makes the layer-0 self-attention
boundary diagnostic coherent:

| Boundary, Python fenc/pos/prompt fixed | mean_abs | max_abs | p99_abs |
|---|---:|---:|---:|
| `ddec_layer0_sa_pre_norm` | `0.0000924` | `0.001284` | `0.000472` |
| `ddec_layer0_after_sa` | `0.000553` | `0.005605` | `0.002623` |
| `ddec_layer0_after_text_ca` | `0.000474` | `0.005920` | `0.002139` |
| `ddec_layer0_img_ca_out` | `0.021464` | `0.503474` | `0.140438` |
| `ddec_layer0_after_img_ca` | `0.013881` | `0.296288` | `0.091069` |
| final `ddec_normed_output` | `0.051041` | `1.853194` | `0.389376` |

Evidence:
`outputs/sam3-text-detector-boundary-20260628j/boundary-summary.json` and
`outputs/sam3-text-detector-boundary-20260628k/boundary-summary.json`.

This leaves the remaining DDEC parity target narrower: image cross-attention and
its BF16/TF32 activation contract. Keeping `ddec.reference_points.weight` in F32
for newly converted BF16 GGML files made the reference points exact, but did not
materially reduce the DDEC/mask drift (`ddec_normed_output mean_abs=0.051056`,
`seg_mask_logits mean_abs=0.452604`), so it is a converter precision cleanup,
not the root parity fix
(`outputs/sam3-text-detector-boundary-20260628l/boundary-summary.json`).

The refreshed normal C++ path remains in the same binary-mask quality envelope:
frame-0 mask IoU `0.9979661879`, XOR `64 px`, C++ area `31421`, Python area
`31451`. The corrected normal boundary is `fenc_output mean_abs=0.118545`,
`ddec_normed_output mean_abs=0.099623`, and `seg_mask_logits mean_abs=0.774092`
(`outputs/sam3-text-detector-boundary-20260628m/boundary-summary.json`).

The earlier raw-logits pass also identified one quality-side mismatch in the C++
text-init tracker path. PCS detections now retain their low-resolution detector
mask logits for diagnostics, but the retained SAM3 mask-as-output default uses
the official-style video-resolution binary mask resized fractionally to the
tracker input-mask resolution. The raw-logits-only path was useful as an
intermediate check and improved propagated mask IoU versus the old nearest-mask
reconstruction:

| Frame | Before mask IoU | After mask IoU |
|---:|---:|---:|
| 0 | `0.997966` | `0.997966` |
| 1 | `0.965513` | `0.981875` |
| 2 | `0.969832` | `0.986759` |

JSONL box quality also improved on propagated frames (`bbox_iou` now about
`0.9949` on frames 1 and 2), but strict Python mask parity is not closed yet:
`min_mask_iou=0.9818752720263633`, `mean_mask_iou=0.9888666935298045`, and mask
hashes still differ. Evidence:
`outputs/sam3-boundary-current-mask-initlogits-20260628a/python-vs-cpp.json`.

An internal mask-prompt SAM decoder attempt was tested and rejected rather than
kept. The branch built and ran after correcting the prompt tensor size to the
SAM prompt-encoder contract, but it regressed the three-frame probe from
`min_mask_iou=0.981875`, `mean_mask_iou=0.988867` to
`min_mask_iou=0.979356`, `mean_mask_iou=0.987514`. It also added an extra
decoder call during text-init and did not produce a speed-relevant improvement.
Rejected evidence:
`outputs/sam3-boundary-current-mask-prompt-20260628a/python-vs-cpp.json`.

Current SAM3 quality conclusion: the earlier "empty mask" symptom was mostly a
measurement-contract error from bbox-only mode. The retained fixes narrow the
strict mask gap by using the official-style fractional video-mask input for
initial memory conditioning and by using the official mask-input object-pointer
path. The next parity slice should target the still-failing
`memory_mask_input`, `stored_spatial_feats`, and `image_features` boundaries,
then compare first propagation logits under the same mask-output speed contract.

A follow-up diagnostic split added env-only C++ ref injection controls for the
initial SAM3 conditioning memory boundary: `SAM3_DISABLE_DIRECT_INITIAL_MEM_MASK`
forces the older `1152 -> 1008 -> 1152` mask path, while
`SAM3_MEM_MASK_INPUT_REF`, `SAM3_MEM_IMAGE_FEATURES_REF`, and
`SAM3_MEM_OBJ_PTR_REF` can replace the dumped Python tensors for one controlled
boundary probe. These controls are diagnostic only and are not default behavior.

The ref probes show that the remaining SAM3 mask gap is not fixed by forcing the
mask preprocessing path alone. Disabling the direct `1152` initial-memory mask
path reduced raw `memory_mask_input` mean error from `0.023489` to `0.016936`,
but worsened `stored_spatial_feats` mean error from `0.064292` to `0.065353` and
reduced final three-frame mask IoU from `0.983801` min / `0.989946` mean to
`0.983056` / `0.989380`. Therefore the current direct initial mask path stays
default.

Replacing only the initial conditioning `image_features` with the official
Python dump made the initial `stored_spatial_feats` much closer
(`mean_abs=0.008026`, down from `0.064292`), but final masks did not improve
(`0.983582` min / `0.989759` mean IoU). Replacing only `obj_ptr` gave a small
final-mask improvement (`0.984018` min / `0.990079` mean IoU), but it is far
from strict parity. Replacing all three initial conditioning tensors exactly
made `image_features`, `memory_mask_input`, and `obj_ptr` exact, but still left
`stored_spatial_feats mean_abs=0.008618` and broke frame 1 mask area almost
completely. Evidence:
`outputs/sam3-memory-boundary-ref-probes-20260628a-summary.json` plus the per-run
directories `outputs/sam3-memory-boundary-py-maskinput-20260628a`,
`outputs/sam3-memory-boundary-py-imagefeat-condonly-20260628a`,
`outputs/sam3-memory-boundary-py-objptr-20260628a`, and
`outputs/sam3-memory-boundary-py-condmem-all-20260628a`.

Decision: do not promote any of these ref paths. The next useful SAM3 parity
target is the normal C++ tracker image-feature path and SAM mask-input decoder
numerics, not another mask-resize toggle. Any speed comparison should continue
to use the retained C++ default because the diagnostic ref paths either do not
improve output quality or require Python tensors.

### 2026-06-28 SAM3.1 F16 Cache-Contract and ViT Variant Refresh

The current SAM3.1 F16 mask-init comparison was refreshed on the same synthetic
`320x240`, `center@7`, fp16/TF32 contract. The measurement helper now records
the official Python timing contract explicitly. On the two-frame single-GPU
path, official Python prefetches the next frame during the frame-0 cache step,
so compare Python `full_cache_step_ms` against C++ `full_frame_step_ms`. The
encoded-propagation slice is still reported separately for cache-equivalent
diagnosis.

Current default versus legacy materialized-QKV disable:

| Row | Mask parity vs default | Python IoU / XOR | `propagate_encoded_ms` mean / median / sd | Notes |
|---|---:|---:|---:|---|
| C++ default | `0` XOR | `0.9809049 / 276` | `16.313 / 16.318 / 0.225 ms` | materialized memory SA QKV default-on |
| C++ legacy default | `0` XOR | `0.9809049 / 276` | `16.898 / 16.893 / 0.060 ms` | disables materialized QKV |
| official Python fp16/TF32 | n/a | baseline mask | `16.778 ms` | cached-frame propagation event |

Evidence:
`outputs/sam31-mask-init-matrix-current-f16-default-legacy-20260628a/summary.json`.
This means the accepted C++ encoded-propagation slice is now slightly faster
than official Python on this narrow cache-equivalent timing scope, but the
SAM3.1 full frame row remains dominated by image encode (`encode_frame1_ms`
about `150 ms`) and is not a same-contract Python comparison.

A current synchronized CUDA-node profile was also refreshed. It confirms that
normal SAM3.1 encode is still dominated by ViT projection work rather than
preprocess or graph setup: after warmup, `sam31_encode` is about `153-166 ms`,
with top stage sums in the linear projections (`proj-or-mlp-fc2`, `mlp-fc1`,
and QKV), then head64 window/global FATTN. The comparable propagation graph is
about `18-19 ms` after warmup, with `sam31_mem_attn_layer0_sa_fattn` still the
largest single node at about `3.9 ms`. Evidence:
`outputs/sam31-current-profile-20260628a/profile-summary.json`,
`outputs/sam31-current-profile-20260628a/encode-nodes.json`, and
`outputs/sam31-current-profile-20260628a/propagate-nodes.json`.

Several ViT precision/output variants were rechecked before promoting anything:

| Variant family | Parity result | Speed result | Decision |
|---|---|---|---|
| broad `SAM3_BF16_VIT_LINEAR_OUTPUT=1` | exact C++ mask on center, small-center, left-wide, bottom-band | mixed: center/left-wide sometimes faster, bottom-band flat, small-center slower | keep opt-in; not robust enough for default |
| F16 ViT linear-output variants | changed C++ mask by `6-15` pixels depending on split | slower in this contract | reject |
| window/global TILE override | changed C++ mask by `75` pixels and slowed encode heavily | slower | reject |

Evidence:
`outputs/sam31-vit-variant-ab-f16-20260628a/summary.json`,
`outputs/sam31-f16-vit-output-ab-20260628a/summary.json`, and
`outputs/sam31-vit-linear-output-coverage-f16-20260628a/summary.json`.

Current SAM3.1 conclusion at this point: the safe, accepted propagation
optimization remains materialized memory-attention SA QKV. C++ can beat official
Python on the cache-equivalent encoded-propagation slice, but the remaining root
work is either a parity-preserving D32 CUDA FATTN improvement for layer-0 memory
self-attention or a SAM3.1 ViT encode reduction in the projection/attention
path.

Follow-up probes on the same date did not find a new default candidate.
cuBLASLt ViT shape algo overrides were exact only for QKV/proj/fc1, and those
were noise-level or slower; `vit-mlp-fc2-algo1` changed the C++ mask by `7`
pixels. The existing propagation toggles also stayed rejected: graph-cache and
direct-output/copy-prune paths were exact but slower or flat, while forcing
layer-0 memory SA from TILE to D32 MMA reduced encoded propagation
(`16.399 ms` to as low as `13.366 ms`) but changed the mask by about `110`
pixels. Evidence:
`outputs/sam31-vit-algo-ab-f16-20260628a/summary.json` and
`outputs/sam31-prop-kernel-existing-ab-f16-20260628a/summary.json`.

The D32 layer-0 mismatch is not an input-layout bug. Existing Q/K/V dumps show
that Q, K, and V match exactly between TILE and MMA, and the difference starts
at `sam31_mem_attn_layer0_sa_fattn`. Recomputing attention from the real dumps
shows MMA is much closer to simple F32 attention (`mean_abs=0.000512` versus
MMA) than TILE is (`mean_abs=0.007326` versus TILE), but the current TILE
contract remains closer to the retained C++/Python mask envelope. Evidence:
`outputs/sam31-prop-d32-real-snap-20260611/numeric-mode-analysis-20260628.json`.

The cuDNN MLP diagnostic path was extended to accept F16 ViT `mlp.lin1` inputs
for F32 output. It now fires for the SAM3.1 F16 model, but it is not accepted:
`cudnn-mlp-f16-f32` changed the C++ mask by `35` pixels and slowed two-frame
encode from `317.975 ms` to `340.054 ms` on the five-run A/B. It slightly
improved Python mask XOR (`276` to `273`) but not enough to justify the speed
loss or default-parity break. Evidence:
`outputs/sam31-cudnn-mlp-f16-f32-ab-20260628a/summary.json`.

The next accepted CUDA optimization is the SAM3.1 F16 window unpartition
boundary. `WIN_UNPART -> residual ADD` is now default-on and can be disabled
with `GGML_CUDA_DISABLE_WIN_UNPART_ADD_FUSION=1`. The fused kernel writes the
residual-added unpartitioned tensor directly for F32 contiguous same-shape
tensors, guarded against unsafe overlap. The related `WIN_PART -> CPY` fusion is
not accepted as default: even with `GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION=1`, the
current SAM3.1 F16 graph did not satisfy the memory/layout requirements in the
profiled contract, so it remains an opt-in diagnostic candidate.

Final evidence was collected with eight interleaved repeats on the same
`320x240`, `center@7`, fp16/TF32 contract:

| Row | Mask parity vs C++ default | Python IoU / XOR | Full step mean / median / sd | Encoded propagation mean / median / sd |
|---|---:|---:|---:|---:|
| C++ default, `WIN_UNPART -> ADD` fused | `0` XOR | `0.9809049 / 276` | `368.165 / 368.080 / 1.384 ms` | `16.281 / 16.287 / 0.090 ms` |
| C++ old window path | `0` XOR | `0.9809049 / 276` | `373.489 / 373.586 / 2.274 ms` | `16.603 / 16.557 / 0.368 ms` |
| C++ with only unpartition-add disabled | `0` XOR | `0.9809049 / 276` | `373.117 / 373.157 / 1.710 ms` | `17.287 / 16.619 / 1.307 ms` |
| official Python fp16/TF32 | n/a | baseline mask | `454.638 ms` full cache step | `17.013 ms` cached propagation |

The paired delta for old window path minus default is `+5.323 ms` full step
with 95% CI `[3.653, 6.994]`, and `+3.648 ms` for two-frame encode with 95% CI
`[2.916, 4.379]`. The paired delta for only disabling `WIN_UNPART -> ADD` is
`+4.952 ms` full step with 95% CI `[3.150, 6.754]`. The profile smoke confirmed
`WIN_UNPART -> ADD` launched `112/112` candidates with zero rejects, while
`WIN_PART -> CPY` had zero successes in the final default configuration.
Evidence:
`outputs/sam31-win-unpart-default-final-f16-20260628a/summary.json` and
`outputs/sam31-win-unpart-default-final-f16-20260628a/profile.log`.

Under the corrected full-cache contract, this final C++ default is `1.235x`
faster than official Python for the two-frame full step
(`454.638 / 368.165`). On the cache-equivalent propagation-only slice it is
`1.045x` faster (`17.013 / 16.281`). The Python mask envelope is unchanged
(`276` XOR pixels, IoU `0.9809049`), so this is a speed-only accepted change
relative to the previous C++ default.

### 2026-06-28 SAM3.1 Mask-Init Precision Contract And PE Cache Follow-up

The official Python `--dtype fp32 --tf32 off` mask-init path was inspected with
autocast disabled. The hooked ViT linear modules still produced
`torch.bfloat16` outputs, even though their weights and some inputs were
`torch.float32`. Therefore the fair official-Python comparison for this
checkpoint remains the observed BF16/FP16 CUDA contract; the GGML F32 model is a
higher-precision C++ mode, not a same-dtype official-Python speed row. Evidence:
`outputs/sam31-python-dtypes-fp32-20260628/summary.json`.

F32 GGML image encode was also fixed to use cuDNN for F32 input, F32 kernel, and
F32 output `CONV_2D` on CUDA. This removes the old naive F32 convolution
bottleneck. With TF32 enabled, F32 full step improved to `537.092 ms` after the
later PE-cache change, but it is still slower than the official-Python
mixed/BF16-output row and should be reported separately. Evidence:
`outputs/sam31-mask-init-f32-tf32-pe-cache-opt-960x540-20260628/summary.json`.

The accepted default SAM3.1 mask-init optimization in this pass is caching the
spatial PE used by memory-slot storage. The previous path regenerated
`sam3_sinusoidal_pe_2d(H,H,D)` during every add-detection memory encode. The new
path reuses the tracker PE cache and permits empty `spatial_pe_cpu` for SAM3.1
slots, because SAM3.1 precompute/propagation uses `image_pe_cpu` or the tracker
cache for this path.

Five interleaved BF16/TF32 runs on the `960x540`, `center@3` two-frame contract:

| Row | Full step mean / median / sd | Add-detection mean | Encoded propagation mean / median / sd | Python/C++ full ratio |
|---|---:|---:|---:|---:|
| C++ BF16 default before PE cache | `363.220 / 363.506 / 1.931 ms` | about `40 ms` | `15.954 / 15.938 / 0.096 ms` | `1.174x` |
| C++ BF16 default after PE cache | `355.840 / 356.348 / 0.880 ms` | `35.438 ms` | `15.822 / 15.811 / 0.092 ms` | `1.199x` |
| official Python BF16/TF32 | `426.633 ms` full cache step | `10.155 ms` add mask | `17.350 ms` cached propagation | baseline |

The C++ mask hash stayed unchanged versus its default baseline; Python IoU/XOR
remained `0.9873699 / 1590`. Evidence:
`outputs/sam31-mask-init-prop-variants-bf16-960x540-20260628/summary.json`,
`outputs/sam31-mask-init-bf16-pe-cache-opt-960x540-20260628/summary.json`, and
`outputs/sam31-profile-bf16-pe-cache-opt-20260628/profile.log`.

The same variant sweep under official Python fp16/TF32 also remained a C++ win:
the best exact C++ rows were `15.996-16.008 ms` encoded propagation, and the
default row was `16.032 ms`; official Python cached propagation was
`17.371 ms`, and the full-cache/full-frame ratio was about `1.25x` in favor of
C++. Evidence:
`outputs/sam31-mask-init-prop-variants-fp16-960x540-20260628/summary.json`.

An experimental backend memory-slot path was added behind
`SAM31_ENABLE_BACKEND_MEMORY_SLOT=1`. It stores the SAM3.1 mask-init memory
encoder output as a backend tensor and precomputes static memory-attention K/V
from backend tensors, instead of reading `sam31_mem_out`, image features, and PE
back to CPU first. A profiled smoke run confirmed that the opt-in path did not
enter `sam31_mem_output_materialize`; `sam31_mem_ca_kv_precompute_backend` took
about `0.52 ms`, and `sam31_mem_store_slot_backend` took about `0.69 ms`.
Evidence:
`outputs/sam31-profile-backend-memory-slot-20260628/profile.log`.

Five-run BF16/TF32 A/B on the same `960x540`, `center@3` contract:

| Row | Full step mean / sd | Add-detection mean / min / max | Encoded propagation mean | C++ default parity |
|---|---:|---:|---:|---:|
| C++ default after PE cache | `355.553 / 1.726 ms` | `34.752 / 34.563 / 35.124 ms` | `15.892 ms` | baseline |
| C++ `backend-memory-slot` opt-in | `354.335 / 2.699 ms` | `31.672 / 31.071 / 32.447 ms` | `15.857 ms` | IoU `1.0`, XOR `0` |
| official Python BF16/TF32 | `418.957 ms` full cache step | `9.890 ms` add mask | `16.802 ms` cached propagation | reference |

The opt-in path is a real add-detection improvement, but it is not a default
yet in this snapshot. At this point the backend-only slot was still valid only
for the one selected spatial memory slot path that uses precomputed K/V.
Evidence:
`outputs/sam31-backend-memory-slot-ab-bf16-960x540-20260628/summary.json`.

The interactive mask object-pointer graph cache was then moved to model scope,
so the smoke benchmark's warmup run primes the graph and constant dense PE even
though each measured run creates a new state/tracker. This removes graph build
and constant upload from the measured single-detection path. Profile evidence:
`outputs/sam31-profile-interactive-model-cache-20260628/profile.log` and
`outputs/sam31-profile-interactive-backend-model-cache-20260628/profile.log`.

Five-run BF16/TF32 A/B after the model-scoped interactive cache:

| Row | Full step mean / median / sd | Add-detection mean / median / min / max | Encoded propagation mean / median | Python/C++ full ratio |
|---|---:|---:|---:|---:|
| C++ default | `352.603 / 352.166 / 2.092 ms` | `30.886 / 30.855 / 30.251 / 31.632 ms` | `16.012 / 16.016 ms` | `1.209x` |
| C++ `backend-memory-slot` opt-in | `349.389 / 347.079 / 4.855 ms` | `27.870 / 25.741 / 25.416 / 36.796 ms` | `15.869 / 15.862 ms` | `1.220x` |
| official Python BF16/TF32 | `426.150 ms` full cache step | `10.370 ms` add mask | `16.939 ms` cached propagation | baseline |

Both C++ rows stayed exact against the C++ default mask (`default_iou=1.0`,
`default_xor=0`) and retained the same official-Python comparison
(`IoU=0.9873699`, XOR `1590`). In the backend-memory-slot profiled run,
`add_detection_encode_memory` was about `17.0 ms`, model-cached
`interactive_obj_ptr` was about `3.47 ms`, and total `add_detection_ms` was
`26.497 ms`. Evidence:
`outputs/sam31-interactive-model-cache-ab-bf16-960x540-20260628/summary.json`.

The next same-contract reduction kept the mask-prompt math on CPU but removed
avoidable work: bilinear axis maps, the interactive mask-downsample weights, and
the intermediate buffers are now held in the model-scoped interactive cache, and
the resize/conv rows use the existing bounded row parallelism. The computation
order is unchanged for each output element. In the backend-memory-slot profiled
run, measured `sam31_interactive_mask_prompt_prepare` on the warmed path dropped
from about `2.07 ms` to `1.21 ms`; total `interactive_obj_ptr` dropped from
about `3.47 ms` to `2.62 ms`; total `add_detection_ms` was `25.890 ms`.
Evidence:
`outputs/sam31-profile-mask-prompt-cache-20260628/profile.log`.

Five-run BF16/TF32 A/B after the mask-prompt cache/parallel update on
`960x540`, `center@3`:

| Row | Full step mean / median / sd | Add-detection mean / median / min / max | Encoded propagation mean / median | Python/C++ full ratio |
|---|---:|---:|---:|---:|
| C++ default | `350.748 / 350.789 / 0.765 ms` | `29.928 / 29.835 / 29.062 / 30.990 ms` | `15.878 / 15.880 ms` | `1.215x` |
| C++ `backend-memory-slot` opt-in | `346.431 / 346.010 / 1.047 ms` | `24.779 / 24.765 / 23.881 / 25.500 ms` | `15.905 / 15.930 ms` | `1.230x` |
| official Python BF16/TF32 | `426.129 ms` full cache step | `10.011 ms` add mask | `17.213 ms` cached propagation | baseline |

Both C++ rows remained exact against the C++ default mask (`default_iou=1.0`,
`default_xor=0`) and kept the same official-Python comparison (`IoU=0.9873699`,
XOR `1590`). Evidence:
`outputs/sam31-mask-prompt-cache-ab-bf16-960x540-20260628/summary.json`.

A second shape/case check on `320x240`, `small-center@7` also kept exact C++
parity and a Python/C++ full-step win:

| Row | Full step mean / median | Add-detection mean / median | Python/C++ full ratio |
|---|---:|---:|---:|
| C++ default | `348.861 / 348.875 ms` | `27.965 / 28.102 ms` | `1.228x` |
| C++ `backend-memory-slot` opt-in | `345.447 / 344.831 ms` | `23.323 / 23.344 ms` | `1.240x` |
| official Python BF16/TF32 | `428.488 ms` full cache step | `9.877 ms` add mask | baseline |

Evidence:
`outputs/sam31-mask-prompt-cache-ab-bf16-320x240-20260628/summary.json`.

The backend-resident SAM3.1 memory-slot path was then generalized from the
single selected spatial slot case to all selected slots when every slot has
precomputed spatial K/V, and it is now enabled by default. The propagation graph
receives a vector of selected slots, aliases each slot's precomputed K/V tensors,
concatenates the spatial K/V inside the graph, and uploads a small
`D x n_slots` temporal-position delta input so the cached K path still applies
the current `maskmem_tpos_enc` row at propagation time. Non-conditioning frames
are also eligible for the SAM3.1 K/V precompute, because temporal position is no
longer baked into the cached tensor. Set `SAM31_DISABLE_BACKEND_MEMORY_SLOT=1`
or `SAM31_ENABLE_BACKEND_MEMORY_SLOT=0` to force the older CPU-materialized slot
path.

Validation on a short forced multi-slot contract uses
`sam31_tracking_mask_init_smoke --num-frames 4 --recondition-every 1` so frame 2
and frame 3 exercise multiple selected memory slots without waiting for the
default `recondition_every=16` cadence. BF16/TF32 `320x240`, `center@3` results:

| Row | Add-detection | Encoded propagation avg | Track-step avg | C++ parity |
|---|---:|---:|---:|---:|
| C++ default | `28.036 ms` | `39.866 ms` | `196.654 ms` | baseline |
| C++ `backend-memory-slot` opt-in | `21.749 ms` | `34.983 ms` | `192.895 ms` | exact mask hash |

The PGM mask hash for both rows was
`9411ca80e3b82fc12962811a380096c9b7fa13953aaea17f56119597391bcb1a`, and
`score` / `obj_score_logit` were identical. A profile run with the same forced
multi-slot contract showed cached propagation graph sizes of `804`, `824`, and
`844` nodes as the selected slot count grew from one to three, with each
non-conditioning memory store running `sam31_mem_ca_kv_precompute_backend`.
Evidence:
`outputs/sam31-multislot-recond1-default-20260628a/summary.json`,
`outputs/sam31-multislot-recond1-backend-20260628a/summary.json`, and
`outputs/sam31-multislot-recond1-profile-20260628a/profile.log`.

The same forced multi-slot contract also passed F16 and F32 default/backend
checks with exact C++ masks:

| Precision | Default add / prop avg | Backend add / prop avg | Mask hash |
|---|---:|---:|---|
| F16 | `24.378 / 41.225 ms` | `21.572 / 34.659 ms` | `103c428cdd2bcd81afd794a02c1aba1eb7ea77c3a8a36e5a15d4c00d7d45ed4d` |
| F32 | `25.051 / 42.422 ms` | `23.054 / 36.956 ms` | `15b568762cd55abf2a8c52caaf6d714578346e3b237bf15b570bd3e09d0a4d61` |

Evidence:
`outputs/sam31-multislot-recond1-f16-default-20260628a/summary.json`,
`outputs/sam31-multislot-recond1-f16-backend-20260628a/summary.json`,
`outputs/sam31-multislot-recond1-f32-default-20260628a/summary.json`, and
`outputs/sam31-multislot-recond1-f32-backend-20260628a/summary.json`.

After default-enabling backend memory slots, the same BF16 forced multi-slot
contract was rerun without `SAM31_ENABLE_BACKEND_MEMORY_SLOT`; it produced the
same mask hash and `score` / `obj_score_logit` as the explicit backend row.
The opt-out CPU slot path remained exact but slower on the propagation slice:

| Row | Add-detection | Encoded propagation avg | Track-step avg | Mask hash |
|---|---:|---:|---:|---|
| Current default backend slot | `22.136 ms` | `35.218 ms` | `193.311 ms` | `9411ca80e3b82fc12962811a380096c9b7fa13953aaea17f56119597391bcb1a` |
| `SAM31_DISABLE_BACKEND_MEMORY_SLOT=1` | `23.822 ms` | `39.516 ms` | `195.847 ms` | `9411ca80e3b82fc12962811a380096c9b7fa13953aaea17f56119597391bcb1a` |

Evidence:
`outputs/sam31-multislot-recond1-default-backend-on-20260628a/summary.json` and
`outputs/sam31-multislot-recond1-cpu-slot-20260628a/summary.json`.

The existing matrix runner was also refreshed on the standard two-frame,
single-selected-slot contract after default-enabling backend memory slots. On
BF16/TF32 `320x240`, `center@3`, three interleaved repeats showed exact C++
parity between current default and `cpu-memory-slot`; both rows produced mask
hash `1901a8c3f2e46fd7442e8b98e4ea96f1813f7c51dc6beda17f6b7cd353c96121`.

| Row | Full-frame step mean | Add-detection mean | Encoded propagation mean | Python ratio |
|---|---:|---:|---:|---:|
| Current default backend slot | `344.341 ms` | `23.891 ms` | `14.965 ms` | full `1.243x`, prop `1.137x` |
| `cpu-memory-slot` | `349.283 ms` | `29.357 ms` | `14.962 ms` | full `1.225x`, prop `1.138x` |
| official Python BF16/TF32 | `428.009 ms` full cache step | `9.953 ms` add mask | `17.020 ms` cached propagation | baseline |

Evidence:
`outputs/sam31-default-backend-on-matrix-bf16-320x240-20260628a/summary.json`.

Remaining root targets after this pass:

1. Run the broader SAM3/SAM3.1 model matrix with the new default backend memory
   slot path and the new `cpu-memory-slot` A/B row to quantify the memory/time
   tradeoff beyond the short synthetic multi-slot contract.
2. Move the remaining `sam31_prepare_interactive_mask_prompt` work off the CPU
   path or fuse it with the mask-downsample graph; after caching/parallelism it
   is still about `1.2 ms` per mask-init.
3. Replace the duplicated CPU mask interpolation chain in mask-init memory
   preparation with a fused parity-preserving path.

### 2026-06-28 SAM3 F16 e2e Stage Split and Model PE Cache

The SAM3 benchmark now emits machine-readable per-stage rows with
`SAM3_BENCH_RESULT_JSON`, and the matrix runner records C++ `frame0_encode`,
`frame0_prompt`, tail `track_encode`, tail `track_propagate`, optional
preencode, and full model-e2e session time separately from official Python
`start_session`, `add_prompt`, and `propagate_wall` timings. It also accepts
`--cpp-variant` and repeated `--cpp-env KEY=VALUE` so opt-in CUDA paths can be
measured without losing the exact environment in `summary.json`.

On the 4-frame SAM3 F16/FP16/TF32 contract, the refreshed stage split showed
that tail tracking is still encode dominated: C++ tail encode is about
`147.117 ms/frame`, propagation is about `38.589 ms/frame`, and encode accounts
for about `79.2%` of the split tail track time. Official Python measured
`844.193 ms` model-e2e while C++ measured `1008.984 ms`, so C++ is still about
`1.195x` slower on the same prompt, frame range, input resolution, and warmup
policy. Evidence:
`outputs/sam3-e2e-stage-model-pe-cache-python-r3-20260628a/summary.json`.

The first accepted e2e optimization from this split moves the regular SAM3 neck
sinusoidal PE tensors from per-state lazy cache to a model-level backend cache.
Those tensors depend only on the model/backend and SAM3 encode size, not on the
video frame, prompt, or tracker state. The old state-owned path can still be
forced with `SAM3_DISABLE_MODEL_NECK_PE_CACHE=1`.

With the model PE cache warmed by the same two-session policy used for the
Python comparison, C++ frame-0 encode dropped from `295.711 ms` to
`158.986 ms` in the same binary A/B, and model-e2e dropped from `1127.798 ms`
to `1001.668 ms`. Tail encode did not materially change (`146.673 ms/frame`
old path versus `146.531 ms/frame` model cache), which confirms this is an e2e
setup-cache improvement rather than a steady image-encoder GEMM improvement.
The profile shows the PE build cost only on the first warmup encode
(`140.630 ms`) and about `0.002-0.003 ms` on later encodes. Evidence:
`outputs/sam3-e2e-stage-model-pe-cache-r3-20260628a/summary.json`,
`outputs/sam3-e2e-stage-state-pe-cache-r3-20260628a/summary.json`, and
`outputs/sam3-e2e-stage-model-pe-cache-profile-20260628a/summary.json`.

Full-mask C++ parity between model PE cache and the old state PE path was exact
on the same 4-frame contract: `mask_hash_equal_rows=4/4`, `min_bbox_iou=1.0`,
and zero bbox, score, and mask-area deltas. Evidence:
`outputs/sam3-model-pe-cache-parity-20260628a/compare.json`.

Rejected same-contract probes in this pass:

| Probe | Result |
|---|---|
| `GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_F32=1` plus unsafe run gate | Slower: model-e2e `1170.148 ms`, track `194.955 ms/frame`. |
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` | Tail encode was only noise-level different and model-e2e regressed to `1122.774 ms`. |
| `SAM3_F16_VIT_LINEAR_OUTPUT=1` | Slower: model-e2e `1209.883 ms`, track `202.772 ms/frame`. |
| `SAM3_F16_VIT_LINEAR_OUTPUT=1` plus `GGML_CUDA_CUBLASLT_DIRECT_F16_DST=1` | Still slower: model-e2e `1188.106 ms`, track `198.608 ms/frame`. |

Remaining SAM3 F16 root target:

1. The steady tail encode graph still needs a real GEMM/dataflow improvement.
   The current synchronized node profile ranks SAM3 ViT `MUL_MAT` groups first:
   `fc2/attn_proj`, `fc1`, then `qkv`, followed by head64 FlashAttention. The
   PE cache improves request e2e setup cost but does not close the steady
   encoder gap to official Python. The post-cache profile evidence is in
   `outputs/sam3-e2e-stage-model-pe-cache-node-profile-20260628a/hotspots-sam3-encode.json`.

### 2026-06-28 SAM3 F16 Artifact-Free e2e Split and Attn Projection Input

The e2e comparison now separates optional benchmark artifacts from model work.
`sam3_benchmark` emits `model_e2e_ms`, `artifact_ms`,
`frame0_artifact_ms`, and tail `track_artifact_*` fields in addition to the
artifact-inclusive `session_ms`. It also accepts `--no-output-artifacts` so
perf runs can skip JSONL/mask/logits generation entirely. The matrix runner
passes this through as `--cpp-no-output-artifacts` and uses artifact-excluded
`model_e2e_ms` for Python/C++ e2e comparisons while preserving
`cpp_session_with_artifacts_ms` for diagnostics.

This clarified the current SAM3 F16 baseline: with artifact generation disabled
and the same 4-frame FP16/TF32/1008px contract, C++ is still behind official
Python. Before the new attn-projection default, 10 C++ repeats measured
`1004.925 ms` model-e2e versus Python `844.193 ms`; tail tracking was
`185.624 ms/frame`, with `147.294 ms/frame` in encode and
`38.329 ms/frame` in propagation. Evidence:
`outputs/sam3-f16-default-no-artifacts-r10-20260628a/summary.json`.

The accepted dataflow optimization is to cast F16 SAM3 ViT attention output to
F16 before the projection GEMM by default. This removes the cuBLASLt-side
`src1=f32` conversion in `vit_attn_proj`; the fallback can be forced with
`SAM3_DISABLE_FAST_F16_VIT_ATTN_PROJ_INPUT=1` or
`SAM3_FAST_F16_VIT_ATTN_PROJ_INPUT=0`.

The 10-repeat artifact-free A/B showed a small but consistent encode-side win:

| Row | Model e2e mean | Frame0 encode | Tail track | Tail encode | Tail propagate |
|---|---:|---:|---:|---:|---:|
| Previous default | `1004.925 ms` | `160.457 ms` | `185.624 ms/fr` | `147.294 ms/fr` | `38.329 ms/fr` |
| F16 attn-proj input | `998.213 ms` | `158.580 ms` | `183.940 ms/fr` | `145.870 ms/fr` | `38.069 ms/fr` |

The model-e2e delta is `6.712 ms` (`0.668%`) and is still partially hidden by
session noise, but the targeted encode stages moved in the expected direction:
frame0 encode improved by `1.877 ms` and tail encode by `1.424 ms/frame`.
Welch t values from the 10-repeat samples were `5.25` for frame0 encode and
`6.14` for tail encode; propagate did not materially move (`0.26 ms/frame`).
Evidence:
`outputs/sam3-f16-default-no-artifacts-r10-20260628a/summary.json` and
`outputs/sam3-f16-attn-proj-input-no-artifacts-r10-20260628a/summary.json`.

Full-mask parity was exact against the old path on the same 10-frame clip:
`mask_hash_equal_rows=10/10`, `min_bbox_iou=1.0`, and zero bbox, score, and
mask-area deltas. The post-default 4-frame disable-control comparison was also
exact (`mask_hash_equal_rows=4/4`). Evidence:
`outputs/sam3-f16-attn-proj-input-parity-10f-20260628a/compare.json` and
`outputs/sam3-f16-default-enabled-parity-20260628a/compare.json`.

cuBLASLt timing confirms the mechanism. In the pre-default timing row,
`vit_attn_proj` had `src1_types=["f32"]` and `convert_sum_ms=30.720928` over
384 calls. With the new default, the same group has `src1_types=["f16"]` and
`convert_sum_ms=0.0` over 128 calls in the profile run. Evidence:
`outputs/sam3-f16-cublaslt-timing-20260628a/group-summary.json` and
`outputs/sam3-f16-default-enabled-cublaslt-timing-20260628a/group-summary.json`.

Remaining root targets after this pass:

1. The C++/Python gap is still not closed: the 10-repeat optimized C++ row is
   `998.213 ms` versus Python `844.193 ms`, so C++ remains about `1.18x`
   slower on this e2e contract.
2. Tail tracking remains encode dominated at about `79.3%` encode and
   `20.7%` propagation. The next CUDA work should target the large ViT GEMM
   groups (`fc1`, `fc2`, `qkv`) and the remaining FlashAttention path rather
   than cached-tail shortcuts.
3. Artifact-producing parity runs and artifact-free perf runs must stay
   separate. Artifact timing is now visible, but speed claims should use
   `--cpp-no-output-artifacts` unless the claim is explicitly about output
   serialization.

### 2026-06-28 SAM3 Required-e2e Input Split

The SAM3 e2e runner now separates the input-staging work needed before a model
session from the model session itself. `sam3_benchmark` emits
`input_load_ms`, `required_e2e_ms`, `input_plus_session_ms`,
`session_overhead_ms`, and `model_remainder_ms` in addition to the existing
artifact-free `model_e2e_ms`. The matrix comparison uses
`cpp_required_e2e_ms = cpp_input_load_ms + cpp_model_e2e_ms` for
`comparable_session_e2e_claim`, and requires `cpp_has_input_load_ms=true` so
old JSON rows without this measurement cannot be used for session-e2e speed
claims.

On the same 4-frame SAM3 F16/FP16/TF32/1008px contract, the refreshed
artifact-free measurement was:

| Stage | C++ CUDA | official Python |
|---|---:|---:|
| Input/session staging | `28.008 ms` input load | `58.012 ms` start_session |
| Model e2e | `982.604 ms` | `844.193 ms` |
| Required/session e2e | `1010.612 ms` | `902.230 ms` |
| Tail encode | `144.375 ms/fr` | included in Python propagate profile |
| Tail propagate | `37.462 ms/fr` | `36.952 ms/fr` timed-tail tracker propagation |
| Unclassified model remainder | `8.167 ms` | n/a |

The corrected session-e2e ratio is `0.893x` Python/C++, so C++ is still about
`1.12x` slower even though its input load is faster than Python
`start_session`. This removes input loading, optional output artifacts, and
unclassified session overhead as plausible explanations for the gap. The
dominant remaining target is still the SAM3 ViT encoder path: frame-0
encode/prompt plus tail encode account for the useful gap, while tail
propagation is already close to Python. Evidence:
`outputs/sam3-f16-required-e2e-r2-20260628a/summary.json`.

Rejected same-contract CUDA probes since the artifact-free split:

| Probe | Result |
|---|---|
| cuDNN SDPA head64 for global and window attention | Slower: model-e2e `1021.925 ms`; Q/K/V conversion and output transpose cost outweighed the cuDNN kernel. |
| cuDNN SDPA head64 global-only | Not enough: model-e2e `996.080 ms`, still slower than current default noise band. |
| Materialized F16 QKV output | Slower: model-e2e `1003.290 ms`, tail encode `148.867 ms/fr`. |
| Current ViT batch graph | No per-frame win: batch sizes 2 and 4 were slower per frame despite exact lane parity. |
| Neck 1x1 `mul_mat` path | Slightly slower: model-e2e `992.760 ms`. |

Next root optimization should therefore avoid conversion-wrapper approaches and
avoid moving work out of the measured session. The useful directions are:

1. Fused or producer-side low-precision QKV/FATTN dataflow that avoids
   materialized casts and output transposes.
2. Shape-specific cuBLASLt or custom CUDA paths for the large SAM3 ViT GEMM
   groups (`fc1`, `fc2`, `qkv`) without adding graph materialization.
3. A Python/C++ report that always shows both `model_e2e` and
   `required/session_e2e`, because Python `start_session` and C++ input loading
   are not equivalent enough to leave implicit.

### 2026-06-28 SAM3 Required-e2e Substage Split

The C++ benchmark now reports a finer required-e2e split:
`state_create_ms`, `tracker_create_ms`, `session_setup_ms`,
`frame0_prompt_model_ms`, `track_all_model_ms`, `track_encode_total_ms`,
`track_propagate_total_ms`, `track_unattributed_total_ms`,
`preencode_state_create_total_ms`, and `preencode_total_ms`. These fields make
it possible to separate required input loading, fixed session setup, frame-0
model work, tail encode, tail propagation, optional artifact work, and any
remaining unclassified wrapper time.

The same 4-frame SAM3 F16/FP16/TF32/1008px full-mask contract was rerun three
times with artifacts disabled:

| C++ row | Required e2e | Model e2e | Input load | Session setup | Frame0 encode | Frame0 prompt | Tail encode total | Tail propagate total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-mask normal | `1022.778 ms` | `995.323 ms` | `27.455 ms` | `8.123 ms` | `156.615 ms` | `276.358 ms` | `435.771 ms` | `118.455 ms` |
| Full-mask cached-tail | `1040.272 ms` | `1011.776 ms` | `28.496 ms` | `8.257 ms` | `156.565 ms` | `277.379 ms` | `149.734 ms` | `129.698 ms` |
| BBox-only normal | `1014.486 ms` | `986.650 ms` | `27.836 ms` | `8.596 ms` | `156.926 ms` | `275.365 ms` | `433.908 ms` | `111.852 ms` |

Evidence:
`outputs/sam3-f16-required-e2e-split-detail-1008-r3-20260628a/split-summary.json`,
`outputs/sam3-f16-required-e2e-cached-tail-detail-1008-r3-20260628a/split-summary.json`,
and
`outputs/sam3-f16-required-e2e-bbox-only-detail-1008-r3-20260628a/split-summary.json`.

The cached-tail row demonstrates why track/fr alone is not an e2e speed claim.
It moves frames 2..N-1 image encoding into `preencode_total_ms`, reducing the
reported track average from `184.743 ms/fr` to `93.145 ms/fr`, but it does not
remove the work. Once `preencode_total_ms=290.139 ms` is included in
`model_e2e_ms`, required e2e is slower than the normal row. Cached-tail remains
a diagnostic for matching stage boundaries, not an accepted optimization.

The bbox-only row also does not explain the Python gap. Skipping full-resolution
mask materialization improves required e2e by only about `8.3 ms` on this
contract, while the full-mask normal row remains about `120.5 ms` slower than
the previously measured official Python session-e2e row of `902.230 ms`
(`outputs/sam3-f16-required-e2e-r2-20260628a/summary.json`).

The remaining measurable gap is therefore not input loading, session setup,
optional artifact generation, cached-tail accounting, or full-mask output. The
dominant required work is still frame-0 encode/prompt plus tail ViT encode.
Future CUDA changes should be judged against `required_e2e_ms` and the explicit
tail encode/propagate totals, not against a track/fr value that excludes
preencoded work.

### 2026-06-28 SAM3 e2e Stage Attribution and Selected-Only Probe

The required-e2e split was tightened again so the C++ row accounts for every
model-session component explicitly. `sam3_benchmark` now emits
`model_accounted_ms`, `frame0_segment_ms`, `frame0_tracker_add_ms`,
`frame0_prompt_unattributed_ms`, `frame0_candidates`,
`frame0_added_instances`, and `text_init_selected_only`. The model remainder is
now computed after subtracting setup, frame-0 encode, frame-0 prompt model work,
tail model work, and any preencoded work. This makes a run invalid as an e2e
speed claim if it only moves encode work outside the timed tracking loop.

The official Python runner now reports matching stage hooks where available:
`start_session`, `add_prompt`, `propagate_wall`, timed-tail stage totals, and
per-frame backbone/detection timing. On the same SAM3 F16/FP16/TF32/1008px
4-frame contract, the Python tail has heavy backbone/detection work on frames 1
and 2 (`165.456 ms` and `165.014 ms`) and a light frame 3
(`0.233 ms`). This is not skipped work in the speed claim; it is official
SAM3's next-chunk prefetch/cache behavior inside propagation. The session row
still includes the wall time that paid for that cached work.

The current full all-detections row remains slower than official Python:

| Stage | C++ CUDA | official Python |
|---|---:|---:|
| Required/session e2e | `1021.900 ms` | `895.990 ms` |
| Model e2e | `994.204 ms` | `838.620 ms` |
| Frame-0 text detection | `101.224 ms` | included in `add_prompt` |
| Frame-0 tracker add/memory encode | `174.695 ms` for 3 instances | included in `add_prompt` |
| Tail encode total | `434.413 ms` | `330.704 ms` timed-tail backbone/detection |
| Tail propagate total | `119.007 ms` | `106.908 ms` timed-tail tracker propagation |

Evidence:
`outputs/sam3-f16-e2e-stage-split-1008-r3-20260628d/summary.json`.

This row is a valid same-cardinality speed comparison
(`same_text_init_cardinality=true`) and the result is still a loss:
official Python is about `0.877x` C++ required-e2e time, so C++ must improve by
about `14%` to match this all-detections contract. The largest measured C++
components are tail encode, frame-0 tracker add/memory encode for three
detections, and frame-0 image encode/text detection.

An application-contract probe was added for the case where the caller only
needs to track the selected text-detection result. `--text-init-selected-only`
still runs text detection over all initial candidates, but it only adds the
selected detection to the tracker/memory bank. This reduces the frame-0
tracker-add work from `174.695 ms` for three instances to `61.370 ms` for one
instance and reduces tail propagation from `119.007 ms` to `43.989 ms`.

| Row | Required e2e | Model e2e | Frame-0 added instances | Official comparison status |
|---|---:|---:|---:|---|
| Full all-detections | `1021.900 ms` | `994.204 ms` | `3` | comparable, slower than Python |
| Selected-only | `851.924 ms` | `824.226 ms` | `1` | not comparable to all-detections Python |
| official Python | `895.990 ms` | `838.620 ms` | all detections | reference |

Evidence:
`outputs/sam3-f16-e2e-selected-only-1008-r3-20260628a/summary.json`.

The selected-only row numerically beats the official Python all-detections row,
but it is not an official Python parity claim because
`same_text_init_cardinality=false`. It is an application-level optimization only
when the product contract is "track the selected person/object" rather than
"track every initial text-detection candidate".

Selected-only output was compared against the normal C++ selected target on the
same four frames. Exact parity is not met (`diff_rows=3`) because the tracking
mask hashes differ after frame 0, but the displayed target bbox is identical and
the mask drift is very small:
`min_bbox_iou=1.0`, `max_bbox_delta_px=0`, `max_score_abs_delta=0.000734`, and
`max_abs_mask_area_rel_delta=0.000653`. With the app-display tolerance used for
this probe, `tolerance_diff_rows=0`. Evidence:
`outputs/sam3-f16-selected-only-parity-1008-20260628a/selected_vs_normal.json`.

The next root optimization for the official all-detections claim should not be
cached-tail accounting or output-artifact removal. The measured targets are:

1. Reduce SAM3 ViT tail encode, especially the large `fc1`, `fc2`, `qkv`, and
   head64 attention groups.
2. Reduce the per-instance frame-0 tracker-add/memory-encode cost without
   changing the all-detections contract.
3. Keep every speed table reporting both required/session e2e and model e2e, and
   mark selected-only rows as non-comparable unless official Python is run with
   the same one-selected-instance contract.

### 2026-06-28 SAM3 Mask-Input Object-Pointer Transfer Fix

The first accepted root fix from the new split is in
`sam3_compute_mask_input_obj_ptr`. The old path staged tracker FPN features
through CPU for each initial text-detection candidate before running the
mask-input SAM decoder used to create an object pointer. At 1008px this meant
re-reading and re-uploading the same tracker features per candidate, including
the `84.9 MB` high-resolution FPN tensor. The function now uses the existing
backend-copy helper for F32 tensors, so the same features stay on the backend
and are copied device-to-device.

This is not a cached-tail or selected-only shortcut. It preserves the
all-detections contract and only changes the transfer path for required
frame-0 object-pointer construction.

The benchmark JSON now also exposes the required frame-0 add split:
`frame0_tracker_mask_prepare_ms`, `frame0_tracker_memory_encode_ms`,
`frame0_tracker_obj_ptr_ms`, `frame0_tracker_store_ms`, and
`frame0_tracker_unattributed_ms`. A representative post-fix single run shows the
three-candidate add cost as:

| Add substage | Time |
|---|---:|
| Mask prepare | `8.567 ms` |
| Memory encode | `17.402 ms` |
| Object pointer | `24.066 ms` |
| Store | `0.046 ms` |
| Unattributed | `0.009 ms` |

Evidence:
`outputs/sam3-f16-add-split-gpucopy-1008-smoke-20260628a/normal.stdout.log`.

The same 4-frame SAM3 F16/FP16/TF32/1008px all-detections contract was rerun
five times after the transfer fix:

| Metric | Before split row | After GPU-copy fix |
|---|---:|---:|
| Required e2e | `1021.900 ms` | `924.404 ms` mean, `4.658 ms` sd |
| Model e2e | `994.204 ms` | `896.720 ms` mean, `4.890 ms` sd |
| Frame-0 tracker add | `174.695 ms` | `49.661 ms` mean, `1.799 ms` sd |
| Tail encode total | `434.413 ms` | `452.896 ms` mean, `2.409 ms` sd |
| Tail propagate total | `119.007 ms` | `126.465 ms` mean, `4.307 ms` sd |

The frame-0 add stage improved by about `3.5x`; required e2e improved by about
`97.5 ms` (`9.5%`) against the previous r3 all-detections row. Full-mask parity
against the pre-fix normal row is exact: `diff_rows=0`, `mask_hash_equal_rows=4`,
zero bbox, score, and mask-area deltas. Evidence:
`outputs/sam3-f16-mask-input-objptr-gpucopy-1008-r5-20260628a/summary.json` and
`outputs/sam3-f16-mask-input-objptr-gpucopy-parity-1008-20260628a/before_vs_after.json`.

This brings C++ much closer to official Python but does not finish the
all-detections goal. Against the same official Python row
(`895.990 ms` required/session e2e, `838.620 ms` model e2e), post-fix C++ is
still about `28.4 ms` slower on required e2e and about `58.1 ms` slower on model
e2e. The remaining required work is now dominated by tail encode plus the
remaining mask-input object-pointer and memory-encode substeps, not by input
loading, output artifacts, CPU staging of tracker features, or selected-only
cardinality.

### 2026-06-28 SAM3 Required-E2E Accounting Fields and Object-Pointer Cache

The benchmark accounting was tightened again so required e2e can be audited by
addition rather than interpretation. `sam3_benchmark` now emits
`model_load_ms` as an explicit alias for the existing model-load `load_ms`,
`cold_required_e2e_ms`, `required_accounted_ms`, `required_remainder_ms`,
`frame0_model_ms`, `tail_model_ms`, and `tail_encode_required_total_ms`.
The hot comparable claim remains `required_e2e_ms = input_load_ms +
model_e2e_ms`; `cold_required_e2e_ms` is diagnostic and includes model load.

This makes cached-tail runs self-policing. A cached-tail run can still reduce
`track_ms`, but `tail_encode_required_total_ms` adds the normal loop encode and
the cached/preencoded work back together, so the row cannot claim an e2e win by
moving required encode work outside the visible tracking loop.

The SAM3 mask-input object-pointer path was also optimized beyond the previous
GPU-copy fix. `sam3_compute_mask_input_obj_ptr` now reuses a graph/cache for the
fixed SAM3 mask-input decoder shape and builds only the tensors needed for
object-pointer construction (`sam_token` and `obj_score`). This keeps the
all-detections contract intact; it does not use selected-only cardinality or
cached-tail accounting.

Parity against the previous cached object-pointer artifact is exact:
`diff_rows=0`, `tolerance_diff_rows=0`, `mask_hash_equal_rows=4`, and all bbox,
score, and mask-area deltas are zero. Evidence:
`outputs/sam3-f16-e2e-fields-parity-1008-20260628a/cached_before_vs_e2e_fields.json`.

The warmed, artifact-free 5-run SAM3 F16/FP16/TF32/1008px all-detections row is:

| Metric | Mean | Median | sd |
|---|---:|---:|---:|
| Required e2e | `913.859 ms` | `914.396 ms` | `2.388 ms` |
| Model e2e | `886.373 ms` | `885.909 ms` | `2.438 ms` |
| Required accounted | `913.858 ms` | `914.394 ms` | `2.388 ms` |
| Required remainder | `0.001 ms` | `0.001 ms` | `0.001 ms` |
| Frame-0 model | `295.682 ms` | `296.232 ms` | `1.327 ms` |
| Tail model | `582.577 ms` | `583.606 ms` | `2.595 ms` |
| Tail encode required total | `459.027 ms` | `459.114 ms` | `2.980 ms` |
| Tail propagate total | `123.549 ms` | `123.696 ms` | `0.926 ms` |
| Frame-0 tracker add | `41.723 ms` | `41.096 ms` | `1.156 ms` |
| Frame-0 memory encode | `17.980 ms` | `17.394 ms` | `1.200 ms` |
| Frame-0 object pointer | `15.251 ms` | `15.271 ms` | `0.111 ms` |

Evidence:
`outputs/sam3-f16-e2e-fields-1008-r5-20260628a/summary.json`.

Against the same official Python row (`895.990 ms` session e2e and
`838.620 ms` model e2e), the warmed C++ row is still slower by `17.869 ms` on
required e2e and `47.753 ms` on model e2e. That is a remaining `1.99%`
required-e2e gap and `5.69%` model-e2e gap. The current useful target is
therefore no longer benchmark accounting or output artifact handling; it is
still the model work, especially tail encode and the remaining frame-0
memory/object-pointer construction.

The cached-tail diagnostic confirms the new accounting. With
`--preencode-cached-tail-frames`, `track_ms` drops to `94.433 ms/fr`, but
`preencode_total_ms=306.008 ms` is reported separately and
`tail_encode_required_total_ms=456.736 ms`, effectively the same required tail
encode work as the normal row. Required e2e is slower (`927.532 ms` mean), so
cached-tail remains a boundary diagnostic, not an accepted optimization.
Evidence:
`outputs/sam3-f16-e2e-fields-cached-tail-1008-r3-20260628a/summary.json`.

### 2026-06-28 Required-E2E Frame Timing Rows

`sam3_benchmark` now has `--output-frame-timing-jsonl <path>`. The file is
written after the measured session, so it does not enter `model_e2e_ms` or
`required_e2e_ms`. It emits one `frame0` row and one `tail` row per propagated
frame. The tail rows separate inline encode, preencoded state creation,
preencode encode, required encode total, propagate, artifact, and unattributed
time.

`scripts/model_matrix_compare.py` also has `--cpp-frame-timing-dir`. Relative
paths are created under `--out-dir`, and the summary records the generated
JSONL paths in `benchmark_context.cpp_frame_timing_jsonl` plus each C++ row's
`cpp_frame_timing_jsonl_runs`. This makes same-contract Python comparisons
auditable at the frame level without changing the measured C++ session time.

The SAM3 F16/1008px 4-frame smoke validates the accounting closure:

| Metric | Benchmark JSON | Frame timing sum |
|---|---:|---:|
| Required tail encode | `446.958 ms` | `446.958 ms` |
| Required tail propagate | `121.255 ms` | `121.255 ms` |
| Required tail unattributed | `0.001 ms` | `0.001 ms` |

The per-frame tail rows were:

| Frame | Required encode | Propagate | Detections |
|---:|---:|---:|---:|
| 1 | `156.101 ms` | `41.851 ms` | `3` |
| 2 | `146.291 ms` | `40.538 ms` | `3` |
| 3 | `144.566 ms` | `38.866 ms` | `3` |

Evidence:
`outputs/sam3-f16-required-e2e-frame-timing-smoke-20260628a/`.

The comparison-script smoke also writes and references the timing file
correctly. Its summary points to
`cpp-frame-timing/cpp-frame-timing-01.jsonl`, and the tail-row sums match the
aggregate row: required tail encode `469.108 ms` and tail propagate
`80.572 ms`. Evidence:
`outputs/model-matrix-frame-timing-smoke-20260628a/summary.json`.

With the frame split in place, a fresh `SAM3_PROFILE=1` run shows that tail
encode is not limited by graph construction or upload. On steady tail frames,
`sam3_encode_graph_build`, `sam3_encode_graph_alloc`, and
`sam3_encode_input_upload` are each below `1 ms`; `sam3_encode` compute is about
`143-145 ms` per tail frame. Evidence:
`outputs/sam3-f16-required-e2e-encode-profile-20260628a/`.

The synchronized CUDA node profile confirms the remaining root bottleneck. On
the steady `sam3_encode#02` call, drop-max stage totals were:

| Stage | Drop-max sum |
|---|---:|
| ViT projection or MLP FC2 matmul | `35.390 ms` |
| ViT MLP FC1 matmul | `30.022 ms` |
| ViT QKV matmul | `19.609 ms` |
| ViT window head64 attention | `12.338 ms` |
| Other | `11.185 ms` |
| ViT MLP GELU | `7.141 ms` |
| ViT global head64 attention | `5.793 ms` |

Evidence:
`outputs/sam3-f16-encode-node-profile-20260628a/encode-node-names.json`.

Two direct optimization probes were rechecked and rejected as defaults:

| Candidate | Result | Decision |
|---|---:|---|
| cuDNN MLP FC1+GELU F32 opt-in | required e2e `970.346 ms` mean vs default `920.718 ms`; tail encode `478.802 ms` vs `452.363 ms` | reject |
| cuBLASLt FC2 residual epilogue opt-in | required e2e `921.729 ms` mean vs default `922.392 ms`, within noise | keep diagnostic |

Evidence:
`outputs/sam3-f16-cudnn-mlp-fc1-gelu-ab-20260628a/` and
`outputs/sam3-f16-cublaslt-residual-ab-20260628a/`.

The next root optimization target remains the SAM3 ViT projection path or a
larger parity-safe ViT block dataflow change. Cached-tail, graph caching,
output-artifact accounting, cuDNN MLP FC1+GELU, and small cuBLASLt residual
epilogues do not close the same-contract Python gap.

### 2026-06-28 Python Phase Split for Required E2E

The official SAM3 Python baseline runner was split out of
`scripts/model_matrix_compare.py` into `scripts/run_sam3_python_baseline.py`.
This removes the fragile embedded `python -c` body and lets the runner expose
phase-scoped stage timing in the comparison summary:

- `start_session`: frame loading plus initial input-batch construction.
- `add_prompt`: frame-0 prompt/detector/tracker initialization.
- `propagate`: tail-frame detector/backbone and tracker propagation.

The runner now wraps both SAM3.1's `load_resource_as_video_frames` path and
SAM3's `load_video_frames` path, so the Python required/session E2E path has a
visible input-load split comparable to C++ `input_load_ms`.

Fresh 4-frame SAM3 F16/BF16, 1008px, `person` text-prompt comparison with
`repeats=3`, C++ output artifacts disabled, and Python/C++ warmups both set to
2:

| Metric | C++ CUDA F16 | Official Python BF16 | Ratio / Delta |
|---|---:|---:|---:|
| Model E2E | `841.894 ms` | `791.790 ms` | Python/C++ `0.9405` |
| Required/session E2E | `863.666 ms` | `856.371 ms` | Python/C++ `0.9916` |
| C++ input load | `21.772 ms` | n/a | included in required E2E |
| Python start_session | n/a | `64.558 ms` | includes load/normalize/staging |
| Python start load_video_frames | n/a | `65.334 ms` mean | profile-run stage |
| C++ tail encode | `470.646 ms` | n/a | required tail total |
| Python timed-tail backbone_detection | n/a | `335.811 ms` mean | profile-run stage |
| C++ tail propagate | `83.027 ms` | n/a | required tail total |
| Python timed-tail tracker_propagation | n/a | `71.699 ms` mean | profile-run stage |

Evidence:
`outputs/e2e-required-split-sam3-f16-r3-20260628b/summary.json`.
The C++ frame timing files for the same run are under
`outputs/e2e-required-split-sam3-f16-r3-20260628b/cpp-frame-timing/`.

Interpretation: required/session E2E is now within about `7.3 ms` on this short
clip, but model-only E2E is still about `50.1 ms` slower in C++. The remaining
meaningful gap is not decode, artifact writing, or accounting. It is the SAM3
tail image-encode path: C++ spends about `470.6 ms` on three required tail
encodes while the official Python timed-tail detector/backbone work is about
`335.8 ms` in the same measured range. The next optimization should therefore
target CUDA ViT image encode throughput, especially projection/MLP/QKV matmul
dataflow, rather than moving work into cached-tail preencode buckets.

The same Python BF16 baseline was reused against the C++ BF16 GGML model. C++
BF16 improves the required tail encode and is the first row in this harness that
wins required/session E2E, although it still does not beat Python on model-only
E2E:

| Metric | C++ CUDA BF16 | Official Python BF16 | Ratio / Delta |
|---|---:|---:|---:|
| Model E2E | `821.030 ms` | `791.790 ms` | Python/C++ `0.9644` |
| Required/session E2E | `842.993 ms` | `856.371 ms` | Python/C++ `1.0159` |
| C++ tail encode | `458.198 ms` | n/a | `12.448 ms` faster than C++ F16 |
| C++ tail propagate | `79.005 ms` | n/a | `4.022 ms` faster than C++ F16 |

Evidence:
`outputs/e2e-required-split-sam3-bf16-r3-20260628a/summary.json`.

This should not be reported as full goal completion until quality parity is
checked for the BF16 row, because the current table is a performance-contract
measurement only. It does show that BF16 is the better CUDA performance baseline
for this model on this GPU.

The same BF16 contract was rerun after the comparison runner started emitting
explicit nested required-e2e breakdowns:
`cpp_required_e2e_breakdown_ms` and `python_required_e2e_breakdown_ms`. The
summary now records the C++ required components, C++ non-required diagnostics,
Python `start_session` / `add_prompt` / `propagate` wall times, and the Python
phase-stage means in a machine-readable shape.

Fresh BF16 numbers with the new fields were:

| Metric | C++ CUDA BF16 | Official Python BF16 |
|---|---:|---:|
| Required/session E2E | `843.199 ms` | `848.779 ms` |
| Model E2E | `821.630 ms` | `790.391 ms` |
| C++ tail encode | `453.739 ms` | n/a |
| C++ tail propagate | `81.315 ms` | n/a |
| Python timed-tail backbone/detection | n/a | `333.697 ms` |
| Python timed-tail tracker propagation | n/a | `71.960 ms` |

Evidence:
`outputs/e2e-required-breakdown-sam3-bf16-r3-20260628a/summary.json`.

This confirms the current shape of the gap: C++ wins required/session E2E on
this short clip only because Python pays more visible `start_session` work.
C++ still loses model-only E2E by about `31.2 ms`, and the dominant model gap is
the SAM3 tail image encode path.

The BF16 row was also checked against official Python BF16 tracking artifacts
with pixel-mask IoU, not just mask area/hash. It is not exact parity:
`diff_rows=4` and `mask_hash_equal_rows=0`. Under the current relaxed tracking
tolerance it passes (`tolerance_diff_rows=0`) with
`min_bbox_iou=0.978805`, `max_bbox_delta_px=2.552`,
`max_score_abs_delta=0.016051`, `max_abs_mask_area_rel_delta=0.012810`,
`min_mask_pixel_iou=0.986339`, and `max_mask_pixel_xor=447`.
Evidence:
`outputs/sam3-bf16-quality-parity-20260628a/python-vs-cpp-summary.json`.

Several existing CUDA/SAM3 BF16 ViT knobs were rechecked against the same
artifact-free required-e2e contract. None beat the default:

| Candidate | Required E2E | Model E2E | Tail encode | Decision |
|---|---:|---:|---:|---|
| Default BF16 | `843.199 ms` | `821.630 ms` | `453.739 ms` | keep |
| `SAM3_BF16_VIT_LINEAR_OUTPUT=1` | `850.224 ms` | `828.657 ms` | `460.931 ms` | reject |
| `SAM3_BF16_VIT_QKV_CHAIN=1` | `890.812 ms` | `869.110 ms` | `489.094 ms` | reject |
| `SAM3_BF16_VIT_MLP_CHAIN=1` | `912.266 ms` | `890.382 ms` | `502.886 ms` | reject |
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` | `848.060 ms` | `825.981 ms` | `457.427 ms` | reject |
| `SAM3_ENABLE_VIT_CONTIGUOUS_ATTENTION_V=1` | `860.543 ms` | `839.251 ms` | `466.598 ms` | reject |
| `SAM3_DISABLE_BF16_VIT_LINEAR_INPUTS=1` | `861.607 ms` | `840.103 ms` | `468.111 ms` | diagnostic reject |

Evidence:
`outputs/e2e-required-breakdown-sam3-bf16-linear-output-r3-20260628a/summary.json`,
`outputs/e2e-required-breakdown-sam3-bf16-qkv-chain-r3-20260628a/summary.json`,
`outputs/e2e-required-breakdown-sam3-bf16-mlp-chain-r3-20260628a/summary.json`,
`outputs/e2e-required-breakdown-sam3-bf16-flat-mlp-r3-20260628a/summary.json`,
`outputs/e2e-required-breakdown-sam3-bf16-contig-v-r3-20260628a/summary.json`,
and
`outputs/e2e-required-breakdown-sam3-bf16-disable-linear-inputs-r3-20260628a/summary.json`.

The diagnostic reject is useful: disabling BF16 ViT linear inputs makes tail
encode about `14.4 ms` slower, so the current default already depends on that
optimization. Further speedup needs a real ViT matmul/attention dataflow change,
not broader BF16 output chaining or cached-tail accounting.

The existing ViT batch bench was also checked as a possible root optimization
for the tail encode gap. It is not useful as-is on this GPU:

| Bench | Batch | Mean | Mean / frame | Lane diff |
|---|---:|---:|---:|---:|
| ViT only | 1 | `135.785 ms` | `135.785 ms` | n/a |
| ViT only | 2 | `313.460 ms` | `156.730 ms` | `0` |
| ViT only | 3 | `468.426 ms` | `156.142 ms` | `0` |
| ViT + tracker neck | 1 | `148.547 ms` | `148.547 ms` | n/a |
| ViT + tracker neck | 2 | `337.535 ms` | `168.768 ms` | `0` |
| ViT + tracker neck | 3 | `494.016 ms` | `164.672 ms` | `0` |

Batching repeated frames preserves lane parity, but per-frame latency gets worse
at batch sizes 2 and 3. The next optimization should not be a naive multi-frame
batch encode. It should instead reduce the single-frame ViT path directly
through CUDA matmul/attention dataflow or a parity-safe fused block path.

### 2026-06-28 Required E2E Frame-Timing Split

`sam3_benchmark --output-frame-timing-jsonl` now records the C++ frame-level
required-E2E split and the encode-internal timing from `sam3_last_encode_timing`.
The comparison runner aggregates those JSONL files into
`cpp_frame_timing_aggregate_ms`, so model/session E2E rows can separate frame
load, frame-0 prompt/add, tail encode, tail propagation, artifacts, and encode
substeps without using cached-tail shortcuts.

On the fresh SAM3 BF16 same-contract run, the tail encode substeps are:

| Tail encode component | Three-frame total |
|---|---:|
| Required tail encode | `456.002 ms` |
| `encode_graph_compute` | `453.340 ms` |
| `encode_graph_build` | `0.653 ms` |
| `encode_graph_alloc` | `1.042 ms` |
| `encode_input_upload` | `0.613 ms` |
| `encode_state_update` | `0.340 ms` |

Evidence:
`outputs/e2e-separated-sam3-bf16-r3-20260628b/summary.json`.

This confirms that the E2E-required tail work is not being hidden in decode,
upload, graph construction, or benchmark artifacts. The optimization target is
the CUDA graph compute itself, dominated by SAM3 ViT projection matmuls,
head64 attention, GELU, residual adds, and Q/K layout materialization.

The cuBLASLt timing probe for the same short run split the dominant projection
matmuls as:

| ViT GEMM group | Calls | Matmul sum |
|---|---:|---:|
| MLP FC2 | `128` | `121.890 ms` |
| MLP FC1 | `128` | `120.600 ms` |
| QKV | `128` | `79.201 ms` |
| Attention projection | `128` | `29.246 ms` |

Evidence:
`outputs/sam3-bf16-cublaslt-timing-20260628a/cublaslt-groups.json`.

Two additional shallow fusion probes were rejected with the same separated
measurement:

| Candidate | Required E2E | Model E2E | Tail encode | Decision |
|---|---:|---:|---:|---|
| Default BF16 | `845.661 ms` | `823.816 ms` | `456.002 ms` | keep |
| `GGML_CUDA_ENABLE_CUBLASLT_BIAS_GELU_ERF_FUSION=1` | `886.393 ms` | `864.830 ms` | `486.121 ms` | reject |
| `GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION=1` | `847.964 ms` | `826.315 ms` | `461.437 ms` | reject |

Evidence:
`outputs/e2e-separated-sam3-bf16-gelu-erf-r3-20260628a/summary.json` and
`outputs/e2e-separated-sam3-bf16-winpart-cpy-r3-20260628a/summary.json`.

The next root optimization should therefore avoid another env-only toggle. It
should either improve the ViT projection/GEMM path directly or remove a real
QKV/RoPE/FATTN/materialization boundary while preserving the same precision and
tracking-quality contract.

### 2026-06-28 Required E2E Process Split Refresh

`scripts/summarize_e2e_component_splits.py --process-markdown` now emits a
C++/Python process table from the comparison rows. It keeps formal wall-time
columns separate from Python diagnostic hook timings, so the table can be used
both for speed claims and for bottleneck attribution without mixing those two
contracts.

The fresh same-contract SAM3 BF16 r3 run is in
`outputs/e2e-required-process-split-sam3-bf16-r3-20260628a/`:

| Process | C++ | Python | Python/C++ | Read |
|---|---:|---:|---:|---|
| Required/session E2E | `846.238 ms` | `842.388 ms` | `0.995` | effectively tied in this short r3 |
| Model E2E | `825.034 ms` | `792.236 ms` | `0.960` | Python is still faster once start-session/input prep is excluded |
| Input prep/start-session | `21.204 ms` | `50.127 ms` | `2.364` | C++ wins host input prep |
| Frame-0 prompt/add | `122.415 ms` | `349.319 ms` | `2.854` | C++ wins prompt/detection wall split |
| Tail image encode/backbone | `454.398 ms` | `335.531 ms` | `0.738` | main C++ loss |
| Tail image encode graph compute | `451.321 ms` | `335.531 ms` | `0.743` | loss is kernel/dataflow, not upload/build |
| Tail propagate total | `84.726 ms` | `71.533 ms` | `0.844` | total still loses |
| Tail propagate graph compute | `60.832 ms` | `71.533 ms` | `1.176` | C++ graph core is faster; non-core work makes total slower |

The tail-propagate non-core split is small compared with the tail-encode gap:
`input_upload=14.103 ms`, `rope_setup=4.534 ms`, `graph_admin=1.515 ms`, and
`output_read=0.546 ms`. These are worth keeping visible, but they cannot close
the model-E2E gap by themselves.

The r5 run in
`outputs/e2e-required-core-split-sam3-bf16-python-sam3-warmup2-r5-20260628/`
shows the same shape with lower decision risk: tail image encode is
`458.979 ms` versus Python timed-tail backbone `337.126 ms`, while tail
propagate graph compute is `61.429 ms` and the total tail propagate split is
`84.800 ms`.

A synchronized node profile for the same input contract is in
`outputs/e2e-process-node-profile-sam3-bf16-20260628a/`. The steady
`sam3_encode#04` attribution is:

| SAM3 ViT stage | Profiled drop-max sum |
|---|---:|
| Projection or MLP FC2 matmul | `36.431 ms` |
| MLP FC1 matmul | `28.867 ms` |
| QKV matmul | `18.865 ms` |
| Window attention | `11.372 ms` |
| Other neck/non-ViT nodes | `11.357 ms` |
| MLP GELU | `7.091 ms` |
| Global attention | `5.394 ms` |
| Q/K layout + RoPE | `4.624 ms` |
| Residual/norm/copy | `3.703 ms` |

Decision: the next accepted optimization should reduce the single-frame SAM3
ViT tail-encode graph compute directly, especially projection/MLP/QKV matmul
dataflow. Naive multi-frame ViT batching and shallow env-toggle fusions have
already been rejected; decode, artifact output, graph construction, and cached
tail accounting are not the current root cause.

The matmul precision boundary was also checked with a C++-only r3 A/B. Forcing
strict `CUBLAS_COMPUTE_32F` with `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` is much
slower than the default TF32-fast path:

| C++ variant | Required E2E | Model E2E | Tail encode graph compute | Required core compute |
|---|---:|---:|---:|---:|
| Default TF32-fast | `850.165 ms` | `828.471 ms` | `452.393 ms` | `666.711 ms` |
| Forced strict 32F | `1046.740 ms` | `1025.221 ms` | `544.941 ms` | `844.365 ms` |

Evidence:
`outputs/e2e-required-process-split-sam3-bf16-cpp-tf32off-r3-20260628a/default-vs-tf32off-component-table.md`.
This confirms that the current default is already using the needed TF32-fast
matmul route for same-policy Python comparisons; the remaining win must come
from better per-op dataflow or kernel structure, not from enabling TF32.

The same process-split harness rechecked the largest MLP FC2 cuBLASLt heuristic
surface. The FC2 shape reports three usable heuristics, but changing the
ViT-specific `mlp.lin2` heuristic from the current default made E2E worse:

| C++ variant | Required E2E | Model E2E | Tail encode graph compute | Decision |
|---|---:|---:|---:|---|
| Default FC2 heuristic | `848.107 ms` | `827.228 ms` | `450.922 ms` | keep |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2=1` | `857.387 ms` | `836.126 ms` | `457.006 ms` | reject |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2=2` | `860.572 ms` | `839.754 ms` | `458.075 ms` | reject |

Evidence:
`outputs/e2e-required-process-split-sam3-bf16-fc2algo2-r3-20260628a/fc2-algo-component-table.md`.
This keeps the next projection work focused on a structural GEMM/dataflow
change, not another heuristic-index default.

The propagation alias probe was reopened with the required-E2E process split.
`sam3_propagate_timing` and the benchmark frame-timing JSONL now report actual
alias/upload counts, so this optimization is no longer judged from wall time
alone. For SAM3 CUDA, the default path now aliases the three encoded tracker
feature inputs and four constant PE/RoPE inputs in each required tail frame. The
old upload path remains available with
`SAM3_DISABLE_PROP_ALIAS_FEATURE_INPUTS=1` and
`SAM3_DISABLE_PROP_ALIAS_CONSTANT_INPUTS=1`.

Five-run C++-only A/B on the same SAM3 BF16/TF32, 4-frame, 1008px,
artifact-free cached-tail contract:

| C++ variant | Required E2E | Tail propagate | Tail prop graph compute | Tail prop input upload | Tail alias counts |
|---|---:|---:|---:|---:|---|
| Alias disabled | `848.649 +/- 6.258 ms` | `83.680 +/- 5.491 ms` | `61.442 +/- 0.373 ms` | `12.639 +/- 3.691 ms` | feature `0/9`, constant `0/12` |
| Default prop alias | `847.267 +/- 5.177 ms` | `80.136 +/- 3.915 ms` | `60.089 +/- 0.296 ms` | `11.287 +/- 2.792 ms` | feature `9/9`, constant `12/12` |

The conservative r5 E2E win is small (`-1.382 ms`) because tail image encode is
still the dominant cost and run-to-run variance is several milliseconds. The
tail-propagation split is consistently better (`-3.544 ms` total, with graph
compute and upload both about `-1.35 ms`), so the alias path is accepted as the
CUDA default but not counted as closing the Python model-E2E gap.

Parity was checked separately with output artifacts enabled against the disabled
path: `diff_rows=0`, `mask_hash_equal_rows=4/4`, `min_mask_pixel_iou=1.0`, and
`max_mask_pixel_xor=0`.

Evidence:
`outputs/e2e-required-process-split-sam3-bf16-propalias-disabled-r5-20260628a/summary.json`,
`outputs/e2e-required-process-split-sam3-bf16-propalias-default-r5-20260628a/disabled-vs-default-component-table.md`,
and
`outputs/e2e-prop-alias-default-vs-disabled-parity-20260628a/disabled-vs-default-parity.json`.

### 2026-06-28 Required E2E Component Split And cuBLASLt Plan Cache

The comparison summary now emits an explicit
`required_e2e_component_split_ms` object for each C++/Python comparison row.
This is the compact E2E view to use before claiming a speed win:

- C++: `input_load_ms`, `session_setup_ms`, `frame0_encode_ms`,
  `frame0_prompt_model_ms`, `tail_state_create_ms`,
  `tail_encode_required_ms`, `tail_propagate_ms`,
  `tail_unattributed_ms`, and remainder fields.
- Python: `start_session_ms`, `add_prompt_ms`, `propagate_wall_ms`, and
  diagnostic profile-stage fields for `load_video_frames`,
  `construct_initial_input_batch`, timed-tail `backbone_detection`, and
  timed-tail `tracker_propagation`.

This keeps the E2E contract explicit: model-only E2E excludes optional C++
artifact writing, required/session E2E includes C++ input load, and Python
`start_session` remains separated from `add_prompt + propagate`.

The first CUDA optimization accepted under this separated E2E contract is a
cuBLASLt bias-fusion plan cache in ggml CUDA. The math path is unchanged; the
cache only reuses the per-shape cuBLASLt matmul descriptor, matrix layouts, and
preference object for the existing bias-fused GEMM path. It is enabled by
default and can be disabled with
`GGML_CUDA_DISABLE_CUBLASLT_BIAS_PLAN_CACHE=1`.

10-run C++-only A/B on SAM3 BF16, 4 frames, 1008px, output artifacts disabled,
and two C++ warmups:

| Metric | Plan cache | Disabled | Delta |
|---|---:|---:|---:|
| Model E2E | `824.162 ms` | `829.037 ms` | `+4.875 ms` disabled |
| Required/session E2E | `845.874 ms` | `850.569 ms` | `+4.696 ms` disabled |
| Tail encode required total | `458.958 ms` | `462.426 ms` | `+3.469 ms` disabled |
| Tail encode graph compute | `456.365 ms` | `459.564 ms` | `+3.199 ms` disabled |
| Frame-0 encode | `154.832 ms` | `155.542 ms` | `+0.710 ms` disabled |
| Frame-0 prompt | `121.881 ms` | `123.186 ms` | `+1.306 ms` disabled |

Evidence:
`outputs/e2e-separated-sam3-bf16-plan-cache-cpponly-r10-20260628a/summary.json`
and
`outputs/e2e-separated-sam3-bf16-plan-cache-disabled-cpponly-r10-20260628a/summary.json`.

The same change was checked for output parity against the disabled path:
`diff_rows=0`, `mask_hash_equal_rows=4`, `min_mask_pixel_iou=1.0`, and
`max_mask_pixel_xor=0`.
Evidence:
`outputs/sam3-bf16-plan-cache-parity-20260628b/plan-vs-disabled.json`.

cuBLASLt timing logs confirm that the normal run hits the plan cache for the
dominant ViT groups after the first call (`plan_cache_hit_rate=0.992` for QKV,
MLP FC1, MLP FC2, and attention projection in the short profile). The
synchronized timing probe is still too invasive to use as the primary speed
claim, so the accepted evidence is the normal 10-run E2E A/B above.

This is a small but real root-side win. It does not close the official Python
model-E2E gap by itself. The remaining target is still the SAM3 ViT image encode
graph compute, especially the projection GEMMs, FATTN boundary, RoPE/QK
materialization, and adjacent data movement.

### 2026-06-28 Required E2E Propagation Split And CUDA A/B

The required-E2E frame timing now also separates propagation internals through
`sam3_propagate_timing` and `sam3_last_propagate_timing()`. Benchmark frame
timing JSONL rows include graph build/allocation, input upload, graph compute,
output read, memory update, bbox/result construction, and accounting remainder
for `sam3_propagate_encoded_frame`.

This matters for optimization because it distinguishes the necessary model
work from benchmark artifacts and from cached-tail shortcuts. On the current
SAM3 BF16 same-contract 4-frame C++ run, accounting still closes to zero:

| Component | Three-frame tail total |
|---|---:|
| Tail encode required | `454.515 ms` |
| Tail encode graph compute | `451.885 ms` |
| Tail propagate required | `78.708 ms` |
| Tail propagate graph compute | `66.367 ms` |
| Tail propagate input upload | `2.986 ms` |
| Tail propagate graph alloc | `1.143 ms` |
| Tail propagate graph build | `0.412 ms` |
| Tail propagate output read | `0.549 ms` |
| Tail propagate accounted | `78.154 ms` |
| Tail propagate remainder | `0.554 ms` |

Evidence:
`outputs/e2e-separated-sam3-bf16-prop-split-r3-20260628b/summary.json`,
`outputs/e2e-separated-sam3-bf16-prop-split-r3-20260628b/component-table.md`,
and
`outputs/e2e-separated-sam3-bf16-prop-split-r3-20260628b/component-split.json`.

The measured optimization denominator is still required/session E2E. The new
propagation split shows that the largest remaining required work is not
decode, artifact output, graph build, or CPU bookkeeping. It is still the
CUDA graph compute of the per-frame image encode; propagation graph compute is
the next but much smaller target.

Additional CUDA A/B runs under the same required-E2E contract rejected three
candidate changes:

| Candidate | Required E2E | Tail encode graph compute | Decision |
|---|---:|---:|---|
| Default BF16 | `837.850 ms` | `452.312 ms` | keep |
| `SAM3_ENABLE_FATTN56_INLINE_LARGE=1` | `841.508 ms` | `453.657 ms` | reject |
| `SAM3_ENABLE_FATTN56_NATIVE_V=1` | `840.970 ms` | `454.514 ms` | reject |
| `GGML_CUDA_DISABLE_FATTN56_DIRECT_OUT=1` | `847.382 ms` | `454.185 ms` | reject; direct output is beneficial |
| `GGML_CUDA_DISABLE_ROPE_PAIR_FUSION=1` | `976.024 ms` | `523.949 ms` | reject; RoPE pair fusion is required |

Evidence:
`outputs/e2e-separated-sam3-bf16-kernel-ab-r3-20260628b/component-table.md`.

The synchronized node profile remains diagnostic only because it inserts
per-node synchronization and disables normal CUDA graph behavior. Its hotspot
ranking is still useful: SAM3 encode is dominated by ViT projection GEMMs
(`MLP FC2 / attention projection`, `MLP FC1`, `QKV`), window/global
FlashAttention, GELU, residual adds, and Q/K contiguous materialization before
RoPE. Propagation is mostly memory-attention and decoder graph compute, but at
about one seventh of the tail encode graph-compute total in this short E2E
contract.

Evidence:
`outputs/e2e-separated-sam3-bf16-profile-nodes-r1-20260628b/sam3-encode-hotspots.json`
and
`outputs/e2e-separated-sam3-bf16-profile-nodes-r1-20260628b/propagate-hotspots.json`.

The next optimization should therefore be a root CUDA implementation change in
the SAM3 ViT encode path, not a cached-tail benchmark change. The strongest
candidate classes are projection-GEMM dataflow, attention/RoPE materialization
boundaries, GELU/MLP fusion where it wins under normal E2E timing, and avoiding
unnecessary contiguous/cast bridges while preserving the same BF16/TF32
precision contract and output parity.

### 2026-06-28 Required E2E Input Split

The C++ benchmark now separates required input preparation from model-session
work instead of reporting `input_load_ms` as an opaque value. Result JSON and
frame timing JSONL expose:

- `input_directory_scan_ms`
- `input_sort_ms`
- `input_frame_decode_ms`
- `input_frame_store_ms`
- `input_remainder_ms`
- a `stage:"input"` frame-timing row

This keeps the required E2E denominator unchanged:

`required_e2e_ms = input_load_ms + model_e2e_ms`

but makes it clear which part is host input preparation and which part is model
execution. `scripts/model_matrix_compare.py` and
`scripts/summarize_e2e_component_splits.py` aggregate these fields across
repeats.

Validation run:
`outputs/e2e-input-split-sam3-bf16-r2-20260628/summary.json`,
`outputs/e2e-input-split-sam3-bf16-r2-20260628/component-table.md`, and
`outputs/e2e-input-split-sam3-bf16-r2-20260628/component-split.json`.

For SAM3 BF16, 4 frames, C++ only, no output artifacts, and 2 repeats:

| Component | Mean |
|---|---:|
| Required E2E | `848.343 +/- 0.966 ms` |
| Input load | `21.181 +/- 0.284 ms` |
| Input frame decode | `21.085 +/- 0.291 ms` |
| Model E2E | `827.161 +/- 1.249 ms` |
| Frame 0 encode | `153.661 +/- 0.400 ms` |
| Frame 0 prompt model | `130.872 +/- 3.449 ms` |
| Tail encode required | `455.977 +/- 1.591 ms` |
| Tail encode graph compute | `453.322 +/- 1.592 ms` |
| Tail propagate required | `78.418 +/- 3.569 ms` |
| Tail propagate graph compute | `63.962 +/- 0.069 ms` |
| Required accounting delta | `0.000 ms` |

The input split confirms that required host input preparation is measurable but
not the bottleneck in this contract. It is about 2.5% of required E2E, while
tail encode graph compute alone is about 53%. Optimization work should continue
on the SAM3 ViT encode CUDA graph and its ggml kernels/dataflow.

One tested root-side fusion candidate is rejected under normal E2E timing:
cuDNN MLP FC1+GELU F32 matched the SAM3 ViT MLP shape contract but regressed
the same C++ required-E2E run from `841.451 +/- 3.632 ms` to
`889.149 +/- 4.015 ms`. The regression came mostly from frame0 encode and tail
encode graph compute, so it remains opt-in diagnostic code rather than a default
optimization.

Evidence:
`outputs/e2e-separated-sam3-bf16-cudnn-mlp-fc1-gelu-f32-r5-20260628d/component-table.md`.

The same input-split denominator also rejects cuDNN SDPA head64 with both
global and window attention enabled. Required E2E regressed from
`848.343 +/- 0.966 ms` to `871.821 +/- 1.141 ms`; model E2E regressed by
`+23.196 ms`, and tail encode graph compute regressed by `+16.185 ms`. The
slight propagation graph-compute improvement does not compensate for the encode
regression, so this remains diagnostic-only.

Evidence:
`outputs/e2e-input-split-sam3-bf16-cudnn-sdpa-head64-r2-20260628/component-table.md`.

### 2026-06-28 Required E2E Core Compute Split

The C++ comparison pipeline now derives the required core graph-compute portion
from frame timing JSONL. This separates the benchmark denominator into:

- required input preparation: video/frame decode and storage before inference
- model session E2E: frame 0 encode/prompt work plus required tail work
- required core graph compute: frame 0 encode graph compute, tail encode graph
  compute, tail propagate graph compute, and tail propagate cache compute
- non-core required work: graph allocation/build, prompt/model overhead, output
  readback, and benchmark bookkeeping that is still required by the API path

The required E2E denominator is unchanged. The new fields only make the measured
work attributable:

`required_e2e_ms = input_load_ms + model_e2e_ms`

Validation run:
`outputs/e2e-required-core-split-sam3-bf16-r2b-20260628/summary.json` and
`outputs/e2e-required-core-split-sam3-bf16-r2b-20260628/component-table.md`.

For SAM3 BF16, 4 frames, C++ only, no output artifacts, and 2 repeats:

| Component | Mean |
|---|---:|
| Required E2E | `847.550 +/- 0.315 ms` |
| Input load | `21.439 +/- 0.132 ms` |
| Input frame decode | `21.345 +/- 0.132 ms` |
| Model E2E | `826.111 +/- 0.183 ms` |
| Frame 0 encode | `154.233 +/- 0.107 ms` |
| Frame 0 prompt model | `131.208 +/- 1.208 ms` |
| Tail encode required | `455.862 +/- 0.848 ms` |
| Tail encode graph compute | `452.743 +/- 0.105 ms` |
| Tail propagate required | `76.089 +/- 0.175 ms` |
| Tail propagate graph compute | `63.421 +/- 0.096 ms` |
| Required core graph compute | `669.324 ms` |
| Required core graph compute share | `79.0%` |
| Tail encode graph compute share | `99.3%` |
| Required accounting delta | `0.000 ms` |

This confirms that the optimization target is still model compute, not input
decode or output artifacts. Input frame decode is about 2.5% of required E2E,
while required core graph compute is about 79.0%. Tail encode is almost entirely
graph compute, so improvements must come from the SAM3 ViT encode CUDA/ggml
dataflow and kernels.

Additional existing-path A/B runs were tested under the same denominator and
rejected:

| Candidate | Required E2E | Required core graph compute | Tail encode graph compute | Decision |
|---|---:|---:|---:|---|
| Default BF16 | `847.550 +/- 0.315 ms` | `669.324 ms` | `452.743 +/- 0.105 ms` | keep |
| Disable direct QKV views | `872.145 +/- 0.642 ms` | `692.302 ms` | `468.378 +/- 1.353 ms` | reject |
| Materialize attention V | `859.581 +/- 2.332 ms` | `680.641 ms` | `460.596 +/- 0.068 ms` | reject |
| MLP flat chain | `852.805 +/- 2.345 ms` | `669.586 ms` | `452.113 +/- 2.133 ms` | reject |
| MLP FC1 cuBLASLt algo1 | `852.369 +/- 1.020 ms` | `676.299 ms` | `457.865 +/- 0.083 ms` | reject |
| Attention projection cuBLASLt algo1 | `849.279 +/- 1.898 ms` | `673.690 ms` | `456.168 +/- 2.179 ms` | reject |

Evidence:
`outputs/e2e-required-core-split-sam3-bf16-cublaslt-algo-r2-20260628/component-ab.md`
and
`outputs/e2e-required-core-split-sam3-bf16-existing-path-ab-r2-20260628/component-ab.md`.

Next optimization work should avoid cached-tail changes or input-pipeline tuning
as the primary strategy. The remaining high-value area is a root CUDA/ggml
change for ViT encode, especially projection GEMM dataflow, attention/RoPE
materialization boundaries, GELU/MLP fusion only if it wins under normal E2E,
and avoiding unnecessary contiguous/cast bridges while preserving parity.

### 2026-06-28 Required E2E Python Contract Diagnostics

`scripts/model_matrix_compare.py` now keeps formal comparisons and SAM3-family
diagnostics separate. `comparable_*_rows` still require an exact C++/Python
family match. New `same_contract_*_diagnostic_rows` pair SAM3-family C++ rows
with the configured official SAM3/SAM3.1 Python baseline only for bottleneck
analysis. They are not parity or speed claims unless `exact_family_match=true`.

The exact SAM3 BF16/TF32 contract was remeasured with 4 PNG frames, input size
1008, 2 warmups, 5 repeats, no output artifacts, and C++ cached-tail accounting
enabled. C++ cached-tail preencoding is still included in `model_e2e_ms`, so it
does not remove required E2E work.

| Row | C++ | Official Python SAM3 | Ratio / note |
|---|---:|---:|---|
| Track scope | `78.647 +/- 2.404 ms/fr` | `141.311 +/- 1.438 ms/fr` | Python/C++ `1.797x` |
| Model E2E | `829.421 +/- 7.385 ms` | `797.208 +/- 5.620 ms` | Python/C++ `0.961x` |
| Required/session E2E | `850.818 +/- 7.689 ms` | `856.262 +/- 6.691 ms` | Python/C++ `1.006x` |
| C++ tail encode required | `458.979 +/- 2.311 ms` | n/a | C++ required work |
| C++ tail encode graph compute | `454.781 ms` | n/a | `99.1%` of tail encode |

Evidence:
`outputs/e2e-required-core-split-sam3-bf16-python-sam3-warmup2-r5-20260628/summary.json`
and
`outputs/e2e-required-core-split-sam3-bf16-python-sam3-warmup2-r5-20260628/component-table.md`.

Interpretation: C++ is slightly faster on the required/session E2E denominator,
but the margin is small relative to run-to-run variance. The stronger, stable
C++ advantage is the track-scope row; the weaker part is model E2E when Python
`start_session` is excluded. The C++ bottleneck remains ViT image encode:
required core graph compute is `670.828 ms`, and tail encode graph compute alone
is `454.781 ms`.

The same machinery also produced a SAM3-family diagnostic against official
Python `sam3.1`, but this row is not a formal SAM3.1 parity/speed claim because
the C++ row is still `family=sam3` and `exact_family_match=false`.

| Diagnostic | C++ SAM3 BF16 | Official Python SAM3.1 BF16/TF32 | Note |
|---|---:|---:|---|
| Required/session E2E | `847.019 +/- 0.864 ms` | `1441.089 +/- 12.677 ms` | diagnostic only |
| Track scope | `76.394 +/- 0.127 ms/fr` | `22.001 +/- 0.247 ms/fr` | Python cached tail is faster |
| C++ tail encode graph compute | `452.040 ms` | n/a | still the C++ root target |

Evidence:
`outputs/e2e-required-core-split-sam3-bf16-cached-tail-contract-warmup2-r2-20260628/summary.json`.

This confirms why cached-tail numbers must be separated. C++ can look faster on
session E2E while still losing the cached-tail propagation slice to Python
SAM3.1. That is a boundary diagnostic, not proof that SAM3.1 tracking has been
beaten at the same implementation contract.

One additional ggml CUDA root candidate was checked and rejected:
`GGML_CUDA_ENABLE_UNARY_CPY_VEC4_PACKED=1` for the fused
`GELU_ERF -> CPY(F32->BF16)` path. It slightly reduced tail encode timing, but
did not improve the required E2E denominator:

| Variant | Required E2E | Tail encode required | Required core graph compute | Decision |
|---|---:|---:|---:|---|
| Default | `850.818 +/- 7.689 ms` | `458.979 +/- 2.311 ms` | `670.828 ms` | keep |
| Packed vec4 BF16 unary-copy | `851.991 +/- 4.113 ms` | `457.490 +/- 2.001 ms` | `670.678 ms` | reject |

Evidence:
`outputs/e2e-required-core-split-sam3-bf16-unary-packed-r5-20260628/summary.json`
and
`outputs/e2e-required-core-split-sam3-bf16-unary-packed-r5-20260628/component-table.md`.

Next root optimization should therefore stay on the ViT encode graph itself:
projection/MLP GEMM dataflow, QKV/RoPE/FATTN materialization, or a larger
same-precision fusion that reduces the `~455 ms` tail encode graph-compute
bucket under the required E2E denominator.

### 2026-06-28 Required E2E Tail Work Separation

The C++ benchmark now separates the tail work that is visible during the timed
tracking loop from the tail image-encode work that is precomputed for
`--cpp-preencode-cached-tail-frames`. This keeps cached-tail runs usable for
diagnosis without treating precomputed work as a speed win. The new frame-timing
and summary fields are:

- `tail_runtime_wall_ms`: wall time actually spent in the measured tail loop
- `tail_visible_model_ms`: tail-loop model time after optional artifact time is
  removed
- `tail_preencode_required_model_ms`: required model work moved before the
  timed tail loop
- `tail_required_model_ms`: visible tail model work plus required preencode
  model work
- `tail_required_wall_ms`: per-frame wall equivalent after adding required
  preencode work back

The accounting run uses SAM3 BF16, 4 PNG frames, 1008 encode size, C++ only, no
output artifacts, cached-tail preencoding enabled, 2 warmups, and 3 measured
repeats:
`outputs/e2e-required-process-split-sam3-bf16-tail-required-r3-20260628a/component-table.md`.

| Component | Mean |
|---|---:|
| Required/session E2E | `839.315 +/- 4.888 ms` |
| Model E2E | `817.904 +/- 4.582 ms` |
| Tail runtime wall | `231.754 +/- 4.398 ms` |
| Tail visible model | `231.752 +/- 4.399 ms` |
| Tail preencode required model | `303.556 +/- 0.679 ms` |
| Tail required model | `535.308 +/- 4.514 ms` |
| Tail encode required total | `454.538 +/- 2.468 ms` |
| Tail encode graph compute | `450.402 +/- 0.256 ms` |
| Tail propagate required | `80.768 +/- 4.901 ms` |
| Tail propagate graph compute | `59.392 +/- 0.168 ms` |
| Tail propagate input upload | `11.953 +/- 3.625 ms` |
| Required core graph compute | `662.705 ms` |
| Required core graph compute share | `79.0%` |
| Tail encode graph compute share | `99.1%` |
| Tail required model accounting delta | `-0.000 ms` |

The important result is that the cached-tail path no longer hides E2E-required
work: `tail_runtime_wall_ms + tail_preencode_required_model_ms` accounts for
`tail_required_model_ms` within rounding noise. The remaining optimization
target is still the SAM3 ViT tail image encode path, not decode, artifacts, or a
benchmark accounting shortcut.

The follow-up required-process split keeps the same execution contract but
separates required tail image encode by where the work is executed:
`outputs/e2e-required-process-split-sam3-bf16-tail-required-resplit-r3-20260628a/component-table.md`.

| Required tail encode split | Mean |
|---|---:|
| Tail encode required total | `453.896 +/- 2.256 ms` |
| Tail encode graph compute | `450.937 +/- 2.213 ms` |
| Inline tail encode | `150.121 +/- 0.499 ms` |
| Inline tail encode graph compute | `149.061 +/- 0.579 ms` |
| Cached-tail preencode encode | `303.775 +/- 1.899 ms` |
| Cached-tail preencode encode graph compute | `301.876 +/- 1.853 ms` |
| Cached-tail preencode state create | `0.001 +/- 0.001 ms` |
| Tail encode split accounting delta | `0.000 ms` |
| Tail encode graph-compute split accounting delta | `0.000 ms` |

This shows that the cached-tail mode only changes when image encoding executes.
It does not change the required work: roughly one encoded tail frame is visible
inside the tail loop and two encoded tail frames are accounted as preencode
work. The optimization target is therefore the shared image-encode graph used by
both rows, not the cached-tail scheduler or state creation.

One root CUDA candidate was tried against this split. The SAM3 BF16 model stores
the neck deconvolution weights as BF16, but the SAM3 graph historically casts
them to F16 before `ggml_conv_transpose_2d_p0`. A guarded BF16 deconvolution path
was added behind `SAM3_CONV_TRANSPOSE_KEEP_BF16=1` and the ggml CUDA
`CONV_TRANSPOSE_2D` k=2/stride=2 GEMM path now accepts BF16 input when such a
graph reaches it. This is not enabled by default.

Evidence:
`outputs/e2e-required-process-split-sam3-bf16-keep-bf16-deconv-r3-20260628a/component-ab.md`
and
`outputs/e2e-required-process-split-sam3-bf16-keep-bf16-deconv-parity-20260628a/default-vs-keep-bf16-deconv.json`.

| Variant | Required E2E | Tail encode required | Tail encode graph compute | Decision |
|---|---:|---:|---:|---|
| Default F16-cast deconv | `844.057 +/- 5.796 ms` | `453.896 +/- 2.256 ms` | `450.937 +/- 2.213 ms` | keep |
| Opt-in BF16 deconv | `842.068 +/- 1.651 ms` | `455.161 +/- 3.406 ms` | `451.161 +/- 2.018 ms` | reject |

The opt-in path did not reduce the required tail image-encode work. It also
changed the frame-0 mask by 4 pixels and produced tiny score deltas up to
`2.3e-05`, so it is not acceptable as the default exact-parity path. Default
determinism remains exact:
`outputs/e2e-required-process-split-sam3-bf16-default-determinism-20260628a/default-determinism.json`
reports `diff_rows=0`.

Two existing ViT dataflow diagnostics were remeasured with the same
required-tail split so they cannot accidentally benefit from cached-tail
accounting:

| Variant | Required E2E | Tail encode required | Tail encode graph compute | Decision |
|---|---:|---:|---:|---|
| Default split baseline | `844.057 +/- 5.796 ms` | `453.896 +/- 2.256 ms` | `450.937 +/- 2.213 ms` | keep |
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` | `843.858 +/- 7.646 ms` | `453.626 +/- 1.528 ms` | `450.607 +/- 1.856 ms` | reject as neutral |
| `SAM3_ENABLE_VIT_CONTIGUOUS_ATTENTION_V=1` | `858.743 +/- 7.797 ms` | `464.216 +/- 2.681 ms` | `460.577 +/- 1.491 ms` | reject as slower |

Evidence:
`outputs/e2e-required-process-split-sam3-bf16-mlp-flat-chain-r3-20260628a/component-table.md`
and
`outputs/e2e-required-process-split-sam3-bf16-contiguous-v-r3-20260628a/component-table.md`.
The MLP flat-chain path is effectively noise-level for the current SAM3 BF16
required E2E denominator, and forcing attention V contiguous is a clear
regression. Neither is the missing root speedup.

A short cuBLASLt timing profile was taken to decide whether the next target
should be hidden BF16 conversion or the GEMM/attention dataflow itself:
`outputs/e2e-required-process-split-sam3-bf16-cublaslt-profile-20260628a/cublaslt-bias-timing-by-group.json`.
This is a profiling run only; it is not normal wall-clock evidence because it
uses synchronized timing hooks. For the main ViT linear layers, all major rows
hit the cuBLASLt fused-bias path as `bf16 x bf16 -> f32` with no input or output
conversion:

| ViT group | Calls | Median matmul | Convert sum | Dst convert sum |
|---|---:|---:|---:|---:|
| `vit_mlp_fc2` | `128` | `0.943168 ms` | `0.000 ms` | `0.000 ms` |
| `vit_mlp_fc1` | `128` | `0.934864 ms` | `0.000 ms` | `0.000 ms` |
| `vit_qkv` | `128` | `0.609344 ms` | `0.000 ms` | `0.000 ms` |
| `vit_attn_proj` | `128` | `0.226304 ms` | `0.000 ms` | `0.000 ms` |

This rules out a simple "remove hidden cuBLASLt conversion" explanation for the
current E2E gap. The next optimization should target a root ViT encode change:
GEMM dataflow that can beat the current cuBLASLt fused-bias baseline, an
attention/RoPE/layout change that avoids materialization around those GEMMs, or
a larger same-contract fusion that reduces the `tail_encode_graph_compute_ms`
bucket under the required-E2E accounting above.

### 2026-06-28 Required E2E Direct Component Fields

The benchmark now emits the required-E2E compute split directly in each C++ JSON
row, so the process split does not depend on a sidecar frame-timing JSONL file.
The direct fields include:

- `tail_inline_encode_ms` and `tail_preencode_encode_ms`
- `tail_inline_encode_graph_compute_ms` and
  `tail_preencode_encode_graph_compute_ms`
- `tail_encode_graph_compute_ms`
- `tail_propagate_graph_compute_ms`
- `tail_propagate_cache_compute_ms`
- `required_core_compute_ms`
- `required_non_core_compute_ms`
- `required_core_compute_fraction`
- `tail_encode_graph_compute_fraction`

This also fixes required-tail accounting when `timed_start_frame` is not frame
1: preencoded tail frames are always counted as required work, regardless of
whether their execution was moved before the timed tail loop.

Latest same-contract SAM3 BF16 direct-field comparison against official Python
SAM3 BF16/TF32 uses 4 PNG frames, 1008 encode size, the same selected person
prompt, `timed_start_frame=1`, cached-tail preencoding enabled for C++, no C++
output artifacts, 2 C++ warmups, and 3 measured repeats:
`outputs/model-matrix-sam3-bf16-required-e2e-direct-split-20260628a/summary.json`.

| Process slice | C++ SAM3 BF16 | Official Python SAM3 BF16/TF32 | Python/C++ |
|---|---:|---:|---:|
| Required/session E2E | `842.949 +/- 2.390 ms` | `842.986 ms` | `1.000x` |
| Model E2E | `821.567 +/- 3.101 ms` | `793.347 ms` | `0.966x` |
| Input prepare / start session | `21.381 +/- 0.713 ms` | `49.615 ms` | `2.320x` |
| Frame-0 prompt / detection | `120.566 +/- 0.207 ms` | `349.748 ms` | `2.901x` |
| Tail image encode / backbone | `455.794 +/- 1.190 ms` | `335.916 ms` | `0.737x` |
| Tail image encode graph compute | `452.369 +/- 1.356 ms` | `335.916 ms` | `0.743x` |
| Tail propagate / tracker | `82.360 +/- 4.612 ms` | `71.996 ms` | `0.874x` |
| Tail propagate graph compute | `59.597 +/- 0.185 ms` | `71.996 ms` | `1.208x` |

Interpretation: the session-level result is effectively tied because C++ input
prepare and frame-0 prompt setup are faster, but C++ still loses the model-only
denominator. The root gap is the SAM3 ViT tail image encode path: direct
required-core compute is `665.641 ms`, `79.0%` of required E2E, and tail image
encode graph compute is `99.2%` of tail image encode required wall time.

Existing CUDA switches and prototype kernels were retested against this direct
split. None are acceptable default optimizations for same-precision SAM3 BF16:

| Variant | Required E2E | Model E2E | Tail encode graph compute | Decision |
|---|---:|---:|---:|---|
| Baseline direct split | `844.667 +/- 2.705 ms` | `823.469 +/- 2.118 ms` | `451.357 +/- 1.046 ms` | keep |
| cuBLASLt `mlp.lin2 + residual` | `844.499 +/- 6.171 ms` | `823.167 +/- 5.753 ms` | `452.793 +/- 0.610 ms` | reject as neutral/slower |
| cuDNN `fc1 + GELU` F32 | `887.529 +/- 3.756 ms` | `865.987 +/- 3.437 ms` | `482.400 +/- 1.497 ms` | reject |
| cuDNN `fc1 + GELU` BF16 | `886.271 +/- 1.281 ms` | `865.158 +/- 1.447 ms` | `484.009 +/- 1.307 ms` | reject |
| Direct cuBLASLt BF16 dst | `849.714 +/- 3.480 ms` | `828.227 +/- 3.029 ms` | `453.661 +/- 1.595 ms` | reject |

Evidence:
`outputs/model-matrix-sam3-bf16-cpp-baseline-direct-split-20260628b/summary.json`,
`outputs/model-matrix-sam3-bf16-cpp-cublaslt-residual-20260628b/summary.json`,
`outputs/model-matrix-sam3-bf16-cpp-cudnn-mlp-f32-20260628b/summary.json`,
`outputs/model-matrix-sam3-bf16-cpp-cudnn-mlp-bf16-20260628b/summary.json`,
and
`outputs/model-matrix-sam3-bf16-cpp-cublaslt-direct-bf16-dst-20260628b/summary.json`.

cuBLASLt shape-specific heuristic screening was also run for the returned
SAM3 ViT candidates. Every tested non-zero index regressed the direct split, so
the default heuristic remains the current baseline:

| Variant | Required E2E | Tail encode graph compute | Decision |
|---|---:|---:|---|
| Baseline | `845.154 ms` | `451.958 ms` | keep |
| `VIT_MLP_FC1=1` | `848.531 ms` | `458.139 ms` | reject |
| `VIT_MLP_FC2=1` | `860.585 ms` | `458.328 ms` | reject |
| `VIT_MLP_FC2=2` | `861.042 ms` | `461.857 ms` | reject |
| `VIT_QKV=1` | `846.423 ms` | `456.044 ms` | reject |
| `VIT_ATTN_PROJ=1` | `858.796 ms` | `459.190 ms` | reject |

Evidence:
`outputs/model-matrix-sam3-bf16-cpp-algo-baseline-screen-20260628b/summary.json`
and the sibling `outputs/model-matrix-sam3-bf16-cpp-algo-*-screen-20260628b`
screening directories.

Finally, the standalone BF16 MLP chain and WMMA prototypes were rechecked for
the SAM3 MLP shape `1024 -> 4736 -> 1024, cols=5184`. The chain path is not
faster than the current graph path under the same precision contract, and the
faster direct-FC1 prototype changes the checksum. These remain benchmark-only
experiments, not production paths. Evidence:
`outputs/sam3-bf16-mlp-chain-bench-20260628b/mlp-chain.log` and
`outputs/sam3-bf16-gelu-cast-bench-20260628b/gelu-wmma.log`.

The next root optimization must therefore be a real ViT image-encode dataflow
change, not an environment-variable enablement: either a same-contract fused
MLP/attention path that beats the current cuBLASLt matmul baseline, or a layout
change that removes materialization around QKV/RoPE/FATTN without moving work
out of the required-E2E denominator.

### 2026-06-28 Required E2E Propagation Upload Split

`sam3_propagate_timing`, benchmark frame-timing JSONL, and
`scripts/model_matrix_compare.py` now split propagation input upload into:

- `propagate_input_prompt_upload_ms`
- `propagate_input_rope_upload_ms`
- `propagate_input_memory_upload_ms`
- `propagate_input_feature_upload_ms`
- `propagate_input_constant_upload_ms`
- `propagate_input_sparse_upload_ms`

The same totals are also exposed as `tail_propagate_input_*_upload_ms` in the
benchmark summary row and in `scripts/summarize_e2e_component_splits.py`.
This keeps required-E2E accounting direct: `propagate_input_upload_ms` remains
the parent timing, and these fields are sub-splits rather than additional
accounted time.

Smoke evidence:
`outputs/model-matrix-sam3-bf16-upload-split-smoke-20260628a/summary.json` and
`outputs/model-matrix-sam3-bf16-upload-split-smoke-20260628a/component_split.md`.
On SAM3 BF16, 4 frames, 1008 encode size, 5 repeats, default CUDA path:

| Component | Mean |
|---|---:|
| Required E2E | `836.666 +/- 3.517 ms` |
| Model E2E | `826.305 +/- 3.552 ms` |
| Tail encode graph compute | `456.875 +/- 1.416 ms` |
| Tail propagate graph compute | `62.127 +/- 0.210 ms` |
| Tail propagate input upload | `2.825 +/- 0.063 ms` |
| Tail propagate prompt upload | `1.260 +/- 0.064 ms` |
| Tail propagate RoPE upload | `1.524 +/- 0.010 ms` |

Interpretation: propagation upload is measurable but not the root gap. The
required-E2E optimization target remains SAM3 ViT image encode.

An experimental backend `rope_k` cache was added behind
`SAM3_ENABLE_PROP_ROPE_K_BACKEND_CACHE=1`. It is not default-on. It removes the
per-frame `rope_k` build/upload work but pays a backend cache construction cost
inside short-video E2E:

| Variant | Required E2E | Tail propagate | RoPE K build | RoPE upload | Decision |
|---|---:|---:|---:|---:|---|
| Default | `836.666 +/- 3.517 ms` | `74.022 +/- 2.162 ms` | `2.174 ms` | `1.524 ms` | keep |
| `SAM3_ENABLE_PROP_ROPE_K_BACKEND_CACHE=1` | `846.587 +/- 4.253 ms` | `84.566 +/- 1.920 ms` | `0.001 ms` | `0.000 ms` | reject for short E2E |

Parity for the opt-in path is exact against default on the 4-frame bbox JSONL
check: `diff_rows == 0`, all bbox IoUs `1.0`, mask hashes equal. Evidence:
`outputs/sam3-bf16-ropek-cache-parity-20260628a/diff.json`.

### 2026-06-28 Required E2E Encode/Admin Split And Model Prompt PE Cache

The required-E2E component split now separates the image-encode side enough to
distinguish compute, graph administration, input upload, preprocessing,
state-update, prompt-PE construction, and residual/unattributed time. The new
summary fields include `required_encode_non_compute_ms`,
`required_encode_graph_admin_ms`, `tail_encode_graph_admin_ms`, and
`tail_propagate_graph_admin_ms`; the process summary also reports
`tail_image_encode.non_compute_required`, `tail_image_encode.graph_admin`,
`tail_image_encode.preprocess`, and `tail_image_encode.remainder`.

The SAM3 BF16 baseline on the 4-frame, encode-size `1008`, cached-tail contract
shows that tail image encode is already almost entirely graph compute:

| Component | Mean |
|---|---:|
| Required E2E | `834.843 +/- 7.085 ms` |
| Model E2E | `824.475 +/- 7.191 ms` |
| Tail image encode required | `453.152 +/- 1.494 ms` |
| Tail image encode graph compute | `449.983 +/- 1.618 ms` |
| Required encode non-compute | `4.075 ms` |
| Tail propagate | `84.486 +/- 5.966 ms` |
| Tail propagate input upload | `13.955 +/- 3.805 ms` |
| Tail propagate constant upload | `11.193 +/- 3.803 ms` |

Evidence:
`outputs/e2e-required-encode-split-sam3-bf16-baseline-r3-20260628a/component_split.md`.

`SAM3_BF16_VIT_WIN_PART_INPUT=1` remains rejected under this split. It increased
required E2E from `834.843 +/- 7.085 ms` to `847.050 +/- 9.894 ms`, model E2E
from `824.475 +/- 7.191 ms` to `836.676 +/- 9.942 ms`, and tail image encode
graph compute from `449.983 +/- 1.618 ms` to `454.288 +/- 1.123 ms`. Evidence:
`outputs/e2e-required-encode-split-sam3-bf16-winpart-input-r3-20260628a/baseline-vs-winpart-input-component.md`.

The accepted root-side cleanup is a model-owned prompt positional-encoding
cache. It keeps the dense prompt PE and small prompt constants in the model for
reuse by new tracking states with the same model and input geometry. It is
enabled by default and can be disabled with `SAM3_DISABLE_MODEL_PROMPT_PE_CACHE=1`.
This is not a ViT encode optimization; it removes repeated per-session constant
construction/upload from the required tail propagation path.

Same-binary r5 A/B, with the cache disabled versus default-on:

| Component | Disabled | Default-on | Delta |
|---|---:|---:|---:|
| Required E2E | `833.428 +/- 5.738 ms` | `827.459 +/- 1.379 ms` | `-5.969 ms` |
| Model E2E | `823.001 +/- 5.702 ms` | `817.164 +/- 1.382 ms` | `-5.838 ms` |
| Tail propagate | `79.526 +/- 3.772 ms` | `71.891 +/- 0.608 ms` | `-7.635 ms` |
| Tail propagate input upload | `10.986 +/- 2.823 ms` | `2.808 +/- 0.292 ms` | `-8.179 ms` |
| Tail propagate constant upload | `8.232 +/- 2.839 ms` | `0.011 +/- 0.001 ms` | `-8.221 ms` |
| Required graph admin | `18.667 ms` | `8.966 ms` | `-9.701 ms` |
| Tail image encode graph compute | `451.267 +/- 2.755 ms` | `459.963 +/- 1.158 ms` | `+8.696 ms` |

Evidence:
`outputs/e2e-required-encode-split-sam3-bf16-model-pe-cache-default-r5-20260628a/disabled-vs-default-component.md`.

Parity against the disabled path is exact for the 4-frame bbox JSONL check:
`diff_rows == 0`, `tolerance_diff_rows == 0`, all bbox IoUs are `1.0`, max bbox
delta is `0.0 px`, max score delta is `0.0`, and all four mask hashes match.
Evidence:
`outputs/sam3-bf16-model-pe-cache-parity-20260628a/disabled-vs-default.json`.

Interpretation: the cache is accepted because it reduces measured required E2E
non-compute work with exact parity. It does not close the official Python
model-E2E gap. The next performance target is still the SAM3 ViT image encode
graph compute path, where the required split shows about `99%` of tail image
encode time is spent in graph compute rather than preprocessing or upload.

### 2026-06-28 Full-Graph CUDA Stage Profile For SAM3 Encode

The next profiling layer now separates `sam3_encode` within the E2E-required
image-encode graph without executing unsafe partial graphs. A direct
`SAM3_PROFILE_ENCODE_CUTS=1` partial-graph attempt reached the prefix cut, then
the next CUDA call reported an illegal-memory-access from the diagnostic graph.
CUDA partial cuts are therefore disabled in `sam3_encode`; the log prints
`SAM3_PROFILE_ENCODE_CUT status=disabled_cuda ... use=GGML_CUDA_PROFILE_NODES`
and normal inference continues. The safe path is:

- collect a normal full-graph CUDA node profile with `GGML_CUDA_PROFILE_NODES=1`
- summarize it with `scripts/summarize_cuda_stage_profile.py`
- use the result for hotspot ranking only, because node profiling synchronizes
  per node and is not a profile-free runtime measurement

Evidence:
`outputs/sam3-bf16-cuda-stage-profile-20260628d/stage-summary.json`,
`outputs/sam3-bf16-cuda-stage-profile-20260628d/hotspots-sam3-encode.json`.

On the same SAM3 BF16, 4-frame, encode-size `1008`, cached-tail contract, the
profiled `sam3_encode` compute calls were:

| Call type | Compute calls | Profiled graph compute |
|---|---:|---:|
| Frame-0 encode | `1` | `431.050 ms` |
| Tail tracking encode | `3` | `144.665`, `146.368`, `146.518 ms` |

The steady tracking-stage ranking from the full-graph node stream is:

| Tracking stage | Mean | Drop-max mean |
|---|---:|---:|
| Neck | `11.658 ms` | `11.651 ms` |
| ViT block 23 | `4.579 ms` | `4.574 ms` |
| ViT block 15 | `4.568 ms` | `4.564 ms` |
| ViT block 31 | `4.538 ms` | `4.536 ms` |
| ViT block 30 | `4.524 ms` | `4.524 ms` |
| ViT block 22 | `4.523 ms` | `4.521 ms` |
| ViT prefix | `0.585 ms` | `0.581 ms` |

Representative per-stage op splits show the same bottleneck as the signature
hotspot view. In the tracking neck, `CONV_2D` is `3.554 ms`,
`CONV_TRANSPOSE_2D` is `3.468 ms`, and layout/elementwise work is about
`3.2 ms`. In global ViT blocks, each stage is mostly `MUL_MAT` at about
`2.09 ms` plus `FLASH_ATTN_EXT` at about `1.80 ms`. The signature hotspot
ranking across all `sam3_encode` calls is still dominated by ViT matmuls:

| Hotspot signature | Drop-max sum |
|---|---:|
| MLP `fc2` / attention projection `MUL_MAT` | `147.035 ms` |
| MLP `fc1` `MUL_MAT` | `116.981 ms` |
| QKV `MUL_MAT` | `76.524 ms` |
| Window `FLASH_ATTN_EXT` | `47.573 ms` |
| MLP `GELU` | `30.707 ms` |
| Global `FLASH_ATTN_EXT` | `26.795 ms` |

Parity for the disabled CUDA partial-cut guard is exact against the default
path on the 4-frame bbox JSONL check: `diff_rows == 0`,
`tolerance_diff_rows == 0`, all bbox IoUs `1.0`, max bbox delta `0.0 px`, max
score delta `0.0`, and all four mask hashes match. Evidence:
`outputs/sam3-bf16-profile-cut-disabled-parity-20260628d/diff.json`.

Interpretation: the E2E-required decomposition is now deep enough to separate
tail image-encode admin/upload from full-graph compute and to rank compute
inside the encoder by stage. No new speed claim is made by this profiling
change. The next accepted optimization still needs to attack the ViT block
dataflow itself, primarily the repeated `MUL_MAT` and `FLASH_ATTN_EXT` work,
rather than another shallow environment toggle or an unsafe partial-graph
measurement.

### 2026-06-28 Required E2E Process Split Refresh

The current comparison tooling now treats cached-tail runs as an accounting
mode, not as a shortcut. C++ work moved before the timed tail loop is added
back through `tail_preencode_required_model_ms`,
`tail_encode_required_total_ms`, and `tail_required_model_ms`, so
`model_e2e_ms` and `required_e2e_ms` still include the image encode work needed
for an end-to-end session. The process summary table also marks each row with
whether it is part of C++ required E2E and whether the Python measurement is a
formal session-E2E field or a diagnostic hook.

Fresh SAM3 BF16/TF32 evidence, 5 decoded frames, `timed_start_frame=2`, C++
cached-tail preencode, no output artifacts, 2 C++ warmups, and 2 measured
repeats:

| Process slice | C++ SAM3 BF16 | Official Python SAM3 BF16/TF32 | Python/C++ |
|---|---:|---:|---:|
| Required/session E2E | `1001.118 ms` | `1078.052 ms` | `1.077x` |
| Model E2E | `988.438 ms` | `1013.891 ms` | `1.026x` |
| Input prepare / start session | `12.680 ms` | `64.138 ms` | `5.058x` |
| Frame-0 prompt / detection | `130.396 ms` | `344.800 ms` | `2.644x` |
| Tail image encode / backbone | `574.208 ms` | `358.416 ms` | `0.624x` |
| Tail image encode graph compute | `570.399 ms` | `358.416 ms` | `0.628x` |
| Tail propagate / tracker | `99.192 ms` | `96.808 ms` | `0.976x` |
| Tail propagate graph compute | `84.332 ms` | `96.808 ms` | `1.148x` |

Evidence:
`outputs/e2e-required-split-sam3-bf16-cached-tail-r2-20260628e/summary.json`,
`outputs/e2e-required-split-sam3-bf16-cached-tail-r2-20260628e/process_split.md`,
and
`outputs/e2e-required-split-sam3-bf16-cached-tail-r2-20260628e/component_split.md`.

Interpretation: C++ wins this short required/session E2E row because input
prepare and frame-0 prompt/detection are faster. That is not the same as saying
the tracking tail is faster. The direct tail comparison still shows the image
encoder as the root performance gap: required core compute is `829.633 ms`
(`82.9%` of required E2E), and tail image encode graph compute is `99.3%` of
tail image encode required time.

The cuBLASLt timing profile for the same warmed shape confirms that the large
ViT projection matmuls remain dominant:

| ViT GEMM group | Calls | Matmul sum |
|---|---:|---:|
| MLP FC2 | `480` | `468.995 ms` |
| MLP FC1 | `480` | `464.321 ms` |
| QKV | `480` | `303.394 ms` |
| Attention projection | `480` | `112.494 ms` |

Evidence:
`outputs/e2e-required-split-sam3-bf16-cached-tail-r2-20260628e/cublaslt-profile/cublaslt-groups.json`
and
`outputs/e2e-required-split-sam3-bf16-cached-tail-r2-20260628e/cublaslt-profile/cublaslt-shapes.json`.

Two shallow CUDA policy probes were rejected on this refreshed split:

| Candidate | Required E2E | Tail encode graph compute | Decision |
|---|---:|---:|---|
| Baseline | `1001.118 ms` | `570.399 ms` | keep |
| cuBLASLt residual epilogue | `1006.774 ms` | `569.888 ms` | reject; graph compute is neutral and E2E regresses |
| `VIT_QKV=1` | `1009.191 ms` | `572.764 ms` | reject |
| `VIT_ATTN_PROJ=1` | `1013.973 ms` | `572.531 ms` | reject |
| `VIT_MLP_FC1=1` | `1019.535 ms` | `575.125 ms` | reject |
| `VIT_MLP_FC2=1` | `1019.380 ms` | `577.955 ms` | reject |
| `VIT_MLP_FC2=2` | `1025.643 ms` | `579.776 ms` | reject |

Evidence:
`outputs/e2e-required-split-sam3-bf16-residual-fusion-r2-20260628e/summary.json`
and
`outputs/e2e-algo-shape-sweep-20260628a/sweep_summary.md`.

Accepted next direction: the E2E accounting is now strict enough to avoid
claiming wins from cached-tail placement or artifact removal. Further speedups
must reduce the actual SAM3 ViT image-encode graph compute, most likely by a
same-contract fused MLP/attention path or by removing QKV/RoPE/FATTN
materialization boundaries without changing the model output contract.

### 2026-06-28 Same-Contract E2E Split Guard

The required-E2E split now has a direct `just` entrypoint:
`just e2e-required-split`. It builds `sam3_benchmark`, runs the SAM3 BF16
same-contract model matrix, and writes `summary.json`,
`component_split.{json,md}`, and `process_split.md`. The recipe defaults to
`models/sam3`, 5 decoded frames, `timed_start_frame=1`, BF16 Python, TF32 on,
C++ cached-tail accounting, selected-only text init, no C++ output artifacts,
and C++ frame-timing JSONL. `MODEL_MATRIX_PYTHON_RESULTS` reuse is now guarded:
reused Python rows must match `frames`, `timed_start_frame`, dtype, and TF32
policy, otherwise `scripts/model_matrix_compare.py` exits before producing a
mixed-contract comparison.

While checking the target-selection contract, PCS/text-init detections were
also fixed to populate `sam3_detection::iou_score` with the same score already
stored in `sam3_detection::score` and `mask.iou_score`. Before the fix, the
benchmark selected text-init detections by an all-zero `iou_score` field; the
sample happened to keep candidate 0 because it was also the highest score, but
other candidate orders could select a different person than official Python's
`out_probs`-max rule.

Fresh same-contract SAM3 BF16/TF32 evidence, 5 decoded frames,
`timed_start_frame=1`, 2 warmups, and 2 measured repeats:

| Process slice | C++ SAM3 BF16 | Official Python SAM3 BF16/TF32 | Python/C++ |
|---|---:|---:|---:|
| Required/session E2E | `939.263 ms` | `1073.821 ms` | `1.143x` |
| Model E2E | `926.435 ms` | `1015.869 ms` | `1.097x` |
| Input prepare / start session | `12.828 ms` | `57.929 ms` | `4.516x` |
| Frame-0 prompt / detection | `119.663 ms` | `346.141 ms` | `2.893x` |
| Tail image encode / backbone | `587.978 ms` | `521.797 ms` | `0.887x` |
| Tail image encode graph compute | `583.486 ms` | `521.797 ms` | `0.894x` |
| Tail propagate / tracker | `49.097 ms` | `113.807 ms` | `2.318x` |
| Tail propagate graph compute | `41.572 ms` | `113.807 ms` | `2.738x` |

Evidence:
`outputs/e2e-required-split-sam3-bf16-same-contract-r2-20260628a/summary.json`,
`outputs/e2e-required-split-sam3-bf16-same-contract-r2-20260628a/process_split.md`,
and
`outputs/e2e-required-split-sam3-bf16-same-contract-r2-20260628a/component_split.md`.

Interpretation: C++ is faster on the formal same-contract short-session E2E
row, but the root image-encode work is still behind Python. Required core graph
compute is `785.541 ms` (`83.6%` of required E2E), and tail image encode graph
compute is `99.2%` of tail image encode time. The next accepted speedup must
reduce that ViT image-encode graph compute, not only move work across the
cached-tail boundary.

Two fusion probes were rejected before promotion to the formal A/B path. The
cuDNN FC1+GELU path hit the SAM3 ViT FC1 shape but regressed the 2-frame smoke
required E2E to `1020.940 ms`. The cuBLASLt GELU-ERF epilogue probe changed
the initial mask hashes and also regressed the 2-frame smoke required E2E to
`970.326 ms`, so it is not a same-precision optimization candidate.

### 2026-06-28 E2E Attribution Tooling Refresh

The same-contract E2E loop now has three separate measurement layers:

- `just e2e-required-split`: normal unsynchronized speed/parity denominator.
- `just e2e-required-profile`: synchronized CUDA-node attribution for the same
  required-E2E C++ contract.
- `just e2e-required-mulmat-profile`: GEMM attribution for the same contract,
  including both normal `GGML_CUDA_PROFILE_MUL_MAT` rows and cuBLASLt
  bias-fusion timing rows.

The ViT block graph now names QKV, attention projection, MLP FC1, and MLP FC2
matmul nodes explicitly, so `scripts/summarize_cuda_node_names.py` can separate
`sam3-vit:qkv-matmul`, `sam3-vit:proj-matmul`,
`sam3-vit:mlp-fc1-matmul`, and `sam3-vit:mlp-fc2-matmul` instead of grouping
them as anonymous nodes.

Fresh normal same-contract rerun after the measurement naming change stayed
within noise of the previous result: required/session E2E `935.300 +/- 1.742 ms`,
model E2E `922.590 +/- 1.721 ms`, tail image encode graph compute
`583.665 +/- 1.156 ms`, and tail propagate graph compute
`41.195 +/- 0.260 ms`
(`outputs/e2e-required-split-sam3-bf16-measurement-names-r3-20260628a/`).

The synchronized node attribution for a short 3-frame run shows the tracking
image encoder is dominated by ViT GEMM and then attention/GELU:

| Tracking encode group | Calls | Sum ms |
|---|---:|---:|
| MLP FC2 matmul | `32` | `29.777` |
| MLP FC1 matmul | `32` | `29.448` |
| QKV matmul | `32` | `19.266` |
| Window attention | `28` | `11.691` |
| GELU | `32` | `7.335` |
| Attention projection matmul | `32` | `7.178` |
| Global attention | `4` | `7.179` |

Evidence:
`outputs/e2e-required-profile-named-matmul-20260628a/node_names_split.json` and
`outputs/e2e-required-profile-named-matmul-20260628a/stage_profile.json`.

The GEMM-specific profile confirms that the dominant ViT projection work is the
cuBLASLt bias-fusion path, not the plain `MUL_MAT` profile path:

| cuBLASLt ViT group | Calls | Matmul sum | Mean |
|---|---:|---:|---:|
| MLP FC2 | `96` | `91.284 ms` | `0.951 ms` |
| MLP FC1 | `96` | `90.312 ms` | `0.941 ms` |
| QKV | `96` | `59.462 ms` | `0.619 ms` |
| Attention projection | `96` | `21.896 ms` | `0.228 ms` |

Evidence:
`outputs/e2e-required-gemm-profile-sam3-bf16-timing-20260628a/cublaslt_bias_by_group.json`.

Decision: no new speedup is accepted from this measurement slice. The actionable
optimization target is now constrained: reduce actual SAM3 ViT image-encode
graph compute, especially the FC1/FC2/QKV cuBLASLt path or the F32
materialization around it, and keep acceptance tied to
`just e2e-required-split` rather than profiling-only timings.

### 2026-06-28 Required-E2E Process Isolation Update

The required-E2E measurement path now records enough sub-processes to separate
accounting changes from real model work. C++ cached-tail mode is still allowed
for the application contract, but `tail_inline_encode_ms` and
`tail_preencode_encode_ms` are added back into `tail_encode_required_total_ms`,
and the graph-only denominator is exposed as `tail_encode_graph_compute_ms`.
This avoids treating cached-tail placement as a speedup unless the actual
encode graph compute also drops.

The profile path now also names SAM3 neck nodes. `sam3_encode#02` in the
3-frame synchronized profile is split as follows:

| Tracking encode group | Calls | Sum ms |
|---|---:|---:|
| MLP FC2 matmul | `32` | `29.781` |
| MLP FC1 matmul | `32` | `29.464` |
| QKV matmul | `32` | `19.243` |
| Window attention | `28` | `11.707` |
| GELU | `32` | `7.341` |
| Attention projection matmul | `32` | `7.190` |
| Global attention | `4` | `7.150` |
| Neck conv3x3 | `4` | `3.551` |
| Neck deconv | `3` | `3.466` |
| Neck conv1x1 | `12` | `1.065` |

Evidence:
`outputs/e2e-required-profile-neck-names-20260628a/node_names_split.json`.

The normal unsynchronized same-contract rerun after neck naming stayed at the
same performance level: required/session E2E `936.705 +/- 1.499 ms`, model E2E
`923.926 +/- 1.568 ms`, tail image encode total
`588.266 +/- 2.808 ms`, tail image encode graph compute
`584.007 +/- 2.474 ms`, tail propagate `50.182 +/- 1.420 ms`, and tail
propagate graph compute `42.069 +/- 0.914 ms`.

Evidence:
`outputs/e2e-required-split-sam3-bf16-neck-names-r3-20260628a/summary.json`,
`outputs/e2e-required-split-sam3-bf16-neck-names-r3-20260628a/component_split.md`,
and
`outputs/e2e-required-split-sam3-bf16-neck-names-r3-20260628a/process_split.md`.

Candidate optimizations tested against the same required-E2E denominator were
not promoted:

| Candidate | Required E2E | Tail encode graph | Decision |
|---|---:|---:|---|
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` | `938.900 +/- 2.446 ms` | `583.477 +/- 2.375 ms` | no clear win |
| `SAM3_DISABLE_DIRECT_NECK_3X3_CONV=1` | `977.341 +/- 1.847 ms` | `595.688 +/- 0.587 ms` | worse |
| `SAM3_BF16_VIT_LINEAR_OUTPUT=1` | `956.885 +/- 5.970 ms` | `588.028 +/- 1.453 ms` | worse |
| `SAM3_BF16_VIT_MLP_CHAIN=1` | `1043.964 +/- 0.772 ms` | `644.520 +/- 0.437 ms` | worse |
| MLP chain + direct BF16 dst | `1011.531 +/- 6.795 ms` | `620.798 +/- 1.085 ms` | worse |
| `SAM3_BF16_VIT_QKV_CHAIN=1` | `1017.038 +/- 0.605 ms` | `625.722 +/- 0.893 ms` | worse |
| cuBLASLt ViT algo index `1` | `969.148 +/- 3.530 ms` | `597.738 +/- 2.215 ms` | worse |
| cuBLASLt FC2 algo index `2` | `958.510 +/- 0.599 ms` | `590.577 +/- 0.653 ms` | worse |

The `--cpp-env` parser now accepts either repeated `--cpp-env KEY=VALUE`
arguments or a shell-style space-separated string such as
`--cpp-env 'A=1 B=2'`. This matters for A/B runs from `just` because
`MODEL_MATRIX_CPP_ENV` is passed as one argument by the recipe.

Current accepted optimization target: do not default-enable BF16 chain,
direct-BF16-dst, or alternate cuBLASLt heuristics for SAM3 BF16. The remaining
gap is still the actual ViT image-encode graph, mostly FC1/FC2/QKV cuBLASLt
work. The next useful implementation should introduce a real fused path or
remove a materialization boundary, then pass `just e2e-required-split` with
equal parity and lower `tail_encode_graph_compute_ms`.

### 2026-06-28 Required-E2E Optimization Recheck

The required-E2E split was used as the acceptance denominator for another
kernel/dataflow pass. The split now separates:

- `input_load_ms` and `input_frame_decode_ms` for required host input work.
- `frame0_encode_ms` and `frame0_prompt_model_ms` for prompt/setup work.
- `tail_encode_required_total_ms`, split into inline encode and cached-tail
  preencode, so cached placement is not counted as a speedup.
- `tail_encode_graph_compute_ms` as the main image-encode kernel/dataflow
  denominator.
- `required_core_compute_ms` versus `required_non_core_compute_ms` to avoid
  mixing graph compute with upload/build/accounting changes.

The baseline for this slice is the artifact-free same-contract SAM3 BF16 r3
run:

| Variant | Required E2E | Model E2E | Tail encode graph | Required core compute | Tail propagate |
|---|---:|---:|---:|---:|---:|
| Baseline neck-named | `936.705 +/- 1.499 ms` | `923.926 +/- 1.568 ms` | `584.007 +/- 2.474 ms` | `785.855 +/- 1.524 ms` | `50.182 +/- 1.420 ms` |
| FC2 residual epilogue opt-in | `951.877 +/- 2.300 ms` | `939.134 +/- 2.300 ms` | `584.807 +/- 1.021 ms` | `790.528 +/- 0.095 ms` | `51.912 +/- 1.328 ms` |
| cuBLASLt bias auto-tune opt-in | `956.306 +/- 2.771 ms` | `943.333 +/- 2.689 ms` | `589.229 +/- 0.886 ms` | `798.897 +/- 1.617 ms` | `51.755 +/- 1.765 ms` |
| QKV heuristic index `1` only | `954.298 +/- 3.101 ms` | `941.295 +/- 3.049 ms` | `589.279 +/- 0.986 ms` | `797.478 +/- 2.263 ms` | `51.964 +/- 1.892 ms` |

Evidence:
`outputs/e2e-required-split-sam3-bf16-neck-names-r3-20260628a/summary.json`,
`outputs/e2e-required-split-sam3-bf16-cublaslt-fc2-residual-enabled-r3-20260628a/summary.json`,
`outputs/e2e-required-split-sam3-bf16-cublaslt-autotune-r3-20260628a/summary.json`,
and
`outputs/e2e-required-split-sam3-bf16-qkv-algo1-r3-20260628a/summary.json`.

The residual epilogue path now has an exact-overlap memory-range exception for
the opt-in FC2 residual case, so the cuBLASLt beta/residual epilogue can be
tested instead of being rejected by the fusion memory check. It is still not a
default optimization because the required-E2E run regressed.

`GGML_CUDA_CUBLASLT_BIAS_AUTOTUNE=1` was added as a diagnostic opt-in. It
performs one non-captured warmup execution, times returned cuBLASLt bias
heuristics by shape, caches the selected algorithm, and then lets later CUDA
graph captures use that cached algorithm. The auto-tune profile selected QKV
heuristic index `1` and kept index `0` for the main FC1/FC2/projection shapes:

| cuBLASLt group | Selected indices | Profiled matmul sum | E2E decision |
|---|---:|---:|---|
| ViT MLP FC2 | `0` | `59.824 ms` | no promotion |
| ViT MLP FC1 | `0` | `59.100 ms` | no promotion |
| ViT QKV | `1` | `39.411 ms` | rejected by E2E |
| ViT attention projection | `0` | `14.329 ms` | no promotion |

Evidence:
`outputs/e2e-required-mulmat-profile-sam3-bf16-autotune-20260628a/cublaslt_bias_by_group.json`.

Decision: none of these candidates is accepted as a default speedup. The
profile-only QKV improvement does not translate into lower
`tail_encode_graph_compute_ms`; QKV index `1` alone also regresses E2E. The next
root optimization should move beyond cuBLASLt heuristic selection and target an
actual materialization or kernel boundary in the ViT image encoder, with the
same required-E2E split as the acceptance gate.

### 2026-06-28 Required-E2E Optimization Target Audit

The process split is now also summarized by
`scripts/summarize_e2e_optimization_targets.py` and the
`just e2e-required-audit` target. This combines:

- normal required-E2E component timing from `component_split.json`;
- C++/Python process rows from `process_split.md` and comparison summaries;
- synchronized CUDA node attribution from `node_names_split.json`;
- cuBLASLt bias-GEMM attribution from `cublaslt_bias_by_group.json`.

Fresh audit output:
`outputs/e2e-required-audit-sam3-bf16-20260628a/optimization_targets.md` and
`outputs/e2e-required-audit-sam3-bf16-20260628a/optimization_targets.json`.

The audit keeps the same acceptance denominator as the normal split. Current
required/session E2E is `936.705 ms`; required core graph compute is
`785.855 ms` (`83.9%`). The only C++/Python process gap that is still negative
for C++ is tail image encode:

| Process | C++ | Python diagnostic | Python/C++ | C++ gap |
|---|---:|---:|---:|---:|
| Tail image encode / backbone | `588.266 ms` | `521.797 ms` | `0.887x` | `+66.469 ms` |
| Tail image encode graph compute | `584.007 ms` | `521.797 ms` | `0.893x` | `+62.210 ms` |

Required non-core work is not the bottleneck for the image encoder:
`tail_encode_graph_admin_ms` is `4.115 ms`, tail encode preprocess is
`0.001 ms`, and tail encode remainder is `0.007 ms`. Cached-tail placement is
therefore accounting context only; it cannot count as an accepted speedup unless
`tail_encode_graph_compute_ms` or required/session E2E also drops.

The tail encode node-stage ranking points at the same root target:

| Tail encode node stage | Calls | Profile sum |
|---|---:|---:|
| ViT MLP FC2 matmul | `32` | `29.781 ms` |
| ViT MLP FC1 matmul | `32` | `29.464 ms` |
| ViT QKV matmul | `32` | `19.243 ms` |
| ViT window attention | `28` | `11.707 ms` |
| ViT GELU | `32` | `7.341 ms` |
| ViT attention projection matmul | `32` | `7.190 ms` |
| ViT global attention | `4` | `7.150 ms` |

cuBLASLt attribution for the same required-E2E C++ contract ranks the dominant
BF16-to-F32 projection groups as MLP FC2, MLP FC1, and QKV. Those remain the
right implementation targets. Existing toggles for BF16 linear output, QKV/MLP
chain output, cuBLASLt heuristic selection, residual epilogues, window-part
input, and contiguous attention V are not promoted because they either regress
the required-E2E denominator or fail parity.

Decision: the next root change should be a real SAM3 ViT dataflow/kernel change
for the MLP/QKV projection path or the head64 attention path. Optimizations that
only reduce input load, graph admin, cached-tail placement, or profiling-local
matmul timing are insufficient unless the required-E2E audit shows the tail
image encode graph itself moved.

The audit script now also accepts candidate component splits through
`--candidate-component-split`, wired from
`E2E_REQUIRED_AUDIT_CANDIDATE_COMPONENT_SPLITS` in `just e2e-required-audit`.
This keeps future kernel/dataflow experiments on the same acceptance denominator:

- `required_e2e_ms` must improve, not just a profiling-local timer;
- `tail_encode_graph_compute_ms` must improve, because tail image encode is the
  remaining C++/Python gap;
- graph-admin or cached-tail-only movement is diagnostic context, not an accepted
  model-speed win.

Smoke comparison output:
`outputs/e2e-required-audit-sam3-bf16-candidates-20260628a/optimization_targets.md`
and
`outputs/e2e-required-audit-sam3-bf16-candidates-20260628a/optimization_targets.json`.

| Candidate | Required E2E delta | Tail graph delta | Core compute delta | Decision |
|---|---:|---:|---:|---|
| `qkv-algo1` | `+17.593 ms` | `+5.271 ms` | `+11.623 ms` | reject |
| `cublaslt-autotune` | `+19.601 ms` | `+5.222 ms` | `+13.042 ms` | reject |

This confirms that the next optimization should not be another cuBLASLt
heuristic toggle. The useful implementation work is a projection dataflow/kernel
change for ViT MLP FC2/FC1/QKV, or a head64 attention kernel change, validated
by this same candidate-delta audit.

## 2026-06-28 required-E2E A/B target and cuDNN head64 rejection

`just e2e-required-ab` now runs a baseline and a candidate under one
required-E2E contract, reuses the baseline Python result for the candidate, and
then writes a combined `optimization_targets.json` / `optimization_targets.md`.
The candidate report includes both component deltas and process deltas, so a
change must move the real E2E denominator and the relevant required process, not
only a local kernel stopwatch.

The default split still uses the application-style cached-tail,
selected-target, no-artifact C++ contract. These are now explicit switches:
`E2E_REQUIRED_SPLIT_CPP_PREENCODE_CACHED_TAIL_FRAMES`,
`E2E_REQUIRED_SPLIT_CPP_TEXT_INIT_SELECTED_ONLY`, and
`E2E_REQUIRED_SPLIT_CPP_NO_OUTPUT_ARTIFACTS`. Set the selected-only switch empty
when measuring the official all-detections Python contract.

The first run used SAM3 F16 C++ versus official SAM3 Python `fp16`/TF32, with
the same 5-frame cached-tail accounting used by the optimization split:

`outputs/e2e-required-ab-sam3-f16-cudnn-head64-window-r3-20260628a/optimization_targets.md`.

This A/B is valid for C++ baseline-versus-candidate acceptance. The C++ rows
still use selected-target text init, so the Python rows remain diagnostic for
the official all-detections contract rather than a formal C++/Python parity
claim.

| Candidate | Required E2E delta | Model E2E delta | Tail encode graph delta | Core compute delta | Decision |
|---|---:|---:|---:|---:|---|
| `cudnn-head64-window` | `+53.788 ms` | `+53.786 ms` | `+29.232 ms` | `+56.677 ms` | reject |

The process split shows where the regression lands:

| Process | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Required session E2E | `965.690 ms` | `1019.478 ms` | `+53.788 ms` |
| Model E2E | `953.023 ms` | `1006.809 ms` | `+53.786 ms` |
| Tail image encode graph compute | `595.055 ms` | `624.287 ms` | `+29.232 ms` |
| Frame-0 encode | `173.075 ms` | `198.352 ms` | `+25.277 ms` |
| Tail propagate graph compute | `42.798 ms` | `45.043 ms` | `+2.244 ms` |

Short same-contract profiles explain the rejection. With existing FATTN:

`outputs/e2e-required-profile-sam3-f16-fattn-baseline-20260628a/profile.log`

and with cuDNN head64/window routing:

`outputs/e2e-required-profile-sam3-f16-cudnn-head64-window-20260628a/profile.log`

the SAM3 ViT attention totals were:

| Path | Shape group | Calls | Kernel/execute sum | Convert/output sum | Total |
|---|---|---:|---:|---:|---:|
| Existing FATTN | window | `140` | `51.054 ms` kernel | `12.505 ms` other | `64.034 ms` |
| cuDNN head64 | window | `140` | `50.442 ms` execute | `51.451 ms` convert/out | `101.893 ms` |
| Existing FATTN | global | `20` | `37.160 ms` kernel | `1.853 ms` other | `39.013 ms` |
| cuDNN head64 | global | `20` | `42.596 ms` execute | `7.359 ms` convert/out | `49.954 ms` |

The cuDNN route is therefore not just hidden by noise. For the current SAM3
graph, Q/K/V are still F32 at the FATTN boundary, so the cuDNN path pays
F32-to-BF16 conversion, V packing, and output transpose costs on every call.
Existing FATTN is faster in the actual required-E2E context.

Next optimization target remains the ViT projection/MLP path. The required
tail encode graph is still dominated by MLP FC2, MLP FC1, and QKV matmuls; a
future accepted change should reduce those dataflow/kernel costs and pass the
same `e2e-required-ab` candidate decision.

## 2026-06-28 required-E2E process split and official F16 audit

`just e2e-required-split` now writes an additional
`required_process_split.json` / `required_process_split.md`. This is a
C++-only required-E2E decomposition, separate from the C++/Python diagnostic
process table. The rollup rows are exclusive and use
`required_e2e.total = input.prepare + model.session_setup + frame0.image_encode
+ frame0.prompt_and_tracker_init + tail.image_encode + tail.propagate +
tail.unattributed + model.remainder`.

`scripts/summarize_e2e_optimization_targets.py` now consumes those
`required_processes` rows and reports `Candidate Required Process Deltas`, so
future A/B reports can distinguish real required model work from input,
graph-admin, cached-tail placement, and accounting movement.

The official all-detections SAM3 F16 baseline used:

`outputs/e2e-required-ab-sam3-f16-linear-output-all-r3-20260628a/baseline/`.

| Required process | Mean |
|---|---:|
| `required_e2e.total` | `1033.430 ms` |
| `tail.image_encode` | `589.506 ms` |
| `tail.image_encode.graph_compute` | `585.332 ms` |
| `frame0.image_encode` | `188.616 ms` |
| `frame0.prompt_and_tracker_init` | `131.938 ms` |
| `tail.propagate` | `102.425 ms` |
| `input.prepare` | `12.704 ms` |
| `model.session_setup` | `8.234 ms` |

Same-contract profiling for this baseline is recorded in:

- `outputs/e2e-required-profile-sam3-f16-official-all-detections-20260628a/`;
- `outputs/e2e-required-mulmat-profile-sam3-f16-official-all-detections-20260628a/`;
- `outputs/e2e-required-audit-sam3-f16-official-all-detections-20260628a/optimization_targets.md`.

The tail image-encode node profile for `sam3_encode#02` ranks the work as:

| Tail encode stage | Calls | Profile sum |
|---|---:|---:|
| `sam3-vit:mlp-fc1-matmul` | `32` | `30.886 ms` |
| `sam3-vit:mlp-fc2-matmul` | `32` | `28.851 ms` |
| `sam3-vit:qkv-matmul` | `32` | `20.220 ms` |
| `sam3-vit:window-attn` | `56` | `12.774 ms` |
| `sam3-vit:global-attn` | `8` | `7.570 ms` |
| `sam3-vit:proj-matmul` | `32` | `7.443 ms` |
| `sam3-vit:mlp-gelu` | `32` | `7.370 ms` |

cuBLASLt attribution under the same contract shows the same target order:

| cuBLASLt group | Calls | Matmul sum | Selected algo |
|---|---:|---:|---|
| `vit_mlp_fc1` | `160` | `157.599 ms` | `[2]` |
| `vit_mlp_fc2` | `160` | `145.149 ms` | `[2]` |
| `vit_qkv` | `160` | `103.127 ms` | `[2]` |
| `vit_attn_proj` | `160` | `37.670 ms` | `[2]` |

Candidate decisions with the new required-process delta table:

| Candidate | Required E2E delta | Tail graph delta | Decision |
|---|---:|---:|---|
| `cublaslt-autotune` | `-5.194 ms` | `+1.160 ms` | reject/diagnostic |
| `vit-mlp-flat-chain` | `+3.917 ms` | `+6.160 ms` | reject |
| `vit-contiguous-attention-v` | `+16.722 ms` | `+13.946 ms` | reject |
| `f16-linear-output-all` | `+148.265 ms` | `+149.481 ms` | reject |

`cublaslt-autotune` is not promoted despite the lower mean required-E2E value,
because the improvement is not statistically strong (`e2e z = -0.418`), and the
tail image-encode graph itself regresses. The next accepted optimization must
move `tail.image_encode.graph_compute`, preferably MLP FC1/FC2/QKV dataflow or
kernel execution, under the same official all-detections required-E2E contract.

## 2026-06-29 required-E2E acceptance gate refresh

The required-E2E audit now treats the process split as the acceptance surface,
not just a descriptive table. `scripts/summarize_e2e_optimization_targets.py`
classifies a candidate as `candidate_speedup` only when:

- `required_e2e_ms` improves;
- `tail_encode_graph_compute_ms` improves;
- both improvements have standard-error signal `z <= -2.0`.

If both means improve but the signal is weak or unavailable, the result is
`needs_more_repeats`. If either required E2E or tail graph compute regresses, the
result stays `reject_or_keep_diagnostic`.

The official all-detections SAM3 F16 audit was regenerated with the process
deltas and the additional direct-F16-dst probe:

`outputs/e2e-required-audit-sam3-f16-official-all-detections-20260629a/optimization_targets.md`.

| Candidate | Required E2E delta | Tail graph delta | E2E z | Tail z | Decision |
|---|---:|---:|---:|---:|---|
| `cublaslt-autotune` | `-5.194 ms` | `+1.160 ms` | `-0.418` | `+0.207` | reject |
| `cublaslt-workspace128` | `-3.429 ms` | `+0.595 ms` | `-0.519` | `+1.412` | reject |
| `cublaslt-fc2-residual` | `-0.436 ms` | `+2.996 ms` | `-0.064` | `+4.120` | reject |
| `cublaslt-fast16f-compute` | `+1.694 ms` | `+0.896 ms` | `+0.245` | `+3.158` | reject |
| `vit-mlp-flat-chain` | `+3.917 ms` | `+6.160 ms` | `+0.601` | `+8.398` | reject |
| `vit-contiguous-attention-v` | `+16.722 ms` | `+13.946 ms` | `+2.523` | `+18.993` | reject |
| `cublaslt-gelu-erf-fusion` | `+70.719 ms` | `+59.730 ms` | `+10.684` | `+65.230` | reject |
| `f16-mlp-fc1-output-direct-f16dst` | `+79.268 ms` | `+69.280 ms` | `+12.089` | `+58.295` | reject |
| `f16-mlp-fc1-output` | `+97.417 ms` | `+85.883 ms` | `+14.856` | `+64.158` | reject |
| `cublaslt-compute16f` | `+147.377 ms` | `+125.938 ms` | `+22.526` | `+134.796` | reject |
| `f16-linear-output-all` | `+148.265 ms` | `+149.481 ms` | `+22.181` | `+74.924` | reject |

The direct-F16-dst probe confirms that the FC1-output shortcut is not merely
paying an extra F32-to-F16 conversion. It reduces some non-core accounting, but
the required tail image encode grows from `589.506 ms` to `658.525 ms`, and the
tail graph compute grows from `585.332 ms` to `654.612 ms`.

ggml CUDA now also has opt-in cuBLASLt bias compute-type probes for this path:
`GGML_CUDA_CUBLASLT_BIAS_FAST_16F_FOR_F16=1`,
`GGML_CUDA_CUBLASLT_BIAS_COMPUTE_16F_FOR_F16=1`, and
`GGML_CUDA_CUBLASLT_BIAS_FAST_16BF_FOR_BF16=1`. Defaults are unchanged. On the
official SAM3 F16 contract, `FAST_16F` is nearly neutral but still misses the
acceptance gate (`tail graph +0.896 ms`), while `COMPUTE_16F` is much slower
(`tail graph +125.938 ms`). These should remain diagnostic switches, not
defaults.

Current implementation conclusion: cached-tail scheduling, graph admin, cuBLASLt
workspace size, cuBLASLt residual epilogue, direct F16 output, and cuDNN
FC1+GELU do not reduce the required SAM3 ViT image-encode graph. The next
optimization should be a real kernel/dataflow change for the ViT projection path:
MLP FC1, MLP FC2, and QKV remain the measured targets, and every candidate must
pass the same required-E2E process split before it can be promoted.

## 2026-06-29 required-E2E parity and mask-contract split

The required-E2E A/B path now separates three concerns:

- speed denominator: `e2e-required-split` keeps using
  `--cpp-no-output-artifacts`, so JSONL/PNG/debug output does not enter
  `required_e2e_ms`;
- artifact parity: `e2e-required-parity` runs baseline and candidate separately
  with JSONL/mask output, then writes `compare.json`;
- candidate promotion: `summarize_e2e_optimization_targets.py` accepts
  `--candidate-parity NAME=PATH`, and a speed candidate can only become
  `candidate_speedup` when both required speed gates pass and parity is exact.

`e2e-required-ab` now invokes the parity run after the speed A/B and feeds that
`compare.json` into the final optimization report. `e2e-required-audit` also
accepts `E2E_REQUIRED_AUDIT_CANDIDATE_PARITY` for already-measured candidates.

The same split can be used for the mask contract. By default the historical
required-E2E optimization run remains bbox-only. Setting
`E2E_REQUIRED_SPLIT_CPP_MASK_OUTPUT=1` passes `--cpp-mask-output` to
`model_matrix_compare.py`, so full-mask model work is included in the speed
denominator while output artifacts remain disabled.

New artifacts:

- `outputs/e2e-required-parity-sam3-f16-cublaslt-fast16f-20260629a/compare.json`;
- `outputs/e2e-required-parity-sam3-f16-qkv-output-direct-f16dst-20260629a/compare.json`;
- `outputs/e2e-required-parity-sam3-f16-fc1-add-gelu-20260629a/compare.json`;
- `outputs/e2e-required-parity-sam3-f16-qkv-materialized-layout-20260629a/compare.json`;
- `outputs/e2e-required-audit-sam3-f16-parity-candidates-20260629d/optimization_targets.md`;
- `outputs/e2e-required-profile-sam3-f16-rope-probe-20260629a/profile.log`;
- `outputs/e2e-required-split-sam3-f16-cuda-graphs-r3-20260629a/component_split.json`;
- `outputs/e2e-required-split-sam3-f16-vit-inplace-ln-affine-r3-20260629a/component_split.json`;
- `outputs/e2e-required-split-sam3-f16-full-mask-baseline-r3-20260629a/component_split.json`.

Candidate results under the bbox-only required contract:

| Candidate | Required E2E delta | Tail graph delta | Parity | Diff rows | Decision |
|---|---:|---:|---|---:|---|
| `cublaslt-fast16f-compute` | `+1.694 ms` | `+0.896 ms` | exact | `0` | reject |
| `f16-qkv-output-direct-f16dst` | `+5.800 ms` | `+8.297 ms` | failed | `5` | reject |
| `f16-vit-inplace-ln-affine` | `+6.740 ms` | `+2.160 ms` | not run | `-` | reject |
| `f16-cuda-graphs` | `+9.573 ms` | `+7.401 ms` | not run | `-` | reject |
| `f16-qkv-materialized-layout` | `+37.023 ms` | `+30.899 ms` | exact | `0` | reject |
| `f16-fc1-add-gelu` | `+71.848 ms` | `+62.675 ms` | failed | `4` | reject |

The QKV direct-F16-output probe is both slower and not exact: min bbox IoU is
`0.999931`, max bbox delta is `0.008 px`, max score delta is `0.000065`, min
mask pixel IoU is `0.999714`, and max mask XOR is `9` pixels. This is a small
numeric drift, but it is not eligible for the exact-parity speed gate and it
also regresses the measured tail encode graph.

Two additional dataflow probes were rejected by the same gate:

- disabling direct QKV views is exact, but it makes the required tail image
  encode graph slower by `+30.899 ms`; the current direct-view path should stay
  default;
- forcing FC1 away from the cuBLASLt GELU epilogue toward ADD+GELU fusion makes
  tail graph compute slower by `+62.675 ms` and introduces small score-only
  JSONL drift, so the current cuBLASLt epilogue path should stay default.

Two launch/layout-administration probes were also rejected:

- `SAM3_CUDA_ENABLE_GRAPHS=1` allows ggml CUDA graph capture but regresses this
  required-E2E contract (`tail graph +7.401 ms`), so SAM3 should keep disabling
  CUDA graphs around its current per-frame graph construction;
- a local SAM3 ViT-only in-place LayerNorm affine prototype reduced neither the
  required denominator nor tail graph compute (`tail graph +2.160 ms`). The
  prototype was discarded instead of adding another dead tuning switch.

`GGML_CUDA_PROFILE_ROPE_PAIR_FUSION=1` confirms that the ViT Q/K
contiguous-pack plus RoPE path is already taking the fused CUDA route:
`GGML_CUDA_CONT_ROPE_PAIR_FUSION success` appears `1056` times in the 5-frame
profile. The remaining `CONT_ROPE_PAIR_FUSION reject` lines are ordinary
non-RoPE `CONT` nodes or unrelated pattern misses. RoPE is still a small
cleanup target, but it is not the root gap compared with MLP FC1/FC2/QKV GEMM.

The full-mask baseline is now available as a separate denominator, not as a
candidate against bbox-only:

| Contract | Required E2E | Tail encode | Tail graph | Tail propagate |
|---|---:|---:|---:|---:|
| bbox-only baseline | `1033.430 ms` | `589.506 ms` | `585.332 ms` | `102.425 ms` |
| full-mask baseline | `1035.561 ms` | `592.924 ms` | `588.650 ms` | `103.585 ms` |

At this sample size the full-mask delta is small, but future Rust-wrapper-facing
claims should use the full-mask contract when the application needs masks. The
optimization target is unchanged: reduce the actual SAM3 ViT image-encode
compute, especially MLP FC1/FC2/QKV, with a candidate that passes both the speed
and parity gates.

## 2026-06-29 F16 required-E2E contract guard and cuDNN MLP recheck

The required-E2E split now carries `python_dtype`, `tf32_policy`, and the
C++ text-init cardinality into `component_split.json`, and the optimization
audit prints the Python dtype/TF32 policy in its header. This prevents a
`sam3-f16` C++ row from being read as a same-precision Python comparison when
the Python run was actually BF16.

Two explicit just entrypoints were added for the same reason:

- `just e2e-required-split-sam3-f16 <out>`;
- `just e2e-required-ab-sam3-f16 <out>`.

Both set `E2E_REQUIRED_SPLIT_FILTER=sam3-f16` and
`E2E_REQUIRED_SPLIT_PYTHON_DTYPE=fp16`; the A/B target also sets
`E2E_REQUIRED_PARITY_FILTER=sam3-f16`.

The cuDNN FC1+GELU hook was rechecked with that F16/TF32 contract:

`outputs/e2e-required-ab-sam3-f16-cudnn-mlp-f32-fp16py-r3-20260629a/optimization_targets.md`.

| Candidate | Required E2E delta | Tail graph delta | E2E z | Tail z | Parity | Decision |
|---|---:|---:|---:|---:|---|---|
| `cudnn-mlp-fc1-gelu-f32` | `+69.723 ms` | `+41.893 ms` | `+45.240` | `+31.566` | failed, `diff_rows=5` | reject |

The corrected baseline still shows the essential gap clearly: C++ required E2E
is faster than Python session E2E on this short selected-target application
contract (`974.817 ms` versus Python `1107.488 ms`), but C++ tail image encode
is still slower than the official Python backbone-like tail work (`604.662 ms`
versus `529.865 ms`). The remaining root target is therefore not setup,
artifacts, cached-tail placement, or graph admin; it is the ViT image-encode
graph compute itself.

The standalone cuDNN MLP bench explains why the hook is not promotable. On the
SAM3 FC1 shape, cuDNN FC1+GELU with FP16 output measures about `0.943 ms`, but
that changes the production output contract. The production-like FP32 output
path measures about `1.259 ms`, and full FC1+GELU+FC2 graph construction is not
supported by the available cuDNN engine on this build. Evidence:
`outputs/sam3-f16-cudnn-mlp-bench-20260629a/`.

## 2026-06-29 required-process isolation for SAM3 F16 tail encode

The cuBLASLt timing summarizer now accepts the current
`GGML_CUDA_CUBLASLT_BIAS_TIMING` line format with `compute_type=...`, so the
same required-E2E profile can again attribute ViT GEMM time by group and shape.
The refreshed F16/TF32 audit is:

`outputs/e2e-required-audit-sam3-f16-fp16py-baseline-20260629e/optimization_targets.md`.

Under the required 5-frame selected-target contract, the exclusive required
process split is:

| Required process | C++ ms | Required E2E % |
|---|---:|---:|
| `tail.image_encode` | `604.662` | `62.0%` |
| `frame0.image_encode` | `178.221` | `18.3%` |
| `frame0.prompt_and_tracker_init` | `118.723` | `12.2%` |
| `tail.propagate` | `52.170` | `5.4%` |
| `input.prepare` | `12.595` | `1.3%` |
| `model.session_setup` | `8.441` | `0.9%` |

The tail image encoder remains the only C++ process that is slower than the
official Python counterpart in this contract: C++ `604.662 ms` versus Python
`529.865 ms`. The CUDA-node and cuBLASLt attribution agree that this is mostly
SAM3 ViT compute, not graph admin, cached-tail placement, or output artifacts.
For the same 5-frame profile, cuBLASLt ViT GEMM sums are:

| GEMM group | Calls | Matmul sum |
|---|---:|---:|
| `vit_mlp_fc1` | `160` | `157.557 ms` |
| `vit_mlp_fc2` | `160` | `145.666 ms` |
| `vit_qkv` | `160` | `103.162 ms` |
| `vit_attn_proj` | `160` | `37.661 ms` |

`just e2e-required-vit-bench <out>` was added as a cheap pre-gate for this
specific required process. It runs `sam3_vit_batch_bench` on the same frame and
can be included in `e2e-required-audit` with
`E2E_REQUIRED_AUDIT_ISOLATED_BENCHES=NAME=PATH[:...]`.

Isolated ViT+tracker-neck probes did not find a promotable existing switch:

| Probe | Mean/frame | Median/frame | Decision |
|---|---:|---:|---|
| baseline | `159.881 ms` | `151.486 ms` | reference |
| batch 2 | `166.743 ms` | `166.001 ms` | reject |
| batch 4 | `166.120 ms` | `166.964 ms` | reject |
| cuBLASLt GELU epilogue | `160.329 ms` | `151.177 ms` | keep diagnostic |
| MLP flat chain | `160.711 ms` | `151.613 ms` | keep diagnostic |
| FC1/QKV/attn heuristic 0 | `165.571 ms` | `157.669 ms` | reject |
| contiguous V | `164.326 ms` | `156.060 ms` | reject |
| disable direct QKV views | `168.495 ms` | `166.037 ms` | reject |
| F16 linear output + direct F16 dst | `175.997 ms` | `175.807 ms` | reject |
| approximate GELU | `172.759 ms` | `173.136 ms` | reject |

The next optimization work should therefore avoid adding more coarse env
switches around the current graph. The target is a real CUDA/GGML implementation
change inside the SAM3 ViT image encoder, with the required-E2E gate remaining:
reduce `tail.image_encode.graph_compute`, then prove exact parity and a
negative required-E2E delta.

The full required-E2E A/B for cuBLASLt FC2 residual epilogue fusion was rerun on
the same F16/TF32 contract and rejected again. With
`GGML_CUDA_ENABLE_CUBLASLT_BIAS_RESIDUAL_FUSION=1`, required E2E regressed by
`+5.027 ms`, tail image-encode graph compute regressed by `+5.110 ms`, and
strict parity failed with `diff_rows=5`. The numerical differences are small
(`max_score_abs_delta=0.000062`, one row with `mask_pixel_xor=3`), but this does
not meet the exact parity gate and it is also slower. Evidence:
`outputs/e2e-required-ab-sam3-f16-cublaslt-residual-r3-20260629a/`.

cuBLASLt heuristic/autotune checks also do not expose an easy GEMM-selection
win. On the isolated ViT+tracker-neck bench with 20 repeats:

| Probe | Mean/frame | Median/frame | Decision |
|---|---:|---:|---|
| refreshed baseline | `159.411 ms` | `155.485 ms` | reference |
| cuBLASLt autotune | `165.142 ms` | `164.971 ms` | reject |
| FC2 heuristic 1 only | `159.585 ms` | `155.539 ms` | neutral |
| autotuned shape envs | `165.550 ms` | `162.653 ms` | reject |

`sam3_vit_batch_bench` now supports cumulative `--block0-stage` stops through
the MLP side of block 0, and `just e2e-required-vit-bench` exposes this through
`E2E_REQUIRED_VIT_BENCH_BLOCK0_STAGE` and
`E2E_REQUIRED_VIT_BENCH_VIT_BLOCKS`. This keeps the required process split
honest: the full required E2E gate still decides acceptance, while cheap
isolated runs can now separate prefix, attention projection, MLP FC1/GELU/FC2,
single-block, and full ViT+neck costs before a CUDA kernel or fusion change is
promoted.

`just e2e-required-vit-stage-sweep <out>` was added for a reproducible block0
stage sweep. On the F16 baseline, the cumulative block0 sweep shows the local
incremental targets are consistent with the full cuBLASLt profile:

| Incremental stage | Mean delta |
|---|---:|
| `mlp_fc1` | `0.923 ms` |
| `attn_core` | `0.656 ms` |
| `mlp_fc2` | `0.642 ms` |
| `qkv_proj` | `0.565 ms` |
| `mlp_gelu` | `0.384 ms` |
| `attn_proj` | `0.228 ms` |

Evidence: `outputs/e2e-required-vit-stage-sweep-sam3-f16-block0-r10-20260629a/`.
The next CUDA-side work should target these block-local stages first, then
promote only changes that also reduce the full required
`tail.image_encode.graph_compute` and pass strict tracking parity.

## 2026-06-29 E2E-isolated CUDA candidate check

`scripts/summarize_vit_stage_sweep.py` now writes machine-friendly
`stage`, `name`, and `median_ms_per_frame` fields in addition to the existing
Markdown table. This keeps the block-local pre-gate usable from scripts without
parsing the rendered report.

A same-type F32 unary `float4` probe was tested as an opt-in diagnostic and was
not kept. The broad unary variant preserved exact parity but regressed required
E2E:

| Probe | Isolated ViT mean | Isolated ViT median | Required E2E delta | Tail graph delta | Parity | Decision |
|---|---:|---:|---:|---:|---|---|
| F32 unary `float4` all unary ops | `158.581 ms` | `154.826 ms` | `+10.122 ms` | `+3.915 ms` | exact, `diff_rows=0` | reject |
| F32 unary `float4` limited to GELU_ERF | `160.267 ms` | `155.060 ms` | not run | not run | not run | reject at isolated pre-gate |

Evidence:
`outputs/e2e-required-vit-bench-sam3-f16-unary-f32-vec4-r20-20260629a/`,
`outputs/e2e-required-ab-sam3-f16-unary-f32-vec4-r3-20260629a/`, and
`outputs/e2e-required-vit-bench-sam3-f16-gelu-erf-f32-vec4-r20-20260629a/`.

The existing `UNARY(GELU_ERF) + CPY(F16)` fusion was also quantified by
disabling it in the block0 stage sweep. The GELU stop itself is unchanged, but
FC2-inclusive stage 9 grows from `4.141 ms` to `4.502 ms` when the fusion is
disabled. That means the current graph already avoids a meaningful part of the
GELU-to-FC2 input conversion cost.

| Stage | Baseline mean | Disabled mean | Delta |
|---|---:|---:|---:|
| `mlp_gelu` | `3.499 ms` | `3.498 ms` | `-0.001 ms` |
| `mlp_fc2` | `4.141 ms` | `4.502 ms` | `+0.361 ms` |
| `mlp_fc1_gelu` | `3.499 ms` | `3.498 ms` | `-0.000 ms` |

Evidence:
`outputs/e2e-required-vit-stage-sweep-sam3-f16-disable-unary-cpy-fusion-r10-20260629a/`.

Current conclusion: the easy activation/copy path is already optimized enough
to show up in the block-local split. The remaining F16 SAM3 gap is still the
GEMM body inside `vit_mlp_fc1`, `vit_mlp_fc2`, and `vit_qkv`; future candidates
should change those CUDA/cuBLASLt execution paths directly and must pass the
same process-split E2E gate before promotion.

The cuBLASLt timing log now includes `gemm=[m,n,k,batch]`, and
`scripts/summarize_cublaslt_bias_timing.py --group-by shape` uses that GEMM
shape when present. This avoids conflating output-equivalent projections with
different K dimensions. A 2-frame smoke profile confirms the main F16 ViT GEMMs:

| Group | GEMM `[m,n,k,batch]` | Calls | Matmul sum | Compute type | Returned algos | Selected algo |
|---|---|---:|---:|---:|---:|---:|
| `vit_mlp_fc1` | `[4736,5184,1024,1]` | `64` | `63.936 ms` | `68` | `8` | `2` |
| `vit_mlp_fc2` | `[1024,5184,4736,1]` | `64` | `58.909 ms` | `68` | `8` | `2` |
| `vit_qkv` | `[3072,5184,1024,1]` | `64` | `41.986 ms` | `68` | `8` | `2` |
| `vit_attn_proj` | `[1024,5184,1024,1]` | `64` | `15.267 ms` | `68` | `8` | `2` |

Evidence:
`outputs/e2e-required-mulmat-profile-sam3-f16-gemm-dims-smoke-20260629b/`.
Since cuBLASLt returns only 8 heuristics for these shapes, increasing the local
16-result request alone is not expected to reveal hidden candidates.

## 2026-06-29 F16 GEMM microbench versus required-E2E gate

`sam3_f32_batched_bench` now has a production-like F16 cuBLASLt path
(`--f16-cublaslt`) for the SAM3 ViT projection shapes. It uses F16 A/B inputs,
F32 output, F32 bias epilogue, and the same F32 accumulation mode used by the
current required-E2E path. This is only a pre-gate: candidates must still pass
the required-E2E process split and strict tracking parity.

Standalone shape sweep evidence:
`outputs/e2e-required-f16-cublaslt-shape-sweep-20260629a/`.

| GEMM group | Baseline algo2 median | Best median | Best algo | Decision before E2E |
|---|---:|---:|---:|---|
| `vit_mlp_fc1` | `0.878272 ms` | `0.878272 ms` | `2` | keep current |
| `vit_mlp_fc2` | `0.824096 ms` | `0.810208 ms` | `0` | test in E2E |
| `vit_qkv` | `0.576896 ms` | `0.533536 ms` | `0` | test in E2E |
| `vit_attn_proj` | `0.209664 ms` | `0.188320 ms` | `0` | test in E2E |

The resulting candidate kept the same F16/F32-accumulate precision policy and
only changed shape-local cuBLASLt heuristic indices:

`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV=0`,
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_ATTN_PROJ=0`,
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1=2`, and
`GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2=0`.

It was rejected by the 5-frame, 5-repeat required-E2E gate:

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Required E2E | `969.523 ms` | `985.556 ms` | `+16.032 ms` |
| Required core graph compute | `812.825 ms` | `826.724 ms` | `+13.899 ms` |
| Tail image-encode graph compute | `595.541 ms` | `601.888 ms` | `+6.347 ms` |
| Frame0 image-encode graph compute | `174.276 ms` | `180.057 ms` | `+5.781 ms` |

Strict parity also failed: `diff_rows=5`, max bbox delta `0.009 px`, max score
delta `0.000045`, and max mask XOR `4` pixels. These are tiny numeric changes,
but they are not exact parity and the speed gate is negative. Evidence:
`outputs/e2e-required-ab-sam3-f16-shape-algo0240-r5-20260629a/`.

A short cuBLASLt timing profile confirms the candidate env selected the intended
heuristics (`fc1=2`, `fc2/qkv/attn_proj=0`). The main ViT GEMM timing moved in
the expected direction in that smoke profile for QKV and attention projection,
but the full required process still regressed. This makes the rule explicit:
standalone GEMM and 2-frame cuBLASLt profiles are diagnostic filters only; the
promotion gate is the required-E2E split plus exact tracking parity.

## 2026-06-29 required-E2E measurement bundle and stage split

`just e2e-required-measure` now produces a single same-contract measurement
bundle containing:

- `split/`: Python/C++ same-contract summary, component split, and required
  process split;
- `profile/`: synchronized CUDA node profile, node-name stage summary, and
  `stage_profile.json`;
- `mulmat-profile/`: cuBLASLt/GEMM attribution;
- `vit-bench/`: isolated required tail image-encode bench;
- `audit/`: one combined optimization target report.

`just e2e-required-measure-sam3-f16 <out>` specializes this bundle for the
current SAM3 F16 contract against official Python fp16 with TF32 on. Evidence:
`outputs/e2e-required-measure-sam3-f16-baseline-20260629a/`.

The new audit also consumes `stage_profile.json`, so the required E2E process
split can be read together with synchronized CUDA stage attribution. The F16
baseline measurement is:

| Required process | Mean |
|---|---:|
| `required_e2e.total` | `967.318 ms` |
| `tail.image_encode` | `598.472 ms` |
| `tail.image_encode.graph_compute` | `594.484 ms` |
| `frame0.image_encode` | `173.642 ms` |
| `frame0.prompt_and_tracker_init` | `123.010 ms` |
| `tail.propagate` | `51.024 ms` |

The same run compares C++ tail image encode to the official Python
backbone-like tail work at the same dtype/TF32 policy: C++ tail image encode is
`598.472 ms` versus Python `522.036 ms`, and C++ tail graph compute is
`594.484 ms` versus Python `522.036 ms`. The remaining speed gap is therefore
still inside the required image encoder graph, not input decode, cached-tail
placement, graph admin, or output artifacts.

The synchronized tail encode stage profile ranks the tracking image encoder as:

| Tracking encode CUDA stage | Calls | Profile sum | Mean/call | Top ops |
|---|---:|---:|---:|---|
| `neck` | `4` | `47.418 ms` | `11.855 ms` | `CONV_2D`, `CONV_TRANSPOSE_2D`, `CONT`, `IM2COL` |
| `vit_block_23` | `4` | `21.663 ms` | `5.416 ms` | `MUL_MAT`, `FLASH_ATTN_EXT` |
| `vit_block_15` | `4` | `21.639 ms` | `5.410 ms` | `MUL_MAT`, `FLASH_ATTN_EXT` |
| `vit_block_31` | `4` | `21.544 ms` | `5.386 ms` | `MUL_MAT`, `FLASH_ATTN_EXT` |
| `vit_block_07` | `4` | `21.142 ms` | `5.285 ms` | `MUL_MAT`, `FLASH_ATTN_EXT` |

The one-call tail node-stage view still shows the aggregate ViT GEMM body as
the larger target:

| cuBLASLt/GEMM group | Calls | Matmul sum |
|---|---:|---:|
| `vit_mlp_fc1` | `160` | `158.505 ms` |
| `vit_mlp_fc2` | `160` | `146.058 ms` |
| `vit_qkv` | `160` | `103.700 ms` |
| `vit_attn_proj` | `160` | `37.883 ms` |

Next optimization work should continue to promote only changes that reduce
`tail.image_encode.graph_compute` under this measurement bundle and then pass
strict tracking parity. The neck is now visible and non-trivial, but the larger
aggregate target remains the ViT MLP/QKV GEMM execution path.

The first follow-up neck check swept `GGML_CUDA_CUDNN_CONV2D_ALGO` on the
isolated F16 tail image-encode bench with tracker neck enabled. It did not find
a promotion candidate:

| cuDNN conv2d algo | Mean/frame | Median | Decision |
|---|---:|---:|---|
| default | `158.945 ms` | `154.696 ms` | keep |
| `1` | `159.668 ms` | `156.398 ms` | reject |
| `0` | `168.485 ms` | `167.841 ms` | reject |
| `2` | `170.963 ms` | `169.929 ms` | reject |
| `5` | `225.302 ms` | `221.956 ms` | reject |
| `3` | `466.490 ms` | `466.121 ms` | reject |
| `4` | `507.524 ms` | `507.681 ms` | reject |

Evidence:
`outputs/e2e-required-vit-bench-sam3-f16-cudnn-conv2d-algo-sweep-20260629a/`.
This keeps the next root target on ViT GEMM/dataflow rather than cuDNN conv2d
algorithm selection.

## 2026-06-29 required-E2E tail placement split and isolated pre-gates

The required-process split now separates tail image encode into both required
work and cached-tail placement detail. For the same F16 baseline bundle:

`outputs/e2e-required-measure-sam3-f16-baseline-20260629a/`.

The split also carries an explicit measurement contract, so cached-tail cannot
be mistaken for a Python-equivalent shortcut:

| Contract item | Value |
|---|---:|
| Required denominator | `required_e2e_ms` |
| Exclusive rollup delta | `0.001 ms` |
| Cached-tail work added back | `true` |
| Python cached-tail equivalent | `false` |
| Tail encode graph-compute share | `99.3%` |

C++ may move tail-frame image encoding into a cached preencode placement bucket,
but that work is still added back into the required-E2E denominator. Python tail
backbone rows remain diagnostic hooks, not proof that Python used the same
cached-tail placement. This keeps acceptance tied to actual required work:
`input_load_ms + model_e2e_ms`, with `tail.image_encode.graph_compute` tracked
inside the denominator.

The regenerated split shows that the tail image encoder is almost entirely graph
compute:

| Tail image encode slice | Mean | Required E2E % |
|---|---:|---:|
| `tail.image_encode` | `598.472 ms` | `61.9%` |
| `tail.image_encode.graph_compute` | `594.484 ms` | `61.5%` |
| `tail.image_encode.non_compute` | `3.988 ms` | `0.4%` |
| `tail.image_encode.non_compute.graph_admin` | `3.843 ms` | `0.4%` |
| `tail.image_encode.non_compute.state_update` | `0.131 ms` | `0.0%` |
| `tail.image_encode.non_compute.pe_build` | `0.007 ms` | `0.0%` |

Placement is now explicit and is not counted as a speedup by itself:

| Placement slice | Mean | Required E2E % |
|---|---:|---:|
| `tail.image_encode.inline.total` | `151.663 ms` | `15.7%` |
| `tail.image_encode.inline.graph_compute` | `150.533 ms` | `15.6%` |
| `tail.image_encode.inline.non_compute` | `1.130 ms` | `0.1%` |
| `tail.image_encode.preencode.total` | `446.809 ms` | `46.2%` |
| `tail.image_encode.preencode.graph_compute` | `443.952 ms` | `45.9%` |
| `tail.image_encode.preencode.non_compute` | `2.858 ms` | `0.3%` |

This confirms that cached-tail scheduling, graph build/allocation/upload, PE
lookup/build, and state update are not large enough to close the Python tail
backbone gap. A candidate must reduce `tail.image_encode.graph_compute`.

Two isolated pre-gates were run against the same tail ViT+tracker-neck bench and
fed into the combined audit report:

`outputs/e2e-required-measure-sam3-f16-baseline-20260629a/audit/optimization_targets.md`.

Batching repeated tail frames is exact for repeated lanes, but slower per frame:

| Batch | Mean/frame | Delta vs batch-1 | Lane max diff | Decision |
|---:|---:|---:|---:|---|
| `1` | `159.233 ms` | `0.000 ms` | `-` | baseline |
| `2` | `166.252 ms` | `+7.019 ms` | `0.000` | reject |
| `3` | `164.339 ms` | `+5.106 ms` | `0.000` | reject |
| `4` | `167.351 ms` | `+8.118 ms` | `0.000` | reject |
| `5` | `162.768 ms` | `+3.535 ms` | `0.000` | reject |
| `8` | `163.341 ms` | `+4.108 ms` | `0.000` | reject |

FATTN execution-policy switches also did not expose a same-contract win:

| FATTN policy | Mean/frame | Delta vs baseline | Decision |
|---|---:|---:|---|
| baseline | `159.034 ms` | `0.000 ms` | keep |
| force MMA | `159.188 ms` | `+0.154 ms` | neutral/reject |
| disable SAM3 ViT MMA | `160.687 ms` | `+1.653 ms` | reject |
| disable direct-out56 | `161.174 ms` | `+2.140 ms` | reject |
| disable FATTN56 fast path | `161.226 ms` | `+2.192 ms` | reject |
| disable FATTN64 fast path | `161.445 ms` | `+2.411 ms` | reject |
| force tile | `230.177 ms` | `+71.144 ms` | reject |
| force vec | `230.650 ms` | `+71.616 ms` | reject |

Evidence:
`outputs/e2e-required-vit-bench-sam3-f16-batch-sweep-20260630a/` and
`outputs/e2e-required-vit-bench-sam3-f16-fattn-policy-sweep-20260630a/`.

The current F16 tracker-neck pre-gate also rechecked the neck 1x1 matmul path
against the same input and tracker-neck scope:

| Neck 1x1 path | Mean/frame | Delta vs baseline | Decision |
|---|---:|---:|---|
| default | `156.064 ms` | `0.000 ms` | keep |
| `SAM3_ENABLE_NECK_1X1_MULMAT=1` | `156.841 ms` | `+0.777 ms` | reject |

Evidence:
`outputs/e2e-required-vit-bench-sam3-f16-neck1x1-mulmat-r10-20260630a/`.

The isolated ViT bench now emits raw timing samples in addition to aggregate
statistics, and the audit report derives sample standard deviation, 95% CI, and
a baseline-delta signal. This was used to split the previously rejected combined
cuBLASLt shape heuristic into individual shape probes:

| Shape heuristic probe | n | Mean/frame | 95% CI | Delta vs baseline | Signal | Decision |
|---|---:|---:|---:|---:|---:|---|
| baseline | `30` | `157.331 ms` | `4.035 ms` | `0.000 ms` | `0.000` | keep |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2=0` | `30` | `157.172 ms` | `4.157 ms` | `-0.159 ms` | `-0.054` | reject/no signal |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_ATTN_PROJ=0` | `30` | `158.114 ms` | `4.009 ms` | `+0.783 ms` | `0.270` | reject |
| `GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV=0` | `30` | `159.836 ms` | `4.405 ms` | `+2.505 ms` | `0.822` | reject |
| QKV+attention projection algo0 | `30` | `160.969 ms` | `4.326 ms` | `+3.637 ms` | `1.205` | reject |

Evidence:
`outputs/e2e-required-vit-bench-sam3-f16-shape-individual-r30-20260630a/`.
The only negative delta is far inside timing noise and has worse median timing,
so no individual heuristic is promoted to full required-E2E A/B. This reinforces
that the remaining work is not another cuBLASLt heuristic selection; it needs a
different ViT GEMM/dataflow implementation.

A standalone F16 MLP chain microbench was added as the next dataflow pre-gate.
It models the SAM3 F16 MLP body as FC1 bias GEMM, GELU_ERF-to-F16, then FC2
bias GEMM, with separate FC1/FC2 cuBLASLt heuristic controls:

| F16 MLP probe | Median | Decision |
|---|---:|---|
| FC1 algo2 only | `0.988 ms` | reference |
| FC2 algo2 only | `1.059 ms` | reference |
| FC2 algo0 only | `1.010 ms` | reference |
| chain FC1 algo2 + FC2 algo2 | `2.487 ms` | reject |
| chain FC1 algo2 + FC2 algo0 | `2.276 ms` | reject |
| chain FC1 algo2 + FC2 algo0 + FAST_16F | `2.180 ms` | reject |
| chain FC1 algo2 + FC2 algo0 + COMPUTE_16F | failed | reject |

Evidence: `outputs/e2e-required-f16-mlp-chain-bench-20260630a/`.
Even the fastest chain row is `+0.217 ms` slower than the existing block0
stage-sweep MLP median (`norm2 -> mlp_fc2`, about `1.963 ms`), and
`FAST_16F` is already rejected by the required-E2E acceptance gate. This makes a
production F16 MLP chain fusion a poor next candidate unless the implementation
changes beyond the measured FC1/GELU/FC2 sequence.

Current implementation decision: the next root change should not be another
batching, cached-tail, graph-admin, conv-algo, FATTN-policy, or cuBLASLt
heuristic toggle. The measured target remains a real SAM3 ViT projection
dataflow/kernel change, especially `vit_mlp_fc1`, `vit_mlp_fc2`, and `vit_qkv`,
with acceptance gated by lower `required_e2e_ms`, lower
`tail.image_encode.graph_compute`, and exact tracking parity.

The block0 ViT stage sweep now separates the attention pre-core boundary into
`qkv_layout`, `qkv_rope`, and `attn_core` instead of treating all QKV layout,
RoPE, and FATTN input work as one `attn_core` bucket. The updated stage order is
`norm1`, `window_part`, `qkv_proj`, `qkv_layout`, `qkv_rope`, `attn_core`,
`attn_proj`, `window_unpart`, `norm2`, `mlp_fc1`, `mlp_gelu`, `mlp_fc2`,
`mlp`, `block`, and the alternate `mlp_fc1_gelu` stop.

Short SAM3 F16 block0 measurements with the current default CUDA path:

| Stage boundary | Mean/frame | Incremental mean | Interpretation |
|---|---:|---:|---|
| `qkv_proj` | `1.169 ms` | `0.565 ms` | QKV GEMM and bias |
| `qkv_layout` | `1.447 ms` | `0.279 ms` | diagnostic Q/K layout stop without RoPE |
| `qkv_rope` | `1.516 ms` | `0.069 ms` | E2E-relevant layout+RoPE boundary |
| `attn_core` | `1.823 ms` | `0.307 ms` | FATTN output boundary |
| `attn_proj` | `2.049 ms` | `0.226 ms` | attention projection |

Evidence:
`outputs/e2e-required-vit-stage-sweep-sam3-f16-block0-fine-current-r10-20260630a/`.

The important interpretation is that `qkv_layout` is a diagnostic stop, not a
standalone E2E cost when RoPE is present. CUDA already has a `CONT+RoPE` fusion.
Profiling showed Q-side `CONT_ROPE_PAIR_FUSION` succeeds, while the K side is
rejected by the fusion memory-range check because the final K RoPE output can
reuse the original qkv/K source storage under the normal non-fused liveness
model. Disabling `CONT+RoPE` fusion increased block0 `attn_core` from
`1.823 ms` to `1.922 ms`, so the existing Q-side fusion is worth about
`0.10 ms` for this block0 window case:

| Mode | `qkv_rope` mean | `attn_core` mean | Decision |
|---|---:|---:|---|
| default | `1.516 ms` | `1.823 ms` | keep |
| `GGML_CUDA_DISABLE_CONT_ROPE_PAIR_FUSION=1` | `1.558 ms` | `1.922 ms` | reject |

Evidence:
`outputs/e2e-required-vit-stage-sweep-sam3-f16-block0-no-cont-rope-r10-20260630a/`
and
`outputs/e2e-required-vit-stage-sweep-sam3-f16-block0-fine-qk-pair-vec-r10-20260630a/profile_rope_stage4.stderr`.

A two-output adjacent `CONT` copy fusion was prototyped for the RoPE-free
diagnostic `qkv_layout` stop, including a vectorized F32 row-group kernel, but it
did not improve the measured boundary (`qkv_layout` became about `+0.010 ms`
slower in the short run) and it does not fire on the RoPE-present E2E path
because Q `CONT` is immediately followed by Q RoPE ops. It is left opt-in only
under `GGML_CUDA_ENABLE_CPY_PAIR_CONT_FUSION=1` and is not an accepted
production optimization.

Current next target: make the K-side layout+RoPE fusion safe without violating
allocator liveness, or move RoPE/layout into the attention kernel itself. A valid
optimization must reduce the `qkv_rope` or `attn_core` boundary and then show a
statistically meaningful improvement in full required-E2E, not only in the
RoPE-free `qkv_layout` diagnostic stop.

## 2026-06-30 BF16 required-E2E process split refresh

The required-E2E split now keeps cached-tail placement separate from required
work and preserves missing timing components instead of converting absent
measurements to `0.0`. The table generator also reads the current
`tail_graph_compute_ms` field for `tail.image_encode.graph_compute`, so the
tail image encoder is no longer misreported as mostly non-compute work.

Fresh SAM3 BF16 required-E2E numbers, with official Python BF16/TF32 as the
same application-contract comparison:

| Metric | C++ | Python | Python/C++ |
|---|---:|---:|---:|
| required/session E2E | `935.425 ms` | `1072.642 ms` | `1.147x` |
| model E2E | `922.839 ms` | `1014.266 ms` | `1.099x` |
| timed track scope | `48.595 ms` | `164.595 ms` | `3.387x` |
| tail image encode/backbone | `588.621 ms` | `523.416 ms` | `0.889x` |
| tail propagate/tracker | `49.266 ms` | `114.666 ms` | `2.327x` |

The required C++ denominator is dominated by image encode graph compute:

| Required process | Mean | Required E2E % |
|---|---:|---:|
| `required_e2e.total` | `935.425 ms` | `100.0%` |
| `tail.image_encode` | `588.621 ms` | `62.9%` |
| `tail.image_encode.graph_compute` | `583.758 ms` | `62.4%` |
| `frame0.image_encode` | `160.494 ms` | `17.2%` |
| `frame0.image_encode.graph_compute` | `159.341 ms` | `17.0%` |
| `frame0.prompt_and_tracker_init` | `116.138 ms` | `12.4%` |
| `tail.propagate` | `49.266 ms` | `5.3%` |
| `tail.image_encode.non_compute` | `4.863 ms` | `0.5%` |

Cached-tail placement is explicit and is not treated as a Python-equivalent
omission:

| Tail placement slice | Mean | Required E2E % |
|---|---:|---:|
| `tail.image_encode.inline.total` | `145.111 ms` | `15.5%` |
| `tail.image_encode.inline.graph_compute` | `143.162 ms` | `15.3%` |
| `tail.image_encode.preencode.total` | `443.511 ms` | `47.4%` |
| `tail.image_encode.preencode.graph_compute` | `440.596 ms` | `47.1%` |

Evidence:
`outputs/e2e-required-split-sam3-bf16-e2e-process-current-r3-20260630a/`.

The fresh BF16 block0 stage sweep keeps the local target focused on MLP and QKV:

| Stage boundary | Mean | Incremental mean |
|---|---:|---:|
| `qkv_proj` | `1.203 ms` | `0.555 ms` |
| `qkv_layout` | `1.482 ms` | `0.278 ms` |
| `qkv_rope` | `1.551 ms` | `0.069 ms` |
| `attn_core` | `1.863 ms` | `0.312 ms` |
| `attn_proj` | `2.094 ms` | `0.231 ms` |
| `mlp_fc1` | `3.153 ms` | `0.915 ms` |
| `mlp_gelu` | `3.342 ms` | `0.189 ms` |
| `mlp_fc2` | `4.305 ms` | `0.963 ms` |
| `block` | `4.392 ms` | `0.082 ms` |

Evidence:
`outputs/e2e-required-vit-stage-sweep-sam3-bf16-block0-current-r10-20260630a/`.

Several small dataflow toggles were rechecked against the fresh BF16 isolated
ViT+tracker-neck scope. The old `478 ms` isolated output is not used for
decision-making because a fresh baseline aligns with the required-E2E tail
encode slice at about one encoded frame.

| Candidate | Mean/frame | Delta vs fresh baseline | Decision |
|---|---:|---:|---|
| fresh baseline | `151.475 ms` | `0.000 ms` | keep |
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` full E2E A/B | `+10.182 ms` required E2E | `+1.1%` | reject |
| cuBLASLt `BIAS+GELU_ERF` | `498.623 ms` | `+347.149 ms` | reject |
| cuDNN BF16 MLP chain | `465.756 ms` | `+314.281 ms` | reject |
| approximate GELU BF16 chain | `501.566 ms` | `+350.091 ms` | reject |

The full required-E2E A/B for `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` had exact
tracking parity (`diff_rows=0`, mask hashes equal) but regressed
`required_e2e_ms` from `936.479 ms` to `946.662 ms` and regressed
`tail.image_encode.graph_compute` from `583.266 ms` to `585.106 ms`.

Evidence:
`outputs/e2e-required-ab-sam3-bf16-mlp-flat-chain-r3-20260630a/`,
`outputs/e2e-required-vit-bench-sam3-bf16-baseline-fresh-r20-20260630b/`,
`outputs/e2e-required-vit-bench-sam3-bf16-cublaslt-geluerf-r20-20260630b/`,
`outputs/e2e-required-vit-bench-sam3-bf16-cudnn-mlp-chain-r20-20260630b/`, and
`outputs/e2e-required-vit-bench-sam3-bf16-approx-gelu-chain-r20-20260630b/`.

Current optimization target: C++ already wins the selected-target required E2E
contract overall, but it does not yet beat official Python on the tail image
encoder/backbone slice. The next accepted change must reduce
`tail.image_encode.graph_compute` by about `17 ms` per encoded tail frame, while
keeping exact tracking parity. Since all ViT `qkv`, `attn.proj`, `mlp_fc1`, and
`mlp_fc2` nodes already hit cuBLASLt `mul_mat+bias` fusion, the remaining work is
not another bias-fusion toggle; it needs a real GEMM/dataflow change for the
SAM3 ViT MLP and QKV paths.

## 2026-06-30 BF16 cuBLASLt shape-algo pre-gate

The isolated ViT+tracker-neck pre-gate must be run serially on the GPU. A
parallel attempt produced about `478 ms` per frame because the candidate benches
contended for the same device; those numbers are invalid for optimization
decisions. The serial sweep now has a dedicated target:

```sh
E2E_REQUIRED_VIT_BENCH_MODEL=models/sam3/sam3-bf16.ggml \
E2E_REQUIRED_VIT_BENCH_IMAGE=outputs/e2e-required-split-sam3-bf16-e2e-process-current-r3-20260630a/frames/00001.jpg \
E2E_REQUIRED_VIT_BENCH_REPEATS=30 \
E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_COMPONENT_SPLIT=outputs/e2e-required-split-sam3-bf16-e2e-process-current-r3-20260630a/component_split.json \
E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_CUBLASLT=outputs/e2e-required-mulmat-profile-sam3-bf16-fresh-20260630c/cublaslt_bias_by_group.json \
just e2e-required-cublaslt-algo-sweep outputs/e2e-required-cublaslt-algo-sweep-sam3-bf16
```

Serial BF16 results for the hot ViT bias-GEMM shape algorithm candidates:

| Candidate | Mean/frame | Median/frame | Delta vs baseline | Signal |
|---|---:|---:|---:|---:|
| baseline | `151.545 ms` | `150.155 ms` | `0.000 ms` | `0.000` |
| `VIT_MLP_FC1=1` | `153.049 ms` | `151.004 ms` | `+1.503 ms` | `0.737` |
| `VIT_MLP_FC2=1` | `153.728 ms` | `152.504 ms` | `+2.183 ms` | `1.043` |
| `VIT_MLP_FC2=2` | `154.690 ms` | `153.336 ms` | `+3.145 ms` | `1.475` |
| `VIT_QKV=1` | `153.648 ms` | `152.348 ms` | `+2.103 ms` | `1.011` |
| `VIT_ATTN_PROJ=1` | `153.451 ms` | `151.997 ms` | `+1.906 ms` | `0.916` |

Decision: no cuBLASLt shape-algo override is promoted to full required-E2E A/B.
The required-E2E denominator is already separated into input, session setup,
frame0 image encode, frame0 prompt/tracker init, tail image encode, and tail
propagate; candidate changes should first reduce the isolated tail image encode
scope and then pass full required-E2E parity/perf. The next useful CUDA work is
therefore a real SAM3 ViT MLP/QKV dataflow or kernel change, not another
cuBLASLt heuristic-index toggle.

Evidence:
`outputs/e2e-required-vit-bench-sam3-bf16-baseline-seq-r30-20260630c/`,
`outputs/e2e-required-vit-bench-sam3-bf16-algo-mlp-fc1-1-seq-r30-20260630c/`,
`outputs/e2e-required-vit-bench-sam3-bf16-algo-mlp-fc2-1-seq-r30-20260630c/`,
`outputs/e2e-required-vit-bench-sam3-bf16-algo-mlp-fc2-2-seq-r30-20260630c/`,
`outputs/e2e-required-vit-bench-sam3-bf16-algo-qkv-1-seq-r30-20260630c/`,
`outputs/e2e-required-vit-bench-sam3-bf16-algo-attn-proj-1-seq-r30-20260630c/`,
and `outputs/e2e-required-audit-sam3-bf16-seq-cublaslt-algos-20260630c/`.

## 2026-06-30 E2E-required block/neck isolation

The isolated pre-gate now has a cumulative ViT block sweep target. Unlike the
stage-stop sweep, `--vit-blocks N` keeps the output contract stable as
`[1024,72,72,1]`, so prefix and block increments are usable for E2E tail
image-encode attribution. The target also measures the tracker neck increment
over full ViT:

```sh
E2E_REQUIRED_VIT_BENCH_MODEL=models/sam3/sam3-bf16.ggml \
E2E_REQUIRED_VIT_BENCH_IMAGE=outputs/e2e-required-split-sam3-bf16-e2e-process-current-r3-20260630a/frames/00001.jpg \
E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS=1 \
E2E_REQUIRED_VIT_BENCH_REPEATS=5 \
just e2e-required-vit-block-sweep outputs/e2e-required-vit-block-sweep-sam3-bf16
```

Fresh SAM3 BF16 block/neck split:

| Slice | Mean/frame | Increment |
|---|---:|---:|
| prefix only | `0.553 ms` | `0.553 ms` |
| after block 0 | `4.360 ms` | `3.807 ms` |
| after global block 7 | `33.292 ms` | `5.347 ms` |
| after global block 15 | `67.533 ms` | `6.956 ms` |
| after global block 23 | `98.074 ms` | `5.092 ms` |
| full ViT through block 31 | `133.362 ms` | `7.595 ms` |
| full ViT + tracker neck | `147.851 ms` | `14.489 ms` |

Evidence:
`outputs/e2e-required-vit-block-sweep-sam3-bf16-r5-20260630d/`.

This changes the optimization interpretation. The required tail image encode
slice is not only a ViT MLP/QKV problem; the tracker neck is about `11-15 ms`
per encoded frame under the same isolated contract and is large enough to
explain most of the remaining Python-backbone gap when compared naively. ViT
without tracker neck, measured with a longer warmup, is still slower than the
Python tail-backbone diagnostic:

| Scope | Mean/frame | Median/frame | Notes |
|---|---:|---:|---|
| C++ full ViT, no tracker neck | `139.123 ms` | `137.611 ms` | `warmup=10`, `repeats=30` |
| Python tail backbone diagnostic | `130.854 ms` | - | `523.416 ms / 4` tail frames |
| C++ full ViT + tracker neck | `151.008 ms` | `148.294 ms` | `warmup=3`, `repeats=20` |

Evidence:
`outputs/e2e-required-vit-bench-sam3-bf16-chain-ab-r20-20260630d/`.

Additional pre-gates did not produce an accepted speedup:

| Candidate | Mean/frame | Delta vs baseline | Decision |
|---|---:|---:|---|
| baseline, full ViT + tracker neck | `151.008 ms` | `0.000 ms` | keep |
| `SAM3_BF16_VIT_QKV_CHAIN=1` | `163.435 ms` | `+12.426 ms` | reject |
| `SAM3_BF16_VIT_MLP_CHAIN=1` | `169.114 ms` | `+18.106 ms` | reject |
| `SAM3_BF16_VIT_LINEAR_OUTPUT=1` | `153.865 ms` | `+2.857 ms` | reject |
| `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` | `151.965 ms` | `+0.956 ms` | reject |
| `SAM3_DISABLE_NECK_SCALE3=1` | `151.894 ms` | `+0.885 ms` | reject |
| `SAM3_ENABLE_NECK_1X1_MULMAT=1` | `152.129 ms` | `+1.121 ms` | reject |
| `SAM3_CONV_TRANSPOSE_KEEP_BF16=1` | `152.541 ms` | `+1.532 ms` | reject |
| batch size 2, full ViT + tracker neck | `156.945 ms` | `+5.937 ms` | reject |
| batch size 4, full ViT + tracker neck | `155.007 ms` | `+3.999 ms` | reject |
| cuDNN MLP FC1+GELU F32 fusion | `162.662 ms` | `+11.654 ms` | reject |
| cuDNN MLP FC1+GELU BF16 chain | `172.540 ms` | `+21.531 ms` | reject |

cuDNN conv2d is active for the tracker-neck 3x3 convolutions. The high-res
tracker neck `s0` 3x3 BF16 conv has shape `[1,256,288,288]` and dominates the
neck cost. Forcing cuDNN forward algos confirmed the default algo `1` is the
best measured choice in this environment:

| cuDNN algo | Mean/frame | Median/frame | Decision |
|---:|---:|---:|---|
| `0` | `159.393 ms` | `154.650 ms` | reject |
| `1` | `149.760 ms` | `143.062 ms` | keep |
| `2` | `160.801 ms` | `154.231 ms` | reject |
| `3` | `458.902 ms` | `457.864 ms` | reject |
| `4` | `460.019 ms` | `458.514 ms` | reject |
| `5` | `460.413 ms` | `460.179 ms` | reject |
| `6` | `462.023 ms` | `460.450 ms` | reject |
| `7` | `214.768 ms` | `210.101 ms` | reject |

An opt-in mixed cuDNN conv2d path,
`GGML_CUDA_ENABLE_CUDNN_CONV2D_F32_LOWP=1`, was added to test whether
F32-input/BF16-weight/F32-output convs could avoid the explicit F32-to-BF16
input conversion. cuDNN rejected that descriptor combination on this system, so
it falls back to the existing lowp-input path and is not an accepted
optimization.

Current acceptance gate: future changes must report the split they improve.
For same-precision Python comparison, use `required_e2e_ms` and
`tail.image_encode.graph_compute` as the formal E2E metrics, and use
`e2e-required-vit-block-sweep` to distinguish full ViT, tracker neck, and
individual global/local block contributions before promoting a candidate to full
required-E2E parity/perf.

## 2026-06-30 Required-E2E process split refresh

The required-E2E measurement now separates the work needed for a fair
end-to-end claim from diagnostic placement effects:

- `required_e2e_ms = input.prepare + model_e2e`.
- `model_e2e` includes frame-0 image encode, frame-0 prompt/tracker init, tail
  image encode, tail propagate, session setup, and accounting remainder.
- cached-tail preencode work is added back to `tail.image_encode` and the
  required-E2E denominator; it is not treated as a Python-equivalent omission.
- optional output artifacts are disabled for the performance rows used below.

Fresh BF16/TF32 required split (`repeats=5`, same 5 decoded frames):

| Component | Mean | Required E2E % | sd | n |
|---|---:|---:|---:|---:|
| `required_e2e_ms` | `940.296 ms` | 100.0% | `10.446` | 5 |
| `model_e2e_ms` | `927.667 ms` | 98.7% | `10.462` | 5 |
| `required_core_compute_ms` | `784.753 ms` | 83.5% | - | - |
| `tail.image_encode` | `587.391 ms` | 62.5% | `2.336` | 5 |
| `tail.image_encode.graph_compute` | `583.536 ms` | 62.1% | `2.285` | 5 |
| `frame0.image_encode` | `160.344 ms` | 17.1% | `0.500` | 5 |
| `frame0.prompt_and_tracker_init` | `119.809 ms` | 12.7% | `6.907` | 5 |
| `tail.propagate` | `51.809 ms` | 5.5% | `5.403` | 5 |
| `input.prepare` | `12.628 ms` | 1.3% | `0.138` | 5 |
| `model.session_setup` | `8.309 ms` | 0.9% | `0.241` | 5 |

This makes the optimization target explicit: the required tail image encode is
`99.3%` graph compute, so host decode, graph admin, cached-tail placement, and
output artifacts are not the current bottleneck. The remaining Python-backbone
gap is in ViT image-encoder graph compute:

| Process | C++ | Python diagnostic | py/cpp | C++ slower |
|---|---:|---:|---:|---:|
| `tail_image_encode_or_backbone` | `587.391 ms` | `527.499 ms` | `0.898` | `59.892 ms` |
| `tail_image_encode.graph_compute` | `583.536 ms` | `527.499 ms` | `0.904` | `56.037 ms` |
| `required_session_e2e` | `940.296 ms` | `1078.028 ms` | `1.146` | - |
| `model_e2e` | `927.667 ms` | `1024.839 ms` | `1.105` | - |

The neck bias-broadcast cleanup was also checked under this split. Removing the
explicit `ggml_repeat` from SAM3 neck bias adds is parity-safe, but by itself is
too small to count as an E2E optimization. Extending the existing CUDA
`ADD+bias -> PERMUTE -> CONT` fusion matcher to the SAM3 neck permutation
`(1,2,0,3)` did fuse the expected neck output nodes, but the full
required-E2E A/B did not improve:

| Candidate | Required E2E delta | Tail graph delta | Core compute delta | Parity | Decision |
|---|---:|---:|---:|---|---|
| default neck bias broadcast | `-0.387 ms` vs previous split | `+0.093 ms` | near zero | not run in that split | diagnostic cleanup |
| `(1,2,0,3)` ADD/PERMUTE/CONT fusion | `+5.742 ms` | `+2.131 ms` | `+5.320 ms` | exact, `diff_rows=0` | reject |

Evidence:
`outputs/e2e-required-split-sam3-bf16-neck-bias-broadcast-r3-20260630e/`,
`outputs/e2e-required-vit-bench-sam3-bf16-neck-bias-broadcast-r30-20260630e/`,
`outputs/e2e-required-vit-bench-sam3-bf16-neck-add-permute-fusion-r30-20260630f/`,
`outputs/e2e-required-parity-sam3-bf16-neck-add-permute-fusion-20260630f/`,
and
`outputs/e2e-required-ab-sam3-bf16-neck-add-permute-fusion-r5-20260630g/`.

The next accepted optimization must therefore reduce one of the warmed ViT
graph-compute groups rather than only moving accounting around. The current
steady node profile points first at ViT MLP FC1/FC2 and then QKV/attention:

| Group | Steady profile total |
|---|---:|
| `vit.mlp_fc2` | `30.262 ms` |
| `vit.mlp_fc1` | `29.974 ms` |
| `vit.qkv_matmul` | `19.600 ms` |
| `vit.attn_core` | `19.023 ms` |
| `neck` | `8.863 ms` |
| `vit.mlp_gelu` | `7.312 ms` |
| `vit.attn_proj` | `7.307 ms` |

Evidence: `outputs/e2e-required-vit-profile-warm-sam3-bf16-20260630f/`.

## 2026-06-30 Required-E2E isolated pre-gate hardening

The isolated `sam3_vit_batch_bench` output now reports `sd_ms`,
`ci95_ms`, and per-frame equivalents in addition to mean/median/min/max.
This keeps the E2E-required pre-gate self-contained: every candidate row can be
judged from the bench JSON without recomputing dispersion from `samples_ms`.

Two gated CUDA fusion probes were rechecked under the same SAM3 BF16 required
tail-image-encode contract (`warmup=3`, `repeats=30`, tracker neck enabled):

| Candidate | Mean/frame | Median/frame | sd | ci95 | Delta vs baseline | Signal | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | `150.395 ms` | `149.097 ms` | `7.887` | `2.822` | `0.000 ms` | - | keep |
| guarded `GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16=1` | `150.719 ms` | `148.031 ms` | `8.199` | `2.934` | `+0.324 ms` | `0.156` | neutral |
| `GGML_CUDA_ENABLE_CUBLASLT_BIAS_RESIDUAL_FUSION=1` | `152.211 ms` | `150.581 ms` | `8.245` | `2.950` | `+1.816 ms` | `0.872` | reject |

The cuDNN MLP BF16 probe previously enabled the GELU_ERF fusion candidate even
when the actual unary output tensor was F32. That did not route through cuDNN
BF16; it rejected on `dst_type=f32` and then fell through to a slower fused
GELU_ERF path. The fusion gate now requires the cuDNN env to match the unary
output type, so the BF16 env no longer changes the F32-output SAM3 BF16 graph.
Parity against default is exact for the guarded BF16 env:
`diff_rows=0`, `mask_hash_equal_rows=5`.

For comparison, the F32 cuDNN FC1+GELU path does route, but it remains slower
in the warmed profile: `157.051 ms` for the single profiled repeat versus
`145.887 ms` baseline, with the MLP FC1 group increasing from `30.007 ms` to
`40.953 ms`. This stays diagnostic-only.

Evidence:
`outputs/e2e-required-vit-bench-sam3-bf16-cudnn-mlp-r30-20260630i/`,
`outputs/e2e-required-vit-profile-sam3-bf16-cudnn-mlp-20260630i/`,
`outputs/e2e-required-vit-bench-sam3-bf16-guarded-fusions-r30-20260630j/`,
and
`outputs/e2e-required-parity-sam3-bf16-cudnn-bf16-guarded-20260630j/`.

Current optimization status is unchanged: no cuDNN/cuBLASLt toggle measured here
is an accepted speedup. The next root path should target custom dataflow or
kernels for the dominant ViT MLP FC1/FC2 and QKV/attention groups, then promote
only candidates that pass this isolated pre-gate to full required-E2E A/B.

## 2026-06-30 Required-E2E ViT isolated stage split

The ViT stage measurement is now split into two different contracts:

- `e2e-required-vit-stage-sweep` remains the cumulative stop-point sweep from
  image input. It is useful for "how much work is included up to this stop", but
  adjacent deltas become noisy for later blocks.
- `e2e-required-vit-isolated-stage-sweep` first captures the real E2E ViT
  intermediate tensors with `sam3_vit_stage_capture`, then runs each requested
  stage as an independent graph with `sam31_vit_block_case`. The manifest maps
  each stage to its exact captured input tensor, so global blocks no longer
  produce invalid empty window-stage rows.

The block-case JSON now includes `sd_ms`, `ci95_ms`, `samples_ms`,
`input_ne`, `output_ne`, `input_prefix`, and `bench_mode=isolated_stage`.
`scripts/summarize_vit_stage_sweep.py` detects this mode and avoids treating
adjacent table rows as local deltas. Composite stop rows such as `qkv_rope`,
`attn_core`, `mlp`, `block`, and `mlp_fc1_gelu` are marked as independent
stop-point graphs, with derived local estimates where applicable.

SAM3 BF16 block 0 window-attention isolated stage results
(`warmup=3`, `repeats=20`) show the hot local groups clearly:

| Stage | Mean | Notes |
|---|---:|---|
| `mlp` | `2.102 ms` | FC1+GELU+FC2 from `norm2` input |
| `mlp_fc2` | `1.168 ms` | largest pure MLP op |
| `mlp_fc1` | `0.937 ms` | second largest pure MLP op |
| `attn_core - qkv_rope` | `0.375 ms` | derived local attention-core estimate |
| `qkv_proj` | `0.632 ms` | window QKV GEMM |
| `block` | `3.668 ms` | full isolated block |

SAM3 BF16 block 7 global-attention isolated stage results show why global
blocks need separate handling:

| Stage | Mean | Notes |
|---|---:|---|
| `block` | `5.166 ms` | full isolated global block |
| `mlp` | `2.044 ms` | FC1+GELU+FC2 from `norm2` input |
| `attn_core - qkv_rope` | `1.620 ms` | global attention core dominates attention |
| `qkv_proj` | `0.632 ms` | global QKV GEMM |
| `mlp_fc2` | `1.164 ms` | largest pure MLP op |
| `mlp_fc1` | `0.937 ms` | second largest pure MLP op |

The existing `SAM3_ENABLE_VIT_MLP_FLAT_CHAIN=1` probe was rechecked with the
new isolated harness on block 7 MLP-only stages. It is still not an accepted
optimization: baseline isolated `mlp` is `2.044 ms`, while flat-chain `mlp` is
`2.092 ms`. Flat-chain improves the standalone GELU layout row (`0.396 ms` to
`0.336 ms`) but loses that benefit in the full MLP graph.

Evidence:
`outputs/e2e-required-vit-isolated-stage-sweep-sam3-bf16-block0-r20-20260630l/`,
`outputs/e2e-required-vit-isolated-stage-sweep-sam3-bf16-block7-r20-20260630l/`,
and
`outputs/e2e-required-vit-isolated-stage-sweep-sam3-bf16-block7-flat-mlp-r20-20260630l/`.

The next CUDA-side optimization target should therefore be one of:

- exact-GELU-compatible MLP dataflow or a custom GEMM epilogue path that does
  not switch SAM3 from GELU_ERF to approximate GELU;
- FC2/FC1 GEMM shape tuning that improves the warmed MLP graph, not just an
  isolated single GEMM row;
- global `attn_core` kernel/dataflow work, because the global block local
  attention core is `1.620 ms` versus `0.375 ms` for block 0 window attention.

## 2026-06-30 Required-E2E isolated audit and global FATTN gate

The required-E2E optimization audit now accepts isolated ViT stage summaries via
`E2E_REQUIRED_AUDIT_ISOLATED_BENCHES=NAME=PATH[:...]` and writes those local
block-stage rows into the same Markdown/JSON report as the required process
split, node-stage profile, and cuBLASLt group summary. This keeps candidate
work ordered by the formal E2E denominator first, while still exposing the small
stage graphs needed for root CUDA changes.

Fresh SAM3 BF16 evidence with official Python BF16/TF32 as the comparison row:

| Metric | C++ | Python | Interpretation |
| --- | ---: | ---: | --- |
| required/session E2E | `934.745 ms` | `1077.372 ms` | C++ wins the selected-target wall contract |
| model E2E | `921.974 ms` | `1018.129 ms` | C++ wins after input/session setup is excluded |
| tail image encode/backbone | `586.292 ms` | `524.232 ms` | C++ is still `62.060 ms` slower on the comparable tail image work |
| tail image encode graph compute | `582.419 ms` | `524.232 ms` | `99.3%` of tail encode is graph compute |

Cached-tail placement remains an accounting detail, not a shortcut. The
preencoded cached-tail work is added back to `required_e2e_ms`, and the Python
row is not treated as a cached-tail-equivalent measurement. The optimization
target is therefore the actual tail image encoder graph, not decode, graph
admin, output artifacts, or cached-tail placement.

The combined audit points to these root targets:

| Target | Current evidence |
| --- | ---: |
| `sam3-vit:mlp-fc2-matmul` | `29.781 ms` in the profiled tail encode |
| `sam3-vit:mlp-fc1-matmul` | `29.464 ms` |
| `sam3-vit:qkv-matmul` | `19.243 ms` |
| `sam3-vit:window-attn` | `11.707 ms` |
| `sam3-vit:global-attn` | `7.150 ms` |
| isolated block 7 global `attn_core - qkv_rope` | `1.620 ms` local estimate |
| isolated block 0 window `attn_core - qkv_rope` | `0.375 ms` local estimate |

A Blackwell-only probe that stopped using the SAM3 ViT MMA FATTN path for
global blocks was checked under the full required-E2E A/B gate. It preserved
strict parity but regressed the formal metrics, so it is not kept:

| Candidate | Required E2E delta | Tail graph delta | Core compute delta | Parity | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| default global generic FATTN instead of SAM3 global MMA | `+6.319 ms` | `+1.807 ms` | `+4.538 ms` | exact, `diff_rows=0`, mask hashes `5/5` | reject |

Evidence:
`outputs/e2e-required-audit-sam3-bf16-current-isolated-20260630m/` and
`outputs/e2e-required-ab-sam3-bf16-global-fattn-select-r3-20260630m/`.

The next promotable optimization should be a real SAM3 ViT dataflow/kernel
change that reduces `tail_image_encode.graph_compute` under this audit, then
passes exact tracking parity and required-E2E A/B. Small isolated wins are only
pre-gates unless they also reduce the formal required-E2E denominator.

## 2026-06-30 Required-E2E Python quality parity gate

`e2e-required-measure` now runs the required split, CUDA profile, cuBLASLt
profile, isolated ViT bench, official Python artifact dump, C++ artifact dump,
and final audit as one measurement bundle. The audit accepts
`E2E_REQUIRED_AUDIT_PYTHON_PARITY=NAME=PATH` and records official Python versus
C++ quality parity next to the process split and CUDA attribution.

This gate is intentionally separate from `e2e-required-parity`:

- `e2e-required-parity` compares C++ baseline versus C++ candidate and should
  stay exact for performance candidates.
- `e2e-required-python-parity` compares official Python versus C++ under the
  same required frames, prompt, dtype, TF32 policy, and model filter. It allows
  small mask/hash differences while reporting the numeric deltas.

Current SAM3 BF16 evidence:

| Metric | Value |
| --- | ---: |
| Python/C++ rows | `5 / 5` |
| exact diff rows | `5` |
| tolerance diff rows | `0` |
| min bbox IoU | `0.979372` |
| max bbox delta px | `2.537` |
| max score abs delta | `0.016209` |
| max mask area rel delta | `0.013912` |
| min mask pixel IoU | `0.985238` |
| max mask pixel xor | `483` |
| mask hash equal rows | `0` |

Evidence:
`outputs/e2e-required-python-parity-sam3-bf16-current-20260630n/` and
`outputs/e2e-required-audit-sam3-bf16-current-python-parity-20260630n/`.

Acceptance implication: current BF16 is not exact official Python parity, but it
passes the relaxed quality gate. A speed candidate should first pass exact
C++/C++ artifact parity, then keep this official Python quality parity passing,
unless the candidate intentionally changes outputs and the changed contract is
documented.

## SAM3.1 mask-init E2E split and variant gate

`sam31-mask-init-audit` now summarizes the SAM3.1 synthetic mask-init matrix as
an E2E optimization gate. The denominator is the two-frame
`full_frame_step_ms`, with separate rows for frame0 image encode, detection-mask
initialization, frame1 image encode, encoded propagation, and tail remainder.
The same audit also records official Python timing for the matching synthetic
contract and rejects variants unless they keep the C++ mask identical to default
and improve the full-frame denominator with a meaningful signal.

Current BF16, TF32-on evidence for `320x240 center@3`, 5 interleaved repeats:

| Process | Mean ms | E2E % |
| --- | ---: | ---: |
| `full_frame_step` | `341.313` | `100.0%` |
| `frame0.image_encode` | `159.668` | `46.8%` |
| `frame0.add_detection` | `23.049` | `6.8%` |
| `tail.image_encode` | `143.451` | `42.0%` |
| `tail.propagate_encoded` | `15.093` | `4.4%` |
| `tail.remainder` | `0.053` | `0.0%` |
| `encode.total` | `303.118` | `88.8%` |

The official Python full cache-step equivalent was `1.252x` slower than C++
for this contract, while the C++ mask remained a quality-but-not-exact match to
official Python (`min mask IoU 0.981628`, `max XOR 270`). The optimization
target is therefore still the image encoder: almost all useful SAM3.1 mask-init
E2E time is in the two image encodes.

The current `WIN_PART -> CPY` probes were rechecked under this gate and are not
promoted:

| Candidate | Full-frame delta | Signal | Encode delta | C++ mask parity | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `win-part-cpy-fusion` | `+3.799 ms` | `+3.95` | `+2.843 ms` | same as default | reject |
| `win-window-fusions` | `+4.779 ms` | `+3.96` | `+3.340 ms` | same as default | reject |

Evidence:
`outputs/sam31-mask-init-matrix-winpart-current-phase3-20260630o/`,
`outputs/sam31-mask-init-audit-winpart-current-phase3-20260630o/`, and
`outputs/sam3-sam31-goal-audit-phase3-e2e-split-20260630o/`.

## Goal audit required-E2E SAM3 proof

The active SAM3/SAM3.1 goal audit now accepts the required-E2E audit JSON
directly through `--sam3-e2e-audit`. This avoids treating older
`model-matrix` tracking scopes as the authoritative SAM3 proof when the current
optimization contract is the stricter required-E2E split.

With
`outputs/e2e-required-audit-sam3-bf16-current-python-parity-20260630n/optimization_targets.json`
attached, the goal audit reports the SAM3 criterion as met:

| Metric | Value |
| --- | ---: |
| required/session Python over C++ | `1.152583x` |
| model E2E Python over C++ | `1.104293x` |
| official Python quality parity | `quality_pass` |
| Python/C++ quality rows | `5 / 5` |

The same run keeps the SAM3.1 mask-init E2E split criterion met, but the full
goal remains incomplete because SAM3.1 full tracking sequence parity is still
partial. Evidence:
`outputs/sam3-sam31-goal-audit-phase3-e2e-split-20260630p/summary.json`.

## SAM3.1 current sequence quality refresh

The stale SAM3.1 sequence comparison that reported `0.883095` mask IoU was
regenerated with the current C++ smoke binary and official Python BF16/TF32-on
contract. The current comparison is a quality pass, but not exact hash parity:

| Metric | Value |
| --- | ---: |
| C++ foreground pixels | `14530` |
| Python foreground pixels | `14592` |
| intersection / union | `14426 / 14696` |
| mask IoU | `0.981628` |
| XOR pixels | `270` |
| C++ only / Python only pixels | `104 / 166` |
| C++ bbox | `[83, 59, 241, 178]` |
| Python bbox | `[84, 59, 241, 178]` |
| XOR pixels on Python 8-neighbor edge | `232 / 270` |

The remaining disagreement is therefore concentrated on mask boundaries, not a
gross tracking target shift. The refreshed goal audit records
`official_mask_sequence_quality.status = quality_pass` and keeps the SAM3.1
tracking criterion `partial` until exact sequence hash parity, or an explicitly
accepted quality-parity contract, is chosen.

Evidence:
`outputs/sam31-mask-sequence-python/summary.json`,
`outputs/sam31-mask-sequence-python/mask_diff_rgb.png`, and
`outputs/sam3-sam31-goal-audit-phase3-sam31-sequence-current-20260630q/summary.json`.

## SAM3.1 required-E2E split

The SAM3.1 mask-init smoke now writes the process boundaries needed for a
stricter E2E comparison:

- model load, state creation, tracker creation, synthetic frame/mask
  construction, and output write are measured separately from model execution.
- the summary also emits grouped required-E2E fields:
  `required_session_setup_ms`, `required_input_prepare_ms`,
  `required_model_execute_ms`, `required_accounted_ms`, and
  `required_remainder_ms`.
- frame0/frame1 encode timing now includes preprocess, graph build/allocation,
  input upload, graph compute, state update, and PE build.
- mask-init and encoded propagation expose their internal timing structs in the
  same summary JSON.
- `sam31-mask-init-audit` uses `run_required_e2e_ms` as the variant decision
  metric when present, falling back to the older `full_frame_step_ms` only for
  older summaries.

Current BF16, TF32-on evidence for `320x240 center@3`, 3 interleaved repeats:

| Required-E2E process | Mean ms | Required-E2E % |
| --- | ---: | ---: |
| `run_required_e2e` | `354.471` | `100.0%` |
| `required.session_setup` | `14.032` | `4.0%` |
| `required.input_prepare` | `0.124` | `0.0%` |
| `required.model_execute` | `340.378` | `96.0%` |
| `tracker.create` | `14.031` | `4.0%` |
| `frame0.image_encode` | `159.362` | `45.0%` |
| `frame0.add_detection` | `23.147` | `6.5%` |
| `tail.image_encode` | `142.815` | `40.3%` |
| `tail.propagate_encoded` | `15.001` | `4.2%` |
| `required.remainder` | `-0.062` | `-0.0%` |

This reframes the next optimization target: C++ still beats the official
Python full cache-step on this small contract (`Python/required = 1.206x`), but
`85.2%` of the required C++ E2E is still the two image encodes. The
propagation decoder itself is already a small part of this case. Variant probes
that only change memory-attention or propagation kernels are therefore unlikely
to move the E2E result unless the image encoder path is also addressed.

The CUDA interactive mask-head warmup is now guarded at model scope so repeated
tracker creation in one process does not rebuild and execute that warmup graph
for every tracker. Under the current `cpp_warmup_runs=1` matrix this is not the
dominant measured cost; the remaining `tracker.create` time is mostly PE/backend
cache preparation and is only about `4%` of required E2E.

Current variant decisions under the new required-E2E gate:

| Variant | Decision | Required-E2E delta | Signal | C++ default parity |
| --- | --- | ---: | ---: | --- |
| `default` | baseline | `0.000 ms` | `0.00` | same |
| `all` | reject quality diff | `+0.194 ms` | `+0.34` | differs by 10 pixels |
| `sa123` | reject no E2E win | `+0.723 ms` | `+0.69` | same |

Evidence:
`outputs/sam31-e2e-split-matrix-warmup-once-20260630r/summary.json` and
`outputs/sam31-e2e-split-audit-warmup-once-20260630r/optimization_targets.md`.

A one-run smoke with the grouped fields confirms that the runner and summarizer
now preserve this split end to end. It reported
`required_model_execute_ms = 340.637 ms` out of
`run_required_e2e_ms = 354.606 ms`; the two image encodes were
`301.832 ms` (`85.1%` of required E2E). Evidence:
`outputs/sam31-required-split-smoke-20260630t/summary.json`,
`outputs/sam31-required-split-matrix-smoke-20260630t/summary.json`, and
`outputs/sam31-required-split-audit-smoke-20260630t.md`.

Encoder-level candidate probes:

- Existing ViT graph batch execution is correct for duplicated lanes but is not
  faster for this contract. With the SAM3.1 BF16 propagation neck, `B=1` was
  `152.152 ms/frame`, while `B=2` was `318.787 ms total` or
  `159.394 ms/frame`; `lane_max_abs_diff = 0`.
- Existing generic fusion toggles do not produce a measurable win on this
  SAM3.1 BF16 encoder contract. In a 12-repeat isolated encoder A/B, default
  was `153.796 ms` mean / `152.476 ms` median; cuBLASLt `GELU_ERF` fusion was
  `164.100 ms` mean / `161.726 ms` median; cuDNN FC1+GELU was
  `154.470 ms` mean / `152.666 ms` median; cuBLASLt bias+residual was
  `154.712 ms` mean / `153.127 ms` median. These remain rejected toggles, not
  defaults.
- Skipping the interactive object-pointer path is not a valid current
  optimization. It leaves the initial SAM3.1 memory slot without the saved image
  features/precomputed backend K/V required by propagation, producing
  `SAM3.1 memory slot does not contain saved image features`. The diagnostic
  variant was removed from the matrix runner instead of being kept as an
  opt-in candidate.

The next root optimization target remains the CUDA image encoder graph compute,
not simple two-frame batching and not mask-init/propagation-only variants. In
the grouped smoke, frame0 encode spent `152.268 ms` in graph compute and frame1
spent `141.773 ms`; graph build, allocation, upload, and PE construction are
much smaller secondary costs. The remaining accepted path should therefore
change the actual ViT GEMM/attention dataflow or kernels, not enable one of the
existing coarse toggles.

Evidence:
`outputs/sam31-vit-geluerf-ab-20260630t/`,
`outputs/sam31-vit-batch-bench-b1-20260630s/vit_bench.json`,
`outputs/sam31-vit-batch-bench-b2-20260630s/vit_bench.json`,
`outputs/sam31-no-interactive-objptr-matrix-20260630s/`, and
`outputs/sam31-no-interactive-objptr-backend-slot-smoke-20260630s/`.

### Required-E2E accounting correction

The SAM3.1 mask-init smoke now separates the required model work from input
construction more strictly. `required_model_execute_ms` is computed as
`frame0_ms + track_step_total_ms`, not `frame0_ms + propagate_ms`, because
`propagate_ms` includes the first tail frame construction wall time. The summary
also emits:

- `tail_image_encode_total_ms` and `tail_image_encode_avg_ms`
- `required_image_encode_ms`
- `required_mask_init_ms`
- `required_propagate_encoded_ms`
- `required_model_accounted_ms`
- `required_model_remainder_ms`

The matrix runner and audit summarizer prefer these fields when present and
fall back to older two-frame summaries when reading historical output.

One smoke run after this correction reported:

| Required-E2E process | ms |
| --- | ---: |
| `run_required_e2e_ms` | `356.166` |
| `required_session_setup_ms` | `13.980` |
| `required_input_prepare_ms` | `0.130` |
| `required_model_execute_ms` | `342.063` |
| `required_image_encode_ms` | `302.062` |
| `required_mask_init_ms` | `25.160` |
| `required_propagate_encoded_ms` | `14.841` |
| `required_model_remainder_ms` | `0.000` |
| `required_remainder_ms` | `-0.008` |

Evidence:
`outputs/sam31-e2e-required-split-fix-smoke-20260630u/summary.json`.

Under the corrected gate, a default vs CUDA-graph E2E matrix preserved the same
mask as default but rejected CUDA graphs: default `run_required_e2e_ms` was
`355.529 ms`, while `cuda-graphs` was `359.318 ms` (`+3.789 ms`, signal
`2.45`). The same corrected split shows the default image encoder share is
`303.532 ms / 355.529 ms = 85.4%`, while `required_mask_init_ms` is `23.071 ms`
and `required_propagate_encoded_ms` is `14.954 ms`.

Evidence:
`outputs/sam31-e2e-cuda-graphs-matrix-20260630u/summary.json` and
`outputs/sam31-e2e-cuda-graphs-audit-20260630u/optimization_targets.md`.

The isolated SAM3.1 BF16 encoder also rejects CUDA graphs and broader BF16
linear-output boundaries on this contract:

| Candidate | Mean ms | Median ms | Delta vs local default | Decision |
| --- | ---: | ---: | ---: | --- |
| `default` for CUDA graph A/B | `153.836` | `152.554` | `0.000` | baseline |
| `SAM3_CUDA_ENABLE_GRAPHS=1` | `154.738` | `151.809` | `+0.902` | reject/no win |
| `default` for BF16 boundary A/B | `154.445` | `152.886` | `0.000` | baseline |
| `SAM3_BF16_VIT_LINEAR_OUTPUT=1` | `155.955` | `152.832` | `+1.510` | reject/no win |
| `SAM3_BF16_VIT_LINEAR_OUTPUT=1 SAM3_BF16_VIT_ATTN_PROJ_OUTPUT=all` | `160.868` | `157.838` | `+6.423` | reject |

Evidence:
`outputs/sam31-vit-cuda-graphs-ab-20260630u/` and
`outputs/sam31-vit-linear-output-ab-20260630u/`.

RoPE is not the next unimplemented CUDA-vs-Metal gap for this ViT contract:
the CUDA backend already contains `rope_pair_fused` and
`cont_rope_pair_fused`, and prior profiling confirms the SAM3 ViT Q/K paths hit
the fused route. The remaining large target is still the actual BF16 ViT
GEMM/attention dataflow: QKV/projection and especially MLP FC1/FC2 need a
same-contract fused or lower-overhead CUDA path to move the required E2E
denominator.

### Required-E2E image graph split

The SAM3.1 mask-init smoke now also writes `tail_encode_timing_total`, the sum
of all required tail image-encode timing rows. The matrix runner derives these
additional E2E fields from it:

- `required_image_encode_graph_compute_ms`
- `required_image_encode_non_compute_ms`
- `tail_encode_graph_compute_total_ms`
- `tail_encode_non_compute_total_ms`

This makes the acceptance surface stricter: a candidate must move the required
E2E denominator and the dominant image-encoder graph-compute slice, not only a
noisy wall-clock subtotal.

A fresh default vs CUDA-graph run with the deeper split reported:

| Variant | Required E2E | Required image graph delta | Tail image graph delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `default` | `355.105 ms` | `0.000 ms` | `0.000 ms` | baseline |
| `cuda-graphs` | `360.396 ms` | `+3.972 ms` | `+2.021 ms` | reject |

The default required-E2E split in that run was `301.541 ms` image encode
(`84.9%` of required E2E), of which `293.890 ms` was image-encoder graph
compute (`82.8%`). Tail image encode was `142.473 ms`, with `141.359 ms` in
graph compute. Evidence:
`outputs/sam31-e2e-split-fields-matrix-20260630v/summary.json` and
`outputs/sam31-e2e-split-fields-audit-20260630v/optimization_targets.md`.

The small `GGML_CUDA_ENABLE_UNARY_CPY_VEC4_PACKED=1` probe was also measured
through this same gate. It preserved the default C++ mask, but it is not an
optimization: required E2E improved only as a noise-level wall-clock subtotal
(`-1.086 ms`, signal `-1.31`), while the image graph-compute slice did not move
in the right direction (`+0.044 ms`) and tail image graph compute regressed by
`+0.118 ms`.

| Variant | Required E2E | Required E2E delta | Image graph delta | Tail graph delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `default` | `355.066 ms` | `0.000 ms` | `0.000 ms` | `0.000 ms` | baseline |
| `unary-cpy-packed` | `353.979 ms` | `-1.086 ms` | `+0.044 ms` | `+0.118 ms` | reject/noise only |

Evidence:
`outputs/sam31-e2e-unary-cpy-packed-matrix-20260630v/summary.json` and
`outputs/sam31-e2e-unary-cpy-packed-audit-20260630v/optimization_targets.md`.

The remaining optimization target is therefore unchanged but now measured more
cleanly: reduce SAM3/SAM3.1 ViT image-encoder graph compute itself, especially
the MLP FC1/FC2, QKV/projection, and attention groups, then pass exact tracking
parity and the required-E2E split gate.

### Required-E2E Python process split

The official Python SAM3.1 mask-sequence runner now reports the same required
E2E process groups used by the C++ mask-init smoke: synthetic input
construction, `init_state`, mask tensor preparation, frame-0/frame-1
backbone/cache work, mask initialization, cached propagation, accounted total,
and accounting remainder. The matrix runner and audit now preserve those fields
and emit:

- `python_required_e2e_over_cpp_required_e2e_ratio`
- `python_backbone_over_cpp_image_graph_ratio`
- `required_core_compute_ms`
- `required_non_image_model_ms`
- `required_non_graph_or_setup_ms`

This answers the cached-tail accounting question directly: the Python required
E2E denominator is no longer only the cached model step. It includes the Python
setup/input work needed for the same two-frame mask-init contract, while C++
still includes its state/tracker setup, synthetic input construction, image
encodes, mask initialization, and encoded propagation.

Fresh BF16/TF32-on evidence for `320x240 center@3`, 5 interleaved repeats,
`cpp_warmup_runs=1`, and `python_warmup_runs=1`:

| Required-E2E process | C++ default mean | C++ E2E % | Python mean | Python E2E % |
| --- | ---: | ---: | ---: | ---: |
| required E2E | `354.681 ms` | `100.0%` | `445.995 ms` | `100.0%` |
| session setup | `13.697 ms` | `3.9%` | `17.154 ms` | `3.8%` |
| input prepare | `0.119 ms` | `0.0%` | `1.710 ms` | `0.4%` |
| model execute | `340.873 ms` | `96.1%` | `427.093 ms` | `95.8%` |
| image/backbone graph slice | `294.936 ms` | `83.2%` | `400.318 ms` | `89.8%` |
| mask init | `23.372 ms` | `6.6%` | `9.924 ms` | `2.2%` |
| cached/encoded propagate | `14.987 ms` | `4.2%` | `16.851 ms` | `3.8%` |

The Python required-E2E over C++ required-E2E ratio is `1.257x`; the Python
backbone/cache over C++ image-graph ratio is `1.357x`. C++ therefore already
wins this small SAM3.1 mask-init required-E2E contract when both sides include
the required setup/input/model work. The remaining C++ bottleneck is still the
image encoder graph because it is `83.2%` of C++ required E2E.

Current candidate decisions under the same stricter split:

| Variant | Required E2E | Required delta | Signal | Image graph delta | Tail graph delta | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `default` | `354.681 ms` | `0.000 ms` | `0.00` | `0.000 ms` | `0.000 ms` | baseline |
| `unary-cpy-packed` | `354.691 ms` | `+0.010 ms` | `0.03` | `-0.034 ms` | `-0.069 ms` | reject/no E2E win |

`unary-cpy-packed` preserves the default C++ mask, but the graph deltas are far
inside noise and the required denominator does not improve. It remains a reject.

`vit-pos-cache` was also rechecked because frame-0 PE build is still visible in
the split. It is not an optimization in the current graph:

| Variant | Required E2E | Required delta | Signal | Image graph delta | Frame0 graph build | Frame0 PE build | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `default` | `355.825 ms` | `0.000 ms` | `0.00` | `0.000 ms` | `0.550 ms` | `5.206 ms` | baseline |
| `vit-pos-cache` | `364.905 ms` | `+9.080 ms` | `9.90` | `-0.374 ms` | `11.805 ms` | `5.715 ms` | reject/no E2E win |

Although `vit-pos-cache` slightly lowers measured image graph compute, it adds
about `+11.3 ms` to frame-0 graph build and increases total image encode by
`+11.353 ms`, so it is rejected.

Evidence:
`outputs/sam31-required-e2e-process-split-matrix-20260629w/summary.json`,
`outputs/sam31-required-e2e-process-split-audit-20260629w/optimization_targets.md`,
`outputs/sam31-required-e2e-vit-pos-cache-matrix-20260629w/summary.json`, and
`outputs/sam31-required-e2e-vit-pos-cache-audit-20260629w/optimization_targets.md`.

The next acceptable optimization must reduce
`required_image_encode_graph_compute_ms` without moving the cost into graph
build/allocation or setup, and still keep C++/C++ mask identity plus the current
official Python quality gate. Exact SAM3.1 sequence hash parity remains a
separate unresolved gate.

### SAM3.1 selection and parity diagnostics

The SAM3.1 mask-init smoke now separates selection diagnostics from the E2E
timing denominator. `summary.json` records `selected_mask_index`,
`mask_foreground_pixels`, and `decoder_iou_scores`; the matrix runner and audit
preserve those fields so a candidate can be rejected for quality drift without
confusing that with a speed change.

Fresh BF16/TF32-on evidence for `320x240 center@3`, 5 interleaved repeats,
`cpp_warmup_runs=1`, and `python_warmup_runs=1`, confirms the new fields flow
through the audit:

| Field | Value |
| --- | ---: |
| C++ required E2E | `354.740 ms` |
| Python required E2E | `444.157 ms` |
| Python required E2E / C++ required E2E | `1.252x` |
| C++ required image encode | `301.826 ms` / `85.1%` |
| C++ required image graph compute | `293.671 ms` / `82.8%` |
| C++ required mask init | `24.260 ms` / `6.8%` |
| C++ required encoded propagation | `14.854 ms` / `4.2%` |
| C++ selected mask index | `0` |
| C++ foreground pixels | `14530` |
| C++ decoder scores | `[0.979781, 0.444142, -1.188118]` |
| Python mask IoU vs C++ | `0.981628` |
| XOR pixels | `270` |

The same run also confirms that the current SAM3.1 mismatch is not a gross mask
candidate-selection error. The dumped `sam31_mux_iou` tensor has argmax `0` for
both official Python and C++, and C++ selects slot `0` from the three mask
candidates. The drift is already present in the selected mask logits.

Dumped tensor comparison on the same case localizes the parity issue further:

| Boundary | Max abs | Mean abs |
| --- | ---: | ---: |
| `sam31_mem_attn_output` | `0.919121` | `0.049223` |
| `sam31_mux_final_q` | `0.214683` | `0.014677` |
| `sam31_mux_final_k` | `2.792472` | `0.797503` |
| `sam31_mux_upscaled` | `0.148115` | `0.008387` |
| `sam31_mux_iou` | `0.671670` | `0.180155` |
| `sam31_mux_obj_score` | `0.410803` | `0.261063` |
| `sam31_mux_masks` | `1.177200` | `0.065582` |

The dump run is intentionally diagnostic only because tensor dump hooks change
wall timing. Its useful conclusion is the location of the parity blocker:
`upscaled` and final-query side are comparatively close, while the image-key
side feeding final attention is highly divergent. The next root fix should
therefore target the memory-attention/image-feature path into the multiplex
decoder, especially the final K path, before spending more time on thresholding,
PNG resize, selected-index handling, or cached-tail accounting.

Evidence:
`outputs/sam31-selection-diagnostics-matrix-r5-20260629x/summary.json`,
`outputs/sam31-selection-diagnostics-audit-r5-20260629x/optimization_targets.md`,
`outputs/sam31-selection-diagnostics-dump-20260629x/mux-summary.json`, and
`outputs/sam31-selection-diagnostics-dump-20260629x/python-run/summary.json`.

## 2026-06-29 SAM3.1 Required E2E Component Attribution

The SAM3.1 mask-init matrix audit now separates the required two-frame E2E
contract into process groups for both official Python and C++, and projects a
synchronized CUDA node profile onto the normal non-profiled image-encoder graph
denominator. This makes the optimization target explicit: candidates must
reduce `required_image_encode_graph_compute_ms` and the required E2E
denominator while preserving the default C++ mask and the Python quality gate.

Fresh BF16/TF32-on evidence for `320x240 center@3`, 5 interleaved repeats,
`cpp_warmup_runs=1`, and `python_warmup_runs=1`:

| Required-E2E process | C++ default mean | C++ E2E % | Python mean | Python E2E % |
| --- | ---: | ---: | ---: | ---: |
| required E2E | `357.924 ms` | `100.0%` | `474.038 ms` | `100.0%` |
| session setup | `14.263 ms` | `4.0%` | `45.953 ms` | `9.7%` |
| input prepare | `0.122 ms` | `0.0%` | `5.501 ms` | `1.2%` |
| model execute | `343.547 ms` | `96.0%` | `422.542 ms` | `89.1%` |
| image/backbone | `304.508 ms` | `85.1%` | `395.484 ms` | `83.4%` |
| image graph compute | `296.086 ms` | `82.7%` | - | - |
| mask init | `24.018 ms` | `6.7%` | `10.264 ms` | `2.2%` |
| cached/encoded propagate | `15.021 ms` | `4.2%` | `16.795 ms` | `3.5%` |

The image-encoder graph attribution uses synchronized profile rows only for
ratios, then projects those ratios onto the normal `296.086 ms` image graph
denominator:

| Component | Projected normal ms | Required E2E % | Main stages |
| --- | ---: | ---: | --- |
| `image.vit.mlp_matmul` | `121.144` | `33.8%` | FC2 and FC1 matmuls |
| `image.vit.qkv_matmul` | `38.778` | `10.8%` | QKV matmul |
| `image.vit.attention` | `37.779` | `10.6%` | window/global attention |
| `image.vit.layout_rope_copy` | `26.009` | `7.3%` | Q/K layout, RoPE, window copy |
| `image.other` | `21.244` | `5.9%` | non-classified image graph nodes |
| `image.neck` | `20.885` | `5.8%` | deconv and conv |
| `image.vit.proj_matmul` | `14.537` | `4.1%` | attention projection |
| `image.vit.mlp_gelu` | `14.519` | `4.1%` | MLP GELU |

Evidence:
`outputs/sam31-e2e-component-kernel-screen-r5-20260629b/summary.json`,
`outputs/sam31-e2e-component-kernel-screen-r5-20260629b/audit.json`, and
`outputs/sam31-e2e-component-kernel-screen-r5-20260629b/optimization_targets.md`.

The same stricter gate rejects the current switch-level CUDA candidates. These
results are useful because they distinguish already-essential optimizations from
remaining root work:

| Screen | Candidate | Required E2E delta | Image graph / tail graph signal | Decision |
| --- | --- | ---: | --- | --- |
| conv/deconv | `no-cudnn-conv2d` | `+918.481 ms` | image graph `+918.647 ms` | cuDNN conv2d is required |
| conv/deconv | `no-conv-transpose-k2s2` | `+1788.059 ms` | image graph `+1780.863 ms` | k2s2 deconv path is required |
| conv/deconv | `no-conv-transpose-k2s2-gemm` | `+262.500 ms` | image graph `+261.704 ms` | GEMM deconv subpath is required |
| conv/deconv | `cudnn-conv2d-f32-lowp` | `+0.162 ms` | no graph win | reject/no E2E win |
| conv/deconv | `cublaslt-autotune` | `+1.156 ms` | mask differs | reject/quality diff |
| cuBLASLt algo | `vit-qkv-algo1` | `+0.367 ms` | image graph `+1.803 ms` | reject/no E2E win |
| cuBLASLt algo | `vit-mlp-fc1-algo1` | `+1.338 ms` | image graph `+1.986 ms` | reject/no E2E win |
| cuBLASLt algo | `vit-mlp-fc2-algo1` | `+1.429 ms` | image graph `+2.373 ms` | reject/no E2E win |
| cuBLASLt algo | `vit-attn-proj-algo1` | `+1.647 ms` | image graph `+0.709 ms` | reject/no E2E win |
| copy/layout | `no-rope-pair-fusion` | `+65.495 ms` | image graph `+67.358 ms` | existing RoPE fusion is required |
| copy/layout | `no-unary-cpy-fusion` | `+21.432 ms` | image graph `+22.096 ms` | existing unary-copy fusion is required |
| copy/layout | `no-cont-rope-pair-fusion` | `+8.908 ms` | image graph `+6.173 ms` | existing cont-RoPE fusion is required |
| copy/layout | `unary-cpy-packed` | `-0.870 ms` | graph does not improve | reject/noise only |
| MLP fusion | `cudnn-mlp-bf16` | `+1.079 ms` | same mask, no graph win | reject/no E2E win |
| MLP fusion | `cublaslt-bias-residual` | `-0.038 ms` | mask differs | reject/quality diff |
| MLP fusion | `cublaslt-gelu-erf` | `+23.366 ms` | mask differs | reject/quality diff |
| MLP fusion | `cublaslt-gelu-erf-residual` | `+22.858 ms` | mask differs | reject/quality diff |
| MLP fusion | `cudnn-mlp-f16-f32` | `+20.685 ms` | mask differs | reject/quality diff |

Evidence:
`outputs/sam31-e2e-cublaslt-algo-r5-20260629b/optimization_targets.md`,
`outputs/sam31-e2e-copy-layout-fusion-r5-20260629b/optimization_targets.md`,
and `outputs/sam31-e2e-mlp-fusion-r5-20260629b/optimization_targets.md`.

Conclusion: the current CUDA backend already depends on the important conv,
deconv, RoPE, and unary-copy fusions. The remaining high-value root target is
not another environment switch. It is a same-precision, same-mask CUDA path for
the SAM3/SAM3.1 ViT image encoder, especially MLP FC1/FC2, QKV/projection, and
attention/layout around those matmuls.

### SAM3.1 ViT Batch And Structure Recheck

The next root candidate was to amortize the required two-frame image encoding by
using the existing ViT batch dimension. `sam3_vit_batch_bench` confirms that the
batch graph is numerically lane-stable for repeated images (`lane_max_abs_diff=0`
for batch 2 and 4), but it does not improve per-frame latency on this GPU:

| Contract | Batch | Mean ms | Mean ms/frame | Lane max abs diff |
| --- | ---: | ---: | ---: | ---: |
| ViT only | `1` | `141.327` | `141.327` | - |
| ViT only | `2` | `292.379` | `146.190` | `0` |
| ViT only | `4` | `578.576` | `144.644` | `0` |
| ViT + tracker neck | `1` | `153.333` | `153.333` | - |
| ViT + tracker neck | `2` | `317.618` | `158.809` | `0` |

Evidence is the terminal run recorded with
`./build/xmake-release-cuda/examples/sam3_vit_batch_bench` against
`models/sam3.1/sam3.1_multiplex-bf16.ggml`.

The local block0 cumulative stop-point sweep explains why batch encoding is not
a simple win. With batch 2, several per-frame stages get worse despite exact
lane agreement:

| Stage | Batch 1 mean/frame | Batch 2 mean/frame | Delta |
| --- | ---: | ---: | ---: |
| `qkv_proj` | `1.311 ms` | `1.512 ms` | `+0.202 ms` |
| `qkv_layout` | `1.535 ms` | `1.826 ms` | `+0.291 ms` |
| `qkv_rope` | `1.554 ms` | `1.919 ms` | `+0.364 ms` |
| `attn_core` | `1.861 ms` | `2.044 ms` | `+0.183 ms` |
| `attn_proj` | `2.097 ms` | `2.500 ms` | `+0.403 ms` |
| `mlp_fc1` | `3.173 ms` | `3.430 ms` | `+0.258 ms` |
| `mlp_gelu` | `3.549 ms` | `4.047 ms` | `+0.498 ms` |
| `mlp_fc2` | `4.421 ms` | `4.927 ms` | `+0.506 ms` |
| `block` | `4.460 ms` | `4.757 ms` | `+0.297 ms` |

Evidence:
`outputs/sam31-vit-stage-batch1-block0-r12-20260629c/summary.md` and
`outputs/sam31-vit-stage-batch2-block0-r12-20260629c/summary.md`.

The current Required E2E gate also rejects the remaining graph-structure
variants:

| Variant | Required E2E delta | Image graph delta | Tail graph delta | Quality | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `vit-mlp-flat-chain` | `+0.559 ms` | `+0.389 ms` | `+0.167 ms` | same mask | reject/no win |
| `bf16-vit-win-part-input` | `+3.488 ms` | `+3.245 ms` | `+1.605 ms` | same mask | reject/no win |
| `vit-contiguous-attention-v` | `+10.672 ms` | `+5.784 ms` | `+2.789 ms` | same mask | reject/no win |
| `no-vit-direct-qkv-views` | `+15.864 ms` | `+13.799 ms` | `+7.078 ms` | same mask | reject/no win |
| `vit-qkv-chain` | `+26.554 ms` | `+23.305 ms` | `+11.699 ms` | mask differs | reject |
| `vit-mlp-chain` | `+35.022 ms` | `+33.397 ms` | `+17.112 ms` | mask differs | reject |
| `vit-qkv-mlp-chain` | `+61.564 ms` | `+59.044 ms` | `+30.944 ms` | mask differs | reject |

Evidence:
`outputs/sam31-e2e-vit-structure-r5-20260629c/summary.json`,
`outputs/sam31-e2e-vit-structure-r5-20260629c/audit.json`, and
`outputs/sam31-e2e-vit-structure-r5-20260629c/optimization_targets.md`.

The next implementation should therefore not be a two-frame batch path or a
reshape-only graph variant. The remaining root work is a CUDA implementation
change inside the hot ViT groups: faster BF16 QKV/MLP matmul epilogues that
preserve the exact default mask, or attention/layout kernels that reduce the
`layout_rope_copy` and attention slices without adding extra materialization.

### SAM3.1 Required-E2E Split Recheck

The SAM3.1 mask-init matrix is now usable as a repeatable E2E optimization
gate from `just sam31-mask-init-audit`. The just entrypoint exposes the C++
thread count and can attach the synchronized default CUDA profile through
`SAM31_MASK_INIT_MATRIX_PROFILE_DEFAULT=1`. The normal variant runs remain the
speed source; the profile pass is used only for attribution.

The current required-E2E process split on `320x240 center@3` is:

| Process | Mean ms | Required E2E share |
| --- | ---: | ---: |
| `required_e2e` | `357.052` | `100.0%` |
| `required.image_encode` | `304.320` | `85.2%` |
| `required.image_encode.graph_compute` | `295.727` | `82.8%` |
| `required.mask_init` | `23.730` | `6.6%` |
| `required.propagate_encoded` | `15.107` | `4.2%` |
| `required.session_setup` | `13.785` | `3.9%` |

The synchronized profile attributes the image encoder graph to:

| Component | Projected normal ms | Required E2E share |
| --- | ---: | ---: |
| `image.vit.mlp_matmul` | `120.804` | `33.8%` |
| `image.vit.qkv_matmul` | `38.740` | `10.9%` |
| `image.vit.attention` | `37.784` | `10.6%` |
| `image.vit.layout_rope_copy` | `26.163` | `7.3%` |
| `image.neck` | `20.796` | `5.8%` |
| `image.vit.mlp_gelu` | `14.612` | `4.1%` |
| `image.vit.proj_matmul` | `14.532` | `4.1%` |

The latest opt-in fusion checks did not produce an accepted speedup:

| Candidate | Repeats | Required E2E delta | Image graph delta | Quality | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `norm-affine-cpy-fusion` | `3` | `-0.769 ms` | `+0.949 ms` | same mask | reject/noise only |
| `mixed-add-norm-fusion` | `3` | `+0.056 ms` | `+0.636 ms` | same mask | reject/no win |
| `cudnn-mlp-bf16` | `10` | `+0.547 ms` | `+0.116 ms` | same mask | reject/no win |
| `cudnn-mlp-f16-f32` | `3` | `+20.962 ms` | `+20.343 ms` | mask differs | reject |
| `vit-mlp-chain-b0-7` | `3` | `+6.342 ms` | `+8.338 ms` | mask differs | reject |
| `vit-qkv-chain-b0-7` | `3` | `+5.265 ms` | `+6.012 ms` | mask differs | reject |
| `vit-qkv-mlp-chain-b0-7` | `3` | `+12.105 ms` | `+14.025 ms` | mask differs | reject |
| `vit-window-late-global-tile` | `3` | `+127.216 ms` | `+128.516 ms` | mask differs | reject |

Evidence:
`outputs/sam31-e2e-final-fusion-screen-r3-20260629a/optimization_targets.md`,
`outputs/sam31-e2e-hotpath-screen-audit-r3-20260629a/optimization_targets.md`,
`outputs/sam31-e2e-cudnn-mlp-screen-audit-r3-20260629a/optimization_targets.md`,
and
`outputs/sam31-e2e-cudnn-mlp-bf16-audit-r10-20260629a/optimization_targets.md`.

This means the next optimization should not be another cached-tail accounting
change, process-level shortcut, or opt-in graph reshape. The remaining target is
the actual image-encoder graph, especially BF16 MLP/QKV matmul epilogues and the
attention/layout kernels around them. Any future candidate must lower
`required_e2e`, `required.image_encode.graph_compute`, and
`tail.image_encode.graph_compute` while keeping the default selected mask,
foreground pixel count, decoder scores, and Python mask IoU gate intact.

### SAM3 Required-E2E Process Map

`summarize_e2e_optimization_targets.py` now emits
`e2e_process_measurements` and a `Required E2E Process Map` Markdown section.
This keeps formal C++ required work, Python session wall timing, and Python hook
diagnostics in separate columns so cached-tail placement cannot be mistaken for
skipped work.

The current SAM3 BF16 smoke map for three frames shows that C++ still wins the
formal session denominator, but the actual tail image encoder graph is slower
than the official Python timed-tail backbone hook:

| Process | C++ ms | Python ms | C++ - Python | Contract role |
| --- | ---: | ---: | ---: | --- |
| `required_session_e2e` | `592.421` | `635.364` | `-42.943` | formal denominator |
| `model_e2e` | `584.315` | `584.124` | `+0.191` | formal subtotal |
| `tail_image_encode_or_backbone` | `284.144` | `167.175` | `+116.969` | diagnostic hook |
| `tail_image_encode.graph_compute` | `282.341` | `167.175` | `+115.166` | primary root target |
| `tail_propagate_or_tracker` | `25.356` | `42.661` | `-17.305` | diagnostic hook |

The same report projects the required image graph budget to
`image.vit.mlp_matmul = 141.117 ms`, `image.neck = 65.668 ms`,
`image.vit.qkv_matmul = 60.256 ms`, and `image.vit.attention = 58.557 ms`.
It also now emits `required_image_graph_stage_budget`, so single-stage isolated
A/B runs can be projected against atomic required-image work rather than only
group buckets. The largest atomic stages are tail `mlp_fc2 = 60.258 ms`, tail
`mlp_fc1 = 59.518 ms`, tail `qkv = 38.958 ms`, and frame0
`prefix-matmul = 34.152 ms`.
Non-compute tail encode work is only `2.747 ms`, so the next accepted
optimization must move graph compute rather than cached-tail placement,
decode/input handling, or graph accounting.

I also rechecked the obvious BF16-output chain shortcut on the captured block0
E2E tensors. The isolated A/B summarizer now projects each local stage delta
onto the required tail graph stage budget instead of the coarser graph group, so
`mlp_fc1` and `mlp_fc2` no longer double-count the whole MLP matmul bucket. The
candidate remains a clear reject:

| Stage | Candidate | Baseline steady mean | Candidate steady mean | Local delta | Tail stage budget | Projected required-E2E delta | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qkv_proj` | `SAM3_BF16_VIT_QKV_CHAIN=1 GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1` | `0.700 ms` | `0.744 ms` | `+0.044 ms` | `38.958 ms` | `+2.470 ms` | reject/slower |
| `mlp_fc1` | `SAM3_BF16_VIT_MLP_CHAIN=1 GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1` | `0.928 ms` | `1.133 ms` | `+0.205 ms` | `59.518 ms` | `+13.174 ms` | reject/slower |
| `mlp_fc2` | `SAM3_BF16_VIT_LINEAR_OUTPUT=1 GGML_CUDA_CUBLASLT_DIRECT_BF16_DST=1` | `1.149 ms` | `1.204 ms` | `+0.055 ms` | `60.258 ms` | `+2.897 ms` | reject/slower |

Evidence:
`outputs/e2e-required-process-map-sam3-bf16-20260629a/optimization_targets.md`
and
`outputs/e2e-required-vit-isolated-bf16-output-current-20260629a/{qkv,fc1,fc2}_ab.md`.
The same projection can now be regenerated through
`just e2e-required-vit-stage-ab` by pointing
`E2E_REQUIRED_VIT_STAGE_AB_BASELINE`,
`E2E_REQUIRED_VIT_STAGE_AB_CANDIDATE`, `E2E_REQUIRED_VIT_STAGE_AB_STAGE_NAME`,
and `E2E_REQUIRED_VIT_STAGE_AB_AUDIT` at the desired summaries/audit.

Conclusion: the current optimization path should move to explicit CUDA/ggml
implementation work inside the image graph. The highest-value areas are a
same-output ViT MLP/QKV epilogue or fused attention/layout kernels that keep the
default mask parity while reducing `tail_image_encode.graph_compute`.
