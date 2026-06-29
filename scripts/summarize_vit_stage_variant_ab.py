# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


STAGE_BUDGET_GROUPS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("image.vit.qkv_matmul",),
    "qkv_layout": ("image.vit.layout_rope_copy",),
    "qkv_rope": ("image.vit.layout_rope_copy",),
    "attn_core": ("image.vit.attention",),
    "attn_proj": ("image.vit.proj_matmul",),
    "window_part": ("image.vit.layout_rope_copy",),
    "window_unpart": ("image.vit.layout_rope_copy",),
    "mlp_fc1": ("image.vit.mlp_matmul",),
    "mlp_gelu": ("image.vit.mlp_gelu",),
    "mlp_fc2": ("image.vit.mlp_matmul",),
    "mlp": ("image.vit.mlp_matmul", "image.vit.mlp_gelu"),
    "mlp_fc1_gelu": ("image.vit.mlp_matmul", "image.vit.mlp_gelu"),
    "norm1": ("image.vit.norm_residual_copy",),
    "norm2": ("image.vit.norm_residual_copy",),
}

STAGE_BUDGET_STAGES: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("sam3-vit:qkv-matmul",),
    "qkv_layout": ("sam3-vit:qk-layout-rope",),
    "qkv_rope": ("sam3-vit:qk-layout-rope",),
    "attn_core": ("sam3-vit:window-attn", "sam3-vit:global-attn"),
    "attn_proj": ("sam3-vit:proj-matmul",),
    "window_part": ("sam3-vit:window-part",),
    "window_unpart": ("sam3-vit:window-unpart",),
    "mlp_fc1": ("sam3-vit:mlp-fc1-matmul",),
    "mlp_gelu": ("sam3-vit:mlp-gelu",),
    "mlp_fc2": ("sam3-vit:mlp-fc2-matmul",),
    "mlp": (
        "sam3-vit:mlp-fc1-matmul",
        "sam3-vit:mlp-gelu",
        "sam3-vit:mlp-fc2-matmul",
    ),
    "mlp_fc1_gelu": ("sam3-vit:mlp-fc1-matmul", "sam3-vit:mlp-gelu"),
    "norm1": ("sam3-vit:norm",),
    "norm2": ("sam3-vit:norm",),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def fmt_signal(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def ci95(values: list[float]) -> float | None:
    sd = sample_sd(values)
    if sd is None:
        return None
    return 1.96 * sd / math.sqrt(len(values))


def signal(delta_ms: float | None, lhs: list[float], rhs: list[float]) -> float | None:
    if delta_ms is None or len(lhs) < 2 or len(rhs) < 2:
        return None
    lhs_sd = sample_sd(lhs)
    rhs_sd = sample_sd(rhs)
    if lhs_sd is None or rhs_sd is None:
        return None
    stderr = math.sqrt(lhs_sd * lhs_sd / len(lhs) + rhs_sd * rhs_sd / len(rhs))
    if stderr <= 0.0:
        return None
    return delta_ms / stderr


def budget_rows(audit: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = audit.get(key)
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def budget_group_rows(audit: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    key = (
        "tail_graph_group_budget"
        if scope == "tail"
        else "required_image_graph_group_budget"
    )
    return budget_rows(audit, key)


def budget_stage_rows(audit: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    key = (
        "tail_graph_stage_budget"
        if scope == "tail"
        else "required_image_graph_stage_budget"
    )
    return budget_rows(audit, key)


def budget_by_group(audit: dict[str, Any], scope: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in budget_group_rows(audit, scope):
        group = row.get("group")
        if isinstance(group, str):
            result[group] = row
    return result


def budget_by_stage(audit: dict[str, Any], scope: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in budget_stage_rows(audit, scope):
        stage = row.get("stage")
        if isinstance(stage, str):
            result[stage] = row
    return result


def budget_index(audit: dict[str, Any], scope: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "stage": budget_by_stage(audit, scope),
        "group": budget_by_group(audit, scope),
    }


def combined_budget(
    matched: list[dict[str, Any]],
    *,
    basis: str,
    names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "budget_basis": basis,
        "budget_names": [str(name) for name in names],
        "budget_required_ms": sum(
            number(row.get("required_budget_ms")) or 0.0 for row in matched
        ),
        "budget_required_e2e_pct": sum(
            number(row.get("required_e2e_pct")) or 0.0 for row in matched
        ),
        "budget_profile_sum_ms": sum(
            number(row.get("profile_sum_ms")) or 0.0 for row in matched
        ),
        "budget_profile_count": sum(int(row.get("count") or 0) for row in matched),
    }


def stage_budget_projection(
    stage_name: str | None,
    budget: dict[str, dict[str, dict[str, Any]]] | None,
) -> dict[str, Any]:
    if stage_name is None or budget is None:
        return {}

    stage_budget = budget.get("stage", {})
    stages = STAGE_BUDGET_STAGES.get(stage_name)
    if stages:
        matched_stages = [
            row for stage in stages if (row := stage_budget.get(stage)) is not None
        ]
        if len(matched_stages) == len(stages):
            return combined_budget(matched_stages, basis="stage", names=stages)

    group_budget = budget.get("group", {})
    groups = STAGE_BUDGET_GROUPS.get(stage_name)
    if not groups:
        return {
            "e2e_budget_projection": None,
            "e2e_budget_projection_reason": "no direct atomic stage-to-budget mapping",
        }

    matched: list[dict[str, Any]] = []
    for group in groups:
        row = group_budget.get(group)
        if row is not None:
            matched.append(row)
    if not matched:
        missing = stages if stages else groups
        return {
            "e2e_budget_projection": None,
            "e2e_budget_projection_reason": "mapped budget stage/group missing from audit",
            "budget_names": list(missing),
        }

    return combined_budget(matched, basis="group", names=groups)


def resolve_summary_path(raw: str) -> tuple[str, Path]:
    label: str | None = None
    path_text = raw
    if "=" in raw:
        maybe_label, maybe_path = raw.split("=", 1)
        if maybe_label and maybe_path:
            label = maybe_label
            path_text = maybe_path
    path = Path(path_text)
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} does not exist")
    return label or summary_path.parent.name, summary_path


def select_row(summary: dict[str, Any], stage: int | None, stage_name: str | None) -> dict[str, Any]:
    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("summary has no rows list")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if stage is not None and row.get("stage") != stage:
            continue
        if stage_name is not None and row.get("stage_name") != stage_name and row.get("name") != stage_name:
            continue
        selected.append(row)
    if not selected:
        raise SystemExit(f"no matching row for stage={stage!r} stage_name={stage_name!r}")
    if len(selected) > 1:
        raise SystemExit("multiple rows matched; pass --stage or --stage-name")
    return selected[0]


def per_frame_samples(row: dict[str, Any]) -> list[float]:
    direct = row.get("samples_ms_per_frame")
    if isinstance(direct, list) and all(isinstance(value, int | float) for value in direct):
        return [float(value) for value in direct]
    samples = row.get("samples_ms")
    if not isinstance(samples, list):
        return []
    batch_size = row.get("batch_size")
    divisor = float(batch_size) if isinstance(batch_size, int | float) and batch_size > 0 else 1.0
    return [float(value) / divisor for value in samples if isinstance(value, int | float)]


def summarize_samples(samples: list[float], drop_initial: int) -> dict[str, Any]:
    steady = samples[drop_initial:] if len(samples) > drop_initial else samples
    return {
        "n": len(samples),
        "steady_n": len(steady),
        "raw_mean_ms": mean(samples),
        "raw_median_ms": median(samples),
        "raw_sd_ms": sample_sd(samples),
        "raw_ci95_ms": ci95(samples),
        "steady_mean_ms": mean(steady),
        "steady_median_ms": median(steady),
        "steady_sd_ms": sample_sd(steady),
        "steady_ci95_ms": ci95(steady),
        "drop_initial_samples": drop_initial,
        "samples_ms": samples,
        "steady_samples_ms": steady,
    }


def decision(delta_ms: float | None, median_delta_ms: float | None, sig: float | None) -> str:
    if delta_ms is None or median_delta_ms is None or sig is None:
        return "insufficient-data"
    if delta_ms < 0.0 and median_delta_ms < 0.0 and sig <= -2.0:
        return "candidate"
    if delta_ms > 0.0 and median_delta_ms > 0.0 and sig >= 2.0:
        return "reject/slower"
    return "reject/no-signal"


def build_rows(
    inputs: list[str],
    *,
    baseline_label: str,
    stage: int | None,
    stage_name: str | None,
    drop_initial: int,
    e2e_budget: dict[str, dict[str, dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in inputs:
        label, summary_path = resolve_summary_path(raw)
        summary = load_json(summary_path)
        row = select_row(summary, stage, stage_name)
        samples = per_frame_samples(row)
        if not samples:
            mean_ms = number(row.get("mean_ms_per_frame")) or number(row.get("mean_ms"))
            if mean_ms is not None:
                samples = [mean_ms]
        stat = summarize_samples(samples, drop_initial)
        rows.append(
            {
                "variant": label,
                "summary": str(summary_path),
                "stage": row.get("stage"),
                "stage_name": row.get("stage_name") or row.get("name"),
                "input_ne": row.get("input_ne"),
                "output_ne": row.get("output_ne"),
                **stage_budget_projection(
                    str(row.get("stage_name") or row.get("name"))
                    if row.get("stage_name") or row.get("name")
                    else None,
                    e2e_budget,
                ),
                **stat,
            }
        )

    baseline = next((row for row in rows if row["variant"] == baseline_label), None)
    if baseline is None:
        labels = ", ".join(str(row["variant"]) for row in rows)
        raise SystemExit(f"baseline {baseline_label!r} not found in variants: {labels}")

    baseline_steady = baseline["steady_samples_ms"]
    baseline_mean = number(baseline.get("steady_mean_ms"))
    baseline_median = number(baseline.get("steady_median_ms"))
    for row in rows:
        row_mean = number(row.get("steady_mean_ms"))
        row_median = number(row.get("steady_median_ms"))
        delta_ms = row_mean - baseline_mean if row_mean is not None and baseline_mean is not None else None
        median_delta_ms = (
            row_median - baseline_median
            if row_median is not None and baseline_median is not None
            else None
        )
        sig = signal(delta_ms, row["steady_samples_ms"], baseline_steady)
        row["steady_mean_delta_ms"] = delta_ms
        row["steady_median_delta_ms"] = median_delta_ms
        row["steady_mean_signal"] = sig
        budget_required = number(row.get("budget_required_ms"))
        if (
            delta_ms is not None
            and baseline_mean is not None
            and baseline_mean > 0.0
            and budget_required is not None
        ):
            row["steady_relative_delta_pct"] = delta_ms / baseline_mean * 100.0
            row["projected_required_e2e_delta_ms"] = (
                budget_required * delta_ms / baseline_mean
            )
        else:
            row["steady_relative_delta_pct"] = None
            row["projected_required_e2e_delta_ms"] = None
        row["decision"] = "baseline" if row is baseline else decision(delta_ms, median_delta_ms, sig)
    return rows


def markdown(rows: list[dict[str, Any]], baseline_label: str, budget_scope: str | None) -> str:
    budget_note = (
        f" Projected E2E deltas scale the local steady mean delta by the `{budget_scope}` "
        "required graph budget stage from the audit when available, otherwise by its group."
        if budget_scope is not None
        else ""
    )
    lines = [
        "# SAM3 ViT Isolated Stage Variant A/B",
        "",
        f"Baseline: `{baseline_label}`. `steady` metrics drop the configured initial timed samples.{budget_note}",
        "",
        "| variant | stage | raw mean | steady mean | steady median | delta mean | delta median | signal | budget basis | budget | projected E2E delta | ci95 | n | decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | `{row.get('stage_name')}` | "
            f"{fmt_ms(number(row.get('raw_mean_ms')))} | "
            f"{fmt_ms(number(row.get('steady_mean_ms')))} | "
            f"{fmt_ms(number(row.get('steady_median_ms')))} | "
            f"{fmt_ms(number(row.get('steady_mean_delta_ms')))} | "
            f"{fmt_ms(number(row.get('steady_median_delta_ms')))} | "
            f"{fmt_signal(number(row.get('steady_mean_signal')))} | "
            f"`{row.get('budget_basis') or '-'}` | "
            f"{fmt_ms(number(row.get('budget_required_ms')))} | "
            f"{fmt_ms(number(row.get('projected_required_e2e_delta_ms')))} | "
            f"{fmt_ms(number(row.get('steady_ci95_ms')))} | "
            f"{row.get('steady_n')} | `{row.get('decision')}` |"
        )
    unmapped = [
        row
        for row in rows
        if row.get("e2e_budget_projection") is None
        and row.get("e2e_budget_projection_reason")
    ]
    if unmapped:
        reason = unmapped[0].get("e2e_budget_projection_reason")
        lines.extend(
            [
                "",
                f"E2E projection omitted for this stage: {reason}.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="summary.json files or variant directories; label=path is supported")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--stage", type=int)
    parser.add_argument("--stage-name")
    parser.add_argument("--drop-initial-samples", type=int, default=1)
    parser.add_argument(
        "--e2e-audit",
        type=Path,
        help="optimization_targets.json used to project isolated local deltas onto required E2E budget",
    )
    parser.add_argument(
        "--budget-scope",
        choices=["tail", "required-image"],
        default="tail",
        help="which required graph budget table from --e2e-audit to use",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    e2e_budget = budget_index(load_json(args.e2e_audit), args.budget_scope) if args.e2e_audit else None
    rows = build_rows(
        args.inputs,
        baseline_label=args.baseline,
        stage=args.stage,
        stage_name=args.stage_name,
        drop_initial=max(args.drop_initial_samples, 0),
        e2e_budget=e2e_budget,
    )
    result = {
        "baseline": args.baseline,
        "stage": args.stage,
        "stage_name": args.stage_name,
        "drop_initial_samples": max(args.drop_initial_samples, 0),
        "e2e_audit": str(args.e2e_audit) if args.e2e_audit else None,
        "budget_scope": args.budget_scope if args.e2e_audit else None,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(
            markdown(rows, args.baseline, args.budget_scope if args.e2e_audit else None),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
