# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

STAGE_NAMES = {
    0: "norm1",
    1: "window_part",
    2: "qkv_proj",
    3: "qkv_layout",
    4: "qkv_rope",
    5: "attn_core",
    6: "attn_proj",
    7: "window_unpart",
    8: "norm2",
    9: "mlp_fc1",
    10: "mlp_gelu",
    11: "mlp_fc2",
    12: "mlp",
    13: "block",
    14: "mlp_fc1_gelu",
}

STAGE_ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)


def load_json(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def output_shape(row: dict[str, Any]) -> str:
    ne = row.get("output_ne")
    if not isinstance(ne, list):
        return "-"
    return "x".join(str(v) for v in ne)


def input_shape(row: dict[str, Any]) -> str:
    ne = row.get("input_ne")
    if not isinstance(ne, list):
        return "-"
    return "x".join(str(v) for v in ne)


def per_frame_ms(row: dict[str, Any], key: str) -> float | None:
    direct = number(row.get(f"{key}_per_frame"))
    if direct is not None:
        return direct
    value = number(row.get(key))
    if value is None:
        return None
    batch_size = row.get("batch_size")
    divisor = (
        float(batch_size)
        if isinstance(batch_size, int | float) and batch_size > 0
        else 1.0
    )
    return value / divisor


def sample_count(row: dict[str, Any]) -> int | None:
    repeats = row.get("repeats")
    if isinstance(repeats, int | float) and repeats > 0:
        return int(repeats)
    samples = row.get("samples_ms")
    if isinstance(samples, list) and samples:
        return len(samples)
    return None


def delta_signal(lhs: dict[str, Any], rhs: dict[str, Any], delta_ms: float | None) -> float | None:
    if delta_ms is None:
        return None
    lhs_sd = per_frame_ms(lhs, "sd_ms")
    rhs_sd = per_frame_ms(rhs, "sd_ms")
    lhs_n = sample_count(lhs)
    rhs_n = sample_count(rhs)
    if lhs_sd is None or rhs_sd is None or lhs_n is None or rhs_n is None:
        return None
    if lhs_n <= 0 or rhs_n <= 0:
        return None
    stderr = math.sqrt(lhs_sd * lhs_sd / lhs_n + rhs_sd * rhs_sd / rhs_n)
    if stderr <= 0.0:
        return None
    return delta_ms / stderr


def stderr_tail(path: Path, max_lines: int = 6) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def collect_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_stage: dict[int, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("stage_*.json")):
        row = load_json(path)
        stage_text = path.stem.removeprefix("stage_")
        stage = int(stage_text) if stage_text.isdigit() else None
        if row is None:
            skipped.append(
                {
                    "stage": stage,
                    "name": STAGE_NAMES.get(stage, f"stage_{stage}") if stage is not None else None,
                    "source": str(path),
                    "stderr_tail": stderr_tail(path.with_suffix(".stderr")),
                }
            )
            continue
        stage = row.get("block_stage")
        if stage is None:
            stage = row.get("block0_stage")
        if not isinstance(stage, int):
            continue
        stage_name = STAGE_NAMES.get(stage, f"stage_{stage}")
        row["stage"] = stage
        row["name"] = stage_name
        row["stage_name"] = stage_name
        row["source"] = str(path)
        rows_by_stage[stage] = row

    isolated = any(
        row.get("bench_mode") == "isolated_stage" or "input_prefix" in row
        for row in rows_by_stage.values()
    )
    rows: list[dict[str, Any]] = []
    previous_row: dict[str, Any] | None = None
    previous_mean: float | None = None
    previous_median: float | None = None
    for stage in STAGE_ORDER:
        row = rows_by_stage.get(stage)
        if row is None:
            continue
        mean_ms = per_frame_ms(row, "mean_ms")
        median_ms = per_frame_ms(row, "median_ms")
        row["mean_ms_per_frame"] = mean_ms
        row["median_ms_per_frame"] = median_ms
        row["sd_ms_per_frame"] = per_frame_ms(row, "sd_ms")
        row["ci95_ms_per_frame"] = per_frame_ms(row, "ci95_ms")
        row["bench_mode"] = "isolated_stage" if isolated else "cumulative_stop"
        if isolated:
            row["delta_mean_ms"] = None
            row["delta_median_ms"] = None
            row["delta_mean_signal"] = None
        else:
            row["delta_mean_ms"] = (
                mean_ms - previous_mean
                if mean_ms is not None and previous_mean is not None and stage != 14
                else None
            )
            row["delta_median_ms"] = (
                median_ms - previous_median
                if median_ms is not None and previous_median is not None and stage != 14
                else None
            )
            row["delta_mean_signal"] = (
                delta_signal(row, previous_row, row["delta_mean_ms"])
                if previous_row is not None and stage != 14
                else None
            )
        rows.append(row)
        if stage != 14:
            previous_row = row
            previous_mean = mean_ms
            previous_median = median_ms

    if isolated and 14 in rows_by_stage and 9 in rows_by_stage and 10 in rows_by_stage:
        fc1_gelu = rows_by_stage[14]
        fc1_mean = per_frame_ms(rows_by_stage[9], "mean_ms")
        gelu_mean = per_frame_ms(rows_by_stage[10], "mean_ms")
        fc1_gelu_mean = per_frame_ms(fc1_gelu, "mean_ms")
        fc1_median = per_frame_ms(rows_by_stage[9], "median_ms")
        gelu_median = per_frame_ms(rows_by_stage[10], "median_ms")
        fc1_gelu_median = per_frame_ms(fc1_gelu, "median_ms")
        fc1_gelu["split_fc1_plus_gelu_mean_ms"] = (
            fc1_mean + gelu_mean if fc1_mean is not None and gelu_mean is not None else None
        )
        fc1_gelu["split_fc1_plus_gelu_median_ms"] = (
            fc1_median + gelu_median
            if fc1_median is not None and gelu_median is not None
            else None
        )
        split_mean = number(fc1_gelu.get("split_fc1_plus_gelu_mean_ms"))
        split_median = number(fc1_gelu.get("split_fc1_plus_gelu_median_ms"))
        fc1_gelu["delta_from_split_mean_ms"] = (
            fc1_gelu_mean - split_mean
            if fc1_gelu_mean is not None and split_mean is not None
            else None
        )
        fc1_gelu["delta_from_split_median_ms"] = (
            fc1_gelu_median - split_median
            if fc1_gelu_median is not None and split_median is not None
            else None
        )
        if 3 in rows_by_stage and 4 in rows_by_stage:
            qkv_layout_mean = per_frame_ms(rows_by_stage[3], "mean_ms")
            qkv_rope_mean = per_frame_ms(rows_by_stage[4], "mean_ms")
            rows_by_stage[4]["derived_rope_mean_ms"] = (
                qkv_rope_mean - qkv_layout_mean
                if qkv_rope_mean is not None and qkv_layout_mean is not None
                else None
            )
        if 4 in rows_by_stage and 5 in rows_by_stage:
            qkv_rope_mean = per_frame_ms(rows_by_stage[4], "mean_ms")
            attn_core_mean = per_frame_ms(rows_by_stage[5], "mean_ms")
            rows_by_stage[5]["derived_attn_core_mean_ms"] = (
                attn_core_mean - qkv_rope_mean
                if attn_core_mean is not None and qkv_rope_mean is not None
                else None
            )
    elif 14 in rows_by_stage and 8 in rows_by_stage:
        fc1_gelu = rows_by_stage[14]
        norm2 = rows_by_stage[8]
        fc1_gelu_mean = per_frame_ms(fc1_gelu, "mean_ms")
        fc1_gelu_median = per_frame_ms(fc1_gelu, "median_ms")
        norm2_mean = per_frame_ms(norm2, "mean_ms")
        norm2_median = per_frame_ms(norm2, "median_ms")
        fc1_gelu["delta_from_norm2_mean_ms"] = (
            fc1_gelu_mean - norm2_mean
            if fc1_gelu_mean is not None and norm2_mean is not None
            else None
        )
        fc1_gelu["delta_from_norm2_median_ms"] = (
            fc1_gelu_median - norm2_median
            if fc1_gelu_median is not None and norm2_median is not None
            else None
        )
        fc1_gelu["delta_from_norm2_signal"] = delta_signal(
            fc1_gelu, norm2, fc1_gelu.get("delta_from_norm2_mean_ms")
        )

    return rows, skipped


def markdown(rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> str:
    isolated = any(row.get("bench_mode") == "isolated_stage" for row in rows)
    block_indices = {
        row.get("block_stage_index")
        for row in rows
        if isinstance(row.get("block_stage_index"), int)
    }
    block_label = (
        f"Block {next(iter(block_indices))}"
        if len(block_indices) == 1
        else "Selected Block"
    )
    title_mode = "Isolated Stage" if isolated else "Cumulative Stop-Point"
    lines = [f"# SAM3 ViT {block_label} {title_mode} Sweep", ""]
    if isolated:
        lines.extend(
            [
                "Each row runs an independent graph from the captured E2E tensor input. "
                "Rows such as `qkv_rope`, `attn_core`, `mlp`, `block`, and "
                "`mlp_fc1_gelu` are composite stop-points; use the derived notes below "
                "when a local sub-cost is needed.",
                "",
                "| block | stage | name | mean ms | median ms | sd | ci95 | input | output |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| block | stage | name | mean/frame ms | median/frame ms | sd | ci95 | delta mean | delta signal | output |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    for row in rows:
        prefix = (
            f"| {row.get('block_stage_index', 0)} | {row.get('stage')} | "
            f"`{row.get('stage_name')}` | "
            f"{fmt_ms(per_frame_ms(row, 'mean_ms'))} | "
            f"{fmt_ms(per_frame_ms(row, 'median_ms'))} | "
            f"{fmt_ms(number(row.get('sd_ms_per_frame')))} | "
            f"{fmt_ms(number(row.get('ci95_ms_per_frame')))} | "
        )
        if isolated:
            lines.append(prefix + f"`{input_shape(row)}` | `{output_shape(row)}` |")
        else:
            lines.append(
                prefix
                + f"{fmt_ms(number(row.get('delta_mean_ms')))} | "
                f"{fmt_ms(number(row.get('delta_mean_signal')))} | "
                f"`{output_shape(row)}` |"
            )

    fc1_gelu = next((row for row in rows if row.get("stage") == 14), None)
    if fc1_gelu is not None:
        if isolated:
            lines.extend(
                [
                    "",
                    "`mlp_fc1_gelu` runs FC1+GELU in one isolated graph. "
                    f"Separate `mlp_fc1 + mlp_gelu` mean is "
                    f"`{fmt_ms(number(fc1_gelu.get('split_fc1_plus_gelu_mean_ms')))} ms`; "
                    f"direct mean delta is "
                    f"`{fmt_ms(number(fc1_gelu.get('delta_from_split_mean_ms')))} ms`.",
                ]
            )
            qkv_rope = next((row for row in rows if row.get("stage") == 4), None)
            attn_core = next((row for row in rows if row.get("stage") == 5), None)
            if qkv_rope is not None or attn_core is not None:
                lines.append("")
                lines.append("Derived local estimates:")
                if qkv_rope is not None:
                    lines.append(
                        f"- `qkv_rope - qkv_layout`: "
                        f"`{fmt_ms(number(qkv_rope.get('derived_rope_mean_ms')))} ms`"
                    )
                if attn_core is not None:
                    lines.append(
                        f"- `attn_core - qkv_rope`: "
                        f"`{fmt_ms(number(attn_core.get('derived_attn_core_mean_ms')))} ms`"
                    )
        else:
            lines.extend(
                [
                    "",
                    "`mlp_fc1_gelu` is an alternate fused-boundary stop. Its direct "
                    f"delta from `norm2` is mean `{fmt_ms(number(fc1_gelu.get('delta_from_norm2_mean_ms')))} ms`, "
                    f"median `{fmt_ms(number(fc1_gelu.get('delta_from_norm2_median_ms')))} ms`, "
                    f"signal `{fmt_ms(number(fc1_gelu.get('delta_from_norm2_signal')))}`.",
                ]
            )
    if skipped:
        lines.extend(["", "Skipped stages:"])
        for row in skipped:
            tail = " / ".join(line for line in row.get("stderr_tail", []) if line)
            lines.append(
                f"- stage `{row.get('stage')}` `{row.get('name')}`: "
                f"{tail if tail else 'no JSON output'}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    rows, skipped = collect_rows(args.input_dir)
    result = {"input_dir": str(args.input_dir), "rows": rows, "skipped": skipped}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))
    if args.markdown_out:
        args.markdown_out.write_text(markdown(rows, skipped), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
