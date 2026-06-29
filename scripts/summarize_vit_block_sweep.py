# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def samples_per_frame(row: dict[str, Any]) -> list[float]:
    samples = row.get("samples_ms_per_frame")
    if isinstance(samples, list):
        return [float(value) for value in samples if isinstance(value, int | float)]
    samples_ms = row.get("samples_ms")
    batch_size = row.get("batch_size")
    if isinstance(samples_ms, list) and isinstance(batch_size, int | float) and batch_size > 0:
        return [
            float(value) / float(batch_size)
            for value in samples_ms
            if isinstance(value, int | float)
        ]
    return []


def summarize_row(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    samples = samples_per_frame(row)
    mean_ms = statistics.fmean(samples) if samples else float(row.get("mean_ms_per_frame", 0.0))
    median_ms = statistics.median(samples) if samples else mean_ms
    sd_ms = statistics.stdev(samples) if len(samples) > 1 else None
    return {
        "source": str(path),
        "model": row.get("model"),
        "backend": row.get("backend"),
        "batch_size": row.get("batch_size"),
        "vit_blocks": row.get("vit_blocks"),
        "include_tracker_neck": bool(row.get("include_tracker_neck")),
        "repeats": row.get("repeats"),
        "mean_ms_per_frame": mean_ms,
        "median_ms_per_frame": median_ms,
        "sd_ms_per_frame": sd_ms,
        "min_ms_per_frame": min(samples) if samples else None,
        "max_ms_per_frame": max(samples) if samples else None,
        "output_ne": row.get("output_ne"),
    }


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        row = load_json(path)
        if not isinstance(row, dict):
            continue
        rows.append(summarize_row(path, row))
    return rows


def add_deltas(rows: list[dict[str, Any]]) -> None:
    no_neck = sorted(
        (
            row
            for row in rows
            if not row["include_tracker_neck"] and isinstance(row.get("vit_blocks"), int)
        ),
        key=lambda row: row["vit_blocks"],
    )
    previous: dict[str, Any] | None = None
    for row in no_neck:
        if previous is None:
            row["delta_label"] = "prefix"
            row["delta_mean_ms"] = row["mean_ms_per_frame"]
            row["delta_median_ms"] = row["median_ms_per_frame"]
        else:
            prev_blocks = int(previous["vit_blocks"])
            cur_blocks = int(row["vit_blocks"])
            if cur_blocks == prev_blocks + 1:
                row["delta_label"] = f"block_{cur_blocks - 1:02d}"
            else:
                row["delta_label"] = f"blocks_{prev_blocks:02d}_{cur_blocks - 1:02d}"
            row["delta_mean_ms"] = row["mean_ms_per_frame"] - previous["mean_ms_per_frame"]
            row["delta_median_ms"] = row["median_ms_per_frame"] - previous["median_ms_per_frame"]
        previous = row

    full_no_neck = next(
        (row for row in no_neck if row.get("vit_blocks") == 32),
        no_neck[-1] if no_neck else None,
    )
    for row in rows:
        if row["include_tracker_neck"] and full_no_neck is not None:
            row["delta_label"] = "tracker_neck"
            row["delta_mean_ms"] = row["mean_ms_per_frame"] - full_no_neck["mean_ms_per_frame"]
            row["delta_median_ms"] = (
                row["median_ms_per_frame"] - full_no_neck["median_ms_per_frame"]
            )


def fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def output_shape(row: dict[str, Any]) -> str:
    value = row.get("output_ne")
    if not isinstance(value, list):
        return "-"
    return "x".join(str(part) for part in value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            1 if row["include_tracker_neck"] else 0,
            row.get("vit_blocks") if isinstance(row.get("vit_blocks"), int) else 999,
        ),
    )
    lines = [
        "# SAM3 ViT Block Sweep",
        "",
        "Cumulative rows keep the image-encoder output contract stable. "
        "`delta` is the mean/median increment from the previous cumulative row; "
        "the tracker-neck row is the increment over full ViT without tracker neck.",
        "",
        "| blocks | tracker neck | delta label | mean/frame ms | median/frame ms | "
        "delta mean | delta median | sd | output |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ordered:
        lines.append(
            f"| {row.get('vit_blocks')} | `{row['include_tracker_neck']}` | "
            f"`{row.get('delta_label', '-')}` | {fmt(row.get('mean_ms_per_frame'))} | "
            f"{fmt(row.get('median_ms_per_frame'))} | {fmt(row.get('delta_mean_ms'))} | "
            f"{fmt(row.get('delta_median_ms'))} | {fmt(row.get('sd_ms_per_frame'))} | "
            f"`{output_shape(row)}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.input_dir)
    add_deltas(rows)
    payload = {
        "input_dir": str(args.input_dir),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown_out, rows)


if __name__ == "__main__":
    main()
