#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.24",
# ]
# ///
"""Compare ggml-order raw f32 tensor dumps in two directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_shape(path: Path) -> tuple[int, ...]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty shape file: {path}")
    return tuple(int(part) for part in text.replace(",", " ").split())


def read_tensor(prefix: Path) -> np.ndarray:
    shape = read_shape(prefix.with_suffix(".shape"))
    data = np.fromfile(prefix.with_suffix(".bin"), dtype=np.float32)
    expected = int(np.prod(shape, dtype=np.int64))
    if data.size != expected:
        raise ValueError(f"{prefix}: expected {expected} values for shape {shape}, got {data.size}")
    return data.reshape(shape, order="F")


def compare_tensor(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    max_abs_tolerance: float,
    mean_abs_tolerance: float,
) -> dict[str, object]:
    while expected.ndim > 0 and expected.shape[-1] == 1 and actual.ndim == expected.ndim - 1:
        expected = np.squeeze(expected, axis=-1)
    while actual.ndim > 0 and actual.shape[-1] == 1 and expected.ndim == actual.ndim - 1:
        actual = np.squeeze(actual, axis=-1)
    if expected.shape != actual.shape:
        return {
            "status": "shape_mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max(initial=0.0))
    mean_abs = float(abs_diff.mean() if abs_diff.size else 0.0)
    status = "ok"
    if max_abs > max_abs_tolerance or mean_abs > mean_abs_tolerance:
        status = "failed"
    return {
        "status": status,
        "shape": list(expected.shape),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": float(np.sqrt(np.mean(diff * diff)) if diff.size else 0.0),
        "max_expected_abs": float(np.abs(expected).max(initial=0.0)),
        "max_actual_abs": float(np.abs(actual).max(initial=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, help="expected tensor directory")
    parser.add_argument("--actual", required=True, help="actual tensor directory")
    parser.add_argument("--out", required=True, help="summary JSON path")
    parser.add_argument(
        "--tensor",
        action="append",
        required=True,
        help="tensor name to compare; repeat for multiple tensors",
    )
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--mean-abs-tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()

    expected_dir = Path(args.expected)
    actual_dir = Path(args.actual)
    summary = {
        "tensors": {},
        "status": "ok",
        "thresholds": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "mean_abs_tolerance": args.mean_abs_tolerance,
        },
    }

    for name in args.tensor:
        item = compare_tensor(
            read_tensor(expected_dir / name),
            read_tensor(actual_dir / name),
            max_abs_tolerance=args.max_abs_tolerance,
            mean_abs_tolerance=args.mean_abs_tolerance,
        )
        summary["tensors"][name] = item
        if item["status"] != "ok":
            summary["status"] = "failed"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
