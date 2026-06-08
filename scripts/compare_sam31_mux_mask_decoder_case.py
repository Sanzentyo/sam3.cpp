#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.24",
# ]
# ///
"""Compare C++ SAM3.1 multiplex mask-decoder slice dumps against Python reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TENSORS = ("masks", "iou_pred", "mask_tokens_out", "object_score_logits")


def read_shape(path: Path) -> tuple[int, ...]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty shape file: {path}")
    parts = text.replace(",", " ").split()
    return tuple(int(part) for part in parts)


def read_tensor(prefix: Path) -> np.ndarray:
    shape = read_shape(prefix.with_suffix(".shape"))
    data = np.fromfile(prefix.with_suffix(".bin"), dtype=np.float32)
    expected = int(np.prod(shape, dtype=np.int64))
    if data.size != expected:
        raise ValueError(f"{prefix}: expected {expected} values for shape {shape}, got {data.size}")
    return data.reshape(shape, order="F")


def compare_tensor(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
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
    return {
        "status": "ok",
        "shape": list(expected.shape),
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean() if abs_diff.size else 0.0),
        "rmse": float(np.sqrt(np.mean(diff * diff)) if diff.size else 0.0),
        "max_expected_abs": float(np.abs(expected).max(initial=0.0)),
        "max_actual_abs": float(np.abs(actual).max(initial=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, help="expected_cpp_layout directory")
    parser.add_argument("--actual", required=True, help="C++ dump directory")
    parser.add_argument("--out", required=True, help="summary JSON path")
    args = parser.parse_args()

    expected_dir = Path(args.expected)
    actual_dir = Path(args.actual)
    summary = {"tensors": {}, "status": "ok"}

    for name in TENSORS:
        expected = read_tensor(expected_dir / name)
        actual = read_tensor(actual_dir / name)
        item = compare_tensor(expected, actual)
        summary["tensors"][name] = item
        if item["status"] != "ok":
            summary["status"] = "failed"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
