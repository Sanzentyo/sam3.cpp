#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
# ]
# ///
"""Compare SAM3 ViT MLP fused-contract raw f32 tensor dumps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_shape(path: Path) -> tuple[int, ...]:
    return tuple(
        int(part) for part in path.read_text(encoding="utf-8").replace(",", " ").split()
    )


def read_tensor(prefix: Path) -> np.ndarray:
    shape = read_shape(prefix.with_suffix(".shape"))
    values = np.fromfile(prefix.with_suffix(".bin"), dtype=np.float32)
    expected = int(np.prod(shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(
            f"{prefix}: expected {expected} values for shape {shape}, got {values.size}"
        )
    return values.reshape(shape, order="F")


def bf16_round_f32(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def stats(
    expected: np.ndarray, actual: np.ndarray, *, bf16_round_actual: bool
) -> dict[str, object]:
    if expected.shape != actual.shape:
        return {
            "status": "shape_mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }
    if bf16_round_actual:
        actual = bf16_round_f32(actual)
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    abs_diff = np.abs(diff)
    flat_equal = actual == expected
    return {
        "status": "ok",
        "shape": list(expected.shape),
        "bf16_round_actual": bf16_round_actual,
        "equal_values": int(np.count_nonzero(flat_equal)),
        "total_values": int(flat_equal.size),
        "equal_ratio": float(np.count_nonzero(flat_equal) / flat_equal.size)
        if flat_equal.size
        else 1.0,
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean() if abs_diff.size else 0.0),
        "rmse": float(np.sqrt(np.mean(diff * diff)) if diff.size else 0.0),
        "p50_abs": float(np.percentile(abs_diff, 50)) if abs_diff.size else 0.0,
        "p95_abs": float(np.percentile(abs_diff, 95)) if abs_diff.size else 0.0,
        "p99_abs": float(np.percentile(abs_diff, 99)) if abs_diff.size else 0.0,
    }


def parse_run_log(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    records = [
        json.loads(match.group(0))
        for match in re.finditer(r"\{[^{}]*\"mean_ms\"[^{}]*\}", text)
    ]
    cudnn_success = {
        "f32": text.count("GGML_CUDA_CUDNN_MLP_FC1_GELU_F32 success"),
        "bf16": text.count("GGML_CUDA_CUDNN_MLP_FC1_GELU_BF16 success"),
    }
    return {
        "timing": records[-1] if records else None,
        "cudnn_success": cudnn_success,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--actual",
        action="append",
        required=True,
        help="LABEL=PREFIX, repeatable",
    )
    args = parser.parse_args()

    expected = read_tensor(args.expected_prefix)
    summary: dict[str, object] = {
        "expected_prefix": str(args.expected_prefix),
        "comparisons": {},
    }
    comparisons: dict[str, object] = {}
    for item in args.actual:
        label, sep, prefix_text = item.partition("=")
        if not sep:
            raise ValueError(f"--actual must be LABEL=PREFIX, got {item}")
        prefix = Path(prefix_text)
        actual = read_tensor(prefix)
        run_log = prefix.parent / "run.log"
        comparisons[label] = {
            "prefix": str(prefix),
            "raw": stats(expected, actual, bf16_round_actual=False),
            "actual_rounded_to_bf16": stats(expected, actual, bf16_round_actual=True),
            "run_log": parse_run_log(run_log),
        }
    summary["comparisons"] = comparisons
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
