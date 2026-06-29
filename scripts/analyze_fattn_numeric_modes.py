#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
# ]
# ///
"""Compare real FATTN Q/K/V dumps against simple attention numeric modes."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DiffStats:
    count: int = 0
    max_abs: float = 0.0
    sum_abs: float = 0.0
    sum_sq: float = 0.0

    def update(self, got: np.ndarray, ref: np.ndarray) -> None:
        diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
        self.count += int(diff.size)
        self.max_abs = max(self.max_abs, float(diff.max(initial=0.0)))
        self.sum_abs += float(diff.sum(dtype=np.float64))
        self.sum_sq += float(np.square(diff, dtype=np.float32).sum(dtype=np.float64))

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "max_abs": 0.0, "mean_abs": 0.0, "rmse": 0.0}
        return {
            "count": self.count,
            "max_abs": self.max_abs,
            "mean_abs": self.sum_abs / self.count,
            "rmse": math.sqrt(self.sum_sq / self.count),
        }


def read_shape(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in path.read_text().strip().split(",") if part)


def read_dnh(path: Path, shape_path: Path) -> np.ndarray:
    d, n, h = read_shape(shape_path)
    flat = np.fromfile(path, dtype=np.float32)
    expected = d * n * h
    if flat.size != expected:
        raise ValueError(f"{path} has {flat.size} floats, expected {expected}")
    return flat.reshape(h, n, d)


def read_dhn_as_nhd(path: Path, shape_path: Path) -> np.ndarray:
    d, h, n = read_shape(shape_path)
    flat = np.fromfile(path, dtype=np.float32)
    expected = d * h * n
    if flat.size != expected:
        raise ValueError(f"{path} has {flat.size} floats, expected {expected}")
    return flat.reshape(n, h, d)


def softmax(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(centered, dtype=np.float32)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_attention_chunk(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    query_start: int,
    query_end: int,
    mode: str,
) -> np.ndarray:
    d = q.shape[1]
    scale = np.float32(1.0 / math.sqrt(d))
    q_chunk = q[query_start:query_end]

    if mode == "f32":
        scores = (q_chunk @ k.T) * scale
        return (softmax(scores.astype(np.float32)) @ v).astype(np.float32)

    if mode == "qkv_f16_f32_math":
        qh = q_chunk.astype(np.float16).astype(np.float32)
        kh = k.astype(np.float16).astype(np.float32)
        vh = v.astype(np.float16).astype(np.float32)
        scores = (qh @ kh.T) * scale
        return (softmax(scores.astype(np.float32)) @ vh).astype(np.float32)

    if mode == "qk_f16_v_f32_math":
        qh = q_chunk.astype(np.float16).astype(np.float32)
        kh = k.astype(np.float16).astype(np.float32)
        scores = (qh @ kh.T) * scale
        return (softmax(scores.astype(np.float32)) @ v).astype(np.float32)

    if mode == "q_f16_k_f32_v_f32_math":
        qh = q_chunk.astype(np.float16).astype(np.float32)
        scores = (qh @ k.T) * scale
        return (softmax(scores.astype(np.float32)) @ v).astype(np.float32)

    if mode == "q_f32_k_f16_v_f32_math":
        kh = k.astype(np.float16).astype(np.float32)
        scores = (q_chunk @ kh.T) * scale
        return (softmax(scores.astype(np.float32)) @ v).astype(np.float32)

    if mode == "q_f32_k_f32_v_f16_math":
        vh = v.astype(np.float16).astype(np.float32)
        scores = (q_chunk @ k.T) * scale
        return (softmax(scores.astype(np.float32)) @ vh).astype(np.float32)

    if mode == "qk_f32_v_f16_math":
        vh = v.astype(np.float16).astype(np.float32)
        scores = (q_chunk @ k.T) * scale
        return (softmax(scores.astype(np.float32)) @ vh).astype(np.float32)

    raise ValueError(f"unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("outputs/sam31-prop-d32-real-snap-20260611"),
        help="Directory with default/ and mma_all/ propagation tensor dumps.",
    )
    parser.add_argument("--prefix", default="sam31_mem_attn_layer0_sa_fattn")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_dir = args.dump_dir / "default"
    mma_dir = args.dump_dir / "mma_all"
    prefix = args.prefix

    q = read_dnh(default_dir / f"{prefix}_q_dump.bin", default_dir / f"{prefix}_q_dump.shape")
    k = read_dnh(default_dir / f"{prefix}_k_dump.bin", default_dir / f"{prefix}_k_dump.shape")
    v = read_dnh(default_dir / f"{prefix}_v_dump.bin", default_dir / f"{prefix}_v_dump.shape")
    tile = read_dhn_as_nhd(default_dir / f"{prefix}_dump.bin", default_dir / f"{prefix}_dump.shape")
    mma = read_dhn_as_nhd(mma_dir / f"{prefix}_dump.bin", mma_dir / f"{prefix}_dump.shape")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Q/K/V shapes differ: {q.shape} {k.shape} {v.shape}")
    h_count, n_tokens, d_head = q.shape
    if tile.shape != (n_tokens, h_count, d_head) or mma.shape != tile.shape:
        raise ValueError(f"output shape mismatch: tile={tile.shape} mma={mma.shape}")

    report: dict[str, Any] = {
        "dump_dir": str(args.dump_dir),
        "prefix": prefix,
        "shape": {"d": d_head, "n": n_tokens, "heads": h_count},
        "chunk": args.chunk,
        "modes": {},
        "tile_vs_mma": {},
    }

    tile_vs_mma = DiffStats()
    tile_vs_mma.update(tile, mma)
    report["tile_vs_mma"] = tile_vs_mma.as_dict()

    modes = (
        "f32",
        "qkv_f16_f32_math",
        "qk_f16_v_f32_math",
        "q_f16_k_f32_v_f32_math",
        "q_f32_k_f16_v_f32_math",
        "q_f32_k_f32_v_f16_math",
        "qk_f32_v_f16_math",
    )
    for mode in modes:
        vs_tile = DiffStats()
        vs_mma = DiffStats()
        for h in range(h_count):
            for start in range(0, n_tokens, args.chunk):
                end = min(start + args.chunk, n_tokens)
                pred = compute_attention_chunk(q[h], k[h], v[h], start, end, mode)
                vs_tile.update(pred, tile[start:end, h, :])
                vs_mma.update(pred, mma[start:end, h, :])
        report["modes"][mode] = {
            "vs_tile": vs_tile.as_dict(),
            "vs_mma": vs_mma.as_dict(),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
