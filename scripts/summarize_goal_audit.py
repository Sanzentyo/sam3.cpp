# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_comparable(summary: dict[str, Any]) -> dict[str, Any]:
    for row in summary.get("comparisons", []):
        if row.get("comparable_speed_claim"):
            return row
    raise ValueError("no comparable comparison row found")


def quality(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_mask_iou": summary.get("mean_mask_iou"),
        "min_mask_iou": summary.get("min_mask_iou"),
        "frames_compared": summary.get("frames_compared") or summary.get("frames"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-512", type=Path, required=True)
    parser.add_argument("--matrix-1024", type=Path, required=True)
    parser.add_argument("--quality-512", type=Path, required=True)
    parser.add_argument("--quality-1024", type=Path, required=True)
    parser.add_argument("--hiera-gap", type=Path, required=True)
    parser.add_argument("--hotspots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    speed_512 = first_comparable(load_json(args.matrix_512))
    speed_1024 = first_comparable(load_json(args.matrix_1024))
    q512 = quality(load_json(args.quality_512))
    q1024 = quality(load_json(args.quality_1024))
    gap = load_json(args.hiera_gap)
    hotspots = load_json(args.hotspots)

    speed_rows = [
        {
            "encode_size": 512,
            "cpp_track_ms": speed_512["cpp_track_ms"],
            "python_track_ms": speed_512["python_track_ms"],
            "python_over_cpp_ratio": speed_512["python_over_cpp_track_ratio_if_comparable"],
            "cpp_faster_than_python": speed_512["cpp_track_ms"] < speed_512["python_track_ms"],
        },
        {
            "encode_size": 1024,
            "cpp_track_ms": speed_1024["cpp_track_ms"],
            "python_track_ms": speed_1024["python_track_ms"],
            "python_over_cpp_ratio": speed_1024["python_over_cpp_track_ratio_if_comparable"],
            "cpp_faster_than_python": speed_1024["cpp_track_ms"] < speed_1024["python_track_ms"],
        },
    ]
    quality_rows = [
        {
            "encode_size": 512,
            **q512,
            "quality_parity": (q512["mean_mask_iou"] or 0.0) >= 0.99 and (q512["min_mask_iou"] or 0.0) >= 0.99,
        },
        {
            "encode_size": 1024,
            **q1024,
            "quality_parity": (q1024["mean_mask_iou"] or 0.0) >= 0.99 and (q1024["min_mask_iou"] or 0.0) >= 0.99,
        },
    ]

    criteria = [
        {
            "criterion": "official Python comparison uses matched input conditions",
            "status": "met",
            "evidence": [
                str(args.matrix_512),
                str(args.matrix_1024),
                "comparison rows have comparable_speed_claim=true",
            ],
        },
        {
            "criterion": "detailed Hiera/profile attribution exists",
            "status": "met",
            "evidence": [str(args.hiera_gap), str(args.hotspots)],
        },
        {
            "criterion": "speed priority is identified",
            "status": "met",
            "evidence": [
                "1024 Hiera gap and node hotspots identify head_dim=56 global/window FlashAttention first",
                "stage-2 q4 MLP matmuls are secondary",
            ],
        },
        {
            "criterion": "C++ is faster than official PyTorch on matched rows",
            "status": "missing",
            "evidence": speed_rows,
        },
        {
            "criterion": "Base+ fallback quality parity with official PyTorch",
            "status": "missing",
            "evidence": quality_rows,
        },
    ]

    result = {
        "objective": (
            "Extend Hiera encode profiling and Base+ fallback quality checks to the official "
            "Python implementation, prioritize speed work, and continue until C++ is faster "
            "than PyTorch."
        ),
        "overall_status": "not_complete",
        "criteria": criteria,
        "next_required_work": [
            "Implement or upstream a real head_dim=56 FlashAttention kernel improvement.",
            "Re-run matched 512/1024 C++ vs official PyTorch speed comparisons.",
            "Investigate the official-PyTorch quality gap before treating Base+ as a Rust-wrapper baseline.",
        ],
        "inputs": {
            "matrix_512": str(args.matrix_512),
            "matrix_1024": str(args.matrix_1024),
            "quality_512": str(args.quality_512),
            "quality_1024": str(args.quality_1024),
            "hiera_gap": str(args.hiera_gap),
            "hotspots": str(args.hotspots),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
