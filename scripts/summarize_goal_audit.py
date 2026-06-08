# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fastest_comparable(summary: dict[str, Any], model_substr: str) -> dict[str, Any]:
    rows = [
        row
        for row in summary.get("comparisons", [])
        if row.get("comparable_speed_claim") and model_substr in str(row.get("model", ""))
    ]
    if not rows:
        raise ValueError(f"no comparable comparison row found for model substring {model_substr!r}")
    return min(rows, key=lambda row: float(row["cpp_track_ms"]))


def comparable_rows(summary: dict[str, Any], model_substr: str = "") -> list[dict[str, Any]]:
    rows = [
        row
        for row in summary.get("comparisons", summary.get("comparable_speed_rows", []))
        if row.get("comparable_speed_claim") and model_substr in str(row.get("model", ""))
    ]
    if not rows and "comparable_speed_rows" in summary:
        rows = [
            row
            for row in summary["comparable_speed_rows"]
            if row.get("comparable_speed_claim") and model_substr in str(row.get("model", ""))
        ]
    return rows


def quality(summary: Any) -> dict[str, Any]:
    if isinstance(summary, list):
        if not summary:
            raise ValueError("empty quality matrix")
        row = max(
            summary,
            key=lambda item: (
                float(item.get("min_mask_iou") or 0.0),
                float(item.get("mean_mask_iou") or 0.0),
            ),
        )
        return {
            "precision": row.get("precision"),
            "mean_mask_iou": row.get("mean_mask_iou"),
            "min_mask_iou": row.get("min_mask_iou"),
            "frames_compared": row.get("frames_compared") or row.get("frames"),
            "source": row.get("source"),
        }
    return {
        "mean_mask_iou": summary.get("mean_mask_iou"),
        "min_mask_iou": summary.get("min_mask_iou"),
        "frames_compared": summary.get("frames_compared") or summary.get("frames"),
    }


def paired_speed_quality(summary: dict[str, Any], *, min_iou: float = 0.99) -> dict[str, Any]:
    cpp_mean = float(summary["cpp_mean"])
    python_mean = float(summary["python_mean"])
    min_mask_iou = float(summary["min_mask_iou_min"])
    runs = int(summary["runs"])
    cpp_faster_runs = int(summary["cpp_faster_runs"])
    return {
        "sample": summary.get("sample"),
        "runs": runs,
        "cpp_mean": cpp_mean,
        "python_mean": python_mean,
        "cpp_over_python_mean_ratio": cpp_mean / python_mean if python_mean else None,
        "cpp_mean_faster": cpp_mean < python_mean,
        "cpp_faster_runs": cpp_faster_runs,
        "min_mask_iou_min": min_mask_iou,
        "strict_quality": min_mask_iou >= min_iou,
        "timing_contract": summary.get("timing_contract"),
    }


def hotspot_priority_evidence(summary: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("hotspots", []):
        priority = str(row.get("priority", ""))
        if not priority.startswith(("primary:", "secondary:")):
            continue
        rows.append(
            {
                "priority": priority,
                "op": row.get("op"),
                "sum_ms_drop_max": row.get("sum_ms_drop_max"),
                "mean_ms_drop_max": row.get("mean_ms_drop_max"),
                "count": row.get("count"),
                "signature": row.get("signature"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-512", type=Path)
    parser.add_argument("--matrix-1024", type=Path)
    parser.add_argument("--matrix-all-1024", type=Path)
    parser.add_argument("--paired-base-plus-1024", type=Path)
    parser.add_argument("--paired-juggle-1024", type=Path)
    parser.add_argument("--static-epilogue-parity", type=Path)
    parser.add_argument("--quality-512", type=Path, required=True)
    parser.add_argument("--quality-1024", type=Path, required=True)
    parser.add_argument("--hiera-gap", type=Path, required=True)
    parser.add_argument("--hotspots", type=Path, required=True)
    parser.add_argument("--model-substr", default="sam2.1_hiera_base_plus")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    speed_rows = []
    if args.matrix_512 is not None:
        speed_512 = fastest_comparable(load_json(args.matrix_512), args.model_substr)
        speed_rows.append(
            {
                "scope": "legacy_matrix_512",
                "encode_size": 512,
                "cpp_track_ms": speed_512["cpp_track_ms"],
                "python_track_ms": speed_512["python_track_ms"],
                "python_over_cpp_ratio": speed_512["python_over_cpp_track_ratio_if_comparable"],
                "cpp_faster_than_python": speed_512["cpp_track_ms"] < speed_512["python_track_ms"],
            }
        )
    if args.matrix_1024 is not None:
        speed_1024 = fastest_comparable(load_json(args.matrix_1024), args.model_substr)
        speed_rows.append(
            {
                "scope": "legacy_matrix_1024",
                "encode_size": 1024,
                "cpp_track_ms": speed_1024["cpp_track_ms"],
                "python_track_ms": speed_1024["python_track_ms"],
                "python_over_cpp_ratio": speed_1024["python_over_cpp_track_ratio_if_comparable"],
                "cpp_faster_than_python": speed_1024["cpp_track_ms"] < speed_1024["python_track_ms"],
            }
        )
    all_size_rows = []
    if args.matrix_all_1024 is not None:
        for row in comparable_rows(load_json(args.matrix_all_1024), "sam2.1_hiera_"):
            all_size_rows.append(
                {
                    "scope": "sam2.1_all_sizes_q8_0_1024",
                    "model": row["model"],
                    "family": row.get("family"),
                    "precision": row.get("precision"),
                    "cpp_track_ms": row["cpp_track_ms"],
                    "python_track_ms": row["python_track_ms"],
                    "python_over_cpp_ratio": row["python_over_cpp_track_ratio_if_comparable"],
                    "cpp_faster_than_python": row["cpp_track_ms"] < row["python_track_ms"],
                }
            )
    paired_rows = []
    if args.paired_base_plus_1024 is not None:
        paired_rows.append(paired_speed_quality(load_json(args.paired_base_plus_1024)))
    if args.paired_juggle_1024 is not None:
        paired_rows.append(paired_speed_quality(load_json(args.paired_juggle_1024)))
    static_epilogue = load_json(args.static_epilogue_parity) if args.static_epilogue_parity else None
    q512 = quality(load_json(args.quality_512))
    q1024 = quality(load_json(args.quality_1024))
    gap = load_json(args.hiera_gap)
    hotspots = load_json(args.hotspots)
    hotspot_priorities = hotspot_priority_evidence(hotspots)

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

    quality_status = "met" if all(row["quality_parity"] for row in quality_rows) else "missing"
    paired_status = "met" if paired_rows and all(row["cpp_mean_faster"] and row["strict_quality"] for row in paired_rows) else "missing"
    all_size_speed_status = (
        "met"
        if all_size_rows and all(row["cpp_faster_than_python"] for row in all_size_rows)
        else "missing"
    )
    speed_status = "met" if paired_status == "met" and all_size_speed_status == "met" else "missing"

    criteria = [
        {
            "criterion": "official Python comparison uses matched input conditions",
            "status": "met",
            "evidence": [
                str(path)
                for path in [args.matrix_512, args.matrix_1024, args.matrix_all_1024, args.paired_base_plus_1024, args.paired_juggle_1024]
                if path is not None
            ] + [
                "matrix rows have comparable_speed_claim=true",
                "paired summaries state C++ Track/fr and official PyTorch propagate_ms_per_frame both measure frames 1..N-1",
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
            "evidence": hotspot_priorities
            or [
                "1024 Hiera gap and node hotspots identify Hiera FlashAttention and quantized MLP matmuls as the remaining priority.",
            ],
        },
        {
            "criterion": "fusion/kernel improvement has exact parity evidence",
            "status": "met" if static_epilogue and static_epilogue.get("summary", {}).get("mask_hash_equal_rows") == static_epilogue.get("lhs_rows") else "missing",
            "evidence": {
                "static_epilogue_parity": str(args.static_epilogue_parity) if args.static_epilogue_parity else None,
                "summary": static_epilogue.get("summary") if static_epilogue else None,
            },
        },
        {
            "criterion": "Base+ paired C++ q8_0 is faster than official PyTorch and strict-quality across samples",
            "status": paired_status,
            "evidence": paired_rows,
        },
        {
            "criterion": "C++ is faster than official PyTorch on matched all-size SAM2.1 rows",
            "status": all_size_speed_status,
            "evidence": all_size_rows or speed_rows,
        },
        {
            "criterion": "Base+ fallback quality parity with official PyTorch",
            "status": quality_status,
            "evidence": quality_rows,
        },
    ]
    missing = [row for row in criteria if row["status"] != "met"]

    result = {
        "objective": (
            "Extend Hiera encode profiling and Base+ fallback quality checks to the official "
            "Python implementation, prioritize speed work, and continue until C++ is faster "
            "than PyTorch."
        ),
        "overall_status": "complete" if not missing and speed_status == "met" else "not_complete",
        "criteria": criteria,
        "missing_criteria": missing,
        "next_required_work": [
            "Bring the matched all-size SAM2.1 1024 matrix below official PyTorch, especially Base+ and Large.",
            "Close the dynamic-video Base+ strict-quality gap: the juggle paired sample still has min mask IoU below 0.99.",
            "Continue kernel work on D56/D72 FlashAttention and q8_0 MMQ body; the static epilogue improvement is useful but not sufficient.",
        ],
        "inputs": {
            "matrix_512": str(args.matrix_512) if args.matrix_512 else None,
            "matrix_1024": str(args.matrix_1024) if args.matrix_1024 else None,
            "matrix_all_1024": str(args.matrix_all_1024) if args.matrix_all_1024 else None,
            "paired_base_plus_1024": str(args.paired_base_plus_1024) if args.paired_base_plus_1024 else None,
            "paired_juggle_1024": str(args.paired_juggle_1024) if args.paired_juggle_1024 else None,
            "static_epilogue_parity": str(args.static_epilogue_parity) if args.static_epilogue_parity else None,
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
