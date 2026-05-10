# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path: Path, encode_size: int) -> dict[str, Any]:
    data = load_json(path)
    rows = data["comparisons"]
    contract = data.get("comparison_contract", {})
    non_empty = [row for row in rows if row.get("python_mask_area", 0) > 0 and row.get("cpp_mask_area", 0) > 0]
    area_ratios = [
        row["cpp_mask_area"] / row["python_mask_area"]
        for row in non_empty
        if row.get("python_mask_area", 0) > 0
    ]
    bbox_ious = [row["bbox_iou"] for row in rows if row.get("bbox_iou") is not None]
    empty_python = [row["offset"] for row in rows if row.get("python_mask_area") == 0]
    return {
        "encode_size": encode_size,
        "source": str(path),
        "comparison_contract": contract,
        "directly_comparable": all(
            contract.get(key) is True
            for key in (
                "same_decoded_source_resolution",
                "same_frame_count",
                "same_point_prompt",
                "same_encode_img_size",
            )
        ),
        "frames_compared": data["frames_compared"],
        "mean_mask_iou": data["mean_mask_iou"],
        "min_mask_iou": data["min_mask_iou"],
        "mean_bbox_iou_non_empty": mean(bbox_ious) if bbox_ious else None,
        "min_bbox_iou": data.get("min_bbox_iou"),
        "mean_cpp_to_python_mask_area_ratio_non_empty": mean(area_ratios) if area_ratios else None,
        "min_cpp_to_python_mask_area_ratio_non_empty": min(area_ratios) if area_ratios else None,
        "max_cpp_to_python_mask_area_ratio_non_empty": max(area_ratios) if area_ratios else None,
        "python_empty_offsets": empty_python,
        "worst_rows": sorted(
            rows,
            key=lambda row: (row.get("mask_iou") is None, row.get("mask_iou") or 0.0),
        )[:3],
        "interpretation": (
            "C++ masks are generally smaller than official Python masks; at 1024 one "
            "official Python frame is empty, which drives min IoU to zero. This is a "
            "semantic quality gap, not just numeric hash drift."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-512", type=Path, required=True)
    parser.add_argument("--quality-1024", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "quality_parity_target": "mean/min mask IoU close to 1.0 on matched official-Python rows",
        "status": "not_at_parity",
        "rows": [
            summarize(args.quality_512, 512),
            summarize(args.quality_1024, 1024),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
