# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26,<2",
#   "pillow",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def detection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows if "offset" in row and row.get("source") != "sam3cpp-meta"
    ]


def bbox_iou(lhs: list[float], rhs: list[float]) -> float:
    ax0, ay0, ax1, ay1 = lhs
    bx0, by0, bx1, by1 = rhs
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    lhs_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    rhs_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = lhs_area + rhs_area - inter
    return inter / denom if denom > 0.0 else 1.0


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def mask_pixel_stats(
    lhs: dict[str, Any], rhs: dict[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    lhs_path = lhs.get("mask_path")
    rhs_path = rhs.get("mask_path")
    if not lhs_path or not rhs_path:
        return {
            "mask_pixel_iou": None,
            "mask_pixel_xor": None,
            "mask_pixel_intersection": None,
            "mask_pixel_union": None,
        }

    lhs_mask = load_mask(base_dir / str(lhs_path))
    rhs_mask = load_mask(base_dir / str(rhs_path))
    if lhs_mask.shape != rhs_mask.shape:
        return {
            "mask_pixel_iou": None,
            "mask_pixel_xor": None,
            "mask_pixel_intersection": None,
            "mask_pixel_union": None,
            "mask_shape_delta": [list(lhs_mask.shape), list(rhs_mask.shape)],
        }

    intersection = int(np.logical_and(lhs_mask, rhs_mask).sum())
    union = int(np.logical_or(lhs_mask, rhs_mask).sum())
    xor = int(np.logical_xor(lhs_mask, rhs_mask).sum())
    return {
        "mask_pixel_iou": intersection / union if union > 0 else 1.0,
        "mask_pixel_xor": xor,
        "mask_pixel_intersection": intersection,
        "mask_pixel_union": union,
    }


def compare_rows(
    lhs: dict[str, Any], rhs: dict[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "offset": lhs.get("offset"),
        "expected_frame_index": lhs.get("expected_frame_index"),
    }

    lhs_bbox = lhs.get("bbox_xyxy")
    rhs_bbox = rhs.get("bbox_xyxy")
    if lhs_bbox is not None and rhs_bbox is not None:
        deltas = [
            abs(float(a) - float(b)) for a, b in zip(lhs_bbox, rhs_bbox, strict=True)
        ]
        row["bbox_iou"] = bbox_iou(
            [float(v) for v in lhs_bbox], [float(v) for v in rhs_bbox]
        )
        row["max_bbox_delta_px"] = max(deltas)
        row["bbox_delta_px"] = deltas
    else:
        row["bbox_iou"] = None
        row["max_bbox_delta_px"] = None
        row["bbox_delta_px"] = None

    row["score_abs_delta"] = abs(
        float(lhs.get("score", 0.0)) - float(rhs.get("score", 0.0))
    )
    lhs_area = int(lhs.get("mask_area", 0))
    rhs_area = int(rhs.get("mask_area", 0))
    row["mask_area_delta"] = rhs_area - lhs_area
    row["mask_area_rel_delta"] = (rhs_area - lhs_area) / max(lhs_area, 1)
    row["mask_hash_equal"] = lhs.get("mask_fnv1a64") == rhs.get("mask_fnv1a64")
    row.update(mask_pixel_stats(lhs, rhs, base_dir=base_dir))
    return row


def row_has_exact_diff(row: dict[str, Any]) -> bool:
    if "row_count_delta" in row:
        return True
    if row.get("bbox_iou") is None:
        return True
    return (
        float(row.get("max_bbox_delta_px") or 0.0) != 0.0
        or float(row.get("score_abs_delta") or 0.0) != 0.0
        or int(row.get("mask_area_delta") or 0) != 0
        or not bool(row.get("mask_hash_equal"))
        or (
            row.get("mask_pixel_iou") is not None
            and float(row["mask_pixel_iou"]) != 1.0
        )
        or (row.get("mask_pixel_xor") is not None and int(row["mask_pixel_xor"]) != 0)
    )


def row_exceeds_tolerance(
    row: dict[str, Any],
    *,
    min_bbox_iou: float,
    max_bbox_delta_px: float,
    max_score_delta: float,
    max_mask_area_rel_delta: float,
    min_mask_iou: float,
    max_mask_xor_pixels: int | None,
    require_mask_hash_equal: bool,
) -> bool:
    if "row_count_delta" in row:
        return True
    if row.get("bbox_iou") is None:
        return True
    return (
        float(row["bbox_iou"]) < min_bbox_iou
        or float(row["max_bbox_delta_px"]) > max_bbox_delta_px
        or float(row["score_abs_delta"]) > max_score_delta
        or abs(float(row["mask_area_rel_delta"])) > max_mask_area_rel_delta
        or (
            row.get("mask_pixel_iou") is not None
            and float(row["mask_pixel_iou"]) < min_mask_iou
        )
        or (
            max_mask_xor_pixels is not None
            and row.get("mask_pixel_xor") is not None
            and int(row["mask_pixel_xor"]) > max_mask_xor_pixels
        )
        or (require_mask_hash_equal and not bool(row["mask_hash_equal"]))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lhs", type=Path)
    parser.add_argument("rhs", type=Path)
    parser.add_argument("--lhs-label", default="lhs")
    parser.add_argument("--rhs-label", default="rhs")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-bbox-iou", type=float, default=1.0)
    parser.add_argument("--max-bbox-delta-px", type=float, default=0.0)
    parser.add_argument("--max-score-delta", type=float, default=0.0)
    parser.add_argument("--max-mask-area-rel-delta", type=float, default=0.0)
    parser.add_argument("--min-mask-iou", type=float, default=1.0)
    parser.add_argument("--max-mask-xor-pixels", type=int)
    parser.add_argument("--allow-mask-hash-diff", action="store_true")
    args = parser.parse_args()

    lhs_rows = detection_rows(read_jsonl(args.lhs))
    rhs_rows = detection_rows(read_jsonl(args.rhs))
    rows = [
        compare_rows(lhs, rhs, base_dir=Path.cwd())
        for lhs, rhs in zip(lhs_rows, rhs_rows, strict=False)
    ]
    if len(lhs_rows) != len(rhs_rows):
        rows.append({"row_count_delta": [len(lhs_rows), len(rhs_rows)]})

    comparable = [row for row in rows if "bbox_iou" in row]
    diff_rows = sum(1 for row in rows if row_has_exact_diff(row))
    tolerance_diff_rows = sum(
        1
        for row in rows
        if row_exceeds_tolerance(
            row,
            min_bbox_iou=args.min_bbox_iou,
            max_bbox_delta_px=args.max_bbox_delta_px,
            max_score_delta=args.max_score_delta,
            max_mask_area_rel_delta=args.max_mask_area_rel_delta,
            min_mask_iou=args.min_mask_iou,
            max_mask_xor_pixels=args.max_mask_xor_pixels,
            require_mask_hash_equal=not args.allow_mask_hash_diff,
        )
    )
    result = {
        "lhs": args.lhs_label,
        "rhs": args.rhs_label,
        "lhs_rows": len(lhs_rows),
        "rhs_rows": len(rhs_rows),
        "diff_rows": diff_rows,
        "tolerance_diff_rows": tolerance_diff_rows,
        "tolerance": {
            "min_bbox_iou": args.min_bbox_iou,
            "max_bbox_delta_px": args.max_bbox_delta_px,
            "max_score_delta": args.max_score_delta,
            "max_mask_area_rel_delta": args.max_mask_area_rel_delta,
            "min_mask_iou": args.min_mask_iou,
            "max_mask_xor_pixels": args.max_mask_xor_pixels,
            "require_mask_hash_equal": not args.allow_mask_hash_diff,
        },
        "rows": rows,
        "summary": {
            "min_bbox_iou": min(
                (
                    float(row["bbox_iou"])
                    for row in comparable
                    if row["bbox_iou"] is not None
                ),
                default=None,
            ),
            "max_bbox_delta_px": max(
                (
                    float(row["max_bbox_delta_px"])
                    for row in comparable
                    if row["max_bbox_delta_px"] is not None
                ),
                default=None,
            ),
            "max_score_abs_delta": max(
                (float(row["score_abs_delta"]) for row in comparable), default=None
            ),
            "max_abs_mask_area_rel_delta": max(
                (abs(float(row["mask_area_rel_delta"])) for row in comparable),
                default=None,
            ),
            "min_mask_pixel_iou": min(
                (
                    float(row["mask_pixel_iou"])
                    for row in comparable
                    if row.get("mask_pixel_iou") is not None
                ),
                default=None,
            ),
            "max_mask_pixel_xor": max(
                (
                    int(row["mask_pixel_xor"])
                    for row in comparable
                    if row.get("mask_pixel_xor") is not None
                ),
                default=None,
            ),
            "mask_hash_equal_rows": sum(
                1 for row in comparable if row["mask_hash_equal"]
            ),
        },
    }

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
