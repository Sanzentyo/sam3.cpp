# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def compare_rows(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "offset": lhs.get("offset"),
        "expected_frame_index": lhs.get("expected_frame_index"),
    }

    lhs_bbox = lhs.get("bbox_xyxy")
    rhs_bbox = rhs.get("bbox_xyxy")
    if lhs_bbox is not None and rhs_bbox is not None:
        deltas = [abs(float(a) - float(b)) for a, b in zip(lhs_bbox, rhs_bbox, strict=True)]
        row["bbox_iou"] = bbox_iou([float(v) for v in lhs_bbox], [float(v) for v in rhs_bbox])
        row["max_bbox_delta_px"] = max(deltas)
        row["bbox_delta_px"] = deltas
    else:
        row["bbox_iou"] = None
        row["max_bbox_delta_px"] = None
        row["bbox_delta_px"] = None

    row["score_abs_delta"] = abs(float(lhs.get("score", 0.0)) - float(rhs.get("score", 0.0)))
    lhs_area = int(lhs.get("mask_area", 0))
    rhs_area = int(rhs.get("mask_area", 0))
    row["mask_area_delta"] = rhs_area - lhs_area
    row["mask_area_rel_delta"] = (rhs_area - lhs_area) / max(lhs_area, 1)
    row["mask_hash_equal"] = lhs.get("mask_fnv1a64") == rhs.get("mask_fnv1a64")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lhs", type=Path)
    parser.add_argument("rhs", type=Path)
    parser.add_argument("--lhs-label", default="lhs")
    parser.add_argument("--rhs-label", default="rhs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    lhs_rows = read_jsonl(args.lhs)
    rhs_rows = read_jsonl(args.rhs)
    rows = [compare_rows(lhs, rhs) for lhs, rhs in zip(lhs_rows, rhs_rows, strict=False)]
    if len(lhs_rows) != len(rhs_rows):
        rows.append({"row_count_delta": [len(lhs_rows), len(rhs_rows)]})

    comparable = [row for row in rows if "bbox_iou" in row]
    result = {
        "lhs": args.lhs_label,
        "rhs": args.rhs_label,
        "lhs_rows": len(lhs_rows),
        "rhs_rows": len(rhs_rows),
        "rows": rows,
        "summary": {
            "min_bbox_iou": min((float(row["bbox_iou"]) for row in comparable if row["bbox_iou"] is not None), default=None),
            "max_bbox_delta_px": max((float(row["max_bbox_delta_px"]) for row in comparable if row["max_bbox_delta_px"] is not None), default=None),
            "max_score_abs_delta": max((float(row["score_abs_delta"]) for row in comparable), default=None),
            "max_abs_mask_area_rel_delta": max((abs(float(row["mask_area_rel_delta"])) for row in comparable), default=None),
            "mask_hash_equal_rows": sum(1 for row in comparable if row["mask_hash_equal"]),
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
