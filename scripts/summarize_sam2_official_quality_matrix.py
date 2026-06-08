#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PRECISION_RE = re.compile(r"_(f32|f16|q8_0|q4_1|q4_0|mxfp4|nvfp4)\.ggml$")


def precision_from_summary(summary: dict[str, Any], path: Path) -> str:
    metadata = summary.get("metadata_validation", {}).get("metadata", {})
    model_path = str(metadata.get("model_path", ""))
    if match := PRECISION_RE.search(model_path):
        return match.group(1)
    for value in ("f32", "f16", "q8_0", "q4_1", "q4_0", "mxfp4", "nvfp4"):
        if value in str(path):
            return value
    return "unknown"


def image_size_from_summary(summary: dict[str, Any], path: Path) -> int | None:
    metadata = summary.get("metadata_validation", {}).get("metadata", {})
    for value in (
        metadata.get("encode_img_size_effective"),
        summary.get("python_summary", {}).get("image_size"),
        summary.get("metadata_validation", {}).get("expected_encode_img_size"),
    ):
        if value is not None:
            return int(value)
    if match := re.search(r"-(512|1024)(?:-|$)", str(path)):
        return int(match.group(1))
    return None


def mode_from_path(path: Path) -> str:
    text = str(path)
    if "multimask" in text:
        return "multimask"
    if "fullmask" in text:
        return "fullmask"
    return "default"


def read_summary(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize C++ vs official SAM2 quality summaries.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.summaries:
        summary = read_summary(path)
        if summary is None:
            continue
        if "frames_compared" not in summary:
            continue
        rows.append(
            {
                "path": str(path),
                "precision": precision_from_summary(summary, path),
                "image_size": image_size_from_summary(summary, path),
                "mode": mode_from_path(path),
                "frames_compared": summary.get("frames_compared"),
                "mean_mask_iou": summary.get("mean_mask_iou"),
                "min_mask_iou": summary.get("min_mask_iou"),
                "min_bbox_iou": summary.get("min_bbox_iou"),
                "metadata_status": summary.get("metadata_validation", {}).get("status"),
                "initial_candidate_status": summary.get("initial_candidates", {}).get("status"),
                "selected_index_match": summary.get("initial_candidates", {}).get("selected_index_match"),
            }
        )

    rows.sort(
        key=lambda row: (
            row["precision"],
            row["image_size"] if row["image_size"] is not None else -1,
            row["mode"],
            row["path"],
        )
    )
    best_by_precision_size: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['precision']}:{row['image_size']}"
        current = best_by_precision_size.get(key)
        if current is None or (row.get("mean_mask_iou") or -1.0) > (current.get("mean_mask_iou") or -1.0):
            best_by_precision_size[key] = row

    result = {
        "rows": rows,
        "best_by_precision_size": best_by_precision_size,
        "count": len(rows),
    }
    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
