#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Summarize current SAM3/SAM3.1 goal status from concrete local evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_env_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    return None


def criterion(name: str, status: str, evidence: Any) -> dict[str, Any]:
    return {"criterion": name, "status": status, "evidence": evidence}


def with_next_action(item: dict[str, Any], next_action: str) -> dict[str, Any]:
    return {**item, "next_action": next_action}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-health", type=Path, default=Path("outputs/cuda-health/summary.env"))
    parser.add_argument(
        "--sam31-coverage",
        type=Path,
        default=Path("outputs/sam31-checkpoint-contract/sam3-vs-sam31-renamed-coverage.json"),
    )
    parser.add_argument(
        "--sam31-multiplex",
        type=Path,
        default=Path("outputs/sam31-multiplex-state/summary.json"),
    )
    parser.add_argument(
        "--sam31-mux-mask-decoder-case",
        type=Path,
        default=Path("outputs/sam31-mux-mask-decoder-case/summary.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cuda_health = read_env_value(args.cuda_health, "cuda_health")
    sam31_coverage = read_json(args.sam31_coverage)
    sam31_multiplex = read_json(args.sam31_multiplex)
    sam31_mux_mask_decoder_case = read_json(args.sam31_mux_mask_decoder_case)
    manifest_checks = (
        sam31_coverage.get("sam31_implementation_manifest_checks", {})
        if isinstance(sam31_coverage, dict)
        else {}
    )

    criteria = [
        with_next_action(
            criterion(
                "CUDA backend is available for same-device SAM3/SAM3.1 performance measurement",
                "met" if cuda_health == "ok" else "blocked",
                {"summary": str(args.cuda_health), "cuda_health": cuda_health},
            ),
            "Restore CUDA device visibility, then rerun same-contract SAM3/SAM3.1 parity and performance.",
        ),
        with_next_action(
            criterion(
                "SAM3 C++ beats official Python under same input, precision, and TF32 contract",
                "blocked" if cuda_health != "ok" else "missing",
                "Latest recorded same-contract evidence still has C++ slower than official Python; rerun is blocked while CUDA is unavailable.",
            ),
            "Resume CUDA kernel-level optimization and repeat paired C++ vs official Python measurements once CUDA is healthy.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 C++ tracking implementation exists",
                "missing",
                {
                    "summary": (
                        "SAM3.1 C++ has checkpoint contract, multiplex helper parity, v4 file-format support, "
                        "native tensor-manifest registration, legacy tracker tensor separation, a compiled "
                        "multiplex mask-decoder graph builder, and an official Python decoder dump case, but "
                        "no C++ tensor comparison yet."
                    ),
                    "official_mux_mask_decoder_case": sam31_mux_mask_decoder_case,
                },
            ),
            "Add official Python tensor dumps for the multiplex mask-decoder slice, wire a C++ dump runner, and compare masks/iou/object-score before continuing to memory backbone.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 checkpoint contract is loader-ready",
                "met" if manifest_checks.get("diff_rows") == 0 else "missing",
                manifest_checks,
            ),
            "Keep this gate green while adding each SAM3.1 loader/graph slice.",
        ),
        with_next_action(
            criterion(
                "SAM3.1 Object Multiplex helper matches official Python",
                "met"
                if isinstance(sam31_multiplex, dict) and sam31_multiplex.get("diff_rows") == 0
                else "missing",
                sam31_multiplex,
            ),
            "Use the helper as the demux/mux reference when wiring SAM3.1 propagation outputs.",
        ),
    ]
    complete = all(item["status"] == "met" for item in criteria)
    missing_criteria = [
        item
        for item in criteria
        if item["status"] != "met"
    ]
    out = {
        "objective": (
            "SAM3 and SAM3.1 C++ beat official Python under the same precision, TF32 policy, "
            "and input contract, with parity/perf evidence."
        ),
        "complete": complete,
        "missing_criteria": missing_criteria,
        "criteria": criteria,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sam3_sam31_goal_complete={str(complete).lower()} criteria={len(criteria)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
