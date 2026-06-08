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


def cxx_profile(path: Path) -> dict[str, Any]:
    log = load_json(path)["logs"][0]
    labels = log.get("profile_compute_by_label", {})
    return {
        "path": str(path),
        "track_ms": log.get("track_ms"),
        "hiera_encode": labels.get("hiera_encode", {}),
        "propagate_single": labels.get("propagate_single", {}),
        "memory_encode": labels.get("memory_encode", {}),
    }


def python_profile(path: Path) -> dict[str, Any]:
    data = load_json(path)
    stats = data.get("stats", {})
    return {
        "path": str(path),
        "track_ms": data.get("track_ms"),
        "forward_image": stats.get("forward_image", {}),
        "track_step": stats.get("track_step", {}),
        "_run_memory_encoder": stats.get("_run_memory_encoder", {}),
    }


def mean_ms(section: dict[str, Any]) -> float | None:
    value = section.get("mean_ms")
    return float(value) if value is not None else None


def steady_mean_ms(section: dict[str, Any]) -> float | None:
    values = [float(v) for v in section.get("values_ms", [])]
    if len(values) <= 1:
        return mean_ms(section)
    trimmed = values[1:]
    return sum(trimmed) / len(trimmed)


def summarize_pair(cxx: dict[str, Any], py: dict[str, Any], encode_size: int) -> dict[str, Any]:
    cxx_hiera_steady = steady_mean_ms(cxx["hiera_encode"])
    py_forward_mean = mean_ms(py["forward_image"])
    return {
        "encode_size": encode_size,
        "cxx_track_ms": cxx["track_ms"],
        "python_instrumented_track_ms": py["track_ms"],
        "cxx_hiera_steady_ms": cxx_hiera_steady,
        "python_forward_image_mean_ms": py_forward_mean,
        "hiera_gap_ms": (
            cxx_hiera_steady - py_forward_mean
            if cxx_hiera_steady is not None and py_forward_mean is not None
            else None
        ),
        "cxx_propagate_single_mean_ms": mean_ms(cxx["propagate_single"]),
        "python_track_step_mean_ms": mean_ms(py["track_step"]),
        "cxx_memory_encode_mean_ms": mean_ms(cxx["memory_encode"]),
        "python_memory_encoder_mean_ms": mean_ms(py["_run_memory_encoder"]),
        "priority": [
            "Reduce Hiera image encoder steady compute first.",
            "Target head_dim=56 FlashAttention and quantized Hiera MLP matmul shapes.",
            "Treat propagation and memory encoding as secondary until Hiera is near PyTorch forward_image.",
        ],
        "inputs": {
            "cxx_profile": cxx["path"],
            "python_profile": py["path"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx-512", type=Path, required=True)
    parser.add_argument("--py-512", type=Path, required=True)
    parser.add_argument("--cxx-1024", type=Path, required=True)
    parser.add_argument("--py-1024", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "note": (
            "Python stage timings use synchronized wrappers and are for attribution, "
            "not the unsynchronized speed baseline."
        ),
        "pairs": [
            summarize_pair(cxx_profile(args.cxx_512), python_profile(args.py_512), 512),
            summarize_pair(cxx_profile(args.cxx_1024), python_profile(args.py_1024), 1024),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
