# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


TABLE_ROW_RE = re.compile(
    r"\|\s+edgetam_q4_0\s+\|.*?\|\s*CUDA\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([0-9]+)\s*\|\s*OK",
    re.S,
)


def run_command(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(cmd)}")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare_parity(lhs: list[dict[str, object]], rhs: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ["offset", "expected_frame_index", "bbox_xyxy", "score", "mask_area", "mask_fnv1a64"]
    diffs: list[dict[str, object]] = []
    for index, (a, b) in enumerate(zip(lhs, rhs, strict=False)):
        row: dict[str, object] = {"row": index}
        same = True
        for key in keys:
            if key == "bbox_xyxy" and a.get(key) is not None and b.get(key) is not None:
                delta = [abs(float(x) - float(y)) for x, y in zip(a[key], b[key], strict=True)]
                if any(value != 0.0 for value in delta):
                    row[key] = delta
                    same = False
            elif key == "score":
                delta = abs(float(a[key]) - float(b[key]))
                if delta != 0.0:
                    row[key] = delta
                    same = False
            elif a.get(key) != b.get(key):
                row[key] = [a.get(key), b.get(key)]
                same = False
        if not same:
            diffs.append(row)
    if len(lhs) != len(rhs):
        diffs.append({"row_count": [len(lhs), len(rhs)]})
    return diffs


def parse_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = TABLE_ROW_RE.search(text)
    if not match:
        raise RuntimeError(f"could not parse benchmark row from {path}")
    names = ["load", "init", "track", "p50", "p95", "total", "rss", "det"]
    return {name: float(value) for name, value in zip(names, match.groups(), strict=True)}


def summary(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    # Good enough for n=15/20 local benchmarking; exact t tables are not the point here.
    tcrit = 2.145 if len(values) == 15 else 2.093 if len(values) == 20 else 1.96
    return {
        "n": float(len(values)),
        "mean": mean,
        "median": statistics.median(values),
        "sd": sd,
        "min": min(values),
        "max": max(values),
        "ci95_low": mean - tcrit * se,
        "ci95_high": mean + tcrit * se,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", default="outputs/parity-stats")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        args.benchmark,
        "--models-dir",
        args.models_dir,
        "--video",
        args.video,
        "--gpu-only",
        "--filter",
        "edgetam",
        "--n-frames",
        str(args.frames),
        "--recondition-every",
        "16",
    ]

    default_jsonl = out_dir / "default_fullmask.jsonl"
    shape_jsonl = out_dir / "shape_fullmask.jsonl"
    run_command(base_cmd + ["--output-jsonl", str(default_jsonl)], out_dir / "default_fullmask.log")
    shape_env = os.environ.copy()
    shape_env["GGML_CUDA_GRAPH_SHAPE_KEY"] = "1"
    run_command(
        base_cmd + ["--output-jsonl", str(shape_jsonl)],
        out_dir / "shape_fullmask.log",
        env=shape_env,
    )

    parity_diffs = compare_parity(read_jsonl(default_jsonl), read_jsonl(shape_jsonl))

    paired: list[dict[str, float]] = []
    perf_cmd = base_cmd + ["--bbox-only"]
    for index in range(1, args.runs + 1):
        default_log = out_dir / f"default_{index:02d}.log"
        shape_log = out_dir / f"shape_{index:02d}.log"
        run_command(perf_cmd, default_log)
        run_command(perf_cmd, shape_log, env=shape_env)
        default = parse_log(default_log)
        shape = parse_log(shape_log)
        paired.append(
            {
                "run": float(index),
                "default_track": default["track"],
                "shape_track": shape["track"],
                "track_saved": default["track"] - shape["track"],
                "default_p50": default["p50"],
                "shape_p50": shape["p50"],
                "default_p95": default["p95"],
                "shape_p95": shape["p95"],
                "default_total": default["total"],
                "shape_total": shape["total"],
            }
        )
        print(f"completed pair {index:02d}", flush=True)

    default_track = [row["default_track"] for row in paired]
    shape_track = [row["shape_track"] for row in paired]
    saved = [row["track_saved"] for row in paired]
    result = {
        "parity": {
            "default_rows": len(read_jsonl(default_jsonl)),
            "shape_rows": len(read_jsonl(shape_jsonl)),
            "diff_rows": len(parity_diffs),
            "diffs": parity_diffs,
        },
        "track_ms": {
            "default": summary(default_track),
            "shape_key": summary(shape_track),
            "paired_saved": summary(saved),
            "speedup_by_mean_percent": (statistics.mean(default_track) / statistics.mean(shape_track) - 1.0)
            * 100.0,
        },
        "runs": paired,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 1 if parity_diffs else 0


if __name__ == "__main__":
    sys.exit(main())
