# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value


def parse_stages(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return {int(part) for part in raw.replace(",", " ").split()}


def find_block(manifest: dict[str, Any], block_idx: int) -> dict[str, Any]:
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list):
        raise SystemExit("manifest has no blocks list")
    for block in blocks:
        if isinstance(block, dict) and block.get("block") == block_idx:
            return block
    raise SystemExit(f"manifest has no block {block_idx}")


def run_stage(
    *,
    exe: Path,
    model: str,
    block_idx: int,
    stage: dict[str, Any],
    out_dir: Path,
    warmup_runs: int,
    repeats: int,
    threads: int,
    device: str,
) -> dict[str, Any]:
    stage_id = int(stage["stage"])
    stage_name = str(stage["stage_name"])
    input_prefix = Path(str(stage["input_prefix"]))
    out_prefix = out_dir / f"stage_{stage_id}_{stage_name}_out"
    stdout_path = out_dir / f"stage_{stage_id}.json"
    stderr_path = out_dir / f"stage_{stage_id}.stderr"
    cmd = [
        str(exe),
        "--model",
        model,
        "--input-prefix",
        str(input_prefix),
        "--out-prefix",
        str(out_prefix),
        "--stage",
        stage_name,
        "--block",
        str(block_idx),
        "--warmup-runs",
        str(warmup_runs),
        "--repeats",
        str(repeats),
        "--threads",
        str(threads),
        f"--{device}",
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)
    row = {
        "stage": stage_id,
        "stage_name": stage_name,
        "input_prefix": str(input_prefix),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        raise SystemExit(
            f"stage {stage_id} {stage_name} failed; see {stderr_path}"
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--stages")
    args = parser.parse_args()

    selected_stages = parse_stages(args.stages)
    manifest = load_json(args.manifest)
    block = find_block(manifest, args.block)
    stage_inputs = block.get("stage_inputs")
    if not isinstance(stage_inputs, list):
        raise SystemExit(f"manifest block {args.block} has no stage_inputs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stage in stage_inputs:
        if not isinstance(stage, dict) or not isinstance(stage.get("stage"), int):
            continue
        if selected_stages is not None and stage["stage"] not in selected_stages:
            continue
        rows.append(
            run_stage(
                exe=args.exe,
                model=args.model,
                block_idx=args.block,
                stage=stage,
                out_dir=args.out_dir,
                warmup_runs=args.warmup_runs,
                repeats=args.repeats,
                threads=args.threads,
                device=args.device,
            )
        )

    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_manifest": str(args.manifest),
                "model": args.model,
                "block": args.block,
                "warmup_runs": args.warmup_runs,
                "repeats": args.repeats,
                "threads": args.threads,
                "device": args.device,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
