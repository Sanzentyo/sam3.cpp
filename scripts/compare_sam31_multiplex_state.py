#!/usr/bin/env python3
"""Compare the C++ SAM3.1 multiplex state helper with official Python SAM3."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def load_official_multiplex_controller(repo: Path):
    module_path = repo / "sam3" / "model" / "multiplex_utils.py"
    spec = importlib.util.spec_from_file_location("sam3_multiplex_utils_direct", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiplexController


def run_cpp(cpp_exe: Path, objects: int, multiplex_count: int, features: int) -> dict[str, Any]:
    cmd = [
        str(cpp_exe),
        "--dump-assignments",
        "--objects",
        str(objects),
        "--multiplex-count",
        str(multiplex_count),
        "--features",
        str(features),
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return json.loads(proc.stdout)


def python_reference(
    controller_cls,
    objects: int,
    multiplex_count: int,
    features: int,
) -> dict[str, Any]:
    controller = controller_cls(
        multiplex_count=multiplex_count,
        eval_multiplex_count=multiplex_count,
    )
    state = controller.get_state(
        objects,
        device=torch.device("cpu"),
        dtype=torch.float32,
        random=False,
    )
    data = torch.arange(1, objects * features + 1, dtype=torch.float32).reshape(objects, features)
    muxed = state.mux(data).flatten().tolist()
    demuxed = state.demux(state.mux(data)).flatten().tolist()
    return {
        "assignments": state.assignments,
        "input": data.flatten().tolist(),
        "muxed": muxed,
        "demuxed": demuxed,
    }


def assert_equal(name: str, lhs: Any, rhs: Any) -> None:
    if lhs != rhs:
        raise AssertionError(f"{name} mismatch:\nC++={lhs}\nPython={rhs}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/sam3"))
    parser.add_argument(
        "--cpp-exe",
        type=Path,
        default=Path("build/xmake-release-cuda/examples/sam31_multiplex_state"),
    )
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--objects", type=int, nargs="+", default=[1, 16, 17, 33])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    controller_cls = load_official_multiplex_controller(args.repo)
    rows = []
    for objects in args.objects:
        cpp = run_cpp(args.cpp_exe, objects, args.multiplex_count, args.features)
        py = python_reference(controller_cls, objects, args.multiplex_count, args.features)
        assert_equal("assignments", cpp["assignments"], py["assignments"])
        assert_equal("input", cpp["input"], py["input"])
        assert_equal("muxed", cpp["muxed"], py["muxed"])
        assert_equal("demuxed", cpp["demuxed"], py["demuxed"])
        rows.append(
            {
                "objects": objects,
                "multiplex_count": args.multiplex_count,
                "features": args.features,
                "num_buckets": len(cpp["assignments"]),
                "status": "ok",
            }
        )

    summary = {"rows": rows, "diff_rows": 0}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
