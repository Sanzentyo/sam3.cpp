#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.8.0",
#   "torchvision==0.23.0",
#   "numpy>=1.26,<2",
#   "timm>=1.0",
#   "iopath>=0.1.10",
#   "einops>=0.7",
#   "pycocotools>=2.0.11",
#   "psutil>=7.0",
#   "ftfy>=6.3",
#   "regex>=2026.5.9",
#   "huggingface_hub",
#   "setuptools<81",
# ]
# ///
"""Inspect official Python SAM3.1 multiplex tracking state shapes."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rel_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def summarize(obj: Any, *, depth: int = 0, max_depth: int = 5, max_items: int = 8, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()

    if isinstance(obj, torch.Tensor):
        out = {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype).removeprefix("torch."),
            "device": str(obj.device),
            "requires_grad": bool(obj.requires_grad),
        }
        if obj.numel() <= 32:
            out["values"] = obj.detach().cpu().tolist()
        return out
    if isinstance(obj, np.ndarray):
        out = {
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
        if obj.size <= 32:
            out["values"] = obj.tolist()
        return out
    if isinstance(obj, (str, int, float, bool)) or obj is None or isinstance(obj, np.generic):
        return scalar(obj)
    if depth >= max_depth:
        return {"type": type(obj).__name__, "repr": repr(obj)[:240]}

    obj_id = id(obj)
    if obj_id in seen:
        return {"type": type(obj).__name__, "cycle": True}
    seen.add(obj_id)

    if isinstance(obj, dict):
        keys = list(obj.keys())
        shown = keys[:max_items]
        return {
            "type": "dict",
            "len": len(obj),
            "keys": [scalar(key) for key in keys[:32]],
            "items": {
                str(key): summarize(obj[key], depth=depth + 1, max_depth=max_depth, max_items=max_items, seen=seen)
                for key in shown
            },
            **({"truncated_items": len(keys) - len(shown)} if len(keys) > len(shown) else {}),
        }
    if isinstance(obj, (list, tuple)):
        shown_values = list(obj[:max_items])
        return {
            "type": type(obj).__name__,
            "len": len(obj),
            "items": [
                summarize(value, depth=depth + 1, max_depth=max_depth, max_items=max_items, seen=seen)
                for value in shown_values
            ],
            **({"truncated_items": len(obj) - len(shown_values)} if len(obj) > len(shown_values) else {}),
        }
    if hasattr(obj, "__dict__"):
        data = vars(obj)
        keys = list(data.keys())
        shown = keys[:max_items]
        return {
            "type": type(obj).__name__,
            "attrs": {
                key: summarize(data[key], depth=depth + 1, max_depth=max_depth, max_items=max_items, seen=seen)
                for key in shown
                if not key.startswith("_")
            },
            **({"truncated_attrs": len(keys) - len(shown)} if len(keys) > len(shown) else {}),
        }
    return {"type": type(obj).__name__, "repr": repr(obj)[:240]}


def patch_start_session_if_needed(predictor: Any) -> None:
    if "offload_state_to_cpu" in inspect.signature(predictor.model.init_state).parameters:
        return

    def start_session_without_state_offload(self: Any, resource_path: str, session_id: str | None = None, offload_video_to_cpu: bool = False, offload_state_to_cpu: bool = False) -> dict[str, str]:
        _ = offload_state_to_cpu
        init_kwargs: dict[str, Any] = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        inference_state = self.model.init_state(**init_kwargs)
        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = types.MethodType(start_session_without_state_offload, predictor)


def extract_state_facts(state: dict[str, Any]) -> dict[str, Any]:
    sam2_states = state.get("sam2_inference_states") or []
    previous_stages = state.get("previous_stages_out") or {}
    return {
        "state_keys": sorted(str(key) for key in state.keys()),
        "num_frames": state.get("num_frames"),
        "orig_height": state.get("orig_height"),
        "orig_width": state.get("orig_width"),
        "device": str(state.get("device")),
        "text_prompt": state.get("text_prompt"),
        "sam2_inference_states_len": len(sam2_states),
        "previous_stages_out_keys": [str(key) for key in previous_stages.keys()] if isinstance(previous_stages, dict) else [],
        "feature_cache_keys": [str(key) for key in (state.get("feature_cache") or {}).keys()],
        "cached_frame_outputs_keys": [str(key) for key in (state.get("cached_frame_outputs") or {}).keys()],
        "tracker_metadata_keys": [str(key) for key in (state.get("tracker_metadata") or {}).keys()],
        "sam2_inference_states": [extract_sam2_state_facts(sam2_state) for sam2_state in sam2_states[:4]],
    }


def extract_sam2_state_facts(sam2_state: dict[str, Any]) -> dict[str, Any]:
    output_dict = sam2_state.get("output_dict") or {}
    temp_outputs = sam2_state.get("temp_output_dict_per_obj") or {}
    multiplex_state = sam2_state.get("multiplex_state")
    facts: dict[str, Any] = {
        "keys": sorted(str(key) for key in sam2_state.keys()),
        "obj_ids": summarize(sam2_state.get("obj_ids"), max_depth=2),
        "obj_id_to_idx": summarize(sam2_state.get("obj_id_to_idx"), max_depth=2),
        "frames_already_tracked": summarize(sam2_state.get("frames_already_tracked"), max_depth=3),
        "output_dict_keys": [str(key) for key in output_dict.keys()],
        "temp_output_dict_per_obj_keys": [str(key) for key in temp_outputs.keys()],
        "cached_features_keys": [str(key) for key in (sam2_state.get("cached_features") or {}).keys()],
        "num_frames": sam2_state.get("num_frames"),
    }
    facts["output_dict"] = {
        str(key): summarize(value, max_depth=4, max_items=6) for key, value in output_dict.items()
    }
    if multiplex_state is not None:
        facts["multiplex_state"] = summarize(multiplex_state, max_depth=3, max_items=12)
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--frames", type=Path, default=Path("outputs/model-matrix-sam3-bf16-cuda-preprocess-pe-cache/frames"))
    parser.add_argument("--prompt-mode", choices=("text", "box", "text-box"), default="box")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--box-xywh", nargs=4, type=float, metavar=("X", "Y", "W", "H"), default=(0.25, 0.25, 0.5, 0.5))
    parser.add_argument("--box-label", type=int, default=1)
    parser.add_argument("--max-stream-frames", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("outputs/sam31-tracking-state-inspect/summary.json"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for official SAM3.1 tracking inspection")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.frames.exists():
        raise SystemExit(f"frames directory not found: {args.frames}")

    sys.path.insert(0, str(args.sam3_repo.resolve()))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        checkpoint_path=str(args.checkpoint),
        version="sam3.1",
        compile=False,
        warm_up=False,
        use_fa3=False,
        use_rope_real=True,
        async_loading_frames=False,
    )
    patch_start_session_if_needed(predictor)

    start = predictor.handle_request({"type": "start_session", "resource_path": str(args.frames)})
    session_id = start["session_id"]
    state = predictor._all_inference_states[session_id]["state"]
    snapshots: dict[str, Any] = {
        "after_start": {
            "facts": extract_state_facts(state),
            "state": summarize(state, max_depth=args.max_depth),
        }
    }

    add_request: dict[str, Any] = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
    }
    if args.prompt_mode in ("text", "text-box"):
        add_request["text"] = args.prompt
    if args.prompt_mode in ("box", "text-box"):
        add_request["bounding_boxes"] = [list(args.box_xywh)]
        add_request["bounding_box_labels"] = [args.box_label]

    add = predictor.handle_request(add_request)
    state = predictor._all_inference_states[session_id]["state"]
    snapshots["after_add_prompt"] = {
        "request": summarize(add_request, max_depth=args.max_depth),
        "response": summarize(add, max_depth=args.max_depth),
        "facts": extract_state_facts(state),
        "state": summarize(state, max_depth=args.max_depth),
    }

    stream_outputs = []
    propagate_error = None
    try:
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "max_frame_num_to_track": args.max_stream_frames,
            }
        ):
            stream_outputs.append(summarize(response, max_depth=args.max_depth))
            if len(stream_outputs) >= args.max_stream_frames:
                break
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - preserves failing official contracts
        propagate_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    state = predictor._all_inference_states[session_id]["state"]
    snapshots["after_propagate"] = {
        "responses": stream_outputs,
        "error": propagate_error,
        "facts": extract_state_facts(state),
        "state": summarize(state, max_depth=args.max_depth),
    }

    try:
        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": False,
            }
        )
    except Exception as exc:  # pragma: no cover - inspection should keep earlier evidence
        snapshots["close_session_error"] = repr(exc)

    summary = {
        "version": "sam3.1",
        "checkpoint": rel_path(args.checkpoint, repo_root),
        "sam3_repo": rel_path(args.sam3_repo, repo_root),
        "frames": rel_path(args.frames, repo_root),
        "prompt_mode": args.prompt_mode,
        "prompt": args.prompt,
        "box_xywh": list(args.box_xywh),
        "box_label": args.box_label,
        "max_stream_frames": args.max_stream_frames,
        "torch": {
            "version": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "snapshots": snapshots,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
