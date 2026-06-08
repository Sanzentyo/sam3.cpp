# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "einops",
#   "ftfy==6.1.1",
#   "huggingface_hub",
#   "iopath",
#   "numpy==1.26.4",
#   "psutil",
#   "pycocotools",
#   "regex",
#   "setuptools<81",
#   "timm",
#   "torch==2.8.0",
#   "torchvision==0.23.0",
#   "tqdm",
#   "typing_extensions",
# ]
# ///

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

import torch


DEFAULT_MODULES = [
    "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv",
    "detector.backbone.vision_backbone.trunk.blocks.0.attn.proj",
    "detector.backbone.vision_backbone.trunk.blocks.0.mlp.fc1",
    "detector.backbone.vision_backbone.trunk.blocks.0.mlp.fc2",
    "detector.backbone.vision_backbone.trunk.blocks.7.attn.qkv",
    "detector.backbone.vision_backbone.trunk.blocks.7.mlp.fc1",
]


def tensor_desc(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "device": str(value.device),
        }
    return str(type(value))


def patch_start_session(predictor: Any) -> None:
    if "offload_state_to_cpu" in inspect.signature(predictor.model.init_state).parameters:
        return

    def start_session_without_state_offload(
        self: Any,
        resource_path: str,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        del offload_state_to_cpu
        init_kwargs: dict[str, Any] = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        state = self.model.init_state(**init_kwargs)
        session_id = session_id or str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = types.MethodType(start_session_without_state_offload, predictor)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect official SAM3 CUDA autocast dtypes.")
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--tf32", choices=["on", "off"], default="on")
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.sam3_repo))

    allow_tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    from sam3.model_builder import build_sam3_predictor

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor = build_sam3_predictor(
            version=args.version,
            compile=False,
            use_fa3=False,
            async_loading_frames=False,
        )

    patch_start_session(predictor)

    modules = args.module or DEFAULT_MODULES
    named_modules = dict(predictor.model.named_modules())
    dtypes: dict[str, dict[str, Any]] = {}

    def make_hook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if name in dtypes:
                return
            dtypes[name] = {
                "input": tensor_desc(inputs[0] if inputs else None),
                "output": tensor_desc(output),
            }

        return hook

    missing = [name for name in modules if name not in named_modules]
    if missing:
        raise SystemExit(f"missing modules: {missing}")
    for name in modules:
        named_modules[name].register_forward_hook(make_hook(name))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        response = predictor.handle_request(
            {"type": "start_session", "resource_path": str(args.frame_dir)}
        )
        session_id = response["session_id"]
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": args.prompt,
            }
        )
        for _ in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            break
        torch.cuda.synchronize()

    result = {
        "version": args.version,
        "frame_dir": str(args.frame_dir),
        "prompt": args.prompt,
        "tf32": args.tf32,
        "autocast_dtype": "torch.bfloat16",
        "modules": dtypes,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
