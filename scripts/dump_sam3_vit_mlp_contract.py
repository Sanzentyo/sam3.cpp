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

import numpy as np
import torch


def patch_start_session(predictor: Any) -> None:
    if (
        "offload_state_to_cpu"
        in inspect.signature(predictor.model.init_state).parameters
    ):
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

    predictor.start_session = types.MethodType(
        start_session_without_state_offload, predictor
    )


def write_cpp_layout_tensor(prefix: Path, tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.ndim != 4:
        raise ValueError(f"expected [B,H,W,C] tensor, got shape {tuple(tensor.shape)}")
    b, h, w, c = tensor.shape
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cpp = (
        tensor.detach().to(torch.float32).permute(3, 2, 1, 0).contiguous().cpu().numpy()
    )
    cpp.astype(np.float32, copy=False).ravel(order="F").tofile(
        prefix.with_suffix(".bin")
    )
    prefix.with_suffix(".shape").write_text(f"{c} {w} {h} {b}\n", encoding="utf-8")
    return {
        "prefix": str(prefix),
        "python_shape": [b, h, w, c],
        "cpp_shape": [c, w, h, b],
        "dtype": str(tensor.dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump official SAM3 ViT MLP fused addmm activation tensors."
    )
    parser.add_argument("--sam3-repo", type=Path, default=Path("external/sam3"))
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--tf32", choices=["on", "off"], default="on")
    parser.add_argument("--block", type=int, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    blocks = set(args.block or [0])
    sys.path.insert(0, str(args.sam3_repo))

    allow_tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    from sam3.model_builder import build_sam3_predictor
    from sam3.perflib import fused as fused_module
    from sam3.model import vitdet

    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    with (
        torch.autocast(device_type="cuda", dtype=autocast_dtype),
        torch.inference_mode(),
    ):
        predictor = build_sam3_predictor(
            version=args.version,
            compile=False,
            use_fa3=False,
            async_loading_frames=False,
        )

    patch_start_session(predictor)

    module_names = {module: name for name, module in predictor.model.named_modules()}
    original_addmm_act = vitdet.addmm_act
    original_addmm_act_module = fused_module.addmm_act
    captures: dict[str, dict[str, Any]] = {}

    def patched_addmm_act(
        activation: Any, linear: Any, mat1: torch.Tensor
    ) -> torch.Tensor:
        name = module_names.get(linear, "")
        capture = False
        block = None
        for block_idx in blocks:
            suffix = f"trunk.blocks.{block_idx}.mlp.fc1"
            if name.endswith(suffix):
                capture = f"block_{block_idx:02d}" not in captures
                block = block_idx
                break

        if not capture:
            return original_addmm_act(activation, linear, mat1)

        self = linear.bias.detach().to(torch.bfloat16)
        mat1_bf16 = mat1.to(torch.bfloat16)
        mat2 = linear.weight.detach().to(torch.bfloat16)
        mat1_flat = mat1_bf16.view(-1, mat1_bf16.shape[-1])
        y = fused_module.addmm_act_op(
            self,
            mat1_flat,
            mat2.t(),
            beta=1,
            alpha=1,
            use_gelu=True,
        )
        y = y.view(mat1_bf16.shape[:-1] + (y.shape[-1],))

        key = f"block_{block:02d}"
        in_prefix = args.out_dir / f"{key}_mlp_fc1_input"
        out_prefix = args.out_dir / f"{key}_mlp_fused_gelu"
        captures[key] = {
            "module": name,
            "input": write_cpp_layout_tensor(in_prefix, mat1_bf16),
            "fused_gelu": write_cpp_layout_tensor(out_prefix, y),
        }
        return y

    vitdet.addmm_act = patched_addmm_act
    fused_module.addmm_act = patched_addmm_act
    try:
        with (
            torch.autocast(device_type="cuda", dtype=autocast_dtype),
            torch.inference_mode(),
        ):
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
                if captures.keys() >= {f"block_{idx:02d}" for idx in blocks}:
                    break
            torch.cuda.synchronize()
    finally:
        vitdet.addmm_act = original_addmm_act
        fused_module.addmm_act = original_addmm_act_module

    summary = {
        "version": args.version,
        "frame_dir": str(args.frame_dir),
        "prompt": args.prompt,
        "tf32": args.tf32,
        "autocast_dtype": str(autocast_dtype),
        "blocks": sorted(blocks),
        "captures": captures,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if len(captures) == len(blocks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
