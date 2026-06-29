#!/usr/bin/env python3
from __future__ import annotations

import importlib
import inspect
import json
import statistics
import sys
import time
import types
import uuid
from collections.abc import Callable
from typing import Any

import torch


START_SESSION_PHASE = "start_session"
ADD_PROMPT_PHASE = "add_prompt"
PROPAGATE_PHASE = "propagate"


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


class StageRecorder:
    def __init__(self, *, frames: int, timed_start_frame: int) -> None:
        self.enabled = False
        self.current_phase: str | None = None
        self.frames = frames
        self.timed_start_frame = timed_start_frame
        self.events: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.events.clear()

    def is_timed_tail_event(self, event: dict[str, Any]) -> bool:
        frame_index = event.get("frame_index")
        return (
            frame_index is not None
            and self.timed_start_frame <= int(frame_index) < self.frames
        )

    def selected_events(
        self, events: list[dict[str, Any]], *, timed_tail_only: bool
    ) -> list[dict[str, Any]]:
        if not timed_tail_only:
            return list(events)
        return [event for event in events if self.is_timed_tail_event(event)]

    def stats(
        self, events: list[dict[str, Any]], *, timed_tail_only: bool
    ) -> dict[str, Any]:
        grouped: dict[str, list[float]] = {}
        for event in self.selected_events(events, timed_tail_only=timed_tail_only):
            grouped.setdefault(str(event["stage"]), []).append(float(event["ms"]))
        return {stage: summarize(values) for stage, values in grouped.items()}

    def totals(
        self, events: list[dict[str, Any]], *, timed_tail_only: bool
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        for event in self.selected_events(events, timed_tail_only=timed_tail_only):
            stage = str(event["stage"])
            totals[stage] = totals.get(stage, 0.0) + float(event["ms"])
        return totals

    def counts(
        self, events: list[dict[str, Any]], *, timed_tail_only: bool
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.selected_events(events, timed_tail_only=timed_tail_only):
            stage = str(event["stage"])
            counts[stage] = counts.get(stage, 0) + 1
        return counts

    def phase_totals(
        self, events: list[dict[str, Any]], *, timed_tail_only: bool
    ) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = {}
        for event in self.selected_events(events, timed_tail_only=timed_tail_only):
            phase = str(event.get("phase") or "unknown")
            stage = str(event["stage"])
            phase_totals = totals.setdefault(phase, {})
            phase_totals[stage] = phase_totals.get(stage, 0.0) + float(event["ms"])
        return totals

    def stage_frame_ms(
        self, events: list[dict[str, Any]], stage: str, *, timed_tail_only: bool
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in self.selected_events(events, timed_tail_only=timed_tail_only):
            if event["stage"] != stage:
                continue
            rows.append(
                {
                    "frame_index": event.get("frame_index"),
                    "phase": event.get("phase"),
                    "ms": float(event["ms"]),
                }
            )
        return rows

    def wrap(self, stage: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled:
                return fn(*args, **kwargs)
            frame_idx = self._frame_index(signature, args, kwargs)
            cuda_sync()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            cuda_sync()
            event: dict[str, Any] = {
                "stage": stage,
                "phase": self.current_phase,
                "ms": (time.perf_counter() - t0) * 1000.0,
            }
            if frame_idx is not None:
                event["frame_index"] = int(frame_idx)
            self.events.append(event)
            return out

        return wrapper

    @staticmethod
    def _frame_index(
        signature: inspect.Signature | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> int | None:
        if "frame_idx" in kwargs:
            return int(kwargs["frame_idx"])
        if "frame_index" in kwargs:
            return int(kwargs["frame_index"])
        if signature is None:
            return None
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            return None
        for name in ("frame_idx", "frame_index"):
            if name in bound.arguments:
                return int(bound.arguments[name])
        return None


def summarize_stage_totals(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for stage, total_ms in row.get(key, {}).items():
            grouped.setdefault(str(stage), []).append(float(total_ms))
    return {stage: summarize(values) for stage, values in grouped.items()}


def summarize_phase_stage_totals(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        for phase, stages in row.get(key, {}).items():
            phase_group = grouped.setdefault(str(phase), {})
            for stage, total_ms in stages.items():
                phase_group.setdefault(str(stage), []).append(float(total_ms))
    return {
        phase: {stage: summarize(values) for stage, values in stages.items()}
        for phase, stages in grouped.items()
    }


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
        inference_state = self.model.init_state(**init_kwargs)
        session_id = session_id or str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = types.MethodType(
        start_session_without_state_offload, predictor
    )


def maybe_wrap_method(
    recorder: StageRecorder, obj: Any, method_name: str, stage: str
) -> None:
    if obj is None or not hasattr(obj, method_name):
        return
    original = getattr(obj, method_name)
    if getattr(original, "_sam3_cpp_timed_stage", False):
        return
    wrapped = recorder.wrap(stage, original)
    setattr(wrapped, "_sam3_cpp_timed_stage", True)
    setattr(obj, method_name, wrapped)


def maybe_wrap_module_function(
    recorder: StageRecorder, module_name: str, function_name: str, stage: str
) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return
    if not hasattr(module, function_name):
        return
    original = getattr(module, function_name)
    if getattr(original, "_sam3_cpp_timed_stage", False):
        return
    wrapped = recorder.wrap(stage, original)
    setattr(wrapped, "_sam3_cpp_timed_stage", True)
    setattr(module, function_name, wrapped)


def apply_tf32_policy(allow_tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def timed_segment(
    recorder: StageRecorder, phase: str, fn: Callable[[], Any]
) -> tuple[Any, float]:
    previous_phase = recorder.current_phase
    if recorder.enabled:
        recorder.current_phase = phase
    cuda_sync()
    t0 = time.perf_counter()
    try:
        out = fn()
    finally:
        cuda_sync()
        if recorder.enabled:
            recorder.current_phase = previous_phase
    return out, (time.perf_counter() - t0) * 1000.0


def main() -> int:
    (
        sam3_repo,
        video_dir,
        prompt,
        frames_arg,
        python_dtype,
        tf32_policy,
        repeats_arg,
        version,
        use_fa3_arg,
        compile_model_arg,
        timed_start_frame_arg,
        track_scope,
    ) = sys.argv[1:13]
    frames = int(frames_arg)
    repeats = int(repeats_arg)
    use_fa3 = use_fa3_arg == "1"
    compile_model = compile_model_arg == "1"
    timed_start_frame = int(timed_start_frame_arg)

    sys.path.insert(0, sam3_repo)
    if python_dtype == "bf16":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    elif python_dtype == "fp16":
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    allow_tf32 = tf32_policy == "on"
    apply_tf32_policy(allow_tf32)

    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        version=version,
        compile=compile_model,
        use_fa3=use_fa3,
        async_loading_frames=False,
    )
    apply_tf32_policy(allow_tf32)
    patch_start_session(predictor)

    recorder = StageRecorder(frames=frames, timed_start_frame=timed_start_frame)
    maybe_wrap_module_function(
        recorder,
        "sam3.model.io_utils",
        "load_resource_as_video_frames",
        "load_resource_frames",
    )
    maybe_wrap_module_function(
        recorder,
        "sam3.model.sam3_multiplex_tracking",
        "load_resource_as_video_frames",
        "load_resource_frames",
    )
    maybe_wrap_module_function(
        recorder,
        "sam3.model.io_utils",
        "load_video_frames",
        "load_video_frames",
    )
    maybe_wrap_module_function(
        recorder,
        "sam3.model.sam3_tracking_predictor",
        "load_video_frames",
        "load_video_frames",
    )
    maybe_wrap_module_function(
        recorder,
        "sam3.model.utils.sam2_utils",
        "load_video_frames",
        "load_video_frames",
    )
    maybe_wrap_method(
        recorder,
        predictor.model,
        "_construct_initial_input_batch",
        "construct_initial_input_batch",
    )
    maybe_wrap_method(
        recorder, predictor.model, "_run_single_frame_inference", "single_frame"
    )
    maybe_wrap_method(
        recorder, predictor.model, "run_backbone_and_detection", "backbone_detection"
    )
    maybe_wrap_method(
        recorder, predictor.model, "run_tracker_propagation", "tracker_propagation"
    )
    maybe_wrap_method(
        recorder,
        predictor.model,
        "run_tracker_update_planning_phase",
        "tracker_update_planning",
    )
    maybe_wrap_method(
        recorder,
        predictor.model,
        "run_tracker_update_execution_phase",
        "tracker_update_execution",
    )
    maybe_wrap_method(recorder, predictor.model, "build_outputs", "build_outputs")
    maybe_wrap_method(
        recorder, predictor.model, "_tracker_update_memories", "memory_update"
    )
    tracker = getattr(predictor.model, "tracker", None)
    maybe_wrap_method(
        recorder, tracker, "_get_image_feature", "tracker_get_image_feature"
    )
    maybe_wrap_method(recorder, tracker, "track_step", "tracker_track_step")
    maybe_wrap_method(
        recorder, tracker, "_run_memory_encoder", "tracker_memory_encoder"
    )

    def run_once(*, measure: bool = False) -> dict[str, Any]:
        recorder.clear()
        recorder.enabled = measure
        e2e_t0 = time.perf_counter()
        response, start_session_ms = timed_segment(
            recorder,
            START_SESSION_PHASE,
            lambda: predictor.handle_request(
                {"type": "start_session", "resource_path": video_dir}
            ),
        )
        session_id = response["session_id"]
        _, add_prompt_ms = timed_segment(
            recorder,
            ADD_PROMPT_PHASE,
            lambda: predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                }
            ),
        )

        stream_count = 0

        def consume_stream() -> None:
            nonlocal stream_count
            for stream_response in predictor.handle_stream_request(
                {"type": "propagate_in_video", "session_id": session_id}
            ):
                stream_count += 1
                if isinstance(stream_response, dict):
                    frame_index = stream_response.get(
                        "frame_index",
                        stream_response.get("frame_idx", stream_count - 1),
                    )
                else:
                    frame_index = stream_count - 1
                if stream_count >= frames or int(frame_index) + 1 >= frames:
                    break

        _, propagate_wall_ms = timed_segment(recorder, PROPAGATE_PHASE, consume_stream)
        recorder.enabled = False
        cuda_sync()
        e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
        _, reset_ms = timed_segment(
            recorder,
            "reset_session",
            lambda: predictor.handle_request(
                {"type": "reset_session", "session_id": session_id}
            ),
        )
        single_frame_events = [
            event for event in recorder.events if event["stage"] == "single_frame"
        ]
        measured_events = [
            event
            for event in single_frame_events
            if 0 < int(event.get("frame_index", -1)) < frames
            and int(event.get("frame_index", -1)) >= timed_start_frame
        ]
        measured_ms = sum(float(event["ms"]) for event in measured_events)
        result = {
            "track_frames": len(measured_events),
            "measured_ms": measured_ms,
            "single_frame_events": list(single_frame_events),
            "stage_events": list(recorder.events),
            "stage_ms_stats": recorder.stats(recorder.events, timed_tail_only=False),
            "timed_tail_stage_ms_stats": recorder.stats(
                recorder.events, timed_tail_only=True
            ),
            "stage_total_ms": recorder.totals(recorder.events, timed_tail_only=False),
            "timed_tail_stage_total_ms": recorder.totals(
                recorder.events, timed_tail_only=True
            ),
            "stage_phase_total_ms": recorder.phase_totals(
                recorder.events, timed_tail_only=False
            ),
            "timed_tail_stage_phase_total_ms": recorder.phase_totals(
                recorder.events, timed_tail_only=True
            ),
            "stage_event_counts": recorder.counts(
                recorder.events, timed_tail_only=False
            ),
            "timed_tail_stage_event_counts": recorder.counts(
                recorder.events, timed_tail_only=True
            ),
            "timed_tail_backbone_detection_frame_ms": recorder.stage_frame_ms(
                recorder.events, "backbone_detection", timed_tail_only=True
            ),
            "timed_tail_backbone_detection_heavy_frames": [
                row
                for row in recorder.stage_frame_ms(
                    recorder.events, "backbone_detection", timed_tail_only=True
                )
                if row["ms"] > 1.0
            ],
            "start_session_ms": start_session_ms,
            "add_prompt_ms": add_prompt_ms,
            "propagate_wall_ms": propagate_wall_ms,
            "model_e2e_ms": add_prompt_ms + propagate_wall_ms,
            "session_e2e_ms": e2e_ms,
            "reset_ms": reset_ms,
            "stream_count": stream_count,
        }
        return result

    for _ in range(2):
        run_once()

    torch.cuda.reset_peak_memory_stats()
    runs: list[dict[str, Any]] = []
    for index in range(repeats):
        e2e_result = run_once(measure=False)
        profile_result = run_once(measure=True)
        count = profile_result["track_frames"]
        measured_ms = profile_result["measured_ms"]
        runs.append(
            {
                "run": index + 1,
                "track_frames": count,
                "track_ms": measured_ms / max(count, 1),
                "total_ms": measured_ms,
                "single_frame_events": profile_result["single_frame_events"],
                "stage_events": profile_result["stage_events"],
                "stage_ms_stats": profile_result["stage_ms_stats"],
                "timed_tail_stage_ms_stats": profile_result[
                    "timed_tail_stage_ms_stats"
                ],
                "stage_total_ms": profile_result["stage_total_ms"],
                "timed_tail_stage_total_ms": profile_result[
                    "timed_tail_stage_total_ms"
                ],
                "stage_phase_total_ms": profile_result["stage_phase_total_ms"],
                "timed_tail_stage_phase_total_ms": profile_result[
                    "timed_tail_stage_phase_total_ms"
                ],
                "stage_event_counts": profile_result["stage_event_counts"],
                "timed_tail_stage_event_counts": profile_result[
                    "timed_tail_stage_event_counts"
                ],
                "timed_tail_backbone_detection_frame_ms": profile_result[
                    "timed_tail_backbone_detection_frame_ms"
                ],
                "timed_tail_backbone_detection_heavy_frames": profile_result[
                    "timed_tail_backbone_detection_heavy_frames"
                ],
                "start_session_ms": e2e_result["start_session_ms"],
                "add_prompt_ms": e2e_result["add_prompt_ms"],
                "propagate_wall_ms": e2e_result["propagate_wall_ms"],
                "model_e2e_ms": e2e_result["model_e2e_ms"],
                "session_e2e_ms": e2e_result["session_e2e_ms"],
                "reset_ms": e2e_result["reset_ms"],
                "stream_count": e2e_result["stream_count"],
                "profile_model_e2e_ms": profile_result["model_e2e_ms"],
                "profile_session_e2e_ms": profile_result["session_e2e_ms"],
            }
        )

    track_values = [float(row["track_ms"]) for row in runs]
    total_values = [float(row["total_ms"]) for row in runs]
    model_e2e_values = [float(row["model_e2e_ms"]) for row in runs]
    session_e2e_values = [float(row["session_e2e_ms"]) for row in runs]
    add_prompt_values = [float(row["add_prompt_ms"]) for row in runs]
    propagate_wall_values = [float(row["propagate_wall_ms"]) for row in runs]
    start_session_values = [float(row["start_session_ms"]) for row in runs]
    print(
        json.dumps(
            {
                "family": version,
                "backend": f"PyTorch CUDA {python_dtype}",
                "python_dtype": python_dtype,
                "tf32_policy": tf32_policy,
                "track_scope": track_scope,
                "warmup_runs": 2,
                "sam3_version": version,
                "sam3_use_fa3": use_fa3,
                "sam3_compile": compile_model,
                "torch_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "torch_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "torch_float32_matmul_precision": (
                    torch.get_float32_matmul_precision()
                    if hasattr(torch, "get_float32_matmul_precision")
                    else None
                ),
                "frames": frames,
                "timed_start_frame": timed_start_frame,
                "track_frames": runs[-1]["track_frames"],
                "track_ms": sum(track_values) / len(track_values),
                "total_ms": sum(total_values) / len(total_values),
                "model_e2e_ms": sum(model_e2e_values) / len(model_e2e_values),
                "session_e2e_ms": sum(session_e2e_values) / len(session_e2e_values),
                "start_session_ms": sum(start_session_values)
                / len(start_session_values),
                "add_prompt_ms": sum(add_prompt_values) / len(add_prompt_values),
                "propagate_wall_ms": sum(propagate_wall_values)
                / len(propagate_wall_values),
                "track_ms_stats": summarize(track_values),
                "model_e2e_ms_stats": summarize(model_e2e_values),
                "session_e2e_ms_stats": summarize(session_e2e_values),
                "start_session_ms_stats": summarize(start_session_values),
                "add_prompt_ms_stats": summarize(add_prompt_values),
                "propagate_wall_ms_stats": summarize(propagate_wall_values),
                "stage_ms_stats": runs[-1]["stage_ms_stats"],
                "timed_tail_stage_ms_stats": runs[-1]["timed_tail_stage_ms_stats"],
                "profile_stage_total_ms_stats": summarize_stage_totals(
                    runs, "stage_total_ms"
                ),
                "profile_timed_tail_stage_total_ms_stats": summarize_stage_totals(
                    runs, "timed_tail_stage_total_ms"
                ),
                "profile_stage_phase_total_ms_stats": summarize_phase_stage_totals(
                    runs, "stage_phase_total_ms"
                ),
                "profile_timed_tail_stage_phase_total_ms_stats": (
                    summarize_phase_stage_totals(
                        runs, "timed_tail_stage_phase_total_ms"
                    )
                ),
                "profile_stage_event_counts": runs[-1]["stage_event_counts"],
                "profile_timed_tail_stage_event_counts": runs[-1][
                    "timed_tail_stage_event_counts"
                ],
                "timed_tail_backbone_detection_frame_ms": runs[-1][
                    "timed_tail_backbone_detection_frame_ms"
                ],
                "timed_tail_backbone_detection_heavy_frames": runs[-1][
                    "timed_tail_backbone_detection_heavy_frames"
                ],
                "timed_tail_backbone_detection_heavy_threshold_ms": 1.0,
                "runs": runs,
                "rss_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                "image_size": 1008,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
