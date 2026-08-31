# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from core import SceneContext, ScenePaths, ensure_scene_skeleton
from detection_and_tracking.backends import sam3 as sam3_module
from detection_and_tracking.backends.sam3 import (
    SAM3Tracker,
    _sam3_config_overrides,
    _sam3_runtime_dtype,
)
from detection_and_tracking.config import DetectionAndTrackingConfig


class _FakeSAM3Runtime:
    def __init__(self, *, config: DetectionAndTrackingConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        prompts: list[str],
    ) -> Path:
        _ = media_path, prompts
        return scene_paths.sidecars_dir / "sam3" / f"{scene_ctx.media_id}_annotated_id.mp4"


class _FakeCuda:
    def __init__(self, supported: bool) -> None:
        self.supported = supported
        self.including_emulation: bool | None = None

    def is_bf16_supported(self, *, including_emulation: bool = True) -> bool:
        self.including_emulation = including_emulation
        return self.supported


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"

    def __init__(self, *, bf16_supported: bool) -> None:
        self.cuda = _FakeCuda(bf16_supported)


class _FakeFrame:
    shape = (720, 1280, 3)


class _FakeSAM3Output:
    frame_idx = 0


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class _FakeRuntimeCuda:
    def empty_cache(self) -> None:
        return


class _FakeRuntimeTorch:
    cuda = _FakeRuntimeCuda()

    def no_grad(self) -> _FakeNoGrad:
        return _FakeNoGrad()


class _FakeProcessor:
    def init_video_session(self, **kwargs: object) -> object:
        _ = kwargs
        return object()

    def add_text_prompt(self, session: object, prompt: str) -> None:
        _ = session, prompt

    def postprocess_outputs(self, session: object, outputs: object) -> dict[str, object]:
        _ = session, outputs
        return {}


class _FakeModel:
    def propagate_in_video_iterator(
        self,
        *,
        inference_session: object,
        show_progress_bar: bool,
    ) -> list[_FakeSAM3Output]:
        _ = inference_session, show_progress_bar
        return [_FakeSAM3Output()]


class _FakeChunkCv2:
    COLOR_BGR2RGB = 3

    def cvtColor(self, bgr: int, code: int) -> str:  # noqa: N802
        _ = code
        return f"rgb-{bgr}"


def test_sam3_tracker_reports_annotated_video_not_red_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sam3_module, "_SAM3Runtime", _FakeSAM3Runtime)
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    scene_ctx = SceneContext(media_id="clip")
    tracker = SAM3Tracker(
        logger=logging.getLogger("test_sam3_tracker"),
        config=DetectionAndTrackingConfig(tracker="sam3", sam3_prompts=("person",)),
    )

    result = tracker.run(tmp_path / "clip.mp4", scene_paths, scene_ctx=scene_ctx)

    assert result.annotated_video == scene_paths.sidecars_dir / "sam3" / "clip_annotated_id.mp4"
    assert result.tracking_overlay == result.annotated_video
    assert result.tracking_video_red_id is None


def test_sam3_config_overrides_keep_only_explicit_values() -> None:
    config = DetectionAndTrackingConfig(
        sam3_score_threshold_detection=0.6,
        sam3_recondition_on_trk_masks=False,
    )

    assert _sam3_config_overrides(config) == {
        "score_threshold_detection": 0.6,
        "recondition_on_trk_masks": False,
    }


def test_sam3_runtime_dtype_uses_bf16_when_cuda_supports_it() -> None:
    torch = _FakeTorch(bf16_supported=True)

    assert _sam3_runtime_dtype(torch) == "bf16"
    assert torch.cuda.including_emulation is False


def test_sam3_runtime_dtype_falls_back_to_float16_without_bf16_support() -> None:
    torch = _FakeTorch(bf16_supported=False)

    assert _sam3_runtime_dtype(torch) == "fp16"
    assert torch.cuda.including_emulation is False


def test_sam3_iter_frame_chunks_yields_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(sam3_module._SAM3Runtime)
    runtime.config = DetectionAndTrackingConfig(sam3_target_fps=30.0)
    runtime.cv2 = _FakeChunkCv2()
    runtime.np = object()
    monkeypatch.setattr(
        sam3_module,
        "decode_video_bgr",
        lambda *_args, **_kwargs: SimpleNamespace(
            stream=SimpleNamespace(fps=30.0, duration_seconds=5 / 30),
            frames=iter(range(5)),
        ),
    )

    chunks = list(runtime._iter_frame_chunks(Path("clip.mp4"), chunk_size=2))

    assert [chunk.source_indices for chunk in chunks] == [[0, 1], [2, 3], [4]]
    assert [len(chunk.rgb_frames) for chunk in chunks] == [2, 2, 1]


def test_sam3_rejects_video_over_duration_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(sam3_module._SAM3Runtime)
    runtime.config = DetectionAndTrackingConfig(sam3_max_duration_s=30.0)
    runtime.cv2 = _FakeChunkCv2()
    runtime.np = object()
    monkeypatch.setattr(
        sam3_module,
        "decode_video_bgr",
        lambda *_args, **_kwargs: SimpleNamespace(
            stream=SimpleNamespace(fps=30.0, duration_seconds=31.0),
            frames=iter(()),
        ),
    )

    with pytest.raises(RuntimeError, match=r"exceeds sam3_max_duration_s=30\.00s"):
        list(runtime._iter_frame_chunks(Path("clip.mp4"), chunk_size=2))


def test_sam3_rejects_video_over_duration_limit_from_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(sam3_module._SAM3Runtime)
    runtime.config = DetectionAndTrackingConfig(sam3_max_duration_s=30.0)
    runtime.cv2 = _FakeChunkCv2()
    runtime.np = object()

    class _FailIfDecoded:
        def __iter__(self) -> Iterator[int]:
            raise AssertionError("SAM3 should reject from metadata before decoding frames.")

    monkeypatch.setattr(
        sam3_module,
        "decode_video_bgr",
        lambda *_args, **_kwargs: SimpleNamespace(
            stream=SimpleNamespace(fps=None, frame_count=901, duration_seconds=None),
            frames=_FailIfDecoded(),
        ),
    )

    with pytest.raises(RuntimeError, match=r"SAM3 input duration 30\.03s exceeds"):
        list(runtime._iter_frame_chunks(Path("clip.mp4"), chunk_size=2))


def test_sam3_rejects_video_over_duration_limit_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(sam3_module._SAM3Runtime)
    runtime.config = DetectionAndTrackingConfig(sam3_max_duration_s=0.05, sam3_target_fps=30.0)
    runtime.cv2 = _FakeChunkCv2()
    runtime.np = object()
    monkeypatch.setattr(
        sam3_module,
        "decode_video_bgr",
        lambda *_args, **_kwargs: SimpleNamespace(
            stream=SimpleNamespace(fps=None, frame_count=None, duration_seconds=None),
            frames=iter(range(4)),
        ),
    )

    with pytest.raises(RuntimeError, match=r"exceeds sam3_max_duration_s=0\.05s"):
        list(runtime._iter_frame_chunks(Path("clip.mp4"), chunk_size=10))


def test_sam3_runtime_keeps_same_raw_id_separate_across_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = object.__new__(sam3_module._SAM3Runtime)
    runtime.config = DetectionAndTrackingConfig(tracker="sam3", sam3_prompts=("person", "forklift"))
    runtime.logger = logging.getLogger("test_sam3_prompt_instance_keys")
    runtime.torch = _FakeRuntimeTorch()
    runtime._processor = _FakeProcessor()
    runtime._model = _FakeModel()
    runtime._device = "cuda"
    runtime._dtype = "fp16"
    captured: dict[str, Any] = {}

    def fake_iter_frame_chunks(
        media_path: Path,
        chunk_size: int,
    ) -> Iterator[Any]:
        _ = media_path, chunk_size
        yield sam3_module._FrameChunk(
            rgb_frames=[object()],
            bgr_frames=[_FakeFrame()],
            source_indices=[0],
            source_fps=30.0,
            output_fps=30.0,
        )

    def fake_detections(
        processed: dict[str, Any],
        prompts: list[str],
    ) -> list[dict[str, Any]]:
        _ = processed, prompts
        return [
            {
                "prompt": "person",
                "instance_id": 7,
                "box_xyxy": [0.0, 0.0, 10.0, 10.0],
                "contours_xy": [],
            },
            {
                "prompt": "forklift",
                "instance_id": 7,
                "box_xyxy": [20.0, 20.0, 40.0, 40.0],
                "contours_xy": [],
            },
        ]

    def capture_tracking_artifacts(
        *,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        frames: list[dict[str, Any]],
        instances: list[dict[str, Any]],
    ) -> None:
        _ = scene_paths, scene_ctx, frames
        captured["instances"] = instances

    monkeypatch.setattr(runtime, "_ensure_model_loaded", lambda: None)
    monkeypatch.setattr(runtime, "_iter_frame_chunks", fake_iter_frame_chunks)
    monkeypatch.setattr(runtime, "_detections_from_processed", fake_detections)
    monkeypatch.setattr(sam3_module, "write_tracking_artifacts", capture_tracking_artifacts)

    runtime.run(
        media_path=tmp_path / "clip.mp4",
        scene_paths=ensure_scene_skeleton(tmp_path / "scene"),
        scene_ctx=SceneContext(media_id="clip"),
        prompts=["person", "forklift"],
    )

    instances = captured["instances"]
    assert len(instances) == 2
    instances_by_id = {str(instance["object_id"]): instance for instance in instances}
    assert instances_by_id["person_0_7"]["object_type"] == "person"
    assert instances_by_id["forklift_0_7"]["object_type"] == "forklift"
    assert instances_by_id["person_0_7"]["frame_count"] == 1
    assert instances_by_id["forklift_0_7"]["frame_count"] == 1


def test_sam3_runtime_assigns_unique_track_ids_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The same raw SAM3 instance id in two internal sessions must not merge.

    SAM3 resets instance ids per internal session, so a raw id is only unique
    within a chunk. The backend must assign a video-global ``track_id`` so the
    ``track_XXXX`` crop dir (and downstream PAS identity) stay distinct.
    """
    runtime = object.__new__(sam3_module._SAM3Runtime)
    runtime.config = DetectionAndTrackingConfig(tracker="sam3", sam3_prompts=("person",))
    runtime.logger = logging.getLogger("test_sam3_unique_track_ids")
    runtime.torch = _FakeRuntimeTorch()
    runtime._processor = _FakeProcessor()
    runtime._model = _FakeModel()
    runtime._device = "cuda"
    runtime._dtype = "fp16"
    captured: dict[str, Any] = {}

    def fake_iter_frame_chunks(
        media_path: Path,
        chunk_size: int,
    ) -> Iterator[Any]:
        _ = media_path, chunk_size
        for source_index in (0, 1):
            yield sam3_module._FrameChunk(
                rgb_frames=[object()],
                bgr_frames=[_FakeFrame()],
                source_indices=[source_index],
                source_fps=30.0,
                output_fps=30.0,
            )

    def fake_detections(
        processed: dict[str, Any],
        prompts: list[str],
    ) -> list[dict[str, Any]]:
        _ = processed, prompts
        # Same raw instance id reappears in the second session.
        return [
            {
                "prompt": "person",
                "instance_id": 1,
                "box_xyxy": [0.0, 0.0, 10.0, 10.0],
                "contours_xy": [],
            }
        ]

    def capture_tracking_artifacts(
        *,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        frames: list[dict[str, Any]],
        instances: list[dict[str, Any]],
    ) -> None:
        _ = scene_paths, scene_ctx, frames
        captured["instances"] = instances

    monkeypatch.setattr(runtime, "_ensure_model_loaded", lambda: None)
    monkeypatch.setattr(runtime, "_iter_frame_chunks", fake_iter_frame_chunks)
    monkeypatch.setattr(runtime, "_detections_from_processed", fake_detections)
    monkeypatch.setattr(sam3_module, "write_tracking_artifacts", capture_tracking_artifacts)

    runtime.run(
        media_path=tmp_path / "clip.mp4",
        scene_paths=ensure_scene_skeleton(tmp_path / "scene"),
        scene_ctx=SceneContext(media_id="clip"),
        prompts=["person"],
    )

    instances = captured["instances"]
    assert len(instances) == 2
    assert {str(inst["object_id"]) for inst in instances} == {"person_0_1", "person_1_1"}
    track_ids = [inst["track_id"] for inst in instances]
    assert len(set(track_ids)) == 2, "track_ids must be unique across internal sessions"


class _FrameOutput:
    def __init__(self, frame_idx: int) -> None:
        self.frame_idx = frame_idx


class _TwoFrameModel:
    def propagate_in_video_iterator(
        self,
        *,
        inference_session: object,
        show_progress_bar: bool,
    ) -> list[_FrameOutput]:
        _ = inference_session, show_progress_bar
        return [_FrameOutput(0), _FrameOutput(1)]


def test_sam3_runtime_aggregates_max_detection_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = object.__new__(sam3_module._SAM3Runtime)
    runtime.config = DetectionAndTrackingConfig(tracker="sam3", sam3_prompts=("person",))
    runtime.logger = logging.getLogger("test_sam3_detection_score")
    runtime.torch = _FakeRuntimeTorch()
    runtime._processor = _FakeProcessor()
    runtime._model = _TwoFrameModel()
    runtime._device = "cuda"
    runtime._dtype = "fp16"
    captured: dict[str, Any] = {}
    scores = iter([0.42, 0.91])

    def fake_iter_frame_chunks(media_path: Path, chunk_size: int) -> Iterator[Any]:
        _ = media_path, chunk_size
        yield sam3_module._FrameChunk(
            rgb_frames=[object(), object()],
            bgr_frames=[_FakeFrame(), _FakeFrame()],
            source_indices=[0, 1],
            source_fps=30.0,
            output_fps=30.0,
        )

    def fake_detections(processed: dict[str, Any], prompts: list[str]) -> list[dict[str, Any]]:
        _ = processed, prompts
        return [
            {
                "prompt": "person",
                "instance_id": 7,
                "box_xyxy": [0.0, 0.0, 10.0, 10.0],
                "contours_xy": [],
                "score": next(scores),
            }
        ]

    def capture_tracking_artifacts(
        *,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        frames: list[dict[str, Any]],
        instances: list[dict[str, Any]],
    ) -> None:
        _ = scene_paths, scene_ctx, frames
        captured["instances"] = instances

    monkeypatch.setattr(runtime, "_ensure_model_loaded", lambda: None)
    monkeypatch.setattr(runtime, "_iter_frame_chunks", fake_iter_frame_chunks)
    monkeypatch.setattr(runtime, "_detections_from_processed", fake_detections)
    monkeypatch.setattr(sam3_module, "write_tracking_artifacts", capture_tracking_artifacts)

    runtime.run(
        media_path=tmp_path / "clip.mp4",
        scene_paths=ensure_scene_skeleton(tmp_path / "scene"),
        scene_ctx=SceneContext(media_id="clip"),
        prompts=["person"],
    )

    instances = captured["instances"]
    # Same chunk index + raw id collapses both frames into one track.
    assert len(instances) == 1
    assert instances[0]["frame_count"] == 2
    assert instances[0]["detection_score"] == 0.91
