# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import cast

import pytest
from captioning.artifacts import CAPTION_ARTIFACTS_KEY, CaptionArtifactsState
from captioning.captioner import CaptionResult, DenseCaptioner
from captioning.config import CaptioningConfig
from captioning.task import DETECTION_AND_TRACKING_ARTIFACTS_KEY, CaptioningTask
from core import (
    DataEntry,
    EnhancedMediaState,
    SceneContext,
    ScenePaths,
    ScenePipelineState,
    read_pipeline_state,
    write_pipeline_state,
)


class _FakeCaptioner:
    def __init__(self) -> None:
        self.media_path: Path | None = None
        self.scene_ctx: SceneContext | None = None

    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        self.media_path = media_path
        self.scene_ctx = scene_ctx
        video_json = scene_paths.sidecars_dir / "captioning" / "video_captions.json"
        video_json.parent.mkdir(parents=True, exist_ok=True)
        video_json.write_text("{}", encoding="utf-8")
        return CaptionResult(success=True, sidecar_json=None, video_json=video_json)


class _FailingCaptioner:
    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        _ = media_path, scene_paths, scene_ctx
        return CaptionResult(success=False)


class _ImageCaptioner:
    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        _ = media_path, scene_ctx
        sidecar_json = scene_paths.sidecars_dir / "captioning" / "image_caption.json"
        image_json = scene_paths.sidecars_dir / "captioning" / "image_captions.json"
        sidecar_json.parent.mkdir(parents=True, exist_ok=True)
        sidecar_json.write_text("{}", encoding="utf-8")
        image_json.write_text("{}", encoding="utf-8")
        return CaptionResult(success=True, sidecar_json=sidecar_json, image_json=image_json)


class _DaftVideoCaptioner:
    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        _ = media_path, scene_ctx
        sidecar = scene_paths.sidecars_dir / "captioning" / "metadata_chunk.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "duration_span": [0.0, 1.0],
                    "summary": "A car passes through the scene.",
                    "windows": [
                        {
                            "start_s": 0.0,
                            "end_s": 1.0,
                            "description": "A car drives through an intersection.",
                            "parsed": {
                                "scene_description": "A car is in an intersection.",
                                "event_summary": "A car passes through the scene.",
                            },
                        }
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return CaptionResult(success=True, sidecar_json=sidecar)


def test_task_prefers_tracking_overlay_in_auto_mode(tmp_path: Path) -> None:
    original = tmp_path / "original.mp4"
    original.touch()
    overlay = tmp_path / "overlay.mp4"
    overlay.touch()
    scene = tmp_path / "scene"
    write_pipeline_state(
        scene,
        ScenePipelineState(
            task_artifacts={
                DETECTION_AND_TRACKING_ARTIFACTS_KEY: {
                    "success": True,
                    "red_id_overlay_path": str(overlay),
                }
            }
        ),
    )
    fake = _FakeCaptioner()
    task = CaptioningTask(
        config=CaptioningConfig(input_source="auto"),
        captioner=cast(DenseCaptioner, fake),
    )

    task.run(DataEntry(media_path=str(original), data_path=str(scene)))

    assert fake.media_path == overlay
    state = read_pipeline_state(scene)
    caption_artifacts = CaptionArtifactsState.model_validate(
        state.task_artifacts[CAPTION_ARTIFACTS_KEY]
    )
    assert caption_artifacts.success is True
    assert caption_artifacts.metadata_chunk_json is None
    assert caption_artifacts.video_json == str(
        scene / "sidecars" / "captioning" / "video_captions.json"
    )


def test_task_writes_daft_pivots_after_success(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    task = CaptioningTask(
        config=CaptioningConfig(input_source="original"),
        captioner=cast(DenseCaptioner, _DaftVideoCaptioner()),
    )

    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    assert (scene_dir / "contextual" / "chunks.json").exists()
    assert (scene_dir / "task" / "temporal_description.json").exists()
    assert (scene_dir / "task" / "scene_description.json").exists()
    assert (scene_dir / "task" / "video_summarization.json").exists()
    state = read_pipeline_state(scene_dir)
    assert state.annotation_export is not None
    outcomes = {emitter.name: emitter for emitter in state.annotation_export.emitters}
    assert outcomes["chunks"].success is True
    assert outcomes["temporal_description"].success is True
    assert outcomes["scene_description"].success is True
    assert outcomes["video_summarization"].success is True


def test_task_skips_daft_pivots_when_contextual_output_disabled(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    task = CaptioningTask(
        config=CaptioningConfig(input_source="original", write_contextual=False),
        captioner=cast(DenseCaptioner, _DaftVideoCaptioner()),
    )

    task.run(DataEntry(media_path=str(media_path), data_path=str(scene_dir)))

    assert not (scene_dir / "contextual" / "chunks.json").exists()
    assert not (scene_dir / "task" / "temporal_description.json").exists()
    assert read_pipeline_state(scene_dir).annotation_export is None


def test_task_uses_scene_directory_media_id_for_pipeline_active_media(tmp_path: Path) -> None:
    scene = tmp_path / "clip_001"
    active = scene / "sidecars" / "active.jpg"
    active.parent.mkdir(parents=True)
    active.touch()
    fake = _FakeCaptioner()
    task = CaptioningTask(
        config=CaptioningConfig(input_source="original"),
        captioner=cast(DenseCaptioner, fake),
    )

    task.run(DataEntry(media_path=str(active), data_path=str(scene)))

    assert fake.scene_ctx is not None
    assert fake.scene_ctx.media_id == "clip_001"
    assert fake.scene_ctx.is_image is True


def test_task_uses_persisted_media_id_for_staged_pipeline_active_media(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "pipeline_data_tmp"
    active = scene / "sidecars" / "active.jpg"
    active.parent.mkdir(parents=True)
    active.touch()
    write_pipeline_state(scene, ScenePipelineState(media_id="clip_001"))
    fake = _FakeCaptioner()
    task = CaptioningTask(
        config=CaptioningConfig(input_source="original"),
        captioner=cast(DenseCaptioner, fake),
    )

    task.run(DataEntry(media_path=str(active), data_path=str(scene)))

    assert fake.scene_ctx is not None
    assert fake.scene_ctx.media_id == "clip_001"
    assert fake.scene_ctx.is_image is True


def test_image_caption_state_does_not_publish_metadata_chunk_json(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    scene = tmp_path / "frame"
    task = CaptioningTask(
        config=CaptioningConfig(input_source="original"),
        captioner=cast(DenseCaptioner, _ImageCaptioner()),
    )

    task.run(DataEntry(media_path=str(image), data_path=str(scene)))

    state = read_pipeline_state(scene)
    caption_artifacts = CaptionArtifactsState.model_validate(
        state.task_artifacts[CAPTION_ARTIFACTS_KEY]
    )
    assert caption_artifacts.success is True
    assert caption_artifacts.metadata_chunk_json is None
    assert caption_artifacts.video_json is None
    assert caption_artifacts.image_json == str(
        scene / "sidecars" / "captioning" / "image_captions.json"
    )


def test_task_preserves_existing_caption_paths_when_run_has_no_new_outputs(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.mp4"
    original.touch()
    scene = tmp_path / "scene"
    write_pipeline_state(
        scene,
        ScenePipelineState(
            task_artifacts={
                CAPTION_ARTIFACTS_KEY: CaptionArtifactsState(
                    success=True,
                    metadata_chunk_json="sidecars/captioning/metadata_chunk.json",
                    video_json="sidecars/captioning/video_captions.json",
                    image_json="sidecars/captioning/image_captions.json",
                ).model_dump()
            }
        ),
    )
    task = CaptioningTask(
        config=CaptioningConfig(input_source="original"),
        captioner=cast(DenseCaptioner, _FailingCaptioner()),
    )

    task.run(DataEntry(media_path=str(original), data_path=str(scene)))

    state = read_pipeline_state(scene)
    caption_artifacts = CaptionArtifactsState.model_validate(
        state.task_artifacts[CAPTION_ARTIFACTS_KEY]
    )
    assert caption_artifacts.success is False
    assert caption_artifacts.metadata_chunk_json is None
    assert caption_artifacts.video_json is None
    assert caption_artifacts.image_json is None


def test_task_fails_for_explicit_missing_tracking_source(tmp_path: Path) -> None:
    original = tmp_path / "original.mp4"
    original.touch()
    scene = tmp_path / "scene"
    task = CaptioningTask(
        config=CaptioningConfig(input_source="tracking"),
        captioner=cast(DenseCaptioner, _FailingCaptioner()),
    )

    with pytest.raises(ValueError, match="requested tracking caption input source is missing"):
        task.run(DataEntry(media_path=str(original), data_path=str(scene)))


def test_task_fails_for_explicit_missing_enhanced_source(tmp_path: Path) -> None:
    original = tmp_path / "original.mp4"
    original.touch()
    scene = tmp_path / "scene"
    write_pipeline_state(
        scene,
        ScenePipelineState(
            enhanced_media=EnhancedMediaState(
                success=True, output_path=str(tmp_path / "missing.mp4")
            )
        ),
    )
    task = CaptioningTask(
        config=CaptioningConfig(input_source="enhanced"),
        captioner=cast(DenseCaptioner, _FailingCaptioner()),
    )

    with pytest.raises(ValueError, match="requested enhanced caption input source is missing"):
        task.run(DataEntry(media_path=str(original), data_path=str(scene)))
