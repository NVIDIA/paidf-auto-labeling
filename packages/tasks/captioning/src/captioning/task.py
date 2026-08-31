# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captioning task implementation."""

from __future__ import annotations

import logging
from pathlib import Path

from core import (
    DataEntry,
    SceneContext,
    ScenePipelineState,
    ensure_scene_skeleton,
    read_pipeline_state,
    scene_context_for_entry,
    update_annotation_export_state,
    write_pipeline_state,
)
from core.formats.daft import emit_captioning_daft_outputs
from core.tasks import SequentialTask

from captioning.artifacts import CAPTION_ARTIFACTS_KEY, CaptionArtifactsState
from captioning.captioner import DenseCaptioner
from captioning.clients import CaptionEndpointClient, create_endpoint_client
from captioning.config import CaptioningConfig

DETECTION_AND_TRACKING_ARTIFACTS_KEY = "detection_and_tracking"
CAPTIONING_DAFT_EMITTERS = (
    "chunks",
    "scene_description",
    "video_summarization",
    "temporal_description",
)


class CaptioningTask(SequentialTask):
    """Sequential task that runs dense captioning for image or video scenes."""

    def __init__(
        self,
        *,
        config: CaptioningConfig | None = None,
        captioner: DenseCaptioner | None = None,
        name: str = "captioning",
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name, max_retries=max_retries)
        self.config = config or CaptioningConfig()
        self.captioner = captioner or _build_captioner(self.config, self.logger)

    def run(self, data_entry: DataEntry) -> DataEntry:
        """Run captioning and update scene pipeline state."""
        if not self.config.enabled:
            return data_entry
        scene_paths = ensure_scene_skeleton(data_entry.data_path)
        original_media = Path(data_entry.media_path)
        scene_ctx = scene_context_for_entry(data_entry)
        pipeline_state = read_pipeline_state(data_entry.data_path)
        media_path = _select_caption_input(
            original_media_path=original_media,
            pipeline_state=pipeline_state,
            input_source=self.config.input_source,
        )
        result = self.captioner.run(
            media_path=media_path,
            scene_paths=scene_paths,
            scene_ctx=scene_ctx,
        )
        pipeline_state.data_entry_id = pipeline_state.data_entry_id or data_entry.id
        pipeline_state.media_path = pipeline_state.media_path or data_entry.media_path
        existing_caption_artifacts = CaptionArtifactsState.model_validate(
            pipeline_state.task_artifacts.get(CAPTION_ARTIFACTS_KEY, {})
        )
        metadata_chunk_json = (
            None if scene_ctx.is_image else existing_caption_artifacts.metadata_chunk_json
        )
        video_json = None if scene_ctx.is_image else existing_caption_artifacts.video_json
        events_json = None if scene_ctx.is_image else existing_caption_artifacts.events_json
        image_json = existing_caption_artifacts.image_json
        if result.success:
            if not scene_ctx.is_image and result.sidecar_json is not None:
                metadata_chunk_json = str(result.sidecar_json)
            if not scene_ctx.is_image and result.video_json is not None:
                video_json = str(result.video_json)
            if scene_ctx.is_image and result.image_json is not None:
                image_json = str(result.image_json)
        else:
            metadata_chunk_json = None
            video_json = None
            events_json = None
            image_json = None
        pipeline_state.task_artifacts[CAPTION_ARTIFACTS_KEY] = CaptionArtifactsState(
            success=result.success,
            metadata_chunk_json=metadata_chunk_json,
            video_json=video_json,
            events_json=events_json,
            image_json=image_json,
        ).model_dump()
        write_pipeline_state(data_entry.data_path, pipeline_state)
        if self.config.write_contextual:
            self._emit_daft_outputs(
                data_entry,
                scene_ctx=scene_ctx,
                caption_artifacts=pipeline_state.task_artifacts[CAPTION_ARTIFACTS_KEY],
                success=result.success,
            )
        return data_entry

    def _emit_daft_outputs(
        self,
        data_entry: DataEntry,
        *,
        scene_ctx: SceneContext,
        caption_artifacts: dict[str, object],
        success: bool,
    ) -> None:
        written: tuple[Path, ...] = ()
        if success:
            written = emit_captioning_daft_outputs(
                data_entry.data_path,
                scene_ctx,
                caption_artifacts=caption_artifacts,
                logger=self.logger,
            )
            if written:
                self.logger.info("Captioning wrote %d DAFT pivot file(s).", len(written))
        _record_daft_outputs(data_entry, written)


def _build_captioner(config: CaptioningConfig, logger: logging.Logger) -> DenseCaptioner:
    vlm = create_endpoint_client(
        provider=config.vlm_provider,
        endpoint_url=config.vlm_endpoint_url,
        model=config.vlm_model,
        timeout_s=config.timeout_s,
        retries=config.retries,
        retry_backoff_s=config.retry_backoff_s,
    )
    llm: CaptionEndpointClient | None = None
    if config.enable_llm_summary:
        provider = config.llm_provider or config.vlm_provider
        llm = create_endpoint_client(
            provider=provider,
            endpoint_url=config.llm_endpoint_url or config.vlm_endpoint_url,
            model=config.llm_model or config.vlm_model,
            timeout_s=config.timeout_s,
            retries=config.retries,
            retry_backoff_s=config.retry_backoff_s,
        )
    logger.debug("Captioning configured with VLM provider %s", config.vlm_provider)
    return DenseCaptioner(config=config, vlm_client=vlm, llm_client=llm)


def _select_caption_input(
    *,
    original_media_path: Path,
    pipeline_state: ScenePipelineState,
    input_source: str,
) -> Path:
    enhanced_media = getattr(pipeline_state, "enhanced_media", None)
    task_artifacts = getattr(pipeline_state, "task_artifacts", {})
    tracking_artifacts = task_artifacts.get(DETECTION_AND_TRACKING_ARTIFACTS_KEY, {})
    tracking_path = _first_existing_path(
        tracking_artifacts.get("annotated_video_path"),
        tracking_artifacts.get("red_id_overlay_path"),
    )
    enhanced_path = _first_existing_path(
        enhanced_media.output_path if enhanced_media is not None else None
    )
    if input_source == "tracking":
        if tracking_path is None:
            raise ValueError("requested tracking caption input source is missing")
        return tracking_path
    if input_source == "enhanced":
        if enhanced_path is None:
            raise ValueError("requested enhanced caption input source is missing")
        return enhanced_path
    if input_source == "original":
        return original_media_path
    return tracking_path or enhanced_path or original_media_path


def _first_existing_path(*values: object) -> Path | None:
    for value in values:
        if value is None:
            continue
        path = Path(str(value))
        if path.exists():
            return path
    return None


def _record_daft_outputs(data_entry: DataEntry, written: tuple[Path, ...]) -> None:
    written_by_name = {path.stem: path for path in written}
    update_annotation_export_state(
        data_entry.data_path,
        data_entry_id=data_entry.id,
        media_path=data_entry.media_path,
        emitter_artifacts={name: written_by_name.get(name) for name in CAPTIONING_DAFT_EMITTERS},
    )


__all__ = ["CaptioningTask"]
