# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core task adapter for super-resolution."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from core import (
    DataEntry,
    EnhancedMediaState,
    ensure_scene_skeleton,
    read_pipeline_state,
    write_pipeline_state,
)
from core.tasks import SequentialTask

from super_resolution.config import SuperResolutionConfig
from super_resolution.factory import create_resolver
from super_resolution.resolver import Resolver, SrResult


@dataclass(frozen=True)
class MediaResolution:
    """Pixel dimensions for one input media item."""

    width: int
    height: int

    @property
    def short_side(self) -> int:
        """Return the shorter image/video side in pixels."""
        return min(self.width, self.height)

    @property
    def long_side(self) -> int:
        """Return the longer image/video side in pixels."""
        return max(self.width, self.height)


ResolutionProbe = Callable[[Path], MediaResolution]


class SuperResolutionTask(SequentialTask):
    """Run super-resolution and persist enhanced media state."""

    def __init__(
        self,
        config: SuperResolutionConfig | None = None,
        *,
        resolver: Resolver | None = None,
        resolution_probe: ResolutionProbe | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "super_resolution")
        self.config = config or SuperResolutionConfig()
        self.resolver = resolver
        self.resolution_probe = resolution_probe or probe_media_resolution
        if (
            self.config.enabled
            and self.config.resolution_policy == "always"
            and self.resolver is None
        ):
            self.resolver = create_resolver(self.config.resolver, self.logger, self.config)

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        if not self.config.enabled:
            self.logger.info("Super-resolution disabled; leaving media unchanged.")
            return data_entry

        scene_paths = ensure_scene_skeleton(data_entry.data_path)
        media_path = Path(data_entry.media_path)
        pipeline_state = read_pipeline_state(data_entry.data_path)
        pipeline_state.data_entry_id = pipeline_state.data_entry_id or data_entry.id
        pipeline_state.media_path = pipeline_state.media_path or data_entry.media_path

        if not self._should_run_super_resolution(media_path):
            pipeline_state.enhanced_media = EnhancedMediaState(success=False, output_path=None)
            write_pipeline_state(data_entry.data_path, pipeline_state)
            return data_entry

        if self.resolver is None:
            self.resolver = create_resolver(self.config.resolver, self.logger, self.config)

        result: SrResult = self.resolver.run(media_path, scene_paths)
        output_path = Path(result.output_path) if result.output_path else None
        missing_success_output = (
            result.success and output_path is not None and not output_path.exists()
        )
        if missing_success_output:
            self.logger.warning(
                "Super-resolution reported success but output is missing: %s",
                output_path,
            )

        pipeline_state.enhanced_media = EnhancedMediaState(
            success=result.success and not missing_success_output,
            output_path=str(output_path) if output_path else None,
        )
        write_pipeline_state(data_entry.data_path, pipeline_state)
        if result.success and output_path and output_path.exists():
            updated_entry: DataEntry = data_entry.model_copy(
                update={"media_path": str(output_path)}
            )
            return updated_entry
        return data_entry

    def _should_run_super_resolution(self, media_path: Path) -> bool:
        if self.config.resolution_policy == "always":
            return True

        try:
            resolution = self.resolution_probe(media_path)
        except Exception as exc:
            self.logger.warning(
                "Could not probe media resolution for %s; running super-resolution. Error: %s",
                media_path,
                exc,
            )
            return True

        should_run = (
            resolution.short_side < self.config.min_input_short_side
            or resolution.long_side < self.config.min_input_long_side
        )
        if should_run:
            self.logger.info(
                "Running super-resolution for %s: input resolution %dx%d is below "
                "auto policy thresholds short_side>=%d and long_side>=%d.",
                media_path,
                resolution.width,
                resolution.height,
                self.config.min_input_short_side,
                self.config.min_input_long_side,
            )
        else:
            self.logger.info(
                "Skipping super-resolution for %s: input resolution %dx%d meets "
                "auto policy thresholds short_side>=%d and long_side>=%d.",
                media_path,
                resolution.width,
                resolution.height,
                self.config.min_input_short_side,
                self.config.min_input_long_side,
            )
        return should_run


def probe_media_resolution(media_path: Path) -> MediaResolution:
    """Probe the visible media dimensions without running the SR resolver."""
    if not media_path.exists() or not media_path.is_file():
        raise FileNotFoundError(f"Media file does not exist: {media_path}")

    try:
        return _probe_image_resolution(media_path)
    except Exception:
        return _probe_video_resolution(media_path)


def _probe_image_resolution(media_path: Path) -> MediaResolution:
    image_module: Any = importlib.import_module("PIL.Image")
    with image_module.open(media_path) as image:
        width, height = image.size
    return MediaResolution(width=int(width), height=int(height))


def _probe_video_resolution(media_path: Path) -> MediaResolution:
    av_module: Any = importlib.import_module("av")
    with av_module.open(str(media_path)) as container:
        for stream in container.streams.video:
            width = int(getattr(stream, "width", 0) or 0)
            height = int(getattr(stream, "height", 0) or 0)
            if width > 0 and height > 0:
                return MediaResolution(width=width, height=height)
    raise ValueError(f"Could not find a video stream with dimensions: {media_path}")


__all__ = ["MediaResolution", "SuperResolutionTask", "probe_media_resolution"]
