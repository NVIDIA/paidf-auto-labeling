# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM JSON generator — wraps ``VideoPipeline`` (library).

Corresponds to ``vlm_json.enabled = true``.

The ``VideoPipeline`` client (VLM API configuration) is constructed ONCE in
``__init__``; ``generate()`` calls ``process_video()`` directly per sample.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.schema.config import PipelineConfig
from config.normalize import resolve_input_path
from daft_export.paths import scene_paths
from vlm_json.base import BaseVlmJsonGenerator, VlmJsonResult
from vlm_json.runners.video_pipeline import PipelineConfig as BpvPipelineConfig
from vlm_json.runners.video_pipeline import PromptFiles, VideoConfig, VideoPipeline, VLMConfig


def _resolve_prompt_path(
    path_str: Optional[str], *, config_dir: Optional[str], repo_root: Optional[Path], logger: logging.Logger
) -> Optional[Path]:
    """Resolve prompt paths with the same repo-root-first rules as other input assets."""
    raw = str(path_str or "").strip()
    if not raw:
        return None

    config_base = Path(config_dir) if config_dir else Path.cwd()
    repo_base = repo_root or Path.cwd()
    candidate = Path(resolve_input_path(raw, config_dir=config_base, repo_root=repo_base)).expanduser()
    if candidate.exists():
        return candidate
    logger.warning("VLM JSON prompt file not found: %s", raw)
    return None


def _resolve_required_prompt_path(
    path_str: str, *, config_dir: Optional[str], repo_root: Optional[Path], logger: logging.Logger
) -> Path:
    path = _resolve_prompt_path(path_str, config_dir=config_dir, repo_root=repo_root, logger=logger)
    if path is None:
        raise FileNotFoundError(f"VLM JSON prompt file not found: {path_str}")
    return path


class VlmJsonGenerator(BaseVlmJsonGenerator):
    """VLM JSON generation via ``VideoPipeline``.

    Corresponds to ``vlm_json.enabled = true``.

    ``__init__`` constructs the ``VideoPipeline`` (and its ``QwenVLMClient``)
    once; ``generate()`` calls ``process_video()`` directly per sample.
    """

    def __init__(
        self,
        config: PipelineConfig,
        resolver: EndpointResolver,
        logger: logging.Logger,
        config_dir: Optional[str] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        super().__init__(logger)

        vlm_cfg = config.vlm_json
        vlm_url, vlm_model = resolver.resolve_vlm(required=True)

        scene_prompt = _resolve_prompt_path(
            vlm_cfg.scene_prompt_file, config_dir=config_dir, repo_root=repo_root, logger=logger
        )
        events_prompt = _resolve_prompt_path(
            vlm_cfg.events_prompt_file, config_dir=config_dir, repo_root=repo_root, logger=logger
        )
        prompt_files = PromptFiles(
            video_json=scene_prompt
            or _resolve_required_prompt_path(
                vlm_cfg.default_video_json_prompt_file,
                config_dir=config_dir,
                repo_root=repo_root,
                logger=logger,
            ),
            events_json=events_prompt
            or _resolve_required_prompt_path(
                vlm_cfg.default_video_events_prompt_file,
                config_dir=config_dir,
                repo_root=repo_root,
                logger=logger,
            ),
            image_json=scene_prompt
            or _resolve_required_prompt_path(
                vlm_cfg.default_image_json_prompt_file,
                config_dir=config_dir,
                repo_root=repo_root,
                logger=logger,
            ),
        )

        bpv_config = BpvPipelineConfig(
            video_config=VideoConfig(),
            vlm_config=VLMConfig(
                base_url=vlm_url,
                model=vlm_model,
                max_tokens=int(vlm_cfg.max_tokens),
                temperature=float(vlm_cfg.temperature),
                timeout=int(vlm_cfg.timeout),
                retries=resolver.vlm_retries,
                structured_output=str(vlm_cfg.structured_output),
                frame_fps=float(vlm_cfg.frame_fps),
                resolution=int(vlm_cfg.resolution),
                max_frames=int(vlm_cfg.max_frames),
            ),
            rate_limit=float(vlm_cfg.rate_limit),
            prompt_files=prompt_files,
            split_json_calls=bool(vlm_cfg.split_json_calls),
        )

        self._pipeline = VideoPipeline(bpv_config, logger)

    # ------------------------------------------------------------------

    def generate(self, video_path: Path, output_dir: Path) -> VlmJsonResult:
        """Run VLM JSON generation on a single video or image input.

        Writes ``contextual/video.json`` + ``contextual/events.json`` for video
        inputs, or ``contextual/image.json`` for image inputs. The
        ``VideoPipeline`` client is reused across calls.
        """
        video_path = Path(video_path)
        scene_dir = Path(output_dir)
        paths = scene_paths(scene_dir)

        result = self._pipeline.process_video(video_path, scene_dir)

        success = bool(result.get("success", False))
        return VlmJsonResult(
            success=success,
            events_json=paths.contextual_events if paths.contextual_events.exists() else None,
            video_json=paths.contextual_video if paths.contextual_video.exists() else None,
            image_json=paths.contextual_image if paths.contextual_image.exists() else None,
        )
