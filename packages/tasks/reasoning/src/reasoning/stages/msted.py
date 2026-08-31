# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MSTED stage emitter (``contextual/msted.json``).

LLM aggregates per-window VLM captions into a single per-scene event
characterization. See :class:`reasoning.config.MstedConfig`
for tunable knobs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reasoning.common import SceneContext
from reasoning.msted.converter import to_daft_msted
from reasoning.msted.llm import MstedLLMError, generate_msted_with_llm
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError, PromptRegistry
from reasoning.stage_inputs import StageInputs
from reasoning.stages.base import llm_api_key_from_extras


class MstedEmitter:
    name = "msted"
    expected_type = "msted"
    requires_temporal_axis = True

    @staticmethod
    def output(paths: ScenePaths) -> Path:
        return paths.contextual_msted

    def render(
        self,
        *,
        inputs: StageInputs,
        ctx: SceneContext,
        cfg: Any,
        prompts: PromptRegistry,
        llm: tuple[str, str],
        config_dir: Path,  # noqa: ARG002 - unused, present for protocol uniformity
        logger: logging.Logger,
        extras: dict | None = None,  # noqa: ARG002 - unused
    ) -> dict | None:
        # Prefer the clean-prose dense-caption sidecar; fall back to the
        # MCQ window sidecar when ``dense_caption`` didn't run. This matches
        # the service-owned captioning pivot priority in core.
        windows = inputs.dense_windows() or inputs.windows()
        if not windows:
            logger.debug("[msted] no windows in any sidecar; skipping msted.json")
            return None

        scene_desc, event_sum, duration = inputs.scene_prose()

        try:
            structured = generate_msted_with_llm(
                windows=windows,
                scene_description=scene_desc,
                event_summary=event_sum,
                duration=duration,
                prompt=prompts.get(cfg.prompt_variant),
                llm_url=llm[0],
                llm_model=llm[1],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                timeout=cfg.timeout,
                structured_output=cfg.structured_output,
                seed=cfg.seed,
                retries=cfg.retries,
                retry_backoff_s=cfg.retry_backoff_s,
                description_keys=tuple(cfg.description_keys),
                api_key=llm_api_key_from_extras(extras),
                logger=logger,
            )
        except (MstedLLMError, PromptError) as exc:
            logger.warning("[msted] LLM aggregation failed: %s; skipping msted.json", exc)
            return None

        return to_daft_msted(
            structured,
            ctx=ctx,
            sources=["sidecars/metadata.json"],
            duration=duration,
            max_segments=cfg.max_segments,
        )
