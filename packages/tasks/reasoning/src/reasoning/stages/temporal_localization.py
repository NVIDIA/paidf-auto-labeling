# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal-localization stage emitter (``task/temporal_localization.json``).

Grounds caller-supplied query strings against the per-window VLM
captions in a single LLM call. Query bank can come from inline list
(``cfg.queries``), a YAML/JSON file (``cfg.query_file``), or both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reasoning.common import SceneContext
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError, PromptRegistry
from reasoning.stage_inputs import StageInputs
from reasoning.stages.base import llm_api_key_from_extras, resolve_config_input_path
from reasoning.temporal_localization.converter import to_daft_temporal_localization
from reasoning.temporal_localization.llm import (
    TemporalLocalizationLLMError,
    generate_temporal_localizations_with_llm,
    load_query_bank,
)


class TemporalLocalizationEmitter:
    name = "temporal_localization"
    expected_type = "temporal_localization"
    requires_temporal_axis = True

    @staticmethod
    def output(paths: ScenePaths) -> Path:
        return paths.task_temporal_localization

    def render(
        self,
        *,
        inputs: StageInputs,
        ctx: SceneContext,
        cfg: Any,
        prompts: PromptRegistry,
        llm: tuple[str, str],
        config_dir: Path,
        logger: logging.Logger,
        extras: dict | None = None,  # noqa: ARG002 - unused
    ) -> dict | None:
        windows = inputs.dense_windows() or inputs.windows()
        if not windows:
            logger.debug("[temporal_localization] no windows in sidecar; skipping")
            return None

        qf: Path | None = None
        if cfg.query_file:
            qf = resolve_config_input_path(cfg.query_file, config_dir=config_dir)
        try:
            # ``section="temporal_localization"`` lets a unified
            # cookbook bank carry a dedicated section for this stage
            # alongside ``open_qa`` / ``mcq_openended`` / ``bcq_openended``
            # sections; per-stage files with a bare ``queries: [...]``
            # list keep working unchanged.
            inline_queries = list(cfg.queries) if cfg.queries is not None else []
            queries = load_query_bank(
                queries=inline_queries,
                query_file=qf,
                section="temporal_localization",
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[temporal_localization] query bank load failed: %s; skipping", exc)
            return None
        if not queries:
            logger.warning(
                "[temporal_localization] enabled but query bank is empty after dedup; skipping"
            )
            return None

        scene_desc, _event_sum, duration = inputs.scene_prose()

        try:
            items = generate_temporal_localizations_with_llm(
                queries=queries,
                windows=windows,
                scene_description=scene_desc,
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
        except (TemporalLocalizationLLMError, PromptError) as exc:
            logger.warning("[temporal_localization] LLM call failed: %s; skipping", exc)
            return None

        return to_daft_temporal_localization(items, ctx=ctx, duration=duration)
