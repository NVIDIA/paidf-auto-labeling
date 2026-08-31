# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal-linkage stage emitter (``task/causal_linkage.json``).

Two pair-source modes (``cfg.mode``):

- ``auto_from_events`` — derive (t1, t2) pairs from
  ``contextual/events.json`` (consecutive event start_time pairs,
  trailing event paired to scene end). Zero config.
- ``explicit`` — caller supplies pairs inline (``cfg.pairs``) or via
  YAML/JSON file (``cfg.pair_file``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reasoning.causal_linkage.converter import to_daft_causal_linkage
from reasoning.causal_linkage.llm import (
    CausalLinkageLLMError,
    derive_pairs_from_events,
    generate_causal_linkages_with_llm,
    load_pair_bank,
)
from reasoning.common import SceneContext
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError, PromptRegistry
from reasoning.stage_inputs import StageInputs
from reasoning.stages.base import llm_api_key_from_extras, resolve_config_input_path


class CausalLinkageEmitter:
    name = "causal_linkage"
    expected_type = "causal_linkage"
    requires_temporal_axis = True

    @staticmethod
    def output(paths: ScenePaths) -> Path:
        return paths.task_causal_linkage

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
        extras: dict | None = None,
    ) -> dict | None:
        windows = inputs.dense_windows() or inputs.windows()
        if not windows:
            logger.debug("[causal_linkage] no windows in sidecar; skipping")
            return None
        # Only the duration is needed here; the scene description comes from
        # reasoning_context() below (richer, causal), not scene_prose().
        _, _, duration = inputs.scene_prose()
        # Prefer the synthesized MSTED description (richer, causal) as the
        # grounding context; fall back to the plain scene_description.
        scene_desc = inputs.reasoning_context()

        pairs = self._resolve_pairs(
            inputs=inputs,
            cfg=cfg,
            duration=duration,
            config_dir=config_dir,
            logger=logger,
        )
        if not pairs:
            return None

        try:
            items = generate_causal_linkages_with_llm(
                pairs=pairs,
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
        except (CausalLinkageLLMError, PromptError) as exc:
            logger.warning("[causal_linkage] LLM call failed: %s; skipping", exc)
            return None

        # Optional reasoning enrichment (mirrors QA family).
        reasoning_provider = (extras or {}).get("reasoning_provider")
        if reasoning_provider is not None and reasoning_provider.applies_to("causal_linkage"):
            for item in items:
                existing = item.get("reasoning")
                if isinstance(existing, str) and existing.strip():
                    continue
                question = item.get("question")
                answer = item.get("answer")
                if not isinstance(question, str) or not isinstance(answer, str):
                    continue
                trace = reasoning_provider.reasoning_for(
                    target="causal_linkage",
                    question=question,
                    answer=answer,
                    context=scene_desc,
                )
                if trace:
                    item["reasoning"] = trace

        return to_daft_causal_linkage(items, ctx=ctx, duration=duration)

    @staticmethod
    def _resolve_pairs(
        *,
        inputs: StageInputs,
        cfg: Any,
        duration: float | None,
        config_dir: Path,
        logger: logging.Logger,
    ) -> list:
        if cfg.mode == "auto_from_events":
            events = inputs.event_list()
            if not events:
                logger.warning(
                    "[causal_linkage] mode='auto_from_events' but events.json is "
                    "missing or empty; skipping"
                )
                return []
            # Prefer the config-pinned video_type; otherwise inherit the
            # anomaly stage's verdict (anomaly mode) so auto-derived pairs
            # are tagged anomaly/normal without a second classification.
            default_video_type = cfg.default_video_type or inputs.anomaly_video_type()
            pairs = derive_pairs_from_events(
                events,
                duration=duration,
                max_pairs=cfg.max_pairs_auto,
                default_video_type=default_video_type,
            )
            if not pairs:
                logger.debug("[causal_linkage] no usable pairs derived from events.json; skipping")
            return pairs

        # mode == "explicit"
        pf: Path | None = None
        if cfg.pair_file:
            pf = resolve_config_input_path(cfg.pair_file, config_dir=config_dir)
        try:
            inline = [p.model_dump(exclude_none=True) for p in (cfg.pairs or [])]
            pairs = load_pair_bank(pairs=inline, pair_file=pf)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[causal_linkage] pair bank load failed: %s; skipping", exc)
            return []
        if not pairs:
            logger.warning("[causal_linkage] enabled but pair bank is empty after dedup; skipping")
        return pairs


__all__ = ["CausalLinkageEmitter"]
