# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Events stage emitter (``contextual/events.json``).

LLM aggregates per-window dense captions (``sidecars/metadata_chunk.json``)
into a list of noteworthy temporal events grounded in an
``contextual/instances.json`` catalogue. See
:class:`reasoning.config.EventsConfig` for tunable knobs.

This stage **replaces** the events.json that the upstream
``vlm_json`` whole-clip pass writes. The aggregator has access to all
per-window dense prose and the instances catalogue, so it can produce a richer,
multi-entity, full-duration events list.

Ordering: this stage runs as the **first** entry in the LLM-stage
dispatch table so that downstream consumers of events.json (notably
:class:`reasoning.stages.causal_linkage.CausalLinkageEmitter` in
``mode=auto_from_events``) see the freshly aggregated events rather than
the whole-clip VLM's events. The service re-loads ``StageInputs.scene_events``
between this stage and the rest of the LLM-stage loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reasoning.common import SceneContext
from reasoning.events.converter import build_object_id_catalogue, to_daft_events
from reasoning.events.llm import EventsLLMError, generate_events_with_llm
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError, PromptRegistry
from reasoning.stage_inputs import StageInputs
from reasoning.stages.base import llm_api_key_from_extras


class EventsEmitter:
    """LLM-aggregated ``contextual/events.json`` emitter.

    Reads dense per-window captions from ``sidecars/metadata_chunk.json``
    (via :meth:`StageInputs.dense_windows`) and the instances catalogue
    from ``contextual/instances.json`` (via
    :meth:`StageInputs.instances_catalogue_entries`). When ``dense_windows``
    is empty (the dense-caption stage didn't run), falls back to the
    MCQ-window sidecar for windows so the stage still works on legacy
    pipelines — but the dense-caption stage is the recommended source
    because its prose is bank-mapping-free.
    """

    name = "events"
    expected_type = "events"
    requires_temporal_axis = True

    @staticmethod
    def output(paths: ScenePaths) -> Path:
        return paths.contextual_events

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
        # MCQ window sidecar when ``dense_caption`` didn't run. Same
        # priority as the chunks/temporal_description pivots and the
        # MSTED stage — keeps the four window-derived contextual files
        # consistent in their data sourcing.
        windows = inputs.dense_windows() or inputs.windows()
        if not windows:
            logger.debug(
                "[events] no windows in any sidecar; "
                "skipping events.json (would have nothing to ground in)"
            )
            return None

        catalogue_entries = inputs.instances_catalogue_entries()
        valid_object_ids = build_object_id_catalogue(inputs.scene_instances)
        _, _, duration = inputs.scene_prose()

        try:
            structured = generate_events_with_llm(
                windows=windows,
                instances_catalogue=catalogue_entries or None,
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
        except (EventsLLMError, PromptError) as exc:
            logger.warning(
                "[events] LLM aggregation failed: %s; "
                "skipping events.json (existing file, if any, is left untouched)",
                exc,
            )
            return None

        return to_daft_events(
            structured,
            ctx=ctx,
            duration=duration,
            max_events=cfg.max_events,
            valid_object_ids=valid_object_ids if valid_object_ids else None,
        )


__all__ = ["EventsEmitter"]
