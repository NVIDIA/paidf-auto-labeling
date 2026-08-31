# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anomaly-mode stage (``sidecars/reasoning/anomaly.json``).

Unlike every other stage in :mod:`reasoning.stages`, anomaly output is a
*sidecar*, not a DAFT contextual/task type (the DAFT v3 type set is a
closed enum). It therefore cannot ride the :func:`emit_stage` scaffold,
which is hard-wired to :func:`write_daft_json`. This module provides a
parallel, best-effort runner that reuses the same building blocks
(endpoint resolution, prompt registry, the LLM adapter) but writes a
plain JSON sidecar instead.

The runner returns the normalized verdict so the orchestrator can feed
the ``anomaly`` / ``normal`` label into the causal-linkage stage's
``video_type`` auto-tagging, closing the loop between "classify the
clip" and "ground the causal pairs as anomaly/normal".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.utils.io import write_json

from reasoning.anomaly.llm import AnomalyLLMError, classify_anomaly_with_llm
from reasoning.anomaly.sidecar import to_anomaly_sidecar
from reasoning.common import SceneContext
from reasoning.endpoint_resolver import EndpointResolver
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError
from reasoning.stage_inputs import StageInputs
from reasoning.stages.base import build_prompt_registry

_TAG = "anomaly"


def run_anomaly_stage(
    *,
    paths: ScenePaths,
    ctx: SceneContext,
    cfg: Any,
    inputs: StageInputs,
    resolver: EndpointResolver,
    config_dir: Path,
    logger: logging.Logger,
    prompt_cache: dict | None = None,
) -> dict[str, Any] | None:
    """Classify the clip, write the anomaly sidecar, return the verdict.

    Best-effort and side-effecting: any failure (image scene, no
    endpoint, no windows, LLM error, write error) is logged at
    ``WARNING``/``DEBUG`` and yields ``None`` so the pipeline rc is never
    affected. On success the ``sidecars/reasoning/anomaly.json`` file is
    written and the normalized verdict dict is returned.

    Anomaly classification needs a temporal axis (it reasons over
    per-window captions), so image scenes are skipped like the other
    temporal stages.
    """
    if ctx.is_image:
        logger.debug("[%s] image scene; skipping (no temporal axis)", _TAG)
        return None

    windows = inputs.dense_windows() or inputs.windows()
    if not windows:
        logger.debug("[%s] no windows in any sidecar; skipping anomaly.json", _TAG)
        return None

    llm_url, llm_model = resolver.resolve_llm()
    if not (llm_url and llm_model):
        logger.warning("[%s] enabled but no LLM endpoint resolved; skipping", _TAG)
        return None
    resolve_api_key = getattr(resolver, "resolve_llm_api_key", None)
    api_key = resolve_api_key() if callable(resolve_api_key) else None

    prompts = build_prompt_registry(
        prompt_dir=getattr(cfg, "prompt_dir", None),
        config_dir=config_dir,
        logger=logger,
        tag=_TAG,
        cache=prompt_cache,
    )
    if prompts is None:
        return None

    scene_desc = inputs.reasoning_context()

    person_attributes = (
        inputs.person_attributes_block()
        if getattr(cfg, "include_person_attributes", True)
        else None
    )
    if person_attributes:
        logger.debug("[%s] folding person-attribute-search evidence into prompt", _TAG)

    try:
        verdict = classify_anomaly_with_llm(
            windows=windows,
            scene_description=scene_desc,
            person_attributes=person_attributes,
            prompt=prompts.get(cfg.prompt_variant),
            llm_url=llm_url,
            llm_model=llm_model,
            localize_highlight=cfg.localize_highlight,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            timeout=cfg.timeout,
            structured_output=cfg.structured_output,
            seed=cfg.seed,
            retries=cfg.retries,
            retry_backoff_s=cfg.retry_backoff_s,
            description_keys=tuple(cfg.description_keys),
            api_key=api_key,
            logger=logger,
        )
    except (AnomalyLLMError, PromptError) as exc:
        logger.warning("[%s] classification failed: %s; skipping anomaly.json", _TAG, exc)
        return None

    payload = to_anomaly_sidecar(verdict, model=llm_model, sources=["sidecars/metadata.json"])

    out = paths.sidecar_anomaly
    try:
        write_json(out, payload)
    except OSError as exc:
        logger.warning("[%s] failed to write %s: %s; skipping", _TAG, out, exc)
        return None
    logger.info("[%s] wrote %s (classification=%s)", _TAG, out, payload.get("classification"))
    return verdict


__all__ = ["run_anomaly_stage"]
