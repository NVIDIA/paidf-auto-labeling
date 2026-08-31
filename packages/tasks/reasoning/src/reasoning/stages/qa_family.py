# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-ended QA family emitters (open_qa, mcq_openended, bcq_openended).

All three share input plumbing (windows + scene prose, or image
caption for image scenes) and a single LLM dispatcher
(``generate_qa_with_llm``). They differ only in:

- the bank loader / config attribute that supplies the question/item
  bank,
- the QA "kind" enum that selects the guided-JSON schema and answer
  regex,
- the converter that turns the LLM items into a DAFT payload,
- the ``ScenePaths`` field where the file lands.

These emitters apply to BOTH video and image scenes. The DAFT QA
schemas accept ``video_id`` or ``image_id`` via ``oneOf``; for image
scenes we surface ``image.json``'s ``caption`` as the scene-level
prose so the LLM has something concrete to ground in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reasoning.common import SceneContext
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError, PromptRegistry
from reasoning.qa.converter import (
    to_daft_bcq_openended,
    to_daft_mcq_openended,
    to_daft_open_qa,
)
from reasoning.qa.llm import QaKind, QaLLMError, generate_qa_with_llm, load_qa_bank
from reasoning.stage_inputs import StageInputs
from reasoning.stages.base import llm_api_key_from_extras, resolve_config_input_path

# ---------------------------------------------------------------------------
# Bank loaders (per-family; tiny shims around load_qa_bank so each emitter
# stays declarative).
# ---------------------------------------------------------------------------


def _load_questions_bank(
    cfg: Any, *, config_dir: Path, kind: QaKind, logger: logging.Logger
) -> list[dict] | None:
    qf: Path | None = None
    if cfg.question_file:
        qf = resolve_config_input_path(cfg.question_file, config_dir=config_dir)
    try:
        # Pass ``section=kind`` so a unified cookbook bank file
        # (e.g. ``cookbooks/warehouse/question_bank.json`` carrying
        # both ``open_qa: [...]`` and ``bcq_openended: [...]`` sections)
        # serves the right slice for this caller. Per-stage files with
        # a bare ``questions: [...]`` list keep working unchanged --
        # the loader falls back to legacy shapes when the requested
        # section isn't present.
        bank = load_qa_bank(questions=list(cfg.questions), question_file=qf, section=kind)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("[%s] question bank load failed: %s; skipping", kind, exc)
        return None
    if not bank:
        logger.warning("[%s] enabled but question bank is empty after dedup; skipping", kind)
        return None
    return bank


def _load_items_bank(
    cfg: Any, *, config_dir: Path, kind: QaKind, logger: logging.Logger
) -> list[dict] | None:
    fp: Path | None = None
    if cfg.item_file:
        fp = resolve_config_input_path(cfg.item_file, config_dir=config_dir)
    try:
        inline = [it.model_dump(exclude_none=True) for it in cfg.items]
        # ``mcq_openended`` is the only QA family that takes
        # ``items``-shaped entries (with letter-keyed ``options``);
        # ``section=kind`` selects the matching slice from a unified
        # cookbook bank. Bare ``items: [...]`` files still work.
        bank = load_qa_bank(questions=inline, question_file=fp, section=kind)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("[%s] item bank load failed: %s; skipping", kind, exc)
        return None
    if not bank:
        logger.warning("[%s] enabled but item bank is empty after dedup; skipping", kind)
        return None
    return bank


# ---------------------------------------------------------------------------
# Shared driver
# ---------------------------------------------------------------------------


def _drive_qa_emitter(
    *,
    inputs: StageInputs,
    ctx: SceneContext,
    cfg: Any,
    prompts: PromptRegistry,
    llm: tuple[str, str],
    logger: logging.Logger,
    kind: QaKind,
    bank: list[dict],
    converter: Callable[..., dict | None],
    reasoning_provider: Any = None,
    api_key: str | None = None,
) -> dict | None:
    """Three-into-one QA driver.

    Image vs video mode is handled here so each emitter's ``render``
    stays a single function-call: pull the bank for your kind, call
    here, return the payload.
    """
    windows: list[dict] | None = None
    image_caption: str | None = None
    scene_desc: str | None = None
    event_sum: str | None = None

    if ctx.is_image:
        image_caption = inputs.image_caption()
        if image_caption is None:
            logger.info("[%s] image scene has no usable caption in image.json; skipping", kind)
            return None
        # Image scenes don't carry a video.json; surface the caption as the
        # scene-level description so reasoning enrichment has something
        # concrete to ground in (parallels how prose tasks treat
        # ``caption`` for images).
        scene_desc = image_caption
    else:
        windows = inputs.dense_windows() or inputs.windows()
        if not windows:
            logger.debug("[%s] no windows in sidecar; skipping", kind)
            return None
        _scene_desc, event_sum, _duration = inputs.scene_prose()
        # Prefer the synthesized MSTED description (richer, causal) as the
        # scene-level grounding; fall back to the plain scene_description.
        scene_desc = inputs.reasoning_context()

    try:
        items = generate_qa_with_llm(
            kind=kind,
            bank=bank,
            windows=windows,
            image_caption=image_caption,
            scene_description=scene_desc,
            event_summary=event_sum,
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
            api_key=api_key,
            logger=logger,
        )
    except (QaLLMError, PromptError) as exc:
        logger.warning("[%s] LLM call failed: %s; skipping", kind, exc)
        return None

    # Optional reasoning enrichment: fill ``reasoning`` on items the QA
    # LLM left without one. Mutates items in place; converter then sees
    # the enriched list.
    if reasoning_provider is not None and reasoning_provider.applies_to(kind):
        for item in items:
            existing = item.get("reasoning")
            if isinstance(existing, str) and existing.strip():
                continue
            question = item.get("question")
            answer = item.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            trace = reasoning_provider.reasoning_for(
                target=kind,
                question=question,
                answer=answer,
                context=scene_desc,
            )
            if trace:
                item["reasoning"] = trace

    return converter(items, ctx=ctx)


# ---------------------------------------------------------------------------
# Per-family emitters
# ---------------------------------------------------------------------------


class OpenQaEmitter:
    name = "open_qa"
    expected_type = "open_qa"
    requires_temporal_axis = False  # works on both image and video scenes

    @staticmethod
    def output(paths: ScenePaths) -> Path:
        return paths.task_open_qa

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
        bank = _load_questions_bank(cfg, config_dir=config_dir, kind="open_qa", logger=logger)
        if bank is None:
            return None
        return _drive_qa_emitter(
            inputs=inputs,
            ctx=ctx,
            cfg=cfg,
            prompts=prompts,
            llm=llm,
            logger=logger,
            kind="open_qa",
            bank=bank,
            converter=to_daft_open_qa,
            reasoning_provider=(extras or {}).get("reasoning_provider"),
            api_key=llm_api_key_from_extras(extras),
        )


class McqOpenendedEmitter:
    name = "mcq_openended"
    expected_type = "mcq_openended"
    requires_temporal_axis = False

    @staticmethod
    def output(paths: ScenePaths) -> Path:
        return paths.task_mcq_openended

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
        bank = _load_items_bank(cfg, config_dir=config_dir, kind="mcq_openended", logger=logger)
        if bank is None:
            return None
        return _drive_qa_emitter(
            inputs=inputs,
            ctx=ctx,
            cfg=cfg,
            prompts=prompts,
            llm=llm,
            logger=logger,
            kind="mcq_openended",
            bank=bank,
            converter=to_daft_mcq_openended,
            reasoning_provider=(extras or {}).get("reasoning_provider"),
            api_key=llm_api_key_from_extras(extras),
        )


class BcqOpenendedEmitter:
    name = "bcq_openended"
    expected_type = "bcq_openended"
    requires_temporal_axis = False

    @staticmethod
    def output(paths: ScenePaths) -> Path:
        return paths.task_bcq_openended

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
        bank = _load_questions_bank(cfg, config_dir=config_dir, kind="bcq_openended", logger=logger)
        if bank is None:
            return None
        return _drive_qa_emitter(
            inputs=inputs,
            ctx=ctx,
            cfg=cfg,
            prompts=prompts,
            llm=llm,
            logger=logger,
            kind="bcq_openended",
            bank=bank,
            converter=to_daft_bcq_openended,
            reasoning_provider=(extras or {}).get("reasoning_provider"),
            api_key=llm_api_key_from_extras(extras),
        )


__all__ = ["BcqOpenendedEmitter", "McqOpenendedEmitter", "OpenQaEmitter"]
