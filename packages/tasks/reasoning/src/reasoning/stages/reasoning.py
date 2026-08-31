# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-scene LLM reasoning-enrichment provider.

Lives next to the emitters because every emitter that supports the
optional ``reasoning`` task field consumes the same provider object.
Built once at the start of :func:`pipeline.run_pipeline`; threaded into
every emitter that opts in via ``extras={"reasoning_provider": ...}``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from reasoning.common import write_daft_json
from reasoning.config import ReasoningConfig, ReasoningTarget
from reasoning.endpoint_resolver import EndpointResolver
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError, PromptRegistry, PromptVariant
from reasoning.reasoning.llm import ReasoningLLMError, generate_reasoning_for_item
from reasoning.stages.base import (  # noqa: F401  (re-export hint)
    build_prompt_registry,
    resolve_config_input_path,
)


class ReasoningProvider:
    """Per-scene LLM caller for the optional ``reasoning`` task field.

    Built once at the start of :func:`run_pipeline` when
    ``reasoning.reasoning.enabled=true`` *and* an LLM endpoint
    resolves; bundles the resolved endpoint, the loaded prompt variant,
    and the call knobs so individual emitters don't have to know about
    any of that. Emitters receive an ``Optional[ReasoningProvider]``
    and call :meth:`reasoning_for` only when it's not ``None`` — every
    other code path (reasoning disabled, no endpoint, prompt missing)
    looks identical from the emitter's side.

    Use-case agnosticism: the prompt is whatever YAML the user picks
    via ``prompt_variant``; the ``targets`` list controls which task
    files get enrichment. Neither is hard-coded here.
    """

    def __init__(
        self,
        *,
        cfg: ReasoningConfig,
        prompt: PromptVariant,
        llm_url: str,
        llm_model: str,
        logger: logging.Logger,
        api_key: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._prompt = prompt
        self._llm_url = llm_url
        self._llm_model = llm_model
        self._api_key = api_key
        self._logger = logger

    @property
    def targets(self) -> tuple[ReasoningTarget, ...]:
        return tuple(self._cfg.targets)

    def applies_to(self, target: ReasoningTarget) -> bool:
        return target in self._cfg.targets

    def reasoning_for(
        self,
        *,
        target: ReasoningTarget,
        question: str,
        answer: str,
        context: str | None = None,
    ) -> str | None:
        """Return a reasoning trace, or ``None`` if not applicable / failed.

        Returning ``None`` rather than raising is the contract: the
        caller falls back to writing the task item without the optional
        ``reasoning`` field. Per-target gating (``applies_to``) lets
        the caller short-circuit before any LLM cost is incurred.
        """
        if not self.applies_to(target):
            return None
        ctx_arg = context if self._cfg.include_context else None
        try:
            return generate_reasoning_for_item(
                question=question,
                answer=answer,
                context=ctx_arg,
                prompt=self._prompt,
                llm_url=self._llm_url,
                llm_model=self._llm_model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
                timeout=self._cfg.timeout,
                seed=self._cfg.seed,
                retries=self._cfg.retries,
                retry_backoff_s=self._cfg.retry_backoff_s,
                max_chars=self._cfg.max_chars,
                api_key=self._api_key,
                logger=self._logger,
                retry_stage=f"reasoning:{target}",
            )
        except (ReasoningLLMError, PromptError) as exc:
            self._logger.warning(
                "[reasoning] %s call failed: %s; emitting item without reasoning",
                target,
                exc,
            )
            return None
        except Exception as exc:
            self._logger.warning(
                "[reasoning] %s unexpected error: %s; emitting item without reasoning",
                target,
                exc,
            )
            return None


def build_reasoning_provider(
    *,
    cfg: ReasoningConfig | None,
    resolver: EndpointResolver | None,
    config_dir: Path,
    logger: logging.Logger,
) -> ReasoningProvider | None:
    """Resolve endpoint + load prompt variant; return ``None`` on any miss.

    Centralizing this here means the per-emitter call sites stay one
    line (``if reasoning_provider is not None:``) and the "best-effort,
    skip on any failure" semantic is enforced at exactly one place.
    """
    if cfg is None or not cfg.enabled:
        return None
    if resolver is None:
        logger.warning(
            "[reasoning] reasoning.reasoning.enabled=true but no EndpointResolver "
            "passed to run_pipeline; reasoning enrichment disabled"
        )
        return None
    llm_url, llm_model = resolver.resolve_llm()
    if not llm_url or not llm_model:
        logger.warning(
            "[reasoning] enabled but no LLM endpoint resolved (config + env); "
            "reasoning enrichment disabled"
        )
        return None

    registry = _registry_or_none(cfg=cfg, config_dir=config_dir, logger=logger)
    if registry is None:
        return None
    try:
        prompt = registry.get(cfg.prompt_variant)
    except PromptError as exc:
        logger.warning("[reasoning] prompt setup failed: %s; reasoning enrichment disabled", exc)
        return None

    return ReasoningProvider(
        cfg=cfg,
        prompt=prompt,
        llm_url=llm_url,
        llm_model=llm_model,
        api_key=resolver.resolve_llm_api_key(),
        logger=logger,
    )


def _registry_or_none(
    *, cfg: ReasoningConfig, config_dir: Path, logger: logging.Logger
) -> PromptRegistry | None:
    """Build a :class:`PromptRegistry` for the reasoning prompt directory."""
    extra_dirs: list[Path] = []
    if cfg.prompt_dir:
        pd = resolve_config_input_path(cfg.prompt_dir, config_dir=config_dir)
        if not pd.is_dir():
            logger.warning(
                "[reasoning] prompt_dir %s does not exist; falling back to bundled prompts only",
                pd,
            )
        else:
            extra_dirs.append(pd)
    try:
        return PromptRegistry(extra_dirs=extra_dirs)
    except PromptError as exc:
        logger.warning(
            "[reasoning] prompt registry setup failed: %s; reasoning enrichment disabled",
            exc,
        )
        return None


def strip_reasoning_from_task_outputs(
    paths: ScenePaths,
    *,
    logger: logging.Logger,
    only_paths: Iterable[Path] | None = None,
) -> None:
    """Drop the optional ``reasoning`` key from every emitted task item.

    Companion to :func:`build_reasoning_provider`: when reasoning is
    disabled (``reasoning.reasoning.enabled=false`` or the section is
    omitted), the upstream LLMs (MCQ generation, QA family, causal
    linkage, temporal localization, the prose reasoning enricher) may
    still surface a ``reasoning`` field as part of their JSON schema.
    Without an explicit strip, the on-disk artifact contains reasoning
    traces even though the operator asked for a no-reasoning run, which
    is what the pipeline produced before this hook existed.

    This walks every task JSON the pipeline writes, deletes the
    ``reasoning`` key from each item it finds, and rewrites only files
    whose content actually changed. Files that don't exist are skipped
    silently (per-stage gating already controls what gets written);
    files where no item had reasoning are left untouched on disk so we
    don't churn mtimes for nothing.

    The chokepoint design (one post-emit pass instead of per-emitter
    flags) keeps the contract simple: any future emitter that adds a
    ``reasoning`` field is automatically covered without needing to
    learn about the disabled-mode policy.
    """
    default_candidates: tuple[Path, ...] = (
        paths.task_mcq,
        paths.task_bcq,
        paths.task_open_qa,
        paths.task_mcq_openended,
        paths.task_bcq_openended,
        paths.task_temporal_localization,
        paths.task_causal_linkage,
        paths.task_scene_description,
        paths.task_video_summarization,
        paths.task_temporal_description,
    )
    allowed = set(default_candidates)
    candidates = (
        tuple(path for path in only_paths if path in allowed)
        if only_paths is not None
        else default_candidates
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[reasoning-strip] could not load %s for cleanup: %s; leaving as-is",
                path,
                exc,
            )
            continue
        if not isinstance(payload, dict):
            continue
        items = payload.get("items")
        if not isinstance(items, list):
            continue
        modified = False
        for item in items:
            if isinstance(item, dict) and "reasoning" in item:
                del item["reasoning"]
                modified = True
        if modified:
            try:
                write_daft_json(path, payload)
                logger.info("[reasoning-strip] removed `reasoning` field from %s", path.name)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(
                    "[reasoning-strip] failed to rewrite %s: %s; leaving as-is",
                    path,
                    exc,
                )


__all__ = [
    "ReasoningProvider",
    "build_reasoning_provider",
    "strip_reasoning_from_task_outputs",
]
