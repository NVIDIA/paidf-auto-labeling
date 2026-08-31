# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Primitives shared by every opt-in LLM-driven DAFT-export stage.

See the package docstring for the eight-step scaffold this module
captures. Public surface:

- :func:`build_prompt_registry` — config-dir-relative ``prompt_dir``
  resolver that returns a :class:`PromptRegistry` (or ``None`` when
  setup fails). Replaces the duplicated 15-line ``extra_dirs``
  blocks in each emitter.
- :class:`LlmStageEmitter` — protocol the per-stage emitter modules
  implement.
- :func:`emit_stage` — single source of truth for the eight-step
  best-effort policy.

New exporters add one emitter under :mod:`reasoning.stages`, then wire it
from the reasoning service. The repeated endpoint resolution, prompt registry
setup, conversion-error handling, write-error handling, and success logging stay
here rather than being copied into each stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from reasoning.common import DaftConvertError, DaftType, SceneContext, write_daft_json
from reasoning.endpoint_resolver import EndpointResolver
from reasoning.paths import ScenePaths
from reasoning.prompts import PromptError, PromptRegistry
from reasoning.stage_inputs import StageInputs

# ---------------------------------------------------------------------------
# Prompt registry helper
# ---------------------------------------------------------------------------


PromptRegistryCache = dict[tuple[str, str], PromptRegistry]


def llm_api_key_from_extras(extras: dict | None) -> str | None:
    """Return the explicit LLM API key threaded through stage extras, if any."""
    if extras is None:
        return None
    value = extras.get("llm_api_key")
    return value if isinstance(value, str) else None


def build_prompt_registry(
    *,
    prompt_dir: str | None,
    config_dir: Path,
    logger: logging.Logger,
    tag: str,
    cache: PromptRegistryCache | None = None,
) -> PromptRegistry | None:
    """Resolve ``prompt_dir`` and build a registry.

    ``prompt_dir`` is typically a relative path inside a cookbook. Source
    cookbook configs use repo-root-style values such as
    ``cookbooks/<domain>/prompts``; explicit ``./`` and ``../`` paths remain
    config-relative for cloned experiment configs.
    Missing or non-directory ``prompt_dir`` values fall back to
    bundled prompts only with a single warning. Returns ``None`` only
    when :class:`PromptRegistry` itself fails to load (corrupt YAML,
    missing bundled defaults, etc.).
    """
    extra_dirs: list[Path] = []
    cache_prompt_dir = ""
    pd: Path | None = None
    if prompt_dir:
        pd = resolve_config_input_path(prompt_dir, config_dir=config_dir)
        cache_prompt_dir = str(pd)
    cache_key = (str(config_dir.resolve()), cache_prompt_dir)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    if pd is not None:
        if not pd.is_dir():
            logger.warning(
                "[%s] prompt_dir %s does not exist; falling back to bundled prompts only",
                tag,
                pd,
            )
        else:
            extra_dirs.append(pd)
    try:
        registry = PromptRegistry(extra_dirs=extra_dirs)
    except PromptError as exc:
        logger.warning("[%s] prompt registry setup failed: %s; skipping", tag, exc)
        return None
    if cache is not None:
        cache[cache_key] = registry
    return registry


def resolve_config_input_path(value: str | Path, *, config_dir: Path) -> Path:
    """Resolve source-compatible local config paths.

    Absolute paths pass through. Explicit ``./`` and ``../`` paths resolve
    relative to the config file. Other relative paths first try the current
    working directory/repo root and then the config directory, matching the
    source CLI's handling of cookbook paths such as
    ``cookbooks/traffic/question_bank.json``.
    """
    raw = str(value).strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if raw.startswith(("./", "../")):
        return (config_dir / path).resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    config_candidate = (config_dir / path).resolve()
    if config_candidate.exists():
        return config_candidate

    return cwd_candidate


# ---------------------------------------------------------------------------
# Emitter protocol + scaffold
# ---------------------------------------------------------------------------


class LlmStageEmitter(Protocol):
    """One-method protocol for opt-in LLM-driven DAFT-export stages.

    Implementors live under :mod:`reasoning.stages` (one file per
    stage) and provide:

    - ``name``: short tag used in log lines and skip messages.
    - ``requires_temporal_axis``: ``True`` for stages that don't
      apply to image scenes (msted, temporal_localization,
      temporal_description, chunks). The scaffold short-circuits
      these on image scenes without calling :meth:`render`.
    - ``output(paths)``: where the produced file lands.
    - ``render(...)``: build and return the DAFT payload, or
      ``None`` to mean "nothing to write" (e.g., empty windows).
      The scaffold catches :class:`PromptError` and
      :class:`DaftConvertError`; any other exception is the
      emitter's responsibility (typed adapter errors should be
      caught inside ``render`` and logged at ``WARNING``).
    """

    name: str
    expected_type: DaftType | str
    requires_temporal_axis: bool

    def output(self, paths: ScenePaths) -> Path: ...

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
    ) -> dict | None: ...


def emit_stage(
    emitter: LlmStageEmitter,
    *,
    paths: ScenePaths,
    ctx: SceneContext,
    cfg: Any,
    inputs: StageInputs,
    resolver: EndpointResolver,
    config_dir: Path,
    logger: logging.Logger,
    extras: dict | None = None,
    prompt_cache: PromptRegistryCache | None = None,
) -> Path | None:
    """Run one emitter through the eight-step best-effort policy.

    The single source of truth for image-skip / endpoint-resolution /
    prompt-registry-build / typed-error handling / payload write that
    every legacy ``_emit_*_with_llm`` repeated. Errors are logged at
    ``WARNING`` and the file is skipped — the pipeline rc is never
    affected by an LLM stage failure.

    ``extras``: optional opaque dict forwarded to the emitter's
    :meth:`render`. Used by emitters that need stage-specific
    auxiliaries the protocol doesn't model (today: the QA family
    pulls a ``reasoning_provider`` from here for in-place item
    enrichment before the converter runs).
    """
    tag = emitter.name
    if emitter.requires_temporal_axis and ctx.is_image:
        logger.debug("[%s] image scene; skipping (no temporal axis)", tag)
        return None

    llm_url, llm_model = resolver.resolve_llm()
    if not (llm_url and llm_model):
        logger.warning("[%s] enabled but no LLM endpoint resolved; skipping", tag)
        return None
    render_extras = dict(extras or {})
    resolve_api_key = getattr(resolver, "resolve_llm_api_key", None)
    llm_api_key = resolve_api_key() if callable(resolve_api_key) else None
    if llm_api_key is not None:
        render_extras.setdefault("llm_api_key", llm_api_key)

    prompts = build_prompt_registry(
        prompt_dir=getattr(cfg, "prompt_dir", None),
        config_dir=config_dir,
        logger=logger,
        tag=tag,
        cache=prompt_cache,
    )
    if prompts is None:
        return None

    try:
        payload = emitter.render(
            inputs=inputs,
            ctx=ctx,
            cfg=cfg,
            prompts=prompts,
            llm=(llm_url, llm_model),
            config_dir=config_dir,
            logger=logger,
            extras=render_extras or None,
        )
    except PromptError as exc:
        logger.warning("[%s] prompt setup failed: %s; skipping", tag, exc)
        return None
    except DaftConvertError as exc:
        logger.warning("[%s] schema conversion failed: %s; skipping", tag, exc)
        return None

    if payload is None:
        return None

    out = emitter.output(paths)
    try:
        write_daft_json(out, payload, expected_type=emitter.expected_type)
    except Exception as exc:
        logger.warning("[%s] failed to write %s: %s; skipping", tag, out, exc, exc_info=True)
        return None
    logger.info("[%s] wrote %s", tag, out)
    return out
