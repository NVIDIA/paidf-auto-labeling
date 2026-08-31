# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning task and compatibility exports for service-owned DAFT task pivots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, override

from core import (
    DataEntry,
    SceneContext,
    scene_context_for_entry,
    stage_raw_media,
    update_annotation_export_state,
)
from core.formats.daft.converters.tasks import to_daft_tasks
from core.tasks import SequentialTask

from reasoning.artifacts import load_stage_inputs
from reasoning.config import DaftExportConfig
from reasoning.endpoint_resolver import EndpointResolver
from reasoning.paths import ScenePaths, ensure_scene_skeleton
from reasoning.stages.anomaly import run_anomaly_stage
from reasoning.stages.base import PromptRegistryCache, StageInputs, emit_stage
from reasoning.stages.causal_linkage import CausalLinkageEmitter
from reasoning.stages.events import EventsEmitter
from reasoning.stages.msted import MstedEmitter
from reasoning.stages.qa_family import BcqOpenendedEmitter, McqOpenendedEmitter, OpenQaEmitter
from reasoning.stages.reasoning import (
    ReasoningProvider,
    build_reasoning_provider,
    strip_reasoning_from_task_outputs,
)
from reasoning.stages.temporal_localization import TemporalLocalizationEmitter

ReasoningMode = Literal["config", "keep", "strip"]


class ReasoningTask(SequentialTask):
    """Run opt-in LLM reasoning/export stages over an existing DAFT scene."""

    def __init__(
        self,
        *,
        config: DaftExportConfig | None = None,
        resolver: EndpointResolver | None = None,
        config_dir: Path | None = None,
        reasoning_mode: ReasoningMode = "config",
        name: str | None = None,
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name or "reasoning", max_retries=max_retries)
        self.config = config
        self.resolver = resolver or EndpointResolver(None, logger=self.logger)
        self.config_dir = config_dir or Path.cwd()
        self.reasoning_mode = reasoning_mode

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        paths = ensure_scene_skeleton(data_entry.data_path)
        ctx = scene_context_for_entry(data_entry)
        # Stage the analyzed media under ``raw/<media_id>.<ext>`` so the scene is
        # self-contained for a downstream ``training_export`` stage. The reasoning
        # pipeline has no detection stage to materialize it, and when the run uses
        # a reasoning-only output directory (separate from an EPA/detection run)
        # nothing else populates ``raw/``. Symlink to avoid duplicating video.
        stage_raw_media(
            raw_dir=paths.raw_dir,
            media_path=Path(data_entry.media_path),
            media_id=ctx.media_id,
            copy_media=False,
            logger=self.logger,
        )
        emitted_paths = self._run_entry(
            paths=paths,
            ctx=ctx,
            cfg=self.config,
            resolver=self.resolver,
            config_dir=self.config_dir,
            reasoning_mode=self.reasoning_mode,
        )
        _record_pipeline_state(paths, data_entry, cfg=self.config, emitted_paths=emitted_paths)
        return data_entry

    def _run_entry(
        self,
        *,
        paths: ScenePaths,
        ctx: SceneContext,
        cfg: DaftExportConfig | None,
        resolver: EndpointResolver,
        config_dir: Path,
        reasoning_mode: ReasoningMode,
    ) -> set[Path]:
        inputs = load_stage_inputs(paths, logger=self.logger)
        reasoning_provider = build_reasoning_provider(
            cfg=cfg.reasoning if cfg is not None else None,
            resolver=resolver,
            config_dir=config_dir,
            logger=self.logger,
        )
        emitted_paths: set[Path] = set()

        if cfg is not None:
            _, llm_emitted_paths = self._run_llm_stages(
                paths=paths,
                ctx=ctx,
                cfg=cfg,
                inputs=inputs,
                resolver=resolver,
                config_dir=config_dir,
                reasoning_provider=reasoning_provider,
            )
            emitted_paths.update(llm_emitted_paths)

        if _should_strip_reasoning(cfg=cfg, mode=reasoning_mode):
            strip_reasoning_from_task_outputs(
                paths,
                logger=self.logger,
                only_paths=emitted_paths,
            )
        return emitted_paths

    def _run_llm_stages(
        self,
        *,
        paths: ScenePaths,
        ctx: SceneContext,
        cfg: DaftExportConfig,
        inputs: StageInputs,
        resolver: EndpointResolver,
        config_dir: Path,
        reasoning_provider: ReasoningProvider | None,
    ) -> tuple[StageInputs, list[Path]]:
        extras = {"reasoning_provider": reasoning_provider}
        prompt_cache: PromptRegistryCache = {}
        emitted_paths: list[Path] = []

        # Anomaly mode runs first: it only needs the captioning windows,
        # and its verdict feeds the causal-linkage ``video_type`` tagging
        # below. The verdict is a non-DAFT sidecar, so it bypasses the
        # ``emit_stage`` (DAFT-only) scaffold and uses its own runner.
        if cfg.anomaly is not None and cfg.anomaly.enabled:
            verdict = run_anomaly_stage(
                paths=paths,
                ctx=ctx,
                cfg=cfg.anomaly,
                inputs=inputs,
                resolver=resolver,
                config_dir=config_dir,
                logger=self.logger,
                prompt_cache=prompt_cache,
            )
            # The runner returns the verdict on success, so ingest it
            # in-memory and skip the redundant sidecar re-read. Fall back
            # to disk only when this run produced no verdict (e.g. a stale
            # sidecar from a prior run may still be on disk).
            if verdict is not None:
                inputs = inputs.with_anomaly_verdict(verdict)
            else:
                inputs = inputs.with_refreshed_anomaly(paths, logger=self.logger)
            if paths.sidecar_anomaly.exists():
                emitted_paths.append(paths.sidecar_anomaly)

        if cfg.events is not None and cfg.events.enabled:
            emitted = emit_stage(
                EventsEmitter(),
                paths=paths,
                ctx=ctx,
                cfg=cfg.events,
                inputs=inputs,
                resolver=resolver,
                config_dir=config_dir,
                logger=self.logger,
                extras=extras,
                prompt_cache=prompt_cache,
            )
            if emitted is not None:
                emitted_paths.append(emitted)
            inputs = inputs.with_refreshed_events(paths, logger=self.logger)

        # MSTED runs next and is refreshed so the downstream reasoning
        # stages (QA family, causal linkage) can ground in the freshly
        # synthesized structured description (the description-to-QA flow),
        # exactly as events.json feeds auto-derived causal pairs above.
        if cfg.msted is not None and cfg.msted.enabled:
            emitted = emit_stage(
                MstedEmitter(),
                paths=paths,
                ctx=ctx,
                cfg=cfg.msted,
                inputs=inputs,
                resolver=resolver,
                config_dir=config_dir,
                logger=self.logger,
                extras=extras,
                prompt_cache=prompt_cache,
            )
            if emitted is not None:
                emitted_paths.append(emitted)
            inputs = inputs.with_refreshed_msted(paths, logger=self.logger)

        emitter_configs = (
            (TemporalLocalizationEmitter(), cfg.temporal_localization),
            (OpenQaEmitter(), cfg.open_qa),
            (McqOpenendedEmitter(), cfg.mcq_openended),
            (BcqOpenendedEmitter(), cfg.bcq_openended),
            (CausalLinkageEmitter(), cfg.causal_linkage),
        )
        for emitter, stage_cfg in emitter_configs:
            if stage_cfg is None or not stage_cfg.enabled:
                continue
            emitted = emit_stage(
                emitter,
                paths=paths,
                ctx=ctx,
                cfg=stage_cfg,
                inputs=inputs,
                resolver=resolver,
                config_dir=config_dir,
                logger=self.logger,
                extras=extras,
                prompt_cache=prompt_cache,
            )
            if emitted is not None:
                emitted_paths.append(emitted)

        return inputs, emitted_paths


def _should_strip_reasoning(*, cfg: DaftExportConfig | None, mode: ReasoningMode) -> bool:
    if mode == "strip":
        return True
    if mode == "keep":
        return False
    return cfg is not None and (cfg.reasoning is None or not cfg.reasoning.enabled)


def _record_pipeline_state(
    paths: ScenePaths,
    entry: DataEntry,
    *,
    cfg: DaftExportConfig | None,
    emitted_paths: set[Path],
) -> None:
    emitted = {path.resolve() for path in emitted_paths if path.exists()}
    update_annotation_export_state(
        paths.scene_dir,
        data_entry_id=entry.id,
        media_path=entry.media_path,
        emitter_artifacts={
            name: path if path.resolve() in emitted else None
            for name, path in _owned_outputs(paths, cfg=cfg).items()
        },
    )


def _owned_outputs(paths: ScenePaths, *, cfg: DaftExportConfig | None) -> dict[str, Path]:
    outputs = {
        "events": paths.contextual_events,
        "msted": paths.contextual_msted,
        "temporal_localization": paths.task_temporal_localization,
        "mcq_openended": paths.task_mcq_openended,
        "bcq_openended": paths.task_bcq_openended,
        "causal_linkage": paths.task_causal_linkage,
    }
    if cfg is not None and cfg.open_qa is not None and cfg.open_qa.enabled:
        outputs["open_qa"] = paths.task_open_qa
    if cfg is not None and cfg.anomaly is not None and cfg.anomaly.enabled:
        outputs["anomaly"] = paths.sidecar_anomaly
    return outputs


__all__ = ["ReasoningMode", "ReasoningTask", "to_daft_tasks"]
