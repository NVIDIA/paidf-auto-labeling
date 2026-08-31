# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from core import DataEntry
from reasoning.common import DAFT_VERSION, SceneContext
from reasoning.endpoint_resolver import EndpointResolver, _endpoint_value
from reasoning.paths import ScenePaths, ensure_scene_skeleton, resolve_raw_media, scene_paths
from reasoning.stages import base
from reasoning.stages.base import StageInputs, emit_stage
from reasoning.task import ReasoningTask


class FakeResolver:
    def resolve_llm(self) -> tuple[str, str]:
        return "http://llm/v1", "model"


class FakeEmitter:
    name = "fake-stage"
    expected_type = "fake"
    requires_temporal_axis = False

    def output(self, paths: ScenePaths) -> Path:
        return paths.task_dir / "fake.json"

    def render(
        self,
        *,
        inputs: StageInputs,
        ctx: SceneContext,
        cfg: Any,
        prompts: Any,
        llm: tuple[str, str],
        config_dir: Path,
        logger: logging.Logger,
        extras: dict | None = None,
    ) -> dict[str, Any]:
        return {"version": "3.0.0", "metadata": {"type": "fake"}}


class TestScenePaths:
    def test_expected_subpaths(self, tmp_path):
        paths = scene_paths(tmp_path)
        assert paths.scene_dir == tmp_path
        assert paths.raw_dir == tmp_path / "raw"
        assert paths.contextual_dir == tmp_path / "contextual"
        assert paths.contextual_video == tmp_path / "contextual" / "video.json"
        assert paths.contextual_events == tmp_path / "contextual" / "events.json"
        assert paths.contextual_instances == tmp_path / "contextual" / "instances.json"
        assert paths.contextual_objects == tmp_path / "contextual" / "objects.json"
        assert paths.task_dir == tmp_path / "task"
        assert paths.task_mcq == tmp_path / "task" / "mcq.json"
        assert paths.task_bcq == tmp_path / "task" / "bcq.json"
        assert paths.task_open_qa == tmp_path / "task" / "open_qa.json"
        assert paths.sidecars_dir == tmp_path / "sidecars"

    def test_no_filesystem_side_effects(self, tmp_path):
        paths = scene_paths(tmp_path / "new_scene")
        assert not paths.raw_dir.exists()
        assert not paths.contextual_dir.exists()
        assert not paths.task_dir.exists()
        assert not paths.sidecars_dir.exists()


class TestEnsureSceneSkeleton:
    def test_creates_all_subdirs(self, tmp_path):
        paths = ensure_scene_skeleton(tmp_path / "scene1")
        assert paths.raw_dir.is_dir()
        assert paths.contextual_dir.is_dir()
        assert paths.task_dir.is_dir()
        assert paths.sidecars_dir.is_dir()

    def test_idempotent(self, tmp_path):
        scene = tmp_path / "scene1"
        ensure_scene_skeleton(scene)
        (scene / "sidecars" / "note.txt").write_text("keep me")
        ensure_scene_skeleton(scene)
        assert (scene / "sidecars" / "note.txt").read_text() == "keep me"


class TestResolveRawMedia:
    def test_returns_none_when_raw_dir_missing(self, tmp_path):
        assert resolve_raw_media(tmp_path / "no-scene") is None

    def test_returns_none_when_empty(self, tmp_path):
        ensure_scene_skeleton(tmp_path)
        assert resolve_raw_media(tmp_path) is None

    def test_resolves_mp4(self, tmp_path):
        paths = ensure_scene_skeleton(tmp_path)
        (paths.raw_dir / "clip.mp4").write_bytes(b"\x00")
        assert resolve_raw_media(tmp_path) == paths.raw_dir / "clip.mp4"

    def test_resolves_non_mp4_suffix(self, tmp_path):
        paths = ensure_scene_skeleton(tmp_path)
        (paths.raw_dir / "dashcam.mov").write_bytes(b"\x00")
        assert resolve_raw_media(tmp_path) == paths.raw_dir / "dashcam.mov"

    def test_ambiguous_raw_media_raises(self, tmp_path):
        paths = ensure_scene_skeleton(tmp_path)
        (paths.raw_dir / "a.mp4").write_bytes(b"\x00")
        (paths.raw_dir / "b.mp4").write_bytes(b"\x00")
        with pytest.raises(ValueError, match="ambiguous raw media"):
            resolve_raw_media(tmp_path)


def test_endpoint_value_preserves_falsy_dict_values():
    endpoints = {"llm": {"retries": 0, "retry_backoff_s": 0.0}}

    assert _endpoint_value(endpoints, "llm", "retries", 3) == 0
    assert _endpoint_value(endpoints, "llm", "retry_backoff_s", 5.0) == 0.0


class TestReasoningTaskStagesRawMedia:
    """The reasoning task must make the scene self-contained by staging
    ``raw/<media_id>.<ext>`` so a downstream training_export stage can resolve
    the analyzed media even in a reasoning-only output directory (no detection
    stage to populate ``raw/``)."""

    def _entry(self, tmp_path):
        scene = tmp_path / "clip.mp4"
        paths = ensure_scene_skeleton(scene)
        media = paths.sidecars_dir / "active.mp4"
        media.write_bytes(b"video-bytes")
        return DataEntry(id="e1", media_path=str(media), data_path=str(scene)), scene

    def test_run_stages_single_extension_symlink(self, tmp_path):
        entry, scene = self._entry(tmp_path)
        ReasoningTask(config=None, reasoning_mode="keep").run(entry)

        staged = resolve_raw_media(scene)
        assert staged == scene / "raw" / "clip.mp4"
        assert staged.is_symlink()
        assert staged.read_bytes() == b"video-bytes"


def test_endpoint_value_defaults_only_when_missing_or_none():
    endpoints = SimpleNamespace(vlm=SimpleNamespace(url="", model=None))

    assert _endpoint_value(endpoints, "vlm", "url", "fallback") == ""
    assert _endpoint_value(endpoints, "vlm", "model", "fallback") == "fallback"
    assert _endpoint_value(endpoints, "missing", "url", "fallback") == "fallback"


def test_endpoint_resolver_prefers_endpoint_url_env_alias(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://old/v1")
    monkeypatch.setenv("LLM_ENDPOINT_URL", "http://new/v1")
    monkeypatch.setenv("LLM_MODEL", "model")

    resolver = EndpointResolver(None, logger=logging.getLogger("test.endpoint_resolver"))

    assert resolver.resolve_llm() == ("http://new/v1", "model")


def test_endpoint_resolver_reads_nvidia_api_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    resolver = EndpointResolver(
        None,
        logger=logging.getLogger("test.endpoint_resolver"),
    )

    assert resolver.resolve_llm_api_key() == "secret"


def test_endpoint_resolver_returns_empty_when_nvidia_api_key_unset(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    resolver = EndpointResolver(
        None,
        logger=logging.getLogger("test.endpoint_resolver"),
    )

    assert resolver.resolve_llm_api_key() == "EMPTY"


def test_emit_stage_logs_and_skips_write_failure(tmp_path, monkeypatch, caplog):
    paths = ensure_scene_skeleton(tmp_path)
    inputs = StageInputs(
        sidecar_metadata=None,
        sidecar_metadata_chunk=None,
        scene_video=None,
        scene_image=None,
        scene_events=None,
        scene_instances=None,
    )

    monkeypatch.setattr(base, "build_prompt_registry", lambda **_: object())

    def fail_write(path, payload, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(base, "write_daft_json", fail_write)

    logger = logging.getLogger("test.emit_stage")
    logger.handlers.clear()
    logger.propagate = True
    with caplog.at_level(logging.WARNING, logger=logger.name):
        emit_stage(
            FakeEmitter(),
            paths=paths,
            ctx=SceneContext(media_id="clip"),
            cfg=SimpleNamespace(prompt_dir=None),
            inputs=inputs,
            resolver=FakeResolver(),
            config_dir=tmp_path,
            logger=logger,
        )

    assert "fake-stage" in caplog.text
    assert "failed to write" in caplog.text
    assert "disk full" in caplog.text


def test_emit_stage_rejects_wrong_payload_type(tmp_path, monkeypatch, caplog):
    paths = ensure_scene_skeleton(tmp_path)
    inputs = StageInputs(
        sidecar_metadata=None,
        sidecar_metadata_chunk=None,
        scene_video=None,
        scene_image=None,
        scene_events=None,
        scene_instances=None,
    )

    class WrongTypeEmitter(FakeEmitter):
        name = "wrong-type"
        expected_type = "open_qa"

        def output(self, paths: ScenePaths) -> Path:
            return paths.task_open_qa

        def render(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "version": DAFT_VERSION,
                "metadata": {"type": "mcq"},
                "items": [{"question": "Q?", "options": {"A": "Yes"}, "answer": "A"}],
            }

    monkeypatch.setattr(base, "build_prompt_registry", lambda **_: object())
    logger = logging.getLogger("test.emit_stage_type")
    logger.handlers.clear()
    logger.propagate = True

    with caplog.at_level(logging.WARNING, logger=logger.name):
        emitted = emit_stage(
            WrongTypeEmitter(),
            paths=paths,
            ctx=SceneContext(media_id="clip"),
            cfg=SimpleNamespace(prompt_dir=None),
            inputs=inputs,
            resolver=FakeResolver(),
            config_dir=tmp_path,
            logger=logger,
        )

    assert emitted is None
    assert not paths.task_open_qa.exists()
    assert "wrong-type" in caplog.text
    assert "metadata.type" in caplog.text


def test_build_prompt_registry_reuses_cache(tmp_path, monkeypatch):
    calls = []

    class FakePromptRegistry:
        def __init__(self, *, extra_dirs):
            calls.append(list(extra_dirs))

    monkeypatch.setattr(base, "PromptRegistry", FakePromptRegistry)

    cache = {}
    first = base.build_prompt_registry(
        prompt_dir=None,
        config_dir=tmp_path,
        logger=logging.getLogger("test.prompt-cache"),
        tag="stage-a",
        cache=cache,
    )
    second = base.build_prompt_registry(
        prompt_dir=None,
        config_dir=tmp_path,
        logger=logging.getLogger("test.prompt-cache"),
        tag="stage-b",
        cache=cache,
    )

    assert first is second
    assert len(calls) == 1
