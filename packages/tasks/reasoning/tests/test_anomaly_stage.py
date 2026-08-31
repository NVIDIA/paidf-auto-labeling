# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the anomaly sidecar stage runner and the StageInputs hook.

The LLM adapter is mocked at the stage-module boundary; these tests
verify the runner's filesystem contract (writes the sidecar, returns the
verdict), its best-effort skips, and the ``anomaly_video_type`` accessor
that feeds causal-linkage auto-tagging."""

from __future__ import annotations

import json
import logging

from core.scene import SceneContext
from reasoning.config import AnomalyConfig
from reasoning.paths import ensure_scene_skeleton, scene_paths
from reasoning.stage_inputs import StageInputs
from reasoning.stages import anomaly as anomaly_stage
from reasoning.stages.anomaly import run_anomaly_stage

_LOGGER = logging.getLogger("test.anomaly_stage")


class FakeResolver:
    def resolve_llm(self) -> tuple[str, str]:
        return "http://llm/v1", "model"

    def resolve_llm_api_key(self) -> str | None:
        return None


class FakePromptRegistry:
    def get(self, name: str):
        return object()


def _inputs_with_windows() -> StageInputs:
    return StageInputs(
        sidecar_metadata={"windows": [{"start_s": 0.0, "end_s": 4.0, "caption": "c"}]},
        sidecar_metadata_chunk=None,
        scene_video=None,
        scene_image=None,
        scene_events=None,
        scene_instances=None,
    )


def _run(paths, ctx, monkeypatch, *, verdict):
    monkeypatch.setattr(anomaly_stage, "build_prompt_registry", lambda **_: FakePromptRegistry())
    monkeypatch.setattr(anomaly_stage, "classify_anomaly_with_llm", lambda **_: verdict)
    return run_anomaly_stage(
        paths=paths,
        ctx=ctx,
        cfg=AnomalyConfig(enabled=True),
        inputs=_inputs_with_windows(),
        resolver=FakeResolver(),
        config_dir=paths.scene_dir,
        logger=_LOGGER,
    )


class TestRunAnomalyStage:
    def test_writes_sidecar_and_returns_verdict(self, tmp_path, monkeypatch):
        paths = ensure_scene_skeleton(tmp_path)
        verdict = {
            "classification": "anomaly",
            "confidence": "high",
            "reasoning": "collision",
            "highlight": {"start": "00:04", "end": "00:07", "description": "impact"},
        }
        out = _run(paths, SceneContext(media_id="clip"), monkeypatch, verdict=verdict)
        assert out == verdict
        assert paths.sidecar_anomaly.exists()
        payload = json.loads(paths.sidecar_anomaly.read_text(encoding="utf-8"))
        assert payload["classification"] == "anomaly"
        assert payload["reperception_request"]["target"] == "captioning"
        assert payload["model"] == "model"

    def test_image_scene_skipped(self, tmp_path, monkeypatch):
        paths = ensure_scene_skeleton(tmp_path)
        out = _run(
            paths,
            SceneContext(media_id="img", is_image=True),
            monkeypatch,
            verdict={"classification": "anomaly", "reasoning": "x"},
        )
        assert out is None
        assert not paths.sidecar_anomaly.exists()

    def test_no_windows_skipped(self, tmp_path, monkeypatch):
        paths = ensure_scene_skeleton(tmp_path)
        monkeypatch.setattr(
            anomaly_stage, "build_prompt_registry", lambda **_: FakePromptRegistry()
        )
        monkeypatch.setattr(
            anomaly_stage, "classify_anomaly_with_llm", lambda **_: {"classification": "anomaly"}
        )
        empty = StageInputs(
            sidecar_metadata=None,
            sidecar_metadata_chunk=None,
            scene_video=None,
            scene_image=None,
            scene_events=None,
            scene_instances=None,
        )
        out = run_anomaly_stage(
            paths=paths,
            ctx=SceneContext(media_id="clip"),
            cfg=AnomalyConfig(enabled=True),
            inputs=empty,
            resolver=FakeResolver(),
            config_dir=paths.scene_dir,
            logger=_LOGGER,
        )
        assert out is None
        assert not paths.sidecar_anomaly.exists()


class TestAnomalyVideoType:
    def _inputs(self, anomaly: dict | None) -> StageInputs:
        return StageInputs(
            sidecar_metadata=None,
            sidecar_metadata_chunk=None,
            scene_video=None,
            scene_image=None,
            scene_events=None,
            scene_instances=None,
            scene_anomaly=anomaly,
        )

    def test_returns_anomaly_label(self):
        assert self._inputs({"classification": "Anomaly"}).anomaly_video_type() == "anomaly"

    def test_returns_normal_label(self):
        assert self._inputs({"classification": "normal"}).anomaly_video_type() == "normal"

    def test_returns_none_for_missing(self):
        assert self._inputs(None).anomaly_video_type() is None

    def test_returns_none_for_invalid(self):
        assert self._inputs({"classification": "weird"}).anomaly_video_type() is None

    def test_with_anomaly_verdict_ingests_in_memory(self):
        base = self._inputs(None)
        updated = base.with_anomaly_verdict({"classification": "anomaly", "reasoning": "x"})
        # Original is untouched (frozen dataclass) and the copy carries the verdict.
        assert base.anomaly_video_type() is None
        assert updated.anomaly_video_type() == "anomaly"

    def test_with_anomaly_verdict_ignores_non_dict(self):
        base = self._inputs(None)
        assert base.with_anomaly_verdict(None) is base


class TestStageInputsLoadsAnomaly:
    def test_load_reads_anomaly_sidecar(self, tmp_path):
        paths = scene_paths(tmp_path)
        paths.sidecar_anomaly.parent.mkdir(parents=True, exist_ok=True)
        paths.sidecar_anomaly.write_text(
            json.dumps({"classification": "anomaly"}), encoding="utf-8"
        )
        loaded = StageInputs.load(paths, logger=_LOGGER)
        assert loaded.anomaly_video_type() == "anomaly"


def _pas_payload() -> dict:
    return {
        "chunk_id": "clip",
        "pas": {
            "n_people": 1,
            "people": [
                {
                    "track_id": 0,
                    "attributes": {
                        "motion or action": "falling",
                        "potential anomaly": "yes",
                        "anomaly type": "fall",
                        "natural language caption": "A person falls to the ground.",
                    },
                    "natural_language_caption": "A person falls to the ground.",
                }
            ],
        },
    }


def _inputs_with_windows_and_pas(pas: dict | None) -> StageInputs:
    return StageInputs(
        sidecar_metadata={"windows": [{"start_s": 0.0, "end_s": 4.0, "caption": "c"}]},
        sidecar_metadata_chunk=None,
        scene_video=None,
        scene_image=None,
        scene_events=None,
        scene_instances=None,
        scene_pas=pas,
    )


class TestAnomalyPersonAttributes:
    """The anomaly stage folds PAS evidence into the LLM call when present."""

    def _capture_run(self, paths, monkeypatch, *, inputs, cfg):
        captured: dict = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return {"classification": "normal", "reasoning": "x"}

        monkeypatch.setattr(
            anomaly_stage, "build_prompt_registry", lambda **_: FakePromptRegistry()
        )
        monkeypatch.setattr(anomaly_stage, "classify_anomaly_with_llm", stub)
        run_anomaly_stage(
            paths=paths,
            ctx=SceneContext(media_id="clip"),
            cfg=cfg,
            inputs=inputs,
            resolver=FakeResolver(),
            config_dir=paths.scene_dir,
            logger=_LOGGER,
        )
        return captured

    def test_pas_evidence_passed_when_present(self, tmp_path, monkeypatch):
        paths = ensure_scene_skeleton(tmp_path)
        captured = self._capture_run(
            paths,
            monkeypatch,
            inputs=_inputs_with_windows_and_pas(_pas_payload()),
            cfg=AnomalyConfig(enabled=True),
        )
        pa = captured["person_attributes"]
        assert pa is not None
        assert "Person track 0" in pa
        assert "potential anomaly=yes" in pa

    def test_pas_evidence_suppressed_by_toggle(self, tmp_path, monkeypatch):
        paths = ensure_scene_skeleton(tmp_path)
        captured = self._capture_run(
            paths,
            monkeypatch,
            inputs=_inputs_with_windows_and_pas(_pas_payload()),
            cfg=AnomalyConfig(enabled=True, include_person_attributes=False),
        )
        assert captured["person_attributes"] is None

    def test_no_pas_passes_none(self, tmp_path, monkeypatch):
        paths = ensure_scene_skeleton(tmp_path)
        captured = self._capture_run(
            paths,
            monkeypatch,
            inputs=_inputs_with_windows_and_pas(None),
            cfg=AnomalyConfig(enabled=True),
        )
        assert captured["person_attributes"] is None


class TestPersonAttributesBlock:
    def test_formats_people_lines(self):
        block = _inputs_with_windows_and_pas(_pas_payload()).person_attributes_block()
        assert block is not None
        assert block.startswith("Person track 0:")
        assert "motion or action=falling" in block
        assert '"A person falls to the ground."' in block

    def test_none_when_no_people(self):
        inputs = _inputs_with_windows_and_pas({"pas": {"people": []}})
        assert inputs.person_attributes_block() is None

    def test_none_when_no_pas(self):
        assert _inputs_with_windows_and_pas(None).person_attributes_block() is None

    def test_load_reads_pas_sidecar(self, tmp_path):
        paths = scene_paths(tmp_path)
        paths.sidecar_pas.parent.mkdir(parents=True, exist_ok=True)
        paths.sidecar_pas.write_text(json.dumps(_pas_payload()), encoding="utf-8")
        loaded = StageInputs.load(paths, logger=_LOGGER)
        block = loaded.person_attributes_block()
        assert block is not None and "Person track 0" in block
