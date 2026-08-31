# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PAS output/state helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from core import DataEntry, SceneContext, read_pipeline_state
from person_attribute_search.artifacts import PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY
from person_attribute_search.outputs import (
    emit_anomaly_deliverable,
    emit_daft_contextual,
    record_state,
)


def _entry(scene_dir: Path) -> DataEntry:
    return DataEntry(id="entry-1", media_path="chunk_000.mp4", data_path=str(scene_dir))


def test_record_state_persists_optional_failures(tmp_path: Path) -> None:
    attributes = tmp_path / "sidecars" / "person_attribute_search" / "attributes.json"
    queries = tmp_path / "sidecars" / "person_attribute_search" / "queries.json"

    record_state(
        _entry(tmp_path),
        success=True,
        attributes_json=attributes,
        queries_json=queries,
        n_people=2,
        warnings=["low confidence"],
        optional_failures=["bucket_query_generation_failed: timeout"],
    )

    state = read_pipeline_state(tmp_path)
    artifacts = state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY]
    assert artifacts["success"] is True
    assert artifacts["attributes_json"] == str(attributes)
    assert artifacts["queries_json"] == str(queries)
    assert artifacts["n_people"] == 2
    assert artifacts["warnings"] == ["low confidence"]
    assert artifacts["optional_failures"] == ["bucket_query_generation_failed: timeout"]


def test_emit_daft_contextual_writes_both_contextual_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Writer:
        def __init__(self, scene_dir: Path, *, ctx: SceneContext) -> None:
            assert scene_dir == tmp_path
            assert ctx.media_id == "chunk_000"

        def write_contextual(self, name: str, payload: dict[str, Any]) -> None:
            calls.append((name, payload))

    monkeypatch.setattr("person_attribute_search.outputs.DaftSceneWriter", _Writer)

    failure = emit_daft_contextual(
        data_entry=_entry(tmp_path),
        ctx=SceneContext(media_id="chunk_000", is_image=False),
        pas_document={"pas": {"n_people": 1}},
        queries_document={"queries": [{"query": "person in vest"}]},
        logger=logging.getLogger(__name__),
    )

    assert failure is None
    assert [name for name, _payload in calls] == ["person_attributes", "pas_queries"]
    assert calls[0][1]["pas"] == {"n_people": 1}
    assert calls[1][1]["queries"] == [{"query": "person in vest"}]


def test_emit_daft_contextual_returns_structured_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Writer:
        def __init__(self, _scene_dir: Path, *, ctx: SceneContext) -> None:
            assert ctx.media_id == "chunk_000"

        def write_contextual(self, _name: str, _payload: dict[str, Any]) -> None:
            raise ValueError("writer unavailable")

    monkeypatch.setattr("person_attribute_search.outputs.DaftSceneWriter", _Writer)

    failure = emit_daft_contextual(
        data_entry=_entry(tmp_path),
        ctx=SceneContext(media_id="chunk_000", is_image=False),
        pas_document={},
        queries_document={},
        logger=logging.getLogger(__name__),
    )

    assert failure == "daft_contextual_mirror_failed: writer unavailable"


def test_emit_anomaly_deliverable_returns_none_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("person_attribute_search.outputs.convert_scene", lambda scene_dir: True)

    failure = emit_anomaly_deliverable(
        data_entry=_entry(tmp_path),
        logger=logging.getLogger(__name__),
    )

    assert failure is None


def test_emit_anomaly_deliverable_returns_skip_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("person_attribute_search.outputs.convert_scene", lambda scene_dir: False)

    failure = emit_anomaly_deliverable(
        data_entry=_entry(tmp_path),
        logger=logging.getLogger(__name__),
    )

    assert failure == "anomaly_deliverable_skipped: no pas.json after PAS write"


def test_emit_anomaly_deliverable_returns_structured_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_scene_dir: Path) -> bool:
        raise ValueError("adapter failed")

    monkeypatch.setattr("person_attribute_search.outputs.convert_scene", _raise)

    failure = emit_anomaly_deliverable(
        data_entry=_entry(tmp_path),
        logger=logging.getLogger(__name__),
    )

    assert failure == "anomaly_deliverable_failed: adapter failed"
