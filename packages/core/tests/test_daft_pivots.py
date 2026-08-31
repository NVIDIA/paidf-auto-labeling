# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, cast

import pytest
from core import SceneContext
from core.formats.daft import (
    DaftConvertError,
    emit_captioning_daft_outputs,
    emit_persona_qa_daft_outputs,
    emit_visual_qa_daft_outputs,
)
from core.formats.daft.converters import to_daft_tasks


def test_emit_captioning_daft_outputs_writes_captioning_pivots(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecars" / "captioning" / "metadata_chunk.json"
    _write_json(
        sidecar,
        {
            "duration_span": [0.0, 1.0],
            "summary": "A car passes through the scene.",
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "description": "A car drives through an intersection.",
                    "parsed": {
                        "scene_description": "A car is in an urban intersection.",
                        "event_summary": "A car passes through the scene.",
                    },
                }
            ],
        },
    )

    written = emit_captioning_daft_outputs(
        tmp_path,
        SceneContext(media_id="clip"),
        caption_artifacts={"success": True, "metadata_chunk_json": str(sidecar)},
    )

    assert {path.name for path in written} == {
        "chunks.json",
        "scene_description.json",
        "temporal_description.json",
        "video_summarization.json",
    }
    scene_description = json.loads((tmp_path / "task" / "scene_description.json").read_text())
    assert scene_description["items"][0]["answer"] == "A car is in an urban intersection."


def test_emit_visual_qa_daft_outputs_writes_task_pivots(tmp_path: Path) -> None:
    items_json = tmp_path / "sidecars" / "visual_qa" / "items.json"
    _write_json(
        items_json,
        {
            "schema_version": "1",
            "media_id": "clip",
            "items": [
                {
                    "id": "q1",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "answer": "Yes",
                },
                {
                    "id": "q2",
                    "question": "Describe the incident.",
                    "answer": "A cyclist falls near a stopped car.",
                },
            ],
        },
    )

    written = emit_visual_qa_daft_outputs(
        tmp_path,
        SceneContext(media_id="clip"),
        visual_qa_artifacts={"success": True, "items_json": str(items_json)},
    )

    assert {path.name for path in written} == {"bcq.json", "open_qa.json"}


def _person_window(track_id: int, *, window_index: int) -> dict[str, object]:
    return {
        "window_index": window_index,
        "track_id": track_id,
        "n_crops_in_chunk": 12,
        "items": [
            {
                "id": "gender",
                "question": "Gender?",
                "options": ["male", "female"],
                "answer": "male",
            },
            {"id": "mask", "question": "Mask?", "options": ["Yes", "No"], "answer": "No"},
            {"id": "caption", "question": "Describe the person.", "answer": "Walking."},
        ],
    }


def test_emit_persona_qa_groups_per_track(tmp_path: Path) -> None:
    windows_json = tmp_path / "sidecars" / "visual_qa" / "windows.normalized.json"
    _write_json(
        windows_json,
        {
            "schema_version": "1",
            "media_id": "clip",
            "windows": [
                _person_window(1, window_index=1),
                _person_window(0, window_index=0),
                # Whole-clip window (no track) must be ignored.
                {"window_index": 2, "track_id": None, "items": []},
            ],
        },
    )

    written = emit_persona_qa_daft_outputs(
        tmp_path,
        SceneContext(media_id="clip"),
        visual_qa_artifacts={"success": True, "windows_json": str(windows_json)},
    )

    assert written == tmp_path / "sidecars" / "visual_qa" / "persona_qa.json"
    payload = json.loads(written.read_text())
    assert payload["type"] == "persona_qa"
    assert payload["n_personas"] == 2
    # Personas are sorted deterministically by track_id.
    assert [p["track_id"] for p in payload["personas"]] == [0, 1]
    persona = payload["personas"][0]
    # gender -> 1 MCQ; mask -> 1 BCQ; caption -> 1 open_qa.
    assert len(persona["mcq"]) == 1
    assert len(persona["bcq"]) == 1
    assert len(persona["open_qa"]) == 1


def test_emit_persona_qa_skips_when_no_per_track_windows(tmp_path: Path) -> None:
    windows_json = tmp_path / "sidecars" / "visual_qa" / "windows.normalized.json"
    _write_json(
        windows_json,
        {
            "schema_version": "1",
            "media_id": "clip",
            "windows": [{"window_index": 0, "track_id": None, "items": []}],
        },
    )

    written = emit_persona_qa_daft_outputs(
        tmp_path,
        SceneContext(media_id="clip"),
        visual_qa_artifacts={"success": True, "windows_json": str(windows_json)},
    )

    assert written is None
    assert not (tmp_path / "sidecars" / "visual_qa" / "persona_qa.json").exists()


def test_to_daft_tasks_rejects_non_dict_items() -> None:
    items = cast("list[dict[str, Any]]", ["not-a-task"])

    with pytest.raises(DaftConvertError, match="is not a dict/mapping"):
        to_daft_tasks(items, ctx=SceneContext(media_id="clip"))


def test_emit_captioning_daft_outputs_rejects_media_id_mismatch(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecars" / "captioning" / "metadata_chunk.json"
    _write_json(
        sidecar,
        {
            "video_id": "other",
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "description": "A car drives through an intersection.",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="does not match expected"):
        emit_captioning_daft_outputs(
            tmp_path,
            SceneContext(media_id="clip"),
            caption_artifacts={"success": True, "metadata_chunk_json": str(sidecar)},
        )


def test_emit_visual_qa_daft_outputs_rejects_media_id_mismatch(tmp_path: Path) -> None:
    items_json = tmp_path / "sidecars" / "visual_qa" / "items.json"
    _write_json(
        items_json,
        {
            "schema_version": "1",
            "media_id": "other",
            "items": [
                {
                    "id": "q1",
                    "question": "Is there an incident?",
                    "options": ["Yes", "No"],
                    "answer": "Yes",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="does not match expected"):
        emit_visual_qa_daft_outputs(
            tmp_path,
            SceneContext(media_id="clip"),
            visual_qa_artifacts={"success": True, "items_json": str(items_json)},
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
