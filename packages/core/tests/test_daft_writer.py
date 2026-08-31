# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from core.formats.daft import (
    DAFT_VERSION,
    CompositeDaftValidator,
    ContextualType,
    DaftKind,
    DaftPayloadValidator,
    DaftSceneWriter,
    DaftValidationError,
    DaftWriteError,
    JsonSchemaDaftPayloadValidator,
    StructuralDaftValidator,
    TaskType,
    daft_envelope,
    write_daft_json,
)
from core.scene import SceneContext


class RejectingValidator:
    def validate_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_type: str | None = None,
        kind: DaftKind | None = None,
        path: Path | None = None,
    ) -> list[str]:
        _ = payload, expected_type, kind, path
        return ["rejected"]


def _video_payload(ctx: SceneContext) -> dict[str, Any]:
    payload = daft_envelope("video", ctx)
    payload.update(
        {
            "format": "mp4",
            "fps": 30,
            "duration": 1.5,
            "height": 720,
            "width": 1280,
        }
    )
    return payload


def test_daft_envelope_includes_version_and_scene_id() -> None:
    ctx = SceneContext(
        media_id="clip",
        iso_date="2026-05-11",
        license_str="proprietary",
        tags=("traffic",),
    )
    env = daft_envelope("events", ctx, description="test")
    assert env["version"] == DAFT_VERSION
    assert env["video_id"] == "clip"
    assert env["metadata"]["type"] == "events"
    assert env["metadata"]["date"] == "2026-05-11"
    assert env["metadata"]["description"] == "test"
    assert env["metadata"]["license"] == "proprietary"
    assert env["metadata"]["tags"] == ["traffic"]


def test_daft_envelope_can_omit_scene_id() -> None:
    ctx = SceneContext(media_id="clip")
    env = daft_envelope("instances", ctx, include_scene_id=False)
    assert "video_id" not in env and "image_id" not in env


def test_scene_writer_routes_contextual_payloads(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip", iso_date="2026-05-20")
    writer = DaftSceneWriter(tmp_path / "scene", ctx=ctx)

    result = writer.write_payload(_video_payload(ctx))

    assert result.kind == "contextual"
    assert result.type_name == "video"
    assert result.path == tmp_path / "scene" / "contextual" / "video.json"
    assert result.path.exists()


def test_scene_writer_routes_person_attributes_contextual(tmp_path: Path) -> None:
    # person_attributes is a registered contextual type; the writer must route it
    # to contextual/person_attributes.json and the default structural validator
    # must accept the open attribute bag without a closed enum.
    ctx = SceneContext(media_id="clip", iso_date="2026-05-20")
    writer = DaftSceneWriter(tmp_path / "scene", ctx=ctx)
    payload = daft_envelope("person_attributes", ctx)
    payload["chunk_id"] = "chunk_000"
    payload["pas"] = {
        "n_people": 1,
        "people": [{"track_id": 0, "attributes": {"top_outer_color": "maroon red"}}],
    }
    payload["crop_root"] = "crops"

    result = writer.write_contextual("person_attributes", payload)

    assert result.kind == "contextual"
    assert result.type_name == "person_attributes"
    assert result.path == tmp_path / "scene" / "contextual" / "person_attributes.json"
    assert result.path.exists()


def test_scene_writer_routes_pas_queries_contextual(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip", iso_date="2026-05-20")
    writer = DaftSceneWriter(tmp_path / "scene", ctx=ctx)
    payload = daft_envelope("pas_queries", ctx)
    payload["chunk_id"] = "chunk_000"
    payload["queries"] = [{"query": "person in a red jacket"}]

    result = writer.write_contextual("pas_queries", payload)

    assert result.kind == "contextual"
    assert result.type_name == "pas_queries"
    assert result.path == tmp_path / "scene" / "contextual" / "pas_queries.json"
    assert result.path.exists()


def test_write_contextual_rejects_task_type_before_writing(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip")
    writer = DaftSceneWriter(tmp_path / "scene", ctx=ctx)
    payload = daft_envelope("mcq", ctx)
    payload["items"] = [{"question": "q", "answer": "A", "options": {"A": "yes", "B": "no"}}]

    with pytest.raises(DaftWriteError, match="'mcq' is not a contextual DAFT type"):
        writer.write_contextual(cast(ContextualType, "mcq"), payload)

    assert not (tmp_path / "scene" / "task" / "mcq.json").exists()


def test_write_task_rejects_contextual_type_before_writing(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip", iso_date="2026-05-20")
    writer = DaftSceneWriter(tmp_path / "scene", ctx=ctx)

    with pytest.raises(DaftWriteError, match="'video' is not a task DAFT type"):
        writer.write_task(cast(TaskType, "video"), _video_payload(ctx))

    assert not (tmp_path / "scene" / "contextual" / "video.json").exists()


def test_write_daft_json_rejects_wrong_version_before_writing(tmp_path: Path) -> None:
    payload = {
        "version": "3.0",
        "metadata": {"type": "mcq"},
        "items": [{"question": "q", "answer": "A"}],
    }
    path = tmp_path / "mcq.json"

    with pytest.raises(DaftValidationError, match=DAFT_VERSION):
        write_daft_json(path, payload)

    assert not path.exists()


def test_write_daft_json_exclusive_create_preserves_existing_file(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip", iso_date="2026-05-20")
    path = tmp_path / "video.json"
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(DaftWriteError, match="Refusing to overwrite existing DAFT file"):
        write_daft_json(path, _video_payload(ctx), overwrite=False)

    assert path.read_text(encoding="utf-8") == "existing\n"


def test_write_daft_json_removes_unique_temp_on_replace_failure(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip", iso_date="2026-05-20")
    path = tmp_path / "video.json"
    path.mkdir()

    with pytest.raises(DaftWriteError, match="Failed to write DAFT file"):
        write_daft_json(path, _video_payload(ctx), overwrite=True)

    assert path.is_dir()
    assert not list(tmp_path.glob(".video.json.*.tmp"))
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_scene_writer_rejects_path_traversal_filename(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip")
    writer = DaftSceneWriter(tmp_path / "scene", ctx=ctx)

    with pytest.raises(DaftWriteError, match="plain file name"):
        writer.write_payload(_video_payload(ctx), filename="../video.json")


def test_scene_writer_runs_injected_validator_before_writing(tmp_path: Path) -> None:
    ctx = SceneContext(media_id="clip")
    writer = DaftSceneWriter(
        tmp_path / "scene",
        ctx=ctx,
        validator=CompositeDaftValidator([StructuralDaftValidator(), RejectingValidator()]),
    )

    with pytest.raises(DaftValidationError, match="rejected"):
        writer.write_payload(_video_payload(ctx))

    assert not (tmp_path / "scene" / "contextual" / "video.json").exists()


def test_validator_protocol_accepts_structural_validator() -> None:
    validator: DaftPayloadValidator = StructuralDaftValidator()

    assert not validator.validate_payload(
        {
            "version": DAFT_VERSION,
            "metadata": {"type": "open_qa"},
            "items": [{"question": "q", "answer": "a"}],
        }
    )


def test_json_schema_validator_reports_schema_errors(tmp_path: Path) -> None:
    schema_path = tmp_path / "schemas" / "tasks" / "open_qa.schema.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        """
        {
          "$schema": "http://json-schema.org/draft-07/schema#",
          "type": "object",
          "required": ["version", "metadata", "items"],
          "properties": {
            "version": {"const": "metropolis-v3.0"},
            "metadata": {
              "type": "object",
              "required": ["type"],
              "properties": {"type": {"const": "open_qa"}}
            },
            "items": {
              "type": "array",
              "minItems": 1,
              "items": {
                "type": "object",
                "required": ["question", "answer"],
                "properties": {
                  "question": {"type": "string"},
                  "answer": {"type": "string"}
                }
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    validator = JsonSchemaDaftPayloadValidator(tmp_path / "schemas")

    errors = validator.validate_payload(
        {"version": DAFT_VERSION, "metadata": {"type": "open_qa"}, "items": [{"question": "q"}]},
        expected_type="open_qa",
        kind="task",
        path=Path("open_qa.json"),
    )

    assert len(errors) == 1
    assert "answer" in errors[0]
