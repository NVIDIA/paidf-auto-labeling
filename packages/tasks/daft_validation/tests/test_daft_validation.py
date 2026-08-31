# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from core import DataEntry, ScenePipelineState, write_pipeline_state
from daft_validation import DaftSceneValidationError, DaftValidationTask, validate_scene


def test_daft_validation_task_accepts_valid_scene(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(
        scene_dir / "contextual" / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "clip",
            "metadata": {"type": "video", "date": "2026-05-20"},
        },
    )
    _write_json(
        scene_dir / "task" / "open_qa.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "open_qa", "date": "2026-05-20"},
            "items": [{"video_id": "clip", "question": "Q?", "answer": "A"}],
        },
    )
    _write_json(scene_dir / "sidecars" / "metadata.json", {"schema_version": "1"})

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    assert validate_scene(entry) == []
    assert DaftValidationTask().run(entry) == entry


def test_daft_validation_uses_scene_id_for_transient_sidecar_media(tmp_path: Path) -> None:
    scene_dir = tmp_path / "clip"
    sr_output = scene_dir / "sidecars" / "sr_output.mp4"
    sr_output.parent.mkdir(parents=True)
    sr_output.touch()
    _write_json(
        scene_dir / "contextual" / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "clip",
            "metadata": {"type": "video", "date": "2026-05-20"},
        },
    )

    entry = DataEntry(media_path=str(sr_output), data_path=str(scene_dir))

    assert validate_scene(entry) == []


def test_daft_validation_prefers_persisted_media_id_for_transient_sidecar_media(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "pipeline_data_x"
    sr_output = scene_dir / "sidecars" / "sr_output.mp4"
    sr_output.parent.mkdir(parents=True)
    sr_output.touch()
    write_pipeline_state(scene_dir, ScenePipelineState(media_id="clip"))
    _write_json(
        scene_dir / "contextual" / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "clip",
            "metadata": {"type": "video", "date": "2026-05-20"},
        },
    )

    entry = DataEntry(media_path=str(sr_output), data_path=str(scene_dir))

    assert validate_scene(entry) == []


def test_daft_validation_tolerates_persona_qa_task_artifact(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(
        scene_dir / "task" / "open_qa.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "open_qa", "date": "2026-05-20"},
            "items": [{"video_id": "clip", "question": "Q?", "answer": "A"}],
        },
    )
    # Auxiliary per-person artifact: not a registered DAFT type, must be tolerated.
    _write_json(
        scene_dir / "task" / "persona_qa.json",
        {
            "schema_version": "1",
            "media_id": "clip",
            "type": "persona_qa",
            "n_personas": 1,
            "personas": [{"track_id": 0, "mcq": [], "bcq": [], "open_qa": []}],
        },
    )

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    assert validate_scene(entry) == []
    assert DaftValidationTask().run(entry) == entry


def test_daft_validation_rejects_stage_deliverables_under_task(tmp_path: Path) -> None:
    """Referring/grounding JSON is non-DAFT and must not live under task/."""
    media_path = tmp_path / "frame.jpg"
    media_path.touch()
    scene_dir = tmp_path / "frame"
    _write_json(
        scene_dir / "task" / "referring_expressions.json",
        {"schema_version": "1", "regions": []},
    )
    _write_json(
        scene_dir / "task" / "grounding_2d.json",
        {"schema_version": "1", "expressions": []},
    )

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    with pytest.raises(DaftSceneValidationError, match="unknown DAFT annotation filename"):
        DaftValidationTask().run(entry)


def test_daft_validation_allows_stage_deliverables_under_sidecars(tmp_path: Path) -> None:
    """Sidecar stage outputs must not fail DAFT validation on scene reruns."""
    media_path = tmp_path / "frame.jpg"
    media_path.touch()
    scene_dir = tmp_path / "frame"
    _write_json(
        scene_dir / "contextual" / "objects.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "objects", "date": "2026-05-20"},
            "frames": [],
        },
    )
    _write_json(
        scene_dir / "sidecars" / "referring_expressions" / "referring_expressions.json",
        {"schema_version": "1", "regions": []},
    )
    _write_json(
        scene_dir / "sidecars" / "grounding_2d" / "grounding_2d.json",
        {"schema_version": "1", "expressions": []},
    )

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    assert validate_scene(entry) == []
    assert DaftValidationTask().run(entry) == entry


def test_daft_validation_rejects_non_daft_contextual_files(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(scene_dir / "contextual" / "video_captions.json", {"metadata": {}})

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    with pytest.raises(DaftSceneValidationError, match="unknown DAFT annotation filename"):
        DaftValidationTask().run(entry)


def test_daft_validation_tolerates_adapter_contextual_deliverables(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(
        scene_dir / "contextual" / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "clip",
            "metadata": {"type": "video", "date": "2026-05-20"},
        },
    )
    # Legacy contextual deliverables emitted by the annotation-schema adapter.
    _write_json(scene_dir / "contextual" / "pas.json", {"pas": {"n_people": 0}})
    _write_json(scene_dir / "contextual" / "pas_anomaly.json", {"official_anomaly": None})

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    assert validate_scene(entry) == []
    assert DaftValidationTask().run(entry) == entry


def test_daft_validation_rejects_contextual_media_id_mismatch(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(
        scene_dir / "contextual" / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "different_clip",
            "metadata": {"type": "video", "date": "2026-05-20"},
        },
    )

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    with pytest.raises(DaftSceneValidationError, match="expected 'clip'"):
        DaftValidationTask().run(entry)


def test_daft_validation_rejects_task_item_media_id_mismatch(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(
        scene_dir / "task" / "open_qa.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "open_qa", "date": "2026-05-20"},
            "items": [{"video_id": "different_clip", "question": "Q?", "answer": "A"}],
        },
    )

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))
    issues = validate_scene(entry)

    assert any("items[0].video_id expected 'clip'" in issue for issue in issues)


def test_daft_validation_rejects_task_item_wrong_scene_id_field(tmp_path: Path) -> None:
    media_path = tmp_path / "frame.png"
    media_path.touch()
    scene_dir = tmp_path / "frame"
    _write_json(
        scene_dir / "task" / "open_qa.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "open_qa", "date": "2026-05-20"},
            "items": [{"video_id": "frame", "question": "Q?", "answer": "A"}],
        },
    )

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))
    issues = validate_scene(entry)

    assert any("items[0].image_id expected 'frame'" in issue for issue in issues)
    assert any("items[0] must not contain video_id" in issue for issue in issues)


def test_daft_validation_rejects_structurally_invalid_task_payload(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    _write_json(
        scene_dir / "task" / "open_qa.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "mcq", "date": "2026-05-20"},
            "items": [],
        },
    )

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))
    issues = validate_scene(entry)

    assert any("metadata.type must be 'open_qa'" in issue for issue in issues)
    assert any("requires a non-empty items list" in issue for issue in issues)


def test_daft_validation_rejects_malformed_json(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    path = scene_dir / "task" / "open_qa.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json\n", encoding="utf-8")

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    with pytest.raises(DaftSceneValidationError, match="invalid JSON"):
        DaftValidationTask().run(entry)


def test_daft_validation_rejects_non_utf8_json(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()
    scene_dir = tmp_path / "clip"
    path = scene_dir / "task" / "open_qa.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff")

    entry = DataEntry(media_path=str(media_path), data_path=str(scene_dir))

    with pytest.raises(DaftSceneValidationError, match="could not decode JSON file as UTF-8"):
        DaftValidationTask().run(entry)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
