# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from core import (
    AnnotationExportState,
    DataEntry,
    EmitterOutcomeState,
    ScenePipelineState,
    read_pipeline_state,
    write_pipeline_state,
)
from core.policy import EmptyOutputPolicy
from reasoning.artifacts import CAPTION_ARTIFACTS_KEY
from reasoning_service.main import ReasoningService, build_resolver, main


def test_container_installs_ffmpeg_without_unused_python_media_packages() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    # FFmpeg comes from the shared input-only media base; the service no longer
    # compiles it and installs no OpenCV/PyAV wheels.
    assert "FROM ${BUILDER_BASE_IMAGE} AS builder" in text
    assert "ffmpeg-install" not in text
    assert text.count("media_toolchain.py verify") == 2
    for unused_media_component in (
        "opencv-python",
        "opencv-headless",
        "pyav",
        "find-links",
    ):
        assert unused_media_component not in text.lower()


def test_reasoning_service_parser_maps_llm_endpoint_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = argparse.ArgumentParser()
    ReasoningService().add_service_args(parser)
    monkeypatch.setenv("LLM_ENDPOINT_URL", "http://env-only/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")

    args = parser.parse_args(
        [
            "--llm-endpoint-url",
            "http://localhost:8000/v1",
            "--llm-model",
            "test-model",
        ]
    )
    resolver = build_resolver(args, logger=logging.getLogger("test.reasoning_service"))

    assert resolver.resolve_llm() == ("http://localhost:8000/v1", "test-model")
    assert resolver.resolve_llm_api_key() == "secret"
    assert not hasattr(args, "llm_api_key_env")


def test_reasoning_service_validates_before_and_after_reasoning() -> None:
    service = ReasoningService()
    parser = argparse.ArgumentParser()
    service.add_service_args(parser)
    args = parser.parse_args([])
    entries = [
        DataEntry(
            id="entry-1",
            media_path="/data/source/clip.mp4",
            data_path="/data/scene",
        )
    ]
    validation_before = object()
    validation_after = object()
    reasoning_task = object()

    with (
        patch("reasoning_service.main.DaftValidationTask") as validation_task_cls,
        patch("reasoning_service.main.ReasoningTask") as task_cls,
        patch("reasoning_service.main.LinearPipeline") as pipeline_cls,
    ):
        validation_task_cls.side_effect = [validation_before, validation_after]
        task_cls.return_value = reasoning_task
        pipeline = pipeline_cls.return_value
        pipeline.run.return_value = entries

        service.execute(args, entries)

    assert validation_task_cls.call_count == 2
    task_cls.assert_called_once()
    pipeline_cls.assert_called_once()
    _, kwargs = pipeline_cls.call_args
    assert kwargs["name"] == "reasoning_pipeline"
    assert kwargs["policy"] is EmptyOutputPolicy.FAIL
    assert kwargs["tasks"] == [validation_before, reasoning_task, validation_after]
    pipeline.run.assert_called_once_with(entries)


def test_reasoning_service_does_not_emit_service_owned_pivots(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()

    scene_dir = tmp_path / "clip"
    (scene_dir / "contextual").mkdir(parents=True)
    (scene_dir / "sidecars").mkdir()

    _write_json(
        scene_dir / "contextual" / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "clip",
            "metadata": {"type": "video", "date": "2026-05-20"},
            "format": "mp4",
            "fps": 30,
            "duration": 1.0,
            "height": 720,
            "width": 1280,
            "scene_description": "A car drives through an intersection.",
            "event_summary": "A car passes through the scene.",
        },
    )
    _write_json(
        scene_dir / "sidecars" / "metadata_chunk.json",
        {
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "description": "A car drives through an intersection.",
                }
            ]
        },
    )

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"media_path": str(media_path), "data_path": str(scene_dir)}) + "\n",
        encoding="utf-8",
    )

    argv = ["reasoning-service", "--input-file", str(input_file)]
    with patch.object(sys, "argv", argv):
        main()

    assert not (scene_dir / "contextual" / "chunks.json").exists()
    assert not (scene_dir / "task" / "scene_description.json").exists()
    assert not (scene_dir / "task" / "video_summarization.json").exists()
    assert not (scene_dir / "task" / "temporal_description.json").exists()

    state = read_pipeline_state(scene_dir)
    assert state.annotation_export is not None
    assert state.annotation_export.success is False
    outcomes = {emitter.name: emitter for emitter in state.annotation_export.emitters}
    assert "chunks" not in outcomes
    assert "scene_description" not in outcomes
    assert "temporal_description" not in outcomes
    assert "video_summarization" not in outcomes
    assert outcomes["events"].success is False
    assert outcomes["msted"].success is False
    assert outcomes["temporal_localization"].success is False


def test_reasoning_service_preserves_existing_pipeline_media_path(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()

    scene_dir = tmp_path / "clip"
    (scene_dir / "contextual").mkdir(parents=True)
    sidecars_dir = scene_dir / "sidecars"
    sidecars_dir.mkdir()
    active_media = sidecars_dir / "active.mp4"
    active_media.touch()

    _write_json(
        scene_dir / "contextual" / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "clip",
            "metadata": {"type": "video", "date": "2026-05-20"},
            "format": "mp4",
            "fps": 30,
            "duration": 1.0,
            "height": 720,
            "width": 1280,
            "scene_description": "A car drives through an intersection.",
            "event_summary": "A car passes through the scene.",
        },
    )
    _write_json(
        scene_dir / "sidecars" / "metadata_chunk.json",
        {
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "description": "A car drives through an intersection.",
                }
            ]
        },
    )
    write_pipeline_state(
        scene_dir,
        ScenePipelineState(
            media_path=str(active_media),
        ),
    )

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"media_path": str(media_path), "data_path": str(scene_dir)}) + "\n",
        encoding="utf-8",
    )

    argv = ["reasoning-service", "--input-file", str(input_file)]
    with patch.object(sys, "argv", argv):
        main()

    state = read_pipeline_state(scene_dir)
    assert state.media_path == str(active_media)
    assert state.annotation_export is not None
    assert state.annotation_export.success is False


def test_reasoning_service_preserves_service_owned_emitter_state(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()

    scene_dir = tmp_path / "clip"
    (scene_dir / "contextual").mkdir(parents=True)
    (scene_dir / "task").mkdir()
    chunks = scene_dir / "contextual" / "chunks.json"
    _write_json(
        chunks,
        {"version": "metropolis-v3.0", "metadata": {"type": "chunks"}, "chunks": []},
    )
    mcq = scene_dir / "task" / "mcq.json"
    _write_json(
        mcq,
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "mcq"},
            "items": [
                {
                    "video_id": "clip",
                    "question": "Q?",
                    "options": {"A": "A", "B": "B"},
                    "answer": "A",
                }
            ],
        },
    )
    write_pipeline_state(
        scene_dir,
        ScenePipelineState(
            annotation_export=AnnotationExportState(
                success=True,
                emitters=[
                    EmitterOutcomeState(name="chunks", success=True, artifact=str(chunks)),
                    EmitterOutcomeState(name="mcq", success=True, artifact=str(mcq)),
                ],
            )
        ),
    )

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"media_path": str(media_path), "data_path": str(scene_dir)}) + "\n",
        encoding="utf-8",
    )

    argv = ["reasoning-service", "--input-file", str(input_file)]
    with patch.object(sys, "argv", argv):
        main()

    state = read_pipeline_state(scene_dir)
    assert state.annotation_export is not None
    assert state.annotation_export.success is True
    outcomes = {emitter.name: emitter for emitter in state.annotation_export.emitters}
    assert outcomes["chunks"].success is True
    assert outcomes["chunks"].artifact == str(chunks)
    assert outcomes["mcq"].success is True
    assert outcomes["mcq"].artifact == str(mcq)
    assert outcomes["events"].success is False


def test_reasoning_service_runs_llm_stage_from_captioning_artifact_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()

    scene_dir = tmp_path / "clip"
    (scene_dir / "contextual").mkdir(parents=True)
    (scene_dir / "sidecars" / "captioning").mkdir(parents=True)

    metadata_chunk = scene_dir / "sidecars" / "captioning" / "metadata_chunk.json"
    _write_json(
        metadata_chunk,
        {
            "duration_span": [0.0, 1.0],
            "summary": "A cyclist falls near a stopped car.",
            "windows": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "description": "A car stops after a cyclist falls in the roadway.",
                    "parsed": {
                        "scene_description": "A road incident is visible.",
                        "event_summary": "A cyclist falls near a stopped car.",
                    },
                }
            ],
        },
    )
    write_pipeline_state(
        scene_dir,
        ScenePipelineState(
            task_artifacts={
                CAPTION_ARTIFACTS_KEY: {
                    "success": True,
                    "metadata_chunk_json": str(metadata_chunk),
                }
            }
        ),
    )

    repo_root = tmp_path / "repo"
    bank_path = repo_root / "cookbooks" / "traffic" / "question_bank.json"
    bank_path.parent.mkdir(parents=True)
    _write_json(
        bank_path,
        {
            "questions": [{"question": "Legacy MCQ question should not be used."}],
            "open_qa": [{"question": "Describe the traffic safety event."}],
        },
    )
    config_file = tmp_path / "configs" / "config.yaml"
    config_file.parent.mkdir()
    config_file.write_text(
        "\n".join(
            [
                "reasoning:",
                "  open_qa:",
                "    enabled: true",
                "    question_file: cookbooks/traffic/question_bank.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"media_path": str(media_path), "data_path": str(scene_dir)}) + "\n",
        encoding="utf-8",
    )

    def _stub_qa_response(**_kwargs: object) -> tuple[dict[str, object], str]:
        payload: dict[str, object] = {
            "items": [
                {
                    "question": "Describe the traffic safety event.",
                    "answer": "A cyclist falls near a stopped car.",
                }
            ]
        }
        return payload, json.dumps(payload)

    argv = [
        "reasoning-service",
        "--input-file",
        str(input_file),
        "--config-file",
        str(config_file),
        "--llm-endpoint-url",
        "http://localhost:8000/v1",
        "--llm-model",
        "test-model",
    ]
    monkeypatch.chdir(repo_root)
    with (
        patch.object(sys, "argv", argv),
        patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=_stub_qa_response,
        ),
    ):
        main()

    open_qa = json.loads((scene_dir / "task" / "open_qa.json").read_text(encoding="utf-8"))
    assert open_qa["items"][0]["answer"] == "A cyclist falls near a stopped car."
    state = read_pipeline_state(scene_dir)
    assert state.annotation_export is not None
    outcomes = {emitter.name: emitter for emitter in state.annotation_export.emitters}
    assert outcomes["open_qa"].success is True
    assert outcomes["open_qa"].artifact == str(scene_dir / "task" / "open_qa.json")


def test_reasoning_service_records_only_current_run_outputs(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.touch()

    scene_dir = tmp_path / "clip"
    (scene_dir / "task").mkdir(parents=True)
    _write_json(
        scene_dir / "task" / "open_qa.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "open_qa", "date": "2026-05-20"},
            "items": [{"video_id": "clip", "question": "Q?", "answer": "A"}],
        },
    )

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"media_path": str(media_path), "data_path": str(scene_dir)}) + "\n",
        encoding="utf-8",
    )

    argv = ["reasoning-service", "--input-file", str(input_file)]
    with patch.object(sys, "argv", argv):
        main()

    state = read_pipeline_state(scene_dir)
    assert state.annotation_export is not None
    assert state.annotation_export.success is False
    outcomes = {emitter.name: emitter for emitter in state.annotation_export.emitters}
    assert "open_qa" not in outcomes


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
