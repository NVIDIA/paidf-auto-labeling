# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcq_generation.mcq.runners.metadata_llm import MetadataLlmRunner


def _write(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_metadata_llm_outputs_windowing_when_missing(monkeypatch, tmp_path):
    meta_in = tmp_path / "metadata.json"
    _write(
        meta_in,
        {
            "video_id": "v",
            "framerate": 30.0,
            "windows": [
                {"start_frame": 0, "end_frame": 59, "vlm_caption": "cap0"},
                {"start_frame": 60, "end_frame": 119, "vlm_caption": "cap1"},
            ],
        },
    )

    scene_dir = tmp_path / "scene"

    runner = MetadataLlmRunner(
        mcq_prompt="PROMPT",
        llm_base_url="u",
        llm_model="m",
        aggregate_windows=False,
    )

    def _fake_call_one(*, prompt_text, caption, logger):
        return (
            {
                "version": 2.0,
                "video_id": "v",
                "mcq": [{"id": "1_01", "question": "q", "options": ["A. Yes", "B. No"], "answer": "B. No"}],
            },
            "RAW",
        )

    monkeypatch.setattr(runner, "_call_one", _fake_call_one)

    runner.run_single(input_metadata_json=meta_in, output_dir=scene_dir, verbose=False)

    out = json.loads((scene_dir / "sidecars" / "metadata.json").read_text(encoding="utf-8"))
    assert "windowing" in out
    assert out["windowing"]["window_mode"] == "frames"
    assert out["windowing"]["window_frames"] == 60
    assert out["windowing"]["sampling_fps"] > 0
    assert out["windowing"]["resolution"] > 0
    assert out["windowing"]["max_frames"] > 0
    assert isinstance(out["windows"][0]["llm_enhanced_caption"], dict)
    assert out["windows"][0]["llm_enhanced_caption"]["mcq"][0]["answer"] == "B. No"


def test_metadata_llm_preserves_windowing_and_inherits_verify_sampling(monkeypatch, tmp_path):
    meta_in = tmp_path / "metadata.json"
    _write(
        meta_in,
        {
            "video_id": "v",
            "framerate": 30.0,
            "windowing": {"sampling_fps": 1.0, "resolution": 360, "max_frames": 24},
            "windows": [{"start_frame": 0, "end_frame": 59, "vlm_caption": "cap0"}],
        },
    )
    scene_dir = tmp_path / "scene"
    task_dir = scene_dir / "task"
    sidecars_dir = scene_dir / "sidecars"
    task_dir.mkdir(parents=True)
    sidecars_dir.mkdir(parents=True)
    (task_dir / "mcq.json").write_text('{"items": [{"question": "old", "answer": "A"}]}', encoding="utf-8")
    _write(sidecars_dir / "mcq.empty.json", {"version": 2.0, "video_id": "old", "mcq": [], "_error": "zero_task_items"})

    runner = MetadataLlmRunner(
        mcq_prompt="PROMPT",
        llm_base_url="u",
        llm_model="m",
        # verify_* left as None -> should inherit from metadata.windowing
        verify_sampling_fps=None,
        verify_resolution=None,
        verify_max_frames=None,
        aggregate_windows=False,
    )

    def _fake_call_one(*, prompt_text, caption, logger):
        return (
            {
                "version": 2.0,
                "video_id": "v",
                "mcq": [{"id": "1_01", "question": "q", "options": ["A. Yes", "B. No"], "answer": "B. No"}],
            },
            "RAW",
        )

    monkeypatch.setattr(runner, "_call_one", _fake_call_one)

    runner.run_single(input_metadata_json=meta_in, output_dir=scene_dir, verbose=False)

    out = json.loads((scene_dir / "sidecars" / "metadata.json").read_text(encoding="utf-8"))
    assert out["windowing"]["sampling_fps"] == pytest.approx(1.0)
    assert out["windowing"]["resolution"] == 360
    assert out["windowing"]["max_frames"] == 24


def test_metadata_llm_writes_empty_marker_for_zero_task_result(monkeypatch, tmp_path):
    meta_in = tmp_path / "metadata.json"
    _write(
        meta_in,
        {
            "video_id": "v",
            "framerate": 30.0,
            "windows": [{"start_frame": 0, "end_frame": 59, "vlm_caption": "cap0"}],
        },
    )
    scene_dir = tmp_path / "scene"

    runner = MetadataLlmRunner(
        mcq_prompt="PROMPT",
        llm_base_url="u",
        llm_model="m",
        aggregate_windows=False,
    )

    def _fake_call_one(*, prompt_text, caption, logger):
        return ({"version": 2.0, "video_id": "v", "mcq": []}, "RAW")

    monkeypatch.setattr(runner, "_call_one", _fake_call_one)

    runner.run_single(input_metadata_json=meta_in, output_dir=scene_dir, verbose=False)

    empty = json.loads((scene_dir / "sidecars" / "mcq.empty.json").read_text(encoding="utf-8"))
    assert empty["_error"] == "zero_task_items"
    assert not (scene_dir / "task" / "mcq.json").exists()


def test_metadata_llm_parse_failure_does_not_write_empty_marker(monkeypatch, tmp_path):
    meta_in = tmp_path / "metadata.json"
    bank = tmp_path / "bank.json"
    _write(
        bank,
        {
            "questions": [
                {"id": "1_01", "question": "q", "options": ["A. Yes", "B. No"]},
            ]
        },
    )
    _write(
        meta_in,
        {
            "video_id": "v",
            "framerate": 30.0,
            "windows": [{"start_frame": 0, "end_frame": 59, "vlm_caption": "cap0"}],
        },
    )
    scene_dir = tmp_path / "scene"

    runner = MetadataLlmRunner(
        mcq_prompt="PROMPT",
        llm_base_url="u",
        llm_model="m",
        aggregate_windows=False,
        retry_missing_questions=True,
        question_bank_file=bank,
    )

    def _fake_call_one(*, prompt_text, caption, logger):
        return (None, '{"version": 2.0}')

    monkeypatch.setattr(runner, "_call_one", _fake_call_one)
    monkeypatch.setattr("mcq_generation.mcq.runners.metadata_llm.retry_fill_missing_questions", lambda **_kwargs: None)

    runner.run_single(input_metadata_json=meta_in, output_dir=scene_dir, verbose=False)

    out = json.loads((scene_dir / "sidecars" / "metadata.json").read_text(encoding="utf-8"))
    window = out["windows"][0]
    assert window["llm_enhanced_caption"] == {}
    assert "llm_mcq_not_parseable:missing_mcq_list" in window["_errors"]
    assert not (scene_dir / "sidecars" / "mcq.empty.json").exists()
    assert not (scene_dir / "task" / "mcq.json").exists()


def test_metadata_llm_skip_existing_uses_resume_marker(monkeypatch, tmp_path):
    meta_in = tmp_path / "metadata.json"
    _write(
        meta_in,
        {
            "video_id": "v",
            "framerate": 30.0,
            "windows": [{"start_frame": 0, "end_frame": 59, "vlm_caption": "cap0"}],
        },
    )
    scene_dir = tmp_path / "scene"
    sidecars = scene_dir / "sidecars"
    sidecars.mkdir(parents=True)
    _write(
        sidecars / "metadata.json",
        {
            "video_id": "v",
            "windows": [
                {
                    "start_frame": 0,
                    "end_frame": 59,
                    "vlm_caption": "cap0",
                    "llm_enhanced_caption": {"mcq": []},
                }
            ],
        },
    )
    _write(sidecars / "mcq.empty.json", {"version": 2.0, "video_id": "v", "mcq": [], "_error": "zero_task_items"})

    runner = MetadataLlmRunner(
        mcq_prompt="PROMPT",
        llm_base_url="u",
        llm_model="m",
        skip_existing=True,
    )

    def _fake_call_one(*, prompt_text, caption, logger):
        raise AssertionError("skip_existing should avoid LLM calls")

    monkeypatch.setattr(runner, "_call_one", _fake_call_one)

    runner.run_single(input_metadata_json=meta_in, output_dir=scene_dir, verbose=False)
