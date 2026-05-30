# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from pathlib import Path

import mcq_generation.mcq.runners.window_direct_vlm as direct_r
import mcq_generation.mcq.runners.window_vlm_llm as window_r
import mcq_generation.mcq.utils.retry_missing as retry_m
import mcq_generation.mcq.utils.vlm_verify as vlm_verify_m
import pytest
from mcq_generation.mcq.runners.window_direct_vlm import WindowDirectVlmRunner
from mcq_generation.mcq.runners.window_vlm_llm import WindowVlmLlmRunner
from mcq_generation.mcq.utils.video import VideoInfo
from mcq_generation.mcq.utils.vlm_verify import parse_vlm_verify_items


def test_window_metadata_extraction_writes_metadata_and_mcq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake input layout: <input_root>/<clip_id>/video.mp4
    input_root = tmp_path / "inputs"
    clip_id = "clip_001"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    output_root = tmp_path / "out"

    # Avoid ffprobe/ffmpeg. Patch video info and frame extraction.
    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=8.0, fps=10.0, width=1280, height=720, num_frames=80),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)

    def _fake_call_chat_raw(*, base_url: str, messages, **_kwargs):
        # VLM caption call: multimodal user message
        if str(base_url).startswith("https://vlm."):
            return "[CATEGORY: NORMAL TRAFFIC]\n[CONFIDENCE: high]\n\n[Weather & Lighting]: Sunny.\n"
        return ""

    monkeypatch.setattr(window_r, "call_chat_raw", _fake_call_chat_raw)
    monkeypatch.setattr(
        window_r,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [
                    {
                        "id": "1_1",
                        "question": "Is there a traffic accident taking place?",
                        "options": ["Yes", "No"],
                        "answer": "No",
                    }
                ],
            },
            json.dumps(
                {
                    "version": 2.0,
                    "video_id": "",
                    "mcq": [
                        {
                            "id": "1_1",
                            "question": "Is there a traffic accident taking place?",
                            "options": ["Yes", "No"],
                            "answer": "No",
                        }
                    ],
                }
            ),
        ),
    )

    scene_dir = tmp_path / "scene"
    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={},
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
    )
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    meta_path = scene_dir / "sidecars" / "metadata.json"
    bcq_path = scene_dir / "task" / "bcq.json"
    assert meta_path.exists()
    assert bcq_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["video_id"] == clip_id
    assert isinstance(meta["windows"], list)
    assert len(meta["windows"]) >= 1
    assert "vlm_caption" in meta["windows"][0]
    assert "llm_enhanced_caption" in meta["windows"][0]
    assert isinstance(meta["windows"][0]["llm_enhanced_caption"], dict)
    assert meta["windows"][0]["llm_enhanced_caption"]["mcq"][0]["answer"] == "No"

    bcq = json.loads(bcq_path.read_text(encoding="utf-8"))
    items = bcq["items"]
    assert len(items) == 1
    assert items[0]["video_id"] == "main"
    assert items[0]["question"] == "Is there a traffic accident taking place?"
    assert items[0]["answer"] == "No"
    assert "options" not in items[0]


def test_window_vlm_llm_parse_failure_does_not_write_empty_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_parse_fail"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(window_r, "call_chat_raw", lambda **_kwargs: "caption")
    monkeypatch.setattr(
        window_r, "call_chat_json_with_structured_fallback", lambda **_kwargs: (None, '{"version": 2.0}')
    )
    monkeypatch.setattr(window_r, "retry_fill_missing_questions", lambda **_kwargs: None)

    scene_dir = tmp_path / "scene"
    (scene_dir / "task").mkdir(parents=True)
    (scene_dir / "sidecars").mkdir(parents=True)
    (scene_dir / "task" / "mcq.json").write_text('{"items": [{"question": "old", "answer": "A"}]}', encoding="utf-8")
    (scene_dir / "sidecars" / "mcq.empty.json").write_text(
        '{"version": 2.0, "video_id": "old", "mcq": [], "_error": "zero_task_items"}',
        encoding="utf-8",
    )
    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={},
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        retry_missing_questions=True,
        question_bank={
            "questions": [
                {"id": "1_1", "question": "Q", "options": ["A", "B"]},
            ]
        },
    )

    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=tmp_path / "out",
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    assert (scene_dir / "sidecars" / "metadata.json").exists()
    meta = json.loads((scene_dir / "sidecars" / "metadata.json").read_text(encoding="utf-8"))
    window = meta["windows"][0]
    assert window["llm_enhanced_caption"] == {}
    assert "llm_mcq_not_parseable:missing_mcq_list" in window["_errors"]
    assert not (scene_dir / "sidecars" / "mcq.empty.json").exists()
    assert not (scene_dir / "task" / "mcq.json").exists()


def test_window_direct_vlm_parse_failure_retry_does_not_write_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_direct_parse_fail"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(
        direct_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(direct_r, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(
        direct_r, "call_chat_json_with_structured_fallback", lambda **_kwargs: (None, '{"version": 2.0}')
    )

    scene_dir = tmp_path / "scene"
    runner = WindowDirectVlmRunner(
        mcq_prompt="mcq prompt",
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        question_bank={
            "questions": [
                {"id": "1_1", "question": "Q", "options": ["A", "B"]},
            ]
        },
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        retry_missing_questions=True,
    )

    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=tmp_path / "out",
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    meta = json.loads((scene_dir / "sidecars" / "metadata.json").read_text(encoding="utf-8"))
    window = meta["windows"][0]
    assert window["llm_enhanced_caption"] == {}
    assert "vlm_direct_mcq_not_parseable:missing_mcq_list" in window["_errors"]
    assert not (scene_dir / "task" / "mcq.json").exists()


def test_window_metadata_extraction_allows_custom_output_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_002"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    output_root = tmp_path / "out"

    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)

    def _fake_call_chat_raw(*, base_url: str, messages, **_kwargs):
        if str(base_url).startswith("https://vlm."):
            return "custom caption"
        return ""

    monkeypatch.setattr(window_r, "call_chat_raw", _fake_call_chat_raw)
    monkeypatch.setattr(
        window_r,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "1_1", "question": "Q", "options": ["A", "B"], "answer": "A"}],
            },
            json.dumps(
                {
                    "version": 2.0,
                    "video_id": "",
                    "mcq": [{"id": "1_1", "question": "Q", "options": ["A", "B"], "answer": "A"}],
                }
            ),
        ),
    )

    cap_key = "caption_custom"
    enh_key = "enhanced_custom"
    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={},
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        caption_key=cap_key,
        enhanced_caption_key=enh_key,
    )
    scene_dir = tmp_path / "scene"
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    meta_path = scene_dir / "sidecars" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert cap_key in meta["windows"][0]
    assert enh_key in meta["windows"][0]
    assert "llm_enhanced_caption" not in meta["windows"][0]


def test_window_metadata_extraction_retry_missing_questions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_003"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    output_root = tmp_path / "out"

    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)

    monkeypatch.setattr(
        window_r,
        "call_chat_raw",
        lambda *_, **__: "[CATEGORY: NORMAL TRAFFIC]\n[CONFIDENCE: high]\n\n[Weather & Lighting]: Sunny.\n",
    )

    # Minimal bank with 2 questions (expect both).
    bank = {
        "name": "test_bank",
        "questions": [
            {"id": "1_1", "question": "Q1", "options": ["Yes", "No"], "aggregation": "any"},
            {"id": "2_1", "question": "Q2", "options": ["Yes", "No"], "aggregation": "any"},
        ],
    }

    # First LLM MCQ call returns only 1_1.
    monkeypatch.setattr(
        window_r,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "1_1", "question": "Q1", "options": ["Yes", "No"], "answer": "No"}],
            },
            "{}",
        ),
    )

    # Retry call returns only the missing id 2_1.
    monkeypatch.setattr(
        retry_m,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "2_1", "question": "Q2", "options": ["Yes", "No"], "answer": "Yes"}],
            },
            "{}",
        ),
    )

    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={},
        aggregation_specs={},
        question_bank=bank,
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        retry_missing_questions=True,
        retry_missing_max_rounds=2,
    )
    scene_dir = tmp_path / "scene"
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    bcq_path = scene_dir / "task" / "bcq.json"
    bcq = json.loads(bcq_path.read_text(encoding="utf-8"))
    questions = {it["question"] for it in bcq["items"]}
    assert questions == {"Q1", "Q2"}


def test_retry_missing_includes_dependents_and_filters_include_if(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_004"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    output_root = tmp_path / "out"

    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(window_r, "call_chat_raw", lambda *_, **__: "caption")

    # Bank: 5_2 depends on 5_1 == Yes
    bank = {
        "name": "test_bank_inc",
        "questions": [
            {"id": "5_1", "question": "Gate", "options": ["Yes", "No"], "aggregation": "any"},
            {
                "id": "5_2",
                "question": "Dependent",
                "options": ["A", "B"],
                "aggregation": "majority",
                "include_if": {"5_1": "Yes"},
            },
        ],
    }

    # First MCQ call returns only dependent (missing gate) -> should be filtered out eventually.
    monkeypatch.setattr(
        window_r,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "5_2", "question": "Dependent", "options": ["A", "B"], "answer": "A"}],
            },
            "{}",
        ),
    )

    # Retry call returns both unanswered (5_1 is missing). It may also repeat 5_2.
    monkeypatch.setattr(
        retry_m,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [
                    {"id": "5_1", "question": "Gate", "options": ["Yes", "No"], "answer": "No"},
                    {"id": "5_2", "question": "Dependent", "options": ["A", "B"], "answer": "A"},
                ],
            },
            "{}",
        ),
    )

    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={"5_2": {"5_1": "Yes"}},
        aggregation_specs={},
        question_bank=bank,
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        retry_missing_questions=True,
        retry_missing_max_rounds=1,
    )
    scene_dir = tmp_path / "scene"
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    bcq_path = scene_dir / "task" / "bcq.json"
    mcq_path = scene_dir / "task" / "mcq.json"
    bcq = json.loads(bcq_path.read_text(encoding="utf-8"))
    assert [it["question"] for it in bcq["items"]] == ["Gate"]
    # 5_2 must be filtered out since 5_1 == No, so no MCQ file is written.
    assert not mcq_path.exists()


def test_window_level_retry_fills_missing_dependent_when_gate_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_005"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    output_root = tmp_path / "out"

    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(window_r, "call_chat_raw", lambda *_, **__: "caption")

    bank = {
        "name": "bank_vlvl",
        "questions": [
            {"id": "4_1", "question": "Gate", "options": ["Yes", "No"], "aggregation": "any"},
            {
                "id": "4_9",
                "question": "Dependent",
                "options": ["A", "B"],
                "aggregation": "majority",
                "include_if": {"4_1": "Yes"},
            },
        ],
    }

    # Initial MCQ call returns gate yes but no dependent.
    monkeypatch.setattr(
        window_r,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "4_1", "question": "Gate", "options": ["Yes", "No"], "answer": "Yes"}],
            },
            "{}",
        ),
    )

    def _fake_retry_call(**kwargs):
        # Required-missing retry should include known gate answers in the system prompt.
        messages = kwargs.get("messages") or []
        sys_msg = messages[0]["content"] if messages and isinstance(messages[0], dict) else ""
        assert "Known answers from previous pass" in sys_msg
        assert '"4_1": "Yes"' in sys_msg
        return (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "4_9", "question": "Dependent", "options": ["A", "B"], "answer": "A"}],
            },
            "{}",
        )

    monkeypatch.setattr(retry_m, "call_chat_json_with_structured_fallback", _fake_retry_call)

    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={"4_9": {"4_1": "Yes"}},
        aggregation_specs={},
        question_bank=bank,
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        retry_missing_questions=True,
        retry_missing_max_rounds=1,
    )
    scene_dir = tmp_path / "scene"
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    bcq = json.loads((scene_dir / "task" / "bcq.json").read_text(encoding="utf-8"))
    mcq = json.loads((scene_dir / "task" / "mcq.json").read_text(encoding="utf-8"))
    assert [it["question"] for it in bcq["items"]] == ["Gate"]
    assert [it["question"] for it in mcq["items"]] == ["Dependent"]


def test_window_vlm_verify_writes_sidecars_and_metadata_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_verify_001"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    output_root = tmp_path / "out"

    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(window_r, "call_chat_raw", lambda *_, **__: "caption")

    def _fake_llm_json_call(**kwargs):
        _ = kwargs
        return (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "1_1", "question": "Q1", "options": ["Yes", "No"], "answer": "No"}],
            },
            "{}",
        )

    def _fake_verify_object_call(**kwargs):
        _ = kwargs
        return (
            {
                "verifications": [
                    {
                        "id": "1_1",
                        "verdict": "not_supported",
                        "reasoning_trace": "Frame shows clear collision evidence.",
                        "suggested_answer": "Yes",
                        "echo_current_answer": "No",
                    }
                ]
            },
            "{}",
        )

    monkeypatch.setattr(window_r, "call_chat_json_with_structured_fallback", _fake_llm_json_call)
    monkeypatch.setattr(vlm_verify_m, "call_chat_object_with_structured_fallback", _fake_verify_object_call)

    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={},
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        vlm_verify_enabled=True,
        vlm_verify_apply_corrections=True,
        vlm_verify_max_tokens=256,
        vlm_verify_temperature=0.0,
    )
    scene_dir = tmp_path / "scene"
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    sidecars_dir = scene_dir / "sidecars"
    meta = json.loads((sidecars_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "vlm_verify" in meta["windows"][0]
    assert meta["windows"][0]["vlm_verify"]["status"] == "ok"
    assert meta["windows"][0]["vlm_verify"]["verifications"][0]["id"] == "1_1"

    verify_sidecar = json.loads((sidecars_dir / "mcq.vlm_verify.json").read_text(encoding="utf-8"))
    assert verify_sidecar["summary"]["questions_verified"] == 1
    assert verify_sidecar["summary"]["questions_corrected"] == 1

    bcq = json.loads((scene_dir / "task" / "bcq.json").read_text(encoding="utf-8"))
    assert bcq["items"][0]["answer"] == "Yes"


def test_window_vlm_verify_skips_open_qa_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_open_qa"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")
    output_root = tmp_path / "out"

    monkeypatch.setattr(
        window_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=1.0, fps=1.0, width=640, height=480, num_frames=1),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"fake frame")
        return [(start_sec, fp)]

    monkeypatch.setattr(window_r, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(window_r, "call_chat_raw", lambda **_kwargs: "A person wearing a dark jacket.")
    monkeypatch.setattr(
        window_r,
        "call_chat_json_with_structured_fallback",
        lambda **_kwargs: (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [
                    {
                        "id": "caption_1",
                        "question": "Describe the visible clothing.",
                        "answer": "A person wearing a dark jacket.",
                    }
                ],
            },
            "{}",
        ),
    )

    def _unexpected_verify_call(**_kwargs):
        pytest.fail("Open-QA items without options should not call VLM verify")

    monkeypatch.setattr(vlm_verify_m, "call_chat_object_with_structured_fallback", _unexpected_verify_call)

    runner = WindowVlmLlmRunner(
        scene_prompt="scene prompt",
        mcq_prompt="mcq prompt",
        include_if_map={},
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        window_seconds=4.0,
        single_window=True,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        llm_max_tokens=256,
        timeout=60,
        rate_limit=0.0,
        vlm_verify_enabled=True,
        vlm_verify_prompt_template="verify prompt {current_mcq_answers}",
    )
    scene_dir = tmp_path / "scene"
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    assert not (scene_dir / "sidecars" / "mcq.vlm_verify.json").exists()
    assert not (scene_dir / "task" / "mcq.json").exists()
    assert (scene_dir / "task" / "open_qa.json").exists()
    meta = json.loads((scene_dir / "sidecars" / "metadata.json").read_text(encoding="utf-8"))
    assert "vlm_verify" not in meta["windows"][0]


def test_window_direct_vlm_retry_fills_missing_dependent_when_gate_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    clip_id = "clip_direct_001"
    clip_dir = input_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "video.mp4"
    clip_path.write_bytes(b"fake mp4 bytes")

    output_root = tmp_path / "out"

    monkeypatch.setattr(
        direct_r,
        "probe_video",
        lambda _p: VideoInfo(duration_sec=4.0, fps=10.0, width=1280, height=720, num_frames=40),
    )

    def _fake_extract_frames(*, out_dir: Path, start_sec: float, end_sec: float, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "frame_000001.jpg"
        fp.write_bytes(b"not a real jpeg, but good enough for base64")
        return [(start_sec, fp)]

    monkeypatch.setattr(direct_r, "extract_frames", _fake_extract_frames)

    bank = {
        "name": "bank_direct",
        "questions": [
            {"id": "4_1", "question": "Gate", "options": ["Yes", "No"], "aggregation": "any"},
            {
                "id": "4_9",
                "question": "Dependent",
                "options": ["A", "B"],
                "aggregation": "majority",
                "include_if": {"4_1": "Yes"},
            },
        ],
    }

    calls = {"n": 0}

    def _fake_vlm_call(**kwargs):
        calls["n"] += 1
        messages = kwargs.get("messages") or []
        sys_prompt = messages[0].get("content") if messages and isinstance(messages[0], dict) else ""
        if calls["n"] == 1:
            # Initial direct MCQ: gate Yes, missing dependent.
            assert "Known answers from previous pass" not in sys_prompt
            return (
                {
                    "version": 2.0,
                    "video_id": "",
                    "mcq": [{"id": "4_1", "question": "Gate", "options": ["Yes", "No"], "answer": "Yes"}],
                },
                "{}",
            )
        # Retry: should include known gate answer in system prompt.
        assert "Known answers from previous pass" in sys_prompt
        assert '"4_1": "Yes"' in sys_prompt
        return (
            {
                "version": 2.0,
                "video_id": "",
                "mcq": [{"id": "4_9", "question": "Dependent", "options": ["A", "B"], "answer": "A"}],
            },
            "{}",
        )

    monkeypatch.setattr(direct_r, "call_chat_json_with_structured_fallback", _fake_vlm_call)

    runner = WindowDirectVlmRunner(
        mcq_prompt="mcq prompt",
        vlm_base_url="https://vlm.example.com/v1",
        vlm_model="vlm-model",
        include_if_map={"4_9": {"4_1": "Yes"}},
        aggregation_specs={},
        question_bank=bank,
        window_seconds=4.0,
        sampling_fps=2.0,
        resolution=480,
        max_frames=10,
        vlm_max_tokens=256,
        vlm_temperature=0.0,
        timeout=60,
        rate_limit=0.0,
        aggregate_windows=True,
        retry_missing_questions=True,
        retry_missing_max_rounds=2,
    )
    scene_dir = tmp_path / "scene"
    runner.build_for_clip(
        clip_path=clip_path,
        input_root=input_root,
        output_root=output_root,
        output_dir=scene_dir,
        logger=logging.getLogger("test"),
    )

    # BCQ gate (`4_1` Yes/No) and MCQ dependent (`4_9` A/B) split into their
    # respective DAFT task files. Both must be present for the retry flow to
    # be considered working: the dependent only appears when the gate answer
    # from pass 1 is carried into the retry-system prompt.
    bcq = json.loads((scene_dir / "task" / "bcq.json").read_text(encoding="utf-8"))
    bcq_questions = {it["question"] for it in bcq["items"]}
    assert bcq_questions == {"Gate"}

    mcq = json.loads((scene_dir / "task" / "mcq.json").read_text(encoding="utf-8"))
    assert len(mcq["items"]) == 1


def test_vlm_verify_parser_accepts_alias_fields_and_verdict_synonyms() -> None:
    verify_obj = {
        "items": [
            {
                "question_id": "1_1",
                "status": "incorrect",
                "evidence": "Collision is visible in middle frames.",
                "correct_answer": "Yes",
            }
        ]
    }
    fallback_mcq = [{"id": "1_1", "answer": "No", "options": ["Yes", "No"]}]
    items = parse_vlm_verify_items(verify_obj, fallback_mcq=fallback_mcq)
    assert len(items) == 1
    assert items[0]["id"] == "1_1"
    assert items[0]["verdict"] == "not_supported"
    assert items[0]["reasoning_trace"] == "Collision is visible in middle frames."
    assert items[0]["suggested_answer"] == "Yes"
    assert items[0]["current_answer"] == "No"
