# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from al_utils.schema.vlm_json import VlmJsonConfig
from vlm_json.runners.video_pipeline import (
    PipelineConfig,
    PromptFiles,
    VideoConfig,
    VideoPipeline,
    VLMConfig,
    _extract_two_json_objects,
)
from vlm_json.vlm_json_generator import VlmJsonGenerator

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "cookbooks" / "shared" / "prompts" / "vlm_json"


def _default_prompt_files() -> PromptFiles:
    return PromptFiles(
        video_json=_PROMPT_DIR / "video_json_prompt.md",
        events_json=_PROMPT_DIR / "video_events_prompt.md",
        image_json=_PROMPT_DIR / "image_caption_prompt.md",
    )


def _make_pipeline(
    *,
    split_json_calls: bool = True,
    prompt_files: Optional[PromptFiles] = None,
    rate_limit: float = 0.0,
) -> VideoPipeline:
    cfg = PipelineConfig(
        video_config=VideoConfig(),
        vlm_config=VLMConfig(base_url="http://test", model="test-model", retries=0),
        prompt_files=prompt_files or _default_prompt_files(),
        split_json_calls=split_json_calls,
        rate_limit=rate_limit,
    )
    pipeline = VideoPipeline(cfg, logging.getLogger("test_video_runner_fallback"))
    pipeline.preprocessor.extract_frames = lambda *args, **kwargs: (True, {"success": True, "frame_count": 1})
    pipeline.preprocessor.get_video_info = lambda path: {
        "width": 1280,
        "height": 720,
        "fps": 29.97,
        "nb_frames": 90,
        "duration": 3.0,
    }
    return pipeline


def _write_dummy_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-video")


def _fenced(payload: dict) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def test_vlm_json_config_defaults_to_split_calls() -> None:
    assert VlmJsonConfig().split_json_calls is True


def test_vlm_json_sampling_defaults_match_runner_defaults() -> None:
    cfg = VlmJsonConfig()
    runner_cfg = VLMConfig()

    assert cfg.frame_fps == runner_cfg.frame_fps == 1.0
    assert cfg.resolution == runner_cfg.resolution == 360
    assert cfg.max_frames == runner_cfg.max_frames == 24


def test_extract_two_json_objects_handles_nested_freeform_objects() -> None:
    video_obj = {"scene_description": "A {literal} brace appears.", "nested": {"ok": True}}
    events_obj = {"events": [{"event_id": "event_001", "metadata": {"severity": "low"}}]}
    text = f"first object: {json.dumps(video_obj)}\nsecond object: {json.dumps(events_obj)}"

    parsed_video, parsed_events = _extract_two_json_objects(text)

    assert parsed_video == video_obj
    assert parsed_events == events_obj


def test_video_vlm_call_failure_writes_daft_fallback_outputs(tmp_path: Path, caplog) -> None:
    video = tmp_path / "scene.mp4"
    _write_dummy_video(video)
    scene_dir = tmp_path / "scene_out"
    pipeline = _make_pipeline(split_json_calls=True)

    def _fail(*args, **kwargs):
        return False, "retry exhausted"

    pipeline.vlm_client.analyze_frames = _fail

    with caplog.at_level(logging.WARNING, logger="test_video_runner_fallback"):
        result = pipeline.process_video(video, scene_dir)

    assert result["success"] is False
    assert result["json_extraction"]["fallback_used"] is True

    video_payload = json.loads((scene_dir / "contextual" / "video.json").read_text(encoding="utf-8"))
    assert video_payload["metadata"]["type"] == "video"
    assert video_payload["video_id"] == "main"
    assert video_payload["format"] == "mp4"
    assert video_payload["fps"] == 30
    assert video_payload["duration"] == 3.0
    assert video_payload["height"] == 720
    assert video_payload["width"] == 1280
    assert "scene_description" not in video_payload
    assert "event_summary" not in video_payload

    events_payload = json.loads((scene_dir / "contextual" / "events.json").read_text(encoding="utf-8"))
    assert events_payload["metadata"]["type"] == "events"
    assert events_payload["video_id"] == "main"
    assert events_payload["events"] == []
    assert "VLM JSON fallback output written for video scene" in caplog.text


def test_warn_policy_style_vlm_failure_outputs_match_daft_schema(tmp_path: Path) -> None:
    """The default/warn fallback writes only fields accepted by DAFT schemas."""
    video = tmp_path / "scene.mp4"
    _write_dummy_video(video)
    scene_dir = tmp_path / "scene_out"
    pipeline = _make_pipeline(split_json_calls=True)
    pipeline.vlm_client.analyze_frames = lambda *args, **kwargs: (False, "retry exhausted")

    result = pipeline.process_video(video, scene_dir)

    assert result["success"] is False
    assert result["json_extraction"]["fallback_used"] is True

    video_payload = json.loads((scene_dir / "contextual" / "video.json").read_text(encoding="utf-8"))
    assert set(video_payload) == {
        "version",
        "video_id",
        "format",
        "fps",
        "duration",
        "height",
        "width",
        "metadata",
    }
    assert set(video_payload["metadata"]) == {"type", "date"}
    assert video_payload["metadata"]["type"] == "video"

    events_payload = json.loads((scene_dir / "contextual" / "events.json").read_text(encoding="utf-8"))
    assert set(events_payload) == {"version", "video_id", "events", "metadata"}
    assert set(events_payload["metadata"]) == {"type", "date"}
    assert events_payload["metadata"]["type"] == "events"
    assert events_payload["events"] == []


def test_split_mode_keeps_video_json_when_events_call_fails(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_dummy_video(video)
    scene_dir = tmp_path / "scene_out"
    pipeline = _make_pipeline(split_json_calls=True)
    calls: list[str] = []

    def _analyze_frames(frames_dir, prompt, output_dir, **kwargs):
        kind = kwargs["retry_stage"].split("_")[-1]
        calls.append(kind)
        if kind == "video":
            (output_dir / "output.txt").write_text(
                _fenced(
                    {
                        "scene_description": "A real model-generated scene description.",
                        "event_summary": "A real model-generated event summary.",
                    }
                ),
                encoding="utf-8",
            )
            return True, None
        return False, "events retry exhausted"

    pipeline.vlm_client.analyze_frames = _analyze_frames

    result = pipeline.process_video(video, scene_dir)

    assert calls == ["video", "events"]
    assert result["success"] is False
    assert result["vlm_analysis"]["success"] is False
    assert result["json_extraction"]["fallback_used"] is True

    video_payload = json.loads((scene_dir / "contextual" / "video.json").read_text(encoding="utf-8"))
    assert video_payload["scene_description"] == "A real model-generated scene description."
    assert video_payload["event_summary"] == "A real model-generated event summary."

    events_payload = json.loads((scene_dir / "contextual" / "events.json").read_text(encoding="utf-8"))
    assert events_payload["events"] == []


def test_video_prompt_overrides_are_used(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_dummy_video(video)
    scene_dir = tmp_path / "scene_out"
    video_prompt = tmp_path / "custom_video.md"
    video_prompt.write_text("CUSTOM VIDEO PROMPT", encoding="utf-8")
    pipeline = _make_pipeline(
        split_json_calls=True,
        prompt_files=PromptFiles(
            video_json=video_prompt,
            events_json=video_prompt,
            image_json=video_prompt,
        ),
    )
    captured: list[str] = []

    def _analyze_frames(frames_dir, prompt, output_dir, **kwargs):
        captured.append(prompt)
        kind = kwargs["retry_stage"].split("_")[-1]
        if kind == "video":
            (output_dir / "output.txt").write_text(
                _fenced({"scene_description": "Custom video scene.", "event_summary": "No visible motion."}),
                encoding="utf-8",
            )
        else:
            (output_dir / "output.txt").write_text(_fenced({"events": []}), encoding="utf-8")
        return True, None

    pipeline.vlm_client.analyze_frames = _analyze_frames

    result = pipeline.process_video(video, scene_dir)

    assert result["success"] is True
    assert "CUSTOM VIDEO PROMPT" in captured[0]
    assert "CUSTOM VIDEO PROMPT" in captured[1]


def test_video_non_split_writes_video_and_events_json(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_dummy_video(video)
    scene_dir = tmp_path / "scene_out"
    pipeline = _make_pipeline(split_json_calls=False)
    captured: list[str] = []

    def _analyze_frames(frames_dir, prompt, output_dir, **kwargs):
        captured.append(prompt)
        (output_dir / "output.txt").write_text(
            _fenced(
                {
                    "video_json": {
                        "scene_description": "A compact combined-call scene.",
                        "event_summary": "A single visible action happens.",
                    },
                    "events_json": {
                        "events": [
                            {
                                "event_id": "event_001",
                                "start_time": 0.0,
                                "end_time": 1.0,
                                "event_caption": "A single visible action happens.",
                                "instances": [],
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        return True, None

    pipeline.vlm_client.analyze_frames = _analyze_frames

    result = pipeline.process_video(video, scene_dir)

    assert result["success"] is True
    assert result["vlm_analysis"]["split_json_calls"] is False
    assert result["vlm_analysis"]["structured_output_effective"] == "openai"
    assert len(captured) == 1
    assert "Prompt for video.json" in captured[0]
    assert "Prompt for events.json" in captured[0]
    assert "video_json" in captured[0]
    assert "events_json" in captured[0]

    video_payload = json.loads((scene_dir / "contextual" / "video.json").read_text(encoding="utf-8"))
    assert video_payload["scene_description"] == "A compact combined-call scene."
    assert video_payload["event_summary"] == "A single visible action happens."

    events_payload = json.loads((scene_dir / "contextual" / "events.json").read_text(encoding="utf-8"))
    assert len(events_payload["events"]) == 1
    assert events_payload["events"][0]["event_caption"] == "A single visible action happens."


def test_split_mode_rate_limit_sleeps_between_video_vlm_calls(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_dummy_video(video)
    scene_dir = tmp_path / "scene_out"
    pipeline = _make_pipeline(split_json_calls=True, rate_limit=0.25)
    calls: list[str] = []
    sleeps: list[float] = []

    def _analyze_frames(frames_dir, prompt, output_dir, **kwargs):
        kind = kwargs["retry_stage"].split("_")[-1]
        calls.append(kind)
        if kind == "video":
            (output_dir / "output.txt").write_text(
                _fenced({"scene_description": "A scene.", "event_summary": "Movement is visible."}),
                encoding="utf-8",
            )
        else:
            (output_dir / "output.txt").write_text(
                _fenced(
                    {
                        "events": [
                            {
                                "event_id": "event_001",
                                "start_time": 0.0,
                                "end_time": 1.0,
                                "event_caption": "Movement is visible.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
        return True, None

    pipeline.vlm_client.analyze_frames = _analyze_frames

    with patch("vlm_json.runners.video_pipeline.time.sleep", side_effect=lambda seconds: sleeps.append(seconds)):
        result = pipeline.process_video(video, scene_dir)

    assert result["success"] is True
    assert calls == ["video", "events"]
    assert sleeps == [0.25]


def test_generator_returns_fallback_paths_when_result_is_failure(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_dummy_video(video)
    scene_dir = tmp_path / "scene_out"

    class _FakePipeline:
        def process_video(self, video_path: Path, output_dir: Path) -> dict:
            contextual = output_dir / "contextual"
            contextual.mkdir(parents=True, exist_ok=True)
            (contextual / "video.json").write_text("{}", encoding="utf-8")
            (contextual / "events.json").write_text("{}", encoding="utf-8")
            return {"success": False, "json_extraction": {"fallback_used": True}}

    gen = VlmJsonGenerator.__new__(VlmJsonGenerator)
    gen._pipeline = _FakePipeline()

    result = VlmJsonGenerator.generate(gen, video, scene_dir)

    assert result.success is False
    assert result.video_json == scene_dir / "contextual" / "video.json"
    assert result.events_json == scene_dir / "contextual" / "events.json"
