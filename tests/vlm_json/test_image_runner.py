# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for VideoPipeline.process_video on image inputs.

The image-input path was originally landed (MR 24, Mar 2026) without any test
that exercised the VLM call shape, so when the runner appended a one-line
"this is a still image" override to the video event-detection prompt the model
ignored it and fabricated an `events` block from the few-shot examples — which
resulted in `image.json` carrying no `caption`.

These tests pin down the post-fix behavior:
- The runner sends the dedicated image prompt (not the video prompt) to the VLM.
- A well-formed VLM response lands in `contextual/image.json` with `caption`.
- Image scenes do not run the events prompt or write `contextual/events.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

from PIL import Image
from vlm_json.runners.video_pipeline import (
    PipelineConfig,
    PromptFiles,
    VideoConfig,
    VideoPipeline,
    VLMConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "cookbooks" / "shared" / "prompts" / "vlm_json"


def _default_prompt_files() -> PromptFiles:
    return PromptFiles(
        video_json=_PROMPT_DIR / "video_json_prompt.md",
        events_json=_PROMPT_DIR / "video_events_prompt.md",
        image_json=_PROMPT_DIR / "image_caption_prompt.md",
    )


def _write_png(path: Path, *, width: int = 640, height: int = 360) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(40, 80, 120)).save(path)


def _make_pipeline(*, split_json_calls: bool = True, prompt_files: Optional[PromptFiles] = None) -> VideoPipeline:
    cfg = PipelineConfig(
        video_config=VideoConfig(),
        vlm_config=VLMConfig(base_url="http://test", model="test-model", retries=0),
        prompt_files=prompt_files or _default_prompt_files(),
        split_json_calls=split_json_calls,
    )
    return VideoPipeline(cfg, logging.getLogger("test_image_runner"))


def _vlm_response(payload: dict[str, Any]) -> str:
    """Wrap a JSON object in the fenced block the runner expects."""
    return "```json\n" + json.dumps(payload) + "\n```"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_image_input_writes_image_json_with_caption(tmp_path: Path) -> None:
    """Happy path: well-formed VLM response → image.json carries the caption."""
    img = tmp_path / "scene.png"
    _write_png(img, width=640, height=360)
    scene_dir = tmp_path / "scene_out"

    pipeline = _make_pipeline()

    response = _vlm_response(
        {
            "caption": "An empty four-way intersection with traffic signals at night.",
            "scenario_info": "urban intersection",
        }
    )

    with patch(
        "vlm_json.runners.video_pipeline.call_chat_object_with_structured_fallback",
        return_value=(None, response),
    ):
        result = pipeline.process_video(img, scene_dir)

    assert result["success"] is True
    assert result["vlm_analysis"]["split_json_calls"] is True
    assert result["json_extraction"]["scene_kind"] == "image"
    assert "events_json" not in result["json_extraction"]

    image_json_path = scene_dir / "contextual" / "image.json"
    assert image_json_path.exists(), "expected contextual/image.json to be written"
    events_path = scene_dir / "contextual" / "events.json"
    assert not events_path.exists(), "image inputs should not write contextual/events.json"

    payload = json.loads(image_json_path.read_text(encoding="utf-8"))
    assert payload["image_id"] == "main"
    assert payload["format"] == "png"
    assert payload["width"] == 640
    assert payload["height"] == 360
    assert payload["caption"] == "An empty four-way intersection with traffic signals at night."
    assert payload["scenario_info"] == "urban intersection"
    # Video-only fields must be absent on image scenes.
    assert "scene_description" not in payload
    assert "event_summary" not in payload
    assert "fps" not in payload
    assert "duration" not in payload


def test_image_input_uses_image_prompt_not_video_prompt(tmp_path: Path) -> None:
    """The runner must send the dedicated image prompt for image inputs.

    Regression guard for the original bug: appending a one-line override to the
    video event-detection prompt let the VLM regurgitate the prompt's few-shot
    event examples instead of producing a caption.
    """
    img = tmp_path / "scene.png"
    _write_png(img)
    scene_dir = tmp_path / "scene_out"

    pipeline = _make_pipeline()

    captured: list[str] = []

    def _capture(**kwargs: Any) -> tuple[None, str]:
        text_items = [c for c in kwargs["messages"][0]["content"] if c.get("type") == "text"]
        # The textual prompt is the last text item (frame timestamps come before).
        prompt = text_items[-1]["text"]
        captured.append(prompt)
        return (None, _vlm_response({"caption": "A static test image."}))

    with patch(
        "vlm_json.runners.video_pipeline.call_chat_object_with_structured_fallback",
        side_effect=_capture,
    ):
        pipeline.process_video(img, scene_dir)

    assert len(captured) == 1
    prompt = captured[0]
    # Hallmarks of the image-only prompt.
    assert "still image" in prompt.lower()
    assert "caption" in prompt.lower()
    # Hallmarks of the video event-analysis prompt that MUST NOT leak through.
    # (Note: `event_summary` is intentionally NOT asserted here because the
    # image prompt names it in a negative-list — "do NOT emit `event_summary`".
    # The markers below appear ONLY in the video prompt.)
    assert "RED-ID" not in prompt
    assert "tracking id" not in prompt.lower()


def test_image_input_does_not_write_events_json_file(tmp_path: Path) -> None:
    """Image scenes produce image.json only; events.json is video-only."""
    img = tmp_path / "scene.png"
    _write_png(img)
    scene_dir = tmp_path / "scene_out"

    pipeline = _make_pipeline()

    with patch(
        "vlm_json.runners.video_pipeline.call_chat_object_with_structured_fallback",
        return_value=(None, _vlm_response({"caption": "A static test image."})),
    ):
        pipeline.process_video(img, scene_dir)

    assert (scene_dir / "contextual" / "image.json").exists()
    assert not (scene_dir / "contextual" / "events.json").exists()
    assert not (scene_dir / "contextual" / "video.json").exists()


def test_image_input_caption_fallback_from_older_field(tmp_path: Path) -> None:
    """Defensive: if the VLM leaks the older video shape (scene_description /
    event_summary), `_enrich_image_with_probe` collapses it into `caption` so
    we still ship a non-empty caption rather than dropping the field on the
    floor.
    """
    img = tmp_path / "scene.png"
    _write_png(img)
    scene_dir = tmp_path / "scene_out"

    pipeline = _make_pipeline()

    older_response = _vlm_response(
        {
            "scene_description": "Older-shaped description from a confused model.",
            "event_summary": "Should not appear on the wire.",
        }
    )

    with patch(
        "vlm_json.runners.video_pipeline.call_chat_object_with_structured_fallback",
        return_value=(None, older_response),
    ):
        pipeline.process_video(img, scene_dir)

    payload = json.loads((scene_dir / "contextual" / "image.json").read_text(encoding="utf-8"))
    assert payload["caption"] == "Older-shaped description from a confused model."
    assert "scene_description" not in payload
    assert "event_summary" not in payload


def test_image_input_non_split_writes_image_json_only(tmp_path: Path) -> None:
    """Combined-call mode also treats image inputs as image.json-only."""
    img = tmp_path / "scene.png"
    _write_png(img)
    scene_dir = tmp_path / "scene_out"
    pipeline = _make_pipeline(split_json_calls=False)

    response = _vlm_response({"caption": "A quiet sidewalk scene."})

    with patch(
        "vlm_json.runners.video_pipeline.call_chat_object_with_structured_fallback",
        return_value=(None, response),
    ):
        result = pipeline.process_video(img, scene_dir)

    assert result["success"] is True
    assert (scene_dir / "contextual" / "image.json").exists()
    assert not (scene_dir / "contextual" / "events.json").exists()


def test_image_prompt_override_is_used(tmp_path: Path) -> None:
    """User-provided scene prompt overrides shipped image default."""
    img = tmp_path / "scene.png"
    _write_png(img)
    scene_dir = tmp_path / "scene_out"
    image_prompt = tmp_path / "custom_image.md"
    image_prompt.write_text("CUSTOM IMAGE PROMPT", encoding="utf-8")
    pipeline = _make_pipeline(
        prompt_files=PromptFiles(
            video_json=image_prompt,
            events_json=image_prompt,
            image_json=image_prompt,
        )
    )

    captured: list[str] = []

    def _capture(**kwargs: Any) -> tuple[None, str]:
        text_items = [c for c in kwargs["messages"][0]["content"] if c.get("type") == "text"]
        prompt = text_items[-1]["text"]
        captured.append(prompt)
        return (None, _vlm_response({"caption": "Custom prompt caption."}))

    with patch(
        "vlm_json.runners.video_pipeline.call_chat_object_with_structured_fallback",
        side_effect=_capture,
    ):
        pipeline.process_video(img, scene_dir)

    assert "CUSTOM IMAGE PROMPT" in captured[0]
    assert "RED-ID" not in captured[0]
    assert len(captured) == 1
