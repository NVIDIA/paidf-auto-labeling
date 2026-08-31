# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from core import DataEntry, read_json, read_pipeline_state, write_json
from core.exceptions import InvalidInputError
from core.model_clients import ChatRequest, MediaPayload
from PIL import Image
from pytest import MonkeyPatch, raises
from referring_expressions.clients import ReferringEndpointError, create_endpoint_client
from referring_expressions.config import ReferringExpressionsConfig
from referring_expressions.media import read_image_payload
from referring_expressions.overlay import draw_marked_boxes
from referring_expressions.parsing import parse_json_array
from referring_expressions.prompts import (
    greedy_match_by_iou,
    match_region_to_candidate,
    region_expression_prompt,
)
from referring_expressions.task import ReferringExpressionsTask
from referring_expressions.types import normalize_object_type


class _FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.image_names: list[str] = []

    def generate(self, request: ChatRequest) -> str:
        self.prompts.append(str(request.prompt))
        if request.media:
            self.image_names.append(str(request.media[0].filename or ""))
        return self.response


def test_parse_json_array_strips_fences() -> None:
    raw = (
        '```json\n[{"mark":1,"bbox_2d":[1,2,3,4],"type":"car","color":"red",'
        '"description":"A red car"}]\n```'
    )
    parsed = parse_json_array(raw)
    assert parsed[0]["type"] == "car"


def test_parse_json_array_salvages_when_slice_is_malformed() -> None:
    # Trailing junk after a broken slice should not abort before salvage.
    raw = (
        '[{"mark":1,"type":"car","color":"red","description":"A red car"},'
        '{"mark":2,"type":"truck","color":"blue","description":"A blue truck"'
        "] trailing"
    )
    parsed = parse_json_array(raw)
    assert len(parsed) == 1
    assert parsed[0]["type"] == "car"


def test_region_prompt_includes_marks_without_closed_type_enum() -> None:
    prompt = region_expression_prompt([{"bbox": [10, 20, 30, 40], "mark": 1}])
    assert "[1] [10, 20, 30, 40]" in prompt
    assert "no closed enum" in prompt
    assert "MUST be one of" not in prompt
    assert '"mark": 1' in prompt


def test_normalize_object_type_is_open_vocabulary() -> None:
    assert normalize_object_type("Semi-Truck") == "truck"
    assert normalize_object_type("unknown") == "other"
    assert normalize_object_type("sedan") == "sedan"
    assert normalize_object_type("Red Chair") == "red_chair"
    assert normalize_object_type("traffic light") == "traffic_light"


def test_greedy_iou_prefers_best_unique_match() -> None:
    predicted = [
        {"bbox": [0, 0, 10, 10], "description": "a"},
        {"bbox": [100, 100, 120, 120], "description": "b"},
    ]
    candidates = [
        {"object_id": "a1", "bbox": [1, 1, 9, 9]},
        {"object_id": "b1", "bbox": [101, 101, 119, 119]},
    ]
    pairs = greedy_match_by_iou(predicted=predicted, candidates=candidates, min_iou=0.3)
    matched_a = pairs[0][1]
    matched_b = pairs[1][1]
    assert matched_a is not None
    assert matched_b is not None
    assert matched_a["object_id"] == "a1"
    assert matched_b["object_id"] == "b1"


def test_draw_marked_boxes_writes_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (80, 60), color=(20, 20, 20)).save(image_path)
    out = tmp_path / "marked.jpg"
    draw_marked_boxes(
        image_path,
        [{"bbox": [10, 10, 40, 40], "object_id": "x"}],
        out,
    )
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (80, 60)


def test_referring_task_uses_mark_ids_and_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), color=(255, 255, 255)).save(image_path)
    scene_dir = tmp_path / "scene"
    write_json(
        scene_dir / "contextual" / "objects.json",
        {
            "frames": [
                {
                    "frame_number": 0,
                    "instances": [
                        {
                            "object_id": "car_0",
                            "bounding_box_2d_tight": [10, 10, 40, 40],
                        }
                    ],
                }
            ]
        },
    )
    client = _FakeClient(
        '[{"mark":1,"bbox_2d":[12,12,38,38],"type":"sedan","color":"white",'
        '"description":"The white sedan in the center"}]'
    )
    task = ReferringExpressionsTask(
        config=ReferringExpressionsConfig(
            vlm_endpoint_url="http://unused",
            draw_box_overlay=True,
        ),
        client=client,
    )
    task.run(DataEntry(id="e1", media_path=str(image_path), data_path=str(scene_dir)))

    artifact = read_json(
        scene_dir / "sidecars" / "referring_expressions" / "referring_expressions.json"
    )
    region = artifact["regions"][0]
    assert region["object_id"] == "car_0"
    assert region["type"] == "sedan"
    assert region["bbox"] == [10, 10, 40, 40]  # DAFT box kept
    assert region["mark"] == 1
    assert Path(artifact["overlay_path"]).exists()
    assert "marked_boxes.jpg" in client.image_names[0]
    state = read_pipeline_state(scene_dir)
    assert state.task_artifacts["referring_expressions"]["region_count"] == 1


def test_create_endpoint_client_uses_core_factory(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_factory(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("referring_expressions.clients._create_endpoint_client", fake_factory)
    create_endpoint_client(
        provider="openai-compatible",
        endpoint_url="http://endpoint/v1",
        model="model-name",
        timeout_s=12.0,
        retries=3,
        retry_backoff_s=0.5,
    )
    assert captured == {
        "provider": "openai-compatible",
        "endpoint_url": "http://endpoint/v1",
        "model": "model-name",
        "timeout_s": 12.0,
        "retries": 3,
        "retry_backoff_s": 0.5,
        "error_cls": ReferringEndpointError,
    }


def test_read_image_payload_builds_core_media_payload(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (20, 10), color=(255, 255, 255)).save(image_path)
    payload = read_image_payload(image_path)
    assert isinstance(payload, MediaPayload)
    assert payload.filename == "frame.jpg"
    assert payload.mime_type == "image/jpeg"
    assert payload.data_base64


def test_match_region_uses_iou_not_exact_equality() -> None:
    candidates = [{"object_id": "truck_1", "bbox": [0, 385, 332, 623]}]
    matched = match_region_to_candidate(
        bbox=[10, 400, 320, 610],
        candidates=candidates,
        min_iou=0.3,
    )
    assert matched is not None
    assert matched["object_id"] == "truck_1"


def test_referring_task_fails_when_frame_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (40, 40), color=(255, 255, 255)).save(image_path)
    scene_dir = tmp_path / "scene"
    write_json(
        scene_dir / "contextual" / "objects.json",
        {"frames": [{"frame_number": 0, "instances": []}]},
    )
    task = ReferringExpressionsTask(
        config=ReferringExpressionsConfig(
            vlm_endpoint_url="http://unused",
            frame_number=3,
        ),
        client=_FakeClient("[]"),
    )
    with raises(InvalidInputError, match="no frame_number=3"):
        task.run(DataEntry(id="e1", media_path=str(image_path), data_path=str(scene_dir)))


def test_referring_task_empty_frame_is_successful(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (40, 40), color=(255, 255, 255)).save(image_path)
    scene_dir = tmp_path / "scene"
    write_json(
        scene_dir / "contextual" / "objects.json",
        {"frames": [{"frame_number": 0, "instances": []}]},
    )
    task = ReferringExpressionsTask(
        config=ReferringExpressionsConfig(vlm_endpoint_url="http://unused"),
        client=_FakeClient("[]"),
    )
    task.run(DataEntry(id="e1", media_path=str(image_path), data_path=str(scene_dir)))
    artifact = read_json(
        scene_dir / "sidecars" / "referring_expressions" / "referring_expressions.json"
    )
    assert artifact["regions"] == []
    state = read_pipeline_state(scene_dir)
    assert state.task_artifacts["referring_expressions"]["success"] is True


def test_referring_task_fails_when_regions_incomplete(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), color=(255, 255, 255)).save(image_path)
    scene_dir = tmp_path / "scene"
    write_json(
        scene_dir / "contextual" / "objects.json",
        {
            "frames": [
                {
                    "frame_number": 0,
                    "instances": [
                        {
                            "object_id": "car_0",
                            "bounding_box_2d_tight": [10, 10, 40, 40],
                        },
                        {
                            "object_id": "car_1",
                            "bounding_box_2d_tight": [50, 10, 80, 40],
                        },
                    ],
                }
            ]
        },
    )
    task = ReferringExpressionsTask(
        config=ReferringExpressionsConfig(vlm_endpoint_url="http://unused"),
        client=_FakeClient(
            '[{"mark":1,"bbox_2d":[12,12,38,38],"type":"sedan","color":"white",'
            '"description":"only one matched"}]'
        ),
    )
    with raises(InvalidInputError, match="matched 1/2"):
        task.run(DataEntry(id="e1", media_path=str(image_path), data_path=str(scene_dir)))
    state = read_pipeline_state(scene_dir)
    assert state.task_artifacts["referring_expressions"]["success"] is False

    final_path = scene_dir / "sidecars" / "referring_expressions" / "referring_expressions.json"
    assert final_path.exists()
    cache_client = _FakeClient(
        '[{"mark":1,"bbox_2d":[12,12,38,38],"type":"sedan","color":"white",'
        '"description":"first matched"},'
        '{"mark":2,"bbox_2d":[52,12,78,38],"type":"sedan","color":"white",'
        '"description":"second matched"}]'
    )
    cached_task = ReferringExpressionsTask(
        config=ReferringExpressionsConfig(vlm_endpoint_url="http://unused"),
        client=cache_client,
    )
    cached_task.run(DataEntry(id="e1", media_path=str(image_path), data_path=str(scene_dir)))
    assert len(cache_client.prompts) == 1
    artifact = read_json(final_path)
    assert len(artifact["regions"]) == 2
    state = read_pipeline_state(scene_dir)
    assert state.task_artifacts["referring_expressions"]["success"] is True
    assert state.task_artifacts["referring_expressions"]["region_count"] == 2
