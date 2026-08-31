# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from core import DataEntry, read_json, read_pipeline_state, write_json
from core.exceptions import InvalidInputError
from core.model_clients import ChatRequest, MediaPayload
from grounding_2d.clients import GroundingEndpointError, create_endpoint_client
from grounding_2d.config import Grounding2DConfig
from grounding_2d.media import read_image_payload
from grounding_2d.prompts import expression_prompt
from grounding_2d.task import Grounding2DTask, _resolve_caption, _resolve_char_span
from PIL import Image
from pytest import MonkeyPatch, raises


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, request: ChatRequest) -> str:
        self.prompts.append(str(request.prompt))
        return self.responses.pop(0)


def test_grounding_task_vlm_expressions_then_sam3(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    image_path = _image(tmp_path / "image.jpg", size=(100, 50))
    scene_dir = tmp_path / "scene"
    write_json(scene_dir / "sidecars" / "input.json", {"caption": "A pedestrian."})
    client = _FakeClient(
        [
            '{"cleaned_caption":"A pedestrian.","expressions":[{"text":"pedestrian",'
            '"char_span":[2,12],"noun_chunk":"pedestrian"}]}',
        ]
    )

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeDetectionTask:
        def __init__(self, *, config: object, name: str) -> None:
            self.config = config
            self.name = name

        def run(self, data_entry: DataEntry) -> DataEntry:
            write_json(
                Path(data_entry.data_path) / "contextual" / "objects.json",
                {
                    "frames": [
                        {
                            "instances": [
                                {
                                    "object_id": "pedestrian_0_1",
                                    "prompt": "pedestrian",
                                    "bounding_box_2d_tight": [1, 2, 30, 40],
                                    "contours_xy": [[1, 2, 30, 2, 30, 40]],
                                    "score": 0.91,
                                }
                            ]
                        }
                    ]
                },
            )
            return data_entry

    monkeypatch.setattr("grounding_2d.task.DetectionAndTrackingConfig", FakeConfig)
    monkeypatch.setattr("grounding_2d.task.DetectionAndTrackingTask", FakeDetectionTask)
    task = Grounding2DTask(
        config=Grounding2DConfig(vlm_endpoint_url="http://unused"),
        client=client,
    )

    task.run(DataEntry(id="entry", media_path=str(image_path), data_path=str(scene_dir)))

    artifact = read_json(scene_dir / "sidecars" / "grounding_2d" / "grounding_2d.json")
    instance = artifact["expressions"][0]["instances"][0]
    assert instance["source"] == "sam3"
    assert instance["bbox"] == [1, 2, 30, 40]
    assert instance["bbox_score"] == 0.91
    assert instance["mask_score"] == 0.91
    assert instance["area"] == 29 * 38
    assert instance["segmentation"]["format"] == "contours_xy"
    state = read_pipeline_state(scene_dir)
    assert state.task_artifacts["grounding_2d"]["success"] is True
    assert state.task_artifacts["grounding_2d"]["expression_count"] == 1
    assert state.task_artifacts["grounding_2d"]["instance_count"] == 1
    assert len(client.prompts) == 1


def test_grounding_task_skips_ungroundable_and_low_score_instances(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    image_path = _image(tmp_path / "image.jpg", size=(200, 100))
    scene_dir = tmp_path / "scene"
    write_json(
        scene_dir / "sidecars" / "input.json",
        {"caption": "Cars during a collision event."},
    )
    client = _FakeClient(
        [
            '{"cleaned_caption":"Cars during a collision event.",'
            '"expressions":['
            '{"text":"Cars","char_span":[0,4],"noun_chunk":"Cars","groundable":true},'
            '{"text":"collision event","char_span":[13,28],"noun_chunk":"event","groundable":false}'
            "]}",
        ]
    )
    captured_configs: list[dict[str, object]] = []

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            prompts = kwargs.get("sam3_prompts", ())
            if not isinstance(prompts, (list, tuple)):
                prompts = ()
            captured_configs.append(
                {
                    "prompts": tuple(str(prompt) for prompt in prompts),
                    "sam3_version": kwargs.get("sam3_version"),
                    "sam3_runtime": kwargs.get("sam3_runtime"),
                }
            )

    class FakeDetectionTask:
        def __init__(self, *, config: object, name: str) -> None:
            self.config = config

        def run(self, data_entry: DataEntry) -> DataEntry:
            write_json(
                Path(data_entry.data_path) / "contextual" / "objects.json",
                {
                    "frames": [
                        {
                            "instances": [
                                {
                                    "object_id": "cars_hi",
                                    "prompt": "Cars",
                                    "bounding_box_2d_tight": [10, 10, 50, 50],
                                    "contours_xy": [[10, 10, 50, 10, 50, 50]],
                                    "score": 0.88,
                                },
                                {
                                    "object_id": "cars_lo",
                                    "prompt": "Cars",
                                    "bounding_box_2d_tight": [60, 10, 90, 40],
                                    "contours_xy": [[60, 10, 90, 10, 90, 40]],
                                    "score": 0.2,
                                },
                                {
                                    "object_id": "cars_tiny",
                                    "prompt": "Cars",
                                    "bounding_box_2d_tight": [100, 10, 105, 15],
                                    "contours_xy": [[100, 10, 105, 10, 105, 15]],
                                    "score": 0.95,
                                },
                            ]
                        }
                    ]
                },
            )
            return data_entry

    monkeypatch.setattr("grounding_2d.task.DetectionAndTrackingConfig", FakeConfig)
    monkeypatch.setattr("grounding_2d.task.DetectionAndTrackingTask", FakeDetectionTask)
    task = Grounding2DTask(
        config=Grounding2DConfig(
            vlm_endpoint_url="http://unused",
            sam3_version="sam3.1",
            sam3_runtime="native",
            min_instance_score=0.5,
            min_bbox_area=64,
        ),
        client=client,
    )
    task.run(DataEntry(id="entry", media_path=str(image_path), data_path=str(scene_dir)))

    assert captured_configs == [
        {"prompts": ("Cars",), "sam3_version": "sam3.1", "sam3_runtime": "native"}
    ]
    artifact = read_json(scene_dir / "sidecars" / "grounding_2d" / "grounding_2d.json")
    by_text = {expr["text"]: expr for expr in artifact["expressions"]}
    assert by_text["collision event"]["instances"] == []
    assert len(by_text["Cars"]["instances"]) == 1
    assert by_text["Cars"]["instances"][0]["mask_id"] == "cars_hi"
    assert by_text["Cars"]["instances"][0]["bbox_score"] == 0.88


def test_create_endpoint_client_uses_core_factory(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_factory(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("grounding_2d.clients._create_endpoint_client", fake_factory)
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
        "error_cls": GroundingEndpointError,
    }


def test_read_image_payload_builds_core_media_payload(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "frame.jpg", size=(20, 10))
    payload = read_image_payload(image_path)
    assert isinstance(payload, MediaPayload)
    assert payload.filename == "frame.jpg"
    assert payload.mime_type == "image/jpeg"
    assert payload.data_base64


def test_read_image_payload_rejects_unknown_mime(tmp_path: Path) -> None:
    path = tmp_path / "frame.bin"
    path.write_bytes(b"not-an-image")
    with raises(InvalidInputError, match="MIME type"):
        read_image_payload(path)


def test_expression_prompt_escapes_quotes_and_newlines() -> None:
    prompt = expression_prompt('A "quoted"\ncaption')
    assert 'Caption: "A \\"quoted\\"\\ncaption"' in prompt


def test_resolve_char_span_derives_or_leaves_unset() -> None:
    caption = "A pedestrian near the curb."
    assert _resolve_char_span([2, 12], text="pedestrian", cleaned_caption=caption) == [2, 12]
    assert _resolve_char_span(None, text="pedestrian", cleaned_caption=caption) == [2, 12]
    assert _resolve_char_span([0, 10], text="pedestrian", cleaned_caption=caption) == [2, 12]
    assert _resolve_char_span(None, text="missing", cleaned_caption=caption) is None
    repeated = "A car beside another car."
    assert _resolve_char_span(None, text="car", cleaned_caption=repeated) is None
    assert _resolve_char_span([2, 5], text="car", cleaned_caption=repeated) == [2, 5]
    assert _resolve_char_span([0, 3], text="car", cleaned_caption=repeated) is None


def test_resolve_caption_prefers_config_then_input_then_captioning(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene"
    entry = DataEntry(id="e", media_path="x.jpg", data_path=str(scene))
    write_json(
        scene / "sidecars" / "captioning" / "image_captions.json",
        {"caption": "from captioning"},
    )
    write_json(scene / "sidecars" / "input.json", {"caption": "from input"})

    assert _resolve_caption(entry, Grounding2DConfig(caption="from config")) == "from config"
    assert _resolve_caption(entry, Grounding2DConfig()) == "from input"

    (scene / "sidecars" / "input.json").unlink()
    assert _resolve_caption(entry, Grounding2DConfig()) == "from captioning"


def test_resolve_caption_reads_captioning_model_output(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    entry = DataEntry(id="e", media_path="x.jpg", data_path=str(scene))
    write_json(
        scene / "contextual" / "image_captions.json",
        {"model_output": {"caption": "nested caption"}},
    )
    assert _resolve_caption(entry, Grounding2DConfig()) == "nested caption"


def _image(path: Path, *, size: tuple[int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)
    return path
