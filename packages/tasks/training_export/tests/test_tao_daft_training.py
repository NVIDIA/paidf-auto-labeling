# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.formats.daft import daft_envelope
from core.scene import SceneContext, ensure_scene_skeleton
from training_export.formats import (
    COSMOS_REASON_VERSION,
    DEFAULT_TAO_VL_REASON_LICENSE,
    TAO_VL_REASON_FORMAT,
    build_cosmos_reason_conversation,
    build_tao_vl_reason_annotation,
    convert_metropolis_dataset_to_cosmos_reason,
    convert_metropolis_dataset_to_tao_vl_reason,
    convert_metropolis_scene_to_cosmos_reason,
    convert_metropolis_scene_to_tao_vl_reason,
)


def test_tao_vl_reason_annotation_builder_defaults_required_license() -> None:
    payload = build_tao_vl_reason_annotation(
        "open_qa",
        [{"video_id": "videos/clip.mp4", "question": "What happened?", "answer": "A crash."}],
    )

    assert payload["format"] == TAO_VL_REASON_FORMAT
    assert payload["metadata"]["type"] == "annotation"
    assert payload["metadata"]["task"] == "open_qa"
    assert payload["metadata"]["license"] == DEFAULT_TAO_VL_REASON_LICENSE
    assert payload["media_root"] is None


def test_cosmos_reason_conversation_adds_media_placeholder_options_and_reasoning() -> None:
    conversation = build_cosmos_reason_conversation(
        "mcq",
        {
            "video_id": "clip",
            "question": "Which object moves?",
            "options": {"A": "Car", "B": "Bike"},
            "answer": "A",
            "reasoning": "The car changes position.",
        },
        is_image=False,
    )

    assert conversation["version"] == COSMOS_REASON_VERSION
    user_content = conversation["conversations"][0]["content"]
    assert user_content[0] == {"type": "video", "video": "video_0"}
    assert "A) Car" in user_content[1]["text"]
    assert "Choose the correct option by letter, optionally followed" in user_content[1]["text"]
    assistant = conversation["conversations"][1]
    assert assistant["content"] == [{"type": "text", "text": "A) Car"}]
    assert assistant["reasoning_content"] == [{"type": "text", "text": "The car changes position."}]


def test_convert_metropolis_scene_to_tao_vl_reason_writes_grouped_annotation(
    tmp_path: Path,
) -> None:
    scene = _scene_with_mcq(tmp_path / "scene")
    output = tmp_path / "tao-vl"

    result = convert_metropolis_scene_to_tao_vl_reason(
        scene,
        output,
        metadata={"description": "training split", "license": "internal"},
    )

    assert result.is_success()
    assert result.samples_written == 1
    annotation = _read_json(output / "mcq.json")
    assert annotation["format"] == TAO_VL_REASON_FORMAT
    assert annotation["metadata"]["description"] == "training split"
    assert annotation["metadata"]["license"] == "internal"
    assert annotation["items"][0]["video_id"] == "videos/raw--clip.mp4"
    assert annotation["items"][0]["question"].endswith(
        "Choose the correct option by letter, optionally followed by the option label "
        "or a brief explanation."
    )
    assert annotation["items"][0]["answer"] == "A) Car"
    assert (output / "videos" / "raw--clip.mp4").read_bytes() == b"video"


def test_convert_resolves_single_extension_raw_media(tmp_path: Path) -> None:
    # Production scenes are named "<stem>.<ext>", so media_id carries the
    # extension and the analyzed media is staged as raw/<media_id> with a single
    # extension. The exporter must resolve it (regression: raw/<id>.<ext> ->
    # <id>.mp4.mp4 doubling left raw media unresolved).
    scene = tmp_path / "clip.mp4"
    paths = ensure_scene_skeleton(scene)
    ctx = SceneContext(media_id="clip.mp4")
    (paths.raw_dir / "clip.mp4").write_bytes(b"video")
    video_payload = daft_envelope("video", ctx)
    video_payload.update(
        {"format": "mp4", "fps": 30, "duration": 1.0, "height": 720, "width": 1280}
    )
    _write_json(paths.contextual_dir / "video.json", video_payload)
    payload = daft_envelope("mcq", ctx, include_scene_id=False)
    payload["items"] = [
        {
            "video_id": "clip.mp4",
            "question": "Which object moves?",
            "options": {"A": "Car", "B": "Bike"},
            "answer": "A",
        }
    ]
    _write_json(paths.task_dir / "mcq.json", payload)

    output = tmp_path / "tao-vl"
    result = convert_metropolis_scene_to_tao_vl_reason(scene, output)

    assert result.is_success()
    assert result.samples_written == 1
    annotation = _read_json(output / "mcq.json")
    assert annotation["items"][0]["video_id"] == "videos/raw--clip.mp4"
    assert (output / "videos" / "raw--clip.mp4").read_bytes() == b"video"


def test_convert_metropolis_scene_to_cosmos_reason_writes_meta_and_conversation(
    tmp_path: Path,
) -> None:
    scene = _scene_with_mcq(tmp_path / "scene")
    output = tmp_path / "cosmos"

    result = convert_metropolis_scene_to_cosmos_reason(
        scene,
        output,
        metadata={"description": "cosmos split"},
    )

    assert result.is_success()
    meta = _read_json(output / "meta.json")
    assert meta["version"] == COSMOS_REASON_VERSION
    assert meta["metadata"]["description"] == "cosmos split"
    assert meta["samples"] == [
        {
            "id": "scene__mcq__mcq__clip__0000",
            "conversation": "text/scene__mcq__mcq__clip__0000.json",
            "media": "media/raw--clip.mp4",
        }
    ]
    conversation = _read_json(output / meta["samples"][0]["conversation"])
    assert conversation["metadata"]["tags"] == ["mcq"]
    assert conversation["conversations"][0]["content"][0] == {"type": "video", "video": "video_0"}
    assert (output / "media" / "raw--clip.mp4").read_bytes() == b"video"


def test_convert_metropolis_dataset_to_cosmos_reason_aggregates_scenes(tmp_path: Path) -> None:
    _scene_with_mcq(tmp_path / "dataset" / "scene-a", media_id="clip_a")
    _scene_with_mcq(tmp_path / "dataset" / "nested" / "scene-b", media_id="clip_b")
    output = tmp_path / "cosmos"

    result = convert_metropolis_dataset_to_cosmos_reason(tmp_path / "dataset", output)

    assert result.is_success()
    meta = _read_json(output / "meta.json")
    assert sorted(sample["id"] for sample in meta["samples"]) == [
        "scene-a__mcq__mcq__clip_a__0000",
        "scene-b__mcq__mcq__clip_b__0000",
    ]
    assert sorted(path.name for path in (output / "media").iterdir()) == [
        "nested--scene-b--raw--clip_b.mp4",
        "scene-a--raw--clip_a.mp4",
    ]


def test_convert_metropolis_dataset_to_tao_vl_reason_aggregates_scenes(tmp_path: Path) -> None:
    _scene_with_mcq(tmp_path / "dataset" / "scene-a", media_id="clip_a")
    _scene_with_mcq(tmp_path / "dataset" / "nested" / "scene-b", media_id="clip_b")
    output = tmp_path / "tao-vl"

    result = convert_metropolis_dataset_to_tao_vl_reason(tmp_path / "dataset", output)

    assert result.is_success()
    annotation = _read_json(output / "mcq.json")
    assert annotation["format"] == TAO_VL_REASON_FORMAT
    assert annotation["metadata"]["task"] == "mcq"
    assert sorted(item["video_id"] for item in annotation["items"]) == [
        "videos/nested--scene-b--raw--clip_b.mp4",
        "videos/scene-a--raw--clip_a.mp4",
    ]
    assert sorted(path.name for path in (output / "videos").iterdir()) == [
        "nested--scene-b--raw--clip_b.mp4",
        "scene-a--raw--clip_a.mp4",
    ]


def test_cosmos_reason_sample_ids_include_task_file_stem(tmp_path: Path) -> None:
    scene = _scene_with_mcq(tmp_path / "scene")
    paths = ensure_scene_skeleton(scene)
    alternate = _read_json(paths.task_dir / "mcq.json")
    _write_json(paths.task_dir / "mcq_extra.json", alternate)

    result = convert_metropolis_scene_to_cosmos_reason(scene, tmp_path / "cosmos")

    assert result.is_success()
    meta = _read_json(tmp_path / "cosmos" / "meta.json")
    assert sorted(sample["id"] for sample in meta["samples"]) == [
        "scene__mcq__mcq__clip__0000",
        "scene__mcq__mcq_extra__clip__0000",
    ]


def test_training_export_rejects_traversal_media_id(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    ctx = SceneContext(media_id="clip")
    (paths.raw_dir / "clip.mp4").write_bytes(b"video")
    video_payload = daft_envelope("video", ctx)
    video_payload.update(
        {"format": "mp4", "fps": 30, "duration": 1.0, "height": 720, "width": 1280}
    )
    video_payload["video_id"] = "../clip"
    _write_json(paths.contextual_dir / "video.json", video_payload)
    task_payload = daft_envelope("mcq", ctx, include_scene_id=False)
    task_payload["items"] = [
        {
            "video_id": "../clip",
            "question": "Which object moves?",
            "options": {"A": "Car", "B": "Bike"},
            "answer": "A",
        }
    ]
    _write_json(paths.task_dir / "mcq.json", task_payload)

    result = convert_metropolis_scene_to_cosmos_reason(paths.scene_dir, tmp_path / "cosmos")

    assert not result.is_success()
    assert result.samples_skipped == 1
    assert not (tmp_path / "cosmos" / "meta.json").exists()


def test_convert_metropolis_scene_infers_media_format_from_raw_file(
    tmp_path: Path,
) -> None:
    scene = _scene_with_mcq(tmp_path / "scene", include_contextual=False)
    output = tmp_path / "cosmos"

    result = convert_metropolis_scene_to_cosmos_reason(scene, output)

    assert result.is_success()
    assert result.samples_written == 1
    meta = _read_json(output / "meta.json")
    assert meta["samples"][0]["media"] == "media/raw--clip.mp4"


def test_convert_metropolis_scene_accepts_raw_symlink_to_scene_sidecar(
    tmp_path: Path,
) -> None:
    scene = _scene_with_mcq(tmp_path / "scene", include_contextual=False)
    raw = scene / "raw" / "clip.mp4"
    sidecar = scene / "sidecars" / "active.mp4"
    sidecar.write_bytes(b"active")
    raw.unlink()
    raw.symlink_to("../sidecars/active.mp4")
    output = tmp_path / "cosmos"

    result = convert_metropolis_scene_to_cosmos_reason(scene, output)

    assert result.is_success()
    assert (output / "media" / "raw--clip.mp4").read_bytes() == b"active"


def _scene_with_mcq(
    scene: Path,
    *,
    media_id: str = "clip",
    include_contextual: bool = True,
) -> Path:
    paths = ensure_scene_skeleton(scene)
    ctx = SceneContext(media_id=media_id)
    raw = paths.raw_dir / f"{media_id}.mp4"
    raw.write_bytes(b"video")
    if include_contextual:
        video_payload = daft_envelope("video", ctx)
        video_payload.update(
            {
                "format": "mp4",
                "fps": 30,
                "duration": 1.0,
                "height": 720,
                "width": 1280,
            }
        )
        _write_json(paths.contextual_dir / "video.json", video_payload)
    payload = daft_envelope("mcq", ctx, include_scene_id=False)
    payload["items"] = [
        {
            "video_id": media_id,
            "question": "Which object moves?",
            "options": {"A": "Car", "B": "Bike"},
            "answer": "A",
        }
    ]
    _write_json(paths.task_dir / "mcq.json", payload)
    return scene


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
