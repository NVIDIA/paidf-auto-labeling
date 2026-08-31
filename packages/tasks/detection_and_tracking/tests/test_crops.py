# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for per-track crop planning and extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from core import ensure_scene_skeleton, read_json, write_json
from detection_and_tracking import crops as crops_module
from detection_and_tracking.config import DetectionAndTrackingConfig, pas_tracking_config
from detection_and_tracking.crops import (
    extract_track_crops,
    pad_and_clamp_box,
    plan_track_crops,
    sample_frame_numbers,
    write_crops,
)


def test_pas_tracking_config_presets_person_crops() -> None:
    config = pas_tracking_config()
    assert config.tracker == "sam3"
    assert config.sam3_prompts == ("person",)
    assert config.extract_crops is True
    assert config.crop_classes == ("person",)


def test_pas_tracking_config_allows_overrides() -> None:
    config = pas_tracking_config(crops_per_track=4, crop_padding=0.0)
    assert config.crops_per_track == 4
    assert config.crop_padding == 0.0
    assert config.extract_crops is True


def test_sample_frame_numbers_even_spacing() -> None:
    assert sample_frame_numbers([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3) == [0, 4, 9]
    assert sample_frame_numbers([5, 6, 7], 8) == [5, 6, 7]
    assert sample_frame_numbers([], 4) == []
    assert sample_frame_numbers([2, 4, 6], 1) == [4]


def test_pad_and_clamp_box_expands_and_clamps() -> None:
    assert pad_and_clamp_box([10, 10, 20, 20], width=100, height=100, padding=0.0) == (
        10,
        10,
        20,
        20,
    )
    # 100% padding doubles the box, clamped to the frame edge at 0.
    assert pad_and_clamp_box([10, 10, 20, 20], width=100, height=100, padding=1.0) == (
        5,
        5,
        25,
        25,
    )


def test_pad_and_clamp_box_rejects_degenerate() -> None:
    assert pad_and_clamp_box([20, 20, 10, 10], width=100, height=100, padding=0.0) is None
    assert pad_and_clamp_box([0, 0, 0, 0], width=100, height=100, padding=0.0) is None


def _objects_payload() -> dict[str, Any]:
    def frame(n: int, box: list[int]) -> dict[str, Any]:
        return {
            "frame_number": n,
            "width": 200,
            "height": 200,
            "instances": [{"object_id": "person_0_2", "bounding_box_2d_tight": box}],
        }

    return {"frames": [frame(n, [10, 10, 50, 90]) for n in range(8)]}


def _instances_payload() -> dict[str, Any]:
    return {
        "instances": [
            {
                "object_id": "person_0_2",
                "object_type": "person",
                "track_id": 2,
                "first_frame": 0,
                "last_frame": 7,
                "start_time_s": 0.0,
                "end_time_s": 5.667,
                "detection_score": 0.8552,
            }
        ]
    }


def test_plan_track_crops_basic() -> None:
    plans = plan_track_crops(
        _objects_payload(),
        _instances_payload(),
        crop_classes=("person",),
        crops_per_track=4,
        crop_padding=0.0,
        min_crop_size=0,
    )
    assert len(plans) == 1
    plan = plans[0]
    assert plan.track_id == 2
    assert plan.detection_score == 0.8552
    assert plan.duration_sec == 5.667
    assert len(plan.frames) == 4
    assert plan.frames[0].box == (10, 10, 50, 90)


def test_plan_track_crops_filters_class() -> None:
    plans = plan_track_crops(
        _objects_payload(),
        _instances_payload(),
        crop_classes=("forklift",),
    )
    assert plans == []


def test_plan_track_crops_filters_min_size() -> None:
    plans = plan_track_crops(
        _objects_payload(),
        _instances_payload(),
        crop_classes=("person",),
        min_crop_size=1000,
    )
    assert plans == []


def test_plan_track_crops_filters_min_detection_score() -> None:
    # Fixture track detection_score is 0.8552.
    assert (
        plan_track_crops(
            _objects_payload(),
            _instances_payload(),
            crop_classes=("person",),
            min_detection_score=0.9,
        )
        == []
    )
    kept = plan_track_crops(
        _objects_payload(),
        _instances_payload(),
        crop_classes=("person",),
        min_detection_score=0.5,
    )
    assert len(kept) == 1


def test_plan_track_crops_filters_min_track_seconds() -> None:
    # Fixture track spans start_time_s=0.0 .. end_time_s=5.667 (duration 5.667s).
    assert (
        plan_track_crops(
            _objects_payload(),
            _instances_payload(),
            crop_classes=("person",),
            min_track_seconds=6.0,
        )
        == []
    )
    kept = plan_track_crops(
        _objects_payload(),
        _instances_payload(),
        crop_classes=("person",),
        min_track_seconds=3.0,
    )
    assert len(kept) == 1


class _FakeCv2:
    def __init__(self, frames: list[Any]) -> None:
        self._frames = frames
        self.written: list[str] = []

    def imread(self, path: str) -> Any:  # noqa: N802 - cv2 API name
        _ = path
        return self._frames[0]

    def imwrite(self, path: str, image: Any) -> bool:  # noqa: N802 - cv2 API name
        _ = image
        Path(path).write_bytes(b"x")
        self.written.append(path)
        return True


def _patch_selected_frames(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[Any],
) -> None:
    """Stub the approved FFmpeg decode path used by crop extraction."""

    def _fake_iter(
        media_path: Path,
        frame_numbers: list[int],
        *,
        cv2: Any,
        np: Any,
    ) -> Any:
        _ = media_path, cv2, np
        for frame_number in frame_numbers:
            yield frame_number, frames[frame_number]

    monkeypatch.setattr(crops_module, "iter_selected_video_frames_bgr", _fake_iter)
    monkeypatch.setattr(
        crops_module,
        "import_optional",
        lambda name, *, backend: np if name == "numpy" else _FakeCv2(frames),
    )


def test_write_crops_writes_files_and_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = plan_track_crops(
        _objects_payload(),
        _instances_payload(),
        crop_classes=("person",),
        crops_per_track=3,
    )
    frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(8)]
    fake = _FakeCv2(frames)
    _patch_selected_frames(monkeypatch, frames)

    records = write_crops(
        Path("clip.mp4"),
        plans,
        output_root=tmp_path,
        crop_subdir="tracks/crops",
        crop_format="jpg",
        cv2=fake,
    )

    assert len(records) == 1
    record = records[0]
    assert record.n_crops == 3
    assert record.crop_dir == "tracks/crops/track_0002"
    assert all(c.startswith("tracks/crops/track_0002/crop_") for c in record.crops)
    assert len(fake.written) == 3


@pytest.mark.parametrize("crop_subdir", ["../escape", "/abs/escape", "tracks/../../escape"])
def test_write_crops_rejects_escaping_crop_subdir(
    tmp_path: Path,
    crop_subdir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator-supplied crop_subdir cannot write outside the sidecars root."""
    plans = plan_track_crops(
        _objects_payload(),
        _instances_payload(),
        crop_classes=("person",),
        crops_per_track=3,
    )
    frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(8)]
    fake = _FakeCv2(frames)
    _patch_selected_frames(monkeypatch, frames)
    with pytest.raises(ValueError, match="escapes scene sidecars root"):
        write_crops(
            Path("clip.mp4"),
            plans,
            output_root=tmp_path,
            crop_subdir=crop_subdir,
            crop_format="jpg",
            cv2=fake,
        )
    assert fake.written == []


def test_extract_track_crops_writes_tracks_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_paths = ensure_scene_skeleton(tmp_path / "chunk_000")
    objects_json = scene_paths.contextual_dir / "objects.json"
    instances_json = scene_paths.contextual_dir / "instances.json"
    write_json(objects_json, _objects_payload())
    write_json(instances_json, _instances_payload())

    frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(8)]
    _patch_selected_frames(monkeypatch, frames)

    tracks_path = extract_track_crops(
        Path("clip.mp4"),
        scene_paths,
        pas_tracking_config(crops_per_track=3, crop_padding=0.0, min_crop_size=0),
        objects_json=objects_json,
        instances_json=instances_json,
        logger=logging.getLogger("test_crops"),
    )

    assert tracks_path is not None
    payload = read_json(tracks_path)
    assert payload["crop_root"] == "tracks/crops"
    assert payload["n_tracks"] == 1
    assert payload["tracks"][0]["track_id"] == 2
    assert payload["tracks"][0]["detection_score"] == 0.8552


def test_extract_track_crops_skips_without_artifacts(tmp_path: Path) -> None:
    scene_paths = ensure_scene_skeleton(tmp_path / "chunk_001")
    result = extract_track_crops(
        Path("clip.mp4"),
        scene_paths,
        DetectionAndTrackingConfig(extract_crops=True),
        objects_json=scene_paths.contextual_dir / "missing_objects.json",
        instances_json=scene_paths.contextual_dir / "missing_instances.json",
        logger=logging.getLogger("test_crops"),
    )
    assert result is None
