# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end DAFT validation: drive the auto-labeling converters to build a scene,
then shell out to ``tao-daft validate --strict`` to confirm it's schema-valid.

Skipped automatically when ``tao-daft`` is not on ``PATH`` (which is the
default in the auto-labeling container — ``nvidia-tao-daft`` is an opt-in dev install).
This is the contract the container produces.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from daft_export.common import write_daft_json
from daft_export.contextual import to_daft_events, to_daft_video
from daft_export.paths import ScenePaths, ensure_scene_skeleton
from daft_export.task import to_daft_tasks
from daft_export.tracking import to_daft_instances, to_daft_objects

pytestmark = pytest.mark.skipif(
    shutil.which("tao-daft") is None,
    reason="tao-daft CLI not on PATH; install nvidia-tao-daft to exercise strict validation",
)

SCENE_VIDEO_ID = "main"
VIDEO_DURATION = 10.0


def _build_full_scene(scene_dir: Path) -> ScenePaths:
    """Build a complete DAFT scene (all contextual + task files) by driving
    the converters with realistic auto-labeling-internal payloads."""
    paths = ensure_scene_skeleton(scene_dir)

    (paths.raw_dir / "main.mp4").write_bytes(b"\x00" * 16)

    write_daft_json(
        paths.contextual_video,
        to_daft_video(
            {
                "format": "mp4",
                "fps": 30.0,
                "duration": VIDEO_DURATION,
                "height": 720,
                "width": 1280,
                "scene_description": "A car drives down a residential street in daylight.",
            },
            video_id=SCENE_VIDEO_ID,
        ),
    )

    write_daft_json(
        paths.contextual_events,
        to_daft_events(
            {
                "events": [
                    {
                        "event_id": "evt_001",
                        "start_time": 1.5,
                        "end_time": 3.25,
                        "event_caption": "Car begins accelerating.",
                        "category": "driving",
                        "instances": ["car_1"],
                    },
                    {
                        "event_id": "evt_002",
                        "start_time": 4.0,
                        "end_time": 7.0,
                        "event_caption": "Car turns right at intersection.",
                        "category": "driving",
                        "instances": ["car_1"],
                    },
                ]
            },
            video_id=SCENE_VIDEO_ID,
            duration=VIDEO_DURATION,
        ),
    )

    write_daft_json(
        paths.contextual_instances,
        to_daft_instances(
            {
                "instances": {
                    "car_1": {
                        "object_type": "car",
                        "instance_id": 1,
                        "semantic_id": 0,
                        "caption": "Silver sedan",
                    },
                }
            }
        ),
    )

    write_daft_json(
        paths.contextual_objects,
        to_daft_objects(
            {
                "frames": {
                    "frame_000001": {
                        "format": "jpg",
                        "frame_number": 1,
                        "width": 1280,
                        "height": 720,
                        "instances": [
                            {
                                "object_id": "car_1",
                                "bounding_box_2d_tight": [100.0, 200.0, 300.0, 400.0],
                                "confidence": 0.92,
                            }
                        ],
                    },
                    "frame_000030": {
                        "format": "jpg",
                        "frame_number": 30,
                        "width": 1280,
                        "height": 720,
                        "instances": [
                            {
                                "object_id": "car_1",
                                "bounding_box_2d_tight": [120.0, 210.0, 320.0, 410.0],
                            }
                        ],
                    },
                }
            },
            video_id=SCENE_VIDEO_ID,
        ),
    )

    mcq_payload, bcq_payload, open_qa_payload = to_daft_tasks(
        [
            {
                "id": "q1",
                "question": "What type of vehicle is shown?",
                "options": ["Sedan", "Truck", "SUV", "Motorcycle"],
                "answer": "Sedan",
                "reasoning_trace": "The vehicle has four doors and a trunk silhouette typical of a sedan.",
            },
            {
                "id": "q2",
                "question": "What does the car do at the intersection?",
                "options": ["Turns right", "Turns left", "Goes straight", "Stops"],
                "answer": "Turns right",
            },
            {
                "id": "q3",
                "question": "Does a collision occur in the video?",
                "options": ["Yes", "No"],
                "answer": "No",
                "reasoning_trace": "The car drives smoothly through the intersection without impact.",
            },
        ],
        video_id=SCENE_VIDEO_ID,
    )
    assert mcq_payload is not None
    assert bcq_payload is not None
    assert open_qa_payload is None
    write_daft_json(paths.task_mcq, mcq_payload)
    write_daft_json(paths.task_bcq, bcq_payload)

    return paths


def _validate(scene_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "tao-daft",
            "validate",
            "metropolis-v3.0",
            "--path",
            str(scene_dir),
            "--raw",
            "auto",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_full_scene_passes_strict_validation(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _build_full_scene(scene)
    proc = _validate(scene)
    assert proc.returncode == 0, f"tao-daft validate failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_partial_scene_passes_strict_validation(tmp_path: Path) -> None:
    """Scenes with some stages disabled (no tracking/events/etc.) still
    validate, so we don't need a "complete scene" guard in the CLI."""
    scene = tmp_path / "scene"
    paths = _build_full_scene(scene)
    paths.contextual_events.unlink()
    paths.contextual_instances.unlink()
    paths.contextual_objects.unlink()
    paths.task_bcq.unlink()

    proc = _validate(scene)
    assert proc.returncode == 0, (
        f"tao-daft validate failed on partial scene:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_corrupt_mcq_answer_is_caught(tmp_path: Path) -> None:
    """Guard: the validator is actually enforcing the MCQ answer regex. Without
    this we could silently regress to 'DAFT-shaped but not DAFT-valid' output."""
    import json

    scene = tmp_path / "scene"
    paths = _build_full_scene(scene)
    data = json.loads(paths.task_mcq.read_text(encoding="utf-8"))
    data["items"][0]["answer"] = "AB"
    paths.task_mcq.write_text(json.dumps(data), encoding="utf-8")

    proc = _validate(scene)
    assert proc.returncode != 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "answer" in combined
