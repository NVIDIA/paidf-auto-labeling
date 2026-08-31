# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"
# ruff: noqa: E501

"""End-to-end DAFT validation: drive the PL converters to build a scene,
then shell out to ``tao-daft validate --strict`` to confirm it's schema-valid.

Skipped automatically when ``tao-daft`` is not on ``PATH`` (which is the
default in the PL container — ``nvidia-tao-daft`` is an opt-in dev install).
Per DAFT_CONVERTER_PLAN.md §9.4 this is the contract the container produces.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from reasoning.causal_linkage.converter import to_daft_causal_linkage
from reasoning.chunks.converter import to_daft_chunks
from reasoning.common import SceneContext, write_daft_json
from reasoning.contextual import to_daft_events, to_daft_video
from reasoning.msted.converter import to_daft_msted
from reasoning.paths import ScenePaths, ensure_scene_skeleton
from reasoning.prose_tasks.converter import to_daft_scene_description, to_daft_video_summarization
from reasoning.qa.converter import (
    to_daft_bcq_openended,
    to_daft_mcq_openended,
    to_daft_open_qa,
)
from reasoning.task import to_daft_tasks
from reasoning.temporal.converter import to_daft_temporal_description
from reasoning.temporal_localization.converter import to_daft_temporal_localization
from reasoning.tracking import to_daft_instances, to_daft_objects, to_daft_tracking

pytestmark = pytest.mark.skipif(
    shutil.which("tao-daft") is None,
    reason="tao-daft CLI not on PATH; install nvidia-tao-daft to exercise strict validation",
)

SCENE_VIDEO_ID = "main"
SCENE_CTX = SceneContext(media_id=SCENE_VIDEO_ID, iso_date="2026-04-20")
VIDEO_DURATION = 10.0


def _build_full_scene(scene_dir: Path) -> ScenePaths:
    """Build a complete DAFT scene (all contextual + task files) by driving
    the converters with realistic PL-internal payloads."""
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
            ctx=SCENE_CTX,
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
            ctx=SCENE_CTX,
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
            },
            ctx=SCENE_CTX,
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
            ctx=SCENE_CTX,
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
        ctx=SCENE_CTX,
    )
    assert mcq_payload is not None
    assert bcq_payload is not None
    assert open_qa_payload is None
    write_daft_json(paths.task_mcq, mcq_payload)
    write_daft_json(paths.task_bcq, bcq_payload)

    # scene_description + video_summarization with synthetic reasoning
    # traces (the shape the optional LLM enrichment pass would produce).
    # The reasoning field is optional in both schemas; including it here
    # exercises the validator round-trip for the enriched output too.
    sd_payload = to_daft_scene_description(
        "A car drives down a residential street in daylight.",
        ctx=SCENE_CTX,
        reasoning="The captions consistently describe a sedan moving down a residential street under daylight conditions.",
    )
    assert sd_payload is not None
    write_daft_json(paths.task_scene_description, sd_payload)

    vs_payload = to_daft_video_summarization(
        "A silver sedan accelerates and turns right at an intersection without incident.",
        ctx=SCENE_CTX,
        reasoning="The per-segment captions show a steady acceleration followed by a right turn through the intersection with no collision.",
        timestamp=(0.0, VIDEO_DURATION),
    )
    assert vs_payload is not None
    write_daft_json(paths.task_video_summarization, vs_payload)

    # chunks.json — dense temporal segmentation. Mirrors the per-window
    # output that ``window_vlm_llm`` writes to ``sidecars/metadata.json``
    # so the converter and pipeline-level re-pivot share a fixture shape.
    chunks_payload = to_daft_chunks(
        [
            {
                "start_s": 0.0,
                "end_s": 4.0,
                "caption": "Traffic flows normally; one car in lane 5, one in lane 3.",
                "tags": ["normal_flow"],
            },
            {
                "start_s": 4.0,
                "end_s": 7.0,
                "caption": "The car in lane 3 turns right through the intersection.",
                "tags": ["turn"],
            },
            {
                "start_s": 7.0,
                "end_s": VIDEO_DURATION,
                "caption": "The car continues out of frame; intersection clears.",
            },
        ],
        ctx=SCENE_CTX,
        duration=VIDEO_DURATION,
    )
    assert chunks_payload is not None
    write_daft_json(paths.contextual_chunks, chunks_payload)

    # temporal_description.json — same per-window source as chunks.json,
    # but rendered into the task namespace as (question, answer) items.
    # Uses the converter's default question template (the schema example,
    # use-case agnostic) so the validator round-trip exercises the
    # default code path; domain-specific overrides are tested in unit tests.
    td_payload = to_daft_temporal_description(
        [
            {
                "start_s": 0.0,
                "end_s": 4.0,
                "caption": "Traffic flows normally; one car in lane 5, one in lane 3.",
            },
            {
                "start_s": 4.0,
                "end_s": 7.0,
                "caption": "The car in lane 3 turns right through the intersection.",
            },
            {
                "start_s": 7.0,
                "end_s": VIDEO_DURATION,
                "caption": "The car continues out of frame; intersection clears.",
            },
        ],
        ctx=SCENE_CTX,
        duration=VIDEO_DURATION,
    )
    assert td_payload is not None
    write_daft_json(paths.task_temporal_description, td_payload)

    # temporal_localization.json — synthetic LLM grounding output (the
    # shape ``generate_temporal_localizations_with_llm`` produces). Hand-
    # built here so the validator round-trip exercises the converter
    # without dragging in an LLM mock.
    tl_payload = to_daft_temporal_localization(
        [
            {
                "question": "When does the car turn right?",
                "answer": {"start": 4.0, "end": 7.0},
                "reasoning": "The per-segment captions describe the right turn between 00:04 and 00:07.",
            },
            {
                "question": "When does the intersection clear?",
                "answer": {"start": 7.0, "end": VIDEO_DURATION},
                "reasoning": "The final segment caption notes the intersection returning to its idle state.",
            },
        ],
        ctx=SCENE_CTX,
        duration=VIDEO_DURATION,
    )
    assert tl_payload is not None
    write_daft_json(paths.task_temporal_localization, tl_payload)

    # msted.json — synthetic structured event output (the shape the LLM
    # aggregator would produce). Hand-built here so the validator
    # round-trip exercises the converter without dragging in an LLM
    # mock; the actual LLM glue is unit-tested separately.
    msted_payload = to_daft_msted(
        {
            "scene_description": "A silver sedan accelerates and turns right at an intersection without incident.",
            "temporal_spatial_localization": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "description": "Traffic flows normally; one car in lane 5, one in lane 3.",
                },
                {
                    "start": 4.0,
                    "end": 7.0,
                    "description": "The car in lane 3 turns right through the intersection.",
                    "spatial_region": "center frame",
                },
                {
                    "start": 7.0,
                    "end": VIDEO_DURATION,
                    "description": "The car continues out of frame; intersection clears.",
                },
            ],
            "event_description": {
                "category": "right turn at intersection",
                "description": "A silver sedan executes a right turn through a residential intersection.",
                "spatial_location": "intersection at the end of the residential street",
                "consequence": "the intersection returns to its idle state with no impact",
            },
        },
        ctx=SCENE_CTX,
        sources=["sidecars/metadata.json"],
        duration=VIDEO_DURATION,
    )
    write_daft_json(paths.contextual_msted, msted_payload)

    # open_qa.json — synthetic free-form QA items (the shape
    # ``generate_qa_with_llm(kind="open_qa")`` produces). Hand-built
    # here so the validator round-trip exercises the converter without
    # dragging in an LLM mock; the LLM glue is unit-tested separately.
    open_qa_payload = to_daft_open_qa(
        [
            {
                "question": "What does the car do at the intersection?",
                "answer": "It executes a right turn through the intersection without incident.",
                "reasoning": "The captions describe a steady right turn between 00:04 and 00:07.",
            },
        ],
        ctx=SCENE_CTX,
    )
    assert open_qa_payload is not None
    write_daft_json(paths.task_open_qa, open_qa_payload)

    # mcq_openended.json — synthetic open-ended MCQ items.
    mcq_oe_payload = to_daft_mcq_openended(
        [
            {
                "question": "What type of vehicle is shown?",
                "answer": "A. The vehicle has four doors and a trunk silhouette typical of a sedan.",
                "options": {"A": "Sedan", "B": "Truck", "C": "SUV", "D": "Motorcycle"},
                "reasoning": "Captions consistently describe a silver sedan.",
            },
        ],
        ctx=SCENE_CTX,
    )
    assert mcq_oe_payload is not None
    write_daft_json(paths.task_mcq_openended, mcq_oe_payload)

    # bcq_openended.json — synthetic open-ended Yes/No items.
    bcq_oe_payload = to_daft_bcq_openended(
        [
            {
                "question": "Does the vehicle complete the turn safely?",
                "answer": "Yes. The captions describe the right turn proceeding without impact and the intersection clearing afterwards.",
                "reasoning": "No collision, debris, or evasive maneuver is mentioned in any segment.",
            },
        ],
        ctx=SCENE_CTX,
    )
    assert bcq_oe_payload is not None
    write_daft_json(paths.task_bcq_openended, bcq_oe_payload)

    # causal_linkage.json — synthetic LLM grounding output for two
    # consecutive events. Hand-built so the validator round-trip
    # exercises the converter without an LLM mock.
    cl_payload = to_daft_causal_linkage(
        [
            {
                "t1": 1.5,
                "t2": 4.0,
                "question": "Explain the relationship between the event at 00:01.500 and the situation at 00:04.",
                "answer": "The car begins accelerating at t1 which positions it to enter the intersection at t2 where it then executes the right turn.",
                "video_type": "normal",
                "reasoning": "Captions show a smooth acceleration immediately before the turn maneuver.",
            },
        ],
        ctx=SCENE_CTX,
        duration=VIDEO_DURATION,
    )
    assert cl_payload is not None
    write_daft_json(paths.task_causal_linkage, cl_payload)

    # tracking.json (4D MOT) — synthetic 3D pose data simulating output from
    # an external monocular-depth estimator or simulator. Required because
    # tracking.json's ``3d_location`` field has no PL-internal source today;
    # the validator round-trip here is the contract for future 3D producers.
    write_daft_json(
        paths.contextual_tracking,
        to_daft_tracking(
            {
                "frames": {
                    "frame_000001": [
                        {
                            "object_id": "car_1",
                            "3d_location": [5.2, 10.3, 0.0],
                            "3d_bounding_box_scale": [2.0, 4.5, 1.5],
                            "3d_bounding_box_rotation": [0.0, 0.0, 1.57],
                            "2d_bounding_box_visible": {
                                "cam_001": [100.0, 200.0, 300.0, 400.0],
                            },
                        },
                    ],
                    "frame_000030": [
                        {
                            "object_id": "car_1",
                            "3d_location": [6.0, 10.5, 0.0],
                            "3d_bounding_box_scale": [2.0, 4.5, 1.5],
                            "3d_bounding_box_rotation": [0.0, 0.0, 1.60],
                            "2d_bounding_box_visible": {
                                "cam_001": [120.0, 210.0, 320.0, 410.0],
                            },
                        },
                    ],
                }
            },
            ctx=SCENE_CTX,
        ),
    )

    return paths


def _validate(scene_dir: Path) -> subprocess.CompletedProcess[str]:
    # The format is a positional subcommand on ``tao-daft validate``, not a
    # ``--version`` flag (the format-name *is* the CLI sub-parser name).
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
    assert proc.returncode == 0, (
        f"tao-daft validate failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_partial_scene_passes_strict_validation(tmp_path: Path) -> None:
    """Scenes with some stages disabled (no tracking/events/etc.) still
    validate. This answers DAFT_CONVERTER_PLAN.md §8.10: we don't need a
    "complete scene" guard in the CLI."""
    scene = tmp_path / "scene"
    paths = _build_full_scene(scene)
    paths.contextual_events.unlink()
    paths.contextual_instances.unlink()
    paths.contextual_objects.unlink()
    paths.contextual_tracking.unlink()
    paths.contextual_chunks.unlink()
    paths.contextual_msted.unlink()
    paths.task_temporal_description.unlink()
    paths.task_bcq.unlink()

    proc = _validate(scene)
    assert proc.returncode == 0, (
        f"tao-daft validate failed on partial scene:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_corrupt_mcq_answer_is_caught(tmp_path: Path) -> None:
    """Guard: the validator is actually enforcing the MCQ answer regex. Without
    this we could silently regress to 'DAFT-shaped but not DAFT-valid' output."""
    scene = tmp_path / "scene"
    paths = _build_full_scene(scene)
    data = json.loads(paths.task_mcq.read_text(encoding="utf-8"))
    data["items"][0]["answer"] = "AB"
    paths.task_mcq.write_text(json.dumps(data), encoding="utf-8")

    proc = _validate(scene)
    assert proc.returncode != 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "answer" in combined


# ---------------------------------------------------------------------------
# Image-scene contract guard: ``objects.json`` is supported for image scenes,
# ``instances.json`` is rejected. Empirically established against
# ``tao-daft validate --raw image`` (the validator's default contextual set
# for image scenes is ``['objects', 'tracking']``; ``instances`` is
# explicitly rejected with "Contextual type 'instances' not valid for raw
# type 'image'"). This test is the safety net that prevents drift back to
# the previous (inverted) implementation that skipped objects.json for
# images and kept writing instances.json.
# ---------------------------------------------------------------------------

IMAGE_CTX = SceneContext(media_id="img_001", is_image=True, iso_date="2026-04-20")


def _build_image_scene_with_objects(scene_dir: Path) -> ScenePaths:
    """Build a minimal image scene with ``objects.json`` (and the
    ``image.json`` companion required by the validator)."""
    paths = ensure_scene_skeleton(scene_dir)

    (paths.raw_dir / "img_001.png").write_bytes(b"\x00" * 16)

    # contextual/image.json — scene-level metadata for the image input.
    write_daft_json(
        paths.contextual_image,
        {
            "version": "metropolis-v3.0",
            "image_id": "img_001",
            "format": "png",
            "height": 1080,
            "width": 1920,
            "caption": "A red car at an intersection.",
            "metadata": {"type": "image", "date": "2026-04-20"},
        },
    )

    # contextual/objects.json — driven via the actual converter so any
    # regression to the old "image rejection" raises the test.
    write_daft_json(
        paths.contextual_objects,
        to_daft_objects(
            {
                "frames": {
                    "frame_000000": {
                        "format": "png",
                        "frame_number": 0,
                        "instances": [
                            {
                                "object_id": "car_1",
                                "bounding_box_2d_tight": [10.0, 20.0, 100.0, 200.0],
                            }
                        ],
                    }
                }
            },
            ctx=IMAGE_CTX,
        ),
    )

    return paths


def test_image_scene_objects_json_passes_strict_validation(tmp_path: Path) -> None:
    """Empirical contract: ``objects.json`` for an image scene passes
    ``tao-daft validate --raw image --strict``. This is the
    authoritative source of truth for which contextual files are
    allowed per raw type."""
    scene = tmp_path / "scene"
    _build_image_scene_with_objects(scene)
    proc = subprocess.run(
        [
            "tao-daft",
            "validate",
            "metropolis-v3.0",
            "--path",
            str(scene),
            "--raw",
            "image",
            "--contextual",
            "objects",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # The scene has no task/ files, which surfaces as the warning
    # "task/ directory is empty" under --strict. That's unrelated to
    # the contract we're guarding here — what we require is:
    #   (a) tao-daft did NOT reject 'objects' as a contextual type for
    #       --raw image (the regression we'd see if the previous
    #       inverted implementation was actually correct), AND
    #   (b) the report says 'Errors: 0' (the contextual phase passed).
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "Contextual type 'objects' not valid" not in combined, (
        "tao-daft must accept objects.json for --raw image; if this fires, "
        "the validator contract has changed and the converter's "
        "image-scene support needs re-evaluation. Output:\n" + combined
    )
    assert "Errors: 0" in combined, (
        "image-scene objects.json must produce zero contextual errors "
        f"under tao-daft --raw image --strict; got:\n{combined}"
    )


def _build_image_scene_with_objects_and_instances(scene_dir: Path) -> ScenePaths:
    """Like :func:`_build_image_scene_with_objects` but also writes an
    image-scene ``instances.json`` driven through the actual converter.

    Used by the two image-scene instances.json validator tests below
    to anchor the "schema-valid + default-validator-ignored" contract.
    """
    paths = _build_image_scene_with_objects(scene_dir)
    write_daft_json(
        paths.contextual_instances,
        to_daft_instances(
            {
                "instances": {
                    "car_1": {
                        "object_type": "car",
                        "instance_id": 1,
                        "semantic_id": 10,
                    }
                }
            },
            ctx=IMAGE_CTX,
        ),
    )
    return paths


def test_image_scene_instances_json_default_validation_passes(tmp_path: Path) -> None:
    """Empirical contract: an image scene that ALSO carries
    ``instances.json`` validates cleanly under the default
    ``tao-daft validate --raw image --strict`` flow.

    The default contextual set for ``--raw image`` is
    ``[objects, tracking]``, so the validator silently ignores any
    ``instances.json`` present in the directory. This means writing
    the file is *free* under default validation — it surfaces as
    neither error nor warning.

    This test guards against a future TAO release that might tighten
    the default set to actively reject unknown contextual files
    (which would force us to gate emission on raw type again).
    """
    scene = tmp_path / "scene"
    _build_image_scene_with_objects_and_instances(scene)
    proc = subprocess.run(
        [
            "tao-daft",
            "validate",
            "metropolis-v3.0",
            "--path",
            str(scene),
            "--raw",
            "image",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    # The scene has no task/ files, which surfaces as "task/ directory
    # not found" — that's unrelated to the contract here. What we
    # require is that the contextual phase reports zero errors AND
    # that the validator did NOT escalate the presence of
    # instances.json into a rejection.
    assert "Contextual type 'instances' not valid" not in combined, (
        "Default --strict validation must not reject the mere presence "
        "of instances.json in an image scene (it's not in the default "
        "contextual set, so it should be ignored). Output:\n" + combined
    )


def test_image_scene_instances_json_explicit_contextual_rejected(tmp_path: Path) -> None:
    """Empirical contract (the explicit-ask path): asking the
    validator to specifically validate ``instances`` for ``--raw image``
    via ``--contextual instances`` IS rejected with the canonical
    error string.

    This is the validator's opinion about which contextual types
    "belong" to an image dataset; it's a downstream-tool stance and
    not a schema constraint (``instances.schema.json`` itself defines
    the per-instance ``images`` array specifically to support image
    scenes). We document the behavior here so a future TAO release
    that loosens this stance trips this test and prompts us to
    re-examine whether the explicit-ask contract is still in force.
    """
    scene = tmp_path / "scene"
    _build_image_scene_with_objects_and_instances(scene)

    proc = subprocess.run(
        [
            "tao-daft",
            "validate",
            "metropolis-v3.0",
            "--path",
            str(scene),
            "--raw",
            "image",
            "--contextual",
            "instances",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0, (
        "tao-daft must reject EXPLICIT --contextual instances for --raw "
        "image; if this passes, the validator contract has loosened. "
        "Output:\n" + combined
    )
    assert "instances" in combined and "image" in combined, (
        "expected the validator's 'Contextual type instances not valid "
        "for raw type image' error; got:\n" + combined
    )
