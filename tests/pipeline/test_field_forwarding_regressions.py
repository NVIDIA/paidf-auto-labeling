# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Config field-forwarding regression tests.
#
# With the new stage-object architecture, the pipeline makes direct Python calls to
# pre-built stage objects (no subprocess argv for tracking/vlm/mcq).  These tests
# verify that config knobs are correctly parsed into the Pydantic model fields that
# stage factories will consume when constructing stage objects.

from __future__ import annotations

from pathlib import Path

import config.loader
import config.schema
import pytest

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
PIPELINE_EXAMPLE_CONFIG = CONFIGS_DIR / "pipeline_example.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_and_validate(tmp_path: Path, extra_overrides: list[str]):
    """Load blueprint config with overrides; return the validated PipelineConfig."""
    if not PIPELINE_EXAMPLE_CONFIG.exists():
        pytest.skip(f"Config not found: {PIPELINE_EXAMPLE_CONFIG}")

    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"")

    base = [
        f"data.0.inputs.video_path={video}",
        f"data.0.output.out_dir={tmp_path / 'out'}",
        f"data.0.output.log_dir={tmp_path / 'out' / 'logs'}",
        "endpoints.vlm.url=http://example.invalid/v1",
        "endpoints.vlm.model=dummy-vlm",
        "endpoints.llm.url=http://example.invalid/v1",
        "endpoints.llm.model=dummy-llm",
    ]

    cfg, _ = config.loader.load_config_with_overrides(str(PIPELINE_EXAMPLE_CONFIG), base + extra_overrides, logger=None)
    validated = config.schema.validate_schema(cfg, logger=None)
    assert validated is not None
    return validated


# ---------------------------------------------------------------------------
# SR knobs
# ---------------------------------------------------------------------------


def test_sr_out_fps_is_forwarded(tmp_path: Path) -> None:
    validated = _load_and_validate(
        tmp_path,
        [
            "detection_and_tracking.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=false",
            "super_resolution.enabled=true",
            "super_resolution.out_fps=15.5",
        ],
    )
    assert validated.super_resolution is not None
    assert validated.super_resolution.out_fps == pytest.approx(15.5)


# ---------------------------------------------------------------------------
# Tracking knobs
# ---------------------------------------------------------------------------


def test_tracking_advanced_knobs_are_forwarded(tmp_path: Path) -> None:
    validated = _load_and_validate(
        tmp_path,
        [
            "super_resolution.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=false",
            "detection_and_tracking.enabled=true",
            "detection_and_tracking.per_class=true",
            "detection_and_tracking.asso_func=giou",
            "detection_and_tracking.min_hits=7",
            "detection_and_tracking.max_age=99",
            "detection_and_tracking.min_track_frames=11",
            "detection_and_tracking.deepocsort_stage2_off=true",
            "detection_and_tracking.deepocsort_min_hits_nonconsecutive=true",
            "detection_and_tracking.cross_class_iou_threshold=0.88",
            "detection_and_tracking.dedup_iou_threshold=0.22",
            "detection_and_tracking.dedup_priority=prev_iou",
            "detection_and_tracking.copy_video=false",
            "detection_and_tracking.save_rgb=false",
        ],
    )
    dt = validated.detection_and_tracking
    assert dt is not None
    assert dt.per_class is True
    assert dt.asso_func == "giou"
    assert dt.min_hits == 7
    assert dt.max_age == 99
    assert dt.min_track_frames == 11
    assert dt.deepocsort_stage2_off is True
    assert dt.deepocsort_min_hits_nonconsecutive is True
    assert dt.cross_class_iou_threshold == pytest.approx(0.88)
    assert dt.dedup_iou_threshold == pytest.approx(0.22)
    assert dt.dedup_priority == "prev_iou"
    assert dt.copy_video is False
    assert dt.save_rgb is False


def test_tracking_save_rgb_true_is_forwarded(tmp_path: Path) -> None:
    validated = _load_and_validate(
        tmp_path,
        [
            "super_resolution.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=false",
            "detection_and_tracking.enabled=true",
            "detection_and_tracking.save_rgb=true",
        ],
    )
    assert validated.detection_and_tracking is not None
    assert validated.detection_and_tracking.save_rgb is True


# ---------------------------------------------------------------------------
# VLM JSON knobs
# ---------------------------------------------------------------------------


def test_vlm_json_knobs_forwarded(tmp_path: Path) -> None:
    validated = _load_and_validate(
        tmp_path,
        [
            "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false",
            "mcq_generation.enabled=false",
            "vlm_json.enabled=true",
            "vlm_json.max_tokens=1234",
            "vlm_json.structured_output=nim",
            "endpoints.vlm.retries=7",
        ],
    )
    assert validated.vlm_json is not None
    assert validated.vlm_json.max_tokens == 1234
    assert validated.vlm_json.structured_output == "nim"
    assert validated.endpoints is not None
    assert validated.endpoints.vlm is not None
    assert validated.endpoints.vlm.retries == 7


# ---------------------------------------------------------------------------
# MCQ knobs
# ---------------------------------------------------------------------------


def test_mcq_window_vlm_llm_tokens_and_structured_output_forwarded(tmp_path: Path) -> None:
    validated = _load_and_validate(
        tmp_path,
        [
            "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=true",
            "mcq_generation.mode=window-vlm-llm",
            "mcq_generation.window_metadata_extraction.vlm_max_tokens=1111",
            "mcq_generation.window_metadata_extraction.llm_max_tokens=2222",
            "mcq_generation.window_metadata_extraction.llm_structured_output=nim",
            "mcq_generation.window_metadata_extraction.vlm_verify_max_tokens=3333",
        ],
    )
    wme = validated.mcq_generation.window_metadata_extraction
    assert wme is not None
    assert wme.vlm_max_tokens == 1111
    assert wme.llm_max_tokens == 2222
    assert wme.llm_structured_output == "nim"
    assert wme.vlm_verify_max_tokens == 3333


def test_mcq_window_direct_vlm_structured_output_forwarded(tmp_path: Path) -> None:
    validated = _load_and_validate(
        tmp_path,
        [
            "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=true",
            "mcq_generation.mode=window-direct-vlm",
            "mcq_generation.window_metadata_extraction.vlm_max_tokens=4444",
            "mcq_generation.window_metadata_extraction.vlm_structured_output=nim",
        ],
    )
    wme = validated.mcq_generation.window_metadata_extraction
    assert wme is not None
    assert wme.vlm_max_tokens == 4444
    assert wme.vlm_structured_output == "nim"


def test_mcq_qd_prompt_gen_max_tokens_forwarded(tmp_path: Path) -> None:
    validated = _load_and_validate(
        tmp_path,
        [
            "super_resolution.enabled=false",
            "detection_and_tracking.enabled=false",
            "vlm_json.enabled=false",
            "mcq_generation.enabled=true",
            "mcq_generation.mode=question-driven-vlm-llm",
            "mcq_generation.window_metadata_extraction.prompt_gen_llm_max_tokens=5555",
        ],
    )
    wme = validated.mcq_generation.window_metadata_extraction
    assert wme is not None
    assert wme.prompt_gen_llm_max_tokens == 5555
