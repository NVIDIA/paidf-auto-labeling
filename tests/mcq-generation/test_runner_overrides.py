# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Tests that MCQ runner configuration knobs are correctly parsed from YAML overrides
# into the Pydantic schema.  With the new stage-object architecture, MCQ stages are
# direct Python calls (no subprocess argv), so config forwarding is verified at the
# schema level rather than by inspecting log output.

from __future__ import annotations

import pytest
from al_utils.schema.mcq import McqGenerationConfig, WindowMetadataExtractionConfig


def test_pipeline_forwards_caption_keys_window_vlm_llm() -> None:
    """window-vlm-llm: caption_key and enhanced_caption_key are stored in the config model."""
    wme = WindowMetadataExtractionConfig(
        caption_key="cap_x",
        enhanced_caption_key="enh_y",
        skip_existing=True,
        single_window=True,  # window_frames guard: single_window bypasses window_frames>0 check
    )
    mcq = McqGenerationConfig(enabled=True, mode="window-vlm-llm", window_metadata_extraction=wme)
    assert mcq.window_metadata_extraction is not None
    assert mcq.window_metadata_extraction.caption_key == "cap_x"
    assert mcq.window_metadata_extraction.enhanced_caption_key == "enh_y"
    assert mcq.window_metadata_extraction.skip_existing is True


def test_pipeline_forwards_overrides_question_driven() -> None:
    """question-driven-vlm-llm: all window_metadata_extraction knobs are stored correctly."""
    wme = WindowMetadataExtractionConfig(
        window_seconds=9.5,
        window_frames=123,
        sampling_fps=7.0,
        resolution=321,
        max_frames=77,
        vlm_max_tokens=1111,
        llm_max_tokens=2222,
        vlm_temperature=0.0,
        llm_temperature=0.0,
        timeout=12,
        rate_limit=0.0,
        caption_key="cap_qd",
        enhanced_caption_key="enh_qd",
        aggregate_windows=True,
        write_empty_mcq_marker=True,
        skip_existing=True,
    )
    mcq = McqGenerationConfig(
        enabled=True,
        mode="question-driven-vlm-llm",
        window_metadata_extraction=wme,
    )
    assert mcq.window_metadata_extraction is not None
    wme = mcq.window_metadata_extraction
    assert wme.window_seconds == 9.5
    assert wme.max_frames == 77
    assert wme.caption_key == "cap_qd"
    assert wme.enhanced_caption_key == "enh_qd"
    assert wme.aggregate_windows is True
    assert wme.write_empty_mcq_marker is True


def test_window_seconds_only_is_valid() -> None:
    """window_seconds > 0 without window_frames is a valid time-based windowing config."""
    wme = WindowMetadataExtractionConfig(window_seconds=5.0)
    assert wme.window_seconds == 5.0
    assert wme.window_frames == 0


def test_no_windowing_strategy_raises() -> None:
    """Neither window_frames nor window_seconds set (and single_window=False) must raise."""
    with pytest.raises(ValueError, match="window_frames"):
        WindowMetadataExtractionConfig(window_seconds=0.0, window_frames=0, single_window=False)
