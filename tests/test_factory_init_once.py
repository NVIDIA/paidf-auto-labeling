# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that stage factories are called once before the sample loop in cli.main().

The factory pattern requires that create_tracker / create_vlm_json_generator /
create_mcq_generator are invoked ONCE regardless of how many samples are in the config.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cli


def _make_two_sample_validated(tmp_path: Path):
    """Return a mock validated PipelineConfig with 2 samples and all stages enabled."""

    def _make_sample(video: str, out: str):
        s = MagicMock()
        s.inputs.video_path = video
        s.inputs.vlm_video_path = None
        s.inputs.metadata_json_path = None
        s.inputs.model_copy.return_value = s.inputs
        s.model_copy.return_value = s
        s.model_dump.return_value = {}
        s.output.out_dir = out
        s.output.log_dir = None
        s.output.config_path = None
        return s

    sample1 = _make_sample("video1.mp4", str(tmp_path / "out1"))
    sample2 = _make_sample("video2.mp4", str(tmp_path / "out2"))

    mock_validated = MagicMock()
    mock_validated.data = [sample1, sample2]
    mock_validated.endpoints = None
    mock_validated.model_dump.return_value = {}
    mock_validated.super_resolution = None
    mock_validated.vlm_json = None
    mock_validated.mcq_generation = None

    # Enable detection_and_tracking
    mock_validated.detection_and_tracking = MagicMock()
    mock_validated.detection_and_tracking.enabled = True

    return mock_validated, sample1, sample2


def test_create_tracker_called_once_for_two_samples(tmp_path):
    """create_tracker must be called ONCE before the sample loop, not once per sample."""
    mock_validated, _, _ = _make_two_sample_validated(tmp_path)

    mock_tracker_instance = MagicMock()

    with (
        patch("cli.load_config_with_overrides", return_value=({}, Path("."))),
        patch("cli.validate_schema", return_value=mock_validated),
        patch("cli.run_pipeline", return_value=0) as mock_run_pipe,
        patch("cli.setup_msc_config"),
        patch("cli.NVCFProgressTracker"),
        patch("cli.localize_path_to_dir", return_value=Path("local/video.mp4")),
        patch("cli.EndpointResolver"),
        patch("cli.create_tracker", return_value=mock_tracker_instance) as mock_create,
    ):
        rc = cli.main(["--config", "c.yaml"])

    assert rc == 0
    # Factory called exactly once, regardless of sample count
    assert mock_create.call_count == 1, (
        f"create_tracker should be called once before the loop, got {mock_create.call_count}"
    )
    # run_pipeline called once per sample
    assert mock_run_pipe.call_count == 2, (
        f"run_pipeline should be called once per sample (2), got {mock_run_pipe.call_count}"
    )
    # Same tracker instance passed to both run_pipeline calls
    calls = mock_run_pipe.call_args_list
    tracker_in_call_0 = calls[0].kwargs.get("det_tracker")
    tracker_in_call_1 = calls[1].kwargs.get("det_tracker")
    assert tracker_in_call_0 is mock_tracker_instance
    assert tracker_in_call_1 is mock_tracker_instance
    assert tracker_in_call_0 is tracker_in_call_1, "Same tracker instance must be reused across samples"
