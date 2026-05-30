# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pipeline as pipeline_mod
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from al_utils.schema.sr import SuperResolutionConfig
from detection_and_tracking.base import BaseTracker, TrackingResult
from sr_runner.base import BaseSuperResolver

run_pipeline = pipeline_mod.run_pipeline


def _make_config(tmp_path, *, sr_enabled=False, vlm_video_path=None):
    inputs_kwargs: dict = {"video_path": str(tmp_path / "video.mp4")}
    if vlm_video_path:
        inputs_kwargs["vlm_video_path"] = vlm_video_path

    sample = SampleConfig(
        inputs=SampleInputsConfig(**inputs_kwargs),
        output=SampleOutputConfig(out_dir=str(tmp_path / "out")),
    )
    config = PipelineConfig(
        pipeline=PipelineSettings(),
        data=[sample],
        super_resolution=SuperResolutionConfig(enabled=True) if sr_enabled else None,
    )
    return sample, config


def test_pipeline_tracking_uses_sr_output_when_sr_ran(tmp_path):
    """When SR runs and produces an output file, tracking receives that file."""
    video = tmp_path / "video.mp4"
    video.touch()

    sample, config = _make_config(tmp_path, sr_enabled=True)

    # The SR runner materializes the file at the path it was asked to write to.
    def _fake_sr_run(input_video, sr_out, **_kwargs):
        Path(sr_out).write_bytes(b"")

    mock_sr = MagicMock(spec=BaseSuperResolver)
    mock_sr.run.side_effect = _fake_sr_run

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.return_value = TrackingResult(success=True)

    run_pipeline(
        sample,
        config,
        sr_runner=mock_sr,
        det_tracker=mock_tracker,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        dry_run=False,
    )

    call_args = mock_tracker.run.call_args
    tracking_in = Path(call_args[0][0])
    expected_sr_out = tmp_path / "out" / "sidecars" / "sr_output.mp4"
    assert tracking_in == expected_sr_out
    assert not str(tracking_in).startswith("s3://")


def test_pipeline_tracking_uses_input_when_sr_disabled(tmp_path):
    """When SR is disabled, tracking receives the original input video."""
    video = tmp_path / "video.mp4"
    video.touch()

    sample, config = _make_config(tmp_path)

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.return_value = TrackingResult(success=True)

    run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=mock_tracker,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        dry_run=False,
    )

    call_args = mock_tracker.run.call_args
    tracking_in = Path(call_args[0][0])
    assert tracking_in == video
