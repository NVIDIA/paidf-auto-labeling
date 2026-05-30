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
from mcq_generation.base import BaseMCQGenerator, MCQResult
from sr_runner.base import BaseSuperResolver
from vlm_json.base import BaseVlmJsonGenerator, VlmJsonResult

run_pipeline = pipeline_mod.run_pipeline


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    *,
    sr_enabled: bool = False,
    tracking_enabled: bool = False,
    vlm_enabled: bool = False,
    mcq_enabled: bool = False,
    mcq_mode: str = "question-driven-vlm-llm",
    empty_output_policy: str = "warn",
    vlm_video_path: str = None,
    metadata_json_path: str = None,
) -> tuple[SampleConfig, PipelineConfig]:
    """Build minimal SampleConfig + PipelineConfig for pipeline unit tests."""
    inputs_kwargs: dict = {"video_path": str(tmp_path / "video.mp4")}
    if vlm_video_path:
        inputs_kwargs["vlm_video_path"] = vlm_video_path
    if metadata_json_path:
        inputs_kwargs["metadata_json_path"] = metadata_json_path

    sample = SampleConfig(
        inputs=SampleInputsConfig(**inputs_kwargs),
        output=SampleOutputConfig(out_dir=str(tmp_path / "out")),
    )

    sr_cfg = None
    if sr_enabled:
        sr_cfg = SuperResolutionConfig(enabled=True)

    config = PipelineConfig(
        pipeline=PipelineSettings(
            empty_output_policy=empty_output_policy,
        ),
        data=[sample],
        super_resolution=sr_cfg,
    )
    return sample, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_dry_run_does_not_execute(tmp_path):
    """Dry-run must not call any stage methods."""
    (tmp_path / "video.mp4").touch()
    sample, config = _make_config(tmp_path)

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_vlm = MagicMock(spec=BaseVlmJsonGenerator)
    mock_mcq = MagicMock(spec=BaseMCQGenerator)

    run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=mock_tracker,
        vlm_json_gen=mock_vlm,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        dry_run=True,
    )

    mock_tracker.run.assert_not_called()
    mock_vlm.generate.assert_not_called()
    mock_mcq.generate.assert_not_called()
    mock_mcq.run_pre_step.assert_not_called()


def test_pipeline_stages_called_when_objects_provided(tmp_path):
    """Provided stage objects should have their methods called."""
    (tmp_path / "video.mp4").touch()
    sample, config = _make_config(tmp_path)

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.return_value = TrackingResult(success=True)

    mock_vlm = MagicMock(spec=BaseVlmJsonGenerator)
    mock_vlm.generate.return_value = VlmJsonResult(success=True)

    mock_mcq = MagicMock(spec=BaseMCQGenerator)
    mock_mcq.generate.return_value = MCQResult(success=True)

    run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=mock_tracker,
        vlm_json_gen=mock_vlm,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        dry_run=False,
    )

    mock_tracker.run.assert_called_once()
    mock_vlm.generate.assert_called_once()
    mock_mcq.run_pre_step.assert_called_once()
    mock_mcq.generate.assert_called_once()


def test_pipeline_none_stages_skipped(tmp_path):
    """None stage objects must not cause errors; pipeline returns 0."""
    (tmp_path / "video.mp4").touch()
    sample, config = _make_config(tmp_path)

    rc = run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        dry_run=False,
    )
    assert rc == 0


def test_pipeline_tracking_uses_sr_output_when_available(tmp_path):
    """Tracking input should be the SR output file when SR ran successfully."""
    (tmp_path / "video.mp4").touch()

    sample, config = _make_config(tmp_path, sr_enabled=True)

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
    assert tracking_in == tmp_path / "out" / "sidecars" / "sr_output.mp4"


def test_pipeline_tracking_falls_back_to_input_when_sr_missing(tmp_path):
    """Tracking input falls back to original input if SR produced no output."""
    video = tmp_path / "video.mp4"
    video.touch()

    sample, config = _make_config(tmp_path, sr_enabled=True)

    mock_sr = MagicMock(spec=BaseSuperResolver)
    mock_sr.run.return_value = None  # SR produces nothing

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
    assert tracking_in == video


def test_pipeline_mcq_pre_step_called_before_generate(tmp_path):
    """run_pre_step() should be called before generate()."""
    (tmp_path / "video.mp4").touch()
    sample, config = _make_config(tmp_path)

    call_order = []
    mock_mcq = MagicMock(spec=BaseMCQGenerator)
    mock_mcq.run_pre_step.side_effect = lambda *a, **kw: call_order.append("pre_step")
    mock_mcq.generate.side_effect = lambda *a, **kw: (call_order.append("generate"), MCQResult(success=True))[1]

    run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        dry_run=False,
    )

    assert call_order.index("pre_step") < call_order.index("generate")
