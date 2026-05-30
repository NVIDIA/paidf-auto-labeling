# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pipeline as pipeline_mod
import pytest
from al_utils.io import read_json
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from al_utils.schema.sr import SuperResolutionConfig
from detection_and_tracking.base import BaseTracker, TrackingResult
from sr_runner.base import BaseSuperResolver
from vlm_json.base import BaseVlmJsonGenerator, VlmJsonResult

run_pipeline = pipeline_mod.run_pipeline


def _make_config(
    tmp_path: Path,
    *,
    sr_enabled: bool = False,
    empty_output_policy: str = "warn",
    vlm_video_path: str = None,
) -> tuple[SampleConfig, PipelineConfig]:
    inputs_kwargs: dict = {"video_path": str(tmp_path / "video.mp4")}
    if vlm_video_path:
        inputs_kwargs["vlm_video_path"] = vlm_video_path

    sample = SampleConfig(
        inputs=SampleInputsConfig(**inputs_kwargs),
        output=SampleOutputConfig(out_dir=str(tmp_path / "out")),
    )
    config = PipelineConfig(
        pipeline=PipelineSettings(empty_output_policy=empty_output_policy),
        data=[sample],
        super_resolution=SuperResolutionConfig(enabled=True) if sr_enabled else None,
    )
    return sample, config


# ---------------------------------------------------------------------------


def test_sr_empty_outputs_is_warning(monkeypatch, tmp_path):
    """SR runs but produces no output file → pipeline warns and returns 0."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")

    sample, config = _make_config(tmp_path, sr_enabled=True)

    mock_sr = MagicMock(spec=BaseSuperResolver)
    # sr_runner.run() returns without creating the file.
    mock_sr.run.return_value = None

    rc = run_pipeline(
        sample,
        config,
        sr_runner=mock_sr,
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
    assert not (tmp_path / "sr.mp4").exists(), "mock sr_runner must not produce output"


def test_sr_unsupported_video_decoder_fallback_is_actionable(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")

    sample, config = _make_config(tmp_path, sr_enabled=True)

    def fake_sr_run(input_video, output_video, log_dir=None, **kwargs):
        assert log_dir is not None
        Path(log_dir, "sr.log").write_text(
            "h264_cuvid CUDA_ERROR_NOT_SUPPORTED cuvid decode callback error\n",
            encoding="utf-8",
        )

    mock_sr = MagicMock(spec=BaseSuperResolver)
    mock_sr.run.side_effect = fake_sr_run

    rc = run_pipeline(
        sample,
        config,
        sr_runner=mock_sr,
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
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["reason"] == "unsupported_video_decoder"
    assert "Unsupported video decoder" in fallback["message"]
    assert "could not read this video format" in fallback["message"]
    assert "hardware decoder does not support" in fallback["message"]
    assert "stage is skipped" in fallback["message"]
    assert "original input or the latest successfully generated media" in fallback["message"]


def test_sr_mpeg4_decode_failure_fallback_is_connected(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")

    sample, config = _make_config(tmp_path, sr_enabled=True)

    def fake_sr_run(input_video, output_video, log_dir=None, **kwargs):
        assert log_dir is not None
        Path(log_dir, "sr.log").write_text(
            "mpeg4 error while decoding stream #0:0: Invalid data found when processing input\n",
            encoding="utf-8",
        )

    mock_sr = MagicMock(spec=BaseSuperResolver)
    mock_sr.run.side_effect = fake_sr_run

    rc = run_pipeline(
        sample,
        config,
        sr_runner=mock_sr,
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
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "super_resolution"
    assert fallback["reason"] == "video_decode_failed"
    assert "Video decode failure" in fallback["message"]


def test_sr_empty_outputs_can_fail(monkeypatch, tmp_path):
    """empty_output_policy=fail → SystemExit when SR produces no output."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")

    sample, config = _make_config(tmp_path, sr_enabled=True, empty_output_policy="fail")

    mock_sr = MagicMock(spec=BaseSuperResolver)
    mock_sr.run.return_value = None  # returns without creating file

    with pytest.raises(SystemExit):
        run_pipeline(
            sample,
            config,
            sr_runner=mock_sr,
            det_tracker=None,
            vlm_json_gen=None,
            mcq_gen=None,
            config_dir=tmp_path,
            repo_root=tmp_path,
            out_dir=tmp_path / "out",
            log_dir=tmp_path / "logs",
            dry_run=False,
        )


def test_tracking_failure_warns_when_vlm_disabled(tmp_path):
    """Tracking failure → warns and returns 0 when no downstream dependency."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.return_value = TrackingResult(success=False)

    rc = run_pipeline(
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
    assert rc == 0


def test_tracking_unsupported_video_decoder_fallback_is_actionable(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)

    def fake_tracking_run(input_video, out_dir):
        log_dir = Path(out_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "tracking.log").write_text(
            "h264_cuvid CUDA_ERROR_NOT_SUPPORTED cuvid decode callback error\n",
            encoding="utf-8",
        )
        raise RuntimeError("decode failed")

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.side_effect = fake_tracking_run

    rc = run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=mock_tracker,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "out" / "logs",
        dry_run=False,
    )

    assert rc == 0
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "tracking"
    assert fallback["reason"] == "unsupported_video_decoder"
    assert "Unsupported video decoder" in fallback["message"]
    assert "could not read this video format" in fallback["message"]
    assert status["stages"]["tracking"]["failure_reason"] == "unsupported_video_decoder"


def test_tracking_mpeg4_decode_failure_fallback_is_connected(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)

    def fake_tracking_run(input_video, out_dir):
        log_dir = Path(out_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "tracking.log").write_text(
            "mpeg4 error while decoding stream #0:0: Invalid data found when processing input\n",
            encoding="utf-8",
        )
        raise RuntimeError("decode failed")

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.side_effect = fake_tracking_run

    rc = run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=mock_tracker,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "out" / "logs",
        dry_run=False,
    )

    assert rc == 0
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "tracking"
    assert fallback["reason"] == "video_decode_failed"
    assert status["stages"]["tracking"]["failure_reason"] == "video_decode_failed"


def test_tracking_failure_can_fail_when_policy_fail(tmp_path):
    """empty_output_policy=fail → SystemExit when tracking fails."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path, empty_output_policy="fail")

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.return_value = TrackingResult(success=False)

    with pytest.raises(SystemExit):
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


def test_tracking_exception_warns_by_default(tmp_path):
    """Tracking exception → warns and continues when policy=warn."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path, empty_output_policy="warn")

    mock_tracker = MagicMock(spec=BaseTracker)
    mock_tracker.run.side_effect = RuntimeError("GPU OOM")

    rc = run_pipeline(
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
    assert rc == 0


def test_vlm_input_falls_back_to_input_when_explicit_path_missing(tmp_path):
    """VLM uses original input video when explicit vlm_video_path doesn't exist."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    missing = str(tmp_path / "nonexistent.mp4")

    sample, config = _make_config(tmp_path, vlm_video_path=missing)

    mock_vlm = MagicMock(spec=BaseVlmJsonGenerator)
    mock_vlm.generate.return_value = VlmJsonResult(success=True)

    run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=mock_vlm,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        dry_run=False,
    )

    call_args = mock_vlm.generate.call_args
    vlm_in = call_args[0][0]
    assert Path(vlm_in) == video


def test_vlm_mpeg4_decode_failure_fallback_is_connected(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)
    log_dir = tmp_path / "logs"

    def fake_vlm_generate(input_video, out_dir):
        (log_dir / "vlm_json.log").write_text(
            "mpeg4 error while decoding stream #0:0: Invalid data found when processing input\n",
            encoding="utf-8",
        )
        raise RuntimeError("decode failed")

    mock_vlm = MagicMock(spec=BaseVlmJsonGenerator)
    mock_vlm.generate.side_effect = fake_vlm_generate

    rc = run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=mock_vlm,
        mcq_gen=MagicMock(),
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=log_dir,
        dry_run=False,
    )

    assert rc == 0
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "vlm_json"
    assert fallback["reason"] == "video_decode_failed"
    assert status["stages"]["vlm_json"]["failure_reason"] == "video_decode_failed"


def test_required_vlm_decode_failure_warns_without_mcq(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)
    log_dir = tmp_path / "logs"

    def fake_vlm_generate(input_video, out_dir):
        (log_dir / "vlm_json.log").write_text(
            "h264_cuvid CUDA_ERROR_NOT_SUPPORTED cuvid decode callback error\n",
            encoding="utf-8",
        )
        raise RuntimeError("decode failed")

    mock_vlm = MagicMock(spec=BaseVlmJsonGenerator)
    mock_vlm.generate.side_effect = fake_vlm_generate

    rc = run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=mock_vlm,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=log_dir,
        dry_run=False,
    )

    assert rc == 0
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "vlm_json"
    assert fallback["reason"] == "unsupported_video_decoder"
    assert status["stages"]["vlm_json"]["failure_reason"] == "unsupported_video_decoder"


def test_mcq_unsupported_video_decoder_fallback_is_actionable(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)
    log_dir = tmp_path / "logs"

    def fake_mcq_generate(input_video, out_dir, *, metadata_json=None):
        (log_dir / "mcq.log").write_text(
            "h264_cuvid CUDA_ERROR_NOT_SUPPORTED cuvid decode callback error\n",
            encoding="utf-8",
        )
        raise RuntimeError("decode failed")

    mock_mcq = MagicMock()
    mock_mcq.generate.side_effect = fake_mcq_generate

    rc = run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=log_dir,
        dry_run=False,
    )

    assert rc == 0
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "mcq_generation"
    assert fallback["reason"] == "unsupported_video_decoder"
    assert "Unsupported video decoder" in fallback["message"]
    assert "could not read this video format" in fallback["message"]
    assert status["stages"]["mcq_generation"]["failure_reason"] == "unsupported_video_decoder"


def test_mcq_unsupported_video_decoder_from_exception_is_actionable(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)

    mock_mcq = MagicMock()
    mock_mcq.generate.side_effect = RuntimeError(
        "h264_cuvid CUDA_ERROR_NOT_SUPPORTED: unsupported H.264 input for this image"
    )

    rc = run_pipeline(
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

    assert rc == 0
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "mcq_generation"
    assert fallback["reason"] == "unsupported_video_decoder"
    assert status["stages"]["mcq_generation"]["failure_reason"] == "unsupported_video_decoder"


def test_mcq_mpeg4_decode_failure_fallback_is_connected(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    sample, config = _make_config(tmp_path)
    log_dir = tmp_path / "logs"

    def fake_mcq_generate(input_video, out_dir, *, metadata_json=None):
        (log_dir / "mcq.log").write_text(
            "mpeg4 error while decoding stream #0:0: Invalid data found when processing input\n",
            encoding="utf-8",
        )
        raise RuntimeError("decode failed")

    mock_mcq = MagicMock()
    mock_mcq.generate.side_effect = fake_mcq_generate

    rc = run_pipeline(
        sample,
        config,
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=log_dir,
        dry_run=False,
    )

    assert rc == 0
    status = read_json(tmp_path / "out" / "sidecars" / "pipeline_status.json")
    fallback = status["fallbacks"][0]
    assert fallback["stage"] == "mcq_generation"
    assert fallback["reason"] == "video_decode_failed"
    assert status["stages"]["mcq_generation"]["failure_reason"] == "video_decode_failed"
