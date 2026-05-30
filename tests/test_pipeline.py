# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pipeline.py::run_pipeline().

Covers: dry_run, MCQ pre-step ordering, stage isolation, tracking exception handling,
and MCQ stage routing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.endpoints import EndpointsConfig, LlmEndpointConfig, VlmEndpointConfig
from al_utils.schema.mcq import McqGenerationConfig, WindowMetadataExtractionConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from al_utils.schema.sr import SuperResolutionConfig
from al_utils.schema.tracking import DetectionAndTrackingConfig
from al_utils.schema.vlm_json import VlmJsonConfig
from detection_and_tracking.base import TrackingResult
from mcq_generation.base import MCQResult
from pipeline import run_pipeline
from vlm_json.base import VlmJsonResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    *,
    sr: bool = False,
    tracking: bool = False,
    vlm: bool = False,
    mcq: bool = False,
    mcq_mode: str = "question-driven-vlm-llm",
    empty_output_policy: str = "warn",
) -> PipelineConfig:
    sr_cfg = SuperResolutionConfig(enabled=True) if sr else None
    dt_cfg = DetectionAndTrackingConfig(enabled=True, save_video=False, save_video_red_id=False) if tracking else None
    vlm_cfg = VlmJsonConfig(enabled=True) if vlm else None
    wme = WindowMetadataExtractionConfig(single_window=True) if mcq else None
    mcq_cfg = McqGenerationConfig(enabled=True, mode=mcq_mode, window_metadata_extraction=wme) if mcq else None

    meta_sidecar = "fake_meta.json" if (mcq and mcq_mode == "metadata-llm") else None

    return PipelineConfig(
        pipeline=PipelineSettings(empty_output_policy=empty_output_policy),
        data=[
            SampleConfig(
                inputs=SampleInputsConfig(
                    video_path="v.mp4",
                    metadata_json_path=meta_sidecar,
                ),
                output=SampleOutputConfig(out_dir="out"),
            )
        ],
        endpoints=EndpointsConfig(
            vlm=VlmEndpointConfig(url="http://vlm/v1", model="fake-vlm"),
            llm=LlmEndpointConfig(url="http://llm/v1", model="fake-llm"),
        ),
        super_resolution=sr_cfg,
        detection_and_tracking=dt_cfg,
        vlm_json=vlm_cfg,
        mcq_generation=mcq_cfg,
    )


def _sample(video_path: str) -> SampleConfig:
    return SampleConfig(
        inputs=SampleInputsConfig(video_path=video_path),
        output=SampleOutputConfig(out_dir=str(Path(video_path).parent / "out")),
    )


def _run(
    tmp_path: Path,
    config: PipelineConfig,
    *,
    sr_runner=None,
    det_tracker=None,
    vlm_json_gen=None,
    mcq_gen=None,
    dry_run: bool = False,
) -> int:
    video = tmp_path / "video.mp4"
    if not dry_run:
        video.write_bytes(b"fake")
    sample = _sample(str(video))
    return run_pipeline(
        sample,
        config,
        sr_runner=sr_runner,
        det_tracker=det_tracker,
        vlm_json_gen=vlm_json_gen,
        mcq_gen=mcq_gen,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test_pipeline"),
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_stages_none_returns_zero(tmp_path: Path) -> None:
    rc = _run(tmp_path, _config())
    assert rc == 0


def test_dry_run_no_stage_called(tmp_path: Path) -> None:
    mock_tracker = MagicMock()
    mock_vlm = MagicMock()
    mock_mcq = MagicMock()

    rc = _run(
        tmp_path,
        _config(tracking=True, vlm=True, mcq=True),
        det_tracker=mock_tracker,
        vlm_json_gen=mock_vlm,
        mcq_gen=mock_mcq,
        dry_run=True,
    )

    assert rc == 0
    mock_tracker.run.assert_not_called()
    mock_vlm.generate.assert_not_called()
    mock_mcq.generate.assert_not_called()
    mock_mcq.run_pre_step.assert_not_called()


def test_mcq_pre_step_called_before_tracking(tmp_path: Path) -> None:
    """run_pre_step() is stage 0 — called before det_tracker.run()."""
    call_order: list[str] = []

    mock_tracker = MagicMock()
    mock_tracker.run.side_effect = lambda *a, **kw: call_order.append("tracking") or TrackingResult(success=True)

    mock_mcq = MagicMock()
    mock_mcq.run_pre_step.side_effect = lambda *a, **kw: call_order.append("pre_step")

    _run(tmp_path, _config(tracking=True, mcq=True), det_tracker=mock_tracker, mcq_gen=mock_mcq)

    assert call_order.index("pre_step") < call_order.index("tracking")


def test_mcq_generate_does_not_route_vlm_json_sidecars(tmp_path: Path) -> None:
    """Current MCQ modes are window/question-bank driven, not event-sidecar driven."""
    events = tmp_path / "events.json"
    video_json = tmp_path / "video.json"
    events.write_text("{}")
    video_json.write_text("{}")

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VlmJsonResult(success=True, events_json=events, video_json=video_json)

    mock_mcq = MagicMock()

    _run(
        tmp_path,
        _config(vlm=True, mcq=True, mcq_mode="question-driven-vlm-llm"),
        vlm_json_gen=mock_vlm,
        mcq_gen=mock_mcq,
    )

    mock_mcq.generate.assert_called_once()
    _, kwargs = mock_mcq.generate.call_args
    assert "events_json" not in kwargs
    assert "video_json" not in kwargs


def test_vlm_fallback_success_false_can_continue_under_warn_policy(tmp_path: Path) -> None:
    """empty_output_policy=warn lets MCQ modes continue without VLM JSON output."""

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VlmJsonResult(success=False)
    mock_mcq = MagicMock()

    rc = _run(
        tmp_path,
        _config(vlm=True, mcq=True, mcq_mode="question-driven-vlm-llm", empty_output_policy="warn"),
        vlm_json_gen=mock_vlm,
        mcq_gen=mock_mcq,
    )

    assert rc == 0
    mock_mcq.generate.assert_called_once()
    status = json.loads((tmp_path / "out" / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["degraded"] is True
    assert status["degraded_stages"] == ["vlm_json"]
    assert status["fallbacks"][0]["stage"] == "vlm_json"
    assert status["fallbacks"][0]["reason"] == "reported_failure"
    assert status["stages"]["vlm_json"]["degraded"] is True


def test_vlm_exception_warn_preserves_exception_failure_reason(tmp_path: Path) -> None:
    """Exception paths keep the exception class in stage status."""

    mock_vlm = MagicMock()
    mock_vlm.generate.side_effect = ValueError("bad vlm response")
    mock_mcq = MagicMock()

    rc = _run(
        tmp_path,
        _config(vlm=True, mcq=True, mcq_mode="question-driven-vlm-llm", empty_output_policy="warn"),
        vlm_json_gen=mock_vlm,
        mcq_gen=mock_mcq,
    )

    assert rc == 0
    mock_mcq.generate.assert_called_once()
    status = json.loads((tmp_path / "out" / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["fallbacks"][0]["stage"] == "vlm_json"
    assert status["fallbacks"][0]["reason"] == "reported_failure"
    assert status["stages"]["vlm_json"]["failure_reason"] == "ValueError"


def test_vlm_fallback_success_false_fails_under_fail_policy(tmp_path: Path) -> None:
    """empty_output_policy=fail keeps reported VLM failure as a hard failure."""
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VlmJsonResult(success=False)

    with pytest.raises(SystemExit):
        _run(
            tmp_path,
            _config(vlm=True, mcq=True, mcq_mode="question-driven-vlm-llm", empty_output_policy="fail"),
            vlm_json_gen=mock_vlm,
            mcq_gen=MagicMock(),
        )


def test_required_vlm_failure_without_mcq_fails(tmp_path: Path) -> None:
    """VLM-only runs keep the VLM JSON stage required."""
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VlmJsonResult(success=False)

    with pytest.raises(SystemExit):
        _run(
            tmp_path,
            _config(vlm=True, mcq=False, empty_output_policy="warn"),
            vlm_json_gen=mock_vlm,
            mcq_gen=None,
        )


def test_tracking_exception_warns_and_continues(tmp_path: Path) -> None:
    """Tracking raises → warning logged, pipeline continues (policy=warn)."""
    mock_tracker = MagicMock()
    mock_tracker.run.side_effect = RuntimeError("GPU OOM")
    mock_mcq = MagicMock()

    rc = _run(
        tmp_path,
        _config(tracking=True, mcq=True, empty_output_policy="warn"),
        det_tracker=mock_tracker,
        mcq_gen=mock_mcq,
    )

    assert rc == 0
    mock_mcq.generate.assert_called_once()
    status = json.loads((tmp_path / "out" / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["degraded"] is True
    assert status["degraded_stages"] == ["tracking"]
    assert status["fallbacks"][0]["stage"] == "tracking"
    assert status["fallbacks"][0]["reason"] == "reported_failure"
    assert status["stages"]["tracking"]["degraded"] is True
    assert status["stages"]["tracking"]["failure_reason"] == "RuntimeError"


def test_tracking_reported_failure_warn_writes_degraded_pipeline_status(tmp_path: Path) -> None:
    """Tracking success=False is degraded, not skipped or fatal, under warn policy."""
    mock_tracker = MagicMock()
    mock_tracker.run.return_value = TrackingResult(success=False)
    mock_mcq = MagicMock()

    rc = _run(
        tmp_path,
        _config(tracking=True, mcq=True, empty_output_policy="warn"),
        det_tracker=mock_tracker,
        mcq_gen=mock_mcq,
    )

    assert rc == 0
    mock_mcq.generate.assert_called_once()
    status = json.loads((tmp_path / "out" / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed_degraded"
    assert status["degraded"] is True
    assert status["degraded_stages"] == ["tracking"]
    assert status["fallbacks"][0]["stage"] == "tracking"
    assert status["fallbacks"][0]["reason"] == "reported_failure"
    assert status["stages"]["tracking"]["status"] == "failed"
    assert status["stages"]["tracking"]["degraded"] is True


def test_only_mcq_no_other_stages(tmp_path: Path) -> None:
    """MCQ generate() is called even when tracker and VLM are absent."""
    mock_mcq = MagicMock()
    _run(tmp_path, _config(mcq=True), mcq_gen=mock_mcq)

    mock_mcq.run_pre_step.assert_called_once()
    mock_mcq.generate.assert_called_once()


def test_vlm_only_no_mcq(tmp_path: Path) -> None:
    """VLM generate() is called; MCQ is not called when mcq_gen=None."""
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VlmJsonResult(success=True)

    rc = _run(tmp_path, _config(vlm=True), vlm_json_gen=mock_vlm)

    assert rc == 0
    mock_vlm.generate.assert_called_once()


def test_metadata_json_sidecar_forwarded(tmp_path: Path) -> None:
    """metadata_json from inputs is forwarded to mcq_gen.generate()."""
    meta = tmp_path / "meta.json"
    meta.write_text("{}")

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    sample = SampleConfig(
        inputs=SampleInputsConfig(video_path=str(video), metadata_json_path=str(meta)),
        output=SampleOutputConfig(out_dir=str(tmp_path / "out")),
    )
    mock_mcq = MagicMock()

    run_pipeline(
        sample,
        _config(mcq=True, mcq_mode="metadata-llm"),
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    _, kwargs = mock_mcq.generate.call_args
    assert kwargs["metadata_json"] == meta


def test_per_stage_log_files_created(tmp_path: Path) -> None:
    """tracking.log, vlm_json.log, mcq.log are created under log_dir when stages run."""
    log_dir = tmp_path / "logs"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    events = tmp_path / "events.json"
    events.write_text("{}")

    mock_tracker = MagicMock()
    mock_tracker.run.return_value = TrackingResult(success=True)

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VlmJsonResult(success=True, events_json=events)

    mock_mcq = MagicMock()

    sample = SampleConfig(
        inputs=SampleInputsConfig(video_path=str(video)),
        output=SampleOutputConfig(out_dir=str(tmp_path / "out")),
    )

    run_pipeline(
        sample,
        _config(tracking=True, vlm=True, mcq=True, mcq_mode="question-driven-vlm-llm"),
        sr_runner=None,
        det_tracker=mock_tracker,
        vlm_json_gen=mock_vlm,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=log_dir,
        logger=logging.getLogger("test"),
    )

    assert (log_dir / "tracking.log").exists(), "tracking.log not created"
    assert (log_dir / "vlm_json.log").exists(), "vlm_json.log not created"
    assert (log_dir / "mcq.log").exists(), "mcq.log not created"


def test_raw_symlinks_to_input_when_no_sr(tmp_path: Path) -> None:
    """When SR does not run, raw/<stem>.<ext> symlinks to the input video."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    sample = _sample(str(video))

    run_pipeline(
        sample,
        _config(),
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=out_dir,
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    link = out_dir / "raw" / "video.mp4"
    assert link.is_symlink()
    assert link.resolve() == video.resolve()


def test_pipeline_status_records_non_degraded_success(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    run_pipeline(
        _sample(str(video)),
        _config(),
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=out_dir,
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    status = json.loads((out_dir / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["degraded"] is False
    assert status["status"] == "completed"
    assert status["fallbacks"] == []
    assert status["degraded_stages"] == []


def test_sr_missing_output_warn_writes_degraded_pipeline_status(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    mock_sr = MagicMock()
    mock_tracker = MagicMock()
    mock_tracker.run.return_value = TrackingResult(success=True)

    rc = run_pipeline(
        _sample(str(video)),
        _config(sr=True, tracking=True, empty_output_policy="warn"),
        sr_runner=mock_sr,
        det_tracker=mock_tracker,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=out_dir,
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    assert rc == 0
    tracking_input, _ = mock_tracker.run.call_args.args
    assert tracking_input == video

    status = json.loads((out_dir / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["degraded"] is True
    assert status["status"] == "completed_degraded"
    assert status["degraded_stages"] == ["super_resolution"]
    assert status["fallbacks"][0]["stage"] == "super_resolution"
    assert status["fallbacks"][0]["reason"] == "missing_output"
    assert status["fallbacks"][0]["fallback_input"] == str(video)
    assert status["stages"]["super_resolution"]["status"] == "failed"
    assert status["stages"]["super_resolution"]["degraded"] is True
    assert status["stages"]["tracking"]["status"] == "completed"


def test_sr_missing_output_fail_policy_aborts_without_status(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit):
        run_pipeline(
            _sample(str(video)),
            _config(sr=True, empty_output_policy="fail"),
            sr_runner=MagicMock(),
            det_tracker=None,
            vlm_json_gen=None,
            mcq_gen=None,
            config_dir=tmp_path,
            repo_root=tmp_path,
            out_dir=out_dir,
            log_dir=tmp_path / "logs",
            logger=logging.getLogger("test"),
        )

    assert not (out_dir / "sidecars" / "pipeline_status.json").exists()


def test_raw_copy_for_staged_remote_input(tmp_path: Path) -> None:
    """Remote-staged inputs are copied into raw/ so cleanup cannot break the scene."""
    staged = tmp_path / "downloaded" / "video.mp4"
    staged.parent.mkdir()
    staged.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    run_pipeline(
        _sample(str(staged)),
        _config(),
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=out_dir,
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
        copy_raw=True,
    )

    raw = out_dir / "raw" / "video.mp4"
    assert raw.is_file()
    assert not raw.is_symlink()
    assert raw.read_bytes() == b"fake"


def test_vlm_input_prefers_tracking_red_id_over_sr_and_input(tmp_path: Path) -> None:
    """When tracking emits a red-id overlay, VLM consumes it in preference to SR / input.

    This preserves the desired routing where the VLM gets the track-ID-annotated
    video so it can reason about per-instance events.
    """
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    # Pre-create the red-id overlay at its canonical sidecar location.
    red_id = out_dir / "sidecars" / "video_tracking_red_id.mp4"
    red_id.parent.mkdir(parents=True, exist_ok=True)
    red_id.write_bytes(b"red")

    mock_tracker = MagicMock()
    mock_tracker.run.return_value = TrackingResult(success=True, tracking_video_red_id=red_id)

    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VlmJsonResult(success=True)

    run_pipeline(
        _sample(str(video)),
        _config(tracking=True, vlm=True),
        sr_runner=None,
        det_tracker=mock_tracker,
        vlm_json_gen=mock_vlm,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=out_dir,
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    mock_vlm.generate.assert_called_once()
    args, _ = mock_vlm.generate.call_args
    assert args[0] == red_id


def test_mcq_reported_failure_warn_writes_degraded_pipeline_status(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    mock_mcq = MagicMock()
    mock_mcq.generate.return_value = MCQResult(success=False)

    rc = run_pipeline(
        _sample(str(video)),
        _config(mcq=True, empty_output_policy="warn"),
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=out_dir,
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    assert rc == 0
    status = json.loads((out_dir / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["degraded"] is True
    assert status["degraded_stages"] == ["mcq_generation"]
    assert status["fallbacks"][0]["stage"] == "mcq_generation"
    assert status["fallbacks"][0]["reason"] == "reported_failure"
    assert status["stages"]["mcq_generation"]["degraded"] is True


def test_mcq_exception_warn_preserves_exception_failure_reason(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    mock_mcq = MagicMock()
    mock_mcq.generate.side_effect = RuntimeError("llm unavailable")

    rc = run_pipeline(
        _sample(str(video)),
        _config(mcq=True, empty_output_policy="warn"),
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=out_dir,
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    assert rc == 0
    status = json.loads((out_dir / "sidecars" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["fallbacks"][0]["stage"] == "mcq_generation"
    assert status["fallbacks"][0]["reason"] == "reported_failure"
    assert status["stages"]["mcq_generation"]["failure_reason"] == "RuntimeError"


def test_pipeline_aborts_when_input_missing(tmp_path: Path) -> None:
    """Missing input media → SystemExit on stage input validation."""
    sample = _sample(str(tmp_path / "missing.mp4"))
    with pytest.raises(SystemExit):
        run_pipeline(
            sample,
            _config(),
            sr_runner=None,
            det_tracker=None,
            vlm_json_gen=None,
            mcq_gen=None,
            config_dir=tmp_path,
            repo_root=tmp_path,
            out_dir=tmp_path / "out",
            log_dir=tmp_path / "logs",
            logger=logging.getLogger("test"),
        )


def test_sr_image_input_uses_png_sidecar_output(tmp_path: Path) -> None:
    """Image SR outputs use a stable PNG sidecar regardless of input extension."""
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake-jpg")
    sample = _sample(str(image))
    mock_sr = MagicMock()
    mock_sr.run.side_effect = lambda _inp, out, **_kwargs: Path(out).write_bytes(b"fake-png")

    rc = run_pipeline(
        sample,
        _config(sr=True),
        sr_runner=mock_sr,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=None,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    assert rc == 0
    sr_output = tmp_path / "out" / "sidecars" / "sr_output.png"
    assert sr_output.exists()
    mock_sr.run.assert_called_once()
    assert mock_sr.run.call_args.args[1] == sr_output
    assert (tmp_path / "out" / "raw" / "frame.png").is_symlink()


def test_non_event_driven_mcq_runs_on_image_input(tmp_path: Path) -> None:
    """metadata-llm + image input → MCQ generator IS invoked (no events needed)."""
    image = tmp_path / "frame.png"
    image.write_bytes(b"fake-png")
    meta = tmp_path / "meta.json"
    meta.write_text("{}")

    sample = SampleConfig(
        inputs=SampleInputsConfig(video_path=str(image), metadata_json_path=str(meta)),
        output=SampleOutputConfig(out_dir=str(tmp_path / "out")),
    )
    mock_mcq = MagicMock()

    rc = run_pipeline(
        sample,
        _config(mcq=True, mcq_mode="metadata-llm"),
        sr_runner=None,
        det_tracker=None,
        vlm_json_gen=None,
        mcq_gen=mock_mcq,
        config_dir=tmp_path,
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        log_dir=tmp_path / "logs",
        logger=logging.getLogger("test"),
    )

    assert rc == 0
    mock_mcq.generate.assert_called_once()
