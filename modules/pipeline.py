# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run SR → Tracking → VLM JSON → MCQ generation.

Receives pre-built stage objects from cli.py — models are loaded ONCE before
the sample loop. Each stage object is called via a direct Python method; SR
remains a ``torchrun`` subprocess.

The per-sample ``out_dir`` is the DAFT scene root. Stages derive their target
paths from :func:`daft_export.paths.scene_paths` — no per-file path plumbing.

Stage objects are None when the stage is disabled.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from al_utils.common import ensure_dir, stage_log_file
from al_utils.io import write_json
from al_utils.media_decode import classify_decode_fallback
from al_utils.media_paths import is_image_path
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig
from daft_export.common import set_scene_media_id
from daft_export.paths import ensure_scene_skeleton, scene_paths
from detection_and_tracking.base import BaseTracker, TrackingResult
from mcq_generation.base import BaseMCQGenerator, MCQResult
from sr_runner.base import BaseSuperResolver
from vlm_json.base import BaseVlmJsonGenerator, VlmJsonResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_policy_value(raw: Any, default: str) -> str:
    v = str(raw or "").strip().lower()
    if v in {"warn", "fail"}:
        return v
    return default


def _log_plan(logger: Optional[logging.Logger], msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _append_pipeline_log(pipeline_log: Optional[Path], msg: str) -> None:
    if pipeline_log is None:
        return
    try:
        pipeline_log.parent.mkdir(parents=True, exist_ok=True)
        with pipeline_log.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip("\n") + "\n")
    except Exception as exc:
        logging.getLogger(__name__).debug("failed to append pipeline log %s: %s", pipeline_log, exc)


def _record_stage_fallback(
    *,
    fallback_records: list[dict[str, Any]],
    stage_status: dict[str, dict[str, Any]],
    stage: str,
    policy: str,
    reason: str,
    message: str,
    fallback_input: Path,
    downstream_input: Path,
    expected_output: Optional[Path] = None,
) -> None:
    stage_status[stage].update({"status": "failed", "degraded": True})
    stage_status[stage].setdefault("failure_reason", reason)
    record = {
        "stage": stage,
        "policy": policy,
        "reason": reason,
        "message": message,
        "fallback_input": str(fallback_input),
        "downstream_input": str(downstream_input),
    }
    if expected_output is not None:
        record["expected_output"] = str(expected_output)
    fallback_records.append(record)


def _write_pipeline_status(
    *,
    status_path: Path,
    fallback_records: list[dict[str, Any]],
    stage_status: dict[str, dict[str, Any]],
) -> None:
    degraded_stages = sorted({str(record.get("stage")) for record in fallback_records if record.get("stage")})
    payload = {
        "version": "1.0",
        "degraded": bool(fallback_records),
        "status": "completed_degraded" if fallback_records else "completed",
        "degraded_stages": degraded_stages,
        "fallbacks": fallback_records,
        "stages": stage_status,
    }
    write_json(status_path, payload, sanitize_paths=True)


def _symlink_raw(raw_dir: Path, target: Path, media_id: str, logger: logging.Logger, *, copy_raw: bool = False) -> None:
    """Point ``<scene>/raw/<media_id>.<ext>`` at the analyzed video.

    Uses a relative symlink for normal local files. Remote inputs are staged
    through temporary local files, so callers can request a real copy to avoid
    leaving ``raw/`` pointing at a temp directory after cleanup.
    """
    if not target.exists() or not target.is_file():
        logger.info("[scene] raw/%s.* not linked (target missing or non-local): %s", media_id, target)
        return

    link = raw_dir / f"{media_id}{target.suffix.lower()}"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        if copy_raw:
            shutil.copy2(target, link)
            logger.info("[scene] copied raw/%s ← %s", link.name, target)
            return
        rel = os.path.relpath(target.resolve(), raw_dir.resolve())
        link.symlink_to(rel)
        logger.info("[scene] linked raw/%s → %s", link.name, target)
    except OSError as exc:
        logger.warning("[scene] failed to symlink raw/%s: %s (target=%s)", media_id, exc, target)


def _resolve_vlm_input(
    sample: SampleConfig,
    tracking_red_id: Optional[Path],
    sr_out: Optional[Path],
    input_video: Path,
    dry_run: bool,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Resolve which media file VLM JSON / MCQ should consume.

    Priority (highest → lowest):

    1. Explicit ``vlm_video_path`` from config.
    2. Tracking red-id overlay (``sidecars/<stem>_tracking_red_id.<ext>``)
       when the tracking stage produced one. Gives the VLM visible
       ``id_<number>`` labels to reference.
    3. SR output, if SR ran.
    4. Original input media.
    """
    explicit = sample.inputs.vlm_video_path
    if explicit and explicit.strip():
        p = Path(explicit)
        if dry_run or p.exists():
            return p
        if logger is not None:
            logger.warning("[vlm_input] explicit vlm_video_path not found: %s — falling back", p)

    if tracking_red_id is not None and tracking_red_id.exists():
        return tracking_red_id

    if sr_out is not None and sr_out.exists():
        return sr_out

    return input_video


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    sample: SampleConfig,
    config: PipelineConfig,
    *,
    sr_runner: Optional[BaseSuperResolver],
    det_tracker: Optional[BaseTracker],
    vlm_json_gen: Optional[BaseVlmJsonGenerator],
    mcq_gen: Optional[BaseMCQGenerator],
    config_dir: Path,
    repo_root: Path,
    out_dir: Path,
    log_dir: Path,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
    copy_raw: bool = False,
) -> int:
    """Run SR → Tracking → VLM JSON → MCQ on a single sample.

    Args:
        sample:       Per-sample config (inputs + ``output.out_dir``).
        config:       Root pipeline config (Pydantic model).
        sr_runner:    Pre-built SR resolver (None skips SR entirely).
        det_tracker:  Pre-built tracker (None → tracking disabled).
        vlm_json_gen: Pre-built VLM JSON generator (None → VLM disabled).
        mcq_gen:      Pre-built MCQ generator (None → MCQ disabled).
        config_dir:   Directory of the config file (for path resolution).
        repo_root:    Repository root.
        out_dir:      Per-sample output directory. Treated as the DAFT scene root.
        log_dir:      Per-sample log directory.
        logger:       Logger instance.
        dry_run:      If True, log planned actions but skip all execution.
        copy_raw:     Copy analyzed media into raw/ instead of symlinking.
    """
    del config_dir, repo_root  # reserved for future use
    if logger is None:
        logger = logging.getLogger(__name__)

    # --- Setup ---
    empty_output_policy = _empty_policy_value(config.pipeline.empty_output_policy, "warn")
    fallback_records: list[dict[str, Any]] = []
    stage_status: dict[str, dict[str, Any]] = {
        "super_resolution": {
            "enabled": bool(config.super_resolution is not None and config.super_resolution.enabled),
            "status": "skipped",
            "degraded": False,
        },
        "tracking": {
            "enabled": det_tracker is not None,
            "status": "skipped",
            "degraded": False,
        },
        "vlm_json": {
            "enabled": vlm_json_gen is not None,
            "status": "skipped",
            "degraded": False,
        },
        "mcq_generation": {
            "enabled": mcq_gen is not None,
            "status": "skipped",
            "degraded": False,
        },
    }

    if dry_run:
        pipeline_log = None
        _log_plan(logger, "DRY RUN: printing planned actions only.")
        paths = scene_paths(out_dir)
    else:
        ensure_dir(out_dir)
        ensure_dir(log_dir)
        pipeline_log = log_dir / "pipeline.log"
        paths = ensure_scene_skeleton(out_dir)

    # --- Input validation ---
    input_video = Path(sample.inputs.video_path)
    if not dry_run and (not input_video.exists() or not input_video.is_file()):
        raise SystemExit(f"[inputs] input_video_path must be a single media file: {input_video}")

    media_id = set_scene_media_id(input_video)

    if dry_run:
        _log_plan(logger, f"[inputs] DRY RUN: would process: {input_video}")

    # --- Stage 0: MCQ pre-step (QD prompt gen — runs before SR, no video needed) ---
    if mcq_gen is not None:
        if dry_run:
            _log_plan(logger, "[mcq] DRY RUN: would run pre-step (e.g. QD prompt gen)")
        else:
            mcq_gen.run_pre_step(out_dir, sample)

    # --- Stage 1: Super-Resolution ---
    # SR output lives under ``sidecars/`` so it's invisible to the DAFT validator
    # but still addressable via ``raw/<media_id>.<ext>``.
    sr_enabled = bool(config.super_resolution is not None and config.super_resolution.enabled)
    sr_out: Optional[Path] = None

    if sr_runner is not None and sr_enabled:
        sr_suffix = ".png" if is_image_path(input_video) else input_video.suffix.lower()
        sr_out = paths.sidecars_dir / f"sr_output{sr_suffix}"
        stage_status["super_resolution"].update({"status": "planned", "output": str(sr_out)})
        if dry_run:
            _log_plan(logger, f"[sr] DRY RUN: would run SR on {input_video} → {sr_out}")
        else:
            ensure_dir(sr_out.parent)
            sr_runner.run(input_video, sr_out, log_dir=log_dir, pipeline_log=pipeline_log)
            if not sr_out.exists():
                fallback = classify_decode_fallback(
                    log_path=log_dir / "sr.log" if log_dir is not None else None,
                    stage_label="SR",
                    default_reason="missing_output",
                    default_message="Expected SR output was not produced; continuing with original input.",
                )
                if fallback.detected_from_video_decode:
                    msg = f"[sr] WARNING: {fallback.message} expected_output={sr_out}"
                else:
                    msg = f"[sr] WARNING: expected output not found: {sr_out}"
                _append_pipeline_log(pipeline_log, msg)
                _record_stage_fallback(
                    fallback_records=fallback_records,
                    stage_status=stage_status,
                    stage="super_resolution",
                    policy=empty_output_policy,
                    reason=fallback.reason,
                    message=fallback.message,
                    expected_output=sr_out,
                    fallback_input=input_video,
                    downstream_input=input_video,
                )
                if empty_output_policy == "fail":
                    raise SystemExit(msg)
                logger.warning(msg + " (continuing pipeline)")
                sr_out = None
            else:
                stage_status["super_resolution"].update({"status": "completed", "output": str(sr_out)})

    # Link ``raw/<media_id>.<ext>`` to the analyzed video (SR output if it ran, else input).
    if not dry_run:
        analyzed = sr_out if (sr_out is not None and sr_out.exists()) else input_video
        _symlink_raw(paths.raw_dir, analyzed, media_id, logger, copy_raw=copy_raw)

    # --- Stage 2: Detection & Tracking ---
    tracking_result = TrackingResult(success=True)
    if det_tracker is not None:
        tracking_in = sr_out if (sr_out is not None and sr_out.exists()) else input_video
        stage_status["tracking"].update({"status": "planned", "input": str(tracking_in)})

        if dry_run:
            _log_plan(logger, f"[tracking] DRY RUN: would run on {tracking_in}")
        else:
            tracking_exc: Optional[Exception] = None
            try:
                with stage_log_file("tracking", log_dir):
                    tracking_result = det_tracker.run(tracking_in, out_dir)
                if tracking_result.success:
                    stage_status["tracking"].update({"status": "completed"})
            except Exception as exc:
                tracking_exc = exc
                tracking_result = TrackingResult(success=False)
                msg = f"[tracking] stage raised {exc.__class__.__name__}: {exc}"
                _append_pipeline_log(pipeline_log, msg)
                stage_status["tracking"].update(
                    {"status": "failed", "degraded": True, "failure_reason": exc.__class__.__name__}
                )

            if not tracking_result.success:
                fallback = classify_decode_fallback(
                    log_path=log_dir / "tracking.log" if log_dir is not None else None,
                    stage_label="tracking",
                    default_reason="reported_failure",
                    default_message="Tracking output was not produced; continuing without tracking overlays.",
                    extra_text=str(tracking_exc or ""),
                )
                if fallback.detected_from_video_decode:
                    stage_status["tracking"]["failure_reason"] = fallback.reason
                msg = (
                    f"[tracking] WARNING: {fallback.message}"
                    if fallback.detected_from_video_decode
                    else "[tracking] WARNING: stage reported failure"
                )
                _append_pipeline_log(pipeline_log, msg)
                _record_stage_fallback(
                    fallback_records=fallback_records,
                    stage_status=stage_status,
                    stage="tracking",
                    policy=empty_output_policy,
                    reason=fallback.reason,
                    message=fallback.message,
                    fallback_input=tracking_in,
                    downstream_input=tracking_in,
                )
                if empty_output_policy == "fail":
                    if tracking_exc is not None:
                        raise SystemExit(msg) from tracking_exc
                    raise SystemExit(msg)
                logger.warning(msg + " (continuing pipeline)")

    # --- Stage 3: VLM JSON ---
    vlm_json_out = VlmJsonResult(success=True)
    vlm_json_best_effort = mcq_gen is not None
    vlm_json_required = not vlm_json_best_effort

    tracking_red_id = tracking_result.tracking_video_red_id if tracking_result.success else None

    if vlm_json_gen is not None:
        vlm_in = _resolve_vlm_input(sample, tracking_red_id, sr_out, input_video, dry_run, logger)
        stage_status["vlm_json"].update({"status": "planned", "input": str(vlm_in)})
        if not dry_run and not vlm_in.exists():
            msg = f"[vlm_json] input media not found: {vlm_in}"
            _append_pipeline_log(pipeline_log, msg)
            _record_stage_fallback(
                fallback_records=fallback_records,
                stage_status=stage_status,
                stage="vlm_json",
                policy=empty_output_policy,
                reason="input_missing",
                message="VLM input media was not found; skipping VLM JSON.",
                fallback_input=vlm_in,
                downstream_input=vlm_in,
            )
            if vlm_json_required or empty_output_policy == "fail":
                raise SystemExit(msg)
            logger.warning(msg + " (skipping VLM)")
        else:
            if dry_run:
                _log_plan(logger, f"[vlm_json] DRY RUN: would run on {vlm_in}")
            else:
                vlm_exc: Optional[Exception] = None
                vlm_failure_msg = "[vlm_json] WARNING: stage reported failure"
                try:
                    with stage_log_file("vlm_json", log_dir):
                        vlm_json_out = vlm_json_gen.generate(vlm_in, out_dir)
                except Exception as exc:
                    vlm_exc = exc
                    vlm_json_out = VlmJsonResult(success=False)
                    vlm_failure_msg = f"[vlm_json] stage raised {exc.__class__.__name__}: {exc}"
                    _append_pipeline_log(pipeline_log, vlm_failure_msg)
                    stage_status["vlm_json"].update(
                        {"status": "failed", "degraded": True, "failure_reason": exc.__class__.__name__}
                    )
                if not vlm_json_out.success:
                    fallback = classify_decode_fallback(
                        log_path=log_dir / "vlm_json.log" if log_dir is not None else None,
                        stage_label="VLM JSON",
                        default_reason="reported_failure",
                        default_message="VLM JSON output was not produced; continuing without VLM JSON.",
                        extra_text=str(vlm_exc or ""),
                    )
                    if fallback.detected_from_video_decode:
                        stage_status["vlm_json"]["failure_reason"] = fallback.reason
                        vlm_failure_msg = f"[vlm_json] WARNING: {fallback.message}"
                    if vlm_exc is None:
                        _append_pipeline_log(pipeline_log, vlm_failure_msg)
                    _record_stage_fallback(
                        fallback_records=fallback_records,
                        stage_status=stage_status,
                        stage="vlm_json",
                        policy=empty_output_policy,
                        reason=fallback.reason,
                        message=fallback.message,
                        fallback_input=vlm_in,
                        downstream_input=vlm_in,
                    )
                    vlm_has_contextual = (
                        bool(vlm_json_out.image_json)
                        if is_image_path(vlm_in)
                        else bool(vlm_json_out.video_json and vlm_json_out.events_json)
                    )
                    if empty_output_policy == "fail" or (
                        vlm_json_required and not vlm_has_contextual and not fallback.detected_from_video_decode
                    ):
                        if vlm_exc is not None:
                            raise SystemExit(vlm_failure_msg) from vlm_exc
                        raise SystemExit(vlm_failure_msg)
                    logger.warning(vlm_failure_msg + " (continuing pipeline)")
                else:
                    stage_status["vlm_json"].update({"status": "completed"})

    # --- Stage 4: MCQ Generation ---
    if mcq_gen is not None:
        mcq_video = _resolve_vlm_input(sample, tracking_red_id, sr_out, input_video, dry_run, logger)
        stage_status["mcq_generation"].update({"status": "planned", "input": str(mcq_video)})
        # metadata_json: from sidecar input (metadata-llm mode).
        metadata_json: Optional[Path] = (
            Path(sample.inputs.metadata_json_path) if sample.inputs.metadata_json_path else None
        )

        if dry_run:
            _log_plan(logger, f"[mcq] DRY RUN: would generate on {mcq_video}")
        else:
            mcq_exc: Optional[Exception] = None
            mcq_failure_msg = "[mcq] WARNING: stage reported failure"
            try:
                with stage_log_file("mcq", log_dir):
                    mcq_result = mcq_gen.generate(
                        mcq_video,
                        out_dir,
                        metadata_json=metadata_json,
                    )
            except Exception as exc:
                mcq_exc = exc
                mcq_result = MCQResult(success=False)
                mcq_failure_msg = f"[mcq] stage raised {exc.__class__.__name__}: {exc}"
                _append_pipeline_log(pipeline_log, mcq_failure_msg)
                stage_status["mcq_generation"].update(
                    {"status": "failed", "degraded": True, "failure_reason": exc.__class__.__name__}
                )
            if not mcq_result.success:
                fallback = classify_decode_fallback(
                    log_path=log_dir / "mcq.log" if log_dir is not None else None,
                    stage_label="MCQ generation",
                    default_reason="reported_failure",
                    default_message="MCQ output was not produced; continuing without MCQ task files.",
                    extra_text=str(mcq_exc or ""),
                )
                if fallback.detected_from_video_decode:
                    stage_status["mcq_generation"]["failure_reason"] = fallback.reason
                    mcq_failure_msg = f"[mcq] WARNING: {fallback.message}"
                if mcq_exc is None:
                    _append_pipeline_log(pipeline_log, mcq_failure_msg)
                _record_stage_fallback(
                    fallback_records=fallback_records,
                    stage_status=stage_status,
                    stage="mcq_generation",
                    policy=empty_output_policy,
                    reason=fallback.reason,
                    message=fallback.message,
                    fallback_input=mcq_video,
                    downstream_input=mcq_video,
                )
                if empty_output_policy == "fail":
                    if mcq_exc is not None:
                        raise SystemExit(mcq_failure_msg) from mcq_exc
                    raise SystemExit(mcq_failure_msg)
                logger.warning(mcq_failure_msg + " (continuing pipeline)")
            else:
                stage_status["mcq_generation"].update({"status": "completed"})

    if not dry_run:
        _write_pipeline_status(
            status_path=paths.sidecars_dir / "pipeline_status.json",
            fallback_records=fallback_records,
            stage_status=stage_status,
        )
    logger.info("DONE: out=%s", out_dir)
    return 0
