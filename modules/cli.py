#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified CLI for Auto-Labeling Pipeline.

Supports local, NVCF, and cloud execution with automatic remote storage handling.

Usage:
    # Local execution with config file
    python modules/cli.py --config configs/pipeline_example.yaml

    # With OmegaConf dotlist overrides (key=value)
    python modules/cli.py --config configs/pipeline_example.yaml \\
        data.0.inputs.video_path=./videos/clip.mp4 data.0.output.out_dir=./out/work

    # Remote input/output (per-sample paths can be remote via MSC)
    python modules/cli.py --config s3://bucket/config.yaml

OmegaConf dotlist overrides (key=value) override config file values.
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.path_sanitize import default_artifact_root, sanitize_paths_for_publish
from config.loader import load_config_with_overrides
from config.normalize import resolve_input_path, resolve_path
from config.schema import validate_schema
from config.validate import validate_environment
from detection_and_tracking.base import BaseTracker
from detection_and_tracking.factory import create_tracker
from mcq_generation.base import BaseMCQGenerator
from mcq_generation.factory import create_mcq_generator
from nvcf_msc_utils import (
    VIDEO_EXTENSIONS,
    NVCFProgressTracker,
    detect_nvcf_llm_endpoint,
    detect_nvcf_vlm_endpoint,
    is_remote_path,
    localize_path_to_dir,
    materialize_move,
    normalize_remote_prefix,
    normalize_remote_prefix_for_compare,
    remote_child_prefix,
    setup_msc_config,
    sync_local_to_remote,
)
from omegaconf import OmegaConf
from pipeline import run_pipeline
from sr_runner.base import BaseSuperResolver
from sr_runner.factory import create_sr_runner
from vlm_json.base import BaseVlmJsonGenerator
from vlm_json.factory import create_vlm_json_generator

MODULES_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULES_DIR.parent

_MCQ_MODES_REQUIRE_VLM: frozenset[str] = frozenset(
    {
        "window-direct-vlm",
        "window-vlm-llm",
        "question-driven-vlm-llm",
    }
)
_MCQ_MODES_REQUIRE_LLM: frozenset[str] = frozenset(
    {
        "window-vlm-llm",
        "question-driven-vlm-llm",
        "metadata-llm",
    }
)


def _get_endpoint_strings(validated) -> tuple[str, str, int, float, str, str, int, float]:
    """
    Extract endpoints from the validated Pydantic schema.

    Returns:
      (vlm_url, vlm_model, vlm_retries, vlm_retry_backoff_s,
       llm_url, llm_model, llm_retries, llm_retry_backoff_s)

    Note: schema validators already strip/validate these fields (and disallow __REQUIRED__),
    so this function is intentionally simple and does not re-sanitize.
    """
    ep = getattr(validated, "endpoints", None)
    vlm = getattr(ep, "vlm", None) if ep is not None else None
    llm = getattr(ep, "llm", None) if ep is not None else None
    vlm_url = getattr(vlm, "url", "") if vlm is not None else ""
    vlm_model = getattr(vlm, "model", "") if vlm is not None else ""
    vlm_retries = getattr(vlm, "retries", 3) if vlm is not None else 3
    vlm_retry_backoff_s = getattr(vlm, "retry_backoff_s", 5.0) if vlm is not None else 5.0
    llm_url = getattr(llm, "url", "") if llm is not None else ""
    llm_model = getattr(llm, "model", "") if llm is not None else ""
    llm_retries = getattr(llm, "retries", 3) if llm is not None else 3
    llm_retry_backoff_s = getattr(llm, "retry_backoff_s", 5.0) if llm is not None else 5.0
    return (
        str(vlm_url or ""),
        str(vlm_model or ""),
        int(vlm_retries or 0),
        float(vlm_retry_backoff_s or 5.0),
        str(llm_url or ""),
        str(llm_model or ""),
        int(llm_retries or 0),
        float(llm_retry_backoff_s or 5.0),
    )


def _normalize_config_paths_inplace(config: Dict[str, object], *, config_dir: Path, repo_root: Path) -> None:
    """Normalize known path-like fields to absolute paths early in the CLI.

    This prevents relative repo paths like "modules/..." from being misinterpreted
    as config-relative (e.g. "configs/modules/...").
    """

    def _resolve_input(v: object) -> object:
        s = str(v or "").strip()
        if not s or "${" in s:
            return v
        if is_remote_path(s):
            return v
        return resolve_input_path(s, config_dir=config_dir, repo_root=repo_root)

    def _get(d: Dict[str, object], *keys: str) -> object:
        cur: object = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    # pipeline.model_cache_path
    v = _get(config, "pipeline", "model_cache_path")
    if v is not None and isinstance(config.get("pipeline"), dict):
        config["pipeline"]["model_cache_path"] = _resolve_input(v)  # type: ignore[index]

    # vlm_json prompt overrides and shipped cookbook defaults
    vlm_json = config.get("vlm_json")
    if isinstance(vlm_json, dict):
        for k in (
            "scene_prompt_file",
            "events_prompt_file",
            "default_video_json_prompt_file",
            "default_video_events_prompt_file",
            "default_image_json_prompt_file",
        ):
            if k in vlm_json and vlm_json.get(k) is not None:
                vlm_json[k] = _resolve_input(vlm_json.get(k))

    # mcq_generation.window_metadata_extraction.* prompt + bank files (all modes, incl QD templates)
    mcq = config.get("mcq_generation")
    if isinstance(mcq, dict):
        w = mcq.get("window_metadata_extraction")
        if isinstance(w, dict):
            for k in (
                "scene_prompt_file",
                "mcq_prompt_file",
                "question_bank_file",
                "qd_vlm_scene_prompt_template_file",
                "qd_mcq_mapper_prompt_template_file",
                "vlm_verify_prompt_file",
            ):
                if k in w and w.get(k) is not None:
                    w[k] = _resolve_input(w.get(k))


_DAFT_VALIDATE_TIMEOUT_S = int(os.environ.get("DAFT_VALIDATE_TIMEOUT_S", 120))


def _run_daft_validate(scene_dir: Path, logger: logging.Logger) -> None:
    """Run ``tao-daft validate`` on a completed scene directory.

    Best-effort and non-blocking: the validator is invoked via subprocess and
    its output is logged. Never raises and never changes the pipeline's exit
    code -- the converter emits DAFT-compliant output unconditionally, so this
    hook is a sanity check for dev workflows, not a gate.

    Skips silently when ``tao-daft`` is not on PATH; the ``nvidia-tao-daft``
    package is NVIDIA-internal and not shipped in the container by default.
    Install it locally (``pip install -e /path/to/nvidia-tao-daft``) to enable.
    """
    if shutil.which("tao-daft") is None:
        logger.info("[daft] tao-daft not on PATH; skipping validation (install nvidia-tao-daft to enable)")
        return

    cmd = [
        "tao-daft",
        "validate",
        "metropolis-v3.0",
        "--path",
        str(scene_dir),
        "--raw",
        "auto",
        "--strict",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_DAFT_VALIDATE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        logger.warning("[daft] validator timed out after %ds on %s", _DAFT_VALIDATE_TIMEOUT_S, scene_dir)
        return
    except Exception as e:
        logger.warning("[daft] validator invocation failed: %s", e)
        return

    if proc.returncode == 0:
        logger.info("[daft] scene validated OK: %s", scene_dir)
        return

    logger.warning(
        "[daft] validator reported issues (rc=%d) for %s\nstdout:\n%s\nstderr:\n%s",
        proc.returncode,
        scene_dir,
        (proc.stdout or "").strip(),
        (proc.stderr or "").strip(),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-labeling pipeline CLI (unified local/cloud execution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file (required)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (local or remote: s3://, msc://)",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned stage commands without executing them.",
    )
    # NOTE: We intentionally do not expose "staging" knobs. Schema does not define them;
    # remote inputs are downloaded to a temp dir to avoid polluting out_dir with _work/_inputs.

    args, unknown = parser.parse_known_args(argv)

    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Auto-labeling Pipeline (SR + Tracking + VLM + MCQ)")
    logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Setup MSC configuration (auto-detects from environment)
    setup_msc_config()

    # Initialize NVCF progress tracker (auto-detects NVCF environment)
    progress = NVCFProgressTracker()

    final_progress: Optional[tuple[int, str]] = None

    try:
        # Load config (supports local or remote --config) and apply OmegaConf dotlist overrides.
        progress.update(10, "Loading configuration")
        config, config_dir = load_config_with_overrides(args.config, unknown, logger=logger)

        if isinstance(config, dict):
            _normalize_config_paths_inplace(config, config_dir=config_dir, repo_root=REPO_ROOT)

        validated = validate_schema(config, logger=logger)
        if validated is None:
            final_progress = (0, "Failed: config schema validation")
            return 2

        # --- Build EndpointResolver once (replaces per-sample _apply_endpoint_fallbacks) ---
        resolver = EndpointResolver(validated.endpoints, logger=logger)
        # Best-effort NVCF VLM endpoint auto-detection.
        if validated.vlm_json and validated.vlm_json.enabled:
            vlm_url, vlm_model = resolver.resolve_vlm()
            if not vlm_url or not vlm_model:
                try:
                    detected_url, detected_model = detect_nvcf_vlm_endpoint()
                    resolver.apply_vlm_defaults(url=detected_url, model=detected_model)
                    vlm_url, vlm_model = resolver.resolve_vlm()
                except Exception as e:
                    logger.warning("NVCF VLM endpoint auto-detection failed: %s", e)
                if not vlm_url or not vlm_model:
                    if bool(getattr(progress, "is_nvcf", False)):
                        updates: dict = {"vlm_json": validated.vlm_json.model_copy(update={"enabled": False})}
                        logger.warning(
                            "vlm_json.enabled=true but VLM endpoint not configured. Disabling vlm_json stage on NVCF."
                        )
                        mcq_mode = str(getattr(validated.mcq_generation, "mode", "") or "").strip()
                        if (
                            validated.mcq_generation is not None
                            and validated.mcq_generation.enabled
                            and mcq_mode in _MCQ_MODES_REQUIRE_VLM
                        ):
                            logger.warning(
                                "mcq_generation.mode=%s requires a VLM endpoint. "
                                "Disabling mcq_generation stage on NVCF.",
                                mcq_mode,
                            )
                            updates["mcq_generation"] = validated.mcq_generation.model_copy(update={"enabled": False})
                        validated = validated.model_copy(update=updates)
                    else:
                        logger.error(
                            "vlm_json.enabled=true requires VLM endpoint "
                            "(endpoints.vlm.url + endpoints.vlm.model or VLM_BASE_URL/VLM_MODEL)."
                        )
                        final_progress = (0, "Failed: missing VLM endpoint config")
                        return 2

        # MCQ VLM endpoint check (independent of vlm_json.enabled — window MCQ modes call VLM directly).
        if validated.mcq_generation is not None and validated.mcq_generation.enabled:
            mcq_mode = str(getattr(validated.mcq_generation, "mode", "") or "").strip()
            if mcq_mode in _MCQ_MODES_REQUIRE_VLM:
                vlm_url, vlm_model = resolver.resolve_vlm()
                if not vlm_url or not vlm_model:
                    try:
                        detected_url, detected_model = detect_nvcf_vlm_endpoint()
                        resolver.apply_vlm_defaults(url=detected_url, model=detected_model)
                        vlm_url, vlm_model = resolver.resolve_vlm()
                    except Exception as e:
                        logger.warning("NVCF VLM endpoint auto-detection failed: %s", e)
                    if not vlm_url or not vlm_model:
                        if bool(getattr(progress, "is_nvcf", False)):
                            logger.warning(
                                "mcq_generation.mode=%s requires a VLM endpoint but none is configured. "
                                "Disabling mcq_generation stage on NVCF.",
                                mcq_mode,
                            )
                            validated = validated.model_copy(
                                update={
                                    "mcq_generation": validated.mcq_generation.model_copy(update={"enabled": False})
                                }
                            )
                        else:
                            logger.error(
                                "mcq_generation.mode=%s requires a VLM endpoint "
                                "(endpoints.vlm.url + endpoints.vlm.model or VLM_BASE_URL/VLM_MODEL).",
                                mcq_mode,
                            )
                            final_progress = (0, "Failed: missing VLM endpoint config")
                            return 2

        # Best-effort NVCF LLM endpoint auto-detection.
        if validated.mcq_generation is not None and validated.mcq_generation.enabled:
            mcq_mode = str(getattr(validated.mcq_generation, "mode", "") or "").strip()
            if mcq_mode in _MCQ_MODES_REQUIRE_LLM:
                llm_url, llm_model = resolver.resolve_llm()
                if not llm_url or not llm_model:
                    try:
                        detected_llm_url, detected_llm_model = detect_nvcf_llm_endpoint()
                        resolver.apply_llm_defaults(url=detected_llm_url, model=detected_llm_model)
                        llm_url, llm_model = resolver.resolve_llm()
                    except Exception as e:
                        logger.warning("NVCF LLM endpoint auto-detection failed: %s", e)
                    if not llm_url or not llm_model:
                        if bool(getattr(progress, "is_nvcf", False)):
                            logger.warning(
                                "mcq_generation.mode=%s requires an LLM endpoint but none is configured. "
                                "Disabling mcq_generation stage on NVCF.",
                                mcq_mode,
                            )
                            validated = validated.model_copy(
                                update={
                                    "mcq_generation": validated.mcq_generation.model_copy(update={"enabled": False})
                                }
                            )
                        else:
                            logger.error(
                                "mcq_generation.mode=%s requires an LLM endpoint "
                                "(endpoints.llm.url + endpoints.llm.model or LLM_BASE_URL/LLM_MODEL).",
                                mcq_mode,
                            )
                            final_progress = (0, "Failed: missing LLM endpoint config")
                            return 2

        validate_environment(logger, dict(config))

        empty_output_policy = str(getattr(validated.pipeline, "empty_output_policy", None) or "warn").strip().lower()
        if empty_output_policy not in {"warn", "fail"}:
            empty_output_policy = "warn"

        # --- Init SR runner (always; handles subprocess torchrun per sample) ---
        sr_runner: Optional[BaseSuperResolver] = None
        if validated.super_resolution is not None and validated.super_resolution.enabled:
            try:
                sr_runner = create_sr_runner(validated, logger)
                logger.info("SR runner initialized successfully")
            except Exception as e:
                if empty_output_policy == "fail":
                    logger.exception("Failed to initialize SR runner")
                    final_progress = (0, f"Failed: SR runner init: {e}")
                    return 2
                logger.warning(
                    "SR runner init failed; disabling SR stage (pipeline.empty_output_policy=warn). (%s)",
                    e,
                )
                sr_runner = None

        # --- Init detection & tracking ---
        det_tracker: Optional[BaseTracker] = None
        if validated.detection_and_tracking is not None and validated.detection_and_tracking.enabled:
            try:
                det_tracker = create_tracker(validated, logger)
                logger.info("Tracker initialized successfully")
            except Exception as e:
                if empty_output_policy == "fail":
                    logger.exception("Failed to initialize tracker")
                    final_progress = (0, f"Failed: tracker init: {e}")
                    return 2
                logger.warning(
                    "Tracker init failed; disabling tracking stage (pipeline.empty_output_policy=warn). (%s)",
                    e,
                )
                det_tracker = None

        # --- Init VLM JSON generator ---
        vlm_json_gen: Optional[BaseVlmJsonGenerator] = None
        if validated.vlm_json is not None and validated.vlm_json.enabled:
            try:
                vlm_json_gen = create_vlm_json_generator(
                    validated,
                    resolver,
                    logger,
                    config_dir=str(config_dir),
                    repo_root=REPO_ROOT,
                )
                logger.info("VLM JSON generator initialized successfully")
            except Exception as e:
                logger.exception("Failed to initialize VLM JSON generator")
                final_progress = (0, f"Failed: VLM JSON generator init: {e}")
                return 2

        # --- Init MCQ generator ---
        mcq_gen: Optional[BaseMCQGenerator] = None
        if validated.mcq_generation is not None and validated.mcq_generation.enabled:
            try:
                mcq_gen = create_mcq_generator(validated, resolver, logger, config_dir=str(config_dir))
                logger.info("MCQ generator initialized successfully")
            except Exception as e:
                logger.exception("Failed to initialize MCQ generator")
                final_progress = (0, f"Failed: MCQ generator init: {e}")
                return 2

        auto_base_out_dir: Optional[Path] = None
        sample_failures: list[tuple[int, str, int]] = []
        sample_successes = 0

        total = len(validated.data)
        for i, sample in enumerate(validated.data):
            # Keep percent monotonic and meaningful.
            pct = 10 + int((i / max(total, 1)) * 80)
            progress.update(pct, f"Running sample {i + 1}/{total}")

            # Output root:
            # - local out_dir: stages write directly under that folder
            # - remote out_dir (s3://, ...): stages write to a local folder, then the CLI uploads it
            #   to the remote prefix after the run completes successfully.
            remote_out_dir: Optional[str] = None
            if sample.output.out_dir:
                out_dir_raw = str(resolve_path(sample.output.out_dir, config_dir=config_dir, repo_root=REPO_ROOT))
                if is_remote_path(out_dir_raw):
                    remote_out_dir = normalize_remote_prefix(out_dir_raw).rstrip("/") + "/"
                    if auto_base_out_dir is None:
                        auto_base_out_dir = REPO_ROOT / "output" / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}"
                    run_out_dir = auto_base_out_dir / f"sample_{i:04d}"
                else:
                    run_out_dir = Path(out_dir_raw)
            else:
                if auto_base_out_dir is None:
                    auto_base_out_dir = REPO_ROOT / "output" / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}"
                run_out_dir = auto_base_out_dir / f"sample_{i:04d}"
            try:
                temp_ctx: Optional[tempfile.TemporaryDirectory] = None
                copy_raw = False
                if bool(args.dry_run):
                    # Dry-run should not touch filesystem (no mkdir, no downloading/staging).
                    sample_local = sample
                else:
                    run_out_dir.mkdir(parents=True, exist_ok=True)
                    # Only stage inputs when needed:
                    # - remote paths: download into a unique temp dir
                    # - local paths: use as-is (no copying/symlinking)
                    # Cleanup happens after the sample finishes.
                    raw_video = str(sample.inputs.video_path or "").strip()
                    raw_vlm_video = str(getattr(sample.inputs, "vlm_video_path", "") or "").strip()
                    need_staging = any(
                        is_remote_path(str(p).strip())
                        for p in (
                            sample.inputs.video_path,
                            sample.inputs.metadata_json_path,
                            raw_vlm_video,
                        )
                        if p and str(p).strip()
                    )
                    tmp_dir = run_out_dir
                    if need_staging:
                        temp_ctx = tempfile.TemporaryDirectory(prefix="auto_labeling_inputs_")
                        tmp_dir = Path(temp_ctx.name)
                        copy_raw = True
                    local_video = localize_path_to_dir(
                        sample.inputs.video_path,
                        dst_dir=tmp_dir,
                        logger=logger,
                        extensions=VIDEO_EXTENSIONS,
                        config_dir=config_dir,
                        repo_root=REPO_ROOT,
                    )
                    if local_video is None:
                        raise RuntimeError(
                            f"Failed to resolve video path: {sample.inputs.video_path!r}. "
                            "Check that the path exists and has a supported video/image extension."
                        )
                    local_meta = localize_path_to_dir(
                        sample.inputs.metadata_json_path,
                        dst_dir=tmp_dir,
                        logger=logger,
                        extensions=(".json",),
                        config_dir=config_dir,
                        repo_root=REPO_ROOT,
                    )

                    local_vlm_override: Optional[Path] = None
                    if raw_vlm_video and (raw_vlm_video != raw_video):
                        if is_remote_path(raw_vlm_video):
                            local_vlm_override = localize_path_to_dir(
                                raw_vlm_video,
                                dst_dir=tmp_dir,
                                logger=logger,
                                extensions=VIDEO_EXTENSIONS,
                                config_dir=config_dir,
                                repo_root=REPO_ROOT,
                            )
                        else:
                            # Only localize if it resolves to an existing local file; otherwise treat it as
                            # a pipeline artifact path under the scene layout, such as sidecars/<stem>_tracking_red_id.<ext>.
                            p = Path(raw_vlm_video).expanduser()
                            if not p.is_absolute() and not p.exists():
                                try:
                                    resolved = resolve_path(raw_vlm_video, config_dir=config_dir, repo_root=REPO_ROOT)
                                    p2 = Path(str(resolved)).expanduser()
                                    if p2.exists():
                                        p = p2
                                except Exception:
                                    pass
                            if p.exists():
                                local_vlm_override = localize_path_to_dir(
                                    str(p),
                                    dst_dir=tmp_dir,
                                    logger=logger,
                                    extensions=VIDEO_EXTENSIONS,
                                    config_dir=config_dir,
                                    repo_root=REPO_ROOT,
                                )
                    sample_local = sample.model_copy(
                        update={
                            "inputs": sample.inputs.model_copy(
                                update={
                                    "video_path": str(local_video),
                                    # If vlm_video_path equals the original video_path, rewrite it to the staged local path.
                                    # If it points to a pipeline artifact under the scene layout, keep it as-is
                                    # (it will exist after upstream stages run).
                                    "vlm_video_path": (
                                        str(local_video)
                                        if raw_vlm_video and (raw_vlm_video == raw_video)
                                        else str(local_vlm_override)
                                        if local_vlm_override
                                        else sample.inputs.vlm_video_path
                                    ),
                                    "metadata_json_path": str(local_meta) if local_meta else None,
                                }
                            )
                        }
                    )

                # Per-sample log dir:
                # - If output.log_dir is set, honor it.
                # - Otherwise default to <out_dir>/logs.
                #
                # Note: if out_dir is remote and log_dir is remote, we still write logs locally and
                # upload them to the requested remote log_dir at the end (since stages write locally).
                run_log_dir_remote: str | None = None
                if getattr(sample_local.output, "log_dir", None):
                    log_dir_raw = str(
                        resolve_path(str(sample_local.output.log_dir), config_dir=config_dir, repo_root=REPO_ROOT)
                    ).strip()
                    if is_remote_path(log_dir_raw):
                        run_log_dir_remote = normalize_remote_prefix(log_dir_raw)
                        run_log_dir = run_out_dir / "logs"
                    else:
                        run_log_dir = Path(log_dir_raw)
                else:
                    run_log_dir = run_out_dir / "logs"

                _pipeline_log_handler: logging.FileHandler | None = None
                if not bool(args.dry_run):
                    run_log_dir.mkdir(parents=True, exist_ok=True)
                    _pipeline_log_path = run_log_dir / "pipeline.log"
                    _pipeline_log_handler = logging.FileHandler(_pipeline_log_path, encoding="utf-8")
                    _pipeline_log_handler.setFormatter(
                        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                    )
                    logging.getLogger().addHandler(_pipeline_log_handler)

                if remote_out_dir:
                    logger.info(f"[outputs] remote out_dir requested: {remote_out_dir}")
                    logger.info(f"[outputs] local working out_dir: {run_out_dir}")

                # Persist the effective config used for this sample (for reproducibility).
                config_dst = str(getattr(sample_local.output, "config_path", "") or "").strip()
                if not config_dst:
                    config_dst = str(run_out_dir / "config.yaml")
                if remote_out_dir:
                    # Keep the config under the local out_dir; it will be uploaded by the folder sync.
                    config_dst = str(run_out_dir / "config.yaml")
                config_dst = resolve_path(config_dst, config_dir=config_dir, repo_root=REPO_ROOT)
                if bool(args.dry_run):
                    logger.info(f"[config] Dry-run: would write used config -> {config_dst}")
                else:
                    # Write a runnable config (can be passed back to this CLI via --config).
                    # Keep it per-sample to avoid requiring dotlist overrides on rerun.
                    runnable_cfg = validated.model_dump(exclude_none=True)
                    runnable_cfg["data"] = [sample_local.model_dump(exclude_none=True)]
                    runnable_cfg = sanitize_paths_for_publish(
                        runnable_cfg,
                        artifact_root=default_artifact_root(REPO_ROOT),
                        extra_roots=(REPO_ROOT,),
                    )
                    used_yaml = OmegaConf.to_yaml(OmegaConf.create(runnable_cfg), resolve=True)
                    if is_remote_path(config_dst) and not remote_out_dir:
                        tmp_cfg = run_out_dir / "_config_used_tmp.yaml"
                        tmp_cfg.write_text(used_yaml, encoding="utf-8")
                        materialize_move(src=tmp_cfg, dst=config_dst, dry_run=False, logger=logger)
                        try:
                            tmp_cfg.unlink()
                        except Exception:
                            pass
                    else:
                        dp = Path(config_dst).expanduser()
                        dp.parent.mkdir(parents=True, exist_ok=True)
                        dp.write_text(used_yaml, encoding="utf-8")
                        logger.info(f"[config] wrote used config: {dp}")

                old_dry = os.environ.get("AUTO_LABELING_DRY_RUN")
                if bool(args.dry_run):
                    os.environ["AUTO_LABELING_DRY_RUN"] = "1"
                try:
                    result = run_pipeline(
                        sample_local,
                        validated,
                        sr_runner=sr_runner,
                        det_tracker=det_tracker,
                        vlm_json_gen=vlm_json_gen,
                        mcq_gen=mcq_gen,
                        config_dir=config_dir,
                        repo_root=REPO_ROOT,
                        out_dir=run_out_dir,
                        log_dir=run_log_dir,
                        logger=logger,
                        dry_run=bool(args.dry_run),
                        copy_raw=copy_raw,
                    )
                except SystemExit as e:
                    code = getattr(e, "code", 1)
                    # SystemExit can carry an int or a message string; normalize to non-zero rc.
                    result = int(code) if isinstance(code, int) else 1
                    logger.error(f"[sample {i}] pipeline aborted (SystemExit): {code}")
                except Exception as e:
                    # Treat unexpected per-sample exceptions the same as a non-zero rc.
                    result = 1
                    logger.error(f"[sample {i}] pipeline raised exception: {e}")
                    logger.debug("Exception details", exc_info=True)
                finally:
                    if _pipeline_log_handler is not None:
                        logging.getLogger().removeHandler(_pipeline_log_handler)
                        _pipeline_log_handler.close()
                        _pipeline_log_handler = None
                    if bool(args.dry_run):
                        if old_dry is None:
                            os.environ.pop("AUTO_LABELING_DRY_RUN", None)
                        else:
                            os.environ["AUTO_LABELING_DRY_RUN"] = old_dry
                    if temp_ctx is not None:
                        try:
                            temp_ctx.cleanup()
                        except Exception:
                            pass
                if result == 0 and bool(validated.pipeline.daft_validate) and not bool(args.dry_run):
                    _run_daft_validate(Path(run_out_dir), logger)

                if result != 0:
                    # Respect pipeline.empty_output_policy for per-sample failures:
                    # - warn: keep running remaining samples; report failures at the end
                    # - fail: stop immediately
                    sample_failures.append((i, str(run_out_dir), int(result)))
                    logger.error(
                        f"[sample {i}] failed (rc={result}). "
                        f"empty_output_policy={empty_output_policy}. out_dir={run_out_dir}"
                    )
                    if empty_output_policy == "fail":
                        progress.update(0, "Failed: Pipeline error")
                        final_progress = (0, "Failed: Pipeline error")
                        return int(result)
                    # warn: continue to next sample
                    continue
                sample_successes += 1

                # If the user requested a remote out_dir, upload the whole local out_dir folder to that remote prefix.
                if remote_out_dir and not bool(args.dry_run):
                    try:
                        logger.info(f"[outputs] uploading local out_dir -> remote: {run_out_dir} -> {remote_out_dir}")
                        sync_local_to_remote(local_dir=str(run_out_dir), remote_path=remote_out_dir, verbose=True)
                        logger.info(f"[outputs] uploaded: {remote_out_dir}")
                    except Exception as e:
                        raise SystemExit(f"Failed to upload out_dir to remote: {remote_out_dir}: {e}") from e

                # Optional: if log_dir was explicitly set to a separate remote path, upload logs there as well.
                logs_already_in_remote_scene = (
                    bool(remote_out_dir)
                    and bool(run_log_dir_remote)
                    and normalize_remote_prefix_for_compare(run_log_dir_remote)
                    == remote_child_prefix(remote_out_dir, "logs")
                )
                if run_log_dir_remote and not logs_already_in_remote_scene and not bool(args.dry_run):
                    try:
                        logger.info(f"[outputs] uploading logs -> remote: {run_log_dir} -> {run_log_dir_remote}")
                        sync_local_to_remote(local_dir=str(run_log_dir), remote_path=run_log_dir_remote, verbose=True)
                        logger.info(f"[outputs] uploaded logs: {run_log_dir_remote}")
                    except Exception as e:
                        raise SystemExit(f"Failed to upload log_dir to remote: {run_log_dir_remote}: {e}") from e
                elif logs_already_in_remote_scene:
                    logger.info("[outputs] logs are already included in remote out_dir upload: %s", remote_out_dir)
            finally:
                # Do not auto-clean: intermediate artifacts remain under each sample's output.out_dir.
                pass

        if sample_failures:
            logger.error(
                f"Pipeline finished with failures: {len(sample_failures)}/{total} failed, "
                f"{sample_successes}/{total} succeeded."
            )
            for idx, outd, rc in sample_failures[:50]:
                logger.error(f"  - sample {idx}: rc={rc} out_dir={outd}")
            if len(sample_failures) > 50:
                logger.error(f"  ... {len(sample_failures) - 50} more failures (omitted)")
            final_progress = (100, f"Completed with {len(sample_failures)} failures")
            progress.update(*final_progress)
            return 1

        final_progress = (100, "Completed successfully")
        progress.update(*final_progress)
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Pipeline failed")
        else:
            logger.error("Pipeline failed: %s (set LOG_LEVEL=DEBUG for stack trace)", e)
        final_progress = (0, f"Failed: {str(e)[:100]}")
        progress.update(*final_progress)
        return 1

    finally:
        # Always emit a final NVCF progress update (best-effort), even on early returns.
        try:
            if bool(getattr(progress, "is_nvcf", False)):
                progress.update(*(final_progress or (0, "Finished")))
        except Exception:
            pass

        # Per-sample work dirs are cleaned up inside the loop.


if __name__ == "__main__":
    raise SystemExit(main())
