# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SeedVR2 super-resolution runner.

Wraps the ``torchrun`` subprocess that runs ``inference_seedvr2_window``.
SR is a permanent subprocess (multi-GPU diffusion cannot be inlined).

The resolver is constructed ONCE before the sample loop; ``run()`` is called
per-sample with explicit input/output paths.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from al_utils.ckpts import ensure_hf_file, resolve_ckpts_root
from al_utils.common import ensure_dir, run_cmd
from al_utils.media_decode import (
    unsupported_video_decoder_message,
    unsupported_video_decoder_seen,
    video_decode_failure_message,
    video_decode_failure_seen,
)
from al_utils.media_paths import is_image_path
from al_utils.schema.config import PipelineConfig
from sr_runner.base import BaseSuperResolver

_SEEDVR_RUNTIME_LINKS = (
    "common",
    "configs_3b",
    "configs_7b",
    "data",
    "models",
    "pos_emb.pt",
    "neg_emb.pt",
    "projects",
)


class SeedVR2Resolver(BaseSuperResolver):
    """SeedVR2 super-resolution via ``torchrun`` subprocess.

    ``__init__`` resolves paths and GPU configuration once; ``run()`` builds
    and executes the torchrun command per sample.
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger, *, gpu_list: list[int]) -> None:
        super().__init__(logger)

        sr = config.super_resolution
        self._variant = str(sr.variant)
        self._seed = int(sr.seed)
        self._res_h = int(sr.res_h)
        self._res_w = int(sr.res_w)
        self._window_frames = int(sr.window_frames)
        self._overlap_frames = int(sr.overlap_frames)
        self._window_timeout_s = int(sr.window_timeout)
        self._out_fps: Optional[float] = float(sr.out_fps) if sr.out_fps is not None else None
        self._empty_output_policy = str(config.pipeline.empty_output_policy or "warn").strip().lower()
        if self._empty_output_policy not in {"warn", "fail"}:
            self._empty_output_policy = "warn"

        # Derive sp_size: multi-GPU uses all GPUs in the list; single-GPU uses only the first.
        if not gpu_list:
            raise ValueError("[sr] gpu_list is empty; at least one GPU index is required")
        if config.pipeline.use_multi_gpu:
            self._gpu_list = list(gpu_list)
            self._sp_size = len(self._gpu_list)
        else:
            self._gpu_list = [gpu_list[0]]
            self._sp_size = 1

        # Resolve the repo root (two levels up from this file: sr_runner/ → modules/ → repo/).
        self._repo_root = Path(__file__).resolve().parent.parent.parent

        # Resolve ckpts root: env MODEL_CACHE_PATH > config.pipeline.model_cache_path > /workspace/ckpts > repo_root/ckpts
        ckpts_root = resolve_ckpts_root(repo_root=self._repo_root, model_cache_path=config.pipeline.model_cache_path)
        # SeedVR checkpoints live in the user-visible ckpts root. Upstream
        # SeedVR resolves them from ./ckpts under its repo cwd, so Docker runs
        # from a writable runtime mirror rooted in the configured cache.
        self._sr_seedvr2_dir = ckpts_root / "seedvr2"

        # Resolve SeedVR code root: prefer SEEDVR_ROOT env; fallback to vendored code.
        env_seedvr_root = str(os.getenv("SEEDVR_ROOT", "")).strip()
        if env_seedvr_root:
            candidate = Path(env_seedvr_root).expanduser().resolve()
            marker_ok = (
                (candidate / "common").exists()
                and (candidate / "projects").exists()
                and (candidate / "pos_emb.pt").exists()
            )
            seedvr_source_root = candidate if marker_ok else (self._repo_root / "modules" / "super_resolution")
        else:
            seedvr_source_root = self._repo_root / "modules" / "super_resolution"

        # Ensure required SeedVR2 ckpts exist at init time (per your design goal).
        ensure_seedvr2_ckpts(ckpts_root=ckpts_root, variant=self._variant, logger=self.logger)
        self._seedvr_root = seedvr_source_root
        if (seedvr_source_root / "projects").exists():
            self._seedvr_root = _prepare_seedvr_runtime_root(
                source_root=seedvr_source_root,
                runtime_root=ckpts_root / "seedvr_runtime",
                ckpts_dir=self._sr_seedvr2_dir,
                logger=self.logger,
            )

        # Determine torchrun module path.
        if (self._seedvr_root / "projects" / "inference_seedvr2_window.py").exists():
            self._sr_module = "projects.inference_seedvr2_window"
            self._sr_cwd = self._seedvr_root
        else:
            self._sr_module = "modules.sr_runner.inference_seedvr2_window"
            self._sr_cwd = self._repo_root

    # ------------------------------------------------------------------

    def run(
        self,
        input_video: Path,
        output_video: Path,
        *,
        log_dir: Optional[Path] = None,
        pipeline_log: Optional[Path] = None,
    ) -> Path:
        """Run SeedVR2 super-resolution on a single media sample.

        Args:
            input_video:   Path to the input media.
            output_video:  Desired path for the SR output media.
            log_dir:       Optional directory for per-stage log file.
            pipeline_log:  Optional pipeline-wide log file.

        Returns:
            ``output_video`` on success.

        Raises:
            SystemExit: If SR fails and ``empty_output_policy="fail"``.
        """
        input_video = Path(input_video)
        output_video = Path(output_video)

        is_image_input = is_image_path(input_video)
        if is_image_input and not is_image_path(output_video):
            raise RuntimeError(
                f"[sr] image input requires an image output path (e.g. .png/.jpg). got output={output_video}"
            )

        # The SeedVR2 image path supports real model inference, but upstream requires
        # sequence parallel size 1 for single-image inputs.
        run_sp_size = 1 if is_image_input else self._sp_size
        sr_visible_gpus = ",".join(str(g) for g in self._gpu_list[:run_sp_size])
        extra_env = {
            "CUDA_VISIBLE_DEVICES": sr_visible_gpus,
            "SEEDVR_ROOT": str(self._seedvr_root),
        }
        self.logger.info(
            "[sr] GPU routing: physical CUDA devices %s exposed to SeedVR subprocess as local cuda:0..%d",
            sr_visible_gpus,
            run_sp_size - 1,
        )
        self.logger.info(
            "[sr] config: variant=%s resolution=%dx%d window_frames=%d overlap_frames=%d window_timeout=%ds",
            self._variant,
            self._res_w,
            self._res_h,
            self._window_frames,
            self._overlap_frames,
            self._window_timeout_s,
        )

        # Intermediate SR work directories under the scene's unified sidecar work area.
        sr_work_dir = output_video.parent / "_work" / "sr"
        sr_tmp_root = sr_work_dir / "_tmp_window_segments"
        sr_out_dir = sr_work_dir / "out"
        ensure_dir(sr_tmp_root)
        ensure_dir(sr_out_dir)

        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={run_sp_size}",
            "--module",
            self._sr_module,
            "--variant",
            self._variant,
            "--video_path",
            str(input_video),
            "--output_dir",
            str(sr_out_dir),
            "--output_path",
            str(output_video),
            "--tmp_dir",
            str(sr_tmp_root),
            "--seed",
            str(self._seed),
            "--res_h",
            str(self._res_h),
            "--res_w",
            str(self._res_w),
            "--sp_size",
            str(run_sp_size),
            "--window_frames",
            str(self._window_frames),
            "--overlap_frames",
            str(self._overlap_frames),
            "--window_timeout",
            str(self._window_timeout_s),
        ]
        if self._out_fps is not None:
            cmd += ["--out_fps", str(self._out_fps)]

        sr_ok = False
        try:
            run_cmd(
                name="sr",
                cmd=cmd,
                cwd=self._sr_cwd,
                extra_env=extra_env,
                log_dir=log_dir,
                pipeline_log=pipeline_log,
                logger=self.logger,
            )
            sr_ok = output_video.exists()
            if not sr_ok:
                msg = f"[sr] expected output media not found: {output_video}"
                decode_hint = _seedvr_decode_failure_hint(log_dir)
                if decode_hint:
                    msg = f"{msg}; {decode_hint}"
                if self._empty_output_policy == "fail":
                    raise SystemExit(msg)
        except SystemExit as exc:
            msg = f"[sr] subprocess exited with non-zero code: {exc.code}"
            decode_hint = _seedvr_decode_failure_hint(log_dir)
            if decode_hint:
                msg = f"{msg}; {decode_hint}"
            if self._empty_output_policy == "fail":
                raise SystemExit(msg) from exc
        except Exception as exc:
            msg = f"[sr] stage raised {exc.__class__.__name__}: {exc}"
            if self._empty_output_policy == "fail":
                raise SystemExit(msg) from exc

        # Best-effort cleanup: remove heavy intermediate artifacts when SR succeeded.
        # Keep on failure for debugging; also keep on LOG_LEVEL=DEBUG.
        if sr_ok and str(os.getenv("LOG_LEVEL", "")).strip().upper() != "DEBUG":
            try:
                shutil.rmtree(sr_work_dir, ignore_errors=True)
            except Exception:
                pass

        return output_video


def _variant_mode(variant: str) -> str:
    v = str(variant or "").strip().lower()
    if "3b" in v:
        return "3b"
    return "7b"


def _seedvr_decode_failure_hint(log_dir: Optional[Path]) -> Optional[str]:
    """Return an actionable hint when the SR subprocess log shows decode failure."""
    if log_dir is None:
        return None

    log_path = Path(log_dir) / "sr.log"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if unsupported_video_decoder_seen(text):
        return unsupported_video_decoder_message("SeedVR")
    if video_decode_failure_seen(text):
        return video_decode_failure_message("SeedVR")
    return None


def _ensure_symlink(*, link_path: Path, target: Path, description: str) -> None:
    """Create or update a symlink without overwriting non-empty real paths."""
    link_path = Path(link_path)
    target = Path(target).expanduser().resolve()
    if link_path.is_symlink():
        current_target = link_path.resolve(strict=False)
        if current_target == target:
            return
        link_path.unlink()
    elif link_path.exists():
        try:
            if link_path.resolve() == target:
                return
        except OSError:
            pass
        if link_path.is_dir() and not any(link_path.iterdir()):
            link_path.rmdir()
        else:
            raise RuntimeError(
                f"[sr] cannot create SeedVR {description} symlink at {link_path}; path already exists and is not "
                "an empty directory or symlink. Remove it or set pipeline.model_cache_path to a clean cache."
            )

    try:
        link_path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        raise RuntimeError(f"[sr] failed to create SeedVR {description} symlink {link_path} -> {target}") from exc


def _prepare_seedvr_runtime_root(
    *,
    source_root: Path,
    runtime_root: Path,
    ckpts_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Build a writable SeedVR runtime root with only the paths inference needs."""
    source_root = Path(source_root).expanduser().resolve()
    runtime_root = Path(runtime_root).expanduser().resolve()
    ckpts_dir = Path(ckpts_dir).expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    expected_links = set(_SEEDVR_RUNTIME_LINKS) | {"ckpts"}
    for stale in runtime_root.iterdir():
        if stale.name not in expected_links and stale.is_symlink():
            stale.unlink()

    for name in _SEEDVR_RUNTIME_LINKS:
        item = source_root / name
        if not item.exists():
            raise RuntimeError(f"[sr] SeedVR source root is missing required path: {item}")
        _ensure_symlink(link_path=runtime_root / name, target=item, description=name)
    _ensure_symlink(link_path=runtime_root / "ckpts", target=ckpts_dir, description="ckpts")

    logger.info("[sr] SeedVR runtime root: %s (source=%s, ckpts=%s)", runtime_root, source_root, ckpts_dir)
    return runtime_root


def ensure_seedvr2_ckpts(*, ckpts_root: Path, variant: str, logger: logging.Logger) -> None:
    """
    Ensure SeedVR2 checkpoints exist under <ckpts_root>/seedvr2.

    This is intentionally called from SeedVR2Resolver.__init__ so stage init is sufficient to
    validate/download required weights.
    """
    mode = _variant_mode(variant)
    seedvr_dir = Path(ckpts_root).expanduser().resolve() / "seedvr2"
    seedvr_dir.mkdir(parents=True, exist_ok=True)

    # Required files by mode.
    required: list[str] = ["ema_vae.pth"]
    if mode == "3b":
        required.append("seedvr2_ema_3b.pth")
    else:
        required.append("seedvr2_ema_7b.pth")

    missing = [f for f in required if not (seedvr_dir / f).exists()]
    if not missing:
        return

    logger.info("[sr] missing ckpts (%s): %s", mode, ", ".join(missing))

    hf_token = str(os.getenv("HF_TOKEN", "")).strip() or None
    # Default to official repos, but allow override for mirrors/internal forks.
    hf_repo_3b = str(os.getenv("HF_REPO_SEEDVR2_3B", "")).strip() or "ByteDance-Seed/SeedVR2-3B"
    hf_repo_7b = str(os.getenv("HF_REPO_SEEDVR2_7B", "")).strip() or "ByteDance-Seed/SeedVR2-7B"

    # Warn early when ckpts are missing and no HF token is set.
    if hf_token is None:
        logger.warning(
            "[sr] HF_TOKEN is not set. If the HuggingFace repo is gated/private, download may fail. "
            "To avoid this, set HF_TOKEN (with access)."
        )

    def _ensure_one(filename: str) -> None:
        dst = seedvr_dir / filename
        if dst.exists() and dst.stat().st_size > 0:
            return
        # HuggingFace download (official repos; token only needed if gated).
        if filename == "ema_vae.pth":
            # Prefer 7B then 3B.
            last_exc: Exception | None = None
            for repo_id in (hf_repo_7b, hf_repo_3b):
                try:
                    logger.info("[sr] hf download %s:%s -> %s", repo_id, filename, dst)
                    ensure_hf_file(repo_id=repo_id, filename=filename, dst=dst, hf_token=hf_token, timeout_s=600)
                    return
                except Exception as e:  # noqa: BLE001
                    last_exc = e
            raise RuntimeError(f"Failed to download {filename} from HuggingFace repos. Set HF_TOKEN.") from last_exc

        # Model ckpts live in their respective repos.
        if filename == "seedvr2_ema_3b.pth":
            repo_id = hf_repo_3b
        else:
            repo_id = hf_repo_7b

        logger.info("[sr] hf download %s:%s -> %s", repo_id, filename, dst)
        ensure_hf_file(repo_id=repo_id, filename=filename, dst=dst, hf_token=hf_token, timeout_s=600)

    for f in required:
        _ensure_one(f)

    still_missing = [f for f in required if not (seedvr_dir / f).exists()]
    if still_missing:
        raise RuntimeError(f"[sr] ckpts still missing after download: {', '.join(still_missing)} (dir={seedvr_dir})")
