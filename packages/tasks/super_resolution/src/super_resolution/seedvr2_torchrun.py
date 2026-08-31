# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torchrun module that adapts SeedVR2 to explicit model paths."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> None:
    """Run SeedVR2 window inference with checkpoints from ``SEEDVR_CKPT_DIR``."""
    args = _parse_args(argv)
    seedvr_root = _require_dir_env("SEEDVR_ROOT")
    ckpt_dir = _require_dir_env("SEEDVR_CKPT_DIR")
    _prepend_sys_path(seedvr_root)
    _patch_seedvr_checkpoint_paths(ckpt_dir)

    old_cwd = os.getcwd()
    try:
        os.chdir(seedvr_root)
        window_module = importlib.import_module("super_resolution.seedvr2_window")
        runner = window_module.configure_runner(args.sp_size, variant=args.variant)
        args.output_path = args.output_path.strip() or None
        args.tmp_dir = args.tmp_dir.strip() or None
        window_module.generation_loop(runner, **vars(args))
    finally:
        os.chdir(old_cwd)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["seedvr2_3b", "seedvr2_7b"],
        default="seedvr2_3b",
    )
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--tmp_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    parser.add_argument("--window_frames", type=int, default=128)
    parser.add_argument("--overlap_frames", type=int, default=64)
    parser.add_argument(
        "--video_decoder",
        choices=["h264_cuvid", "vp9_cuvid", "vp9", "mpeg4"],
        default="vp9",
    )
    parser.add_argument("--no_blend_overlap", action="store_true")
    return parser.parse_args(argv)


def _require_dir_env(name: str) -> Path:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"{name} must be set for SeedVR2 inference.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{name} must point to an existing directory: {path}")
    return path


def _prepend_sys_path(path: Path) -> None:
    rendered = str(path)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)


def _patch_seedvr_checkpoint_paths(ckpt_dir: Path) -> None:
    infer_module = importlib.import_module("projects.video_diffusion_sr.infer")
    video_diffusion_infer = infer_module.VideoDiffusionInfer
    original_dit = video_diffusion_infer.configure_dit_model
    original_vae = video_diffusion_infer.configure_vae_model

    def configure_dit_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "checkpoint" in kwargs:
            kwargs["checkpoint"] = _rewrite_checkpoint(kwargs["checkpoint"], ckpt_dir)
        return original_dit(self, *args, **kwargs)

    def configure_vae_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            self.config.vae.checkpoint = _rewrite_checkpoint(
                self.config.vae.checkpoint,
                ckpt_dir,
            )
        except AttributeError:
            pass
        return original_vae(self, *args, **kwargs)

    video_diffusion_infer.configure_dit_model = configure_dit_model
    video_diffusion_infer.configure_vae_model = configure_vae_model


def _rewrite_checkpoint(value: object, ckpt_dir: Path) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    if normalized.startswith("./ckpts/") or normalized.startswith("ckpts/"):
        return str(ckpt_dir / Path(normalized).name)
    return value


if __name__ == "__main__":
    main()
