#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored notice:
# - This file is derived from upstream SeedVR/SeedVR2 inference code (Apache-2.0).
# - SPDX-License-Identifier remains Apache-2.0 for compliance; NVIDIA modifications are covered by the
#   SPDX-FileCopyrightText line(s) above.
"""
SeedVR2 long-video inference via sliding windows + overlap stitching.

This script is intentionally standalone: it does NOT modify the original inference scripts.
Instead, it imports them as "variants" and reuses their:
- configure_runner(sp_size)
- generation_step(runner, text_embeds_dict, cond_latents)

Design goals:
- Handle long videos without decoding / processing the full T frames at once.
- Keep overlap alignment stable (so blending does not produce ghosting).
- Fail-fast per video: the first failed window stops that video, but the batch continues.
- Write failures to a single log (failures.log); do not crash the whole batch for partial failures.

Decoding:
- Uses torchvision.io.VideoReader in streaming mode to build windows in frame-space.
  This avoids seek-based decoding misalignment that can cause overlap "ghosting".

Stitching:
- Default: blend overlap frames.
- Use --no_blend_overlap to disable blending (overlap frames are dropped from the later window).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import importlib.util
import logging
import os
import shutil
import signal
import tempfile
import traceback
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional, Tuple

import av
import mediapy
import numpy as np
import torch
from common.distributed import get_device
from common.distributed.advanced import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from common.partition import partition_by_groups, partition_by_size
from common.seed import set_seed
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
from einops import rearrange
from torchvision.io import VideoReader, read_image, write_video
from torchvision.transforms import Compose, Lambda, Normalize
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Keep these standalone: this script runs inside the SeedVR torchrun runtime,
# where repo-local al_utils modules are not necessarily importable. If the
# canonical media policy changes in al_utils.media_paths, update these literals
# too; tests assert the two allowlists stay in sync.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}

# MP4 writer encoder selection. This image's FFmpeg is built LGPL-pure and
# ships exactly two encoders that can legally produce MP4 output:
#   * h264_nvenc (HW) - preferred. Requires a GPU with NVENC silicon
#     (consumer RTX 20+/40/50, RTX PRO, A10/A40/L40-class GPUs). Datacenter SKUs without
#     NVENC (e.g. H100 NVL) and CI runners without a GPU will fail the probe
#     below and fall back to mpeg4.
#   * mpeg4 (SW) - MPEG-4 Part 2 fallback. Lower compression efficiency than
#     H.264 but LGPL-clean and runs anywhere. Selected only when NVENC is
#     unavailable.
_NVENC_PROBE_LOCK = Lock()
_NVENC_PROBE_RESULT: bool | None = None
_MPEG4_FALLBACK_WARNED = False


class WindowTimeoutError(TimeoutError):
    pass


def _is_nvenc_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "h264_nvenc",
            "libnvidia-encode.so",
            "nvenc",
            "minimum required nvidia driver",
        )
    )


def _probe_nvenc_available(*, force: bool = False) -> bool:
    global _NVENC_PROBE_RESULT
    with _NVENC_PROBE_LOCK:
        if _NVENC_PROBE_RESULT is not None and not force:
            return _NVENC_PROBE_RESULT

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            probe_path = Path(tmp.name)
        try:
            container = av.open(str(probe_path), mode="w")
            try:
                stream = container.add_stream("h264_nvenc", rate=Fraction(30, 1))
                stream.width = 256
                stream.height = 256
                stream.pix_fmt = "yuv420p"
                stream.options = {"preset": "p4", "tune": "hq"}

                blank = np.zeros((256, 256, 3), dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(blank, format="rgb24")):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            finally:
                container.close()

            _NVENC_PROBE_RESULT = True
        except Exception as exc:
            # Probe only: _select_video_encoder() converts a False result into
            # the intended mpeg4 fallback; unrelated write errors should surface.
            if _is_nvenc_error(exc):
                _NVENC_PROBE_RESULT = False
            else:
                raise
        finally:
            try:
                os.remove(probe_path)
            except OSError:
                pass

    return _NVENC_PROBE_RESULT


def _select_video_encoder() -> tuple[str, dict[str, str]]:
    """Return (codec_name, encoder_options) for PyAV stream creation.

    Prefers h264_nvenc; falls back to mpeg4 if NVENC is unavailable. The
    mpeg4 fallback is logged once per process so the operator notices the
    lower-quality output codec.
    """
    if _probe_nvenc_available():
        return "h264_nvenc", {"preset": "p4", "tune": "hq"}
    _warn_mpeg4_fallback_once()
    return "mpeg4", {}


def _warn_mpeg4_fallback_once() -> None:
    global _MPEG4_FALLBACK_WARNED
    if not _MPEG4_FALLBACK_WARNED:
        logger.warning(
            "h264_nvenc unavailable; SR video writer falling back to mpeg4 (LGPL-clean, lower compression than H.264)."
        )
        _MPEG4_FALLBACK_WARNED = True


@contextmanager
def _window_timeout(seconds: int, *, label: str):
    """Best-effort POSIX SIGALRM timeout around Python-level SR window work.

    This cannot interrupt in-progress CUDA kernels or native calls such as
    vae.encode(), generation_step(), or diffusion model forward passes. Wall
    time can therefore exceed the configured timeout until control returns to
    Python. Non-POSIX platforms are handled below by disabling the alarm.
    """
    timeout_s = int(seconds or 0)
    if timeout_s <= 0 or os.name != "posix":
        yield
        return

    def _handle_timeout(signum, frame):  # noqa: ARG001
        raise WindowTimeoutError(f"{label} timed out after {timeout_s}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTS


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return isinstance(err, RuntimeError) and (
        "cuda out of memory" in msg
        or "out of memory" in msg
        or ("cublas" in msg and "alloc" in msg)
        or ("memory" in msg and "allocation" in msg)
    )


def _cleanup_cuda():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _cleanup_distributed():
    """Best-effort shutdown for torchrun ranks before process exit."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as exc:
        logger.debug("failed to destroy process group: %s", exc)


def _install_shutdown_handlers():
    """Clean up NCCL process groups on common external termination signals."""

    def _handle_shutdown(signum, frame):  # noqa: ARG001
        _cleanup_distributed()
        _cleanup_cuda()
        raise SystemExit(128 + int(signum))

    if os.name == "posix":
        signal.signal(signal.SIGTERM, _handle_shutdown)
        signal.signal(signal.SIGINT, _handle_shutdown)
    else:
        logger.warning("shutdown signal handlers are not installed on unsupported OS type: %s", os.name)


def _is_cuda_oom_any(err: BaseException) -> bool:
    """Return True if any exception in the chain looks like a CUDA OOM."""
    cur: Optional[BaseException] = err
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _is_cuda_oom(cur):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _blend_overlap(prev_tail: torch.Tensor, curr_head: torch.Tensor) -> torch.Tensor:
    """
    Blend overlap frames linearly.
    prev_tail, curr_head: (T, C, H, W), float32 in [0,1]
    """
    t = int(prev_tail.shape[0])
    if t <= 0:
        return curr_head
    w = torch.linspace(0.0, 1.0, steps=t, device=prev_tail.device).view(t, 1, 1, 1)
    return prev_tail * (1.0 - w) + curr_head * w


def _stitch_segments(segments: List[torch.Tensor], overlap_frames: int, *, blend: bool) -> torch.Tensor:
    """
    segments: list of (T, C, H, W) float32 in [0,1]
    Returns concatenated (T_total, C, H, W)

    If blend=False, we drop the overlap frames from the later segment (simple stitching).
    """
    if not segments:
        return torch.empty(0)
    if overlap_frames <= 0:
        return torch.cat(segments, dim=0)
    out = segments[0]
    for seg in segments[1:]:
        ov = min(overlap_frames, out.shape[0], seg.shape[0])
        if ov <= 0:
            out = torch.cat([out, seg], dim=0)
            continue
        if blend:
            blended = _blend_overlap(out[-ov:], seg[:ov])
            out = torch.cat([out[:-ov], blended, seg[ov:]], dim=0)
        else:
            out = torch.cat([out, seg[ov:]], dim=0)
    return out


def _iter_windows_by_streaming(
    path: str,
    *,
    window_frames: int,
    overlap_frames: int,
) -> Iterable[Tuple[int, int, torch.Tensor]]:
    """Yield windows by sequential decode so overlaps share the exact same frames.

    Yields: (window_index, start_frame_index, video_TCHW_uint8)
    """
    if window_frames <= 0:
        raise ValueError("window_frames must be > 0")
    overlap_frames = max(0, int(overlap_frames))
    stride = max(1, int(window_frames - overlap_frames))

    vr = VideoReader(path, "video")
    buf: List[torch.Tensor] = []

    def _stack(frames_hwc: List[torch.Tensor]) -> torch.Tensor:
        # VideoReader returns HWC uint8; convert to TCHW uint8.
        if not frames_hwc:
            return torch.empty((0, 3, 1, 1), dtype=torch.uint8)
        f0 = frames_hwc[0]
        if f0.ndim == 3 and f0.shape[-1] in (1, 3, 4):  # HWC
            frames_chw = [f[..., :3].permute(2, 0, 1).contiguous() for f in frames_hwc]
        elif f0.ndim == 3 and f0.shape[0] in (1, 3, 4):  # CHW
            frames_chw = [f[:3].contiguous() for f in frames_hwc]
        else:
            raise ValueError(f"Unexpected frame tensor shape from VideoReader: {tuple(f0.shape)}")
        return torch.stack(frames_chw, dim=0)

    for item in vr:
        buf.append(item["data"])
        if len(buf) >= window_frames:
            break

    if not buf:
        yield 0, 0, torch.empty((0, 3, 1, 1), dtype=torch.uint8)
        return

    win_i = 0
    start_idx = 0
    yield win_i, start_idx, _stack(buf)
    win_i += 1

    overlap_keep = overlap_frames
    while True:
        prefix = buf[-overlap_keep:] if overlap_keep > 0 else []
        new_frames: List[torch.Tensor] = []
        for item in vr:
            new_frames.append(item["data"])
            if len(new_frames) >= stride:
                break
        if not new_frames:
            break
        buf = prefix + new_frames
        start_idx = start_idx + stride
        yield win_i, start_idx, _stack(buf)
        win_i += 1


def _write_output(sample_tchw_01: torch.Tensor, filename: str, fps: float):
    sample = rearrange(sample_tchw_01, "t c h w -> t h w c")
    sample = (sample * 255.0).round().clamp(0, 255).to(torch.uint8)
    if torch.is_tensor(sample):
        sample = sample.detach().cpu().numpy()
    sample = np.asarray(sample, dtype=np.uint8)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if sample.shape[0] == 1:
        mediapy.write_image(filename, sample.squeeze(0))
    else:
        try:
            mediapy.write_video(filename, sample, fps=float(fps))
        except Exception:
            write_video(filename, torch.from_numpy(sample), fps=float(fps))


def _safe_stem(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    keep = []
    for ch in stem:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
        else:
            keep.append("_")
    stem = "".join(keep).strip().replace(os.sep, "_")
    return stem or "video"


def _write_video_streaming_from_segments_u8(
    segment_paths: List[str],
    out_path: str,
    *,
    fps: float,
    overlap_frames: int,
    blend: bool,
) -> None:
    if not segment_paths:
        raise RuntimeError("No segments to stitch.")

    overlap_frames = max(0, int(overlap_frames))

    def _load_u8(path: str) -> torch.Tensor:
        t = torch.load(path, map_location="cpu")
        if not torch.is_tensor(t):
            raise TypeError(f"Expected a torch Tensor in {path}, got {type(t)}")
        if t.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 tensor in {path}, got {t.dtype}")
        if t.ndim != 4:
            raise ValueError(f"Expected (T,C,H,W) in {path}, got shape {tuple(t.shape)}")
        return t.contiguous()

    def _write_u8_tchw(container, stream, tchw_u8: torch.Tensor) -> None:
        if tchw_u8.numel() == 0:
            return
        thwc = tchw_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        thwc = np.asarray(thwc, dtype=np.uint8)
        for frame in thwc:
            vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)

    def _blend_u8(prev_tail_u8: torch.Tensor, curr_head_u8: torch.Tensor) -> torch.Tensor:
        t = min(int(prev_tail_u8.shape[0]), int(curr_head_u8.shape[0]))
        if t <= 0:
            return torch.empty((0,) + tuple(prev_tail_u8.shape[1:]), dtype=torch.uint8)
        a = prev_tail_u8[-t:].float().div(255.0)
        b = curr_head_u8[:t].float().div(255.0)
        blended = _blend_overlap(a, b)
        return (blended * 255.0).round().clamp(0, 255).to(torch.uint8)

    first = _load_u8(segment_paths[0])
    h = int(first.shape[2])
    w = int(first.shape[3])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    codec_name, codec_options = _select_video_encoder()

    def _write_with_codec(selected_codec: str, selected_options: dict[str, str]) -> None:
        container = av.open(out_path, mode="w")
        try:
            stream = container.add_stream(selected_codec, rate=Fraction(fps).limit_denominator(1000))
            if selected_options:
                stream.options = selected_options
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"

            if overlap_frames <= 0:
                for p in segment_paths:
                    seg = _load_u8(p)
                    _write_u8_tchw(container, stream, seg)
            else:
                if not blend:
                    seg0 = first
                    _write_u8_tchw(container, stream, seg0)
                    for p in segment_paths[1:]:
                        seg = _load_u8(p)
                        ov = min(overlap_frames, int(seg.shape[0]))
                        _write_u8_tchw(container, stream, seg[ov:])
                else:
                    seg0 = first
                    if int(seg0.shape[0]) > overlap_frames:
                        _write_u8_tchw(container, stream, seg0[:-overlap_frames])
                        prev_tail = seg0[-overlap_frames:]
                    else:
                        prev_tail = seg0

                    for p in segment_paths[1:]:
                        seg = _load_u8(p)
                        ov = min(overlap_frames, int(seg.shape[0]), int(prev_tail.shape[0]))
                        head = seg[:ov]

                        if int(prev_tail.shape[0]) > ov:
                            _write_u8_tchw(container, stream, prev_tail[:-ov])
                        blended_u8 = _blend_u8(prev_tail, head)
                        _write_u8_tchw(container, stream, blended_u8)

                        rest = seg[ov:]
                        if int(rest.shape[0]) > overlap_frames:
                            _write_u8_tchw(container, stream, rest[:-overlap_frames])
                            prev_tail = rest[-overlap_frames:]
                        else:
                            prev_tail = rest

                    _write_u8_tchw(container, stream, prev_tail)

            for pkt in stream.encode(None):
                container.mux(pkt)
        finally:
            container.close()

    try:
        _write_with_codec(codec_name, codec_options)
    except Exception as exc:
        if codec_name != "h264_nvenc" or not _is_nvenc_error(exc):
            raise

        global _NVENC_PROBE_RESULT
        with _NVENC_PROBE_LOCK:
            _NVENC_PROBE_RESULT = False
        _warn_mpeg4_fallback_once()
        try:
            os.remove(out_path)
        except OSError:
            pass
        _write_with_codec("mpeg4", {})


def _resolve_variant_module(variant: str):
    mapping = {
        "seedvr2_3b": ("projects.inference_seedvr2_3b", "inference_seedvr2_3b.py"),
        "seedvr2_7b": ("projects.inference_seedvr2_7b", "inference_seedvr2_7b.py"),
    }
    if variant not in mapping:
        raise ValueError(f"Unknown --variant: {variant}. Choose one of: {', '.join(mapping.keys())}")

    module_name, rel_py = mapping[variant]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    py_path = os.path.join(this_dir, rel_py)
    seedvr_root = str(os.getenv("SEEDVR_ROOT", "")).strip()

    # Some upstream SeedVR scripts check for optional resources via relative paths at import-time
    # (e.g. "./projects/video_diffusion_sr/color_fix.py"). Import them from the SeedVR repo root
    # when available so those checks behave as intended.
    old_cwd = os.getcwd()
    try:
        if seedvr_root and os.path.isdir(seedvr_root):
            os.chdir(seedvr_root)

        if os.path.isfile(py_path):
            unique_name = f"_seedvr_variant_{variant}"
            spec = importlib.util.spec_from_file_location(unique_name, py_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load variant module spec for {variant} from {py_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        return __import__(module_name, fromlist=["*"])
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass


def configure_runner(sp_size: int, *, variant: str = "seedvr2_3b"):
    module = _resolve_variant_module(str(variant))
    # Upstream SeedVR variant scripts assume they are executed from the SeedVR repo root
    # (they load configs like "./configs_7b/main.yaml"). When we run from a different cwd
    # (e.g. /workspace), those relative paths break. Fix by temporarily switching cwd.
    seedvr_root = str(os.getenv("SEEDVR_ROOT", "")).strip()
    old_cwd = os.getcwd()
    try:
        if seedvr_root and os.path.isdir(seedvr_root):
            os.chdir(seedvr_root)
        runner = module.configure_runner(int(sp_size))
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
    setattr(runner, "_seedvr_window_variant", module)
    return runner


def generation_step(runner, text_embeds_dict, cond_latents):
    module = getattr(runner, "_seedvr_window_variant", None)
    if module is None:
        raise RuntimeError("Runner is missing '_seedvr_window_variant'.")
    return module.generation_step(runner, text_embeds_dict, cond_latents=cond_latents)


def generation_loop(
    runner,
    video_path: str = "./test_videos",
    output_dir: str = "./results",
    output_path: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    batch_size: int = 1,
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.0,
    sample_steps: int = 1,
    seed: int = 666,
    res_h: int = 720,
    res_w: int = 1280,
    sp_size: int = 1,
    out_fps: Optional[float] = None,
    window_frames: int = 128,
    overlap_frames: int = 64,
    window_timeout: int = 3600,
    no_blend_overlap: bool = False,
    variant: str = "seedvr2_3b",
):
    module = getattr(runner, "_seedvr_window_variant", None)
    if module is None:
        raise RuntimeError("Runner is missing '_seedvr_window_variant'.")

    os.makedirs(output_dir, exist_ok=True)
    failure_log_path = os.path.join(output_dir, "failures.log")
    tgt_path = output_dir

    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    runner.configure_diffusion()

    set_seed(seed, same_across_ranks=True)

    if os.path.isdir(video_path):
        if output_path:
            raise ValueError("--output_path requires --video_path to be a single file (not a directory).")
        video_root = video_path
        video_list_for_prompts = os.listdir(video_root)
    else:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"--video_path not found: {video_path}")
        video_root = os.path.dirname(video_path) or "."
        video_list_for_prompts = [os.path.basename(video_path)]

    original_videos = []
    for f in video_list_for_prompts:
        if is_image_file(f) or os.path.splitext(f.lower())[1] in VIDEO_EXTS:
            original_videos.append(f)
    print(f"Total prompts to be generated: {len(original_videos)}")

    original_videos_group = partition_by_groups(
        original_videos,
        get_data_parallel_world_size() // get_sequence_parallel_world_size(),
    )
    original_videos_local = original_videos_group[get_data_parallel_rank() // get_sequence_parallel_world_size()]
    original_videos_local = partition_by_size(original_videos_local, batch_size)

    def _extract_text_embeds():
        def _find_asset(name: str) -> Path:
            # Prefer cwd (matches upstream behavior), but fall back to SEEDVR_ROOT
            # so this runner can be executed from other working directories.
            candidates: list[Path] = [Path.cwd() / name]
            seedvr_root = str(os.getenv("SEEDVR_ROOT", "")).strip()
            if seedvr_root:
                candidates.append(Path(seedvr_root) / name)
            for p in candidates:
                if p.exists():
                    return p
            raise FileNotFoundError(
                f"Expected {name} in current working directory or $SEEDVR_ROOT. "
                f"cwd={Path.cwd()} SEEDVR_ROOT={seedvr_root!r}"
            )

        pos_path = _find_asset("pos_emb.pt")
        neg_path = _find_asset("neg_emb.pt")
        text_pos_embeds = torch.load(str(pos_path), map_location="cpu")
        text_neg_embeds = torch.load(str(neg_path), map_location="cpu")
        return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}

    positive_prompts_embeds = []
    for _ in tqdm(original_videos_local):
        positive_prompts_embeds.append(_extract_text_embeds())
    gc.collect()
    torch.cuda.empty_cache()

    video_transform = Compose(
        [
            NaResize(
                resolution=(res_h * res_w) ** 0.5,
                mode="area",
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ]
    )

    def cut_videos(videos, sp_size):
        t = videos.size(1)
        if t == 1:
            return videos
        if t <= 4 * sp_size:
            padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos

    def _infer_fps(src_path: str, out_fps: float | None) -> float:
        if out_fps is not None:
            return float(out_fps)
        try:
            vr = VideoReader(src_path, "video")
            md = vr.get_metadata()
            fps = None
            if isinstance(md, dict):
                v = md.get("video", None)
                if isinstance(v, dict):
                    fps = v.get("fps", None)
            if isinstance(fps, (list, tuple)) and fps:
                fps = fps[0]
            if fps is not None and float(fps) > 0:
                return float(fps)
        except Exception:
            pass
        try:
            with av.open(src_path) as c:
                if c.streams.video:
                    st = c.streams.video[0]
                    rate = st.average_rate or st.base_rate
                    if rate is not None:
                        return float(rate)
        except Exception:
            pass
        return 30.0

    for videos, text_embeds in tqdm(zip(original_videos_local, positive_prompts_embeds)):
        for i, emb in enumerate(text_embeds["texts_pos"]):
            text_embeds["texts_pos"][i] = emb.to(get_device())
        for i, emb in enumerate(text_embeds["texts_neg"]):
            text_embeds["texts_neg"][i] = emb.to(get_device())

        for video in videos:
            src_path = os.path.abspath(os.path.join(video_root, video))
            if output_path:
                out_file = os.path.abspath(str(output_path))
            else:
                out_file = os.path.join(tgt_path, os.path.basename(video))

            try:
                if is_image_file(video):
                    with _window_timeout(
                        window_timeout,
                        label=f"SR image inference for {src_path}",
                    ):
                        if sp_size > 1:
                            raise ValueError("Sp size should be set to 1 for image inputs!")
                        img = read_image(src_path).unsqueeze(0) / 255.0
                        cond = video_transform(img.to(get_device()))
                        cond_cut = cut_videos(cond, sp_size)

                        runner.dit.to("cpu")
                        runner.vae.to(get_device())
                        cond_latents = runner.vae_encode([cond_cut])
                        runner.vae.to("cpu")
                        runner.dit.to(get_device())

                        set_seed(seed, same_across_ranks=True)
                        samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
                        runner.dit.to("cpu")

                        if get_sequence_parallel_rank() == 0:
                            sample = samples[0].to("cpu")
                            out_01 = sample.clip(-1, 1).mul_(0.5).add_(0.5).float()
                            _write_output(out_01, out_file, fps=float(out_fps or 24.0))
                    _cleanup_cuda()
                    continue

                save_fps = _infer_fps(src_path, out_fps)
                tmp_root = os.path.abspath(str(tmp_dir)) if tmp_dir else os.path.join(tgt_path, "_tmp_window_segments")
                tmp_dir = os.path.join(
                    tmp_root,
                    f"{_safe_stem(video)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}",
                )
                segment_paths: List[str] = []
                if get_sequence_parallel_rank() == 0:
                    os.makedirs(tmp_dir, exist_ok=True)
                window_iter = _iter_windows_by_streaming(
                    src_path, window_frames=int(window_frames), overlap_frames=int(overlap_frames)
                )
                for win_i, start_frame, win_tchw_u8 in window_iter:
                    with _window_timeout(
                        window_timeout,
                        label=f"SR window {win_i} for {src_path} starting at frame {start_frame}",
                    ):
                        if win_tchw_u8.numel() == 0 or (hasattr(win_tchw_u8, "shape") and win_tchw_u8.shape[0] == 0):
                            raise ValueError(
                                f"decode_failed: got 0 frames for {src_path} window starting at frame {start_frame}"
                            )
                        win = win_tchw_u8 / 255.0
                        cond = video_transform(win.to(get_device()))
                        ori_len = int(cond.size(1))
                        cond_cut = cut_videos(cond, sp_size)

                        runner.dit.to("cpu")
                        runner.vae.to(get_device())
                        cond_latents = runner.vae_encode([cond_cut])
                        runner.vae.to("cpu")
                        runner.dit.to(get_device())

                        set_seed(seed, same_across_ranks=True)
                        samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
                        runner.dit.to("cpu")

                        if get_sequence_parallel_rank() == 0:
                            sample = samples[0]
                            if ori_len < sample.shape[0]:
                                sample = sample[:ori_len]
                            if getattr(module, "use_colorfix", False):
                                inp_tchw = rearrange(cond, "c t h w -> t c h w")
                                sample = module.wavelet_reconstruction(
                                    sample.to("cpu"), inp_tchw[: sample.size(0)].to("cpu")
                                )
                            else:
                                sample = sample.to("cpu")
                            out_01 = sample.clip(-1, 1).mul_(0.5).add_(0.5).float()
                            out_u8 = (out_01 * 255.0).round().clamp(0, 255).to(torch.uint8)
                            seg_path = os.path.join(tmp_dir, f"seg_{win_i:06d}_start{start_frame:09d}.pt")
                            torch.save(out_u8.contiguous().cpu(), seg_path)
                            segment_paths.append(seg_path)

                    _cleanup_cuda()

                if get_sequence_parallel_rank() == 0:
                    if not segment_paths:
                        raise RuntimeError(f"No successful windows for {src_path} (see failures.log)")
                    _write_video_streaming_from_segments_u8(
                        segment_paths,
                        out_file,
                        fps=float(save_fps),
                        overlap_frames=int(overlap_frames),
                        blend=not bool(no_blend_overlap),
                    )
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    # Best-effort cleanup of the temp root if it's now empty.
                    try:
                        if tmp_root and os.path.isdir(tmp_root) and not os.listdir(tmp_root):
                            os.rmdir(tmp_root)
                    except Exception:
                        pass

            except Exception as e:
                # In single-file mode we re-raise; still try to clean temp segments if they were created.
                try:
                    if "tmp_dir" in locals() and isinstance(tmp_dir, str):
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass
                if not os.path.isdir(video_path):
                    raise
                if get_sequence_parallel_rank() == 0:
                    kind = "OOM" if _is_cuda_oom_any(e) else "ERROR"
                    os.makedirs(os.path.dirname(failure_log_path) or ".", exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.datetime.now().isoformat()}] {kind}: {src_path}\n")
                        f.write(f"{str(e)}\n")
                        f.write(traceback.format_exc())
                        f.write("\n")
                    try:
                        if "tmp_dir" in locals() and isinstance(tmp_dir, str):
                            shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass
                _cleanup_cuda()
                continue

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="seedvr2_3b", choices=["seedvr2_3b", "seedvr2_7b"])
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional: write output to this exact file path (single-file mode only).",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="",
        help="Optional: directory root for window segment temp files (defaults to <output_dir>/_tmp_window_segments).",
    )
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    parser.add_argument("--window_frames", type=int, default=128, help="Frames per window.")
    parser.add_argument("--overlap_frames", type=int, default=64, help="Overlap frames between windows.")
    parser.add_argument(
        "--window_timeout",
        type=int,
        default=3600,
        help="Wall-clock timeout in seconds for each SR window; <=0 disables the per-window timeout.",
    )
    parser.add_argument(
        "--no_blend_overlap", action="store_true", help="Disable overlap blending (overlap frames will be dropped)."
    )
    args = parser.parse_args()

    args.output_path = args.output_path.strip()
    args.tmp_dir = args.tmp_dir.strip()
    if not args.output_path:
        args.output_path = None
    if not args.tmp_dir:
        args.tmp_dir = None
    _install_shutdown_handlers()
    try:
        runner = configure_runner(args.sp_size, variant=args.variant)
        generation_loop(runner, **vars(args))
    finally:
        _cleanup_distributed()
        _cleanup_cuda()
