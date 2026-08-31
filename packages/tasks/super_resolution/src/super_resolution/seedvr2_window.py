# Vendored notice:
# - This file is vendored/adapted from upstream ByteDance-Seed/SeedVR2 (Apache-2.0): https://github.com/ByteDance-Seed/SeedVR
#
# License details can be found in the UPSTREAM_LICENSE.md file.

# SPDX-FileCopyrightText: Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

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
- Uses a selected FFmpeg hardware/software decoder in streaming mode to build windows
  in frame-space. This avoids seek-based decoding misalignment that can cause overlap
  "ghosting".

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
import sys
import traceback
import types
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

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
from core.media.video_codecs import (
    VideoDecodePlan,
    iter_decoded_rgb24_frames,
    probe_video_stream,
)
from core.media.vp9_output import configure_vp9_output_stream
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
from einops import rearrange
from torchvision.io import read_image, write_video
from torchvision.transforms import Compose, Lambda, Normalize
from tqdm import tqdm

from super_resolution.media_formats import (
    is_supported_sr_image_path,
    is_supported_sr_media_path,
    is_supported_sr_video_path,
    sr_output_name_for_input,
    sr_supported_extensions_message,
)

logger = logging.getLogger(__name__)


def is_image_file(filename: str) -> bool:
    return is_supported_sr_image_path(filename)


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return isinstance(err, RuntimeError) and (
        "cuda out of memory" in msg
        or "out of memory" in msg
        or ("cublas" in msg and "alloc" in msg)
        or ("memory" in msg and "allocation" in msg)
    )


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        logger.debug("CUDA cache cleanup failed.", exc_info=True)


def _is_cuda_oom_any(err: BaseException) -> bool:
    """Return True if any exception in the chain looks like a CUDA OOM."""
    cur: BaseException | None = err
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
    blended: torch.Tensor = prev_tail * (1.0 - w) + curr_head * w
    return blended


def _stitch_segments(
    segments: list[torch.Tensor],
    overlap_frames: int,
    *,
    blend: bool,
) -> torch.Tensor:
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


def _iter_decoded_tensors(path: str, plan: VideoDecodePlan) -> Iterable[torch.Tensor]:
    """Adapt dependency-light core RGB frame bytes to SeedVR2 HWC tensors."""
    for payload in iter_decoded_rgb24_frames(path, plan):
        array = np.frombuffer(payload, dtype=np.uint8).copy()
        array = array.reshape(plan.stream.height, plan.stream.width, 3)
        yield torch.from_numpy(array)


def _iter_windows_by_streaming(
    path: str,
    *,
    decoder_name: str,
    window_frames: int,
    overlap_frames: int,
) -> Iterable[tuple[int, int, torch.Tensor]]:
    """Yield windows by sequential decode so overlaps share the exact same frames.

    Yields: (window_index, start_frame_index, video_TCHW_uint8)
    """
    if window_frames <= 0:
        raise ValueError("window_frames must be > 0")
    overlap_frames = int(overlap_frames)
    if overlap_frames < 0 or overlap_frames >= window_frames:
        raise ValueError(
            "overlap_frames must be >= 0 and < window_frames "
            f"(overlap_frames={overlap_frames}, window_frames={window_frames}); "
            "then stride = window_frames - overlap_frames is guaranteed >= 1."
        )
    stride = window_frames - overlap_frames

    decode_plan = VideoDecodePlan(
        stream=probe_video_stream(path),
        decoder_name=decoder_name,
    )
    frames = iter(_iter_decoded_tensors(path, decode_plan))
    buf: list[torch.Tensor] = []

    def _stack(frames_hwc: list[torch.Tensor]) -> torch.Tensor:
        # The decoder returns HWC uint8; convert to TCHW uint8.
        if not frames_hwc:
            return torch.empty((0, 3, 1, 1), dtype=torch.uint8)
        f0 = frames_hwc[0]
        if f0.ndim == 3 and f0.shape[-1] in (1, 3, 4):  # HWC
            frames_chw = [f[..., :3].permute(2, 0, 1).contiguous() for f in frames_hwc]
        elif f0.ndim == 3 and f0.shape[0] in (1, 3, 4):  # CHW
            frames_chw = [f[:3].contiguous() for f in frames_hwc]
        else:
            raise ValueError(f"Unexpected decoded frame tensor shape: {tuple(f0.shape)}")
        return torch.stack(frames_chw, dim=0)

    for frame in frames:
        buf.append(frame)
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
        new_frames: list[torch.Tensor] = []
        for frame in frames:
            new_frames.append(frame)
            if len(new_frames) >= stride:
                break
        if not new_frames:
            break
        buf = prefix + new_frames
        start_idx = start_idx + stride
        yield win_i, start_idx, _stack(buf)
        win_i += 1


def _write_output(sample_tchw_01: torch.Tensor, filename: str, fps: float) -> None:
    sample_thwc = rearrange(sample_tchw_01, "t c h w -> t h w c")
    sample_u8 = (sample_thwc * 255.0).round().clamp(0, 255).to(torch.uint8)
    sample = np.asarray(sample_u8.detach().cpu().numpy(), dtype=np.uint8)
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


def _output_name_for_input(f: str) -> str:
    stem = os.path.splitext(os.path.basename(f))[0]
    return sr_output_name_for_input(f, stem=stem)


def _write_video_streaming_from_segments_u8(
    segment_paths: list[str],
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

    def _write_u8_tchw(container: Any, stream: Any, tchw_u8: torch.Tensor) -> None:
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

    def _write_segments(container: Any, stream: Any) -> None:
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

    def _write_once() -> None:
        container = av.open(out_path, mode="w")
        try:
            stream: Any = container.add_stream(
                "libvpx-vp9",
                rate=Fraction(fps).limit_denominator(1000),
            )
            configure_vp9_output_stream(stream, width=w, height=h)
            _write_segments(container, stream)
        finally:
            container.close()

    _write_once()


def _resolve_variant_module(variant: str) -> types.ModuleType:
    mapping = {
        "seedvr2_3b": ("projects.inference_seedvr2_3b", "inference_seedvr2_3b.py"),
        "seedvr2_7b": ("projects.inference_seedvr2_7b", "inference_seedvr2_7b.py"),
    }
    if variant not in mapping:
        raise ValueError(
            f"Unknown --variant: {variant}. Choose one of: {', '.join(mapping.keys())}"
        )

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
                raise ImportError(
                    f"Failed to load variant module spec for {variant} from {py_path}"
                )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        return __import__(module_name, fromlist=["*"])
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            logger.debug("Failed to restore working directory.", exc_info=True)


def _install_apex_normalization_fallback() -> types.ModuleType:
    try:
        from apex import normalization as _normalization  # noqa: PLC0415

        return cast(types.ModuleType, _normalization)
    except ModuleNotFoundError:
        pass

    from diffusers.models.normalization import RMSNorm  # noqa: PLC0415

    # Without a valid __spec__, importlib.util.find_spec("apex") raises, and
    # diffusers' apex check calls find_spec — turning "apex absent" into a crash.
    apex_module = sys.modules.get("apex")
    if apex_module is None:
        apex_module = types.ModuleType("apex")
        sys.modules["apex"] = apex_module
    if getattr(apex_module, "__spec__", None) is None:
        apex_module.__spec__ = importlib.util.spec_from_loader("apex", loader=None)

    normalization_module = types.ModuleType("apex.normalization")
    normalization_module.__spec__ = importlib.util.spec_from_loader(
        "apex.normalization", loader=None
    )
    normalization_module.__dict__["FusedLayerNorm"] = torch.nn.LayerNorm

    class FusedRMSNorm(RMSNorm):
        def __init__(
            self,
            normalized_shape: int | list[int] | tuple[int, ...],
            elementwise_affine: bool = True,
            eps: float = 1e-5,
        ) -> None:
            if isinstance(normalized_shape, (list, tuple)):
                dim = int(normalized_shape[0])
            else:
                dim = int(normalized_shape)
            super().__init__(dim=dim, eps=eps, elementwise_affine=elementwise_affine)

    normalization_module.__dict__["FusedRMSNorm"] = FusedRMSNorm
    apex_module.__dict__["normalization"] = normalization_module
    sys.modules["apex.normalization"] = normalization_module
    return normalization_module


def configure_runner(sp_size: int, *, variant: str = "seedvr2_3b") -> Any:
    module = _resolve_variant_module(str(variant))
    _install_apex_normalization_fallback()
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
            logger.debug("Failed to restore working directory.", exc_info=True)
    runner._seedvr_window_variant = module
    return runner


def generation_step(runner: Any, text_embeds_dict: dict[str, Any], cond_latents: Any) -> Any:
    module = getattr(runner, "_seedvr_window_variant", None)
    if module is None:
        raise RuntimeError("Runner is missing '_seedvr_window_variant'.")
    return module.generation_step(runner, text_embeds_dict, cond_latents=cond_latents)


def generation_loop(
    runner: Any,
    video_path: str = "./test_videos",
    output_dir: str = "./results",
    output_path: str | None = None,
    tmp_dir: str | None = None,
    batch_size: int = 1,
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.0,
    sample_steps: int = 1,
    seed: int = 666,
    res_h: int = 720,
    res_w: int = 1280,
    sp_size: int = 1,
    out_fps: float | None = None,
    window_frames: int = 128,
    overlap_frames: int = 64,
    no_blend_overlap: bool = False,
    video_decoder: str = "vp9",
    variant: str = "seedvr2_3b",
) -> None:
    module = getattr(runner, "_seedvr_window_variant", None)
    if module is None:
        raise RuntimeError("Runner is missing '_seedvr_window_variant'.")

    os.makedirs(output_dir, exist_ok=True)
    failure_log_path = os.path.join(output_dir, "failures.log")
    tgt_path = output_dir
    tmp_root = (
        os.path.abspath(str(tmp_dir)) if tmp_dir else os.path.join(tgt_path, "_tmp_window_segments")
    )

    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    runner.configure_diffusion()

    set_seed(seed, same_across_ranks=True)

    if os.path.isdir(video_path):
        if output_path:
            raise ValueError(
                "--output_path requires --video_path to be a single file (not a directory)."
            )
        video_root = video_path
        video_list_for_prompts = sorted(os.listdir(video_root))
    else:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"--video_path not found: {video_path}")
        if not is_supported_sr_media_path(video_path):
            raise ValueError(
                f"Unsupported --video_path extension for {video_path!r}; "
                f"{sr_supported_extensions_message()}"
            )
        video_root = os.path.dirname(video_path) or "."
        video_list_for_prompts = [os.path.basename(video_path)]

    original_videos: list[str] = []
    for f in video_list_for_prompts:
        if is_supported_sr_media_path(f):
            original_videos.append(f)
    print(f"Total prompts to be generated: {len(original_videos)}")

    original_videos_group = partition_by_groups(
        original_videos,
        get_data_parallel_world_size() // get_sequence_parallel_world_size(),
    )
    original_videos_local = original_videos_group[
        get_data_parallel_rank() // get_sequence_parallel_world_size()
    ]
    original_videos_local = partition_by_size(original_videos_local, batch_size)

    def _extract_text_embeds() -> dict[str, list[torch.Tensor]]:
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
        text_pos_embeds = cast(torch.Tensor, torch.load(str(pos_path), map_location="cpu")).cpu()
        text_neg_embeds = cast(torch.Tensor, torch.load(str(neg_path), map_location="cpu")).cpu()
        return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}

    positive_prompts_embeds: list[dict[str, list[torch.Tensor]]] = []
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

    def cut_videos(videos: torch.Tensor, sp_size: int) -> torch.Tensor:
        t = videos.size(1)
        if t == 1:
            return videos
        if t <= 4 * sp_size:
            padding_frames = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding_frames, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        padding_frames = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
        padding = torch.cat(padding_frames, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos

    def _infer_fps(src_path: str, out_fps: float | None) -> float:
        if out_fps is not None:
            return float(out_fps)
        try:
            fps = probe_video_stream(src_path).fps
            if fps is not None:
                return fps
        except Exception:
            logger.debug("Video metadata FPS inference failed.", exc_info=True)
        try:
            with av.open(src_path) as c:
                if c.streams.video:
                    st = c.streams.video[0]
                    rate = st.average_rate or st.base_rate
                    if rate is not None:
                        return float(rate)
        except Exception:
            logger.debug("PyAV FPS inference failed.", exc_info=True)
        return 30.0

    def _to_device_text_embeds(
        cpu_text_embeds: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        return {
            key: [emb.to(get_device()) for emb in embeds] for key, embeds in cpu_text_embeds.items()
        }

    video_prompt_pairs = zip(original_videos_local, positive_prompts_embeds, strict=True)
    for videos, cpu_text_embeds in tqdm(video_prompt_pairs):
        for video in videos:
            src_path = os.path.abspath(os.path.join(video_root, video))
            if output_path:
                out_file = os.path.abspath(str(output_path))
            else:
                out_file = os.path.join(tgt_path, _output_name_for_input(video))
            per_video_tmp_dir: str | None = None

            try:
                if is_supported_sr_image_path(video):
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
                    temp_device_embeds: dict[str, list[torch.Tensor]] | None = None
                    try:
                        temp_device_embeds = _to_device_text_embeds(cpu_text_embeds)
                        samples = generation_step(
                            runner, temp_device_embeds, cond_latents=cond_latents
                        )
                    finally:
                        del temp_device_embeds
                        _cleanup_cuda()
                    runner.dit.to("cpu")

                    if get_sequence_parallel_rank() == 0:
                        sample = samples[0].to("cpu")
                        out_01 = sample.clip(-1, 1).mul_(0.5).add_(0.5).float()
                        _write_output(out_01, out_file, fps=float(out_fps or 24.0))
                    _cleanup_cuda()
                    continue

                if not is_supported_sr_video_path(video):
                    raise ValueError(
                        f"Unsupported video extension for {src_path!r}; "
                        f"{sr_supported_extensions_message()}"
                    )
                save_fps = _infer_fps(src_path, out_fps)
                per_video_tmp_dir = os.path.join(
                    tmp_root,
                    f"{_safe_stem(video)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}",
                )
                segment_paths: list[str] = []
                if get_sequence_parallel_rank() == 0:
                    os.makedirs(per_video_tmp_dir, exist_ok=True)
                window_iter = _iter_windows_by_streaming(
                    src_path,
                    decoder_name=video_decoder,
                    window_frames=int(window_frames),
                    overlap_frames=int(overlap_frames),
                )
                for win_i, start_frame, win_tchw_u8 in window_iter:
                    if win_tchw_u8.numel() == 0 or (
                        hasattr(win_tchw_u8, "shape") and win_tchw_u8.shape[0] == 0
                    ):
                        raise ValueError(
                            "decode_failed: got 0 frames for "
                            f"{src_path} window starting at frame {start_frame}"
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
                    temp_device_embeds = None
                    try:
                        temp_device_embeds = _to_device_text_embeds(cpu_text_embeds)
                        samples = generation_step(
                            runner, temp_device_embeds, cond_latents=cond_latents
                        )
                    finally:
                        del temp_device_embeds
                        _cleanup_cuda()
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
                        seg_path = os.path.join(
                            per_video_tmp_dir, f"seg_{win_i:06d}_start{start_frame:09d}.pt"
                        )
                        torch.save(out_u8.contiguous().cpu(), seg_path)
                        segment_paths.append(seg_path)

                    _cleanup_cuda()

                if get_sequence_parallel_rank() == 0:
                    if not segment_paths:
                        raise RuntimeError(
                            f"No successful windows for {src_path} (see failures.log)"
                        )
                    _write_video_streaming_from_segments_u8(
                        segment_paths,
                        out_file,
                        fps=float(save_fps),
                        overlap_frames=int(overlap_frames),
                        blend=not bool(no_blend_overlap),
                    )
                    shutil.rmtree(per_video_tmp_dir, ignore_errors=True)
                    # Best-effort cleanup of the temp root if it's now empty.
                    try:
                        if tmp_root and os.path.isdir(tmp_root) and not os.listdir(tmp_root):
                            os.rmdir(tmp_root)
                    except Exception:
                        logger.debug("Failed to remove empty temp root.", exc_info=True)

            except Exception as exc:
                # In single-file mode we re-raise; still try to clean temp segments if
                # they were created.
                try:
                    if per_video_tmp_dir is not None:
                        shutil.rmtree(per_video_tmp_dir, ignore_errors=True)
                except Exception:
                    logger.debug("Failed to clean per-video temp directory.", exc_info=True)
                if not os.path.isdir(video_path):
                    raise
                if get_sequence_parallel_rank() == 0:
                    kind = "OOM" if _is_cuda_oom_any(exc) else "ERROR"
                    os.makedirs(os.path.dirname(failure_log_path) or ".", exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(
                            f"[{datetime.datetime.now().isoformat()}] {kind}: {src_path}\n"
                        )
                        log_file.write(f"{str(exc)}\n")
                        log_file.write(traceback.format_exc())
                        log_file.write("\n")
                    try:
                        if per_video_tmp_dir is not None:
                            shutil.rmtree(per_video_tmp_dir, ignore_errors=True)
                    except Exception:
                        logger.debug("Failed to clean per-video temp directory.", exc_info=True)
                _cleanup_cuda()
                continue

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", type=str, default="seedvr2_3b", choices=["seedvr2_3b", "seedvr2_7b"]
    )
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
        help=(
            "Optional: directory root for window segment temp files "
            "(defaults to <output_dir>/_tmp_window_segments)."
        ),
    )
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    parser.add_argument("--window_frames", type=int, default=128, help="Frames per window.")
    parser.add_argument(
        "--overlap_frames", type=int, default=64, help="Overlap frames between windows."
    )
    parser.add_argument(
        "--no_blend_overlap",
        action="store_true",
        help="Disable overlap blending (overlap frames will be dropped).",
    )
    parser.add_argument(
        "--video_decoder",
        choices=["h264_cuvid", "vp9_cuvid", "vp9", "mpeg4"],
        default="vp9",
    )
    args = parser.parse_args()

    runner = configure_runner(args.sp_size, variant=args.variant)
    args.output_path = args.output_path.strip()
    args.tmp_dir = args.tmp_dir.strip()
    if not args.output_path:
        args.output_path = None
    if not args.tmp_dir:
        args.tmp_dir = None
    generation_loop(runner, **vars(args))
