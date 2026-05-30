# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for classifying video decode failures from stage logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

UNSUPPORTED_VIDEO_DECODER_MESSAGE_TEMPLATE = (
    "Unsupported video decoder: {stage} could not read this video format. "
    "Detected reason: the available H.264 hardware decoder does not support this input encoding/profile "
    "(commonly H.264 High 4:4:4 Predictive / yuv444p). "
    "With pipeline.empty_output_policy=warn, this stage is skipped and downstream stages continue "
    "from the original input or the latest successfully generated media. "
    "Use a supported encoding such as H.264 Main/yuv420p, or an approved image that includes software H.264 decode."
)

VIDEO_DECODE_FAILURE_MESSAGE_TEMPLATE = (
    "Video decode failure: {stage} could not read frames from this video. "
    "Detected reason: the enabled decoder for this input failed while decoding. "
    "With pipeline.empty_output_policy=warn, this stage is skipped and downstream stages continue "
    "from the original input or the latest successfully generated media. "
    "Supported decode paths in this image include H.264 via h264_cuvid/NVDEC, MPEG-4 Part 2 via mpeg4, "
    "and MJPEG via mjpeg."
)


@dataclass(frozen=True)
class DecodeFallbackClassification:
    reason: str
    message: str
    detected_from_video_decode: bool = False


def unsupported_video_decoder_message(stage: str) -> str:
    return UNSUPPORTED_VIDEO_DECODER_MESSAGE_TEMPLATE.format(stage=stage)


def video_decode_failure_message(stage: str) -> str:
    return VIDEO_DECODE_FAILURE_MESSAGE_TEMPLATE.format(stage=stage)


def h264_stream_requires_unsupported_decoder(stream: Mapping[str, object]) -> bool:
    codec = str(stream.get("codec_name") or "").strip().lower()
    if codec not in {"h264", "avc1"}:
        return False

    profile = str(stream.get("profile") or "").strip().lower()
    pix_fmt = str(stream.get("pix_fmt") or "").strip().lower()

    if "4:2:2" in profile or "4:4:4" in profile:
        return True
    if "422" in pix_fmt or "444" in pix_fmt:
        return True

    # The FFmpeg build intentionally ships only h264_cuvid for H.264 decode.
    # Keep this gate conservative: non-420 H.264 paths are known to fail or hang
    # before pipeline fallback can run.
    known_420_pix_fmts = {"", "yuv420p", "yuvj420p", "nv12"}
    known_420_profiles = {"", "baseline", "constrained baseline", "main", "high"}
    return pix_fmt not in known_420_pix_fmts or profile not in known_420_profiles


def classify_decode_fallback(
    *,
    log_path: Optional[Path],
    stage_label: str,
    default_reason: str,
    default_message: str,
    extra_text: str = "",
) -> DecodeFallbackClassification:
    """Classify media-read failures without losing the generic fallback path."""
    text = read_log_text(log_path)
    if extra_text:
        text = f"{text}\n{extra_text}"
    if unsupported_video_decoder_seen(text):
        return DecodeFallbackClassification(
            reason="unsupported_video_decoder",
            message=unsupported_video_decoder_message(stage_label),
            detected_from_video_decode=True,
        )
    if video_decode_failure_seen(text):
        return DecodeFallbackClassification(
            reason="video_decode_failed",
            message=video_decode_failure_message(stage_label),
            detected_from_video_decode=True,
        )
    return DecodeFallbackClassification(reason=default_reason, message=default_message)


def read_log_text(log_path: Optional[Path]) -> str:
    if log_path is None:
        return ""
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def unsupported_video_decoder_seen(text: str) -> bool:
    return "h264_cuvid" in text and (
        "CUDA_ERROR_NOT_SUPPORTED" in text
        or "cuvid decode callback error" in text
        or "avcodec_send_packet()" in text
        or "Generic error in an external library" in text
    )


def video_decode_failure_seen(text: str) -> bool:
    lower_text = text.lower()
    if not any(codec in lower_text for codec in ("h264_cuvid", "mpeg4", "mjpeg")):
        return False
    if any(marker in lower_text for marker in ("out of memory", "cuda out of memory", "oom")):
        return False
    return any(
        marker in lower_text
        for marker in (
            "error while decoding",
            "failed to decode",
            "error submitting packet to decoder",
            "invalid data found when processing input",
            "could not find codec parameters",
            "avcodec_send_packet()",
        )
    )
