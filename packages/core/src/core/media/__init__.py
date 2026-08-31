# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, dependency-light media utilities."""

from core.media.image_groups import discover_image_group
from core.media.video_codecs import (
    H264_GPU_REQUIRED_MESSAGE,
    UnsupportedVideoCodecError,
    VideoDecodeError,
    VideoDecodePlan,
    VideoDecoderUnavailableError,
    VideoStreamInfo,
    iter_decoded_rgb24_frames,
    prepare_video_decode,
    probe_video_stream,
)
from core.media.vp9_output import VP9_ENCODER_OPTIONS, Vp9VideoWriter, configure_vp9_output_stream

__all__ = [
    "H264_GPU_REQUIRED_MESSAGE",
    "UnsupportedVideoCodecError",
    "VideoDecodeError",
    "VideoDecodePlan",
    "VideoDecoderUnavailableError",
    "VideoStreamInfo",
    "VP9_ENCODER_OPTIONS",
    "Vp9VideoWriter",
    "configure_vp9_output_stream",
    "discover_image_group",
    "iter_decoded_rgb24_frames",
    "prepare_video_decode",
    "probe_video_stream",
]
