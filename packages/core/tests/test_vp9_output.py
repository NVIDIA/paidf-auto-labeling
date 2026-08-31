# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from core.media import VP9_ENCODER_OPTIONS, configure_vp9_output_stream


def test_configure_vp9_output_stream_sets_quality_and_color_metadata() -> None:
    class FakeCodecContext:
        color_range = 0
        color_primaries = 0
        color_trc = 0
        colorspace = 0

    class FakeStream:
        width = 0
        height = 0
        pix_fmt = ""

        def __init__(self) -> None:
            self.options: dict[str, str] = {}
            self.codec_context = FakeCodecContext()

    stream = FakeStream()

    configure_vp9_output_stream(stream, width=1280, height=720)

    assert stream.width == 1280
    assert stream.height == 720
    assert stream.pix_fmt == "yuv420p"
    assert stream.options == VP9_ENCODER_OPTIONS
    assert stream.options is not VP9_ENCODER_OPTIONS
    assert stream.codec_context.color_range == 1
    assert stream.codec_context.color_primaries == 1
    assert stream.codec_context.color_trc == 1
    assert stream.codec_context.colorspace == 1
