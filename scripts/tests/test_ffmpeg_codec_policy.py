# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ffmpeg_codec_policy import (
    _EXPECTED_FFMPEG_COMPONENTS,
    _forbidden_ffmpeg_component_names,
)


def test_validator_expected_encoders_are_vp9_and_internal_rawvideo() -> None:
    assert _EXPECTED_FFMPEG_COMPONENTS["encoders"] == {"libvpx-vp9", "rawvideo"}


def test_validator_allows_mpeg4_part2_and_its_ffmpeg_h263_dependency() -> None:
    assert _EXPECTED_FFMPEG_COMPONENTS["decoders"] == {
        "h263",
        "h264_cuvid",
        "mpeg4",
        "vp9",
        "vp9_cuvid",
    }


def test_validator_allows_ffmpeg_vp9_bitstream_filters() -> None:
    assert {"vp9_superframe", "vp9_superframe_split"} <= _EXPECTED_FFMPEG_COMPONENTS[
        "bitstream filters"
    ]


def test_validator_rejects_h264_software_codecs() -> None:
    findings = _forbidden_ffmpeg_component_names(
        ["libvpx-vp9", "rawvideo", "libx264", "libopenh264", "h264"],
        allowed_components=_EXPECTED_FFMPEG_COMPONENTS["encoders"],
    )

    assert findings == ["h264", "libopenh264", "libx264"]
