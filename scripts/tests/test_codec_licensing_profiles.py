# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ffmpeg_codec_policy
import pytest
from ffmpeg_codec_policy import (
    COMMON_INPUT_FLAGS,
    expected_ffmpeg_components,
    flags_for_profile,
)


def test_vp9_output_profile_includes_common_input_and_vp9_output() -> None:
    flags = flags_for_profile("vp9-output")

    assert set(COMMON_INPUT_FLAGS) <= set(flags)
    assert "--enable-encoder=libvpx_vp9" in flags
    assert "--enable-encoder=h264_nvenc" not in flags
    assert "--enable-filter=select" in flags
    assert "--enable-filter=setpts" in flags
    assert "--enable-decoder=mpeg4" in flags
    assert "--enable-parser=mpeg4video" in flags


def test_common_input_flags_enable_window_filters() -> None:
    """Window decode/re-encode needs select+setpts under --disable-everything."""
    assert "--enable-filter=select" in COMMON_INPUT_FLAGS
    assert "--enable-filter=setpts" in COMMON_INPUT_FLAGS


def test_input_only_profile_does_not_force_persisted_output_codec() -> None:
    flags = flags_for_profile("input-only")

    assert "--enable-encoder=rawvideo" in flags
    assert "--enable-encoder=libvpx_vp9" not in flags


def test_input_only_licensing_profile_keeps_common_decoders() -> None:
    components = expected_ffmpeg_components("input-only")

    assert components["encoders"] == {"rawvideo"}
    assert components["decoders"] == {
        "h263",
        "h264_cuvid",
        "mpeg4",
        "vp9",
        "vp9_cuvid",
    }


def test_mpeg4_profile_does_not_explicitly_enable_h263_input() -> None:
    assert "--enable-decoder=h263" not in flags_for_profile("input-only")


def test_vp9_output_licensing_profile_adds_vp9_encoder() -> None:
    components = expected_ffmpeg_components("vp9-output")

    assert components["encoders"] == {"libvpx-vp9", "rawvideo"}


def test_unknown_licensing_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="input-only, vp9-output"):
        expected_ffmpeg_components("unknown")


def test_unknown_configure_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="input-only, vp9-output"):
        flags_for_profile("unknown")


def test_internal_policy_cli_only_accepts_configure_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = ffmpeg_codec_policy.main(["ffmpeg_codec_policy.py", "--profile"])

    assert result == 2
    assert "configure-flags <profile>" in capsys.readouterr().err
