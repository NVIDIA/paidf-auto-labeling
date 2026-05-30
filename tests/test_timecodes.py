# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
from daft_export.timecodes import frames_to_timecode, seconds_to_timecode, timecode_to_seconds

DAFT_TIMECODE_RE = re.compile(r"^(\d{2}:)?\d{2}:\d{2}(\.\d+)?$")


class TestSecondsToTimecode:
    def test_zero(self):
        assert seconds_to_timecode(0) == "00:00"

    def test_sub_second(self):
        assert seconds_to_timecode(0.5) == "00:00.500"

    def test_whole_seconds(self):
        assert seconds_to_timecode(5.0) == "00:05"

    def test_minutes_and_seconds(self):
        assert seconds_to_timecode(95.5) == "01:35.500"

    def test_exactly_one_hour(self):
        assert seconds_to_timecode(3600.0) == "01:00:00"

    def test_over_one_hour(self):
        assert seconds_to_timecode(3723.0) == "01:02:03"

    def test_over_one_hour_with_fraction(self):
        assert seconds_to_timecode(3723.456) == "01:02:03.456"

    def test_small_fraction_rounds_to_millisecond(self):
        tc = seconds_to_timecode(1.0006)
        assert tc == "00:01.001"

    def test_fraction_rounds_away(self):
        tc = seconds_to_timecode(1.0001)
        assert tc == "00:01"

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            seconds_to_timecode(-1.0)

    @pytest.mark.parametrize("secs", [0, 0.33, 1.5, 59.999, 60, 95.5, 3599.99, 3600, 7261.123])
    def test_always_matches_daft_regex(self, secs):
        tc = seconds_to_timecode(secs)
        assert DAFT_TIMECODE_RE.match(tc), f"{tc!r} doesn't match DAFT pattern"


class TestFramesToTimecode:
    def test_frame_zero(self):
        assert frames_to_timecode(0, 30.0) == "00:00"

    def test_one_second(self):
        assert frames_to_timecode(30, 30.0) == "00:01"

    def test_fractional_fps(self):
        tc = frames_to_timecode(450, 30.0)
        assert tc == "00:15"

    def test_non_integer_result(self):
        tc = frames_to_timecode(1, 30.0)
        assert tc == "00:00.033"

    def test_zero_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            frames_to_timecode(10, 0)

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            frames_to_timecode(10, -24.0)


class TestTimecodeToSeconds:
    def test_mm_ss(self):
        assert timecode_to_seconds("01:35") == 95.0

    def test_mm_ss_frac(self):
        assert timecode_to_seconds("01:35.500") == 95.5

    def test_hh_mm_ss(self):
        assert timecode_to_seconds("01:02:03") == 3723.0

    def test_hh_mm_ss_frac(self):
        assert timecode_to_seconds("01:02:03.456") == pytest.approx(3723.456)

    def test_zero(self):
        assert timecode_to_seconds("00:00") == 0.0

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="invalid DAFT timecode"):
            timecode_to_seconds("5")

    def test_single_digit_rejected(self):
        with pytest.raises(ValueError, match="invalid DAFT timecode"):
            timecode_to_seconds("1:02")


class TestRoundTrip:
    @pytest.mark.parametrize("secs", [0, 0.5, 1.0, 59.0, 60.0, 95.5, 3600.0, 3723.456])
    def test_seconds_round_trip(self, secs):
        tc = seconds_to_timecode(secs)
        result = timecode_to_seconds(tc)
        assert result == pytest.approx(secs, abs=1e-3)

    @pytest.mark.parametrize("frame", [0, 1, 30, 450, 1800, 108000])
    def test_frames_round_trip(self, frame):
        fps = 30.0
        tc = frames_to_timecode(frame, fps)
        recovered = timecode_to_seconds(tc)
        assert recovered == pytest.approx(frame / fps, abs=1e-3)
