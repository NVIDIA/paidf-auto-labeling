# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timecode helpers for DAFT (``MM:SS`` / ``HH:MM:SS`` with optional ``.ms``)."""

from __future__ import annotations

import re

# Pattern enforced by the DAFT schema on every timecode field.
_TIMECODE_RE = re.compile(r"^(\d{2}:)?\d{2}:\d{2}(\.\d+)?$")


def seconds_to_timecode(seconds: float) -> str:
    """Convert seconds to a DAFT timecode string.

    Returns ``MM:SS`` for durations under one hour, ``HH:MM:SS`` otherwise.
    Fractional seconds are included as milliseconds when non-zero.

    The output always matches the schema pattern ``_TIMECODE_RE``.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")

    total_ms = round(seconds * 1000)
    frac_ms = total_ms % 1000
    total_secs = total_ms // 1000

    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60

    frac = f".{frac_ms:03d}" if frac_ms else ""

    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}{frac}"
    return f"{mins:02d}:{secs:02d}{frac}"


def frames_to_timecode(frame: int, fps: float) -> str:
    """Convert a frame number to a DAFT timecode string."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return seconds_to_timecode(frame / fps)


def timecode_to_seconds(tc: str) -> float:
    """Parse a DAFT timecode string back to seconds."""
    if not _TIMECODE_RE.match(tc):
        raise ValueError(f"invalid DAFT timecode: {tc!r}")

    parts = tc.split(":")
    if len(parts) == 2:
        mins, secs = parts
        hrs = "0"
    else:
        hrs, mins, secs = parts

    return int(hrs) * 3600 + int(mins) * 60 + float(secs)
