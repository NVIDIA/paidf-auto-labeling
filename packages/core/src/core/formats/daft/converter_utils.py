# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small validation helpers shared by DAFT converters."""

from __future__ import annotations

import math
from typing import Any

from core.formats.daft.errors import DaftConvertError
from core.formats.daft.timecodes import seconds_to_timecode, timecode_to_seconds


def require_nonempty_string(
    value: Any,
    *,
    field: str,
    prefix: str = "",
    quote_field: bool = False,
) -> str:
    """Return ``value.strip()`` when it is a non-empty string."""
    field_label = repr(field) if quote_field else field
    subject = f"{prefix} {field_label}".strip()
    if not isinstance(value, str):
        raise DaftConvertError(f"{subject} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise DaftConvertError(f"{subject} is empty after strip()")
    return stripped


def clamp_nonnegative_seconds(seconds: float, duration: float | None) -> float:
    """Clamp seconds to ``[0, duration]`` when a duration is supplied."""
    clamped = max(0.0, seconds)
    if duration is not None:
        clamped = min(clamped, duration)
    return clamped


def validate_nonnegative_duration(
    duration: float | None,
    *,
    field: str,
) -> float | None:
    """Return duration as a finite non-negative float."""
    if duration is None:
        return None
    duration_s = float(duration)
    if not math.isfinite(duration_s) or duration_s < 0:
        raise DaftConvertError(f"{field} must be finite and >= 0, got {duration!r}")
    return duration_s


def coerce_timecode(
    value: Any,
    *,
    field: str,
    duration: float | None,
    duration_field: str,
) -> str:
    """Accept numeric seconds or DAFT timecode strings and return a timecode."""
    if isinstance(value, bool) or value is None:
        raise DaftConvertError(f"{field} must be a number or timecode string, got {value!r}")

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise DaftConvertError(f"{field} is empty")
        try:
            seconds = timecode_to_seconds(stripped)
        except ValueError as exc:
            raise DaftConvertError(
                f"{field} is not a valid DAFT timecode: {stripped!r} ({exc})"
            ) from exc
    elif isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds):
            raise DaftConvertError(f"{field} must be a finite number, got {seconds}")
        if seconds < 0:
            raise DaftConvertError(f"{field} must be >= 0, got {seconds}")
    else:
        raise DaftConvertError(
            f"{field} must be a number or timecode string, got {type(value).__name__}"
        )

    seconds = max(seconds, 0.0)
    if duration is not None:
        duration_s = float(duration)
        invalid_duration = not math.isfinite(duration_s) or duration_s < 0
        if invalid_duration:
            raise DaftConvertError(f"{duration_field} must be finite and >= 0, got {duration!r}")
        seconds = min(seconds, duration_s)
    return seconds_to_timecode(seconds)


__all__ = [
    "clamp_nonnegative_seconds",
    "coerce_timecode",
    "require_nonempty_string",
    "validate_nonnegative_duration",
]
