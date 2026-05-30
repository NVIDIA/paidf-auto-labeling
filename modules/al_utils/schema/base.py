# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

_REQUIRED_SENTINEL = "__REQUIRED__"


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _clean_optional_str(value: Any, *, field: str) -> Optional[str]:
    v = _clean_str(value)
    if v == _REQUIRED_SENTINEL:
        raise ValueError(f"{field} must be set (do not leave as {_REQUIRED_SENTINEL!r})")
    if not v:
        return None
    return v


def _clean_optional_str_allow_required_sentinel(value: Any, *, field: str) -> Optional[str]:
    """
    Like _clean_optional_str, but treats the required sentinel as "unset".

    This is useful for fields that are optional at the schema level and only become
    required under certain stage combinations (validated elsewhere).
    """
    v = _clean_str(value)
    if not v or v == _REQUIRED_SENTINEL:
        return None
    return v


def _clean_required_str(value: Any, *, field: str) -> str:
    v = _clean_str(value)
    if not v or v == _REQUIRED_SENTINEL:
        raise ValueError(f"{field} is required and cannot be empty (do not leave as {_REQUIRED_SENTINEL!r})")
    return v
