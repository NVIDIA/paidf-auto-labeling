# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for al_utils.common.resolve_gpu_list()."""

from __future__ import annotations

import pytest
from al_utils.common import resolve_gpu_list


def test_int_zero_returns_zero() -> None:
    """int 0 must not be treated as falsy and resolved to 'all'."""
    assert resolve_gpu_list(0) == [0]


def test_int_positive() -> None:
    assert resolve_gpu_list(3) == [3]


def test_int_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        resolve_gpu_list(-1)


def test_string_single() -> None:
    assert resolve_gpu_list("0") == [0]


def test_string_comma_separated() -> None:
    assert resolve_gpu_list("2,3") == [2, 3]


def test_string_whitespace_stripped() -> None:
    assert resolve_gpu_list(" 2 , 3 ") == [2, 3]


def test_string_all_returns_nonempty() -> None:
    result = resolve_gpu_list("all")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(g, int) and g >= 0 for g in result)


def test_none_returns_nonempty() -> None:
    result = resolve_gpu_list(None)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_empty_string_returns_nonempty() -> None:
    result = resolve_gpu_list("")
    assert isinstance(result, list)
    assert len(result) >= 1


def test_nonnumeric_raises() -> None:
    with pytest.raises(ValueError, match="Invalid GPU ID"):
        resolve_gpu_list("abc")


def test_float_string_raises() -> None:
    with pytest.raises(ValueError, match="Invalid GPU ID"):
        resolve_gpu_list("2.0")


def test_negative_string_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        resolve_gpu_list("-1")


def test_empty_elements_skipped() -> None:
    assert resolve_gpu_list("2,,3") == [2, 3]


def test_trailing_comma_skipped() -> None:
    assert resolve_gpu_list("2,3,") == [2, 3]
