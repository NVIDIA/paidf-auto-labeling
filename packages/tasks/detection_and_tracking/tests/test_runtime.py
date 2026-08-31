# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from detection_and_tracking.backends.runtime import first_configured_gpu, list_from_runtime


class _ToListValue:
    def __init__(self, value: Any) -> None:
        self.value = value

    def tolist(self) -> Any:
        return self.value


class _NonIterable:
    pass


def test_list_from_runtime_returns_tolist_lists_as_is() -> None:
    assert list_from_runtime(_ToListValue([1, 2])) == [1, 2]


def test_list_from_runtime_wraps_tolist_scalars() -> None:
    assert list_from_runtime(_ToListValue(7)) == [7]


def test_list_from_runtime_wraps_non_iterables() -> None:
    value = _NonIterable()

    assert list_from_runtime(value) == [value]


def test_list_from_runtime_uses_iterable_fallback() -> None:
    assert list_from_runtime((1, 2)) == [1, 2]


def test_first_configured_gpu_skips_malformed_tokens() -> None:
    assert first_configured_gpu("abc, 2") == 2


def test_first_configured_gpu_treats_negative_string_as_disabled() -> None:
    assert first_configured_gpu("-1, 2") is None
