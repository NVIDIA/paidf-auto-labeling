# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest
from al_utils.common import resolve_gpu_list


def test_resolve_gpu_list_all_uses_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_gpu_list('all') returns list derived from torch.cuda.device_count."""
    with patch("al_utils.common._all_gpu_ids", return_value=[0, 1, 2, 3]):
        assert resolve_gpu_list("all") == [0, 1, 2, 3]

    with patch("al_utils.common._all_gpu_ids", return_value=[0]):
        assert resolve_gpu_list("all") == [0]


def test_resolve_gpu_list_none_uses_torch() -> None:
    """resolve_gpu_list(None) returns list derived from torch.cuda.device_count."""
    with patch("al_utils.common._all_gpu_ids", return_value=[0, 1]):
        assert resolve_gpu_list(None) == [0, 1]


def test_resolve_gpu_list_fallback_on_torch_error() -> None:
    """resolve_gpu_list falls back to [0] when torch is unavailable."""
    with patch("al_utils.common._all_gpu_ids", return_value=[0]):
        assert resolve_gpu_list(None) == [0]
