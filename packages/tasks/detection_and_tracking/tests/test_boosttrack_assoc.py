# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from detection_and_tracking.backends.boosttrack.assoc import associate


def test_associate_no_trackers_returns_1d_unmatched_trackers() -> None:
    matched, unmatched_dets, unmatched_trks, cost_matrix = associate(
        np.zeros((2, 6), dtype=float),
        np.empty((0, 5), dtype=float),
        0.3,
    )

    assert matched.shape == (0, 2)
    assert unmatched_dets.tolist() == [0, 1]
    assert unmatched_trks.shape == (0,)
    assert cost_matrix.shape == (0, 0)
