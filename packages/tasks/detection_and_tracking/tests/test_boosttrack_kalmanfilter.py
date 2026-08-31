# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from detection_and_tracking.backends.boosttrack.kalmanfilter import KalmanFilter


def test_kalman_update_accepts_flat_measurement_without_broadcasting() -> None:
    kf = KalmanFilter(np.array([1.0, 2.0, 3.0, 4.0]))

    kf.update(np.array([1.5, 2.5, 3.5, 4.5]))

    assert kf.x.shape == (8, 1)


def test_kalman_update_accepts_column_measurement() -> None:
    kf = KalmanFilter(np.array([1.0, 2.0, 3.0, 4.0]))

    kf.update(np.array([[1.5], [2.5], [3.5], [4.5]]))

    assert kf.x.shape == (8, 1)


def test_kalman_update_rejects_unexpected_measurement_shape() -> None:
    kf = KalmanFilter(np.array([1.0, 2.0, 3.0, 4.0]))

    with pytest.raises(ValueError, match="Kalman measurement must have shape"):
        kf.update(np.array([[1.0, 2.0, 3.0, 4.0]]))
