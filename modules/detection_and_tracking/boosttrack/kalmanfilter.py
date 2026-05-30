# SPDX-FileCopyrightText: Copyright (c) 2024 vukasin-stanojevic
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Vendored from BoostTrack (MIT).
Upstream: https://github.com/vukasin-stanojevic/BoostTrack

Vendored notice:
- SPDX-License-Identifier is `MIT AND Apache-2.0`: upstream MIT is preserved and NVIDIA
  modifications are licensed under Apache-2.0; see the SPDX-FileCopyrightText headers above.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    def __init__(self, init_state: np.ndarray):
        # Upstream uses an 8D constant-velocity model:
        # x = [cx, cy, h, r, vx, vy, vh, vr]^T
        init_state = init_state.astype(float).reshape((4, 1))
        self.x = np.zeros((8, 1), dtype=float)
        self.x[:4] = init_state

        # F, Q, H, R are adapted from upstream.
        self._ndim = 4

        self.dt = 1.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        self.P = np.eye(8) * 10.0
        self.Q = np.eye(8)
        self.R = np.eye(4)

        # Upstream keeps covariance for Mahalanobis distance; we mirror that.
        self.covariance = self.P.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.covariance = self.P.copy()

    def update(self, z: np.ndarray, score: float = 0.0):  # noqa: ARG002
        # Standard KF update.
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        identity_matrix = np.eye(self.P.shape[0])
        self.P = (identity_matrix - (K @ self.H)) @ self.P
        self.covariance = self.P.copy()
