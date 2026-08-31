# Vendored notice:
# - This file is vendored/adapted from upstream BoostTrack (MIT): https://github.com/vukasin-stanojevic/BoostTrack
#
# License text:
# MIT License
#
# Copyright (c) 2024 vukasin-stanojevic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-License-Identifier: MIT

"""
Vendored from BoostTrack (MIT).
Upstream: https://github.com/vukasin-stanojevic/BoostTrack
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    def __init__(self, init_state: np.ndarray) -> None:
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

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.covariance = self.P.copy()

    def update(self, z: np.ndarray, score: float = 0.0) -> None:  # noqa: ARG002
        # Standard KF update.
        z = np.asarray(z, dtype=float)
        if z.ndim == 1 and z.shape == (4,):
            z = z.reshape((4, 1))
        elif z.shape != (4, 1):
            raise ValueError(f"Kalman measurement must have shape (4,) or (4, 1), got {z.shape}.")
        y = z - (self.H @ self.x)
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + (kalman_gain @ y)
        identity_matrix = np.eye(self.P.shape[0])
        self.P = (identity_matrix - (kalman_gain @ self.H)) @ self.P
        self.covariance = self.P.copy()
