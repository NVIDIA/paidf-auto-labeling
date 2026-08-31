# Vendored notice:
# - This file is vendored/adapted from upstream BoostTrack (MIT): https://github.com/vukasin-stanojevic/BoostTrack
#
# Key differences:
# - No ECC / no embedding by default (keeps dependencies light and avoids extra weights).
# - Assignment uses the vendored assoc.py which requires `lap`.
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
Minimal BoostTrack implementation (MIT), adapted for our pipeline.

Upstream: https://github.com/vukasin-stanojevic/BoostTrack


"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from .assoc import associate
from .kalmanfilter import KalmanFilter


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=float).ravel()
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    r = w / (h + 1e-6)
    return np.array([x, y, h, r], dtype=float).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    h = x[2]
    r = x[3]
    w = 0.0 if r <= 0 else r * h
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0], dtype=float
        ).reshape((1, 4))
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score], dtype=float
    ).reshape((1, 5))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: np.ndarray, emb: np.ndarray | None = None) -> None:
        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.hit_streak = 0
        self.age = 0
        self.emb: np.ndarray | None = emb

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7
        if self.age < n:
            return float(coef ** (n - self.age))
        return float(coef ** (self.time_since_update - 1))

    def update(self, bbox: np.ndarray, score: float = 0.0) -> None:
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb: np.ndarray, alpha: float = 0.9) -> None:
        if self.emb is None:
            self.emb = emb
            return
        self.emb = alpha * self.emb + (1 - alpha) * emb
        n = np.linalg.norm(self.emb) + 1e-12
        self.emb = self.emb / n

    def get_emb(self) -> np.ndarray | None:
        return self.emb


class BoostTrack:
    """
    Online tracker.

    Input dets: np.ndarray (N, 5) [x1,y1,x2,y2,score]
    Output tracks: np.ndarray (M, 6) [x1,y1,x2,y2,track_id,track_conf]
    """

    def __init__(
        self,
        *,
        max_age: int = 30,
        min_hits: int = 3,
        det_thresh: float = 0.5,
        iou_threshold: float = 0.3,
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25,
    ) -> None:
        self.frame_count = 0
        self.trackers: list[KalmanBoxTracker] = []

        self.max_age = int(max_age)
        self.iou_threshold = float(iou_threshold)
        self.det_thresh = float(det_thresh)
        self.min_hits = int(min_hits)

        self.lambda_iou = float(lambda_iou)
        self.lambda_mhd = float(lambda_mhd)
        self.lambda_shape = float(lambda_shape)

        # Optional embedder callable: (frame_bgr, dets_xyxy) -> np.ndarray (N, D) normalized
        self.embedder: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1,))[:n_dims]
        for i in range(len(self.trackers)):
            x[i] = self.trackers[i].kf.x[:n_dims].reshape((-1,))
            sigma = np.diag(self.trackers[i].kf.covariance[:n_dims, :n_dims])
            sigma_inv[i] = np.reciprocal(sigma + 1e-12)
        distance = (
            (z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2
            * sigma_inv.reshape((1, -1, n_dims))
        ).sum(axis=2)
        return cast(np.ndarray, distance)

    def update(self, dets: np.ndarray | None, *, frame_bgr: np.ndarray | None = None) -> np.ndarray:
        if dets is None:
            dets = np.empty((0, 6), dtype=float)
        dets = dets.astype(float, copy=False)

        self.frame_count += 1

        # Predict existing trackers
        trks = np.zeros((len(self.trackers), 5), dtype=float)
        confs = np.zeros((len(self.trackers), 1), dtype=float)
        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            confs[t] = self.trackers[t].get_confidence()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

        remain_inds = dets[:, 4] >= self.det_thresh if dets.size else np.array([], dtype=bool)
        dets = dets[remain_inds] if dets.size else dets
        scores = dets[:, 4] if dets.size else np.empty((0,), dtype=float)

        # ReID embeddings (optional)
        dets_embs: np.ndarray | None = None
        emb_cost: np.ndarray | None = None
        if self.embedder is not None and frame_bgr is not None and dets.size:
            dets_embs = self.embedder(frame_bgr, dets[:, :4])
            trk_emb_list: list[np.ndarray] = []
            for t in range(len(self.trackers)):
                e = self.trackers[t].get_emb()
                if e is None:
                    # placeholder; association will still work via IoU/MH/shape
                    trk_emb_list.append(np.zeros((dets_embs.shape[1],), dtype=np.float32))
                else:
                    trk_emb_list.append(e.astype(np.float32, copy=False))
            trk_embs = (
                np.stack(trk_emb_list, axis=0)
                if trk_emb_list
                else np.empty((0, dets_embs.shape[1]), dtype=np.float32)
            )
            if trk_embs.size:
                emb_cost = dets_embs.astype(np.float32, copy=False) @ trk_embs.T

        matched, unmatched_dets, unmatched_trks, _ = associate(
            dets,
            trks,
            self.iou_threshold,
            mahalanobis_distance=self.get_mh_dist_matrix(dets),
            track_confidence=confs,
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4], scores[m[0]])
            if dets_embs is not None:
                self.trackers[m[1]].update_emb(dets_embs[m[0]])

        for i in unmatched_dets:
            if dets[i, 4] >= self.det_thresh:
                emb_i = dets_embs[i] if dets_embs is not None else None
                self.trackers.append(KalmanBoxTracker(dets[i, :4], emb=emb_i))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.get_confidence()])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if ret:
            return np.concatenate(ret, axis=0)
        return np.empty((0, 6), dtype=float)
