# SPDX-FileCopyrightText: Copyright (c) 2024 vukasin-stanojevic
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

# Vendored notice:
# - This file is vendored/adapted from upstream BoostTrack (MIT): https://github.com/vukasin-stanojevic/BoostTrack
# - SPDX-License-Identifier is `MIT AND Apache-2.0`: upstream MIT is preserved and NVIDIA
#   modifications are licensed under Apache-2.0; see the SPDX-FileCopyrightText headers above.

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Optional

import lap  # type: ignore
import numpy as np


def shape_similarity_v1(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    # NOTE: This matches the upstream default implementation (it intentionally uses width in both denominators).
    return np.exp(-(np.abs(dw - tw) / np.maximum(dw, tw) + np.abs(dh - th) / np.maximum(dw, tw)))


def mahalanobis_distance_similarity(mahalanobis_distance: np.ndarray, softmax_temp: float = 1.0) -> np.ndarray:
    limit = 13.2767  # 99% conf interval (chi2inv)
    mahalanobis_distance = deepcopy(mahalanobis_distance)
    mask = mahalanobis_distance > limit
    mahalanobis_distance[mask] = limit
    mahalanobis_distance = limit - mahalanobis_distance

    md = np.exp(mahalanobis_distance / softmax_temp)
    md = md / md.sum(0).reshape((1, -1))
    md = np.where(mask, 0, md)
    return md


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes IoU between two bbox sets in [x1,y1,x2,y2] form."""
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
        + 1e-12
    )
    return o


def _match(cost_matrix: np.ndarray, threshold: float) -> np.ndarray:
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    a = (cost_matrix > threshold).astype(np.int32)
    if a.sum(1).max(initial=0) == 1 and a.sum(0).max(initial=0) == 1:
        return np.stack(np.nonzero(a), axis=1).astype(int)

    # maximize cost -> minimize negative cost
    _, x, y = lap.lapjv(-cost_matrix, extend_cost=True)  # type: ignore[attr-defined]
    return np.array([[y[i], i] for i in x if i >= 0], dtype=int)


def linear_assignment(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_matrix: np.ndarray | None,
    cost_matrix: np.ndarray | None,
    threshold: float,
    emb_cost: Optional[np.ndarray] = None,
):
    if iou_matrix is None and cost_matrix is None:
        raise ValueError("Both iou_matrix and cost_matrix are None!")
    if iou_matrix is None:
        iou_matrix = deepcopy(cost_matrix)
    if cost_matrix is None:
        cost_matrix = deepcopy(iou_matrix)

    matched_indices = _match(cost_matrix, threshold)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        ok = (iou_matrix[m[0], m[1]] >= threshold) or (
            False if emb_cost is None else (iou_matrix[m[0], m[1]] >= threshold / 2 and emb_cost[m[0], m[1]] >= 0.75)
        )
        if ok:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(int(m[0]))
            unmatched_trackers.append(int(m[1]))

    matches_arr = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
    return matches_arr, np.array(unmatched_detections), np.array(unmatched_trackers), cost_matrix


def associate(
    detections,
    trackers,
    iou_threshold,
    *,
    mahalanobis_distance: Optional[np.ndarray] = None,
    track_confidence: Optional[np.ndarray] = None,
    detection_confidence: Optional[np.ndarray] = None,
    emb_cost: Optional[np.ndarray] = None,
    lambda_iou: float = 0.5,
    lambda_mhd: float = 0.25,
    lambda_shape: float = 0.25,
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
            np.empty((0, 0)),
        )

    iou_matrix = iou_batch(detections, trackers)
    cost_matrix = deepcopy(iou_matrix)

    if detection_confidence is not None and track_confidence is not None:
        conf = np.multiply(detection_confidence.reshape((-1, 1)), track_confidence.reshape((1, -1)))
        conf[iou_matrix < iou_threshold] = 0
        cost_matrix += lambda_iou * conf * iou_batch(detections, trackers)
    else:
        warnings.warn("Detections or tracklet confidence is None; detection-tracklet confidence cannot be computed!")
        conf = None

    if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
        md = mahalanobis_distance_similarity(mahalanobis_distance)
        cost_matrix += lambda_mhd * md
        if conf is not None:
            cost_matrix += lambda_shape * conf * shape_similarity_v1(detections, trackers)

    if emb_cost is not None:
        lambda_emb = (1 + lambda_iou + lambda_shape + lambda_mhd) * 1.5
        cost_matrix += lambda_emb * emb_cost

    return linear_assignment(detections, trackers, iou_matrix, cost_matrix, iou_threshold, emb_cost)
