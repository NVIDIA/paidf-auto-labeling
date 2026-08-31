# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection and tracking task package.

Importing this package populates the tracker registry with the shipped
backends (see :mod:`detection_and_tracking.backends`). Out-of-tree
backends can register themselves at any time by calling
:func:`register_tracker`.
"""

from detection_and_tracking import (  # noqa: F401  (import side-effect: register builtins)
    backends as _builtin_backends,
)
from detection_and_tracking.artifacts import TRACKING_ARTIFACTS_KEY, TrackingArtifactsState
from detection_and_tracking.backends import StubTracker
from detection_and_tracking.config import DetectionAndTrackingConfig, pas_tracking_config
from detection_and_tracking.crops import (
    CropRecord,
    extract_track_crops,
    plan_track_crops,
)
from detection_and_tracking.factory import (
    PLANNED_TRACKER_KINDS,
    TrackerKind,
    create_tracker,
    list_trackers,
    register_tracker,
)
from detection_and_tracking.task import (
    DetectionAndTrackingTask,
)
from detection_and_tracking.tracker import (
    Tracker,
    TrackingResult,
)

__all__ = [
    "CropRecord",
    "DetectionAndTrackingConfig",
    "DetectionAndTrackingTask",
    "PLANNED_TRACKER_KINDS",
    "StubTracker",
    "TRACKING_ARTIFACTS_KEY",
    "Tracker",
    "TrackerKind",
    "TrackingArtifactsState",
    "TrackingResult",
    "create_tracker",
    "extract_track_crops",
    "list_trackers",
    "pas_tracking_config",
    "plan_track_crops",
    "register_tracker",
]
