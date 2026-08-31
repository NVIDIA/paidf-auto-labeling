# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in tracker backends."""

from detection_and_tracking.backends.rfdetr import RFDetrTracker
from detection_and_tracking.backends.sam3 import SAM3Tracker
from detection_and_tracking.backends.stub import StubTracker
from detection_and_tracking.factory import register_tracker

register_tracker("stub", lambda logger, config: StubTracker(logger))
register_tracker(
    "rfdetr-bytetrack",
    lambda logger, config: RFDetrTracker(logger, config, tracker_backend="bytetrack"),
)
register_tracker(
    "rfdetr-boosttrack",
    lambda logger, config: RFDetrTracker(logger, config, tracker_backend="boosttrack"),
)
register_tracker("sam3", lambda logger, config: SAM3Tracker(logger, config))


__all__ = ["RFDetrTracker", "SAM3Tracker", "StubTracker"]
