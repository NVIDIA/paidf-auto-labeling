# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from detection_and_tracking.backends.boosttrack.boost_track import BoostTrack, KalmanBoxTracker


def test_boosttrack_update_none_advances_and_prunes_stale_trackers() -> None:
    tracker = BoostTrack(max_age=0, min_hits=1)
    tracker.trackers.append(KalmanBoxTracker(np.array([0.0, 0.0, 10.0, 10.0])))

    result = tracker.update(None)

    assert result.shape == (0, 6)
    assert tracker.frame_count == 1
    assert tracker.trackers == []
