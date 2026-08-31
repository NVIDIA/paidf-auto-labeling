# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import pytest
from reasoning.common import DaftConvertError, SceneContext
from reasoning.events.converter import build_object_id_catalogue, to_daft_events

VIDEO_CTX = SceneContext(media_id="clip_001", iso_date="2026-04-20")


def event(**overrides):
    base = {"event_id": "e1", "start_time": 0.0, "end_time": 2.0}
    base.update(overrides)
    return base


def test_non_finite_event_time_rejected():
    with pytest.raises(DaftConvertError, match="finite"):
        to_daft_events({"events": [event(start_time=float("nan"))]}, ctx=VIDEO_CTX)


def test_non_finite_duration_rejected():
    with pytest.raises(DaftConvertError, match="duration"):
        to_daft_events({"events": [event(end_time=10.0)]}, ctx=VIDEO_CTX, duration=float("inf"))


def test_negative_duration_rejected():
    with pytest.raises(DaftConvertError, match="duration"):
        to_daft_events({"events": [event(end_time=10.0)]}, ctx=VIDEO_CTX, duration=-1.0)


def test_build_object_id_catalogue_none_for_missing_or_malformed_payload():
    assert build_object_id_catalogue(None) is None
    assert build_object_id_catalogue({"instances": []}) is None


def test_build_object_id_catalogue_returns_ids_for_valid_payload():
    assert build_object_id_catalogue({"instances": {"car_1": {}, "": {}, 7: {}}}) == frozenset(
        {"car_1"}
    )
