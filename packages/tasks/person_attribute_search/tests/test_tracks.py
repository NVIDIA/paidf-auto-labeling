# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for per-track aggregation."""

from __future__ import annotations

from person_attribute_search.tracks import (
    TrackRecord,
    build_people,
    build_person_entry,
    track_record_from_mapping,
)


def _track(track_id: int) -> TrackRecord:
    return TrackRecord(
        track_id=track_id,
        detection_score=0.85,
        duration_sec=5.6,
        first_frame=8,
        last_frame=143,
        n_crops=16,
        n_crops_in_chunk=13,
        first_frame_in_chunk=8,
        last_frame_in_chunk=119,
        crop_dir=f"../tracks/crops/track_{track_id:04d}",
        items=[
            {"id": "gender", "answer": "male"},
            {"id": "top outer color", "answer": "yellow"},
            {"id": "top outer type", "answer": "vest"},
            {"id": "headwear", "answer": "yes"},
            {"id": "headwear type", "answer": "helmet"},
            {"id": "natural language caption", "answer": "A man in a yellow vest."},
        ],
    )


def test_track_record_from_mapping_ignores_unknown_keys() -> None:
    record = track_record_from_mapping(
        {"track_id": 2, "detection_score": 0.5, "unexpected": "drop-me"}
    )
    assert record.track_id == 2
    assert record.detection_score == 0.5


def test_track_record_carries_object_id() -> None:
    record = track_record_from_mapping({"track_id": 4, "object_id": "person_alpha"})
    assert record.object_id == "person_alpha"


def test_build_person_entry_includes_object_id_when_present() -> None:
    track = TrackRecord(
        track_id=4,
        object_id="person_alpha",
        items=[{"id": "gender", "answer": "female"}],
    )
    entry, _ = build_person_entry(track)
    assert entry["object_id"] == "person_alpha"


def test_build_person_entry_omits_object_id_when_absent() -> None:
    entry, _ = build_person_entry(_track(2))
    assert "object_id" not in entry


def test_build_person_entry_shape() -> None:
    entry, query_set = build_person_entry(_track(2))
    assert entry["track_id"] == 2
    assert entry["detection_score"] == 0.85
    assert entry["crop_dir"] == "../tracks/crops/track_0002"
    assert entry["attributes"]["top outer color"] == "yellow"
    assert entry["attributes"]["headwear type"] == "helmet"
    assert entry["attributes"]["natural language caption"] == "A man in a yellow vest."
    assert entry["natural_language_caption"] == "A man in a yellow vest."
    assert query_set.easy


def test_build_person_entry_uses_track_caption_fallback() -> None:
    track = TrackRecord(
        track_id=3,
        caption="Fallback caption from track.",
        items=[{"id": "gender", "answer": "female"}],
    )
    entry, _ = build_person_entry(track)
    assert entry["natural_language_caption"] == "Fallback caption from track."


def test_build_people_preserves_order() -> None:
    people, query_sets = build_people([_track(5), _track(2)])
    assert [p["track_id"] for p in people] == [5, 2]
    assert len(query_sets) == 2
