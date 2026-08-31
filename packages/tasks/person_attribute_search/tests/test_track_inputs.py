# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-track inputs seam assembler."""

from __future__ import annotations

from person_attribute_search.track_inputs import assemble_track_inputs


def _tracks_payload() -> dict[str, object]:
    return {
        "crop_root": "tracks/crops",
        "n_tracks": 1,
        "tracks": [
            {
                "track_id": 2,
                "object_id": "person_2",
                "detection_score": 0.85,
                "duration_sec": 5.6,
                "first_frame": 8,
                "last_frame": 143,
                "n_crops": 16,
                "crop_dir": "tracks/crops/track_0002",
                "crops": ["a.jpg", "b.jpg"],
            }
        ],
    }


def test_assemble_merges_bookkeeping_and_model_outputs() -> None:
    per_track = {
        2: {
            "items": [{"id": "gender", "answer": "male"}],
            "caption": "A man in a vest.",
            "queries": ["person in yellow vest"],
            "n_crops_in_chunk": 13,
            "unknown_field": "dropped",
        }
    }
    seam = assemble_track_inputs(
        chunk_id="chunk_000",
        tracks_payload=_tracks_payload(),
        per_track=per_track,
        source_annotation="../vlm_annotations/chunk_000.json",
        anomaly_labels=["warehouse safety violation"],
        caption_queries=["A man walking in the warehouse."],
    )

    assert seam["chunk_id"] == "chunk_000"
    assert seam["crop_root"] == "tracks/crops"
    assert seam["source_annotation"] == "../vlm_annotations/chunk_000.json"
    assert seam["anomaly_labels"] == ["warehouse safety violation"]
    assert seam["caption_queries"] == ["A man walking in the warehouse."]

    [track] = seam["tracks"]
    assert track["track_id"] == 2
    assert track["detection_score"] == 0.85
    assert track["n_crops_in_chunk"] == 13
    assert track["items"] == [{"id": "gender", "answer": "male"}]
    assert track["queries"] == ["person in yellow vest"]
    # Unknown fields and the raw ``crops`` list are not part of the seam contract.
    assert "unknown_field" not in track
    assert "crops" not in track


def test_assemble_crop_root_override_and_empty_optionals() -> None:
    seam = assemble_track_inputs(
        chunk_id="chunk_001",
        tracks_payload={"tracks": []},
        per_track={},
        crop_root="custom/root",
    )
    assert seam["crop_root"] == "custom/root"
    assert seam["tracks"] == []
    assert "source_annotation" not in seam
    assert "anomaly_labels" not in seam
    assert "caption_queries" not in seam


def test_assemble_skips_malformed_track_ids() -> None:
    seam = assemble_track_inputs(
        chunk_id="chunk_002",
        tracks_payload={"tracks": [{"track_id": True}, {"track_id": "abc"}, {"track_id": " 3 "}]},
        per_track={3: {"caption": "valid"}},
    )
    assert seam["tracks"] == [{"track_id": 3, "caption": "valid"}]
