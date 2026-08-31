# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adaptive attribute-source resolution."""

from __future__ import annotations

from pathlib import Path

from core import ensure_scene_skeleton, write_json
from person_attribute_search.sources import (
    attribute_entries,
    extract_caption,
    items_from_captioning,
    items_from_visual_qa,
    load_attribute_image_sources,
    merge_attribute_json,
    per_track_from_visual_qa_windows,
    resolve_attribute_items,
)


def test_attribute_entries_accepts_supported_envelopes() -> None:
    entry = {"attributes": {"top_outer_color": "red"}}
    assert attribute_entries(entry) == [entry]
    assert attribute_entries([entry]) == [entry]
    assert attribute_entries({"entries": [entry, "ignored"]}) == [entry]


def test_merge_attribute_json_votes_and_unions_accessories(tmp_path: Path) -> None:
    path = tmp_path / "person_attributes.json"
    write_json(
        path,
        {
            "entries": [
                {
                    "person_key": "person_7",
                    "person_id": "7",
                    "dataset": "test",
                    "image_id": "front.jpg",
                    "attributes": {
                        "top_outer_color": "red",
                        "bottom_type": "pants",
                        "shoe_type": "unknown",
                        "accessories": ["bag", "unknown"],
                    },
                },
                {
                    "image_id": "side.jpg",
                    "attributes": {
                        "top_outer_color": "blue",
                        "bottom_type": "pants",
                        "accessories": ["bag", "watch"],
                    },
                },
                {
                    "image_id": "rear.jpg",
                    "attributes": {
                        "top_outer_color": "red",
                        "bottom_type": "jeans",
                    },
                },
            ]
        },
    )

    merged = merge_attribute_json(path)

    assert merged.person_key == "person_7"
    assert merged.person_id == "7"
    assert merged.dataset == "test"
    assert merged.image_ids == ("front.jpg", "side.jpg", "rear.jpg")
    assert merged.attributes.top_outer_color == "red"
    assert merged.attributes.bottom_type == "pants"
    assert merged.attributes.shoe_type is None
    assert [accessory.item for accessory in merged.attributes.accessories] == ["bag", "watch"]


def test_merge_attribute_json_breaks_vote_ties_by_input_order(tmp_path: Path) -> None:
    path = tmp_path / "attributes.json"
    write_json(
        path,
        [
            {"attributes": {"viewpoint": "left"}},
            {"attributes": {"viewpoint": "right"}},
        ],
    )

    assert merge_attribute_json(path).attributes.viewpoint == "left"


def test_merge_attribute_json_preserves_missing_identity_metadata(tmp_path: Path) -> None:
    path = tmp_path / "attributes.json"
    write_json(
        path,
        [{"image_id": "person.jpg", "attributes": {"viewpoint": "front"}}],
    )

    merged = merge_attribute_json(path)

    assert merged.person_key is None
    assert merged.person_id is None
    assert merged.dataset is None


def test_load_attribute_image_sources_keeps_each_entry_separate(tmp_path: Path) -> None:
    path = tmp_path / "attributes.json"
    write_json(
        path,
        {
            "entries": [
                {
                    "person_key": "person_aug0",
                    "augmentation_id": "0",
                    "image_id": "person_aug0/front.jpg",
                    "images": ["full/path/front.jpg"],
                    "attributes": {"viewpoint": "front"},
                },
                {
                    "person_key": "person_aug0",
                    "augmentation_id": "0",
                    "image_id": "person_aug0/side.jpg",
                    "images": ["full/path/side.jpg"],
                    "attributes": {"viewpoint": "side"},
                },
            ]
        },
    )

    sources = load_attribute_image_sources(path)

    assert [source.image_id for source in sources] == [
        "person_aug0/front.jpg",
        "person_aug0/side.jpg",
    ]
    assert [source.attributes.viewpoint for source in sources] == ["front", "side"]
    assert sources[0].source_entry["images"] == ["full/path/front.jpg"]


def test_items_from_visual_qa_filters_non_dicts() -> None:
    payload = {"items": [{"id": "gender", "answer": "male"}, "junk", 7]}
    assert items_from_visual_qa(payload) == [{"id": "gender", "answer": "male"}]


def test_items_from_visual_qa_returns_none_for_unusable_payloads() -> None:
    """Missing/empty/malformed payloads are no-source (None), never empty-success."""
    assert items_from_visual_qa({"other": 1}) is None
    assert items_from_visual_qa("nope") is None
    assert items_from_visual_qa({"items": []}) is None
    assert items_from_visual_qa({"items": ["junk", 7]}) is None
    assert items_from_visual_qa({"items": "nope"}) is None


def test_resolve_empty_visual_qa_falls_through_to_captioning(tmp_path: Path) -> None:
    """A present-but-empty visual_qa sidecar must not short-circuit as a source."""
    paths = ensure_scene_skeleton(tmp_path / "scene")
    write_json(paths.sidecars_dir / "visual_qa" / "items.json", {"items": []})
    write_json(
        paths.sidecars_dir / "captioning" / "attributes.json",
        {"parsed": {"gender": "female"}},
    )

    items, source = resolve_attribute_items(
        paths,
        visual_qa_sidecars=("visual_qa/items.json",),
        caption_attribute_sidecars=("captioning/attributes.json",),
    )
    assert source == "captioning"
    assert items == [{"id": "gender", "answer": "female"}]


def test_resolve_empty_visual_qa_no_captioning_is_no_source(tmp_path: Path) -> None:
    """Empty visual_qa with no captioning fallback resolves to no source (failure)."""
    paths = ensure_scene_skeleton(tmp_path / "scene")
    write_json(paths.sidecars_dir / "visual_qa" / "items.json", {"items": []})

    items, source = resolve_attribute_items(
        paths,
        visual_qa_sidecars=("visual_qa/items.json",),
        caption_attribute_sidecars=(),
    )
    assert items is None
    assert source is None


def test_items_from_captioning_parsed_object() -> None:
    payload = {"parsed": {"gender": "male", "top outer color": "yellow"}}
    items = items_from_captioning(payload)
    assert items == [
        {"id": "gender", "answer": "male"},
        {"id": "top outer color", "answer": "yellow"},
    ]


def test_items_from_captioning_falls_back_to_model_output() -> None:
    payload = {"model_output": {"gender": "female"}}
    assert items_from_captioning(payload) == [{"id": "gender", "answer": "female"}]


def test_items_from_captioning_without_parsed_returns_none() -> None:
    assert items_from_captioning({"caption": "text only"}) is None


def test_extract_caption_reads_nested_caption_containers() -> None:
    assert extract_caption({"parsed": {"caption": " Nested caption. "}}) == "Nested caption."
    assert extract_caption({"model_output": {"summary": " Model summary. "}}) == "Model summary."
    assert extract_caption({"parsed": " Direct caption. "}) == "Direct caption."


def test_resolve_prefers_visual_qa(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    write_json(
        paths.sidecars_dir / "visual_qa" / "items.json",
        {"items": [{"id": "gender", "answer": "male"}]},
    )
    write_json(
        paths.sidecars_dir / "captioning" / "attributes.json",
        {"parsed": {"gender": "female"}},
    )

    items, source = resolve_attribute_items(
        paths,
        visual_qa_sidecars=("visual_qa/items.json",),
        caption_attribute_sidecars=("captioning/attributes.json",),
    )
    assert source == "visual_qa"
    assert items == [{"id": "gender", "answer": "male"}]


def test_resolve_uses_captioning_when_visual_qa_absent(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    write_json(
        paths.sidecars_dir / "captioning" / "attributes.json",
        {"parsed": {"gender": "female"}},
    )

    items, source = resolve_attribute_items(
        paths,
        visual_qa_sidecars=("visual_qa/items.json",),
        caption_attribute_sidecars=("captioning/attributes.json",),
    )
    assert source == "captioning"
    assert items == [{"id": "gender", "answer": "female"}]


def test_resolve_returns_none_when_no_source(tmp_path: Path) -> None:
    paths = ensure_scene_skeleton(tmp_path / "scene")
    items, source = resolve_attribute_items(
        paths,
        visual_qa_sidecars=("visual_qa/items.json",),
        caption_attribute_sidecars=(),
    )
    assert items is None
    assert source is None


def test_per_track_skips_malformed_track_ids() -> None:
    payload = {
        "windows": [
            {"track_id": 1, "items": [{"id": "gender", "answer": "male"}]},
            {"track_id": "2", "items": []},
            {"track_id": True, "items": []},
            {"track_id": 1.5, "items": []},
            {"track_id": "x", "items": []},
            {"track_id": None, "items": []},
        ]
    }
    out = per_track_from_visual_qa_windows(payload)
    assert set(out) == {1, 2}
