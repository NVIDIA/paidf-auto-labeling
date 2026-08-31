# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tiered query-bundle export document assembly."""

from __future__ import annotations

from person_attribute_search.export.bundle import (
    build_bundle_documents,
    build_image_attribute_bundle_document,
    build_image_bundle_document,
    build_image_hitl_bundle_document,
)
from person_attribute_search.queries import QuerySet
from person_attribute_search.schema import PersonAttributes
from person_attribute_search.sources import AttributeImageSource


def _query_set(tag: str) -> QuerySet:
    return QuerySet(
        easy=[(f"{tag}-easy", "")],
        medium=[(f"{tag}-medium", "")],
        hard=[f"{tag}-hard"],
    )


def test_build_bundle_documents_keys_by_object_id() -> None:
    people = [
        {"track_id": 0, "object_id": "person_alpha"},
        {"track_id": 1, "object_id": "person_beta"},
    ]
    query_sets = [_query_set("a"), _query_set("b")]

    aggregated, per_person = build_bundle_documents(
        chunk_id="scene_0", people=people, query_sets=query_sets
    )

    assert aggregated["chunk_id"] == "scene_0"
    assert aggregated["n_people"] == 2
    assert set(aggregated["people"]) == {"person_alpha", "person_beta"}
    assert aggregated["people"]["person_alpha"]["queries"] == {
        "easy": ["a-easy"],
        "medium": ["a-medium"],
        "hard": ["a-hard"],
    }
    assert aggregated["people"]["person_alpha"]["track_id"] == 0

    filenames = [name for name, _ in per_person]
    assert filenames == ["person_alpha.json", "person_beta.json"]
    first_doc = per_person[0][1]
    assert first_doc["person_key"] == "person_alpha"
    assert first_doc["queries"]["hard"] == ["a-hard"]


def test_build_bundle_documents_falls_back_to_track_id() -> None:
    people = [{"track_id": 7}]
    aggregated, per_person = build_bundle_documents(
        chunk_id="scene_0", people=people, query_sets=[_query_set("x")]
    )
    assert set(aggregated["people"]) == {"track_0007"}
    assert per_person[0][0] == "track_0007.json"


def test_build_image_bundle_uses_image_ids_for_duplicate_person_keys() -> None:
    sources = [
        AttributeImageSource(
            attributes=PersonAttributes(top_outer_color=color),
            person_key="person_aug0",
            person_id="person",
            dataset="test",
            image_id=f"person_aug0/{view}.jpg",
            source_entry={
                "person_key": "person_aug0",
                "person_id": "person",
                "source_person_key": "person",
                "augmentation_id": "0",
                "images": [f"full/path/{view}.jpg"],
                "dataset": "test",
                "attributes": {"color": color},
                "selected_attributes": {"color": color},
                "attribute_verification": {"passed": True},
            },
        )
        for color, view in (("red", "front"), ("blue", "side"))
    ]

    document = build_image_bundle_document(
        chunk_id="0", sources=sources, query_sets=[_query_set("front"), _query_set("side")]
    )

    assert document["chunk_id"] == "0"
    assert document["n_people"] == 2
    assert set(document["people"]) == {"person_aug0/front.jpg", "person_aug0/side.jpg"}
    first = document["people"]["person_aug0/front.jpg"]
    assert first == {
        "image_filename": "front.jpg",
        "queries": {
            "easy": ["front-easy"],
            "medium": ["front-medium"],
            "hard": ["front-hard"],
        },
    }


def test_build_image_attribute_bundle_matches_query_keys_and_track_ids() -> None:
    sources = [
        AttributeImageSource(
            attributes=PersonAttributes(top_outer_color=color, viewpoint=view),
            person_key="person_aug0",
            person_id="person",
            dataset="test",
            image_id=f"person_aug0/{view}.jpg",
            source_entry={"person_key": "person_aug0", "track_id": track_id},
        )
        for color, view, track_id in (("red", "front", 4), ("blue", "side", 9))
    ]

    attributes = build_image_attribute_bundle_document(chunk_id="0", sources=sources)
    queries = build_image_bundle_document(
        chunk_id="0", sources=sources, query_sets=[_query_set("front"), _query_set("side")]
    )

    assert attributes["chunk_id"] == queries["chunk_id"] == "0"
    assert attributes["n_people"] == queries["n_people"] == 2
    assert attributes["people"].keys() == queries["people"].keys()
    assert attributes["people"]["person_aug0/front.jpg"] == {
        "track_id": 4,
        "attributes": {"top outer color": "red", "viewpoint": "front"},
    }
    assert attributes["people"]["person_aug0/side.jpg"]["track_id"] == 9


def test_build_image_attribute_bundle_preserves_source_accessories() -> None:
    sources = [
        AttributeImageSource(
            attributes=PersonAttributes(),
            person_key=f"person_{index}",
            person_id=f"person_{index}",
            dataset="test",
            image_id=f"image_{index}.jpg",
            source_entry={"attributes": {"accessories": accessories}},
        )
        for index, accessories in enumerate(([], ["hoodie"], ["backpack", "watch"]))
    ]

    document = build_image_attribute_bundle_document(chunk_id="0", sources=sources)

    assert document["people"]["person_0"]["attributes"]["accessories"] == []
    assert document["people"]["person_1"]["attributes"]["accessories"] == ["hoodie"]
    assert document["people"]["person_2"]["attributes"]["accessories"] == [
        "backpack",
        "watch",
    ]


def test_build_image_hitl_bundle_matches_keys_track_ids_and_captions() -> None:
    sources = [
        AttributeImageSource(
            attributes=PersonAttributes(
                top_outer_color=color,
                natural_caption=caption,
            ),
            person_key="person_aug0",
            person_id="person",
            dataset="test",
            image_id=f"person_aug0/{view}.jpg",
            source_entry={"track_id": track_id},
        )
        for color, view, caption, track_id in (
            ("red", "front", "A front view.", 4),
            ("blue", "side", None, 9),
        )
    ]

    hitl = build_image_hitl_bundle_document(
        chunk_id="0",
        sources=sources,
        query_sets=[_query_set("front"), _query_set("side")],
        image_url_base="https://images/",
    )
    attributes = build_image_attribute_bundle_document(chunk_id="0", sources=sources)

    assert hitl["chunk_id"] == attributes["chunk_id"] == "0"
    assert hitl["n_people"] == attributes["n_people"] == 2
    assert hitl["people"].keys() == attributes["people"].keys()
    front = hitl["people"]["person_aug0/front.jpg"]
    assert front["track_id"] == 4
    assert front["image_url"] == "https://images/person_aug0/front.jpg"
    assert [item["caption_type"] for item in front["preannotations"]] == [
        "Easy",
        "Medium",
        "Hard",
        "Natural",
    ]
    side = hitl["people"]["person_aug0/side.jpg"]
    assert side["track_id"] == 9
    assert all(item["caption_type"] != "Natural" for item in side["preannotations"])
