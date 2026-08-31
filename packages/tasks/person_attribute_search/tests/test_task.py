# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the PAS assembly task over a temp scene directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core import DataEntry, ensure_scene_skeleton, read_json, read_pipeline_state, write_json
from core.model_clients import ChatRequest
from person_attribute_search.artifacts import PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY
from person_attribute_search.config import PersonAttributeSearchConfig
from person_attribute_search.task import PersonAttributeSearchTask


def _seed_visual_qa_items(scene_dir: Path) -> None:
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "visual_qa" / "items.json",
        {
            "items": [
                {"id": "gender", "answer": "female"},
                {"id": "top outer color", "answer": "red (crimson)"},
                {"id": "top outer type", "answer": "t-shirt"},
                {"id": "bottom color", "answer": "blue (navy)"},
                {"id": "bottom type", "answer": "jeans"},
                {"id": "natural language caption", "answer": "A woman in a red t-shirt."},
                {"id": "hard", "answer": "woman crimson t-shirt navy jeans"},
            ]
        },
    )


def test_task_writes_pas_artifacts(tmp_path: Path) -> None:
    scene_dir = tmp_path / "00001"
    _seed_visual_qa_items(scene_dir)
    entry = DataEntry(media_path="00001.jpg", data_path=str(scene_dir))

    task = PersonAttributeSearchTask(PersonAttributeSearchConfig(dataset="rstp"))
    task.run(entry)

    attributes = read_json(scene_dir / "sidecars" / "person_attribute_search" / "attributes.json")
    assert attributes["person_key"] == "00001_rstp"
    assert attributes["attributes"]["top outer color"] == "red"
    assert attributes["attributes"]["top outer color (fine)"] == "crimson"

    queries = read_json(scene_dir / "sidecars" / "person_attribute_search" / "queries.json")
    assert queries["queries"]["hard"] == ["woman crimson t-shirt navy jeans"]
    assert queries["queries"]["easy"]

    hitl = read_json(scene_dir / "sidecars" / "person_attribute_search" / "hitl.json")
    assert hitl["preannotations"][-1]["caption_type"] == "Natural"


def test_task_llm_query_generation_single_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With llm_query_generation, medium+hard queries come from the LLM op."""
    scene_dir = tmp_path / "00020"
    _seed_visual_qa_items(scene_dir)
    entry = DataEntry(media_path="00020.jpg", data_path=str(scene_dir))

    captured: dict[str, object] = {}

    class _FakeClient:
        def generate(self, request: ChatRequest) -> str:
            captured["prompt"] = request.prompt
            return (
                '{"medium": [["red t-shirt and navy jeans", "top outer color, bottom color"]],'
                ' "hard": ["woman in crimson tee and navy denim", "crimson top navy jeans"]}'
            )

    def _fake_factory(**_kwargs: object) -> _FakeClient:
        return _FakeClient()

    monkeypatch.setattr("person_attribute_search.task.create_endpoint_client", _fake_factory)

    config = PersonAttributeSearchConfig(
        dataset="rstp",
        llm_query_generation=True,
        llm_endpoint_url="http://localhost:8082/v1",
        llm_model="default-llm",
    )
    PersonAttributeSearchTask(config).run(entry)

    queries = read_json(scene_dir / "sidecars" / "person_attribute_search" / "queries.json")
    assert queries["queries"]["medium"] == [
        ["red t-shirt and navy jeans", "top outer color, bottom color"]
    ]
    assert queries["queries"]["hard"] == [
        "woman in crimson tee and navy denim",
        "crimson top navy jeans",
    ]
    # Easy queries remain template-generated.
    assert queries["queries"]["easy"]
    # The attributes block + caption were rendered into the LLM prompt.
    assert "A woman in a red t-shirt." in str(captured["prompt"])


def test_task_records_state(tmp_path: Path) -> None:
    scene_dir = tmp_path / "00002"
    _seed_visual_qa_items(scene_dir)
    entry = DataEntry(media_path="00002.jpg", data_path=str(scene_dir))

    PersonAttributeSearchTask().run(entry)

    state = read_pipeline_state(scene_dir)
    artifacts = state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY]
    assert artifacts["success"] is True


def test_task_sources_attributes_from_captioning(tmp_path: Path) -> None:
    scene_dir = tmp_path / "00010"
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "captioning" / "attributes.json",
        {
            "parsed": {
                "gender": "male",
                "top outer color": "yellow (hi-vis)",
                "top outer type": "vest",
                "natural language caption": "A man in a yellow vest.",
            }
        },
    )
    entry = DataEntry(media_path="00010.jpg", data_path=str(scene_dir))

    config = PersonAttributeSearchConfig(
        dataset="rstp",
        caption_attribute_sidecars=("captioning/attributes.json",),
    )
    PersonAttributeSearchTask(config).run(entry)

    attributes = read_json(scene_dir / "sidecars" / "person_attribute_search" / "attributes.json")
    assert attributes["attributes"]["top outer color"] == "yellow"
    assert attributes["attributes"]["top outer color (fine)"] == "hi-vis"


def test_task_skips_when_no_items(tmp_path: Path) -> None:
    scene_dir = tmp_path / "00003"
    ensure_scene_skeleton(scene_dir)
    entry = DataEntry(media_path="00003.jpg", data_path=str(scene_dir))

    PersonAttributeSearchTask().run(entry)

    state = read_pipeline_state(scene_dir)
    artifacts = state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY]
    assert artifacts["success"] is False


def _seed_track_inputs(scene_dir: Path) -> None:
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "person_attribute_search" / "track_inputs.json",
        {
            "chunk_id": "chunk_000",
            "crop_root": "../tracks/crops",
            "source_annotation": "../vlm_annotations/chunk_000.json",
            "anomaly_labels": ["warehouse_safety_violation"],
            "caption_queries": ["a man in a yellow helmet walking in a warehouse"],
            "tracks": [
                {
                    "track_id": 2,
                    "detection_score": 0.855,
                    "crop_dir": "../tracks/crops/track_0002",
                    "items": [
                        {"id": "gender", "answer": "male"},
                        {"id": "top outer color", "answer": "yellow"},
                        {"id": "top outer type", "answer": "vest"},
                        {"id": "natural language caption", "answer": "A man in a yellow vest."},
                    ],
                },
                {
                    "track_id": 3,
                    "detection_score": 0.74,
                    "crop_dir": "../tracks/crops/track_0003",
                    "items": [
                        {"id": "gender", "answer": "female"},
                        {"id": "top outer color", "answer": "red"},
                        {"id": "top outer type", "answer": "jacket"},
                    ],
                },
            ],
        },
    )


def test_task_per_track_emits_chunk_documents(tmp_path: Path) -> None:
    scene_dir = tmp_path / "chunk_000"
    _seed_track_inputs(scene_dir)
    entry = DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))

    PersonAttributeSearchTask().run(entry)

    pas = read_json(scene_dir / "sidecars" / "person_attribute_search" / "pas.json")
    assert pas["chunk_id"] == "chunk_000"
    assert pas["crop_root"] == "../tracks/crops"
    assert pas["pas"]["n_people"] == 2
    assert [p["track_id"] for p in pas["pas"]["people"]] == [2, 3]
    assert pas["pas"]["people"][0]["attributes"]["top outer color"] == "yellow"

    queries = read_json(scene_dir / "sidecars" / "person_attribute_search" / "chunk_queries.json")
    flat = [q["query"] for q in queries["queries"]]
    assert "warehouse safety violation" in flat
    assert "a man in a yellow helmet walking in a warehouse" in flat
    assert len(flat) == len(set(flat))

    state = read_pipeline_state(scene_dir)
    artifacts = state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY]
    assert artifacts["success"] is True
    assert artifacts["n_people"] == 2


def test_task_assembles_per_track_from_visual_qa_windows(tmp_path: Path) -> None:
    """No explicit seam: PAS assembles it from detection tracks + visual_qa windows."""
    scene_dir = tmp_path / "chunk_000"
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "detection_and_tracking" / "tracks.json",
        {
            "crop_root": "detection_and_tracking/crops",
            "tracks": [
                {
                    "track_id": 2,
                    "detection_score": 0.86,
                    "crop_dir": "detection_and_tracking/crops/track_0002",
                },
                {
                    "track_id": 5,
                    "detection_score": 0.71,
                    "crop_dir": "detection_and_tracking/crops/track_0005",
                },
            ],
        },
    )
    write_json(
        paths.sidecars_dir / "visual_qa" / "windows.normalized.json",
        {
            "schema_version": "1",
            "media_id": "chunk_000",
            "windows": [
                {
                    "window_index": 0,
                    "track_id": 2,
                    "n_crops_in_chunk": 8,
                    "description": "A man in a yellow vest.",
                    "items": [
                        {"id": "gender", "answer": "male"},
                        {"id": "top outer color", "answer": "yellow"},
                        {"id": "top outer type", "answer": "vest"},
                        {"id": "hard", "answer": "man in a yellow safety vest"},
                    ],
                },
                {
                    "window_index": 1,
                    "track_id": 5,
                    "n_crops_in_chunk": 4,
                    "description": "A woman in a red jacket.",
                    "items": [
                        {"id": "gender", "answer": "female"},
                        {"id": "top outer color", "answer": "red"},
                        {"id": "top outer type", "answer": "jacket"},
                    ],
                },
            ],
        },
    )
    entry = DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))

    PersonAttributeSearchTask().run(entry)

    pas = read_json(scene_dir / "sidecars" / "person_attribute_search" / "pas.json")
    assert pas["chunk_id"] == "chunk_000"
    assert pas["crop_root"] == "detection_and_tracking/crops"
    assert [p["track_id"] for p in pas["pas"]["people"]] == [2, 5]
    # Detection bookkeeping merged with visual_qa items.
    assert pas["pas"]["people"][0]["detection_score"] == 0.86
    assert pas["pas"]["people"][0]["attributes"]["top outer color"] == "yellow"

    queries = read_json(scene_dir / "sidecars" / "person_attribute_search" / "chunk_queries.json")
    flat = [q["query"] for q in queries["queries"]]
    # Hard query sourced from the visual_qa item flows through to chunk queries.
    assert "man in a yellow safety vest" in flat


def test_task_per_track_emits_query_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With bucket_query_generation, chunk_queries carries PAS/Anomaly/Caption buckets."""
    scene_dir = tmp_path / "chunk_000"
    _seed_track_inputs(scene_dir)
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "captioning" / "video_captions.json",
        {
            "summary": "A busy warehouse aisle with workers.",
            "windows": [
                {"start_time": 0.0, "end_time": 1.0, "caption": "Workers walking."},
                {"start_time": 1.0, "end_time": 2.0, "caption": "A person slips."},
            ],
        },
    )
    write_json(
        paths.sidecars_dir / "visual_qa_anomaly" / "items.json",
        {"items": [{"id": "anom_person_falling_or_collapsing", "answer": "A. Yes"}]},
    )

    captured: dict[str, object] = {}

    class _FakeClient:
        def generate(self, request: ChatRequest) -> str:
            captured["prompt"] = request.prompt
            return (
                '{"PAS": [["person wearing yellow vest", "top outer"]],'
                ' "Anomaly": [["person slipping on warehouse floor", "fall category"]],'
                ' "Caption": [["busy warehouse aisle with workers", "scene"]]}'
            )

    monkeypatch.setattr(
        "person_attribute_search.task.create_endpoint_client",
        lambda **_kwargs: _FakeClient(),
    )

    config = PersonAttributeSearchConfig(
        bucket_query_generation=True,
        llm_endpoint_url="http://localhost:8082/v1",
        llm_model="default-llm",
    )
    PersonAttributeSearchTask(config).run(
        DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))
    )

    queries = read_json(scene_dir / "sidecars" / "person_attribute_search" / "chunk_queries.json")
    buckets = queries["query_buckets"]
    # PAS bucket mirrors the flat per-person queries (unchanged behavior).
    flat = [q["query"] for q in queries["queries"]]
    assert buckets["PAS"] == [[q, ""] for q in flat]
    # Anomaly + Caption come from the chunk-level LLM call.
    assert buckets["Anomaly"] == [["person slipping on warehouse floor", "fall category"]]
    assert buckets["Caption"] == [["busy warehouse aisle with workers", "scene"]]
    # The voted anomaly category and dense caption were rendered into the prompt.
    assert "person_falling_or_collapsing" in str(captured["prompt"])
    assert "A person slips." in str(captured["prompt"])


def test_task_reuses_llm_client_for_tiered_and_bucket_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two optional PAS LLM paths share one endpoint client instance."""
    scene_dir = tmp_path / "chunk_000"
    _seed_track_inputs(scene_dir)
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "captioning" / "video_captions.json",
        {"summary": "A warehouse scene.", "windows": [{"caption": "Workers walking."}]},
    )
    write_json(
        paths.sidecars_dir / "visual_qa_anomaly" / "items.json",
        {"items": [{"id": "anom_person_falling_or_collapsing", "answer": "A. Yes"}]},
    )

    factory_calls = 0

    class _FakeClient:
        def generate(self, request: ChatRequest) -> str:
            if "Scene/anomaly and dense caption context" in request.prompt:
                return (
                    '{"PAS": [], "Anomaly": [["person slipping", "fall category"]],'
                    ' "Caption": [["warehouse scene", "scene"]]}'
                )
            return '{"medium": [["yellow vest", "top outer color"]], "hard": ["worker vest"]}'

    def _fake_factory(**_kwargs: object) -> _FakeClient:
        nonlocal factory_calls
        factory_calls += 1
        return _FakeClient()

    monkeypatch.setattr("person_attribute_search.task.create_endpoint_client", _fake_factory)

    config = PersonAttributeSearchConfig(
        llm_query_generation=True,
        bucket_query_generation=True,
        llm_endpoint_url="http://localhost:8082/v1",
        llm_model="default-llm",
    )
    PersonAttributeSearchTask(config).run(
        DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))
    )

    assert factory_calls == 1


def test_task_records_optional_bucket_generation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bucket LLM failures degrade to PAS-only buckets and are recorded in state."""
    scene_dir = tmp_path / "chunk_000"
    _seed_track_inputs(scene_dir)
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "captioning" / "video_captions.json",
        {"summary": "A warehouse scene.", "windows": [{"caption": "Workers walking."}]},
    )
    write_json(
        paths.sidecars_dir / "visual_qa_anomaly" / "items.json",
        {"items": [{"id": "anom_person_falling_or_collapsing", "answer": "A. Yes"}]},
    )

    class _FailingClient:
        def generate(self, _request: ChatRequest) -> str:
            raise RuntimeError("test bucket failure")

    monkeypatch.setattr(
        "person_attribute_search.task.create_endpoint_client",
        lambda **_kwargs: _FailingClient(),
    )

    config = PersonAttributeSearchConfig(
        bucket_query_generation=True,
        llm_endpoint_url="http://localhost:8082/v1",
        llm_model="default-llm",
    )
    PersonAttributeSearchTask(config).run(
        DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))
    )

    queries = read_json(scene_dir / "sidecars" / "person_attribute_search" / "chunk_queries.json")
    buckets = queries["query_buckets"]
    assert buckets["PAS"]
    assert buckets["Anomaly"] == []
    assert buckets["Caption"] == []

    state = read_pipeline_state(scene_dir)
    artifacts = state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY]
    assert artifacts["success"] is True
    assert artifacts["optional_failures"] == ["bucket_query_generation_failed: test bucket failure"]


def test_task_per_track_skips_buckets_when_disabled(tmp_path: Path) -> None:
    """Without bucket_query_generation, no query_buckets are written (default)."""
    scene_dir = tmp_path / "chunk_000"
    _seed_track_inputs(scene_dir)
    PersonAttributeSearchTask().run(DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir)))
    queries = read_json(scene_dir / "sidecars" / "person_attribute_search" / "chunk_queries.json")
    assert "query_buckets" not in queries


def test_task_per_track_emits_anomaly_record_when_enabled(tmp_path: Path) -> None:
    """With emit_contextual, the terminal PAS flow writes the merged sidecar record."""
    scene_dir = tmp_path / "chunk_000.mp4"
    _seed_track_inputs(scene_dir)
    config = PersonAttributeSearchConfig(emit_contextual=True)
    PersonAttributeSearchTask(config).run(
        DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))
    )

    pas_sidecar = scene_dir / "sidecars" / "person_attribute_search"
    # pas.json remains the PAS stage's own sidecar; it is not copied to contextual/.
    assert read_json(pas_sidecar / "pas.json")["pas"]["n_people"] == 2
    assert not (scene_dir / "contextual" / "pas.json").exists()
    anomaly = read_json(pas_sidecar / "pas_anomaly.json")
    # Merged record carries the per-chunk block sourced from the PAS sidecar.
    assert anomaly["chunk"]["pas"]["n_people"] == 2
    # official_anomaly is not populated by the in-task fold-in (CLI-only).
    assert anomaly["official_anomaly"] is None
    # No anomaly visual_qa sidecar in this fixture: fail closed (n_models_ok == 0)
    # rather than minting a spurious 'normal' vote.
    assert anomaly["chunk"]["anomaly_gt"]["n_models_ok"] == 0
    # The three legacy query buckets are carried through from chunk_queries.json.
    assert set(anomaly["chunk"]["queries"]) == {"PAS", "Anomaly", "Caption"}


def test_task_per_track_skips_anomaly_record_by_default(tmp_path: Path) -> None:
    """Without emit_contextual, no merged pas_anomaly.json deliverable is written."""
    scene_dir = tmp_path / "chunk_000.mp4"
    _seed_track_inputs(scene_dir)
    PersonAttributeSearchTask().run(DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir)))
    assert not (scene_dir / "sidecars" / "person_attribute_search" / "pas_anomaly.json").exists()


def test_task_per_track_skips_when_no_tracks(tmp_path: Path) -> None:
    scene_dir = tmp_path / "chunk_001"
    paths = ensure_scene_skeleton(scene_dir)
    write_json(
        paths.sidecars_dir / "person_attribute_search" / "track_inputs.json",
        {"chunk_id": "chunk_001", "tracks": []},
    )
    entry = DataEntry(media_path="chunk_001.mp4", data_path=str(scene_dir))

    PersonAttributeSearchTask().run(entry)

    state = read_pipeline_state(scene_dir)
    artifacts = state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY]
    assert artifacts["success"] is False


def test_explicit_attribute_json_overrides_visual_qa_and_writes_legacy_output(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "person"
    _seed_visual_qa_items(scene_dir)
    attribute_json = tmp_path / "attributes.json"
    write_json(
        attribute_json,
        {
            "entries": [
                {
                    "person_key": "person_42",
                    "person_id": "42",
                    "dataset": "custom",
                    "image_id": "view_a.jpg",
                    "attributes": {
                        "top_outer_color": "green",
                        "top_outer_type": "jacket",
                        "bottom_color": "black",
                        "bottom_type": "pants",
                    },
                }
            ]
        },
    )
    entry = DataEntry(media_path="sentinel.jpg", data_path=str(scene_dir))

    PersonAttributeSearchTask(
        PersonAttributeSearchConfig(attribute_json=str(attribute_json), write_hitl=False)
    ).run(entry)

    output_dir = scene_dir / "sidecars" / "person_attribute_search"
    attributes = read_json(output_dir / "attributes.json")
    queries = read_json(output_dir / "queries.json")
    assert attributes["person_key"] == "person_42"
    assert attributes["dataset"] == "custom"
    assert attributes["attributes"]["top outer color"] == "green"
    assert "natural_caption" not in attributes
    assert queries["queries"]["easy"]
    assert isinstance(queries["queries"]["easy"][0], list)
    assert "natural_caption" not in queries


def test_explicit_attribute_json_uses_single_identity_metadata_fallbacks(tmp_path: Path) -> None:
    scene_dir = tmp_path / "person"
    attribute_json = tmp_path / "attributes.json"
    write_json(
        attribute_json,
        {
            "attributes": {"top_outer_color": "green"},
            "image_id": "view_a.jpg",
        },
    )
    entry = DataEntry(media_path="sentinel.jpg", data_path=str(scene_dir))

    PersonAttributeSearchTask(
        PersonAttributeSearchConfig(
            attribute_json=str(attribute_json), dataset="custom", write_hitl=False
        )
    ).run(entry)

    attributes = read_json(scene_dir / "sidecars" / "person_attribute_search" / "attributes.json")
    assert attributes["person_key"] == "sentinel_custom"
    assert attributes["person_id"] == "sentinel"
    assert attributes["dataset"] == "custom"


def test_explicit_attributes_use_bundle_generator_with_flat_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attribute_json = tmp_path / "attributes.json"
    write_json(
        attribute_json,
        {
            "entries": [
                {
                    "person_key": "person_aug0",
                    "person_id": "person",
                    "augmentation_id": "0",
                    "image_id": "person_aug0/front.jpg",
                    "attributes": {
                        "top_outer_color": "red",
                        "top_outer_type": "jacket",
                        "viewpoint": "front",
                        "accessories": [],
                    },
                },
                {
                    "person_key": "person_aug0",
                    "person_id": "person",
                    "augmentation_id": "0",
                    "image_id": "person_aug0/side.jpg",
                    "attributes": {
                        "top_outer_color": "red",
                        "top_outer_type": "jacket",
                        "viewpoint": "side",
                        "accessories": ["backpack"],
                    },
                },
            ]
        },
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.prompt = ""
            self.calls = 0

        def generate(self, request: ChatRequest) -> str:
            self.prompt = request.prompt
            self.calls += 1
            return json.dumps({"queries": {"easy": ["e1"], "medium": ["m1"], "hard": ["h1"]}})

    client = _FakeClient()
    monkeypatch.setattr(
        "person_attribute_search.task.create_endpoint_client",
        lambda **_kwargs: client,
    )
    scene_dir = tmp_path / "scene"
    entry = DataEntry(media_path="sentinel.jpg", data_path=str(scene_dir))
    config = PersonAttributeSearchConfig(
        attribute_json=str(attribute_json),
        bundle_query_generation=True,
        query_prompt_text="Generate from {attributes}",
        bundle_query_count=1,
        bundle_easy_count=1,
        bundle_medium_count=1,
        bundle_hard_count=1,
        llm_model="model",
        hitl_image_url_base="https://images/",
    )

    PersonAttributeSearchTask(config).run(entry)

    output_dir = scene_dir / "sidecars" / "person_attribute_search"
    assert not (output_dir / "queries.json").exists()
    assert not (output_dir / "attributes.json").exists()
    query_bundle = read_json(output_dir / "bundle_queries.json")
    attribute_bundle = read_json(output_dir / "bundle_attributes.json")
    hitl_bundle = read_json(output_dir / "bundle_hitl.json")
    assert {
        query_bundle["chunk_id"],
        attribute_bundle["chunk_id"],
        hitl_bundle["chunk_id"],
    } == {"0"}
    assert {
        query_bundle["n_people"],
        attribute_bundle["n_people"],
        hitl_bundle["n_people"],
    } == {2}
    expected_keys = {"person_aug0/front.jpg", "person_aug0/side.jpg"}
    assert set(query_bundle["people"]) == expected_keys
    assert set(attribute_bundle["people"]) == expected_keys
    assert set(hitl_bundle["people"]) == expected_keys
    assert query_bundle["people"]["person_aug0/front.jpg"]["queries"] == {
        "easy": ["e1"],
        "medium": ["m1"],
        "hard": ["h1"],
    }
    assert attribute_bundle["people"]["person_aug0/front.jpg"] == {
        "track_id": 0,
        "attributes": {
            "top outer type": "jacket",
            "top outer color": "red",
            "viewpoint": "front",
            "accessories": [],
        },
    }
    assert attribute_bundle["people"]["person_aug0/side.jpg"]["track_id"] == 1
    assert attribute_bundle["people"]["person_aug0/side.jpg"]["attributes"]["accessories"] == [
        "backpack"
    ]
    state = read_pipeline_state(scene_dir)
    artifacts = state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY]
    assert artifacts["attributes_json"] is None
    assert artifacts["bundle_attributes_json"] == str(output_dir / "bundle_attributes.json")
    assert artifacts["hitl_json"] is None
    assert artifacts["bundle_hitl_json"] == str(output_dir / "bundle_hitl.json")
    assert not (output_dir / "hitl.json").exists()
    assert hitl_bundle["people"]["person_aug0/front.jpg"]["image_url"] == (
        "https://images/person_aug0/front.jpg"
    )
    assert client.calls == 2
    assert "red" in client.prompt
