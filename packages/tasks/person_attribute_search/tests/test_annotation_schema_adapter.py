# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the annotation schema adapter (``person_attribute_search.annotation_schema_adapter``).

A synthetic UPA two-pass scene directory is built in a tmp path (no model calls,
no real media) to verify the model-free merge into the legacy schema and the
merged ``sidecars/person_attribute_search/pas_anomaly.json`` output. Pure helpers
are tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from person_attribute_search import annotation_schema_adapter as adapter


@pytest.mark.parametrize(
    "scene_name, expected",
    [
        (
            "gopro_v2_person_falling_0002_img2video_chunk_000.mp4",
            (
                "gopro_v2_person_falling_0002_img2video_chunk_000",
                "gopro_v2_person_falling_0002_img2video",
                0,
            ),
        ),
        ("clip_chunk_012", ("clip_chunk_012", "clip", 12)),
        ("standalone.mp4", ("standalone", "standalone", 0)),
    ],
)
def test_split_chunk_label(scene_name: str, expected: tuple[str, str, int]) -> None:
    """Scene dir names split into (chunk_label, video_stem, chunk_id)."""
    assert adapter.split_chunk_label(scene_name) == expected


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("A. Yes", True),
        ("yes", True),
        ("Yes, clearly.", True),
        ("B. No", False),
        ("no", False),
        ("eyes", False),
        ("yesterday", False),
        (None, False),
    ],
)
def test_is_yes(answer: object, expected: bool) -> None:
    """BCQ answer strings are robustly classified as Yes/No (word-boundary)."""
    assert adapter._is_yes(answer) is expected


def test_fmt_timestamp() -> None:
    """Seconds render as MM:SS."""
    assert adapter._fmt_timestamp(0.0) == "00:00"
    assert adapter._fmt_timestamp(65.4) == "01:05"


def test_extract_anomaly_categories_only_yes() -> None:
    """Only anom_* items answered Yes become categories, order preserved."""
    items = {
        "items": [
            {"id": "anom_physical_fight", "answer": "B. No"},
            {"id": "anom_person_falling_or_collapsing", "answer": "A. Yes"},
            {"id": "anom_warehouse_safety_violation", "answer": "A. Yes"},
            {"id": "age", "answer": "adult"},
        ]
    }
    assert adapter.extract_anomaly_categories(items) == [
        "person_falling_or_collapsing",
        "warehouse_safety_violation",
    ]


def test_extract_anomaly_categories_skips_malformed_items() -> None:
    items = {"items": ["junk", {"id": "anom_vandalism", "answer": "A. Yes"}, 7]}
    assert adapter.extract_anomaly_categories(items) == ["vandalism"]


def test_build_anomaly_gt_single_model() -> None:
    """anomaly_gt is single-model with n_models_ok == 1 and a voted_gt."""
    items = {"items": [{"id": "anom_vandalism", "answer": "A. Yes"}]}
    block = adapter.build_anomaly_gt(items, model_name="upa/visual_qa")
    assert block["voted_categories"] == ["vandalism"]
    assert block["per_category_votes"] == {"vandalism": 1}
    assert block["per_model_votes"] == {"upa/visual_qa": ["vandalism"]}
    assert block["n_models_ok"] == 1
    assert block["voted_gt"] == "vandalism"
    assert block["is_strict_majority"] is True


def test_build_anomaly_gt_confirmed_normal_when_all_no() -> None:
    """Anomaly BCQs present and all answered 'No' is a confirmed normal vote."""
    items = {"items": [{"id": "anom_vandalism", "answer": "B. No"}]}
    block = adapter.build_anomaly_gt(items, model_name="m")
    assert block["voted_categories"] == []
    assert block["voted_gt"] is None
    assert block["is_strict_majority"] is False
    assert block["n_models_ok"] == 1
    assert block["per_model_votes"] == {"m": []}


def test_build_anomaly_gt_missing_evidence_fails_closed() -> None:
    """No anomaly BCQ items means missing pass-2 output: fail closed, not normal."""
    block = adapter.build_anomaly_gt({"items": []}, model_name="m")
    assert block["voted_categories"] == []
    assert block["voted_gt"] is None
    assert block["is_strict_majority"] is False
    assert block["n_models_ok"] == 0
    assert block["per_model_votes"] == {}


def test_build_anomaly_gt_missing_sidecar_fails_closed() -> None:
    """A missing anomaly sidecar (None) fails closed rather than minting normal."""
    block = adapter.build_anomaly_gt(None, model_name="m")
    assert block["voted_categories"] == []
    assert block["voted_gt"] is None
    assert block["n_models_ok"] == 0
    assert block["per_model_votes"] == {}


def test_build_captions_aggregates_windows() -> None:
    """Per-window captions fold into legacy scene + dense caption shapes."""
    video_captions = {
        "summary": "A warehouse scene.",
        "windows": [
            {"start_time": 0.0, "end_time": 4.0, "caption": "Workers walk."},
            {"start_time": 4.0, "end_time": 5.0, "caption": "One slips."},
        ],
    }
    scene, dense = adapter.build_captions(video_captions, has_anomaly=True)
    assert scene["parsed"] == {"scene_caption": "A warehouse scene."}
    assert dense["parsed"]["anomaly"] == "yes"
    assert dense["parsed"]["dense caption"] == (
        "00:00 - 00:04: Workers walk.\n00:04 - 00:05: One slips."
    )


def test_build_captions_empty_is_null_parsed() -> None:
    """No caption windows yields null parsed blocks (no crash)."""
    scene, dense = adapter.build_captions({"windows": []}, has_anomaly=False)
    assert scene["parsed"] is None
    assert dense["parsed"] is None


def test_reshape_queries_pas_only() -> None:
    """Without buckets, UPA flat queries land under PAS; others empty (fallback)."""
    chunk_queries = {"queries": [{"query": "yellow vest"}, {"query": "black pants"}]}
    out = adapter.reshape_queries(chunk_queries)
    assert out["PAS"] == [["yellow vest", ""], ["black pants", ""]]
    assert out["Anomaly"] == []
    assert out["Caption"] == []


def test_reshape_queries_passes_through_buckets() -> None:
    """When query_buckets are present, all three buckets pass through as pairs."""
    chunk_queries = {
        "queries": [{"query": "yellow vest"}],
        "query_buckets": {
            "PAS": [["person wearing yellow vest", "top outer"]],
            "Anomaly": [["person slipping on floor", "fall category"]],
            "Caption": [["busy warehouse aisle", "scene"]],
        },
    }
    out = adapter.reshape_queries(chunk_queries)
    assert out["PAS"] == [["person wearing yellow vest", "top outer"]]
    assert out["Anomaly"] == [["person slipping on floor", "fall category"]]
    assert out["Caption"] == [["busy warehouse aisle", "scene"]]


def test_reshape_queries_buckets_tolerate_strings_and_drop_empty() -> None:
    """Bucket entries may be bare strings; empty-query entries are dropped."""
    chunk_queries = {
        "query_buckets": {
            "PAS": ["bare string", "", ["", "evidence"]],
            "Anomaly": [],
            "Caption": [],
        }
    }
    out = adapter.reshape_queries(chunk_queries)
    assert out["PAS"] == [["bare string", ""]]
    assert out["Anomaly"] == []
    assert out["Caption"] == []


def test_official_for() -> None:
    """official_for resolves by stem or stem.mp4, else None."""
    lookup = {"clip": "vandalism", "other.mp4": "fighting"}
    assert adapter.official_for("clip", lookup) == "vandalism"
    assert adapter.official_for("other", lookup) == "fighting"
    assert adapter.official_for("missing", lookup) is None
    assert adapter.official_for("clip", None) is None


def _make_scene(out_dir: Path, scene_name: str) -> Path:
    """Create a minimal combined pass1+pass2 scene directory under out_dir."""
    scene = out_dir / scene_name
    pas = scene / "sidecars" / "person_attribute_search"
    vqa = scene / "sidecars" / "visual_qa_anomaly"
    cap = scene / "sidecars" / "captioning"
    for directory in (pas, vqa, cap):
        directory.mkdir(parents=True, exist_ok=True)

    (pas / "pas.json").write_text(
        json.dumps(
            {
                "chunk_id": scene_name,
                "crop_root": "tracks/crops",
                "pas": {
                    "n_people": 1,
                    "people": [
                        {
                            "track_id": 0,
                            "detection_score": 0.9,
                            "n_crops": 16,
                            "attributes": {"age": "adult", "gender": "male"},
                            "natural_language_caption": "An adult male.",
                        }
                    ],
                },
            }
        )
    )
    (pas / "chunk_queries.json").write_text(
        json.dumps({"chunk_id": scene_name, "queries": [{"query": "yellow vest"}]})
    )
    (vqa / "items.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "anom_person_falling_or_collapsing",
                        "options": ["A. Yes", "B. No"],
                        "answer": "A. Yes",
                    }
                ]
            }
        )
    )
    (cap / "video_captions.json").write_text(
        json.dumps(
            {
                "summary": "Warehouse aisle.",
                "windows": [{"start_time": 0.0, "end_time": 4.0, "caption": "Workers walk."}],
            }
        )
    )
    (cap / "metadata_chunk.json").write_text(
        json.dumps({"media": {"framerate": 24.0, "width": 1280, "height": 720, "num_frames": 120}})
    )
    return scene


def test_build_chunk_record_merges_all(tmp_path: Path) -> None:
    """A combined scene produces a full legacy-shaped record."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    record = adapter.build_chunk_record(scene, model_name="upa/visual_qa", official_anomaly=None)
    assert record is not None
    assert record["video_stem"] == "clip"
    assert record["chunk_label"] == "clip_chunk_000"
    assert record["chunk_id"] == 0
    assert record["fps"] == 24.0
    assert record["width"] == 1280
    assert record["height"] == 720
    chunk = record["chunk"]
    assert chunk["pas"]["n_people"] == 1
    assert chunk["anomaly_gt"]["voted_gt"] == "person_falling_or_collapsing"
    assert chunk["dense_caption"]["parsed"]["anomaly"] == "yes"
    assert chunk["frame_range"] == [0, 119]
    assert chunk["queries"]["PAS"] == [["yellow vest", ""]]


def test_build_chunk_record_skips_non_scene(tmp_path: Path) -> None:
    """A directory without a PAS sidecar is not a scene (returns None)."""
    (tmp_path / "helper").mkdir()
    assert adapter.build_chunk_record(tmp_path / "helper", "m", None) is None


def test_convert_out_dir_writes_merged_record(tmp_path: Path) -> None:
    """convert_out_dir writes sidecars/person_attribute_search/pas_anomaly.json per scene."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    pas_sidecar = scene / "sidecars" / "person_attribute_search"
    pas_doc = json.loads((pas_sidecar / "pas.json").read_text())

    counts = adapter.convert_out_dir(tmp_path, model_name="upa/visual_qa")
    assert counts == {"scenes": 1}

    written_merged = json.loads((pas_sidecar / "pas_anomaly.json").read_text())
    assert written_merged["chunk"]["anomaly_gt"]["voted_gt"] == "person_falling_or_collapsing"
    assert written_merged["chunk"]["pas"] == pas_doc["pas"]
    # pas.json is not duplicated into contextual/; nothing lands under contextual/.
    assert not (scene / "contextual" / "pas.json").exists()
    assert not (scene / "contextual" / "pas_anomaly.json").exists()


def test_convert_out_dir_applies_official_lookup(tmp_path: Path) -> None:
    """official_lookup populates official_anomaly by video stem."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    adapter.convert_out_dir(
        tmp_path, model_name="m", official_lookup={"clip": "person_falling_or_collapsing"}
    )
    merged = json.loads(
        (scene / "sidecars" / "person_attribute_search" / "pas_anomaly.json").read_text()
    )
    assert merged["official_anomaly"] == "person_falling_or_collapsing"


def test_convert_preserves_original_sidecars(tmp_path: Path) -> None:
    """The adapter only adds pas_anomaly.json; pre-existing sidecars stay intact."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    before = {p: p.read_bytes() for p in (scene / "sidecars").rglob("*.json")}

    adapter.convert_out_dir(tmp_path, model_name="m")

    after = {p: p.read_bytes() for p in (scene / "sidecars").rglob("*.json")}
    # Every originally-present sidecar is byte-for-byte unchanged.
    assert {p: after[p] for p in before} == before
    # The only new sidecar is the merged anomaly record.
    new_paths = set(after) - set(before)
    assert new_paths == {scene / "sidecars" / "person_attribute_search" / "pas_anomaly.json"}


def test_build_official_lookup_list_form(tmp_path: Path) -> None:
    """A list dataset.json indexes the category under filename/stem variants."""
    ds = tmp_path / "dataset.json"
    ds.write_text(
        json.dumps(
            [
                {"filename": "clip_a.mp4", "anomaly_type": "vandalism"},
                {"target_stem": "clip_b", "category": "fighting"},
            ]
        )
    )
    lookup = adapter.build_official_lookup(ds)
    assert lookup["clip_a.mp4"] == "vandalism"
    assert lookup["clip_a"] == "vandalism"
    # Source label "fighting" is normalized onto the canonical KPI id.
    assert lookup["clip_b"] == "physical_fight"


def test_build_official_lookup_normalizes_and_keeps_unknown(tmp_path: Path) -> None:
    """Known source labels normalize to canonical ids; unknown labels stay verbatim."""
    ds = tmp_path / "dataset.json"
    ds.write_text(
        json.dumps(
            [
                {"filename": "a.mp4", "anomaly_type": "shoplifting"},
                {"filename": "b.mp4", "anomaly_type": "person_falling"},
                {"filename": "c.mp4", "anomaly_type": "something_not_in_ontology"},
            ]
        )
    )
    lookup = adapter.build_official_lookup(ds)
    assert lookup["a"] == "stealing_or_shoplifting"
    assert lookup["b"] == "person_falling_or_collapsing"
    assert lookup["c"] == "something_not_in_ontology"


def test_build_official_lookup_drops_intentional_aliases(tmp_path: Path) -> None:
    """Recognized-but-dropped aliases (e.g. 'abnormal') are excluded, not kept verbatim."""
    ds = tmp_path / "dataset.json"
    ds.write_text(
        json.dumps(
            [
                {"filename": "a.mp4", "anomaly_type": "abnormal"},
                {"filename": "b.mp4", "anomaly_type": "anomalous"},
                {"filename": "c.mp4", "anomaly_type": "shoplifting"},
            ]
        )
    )
    lookup = adapter.build_official_lookup(ds)
    assert "a" not in lookup
    assert "b" not in lookup
    assert lookup["c"] == "stealing_or_shoplifting"


def test_build_official_lookup_videos_dict_form(tmp_path: Path) -> None:
    """A {"videos": {...}} dataset.json is parsed and keyed by stem."""
    ds = tmp_path / "dataset.json"
    ds.write_text(json.dumps({"videos": {"clip_c.mp4": {"label": "person_falling_or_collapsing"}}}))
    lookup = adapter.build_official_lookup(ds)
    assert lookup["clip_c"] == "person_falling_or_collapsing"


def test_build_official_lookup_missing_file(tmp_path: Path) -> None:
    """A missing dataset.json yields an empty lookup (not an error)."""
    assert adapter.build_official_lookup(tmp_path / "nope.json") == {}


def test_main_writes_merged_record_with_official_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI converts a scene and applies official_anomaly from --official-json."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"clip": "person_falling_or_collapsing"}))

    code = adapter.main(
        [
            "--out-dir",
            str(tmp_path),
            "--model-name",
            "upa/visual_qa",
            "--official-json",
            str(official),
        ]
    )
    assert code == 0
    assert "1 scene(s)" in capsys.readouterr().out

    merged = json.loads(
        (scene / "sidecars" / "person_attribute_search" / "pas_anomaly.json").read_text()
    )
    assert merged["official_anomaly"] == "person_falling_or_collapsing"
    assert merged["chunk"]["anomaly_gt"]["per_model_votes"] == {
        "upa/visual_qa": ["person_falling_or_collapsing"]
    }


def test_load_official_lookup_rejects_non_string_values(tmp_path: Path) -> None:
    """--official-json values must be non-empty strings (null/empty are rejected)."""
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"clip": None}))
    with pytest.raises(ValueError, match="non-empty string"):
        adapter._load_official_lookup(None, official)

    official.write_text(json.dumps({"clip": "  "}))
    with pytest.raises(ValueError, match="non-empty string"):
        adapter._load_official_lookup(None, official)


def test_main_rejects_non_directory(tmp_path: Path) -> None:
    """CLI errors (SystemExit) when --out-dir is not a directory."""
    with pytest.raises(SystemExit):
        adapter.main(["--out-dir", str(tmp_path / "missing")])


def _write_anomaly_windows(scene: Path, sub: str, model: str) -> None:
    """Write a minimal windows sidecar that records the visual_qa model."""
    target = scene / "sidecars" / sub
    target.mkdir(parents=True, exist_ok=True)
    (target / "windows.json").write_text(
        json.dumps({"windows": [{"visual_qa_call": {"model": model}}]})
    )


def test_detect_voter_model_prefers_anomaly_namespace(tmp_path: Path) -> None:
    """detect_voter_model reads visual_qa_anomaly/ before the default visual_qa/."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    _write_anomaly_windows(scene, "visual_qa", "pas/model")
    _write_anomaly_windows(scene, "visual_qa_anomaly", "anomaly/model")
    assert adapter.detect_voter_model(scene / "sidecars") == "anomaly/model"


def test_detect_voter_model_falls_back_to_visual_qa(tmp_path: Path) -> None:
    """Anomaly-only runs (no visual_qa_anomaly/) fall back to visual_qa/."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    _write_anomaly_windows(scene, "visual_qa", "only/model")
    assert adapter.detect_voter_model(scene / "sidecars") == "only/model"


def test_detect_voter_model_none_when_absent(tmp_path: Path) -> None:
    """No windows sidecar yields None (caller applies the default)."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    assert adapter.detect_voter_model(scene / "sidecars") is None


def test_build_chunk_record_auto_detects_model(tmp_path: Path) -> None:
    """With model_name=None the voter is read from the sidecar."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    _write_anomaly_windows(scene, "visual_qa_anomaly", "Qwen/Qwen3.6-35B-A3B-FP8")
    record = adapter.build_chunk_record(scene)
    assert record is not None
    assert list(record["chunk"]["anomaly_gt"]["per_model_votes"]) == ["Qwen/Qwen3.6-35B-A3B-FP8"]


def test_build_chunk_record_default_model_when_undetectable(tmp_path: Path) -> None:
    """No sidecar model and no override falls back to DEFAULT_MODEL_NAME."""
    scene = _make_scene(tmp_path, "clip_chunk_000.mp4")
    record = adapter.build_chunk_record(scene)
    assert record is not None
    assert list(record["chunk"]["anomaly_gt"]["per_model_votes"]) == [adapter.DEFAULT_MODEL_NAME]
