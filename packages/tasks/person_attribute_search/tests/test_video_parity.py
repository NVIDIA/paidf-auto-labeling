# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-chunk video output-contract parity harness.

The legacy query strings are LLM-generated, so byte-identical *generation* is
impossible to reproduce offline. What this harness locks is the **output
contract**: given a faithfully reconstructed ``track_inputs.json`` seam (built
from the golden ``pas.json`` attributes + the golden query strings), the UPA
per-track assembly must reproduce the golden ``pas.json`` and ``queries.json``
documents structurally. This guards against any future drift in the assembled
shapes during refactors.

Fixtures under ``fixtures/video/`` are copied verbatim from the upstream
video pseudo-labelling ``sample_output``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import DataEntry, ensure_scene_skeleton, read_json, write_json
from core.formats.daft import DAFT_VERSION
from person_attribute_search.config import PersonAttributeSearchConfig
from person_attribute_search.task import PersonAttributeSearchTask
from person_attribute_search.track_inputs import assemble_track_inputs

_FIXTURES = Path(__file__).parent / "fixtures" / "video"


def _load_golden() -> tuple[dict[str, Any], dict[str, Any]]:
    pas = json.loads((_FIXTURES / "pas_chunk_000.json").read_text(encoding="utf-8"))
    queries = json.loads((_FIXTURES / "queries_chunk_000.json").read_text(encoding="utf-8"))
    return pas, queries


def _reconstruct_seam(pas_golden: dict[str, Any], queries_golden: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the seam that *would* have produced the golden documents."""
    people = pas_golden["pas"]["people"]
    assert len(people) == 1, "fixture assumes a single-person chunk"
    person = people[0]
    track_id = int(person["track_id"])

    # The golden query list is: per-person tiered queries, then anomaly-derived,
    # then caption-derived. With one person, the first six are this track's
    # queries and the remainder are chunk-level caption/anomaly queries.
    all_queries = [entry["query"] for entry in queries_golden["queries"]]
    person_queries = all_queries[:6]
    chunk_queries = all_queries[6:]

    attributes = person["attributes"]
    tracks_payload = {
        "crop_root": pas_golden["crop_root"],
        "tracks": [
            {
                "track_id": track_id,
                "detection_score": person["detection_score"],
                "duration_sec": person["duration_sec"],
                "first_frame": person["first_frame"],
                "last_frame": person["last_frame"],
                "n_crops": person["n_crops"],
                "crop_dir": person["crop_dir"],
            }
        ],
    }
    per_track = {
        track_id: {
            "items": [{"id": key, "answer": value} for key, value in attributes.items()],
            "caption": person["natural_language_caption"],
            "queries": person_queries,
            "n_crops_in_chunk": person["n_crops_in_chunk"],
            "first_frame_in_chunk": person["first_frame_in_chunk"],
            "last_frame_in_chunk": person["last_frame_in_chunk"],
        }
    }
    return assemble_track_inputs(
        chunk_id=pas_golden["chunk_id"],
        tracks_payload=tracks_payload,
        per_track=per_track,
        source_annotation=pas_golden["source_annotation"],
        caption_queries=chunk_queries,
        crop_root=pas_golden["crop_root"],
    )


def _run_task(tmp_path: Path, seam: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_dir = tmp_path / "chunk_000"
    paths = ensure_scene_skeleton(scene_dir)
    config = PersonAttributeSearchConfig()
    write_json(paths.sidecars_dir / config.track_inputs_sidecar, seam)

    entry = DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))
    PersonAttributeSearchTask(config).run(entry)

    produced_pas = read_json(paths.sidecars_dir / config.output_pas_sidecar)
    produced_queries = read_json(paths.sidecars_dir / config.output_chunk_queries_sidecar)
    return produced_pas, produced_queries


def test_video_pas_document_matches_golden(tmp_path: Path) -> None:
    pas_golden, queries_golden = _load_golden()
    seam = _reconstruct_seam(pas_golden, queries_golden)
    produced_pas, _ = _run_task(tmp_path, seam)

    assert produced_pas == pas_golden


def test_video_queries_document_matches_golden(tmp_path: Path) -> None:
    pas_golden, queries_golden = _load_golden()
    seam = _reconstruct_seam(pas_golden, queries_golden)
    _, produced_queries = _run_task(tmp_path, seam)

    # The query list is the meaningful contract. ``source_annotation`` is a
    # provenance pointer that the legacy video flow stores differently in pas vs queries
    # (a known legacy inconsistency), so it is asserted to exist but excluded
    # from the list comparison.
    assert produced_queries["chunk_id"] == queries_golden["chunk_id"]
    assert produced_queries["queries"] == queries_golden["queries"]
    assert "source_annotation" in produced_queries


def _strip_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the DAFT body with the envelope keys removed."""
    return {k: v for k, v in payload.items() if k not in ("version", "metadata", "video_id")}


def test_daft_contextual_mirror_dual_writes_and_preserves_body(tmp_path: Path) -> None:
    # The DAFT mirror is additive: sidecars remain the source of truth and the
    # enveloped contextual copies carry the identical body verbatim.
    pas_golden, queries_golden = _load_golden()
    seam = _reconstruct_seam(pas_golden, queries_golden)
    scene_dir = tmp_path / "chunk_000"
    paths = ensure_scene_skeleton(scene_dir)
    config = PersonAttributeSearchConfig()
    write_json(paths.sidecars_dir / config.track_inputs_sidecar, seam)
    entry = DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))
    PersonAttributeSearchTask(config).run(entry)

    # Sidecars still written (dual-write, lossless round-trip).
    sidecar_pas = read_json(paths.sidecars_dir / config.output_pas_sidecar)
    sidecar_queries = read_json(paths.sidecars_dir / config.output_chunk_queries_sidecar)

    # DAFT contextual copies exist with a valid envelope + matching body.
    daft_pas = read_json(paths.contextual_dir / "person_attributes.json")
    daft_queries = read_json(paths.contextual_dir / "pas_queries.json")

    assert daft_pas["version"] == DAFT_VERSION
    assert daft_pas["metadata"]["type"] == "person_attributes"
    assert daft_queries["version"] == DAFT_VERSION
    assert daft_queries["metadata"]["type"] == "pas_queries"

    assert _strip_envelope(daft_pas) == sidecar_pas
    assert _strip_envelope(daft_queries) == sidecar_queries


def test_daft_contextual_mirror_can_be_disabled(tmp_path: Path) -> None:
    pas_golden, queries_golden = _load_golden()
    seam = _reconstruct_seam(pas_golden, queries_golden)
    scene_dir = tmp_path / "chunk_000"
    paths = ensure_scene_skeleton(scene_dir)
    config = PersonAttributeSearchConfig(emit_daft_contextual=False)
    write_json(paths.sidecars_dir / config.track_inputs_sidecar, seam)
    entry = DataEntry(media_path="chunk_000.mp4", data_path=str(scene_dir))
    PersonAttributeSearchTask(config).run(entry)

    assert (paths.sidecars_dir / config.output_pas_sidecar).exists()
    assert not (paths.contextual_dir / "person_attributes.json").exists()
    assert not (paths.contextual_dir / "pas_queries.json").exists()
