# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed scene pipeline state cross-task contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from core import (
    PIPELINE_STATE_FILENAME,
    AnnotationExportState,
    EmitterOutcomeState,
    EnhancedMediaState,
    ScenePipelineState,
    pipeline_state_path,
    read_pipeline_state,
    update_annotation_export_state,
    write_pipeline_state,
)


def test_pipeline_state_path_is_under_sidecars(tmp_path: Path) -> None:
    """State lives under ``sidecars/`` so it stays outside the DAFT contract."""
    p = pipeline_state_path(tmp_path)
    assert p == tmp_path / "sidecars" / PIPELINE_STATE_FILENAME


def test_read_pipeline_state_returns_empty_when_missing(tmp_path: Path) -> None:
    """First-touch tasks expect an empty pipeline state, not an error."""
    pipeline_state = read_pipeline_state(tmp_path)
    assert isinstance(pipeline_state, ScenePipelineState)
    assert pipeline_state.enhanced_media is None
    assert pipeline_state.task_artifacts == {}
    assert pipeline_state.annotation_export is None


def test_write_then_read_roundtrips(tmp_path: Path) -> None:
    """Atomic write + read preserves every field."""
    pipeline_state = ScenePipelineState(
        data_entry_id="entry-1",
        media_id="clip",
        media_path="/in/clip.mp4",
        enhanced_media=EnhancedMediaState(success=True, output_path="/out/sr.mp4"),
        task_artifacts={
            "detection_and_tracking": {
                "success": True,
                "objects_json": "/out/objects.json",
                "instances_json": "/out/instances.json",
                "red_id_overlay_path": "/out/red.mp4",
            },
            "captioning": {
                "success": True,
                "video_json": "/out/video.json",
            },
            "visual_qa": {
                "success": True,
                "items_json": "/out/sidecars/visual_qa/items.json",
                "windows_json": "/out/sidecars/visual_qa/windows.normalized.json",
                "source_sidecar": "/out/sidecars/visual_qa/windows.json",
            },
        },
        annotation_export=AnnotationExportState(
            success=True,
            emitters=[EmitterOutcomeState(name="mcq", success=True, artifact="/out/mcq.json")],
        ),
    )
    written = write_pipeline_state(tmp_path, pipeline_state)
    assert written.exists()
    assert written.read_text(encoding="utf-8").endswith("\n")

    loaded = read_pipeline_state(tmp_path)
    assert loaded == pipeline_state


def test_caption_artifacts_preserve_dense_caption_sidecar(tmp_path: Path) -> None:
    """Captioning records dense sidecars that DAFT export consumes later."""
    pipeline_state = ScenePipelineState(
        task_artifacts={
            "captioning": {
                "success": True,
                "metadata_chunk_json": "/out/metadata_chunk.json",
                "video_json": "/out/video.json",
                "image_json": "/out/image.json",
            }
        }
    )
    write_pipeline_state(tmp_path, pipeline_state)

    loaded = read_pipeline_state(tmp_path)
    assert loaded.task_artifacts["captioning"]["metadata_chunk_json"] == "/out/metadata_chunk.json"
    assert loaded.task_artifacts["captioning"]["video_json"] == "/out/video.json"
    assert loaded.task_artifacts["captioning"]["image_json"] == "/out/image.json"


def test_visual_qa_artifacts_preserve_sidecar_contract(tmp_path: Path) -> None:
    """Visual QA records normalized item sidecars that DAFT export consumes later."""
    pipeline_state = ScenePipelineState(
        task_artifacts={
            "visual_qa": {
                "success": True,
                "items_json": "/out/sidecars/visual_qa/items.json",
                "windows_json": "/out/sidecars/visual_qa/windows.normalized.json",
                "source_sidecar": "/out/sidecars/metadata.json",
            }
        }
    )
    write_pipeline_state(tmp_path, pipeline_state)

    loaded = read_pipeline_state(tmp_path)
    artifacts = loaded.task_artifacts["visual_qa"]
    assert artifacts["success"] is True
    assert artifacts["items_json"] == "/out/sidecars/visual_qa/items.json"
    assert artifacts["windows_json"] == "/out/sidecars/visual_qa/windows.normalized.json"
    assert artifacts["source_sidecar"] == "/out/sidecars/metadata.json"


def test_write_is_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    """Tmp file is renamed into place; nothing left behind."""
    write_pipeline_state(tmp_path, ScenePipelineState(data_entry_id="x"))
    sidecars = tmp_path / "sidecars"
    leftovers = [p.name for p in sidecars.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_partial_update_preserves_other_sub_models(tmp_path: Path) -> None:
    """Each task updates its slice; sibling state survives."""
    initial = ScenePipelineState(
        enhanced_media=EnhancedMediaState(success=True, output_path="/out/sr.mp4"),
    )
    write_pipeline_state(tmp_path, initial)

    # Tracking task runs: load, mutate its slice, save.
    loaded = read_pipeline_state(tmp_path)
    loaded.task_artifacts["detection_and_tracking"] = {
        "success": True,
        "objects_json": "/out/objects.json",
    }
    write_pipeline_state(tmp_path, loaded)

    final = read_pipeline_state(tmp_path)
    assert final.enhanced_media is not None
    assert final.enhanced_media.output_path == "/out/sr.mp4"
    assert final.task_artifacts["detection_and_tracking"]["objects_json"] == "/out/objects.json"


def test_namespaced_visual_qa_variants_coexist_across_passes(tmp_path: Path) -> None:
    """Distinct visual_qa state keys from separate passes do not clobber each other."""
    # Pass 1 (anomaly) records under a namespaced key.
    pass1 = read_pipeline_state(tmp_path)
    pass1.task_artifacts["visual_qa_anomaly"] = {
        "success": True,
        "items_json": "/out/sidecars/visual_qa_anomaly/items.json",
    }
    write_pipeline_state(tmp_path, pass1)

    # Pass 3 (anomaly-search) records under a different namespaced key + canonical latest.
    pass3 = read_pipeline_state(tmp_path)
    pass3.task_artifacts["visual_qa_anomaly_search"] = {
        "success": True,
        "items_json": "/out/sidecars/visual_qa_anomaly_search/items.json",
    }
    pass3.task_artifacts["visual_qa"] = {
        "success": True,
        "items_json": "/out/sidecars/visual_qa_anomaly_search/items.json",
        "variant": "visual_qa_anomaly_search",
    }
    write_pipeline_state(tmp_path, pass3)

    final = read_pipeline_state(tmp_path)
    # Both passes' provenance survives.
    assert (
        final.task_artifacts["visual_qa_anomaly"]["items_json"]
        == "/out/sidecars/visual_qa_anomaly/items.json"
    )
    assert (
        final.task_artifacts["visual_qa_anomaly_search"]["items_json"]
        == "/out/sidecars/visual_qa_anomaly_search/items.json"
    )
    # Canonical key points at the most recent pass.
    assert final.task_artifacts["visual_qa"]["variant"] == "visual_qa_anomaly_search"


def test_update_annotation_export_state_replaces_only_owned_emitters(tmp_path: Path) -> None:
    existing_artifact = tmp_path / "task" / "mcq.json"
    existing_artifact.parent.mkdir()
    existing_artifact.touch()
    chunks_artifact = tmp_path / "contextual" / "chunks.json"
    chunks_artifact.parent.mkdir()
    chunks_artifact.touch()
    write_pipeline_state(
        tmp_path,
        ScenePipelineState(
            data_entry_id="old-entry",
            media_path="/out/active.mp4",
            annotation_export=AnnotationExportState(
                success=True,
                emitters=[
                    EmitterOutcomeState(
                        name="mcq",
                        success=True,
                        artifact=str(existing_artifact),
                    ),
                    EmitterOutcomeState(
                        name="chunks",
                        success=False,
                    ),
                ],
            ),
        ),
    )

    update_annotation_export_state(
        tmp_path,
        data_entry_id="new-entry",
        media_path="/in/original.mp4",
        emitter_artifacts={
            "chunks": chunks_artifact,
            "temporal_description": None,
        },
    )

    loaded = read_pipeline_state(tmp_path)
    assert loaded.data_entry_id == "new-entry"
    assert loaded.media_path == "/out/active.mp4"
    assert loaded.annotation_export is not None
    assert loaded.annotation_export.success is True
    outcomes = {emitter.name: emitter for emitter in loaded.annotation_export.emitters}
    assert outcomes["mcq"].success is True
    assert outcomes["mcq"].artifact == str(existing_artifact)
    assert outcomes["chunks"].success is True
    assert outcomes["chunks"].artifact == str(chunks_artifact)
    assert outcomes["temporal_description"].success is False
    assert outcomes["temporal_description"].artifact is None


def test_corrupt_pipeline_state_surfaces_error(tmp_path: Path) -> None:
    """Schema-validation errors are propagated so corrupt files are loud."""
    p = pipeline_state_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017 - pydantic-version-agnostic
        read_pipeline_state(tmp_path)
