# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output/state helpers for the Person Attribute Search task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core import DataEntry, SceneContext, read_pipeline_state, write_pipeline_state
from core.formats.daft import DaftSceneWriter, daft_envelope

from person_attribute_search.annotation_schema_adapter import convert_scene
from person_attribute_search.artifacts import (
    PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY,
    PersonAttributeSearchArtifactsState,
)


def emit_daft_contextual(
    *,
    data_entry: DataEntry,
    ctx: SceneContext,
    pas_document: dict[str, Any],
    queries_document: dict[str, Any],
    logger: logging.Logger,
) -> str | None:
    """
    Mirror per-chunk PAS sidecars into DAFT contextual annotations.

    This is additive. The PAS sidecars remain the source of truth; the DAFT
    contextual copies make the same content discoverable by DAFT consumers.
    """
    try:
        writer = DaftSceneWriter(Path(data_entry.data_path), ctx=ctx)
        writer.write_contextual(
            "person_attributes",
            {**daft_envelope("person_attributes", ctx), **pas_document},
        )
        writer.write_contextual(
            "pas_queries",
            {**daft_envelope("pas_queries", ctx), **queries_document},
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("DAFT contextual mirror failed for %s: %s", data_entry.data_path, exc)
        return f"daft_contextual_mirror_failed: {exc}"
    logger.info(
        "PAS mirrored DAFT contextual person_attributes/pas_queries for %s",
        data_entry.data_path,
    )
    return None


def emit_anomaly_deliverable(*, data_entry: DataEntry, logger: logging.Logger) -> str | None:
    """
    Write the merged pass1+pass2 anomaly deliverable for this scene.

    Fails soft because it is an additive terminal artifact; already-written PAS
    sidecars should not be discarded when the merged adapter cannot run.
    """
    scene_dir = Path(data_entry.data_path)
    try:
        written = convert_scene(scene_dir)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Anomaly deliverable export failed for %s: %s", scene_dir, exc)
        return f"anomaly_deliverable_failed: {exc}"
    if written:
        logger.info("PAS wrote pas_anomaly deliverable for %s", scene_dir)
        return None
    else:
        logger.warning(
            "Anomaly deliverable export skipped for %s: no pas.json after PAS write.",
            scene_dir,
        )
        return "anomaly_deliverable_skipped: no pas.json after PAS write"


def record_state(
    data_entry: DataEntry,
    *,
    success: bool,
    attributes_json: Path | None = None,
    bundle_attributes_json: Path | None = None,
    queries_json: Path | None = None,
    bundle_queries_json: Path | None = None,
    hitl_json: Path | None = None,
    bundle_hitl_json: Path | None = None,
    pas_json: Path | None = None,
    chunk_queries_json: Path | None = None,
    n_people: int | None = None,
    warnings: list[str] | None = None,
    optional_failures: list[str] | None = None,
) -> None:
    """Persist PAS artifact references into the scene pipeline state."""
    state = read_pipeline_state(data_entry.data_path)
    state.data_entry_id = state.data_entry_id or data_entry.id
    state.media_path = state.media_path or data_entry.media_path
    state.task_artifacts[PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY] = (
        PersonAttributeSearchArtifactsState(
            success=success,
            attributes_json=str(attributes_json) if attributes_json else None,
            bundle_attributes_json=(
                str(bundle_attributes_json) if bundle_attributes_json else None
            ),
            queries_json=str(queries_json) if queries_json else None,
            bundle_queries_json=str(bundle_queries_json) if bundle_queries_json else None,
            hitl_json=str(hitl_json) if hitl_json else None,
            bundle_hitl_json=str(bundle_hitl_json) if bundle_hitl_json else None,
            pas_json=str(pas_json) if pas_json else None,
            chunk_queries_json=str(chunk_queries_json) if chunk_queries_json else None,
            n_people=n_people,
            warnings=warnings or [],
            optional_failures=optional_failures or [],
        ).model_dump()
    )
    write_pipeline_state(data_entry.data_path, state)


__all__ = [
    "emit_anomaly_deliverable",
    "emit_daft_contextual",
    "record_state",
]
