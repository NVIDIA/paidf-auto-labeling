# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic data models shared by the core framework.

``DataEntry`` is the in-process handle passed through pipelines. The
pipeline state models describe shared artifact references persisted at
``<scene>/sidecars/pipeline_state.json`` so cross-task state can be recovered
from the scene directory itself. Task-specific payload schemas belong to the
task packages that write them and are stored under ``task_artifacts``.
"""

from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DataEntry(BaseModel):
    """Pipeline handle for a single scene.

    ``DataEntry`` carries the stable fields every task needs to locate the
    input media and the DAFT scene directory. The service input JSONL can be
    reused across containers because ``media_path`` is treated as immutable
    caller input; the pipeline preserves the original media under
    ``data_path/sidecars`` as ``raw<source suffix>`` and stages the currently
    active media under ``data_path/sidecars`` for task execution. Tasks should
    write durable annotations, sidecars, and any updated media under
    ``data_path``. Shared artifact references that need to survive across
    tasks, processes, or containers belong in ``core.ScenePipelineState`` at
    ``<scene>/sidecars/pipeline_state.json``. Task-specific schemas should live
    in the task package and be stored under ``ScenePipelineState.task_artifacts``.

    Unknown fields are forbidden on the base model so typos and implicit
    in-memory state do not silently enter the pipeline. If a future workflow
    needs task-specific entry fields, that should be modeled with an explicit
    typed interface instead of ad hoc extras.

    Fields:
        id: Unique identifier for this data entry.
        media_path: Local or remote path to the original media file to annotate.
        data_path: Local or remote path to the DAFT scene directory.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="The unique identifier for the data entry.",
    )
    media_path: str = Field(..., description="The path to the media file to be annotated.")
    data_path: str = Field(
        ...,
        description=(
            "Path to an existing or new DAFT data directory containing annotations, active media, "
            "and other scene metadata."
        ),
    )


PIPELINE_STATE_SCHEMA_VERSION: str = "1"
"""Bumped on a non-backwards-compatible change to the pipeline state layout."""

PIPELINE_STATE_FILENAME: str = "pipeline_state.json"
"""State is persisted at ``<scene>/sidecars/<PIPELINE_STATE_FILENAME>``."""


class EnhancedMediaState(BaseModel):
    """State slice for enhanced media consumed by downstream tasks."""

    success: bool = False
    output_path: str | None = None


class EmitterOutcomeState(BaseModel):
    """One annotation emitter's persisted outcome."""

    name: str
    success: bool
    artifact: str | None = None


class AnnotationExportState(BaseModel):
    """State slice for annotation export outcomes."""

    success: bool = False
    emitters: list[EmitterOutcomeState] = Field(default_factory=list)


class ScenePipelineState(BaseModel):
    """Aggregate per-scene state shared across tasks.

    Persisted to ``<scene>/sidecars/pipeline_state.json``. Tasks load the
    existing pipeline state (empty if missing), update their own artifact
    contract slice or ``task_artifacts`` entry, and save the result. Other
    slices are passed through unchanged so a task running in isolation does not
    destroy state produced by a sibling task in a prior run.
    """

    schema_version: str = PIPELINE_STATE_SCHEMA_VERSION
    data_entry_id: str | None = None
    media_id: str | None = None
    media_path: str | None = None
    enhanced_media: EnhancedMediaState | None = None
    task_artifacts: dict[str, dict[str, object]] = Field(default_factory=dict)
    annotation_export: AnnotationExportState | None = None


def pipeline_state_path(scene_dir: Path | str) -> Path:
    """Return the canonical pipeline state path for ``scene_dir``."""
    return Path(scene_dir) / "sidecars" / PIPELINE_STATE_FILENAME


def read_pipeline_state(scene_dir: Path | str) -> ScenePipelineState:
    """Read the scene pipeline state if present; otherwise return an empty one.

    A missing or empty pipeline state is *not* an error; first-touch tasks
    encounter this state by design. Schema-validation errors are
    propagated so corrupt files surface loudly.
    """
    path = pipeline_state_path(scene_dir)
    if not path.exists():
        return ScenePipelineState()
    state: ScenePipelineState = ScenePipelineState.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return state


def write_pipeline_state(scene_dir: Path | str, pipeline_state: ScenePipelineState) -> Path:
    """Atomically write the scene pipeline state to disk.

    Writes to ``<path>.tmp`` then renames into place so a partially
    written file is never visible to concurrent readers. Creates the
    ``sidecars/`` directory if missing.
    """
    path = pipeline_state_path(scene_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(pipeline_state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def update_annotation_export_state(
    scene_dir: Path | str,
    *,
    emitter_artifacts: Mapping[str, Path | str | None],
    data_entry_id: str | None = None,
    media_path: str | None = None,
) -> Path:
    """Merge annotation-export emitter outcomes into the scene pipeline state.

    Only emitters named in ``emitter_artifacts`` are replaced. Existing
    outcomes for other emitters are preserved so independently run services do
    not erase each other's artifact inventory.
    """
    state = read_pipeline_state(scene_dir)
    owned_names = set(emitter_artifacts)
    existing_emitters = (
        []
        if state.annotation_export is None
        else [
            emitter
            for emitter in state.annotation_export.emitters
            if emitter.name not in owned_names
        ]
    )
    updated_emitters = [
        _emitter_outcome_from_artifact(name, artifact)
        for name, artifact in emitter_artifacts.items()
    ]
    emitters = [*existing_emitters, *updated_emitters]

    updates: dict[str, object] = {
        "annotation_export": AnnotationExportState(
            success=any(emitter.success for emitter in emitters),
            emitters=emitters,
        )
    }
    if data_entry_id is not None:
        updates["data_entry_id"] = data_entry_id
    if media_path is not None:
        updates["media_path"] = state.media_path or media_path

    updated = state.model_copy(update=updates)
    return write_pipeline_state(scene_dir, updated)


def _emitter_outcome_from_artifact(
    name: str,
    artifact: Path | str | None,
) -> EmitterOutcomeState:
    if artifact is None:
        return EmitterOutcomeState(name=name, success=False)
    artifact_path = Path(artifact)
    if not artifact_path.exists():
        return EmitterOutcomeState(name=name, success=False)
    return EmitterOutcomeState(name=name, success=True, artifact=str(artifact_path))


__all__ = [
    "PIPELINE_STATE_FILENAME",
    "PIPELINE_STATE_SCHEMA_VERSION",
    "AnnotationExportState",
    "DataEntry",
    "EmitterOutcomeState",
    "EnhancedMediaState",
    "ScenePipelineState",
    "pipeline_state_path",
    "read_pipeline_state",
    "write_pipeline_state",
    "update_annotation_export_state",
]
