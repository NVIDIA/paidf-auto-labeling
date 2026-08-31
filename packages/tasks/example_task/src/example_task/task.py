# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains the implementation of several example tasks which may be used as a template
for new tasks entering the repo.
"""

from typing import override

from core import (
    AnnotationExportState,
    EmitterOutcomeState,
    SceneContext,
    ensure_scene_skeleton,
    read_pipeline_state,
    write_json,
    write_pipeline_state,
)
from core.formats.daft import daft_envelope
from core.models import DataEntry
from core.tasks import SequentialTask

EXAMPLE_TASK_FILENAME = "example_task.json"
"""Sidecar filename used by the template task's example annotation payload."""

EXAMPLE_TASK_MANIFEST_FILENAME = "example_task_manifest.json"
"""Sidecar filename used by the template task's example artifact manifest."""


class SimpleTask(SequentialTask):
    """
    Template task that writes one example annotation artifact for a data entry.

    The task creates the DAFT scene skeleton, emits a small sidecar payload and manifest, and
    records its published artifact in ``ScenePipelineState``. New task packages can copy this
    shape when they need to write task outputs and update shared pipeline state.
    """

    def __init__(self, name: str | None = None, max_retries: int = 0) -> None:
        """
        Initialize the example task.

        Args:
            name: Optional task name used in logs and emitted artifact state.
            max_retries: Number of sequential retries for transient per-entry failures.
        """
        super().__init__(name=name, max_retries=max_retries)

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        """
        Write the example task sidecars and update scene pipeline state.

        Args:
            data_entry: Data entry whose DAFT directory should receive the example artifacts.
        Returns:
            The unchanged input data entry.
        """
        self.logger.debug(
            f"Running simple task on {data_entry.media_path} "
            f"and outputting to {data_entry.data_path}"
        )

        scene_paths = ensure_scene_skeleton(data_entry.data_path)
        scene_ctx = SceneContext.from_input(data_entry.media_path)

        task_output_path = scene_paths.sidecars_dir / EXAMPLE_TASK_FILENAME
        payload = daft_envelope(
            "example_task",
            scene_ctx,
            description="Garbage example annotations emitted by the template task.",
        )
        payload["data_entry_id"] = data_entry.id
        payload["source_media_path"] = data_entry.media_path
        payload["items"] = [
            {
                "id": "garbage-0",
                "label": "probably-a-thing",
                "confidence": 0.42,
                "bbox_xyxy": [12, 34, 156, 178],
            }
        ]
        write_json(task_output_path, payload)

        manifest_path = scene_paths.sidecars_dir / EXAMPLE_TASK_MANIFEST_FILENAME
        task_output_scene_path = task_output_path.relative_to(scene_paths.scene_dir).as_posix()
        manifest = {
            "data_entry_id": data_entry.id,
            "task_name": self.name,
            "published_artifacts": [task_output_scene_path],
        }
        write_json(manifest_path, manifest)

        pipeline_state = read_pipeline_state(data_entry.data_path)
        pipeline_state.data_entry_id = data_entry.id
        pipeline_state.media_path = data_entry.media_path
        existing_emitters = (
            []
            if pipeline_state.annotation_export is None
            else [
                emitter
                for emitter in pipeline_state.annotation_export.emitters
                if emitter.name != self.name
            ]
        )
        pipeline_state.annotation_export = AnnotationExportState(
            success=all(emitter.success for emitter in existing_emitters),
            emitters=[
                *existing_emitters,
                EmitterOutcomeState(
                    name=self.name,
                    success=True,
                    artifact=str(task_output_path),
                ),
            ],
        )
        write_pipeline_state(data_entry.data_path, pipeline_state)

        return data_entry
