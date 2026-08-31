# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from core import (
    AnnotationExportState,
    EmitterOutcomeState,
    ScenePipelineState,
    read_pipeline_state,
    write_pipeline_state,
)
from core.models import DataEntry
from example_task.task import (
    EXAMPLE_TASK_FILENAME,
    EXAMPLE_TASK_MANIFEST_FILENAME,
    SimpleTask,
)


def test_simple_task_publishes_example_data(tmp_path: Path) -> None:
    simple_task = SimpleTask()
    data_entry = DataEntry(media_path="path/to/media/file.mp4", data_path=str(tmp_path))
    write_pipeline_state(
        tmp_path,
        ScenePipelineState(
            annotation_export=AnnotationExportState(
                success=True,
                emitters=[
                    EmitterOutcomeState(
                        name="already_published",
                        success=True,
                        artifact="/tmp/already.json",
                    )
                ],
            )
        ),
    )
    annotated_data_entry = simple_task.run(data_entry)

    assert annotated_data_entry is not None
    assert annotated_data_entry.media_path == "path/to/media/file.mp4"
    assert annotated_data_entry.data_path == str(tmp_path)

    task_output_path = tmp_path / "sidecars" / EXAMPLE_TASK_FILENAME
    payload = json.loads(task_output_path.read_text(encoding="utf-8"))
    assert payload["video_id"] == "file"
    assert payload["metadata"]["type"] == "example_task"
    assert payload["data_entry_id"] == data_entry.id
    assert payload["items"][0]["label"] == "probably-a-thing"

    manifest_path = tmp_path / "sidecars" / EXAMPLE_TASK_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["published_artifacts"] == [f"sidecars/{EXAMPLE_TASK_FILENAME}"]

    pipeline_state = read_pipeline_state(tmp_path)
    assert pipeline_state.annotation_export is not None
    assert pipeline_state.annotation_export.success is True
    emitters = {
        emitter.name: emitter.artifact for emitter in pipeline_state.annotation_export.emitters
    }
    assert emitters == {
        "already_published": "/tmp/already.json",
        simple_task.name: str(task_output_path),
    }
