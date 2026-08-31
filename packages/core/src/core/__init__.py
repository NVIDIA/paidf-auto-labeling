# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core abstractions for the Auto-Labeling framework.

Re-exports the most frequently used public symbols so downstream task and
service packages can import from ``core`` directly. Submodules remain
importable individually (e.g. ``from core.scene import ScenePaths``).
"""

from core.cost_performance import (
    COST_PERFORMANCE_REPORT_SIDECAR,
    ModelCallRecord,
    ModelUsageSnapshot,
    TaskRunReport,
    collect_model_usage,
    model_usage_entry,
    normalize_token_counts,
    record_model_call,
    write_cost_performance_report,
)
from core.models import (
    PIPELINE_STATE_FILENAME,
    PIPELINE_STATE_SCHEMA_VERSION,
    AnnotationExportState,
    DataEntry,
    EmitterOutcomeState,
    EnhancedMediaState,
    ScenePipelineState,
    pipeline_state_path,
    read_pipeline_state,
    update_annotation_export_state,
    write_pipeline_state,
)
from core.pipelines import LinearPipeline
from core.policy import (
    EmptyOutputPolicy,
    StageOutcome,
    StagePolicyError,
    coerce_policy,
)
from core.scene import (
    ACTIVE_MEDIA_GLOB,
    CONTEXTUAL_DIRNAME,
    RAW_DIRNAME,
    RAW_MEDIA_BASENAME,
    SIDECARS_DIRNAME,
    TASK_DIRNAME,
    SceneContext,
    ScenePaths,
    active_media_path_for_input,
    ensure_scene_skeleton,
    find_active_media_path,
    is_image_path,
    raw_analyzed_media_name,
    raw_media_path,
    resolve_sidecar,
    scene_context_for_entry,
    scene_media_id_from_path,
    stage_raw_media,
)
from core.utils.io import (
    ensure_dir,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

__all__ = [
    "PIPELINE_STATE_FILENAME",
    "PIPELINE_STATE_SCHEMA_VERSION",
    "AnnotationExportState",
    "COST_PERFORMANCE_REPORT_SIDECAR",
    "DataEntry",
    "EmitterOutcomeState",
    "EmptyOutputPolicy",
    "EnhancedMediaState",
    "ACTIVE_MEDIA_GLOB",
    "CONTEXTUAL_DIRNAME",
    "RAW_MEDIA_BASENAME",
    "RAW_DIRNAME",
    "SIDECARS_DIRNAME",
    "TASK_DIRNAME",
    "LinearPipeline",
    "ModelCallRecord",
    "ModelUsageSnapshot",
    "SceneContext",
    "ScenePipelineState",
    "ScenePaths",
    "StageOutcome",
    "StagePolicyError",
    "TaskRunReport",
    "active_media_path_for_input",
    "coerce_policy",
    "collect_model_usage",
    "ensure_dir",
    "ensure_scene_skeleton",
    "find_active_media_path",
    "is_image_path",
    "model_usage_entry",
    "normalize_token_counts",
    "pipeline_state_path",
    "read_json",
    "read_jsonl",
    "read_pipeline_state",
    "record_model_call",
    "raw_analyzed_media_name",
    "raw_media_path",
    "resolve_sidecar",
    "scene_context_for_entry",
    "scene_media_id_from_path",
    "stage_raw_media",
    "update_annotation_export_state",
    "write_json",
    "write_jsonl",
    "write_cost_performance_report",
    "write_pipeline_state",
]
