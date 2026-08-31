# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFT file-contract primitives shared across task packages.

This package is intentionally limited to reusable DAFT infrastructure:
envelope construction, canonical type routing, deterministic service pivots,
validation, and atomic writes. Task-specific prompt logic and model execution
belong in task packages.
"""

from core.formats.daft.envelope import DAFT_VERSION, daft_envelope, metadata_block
from core.formats.daft.errors import DaftConvertError
from core.formats.daft.pivots import (
    CAPTION_ARTIFACTS_KEY,
    PERSONA_QA_FILENAME,
    VISUAL_QA_ARTIFACTS_KEY,
    emit_captioning_daft_outputs,
    emit_persona_qa_daft_outputs,
    emit_visual_qa_daft_outputs,
)
from core.formats.daft.types import (
    CONTEXTUAL_DEFAULT_FILENAMES,
    CONTEXTUAL_TYPES,
    TASK_DEFAULT_FILENAMES,
    TASK_TYPES,
    ContextualType,
    DaftKind,
    DaftType,
    TaskType,
    daft_kind,
)
from core.formats.daft.writer import (
    CompositeDaftValidator,
    DaftPayloadValidator,
    DaftSceneWriter,
    DaftValidationError,
    DaftWriteError,
    DaftWriteResult,
    JsonSchemaDaftPayloadValidator,
    NvidiaTaoDaftPayloadValidator,
    StructuralDaftValidator,
    write_daft_json,
)

__all__ = [
    "CONTEXTUAL_DEFAULT_FILENAMES",
    "CONTEXTUAL_TYPES",
    "CAPTION_ARTIFACTS_KEY",
    "DAFT_VERSION",
    "TASK_DEFAULT_FILENAMES",
    "TASK_TYPES",
    "CompositeDaftValidator",
    "ContextualType",
    "DaftKind",
    "DaftConvertError",
    "DaftPayloadValidator",
    "DaftSceneWriter",
    "DaftType",
    "DaftValidationError",
    "DaftWriteError",
    "DaftWriteResult",
    "JsonSchemaDaftPayloadValidator",
    "NvidiaTaoDaftPayloadValidator",
    "PERSONA_QA_FILENAME",
    "StructuralDaftValidator",
    "TaskType",
    "VISUAL_QA_ARTIFACTS_KEY",
    "daft_envelope",
    "daft_kind",
    "emit_captioning_daft_outputs",
    "emit_persona_qa_daft_outputs",
    "emit_visual_qa_daft_outputs",
    "metadata_block",
    "write_daft_json",
]
