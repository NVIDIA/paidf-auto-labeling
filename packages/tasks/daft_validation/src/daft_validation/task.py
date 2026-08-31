# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural DAFT scene validation task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from core import DataEntry, SceneContext, ensure_scene_skeleton, scene_context_for_entry
from core.formats.daft import (
    CONTEXTUAL_DEFAULT_FILENAMES,
    PERSONA_QA_FILENAME,
    TASK_DEFAULT_FILENAMES,
    DaftKind,
    DaftPayloadValidator,
    StructuralDaftValidator,
)
from core.tasks import SequentialTask
from pydantic import BaseModel, ConfigDict

# PAS deliverable filenames (owned by person_attribute_search). Kept as local
# string constants so daft_validation does not depend on the PAS task package.
_PAS_FILENAME = "pas.json"
_PAS_ANOMALY_FILENAME = "pas_anomaly.json"

# Current pipelines write persona_qa.json under sidecars/visual_qa/, but scenes
# produced by older images may still carry it under task/. Tolerate it there for
# backward-compatible re-runs rather than flagging it as an unknown DAFT filename.
_IGNORED_TASK_FILENAMES: frozenset[str] = frozenset({PERSONA_QA_FILENAME})


class DaftValidationConfig(BaseModel):
    """Configuration for DAFT validation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class DaftSceneValidationError(ValueError):
    """Raised when a DAFT scene fails structural validation."""

    def __init__(self, scene_dir: Path | str, issues: list[str]) -> None:
        self.scene_dir = Path(scene_dir)
        self.issues = tuple(issues)
        joined = "; ".join(issues)
        super().__init__(f"DAFT validation failed for {self.scene_dir}: {joined}")


class DaftValidationTask(SequentialTask):
    """Validate canonical DAFT JSON files in a scene directory."""

    def __init__(
        self,
        *,
        config: DaftValidationConfig | None = None,
        validator: DaftPayloadValidator | None = None,
        name: str | None = None,
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name or "daft_validation", max_retries=max_retries)
        self.config = config or DaftValidationConfig()
        self.validator = validator or StructuralDaftValidator()

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        if not self.config.enabled:
            self.logger.info("DAFT validation disabled; skipping.")
            return data_entry

        issues = validate_scene(data_entry, validator=self.validator)
        if issues:
            raise DaftSceneValidationError(data_entry.data_path, issues)
        self.logger.info("DAFT validation passed for %s.", data_entry.data_path)
        return data_entry


def validate_scene(
    data_entry: DataEntry,
    *,
    validator: DaftPayloadValidator | None = None,
) -> list[str]:
    """Validate structural DAFT contracts for a scene and return all issues."""
    paths = ensure_scene_skeleton(data_entry.data_path)
    scene_ctx = scene_context_for_entry(data_entry)
    payload_validator = validator or StructuralDaftValidator()
    issues: list[str] = []
    _validate_directory(
        paths.contextual_dir,
        kind="contextual",
        filename_to_type=_CONTEXTUAL_TYPE_BY_FILENAME,
        validator=payload_validator,
        scene_ctx=scene_ctx,
        issues=issues,
        ignored_filenames=_IGNORED_CONTEXTUAL_FILENAMES,
    )
    _validate_directory(
        paths.task_dir,
        kind="task",
        filename_to_type=_TASK_TYPE_BY_FILENAME,
        validator=payload_validator,
        scene_ctx=scene_ctx,
        issues=issues,
        ignored_filenames=_IGNORED_TASK_FILENAMES,
    )
    return issues


def _validate_directory(
    root: Path,
    *,
    kind: DaftKind,
    filename_to_type: dict[str, str],
    validator: DaftPayloadValidator,
    scene_ctx: SceneContext,
    issues: list[str],
    ignored_filenames: frozenset[str] = frozenset(),
) -> None:
    for path in sorted(root.glob("*.json")):
        if path.name in ignored_filenames:
            continue
        expected_type = filename_to_type.get(path.name)
        if expected_type is None:
            issues.append(
                f"{path}: unknown DAFT annotation filename; task-internal artifacts belong "
                "under sidecars/"
            )
            continue

        payload = _read_json_object(path, issues)
        if payload is None:
            continue

        for error in validator.validate_payload(
            payload,
            expected_type=expected_type,
            kind=kind,
            path=path,
        ):
            issues.append(str(error))
        if kind == "task":
            _validate_task_item_media_contract(path, payload=payload, ctx=scene_ctx, issues=issues)
        _validate_media_contract(
            path,
            expected_type=expected_type,
            payload=payload,
            ctx=scene_ctx,
            issues=issues,
        )


def _validate_task_item_media_contract(
    path: Path,
    *,
    payload: dict[str, Any],
    ctx: SceneContext,
    issues: list[str],
) -> None:
    items = payload.get("items")
    if not isinstance(items, list):
        return

    field = ctx.scene_id_field
    other = "video_id" if ctx.is_image else "image_id"
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item.get(field) != ctx.media_id:
            issues.append(f"{path}: items[{index}].{field} expected {ctx.media_id!r}")
        if other in item:
            issues.append(f"{path}: items[{index}] must not contain {other}")


def _read_json_object(path: Path, issues: list[str]) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        issues.append(f"{path}: could not read JSON file: {exc}")
        return None
    except UnicodeDecodeError as exc:
        issues.append(f"{path}: could not decode JSON file as UTF-8: {exc}")
        return None
    except json.JSONDecodeError as exc:
        issues.append(f"{path}: invalid JSON: {exc}")
        return None
    if not isinstance(loaded, dict):
        issues.append(f"{path}: DAFT payload must be a JSON object, got {type(loaded).__name__}")
        return None
    return loaded


def _validate_media_contract(
    path: Path,
    *,
    expected_type: str,
    payload: dict[str, Any],
    ctx: SceneContext,
    issues: list[str],
) -> None:
    if expected_type == "video":
        if ctx.is_image:
            issues.append(f"{path}: media_path is an image but contextual/video.json is present")
        _append_id_mismatch(
            path,
            payload=payload,
            field="video_id",
            expected=ctx.media_id,
            issues=issues,
        )
    if expected_type == "image":
        if not ctx.is_image:
            issues.append(f"{path}: media_path is a video but contextual/image.json is present")
        _append_id_mismatch(
            path,
            payload=payload,
            field="image_id",
            expected=ctx.media_id,
            issues=issues,
        )


def _append_id_mismatch(
    path: Path,
    *,
    payload: dict[str, Any],
    field: str,
    expected: str,
    issues: list[str],
) -> None:
    value = payload.get(field)
    if isinstance(value, str) and value.strip() and value.strip() != expected:
        issues.append(f"{path}: {field}={value!r}; expected {expected!r}")


# Current pipelines write the PAS deliverable under
# sidecars/person_attribute_search/, but scenes produced by older images may
# still carry pas.json / pas_anomaly.json under contextual/. These are not DAFT
# contextual annotations; tolerate them there for backward-compatible re-runs.
_IGNORED_CONTEXTUAL_FILENAMES: frozenset[str] = frozenset({_PAS_FILENAME, _PAS_ANOMALY_FILENAME})

_CONTEXTUAL_TYPE_BY_FILENAME: dict[str, str] = {
    filename: type_name for type_name, filename in CONTEXTUAL_DEFAULT_FILENAMES.items()
}
_TASK_TYPE_BY_FILENAME: dict[str, str] = {
    filename: type_name for type_name, filename in TASK_DEFAULT_FILENAMES.items()
}


__all__ = [
    "DaftSceneValidationError",
    "DaftValidationConfig",
    "DaftValidationTask",
    "validate_scene",
]
