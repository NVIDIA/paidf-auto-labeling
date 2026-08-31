# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFT write API for contextual and task annotations."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from core.formats.daft.envelope import DAFT_VERSION
from core.formats.daft.types import (
    CONTEXTUAL_DEFAULT_FILENAMES,
    TASK_DEFAULT_FILENAMES,
    ContextualType,
    DaftKind,
    DaftType,
    TaskType,
    daft_kind,
)
from core.scene import SceneContext, ScenePaths, ensure_scene_skeleton
from core.utils.io import JSON_INDENT


class DaftWriteError(ValueError):
    """Raised when a DAFT payload cannot be safely written."""


class DaftValidationError(DaftWriteError):
    """Raised when structural or schema validation fails before writing."""


class DaftPayloadValidator(Protocol):
    """Protocol for pluggable DAFT payload validators."""

    def validate_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_type: str | None = None,
        kind: DaftKind | None = None,
        path: Path | None = None,
    ) -> Sequence[str]:
        """Return validation error messages for ``payload``."""
        ...


@dataclass(frozen=True)
class DaftWriteResult:
    """Result of one DAFT payload write."""

    path: Path
    type_name: str
    kind: DaftKind


class StructuralDaftValidator:
    """Fast structural validator used before every DAFT write.

    This intentionally does not replace the full ``nvidia-tao-daft`` validator.
    It catches writer/API mistakes early and can be composed with
    ``NvidiaTaoDaftPayloadValidator`` for schema checks.
    """

    def validate_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_type: str | None = None,
        kind: DaftKind | None = None,
        path: Path | None = None,
    ) -> Sequence[str]:
        location = f" ({path})" if path is not None else ""
        errors: list[str] = []
        if payload.get("version") != DAFT_VERSION:
            got = payload.get("version")
            errors.append(f"DAFT payload version must be {DAFT_VERSION!r}, got {got!r}{location}")

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"DAFT payload metadata must be an object{location}")
            return errors

        type_value = metadata.get("type")
        if not isinstance(type_value, str) or not type_value:
            errors.append(f"DAFT payload metadata.type must be a non-empty string{location}")
            return errors

        if expected_type is not None and type_value != expected_type:
            errors.append(
                f"DAFT payload metadata.type must be {expected_type!r}, "
                f"got {type_value!r}{location}"
            )

        try:
            actual_kind = daft_kind(type_value)
        except ValueError as exc:
            errors.append(f"{exc}{location}")
            return errors

        if kind is not None and actual_kind != kind:
            errors.append(
                f"DAFT payload type {type_value!r} belongs in {actual_kind}/, not {kind}/{location}"
            )

        if actual_kind == "task":
            items = payload.get("items")
            if not isinstance(items, list) or not items:
                errors.append(
                    f"DAFT task payload {type_value!r} requires a non-empty items list{location}"
                )

        return errors


class CompositeDaftValidator:
    """Run multiple DAFT validators and concatenate their errors."""

    def __init__(self, validators: Iterable[DaftPayloadValidator]) -> None:
        self._validators = tuple(validators)

    def validate_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_type: str | None = None,
        kind: DaftKind | None = None,
        path: Path | None = None,
    ) -> Sequence[str]:
        errors: list[str] = []
        for validator in self._validators:
            errors.extend(
                validator.validate_payload(
                    payload,
                    expected_type=expected_type,
                    kind=kind,
                    path=path,
                )
            )
        return errors


class NvidiaTaoDaftPayloadValidator:
    """Validate a single payload with the optional ``nvidia-tao-daft`` package."""

    def __init__(self) -> None:
        try:
            module = importlib.import_module("nvidia_tao_daft.validators.metropolis_v3_0.validator")
            validator_cls = module.MetropolisV3_0Validator
            self._validator: Any = validator_cls()
        except Exception as exc:
            raise DaftValidationError(
                "nvidia-tao-daft is not importable; install it or inject another validator"
            ) from exc

    def validate_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_type: str | None = None,
        kind: DaftKind | None = None,
        path: Path | None = None,
    ) -> Sequence[str]:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("type"), str):
            return ["metadata.type is required before schema validation"]

        type_name = metadata["type"]
        try:
            actual_kind = daft_kind(type_name)
        except ValueError as exc:
            return [str(exc)]

        if kind is not None and actual_kind != kind:
            return [f"metadata.type {type_name!r} belongs in {actual_kind}/, not {kind}/"]
        if expected_type is not None and type_name != expected_type:
            return [f"metadata.type {type_name!r} does not match expected {expected_type!r}"]

        schema_map_name = "_CONTEXTUAL_SCHEMAS" if actual_kind == "contextual" else "_TASK_SCHEMAS"
        schema_map = cast(Mapping[str, str], getattr(self._validator, schema_map_name))
        schema_name = schema_map.get(type_name)
        if schema_name is None:
            return [f"No nvidia-tao-daft schema registered for metadata.type {type_name!r}"]

        errors = self._validator._validate_data(payload, schema_name)
        location = f"{path.name}: " if path is not None else ""
        return [f"{location}{error}" for error in errors]


class JsonSchemaDaftPayloadValidator:
    """Validate DAFT payloads against a metropolis-v3.0 schema directory."""

    _CONTEXTUAL_SCHEMAS: dict[str, str] = {
        "calibration": "contextual/calibration.schema.json",
        "chunks": "contextual/chunks.schema.json",
        "events": "contextual/events.schema.json",
        "image": "contextual/image.schema.json",
        "instances": "contextual/instances.schema.json",
        "msted": "contextual/msted.schema.json",
        "objects": "contextual/objects.schema.json",
        "pas_queries": "contextual/pas_queries.schema.json",
        "person_attributes": "contextual/person_attributes.schema.json",
        "tracking": "contextual/tracking.schema.json",
        "video": "contextual/video.schema.json",
    }
    _TASK_SCHEMAS: dict[str, str] = {
        "bcq": "tasks/bcq.schema.json",
        "bcq_openended": "tasks/bcq_openended.schema.json",
        "causal_linkage": "tasks/causal_linkage.schema.json",
        "mcq": "tasks/mcq.schema.json",
        "mcq_openended": "tasks/mcq_openended.schema.json",
        "open_qa": "tasks/open_qa.schema.json",
        "scene_description": "tasks/scene_description.schema.json",
        "temporal_description": "tasks/temporal_description.schema.json",
        "temporal_localization": "tasks/temporal_localization.schema.json",
        "video_summarization": "tasks/video_summarization.schema.json",
    }

    def __init__(self, schema_root: Path | str) -> None:
        self.schema_root = Path(schema_root)
        self._validators: dict[str, Any] = {}
        try:
            jsonschema = importlib.import_module("jsonschema")
            self._draft7_validator_cls: Any = jsonschema.Draft7Validator
        except Exception as exc:
            raise DaftValidationError("jsonschema is required for schema validation") from exc

    def validate_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_type: str | None = None,
        kind: DaftKind | None = None,
        path: Path | None = None,
    ) -> Sequence[str]:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("type"), str):
            return ["metadata.type is required before schema validation"]

        type_name = metadata["type"]
        if expected_type is not None and type_name != expected_type:
            return [f"metadata.type {type_name!r} does not match expected {expected_type!r}"]
        try:
            actual_kind = daft_kind(type_name)
        except ValueError as exc:
            return [str(exc)]
        if kind is not None and actual_kind != kind:
            return [f"metadata.type {type_name!r} belongs in {actual_kind}/, not {kind}/"]

        schema_name = self._schema_name(type_name, actual_kind)
        validator = self._validator_for(schema_name)
        errors: list[str] = []
        for error in validator.iter_errors(payload):
            field_path = ".".join(str(part) for part in error.absolute_path)
            message = f"[{field_path}] {error.message}" if field_path else error.message
            if path is not None:
                message = f"{path.name}: {message}"
            errors.append(message)
        return errors

    def _schema_name(self, type_name: str, kind: DaftKind) -> str:
        schema_map = self._CONTEXTUAL_SCHEMAS if kind == "contextual" else self._TASK_SCHEMAS
        schema_name = schema_map.get(type_name)
        if schema_name is None:
            raise DaftValidationError(f"No schema registered for metadata.type {type_name!r}")
        return schema_name

    def _validator_for(self, schema_name: str) -> Any:
        cached = self._validators.get(schema_name)
        if cached is not None:
            return cached
        schema_path = self.schema_root / schema_name
        if not schema_path.exists():
            raise DaftValidationError(f"DAFT schema not found: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = self._draft7_validator_cls(schema)
        self._validators[schema_name] = validator
        return validator


class DaftSceneWriter:
    """Route, validate and atomically write DAFT annotations for one scene."""

    def __init__(
        self,
        scene_dir: Path | str,
        *,
        ctx: SceneContext | None = None,
        validator: DaftPayloadValidator | None = None,
    ) -> None:
        self.paths: ScenePaths = ensure_scene_skeleton(scene_dir)
        self.ctx = ctx
        self.validator = validator or StructuralDaftValidator()

    def target_path(self, type_name: DaftType | str, *, filename: str | None = None) -> Path:
        """Return the canonical path for ``type_name`` under this scene."""
        kind = daft_kind(type_name)
        name = _safe_filename(filename or _default_filename(type_name, kind))
        if kind == "contextual":
            return self.paths.contextual_dir / name
        return self.paths.task_dir / name

    def write_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_type: DaftType | str | None = None,
        filename: str | None = None,
        overwrite: bool = True,
    ) -> DaftWriteResult:
        """Validate and write one DAFT payload to the directory implied by ``metadata.type``."""
        type_name = _metadata_type(payload)
        if expected_type is not None and type_name != expected_type:
            raise DaftWriteError(
                f"payload metadata.type {type_name!r} does not match expected {expected_type!r}"
            )
        kind = daft_kind(type_name)
        path = self.target_path(type_name, filename=filename)
        _write_daft_json(
            path,
            payload,
            validator=self.validator,
            expected_type=type_name,
            kind=kind,
            overwrite=overwrite,
        )
        return DaftWriteResult(path=path, type_name=type_name, kind=kind)

    def write_contextual(
        self,
        type_name: ContextualType,
        payload: dict[str, Any],
        *,
        filename: str | None = None,
        overwrite: bool = True,
    ) -> DaftWriteResult:
        """Write one contextual annotation payload."""
        if daft_kind(type_name) != "contextual":
            raise DaftWriteError(f"{type_name!r} is not a contextual DAFT type")
        result = self.write_payload(
            payload,
            expected_type=type_name,
            filename=filename,
            overwrite=overwrite,
        )
        if result.kind != "contextual":
            raise DaftWriteError(f"{type_name!r} is not a contextual DAFT type")
        return result

    def write_task(
        self,
        type_name: TaskType,
        payload: dict[str, Any],
        *,
        filename: str | None = None,
        overwrite: bool = True,
    ) -> DaftWriteResult:
        """Write one task annotation payload."""
        if daft_kind(type_name) != "task":
            raise DaftWriteError(f"{type_name!r} is not a task DAFT type")
        result = self.write_payload(
            payload,
            expected_type=type_name,
            filename=filename,
            overwrite=overwrite,
        )
        if result.kind != "task":
            raise DaftWriteError(f"{type_name!r} is not a task DAFT type")
        return result


def write_daft_json(
    path: Path | str,
    payload: dict[str, Any],
    *,
    validator: DaftPayloadValidator | None = None,
    expected_type: DaftType | str | None = None,
    kind: DaftKind | None = None,
    overwrite: bool = True,
) -> Path:
    """Validate and atomically write a DAFT payload to an explicit path."""
    type_name = _metadata_type(payload)
    actual_kind = daft_kind(type_name)
    _write_daft_json(
        Path(path),
        payload,
        validator=validator or StructuralDaftValidator(),
        expected_type=expected_type or type_name,
        kind=kind or actual_kind,
        overwrite=overwrite,
    )
    return Path(path)


def _write_daft_json(
    path: Path,
    payload: dict[str, Any],
    *,
    validator: DaftPayloadValidator,
    expected_type: str | None,
    kind: DaftKind | None,
    overwrite: bool,
) -> None:
    errors = validator.validate_payload(payload, expected_type=expected_type, kind=kind, path=path)
    if errors:
        joined = "; ".join(errors)
        raise DaftValidationError(f"DAFT validation failed for {path}: {joined}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = (json.dumps(payload, indent=JSON_INDENT, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    if not overwrite:
        _write_exclusive(path, payload_bytes)
        return

    tmp = _write_unique_temp(path, payload_bytes)
    try:
        tmp.replace(path)
    except OSError as exc:
        raise DaftWriteError(f"Failed to write DAFT file {path}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _write_unique_temp(path: Path, payload_bytes: bytes) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as file:
            fd = -1
            file.write(payload_bytes)
            file.flush()
            os.fsync(file.fileno())
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise DaftWriteError(f"Failed to write DAFT temp file {tmp}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    return tmp


def _write_exclusive(path: Path, payload_bytes: bytes) -> None:
    fd = -1
    created = False
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        created = True
        with os.fdopen(fd, "wb") as file:
            fd = -1
            file.write(payload_bytes)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError as exc:
        raise DaftWriteError(f"Refusing to overwrite existing DAFT file: {path}") from exc
    except OSError as exc:
        if created:
            path.unlink(missing_ok=True)
        raise DaftWriteError(f"Failed to write DAFT file {path}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _metadata_type(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise DaftWriteError(f"DAFT payload must be a dict, got {type(payload).__name__}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise DaftWriteError("DAFT payload missing metadata object")
    type_name = metadata.get("type")
    if not isinstance(type_name, str) or not type_name:
        raise DaftWriteError("DAFT payload metadata.type must be a non-empty string")
    return type_name


def _default_filename(type_name: str, kind: DaftKind) -> str:
    if kind == "contextual":
        return CONTEXTUAL_DEFAULT_FILENAMES[cast(ContextualType, type_name)]
    return TASK_DEFAULT_FILENAMES[cast(TaskType, type_name)]


def _safe_filename(filename: str) -> str:
    path = Path(filename)
    if path.name != filename or path.is_absolute():
        raise DaftWriteError(f"DAFT filename must be a plain file name, got {filename!r}")
    if not filename.endswith(".json"):
        raise DaftWriteError(f"DAFT filename must end with .json, got {filename!r}")
    return filename


__all__ = [
    "CompositeDaftValidator",
    "DaftPayloadValidator",
    "DaftSceneWriter",
    "DaftValidationError",
    "DaftWriteError",
    "DaftWriteResult",
    "JsonSchemaDaftPayloadValidator",
    "NvidiaTaoDaftPayloadValidator",
    "StructuralDaftValidator",
    "write_daft_json",
]
