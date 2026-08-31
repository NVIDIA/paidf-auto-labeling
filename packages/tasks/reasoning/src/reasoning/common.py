# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFT task compatibility exports.

DAFT file-contract primitives live in :mod:`core.formats.daft` so other task
packages can share the same version literal, envelope construction,
validation, and atomic write behavior. This module keeps the historical
``reasoning.common`` import surface thin while DAFT-specific converters keep
owning DAFT-specific conversion errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.formats.daft import (
    DAFT_VERSION,
    ContextualType,
    DaftConvertError,
    DaftKind,
    DaftType,
    TaskType,
    daft_envelope,
    metadata_block,
)
from core.formats.daft import (
    write_daft_json as _write_daft_json,
)
from core.scene import SceneContext


def write_daft_json(
    path: Path | str,
    payload: dict[str, Any],
    *,
    expected_type: DaftType | str | None = None,
    kind: DaftKind | None = None,
    overwrite: bool = True,
) -> Path:
    """Validate and atomically write a DAFT payload through core's writer."""
    return _write_daft_json(
        path,
        payload,
        expected_type=expected_type,
        kind=kind,
        overwrite=overwrite,
    )


__all__ = [
    "DAFT_VERSION",
    "ContextualType",
    "DaftConvertError",
    "DaftKind",
    "DaftType",
    "SceneContext",
    "TaskType",
    "daft_envelope",
    "metadata_block",
    "write_daft_json",
]
