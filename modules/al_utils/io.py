# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic file I/O helpers used across the pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from al_utils.path_sanitize import default_artifact_root, sanitize_paths_for_publish


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_json(
    path: Path,
    obj: Any,
    *,
    sanitize_paths: bool = False,
    artifact_root: Path | str | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        sanitize_paths_for_publish(
            obj,
            artifact_root=artifact_root or default_artifact_root(Path.cwd()),
            extra_roots=(),
        )
        if sanitize_paths
        else obj
    )
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


__all__ = ["read_text", "read_json", "write_text", "write_json", "sha256_text"]
