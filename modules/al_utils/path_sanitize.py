# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for publishing path-stable artifacts.

Runtime config values often contain host/container absolute paths. Published
artifacts should not preserve developer home directories, so local paths are
rewritten under a stable ``{artifact_root}`` token at write boundaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

ARTIFACT_ROOT_TOKEN = "{artifact_root}"
ARTIFACT_ROOT_ENV = "AUTO_LABELING_ARTIFACT_ROOT"
_REMOTE_SCHEMES = {"msc", "s3", "gs", "ais", "http", "https"}
_COMMON_ARTIFACT_ROOTS = (Path("/workspace"), Path("/app/data"))


def default_artifact_root(repo_root: Path | str) -> Path:
    """Return the artifact root used for ``{artifact_root}`` substitution."""
    env_root = str(os.environ.get(ARTIFACT_ROOT_ENV, "")).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    root = Path(repo_root).expanduser().resolve()
    # Local developer checkouts often live in a monorepo-like parent
    # (e.g. .../pythonic/auto-labeling), while the Docker image uses
    # /workspace directly. Keep both layouts stable.
    return root.parent if root.name == "auto-labeling" else root


def expand_artifact_root_token(value: str, *, repo_root: Path | str) -> str:
    """Expand a leading ``{artifact_root}`` token for runtime path resolution."""
    text = str(value)
    if text == ARTIFACT_ROOT_TOKEN:
        return str(default_artifact_root(repo_root))
    prefix = ARTIFACT_ROOT_TOKEN + "/"
    if text.startswith(prefix):
        rel = text[len(prefix) :]
        artifact_candidate = default_artifact_root(repo_root) / rel
        repo_candidate = Path(repo_root).expanduser().resolve() / rel
        if not artifact_candidate.exists() and repo_candidate.exists():
            return str(repo_candidate)
        return str(artifact_candidate)
    return text


def _is_remote_uri(value: str) -> bool:
    return urlsplit(value).scheme in _REMOTE_SCHEMES


def sanitize_path_string(
    value: str,
    *,
    artifact_root: Path | str,
    extra_roots: Iterable[Path | str] = (),
) -> str:
    """Rewrite one local absolute path as ``{artifact_root}/relative/path``.

    Remote URIs, non-absolute strings, and paths outside the configured roots
    are left unchanged.
    """
    text = str(value)
    if not text or _is_remote_uri(text) or text.startswith(ARTIFACT_ROOT_TOKEN):
        return text

    p = Path(text).expanduser()
    if not p.is_absolute():
        return text

    roots = [Path(artifact_root).expanduser().resolve()]
    roots.extend(Path(root).expanduser().resolve() for root in extra_roots)
    roots.extend(root.resolve() for root in _COMMON_ARTIFACT_ROOTS)

    for root in roots:
        try:
            rel = p.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        return f"{ARTIFACT_ROOT_TOKEN}/{rel.as_posix()}"
    return text


def sanitize_paths_for_publish(
    obj: Any,
    *,
    artifact_root: Path | str,
    extra_roots: Iterable[Path | str] = (),
) -> Any:
    """Recursively sanitize local absolute paths in JSON/YAML-like data."""
    if isinstance(obj, str):
        return sanitize_path_string(obj, artifact_root=artifact_root, extra_roots=extra_roots)
    if isinstance(obj, list):
        return [sanitize_paths_for_publish(v, artifact_root=artifact_root, extra_roots=extra_roots) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_paths_for_publish(v, artifact_root=artifact_root, extra_roots=extra_roots) for v in obj)
    if isinstance(obj, dict):
        return {
            k: sanitize_paths_for_publish(v, artifact_root=artifact_root, extra_roots=extra_roots)
            for k, v in obj.items()
        }
    return obj


__all__ = [
    "ARTIFACT_ROOT_ENV",
    "ARTIFACT_ROOT_TOKEN",
    "default_artifact_root",
    "expand_artifact_root_token",
    "sanitize_path_string",
    "sanitize_paths_for_publish",
]
