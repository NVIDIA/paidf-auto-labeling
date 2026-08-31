# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Filesystem I/O helpers shared across the framework.

Centralizes JSON / JSONL formatting for framework-managed files. DAFT
artifacts rely on the pretty-printed JSON shape (UTF-8, 2-space indent,
trailing newline), while service inputs can reuse the JSONL reader/writer
without duplicating parsing loops.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

JSON_INDENT: int = 2
"""Two-space indentation used for framework-managed JSON artifacts."""


def ensure_dir(path: Path | str) -> Path:
    """
    Create ``path`` (and parents) if missing.

    Args:
        path: Directory path to ensure exists.
    Returns:
        The resolved ``Path`` for ``path``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path | str, payload: Any) -> Path:
    """
    Write ``payload`` to ``path`` as pretty-printed JSON.

    Creates the parent directory if missing. Always writes a trailing newline
    so the file plays nicely with POSIX tooling and pre-commit.

    Args:
        path: Destination file path.
        payload: JSON-serializable object to write.
    Returns:
        The ``Path`` written to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=JSON_INDENT, ensure_ascii=False)
    p.write_text(text + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    """
    Read a JSON document from disk.

    Args:
        path: File path to read.
    Returns:
        The parsed JSON value.
    Raises:
        json.JSONDecodeError: If the file does not parse as JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path | str, records: Iterable[Any]) -> Path:
    """
    Write an iterable of records as JSON Lines.

    Args:
        path: Destination file path.
        records: Iterable of JSON-serializable records.
    Returns:
        The ``Path`` written to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: Path | str) -> Iterator[Any]:
    """
    Yield each record from a JSON Lines file. Skips empty lines.

    Args:
        path: File path to read.
    Yields:
        Each parsed JSON record in the file.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


__all__ = [
    "JSON_INDENT",
    "ensure_dir",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]
