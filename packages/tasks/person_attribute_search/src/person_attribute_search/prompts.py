# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PAS prompt assets and rendering.

The PAS pipeline drives three model prompts:

* ``pas_attributes`` — VLM attribute extraction (the v3 attribute schema and the
  ``"primary (fine)"`` color convention). Consumed by ``visual_qa`` /
  ``captioning`` via their ``prompt_file`` seam (a standalone instruction, no
  placeholders).
* ``pas_queries`` — LLM difficulty-tiered query generation from a person's
  attributes + caption. Has ``{attributes}`` and ``{caption}`` placeholders.
* ``pas_image_queries`` — per-image VLM caption + hard queries. Has an
  ``{attributes}`` placeholder.

The prompt text lives as plain ``.txt`` assets under ``data/prompts/`` so it can
be pointed at directly by a cookbook's ``prompt_file``. Prompt stems are
unversioned: git is the source of truth for prompt history. Placeholder
substitution uses ``str.replace`` (not ``str.format``) so the literal JSON braces
in the prompts never need escaping.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR: Path = Path(__file__).parent / "data" / "prompts"

ATTRIBUTES_PROMPT = "pas_attributes"
QUERIES_PROMPT = "pas_queries"
QUERIES_HARD_ONLY_PROMPT = "pas_queries_hard_only"
IMAGE_QUERIES_PROMPT = "pas_image_queries"


def prompt_path(name: str) -> Path:
    """Return the absolute path to a named prompt asset (no extension)."""
    if (
        not name
        or Path(name).is_absolute()
        or Path(name).suffix
        or "/" in name
        or "\\" in name
        or ".." in name
    ):
        raise ValueError(f"Invalid PAS prompt asset name: {name!r}")
    prompt_dir = PROMPT_DIR.resolve()
    path = (prompt_dir / f"{name}.txt").resolve()
    if path.parent != prompt_dir:
        raise ValueError(f"Invalid PAS prompt asset name: {name!r}")
    return path


def load_prompt(name: str) -> str:
    """
    Load a named prompt asset's text.

    Args:
        name: Prompt stem, e.g. ``"pas_attributes"`` (use the module
            constants).

    Returns:
        The prompt text, trailing whitespace stripped.

    Raises:
        FileNotFoundError: If the named prompt asset does not exist.
    """
    path = prompt_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"PAS prompt asset not found: {path}")
    return path.read_text(encoding="utf-8").rstrip()


def render_query_prompt(*, attributes: str, caption: str, hard_only: bool = False) -> str:
    """
    Render the LLM query prompt with the person's attributes and caption.

    Args:
        attributes: The person's attribute block (JSON string).
        caption: The person's natural-language caption.
        hard_only: When ``True`` render the hard-only variant (used when medium
            queries are template-generated, mirroring the legacy
            ``PROMPT_V3_HARD_ONLY`` path); otherwise render the medium+hard
            variant (``PROMPT_V3``).

    Returns:
        The rendered prompt text.
    """
    name = QUERIES_HARD_ONLY_PROMPT if hard_only else QUERIES_PROMPT
    template = load_prompt(name)
    return template.replace("{attributes}", attributes).replace("{caption}", caption)


def render_image_query_prompt(*, attributes: str) -> str:
    """Render the per-image VLM prompt with the person's attributes block."""
    template = load_prompt(IMAGE_QUERIES_PROMPT)
    return template.replace("{attributes}", attributes)


__all__ = [
    "ATTRIBUTES_PROMPT",
    "IMAGE_QUERIES_PROMPT",
    "PROMPT_DIR",
    "QUERIES_HARD_ONLY_PROMPT",
    "QUERIES_PROMPT",
    "load_prompt",
    "prompt_path",
    "render_image_query_prompt",
    "render_query_prompt",
]
