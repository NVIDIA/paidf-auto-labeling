# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard prompt templates and prompt loading helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

STANDARD_DENSE_VIDEO_PROMPT = (
    "Elaborate on the visual and narrative elements of the video in detail."
)

STANDARD_IMAGE_PROMPT = "Describe the image in detail."

STANDARD_SUMMARY_PROMPT = """Summarize the following per-window video captions into one concise,
factual scene-level description. Preserve concrete actions and avoid inventing
events not present in the window captions.
"""


def load_prompt(
    *,
    prompt_text: str | None,
    prompt_file: str | None,
    default_prompt: str,
) -> str:
    """Load a prompt from direct text, then file, then default."""
    if prompt_text and prompt_text.strip():
        return prompt_text.strip()
    if prompt_file and prompt_file.strip():
        contents = Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
        if contents:
            return contents
    return default_prompt.strip()


def prompt_metadata(
    *,
    prompt: str,
    prompt_text: str | None,
    prompt_file: str | None,
    default_name: str,
) -> dict[str, object]:
    """Build reproducibility metadata for a prompt without storing its full text."""
    source = "default"
    source_ref = default_name
    if prompt_text and prompt_text.strip():
        source = "inline"
        source_ref = "cli"
    elif prompt_file and prompt_file.strip():
        source = "file"
        source_ref = prompt_file
    return {
        "source": source,
        "source_ref": source_ref,
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "num_chars": len(prompt),
    }


__all__ = [
    "STANDARD_DENSE_VIDEO_PROMPT",
    "STANDARD_IMAGE_PROMPT",
    "STANDARD_SUMMARY_PROMPT",
    "load_prompt",
    "prompt_metadata",
]
