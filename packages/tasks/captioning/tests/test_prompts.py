# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from captioning.prompts import STANDARD_DENSE_VIDEO_PROMPT, STANDARD_IMAGE_PROMPT, load_prompt


def test_load_prompt_falls_back_when_prompt_file_is_empty(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(" \n\t ", encoding="utf-8")

    assert (
        load_prompt(
            prompt_text=None,
            prompt_file=str(prompt_file),
            default_prompt="default prompt",
        )
        == "default prompt"
    )


def test_load_prompt_uses_non_empty_prompt_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(" custom prompt \n", encoding="utf-8")

    assert (
        load_prompt(
            prompt_text=None,
            prompt_file=str(prompt_file),
            default_prompt="default prompt",
        )
        == "custom prompt"
    )


def test_standard_prompts_are_generic() -> None:
    assert (
        STANDARD_DENSE_VIDEO_PROMPT
        == "Elaborate on the visual and narrative elements of the video in detail."
    )
    assert STANDARD_IMAGE_PROMPT == "Describe the image in detail."
