# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for migrated 2D grounding stages."""

from __future__ import annotations

from typing import Any

EXPRESSION_PROMPT_TEMPLATE = """\
You are analyzing a scene image and its caption to extract referring expressions
for a box/mask detector (text → region). Do not assume any particular domain
(traffic, retail, indoor, etc.); judge only from this image and caption.

Caption: "{caption}"

Tasks:
1. Clean the caption — remove speech artifacts such as "we can see", "there is/are",
   "you can see", "in the image", "in this photo", "I can see". Rephrase into natural
   written English.
2. Extract referring expressions — short noun phrases for whole visible objects or
   small groups. Add a brief spatial cue only when it disambiguates.
3. For each expression provide its exact character span in the cleaned caption, the
   core noun, and whether a detector can box it.

Rules:
- Only include expressions referring to something visible in the image.
- Prefer whole countable objects, not object parts, ambience, or scene layout.
- Use the most specific object type supported by what you see in the image.
- Avoid abstract or uncountable nouns unless they name a specific countable thing.
- Each expression must be a noun phrase, not a full sentence.
- Keep each expression short (about 6 words or fewer).
- Cap the list at the ~12 most important countable objects.
- Set "groundable" to true only if a single tight box around that object is sensible.
  Set "groundable" to false for parts, pure background, abstract events, or anything
  a detector should skip.

Respond ONLY with a JSON object — no markdown, no explanation:
{{
  "cleaned_caption": "...",
  "expressions": [
    {{"text": "...", "char_span": [start, end], "noun_chunk": "...", "groundable": true}},
    ...
  ]
}}"""

GROUNDING_PROMPT_TEMPLATE = """\
You are a visual grounding model. Locate visible instances of each referring expression below.

Referring expressions:
{expressions_block}

Rules:
- Coordinates are pixel-space: [x1, y1, x2, y2].
- (x1,y1) is top-left and (x2,y2) is bottom-right.
- Return at most {max_instances} instances per expression.
- Estimate a confidence score from 0.0 to 1.0 for each bbox.
- If an expression matches nothing visible, use empty lists.

Respond only with a valid JSON object:
{{
  "<expression_text>": {{"bboxes": [[x1,y1,x2,y2]], "scores": [0.9]}}
}}"""


def _escape_prompt_value(value: str) -> str:
    """Escape dynamic values so quotes/newlines do not break prompt structure."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def expression_prompt(caption: str) -> str:
    """Build the expression-extraction prompt."""
    return EXPRESSION_PROMPT_TEMPLATE.format(caption=_escape_prompt_value(caption))


def grounding_prompt(expressions: list[dict[str, Any]], *, max_instances: int) -> str:
    """Build the phrase-grounding prompt."""
    lines = "\n".join(f'  - "{_escape_prompt_value(str(expr["text"]))}"' for expr in expressions)
    return GROUNDING_PROMPT_TEMPLATE.format(
        expressions_block=lines,
        max_instances=max_instances,
    )


__all__ = ["expression_prompt", "grounding_prompt"]
