# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HITL data-factory preannotation export.

Converts an assembled ``QuerySet`` plus the person/image natural caption into
the human-in-the-loop preannotation schema consumed by the
``person-attribute-search-data-augmentation`` data-factory team
(``2 Easy + 2 Medium + 1 Hard + 1 Natural`` per image). Pure dict construction,
no filesystem or S3 side effects.
"""

from __future__ import annotations

from person_attribute_search.queries import QuerySet

_DEFAULT_EASY = 2
_DEFAULT_MEDIUM = 2
_DEFAULT_HARD = 1


def build_preannotations(
    query_set: QuerySet,
    natural_caption: str | None,
    *,
    easy: int = _DEFAULT_EASY,
    medium: int = _DEFAULT_MEDIUM,
    hard: int = _DEFAULT_HARD,
) -> list[dict[str, str]]:
    """
    Build the HITL ``preannotations`` list for a single image.

    Args:
        query_set: Assembled easy/medium/hard queries for the image.
        natural_caption: The image's natural-language caption (may be empty).
        easy: Number of Easy captions to include.
        medium: Number of Medium captions to include.
        hard: Number of Hard captions to include.

    Returns:
        Preannotation dicts, each ``{caption_type, caption, attribute_type}``.
        Natural is appended only when a non-empty caption is available.
    """
    easy = max(0, easy)
    medium = max(0, medium)
    hard = max(0, hard)
    preannotations: list[dict[str, str]] = []

    for query, used in query_set.easy[:easy]:
        preannotations.append({"caption_type": "Easy", "caption": query, "attribute_type": used})
    for query, used in query_set.medium[:medium]:
        preannotations.append({"caption_type": "Medium", "caption": query, "attribute_type": used})
    for query in query_set.hard[:hard]:
        preannotations.append({"caption_type": "Hard", "caption": query, "attribute_type": ""})

    caption = (natural_caption or "").strip()
    if caption:
        preannotations.append({"caption_type": "Natural", "caption": caption, "attribute_type": ""})

    return preannotations


def build_preannotation_file(
    image_url: str,
    query_set: QuerySet,
    natural_caption: str | None,
) -> dict[str, object]:
    """
    Build a complete per-image HITL preannotation document.

    Args:
        image_url: Public (S3) URL the annotator tool will render.
        query_set: Assembled queries for the image.
        natural_caption: Natural-language caption for the image.

    Returns:
        ``{"image_url": ..., "preannotations": [...]}``.
    """
    return {
        "image_url": image_url,
        "preannotations": build_preannotations(query_set, natural_caption),
    }


__all__ = ["build_preannotation_file", "build_preannotations"]
