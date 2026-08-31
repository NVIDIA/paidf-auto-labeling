# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Identity aggregation.

PAS reasons about a *person identity* that spans many images/crops, whereas the
UPA data model is per-asset (one ``DataEntry`` == one media). This module is the
bridge: it groups flat image records into per-identity groups while preserving
input order, so attribute extraction and query generation can operate per
person.

It is pure (no I/O) and operates on plain records so it can sit either before
ingestion (grouping a dataset manifest) or after detection/tracking (grouping
track crops by track/person id).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PersonKey = tuple[str, str]


class PersonGroup(BaseModel):
    """One person identity and the images/crops that belong to it."""

    model_config = ConfigDict(extra="forbid")

    person_key: PersonKey
    person_id: str
    dataset: str
    images: list[str] = Field(default_factory=list)


def make_person_key(dataset: str, person_id: str) -> PersonKey:
    """Build the stable internal person identity key."""
    return (dataset, person_id)


def person_key_to_string(person_key: PersonKey) -> str:
    """Render an internal key for legacy JSON output only."""
    dataset, person_id = person_key
    return f"{person_id}_{dataset}"


def group_by_identity(
    records: Iterable[Mapping[str, Any]],
    *,
    person_id_key: str = "person_id",
    dataset_key: str = "dataset",
    image_key: str = "image_path",
) -> list[PersonGroup]:
    """
    Group flat image records into per-identity groups, preserving first-seen order.

    Args:
        records: Iterable of mappings, each describing one image with at least a
            person id, dataset, and image path.
        person_id_key: Mapping key holding the person id.
        dataset_key: Mapping key holding the dataset name.
        image_key: Mapping key holding the image path.

    Returns:
        A list of ``PersonGroup`` objects in first-seen identity order, each with
        de-duplicated images in first-seen order.

    Raises:
        ValueError: If a record is missing the person id or dataset.
    """
    groups: dict[PersonKey, PersonGroup] = {}
    seen_images: dict[PersonKey, set[str]] = {}

    for index, record in enumerate(records):
        person_id = str(record.get(person_id_key) or "").strip()
        dataset = str(record.get(dataset_key) or "").strip()
        if not person_id or not dataset:
            raise ValueError(
                f"record[{index}] missing {person_id_key!r} or {dataset_key!r}: {record!r}"
            )
        person_key = make_person_key(dataset, person_id)
        if person_key not in groups:
            groups[person_key] = PersonGroup(
                person_key=person_key,
                person_id=person_id,
                dataset=dataset,
                images=[],
            )
            seen_images[person_key] = set()

        image_path = str(record.get(image_key) or "").strip()
        if image_path and image_path not in seen_images[person_key]:
            seen_images[person_key].add(image_path)
            groups[person_key].images.append(image_path)

    return list(groups.values())


__all__ = [
    "PersonGroup",
    "PersonKey",
    "group_by_identity",
    "make_person_key",
    "person_key_to_string",
]
