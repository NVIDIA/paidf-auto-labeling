# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Structured person-attribute domain model (PAS "v3" schema).

This module owns the *vocabulary* and the typed representation of a person's
attributes. It is intentionally free of I/O, model calls, and framework
imports so it can be unit-tested in isolation and reused by parsing, query
generation, and export modules.

Taxonomy tolerance: the vocabularies below are seed lists, not allowlists. The
legacy PAS data-factory accepts synonyms and fine-grained values (e.g.
``"maroon red" ~ red``). Validation therefore reports unknown values as
*warnings* and never rejects them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Primary color palette shared by every color-bearing attribute.
PRIMARY_COLORS: tuple[str, ...] = (
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
    "camouflage",
    "beige",
)

# Seed vocabularies for the categorical attribute slots (see PAS attribute_v3).
ATTRIBUTE_VOCABULARY: dict[str, tuple[str, ...]] = {
    "age": ("child", "teenager", "adult", "senior"),
    "gender": ("male", "female"),
    "skin_tone": ("light", "medium", "dark"),
    "body_shape": ("thin", "fat", "normal"),
    "hair_length": ("short", "long"),
    "hair_style": ("straight", "curly", "wavy", "bald", "other"),
    "hair_color": ("black", "brown", "blonde", "red", "other"),
    "top_outer_length": ("sleeveless", "short sleeve", "long sleeve"),
    "top_outer_type": (
        "t-shirt",
        "shirt",
        "sweater",
        "hoodie",
        "vest",
        "jacket",
        "coat",
        "robe",
        "camisole",
        "other",
    ),
    "top_inner_length": ("sleeveless", "short sleeve", "long sleeve"),
    "top_inner_type": ("not visible", "collared", "round-neck", "hoodie", "sweater"),
    "bottom_length": ("short", "long"),
    "bottom_type": ("pants", "shorts", "skirt", "dress", "leggings"),
    "shoe_type": ("sneakers", "boots", "high heels", "sandals", "flip-flops", "barefoot"),
    "headwear": ("yes", "no"),
    "mask": ("yes", "no"),
    "glasses": ("yes", "no"),
    "image_quality": ("high", "medium", "low"),
}

# Demographic slots are excluded from search queries by PAS policy.
DEMOGRAPHIC_FIELDS: frozenset[str] = frozenset({"age", "gender", "skin_tone"})

# Field split: the still-image legacy PAS "v3" schema vs. the extra fields the
# per-chunk video flow adds. Kept explicit so callers/tests can reason about
# which "view" a populated ``PersonAttributes`` belongs to without guessing.
VIDEO_EXTRA_FIELDS: frozenset[str] = frozenset(
    {
        "carrying",
        "motion_or_action",
        "potential_anomaly",
        "anomaly_type",
    }
)
STRICT_V3_FIELDS: frozenset[str] = frozenset(
    {
        "age",
        "gender",
        "skin_tone",
        "body_shape",
        "hair_length",
        "hair_style",
        "hair_color",
        "top_outer_length",
        "top_outer_type",
        "top_outer_color",
        "top_outer_color_fine",
        "top_inner_length",
        "top_inner_type",
        "top_inner_color",
        "top_inner_color_fine",
        "bottom_length",
        "bottom_type",
        "bottom_color",
        "bottom_color_fine",
        "shoe_type",
        "shoe_color",
        "shoe_color_fine",
        "headwear",
        "headwear_type",
        "headwear_color",
        "headwear_color_fine",
        "mask",
        "glasses",
        "multiview_consistency",
        "image_quality",
        "viewpoint",
        "accessories",
        "natural_caption",
    }
)


class Accessory(BaseModel):
    """A single accessory expressed as (relationship, color, item)."""

    model_config = ConfigDict(extra="forbid")

    relationship: str
    color: str
    item: str

    def phrase(self) -> str:
        """Render a human phrase such as ``"carrying a red handbag"``."""
        color = self.color.strip()
        item = self.item.strip()
        descriptor = f"{color} {item}".strip()
        return f"{self.relationship.strip()} {descriptor}".strip()


class PersonAttributes(BaseModel):
    """
    Typed person attributes extracted from one or more images of an identity.

    Every categorical field is optional because real extractions are partial:
    occluded views routinely omit shoes, inner tops, etc. ``color (fine)``
    grades are kept separate so easy/medium template queries can use either the
    primary color (coarse) or the fine-grained color (specific).
    """

    model_config = ConfigDict(extra="forbid")

    age: str | None = None
    gender: str | None = None
    skin_tone: str | None = None
    body_shape: str | None = None
    hair_length: str | None = None
    hair_style: str | None = None
    hair_color: str | None = None

    top_outer_length: str | None = None
    top_outer_type: str | None = None
    top_outer_color: str | None = None
    top_outer_color_fine: str | None = None

    top_inner_length: str | None = None
    top_inner_type: str | None = None
    top_inner_color: str | None = None
    top_inner_color_fine: str | None = None

    bottom_length: str | None = None
    bottom_type: str | None = None
    bottom_color: str | None = None
    bottom_color_fine: str | None = None

    shoe_type: str | None = None
    shoe_color: str | None = None
    shoe_color_fine: str | None = None

    headwear: str | None = None
    headwear_type: str | None = None
    headwear_color: str | None = None
    headwear_color_fine: str | None = None

    mask: str | None = None
    glasses: str | None = None
    carrying: str | None = None
    motion_or_action: str | None = None
    potential_anomaly: str | None = None
    anomaly_type: str | None = None

    accessories: list[Accessory] = Field(default_factory=list)
    multiview_consistency: str | None = None
    image_quality: str | None = None
    viewpoint: str | None = None
    natural_caption: str | None = None


def split_color(value: str) -> tuple[str, str | None]:
    """
    Split a ``"primary (fine)"`` color string into its components.

    Args:
        value: A color string such as ``"blue (navy blue)"`` or ``"red"``.

    Returns:
        A ``(primary, fine)`` tuple. ``fine`` is ``None`` when no parenthetical
        grade is present.
    """
    text = value.strip()
    if "(" in text and text.endswith(")"):
        primary, _, remainder = text.partition("(")
        fine = remainder[:-1].strip()
        return primary.strip(), (fine or None)
    return text, None


def validate_attributes(attributes: PersonAttributes) -> list[str]:
    """
    Return human-readable warnings for values outside the seed vocabularies.

    This never raises and never mutates input. Unknown values are allowed
    (taxonomy tolerance) but surfaced so callers can log or audit drift.

    Args:
        attributes: The attributes to check.

    Returns:
        A list of warning strings, empty when every populated slot is known.
    """
    warnings: list[str] = []
    for field_name, allowed in ATTRIBUTE_VOCABULARY.items():
        value = getattr(attributes, field_name)
        if value is not None and value.strip().lower() not in allowed:
            warnings.append(f"{field_name}={value!r} not in seed vocabulary")

    for color_field in (
        "top_outer_color",
        "top_inner_color",
        "bottom_color",
        "shoe_color",
        "headwear_color",
    ):
        value = getattr(attributes, color_field)
        if value is not None and value.strip().lower() not in PRIMARY_COLORS:
            warnings.append(f"{color_field}={value!r} not a primary color")

    return warnings


__all__ = [
    "ATTRIBUTE_VOCABULARY",
    "DEMOGRAPHIC_FIELDS",
    "PRIMARY_COLORS",
    "STRICT_V3_FIELDS",
    "VIDEO_EXTRA_FIELDS",
    "Accessory",
    "PersonAttributes",
    "split_color",
    "validate_attributes",
]
