# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fixed KPI anomaly ontology (single source of truth).

This is the in-code home of the 22-category KPI anomaly ontology migrated
verbatim from the legacy ``vad_pseudo_labelling_pipeline`` v1.0.0
(``script/utils/anomaly_ontology.py``). The cookbook anomaly question bank
(``question_bank.anomaly.json``) encodes the same 21 anomaly categories as
independent Yes/No BCQs (``normal`` is implied when none fire); this module is
what the PAS annotation-schema adapter's official-GT lookup
uses to normalize arbitrary source labels onto the canonical category ids.

The ontology is 22 categories: ``normal`` plus 21 anomaly categories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

KPI_NORMAL_CATEGORY = "normal"
"""Canonical id used when no listed anomaly is present in a clip."""


@dataclass(frozen=True)
class AnomalyCategory:
    """A single KPI anomaly category: its id, coarse group, and description."""

    category_id: str
    group: str
    description: str


KPI_ANOMALY_ONTOLOGY: tuple[AnomalyCategory, ...] = (
    AnomalyCategory(
        "normal",
        "Normal",
        "No listed anomaly is present in the clip.",
    ),
    AnomalyCategory(
        "traffic_accident_or_crash",
        "Traffic",
        "Traffic accident, vehicle crash, collision, rollover, or immediate crash aftermath.",
    ),
    AnomalyCategory(
        "traffic_violation",
        "Traffic",
        "Unsafe or illegal traffic behavior, including wrong-way driving, illegal lane changing, "
        "red-light running, illegal parking or blocking, and illegal pedestrian crossing.",
    ),
    AnomalyCategory(
        "traffic_obstruction",
        "Traffic",
        "Obstacles, debris, stopped objects, animals, or other visible blockage obstructing "
        "traffic or creating a road hazard.",
    ),
    AnomalyCategory(
        "animal_incident",
        "Animals",
        "Animal attacking, chasing, hurting, threatening people, entering an unsafe public area, "
        "or obstructing traffic or pathways.",
    ),
    AnomalyCategory(
        "robbery_or_mugging",
        "Crime / Violence",
        "Robbery, mugging, or forceful theft from a person.",
    ),
    AnomalyCategory(
        "stealing_or_shoplifting",
        "Crime / Violence",
        "Person stealing, shoplifting, or taking property unlawfully.",
    ),
    AnomalyCategory(
        "physical_fight",
        "Crime / Violence",
        "Mutual physical fight between two or more people who are actively engaged.",
    ),
    AnomalyCategory(
        "physical_abuse",
        "Crime / Violence",
        "One-sided physical assault, beating, kicking, pushing, dragging, restraint, or abuse "
        "of another person.",
    ),
    AnomalyCategory(
        "shooting_or_gunfire",
        "Crime / Violence",
        "Shooting, gunfire, visible gun attack, or immediate gunfire aftermath.",
    ),
    AnomalyCategory(
        "riot_or_large_violent_disturbance",
        "Crime / Violence",
        "Riot, crowd violence, or large-scale violent disturbance.",
    ),
    AnomalyCategory(
        "vandalism",
        "Crime / Violence",
        "Vandalism, property destruction, or deliberate damage to public or private property.",
    ),
    AnomalyCategory(
        "fire_smoke_or_explosion",
        "Hazards / Disasters",
        "Visible fire, smoke, blast, explosion, or immediate explosive or fire aftermath.",
    ),
    AnomalyCategory(
        "flooding_or_tsunami",
        "Hazards / Disasters",
        "Flooding, tsunami, or dangerous water inundation.",
    ),
    AnomalyCategory(
        "tornado_or_severe_windstorm",
        "Hazards / Disasters",
        "Tornado, severe windstorm, or wind-driven hazardous event.",
    ),
    AnomalyCategory(
        "avalanche_or_landslide",
        "Hazards / Disasters",
        "Avalanche, landslide, mudslide, or earth or snow collapse.",
    ),
    AnomalyCategory(
        "falling_objects",
        "Objects / Falls",
        "Objects falling from height, falling from shelves or racks, or creating danger after "
        "falling.",
    ),
    AnomalyCategory(
        "suspicious_or_dangerous_object",
        "Objects / Falls",
        "Suspicious, dangerous, thrown, abandoned, or hazardous object.",
    ),
    AnomalyCategory(
        "person_falling_or_collapsing",
        "Objects / Falls",
        "Person falling, slipping, tripping, collapsing, or lying down after a fall.",
    ),
    AnomalyCategory(
        "suspicious_running",
        "Objects / Falls",
        "Person running in a suspicious, fleeing, panic, chase, or hazardous context.",
    ),
    AnomalyCategory(
        "warehouse_safety_violation",
        "Warehouse / Workplace",
        "Unsafe behavior in a warehouse, warehouse aisle, loading or storage area, industrial "
        "stockroom, or similar warehouse-like workplace, including PPE violation, unsafe climbing, "
        "unsafe forklift interaction, smoking or vaping violation, or industrial near miss.",
    ),
    AnomalyCategory(
        "industrial_spill_or_leak",
        "Warehouse / Workplace",
        "Industrial liquid spill, leak, overflow, or spillover involving containers, barrels, "
        "boxes, forklifts, or warehouse materials.",
    ),
)

KPI_ANOMALY_CATEGORIES: tuple[str, ...] = tuple(item.category_id for item in KPI_ANOMALY_ONTOLOGY)
"""All 22 canonical category ids (``normal`` first, then the 21 anomalies)."""

_CATEGORY_BY_ID = {item.category_id: item for item in KPI_ANOMALY_ONTOLOGY}


def _label_key(value: str) -> str:
    """Normalize an arbitrary label to a comparison key (lowercase snake)."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


_SOURCE_LABEL_ALIASES: dict[str, str | None] = {
    "abnormal": None,
    "anomalous": None,
    "chad_anomalous": None,
    "normal": "normal",
    "nonfight": "normal",
    "activity": "normal",
    "hard_negative": "normal",
    "seed_frame": "normal",
    "assault": "physical_abuse",
    "animals_hurt_human": "animal_incident",
    "animals_in_urban_areas": "animal_incident",
    "boxes_falling_rack": "falling_objects",
    "building_collapse": "falling_objects",
    "climbing": "warehouse_safety_violation",
    "damaged_boxes": "falling_objects",
    "falling": "person_falling_or_collapsing",
    "fight": "physical_fight",
    "fighting": "physical_fight",
    "fire_smoke": "fire_smoke_or_explosion",
    "flooding_or_tsunami": "flooding_or_tsunami",
    "forklift_collision": "warehouse_safety_violation",
    "forklift_spillover": "industrial_spill_or_leak",
    "hurt": "physical_abuse",
    "illegal_lane_changing": "traffic_violation",
    "liquid_spillover_barrel": "industrial_spill_or_leak",
    "liquid_spillover_boxes": "industrial_spill_or_leak",
    "near_miss_forklift_person": "warehouse_safety_violation",
    "person_falling": "person_falling_or_collapsing",
    "ppe_violation": "warehouse_safety_violation",
    "shoplifting": "stealing_or_shoplifting",
    "throwing": "suspicious_or_dangerous_object",
    "vaping_smoking": "warehouse_safety_violation",
}
"""Maps source-dataset label keys onto canonical ids (``None`` drops the label)."""


def kpi_anomaly_categories() -> list[str]:
    """Return a copy of all 22 canonical category ids."""
    return list(KPI_ANOMALY_CATEGORIES)


def format_category_options(categories: list[str]) -> str:
    """Render ``categories`` as a ``- id (group): description`` bullet list."""
    lines: list[str] = []
    for category in categories:
        item = _CATEGORY_BY_ID.get(category)
        if item:
            lines.append(f"- {item.category_id} ({item.group}): {item.description}")
        else:
            lines.append(f"- {category}")
    return "\n".join(lines)


def is_dropped_source_label(label: Any) -> bool:
    """True when ``label`` is a recognized but intentionally-dropped source alias.

    Distinguishes deliberately-ambiguous labels the ontology maps to nothing
    (e.g. ``abnormal``/``anomalous``) from genuinely-unknown labels. Callers can
    use this to drop the former while still keeping the latter verbatim, instead
    of treating every ``normalize_kpi_category() is None`` result the same way.

    Args:
        label: The raw source label (any type).
    Returns:
        ``True`` only when the normalized key is present in the alias table and
        maps to ``None``.
    """
    if not isinstance(label, str) or not label.strip():
        return False
    key = _label_key(label)
    return key in _SOURCE_LABEL_ALIASES and _SOURCE_LABEL_ALIASES[key] is None


def normalize_kpi_category(label: Any) -> str | None:
    """
    Normalize an arbitrary label onto a canonical KPI category id.

    Mirrors the legacy ``anomaly_ontology.normalize_kpi_category``: empty or
    non-string input yields ``None``; a ``*_negative`` suffix maps to ``normal``;
    a direct ontology match is returned as-is; otherwise the source-label alias
    table is consulted (which may itself map to ``None`` to drop the label).

    Args:
        label: The raw label (any type; only non-empty strings can match).
    Returns:
        The canonical category id, or ``None`` when the label is unknown/dropped.
    """
    if not isinstance(label, str) or not label.strip():
        return None
    key = _label_key(label)
    if key.endswith("_negative"):
        return KPI_NORMAL_CATEGORY
    if key in _CATEGORY_BY_ID:
        return key
    return _SOURCE_LABEL_ALIASES.get(key)


__all__ = [
    "KPI_ANOMALY_CATEGORIES",
    "KPI_ANOMALY_ONTOLOGY",
    "KPI_NORMAL_CATEGORY",
    "AnomalyCategory",
    "format_category_options",
    "is_dropped_source_label",
    "kpi_anomaly_categories",
    "normalize_kpi_category",
]
