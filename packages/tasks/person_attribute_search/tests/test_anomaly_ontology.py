# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the KPI anomaly ontology (single source of truth)."""

from __future__ import annotations

import pytest
from person_attribute_search.anomaly_ontology import (
    KPI_ANOMALY_CATEGORIES,
    KPI_NORMAL_CATEGORY,
    format_category_options,
    is_dropped_source_label,
    kpi_anomaly_categories,
    normalize_kpi_category,
)


def test_ontology_has_22_categories_with_normal_first() -> None:
    """The ontology is 22 categories: normal plus 21 anomalies."""
    assert len(KPI_ANOMALY_CATEGORIES) == 22
    assert KPI_ANOMALY_CATEGORIES[0] == KPI_NORMAL_CATEGORY
    # 21 anomaly categories follow normal, all unique.
    assert len(set(KPI_ANOMALY_CATEGORIES)) == 22


def test_kpi_anomaly_categories_returns_independent_copy() -> None:
    """The accessor returns a fresh list that callers may mutate safely."""
    first = kpi_anomaly_categories()
    first.append("mutation")
    assert "mutation" not in kpi_anomaly_categories()


def test_normalize_direct_ontology_match() -> None:
    """A canonical id (in any spacing/case) normalizes to itself."""
    assert normalize_kpi_category("physical_fight") == "physical_fight"
    assert normalize_kpi_category("Physical Fight") == "physical_fight"


def test_normalize_alias_mapping() -> None:
    """Source-dataset labels map to their canonical KPI id via the alias table."""
    assert normalize_kpi_category("shoplifting") == "stealing_or_shoplifting"
    assert normalize_kpi_category("fighting") == "physical_fight"
    assert normalize_kpi_category("person_falling") == "person_falling_or_collapsing"
    assert normalize_kpi_category("ppe_violation") == "warehouse_safety_violation"


def test_normalize_negative_suffix_maps_to_normal() -> None:
    """Any ``*_negative`` label collapses to the normal category."""
    assert normalize_kpi_category("fight_negative") == KPI_NORMAL_CATEGORY
    assert normalize_kpi_category("anything-negative") == KPI_NORMAL_CATEGORY


def test_normalize_dropped_and_unknown_labels_return_none() -> None:
    """Explicitly-dropped aliases and unknown labels both yield None."""
    assert normalize_kpi_category("anomalous") is None
    assert normalize_kpi_category("totally_unknown_label") is None


def test_is_dropped_source_label_distinguishes_dropped_from_unknown() -> None:
    """Only recognized aliases mapped to None are 'dropped'; unknowns are not."""
    # Intentionally-dropped ambiguous aliases (case/spacing-insensitive).
    assert is_dropped_source_label("abnormal") is True
    assert is_dropped_source_label("Anomalous") is True
    # A genuinely-unknown label is not "dropped" (callers keep it verbatim).
    assert is_dropped_source_label("totally_unknown_label") is False
    # A label that maps to a canonical id is not "dropped".
    assert is_dropped_source_label("shoplifting") is False
    # Non-strings / blanks are not dropped.
    assert is_dropped_source_label(None) is False
    assert is_dropped_source_label("  ") is False


@pytest.mark.parametrize("value", [None, "", "   ", 123, ["fighting"]])
def test_normalize_rejects_non_string_or_empty(value: object) -> None:
    """Non-strings and blank strings normalize to None."""
    assert normalize_kpi_category(value) is None


def test_format_category_options_known_and_unknown() -> None:
    """Known ids render with group + description; unknown ids render bare."""
    rendered = format_category_options(["physical_fight", "made_up"])
    assert "- physical_fight (Crime / Violence):" in rendered
    assert "- made_up" in rendered
