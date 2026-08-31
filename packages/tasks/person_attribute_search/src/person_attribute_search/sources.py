# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Adaptive attribute-source resolution.

Decouples the PAS task from any single upstream producer. Structured person
attributes can be produced by different model stages:

* ``visual_qa`` — normalized ``items.json`` (``{"items": [{"id", "answer"}, ...]}``).
* ``captioning`` — a ``parsed`` JSON object captured when the captioning VLM is
  driven by the PAS attribute prompt.

Both are normalized to a single ``items`` list so the rest of the task is
producer-agnostic. Adding a new producer for a new domain is one new
``items_from_*`` adapter plus a config entry — the task itself does not change.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import ScenePaths, read_json, resolve_sidecar

from person_attribute_search.attributes import attributes_from_answers
from person_attribute_search.schema import Accessory, PersonAttributes
from person_attribute_search.track_inputs import assemble_track_inputs

Item = dict[str, Any]

_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "n/a",
        "na",
        "none",
        "not available",
        "not visible",
        "null",
        "other",
        "unavailable",
        "unknown",
    }
)


@dataclass(frozen=True)
class MergedAttributeSource:
    """One person assembled from one or more attribute-file observations."""

    attributes: PersonAttributes
    person_key: str | None
    person_id: str | None
    dataset: str | None
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class AttributeImageSource:
    """One image-level observation from an explicit attribute JSON file."""

    attributes: PersonAttributes
    person_key: str
    person_id: str
    dataset: str
    image_id: str
    source_entry: dict[str, Any]


def attribute_entries(payload: Any) -> list[dict[str, Any]]:
    """Return entries from supported PAS attribute JSON envelopes."""
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        entries = [entry for entry in payload["entries"] if isinstance(entry, dict)]
    elif isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("attributes"), dict):
        entries = [payload]
    else:
        raise ValueError(
            "attribute JSON must be an entry, a list of entries, or {'entries': [...]}"
        )
    if not entries:
        raise ValueError("attribute JSON did not contain any object entries")
    return entries


def merge_attribute_json(path: Path) -> MergedAttributeSource:
    """Load and deterministically merge attribute observations for one person."""
    entries = attribute_entries(read_json(path))
    observations: list[PersonAttributes] = []
    usable_entries: list[dict[str, Any]] = []
    for entry in entries:
        raw_attributes = entry.get("attributes")
        if not isinstance(raw_attributes, Mapping) or not raw_attributes:
            continue
        cleaned = {
            str(key): value for key, value in raw_attributes.items() if not _placeholder(value)
        }
        attributes = attributes_from_answers(cleaned)
        attributes = attributes.model_copy(
            update={"accessories": _clean_accessories(attributes.accessories)}
        )
        if not _has_attributes(attributes):
            continue
        observations.append(attributes)
        usable_entries.append(entry)
    if not observations:
        raise ValueError("attribute JSON did not contain any entries with usable attributes")

    merged = _merge_attribute_observations(observations)
    identity = _first_string(usable_entries, ("person_key",))
    person_id = _first_string(usable_entries, ("person_id",))
    dataset = _first_string(usable_entries, ("dataset",))
    image_ids = tuple(
        value
        for entry in usable_entries
        if (
            value := _entry_identifier(
                entry,
                ("image_id", "image", "image_path", "output_image", "crop_path"),
            )
        )
    )
    return MergedAttributeSource(
        attributes=merged,
        person_key=identity,
        person_id=person_id,
        dataset=dataset,
        image_ids=image_ids,
    )


def load_attribute_image_sources(path: Path) -> list[AttributeImageSource]:
    """Load explicit attribute records without merging distinct images.

    An entry containing several values in ``images`` is expanded so every image
    receives its own query bundle. The source entry is retained for lossless,
    additive export metadata.
    """
    sources: list[AttributeImageSource] = []
    for entry_index, entry in enumerate(attribute_entries(read_json(path))):
        raw_attributes = entry.get("attributes")
        if not isinstance(raw_attributes, Mapping) or not raw_attributes:
            continue
        cleaned = {
            str(key): value for key, value in raw_attributes.items() if not _placeholder(value)
        }
        attributes = attributes_from_answers(cleaned)
        attributes = attributes.model_copy(
            update={"accessories": _clean_accessories(attributes.accessories)}
        )
        if not _has_attributes(attributes):
            continue

        image_ids = _entry_image_ids(entry)
        if not image_ids:
            image_ids = [f"image_{entry_index:05d}"]
        person_key = _entry_identifier(entry, ("person_key", "person_id")) or image_ids[0]
        person_id = _entry_identifier(entry, ("person_id", "person_key")) or person_key
        dataset = _entry_identifier(entry, ("dataset",)) or "upa"
        for image_id in image_ids:
            source_entry = dict(entry)
            source_entry["image_id"] = image_id
            sources.append(
                AttributeImageSource(
                    attributes=attributes,
                    person_key=person_key,
                    person_id=person_id,
                    dataset=dataset,
                    image_id=image_id,
                    source_entry=source_entry,
                )
            )
    if not sources:
        raise ValueError("attribute JSON did not contain any entries with usable attributes")
    return sources


def _entry_image_ids(entry: Mapping[str, Any]) -> list[str]:
    """Return stable, de-duplicated image identifiers from one source entry."""
    values: list[str] = []
    direct = _entry_identifier(
        entry,
        ("image_id", "image", "image_path", "output_image", "crop_path"),
    )
    if direct:
        return [direct]
    raw_images = entry.get("images")
    if isinstance(raw_images, list):
        values.extend(str(value).strip() for value in raw_images if str(value).strip())
    return list(dict.fromkeys(values))


def _merge_attribute_observations(observations: list[PersonAttributes]) -> PersonAttributes:
    """Merge scalar fields by vote and accessories by stable union."""
    merged: dict[str, Any] = {}
    for field_name in PersonAttributes.model_fields:
        if field_name == "accessories":
            continue
        values = [
            value
            for observation in observations
            if (value := getattr(observation, field_name)) is not None and not _placeholder(value)
        ]
        if values:
            counts = Counter(str(value).casefold() for value in values)
            winner = max(
                counts,
                key=lambda key: (
                    counts[key],
                    -next(
                        index for index, value in enumerate(values) if str(value).casefold() == key
                    ),
                ),
            )
            merged[field_name] = next(value for value in values if str(value).casefold() == winner)

    accessories: list[Accessory] = []
    seen: set[tuple[str, str, str]] = set()
    for observation in observations:
        for accessory in observation.accessories:
            key = (
                accessory.relationship.casefold(),
                accessory.color.casefold(),
                accessory.item.casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            accessories.append(accessory)
    if accessories:
        merged["accessories"] = accessories
    return PersonAttributes(**merged)


def _placeholder(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in _PLACEHOLDER_VALUES
    return False


def _clean_accessories(accessories: list[Accessory]) -> list[Accessory]:
    """Drop placeholder accessories and blank placeholder qualifiers."""
    return [
        Accessory(
            relationship="" if _placeholder(accessory.relationship) else accessory.relationship,
            color="" if _placeholder(accessory.color) else accessory.color,
            item=accessory.item,
        )
        for accessory in accessories
        if not _placeholder(accessory.item)
    ]


def _has_attributes(attributes: PersonAttributes) -> bool:
    return any(
        bool(value) if isinstance(value, list) else value is not None
        for value in attributes.model_dump().values()
    )


def _first_string(entries: list[dict[str, Any]], keys: tuple[str, ...]) -> str | None:
    for entry in entries:
        value = _entry_identifier(entry, keys)
        if value:
            return value
    return None


def _entry_identifier(entry: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def items_from_visual_qa(payload: Any) -> list[Item] | None:
    """Normalize a ``visual_qa`` sidecar payload into items, or ``None``.

    Returns ``None`` (no usable source) when the payload is not a dict, lacks a
    valid ``items`` list, or yields no item objects. A present-but-empty or
    malformed sidecar is therefore never reported as a successful-but-empty
    source, which would otherwise let an empty upstream VQA run produce a PAS
    output built from fabricated attributes.
    """
    if not isinstance(payload, dict):
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    normalized = [item for item in items if isinstance(item, dict)]
    return normalized or None


def items_from_captioning(payload: Any) -> list[Item] | None:
    """
    Normalize a ``captioning`` sidecar's parsed JSON object into items.

    Looks for ``parsed`` (image/video sidecar) and falls back to
    ``model_output`` (contextual payload). Each top-level key becomes an item
    ``{"id": key, "answer": value}`` so the standard attribute parser applies.
    """
    if not isinstance(payload, dict):
        return None
    parsed = payload.get("parsed")
    if not isinstance(parsed, Mapping):
        parsed = payload.get("model_output")
    if not isinstance(parsed, Mapping):
        return None
    return [{"id": str(key), "answer": value} for key, value in parsed.items()]


def per_track_from_visual_qa_windows(payload: Any) -> dict[int, dict[str, Any]]:
    """
    Build per-track model outputs from a ``visual_qa`` normalized windows sidecar.

    In the video flow ``visual_qa`` runs once per track (each track's crops are a
    window) and stamps ``track_id`` into every window. This turns those windows
    into the ``{track_id: {items, caption, n_crops_in_chunk}}`` fragments that
    :func:`~person_attribute_search.track_inputs.assemble_track_inputs` merges
    with the detection ``tracks.json`` bookkeeping.

    Args:
        payload: Parsed ``visual_qa/windows.normalized.json`` payload.

    Returns:
        Per-track fragments keyed by ``track_id`` (empty when no track windows).
    """
    if not isinstance(payload, dict):
        return {}
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for window in windows:
        if not isinstance(window, dict):
            continue
        value = window.get("track_id")
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, str):
            if not value.strip().isdigit():
                continue
            track_id = int(value)
        elif isinstance(value, int):
            track_id = value
        else:
            continue
        raw_items = window.get("items")
        fragment: dict[str, Any] = {
            "items": [item for item in raw_items if isinstance(item, dict)]
            if isinstance(raw_items, list)
            else [],
        }
        description = window.get("description")
        if isinstance(description, str) and description.strip():
            fragment["caption"] = description.strip()
        n_crops = window.get("n_crops_in_chunk")
        if isinstance(n_crops, int):
            fragment["n_crops_in_chunk"] = n_crops
        out[track_id] = fragment
    return out


def find_sidecar(paths: ScenePaths, candidates: tuple[str, ...]) -> Path | None:
    """Return the first existing sidecar among ``candidates`` (in order)."""
    for candidate in candidates:
        try:
            path = resolve_sidecar(paths.sidecars_dir, candidate)
        except ValueError:
            continue
        if path.is_file():
            return path
    return None


def load_track_inputs(paths: ScenePaths, track_inputs_sidecar: str) -> dict[str, Any] | None:
    """Read the explicit per-track inputs seam if present and object-shaped."""
    sidecar = resolve_sidecar(paths.sidecars_dir, track_inputs_sidecar)
    if not sidecar.is_file():
        return None
    payload = read_json(sidecar)
    return payload if isinstance(payload, dict) else None


def assemble_track_inputs_from_upstream(
    paths: ScenePaths,
    *,
    chunk_id: str,
    tracks_sidecars: tuple[str, ...],
    visual_qa_window_sidecars: tuple[str, ...],
) -> dict[str, Any] | None:
    """
    Assemble per-track inputs from detection tracks and per-track VQA windows.

    Returns ``None`` when either producer is absent or when no usable per-track
    VQA windows exist, allowing the caller to fall back to the single-identity
    flow.
    """
    tracks_sidecar = find_sidecar(paths, tracks_sidecars)
    windows_sidecar = find_sidecar(paths, visual_qa_window_sidecars)
    if tracks_sidecar is None or windows_sidecar is None:
        return None

    per_track = per_track_from_visual_qa_windows(read_json(windows_sidecar))
    if not per_track:
        return None

    tracks_payload = read_json(tracks_sidecar)
    if not isinstance(tracks_payload, dict):
        return None

    return assemble_track_inputs(
        chunk_id=chunk_id,
        tracks_payload=tracks_payload,
        per_track=per_track,
        crop_root=str(tracks_payload.get("crop_root") or ""),
    )


def resolve_attribute_items(
    paths: ScenePaths,
    *,
    visual_qa_sidecars: tuple[str, ...],
    caption_attribute_sidecars: tuple[str, ...] = (),
) -> tuple[list[Item] | None, str | None]:
    """
    Resolve structured attribute items from the first available producer.

    ``visual_qa`` is tried first to preserve existing behavior; captioning-parsed
    sources are only consulted when ``caption_attribute_sidecars`` is configured.

    Args:
        paths: Scene paths to resolve sidecars against.
        visual_qa_sidecars: Candidate ``visual_qa`` item sidecars, in order.
        caption_attribute_sidecars: Optional captioning sidecars whose ``parsed``
            JSON carries the attributes (empty disables this source).

    Returns:
        ``(items, source_name)`` where ``source_name`` is ``"visual_qa"`` or
        ``"captioning"``; ``(None, None)`` when no producer is available.
    """
    sidecar = find_sidecar(paths, visual_qa_sidecars)
    if sidecar is not None:
        items = items_from_visual_qa(read_json(sidecar))
        if items is not None:
            return items, "visual_qa"

    caption_sidecar = find_sidecar(paths, caption_attribute_sidecars)
    if caption_sidecar is not None:
        items = items_from_captioning(read_json(caption_sidecar))
        if items is not None:
            return items, "captioning"

    return None, None


def extract_caption(payload: Any) -> str | None:
    """Best-effort extraction of a caption string from a captioning sidecar."""
    if not isinstance(payload, dict):
        return None
    caption_keys = ("caption", "natural_caption", "vlm_caption", "description", "summary")
    for key in caption_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for container_key in ("parsed", "model_output"):
        container = payload.get(container_key)
        if isinstance(container, str) and container.strip():
            return container.strip()
        if not isinstance(container, dict):
            continue
        for key in caption_keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def string_list(value: Any) -> list[str]:
    """Coerce a payload value into a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


__all__ = [
    "Item",
    "assemble_track_inputs_from_upstream",
    "extract_caption",
    "find_sidecar",
    "items_from_captioning",
    "items_from_visual_qa",
    "load_track_inputs",
    "per_track_from_visual_qa_windows",
    "resolve_attribute_items",
    "string_list",
]
