# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-track inputs seam assembler (model-free).

The video (per-chunk) flow has three upstream producers feeding the PAS
per-track assembly:

1. ``detection_and_tracking`` crop extraction -> ``tracks.json`` (crop bookkeeping
   per track: scores, frames, crop dirs).
2. The per-track VLM/LLM fan-out -> per-track model outputs (attribute ``items``,
   a ``caption``, and pre-generated retrieval ``queries``).
3. The chunk-level annotation (anomaly labels + caption-derived queries).

This module merges those three into the single ``track_inputs.json`` seam that
``PersonAttributeSearchTask`` consumes. It is pure assembly: no model calls and
no filesystem access — only dict shaping. The (live) model fan-out that produces
input #2 is the one remaining step that calls a model; everything downstream of
this seam is deterministic and unit-tested.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from person_attribute_search.tracks import TrackRecord


def _string_list(values: Iterable[str] | None) -> list[str]:
    """Coerce an iterable into a list of stripped, non-empty strings."""
    if values is None:
        return []
    return [text.strip() for text in values if isinstance(text, str) and text.strip()]


def assemble_track_inputs(
    *,
    chunk_id: str,
    tracks_payload: Mapping[str, Any],
    per_track: Mapping[int, Mapping[str, Any]],
    source_annotation: str | None = None,
    anomaly_labels: Iterable[str] | None = None,
    caption_queries: Iterable[str] | None = None,
    crop_root: str | None = None,
) -> dict[str, Any]:
    """
    Merge crop bookkeeping and per-track model outputs into the PAS seam.

    Args:
        chunk_id: Identifier of the video chunk (e.g. ``"chunk_000"``).
        tracks_payload: Parsed ``tracks.json`` from crop extraction, with a
            ``tracks`` list and an optional ``crop_root``.
        per_track: Per-track model outputs keyed by ``track_id``. Each value may
            carry ``items``, ``caption``, ``queries``, and chunk-relative
            bookkeeping (``n_crops_in_chunk``, ``first_frame_in_chunk``, ...).
        source_annotation: Optional provenance pointer to the upstream annotation.
        anomaly_labels: Chunk-level anomaly query labels (normalized downstream).
        caption_queries: Chunk-level caption-derived query strings (verbatim).
        crop_root: Crop root override; defaults to ``tracks_payload["crop_root"]``.

    Returns:
        The ``track_inputs.json`` seam dict consumed by the PAS per-track flow.
    """
    known = set(TrackRecord.model_fields)
    base_tracks = tracks_payload.get("tracks")
    base_tracks = base_tracks if isinstance(base_tracks, list) else []

    seam_tracks: list[dict[str, Any]] = []
    for entry in base_tracks:
        if not isinstance(entry, Mapping):
            continue
        val = entry.get("track_id")
        if val is None or isinstance(val, bool):
            continue
        if isinstance(val, int):
            track_id = val
        else:
            raw_track_id = str(val).strip()
            if not raw_track_id.isdigit():
                continue
            try:
                track_id = int(raw_track_id)
            except (TypeError, ValueError):
                continue
        merged = {key: value for key, value in entry.items() if key in known}
        merged["track_id"] = track_id
        for key, value in per_track.get(track_id, {}).items():
            if key in known:
                merged[key] = value
        seam_tracks.append(merged)

    resolved_crop_root = (
        crop_root if crop_root is not None else str(tracks_payload.get("crop_root") or "")
    )

    seam: dict[str, Any] = {"chunk_id": chunk_id}
    if source_annotation is not None:
        seam["source_annotation"] = source_annotation
    seam["crop_root"] = resolved_crop_root
    anomaly = _string_list(anomaly_labels)
    if anomaly:
        seam["anomaly_labels"] = anomaly
    captions = _string_list(caption_queries)
    if captions:
        seam["caption_queries"] = captions
    seam["tracks"] = seam_tracks
    return seam


__all__ = ["assemble_track_inputs"]
