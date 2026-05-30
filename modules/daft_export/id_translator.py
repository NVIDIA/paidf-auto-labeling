# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge VLM ``id_<n>`` event refs to tracker ``<class>_<n>`` keys.

The VLM reads numeric track ids off the red-overlay video produced by the
detection-and-tracking stage (the overlay paints only the numeric track id;
see ``rfdetr_tracking.draw_tracks`` red-id branch). The prompt asks it to wrap
those labels as ``id_<n>``, but real model outputs sometimes use bare numeric
strings or JSON numbers. The tracker, meanwhile, persists each object in
``instances.json`` under ``<class>_<n>``. Both are intended to refer to the
same physical object via the shared numeric suffix.

A strict cross-reference check (DAFT validator, downstream consumers) sees
``id_76`` and ``car_76`` as unrelated strings, so this module performs the
mechanical translation. VLM ids whose numeric suffix isn't in the catalogue
are dropped: in our runs we've observed these in two situations — det/track
enabled (small handful of mismatches per scene) and det/track disabled
(every event id is a mismatch). The exact mechanism (VLM misread,
prompt-example leak, ungrounded inference) hasn't been pinned down; the
treatment is the same in both cases (drop and warn).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

# ``{id: <int>}`` or ``{id: <int>, id: <int>, ...}`` annotation, capturing
# any leading whitespace so deletion leaves clean prose. The few-shot
# examples in the video prompt (e.g. "motorcycle {id: 6127}") teach this
# exact shape, which is also what the model emits when it leaks.
_ID_ANNOTATION_RE = re.compile(r"\s*\{\s*id\s*:\s*\d+(?:\s*,\s*id\s*:\s*\d+)*\s*\}")


def _canonical_numeric_suffix(s: str) -> str:
    """Normalize VLM numeric ID suffixes so ``id_001`` grounds to track ``*_1``."""
    stripped = s.lstrip("0")
    return stripped or "0"


def _vlm_numeric_suffix(value: object) -> str | None:
    """Extract a canonical numeric suffix from VLM ID variants."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value >= 0 else None
    if not isinstance(value, str):
        return None
    raw = value.strip()
    suffix = raw[3:] if raw.startswith("id_") else raw
    if not suffix.isdigit():
        return None
    return _canonical_numeric_suffix(suffix)


def index_instances_by_suffix(instances_keys: Iterable[str]) -> dict[str, str]:
    """Map ``<class>_<n>`` keys to ``{<n>: <class>_<n>}`` for O(1) lookup.

    On suffix collision (rare: same numeric track id ending up under two
    class prefixes due to a mid-track class flip) the last-inserted key
    wins. Picking deterministically beats picking-with-warning since the
    "right" answer is undefined either way — the VLM only ever saw the
    number, not the class. Non-integer suffixes are skipped: they can't
    be hit by an ``id_<int>`` lookup and would only add noise.
    """
    return {
        k.rsplit("_", 1)[-1]: k
        for k in instances_keys
        if isinstance(k, str) and "_" in k and k.rsplit("_", 1)[-1].isdigit()
    }


def ground_vlm_ids(
    vlm_ids: Iterable[object],
    suffix_to_key: dict[str, str],
    *,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Translate VLM ``id_<n>`` to tracker ``<class>_<n>``; drop the rest.

    Anything that doesn't parse as ``id_<int>``, a bare integer string, or a
    JSON integer is dropped. Leading zeros from VLM/OCR variants are normalized
    (``id_001`` -> ``id_1``). Pass an empty dict to drop everything; the caller
    is expected to log the "instances catalogue empty/missing" case once (see
    ``contextual.to_daft_events``) so we don't spam a per-id warning for every
    event when the catalogue is empty.
    """
    out: list[str] = []
    for vid in vlm_ids:
        suffix = _vlm_numeric_suffix(vid)
        if suffix is None:
            continue
        key = suffix_to_key.get(suffix)
        if key is None:
            if logger is not None and suffix_to_key:
                # Only warn when there *was* a catalogue to ground against;
                # the empty-catalogue case is already logged once at the
                # call site to avoid per-id spam.
                logger.warning("[id_translator] dropping ungrounded VLM id %r", vid)
            continue
        out.append(key)
    return out


def strip_ungrounded_id_annotations(text: str, suffix_to_key: dict[str, str]) -> str:
    """Drop ``{id: <n>}`` prose annotations whose ids aren't in the catalogue.

    Prose-side companion to :func:`ground_vlm_ids`: same grounding policy,
    different data shape. The structured ``instances`` array is cleaned by
    id translation, but the same VLM run can also leak example numbers
    into ``event_caption`` / ``event_summary`` via the ``{id: <n>}``
    annotation pattern from the few-shot examples. Pass the same
    ``suffix_to_key`` used for the structured side so prose and structured
    views stay consistent.

    An annotation is dropped only when *every* numeric id inside it is
    missing from ``suffix_to_key`` (mixed annotations are kept verbatim:
    partial rewrites would distort the model's intent). An empty
    catalogue (no overlay) drops everything.
    """
    if not isinstance(text, str):
        return text

    def _replace(m: re.Match[str]) -> str:
        if any(_canonical_numeric_suffix(s) in suffix_to_key for s in re.findall(r"\d+", m.group(0))):
            return m.group(0)
        return ""

    return _ID_ANNOTATION_RE.sub(_replace, text).lstrip()


__all__ = [
    "ground_vlm_ids",
    "index_instances_by_suffix",
    "strip_ungrounded_id_annotations",
]
