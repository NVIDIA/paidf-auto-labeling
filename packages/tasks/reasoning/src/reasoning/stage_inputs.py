# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-once DAFT stage input cache and JSON helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reasoning.paths import ScenePaths


def safe_read_json(path: Path, *, logger: logging.Logger, tag: str) -> Any | None:
    """Read ``path`` as JSON, returning ``None`` for expected I/O failures."""
    if not path.exists():
        logger.debug("[%s] no file at %s; skipping", tag, path)
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[%s] failed to read %s: %s; skipping", tag, path, exc)
        return None


#: Person-attribute keys surfaced (in order) into the anomaly evidence
#: block. Kept small and anomaly-relevant; the full attribute map stays
#: on disk in ``pas.json`` and is not duplicated into the prompt.
_PAS_EVIDENCE_ATTRS: tuple[str, ...] = (
    "motion or action",
    "potential anomaly",
    "anomaly type",
)

#: Cap on persons rendered so a crowded scene cannot blow up the prompt.
_PAS_MAX_PEOPLE: int = 20


def _format_person_evidence(person: dict[str, Any]) -> str | None:
    """Render one PAS person entry as an anomaly-evidence line, or ``None``.

    Returns ``None`` when the entry carries neither a usable attribute
    nor a caption — a bare track id is not evidence worth a prompt line.
    """
    track_id = person.get("track_id")
    label = f"Person track {track_id}" if track_id is not None else "Person"
    attributes = person.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}

    body: list[str] = []
    for key in _PAS_EVIDENCE_ATTRS:
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            body.append(f"{key}={value.strip()}")

    caption = person.get("natural_language_caption")
    if not (isinstance(caption, str) and caption.strip()):
        caption = attributes.get("natural language caption")
    if isinstance(caption, str) and caption.strip():
        body.append(f'"{caption.strip()}"')

    if not body:
        return None
    return f"{label}: " + "; ".join(body)


@dataclass(frozen=True)
class StageInputs:
    """Snapshot of sidecar and contextual JSON used by DAFT emitters."""

    sidecar_metadata: dict | None
    sidecar_metadata_chunk: dict | None
    scene_video: dict | None
    scene_image: dict | None
    scene_events: dict | None
    scene_instances: dict | None
    # The LLM-aggregated structured description (``contextual/msted.json``).
    # Optional, defaulted, and last so existing positional/keyword
    # constructions (and the runtime-contract tests) keep working without
    # naming it. Populated by :meth:`load` and by ``load_stage_inputs``.
    scene_msted: dict | None = None
    # The anomaly verdict sidecar (``sidecars/reasoning/anomaly.json``).
    # Like ``scene_msted`` it is optional, defaulted, and last so existing
    # constructions keep working. Lets the causal-linkage stage auto-tag
    # ``video_type`` from the anomaly classification.
    scene_anomaly: dict | None = None
    # The person-attribute-search deliverable
    # (``sidecars/person_attribute_search/pas.json``) produced by the PAS pass.
    # Optional and defaulted so existing constructions keep working. Lets the
    # anomaly stage fold grounded per-person attributes (actions, potential-
    # anomaly flags, captions) into its prompt as corroborating evidence.
    scene_pas: dict | None = None

    @classmethod
    def load(cls, paths: ScenePaths, *, logger: logging.Logger) -> StageInputs:
        """Read every input the emitter family may need, exactly once."""
        sm = safe_read_json(paths.sidecar_metadata, logger=logger, tag="stage-inputs")
        smc = safe_read_json(paths.sidecar_metadata_chunk, logger=logger, tag="stage-inputs")
        sv = safe_read_json(paths.contextual_video, logger=logger, tag="stage-inputs")
        si = safe_read_json(paths.contextual_image, logger=logger, tag="stage-inputs")
        se = safe_read_json(paths.contextual_events, logger=logger, tag="stage-inputs")
        sinst = safe_read_json(paths.contextual_instances, logger=logger, tag="stage-inputs")
        smsted = safe_read_json(paths.contextual_msted, logger=logger, tag="stage-inputs")
        sanom = safe_read_json(paths.sidecar_anomaly, logger=logger, tag="stage-inputs")
        spas = safe_read_json(paths.sidecar_pas, logger=logger, tag="stage-inputs")
        return cls(
            sidecar_metadata=sm if isinstance(sm, dict) else None,
            sidecar_metadata_chunk=smc if isinstance(smc, dict) else None,
            scene_video=sv if isinstance(sv, dict) else None,
            scene_image=si if isinstance(si, dict) else None,
            scene_events=se if isinstance(se, dict) else None,
            scene_instances=sinst if isinstance(sinst, dict) else None,
            scene_msted=smsted if isinstance(smsted, dict) else None,
            scene_anomaly=sanom if isinstance(sanom, dict) else None,
            scene_pas=spas if isinstance(spas, dict) else None,
        )

    def windows(self) -> list[dict]:
        """Per-window list from the sidecar metadata, or ``[]`` if absent."""
        meta = self.sidecar_metadata or {}
        windows = meta.get("windows")
        return windows if isinstance(windows, list) else []

    def dense_windows(self) -> list[dict]:
        """Per-window list from the dense-caption sidecar, or ``[]`` if absent."""
        meta = self.sidecar_metadata_chunk or {}
        windows = meta.get("windows")
        return windows if isinstance(windows, list) else []

    def scene_prose(self) -> tuple[str | None, str | None, float | None]:
        """Return ``(scene_description, event_summary, duration)`` from video metadata."""
        video = self.scene_video or {}
        scene_description = video.get("scene_description")
        event_summary = video.get("event_summary")
        duration = video.get("duration")
        return (
            scene_description.strip()
            if isinstance(scene_description, str) and scene_description.strip()
            else None,
            event_summary.strip()
            if isinstance(event_summary, str) and event_summary.strip()
            else None,
            float(duration) if isinstance(duration, (int, float)) and duration > 0 else None,
        )

    def synthesized_description(self) -> str | None:
        """Return the synthesized structured description from ``msted.json``.

        MSTED is this package's "description synthesis" substrate: an
        LLM aggregates per-window captions into a scene-level overview
        plus a salient-event characterization (category, cause,
        consequence, ...). Surfacing it as a single grounding string lets
        the downstream reasoning stages (QA family, causal linkage)
        reason *from the synthesized description* rather than from the raw
        per-window prose alone — the description-to-QA flow that gives
        reasoning traces their depth.

        Returns ``None`` when no usable ``msted.json`` is present so
        callers transparently fall back to the lighter
        :meth:`scene_prose` description.
        """
        msted = self.scene_msted or {}
        parts: list[str] = []

        scene_description = msted.get("scene_description")
        if isinstance(scene_description, str) and scene_description.strip():
            parts.append(scene_description.strip())

        event = msted.get("event_description")
        if isinstance(event, dict):
            # Preferred ordering puts the causal keys (cause/consequence/
            # root_cause) last so the salient-event line reads as a small
            # narrative. Any extra free-form keys are appended in their
            # original order — MSTED's event_description is intentionally
            # an open string-map.
            preferred = (
                "category",
                "description",
                "temporal_description",
                "spatial_location",
                "cause",
                "consequence",
                "root_cause",
            )
            bits: list[str] = []
            seen: set[str] = set()
            for key in preferred:
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    bits.append(f"{key}: {value.strip()}")
                    seen.add(key)
            for key, value in event.items():
                if key in seen:
                    continue
                if isinstance(value, str) and value.strip():
                    bits.append(f"{key}: {value.strip()}")
            if bits:
                parts.append("Salient event — " + "; ".join(bits))

        if not parts:
            return None
        return "\n".join(parts)

    def reasoning_context(self) -> str | None:
        """Return the best available scene-level grounding for reasoning.

        Prefers the synthesized MSTED description (richer, causal) and
        falls back to the plain ``scene_description`` from ``video.json``.
        Returns ``None`` when neither is present — callers then ground
        purely in the per-window captions.
        """
        synthesized = self.synthesized_description()
        if synthesized:
            return synthesized
        scene_description, _event_summary, _duration = self.scene_prose()
        return scene_description

    def anomaly_video_type(self) -> str | None:
        """Return the anomaly verdict as a causal ``video_type``, if any.

        Reads ``classification`` from the anomaly sidecar and returns it
        only when it is one of the two values the causal-linkage
        ``video_type`` enum accepts (``"anomaly"`` / ``"normal"``). Any
        other / missing value yields ``None`` so the causal stage falls
        back to its own ``default_video_type`` (or leaves pairs untagged).
        """
        anomaly = self.scene_anomaly or {}
        label = anomaly.get("classification")
        if isinstance(label, str) and label.strip().lower() in ("anomaly", "normal"):
            return label.strip().lower()
        return None

    def person_attributes_block(self) -> str | None:
        """Return a compact per-person evidence block from ``pas.json``.

        Renders one bounded line per tracked person with the anomaly-
        relevant attributes (action, potential-anomaly flag/type) plus the
        natural-language caption, e.g.::

            Person track 0: motion or action=other; potential anomaly=no; "A person ..."

        The list is capped at :data:`_PAS_MAX_PEOPLE` so a crowded scene
        cannot blow up the prompt. Returns ``None`` when no usable PAS
        payload is present so callers transparently fall back to
        caption-only anomaly reasoning.
        """
        pas = self.scene_pas or {}
        block = pas.get("pas")
        people = block.get("people") if isinstance(block, dict) else None
        if not isinstance(people, list) or not people:
            return None
        lines: list[str] = []
        for person in people[:_PAS_MAX_PEOPLE]:
            if not isinstance(person, dict):
                continue
            line = _format_person_evidence(person)
            if line:
                lines.append(line)
        return "\n".join(lines) if lines else None

    def image_caption(self) -> str | None:
        """Return ``caption`` from image metadata for image scenes."""
        image = self.scene_image or {}
        caption = image.get("caption")
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
        return None

    def event_list(self) -> list[dict]:
        """Per-event list from ``events.json``, or ``[]`` if absent."""
        events = self.scene_events or {}
        event_list = events.get("events")
        return event_list if isinstance(event_list, list) else []

    def instances_catalogue_entries(self) -> list[dict]:
        """Flat catalogue of object ids from ``contextual/instances.json``."""
        instances = self.scene_instances or {}
        raw = instances.get("instances")
        if not isinstance(raw, dict):
            return []
        out: list[dict] = []
        for object_id, entry in raw.items():
            if not isinstance(object_id, str) or not object_id.strip():
                continue
            if not isinstance(entry, dict):
                continue
            label = entry.get("object_type") or entry.get("label")
            track_id = entry.get("track_id")
            if track_id is None:
                tail = object_id.rsplit("_", 1)
                if len(tail) == 2 and tail[1].isdigit():
                    track_id = int(tail[1])
            row: dict[str, Any] = {"object_id": object_id.strip()}
            if isinstance(label, str) and label.strip():
                row["label"] = label.strip()
            if track_id is not None:
                row["track_id"] = track_id
            out.append(row)
        return out

    def with_refreshed_events(self, paths: ScenePaths, *, logger: logging.Logger) -> StageInputs:
        """Return a copy with ``scene_events`` re-read from disk when usable."""
        scene_events = safe_read_json(
            paths.contextual_events,
            logger=logger,
            tag="stage-inputs:refresh",
        )
        if not isinstance(scene_events, dict):
            return self
        return replace(self, scene_events=scene_events)

    def with_refreshed_msted(self, paths: ScenePaths, *, logger: logging.Logger) -> StageInputs:
        """Return a copy with ``scene_msted`` re-read from disk when usable.

        The MSTED stage may write ``contextual/msted.json`` during the
        same run; refreshing here lets the downstream reasoning stages
        (QA family, causal linkage) ground in the freshly synthesized
        description rather than a pre-run snapshot. Mirrors
        :meth:`with_refreshed_events`.
        """
        scene_msted = safe_read_json(
            paths.contextual_msted,
            logger=logger,
            tag="stage-inputs:refresh",
        )
        if not isinstance(scene_msted, dict):
            return self
        return replace(self, scene_msted=scene_msted)

    def with_anomaly_verdict(self, verdict: dict | None) -> StageInputs:
        """Return a copy carrying an in-memory anomaly verdict.

        The anomaly stage already returns its normalized verdict, so the
        orchestrator can ingest it directly instead of re-reading the
        just-written sidecar from disk. :meth:`anomaly_video_type` only
        needs the ``classification`` field, which the verdict carries.
        Returns ``self`` unchanged when ``verdict`` is not a usable dict.
        """
        if not isinstance(verdict, dict):
            return self
        return replace(self, scene_anomaly=verdict)

    def with_refreshed_anomaly(self, paths: ScenePaths, *, logger: logging.Logger) -> StageInputs:
        """Return a copy with ``scene_anomaly`` re-read from disk when usable.

        The anomaly stage writes ``sidecars/reasoning/anomaly.json`` early
        in the run; refreshing here lets the causal-linkage stage auto-tag
        ``video_type`` from the freshly written verdict. Mirrors
        :meth:`with_refreshed_msted`.
        """
        scene_anomaly = safe_read_json(
            paths.sidecar_anomaly,
            logger=logger,
            tag="stage-inputs:refresh",
        )
        if not isinstance(scene_anomaly, dict):
            return self
        return replace(self, scene_anomaly=scene_anomaly)


__all__ = ["StageInputs", "safe_read_json"]
