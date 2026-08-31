# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the ``sidecars/reasoning/anomaly.json`` payload.

Pure shaping logic, deliberately separate from the LLM adapter so it can
be unit-tested without mocking an endpoint and reused by a caller that
already has a cached verdict.

Why a sidecar and not a DAFT type: the DAFT v3 contextual/task type set
is a closed enum (see ``core.formats.daft.types``). An "anomaly" type
cannot be added without changing the core schema and the external
``nvidia-tao-daft`` contract, so the verdict lives in a plain JSON
sidecar that ``write_daft_json`` never sees.

The re-perception request
--------------------------
When the verdict is an anomaly and a highlight window was localized, the
payload carries a ``reperception_request`` block. This is the seam to
the captioning service: a downstream VLM pass can re-caption just that
window at higher fidelity. Localizing the moment is *text* reasoning over
captions and belongs here; issuing the VLM re-caption is the captioning
service's job. The reasoning service stays LLM-only by design and never
calls a VLM itself.
"""

from __future__ import annotations

from typing import Any

#: Schema tag stamped on every sidecar so downstream consumers (and a
#: future captioning re-perception reader) can version-gate the shape.
ANOMALY_SIDECAR_SCHEMA: str = "reasoning.anomaly/v1"

#: Target service expected to honor a re-perception request.
_REPERCEPTION_TARGET: str = "captioning"


def to_anomaly_sidecar(
    verdict: dict[str, Any],
    *,
    model: str | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Turn a normalized verdict into the on-disk sidecar payload.

    ``verdict`` is the dict returned by
    :func:`reasoning.anomaly.llm.classify_anomaly_with_llm`. Only the
    keys that verdict guarantees are read; unknown keys are ignored so a
    richer future verdict never leaks un-vetted fields onto disk.

    The ``reperception_request`` block is emitted only for an anomaly
    that carries a valid ``highlight`` — a normal scene has nothing to
    re-perceive, and an anomaly without a localized window gives the
    captioning service no actionable target.
    """
    classification = verdict.get("classification")
    payload: dict[str, Any] = {
        "schema": ANOMALY_SIDECAR_SCHEMA,
        "classification": classification,
        "confidence": verdict.get("confidence"),
        "reasoning": verdict.get("reasoning", ""),
    }

    if classification == "anomaly":
        for key in ("root_cause", "consequence"):
            value = verdict.get(key)
            if isinstance(value, str) and value.strip():
                payload[key] = value.strip()

        highlight = verdict.get("highlight")
        if isinstance(highlight, dict) and highlight.get("start") and highlight.get("end"):
            sanitized = _sanitize_highlight(highlight)
            payload["highlight"] = sanitized
            payload["reperception_request"] = _reperception_request(sanitized)

    if model:
        payload["model"] = model
    if sources:
        payload["sources"] = list(sources)

    return payload


def _sanitize_highlight(highlight: dict[str, Any]) -> dict[str, Any]:
    """Whitelist the highlight fields that reach the sidecar.

    Only ``start`` / ``end`` (and an optional, non-empty string
    ``description``) are carried through. A richer-than-expected verdict
    — for example a cached one with extra model bookkeeping — therefore
    cannot leak un-vetted keys onto disk or into the captioning hand-off.
    The caller guarantees ``start`` / ``end`` are present and truthy.
    """
    sanitized: dict[str, Any] = {
        "start": highlight["start"],
        "end": highlight["end"],
    }
    description = highlight.get("description")
    if isinstance(description, str) and description.strip():
        sanitized["description"] = description.strip()
    return sanitized


def _reperception_request(highlight: dict[str, Any]) -> dict[str, Any]:
    """Build the captioning hand-off block for a localized anomaly window.

    The reasoning service does not re-caption; it *requests* that the
    captioning service do so. Keeping the request declarative (a target,
    a window, a reason) means the captioning side owns the policy — when
    to act, at what fidelity — without the reasoning service reaching
    across the boundary into a VLM.
    """
    request: dict[str, Any] = {
        "status": "requested",
        "target": _REPERCEPTION_TARGET,
        "reason": "anomaly_highlight",
        "window": {"start": highlight.get("start"), "end": highlight.get("end")},
        "note": (
            "Re-caption this window at higher fidelity to confirm the anomaly. "
            "VLM re-perception is the captioning service's responsibility; the "
            "reasoning service only requests it."
        ),
    }
    description = highlight.get("description")
    if isinstance(description, str) and description.strip():
        request["description"] = description.strip()
    return request


__all__ = [
    "ANOMALY_SIDECAR_SCHEMA",
    "to_anomaly_sidecar",
]
