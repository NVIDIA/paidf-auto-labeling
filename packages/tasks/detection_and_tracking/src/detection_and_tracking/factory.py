# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracker factory.

Thin wrapper around :class:`core.registry.Registry`. Concrete backends
self-register via :func:`register_tracker` (typically from a
``backends/*.py`` module that is imported at package import time).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

from core.registry import Registry

from detection_and_tracking.config import DetectionAndTrackingConfig, TrackerKindValue
from detection_and_tracking.tracker import Tracker

TrackerKind = str
PLANNED_TRACKER_KINDS: frozenset[str] = frozenset(
    {
        "rfdetr-boosttrack",
        "rfdetr-deepocsort",
        "rfdetr-bytetrack",
        "sam3",
    }
)

TrackerBuilder = Callable[[logging.Logger, DetectionAndTrackingConfig], Tracker]

_REGISTRY: Registry[Tracker] = Registry("tracker kind")


def register_tracker(kind: str, builder: TrackerBuilder) -> None:
    """Register a tracker builder under ``kind``.

    Re-registering the same kind silently overwrites the previous entry
    so tests and plug-ins can replace builders without teardown.
    """
    _REGISTRY.register(kind, builder)


def create_tracker(
    kind: TrackerKind | str,
    logger: logging.Logger,
    config: DetectionAndTrackingConfig | None = None,
) -> Tracker:
    """Construct a registered tracker instance.

    Raises ``NotImplementedError`` for planned backends whose real
    implementation has not landed yet, and ``ValueError`` for unknown
    kinds.
    """
    if kind in PLANNED_TRACKER_KINDS and not _REGISTRY.is_registered(kind):
        registered = ", ".join(_REGISTRY.list()) or "<none>"
        raise NotImplementedError(
            f"Tracker backend {kind!r} is planned but not implemented in this scaffold. "
            f"Registered tracker backends: {registered}. Select a registered backend until "
            "the requested backend migration lands."
        )
    if config is None:
        if not _REGISTRY.is_registered(kind):
            return _REGISTRY.create(kind, logger, DetectionAndTrackingConfig())
        config = DetectionAndTrackingConfig(tracker=cast(TrackerKindValue, kind))
    return _REGISTRY.create(kind, logger, config)


def list_trackers() -> list[str]:
    """Return the sorted list of registered tracker kinds."""
    return _REGISTRY.list()


__all__ = [
    "TrackerBuilder",
    "TrackerKind",
    "PLANNED_TRACKER_KINDS",
    "create_tracker",
    "list_trackers",
    "register_tracker",
]
