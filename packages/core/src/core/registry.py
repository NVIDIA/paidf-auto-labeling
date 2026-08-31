# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generic string-keyed factory registry shared by task packages.

The service decides which task stages to compose. Each task package then
decides which implementation to use for its own backend/emitter config,
such as ``tracker="stub"`` or ``captioner="openai-vlm"``.

A registry is the small indirection that turns those config strings into
concrete objects without making the service import every implementation
or maintain a long ``if/elif`` chain. Registering builders instead of
instances keeps heavy backends lazy: the process can list available
kinds or parse config without loading a model, and only the selected
kind is instantiated when ``create_*`` is called.

Task packages still own their public factory functions
(``register_tracker``, ``create_tracker``, ``list_trackers``, etc.).
This class only centralizes the repeated mechanics behind those
factories: store builders, construct by name, list known names, and
produce a consistent "unknown kind" error.

Usage::

    _TRACKERS: Registry[Tracker] = Registry("tracker")

    def register_tracker(kind: str, builder: TrackerBuilder) -> None:
        _TRACKERS.register(kind, builder)

    def create_tracker(kind: str, logger: logging.Logger) -> Tracker:
        return _TRACKERS.create(kind, logger)

    def list_trackers() -> list[str]:
        return _TRACKERS.list()

Builders accept arbitrary positional and keyword arguments, so each task
package can keep its type-specific builder signature. Re-registering the
same kind overwrites the previous entry so tests and plug-ins can replace
builders without explicit teardown.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry[T]:
    """
    Process-global registry mapping string kinds to builder callables.
    """

    def __init__(self, label: str) -> None:
        """
        Initialize a registry.

        Args:
            label: Singular noun used in error messages (e.g. ``"tracker"``,
                ``"emitter"``).
        """
        self._label = label
        self._builders: dict[str, Callable[..., T]] = {}

    def register(self, kind: str, builder: Callable[..., T]) -> None:
        """
        Register a builder under ``kind``. Re-registering silently overwrites the
        previous entry — task packages and tests rely on this for in-tree
        backend swaps.

        Args:
            kind: String identifier used to look up the builder later.
            builder: Callable invoked to construct an instance for ``kind``.
        """
        self._builders[kind] = builder

    def create(self, kind: str, *args: Any, **kwargs: Any) -> T:
        """
        Construct an instance for ``kind``.

        Args:
            kind: String identifier of a registered builder.
            *args: Positional arguments forwarded to the builder.
            **kwargs: Keyword arguments forwarded to the builder.
        Returns:
            The instance produced by the registered builder.
        Raises:
            ValueError: If ``kind`` is not registered. The message lists every
                registered kind so the failure is self-explanatory.
        """
        if kind not in self._builders:
            known = ", ".join(sorted(self._builders)) or "<none>"
            raise ValueError(f"Unknown {self._label}: {kind!r}. Registered: {known}")
        return self._builders[kind](*args, **kwargs)

    def list(self) -> list[str]:
        """
        Return the sorted list of registered kinds.

        Returns:
            Sorted list of registered kind strings.
        """
        return sorted(self._builders)

    def is_registered(self, kind: str) -> bool:
        """
        Check whether a builder is registered under ``kind``.

        Args:
            kind: String identifier to check.
        Returns:
            True if a builder is registered under ``kind``, else False.
        """
        return kind in self._builders

    def __contains__(self, kind: str) -> bool:
        return kind in self._builders


__all__ = ["Registry"]
