# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolver registry for the super-resolution package."""

from __future__ import annotations

import logging
from collections.abc import Callable

from core.registry import Registry

from super_resolution.config import SuperResolutionConfig
from super_resolution.resolver import Resolver

ResolverKind = str
ResolverBuilder = Callable[[logging.Logger, SuperResolutionConfig], Resolver]

_REGISTRY: Registry[Resolver] = Registry("super-resolution resolver")


def register_resolver(kind: str, builder: ResolverBuilder) -> None:
    """Register a resolver builder under ``kind``."""
    _REGISTRY.register(kind, builder)


def create_resolver(
    kind: ResolverKind | str,
    logger: logging.Logger,
    config: SuperResolutionConfig,
) -> Resolver:
    """Construct a registered resolver."""
    return _REGISTRY.create(kind, logger, config)


def list_resolvers() -> list[str]:
    """Return registered resolver names."""
    return _REGISTRY.list()


__all__ = [
    "ResolverBuilder",
    "ResolverKind",
    "create_resolver",
    "list_resolvers",
    "register_resolver",
]
