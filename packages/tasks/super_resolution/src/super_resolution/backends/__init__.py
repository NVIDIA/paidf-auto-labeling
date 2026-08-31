# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in super-resolution resolvers."""

import logging

from super_resolution.backends.seedvr2 import SeedVR2Resolver
from super_resolution.config import SuperResolutionConfig
from super_resolution.factory import register_resolver
from super_resolution.resolver import Resolver


def _build_seedvr2(logger: logging.Logger, config: SuperResolutionConfig) -> Resolver:
    return SeedVR2Resolver(logger=logger, config=config.seedvr2)


register_resolver("seedvr2", _build_seedvr2)

__all__ = ["SeedVR2Resolver"]
