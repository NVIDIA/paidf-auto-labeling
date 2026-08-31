# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Super-resolution task package."""

from super_resolution.backends import SeedVR2Resolver
from super_resolution.config import ResolutionPolicyValue, SeedVR2Config, SuperResolutionConfig
from super_resolution.factory import (
    ResolverBuilder,
    ResolverKind,
    create_resolver,
    list_resolvers,
    register_resolver,
)
from super_resolution.resolver import Resolver, SrResult
from super_resolution.task import SuperResolutionTask

__all__ = [
    "Resolver",
    "ResolverBuilder",
    "ResolverKind",
    "ResolutionPolicyValue",
    "SeedVR2Config",
    "SeedVR2Resolver",
    "SrResult",
    "SuperResolutionConfig",
    "SuperResolutionTask",
    "create_resolver",
    "list_resolvers",
    "register_resolver",
]
