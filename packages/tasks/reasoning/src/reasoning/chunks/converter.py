# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for service-owned DAFT chunks pivots."""

from core.formats.daft.converters.chunks import to_daft_chunks

__all__ = ["to_daft_chunks"]
