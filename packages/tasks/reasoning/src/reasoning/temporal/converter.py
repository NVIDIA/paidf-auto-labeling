# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for service-owned DAFT temporal-description pivots."""

from core.formats.daft.converters.temporal import (
    DEFAULT_QUESTION_TEMPLATE,
    VideoType,
    to_daft_temporal_description,
)

__all__ = ["DEFAULT_QUESTION_TEMPLATE", "VideoType", "to_daft_temporal_description"]
