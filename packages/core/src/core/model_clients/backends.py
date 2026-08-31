# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical OpenAI-compatible endpoint defaults.

OpenAI-compatible callers always authenticate with ``NVIDIA_API_KEY`` and supply
an explicit ``endpoint_url`` (typically a local NIM or other OpenAI-compatible
host). Public OpenAI and internal NVIDIA inference URLs are intentionally not
advertised as first-class backends.
"""

from __future__ import annotations

NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


__all__ = ["GEMINI_API_KEY_ENV", "NVIDIA_API_KEY_ENV"]
