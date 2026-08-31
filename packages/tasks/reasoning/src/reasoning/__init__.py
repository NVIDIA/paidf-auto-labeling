# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning task package.

Transforms PL's internal stage outputs and optional LLM responses into
DAFT-compliant payloads, then writes them to canonical scene paths. Single
source of truth for the schema version is ``DAFT_VERSION`` in ``common``.
"""
