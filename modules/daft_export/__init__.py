# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFT converter.

Transforms the auto-labeling-internal stage outputs (in-memory dicts) into DAFT-compliant
payloads and writes them to canonical scene paths. Single source of truth for
the schema version is ``DAFT_VERSION`` in ``common``.
"""
