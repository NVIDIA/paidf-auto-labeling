# SPDX-FileCopyrightText: Copyright (c) 2024 vukasin-stanojevic
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Minimal vendored BoostTrack tracker (MIT licensed).

Upstream: https://github.com/vukasin-stanojevic/BoostTrack

This package is intentionally slimmed down for integration with our pipeline.

Vendored notice:
- This directory contains code vendored from the upstream BoostTrack repository (MIT).
- SPDX-License-Identifier is `MIT AND Apache-2.0`: upstream MIT is preserved and NVIDIA
  modifications are licensed under Apache-2.0; see the SPDX-FileCopyrightText headers above.
"""

from .boost_track import BoostTrack  # noqa: F401
