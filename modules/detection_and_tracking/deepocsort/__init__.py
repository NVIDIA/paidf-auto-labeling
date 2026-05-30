# SPDX-FileCopyrightText: Copyright (c) 2023 Gerard Maggiolino
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Deep-OC-SORT (MIT) vendored wrapper.

This package contains a lightly-adapted copy of the standalone Deep-OC-SORT repository
implementation (NOT BoxMOT) to keep this repo commercial-friendly.

Upstream: GerardMaggiolino/Deep-OC-SORT (MIT)

Vendored notice:
- SPDX-License-Identifier is `MIT AND Apache-2.0`: upstream MIT is preserved and NVIDIA
  modifications are licensed under Apache-2.0; see the SPDX-FileCopyrightText headers above.
"""

from .ocsort import OCSort  # noqa: F401
