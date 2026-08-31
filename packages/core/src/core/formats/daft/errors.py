# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFT conversion errors shared by core pivot helpers."""


class DaftConvertError(ValueError):
    """Raised when source task data cannot be converted to DAFT shape."""


__all__ = ["DaftConvertError"]
