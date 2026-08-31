# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media helpers for referring-expression VLM requests."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from referring_expressions.clients import MediaPayload


def read_image_payload(path: Path) -> MediaPayload:
    """Read an image file into a core ``MediaPayload``."""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return MediaPayload(
        mime_type=mime_type,
        data_base64=encoded,
        filename=path.name,
    )


__all__ = ["read_image_payload"]
