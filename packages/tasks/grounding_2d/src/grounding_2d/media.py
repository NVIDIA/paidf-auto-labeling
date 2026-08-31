# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media helpers for grounding VLM requests."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from core.exceptions import InvalidInputError

from grounding_2d.clients import MediaPayload


def read_image_payload(path: Path) -> MediaPayload:
    """Read an image file into a core ``MediaPayload``."""
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type is None or not mime_type.startswith("image/"):
        raise InvalidInputError(
            f"Unsupported or unknown image MIME type for {path.name}: {mime_type!r}"
        )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return MediaPayload(
        mime_type=mime_type,
        data_base64=encoded,
        filename=path.name,
    )


__all__ = ["read_image_payload"]
