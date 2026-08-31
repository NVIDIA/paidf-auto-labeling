# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral multimodal model request types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

EndpointProvider = Literal["openai-compatible", "gemini"]


class EndpointError(RuntimeError):
    """Raised when a model endpoint response cannot be used."""


@dataclass(frozen=True)
class MediaPayload:
    """Base64-encoded media payload for a multimodal chat request."""

    mime_type: str
    data_base64: str
    filename: str | None = None
    frame_index: int | None = None
    time_s: float | None = None
    width: int | None = None
    height: int | None = None
    num_bytes: int | None = None

    @property
    def data_url(self) -> str:
        """Return an RFC 2397 data URL."""
        return f"data:{self.mime_type};base64,{self.data_base64}"


@dataclass(frozen=True)
class ChatRequest:
    """Provider-neutral text or multimodal chat request."""

    prompt: str
    media: tuple[MediaPayload, ...] = ()
    system_prompt: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9


class EndpointClient(Protocol):
    """Protocol implemented by model endpoint clients."""

    def generate(self, request: ChatRequest) -> str:
        """Generate text for ``request``."""
