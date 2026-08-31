# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral LLM request types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

type ChatMessage = dict[str, Any]


@dataclass(frozen=True)
class ChatRequest:
    """Text chat request shared by task-level LLM adapters."""

    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    extra_body: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


class ChatTextClient(Protocol):
    """Protocol for text-generation clients."""

    def generate(self, request: ChatRequest) -> str:
        """Return the generated text for ``request``."""
