# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model endpoint clients for task packages."""

from core.model_clients.backends import GEMINI_API_KEY_ENV, NVIDIA_API_KEY_ENV
from core.model_clients.factory import create_endpoint_client
from core.model_clients.gemini import GeminiClient, gemini_generate_url
from core.model_clients.http import post_json
from core.model_clients.openai_compatible import (
    OpenAICompatibleClient,
    content_text,
    create_openai_chat_completion,
    create_openai_client,
    openai_base_url,
)
from core.model_clients.reasoning import (
    DEFAULT_PARSER,
    ReasoningParser,
    extra_body_for_parser,
    merge_extra_body,
    strip_think_blocks,
)
from core.model_clients.types import (
    ChatRequest,
    EndpointClient,
    EndpointError,
    EndpointProvider,
    MediaPayload,
)

__all__ = [
    "DEFAULT_PARSER",
    "GEMINI_API_KEY_ENV",
    "NVIDIA_API_KEY_ENV",
    "ChatRequest",
    "EndpointClient",
    "EndpointError",
    "EndpointProvider",
    "GeminiClient",
    "MediaPayload",
    "OpenAICompatibleClient",
    "ReasoningParser",
    "content_text",
    "create_endpoint_client",
    "create_openai_chat_completion",
    "create_openai_client",
    "extra_body_for_parser",
    "gemini_generate_url",
    "merge_extra_body",
    "openai_base_url",
    "post_json",
    "strip_think_blocks",
]
