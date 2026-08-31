# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LLM endpoint helpers.

The modules in this package are intentionally transport-level utilities.
Task packages own prompts, schemas, converters, and artifact contracts.
"""

from core.llm.clients import (
    OpenAICompatibleTextClient,
    call_chat_raw,
    ensure_v1,
    get_api_key,
    get_llm_api_key,
    get_vlm_api_key,
    reset_openai_client_cache,
)
from core.llm.json_extract import (
    extract_json_object,
    extract_json_object_from_llm_text,
    parse_strict_json_object,
    wrap_json_fence,
)
from core.llm.structured import (
    StructuredObjectCaller,
    build_structured_request_options,
    call_chat_object_with_structured_fallback,
    call_required_json_object_with_fallback,
    require_items_list,
    resolve_structured_output_mode,
)
from core.llm.types import ChatMessage, ChatRequest, ChatTextClient
from core.llm.vlm_response import (
    extract_visual_qa_items,
    extract_window_description,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatTextClient",
    "OpenAICompatibleTextClient",
    "StructuredObjectCaller",
    "build_structured_request_options",
    "call_chat_object_with_structured_fallback",
    "call_chat_raw",
    "call_required_json_object_with_fallback",
    "ensure_v1",
    "extract_json_object",
    "extract_json_object_from_llm_text",
    "extract_visual_qa_items",
    "extract_window_description",
    "get_api_key",
    "get_llm_api_key",
    "get_vlm_api_key",
    "parse_strict_json_object",
    "require_items_list",
    "reset_openai_client_cache",
    "resolve_structured_output_mode",
    "wrap_json_fence",
]
