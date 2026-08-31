# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-aggregated ``contextual/events.json`` producer.

Mirror of the ``msted`` package: a pure converter (``converter.py``) plus
an LLM caller (``llm.py``). Callers that already have structured event
data (cached, hand-written, produced by a non-LLM source) can import
:func:`to_daft_events` directly without dragging in any LLM
infrastructure.
"""

from reasoning.events.converter import to_daft_events
from reasoning.events.llm import (
    EventsLLMError,
    generate_events_with_llm,
    parse_cached_events,
)

__all__ = [
    "EventsLLMError",
    "generate_events_with_llm",
    "parse_cached_events",
    "to_daft_events",
]
