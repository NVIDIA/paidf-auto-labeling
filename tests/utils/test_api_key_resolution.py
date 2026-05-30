# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for get_vlm_api_key() / get_llm_api_key() / get_api_key().

Verifies priority order and fallback behaviour for all three functions:
  VLM_API_KEY > NVIDIA_API_KEY > OPENAI_API_KEY > "EMPTY"
  LLM_API_KEY > NVIDIA_API_KEY > OPENAI_API_KEY > "EMPTY"
  get_api_key() delegates to get_vlm_api_key() (which already covers shared fallbacks)
"""

from __future__ import annotations

import pytest
from mcq_generation.mcq.utils.openai import get_api_key, get_llm_api_key, get_vlm_api_key

_ALL_KEYS = ("VLM_API_KEY", "LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def _clear_api_keys(monkeypatch):
    """Remove all API key env vars before each test."""
    for k in _ALL_KEYS:
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# get_vlm_api_key
# ---------------------------------------------------------------------------


class TestGetVlmApiKey:
    def test_vlm_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "vlm-secret")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_vlm_api_key() == "vlm-secret"

    def test_nvidia_fallback_when_no_vlm_key(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_vlm_api_key() == "nvidia-secret"

    def test_openai_fallback_when_no_vlm_or_nvidia(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_vlm_api_key() == "openai-secret"

    def test_empty_placeholder_when_nothing_set(self):
        assert get_vlm_api_key() == "EMPTY"

    def test_llm_key_not_used_for_vlm(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "llm-secret")
        assert get_vlm_api_key() == "EMPTY"

    def test_empty_string_vlm_key_falls_through(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
        assert get_vlm_api_key() == "nvidia-secret"

    def test_empty_string_nvidia_falls_through(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_vlm_api_key() == "openai-secret"


# ---------------------------------------------------------------------------
# get_llm_api_key
# ---------------------------------------------------------------------------


class TestGetLlmApiKey:
    def test_llm_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "llm-secret")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_llm_api_key() == "llm-secret"

    def test_nvidia_fallback_when_no_llm_key(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_llm_api_key() == "nvidia-secret"

    def test_openai_fallback_when_no_llm_or_nvidia(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_llm_api_key() == "openai-secret"

    def test_empty_placeholder_when_nothing_set(self):
        assert get_llm_api_key() == "EMPTY"

    def test_vlm_key_not_used_for_llm(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "vlm-secret")
        assert get_llm_api_key() == "EMPTY"

    def test_empty_string_llm_key_falls_through(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
        assert get_llm_api_key() == "nvidia-secret"

    def test_empty_string_nvidia_falls_through(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_llm_api_key() == "openai-secret"


# ---------------------------------------------------------------------------
# get_api_key (generic fallback used when api_key=None in call_chat_raw)
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_vlm_key_preferred(self, monkeypatch):
        monkeypatch.setenv("VLM_API_KEY", "vlm-secret")
        monkeypatch.setenv("LLM_API_KEY", "llm-secret")
        assert get_api_key() == "vlm-secret"

    def test_nvidia_when_no_vlm_or_llm(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
        assert get_api_key() == "nvidia-secret"

    def test_openai_last_resort(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        assert get_api_key() == "openai-secret"

    def test_empty_placeholder_when_nothing_set(self):
        assert get_api_key() == "EMPTY"

    def test_result_is_nonempty_string(self):
        # Whatever the env, the result must always be a non-empty string
        # (OpenAI client rejects empty api_key).
        key = get_api_key()
        assert isinstance(key, str) and len(key) > 0
