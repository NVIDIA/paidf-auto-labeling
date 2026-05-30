# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mcq_generation/mcq/utils/openai.py.

Covers:
  - Pure helpers: ensure_v1, get_api_key, guided_json_enabled, _is_nim_endpoint,
    _response_format_unsupported, resolve_structured_output_mode,
    build_structured_request_options
  - JSON parsing: parse_strict_json_object, extract_json_object_from_llm_text
  - API call logic: call_chat_raw retry / non-retry / empty-content,
    call_chat_object_with_structured_fallback structured→raw fallback,
    call_chat_structured_guided_json mode dispatch
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from mcq_generation.mcq.utils.openai import (
    _RESPONSE_FORMAT_SUPPORT_CACHE,
    _is_nim_endpoint,
    _response_format_unsupported,
    build_structured_request_options,
    call_chat_object_with_structured_fallback,
    call_chat_raw,
    call_chat_structured_guided_json,
    ensure_v1,
    extract_json_object_from_llm_text,
    get_api_key,
    guided_json_enabled,
    parse_strict_json_object,
    resolve_structured_output_mode,
)
from openai import APIConnectionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MSGS = [{"role": "user", "content": "hello"}]
BASE = "http://llm:8000"
MODEL = "fake-llm"


def _mock_client(content: str = '{"mcq":[]}', finish_reason: str = "stop") -> MagicMock:
    """Return a mock OpenAI client whose first choice has the given content."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# ensure_v1
# ---------------------------------------------------------------------------


def test_ensure_v1_already_has_v1() -> None:
    assert ensure_v1("http://host:8000/v1") == "http://host:8000/v1"


def test_ensure_v1_adds_v1() -> None:
    assert ensure_v1("http://host:8000") == "http://host:8000/v1"


def test_ensure_v1_strips_trailing_slash() -> None:
    assert ensure_v1("http://host:8000/") == "http://host:8000/v1"


def test_ensure_v1_empty_returns_empty() -> None:
    assert ensure_v1("") == ""


# ---------------------------------------------------------------------------
# get_api_key
# ---------------------------------------------------------------------------


def test_get_api_key_vlm_key_takes_priority(monkeypatch) -> None:
    monkeypatch.setenv("VLM_API_KEY", "vlm-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    assert get_api_key() == "vlm-key"


def test_get_api_key_nvidia_fallback(monkeypatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    assert get_api_key() == "nv-key"


def test_get_api_key_fallback_to_openai(monkeypatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    assert get_api_key() == "oai-key"


def test_get_api_key_default_empty(monkeypatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_api_key() == "EMPTY"


# ---------------------------------------------------------------------------
# guided_json_enabled
# ---------------------------------------------------------------------------


def test_guided_json_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MCQ_DISABLE_GUIDED_JSON", raising=False)
    assert guided_json_enabled() is True


@pytest.mark.parametrize("val", ["1", "true", "yes", "y", "on", "TRUE", "YES"])
def test_guided_json_disabled_by_env(monkeypatch, val: str) -> None:
    monkeypatch.setenv("MCQ_DISABLE_GUIDED_JSON", val)
    assert guided_json_enabled() is False


# ---------------------------------------------------------------------------
# _is_nim_endpoint
# ---------------------------------------------------------------------------


def test_nim_detected_by_nvidia_com() -> None:
    assert _is_nim_endpoint(base_url="https://api.nvidia.com/v1", model="llama") is True


def test_nim_detected_by_nvcf_in_url() -> None:
    assert _is_nim_endpoint(base_url="http://nvcf.internal/v1", model="llama") is True


def test_nim_detected_by_model_prefix() -> None:
    assert _is_nim_endpoint(base_url="http://localhost:8000/v1", model="nvidia/llama-3") is True


def test_non_nim_endpoint() -> None:
    assert _is_nim_endpoint(base_url="http://localhost:8000/v1", model="llama3") is False


def test_empty_url_not_nim() -> None:
    assert _is_nim_endpoint(base_url="", model="llama3") is False


# ---------------------------------------------------------------------------
# _response_format_unsupported
# ---------------------------------------------------------------------------


def test_response_format_unsupported_detected() -> None:
    exc = Exception("response_format is an unrecognized field")
    assert _response_format_unsupported(exc) is True


def test_response_format_unsupported_not_triggered_without_keyword() -> None:
    exc = Exception("connection refused")
    assert _response_format_unsupported(exc) is False


def test_response_format_supported_extra_fields() -> None:
    exc = Exception("response_format: additional properties are not allowed")
    assert _response_format_unsupported(exc) is True


# ---------------------------------------------------------------------------
# resolve_structured_output_mode
# ---------------------------------------------------------------------------


def test_resolve_mode_nim_for_nvidia_url() -> None:
    mode = resolve_structured_output_mode(
        structured_output="auto",
        base_url="https://api.nvidia.com/v1",
        model="llama",
    )
    assert mode == "nim"


def test_resolve_mode_openai_for_local_url() -> None:
    mode = resolve_structured_output_mode(
        structured_output="auto",
        base_url="http://localhost:8000/v1",
        model="llama",
    )
    assert mode == "openai"


def test_resolve_mode_explicit_nim() -> None:
    assert resolve_structured_output_mode(structured_output="nim", base_url="http://x", model="m") == "nim"


def test_resolve_mode_explicit_off() -> None:
    assert resolve_structured_output_mode(structured_output="off", base_url="http://x", model="m") == "off"


def test_resolve_mode_invalid_falls_back_to_auto() -> None:
    mode = resolve_structured_output_mode(
        structured_output="invalid_mode",
        base_url="http://localhost:8000",
        model="llama",
        invalid_fallback="auto",
    )
    assert mode in {"openai", "nim"}


# ---------------------------------------------------------------------------
# build_structured_request_options
# ---------------------------------------------------------------------------


def test_build_options_nim_returns_guided_json() -> None:
    mode, extra_body, response_format = build_structured_request_options(
        structured_output="nim",
        base_url="http://x",
        model="m",
    )
    assert mode == "nim"
    assert extra_body is not None
    assert "nvext" in extra_body
    assert response_format is None


def test_build_options_openai_returns_response_format() -> None:
    mode, extra_body, response_format = build_structured_request_options(
        structured_output="openai",
        base_url="http://x",
        model="m",
    )
    assert mode == "openai"
    assert extra_body is None
    assert response_format == {"type": "json_object"}


def test_build_options_off_returns_nones() -> None:
    mode, extra_body, response_format = build_structured_request_options(
        structured_output="off",
        base_url="http://x",
        model="m",
    )
    assert mode == "off"
    assert extra_body is None
    assert response_format is None


# ---------------------------------------------------------------------------
# parse_strict_json_object
# ---------------------------------------------------------------------------


def test_parse_strict_valid_object() -> None:
    obj = parse_strict_json_object('{"mcq": []}')
    assert obj == {"mcq": []}


def test_parse_strict_array_returns_none() -> None:
    assert parse_strict_json_object("[1, 2, 3]") is None


def test_parse_strict_empty_returns_none() -> None:
    assert parse_strict_json_object("") is None


def test_parse_strict_invalid_json_returns_none() -> None:
    assert parse_strict_json_object("{broken json") is None


# ---------------------------------------------------------------------------
# extract_json_object_from_llm_text
# ---------------------------------------------------------------------------


def test_extract_direct_json() -> None:
    obj = extract_json_object_from_llm_text('{"mcq": []}')
    assert obj == {"mcq": []}


def test_extract_from_fence() -> None:
    text = '```json\n{"mcq": [{"id": "q1"}]}\n```'
    obj = extract_json_object_from_llm_text(text)
    assert obj is not None
    assert obj["mcq"][0]["id"] == "q1"


def test_extract_embedded_in_text() -> None:
    text = 'Here is the result:\n{"mcq": []} \nDone.'
    obj = extract_json_object_from_llm_text(text)
    assert obj == {"mcq": []}


def test_extract_trailing_comma_repaired() -> None:
    text = '{"mcq": [{"id": "q1",}],}'
    obj = extract_json_object_from_llm_text(text)
    assert obj is not None
    assert "mcq" in obj


def test_extract_missing_close_brace_repaired() -> None:
    text = '{"mcq": [{"id": "q1"}'
    obj = extract_json_object_from_llm_text(text)
    # Best-effort: may or may not parse, but should not raise
    # The important thing is it returns something or None (not an exception)
    assert obj is None or isinstance(obj, dict)


def test_extract_no_json_returns_none() -> None:
    assert extract_json_object_from_llm_text("No JSON here.") is None


def test_extract_empty_returns_none() -> None:
    assert extract_json_object_from_llm_text("") is None


# ---------------------------------------------------------------------------
# call_chat_raw — success
# ---------------------------------------------------------------------------


def test_call_chat_raw_success(monkeypatch) -> None:
    mock_client = _mock_client('{"mcq": []}')
    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        result = call_chat_raw(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            retries=0,
        )
    assert result == '{"mcq": []}'
    mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# call_chat_raw — retry on transient error
# ---------------------------------------------------------------------------


def test_call_chat_raw_retries_on_transient_error(monkeypatch) -> None:
    mock_client = MagicMock()
    # First call: transient error; second call: success
    choice = MagicMock()
    choice.message.content = "good response"
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    mock_client.chat.completions.create.side_effect = [
        APIConnectionError(request=MagicMock()),
        resp,
    ]

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        result = call_chat_raw(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            retries=2,
            retry_backoff_s=0.0,
        )

    assert result == "good response"
    assert mock_client.chat.completions.create.call_count == 2


def test_call_chat_raw_raises_on_non_retryable(monkeypatch) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ValueError("bad request")

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
        pytest.raises(ValueError, match="bad request"),
    ):
        call_chat_raw(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            retries=3,
            retry_backoff_s=0.0,
        )

    # Non-retryable: only attempted once
    mock_client.chat.completions.create.assert_called_once()


def test_call_chat_raw_exhausts_retries_and_raises(monkeypatch) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
        pytest.raises(APIConnectionError),
    ):
        call_chat_raw(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            retries=2,
            retry_backoff_s=0.0,
        )

    # retries=2 means 3 total attempts (0, 1, 2)
    assert mock_client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# call_chat_object_with_structured_fallback
# ---------------------------------------------------------------------------


def test_structured_fallback_structured_succeeds(monkeypatch) -> None:
    """Structured call returns valid JSON → returned directly, raw not called."""
    payload = json.dumps({"mcq": [{"id": "q1", "question": "Q?", "options": ["A"], "answer": "A"}]})
    mock_client = _mock_client(payload)

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        obj, raw = call_chat_object_with_structured_fallback(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            retries=0,
            structured_output="openai",
        )

    assert isinstance(obj, dict)
    assert "mcq" in obj
    # Only one call: structured succeeded
    assert mock_client.chat.completions.create.call_count == 1


def test_structured_fallback_falls_back_to_raw(monkeypatch) -> None:
    """Structured call returns garbage → falls back to raw call which returns valid JSON."""
    garbage = "not json at all"
    payload = json.dumps({"mcq": []})

    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        choice = MagicMock()
        choice.message.content = garbage if call_count[0] == 1 else payload
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        obj, raw = call_chat_object_with_structured_fallback(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            retries=0,
            structured_output="openai",
        )

    assert isinstance(obj, dict)
    # 2 calls: 1 structured (failed parse) + 1 raw (succeeded)
    assert mock_client.chat.completions.create.call_count == 2


def test_structured_fallback_mode_off_skips_structured(monkeypatch) -> None:
    """mode=off skips structured call entirely; only raw call made."""
    payload = json.dumps({"mcq": []})
    mock_client = _mock_client(payload)

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        obj, raw = call_chat_object_with_structured_fallback(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            retries=0,
            structured_output="off",
        )

    assert isinstance(obj, dict)
    mock_client.chat.completions.create.assert_called_once()


def test_structured_fallback_validator_rejects_then_recovers(monkeypatch) -> None:
    """Wrong-shape JSON is rejected by validator; retry returns right shape."""
    wrong_shape = json.dumps({"video_id": "x", "scene_description": "wrong call"})
    right_shape = json.dumps({"events": [{"event_id": "e1"}]})

    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        choice = MagicMock()
        choice.message.content = wrong_shape if call_count[0] == 1 else right_shape
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        obj, _raw = call_chat_object_with_structured_fallback(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            retries=2,
            structured_output="openai",
            validator=lambda o: isinstance(o.get("events"), list),
        )

    assert isinstance(obj, dict)
    assert isinstance(obj.get("events"), list)
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# call_chat_structured_guided_json — mode dispatch
# ---------------------------------------------------------------------------


def test_call_chat_structured_mode_off_returns_none(monkeypatch) -> None:
    with patch("mcq_generation.mcq.utils.openai.OpenAI"):
        result = call_chat_structured_guided_json(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            structured_output="off",
        )
    assert result is None


def test_call_chat_structured_openai_mode(monkeypatch) -> None:
    _RESPONSE_FORMAT_SUPPORT_CACHE.clear()
    payload = json.dumps({"mcq": [{"id": "q1", "question": "Q?", "options": ["A"], "answer": "A"}]})
    mock_client = _mock_client(payload)

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        result = call_chat_structured_guided_json(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            structured_output="openai",
        )

    assert isinstance(result, dict)
    assert isinstance(result.get("mcq"), list)
    # response_format={"type":"json_object"} was passed
    create_kwargs = mock_client.chat.completions.create.call_args[1]
    assert create_kwargs.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# call_chat_raw — list-content response (multi-part, e.g. some VLM APIs)
# ---------------------------------------------------------------------------


def test_call_chat_raw_list_content_joined(monkeypatch) -> None:
    """When message.content is a list of text parts, they are joined."""
    choice = MagicMock()
    # content is a list of dicts with "text" key
    choice.message.content = [{"text": "part1"}, {"text": "part2"}]
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = resp

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        result = call_chat_raw(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            retries=0,
        )

    assert "part1" in result
    assert "part2" in result


def test_call_chat_raw_list_content_object_parts(monkeypatch) -> None:
    """List-content parts as objects with .text attribute (not dict)."""
    part = MagicMock()
    part.text = "object part"
    choice = MagicMock()
    choice.message.content = [part]
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = resp

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        result = call_chat_raw(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            retries=0,
        )

    assert result == "object part"


# ---------------------------------------------------------------------------
# call_chat_raw — reasoning field fallback (Qwen-style models)
# ---------------------------------------------------------------------------


def test_call_chat_raw_reasoning_field_fallback(monkeypatch) -> None:
    """Empty content + message.reasoning → reasoning text returned."""
    choice = MagicMock()
    choice.message.content = ""
    choice.message.reasoning = "thinking step by step"
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = resp

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
    ):
        result = call_chat_raw(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            retries=0,
        )

    assert result == "thinking step by step"


# ---------------------------------------------------------------------------
# call_chat_structured_guided_json — NIM guided_json path
# ---------------------------------------------------------------------------


def test_call_chat_structured_nim_guided_json(monkeypatch) -> None:
    """mode=nim tries guided_json; returns parsed MCQ dict on success."""
    _RESPONSE_FORMAT_SUPPORT_CACHE.clear()
    payload = json.dumps({"mcq": [{"id": "q1", "question": "Q?", "options": ["A"], "answer": "A"}]})
    mock_client = _mock_client(payload)

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI", return_value=mock_client),
        patch("mcq_generation.mcq.utils.openai.time.sleep"),
        patch("mcq_generation.mcq.utils.openai.guided_json_enabled", return_value=True),
    ):
        result = call_chat_structured_guided_json(
            base_url=BASE,
            model=MODEL,
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            structured_output="nim",
        )

    assert isinstance(result, dict)
    assert isinstance(result.get("mcq"), list)
    # guided_json was passed via extra_body (nvext or root-level)
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "extra_body" in call_kwargs


def test_call_chat_structured_guided_json_disabled(monkeypatch) -> None:
    """When guided_json is disabled via env, NIM mode falls through to openai or returns None."""
    _RESPONSE_FORMAT_SUPPORT_CACHE.clear()

    with (
        patch("mcq_generation.mcq.utils.openai.OpenAI"),
        patch("mcq_generation.mcq.utils.openai.guided_json_enabled", return_value=False),
    ):
        # NIM mode with guided_json disabled → _try_guided_json returns None
        # _try_response_format_json_object also called; but since it's a non-nim URL,
        # for mode=nim the fallback is skipped if not is_nim
        result = call_chat_structured_guided_json(
            base_url="http://localhost:8000",  # non-nim
            model="llama",
            messages=MSGS,
            timeout=10,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            structured_output="nim",
        )

    # Non-NIM endpoint with guided_json disabled and mode=nim → None
    assert result is None
