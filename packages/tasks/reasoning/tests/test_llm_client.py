# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import pytest
from reasoning.llm_client import (
    build_structured_request_options,
    call_required_json_object_with_fallback,
    require_items_list,
)


class HelperError(RuntimeError):
    pass


def test_nim_structured_options_include_root_and_nvext_guided_json():
    schema = {"type": "object", "properties": {"items": {"type": "array"}}}

    mode, extra_body, response_format = build_structured_request_options(
        structured_output="nim",
        base_url="http://nim/v1",
        model="model",
        guided_json_schema=schema,
    )

    assert mode == "nim"
    assert extra_body == {"nvext": {"guided_json": schema}, "guided_json": schema}
    assert response_format is None


def test_required_json_object_helper_preserves_call_options():
    captured = {}

    def caller(**kwargs):
        captured.update(kwargs)
        return {"items": [{"id": "1"}]}, '{"items": [{"id": "1"}]}'

    result = call_required_json_object_with_fallback(
        base_url="http://llm/v1",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        timeout=30,
        max_tokens=128,
        temperature=0.1,
        top_p=0.9,
        logger=None,
        retries=2,
        retry_backoff_s=0.5,
        structured_output="openai",
        seed=7,
        guided_json_schema={"type": "object"},
        retry_stage="stage",
        api_key="key",
        error_type=HelperError,
        caller=caller,
    )

    assert result == {"items": [{"id": "1"}]}
    assert captured["base_url"] == "http://llm/v1"
    assert captured["model"] == "model"
    assert captured["retry_stage"] == "stage"
    assert captured["guided_json_schema"] == {"type": "object"}


def test_required_json_object_helper_raises_stage_error_on_parse_failure():
    def caller(**kwargs):
        return None, "not json"

    with pytest.raises(HelperError, match="custom parse failure.*not json"):
        call_required_json_object_with_fallback(
            base_url="http://llm/v1",
            model="model",
            messages=[{"role": "user", "content": "hi"}],
            timeout=30,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            logger=None,
            error_type=HelperError,
            no_parseable_message="custom parse failure: {snippet_repr}",
            caller=caller,
        )


def test_require_items_list_validates_shape():
    assert require_items_list(
        {"items": [{"id": "1"}]},
        error_type=HelperError,
        missing_message="missing {type_name}",
        empty_message="empty",
    ) == [{"id": "1"}]

    with pytest.raises(HelperError, match="missing str"):
        require_items_list(
            {"items": "bad"},
            error_type=HelperError,
            missing_message="missing {type_name}",
            empty_message="empty",
        )
