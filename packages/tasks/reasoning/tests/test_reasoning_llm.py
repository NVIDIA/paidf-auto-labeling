# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the reasoning enrichment LLM adapter.

Mocks ``call_chat_raw`` at the module boundary — verifies adapter logic
(prompt rendering, context block composition, post-processing,
error handling). LLM behavior itself is out of scope."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from reasoning.prompts import PromptVariant
from reasoning.reasoning.llm import (
    ReasoningLLMError,
    generate_reasoning_for_item,
)


def fake_prompt() -> PromptVariant:
    """Echo prompt: surfaces every placeholder so tests can assert what
    the adapter rendered."""
    return PromptVariant(
        name="fake_reasoning",
        system="REASONING_SYSTEM",
        user_template="Q={question}|A={answer}|CTX={context_block}",
    )


class TestHappyPath:
    def test_returns_text(self):
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value="The car turns right because the captions show steady deceleration.",
        ):
            out = generate_reasoning_for_item(
                question="What does the car do?",
                answer="It turns right.",
                context="00:00-00:04: car decelerating",
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert "turns right" in out

    def test_messages_carry_system_and_rendered_user(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "trace"

        with patch("reasoning.reasoning.llm.call_chat_raw", side_effect=stub):
            generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context="CTX_TEXT",
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        msgs = captured["messages"]
        assert msgs[0] == {"role": "system", "content": "REASONING_SYSTEM"}
        user = msgs[1]
        assert user["role"] == "user"
        # Adapter substituted both required placeholders + composed context block.
        assert "Q=Q?" in user["content"]
        assert "A=A." in user["content"]
        assert "Context:" in user["content"]
        assert "CTX_TEXT" in user["content"]

    def test_no_context_renders_empty_context_block(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return "trace"

        with patch("reasoning.reasoning.llm.call_chat_raw", side_effect=stub):
            generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        # Empty context -> no "Context:" header in the prompt.
        assert "Context:" not in captured["messages"][1]["content"]

    def test_blank_context_treated_as_no_context(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return "trace"

        with patch("reasoning.reasoning.llm.call_chat_raw", side_effect=stub):
            generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context="    ",
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert "Context:" not in captured["messages"][1]["content"]

    def test_endpoint_args_threaded_through(self):
        captured: dict[str, Any] = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return "trace"

        with patch("reasoning.reasoning.llm.call_chat_raw", side_effect=stub):
            generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://endpoint/v1",
                llm_model="my-model",
                max_tokens=256,
                temperature=0.3,
                top_p=0.95,
                timeout=30,
                seed=7,
                retries=4,
                retry_backoff_s=1.5,
                api_key="secret",
                retry_stage="reasoning:scene_description",
            )

        assert captured["base_url"] == "http://endpoint/v1"
        assert captured["model"] == "my-model"
        assert captured["max_tokens"] == 256
        assert captured["temperature"] == 0.3
        assert captured["top_p"] == 0.95
        assert captured["timeout"] == 30
        assert captured["seed"] == 7
        assert captured["retries"] == 4
        assert captured["retry_backoff_s"] == 1.5
        assert captured["api_key"] == "secret"
        assert captured["retry_stage"] == "reasoning:scene_description"


class TestPostProcessing:
    def test_strips_leading_label_reasoning(self):
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value="Reasoning: The captions clearly show acceleration.",
        ):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert out == "The captions clearly show acceleration."

    def test_strips_leading_label_rationale_case_insensitive(self):
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value="RATIONALE: Because the data supports it.",
        ):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert out == "Because the data supports it."

    def test_strips_outer_double_quotes(self):
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value='"Quoted reasoning trace."',
        ):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert out == "Quoted reasoning trace."

    def test_strips_outer_single_quotes(self):
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value="'Single-quoted trace.'",
        ):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert out == "Single-quoted trace."

    def test_does_not_strip_unbalanced_quotes(self):
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value='"unbalanced',
        ):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert out == '"unbalanced'

    def test_does_not_strip_when_inner_contains_same_quote(self):
        # "She said 'hi'" — outer doubles wrap a sentence containing single
        # quotes (different chars). But "'a' and 'b'" — outer singles
        # wrap a string that itself contains singles, so we refuse to
        # strip (would corrupt the inner quotes).
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value="'a' and 'b'",
        ):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert out == "'a' and 'b'"

    def test_truncates_to_max_chars(self, caplog):
        caplog.set_level(logging.WARNING)
        long_text = "x" * 5000
        with patch("reasoning.reasoning.llm.call_chat_raw", return_value=long_text):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
                max_chars=100,
            )
        assert len(out) == 100

    def test_strips_trailing_whitespace(self):
        with patch(
            "reasoning.reasoning.llm.call_chat_raw",
            return_value="   trace text   \n\n",
        ):
            out = generate_reasoning_for_item(
                question="Q?",
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert out == "trace text"


class TestErrors:
    def test_empty_response_raises(self):
        with patch("reasoning.reasoning.llm.call_chat_raw", return_value=""):
            with pytest.raises(ReasoningLLMError, match="empty/whitespace"):
                generate_reasoning_for_item(
                    question="Q?",
                    answer="A.",
                    context=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )

    def test_whitespace_only_response_raises(self):
        with patch("reasoning.reasoning.llm.call_chat_raw", return_value="   \n\t"):
            with pytest.raises(ReasoningLLMError, match="empty/whitespace"):
                generate_reasoning_for_item(
                    question="Q?",
                    answer="A.",
                    context=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )

    def test_label_only_response_raises_after_stripping(self):
        # "Reasoning:" alone -> empty after label stripping -> raise.
        with patch("reasoning.reasoning.llm.call_chat_raw", return_value="Reasoning: "):
            with pytest.raises(ReasoningLLMError, match="empty after post-processing"):
                generate_reasoning_for_item(
                    question="Q?",
                    answer="A.",
                    context=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )

    def test_blank_question_rejected(self):
        with patch("reasoning.reasoning.llm.call_chat_raw") as mock:
            with pytest.raises(ReasoningLLMError, match="question"):
                generate_reasoning_for_item(
                    question="   ",
                    answer="A.",
                    context=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )
        mock.assert_not_called()

    def test_blank_answer_rejected(self):
        with patch("reasoning.reasoning.llm.call_chat_raw") as mock:
            with pytest.raises(ReasoningLLMError, match="answer"):
                generate_reasoning_for_item(
                    question="Q?",
                    answer="",
                    context=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )
        mock.assert_not_called()

    def test_non_string_question_rejected(self):
        with pytest.raises(ReasoningLLMError, match="question"):
            generate_reasoning_for_item(
                question=42,  # type: ignore[arg-type]
                answer="A.",
                context=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
