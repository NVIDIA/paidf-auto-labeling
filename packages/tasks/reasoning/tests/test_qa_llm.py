# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the open-ended QA LLM adapter and bank loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from reasoning.prompts import PromptVariant
from reasoning.qa.llm import (
    QaLLMError,
    generate_qa_with_llm,
    load_qa_bank,
)


def fake_prompt() -> PromptVariant:
    return PromptVariant(
        name="fake_qa",
        system="QA_SYSTEM",
        user_template=(
            "WIN={windows_block}\n"
            "SCENE={scene_description_block}\n"
            "EVT={event_summary_block}\n"
            "Q={questions_block}"
        ),
    )


def windows_fixture() -> list[dict]:
    return [
        {"start_s": 0.0, "end_s": 4.0, "caption": "captionA"},
        {"start_s": 4.0, "end_s": 7.0, "caption": "captionB"},
    ]


def stub_object_response(obj: dict) -> Any:
    """Build a stub for ``call_chat_object_with_structured_fallback`` returning
    ``(parsed, raw)``."""

    def _stub(**kwargs):
        return obj, json.dumps(obj)

    return _stub


# ---------------------------------------------------------------------------
# load_qa_bank
# ---------------------------------------------------------------------------


class TestQaBankInline:
    def test_inline_strings(self):
        out = load_qa_bank(questions=["a", "b"], question_file=None)
        assert out == [{"question": "a"}, {"question": "b"}]

    def test_inline_dicts(self):
        out = load_qa_bank(
            questions=[
                {"question": "a"},
                {"question": "b", "options": {"A": "x", "B": "y"}},
            ],
            question_file=None,
        )
        assert out == [
            {"question": "a"},
            {"question": "b", "options": {"A": "x", "B": "y"}},
        ]

    def test_dedupe_and_strip(self):
        out = load_qa_bank(
            questions=["  a  ", "a", "b", "", "  ", {"question": "  c  "}],
            question_file=None,
        )
        assert out == [{"question": "a"}, {"question": "b"}, {"question": "c"}]

    def test_text_alias(self):
        out = load_qa_bank(
            questions=[{"text": "hello"}, {"query": "world"}],
            question_file=None,
        )
        assert out == [{"question": "hello"}, {"question": "world"}]


class TestQaBankFile:
    def test_yaml_bare_list(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump(["q1", "q2"]), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f)
        assert out == [{"question": "q1"}, {"question": "q2"}]

    def test_yaml_object_with_questions(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump({"questions": ["q1", "q2"]}), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f)
        assert out == [{"question": "q1"}, {"question": "q2"}]

    def test_mcq_style_with_options(self, tmp_path):
        f = tmp_path / "q.yaml"
        bank = {
            "questions": [
                {"question": "Pick A or B", "options": {"A": "alpha", "B": "beta"}},
                {"text": "Free-form Q"},
            ]
        }
        f.write_text(yaml.safe_dump(bank), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f)
        assert out == [
            {"question": "Pick A or B", "options": {"A": "alpha", "B": "beta"}},
            {"question": "Free-form Q"},
        ]

    def test_json_file(self, tmp_path):
        f = tmp_path / "q.json"
        f.write_text(json.dumps({"items": [{"question": "X"}]}), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f)
        assert out == [{"question": "X"}]

    def test_inline_plus_file_concatenated_and_deduped(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump(["b", "c"]), encoding="utf-8")
        out = load_qa_bank(questions=["a", "b"], question_file=f)
        assert out == [{"question": "a"}, {"question": "b"}, {"question": "c"}]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_qa_bank(questions=None, question_file=tmp_path / "missing.yaml")

    def test_unknown_shape_raises(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a list"):
            load_qa_bank(questions=None, question_file=f)

    def test_empty_returns_empty(self):
        assert load_qa_bank(questions=None, question_file=None) == []


class TestQaBankSection:
    """The unified-cookbook bank shape: file is a dict with per-task sections.

    When ``section=<task>`` is passed, the loader prefers entries from
    that section over the legacy top-level ``questions`` array, which
    means a single bank file can drive ``open_qa`` / ``mcq_openended``
    / ``bcq_openended`` / ``temporal_localization`` from disjoint slices
    while remaining backward-compatible with consumers that don't pass
    ``section`` (notably the ``mcq_generation`` ``question-driven-vlm-llm``
    mode, which keeps reading the top-level array).
    """

    @staticmethod
    def _unified_bank() -> dict:
        return {
            "name": "test_bank",
            "version": "2.0",
            "questions": [
                {"id": "legacy_1", "question": "legacy question 1"},
            ],
            "open_qa": [
                {"id": "oq_1", "question": "open Q1"},
                {"id": "oq_2", "question": "open Q2"},
            ],
            "mcq_openended": [
                {
                    "id": "mo_1",
                    "question": "MCQ Q1",
                    "options": {"A": "alpha", "B": "beta"},
                },
            ],
            "bcq_openended": [
                {"id": "bo_1", "question": "BCQ Q1"},
            ],
        }

    def test_section_open_qa_skips_legacy_array(self, tmp_path):
        """When ``section`` is supplied, the section wins over top-level ``questions``."""
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f, section="open_qa")
        assert out == [{"question": "open Q1"}, {"question": "open Q2"}]

    def test_section_mcq_openended_preserves_options(self, tmp_path):
        """``options`` dict from a sectioned MCQ entry survives normalization."""
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f, section="mcq_openended")
        assert out == [
            {"question": "MCQ Q1", "options": {"A": "alpha", "B": "beta"}},
        ]

    def test_section_bcq_openended(self, tmp_path):
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f, section="bcq_openended")
        assert out == [{"question": "BCQ Q1"}]

    def test_no_section_arg_falls_back_to_legacy_array(self, tmp_path):
        """Without ``section``, the existing top-level ``questions`` array is used."""
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f)
        assert out == [{"question": "legacy question 1"}]

    def test_missing_section_falls_back_to_legacy_array(self, tmp_path):
        """Requesting a section absent from the file falls back to legacy shape."""
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f, section="not_in_file")
        assert out == [{"question": "legacy question 1"}]

    def test_yaml_sectioned_bank(self, tmp_path):
        """YAML files honour ``section`` the same way JSON does."""
        f = tmp_path / "bank.yaml"
        f.write_text(yaml.safe_dump(self._unified_bank()), encoding="utf-8")
        out = load_qa_bank(questions=None, question_file=f, section="open_qa")
        assert out == [{"question": "open Q1"}, {"question": "open Q2"}]

    def test_inline_plus_section_file_concatenated_and_deduped(self, tmp_path):
        """Inline ``questions`` are prepended to the resolved section list and deduped."""
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_qa_bank(
            questions=["open Q2", "extra inline"],
            question_file=f,
            section="open_qa",
        )
        assert out == [
            {"question": "open Q2"},
            {"question": "extra inline"},
            {"question": "open Q1"},
        ]

    def test_section_only_file_no_legacy_questions_at_top(self, tmp_path):
        """A bank that *only* has sectioned content (no top-level ``questions``)
        works for sectioned consumers but raises for unsectioned ones."""
        f = tmp_path / "bank.json"
        f.write_text(
            json.dumps({"open_qa": [{"question": "only Q"}]}),
            encoding="utf-8",
        )
        out = load_qa_bank(questions=None, question_file=f, section="open_qa")
        assert out == [{"question": "only Q"}]
        # An unsectioned consumer (e.g. legacy mcq_generation reading the
        # same file) sees an unrecognised shape and surfaces a clear error.
        with pytest.raises(ValueError, match="must be a list"):
            load_qa_bank(questions=None, question_file=f)

    def test_section_value_must_be_a_list(self, tmp_path):
        """If the named section exists but isn't a list, reject the bank."""
        f = tmp_path / "bank.json"
        f.write_text(
            json.dumps(
                {
                    "open_qa": "not-a-list",
                    "questions": [{"question": "fallback Q"}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="open_qa"):
            load_qa_bank(questions=None, question_file=f, section="open_qa")

    def test_warehouse_cookbook_bank_round_trip(self):
        """Smoke check against the real cookbook bank we ship.

        Validates that the cookbook JSON parses, has every section the
        consumer code expects, and that the loader returns at least one
        normalized entry per section. This is the regression fence we
        actually care about: if anyone breaks the file structure of
        ``cookbooks/warehouse/question_bank.json``, this test fails.
        """
        bank_path = (
            Path(__file__).resolve().parents[2] / "cookbooks" / "warehouse" / "question_bank.json"
        )
        if not bank_path.is_file():
            pytest.skip(f"warehouse bank not present at {bank_path}")
        for section in ("open_qa", "mcq_openended", "bcq_openended"):
            out = load_qa_bank(questions=None, question_file=bank_path, section=section)
            assert out, f"warehouse cookbook section {section!r} resolved empty"
            assert all(isinstance(it, dict) and "question" in it for it in out)
        mcq = load_qa_bank(questions=None, question_file=bank_path, section="mcq_openended")
        assert any(isinstance(it.get("options"), dict) and len(it["options"]) >= 2 for it in mcq), (
            "warehouse cookbook mcq_openended section must carry letter-keyed options"
        )


# ---------------------------------------------------------------------------
# generate_qa_with_llm — happy path & dispatch
# ---------------------------------------------------------------------------


class TestGenerateOpenQa:
    def test_returns_parsed_items(self):
        good = {
            "items": [
                {"question": "Q?", "answer": "Free text answer."},
            ]
        }
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                windows=windows_fixture(),
                scene_description="Some scene.",
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items == [{"question": "Q?", "answer": "Free text answer."}]

    def test_renders_with_window_and_question_blocks(self):
        good = {"items": [{"question": "Q?", "answer": "ok."}]}
        captured = {}

        def stub(**kwargs):
            captured["messages"] = kwargs["messages"]
            return good, json.dumps(good)

        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                windows=windows_fixture(),
                scene_description="Some scene.",
                event_summary="Evt.",
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        user_msg = captured["messages"][1]["content"]
        assert "captionA" in user_msg
        assert "captionB" in user_msg
        assert "Some scene." in user_msg
        assert "Evt." in user_msg
        assert "Q?" in user_msg


class TestGenerateMcqOpenended:
    def test_dispatches_with_options_block(self):
        good = {"items": [{"question": "Choose the truck option", "answer": "B. The truck."}]}
        captured = {}

        def stub(**kwargs):
            captured["messages"] = kwargs["messages"]
            return good, json.dumps(good)

        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            items = generate_qa_with_llm(
                kind="mcq_openended",
                bank=[
                    {
                        "question": "Pick A or B",
                        "options": {"A": "car", "B": "truck"},
                    }
                ],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        # Options re-attached from the bank when LLM elided them.
        assert items[0]["options"] == {"A": "car", "B": "truck"}
        # Options were rendered into the prompt.
        user_msg = captured["messages"][1]["content"]
        assert "A: car" in user_msg
        assert "B: truck" in user_msg

    def test_reattaches_options_by_question_id(self):
        good = {
            "items": [
                {
                    "question_id": "q2",
                    "question": "Model rewrote the question text.",
                    "answer": "A. The bus.",
                }
            ]
        }
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_qa_with_llm(
                kind="mcq_openended",
                bank=[
                    {
                        "id": "q1",
                        "question": "First",
                        "options": {"A": "car", "B": "truck"},
                    },
                    {
                        "id": "q2",
                        "question": "Second",
                        "options": {"A": "bus", "B": "bike"},
                    },
                ],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items[0]["options"] == {"A": "bus", "B": "bike"}

    def test_does_not_overwrite_llm_options(self):
        good = {
            "items": [
                {
                    "question": "Pick A or B",
                    "answer": "B. The truck.",
                    "options": {"A": "car", "B": "truck", "C": "added"},
                }
            ]
        }
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_qa_with_llm(
                kind="mcq_openended",
                bank=[
                    {
                        "question": "Pick A or B",
                        "options": {"A": "car", "B": "truck"},
                    }
                ],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        # LLM-supplied options win; bank does not clobber.
        assert items[0]["options"] == {"A": "car", "B": "truck", "C": "added"}


class TestGenerateBcqOpenended:
    def test_dispatches(self):
        good = {"items": [{"question": "Yes/no?", "answer": "Yes. ok."}]}
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_qa_with_llm(
                kind="bcq_openended",
                bank=[{"question": "Yes/no?"}],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items[0]["answer"].startswith("Yes. ")


# ---------------------------------------------------------------------------
# generate_qa_with_llm — error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_kind_raises(self):
        with pytest.raises(QaLLMError, match="kind must be"):
            generate_qa_with_llm(
                kind="invalid",  # type: ignore[arg-type]
                bank=[{"question": "Q?"}],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_empty_bank_raises(self):
        with pytest.raises(QaLLMError, match="bank is empty"):
            generate_qa_with_llm(
                kind="open_qa",
                bank=[],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_no_usable_windows_raises(self):
        with pytest.raises(QaLLMError, match="no usable windows"):
            generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                windows=[{"foo": "bar"}],
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_unparseable_response_raises(self):
        def stub(**kwargs):
            return None, "not json"

        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            with pytest.raises(QaLLMError, match="no parseable JSON"):
                generate_qa_with_llm(
                    kind="open_qa",
                    bank=[{"question": "Q?"}],
                    windows=windows_fixture(),
                    scene_description=None,
                    event_summary=None,
                    prompt=fake_prompt(),
                    llm_url="http://x",
                    llm_model="m",
                    api_key="k",
                )

    def test_missing_items_key_raises(self):
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response({"foo": "bar"}),
        ):
            with pytest.raises(QaLLMError, match="missing 'items'"):
                generate_qa_with_llm(
                    kind="open_qa",
                    bank=[{"question": "Q?"}],
                    windows=windows_fixture(),
                    scene_description=None,
                    event_summary=None,
                    prompt=fake_prompt(),
                    llm_url="http://x",
                    llm_model="m",
                    api_key="k",
                )

    def test_empty_items_list_raises(self):
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response({"items": []}),
        ):
            with pytest.raises(QaLLMError, match="empty 'items'"):
                generate_qa_with_llm(
                    kind="open_qa",
                    bank=[{"question": "Q?"}],
                    windows=windows_fixture(),
                    scene_description=None,
                    event_summary=None,
                    prompt=fake_prompt(),
                    llm_url="http://x",
                    llm_model="m",
                    api_key="k",
                )

    def test_non_dict_items_dropped(self):
        good = {"items": ["nope", {"question": "Q?", "answer": "ok."}]}
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items == [{"question": "Q?", "answer": "ok."}]


# ---------------------------------------------------------------------------
# generate_qa_with_llm — image mode
# ---------------------------------------------------------------------------


class TestGenerateQaImageMode:
    """The DAFT QA schemas accept BOTH ``video_id`` and ``image_id`` via
    ``oneOf``. The adapter must therefore serve image scenes too — the
    pipeline used to skip them outright, which silently halved coverage
    on image-only datasets and was inconsistent with the
    ``scene_description`` / ``video_summarization`` emitters that DO
    serve images."""

    def test_image_caption_renders_into_windows_block_slot(self):
        good = {"items": [{"question": "What is shown?", "answer": "A red car."}]}
        captured: dict = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return good, json.dumps(good)

        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            items = generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "What is shown?"}],
                image_caption="A red car at an intersection.",
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items == [{"question": "What is shown?", "answer": "A red car."}]
        user_msg = captured["messages"][1]["content"]
        # The caption is plumbed through the same {windows_block} slot so
        # the bundled prompt set doesn't need an image-specific fork.
        assert "Image caption: A red car at an intersection." in user_msg
        # And the retry stage tag carries the mode for log triage.
        assert captured["retry_stage"] == "qa:open_qa:image"

    def test_image_caption_is_stripped(self):
        """Whitespace-only captions are treated as 'unset' and rejected;
        captions with surrounding whitespace are stripped before
        rendering."""
        good = {"items": [{"question": "Q?", "answer": "ok."}]}
        captured: dict = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return good, json.dumps(good)

        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                image_caption="  trimmed caption  ",
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert "Image caption: trimmed caption" in captured["messages"][1]["content"]

    def test_mcq_image_mode_renders_options_and_dispatches(self):
        """MCQ image-mode is the most complex variant: the per-question
        options must still be rendered AND the kind-specific guided JSON
        schema must be used. Regression guard for the option-bank
        re-attachment path on images."""
        good = {"items": [{"question": "Pick A or B", "answer": "A. The car."}]}
        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_qa_with_llm(
                kind="mcq_openended",
                bank=[
                    {
                        "question": "Pick A or B",
                        "options": {"A": "car", "B": "truck"},
                    }
                ],
                image_caption="A red car at an intersection.",
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items[0]["options"] == {"A": "car", "B": "truck"}

    def test_both_modes_supplied_is_rejected(self):
        """Programmer-error guard: passing both ``windows`` and
        ``image_caption`` is ambiguous (which one drives the prompt
        block?). The check fires before the LLM call so callers see the
        bug at the first pipeline run, not in production logs."""
        with pytest.raises(QaLLMError, match="exactly one of"):
            generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                windows=windows_fixture(),
                image_caption="A red car.",
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_neither_mode_supplied_is_rejected(self):
        """Programmer-error guard: omitting both modes used to fall
        through to ``_format_windows_block(None, ...)`` and explode with
        a confusing TypeError. Surface the actual contract violation
        instead."""
        with pytest.raises(QaLLMError, match="exactly one of"):
            generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                # windows omitted, image_caption omitted
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_empty_image_caption_is_rejected(self):
        """An empty / whitespace-only caption is semantically 'no
        caption' — must be rejected with the same shape error so callers
        can route to the same skip-the-stage branch."""
        with pytest.raises(QaLLMError, match="exactly one of"):
            generate_qa_with_llm(
                kind="open_qa",
                bank=[{"question": "Q?"}],
                image_caption="   ",
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_video_mode_retry_stage_carries_video_label(self):
        """Symmetry guard with the image-mode retry-stage assertion: the
        ``retry_stage`` tag carries the mode so log triage can grep
        ``qa:*:video`` vs ``qa:*:image``."""
        good = {"items": [{"question": "Q?", "answer": "ok."}]}
        captured: dict = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return good, json.dumps(good)

        with patch(
            "reasoning.qa.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            generate_qa_with_llm(
                kind="bcq_openended",
                bank=[{"question": "Did X happen?"}],
                windows=windows_fixture(),
                scene_description=None,
                event_summary=None,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert captured["retry_stage"] == "qa:bcq_openended:video"
