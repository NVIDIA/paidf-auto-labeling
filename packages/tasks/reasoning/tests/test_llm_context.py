# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Unit tests for ``reasoning._llm_context``.

These helpers used to be duplicated across four LLM-adapter modules
(``msted_llm``, ``temporal_localization_llm``, ``qa_llm``,
``causal_linkage_llm``). PR 3 consolidated them into one module so a
sidecar-key change only needs to land in one place. These tests pin the
contract so future contributors can catch behavior drift on the
consolidated module.

See review feedback problem area #4.
"""

from __future__ import annotations

import logging

import pytest
from reasoning._llm_context import (
    WINDOW_END_KEYS,
    WINDOW_START_KEYS,
    block,
    format_windows_block,
    pick_seconds,
    pick_string,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_window_key_tuples_cover_all_known_aliases() -> None:
    """The four-alias set must be preserved verbatim — every existing
    sidecar-emitting stage relies on this exact ordering for fallback.
    Reordering or shrinking the tuples is a behavior change and must be
    intentional + paired with this test update."""
    assert WINDOW_START_KEYS == ("start_s", "start", "start_seconds", "start_time")
    assert WINDOW_END_KEYS == ("end_s", "end", "end_seconds", "end_time")


# ---------------------------------------------------------------------------
# pick_seconds
# ---------------------------------------------------------------------------


class TestPickSeconds:
    def test_returns_first_present_numeric_value(self) -> None:
        win = {"start_s": 3.5, "start": 99.9}
        assert pick_seconds(win, ("start_s", "start")) == 3.5

    def test_falls_through_to_next_key_when_first_missing(self) -> None:
        win = {"start": 7}
        assert pick_seconds(win, ("start_s", "start", "start_seconds")) == 7.0

    def test_returns_none_when_no_key_present(self) -> None:
        assert pick_seconds({"end_s": 1.0}, ("start_s", "start")) is None

    def test_returns_none_for_empty_window(self) -> None:
        assert pick_seconds({}, WINDOW_START_KEYS) is None

    def test_skips_boolean_values(self) -> None:
        """Booleans subclass int in Python, so a sidecar that mistakenly
        stored a flag under a temporal key would otherwise round-trip
        as 0.0 / 1.0 and be silently consumed as a timestamp. The skip
        is the only thing protecting downstream timecode formatting."""
        win = {"start_s": True, "start": 4.2}
        assert pick_seconds(win, ("start_s", "start")) == 4.2

    def test_skips_non_numeric_values(self) -> None:
        win = {"start_s": "3.5", "start": 4.2}
        assert pick_seconds(win, ("start_s", "start")) == 4.2

    def test_returns_none_when_only_invalid_types_present(self) -> None:
        win = {"start_s": "not a number", "start": None}
        assert pick_seconds(win, ("start_s", "start")) is None

    def test_zero_is_a_valid_value(self) -> None:
        """Boundary case: an event starting at t=0 must not collapse to
        ``None`` (which would silently drop the very first window of a
        clip from the rendered LLM context)."""
        assert pick_seconds({"start_s": 0}, ("start_s",)) == 0.0
        assert pick_seconds({"start_s": 0.0}, ("start_s",)) == 0.0


# ---------------------------------------------------------------------------
# pick_string
# ---------------------------------------------------------------------------


class TestPickString:
    def test_returns_first_present_non_empty_string_stripped(self) -> None:
        win = {"caption": "  hello  ", "description": "world"}
        assert pick_string(win, ("caption", "description")) == "hello"

    def test_skips_empty_string(self) -> None:
        win = {"caption": "   ", "description": "fallback"}
        assert pick_string(win, ("caption", "description")) == "fallback"

    def test_skips_non_string_values(self) -> None:
        win = {"caption": 42, "description": "ok"}
        assert pick_string(win, ("caption", "description")) == "ok"

    def test_returns_none_when_none_present(self) -> None:
        assert pick_string({}, ("caption",)) is None

    def test_returns_none_when_all_empty(self) -> None:
        win = {"caption": "", "description": "   "}
        assert pick_string(win, ("caption", "description")) is None


# ---------------------------------------------------------------------------
# block
# ---------------------------------------------------------------------------


class TestBlock:
    def test_renders_header_and_stripped_body(self) -> None:
        out = block("Scene description:", "  some prose  ", fallback="")
        assert out == "Scene description:\nsome prose"

    def test_falls_back_when_body_empty(self) -> None:
        out = block("Scene description:", "", fallback="(no scene-level description provided)")
        assert out == "Scene description:\n(no scene-level description provided)"

    def test_falls_back_when_body_none(self) -> None:
        out = block("Header:", None, fallback="missing")
        assert out == "Header:\nmissing"

    def test_returns_empty_when_body_empty_and_no_fallback(self) -> None:
        """Critical for the LLM-prompt template join pattern
        ``"\\n\\n".join([scene_block, event_block, ...])``: an empty
        return must elide the section entirely (not leave a dangling
        header) so the rendered prompt stays clean when an optional
        signal is absent."""
        assert block("Header:", None, fallback="") == ""
        assert block("Header:", "   ", fallback="") == ""

    def test_falls_back_when_body_is_non_string(self) -> None:
        """Defensive: a sidecar could shove a list / dict / int into a
        text field. The function must not crash; it should fall through
        to the fallback path."""
        out = block("Header:", 42, fallback="fb")  # type: ignore[arg-type]
        assert out == "Header:\nfb"


# ---------------------------------------------------------------------------
# format_windows_block
# ---------------------------------------------------------------------------


class TestFormatWindowsBlock:
    @pytest.fixture
    def log(self) -> logging.Logger:
        return logging.getLogger("test_llm_context")

    def test_renders_timecoded_caption_lines(self, log: logging.Logger) -> None:
        windows = [
            {"start_s": 0.0, "end_s": 5.0, "caption": "first scene"},
            {"start_s": 5.0, "end_s": 10.5, "description": "second scene"},
        ]
        out = format_windows_block(
            windows,
            description_keys=("caption", "description"),
            log=log,
            tag="test",
        )
        # Format is "MM:SS-MM:SS: <caption>"; we don't pin the exact
        # timecode formatting (that's seconds_to_timecode's contract)
        # but we do assert the structural shape.
        assert len(out) == 2
        assert out[0].endswith(": first scene")
        assert out[1].endswith(": second scene")
        assert "-" in out[0] and ": " in out[0]

    def test_skips_non_dict_window(self, log: logging.Logger) -> None:
        windows = [
            "not a dict",  # type: ignore[list-item]
            {"start_s": 1.0, "end_s": 2.0, "caption": "ok"},
        ]
        out = format_windows_block(windows, description_keys=("caption",), log=log, tag="test")
        assert len(out) == 1

    def test_skips_window_without_start_or_end(self, log: logging.Logger) -> None:
        windows = [
            {"end_s": 1.0, "caption": "no start"},
            {"start_s": 1.0, "caption": "no end"},
            {"start_s": 1.0, "end_s": 2.0, "caption": "ok"},
        ]
        out = format_windows_block(windows, description_keys=("caption",), log=log, tag="test")
        assert len(out) == 1

    def test_skips_window_without_caption(self, log: logging.Logger) -> None:
        windows = [
            {"start_s": 1.0, "end_s": 2.0},  # no caption
            {"start_s": 3.0, "end_s": 4.0, "caption": "ok"},
        ]
        out = format_windows_block(windows, description_keys=("caption",), log=log, tag="test")
        assert len(out) == 1

    def test_returns_empty_list_for_empty_input(self, log: logging.Logger) -> None:
        out = format_windows_block([], description_keys=("caption",), log=log, tag="test")
        assert out == []

    def test_uses_tag_in_debug_logs(self, caplog) -> None:
        """The ``tag`` parameter must propagate into every debug-skip
        log message so per-stage logs stay distinguishable when several
        adapters run on the same scene. Regression guard: a previous
        copy-paste hardcoded the tag to ``[msted_llm]`` and we lost the
        ability to tell which stage skipped a window."""
        log = logging.getLogger("test_format_tag")
        windows = [{"end_s": 1.0}]  # missing start → triggers debug log
        with caplog.at_level(logging.DEBUG, logger="test_format_tag"):
            format_windows_block(windows, description_keys=("caption",), log=log, tag="my_stage")
        assert any("[my_stage]" in rec.message for rec in caplog.records), (
            f"expected '[my_stage]' tag in debug logs; got {[r.message for r in caplog.records]}"
        )

    def test_default_start_end_keys_match_module_constants(self, log: logging.Logger) -> None:
        """Sanity: callers omitting ``start_keys`` / ``end_keys`` must
        get the package-level defaults so all four LLM adapters agree
        on the alias set."""
        windows = [{"start_seconds": 1.0, "end_seconds": 2.0, "caption": "ok"}]
        out = format_windows_block(windows, description_keys=("caption",), log=log, tag="test")
        assert len(out) == 1

    def test_caller_can_override_start_end_keys(self, log: logging.Logger) -> None:
        windows = [{"begin": 1.0, "finish": 2.0, "caption": "ok"}]
        out = format_windows_block(
            windows,
            description_keys=("caption",),
            log=log,
            tag="test",
            start_keys=("begin",),
            end_keys=("finish",),
        )
        assert len(out) == 1
