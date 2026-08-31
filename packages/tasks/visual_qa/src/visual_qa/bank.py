# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Question-bank and aggregation helpers for visual QA sidecars."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LETTER_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
AGGREGATION_TYPES = frozenset({"majority", "majority_tie_first", "first", "any", "supermajority"})

_PREFIX_RE = re.compile(r"^\s*[\(\[\{]?([A-Za-z0-9])[\)\]\}]?\s*[.)\-:：、]\s*(.+?)\s*$")


@dataclass(frozen=True)
class QuestionSpec:
    """Normalized question-bank entry."""

    qid: str
    question: str
    options: tuple[str, ...] = ()
    include_if: dict[str, str] = field(default_factory=dict)
    aggregation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionBank:
    """Normalized top-level ``questions`` bank."""

    questions: dict[str, QuestionSpec]
    source_path: Path | None = None

    @property
    def ordered_ids(self) -> list[str]:
        return list(self.questions)


def load_question_bank(path: Path | str | None) -> QuestionBank:
    """Load a JSON question bank.

    The visual QA stage consumes the top-level ``questions`` array. Per-task
    sections such as ``open_qa`` remain DAFT-export concerns.
    """
    if path is None:
        return QuestionBank(questions={})
    bank_path = Path(path)
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"question bank must be a JSON object: {bank_path}")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError(f"question bank missing top-level questions list: {bank_path}")

    questions: dict[str, QuestionSpec] = {}
    for idx, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            raise ValueError(
                f"question bank entry {idx} must be an object, got {type(raw).__name__}: {raw!r}"
            )
        qid = str(raw.get("id") or "").strip()
        question = str(raw.get("question") or raw.get("text") or "").strip()
        if not qid or not question:
            raise ValueError(f"question bank entry {idx} requires non-empty id and question")
        if qid in questions:
            raise ValueError(f"question bank contains duplicate id {qid!r} at entry {idx}")
        options = tuple(_clean_option_list(raw.get("options")))
        include_if = _clean_include_if(raw.get("include_if"))
        aggregation = _normalize_aggregation(raw.get("aggregation"))
        questions[qid] = QuestionSpec(
            qid=qid,
            question=question,
            options=options,
            include_if=include_if,
            aggregation=aggregation,
        )
    return QuestionBank(questions=questions, source_path=bank_path)


def normalize_items(
    raw_items: list[Any],
    *,
    bank: QuestionBank,
    strict_answers: bool,
    open_ended: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize raw QA items into the sidecar item contract."""
    out: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            errors.append(f"item[{idx}] skipped: expected object")
            continue
        qid = str(raw.get("id") or raw.get("question_id") or "").strip()
        spec = bank.questions.get(qid)
        question = _clean_string(raw.get("question")) or (spec.question if spec else "")
        if not qid or not question:
            errors.append(f"item[{idx}] skipped: missing id or question")
            continue

        raw_options = raw.get("options")
        if isinstance(raw_options, list):
            options = _clean_option_list(raw_options)
        elif spec is not None:
            options = list(spec.options)
        else:
            options = []

        raw_answer = raw.get("answer")
        answer = normalize_answer(raw_answer, options)
        if answer is None:
            errors.append(f"item[{idx}] skipped: empty answer for {qid}")
            continue
        if strict_answers and options and not answer_matches_options(answer, options):
            errors.append(f"item[{idx}] skipped: answer not in options for {qid}")
            continue

        normalized: dict[str, Any] = {
            "id": qid,
            "question": question,
            "options": options,
            "answer": answer,
        }
        reasoning = _clean_string(raw.get("reasoning_trace")) or _clean_string(raw.get("reasoning"))
        if reasoning is not None:
            normalized["reasoning_trace"] = reasoning

        if open_ended and options:
            normalized["answer"] = strip_option_prefix(answer)
            normalized["options"] = []

        out.append(normalized)

    return out, errors


def aggregate_window_items(
    windows: list[list[dict[str, Any]]],
    *,
    bank: QuestionBank,
    media_id: str,
    aggregate: bool,
) -> dict[str, Any]:
    """Aggregate per-window answers and return a visual-QA items sidecar."""
    if not windows:
        items: list[dict[str, Any]] = []
    elif not aggregate:
        items = _apply_include_if(list(windows[0]), bank=bank)
    elif len(windows) == 1:
        items = _apply_include_if(list(windows[0]), bank=bank)
    else:
        items = _aggregate_multi_window(windows, bank=bank)

    return {
        "schema_version": "1",
        "media_id": media_id,
        "items": items,
        "aggregation": {
            "enabled": bool(aggregate),
            "num_windows": len(windows),
            "method": "bank_rules" if bank.questions else "input_order",
        },
    }


def normalize_answer(answer: Any, options: list[str]) -> str | None:
    """Resolve answer text against options while preserving a DAFT-compatible surface."""
    ans = _clean_string(answer)
    if ans is None:
        return None
    if not options:
        return ans

    if len(ans) == 1 and ans.isalpha():
        letter = ans.upper() if len(options) <= 26 else ans
        letter_idx = LETTER_ALPHABET.find(letter)
        if 0 <= letter_idx < len(options):
            return options[letter_idx]

    ans_stripped = strip_option_prefix(ans)
    for option in options:
        if ans == option or ans_stripped == strip_option_prefix(option):
            return option
    return ans


def answer_matches_options(answer: str, options: list[str]) -> bool:
    """True when ``answer`` resolves to one of ``options``."""
    if not options:
        return bool(answer.strip())
    normalized = normalize_answer(answer, options)
    if normalized is None:
        return False
    norm_display = strip_option_prefix(normalized)
    return any(
        normalized == option or norm_display == strip_option_prefix(option) for option in options
    )


def strip_option_prefix(value: Any) -> str:
    text = str(value or "").strip()
    match = _PREFIX_RE.match(text)
    return match.group(2).strip() if match else text


def _aggregate_multi_window(
    windows: list[list[dict[str, Any]]], *, bank: QuestionBank
) -> list[dict[str, Any]]:
    slots: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for window in windows:
        for item in window:
            qid = str(item.get("id") or "").strip()
            if not qid:
                continue
            if qid not in slots:
                order.append(qid)
                slots[qid] = {
                    "question": item.get("question", ""),
                    "options": item.get("options", []),
                    "answers": [],
                    "reasoning_traces": [],
                }
            if not slots[qid].get("options") and item.get("options"):
                slots[qid]["options"] = item.get("options")
            slots[qid]["answers"].append(item.get("answer"))
            slots[qid]["reasoning_traces"].append(item.get("reasoning_trace"))

    ordered = [qid for qid in bank.ordered_ids if qid in slots]
    seen = set(ordered)
    for qid in order:
        if qid not in seen:
            ordered.append(qid)
            seen.add(qid)

    by_id: dict[str, dict[str, Any]] = {}
    for qid in ordered:
        slot = slots[qid]
        spec = bank.questions.get(qid)
        options = _clean_option_list(slot.get("options")) or (list(spec.options) if spec else [])
        aggregation = (
            spec.aggregation
            if spec is not None and spec.aggregation
            else _default_aggregation(options)
        )
        answer, reasoning_trace = _aggregate_answer_with_reasoning(
            answers=_answer_reasoning_pairs(
                answers=slot.get("answers", []),
                reasoning_traces=slot.get("reasoning_traces", []),
            ),
            options=options,
            spec=aggregation,
        )
        if not answer:
            continue
        aggregated_item: dict[str, Any] = {
            "id": qid,
            "question": str(slot.get("question") or (spec.question if spec else "")).strip(),
            "options": options,
            "answer": answer,
        }
        reasoning = _clean_string(reasoning_trace)
        if reasoning is not None:
            aggregated_item["reasoning_trace"] = reasoning
        by_id[qid] = aggregated_item

    final: list[dict[str, Any]] = []
    for qid in ordered:
        final_item = by_id.get(qid)
        if final_item is None:
            continue
        spec = bank.questions.get(qid)
        if spec is not None and spec.include_if:
            if not _include_if_satisfied(spec.include_if, by_id=by_id, bank=bank):
                continue
        final.append(final_item)
    return final


def _apply_include_if(items: list[dict[str, Any]], *, bank: QuestionBank) -> list[dict[str, Any]]:
    if not bank.questions:
        return items
    by_id = {
        str(item.get("id")): item
        for item in items
        if isinstance(item.get("id"), str) and str(item.get("id")).strip()
    }
    out: list[dict[str, Any]] = []
    for item in items:
        qid = str(item.get("id") or "").strip()
        spec = bank.questions.get(qid)
        if spec is not None and spec.include_if:
            if not _include_if_satisfied(spec.include_if, by_id=by_id, bank=bank):
                continue
        out.append(item)
    return out


def _aggregate_answers(*, answers: list[str], options: list[str], spec: dict[str, Any]) -> str:
    answer, _reasoning_trace = _aggregate_answer_with_reasoning(
        answers=[(answer, None) for answer in answers],
        options=options,
        spec=spec,
    )
    return answer


def _aggregate_answer_with_reasoning(
    *,
    answers: list[tuple[str, Any]],
    options: list[str],
    spec: dict[str, Any],
) -> tuple[str, Any | None]:
    normalized = [(normalize_answer(answer, options), reasoning) for answer, reasoning in answers]
    cleaned = [(answer, reasoning) for answer, reasoning in normalized if answer]
    if not cleaned:
        return "", None

    agg_type = str(spec.get("type") or "").strip().lower()
    if agg_type == "first":
        return cleaned[0]
    if agg_type == "any":
        yn = _yes_no_options(options)
        if yn is not None:
            yes_option, no_option = yn
            yes_match = _first_yes_no_match(
                cleaned, selected_norm="yes", selected_answer=yes_option
            )
            if yes_match is not None:
                return yes_match
            no_match = _first_yes_no_match(
                cleaned,
                selected_norm="no",
                selected_answer=no_option,
            )
            if no_match is not None:
                return no_match
            return no_option, None
        return cleaned[0]
    if agg_type == "supermajority":
        yn = _yes_no_options(options)
        if yn is not None:
            yes_option, no_option = yn
            yes_count = sum(1 for value, _reasoning in cleaned if _yes_no_norm(value) == "yes")
            no_count = sum(1 for value, _reasoning in cleaned if _yes_no_norm(value) == "no")
            total = yes_count + no_count
            if total > 0:
                threshold = min(1.0, max(0.0, float(spec.get("threshold", 0.6))))
                min_yes = max(1, int(spec.get("min_yes", 1)))
                required_yes = max(min_yes, math.ceil(threshold * total))
                selected = yes_option if yes_count >= required_yes else no_option
                selected_match = _first_yes_no_match(
                    cleaned,
                    selected_norm="yes" if yes_count >= required_yes else "no",
                    selected_answer=selected,
                )
                return selected_match if selected_match is not None else (selected, None)
        return _majority_tie_first(cleaned)
    return _majority_tie_first(cleaned)


def _answer_reasoning_pairs(*, answers: Any, reasoning_traces: Any) -> list[tuple[str, Any]]:
    if not isinstance(answers, list):
        return []
    traces = reasoning_traces if isinstance(reasoning_traces, list) else []
    return [
        (str(answer), traces[index] if index < len(traces) else None)
        for index, answer in enumerate(answers)
        if answer is not None
    ]


def _majority_tie_first(answers: list[tuple[str, Any]]) -> tuple[str, Any | None]:
    counts = Counter(answer for answer, _reasoning in answers)
    top_count = counts.most_common(1)[0][1]
    tied = {answer for answer, count in counts.items() if count == top_count}
    for answer, reasoning in answers:
        if answer in tied:
            return answer, reasoning
    return answers[0]


def _first_yes_no_match(
    answers: list[tuple[str, Any]],
    *,
    selected_norm: str,
    selected_answer: str,
) -> tuple[str, Any | None] | None:
    for answer, reasoning in answers:
        if _yes_no_norm(answer) == selected_norm:
            return selected_answer, reasoning
    return None


def _include_if_satisfied(
    include_if: dict[str, str],
    *,
    by_id: dict[str, dict[str, Any]],
    bank: QuestionBank,
) -> bool:
    for gate_id, required in include_if.items():
        gate = by_id.get(gate_id)
        if gate is None:
            return False
        gate_spec = bank.questions.get(gate_id)
        options = (
            list(gate_spec.options)
            if gate_spec is not None
            else _clean_option_list(gate.get("options"))
        )
        required_normalized = normalize_answer(required, options) or required
        gate_answer = normalize_answer(gate.get("answer"), options) or str(gate.get("answer") or "")
        if gate_answer != required_normalized:
            return False
    return True


def _normalize_aggregation(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        agg_type = raw.strip().lower()
        if agg_type not in AGGREGATION_TYPES:
            raise ValueError(f"invalid aggregation type: {raw!r}")
        return {"type": agg_type}
    if isinstance(raw, dict):
        agg_type = str(raw.get("type") or "").strip().lower()
        if agg_type not in AGGREGATION_TYPES:
            raise ValueError(f"invalid aggregation type: {agg_type!r}")
        out: dict[str, Any] = {"type": agg_type}
        if "threshold" in raw:
            out["threshold"] = float(raw["threshold"])
        if "min_yes" in raw:
            out["min_yes"] = int(raw["min_yes"])
        return out
    raise ValueError(f"invalid aggregation spec: {raw!r}")


def _default_aggregation(options: list[str]) -> dict[str, Any]:
    if _yes_no_options(options) is not None:
        return {"type": "supermajority", "threshold": 0.6, "min_yes": 1}
    return {"type": "majority_tie_first"}


def _yes_no_options(options: list[str]) -> tuple[str, str] | None:
    yes: str | None = None
    no: str | None = None
    for option in options:
        value = _yes_no_norm(option)
        if value == "yes":
            yes = option
        elif value == "no":
            no = option
    return (yes, no) if yes is not None and no is not None else None


def _yes_no_norm(value: Any) -> str | None:
    stripped = strip_option_prefix(value).strip().strip(".,!?;:").lower()
    if stripped == "yes":
        return "yes"
    if stripped == "no":
        return "no"
    return None


def _clean_include_if(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(key).strip(): str(value).strip()
        for key, value in raw.items()
        if str(key).strip() and str(value).strip()
    }


def _clean_option_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for option in raw:
        value = str(option or "").strip()
        display = strip_option_prefix(value)
        if not value or display in seen:
            continue
        seen.add(display)
        out.append(value)
    return out


def _clean_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


__all__ = [
    "QuestionBank",
    "QuestionSpec",
    "aggregate_window_items",
    "answer_matches_options",
    "load_question_bank",
    "normalize_answer",
    "normalize_items",
    "strip_option_prefix",
]
