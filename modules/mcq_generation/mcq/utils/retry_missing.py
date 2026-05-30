# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from mcq_generation.mcq.utils.bank import filter_mcq_items_strict, options_map_from_bank
from mcq_generation.mcq.utils.openai import call_chat_json_with_structured_fallback, get_llm_api_key


def bank_question_map(bank: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for q in bank.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        if not qid:
            continue
        out[qid] = q
    return out


def mcq_by_id(mcq_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for it in mcq_obj.get("mcq") or []:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("id") or "").strip()
        if not qid:
            continue
        by_id[qid] = it
    return by_id


def known_answers_for_retry(
    mcq_obj: Dict[str, Any],
    *,
    include_if_map: Dict[str, Dict[str, str]],
    target_ids: List[str],
) -> Dict[str, str]:
    """Return answers for gate questions needed to retry the given target question ids."""
    target = {str(x).strip() for x in (target_ids or []) if str(x).strip()}
    if not target:
        return {}
    gates: set[str] = set()
    for qid in target:
        for gate_qid in (include_if_map.get(qid) or {}).keys():
            gates.add(str(gate_qid).strip())
    if not gates:
        return {}
    by_id = mcq_by_id(mcq_obj)
    known: Dict[str, str] = {}
    for gid in sorted(gates):
        ans = (by_id.get(gid) or {}).get("answer")
        if isinstance(ans, str) and ans.strip():
            known[gid] = ans.strip()
    return known


def expected_question_ids(
    *,
    bank: Dict[str, Any],
    include_if_map: Dict[str, Dict[str, str]],
    current_mcq_obj: Dict[str, Any],
    max_rounds: int = 3,
) -> List[str]:
    """
    Compute expected question ids under include_if gating, using a small fixpoint iteration.

    Conservative rule:
    - If a gate question is missing, dependent questions are NOT considered expected.
    """
    bank_questions = bank_question_map(bank)
    current_by_id = mcq_by_id(current_mcq_obj)

    expected: set[str] = set()
    for _ in range(max(1, int(max_rounds))):
        changed = False
        for qid in bank_questions.keys():
            inc = include_if_map.get(qid)
            if not inc:
                if qid not in expected:
                    expected.add(qid)
                    changed = True
                continue

            ok = True
            for gate_qid, required in inc.items():
                gate = current_by_id.get(gate_qid)
                gate_ans = gate.get("answer") if isinstance(gate, dict) else None
                if gate_ans != required:
                    ok = False
                    break
            if ok and qid not in expected:
                expected.add(qid)
                changed = True
        if not changed:
            break
    return sorted(expected)


def build_retry_system_prompt(
    *,
    base_prompt: str,
    missing_qs: List[Dict[str, Any]],
    known_answers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Append a strict "missing-only" instruction + a small bank subset to the base MCQ prompt.
    """
    ids = [str(q.get("id") or "").strip() for q in missing_qs if str(q.get("id") or "").strip()]
    bank_subset = [
        {"id": q.get("id"), "question": q.get("question"), "options": q.get("options", [])} for q in missing_qs
    ]
    known = {
        str(k).strip(): str(v).strip() for k, v in (known_answers or {}).items() if str(k).strip() and str(v).strip()
    }
    known_block = ""
    if known:
        known_block = (
            "\nKnown answers from previous pass (treat as ground truth; do not contradict):\n"
            + "```json\n"
            + json.dumps({"answers": known}, ensure_ascii=False, indent=2)
            + "\n```\n"
        )
    suffix = (
        "\n\n"
        "### Retry mode (missing questions only)\n"
        "You MUST return a JSON object with keys: version, video_id, mcq.\n"
        "Only include the following question ids in mcq (no extra questions):\n"
        f"{ids}\n\n"
        + known_block
        + "Use exactly the question text and options below.\n"
        + "- If options is non-empty: choose answer from the provided options (copy exactly).\n"
        + "- If options is empty ([]): include answer as a concise free-text string, using the pattern "
        '"answer": "<free-form description>" and replacing the placeholder with the actual answer.\n'
        "Questions (bank subset):\n"
        + "```json\n"
        + json.dumps({"questions": bank_subset}, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    return (base_prompt or "").rstrip() + suffix


def call_retry_missing_questions(
    *,
    llm_base_url: str,
    llm_model: str,
    llm_structured_output: str,
    llm_max_tokens: int,
    llm_temperature: float,
    timeout: int,
    llm_retries: int,
    llm_retry_backoff_s: float,
    retry_stage: str,
    retry_prompt: str,
    caption: str,
    logger,
) -> Tuple[Optional[Dict[str, Any]], str]:
    messages = [{"role": "system", "content": retry_prompt}, {"role": "user", "content": caption}]
    obj, raw = call_chat_json_with_structured_fallback(
        base_url=llm_base_url,
        model=llm_model,
        messages=messages,
        timeout=int(timeout),
        max_tokens=int(llm_max_tokens),
        temperature=float(llm_temperature),
        top_p=0.9,
        logger=logger,
        retries=int(llm_retries or 0),
        retry_backoff_s=float(llm_retry_backoff_s or 5.0),
        structured_output=str(llm_structured_output or "openai"),
        retry_stage=str(retry_stage or "").strip() or "retry_missing_questions",
        api_key=get_llm_api_key(),
    )
    return (obj if isinstance(obj, dict) else None), raw


def retry_fill_missing_questions(
    *,
    bank: Dict[str, Any],
    include_if_map: Dict[str, Dict[str, str]],
    base_prompt: str,
    caption: str,
    current_mcq_obj: Optional[Dict[str, Any]],
    target_ids: Optional[List[str]] = None,
    known_answers: Optional[Dict[str, str]] = None,
    video_id: str,
    llm_base_url: str,
    llm_model: str,
    llm_structured_output: str,
    llm_max_tokens: int,
    llm_temperature: float,
    timeout: int,
    llm_retries: int,
    llm_retry_backoff_s: float,
    retry_stage: str,
    max_rounds: int = 2,
    logger=None,
) -> Optional[Dict[str, Any]]:
    """
    Best-effort: retry only missing question ids and merge them into current_mcq_obj.

    Returns the updated MCQ object, or None if no parseable MCQ could be produced.
    """
    if not isinstance(bank, dict):
        return current_mcq_obj if isinstance(current_mcq_obj, dict) else None
    if not str(caption or "").strip():
        return current_mcq_obj if isinstance(current_mcq_obj, dict) else None

    bank_questions = bank_question_map(bank)
    if not bank_questions:
        return current_mcq_obj if isinstance(current_mcq_obj, dict) else None

    cur: Dict[str, Any] = dict(current_mcq_obj) if isinstance(current_mcq_obj, dict) else {"version": 2.0, "mcq": []}
    cur["video_id"] = str(video_id or "").strip()
    if "version" not in cur:
        cur["version"] = 2.0

    options_map = options_map_from_bank(bank)

    # attempt 0 is the first retry call after the original full-bank call.
    for attempt in range(max(1, int(max_rounds))):
        current_by_id = mcq_by_id(cur)
        # Retry strategy:
        # - attempt 0: the "first round" already happened before calling this helper (full prompt).
        # - attempt >= 0: when retrying, ask for all unanswered questions (bank - present),
        #   including gated dependents. We'll enforce include_if after merge.
        all_ids = sorted(bank_question_map(bank).keys())
        target = [str(x).strip() for x in (target_ids or []) if str(x).strip()]
        if target:
            target = [qid for qid in target if qid in bank_questions]
        desired = target if target else all_ids
        missing_ids = [qid for qid in desired if qid not in current_by_id]
        if not missing_ids:
            break

        missing_qs = [bank_questions[qid] for qid in missing_ids if qid in bank_questions]
        if not missing_qs:
            break

        retry_prompt = build_retry_system_prompt(
            base_prompt=base_prompt, missing_qs=missing_qs, known_answers=known_answers
        )
        obj, _raw = call_retry_missing_questions(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_structured_output=llm_structured_output,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            timeout=timeout,
            llm_retries=llm_retries,
            llm_retry_backoff_s=llm_retry_backoff_s,
            retry_stage=retry_stage,
            retry_prompt=retry_prompt,
            caption=caption,
            logger=logger,
        )
        if not isinstance(obj, dict):
            break

        returned = [it for it in (obj.get("mcq") or []) if isinstance(it, dict)]
        if not returned:
            break

        merged = dict(current_by_id)
        for it in returned:
            qid = str(it.get("id") or "").strip()
            if qid in missing_ids:
                merged[qid] = it
        # Keep bank order for stability.
        merged_list = [merged[qid] for qid in all_ids if qid in merged]
        # Enforce include_if + answer validity after merge.
        cur["mcq"] = filter_mcq_items_strict(merged_list, include_if_map=include_if_map, options_map=options_map)

    return cur if isinstance(cur.get("mcq"), list) else None
