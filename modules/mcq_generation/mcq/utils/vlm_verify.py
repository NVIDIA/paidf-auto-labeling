# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from mcq_generation.mcq.utils.openai import call_chat_object_with_structured_fallback, get_vlm_api_key

VLM_VERIFY_GUIDED_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string"},
                    "reasoning_trace": {"type": "string"},
                    "suggested_answer": {"type": "string"},
                    # Must echo the input CURRENT_ANSWER, otherwise the verify output is discarded.
                    "echo_current_answer": {"type": "string"},
                },
                "required": ["id", "verdict", "reasoning_trace", "suggested_answer", "echo_current_answer"],
            },
        }
    },
    "required": ["verifications"],
}


def _normalize_verdict(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in {"supported", "support", "yes", "correct", "matched", "match", "confirmed"}:
        return "supported"
    if s in {
        "not_supported",
        "unsupported",
        "not support",
        "no",
        "incorrect",
        "mismatch",
        "conflict",
        "contradicted",
    }:
        return "not_supported"
    if s in {"uncertain", "unknown", "cannot_determine", "not_sure", "insufficient_evidence"}:
        return "uncertain"
    return "uncertain"


def _raw_verify_items_list(verify_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Best-effort extraction of the raw per-question verification items from a verifier response object.
    """
    raw_items = verify_obj.get("verifications")
    if not isinstance(raw_items, list):
        raw_items = verify_obj.get("items")
    if not isinstance(raw_items, list):
        raw_items = verify_obj.get("results")
    if not isinstance(raw_items, list):
        raw_items = verify_obj.get("mcq")
    if not isinstance(raw_items, list):
        by_id_obj = verify_obj.get("by_id")
        if isinstance(by_id_obj, dict):
            raw_items = []
            for qid, item in by_id_obj.items():
                if not isinstance(item, dict):
                    continue
                merged = dict(item)
                merged.setdefault("id", str(qid))
                raw_items.append(merged)
    if not isinstance(raw_items, list):
        return []
    return [it for it in raw_items if isinstance(it, dict)]


def _expected_current_by_id(fallback_mcq: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in fallback_mcq:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("id") or "").strip()
        if not qid:
            continue
        ans = it.get("answer") or it.get("CURRENT_ANSWER") or it.get("current_answer") or it.get("current") or ""
        if isinstance(ans, str) and ans.strip():
            out[qid] = ans.strip()
    return out


def _render_verify_prompt(template_text: str, payload: Dict[str, Any]) -> str:
    payload_text = json.dumps(payload, ensure_ascii=True)
    return (
        template_text.replace("{current_mcq_answers}", payload_text)
        if "{current_mcq_answers}" in template_text
        else template_text.rstrip() + f"\n\nCurrent MCQ answers:\n{payload_text}"
    )


def render_vlm_verify_prompt_template(*, prompt_template: str, apply_corrections: bool) -> str:
    """Render policy placeholders while keeping the per-window answer payload placeholder intact."""
    rendered = prompt_template
    for key, value in _correction_policy_vars(apply_corrections=bool(apply_corrections)).items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _correction_policy_vars(*, apply_corrections: bool) -> Dict[str, str]:
    if not apply_corrections:
        return {
            "correction_policy": (
                "Correction policy: corrections are disabled.\n"
                "- This verification pass must not propose corrections and must not re-answer.\n"
                "- NEVER output not_supported.\n"
                "- NEVER change suggested_answer."
            ),
            "not_supported_rule": "",
            "uncertain_rule": (
                "2) uncertain: evidence is insufficient/ambiguous.\n"
                "   - suggested_answer MUST be exactly the same as CURRENT_ANSWER (do not guess-change)."
            ),
            "not_supported_constraints": "- NEVER output not_supported.",
            "not_supported_reasoning": "",
            "domain_safety_rules": "",
            "verdict_values": "supported, uncertain",
        }

    return {
        "correction_policy": (
            "Correction policy: corrections are enabled.\n"
            "- Only in rare cases, when you are very confident the CURRENT_ANSWER is clearly wrong "
            "AND you have explicit counter-evidence in frames, you may propose a correction.\n"
            "- Only if you can DISPROVE CURRENT_ANSWER with positive, checkable visual evidence, consider not_supported.\n"
            "- Only after setting suggested_answer = CURRENT_ANSWER first, if and only if you choose "
            "verdict=not_supported, you may replace suggested_answer with a DIFFERENT valid option."
        ),
        "not_supported_rule": (
            "2) not_supported: evidence contradicts CURRENT_ANSWER and supports another option.\n"
            "   - suggested_answer MUST be a different valid option (this is the only case where changing is appropriate)."
        ),
        "uncertain_rule": (
            "3) uncertain: evidence is insufficient/ambiguous.\n"
            "   - suggested_answer MUST be exactly the same as CURRENT_ANSWER (do not guess-change)."
        ),
        "not_supported_constraints": "- NEVER output not_supported when suggested_answer equals CURRENT_ANSWER.",
        "not_supported_reasoning": (
            "  - not_supported: reasoning_trace MUST explain what in the frames contradicts CURRENT_ANSWER "
            "and what supports the suggested_answer."
        ),
        "domain_safety_rules": (
            '- For binary accident/incident questions (e.g., "Is a traffic accident happening in this scene?"):\n'
            '  - NEVER use only the absence of an event (e.g., "no visible crash/damage") as the sole reason to mark not_supported.\n'
            "  - To change Yes->No, you MUST cite at least two independent, checkable positive observations from frames that\n"
            "    support a non-incident state (not merely that something is not visible).\n"
            "  - If you cannot meet this bar, use verdict=uncertain and keep suggested_answer equal to CURRENT_ANSWER.\n"
            "- If (verdict=not_supported) AND (suggested_answer differs from CURRENT_ANSWER), reasoning_trace MUST contain\n"
            "  an explicit evidence cue and be directly checkable from frames. Use the pattern:\n"
            '  "EVIDENCE: <what is visible in frames that contradicts CURRENT_ANSWER> + <what is visible in frames that supports suggested_answer>".\n'
            "  The sentence MUST include at least one of these tokens: frame, frames, visible, see, shows.\n"
            "  If you cannot provide this, DO NOT change the answer; use verdict=uncertain and keep suggested_answer equal to CURRENT_ANSWER."
        ),
        "verdict_values": "supported, not_supported, uncertain",
    }


def build_vlm_verify_prompt(
    mcq_items: List[Dict[str, Any]],
    *,
    prompt_template: str,
    apply_corrections: bool = False,
) -> Tuple[str, List[str]]:
    """
    Build the VLM verification prompt for a list of MCQ items.

    Notes:
    - The verifier MUST echo CURRENT_ANSWER in `echo_current_answer` so we can verify it read the input.
    - `echo_current_answer` is not persisted in outputs; it's only used for gating.
    """
    # Keep the payload minimal and explicit so the verifier treats the CURRENT answer as an input
    # to be verified (not something to be recomputed).
    slim_items: List[Dict[str, Any]] = []
    included_ids: List[str] = []
    for it in mcq_items:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("id") or "").strip()
        if not qid:
            continue
        question = it.get("question") or it.get("prompt") or it.get("text") or ""
        options = it.get("options") or it.get("choices") or it.get("candidates") or []
        # Open questions (options=[]) are not supported by the current verifier contract
        # (it assumes a fixed option set for "suggested_answer"). Skip them to avoid
        # forced invalid outputs and echo-gate discards.
        if not isinstance(options, list) or len(options) == 0:
            continue
        current_answer = it.get("answer") or it.get("current_answer") or it.get("current") or ""
        included_ids.append(qid)
        slim_items.append(
            {
                "id": qid,
                "question": question,
                "options": options,
                "CURRENT_ANSWER": current_answer,
            }
        )
    rendered_template = render_vlm_verify_prompt_template(
        prompt_template=prompt_template,
        apply_corrections=bool(apply_corrections),
    )
    rendered = _render_verify_prompt(rendered_template, {"mcq": slim_items})
    return rendered, included_ids


def verify_echo_current_answer_or_discard(
    verify_obj: Dict[str, Any],
    *,
    fallback_mcq: List[Dict[str, Any]],
    expected_ids: List[str] | None = None,
    video_id: str,
    w_idx: int,
    logger: logging.Logger,
) -> bool:
    """
    Return True if the verifier output should be discarded because the model did not echo CURRENT_ANSWER correctly.
    """
    expected_by_id = _expected_current_by_id(fallback_mcq)
    if expected_ids is not None:
        expected_set = {str(x).strip() for x in expected_ids if str(x).strip()}
        expected_by_id = {k: v for k, v in expected_by_id.items() if k in expected_set}
    if not expected_by_id:
        return False

    echo_by_id: Dict[str, str] = {}
    for it in _raw_verify_items_list(verify_obj):
        qid = str(it.get("id") or it.get("qid") or it.get("question_id") or "").strip()
        if not qid:
            continue
        echo = str(it.get("echo_current_answer") or "").strip()
        if echo:
            echo_by_id[qid] = echo

    mismatches: List[Tuple[str, str, str]] = []
    for qid, expected in expected_by_id.items():
        echoed = str(echo_by_id.get(qid, "")).strip()
        if (not echoed) or echoed != expected:
            mismatches.append((qid, expected, echoed))

    if not mismatches:
        logger.info(
            "VLM verify echo_current_answer OK (clip=%s window=%d items=%d)",
            video_id,
            int(w_idx),
            len(expected_by_id),
        )
        return False

    logger.warning(
        "VLM verify discarded due to CURRENT_ANSWER echo mismatch (clip=%s window=%d mismatches=%d sample=%s)",
        video_id,
        int(w_idx),
        len(mismatches),
        mismatches[: min(5, len(mismatches))],
    )
    return True


def parse_vlm_verify_items(verify_obj: Dict[str, Any], *, fallback_mcq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current_by_id = _expected_current_by_id(fallback_mcq)
    raw_items = _raw_verify_items_list(verify_obj)

    out: List[Dict[str, Any]] = []
    for it in raw_items:
        qid = str(it.get("id") or it.get("qid") or it.get("question_id") or "").strip()
        if not qid:
            continue
        reasoning = str(
            it.get("reasoning_trace") or it.get("reasoning") or it.get("rationale") or it.get("evidence") or ""
        ).strip()
        suggested = str(
            it.get("suggested_answer") or it.get("suggestion") or it.get("correct_answer") or it.get("answer") or ""
        ).strip()
        out.append(
            {
                "id": qid,
                "verdict": _normalize_verdict(it.get("verdict") or it.get("label") or it.get("status")),
                "reasoning_trace": reasoning,
                "suggested_answer": suggested,
                "current_answer": current_by_id.get(qid, ""),
            }
        )
    if out:
        return out

    # Fallback: preserve IDs so metadata windows still have a stable verify shape even on parse failures.
    fallback: List[Dict[str, Any]] = []
    for q in fallback_mcq:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        if not qid:
            continue
        fallback.append(
            {
                "id": qid,
                "verdict": "uncertain",
                "reasoning_trace": "unparsed_verify_output",
                "suggested_answer": str(q.get("answer") or "").strip(),
                "current_answer": str(q.get("answer") or "").strip(),
            }
        )
    return fallback


def attach_reasoning_traces_from_verify(
    mcq_obj: Dict[str, Any], *, verify_windows_out: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Best-effort: attach one `reasoning_trace` per question to the final per-video mcq output.

    We pick a trace whose `suggested_answer` matches the final `answer` when possible
    (supported > uncertain > not_supported), and fall back to the earliest available trace.
    """
    if not isinstance(mcq_obj, dict):
        return mcq_obj
    mcq_items = mcq_obj.get("mcq")
    if not isinstance(mcq_items, list) or not mcq_items:
        return mcq_obj
    if not isinstance(verify_windows_out, list) or not verify_windows_out:
        return mcq_obj

    target_answer_by_qid: Dict[str, str] = {}
    for it in mcq_items:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("id") or "").strip()
        ans = it.get("answer")
        if qid and isinstance(ans, str) and ans.strip():
            target_answer_by_qid[qid] = ans.strip()

    def verdict_rank(v: str) -> int:
        if v == "supported":
            return 0
        if v == "uncertain":
            return 1
        if v == "not_supported":
            return 2
        return 3

    candidates: Dict[str, List[Tuple[int, int, str, str, str]]] = {}
    # tuple: (score, window_index, reasoning_trace, verdict, suggested_answer)
    for w in verify_windows_out:
        if not isinstance(w, dict):
            continue
        widx = int(w.get("window_index") or 0)
        vobj = w.get("vlm_verify")
        if not isinstance(vobj, dict):
            continue
        if str(vobj.get("status") or "") != "ok":
            continue
        for it in vobj.get("verifications") or []:
            if not isinstance(it, dict):
                continue
            qid = str(it.get("id") or "").strip()
            if not qid:
                continue
            rt = str(it.get("reasoning_trace") or "").strip()
            if not rt:
                continue
            sugg = str(it.get("suggested_answer") or "").strip()
            verdict = str(it.get("verdict") or "").strip()
            target = target_answer_by_qid.get(qid, "")
            match = bool(target) and bool(sugg) and (sugg == target)
            score = (0 if match else 10) + verdict_rank(verdict)
            candidates.setdefault(qid, []).append((score, widx, rt, verdict, sugg))

    if not candidates:
        return mcq_obj

    chosen: Dict[str, str] = {}
    for qid, xs in candidates.items():
        target = target_answer_by_qid.get(qid, "")
        # Prefer traces that (a) agree with final answer and (b) claim support for it.
        # This avoids injecting verifier text that contradicts the final answer.
        agree_supported = [t for t in xs if bool(target) and t[4] == target and t[3] == "supported"]
        agree_uncertain = [t for t in xs if bool(target) and t[4] == target and t[3] == "uncertain"]
        pool = agree_supported or agree_uncertain
        if pool:
            xs2 = sorted(pool, key=lambda t: (t[0], t[1]))
            chosen[qid] = xs2[0][2]
        else:
            # No safe agreeing trace exists; emit a non-contradictory placeholder pointing to the verify sidecar.
            chosen[qid] = "verifier_no_safe_reasoning_trace; see sidecars/mcq.vlm_verify.json"

    for it in mcq_items:
        if not isinstance(it, dict):
            continue
        if isinstance(it.get("reasoning_trace"), str) and it.get("reasoning_trace").strip():
            continue
        qid = str(it.get("id") or "").strip()
        rt = chosen.get(qid)
        if qid and isinstance(rt, str) and rt.strip():
            it["reasoning_trace"] = rt
    return mcq_obj


def run_window_vlm_verify(
    *,
    messages: List[Dict[str, Any]],
    fallback_mcq: List[Dict[str, Any]],
    verify_expected_ids: List[str],
    mcq_obj: Dict[str, Any],
    video_id: str,
    w_idx: int,
    vlm_base_url: str,
    vlm_model: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    structured_output: str,
    apply_corrections: bool,
    logger: logging.Logger,
    win_errors: List[str],
    retries: int = 3,
    retry_backoff_s: float = 5.0,
    retry_stage: str = "vlm_verify",
) -> Dict[str, Any]:
    """Execute a VLM verify API call and parse the result.

    Returns a dict with:
      - corrected_obj: the (possibly corrected) MCQ object
      - verify_items: list of per-question verify results
      - corrected_count: number of answers corrected
      - verify_status: "ok", "discarded_current_echo_mismatch", or "call_failed"
      - error: error class name (only present if verify_status == "call_failed")
    """
    try:
        verify_obj, _verify_text = call_chat_object_with_structured_fallback(
            base_url=vlm_base_url,
            model=vlm_model,
            messages=messages,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            logger=logger,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            structured_output=structured_output,
            guided_json_schema=VLM_VERIFY_GUIDED_JSON_SCHEMA,
            invalid_fallback="auto",
            retry_stage=retry_stage,
            api_key=get_vlm_api_key(),
        )
        parsed = verify_obj if isinstance(verify_obj, dict) else {}
        discard = verify_echo_current_answer_or_discard(
            parsed,
            fallback_mcq=fallback_mcq,
            expected_ids=verify_expected_ids,
            video_id=video_id,
            w_idx=w_idx,
            logger=logger,
        )
        verify_items = parse_vlm_verify_items(parsed, fallback_mcq=fallback_mcq)
        if discard:
            return {
                "corrected_obj": dict(mcq_obj),
                "verify_items": verify_items,
                "corrected_count": 0,
                "verify_status": "discarded_current_echo_mismatch",
            }
        if apply_corrections:
            corrected_obj, corrected_n = apply_vlm_verify_corrections(mcq_obj, verify_items)
        else:
            corrected_obj, corrected_n = dict(mcq_obj), 0
        return {
            "corrected_obj": corrected_obj,
            "verify_items": verify_items,
            "corrected_count": int(corrected_n),
            "verify_status": "ok",
        }
    except Exception as e:
        win_errors.append(f"vlm_verify_failed:{e.__class__.__name__}")
        logger.exception("VLM verify failed (clip=%s window=%d)", video_id, w_idx)
        return {
            "corrected_obj": dict(mcq_obj),
            "verify_items": [],
            "corrected_count": 0,
            "verify_status": "call_failed",
            "error": f"{e.__class__.__name__}",
        }


def apply_vlm_verify_corrections(
    mcq_obj: Dict[str, Any], verify_items: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    corrected = dict(mcq_obj)
    mcq_items = [dict(it) for it in (mcq_obj.get("mcq") or []) if isinstance(it, dict)]
    corrected["mcq"] = mcq_items
    by_id: Dict[str, Dict[str, Any]] = {}
    for q in mcq_items:
        qid = str(q.get("id") or "").strip()
        if qid:
            by_id[qid] = q

    changed = 0
    for it in verify_items:
        qid = str(it.get("id") or "").strip()
        if not qid or qid not in by_id:
            continue
        q = by_id[qid]
        current_answer = str(q.get("answer") or "").strip()
        suggested = str(it.get("suggested_answer") or "").strip()
        options = [str(x).strip() for x in (q.get("options") or []) if str(x).strip()]
        verdict = _normalize_verdict(it.get("verdict"))
        if verdict != "not_supported":
            continue
        if suggested and suggested in options and suggested != current_answer:
            q["answer"] = suggested
            changed += 1
    return corrected, changed
