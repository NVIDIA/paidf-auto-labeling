# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from mcq_generation.mcq.utils.bank import answer_matches_options

AGGREGATION_TYPES = {
    "majority",
    "majority_tie_first",
    "first",
    "any",
    "supermajority",
}


def aggregation_specs_from_bank(bank: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-question aggregation specs from a question bank.

    Supported per-question formats:
      - "aggregation": "majority" | "majority_tie_first" | "first" | "any" | "supermajority"
      - "aggregation": {"type": "...", "threshold": 0.6, "min_yes": 1}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(bank, dict):
        return out
    for q in bank.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        if not qid:
            continue
        spec_raw = q.get("aggregation", None)
        spec = normalize_aggregation_spec(spec_raw)
        if spec is not None:
            out[qid] = spec
    return out


def normalize_aggregation_spec(spec_raw: Any) -> Optional[Dict[str, Any]]:
    if spec_raw is None:
        return None
    if isinstance(spec_raw, str):
        t = spec_raw.strip().lower()
        if not t:
            return None
        if t not in AGGREGATION_TYPES:
            raise SystemExit(
                f"Invalid aggregation type in question bank: {spec_raw!r} (allowed: {sorted(AGGREGATION_TYPES)})"
            )
        return {"type": t}
    if isinstance(spec_raw, dict):
        t = str(spec_raw.get("type") or "").strip().lower()
        if not t:
            raise SystemExit("Invalid aggregation spec in question bank: missing 'type'")
        if t not in AGGREGATION_TYPES:
            raise SystemExit(f"Invalid aggregation type in question bank: {t!r} (allowed: {sorted(AGGREGATION_TYPES)})")
        spec: Dict[str, Any] = {"type": t}
        if "threshold" in spec_raw:
            try:
                spec["threshold"] = float(spec_raw["threshold"])
            except Exception:
                raise SystemExit(f"Invalid aggregation threshold: {spec_raw.get('threshold')!r}") from None
        if "min_yes" in spec_raw:
            try:
                spec["min_yes"] = int(spec_raw["min_yes"])
            except Exception:
                raise SystemExit(f"Invalid aggregation min_yes: {spec_raw.get('min_yes')!r}") from None
        return spec
    raise SystemExit(f"Invalid aggregation spec in question bank: {spec_raw!r} (expected string or object)")


_YN_WORD_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


def _yn_norm(x: Any) -> str | None:
    s = str(x or "").strip()
    if not s:
        return None

    # Accept plain "Yes"/"No" and common prefixed variants like:
    # - "A. Yes", "A) Yes", "(A) Yes", "[A] Yes", "{A} Yes"
    # - "A: Yes", "A - Yes", "A — Yes"
    # - "1. Yes", "2) No"
    # Also tolerate trailing punctuation: "Yes.", "No,".

    t = s.strip()

    # Strip wrapped label prefix: "(A) Yes", "[1] No", "{B} Yes"
    if len(t) >= 4 and t[0] in "([{" and t[2] in ")]}" and t[1].isalnum():
        t = t[3:].lstrip()

    # Strip plain label prefix: "A. Yes", "1) No", "B - Yes"
    if t and t[0].isalnum():
        i = 1
        while i < len(t) and t[i].isspace():
            i += 1
        if i < len(t) and t[i] in ".)-–—:：、":
            i += 1
            while i < len(t) and t[i].isspace():
                i += 1
            t = t[i:]

    # Strip trailing punctuation
    while t and t[-1] in ".,!?;:。，“”\"'":
        t = t[:-1]

    t = t.strip().lower()
    if t in {"yes", "no"}:
        return t

    # Fallback: accept yes/no appearing as a standalone word in a sentence-like string.
    found: set[str] = set()
    for m in _YN_WORD_RE.finditer(t):
        found.add(str(m.group(1)).lower())
        if len(found) > 1:
            return None
    if len(found) == 1:
        return next(iter(found))
    return None


def _binary_yes_no_options(options: List[Any]) -> tuple[str, str] | None:
    """
    Return (yes_option, no_option) if options represent a binary Yes/No choice.

    Supports both plain values ("Yes"/"No") and letter-prefixed variants ("A. Yes"/"B. No").
    """
    yes_opt: str | None = None
    no_opt: str | None = None
    for o in options or []:
        s = str(o or "").strip()
        n = _yn_norm(s)
        if n == "yes":
            yes_opt = s
        elif n == "no":
            no_opt = s
    if yes_opt and no_opt:
        return (yes_opt, no_opt)
    return None


def _is_binary_yes_no(options: List[Any]) -> bool:
    return _binary_yes_no_options(options) is not None


def default_aggregation_spec_for_options(options: List[Any]) -> Dict[str, Any]:
    """
    Default behavior (when the bank does not specify an aggregation rule):
      - Yes/No questions: "supermajority" (reduce false positives)
      - Other multiple-choice: "majority_tie_first"
    """
    if _is_binary_yes_no(options):
        return {"type": "supermajority", "threshold": 0.6, "min_yes": 1}
    return {"type": "majority_tie_first"}


def _majority_with_tie_first(answers: List[str]) -> str:
    counter = Counter(answers)
    if not counter:
        return answers[0]
    most = counter.most_common()
    top = most[0][1]
    tied = {a for a, c in most if c == top}
    if len(tied) == 1:
        return most[0][0]
    for a in answers:
        if a in tied:
            return a
    return answers[0]


def aggregate_answers(*, answers: List[str], options: List[Any], spec: Dict[str, Any]) -> str:
    a = [str(x).strip() for x in answers if str(x).strip()]
    if not a:
        return ""
    t = str(spec.get("type") or "").strip().lower()
    if not t:
        t = str(default_aggregation_spec_for_options(options).get("type"))

    if t == "first":
        return a[0]

    if t == "any":
        yn = _binary_yes_no_options(options)
        if yn is not None:
            yes_opt, no_opt = yn
            if any(_yn_norm(x) == "yes" for x in a):
                return yes_opt
            if any(_yn_norm(x) == "no" for x in a):
                return no_opt
        return a[0]

    if t == "supermajority":
        yn = _binary_yes_no_options(options)
        if yn is not None:
            yes_opt, no_opt = yn
            yes = sum(1 for x in a if _yn_norm(x) == "yes")
            no = sum(1 for x in a if _yn_norm(x) == "no")
            n = yes + no
            if n <= 0:
                return _majority_with_tie_first(a)
            thr = float(spec.get("threshold", 0.6))
            thr = max(0.0, min(1.0, thr))
            min_yes = int(spec.get("min_yes", 1))
            required_yes = max(min_yes, int(math.ceil(thr * n)))
            return yes_opt if yes >= required_yes else no_opt
        return _majority_with_tie_first(a)

    if t in {"majority_tie_first", "majority"}:
        return _majority_with_tie_first(a)

    # Defensive fallback: should be unreachable due to validation.
    return _majority_with_tie_first(a)


def aggregate_window_mcqs(
    window_mcqs: List[Dict[str, Any]],
    *,
    video_id: str,
    include_if_map: Optional[Dict[str, Dict[str, str]]] = None,
    aggregation_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Aggregate per-window MCQ objects into a single per-video MCQ object.

    - Per-question aggregation rules come from aggregation_specs (from question bank).
    - If a question has no explicit rule, use default_aggregation_spec_for_options().
    - include_if_map is applied after aggregation (bank gating).
    """
    if len(window_mcqs) == 1:
        mcq = dict(window_mcqs[0])
        mcq["video_id"] = video_id
        return mcq

    include_if_map = include_if_map or {}
    aggregation_specs = aggregation_specs or {}

    aggregated: Dict[str, Dict[str, Any]] = {}
    for wmcq in window_mcqs:
        for item in wmcq.get("mcq", []) or []:
            if not isinstance(item, dict):
                continue
            qid = item.get("id")
            if not isinstance(qid, str) or not qid:
                continue
            slot = aggregated.setdefault(
                qid,
                {"question": item.get("question", ""), "options": item.get("options", []), "answers": []},
            )
            # Keep the first non-empty options list we see (best-effort).
            if not slot.get("options") and item.get("options"):
                slot["options"] = item.get("options", [])
            slot["answers"].append(item.get("answer"))

    final_items: List[Dict[str, Any]] = []
    for qid in sorted(aggregated.keys()):
        data = aggregated[qid]
        answers = [a for a in data.get("answers", []) if a is not None]
        if not answers:
            continue
        opts = data.get("options", [])
        if not isinstance(opts, list):
            opts = []
        spec = aggregation_specs.get(qid) or default_aggregation_spec_for_options(opts)
        final_answer = aggregate_answers(answers=[str(x) for x in answers], options=opts, spec=spec)
        if not final_answer:
            continue
        final_items.append(
            {
                "id": qid,
                "question": data.get("question", ""),
                "options": opts,
                "answer": final_answer,
            }
        )

    # Enforce option validity + include_if gating after aggregation.
    by_id: Dict[str, Dict[str, Any]] = {
        str(it.get("id")): it for it in final_items if isinstance(it, dict) and it.get("id")
    }
    filtered: List[Dict[str, Any]] = []
    for it in final_items:
        qid = str(it.get("id", "")).strip()
        if not qid:
            continue
        opts = it.get("options", [])
        ans = it.get("answer")
        if isinstance(opts, list) and opts and not answer_matches_options(str(ans or "").strip(), opts):
            continue
        inc = include_if_map.get(qid)
        if inc:
            ok = True
            for gate_qid, required in inc.items():
                gate = by_id.get(gate_qid)
                gate_ans = gate.get("answer") if isinstance(gate, dict) else None
                if gate_ans != required:
                    ok = False
                    break
            if not ok:
                continue
        filtered.append(it)

    used_bank_rules = any(qid in aggregation_specs for qid in aggregated.keys())
    method = "bank_rules" if used_bank_rules else "default_rules"
    return {
        "version": window_mcqs[0].get("version", 2.0),
        "video_id": video_id,
        "mcq": filtered,
        "_aggregation_info": {"num_windows": len(window_mcqs), "aggregation_method": method},
    }
