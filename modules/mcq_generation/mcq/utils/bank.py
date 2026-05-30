# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore
from al_utils.io import read_text
from al_utils.text import LETTER_ALPHABET, match_letter_prefix, strip_letter_prefix

_EMBEDDED_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _normalize_option_list(options: List[Any]) -> List[str]:
    """Deduplicate bank options before prompts are generated.

    Duplicate comparison uses the DAFT display value (after stripping optional
    letter prefixes) so banks cannot present two choices that would collapse to
    the same MCQ option later. If every option was letter-prefixed, re-letter the
    remaining choices to keep prompt choices contiguous after dedupe.
    """
    entries: List[tuple[str, str, bool]] = []
    for option in options:
        raw = str(option)
        entries.append((raw, strip_letter_prefix(raw), match_letter_prefix(raw) is not None))

    out_raw: List[str] = []
    out_values: List[str] = []
    seen_values: set[str] = set()
    for raw, value, _has_prefix in entries:
        if value in seen_values:
            continue
        seen_values.add(value)
        out_raw.append(raw.strip())
        out_values.append(value)

    if entries and all(has_prefix for _raw, _value, has_prefix in entries):
        if len(out_values) > len(LETTER_ALPHABET):
            raise SystemExit(
                f"Question bank has {len(out_values)} unique prefixed options; DAFT caps MCQ choices at "
                f"{len(LETTER_ALPHABET)}"
            )
        return [f"{LETTER_ALPHABET[i]}. {value}" for i, value in enumerate(out_values)]
    return out_raw


def _option_answer_aliases(options: List[Any]) -> set[str]:
    aliases: set[str] = set()
    for option in options:
        raw = str(option).strip()
        if not raw:
            continue
        aliases.add(raw)
        value = strip_letter_prefix(raw).strip()
        if value:
            aliases.add(value)
        m = match_letter_prefix(raw)
        if m:
            aliases.add(m.group(1))
    return aliases


def answer_matches_options(answer: str, options: List[Any]) -> bool:
    return answer in _option_answer_aliases(options)


def normalize_bank_options(bank: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-normalized bank with duplicate question options removed."""
    questions = bank.get("questions")
    if not isinstance(questions, list):
        return bank

    normalized = dict(bank)
    normalized_questions: List[Any] = []
    for q in questions:
        if not isinstance(q, dict):
            normalized_questions.append(q)
            continue
        qq = dict(q)
        opts = qq.get("options")
        if isinstance(opts, list):
            qq["options"] = _normalize_option_list(opts)
        normalized_questions.append(qq)
    normalized["questions"] = normalized_questions
    return normalized


def read_bank(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Question bank not found: {p}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        obj = yaml.safe_load(read_text(p))
    else:
        obj = json.loads(read_text(p))
    if not isinstance(obj, dict):
        raise SystemExit(f"Invalid bank payload (expected object): {p}")
    if not isinstance(obj.get("questions"), list) or not obj["questions"]:
        raise SystemExit(f"Invalid bank payload: missing non-empty 'questions' list: {p}")
    return normalize_bank_options(obj)


def wrap_bank_json(bank_payload: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(bank_payload, ensure_ascii=False, indent=2) + "\n```"


def inject_bank_into_template(template_text: str, *, bank_payload: Dict[str, Any]) -> str:
    """
    Replace bank placeholders in a template.

    Supports both placeholders:
    - {{QUESTION_BANK_JSON}}
    - {{QUESTION_BANK_MARKDOWN}}

    (Today both are filled with a fenced JSON block.)
    """
    bank_block = wrap_bank_json(bank_payload)
    fused = str(template_text or "")
    fused = fused.replace("{{QUESTION_BANK_JSON}}", bank_block)
    fused = fused.replace("{{QUESTION_BANK_MARKDOWN}}", bank_block)
    return fused


def extract_include_if_map_from_prompt_text(prompt_text: str) -> Dict[str, Dict[str, str]]:
    """
    Best-effort: extract question-bank include_if rules from a fused MCQ prompt.

    The question bank is embedded as a fenced JSON object with:
      - questions: [{id, question, options, include_if?}, ...]

    Returns:
      {qid: {gate_qid: required_answer, ...}, ...}
    """

    def _brace_match_first_object(s: str) -> Optional[str]:
        start = s.find("{")
        if start == -1:
            return None
        brace = 0
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    return s[start : i + 1]
        return None

    text = str(prompt_text or "")
    idx = 0
    while True:
        start = text.find("```json", idx)
        if start == -1:
            break
        block_start = text.find("\n", start)
        if block_start == -1:
            break
        end = text.find("```", block_start + 1)
        if end == -1:
            break
        payload = text[block_start + 1 : end]
        idx = end + 3

        cand = _brace_match_first_object(payload)
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        qlist = obj.get("questions")
        if not isinstance(qlist, list) or not qlist:
            continue

        include_if: Dict[str, Dict[str, str]] = {}
        for q in qlist:
            if not isinstance(q, dict):
                continue
            qid = q.get("id")
            inc = q.get("include_if")
            if isinstance(qid, str) and qid and isinstance(inc, dict) and inc:
                include_if[qid] = {
                    str(k).strip(): str(v).strip() for k, v in inc.items() if str(k).strip() and str(v).strip()
                }
        return include_if

    return {}


def collect_embedded_bank_from_prompt(prompt_text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort: find the first fenced JSON block that looks like a question bank.

    Expected shape:
      {"questions": [{id, question, options?, include_if?}, ...], ...}
    """
    t = str(prompt_text or "")
    if not t.strip():
        return None

    for m in _EMBEDDED_JSON_BLOCK_RE.finditer(t):
        raw = m.group(1)
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        qs = obj.get("questions")
        if not isinstance(qs, list) or not qs:
            continue
        ok = 0
        for q in qs[:5]:
            if isinstance(q, dict) and str(q.get("id", "")).strip() and str(q.get("question", "")).strip():
                ok += 1
        if ok > 0:
            return normalize_bank_options(obj)
    return None


def include_if_map_from_bank(bank: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not isinstance(bank, dict):
        return out
    for q in bank.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        inc = q.get("include_if")
        if not qid or not isinstance(inc, dict) or not inc:
            continue
        cond: Dict[str, str] = {}
        for k, v in inc.items():
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and vv:
                cond[kk] = vv
        if cond:
            out[qid] = cond
    return out


def options_map_from_bank(bank: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(bank, dict):
        return out
    for q in bank.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        raw_opts = q.get("options") or []
        opts = _normalize_option_list(raw_opts) if isinstance(raw_opts, list) else []
        if qid:
            out[qid] = [str(o) for o in opts]
    return out


def filter_mcq_items_strict(
    items: List[Dict[str, Any]],
    *,
    include_if_map: Dict[str, Dict[str, str]],
    options_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Enforce:
      - answer must be one of options (if options known)
      - include_if conditions must be satisfied (iteratively)
    """
    cur = [it for it in items if isinstance(it, dict) and str(it.get("id") or "").strip()]

    # Drop invalid answers first.
    filtered: List[Dict[str, Any]] = []
    for it in cur:
        qid = str(it.get("id") or "").strip()
        ans = str(it.get("answer") or "").strip()
        if not ans:
            continue
        opts = options_map.get(qid)
        if opts and not answer_matches_options(ans, opts):
            continue
        filtered.append(it)
    cur = filtered

    # Iteratively apply include_if: if parent answers missing or mismatched, drop the conditional question.
    for _ in range(10):  # safety cap; banks are tiny
        answers = {
            str(it.get("id") or "").strip(): str(it.get("answer") or "").strip() for it in cur if isinstance(it, dict)
        }
        next_items: List[Dict[str, Any]] = []
        changed = False
        for it in cur:
            qid = str(it.get("id") or "").strip()
            cond = include_if_map.get(qid)
            if not cond:
                next_items.append(it)
                continue
            ok = True
            for parent_qid, expected in cond.items():
                if answers.get(parent_qid) != expected:
                    ok = False
                    break
            if ok:
                next_items.append(it)
            else:
                changed = True
        cur = next_items
        if not changed:
            break

    return cur
