# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Dict, List, Optional


def fmt_ids(ids: List[str], *, limit: int = 30) -> str:
    """Format a list of question IDs for log messages, truncating if needed."""
    if not ids:
        return "[]"
    shown = ids[: max(0, int(limit))]
    if len(ids) <= len(shown):
        return str(shown)
    return str(shown)[:-1] + f", ... +{len(ids) - len(shown)}]"


def bank_all_ids(bank: Dict[str, Any]) -> List[str]:
    """Return sorted unique question IDs from a bank dict."""
    out: List[str] = []
    for q in bank.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        if qid:
            out.append(qid)
    return sorted(set(out))


def present_ids(mcq_obj: Optional[Dict[str, Any]]) -> List[str]:
    """Return sorted unique question IDs with non-empty answers in a parsed MCQ dict."""
    if not isinstance(mcq_obj, dict):
        return []
    out: List[str] = []
    for it in mcq_obj.get("mcq") or []:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("id") or "").strip()
        ans = str(it.get("answer") or "").strip()
        if qid and ans:
            out.append(qid)
    return sorted(set(out))


def gated_off_ids(
    *,
    all_ids: List[str],
    present_ids: List[str],
    include_if_map: Dict[str, Dict[str, str]],
    answers: Dict[str, str],
) -> List[str]:
    """
    Return IDs that are absent AND gated off by an include_if condition that is not satisfied.
    Used to distinguish "legitimately skipped" from "mistakenly missing" questions.
    """
    present = set(present_ids)
    gated: List[str] = []
    for qid in all_ids:
        if qid in present:
            continue
        cond = include_if_map.get(qid)
        if not cond:
            continue
        ok = True
        for parent_qid, expected in cond.items():
            if answers.get(parent_qid) != expected:
                ok = False
                break
        if not ok:
            gated.append(qid)
    return gated
