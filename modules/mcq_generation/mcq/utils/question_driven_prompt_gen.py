# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from al_utils.io import read_text, sha256_text
from mcq_generation.mcq.utils.bank import read_bank
from mcq_generation.mcq.utils.openai import call_chat_raw, get_llm_api_key


def _bank_minimal_view_for_scene_prompt(bank: Dict[str, Any]) -> Dict[str, Any]:
    out_questions: List[Dict[str, Any]] = []
    for q in bank.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        qtext = str(q.get("question") or "").strip()
        if not qid or not qtext:
            continue
        item: Dict[str, Any] = {"id": qid, "question": qtext}
        inc = q.get("include_if")
        if isinstance(inc, dict) and inc:
            item["include_if"] = {str(k).strip(): str(v).strip() for k, v in inc.items() if str(k).strip()}
        out_questions.append(item)
    return {"questions": out_questions}


def _bank_minimal_view_for_mapper_rules(bank: Dict[str, Any]) -> Dict[str, Any]:
    out_questions: List[Dict[str, Any]] = []
    for q in bank.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        qtext = str(q.get("question") or "").strip()
        opts = q.get("options") or []
        if not qid or not qtext or not isinstance(opts, list) or not opts:
            continue
        item: Dict[str, Any] = {"id": qid, "question": qtext, "options": [str(o) for o in opts]}
        inc = q.get("include_if")
        if isinstance(inc, dict) and inc:
            item["include_if"] = {str(k).strip(): str(v).strip() for k, v in inc.items() if str(k).strip()}
        out_questions.append(item)
    return {"questions": out_questions}


def generate_vlm_scene_prompt(
    *,
    question_bank_file: Path,
    system_template_file: Optional[Path],
    llm_base_url: str,
    llm_model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    seed: Optional[int],
    retries: int = 3,
    logger: Any = None,
) -> Dict[str, Any]:
    bank = read_bank(question_bank_file)
    payload = _bank_minimal_view_for_scene_prompt(bank)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)

    if system_template_file:
        system_prompt = read_text(system_template_file)
    else:
        system_prompt = (
            "You are a prompt engineer. Create a domain-agnostic VLM prompt that requests structured evidence only.\n"
        )

    text = call_chat_raw(
        base_url=str(llm_base_url),
        model=str(llm_model),
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": payload_text}],
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=1.0,
        timeout=int(timeout),
        seed=(int(seed) if seed is not None else None),
        retries=int(retries or 0),
        logger=logger,
        retry_stage="question_driven_prompt_gen:scene_prompt",
        api_key=get_llm_api_key(),
    ).strip()
    if not text:
        raise SystemExit("LLM returned empty prompt text")

    return {
        "prompt_text": text,
        "question_bank_sha256": sha256_text(read_text(question_bank_file)),
        "minimal_bank_sha256": sha256_text(payload_text),
        "system_template_file": str(system_template_file) if system_template_file else None,
        "llm_base_url": str(llm_base_url),
        "llm_model": str(llm_model),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 1.0,
        "timeout": int(timeout),
        "seed": (int(seed) if seed is not None else None),
    }


def generate_mapper_rules(
    *,
    question_bank_file: Path,
    system_template_file: Optional[Path],
    llm_base_url: str,
    llm_model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    seed: Optional[int],
    retries: int = 3,
    logger: Any = None,
) -> Dict[str, Any]:
    bank = read_bank(question_bank_file)
    payload = _bank_minimal_view_for_mapper_rules(bank)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)

    system_prompt = read_text(system_template_file) if system_template_file else ""
    if not system_prompt.strip():
        system_prompt = "Write decision rules for mapping a structured description to MCQ answers."

    text_raw = call_chat_raw(
        base_url=str(llm_base_url),
        model=str(llm_model),
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": payload_text}],
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=1.0,
        timeout=int(timeout),
        seed=(int(seed) if seed is not None else None),
        retries=int(retries or 0),
        logger=logger,
        retry_stage="question_driven_prompt_gen:mapper_rules",
        api_key=get_llm_api_key(),
    ).strip()
    if not text_raw:
        raise SystemExit("LLM returned empty rules text")
    if "## DECISION / DISAMBIGUATION RULES" not in text_raw:
        raise SystemExit("LLM rules output missing required heading: '## DECISION / DISAMBIGUATION RULES'")

    return {
        "rules_text": text_raw.strip(),
        "rules_text_raw": text_raw,
        "mode": "llm",
        "question_bank_sha256": sha256_text(read_text(question_bank_file)),
        "minimal_bank_sha256": sha256_text(payload_text),
        "system_template_file": str(system_template_file) if system_template_file else None,
        "llm_base_url": str(llm_base_url),
        "llm_model": str(llm_model),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 1.0,
        "timeout": int(timeout),
        "seed": (int(seed) if seed is not None else None),
    }
