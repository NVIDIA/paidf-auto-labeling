# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy-compatible question-driven prompt generation for Visual QA."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visual_qa.clients import ChatRequest, EndpointClient

_PROMPTS_DIR = Path(__file__).with_name("prompts")
SCENE_PROMPT_SYSTEM_TEMPLATE = (_PROMPTS_DIR / "question_driven_scene_system.md").read_text(
    encoding="utf-8"
)
MAPPER_PROMPT_TEMPLATE = (_PROMPTS_DIR / "question_driven_mapper.md").read_text(encoding="utf-8")


@dataclass(frozen=True)
class QuestionDrivenPrompts:
    """Generated evidence prompt and deterministic bank mapper."""

    evidence_prompt: str
    mapper_prompt: str
    bank_sha256: str


def generate_question_driven_prompts(
    *,
    bank_payload: dict[str, Any],
    llm_client: EndpointClient,
    max_tokens: int,
    system_template: str = SCENE_PROMPT_SYSTEM_TEMPLATE,
) -> QuestionDrivenPrompts:
    """Generate the legacy scene-evidence prompt and inject the mapper bank."""
    questions = bank_payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("question-driven mode requires a question bank questions list")

    minimal_questions: list[dict[str, Any]] = []
    for raw in questions:
        if not isinstance(raw, dict):
            continue
        question_id = str(raw.get("id") or "").strip()
        question = str(raw.get("question") or "").strip()
        if not question_id or not question:
            continue
        item: dict[str, Any] = {"id": question_id, "question": question}
        include_if = raw.get("include_if")
        if isinstance(include_if, dict) and include_if:
            item["include_if"] = include_if
        minimal_questions.append(item)

    minimal_payload = {"questions": minimal_questions}
    evidence_prompt = llm_client.generate(
        ChatRequest(
            prompt=json.dumps(minimal_payload, ensure_ascii=False, indent=2),
            system_prompt=system_template,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
    ).strip()
    if not evidence_prompt:
        raise ValueError("question-driven prompt generation returned an empty prompt")

    bank_text = json.dumps({"questions": questions}, ensure_ascii=False, indent=2)
    bank_block = f"```json\n{bank_text}\n```"
    return QuestionDrivenPrompts(
        evidence_prompt=evidence_prompt,
        mapper_prompt=MAPPER_PROMPT_TEMPLATE.replace("{{QUESTION_BANK_JSON}}", bank_block),
        bank_sha256=hashlib.sha256(bank_text.encode("utf-8")).hexdigest(),
    )


def write_question_driven_prompts(
    output_dir: Path,
    prompts: QuestionDrivenPrompts,
) -> None:
    """Persist generated prompts for parity diagnostics and reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scene_prompt.used.md").write_text(
        prompts.evidence_prompt.rstrip() + "\n",
        encoding="utf-8",
    )
    (output_dir / "mcq_prompt.used.md").write_text(
        prompts.mapper_prompt.rstrip() + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompts.used.json").write_text(
        json.dumps({"question_bank_sha256": prompts.bank_sha256}, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "MAPPER_PROMPT_TEMPLATE",
    "QuestionDrivenPrompts",
    "SCENE_PROMPT_SYSTEM_TEMPLATE",
    "generate_question_driven_prompts",
    "write_question_driven_prompts",
]
