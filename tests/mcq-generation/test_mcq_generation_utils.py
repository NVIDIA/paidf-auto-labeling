# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from al_utils.schema.mcq import WindowMetadataExtractionConfig
from mcq_generation.mcq.utils.aggregation import default_aggregation_spec_for_options
from mcq_generation.mcq.utils.bank import extract_include_if_map_from_prompt_text, inject_bank_into_template
from mcq_generation.mcq.utils.openai import classify_mcq_json_parse_failure
from mcq_generation.mcq.utils.validate import outputs_complete
from mcq_generation.mcq.utils.vlm_verify import build_vlm_verify_prompt, render_vlm_verify_prompt_template


def test_inject_bank_into_template_replaces_placeholders() -> None:
    template = "X\n{{QUESTION_BANK_JSON}}\nY\n{{QUESTION_BANK_MARKDOWN}}\nZ\n"
    bank = {"name": "bank", "questions": [{"id": "1_1", "question": "Q", "options": ["A", "B"]}]}
    fused = inject_bank_into_template(template, bank_payload=bank)
    assert "{{QUESTION_BANK_JSON}}" not in fused
    assert "{{QUESTION_BANK_MARKDOWN}}" not in fused
    assert "```json" in fused
    assert '"questions"' in fused


def test_classify_mcq_json_parse_failure_is_safe_summary() -> None:
    assert classify_mcq_json_parse_failure("") == "empty_output"
    assert classify_mcq_json_parse_failure("not json") == "non_json_output"
    assert classify_mcq_json_parse_failure('{"version": 2.0}') == "missing_mcq_list"
    assert classify_mcq_json_parse_failure("```json\n{bad\n```") == "fenced_json_not_parseable"


def test_qd_mapper_template_supports_free_form_questions() -> None:
    template = Path(
        "cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/mcq_mapper_bank_injected_template.md"
    ).read_text(encoding="utf-8")
    bank = {"name": "open", "questions": [{"id": "caption_1", "question": "Describe the visible clothing."}]}
    fused = inject_bank_into_template(template, bank_payload=bank)

    assert "missing `options` or `options: []`" in fused
    assert "Do NOT invent `options` for free-form questions." in fused
    assert '"id": "caption_1"' in fused


def test_person_attributes_bank_is_free_form() -> None:
    bank = json.loads(Path("cookbooks/person_attributes/question_bank.json").read_text(encoding="utf-8"))

    assert bank["name"] == "person_attributes"
    assert bank["questions"]
    assert all(not q.get("options") for q in bank["questions"])


def test_extract_include_if_map_from_prompt_text_parses_embedded_bank() -> None:
    prompt = (
        "rules...\n\n```json\n"
        + json.dumps(
            {
                "name": "bank",
                "questions": [
                    {"id": "1", "question": "Q1", "options": ["A", "B"]},
                    {"id": "2", "question": "Q2", "options": ["Yes", "No"], "include_if": {"1": "A"}},
                ],
            }
        )
        + "\n```\n"
    )
    inc = extract_include_if_map_from_prompt_text(prompt)
    assert inc == {"2": {"1": "A"}}


def _write_scene_metadata(scene: Path, *, include_caption: bool = True) -> None:
    (scene / "sidecars").mkdir(parents=True, exist_ok=True)
    window = {"start_frame": 0, "end_frame": 1, "llm_enhanced_caption": {"mcq": []}}
    if include_caption:
        window["vlm_caption"] = "caption"
    (scene / "sidecars" / "metadata.json").write_text(
        json.dumps({"video_id": "clip", "windows": [window]}), encoding="utf-8"
    )


def _write_task_mcq(scene: Path) -> None:
    (scene / "task").mkdir(parents=True, exist_ok=True)
    (scene / "task" / "mcq.json").write_text(
        json.dumps(
            {
                "version": "metropolis-v3.0",
                "items": [{"video_id": "clip", "question": "q", "answer": "A", "options": {"A": "x"}}],
            }
        ),
        encoding="utf-8",
    )


def test_outputs_complete_respects_metadata_and_task_output(tmp_path: Path) -> None:
    scene = tmp_path / "clip"
    _write_scene_metadata(scene)
    _write_task_mcq(scene)
    assert outputs_complete(
        out_dir=scene,
        direct_mcq_from_vlm=False,
        caption_key="vlm_caption",
        enhanced_caption_key="llm_enhanced_caption",
    )

    # Remove the task file; no empty stub either -> incomplete.
    (scene / "task" / "mcq.json").unlink()
    assert not outputs_complete(
        out_dir=scene,
        direct_mcq_from_vlm=False,
        caption_key="vlm_caption",
        enhanced_caption_key="llm_enhanced_caption",
    )


def test_outputs_complete_direct_vlm_does_not_require_caption_key(tmp_path: Path) -> None:
    scene = tmp_path / "clip"
    _write_scene_metadata(scene, include_caption=False)
    _write_task_mcq(scene)
    assert outputs_complete(
        out_dir=scene,
        direct_mcq_from_vlm=True,
        caption_key="vlm_caption",
        enhanced_caption_key="llm_enhanced_caption",
    )


def test_default_aggregation_spec_yes_no_variants() -> None:
    # Accept common MCQ option prefix patterns and punctuation.
    opts = ["(A) Yes.", "B) No", "C: maybe"]
    assert default_aggregation_spec_for_options(opts)["type"] == "supermajority"

    # Also accept sentence-like noise around the standalone yes/no word.
    opts2 = ["Answer: yes, because it is visible.", "No - not observed"]
    assert default_aggregation_spec_for_options(opts2)["type"] == "supermajority"


def test_vlm_verify_prompt_template_expands_correction_policy() -> None:
    template = (
        "{correction_policy}\n"
        "{not_supported_rule}\n"
        "{uncertain_rule}\n"
        "{not_supported_constraints}\n"
        "{not_supported_reasoning}\n"
        "{domain_safety_rules}\n"
        "{verdict_values}\n"
        "{current_mcq_answers}"
    )
    mcq_items = [{"id": "1_1", "question": "Q?", "options": ["Yes", "No"], "answer": "No"}]

    prompt, expected_ids = build_vlm_verify_prompt(
        mcq_items,
        prompt_template=template,
        apply_corrections=True,
    )

    assert expected_ids == ["1_1"]
    assert "{correction_policy}" not in prompt
    assert "corrections are enabled" in prompt
    assert "not_supported" in prompt
    assert "supported, not_supported, uncertain" in prompt
    assert '"CURRENT_ANSWER": "No"' in prompt


def test_vlm_verify_prompt_template_expands_corrections_disabled_policy() -> None:
    template = (
        "{correction_policy}\n"
        "{not_supported_rule}\n"
        "{uncertain_rule}\n"
        "{not_supported_constraints}\n"
        "{not_supported_reasoning}\n"
        "{domain_safety_rules}\n"
        "{verdict_values}\n"
        "{current_mcq_answers}"
    )
    mcq_items = [{"id": "1_1", "question": "Q?", "options": ["Yes", "No"], "answer": "No"}]

    prompt, expected_ids = build_vlm_verify_prompt(
        mcq_items,
        prompt_template=template,
        apply_corrections=False,
    )

    assert expected_ids == ["1_1"]
    assert "{correction_policy}" not in prompt
    assert "corrections are disabled" in prompt
    assert "NEVER output not_supported" in prompt
    assert "supported, uncertain" in prompt
    assert "supported, not_supported, uncertain" not in prompt


def test_vlm_verify_corrections_default_to_disabled() -> None:
    template = "{correction_policy}\n{verdict_values}\n{current_mcq_answers}"
    mcq_items = [{"id": "1_1", "question": "Q?", "options": ["Yes", "No"], "answer": "No"}]

    assert WindowMetadataExtractionConfig().vlm_verify_apply_corrections is False
    prompt, expected_ids = build_vlm_verify_prompt(mcq_items, prompt_template=template)

    assert expected_ids == ["1_1"]
    assert "corrections are disabled" in prompt
    assert "supported, uncertain" in prompt
    assert "supported, not_supported, uncertain" not in prompt


def test_vlm_verify_prompt_template_artifact_keeps_answer_placeholder() -> None:
    template = "{correction_policy}\n{verdict_values}\n{current_mcq_answers}"

    rendered = render_vlm_verify_prompt_template(prompt_template=template, apply_corrections=False)

    assert "corrections are disabled" in rendered
    assert "supported, uncertain" in rendered
    assert "{current_mcq_answers}" in rendered
