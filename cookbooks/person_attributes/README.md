<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Person Attributes Cookbook

Free-form person attribute captioning cookbook for image-focused PAS workflows.

## Assets

- Question bank: `question_bank.json`
- Person-attribute QD VLM scene prompt: `prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md`

The bank intentionally omits `options`; answers export to DAFT `task/open_qa.json`.

This cookbook overrides the VLM scene prompt because person-attribute captions need PAS-specific evidence. The generic QD mapper template still comes from `../shared/prompts/mcq/question_driven_vlm_llm/templates/` unless explicitly overridden.

## Config

Use `cookbooks/person_attributes/configs/mcq_generation.yaml` as the PAS-specific MCQ override snippet with the root `configs/pipeline_example.yaml`.
