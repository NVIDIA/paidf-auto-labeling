<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared Cookbook Assets

Cross-use-case prompts and templates live here. For the full cookbook ownership model, see `../../docs/mcq-modes.md`.

## Assets

- VLM JSON prompts: `prompts/vlm_json/`
- VLM verification prompt: `prompts/mcq/vlm_verify/verify_prompt.md`
- Generic question-driven templates: `prompts/mcq/question_driven_vlm_llm/templates/`

Use the shared question-driven templates by default unless a use case needs a domain-specific override, such as the PAS VLM scene prompt in `../person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md`.
