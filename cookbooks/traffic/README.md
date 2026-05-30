<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Traffic Cookbook

Traffic surveillance cookbook for the default MCQ generation blueprint.

## Assets

- Question bank: `question_bank.json`
- Window VLM + LLM prompts: `prompts/mcq/window_vlm_llm/`
- Direct VLM prompt: `prompts/mcq/window_direct_vlm/`
- Metadata LLM prompt: `prompts/mcq/metadata_llm/`

The traffic prompt files above embed the traffic question bank for modes that consume a prompt+bank as one unit. Question-driven mode uses the traffic bank plus the generic templates in `../shared/prompts/mcq/question_driven_vlm_llm/templates/`.

## Config

Use `cookbooks/traffic/configs/mcq_generation.yaml` as the traffic-specific MCQ override snippet with the root `configs/pipeline_example.yaml`.
