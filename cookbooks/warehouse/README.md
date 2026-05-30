<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Warehouse Cookbook

Warehouse, shelving, and floor safety question bank cookbook.

## Assets

- Question bank: `question_bank.json`
- Config snippet: `cookbooks/warehouse/configs/mcq_generation.yaml`

This cookbook currently provides a domain-specific question bank and uses the generic question-driven templates from `../shared/prompts/mcq/question_driven_vlm_llm/templates/`.

## Config

Use `cookbooks/warehouse/configs/mcq_generation.yaml` as the warehouse-specific MCQ override snippet with the root `configs/pipeline_example.yaml`.
