# MCQ Mode Agent Guide

Use this as the short agent checklist. Repo paths below are repo-root relative. The full user-facing mode guide lives in `docs/mcq-modes.md`; the question bank schema and export rules live in `modules/mcq_generation/README.md#question-bank-format`.

## Pick A Mode

| Mode | Use when | Endpoints | Required input |
|------|----------|-----------|----------------|
| `question-driven-vlm-llm` | Generic route for traffic, robotics, warehouse, PAS/open-QA, or custom banks | VLM + LLM | `question_bank_file` |
| `window-vlm-llm` | Traffic blueprint prompt: VLM captions, then LLM maps to embedded traffic questions | VLM + LLM | none |
| `window-direct-vlm` | Traffic blueprint prompt: VLM answers embedded traffic questions directly | VLM | none |
| `metadata-llm` | Reuse an existing `sidecars/metadata.json` and remap with LLM only | LLM | `metadata_json_path` plus an existing `video_path` for pipeline validation |

`metadata-llm` needs no VLM at runtime; the schema auto-disables VLM verification for this mode.

## Cookbooks

- Default traffic bank: `cookbooks/traffic/question_bank.json`
- Robotics bank: `cookbooks/robotics/question_bank.json`
- Warehouse bank: `cookbooks/warehouse/question_bank.json`
- PAS/open-QA bank: `cookbooks/person_attributes/question_bank.json`

For `question-driven-vlm-llm`, pass the bank directly:

```bash
mcq_generation.window_metadata_extraction.question_bank_file="/path/to/my_bank.json"
```

For `window-*` and `metadata-llm`, the bank is embedded in `mcq_prompt_file`. To change those questions, create a replacement prompt file that contains both instructions and the bank, then override:

```bash
mcq_generation.window_metadata_extraction.mcq_prompt_file="/path/to/my_prompt_with_bank.md"
```

## Endpoint Notes

Use [`endpoint-configuration.md`](endpoint-configuration.md) for canonical endpoint URL patterns, NVCF/API Catalog notes, served model ID verification, and credential handling rules.
