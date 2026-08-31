# Visual QA

## What it does

The visual QA service answers a fixed set of questions (a **question bank**)
about a scene — multiple-choice (MCQ), binary-choice (BCQ), and open-ended
questions. It can answer directly from image/video windows with a VLM, answer
from an intermediate VLM extraction step with an LLM, or normalize QA evidence
that another stage already produced.

## Before you start

- A question bank JSON file. See the checked-in examples referenced from
  [Samples and Cookbooks](../samples-and-cookbooks.md), or your own domain
  question bank.
- A reachable VLM endpoint for `window-direct-vlm` or `window-vlm-llm`
  generation modes, and an LLM endpoint if you additionally use
  `metadata-llm` or `window-vlm-llm`. See
  [VLM and LLM Endpoints](../vlm-llm-endpoints.md).
- If you only want to **normalize** QA evidence a previous stage already wrote
  (default mode), no endpoint is required — see step 2 below.

## Step by step

### 1. Check the full option list

```bash
make run SCRIPT=visual-qa-service:main ARGS='--help'
```

### 2. Normalize existing evidence (no model call)

If the scene already contains Visual QA sidecars from an earlier run or a
different producer, the default `normalize-only` mode converts them into the
standard `task/mcq.json`, `task/bcq.json`, and `task/open_qa.json` files
without calling any model:

```bash
make run SCRIPT=visual-qa-service:main ARGS='--input-file <input.jsonl>'
```

### 3. Generate answers directly from the VLM

```bash
make run SCRIPT=visual-qa-service:main ARGS="\
  --input-file <input.jsonl> \
  --generation-mode window-direct-vlm \
  --question-bank-file <question_bank.json> \
  --vlm-endpoint-url http://localhost:8000/v1 \
  --vlm-model <served-vlm-model>"
```

### 4. Generate answers via VLM evidence extraction, then LLM reasoning

```bash
make run SCRIPT=visual-qa-service:main ARGS="\
  --input-file <input.jsonl> \
  --generation-mode window-vlm-llm \
  --question-bank-file <question_bank.json> \
  --vlm-endpoint-url http://localhost:8000/v1 --vlm-model <served-vlm-model> \
  --llm-endpoint-url http://localhost:8002/v1 --llm-model <served-llm-model>"
```

### 5. Answer from existing captioning/metadata sidecars only

```bash
make run SCRIPT=visual-qa-service:main ARGS="\
  --input-file <input.jsonl> \
  --generation-mode metadata-llm \
  --question-bank-file <question_bank.json> \
  --llm-endpoint-url http://localhost:8002/v1 --llm-model <served-llm-model>"
```

## Verify it worked

- The command exits `0` with no traceback.
- `<data_path>/task/mcq.json`, `<data_path>/task/bcq.json`, and
  `<data_path>/task/open_qa.json` exist for questions of the matching type in
  your question bank.
- `<data_path>/sidecars/visual_qa/items.json` contains the normalized,
  per-question evidence.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| "No question bank" error in a generation mode | A question bank is required for generation modes; pass `--question-bank-file` |
| `normalize-only` produces empty task files | The scene has no supported Visual QA sidecar yet — run a generation mode first, or confirm the upstream producer wrote to `sidecars/visual_qa/windows.json` |
| `metadata-llm` finds nothing to answer from | No supported captioning/metadata sidecar exists in the scene — run captioning first |
| Endpoint auth error | Confirm `NVIDIA_API_KEY`/`GEMINI_API_KEY` is exported and matches the endpoint provider |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/visual_qa_service/README.md`](../../../services/visual_qa_service/README.md)

## Next

[Reasoning](reasoning.md) can combine Visual QA output with captioning to
derive higher-level events.
