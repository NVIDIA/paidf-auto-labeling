# visual_qa

Visual QA is the upstream media-grounded question-answering task. It owns QA
evidence generation, normalization, and DAFT `task/` pivots.

Current sidecars:

Normalized outputs:

```text
sidecars/visual_qa/items.json
sidecars/visual_qa/windows.normalized.json
```

Raw evidence consumed or generated:

```text
sidecars/visual_qa/windows.json
```

The normalized outputs list is not exhaustive; deployments may write additional
sidecars for debugging or pipeline state. Pipeline-state references are written
under `task_artifacts["visual_qa"]`.

Generation modes:

- `normalize-only`: consume existing QA evidence such as
  `sidecars/visual_qa/windows.json` or legacy `sidecars/metadata.json`.
- `window-direct-vlm`: call the VLM directly with each image or video window and
  the question bank.
- `window-vlm-llm`: call the VLM to extract visual evidence, then call the LLM
  to answer the question bank from that evidence.
- `question-driven-vlm-llm`: legacy-compatible mode. An LLM generates a
  bank-specific VLM scene prompt and MCQ mapper from the question bank, then
  runs the same window VLM → LLM mapper sequence as pseudo-labeling QD mode.
  Generated prompts are written under `sidecars/visual_qa/prompts/`.
- `metadata-llm`: call the LLM to answer the question bank from existing
  metadata sidecars such as `sidecars/captioning/metadata_chunk.json` or
  `sidecars/captioning/image_caption.json`.

Video generation uses `--media-mode auto` by default: each window is sent as a
`video/mp4` payload first and falls back to JPEG frames if extraction or the
video-payload model call fails. Video probing and decoding use the approved
`core.media` implementation, and generated window clips use its shared PyAV
`libvpx-vp9` writer. `--media-mode frames` forces JPEG frame payloads.

Non-normalize generation modes require a question bank. Normalization validates
answers against the bank, applies `include_if` gates, aggregates multi-window
answers, records a stable sidecar contract, and writes DAFT task pivots:
`task/mcq.json`, `task/bcq.json`, and `task/open_qa.json`.

API keys are read from environment variables. Explicit `vlm_api_key_env` and
`llm_api_key_env` config values are honored first. Without explicit overrides,
VLM calls prefer `VLM_API_KEY` and LLM calls prefer `LLM_API_KEY`;
OpenAI-compatible endpoints then fall back to `NVIDIA_API_KEY` and
`OPENAI_API_KEY`, while Gemini endpoints fall back to `GEMINI_API_KEY`.

For local video decoding checks, use `uv sync --package visual-qa --extra video`.
