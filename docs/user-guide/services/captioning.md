# Captioning

## What it does

The captioning service asks a **VLM** (Vision-Language Model) to describe what
is happening in an image or video. For video, it splits the clip into
overlapping time windows, captions each window, and can optionally ask an LLM
to summarize the whole clip afterward.

## Before you start

- A reachable OpenAI-compatible (or Gemini) VLM endpoint. See
  [VLM and LLM Endpoints](../vlm-llm-endpoints.md) — you need this before any
  real run; captioning has no local-model mode.
- An API key exported in your shell (`NVIDIA_API_KEY` for OpenAI-compatible
  endpoints, `GEMINI_API_KEY` for Gemini). Never put the key value in a
  cookbook, payload, or command-line argument.
- Input video must be H.264, VP9, or MPEG-4 Part 2 — see
  [Media Policy](../installation.md#media-policy).

## Step by step

### 1. Check the full option list

```bash
make run SCRIPT=captioning-service:main ARGS='--help'
```

### 2. Run against a local OpenAI-compatible endpoint

```bash
export NVIDIA_API_KEY="your-api-key"

make run SCRIPT=captioning-service:main ARGS="\
  --input-file payloads/simple.jsonl \
  --vlm-provider openai-compatible \
  --vlm-endpoint-url http://localhost:8000/v1 \
  --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct"
```

### 3. Run against a remote/hosted endpoint

```bash
make run SCRIPT=captioning-service:main ARGS="\
  --input-file payloads/simple.jsonl \
  --vlm-provider openai-compatible \
  --vlm-endpoint-url https://<your-vlm-endpoint>/v1 \
  --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
```

### 4. (Optional) Add a video-level summary

Add `--enable-llm-summary` plus `--llm-endpoint-url`/`--llm-model` to have an
LLM synthesize one summary paragraph from all window captions:

```bash
make run SCRIPT=captioning-service:main ARGS="\
  --input-file payloads/simple.jsonl \
  --vlm-provider openai-compatible \
  --vlm-endpoint-url https://<your-vlm-endpoint>/v1 \
  --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --enable-llm-summary \
  --llm-provider openai-compatible \
  --llm-endpoint-url https://<your-llm-endpoint>/v1 \
  --llm-model Qwen/Qwen2.5-14B-Instruct"
```

## Verify it worked

- The command exits `0` with no traceback.
- For video scenes: `<data_path>/task/temporal_description.json` and
  `<data_path>/contextual/chunks.json` exist and are non-empty.
- For image scenes: `<data_path>/task/scene_description.json` exists.
- If `--enable-llm-summary` was set:
  `<data_path>/task/video_summarization.json` exists.
- Raw model responses and provenance are kept under
  `<data_path>/sidecars/captioning/` if you need to inspect what the model
  actually said.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Connection refused / timeout | Endpoint URL is wrong or unreachable from where the command runs — verify with `curl <url>/models` first |
| `401`/`403` from the endpoint | Missing or wrong API key; confirm `NVIDIA_API_KEY`/`GEMINI_API_KEY` is exported in the same shell |
| Truncated or empty captions from a reasoning-style model | Raise `--max-tokens`; reasoning models spend part of their output budget on internal reasoning before the visible answer |
| Malformed input JSONL rejected immediately | Check the manifest is one valid `DataEntry` JSON object per line, not a JSON array |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/captioning_service/README.md`](../../../services/captioning_service/README.md)

## Next

[Visual QA](visual-qa.md) or [Reasoning](reasoning.md) typically consume
captioning output next.
