# Captioning Service

The captioning service is responsible for creating dense captions for images and videos.
It answers the question "What happened in this image or video?".

## Inputs

The service accepts `DataEntry` records through `--input` or `--input-file`.

```jsonl
{"id": "scene-001", "media_path": "/data/a.mp4", "data_path": "/scenes/a"}
{"id": "scene-002", "media_path": "/data/b.jpg", "data_path": "/scenes/b"}
```

`media_path` points at the caller-provided image or video. `data_path` points at
the DAFT scene directory where sidecars and task outputs are written.

## Outputs

Successful runs write captioning artifacts into each data_path output directory.

### DAFT outputs

The primary deliverables are DAFT-validated files under `contextual/` and
`task/`. Downstream tasks and export pipelines consume these directly.

| Output | Location | Description |
|---|---|---|
| Chunk metadata | `contextual/chunks.json` | Per-window video chunk metadata derived from captioning. Video scenes only. |
| Temporal description | `task/temporal_description.json` | Window-level temporal captions in DAFT form. Video scenes only. |
| Scene description | `task/scene_description.json` | Scene-level description derived from captioning output. |
| Video summarization | `task/video_summarization.json` | Video-level summary when LLM summarization is enabled. Video scenes only. |

Image scenes do not produce video-only outputs such as `contextual/chunks.json`,
`task/temporal_description.json`, or `task/video_summarization.json`.

Pass `--no-contextual` to skip these DAFT files and write only captioning
sidecars.

### Captioning sidecars

The service also writes captioning sidecars under `sidecars/captioning/`. These
are not the primary export format; they exist so later pipeline steps can
inspect reasoning traces, reuse compact caption handoffs, and recover model
provenance without re-running the VLM.

| Output | Location | Description |
|---|---|---|
| Video dense metadata | `sidecars/captioning/metadata_chunk.json` | Per-window video captions, timing, model settings, and compact status metadata. The filename can be changed with `--sidecar-filename`. |
| Video compact captions | `sidecars/captioning/video_captions.json` | Compact video caption handoff derived from the dense metadata. |
| Image raw sidecar | `sidecars/captioning/image_caption.json` | Raw image caption sidecar and model provenance. |
| Image compact captions | `sidecars/captioning/image_captions.json` | Compact image caption handoff. |
| Pipeline state | `sidecars/pipeline_state.json` | Stores `task_artifacts["captioning"]` references for downstream tasks. |

Use `--preserve-raw-model-output` when sidecars should also retain full per-window
VLM responses and call metadata for downstream reasoning or debugging.

## Run Locally

Show the service help:

```bash
make run SCRIPT=captioning-service:main ARGS='--help'
```

Run against a local OpenAI-compatible VLM endpoint:

```bash
make run SCRIPT=captioning-service:main ARGS='--input-file payloads/simple.jsonl --vlm-provider openai-compatible --vlm-endpoint-url http://localhost:8000/v1 --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct'
```

Run against a remote OpenAI-compatible VLM NIM endpoint:

```bash
make run SCRIPT=captioning-service:main ARGS="--input-file payloads/simple.jsonl --vlm-provider openai-compatible --vlm-endpoint-url https://YOUR_VLM_ENDPOINT/v1 --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
```

Run against remote VLM and summary LLM NIM endpoint with video summarization enabled:

```bash
make run SCRIPT=captioning-service:main ARGS="--input-file payloads/simple.jsonl --vlm-provider openai-compatible --vlm-endpoint-url https://YOUR_VLM_ENDPOINT/v1 --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --enable-llm-summary --llm-provider openai-compatible --llm-endpoint-url https://YOUR_LLM_ENDPOINT/v1 --llm-model Qwen/Qwen2.5-14B-Instruct"
```

## Arguments

Common service arguments:

| Argument | Required | Description |
|---|---:|---|
| `--input` | No | JSON array of `DataEntry` objects. Mutually exclusive with `--input-file`. |
| `--input-file` | No | JSONL file containing one `DataEntry` object per line. |
| `--dev-data-root` | No | Copies each input scene to `<root>/<entry.id>` before running. Useful for testing without mutating source scenes. |
| `--log-level` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |

Captioning control:

| Argument | Default | Description |
|---|---|---|
| `--disabled` | `false` | Skip caption generation and leave entries unchanged. Useful for smoke tests. |
| `--input-source` | `auto` | Media source to caption: `auto`, `original`, `enhanced`, or `tracking`. `auto` prefers tracking output, then enhanced media, then original media. |
| `--no-contextual` | `false` | Skip DAFT contextual/task pivots and write only captioning sidecars. |
| `--sidecar-filename` | `metadata_chunk.json` | Filename for the dense video caption sidecar under `sidecars/captioning/`. |

Model endpoints:

| Argument | Default | Description |
|---|---|---|
| `--vlm-provider` | `openai-compatible` | Provider adapter for image and per-window video caption generation: `openai-compatible` or `gemini`. |
| `--vlm-endpoint-url` | unset | Base URL for VLM requests. OpenAI-compatible endpoints accept either a `/v1` base URL or a `/chat/completions` URL. Required for openai-compatible. |
| `--vlm-model` | `default` | Model name sent to the VLM endpoint. |
| `--enable-llm-summary` | `false` | Call an LLM after video window captioning to generate a video-level summary. |
| `--llm-provider` | VLM provider | Provider adapter for summary generation. |
| `--llm-endpoint-url` | VLM endpoint URL | Base URL for summary LLM requests. |
| `--llm-model` | VLM model | Model name for summary generation. Required when `--enable-llm-summary` is used. |

Prompts:

| Argument | Default | Description |
|---|---|---|
| `--prompt` | built-in | Inline prompt for video window captioning. Mutually exclusive with `--prompt-file`. |
| `--prompt-file` | unset | Prompt text file for video window captioning. |
| `--image-prompt` | built-in | Inline prompt for image captioning. Mutually exclusive with `--image-prompt-file`. |
| `--image-prompt-file` | unset | Prompt text file for image captioning. |
| `--summary-prompt` | built-in | Inline prompt for LLM summary generation. Mutually exclusive with `--summary-prompt-file`. |
| `--summary-prompt-file` | unset | Prompt text file for LLM summary generation. |
| `--system-prompt` | unset | Optional system prompt sent with VLM caption requests. |

Video windowing and payloads:

| Argument | Default | Description |
|---|---|---|
| `--window-frames` | `256` | Target frame count per video window. Set to `0` to use `--window-seconds`. |
| `--remainder-threshold` | `128` | Minimum leftover frames required to create a final partial window. |
| `--window-seconds` | `10.0` | Video window duration in seconds when `--window-frames` is `0`. |
| `--single-window` | `false` | Caption the whole video as one window instead of splitting it. |
| `--media-mode` | `auto` | Video payload mode: `auto`, `video`, or `frames`. `auto` tries MP4 clips and falls back to JPEG frames. |
| `--sampling-fps` | `2.0` | Frames-per-second sample rate when JPEG frame payloads are used. |
| `--max-frames` | `8` | Maximum JPEG frames sent per window when frame payloads are used. |
| `--resolution` | `768` | Longest-side pixel size for resized JPEG frame payloads. |

Generation and reliability:

| Argument | Default | Description |
|---|---|---|
| `--max-tokens` | `1024` | Maximum output tokens requested from each model call. |
| `--summary-input-token-budget` | `6000` | Approximate input-token budget per LLM summary request. |
| `--temperature` | `0.2` | Sampling temperature sent to each model call. |
| `--top-p` | `0.9` | Nucleus sampling top-p value sent to each model call. |
| `--timeout-s` | `120.0` | HTTP request timeout in seconds. |
| `--retries` | `2` | Retry count for retryable model endpoint failures. |
| `--retry-backoff-s` | `1.0` | Base retry backoff in seconds. |
| `--preserve-raw-model-output` | `false` | Keep raw VLM responses, parsed JSON, full per-window call metadata, token counts, finish reasons, and input-media request metadata. |

## Media toolchain

The container uses the repository's checksum-pinned media toolchain to build an
LGPL-only FFmpeg from source. Video inputs are restricted to H.264 through the
approved NVIDIA CUVID decoder, VP9 with the approved hardware/software fallback, or MPEG-4
Part 2 with FFmpeg's native software decoder, all implemented by `core.media`. Generated
window clips are video-only VP9
MP4 files encoded through the shared PyAV writer.

PyAV is built from source against that FFmpeg. OpenCV is retained only for image
decode, resize, color conversion, and JPEG encoding, and the container installs
only source-built `opencv-python-headless` with its video backends disabled. The
media policy tool verifies the builder and final runtime for codec allow-list,
GPL/nonfree configuration, and wheel-bundled FFmpeg/libav payloads.

API keys are read from environment variables. OpenAI-compatible endpoints use
`NVIDIA_API_KEY` and require an explicit `--vlm-endpoint-url` (and
`--llm-endpoint-url` when summarizing). Gemini endpoints use `GEMINI_API_KEY`.
