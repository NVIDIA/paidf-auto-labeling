# Captioning

The captioning task provides image and video caption generation. It owns media
sampling, endpoint adapters, prompt handling, response parsing, caption
sidecars, and pipeline-state references. The runnable CLI and container live in
`services/captioning_service/`.

## Model Clients

```text
CaptioningTask
  -> DenseCaptioner
       -> OpenAICompatibleClient  # local vLLM, NIM, or hosted endpoint
       -> GeminiClient
```

Deployments are treated as endpoints; model weights are not bundled with the
task. OpenAI-compatible endpoints use `NVIDIA_API_KEY`, and Gemini endpoints
use `GEMINI_API_KEY`.

## Media and Windowing

Video captioning uses frame-count windows by default. It can instead use
time-based windows or one whole-clip request. `media-mode=auto` sends a video
payload first and falls back to JPEG frames if extraction or the endpoint call
fails. Shared decode and encode behavior comes from `core.media`.

Custom deployments can provide separate image and video prompts. An optional
LLM endpoint can summarize all video-window captions.

## Outputs

The task writes caption evidence under `sidecars/captioning/`:

- `metadata_chunk.json` for detailed video-window evidence.
- `image_caption.json` for raw image-caption evidence.
- `video_captions.json` and `image_captions.json` for compact outputs.
- `sidecars/pipeline_state.json` references under
  `task_artifacts["captioning"]`.

The service pivots successful evidence into DAFT `contextual/` and `task/`
files through `core.formats.daft`. The complete product artifact contract is
documented in
[Artifact Contract](../../../docs/developer/architecture/artifact-contract.md).

## Development

```bash
make run SCRIPT=captioning-service:main ARGS='--help'
make build IMAGE=captioning-service:main
```

Use the CLI `--help` output as the authoritative argument reference.
