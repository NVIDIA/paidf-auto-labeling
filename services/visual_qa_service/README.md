# Auto-Labeling Visual QA Service

Runs the `visual_qa` sidecar task over existing scene directories.

This service is the upstream visual question-answering stage. It prepares
raw and normalized QA sidecars through the `visual_qa` task. The task also
pivots normalized items into DAFT `task/` files through `core.formats.daft`.

Registered entrypoint:

```bash
make run SCRIPT=visual-qa-service:main
make run SCRIPT=visual-qa-service:main ARGS='--input-file <input.jsonl>'
make run SCRIPT=visual-qa-service:main ARGS='--input-file <input.jsonl> --question-bank-file <question_bank.json>'
make run SCRIPT=visual-qa-service:main ARGS='--input-file <input.jsonl> --generation-mode metadata-llm --question-bank-file <question_bank.json> --llm-model <model>'
make run SCRIPT=visual-qa-service:main ARGS='--input-file payloads/simple.jsonl'
```

`make run SCRIPT=visual-qa-service:main` offers a checked-in
`payloads/simple.jsonl` preset and a help preset. Pass `ARGS=...` to use a
scene that already contains visual-QA evidence.

The default `normalize-only` mode expects scene directories that already contain
a supported Visual QA sidecar. Generation modes call model endpoints before
normalization:

- `window-direct-vlm` answers the question bank directly from each image or
  video window.
- `window-vlm-llm` extracts visual evidence with the VLM, then answers with the
  LLM.
- `metadata-llm` answers from captioning or metadata sidecars.

A question bank is optional for normalization and required for generation.
Pass it with `--question-bank-file`; the value may be a local JSON file or a
remote file URL supported by the framework storage layer, such as `s3://`,
`gs://`, `msc://`, `http://`, or `https://`.
API keys are read from environment variables. OpenAI-compatible endpoints use
`NVIDIA_API_KEY` and require an explicit endpoint URL. Gemini endpoints use
`GEMINI_API_KEY`.

Registered image:

```bash
make build IMAGE=visual-qa-service:build
```

The service container builds the manifest-pinned, LGPL-only FFmpeg and PyAV from
source, then verifies its codec and license surface in both build stages. Video inputs may use
H.264, VP9, or MPEG-4 Part 2; generated window clips remain VP9-only. OpenCV
is source-built as `opencv-python-headless` with video backends disabled and is
used only for image decode, resize, color conversion, and JPEG encoding. The
container does not bundle VLM/LLM weights; point it at local or hosted model
endpoints at runtime.

Default input sidecars are checked in order:

```text
sidecars/visual_qa/windows.json
sidecars/metadata.json
```

Default metadata sidecars for `metadata-llm` are checked in order:

```text
sidecars/captioning/metadata_chunk.json
sidecars/captioning/image_caption.json
sidecars/metadata.json
```

Outputs:

```text
sidecars/visual_qa/items.json
sidecars/visual_qa/windows.normalized.json
task/mcq.json
task/bcq.json
task/open_qa.json
```

Pipeline-state references are recorded under `task_artifacts["visual_qa"]`.
