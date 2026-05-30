# Custom Caption

Use this reference when the user wants to analyse images or videos with their own questions, or to build a reusable question bank for later batch runs. For the shipped person-attribute / PAS preset, use [`person-attribute-caption.md`](person-attribute-caption.md) instead of redesigning a bank.

Two sub-flows share one config:

- **Analyze now** — ask closed-choice questions about specific files and return answers immediately. Closed-choice is the default; omit `options` or set `options: []` only for intentionally free-form answers.
- **Create bank** — design a reusable question set under `cookbooks/<slug>/question_bank.json` for later use across many files. Use closed-choice by default; omit `options` only when fixed options would be misleading.

---

## Endpoints Required (for "analyze now" only)

Use [`endpoint-configuration.md`](endpoint-configuration.md) for base URL, served model ID, credential handling rules, and the `VLM_URL` / `VLM_MODEL` / `LLM_URL` / `LLM_MODEL` variables used in the commands below.

- Scene-caption-only runs need VLM only.
- MCQ/BCQ and "both" runs need both VLM and LLM.
- Confirm the served model ID before running; the user-facing model name can differ from the served ID (e.g. FP8 suffix).

---

## Flow

### Step 1 — Interview (one message, skip what's already known)

Ask all of these together:

1. **Goal** — analyze files now, or create a question bank for future use?
2. **Files** — which image(s) or video(s)?
3. **What to know** — what do they want to find out or describe?
4. **Answer format**:
   - *Scene caption only* — use VLM JSON scene metadata, not a question bank
   - *MCQ/BCQ labels* — fixed yes/no or multiple-choice (good for labeling/training data)
   - *Both* — run VLM JSON for scene caption plus MCQ/BCQ labels

### Step 2 — Build the question bank

- Read reference images directly with the Read tool; for videos extract frames first:
  ```bash
  ffmpeg -i /path/to/video.mp4 -vf "select='not(mod(n,floor(nb_frames/8+1)))',setpts=N/TB" \
    -frames:v 8 -q:v 2 /tmp/frame_%03d.jpg -y -loglevel error
  ```
- Design 5–15 questions based on what the user wants to know and what you see; use closed-choice options by default, and use free-form entries only when fixed options would be misleading.
- If a bank already exists at the chosen slug path (for example, `cookbooks/traffic/question_bank.json`), ask whether to reuse or regenerate.
- Always save custom reusable banks under `cookbooks/<slug>/question_bank.json`; create the cookbook directory when needed.
- For full bank/export rules (`mcq`, `bcq`, `open_qa`, duplicate options, aggregation), see `modules/mcq_generation/README.md#question-bank-format`.

**Question bank format:**

```json
{
  "name": "<slug>",
  "questions": [
    {
      "id": "1_1",
      "question": "Is there a traffic accident taking place?",
      "options": ["Yes", "No"],
      "aggregation": "any"
    },
    {
      "id": "1_2",
      "question": "What type of incident is it?",
      "options": ["A. Collision", "B. Rollover", "C. Near-miss", "D. Other"],
      "aggregation": "majority"
    }
  ]
}
```

### Step 3 — If "analyze now": run the pipeline

For scene-caption-only requests, run the VLM JSON stage and read `contextual/video.json` or `contextual/image.json` from the output:

```bash
cd <REPO_ROOT> && ./docker/deploy.sh shell -lc "
  uv run python modules/cli.py --config configs/pipeline_example.yaml \
    super_resolution.enabled=false \
    detection_and_tracking.enabled=false \
    vlm_json.enabled=true \
    mcq_generation.enabled=false \
    data.0.inputs.video_path='/workspace/<file>' \
    data.0.output.out_dir='output/caption_<slug>' \
    endpoints.vlm.url='${VLM_URL}' \
    endpoints.vlm.model='${VLM_MODEL}'
"
```

For MCQ/BCQ label requests, confirm VLM/LLM endpoints first, then run the command below. The `single_window=true` override is intentional for caption-style runs so the whole media sample is treated as one item; the blueprint default remains `false`.

Replace placeholders before running:

- `<REPO_ROOT>` — absolute path to this repo
- `<slug>` — short scenario name, e.g. `traffic_incident`; appears in the cookbook path and output dir
- `<file>` — repo-relative path to the input file, e.g. `input/video.mp4` or `output/myrun/sidecars/<track_stem>_tracking.mp4`. The entire repo is mounted at `/workspace/` inside Docker, so any file in the repo tree is accessible.

```bash
cd <REPO_ROOT> && ./docker/deploy.sh shell -lc "
  uv run python modules/cli.py --config configs/pipeline_example.yaml \
    super_resolution.enabled=false \
    detection_and_tracking.enabled=false \
    vlm_json.enabled=false \
    mcq_generation.enabled=true \
    mcq_generation.mode=question-driven-vlm-llm \
    mcq_generation.window_metadata_extraction.single_window=true \
    mcq_generation.window_metadata_extraction.vlm_verify_enabled=false \
    mcq_generation.window_metadata_extraction.question_bank_file='/workspace/cookbooks/<slug>/question_bank.json' \
    data.0.inputs.video_path='/workspace/<file>' \
    data.0.output.out_dir='output/caption_<slug>' \
    endpoints.vlm.url='${VLM_URL}' \
    endpoints.vlm.model='${VLM_MODEL}' \
    endpoints.llm.url='${LLM_URL}' \
    endpoints.llm.model='${LLM_MODEL}'
"
```

Multi-file: expand `data.0`, `data.1`, ... inline.

For "both" requests, keep `vlm_json.enabled=true`, enable `mcq_generation.enabled=true`, and provide both VLM and LLM endpoints. The run produces scene metadata (`contextual/video.json` or `contextual/image.json`) plus task answers (`task/mcq.json`, `task/bcq.json`, and/or `task/open_qa.json`):

```bash
cd <REPO_ROOT> && ./docker/deploy.sh shell -lc "
  uv run python modules/cli.py --config configs/pipeline_example.yaml \
    super_resolution.enabled=false \
    detection_and_tracking.enabled=false \
    vlm_json.enabled=true \
    mcq_generation.enabled=true \
    mcq_generation.mode=question-driven-vlm-llm \
    mcq_generation.window_metadata_extraction.single_window=true \
    mcq_generation.window_metadata_extraction.vlm_verify_enabled=false \
    mcq_generation.window_metadata_extraction.question_bank_file='/workspace/cookbooks/<slug>/question_bank.json' \
    data.0.inputs.video_path='/workspace/<file>' \
    data.0.output.out_dir='output/caption_<slug>' \
    endpoints.vlm.url='${VLM_URL}' \
    endpoints.vlm.model='${VLM_MODEL}' \
    endpoints.llm.url='${LLM_URL}' \
    endpoints.llm.model='${LLM_MODEL}'
"
```

### Step 4A — If "analyze now": present results

Show each question and answer clearly. Don't expose JSON to the user unless they ask.

### Step 4B — If "create bank": confirm and stop

Show the generated questions in a table. Confirm the saved path. Tell the user how to use it later: pass the bank back via `mcq_generation.window_metadata_extraction.question_bank_file='/workspace/cookbooks/<slug>/question_bank.json'` in a future run.

---

## Key Facts

- Full question bank/export rules: `modules/mcq_generation/README.md#question-bank-format`.
- Custom captioning sets `mcq_generation.window_metadata_extraction.single_window=true` so the whole media sample is treated as one window. Blueprint default is `false`. `max_frames=100` and `sampling_fps=2.0` are built-in defaults — no need to set them explicitly.
- Image inputs (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) work natively.
- When SR and tracking are disabled, MCQ stage uses the original `video_path` automatically.
- `vlm_verify_enabled=false` — skip post-verification for captions.
- No GPU override needed — VLM calls go to the remote endpoint.
