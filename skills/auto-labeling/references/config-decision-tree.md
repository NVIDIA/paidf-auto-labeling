# Config Decision Tree — Edge Cases

Run pipeline commands inside Docker. Use the generic NGC-image `docker run` pattern in the sibling `docker-run.md`; repo-local users may use `./docker/deploy.sh shell`. Inside the container, use `uv run python modules/cli.py ...` so the command uses the uv-managed runtime environment baked into the image.

## Partial Pipelines (disabling stages)

Instead of switching config files, disable stages inline:

```bash
# Full pipeline minus SR
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false \
  data.0.inputs.video_path="video.mp4" \
  data.0.output.out_dir="output/run1" \
  endpoints.vlm.url="http://vlm:8000/v1" endpoints.vlm.model="model_id" \
  endpoints.llm.url="http://llm:8000/v1" endpoints.llm.model="llm_id"

# Full pipeline minus MCQ
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  mcq_generation.enabled=false \
  data.0.inputs.video_path="video.mp4" \
  data.0.output.out_dir="output/run1" \
  endpoints.vlm.url="http://vlm:8000/v1" endpoints.vlm.model="model_id"
```

## `metadata-llm` Mode (no standalone preset)

Use `configs/pipeline_example.yaml` with a mode override. The schema auto-disables
`vlm_verify_enabled` for this LLM-only mode.
The unified pipeline still validates `video_path`, so point it to an existing
source media file even though inference reads captions from `metadata_json_path`.

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false detection_and_tracking.enabled=false vlm_json.enabled=false \
  mcq_generation.enabled=true \
  mcq_generation.mode=metadata-llm \
  data.0.inputs.video_path="<source_video.mp4>" \
  data.0.inputs.metadata_json_path="<prior_out_dir>/sidecars/metadata.json" \
  data.0.output.out_dir="output/run1" \
  endpoints.llm.url="http://llm:8000/v1" endpoints.llm.model="llm_id"
```

This mode reads captions from `metadata_json_path` and runs LLM-only MCQ generation.

## Stage Chaining (running stages across separate CLI invocations)

Sometimes users run stages separately (e.g., SR on day 1, MCQ on day 2). The output of each stage is a file that the next stage reads:

| Completed stage | Output artifact | Next stage reads it via |
|-----------------|-----------------|------------------------|
| Tracking | `contextual/instances.json`, `contextual/objects.json`, `sidecars/<track_stem>_tracking_red_id.<ext>` | VLM auto-picks the red-ID overlay when you reuse the same `out_dir`. |
| VLM JSON | Video: `contextual/events.json`, `contextual/video.json`; image: `contextual/image.json` only | Downstream inspection or custom tooling; MCQ modes use media frames or `sidecars/metadata.json` depending on mode. |
| Window MCQ | `sidecars/metadata.json` | `data.0.inputs.metadata_json_path` (for `metadata-llm` mode) |

`<track_stem>` is usually `sr_output` if SR ran, else the original input stem. For a different `out_dir`, set `data.0.inputs.vlm_video_path="<prior_out_dir>/sidecars/<track_stem>_tracking_red_id.<ext>"` explicitly.

Note: when the primary input is an image, tracking output paths are typically `.png` files (not `.mp4`).

**Typical chain when running stages separately:**

1. **Tracking only** → outputs go to `out_dir/`
2. **VLM only** (reads `sidecars/<track_stem>_tracking_red_id.<ext>` from step 1 when available):
   ```bash
   uv run python modules/cli.py --config configs/pipeline_example.yaml \
     super_resolution.enabled=false detection_and_tracking.enabled=false \
     vlm_json.enabled=true mcq_generation.enabled=false \
     data.0.inputs.video_path="<original_video.mp4>" \
     data.0.output.out_dir="<same_out_dir_as_step1>" \
     endpoints.vlm.url="..." endpoints.vlm.model="..."
   ```
3. **MCQ only** (question bank -> VLM evidence -> LLM MCQ):
   ```bash
   uv run python modules/cli.py --config configs/pipeline_example.yaml \
     super_resolution.enabled=false detection_and_tracking.enabled=false vlm_json.enabled=false \
     mcq_generation.enabled=true mcq_generation.mode=question-driven-vlm-llm \
     data.0.inputs.video_path="<original_video.mp4>" \
     data.0.output.out_dir="<out_dir>" \
     endpoints.vlm.url="..." endpoints.vlm.model="..." \
     endpoints.llm.url="..." endpoints.llm.model="..."
   ```

## Windowing Strategy

When `single_window=false` (default), at least one of `window_frames` or `window_seconds` must be > 0.
The blueprint config already sets `window_frames: 60`, so no override is needed unless you want a different window size.

- `mcq_generation.window_metadata_extraction.window_frames=N` — split by frame count (blueprint default: `60`)
- `mcq_generation.window_metadata_extraction.window_seconds=N` — split by duration (alternative to `window_frames`)

If you explicitly zero out both in a custom config (no blueprint), that is a schema validation error. `single_window=true` bypasses both and treats the whole media sample as one window.

## Resuming a Failed Batch

For window MCQ modes, add `mcq_generation.window_metadata_extraction.skip_existing=true` to skip windows that already have outputs. This lets you resume a long batch run after a failure without reprocessing completed windows.

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false detection_and_tracking.enabled=false vlm_json.enabled=false \
  mcq_generation.enabled=true mcq_generation.mode=window-vlm-llm \
  mcq_generation.window_metadata_extraction.skip_existing=true \
  data.0.inputs.video_path="video.mp4" \
  data.0.output.out_dir="output/run1" \
  endpoints.vlm.url="..." endpoints.vlm.model="..." \
  endpoints.llm.url="..." endpoints.llm.model="..."
```
