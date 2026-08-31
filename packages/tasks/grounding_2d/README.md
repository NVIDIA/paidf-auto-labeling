# 2D Grounding Task

Image-level referring-expression grounding on unified `DataEntry` scenes.

Flow:

1. Resolve caption (precedence): `--caption` / config → `sidecars/input.json`
   → captioning artifacts (`sidecars/captioning/image_captions.json`, DAFT
   `contextual/image_captions.json`). Prefer omitting `input.json` when the
   cookbook runs `captioning` first.
2. VLM extracts referring expressions and marks each with ``groundable``
   (domain-agnostic: whole countable objects only).
3. Before SAM3, keep expressions that are VLM-groundable and fit SAM3 text
   limits. Optional `--expression-filter-policy-path` exists for advanced
   custom deny lists; default is no policy file.
4. SAM3 grounds each remaining expression via the `detection_and_tracking`
   **library** (in-process import so prompts are dynamic). Packaging uses the
   `services/grounding_2d_service/docker/Dockerfile.gpu` image flavor — not the detection
   service container as a runtime dependency.
5. Keep instances with `score >= min_instance_score` and bbox area `>= min_bbox_area`.
6. Write `sidecars/grounding_2d/grounding_2d.json` plus step diagnostics under
   `sidecars/grounding_2d/`.

VLM calls use core `create_endpoint_client` / `ChatRequest` (same pattern as
captioning). There is no SAM2 path and no VLM phrase-box path. Prefer VLM
``groundable`` over keyword denylists.
