# Person-Attribute Caption Preset (PAS)

Use this reference for the shipped image / person-attribute captioning preset. It is not a general MCQ workflow and should not redesign the question bank unless the user explicitly asks. For arbitrary custom questions, see [`custom-caption.md`](custom-caption.md).

## Fixed Preset

- Question bank: `cookbooks/person_attributes/question_bank.json`
- QD VLM scene prompt template: `cookbooks/person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md`
- MCQ mode: `question-driven-vlm-llm`
- Output task type: free-form answers in `task/open_qa.json`
- Intended input: image files. Video inputs can run, but this preset is image-captioning focused.

The bank intentionally omits `options`; this means free-text / open QA, not MCQ/BCQ.

## Endpoints

Use [`endpoint-configuration.md`](endpoint-configuration.md) for base URL, served model ID, credential handling rules, and the `VLM_URL` / `VLM_MODEL` / `LLM_URL` / `LLM_MODEL` variables used in the command below.

## Run Command

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
    mcq_generation.window_metadata_extraction.question_bank_file='/workspace/cookbooks/person_attributes/question_bank.json' \
    mcq_generation.window_metadata_extraction.qd_vlm_scene_prompt_template_file='/workspace/cookbooks/person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md' \
    data.0.inputs.video_path='/workspace/<image-or-video>' \
    data.0.output.out_dir='output/person_attribute_caption_<slug>' \
    endpoints.vlm.url='${VLM_URL}' \
    endpoints.vlm.model='${VLM_MODEL}' \
    endpoints.llm.url='${LLM_URL}' \
    endpoints.llm.model='${LLM_MODEL}'
"
```

For multiple images, expand `data.0`, `data.1`, … and give each sample its own `out_dir`.

## Present Results

Read `task/open_qa.json` and present each generated caption grouped by difficulty:

- `easy_*`: short garment-only captions.
- `medium_*`: simple natural person descriptions with clothing and footwear.
- `hard_*`: richer visible-detail descriptions.

Do not present these as MCQ answers. Do not expose raw JSON unless the user asks.
