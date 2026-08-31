# Image Spatial Grounding

Image Spatial Grounding is the user-facing image workflow for turning an image
into grounded regions plus discriminative phrases. It keeps the stage contracts
separate while presenting one complete use case:

```text
captioning -> grounding_2d -> referring_expressions
```

Use this cookbook when you want to caption a scene, ground visible objects with
SAM3 boxes and masks, then generate concise referring phrases for those grounded
regions.

## Configs

Two configs, one per half of the chain — not every stage combination, just the
two genuinely distinct entry points:

- `pipeline_grounding.yaml` runs `captioning -> grounding_2d`: caption an
  image, then ground the visible objects with SAM3 boxes/masks. Sample
  media is `data/input_media/images/traffic_intersection_frames/` after you
  stage NGC traffic clips and extract one still per clip. All staged
  frames run in one pass. Scenes land under
  `output/auto_labeling/image_spatial_grounding/grounding/<filename>/`.
- `pipeline_referring.yaml` runs `referring_expressions` only. Point it at
  the same frames directory and the same grounding `out_dir`. Run grounding
  first so each scene already has `contextual/objects.json`. Use this when
  you already have boxes and only need phrases.

## Run

The tracked configs point at sample paths, so you can inspect either plan
with no changes. Stage NGC frames, run grounding, then run referring on
those scenes:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/image_spatial_grounding/configs/pipeline_grounding.yaml --container-dry-run'

make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/image_spatial_grounding/configs/pipeline_referring.yaml --container-dry-run'
```

For a real model-backed run, copy the tracked config to a gitignored local
config and set the output path, VLM endpoint, model name, and (for
`pipeline_grounding.yaml`) the SAM3 weights mount:

```bash
cp cookbooks/image_spatial_grounding/configs/pipeline_grounding.yaml \
  cookbooks/image_spatial_grounding/configs/pipeline_grounding.local.yaml
```

## Outputs

The workflow writes a normal DAFT scene under the configured output root. The
main stage artifacts are:

- `contextual/objects.json` from SAM3 grounding
- `sidecars/grounding_2d/grounding_2d.json`
- `sidecars/referring_expressions/referring_expressions.json`
- `sidecars/referring_expressions/marked_boxes.jpg`
