# Reasoning

## What it does

The reasoning service reads a scene's existing captioning and Visual QA
artifacts and asks an **LLM** to derive higher-level output: events, causal
links, temporal localization, and open-ended QA. Every stage it can produce is
opt-in and disabled by default, so you explicitly choose what to generate.

## Before you start

- A scene that already contains captioning output (`sidecars/metadata*.json`
  or `sidecars/captioning/*`). Reasoning does not decode media itself — run
  [Captioning](captioning.md) first.
- A reachable LLM endpoint. See
  [VLM and LLM Endpoints](../vlm-llm-endpoints.md). Only
  `openai-compatible` is currently supported.
- A `DaftExportConfig` file that turns on the stages you want (see step 3).

## Step by step

### 1. Smoke-test with the checked-in sample payload

```bash
make run SCRIPT=reasoning-service:main ARGS='--input-file payloads/simple.jsonl'
```

With no config file, every LLM-driven stage stays disabled, so this validates
wiring without needing an endpoint.

### 2. Check the full option list

```bash
make run SCRIPT=reasoning-service:main ARGS='--help'
```

### 3. Enable specific reasoning stages

Write a config file (JSON or YAML):

```yaml
# reasoning.local.yaml
reasoning:
  events:
    enabled: true
    prompt_variant: events_from_chunks
  msted:
    enabled: true
    prompt_variant: msted_default
```

Then run:

```bash
make run SCRIPT=reasoning-service:main ARGS="\
  --input-file <input.jsonl> \
  --config-file reasoning.local.yaml \
  --llm-endpoint-url http://localhost:8002/v1 \
  --llm-model <served-llm-model>"
```

Each stage with `enabled: true` but no resolvable endpoint logs a warning and
skips writing rather than failing the whole run — check the log if an expected
output file is missing.

## Verify it worked

- The command exits `0` with no traceback.
- Files matching the stages you enabled exist, for example
  `<data_path>/contextual/events.json` for the `events` stage or
  `<data_path>/contextual/msted.json` for `msted`.
- `<data_path>/sidecars/pipeline_state.json` records an
  `annotation_export` entry for each file emitted by this run.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Expected output file never appears | The stage was enabled but had no resolvable LLM endpoint — check `--llm-endpoint-url`/`--llm-model` and the run log for a skip warning |
| Reasoning finds no captions to work from | Run [Captioning](captioning.md) on the scene first; reasoning discovers captions from `sidecars/captioning/*` and `sidecars/metadata*.json` |
| `daft-validation` fails before any stage runs | The scene has an unknown DAFT filename or a media contract mismatch from an earlier stage — see [Troubleshooting](../troubleshooting.md) |

## Full argument reference

[`services/reasoning_service/README.md`](../../../services/reasoning_service/README.md)

## Next

Run [Training Export](training-export.md) as a separate batch step once you
have a directory of completed scenes ready to package.
