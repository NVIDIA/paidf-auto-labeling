# Experiment Output Layout

Auto-Labeling writes one **scene** per input: a directory with `raw/`, `contextual/`,
`task/`, and `sidecars/`. Services share that directory instead of calling each
other. Two ways to produce it:

- **Cookbook run (recommended).** Copy a tracked cookbook to `*.local.yaml`,
  set `data[*].output.out_dir`, and run the
  [workflow runner](operations-workflow-runner.md). That is the product path.
  See [Getting Started](getting-started.md).
- **Manual experiment.** Invoke one service yourself with
  `make run SCRIPT=<service>:main`. Use this to debug a single stage or to
  learn the scene contract without compiling a cookbook.

The rest of this page is the directory both paths should use.

## How to run a manual experiment

A manual experiment is a one-service run against a `DataEntry` JSONL file you
create. The workflow runner is not involved.

1. Create the experiment tree and stage source media:

   ```bash
   EXPERIMENT=/path/to/my-experiment
   mkdir -p "$EXPERIMENT"/{source,logs,data/clip-001}
   cp /path/to/clip.mp4 "$EXPERIMENT/source/clip-001.mp4"
   ```

2. Write `$EXPERIMENT/input.jsonl`. Each line is one `DataEntry`. `media_path`
   is the source file; `data_path` is the scene directory that service will
   write:

   ```json
   {
     "id": "clip-001",
     "media_path": "/path/to/my-experiment/source/clip-001.mp4",
     "data_path": "/path/to/my-experiment/data/clip-001"
   }
   ```

   In the file, keep that object on one JSONL line.

3. Run one service. Captioning is a typical first example once a
   [VLM endpoint](vlm-llm-endpoints.md) is reachable:

   ```bash
   make run SCRIPT=captioning-service:main \
     ARGS="--input-file $EXPERIMENT/input.jsonl"
   ```

   Use `make run SCRIPT=<service>:main ARGS='--help'` for that service's
   flags. GPU stages also need the checkpoints from
   [Model Provisioning](model-provisioning.md).

4. Confirm the scene: `$EXPERIMENT/data/clip-001/sidecars/` should contain
   `pipeline_state.json` plus that stage's artifacts.

To chain stages by hand, keep the same `data_path` and feed the next service
the same JSONL. Prefer a cookbook once you are running more than one stage.

Per-service walkthroughs: [Services](services/README.md).

## Directory layout

The inner `data/<entry-id>` directory is the scene passed as
`DataEntry.data_path`. Folders mean:

- `raw/` — media the pipeline is analyzing
- `contextual/` — scene-level objects, tracks, events, and similar metadata
- `task/` — questions, captions, and other task JSON
- `sidecars/` — handoff files: a copy of the original media (`raw.<ext>`),
  the current media for the next stage (`active.<ext>`),
  `pipeline_state.json`, and per-stage extras

```text
<experiment-root>/
  input.jsonl
  source/
    <entry-id>.<ext>
  logs/
    <stage-or-service>.log
  data/
    <entry-id>/
      raw/
      contextual/
      task/
      sidecars/
        raw.<ext>
        active.<ext>
        pipeline_state.json
        logs/
          <task>.log
        <task-artifacts>
```

On a cookbook run, `<experiment-root>` is the cookbook `out_dir`. The runner
creates `data/<entry-id>` for you; you still need `media_path` to point at
real media.

## Rules

- `input.jsonl` is the exact input used for the run.
- `source/` contains the staged source media used by that input.
- `logs/` contains outer service or container logs.
- `data/<entry-id>` is always the value of `DataEntry.data_path`.
- `sidecars/raw.<ext>` preserves the original media once.
- `sidecars/active.<ext>` is the current media handoff for the next stage.

Services should depend on `media_path`, `data_path`, and the scene handoff
inside `sidecars/`, not on the outer experiment-root name. File-level
ownership is in the
[Artifact Contract](../developer/architecture/artifact-contract.md).
