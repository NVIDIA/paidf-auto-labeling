# Auto-Labeling — User Guide (v1.1.0)

How to **run** Auto-Labeling: install it, pick a cookbook, wire
endpoints and checkpoints, and read the outputs.

This guide is for external customers and for NVIDIA operators using the
product. If you are changing the code, follow this guide once so you can run
a cookbook, then switch to [Developer Docs](../developer/README.md).

Typical commands:

- `make sync` — install the workspace
- `make run SCRIPT=workflow-runner:main ARGS='...'` — run a cookbook
- `make build IMAGE=<service>:<target>` — build a stage image when needed

Read the pages in this order the first time. After that, use the table as a
lookup.

1. [Installation](installation.md) — clone, `make sync`, confirm `--help`
2. [Model Provisioning](model-provisioning.md) — download the checkpoints
   your cookbook needs
3. [VLM and LLM Endpoints](vlm-llm-endpoints.md) — stand up or point at a
   model server and export an API key
4. [Getting Started](getting-started.md) — copy a cookbook, dry-run, run
5. [Experiment Output Layout](experiment-output-layout.md) — find the
   results on disk

Open a [service page](services/README.md) only when you need one stage on
its own. Use [Remote Storage](remote-storage.md) only for S3/GCS/Azure/HTTP
paths. If something fails, use [Troubleshooting](troubleshooting.md).

## Start Here By Goal

| Goal | Page |
| --- | --- |
| First end-to-end run | [Getting Started](getting-started.md) |
| Clone, sync, host requirements, and limitations | [Installation](installation.md) |
| Pick a sample cookbook | [Samples and Cookbooks](samples-and-cookbooks.md) |
| Run, debug, or understand one service | [Services](services/README.md) |
| Compile and launch a cookbook | [Operations: Workflow Runner](operations-workflow-runner.md) |
| Wire VLM and LLM endpoints | [VLM and LLM Endpoints](vlm-llm-endpoints.md) |
| Download checkpoints | [Model Provisioning](model-provisioning.md) |
| Use S3/GCS/Azure/HTTP scene or media paths | [Remote Storage](remote-storage.md) |
| Understand the output tree, or run one service by hand | [Experiment Output Layout](experiment-output-layout.md) |
| Fix a failed run | [Troubleshooting](troubleshooting.md) |

## What you can run from this guide

- Video auto-labeling (`video_data_augmentation`)
- Visual Attribute Search cookbooks
- Image spatial grounding (2D grounding plus referring expressions)
- Smart Spaces warehouse event-reasoning
- Training export as a last stage or a standalone batch job

The workflow runner runs those stages locally, in order, on your machine.

## Related Material

- Repository hub: [Project README](../../README.md)
- Changing the code: [Developer Docs](../developer/README.md)
- Cookbook index: [Workflow Runner Cookbooks](../../cookbooks/README.md)
- Release history: [Changelog](../../CHANGELOG.md)
