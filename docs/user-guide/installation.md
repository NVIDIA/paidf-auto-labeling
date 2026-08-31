# Installation

Install Auto-Labeling so you can run a cookbook. This is the product
install path for customers and operators.

If you are changing the code, do this install first, then continue with
[Local Development](../developer/local-development.md).

## Requirements

Install only what the selected workflow needs.

| Item | Requirement |
| --- | --- |
| OS | Linux-like host with bind mounts and a container runtime |
| Python | 3.12 or newer |
| Package manager | [`uv`](https://docs.astral.sh/uv/getting-started/installation/) |
| Build/launcher glue | GNU Make |
| Container runtime | Docker or Podman for cookbook execution |
| GPU runtime | NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit |

| Workflow | Extra requirements |
| --- | --- |
| Workflow-runner dry run | Docker or Podman |
| Remote VLM/LLM-backed runs | Reachable endpoint URLs and matching credentials |
| GPU-backed stages | Local checkpoints plus NVIDIA Container Toolkit |
| Remote storage | [Multi-Storage Client configuration](remote-storage.md) |

GPU stages need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(or an equivalent GPU runtime) so stage containers can see the GPU.

`ffmpeg` / `ffprobe` on the host is required to chunk the NGC VSS sample
clips and is useful for other local validation. SAM3,
SeedVR2, and RF-DETR checkpoints are required only for the stages that use
them. See [Model Provisioning](model-provisioning.md).

## Clone And Sync

```bash
git clone <repo-url>
cd paidf-auto-labeling
make sync
```

`make sync` installs the workspace packages needed to run services and
cookbooks.

Copy tracked cookbooks to `*.local.yaml` before you edit paths. Those
`*.local.yaml` files are gitignored so machine-specific paths and endpoints
do not get committed.

## Verify The Install

Confirm the workflow-runner entrypoint resolves:

```bash
make run SCRIPT=workflow-runner:main ARGS='--help'
```

Then confirm the host can compile a cookbook plan without running containers:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
```

A successful dry-run only means the runner and container runtime can parse
that cookbook. It is not a real labeling run. Tracked configs contain
placeholders for media, checkpoints, and endpoints.

You still need, before [Getting Started](getting-started.md):

1. Checkpoints for the stages you will run —
   [Model Provisioning](model-provisioning.md)
2. A VLM/LLM endpoint and an API key in your shell —
   [VLM and LLM Endpoints](vlm-llm-endpoints.md)
3. Staged NGC traffic/warehouse clips for a real sample run —
   [Samples and Cookbooks](samples-and-cookbooks.md#download-vss-sample-clips)
4. Remote storage config only if media or outputs are not on local disk —
   [Remote Storage](remote-storage.md)

Never put API keys or cloud credentials in cookbook YAML. Export them in the
shell and pass the variable name with `--container-env`.

## Limitations

- `workflow-runner` is a local sequential launcher, not a scheduler.
- `grounding_2d` and `referring_expressions` are image-only.
- Person Attribute Search is an assembly stage: it needs upstream sidecars or
  explicit attribute JSON.
- `reasoning-service` documents `openai-compatible` as its LLM provider surface.
- Secrets belong in the environment or the execution platform, not in tracked
  configs.
- The repo does not publish a qualified performance matrix. Time a short
  representative asset on your cookbook and serving stack before you scale.

### Media Policy

- Supported input video codecs are H.264, VP9, and MPEG-4 Part 2 only.
- H.264 requires the NVIDIA CUVID hardware decoder.
- Generated videos are standardized on VP9 output.
