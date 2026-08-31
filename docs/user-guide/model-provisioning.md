# Model Provisioning

Model-backed stages use a shared model-cache root plus optional explicit
checkpoint environment variables. Mount checkpoints read-only in controlled or
offline environments.

This page has two parts: downloading the checkpoints (below), then wiring the
resulting cache directory into a service or cookbook (further down).

## Downloading The Required Checkpoints

Four checkpoints cover every model-backed service in this repo:

| Purpose | Used by | Download source | Expected location |
| --- | --- | --- | --- |
| SAM 3.1 Object Multiplex | [Detection and Tracking](services/detection-and-tracking.md) native SAM3.1 tracking | [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) | `<model-cache>/sam3.1/` |
| RF-DETR | [Detection and Tracking](services/detection-and-tracking.md) RF-DETR backends | [RF-DETR Base COCO](https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth) | `<model-cache>/rfdetr/rf-detr-base.pth` |
| SAM3 (Transformers) | [2D Grounding](services/grounding-2d.md), SAM3 (non-3.1) tracking, [Referring Expressions](services/referring-expressions.md) upstream boxes | [`facebook/sam3`](https://huggingface.co/facebook/sam3) | `<model-cache>/sam3/` |
| SeedVR2 3B | [Super Resolution](services/super-resolution.md) | [`ByteDance-Seed/SeedVR2-3B`](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) | `<model-cache>/seedvr2/` |

You only need the checkpoints for the services you plan to run — see each
service's own page for which one applies.

### 1. Create or use a Hugging Face account

1. Sign in at [huggingface.co/join](https://huggingface.co/join) (or use an
   existing account).
2. Confirm the account email if prompted.

### 2. Request access to gated model repos

`facebook/sam3` and `facebook/sam3.1` are **gated** — you must request and be
granted access individually before you can download them.
`ByteDance-Seed/SeedVR2-3B` is not gated.

While signed in, open each model page below. If the page shows the repo is
gated, submit the access request and wait until access is **granted** before
downloading:

| Model | Repo page |
| --- | --- |
| SAM3.1 | [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1) |
| SAM3 | [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) |
| SeedVR2 3B | [huggingface.co/ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) |

Gated-access approval can take time; do not start a download for a gated repo
until your account shows approved access on that repo's page.

### 3. Install the Hugging Face CLI

Pick one installation method:

```bash
# Preferred: standalone installer
curl -LsSf https://hf.co/cli/install.sh | bash

# Or: pipx
pipx install huggingface_hub[cli]

# Or: pip (user or venv)
python -m pip install -U "huggingface_hub[cli]"
```

Confirm the CLI is on `PATH`:

```bash
hf version
```

### 4. Authenticate

Create a user access token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with
permission to read gated repositories, then log in:

```bash
hf auth login
```

Paste the token when prompted, or pass it directly:

```bash
hf auth login --token "$HF_TOKEN"
```

Verify the active identity:

```bash
hf auth whoami
```

Never commit tokens, put them in cookbooks, or write them into logs. If a
token may have leaked into a shell history or log, revoke it in your Hugging
Face token settings and create a new one.

### 5. Download the checkpoints

```bash
export MODEL_CACHE="<absolute-model-cache-path>"
mkdir -p "$MODEL_CACHE"/{sam3.1,rfdetr,sam3,seedvr2}

hf download facebook/sam3.1 \
  sam3.1_multiplex.pt \
  --local-dir "$MODEL_CACHE/sam3.1"

hf download facebook/sam3 \
  --local-dir "$MODEL_CACHE/sam3"

curl --fail --location \
  --output "$MODEL_CACHE/rfdetr/rf-detr-base.pth" \
  https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth

hf download ByteDance-Seed/SeedVR2-3B \
  ema_vae.pth seedvr2_ema_3b.pth \
  --local-dir "$MODEL_CACHE/seedvr2"
```

### 6. Verify the download

```bash
test -f "$MODEL_CACHE/sam3.1/sam3.1_multiplex.pt"
test -f "$MODEL_CACHE/sam3/config.json"
test -f "$MODEL_CACHE/sam3/model.safetensors"
test -f "$MODEL_CACHE/rfdetr/rf-detr-base.pth"
test -f "$MODEL_CACHE/seedvr2/ema_vae.pth"
test -f "$MODEL_CACHE/seedvr2/seedvr2_ema_3b.pth"
```

Each command should produce no output and exit `0`. A missing file means that
download did not complete — re-run the matching `hf download`/`curl` command
above.

## Default Cache Layout

```text
<model-cache>/
├── seedvr2/
│   ├── ema_vae.pth
│   └── seedvr2_ema_3b.pth
├── rfdetr/
│   └── rf-detr-base.pth
├── sam3/
│   └── <Transformers-format SAM3 files>
└── sam3.1/
    └── sam3.1_multiplex.pt
```

## Expected Assets

- Super resolution resolves SeedVR2 under `<model-cache>/seedvr2`.
- RF-DETR resolves `rfdetr/rf-detr-base.pth` unless `RFDETR_MODEL_PATH`
  overrides it.
- SAM3 Transformers uses the SAM3 cache directory.
- Native SAM 3.1 uses `sam3.1/sam3.1_multiplex.pt` unless `SAM3_MODEL_PATH`
  overrides it.

## Workflow Configuration

Set the shared cache in a cookbook:

```yaml
runtime:
  model_cache_path: /models/cache
```

Mount explicit model directories or files read-only:

```yaml
container:
  mounts:
    - /host/models:/models:ro
  env:
    - SAM3_MODEL_PATH=/models/sam3.1/sam3.1_multiplex.pt
```

## Download Policy

Automatic downloads are opt-in:

- super resolution: `--allow-checkpoint-download`
- RF-DETR: `--allow-model-download`
- SAM3 production runs should use provisioned local weights

Enable downloads only when outbound access and model licensing permit them.
