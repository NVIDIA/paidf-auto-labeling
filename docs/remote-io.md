<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote I/O (Cloud Storage)

The primary remote storage schemes are `s3://` and `msc://`. The pipeline supports remote media inputs, remote configs, and remote output upload:

- Inputs and `--config`: `s3://`, `msc://`, plus direct single-file `http://` and `https://` downloads.
- Output upload (`out_dir`, remote `log_dir`, remote `config_path`): `s3://` or `msc://`.
- Plain HTTP(S) upload is not supported. Use `path_mapping` to rewrite an HTTP(S) storage prefix to an MSC-backed prefix.

Local unit tests can run outside Docker. When running the pipeline itself for development or validation, use Docker (`./docker/deploy.sh shell`) so SR, VLM/LLM clients, DAFT validation, GPU/container setup, and mounted paths match the supported runtime. Pipeline examples below use `uv run python modules/cli.py` from inside the container.

The config is identical to local runs — only `video_path` and `out_dir` change:

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  data.0.inputs.video_path="s3://my-bucket/clip.mp4" \
  data.0.output.out_dir="output/local_out"
```

If `out_dir` is a remote prefix (`s3://…` or `msc://…`), the pipeline runs locally and uploads the whole folder to the remote prefix after a successful run. If `video_path` is a plain HTTP(S) URL, it must point to a single media file, not a directory listing.

The per-sample `config.yaml` written under `out_dir/` is fully runnable — pass it back to `--config` to reproduce the run.

---

## MSC credentials

Set `MULTISTORAGECLIENT_CONFIGURATION` as a JSON string. The CLI writes it to a temp file and sets `MSC_CONFIG` for the MSC library.

### Case 1 — Plain `s3://` with AWS credentials (simplest)

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"
export AWS_ENDPOINT_URL="https://my-s3-endpoint.example.com"   # optional: S3-compatible backend
export AWS_S3_ADDRESSING_STYLE="path"                          # needed for some backends

export MULTISTORAGECLIENT_CONFIGURATION='{}'
```

MSC creates implicit profiles from AWS env vars automatically when the config is an empty object.

### Case 2 — Direct HTTP(S) file inputs

Public or signed HTTP(S) file URLs can be used directly for `data[*].inputs.video_path`, `data[*].inputs.vlm_video_path`, `data[*].inputs.metadata_json_path`, and `--config` when the URL points to one file:

```bash
uv run python modules/cli.py --config "https://example.com/configs/pipeline_example.yaml" \
  data.0.inputs.video_path="https://example.com/videos/clip%20name.mp4" \
  data.0.output.out_dir="output/run"
```

The downloaded filename is URL-decoded before it is staged locally, so percent-encoded names such as `clip%20name.mp4` become `clip name.mp4`. Direct HTTP(S) is download-only; use `path_mapping` for HTTP(S)-backed storage prefixes or uploads.

Shell-escape or quote URL values passed to `data[*].inputs.video_path`, `data[*].inputs.vlm_video_path`, `data[*].inputs.metadata_json_path`, or `--config` when they contain special characters. For example, wrap the URL in single quotes or escape spaces and ampersands; percent-encoded names such as `%20` are still URL-decoded after download.

### Case 3 — `https://` → `s3://` path mapping (CDN / presigned URLs)

```bash
export MULTISTORAGECLIENT_CONFIGURATION='{"path_mapping":{"https://my-bucket.s3.amazonaws.com/":"s3://my-bucket/","https://s3.amazonaws.com/my-bucket/":"s3://my-bucket/"}}'
```

> `path_mapping` source prefixes should be specific and end with `/` — e.g. `"https://my-bucket.s3.amazonaws.com/"` or `"s3://my-bucket/"` ✓, `"s3://"` ✗.

### Case 4 — Named profile with explicit credentials (MSC ≥ 0.36)

```bash
export MULTISTORAGECLIENT_CONFIGURATION='{
  "profiles": {
    "my_s3": {
      "storage_provider": {
        "type": "s3",
        "options": {
          "base_path": "",
          "region_name": "us-west-2",
          "endpoint_url": "https://my-s3-endpoint.example.com",
          "aws_access_key_id": "...",
          "aws_secret_access_key": "...",
          "addressing_style": "path"
        }
      }
    }
  },
  "path_mapping": {
    "s3://my-bucket/": "msc://my_s3/my-bucket/"
  }
}'
```

> Do **not** name the profile `"default"` — it conflicts with MSC's built-in `file` profile.
> MSC ≥ 0.36 requires `storage_provider` (not `storage`); `storage` fails with a schema error.

---

## NVCF / Kubernetes secrets file

If `/var/secrets/secrets.json` is mounted, the CLI reads and exports its keys automatically:

```json
{
  "MULTISTORAGECLIENT_CONFIGURATION": "{}",
  "AWS_ACCESS_KEY_ID": "...",
  "AWS_SECRET_ACCESS_KEY": "...",
  "AWS_DEFAULT_REGION": "us-west-2",
  "AWS_ENDPOINT_URL": "https://my-s3-endpoint.example.com"
}
```

Do not commit secrets. Prefer mounted secrets files or your runtime's secret manager / IAM roles.

---

## Remote config file

The `--config` argument itself can be a remote path:

```bash
uv run python modules/cli.py --config "s3://my-bucket/configs/pipeline_example.yaml" \
  data.0.inputs.video_path="s3://my-bucket/clip.mp4" \
  data.0.output.out_dir="output/run"
```
