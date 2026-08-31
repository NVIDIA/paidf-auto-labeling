# Remote Storage

Auto-Labeling uses [NVIDIA Multi-Storage Client (MSC)](https://nvidia.github.io/multi-storage-client/)
so a cookbook can point `media_path` and output paths at object storage the same
way it points at local files. MSC is the client that talks to S3, GCS, Azure,
AIStore, and HTTP(S). Local paths need no MSC configuration.

If you are only running staged local sample media on disk, skip this page.

## What MSC is for

Without MSC, every stage container would need its own S3/GCS/Azure SDK wiring.
MSC gives Auto-Labeling one profile-based config for those backends: you describe the
bucket once, then use `s3://...`, `gs://...`, `azure://...`, `ais://...`,
`msc://<profile>/...`, or HTTP(S) URLs in the cookbook. The same config must
be visible inside every stage container, because the workflow runner does not
download remote URIs on the host or bind-mount them.

Write the MSC profile itself using the upstream schema, not this page:

- [MSC configuration reference](https://nvidia.github.io/multi-storage-client/references/configuration.html)
- [MSC quickstart](https://nvidia.github.io/multi-storage-client/user_guide/quickstart.html)

## Set it up

1. Create an MSC config file with a profile for your bucket. Prefer workload
   identity, IAM roles, or provider environment variables over long-lived keys.
   If you must pass static credentials, use MSC env-var expansion in that file;
   never put keys in a cookbook, payload, or Markdown file.

   Minimal S3 example:

   ```yaml
   profiles:
     my-s3:
       storage_provider:
         type: s3
         options:
           base_path: my-bucket
           region_name: us-east-1
   ```

2. Export the config on the host. Either:

   ```bash
   export MSC_CONFIG=/path/to/msc.yaml
   ```

   or inline JSON (the form shown in `.env.example`):

   ```bash
   export MULTISTORAGECLIENT_CONFIGURATION="$(cat /path/to/msc.json)"
   ```

   The JSON object is the same schema as the YAML file above. Keep it in the
   environment, not in a cookbook.

   `MSC_CONFIG` wins if both are set. Auto-Labeling writes inline
   `MULTISTORAGECLIENT_CONFIGURATION` to a temp file and points MSC at it.

3. Point the cookbook at remote URIs, for example
   `media_path: s3://my-bucket/clips/clip-001.mp4` or
   `media_path: msc://my-s3/clips/clip-001.mp4`. Use HTTP(S) for remote file
   inputs. Use a configured writable backend for scene outputs.

4. Pass the variable name into every stage container. If you use a config
   file, also mount it:

   ```bash
   make run SCRIPT=workflow-runner:main \
     ARGS='--cookbook-file <cookbook.local.yaml> \
           --container-env MSC_CONFIG \
           --container-mount /path/to/msc.yaml:/path/to/msc.yaml:ro'
   ```

   For inline JSON, pass `--container-env MULTISTORAGECLIENT_CONFIGURATION`
   instead of mounting a file.

## What Auto-Labeling does with a remote path

- A remote `media_path` is one file: the service downloads it into temporary
  local storage, then runs the task.
- A remote `data_path` is a scene prefix: the service syncs it down, runs the
  task, then uploads the completed scene back to that prefix.

Keep credentials out of cookbooks, payloads, logs, and evidence. Mount
credential or config files read-only.
