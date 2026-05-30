# Docker Run Reference

Use this reference when the skill is installed outside the repository and `./docker/deploy.sh` is not available.

## Generic NGC Image Flow

Set the image and mount input/output roots at stable container paths. This form assumes the NGC image already contains the pipeline code at `/workspace`:

```bash
export AUTO_LABELING_IMAGE="<NGC_IMAGE>"

docker run --rm --gpus all --ipc=host --shm-size=32g \
  --add-host=host.docker.internal:host-gateway \
  --env-file "<APPROVED_CREDENTIAL_ENV_FILE>" \
  -v "<INPUT_ROOT>:/input:ro" \
  -v "<OUTPUT_ROOT>:/output" \
  -w /workspace \
  "${AUTO_LABELING_IMAGE}" \
  bash -lc '
    uv run python modules/cli.py --config configs/pipeline_example.yaml \
      data.0.inputs.video_path="/input/<VIDEO_FILE>" \
      data.0.output.out_dir="/output/<RUN_NAME>" \
      endpoints.vlm.url="<VLM_URL>" endpoints.vlm.model="<VLM_MODEL>" \
      endpoints.llm.url="<LLM_URL>" endpoints.llm.model="<LLM_MODEL>"
  '
```

Use container paths in CLI overrides:

- Inputs mounted under `/input/...`
- Outputs under `/output/...`
- Configs and cookbooks under `/workspace/...` from the image

If the image does not contain the pipeline checkout, mount one explicitly:

```bash
-v "<REPO_ROOT>:/workspace"
```

## Notes

- Keep `--gpus all` for SR/tracking workloads. Use `pipeline.gpu_ids=<ids>` to select logical GPUs inside the container.
- Keep `--ipc=host --shm-size=32g` for video and model workloads.
- Use [endpoint-configuration.md](endpoint-configuration.md) before adding credentials. Prefer an approved env file, Docker secret, or mounted credential file.
- If the user explicitly approves host environment passthrough instead, pass only variable names (for example `-e VLM_API_KEY`) and never expand or print secret values in the command string.
- Include Hugging Face credentials only when checkpoint/model downloads require them.
- If VLM/LLM endpoints run on the Docker host, use `host.docker.internal` from inside the container. On Linux, include `--add-host=host.docker.internal:host-gateway` in the `docker run` command (repo-local `./docker/deploy.sh` already maps this via `extra_hosts` in `docker/docker-compose.yml`).
- For remote inputs/outputs (`s3://`, `msc://`), also pass the provider credentials and any required `MULTISTORAGECLIENT_CONFIGURATION`.

## Repo-Local Convenience Alternative

When working inside this repository, `./docker/deploy.sh` may be used as a convenience wrapper:

```bash
cd <REPO_ROOT>
./docker/deploy.sh shell -lc '
  uv run python modules/cli.py --config configs/pipeline_example.yaml \
    data.0.inputs.video_path="/workspace/input/<VIDEO_FILE>" \
    data.0.output.out_dir="/workspace/output/<RUN_NAME>" \
    endpoints.vlm.url="<VLM_URL>" endpoints.vlm.model="<VLM_MODEL>" \
    endpoints.llm.url="<LLM_URL>" endpoints.llm.model="<LLM_MODEL>"
'
```

Do not rely on this wrapper in published/common skills unless the target environment is known to have this repo checkout and script.
