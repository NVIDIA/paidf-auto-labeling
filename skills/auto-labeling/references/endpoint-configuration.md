# Endpoint Configuration

Use this when a run needs VLM or LLM endpoints.

## URL And Model

- Docker cannot reach host services through `localhost`; use `host.docker.internal`.
- Endpoint URLs must end in `/v1`.
- Verify `/models` and use the served model ID, not just the display name.

```bash
export VLM_URL="http://host.docker.internal:<vlm-port>/v1"
export VLM_MODEL="<served-vlm-model-id>"
export LLM_URL="http://host.docker.internal:<llm-port>/v1"
export LLM_MODEL="<served-llm-model-id>"
curl -s "<endpoint-url>/models" | python3 -m json.tool
```

For NVCF/API Catalog, use the deployment `/v1` URL or `https://integrate.api.nvidia.com/v1`.

## Keys

Tell the user which credential source will be used; never print secret values.

- Auth-free local endpoint: set no key.
- Shared NVCF/API Catalog key: use `NVIDIA_API_KEY`.
- Separate endpoint keys: use `VLM_API_KEY` and `LLM_API_KEY`.
- Shared OpenAI-compatible key: use `OPENAI_API_KEY` only if endpoint-specific keys are unset.

If approved, load a key file into env without echoing it:

```bash
export NVIDIA_API_KEY="$(tr -d '\r\n' < "<KEY_FILE>")"
```

For Docker, pass approved variables by name, not by value:

```bash
docker run ... -e NVIDIA_API_KEY ...
docker run ... -e VLM_API_KEY -e LLM_API_KEY ...
```

Do not expand key values into CLI args, URLs, headers, or summaries.
