# VLM And LLM Endpoints

Most Auto-Labeling cookbook flows rely on external model endpoints. The repo’s shared
contract is intentionally simple: the cookbook or CLI supplies endpoint URLs
and model names, while credentials come from environment variables.

## Provider Surface

| Provider family | Typical auth variable | Used by |
| --- | --- | --- |
| OpenAI-compatible | `NVIDIA_API_KEY` | captioning, visual QA, reasoning, grounding, referring expressions, PAS query generation |
| Gemini | `GEMINI_API_KEY` | Gemini-backed captioning and Visual QA flows |

Visual QA also supports task-level fallback variables such as `VLM_API_KEY`,
`LLM_API_KEY`, and `OPENAI_API_KEY`, but the repo-wide cookbook contract should
prefer `NVIDIA_API_KEY` or `GEMINI_API_KEY`.

## Setting Up An Endpoint

Choose one of the two scenarios below before your first real run. Both VLM and
LLM endpoints must expose an OpenAI-compatible `/v1` API.

### Scenario A — Host the models yourself

Deploy models locally with an OpenAI-compatible serving stack, such as the
NVIDIA vLLM container, giving one GPU to each model. A previously validated
local pairing is:

| Role | Model | Example endpoint |
| --- | --- | --- |
| VLM | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | `http://localhost:8000/v1` |
| LLM | `Qwen/Qwen2.5-14B-Instruct` | `http://localhost:8002/v1` |

Follow your serving stack's own deployment instructions. Once both are up:

```bash
export UAL_VLM_URL="http://localhost:8000/v1"
export UAL_VLM_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
export UAL_LLM_URL="http://localhost:8002/v1"
export UAL_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
export NVIDIA_API_KEY="EMPTY"
```

`NVIDIA_API_KEY=EMPTY` satisfies clients that require an `Authorization`
header even though a local, unauthenticated endpoint ignores its value.

### Scenario B — Use a hosted endpoint

Use a model from [NVIDIA Inference](https://inference.nvidia.com/) or another
hosted OpenAI-compatible deployment. Obtain the exact endpoint URL and model ID
from your deployment — do not reuse an endpoint ID from an older project, since
hosted deployments can rotate:

```bash
export UAL_VLM_URL="<provided-vlm-endpoint>/v1"
export UAL_VLM_MODEL="<provided-vlm-model>"
export UAL_LLM_URL="<provided-llm-endpoint>/v1"
export UAL_LLM_MODEL="<provided-llm-model>"
export NVIDIA_API_KEY="<your-personal-api-key>"
```

Obtain your own API key through the standard NVIDIA/NGC access process. Keep
it only in your shell environment — never in cookbook YAML, command-line
arguments, or logs.

If you select a reasoning-style model (for example, a Gemini 3 family model),
raise the affected `--max-tokens` values to `32768` for captioning/Visual
QA/reasoning calls. These models spend part of their output budget on internal
reasoning and can otherwise return truncated or empty results.

### Verify the endpoint before your first real run

```bash
curl --fail --silent --show-error \
  --header "Authorization: Bearer $NVIDIA_API_KEY" \
  "$UAL_VLM_URL/models" >/dev/null
curl --fail --silent --show-error \
  --header "Authorization: Bearer $NVIDIA_API_KEY" \
  "$UAL_LLM_URL/models" >/dev/null
```

Both commands should exit `0` with no output. A non-zero exit means the
endpoint is unreachable, the wrong URL was used, or the key is invalid — fix
this before starting a real cookbook run so failures aren't confused with a
pipeline bug later.

## Workflow Runner Pattern

Export the secret in the host shell, then pass only the variable name:

```bash
export NVIDIA_API_KEY="your-api-key"

make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <cookbook.local.yaml> --container-env NVIDIA_API_KEY'
```

Do not put secret values in:

- cookbooks
- payloads
- Markdown
- command history retained in evidence

## Host Networking

The workflow runner uses host networking by default. On Linux, a model endpoint
running on the same machine can therefore be addressed as:

```text
http://localhost:<port>/v1
```

If you override `--container-network`, use a hostname or address reachable from
that network instead.

## Cookbook Endpoint Blocks

Cookbooks normally declare:

```yaml
endpoints:
  vlm:
    url: http://localhost:8081/v1
    model: <served-vlm-model>
  llm:
    url: http://localhost:8082/v1
    model: <served-llm-model>
```

Replace the model names with the exact identifiers exposed by your serving
stack. The repo ships example Qwen model names in several cookbook configs, but
those are defaults, not hard requirements.

## Minimal Validation

Before a real workflow run, verify:

1. the endpoint URL is reachable from the selected container network
2. the served model name matches the cookbook
3. the required auth variable name is passed into the stage containers
4. one dry-run and one short smoke run complete without hidden retries or auth
   errors

Developers adding a new client: [Model Client Architecture](../developer/architecture/model-client-architecture.md).
