# Shared Model Client Architecture

## Goal

Keep provider mechanics out of task and service code. Tasks should own prompts,
parsing, artifact contracts, and task defaults. Services should own CLI and
container orchestration. Shared model clients own endpoint URLs, API key lookup,
request formatting, retries, and response text extraction.

This refactor preserves existing behavior while creating a shared place for new
VLM/LLM providers.

## Folder Structure

```text
packages/core/src/core/model_clients/
  __init__.py              # public shared client API
  types.py                 # ChatRequest, MediaPayload, EndpointClient
  http.py                  # shared JSON POST and retry helper
  openai_compatible.py     # OpenAI/vLLM/NIM-compatible chat client
  gemini.py                # Gemini generateContent client

packages/tasks/captioning/src/captioning/clients.py
packages/tasks/visual_qa/src/visual_qa/clients.py
```

The task-level `clients.py` files are now adapters. They keep task-specific
error names, factories, and import paths stable, but delegate provider behavior
to `core.model_clients`.

## Dependency Direction

```text
services -> tasks -> core.model_clients
```

Services do not instantiate provider SDKs directly. Tasks may inject clients for
tests, but provider implementations live in core. Core model clients do not
import task packages.

## Ownership Boundaries

- `core.model_clients.types`: provider-neutral request and media payload types.
- `core.model_clients.http`: transport validation, JSON POST, transient retry
  policy for REST providers.
- `core.model_clients.openai_compatible`: OpenAI-compatible chat-completions
  message shape, SDK setup, API key resolution, response metadata.
- `core.model_clients.gemini`: Gemini generateContent request and response
  shape.
- Task adapters: task-specific factory signatures, error class names, and
  compatibility aliases.

## Current Scope

This MR extracts the duplicated captioning and visual QA model clients into the
shared layer. It intentionally does not redesign every task model interaction.
The 2D grounding task keeps its image-path-oriented adapter and can move onto
the shared request types in a later, behavior-preserving step.

## Extension Rule

When adding a new provider, add the provider implementation under
`core.model_clients` and expose it through a thin task adapter only when a task
needs custom defaults or compatibility names. Avoid adding provider HTTP, retry,
or SDK code directly in services.
