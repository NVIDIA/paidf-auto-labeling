# Adding A New Task Or Service

This page is the map, not the full instructions — the authoritative,
up-to-date steps live in the root [`AGENTS.md`](../../AGENTS.md), which both
human contributors and coding agents follow. Read the relevant section there
before starting; this page just tells you which section to read and where it
fits relative to everything else in `docs/developer/`.

## Before You Start

1. Follow [Local Development](local-development.md) so `make lint-check`,
   `make mypy`, and `make test` pass on a clean tree. Then read
   [`AGENTS.md`](../../AGENTS.md) for the `services -> tasks -> core`
   dependency direction, script/image discovery, and code conventions.
2. Read [Architecture: Services Overview](architecture/services-overview.md)
   and [Architecture: Artifact Contract](architecture/artifact-contract.md) so
   you know where a new stage fits in the stage order and what it's allowed to
   read or write in a scene.
3. If you're adding VLM/LLM behavior, also read
   [Architecture: Model Client Architecture](architecture/model-client-architecture.md).

## Adding A Task Package

Task packages hold reusable annotation behavior and depend only on `core`.
Follow **["Adding a New Task"](../../AGENTS.md#adding-a-new-task)** in
`AGENTS.md`: duplicate `packages/tasks/example_task/`, keep the same
structure and README shape as that template, register it by running
`make sync` (the workspace glob picks it up automatically — no manual
registration step).

## Adding A Service

Services package a task (or several) as a CLI and container. Follow
**["Adding a New Service"](../../AGENTS.md#adding-a-new-service)** in
`AGENTS.md`: duplicate `services/example_service/`, register its entrypoint
under `[project.scripts]` so it's discoverable from `make run`, and register
its image under `[tool.build.images]` so it's discoverable from `make build`.

Once your service exists, add its own step-by-step page under
[`docs/user-guide/services/`](../user-guide/services/README.md) following the
existing pages there (what it does, prerequisites, numbered run steps, verify,
troubleshoot) so operators can find and run it the same way they find every
other service.

## Related Guidance

- [Local Development](local-development.md) — contributor checks, templates,
  and image registration
- [`AGENTS.md`](../../AGENTS.md) — code conventions, testing, CI/CD, and the
  full command reference (`make lint`, `make mypy`, `make test`, `make check`)
- [`skills/`](../../skills/README.md) — task-shaped agent guidance, including
  [workflow-stage-integration](../../skills/paidf-auto-labeling/references/workflow-stage-integration.md)
  for implementing or reviewing a single new stage or service end to end
- [Package And Service Index](README.md#package-and-service-index) — every
  existing task/service README, for finding a close analog to model yours on
