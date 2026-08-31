# Developer Docs

Documentation for people **changing** Auto-Labeling: architecture,
contracts, contributor setup, and how to add a task or service.

To **run** a cookbook as a customer or operator, use the
[User Guide](../user-guide/README.md). Complete that product path once, then
return here.

## Getting Started As A Contributor

- [Local Development](local-development.md) — tests, lint, hooks, templates,
  image registration, and telemetry internals
- [Adding A New Task Or Service](adding-a-task-or-service.md) — the map for
  extending the repo; points into `AGENTS.md` for the authoritative steps
- [`AGENTS.md`](../../AGENTS.md) — setup, code conventions, testing, and CI/CD
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — license, sign-off, and review
  expectations
- [Skills](../../skills/README.md) — task-shaped agent guidance for running,
  debugging, authoring, and integrating stages

## Architecture And Contracts

- [Architecture: Services Overview](architecture/services-overview.md)
- [Architecture: Artifact Contract](architecture/artifact-contract.md)
- [Architecture: Model Client Architecture](architecture/model-client-architecture.md)

## Package And Service Index

Every workspace package ships its own `README.md` next to its `pyproject.toml`
(uv/PyPI packaging resolves `readme = "README.md"` relative to that file, so
these stay put rather than moving under `docs/`). This table is the single
place to find all of them, grouped by the repo's dependency direction:
`services -> tasks -> core`.

For the operator-facing, step-by-step version of the service list, see
[Services](../user-guide/services/README.md) in the user guide.

### Core

| Package | README |
| --- | --- |
| `core` | [packages/core/README.md](../../packages/core/README.md) |

### Task Packages

| Package | README |
| --- | --- |
| `captioning` | [packages/tasks/captioning/README.md](../../packages/tasks/captioning/README.md) |
| `daft_validation` | [packages/tasks/daft_validation/README.md](../../packages/tasks/daft_validation/README.md) |
| `detection_and_tracking` | [packages/tasks/detection_and_tracking/README.md](../../packages/tasks/detection_and_tracking/README.md) |
| `example_task` | [packages/tasks/example_task/README.md](../../packages/tasks/example_task/README.md) |
| `grounding_2d` | [packages/tasks/grounding_2d/README.md](../../packages/tasks/grounding_2d/README.md) |
| `person_attribute_search` | [packages/tasks/person_attribute_search/README.md](../../packages/tasks/person_attribute_search/README.md) |
| `reasoning` | [packages/tasks/reasoning/README.md](../../packages/tasks/reasoning/README.md) |
| `referring_expressions` | [packages/tasks/referring_expressions/README.md](../../packages/tasks/referring_expressions/README.md) |
| `super_resolution` | [packages/tasks/super_resolution/README.md](../../packages/tasks/super_resolution/README.md) |
| `training_export` | [packages/tasks/training_export/README.md](../../packages/tasks/training_export/README.md) |
| `visual_qa` | [packages/tasks/visual_qa/README.md](../../packages/tasks/visual_qa/README.md) |

### Services

| Package | README |
| --- | --- |
| `captioning_service` | [services/captioning_service/README.md](../../services/captioning_service/README.md) |
| `detection_and_tracking_service` | [services/detection_and_tracking_service/README.md](../../services/detection_and_tracking_service/README.md) |
| `event_and_person_attribute_search_service` | [services/event_and_person_attribute_search_service/README.md](../../services/event_and_person_attribute_search_service/README.md) |
| `example_service` | [services/example_service/README.md](../../services/example_service/README.md) |
| `grounding_2d_service` | [services/grounding_2d_service/README.md](../../services/grounding_2d_service/README.md) |
| `reasoning_service` | [services/reasoning_service/README.md](../../services/reasoning_service/README.md) |
| `referring_expressions_service` | [services/referring_expressions_service/README.md](../../services/referring_expressions_service/README.md) |
| `super_resolution_service` | [services/super_resolution_service/README.md](../../services/super_resolution_service/README.md) |
| `training_export_service` | [services/training_export_service/README.md](../../services/training_export_service/README.md) |
| `visual_qa_service` | [services/visual_qa_service/README.md](../../services/visual_qa_service/README.md) |
| `workflow_runner` | [services/workflow_runner/README.md](../../services/workflow_runner/README.md) |
