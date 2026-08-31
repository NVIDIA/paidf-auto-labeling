# unified-pseudo-annotation

A Python framework for pseudo-annotating large, unlabeled video and image datasets. Structured as a **uv workspace** (monorepo) with three layers:

- `packages/core/` — shared abstractions and registry utilities
- `packages/tasks/*/` — individual annotation task modules (each is its own package)
- `services/*/` — complete annotation services that compose tasks

Infrastructure (CI/CD, linting, type checking, testing) is fully set up. Task and service implementations are the primary area of active development.

---

## Setup

Requires **uv** and **Python 3.12+**.

```bash
make sync   # installs all workspace packages + dev dependencies
```

This runs `uv sync --all-packages --all-extras` under the hood.

---

## Common Commands

```bash
make sync          # Install/sync all workspace packages
make lint          # Auto-fix linting and format code (ruff)
make lint-check    # Check linting/formatting without making changes
make mypy          # Run mypy type checker
make test          # Run all tests with coverage report
make build         # Interactively select and build a registered Docker image
make run           # Interactively select and run a registered script
make clean         # Remove __pycache__, egg-info, .venv, dist, build
```

Run a specific script directly:
```bash
make run SCRIPT=example-service:main
make run SCRIPT=example-service:main ARGS='--foo bar'
make run SCRIPT=example-service:main ARGS='--input-file payloads/simple.jsonl'
```

Build a specific registered Docker image:
```bash
make build IMAGE=example-service:build
```

---

## Architecture

### Package layout

```
packages/core/                    # Foundation package with shared functionality
packages/tasks/<task-name>/       # Task packages — depend on core
services/<service-name>/          # Services — depend on core + tasks
scripts/run.py                    # Unified script runner
scripts/build.py                  # Unified Docker image build runner
pyproject.toml                    # Root workspace config
Makefile                          # All dev commands
```

### Dependency direction

```
services → tasks → core
```

Services import tasks; tasks import core. Never import upward (core must not import tasks; tasks must not import services).

### Script discovery

`scripts/run.py` reads the root `pyproject.toml` to find workspace members, then scans each package's `pyproject.toml` for `[project.scripts]` entries. Only packages with `[project.scripts]` appear in `make run`. Scripts are executed via `uv run --package <name>`.

The Makefile passes `ARGS` after a `--` separator. `scripts/run.py` treats everything after that separator as the selected script's arguments and does not parse script-specific options. Use normal shell quoting for inline structured values, or prefer `--input-file` for larger payloads.

New services should register their entrypoints as project scripts to make them discoverable from the run script.

### Default args

Each package can declare per-script default arg presets under `[tool.run.default-args]` in its own `pyproject.toml`. Keys must match the script names in `[project.scripts]`, and each preset defines a selector `name` plus the editable `args` string:

```toml
[project.scripts]
main = "example_service.main:main"

[tool.run.default-args]
main = [
    { name = "simple payload", args = "--log-level DEBUG --input-file payloads/simple.jsonl" },
    { name = "remote payload", args = "--log-level DEBUG --input-file payloads/remote.jsonl" },
]
```

When `make run` is invoked without `ARGS`:
- In an interactive terminal, the selected script is followed by an args preset selector, then the selected args are shown in an editable prompt — edit inline and press Enter, or press Enter to accept as-is.
- In a non-interactive context (CI, piped stdin), the first preset is used directly with no prompt.

Passing `ARGS=...` always overrides the default and skips the prompt.

### Image discovery

`scripts/build.py` reads the root `pyproject.toml` to find workspace members, then scans each package's `pyproject.toml` for `[tool.build.images]` entries. Only packages with registered images appear in `make build`. Each image target is addressed as `<project-name>:<image-name>`.

Register an image with the exact Docker build command to run:

```toml
[tool.build.images]
main = "docker buildx build -t example-service -f services/example_service/docker/Dockerfile ."
```

Registered commands are split with `shlex` and executed from the repository root, so write Dockerfile and build context paths relative to the repo root.

A service image whose builder stage starts from a shared base image can declare that base under `[tool.build.image-bases]`, keyed by the same image name:

```toml
[tool.build.images]
main = "docker buildx build -t captioning-service -f services/captioning_service/docker/Dockerfile ."

[tool.build.image-bases]
main = "upa-media-base:ubuntu-vp9"
```

In the generated GitLab config the base builds first (with a longer timeout), the service job gains a `needs:` on it, and the base's pushed ref is passed to the service build as the `BUILDER_BASE_IMAGE` build arg. Pass extra build args to any target with `scripts/build.py <target> --build-arg KEY=VALUE`.

`make build` does not build a declared base automatically, so building a base-consuming service locally is a two-step: build the base first, then the service (which picks up the local base image via the Dockerfile's `BUILDER_BASE_IMAGE` default). For example, `make build IMAGE=upa-media-base:ubuntu-vp9` then `make build IMAGE=captioning-service:main`.

Images build for both `linux/amd64` and `linux/arm64`. Use `docker buildx build` so `scripts/build.py` can add the platform list. In CI each image builds twice: an amd64 job on the `sdg` runners and an arm64 job on the `arm-sdg` runners. Each job pushes a per-arch `<ref>-<arch>` tag, then a merge job runs `docker buildx imagetools create` to combine the two into one manifest-list tag. The per-arch tags stay in the registry as intermediate tags; they share layers and manifests with the final tag so they cost little, but nothing prunes them (registry retention handles cleanup). The two arch jobs run at the same time, but only one build runs per pool at once (`resource_group: docker-build-<arch>`). Each arch keeps its own registry cache (`buildcache-<image>-<version>-<arch>`) so the builds don't overwrite each other. A local `make build` (no push) can't load a manifest list, so it only builds the host architecture. Every base image and `COPY --from` reference must resolve on both platforms — pin bases to a manifest-list (index) digest, not a single-arch one. Override the platform list with `--platform`, e.g. `make build IMAGE=example-service:build` builds host-only, and `scripts/build.py example-service:build --platform linux/arm64 --tag <ref> --push` builds arm64 only.

---

## Video Codec Policy

All services that read or write compressed video must follow the same media policy.
Keep codec selection and validation in shared `core.media` utilities instead of
implementing service-specific codec behavior.

### Media library responsibilities

- **FFmpeg CLI owns compressed-video probing and decoding.** Use shared `core.media`
  utilities, which call `ffprobe` and `ffmpeg` rawvideo pipes, for codec validation, decoder
  selection, and frame decoding.
- **PyAV owns compressed-video encoding and muxing for generated outputs.** Use the shared VP9
  output utilities backed by PyAV and `libvpx-vp9`, built from source against the controlled
  FFmpeg build.
- **OpenCV must never provide video I/O.** Do not use `cv2.VideoCapture`, `cv2.VideoWriter`, or
  OpenCV video-encoding/decoding APIs. OpenCV may be used for image loading, image writing,
  color conversion, contours, masks, and annotation drawing.
- Container images that include OpenCV must install only source-built `opencv-python-headless`,
  configured with FFmpeg, GStreamer, and other general-purpose video backends disabled. Do not
  install `opencv-python` or upstream OpenCV wheels containing bundled media libraries.
- Container images that include PyAV must build it from source against the repository's
  controlled FFmpeg build. Do not use PyAV wheels that bundle private FFmpeg/libav libraries.

### Approved compression formats

- Supported input video codecs are **H.264**, **VP9**, and **MPEG-4 Part 2** only. Container
  extensions such as `.mp4`, `.mov`, and `.webm` do not determine the codec; inspect and
  validate the video stream before processing it.
- H.264 input must use the approved NVIDIA CUVID hardware decoder. If hardware decoding is
  unavailable, reject the input and ask for VP9 or MPEG-4 Part 2 rather than falling back to a
  software H.264 decoder.
- VP9 input should prefer NVIDIA CUVID and may fall back to FFmpeg's native software VP9
  decoder.
- MPEG-4 Part 2 input must use FFmpeg's native software `mpeg4` decoder.
- Every generated video must use **VP9** via `libvpx-vp9`. MP4 is the standard output
  container unless a task contract explicitly requires another container. Do not generate
  H.264, MPEG-4 Part 2 (`mp4v`), H.265/HEVC, or other compressed-video formats.
- Generated annotation videos contain video only; do not preserve or add audio unless a task
  contract explicitly requires it and the codec policy is extended first.

### Container requirements

- Build FFmpeg from a pinned, checksum-verified source release using the shared codec profile
  and build helpers under `scripts/`.
- FFmpeg builds must use an explicit component allowlist with GPL and nonfree components
  disabled. Do not enable `libx264`, `libx265`, `libopenh264`, or other unapproved codecs.
- Use the `input-only` media-toolchain profile for services that only probe or decode video, and
  `vp9-output` for services that generate VP9 video outputs.
- Run `python scripts/media_toolchain.py verify --profile <profile> --python <venv-python>`
  during image builds in both the builder stage and final runtime stage. The build must fail if
  unexpected encoders/decoders, GPL/nonfree configuration, or wheel-bundled FFmpeg/libav payloads
  are present. `scripts/ffmpeg_codec_policy.py` is an internal helper for configure flags and
  marker checks, not the public container-build entrypoint.
- When adding media behavior, reuse or extend `core.media` probing, decoder-selection, and VP9
  output configuration so all services enforce identical errors and fallback behavior.

---

## Adding a New Task

1. Create the directory: `packages/tasks/<task-name>/` by duplicating `packages/tasks/example_task/` and filling in the gaps. Do not try creating a new task package from scratch.
2. The root `pyproject.toml` glob `"packages/tasks/*"` automatically picks it up — no manual registration needed.
3. Run `make sync` to install the new package into the uv workspace.
4. When developing the task, try to keep the overall structure of the project, README, and code similar to the golden example task. It is important for the different tasks to have a predictable and consistent structure.

---

## Adding a New Service

1. Create the directory: `services/<service-name>/` by duplicating `services/example_service/` and filling in the gaps. Do not try creating a new service package from scratch.
2. The root glob `"services/*"` picks it up automatically.
3. Run `make sync`, then `make run SCRIPT=<service-name>:main` to invoke it.
4. When developing the service, try to keep the overall structure of the project, README, and code similar to the golden example service. It is important for the different tasks to have a predictable and consistent structure.

---

## Code Conventions

- **Python 3.12+** — use modern syntax (e.g., `X | Y` union types)
- **Full type annotations required** — mypy runs in strict mode (`disallow_untyped_defs = true`). Every function parameter and return type must be annotated.
- **Line length**: 100 characters (ruff enforced)
- **Import style**: ruff enforces `isort`-compatible import ordering (rule `I`)
- **No implicit `Optional`** — use `X | None` explicitly
- **Ruff rules in effect**: `E, F, I, N, UP, B, C4, PLC0415`
- Pre-commit hooks auto-fix ruff issues on every commit; `make lint` does the same manually.

---

## Before Submitting Code

After writing some code, ensure the following commands all come back green:
- `make test` # Runs unit tests and checks if you created a bug
- `make mypy` # Checks if the typing is correct and passes all mypy checks
- `make lint` # Lints the code, and fixes any linting or formatting errors

## Testing

```bash
make test    # runs pytest --cov across all packages and services
```

- Test files live in `tests/` within each package: `packages/core/tests/`, `packages/tasks/<name>/tests/`, `services/<name>/tests/`
- Coverage tracks `packages/` and `services/` source; excludes `*/tests/*`
- Coverage report shows missing lines and percentages

---

## CI/CD

GitLab CI (`.gitlab-ci.yml`) runs three jobs on every MR and push to `main`:

| Stage | Job | Command |
|-------|-----|---------|
| lint | ruff | `make lint-check` |
| lint | mypy | `make mypy` |
| test | unit-test | `make test` |

All three must pass before merging. The test job produces `test_report.xml` and `coverage_report.xml` artifacts.

Docker image: `ghcr.io/astral-sh/uv:0.5-python3.12-bookworm-slim`

---

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Root workspace config; defines workspace member globs and shared tool config (ruff, mypy, pytest, coverage) |
| `Makefile` | All developer commands |
| `scripts/run.py` | Script discovery and unified runner |
| `scripts/build.py` | Docker image discovery and build runner |
| `packages/core/src/core/__init__.py` | Core abstractions entry point |
| `packages/tasks/example_task/` | Template for new task packages |
| `services/example_service/` | Template for new service packages |
| `.gitlab-ci.yml` | CI pipeline definition |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff check + format) |
