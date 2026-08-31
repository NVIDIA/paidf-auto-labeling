# End-to-end (E2E) service tests

This suite drives **every service** through its real CLI entrypoint and asserts on
the artifacts it produces. It runs as a merge-request gate on a GPU runner and is
also runnable locally.

```bash
make test-e2e            # run the whole suite
uv run pytest -m e2e tests/e2e -k captioning   # run one service's tests
```

E2E tests are marked `@pytest.mark.e2e` and are **deselected by default**, so the
regular `make test` (unit) run stays fast and free of GPU/endpoint/ffmpeg
dependencies.

## How services are launched

Tests launch services in **source mode**, exactly the way `scripts/run.py`
(`make run`) does — `uv run --package <project> python -c <entrypoint-runner>` —
which avoids the `main` console-script name collision across packages. See
`e2e_harness.build_service_runner`.

## Execution model and prerequisites

Model endpoints (LLM and VLM) are **external**. Local GPU model inference
(SeedVR super-resolution, RF-DETR / SAM3 tracking) is **not supported for now**;
those services are covered through their stub / disabled / skip paths. Each test
self-skips when a prerequisite is missing:

| Service | Test(s) | Requires |
| --- | --- | --- |
| `example-service` | seeds scene | — (always-on) |
| `training-export-service` | cosmos export | — (always-on) |
| `workflow-runner` | `--container-dry-run` | — (always-on) |
| `super-resolution-service` | `--disabled` | — (always-on) |
| `detection-and-tracking-service` | `--disabled` | — (always-on) |
| `reasoning-service` | no-config validation | — (always-on) |
| `captioning-service` | image caption | VLM endpoint |
| `reasoning-service` | LLM stage | LLM endpoint + `UPA_E2E_REASONING_CONFIG` |
| `captioning-service` | video windows | VLM endpoint + ffmpeg |
| `visual-qa-service` | window-direct-vlm | VLM endpoint + ffmpeg |
| `detection-and-tracking-service` | `stub` backend | ffmpeg |
| `super-resolution-service` | auto-skip high-res | ffmpeg |

## Environment variables

External endpoints (set as masked CI/CD variables, or exported locally):

These reuse the repo's existing env var names (see reasoning's `EndpointResolver`);
`UPA_E2E_*` are the only test-specific knobs.

| Variable | Purpose |
| --- | --- |
| `VLM_ENDPOINT_URL` | **Required** for VLM tests. Base URL (OpenAI-compatible `/v1` or `/chat/completions`); `VLM_BASE_URL` accepted too |
| `VLM_MODEL` | Optional VLM model name (forwarded via `--vlm-model`; service default used if unset) |
| `LLM_ENDPOINT_URL` | **Required** for the LLM stage. Base URL (`LLM_BASE_URL` accepted too) |
| `LLM_MODEL` | Optional LLM model name (forwarded via `--llm-model`) |
| `NVIDIA_API_KEY` | Optional API key for OpenAI-compatible VLM/LLM endpoints (inherited by the service subprocess) |
| `UPA_E2E_REASONING_CONFIG` | Path to a reasoning config file that enables an LLM stage |
| `UPA_E2E_PREFLIGHT` | Set to `0` to skip the endpoint reachability probe |
| `UPA_E2E_SAMPLE_VIDEO` | Optional path to a real input video for the decode tests (e.g. a staged `data/input_media/videos/traffic_video_analytics/traffic_sample_000.mp4`); defaults to a generated synthetic VP9 clip |

Services take no API-key argument: OpenAI-compatible clients always authenticate
with `NVIDIA_API_KEY`, matching the main service contract.

Before running endpoint tests the harness performs a best-effort `GET` liveness
probe against the endpoint URL and skips (rather than failing) when nothing is
answering. Set `UPA_E2E_PREFLIGHT=0` to bypass the probe.

## Fixtures and sample media

`tests/e2e/e2e_harness.py` holds the shared, typed logic; `conftest.py` exposes it
as fixtures:

- `run_service` — launch a service by project name, capture stdout/stderr/exit code.
- `endpoints` — resolved LLM/VLM config with `require_vlm()` / `require_llm()`.
- `make_image` — generate a JPEG via Pillow.
- `sample_video` — resolve the input video for decode tests: use
  `UPA_E2E_SAMPLE_VIDEO` if set (e.g. a staged traffic H.264 sample under
  `data/input_media/videos/traffic_video_analytics`, which exercises the NVIDIA
  CUVID path on a GPU runner), otherwise generate a
  policy-compliant VP9 clip via the shared `core.media` PyAV writer (encode needs
  PyAV's bundled libvpx; **decode inside a service needs system ffmpeg** on PATH —
  H.264 additionally needs NVIDIA CUVID + a GPU).
- `completed_scene` — build a completed DAFT scene for training-export input.

## Enabling the GPU / video-decode paths in CI

The video tests decode compressed video through `core.media`, which shells out to
`ffmpeg`/`ffprobe` on PATH. To activate them, the runner image must provide the
controlled ffmpeg build (see `scripts/media_toolchain.py`) and, for H.264 input,
the NVIDIA CUVID decoder and a visible GPU. Until then those tests skip and the
always-on + endpoint-image tests still gate every service. Set
`UPA_E2E_RUNNER_TAG` to the GPU runner tag in the project CI/CD settings.
