# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers and types for the end-to-end (E2E) service test suite.

This module is importable from both ``conftest.py`` and the test modules because
``tests/e2e`` is on the pytest ``pythonpath``. Keeping the logic here (rather than
in ``conftest.py``) lets tests share strongly typed helpers without depending on
conftest module-import semantics.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import run as workspace_run
from core.formats.daft import daft_envelope
from core.media import Vp9VideoWriter
from core.model_clients import NVIDIA_API_KEY_ENV
from core.scene import SceneContext, ensure_scene_skeleton
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]

# Generous per-service timeout: real endpoint round-trips and uv resolution can be slow.
DEFAULT_SERVICE_TIMEOUT_S = 900


@dataclass
class ServiceResult:
    """Outcome of launching a workspace service entrypoint."""

    target: str
    returncode: int
    stdout: str
    stderr: str

    def assert_ok(self) -> None:
        """Fail with captured output when the service exited non-zero."""
        if self.returncode != 0:
            raise AssertionError(
                f"Service `{self.target}` exited with code {self.returncode}.\n"
                f"--- stdout ---\n{self.stdout}\n--- stderr ---\n{self.stderr}"
            )

    @property
    def output(self) -> str:
        """Combined stdout and stderr, useful for log assertions."""
        return f"{self.stdout}\n{self.stderr}"


# Signature: run(project_name, args, *, env=None, timeout=...) -> ServiceResult
RunService = Callable[..., ServiceResult]


def build_service_runner() -> RunService:
    """Build a callable that launches workspace services the way ``make run`` does."""

    # ``run`` (scripts/run.py) is importable because scripts/ is on the pytest pythonpath.
    runnables = {runnable.project_name: runnable for runnable in workspace_run.discover_runnables()}

    def _run(
        project_name: str,
        args: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        timeout: int = DEFAULT_SERVICE_TIMEOUT_S,
    ) -> ServiceResult:
        try:
            runnable = runnables[project_name]
        except KeyError:
            available = ", ".join(sorted(runnables))
            raise KeyError(
                f"Unknown service `{project_name}`. Available services: {available}."
            ) from None

        command = [
            "uv",
            "run",
            "--package",
            runnable.project_name,
            "python",
            "-c",
            workspace_run.ENTRYPOINT_RUNNER,
            runnable.entrypoint,
            *[str(arg) for arg in args],
        ]
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=proc_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ServiceResult(
            target=f"{project_name}:{runnable.script_name}",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return _run


@dataclass
class EndpointConfig:
    """Configuration for one external model endpoint (LLM or VLM)."""

    kind: str
    url: str | None
    model: str | None
    api_key: str | None

    @property
    def configured(self) -> bool:
        # Only the URL is required; the model is optional (services default it) and
        # forwarded only when set.
        return bool(self.url)


@dataclass
class Endpoints:
    """External LLM and VLM endpoints resolved from the environment."""

    vlm: EndpointConfig
    llm: EndpointConfig

    def require_vlm(self) -> EndpointConfig:
        """Return the VLM config or skip when it is unset/unreachable."""
        return _require_endpoint(self.vlm, "VLM_ENDPOINT_URL")

    def require_llm(self) -> EndpointConfig:
        """Return the LLM config or skip when it is unset/unreachable."""
        return _require_endpoint(self.llm, "LLM_ENDPOINT_URL")


def vlm_cli_args(config: EndpointConfig) -> list[str]:
    """Build ``--vlm-*`` CLI arguments from a resolved VLM endpoint config.

    ``--vlm-model`` is only added when ``VLM_MODEL`` is set; otherwise the service
    falls back to its own default model name.
    """
    args = ["--vlm-endpoint-url", str(config.url)]
    if config.model:
        args += ["--vlm-model", config.model]
    return args


def llm_cli_args(config: EndpointConfig) -> list[str]:
    """Build ``--llm-*`` CLI arguments from a resolved LLM endpoint config.

    ``--llm-model`` is only added when ``LLM_MODEL`` is set; otherwise the service
    falls back to its own default model name.
    """
    args = ["--llm-endpoint-url", str(config.url)]
    if config.model:
        args += ["--llm-model", config.model]
    return args


def resolve_endpoints() -> Endpoints:
    """Resolve external LLM/VLM endpoint configuration from the environment.

    Reuses the repo's existing env var convention (see reasoning's
    ``EndpointResolver``): ``VLM_ENDPOINT_URL`` (or ``VLM_BASE_URL``) / ``VLM_MODEL``
    and the ``LLM_*`` equivalents. Auth for OpenAI-compatible clients always uses
    ``NVIDIA_API_KEY``, which is inherited by the service subprocess. Only the URL
    is required; the model is forwarded as ``--vlm-model`` / ``--llm-model`` only
    when set.
    """
    api_key = os.environ.get(NVIDIA_API_KEY_ENV) or None

    def _config(kind: str) -> EndpointConfig:
        role = kind.upper()  # VLM or LLM
        return EndpointConfig(
            kind=kind,
            url=os.environ.get(f"{role}_ENDPOINT_URL")
            or os.environ.get(f"{role}_BASE_URL")
            or None,
            model=os.environ.get(f"{role}_MODEL") or None,
            api_key=api_key,
        )

    return Endpoints(vlm=_config("vlm"), llm=_config("llm"))


def _require_endpoint(config: EndpointConfig, env_hint: str) -> EndpointConfig:
    if not config.configured:
        pytest.skip(f"Set {env_hint} to run {config.kind.upper()} endpoint E2E tests.")
    if os.environ.get("UPA_E2E_PREFLIGHT", "1") != "0" and not endpoint_reachable(config):
        pytest.skip(
            f"{config.kind.upper()} endpoint {config.url!r} is not reachable "
            "(set UPA_E2E_PREFLIGHT=0 to bypass this probe)."
        )
    return config


def endpoint_reachable(config: EndpointConfig) -> bool:
    """Best-effort liveness probe: is anything answering at the endpoint URL?

    This is only a skip-vs-run gate, so any HTTP response below 500 (including a
    404/405 from a POST-only route) counts as reachable. The test itself fails
    loudly if the endpoint is actually broken.
    """
    if not config.url:
        return False
    try:
        request = urllib.request.Request(config.url, method="GET")
        if config.api_key:
            request.add_header("Authorization", f"Bearer {config.api_key}")
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            return bool(response.status < 500)
    except urllib.error.HTTPError as error:
        # An auth or other 4xx response still proves the endpoint is answering.
        return bool(error.code < 500)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def ffmpeg_on_path() -> bool:
    """Whether ``ffmpeg`` and ``ffprobe`` are both on PATH for source-mode decode."""
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def require_ffmpeg() -> None:
    """Skip the calling test when ffmpeg/ffprobe are not available on PATH."""
    if not ffmpeg_on_path():
        pytest.skip("ffmpeg/ffprobe not found on PATH; required for source-mode video decode.")


def generate_image(path: Path, *, width: int = 1280, height: int = 720) -> Path:
    """Write a small, decodable JPEG image fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (28, 42, 64))
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 4, height // 4, width // 2, height // 2), fill=(220, 80, 40))
    draw.ellipse((width // 2, height // 3, width * 3 // 4, height * 2 // 3), fill=(60, 200, 120))
    image.save(path, format="JPEG", quality=90)
    return path


def generate_vp9_video(
    path: Path,
    *,
    seconds: float = 2.0,
    fps: float = 8.0,
    width: int = 256,
    height: int = 144,
) -> Path:
    """Write a small policy-compliant VP9 video fixture via the shared PyAV writer.

    Encoding uses PyAV's bundled libvpx, so it does not require a system ffmpeg — but
    decoding the fixture inside a service still needs ffmpeg on PATH.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(int(seconds * fps), 1)
    with Vp9VideoWriter(path, width=width, height=height, fps=fps, frame_format="bgr24") as writer:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            progress = index / max(frame_count - 1, 1)
            box_x = int(progress * (width - 40))
            frame[height // 2 - 20 : height // 2 + 20, box_x : box_x + 40] = (0, 0, 255)
            writer.write(frame)
    return path


def resolve_sample_video(dest: Path, **generate_kwargs: Any) -> Path:
    """Return a real override video, or a freshly generated synthetic VP9 clip.

    When ``UPA_E2E_SAMPLE_VIDEO`` points at a file, that real clip is used, which
    exercises the NVIDIA CUVID decode path on a GPU runner for H.264 input. Otherwise
    a small synthetic VP9 clip is generated so the video tests run without a GPU.
    Extra keyword arguments are forwarded to :func:`generate_vp9_video` and ignored
    when an override is used.
    """
    override = os.environ.get("UPA_E2E_SAMPLE_VIDEO")
    if override:
        path = Path(override)
        if not path.is_file():
            pytest.skip(f"UPA_E2E_SAMPLE_VIDEO={override!r} is not a file.")
        return path
    return generate_vp9_video(dest, **generate_kwargs)


def build_completed_scene(scene_dir: Path, *, media_id: str = "clip") -> Path:
    """Build a completed DAFT scene with one MCQ task item for training-export input."""
    paths = ensure_scene_skeleton(scene_dir)
    (paths.raw_dir / f"{media_id}.mp4").write_bytes(b"video")

    context = SceneContext(media_id=media_id)
    video_payload = daft_envelope("video", context)
    video_payload.update(
        {"format": "mp4", "fps": 30, "duration": 1.0, "height": 720, "width": 1280}
    )
    _write_json(paths.contextual_dir / "video.json", video_payload)

    mcq_payload = daft_envelope("mcq", context, include_scene_id=False)
    mcq_payload["items"] = [
        {
            "video_id": media_id,
            "question": "Which object moves?",
            "options": {"A": "Car", "B": "Bike"},
            "answer": "A",
        }
    ]
    _write_json(paths.task_dir / "mcq.json", mcq_payload)
    return scene_dir


def entries_payload(entries: list[dict[str, Any]]) -> str:
    """Serialize DataEntry records for a service ``--input`` argument."""
    return json.dumps(entries)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
