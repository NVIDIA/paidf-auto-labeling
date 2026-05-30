# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _docker_image_exists(image: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def test_docker_cli_works() -> None:
    """
    Smoke test: docker is installed and the daemon is reachable.

    Skips automatically when docker isn't available (common in some CI runners).
    """
    if not _docker_available():
        pytest.skip("docker not available (missing CLI, permission issue, or daemon not running)")


def test_docker_can_run_auto_labeling_image() -> None:
    """
    Smoke test: if the auto-labeling docker image exists locally, ensure we can run it.

    We bypass the image entrypoint to avoid side effects like checkpoint downloads.
    """
    if not _docker_available():
        pytest.skip("docker not available (missing CLI, permission issue, or daemon not running)")

    image = os.getenv("AUTO_LABELING_DOCKER_IMAGE", "paidf-auto-labeling:local").strip() or "paidf-auto-labeling:local"
    if not _docker_image_exists(image):
        pytest.skip(f"docker image not found locally: {image}")

    repo_root = _repo_root()

    r = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            # Mount the current repo so we validate *this* checkout's submodules.
            "-v",
            f"{repo_root}:/workspace:ro",
            "-w",
            "/workspace",
            "--entrypoint",
            "bash",
            image,
            "-lc",
            # Validate:
            # - core deps import (schema/config/pipeline wiring)
            # - repo submodules import (mcq / vlm_json / detection_and_tracking / sr_runner)
            # - SeedVR importability via cwd ($SEEDVR_ROOT) (e.g. `import common`)
            # - common binaries exist (uv/ffmpeg/ffprobe/torchrun)
            # We intentionally avoid running the image entrypoint (ckpt downloads).
            "set -euo pipefail; "
            "python -V; "
            "export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/modules; "
            'python -c "'
            "from importlib.util import find_spec; "
            "mods=['omegaconf','pydantic','yaml','multistorageclient','numpy']; "
            "missing=[m for m in mods if find_spec(m) is None]; "
            "assert not missing, f'missing python modules: {missing}'; "
            "import modules.cli; "
            "import modules.al_utils.io; "
            "import modules.mcq_generation.mcq.runners.window_vlm_llm; "
            "import modules.mcq_generation.question_driven_vlm_llm; "
            "print('python imports OK');"
            '"; '
            # Validate importable modules with their native import semantics.
            "python -c \"import modules.detection_and_tracking.rfdetr_tracking; print('rfdetr_tracking import OK')\" >/dev/null; "
            "python -c \"import modules.vlm_json.runners.video_pipeline; print('vlm_json pipeline import OK')\" >/dev/null; "
            "python -c \"import modules.mcq_generation.mcq.runners.window_vlm_llm; print('mcq window_vlm_llm import OK')\" >/dev/null; "
            "python -c \"import modules.mcq_generation.question_driven_vlm_llm; print('mcq question_driven import OK')\" >/dev/null; "
            # SeedVR: validate it is importable when cwd points at SEEDVR_ROOT.
            # (This mirrors how SR is executed via torchrun --module with cwd set.)
            'seedvr_root="${SEEDVR_ROOT:-}"; '
            'if [ -n "$seedvr_root" ] && [ -d "$seedvr_root" ]; then '
            '  (cd "$seedvr_root" && python -c "'
            "import importlib; "
            "from importlib.util import find_spec; "
            "import common; "
            "candidates=["
            "'projects.inference_seedvr2_window',"
            "'projects.inference_seedvr2_7b',"
            "'projects.inference_seedvr2_3b',"
            "'projects.inference_seedvr_7b',"
            "'projects.inference_seedvr_3b',"
            "]; "
            "picked=next((m for m in candidates if find_spec(m) is not None), None); "
            "assert picked is not None, 'no SeedVR inference entry found under projects/'; "
            "importlib.import_module(picked); "
            "print('seedvr imports OK', picked);"
            '"); '
            "else "
            "  echo 'SEEDVR_ROOT not set or not a dir; skipping seedvr import check'; "
            "fi; "
            "command -v uv >/dev/null; "
            "command -v ffmpeg >/dev/null; "
            "command -v ffprobe >/dev/null; "
            "command -v torchrun >/dev/null; "
            "echo 'binaries OK'",
        ],
        check=False,
        timeout=60,
    )
    assert r.returncode == 0
