# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_docker_compose_wires_puid_pgid_and_ckpts_mount() -> None:
    compose = (_repo_root() / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    # Compose should pass PUID/PGID into the container so entrypoint can remap the user.
    assert "PUID:" in compose and "${PUID" in compose
    assert "PGID:" in compose and "${PGID" in compose
    # ckpts should be bind-mounted into /workspace/ckpts (persist across runs).
    assert "/workspace/ckpts" in compose


def test_entrypoint_remaps_user_and_chowns_ckpts() -> None:
    entrypoint = (_repo_root() / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    # UID/GID remap via gosu
    assert "groupmod" in entrypoint
    assert "usermod" in entrypoint
    assert "gosu" in entrypoint
    # Ensure ckpts (bind-mount) is included in chown so permissions are repaired at runtime.
    assert "/workspace/ckpts" in entrypoint


def test_dockerfile_keeps_seedvr_ckpts_unpatched() -> None:
    dockerfile = (_repo_root() / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV SEEDVR_ROOT=/opt/seedvr" in dockerfile
    assert "SEEDVR_CKPTS_DIR" not in dockerfile
    assert "checkpoint='./ckpts/seedvr2_ema_3b.pth'" not in dockerfile
    assert "checkpoint='./ckpts/seedvr2_ema_7b.pth'" not in dockerfile
    assert "checkpoint: ./ckpts/ema_vae.pth" not in dockerfile
