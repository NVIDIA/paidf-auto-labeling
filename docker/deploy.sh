#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.yml"
# Pass host UID/GID at runtime so the entrypoint remaps the internal "nvidia"
# user via gosu — no image rebuild needed when the host user changes.
PUID="${PUID:-$(id -u)}"
PGID="${PGID:-$(id -g)}"

usage() {
  cat <<'EOF'
Usage: ./docker/deploy.sh <command> [args...]

Commands:
  build        Build image(s)
  up           Build (if needed) and start service (detached)
  down         Stop and remove containers/networks
  shell        Start an interactive shell in the service container
  logs         Tail logs
  ps           Show compose status
  check        Quick sanity checks (GPU + container startup)

Notes:
  - Uses docker compose file: docker/docker-compose.yml
  - Auto-injects PUID/PGID from the host (id -u / id -g) for runtime UID remapping via gosu
  - Runs containers as root (--user 0:0) so the gosu entrypoint activates
  - Does NOT auto-build before running; run `./docker/deploy.sh build` once when needed
  - Will auto-use sudo if current user cannot access /var/run/docker.sock
EOF
}

need_sudo=false
if ! docker info >/dev/null 2>&1; then
  need_sudo=true
fi

if [[ "${need_sudo}" == "true" ]]; then
  DOCKER=(sudo -n docker)
  DOCKER_COMPOSE=(sudo -n docker compose)
else
  DOCKER=(docker)
  DOCKER_COMPOSE=(docker compose)
fi

COMPOSE=("${DOCKER_COMPOSE[@]}" -f "${COMPOSE_FILE}")

# Pass PUID/PGID directly via --env flags on `run --user 0:0` commands so the
# gosu entrypoint remaps the "nvidia" user to the host caller at runtime.
# Using --env (not process-env) avoids the nested-sudo env-reset problem when
# docker requires sudo.
_compose_run_root() {
  "${COMPOSE[@]}" run --rm --user 0:0 --env "PUID=${PUID}" --env "PGID=${PGID}" --env "HF_TOKEN=${HF_TOKEN:-}" "$@"
}
SERVICE="auto-labeling"

cmd="${1:-}"
shift || true

# Ensure bind-mounted host folders exist and are writable before compose.
# This keeps the workflow robust for any user who clones the repo.
mkdir -p "${ROOT_DIR}/ckpts"
if [[ ! -d "${ROOT_DIR}/ckpts" ]]; then
  echo "ERROR: expected directory but found a file: ${ROOT_DIR}/ckpts" >&2
  exit 2
fi

case "${cmd}" in
  build)
    "${COMPOSE[@]}" build "$@"
    ;;
  up)
    "${COMPOSE[@]}" up -d --build "$@"
    ;;
  down)
    "${COMPOSE[@]}" down "$@"
    ;;
  shell)
    _compose_run_root "${SERVICE}" bash "$@"
    ;;
  logs)
    "${COMPOSE[@]}" logs -f --tail=200 "$@"
    ;;
  ps)
    "${COMPOSE[@]}" ps "$@"
    ;;
  check)
    # GPU sanity check (requires NVIDIA Container Toolkit)
    "${DOCKER[@]}" run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L
    # Image + startup sanity check
    "${COMPOSE[@]}" build
    _compose_run_root "${SERVICE}" python -c "print('python OK')"
    echo "ALL CHECKS PASSED"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
