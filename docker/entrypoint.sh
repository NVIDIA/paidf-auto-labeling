#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# ---------------------------------------------------------------------------
# Runtime UID/GID remapping via gosu.
# When the container is started as root (--user 0:0), the entrypoint remaps
# the internal "nvidia" user to match the host caller (PUID/PGID env vars),
# fixes ownership of writable bind-mount dirs, then re-execs as that user —
# no build-time UID/GID baking needed.
# ---------------------------------------------------------------------------
if [[ "$(id -u)" == "0" ]]; then
    PUID="${PUID:-1000}"
    PGID="${PGID:-1000}"
    groupmod -o -g "${PGID}" nvidia
    usermod  -o -u "${PUID}" nvidia
    # Only touch runtime-writable bind mounts here. Model/source assets baked
    # into the image are prepared at build time, and checkpoints are handled by
    # the Python model initializers or the explicit prefetch command.
    chown -R nvidia:nvidia /workspace/output /workspace/logs /workspace/ckpts 2>/dev/null || true
    exec gosu nvidia bash "$0" "$@"
fi

if [[ $# -eq 0 ]]; then
  exec bash
else
  exec "$@"
fi
