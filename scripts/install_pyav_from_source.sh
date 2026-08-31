#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

python_bin="${1:?venv Python path is required}"
artifact_root="${2:-/opt/media-runtime}"
source_dir="${3:-/usr/share/oss-sources}"
prefix="${4:-/usr/local}"

: "${PYAV_VERSION:?PYAV_VERSION is required}"
: "${PYAV_SHA256:?PYAV_SHA256 is required}"
: "${MEDIA_BUILD_REQUIREMENTS:?MEDIA_BUILD_REQUIREMENTS is required}"

"${python_bin}" -m ensurepip --upgrade
"${python_bin}" -m pip uninstall -y av || true
rm -rf "$(dirname "${python_bin}")/../lib"/python*/site-packages/av \
    "$(dirname "${python_bin}")/../lib"/python*/site-packages/av-*.dist-info \
    "$(dirname "${python_bin}")/../lib"/python*/site-packages/av.libs
export PKG_CONFIG_PATH="${prefix}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
mapfile -t build_requirements <<<"${MEDIA_BUILD_REQUIREMENTS}"
"${python_bin}" -m pip install --no-cache-dir --upgrade "${build_requirements[@]}"
mkdir -p "${source_dir}/source-builds" "${artifact_root}/oss-sources/source-builds"
pyav_sdist="${source_dir}/source-builds/av-${PYAV_VERSION}.tar.gz"
"${python_bin}" -m pip download --no-cache-dir --no-deps --no-binary=:all: \
    --dest "${source_dir}/source-builds" \
    "av==${PYAV_VERSION}"
echo "${PYAV_SHA256}  ${pyav_sdist}" | sha256sum -c -
cp -a "${pyav_sdist}" "${artifact_root}/oss-sources/source-builds/"
if [ -n "${MEDIA_WHEEL_DIR:-}" ]; then
    # Shared-base mode: compile once into a wheel, then install it. Services reuse
    # the wheel (linked against this /usr/local FFmpeg) without recompiling.
    mkdir -p "${MEDIA_WHEEL_DIR}"
    "${python_bin}" -m pip wheel --no-cache-dir --no-build-isolation \
        --no-binary=av \
        --wheel-dir "${MEDIA_WHEEL_DIR}" \
        "${pyav_sdist}"
    "${python_bin}" -m pip install --no-cache-dir --no-index \
        --find-links "${MEDIA_WHEEL_DIR}" \
        av
else
    "${python_bin}" -m pip install --no-cache-dir --no-build-isolation \
        --no-binary=av \
        "${pyav_sdist}"
fi
