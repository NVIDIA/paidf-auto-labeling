#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

python_bin="${1:?venv Python path is required}"
artifact_root="${2:-/opt/media-runtime}"
source_dir="${3:-/usr/share/oss-sources}"
prefix="${4:-/usr/local}"
export PATH="$(dirname "${python_bin}"):${PATH}"

: "${OPENCV_PYTHON_HEADLESS_VERSION:?OPENCV_PYTHON_HEADLESS_VERSION is required}"
: "${OPENCV_PYTHON_HEADLESS_SHA256:?OPENCV_PYTHON_HEADLESS_SHA256 is required}"
: "${MEDIA_BUILD_REQUIREMENTS:?MEDIA_BUILD_REQUIREMENTS is required}"

export CMAKE_PREFIX_PATH="${prefix}:${CMAKE_PREFIX_PATH:-}"

"${python_bin}" -m ensurepip --upgrade
"${python_bin}" -m pip uninstall -y opencv-python opencv-python-headless || true
rm -rf "$(dirname "${python_bin}")/../lib"/python*/site-packages/cv2 \
    "$(dirname "${python_bin}")/../lib"/python*/site-packages/opencv_python-*.dist-info \
    "$(dirname "${python_bin}")/../lib"/python*/site-packages/opencv_python.libs \
    "$(dirname "${python_bin}")/../lib"/python*/site-packages/opencv_python_headless-*.dist-info \
    "$(dirname "${python_bin}")/../lib"/python*/site-packages/opencv_python_headless.libs
mapfile -t build_requirements <<<"${MEDIA_BUILD_REQUIREMENTS}"
"${python_bin}" -m pip install --no-cache-dir --upgrade "${build_requirements[@]}"
mkdir -p "${source_dir}/source-builds" "${artifact_root}/oss-sources/source-builds"
opencv_sdist="${source_dir}/source-builds/opencv-python-headless-${OPENCV_PYTHON_HEADLESS_VERSION}.tar.gz"
# Reuse the Python 3.12-compatible build tools installed above while pip prepares
# sdist metadata; OpenCV's isolated build requirements pin an incompatible setuptools.
"${python_bin}" -m pip download --no-cache-dir --no-deps --no-build-isolation \
    --no-binary=:all: \
    --dest "${source_dir}/source-builds" \
    "opencv-python-headless==${OPENCV_PYTHON_HEADLESS_VERSION}"
echo "${OPENCV_PYTHON_HEADLESS_SHA256}  ${opencv_sdist}" | sha256sum -c -
cp -a "${opencv_sdist}" "${artifact_root}/oss-sources/source-builds/"
export CMAKE_ARGS="-DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF \
    -DWITH_MSMF=OFF -DWITH_DSHOW=OFF -DWITH_AVFOUNDATION=OFF"
if [ -n "${MEDIA_WHEEL_DIR:-}" ]; then
    # Shared-base mode: compile once into a wheel, then install that wheel so the
    # policy check below still runs. Services reuse the wheel without recompiling.
    mkdir -p "${MEDIA_WHEEL_DIR}"
    "${python_bin}" -m pip wheel --no-cache-dir --no-build-isolation \
        --no-binary=opencv-python-headless \
        --wheel-dir "${MEDIA_WHEEL_DIR}" \
        "${opencv_sdist}"
    "${python_bin}" -m pip install --no-cache-dir --no-index \
        --find-links "${MEDIA_WHEEL_DIR}" \
        opencv-python-headless
else
    "${python_bin}" -m pip install --no-cache-dir --no-build-isolation \
        --no-binary=opencv-python-headless \
        "${opencv_sdist}"
fi
"${python_bin}" - <<'PY'
import cv2

build_info = cv2.getBuildInformation()
for backend in ("FFMPEG", "GStreamer", "v4l/v4l2", "Media Foundation", "DirectShow"):
    for line in build_info.splitlines():
        if line.strip().startswith(f"{backend}:") and line.rsplit(maxsplit=1)[-1] == "YES":
            raise SystemExit(f"ERROR: source-built OpenCV unexpectedly enabled {backend}")
print("OpenCV source build has general-purpose video backends disabled.")
PY
