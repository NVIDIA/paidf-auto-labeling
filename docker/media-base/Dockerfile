# syntax=docker/dockerfile:1.7
#
# Shared media-toolchain base image.
#
# Compiles the controlled media stack (FFmpeg for every profile, plus OpenCV and
# PyAV wheels for the vp9-output profile) exactly once. Service builder stages
# start FROM this image instead of recompiling the stack per service, which is
# what pushes the emulated arm64 builds over the CI timeout.
#
# Build one image per (base family, profile):
#   docker buildx build -f docker/media-base/Dockerfile \
#     --build-arg BASE_IMAGE=nvcr.io/nvidia/base/ubuntu:noble-<digest> \
#     --build-arg PROFILE=vp9-output -t <registry>/media-base:ubuntu-vp9 .
#
# The wheels are ABI-tied to MEDIA_PYTHON, so the base must share the same
# interpreter the consuming service venv uses (uv's UV_PYTHON).

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS media-base

ARG PROFILE=vp9-output
ARG MEDIA_PYTHON=/usr/bin/python3.12
ARG WHEEL_DIR=/opt/media-wheels

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libvpx-dev \
        nasm \
        pkg-config \
        python3 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        xz-utils \
        yasm \
    && rm -rf /var/lib/apt/lists/*

# Copy only the FFmpeg build inputs first. The OpenCV/PyAV installers are copied
# after the FFmpeg build below, so editing a wheel installer cannot invalidate
# the hours-long FFmpeg layer.
COPY scripts/media-toolchain.toml scripts/media_toolchain.py scripts/ffmpeg_codec_policy.py \
    scripts/build_restricted_ffmpeg.sh /opt/media-build/

# FFmpeg is system-wide (/usr/local + /opt/media-runtime) and shared by every profile.
RUN python3 /opt/media-build/media_toolchain.py ffmpeg-install --profile "${PROFILE}"

# Copied after FFmpeg so wheel-installer edits reuse the cached FFmpeg build.
COPY scripts/install_opencv_headless_from_source.sh \
    scripts/install_pyav_from_source.sh scripts/prune_runtime_python.sh /opt/media-build/

# vp9-output additionally prebuilds the OpenCV/PyAV wheels into WHEEL_DIR. The
# install-and-verify inside the scripts still runs, so a policy violation fails
# the base build rather than every downstream service.
# Build the wheels inside a venv. Ubuntu disables ensurepip for the system
# python and marks it externally-managed (PEP 668), so pip can't run against
# /usr/bin/python3.12 directly. A venv off the same interpreter has its own pip
# and no PEP 668 guard; the wheels stay cp312 so service venvs still consume them.
RUN if [ "${PROFILE}" = "vp9-output" ]; then \
        "${MEDIA_PYTHON}" -m venv /opt/media-venv \
        && /opt/media-venv/bin/python /opt/media-build/media_toolchain.py python-install \
            --python /opt/media-venv/bin/python \
            --wheel-dir "${WHEEL_DIR}" \
            --component opencv-headless \
            --component pyav; \
    else \
        mkdir -p "${WHEEL_DIR}"; \
    fi

# Fail fast on the FFmpeg codec/license surface (GPL/nonfree, unexpected codecs).
# `verify` scans the interpreter it's pointed at, so for vp9-output check both the
# system python and the venv that holds the OpenCV/PyAV wheels; a wheel-bundled
# FFmpeg/libav in either must fail the build. input-only has no venv.
RUN python3 /opt/media-build/media_toolchain.py verify --profile "${PROFILE}"
RUN if [ "${PROFILE}" = "vp9-output" ]; then \
        python3 /opt/media-build/media_toolchain.py verify --profile "${PROFILE}" \
            --python /opt/media-venv/bin/python; \
    fi

# The venv exists only to build and verify the shared media wheels.  Remove
# distribution-build tooling before publishing the base image.
RUN bash /opt/media-build/prune_runtime_python.sh /opt/media-venv

ENV UPA_MEDIA_WHEEL_DIR=${WHEEL_DIR} \
    UPA_MEDIA_PROFILE=${PROFILE}
