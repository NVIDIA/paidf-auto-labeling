# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Fixed non-root runtime user (Sonar S6471).
# UID/GID are remapped at container startup by the entrypoint via gosu — no build-time args needed.
RUN groupadd -g 1000 nvidia && useradd -m -u 1000 -g 1000 -s /bin/bash nvidia

# Use bash for better RUN scripting.
SHELL ["/bin/bash", "-lc"]

# -------- OSS compliance: snapshot base image package list --------
# Used at the END of the build (after every apt install/upgrade/purge) to
# compute the delta of packages we install on top of the base, so we can
# ship their upstream source code at /usr/share/oss-sources/ so the source
# travels with every pull. Capturing the snapshot after the FINAL purge means
# build-only deps (cmake, build-essential, yasm, ...) that we install and
# then remove don't bloat the bake — only sources for packages actually
# present in the final image get fetched.
RUN dpkg-query -W -f='${Package}\n' | LC_ALL=C sort -u > /tmp/__oss_base_pkgs.txt

# -------- OS deps (video IO + build toolchain) --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    yasm \
    nasm \
    # OpenCV (built from sdist with WITH_FFMPEG=OFF) needs these image codec headers.
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    # Keep FFmpeg LGPL (avoid OpenSSL); enable TLS via GnuTLS
    libgnutls28-dev \
    # gosu: safe privilege-drop in entrypoint (runtime UID/GID remapping)
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# -------- FFmpeg (LGPL-only build, allow-list whitelist) --------
# Rationale (per OSS Review Board / legal guidance):
# - Ubuntu's `apt install ffmpeg` typically enables GPL components (e.g. libx264/libx265).
# - `https://git.ffmpeg.org/ffmpeg.git` can be flaky/blocked in CI (exit code 128 on clone).
# - We build FFmpeg from a release tarball + nv-codec-headers, with an allow-list
#   approach: `--disable-everything` strips every encoder / decoder / muxer /
#   demuxer / parser / protocol / bsf / filter, then we explicitly re-enable
#   only the LGPL-clean codecs we need + the minimal IO scaffolding to actually
#   read and write mp4 / mjpeg files end-to-end.
# - Allowed video codecs:
#     encoders:  h264_nvenc (NVIDIA NVENC HW), mpeg4 (SW fallback), mjpeg (SW)
#     decoders:  h264_cuvid (NVIDIA NVDEC HW, full HW decode — no SW H.264
#                 algorithm code in our binary), mpeg4 (SW), mjpeg (SW)
# - We intentionally do NOT ship the SW `h264` decoder: MPEG-LA's AVC patent
#   pool covers H.264 decode as well as encode, and shipping a SW H.264 decoder
#   in an OSS container would put royalty obligations on downstream redistribution.
#   The standalone `h264_cuvid` decoder dispatches the full decode to NVDEC
#   silicon (NVIDIA already pays the silicon-level MPEG-LA license), so no
#   H.264 algorithm code lives in our binary.
# - PyAV writers prefer h264_nvenc on NVENC-equipped GPUs (for example RTX PRO,
#   RTX 40/50, A10/A40/L40-class GPUs) and fall back to mpeg4 on GPUs without
#   NVENC (for example A100 / H100 NVL). H.264 input always goes through NVDEC.
ARG FFMPEG_VERSION=8.1
# sha256 of https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz —
# bump together with FFMPEG_VERSION. Verified against the tarball used by the
# published image.
ARG FFMPEG_SHA256=b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a
ARG NV_CODEC_HEADERS_VERSION=n12.1.14.0
RUN set -euxo pipefail; \
    mkdir -p /tmp/ffmpeg-build /usr/share/oss-sources/source-builds && cd /tmp/ffmpeg-build; \
    git clone --depth 1 --branch "${NV_CODEC_HEADERS_VERSION}" https://github.com/FFmpeg/nv-codec-headers.git; \
    # OSS compliance: preserve upstream source for nv-codec-headers (MIT).
    tar -C /tmp/ffmpeg-build --exclude='.git' \
        -czf "/usr/share/oss-sources/source-builds/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" nv-codec-headers; \
    make -C nv-codec-headers -j"$(nproc)" && make -C nv-codec-headers install; \
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz; \
    # Verify upstream tarball before preserve / extract. Abort on mismatch.
    echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c -; \
    # OSS compliance: preserve upstream FFmpeg source tarball (LGPL).
    cp ffmpeg.tar.xz "/usr/share/oss-sources/source-builds/ffmpeg-${FFMPEG_VERSION}.tar.xz"; \
    tar -xf ffmpeg.tar.xz; \
    cd "ffmpeg-${FFMPEG_VERSION}"; \
    ./configure \
      `# --- Build setup (install path, link mode, deps) ---` \
      --prefix=/usr/local \
      --disable-debug \
      --disable-doc \
      --disable-static \
      --enable-shared \
      --enable-pic \
      --enable-gnutls \
      `# --- LGPL guard (no GPL/nonfree/HW-licensed components) ---` \
      --disable-nonfree \
      --disable-gpl \
      --disable-vulkan \
      --disable-cuda-nvcc \
      `# --- Legal-approved codec allow-list (per OSRB, 2026-05-23) ---` \
      `# Strip every component, then re-enable only:` \
      `#   encoders: h264_nvenc (HW), mpeg4 (SW fallback), mjpeg (SW)` \
      `#   decoders: h264_cuvid (HW-only, no SW H.264 algorithm in binary),` \
      `#             mpeg4 (SW), mjpeg (SW)` \
      `# We deliberately do NOT enable --enable-decoder=h264 (the SW H.264` \
      `# decoder) because the MPEG-LA AVC patent pool covers H.264 decode as` \
      `# well as encode. h264_cuvid dispatches the full decode to NVDEC silicon` \
      `# (NVIDIA has already paid the silicon-level MPEG-LA license).` \
      --disable-encoders \
      --disable-decoders \
      --disable-everything \
      --enable-encoder=h264_nvenc \
      --enable-encoder=mpeg4 \
      --enable-encoder=mjpeg \
      --enable-decoder=h264_cuvid \
      --enable-decoder=mpeg4 \
      --enable-decoder=mjpeg \
      --enable-nvenc \
      --enable-nvdec \
      `# --- Minimal IO scaffolding (every flag below is empirically required) ---` \
      `# protocol=file:        all av.open()/ffmpeg -i path/-y path calls` \
      `# muxer=mp4:            SR + detection + tracking write .mp4 output` \
      `# demuxer=mov:          read .mp4/.mov/.m4v input (one demuxer handles all)` \
      `# parser=h264:          slice NAL units from container before feeding cuvid` \
      `# bsf=h264_mp4toannexb: h264_cuvid wants Annex-B; MP4 stores AVCC` \
      `# muxer=image2:         MCQ sample_frames_ffmpeg writes JPEG frames` \
      `# filter=fps,scale,format: sample_frames_ffmpeg filter graph (fallback uses 'format')` \
      --enable-protocol=file \
      --enable-muxer=mp4 \
      --enable-muxer=image2 \
      --enable-demuxer=mov \
      --enable-parser=h264 \
      --enable-bsf=h264_mp4toannexb \
      --enable-filter=fps \
      --enable-filter=scale \
      --enable-filter=format \
      --extra-cflags="-I/usr/local/include" \
      --extra-ldflags="-L/usr/local/lib"; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    rm -rf /tmp/ffmpeg-build; \
    \
    # Build-time verification: must NOT contain GPL/nonfree flags.
    ffmpeg -hide_banner -version | head -n 3; \
    ! ffmpeg -buildconf | grep -E "enable-gpl|enable-nonfree"; \
    # Sanity: the only H.264 encoder in the image must be h264_nvenc.
    h264_encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null | awk '$2 ~ /^h264_/ {print $2}' | sort -u)"; \
    h264_count="$(printf '%s\n' "${h264_encoders}" | awk 'NF {c++} END {print c+0}')"; \
    if [[ "${h264_count}" -ne 1 || "${h264_encoders}" != "h264_nvenc" ]]; then \
      echo "ERROR: expected exactly one h264_* encoder (h264_nvenc), got: ${h264_encoders:-<none>}" >&2; \
      exit 1; \
    fi; \
    # Sanity: GPL / patent-encumbered third-party encoder libs must NOT appear.
    # The h264_*-prefix check above misses libx264 / libopenh264 / libx265 because
    # those don't carry the "h264_" prefix. Belt + suspenders alongside the
    # --disable-gpl / --disable-nonfree configure flags above.
    for forbidden in libx264 libopenh264 libx265; do \
      if ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$forbidden" >/dev/null; then \
        echo "ERROR: forbidden encoder '$forbidden' leaked into the image (GPL/patent-encumbered)" >&2; \
        exit 1; \
      fi; \
    done; \
    # Sanity: mpeg4 + mjpeg SW encoders must be present (used as fallback when NVENC is absent).
    for enc in mpeg4 mjpeg; do \
      ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$enc" >/dev/null \
        || { echo "ERROR: expected encoder '$enc' to be enabled" >&2; exit 1; }; \
    done; \
    # Sanity: mpeg4 + mjpeg + h264_cuvid (HW-only NVDEC) decoders must be present.
    # We must NOT ship the SW `h264` decoder (MPEG-LA royalty exposure).
    for dec in mpeg4 mjpeg h264_cuvid; do \
      ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$dec" >/dev/null \
        || { echo "ERROR: expected decoder '$dec' to be enabled" >&2; exit 1; }; \
    done; \
    # Sanity: h264_cuvid must be the only H.264-family decoder present (royalty exposure).
    h264_decoders="$(ffmpeg -hide_banner -decoders 2>/dev/null | awk '$2 ~ /^h264/ {print $2}' | sort -u)"; \
    forbidden_h264_decoders="$(printf '%s\n' "${h264_decoders}" | awk 'NF && $0 != "h264_cuvid"')"; \
    if [[ -n "${forbidden_h264_decoders}" ]]; then \
      echo "ERROR: forbidden H.264 decoder(s) leaked into the image: ${forbidden_h264_decoders} (h264_cuvid is the only allowed H.264 decode path)" >&2; \
      exit 1; \
    fi; \
    # Sanity: minimal IO scaffolding must be present (else read/write any file fails).
    # The checks below probe the binary directly so we catch silent regressions
    # from --disable-everything dropping things we expect to be enabled above.
    ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx mp4    >/dev/null || { echo "ERROR: muxer mp4 missing"   >&2; exit 1; }; \
    ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx image2 >/dev/null || { echo "ERROR: muxer image2 missing">&2; exit 1; }; \
    `# mov demuxer is registered under combined name 'mov,mp4,m4a,3gp,3g2,mj2'.` \
    ffmpeg -hide_banner -demuxers 2>/dev/null | awk '{print $2}' | grep -E '(^|,)mov(,|$)' >/dev/null || { echo "ERROR: demuxer mov missing" >&2; exit 1; }; \
    `# -protocols / -bsfs output is indented; strip leading whitespace before grep.` \
    ffmpeg -hide_banner -protocols 2>/dev/null | awk '{print $1}' | grep -Fx file              >/dev/null || { echo "ERROR: protocol file missing">&2; exit 1; }; \
    ffmpeg -hide_banner -bsfs      2>/dev/null | awk '{print $1}' | grep -Fx h264_mp4toannexb  >/dev/null || { echo "ERROR: bsf h264_mp4toannexb missing">&2; exit 1; }; \
    for filt in fps scale format; do \
      ffmpeg -hide_banner -filters 2>/dev/null | awk '{print $2}' | grep -Fx "$filt" >/dev/null \
        || { echo "ERROR: filter '$filt' missing" >&2; exit 1; }; \
    done

RUN apt-get purge -y --auto-remove yasm nasm && rm -rf /var/lib/apt/lists/*

# -------- uv (fast Python package manager) --------
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

# -------- Build args for CUDA/PyTorch/Extensions --------
ARG PYTHON_VERSION=3.13.13
# Choose which CUDA wheel index to use for torch/torchvision/torchaudio.
# Common values: cu121, cu128
ARG TORCH_CUDA_INDEX=cu128
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0

# flash-attn (required: SeedVR imports flash_attn directly)
ARG FLASH_ATTN_VERSION=2.8.3

# OpenCV is rebuilt from sdist with WITH_FFMPEG=OFF for LGPL compliance.
# Video I/O lives in PyAV instead; cv2 is only used for image ops + drawing.
ARG OPENCV_PYTHON_VERSION=4.12.0.88

# -------- uv defaults --------
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=0
ENV UV_PYTHON_DOWNLOADS=manual
ENV UV_PYTHON=python${PYTHON_VERSION}

WORKDIR /workspace

# Copy dependency files first for better layer caching (do not copy full repo yet).
COPY pyproject.toml /workspace/pyproject.toml
COPY uv.lock /workspace/uv.lock

# Install Python via uv (cached across code changes).
RUN uv python install ${PYTHON_VERSION}

# Patch vendored deps in uv's bundled setuptools.
RUN /root/.local/share/uv/python/cpython-${PYTHON_VERSION}-linux-x86_64-gnu/bin/python3 \
    -m pip install --no-cache-dir --break-system-packages --upgrade "setuptools>=78.1.1"

# Create venv outside /workspace so it is not shadowed by the repo bind-mount (..:/workspace).
RUN uv venv /opt/venv --python ${PYTHON_VERSION}
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1) Repo python deps from uv.lock into /opt/venv.
#    Export requirements.txt from the lockfile, then install into the venv.
RUN uv export --frozen --no-dev --no-emit-project --output-file /tmp/requirements.txt \
    && uv pip install --no-cache --requirements /tmp/requirements.txt \
    && uv pip install --no-cache "pip>=25.3"

# 1b) Drop unused rfdetr training/dataset transitive deps.
#     `roboflow` (and its only-consumer `rf100vl`) are pulled in by rfdetr's
#     install_requires for training/dataset utilities. We only use rfdetr for
#     inference, never `import roboflow` / `import rf100vl`, so remove them to
#     shrink the image and silence pip's resolver warning about roboflow's
#     hard pin on opencv-python-headless==4.10.0.84 (we ship a source-built
#     opencv-python in step 2 below).
RUN python -m pip uninstall -y roboflow rf100vl

# 2) Replace the FFmpeg-bundled OpenCV wheel with a CMake source build.
#    The PyPI opencv-python wheel ships its own FFmpeg shared libs; we don't want that.
#    Use opencv-python sdist + CMAKE_ARGS to force WITH_FFMPEG=OFF, plus
#    ENABLE_HEADLESS=1 / WITH_QT=OFF / WITH_GTK=OFF to skip GUI bindings.
RUN <<'BASH'
set -euxo pipefail
python -m pip uninstall -y \
    opencv-python opencv-python-headless \
    opencv-contrib-python opencv-contrib-python-headless || true
rm -rf /opt/venv/lib/python*/site-packages/cv2 \
       /opt/venv/lib/python*/site-packages/opencv_python.libs \
       /opt/venv/lib/python*/site-packages/opencv_python-*.dist-info \
       /opt/venv/lib/python*/site-packages/opencv_python_headless.libs \
       /opt/venv/lib/python*/site-packages/opencv_python_headless-*.dist-info \
       /opt/venv/lib/python*/site-packages/opencv_contrib_python*.dist-info \
       /opt/venv/lib/python*/site-packages/opencv_contrib_python*.libs
# Build deps must live in the venv because we use --no-build-isolation
# (default isolation pulls a stale setuptools that breaks on Python 3.13:
#  AttributeError: module 'pkgutil' has no attribute 'ImpImporter').
uv pip install --no-cache "setuptools>=78.1.1" "wheel>=0.46.2" \
    "scikit-build>=0.18,<1" "cmake>=3.27,<4" "numpy>=2.1.0,<3" "pip>=25.3"
export ENABLE_CONTRIB=0
# ENABLE_HEADLESS=1 tells opencv-python's build to skip GUI deps (Qt/GTK).
# We also pass -DWITH_QT=OFF -DWITH_GTK=OFF as belt-and-suspenders.
export ENABLE_HEADLESS=1
export CMAKE_ARGS="\
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_V4L=OFF \
    -DWITH_LIBV4L=OFF \
    -DWITH_QT=OFF \
    -DWITH_GTK=OFF \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF"
python -m pip install --no-cache-dir --no-build-isolation \
    --no-binary=opencv-python \
    "opencv-python==${OPENCV_PYTHON_VERSION}"
python - <<'PY'
import re
import subprocess
from pathlib import Path
import cv2
info = cv2.getBuildInformation()
if re.search(r"FFMPEG:\s+YES", info):
    raise SystemExit("OpenCV unexpectedly built WITH FFmpeg")
cv2_pkg_dir = Path(cv2.__file__).resolve().parent
cv2_sos = sorted(cv2_pkg_dir.rglob("cv2*.so"))
if not cv2_sos:
    raise SystemExit(f"Could not locate cv2 compiled extension under {cv2_pkg_dir}")
forbidden = ("libavcodec", "libavdevice", "libavfilter",
             "libavformat", "libavutil", "libswresample", "libswscale")
for cv2_so in cv2_sos:
    ldd = subprocess.check_output(["ldd", str(cv2_so)], text=True)
    for lib in forbidden:
        if lib in ldd:
            raise SystemExit(f"{cv2_so.name} still links {lib}")
print("opencv", cv2.__version__, "FFmpeg disabled, no libav* linkage in", [p.name for p in cv2_sos])
PY
# Defensive scan: fail if any opencv_python*.libs/libav* slipped back in
# (e.g. a stale wheel reinstall by another layer down the line).
leftovers="$(find /opt/venv/lib/python*/site-packages \( \
    -path '*opencv_python*.libs/libavcodec*' -o \
    -path '*opencv_python*.libs/libavdevice*' -o \
    -path '*opencv_python*.libs/libavfilter*' -o \
    -path '*opencv_python*.libs/libavformat*' -o \
    -path '*opencv_python*.libs/libavutil*' -o \
    -path '*opencv_python*.libs/libswresample*' -o \
    -path '*opencv_python*.libs/libswscale*' \
\) -print 2>/dev/null || true)"
if [ -n "$leftovers" ]; then echo "$leftovers"; exit 1; fi
BASH

# 3) Install PyTorch with CUDA (overrides the PyPI torch from uv sync).
RUN uv pip install --no-cache \
      --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX} \
      torch==${TORCH_VERSION}+${TORCH_CUDA_INDEX} \
      torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_INDEX} \
      torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_INDEX}

# 4) flash-attn (required). Builds from source inside the CUDA devel image.
#    Need setuptools for the build.
RUN uv pip install --no-cache "setuptools>=78.1.1" "wheel>=0.46.2" \
    && uv pip install --no-cache --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"

# 5) PyAV: rebuild from source and link to our LGPL-only /usr/local FFmpeg libs.
# Rationale:
# - The PyPI wheel for `av` bundles its own FFmpeg libs (often including libx264/libx265).
# - For LGPL-friendly builds, force a source build that links against our /usr/local FFmpeg
#   (configured with --disable-gpl --disable-nonfree).
RUN set -euxo pipefail; \
    # Ensure pip exists in this uv-managed venv, then force a *source* build of PyAV.
    python -m ensurepip --upgrade; \
    python -m pip --version; \
    python -m pip uninstall -y av || true; \
    rm -rf /opt/venv/lib/python*/site-packages/av /opt/venv/lib/python*/site-packages/av-*.dist-info /opt/venv/lib/python*/site-packages/av.libs || true; \
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"; \
    # Install PyAV from the git tag (source build) to link against our LGPL-only FFmpeg.
    uv pip install --no-cache "setuptools>=78.1.1" "cython>=3.1.0,<4"; \
    python -m pip install --no-cache-dir --no-build-isolation "av @ git+https://github.com/PyAV-Org/PyAV@v16.1.0"; \
    python -c "import av, av._core; print('pyav', av.__version__, 'core', av._core.__file__)"

# -------- SeedVR clone baked into image for sr_runner --------
# Pinned clone so the image is self-contained; override at build with --build-arg if needed.
ARG SEEDVR_REPO_URL=https://github.com/ByteDance-Seed/SeedVR.git
ARG SEEDVR_GIT_SHA=e4de8c24441a67e1b7df56abea10645059bb1185
COPY modules/sr_runner/video_diffusion_sr/color_fix.py /tmp/seedvr_color_fix.py
COPY modules/sr_runner/inference_seedvr2_window.py /tmp/seedvr_inference_seedvr2_window.py
RUN set -euxo pipefail; \
    git init /opt/seedvr && cd /opt/seedvr && \
    git remote add origin "${SEEDVR_REPO_URL}" && \
    git fetch --depth 1 origin "${SEEDVR_GIT_SHA}" && \
    git checkout --detach FETCH_HEAD && \
    # OSS compliance: preserve upstream SeedVR source (Apache-2.0) before we mutate the tree.
    mkdir -p /usr/share/oss-sources/source-builds && \
    git archive --format=tar.gz \
        --output="/usr/share/oss-sources/source-builds/SeedVR-${SEEDVR_GIT_SHA:0:7}.tar.gz" HEAD && \
    mkdir -p /opt/seedvr/projects/video_diffusion_sr && \
    cp /tmp/seedvr_color_fix.py /opt/seedvr/projects/video_diffusion_sr/color_fix.py && \
    cp /tmp/seedvr_inference_seedvr2_window.py /opt/seedvr/projects/inference_seedvr2_window.py && \
    sed -i 's/fusedrms/rms/g; s/fusedln/layer/g' \
        /opt/seedvr/configs_3b/main.yaml \
        /opt/seedvr/configs_7b/main.yaml && \
    ! grep -E "fusedrms|fusedln" /opt/seedvr/configs_3b/main.yaml /opt/seedvr/configs_7b/main.yaml && \
    test -d /opt/seedvr/common && test -d /opt/seedvr/projects && \
    test -f /opt/seedvr/pos_emb.pt && test -f /opt/seedvr/neg_emb.pt && \
    chown -R nvidia:nvidia /opt/seedvr && \
    rm -f /tmp/seedvr_color_fix.py /tmp/seedvr_inference_seedvr2_window.py
ENV SEEDVR_ROOT=/opt/seedvr

# Remove build-only packages to reduce attack surface.
RUN apt-get purge -y --auto-remove \
    build-essential cmake ninja-build pkg-config \
    git \
    gnupg gnupg2 \
    libgnutls28-dev \
    && rm -rf /var/lib/apt/lists/*

# -------- OSS compliance: bake upstream source for every Debian package in final image (vs base) --------
# Runs AFTER every apt install/upgrade/purge so the diff against the base
# snapshot captures only the packages actually present in the shipped image
# (not build-only deps like cmake/git/yasm that we install + purge mid-build).
# Output goes to /usr/share/oss-sources/debian/ so the source travels with
# every pull.
RUN <<'OSS_DEB_BAKE'
set -euxo pipefail

mkdir -p /usr/share/oss-sources/debian
cd /usr/share/oss-sources/debian

# Enable deb-src in whichever sources format the base image uses.
#  - one-line .list (Ubuntu 22.04 / Debian classic): un-comment `# deb-src` lines.
#  - deb822 .sources (Ubuntu 24.04+ default): add `deb-src` to `Types:`.
sed -i 's/^# deb-src/deb-src/g' /etc/apt/sources.list 2>/dev/null || true
find /etc/apt/sources.list.d -name '*.list' -print0 2>/dev/null \
    | xargs -0 -r sed -i 's/^# deb-src/deb-src/g'
for f in /etc/apt/sources.list.d/*.sources; do
    [ -f "$f" ] || continue
    grep -q '^Types:.*deb-src' "$f" || sed -i 's/^Types: deb$/Types: deb deb-src/' "$f"
done
apt-get update -qq

# Diff currently-installed binary pkgs against the base snapshot, resolve each
# to its source pkg via dpkg, dedupe, fetch. `set -e` makes any apt-get source
# failure fail the build, so we never ship an incomplete bake.
# Name-only diff (no ${Version}) is intentional: base-image package upgrades
# (e.g. `apt-get upgrade` security patches) are out of scope — base image
# owner's responsibility.
dpkg-query -W -f='${Package}\n' | LC_ALL=C sort -u > /tmp/__oss_after_pkgs.txt
LC_ALL=C comm -13 /tmp/__oss_base_pkgs.txt /tmp/__oss_after_pkgs.txt \
    > NEW_PACKAGES.txt
echo "=== OSS: new Debian packages installed on top of base ==="
cat NEW_PACKAGES.txt

src_pkgs=$(while read -r p; do
    [ -z "$p" ] && continue
    dpkg-query -W -f='${source:Package}\n' "$p"
done < NEW_PACKAGES.txt | LC_ALL=C sort -u)
echo "=== OSS: unique source packages to fetch ==="
echo "$src_pkgs" | tee SOURCE_PACKAGES.txt

for src in $src_pkgs; do
    apt-get source --download-only "$src"
done

echo "Bundled $(ls *.dsc 2>/dev/null | wc -l) source packages ($(du -sh . | awk '{print $1}'))"
rm -rf /var/lib/apt/lists/* /tmp/__oss_*.txt
OSS_DEB_BAKE

# Copy the full repo last (so code changes don't bust the dependency cache).
COPY . /workspace

RUN cp /NGC-DL-CONTAINER-LICENSE /workspace/NGC-DL-CONTAINER-LICENSE \
 && chmod 0644 /workspace/NGC-DL-CONTAINER-LICENSE /workspace/LICENSE-3rd-party.txt /workspace/VERSION

# Entrypoint: install outside /workspace so it works without bind-mount.
COPY docker/entrypoint.sh /usr/local/bin/auto-labeling-entrypoint.sh
RUN chmod +x /usr/local/bin/auto-labeling-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/auto-labeling-entrypoint.sh"]

# uv defaults (can be overridden at runtime).
# - Ensure `uv run ...` uses the prebuilt venv instead of creating /workspace/.venv.
# - Don't let `uv run` mutate /opt/venv inside the container.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_NO_SYNC=1

# Helpful defaults for multi-GPU / NCCL; override at runtime as needed.
ENV NCCL_DEBUG=WARN

# Request the NVIDIA Container Runtime to expose the video capability inside
# the container. The default is "compute,utility" which omits libnvcuvid.so
# (NVDEC userspace) and libnvidia-encode.so (NVENC userspace). Without "video"
# the FFmpeg/PyAV h264_cuvid + h264_nvenc paths fail at runtime with
# "Cannot load libnvcuvid.so.1" / "Cannot load libnvidia-encode.so.1".
# Adding "video" here ensures all containers from this image can use the
# hardware H.264 decode/encode paths without the operator having to remember
# to pass `--env NVIDIA_DRIVER_CAPABILITIES=...` at `docker run` time.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# uv-managed Python defaults to /root/.local/share/uv/python when installed as root.
# The venv interpreter may point into that tree; ensure it is executable by the non-root runtime user.
RUN set -eux; \
    chmod 755 /root; \
    if [[ -d /root/.local/share/uv ]]; then chmod -R a+rX /root/.local/share/uv; fi

# Ensure common runtime-write paths exist for the non-root user (for non-bind-mount runs).
RUN set -euxo pipefail; \
    mkdir -p /workspace/output /workspace/logs /workspace/ckpts; \
    chown -R nvidia:nvidia /workspace/output /workspace/logs /workspace/ckpts

USER nvidia
CMD ["/bin/bash"]
