#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

profile="${1:-vp9-output}"
policy_script="${2:-/tmp/media-build/ffmpeg_codec_policy.py}"
artifact_root="${3:-/opt/media-runtime}"
prefix="${4:-/usr/local}"
source_dir="${5:-/usr/share/oss-sources}"

: "${FFMPEG_VERSION:?FFMPEG_VERSION is required}"
: "${FFMPEG_URL:?FFMPEG_URL is required}"
: "${FFMPEG_SHA256:?FFMPEG_SHA256 is required}"
: "${NV_CODEC_HEADERS_VERSION:?NV_CODEC_HEADERS_VERSION is required}"
: "${NV_CODEC_HEADERS_URL:?NV_CODEC_HEADERS_URL is required}"
: "${NV_CODEC_HEADERS_SHA256:?NV_CODEC_HEADERS_SHA256 is required}"
: "${LIBVPX_VERSION:?LIBVPX_VERSION is required}"
: "${LIBVPX_URL:?LIBVPX_URL is required}"
: "${LIBVPX_SHA256:?LIBVPX_SHA256 is required}"
: "${MEDIA_BUILD_JOBS:?MEDIA_BUILD_JOBS is required}"

rm -rf /tmp/ffmpeg-build
mkdir -p /tmp/ffmpeg-build "${source_dir}/source-builds"
cd /tmp/ffmpeg-build
libvpx_archive="libvpx-${LIBVPX_VERSION}.tar.xz"
curl -fsSL "${LIBVPX_URL}" -o "${libvpx_archive}"
echo "${LIBVPX_SHA256}  ${libvpx_archive}" | sha256sum -c -
cp "${libvpx_archive}" "${source_dir}/source-builds/${libvpx_archive}"
headers_archive="nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz"
curl -fsSL "${NV_CODEC_HEADERS_URL}" -o "${headers_archive}"
echo "${NV_CODEC_HEADERS_SHA256}  ${headers_archive}" | sha256sum -c -
cp "${headers_archive}" "${source_dir}/source-builds/${headers_archive}"
mkdir nv-codec-headers
tar -xf "${headers_archive}" --strip-components=1 -C nv-codec-headers
make -C nv-codec-headers -j"${MEDIA_BUILD_JOBS}"
make -C nv-codec-headers PREFIX="${prefix}" install
export PKG_CONFIG_PATH="${prefix}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

curl -fsSL "${FFMPEG_URL}" -o ffmpeg.tar.xz
echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c -
cp ffmpeg.tar.xz \
    "${source_dir}/source-builds/ffmpeg-${FFMPEG_VERSION}.tar.xz"
tar -xf ffmpeg.tar.xz
cd "ffmpeg-${FFMPEG_VERSION}"
mapfile -t codec_flags < <(python3 "${policy_script}" configure-flags "${profile}")
./configure \
    --prefix="${prefix}" \
    --disable-debug \
    --disable-doc \
    --disable-static \
    --enable-shared \
    --enable-pic \
    --disable-nonfree \
    --disable-gpl \
    --disable-vulkan \
    --disable-cuda-nvcc \
    --disable-encoders \
    --disable-decoders \
    --disable-everything \
    "${codec_flags[@]}" \
    --disable-parser=aac \
    --disable-parser=hevc \
    --extra-cflags="-I${prefix}/include" \
    --extra-ldflags="-L${prefix}/lib"
make -j"${MEDIA_BUILD_JOBS}"
make install
ldconfig
rm -rf /tmp/ffmpeg-build

mkdir -p "${artifact_root}/bin" "${artifact_root}/lib"
cp -a "${prefix}/bin/ffmpeg" "${prefix}/bin/ffprobe" "${artifact_root}/bin/"
cp -a "${prefix}"/lib/libavcodec.so* "${artifact_root}/lib/"
cp -a "${prefix}"/lib/libavdevice.so* "${artifact_root}/lib/"
cp -a "${prefix}"/lib/libavfilter.so* "${artifact_root}/lib/"
cp -a "${prefix}"/lib/libavformat.so* "${artifact_root}/lib/"
cp -a "${prefix}"/lib/libavutil.so* "${artifact_root}/lib/"
cp -a "${prefix}"/lib/libswresample.so* "${artifact_root}/lib/"
cp -a "${prefix}"/lib/libswscale.so* "${artifact_root}/lib/"
cp -a /usr/lib/*/libvpx.so* "${artifact_root}/lib/"
mkdir -p "${artifact_root}/oss-sources"
cp -a "${source_dir}/." "${artifact_root}/oss-sources/"
