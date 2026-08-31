#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Remove Python distribution-build tooling from a final runtime virtual
# environment. Application containers do not build or install distributions at
# runtime; keeping these packages only expands the shipped dependency surface.

set -euo pipefail

venv_root="${1:?virtual-environment path is required}"

if [ ! -d "${venv_root}" ]; then
    echo "Virtual environment does not exist; skipping cleanup: ${venv_root}"
    exit 0
fi

shopt -s nullglob
for site_packages in "${venv_root}"/lib/python*/site-packages; do
    rm -rf \
        "${site_packages}/pip" \
        "${site_packages}"/pip-*.dist-info \
        "${site_packages}/setuptools" \
        "${site_packages}"/setuptools-*.dist-info \
        "${site_packages}/wheel" \
        "${site_packages}"/wheel-*.dist-info
done

rm -f \
    "${venv_root}/bin/pip" \
    "${venv_root}/bin/pip3" \
    "${venv_root}"/bin/pip3.* \
    "${venv_root}/bin/easy_install" \
    "${venv_root}"/bin/easy_install-*
