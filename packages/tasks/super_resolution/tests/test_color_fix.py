# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from PIL import Image
from super_resolution.video_diffusion_sr.color_fix import (
    _decompose_multiscale,
    adain_color_fix,
    wavelet_color_fix,
    wavelet_reconstruction,
)
from torch import Tensor


def test_adain_color_fix_happy_path() -> None:
    target = Image.new("RGB", (8, 8), color="black")
    source = Image.new("RGB", (8, 8), color="white")

    output = adain_color_fix(target, source)

    assert isinstance(output, Image.Image)
    assert output.size == target.size
    assert output.mode == "RGB"


def test_wavelet_color_fix_happy_path() -> None:
    target = Image.new("RGB", (8, 8), color="black")
    source = Image.new("RGB", (8, 8), color="white")

    output = wavelet_color_fix(target, source)

    assert isinstance(output, Image.Image)
    assert output.size == target.size
    assert output.mode == "RGB"


def test_decompose_multiscale_rejects_invalid_levels() -> None:
    tensor = torch.zeros((1, 3, 8, 8))

    with pytest.raises(ValueError, match="levels must be >= 1, got levels=0"):
        _decompose_multiscale(tensor, levels=0)


def test_adain_color_fix_rejects_mismatched_channels() -> None:
    target = Image.new("RGB", (8, 8), color="black")
    source = Image.new("L", (8, 8), color=255)

    with pytest.raises(ValueError, match="target_channels=3.*source_channels=1"):
        adain_color_fix(target, source)


def test_wavelet_reconstruction_rejects_mismatched_shapes() -> None:
    content = torch.zeros((1, 3, 8, 8))
    style = torch.zeros((1, 3, 9, 8))

    with pytest.raises(ValueError, match=r"content_feat\.shape=.*style_feat\.shape=.*levels=5"):
        wavelet_reconstruction(content, style)


def test_wavelet_reconstruction_casts_style_to_content_dtype() -> None:
    content = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    style = torch.ones((1, 3, 8, 8), dtype=torch.float64)

    result: Tensor = wavelet_reconstruction(content, style, levels=1)

    assert result.dtype == content.dtype
    assert result.device == content.device


def test_wavelet_color_fix_rejects_mismatched_image_shapes() -> None:
    target = Image.new("RGB", (8, 8), color="black")
    source = Image.new("RGB", (9, 8), color="white")

    with pytest.raises(ValueError, match=r"target\.shape=.*source\.shape=.*_to_pil"):
        wavelet_color_fix(target, source)
