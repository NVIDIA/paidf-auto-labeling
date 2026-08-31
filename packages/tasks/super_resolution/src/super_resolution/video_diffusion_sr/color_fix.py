#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight color-fix utilities used by SeedVR inference scripts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToPILImage, ToTensor


@dataclass(frozen=True)
class _Stats:
    mean: Tensor
    std: Tensor


def _to_bchw(x: Image.Image) -> Tensor:
    tensor: Tensor = ToTensor()(x)
    return tensor.unsqueeze(0)


def _to_pil(x: Tensor) -> Image.Image:
    x = x.detach().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    image: Image.Image = ToPILImage()(x)
    return image


def _channel_stats(x: Tensor, eps: float = 1e-5) -> _Stats:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    return _Stats(mean=mean, std=(var + eps).sqrt())


def adain_color_fix(target: Image.Image, source: Image.Image, eps: float = 1e-5) -> Image.Image:
    """Match target's per-channel mean/std to source while keeping target content."""
    target_tensor = _to_bchw(target)
    source_tensor = _to_bchw(source)
    target_channels = int(target_tensor.shape[1])
    source_channels = int(source_tensor.shape[1])
    if target_channels != source_channels:
        raise ValueError(
            "adain_color_fix requires target and source images to have the same channel count: "
            f"target.mode={target.mode!r} target_channels={target_channels}, "
            f"source.mode={source.mode!r} source_channels={source_channels}"
        )

    target_stats = _channel_stats(target_tensor, eps=eps)
    source_stats = _channel_stats(source_tensor, eps=eps)

    normalized = (target_tensor - target_stats.mean) / target_stats.std
    out = normalized * source_stats.std + source_stats.mean
    return _to_pil(out)


def _gaussian_kernel_3x3(device: torch.device, dtype: torch.dtype) -> Tensor:
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return kernel / kernel.sum()


def _blur_bchw(x: Tensor, dilation: int) -> Tensor:
    kernel = _gaussian_kernel_3x3(x.device, x.dtype).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    x = functional.pad(x, (dilation, dilation, dilation, dilation), mode="replicate")
    return functional.conv2d(x, kernel, groups=x.shape[1], dilation=dilation)


def _decompose_multiscale(x: Tensor, levels: int = 5) -> tuple[Tensor, Tensor]:
    _validate_levels(levels)
    low = x
    high = torch.zeros_like(x)
    for i in range(levels):
        dilation = 2**i
        blurred = _blur_bchw(low, dilation=dilation)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _validate_levels(levels: int) -> None:
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got levels={levels}")


def _shape_tuple(x: Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in x.shape)


def _validate_matching_shapes(
    *,
    target_name: str,
    target: Tensor,
    source_name: str,
    source: Tensor,
    levels: int,
) -> None:
    if target.shape != source.shape:
        raise ValueError(
            f"{target_name} and {source_name} shapes must match before wavelet color fix: "
            f"{target_name}.shape={_shape_tuple(target)}, "
            f"{source_name}.shape={_shape_tuple(source)}, levels={levels}. "
            "_to_pil is the final conversion point after tensor reconstruction."
        )


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor, levels: int = 5) -> Tensor:
    """Backwards-compatible API used by SeedVR/SeedVR2 inference scripts."""
    _validate_levels(levels)
    _validate_matching_shapes(
        target_name="content_feat",
        target=content_feat,
        source_name="style_feat",
        source=style_feat,
        levels=levels,
    )
    style_feat = style_feat.to(device=content_feat.device, dtype=content_feat.dtype)
    content_high, _content_low = _decompose_multiscale(content_feat, levels=levels)
    _style_high, style_low = _decompose_multiscale(style_feat, levels=levels)
    return content_high + style_low


def wavelet_color_fix(target: Image.Image, source: Image.Image, levels: int = 5) -> Image.Image:
    """Preserve target details while taking source low-frequency color/illumination."""
    _validate_levels(levels)
    target_tensor = _to_bchw(target)
    source_tensor = _to_bchw(source)
    _validate_matching_shapes(
        target_name="target",
        target=target_tensor,
        source_name="source",
        source=source_tensor,
        levels=levels,
    )
    source_tensor = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
    target_high, _target_low = _decompose_multiscale(target_tensor, levels=levels)
    _source_high, source_low = _decompose_multiscale(source_tensor, levels=levels)
    return _to_pil(target_high + source_low)
