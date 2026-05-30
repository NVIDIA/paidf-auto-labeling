#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight color-fix utilities used by some SR inference scripts.

This file is intentionally self-contained (PyTorch + PIL + torchvision) and
implements two practical "color alignment" options:

- AdaIN-based mean/variance matching in RGB space
- A simple multi-scale blur decomposition that swaps low-frequency color

All functions operate in float RGB in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToPILImage, ToTensor


@dataclass(frozen=True)
class _Stats:
    mean: Tensor  # (B,C,1,1)
    std: Tensor  # (B,C,1,1)


def _to_bchw(x: Image.Image) -> Tensor:
    t = ToTensor()(x)  # (C,H,W) in [0,1]
    return t.unsqueeze(0)  # (1,C,H,W)


def _to_pil(x: Tensor) -> Image.Image:
    x = x.detach().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    return ToPILImage()(x)


def _channel_stats(x: Tensor, eps: float = 1e-5) -> _Stats:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    std = (var + eps).sqrt()
    return _Stats(mean=mean, std=std)


def adain_color_fix(target: Image.Image, source: Image.Image, eps: float = 1e-5) -> Image.Image:
    """
    Match target's per-channel mean/std to source (AdaIN-style).

    - target: image whose spatial content we keep
    - source: image whose global color/illumination we match
    """
    t = _to_bchw(target)
    s = _to_bchw(source)

    ts = _channel_stats(t, eps=eps)
    ss = _channel_stats(s, eps=eps)

    t_norm = (t - ts.mean) / ts.std
    out = t_norm * ss.std + ss.mean
    return _to_pil(out)


def _gaussian_kernel_3x3(device: torch.device, dtype: torch.dtype) -> Tensor:
    k = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    k = k / k.sum()
    return k


def _blur_bchw(x: Tensor, dilation: int) -> Tensor:
    # Depthwise 3x3 blur (approx. Gaussian), dilation controls effective radius.
    k = _gaussian_kernel_3x3(x.device, x.dtype).view(1, 1, 3, 3)
    k = k.repeat(x.shape[1], 1, 1, 1)  # (C,1,3,3)
    pad = dilation
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(x, k, groups=x.shape[1], dilation=dilation)


def _decompose_multiscale(x: Tensor, levels: int = 5) -> Tuple[Tensor, Tensor]:
    """
    Simple multi-scale decomposition:
    - low: repeated blurred version (color/illumination)
    - high: residuals accumulated across scales (detail)
    """
    low = x
    high = torch.zeros_like(x)
    for i in range(levels):
        d = 2**i
        blurred = _blur_bchw(low, dilation=d)
        high = high + (low - blurred)
        low = blurred
    return high, low


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor, levels: int = 5) -> Tensor:
    """
    Backwards-compatible API used by SeedVR/SeedVR2 inference scripts.

    Both inputs are expected to be 4D tensors shaped like BCHW. In this repo, the
    scripts pass TCHW (treating time as batch), which is also valid.

    Returns:
      content_high_freq + style_low_freq
    """
    content_high, _content_low = _decompose_multiscale(content_feat, levels=levels)
    _style_high, style_low = _decompose_multiscale(style_feat, levels=levels)
    return content_high + style_low


def wavelet_color_fix(target: Image.Image, source: Image.Image, levels: int = 5) -> Image.Image:
    """
    Preserve target details while taking source low-frequency color/illumination.
    """
    t = _to_bchw(target)
    s = _to_bchw(source)
    t_high, _t_low = _decompose_multiscale(t, levels=levels)
    _s_high, s_low = _decompose_multiscale(s, levels=levels)
    out = t_high + s_low
    return _to_pil(out)
