# SPDX-FileCopyrightText: Copyright (c) 2021 OpenAI
# SPDX-FileCopyrightText: Copyright (c) 2023 Syliz517
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Minimal CLIP Vision Transformer (ViT-B/16) image encoder for vehicle ReID.

This is a tiny inference-only implementation compatible with the checkpoint
`ckpts/reid/clip_vehicleid.pt` (keys under `image_encoder.*`).

This implementation follows the CLIP ViT-B/16 architecture and OpenAI CLIP-style parameter naming.
The VehicleID ReID weights we support are sourced from the CLIP-ReID project (MIT).

We vendor this to avoid depending on `open_clip` / `open_clip_torch`.

Vendored notice:
- This file includes a minimal CLIP ViT implementation derived from upstream CLIP/CLIP-ReID codebases (MIT).
- SPDX-License-Identifier is `MIT AND Apache-2.0`: upstream MIT is preserved and NVIDIA
  modifications are licensed under Apache-2.0; see the SPDX-FileCopyrightText headers above.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VehicleCLIPViTConfig:
    image_size: int = 256
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    embed_dim: int = 512


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=False)
        self.ln_2 = nn.LayerNorm(d_model)
        # Match OpenAI CLIP naming: mlp.c_fc, mlp.c_proj
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(d_model, d_model * 4)  # type: ignore[attr-defined]
        self.mlp.gelu = QuickGELU()  # type: ignore[attr-defined]
        self.mlp.c_proj = nn.Linear(d_model * 4, d_model)  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        y = self.ln_2(x)
        y = self.mlp.c_proj(self.mlp.gelu(self.mlp.c_fc(y)))  # type: ignore[attr-defined]
        x = x + y
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.resblocks:
            x = blk(x)
        return x


class VehicleCLIPVisionTransformer(nn.Module):
    """
    Vision encoder that matches the key layout of `image_encoder.*` in clip_vehicleid.pt:
      - conv1.weight (patch embedding conv)
      - class_embedding
      - positional_embedding
      - ln_pre, transformer.resblocks.*, ln_post
      - proj (768x512)
    """

    def __init__(self, cfg: VehicleCLIPViTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        grid = cfg.image_size // cfg.patch_size
        num_patches = grid * grid

        self.conv1 = nn.Conv2d(3, cfg.width, kernel_size=cfg.patch_size, stride=cfg.patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(cfg.width))
        self.positional_embedding = nn.Parameter(torch.empty(num_patches + 1, cfg.width))
        self.ln_pre = nn.LayerNorm(cfg.width)

        self.transformer = Transformer(cfg.width, cfg.layers, cfg.heads)
        self.ln_post = nn.LayerNorm(cfg.width)
        self.proj = nn.Parameter(torch.empty(cfg.width, cfg.embed_dim))

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.proj, std=self.cfg.width**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) with H=W=cfg.image_size
        x = self.conv1(x)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid*grid)
        x = x.permute(0, 2, 1)  # (B, grid*grid, width)

        cls = self.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, width)
        x = x + self.positional_embedding.to(x.dtype).unsqueeze(0)
        x = self.ln_pre(x)

        # transformer expects (seq, batch, dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])  # CLS token
        x = x @ self.proj.to(x.dtype)  # (B, embed_dim)
        return x


@torch.inference_mode()
def load_vehicle_clip_vit_b16_256(
    weights_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> Tuple[VehicleCLIPVisionTransformer, VehicleCLIPViTConfig]:
    """
    Load `ckpts/reid/clip_vehicleid.pt` into a minimal ViT-B/16 @ 256 encoder.
    """
    cfg = VehicleCLIPViTConfig()
    model = VehicleCLIPVisionTransformer(cfg).to(device=device, dtype=dtype)

    raw: Dict[str, torch.Tensor] = torch.load(str(weights_path), map_location="cpu")
    # Keep only image encoder weights and strip prefix.
    prefix = "image_encoder."
    sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # We require all core vision keys present. If something is missing, surface it early.
    core_missing = [k for k in missing if not k.startswith("transformer.resblocks.")]
    if core_missing:
        raise RuntimeError(f"Vehicle CLIP encoder missing keys: {core_missing[:20]} (total={len(core_missing)})")
    if unexpected:
        # Usually safe to ignore if checkpoint has extras; but for our stripped sd this should be empty.
        raise RuntimeError(f"Vehicle CLIP encoder unexpected keys: {unexpected[:20]} (total={len(unexpected)})")

    model.eval()
    return model, cfg


def preprocess_vehicle_clip(
    crops_bgr_uint8: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    crops_bgr_uint8: uint8 tensor (B,3,H,W) in BGR order.
    Returns float tensor normalized for CLIP.
    """
    x = crops_bgr_uint8.to(device=device, dtype=torch.float32) / 255.0
    # BGR -> RGB
    x = x[:, [2, 1, 0], :, :]
    # CLIP mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.to(dtype=dtype)
