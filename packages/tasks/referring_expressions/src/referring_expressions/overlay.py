# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw numbered box overlays so the VLM can ground phrases to marks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def draw_marked_boxes(
    image_path: Path,
    box_records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write an RGB overlay with 1-based mark ids on each box.

    Returns ``output_path``.
    """
    with Image.open(image_path) as src:
        image = src.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    for index, record in enumerate(box_records, start=1):
        bbox = [int(v) for v in record["bbox"]]
        x1, y1, x2, y2 = bbox
        color = _mark_color(index)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(index)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        pad = 2
        label_y = max(0, y1 - th - 2 * pad)
        draw.rectangle(
            [x1, label_y, x1 + tw + 2 * pad, label_y + th + 2 * pad],
            fill=color,
        )
        draw.text((x1 + pad, label_y + pad), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _mark_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 64, 64),
        (64, 200, 64),
        (64, 128, 255),
        (255, 200, 64),
        (200, 64, 255),
        (64, 220, 220),
        (255, 128, 64),
        (180, 180, 180),
    ]
    return palette[(index - 1) % len(palette)]


__all__ = ["draw_marked_boxes"]
