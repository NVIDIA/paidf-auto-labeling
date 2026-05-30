# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple


def b64_image_url(image_path: Path) -> str:
    b = Path(image_path).read_bytes()
    enc = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{enc}"


def vlm_messages_from_frames(frames: List[Tuple[float, Path]], prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for ts, p in frames:
        content.append({"type": "text", "text": f"<{ts:.1f} seconds>"})
        content.append({"type": "image_url", "image_url": {"url": b64_image_url(p)}})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def vlm_direct_mcq_messages_from_frames(
    frames: List[Tuple[float, Path]],
    mcq_prompt: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for ts, p in frames:
        content.append({"type": "text", "text": f"<{ts:.1f} seconds>"})
        content.append({"type": "image_url", "image_url": {"url": b64_image_url(p)}})
    content.append(
        {
            "type": "text",
            "text": (
                "Based on the provided video frames, generate the MCQ JSON for this window.\n"
                "Follow the system instructions strictly. Output the JSON object only."
            ),
        }
    )
    return [{"role": "system", "content": mcq_prompt}, {"role": "user", "content": content}]
