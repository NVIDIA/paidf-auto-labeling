# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import List

from al_utils.media_paths import IMAGE_EXTS, VIDEO_EXTS


def collect_videos(input_dir: Path) -> List[Path]:
    # NOTE: Despite the function name, window runners also support single-image inputs
    # (treated as a 1-frame clip). Extensions come from al_utils.media_paths so the
    # whitelist stays in sync with the OSRB-approved FFmpeg build's demuxers and
    # with mcq_generation.mcq.utils.video and vlm_json.runners.video_pipeline.
    exts = VIDEO_EXTS | IMAGE_EXTS
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []
    if input_dir.is_file():
        return [input_dir] if input_dir.suffix.lower() in exts else []
    try:
        vids = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        return sorted(set(vids))
    except Exception:
        return []


def derive_video_id(input_root: Path, video_path: Path) -> str:
    try:
        rel = Path(video_path).relative_to(Path(input_root))
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    stem = str(Path(video_path).stem or "").strip()
    if stem:
        return stem
    parent = str(Path(video_path).parent.name or "").strip()
    if parent:
        return parent
    return "unknown_video"


def resolve_output_video_id(*, input_root: Path, clip_path: Path, video_id_override: str = "") -> str:
    override = str(video_id_override or "").strip()
    if override:
        return override
    return derive_video_id(input_root, clip_path)
