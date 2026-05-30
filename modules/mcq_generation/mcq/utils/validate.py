# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from daft_export.paths import scene_paths

DEFAULT_CAPTION_KEY = "vlm_caption"
DEFAULT_ENHANCED_CAPTION_KEY = "llm_enhanced_caption"


def is_enhanced_mcq_payload(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("mcq"), list)


def read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def is_task_file_valid(path: Path) -> bool:
    """True if ``path`` is a task-output file with at least one item."""
    obj = read_json_object(path)
    if not obj:
        return False
    items = obj.get("items")
    return isinstance(items, list) and len(items) > 0


def is_empty_marker_valid(path: Path) -> bool:
    """True if ``path`` is the explicit zero-task resume marker."""
    obj = read_json_object(path)
    if not obj:
        return False
    if obj.get("_error") != "zero_task_items":
        return False
    mcq_items = obj.get("mcq")
    return isinstance(mcq_items, list) and len(mcq_items) == 0


def is_metadata_valid(
    path: Path,
    *,
    direct_mcq_from_vlm: bool,
    caption_key: str,
    enhanced_caption_key: str,
) -> bool:
    obj = read_json_object(path)
    if not obj:
        return False
    windows = obj.get("windows", None)
    if not isinstance(windows, list) or len(windows) == 0:
        return False
    cap_key = str(caption_key or "").strip() or DEFAULT_CAPTION_KEY
    enh_key = str(enhanced_caption_key or "").strip() or DEFAULT_ENHANCED_CAPTION_KEY
    for w in windows:
        if not isinstance(w, dict):
            return False
        if "start_frame" not in w or "end_frame" not in w:
            return False
        # Enhanced payload must exist and be a parsed MCQ JSON object.
        enh_v = w.get(enh_key)
        if not is_enhanced_mcq_payload(enh_v):
            return False
        # For VLM->caption->LLM flows, captions must exist and be non-empty.
        if not direct_mcq_from_vlm:
            cap_v = w.get(cap_key)
            if not (isinstance(cap_v, str) and cap_v.strip()):
                return False
    return True


def is_vlm_verify_valid(path: Path, *, expected_windows_total: Optional[int] = None) -> bool:
    obj = read_json_object(path)
    if not obj:
        return False
    summary = obj.get("summary")
    if not isinstance(summary, dict):
        return False
    windows_total = summary.get("windows_total")
    questions_verified = summary.get("questions_verified")
    if not isinstance(windows_total, int) or windows_total <= 0:
        return False
    if expected_windows_total is not None and windows_total != int(expected_windows_total):
        return False
    if not isinstance(questions_verified, int) or questions_verified <= 0:
        return False
    return True


def outputs_complete(
    *,
    out_dir: Path,
    direct_mcq_from_vlm: bool,
    caption_key: str,
    enhanced_caption_key: str,
    require_vlm_verify: bool = False,
) -> bool:
    """True if ``out_dir`` already holds a complete run for the skip-on-resume check.

    ``out_dir`` is the scene root. "Complete" requires:
      * a valid window-metadata sidecar, and
      * either a non-empty task file (MCQ, BCQ, or open QA), or an ``mcq.empty.json``
        marker recording that a prior run completed task composition with zero items.

    When ``require_vlm_verify`` is set, a verifier sidecar covering every
    window must also be present and valid.
    """
    paths = scene_paths(out_dir)
    sidecar_meta = paths.sidecars_dir / "metadata.json"
    if not is_metadata_valid(
        sidecar_meta,
        direct_mcq_from_vlm=direct_mcq_from_vlm,
        caption_key=caption_key,
        enhanced_caption_key=enhanced_caption_key,
    ):
        return False

    task_complete = (
        is_task_file_valid(paths.task_mcq)
        or is_task_file_valid(paths.task_bcq)
        or is_task_file_valid(paths.task_open_qa)
        or is_empty_marker_valid(paths.sidecars_dir / "mcq.empty.json")
    )
    if not task_complete:
        return False

    if require_vlm_verify:
        meta_obj = read_json_object(sidecar_meta) or {}
        wins = meta_obj.get("windows") if isinstance(meta_obj, dict) else None
        expected_total = len(wins) if isinstance(wins, list) else None
        verify_path = paths.sidecars_dir / "mcq.vlm_verify.json"
        if not is_vlm_verify_valid(verify_path, expected_windows_total=expected_total):
            return False

    return True
