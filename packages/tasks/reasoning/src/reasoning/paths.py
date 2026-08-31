# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical on-disk layout of a DAFT scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenePaths:
    """Canonical DAFT file paths for a single scene.

    A DAFT scene has a fixed on-disk layout::

        <scene>/
        ├── raw/<media_id>.<ext>               # analyzed media
        ├── contextual/
        │   ├── video.json | image.json        # scene-level metadata (one or the other)
        │   ├── events.json                    # temporal events (video scenes only)
        │   ├── instances.json                 # tracked-object catalogue
        │   ├── objects.json                   # per-frame 2D detections
        │   ├── tracking.json                  # 4D MOT (per-frame 3D pose; opt-in)
        │   ├── chunks.json                    # dense temporal captions (window MCQ runs)
        │   └── msted.json                     # LLM-aggregated event characterization (opt-in)
        ├── task/
        │   ├── mcq.json                       # multi-choice questions
        │   ├── bcq.json                       # binary (Yes/No) questions
        │   ├── open_qa.json                   # open-ended questions
        │   ├── scene_description.json         # captioning task (image + video)
        │   ├── video_summarization.json       # whole-video summary (video only)
        │   ├── temporal_description.json      # per-window dense captions (video only)
        │   ├── temporal_localization.json     # query-driven event grounding (opt-in)
        │   ├── mcq_openended.json             # MCQ with open-ended explanations (opt-in)
        │   ├── bcq_openended.json             # Yes/No with open-ended explanations (opt-in)
        │   └── causal_linkage.json            # (t1, t2) causal grounding (opt-in)
        └── sidecars/                          # non-DAFT diagnostic files

    ``<media_id>`` is the stem of the original input filename.

    Image scenes use ``image.json`` (no fps/duration) instead of
    ``video.json`` and do not write ``events.json``.
    """

    scene_dir: Path

    raw_dir: Path

    contextual_dir: Path
    contextual_video: Path
    contextual_image: Path
    contextual_events: Path
    contextual_instances: Path
    contextual_objects: Path
    contextual_tracking: Path
    contextual_chunks: Path
    contextual_msted: Path

    task_dir: Path
    task_mcq: Path
    task_bcq: Path
    task_open_qa: Path
    task_scene_description: Path
    task_video_summarization: Path
    task_temporal_description: Path
    task_temporal_localization: Path
    task_mcq_openended: Path
    task_bcq_openended: Path
    task_causal_linkage: Path

    sidecars_dir: Path
    sidecar_metadata: Path
    sidecar_metadata_chunk: Path
    # Reasoning-owned diagnostic sidecars (non-DAFT). ``anomaly.json``
    # carries the anomaly verdict + re-perception request; it lives under
    # ``sidecars/reasoning/`` because the DAFT v3 type set is closed and
    # cannot host an "anomaly" type without a core-schema change.
    sidecar_anomaly: Path
    # Person-attribute-search deliverable (``sidecars/person_attribute_search/
    # pas.json``) produced by the PAS pass. Read-only cross-stage evidence the
    # anomaly stage can fold into its prompt; absent when PAS did not run.
    sidecar_pas: Path


def scene_paths(scene_dir: Path | str) -> ScenePaths:
    """Return canonical DAFT path handles rooted at ``scene_dir``.

    Pure path construction — no filesystem side effects. Call
    ``ensure_scene_skeleton`` to materialize the empty directory tree.
    """
    root = Path(scene_dir)
    return ScenePaths(
        scene_dir=root,
        raw_dir=root / "raw",
        contextual_dir=root / "contextual",
        contextual_video=root / "contextual" / "video.json",
        contextual_image=root / "contextual" / "image.json",
        contextual_events=root / "contextual" / "events.json",
        contextual_instances=root / "contextual" / "instances.json",
        contextual_objects=root / "contextual" / "objects.json",
        contextual_tracking=root / "contextual" / "tracking.json",
        contextual_chunks=root / "contextual" / "chunks.json",
        contextual_msted=root / "contextual" / "msted.json",
        task_dir=root / "task",
        task_mcq=root / "task" / "mcq.json",
        task_bcq=root / "task" / "bcq.json",
        task_open_qa=root / "task" / "open_qa.json",
        task_scene_description=root / "task" / "scene_description.json",
        task_video_summarization=root / "task" / "video_summarization.json",
        task_temporal_description=root / "task" / "temporal_description.json",
        task_temporal_localization=root / "task" / "temporal_localization.json",
        task_mcq_openended=root / "task" / "mcq_openended.json",
        task_bcq_openended=root / "task" / "bcq_openended.json",
        task_causal_linkage=root / "task" / "causal_linkage.json",
        sidecars_dir=root / "sidecars",
        sidecar_metadata=root / "sidecars" / "metadata.json",
        sidecar_metadata_chunk=root / "sidecars" / "metadata_chunk.json",
        sidecar_anomaly=root / "sidecars" / "reasoning" / "anomaly.json",
        sidecar_pas=root / "sidecars" / "person_attribute_search" / "pas.json",
    )


def resolve_raw_media(scene_dir: Path | str) -> Path | None:
    """Return the media file under ``<scene>/raw/``, or ``None`` if absent.

    The raw directory contains the analyzed media, either as a symlink for
    normal local inputs or as a copied file for remote-staged inputs. This
    helper returns the first file found, regardless of name — callers should
    not assume a specific stem.
    """
    raw = Path(scene_dir) / "raw"
    if not raw.is_dir():
        return None
    candidates = sorted(p for p in raw.iterdir() if p.is_file() or p.is_symlink())
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"ambiguous raw media under {scene_dir!s}/raw: {names}")
    return candidates[0]


def ensure_scene_skeleton(scene_dir: Path | str) -> ScenePaths:
    """Create ``raw/``, ``contextual/``, ``task/``, ``sidecars/`` under ``scene_dir``.

    Idempotent. Called once per scene before DAFT export stages run. Returns the
    same ``ScenePaths`` as ``scene_paths(scene_dir)`` for convenience.
    """
    paths = scene_paths(scene_dir)
    for d in (paths.raw_dir, paths.contextual_dir, paths.task_dir, paths.sidecars_dir):
        d.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = ["ScenePaths", "ensure_scene_skeleton", "resolve_raw_media", "scene_paths"]
