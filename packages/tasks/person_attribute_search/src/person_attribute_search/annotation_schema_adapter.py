# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Annotation schema adapter:

Model-free converter that merges the native UPA two-pass sidecars of a scene
into the legacy ``vad_pseudo_labelling_pipeline`` v1.0.0 per-chunk annotation
schema. The two passes of the ``person_attribute_anomaly_gen`` cookbook produce
the same information the legacy pipeline did, but spread across per-stage
sidecars:

    sidecars/person_attribute_search/pas.json            -> chunk.pas
    sidecars/person_attribute_search/chunk_queries.json  -> chunk.queries
    sidecars/visual_qa_anomaly/items.json (or visual_qa/) -> chunk.anomaly_gt
    sidecars/captioning/video_captions.json              -> scene/dense caption
    sidecars/captioning/metadata_chunk.json              -> fps/width/height/frames

For each scene the adapter writes the merged record beside the PAS sidecars:

  * ``sidecars/person_attribute_search/pas_anomaly.json`` - the merged
    pass1+pass2 unified record.

The pass-1 ``pas.json`` already lives under ``sidecars/person_attribute_search/``
(written by the PAS stage) and is not duplicated. Nothing is written into
``contextual/``; the merged record is a task-internal artifact, not a DAFT
contextual annotation.

The UPA sidecars are only read, never modified, so the adapter is additive and
idempotent. It mirrors the per-chunk record from the legacy
``script/utils/chunked_pipeline.py``; the legacy video-level rollup is
intentionally not produced (the two-pass workflow is per-chunk).

Parity caveats (by design):
  * The anomaly vote is single-model (UPA runs one VLM), so ``per_model_votes``
    has one entry and ``n_models_ok == 1``; ``is_strict_majority`` is True
    whenever any category fires. The legacy panel used multiple model votes.
  * ``scene_caption``/``dense_caption`` are aggregated from UPA per-window dense
    captions into the legacy ``{raw, parsed}`` shape; the legacy
    ``anomaly caption`` sub-field has no UPA counterpart and is left empty.
  * ``official_anomaly`` is supplied by the caller (e.g. the CLI's
    ``--dataset-json`` lookup); otherwise it is null.
  * Chunk ``start_time``/``end_time`` are clip-local (UPA chunks are standalone
    clips); the legacy values were offsets within the original video.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.utils.io import read_json, write_json

from person_attribute_search.anomaly_ontology import (
    is_dropped_source_label,
    normalize_kpi_category,
)

CHUNK_RE = re.compile(r"^(?P<stem>.+)_chunk_(?P<idx>\d+)$")
ANOM_PREFIX = "anom_"

PAS_FILENAME = "pas.json"
"""Pass-1 PAS sidecar filename (read from ``sidecars/person_attribute_search/``)."""

PAS_ANOMALY_FILENAME = "pas_anomaly.json"
"""Merged pass1+pass2 unified per-chunk record."""

DEFAULT_MODEL_NAME = "vlm"
"""Fallback anomaly-voter label when none is passed and none is in the sidecars."""


def _strip_mp4(name: str) -> str:
    """Drop a single trailing ``.mp4`` (UPA scene dirs are ``<clip>.mp4/``)."""
    return name[:-4] if name.endswith(".mp4") else name


def split_chunk_label(scene_dir_name: str) -> tuple[str, str, int]:
    """
    Split a scene dir name into ``(chunk_label, video_stem, chunk_id)``.

    ``gopro..._chunk_000.mp4`` -> ``("gopro..._chunk_000", "gopro...", 0)``. A
    clip without a ``_chunk_NNN`` suffix is treated as chunk 0 of itself.

    Args:
        scene_dir_name: The scene directory name (optionally ``.mp4``-suffixed).
    Returns:
        Tuple of chunk label, video stem, and integer chunk id.
    """
    label = _strip_mp4(scene_dir_name)
    match = CHUNK_RE.match(label)
    if match:
        return label, match.group("stem"), int(match.group("idx"))
    return label, label, 0


def _read_optional_json(path: Path) -> Any | None:
    """Load JSON, returning None when the file is absent (not an error)."""
    if not path.exists():
        return None
    return read_json(path)


_YES_RE = re.compile(r"\byes\b")


def _is_yes(answer: Any) -> bool:
    """Whether a BCQ answer string denotes 'Yes' (robust to 'A. Yes'/'Yes').

    Matches ``yes`` only as a standalone word so substrings such as ``eyes`` do
    not count as affirmative.
    """
    return isinstance(answer, str) and _YES_RE.search(answer.lower()) is not None


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS, matching the legacy dense-caption timeline."""
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def extract_anomaly_categories(visual_qa_items: dict[str, Any] | None) -> list[str]:
    """
    Categories whose ``anom_*`` BCQ was answered Yes, order preserved.

    Args:
        visual_qa_items: Parsed ``visual_qa`` items sidecar, or None.
    Returns:
        The fired anomaly category ids (``anom_`` prefix stripped).
    """
    if not isinstance(visual_qa_items, dict):
        return []
    categories: list[str] = []
    for item in visual_qa_items.get("items", []):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id", "")
        if item_id.startswith(ANOM_PREFIX) and _is_yes(item.get("answer")):
            categories.append(item_id[len(ANOM_PREFIX) :])
    return categories


def _has_anomaly_evidence(visual_qa_items: dict[str, Any] | None) -> bool:
    """True when at least one anomaly BCQ item (``anom_*``) is present.

    Distinguishes a confirmed-normal scene (anomaly BCQs were asked and all
    answered "No") from missing/incomplete pass-2 output (no anomaly BCQs
    recorded at all).
    """
    if not isinstance(visual_qa_items, dict):
        return False
    return any(
        isinstance(item, dict) and str(item.get("id", "")).startswith(ANOM_PREFIX)
        for item in visual_qa_items.get("items", [])
    )


def build_anomaly_gt(visual_qa_items: dict[str, Any] | None, model_name: str) -> dict[str, Any]:
    """
    Build the single-model ``anomaly_gt`` block in the legacy shape.

    An empty ``voted_categories`` with ``n_models_ok == 1`` implies a confirmed
    'normal' (anomaly BCQs were asked and all answered "No"). When no anomaly BCQ
    items are present the pass-2 output is missing/incomplete, so this fails
    closed with ``n_models_ok == 0`` and empty votes rather than minting a
    spurious 'normal' label.

    Args:
        visual_qa_items: Parsed anomaly ``visual_qa`` items sidecar, or None.
        model_name: Label recorded as the single voter in ``per_model_votes``.
    Returns:
        The ``anomaly_gt`` dictionary.
    """
    if not _has_anomaly_evidence(visual_qa_items):
        return {
            "voted_categories": [],
            "per_category_votes": {},
            "per_model_votes": {},
            "n_models_ok": 0,
            "voted_gt": None,
            "is_strict_majority": False,
        }
    categories = extract_anomaly_categories(visual_qa_items)
    return {
        "voted_categories": categories,
        "per_category_votes": dict.fromkeys(categories, 1),
        "per_model_votes": {model_name: categories},
        "n_models_ok": 1,
        "voted_gt": categories[0] if categories else None,
        "is_strict_majority": bool(categories),
    }


def build_captions(
    video_captions: dict[str, Any] | None, has_anomaly: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Aggregate UPA per-window captions into legacy scene + dense captions.

    Args:
        video_captions: Parsed ``captioning/video_captions.json``, or None.
        has_anomaly: Whether any anomaly fired (sets the dense ``anomaly`` flag).
    Returns:
        Tuple of ``(scene_caption, dense_caption)`` in legacy ``{raw, parsed}``
        shape; ``parsed`` is None when no caption text is available.
    """
    windows = (video_captions or {}).get("windows") or []
    timeline_lines = [
        f"{_fmt_timestamp(window.get('start_time', 0.0))} - "
        f"{_fmt_timestamp(window.get('end_time', 0.0))}: {window.get('caption', '').strip()}"
        for window in windows
        if window.get("caption")
    ]
    dense_text = "\n".join(timeline_lines)

    summary = (video_captions or {}).get("summary")
    scene_text = summary or (windows[0].get("caption", "").strip() if windows else "")

    scene_caption = {
        "raw": scene_text,
        "parsed": {"scene_caption": scene_text} if scene_text else None,
    }
    dense_caption = {
        "raw": dense_text,
        "parsed": (
            {
                "anomaly": "yes" if has_anomaly else "no",
                "anomaly caption": "",
                "dense caption": dense_text,
            }
            if dense_text
            else None
        ),
    }
    return scene_caption, dense_caption


def _normalize_query_pair(item: Any) -> list[str] | None:
    """Coerce one query entry into a ``[query, evidence]`` pair, or None."""
    if isinstance(item, str):
        text = item.strip()
        return [text, ""] if text else None
    if isinstance(item, (list, tuple)) and item:
        text = str(item[0]).strip()
        evidence = str(item[1]).strip() if len(item) > 1 else ""
        return [text, evidence] if text else None
    if isinstance(item, dict):
        text = str(item.get("query", "")).strip()
        evidence = str(item.get("source") or item.get("evidence") or "").strip()
        return [text, evidence] if text else None
    return None


def reshape_queries(chunk_queries: dict[str, Any] | None) -> dict[str, list]:
    """
    Convert UPA queries to the legacy ``{PAS, Anomaly, Caption}`` shape.

    When the PAS stage emits chunk-level ``query_buckets`` (legacy
    ``query_generation.py`` parity), all three buckets are passed through as
    ``[query, evidence]`` pairs. Otherwise the legacy fallback applies: the flat
    ``queries`` list lands entirely under ``PAS`` (each as ``[text, ""]``) and
    ``Anomaly``/``Caption`` stay empty.

    Args:
        chunk_queries: Parsed ``chunk_queries.json``, or None.
    Returns:
        Mapping with ``PAS``/``Anomaly``/``Caption`` query-pair lists.
    """
    buckets = (chunk_queries or {}).get("query_buckets")
    if isinstance(buckets, dict):
        return {
            bucket: [
                pair
                for item in (buckets.get(bucket) or [])
                if (pair := _normalize_query_pair(item)) is not None
            ]
            for bucket in ("PAS", "Anomaly", "Caption")
        }

    items = (chunk_queries or {}).get("queries") or []
    pas_pairs: list[list[str]] = []
    for entry in items:
        pair = _normalize_query_pair(entry)
        if pair is not None:
            pas_pairs.append(pair)
    return {"PAS": pas_pairs, "Anomaly": [], "Caption": []}


def _media_meta(metadata_chunk: dict[str, Any] | None, raw_video: Path | None) -> dict[str, Any]:
    """fps/width/height/num_frames from captioning metadata, else ffprobe."""
    media = (metadata_chunk or {}).get("media") or {}
    fps = media.get("framerate")
    width = media.get("width")
    height = media.get("height")
    num_frames = media.get("num_frames")
    if None not in (fps, width, height, num_frames):
        return {"fps": fps, "width": width, "height": height, "num_frames": num_frames}
    probed = _ffprobe(raw_video) if raw_video and raw_video.exists() else {}
    return {
        "fps": fps if fps is not None else probed.get("fps"),
        "width": width if width is not None else probed.get("width"),
        "height": height if height is not None else probed.get("height"),
        "num_frames": num_frames if num_frames is not None else probed.get("num_frames"),
    }


def _ffprobe(video: Path) -> dict[str, Any]:
    """Best-effort fps/width/height/num_frames via ffprobe (empty on failure)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {}
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
                "--",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        stream = (json.loads(out).get("streams") or [{}])[0]
        num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
        den_val = float(den) if den else 1.0
        fps = float(num) / den_val if den_val else None
        frames = stream.get("nb_frames")
        return {
            "fps": fps,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "num_frames": int(frames) if frames and str(frames).isdigit() else None,
        }
    except (subprocess.SubprocessError, ValueError, KeyError, OSError):
        return {}


def _find_raw_video(scene_dir: Path) -> Path | None:
    """Locate the chunk clip under ``raw/`` for the ffprobe fallback."""
    raw_dir = scene_dir / "raw"
    if raw_dir.is_dir():
        for candidate in sorted(raw_dir.glob("*.mp4")):
            return candidate
    return None


def _model_from_windows(windows_doc: Any) -> str | None:
    """First ``visual_qa_call.model`` recorded across a windows sidecar."""
    if not isinstance(windows_doc, dict):
        return None
    for window in windows_doc.get("windows", []):
        call = window.get("visual_qa_call") if isinstance(window, dict) else None
        if isinstance(call, dict):
            model = call.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
    return None


def detect_voter_model(sidecars: Path) -> str | None:
    """
    Resolve the VLM that produced the anomaly votes from the stage sidecars.

    Reads ``visual_qa_call.model`` from the anomaly windows sidecar, mirroring the
    items precedence (dedicated ``visual_qa_anomaly/`` first, then the default
    ``visual_qa/`` used by anomaly-only runs).

    Args:
        sidecars: The scene's ``sidecars/`` directory.
    Returns:
        The recorded model name, or None when no sidecar records one.
    """
    for sub in ("visual_qa_anomaly", "visual_qa"):
        model = _model_from_windows(_read_optional_json(sidecars / sub / "windows.json"))
        if model:
            return model
    return None


def build_chunk_record(
    scene_dir: Path, model_name: str | None = None, official_anomaly: str | None = None
) -> dict[str, Any] | None:
    """
    Assemble one legacy per-chunk record from a UPA scene directory.

    Args:
        scene_dir: The UPA scene directory (``<clip>.mp4/``).
        model_name: Label for the single anomaly voter. When None, it is
            auto-detected from the visual_qa sidecar, falling back to
            ``DEFAULT_MODEL_NAME``.
        official_anomaly: Designed anomaly category, or None when unknown.
    Returns:
        The merged legacy-shaped record, or None when the scene has no PAS
        sidecar (i.e. it is not a real chunk folder).
    """
    sidecars = scene_dir / "sidecars"
    pas_doc = _read_optional_json(sidecars / "person_attribute_search" / "pas.json")
    if pas_doc is None:
        return None

    chunk_label, video_stem, chunk_id = split_chunk_label(scene_dir.name)

    # Prefer the dedicated anomaly namespace; fall back to the default visual_qa/
    # (anomaly-only runs write there).
    visual_qa_items = _read_optional_json(sidecars / "visual_qa_anomaly" / "items.json")
    if visual_qa_items is None:
        visual_qa_items = _read_optional_json(sidecars / "visual_qa" / "items.json")

    voter = model_name or detect_voter_model(sidecars) or DEFAULT_MODEL_NAME
    anomaly_gt = build_anomaly_gt(visual_qa_items, voter)
    scene_caption, dense_caption = build_captions(
        _read_optional_json(sidecars / "captioning" / "video_captions.json"),
        has_anomaly=bool(anomaly_gt["voted_categories"]),
    )
    queries = reshape_queries(
        _read_optional_json(sidecars / "person_attribute_search" / "chunk_queries.json")
    )

    meta = _media_meta(
        _read_optional_json(sidecars / "captioning" / "metadata_chunk.json"),
        _find_raw_video(scene_dir),
    )
    num_frames = meta.get("num_frames")
    fps = meta.get("fps")
    duration = (num_frames / fps) if (num_frames and fps) else None
    frame_range = [0, num_frames - 1] if num_frames else None

    chunk_block = {
        "chunk_id": chunk_id,
        "start_time": 0.0,
        "end_time": duration,
        "frame_range": frame_range,
        "file": f"chunk_{chunk_id:03d}.mp4",
        "pas": pas_doc.get("pas", {"n_people": 0, "people": []}),
        "scene_caption": scene_caption,
        "dense_caption": dense_caption,
        "anomaly_gt": anomaly_gt,
        "queries": queries,
    }

    return {
        "video_stem": video_stem,
        "video_name": f"{video_stem}.mp4",
        "chunk_id": chunk_id,
        "chunk_label": chunk_label,
        "official_anomaly": official_anomaly,
        "fps": fps,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "chunk": chunk_block,
    }


def official_for(stem: str, lookup: dict[str, str] | None) -> str | None:
    """Resolve ``official_anomaly`` for a video stem from a lookup mapping."""
    if not lookup:
        return None
    return lookup.get(stem) or lookup.get(f"{stem}.mp4")


def convert_scene(
    scene_dir: Path, model_name: str | None = None, official_anomaly: str | None = None
) -> bool:
    """
    Write ``sidecars/person_attribute_search/pas_anomaly.json`` for a scene.

    Emits the merged pass1+pass2 anomaly record next to the PAS sidecars. The
    verbatim ``pas.json`` already lives under ``sidecars/person_attribute_search/``
    (written by the PAS stage), so it is not duplicated here, and nothing is
    written into ``contextual/``.

    Args:
        scene_dir: The UPA scene directory.
        model_name: Label for the single anomaly voter; auto-detected from the
            sidecars when None.
        official_anomaly: Designed anomaly category, or None.
    Returns:
        True if the scene was converted, False if it is not a chunk folder.
    """
    pas_sidecar_dir = scene_dir / "sidecars" / "person_attribute_search"
    pas_doc = _read_optional_json(pas_sidecar_dir / PAS_FILENAME)
    if pas_doc is None:
        return False
    record = build_chunk_record(scene_dir, model_name, official_anomaly)
    if record is None:
        return False
    # merged pass1+pass2 unified record, co-located with the PAS sidecars.
    write_json(pas_sidecar_dir / PAS_ANOMALY_FILENAME, record)
    return True


def convert_out_dir(
    out_dir: Path,
    model_name: str | None = None,
    official_lookup: dict[str, str] | None = None,
) -> dict[str, int]:
    """
    Convert every scene in a UPA out_dir; return simple counts.

    Args:
        out_dir: UPA output directory containing per-scene ``<clip>.mp4/`` dirs.
        model_name: Label for the single anomaly voter; auto-detected per scene
            from the sidecars when None.
        official_lookup: Optional video-stem -> official_anomaly mapping.
    Returns:
        Mapping ``{"scenes": <count converted>}``.
    """
    scenes_written = 0
    for scene_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        stem = split_chunk_label(scene_dir.name)[1]
        if convert_scene(scene_dir, model_name, official_for(stem, official_lookup)):
            scenes_written += 1
    return {"scenes": scenes_written}


_DATASET_CATEGORY_FIELDS = ("anomaly_type", "category", "label")
_DATASET_KEY_FIELDS = (
    "filename",
    "target_stem",
    "source_stem",
    "original_filename",
    "local_path",
)


def _dataset_candidate_keys(record_key: str, meta: dict[str, Any]) -> set[str]:
    """Filename/name/stem variants a dataset record may be keyed by."""
    candidates = {record_key, Path(record_key).name, Path(record_key).stem}
    for field in _DATASET_KEY_FIELDS:
        value = meta.get(field)
        if value:
            path = Path(str(value))
            candidates.update({str(value), path.name, path.stem})
    return {candidate for candidate in candidates if candidate}


def build_official_lookup(dataset_json: Path) -> dict[str, str]:
    """
    Build a ``{video_key: official_anomaly}`` lookup from a legacy dataset.json.

    Mirrors the legacy ``anomaly_ontology.kpi_official_lookup``: accepts either a
    top-level list of records or a ``{"videos": {...}}`` mapping, reads the
    designed category from the first present of
    ``anomaly_type``/``category``/``label``, and indexes it under every
    filename/stem variant of the record. Each category is normalized onto the
    canonical KPI ontology id via
    :func:`person_attribute_search.anomaly_ontology.normalize_kpi_category` (so source labels
    like ``shoplifting`` or ``person_falling`` resolve to their canonical ids).
    Genuinely-unknown labels are kept verbatim so non-KPI datasets still
    round-trip, but recognized yet intentionally-dropped aliases (e.g.
    ``abnormal``) are excluded rather than resurrected as official labels.

    Args:
        dataset_json: Path to the dataset metadata JSON.
    Returns:
        Mapping of video key (name/stem variants) to official anomaly category.
    """
    if not dataset_json.exists():
        return {}
    data = read_json(dataset_json)
    records: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, list):
        records = [
            (str(item.get("filename") or item.get("target_stem") or ""), item)
            for item in data
            if isinstance(item, dict)
        ]
    elif isinstance(data, dict) and isinstance(data.get("videos"), dict):
        records = [
            (str(key), item) for key, item in data["videos"].items() if isinstance(item, dict)
        ]

    lookup: dict[str, str] = {}
    for key, meta in records:
        raw_category = next(
            (
                str(meta[field]).strip()
                for field in _DATASET_CATEGORY_FIELDS
                if isinstance(meta.get(field), str) and meta[field].strip()
            ),
            None,
        )
        if not raw_category:
            continue
        normalized = normalize_kpi_category(raw_category)
        if normalized is not None:
            category = normalized
        elif is_dropped_source_label(raw_category):
            # Intentionally-dropped ambiguous alias (e.g. "abnormal"): do not
            # reintroduce it verbatim as an official label.
            continue
        else:
            # Genuinely unknown label (non-KPI dataset): keep verbatim so it
            # still round-trips instead of being silently dropped.
            category = raw_category
        for candidate in _dataset_candidate_keys(key, meta):
            lookup[candidate] = category
    return lookup


def _load_official_lookup(dataset_json: Path | None, official_json: Path | None) -> dict[str, str]:
    """Merge dataset.json and direct-mapping sources (the latter wins)."""
    lookup: dict[str, str] = {}
    if dataset_json is not None:
        lookup.update(build_official_lookup(dataset_json))
    if official_json is not None:
        direct = read_json(official_json)
        if not isinstance(direct, dict):
            raise ValueError("--official-json must contain a JSON object mapping")
        for key, value in direct.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"--official-json value for {key!r} must be a non-empty string, got {value!r}"
                )
            lookup[str(key)] = value.strip()
    return lookup


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint: merge a UPA out_dir's two-pass sidecars into the per-scene
    ``sidecars/person_attribute_search/pas_anomaly.json`` record.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="annotation_schema_adapter",
        description=(
            "Merge UPA two-pass sidecars into the legacy per-chunk annotation "
            "schema, writing sidecars/person_attribute_search/pas_anomaly.json "
            "(pass1+pass2 merged) per scene."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="UPA output directory containing per-scene <clip>.mp4/ folders.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Override the anomaly-voter label. Default: auto-detect from each "
            "scene's visual_qa sidecar, falling back to "
            f"'{DEFAULT_MODEL_NAME}'."
        ),
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="Optional legacy dataset.json used to populate official_anomaly.",
    )
    parser.add_argument(
        "--official-json",
        type=Path,
        default=None,
        help=(
            "Optional flat JSON object {video_stem: official_anomaly}; entries "
            "take precedence over --dataset-json."
        ),
    )
    args = parser.parse_args(argv)

    if not args.out_dir.is_dir():
        parser.error(f"--out-dir is not a directory: {args.out_dir}")
    try:
        lookup = _load_official_lookup(args.dataset_json, args.official_json)
    except ValueError as exc:
        parser.error(str(exc))

    result = convert_out_dir(args.out_dir, args.model_name, lookup or None)
    print(
        f"annotation_schema_adapter: wrote pas_anomaly artifacts for "
        f"{result['scenes']} scene(s) under {args.out_dir}"
    )
    return 0


__all__ = [
    "DEFAULT_MODEL_NAME",
    "PAS_FILENAME",
    "PAS_ANOMALY_FILENAME",
    "build_anomaly_gt",
    "build_captions",
    "build_chunk_record",
    "build_official_lookup",
    "convert_out_dir",
    "convert_scene",
    "detect_voter_model",
    "extract_anomaly_categories",
    "main",
    "official_for",
    "reshape_queries",
    "split_chunk_label",
]


if __name__ == "__main__":
    raise SystemExit(main())
