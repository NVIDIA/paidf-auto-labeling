# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Metropolis v3 task annotations to TAO DAFT training formats."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.formats.daft.types import TASK_TYPES
from core.utils.io import read_json, write_json

COSMOS_REASON_VERSION = "cosmos-reason-v1.0"
TAO_VL_REASON_FORMAT = "tao-vl-reason-v1.0"
DEFAULT_TAO_VL_REASON_LICENSE = "CC BY-NC-ND 4.0"

SUPPORTED_TRAINING_TASKS: frozenset[str] = frozenset(TASK_TYPES)
_ANSWER_INSTRUCTIONS: dict[str, str] = {
    "bcq": "Answer with Yes or No, optionally followed by a brief explanation.",
    "bcq_openended": "Answer with Yes or No, optionally followed by a brief explanation.",
    "mcq": (
        "Choose the correct option by letter, optionally followed by the option label "
        "or a brief explanation."
    ),
    "mcq_openended": (
        "Choose the correct option by letter, optionally followed by the option label "
        "or a brief explanation."
    ),
    "temporal_localization": (
        "Provide the result in json format with 'mm:ss' for time depiction. "
        "Use keywords 'start', 'end' in the json output."
    ),
}


class TrainingFormatError(ValueError):
    """Raised when a training-format export cannot be constructed."""


@dataclass
class TrainingConversionResult:
    """Summary returned by TAO DAFT training-format exporters."""

    samples_written: int = 0
    samples_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)

    def is_success(self) -> bool:
        """Return True when conversion produced samples and no hard errors."""
        return self.samples_written > 0 and not self.errors

    def extend(self, other: TrainingConversionResult) -> None:
        """Merge another conversion result into this one."""
        self.samples_written += other.samples_written
        self.samples_skipped += other.samples_skipped
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        self.paths.extend(other.paths)


def convert_metropolis_scene_to_tao_vl_reason(
    scene_dir: Path | str,
    output_dir: Path | str,
    *,
    task_types: Iterable[str] | None = None,
    copy_media: bool = True,
    metadata: Mapping[str, Any] | None = None,
    emit_media_root_as_null: bool = False,
) -> TrainingConversionResult:
    """Convert one Metropolis scene directory to ``tao-vl-reason-v1.0`` files."""
    scene_path = Path(scene_dir)
    return _convert_metropolis_scenes_to_tao_vl_reason(
        [scene_path],
        Path(output_dir),
        media_anchor=scene_path,
        task_types=task_types,
        copy_media=copy_media,
        metadata=metadata,
        emit_media_root_as_null=emit_media_root_as_null,
    )


def convert_metropolis_dataset_to_tao_vl_reason(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    task_types: Iterable[str] | None = None,
    copy_media: bool = True,
    metadata: Mapping[str, Any] | None = None,
    emit_media_root_as_null: bool = False,
) -> TrainingConversionResult:
    """Convert all Metropolis scenes under ``dataset_dir`` to TAO VL annotations."""
    dataset_path = Path(dataset_dir)
    scenes = _find_metropolis_scenes(dataset_path)
    if not scenes:
        return TrainingConversionResult(
            errors=[f"No Metropolis scenes with task/ under {dataset_path}"]
        )
    return _convert_metropolis_scenes_to_tao_vl_reason(
        scenes,
        Path(output_dir),
        media_anchor=dataset_path,
        task_types=task_types,
        copy_media=copy_media,
        metadata=metadata,
        emit_media_root_as_null=emit_media_root_as_null,
    )


def convert_metropolis_scenes_to_tao_vl_reason(
    scene_dirs: Iterable[Path | str],
    output_dir: Path | str,
    *,
    media_anchor: Path | str | None = None,
    task_types: Iterable[str] | None = None,
    copy_media: bool = True,
    metadata: Mapping[str, Any] | None = None,
    emit_media_root_as_null: bool = False,
) -> TrainingConversionResult:
    """Convert explicit Metropolis scene directories to TAO VL annotations."""
    scenes = [Path(scene_dir) for scene_dir in scene_dirs]
    if not scenes:
        return TrainingConversionResult(errors=["No Metropolis scene directories were provided"])
    anchor = Path(media_anchor) if media_anchor is not None else _common_parent(scenes)
    return _convert_metropolis_scenes_to_tao_vl_reason(
        scenes,
        Path(output_dir),
        media_anchor=anchor,
        task_types=task_types,
        copy_media=copy_media,
        metadata=metadata,
        emit_media_root_as_null=emit_media_root_as_null,
    )


def convert_metropolis_scene_to_cosmos_reason(
    scene_dir: Path | str,
    output_dir: Path | str,
    *,
    task_types: Iterable[str] | None = None,
    copy_media: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingConversionResult:
    """Convert one Metropolis scene directory to a ``cosmos-reason-v1.0`` dataset."""
    scene_path = Path(scene_dir)
    return _convert_metropolis_scenes_to_cosmos_reason(
        [scene_path],
        Path(output_dir),
        media_anchor=scene_path,
        task_types=task_types,
        copy_media=copy_media,
        metadata=metadata,
    )


def convert_metropolis_scenes_to_cosmos_reason(
    scene_dirs: Iterable[Path | str],
    output_dir: Path | str,
    *,
    media_anchor: Path | str | None = None,
    task_types: Iterable[str] | None = None,
    copy_media: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingConversionResult:
    """Convert explicit Metropolis scene directories to one Cosmos Reason dataset."""
    scenes = [Path(scene_dir) for scene_dir in scene_dirs]
    if not scenes:
        return TrainingConversionResult(errors=["No Metropolis scene directories were provided"])
    anchor = Path(media_anchor) if media_anchor is not None else _common_parent(scenes)
    return _convert_metropolis_scenes_to_cosmos_reason(
        scenes,
        Path(output_dir),
        media_anchor=anchor,
        task_types=task_types,
        copy_media=copy_media,
        metadata=metadata,
    )


def convert_metropolis_dataset_to_cosmos_reason(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    task_types: Iterable[str] | None = None,
    copy_media: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingConversionResult:
    """Convert all Metropolis scenes under ``dataset_dir`` to one Cosmos Reason dataset."""
    dataset_path = Path(dataset_dir)
    scenes = _find_metropolis_scenes(dataset_path)
    if not scenes:
        return TrainingConversionResult(
            errors=[f"No Metropolis scenes with task/ under {dataset_path}"]
        )
    return _convert_metropolis_scenes_to_cosmos_reason(
        scenes,
        Path(output_dir),
        media_anchor=dataset_path,
        task_types=task_types,
        copy_media=copy_media,
        metadata=metadata,
    )


def build_tao_vl_reason_annotation(
    task_type: str,
    items: Iterable[Mapping[str, Any]],
    *,
    media_root: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``tao-vl-reason-v1.0`` annotation payload."""
    item_list = [dict(item) for item in items]
    if not item_list:
        raise TrainingFormatError("tao-vl-reason annotation requires at least one item")

    meta = _metadata_block("annotation", metadata)
    meta["task"] = task_type
    meta.setdefault("license", DEFAULT_TAO_VL_REASON_LICENSE)
    return {
        "format": TAO_VL_REASON_FORMAT,
        "metadata": meta,
        "media_root": media_root,
        "items": item_list,
    }


def build_cosmos_reason_meta(
    samples: Iterable[Mapping[str, str]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``cosmos-reason-v1.0`` ``meta.json`` payload."""
    sample_list = [dict(sample) for sample in samples]
    if not sample_list:
        raise TrainingFormatError("cosmos-reason meta.json requires at least one sample")
    return {
        "version": COSMOS_REASON_VERSION,
        "metadata": _metadata_block("meta", metadata),
        "samples": sample_list,
    }


def build_cosmos_reason_conversation(
    task_type: str,
    item: Mapping[str, Any],
    *,
    is_image: bool,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``cosmos-reason-v1.0`` conversation payload from a task item."""
    question = _compose_question(task_type, item)
    answer = _format_answer(task_type, item)
    if not question or answer is None:
        raise TrainingFormatError(f"Cannot build conversation for task {task_type!r}")

    media_type = "image" if is_image else "video"
    user_content = [
        {"type": media_type, media_type: f"{media_type}_0"},
        {"type": "text", "text": question},
    ]
    assistant_turn: dict[str, Any] = {"role": "assistant"}
    reasoning = item.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        assistant_turn["reasoning_content"] = [{"type": "text", "text": reasoning.strip()}]
    assistant_turn["content"] = [{"type": "text", "text": answer}]

    meta = _metadata_block("conversation", metadata)
    tags = list(meta.get("tags", []))
    if task_type not in tags:
        tags.append(task_type)
    meta["tags"] = tags
    return {
        "version": COSMOS_REASON_VERSION,
        "metadata": meta,
        "conversations": [
            {"role": "user", "content": user_content},
            assistant_turn,
        ],
    }


def _convert_metropolis_scenes_to_tao_vl_reason(
    scenes: list[Path],
    output_dir: Path,
    *,
    media_anchor: Path,
    task_types: Iterable[str] | None,
    copy_media: bool,
    metadata: Mapping[str, Any] | None,
    emit_media_root_as_null: bool,
) -> TrainingConversionResult:
    result = TrainingConversionResult()
    selected = set(task_types) if task_types is not None else None
    items_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for scene in scenes:
        media_formats = _build_media_format_index(scene, result.warnings)
        for task_file, task_type, index, item in _iter_task_items(scene, selected, result.warnings):
            converted = _convert_task_item(
                scene,
                output_dir,
                media_anchor,
                media_formats,
                task_file,
                task_type,
                index,
                item,
                copy_media=copy_media,
                result=result,
            )
            if converted is not None:
                items_by_type[task_type].append(converted)

    if not items_by_type:
        result.errors.append("No supported Metropolis task items were converted")
        return result

    media_root = None if copy_media or emit_media_root_as_null else str(media_anchor.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    for task_type, items in sorted(items_by_type.items()):
        annotation = build_tao_vl_reason_annotation(
            task_type,
            items,
            media_root=media_root,
            metadata=metadata,
        )
        path = write_json(output_dir / f"{task_type}.json", annotation)
        result.paths.append(path)
        result.samples_written += len(items)
    return result


def _convert_metropolis_scenes_to_cosmos_reason(
    scenes: list[Path],
    output_dir: Path,
    *,
    media_anchor: Path,
    task_types: Iterable[str] | None,
    copy_media: bool,
    metadata: Mapping[str, Any] | None,
) -> TrainingConversionResult:
    result = TrainingConversionResult()
    selected = set(task_types) if task_types is not None else None
    samples: list[dict[str, str]] = []
    media_out = output_dir / "media"
    text_out = output_dir / "text"

    for scene in scenes:
        media_formats = _build_media_format_index(scene, result.warnings)
        for task_file, task_type, index, item in _iter_task_items(scene, selected, result.warnings):
            media_id = _media_id(item)
            if media_id is None:
                result.warnings.append(f"{task_file.name}[{index}]: missing video_id/image_id")
                result.samples_skipped += 1
                continue
            is_image = "image_id" in item
            media_rel = _resolve_training_media(
                scene,
                output_dir,
                media_anchor,
                media_id,
                media_formats.get(media_id),
                is_image=is_image,
                copy_media=copy_media,
                media_subdir="media",
                task_file=task_file,
                index=index,
                result=result,
            )
            if media_rel is None:
                result.samples_skipped += 1
                continue

            try:
                conversation = build_cosmos_reason_conversation(
                    task_type,
                    item,
                    is_image=is_image,
                    metadata={"tags": [task_type]},
                )
            except TrainingFormatError as exc:
                result.warnings.append(f"{task_file.name}[{index}]: {exc}")
                result.samples_skipped += 1
                continue

            sample_id = _sample_id(scene.name, task_type, task_file.stem, media_id, index)
            conversation_path = text_out / f"{sample_id}.json"
            result.paths.append(write_json(conversation_path, conversation))
            samples.append(
                {
                    "id": sample_id,
                    "conversation": f"text/{conversation_path.name}",
                    "media": media_rel,
                }
            )
            result.samples_written += 1

    if not samples:
        result.errors.append("No supported Metropolis task items were converted")
        return result

    meta = build_cosmos_reason_meta(samples, metadata=metadata)
    result.paths.append(write_json(output_dir / "meta.json", meta))
    if copy_media:
        media_out.mkdir(parents=True, exist_ok=True)
    text_out.mkdir(parents=True, exist_ok=True)
    return result


def _convert_task_item(
    scene: Path,
    output_dir: Path,
    media_anchor: Path,
    media_formats: Mapping[str, str],
    task_file: Path,
    task_type: str,
    index: int,
    item: dict[str, Any],
    *,
    copy_media: bool,
    result: TrainingConversionResult,
) -> dict[str, Any] | None:
    media_id = _media_id(item)
    if media_id is None:
        result.warnings.append(f"{task_file.name}[{index}]: missing video_id/image_id")
        result.samples_skipped += 1
        return None

    is_image = "image_id" in item
    media_rel = _resolve_training_media(
        scene,
        output_dir,
        media_anchor,
        media_id,
        media_formats.get(media_id),
        is_image=is_image,
        copy_media=copy_media,
        media_subdir="images" if is_image else "videos",
        task_file=task_file,
        index=index,
        result=result,
    )
    if media_rel is None:
        result.samples_skipped += 1
        return None

    answer = _format_answer(task_type, item)
    if answer is None:
        result.warnings.append(f"{task_file.name}[{index}]: could not format answer")
        result.samples_skipped += 1
        return None

    output_item: dict[str, Any] = {
        "image_id" if is_image else "video_id": media_rel,
        "question": _compose_question(task_type, item),
        "answer": answer,
    }
    reasoning = item.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        output_item["reasoning"] = reasoning.strip()
    item_index = item.get("item_index") or item.get("id")
    if item_index is not None:
        output_item["item_index"] = str(item_index)
    return output_item


def _resolve_training_media(
    scene: Path,
    output_dir: Path,
    media_anchor: Path,
    media_id: str,
    media_format: str | None,
    *,
    is_image: bool,
    copy_media: bool,
    media_subdir: str,
    task_file: Path,
    index: int,
    result: TrainingConversionResult,
) -> str | None:
    media_src = _find_media_file(scene, media_id, media_format=media_format)
    if media_src is None:
        result.warnings.append(
            f"{task_file.name}[{index}]: expected media file for {media_id!r} not found under raw/"
        )
        return None

    if copy_media:
        filename = _media_dest_basename(media_src, media_anchor)
        dest_dir = output_dir / media_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        if not dest.exists():
            shutil.copy2(media_src, dest)
        return f"{media_subdir}/{filename}"

    try:
        return media_src.resolve().relative_to(media_anchor.resolve()).as_posix()
    except ValueError:
        return str(media_src.resolve())


def _iter_task_items(
    scene: Path,
    task_filter: set[str] | None,
    warnings: list[str],
) -> Iterable[tuple[Path, str, int, dict[str, Any]]]:
    task_dir = scene / "task"
    if not task_dir.is_dir():
        warnings.append(f"task/ directory not found in {scene}")
        return

    for task_file in sorted(task_dir.glob("*.json")):
        loaded = read_json(task_file)
        if not isinstance(loaded, dict):
            warnings.append(f"Skipping {task_file.name}: invalid JSON object")
            continue
        task_type = _metadata_type(loaded) or _clean_str(loaded.get("task_type"))
        if task_type is None:
            warnings.append(f"Skipping {task_file.name}: missing metadata.type")
            continue
        if task_type not in SUPPORTED_TRAINING_TASKS:
            warnings.append(f"Skipping {task_file.name}: unsupported task type {task_type!r}")
            continue
        if task_filter is not None and task_type not in task_filter:
            continue

        raw_items = loaded.get("items")
        if not isinstance(raw_items, list):
            warnings.append(f"Skipping {task_file.name}: items must be a list")
            continue
        for index, item in enumerate(raw_items):
            if isinstance(item, dict):
                yield task_file, task_type, index, item
            else:
                warnings.append(f"Skipping {task_file.name}[{index}]: item is not an object")


def _compose_question(task_type: str, item: Mapping[str, Any]) -> str:
    question = str(item.get("question", "")).strip()
    if task_type in ("mcq", "mcq_openended"):
        options = item.get("options")
        if isinstance(options, Mapping) and options:
            options_text = "\n".join(f"{key}) {value}" for key, value in sorted(options.items()))
            question = f"{question}\n\n{options_text}" if question else options_text
    instruction = _ANSWER_INSTRUCTIONS.get(task_type)
    if instruction:
        question = f"{question}\n\n{instruction}" if question else instruction
    return question


def _format_answer(task_type: str, item: Mapping[str, Any]) -> str | None:
    if task_type in {
        "open_qa",
        "bcq_openended",
        "mcq_openended",
        "video_summarization",
        "scene_description",
        "temporal_description",
        "causal_linkage",
    }:
        return _clean_str(item.get("answer"))

    if task_type == "bcq":
        answer = _clean_str(item.get("answer"))
        if answer is None:
            return None
        explanation = _clean_str(item.get("explanation"))
        return f"{answer}. {explanation}" if explanation else answer

    if task_type == "mcq":
        letter = _clean_str(item.get("answer"))
        if letter is None:
            return None
        options = item.get("options")
        if isinstance(options, Mapping) and letter in options:
            label = f"{letter}) {options[letter]}"
        else:
            label = letter
        explanation = _clean_str(item.get("explanation"))
        return f"{label}. {explanation}" if explanation else label

    if task_type == "temporal_localization":
        answer = item.get("answer")
        if answer is None:
            return None
        return f"```json\n{json.dumps(answer, indent=2, ensure_ascii=False)}\n```"

    return None


def _metadata_block(type_name: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type_name}
    if metadata is not None:
        for key, value in metadata.items():
            if key != "type" and value is not None:
                out[key] = value
    return out


def _find_metropolis_scenes(root: Path) -> list[Path]:
    if (root / "task").is_dir():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        scene
        for child in sorted(root.iterdir())
        if child.is_dir()
        for scene in _find_metropolis_scenes(child)
    )


def _common_parent(paths: list[Path]) -> Path:
    resolved = [path.resolve() for path in paths]
    if len(resolved) == 1:
        return resolved[0]
    return Path(os.path.commonpath([str(path) for path in resolved]))


def _build_media_format_index(scene: Path, warnings: list[str]) -> dict[str, str]:
    media_formats: dict[str, str] = {}
    contextual_dir = scene / "contextual"
    if not contextual_dir.is_dir():
        warnings.append(f"contextual/ directory not found in {scene}")
        return media_formats

    for contextual_file in sorted(contextual_dir.glob("*.json")):
        loaded = read_json(contextual_file)
        if not isinstance(loaded, dict):
            warnings.append(f"Skipping {contextual_file.name}: invalid JSON object")
            continue
        metadata_type = _metadata_type(loaded)
        if metadata_type not in {"image", "video"}:
            continue
        media_id = _clean_str(loaded.get(f"{metadata_type}_id"))
        media_format = _clean_str(loaded.get("format"))
        if media_id is None or media_format is None:
            warnings.append(
                f"Skipping {contextual_file.name}: {metadata_type}_id and format are required "
                "for training export media resolution"
            )
            continue
        media_formats[media_id] = media_format.lstrip(".")
    return media_formats


def _find_media_file(scene: Path, media_id: str, *, media_format: str | None) -> Path | None:
    safe_media_id = _safe_media_filename_part(media_id)
    if safe_media_id is None:
        return None

    raw_dir = scene / "raw"
    # Preferred layout: the scene media_id already carries its extension (scenes
    # are named "<stem>.<ext>"), so the analyzed media is staged as
    # ``raw/<media_id>`` with a single extension. Match that verbatim first.
    if "." in safe_media_id:
        direct = raw_dir / safe_media_id
        if _is_safe_media_file(direct, scene, raw_dir):
            return direct
    if media_format is not None:
        safe_media_format = _safe_media_format(media_format)
        if safe_media_format is None:
            return None
        candidate = raw_dir / f"{safe_media_id}.{safe_media_format}"
        if _is_safe_media_file(candidate, scene, raw_dir):
            return candidate
    if not raw_dir.is_dir():
        return None
    for candidate in sorted(raw_dir.iterdir()):
        if candidate.name == safe_media_id and _is_safe_media_file(candidate, scene, raw_dir):
            return candidate
        if not candidate.name.startswith(f"{safe_media_id}."):
            continue
        inferred_format = candidate.name.removeprefix(f"{safe_media_id}.")
        if _safe_media_format(inferred_format) is None:
            continue
        if _is_safe_media_file(candidate, scene, raw_dir):
            return candidate
    return None


def _is_safe_media_file(candidate: Path, scene: Path, raw_dir: Path) -> bool:
    try:
        candidate.parent.resolve(strict=False).relative_to(raw_dir.resolve(strict=False))
        candidate.resolve(strict=False).relative_to(scene.resolve(strict=False))
    except ValueError:
        return False
    return candidate.is_file()


def _media_dest_basename(media_src: Path, dataset_root: Path) -> str:
    resolved = media_src.absolute()
    try:
        relative = resolved.relative_to(dataset_root.resolve())
    except ValueError:
        return media_src.name
    return relative.as_posix().replace("/", "--")


def _media_id(item: Mapping[str, Any]) -> str | None:
    value = item.get("image_id") or item.get("video_id")
    return _clean_str(value)


def _metadata_type(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    return _clean_str(metadata.get("type"))


def _sample_id(
    scene_name: str,
    task_type: str,
    task_file_stem: str,
    media_id: str,
    index: int,
) -> str:
    raw = f"{scene_name}__{task_type}__{task_file_stem}__{media_id}__{index:04d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or f"sample_{index:04d}"


def _safe_media_filename_part(value: str) -> str | None:
    if "/" in value or "\\" in value:
        return None
    path = Path(value)
    if path.name != value or path.is_absolute() or ".." in path.parts:
        return None
    return value or None


def _safe_media_format(value: str) -> str | None:
    cleaned = value.lstrip(".")
    if not re.fullmatch(r"[A-Za-z0-9]+", cleaned):
        return None
    return cleaned


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "COSMOS_REASON_VERSION",
    "DEFAULT_TAO_VL_REASON_LICENSE",
    "SUPPORTED_TRAINING_TASKS",
    "TAO_VL_REASON_FORMAT",
    "TrainingConversionResult",
    "TrainingFormatError",
    "build_cosmos_reason_conversation",
    "build_cosmos_reason_meta",
    "build_tao_vl_reason_annotation",
    "convert_metropolis_dataset_to_cosmos_reason",
    "convert_metropolis_dataset_to_tao_vl_reason",
    "convert_metropolis_scene_to_cosmos_reason",
    "convert_metropolis_scene_to_tao_vl_reason",
    "convert_metropolis_scenes_to_cosmos_reason",
    "convert_metropolis_scenes_to_tao_vl_reason",
]
