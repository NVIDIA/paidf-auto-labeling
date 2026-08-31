# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual QA sidecar task."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, override

from core import (
    DataEntry,
    SceneContext,
    ScenePaths,
    ScenePipelineState,
    ensure_scene_skeleton,
    is_image_path,
    read_pipeline_state,
    scene_context_for_entry,
    update_annotation_export_state,
    write_pipeline_state,
)
from core.formats.daft import emit_persona_qa_daft_outputs, emit_visual_qa_daft_outputs
from core.media import discover_image_group
from core.tasks import SequentialTask
from core.utils.multistorage import MSCStorage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from visual_qa.artifacts import VISUAL_QA_ARTIFACTS_KEY, VisualQaArtifactsState
from visual_qa.bank import QuestionBank, aggregate_window_items, load_question_bank, normalize_items
from visual_qa.clients import (
    ChatRequest,
    EndpointClient,
    EndpointProvider,
    MediaPayload,
    ReasoningParser,
    create_endpoint_client,
)
from visual_qa.media import (
    MediaDecodeError,
    MediaWindow,
    VideoInfo,
    extract_window_frames,
    extract_window_video_payload,
    load_crop_payloads,
    load_crop_payloads_from_files,
    plan_windows,
    probe_video,
    read_image_payload,
)
from visual_qa.parsing import (
    extract_json_object,
    extract_visual_qa_items,
    extract_window_description,
)
from visual_qa.question_driven import (
    generate_question_driven_prompts,
    write_question_driven_prompts,
)
from visual_qa.sidecar import (
    extract_window_item_groups,
    find_first_existing_sidecar,
    read_json_object,
    resolve_sidecar,
    write_json_object,
)

VisualQaGenerationMode = Literal[
    "normalize-only",
    "window-direct-vlm",
    "window-vlm-llm",
    "question-driven-vlm-llm",
    "metadata-llm",
]
VisualQaInputSource = Literal["auto", "original", "enhanced", "tracking"]
VisualQaMediaMode = Literal["auto", "video", "frames"]

STANDARD_VISUAL_QA_PROMPT = """Answer the authoritative question bank from the
provided visual evidence. Return exactly one fenced JSON object with this shape
and no prose outside the JSON fence:

```json
{
  "items": [
    {
      "id": "<copy the question id>",
      "question": "<copy the question text>",
      "options": ["<copy options exactly, or [] for open-ended>"],
      "answer": "<one exact option string, or concise free-form answer>"
    }
  ]
}
```

Omit questions whose `include_if` condition is not satisfied by your own
answers. Do not invent questions, options, objects, actions, or events.
{reasoning_rule}

QUESTION BANK:
```json
{question_bank_json}
```
"""

STANDARD_VISUAL_EVIDENCE_PROMPT = """Describe the visual evidence needed to
answer a downstream question bank. Be concrete about visible objects, actions,
scene state, counts when obvious, and uncertainty. Do not answer questions yet.
"""

DETECTION_AND_TRACKING_ARTIFACTS_KEY = "detection_and_tracking"
VISUAL_QA_DAFT_EMITTERS = ("mcq", "bcq", "open_qa")


class VisualQaConfig(BaseModel):
    """Configuration for visual QA generation and sidecar normalization."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    generation_mode: VisualQaGenerationMode = "normalize-only"
    input_source: VisualQaInputSource = "auto"
    image_group_dir: str | None = None
    max_group_images: int = Field(default=0, ge=0)

    question_bank_file: str | None = None

    # Per-track (PAS video flow) generation. When set, each tracked person's
    # crops become one QA "window": the task reads this detection/tracking
    # tracks sidecar, runs the question bank once per track over that track's
    # crops, and stamps ``track_id`` into each window. Leave ``None`` for the
    # standard per-scene (video time-window / single-image) behavior.
    track_crops_sidecar: str | None = None
    # Even-downsample cap for crops sent per track. ``0`` sends every crop (see
    # ``media._sample_even``), which the image-augmentation flow uses to preserve
    # all pre-extracted augmented views per person.
    max_crops_per_track: int = Field(default=8, ge=0)

    input_sidecars: tuple[str, ...] = (
        "visual_qa/windows.json",
        "metadata.json",
    )
    metadata_input_sidecars: tuple[str, ...] = (
        "captioning/metadata_chunk.json",
        "captioning/image_caption.json",
        "metadata.json",
    )
    raw_windows_sidecar: str = "visual_qa/windows.json"
    output_items_sidecar: str = "visual_qa/items.json"
    output_windows_sidecar: str = "visual_qa/windows.normalized.json"

    # Key under which this run records its slice in
    # ``pipeline_state.task_artifacts``. Defaults to the canonical
    # ``"visual_qa"``. Set a distinct value per pass (e.g. ``visual_qa_anomaly``,
    # ``visual_qa_anomaly_search``) when several visual_qa passes share one
    # output directory, so each pass's provenance survives instead of the last
    # run clobbering the shared key. The canonical ``"visual_qa"`` key is always
    # also written as the most-recent ("latest") pointer for consumers that look
    # it up by name.
    state_artifacts_key: str | None = None

    vlm_provider: EndpointProvider = "openai-compatible"
    vlm_endpoint_url: str | None = None
    vlm_model: str = "default"

    llm_provider: EndpointProvider | None = None
    llm_endpoint_url: str | None = None
    llm_model: str | None = None

    parser: ReasoningParser = "instruct"

    evidence_prompt_text: str | None = None
    evidence_prompt_file: str | None = None
    prompt_text: str | None = None
    prompt_file: str | None = None
    system_prompt: str | None = None
    include_reasoning: bool = False

    window_seconds: float = Field(default=10.0, gt=0.0)
    window_frames: int = Field(default=256, ge=0)
    remainder_threshold: int = Field(default=128, ge=0)
    single_window: bool = False
    sampling_fps: float = Field(default=2.0, gt=0.0)
    max_frames: int = Field(default=8, ge=1)
    resolution: int = Field(default=768, ge=64)
    media_mode: VisualQaMediaMode = "auto"

    max_tokens: int = Field(default=1024, ge=1)
    question_prompt_max_tokens: int = Field(default=8192, ge=1)
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    timeout_s: float = Field(default=120.0, gt=0.0)
    retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=1.0, ge=0.0)

    aggregate_windows: bool = True
    strict_answers: bool = True
    open_ended: bool = False
    write_empty_marker: bool = True
    # Emit the flat DAFT task files (task/mcq.json, task/bcq.json, task/open_qa.json).
    # Defaults on for backward compatibility. Turn off in passes whose flat QA is
    # redundant (e.g. PAS per-track, where persona_qa.json supersedes it) or owned by
    # a dedicated later pass. persona_qa.json emission is independent of this flag.
    emit_flat_qa_tasks: bool = True

    @model_validator(mode="after")
    def _validate_combinations(self) -> Self:
        for text_field, file_field in (
            ("prompt_text", "prompt_file"),
            ("evidence_prompt_text", "evidence_prompt_file"),
        ):
            if getattr(self, text_field) is not None and getattr(self, file_field) is not None:
                raise ValueError(f"{text_field} and {file_field} cannot both be set")
        if self.generation_mode != "normalize-only" and self.question_bank_file is None:
            raise ValueError(f"generation_mode={self.generation_mode} requires question_bank_file")
        if self.track_crops_sidecar is not None and self.generation_mode not in {
            "window-direct-vlm",
            "window-vlm-llm",
            "question-driven-vlm-llm",
        }:
            raise ValueError(
                "track_crops_sidecar requires generation_mode of 'window-direct-vlm', "
                f"'window-vlm-llm', or 'question-driven-vlm-llm', got {self.generation_mode!r}"
            )
        if self.image_group_dir is not None and self.generation_mode not in {
            "window-direct-vlm",
            "window-vlm-llm",
            "question-driven-vlm-llm",
        }:
            raise ValueError(
                "image_group_dir requires generation_mode of 'window-direct-vlm', "
                f"'window-vlm-llm', or 'question-driven-vlm-llm', got {self.generation_mode!r}"
            )
        if self.image_group_dir is not None and self.track_crops_sidecar is not None:
            raise ValueError("image_group_dir and track_crops_sidecar cannot both be set")
        return self


class VisualQaTask(SequentialTask):
    """Generate, normalize, and export media-grounded QA artifacts."""

    def __init__(
        self,
        config: VisualQaConfig | None = None,
        *,
        name: str | None = None,
        max_retries: int = 0,
    ) -> None:
        super().__init__(name=name or "visual_qa", max_retries=max_retries)
        self.config = config or VisualQaConfig()
        self.storage = MSCStorage(self.logger)
        self._resolved_question_bank_file: Path | None = None
        self._question_driven_evidence_prompt: str | None = None
        self.vlm_client = _build_vlm_client(self.config) if _uses_vlm(self.config) else None
        self.llm_client = _build_llm_client(self.config) if _uses_llm(self.config) else None

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        if not self.config.enabled:
            self.logger.info("Visual QA disabled; skipping.")
            return data_entry

        paths = ensure_scene_skeleton(data_entry.data_path)
        generated_sidecar = self._generate_raw_sidecar_if_configured(data_entry, paths)
        input_sidecar = generated_sidecar or find_first_existing_sidecar(
            paths.sidecars_dir,
            self.config.input_sidecars,
        )
        items_sidecar = resolve_sidecar(paths.sidecars_dir, self.config.output_items_sidecar)
        windows_sidecar = resolve_sidecar(paths.sidecars_dir, self.config.output_windows_sidecar)

        if input_sidecar is None:
            self.logger.info(
                "No visual QA input sidecar found for %s; checked %s",
                data_entry.data_path,
                ", ".join(self.config.input_sidecars),
            )
            self._record_state(
                data_entry,
                success=False,
                items_json=None,
                windows_json=None,
                source_sidecar=None,
            )
            _record_daft_outputs(data_entry, ())
            return data_entry

        bank = self._load_bank()
        payload = read_json_object(input_sidecar)
        raw_groups, raw_windows = extract_window_item_groups(payload)

        normalized_groups: list[list[dict[str, object]]] = []
        all_errors: list[str] = []
        normalized_windows: list[dict[str, object]] = []
        for idx, group in enumerate(raw_groups):
            items, errors = normalize_items(
                group,
                bank=bank,
                strict_answers=self.config.strict_answers,
                open_ended=self.config.open_ended,
            )
            all_errors.extend(f"window[{idx}]: {error}" for error in errors)
            normalized_groups.append(items)

        for idx, window in enumerate(raw_windows):
            normalized = dict(window)
            normalized["items"] = normalized_groups[idx] if idx < len(normalized_groups) else []
            normalized_windows.append(normalized)

        ctx = scene_context_for_entry(data_entry)
        item_payload = aggregate_window_items(
            normalized_groups,
            bank=bank,
            media_id=ctx.media_id,
            aggregate=self.config.aggregate_windows,
        )
        item_payload["source_sidecar"] = str(input_sidecar)
        item_payload["question_bank"] = self._bank_metadata(bank)
        if all_errors:
            item_payload["warnings"] = all_errors

        window_payload: dict[str, object] = {
            "schema_version": "1",
            "media_id": ctx.media_id,
            "source_sidecar": str(input_sidecar),
            "windows": normalized_windows,
        }

        write_items = bool(item_payload["items"]) or self.config.write_empty_marker
        if write_items:
            write_json_object(items_sidecar, item_payload)
        write_json_object(windows_sidecar, window_payload)

        visual_qa_artifacts = self._record_state(
            data_entry,
            success=True,
            items_json=items_sidecar if write_items else None,
            windows_json=windows_sidecar,
            source_sidecar=input_sidecar,
        )
        self._emit_daft_outputs(
            data_entry,
            scene_ctx=ctx,
            visual_qa_artifacts=visual_qa_artifacts,
            success=True,
        )
        self.logger.info(
            "Visual QA wrote %d item(s) from %s to %s",
            len(item_payload["items"]) if isinstance(item_payload["items"], list) else 0,
            input_sidecar,
            items_sidecar,
        )
        return data_entry

    def _generate_raw_sidecar_if_configured(
        self,
        data_entry: DataEntry,
        paths: ScenePaths,
    ) -> Path | None:
        if self.config.generation_mode == "normalize-only":
            return None

        bank = self._load_bank()
        if self.config.generation_mode == "question-driven-vlm-llm":
            question_bank_path = self._question_bank_path()
            if question_bank_path is None:
                raise ValueError("question-driven-vlm-llm generation requires question_bank_file")
            llm_client = _require_client(self.llm_client, self.config.generation_mode)
            prompts = generate_question_driven_prompts(
                bank_payload=_load_question_bank_payload(question_bank_path),
                llm_client=llm_client,
                max_tokens=self.config.question_prompt_max_tokens,
            )
            self._question_driven_evidence_prompt = prompts.evidence_prompt
            qa_prompt = prompts.mapper_prompt
            write_question_driven_prompts(
                paths.sidecars_dir / "visual_qa" / "prompts",
                prompts,
            )
        else:
            qa_prompt = _load_visual_qa_prompt(
                self.config,
                question_bank_path=self._question_bank_path(),
            )
        original_media = Path(data_entry.media_path)
        scene_ctx = scene_context_for_entry(data_entry)
        output_sidecar = resolve_sidecar(paths.sidecars_dir, self.config.raw_windows_sidecar)

        extra_payload: dict[str, Any] = {}
        if self.config.track_crops_sidecar is not None:
            tracks_sidecar = resolve_sidecar(paths.sidecars_dir, self.config.track_crops_sidecar)
            payload_source_key = "source_sidecar"
            payload_source_value = str(tracks_sidecar)
            if not tracks_sidecar.is_file():
                # A missing track-crops sidecar means detection found/kept no
                # tracks for this scene (a genuinely person-less clip, or all
                # tracks pruned by a crop gate). Treat it as "no people here":
                # emit empty windows and continue, rather than aborting the whole
                # run. Empty windows flow through normalization as a zero-result
                # scene, identical to a present-but-empty tracks sidecar.
                self.logger.warning(
                    "track_crops_sidecar not found (%s); treating scene as having "
                    "no tracks and writing empty visual_qa windows.",
                    tracks_sidecar,
                )
                windows = []
            else:
                tracks_payload = read_json_object(tracks_sidecar)
                windows = self._generate_from_track_crops(
                    paths=paths, tracks_payload=tracks_payload, qa_prompt=qa_prompt
                )
                crop_root = tracks_payload.get("crop_root")
                if isinstance(crop_root, str):
                    extra_payload["crop_root"] = crop_root
        elif self.config.generation_mode == "metadata-llm":
            source_sidecar = find_first_existing_sidecar(
                paths.sidecars_dir,
                self.config.metadata_input_sidecars,
            )
            if source_sidecar is None:
                raise ValueError(
                    "generation_mode=metadata-llm requires one of "
                    f"{', '.join(self.config.metadata_input_sidecars)}"
                )
            windows = self._generate_from_metadata_sidecar(
                source_sidecar=source_sidecar, qa_prompt=qa_prompt
            )
            payload_source_key = "source_sidecar"
            payload_source_value = str(source_sidecar)
        else:
            state = read_pipeline_state(data_entry.data_path)
            media_path = _select_visual_qa_input(
                original_media_path=original_media,
                pipeline_state=state,
                input_source=self.config.input_source,
            )
            if self.config.image_group_dir is not None:
                if not is_image_path(media_path):
                    raise ValueError("image_group_dir requires an image DataEntry.media_path")
                image_paths = discover_image_group(
                    self.config.image_group_dir,
                    max_images=self.config.max_group_images,
                )
                windows = self._generate_from_image_group(
                    image_paths=image_paths,
                    qa_prompt=qa_prompt,
                )
                payload_source_key = "source_media"
                payload_source_value = str(media_path)
                extra_payload["source_image_group"] = str(
                    Path(self.config.image_group_dir).expanduser()
                )
            else:
                windows = self._generate_from_media(media_path=media_path, qa_prompt=qa_prompt)
                payload_source_key = "source_media"
                payload_source_value = str(media_path)

        payload: dict[str, Any] = {
            "schema_version": "1",
            "artifact_type": "visual_qa_windows",
            "stage": _stage_metadata(),
            "media_id": scene_ctx.media_id,
            "generation_mode": self.config.generation_mode,
            "question_bank": self._bank_metadata(bank),
            payload_source_key: payload_source_value,
            **extra_payload,
            "windows": windows,
        }
        write_json_object(output_sidecar, payload)
        return output_sidecar

    def _generate_from_track_crops(
        self,
        *,
        paths: ScenePaths,
        tracks_payload: dict[str, Any],
        qa_prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Generate one QA window per tracked person from that track's crops.

        Each track in the detection/tracking ``tracks.json`` becomes a single
        window whose media is the track's (evenly sampled) crops. The track's
        ``track_id`` and the number of crops actually sent are stamped into the
        window so the downstream assembler can key results by identity.

        Args:
            paths: Resolved scene paths (locates ``sidecars/`` and crop dirs).
            tracks_payload: Parsed tracks sidecar with a ``tracks`` list.
            qa_prompt: The rendered question-bank prompt.

        Returns:
            One window record per track that yielded crops.
        """
        raw_tracks = tracks_payload.get("tracks")
        if not isinstance(raw_tracks, list):
            return []
        windows: list[dict[str, Any]] = []
        for idx, record in enumerate(raw_tracks):
            if not isinstance(record, dict) or record.get("track_id") is None:
                continue
            try:
                crop_dir = resolve_sidecar(paths.sidecars_dir, str(record.get("crop_dir") or ""))
            except ValueError:
                self.logger.warning(
                    "Track record has an out-of-scene crop_dir %r; skipping.",
                    record.get("crop_dir"),
                )
                continue
            track_id = int(record["track_id"])
            # Prefer the crop files the producer recorded for this track over
            # rescanning the directory, so stale crops from an earlier run into a
            # dirty directory cannot leak in. Fall back to a directory scan only
            # for older sidecars that did not record an explicit crop list.
            recorded_crops = record.get("crops")
            if isinstance(recorded_crops, list) and recorded_crops:
                crop_files: list[Path] = []
                escaped = False
                for rel in recorded_crops:
                    try:
                        crop_files.append(resolve_sidecar(paths.sidecars_dir, str(rel)))
                    except ValueError:
                        escaped = True
                        break
                if escaped:
                    self.logger.warning(
                        "Track %d has an out-of-scene crop path; skipping.", track_id
                    )
                    continue
                crops = load_crop_payloads_from_files(
                    crop_files, max_crops=self.config.max_crops_per_track
                )
            else:
                crops = load_crop_payloads(crop_dir, max_crops=self.config.max_crops_per_track)
            if not crops:
                self.logger.warning("Track %d has no crops under %s; skipping.", track_id, crop_dir)
                continue
            generated = self._generate_for_window(
                media=tuple(crops),
                window_context=(
                    f"These {len(crops)} cropped images all show the same person "
                    f"(track {track_id}) across a video chunk. Describe that one person."
                ),
                description="",
                parsed=None,
                qa_prompt=qa_prompt,
            )
            windows.append(
                {
                    "window_index": idx,
                    "track_id": track_id,
                    "n_crops_in_chunk": len(crops),
                    "description": generated.description,
                    **generated.payload,
                }
            )
        return windows

    def _generate_from_image_group(
        self,
        *,
        image_paths: tuple[Path, ...],
        qa_prompt: str,
    ) -> list[dict[str, Any]]:
        """Generate one QA window from multiple views of a single identity."""
        media = tuple(read_image_payload(path) for path in image_paths)
        generated = self._generate_for_window(
            media=media,
            window_context=(
                f"These {len(image_paths)} images are different views of the same identity. "
                "Answer once for that identity using evidence across all supplied views."
            ),
            description="",
            parsed=None,
            qa_prompt=qa_prompt,
        )
        return [
            {
                "window_index": 0,
                "source_images": [str(path) for path in image_paths],
                "num_views": len(image_paths),
                "description": generated.description,
                **generated.payload,
            }
        ]

    def _generate_from_media(self, *, media_path: Path, qa_prompt: str) -> list[dict[str, Any]]:
        if is_image_path(media_path):
            image_payload = read_image_payload(media_path)
            generated = self._generate_for_window(
                media=(image_payload,),
                window_context="Image input.",
                description="",
                parsed=None,
                qa_prompt=qa_prompt,
            )
            return [
                {
                    "window_index": 0,
                    "description": generated.description,
                    **generated.payload,
                }
            ]

        info = probe_video(media_path)
        windows = plan_windows(info, self.config)
        generated_windows: list[dict[str, Any]] = []
        for window in windows:
            media_payloads, input_mode, media_error = self._window_media_payloads(
                media_path=media_path,
                window=window,
                info=info,
            )
            window_context = (
                f"Window start: {window.start_s:.3f}s.\n"
                f"Window end: {window.end_s:.3f}s.\n"
                "Media timestamps are relative to this window."
            )
            try:
                generated = self._generate_for_window(
                    media=tuple(media_payloads),
                    window_context=window_context,
                    description="",
                    parsed=None,
                    qa_prompt=qa_prompt,
                )
            except Exception as exc:
                if self.config.media_mode != "auto" or input_mode != "video":
                    raise
                media_payloads = extract_window_frames(
                    media_path=media_path,
                    window=window,
                    source_fps=info.fps,
                    sampling_fps=self.config.sampling_fps,
                    max_frames=self.config.max_frames,
                    resolution=self.config.resolution,
                )
                input_mode = "frames"
                media_error = _append_failure_reason(
                    media_error,
                    f"video_model_call_failed:{exc.__class__.__name__}",
                )
                generated = self._generate_for_window(
                    media=tuple(media_payloads),
                    window_context=window_context,
                    description="",
                    parsed=None,
                    qa_prompt=qa_prompt,
                )
            window_record: dict[str, Any] = {
                "window_index": window.index,
                "start_s": round(window.start_s, 3),
                "end_s": round(window.end_s, 3),
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "input_media_mode": input_mode,
                "description": generated.description,
                **generated.payload,
            }
            if media_error:
                window_record["media_failure_reason"] = media_error
            generated_windows.append(window_record)
        return generated_windows

    def _generate_from_metadata_sidecar(
        self,
        *,
        source_sidecar: Path,
        qa_prompt: str,
    ) -> list[dict[str, Any]]:
        payload = read_json_object(source_sidecar)
        source_windows = _metadata_windows(payload)
        generated_windows: list[dict[str, Any]] = []
        for idx, window in enumerate(source_windows):
            description = str(
                window.get("description")
                or window.get("vlm_caption")
                or window.get("caption")
                or ""
            )
            generated = self._generate_for_window(
                media=(),
                window_context=_metadata_window_context(window, index=idx),
                description=description,
                parsed=None,
                qa_prompt=qa_prompt,
            )
            window_record = dict(window)
            window_record["window_index"] = int(window_record.get("window_index", idx))
            window_record["description"] = generated.description or description
            window_record.update(generated.payload)
            generated_windows.append(window_record)
        return generated_windows

    def _generate_for_window(
        self,
        *,
        media: tuple[MediaPayload, ...],
        window_context: str,
        description: str,
        parsed: dict[str, Any] | None,
        qa_prompt: str,
    ) -> _GeneratedQaWindow:
        mode = self.config.generation_mode
        if mode == "window-direct-vlm":
            vlm_client = _require_client(self.vlm_client, mode)
            raw = vlm_client.generate(
                ChatRequest(
                    prompt=f"{qa_prompt}\n\n{window_context}",
                    media=media,
                    system_prompt=self.config.system_prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            )
            parsed_qa = extract_json_object(raw)
            call_metadata = _client_call_metadata(vlm_client)
            return _GeneratedQaWindow(
                description=description,
                payload=_visual_qa_payload(
                    mode=mode,
                    raw_response=raw,
                    parsed=parsed_qa,
                    call_metadata=call_metadata,
                ),
            )

        if mode in {"window-vlm-llm", "question-driven-vlm-llm"}:
            vlm_client = _require_client(self.vlm_client, mode)
            evidence_prompt = (
                self._question_driven_evidence_prompt
                if mode == "question-driven-vlm-llm"
                else _load_prompt(
                    prompt_text=self.config.evidence_prompt_text,
                    prompt_file=self.config.evidence_prompt_file,
                    default_prompt=STANDARD_VISUAL_EVIDENCE_PROMPT,
                )
            )
            if not evidence_prompt:
                raise RuntimeError("question-driven evidence prompt was not generated")
            evidence_raw = vlm_client.generate(
                ChatRequest(
                    prompt=f"{evidence_prompt}\n\n{window_context}",
                    media=media,
                    system_prompt=self.config.system_prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            )
            evidence_parsed = extract_json_object(evidence_raw)
            evidence_description = (
                extract_window_description(evidence_raw, evidence_parsed) or evidence_raw.strip()
            )
            evidence_call = _client_call_metadata(vlm_client)
            if mode == "question-driven-vlm-llm":
                llm_client = _require_client(self.llm_client, mode)
                answer_raw = llm_client.generate(
                    ChatRequest(
                        prompt=evidence_description,
                        system_prompt=qa_prompt,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                )
                llm_payload = _visual_qa_payload(
                    mode=mode,
                    raw_response=answer_raw,
                    parsed=extract_json_object(answer_raw),
                    call_metadata=_client_call_metadata(llm_client),
                )
            else:
                llm_payload = self._generate_llm_items(
                    prompt=qa_prompt,
                    window_context=window_context,
                    description=evidence_description,
                    parsed=evidence_parsed,
                )
            llm_payload["visual_evidence"] = evidence_description
            llm_payload["visual_evidence_raw_response"] = evidence_raw
            llm_payload["visual_evidence_parsed"] = evidence_parsed
            llm_payload["visual_evidence_call"] = evidence_call
            llm_payload["visual_evidence_token_counts"] = _token_counts(evidence_call)
            return _GeneratedQaWindow(description=evidence_description, payload=llm_payload)

        if mode == "metadata-llm":
            llm_payload = self._generate_llm_items(
                prompt=qa_prompt,
                window_context=window_context,
                description=description,
                parsed=parsed,
            )
            return _GeneratedQaWindow(description=description, payload=llm_payload)

        raise ValueError(f"Unsupported visual QA generation mode: {mode}")

    def _generate_llm_items(
        self,
        *,
        prompt: str,
        window_context: str,
        description: str,
        parsed: dict[str, Any] | None,
    ) -> dict[str, Any]:
        mode = self.config.generation_mode
        llm_client = _require_client(self.llm_client, mode)
        raw = llm_client.generate(
            ChatRequest(
                prompt=prompt
                + "\n\n"
                + _visual_qa_text_input(
                    window_context=window_context,
                    description=description,
                    parsed=parsed,
                ),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        )
        parsed_qa = extract_json_object(raw)
        call_metadata = _client_call_metadata(llm_client)
        return _visual_qa_payload(
            mode=mode,
            raw_response=raw,
            parsed=parsed_qa,
            call_metadata=call_metadata,
        )

    def _window_media_payloads(
        self,
        *,
        media_path: Path,
        window: MediaWindow,
        info: VideoInfo,
    ) -> tuple[list[MediaPayload], str, str | None]:
        if self.config.media_mode in {"auto", "video"}:
            try:
                video_payload = extract_window_video_payload(
                    media_path=media_path,
                    window=window,
                    total_frames=info.frame_count,
                )
                return [video_payload], "video", None
            except MediaDecodeError as exc:
                if self.config.media_mode == "video":
                    raise
                media_error = f"video_payload_failed:{exc.__class__.__name__}"
        else:
            media_error = None

        frames = extract_window_frames(
            media_path=media_path,
            window=window,
            source_fps=info.fps,
            sampling_fps=self.config.sampling_fps,
            max_frames=self.config.max_frames,
            resolution=self.config.resolution,
        )
        return frames, "frames", media_error

    def _load_bank(self) -> QuestionBank:
        question_bank_path = self._question_bank_path()
        if question_bank_path is None:
            return load_question_bank(None)
        return load_question_bank(question_bank_path)

    def _question_bank_path(self) -> Path | None:
        question_bank_file = self.config.question_bank_file
        if question_bank_file is None:
            return None
        trimmed_question_bank_file = question_bank_file.strip()
        if not trimmed_question_bank_file:
            return None
        if self._resolved_question_bank_file is not None:
            return self._resolved_question_bank_file

        if self.storage.is_remote_storage_url(trimmed_question_bank_file):
            self._resolved_question_bank_file = Path(
                self.storage.download_if_remote(trimmed_question_bank_file, is_file=True)
            )
        else:
            self._resolved_question_bank_file = Path(trimmed_question_bank_file).expanduser()
        return self._resolved_question_bank_file

    def _bank_metadata(self, bank: QuestionBank) -> dict[str, object]:
        bank_path = self.config.question_bank_file
        if bank_path is None and bank.source_path is not None:
            bank_path = str(bank.source_path)
        return {
            "path": bank_path,
            "num_questions": len(bank.questions),
        }

    def _record_state(
        self,
        data_entry: DataEntry,
        *,
        success: bool,
        items_json: Path | None,
        windows_json: Path | None,
        source_sidecar: Path | None,
    ) -> dict[str, object]:
        state = read_pipeline_state(data_entry.data_path)
        state.data_entry_id = state.data_entry_id or data_entry.id
        state.media_path = state.media_path or data_entry.media_path
        visual_qa_artifacts = VisualQaArtifactsState(
            success=success,
            items_json=str(items_json) if items_json is not None else None,
            windows_json=str(windows_json) if windows_json is not None else None,
            source_sidecar=str(source_sidecar) if source_sidecar is not None else None,
        ).model_dump()
        key = self.config.state_artifacts_key or VISUAL_QA_ARTIFACTS_KEY
        state.task_artifacts[key] = visual_qa_artifacts
        if key != VISUAL_QA_ARTIFACTS_KEY:
            # Preserve per-pass provenance under the namespaced key while keeping
            # the canonical key pointing at the most-recent run so name-based
            # consumers keep working across a consolidated multi-pass directory.
            state.task_artifacts[VISUAL_QA_ARTIFACTS_KEY] = {
                **visual_qa_artifacts,
                "variant": key,
            }
        write_pipeline_state(data_entry.data_path, state)
        return visual_qa_artifacts

    def _emit_daft_outputs(
        self,
        data_entry: DataEntry,
        *,
        scene_ctx: SceneContext,
        visual_qa_artifacts: dict[str, object],
        success: bool,
    ) -> None:
        written: tuple[Path, ...] = ()
        if success:
            if self.config.emit_flat_qa_tasks:
                written = emit_visual_qa_daft_outputs(
                    data_entry.data_path,
                    scene_ctx,
                    visual_qa_artifacts=visual_qa_artifacts,
                    logger=self.logger,
                )
            else:
                self.logger.info(
                    "Visual QA flat DAFT task emission disabled; skipping mcq/bcq/open_qa."
                )
            persona_qa = emit_persona_qa_daft_outputs(
                data_entry.data_path,
                scene_ctx,
                visual_qa_artifacts=visual_qa_artifacts,
                logger=self.logger,
            )
            if persona_qa is not None:
                written = (*written, persona_qa)
            if written:
                self.logger.info("Visual QA wrote %d DAFT pivot file(s).", len(written))
        _record_daft_outputs(data_entry, written)


class _GeneratedQaWindow(BaseModel):
    description: str
    payload: dict[str, Any]


def _build_vlm_client(config: VisualQaConfig) -> EndpointClient:
    return create_endpoint_client(
        provider=config.vlm_provider,
        endpoint_url=config.vlm_endpoint_url,
        model=config.vlm_model,
        timeout_s=config.timeout_s,
        retries=config.retries,
        retry_backoff_s=config.retry_backoff_s,
        parser=config.parser,
    )


def _build_llm_client(config: VisualQaConfig) -> EndpointClient:
    provider = config.llm_provider or config.vlm_provider
    return create_endpoint_client(
        provider=provider,
        endpoint_url=config.llm_endpoint_url or config.vlm_endpoint_url,
        model=config.llm_model or config.vlm_model,
        timeout_s=config.timeout_s,
        retries=config.retries,
        retry_backoff_s=config.retry_backoff_s,
        parser=config.parser,
    )


def _uses_vlm(config: VisualQaConfig) -> bool:
    return config.generation_mode in {
        "window-direct-vlm",
        "window-vlm-llm",
        "question-driven-vlm-llm",
    }


def _uses_llm(config: VisualQaConfig) -> bool:
    return config.generation_mode in {
        "window-vlm-llm",
        "question-driven-vlm-llm",
        "metadata-llm",
    }


def _require_client(
    client: EndpointClient | None,
    mode: str,
) -> EndpointClient:
    if client is None:
        raise RuntimeError(f"generation_mode={mode} requires an endpoint client")
    return client


def _load_prompt(
    *,
    prompt_text: str | None,
    prompt_file: str | None,
    default_prompt: str,
) -> str:
    if prompt_text and prompt_text.strip():
        return prompt_text.strip()
    if prompt_file and prompt_file.strip():
        contents = Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
        if contents:
            return contents
    return default_prompt.strip()


def _load_visual_qa_prompt(
    config: VisualQaConfig,
    *,
    question_bank_path: Path | None = None,
) -> str:
    prompt = _load_prompt(
        prompt_text=config.prompt_text,
        prompt_file=config.prompt_file,
        default_prompt=STANDARD_VISUAL_QA_PROMPT,
    )
    if "{question_bank_json}" not in prompt and "{reasoning_rule}" not in prompt:
        return prompt
    if question_bank_path is None:
        raise ValueError("visual QA generation requires question_bank_file")
    bank_payload = _load_question_bank_payload(question_bank_path)
    reasoning_rule = (
        "Include a short `reasoning_trace` for each answer."
        if config.include_reasoning
        else "Do not include reasoning traces."
    )
    return (
        prompt.replace("{question_bank_json}", json.dumps(bank_payload, indent=2))
        .replace("{reasoning_rule}", reasoning_rule)
        .strip()
    )


def _load_question_bank_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"visual QA question bank must be a JSON object: {path}")
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"visual QA question bank missing 'questions' list: {path}")
    return {"questions": questions}


def _visual_qa_payload(
    *,
    mode: str,
    raw_response: str,
    parsed: dict[str, Any] | None,
    call_metadata: dict[str, Any],
) -> dict[str, Any]:
    items = extract_visual_qa_items(parsed)
    return {
        "items": items,
        "visual_qa_generation_mode": mode,
        "visual_qa_raw_response": raw_response,
        "visual_qa_parsed": parsed,
        "visual_qa_items": items,
        "visual_qa_token_counts": _token_counts(call_metadata),
        "visual_qa_call": call_metadata,
    }


def _visual_qa_text_input(
    *,
    window_context: str,
    description: str,
    parsed: dict[str, Any] | None,
) -> str:
    metadata: dict[str, Any] = {
        "window_context": window_context,
        "description": description,
    }
    if parsed is not None:
        metadata["parsed_evidence"] = parsed
    return "WINDOW EVIDENCE:\n" + json.dumps(metadata, indent=2)


def _metadata_windows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    windows = payload.get("windows")
    if isinstance(windows, list):
        return [dict(window) for window in windows if isinstance(window, dict)]
    for key in ("caption", "summary", "description", "vlm_caption"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return [{"window_index": 0, "description": value.strip()}]
    return []


def _metadata_window_context(window: dict[str, Any], *, index: int) -> str:
    metadata = {
        key: value
        for key, value in window.items()
        if key
        in {
            "window_index",
            "start_s",
            "end_s",
            "start_frame",
            "end_frame",
            "caption",
            "description",
            "summary",
            "vlm_caption",
        }
    }
    metadata.setdefault("window_index", index)
    return "WINDOW METADATA:\n" + json.dumps(metadata, indent=2)


def _client_call_metadata(client: EndpointClient) -> dict[str, Any]:
    metadata = getattr(client, "last_call_metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _token_counts(call_metadata: dict[str, Any] | None) -> dict[str, int] | None:
    usage = call_metadata.get("usage") if call_metadata else None
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _first_int(usage, "prompt_tokens", "promptTokenCount", "input_tokens")
    output_tokens = _first_int(
        usage,
        "completion_tokens",
        "output_tokens",
        "candidatesTokenCount",
    )
    total_tokens = _first_int(usage, "total_tokens", "totalTokenCount")
    out: dict[str, int] = {}
    if prompt_tokens is not None:
        out["prompt_tokens"] = prompt_tokens
    if output_tokens is not None:
        out["output_tokens"] = output_tokens
    if total_tokens is not None:
        out["total_tokens"] = total_tokens
    return out or None


def _first_int(values: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _stage_metadata() -> dict[str, str]:
    return {
        "name": "visual_qa",
        "implementation": "VisualQaTask",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _append_failure_reason(existing: str | None, new_reason: str) -> str:
    if existing:
        return f"{existing};{new_reason}"
    return new_reason


def _select_visual_qa_input(
    *,
    original_media_path: Path,
    pipeline_state: ScenePipelineState,
    input_source: str,
) -> Path:
    enhanced_media = getattr(pipeline_state, "enhanced_media", None)
    task_artifacts = getattr(pipeline_state, "task_artifacts", {})
    tracking_artifacts = task_artifacts.get(DETECTION_AND_TRACKING_ARTIFACTS_KEY, {})
    tracking_path = _first_existing_path(
        tracking_artifacts.get("annotated_video_path"),
        tracking_artifacts.get("red_id_overlay_path"),
    )
    enhanced_path = _first_existing_path(
        enhanced_media.output_path if enhanced_media is not None else None
    )
    if input_source == "tracking":
        if tracking_path is None:
            raise ValueError("requested tracking visual QA input source is missing")
        return tracking_path
    if input_source == "enhanced":
        if enhanced_path is None:
            raise ValueError("requested enhanced visual QA input source is missing")
        return enhanced_path
    if input_source == "original":
        return original_media_path
    return tracking_path or enhanced_path or original_media_path


def _first_existing_path(*values: object) -> Path | None:
    for value in values:
        if value is None:
            continue
        path = Path(str(value))
        if path.exists():
            return path
    return None


def _record_daft_outputs(data_entry: DataEntry, written: tuple[Path, ...]) -> None:
    written_by_name = {path.stem: path for path in written}
    update_annotation_export_state(
        data_entry.data_path,
        data_entry_id=data_entry.id,
        media_path=data_entry.media_path,
        emitter_artifacts={name: written_by_name.get(name) for name in VISUAL_QA_DAFT_EMITTERS},
    )


__all__ = ["VisualQaConfig", "VisualQaGenerationMode", "VisualQaTask"]
