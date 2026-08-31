# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense captioning implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core import SceneContext, ScenePaths, is_image_path, write_json
from core.formats.daft import daft_envelope
from core.media import discover_image_group

from captioning.clients import CaptionEndpointClient, ChatRequest, MediaPayload
from captioning.config import CaptioningConfig
from captioning.media import (
    CaptionWindow,
    ImageInfo,
    MediaDecodeError,
    VideoInfo,
    extract_window_frames,
    extract_window_video_payload,
    plan_windows,
    probe_image,
    probe_video,
    read_image_payload,
)
from captioning.parsing import (
    extract_inner_chunks,
    extract_json_object,
    extract_window_description,
)
from captioning.prompts import (
    STANDARD_DENSE_VIDEO_PROMPT,
    STANDARD_IMAGE_PROMPT,
    STANDARD_SUMMARY_PROMPT,
    load_prompt,
    prompt_metadata,
)

_SAFE_SIDECAR_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class CaptionResult:
    """Artifact paths written by a captioning run."""

    success: bool
    sidecar_json: Path | None = None
    video_json: Path | None = None
    image_json: Path | None = None


class DenseCaptioner:
    """Dense image/video captioning mechanics using endpoint-backed VLM/LLM clients."""

    def __init__(
        self,
        *,
        config: CaptioningConfig,
        vlm_client: CaptionEndpointClient,
        llm_client: CaptionEndpointClient | None = None,
    ) -> None:
        if config.enable_llm_summary and llm_client is None:
            raise ValueError("enable_llm_summary requires an llm_client")
        self.config = config
        self.vlm_client = vlm_client
        self.llm_client = llm_client

    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        """Run captioning for one media sample."""
        if self.config.image_group_dir is not None:
            if not is_image_path(media_path):
                raise ValueError("image_group_dir requires an image DataEntry.media_path")
            image_paths = discover_image_group(
                self.config.image_group_dir,
                max_images=self.config.max_group_images,
            )
            return self._run_image_group(
                media_path=media_path,
                image_paths=image_paths,
                scene_paths=scene_paths,
                scene_ctx=scene_ctx,
            )
        if is_image_path(media_path):
            return self._run_image(
                media_path=media_path, scene_paths=scene_paths, scene_ctx=scene_ctx
            )
        return self._run_video(media_path=media_path, scene_paths=scene_paths, scene_ctx=scene_ctx)

    def _run_image(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        prompt = load_prompt(
            prompt_text=self.config.image_prompt_text,
            prompt_file=self.config.image_prompt_file,
            default_prompt=STANDARD_IMAGE_PROMPT,
        )
        image_info = probe_image(media_path)
        image_payload = read_image_payload(media_path)
        raw = self.vlm_client.generate(
            ChatRequest(
                prompt=prompt,
                media=(image_payload,),
                system_prompt=self.config.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        )
        call_metadata = _client_call_metadata(self.vlm_client)
        parsed = extract_json_object(raw)
        caption = extract_window_description(raw, parsed)
        sidecar = scene_paths.sidecars_dir / "captioning" / "image_caption.json"
        write_json(
            sidecar,
            {
                "schema_version": "1",
                "artifact_type": "image_caption_sidecar",
                "stage": _stage_metadata(),
                "image_id": scene_ctx.media_id,
                "source_image": str(media_path),
                "media": _image_media_metadata(
                    scene_ctx=scene_ctx,
                    media_path=media_path,
                    image_info=image_info,
                ),
                "model": _endpoint_metadata(
                    provider=self.config.vlm_provider,
                    endpoint_url=self.config.vlm_endpoint_url,
                    model=self.config.vlm_model,
                ),
                "generation": _generation_metadata(self.config),
                "prompt": prompt_metadata(
                    prompt=prompt,
                    prompt_text=self.config.image_prompt_text,
                    prompt_file=self.config.image_prompt_file,
                    default_name="standard_image",
                ),
                "raw_response": raw,
                "parsed": parsed,
                "caption": caption,
                "caption_status": "success" if caption else "empty",
                "caption_failure_reason": None,
                "token_counts": _token_counts(call_metadata),
                "call": call_metadata,
                "valid": bool(caption),
            },
        )
        image_json = None
        if self.config.write_contextual:
            image_json = scene_paths.sidecars_dir / "captioning" / "image_captions.json"
            payload = daft_envelope("image_caption", scene_ctx)
            payload["caption"] = caption
            if parsed:
                payload["model_output"] = parsed
            write_json(image_json, payload)
        return CaptionResult(success=bool(caption), sidecar_json=sidecar, image_json=image_json)

    def _run_image_group(
        self,
        *,
        media_path: Path,
        image_paths: tuple[Path, ...],
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        """Caption multiple views of one identity in a single VLM request."""
        image_group_dir = self.config.image_group_dir
        if image_group_dir is None:
            raise RuntimeError("image-group captioning requires image_group_dir")
        prompt = load_prompt(
            prompt_text=self.config.image_prompt_text,
            prompt_file=self.config.image_prompt_file,
            default_prompt=STANDARD_IMAGE_PROMPT,
        )
        prompt = (
            f"{prompt}\n\nTreat all supplied images as different views of the same identity. "
            "Produce one consolidated description using evidence across the views."
        )
        image_infos = [probe_image(path) for path in image_paths]
        image_payloads = tuple(read_image_payload(path) for path in image_paths)
        raw = self.vlm_client.generate(
            ChatRequest(
                prompt=prompt,
                media=image_payloads,
                system_prompt=self.config.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        )
        call_metadata = _client_call_metadata(self.vlm_client)
        parsed = extract_json_object(raw)
        caption = extract_window_description(raw, parsed)
        source_images = [str(path) for path in image_paths]
        sidecar = scene_paths.sidecars_dir / "captioning" / "image_caption.json"
        write_json(
            sidecar,
            {
                "schema_version": "1",
                "artifact_type": "image_caption_sidecar",
                "stage": _stage_metadata(),
                "image_id": scene_ctx.media_id,
                "source_image": str(media_path),
                "source_images": source_images,
                "media": {
                    "kind": "image_group",
                    "source_directory": str(Path(image_group_dir).expanduser()),
                    "representative_image": str(media_path),
                    "num_images": len(image_paths),
                    "num_views": len(image_paths),
                    "total_bytes": sum(info.num_bytes for info in image_infos),
                    "max_width": max(info.width for info in image_infos),
                    "max_height": max(info.height for info in image_infos),
                    "images": [
                        _image_media_metadata(
                            scene_ctx=scene_ctx,
                            media_path=path,
                            image_info=info,
                        )
                        for path, info in zip(image_paths, image_infos, strict=True)
                    ],
                },
                "model": _endpoint_metadata(
                    provider=self.config.vlm_provider,
                    endpoint_url=self.config.vlm_endpoint_url,
                    model=self.config.vlm_model,
                ),
                "generation": _generation_metadata(self.config),
                "prompt": prompt_metadata(
                    prompt=prompt,
                    prompt_text=self.config.image_prompt_text,
                    prompt_file=self.config.image_prompt_file,
                    default_name="standard_image_group",
                ),
                "raw_response": raw,
                "parsed": parsed,
                "caption": caption,
                "caption_status": "success" if caption else "empty",
                "caption_failure_reason": None,
                "token_counts": _token_counts(call_metadata),
                "call": call_metadata,
                "valid": bool(caption),
            },
        )
        image_json = None
        if self.config.write_contextual:
            image_json = scene_paths.sidecars_dir / "captioning" / "image_captions.json"
            payload = daft_envelope("image_caption", scene_ctx)
            payload["caption"] = caption
            payload["source_images"] = source_images
            payload["num_views"] = len(image_paths)
            if parsed:
                payload["model_output"] = parsed
            write_json(image_json, payload)
        return CaptionResult(success=bool(caption), sidecar_json=sidecar, image_json=image_json)

    def _run_video(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> CaptionResult:
        prompt = load_prompt(
            prompt_text=self.config.prompt_text,
            prompt_file=self.config.prompt_file,
            default_prompt=STANDARD_DENSE_VIDEO_PROMPT,
        )
        info = probe_video(media_path)
        windows = plan_windows(info, self.config)
        windows_out: list[dict[str, Any]] = []
        window_token_counts: list[dict[str, int]] = []
        window_finish_reasons: list[str] = []
        for window in windows:
            media_payloads, input_mode, media_error = self._window_media_payloads(
                media_path=media_path,
                window=window,
                info=info,
            )
            window_prompt = (
                f"{prompt}\n\nWindow start: {window.start_s:.3f}s. "
                f"Window end: {window.end_s:.3f}s. "
                "Media timestamps are relative to this window."
            )
            try:
                raw = self._generate_window_caption(
                    prompt=window_prompt,
                    media_payloads=media_payloads,
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
                raw = self._generate_window_caption(
                    prompt=window_prompt,
                    media_payloads=media_payloads,
                )
            call_metadata = _client_call_metadata(self.vlm_client)
            parsed = extract_json_object(raw)
            description = extract_window_description(raw, parsed)
            caption_status = "success" if description else "empty"
            token_counts = _token_counts(call_metadata)
            if token_counts:
                window_token_counts.append(token_counts)
            finish_reason = call_metadata.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                window_finish_reasons.append(finish_reason)
            inner_chunks = extract_inner_chunks(parsed)
            window_record: dict[str, Any] = {
                "index": window.index,
                "start_s": round(window.start_s, 3),
                "end_s": round(window.end_s, 3),
                "duration_s": round(max(0.0, window.end_s - window.start_s), 3),
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "input_media_mode": input_mode,
                "description": description,
            }
            if inner_chunks:
                window_record["inner_chunks"] = inner_chunks
            if media_error:
                window_record["media_failure_reason"] = media_error
            if caption_status != "success":
                window_record["caption_status"] = caption_status
            if self.config.preserve_raw_model_output:
                window_record["caption_status"] = caption_status
                window_record["caption_failure_reason"] = None
                window_record["call"] = call_metadata
                if token_counts:
                    window_record["token_counts"] = token_counts
                if isinstance(finish_reason, str) and finish_reason:
                    window_record["finish_reason"] = finish_reason
                window_record["input_media"] = _media_payload_metadata(media_payloads)
                window_record["vlm_caption"] = raw
                window_record["parsed"] = parsed
            windows_out.append(window_record)
        summary = None
        summary_call_metadata: dict[str, Any] | None = None
        if self.config.enable_llm_summary:
            summary, summary_call_metadata = self._summarize_windows(windows_out)
        sidecar_filename = _safe_sidecar_filename(self.config.sidecar_filename)
        sidecar = scene_paths.sidecars_dir / "captioning" / sidecar_filename
        has_caption = any(bool(window.get("description")) for window in windows_out)
        windowing: dict[str, Any] = {
            "window_mode": (
                "single"
                if self.config.single_window
                else ("frames" if self.config.window_frames > 0 else "seconds")
            ),
            "window_seconds": self.config.window_seconds,
            "window_frames": self.config.window_frames,
            "remainder_threshold": self.config.remainder_threshold,
            "single_window": self.config.single_window,
            "sampling_fps": self.config.sampling_fps,
            "resolution": self.config.resolution,
            "max_frames": self.config.max_frames,
            "media_mode": self.config.media_mode,
            "caption_key": "description",
        }
        if self.config.preserve_raw_model_output:
            windowing["raw_caption_key"] = "vlm_caption"
            windowing["parsed_key"] = "parsed"
        total_prompt_tokens = _sum_token_count_values(
            window_token_counts, "prompt_tokens"
        ) + _token_count_value(summary_call_metadata, "prompt_tokens")
        total_output_tokens = _sum_token_count_values(
            window_token_counts, "output_tokens"
        ) + _token_count_value(summary_call_metadata, "output_tokens")
        total_tokens = _sum_token_count_values(window_token_counts, "total_tokens") + (
            _token_count_value(summary_call_metadata, "total_tokens")
        )
        metadata: dict[str, Any] = {
            "schema_version": "1",
            "artifact_type": "video_dense_caption_sidecar",
            "stage": _stage_metadata(),
            "video_id": scene_ctx.media_id,
            "source_video": str(media_path),
            "media": _video_media_metadata(scene_ctx=scene_ctx, media_path=media_path, info=info),
            "model": _endpoint_metadata(
                provider=self.config.vlm_provider,
                endpoint_url=self.config.vlm_endpoint_url,
                model=self.config.vlm_model,
            ),
            "generation": _generation_metadata(self.config),
            "prompt": prompt_metadata(
                prompt=prompt,
                prompt_text=self.config.prompt_text,
                prompt_file=self.config.prompt_file,
                default_name="standard_dense_video",
            ),
            "duration_span": [0.0, round(info.duration_s, 3)],
            "width": info.width,
            "height": info.height,
            "framerate": info.fps,
            "num_frames": info.frame_count,
            "windowing": windowing,
            "windows": windows_out,
            "summary": summary,
            "llm_summary": _summary_metadata(
                config=self.config,
                summary=summary,
                call_metadata=summary_call_metadata,
            ),
            "valid": has_caption,
            "has_caption": has_caption,
            "caption_status": "success" if has_caption else "empty",
            "caption_failure_reason": None,
            "token_counts": _total_token_counts(
                prompt_tokens=total_prompt_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_tokens,
            ),
            "finish_reasons": _unique_values(window_finish_reasons),
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "num_caption_windows": sum(1 for window in windows_out if window.get("description")),
        }
        write_json(sidecar, metadata)
        video_json = None
        if self.config.write_contextual:
            video_json = scene_paths.sidecars_dir / "captioning" / "video_captions.json"
            payload = daft_envelope("video_caption", scene_ctx)
            payload["duration_span"] = metadata["duration_span"]
            payload["summary"] = summary
            payload["windows"] = [
                {
                    "start_time": window["start_s"],
                    "end_time": window["end_s"],
                    "start_frame": window["start_frame"],
                    "end_frame": window["end_frame"],
                    "caption": window["description"],
                }
                for window in windows_out
            ]
            write_json(video_json, payload)
        return CaptionResult(success=has_caption, sidecar_json=sidecar, video_json=video_json)

    def _summarize_windows(
        self, windows: list[dict[str, Any]]
    ) -> tuple[str | None, dict[str, Any] | None]:
        llm_client = self.llm_client
        if llm_client is None:
            raise RuntimeError("enable_llm_summary requires an llm_client")
        summary_inputs = _summary_inputs_from_windows(windows)
        if not summary_inputs:
            return None, None
        prompt = load_prompt(
            prompt_text=self.config.summary_prompt_text,
            prompt_file=self.config.summary_prompt_file,
            default_prompt=STANDARD_SUMMARY_PROMPT,
        )
        call_metadata: list[dict[str, Any]] = []
        batch_summaries, batch_metadata = self._generate_summary_batches(
            summary_inputs,
            prompt=prompt,
            input_token_budget=self.config.summary_input_token_budget,
            max_output_tokens=self.config.max_tokens,
        )
        call_metadata.extend(batch_metadata)
        if not batch_summaries:
            return None, None
        merge_prompt = (
            "Merge the following partial video summaries into one concise, factual "
            "scene-level description. Preserve concrete actions and avoid inventing events."
        )
        while len(batch_summaries) > 1:
            previous_count = len(batch_summaries)
            merged_summaries, merge_metadata = self._generate_summary_batches(
                [
                    f"[partial_summary {idx}] {summary}"
                    for idx, summary in enumerate(batch_summaries)
                ],
                prompt=merge_prompt,
                input_token_budget=self.config.summary_input_token_budget,
                max_output_tokens=self.config.max_tokens,
            )
            call_metadata.extend(merge_metadata)
            if not merged_summaries:
                break
            if len(merged_summaries) >= previous_count:
                batch_summaries = ["\n".join(merged_summaries)]
                break
            batch_summaries = merged_summaries
        return "\n".join(batch_summaries), _combine_call_metadata(call_metadata)

    def _generate_summary_batches(
        self,
        inputs: list[str],
        *,
        prompt: str,
        input_token_budget: int,
        max_output_tokens: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        llm_client = self.llm_client
        if llm_client is None:
            raise RuntimeError("enable_llm_summary requires an llm_client")
        summaries: list[str] = []
        metadata: list[dict[str, Any]] = []
        for batch in chunk_descriptions_by_budget(
            inputs,
            prompt=prompt,
            input_token_budget=input_token_budget,
            max_output_tokens=max_output_tokens,
        ):
            raw = llm_client.generate(
                ChatRequest(
                    prompt=prompt + "\n\n" + "\n".join(batch),
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            )
            summary = raw.strip()
            if summary:
                summaries.append(summary)
            metadata.append(_client_call_metadata(llm_client))
        return summaries, metadata

    def _generate_window_caption(
        self,
        *,
        prompt: str,
        media_payloads: list[MediaPayload],
    ) -> str:
        return self.vlm_client.generate(
            ChatRequest(
                prompt=prompt,
                media=tuple(media_payloads),
                system_prompt=self.config.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        )

    def _window_media_payloads(
        self,
        *,
        media_path: Path,
        window: CaptionWindow,
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


def _safe_sidecar_filename(filename: str) -> str:
    candidate = filename.strip()
    path = Path(candidate)
    if (
        not candidate
        or path.is_absolute()
        or path.name != candidate
        or ".." in candidate
        or not _SAFE_SIDECAR_FILENAME_RE.fullmatch(candidate)
    ):
        raise ValueError(
            "sidecar_filename must be a plain filename using only letters, numbers, dots, "
            "dashes, and underscores"
        )
    return candidate


def _stage_metadata() -> dict[str, str]:
    return {
        "name": "captioning",
        "implementation": "DenseCaptioner",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _file_metadata(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source_path": str(path),
        "filename": path.name,
        "stem": path.stem,
        "suffix": path.suffix.lower(),
    }
    try:
        out["num_bytes"] = path.stat().st_size
    except OSError:
        out["num_bytes"] = None
    return out


def _image_media_metadata(
    *,
    scene_ctx: SceneContext,
    media_path: Path,
    image_info: ImageInfo,
) -> dict[str, Any]:
    return {
        "kind": "image",
        "image_id": scene_ctx.media_id,
        **_file_metadata(media_path),
        "mime_type": image_info.mime_type,
        "width": image_info.width,
        "height": image_info.height,
        "decoded": image_info.width > 0 and image_info.height > 0,
    }


def _video_media_metadata(
    *,
    scene_ctx: SceneContext,
    media_path: Path,
    info: VideoInfo,
) -> dict[str, Any]:
    return {
        "kind": "video",
        "video_id": scene_ctx.media_id,
        **_file_metadata(media_path),
        "width": info.width,
        "height": info.height,
        "framerate": info.fps,
        "num_frames": info.frame_count,
        "duration_s": round(info.duration_s, 3),
    }


def _endpoint_metadata(
    *,
    provider: str | None,
    endpoint_url: str | None,
    model: str | None,
) -> dict[str, str | None]:
    return {
        "provider": provider,
        "model": model,
        "endpoint_url": endpoint_url,
    }


def _generation_metadata(config: CaptioningConfig) -> dict[str, int | float | bool]:
    return {
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "timeout_s": config.timeout_s,
        "retries": config.retries,
        "retry_backoff_s": config.retry_backoff_s,
        "preserve_raw_model_output": config.preserve_raw_model_output,
    }


def _media_payload_metadata(media: list[MediaPayload]) -> list[dict[str, Any]]:
    return [
        {
            "filename": item.filename,
            "mime_type": item.mime_type,
            "frame_index": item.frame_index,
            "time_s": item.time_s,
            "width": item.width,
            "height": item.height,
            "num_bytes": item.num_bytes,
        }
        for item in media
    ]


def _client_call_metadata(client: CaptionEndpointClient) -> dict[str, Any]:
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


def _sum_token_count_values(token_counts: list[dict[str, int]], key: str) -> int:
    total = 0
    for counts in token_counts:
        total += counts.get(key, 0)
    return total


def _token_count_value(call_metadata: dict[str, Any] | None, key: str) -> int:
    counts = _token_counts(call_metadata)
    if counts is None:
        return 0
    return counts.get(key, 0)


def _total_token_counts(
    *,
    prompt_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> dict[str, int] | None:
    out: dict[str, int] = {}
    if prompt_tokens > 0:
        out["prompt_tokens"] = prompt_tokens
    if output_tokens > 0:
        out["output_tokens"] = output_tokens
    if total_tokens > 0:
        out["total_tokens"] = total_tokens
    return out or None


def _unique_values(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _append_failure_reason(existing: str | None, reason: str) -> str:
    if not existing:
        return reason
    return f"{existing};{reason}"


def _summary_inputs_from_windows(windows: list[dict[str, Any]]) -> list[str]:
    inputs: list[str] = []
    for window in windows:
        start_s = _float_value(window.get("start_s"))
        end_s = _float_value(window.get("end_s"))
        chunks = window.get("inner_chunks")
        added_chunk = False
        if isinstance(chunks, list):
            for chunk_idx, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    continue
                description = str(chunk.get("description", "")).strip()
                if not description:
                    continue
                chunk_start = _chunk_time(
                    chunk.get("start"), window_start_s=start_s, fallback=start_s
                )
                chunk_end = _chunk_time(chunk.get("end"), window_start_s=start_s, fallback=end_s)
                inputs.append(
                    f"[{chunk_start:.3f}-{chunk_end:.3f}s chunk {chunk_idx}] {description}"
                )
                added_chunk = True
        if added_chunk:
            continue
        description = str(window.get("description", "")).strip()
        if description and start_s is not None and end_s is not None:
            inputs.append(f"[{start_s:.3f}-{end_s:.3f}s] {description}")
        elif description:
            inputs.append(description)
    return inputs


def chunk_descriptions_by_budget(
    descriptions: list[str],
    *,
    prompt: str,
    input_token_budget: int,
    max_output_tokens: int,
) -> list[list[str]]:
    """Batch all descriptions into prompt-sized groups without dropping text."""
    description_budget = max(
        1,
        input_token_budget - _estimate_tokens(prompt) - max_output_tokens,
    )
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    for description in descriptions:
        if not description.strip():
            continue
        separator_tokens = 1 if current else 0
        token_count = _estimate_tokens(description)
        needed = token_count + separator_tokens
        if current and used + needed > description_budget:
            batches.append(current)
            current = []
            used = 0
        if token_count > description_budget:
            if current:
                batches.append(current)
                current = []
                used = 0
            for part in _split_to_estimated_token_budget(description, description_budget):
                batches.append([part])
            continue
        current.append(description)
        used += token_count + (1 if len(current) > 1 else 0)
    if current:
        batches.append(current)
    return batches


def _split_to_estimated_token_budget(text: str, token_budget: int) -> list[str]:
    max_chars = max(1, token_budget * 4)
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            parts.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return [part for part in parts if part]


def _chunk_time(value: object, *, window_start_s: float | None, fallback: float | None) -> float:
    numeric = _float_value(value)
    if numeric is not None:
        return numeric + window_start_s if window_start_s is not None else numeric
    if isinstance(value, str):
        parsed = _parse_timecode(value)
        if parsed is not None:
            return parsed + window_start_s if window_start_s is not None else parsed
    return fallback if fallback is not None else 0.0


def _parse_timecode(value: str) -> float | None:
    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
    except ValueError:
        return None
    return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _combine_call_metadata(calls: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, int] = {}
    for metadata in calls:
        counts = _token_counts(metadata)
        if counts is None:
            continue
        for key, value in counts.items():
            usage[key] = usage.get(key, 0) + value
    return {"calls": calls, "usage": usage} if usage else {"calls": calls}


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _summary_metadata(
    *,
    config: CaptioningConfig,
    summary: str | None,
    call_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if not config.enable_llm_summary:
        return {"enabled": False}
    return {
        "enabled": config.enable_llm_summary,
        "model": _endpoint_metadata(
            provider=config.llm_provider or config.vlm_provider,
            endpoint_url=config.llm_endpoint_url or config.vlm_endpoint_url,
            model=config.llm_model or config.vlm_model,
        ),
        "caption_status": "success" if summary else "empty",
        "caption_failure_reason": None,
        "token_counts": _token_counts(call_metadata),
        "call": call_metadata,
    }


__all__ = ["CaptionResult", "DenseCaptioner"]
