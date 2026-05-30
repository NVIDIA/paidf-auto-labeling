# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NOTE:
# This module intentionally contains NO CLI/argparse entrypoints.
# It is a pure, importable library used by `vlm_json.vlm_json_generator.VlmJsonGenerator`.

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from al_utils.media_paths import is_image_path
from daft_export.common import get_scene_media_id, write_daft_json
from daft_export.contextual import to_daft_events, to_daft_image, to_daft_video
from daft_export.paths import scene_paths
from mcq_generation.mcq.utils.frame_sampling import sample_frames_ffmpeg
from mcq_generation.mcq.utils.openai import (
    call_chat_object_with_structured_fallback,
    extract_json_object_from_llm_text,
    get_vlm_api_key,
    parse_strict_json_object,
)
from mcq_generation.mcq.utils.video import probe_video
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VideoConfig:
    """Video preprocessing configuration."""

    target_fps: int = 30
    crf: int = 18  # Visually lossless quality
    preset: str = "medium"
    gop_size: int = 30
    profile: str = "high"
    level: str = "4.1"
    pix_fmt: str = "yuv420p"
    timescale: int = 30000


@dataclass
class VLMConfig:
    """VLM API configuration."""

    # NOTE: endpoint/model must be provided by caller (config/CLI).
    base_url: str = ""
    model: str = ""
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.9
    timeout: int = 600
    retries: int = 3
    structured_output: str = "openai"

    # Frame extraction settings
    frame_fps: float = 1.0

    # Frame quality settings
    qscale: int = 5
    # Frame height in pixels (width auto-scaled). Use 0 to auto-pick based on source video.
    resolution: int = 360
    # Cap for auto resolution to avoid blowing up context / cost.
    max_auto_resolution: int = 720
    max_frames: int = 24


@dataclass
class PromptFiles:
    """Resolved prompt files for VLM JSON generation."""

    video_json: Path
    events_json: Path
    image_json: Path


@dataclass
class PipelineConfig:
    """Pipeline configuration (VLM + video + prompt settings).

    Output paths are derived per-call from the ``scene_dir`` passed to
    :meth:`VideoPipeline.process_video`; there is no pipeline-wide output
    directory on this config.
    """

    video_config: VideoConfig
    vlm_config: VLMConfig
    rate_limit: float = 0.0
    prompt_files: Optional[PromptFiles] = field(default=None)
    # Run VLM twice (video.json then events.json). More stable, ~2x cost.
    split_json_calls: bool = True

    def __post_init__(self) -> None:
        if self.prompt_files is None:
            raise ValueError("prompt_files must be provided by the caller")


@dataclass
class PromptBundle:
    """Resolved prompt text used by split and combined VLM calls."""

    video_json: str
    events_json: str
    image_json: str


# =============================================================================
# JSON extraction helpers
# =============================================================================


_EXPECTED_JSON_EXTRACTION_ERRORS = (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError)


def _extract_single_json_object(text: str) -> Dict[str, Any]:
    """Parse one JSON object from a VLM response.

    Structured-output endpoints usually return a raw JSON object, while less
    strict endpoints may wrap the object in prose or fenced Markdown. This
    helper accepts both shapes and raises a clear error when no object can be
    recovered.
    """
    t = str(text or "")
    if not t.strip():
        raise ValueError("Empty model output")
    obj = parse_strict_json_object(t)
    if isinstance(obj, dict):
        return obj
    obj2 = extract_json_object_from_llm_text(t)
    if isinstance(obj2, dict):
        return obj2
    raise ValueError("Could not extract a valid JSON object from model output")


def _extract_two_json_objects(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Recover the two-object VLM response shape.

    Some prompts return separate fenced JSON blocks for scene metadata and
    events. The combined prompt prefers one wrapper object, but this parser
    remains for existing artifacts and best-effort fallback parsing.
    """
    t = str(text or "")
    if not t.strip():
        raise ValueError("Empty model output")

    def _extract_fenced_json_blocks(s: str) -> List[str]:
        """Extract fenced ```json blocks in linear time.

        We intentionally avoid regex here to prevent backtracking/DoS concerns.
        This parser is strict about the fence markers:
        - Opening fence: a line like "```json" (case-insensitive) with only whitespace after "json"
        - Closing fence: a line like "```" (only whitespace after)
        """

        def _is_open_fence(line: str) -> bool:
            t = line.lstrip()
            if not t.startswith("```"):
                return False
            rest = t[3:].strip()
            if not rest:
                return False
            parts = rest.split(None, 1)
            if parts[0].lower() != "json":
                return False
            if len(parts) == 2 and parts[1].strip():
                return False
            return True

        def _is_close_fence(line: str) -> bool:
            t = line.lstrip()
            if not t.startswith("```"):
                return False
            rest = t[3:].strip()
            return rest == ""

        blocks: List[str] = []
        in_block = False
        buf: List[str] = []
        for line in s.splitlines():
            if not in_block:
                if _is_open_fence(line):
                    in_block = True
                    buf = []
                continue
            # in_block
            if _is_close_fence(line):
                payload = "\n".join(buf).strip()
                if payload:
                    blocks.append(payload)
                in_block = False
                buf = []
                continue
            buf.append(line)
        return blocks

    def _repair_common_json_errors(s: str) -> str:
        out = str(s or "")
        out = re.sub(r'"event_id":\s*"(event_\d+),', r'"event_id": "\1",', out)
        out = re.sub(
            r'"([^"]+)":\s*"([^"\n]*),\s*$',
            r'"\1": "\2",',
            out,
            flags=re.MULTILINE,
        )
        return out

    def _scan_json_objects(s: str) -> List[str]:
        """Return complete top-level JSON object substrings from free-form text."""
        objects: List[str] = []
        start: Optional[int] = None
        depth = 0
        in_string = False
        escaped = False

        for i, ch in enumerate(s):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(s[start : i + 1])
                    start = None

        return objects

    def _parse_obj(s: str) -> Optional[Dict[str, Any]]:
        obj = parse_strict_json_object(s)
        if isinstance(obj, dict):
            return obj
        obj2 = extract_json_object_from_llm_text(s)
        return obj2 if isinstance(obj2, dict) else None

    blocks = _extract_fenced_json_blocks(t)
    if len(blocks) >= 2:
        obj1 = _parse_obj(blocks[0])
        obj2 = _parse_obj(blocks[1])
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            return obj1, obj2

    repaired = _repair_common_json_errors(t)
    blocks = _extract_fenced_json_blocks(repaired)
    if len(blocks) >= 2:
        obj1 = _parse_obj(blocks[0])
        obj2 = _parse_obj(blocks[1])
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            return obj1, obj2

    # Best-effort: look for two complete JSON objects in free-form text.
    objs = _scan_json_objects(t)
    if len(objs) >= 2:
        obj1 = _parse_obj(objs[0])
        obj2 = _parse_obj(objs[1])
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            return obj1, obj2

    raise ValueError("Could not extract two JSON objects from model output")


def _extract_combined_video_events(
    text: str, logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract non-split video output.

    Preferred shape is one wrapper object:
      {"video_json": {...}, "events_json": {"events": [...]}}

    The older two-fenced-block shape is still accepted for existing prompts.
    """

    try:
        return _extract_two_json_objects(text)
    except Exception as two_obj_error:
        log = logger or logging.getLogger(__name__)
        log.warning(
            "Combined VLM JSON extraction: two-object parse failed; falling back to wrapper-object parse (%s)",
            two_obj_error,
        )
        obj = _extract_single_json_object(text)
        video_obj = obj.get("video_json") or obj.get("video")
        events_obj = obj.get("events_json")
        if events_obj is None and isinstance(obj.get("events"), list):
            events_obj = {"events": obj.get("events")}
        if not isinstance(video_obj, dict):
            candidate_video = {
                k: v for k, v in obj.items() if k not in {"events", "events_json", "video_json", "video"}
            }
            video_obj = candidate_video if candidate_video else None
        if not isinstance(events_obj, dict):
            raise two_obj_error
        if not isinstance(video_obj, dict):
            raise two_obj_error
        return video_obj, events_obj


def _events_validator(obj: Dict[str, Any]) -> bool:
    """Validate the events.json shape returned by the VLM.

    Required: a top-level ``events`` key whose value is a list. Used by
    ``QwenVLMClient.analyze_frames`` to make schema mismatches retryable
    inside ``call_chat_object_with_structured_fallback``, instead of
    silently falling back to empty events when a model returns valid
    JSON of the wrong shape (observed with qwen3.5-122b-a10b).
    """
    return isinstance(obj, dict) and isinstance(obj.get("events"), list)


def _combined_video_events_validator(obj: Dict[str, Any]) -> bool:
    """Validate the combined video+events wrapper shape (non-split mode).

    The combined prompt asks the model to return one wrapper containing
    both the video metadata object and the events object. Mirrors the
    shapes accepted by ``_extract_combined_video_events`` so that a model
    returning the wrong shape gets re-prompted instead of silently
    falling through to the empty-events fallback.
    """
    if not isinstance(obj, dict):
        return False
    video_obj = obj.get("video_json") or obj.get("video")
    events_obj = obj.get("events_json")
    if events_obj is None and isinstance(obj.get("events"), list):
        events_obj = {"events": obj["events"]}
    return isinstance(video_obj, dict) and isinstance(events_obj, dict) and isinstance(events_obj.get("events"), list)


# =============================================================================
# Video preprocessing
# =============================================================================


class VideoPreprocessor:
    """Frame extraction helper for VLM."""

    def __init__(self, config: VLMConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Probe media container metadata needed for DAFT contextual outputs."""
        try:
            vi = probe_video(video_path)
            return {
                "width": int(vi.width),
                "height": int(vi.height),
                "fps": float(vi.fps),
                "nb_frames": int(vi.num_frames),
                "duration": float(vi.duration_sec),
            }
        except Exception as e:
            self.logger.error("Failed to probe video %s: %s", video_path, e)
            return {}

    def extract_frames(
        self,
        input_path: Path,
        output_dir: Path,
        fps: float = 1.0,
        qscale: int = 5,
        resolution: int = 360,
        *,
        max_auto_resolution: int = 720,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sample video frames, or normalize an image input into one JPEG frame.

        Video inputs are sampled through the shared ffmpeg sampler with even
        max-frame coverage. Image inputs take the same downstream path by
        writing one resized JPEG into the frame directory.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            chosen_resolution = int(resolution)
            if chosen_resolution == 0:
                meta = self.get_video_info(input_path)
                src_h = int(meta.get("height") or 0)
                if src_h > 0:
                    chosen_resolution = min(src_h, int(max_auto_resolution))
                else:
                    chosen_resolution = int(max_auto_resolution)

            # Image input: write a single JPEG frame.
            if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                start_time = time.time()
                with Image.open(input_path) as im0:
                    im = im0.convert("RGB")
                    w0, h0 = im.size
                    if chosen_resolution > 0 and h0 > 0 and int(h0) != chosen_resolution:
                        scale = float(chosen_resolution) / float(h0)
                        target_w = int(round(float(w0) * scale))
                        if target_w % 2 == 1:
                            target_w = max(2, target_w - 1)
                        target_h = int(chosen_resolution)
                        if target_h % 2 == 1:
                            target_h = max(2, target_h - 1)
                        im = im.resize((int(target_w), int(target_h)), resample=Image.Resampling.LANCZOS)

                out_path = output_dir / "frame_000001.jpg"
                jpeg_quality = 95 if int(qscale or 5) <= 4 else 90
                try:
                    im.save(out_path, format="JPEG", quality=int(jpeg_quality), optimize=True)
                except Exception:
                    im.save(out_path, format="JPEG", quality=int(jpeg_quality))
                elapsed = time.time() - start_time
                return True, {
                    "success": True,
                    "frame_count": 1,
                    "fps": 1.0,
                    "resolution": chosen_resolution,
                    "extraction_time_sec": elapsed,
                    "frame_interval_sec": 1.0,
                    "mode": "image",
                }

            meta = self.get_video_info(input_path)
            duration = float(meta.get("duration") or 0.0)
            if duration <= 0:
                duration = 1.0

            self.logger.info("Extracting frames at %s FPS (scale height=%s)...", fps, chosen_resolution)
            start_time = time.time()
            frames = sample_frames_ffmpeg(
                video_path=input_path,
                out_dir=output_dir,
                start_sec=0.0,
                end_sec=duration,
                sampling_fps=float(fps),
                resolution=int(chosen_resolution),
                max_frames=int(self.config.max_frames),
                logger=self.logger,
                qscale=int(qscale),
            )
            elapsed = time.time() - start_time
            frame_count = len(frames)

            extraction_info: Dict[str, Any] = {
                "success": True,
                "frame_count": frame_count,
                "fps": fps,
                "resolution": chosen_resolution,
                "extraction_time_sec": elapsed,
                "frame_interval_sec": (1.0 / float(fps)) if float(fps) > 0 else 0.0,
            }
            self.logger.info("✓ Extracted %s frames at %s FPS", frame_count, fps)
            return True, extraction_info
        except Exception as e:
            self.logger.error("Frame extraction failed: %s", e)
            return False, {"success": False, "error": str(e)}


# =============================================================================
# VLM API Client
# =============================================================================


class QwenVLMClient:
    """OpenAI-compatible VLM API client with retries."""

    def __init__(self, config: VLMConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    @staticmethod
    def _read_b64(path: Path) -> str:
        """Read an extracted frame as base64 for OpenAI-compatible image input."""
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def _messages_from_frames(
        self, frames_dir: Path, prompt: str, *, frame_fps_param: float, video_fps_param: float
    ) -> List[Dict[str, Any]]:
        """Build a multimodal chat message with timestamp labels before frames."""
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            return [{"role": "user", "content": [{"type": "text", "text": str(prompt)}]}]

        step = max(1, int(float(frame_fps_param) * float(video_fps_param)))
        content: List[Dict[str, Any]] = []
        for i in range(0, len(frames), step):
            sec = f"{i / float(video_fps_param):.1f}"
            content.append({"type": "text", "text": f"<{sec} seconds>"})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._read_b64(frames[i])}"}}
            )
        content.append({"type": "text", "text": str(prompt)})
        return [{"role": "user", "content": content}]

    def analyze_frames(
        self,
        frames_dir: Path,
        prompt: str,
        output_dir: Path,
        *,
        structured_output_override: Optional[str] = None,
        retry_stage: str = "",
        retry_dump_dir: Optional[Path] = None,
        validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Call the VLM on extracted frames and persist the raw text response.

        Pass ``validator`` to make schema mismatches retryable: the underlying
        retry loop re-prompts the model when a response parses to a dict but
        fails the validator (e.g. missing the expected ``events`` key).
        """
        num_frames = len(list(frames_dir.glob("*.jpg")))
        if num_frames == 0:
            self.logger.error("No frames found in %s", frames_dir)
            return False, "No frames found"

        max_frames = int(self.config.max_frames)
        if num_frames > max_frames:
            self.logger.warning("Too many frames (%s); will sample down to %s frames.", num_frames, max_frames)

        try:
            extraction_fps = float(self.config.frame_fps)
            if num_frames > max_frames:
                step = int(num_frames / max_frames)
                frame_fps_param = step / extraction_fps
                video_fps_param = extraction_fps
            else:
                frame_fps_param = 1.0 / extraction_fps
                video_fps_param = extraction_fps

            messages = self._messages_from_frames(
                frames_dir,
                prompt,
                frame_fps_param=float(frame_fps_param),
                video_fps_param=float(video_fps_param),
            )
            structured_mode = (
                str(
                    structured_output_override
                    if structured_output_override is not None
                    else getattr(self.config, "structured_output", "openai")
                ).strip()
                or "openai"
            )

            obj, text = call_chat_object_with_structured_fallback(
                base_url=str(self.config.base_url),
                model=str(self.config.model),
                messages=messages,
                timeout=int(self.config.timeout),
                max_tokens=int(self.config.max_tokens),
                temperature=float(self.config.temperature),
                top_p=float(self.config.top_p),
                retries=int(self.config.retries or 0),
                structured_output=structured_mode,
                guided_json_schema={"type": "object"},
                invalid_fallback="off",
                retry_stage=retry_stage,
                retry_dump_dir=(str(retry_dump_dir) if retry_dump_dir is not None else ""),
                logger=self.logger,
                api_key=get_vlm_api_key(),
                validator=validator,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "output.txt").write_text(text, encoding="utf-8")
            if validator is not None and obj is None:
                self.logger.error(
                    "VLM response failed schema validation after all retries (stage=%s); raw text persisted for diagnostics.",
                    retry_stage or "<unspecified>",
                )
                return False, "schema_validation_failed"
            return True, None
        except Exception as e:
            self.logger.error("Unexpected error during VLM analysis: %s", e)
            return False, str(e)


# =============================================================================
# Main Pipeline
# =============================================================================


class VideoPipeline:
    """Main pipeline orchestrator for VLM JSON generation."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        self.preprocessor = VideoPreprocessor(config.vlm_config, logger)
        self.vlm_client = QwenVLMClient(config.vlm_config, logger)

        self.prompts = self._load_prompt_bundle()

    def _read_prompt_file(self, path: Path, *, label: str) -> Optional[str]:
        """Read an optional prompt file and log recoverable file/encoding errors."""
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            self.logger.warning("Failed to read %s prompt file (%s): %s", label, str(path), str(e))
        return None

    def _load_prompt_text(
        self,
        *,
        label: str,
        path: Path,
    ) -> str:
        """Read a required prompt file or raise a startup-time configuration error."""
        text = self._read_prompt_file(path, label=label)
        if text is not None:
            return text

        raise FileNotFoundError(f"VLM JSON prompt file is missing or unreadable: {path}")

    def _load_prompt_bundle(self) -> PromptBundle:
        """Load all shipped/overridden prompt files once during pipeline construction."""
        prompt_files = self.config.prompt_files
        return PromptBundle(
            video_json=self._load_prompt_text(
                label="video.json",
                path=prompt_files.video_json,
            ),
            events_json=self._load_prompt_text(
                label="events.json",
                path=prompt_files.events_json,
            ),
            image_json=self._load_prompt_text(
                label="image.json",
                path=prompt_files.image_json,
            ),
        )

    @staticmethod
    def _single_output_prompt(prompt: str, *, output_name: str, requirement: str) -> str:
        """Constrain a shipped subprompt to produce exactly one target JSON file."""
        return (
            prompt.rstrip()
            + f"\n\nIMPORTANT: For this call, output ONLY the {output_name} object.\n"
            + requirement
            + "\nOutput exactly ONE fenced JSON block (```json ... ```), and nothing else.\n"
        )

    @staticmethod
    def _combined_output_prompt(*, metadata_name: str, metadata_prompt: str, events_prompt: str) -> str:
        """Build the non-split video prompt that asks for metadata and events together.

        `split_json_calls=false` sends one VLM request for video inputs rather
        than separate `video.json` and `events.json` calls. This wrapper prompt
        preserves the two shipped subprompts but requires the model to return a
        single fenced JSON object that can be split deterministically into the
        scene metadata object and the events object.
        """
        return (
            "You will produce one combined JSON wrapper from the same input frames.\n"
            "The wrapper will later be split into two files: video.json and events.json.\n"
            "Any instruction inside the subprompts that says to return exactly one JSON block applies to the "
            "wrapper response as a whole.\n\n"
            "Required full-response format: output EXACTLY ONE fenced JSON block and nothing else.\n"
            "```json\n"
            "{\n"
            f'  "{metadata_name.replace(".", "_")}": {{\n'
            '    "scene_description": "<scene description>",\n'
            '    "event_summary": "<activity summary>",\n'
            '    "scenario_info": "<scene type>"\n'
            "  },\n"
            '  "events_json": {\n'
            '    "events": [\n'
            "      {\n"
            '        "event_id": "event_001",\n'
            '        "event_caption": "<visible event>",\n'
            '        "start_time": 0.0,\n'
            '        "end_time": 1.0,\n'
            '        "category": "movement",\n'
            '        "instances": []\n'
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
            "```\n\n"
            f"## Prompt for {metadata_name}\n\n"
            + metadata_prompt.rstrip()
            + "\n\n## Prompt for events.json\n\n"
            + events_prompt.rstrip()
            + "\n\nIMPORTANT: Output exactly one wrapper object with top-level keys "
            + f"`{metadata_name.replace('.', '_')}` and `events_json`.\n"
            + f"`{metadata_name.replace('.', '_')}` MUST NOT include an `events` field.\n"
            + "`events_json` MUST include top-level key `events` whose value is a list.\n"
        )

    def process_video(self, video_path: Path, scene_dir: Path) -> Dict[str, Any]:
        """Run VLM JSON generation on one video and emit DAFT outputs.

        Writes ``contextual/video.json`` plus ``contextual/events.json`` for
        videos, or ``contextual/image.json`` for still images. Frame extraction scratch and
        raw VLM text responses live under ``scene_dir/sidecars/_work/vlm_json/``;
        frames are cleaned on success, raw text is retained for debugging.

        Args:
            video_path: Input video (post-SR if SR ran, else the original).
            scene_dir:  Per-sample scene root.
        """
        video_path = Path(video_path)
        scene_dir = Path(scene_dir)
        paths = scene_paths(scene_dir)

        # Ephemeral work area: sampled frames + raw VLM text responses. Not
        # part of the DAFT scene; lives under sidecars/_work so validators ignore it.
        work_dir = paths.sidecars_dir / "_work" / "vlm_json"
        work_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = work_dir / "frames"

        result: Dict[str, Any] = {
            "input_path": str(video_path),
            "success": False,
            "vlm_analysis": {},
            "json_extraction": {},
        }
        is_image_input = is_image_path(video_path)

        self.logger.info("Step 1/3: Frame Extraction")
        extract_success, extract_info = self.preprocessor.extract_frames(
            video_path,
            frames_dir,
            fps=float(self.config.vlm_config.frame_fps),
            qscale=int(self.config.vlm_config.qscale),
            resolution=int(self.config.vlm_config.resolution),
            max_auto_resolution=int(self.config.vlm_config.max_auto_resolution),
        )
        result["frame_extraction"] = extract_info
        if not extract_success:
            self.logger.warning("Frame extraction failed; writing DAFT fallback outputs where possible.")
            result["json_extraction"] = self._write_fallback_outputs(
                video_path=video_path,
                scene_dir=scene_dir,
                is_image_input=is_image_input,
                reason=str(extract_info.get("error") or "frame_extraction_failed"),
            )
            if result["json_extraction"].get("success"):
                self.logger.warning(
                    "VLM JSON fallback output written for %s scene: %s",
                    result["json_extraction"].get("scene_kind"),
                    result["json_extraction"].get("contextual_json"),
                )
            return result

        self.logger.info("Step 2/3: VLM Analysis")
        start_time = time.time()

        split_mode = bool(getattr(self.config, "split_json_calls", False))
        configured_structured = (
            str(getattr(self.config.vlm_config, "structured_output", "openai") or "openai").strip() or "openai"
        )
        effective_structured = configured_structured
        if is_image_input:
            self.logger.info("VLM JSON call plan: image input -> image.json only")
        elif split_mode:
            self.logger.info("VLM JSON call plan: video input -> split video.json then events.json")
        else:
            self.logger.info("VLM JSON call plan: video input -> combined video.json + events.json")

        def _run_one(
            prompt_text: str,
            *,
            kind: str,
            validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
        ) -> Tuple[bool, Optional[str], Optional[Path]]:
            ok, err = self.vlm_client.analyze_frames(
                frames_dir,
                prompt_text,
                work_dir,
                structured_output_override=effective_structured,
                retry_stage=f"vlm_json_{kind}",
                retry_dump_dir=work_dir,
                validator=validator,
            )
            out_txt = work_dir / "output.txt"
            if ok and out_txt.exists():
                raw_path = work_dir / f"raw_output.{kind}.txt"
                try:
                    if raw_path.exists():
                        raw_path.unlink()
                    out_txt.rename(raw_path)
                except Exception:
                    raw_path = out_txt
                return ok, None, raw_path

            failed_raw_path = work_dir / f"raw_output.{kind}.failed.txt"
            try:
                if failed_raw_path.exists():
                    failed_raw_path.unlink()
                if out_txt.exists():
                    out_txt.rename(failed_raw_path)
                else:
                    failed_raw_path.write_text(str(err or ""), encoding="utf-8")
                return ok, err, failed_raw_path
            except Exception:
                return ok, err, None

        vlm_success = False
        vlm_error: Optional[str] = None
        raw_paths: Dict[str, Path] = {}

        if split_mode:
            if is_image_input:
                metadata_key = "image"
                metadata_prompt = self._single_output_prompt(
                    self.prompts.image_json,
                    output_name="image.json",
                    requirement="The JSON object should describe the still image and MUST NOT include an 'events' field.",
                )
                metadata_error = "image_json_call_failed"
            else:
                metadata_key = "video"
                metadata_prompt = self._single_output_prompt(
                    self.prompts.video_json,
                    output_name="video.json",
                    requirement="The JSON object should describe the video and MUST NOT include an 'events' field.",
                )
                events_prompt = self._single_output_prompt(
                    self.prompts.events_json,
                    output_name="events.json",
                    requirement="The JSON object MUST have top-level key 'events' whose value is a list.",
                )
                metadata_error = "video_json_call_failed"
            ok1, err1, raw1 = _run_one(metadata_prompt, kind=metadata_key)
            if not ok1:
                vlm_success = False
                vlm_error = err1 or metadata_error
            else:
                if raw1 is not None:
                    raw_paths[metadata_key] = raw1
                if is_image_input:
                    vlm_success = True
                    vlm_error = None
                else:
                    if self.config.rate_limit and self.config.rate_limit > 0:
                        time.sleep(float(self.config.rate_limit))
                    ok2, err2, raw2 = _run_one(events_prompt, kind="events", validator=_events_validator)
                    vlm_success = bool(ok2)
                    vlm_error = err2 if not ok2 else None
                    if raw2 is not None:
                        raw_paths["events"] = raw2
        else:
            if is_image_input:
                all_prompt = self._single_output_prompt(
                    self.prompts.image_json,
                    output_name="image.json",
                    requirement="The JSON object should describe the still image and MUST NOT include an 'events' field.",
                )
                all_validator = None
            else:
                all_prompt = self._combined_output_prompt(
                    metadata_name="video.json",
                    metadata_prompt=self.prompts.video_json,
                    events_prompt=self.prompts.events_json,
                )
                all_validator = _combined_video_events_validator
            ok, err, raw = _run_one(all_prompt, kind="all", validator=all_validator)
            vlm_success = bool(ok)
            vlm_error = err
            if raw is not None:
                raw_paths["all"] = raw

        result["vlm_analysis"] = {
            "success": vlm_success,
            "processing_time_sec": time.time() - start_time,
            "error": vlm_error,
            "mode": "frame",
            "split_json_calls": split_mode,
            "structured_output_effective": effective_structured,
        }

        self.logger.info("Step 3/3: JSON Extraction")
        files_written = False
        fallback_used = False
        fallback_reasons: List[str] = []
        try:
            if is_image_input:
                if split_mode:
                    raw_v = raw_paths.get("image")
                    obj_v: Dict[str, Any] = {}
                    if raw_v is None:
                        fallback_used = True
                        fallback_reasons.append(vlm_error or "missing_raw_image_output")
                    else:
                        try:
                            obj_v = _extract_single_json_object(Path(raw_v).read_text(encoding="utf-8"))
                        except _EXPECTED_JSON_EXTRACTION_ERRORS as e:
                            fallback_used = True
                            fallback_reasons.append(f"image_json_parse_failed:{e}")
                            self.logger.error("Image JSON extraction failed; using DAFT fallback: %s", e)
                        except Exception as e:  # noqa: BLE001
                            # Keep the pipeline resilient to unexpected parser/client output failures.
                            fallback_used = True
                            fallback_reasons.append(f"image_json_parse_failed:{e}")
                            self.logger.exception("Unexpected image JSON extraction failure; using DAFT fallback")
                else:
                    raw_output = raw_paths.get("all") or (work_dir / "output.txt")
                    try:
                        obj_v = _extract_single_json_object(Path(raw_output).read_text(encoding="utf-8"))
                    except _EXPECTED_JSON_EXTRACTION_ERRORS as e:
                        fallback_used = True
                        fallback_reasons.append(f"image_single_call_json_parse_failed:{e}")
                        obj_v = {}
                        self.logger.error("Image VLM JSON extraction failed; using DAFT fallback: %s", e)
                    except Exception as e:  # noqa: BLE001
                        # Keep the pipeline resilient to unexpected parser/client output failures.
                        fallback_used = True
                        fallback_reasons.append(f"image_single_call_json_parse_failed:{e}")
                        obj_v = {}
                        self.logger.exception("Unexpected image VLM JSON extraction failure; using DAFT fallback")

                enriched_image = _enrich_image_with_probe(obj_v, video_path)
                daft_image = to_daft_image(enriched_image, image_id=get_scene_media_id())
                write_daft_json(paths.contextual_image, daft_image)
                files_written = True
            else:
                if split_mode:
                    raw_v = raw_paths.get("video")
                    raw_e = raw_paths.get("events")
                    obj_v = {}
                    obj_e = {"events": []}
                    if raw_v is None:
                        fallback_used = True
                        fallback_reasons.append("missing_raw_video_output")
                    else:
                        try:
                            obj_v = _extract_single_json_object(Path(raw_v).read_text(encoding="utf-8"))
                        except _EXPECTED_JSON_EXTRACTION_ERRORS as e:
                            fallback_used = True
                            fallback_reasons.append(f"video_json_parse_failed:{e}")
                            self.logger.error("video.json extraction failed; using DAFT fallback: %s", e)
                        except Exception as e:  # noqa: BLE001
                            # Keep the pipeline resilient to unexpected parser/client output failures.
                            fallback_used = True
                            fallback_reasons.append(f"video_json_parse_failed:{e}")
                            self.logger.exception("Unexpected video.json extraction failure; using DAFT fallback")
                    if raw_e is None:
                        fallback_used = True
                        fallback_reasons.append("missing_raw_events_output")
                    else:
                        try:
                            obj_e = _extract_single_json_object(Path(raw_e).read_text(encoding="utf-8"))
                            if "events" not in obj_e or not isinstance(obj_e.get("events"), list):
                                raise ValueError("events.json candidate missing top-level 'events' list")
                        except _EXPECTED_JSON_EXTRACTION_ERRORS as e:
                            fallback_used = True
                            fallback_reasons.append(f"events_json_parse_failed:{e}")
                            obj_e = {"events": []}
                            self.logger.error("events.json extraction failed; using empty-events fallback: %s", e)
                        except Exception as e:  # noqa: BLE001
                            # Keep the pipeline resilient to unexpected parser/client output failures.
                            fallback_used = True
                            fallback_reasons.append(f"events_json_parse_failed:{e}")
                            obj_e = {"events": []}
                            self.logger.exception(
                                "Unexpected events.json extraction failure; using empty-events fallback"
                            )
                else:
                    raw_output = raw_paths.get("all") or (work_dir / "output.txt")
                    try:
                        obj_v, obj_e = _extract_combined_video_events(
                            Path(raw_output).read_text(encoding="utf-8"),
                            logger=self.logger,
                        )
                        if "events" not in obj_e or not isinstance(obj_e.get("events"), list):
                            raise ValueError("events.json candidate missing top-level 'events' list")
                    except _EXPECTED_JSON_EXTRACTION_ERRORS as e:
                        fallback_used = True
                        fallback_reasons.append(f"single_call_json_parse_failed:{e}")
                        obj_v, obj_e = {}, {"events": []}
                        self.logger.error("VLM JSON extraction failed; using DAFT fallback: %s", e)
                    except Exception as e:  # noqa: BLE001
                        # Keep the pipeline resilient to unexpected parser/client output failures.
                        fallback_used = True
                        fallback_reasons.append(f"single_call_json_parse_failed:{e}")
                        obj_v, obj_e = {}, {"events": []}
                        self.logger.exception("Unexpected VLM JSON extraction failure; using DAFT fallback")

                enriched_video = _enrich_video_with_probe(obj_v, video_path, self.preprocessor)
                duration = float(enriched_video.get("duration") or 0.0) or None

                instances_keys = _read_instances_keys(paths.contextual_instances, self.logger)

                daft_video = to_daft_video(
                    enriched_video,
                    video_id=get_scene_media_id(),
                    instances_keys=instances_keys,
                )
                daft_events = to_daft_events(
                    obj_e,
                    video_id=get_scene_media_id(),
                    duration=duration,
                    instances_keys=instances_keys,
                    logger=self.logger,
                )

                write_daft_json(paths.contextual_video, daft_video)
                write_daft_json(paths.contextual_events, daft_events)
                files_written = True
        except Exception as e:
            self.logger.error("DAFT fallback/output writing failed: %s", e)
            files_written = False
            fallback_used = True
            fallback_reasons.append(f"daft_write_failed:{e}")

        result["json_extraction"] = {
            "success": bool(files_written and not fallback_used),
            "fallback_used": bool(fallback_used),
            "fallback_reasons": fallback_reasons,
            "scene_kind": "image" if is_image_input else "video",
            "contextual_json": str(paths.contextual_image if is_image_input else paths.contextual_video),
        }
        if not is_image_input:
            result["json_extraction"]["events_json"] = str(paths.contextual_events)
        if files_written and fallback_used:
            self.logger.warning(
                "VLM JSON fallback output written for %s scene: %s%s",
                result["json_extraction"]["scene_kind"],
                result["json_extraction"]["contextual_json"],
                f", {result['json_extraction']['events_json']}" if "events_json" in result["json_extraction"] else "",
            )
        result["success"] = bool(files_written and vlm_success and not fallback_used)

        try:
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)
        except Exception:
            pass

        return result

    def _write_fallback_outputs(
        self,
        *,
        video_path: Path,
        scene_dir: Path,
        is_image_input: bool,
        reason: str,
    ) -> Dict[str, Any]:
        """Write minimal DAFT-valid contextual outputs after VLM-stage failure.

        Runtime/parse failures should not leave downstream consumers with an
        empty scene directory when fallback policy permits continuing. Images get
        only `image.json`; videos get `video.json` plus an empty `events.json`
        whose event IDs are still grounded against any available instances.
        """
        paths = scene_paths(scene_dir)
        errors: List[str] = [reason] if reason else []
        try:
            if is_image_input:
                enriched_image = _enrich_image_with_probe({}, video_path)
                daft_image = to_daft_image(enriched_image, image_id=get_scene_media_id())
                write_daft_json(paths.contextual_image, daft_image)
                return {
                    "success": False,
                    "fallback_used": True,
                    "fallback_reasons": errors,
                    "scene_kind": "image",
                    "contextual_json": str(paths.contextual_image),
                }

            enriched_video = _enrich_video_with_probe({}, video_path, self.preprocessor)
            duration = float(enriched_video.get("duration") or 0.0) or None
            instances_keys = _read_instances_keys(paths.contextual_instances, self.logger)
            write_daft_json(
                paths.contextual_video,
                to_daft_video(
                    enriched_video,
                    video_id=get_scene_media_id(),
                    instances_keys=instances_keys,
                ),
            )
            write_daft_json(
                paths.contextual_events,
                to_daft_events(
                    {"events": []},
                    video_id=get_scene_media_id(),
                    duration=duration,
                    instances_keys=instances_keys,
                    logger=self.logger,
                ),
            )
            return {
                "success": False,
                "fallback_used": True,
                "fallback_reasons": errors,
                "scene_kind": "video",
                "contextual_json": str(paths.contextual_video),
                "events_json": str(paths.contextual_events),
            }
        except Exception as e:
            self.logger.error("Failed to write DAFT fallback outputs: %s", e)
            errors.append(f"fallback_write_failed:{e}")
            return {
                "success": False,
                "fallback_used": True,
                "fallback_reasons": errors,
                "scene_kind": "image" if is_image_input else "video",
                "contextual_json": str(paths.contextual_image if is_image_input else paths.contextual_video),
                **({} if is_image_input else {"events_json": str(paths.contextual_events)}),
            }


def _enrich_video_with_probe(
    obj_v: Dict[str, Any],
    video_path: Path,
    preprocessor: "VideoPreprocessor",
) -> Dict[str, Any]:
    """Merge VLM-emitted fields with media-probe data required by DAFT.

    The VLM carries scene-level text fields (``scene_description``,
    ``event_summary``, etc.); the media container carries ``format``, ``fps``,
    ``duration``, ``height``, ``width``. Probe data overrides any VLM guesses
    for the latter — the converter rejects missing / out-of-range values.
    """
    out = dict(obj_v or {})
    probe = preprocessor.get_video_info(video_path)
    out["format"] = video_path.suffix.lstrip(".").lower() or "mp4"
    out["fps"] = probe.get("fps")
    out["duration"] = probe.get("duration")
    out["height"] = probe.get("height")
    out["width"] = probe.get("width")
    return out


def _enrich_image_with_probe(obj_v: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
    """Image-input counterpart to :func:`_enrich_video_with_probe`.

    DAFT image.json wants ``format`` + ``height`` + ``width`` (no fps/duration)
    and a single optional ``caption`` instead of the video schema's
    ``scene_description`` / ``event_summary`` pair.

    The image scene prompt asks the VLM to emit ``caption`` directly, so the
    typical path is just dimension + format enrichment. The
    ``scene_description`` / ``event_summary`` collapse remains as a defensive
    fallback in case the VLM leaks the older video-style shape.
    """
    out = dict(obj_v or {})
    try:
        with Image.open(image_path) as im:
            width, height = im.size
    except Exception:
        width = height = 0
    out["height"] = height
    out["width"] = width

    out["format"] = image_path.suffix.lstrip(".").lower() or "jpg"

    if "caption" not in out:
        caption = out.get("scene_description") or out.get("event_summary")
        if caption:
            out["caption"] = caption
    out.pop("scene_description", None)
    out.pop("event_summary", None)
    return out


def _read_instances_keys(instances_path: Path, logger: logging.Logger) -> List[str]:
    """Return the keys of the scene's ``instances.json`` ``"instances"`` dict.

    Used to ground the VLM-emitted ``id_<n>`` event references against the
    tracker's ``<class>_<n>`` catalogue. Returns an empty list when the file
    is absent or unreadable: the caller passes this to ``to_daft_events``,
    which then routes through the translator's ``NO_INSTANCES`` branch and
    drops the (necessarily ungrounded) VLM IDs. We always return a list
    (never ``None``) so the caller's contract with ``to_daft_events`` is
    "always translate"; ``None`` would silently disable translation, which
    is the wrong default for the live pipeline.
    """
    if not instances_path.exists():
        return []
    try:
        with instances_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(
            "[id_translator] failed to read %s: %s; treating as empty",
            instances_path,
            e,
        )
        return []
    inst = data.get("instances") if isinstance(data, dict) else None
    if not isinstance(inst, dict):
        return []
    return list(inst.keys())
