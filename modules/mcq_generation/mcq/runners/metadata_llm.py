# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-only MCQ generation from a precomputed ``metadata.json``.

Consumes an existing window-captioned ``metadata.json`` (produced by a prior
``window-*`` run) and asks the LLM to answer the question bank per window.
Pipeline schema disables VLM verification for ``metadata-llm``; the low-level
runner still accepts optional verify settings for direct compatibility.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from al_utils.io import sha256_text, write_json, write_text
from daft_export.common import get_scene_media_id, write_daft_json
from daft_export.paths import scene_paths
from daft_export.task import to_daft_tasks
from mcq_generation.mcq.utils.aggregation import aggregate_window_mcqs, aggregation_specs_from_bank
from mcq_generation.mcq.utils.bank import (
    collect_embedded_bank_from_prompt,
    filter_mcq_items_strict,
    include_if_map_from_bank,
    options_map_from_bank,
    read_bank,
)
from mcq_generation.mcq.utils.ids import fmt_ids as _fmt_ids
from mcq_generation.mcq.utils.ids import present_ids as _present_ids
from mcq_generation.mcq.utils.logging_utils import setup_runner_logger
from mcq_generation.mcq.utils.openai import (
    call_chat_json_with_structured_fallback,
    classify_mcq_json_parse_failure,
    extract_json_object_from_llm_text,
    get_llm_api_key,
)
from mcq_generation.mcq.utils.retry_missing import (
    expected_question_ids,
    known_answers_for_retry,
    retry_fill_missing_questions,
)
from mcq_generation.mcq.utils.validate import (
    DEFAULT_CAPTION_KEY,
    DEFAULT_ENHANCED_CAPTION_KEY,
    is_enhanced_mcq_payload,
    outputs_complete,
)
from mcq_generation.mcq.utils.video import extract_frames, probe_video
from mcq_generation.mcq.utils.vlm import vlm_messages_from_frames
from mcq_generation.mcq.utils.vlm_verify import (
    attach_reasoning_traces_from_verify,
    build_vlm_verify_prompt,
    run_window_vlm_verify,
)


def _setup_logger(verbose: bool) -> logging.Logger:
    return setup_runner_logger("metadata_llm", verbose)


def _derive_video_id(metadata_path: Path) -> str:
    # Prefer the semantic clip id in the metadata content (stable across staging),
    # then fall back to filesystem-based heuristics.
    try:
        obj = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            vid = str(obj.get("video_id") or "").strip()
            if vid:
                return vid
    except Exception:
        pass

    # Fallback: derive a stable-enough id from the staged metadata location
    # (usually <scene>/sidecars/metadata.json).
    return metadata_path.parent.name or metadata_path.stem


@dataclass
class MetadataLlmRunner:
    mcq_prompt: str
    llm_base_url: str
    llm_model: str
    question_bank_file: Optional[Path] = None
    caption_key: str = DEFAULT_CAPTION_KEY
    enhanced_caption_key: str = DEFAULT_ENHANCED_CAPTION_KEY
    llm_retries: int = 3
    llm_retry_backoff_s: float = 5.0
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.2
    llm_structured_output: str = "openai"
    timeout: int = 600
    rate_limit: float = 0.0
    aggregate_windows: bool = False
    write_empty_mcq_marker: bool = True
    skip_existing: bool = False
    seed: Optional[int] = None
    retry_missing_questions: bool = False
    retry_missing_max_rounds: int = 2

    # Optional per-window VLM verify (frames-based) on finalized per-window MCQ.
    # This allows "metadata-llm + correction" without re-running caption generation.
    vlm_verify_enabled: bool = False
    vlm_verify_apply_corrections: bool = False
    vlm_base_url: str = ""
    vlm_model: str = ""
    vlm_retries: int = 3
    vlm_retry_backoff_s: float = 5.0
    vlm_verify_max_tokens: int = 8192
    vlm_verify_temperature: float = 0.0
    vlm_verify_structured_output: str = "openai"
    vlm_verify_prompt_template: str = ""
    # Frame sampling for verify:
    # If None, inherit from input metadata.windowing when available.
    verify_sampling_fps: Optional[float] = None
    verify_resolution: Optional[int] = None
    verify_max_frames: Optional[int] = None
    # Required when vlm_verify_enabled: source video for frame sampling
    input_video_path: Optional[Path] = None

    def persist_prompts(self, *, output_root: Path, mcq_prompt_file: Optional[Path] = None) -> None:
        prompts_dir = Path(output_root) / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        write_text(
            prompts_dir / "mcq_prompt.used.md",
            self.mcq_prompt + ("\n" if not self.mcq_prompt.endswith("\n") else ""),
        )

        meta: Dict[str, Any] = {
            "mcq_prompt_file": str(mcq_prompt_file) if mcq_prompt_file is not None else "",
            "mcq_prompt_sha256": sha256_text(self.mcq_prompt),
        }

        write_json(prompts_dir / "prompts.used.json", meta)

    def run_single(
        self,
        *,
        input_metadata_json: Path,
        output_dir: Path,
        video_id_override: str = "",
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        logger = logger or _setup_logger(verbose)
        meta = Path(input_metadata_json)
        if not meta.exists():
            raise SystemExit(f"Input does not exist: {meta}")
        if not meta.is_file():
            raise SystemExit(f"Input must be a file: {meta}")

        if self.skip_existing and outputs_complete(
            out_dir=Path(output_dir),
            direct_mcq_from_vlm=False,
            caption_key=self.caption_key,
            enhanced_caption_key=self.enhanced_caption_key,
            require_vlm_verify=bool(self.vlm_verify_enabled),
        ):
            logger.info("Skipping existing complete metadata-llm output: %s", output_dir)
            return None

        self._build_from_metadata(
            metadata_path=meta,
            output_dir=Path(output_dir),
            video_id_override=str(video_id_override or ""),
            logger=logger,
        )

    def _aggregate_mcqs(
        self,
        window_mcqs: List[Dict[str, Any]],
        *,
        video_id: str,
        include_if_map: Dict[str, Dict[str, str]],
        aggregation_specs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        return aggregate_window_mcqs(
            window_mcqs,
            video_id=video_id,
            include_if_map=include_if_map,
            aggregation_specs=aggregation_specs,
        )

    def _call_one(
        self, *, prompt_text: str, caption: str, logger: logging.Logger
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        llm_messages = [{"role": "system", "content": prompt_text}, {"role": "user", "content": caption}]
        obj, raw = call_chat_json_with_structured_fallback(
            base_url=self.llm_base_url,
            model=self.llm_model,
            messages=llm_messages,
            timeout=self.timeout,
            max_tokens=self.llm_max_tokens,
            temperature=float(self.llm_temperature),
            top_p=0.9,
            logger=logger,
            seed=self.seed,
            retries=int(self.llm_retries or 0),
            retry_backoff_s=float(self.llm_retry_backoff_s or 5.0),
            structured_output=str(self.llm_structured_output or "openai"),
            retry_stage="metadata_llm:llm_mcq",
            api_key=get_llm_api_key(),
        )
        return (obj if isinstance(obj, dict) else None), raw

    def _verify_window_mcq(
        self,
        *,
        frames: List[Tuple[float, Path]],
        fallback_mcq: List[Dict[str, Any]],
        mcq_obj: Dict[str, Any],
        video_id: str,
        w_idx: int,
        logger: logging.Logger,
        win_errors: List[str],
    ) -> Dict[str, Any]:
        verify_prompt, verify_expected_ids = build_vlm_verify_prompt(
            fallback_mcq,
            prompt_template=self.vlm_verify_prompt_template,
            apply_corrections=bool(self.vlm_verify_apply_corrections),
        )
        if not verify_expected_ids:
            return {
                "corrected_obj": dict(mcq_obj),
                "verify_items": [],
                "corrected_count": 0,
                "verify_status": "skipped_no_verifiable_items",
            }
        return run_window_vlm_verify(
            messages=vlm_messages_from_frames(frames, verify_prompt),
            fallback_mcq=fallback_mcq,
            verify_expected_ids=verify_expected_ids,
            mcq_obj=mcq_obj,
            video_id=video_id,
            w_idx=w_idx,
            vlm_base_url=str(self.vlm_base_url).strip(),
            vlm_model=str(self.vlm_model).strip(),
            timeout=self.timeout,
            max_tokens=int(self.vlm_verify_max_tokens),
            temperature=float(self.vlm_verify_temperature),
            structured_output=str(self.vlm_verify_structured_output or "openai"),
            apply_corrections=bool(self.vlm_verify_apply_corrections),
            retries=int(self.vlm_retries or 0),
            retry_backoff_s=float(self.vlm_retry_backoff_s or 5.0),
            retry_stage="metadata_llm:vlm_verify",
            logger=logger,
            win_errors=win_errors,
        )

    def _build_from_metadata(
        self,
        *,
        metadata_path: Path,
        output_dir: Path,
        video_id_override: str = "",
        logger: logging.Logger,
    ) -> None:
        video_id = str(video_id_override or "").strip() or _derive_video_id(metadata_path)

        paths = scene_paths(output_dir)
        paths.sidecars_dir.mkdir(parents=True, exist_ok=True)
        sidecar_metadata = paths.sidecars_dir / "metadata.json"
        sidecar_verify = paths.sidecars_dir / "mcq.vlm_verify.json"
        sidecar_empty = paths.sidecars_dir / "mcq.empty.json"
        for stale_path in (paths.task_mcq, paths.task_bcq, paths.task_open_qa, sidecar_verify, sidecar_empty):
            stale_path.unlink(missing_ok=True)

        src = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        windows = list(src.get("windows") or [])
        if not windows:
            logger.warning("No windows found in %s; skipping", metadata_path)
            return None

        # Best-effort: inherit frame sampling settings from the source metadata if present,
        # so downstream verify uses the same sampling scheme unless explicitly overridden.
        if isinstance(src.get("windowing"), dict):
            wcfg = src.get("windowing") or {}
            if self.verify_sampling_fps is None and isinstance(wcfg.get("sampling_fps"), (int, float)):
                self.verify_sampling_fps = float(wcfg.get("sampling_fps") or 0.0)
            if self.verify_resolution is None and isinstance(wcfg.get("resolution"), int):
                self.verify_resolution = int(wcfg.get("resolution") or 0)
            if self.verify_max_frames is None and isinstance(wcfg.get("max_frames"), int):
                self.verify_max_frames = int(wcfg.get("max_frames") or 0)

        cap_key = str(self.caption_key or "").strip() or DEFAULT_CAPTION_KEY
        enh_key = str(self.enhanced_caption_key or "").strip() or DEFAULT_ENHANCED_CAPTION_KEY

        # Persist a reproducibility hint for how windows/frames were sampled.
        # - If the input metadata already contains windowing, keep it.
        # - Otherwise, infer window size from windows[*].start_frame/end_frame (or start_sec/end_sec) best-effort.
        windowing_obj: Dict[str, Any] = (
            dict(src.get("windowing") or {}) if isinstance(src.get("windowing"), dict) else {}
        )
        if not windowing_obj:
            fps0 = float(src.get("framerate") or 0.0)
            # Prefer frame windows (this repo's default for window modes).
            lens: List[int] = []
            for w in windows:
                if not isinstance(w, dict):
                    continue
                sf = w.get("start_frame")
                ef = w.get("end_frame")
                if isinstance(sf, int) and isinstance(ef, int) and ef >= sf:
                    lens.append(int(ef - sf + 1))
            window_frames = lens[0] if lens and all(x == lens[0] for x in lens) else 0
            window_seconds = (float(window_frames) / fps0) if (window_frames > 0 and fps0 > 0) else 0.0
            mode = "frames" if window_frames > 0 else "unknown"

            # Fallback: try seconds windows if no frame windows are present.
            if mode == "unknown":
                spans: List[float] = []
                for w in windows:
                    if not isinstance(w, dict):
                        continue
                    ss = w.get("start_sec")
                    es = w.get("end_sec")
                    if isinstance(ss, (int, float)) and isinstance(es, (int, float)) and float(es) >= float(ss):
                        spans.append(float(es) - float(ss))
                if spans and all(abs(x - spans[0]) < 1e-3 for x in spans):
                    mode = "seconds"
                    window_seconds = float(spans[0])

            windowing_obj = {
                "window_mode": mode,
                "window_frames": int(window_frames),
                "window_seconds": float(window_seconds),
                # These describe the frame sampling strategy used for downstream VLM verify
                # (and should match the window runner when available).
                "sampling_fps": float(self.verify_sampling_fps or 2.0),
                "resolution": int(self.verify_resolution or 480),
                "max_frames": int(self.verify_max_frames or 100),
                "caption_key": str(cap_key),
                "enhanced_caption_key": str(enh_key),
            }
        else:
            # Ensure these are present for downstream tooling.
            windowing_obj.setdefault("caption_key", str(cap_key))
            windowing_obj.setdefault("enhanced_caption_key", str(enh_key))
            if self.verify_sampling_fps is not None:
                windowing_obj["sampling_fps"] = float(self.verify_sampling_fps)
            if self.verify_resolution is not None:
                windowing_obj["resolution"] = int(self.verify_resolution)
            if self.verify_max_frames is not None:
                windowing_obj["max_frames"] = int(self.verify_max_frames)
        src["windowing"] = windowing_obj

        eff_verify_sampling_fps = float(windowing_obj.get("sampling_fps") or 2.0)
        eff_verify_resolution = int(windowing_obj.get("resolution") or 480)
        eff_verify_max_frames = int(windowing_obj.get("max_frames") or 100)

        # If we can parse embedded bank, enforce strict include_if + answer validity on outputs.
        bank_single: Optional[Dict[str, Any]] = None
        if self.question_bank_file is not None:
            try:
                bank_single = read_bank(Path(self.question_bank_file))
            except Exception:
                bank_single = None
        if bank_single is None:
            bank_single = collect_embedded_bank_from_prompt(self.mcq_prompt)
        include_if_map = include_if_map_from_bank(bank_single) if isinstance(bank_single, dict) else {}
        options_map = options_map_from_bank(bank_single) if isinstance(bank_single, dict) else {}
        aggregation_specs = aggregation_specs_from_bank(bank_single) if isinstance(bank_single, dict) else {}

        window_mcq_objs: List[Dict[str, Any]] = []
        verify_windows_out: List[Dict[str, Any]] = []
        window_mcq_corrected_objs: List[Dict[str, Any]] = []
        corrected_total = 0

        video_info = None
        tmp_verify_root: Path | None = None
        if bool(self.vlm_verify_enabled):
            if self.input_video_path is None:
                raise SystemExit("vlm_verify_enabled=true but input_video_path is not set (need video for frames)")
            if not str(self.vlm_base_url or "").strip() or not str(self.vlm_model or "").strip():
                raise SystemExit("vlm_verify_enabled=true but VLM endpoint config is missing (vlm_base_url/vlm_model)")
            video_info = probe_video(Path(self.input_video_path))
            # Frame scratch sits under the scene's unified sidecar work area
            # and gets wiped on both entry and exit.
            tmp_verify_root = Path(output_dir) / "sidecars" / "_work" / "metadata_llm_verify"
            if tmp_verify_root.exists():
                shutil.rmtree(tmp_verify_root, ignore_errors=True)
            tmp_verify_root.mkdir(parents=True, exist_ok=True)

        try:
            for w_idx, w in enumerate(windows):
                if not isinstance(w, dict):
                    continue
                caption = str(w.get(cap_key, "") or "").strip()
                if not caption:
                    w[enh_key] = {}
                    continue

                mcq_obj, mcq_text = self._call_one(prompt_text=self.mcq_prompt, caption=caption, logger=logger)

                if bool(self.retry_missing_questions):
                    base_obj = mcq_obj if isinstance(mcq_obj, dict) else extract_json_object_from_llm_text(mcq_text)
                    if isinstance(bank_single, dict):
                        retry_started_from_empty = not is_enhanced_mcq_payload(base_obj)
                        cur_obj: Dict[str, Any] = (
                            dict(base_obj)
                            if is_enhanced_mcq_payload(base_obj)
                            else {"version": 2.0, "video_id": video_id, "mcq": []}
                        )
                        sf = w.get("start_frame")
                        ef = w.get("end_frame")
                        frame_s = int(sf) if isinstance(sf, int) else -1
                        frame_e = int(ef) if isinstance(ef, int) else -1
                        for attempt in range(max(1, int(self.retry_missing_max_rounds or 1))):
                            present = set(_present_ids(cur_obj))
                            required_missing = [
                                qid
                                for qid in expected_question_ids(
                                    bank=bank_single,
                                    include_if_map=include_if_map,
                                    current_mcq_obj=cur_obj,
                                )
                                if qid not in present
                            ]
                            if not required_missing:
                                break

                            known = known_answers_for_retry(
                                cur_obj,
                                include_if_map=include_if_map,
                                target_ids=required_missing,
                            )
                            logger.info(
                                "Retry required questions (video=%s window=%d frames=%d-%d attempt=%d missing=%d ids=%s known=%s)",
                                video_id,
                                w_idx,
                                frame_s,
                                frame_e,
                                attempt + 1,
                                len(required_missing),
                                _fmt_ids(required_missing),
                                sorted(known.keys()),
                            )
                            retried = retry_fill_missing_questions(
                                bank=bank_single,
                                include_if_map=include_if_map,
                                base_prompt=self.mcq_prompt,
                                caption=caption,
                                current_mcq_obj=cur_obj,
                                target_ids=required_missing,
                                known_answers=known,
                                video_id=video_id,
                                llm_base_url=self.llm_base_url,
                                llm_model=self.llm_model,
                                llm_structured_output=str(self.llm_structured_output or "openai"),
                                llm_max_tokens=int(self.llm_max_tokens),
                                llm_temperature=float(self.llm_temperature),
                                timeout=int(self.timeout),
                                llm_retries=int(self.llm_retries or 0),
                                llm_retry_backoff_s=float(self.llm_retry_backoff_s or 5.0),
                                retry_stage="metadata_llm:llm_mcq:retry_required",
                                max_rounds=1,
                                logger=logger,
                            )
                            if not isinstance(retried, dict):
                                break
                            filled = sorted(set(_present_ids(retried)) - present)
                            cur_obj = retried
                            remain = [
                                qid
                                for qid in expected_question_ids(
                                    bank=bank_single,
                                    include_if_map=include_if_map,
                                    current_mcq_obj=cur_obj,
                                )
                                if qid not in set(_present_ids(cur_obj))
                            ]
                            logger.info(
                                "Retry required done (video=%s window=%d frames=%d-%d filled=%d ids=%s required_missing=%d)",
                                video_id,
                                w_idx,
                                frame_s,
                                frame_e,
                                len(filled),
                                _fmt_ids(filled),
                                len(remain),
                            )
                            if not filled:
                                break
                        has_retry_items = bool(_present_ids(cur_obj))
                        if not retry_started_from_empty or has_retry_items:
                            mcq_obj = cur_obj

                if isinstance(mcq_obj, dict):
                    mcq_obj["video_id"] = video_id
                    if "version" not in mcq_obj:
                        mcq_obj["version"] = 2.0
                    mcq_items = mcq_obj.get("mcq") or []
                    if isinstance(mcq_items, list) and (include_if_map or options_map):
                        mcq_obj["mcq"] = filter_mcq_items_strict(
                            [it for it in mcq_items if isinstance(it, dict)],
                            include_if_map=include_if_map,
                            options_map=options_map,
                        )
                    w[enh_key] = mcq_obj
                    window_mcq_objs.append(mcq_obj)

                    # Optional per-window VLM verify/correction using frames from the input video.
                    if bool(self.vlm_verify_enabled) and self.input_video_path is not None and video_info is not None:
                        # Resolve the window time range.
                        start_sec = 0.0
                        end_sec = float(video_info.duration_sec)
                        if isinstance(w.get("start_sec"), (int, float)) and isinstance(w.get("end_sec"), (int, float)):
                            start_sec = float(w.get("start_sec") or 0.0)
                            end_sec = float(w.get("end_sec") or end_sec)
                        elif isinstance(w.get("start_frame"), int) and isinstance(w.get("end_frame"), int):
                            fps = float(video_info.fps or 30.0)
                            sf = int(w.get("start_frame") or 0)
                            ef = int(w.get("end_frame") or 0)
                            start_sec = max(0.0, sf / fps)
                            # inclusive end_frame -> end time is (ef+1)/fps
                            end_sec = max(start_sec, (ef + 1) / fps)
                        # Clamp to duration
                        start_sec = max(0.0, min(float(video_info.duration_sec), float(start_sec)))
                        end_sec = max(start_sec, min(float(video_info.duration_sec), float(end_sec)))

                        # ``tmp_verify_root`` is guaranteed set: the enclosing
                        # ``vlm_verify_enabled`` guard initializes it above.
                        assert tmp_verify_root is not None
                        win_dir = tmp_verify_root / f"win_{w_idx:03d}"
                        frames = extract_frames(
                            video_path=Path(self.input_video_path),
                            out_dir=win_dir,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            sampling_fps=float(eff_verify_sampling_fps),
                            resolution=int(eff_verify_resolution),
                            max_frames=int(eff_verify_max_frames),
                            logger=logger,
                        )

                        mcq_items2 = [it for it in (mcq_obj.get("mcq") or []) if isinstance(it, dict)]
                        corrected_obj: Dict[str, Any]
                        if frames and mcq_items2:
                            win_errors_verify: List[str] = []
                            verify_result = self._verify_window_mcq(
                                frames=frames,
                                fallback_mcq=mcq_items2,
                                mcq_obj=mcq_obj,
                                video_id=video_id,
                                w_idx=w_idx,
                                logger=logger,
                                win_errors=win_errors_verify,
                            )
                            corrected_obj = verify_result["corrected_obj"]
                            if verify_result["verify_status"] == "skipped_no_verifiable_items":
                                win_verify_obj = None
                            else:
                                corrected_total += verify_result["corrected_count"]
                                win_verify_obj = {
                                    "status": verify_result["verify_status"],
                                    "verifications": verify_result["verify_items"],
                                    "corrected_count": verify_result["corrected_count"],
                                }
                                if "error" in verify_result:
                                    win_verify_obj["error"] = verify_result["error"]
                        else:
                            corrected_obj = dict(mcq_obj)
                            win_verify_obj = {
                                "status": "no_frames_or_empty_mcq",
                                "verifications": [],
                                "corrected_count": 0,
                            }

                        if win_verify_obj is not None:
                            w["vlm_verify"] = win_verify_obj
                            verify_windows_out.append(
                                {
                                    "window_index": int(w_idx),
                                    "start_frame": int(w.get("start_frame") or 0)
                                    if isinstance(w.get("start_frame"), int)
                                    else 0,
                                    "end_frame": int(w.get("end_frame") or 0)
                                    if isinstance(w.get("end_frame"), int)
                                    else 0,
                                    "vlm_verify": win_verify_obj,
                                }
                            )
                        window_mcq_corrected_objs.append(dict(corrected_obj))
                else:
                    reason = classify_mcq_json_parse_failure(mcq_text)
                    errors = w.get("_errors")
                    if not isinstance(errors, list):
                        errors = []
                        w["_errors"] = errors
                    errors.append(f"llm_mcq_not_parseable:{reason}")
                    logger.warning(
                        "metadata-llm: LLM MCQ output not parseable (video=%s window=%d reason=%s has_output=%s output_chars=%d)",
                        video_id,
                        w_idx,
                        reason,
                        bool(str(mcq_text or "").strip()),
                        len(str(mcq_text or "")),
                    )
                    w[enh_key] = {}

                if self.rate_limit and self.rate_limit > 0:
                    time.sleep(float(self.rate_limit))
        finally:
            if tmp_verify_root is not None:
                shutil.rmtree(tmp_verify_root, ignore_errors=True)

        src["video_id"] = video_id
        src["source_video"] = str(src.get("source_video") or "")

        src["windows"] = windows
        src["has_caption"] = True
        src["valid"] = True
        write_json(sidecar_metadata, src, sanitize_paths=True)
        enhanced_objects = sum(1 for w in windows if isinstance(w, dict) and is_enhanced_mcq_payload(w.get(enh_key)))
        logger.info(
            "metadata-llm: metadata sidecar written (windows=%d enhanced_payload_objects=%d verify_enabled=%s)",
            len(windows),
            enhanced_objects,
            bool(self.vlm_verify_enabled),
        )

        output_mcq_objs = window_mcq_corrected_objs if window_mcq_corrected_objs else window_mcq_objs
        mcq_written = False
        task_composed = False
        if output_mcq_objs:
            out_mcq = (
                self._aggregate_mcqs(
                    output_mcq_objs,
                    video_id=video_id,
                    include_if_map=include_if_map,
                    aggregation_specs=aggregation_specs,
                )
                if self.aggregate_windows
                else dict(output_mcq_objs[0])
            )
            out_mcq["video_id"] = video_id
            if bool(self.vlm_verify_enabled) and verify_windows_out:
                attach_reasoning_traces_from_verify(out_mcq, verify_windows_out=verify_windows_out)

            items = list(out_mcq.get("mcq", []))
            task_composed = True
            mcq_payload, bcq_payload, open_qa_payload = to_daft_tasks(items, video_id=get_scene_media_id())
            if mcq_payload is not None:
                write_daft_json(paths.task_mcq, mcq_payload)
                mcq_written = True
            if bcq_payload is not None:
                write_daft_json(paths.task_bcq, bcq_payload)
                mcq_written = True
            if open_qa_payload is not None:
                write_daft_json(paths.task_open_qa, open_qa_payload)
                mcq_written = True

        if not mcq_written:
            if task_composed and self.write_empty_mcq_marker:
                write_json(
                    sidecar_empty,
                    {"version": 2.0, "video_id": video_id, "mcq": [], "_error": "zero_task_items"},
                )
            else:
                logger.warning("No DAFT task items for video=%s; skipping task output", video_id)
        else:
            sidecar_empty.unlink(missing_ok=True)

        if bool(self.vlm_verify_enabled) and mcq_written and verify_windows_out:
            verify_items_total = 0
            for w in verify_windows_out:
                vobj = w.get("vlm_verify")
                if isinstance(vobj, dict) and str(vobj.get("status") or "") == "ok":
                    verify_items_total += len(vobj.get("verifications") or [])
            verify_obj = {
                "version": 1.0,
                "video_id": video_id,
                "source_video": str(self.input_video_path or src.get("source_video") or ""),
                "windows": verify_windows_out,
                "summary": {
                    "windows_total": len(windows),
                    "windows_with_verify": len(verify_windows_out),
                    "questions_verified": verify_items_total,
                    "questions_corrected": int(corrected_total),
                },
            }
            write_json(sidecar_verify, verify_obj, sanitize_paths=True)
