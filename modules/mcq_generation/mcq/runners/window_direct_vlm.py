# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-window direct-VLM MCQ generation.

For each window of ``<clip_path>`` the runner samples frames and asks the VLM
to produce MCQ/BCQ task items in a single call (no separate LLM step). Items
are aggregated across windows, handed to the task converter for the final
output files, and companion artifacts (window metadata, verifier sidecars)
are written alongside.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from al_utils.io import write_json
from daft_export.common import get_scene_media_id, write_daft_json
from daft_export.paths import scene_paths
from daft_export.task import to_daft_tasks
from mcq_generation.mcq.utils.aggregation import aggregate_window_mcqs
from mcq_generation.mcq.utils.bank import filter_mcq_items_strict, options_map_from_bank
from mcq_generation.mcq.utils.ids import fmt_ids, present_ids
from mcq_generation.mcq.utils.layout import derive_video_id
from mcq_generation.mcq.utils.logging_utils import setup_runner_logger
from mcq_generation.mcq.utils.openai import (
    call_chat_json_with_structured_fallback,
    classify_mcq_json_parse_failure,
    extract_json_object_from_llm_text,
    get_vlm_api_key,
)
from mcq_generation.mcq.utils.retry_missing import (
    bank_question_map,
    build_retry_system_prompt,
    expected_question_ids,
    known_answers_for_retry,
)
from mcq_generation.mcq.utils.validate import (
    DEFAULT_CAPTION_KEY,
    DEFAULT_ENHANCED_CAPTION_KEY,
    is_enhanced_mcq_payload,
    outputs_complete,
)
from mcq_generation.mcq.utils.video import extract_frames, iter_windows, iter_windows_by_frames, probe_video
from mcq_generation.mcq.utils.vlm import vlm_direct_mcq_messages_from_frames
from mcq_generation.mcq.utils.vlm_verify import (
    attach_reasoning_traces_from_verify,
    build_vlm_verify_prompt,
    run_window_vlm_verify,
)


def setup_logger(verbose: bool) -> logging.Logger:
    return setup_runner_logger("window_direct_vlm", verbose)


def _missing_required_ids(
    *, bank: Dict[str, Any], include_if_map: Dict[str, Dict[str, str]], mcq_obj: Dict[str, Any]
) -> List[str]:
    expected = expected_question_ids(bank=bank, include_if_map=include_if_map, current_mcq_obj=mcq_obj)
    present = set(present_ids(mcq_obj))
    return [qid for qid in expected if qid not in present]


@dataclass
class WindowDirectVlmRunner:
    mcq_prompt: str
    vlm_base_url: str
    vlm_model: str
    include_if_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    aggregation_specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    window_seconds: float = 4.0
    window_frames: int = 0
    # If true, force a single window covering the whole media sample.
    # Frame sampling still respects sampling_fps/resolution/max_frames.
    single_window: bool = False
    sampling_fps: float = 2.0
    resolution: int = 480
    max_frames: int = 100
    vlm_max_tokens: int = 8192
    vlm_temperature: float = 0.3
    vlm_structured_output: str = "openai"
    vlm_retries: int = 3
    vlm_retry_backoff_s: float = 5.0
    timeout: int = 600
    rate_limit: float = 0.0
    aggregate_windows: bool = True
    video_id_override: str = ""
    caption_key: str = DEFAULT_CAPTION_KEY
    enhanced_caption_key: str = DEFAULT_ENHANCED_CAPTION_KEY
    write_empty_mcq_marker: bool = False
    # Retry missing required question ids (best-effort, per-window) by re-asking the VLM.
    retry_missing_questions: bool = False
    retry_missing_max_rounds: int = 2
    # Optional: keep a copy of the question bank for retry prompt subsets / required-id computation.
    question_bank: Optional[Dict[str, Any]] = None
    # Optional VLM verify pass after per-window MCQ finalize.
    vlm_verify_enabled: bool = False
    vlm_verify_max_tokens: int = 8192
    vlm_verify_temperature: float = 0.0
    vlm_verify_structured_output: str = "openai"
    vlm_verify_apply_corrections: bool = False
    vlm_verify_prompt_template: str = ""

    def _aggregate_mcqs(self, window_mcqs: List[Dict[str, Any]], *, video_id: str) -> Dict[str, Any]:
        return aggregate_window_mcqs(
            window_mcqs,
            video_id=video_id,
            include_if_map=self.include_if_map,
            aggregation_specs=self.aggregation_specs,
        )

    def _verify_window_mcq(
        self,
        *,
        frames: List[Path],
        mcq_obj: Dict[str, Any],
        video_id: str,
        w_idx: int,
        logger: logging.Logger,
        win_errors: List[str],
    ) -> Dict[str, Any]:
        fallback_mcq = list(mcq_obj.get("mcq") or [])
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
            messages=vlm_direct_mcq_messages_from_frames(frames, verify_prompt),
            fallback_mcq=fallback_mcq,
            verify_expected_ids=verify_expected_ids,
            mcq_obj=mcq_obj,
            video_id=video_id,
            w_idx=w_idx,
            vlm_base_url=self.vlm_base_url,
            vlm_model=self.vlm_model,
            timeout=self.timeout,
            max_tokens=int(self.vlm_verify_max_tokens),
            temperature=float(self.vlm_verify_temperature),
            structured_output=str(self.vlm_verify_structured_output or "openai"),
            apply_corrections=bool(self.vlm_verify_apply_corrections),
            retries=int(self.vlm_retries or 0),
            retry_backoff_s=float(self.vlm_retry_backoff_s or 5.0),
            retry_stage="window_direct_vlm:vlm_verify",
            logger=logger,
            win_errors=win_errors,
        )

    def build_for_clip(
        self,
        *,
        clip_path: Path,
        input_root: Path,
        output_root: Path,
        output_dir: Path,
        logger: logging.Logger,
    ) -> None:
        video_id = str(self.video_id_override or "").strip() or derive_video_id(input_root, clip_path)

        paths = scene_paths(output_dir)
        paths.sidecars_dir.mkdir(parents=True, exist_ok=True)
        sidecar_metadata = paths.sidecars_dir / "metadata.json"
        sidecar_verify = paths.sidecars_dir / "mcq.vlm_verify.json"
        sidecar_empty = paths.sidecars_dir / "mcq.empty.json"
        for stale_path in (paths.task_mcq, paths.task_bcq, paths.task_open_qa, sidecar_verify, sidecar_empty):
            stale_path.unlink(missing_ok=True)

        # Ephemeral per-video workspace (frame extraction + caption tmp files).
        # ``output_root`` is the scene's sidecars/_work/<stage> dir; the wrapper
        # cleans it up post-run.
        work_dir = Path(output_root) / video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        cap_key = str(self.caption_key or "").strip() or DEFAULT_CAPTION_KEY
        enh_key = str(self.enhanced_caption_key or "").strip() or DEFAULT_ENHANCED_CAPTION_KEY

        windows_out: List[Dict[str, Any]] = []
        window_mcq_objs: List[Dict[str, Any]] = []
        window_mcq_corrected_objs: List[Dict[str, Any]] = []
        verify_windows_out: List[Dict[str, Any]] = []

        try:
            info = probe_video(clip_path)
        except Exception as e:
            logger.exception("Failed probing video for clip=%s (%s): %s", video_id, clip_path, e)
            base: Dict[str, Any] = {
                "video_id": video_id,
                "source_video": str(clip_path),
                "valid": False,
                "windows": [],
                "filtered_windows": [],
                "_error": f"probe_failed:{e.__class__.__name__}",
            }
            write_json(sidecar_metadata, base, sanitize_paths=True)
            return None

        tmp_root = work_dir / "_tmp_frames"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

        try:
            wf = int(self.window_frames or 0)
            ws = float(self.window_seconds or 0.0)
            if bool(self.single_window):
                wf = 0
                ws = 0.0

            if wf > 0:
                # Frame-index windowing, inclusive end indices.

                # Trust probed video FPS/frame count over sidecar metadata.
                fps_src = float(info.fps)
                if fps_src <= 0:
                    fps_src = 30.0

                last_frame_idx = max(0, int(info.num_frames) - 1)

                win_iter = iter_windows_by_frames(last_frame_idx + 1, wf)
                for w_idx, (start_frame, end_frame) in enumerate(win_iter):
                    win_errors: List[str] = []
                    start_sec = float(start_frame) / float(fps_src)
                    end_sec = float(end_frame + 1) / float(fps_src)
                    win_dir = tmp_root / f"win_{w_idx:03d}"
                    frames = extract_frames(
                        video_path=clip_path,
                        out_dir=win_dir,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        sampling_fps=self.sampling_fps,
                        resolution=self.resolution,
                        max_frames=self.max_frames,
                        logger=logger,
                    )
                    if not frames:
                        win_errors.append("no_frames_extracted")

                    mcq_text = ""
                    mcq_obj = None
                    if frames:
                        try:
                            mcq_obj, mcq_text = call_chat_json_with_structured_fallback(
                                base_url=self.vlm_base_url,
                                model=self.vlm_model,
                                messages=vlm_direct_mcq_messages_from_frames(frames, self.mcq_prompt),
                                timeout=self.timeout,
                                max_tokens=self.vlm_max_tokens,
                                temperature=float(self.vlm_temperature),
                                top_p=0.9,
                                logger=logger,
                                retries=int(self.vlm_retries or 0),
                                retry_backoff_s=float(self.vlm_retry_backoff_s or 5.0),
                                structured_output=str(self.vlm_structured_output or "openai"),
                                retry_stage="window_direct_vlm:vlm_direct_mcq",
                                api_key=get_vlm_api_key(),
                            )
                        except Exception as e:
                            win_errors.append(f"vlm_direct_mcq_failed:{e.__class__.__name__}")
                            logger.exception("VLM direct-MCQ call failed (clip=%s window=%d): %s", video_id, w_idx, e)
                            mcq_text = ""

                    if bool(self.retry_missing_questions) and frames:
                        base_obj = mcq_obj if isinstance(mcq_obj, dict) else extract_json_object_from_llm_text(mcq_text)
                        bank = self.question_bank if isinstance(self.question_bank, dict) else None
                        if bank is not None:
                            options_map = options_map_from_bank(bank)
                            bank_qs = bank_question_map(bank)
                            all_ids = list(bank_qs.keys())
                            retry_started_from_empty = not is_enhanced_mcq_payload(base_obj)
                            cur_obj: Dict[str, Any] = (
                                dict(base_obj)
                                if is_enhanced_mcq_payload(base_obj)
                                else {"version": 2.0, "video_id": video_id, "mcq": []}
                            )
                            cur_obj["video_id"] = video_id
                            if "version" not in cur_obj:
                                cur_obj["version"] = 2.0
                            for attempt in range(max(1, int(self.retry_missing_max_rounds or 1))):
                                required_missing = _missing_required_ids(
                                    bank=bank,
                                    include_if_map=self.include_if_map,
                                    mcq_obj=cur_obj,
                                )
                                if not required_missing:
                                    break
                                before = set(present_ids(cur_obj))
                                known = known_answers_for_retry(
                                    cur_obj,
                                    include_if_map=self.include_if_map,
                                    target_ids=required_missing,
                                )
                                logger.info(
                                    "Retry required questions (clip=%s window=%d frames=%d-%d attempt=%d missing=%d ids=%s known=%s)",
                                    video_id,
                                    w_idx,
                                    int(start_frame),
                                    int(end_frame),
                                    attempt + 1,
                                    len(required_missing),
                                    fmt_ids(required_missing),
                                    sorted(known.keys()),
                                )
                                missing_qs = [bank_qs[qid] for qid in required_missing if qid in bank_qs]
                                retry_prompt = build_retry_system_prompt(
                                    base_prompt=self.mcq_prompt,
                                    missing_qs=missing_qs,
                                    known_answers=known,
                                )
                                try:
                                    retry_obj, retry_text = call_chat_json_with_structured_fallback(
                                        base_url=self.vlm_base_url,
                                        model=self.vlm_model,
                                        messages=vlm_direct_mcq_messages_from_frames(frames, retry_prompt),
                                        timeout=self.timeout,
                                        max_tokens=self.vlm_max_tokens,
                                        temperature=float(self.vlm_temperature),
                                        top_p=0.9,
                                        logger=logger,
                                        retries=int(self.vlm_retries or 0),
                                        retry_backoff_s=float(self.vlm_retry_backoff_s or 5.0),
                                        structured_output=str(self.vlm_structured_output or "openai"),
                                        retry_stage="window_direct_vlm:vlm_direct_mcq:retry_required",
                                        api_key=get_vlm_api_key(),
                                    )
                                except Exception as e:
                                    win_errors.append(f"vlm_direct_mcq_retry_failed:{e.__class__.__name__}")
                                    logger.exception(
                                        "VLM direct-MCQ retry failed (clip=%s window=%d): %s", video_id, w_idx, e
                                    )
                                    break
                                obj2 = (
                                    retry_obj
                                    if isinstance(retry_obj, dict)
                                    else extract_json_object_from_llm_text(str(retry_text))
                                )
                                if not isinstance(obj2, dict):
                                    break
                                returned = [it for it in (obj2.get("mcq") or []) if isinstance(it, dict)]
                                if not returned:
                                    break
                                merged = {
                                    str(it.get("id") or "").strip(): it
                                    for it in (cur_obj.get("mcq") or [])
                                    if isinstance(it, dict)
                                }
                                for it in returned:
                                    qid = str(it.get("id") or "").strip()
                                    if qid and qid in set(required_missing):
                                        merged[qid] = it
                                merged_list = [merged[qid] for qid in all_ids if qid in merged] + [
                                    it for qid, it in merged.items() if qid not in set(all_ids)
                                ]
                                # Enforce include_if + answer validity after merge.
                                cur_obj["mcq"] = filter_mcq_items_strict(
                                    [it for it in merged_list if isinstance(it, dict)],
                                    include_if_map=self.include_if_map,
                                    options_map=options_map,
                                )
                                after = set(present_ids(cur_obj))
                                filled = sorted(after - before)
                                remain = _missing_required_ids(
                                    bank=bank,
                                    include_if_map=self.include_if_map,
                                    mcq_obj=cur_obj,
                                )
                                logger.info(
                                    "Retry required done (clip=%s window=%d frames=%d-%d filled=%d ids=%s required_missing=%d)",
                                    video_id,
                                    w_idx,
                                    int(start_frame),
                                    int(end_frame),
                                    len(filled),
                                    fmt_ids(filled),
                                    len(remain),
                                )
                                if not filled:
                                    break
                            has_retry_items = bool(present_ids(cur_obj))
                            if not retry_started_from_empty or has_retry_items:
                                mcq_obj = cur_obj

                    win_verify_obj: Dict[str, Any] | None = None
                    corrected_obj: Dict[str, Any] | None = None
                    if isinstance(mcq_obj, dict):
                        mcq_obj["video_id"] = video_id
                        if "version" not in mcq_obj:
                            mcq_obj["version"] = 2.0
                        # Optional VLM verify on finalized per-window MCQ.
                        if bool(self.vlm_verify_enabled) and frames:
                            mcq_items = [it for it in (mcq_obj.get("mcq") or []) if isinstance(it, dict)]
                            if mcq_items:
                                verify_result = self._verify_window_mcq(
                                    frames=frames,
                                    mcq_obj=mcq_obj,
                                    video_id=video_id,
                                    w_idx=w_idx,
                                    logger=logger,
                                    win_errors=win_errors,
                                )
                                corrected_obj = verify_result["corrected_obj"]
                                if verify_result["verify_status"] != "skipped_no_verifiable_items":
                                    win_verify_obj = {
                                        "status": verify_result["verify_status"],
                                        "verifications": verify_result["verify_items"],
                                        "corrected_count": verify_result["corrected_count"],
                                    }
                                    if "error" in verify_result:
                                        win_verify_obj["error"] = verify_result["error"]
                                    verify_windows_out.append(
                                        {
                                            "window_index": int(w_idx),
                                            "start_frame": int(start_frame),
                                            "end_frame": int(end_frame),
                                            "vlm_verify": win_verify_obj,
                                        }
                                    )
                        window_mcq_objs.append(mcq_obj)
                        if isinstance(corrected_obj, dict):
                            window_mcq_corrected_objs.append(corrected_obj)
                        else:
                            window_mcq_corrected_objs.append(dict(mcq_obj))
                    elif frames:
                        # VLM call returned without raising, but did not yield parseable MCQ JSON.
                        # Do not log model output contents; prompts/responses can contain user-controlled data.
                        reason = classify_mcq_json_parse_failure(mcq_text)
                        win_errors.append(f"vlm_direct_mcq_not_parseable:{reason}")
                        logger.warning(
                            "VLM direct-MCQ output not parseable (clip=%s window=%d reason=%s has_output=%s output_chars=%d)",
                            video_id,
                            w_idx,
                            reason,
                            bool(str(mcq_text or "").strip()),
                            len(str(mcq_text or "")),
                        )

                    win_obj: Dict[str, Any] = {"start_frame": int(start_frame), "end_frame": int(end_frame)}
                    win_obj[enh_key] = mcq_obj if isinstance(mcq_obj, dict) else {}
                    if win_errors:
                        win_obj["_errors"] = win_errors
                    if win_verify_obj is not None:
                        win_obj["vlm_verify"] = win_verify_obj
                    windows_out.append(win_obj)

                    if self.rate_limit and self.rate_limit > 0:
                        time.sleep(float(self.rate_limit))
            else:
                for w_idx, (start_sec, end_sec) in enumerate(iter_windows(info.duration_sec, ws)):
                    win_errors2: List[str] = []
                    win_dir = tmp_root / f"win_{w_idx:03d}"
                    frames = extract_frames(
                        video_path=clip_path,
                        out_dir=win_dir,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        sampling_fps=self.sampling_fps,
                        resolution=self.resolution,
                        max_frames=self.max_frames,
                        logger=logger,
                    )
                    if not frames:
                        win_errors2.append("no_frames_extracted")

                    start_frame = int(float(start_sec) * float(info.fps))
                    end_frame = int(float(end_sec) * float(info.fps)) - 1
                    if info.num_frames > 0:
                        end_frame = min(end_frame, int(info.num_frames) - 1)
                    end_frame = max(start_frame, end_frame)

                    mcq_text = ""
                    mcq_obj = None
                    if frames:
                        try:
                            mcq_obj, mcq_text = call_chat_json_with_structured_fallback(
                                base_url=self.vlm_base_url,
                                model=self.vlm_model,
                                messages=vlm_direct_mcq_messages_from_frames(frames, self.mcq_prompt),
                                timeout=self.timeout,
                                max_tokens=self.vlm_max_tokens,
                                temperature=float(self.vlm_temperature),
                                top_p=0.9,
                                logger=logger,
                                retries=int(self.vlm_retries or 0),
                                retry_backoff_s=float(self.vlm_retry_backoff_s or 5.0),
                                structured_output=str(self.vlm_structured_output or "openai"),
                                retry_stage="window_direct_vlm:vlm_direct_mcq",
                                api_key=get_vlm_api_key(),
                            )
                        except Exception as e:
                            win_errors2.append(f"vlm_direct_mcq_failed:{e.__class__.__name__}")
                            logger.exception("VLM direct-MCQ call failed (clip=%s window=%d): %s", video_id, w_idx, e)
                            mcq_text = ""

                    if bool(self.retry_missing_questions) and frames:
                        base_obj = mcq_obj if isinstance(mcq_obj, dict) else extract_json_object_from_llm_text(mcq_text)
                        bank = self.question_bank if isinstance(self.question_bank, dict) else None
                        if bank is not None:
                            options_map = options_map_from_bank(bank)
                            bank_qs = bank_question_map(bank)
                            all_ids = list(bank_qs.keys())
                            retry_started_from_empty = not is_enhanced_mcq_payload(base_obj)
                            cur_obj: Dict[str, Any] = (
                                dict(base_obj)
                                if is_enhanced_mcq_payload(base_obj)
                                else {"version": 2.0, "video_id": video_id, "mcq": []}
                            )
                            cur_obj["video_id"] = video_id
                            if "version" not in cur_obj:
                                cur_obj["version"] = 2.0
                            for attempt in range(max(1, int(self.retry_missing_max_rounds or 1))):
                                required_missing = _missing_required_ids(
                                    bank=bank,
                                    include_if_map=self.include_if_map,
                                    mcq_obj=cur_obj,
                                )
                                if not required_missing:
                                    break
                                before = set(present_ids(cur_obj))
                                known = known_answers_for_retry(
                                    cur_obj,
                                    include_if_map=self.include_if_map,
                                    target_ids=required_missing,
                                )
                                logger.info(
                                    "Retry required questions (clip=%s window=%d frames=%d-%d attempt=%d missing=%d ids=%s known=%s)",
                                    video_id,
                                    w_idx,
                                    int(start_frame),
                                    int(end_frame),
                                    attempt + 1,
                                    len(required_missing),
                                    fmt_ids(required_missing),
                                    sorted(known.keys()),
                                )
                                missing_qs = [bank_qs[qid] for qid in required_missing if qid in bank_qs]
                                retry_prompt = build_retry_system_prompt(
                                    base_prompt=self.mcq_prompt,
                                    missing_qs=missing_qs,
                                    known_answers=known,
                                )
                                try:
                                    retry_obj, retry_text = call_chat_json_with_structured_fallback(
                                        base_url=self.vlm_base_url,
                                        model=self.vlm_model,
                                        messages=vlm_direct_mcq_messages_from_frames(frames, retry_prompt),
                                        timeout=self.timeout,
                                        max_tokens=self.vlm_max_tokens,
                                        temperature=float(self.vlm_temperature),
                                        top_p=0.9,
                                        logger=logger,
                                        retries=int(self.vlm_retries or 0),
                                        retry_backoff_s=float(self.vlm_retry_backoff_s or 5.0),
                                        structured_output=str(self.vlm_structured_output or "openai"),
                                        retry_stage="window_direct_vlm:vlm_direct_mcq:retry_required",
                                        api_key=get_vlm_api_key(),
                                    )
                                except Exception as e:
                                    win_errors2.append(f"vlm_direct_mcq_retry_failed:{e.__class__.__name__}")
                                    logger.exception(
                                        "VLM direct-MCQ retry failed (clip=%s window=%d): %s", video_id, w_idx, e
                                    )
                                    break
                                obj2 = (
                                    retry_obj
                                    if isinstance(retry_obj, dict)
                                    else extract_json_object_from_llm_text(str(retry_text))
                                )
                                if not isinstance(obj2, dict):
                                    break
                                returned = [it for it in (obj2.get("mcq") or []) if isinstance(it, dict)]
                                if not returned:
                                    break
                                merged = {
                                    str(it.get("id") or "").strip(): it
                                    for it in (cur_obj.get("mcq") or [])
                                    if isinstance(it, dict)
                                }
                                for it in returned:
                                    qid = str(it.get("id") or "").strip()
                                    if qid and qid in set(required_missing):
                                        merged[qid] = it
                                merged_list = [merged[qid] for qid in all_ids if qid in merged] + [
                                    it for qid, it in merged.items() if qid not in set(all_ids)
                                ]
                                cur_obj["mcq"] = filter_mcq_items_strict(
                                    [it for it in merged_list if isinstance(it, dict)],
                                    include_if_map=self.include_if_map,
                                    options_map=options_map,
                                )
                                after = set(present_ids(cur_obj))
                                filled = sorted(after - before)
                                remain = _missing_required_ids(
                                    bank=bank,
                                    include_if_map=self.include_if_map,
                                    mcq_obj=cur_obj,
                                )
                                logger.info(
                                    "Retry required done (clip=%s window=%d frames=%d-%d filled=%d ids=%s required_missing=%d)",
                                    video_id,
                                    w_idx,
                                    int(start_frame),
                                    int(end_frame),
                                    len(filled),
                                    fmt_ids(filled),
                                    len(remain),
                                )
                                if not filled:
                                    break
                            has_retry_items = bool(present_ids(cur_obj))
                            if not retry_started_from_empty or has_retry_items:
                                mcq_obj = cur_obj

                    win_verify_obj: Dict[str, Any] | None = None
                    corrected_obj: Dict[str, Any] | None = None
                    if isinstance(mcq_obj, dict):
                        mcq_obj["video_id"] = video_id
                        if "version" not in mcq_obj:
                            mcq_obj["version"] = 2.0
                        # Optional VLM verify on finalized per-window MCQ.
                        if bool(self.vlm_verify_enabled) and frames:
                            mcq_items = [it for it in (mcq_obj.get("mcq") or []) if isinstance(it, dict)]
                            if mcq_items:
                                verify_result = self._verify_window_mcq(
                                    frames=frames,
                                    mcq_obj=mcq_obj,
                                    video_id=video_id,
                                    w_idx=w_idx,
                                    logger=logger,
                                    win_errors=win_errors2,
                                )
                                corrected_obj = verify_result["corrected_obj"]
                                if verify_result["verify_status"] != "skipped_no_verifiable_items":
                                    win_verify_obj = {
                                        "status": verify_result["verify_status"],
                                        "verifications": verify_result["verify_items"],
                                        "corrected_count": verify_result["corrected_count"],
                                    }
                                    if "error" in verify_result:
                                        win_verify_obj["error"] = verify_result["error"]
                                    verify_windows_out.append(
                                        {
                                            "window_index": int(w_idx),
                                            "start_frame": int(start_frame),
                                            "end_frame": int(end_frame),
                                            "vlm_verify": win_verify_obj,
                                        }
                                    )
                        window_mcq_objs.append(mcq_obj)
                        if isinstance(corrected_obj, dict):
                            window_mcq_corrected_objs.append(corrected_obj)
                        else:
                            window_mcq_corrected_objs.append(dict(mcq_obj))
                    elif frames:
                        # VLM call returned without raising, but did not yield parseable MCQ JSON.
                        # Do not log model output contents; prompts/responses can contain user-controlled data.
                        reason = classify_mcq_json_parse_failure(mcq_text)
                        win_errors2.append(f"vlm_direct_mcq_not_parseable:{reason}")
                        logger.warning(
                            "VLM direct-MCQ output not parseable (clip=%s window=%d reason=%s has_output=%s output_chars=%d)",
                            video_id,
                            w_idx,
                            reason,
                            bool(str(mcq_text or "").strip()),
                            len(str(mcq_text or "")),
                        )

                    win_obj2: Dict[str, Any] = {"start_frame": start_frame, "end_frame": end_frame}
                    win_obj2[enh_key] = mcq_obj if isinstance(mcq_obj, dict) else {}
                    if win_errors2:
                        win_obj2["_errors"] = win_errors2
                    if win_verify_obj is not None:
                        win_obj2["vlm_verify"] = win_verify_obj
                    windows_out.append(win_obj2)

                    if self.rate_limit and self.rate_limit > 0:
                        time.sleep(float(self.rate_limit))

            metadata = {
                "video_id": video_id,
                "source_video": str(clip_path),
                "duration_span": [0.0, round(info.duration_sec, 3)],
                "width": info.width,
                "height": info.height,
                "framerate": info.fps,
                "num_frames": info.num_frames,
            }
            metadata["windowing"] = {
                "window_mode": ("single" if bool(self.single_window) else ("frames" if wf > 0 else "seconds")),
                "window_frames": int(wf),
                "window_seconds": float(ws),
                "single_window": bool(self.single_window),
                "sampling_fps": float(self.sampling_fps or 0.0),
                "resolution": int(self.resolution or 0),
                "max_frames": int(self.max_frames or 0),
                "caption_key": str(cap_key),
                "enhanced_caption_key": str(enh_key),
            }

            metadata["windows"] = windows_out
            metadata.setdefault("filtered_windows", [])
            metadata["valid"] = True
            metadata["has_caption"] = any(is_enhanced_mcq_payload(w.get(enh_key)) for w in windows_out)
            write_json(sidecar_metadata, metadata, sanitize_paths=True)
            enhanced_objects = sum(1 for w in windows_out if is_enhanced_mcq_payload(w.get(enh_key)))
            logger.info(
                "window-direct-vlm: metadata sidecar written (windows=%d enhanced_payload_objects=%d verify_enabled=%s)",
                len(windows_out),
                enhanced_objects,
                bool(self.vlm_verify_enabled),
            )

            def _compose_mcq_object(mcq_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
                out_obj = (
                    self._aggregate_mcqs(mcq_objs, video_id=video_id) if self.aggregate_windows else dict(mcq_objs[0])
                )
                out_obj["video_id"] = video_id
                if bool(self.vlm_verify_enabled) and verify_windows_out:
                    attach_reasoning_traces_from_verify(out_obj, verify_windows_out=verify_windows_out)
                return out_obj

            mcq_written = False
            task_composed = False
            if window_mcq_objs:
                output_mcq_objs = window_mcq_corrected_objs if window_mcq_corrected_objs else window_mcq_objs
                composed = _compose_mcq_object(output_mcq_objs)
                items = list(composed.get("mcq", []))
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
                    logger.warning("No DAFT task items for clip=%s; skipping task output", video_id)
            else:
                sidecar_empty.unlink(missing_ok=True)

            if bool(self.vlm_verify_enabled) and mcq_written and verify_windows_out:
                verify_items_total = 0
                corrected_total = 0
                for w in verify_windows_out:
                    vobj = w.get("vlm_verify")
                    if isinstance(vobj, dict) and str(vobj.get("status") or "") == "ok":
                        verify_items_total += len(vobj.get("verifications") or [])
                        corrected_total += int(vobj.get("corrected_count") or 0)

                verify_sidecar_payload = {
                    "version": 1.0,
                    "video_id": video_id,
                    "source_video": str(clip_path),
                    "windows": verify_windows_out,
                    "summary": {
                        "windows_total": len(windows_out),
                        "windows_with_verify": len(verify_windows_out),
                        "questions_verified": verify_items_total,
                        "questions_corrected": corrected_total,
                    },
                }
                write_json(sidecar_verify, verify_sidecar_payload, sanitize_paths=True)

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def run_single(
        self,
        *,
        clip_path: Path,
        input_root: Path,
        output_root: Path,
        output_dir: Path,
        skip_existing: bool = False,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        logger = logger or setup_logger(verbose)
        if skip_existing and outputs_complete(
            out_dir=output_dir,
            direct_mcq_from_vlm=True,
            caption_key=str(self.caption_key),
            enhanced_caption_key=str(self.enhanced_caption_key),
            require_vlm_verify=bool(self.vlm_verify_enabled),
        ):
            logger.info("Skip existing outputs for clip=%s (output_dir=%s)", clip_path.name, output_dir)
            return None

        self.build_for_clip(
            clip_path=clip_path,
            input_root=input_root if input_root.is_dir() else input_root.parent,
            output_root=output_root,
            output_dir=output_dir,
            logger=logger,
        )
