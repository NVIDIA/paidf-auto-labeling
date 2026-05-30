# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Window direct-VLM MCQ generator.

Wraps ``WindowDirectVlmRunner`` — per-window direct VLM MCQ (no LLM).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.io import sha256_text, write_json, write_text
from al_utils.schema.config import PipelineConfig
from al_utils.schema.mcq import WindowMetadataExtractionConfig
from daft_export.paths import scene_paths
from mcq_generation.base import BaseMCQGenerator, MCQResult
from mcq_generation.mcq.runners.window_direct_vlm import WindowDirectVlmRunner
from mcq_generation.mcq.utils.aggregation import aggregation_specs_from_bank
from mcq_generation.mcq.utils.bank import collect_embedded_bank_from_prompt, include_if_map_from_bank
from mcq_generation.mcq.utils.prompt_io import load_text as _load_text
from mcq_generation.mcq.utils.vlm_verify import render_vlm_verify_prompt_template


class WindowDirectVlmGenerator(BaseMCQGenerator):
    """Per-window direct-VLM MCQ generation (no LLM step).

    Corresponds to ``mcq_generation.mode = window-direct-vlm``.
    """

    def __init__(
        self,
        config: PipelineConfig,
        resolver: EndpointResolver,
        logger: logging.Logger,
        config_dir: Optional[str] = None,
    ) -> None:
        super().__init__(logger)
        w: WindowMetadataExtractionConfig = config.mcq_generation.window_metadata_extraction

        mcq_prompt = _load_text(
            w.mcq_prompt_file or "cookbooks/traffic/prompts/mcq/window_direct_vlm/mcq_prompt.md",
            config_dir,
        )
        verify_prompt = _load_text(
            w.vlm_verify_prompt_file or "cookbooks/shared/prompts/mcq/vlm_verify/verify_prompt.md",
            config_dir,
        )
        bank = collect_embedded_bank_from_prompt(mcq_prompt)

        vlm_url, vlm_model = resolver.resolve_vlm(required=True)

        self._runner = WindowDirectVlmRunner(
            mcq_prompt=mcq_prompt,
            include_if_map=include_if_map_from_bank(bank),
            aggregation_specs=aggregation_specs_from_bank(bank),
            question_bank=bank if isinstance(bank, dict) else None,
            vlm_base_url=vlm_url,
            vlm_model=vlm_model,
            vlm_retries=resolver.vlm_retries,
            vlm_retry_backoff_s=resolver.vlm_retry_backoff_s,
            vlm_structured_output=str(w.vlm_structured_output),
            caption_key=str(w.caption_key),
            enhanced_caption_key=str(w.enhanced_caption_key),
            window_seconds=float(w.window_seconds),
            window_frames=int(w.window_frames),
            single_window=bool(w.single_window),
            sampling_fps=float(w.sampling_fps),
            resolution=int(w.resolution),
            max_frames=int(w.max_frames),
            vlm_max_tokens=int(w.vlm_max_tokens),
            vlm_temperature=float(w.vlm_temperature),
            timeout=int(w.timeout),
            rate_limit=float(w.rate_limit),
            aggregate_windows=bool(w.aggregate_windows),
            write_empty_mcq_marker=bool(w.write_empty_mcq_marker),
            retry_missing_questions=bool(w.retry_missing_questions),
            retry_missing_max_rounds=int(w.retry_missing_max_rounds),
            vlm_verify_enabled=bool(w.vlm_verify_enabled),
            vlm_verify_max_tokens=int(w.vlm_verify_max_tokens),
            vlm_verify_temperature=float(w.vlm_verify_temperature),
            vlm_verify_structured_output=str(w.vlm_verify_structured_output),
            vlm_verify_apply_corrections=bool(w.vlm_verify_apply_corrections),
            vlm_verify_prompt_template=verify_prompt,
        )
        self._skip_existing = bool(w.skip_existing)
        self._vlm_verify_enabled = bool(w.vlm_verify_enabled)
        self._mcq_prompt = mcq_prompt
        self._verify_prompt = verify_prompt

    def _persist_prompts(self, output_dir: Path) -> None:
        prompts_dir = output_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        def _nl(s: str) -> str:
            return s + ("\n" if not s.endswith("\n") else "")

        write_text(prompts_dir / "mcq_prompt.used.md", _nl(self._mcq_prompt))
        hashes = {"mcq_prompt_sha256": sha256_text(self._mcq_prompt)}
        if self._vlm_verify_enabled:
            rendered_verify_prompt = render_vlm_verify_prompt_template(
                prompt_template=self._verify_prompt,
                apply_corrections=bool(self._runner.vlm_verify_apply_corrections),
            )
            write_text(prompts_dir / "vlm_verify_prompt.used.md", _nl(rendered_verify_prompt))
            hashes["vlm_verify_prompt_sha256"] = sha256_text(rendered_verify_prompt)
        else:
            try:
                (prompts_dir / "vlm_verify_prompt.used.md").unlink()
            except FileNotFoundError:
                pass
        write_json(prompts_dir / "prompts.used.json", hashes)

    def generate(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        events_json: Optional[Path] = None,
        video_json: Optional[Path] = None,
        metadata_json: Optional[Path] = None,
    ) -> MCQResult:
        paths = scene_paths(output_dir)
        work_dir = paths.sidecars_dir / "_work" / "window_direct_vlm"

        self._persist_prompts(output_dir)
        try:
            self.logger.info(
                "window-direct-vlm: starting run (clip=%s, work_dir=%s, output_dir=%s, skip_existing=%s)",
                video_path,
                work_dir,
                output_dir,
                self._skip_existing,
            )
            self._runner.run_single(
                clip_path=Path(video_path),
                input_root=Path(video_path).parent,
                output_root=work_dir,
                output_dir=output_dir,
                skip_existing=self._skip_existing,
                logger=self.logger,
            )
            self.logger.info("window-direct-vlm: completed run (output_dir=%s)", output_dir)
        finally:
            try:
                shutil.rmtree(work_dir, ignore_errors=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.logger.warning("window-direct-vlm: failed to clean work_dir=%s: %s", work_dir, exc)

        sidecar_meta = paths.sidecars_dir / "metadata.json"
        return MCQResult(
            success=True,
            mcq_json=paths.task_mcq if paths.task_mcq.exists() else None,
            bcq_json=paths.task_bcq if paths.task_bcq.exists() else None,
            open_qa_json=paths.task_open_qa if paths.task_open_qa.exists() else None,
            metadata_json=sidecar_meta if sidecar_meta.exists() else None,
        )
