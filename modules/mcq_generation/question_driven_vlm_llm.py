# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Question-driven VLM+LLM MCQ generator.

Two-phase approach:
  1. ``run_pre_step()`` — LLM-only prompt generation from a question bank
     (runs before SR/tracking; no video frames needed).
  2. ``generate()`` — per-window VLM captioning + LLM MCQ using generated prompts.

Corresponds to ``mcq_generation.mode = question-driven-vlm-llm``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.io import read_json, read_text, sha256_text, write_json, write_text
from al_utils.schema.config import PipelineConfig
from al_utils.schema.mcq import WindowMetadataExtractionConfig
from daft_export.paths import scene_paths
from mcq_generation.base import BaseMCQGenerator, MCQResult
from mcq_generation.mcq.runners.window_vlm_llm import WindowVlmLlmRunner
from mcq_generation.mcq.utils.aggregation import aggregation_specs_from_bank
from mcq_generation.mcq.utils.bank import include_if_map_from_bank, inject_bank_into_template, read_bank
from mcq_generation.mcq.utils.prompt_io import resolve_path as _resolve_path
from mcq_generation.mcq.utils.question_driven_prompt_gen import generate_mapper_rules, generate_vlm_scene_prompt
from mcq_generation.mcq.utils.vlm_verify import render_vlm_verify_prompt_template


class QuestionDrivenVlmLlmGenerator(BaseMCQGenerator):
    """Question-driven VLM+LLM: LLM generates prompts from a question bank,
    then window-VLM+LLM runs with those prompts.

    ``run_pre_step()`` is overridden to execute the LLM prompt-gen phase
    before super-resolution (no video needed).  ``generate()`` reads the
    prompts produced by ``run_pre_step()`` and runs the window runner.
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

        # Prompt-gen LLM endpoint (falls back to general LLM endpoint)
        llm_url, llm_model = resolver.resolve_llm(required=True)
        self._prompt_gen_llm_url: str = str(w.prompt_gen_llm_base_url or "").strip() or llm_url
        self._prompt_gen_llm_model: str = str(w.prompt_gen_llm_model or "").strip() or llm_model
        self._prompt_gen_llm_max_tokens: int = int(w.prompt_gen_llm_max_tokens)
        self._prompt_gen_seed: Optional[int] = w.prompt_gen_seed
        self._prompt_gen_llm_retries: int = resolver.llm_retries
        self._timeout: int = int(w.timeout)
        self._append_mapper_rules: bool = bool(w.append_mapper_rules)
        self._skip_existing: bool = bool(w.skip_existing)
        self._vlm_verify_enabled: bool = bool(w.vlm_verify_enabled)
        self._config_dir = config_dir

        self._question_bank_file: Optional[Path] = _resolve_path(w.question_bank_file, config_dir)
        self._qd_vlm_scene_prompt_template_file: Optional[Path] = _resolve_path(
            w.qd_vlm_scene_prompt_template_file
            or "cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/vlm_scene_prompt_template.md",
            config_dir,
        )
        self._qd_mcq_mapper_prompt_template_file: Optional[Path] = _resolve_path(
            w.qd_mcq_mapper_prompt_template_file
            or "cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/mcq_mapper_bank_injected_template.md",
            config_dir,
        )
        self._vlm_verify_prompt_file: Optional[Path] = _resolve_path(
            w.vlm_verify_prompt_file or "cookbooks/shared/prompts/mcq/vlm_verify/verify_prompt.md",
            config_dir,
        )

        # Window runner params (prompts loaded lazily in generate())
        vlm_url, vlm_model = resolver.resolve_vlm(required=True)
        self._vlm_url = vlm_url
        self._vlm_model = vlm_model
        self._llm_url = llm_url
        self._llm_model = llm_model
        self._resolver = resolver
        self._window_cfg = w

    # ------------------------------------------------------------------
    # Pre-step: LLM prompt generation (runs before SR/tracking)
    # ------------------------------------------------------------------

    def run_pre_step(self, out_dir: Path, sample: object) -> None:  # type: ignore[override]
        """Run LLM prompt generation — no video frames needed.

        Writes the final prompts used by downstream inference to
        ``<out_dir>/prompts/``. Intermediate QD prompt-generation artifacts are
        only written when they are not duplicates of the final ``*.used.md``
        files.

        Mirrors the logic of ``run_qd_prompt_gen.py::main()``.
        """
        prompts_dir = Path(out_dir) / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        scene_prompt_used = prompts_dir / "scene_prompt.used.md"
        mcq_prompt_used = prompts_dir / "mcq_prompt.used.md"
        verify_prompt_used = prompts_dir / "vlm_verify_prompt.used.md"

        if self._skip_existing and scene_prompt_used.exists() and mcq_prompt_used.exists():
            prompts_used_json = prompts_dir / "prompts.used.json"
            prompts_used = read_json(prompts_used_json) if prompts_used_json.exists() else {}
            if not isinstance(prompts_used, dict):
                prompts_used = {}
            if self._vlm_verify_enabled:
                if not verify_prompt_used.exists():
                    verify_prompt = read_text(self._vlm_verify_prompt_file)
                    rendered_verify_prompt = render_vlm_verify_prompt_template(
                        prompt_template=verify_prompt,
                        apply_corrections=bool(self._window_cfg.vlm_verify_apply_corrections),
                    )
                    write_text(
                        verify_prompt_used,
                        rendered_verify_prompt + ("\n" if not rendered_verify_prompt.endswith("\n") else ""),
                    )
                else:
                    rendered_verify_prompt = read_text(verify_prompt_used)
                prompts_used["vlm_verify_prompt_sha256"] = sha256_text(rendered_verify_prompt)
            else:
                try:
                    verify_prompt_used.unlink()
                except FileNotFoundError:
                    pass
                prompts_used.pop("vlm_verify_prompt_sha256", None)
            write_json(prompts_used_json, prompts_used)
            self.logger.info("QD prompt gen: skipping — output files already exist.")
            return None

        if self._question_bank_file is None:
            raise ValueError(
                "question_driven-vlm-llm requires mcq_generation.window_metadata_extraction.question_bank_file"
            )

        # 1) question bank -> LLM generates a VLM scene prompt
        scene_meta = generate_vlm_scene_prompt(
            question_bank_file=self._question_bank_file,
            system_template_file=self._qd_vlm_scene_prompt_template_file,
            llm_base_url=self._prompt_gen_llm_url,
            llm_model=self._prompt_gen_llm_model,
            max_tokens=self._prompt_gen_llm_max_tokens,
            temperature=0.0,
            timeout=self._timeout,
            seed=self._prompt_gen_seed,
            retries=self._prompt_gen_llm_retries,
            logger=self.logger,
        )
        scene_text = scene_meta["prompt_text"]
        write_json(prompts_dir / "scene_prompt.generated_by_llm.meta.json", scene_meta)

        # 2) question bank + mapper template -> bank-injected MCQ mapper prompt
        bank = read_bank(self._question_bank_file)
        template = read_text(self._qd_mcq_mapper_prompt_template_file)
        bank_injected_text = inject_bank_into_template(template, bank_payload=bank)

        if self._append_mapper_rules:
            rules_template = _resolve_path(
                "cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/mcq_mapper_rules_appendix_template.md",
                self._config_dir,
            )
            rules_meta = generate_mapper_rules(
                question_bank_file=self._question_bank_file,
                system_template_file=rules_template,
                llm_base_url=self._prompt_gen_llm_url,
                llm_model=self._prompt_gen_llm_model,
                max_tokens=self._prompt_gen_llm_max_tokens,
                temperature=0.0,
                timeout=self._timeout,
                seed=self._prompt_gen_seed,
                retries=self._prompt_gen_llm_retries,
                logger=self.logger,
            )
            rules_text = str(rules_meta.get("rules_text") or "")
            write_text(
                prompts_dir / "mcq_prompt.mapper_rules.generated_by_llm.md",
                rules_text + ("\n" if not rules_text.endswith("\n") else ""),
            )
            fused = bank_injected_text.rstrip() + "\n\n---\n\n" + rules_text.strip() + "\n"
            mcq_text_used = fused
        else:
            mcq_text_used = bank_injected_text

        write_text(scene_prompt_used, scene_text + ("\n" if not scene_text.endswith("\n") else ""))
        write_text(mcq_prompt_used, mcq_text_used + ("\n" if not mcq_text_used.endswith("\n") else ""))
        prompt_hashes = {
            "scene_prompt_sha256": sha256_text(scene_text),
            "mcq_prompt_sha256": sha256_text(mcq_text_used),
        }
        if self._vlm_verify_enabled:
            verify_prompt = read_text(self._vlm_verify_prompt_file)
            rendered_verify_prompt = render_vlm_verify_prompt_template(
                prompt_template=verify_prompt,
                apply_corrections=bool(self._window_cfg.vlm_verify_apply_corrections),
            )
            write_text(
                verify_prompt_used, rendered_verify_prompt + ("\n" if not rendered_verify_prompt.endswith("\n") else "")
            )
            prompt_hashes["vlm_verify_prompt_sha256"] = sha256_text(rendered_verify_prompt)
        else:
            try:
                verify_prompt_used.unlink()
            except FileNotFoundError:
                pass
        if bank_injected_text.strip() != mcq_text_used.strip():
            write_text(
                prompts_dir / "mcq_prompt.bank_injected.md",
                bank_injected_text + ("\n" if not bank_injected_text.endswith("\n") else ""),
            )
        write_json(prompts_dir / "prompts.used.json", prompt_hashes)
        self.logger.info(
            "QD prompt gen complete: %s, %s%s",
            scene_prompt_used,
            mcq_prompt_used,
            f", {verify_prompt_used}" if self._vlm_verify_enabled else "",
        )

    # ------------------------------------------------------------------
    # Generate: window inference with LLM-generated prompts
    # ------------------------------------------------------------------

    def generate(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        events_json: Optional[Path] = None,
        video_json: Optional[Path] = None,
        metadata_json: Optional[Path] = None,
    ) -> MCQResult:
        """Run window VLM+LLM MCQ using prompts produced by ``run_pre_step()``."""
        paths = scene_paths(output_dir)
        prompts_dir = Path(output_dir) / "prompts"
        work_dir = paths.sidecars_dir / "_work" / "question_driven_vlm_llm"

        scene_prompt = read_text(prompts_dir / "scene_prompt.used.md")
        mcq_prompt = read_text(prompts_dir / "mcq_prompt.used.md")
        verify_prompt_used = prompts_dir / "vlm_verify_prompt.used.md"
        if self._vlm_verify_enabled:
            verify_prompt = (
                read_text(verify_prompt_used)
                if verify_prompt_used.exists()
                else render_vlm_verify_prompt_template(
                    prompt_template=read_text(self._vlm_verify_prompt_file),
                    apply_corrections=bool(self._window_cfg.vlm_verify_apply_corrections),
                )
            )
        else:
            verify_prompt = ""

        bank = read_bank(self._question_bank_file) if self._question_bank_file else None

        w = self._window_cfg
        runner = WindowVlmLlmRunner(
            scene_prompt=scene_prompt,
            mcq_prompt=mcq_prompt,
            include_if_map=include_if_map_from_bank(bank),
            aggregation_specs=aggregation_specs_from_bank(bank),
            question_bank=bank if isinstance(bank, dict) else None,
            vlm_base_url=self._vlm_url,
            vlm_model=self._vlm_model,
            llm_base_url=self._llm_url,
            llm_model=self._llm_model,
            vlm_retries=self._resolver.vlm_retries,
            vlm_retry_backoff_s=self._resolver.vlm_retry_backoff_s,
            llm_retries=self._resolver.llm_retries,
            llm_retry_backoff_s=self._resolver.llm_retry_backoff_s,
            window_seconds=float(w.window_seconds),
            window_frames=int(w.window_frames),
            single_window=bool(w.single_window),
            sampling_fps=float(w.sampling_fps),
            resolution=int(w.resolution),
            max_frames=int(w.max_frames),
            vlm_max_tokens=int(w.vlm_max_tokens),
            llm_max_tokens=int(w.llm_max_tokens),
            vlm_temperature=float(w.vlm_temperature),
            llm_temperature=float(w.llm_temperature),
            llm_structured_output=str(w.llm_structured_output),
            timeout=int(w.timeout),
            rate_limit=float(w.rate_limit),
            aggregate_windows=bool(w.aggregate_windows),
            caption_key=str(w.caption_key),
            enhanced_caption_key=str(w.enhanced_caption_key),
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
        try:
            self.logger.info(
                "question-driven-vlm-llm: starting run (clip=%s, work_dir=%s, output_dir=%s, skip_existing=%s)",
                video_path,
                work_dir,
                output_dir,
                self._skip_existing,
            )
            runner.run_single(
                clip_path=Path(video_path),
                input_root=Path(video_path).parent,
                output_root=work_dir,
                output_dir=Path(output_dir),
                skip_existing=self._skip_existing,
                logger=self.logger,
            )
            self.logger.info("question-driven-vlm-llm: completed run (output_dir=%s)", output_dir)
        finally:
            try:
                shutil.rmtree(work_dir, ignore_errors=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.logger.warning("question-driven-vlm-llm: failed to clean work_dir=%s: %s", work_dir, exc)

        sidecar_meta = paths.sidecars_dir / "metadata.json"
        return MCQResult(
            success=True,
            mcq_json=paths.task_mcq if paths.task_mcq.exists() else None,
            bcq_json=paths.task_bcq if paths.task_bcq.exists() else None,
            open_qa_json=paths.task_open_qa if paths.task_open_qa.exists() else None,
            metadata_json=sidecar_meta if sidecar_meta.exists() else None,
        )
