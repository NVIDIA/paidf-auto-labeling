# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata LLM MCQ generator.

Wraps ``MetadataLlmRunner`` — LLM MCQ generation from an existing metadata.json
(window captions already present from a prior run).

Corresponds to ``mcq_generation.mode = metadata-llm``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.schema.config import PipelineConfig
from al_utils.schema.mcq import WindowMetadataExtractionConfig
from daft_export.paths import scene_paths
from mcq_generation.base import BaseMCQGenerator, MCQResult
from mcq_generation.mcq.runners.metadata_llm import (
    DEFAULT_CAPTION_KEY,
    DEFAULT_ENHANCED_CAPTION_KEY,
    MetadataLlmRunner,
)
from mcq_generation.mcq.utils.prompt_io import load_text as _load_text
from mcq_generation.mcq.utils.prompt_io import resolve_path as _resolve_path


class MetadataLlmGenerator(BaseMCQGenerator):
    """LLM MCQ generation from an existing ``metadata.json``.

    Corresponds to ``mcq_generation.mode = metadata-llm``.

    ``generate()`` consumes an input ``metadata.json`` — either explicitly
    via the ``metadata_json`` kwarg (the pipeline passes
    ``data[*].inputs.metadata_json_path``, staged locally by ``cli.py``), or
    from the scene's own ``<scene>/sidecars/metadata.json`` as a fallback for
    in-place re-runs.

    ``video_path`` is not used for LLM inference when schema validation
    auto-disables VLM verification, but the pipeline CLI still requires the
    sample's primary ``data[*].inputs.video_path`` to point to an existing
    media file.
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

        mcq_prompt_path = w.mcq_prompt_file or "cookbooks/traffic/prompts/mcq/metadata_llm/mcq_prompt.md"
        mcq_prompt = _load_text(mcq_prompt_path, config_dir)
        verify_prompt = _load_text(
            w.vlm_verify_prompt_file or "cookbooks/shared/prompts/mcq/vlm_verify/verify_prompt.md",
            config_dir,
        )
        llm_url, llm_model = resolver.resolve_llm(required=True)
        vlm_url, vlm_model = resolver.resolve_vlm(required=False)

        self._runner = MetadataLlmRunner(
            mcq_prompt=mcq_prompt,
            llm_base_url=llm_url,
            llm_model=llm_model,
            llm_structured_output=str(w.llm_structured_output),
            # metadata-llm uses the embedded bank in mcq_prompt_file (keep behavior consistent with window-* modes).
            question_bank_file=None,
            caption_key=str(w.caption_key) or DEFAULT_CAPTION_KEY,
            enhanced_caption_key=str(w.enhanced_caption_key) or DEFAULT_ENHANCED_CAPTION_KEY,
            llm_retries=resolver.llm_retries,
            llm_retry_backoff_s=resolver.llm_retry_backoff_s,
            llm_max_tokens=int(w.llm_max_tokens),
            llm_temperature=float(w.llm_temperature),
            timeout=int(w.timeout),
            rate_limit=float(w.rate_limit),
            aggregate_windows=bool(w.aggregate_windows),
            write_empty_mcq_marker=bool(w.write_empty_mcq_marker),
            skip_existing=bool(w.skip_existing),
            retry_missing_questions=bool(w.retry_missing_questions),
            retry_missing_max_rounds=int(w.retry_missing_max_rounds),
            vlm_verify_enabled=bool(w.vlm_verify_enabled),
            vlm_verify_apply_corrections=bool(w.vlm_verify_apply_corrections),
            vlm_base_url=vlm_url,
            vlm_model=vlm_model,
            vlm_retries=resolver.vlm_retries,
            vlm_retry_backoff_s=resolver.vlm_retry_backoff_s,
            vlm_verify_max_tokens=int(w.vlm_verify_max_tokens),
            vlm_verify_temperature=float(w.vlm_verify_temperature),
            vlm_verify_structured_output=str(w.vlm_verify_structured_output),
            vlm_verify_prompt_template=verify_prompt,
        )
        self._mcq_prompt_file = _resolve_path(mcq_prompt_path, config_dir)

    def generate(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        events_json: Optional[Path] = None,
        video_json: Optional[Path] = None,
        metadata_json: Optional[Path] = None,
    ) -> MCQResult:
        """Generate MCQ from an existing ``metadata.json``.

        Args:
            video_path:    Path to the input media (only used by direct-runner VLM verify).
            output_dir:    Per-sample output directory.
            events_json:   Unused by this mode.
            metadata_json: Path to an existing ``metadata.json`` with window
                           captions. The pipeline passes
                           ``sample.inputs.metadata_json_path`` after staging
                           it locally in ``cli.py``. Falls back to
                           ``output_dir/sidecars/metadata.json`` when not
                           provided.
        """
        paths = scene_paths(output_dir)
        # Input metadata defaults to the scene's own metadata sidecar when the
        # caller hasn't explicitly staged one — useful for re-running the LLM
        # pass over a previously-produced scene in place.
        input_metadata = metadata_json or (paths.sidecars_dir / "metadata.json")

        # Attach video path for optional VLM verify frame extraction.
        if bool(self._runner.vlm_verify_enabled) and video_path is not None:
            self._runner.input_video_path = Path(video_path)

        self._runner.persist_prompts(
            output_root=Path(output_dir),
            mcq_prompt_file=self._mcq_prompt_file,
        )
        self.logger.info(
            "metadata-llm: starting run (metadata=%s, output_dir=%s, vlm_verify=%s)",
            input_metadata,
            output_dir,
            bool(self._runner.vlm_verify_enabled),
        )
        self._runner.run_single(
            input_metadata_json=Path(input_metadata),
            output_dir=Path(output_dir),
            logger=self.logger,
        )
        self.logger.info("metadata-llm: completed run (output_dir=%s)", output_dir)

        sidecar_meta = paths.sidecars_dir / "metadata.json"
        return MCQResult(
            success=True,
            mcq_json=paths.task_mcq if paths.task_mcq.exists() else None,
            bcq_json=paths.task_bcq if paths.task_bcq.exists() else None,
            open_qa_json=paths.task_open_qa if paths.task_open_qa.exists() else None,
            metadata_json=sidecar_meta if sidecar_meta.exists() else None,
        )
