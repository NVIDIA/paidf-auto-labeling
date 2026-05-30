# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WindowMetadataExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_prompt_file: Optional[str] = None
    mcq_prompt_file: Optional[str] = None
    # Question bank (used by question-driven modes to generate prompts)
    question_bank_file: Optional[str] = None
    # Question-driven prompt generation via LLM (used by mcq_generation.mode=question-driven-vlm-llm)
    prompt_gen_llm_base_url: Optional[str] = None
    prompt_gen_llm_model: Optional[str] = None
    prompt_gen_llm_max_tokens: int = 8192
    prompt_gen_seed: Optional[int] = None
    qd_vlm_scene_prompt_template_file: Optional[str] = None
    qd_mcq_mapper_prompt_template_file: Optional[str] = None
    append_mapper_rules: bool = False

    window_frames: int = 0
    window_seconds: float = 4.0
    # If true, ignore window_frames/window_seconds and use one window covering the whole media sample.
    # Frame sampling still respects sampling_fps/resolution/max_frames.
    single_window: bool = False
    sampling_fps: float = 2.0
    resolution: int = 480
    max_frames: int = 100
    # windows[].<caption_key>: where the per-window VLM caption text is written/read.
    # Used by window-vlm-llm and metadata-llm.
    caption_key: str = "vlm_caption"
    # windows[].<enhanced_caption_key>: where the per-window LLM-enhanced payload is written/read.
    # Stores embedded MCQ JSON as an object.
    enhanced_caption_key: str = "llm_enhanced_caption"

    vlm_max_tokens: int = 8192
    llm_max_tokens: int = 8192
    # Default deterministic for batch stability.
    vlm_temperature: float = 0.0
    llm_temperature: float = 0.0
    # Structured output mode for VLM/LLM JSON calls:
    # - auto: NIM endpoints -> guided_json; otherwise -> response_format=json_object
    # - nim: force guided_json
    # - openai: force response_format=json_object
    # - off: disable structured outputs (raw + best-effort JSON extraction)
    vlm_structured_output: Literal["auto", "nim", "openai", "off"] = "openai"
    llm_structured_output: Literal["auto", "nim", "openai", "off"] = "openai"
    timeout: int = 600
    rate_limit: float = 0.0

    aggregate_windows: bool = False
    write_empty_mcq_marker: bool = True
    skip_existing: bool = False
    # Best-effort: retry and fill missing question ids per window (LLM-only, no extra frame extraction).
    retry_missing_questions: bool = True
    retry_missing_max_rounds: int = Field(default=2, ge=1)
    # Post-check for MCQ window outputs:
    # after per-window MCQ (and retry_missing) is finalized, run an extra VLM verification
    # on the same window frames and write per-question reasoning traces.
    vlm_verify_enabled: bool = False
    vlm_verify_max_tokens: int = 8192
    vlm_verify_temperature: float = 0.0
    vlm_verify_structured_output: Literal["auto", "nim", "openai", "off"] = "openai"
    # Defaults to reasoning-only verification. When false, verification will NOT change answers;
    # it only attaches reasoning_trace (and keeps suggested_answer equal to the current answer).
    vlm_verify_apply_corrections: bool = False
    vlm_verify_prompt_file: Optional[str] = None

    @model_validator(mode="after")
    def _check_window_frames(self) -> "WindowMetadataExtractionConfig":
        if not self.single_window and self.window_frames <= 0 and self.window_seconds <= 0:
            raise ValueError(
                "when single_window=False, either window_frames > 0 (frame-based windowing) "
                "or window_seconds > 0 (time-based windowing) must be set; "
                "or set single_window=True to treat the whole media sample as one window"
            )
        return self


class McqGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal[
        "window-vlm-llm",
        "window-direct-vlm",
        "question-driven-vlm-llm",
        "metadata-llm",
    ] = "question-driven-vlm-llm"
    window_metadata_extraction: Optional[WindowMetadataExtractionConfig] = None

    @model_validator(mode="after")
    def _apply_mode_defaults(self) -> "McqGenerationConfig":
        # metadata-llm is LLM-only; VLM verify cannot run without a VLM endpoint.
        # Auto-disable so users don't need to add vlm_verify_enabled=false explicitly.
        if (
            self.mode == "metadata-llm"
            and self.window_metadata_extraction is not None
            and self.window_metadata_extraction.vlm_verify_enabled
        ):
            self.window_metadata_extraction = self.window_metadata_extraction.model_copy(
                update={"vlm_verify_enabled": False}
            )
        return self
