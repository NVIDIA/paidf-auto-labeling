# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from core.interfaces import ServiceInterface
from core.models import DataEntry
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from daft_validation import DaftValidationTask
from visual_qa import VisualQaConfig, VisualQaTask


class VisualQaService(ServiceInterface):
    """Run visual QA generation and normalization over scene directories."""

    def __init__(self) -> None:
        super().__init__(name="visual_qa_service", description="Run the visual QA sidecar task.")

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--generation-mode",
            choices=[
                "normalize-only",
                "window-direct-vlm",
                "window-vlm-llm",
                "question-driven-vlm-llm",
                "metadata-llm",
            ],
            default="normalize-only",
            help="How to create raw visual QA evidence before normalization.",
        )
        parser.add_argument(
            "--input-source",
            choices=["auto", "original", "enhanced", "tracking"],
            default="auto",
            help=(
                "Media source for VLM-backed visual QA. auto prefers tracking output, "
                "then enhanced media, then original input."
            ),
        )
        parser.add_argument(
            "--image-group-dir",
            default=None,
            help=(
                "Local directory recursively containing views of one identity. When set, "
                "all selected images are sent as one Visual QA window."
            ),
        )
        parser.add_argument(
            "--max-group-images",
            type=int,
            default=0,
            help="Maximum image-group views to send; 0 sends all discovered images.",
        )
        parser.add_argument(
            "--question-bank-file",
            type=str,
            default=None,
            help="Optional JSON question bank with a top-level questions array.",
        )
        parser.add_argument(
            "--track-crops-sidecar",
            type=str,
            default=None,
            help=(
                "Relative path to a detection/tracking tracks.json. When set, visual "
                "QA runs once per track over that track's crops (PAS video flow), "
                "stamping track_id into each window."
            ),
        )
        parser.add_argument(
            "--max-crops-per-track",
            type=int,
            default=8,
            help="Maximum crops sent to the VLM per track in track-crops mode.",
        )
        parser.add_argument(
            "--vlm-provider",
            choices=["openai-compatible", "gemini"],
            default="openai-compatible",
            help="Provider adapter for VLM-backed visual QA generation.",
        )
        parser.add_argument(
            "--vlm-endpoint-url",
            default=None,
            help="Base URL for VLM requests.",
        )
        parser.add_argument(
            "--vlm-model",
            default="default",
            help="Model name sent to the VLM endpoint.",
        )
        parser.add_argument(
            "--llm-provider",
            choices=["openai-compatible", "gemini"],
            default=None,
            help="Provider adapter for LLM-backed QA mapping. Defaults to --vlm-provider.",
        )
        parser.add_argument(
            "--llm-endpoint-url",
            default=None,
            help="Base URL for LLM requests. Defaults to --vlm-endpoint-url.",
        )
        parser.add_argument(
            "--llm-model",
            default=None,
            help="Model name for LLM requests. Defaults to --vlm-model.",
        )
        parser.add_argument(
            "--parser",
            choices=["instruct", "reasoning"],
            default="instruct",
            help=(
                "Reasoning mode for hybrid models. 'instruct' (default) disables thinking "
                "for directly parseable answers; 'reasoning' leaves it enabled."
            ),
        )
        parser.add_argument(
            "--evidence-prompt",
            dest="evidence_prompt_text",
            default=None,
            help="Inline VLM evidence prompt for window-vlm-llm mode.",
        )
        parser.add_argument(
            "--evidence-prompt-file",
            default=None,
            help="Path to a VLM evidence prompt file for window-vlm-llm mode.",
        )
        parser.add_argument(
            "--prompt",
            dest="prompt_text",
            default=None,
            help="Inline visual QA answer prompt.",
        )
        parser.add_argument(
            "--prompt-file",
            default=None,
            help="Path to a visual QA answer prompt file.",
        )
        parser.add_argument(
            "--system-prompt",
            default=None,
            help="Optional system prompt sent with VLM visual QA requests.",
        )
        parser.add_argument(
            "--include-reasoning",
            action="store_true",
            help="Ask generated QA evidence to include reasoning_trace fields.",
        )
        parser.add_argument(
            "--window-seconds",
            type=float,
            default=10.0,
            help="Video window duration in seconds when --window-frames is 0.",
        )
        parser.add_argument(
            "--window-frames",
            type=int,
            default=256,
            help="Target frame count per video window. Set 0 to use --window-seconds.",
        )
        parser.add_argument(
            "--remainder-threshold",
            type=int,
            default=128,
            help="Minimum leftover frames required to create a final partial window.",
        )
        parser.add_argument(
            "--single-window",
            action="store_true",
            help="Run visual QA over the whole video as one window.",
        )
        parser.add_argument(
            "--sampling-fps",
            type=float,
            default=2.0,
            help="Frames-per-second sample rate when JPEG frame payloads are used.",
        )
        parser.add_argument(
            "--max-frames",
            type=int,
            default=8,
            help="Maximum JPEG frames sent per video window.",
        )
        parser.add_argument(
            "--resolution",
            type=int,
            default=768,
            help="Longest-side pixel size for resized JPEG frame payloads.",
        )
        parser.add_argument(
            "--media-mode",
            choices=["auto", "video", "frames"],
            default="auto",
            help="Payload mode for video windows.",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=10000,
            help="Maximum output tokens requested from each model call.",
        )
        parser.add_argument(
            "--question-prompt-max-tokens",
            type=int,
            default=8192,
            help=(
                "Maximum output tokens for the deterministic question-bank-to-evidence "
                "prompt call in question-driven-vlm-llm mode."
            ),
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.2,
            help="Sampling temperature sent to each model call.",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=0.9,
            help="Nucleus sampling top-p value sent to each model call.",
        )
        parser.add_argument(
            "--timeout-s",
            type=float,
            default=120.0,
            help="HTTP request timeout, in seconds, for model endpoint calls.",
        )
        parser.add_argument(
            "--retries",
            type=int,
            default=2,
            help="Number of retries for retryable model endpoint failures.",
        )
        parser.add_argument(
            "--retry-backoff-s",
            type=float,
            default=1.0,
            help="Base retry backoff, in seconds, between model endpoint attempts.",
        )
        parser.add_argument(
            "--input-sidecar",
            action="append",
            default=None,
            help=(
                "Relative sidecar path to consume. May be passed multiple times. "
                "Defaults to visual_qa/windows.json then metadata.json."
            ),
        )
        parser.add_argument(
            "--metadata-input-sidecar",
            action="append",
            default=None,
            help=(
                "Relative sidecar path to consume in metadata-llm mode. May be passed "
                "multiple times."
            ),
        )
        parser.add_argument(
            "--raw-windows-sidecar",
            type=str,
            default="visual_qa/windows.json",
            help="Relative sidecar path for generated raw visual QA evidence.",
        )
        parser.add_argument(
            "--output-items-sidecar",
            type=str,
            default="visual_qa/items.json",
            help="Relative sidecar path for normalized visual QA items.",
        )
        parser.add_argument(
            "--output-windows-sidecar",
            type=str,
            default="visual_qa/windows.normalized.json",
            help="Relative sidecar path for normalized per-window visual QA details.",
        )
        parser.add_argument(
            "--state-artifacts-key",
            type=str,
            default=None,
            help=(
                "Key under which this run records its slice in "
                "pipeline_state.task_artifacts (default 'visual_qa'). Set a "
                "distinct value per pass (e.g. 'visual_qa_anomaly', "
                "'visual_qa_anomaly_search') so multiple visual_qa passes sharing "
                "one output directory each keep their provenance. The canonical "
                "'visual_qa' key is always also written as the latest pointer."
            ),
        )
        parser.add_argument(
            "--no-aggregate-windows",
            action="store_true",
            help="Use the first answer window instead of aggregating answers across windows.",
        )
        parser.add_argument(
            "--allow-out-of-bank-answers",
            action="store_true",
            help="Keep closed-choice answers even when they do not match bank options.",
        )
        parser.add_argument(
            "--open-ended",
            action="store_true",
            help="Flatten closed-choice answers to open-ended text and clear options.",
        )
        parser.add_argument(
            "--no-flat-qa-tasks",
            action="store_true",
            help=(
                "Do not write the flat DAFT task files (mcq.json/bcq.json/open_qa.json). "
                "persona_qa.json is unaffected. Use in passes whose flat QA is redundant "
                "or owned by a dedicated later pass."
            ),
        )

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        config = build_config(args)
        pipeline = LinearPipeline(
            tasks=[VisualQaTask(config=config), DaftValidationTask()],
            name="visual_qa_pipeline",
            policy=EmptyOutputPolicy.FAIL,
        )
        processed = pipeline.run(data_entries)
        self.logger.info("Processed %d data entries.", len(processed))


def build_config(args: argparse.Namespace) -> VisualQaConfig:
    """Build visual QA task config from parsed CLI args."""
    defaults = VisualQaConfig()
    input_sidecars = tuple(args.input_sidecar) if args.input_sidecar else defaults.input_sidecars
    metadata_input_sidecars = (
        tuple(args.metadata_input_sidecar)
        if args.metadata_input_sidecar
        else defaults.metadata_input_sidecars
    )
    return VisualQaConfig(
        generation_mode=args.generation_mode,
        input_source=args.input_source,
        image_group_dir=args.image_group_dir,
        max_group_images=args.max_group_images,
        question_bank_file=args.question_bank_file,
        track_crops_sidecar=args.track_crops_sidecar,
        max_crops_per_track=args.max_crops_per_track,
        input_sidecars=input_sidecars,
        metadata_input_sidecars=metadata_input_sidecars,
        raw_windows_sidecar=args.raw_windows_sidecar,
        output_items_sidecar=args.output_items_sidecar,
        output_windows_sidecar=args.output_windows_sidecar,
        state_artifacts_key=args.state_artifacts_key,
        vlm_provider=args.vlm_provider,
        vlm_endpoint_url=args.vlm_endpoint_url,
        vlm_model=args.vlm_model,
        llm_provider=args.llm_provider,
        llm_endpoint_url=args.llm_endpoint_url,
        llm_model=args.llm_model,
        parser=args.parser,
        evidence_prompt_text=args.evidence_prompt_text,
        evidence_prompt_file=args.evidence_prompt_file,
        prompt_text=args.prompt_text,
        prompt_file=args.prompt_file,
        system_prompt=args.system_prompt,
        include_reasoning=bool(args.include_reasoning),
        window_seconds=args.window_seconds,
        window_frames=args.window_frames,
        remainder_threshold=args.remainder_threshold,
        single_window=bool(args.single_window),
        sampling_fps=args.sampling_fps,
        max_frames=args.max_frames,
        resolution=args.resolution,
        media_mode=args.media_mode,
        max_tokens=args.max_tokens,
        question_prompt_max_tokens=args.question_prompt_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
        aggregate_windows=not bool(args.no_aggregate_windows),
        strict_answers=not bool(args.allow_out_of_bank_answers),
        open_ended=bool(args.open_ended),
        emit_flat_qa_tasks=not bool(args.no_flat_qa_tasks),
    )


def main() -> None:
    VisualQaService().run()


if __name__ == "__main__":
    main()
