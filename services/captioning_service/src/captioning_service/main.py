# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI service for the standalone captioning container."""

from __future__ import annotations

import argparse

from captioning.config import CaptioningConfig
from captioning.task import CaptioningTask
from core import DataEntry
from core.interfaces import ServiceInterface
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from daft_validation import DaftValidationTask


class CaptioningService(ServiceInterface):
    """Run the captioning task over supplied ``DataEntry`` records."""

    def __init__(self) -> None:
        super().__init__(name="captioning_service", description="Run the captioning service.")

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        captioning = parser.add_argument_group("captioning options")
        captioning.add_argument(
            "--disabled",
            action="store_true",
            help="Skip caption generation and leave each input entry unchanged.",
        )
        captioning.add_argument(
            "--input-source",
            choices=["auto", "original", "enhanced", "tracking"],
            default="auto",
            help=(
                "Media source to caption. auto prefers tracking output, then enhanced media, "
                "then the original input. enhanced/tracking require that artifact to exist."
            ),
        )
        captioning.add_argument(
            "--image-group-dir",
            default=None,
            help=(
                "Local directory recursively containing views of one identity. When set, "
                "all selected images are sent in one VLM request."
            ),
        )
        captioning.add_argument(
            "--max-group-images",
            type=int,
            default=0,
            help="Maximum image-group views to send; 0 sends all discovered images.",
        )
        captioning.add_argument(
            "--vlm-provider",
            choices=["openai-compatible", "gemini"],
            default="openai-compatible",
            help="Provider adapter for image and per-window video caption generation.",
        )
        captioning.add_argument(
            "--vlm-endpoint-url",
            default=None,
            help=(
                "Base URL for VLM requests. OpenAI-compatible endpoints accept a /v1 base "
                "or a /chat/completions URL."
            ),
        )
        captioning.add_argument(
            "--vlm-model",
            default="default",
            help="Model name sent to the VLM endpoint. Default: %(default)s.",
        )
        captioning.add_argument(
            "--enable-llm-summary",
            action="store_true",
            help="After window captioning, call an LLM to generate a video-level summary.",
        )
        captioning.add_argument(
            "--llm-provider",
            choices=["openai-compatible", "gemini"],
            default=None,
            help="Provider adapter for summary generation. Defaults to --vlm-provider.",
        )
        captioning.add_argument(
            "--llm-endpoint-url",
            default=None,
            help="Base URL for summary LLM requests. Defaults to --vlm-endpoint-url.",
        )
        captioning.add_argument(
            "--llm-model",
            default=None,
            help="Model name for summary generation. Defaults to --vlm-model.",
        )
        captioning.add_argument(
            "--prompt",
            dest="prompt_text",
            default=None,
            help="Inline prompt for video window captioning; overrides the built-in prompt.",
        )
        captioning.add_argument(
            "--prompt-file",
            default=None,
            help="Path to a prompt text file for video window captioning.",
        )
        captioning.add_argument(
            "--image-prompt",
            dest="image_prompt_text",
            default=None,
            help="Inline prompt for image captioning; overrides the built-in image prompt.",
        )
        captioning.add_argument(
            "--image-prompt-file",
            default=None,
            help="Path to a prompt text file for image captioning.",
        )
        captioning.add_argument(
            "--summary-prompt",
            dest="summary_prompt_text",
            default=None,
            help="Inline prompt for LLM summary generation.",
        )
        captioning.add_argument(
            "--summary-prompt-file",
            default=None,
            help="Path to a prompt text file for LLM summary generation.",
        )
        captioning.add_argument(
            "--system-prompt",
            default=None,
            help="Optional system prompt sent with VLM caption requests.",
        )
        captioning.add_argument(
            "--window-seconds",
            type=float,
            default=10.0,
            help="Video window duration in seconds when --window-frames is 0.",
        )
        captioning.add_argument(
            "--window-frames",
            type=int,
            default=256,
            help="Target frame count per video window. Set 0 to use --window-seconds.",
        )
        captioning.add_argument(
            "--remainder-threshold",
            type=int,
            default=128,
            help="Minimum leftover frames required to create a final partial window.",
        )
        captioning.add_argument(
            "--single-window",
            action="store_true",
            help="Caption the whole video as one window instead of splitting it.",
        )
        captioning.add_argument(
            "--sampling-fps",
            type=float,
            default=2.0,
            help="Frames-per-second sample rate when JPEG frame payloads are used.",
        )
        captioning.add_argument(
            "--max-frames",
            type=int,
            default=8,
            help="Maximum JPEG frames sent per window when frame payloads are used.",
        )
        captioning.add_argument(
            "--resolution",
            type=int,
            default=768,
            help="Longest-side pixel size for resized JPEG frame payloads.",
        )
        captioning.add_argument(
            "--media-mode",
            choices=["auto", "video", "frames"],
            default="auto",
            help=(
                "Payload mode for video windows. auto tries MP4 clips and falls back to JPEG "
                "frames; video forces MP4 clips; frames forces JPEG frames."
            ),
        )
        captioning.add_argument(
            "--max-tokens",
            type=int,
            default=1024,
            help="Maximum output tokens requested from each model call.",
        )
        captioning.add_argument(
            "--summary-input-token-budget",
            type=int,
            default=6000,
            help="Approximate input-token budget per LLM summary request.",
        )
        captioning.add_argument(
            "--temperature",
            type=float,
            default=0.2,
            help="Sampling temperature sent to each model call.",
        )
        captioning.add_argument(
            "--top-p",
            type=float,
            default=0.9,
            help="Nucleus sampling top-p value sent to each model call.",
        )
        captioning.add_argument(
            "--timeout-s",
            type=float,
            default=120.0,
            help="HTTP request timeout, in seconds, for model endpoint calls.",
        )
        captioning.add_argument(
            "--retries",
            type=int,
            default=2,
            help="Number of retries for retryable model endpoint failures.",
        )
        captioning.add_argument(
            "--retry-backoff-s",
            type=float,
            default=1.0,
            help="Base retry backoff, in seconds, between model endpoint attempts.",
        )
        captioning.add_argument(
            "--preserve-raw-model-output",
            action="store_true",
            help=(
                "Keep raw VLM responses, parsed JSON, call metadata, and input-media "
                "request metadata in per-window video metadata. By default, "
                "metadata_chunk.json stores compact extracted captions."
            ),
        )
        captioning.add_argument(
            "--no-contextual",
            action="store_true",
            help="Do not write DAFT contextual caption artifacts; only write the sidecar.",
        )
        captioning.add_argument(
            "--sidecar-filename",
            default="metadata_chunk.json",
            help="Filename for the dense video caption sidecar under sidecars/captioning.",
        )

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        if bool(getattr(args, "disabled", False)):
            self.logger.info("Skipping captioning: disabled")
            return

        if not data_entries:
            raise SystemExit("Pass --input or --input-file with at least one DataEntry.")

        pipeline = LinearPipeline(
            tasks=[CaptioningTask(config=build_config(args)), DaftValidationTask()],
            name="captioning_pipeline",
            policy=EmptyOutputPolicy.FAIL,
        )
        processed_entries = pipeline.run(data_entries)
        self.logger.info("Processed %d data entries.", len(processed_entries))


def build_config(args: argparse.Namespace) -> CaptioningConfig:
    """Build task config from parsed CLI args."""
    return CaptioningConfig(
        enabled=not bool(args.disabled),
        input_source=args.input_source,
        image_group_dir=args.image_group_dir,
        max_group_images=args.max_group_images,
        vlm_provider=args.vlm_provider,
        vlm_endpoint_url=args.vlm_endpoint_url,
        vlm_model=args.vlm_model,
        enable_llm_summary=args.enable_llm_summary,
        llm_provider=args.llm_provider,
        llm_endpoint_url=args.llm_endpoint_url,
        llm_model=args.llm_model,
        prompt_text=args.prompt_text,
        prompt_file=args.prompt_file,
        image_prompt_text=args.image_prompt_text,
        image_prompt_file=args.image_prompt_file,
        summary_prompt_text=args.summary_prompt_text,
        summary_prompt_file=args.summary_prompt_file,
        system_prompt=args.system_prompt,
        window_seconds=args.window_seconds,
        window_frames=args.window_frames,
        remainder_threshold=args.remainder_threshold,
        single_window=args.single_window,
        sampling_fps=args.sampling_fps,
        max_frames=args.max_frames,
        resolution=args.resolution,
        media_mode=args.media_mode,
        max_tokens=args.max_tokens,
        summary_input_token_budget=args.summary_input_token_budget,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
        preserve_raw_model_output=args.preserve_raw_model_output,
        write_contextual=not bool(args.no_contextual),
        sidecar_filename=args.sidecar_filename,
    )


def main() -> None:
    """Run the captioning service CLI."""
    CaptioningService().run()


if __name__ == "__main__":
    main()


__all__ = ["CaptioningService", "build_config", "main"]
