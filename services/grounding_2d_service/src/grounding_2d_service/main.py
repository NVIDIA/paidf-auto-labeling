# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI service for the standalone 2D grounding task."""

from __future__ import annotations

import argparse

from core import DataEntry
from core.interfaces import ServiceInterface
from grounding_2d.config import Grounding2DConfig
from grounding_2d.task import Grounding2DTask


class Grounding2DService(ServiceInterface):
    """Run 2D grounding (VLM expressions + SAM3) over image DataEntry records."""

    def __init__(self) -> None:
        super().__init__(
            name="grounding_2d_service",
            description="Run the standalone 2D grounding service.",
        )

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--disabled", action="store_true", help="Skip 2D grounding and exit.")
        parser.add_argument(
            "--caption",
            default=None,
            help=(
                "Caption applied to all DataEntry inputs. Prefer per-scene "
                "sidecars/input.json for dataset runs."
            ),
        )
        parser.add_argument(
            "--input-metadata-filename",
            default="input.json",
            help="Sidecar metadata filename containing caption for native DataEntry inputs.",
        )
        parser.add_argument("--force-reprocess", action="store_true")
        parser.add_argument("--vlm-endpoint-url", default=None)
        parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
        parser.add_argument("--system-prompt", default=None)
        parser.add_argument("--max-tokens", type=int, default=4096)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--top-p", type=float, default=0.9)
        parser.add_argument("--timeout-s", type=float, default=120.0)
        parser.add_argument("--retries", type=int, default=2)
        parser.add_argument("--retry-backoff-s", type=float, default=1.0)
        parser.add_argument("--max-instances-per-expression", type=int, default=10)
        parser.add_argument(
            "--filter-ungroundable-expressions",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Apply VLM groundable flags, optional JSON deny policy, and SAM3 "
                "text-length limits before prompting SAM3 (default: true)."
            ),
        )
        parser.add_argument(
            "--expression-filter-policy-path",
            default=None,
            help=(
                "Optional JSON deny-list policy "
                "(deny_phrases / deny_nouns). Empty lists = no keyword assumptions."
            ),
        )
        parser.add_argument(
            "--min-instance-score",
            type=float,
            default=0.5,
            help="Minimum SAM3 detection score to keep an instance (default: 0.5).",
        )
        parser.add_argument(
            "--min-bbox-area",
            type=int,
            default=64,
            help="Minimum bbox area in pixels to keep an instance (default: 64).",
        )
        parser.add_argument("--sam3-model-cache-path", default=None)
        parser.add_argument("--sam3-gpu-ids", default="all")
        parser.add_argument(
            "--sam3-version",
            choices=["sam3", "sam3.1"],
            default="sam3",
            help="SAM3 model family. sam3.1 selects the native Object Multiplex path.",
        )
        parser.add_argument(
            "--sam3-runtime",
            choices=["auto", "transformers", "native"],
            default="auto",
            help="SAM3 inference stack. auto selects native for sam3.1.",
        )
        parser.add_argument("--sam3-target-fps", type=float, default=10.0)
        parser.add_argument("--sam3-session-reset-s", type=float, default=10.0)
        parser.add_argument("--sam3-max-duration-s", type=float, default=30.0)
        parser.add_argument("--sam3-write-annotated-media", action="store_true")
        parser.add_argument(
            "--sam3-annotated-media-label-style",
            choices=["id", "name", "none"],
            default="name",
        )
        parser.add_argument("--sam3-annotated-media-mask-opacity", type=int, default=0)

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        if bool(args.disabled):
            self.logger.info("2D grounding disabled; skipping task creation and run.")
            return

        if not data_entries:
            raise SystemExit("Pass --input or --input-file with at least one DataEntry.")

        task = Grounding2DTask(config=build_config(args))
        annotated = task.run_batch(data_entries)
        self.logger.info("Processed %d data entries.", len(annotated))


def build_config(args: argparse.Namespace) -> Grounding2DConfig:
    """Build task config from parsed CLI args."""
    return Grounding2DConfig(
        enabled=not bool(args.disabled),
        force_reprocess=args.force_reprocess,
        caption=args.caption,
        input_metadata_filename=args.input_metadata_filename,
        vlm_endpoint_url=args.vlm_endpoint_url,
        vlm_model=args.vlm_model,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
        max_instances_per_expression=args.max_instances_per_expression,
        filter_ungroundable_expressions=bool(args.filter_ungroundable_expressions),
        expression_filter_policy_path=args.expression_filter_policy_path,
        min_instance_score=args.min_instance_score,
        min_bbox_area=args.min_bbox_area,
        sam3_model_cache_path=args.sam3_model_cache_path,
        sam3_gpu_ids=args.sam3_gpu_ids,
        sam3_version=args.sam3_version,
        sam3_runtime=args.sam3_runtime,
        sam3_target_fps=args.sam3_target_fps,
        sam3_session_reset_s=args.sam3_session_reset_s,
        sam3_max_duration_s=args.sam3_max_duration_s,
        sam3_write_annotated_media=args.sam3_write_annotated_media,
        sam3_annotated_media_label_style=args.sam3_annotated_media_label_style,
        sam3_annotated_media_mask_opacity=args.sam3_annotated_media_mask_opacity,
    )


def main() -> None:
    """Run the 2D grounding service CLI."""
    Grounding2DService().run()


if __name__ == "__main__":
    main()
