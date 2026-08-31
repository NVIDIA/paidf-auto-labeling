# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI service for referring expressions on known DAFT instances."""

from __future__ import annotations

import argparse

from core import DataEntry
from core.interfaces import ServiceInterface
from referring_expressions.config import ReferringExpressionsConfig
from referring_expressions.task import ReferringExpressionsTask


class ReferringExpressionsService(ServiceInterface):
    """Run referring expressions (boxes → VLM phrases) over image DataEntry records."""

    def __init__(self) -> None:
        super().__init__(
            name="referring_expressions_service",
            description="Run the referring-expressions service.",
        )

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--disabled", action="store_true")
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
        parser.add_argument("--frame-number", type=int, default=0)
        parser.add_argument(
            "--draw-box-overlay",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Draw numbered boxes on the image before the VLM call (default: true).",
        )
        parser.add_argument(
            "--min-match-iou",
            type=float,
            default=0.3,
            help="Minimum IoU to link a VLM region to a DAFT box when mark id is missing.",
        )
        parser.add_argument(
            "--enable-grouped-expressions",
            action="store_true",
            help="Reserved: Step 1 grouped referring expressions (not implemented in MVP).",
        )
        parser.add_argument(
            "--enable-double-check",
            action="store_true",
            help="Reserved: Step 2 verification (not implemented in MVP).",
        )

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        if bool(args.disabled):
            self.logger.info("Referring expressions disabled; skipping.")
            return
        if not data_entries:
            raise SystemExit("Pass --input or --input-file with at least one DataEntry.")
        task = ReferringExpressionsTask(config=build_config(args))
        annotated = task.run_batch(data_entries)
        self.logger.info("Processed %d data entries.", len(annotated))


def build_config(args: argparse.Namespace) -> ReferringExpressionsConfig:
    """Build task config from parsed CLI args."""
    return ReferringExpressionsConfig(
        enabled=not bool(args.disabled),
        force_reprocess=args.force_reprocess,
        vlm_endpoint_url=args.vlm_endpoint_url,
        vlm_model=args.vlm_model,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
        frame_number=args.frame_number,
        draw_box_overlay=bool(args.draw_box_overlay),
        min_match_iou=args.min_match_iou,
        enable_grouped_expressions=bool(args.enable_grouped_expressions),
        enable_double_check=bool(args.enable_double_check),
    )


def main() -> None:
    """Run the referring-expressions service CLI."""
    ReferringExpressionsService().run()


if __name__ == "__main__":
    main()
