# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI service for the standalone super-resolution container."""

from __future__ import annotations

import argparse

from core import DataEntry
from core.interfaces import ServiceInterface
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from daft_validation import DaftValidationTask
from super_resolution.config import SeedVR2Config, SuperResolutionConfig
from super_resolution.factory import list_resolvers
from super_resolution.task import SuperResolutionTask


class SuperResolutionService(ServiceInterface):
    """Run only the super-resolution task over supplied ``DataEntry`` records."""

    def __init__(self) -> None:
        super().__init__(
            name="super_resolution_service",
            description="Run the standalone super-resolution service.",
        )

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--resolver",
            default="seedvr2",
            choices=list_resolvers(),
            help="Super-resolution resolver to run.",
        )
        parser.add_argument("--disabled", action="store_true", help="Skip SR and exit cleanly.")
        parser.add_argument(
            "--variant",
            choices=["seedvr2_3b", "seedvr2_7b"],
            default="seedvr2_3b",
            help="SeedVR2 model variant.",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--res-h", type=int, default=720)
        parser.add_argument("--res-w", type=int, default=1280)
        parser.add_argument(
            "--resolution-policy",
            choices=["always", "auto"],
            default="always",
            help=(
                "When 'always', run SR for every input. When 'auto', probe each input "
                "and skip SR when it already meets the configured minimum resolution."
            ),
        )
        parser.add_argument(
            "--min-input-short-side",
            type=int,
            default=720,
            help="Minimum short-side pixels required to skip SR when --resolution-policy=auto.",
        )
        parser.add_argument(
            "--min-input-long-side",
            type=int,
            default=1280,
            help="Minimum long-side pixels required to skip SR when --resolution-policy=auto.",
        )
        parser.add_argument("--window-frames", type=int, default=128)
        parser.add_argument("--overlap-frames", type=int, default=64)
        parser.add_argument("--out-fps", type=float, default=None)
        parser.add_argument(
            "--gpu-ids",
            default="all",
            help='GPU ids for SeedVR2, e.g. "0" or "0,1". Use "all" for visible GPUs.',
        )
        parser.add_argument("--use-multi-gpu", action="store_true")
        parser.add_argument("--model-cache-path", default=None)
        parser.add_argument("--seedvr-root", default=None)
        parser.add_argument("--allow-checkpoint-download", action="store_true")
        parser.add_argument("--keep-intermediates", action="store_true")
        parser.add_argument(
            "--empty-output-policy",
            choices=["warn", "fail"],
            default="warn",
            help="How the resolver reacts when SeedVR2 does not create output.",
        )
        parser.add_argument("--command-timeout-s", type=float, default=None)

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        if bool(getattr(args, "disabled", False)):
            self.logger.info("Skipping super-resolution: disabled")
            return

        if not data_entries:
            raise SystemExit("Pass --input or --input-file with at least one DataEntry.")

        pipeline = LinearPipeline(
            tasks=[SuperResolutionTask(config=build_config(args)), DaftValidationTask()],
            name="super_resolution_pipeline",
            policy=EmptyOutputPolicy.FAIL,
        )
        processed = pipeline.run(data_entries)
        self.logger.info("Processed %d data entries.", len(processed))


def build_config(args: argparse.Namespace) -> SuperResolutionConfig:
    """Build task config from parsed CLI args."""
    return SuperResolutionConfig(
        enabled=not bool(args.disabled),
        resolver=str(args.resolver),
        resolution_policy=args.resolution_policy,
        min_input_short_side=args.min_input_short_side,
        min_input_long_side=args.min_input_long_side,
        seedvr2=SeedVR2Config(
            variant=args.variant,
            seed=args.seed,
            res_h=args.res_h,
            res_w=args.res_w,
            window_frames=args.window_frames,
            overlap_frames=args.overlap_frames,
            out_fps=args.out_fps,
            gpu_ids=args.gpu_ids,
            use_multi_gpu=args.use_multi_gpu,
            model_cache_path=args.model_cache_path,
            seedvr_root=args.seedvr_root,
            allow_checkpoint_download=args.allow_checkpoint_download,
            keep_intermediates=args.keep_intermediates,
            empty_output_policy=args.empty_output_policy,
            command_timeout_s=args.command_timeout_s,
        ),
    )


def main() -> None:
    """Run the super-resolution service CLI."""
    SuperResolutionService().run()


if __name__ == "__main__":
    main()
