# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container workflow runner service."""

from __future__ import annotations

import argparse

from core import DataEntry, EmptyOutputPolicy, coerce_policy
from core.interfaces import ServiceInterface
from training_export import (
    COSMOS_REASON_VERSION,
    SUPPORTED_TRAINING_TASKS,
    TAO_VL_REASON_FORMAT,
)

from workflow_runner.container_runner import (
    DEFAULT_BUILD_TARGETS,
    DEFAULT_PIPELINE,
    DEFAULT_STAGE_IMAGES,
    SELECTABLE_STAGES,
    ContainerPipelineRunner,
    build_container_plan,
    default_tracking_build_target,
    default_tracking_image,
    normalize_entries_for_container,
    resolve_container_user,
    rootless_env_with_defaults,
    runner_config_from_args,
    write_runner_input_file,
)
from workflow_runner.cookbook import (
    apply_cookbook_config,
    expand_directory_entries,
    explicit_cli_options,
)

TRAINING_EXPORT_FORMATS = (COSMOS_REASON_VERSION, TAO_VL_REASON_FORMAT)
TRAINING_EXPORT_TASKS = tuple(sorted(SUPPORTED_TRAINING_TASKS))


class WorkflowRunnerService(ServiceInterface):
    """Run Dockerized stage services as a linear workflow."""

    def __init__(self) -> None:
        super().__init__(
            name="workflow_runner",
            description="Run Dockerized workflow stages in sequence.",
        )

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        self._add_stage_flags(parser)
        self._add_container_flags(parser)
        self._add_image_and_build_flags(parser)
        self._add_model_runtime_flags(parser)
        self._add_reasoning_flags(parser)
        self._add_training_export_flags(parser)

    def build_parser(self) -> argparse.ArgumentParser:
        """Build the full CLI parser with common and service-specific flags."""
        parser = argparse.ArgumentParser(description=self.description)
        self._add_common_args(parser)
        self.add_service_args(parser)
        return parser

    def _add_stage_flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--stages",
            nargs="+",
            choices=SELECTABLE_STAGES,
            default=None,
            help=(
                "Stages to enable. The runner executes enabled stages in canonical dataflow "
                "order so sidecar handoffs are produced before consumers run."
            ),
        )
        parser.add_argument(
            "--pipeline",
            choices=["video", "image"],
            default=None,
            help=(
                "Default container pipeline to use when --stages is omitted. "
                f"Default: {DEFAULT_PIPELINE}."
            ),
        )
        parser.add_argument(
            "--cookbook-file",
            default=None,
            help=(
                "Optional scenario YAML/JSON config. Supports the cookbook layout under "
                "cookbooks/ at the repository root."
            ),
        )
        parser.add_argument(
            "--no-expand-input-dirs",
            action="store_true",
            help="Do not expand local directory media inputs into one DataEntry per file.",
        )
        parser.add_argument(
            "--policy",
            choices=[EmptyOutputPolicy.WARN.value, EmptyOutputPolicy.FAIL.value],
            default=EmptyOutputPolicy.FAIL.value,
            help="How to react to a failed stage container. Default: fail.",
        )

    def _add_container_flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--container-runtime",
            default="docker",
            help="Container executable to use, usually docker or podman.",
        )
        parser.add_argument(
            "--container-gpus",
            default="all",
            help='GPU request passed to Docker/Podman. Use "none" to omit --gpus.',
        )
        parser.add_argument(
            "--container-network",
            default="host",
            help='Container network mode. Use "none" to omit --network.',
        )
        parser.add_argument(
            "--container-workdir",
            default=None,
            help="Working directory inside each stage container.",
        )
        parser.add_argument(
            "--container-user",
            default=None,
            help=(
                'Container user passed as Docker/Podman --user, e.g. "$(id -u):$(id -g)". '
                'Use "none" to omit.'
            ),
        )
        parser.add_argument(
            "--container-env",
            action="append",
            default=[],
            metavar="NAME",
            help="Environment variable name to pass through to every stage container.",
        )
        parser.add_argument(
            "--container-mount",
            action="append",
            default=[],
            metavar="HOST[:CONTAINER[:ro|rw]]",
            help=(
                "Additional bind mount for every stage. DataEntry media/data paths and "
                "runner-generated input manifests are mounted automatically."
            ),
        )
        parser.add_argument(
            "--container-build-images",
            action="store_true",
            help="Build each enabled stage image before running containers.",
        )
        parser.add_argument(
            "--container-ensure-images",
            action="store_true",
            help=(
                "Build only the enabled stage images that are missing before "
                "running containers; images that already exist are left "
                "untouched. Ignored when --container-build-images is set."
            ),
        )
        parser.add_argument(
            "--container-dry-run",
            action="store_true",
            help="Log the container plan without building or running containers.",
        )
        parser.add_argument(
            "--stage-arg",
            action="append",
            default=[],
            metavar="STAGE=ARG",
            help=(
                "Pass one raw arg through to one stage. Repeat for flags and values, e.g. "
                "--stage-arg captioning=--window-seconds --stage-arg captioning=15."
            ),
        )

    def _add_image_and_build_flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sr-image",
            default=None,
            help="Override the super-resolution stage image.",
        )
        parser.add_argument(
            "--tracking-image",
            default=None,
            help="Override the detection/tracking stage image.",
        )
        parser.add_argument(
            "--captioning-image",
            default=None,
            help="Override the captioning stage image.",
        )
        parser.add_argument(
            "--visual-qa-image",
            default=None,
            help="Override the visual-QA stage image.",
        )
        parser.add_argument(
            "--reasoning-image",
            default=None,
            help="Override the reasoning stage image.",
        )
        parser.add_argument(
            "--training-export-image",
            default=None,
            help="Override the training-export stage image.",
        )
        parser.add_argument(
            "--pas-image",
            default=None,
            help="Override the event-and-person-attribute-search stage image.",
        )
        parser.add_argument(
            "--referring-expressions-image",
            default=None,
            help="Override the referring-expressions stage image.",
        )
        parser.add_argument(
            "--grounding-2d-image",
            default=None,
            help="Override the 2D grounding stage image.",
        )
        parser.add_argument(
            "--sr-build-target",
            default=None,
            help="Override make build IMAGE target for super-resolution.",
        )
        parser.add_argument(
            "--tracking-build-target",
            default=None,
            help="Override make build IMAGE target for detection/tracking.",
        )
        parser.add_argument(
            "--captioning-build-target",
            default=None,
            help="Override make build IMAGE target for captioning.",
        )
        parser.add_argument(
            "--visual-qa-build-target",
            default=None,
            help="Override make build IMAGE target for visual-QA.",
        )
        parser.add_argument(
            "--reasoning-build-target",
            default=None,
            help="Override make build IMAGE target for reasoning.",
        )
        parser.add_argument(
            "--training-export-build-target",
            default=None,
            help="Override make build IMAGE target for training export.",
        )
        parser.add_argument(
            "--pas-build-target",
            default=None,
            help="Override make build IMAGE target for event-and-person-attribute-search.",
        )
        parser.add_argument(
            "--referring-expressions-build-target",
            default=None,
            help="Override make build IMAGE target for referring expressions.",
        )
        parser.add_argument(
            "--grounding-2d-build-target",
            default=None,
            help="Override make build IMAGE target for 2D grounding.",
        )

    def _add_model_runtime_flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sr-resolver",
            default="seedvr2",
            help="Super-resolution backend forwarded to the SR service.",
        )
        parser.add_argument(
            "--sr-variant",
            default=None,
            help="Optional super-resolution model variant forwarded to the SR service.",
        )
        parser.add_argument(
            "--tracker",
            default="rfdetr-boosttrack",
            help="Detection/tracking backend, e.g. rfdetr-boosttrack or sam3.",
        )
        parser.add_argument(
            "--classes",
            nargs="*",
            default=[],
            help="Optional class names forwarded to detection/tracking.",
        )
        parser.add_argument(
            "--model-cache-path",
            default=None,
            help="Host path mounted and forwarded for model caches/checkpoints.",
        )
        parser.add_argument(
            "--gpu-ids",
            default=None,
            help='GPU ids forwarded to model stages, e.g. "0" or "0,1".',
        )
        parser.add_argument(
            "--vlm-endpoint-url",
            default=None,
            help="VLM endpoint URL forwarded to captioning.",
        )
        parser.add_argument(
            "--vlm-model",
            default=None,
            help="VLM model name forwarded to captioning.",
        )
        parser.add_argument(
            "--llm-endpoint-url",
            default=None,
            help="LLM endpoint URL forwarded to captioning/reasoning stages.",
        )
        parser.add_argument(
            "--llm-model",
            default=None,
            help="LLM model name forwarded to captioning/reasoning stages.",
        )
        parser.add_argument(
            "--question-bank-file",
            default=None,
            help="Question bank forwarded to visual-QA.",
        )
        parser.add_argument(
            "--reasoning-config-file",
            "--daft-config-file",
            dest="reasoning_config_file",
            default=None,
            help=(
                "Reasoning config file forwarded to the reasoning stage. "
                "(--daft-config-file is accepted as a legacy alias.)"
            ),
        )
        parser.add_argument(
            "--pas-config-file",
            default=None,
            help="Optional config file forwarded to the event-and-person-attribute-search stage.",
        )

    def _add_reasoning_flags(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--reasoning-mode",
            choices=["config", "keep", "strip"],
            default="config",
            help="Reasoning trace handling forwarded to the reasoning service.",
        )

    def _add_training_export_flags(self, parser: argparse.ArgumentParser) -> None:
        export = parser.add_argument_group("training export options")
        export.add_argument(
            "--training-export-format",
            dest="training_export_formats",
            action="append",
            choices=TRAINING_EXPORT_FORMATS,
            default=[],
            help=(
                "Export completed Metropolis scene outputs to a TAO DAFT training format. "
                "Repeat to write multiple formats."
            ),
        )
        export.add_argument(
            "--training-export-dir",
            default=None,
            help=(
                "Output directory for training exports. Each selected format is written "
                "under a subdirectory named after the format."
            ),
        )
        export.add_argument(
            "--training-export-task",
            dest="training_export_tasks",
            action="append",
            choices=TRAINING_EXPORT_TASKS,
            default=[],
            help="Optional Metropolis task type filter for training exports. Repeat as needed.",
        )
        export.add_argument(
            "--training-export-description",
            default=None,
            help="Optional description metadata written to exported training datasets.",
        )
        export.add_argument(
            "--training-export-license",
            default=None,
            help="Optional license metadata written to exported training datasets.",
        )
        export.add_argument(
            "--training-export-tag",
            dest="training_export_tags",
            action="append",
            default=[],
            help="Optional tag metadata written to exported training datasets. Repeat as needed.",
        )
        export.add_argument(
            "--training-export-no-copy-media",
            action="store_true",
            help="Reference source raw media in-place instead of copying it into export outputs.",
        )
        export.add_argument(
            "--training-export-emit-media-root-as-null",
            action="store_true",
            help=(
                "For tao-vl-reason-v1.0 no-copy exports, emit media_root as null so "
                "consumers can set it later."
            ),
        )

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        protected_options = explicit_cli_options()
        cookbook_entries = apply_cookbook_config(args, protected_options=protected_options)
        if not data_entries:
            data_entries = cookbook_entries
            if data_entries and args.dev_data_root is not None:
                data_entries = self._copy_data_entries_to_dev_root(
                    data_entries,
                    args.dev_data_root,
                )
        if not bool(args.no_expand_input_dirs):
            data_entries = expand_directory_entries(
                data_entries,
                pipeline=args.pipeline,
            )
        if not data_entries:
            raise SystemExit("Pass --input or --input-file with at least one DataEntry.")

        _apply_container_defaults(args)
        container_entries = normalize_entries_for_container(data_entries)
        input_file, tempdir = write_runner_input_file(container_entries)
        try:
            plan = build_container_plan(args, container_entries, input_file=input_file)
            runner = ContainerPipelineRunner(
                config=runner_config_from_args(args, plan.mounts),
                policy=coerce_policy(args.policy, default=EmptyOutputPolicy.FAIL),
                logger=self.logger,
            )
            results = runner.run(list(plan.stages))
            successes = sum(1 for result in results if result.success)
            failures = sum(1 for result in results if not result.success)
            self.logger.info(
                "Container pipeline finished: %s stage successes, %s stage failures.",
                successes,
                failures,
            )
        finally:
            tempdir.cleanup()


def _apply_container_defaults(args: argparse.Namespace) -> None:
    """Fill image/build defaults after CLI parsing."""
    args.sr_image = args.sr_image or DEFAULT_STAGE_IMAGES["super_resolution"]
    args.tracking_image = args.tracking_image or default_tracking_image(str(args.tracker))
    args.captioning_image = args.captioning_image or DEFAULT_STAGE_IMAGES["captioning"]
    args.visual_qa_image = args.visual_qa_image or DEFAULT_STAGE_IMAGES["visual_qa"]
    args.reasoning_image = args.reasoning_image or DEFAULT_STAGE_IMAGES["reasoning"]
    args.training_export_image = (
        args.training_export_image or DEFAULT_STAGE_IMAGES["training_export"]
    )
    args.pas_image = args.pas_image or DEFAULT_STAGE_IMAGES["person_attribute_search"]
    args.referring_expressions_image = (
        args.referring_expressions_image or DEFAULT_STAGE_IMAGES["referring_expressions"]
    )
    args.grounding_2d_image = args.grounding_2d_image or DEFAULT_STAGE_IMAGES["grounding_2d"]
    args.sr_build_target = args.sr_build_target or DEFAULT_BUILD_TARGETS["super_resolution"]
    args.tracking_build_target = args.tracking_build_target or default_tracking_build_target(
        str(args.tracker)
    )
    args.captioning_build_target = (
        args.captioning_build_target or DEFAULT_BUILD_TARGETS["captioning"]
    )
    args.visual_qa_build_target = args.visual_qa_build_target or DEFAULT_BUILD_TARGETS["visual_qa"]
    args.reasoning_build_target = args.reasoning_build_target or DEFAULT_BUILD_TARGETS["reasoning"]
    args.training_export_build_target = (
        args.training_export_build_target or DEFAULT_BUILD_TARGETS["training_export"]
    )
    args.pas_build_target = (
        args.pas_build_target or DEFAULT_BUILD_TARGETS["person_attribute_search"]
    )
    args.referring_expressions_build_target = (
        args.referring_expressions_build_target or DEFAULT_BUILD_TARGETS["referring_expressions"]
    )
    args.grounding_2d_build_target = (
        args.grounding_2d_build_target or DEFAULT_BUILD_TARGETS["grounding_2d"]
    )
    # Resolve the "auto" user sentinel to the invoking uid:gid, then inject the
    # rootless hygiene env so non-root runs work without the caller listing five
    # --container-env flags by hand.
    args.container_user = resolve_container_user(args.container_user)
    args.container_env = list(
        rootless_env_with_defaults(args.container_user, tuple(args.container_env))
    )


def main() -> None:
    WorkflowRunnerService().run()


if __name__ == "__main__":
    main()
