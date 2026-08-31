# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from core import DataEntry
from core.interfaces import ServiceInterface
from training_export import (
    TRAINING_EXPORT_FORMATS,
    TRAINING_EXPORT_TASKS,
    TrainingExportConfig,
    TrainingExportTask,
)


class TrainingExportService(ServiceInterface):
    def __init__(self) -> None:
        super().__init__(
            name="training_export_service",
            description="Export completed DAFT scene directories to training datasets.",
        )

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--training-export-format",
            dest="training_export_formats",
            action="append",
            choices=TRAINING_EXPORT_FORMATS,
            default=[],
            required=True,
            help=(
                "Export completed DAFT annotations to a TAO DAFT training format. "
                "Repeat to write multiple formats."
            ),
        )
        parser.add_argument(
            "--training-export-dir",
            required=True,
            help=(
                "Output directory for training exports. Each selected format is written "
                "under a subdirectory named after the format."
            ),
        )
        parser.add_argument(
            "--training-export-task",
            dest="training_export_tasks",
            action="append",
            choices=TRAINING_EXPORT_TASKS,
            default=[],
            help="Optional DAFT task type filter for training exports. Repeat as needed.",
        )
        parser.add_argument(
            "--training-export-description",
            default=None,
            help="Optional description metadata written to exported training datasets.",
        )
        parser.add_argument(
            "--training-export-license",
            default=None,
            help="Optional license metadata written to exported training datasets.",
        )
        parser.add_argument(
            "--training-export-tag",
            dest="training_export_tags",
            action="append",
            default=[],
            help="Optional tag metadata written to exported training datasets. Repeat as needed.",
        )
        parser.add_argument(
            "--training-export-no-copy-media",
            action="store_true",
            help="Reference source raw media in-place instead of copying it into export outputs.",
        )
        parser.add_argument(
            "--training-export-emit-media-root-as-null",
            action="store_true",
            help=(
                "For tao-vl-reason-v1.0 no-copy exports, emit media_root as null so "
                "consumers can set it later."
            ),
        )

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        if not data_entries:
            raise SystemExit("Pass --input or --input-file with at least one DataEntry.")
        task = TrainingExportTask(config=training_export_config_from_args(args))
        processed = task.run_batch(data_entries)
        self.logger.info(
            "Exported training datasets from %d data entries into %s.",
            len(processed),
            args.training_export_dir,
        )


def training_export_config_from_args(args: argparse.Namespace) -> TrainingExportConfig:
    metadata: dict[str, object] = {}
    if args.training_export_description:
        metadata["description"] = str(args.training_export_description)
    if args.training_export_license:
        metadata["license"] = str(args.training_export_license)
    if args.training_export_tags:
        metadata["tags"] = list(args.training_export_tags)
    return TrainingExportConfig(
        formats=tuple(args.training_export_formats or ()),
        output_dir=args.training_export_dir,
        task_types=tuple(args.training_export_tasks or ()),
        metadata=metadata,
        copy_media=not bool(args.training_export_no_copy_media),
        emit_media_root_as_null=bool(args.training_export_emit_media_root_as_null),
    )


def main() -> None:
    TrainingExportService().run()


if __name__ == "__main__":
    main()
