# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains the interface to be implemented by all services. It provides:
    - Common CLI arguments to be used by all services.
    - Parsing and handling of the common CLI arguments.
    - A base class for service implementations.

The shared CLI arguments are:
    - ``--log-level``: Set the logging level. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
      Default: INFO.
    - ``--input``: The data entries to be annotated (mutually exclusive with ``--input-file``).
    - ``--input-file``: JSONL data entries to annotate (mutually exclusive with ``--input``).
    - ``--dev-data-root``: Copy each input DAFT directory to ``PATH/<data_entry.id>`` and run
      against that copy. The path may be local or remote. Missing local input directories are
      treated as empty scenes.

To create a new service, subclass the ServiceInterface and implement the execute method.
Optionally, override the add_service_args method to add service-specific CLI arguments.

See the example service in `services/example_service` for a complete example.
"""

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod

from core.cost_performance import (
    ModelUsageSnapshot,
    collect_model_usage,
    write_cost_performance_report,
)
from core.models import DataEntry
from core.utils import telemetry
from core.utils.dev_data_root import DevDataRootCopier
from core.utils.io import read_jsonl
from core.utils.logging import LOG_LEVELS, apply_log_level, get_logger


class ServiceInterface(ABC):
    """
    Abstract base class for annotation service entrypoints.

    Subclass this and implement ``execute()``. Override ``add_service_args()`` to add
    service-specific CLI arguments on top of the shared CLI arguments.
    """

    logger: logging.Logger

    def __init__(self, name: str | None = None, description: str | None = None) -> None:
        """Initialize the service.
        Args:
            name: The name of the service, used for logging.
            description: The description of the service, used in the CLI help message.
        """
        self.name = name or self.__class__.__name__
        self.description = description or f"Run the {self.name} service."
        self.logger = get_logger(self.name, kind="service")

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        """
        Override to add service-specific CLI arguments to the parser.
        Args:
            parser: The argument parser to add the service-specific arguments to.
        """
        return

    @abstractmethod
    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        """
        This method is called by the service to execute the pipeline against the data entries
        provided by the user.
        Args:
            args: The parsed command line arguments.
            data_entries: The data entries to be processed by the pipeline.
        """
        ...

    def run(self) -> None:
        """
        Entrypoint for the service.

        Parses common and service-specific arguments, loads any requested data entries, optionally
        copies DAFT directories into a dev root, and calls ``execute()``.
        """
        parser = argparse.ArgumentParser(description=self.description)
        self._add_common_args(parser)
        self.add_service_args(parser)
        args = parser.parse_args()
        apply_log_level(level=args.log_level)
        telemetry.setup_telemetry(service_name=self.name)
        self.logger.info(f"Starting Service {self.name} with log level {args.log_level}.")
        with telemetry.start_span("service.run") as span:
            data_entries = self._get_data_entries(args)
            telemetry.set_attributes(span, {"entry.count": len(data_entries)})
            if args.dev_data_root is not None:
                data_entries = self._copy_data_entries_to_dev_root(
                    data_entries,
                    args.dev_data_root,
                )
            started_at = time.perf_counter()
            with collect_model_usage() as model_usage:
                self.execute(args, data_entries)
            self._write_generic_cost_performance_reports(
                data_entries,
                model_usage=model_usage.snapshot(),
                service_elapsed_s=time.perf_counter() - started_at,
            )

    def _write_generic_cost_performance_reports(
        self,
        data_entries: list[DataEntry],
        *,
        model_usage: ModelUsageSnapshot,
        service_elapsed_s: float,
    ) -> None:
        """
        Emit a shared cost/performance summary for services that made model calls.

        Reports go through OpenTelemetry rather than a fixed scene sidecar so
        parallel workers on shared remote storage cannot overwrite each other.
        Services with richer product-specific reporting should call
        ``write_cost_performance_report`` themselves from ``execute``.
        """
        for data_entry in data_entries:
            entry_usage = model_usage.for_entry(data_entry.id)
            if not entry_usage.calls:
                continue
            try:
                write_cost_performance_report(
                    data_entry,
                    service_name=self.name,
                    service_elapsed_s=service_elapsed_s,
                    model_usage=entry_usage,
                    notes=[
                        "Generic service-level report emitted by ServiceInterface.",
                        "Task-level timing and scene scale require service-specific enrichment.",
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - observability must not break outputs
                self.logger.warning(
                    "Failed to emit cost/performance report for %s: %s",
                    data_entry.data_path,
                    exc,
                )

    def _add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """
        Add the common CLI arguments to the parser to create a consistent CLI for all services.
        Args:
            parser: The argument parser to add the common arguments to.
        """
        parser.add_argument(
            "--log-level",
            dest="log_level",
            type=str,
            action="store",
            help=("Logging level. Must be one of: " + ", ".join(LOG_LEVELS) + ". Default: INFO."),
            default="INFO",
            choices=LOG_LEVELS,
        )
        # Keep input optional so subclasses can define additional sources.
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--input", type=str, help="The data entries to be annotated.")
        group.add_argument("--input-file", type=str, help="The input file to be annotated.")
        parser.add_argument(
            "--dev-data-root",
            dest="dev_data_root",
            type=str,
            help=(
                "Copy each input DAFT data directory to PATH/<data_entry.id> and run against "
                "that copy. PATH may be local or remote. Missing local data directories start "
                "as empty scenes. Existing ID-named directories under PATH are overwritten."
                "Use in development to preserve the source DAFT directory when testing services."
            ),
            default=None,
        )

    def _get_data_entries(self, args: argparse.Namespace) -> list[DataEntry]:
        """
        Parses the data entries from the command line arguments and validates them.
        Args:
            args: The parsed command line arguments.
        Returns:
            A list of data entries to be processed by the pipeline.
        """
        if args.input is not None:
            entries = self._data_entries_from_input_json(args.input)
        elif args.input_file is not None:
            entries = self._data_entries_from_jsonl_path(args.input_file)
        else:
            entries = []

        self.logger.info(f"Found {len(entries)} data entries to process.")
        self.logger.debug(f"First 3 entries: {entries[:3]}")
        return entries

    def _copy_data_entries_to_dev_root(
        self,
        data_entries: list[DataEntry],
        dev_data_root: str,
    ) -> list[DataEntry]:
        """
        Copy input DAFT directories to a dev root and rewrite ``data_path``.

        Each entry is copied into ``dev_data_root/<data_entry.id>``. If a local source
        ``data_path`` does not exist, the destination is created empty so the pipeline can seed a
        new DAFT scene there. Existing destination directories are replaced after every entry has
        been validated, so duplicate IDs or invalid local sources fail before any destination is
        overwritten.

        Args:
            data_entries: Input entries whose DAFT directories should be copied.
            dev_data_root: Local or remote directory that will contain one child directory per
                entry ID.

        Returns:
            Data entries with ``data_path`` rewritten to their dev-copy directories.

        Raises:
            ValueError: If an ID is duplicate or unsafe, a local source is not a directory, or a
                destination path is invalid.
        """
        return DevDataRootCopier(self.logger).copy_entries(data_entries, dev_data_root)

    def _data_entry_from_record(self, record: object) -> DataEntry:
        """
        Converts a JSON record into a data entry object.
        Args:
            record: The JSON record to convert into a data entry.
        Returns:
            A data entry object.
        """
        if not isinstance(record, dict):
            msg = f"Each entry must be a JSON object, got {type(record).__name__}"
            raise ValueError(msg)
        data_entry: DataEntry = DataEntry.model_validate(record)
        return data_entry

    def _data_entries_from_input_json(self, payload: str) -> list[DataEntry]:
        """
        Parses the data entries from the --input argument and validates them.
        Args:
            payload: The JSON payload to parse into a list of data entries.
        Returns:
            A list of data entries to be processed by the pipeline.
        """
        decoded = json.loads(payload)
        if not isinstance(decoded, list):
            msg = "--input must be a JSON array of objects."
            raise ValueError(msg)
        return [self._data_entry_from_record(item) for item in decoded]

    def _data_entries_from_jsonl_path(self, path_str: str) -> list[DataEntry]:
        """
        Parses the data entries from the --input-file argument and validates them.
        Args:
            path_str: The path to the JSONL file to parse into a list of data entries.
            The JSONL file should contain one JSON object per line with ``media_path`` and
            ``data_path`` fields.
        Returns:
            A list of data entries to be processed by the pipeline.
        """
        return [self._data_entry_from_record(record) for record in read_jsonl(path_str)]
