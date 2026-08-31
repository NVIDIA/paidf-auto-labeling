# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reusable task base classes.

The interfaces package defines the shape that pipelines depend on. This module contains concrete
task abstractions that implement a particular batching policy while leaving the per-entry
annotation logic to individual task packages.
"""

from typing import override

from core.cost_performance import model_usage_entry
from core.exceptions import InvalidInputError, RetryTaskError
from core.interfaces.task import TaskInterface
from core.models import DataEntry
from core.utils import telemetry


class SequentialTask(TaskInterface):
    """
    Base class for tasks that process data entries one at a time.

    Subclasses implement ``run`` for a single ``DataEntry``. ``SequentialTask`` supplies
    ``run_batch`` by iterating through the incoming list in order and applying the framework's
    standard per-entry error handling:

    - ``RetryTaskError`` retries the same task against the same data entry up to ``max_retries``
      times, then re-raises the final error.
    - ``InvalidInputError`` skips the current data entry and continues processing the rest of the
      batch.
    - Any other exception is treated as fatal and propagates to the caller.

    Tasks with model-native batch inference, custom parallelism, or different failure semantics
    should implement ``TaskInterface`` directly or subclass another batching base class.
    """

    max_retries: int

    def __init__(self, name: str | None = None, max_retries: int = 0) -> None:
        """
        Initialize a sequential task.

        Args:
            name: Optional human-readable name used in logs. Defaults to the class name.
            max_retries: Number of times to retry ``run`` after a ``RetryTaskError`` for a single
                data entry.
        Raises:
            ValueError: If ``max_retries`` is negative.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")

        super().__init__(name=name)
        self.max_retries = max_retries

    @override
    def run_batch(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        """
        Annotate data entries sequentially.

        Args:
            data_entries: The data entries to be annotated.
        Returns:
            The successfully annotated data entries, preserving input order for entries that were
            not skipped.
        """
        annotated_data_entries = []
        for data_entry in data_entries:
            with telemetry.start_span(
                "task.run_entry",
                {"task.name": self.name, "entry.id": data_entry.id},
            ):
                telemetry.record_input_frames(self.name, data_entry.media_path)
                try:
                    with model_usage_entry(data_entry):
                        annotated_data_entry = self._run_with_retries(data_entry)
                except InvalidInputError as error:
                    self.logger.warning(
                        "Skipping invalid data entry %s in task %s: %s",
                        data_entry.media_path,
                        self.name,
                        error,
                    )
                    continue
                annotated_data_entries.append(annotated_data_entry)
        return annotated_data_entries

    def _run_with_retries(self, data_entry: DataEntry) -> DataEntry:
        """
        Run this task against a single data entry with sequential retry semantics.

        Args:
            data_entry: The data entry to annotate.
        Returns:
            The annotated data entry.
        Raises:
            RetryTaskError: If ``run`` continues to raise ``RetryTaskError`` after all retries have
                been exhausted.
        """
        attempts = 0
        while True:
            try:
                with telemetry.start_span(
                    "task.run_attempt",
                    {"task.name": self.name, "entry.id": data_entry.id, "attempt": attempts},
                ):
                    return self.run(data_entry)
            except RetryTaskError as error:
                if attempts >= self.max_retries:
                    self.logger.error(
                        "Task %s exhausted %d retries for data entry %s: %s",
                        self.name,
                        self.max_retries,
                        data_entry.media_path,
                        error,
                    )
                    raise

                attempts += 1
                self.logger.warning(
                    "Retrying task %s for data entry %s after transient failure "
                    "(attempt %d of %d): %s",
                    self.name,
                    data_entry.media_path,
                    attempts,
                    self.max_retries,
                    error,
                )
