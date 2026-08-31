# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Task contracts shared by every annotation implementation.

``TaskInterface`` defines the shape that pipelines and services depend on: a task must be able to
process one ``DataEntry`` and must also expose a batch entrypoint. The interface intentionally does
not define how batching happens. Concrete task base classes, such as
``core.tasks.SequentialTask``, own those mechanics so future task types can choose a different
batching strategy without changing the pipeline contract.
"""

import logging
from abc import ABC, abstractmethod

from core.models import DataEntry
from core.utils.logging import get_logger


class TaskInterface(ABC):
    """
    Base interface for all task packages.

    Pipelines should type against this interface so they can compose tasks without depending on a
    specific batching implementation. Task authors should usually subclass a more specific base
    class, such as ``core.tasks.SequentialTask``, unless they need to provide their own
    ``run_batch`` behavior.
    """

    logger: logging.Logger
    name: str

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name, kind="task")
        self.logger.info(f"Initialized Task {self.name}.")

    @abstractmethod
    def run(self, data_entry: DataEntry) -> DataEntry:
        """
        Annotate a single data entry by writing outputs into the scene directory.

        Tasks may do the following:
            - Write DAFT artifacts under ``contextual/``, ``task/``, or ``sidecars/`` in the scene
              tree at ``data_entry.data_path``.
            - Record shared artifact references into the typed ``core.ScenePipelineState`` at
              ``<scene>/sidecars/pipeline_state.json``.
            - Read the current scene media from ``data_entry.media_path``. During pipeline
              execution this points at the local ``sidecars/active.*`` file under
              ``data_entry.data_path``.
            - If the task updates the scene media, such as super-resolution or denoising, either
              overwrite the active media file in place or return a ``DataEntry`` with
              ``media_path`` pointing at the newly produced local/remote media. The pipeline will
              promote that file back into ``sidecars/active.*`` before the next task runs.
        Args:
            data_entry: The data entry to annotate.
        Returns:
            The updated data entry. If the task does not update ``media_path`` or other base
            fields, return the unchanged input data entry.
        Raises:
            RetryTaskError: If the task encountered a transient failure for the current data entry.
            InvalidInputError: If the current data entry is invalid for the task and should be
                excluded.
            Any other exception: If the task encountered a fatal error for the current data entry.
        """
        ...

    @abstractmethod
    def run_batch(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        """
        Annotate a batch of data entries using this task's batching strategy.

        This method is part of the public task contract because pipelines operate on batches.
        Implementations may process entries sequentially, in model-native batches, in parallel, or
        with any other strategy that fits the task. Implementations are also responsible for
        documenting their own retry and invalid-input behavior.

        Args:
            data_entries: The data entries to be annotated.
        Returns:
            A list of annotated data entries.
        """
        ...
