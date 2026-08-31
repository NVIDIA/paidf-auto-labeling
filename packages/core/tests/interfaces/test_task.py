# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import override

from core.interfaces.task import TaskInterface
from core.models import DataEntry


class PassthroughTask(TaskInterface):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        return data_entry

    @override
    def run_batch(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        return data_entries


class MissingBatchTask(TaskInterface):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        return data_entry


def test_task_interface_initializes_task_identity() -> None:
    task = PassthroughTask(name="passthrough")

    assert task.name == "passthrough"


def test_task_interface_requires_batch_strategy() -> None:
    assert "run_batch" in MissingBatchTask.__abstractmethods__


def test_task_interface_accepts_custom_batch_strategy() -> None:
    data_entry = DataEntry(media_path="video.mp4", data_path="data")
    task = PassthroughTask()

    assert task.run_batch([data_entry]) == [data_entry]
