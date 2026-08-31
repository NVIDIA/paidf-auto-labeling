# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import override

import pytest
from core.exceptions import InvalidInputError, RetryTaskError
from core.models import DataEntry
from core.tasks import SequentialTask


class PassthroughTask(SequentialTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        return data_entry


class InvalidInputTask(SequentialTask):
    def __init__(self, invalid_media_path: str) -> None:
        super().__init__()
        self.invalid_media_path = invalid_media_path

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        if data_entry.media_path == self.invalid_media_path:
            raise InvalidInputError("invalid input")
        return data_entry


class FlakyTask(SequentialTask):
    attempts: int

    def __init__(self, failures_before_success: int, max_retries: int) -> None:
        super().__init__(max_retries=max_retries)
        self.failures_before_success = failures_before_success
        self.attempts = 0

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise RetryTaskError("transient failure")
        return data_entry


class FatalTask(SequentialTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        raise RuntimeError("fatal failure")


def test_sequential_task_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        PassthroughTask(max_retries=-1)


def test_run_batch_skips_invalid_input_errors() -> None:
    valid_entry = DataEntry(media_path="valid.mp4", data_path="valid-data")
    invalid_entry = DataEntry(media_path="invalid.mp4", data_path="invalid-data")

    task = InvalidInputTask(invalid_media_path=invalid_entry.media_path)

    assert task.run_batch([valid_entry, invalid_entry]) == [valid_entry]


def test_run_batch_retries_retry_task_errors_until_success() -> None:
    data_entry = DataEntry(media_path="video.mp4", data_path="data")
    task = FlakyTask(failures_before_success=2, max_retries=2)

    assert task.run_batch([data_entry]) == [data_entry]
    assert task.attempts == 3


def test_run_batch_reraises_retry_task_errors_after_retries_are_exhausted() -> None:
    data_entry = DataEntry(media_path="video.mp4", data_path="data")
    task = FlakyTask(failures_before_success=2, max_retries=1)

    with pytest.raises(RetryTaskError, match="transient failure"):
        task.run_batch([data_entry])

    assert task.attempts == 2


def test_run_batch_propagates_fatal_errors() -> None:
    data_entry = DataEntry(media_path="video.mp4", data_path="data")
    task = FatalTask()

    with pytest.raises(RuntimeError, match="fatal failure"):
        task.run_batch([data_entry])
