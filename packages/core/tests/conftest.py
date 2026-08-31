# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import override

import pytest
from core.models import DataEntry
from core.tasks import SequentialTask


class SimpleTask(SequentialTask):
    """
    This class is the implementation of a task which returns the data entry.
    """

    def __init__(self, name: str | None = None, max_retries: int = 0) -> None:
        super().__init__(name=name, max_retries=max_retries)

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        return data_entry


@pytest.fixture
def simple_task() -> SimpleTask:
    return SimpleTask()
