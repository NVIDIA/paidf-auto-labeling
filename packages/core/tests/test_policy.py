# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import override

import pytest
from core import (
    DataEntry,
    EmptyOutputPolicy,
    LinearPipeline,
    StagePolicyError,
)
from core.policy import coerce_policy
from core.tasks import SequentialTask


class _PassThroughTask(SequentialTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        return data_entry


class _FailingTask(SequentialTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        raise RuntimeError("boom")


def _entry(tmp_path: Path) -> DataEntry:
    # PipelineInterface.prepare_input drops entries whose media is
    # missing; touch the file so this entry survives prep and the
    # policy semantics under test are exercised on a real task call.
    media = tmp_path / "x.mp4"
    media.touch()
    return DataEntry(media_path=str(media), data_path=str(tmp_path / "d"))


def test_coerce_policy_accepts_strings_and_enum() -> None:
    assert coerce_policy("warn") is EmptyOutputPolicy.WARN
    assert coerce_policy("FAIL") is EmptyOutputPolicy.FAIL
    assert coerce_policy(EmptyOutputPolicy.FAIL) is EmptyOutputPolicy.FAIL
    assert coerce_policy("nonsense") is EmptyOutputPolicy.WARN
    assert coerce_policy(None) is EmptyOutputPolicy.WARN


def test_warn_policy_continues_after_failure(tmp_path: Path) -> None:
    pipeline = LinearPipeline(
        tasks=[_FailingTask(name="bad"), _PassThroughTask(name="ok")],
        policy="warn",
    )
    out = pipeline.run([_entry(tmp_path)])
    assert len(out) == 1
    outcomes = pipeline.last_outcomes
    assert [o.task_name for o in outcomes] == ["bad", "ok"]
    assert outcomes[0].success is False
    assert isinstance(outcomes[0].error, RuntimeError)
    assert outcomes[1].success is True


def test_fail_policy_aborts(tmp_path: Path) -> None:
    pipeline = LinearPipeline(
        tasks=[_FailingTask(name="bad")],
        policy=EmptyOutputPolicy.FAIL,
    )
    with pytest.raises(StagePolicyError):
        pipeline.run([_entry(tmp_path)])
