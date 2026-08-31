# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline failure policy and stage outcomes.

The pseudo-labeling pipeline is fallback-friendly by default: when a
non-essential task fails, the pipeline records the failure, skips the task's
outputs, and continues. Operators can opt into hard-fail behavior by setting
``EmptyOutputPolicy.FAIL``.

See ``LinearPipeline`` for the runtime semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class EmptyOutputPolicy(StrEnum):
    """
    How the pipeline reacts to per-task failures or empty outputs.
    """

    WARN = "warn"
    """Log the failure and continue running downstream tasks."""

    FAIL = "fail"
    """Abort the pipeline by raising ``StagePolicyError``."""


PolicyValue = EmptyOutputPolicy | Literal["warn", "fail"]


def coerce_policy(
    value: PolicyValue | str | None, default: EmptyOutputPolicy = EmptyOutputPolicy.WARN
) -> EmptyOutputPolicy:
    """
    Normalize policy strings (case-insensitive) into ``EmptyOutputPolicy``.

    Unknown or empty values fall back to ``default`` (``WARN``).

    Args:
        value: Raw policy value — enum member, string, or None.
        default: Fallback returned when ``value`` is None or unrecognized.
    Returns:
        The normalized ``EmptyOutputPolicy`` member.
    """
    if value is None:
        return default
    if isinstance(value, EmptyOutputPolicy):
        return value
    s = str(value).strip().lower()
    if s == EmptyOutputPolicy.FAIL.value:
        return EmptyOutputPolicy.FAIL
    if s == EmptyOutputPolicy.WARN.value:
        return EmptyOutputPolicy.WARN
    return default


class StagePolicyError(RuntimeError):
    """
    Raised when a pipeline enforces ``FAIL`` policy after a task failure.
    """


@dataclass(frozen=True)
class StageOutcome:
    """
    Result of running one task during a pipeline run.

    ``LinearPipeline`` records one outcome per task in ``last_outcomes``.
    Services can use those outcomes for logs, run summaries, or persisted
    pseudo-labeling run metadata:

    ```
    pipeline = LinearPipeline(tasks=tasks, policy="warn")
    annotated = pipeline.run(data_entries)

    stage_metadata = [
        {
            "task_name": outcome.task_name,
            "success": outcome.success,
            "error_type": None
            if outcome.error is None
            else outcome.error.__class__.__name__,
            "error_message": None if outcome.error is None else str(outcome.error),
        }
        for outcome in pipeline.last_outcomes
    ]
    ```

    ``StageOutcome`` keeps the in-memory exception object so callers can
    decide how much detail to log or persist. Persisted metadata should
    serialize the exception into small fields such as type and message.

    Attributes:
        task_name: Name of the task that produced this outcome.
        success: True when the task completed without raising and the
            implementation considers its output valid.
        error: Exception captured from the task, if any.
    """

    task_name: str
    success: bool
    error: BaseException | None = None


__all__ = [
    "EmptyOutputPolicy",
    "PolicyValue",
    "StageOutcome",
    "StagePolicyError",
    "coerce_policy",
]
