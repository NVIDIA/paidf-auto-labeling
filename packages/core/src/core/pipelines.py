# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline implementations.

``LinearPipeline`` is the single in-process pipeline this framework
ships. It honors ``EmptyOutputPolicy`` (warn vs fail) and records per-task
``StageOutcome`` for observability. See the class docstring for details.

New pipeline shapes (DAG, conditional, parallel) should subclass
``PipelineInterface`` rather than extending this class.
"""

from __future__ import annotations

from typing import override

from core.interfaces import PipelineInterface, TaskInterface
from core.models import DataEntry
from core.policy import (
    EmptyOutputPolicy,
    PolicyValue,
    StageOutcome,
    StagePolicyError,
    coerce_policy,
)
from core.utils import telemetry


class LinearPipeline(PipelineInterface):
    """
    Linear pipeline with per-task error policy and stage outcomes.

    On task failure (raised exception or the task's own signaled failure), the
    configured ``EmptyOutputPolicy`` decides behavior:

    - ``WARN`` (default): the failure is logged, the task's outputs are
      skipped, and downstream tasks still run on the upstream data entries
      (i.e. the failed task is treated as a no-op). For example, if SR
      fails, tracking can still run on the original input.
    - ``FAIL``: the pipeline aborts with ``StagePolicyError``.

    Per-task ``StageOutcome`` records are exposed via ``last_outcomes`` so
    services can log per-task success/failure without wrapping every task
    themselves.
    """

    def __init__(
        self,
        tasks: list[TaskInterface],
        *,
        name: str | None = None,
        policy: PolicyValue | None = None,
    ) -> None:
        """
        Initialize a best-effort linear pipeline.

        Args:
            tasks: Ordered list of tasks to run.
            name: Optional pipeline name. Defaults to the class name.
            policy: How to react to per-task failures. Accepts an
                ``EmptyOutputPolicy`` member, ``"warn"`` / ``"fail"`` strings,
                or None (treated as ``WARN``).
        """
        super().__init__(name=name)
        self.tasks = tasks
        self.policy: EmptyOutputPolicy = coerce_policy(policy)
        self._last_outcomes: list[StageOutcome] = []

    @property
    def last_outcomes(self) -> list[StageOutcome]:
        """Per-task outcomes from the most recent ``run`` invocation."""
        return list(self._last_outcomes)

    @override
    def run(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        with telemetry.start_span("pipeline.run", {"pipeline.name": self.name}):
            return self._run(data_entries)

    def _run(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        data_entries = self.prepare_input(data_entries)
        self._last_outcomes = []

        current = data_entries
        for task in self.tasks:
            self.logger.info(
                f"Running task {task.name} on {len(current)} data entries "
                f"(policy={self.policy.value})."
            )
            try:
                with telemetry.start_span("task.run_batch", {"task.name": task.name}):
                    batch_result = task.run_batch(current)
                next_entries = self.prepare_stage_output(batch_result)
            except Exception as exc:
                outcome = StageOutcome(task_name=task.name, success=False, error=exc)
                self._last_outcomes.append(outcome)
                if self.policy is EmptyOutputPolicy.FAIL:
                    self.logger.error(
                        f"Task {task.name} raised {exc.__class__.__name__}; "
                        "aborting pipeline (policy=fail)."
                    )
                    raise StagePolicyError(f"Task {task.name} failed: {exc}") from exc
                self.logger.warning(
                    f"Task {task.name} raised {exc.__class__.__name__}: {exc} "
                    "(continuing pipeline, policy=warn)."
                )
                continue

            self._last_outcomes.append(StageOutcome(task_name=task.name, success=True))
            self.logger.info(f"Completed task {task.name} on {len(next_entries)} data entries.")
            current = next_entries

        return self.prepare_output(current)


__all__ = ["LinearPipeline"]
