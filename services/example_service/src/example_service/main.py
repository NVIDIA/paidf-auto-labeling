# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from core.interfaces import ServiceInterface
from core.models import DataEntry
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from example_task.task import SimpleTask


class ExampleService(ServiceInterface):
    def __init__(self) -> None:
        super().__init__(name="example_service", description="Run the example service pipeline.")

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        pipeline = LinearPipeline(
            tasks=[SimpleTask(name="example_task")],
            name="example_pipeline",
            policy=EmptyOutputPolicy.FAIL,
        )
        annotated = pipeline.run(data_entries)
        self.logger.info(f"Annotated {len(annotated)} data entries.")


def main() -> None:
    ExampleService().run()


if __name__ == "__main__":
    main()
