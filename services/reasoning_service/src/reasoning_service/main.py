# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from core import DataEntry
from core.interfaces import ServiceInterface
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from daft_validation import DaftValidationTask
from reasoning.config import DaftExportConfig
from reasoning.endpoint_resolver import EndpointResolver
from reasoning.task import ReasoningTask


class ReasoningService(ServiceInterface):
    def __init__(self) -> None:
        super().__init__(
            name="reasoning_service",
            description="Run opt-in LLM reasoning stages over existing DAFT scene directories.",
        )

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config-file",
            type=str,
            default=None,
            help="Optional JSON/YAML file containing a reasoning config block.",
        )
        parser.add_argument(
            "--llm-provider",
            choices=("openai-compatible",),
            default="openai-compatible",
            help="Provider adapter for reasoning LLM requests.",
        )
        parser.add_argument(
            "--llm-endpoint-url",
            type=str,
            default="",
            help=(
                "Optional LLM endpoint URL override for opt-in LLM stages. "
                "OpenAI-compatible endpoints accept a /v1 base or a /chat/completions URL."
            ),
        )
        parser.add_argument(
            "--llm-model",
            type=str,
            default="",
            help="Optional LLM model override for opt-in LLM stages.",
        )
        parser.add_argument(
            "--reasoning-mode",
            choices=("config", "keep", "strip"),
            default="config",
            help=(
                "Reasoning trace handling. 'config' strips reasoning only when a config file "
                "is provided and reasoning is disabled; 'keep' preserves reasoning; 'strip' "
                "always removes reasoning from outputs written by this run."
            ),
        )

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        cfg_path = Path(args.config_file).expanduser().resolve() if args.config_file else None
        cfg = _load_config(cfg_path) if cfg_path is not None else None
        resolver = build_resolver(args, logger=self.logger)
        config_dir = cfg_path.parent if cfg_path is not None else Path.cwd()

        pipeline = LinearPipeline(
            tasks=[
                DaftValidationTask(),
                ReasoningTask(
                    config=cfg,
                    resolver=resolver,
                    config_dir=config_dir,
                    reasoning_mode=args.reasoning_mode,
                ),
                DaftValidationTask(),
            ],
            name="reasoning_pipeline",
            policy=EmptyOutputPolicy.FAIL,
        )
        processed = pipeline.run(data_entries)
        self.logger.info("Processed %d data entries.", len(processed))


def _load_config(path: Path) -> DaftExportConfig:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(raw)
    else:
        loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise ValueError(f"Reasoning config must be a JSON/YAML object: {path}")
    section = loaded.get("reasoning", loaded)
    if not isinstance(section, dict):
        raise ValueError(f"Reasoning config section must be an object: {path}")
    return DaftExportConfig.model_validate(section)


def build_resolver(args: argparse.Namespace, *, logger: logging.Logger) -> EndpointResolver:
    """Build the reasoning endpoint resolver from parsed CLI args."""
    resolver = EndpointResolver(None, logger=logger)
    resolver.apply_llm_overrides(
        url=getattr(args, "llm_endpoint_url", None) or "",
        model=getattr(args, "llm_model", None) or "",
    )
    return resolver


def main() -> None:
    ReasoningService().run()


if __name__ == "__main__":
    main()
