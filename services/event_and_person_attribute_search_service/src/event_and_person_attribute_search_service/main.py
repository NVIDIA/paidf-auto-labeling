# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PAS-only service consuming Visual QA sidecars or explicit attribute JSON."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import override

import yaml
from core import (
    DataEntry,
    ModelUsageSnapshot,
    TaskRunReport,
    collect_model_usage,
    read_pipeline_state,
    write_cost_performance_report,
)
from core.interfaces import ServiceInterface
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from core.tasks import SequentialTask
from core.utils.multistorage import MSCStorage
from person_attribute_search.artifacts import PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY
from person_attribute_search.config import PersonAttributeSearchConfig
from person_attribute_search.task import PersonAttributeSearchTask

SERVICE_CONFIG_SECTION_KEYS = (
    "event_and_person_attribute_search",
    "event_and_attribute_search",
)


class EventAndPersonAttributeSearchService(ServiceInterface):
    """Run only the Person Attribute Search assembly/query task."""

    def __init__(self) -> None:
        super().__init__(
            name="event_and_person_attribute_search_service",
            description=(
                "Generate Person Attribute Search artifacts from upstream Visual QA "
                "sidecars or an explicit attribute JSON file."
            ),
        )
        self.storage = MSCStorage(self.logger)

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        """Register PAS input and query-generation options."""
        parser.add_argument(
            "--config-file",
            default=None,
            help="Local or remote JSON/YAML file containing PAS configuration.",
        )
        parser.add_argument(
            "--attribute-json",
            default=None,
            help=(
                "Local or remote attribute JSON. Overrides scene Visual QA sidecars and "
                "requires exactly one service input entry."
            ),
        )
        parser.add_argument(
            "--query-prompt-file",
            default=None,
            help="Prompt file for single-call easy/medium/hard bundle generation.",
        )
        parser.add_argument(
            "--query-prompt-text",
            default=None,
            help="Inline prompt for single-call easy/medium/hard bundle generation.",
        )
        parser.add_argument(
            "--query-count",
            type=int,
            default=None,
            help="Require exactly this many distinct queries in every tier.",
        )
        parser.add_argument(
            "--use-template-for-medium",
            action="store_true",
            help="Use templates for medium queries in the per-tier LLM mode.",
        )
        parser.add_argument(
            "--llm-provider",
            choices=["openai-compatible", "gemini"],
            default=None,
            help="PAS query-generation provider.",
        )
        parser.add_argument("--llm-endpoint-url", default=None)
        parser.add_argument("--llm-model", default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--top-p", type=float, default=None)
        parser.add_argument("--max-tokens", type=int, default=None)
        parser.add_argument(
            "--llm-retries",
            type=int,
            default=None,
            help="Retries for transient LLM endpoint failures.",
        )
        parser.add_argument(
            "--llm-retry-backoff-s",
            type=float,
            default=None,
            help="Base backoff in seconds for transient endpoint retries.",
        )
        parser.add_argument(
            "--llm-response-retries",
            type=int,
            default=None,
            help="Retries for empty, malformed, or contract-invalid LLM responses.",
        )
        parser.add_argument(
            "--llm-response-retry-backoff-s",
            type=float,
            default=None,
            help="Base exponential backoff in seconds for invalid-response retries.",
        )

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        """Load configuration, stage explicit inputs, and run PAS."""
        with tempfile.TemporaryDirectory(prefix="pas_inputs_") as temp_dir:
            staging_root = Path(temp_dir)
            if args.config_file:
                local_config = self.storage.download_if_remote(
                    args.config_file,
                    str(staging_root / "config"),
                    is_file=True,
                )
                config = _load_config(Path(local_config).expanduser().resolve())
            else:
                config = PersonAttributeSearchConfig()
            config = _apply_cli_overrides(config, args)
            config = _stage_remote_inputs(
                config,
                storage=self.storage,
                staging_root=staging_root / "inputs",
            )
            if config.attribute_json and len(data_entries) != 1:
                raise ValueError(
                    "--attribute-json requires exactly one service input entry; "
                    f"received {len(data_entries)}"
                )
            self._execute_pipeline(config, data_entries)

    def _execute_pipeline(
        self,
        config: PersonAttributeSearchConfig,
        data_entries: list[DataEntry],
    ) -> None:
        """Run the one-task PAS pipeline and write cost/performance reports."""
        task = _MeasuredTask(PersonAttributeSearchTask(config=config))
        pipeline = LinearPipeline(
            tasks=[task],
            name="person_attribute_search_pipeline",
            policy=EmptyOutputPolicy.FAIL,
        )
        started_at = time.perf_counter()
        with collect_model_usage() as model_usage:
            processed = pipeline.run(data_entries)
        elapsed_s = time.perf_counter() - started_at
        usage = model_usage.snapshot()
        for entry in processed:
            report = task.entry_reports.get(entry.id)
            task_reports = [report] if report is not None else []
            entry_usage = usage.for_entry(entry.id)
            write_cost_performance_report(
                entry,
                service_name=self.name,
                service_elapsed_s=report.elapsed_s if report is not None else elapsed_s,
                model_usage=entry_usage,
                task_reports=task_reports,
                scale=_scene_scale(entry, config),
                degraded=_is_degraded(entry, task_reports, entry_usage),
                degradation_reasons=_degradation_reasons(entry, task_reports, entry_usage),
                notes=[
                    "PAS consumes precomputed attributes; it does not run captioning or Visual QA.",
                    "Model counts include only PAS query-generation calls.",
                ],
            )
        self.logger.info("Processed %d data entries.", len(processed))


class _MeasuredTask(SequentialTask):
    """Record PAS elapsed time without changing task behavior."""

    def __init__(self, delegate: SequentialTask) -> None:
        super().__init__(name=delegate.name, max_retries=delegate.max_retries)
        self.delegate = delegate
        self.entry_reports: dict[str, TaskRunReport] = {}

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        started_at = time.perf_counter()
        try:
            output = self.delegate.run(data_entry)
        except Exception:
            self.entry_reports[data_entry.id] = TaskRunReport(
                name=self.name,
                action="ran",
                success=False,
                elapsed_s=time.perf_counter() - started_at,
            )
            raise
        self.entry_reports[data_entry.id] = TaskRunReport(
            name=self.name,
            action="ran",
            success=True,
            elapsed_s=time.perf_counter() - started_at,
        )
        return output


def _load_config(path: Path) -> PersonAttributeSearchConfig:
    """Read direct, nested PAS, or legacy composite configuration."""
    raw = path.read_text(encoding="utf-8")
    loaded = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise ValueError(f"Person Attribute Search config must be an object: {path}")

    section: object = loaded
    for key in SERVICE_CONFIG_SECTION_KEYS:
        if key in loaded:
            section = loaded[key]
            break
    if isinstance(section, dict) and "person_attribute_search" in section:
        section = section["person_attribute_search"]
    elif section is loaded and "person_attribute_search" in loaded:
        section = loaded["person_attribute_search"]
    if not isinstance(section, dict):
        raise ValueError(f"Person Attribute Search config section must be an object: {path}")
    return PersonAttributeSearchConfig.model_validate(section)


def _apply_cli_overrides(
    config: PersonAttributeSearchConfig,
    args: argparse.Namespace,
) -> PersonAttributeSearchConfig:
    """Apply explicit CLI options to PAS configuration."""
    updates: dict[str, object] = {}
    scalar_options = {
        "attribute_json": "attribute_json",
        "query_prompt_file": "query_prompt_file",
        "query_prompt_text": "query_prompt_text",
        "query_count": "bundle_query_count",
        "llm_provider": "llm_provider",
        "llm_endpoint_url": "llm_endpoint_url",
        "llm_model": "llm_model",
        "temperature": "llm_temperature",
        "top_p": "llm_top_p",
        "max_tokens": "bundle_max_tokens",
        "llm_retries": "llm_retries",
        "llm_retry_backoff_s": "llm_retry_backoff_s",
        "llm_response_retries": "llm_response_retries",
        "llm_response_retry_backoff_s": "llm_response_retry_backoff_s",
    }
    for argument, field in scalar_options.items():
        value = getattr(args, argument, None)
        if value is not None:
            updates[field] = value
    if getattr(args, "use_template_for_medium", False):
        updates["use_template_for_medium"] = True
    if updates.get("query_prompt_file") or updates.get("query_prompt_text"):
        updates["bundle_query_generation"] = True
    if not updates:
        return config
    return PersonAttributeSearchConfig.model_validate({**config.model_dump(), **updates})


def _stage_remote_inputs(
    config: PersonAttributeSearchConfig,
    *,
    storage: MSCStorage,
    staging_root: Path,
) -> PersonAttributeSearchConfig:
    """Stage remote attribute and prompt files for the task lifetime."""
    updates: dict[str, object] = {}
    for field in ("attribute_json", "query_prompt_file"):
        value = getattr(config, field)
        if not isinstance(value, str) or not storage.is_remote_storage_url(value):
            continue
        updates[field] = storage.download_if_remote(
            value,
            str(staging_root / field),
            is_file=True,
        )
    return config.model_copy(update=updates) if updates else config


def _scene_scale(
    data_entry: DataEntry,
    config: PersonAttributeSearchConfig,
) -> dict[str, object]:
    state = _pas_state(data_entry)
    return {
        "attribute_source": "attribute_json" if config.attribute_json else "visual_qa",
        "people": _int_value(state.get("n_people")) or (1 if state.get("success") else 0),
    }


def _is_degraded(
    data_entry: DataEntry,
    task_reports: list[TaskRunReport],
    model_usage: ModelUsageSnapshot,
) -> bool:
    return bool(_degradation_reasons(data_entry, task_reports, model_usage))


def _degradation_reasons(
    data_entry: DataEntry,
    task_reports: list[TaskRunReport],
    model_usage: ModelUsageSnapshot,
) -> list[str]:
    reasons = [f"task_failed:{report.name}" for report in task_reports if not report.success]
    failed_calls = sum(1 for call in model_usage.calls if not call.success)
    if failed_calls:
        reasons.append(f"model_call_failed:{failed_calls}")
    state = _pas_state(data_entry)
    if not state.get("success"):
        reasons.append("missing_or_invalid_attribute_source")
    optional_failures = state.get("optional_failures")
    if isinstance(optional_failures, list):
        reasons.extend(str(item) for item in optional_failures)
    return reasons


def _pas_state(data_entry: DataEntry) -> dict[str, object]:
    state = read_pipeline_state(data_entry.data_path)
    raw = state.task_artifacts.get(PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY)
    return raw if isinstance(raw, dict) else {}


def _int_value(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def main() -> None:
    """Console-script entrypoint."""
    EventAndPersonAttributeSearchService().run()


if __name__ == "__main__":
    main()
