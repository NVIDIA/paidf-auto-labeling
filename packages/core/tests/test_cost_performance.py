# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core import DataEntry
from core.cost_performance import (
    TaskRunReport,
    collect_model_usage,
    model_usage_entry,
    normalize_token_counts,
    record_model_call,
    write_cost_performance_report,
)


def test_normalize_token_counts_accepts_openai_and_gemini_shapes() -> None:
    assert normalize_token_counts(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    ) == {
        "prompt_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    assert normalize_token_counts({"promptTokenCount": 8, "candidatesTokenCount": 3}) == {
        "prompt_tokens": 8,
        "output_tokens": 3,
        "total_tokens": 11,
    }


def test_normalize_token_counts_only_swallows_expected_model_dump_errors() -> None:
    class _BadUsage:
        def model_dump(self) -> dict[str, int]:
            raise TypeError("bad conversion")

    assert normalize_token_counts(_BadUsage()) == {}


def test_normalize_token_counts_propagates_unexpected_model_dump_errors() -> None:
    class _BuggyUsage:
        def model_dump(self) -> dict[str, int]:
            raise RuntimeError("bug")

    try:
        normalize_token_counts(_BuggyUsage())
    except RuntimeError as exc:
        assert str(exc) == "bug"
    else:
        raise AssertionError("RuntimeError was not propagated")


def test_model_usage_is_attributed_to_active_entry() -> None:
    entry = DataEntry(id="entry-1", media_path="clip.mp4", data_path="scene")

    with collect_model_usage() as usage:
        with model_usage_entry(entry):
            record_model_call(
                kind="vlm",
                provider="openai-compatible",
                model="vlm",
                endpoint_url="http://localhost:8080/v1",
                elapsed_s=0.25,
                retry_count=1,
                success=True,
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            )
        record_model_call(
            kind="llm",
            provider="openai-compatible",
            model="llm",
            endpoint_url="http://localhost:8081/v1",
            elapsed_s=0.1,
            retry_count=0,
            success=True,
        )

    snapshot = usage.snapshot()

    assert snapshot.for_entry("entry-1").summary_json()["vlm_calls_observed"] == 1
    assert snapshot.for_entry("entry-1").summary_json()["retry_count_observed"] == 1
    assert snapshot.unassigned().summary_json()["llm_calls_observed"] == 1


def test_record_model_call_emits_otel_metrics_even_without_accumulator() -> None:
    with patch("core.cost_performance.telemetry.record_model_call_metrics") as record_metrics:
        record_model_call(
            kind="llm",
            provider="openai-compatible",
            model="llm",
            endpoint_url="http://localhost:8081/v1",
            elapsed_s=0.2,
            retry_count=1,
            success=True,
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    record_metrics.assert_called_once_with(
        kind="llm",
        provider="openai-compatible",
        model="llm",
        success=True,
        elapsed_s=0.2,
        retry_count=1,
        token_counts={"prompt_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        data_entry_id=None,
    )


def test_write_cost_performance_report_emits_otel_summary(tmp_path: Path) -> None:
    entry = DataEntry(id="entry-1", media_path="clip.mp4", data_path=str(tmp_path / "scene"))
    with collect_model_usage() as usage:
        with model_usage_entry(entry):
            record_model_call(
                kind="llm",
                provider="openai-compatible",
                model="llm",
                endpoint_url="http://localhost:8081/v1",
                elapsed_s=0.2,
                retry_count=0,
                success=True,
                usage={"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
            )

    with patch("core.cost_performance.telemetry.emit_cost_performance_summary") as emit:
        write_cost_performance_report(
            entry,
            service_name="svc",
            service_elapsed_s=1.25,
            model_usage=usage.snapshot().for_entry(entry.id),
            task_reports=[
                TaskRunReport(name="reasoning", action="ran", success=True, elapsed_s=1.0)
            ],
            scale={"tracks": 3},
            degraded=False,
        )

    emit.assert_called_once()
    kwargs = emit.call_args.kwargs
    assert kwargs["service_name"] == "svc"
    assert kwargs["data_entry_id"] == "entry-1"
    assert kwargs["model_usage"]["llm_calls_observed"] == 1
    assert kwargs["model_usage"]["token_counts_observed"]["total_tokens"] == 15
    assert kwargs["tasks"][0]["name"] == "reasoning"
    assert kwargs["scene_scale"] == {"tracks": 3}
    assert not (tmp_path / "scene" / "sidecars" / "cost_performance" / "report.json").exists()
