# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost and performance reporting helpers for model-backed services."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

from core.models import DataEntry
from core.utils import telemetry

ModelCallKind = Literal["vlm", "llm"]

# Kept for backward-compatible imports. Cost/performance is emitted via OTEL
# rather than a fixed scene sidecar to avoid parallel remote-storage overwrites.
COST_PERFORMANCE_REPORT_SIDECAR = "cost_performance/report.json"
SCHEMA_VERSION = "1"

_ACTIVE_USAGE: ContextVar[ModelUsageAccumulator | None] = ContextVar(
    "active_model_usage",
    default=None,
)
_ACTIVE_ENTRY_ID: ContextVar[str | None] = ContextVar("active_model_usage_entry_id", default=None)


@dataclass(frozen=True)
class TaskRunReport:
    """One task execution outcome for a single scene."""

    name: str
    action: str
    success: bool
    elapsed_s: float
    reason: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        payload: dict[str, object] = {
            "name": self.name,
            "action": self.action,
            "success": self.success,
            "elapsed_s": round(self.elapsed_s, 6),
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class ModelCallRecord:
    """One observed model endpoint call."""

    kind: ModelCallKind
    provider: str
    model: str
    endpoint_url: str | None
    data_entry_id: str | None
    success: bool
    elapsed_s: float
    retry_count: int
    token_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelUsageSnapshot:
    """Immutable model-call snapshot collected during a pipeline run."""

    calls: tuple[ModelCallRecord, ...]

    def for_entry(self, data_entry_id: str) -> ModelUsageSnapshot:
        """Return calls attributed to ``data_entry_id``."""
        return ModelUsageSnapshot(
            calls=tuple(call for call in self.calls if call.data_entry_id == data_entry_id)
        )

    def unassigned(self) -> ModelUsageSnapshot:
        """Return calls that happened outside a per-entry task context."""
        return ModelUsageSnapshot(
            calls=tuple(call for call in self.calls if call.data_entry_id is None)
        )

    def summary_json(self) -> dict[str, object]:
        """Return uniform aggregate counters for this snapshot."""
        total_calls = len(self.calls)
        vlm_calls = sum(1 for call in self.calls if call.kind == "vlm")
        llm_calls = sum(1 for call in self.calls if call.kind == "llm")
        retry_count = sum(call.retry_count for call in self.calls)
        retry_rate = (retry_count / total_calls) if total_calls else 0.0
        failed_calls = sum(1 for call in self.calls if not call.success)
        by_provider_model: Counter[str] = Counter(
            f"{call.provider}:{call.model}:{call.kind}" for call in self.calls
        )
        return {
            "vlm_calls_observed": vlm_calls,
            "llm_calls_observed": llm_calls,
            "total_calls_observed": total_calls,
            "failed_calls_observed": failed_calls,
            "retry_count_observed": retry_count,
            "retry_rate_observed": retry_rate,
            "model_endpoint_elapsed_s": round(sum(call.elapsed_s for call in self.calls), 6),
            "token_counts_observed": _sum_token_counts(call.token_counts for call in self.calls),
            "by_provider_model": dict(sorted(by_provider_model.items())),
        }


class ModelUsageAccumulator:
    """Mutable per-run accumulator for model endpoint usage."""

    def __init__(self) -> None:
        self._calls: list[ModelCallRecord] = []

    def record(self, call: ModelCallRecord) -> None:
        """Record one model endpoint call."""
        self._calls.append(call)

    def snapshot(self) -> ModelUsageSnapshot:
        """Return an immutable snapshot."""
        return ModelUsageSnapshot(calls=tuple(self._calls))


@contextmanager
def collect_model_usage() -> Iterator[ModelUsageAccumulator]:
    """Collect model usage emitted by core model clients in the current context."""
    accumulator = ModelUsageAccumulator()
    token = _ACTIVE_USAGE.set(accumulator)
    try:
        yield accumulator
    finally:
        _ACTIVE_USAGE.reset(token)


@contextmanager
def model_usage_entry(data_entry: DataEntry) -> Iterator[None]:
    """Attribute model calls in this context to ``data_entry``."""
    token = _ACTIVE_ENTRY_ID.set(data_entry.id)
    try:
        yield
    finally:
        _ACTIVE_ENTRY_ID.reset(token)


def record_model_call(
    *,
    kind: ModelCallKind,
    provider: str,
    model: str,
    endpoint_url: str | None,
    elapsed_s: float,
    retry_count: int,
    success: bool,
    usage: object | None = None,
) -> None:
    """Record a model call into the active accumulator and OTEL metrics."""
    token_counts = normalize_token_counts(usage)
    data_entry_id = _ACTIVE_ENTRY_ID.get()
    safe_elapsed_s = max(0.0, float(elapsed_s))
    safe_retry_count = max(0, int(retry_count))
    telemetry.record_model_call_metrics(
        kind=kind,
        provider=provider,
        model=model,
        success=success,
        elapsed_s=safe_elapsed_s,
        retry_count=safe_retry_count,
        token_counts=token_counts,
        data_entry_id=data_entry_id,
    )
    accumulator = _ACTIVE_USAGE.get()
    if accumulator is None:
        return
    accumulator.record(
        ModelCallRecord(
            kind=kind,
            provider=provider,
            model=model,
            endpoint_url=endpoint_url,
            data_entry_id=data_entry_id,
            success=success,
            elapsed_s=safe_elapsed_s,
            retry_count=safe_retry_count,
            token_counts=token_counts,
        )
    )


def normalize_token_counts(usage: object | None) -> dict[str, int]:
    """Normalize provider-specific usage metadata into shared token keys."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            usage = usage.model_dump()
        except (TypeError, ValueError, AttributeError):
            return {}
    if not isinstance(usage, Mapping):
        return {}
    prompt = _first_int(usage, ("prompt_tokens", "promptTokenCount", "input_tokens"))
    output = _first_int(
        usage,
        ("completion_tokens", "candidatesTokenCount", "output_tokens", "response_tokens"),
    )
    total = _first_int(usage, ("total_tokens", "totalTokenCount"))
    if total is None and (prompt is not None or output is not None):
        total = int(prompt or 0) + int(output or 0)

    counts: dict[str, int] = {}
    if prompt is not None:
        counts["prompt_tokens"] = prompt
    if output is not None:
        counts["output_tokens"] = output
    if total is not None:
        counts["total_tokens"] = total
    return counts


def write_cost_performance_report(
    data_entry: DataEntry,
    *,
    service_name: str,
    service_elapsed_s: float,
    model_usage: ModelUsageSnapshot,
    task_reports: Sequence[TaskRunReport] = (),
    scale: Mapping[str, object] | None = None,
    degraded: bool = False,
    degradation_reasons: Sequence[str] = (),
    notes: Sequence[str] = (),
    output_sidecar: str = COST_PERFORMANCE_REPORT_SIDECAR,
) -> None:
    """Emit the universal per-scene cost/performance report via OpenTelemetry.

    A fixed scene sidecar is intentionally not written: parallel service
    containers sharing remote storage would race on the same path. The
    ``output_sidecar`` argument is retained for call-site compatibility and
    ignored.
    """
    del output_sidecar  # retained for API compatibility; sidecar writes removed
    telemetry.emit_cost_performance_summary(
        service_name=service_name,
        data_entry_id=data_entry.id,
        media_path=data_entry.media_path,
        data_path=data_entry.data_path,
        service_elapsed_s=service_elapsed_s,
        model_usage=model_usage.summary_json(),
        tasks=[report.to_json() for report in task_reports],
        scene_scale=dict(scale or {}),
        degraded=bool(degraded),
        degradation_reasons=list(degradation_reasons),
        notes=list(notes),
        schema_version=SCHEMA_VERSION,
    )


def timed_model_call(
    *,
    kind: ModelCallKind,
    provider: str,
    model: str,
    endpoint_url: str | None,
) -> ModelCallTimer:
    """Create a timer object for callers that need manual success/retry reporting."""
    return ModelCallTimer(kind=kind, provider=provider, model=model, endpoint_url=endpoint_url)


@dataclass
class ModelCallTimer:
    """Small helper for recording a model call after retries complete."""

    kind: ModelCallKind
    provider: str
    model: str
    endpoint_url: str | None
    started_at: float = field(default_factory=time.perf_counter)

    def record(self, *, success: bool, retry_count: int, usage: object | None = None) -> None:
        """Record the completed call."""
        record_model_call(
            kind=self.kind,
            provider=self.provider,
            model=self.model,
            endpoint_url=self.endpoint_url,
            elapsed_s=time.perf_counter() - self.started_at,
            retry_count=retry_count,
            success=success,
            usage=usage,
        )


def _first_int(mapping: Mapping[object, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _sum_token_counts(counts: Iterable[Mapping[str, int]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for item in counts:
        for key, value in item.items():
            totals[key] += int(value)
    return dict(sorted(totals.items()))


__all__ = [
    "COST_PERFORMANCE_REPORT_SIDECAR",
    "ModelCallKind",
    "ModelCallRecord",
    "ModelCallTimer",
    "ModelUsageAccumulator",
    "ModelUsageSnapshot",
    "TaskRunReport",
    "collect_model_usage",
    "model_usage_entry",
    "normalize_token_counts",
    "record_model_call",
    "timed_model_call",
    "write_cost_performance_report",
]
