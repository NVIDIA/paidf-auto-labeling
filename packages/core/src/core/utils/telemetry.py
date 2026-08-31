# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry traces and metrics for annotation services.

``setup_telemetry()`` enables telemetry when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
set or a signal exporter is set to ``console``. It is a no-op otherwise, and also
a no-op if ``OTEL_SDK_DISABLED=true``. Any setup failure is logged and swallowed
rather than raised. Configuration uses the standard ``OTEL_*`` env vars.

The Resource carries ``service.name``, ``service.namespace``, and
``service.version``; ``run.id`` and ``host.name`` are added as span attributes by
:func:`start_span`. Explicit metric instruments use delta temporality. The
OpenTelemetry SDK is a ``core`` dependency, but all OpenTelemetry imports
are deferred into the functions that use them to keep module import (and thus
cold-start) cheap when telemetry is never set up.
"""

from __future__ import annotations

import atexit
import os
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.utils.logging import get_logger

logger = get_logger("telemetry")

_TRACER_NAME = "auto-labeling"
_METER_NAME = "auto-labeling"
_DEFAULT_SERVICE_NAME = "auto-labeling"
_DEFAULT_SERVICE_NAMESPACE = "auto-labeling"
_DEFAULT_SERVICE_VERSION = "0.0.0"

# Supported OTLP transports, mirroring opentelemetry.sdk._configuration.
_OTLP_GRPC = "grpc"
_OTLP_HTTP = "http/protobuf"

# Supported exporter selections for OTEL_TRACES_EXPORTER / OTEL_METRICS_EXPORTER.
_EXPORTER_CONSOLE = "console"
_EXPORTER_OTLP = "otlp"
_EXPORTER_NONE = "none"

# Providers are kept module-level so they can be flushed at process exit.
_INITIALIZED = False
_ENABLED = False
_tracer_provider: Any = None
_meter_provider: Any = None
_metrics: dict[str, Any] = {}

RUN_ID: str | None = None
_HOST_NAME = socket.gethostname()


def _is_disabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in ("true", "1", "yes")


def _is_configured() -> bool:
    """Return True when telemetry has a usable exporter target."""
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip():
        return True
    traces = os.getenv("OTEL_TRACES_EXPORTER", _EXPORTER_OTLP).strip().lower()
    metrics = os.getenv("OTEL_METRICS_EXPORTER", _EXPORTER_OTLP).strip().lower()
    return traces == _EXPORTER_CONSOLE and metrics == _EXPORTER_CONSOLE


def is_enabled() -> bool:
    """Return True when telemetry has been initialised with real exporters."""
    return _ENABLED


def _build_resource(service_name: str | None, service_version: str | None) -> Any:
    """Build the OTel Resource carrying ``service.name``, ``service.namespace``,
    and ``service.version``.

    Each field resolves from the explicit argument, then ``OTEL_SERVICE_NAME`` /
    ``OTEL_SERVICE_VERSION``, then a module default.
    """
    from opentelemetry.sdk.resources import Resource

    name = service_name or os.getenv("OTEL_SERVICE_NAME") or _DEFAULT_SERVICE_NAME
    namespace = os.getenv("OTEL_SERVICE_NAMESPACE") or _DEFAULT_SERVICE_NAMESPACE
    version = service_version or os.getenv("OTEL_SERVICE_VERSION") or _DEFAULT_SERVICE_VERSION
    return Resource.create(
        {
            "service.name": name,
            "service.namespace": namespace,
            "service.version": version,
        }
    )


def _resolve_otlp_protocol(signal_env: str) -> str:
    """Resolve the OTLP protocol: signal-specific var, else the global, else grpc.

    Raises ``RuntimeError`` for an unsupported value (matching the SDK).
    """
    protocol = (
        os.getenv(signal_env) or os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or _OTLP_GRPC
    ).strip()
    if protocol not in (_OTLP_GRPC, _OTLP_HTTP):
        raise RuntimeError(f"Unsupported OTLP protocol '{protocol}' is configured")
    return protocol


def _make_span_exporter(exporter_type: str) -> Any:
    if exporter_type == _EXPORTER_CONSOLE:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    if _resolve_otlp_protocol("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") == _OTLP_HTTP:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpSpanExporter,
        )

        return HttpSpanExporter()

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcSpanExporter,
    )

    return GrpcSpanExporter()


def _delta_temporality() -> dict[Any, Any]:
    """Map cumulative instrument kinds to delta temporality."""
    from opentelemetry.sdk.metrics import Counter, Histogram, ObservableCounter
    from opentelemetry.sdk.metrics.export import AggregationTemporality

    return {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
    }


def _make_metric_exporter(exporter_type: str) -> Any:
    if exporter_type == _EXPORTER_CONSOLE:
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        return ConsoleMetricExporter(preferred_temporality=_delta_temporality())

    if _resolve_otlp_protocol("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL") == _OTLP_HTTP:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as HttpMetricExporter,
        )

        return HttpMetricExporter(preferred_temporality=_delta_temporality())

    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as GrpcMetricExporter,
    )

    return GrpcMetricExporter(preferred_temporality=_delta_temporality())


def _instrument_libraries() -> None:
    """Enable auto-instrumentation for supported client libraries.

    Each instrumentor is optional: a missing package or already-instrumented state
    is logged and ignored. ``OpenAIInstrumentor`` produces a span per call made
    through the ``openai`` SDK.
    """
    try:
        from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
    except Exception as e:  # pragma: no cover - depends on optional package
        logger.debug(f"openai auto-instrumentation unavailable: {e}")

    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        # Inject trace_id/span_id into existing log records without rewriting the
        # log format the service already configured.
        LoggingInstrumentor().instrument(set_logging_format=False)
    except Exception as e:  # pragma: no cover - depends on optional package
        logger.debug(f"logging auto-instrumentation unavailable: {e}")


def _create_instruments(meter_provider: Any) -> None:
    """Register explicit metric instruments on ``meter_provider``."""
    meter = meter_provider.get_meter(_METER_NAME)
    _metrics["task.frames_processed"] = meter.create_counter(
        "task.frames_processed",
        unit="1",
        description="Frames of input media processed by a task (labeled by task_name).",
    )
    _metrics["model.calls"] = meter.create_counter(
        "model.calls",
        unit="1",
        description="Model endpoint calls observed by core model clients.",
    )
    _metrics["model.call.duration"] = meter.create_histogram(
        "model.call.duration",
        unit="s",
        description="Model endpoint call duration in seconds.",
    )
    _metrics["model.tokens"] = meter.create_counter(
        "model.tokens",
        unit="1",
        description="Token counts observed on model endpoint calls.",
    )
    _metrics["model.retries"] = meter.create_counter(
        "model.retries",
        unit="1",
        description="Retry attempts observed on model endpoint calls.",
    )


def setup_telemetry(
    service_name: str | None = None,
    service_version: str | None = None,
) -> None:
    """Initialise OpenTelemetry traces + metrics.

    Idempotent and a no-op when ``OTEL_SDK_DISABLED`` is set or when telemetry is
    not configured. Any telemetry failure is logged and swallowed so it can never
    break a service.

    Args:
        service_name: Stable service identity for the Resource ``service.name``.
        service_version: Stable service version for the Resource ``service.version``.
    """
    global _INITIALIZED, _ENABLED, _tracer_provider, _meter_provider, RUN_ID

    if _INITIALIZED:
        return

    # Generate eagerly so callers reading telemetry.RUN_ID get a stable id even on
    # the no-op paths.
    RUN_ID = str(uuid.uuid4())

    if _is_disabled():
        logger.info("OpenTelemetry disabled via OTEL_SDK_DISABLED; telemetry is a no-op")
        _INITIALIZED = True
        return

    if not _is_configured():
        logger.info(
            "OpenTelemetry not configured (set OTEL_EXPORTER_OTLP_ENDPOINT, or an "
            "OTEL_*_EXPORTER=console, to enable); telemetry is a no-op"
        )
        _INITIALIZED = True
        return

    traces_exporter = os.getenv("OTEL_TRACES_EXPORTER", _EXPORTER_OTLP).strip().lower()
    metrics_exporter = os.getenv("OTEL_METRICS_EXPORTER", _EXPORTER_OTLP).strip().lower()

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = _build_resource(service_name, service_version)

        tracer_provider = TracerProvider(resource=resource)
        if traces_exporter != _EXPORTER_NONE:
            tracer_provider.add_span_processor(
                BatchSpanProcessor(_make_span_exporter(traces_exporter))
            )

        metric_readers = []
        if metrics_exporter != _EXPORTER_NONE:
            metric_readers.append(
                PeriodicExportingMetricReader(_make_metric_exporter(metrics_exporter))
            )
        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)

        _create_instruments(meter_provider)
        _instrument_libraries()

        # Install the global providers last so a failure above leaves the globals
        # untouched and a later call can retry cleanly.
        trace.set_tracer_provider(tracer_provider)
        metrics.set_meter_provider(meter_provider)

        _tracer_provider = tracer_provider
        _meter_provider = meter_provider
        _ENABLED = True
        _INITIALIZED = True
        atexit.register(shutdown)
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "n/a"
        logger.info(
            f"OpenTelemetry initialised (run.id={RUN_ID}, traces={traces_exporter}, "
            f"metrics={metrics_exporter}, endpoint={otlp_endpoint})"
        )
    except Exception as e:
        # Never let observability setup break the service; leave _INITIALIZED False
        # so a later call can retry.
        logger.warning(f"OpenTelemetry setup failed; continuing without telemetry: {e}")
        _ENABLED = False
        _tracer_provider = None
        _meter_provider = None


def shutdown() -> None:
    """Flush and shut down providers, registered via ``atexit`` by
    :func:`setup_telemetry`. Safe to call multiple times.
    """
    global _tracer_provider, _meter_provider, _ENABLED
    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush()
            _tracer_provider.shutdown()
        except Exception as exc:  # pragma: no cover - best-effort flush on exit
            logger.debug(f"tracer provider shutdown failed: {exc}")
        _tracer_provider = None
    if _meter_provider is not None:
        try:
            _meter_provider.force_flush()
            _meter_provider.shutdown()
        except Exception as exc:  # pragma: no cover - best-effort flush on exit
            logger.debug(f"meter provider shutdown failed: {exc}")
        _meter_provider = None
    _metrics.clear()
    _ENABLED = False


@contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start a span around a phase of work.

    Yields ``None`` when telemetry is disabled. ``None``-valued attributes are
    dropped, and ``run.id`` / ``host.name`` are added to the span.
    """
    if not _ENABLED:
        yield None
        return

    from opentelemetry import trace

    tracer = trace.get_tracer(_TRACER_NAME)
    attrs = {k: v for k, v in (attributes or {}).items() if v is not None}
    if RUN_ID is not None:
        attrs.setdefault("run.id", RUN_ID)
    attrs.setdefault("host.name", _HOST_NAME)
    with tracer.start_as_current_span(name, attributes=attrs) as span:
        yield span


def set_attributes(span: Any, attributes: dict[str, Any] | None = None) -> None:
    """Set non-None attributes on a span (matches :func:`start_span`'s policy)."""
    if span is None or not attributes:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def record_input_frames(task_name: str, media_path: str | Path) -> None:
    """Best-effort: probe ``media_path`` and add its frame count for ``task_name``.

    Increments the ``task.frames_processed`` counter by the number of frames in a
    task's input media (1 for images). No-op when telemetry is disabled or the
    counter is unregistered, and silently skips when the frame count cannot be
    determined, so frame-count telemetry never affects task execution.
    """
    if not _ENABLED:
        return
    counter = _metrics.get("task.frames_processed")
    if counter is None:
        return

    from core.utils.media_probe import probe_frame_count

    try:
        frames = probe_frame_count(media_path)
        if frames is None:
            return
        counter.add(frames, {"task_name": task_name})
    except Exception as exc:
        logger.debug(f"failed to record input frames for {task_name!r}: {exc}")


def record_model_call_metrics(
    *,
    kind: str,
    provider: str,
    model: str,
    success: bool,
    elapsed_s: float,
    retry_count: int,
    token_counts: dict[str, int] | None = None,
    data_entry_id: str | None = None,
) -> None:
    """Best-effort: record one model endpoint call as OTEL metrics.

    No-op when telemetry is disabled. Failures are logged and swallowed so
    observability never affects model-call execution.
    """
    if not _ENABLED:
        return

    labels = {
        "call.kind": kind,
        "provider": provider,
        "model": model,
        "success": str(bool(success)).lower(),
    }
    if data_entry_id is not None:
        # Keep metric cardinality low: presence only on labels; full id on the span.
        labels["data_entry.present"] = "true"
        try:
            from opentelemetry import trace

            set_attributes(trace.get_current_span(), {"data_entry.id": data_entry_id})
        except Exception as exc:
            logger.debug(f"failed to attach data_entry.id to span: {exc}")

    try:
        calls = _metrics.get("model.calls")
        if calls is not None:
            calls.add(1, labels)

        duration = _metrics.get("model.call.duration")
        if duration is not None:
            duration.record(max(0.0, float(elapsed_s)), labels)

        retries = _metrics.get("model.retries")
        if retries is not None and retry_count > 0:
            retries.add(int(retry_count), labels)

        tokens = _metrics.get("model.tokens")
        if tokens is not None and token_counts:
            for token_type, count in token_counts.items():
                token_labels = {**labels, "token.type": token_type}
                tokens.add(int(count), token_labels)
    except Exception as exc:
        logger.debug(f"failed to record model-call metrics: {exc}")


def emit_cost_performance_summary(
    *,
    service_name: str,
    data_entry_id: str,
    media_path: str,
    data_path: str,
    service_elapsed_s: float,
    model_usage: dict[str, object],
    tasks: list[dict[str, object]] | None = None,
    scene_scale: dict[str, object] | None = None,
    degraded: bool = False,
    degradation_reasons: list[str] | None = None,
    notes: list[str] | None = None,
    schema_version: str = "1",
) -> None:
    """Emit a per-scene cost/performance summary as an OpenTelemetry span.

    Prefer this over writing a fixed scene sidecar: parallel workers sharing
    remote storage would otherwise race on ``cost_performance/report.json``.
    No-op when telemetry is disabled.
    """
    import json

    attributes: dict[str, Any] = {
        "cost_performance.schema_version": schema_version,
        "cost_performance.artifact_type": "cost_performance_report",
        "service.name": service_name,
        "data_entry.id": data_entry_id,
        "media.path": media_path,
        "data.path": data_path,
        "service.elapsed_s": max(0.0, float(service_elapsed_s)),
        "degradation.degraded": bool(degraded),
        "model_usage.json": json.dumps(model_usage, sort_keys=True),
        "tasks.json": json.dumps(list(tasks or []), sort_keys=True),
        "scene_scale.json": json.dumps(dict(scene_scale or {}), sort_keys=True),
        "degradation.reasons.json": json.dumps(list(degradation_reasons or [])),
        "notes.json": json.dumps(list(notes or [])),
    }
    try:
        with start_span("cost_performance.report", attributes=attributes):
            pass
    except Exception as exc:
        logger.debug(f"failed to emit cost/performance summary: {exc}")
