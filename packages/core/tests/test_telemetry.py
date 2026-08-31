# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from pathlib import Path

import pytest
from core.utils import telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset module state and clear OTEL_* env so each test starts from no-op."""
    for var in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_EXPORTER",
        "OTEL_METRICS_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
        "OTEL_SERVICE_NAME",
        "OTEL_SERVICE_VERSION",
        "OTEL_SERVICE_NAMESPACE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(telemetry, "_INITIALIZED", False)
    monkeypatch.setattr(telemetry, "_ENABLED", False)
    monkeypatch.setattr(telemetry, "RUN_ID", None)
    monkeypatch.setattr(telemetry, "_tracer_provider", None)
    monkeypatch.setattr(telemetry, "_meter_provider", None)
    monkeypatch.setattr(telemetry, "_metrics", {})
    yield


def _enable_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable telemetry with no exporters (real providers, no network/stdout)."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    telemetry.setup_telemetry(service_name="svc")


# --- config predicates ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("yes", True), ("false", False), ("", False)],
)
def test_is_disabled(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", value)
    assert telemetry._is_disabled() is expected


def test_is_configured_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    assert telemetry._is_configured() is True


def test_is_configured_false_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "   ")
    assert telemetry._is_configured() is False


def test_is_configured_true_when_both_console_without_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "console")
    assert telemetry._is_configured() is True


@pytest.mark.parametrize("var", ["OTEL_TRACES_EXPORTER", "OTEL_METRICS_EXPORTER"])
def test_is_configured_false_when_only_one_console(
    monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    # Only one signal set to console (the other defaults to otlp) is not enough.
    monkeypatch.setenv(var, "console")
    assert telemetry._is_configured() is False


def test_is_configured_false_for_otlp_exporter_without_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    assert telemetry._is_configured() is False


# --- resource & exporter construction ---------------------------------------


def test_build_resource_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry.sdk.resources")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-name")
    monkeypatch.setenv("OTEL_SERVICE_VERSION", "9.9")
    monkeypatch.setenv("OTEL_SERVICE_NAMESPACE", "ns")
    resource = telemetry._build_resource("arg-name", None)
    assert resource.attributes["service.name"] == "arg-name"  # arg beats env
    assert resource.attributes["service.version"] == "9.9"  # env used when arg is None
    assert resource.attributes["service.namespace"] == "ns"


def test_build_resource_defaults() -> None:
    pytest.importorskip("opentelemetry.sdk.resources")
    resource = telemetry._build_resource(None, None)
    assert telemetry._TRACER_NAME == "auto-labeling"
    assert telemetry._METER_NAME == "auto-labeling"
    assert telemetry._DEFAULT_SERVICE_NAME == "auto-labeling"
    assert telemetry._DEFAULT_SERVICE_NAMESPACE == "auto-labeling"
    assert resource.attributes["service.name"] == "auto-labeling"
    assert resource.attributes["service.namespace"] == "auto-labeling"
    assert resource.attributes["service.version"] == telemetry._DEFAULT_SERVICE_VERSION


def test_resolve_otlp_protocol_defaults_to_grpc() -> None:
    assert telemetry._resolve_otlp_protocol("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") == "grpc"


def test_resolve_otlp_protocol_signal_var_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    assert telemetry._resolve_otlp_protocol("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") == "http/protobuf"


def test_resolve_otlp_protocol_falls_back_to_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    assert telemetry._resolve_otlp_protocol("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") == "http/protobuf"


def test_resolve_otlp_protocol_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "bogus")
    with pytest.raises(RuntimeError):
        telemetry._resolve_otlp_protocol("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")


def test_delta_temporality_maps_cumulative_kinds() -> None:
    pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics import Counter, Histogram, ObservableCounter  # noqa: PLC0415
    from opentelemetry.sdk.metrics.export import AggregationTemporality  # noqa: PLC0415

    mapping = telemetry._delta_temporality()
    assert mapping[Counter] == AggregationTemporality.DELTA
    assert mapping[Histogram] == AggregationTemporality.DELTA
    assert mapping[ObservableCounter] == AggregationTemporality.DELTA


def test_make_span_exporter_console() -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # noqa: PLC0415

    assert isinstance(telemetry._make_span_exporter("console"), ConsoleSpanExporter)


def test_make_metric_exporter_console() -> None:
    pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter  # noqa: PLC0415

    assert isinstance(telemetry._make_metric_exporter("console"), ConsoleMetricExporter)


# --- instruments ------------------------------------------------------------


def test_create_instruments_registers_delta_counter() -> None:
    pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: PLC0415

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    telemetry._create_instruments(provider)

    counter = telemetry._metrics["task.frames_processed"]
    counter.add(3, {"task_name": "captioning"})

    data = reader.get_metrics_data()
    assert data is not None
    values = [
        getattr(point, "value", None)
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        for point in metric.data.data_points
    ]
    assert 3 in values


# --- setup_telemetry --------------------------------------------------------


def test_setup_is_noop_without_endpoint() -> None:
    telemetry.setup_telemetry(service_name="svc")
    assert telemetry.is_enabled() is False
    # RUN_ID is generated even on the no-op path so callers can read it.
    assert telemetry.RUN_ID is not None


def test_setup_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    telemetry.setup_telemetry(service_name="svc")
    assert telemetry.is_enabled() is False


def test_setup_enables_and_sets_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    _enable_local(monkeypatch)
    assert telemetry.is_enabled() is True
    assert telemetry.RUN_ID is not None


def test_setup_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry.setup_telemetry(service_name="svc")
    first = telemetry.RUN_ID
    telemetry.setup_telemetry(service_name="other")
    assert telemetry.RUN_ID == first


# --- shutdown ---------------------------------------------------------------


def test_shutdown_disables_and_is_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    _enable_local(monkeypatch)
    assert telemetry.is_enabled() is True
    telemetry.shutdown()
    assert telemetry.is_enabled() is False
    telemetry.shutdown()  # safe to call twice


# --- start_span -------------------------------------------------------------


def test_start_span_yields_none_when_disabled() -> None:
    telemetry.setup_telemetry(service_name="svc")
    with telemetry.start_span("service.run", {"k": "v"}) as span:
        assert span is None


def test_start_span_records_attributes_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    _enable_local(monkeypatch)
    with telemetry.start_span("service.run", {"k": "v", "drop": None}) as span:
        assert span is not None
        assert span.attributes["k"] == "v"
        assert "drop" not in span.attributes
        assert span.attributes["run.id"] == telemetry.RUN_ID
        assert span.attributes["host.name"] == telemetry._HOST_NAME


# --- set_attributes ---------------------------------------------------------


def test_set_attributes_drops_none_and_sets_rest() -> None:
    class _FakeSpan:
        def __init__(self) -> None:
            self.attrs: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attrs[key] = value

    span = _FakeSpan()
    telemetry.set_attributes(span, {"a": 1, "b": None, "c": "x"})
    assert span.attrs == {"a": 1, "c": "x"}


def test_set_attributes_noop_on_none_span() -> None:
    telemetry.set_attributes(None, {"a": 1})  # must not raise


# --- record_input_frames ----------------------------------------------------


def test_record_input_frames_is_noop_when_disabled() -> None:
    telemetry.setup_telemetry(service_name="svc")
    # Must not probe media or raise on the no-op path.
    telemetry.record_input_frames("captioning", "/does/not/exist.mp4")


def test_record_input_frames_increments_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("opentelemetry.sdk.metrics")
    _enable_local(monkeypatch)
    image = tmp_path / "frame.png"
    image.write_bytes(b"x")
    telemetry.record_input_frames("captioning", image)  # must not raise
    assert telemetry._metrics.get("task.frames_processed") is not None


def test_record_model_call_metrics_is_noop_when_disabled() -> None:
    telemetry.setup_telemetry(service_name="svc")
    telemetry.record_model_call_metrics(
        kind="llm",
        provider="openai-compatible",
        model="test",
        success=True,
        elapsed_s=0.1,
        retry_count=1,
        token_counts={"total_tokens": 3},
    )


def test_record_model_call_metrics_registers_instruments_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("opentelemetry.sdk.metrics")
    _enable_local(monkeypatch)
    telemetry.record_model_call_metrics(
        kind="vlm",
        provider="openai-compatible",
        model="test-vlm",
        success=True,
        elapsed_s=0.25,
        retry_count=2,
        token_counts={"prompt_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        data_entry_id="entry-1",
    )
    assert telemetry._metrics.get("model.calls") is not None
    assert telemetry._metrics.get("model.call.duration") is not None
    assert telemetry._metrics.get("model.tokens") is not None
    assert telemetry._metrics.get("model.retries") is not None


def test_record_model_call_metrics_avoids_high_cardinality_entry_id_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """data_entry.id belongs on spans; metrics only get a low-cardinality presence flag."""
    pytest.importorskip("opentelemetry")

    class _FakeSpan:
        def __init__(self) -> None:
            self.attrs: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attrs[key] = value

    class _CapturingInstrument:
        def __init__(self) -> None:
            self.calls: list[tuple[object, dict[str, object] | None]] = []

        def add(self, amount: object, attributes: dict[str, object] | None = None) -> None:
            self.calls.append((amount, attributes))

        def record(self, amount: object, attributes: dict[str, object] | None = None) -> None:
            self.calls.append((amount, attributes))

    span = _FakeSpan()
    calls = _CapturingInstrument()
    duration = _CapturingInstrument()
    retries = _CapturingInstrument()
    tokens = _CapturingInstrument()

    monkeypatch.setattr(telemetry, "_ENABLED", True)
    monkeypatch.setattr(
        telemetry,
        "_metrics",
        {
            "model.calls": calls,
            "model.call.duration": duration,
            "model.retries": retries,
            "model.tokens": tokens,
        },
    )
    monkeypatch.setattr(
        "opentelemetry.trace.get_current_span",
        lambda: span,
    )

    telemetry.record_model_call_metrics(
        kind="vlm",
        provider="openai-compatible",
        model="test-vlm",
        success=True,
        elapsed_s=0.25,
        retry_count=2,
        token_counts={"prompt_tokens": 4, "total_tokens": 5},
        data_entry_id="entry-1",
    )

    assert span.attrs == {"data_entry.id": "entry-1"}
    assert calls.calls
    metric_labels = calls.calls[0][1]
    assert metric_labels is not None
    assert metric_labels.get("data_entry.present") == "true"
    assert "data_entry.id" not in metric_labels
    assert all("data_entry.id" not in (attrs or {}) for _, attrs in duration.calls)
    assert all("data_entry.id" not in (attrs or {}) for _, attrs in retries.calls)
    assert all("data_entry.id" not in (attrs or {}) for _, attrs in tokens.calls)
    assert all((attrs or {}).get("data_entry.present") == "true" for _, attrs in tokens.calls)


def test_record_model_call_metrics_omits_presence_flag_without_entry_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CapturingInstrument:
        def __init__(self) -> None:
            self.calls: list[tuple[object, dict[str, object] | None]] = []

        def add(self, amount: object, attributes: dict[str, object] | None = None) -> None:
            self.calls.append((amount, attributes))

        def record(self, amount: object, attributes: dict[str, object] | None = None) -> None:
            self.calls.append((amount, attributes))

    calls = _CapturingInstrument()
    monkeypatch.setattr(telemetry, "_ENABLED", True)
    monkeypatch.setattr(telemetry, "_metrics", {"model.calls": calls})

    telemetry.record_model_call_metrics(
        kind="llm",
        provider="openai-compatible",
        model="test",
        success=False,
        elapsed_s=0.1,
        retry_count=0,
    )

    assert calls.calls
    metric_labels = calls.calls[0][1]
    assert metric_labels is not None
    assert "data_entry.present" not in metric_labels
    assert "data_entry.id" not in metric_labels


def test_emit_cost_performance_summary_is_noop_when_disabled() -> None:
    telemetry.setup_telemetry(service_name="svc")
    telemetry.emit_cost_performance_summary(
        service_name="svc",
        data_entry_id="entry-1",
        media_path="clip.mp4",
        data_path="scene",
        service_elapsed_s=1.0,
        model_usage={"total_calls_observed": 0},
    )
