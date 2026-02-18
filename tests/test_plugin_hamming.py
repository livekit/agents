# Copyright 2025 Hamming, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from unittest.mock import MagicMock, patch

import pytest

import livekit.plugins.hamming._setup as _setup_mod
from livekit.plugins.hamming._setup import (
    _OTLP_BASE_PATH,
    DEFAULT_BASE_URL,
    WORKSPACE_KEY_HEADER,
    HammingTelemetry,
    setup_hamming,
)

TEST_API_KEY = "ham_test_key_12345"
CUSTOM_BASE_URL = "https://custom.hamming.ai"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after each test."""
    _setup_mod._telemetry = None
    yield
    # Shutdown providers to avoid leaking background threads (BatchSpanProcessor, etc.)
    telemetry = _setup_mod._telemetry
    if telemetry is not None:
        telemetry.shutdown()
    _setup_mod._telemetry = None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure HAMMING_ env vars don't leak between tests."""
    monkeypatch.delenv("HAMMING_API_KEY", raising=False)
    monkeypatch.delenv("HAMMING_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def mock_livekit_tracer():
    """Mock LiveKit's set_tracer_provider to avoid requiring a full LiveKit setup."""
    with patch("livekit.agents.telemetry.set_tracer_provider") as mock:
        yield mock


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def test_raises_without_api_key():
    """setup_hamming() must raise ValueError when no key is available."""
    with pytest.raises(ValueError, match="API key required"):
        setup_hamming()


def test_api_key_from_param():
    """Explicit api_key parameter takes priority."""
    t = setup_hamming(api_key=TEST_API_KEY)
    assert t.trace_provider is not None


def test_api_key_from_env(monkeypatch):
    """Falls back to HAMMING_API_KEY env var."""
    monkeypatch.setenv("HAMMING_API_KEY", TEST_API_KEY)
    t = setup_hamming()
    assert t.trace_provider is not None


def test_param_key_takes_priority_over_env(monkeypatch):
    """api_key param should win over env var."""
    monkeypatch.setenv("HAMMING_API_KEY", "ham_env_key")

    with patch("livekit.plugins.hamming._setup.OTLPSpanExporter") as mock_exporter:
        setup_hamming(api_key=TEST_API_KEY)
        # The exporter should have been called with the param key, not env key
        call_kwargs = mock_exporter.call_args
        assert call_kwargs.kwargs["headers"][WORKSPACE_KEY_HEADER] == TEST_API_KEY


def test_empty_string_api_key_raises():
    """Empty string api_key should still raise ValueError."""
    with pytest.raises(ValueError, match="API key required"):
        setup_hamming(api_key="")


def test_empty_env_api_key_raises(monkeypatch):
    """Empty HAMMING_API_KEY env var should raise ValueError."""
    monkeypatch.setenv("HAMMING_API_KEY", "")
    with pytest.raises(ValueError, match="API key required"):
        setup_hamming()


# ---------------------------------------------------------------------------
# Base URL resolution
# ---------------------------------------------------------------------------


def test_default_base_url():
    """Uses https://app.hamming.ai when no base_url is specified."""
    with patch("livekit.plugins.hamming._setup.OTLPSpanExporter") as mock_exporter:
        setup_hamming(api_key=TEST_API_KEY)
        endpoint = mock_exporter.call_args.kwargs["endpoint"]
        assert endpoint == f"{DEFAULT_BASE_URL}{_OTLP_BASE_PATH}/traces"


def test_base_url_from_param():
    """Explicit base_url parameter takes priority."""
    with patch("livekit.plugins.hamming._setup.OTLPSpanExporter") as mock_exporter:
        setup_hamming(api_key=TEST_API_KEY, base_url=CUSTOM_BASE_URL)
        endpoint = mock_exporter.call_args.kwargs["endpoint"]
        assert endpoint.startswith(CUSTOM_BASE_URL)


def test_base_url_from_env(monkeypatch):
    """Falls back to HAMMING_BASE_URL env var."""
    monkeypatch.setenv("HAMMING_BASE_URL", CUSTOM_BASE_URL)
    with patch("livekit.plugins.hamming._setup.OTLPSpanExporter") as mock_exporter:
        setup_hamming(api_key=TEST_API_KEY)
        endpoint = mock_exporter.call_args.kwargs["endpoint"]
        assert endpoint.startswith(CUSTOM_BASE_URL)


def test_base_url_trailing_slash_stripped():
    """Trailing slashes in base_url should be stripped."""
    with patch("livekit.plugins.hamming._setup.OTLPSpanExporter") as mock_exporter:
        setup_hamming(api_key=TEST_API_KEY, base_url="https://example.com///")
        endpoint = mock_exporter.call_args.kwargs["endpoint"]
        assert "///" not in endpoint
        assert endpoint == f"https://example.com{_OTLP_BASE_PATH}/traces"


# ---------------------------------------------------------------------------
# Provider creation
# ---------------------------------------------------------------------------


def test_all_providers_created_by_default():
    """All three providers should be created when enable flags are default (True)."""
    t = setup_hamming(api_key=TEST_API_KEY)
    assert t.trace_provider is not None
    assert t.logger_provider is not None
    assert t.meter_provider is not None


def test_disable_traces():
    """enable_traces=False should leave trace_provider as None."""
    t = setup_hamming(api_key=TEST_API_KEY, enable_traces=False)
    assert t.trace_provider is None
    assert t.logger_provider is not None
    assert t.meter_provider is not None


def test_disable_logs():
    """enable_logs=False should leave logger_provider as None."""
    t = setup_hamming(api_key=TEST_API_KEY, enable_logs=False)
    assert t.trace_provider is not None
    assert t.logger_provider is None
    assert t.meter_provider is not None


def test_disable_metrics():
    """enable_metrics=False should leave meter_provider as None."""
    t = setup_hamming(api_key=TEST_API_KEY, enable_metrics=False)
    assert t.trace_provider is not None
    assert t.logger_provider is not None
    assert t.meter_provider is None


def test_disable_all():
    """All providers None when everything is disabled."""
    t = setup_hamming(
        api_key=TEST_API_KEY,
        enable_traces=False,
        enable_logs=False,
        enable_metrics=False,
    )
    assert t.trace_provider is None
    assert t.logger_provider is None
    assert t.meter_provider is None


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------


def test_singleton_returns_same_instance():
    """Calling setup_hamming() twice returns the same HammingTelemetry."""
    t1 = setup_hamming(api_key=TEST_API_KEY)
    t2 = setup_hamming(api_key="ham_different_key")
    assert t1 is t2


def test_singleton_logs_warning():
    """Second call should log a warning about being already configured."""
    setup_hamming(api_key=TEST_API_KEY)
    with patch.object(_setup_mod.logger, "warning") as mock_warn:
        setup_hamming(api_key="ham_other")
        mock_warn.assert_called_once()
        assert "already called" in mock_warn.call_args[0][0]


def test_singleton_not_set_on_error():
    """If setup_hamming raises, singleton should remain None for retry."""
    with pytest.raises(ValueError):
        setup_hamming()  # no key -> raises

    # Should be retryable with a valid key
    t = setup_hamming(api_key=TEST_API_KEY)
    assert t.trace_provider is not None


# ---------------------------------------------------------------------------
# Exporter configuration
# ---------------------------------------------------------------------------


def test_exporters_receive_correct_endpoints():
    """Each exporter should get the right OTLP path suffix."""
    with (
        patch("livekit.plugins.hamming._setup.OTLPSpanExporter") as mock_trace,
        patch("livekit.plugins.hamming._setup.OTLPLogExporter") as mock_log,
        patch("livekit.plugins.hamming._setup.OTLPMetricExporter") as mock_metric,
    ):
        setup_hamming(api_key=TEST_API_KEY)
        base = DEFAULT_BASE_URL + _OTLP_BASE_PATH

        assert mock_trace.call_args.kwargs["endpoint"] == f"{base}/traces"
        assert mock_log.call_args.kwargs["endpoint"] == f"{base}/logs"
        assert mock_metric.call_args.kwargs["endpoint"] == f"{base}/metrics"


def test_exporters_receive_auth_header():
    """All exporters should have the X-Workspace-Key header."""
    with (
        patch("livekit.plugins.hamming._setup.OTLPSpanExporter") as mock_trace,
        patch("livekit.plugins.hamming._setup.OTLPLogExporter") as mock_log,
        patch("livekit.plugins.hamming._setup.OTLPMetricExporter") as mock_metric,
    ):
        setup_hamming(api_key=TEST_API_KEY)
        for mock in (mock_trace, mock_log, mock_metric):
            headers = mock.call_args.kwargs["headers"]
            assert headers[WORKSPACE_KEY_HEADER] == TEST_API_KEY


def test_custom_service_name():
    """service_name should be set on the OTel resource."""
    with patch("livekit.plugins.hamming._setup.Resource.create") as mock_resource:
        mock_resource.return_value = MagicMock()
        setup_hamming(api_key=TEST_API_KEY, service_name="my-custom-agent")
        mock_resource.assert_called_once_with({"service.name": "my-custom-agent"})


# ---------------------------------------------------------------------------
# LiveKit tracer integration
# ---------------------------------------------------------------------------


def test_livekit_set_tracer_provider_called(mock_livekit_tracer):
    """setup_hamming should call LiveKit's set_tracer_provider with metadata."""
    metadata = {"livekit.room_name": "test-room"}
    t = setup_hamming(api_key=TEST_API_KEY, metadata=metadata)
    mock_livekit_tracer.assert_called_once_with(t.trace_provider, metadata=metadata)


def test_livekit_set_tracer_provider_not_called_when_traces_disabled(mock_livekit_tracer):
    """LiveKit tracer should not be set when traces are disabled."""
    setup_hamming(api_key=TEST_API_KEY, enable_traces=False)
    mock_livekit_tracer.assert_not_called()


def test_global_tracer_provider_set():
    """Global OTel trace provider should be set for user spans."""
    with patch("livekit.plugins.hamming._setup.trace.set_tracer_provider") as mock_global:
        t = setup_hamming(api_key=TEST_API_KEY)
        mock_global.assert_called_once_with(t.trace_provider)


# ---------------------------------------------------------------------------
# HammingTelemetry methods
# ---------------------------------------------------------------------------


def test_repr_all_signals():
    t = HammingTelemetry(
        trace_provider=MagicMock(),
        logger_provider=MagicMock(),
        meter_provider=MagicMock(),
    )
    assert repr(t) == "HammingTelemetry(signals=[traces, logs, metrics])"


def test_repr_no_signals():
    t = HammingTelemetry()
    assert repr(t) == "HammingTelemetry(signals=[])"


def test_repr_partial_signals():
    t = HammingTelemetry(trace_provider=MagicMock())
    assert repr(t) == "HammingTelemetry(signals=[traces])"


def test_force_flush_calls_all_providers():
    """force_flush should call force_flush on each configured provider."""
    tp = MagicMock()
    lp = MagicMock()
    mp = MagicMock()
    t = HammingTelemetry(trace_provider=tp, logger_provider=lp, meter_provider=mp)

    t.force_flush(timeout_ms=3000)

    tp.force_flush.assert_called_once_with(timeout_millis=3000)
    lp.force_flush.assert_called_once_with(timeout_millis=3000)
    mp.force_flush.assert_called_once_with(timeout_millis=3000)


def test_force_flush_skips_none_providers():
    """force_flush should not fail when providers are None."""
    t = HammingTelemetry()  # all None
    t.force_flush()  # should not raise


def test_force_flush_resilient_to_errors():
    """If one provider fails flush, others should still be flushed."""
    tp = MagicMock()
    tp.force_flush.side_effect = RuntimeError("trace flush failed")
    lp = MagicMock()
    mp = MagicMock()
    t = HammingTelemetry(trace_provider=tp, logger_provider=lp, meter_provider=mp)

    with pytest.raises(RuntimeError, match="trace flush failed"):
        t.force_flush()

    # All three should have been attempted
    tp.force_flush.assert_called_once()
    lp.force_flush.assert_called_once()
    mp.force_flush.assert_called_once()


def test_force_flush_restores_log_handlers():
    """force_flush should re-attach LoggingHandlers after flushing."""
    from opentelemetry.sdk._logs import LoggingHandler

    root_logger = logging.getLogger()
    handler = LoggingHandler(level=logging.INFO, logger_provider=MagicMock())
    root_logger.addHandler(handler)

    try:
        lp = MagicMock()
        t = HammingTelemetry(logger_provider=lp)

        t.force_flush()

        # Handler should be back on the root logger
        assert handler in root_logger.handlers
    finally:
        root_logger.removeHandler(handler)


def test_force_flush_restores_handlers_on_error():
    """Log handlers should be restored even if flush raises an error."""
    from opentelemetry.sdk._logs import LoggingHandler

    root_logger = logging.getLogger()
    handler = LoggingHandler(level=logging.INFO, logger_provider=MagicMock())
    root_logger.addHandler(handler)

    try:
        lp = MagicMock()
        lp.force_flush.side_effect = RuntimeError("flush boom")
        t = HammingTelemetry(logger_provider=lp)

        with pytest.raises(RuntimeError, match="flush boom"):
            t.force_flush()

        # Handler must be restored despite the error
        assert handler in root_logger.handlers
    finally:
        root_logger.removeHandler(handler)


def test_shutdown_permanently_removes_log_handlers():
    """shutdown should permanently remove LoggingHandlers (terminal operation)."""
    from opentelemetry.sdk._logs import LoggingHandler

    root_logger = logging.getLogger()
    handler = LoggingHandler(level=logging.INFO, logger_provider=MagicMock())
    root_logger.addHandler(handler)

    lp = MagicMock()
    t = HammingTelemetry(logger_provider=lp)

    t.shutdown()

    # Handler should NOT be on the root logger after shutdown
    assert handler not in root_logger.handlers


def test_shutdown_calls_all_providers():
    """shutdown should call shutdown on each configured provider."""
    tp = MagicMock()
    lp = MagicMock()
    mp = MagicMock()
    t = HammingTelemetry(trace_provider=tp, logger_provider=lp, meter_provider=mp)

    t.shutdown()

    tp.shutdown.assert_called_once()
    lp.shutdown.assert_called_once()
    mp.shutdown.assert_called_once()


def test_shutdown_resilient_to_errors():
    """If one provider fails shutdown, others should still be shut down."""
    tp = MagicMock()
    lp = MagicMock()
    lp.shutdown.side_effect = RuntimeError("log shutdown failed")
    mp = MagicMock()
    t = HammingTelemetry(trace_provider=tp, logger_provider=lp, meter_provider=mp)

    with pytest.raises(RuntimeError, match="log shutdown failed"):
        t.shutdown()

    tp.shutdown.assert_called_once()
    lp.shutdown.assert_called_once()
    mp.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Log handler setup
# ---------------------------------------------------------------------------


def test_log_handler_added_to_root_logger():
    """A LoggingHandler should be added to the root logger when logs are enabled."""
    root_logger = logging.getLogger()
    handlers_before = len(root_logger.handlers)

    setup_hamming(api_key=TEST_API_KEY, enable_logs=True)

    handlers_after = len(root_logger.handlers)
    assert handlers_after == handlers_before + 1

    # The last handler should be a LoggingHandler
    added_handler = root_logger.handlers[-1]
    from opentelemetry.sdk._logs import LoggingHandler

    assert isinstance(added_handler, LoggingHandler)

    # Clean up so we don't pollute other tests
    root_logger.removeHandler(added_handler)


def test_no_log_handler_when_logs_disabled():
    """No LoggingHandler should be added when logs are disabled."""
    root_logger = logging.getLogger()
    handlers_before = len(root_logger.handlers)

    setup_hamming(api_key=TEST_API_KEY, enable_logs=False)

    assert len(root_logger.handlers) == handlers_before
