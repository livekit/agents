"""Regression tests for issue #7029.

``_setup_cloud_tracer`` must create OTel SDK providers with
``shutdown_on_exit=False``.  The SDK default is ``True``, which registers
``atexit`` handlers calling ``provider.shutdown() → worker_thread.join()``.
When the OTLP endpoint is unreachable the worker threads may be stuck
exporting, blocking process exit indefinitely.

LiveKit manages shutdown explicitly via ``_shutdown_telemetry()`` with a
bounded wall-clock timeout, so the SDK atexit handlers are redundant and
dangerous.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
from opentelemetry import metrics as metrics_api
from opentelemetry._logs import get_logger_provider
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider

from livekit.agents.telemetry.traces import tracer

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


_ENV = {**os.environ, "LIVEKIT_API_KEY": "dummy", "LIVEKIT_API_SECRET": "dummy"}


def _setup() -> None:
    """Call _setup_cloud_tracer with an unreachable endpoint."""
    from livekit.agents.telemetry.traces import _setup_cloud_tracer

    _setup_cloud_tracer(
        room_id="room",
        job_id="job",
        agent_name="agent",
        observability_url="http://127.0.0.1:1",
        enable_traces=True,
        enable_logs=True,
    )


# ---------------------------------------------------------------------------
# Test A: providers must be created with shutdown_on_exit=False
# ---------------------------------------------------------------------------


@pytest.mark.no_concurrent
def test_providers_created_without_atexit() -> None:
    """All three OTel providers created by _setup_cloud_tracer must have
    shutdown_on_exit=False so they don't register blocking atexit handlers."""
    # save originals
    orig_tp = tracer._tracer_provider
    orig_lp = get_logger_provider()
    orig_mp = metrics_api.get_meter_provider()
    try:
        _setup()

        if isinstance(tracer._tracer_provider, trace_sdk.TracerProvider):
            tp = tracer._tracer_provider
            assert tp._atexit_handler is None, "TracerProvider must not register atexit handler"

        lp = get_logger_provider()
        if isinstance(lp, LoggerProvider):
            assert lp._at_exit_handler is None, "LoggerProvider must not register atexit handler"

        mp = metrics_api.get_meter_provider()
        if isinstance(mp, SdkMeterProvider):
            assert mp._atexit_handler is None, "MeterProvider must not register atexit handler"
    finally:
        from opentelemetry._logs import set_logger_provider

        from livekit.agents.telemetry import set_tracer_provider

        set_tracer_provider(orig_tp)  # type: ignore[arg-type]
        set_logger_provider(orig_lp)  # type: ignore[arg-type]
        metrics_api.set_meter_provider(orig_mp)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test B: process with unreachable endpoint must exit promptly
# ---------------------------------------------------------------------------


@pytest.mark.no_concurrent
def test_offline_exit_no_hang() -> None:
    """A process that sets up telemetry with an unreachable endpoint and
    exits without calling _shutdown_telemetry must not hang on atexit."""
    code = (
        "from livekit.agents.telemetry.traces import _setup_cloud_tracer\n"
        "_setup_cloud_tracer(\n"
        "    room_id='r', job_id='j', agent_name='a',\n"
        "    observability_url='http://127.0.0.1:1',\n"
        "    enable_traces=True, enable_logs=True,\n"
        ")\n"
    )
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        env=_ENV,
    )
    elapsed = time.monotonic() - start
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert elapsed < 5.0, (
        f"Process took {elapsed:.1f}s to exit — likely deadlocked on atexit handler"
    )
