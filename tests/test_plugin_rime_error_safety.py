"""Check Rime error details at the stream, logging, and telemetry boundaries."""

from __future__ import annotations

import logging
import traceback

import aiohttp
import pytest
from aiohttp import web
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import APIConnectionError, APIConnectOptions, APIError, APIStatusError
from livekit.agents.telemetry.traces import _DynamicTracer
from livekit.agents.tts import tts as tts_module
from livekit.plugins.rime import TTS

pytestmark = pytest.mark.unit

_SECRET = "customer-secret-marker"


@pytest.mark.parametrize(
    ("mode", "status", "raise_for_status"),
    [
        ("http", 400, False),
        ("http", 500, False),
        ("handshake", 400, False),
        ("handshake", 500, False),
        ("handshake", 400, True),
        ("handshake", 500, True),
        ("transport", 302, False),
    ],
)
async def test_rime_errors_do_not_expose_provider_or_transport_details(
    mode: str,
    status: int,
    raise_for_status: bool,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0

    async def fail_request(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        if mode == "transport":
            # aiohttp rejects this redirect and includes the URL in its exception.
            return web.Response(
                status=302, headers={"Location": f"ftp://localhost/file?token={_SECRET}"}
            )
        return web.Response(status=status, reason=_SECRET, text=_SECRET)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = _DynamicTracer("test-rime-error-safety")
    tracer.set_provider(provider)
    monkeypatch.setattr(tts_module, "tracer", tracer)
    # The plugin must produce safe exceptions even when telemetry redaction is off.
    monkeypatch.setattr(tts_module.telemetry_utils, "redaction_enabled", lambda *_: False)
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    app = web.Application()
    app.router.add_route("*", "/coda/ws", fail_request)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{runner.addresses[0][1]}/coda/ws"
    conn_options = APIConnectOptions(max_retry=1, retry_interval=0, timeout=1)

    try:
        async with aiohttp.ClientSession(raise_for_status=raise_for_status) as session:
            if mode == "handshake":
                tts = TTS(
                    api_key="test-key",
                    websocket_url=url.replace("http:", "ws:"),
                    http_session=session,
                )
                stream = tts.stream(conn_options=conn_options)
                stream.push_text("hello")
                stream.end_input()
            else:
                tts = TTS(api_key="test-key", base_url=url, http_session=session)
                stream = tts.synthesize("hello", conn_options=conn_options)
            try:
                with pytest.raises(APIError) as exc_info:
                    _ = [event async for event in stream]
            finally:
                await stream.aclose()
                await tts.aclose()
    finally:
        await runner.cleanup()
        provider.shutdown()

    error = exc_info.value
    expected_retryable = mode == "transport" or status == 500
    if mode == "transport":
        assert isinstance(error, APIConnectionError)
    else:
        assert isinstance(error, APIStatusError)
        assert error.status_code == status
        assert error.request_id is None
    assert error.body is None
    assert error.retryable is expected_retryable
    assert requests == (2 if expected_retryable else 1)
    assert _SECRET not in str(error)
    assert _SECRET not in repr(error)
    assert _SECRET not in "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert error.__cause__ is None
    assert _SECRET not in caplog.text
    retry_logs = [record for record in caplog.records if "retrying in" in record.getMessage()]
    assert len(retry_logs) == (1 if expected_retryable else 0)
    spans = exporter.get_finished_spans()
    attempts = [span for span in spans if span.name == "tts_request_run"]
    assert len(attempts) == requests
    assert all(span.events for span in attempts)
    assert all(_SECRET not in span.to_json() for span in spans)
