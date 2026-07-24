"""Unit tests for livekit.agents.beta.gtm_telemetry.

Covers: accumulation of turns and tool records, tool error precedence with
batch backfill, batch-only untimed records, the v3 deferred-merge rule,
metrics aggregation (including None when no data), HMAC webhook signing,
retry/4xx/exhaustion behavior, adapters, lazy http_context resolution,
and an end-to-end fake-session integration test.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import AgentSession, utils
from livekit.agents.beta.gtm_telemetry import (
    CallMetrics,
    PostCallReport,
    PostCallTelemetryCollector,
    ToolInvocationRecord,
    TranscriptTurn,
    WebhookDispatcher,
    to_hubspot_engagement,
    to_salesforce_task,
)
from livekit.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.metrics import LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    ToolCallEnded,
    ToolCallStarted,
    ToolExecutionUpdatedEvent,
)

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


# -- metrics factory helpers --


def _llm_metrics(ttft: float) -> LLMMetrics:
    return LLMMetrics(
        label="fake",
        request_id="r1",
        timestamp=0.0,
        duration=0.5,
        ttft=ttft,
        cancelled=False,
        completion_tokens=10,
        prompt_tokens=20,
        prompt_cached_tokens=0,
        total_tokens=30,
        tokens_per_second=20.0,
    )


def _stt_metrics(audio_duration: float) -> STTMetrics:
    return STTMetrics(
        label="fake",
        request_id="r2",
        timestamp=0.0,
        duration=0.0,
        audio_duration=audio_duration,
        streamed=True,
    )


def _tts_metrics(audio_duration: float) -> TTSMetrics:
    return TTSMetrics(
        label="fake",
        request_id="r3",
        timestamp=0.0,
        ttfb=0.1,
        duration=0.4,
        audio_duration=audio_duration,
        cancelled=False,
        characters_count=42,
        streamed=True,
    )


def _rt_metrics(ttft: float) -> RealtimeModelMetrics:
    return RealtimeModelMetrics(
        request_id="r4",
        timestamp=0.0,
        ttft=ttft,
        input_token_details=RealtimeModelMetrics.InputTokenDetails(),
        output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
    )


# -- test server helper --


@contextlib.asynccontextmanager
async def _webhook_server(handler):  # type: ignore[no-untyped-def]
    """Start an aiohttp TestServer with POST /webhook."""
    app = web.Application()
    app.router.add_post("/webhook", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        url = str(server.make_url("/webhook"))
        yield url, session
    finally:
        await session.close()
        await server.close()


# -- tests --


async def test_collector_accumulates_turns_and_tools() -> None:
    """Collector records transcript turns and timed tool invocations."""
    session = AgentSession()
    collector = PostCallTelemetryCollector(session)
    collector.attach()
    try:
        # user turn
        collector._on_conversation_item_added(
            ConversationItemAddedEvent(
                item=ChatMessage(role="user", content=["hi"], created_at=1.0)
            )
        )
        # tool started
        collector._on_tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallStarted(
                    function_call=FunctionCall(
                        call_id="c1",
                        name="lookup",
                        arguments='{"q": "acme"}',
                        created_at=100.0,
                    )
                ),
                created_at=100.0,
            )
        )
        # tool ended
        collector._on_tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallEnded(id="c1", call_id="c1", status="done", message="found"),
                created_at=100.25,
            )
        )

        report = collector.generate_report()
        assert len(report.turns) == 1
        assert report.turns[0].speaker == "user"
        assert report.turns[0].text == "hi"

        assert len(report.tool_invocations) == 1
        rec = report.tool_invocations[0]
        assert rec.arguments == {"q": "acme"}
        assert rec.duration_ms == pytest.approx(250.0)
        assert rec.status == "done"
        assert rec.result == "found"
    finally:
        await collector.aclose()


async def test_tool_error_precedence_and_batch_backfill() -> None:
    """ToolCallEnded.message (diagnostic text) is never overwritten by the
    redacted batch output."""
    session = AgentSession()
    collector = PostCallTelemetryCollector(session)
    collector.attach()
    try:
        fc = FunctionCall(call_id="c1", name="lookup", arguments="{}", created_at=100.0)
        # started
        collector._on_tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallStarted(function_call=fc),
                created_at=100.0,
            )
        )
        # ended with diagnostic error text
        collector._on_tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallEnded(
                    id="c1", call_id="c1", status="error", message="ValueError: bad input"
                ),
                created_at=100.5,
            )
        )
        # batch arrives with redacted output
        collector._on_function_tools_executed(
            FunctionToolsExecutedEvent(
                function_calls=[fc],
                function_call_outputs=[
                    FunctionCallOutput(
                        call_id="c1",
                        output="An internal error occurred",
                        is_error=True,
                    )
                ],
            )
        )

        report = collector.generate_report()
        assert report.tool_invocations[0].error == "ValueError: bad input"
        assert report.metrics.failed_tool_calls == 1
    finally:
        await collector.aclose()


async def test_batch_only_call_creates_untimed_record() -> None:
    """Calls that never reach the executor still appear as untimed records."""
    session = AgentSession()
    collector = PostCallTelemetryCollector(session)
    collector.attach()
    try:
        fc = FunctionCall(call_id="c2", name="unknown_tool", arguments="{}", created_at=50.0)
        collector._on_function_tools_executed(
            FunctionToolsExecutedEvent(
                function_calls=[fc],
                function_call_outputs=[
                    FunctionCallOutput(call_id="c2", output="tool not found", is_error=True)
                ],
            )
        )

        report = collector.generate_report()
        assert len(report.tool_invocations) == 1
        rec = report.tool_invocations[0]
        assert rec.duration_ms is None
        assert rec.completed_at is None
        assert rec.status == "error"
        assert rec.error == "tool not found"
        assert report.metrics.total_tool_calls == 1
        assert report.metrics.failed_tool_calls == 1
    finally:
        await collector.aclose()


async def test_deferred_terminal_none_message_keeps_batch_result() -> None:
    """v3 deferred-merge: batch backfills result on a pending record;
    terminal ToolCallEnded(message=None) does not erase it."""
    session = AgentSession()
    collector = PostCallTelemetryCollector(session)
    collector.attach()
    try:
        fc = FunctionCall(call_id="c3", name="search", arguments='{"q": "x"}', created_at=200.0)
        # started
        collector._on_tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallStarted(function_call=fc),
                created_at=200.0,
            )
        )
        # batch arrives before terminal (deferred scenario): backfills result
        collector._on_function_tools_executed(
            FunctionToolsExecutedEvent(
                function_calls=[fc],
                function_call_outputs=[
                    FunctionCallOutput(call_id="c3", output="found 3 rows", is_error=False)
                ],
            )
        )
        # terminal with message=None — must NOT erase the batch result
        collector._on_tool_execution_updated(
            ToolExecutionUpdatedEvent(
                update=ToolCallEnded(id="c3", call_id="c3", status="done", message=None),
                created_at=200.8,
            )
        )

        report = collector.generate_report()
        rec = report.tool_invocations[0]
        assert rec.result == "found 3 rows"
        assert rec.status == "done"
        assert rec.duration_ms == pytest.approx(800.0)
        assert rec.completed_at == pytest.approx(200.8)
    finally:
        await collector.aclose()


async def test_metrics_aggregation() -> None:
    """Metrics from events are correctly aggregated (RealtimeModelMetrics ttft=-1 skipped)."""
    session = AgentSession()
    collector = PostCallTelemetryCollector(session)
    collector.attach()
    try:
        collector._on_metrics_collected(MetricsCollectedEvent(metrics=_llm_metrics(0.2)))
        collector._on_metrics_collected(MetricsCollectedEvent(metrics=_stt_metrics(3.0)))
        collector._on_metrics_collected(MetricsCollectedEvent(metrics=_tts_metrics(4.5)))
        collector._on_metrics_collected(MetricsCollectedEvent(metrics=_rt_metrics(-1)))

        report = collector.generate_report()
        assert report.metrics.avg_llm_ttft_ms == pytest.approx(200.0)
        assert report.metrics.user_speech_duration_seconds == pytest.approx(3.0)
        assert report.metrics.agent_speech_duration_seconds == pytest.approx(4.5)
    finally:
        await collector.aclose()


async def test_metrics_none_when_no_data() -> None:
    """Speech durations and avg_llm_ttft_ms are None when no data was observed."""
    session = AgentSession()
    collector = PostCallTelemetryCollector(session)
    collector.attach()
    try:
        report = collector.generate_report()
        assert report.metrics.user_speech_duration_seconds is None
        assert report.metrics.agent_speech_duration_seconds is None
        assert report.metrics.avg_llm_ttft_ms is None
    finally:
        await collector.aclose()


async def test_webhook_hmac_signature() -> None:
    """Dispatcher sends correct HMAC-SHA256 signature and a valid JSON report."""
    captured: dict[str, Any] = {}
    secret = "test-secret-42"

    async def handler(request: web.Request) -> web.Response:
        captured["body_bytes"] = await request.read()
        captured["headers"] = dict(request.headers)
        return web.Response(status=200)

    async with _webhook_server(handler) as (url, http_session):
        dispatcher = WebhookDispatcher(url, webhook_secret=secret, http_session=http_session)
        report = PostCallReport(
            room_name="test-room",
            created_at=1700000000.0,
        )
        result = await dispatcher.dispatch(report)

    assert result is True
    body = captured["body_bytes"]
    expected_sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert captured["headers"]["X-LiveKit-Signature"] == expected_sig

    # verify the body round-trips
    payload = json.loads(body)
    assert payload["room_name"] == "test-room"
    assert payload["type"] == "post_call_report"


async def test_webhook_retry_then_success() -> None:
    """Dispatcher retries on 500 and succeeds on the second attempt."""
    call_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return web.Response(status=500)
        return web.Response(status=200)

    async with _webhook_server(handler) as (url, http_session):
        dispatcher = WebhookDispatcher(url, http_session=http_session, base_delay=0.01)
        result = await dispatcher.dispatch(PostCallReport(created_at=1.0))

    assert result is True
    assert call_count == 2


async def test_webhook_no_retry_on_4xx() -> None:
    """Dispatcher does not retry 4xx responses."""
    call_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal call_count
        call_count += 1
        return web.Response(status=400)

    async with _webhook_server(handler) as (url, http_session):
        dispatcher = WebhookDispatcher(url, http_session=http_session)
        result = await dispatcher.dispatch(PostCallReport(created_at=1.0))

    assert result is False
    assert call_count == 1


async def test_webhook_exhausted_no_trailing_sleep() -> None:
    """Dispatcher sleeps only between attempts, not after the final one.
    With base_delay=d and max_retries=3, exactly 2 sleeps of [d, 2d]."""
    sleeps: list[float] = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        sleeps.append(delay)
        await original_sleep(delay)

    call_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal call_count
        call_count += 1
        return web.Response(status=500)

    async with _webhook_server(handler) as (url, http_session):
        import unittest.mock

        with unittest.mock.patch("asyncio.sleep", side_effect=mock_sleep):
            dispatcher = WebhookDispatcher(
                url, http_session=http_session, base_delay=0.01, max_retries=3
            )
            result = await dispatcher.dispatch(PostCallReport(created_at=1.0))

    assert result is False
    assert call_count == 3
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(0.01)
    assert sleeps[1] == pytest.approx(0.02)


async def test_adapters() -> None:
    """CRM adapter outputs have the expected shape."""
    report = PostCallReport(
        room_name="sales-call-1",
        created_at=1700000000.0,
        turns=[
            TranscriptTurn(speaker="user", text="Hello", timestamp=1700000001.0),
            TranscriptTurn(speaker="agent", text="Hi there!", timestamp=1700000002.0),
        ],
        tool_invocations=[
            ToolInvocationRecord(
                tool_name="lookup",
                call_id="t1",
                arguments={"email": "a@b.com"},
                result="found",
                status="done",
                started_at=1700000003.0,
                completed_at=1700000003.5,
                duration_ms=500.0,
            )
        ],
        metrics=CallMetrics(
            total_duration_seconds=120.0,
            user_speech_duration_seconds=45.0,
            agent_speech_duration_seconds=60.0,
            total_tool_calls=1,
            failed_tool_calls=0,
            avg_llm_ttft_ms=150.0,
        ),
    )

    hs = to_hubspot_engagement(report)
    assert "properties" in hs
    props = hs["properties"]
    assert props["hs_call_duration"] == "120000"
    assert props["hs_timestamp"] == 1700000000000
    assert "Hello" in props["hs_call_body"]
    assert props["hs_call_status"] == "COMPLETED"

    sf = to_salesforce_task(report)
    assert sf["Status"] == "Completed"
    assert sf["CallDurationInSeconds"] == 120
    assert "sales-call-1" in sf["Subject"]
    assert "Hello" in sf["Description"]
    assert sf["ActivityDate"] == "2023-11-14"


async def test_adapters_none_durations() -> None:
    """Adapters handle None speech durations gracefully."""
    report = PostCallReport(
        room_name="rt-call",
        created_at=1700000000.0,
        metrics=CallMetrics(
            total_duration_seconds=60.0,
            user_speech_duration_seconds=None,
            agent_speech_duration_seconds=None,
            avg_llm_ttft_ms=None,
        ),
    )
    hs = to_hubspot_engagement(report)
    assert hs["properties"]["hs_call_duration"] == "60000"

    sf = to_salesforce_task(report)
    assert sf["CallDurationInSeconds"] == 60


async def test_dispatch_resolves_http_context_lazily() -> None:
    """Constructing a WebhookDispatcher without an http_session does not raise;
    resolution happens inside dispatch() via http_context."""
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["body"] = await request.read()
        return web.Response(status=200)

    app = web.Application()
    app.router.add_post("/webhook", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/webhook"))
        # construct OUTSIDE any http context — must not raise
        dispatcher = WebhookDispatcher(url)

        async with utils.http_context.open():
            result = await dispatcher.dispatch(PostCallReport(created_at=1.0))
        assert result is True
        assert len(captured["body"]) > 0
    finally:
        await server.close()


async def test_generate_report_raises_if_not_attached() -> None:
    """generate_report() raises RuntimeError when attach() was never called."""
    session = AgentSession()
    collector = PostCallTelemetryCollector(session)
    with pytest.raises(RuntimeError, match="not attached"):
        collector.generate_report()


async def test_end_to_end_fake_session() -> None:
    """Full integration test: attach to a fake-session run, receive a webhook."""
    from livekit.agents import Agent

    captured_bodies: list[bytes] = []

    async def handler(request: web.Request) -> web.Response:
        captured_bodies.append(await request.read())
        return web.Response(status=200)

    app = web.Application()
    app.router.add_post("/webhook", handler)
    server = TestServer(app)
    await server.start_server()
    http_session = aiohttp.ClientSession()

    try:
        url = str(server.make_url("/webhook"))

        actions = FakeActions()
        actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
        actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.3)
        actions.add_tts(2.0, ttfb=0.2, duration=0.3)

        session = create_session(actions, speed_factor=1)

        class TestAgent(Agent):
            def __init__(self) -> None:
                super().__init__(instructions="You are a test assistant.")

        dispatcher = WebhookDispatcher(url, http_session=http_session)
        collector = PostCallTelemetryCollector(
            session,
            room_name="e2e-test-room",
            dispatcher=dispatcher,
        )
        collector.attach()

        await asyncio.wait_for(run_session(session, TestAgent()), timeout=60.0)
        # run_session calls session.aclose() which emits close -> spawns flush task
        await collector.aflush()

        report = collector.generate_report()
        assert report.metrics.total_duration_seconds > 0
        # The fake session produces at least one user + one agent turn
        user_turns = [t for t in report.turns if t.speaker == "user"]
        agent_turns = [t for t in report.turns if t.speaker == "agent"]
        assert len(user_turns) >= 1
        assert len(agent_turns) >= 1

        # webhook was delivered
        assert len(captured_bodies) >= 1
        payload = json.loads(captured_bodies[0])
        assert payload["room_name"] == "e2e-test-room"
        assert payload["type"] == "post_call_report"
    finally:
        await collector.aclose()
        await http_session.close()
        await server.close()
