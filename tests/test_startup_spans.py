"""Startup, shutdown, and dispatch telemetry.

Covers the dispatch timeline carried from the worker to the job process (``StartJobRequest``
round trip and the ``job_entrypoint`` events/latencies), the ``room_connect`` span around
``JobContext.connect``, and the session-level spans and events a full fake session produces:
``session_start`` grouping the startup work, ``session_close`` grouping teardown, and state-change
events on ``agent_session``."""

from __future__ import annotations

import io
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, JobContext
from livekit.agents.ipc.job_proc_lazy_main import (
    _callback_name,
    _record_dispatch_timeline,
    _server_timestamp_seconds,
)
from livekit.agents.ipc.proto import StartJobRequest
from livekit.agents.job import AutoSubscribe, JobAcceptArguments, RunningJobInfo
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.protocol import agent as agent_proto

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


def _spans(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _job(**state: int) -> agent_proto.Job:
    job = agent_proto.Job(id="AJ_1", dispatch_id="AD_1", agent_name="demo")
    job.room.name = "room-1"
    job.room.sid = "RM_1"
    job.state.worker_id = "W_1"
    job.state.agent_id = "AG_1"
    for k, v in state.items():
        setattr(job.state, k, v)
    return job


def _info(**timestamps: float) -> RunningJobInfo:
    return RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="agent", identity="agent-1", metadata="{}"),
        job=_job(),
        url="wss://example.livekit.cloud",
        token="tok",
        worker_id="W_1",
        fake_job=False,
        **timestamps,
    )


# -- dispatch timeline --


def test_start_job_request_round_trips_dispatch_timestamps() -> None:
    req = StartJobRequest()
    req.running_job = _info(
        received_at=1_700_000_000.1,
        accepted_at=1_700_000_000.2,
        assigned_at=1_700_000_000.5,
        launched_at=1_700_000_000.6,
    )
    buf = io.BytesIO()
    req.write(buf)
    buf.seek(0)

    decoded = StartJobRequest()
    decoded.read(buf)
    out = decoded.running_job
    assert out.job.id == "AJ_1" and out.worker_id == "W_1" and out.token == "tok"
    assert out.received_at == pytest.approx(1_700_000_000.1)
    assert out.accepted_at == pytest.approx(1_700_000_000.2)
    assert out.assigned_at == pytest.approx(1_700_000_000.5)
    assert out.launched_at == pytest.approx(1_700_000_000.6)


def test_dispatch_timeline_events_and_latencies(span_exporter: InMemorySpanExporter) -> None:
    t0 = 1_700_000_000.0  # realistic unix time: the server timestamp unit detection needs it
    info = _info(received_at=t0, accepted_at=t0 + 0.2, assigned_at=t0 + 0.5, launched_at=t0 + 0.6)
    info.job.state.started_at = int((t0 + 0.05) * 1e9)  # server-side unix nanoseconds

    with tracer.start_as_current_span("job_entrypoint", start_time=int(t0 * 1e9)) as span:
        _record_dispatch_timeline(span, info, entrypoint_started_at=t0 + 1.0)

    [entry] = _spans(span_exporter, "job_entrypoint")
    attrs = entry.attributes or {}
    assert attrs[trace_types.ATTR_JOB_ACCEPT_LATENCY] == pytest.approx(0.2)
    assert attrs[trace_types.ATTR_JOB_ASSIGNMENT_LATENCY] == pytest.approx(0.3)
    assert attrs[trace_types.ATTR_JOB_LAUNCH_LATENCY] == pytest.approx(0.5)
    assert attrs[trace_types.ATTR_JOB_DISPATCH_LATENCY] == pytest.approx(1.0)
    assert attrs[trace_types.ATTR_JOB_SERVER_STARTED_AT] == pytest.approx(t0 + 0.05)
    assert attrs[trace_types.ATTR_JOB_ENTRYPOINT_STARTED_AT] == t0 + 1.0

    ns = 1_000_000_000
    events = {e.name: e.timestamp for e in entry.events}
    assert events["job_received"] == pytest.approx(t0 * ns, abs=1000)
    assert events["job_accepted"] == pytest.approx((t0 + 0.2) * ns, abs=1000)
    assert events["job_assigned"] == pytest.approx((t0 + 0.5) * ns, abs=1000)
    assert events["process_assigned"] == pytest.approx((t0 + 0.6) * ns, abs=1000)
    assert events["entrypoint_started"] == pytest.approx((t0 + 1.0) * ns, abs=1000)
    assert events["job_started_on_server"] == pytest.approx((t0 + 0.05) * ns, abs=1000)
    assert [e.name for e in entry.events] == [
        "job_received",
        "job_accepted",
        "job_assigned",
        "process_assigned",
        "entrypoint_started",
        "job_started_on_server",
    ]


def test_unknown_dispatch_stages_are_skipped(span_exporter: InMemorySpanExporter) -> None:
    # simulation / console / resumed jobs carry no timestamps: nothing is guessed
    with tracer.start_as_current_span("job_entrypoint") as span:
        _record_dispatch_timeline(span, _info(), entrypoint_started_at=1001.0)

    [entry] = _spans(span_exporter, "job_entrypoint")
    attrs = entry.attributes or {}
    assert [e.name for e in entry.events] == ["entrypoint_started"]
    assert trace_types.ATTR_JOB_DISPATCH_LATENCY not in attrs
    assert trace_types.ATTR_JOB_ACCEPT_LATENCY not in attrs
    assert trace_types.ATTR_JOB_SERVER_STARTED_AT not in attrs


def test_server_timestamp_units() -> None:
    assert _server_timestamp_seconds(1_700_000_000_123_456_789) == pytest.approx(1_700_000_000.123)
    assert _server_timestamp_seconds(1_700_000_000_123) == pytest.approx(1_700_000_000.123)
    assert _server_timestamp_seconds(1_700_000_000) == 1_700_000_000.0


def test_callback_name() -> None:
    async def cleanup(reason: str) -> None: ...

    assert _callback_name(cleanup) == "test_callback_name.<locals>.cleanup"
    assert _callback_name(MagicMock(__qualname__="x.y", __name__="y")) == "x.y"


# -- room_connect --


def _mock_room() -> MagicMock:
    room = MagicMock()
    room.connect = AsyncMock()
    room.isconnected.return_value = False
    room.remote_participants = {}
    room.local_participant.sid = "PA_agent"
    room.local_participant.identity = "agent-1"
    return room


async def test_room_connect_span(span_exporter: InMemorySpanExporter) -> None:
    room = _mock_room()
    ctx = JobContext(
        proc=MagicMock(),
        info=_info(),
        room=room,
        on_connect=lambda: None,
        on_shutdown=lambda reason: None,
        inference_executor=MagicMock(),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    room.connect.assert_awaited_once()
    [span] = _spans(span_exporter, "room_connect")
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_ROOM_NAME] == "room-1"
    assert attrs[trace_types.ATTR_ROOM_SID] == "RM_1"
    assert attrs[trace_types.ATTR_ROOM_AUTO_SUBSCRIBE] == "audio_only"
    assert attrs[trace_types.ATTR_ROOM_E2EE] is False
    assert attrs[trace_types.ATTR_PARTICIPANT_ID] == "PA_agent"
    assert attrs[trace_types.ATTR_PARTICIPANT_IDENTITY] == "agent-1"
    assert attrs[trace_types.ATTR_ROOM_REMOTE_PARTICIPANT_COUNT] == 0


async def test_room_connect_failure_is_an_error_span(span_exporter: InMemorySpanExporter) -> None:
    room = _mock_room()
    room.connect = AsyncMock(side_effect=RuntimeError("token expired"))
    ctx = JobContext(
        proc=MagicMock(),
        info=_info(),
        room=room,
        on_connect=lambda: None,
        on_shutdown=lambda reason: None,
        inference_executor=MagicMock(),
    )

    with pytest.raises(RuntimeError):
        await ctx.connect()

    [span] = _spans(span_exporter, "room_connect")
    assert span.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in span.events)


def test_shutdown_callback_wrapper_keeps_the_user_name() -> None:
    ctx = JobContext(
        proc=MagicMock(),
        info=_info(),
        room=_mock_room(),
        on_connect=lambda: None,
        on_shutdown=lambda reason: None,
        inference_executor=MagicMock(),
    )

    async def flush_crm() -> None: ...

    ctx.add_shutdown_callback(flush_crm)
    [wrapped] = ctx._shutdown_callbacks
    assert _callback_name(wrapped).endswith("flush_crm")


# -- session_start / session_close --


async def test_session_lifecycle_spans_and_events(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello there", stt_delay=0.1)
    actions.add_llm("Hi!", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)

    session = create_session(actions, speed_factor=2.0)
    await run_session(session, Agent(instructions="test"), drain_delay=1.0)

    [root] = _spans(span_exporter, "agent_session")
    [start] = _spans(span_exporter, "session_start")
    assert start.parent is not None and start.parent.span_id == root.context.span_id

    # startup work nests under session_start
    [activity_start] = _spans(span_exporter, "start_agent_activity")
    assert activity_start.parent is not None
    assert activity_start.parent.span_id == start.context.span_id
    [toolsets] = _spans(span_exporter, "setup_toolsets")
    assert toolsets.parent is not None
    assert toolsets.parent.span_id == activity_start.context.span_id
    assert start.end_time is not None and activity_start.end_time is not None
    assert start.end_time >= activity_start.end_time

    # the long-lived pipeline is not re-parented: turns stay directly under agent_session
    for name in ("user_turn", "agent_turn"):
        for turn in _spans(span_exporter, name):
            assert turn.parent is not None
            assert turn.parent.span_id == root.context.span_id, name

    # teardown as one bar with the reason, drain nested inside it
    [close] = _spans(span_exporter, "session_close")
    assert close.parent is not None and close.parent.span_id == root.context.span_id
    assert (close.attributes or {})[trace_types.ATTR_CLOSE_REASON] == "user_initiated"
    # run_session drains once before closing (a sibling); the close's own drain nests inside
    drains = _spans(span_exporter, "drain_agent_activity")
    assert any(d.parent is not None and d.parent.span_id == close.context.span_id for d in drains)

    # state timeline on the root span
    agent_states = [e for e in root.events if e.name == "agent_state_changed"]
    transitions = [
        (
            (e.attributes or {})[trace_types.ATTR_OLD_STATE],
            (e.attributes or {})[trace_types.ATTR_NEW_STATE],
        )
        for e in agent_states
    ]
    assert ("initializing", "listening") in transitions
    assert any(new == "speaking" for _, new in transitions)
    user_states = [e for e in root.events if e.name == "user_state_changed"]
    assert any((e.attributes or {})[trace_types.ATTR_NEW_STATE] == "speaking" for e in user_states)
