"""Startup, shutdown, and dispatch telemetry.

Covers the dispatch timeline carried from the worker to the job process (``StartJobRequest``
round trip and the ``job_entrypoint`` events/latencies), the ``room_connect`` span around
``JobContext.connect``, and the session-level spans and events a full fake session produces:
``session_start`` grouping the startup work, ``session_close`` grouping teardown, and state-change
events on ``agent_session``."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext
from livekit.agents.ipc.job_proc_lazy_main import (
    _callback_name,
    _record_dispatch_timeline,
    _server_timestamp_seconds,
)
from livekit.agents.ipc.proto import StartJobRequest
from livekit.agents.job import AutoSubscribe, JobAcceptArguments, RunningJobInfo
from livekit.agents.telemetry import session_context, set_tracer_provider, trace_types, tracer
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
    # sub-millisecond, not approx: a 32-bit float would round all four to the same value
    # (128 s resolution at this magnitude) and every latency would come out as 0
    assert abs(out.received_at - 1_700_000_000.1) < 1e-4
    assert abs(out.accepted_at - 1_700_000_000.2) < 1e-4
    assert abs(out.assigned_at - 1_700_000_000.5) < 1e-4
    assert abs(out.launched_at - 1_700_000_000.6) < 1e-4
    assert out.accepted_at - out.received_at == pytest.approx(0.1, abs=1e-4)


def test_dispatch_timeline_events_and_latencies(span_exporter: InMemorySpanExporter) -> None:
    t0 = 1_700_000_000.0  # realistic unix time: the server timestamp unit detection needs it
    info = _info(received_at=t0, accepted_at=t0 + 0.2, assigned_at=t0 + 0.5, launched_at=t0 + 0.6)
    info.job.state.started_at = int((t0 + 0.05) * 1e9)  # server-side unix nanoseconds

    with tracer.start_as_current_span("job_entrypoint", start_time=int(t0 * 1e9)) as span:
        _record_dispatch_timeline(span, info, entrypoint_started_at=t0 + 1.0)

    [entry] = _spans(span_exporter, "job_entrypoint")
    attrs = entry.attributes or {}
    # adjacent stages, summing to the total
    assert attrs[trace_types.ATTR_JOB_ACCEPT_LATENCY] == pytest.approx(0.2)
    assert attrs[trace_types.ATTR_JOB_ASSIGNMENT_LATENCY] == pytest.approx(0.3)
    assert attrs[trace_types.ATTR_JOB_LAUNCH_LATENCY] == pytest.approx(0.1)
    assert attrs[trace_types.ATTR_JOB_ENTRYPOINT_LATENCY] == pytest.approx(0.4)
    assert attrs[trace_types.ATTR_JOB_DISPATCH_LATENCY] == pytest.approx(1.0)
    # instants are events on the timeline, not raw unix timestamps in the attribute list
    assert not any(k.endswith("_at") for k in attrs)

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
    assert not any(k.startswith("lk.job.") and k.endswith("_latency") for k in attrs)


def test_preload_for_jobs_imports_the_lazy_sdk_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The openai SDK imports its resources tree on first client use; inside a job that was a
    300 ms stall at session start. The warm-up does it before any job exists."""
    import sys

    from livekit.agents.ipc.job_proc_lazy_main import _preload_for_jobs

    for name in [m for m in sys.modules if m.startswith("openai.resources")]:
        monkeypatch.delitem(sys.modules, name)
    assert "openai.resources" not in sys.modules
    _preload_for_jobs()
    assert "openai.resources" in sys.modules
    assert "livekit.local_inference" in sys.modules  # the local end-of-turn model is loaded


def test_framework_callbacks_are_not_user_callbacks() -> None:
    from livekit.agents.ipc.job_proc_lazy_main import _is_framework_callback
    from livekit.agents.utils import aio

    async def log_usage() -> None: ...

    assert not _is_framework_callback(log_usage)
    assert not _is_framework_callback(lambda: None)
    assert _is_framework_callback(aio.cancel_and_wait)  # livekit.agents.utils.aio
    assert _is_framework_callback(AgentSession.aclose)


def test_job_span_is_back_dated_and_carries_the_join_keys(
    span_exporter: InMemorySpanExporter,
) -> None:
    """The job's root span starts at the availability request and is ended by the job
    runner after shutdown; here only its creation is checked."""
    from livekit.agents.ipc.job_proc_lazy_main import _start_job_span

    t0 = 1_700_000_000.0
    info = _info(received_at=t0, accepted_at=t0 + 0.2, assigned_at=t0 + 0.5, launched_at=t0 + 0.6)
    ctx = JobContext(
        proc=MagicMock(),
        info=info,
        room=_mock_room(),
        on_connect=lambda: None,
        on_shutdown=lambda reason: None,
        inference_executor=MagicMock(),
    )
    span = _start_job_span(ctx)
    assert span.is_recording()
    # a child created under it (the session, a connect) nests: the root is a real span
    with tracer.start_as_current_span("agent_session", context=trace.set_span_in_context(span)):
        pass
    span.end()

    [root] = _spans(span_exporter, "job_entrypoint")
    assert root.start_time == pytest.approx(t0 * 1e9, abs=1000)
    attrs = root.attributes or {}
    assert attrs[trace_types.ATTR_JOB_ID] == "AJ_1"
    assert attrs[trace_types.ATTR_DISPATCH_ID] == "AD_1"
    assert attrs[trace_types.ATTR_WORKER_ID] == "W_1"
    assert attrs[trace_types.ATTR_JOB_ACCEPT_LATENCY] == pytest.approx(0.2)
    assert [e.name for e in root.events][:4] == [
        "job_received",
        "job_accepted",
        "job_assigned",
        "process_assigned",
    ]
    [session] = _spans(span_exporter, "agent_session")
    assert session.parent is not None and session.parent.span_id == root.context.span_id


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

    # the entrypoint usually connects before session.start(): the span lands under the
    # job's own span right away, no session needed
    with tracer.start_as_current_span("job_entrypoint") as entrypoint:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    room.connect.assert_awaited_once()

    [span] = _spans(span_exporter, "room_connect")
    assert span.parent is not None
    assert span.parent.span_id == entrypoint.get_span_context().span_id
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
    # the wrapper lives in livekit.agents.job; it must not make the user's callback look
    # like one of the framework's own (those get no shutdown_callback span)
    from livekit.agents.ipc.job_proc_lazy_main import _is_framework_callback

    assert wrapped.__module__ == flush_crm.__module__
    assert not _is_framework_callback(wrapped)


# -- SIP join keys --


async def test_sip_participant_attributes_copied_with_only_the_number_tagged(
    span_exporter: InMemorySpanExporter,
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.0, "Hi", stt_delay=0.1)
    actions.add_llm("Hello", ttft=0.05, duration=0.1)
    actions.add_tts(0.2, ttfb=0.05, duration=0.1)
    session = create_session(actions, speed_factor=4.0)

    sip = MagicMock()
    sip.sid, sip.identity = "PA_sip", "sip_+15550001111"
    sip.kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    sip.attributes = {
        "sip.callID": "SCL_abc",
        "sip.trunkID": "ST_xyz",
        "sip.trunkPhoneNumber": "+15550009999",
        "sip.phoneNumber": "+15550001111",
        "sip.h.x-custom": "route-7",
        "unrelated": "ignored",
    }

    # link the SIP participant once the session is up; run_session owns start/close so the
    # fake pipeline and the transcript synchronizer are torn down cleanly
    linked_once = False

    def _link(ev: object) -> None:
        nonlocal linked_once
        if not linked_once:
            linked_once = True
            session._on_room_io_participant_linked(sip)

    session.on("agent_state_changed", _link)
    await run_session(session, Agent(instructions="test"), drain_delay=0.5)

    [root] = _spans(span_exporter, "agent_session")
    attrs = root.attributes or {}
    # the customer's own identifiers stay plain
    assert attrs["lk.sip.callID"] == "SCL_abc"
    assert attrs["lk.sip.trunkID"] == "ST_xyz"
    assert attrs["lk.sip.trunkPhoneNumber"] == "+15550009999"
    assert attrs["lk.sip.h.x-custom"] == "route-7"
    # the end user's number is the one PII value
    assert attrs[trace_types.ATTR_SIP_PHONE_NUMBER] == "+15550001111"
    assert "lk.sip.phoneNumber" not in attrs
    assert "lk.sip.unrelated" not in attrs
    [linked] = [e for e in root.events if e.name == "participant_linked"]
    assert (linked.attributes or {})[trace_types.ATTR_PARTICIPANT_KIND] == "PARTICIPANT_KIND_SIP"


# -- startup spans are never current --


def _live_session_job(root: trace.Span, start: trace.Span | None) -> MagicMock:
    """A job whose primary session is running: root span set, startup span while starting."""
    session = MagicMock()
    session._root_span_context = trace.set_span_in_context(root)
    session._session_start_context = trace.set_span_in_context(start) if start else None
    job = MagicMock()
    job._primary_agent_session = session
    return job


async def test_detached_span_is_not_current_and_takes_its_parent(
    span_exporter: InMemorySpanExporter,
) -> None:
    with tracer.start_as_current_span("agent_session") as root:
        parent = tracer.start_span("session_start")
        with tracer.detached_span(
            "publish_audio_output", context=trace.set_span_in_context(parent)
        ) as span:
            # a task spawned here inherits the ambient context, not the detached span
            assert trace.get_current_span() is root
            inherited = await asyncio.create_task(_current_span_name())
            assert inherited == "agent_session"
        parent.end()
    [published] = _spans(span_exporter, "publish_audio_output")
    assert published.parent is not None
    assert published.parent.span_id == parent.get_span_context().span_id
    assert span.end_time is not None  # type: ignore[attr-defined]


async def _current_span_name() -> str:
    return getattr(trace.get_current_span(), "name", "<none>")


async def test_detached_span_records_the_exception(span_exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError), tracer.detached_span("wait_for_participant"):
        raise RuntimeError("no one came")
    [span] = _spans(span_exporter, "wait_for_participant")
    assert span.status.status_code.name == "ERROR"


async def test_room_connect_during_start_nests_under_session_start_without_leaking(
    span_exporter: InMemorySpanExporter,
) -> None:
    """The room's event tasks are created inside room.connect(); whatever is current there
    becomes the parent of every span those tasks emit for the rest of the session."""
    with tracer.start_as_current_span("agent_session") as root:
        start = tracer.start_span("session_start")
        job = _live_session_job(root, start)
        with session_context.session_span("room_connect", job_ctx=job):
            assert trace.get_current_span() is root
            assert await asyncio.create_task(_current_span_name()) == "agent_session"
        start.end()
        # after startup the same call parents to the ambient span
        with session_context.session_span("room_connect", job_ctx=_live_session_job(root, None)):
            assert trace.get_current_span() is root

    connects = _spans(span_exporter, "room_connect")
    assert len(connects) == 2
    assert connects[0].parent is not None
    assert connects[0].parent.span_id == start.get_span_context().span_id
    assert connects[1].parent is not None
    assert connects[1].parent.span_id == root.get_span_context().span_id


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
    # (user_turn is pinned to the root explicitly: a late STT final during session_close
    # used to nest it under the close span)
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
