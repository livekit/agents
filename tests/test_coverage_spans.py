"""Coverage additions on existing spans: interruption detail on ``agent_turn``, the
``update_agent`` handoff span, and fallback-adapter events on the request span."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, APIConnectionError, llm
from livekit.agents.llm import ChatContext, FallbackAdapter, LLMStream, Tool
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_io import FakeAudioInput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_session import FakeActions, create_session, run_session
from .fake_stt import FakeSTT

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


def _events(span: ReadableSpan, name: str) -> list[Any]:
    return [e for e in span.events if e.name == name]


# -- interruption detail --


async def test_barge_in_records_source_and_playout_position(
    span_exporter: InMemorySpanExporter,
) -> None:
    speed = 2.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at ~3.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # barge-in at ~5.5s

    session = create_session(actions, speed_factor=speed)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="test"), drain_delay=1.0), timeout=60
    )

    interrupted = [
        s
        for s in _spans(span_exporter, "agent_turn")
        if (s.attributes or {}).get(trace_types.ATTR_SPEECH_INTERRUPTED) is True
    ]
    assert len(interrupted) == 1
    turn = interrupted[0]
    [event] = _events(turn, "interrupted")
    assert (event.attributes or {})[trace_types.ATTR_INTERRUPTION_SOURCE] == "audio_activity"
    position = (turn.attributes or {})[trace_types.ATTR_PLAYOUT_POSITION]
    assert isinstance(position, float)
    # ~2 s of the 10 s story had played (5.5 - 3.5), scaled by the speed factor
    assert 0.5 < position < 10.0 / speed

    # the reply to "Stop!" was not interrupted and carries no event
    for turn in _spans(span_exporter, "agent_turn"):
        if turn is not interrupted[0]:
            assert _events(turn, "interrupted") == []


# -- agent handoff --


class _FirstAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="first")


class _SecondAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="second")


async def test_update_agent_span_groups_the_handoff(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello", stt_delay=0.1)
    actions.add_llm("Hi there", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)

    session = create_session(actions, speed_factor=2.0)
    audio_input = session.input.audio
    assert isinstance(audio_input, FakeAudioInput)
    stt = session.stt
    assert isinstance(stt, FakeSTT)
    # the synced audio output owns a transcript synchronizer that run_session normally closes;
    # grab it now, the session detaches its outputs on close
    synchronizer = (
        session.output.audio._synchronizer
        if isinstance(session.output.audio, _SyncedAudioOutput)
        else None
    )

    await session.start(_FirstAgent())
    audio_input.push(0.1)
    await stt.fake_user_speeches_done
    await asyncio.sleep(1.0)

    session.update_agent(_SecondAgent())
    await asyncio.sleep(1.0)

    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()
    if synchronizer is not None:
        await synchronizer.aclose()

    [root] = _spans(span_exporter, "agent_session")
    [handoff] = _spans(span_exporter, "update_agent")
    assert handoff.parent is not None and handoff.parent.span_id == root.context.span_id
    attrs = handoff.attributes or {}
    assert attrs[trace_types.ATTR_PREVIOUS_AGENT_LABEL] == "_first_agent"
    assert attrs[trace_types.ATTR_AGENT_LABEL] == "_second_agent"

    def _children(name: str) -> list[ReadableSpan]:
        return [
            s
            for s in _spans(span_exporter, name)
            if s.parent is not None and s.parent.span_id == handoff.context.span_id
        ]

    # the old agent's drain (with on_exit inside it), then the new agent's start, under the handoff
    [drain] = _children("drain_agent_activity")
    exits = [
        s
        for s in _spans(span_exporter, "on_exit")
        if s.parent is not None and s.parent.span_id == drain.context.span_id
    ]
    assert len(exits) == 1
    [start] = _children("start_agent_activity")
    assert (start.attributes or {})[trace_types.ATTR_AGENT_LABEL] == "_second_agent"

    # the initial start is not a handoff: it lives under session_start, not update_agent
    [session_start] = _spans(span_exporter, "session_start")
    initial = [
        s
        for s in _spans(span_exporter, "start_agent_activity")
        if s.parent is not None and s.parent.span_id == session_start.context.span_id
    ]
    assert len(initial) == 1


# -- fallback adapter events --


class _FailingLLMStream(LLMStream):
    async def _run(self) -> None:
        raise APIConnectionError("primary down")


class _FailingLLM(FakeLLM):
    @property
    def model(self) -> str:
        return "broken-model"

    @property
    def provider(self) -> str:
        return "broken"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        return _FailingLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


async def test_llm_fallback_records_failed_and_serving_provider(
    span_exporter: InMemorySpanExporter,
) -> None:
    primary = _FailingLLM()
    secondary = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hi", content="hello", ttft=0.01, duration=0.02)]
    )
    adapter = FallbackAdapter([primary, secondary], attempt_timeout=1.0, max_retry_per_llm=0)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    try:
        stream = adapter.chat(chat_ctx=chat_ctx)
        chunks = [chunk async for chunk in stream]
        await stream.aclose()
    finally:
        await adapter.aclose()
        await primary.aclose()
        await secondary.aclose()

    assert "".join(c.delta.content or "" for c in chunks if c.delta) == "hello"

    # the adapter's request span nests the attempt span that ran the fallback loop
    [request] = _spans(span_exporter, "llm_fallback_adapter")
    runs = [
        s
        for s in _spans(span_exporter, "llm_request_run")
        if s.parent is not None and s.parent.span_id == request.context.span_id
    ]
    assert runs, "no llm_request_run under the fallback request"
    run = runs[-1]
    [failed] = _events(run, "fallback_provider_failed")
    assert (failed.attributes or {})[trace_types.ATTR_FALLBACK_LABEL] == primary.label
    assert (failed.attributes or {})[trace_types.ATTR_FALLBACK_INDEX] == 0
    assert (run.attributes or {})[trace_types.ATTR_FALLBACK_LABEL] == secondary.label
    assert (run.attributes or {})[trace_types.ATTR_FALLBACK_INDEX] == 1


def test_interrupt_source_first_wins() -> None:
    from livekit.agents.voice.speech_handle import SpeechHandle

    handle = SpeechHandle.__new__(SpeechHandle)
    handle._interrupt_source = None
    handle._set_interrupt_source("audio_activity")
    handle._set_interrupt_source("user_turn")
    assert handle._interrupt_source == "audio_activity"
    # unrelated: keep the module import used
    assert llm is not None and time is not None
