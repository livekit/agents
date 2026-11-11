"""Span shape and attributes: agent turns, end-of-turn waits, user speech, handoffs, adapters."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from livekit.agents import (
    Agent,
    AgentSession,
    APIConnectionError,
    RunContext,
    function_tool,
    llm,
    stt,
    vad,
)
from livekit.agents.llm import ChatContext, FallbackAdapter, FunctionToolCall, LLMStream, Tool
from livekit.agents.telemetry import (
    otel_metrics,
    set_tracer_provider,
    trace_types,
    tracer,
    utils as telemetry_utils,
)
from livekit.agents.tts import FallbackAdapter as TTSFallbackAdapter, StreamAdapter
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice import generation, keyterm_detection
from livekit.agents.voice.agent_activity import (
    AgentActivity,
    _agent_turn,
    _continue_discarded_turn,
)
from livekit.agents.voice.agent_session import _RECORDING_ALL_OFF
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.endpointing import BaseEndpointing
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.io import ModelSettings
from livekit.agents.voice.keyterm_detection import KeytermDetector
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput
from livekit.agents.voice.turn import (
    TurnDetectionEvent,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)

from .fake_io import FakeAudioInput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_session import FakeActions, create_session, run_session
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .trace_schema import assert_trace_well_formed

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


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


def _children(
    exporter: InMemorySpanExporter, parent: ReadableSpan, name: str
) -> list[ReadableSpan]:
    return [
        s
        for s in _spans(exporter, name)
        if s.parent is not None and s.parent.span_id == parent.context.span_id
    ]


def _events(span: ReadableSpan, name: str) -> list[Any]:
    return [e for e in span.events if e.name == name]


# -- agent_turn --


class _WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    @function_tool
    async def get_weather(self, context: RunContext, location: str) -> str:
        return f"sunny in {location}"


async def test_tool_call_is_one_agent_turn(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "What's the weather in Tokyo?")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_llm(content="It is sunny in Tokyo.", input="sunny in Tokyo")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(run_session(session, _WeatherAgent(), drain_delay=1.0), timeout=60)

    [root] = _spans(span_exporter, "agent_session")
    turns = _spans(span_exporter, "agent_turn")
    assert len(turns) == 1, [(t.attributes or {}).get(trace_types.ATTR_SPEECH_ID) for t in turns]
    [turn] = turns
    assert turn.parent is not None and turn.parent.span_id == root.context.span_id

    attrs = turn.attributes or {}
    speech_id = attrs[trace_types.ATTR_SPEECH_ID]
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 2
    assert attrs[trace_types.ATTR_AGENT_TURN_ID] == f"{speech_id}_2"
    generations = [e for e in turn.events if e.name == "generation"]
    assert [(e.attributes or {})[trace_types.ATTR_AGENT_TURN_ID] for e in generations] == [
        f"{speech_id}_1",
        f"{speech_id}_2",
    ]
    assert trace_types.ATTR_AGENT_PARENT_TURN_ID not in (generations[0].attributes or {})
    assert (generations[1].attributes or {})[trace_types.ATTR_AGENT_PARENT_TURN_ID] == (
        f"{speech_id}_1"
    )

    # both generations' inference, the tool between them, and the speech all nest under it
    assert len(_children(span_exporter, turn, "llm_node")) == 2
    [tool] = _children(span_exporter, turn, "function_tool")
    [tts] = _children(span_exporter, turn, "tts_node")
    [speaking] = _children(span_exporter, turn, "agent_speaking")
    assert tool.start_time < tts.start_time
    # and the turn covers everything, ending with the speech rather than with the first step
    for child in (tool, tts, speaking):
        assert child.end_time is not None and turn.end_time is not None
        assert turn.start_time <= child.start_time and child.end_time <= turn.end_time
    assert_trace_well_formed(span_exporter.get_finished_spans())


async def test_plain_reply_is_one_generation(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello", stt_delay=0.1)
    actions.add_llm("Hi there", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="test"), drain_delay=1.0), timeout=60
    )

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 1
    assert attrs[trace_types.ATTR_AGENT_TURN_ID] == f"{attrs[trace_types.ATTR_SPEECH_ID]}_1"
    assert len([e for e in turn.events if e.name == "generation"]) == 1
    assert trace_types.ATTR_AGENT_PARENT_TURN_ID not in attrs
    assert_trace_well_formed(span_exporter.get_finished_spans())


def test_discarded_preemptive_generation_hands_its_turn_to_the_successor(
    span_exporter: InMemorySpanExporter,
) -> None:
    """The successor continues a discarded preemptive attempt's agent_turn instead of opening a second one."""
    root = tracer.start_span("agent_session")
    root_ctx = trace.set_span_in_context(root)
    attempt = SpeechHandle.create(allow_interruptions=True)
    with _agent_turn(attempt, root_context=root_ctx, agent_label="a"):
        pass  # the attempt's first generation ran here

    reply = SpeechHandle.create(allow_interruptions=True)
    _continue_discarded_turn(attempt, reply)
    attempt._mark_done()  # the cancelled attempt finishes: the span must survive it
    assert _spans(span_exporter, "agent_turn") == []

    with _agent_turn(reply, root_context=root_ctx, agent_label="a"):
        pass
    reply._mark_done()
    root.end()

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_SPEECH_ID] == reply.id
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 1  # the reply's own step count
    events = [e.name for e in turn.events]
    assert events == ["generation", "preemptive_generation_discarded", "generation"]
    [discarded] = [e for e in turn.events if e.name == "preemptive_generation_discarded"]
    assert (discarded.attributes or {})[trace_types.ATTR_SPEECH_ID] == attempt.id

    # nothing to hand over: a plain successor is untouched
    _continue_discarded_turn(None, reply)
    _continue_discarded_turn(reply, reply)


def test_llm_failure_stored_on_the_handle_fails_the_turn(
    span_exporter: InMemorySpanExporter,
) -> None:
    """An LLM failure stored on the handle, not raised, still fails the turn."""
    handle = SpeechHandle.create(allow_interruptions=True)
    with _agent_turn(handle, root_context=None, agent_label="a"):
        handle._error = RuntimeError("llm down")
    handle._mark_done()

    [turn] = _spans(span_exporter, "agent_turn")
    assert turn.status.status_code == StatusCode.ERROR
    assert [e.name for e in turn.events if e.name == "exception"] == ["exception"]


def test_turn_duration_metric_is_recorded_when_the_span_is_sampled_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = MagicMock()
    monkeypatch.setattr(otel_metrics, "record_invoke_agent_duration", record)
    handle = SpeechHandle.create(allow_interruptions=True)
    handle._agent_turn_span = trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
    handle._agent_turn_started_at = 1.0
    handle._agent_turn_agent_name = "a"
    handle._mark_done()
    record.assert_called_once()
    assert record.call_args.kwargs == {"agent_name": "a"}


def test_sampled_out_turn_is_still_handed_to_the_successor() -> None:
    """The successor adopts a non-recording turn too, keeping the attempt's start time."""
    attempt = SpeechHandle.create(allow_interruptions=True)
    attempt._agent_turn_span = trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
    attempt._agent_turn_started_at = 1.0
    attempt._agent_turn_agent_name = "a"

    reply = SpeechHandle.create(allow_interruptions=True)
    _continue_discarded_turn(attempt, reply)
    assert attempt._agent_turn_span is None
    assert reply._agent_turn_span is not None
    assert reply._agent_turn_started_at == 1.0
    assert reply._agent_turn_agent_name == "a"


# -- eou_wait --


def _make_recognition(*, min_delay: float, with_detector: bool = False) -> AudioRecognition:
    """Enough of ``AudioRecognition`` to drive ``_run_eou_detection``; commits by default."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._session._room_io = None  # keep participant attributes off the user_turn span
    ar._session.amd = None
    ar._session.options.transcription_timeout = None
    ar._hooks = MagicMock()
    ar._hooks.on_end_of_turn.return_value = True
    ar._hooks.retrieve_chat_ctx.return_value = llm.ChatContext()
    ar._stt = None
    ar._stt_pipeline = None
    ar._stt_model = None
    ar._stt_provider = None
    ar._vad = None
    ar._turn_detection_mode = "vad"
    ar._turn_detector = None
    ar._turn_detector_stream = None
    ar._turn_detector_prediction_fut = None
    ar._turn_detector_flushed = False
    ar._turn_detector_late_prediction_warned = False
    ar._vad_base_turn_detection = False
    ar._agent_speaking = False
    ar._interruption_enabled = False
    ar._interruption_ch = None
    ar._turn_backchannel_over_agent = False
    ar._overlap_in_current_turn = False
    ar._active_vad_speech_started_at = None
    ar._transcription_timeout_handle = None
    ar._turn_speech_duration = 0.0
    ar._turn_transcript_received = False
    ar._audio_transcript = "hello there"
    ar._final_transcript_confidence = []
    ar._stt_request_ids = []
    ar._stt_events = []
    ar._last_speaking_time = None
    ar._last_final_transcript_time = None
    ar._speech_start_time = None
    ar._vad_speech_started = False
    ar._user_silence_ev = asyncio.Event()  # backs the _speaking setter
    ar._speaking = False
    ar._end_of_turn_task = None
    ar._user_turn_committed = False
    ar._last_language = None
    ar._last_emitted_prediction = None
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._eou_wait_span = None
    ar._eou_wait_started_at_ns = None
    ar._eou_wait_rearms = 0
    ar._eou_wait_floor_ns = None
    ar._eou_wait_not_committed = 0
    ar._user_turn_resumes = 0
    ar._eou_detection_span = None
    ar._closing = asyncio.Event()

    endpointing = MagicMock()
    endpointing.min_delay = min_delay
    endpointing.max_delay = max(min_delay, 1.0)
    ar._endpointing = endpointing

    if with_detector:
        ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)
        stream_mock = MagicMock(spec=_StreamingTurnDetectorStream)
        stream_mock.supports_language = AsyncMock(return_value=True)
        stream_mock.unlikely_threshold = AsyncMock(return_value=0.5)
        stream_mock.backchannel_threshold = AsyncMock(return_value=None)
        stream_mock.predict = MagicMock(side_effect=asyncio.Future)
        stream_mock.flush = MagicMock()
        stream_mock.cancel_inference = MagicMock()
        stream_mock.prediction_timeout = 0.01
        ar._turn_detector_stream = stream_mock
        event = TurnDetectionEvent(
            type="eot_prediction",
            last_speaking_time=time.time(),
            end_of_turn_probability=0.9,
            inference_duration=0.01,
            detection_delay=None,
            backchannel_probability=None,
        )
        fut: asyncio.Future[TurnDetectionEvent] = asyncio.Future()
        fut.set_result(event)
        ar._turn_detector_prediction_fut = fut
    return ar


def _start_of_speech() -> vad.VADEvent:
    return vad.VADEvent(
        type=vad.VADEventType.START_OF_SPEECH,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.5,
        silence_duration=0.0,
    )


async def _await_bounce(ar: AudioRecognition) -> None:
    task = ar._end_of_turn_task
    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _cancel_bounce(ar: AudioRecognition) -> None:
    task = ar._end_of_turn_task
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_wait_span_covers_last_speech_to_commit(span_exporter: InMemorySpanExporter) -> None:
    ar = _make_recognition(min_delay=0.05)
    # the user stopped talking 0.3 s ago: the wait is back-dated there and commits at once
    last_speaking = time.time() - 0.3
    ar._last_speaking_time = last_speaking

    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await _await_bounce(ar)

    [wait] = _spans(span_exporter, "eou_wait")
    [user_turn] = _spans(span_exporter, "user_turn")
    assert wait.parent is not None
    assert wait.parent.span_id == user_turn.context.span_id

    assert wait.start_time == int(last_speaking * 1_000_000_000)
    # no VAD opened this turn: created here, it starts at the last speaking time too, so the
    # back-dated wait never starts before its parent
    assert user_turn.start_time <= wait.start_time
    attrs = wait.attributes or {}
    assert attrs[trace_types.ATTR_EOU_OUTCOME] == "committed"
    assert attrs[trace_types.ATTR_EOU_SOURCE] == "vad"
    assert attrs[trace_types.ATTR_EOU_DELAY] == 0.05
    assert attrs[trace_types.ATTR_EOU_REARM_COUNT] == 0
    wait_duration = attrs[trace_types.ATTR_EOU_WAIT_DURATION]
    assert isinstance(wait_duration, float)
    assert 0.3 <= wait_duration < 0.6
    assert wait.end_time is not None
    # the attribute is derived from the span's own start/end nanoseconds: identical, not close
    assert (wait.end_time - wait.start_time) / 1e9 == wait_duration
    # the wait closes before the turn does
    assert user_turn.end_time is not None and wait.end_time <= user_turn.end_time
    assert ar._eou_wait_span is None


async def test_later_trigger_rearms_the_same_span(span_exporter: InMemorySpanExporter) -> None:
    # the wait is long relative to the re-trigger so a slow CI runner cannot let it commit first
    ar = _make_recognition(min_delay=1.0)
    ar._last_speaking_time = time.time()

    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await asyncio.sleep(0.05)
    # a late STT final re-triggers end of turn; the bounce task restarts, the span must not
    ar._run_eou_detection(llm.ChatContext(), trigger="stt")
    await _await_bounce(ar)

    [wait] = _spans(span_exporter, "eou_wait")
    attrs = wait.attributes or {}
    assert attrs[trace_types.ATTR_EOU_OUTCOME] == "committed"
    assert attrs[trace_types.ATTR_EOU_REARM_COUNT] == 1
    assert attrs[trace_types.ATTR_EOU_SOURCE] == "stt"
    [rearmed] = _events(wait, "rearmed")
    assert (rearmed.attributes or {})[trace_types.ATTR_EOU_SOURCE] == "stt"


async def test_resumed_speech_ends_wait_at_speech_start(
    span_exporter: InMemorySpanExporter,
) -> None:
    ar = _make_recognition(min_delay=1.0)
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await asyncio.sleep(0.05)

    before = time.time()
    await ar._on_vad_event(_start_of_speech())  # speech_duration=0.5 -> started 0.5 s ago
    after = time.time()
    await _cancel_bounce(ar)

    [wait] = _spans(span_exporter, "eou_wait")
    attrs = wait.attributes or {}
    assert attrs[trace_types.ATTR_EOU_OUTCOME] == "user_resumed"
    # ended where the resumed speech started, per VAD, clamped to the wait's own start
    assert wait.end_time is not None
    speech_started_between = (int((before - 0.5) * 1e9), int((after - 0.5) * 1e9))
    assert wait.start_time <= wait.end_time
    assert wait.end_time <= max(speech_started_between[1], wait.start_time)
    # the user turn itself stays open: they are still talking
    assert _spans(span_exporter, "user_turn") == []
    assert ar._user_turn_span is not None and ar._user_turn_span.is_recording()
    ar._end_user_turn_span()
    # the turn counts the waits the user cut short
    [user_turn] = _spans(span_exporter, "user_turn")
    assert (user_turn.attributes or {})[trace_types.ATTR_EOU_RESUME_COUNT] == 1


async def test_user_turn_is_pinned_to_the_session_root(span_exporter: InMemorySpanExporter) -> None:
    """A turn committed while session_close is current still sits under agent_session."""
    ar = _make_recognition(min_delay=0.01)
    root = tracer.start_span("agent_session")
    ar._session._root_span_context = trace.set_span_in_context(root)
    with tracer.start_as_current_span("session_close") as close:
        ar._last_speaking_time = time.time()
        ar._run_eou_detection(llm.ChatContext(), trigger="stt")
        await _await_bounce(ar)
    ar._end_user_turn_span()
    root.end()

    [turn] = _spans(span_exporter, "user_turn")
    assert turn.parent is not None
    assert turn.parent.span_id == root.get_span_context().span_id
    assert turn.parent.span_id != close.get_span_context().span_id
    [wait] = _spans(span_exporter, "eou_wait")
    assert wait.parent is not None and wait.parent.span_id == turn.context.span_id


async def test_teardown_drops_an_open_wait(span_exporter: InMemorySpanExporter) -> None:
    ar = _make_recognition(min_delay=1.0)
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await asyncio.sleep(0.02)

    ar._end_user_turn_span()
    await _cancel_bounce(ar)

    [wait] = _spans(span_exporter, "eou_wait")
    assert (wait.attributes or {})[trace_types.ATTR_EOU_OUTCOME] == "dropped"
    [user_turn] = _spans(span_exporter, "user_turn")
    assert user_turn.end_time is not None and wait.end_time is not None
    assert wait.end_time <= user_turn.end_time


async def test_not_committed_turn_keeps_waiting(span_exporter: InMemorySpanExporter) -> None:
    ar = _make_recognition(min_delay=0.01)
    ar._hooks.on_end_of_turn.return_value = False  # e.g. below interruption min_words
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await _await_bounce(ar)

    # the decision is deferred: the span records the rejection and stays open
    assert _spans(span_exporter, "eou_wait") == []
    assert ar._eou_wait_span is not None and ar._eou_wait_span.is_recording()
    ar._end_user_turn_span()
    [wait] = _spans(span_exporter, "eou_wait")
    attrs = wait.attributes or {}
    assert attrs[trace_types.ATTR_EOU_NOT_COMMITTED_COUNT] == 1
    assert attrs[trace_types.ATTR_EOU_OUTCOME] == "dropped"
    assert wait.events == ()


async def test_detection_nests_under_wait(span_exporter: InMemorySpanExporter) -> None:
    ar = _make_recognition(min_delay=0.01, with_detector=True)
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await _await_bounce(ar)

    [detection] = _spans(span_exporter, "eou_detection")
    [wait] = _spans(span_exporter, "eou_wait")
    assert detection.parent is not None
    assert detection.parent.span_id == wait.context.span_id
    assert (detection.attributes or {})[trace_types.ATTR_EOU_PROBABILITY] == 0.9
    # the delay decided by the prediction is stamped on the wait too
    assert (wait.attributes or {})[trace_types.ATTR_EOU_DELAY] == 0.01


async def test_resumed_speech_during_detection_keeps_the_child_inside(
    span_exporter: InMemorySpanExporter,
) -> None:
    """A resume reported after the detector started cannot end the wait before its child."""
    ar = _make_recognition(min_delay=1.0, with_detector=True)
    pending: asyncio.Future[TurnDetectionEvent] = asyncio.Future()
    ar._turn_detector_prediction_fut = pending  # inference never answers
    ar._turn_detector_stream.prediction_timeout = 10.0  # type: ignore[union-attr]
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await asyncio.sleep(0.05)
    assert ar._eou_detection_span is not None and ar._eou_detection_span.is_recording()

    await ar._on_vad_event(_start_of_speech())  # started 0.5 s ago, before the detection began
    await _cancel_bounce(ar)

    [wait] = _spans(span_exporter, "eou_wait")
    [detection] = _spans(span_exporter, "eou_detection")
    assert detection.parent is not None and detection.parent.span_id == wait.context.span_id
    assert wait.start_time <= detection.start_time
    assert detection.end_time is not None and wait.end_time is not None
    assert detection.end_time <= wait.end_time
    assert [e.name for e in detection.events] == ["superseded"]
    attrs = wait.attributes or {}
    assert attrs[trace_types.ATTR_EOU_OUTCOME] == "user_resumed"
    # the delay in force is stamped when the wait opens, so a wait ended by resumed speech
    # before the detector answered still carries it
    assert trace_types.ATTR_EOU_DELAY in attrs
    assert attrs[trace_types.ATTR_EOU_WAIT_DURATION] == pytest.approx(
        (wait.end_time - wait.start_time) / 1e9
    )
    # the real resume time survives as an event inside the bar
    [resumed] = _events(wait, "user_resumed")
    assert wait.start_time <= resumed.timestamp <= wait.end_time
    ar._end_user_turn_span()


async def test_turn_mode_change_drops_an_open_wait(span_exporter: InMemorySpanExporter) -> None:
    ar = _make_recognition(min_delay=1.0)
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await asyncio.sleep(0.02)
    assert ar._eou_wait_span is not None

    ar._update_options(turn_detection="manual")
    await asyncio.sleep(0)  # let the cancelled bounce task unwind

    [wait] = _spans(span_exporter, "eou_wait")
    assert (wait.attributes or {})[trace_types.ATTR_EOU_OUTCOME] == "dropped"
    assert ar._eou_wait_span is None and ar._end_of_turn_task is None
    # the user turn is still open for whatever the manual mode does with the next speech
    assert ar._user_turn_span is not None and ar._user_turn_span.is_recording()
    ar._end_user_turn_span()


class _HookAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        await asyncio.sleep(0.05)


async def test_full_session_turn_handoff_spans(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello, how are you?", stt_delay=0.1)
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(run_session(session, _HookAgent(), drain_delay=1.0), timeout=30)

    [wait] = _spans(span_exporter, "eou_wait")
    [user_turn] = _spans(span_exporter, "user_turn")
    assert wait.parent is not None and wait.parent.span_id == user_turn.context.span_id
    assert (wait.attributes or {})[trace_types.ATTR_EOU_OUTCOME] == "committed"

    [hook] = _spans(span_exporter, "on_user_turn_completed")
    assert hook.end_time is not None
    assert (hook.end_time - hook.start_time) / 1e9 == pytest.approx(0.05, abs=1e-3)
    assert (hook.attributes or {})[trace_types.ATTR_AGENT_LABEL] == "_hook_agent"
    # the hook is part of the turn: it nests under user_turn, which ends after it (a
    # preemptive generation may already have started its agent_turn by then; that is fine)
    assert hook.parent is not None and hook.parent.span_id == user_turn.context.span_id
    assert user_turn.end_time is not None and user_turn.end_time >= hook.end_time

    # the reply's agent_turn records how long it sat in the speech queue
    turns = [
        s
        for s in _spans(span_exporter, "agent_turn")
        if trace_types.ATTR_SPEECH_QUEUE_WAIT in (s.attributes or {})
    ]
    assert turns, "no agent_turn carries the queue wait"
    for turn in turns:
        queue_wait = (turn.attributes or {})[trace_types.ATTR_SPEECH_QUEUE_WAIT]
        assert isinstance(queue_wait, float) and 0.0 <= queue_wait < 5.0
    assert_trace_well_formed(span_exporter.get_finished_spans())

    # the reply that answered the turn carries the user-side stages next to lk.e2e_latency,
    # so the per-turn breakdown reads off one span
    [reply] = [
        s
        for s in _spans(span_exporter, "agent_turn")
        if trace_types.ATTR_E2E_LATENCY in (s.attributes or {})
    ]
    attrs = reply.attributes or {}
    for key in (
        trace_types.ATTR_END_OF_TURN_DELAY,
        trace_types.ATTR_TRANSCRIPTION_DELAY,
        trace_types.ATTR_ON_USER_TURN_COMPLETED_DELAY,
    ):
        assert isinstance(attrs[key], float), key
    assert attrs[trace_types.ATTR_ON_USER_TURN_COMPLETED_DELAY] == pytest.approx(0.05, abs=1e-3)


class _RaisingHookAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        raise RuntimeError(f"lookup failed for {new_message.text_content}")


async def test_hook_exception_honours_session_redaction(
    span_exporter: InMemorySpanExporter,
) -> None:
    """Session-level redaction alone keeps the hook's exception message out of the trace."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "My name is Jane Doe", stt_delay=0.1)
    actions.add_llm("Sorry about that.", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(
        run_session(
            session,
            _RaisingHookAgent(),
            drain_delay=1.0,
            record={**_RECORDING_ALL_OFF, "redaction": True},  # type: ignore[typeddict-item]
        ),
        timeout=30,
    )
    assert session._redaction_enabled

    [hook] = _spans(span_exporter, "on_user_turn_completed")
    assert hook.status.status_code.name == "ERROR"
    attrs = hook.attributes or {}
    assert attrs[trace_types.ATTR_EXCEPTION_TYPE] == "RuntimeError"
    assert attrs[trace_types.ATTR_EXCEPTION_MESSAGE] == telemetry_utils.REDACTED_EXCEPTION_MESSAGE
    rendered = repr(hook.attributes) + repr(
        [(e.name, dict(e.attributes or {})) for e in hook.events]
    )
    assert "Jane Doe" not in rendered and "lookup failed" not in rendered
    assert_trace_well_formed(span_exporter.get_finished_spans())


# -- user_speaking --


def _user_speaking(exporter: InMemorySpanExporter) -> ReadableSpan:
    [span] = [s for s in exporter.get_finished_spans() if s.name == "user_speaking"]
    return span


async def test_backdated_end_is_clamped_to_the_start(span_exporter: InMemorySpanExporter) -> None:
    session = AgentSession()
    started_at = time.time()
    session._update_user_state("speaking", last_speaking_time=started_at)
    # a VAD end-of-speech anchor, backdated past the STT start by the silence it waited on
    session._update_user_state("listening", last_speaking_time=started_at - 0.55)

    span = _user_speaking(span_exporter)
    assert span.end_time == span.start_time


async def test_end_after_the_start_is_kept(span_exporter: InMemorySpanExporter) -> None:
    session = AgentSession()
    started_at = time.time()
    session._update_user_state("speaking", last_speaking_time=started_at)
    session._update_user_state("listening", last_speaking_time=started_at + 1.25)

    span = _user_speaking(span_exporter)
    assert span.end_time - span.start_time == pytest.approx(1.25e9, rel=1e-6)


# -- stt events on user_turn --


@pytest.fixture
async def recognition(span_exporter: InMemorySpanExporter) -> AsyncIterator[AudioRecognition]:
    session = AgentSession(vad=None, turn_handling={"turn_detection": "manual"})
    root = tracer.start_span("agent_session")
    session._root_span_context = trace.set_span_in_context(root)
    hooks = MagicMock()
    hooks.retrieve_chat_ctx.return_value = llm.ChatContext()
    hooks.on_end_of_turn.return_value = True
    recognition = AudioRecognition(
        session,
        hooks=hooks,
        endpointing=BaseEndpointing(min_delay=0, max_delay=0),
        stt=None,
        vad=None,
        interruption_detection=None,
        turn_detection="manual",
    )
    try:
        yield recognition
    finally:
        await recognition._aclose()
        root.end()


def _turn_events(exporter: InMemorySpanExporter) -> list[list[dict]]:
    return [
        json.loads(span.attributes[trace_types.ATTR_STT_EVENTS])
        for span in exporter.get_finished_spans()
        if span.name == "user_turn"
    ]


async def _commit(recognition: AudioRecognition) -> None:
    recognition._run_eou_detection(llm.ChatContext(), trigger="manual")
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task


async def test_transcript_arrivals_survive_buffering_and_reset_between_turns(
    recognition: AudioRecognition, span_exporter: InMemorySpanExporter
) -> None:
    origin = time.time() - 2
    recognition._transcript_gate_active = True
    await recognition._on_stt_event(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
    expected = []
    for i, (kind, text) in enumerate(
        [
            (stt.SpeechEventType.INTERIM_TRANSCRIPT, "你好🙂"),
            (stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, "hello world"),
            (stt.SpeechEventType.FINAL_TRANSCRIPT, "hello world"),
            (stt.SpeechEventType.FINAL_TRANSCRIPT, ""),
        ]
    ):
        received_at = origin + i * 0.1
        await recognition._on_stt_event(
            stt.SpeechEvent(
                type=kind,
                created_at=received_at,
                alternatives=[
                    stt.SpeechData(language="en", text=text),
                    stt.SpeechData(language="en", text="unused alternative"),
                ],
            )
        )
        expected.append(
            {"received_at": received_at, "type": kind.value, "transcript_length": len(text)}
        )

    assert recognition._audio_transcript == ""
    assert recognition._user_turn_span is None
    recognition._ensure_user_turn_span(start_time=origin - 0.1)
    recognition._flush_held_transcripts()
    await _commit(recognition)

    assert _turn_events(span_exporter) == [expected]
    assert recognition._stt_events == []

    received_at = time.time()
    await recognition._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            created_at=received_at,
            alternatives=[stt.SpeechData(language="en", text="next")],
        )
    )
    await _commit(recognition)
    assert _turn_events(span_exporter) == [
        expected,
        [{"received_at": received_at, "type": "final_transcript", "transcript_length": 4}],
    ]


@pytest.mark.parametrize("finish", ["clear", "close"])
async def test_incomplete_turn_keeps_held_transcript_metadata(
    recognition: AudioRecognition, span_exporter: InMemorySpanExporter, finish: str
) -> None:
    received_at = time.time()
    recognition._transcript_gate_active = True
    await recognition._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
            created_at=received_at,
            alternatives=[stt.SpeechData(language="en", text="unfinished")],
        )
    )
    if finish == "clear":
        recognition._clear_user_turn()
    else:
        await recognition._aclose()

    assert _turn_events(span_exporter) == [
        [{"received_at": received_at, "type": "interim_transcript", "transcript_length": 10}]
    ]
    assert recognition._stt_events == []


async def test_transcript_list_is_not_limited_by_span_event_count(
    recognition: AudioRecognition, span_exporter: InMemorySpanExporter
) -> None:
    recognition._transcript_gate_active = True
    for _ in range(130):
        await recognition._on_stt_event(
            stt.SpeechEvent(type=stt.SpeechEventType.INTERIM_TRANSCRIPT)
        )
    await recognition._aclose()

    [events] = _turn_events(span_exporter)
    assert len(events) == 130
    assert all(event["transcript_length"] == 0 for event in events)


# -- keyterm_detection --


def _detector(monkeypatch: pytest.MonkeyPatch) -> tuple[KeytermDetector, object]:
    async def fake_detect(**kwargs: object) -> tuple[list[str], list[str], list[str]]:
        return [], ["Acme", "LiveKit"], []

    monkeypatch.setattr(keyterm_detection, "_detect_keyterms", fake_detect)
    detector = KeytermDetector(static_keyterms=["Zed"])
    detector._llm = FakeLLM()  # type: ignore[assignment]
    root = tracer.start_span("agent_session")
    detector._session = SimpleNamespace(  # type: ignore[assignment]
        _root_span_context=trace.set_span_in_context(root),
        history=ChatContext.empty(),
    )
    return detector, root


async def test_detection_pass_nests_under_the_replying_agent_turn(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector, root = _detector(monkeypatch)
    with tracer.start_as_current_span("agent_turn") as turn:
        detector._on_conversation_item_added(
            ConversationItemAddedEvent(
                item=llm.ChatMessage(role="user", content=["order for Acme"])
            )
        )
        assert detector._detect_task is not None
        await detector._detect_task
    root.end()  # type: ignore[attr-defined]

    [span] = [s for s in span_exporter.get_finished_spans() if s.name == "keyterm_detection"]
    assert span.parent is not None and span.parent.span_id == turn.get_span_context().span_id
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_KEYTERMS_COUNT] == 3  # Zed + Acme + LiveKit
    assert attrs[trace_types.ATTR_KEYTERMS_ADDED] == 2
    assert attrs[trace_types.ATTR_KEYTERMS_REMOVED] == 0
    assert attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == detector._llm.model
    assert detector.keyterms == ["Zed", "Acme", "LiveKit"]


async def test_detection_pass_without_a_reply_falls_back_to_the_session(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector, root = _detector(monkeypatch)
    # no span current: user code edited the history, or the reply was skipped
    await detector._run_once(ChatContext.empty(), detector._session._root_span_context)  # type: ignore[union-attr]
    root.end()  # type: ignore[attr-defined]
    [span] = [s for s in span_exporter.get_finished_spans() if s.name == "keyterm_detection"]
    assert span.parent is not None
    assert span.parent.span_id == root.get_span_context().span_id  # type: ignore[attr-defined]


# -- interruption, handoff and adapter spans --


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
    assert (turn.attributes or {})[trace_types.ATTR_INTERRUPTION_SOURCE] == "audio_activity"
    position = (turn.attributes or {})[trace_types.ATTR_PLAYOUT_POSITION]
    assert isinstance(position, float)
    # ~2 s of the 10 s story had played (5.5 - 3.5), scaled by the speed factor
    assert 0.5 < position < 10.0 / speed

    # the reply to "Stop!" was not interrupted and names no source
    for turn in _spans(span_exporter, "agent_turn"):
        if turn is not interrupted[0]:
            assert trace_types.ATTR_INTERRUPTION_SOURCE not in (turn.attributes or {})
    assert_trace_well_formed(span_exporter.get_finished_spans())


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
    assert_trace_well_formed(span_exporter.get_finished_spans())


@pytest.mark.parametrize("phase", ["drain", "start"])
@pytest.mark.parametrize("redacted", [False, True])
async def test_update_agent_records_failure_once(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    redacted: bool,
) -> None:
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redacted)
    session = AgentSession()
    failure = RuntimeError("private failure details")
    try:
        await session.start(_FirstAgent())
        assert session._activity is not None
        with monkeypatch.context() as patch:
            if phase == "drain":
                patch.setattr(session._activity, "drain", AsyncMock(side_effect=failure))
            else:
                patch.setattr(AgentActivity, "start", AsyncMock(side_effect=failure))
            with pytest.raises(RuntimeError) as raised:
                await session._update_activity_task(None, _SecondAgent())
            assert raised.value is failure
    finally:
        await session.aclose()

    [handoff] = _spans(span_exporter, "update_agent")
    [event] = _events(handoff, "exception")
    expected = telemetry_utils.REDACTED_EXCEPTION_MESSAGE if redacted else str(failure)
    assert event.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == expected
    assert handoff.status.description == expected


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


@pytest.mark.parametrize("through_node", [False, True], ids=["stream", "llm_node"])
async def test_llm_fallback_records_failed_and_serving_provider(
    span_exporter: InMemorySpanExporter,
    through_node: bool,
) -> None:
    primary = _FailingLLM()
    secondary = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hi", content="hello", ttft=0.01, duration=0.02)]
    )
    adapter = FallbackAdapter([primary, secondary], attempt_timeout=1.0, max_retry_per_llm=0)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    try:
        if through_node:

            def node(
                chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
            ) -> LLMStream:
                return adapter.chat(chat_ctx=chat_ctx, tools=tools)

            task, data = generation.perform_llm_inference(
                node=node,
                chat_ctx=chat_ctx,
                tool_ctx=llm.ToolContext([]),
                model_settings=ModelSettings(),
                model=adapter.model,
                provider=adapter.provider,
            )
            assert await task is True
            response = data.generated_text
            [caller] = _spans(span_exporter, "llm_node")
        else:
            with tracer.start_as_current_span("caller") as caller:
                stream = adapter.chat(chat_ctx=chat_ctx)
                chunks = [chunk async for chunk in stream]
                await stream.aclose()
            response = "".join(c.delta.content or "" for c in chunks if c.delta)
    finally:
        await asyncio.gather(*(s.recovering_task for s in adapter._status if s.recovering_task))
        await adapter.aclose()
        await primary.aclose()
        await secondary.aclose()

    assert response == "hello"

    # each instance's request nests directly under the adapter's, with no attempt span between
    [request] = _spans(span_exporter, "llm_fallback_adapter")
    children = [
        s
        for s in _spans(span_exporter, "llm_request")
        if s.parent is not None and s.parent.span_id == request.context.span_id
    ]
    instance_requests = sorted(
        (s for s in children if not (s.attributes or {}).get(trace_types.ATTR_FALLBACK_RECOVERY)),
        key=lambda s: s.start_time or 0,
    )
    models = [
        (s.attributes or {})[trace_types.ATTR_GEN_AI_REQUEST_MODEL] for s in instance_requests
    ]
    assert models == [primary.model, secondary.model]
    # the background check that the primary is back is marked, not mistaken for an attempt
    [probe] = [s for s in children if s not in instance_requests]
    assert (probe.attributes or {})[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == primary.model
    # the failed attempt names the model it asked on its own span
    [failed_run] = [
        s
        for s in _spans(span_exporter, "llm_request_run")
        if s.parent is not None and s.parent.span_id == instance_requests[0].context.span_id
    ]
    assert failed_run.status.status_code is StatusCode.ERROR
    assert (failed_run.attributes or {})[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == primary.model
    assert (failed_run.attributes or {})[
        trace_types.ATTR_GEN_AI_PROVIDER_NAME
    ] == trace_types.gen_ai_provider_name(primary.provider)
    # the adapter's span names the one that served: request = expected (the primary),
    # response = who answered
    request_attrs = request.attributes or {}
    assert request_attrs[trace_types.ATTR_FALLBACK_LABEL] == secondary.label
    assert request_attrs[trace_types.ATTR_FALLBACK_INDEX] == 1
    assert trace_types.ATTR_GEN_AI_OPERATION_NAME not in request_attrs
    assert request_attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == primary.model
    assert request_attrs[trace_types.ATTR_GEN_AI_RESPONSE_MODEL] == secondary.model
    # the caller's span gets the same response side, per request: a concurrent request on
    # the same adapter may be served by a different instance, so this is not read back from
    # the adapter's shared state once the node finishes
    assert isinstance(caller, ReadableSpan)
    caller_attrs = caller.attributes or {}
    assert caller_attrs[trace_types.ATTR_GEN_AI_RESPONSE_MODEL] == secondary.model
    assert caller_attrs[trace_types.ATTR_GEN_AI_PROVIDER_NAME] == trace_types.gen_ai_provider_name(
        secondary.provider
    )
    if through_node:
        assert caller_attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == primary.model
    # and the adapter itself now reports who serves next
    assert adapter.model == secondary.model and adapter.provider == secondary.provider


class _BrokenTTS(FakeTTS):
    @property
    def model(self) -> str:
        return "broken-tts"

    @property
    def provider(self) -> str:
        return "broken"


@pytest.mark.parametrize("streaming", [False, True], ids=["chunked", "stream"])
async def test_tts_fallback_nests_instance_requests_directly(
    span_exporter: InMemorySpanExporter, streaming: bool
) -> None:
    primary = _BrokenTTS(fake_exception=APIConnectionError("primary down"))
    secondary = FakeTTS(fake_audio_duration=0.1)
    adapter = TTSFallbackAdapter([primary, secondary], max_retry_per_tts=0)
    try:
        if streaming:
            stream = adapter.stream(conn_options=APIConnectOptions(max_retry=0))
            stream.push_text("hello world.")
            stream.end_input()
        else:
            stream = adapter.synthesize("hello world.")
        async with stream:
            _ = [audio async for audio in stream]
    finally:
        for status in adapter._status:
            for task in (status.recovering_synthesize_task, status.recovering_stream_task):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()
        await primary.aclose()
        await secondary.aclose()

    [request] = _spans(span_exporter, "tts_fallback_adapter")
    children = [
        s
        for s in span_exporter.get_finished_spans()
        if s.parent is not None and s.parent.span_id == request.context.span_id
    ]
    assert {s.name for s in children} == {"tts_request"}
    probes = [s for s in children if (s.attributes or {}).get(trace_types.ATTR_FALLBACK_RECOVERY)]
    attempts = sorted((s for s in children if s not in probes), key=lambda s: s.start_time or 0)
    # the failed primary, then the secondary that served, each naming its model
    models = [(s.attributes or {})[trace_types.ATTR_GEN_AI_REQUEST_MODEL] for s in attempts]
    assert models == [primary.model, secondary.model]
    [probe] = probes
    assert (probe.attributes or {})[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == primary.model
    # the failed attempt names the model it asked on its own span
    [failed_run] = [
        s
        for s in _spans(span_exporter, "tts_request_run")
        if s.parent is not None and s.parent.span_id == attempts[0].context.span_id
    ]
    assert failed_run.status.status_code is StatusCode.ERROR
    failed_attrs = failed_run.attributes or {}
    assert failed_attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == primary.model
    assert failed_attrs[trace_types.ATTR_GEN_AI_PROVIDER_NAME] == "broken"
    # the adapter's span: request = expected (the primary), response = who answered
    request_attrs = request.attributes or {}
    assert request_attrs[trace_types.ATTR_FALLBACK_LABEL] == secondary.label
    assert request_attrs[trace_types.ATTR_FALLBACK_INDEX] == 1
    assert request_attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == primary.model
    assert request_attrs[trace_types.ATTR_GEN_AI_RESPONSE_MODEL] == secondary.model


async def test_adapter_request_shapes_are_allowed(span_exporter: InMemorySpanExporter) -> None:
    """Each fallback or stream adapter opens the wrapped request directly under its own span."""
    llm_adapter = FallbackAdapter(
        [
            FakeLLM(
                fake_responses=[
                    FakeLLMResponse(input="hi", content="hello", ttft=0.01, duration=0.02)
                ]
            )
        ],
        max_retry_per_llm=0,
    )
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    plain = TTSFallbackAdapter([FakeTTS(fake_audio_duration=0.1)], max_retry_per_tts=0)
    stacked = TTSFallbackAdapter(
        [StreamAdapter(tts=FakeTTS(fake_audio_duration=0.1))], max_retry_per_tts=0
    )
    stream_only = StreamAdapter(tts=FakeTTS(fake_audio_duration=0.1))

    with tracer.start_as_current_span("agent_session"):
        with tracer.start_as_current_span(
            "agent_turn", attributes={"lk.speech_id": "sp", "lk.generation_count": 1}
        ) as turn:
            turn.add_event("generation", {"lk.generation_id": "sp_1"})
            with tracer.start_as_current_span("llm_node"):
                stream = llm_adapter.chat(chat_ctx=chat_ctx)
                _ = [c async for c in stream]
                await stream.aclose()
            with tracer.start_as_current_span("tts_node"):
                chunked = plain.synthesize("hello world.")
                _ = [a async for a in chunked]
                await chunked.aclose()
                for tts in (plain, stacked, stream_only):
                    synth = tts.stream(conn_options=APIConnectOptions(max_retry=0))
                    synth.push_text("hello world.")
                    synth.end_input()
                    _ = [a async for a in synth]
                    await synth.aclose()
    for model in (llm_adapter, plain, stacked, stream_only):
        await model.aclose()

    spans = span_exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"llm_fallback_adapter", "tts_fallback_adapter", "tts_stream_adapter"} <= names
    assert_trace_well_formed(spans)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("redacted", [False, True])
async def test_tts_retry_records_each_exception_once(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    redacted: bool,
) -> None:
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redacted)
    failure = APIConnectionError("private failure details")
    async with FakeTTS(fake_exception=failure) as model:
        conn_options = APIConnectOptions(max_retry=1, retry_interval=0)
        if streaming:
            stream = model.stream(conn_options=conn_options)
            stream.push_text("hello")
            stream.end_input()
        else:
            stream = model.synthesize("hello", conn_options=conn_options)
        async with stream:
            with pytest.raises(APIConnectionError):
                async for _ in stream:
                    pass

    attempts = _spans(span_exporter, "tts_request_run")
    assert len(attempts) == 2
    for attempt in attempts:
        [event] = _events(attempt, "exception")
        expected = telemetry_utils.REDACTED_EXCEPTION_MESSAGE if redacted else str(failure)
        assert event.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == expected


def test_interrupt_source_first_wins() -> None:
    handle = SpeechHandle.create()
    handle.interrupt(source="audio_activity")
    handle.interrupt(source="user_turn")  # already interrupted: the first cause stands
    assert handle._interrupt_source == "audio_activity"
    assert SpeechHandle.create().interrupt()._interrupt_source == "programmatic"
