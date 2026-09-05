"""The ``eot_wait`` span: one per user turn, from the last speech anchor to the turn decision.

``eot_detection`` (the turn-detector inference) nests under it; the wait itself was previously
invisible, so a 2.5 s endpointing delay showed up as an empty gap between ``eot_detection`` and
``agent_turn``. Also covers the ``on_user_turn_completed`` span and the queue-wait attribute on
``agent_turn`` through a full fake session."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, llm, vad
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.turn import (
    TurnDetectionEvent,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)

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


def _events(span: ReadableSpan, name: str) -> list:
    return [e for e in span.events if e.name == name]


def _make_recognition(*, min_delay: float, with_detector: bool = False) -> AudioRecognition:
    """Enough of ``AudioRecognition`` to drive ``_run_eou_detection`` with real spans.

    ``_hooks.on_end_of_turn`` commits by default. VAD-only turn detection unless
    ``with_detector`` wires the streaming turn-detector mocks (for ``eot_detection``)."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
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
    ar._eot_wait_span = None
    ar._eot_wait_started_at = None
    ar._eot_wait_rearms = 0
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

    [wait] = _spans(span_exporter, "eot_wait")
    [user_turn] = _spans(span_exporter, "user_turn")
    assert wait.parent is not None
    assert wait.parent.span_id == user_turn.context.span_id

    assert wait.start_time == int(last_speaking * 1_000_000_000)
    attrs = wait.attributes or {}
    assert attrs[trace_types.ATTR_EOU_OUTCOME] == "committed"
    assert attrs[trace_types.ATTR_EOU_SOURCE] == "vad"
    assert attrs[trace_types.ATTR_EOU_DELAY] == 0.05
    assert attrs[trace_types.ATTR_EOU_REARM_COUNT] == 0
    wait_duration = attrs[trace_types.ATTR_EOU_WAIT_DURATION]
    assert isinstance(wait_duration, float)
    assert 0.3 <= wait_duration < 0.6
    assert wait.end_time is not None
    assert abs((wait.end_time - wait.start_time) / 1e9 - wait_duration) < 1e-6
    # the wait closes before the turn does
    assert user_turn.end_time is not None and wait.end_time <= user_turn.end_time
    assert ar._eot_wait_span is None


async def test_later_trigger_rearms_the_same_span(span_exporter: InMemorySpanExporter) -> None:
    ar = _make_recognition(min_delay=0.3)
    ar._last_speaking_time = time.time()

    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await asyncio.sleep(0.05)
    # a late STT final re-triggers end of turn; the bounce task restarts, the span must not
    ar._run_eou_detection(llm.ChatContext(), trigger="stt")
    await _await_bounce(ar)

    [wait] = _spans(span_exporter, "eot_wait")
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

    [wait] = _spans(span_exporter, "eot_wait")
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


async def test_teardown_drops_an_open_wait(span_exporter: InMemorySpanExporter) -> None:
    ar = _make_recognition(min_delay=1.0)
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await asyncio.sleep(0.02)

    ar._end_user_turn_span()
    await _cancel_bounce(ar)

    [wait] = _spans(span_exporter, "eot_wait")
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
    assert _spans(span_exporter, "eot_wait") == []
    assert ar._eot_wait_span is not None and ar._eot_wait_span.is_recording()
    ar._end_user_turn_span()
    [wait] = _spans(span_exporter, "eot_wait")
    assert len(_events(wait, "not_committed")) == 1
    assert (wait.attributes or {})[trace_types.ATTR_EOU_OUTCOME] == "dropped"


async def test_detection_nests_under_wait_with_new_name(
    span_exporter: InMemorySpanExporter,
) -> None:
    ar = _make_recognition(min_delay=0.01, with_detector=True)
    ar._last_speaking_time = time.time()
    ar._run_eou_detection(llm.ChatContext(), trigger="vad")
    await _await_bounce(ar)

    assert _spans(span_exporter, "eou_detection") == []
    [detection] = _spans(span_exporter, "eot_detection")
    [wait] = _spans(span_exporter, "eot_wait")
    assert detection.parent is not None
    assert detection.parent.span_id == wait.context.span_id
    assert (detection.attributes or {})[trace_types.ATTR_EOU_PROBABILITY] == 0.9
    # the delay decided by the prediction is stamped on the wait too
    assert (wait.attributes or {})[trace_types.ATTR_EOU_DELAY] == 0.01


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

    [wait] = _spans(span_exporter, "eot_wait")
    [user_turn] = _spans(span_exporter, "user_turn")
    assert wait.parent is not None and wait.parent.span_id == user_turn.context.span_id
    assert (wait.attributes or {})[trace_types.ATTR_EOU_OUTCOME] == "committed"

    [hook] = _spans(span_exporter, "on_user_turn_completed")
    assert hook.end_time is not None
    assert (hook.end_time - hook.start_time) / 1e9 >= 0.05
    assert (hook.attributes or {})[trace_types.ATTR_AGENT_LABEL] == "_hook_agent"

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
