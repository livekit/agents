"""The ``user_speaking`` span must never end before it starts.

A span whose end precedes its start is exported with a negative duration, which viewers read
as an unsigned 64-bit nanosecond count: ~18446744073700 ms of "user speaking" that swamps
every other span on the trace (livekit/agents#3396).

Two things keep it from happening:

* ``AudioRecognition`` clamps a provider-supplied speech onset to ``now``, the same way it
  already clamps the provider's ``speech_end_time``. An unclamped onset starts the turn in the
  future while every end anchor is clamped, so the span closes before it opened.
* ``AgentSession._update_user_state`` floors the span's end at its own start, so no back-dated
  anchor from any source (a VAD silence window, a late provider timestamp) can invert it. This
  mirrors the floor ``_end_eou_wait_span`` already applies to ``eou_wait``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import stt
from livekit.agents.telemetry import set_tracer_provider, tracer
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

SKEW = 0.5
"""How far ahead of the local clock the provider's onset timestamp sits."""


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


def _make_recognition() -> AudioRecognition:
    """Enough of ``AudioRecognition`` to drive ``_process_stt_event`` in STT turn detection."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._session._room_io = None
    ar._session.amd = None
    ar._session.options.transcription_timeout = None
    ar._hooks = MagicMock()
    ar._stt = MagicMock()
    ar._stt_pipeline = None
    ar._stt_model = None
    ar._stt_provider = None
    ar._vad = None
    ar._turn_detection_mode = "stt"
    ar._turn_detector_stream = None
    ar._vad_base_turn_detection = False
    ar._agent_speaking = False
    ar._audio_transcript = ""
    ar._audio_interim_transcript = ""
    ar._audio_preflight_transcript = ""
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
    ar._user_turn_resumes = 0
    ar._eou_wait_span = None
    ar._eou_wait_started_at_ns = None
    ar._eou_wait_rearms = 0
    ar._eou_wait_floor_ns = None
    ar._eou_wait_not_committed = 0
    ar._eou_detection_span = None
    ar._transcription_timeout_handle = None
    ar._turn_speech_duration = 0.0
    return ar


def _make_session() -> AgentSession:
    """Enough of ``AgentSession`` to drive ``_update_user_state``'s span handling."""
    s = AgentSession.__new__(AgentSession)
    s._user_turn_claims = 0
    s._user_state = "listening"
    s._agent_state = "listening"
    s._user_speaking_span = None
    s._user_speaking_started_at_ns = None
    s._room_io = None
    s._add_session_event = lambda *args, **kwargs: None  # type: ignore[method-assign]
    s.emit = lambda *args, **kwargs: None  # type: ignore[method-assign]
    s._set_user_away_timer = lambda: None  # type: ignore[method-assign]
    s._cancel_user_away_timer = lambda: None  # type: ignore[method-assign]
    return s


def _user_speaking(exporter: InMemorySpanExporter) -> ReadableSpan:
    [span] = [s for s in exporter.get_finished_spans() if s.name == "user_speaking"]
    return span


@pytest.mark.asyncio
async def test_provider_onset_ahead_of_the_local_clock_is_clamped() -> None:
    """A provider whose stream clock runs ahead (audio pushed faster than realtime, a re-seeded
    stream anchor) reports an onset in the future. It anchors the turn, so it is clamped to
    ``now`` — as ``speech_end_time`` from the same stream already is."""
    ar = _make_recognition()
    now = time.time()

    ar._process_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.START_OF_SPEECH,
            speech_start_time=now + SKEW,
        )
    )

    assert ar._speech_start_time is not None
    assert ar._speech_start_time <= time.time(), (
        f"turn anchored {(ar._speech_start_time - now) * 1000:.0f} ms in the future; "
        "every end anchor is clamped to now, so the spans built from it would end "
        "before they began"
    )
    [call] = ar._hooks.on_start_of_speech.call_args_list
    assert call.kwargs["speech_start_time"] == ar._speech_start_time


@pytest.mark.asyncio
async def test_provider_onset_in_the_past_is_kept() -> None:
    """The clamp is one-sided: a back-dated onset is what the anchor is for."""
    ar = _make_recognition()
    onset = time.time() - 1.5

    ar._process_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.START_OF_SPEECH,
            speech_start_time=onset,
        )
    )

    assert ar._speech_start_time == onset


@pytest.mark.asyncio
async def test_user_speaking_span_never_ends_before_it_starts(
    span_exporter: InMemorySpanExporter,
) -> None:
    """Whatever anchors the two transitions, the span keeps a non-negative duration."""
    session = _make_session()
    start = time.time()

    session._update_user_state("speaking", last_speaking_time=start)
    # a back-dated end anchor: the VAD reports the segment's end minus its silence window
    session._update_user_state("listening", last_speaking_time=start - SKEW)

    span = _user_speaking(span_exporter)
    assert span.end_time is not None and span.start_time is not None
    assert span.end_time >= span.start_time, (
        f"user_speaking ends {(span.start_time - span.end_time) / 1e6:.1f} ms before it starts; "
        "viewers read the negative duration as "
        f"{((span.end_time - span.start_time) % 2**64) / 1e6:.0f} ms"
    )
    assert span.start_time == int(start * 1_000_000_000)


@pytest.mark.asyncio
async def test_user_speaking_span_keeps_its_back_dated_bounds(
    span_exporter: InMemorySpanExporter,
) -> None:
    """The floor only clips an inverted end: ordered anchors are used as given."""
    session = _make_session()
    start = time.time() - 2.0
    end = start + 1.25

    session._update_user_state("speaking", last_speaking_time=start)
    session._update_user_state("listening", last_speaking_time=end)

    span = _user_speaking(span_exporter)
    assert span.start_time == int(start * 1_000_000_000)
    assert span.end_time == int(end * 1_000_000_000)
