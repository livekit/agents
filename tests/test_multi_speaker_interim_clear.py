"""Regression tests for the empty final ``MultiSpeakerAdapter`` emits for a suppressed segment.

With ``suppress_background_speaker=True`` the adapter forwards a background
segment's interim (diarizing providers report interims without a speaker id, so
they bypass the suppression check) and then drops the attributed final. In its
place it emits a ``FINAL_TRANSCRIPT`` whose only alternative has empty text:
"send an empty final transcript to clear the interim results" (see
``stt/multi_speaker_adapter.py``).

The transport cannot express a wire-level clear (an empty text stream carries
zero data chunks, which clients ignore), so ``AudioRecognition`` does not
forward the event. It does use the empty final to drop the interim buffered for
the suppressed segment. Otherwise those stale background words stay in
``_current_transcript`` - where they count toward the ``min_words`` interruption
gate - and ``commit_user_turn`` appends them to the user turn.

The empty final carries no segment identity, so recognition only drops interim
text that can belong to the suppressed segment: a speaker-less (Deepgram-style)
interim is dropped, while an interim the adapter attributed to the primary
speaker is another segment's in-progress words and survives for the commit. When
no primary interim is pending, the empty final also refreshes the manual-commit
freshness anchor and sets the completion signal: the wrapped STT did finalize,
so a commit must not wait for yet another final. While a primary interim is
alive, neither happens: that final closed a different segment, so the commit
keeps waiting for the primary speaker's own final (falling back to the interim
after ``transcript_timeout``) instead of promoting the interim and discarding
the final that follows.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.agents.language import LanguageCode
from livekit.agents.stt import MultiSpeakerAdapter, RecognizeStream
from livekit.agents.stt.multi_speaker_adapter import PrimarySpeakerDetectionOptions
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit

_CONN = APIConnectOptions(max_retry=0, timeout=5.0)


def _make_recognition() -> AudioRecognition:
    """Wire the attributes the transcript path of ``_on_stt_event`` touches."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session._root_span_context = None
    ar._session.amd = None
    ar._session._room_io = None
    ar._hooks = MagicMock()
    ar._vad = None
    ar._turn_detection_mode = "manual"
    ar._user_turn_committed = False
    ar._vad_base_turn_detection = False
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._interruption_enabled = False
    ar._agent_speaking = False
    ar._transcript_buffer = []
    ar._transcript_gate_active = False
    ar._stt_aligned_transcript = False
    ar._backchannel_boundary = None
    ar._turn_transcript_received = False
    ar._transcription_timeout_handle = None
    ar._stt_request_ids = []
    ar._stt_events = []
    ar._end_of_turn_task = None
    ar._audio_transcript = ""
    ar._audio_interim_transcript = ""
    ar._audio_interim_from_primary = False
    ar._audio_preflight_transcript = ""
    ar._final_transcript_confidence = []
    ar._final_transcript_received = asyncio.Event()
    ar._last_language = None
    ar._last_final_transcript_time = None
    ar._turn_tracker = MagicMock()
    ar._last_speaking_time = None
    ar._sample_rate = None
    ar._vad_ch = None
    ar._interruption_ch = None
    ar._stt_pipeline = MagicMock()
    ar._stt_pipeline.input_started_at = time.time()
    # model an STT without manual flush: _commit_user_turn then relies on the
    # final-transcript freshness check, as these tests exercise
    ar._stt_pipeline.manual_flush = False
    ar._stt_flush_requested = False
    ar._check_user_turn_limit = MagicMock()  # type: ignore[method-assign]
    ar._speech_start_time = None
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    # only reached by _commit_user_turn
    ar._stt = MagicMock()
    ar._closing = asyncio.Event()
    ar._commit_user_turn_atask = None
    return ar


def _final_transcript(text: str, *, speaker_id: str | None) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=LanguageCode("en"),
                text=text,
                confidence=0.9,
                speaker_id=speaker_id,
            )
        ],
    )


def _interim_transcript(text: str) -> stt.SpeechEvent:
    # Deepgram sets speaker=None on every non-final result, so a diarizing
    # provider's interim carries no attribution and bypasses the adapter's
    # background-speaker suppression.
    return stt.SpeechEvent(
        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[stt.SpeechData(language=LanguageCode("en"), text=text, speaker_id=None)],
    )


def _empty_final_transcript() -> stt.SpeechEvent:
    # Exactly what MultiSpeakerAdapterWrapper._forward_output emits for a
    # suppressed final: one alternative with empty text and no language.
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language=LanguageCode(""), text="")],
    )


async def _feed_adapter_sequence(ar: AudioRecognition) -> None:
    """Feed the events the adapter emits for a suppressed background segment."""
    await ar._on_stt_event(_final_transcript("hello", speaker_id="S0"))
    await ar._on_stt_event(_interim_transcript("background chatter"))
    await ar._on_stt_event(_empty_final_transcript())


async def test_empty_final_clears_the_background_interim() -> None:
    ar = _make_recognition()

    await ar._on_stt_event(_final_transcript("hello", speaker_id="S0"))
    await ar._on_stt_event(_interim_transcript("background chatter"))

    # control: the speaker-less interim is buffered as usual
    assert ar._audio_interim_transcript == "background chatter"
    assert ar._current_transcript == "hello background chatter"

    await ar._on_stt_event(_empty_final_transcript())

    assert ar._audio_interim_transcript == ""
    assert ar._audio_preflight_transcript == ""
    assert ar._current_transcript == "hello"
    assert ar._final_transcript_received.is_set()


async def test_empty_final_without_buffered_interim_keeps_empty_state() -> None:
    """An empty final with nothing buffered stays a no-op for the turn text."""
    ar = _make_recognition()

    await ar._on_stt_event(_empty_final_transcript())

    assert ar._audio_interim_transcript == ""
    assert ar._audio_preflight_transcript == ""
    assert ar._current_transcript == ""
    assert ar._final_transcript_received.is_set()


async def test_empty_final_keeps_background_text_out_of_commit_user_turn() -> None:
    ar = _make_recognition()

    await _feed_adapter_sequence(ar)

    committed = await ar._commit_user_turn(
        audio_detached=False, transcript_timeout=0.01, stt_flush_duration=0.0
    )

    assert committed == "hello"


async def test_commit_after_empty_final_does_not_wait_for_another_final() -> None:
    """An empty final satisfies the manual-commit freshness check.

    The commit waits for a new final only when the last one is older than 0.5s,
    because it then assumes the STT still holds unfinalized audio. A suppressed
    background segment produces a real final, so once the empty final arrives
    the STT buffer is flushed: waiting for yet another final stalls the commit
    until ``transcript_timeout``.
    """
    ar = _make_recognition()

    await ar._on_stt_event(_final_transcript("hello", speaker_id="S0"))
    await asyncio.sleep(0.55)  # the last final is now outside the 0.5s window
    await ar._on_stt_event(_interim_transcript("background chatter"))
    await ar._on_stt_event(_empty_final_transcript())

    started = time.monotonic()
    committed = await ar._commit_user_turn(
        audio_detached=False, transcript_timeout=1.0, stt_flush_duration=0.0
    )
    elapsed = time.monotonic() - started

    assert committed == "hello"
    assert elapsed < 0.5, f"commit waited {elapsed:.2f}s for a final that already arrived"


# --- the empty final through the real MultiSpeakerAdapter -------------------
#
# The tests above feed the event sequence directly. The adapter's empty final is
# not addressed to a particular interim, so these drive the real adapter to pin
# down which interim it is allowed to clear.

_SAMPLE_RATE = 1000
_SAMPLES_PER_FRAME = 100  # 100 ms, matching frame_size_ms below


def _tone_frame(amplitude: int) -> rtc.AudioFrame:
    """A mono int16 frame whose every sample equals ``amplitude``."""
    return rtc.AudioFrame(
        data=amplitude.to_bytes(2, "little", signed=True) * _SAMPLES_PER_FRAME,
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=_SAMPLES_PER_FRAME,
    )


def _timed_transcript(
    event_type: stt.SpeechEventType,
    text: str,
    *,
    speaker_id: str | None,
    start_time: float,
    end_time: float,
) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=event_type,
        alternatives=[
            stt.SpeechData(
                language=LanguageCode("en"),
                text=text,
                confidence=0.9,
                speaker_id=speaker_id,
                start_time=start_time,
                end_time=end_time,
            )
        ],
    )


class _ScriptedDiarizingSTT(stt.STT):
    """Diarization-capable STT that replays one event sequence per stream.

    The adapter runs RMS-based primary-speaker detection over the audio it
    forwards, so the scripted stream waits for ``expected_frames`` frames before
    emitting, matching the order a provider produces: audio first, then the
    results for it.
    """

    def __init__(self, events: list[stt.SpeechEvent], *, expected_frames: int) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True, diarization=True)
        )
        self._scripted_events = events
        self._expected_frames = expected_frames

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("this fake only supports streaming")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return _ScriptedDiarizingStream(
            stt=self,
            events=self._scripted_events,
            expected_frames=self._expected_frames,
            conn_options=conn_options,
        )


class _ScriptedDiarizingStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: stt.STT,
        events: list[stt.SpeechEvent],
        expected_frames: int,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._scripted_events = events
        self._expected_frames = expected_frames

    async def _run(self) -> None:
        received = 0
        while received < self._expected_frames:
            if isinstance(await self._input_ch.recv(), rtc.AudioFrame):
                received += 1
        for event in self._scripted_events:
            self._event_ch.send_nowait(event)


def _diarizing_adapter(events: list[stt.SpeechEvent], *, frames: int) -> MultiSpeakerAdapter:
    return MultiSpeakerAdapter(
        stt=_ScriptedDiarizingSTT(events, expected_frames=frames),
        suppress_background_speaker=True,
        primary_detection_options=PrimarySpeakerDetectionOptions(
            frame_size_ms=100, min_rms_samples=1
        ),
    )


async def _adapter_events(
    adapter: MultiSpeakerAdapter, frames: list[rtc.AudioFrame], count: int
) -> list[stt.SpeechEvent]:
    stream = adapter.stream(conn_options=_CONN)
    for frame in frames:
        stream.push_frame(frame)
    events: list[stt.SpeechEvent] = []
    async for event in stream:
        events.append(event)
        if len(events) == count:
            break
    await stream.aclose()
    return events


async def test_primary_interim_survives_a_suppressed_background_final() -> None:
    """A background final must not drop the primary speaker's interim.

    Diarizing providers that attribute interim results (AssemblyAI's turns carry
    ``speaker_label``) let the adapter forward a primary interim. A quieter
    background final right after it is suppressed and replaced by the empty
    final, which carries no segment identity: clearing the primary interim here
    would make a manual commit lose speech that has not been finalized yet.
    """
    events = [
        _timed_transcript(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            "hello",
            speaker_id="S0",
            start_time=0.0,
            end_time=0.2,
        ),
        _timed_transcript(
            stt.SpeechEventType.INTERIM_TRANSCRIPT,
            "book a ticket",
            speaker_id="S0",
            start_time=0.2,
            end_time=0.3,
        ),
        # quiet background speaker: the detector keeps S0 as primary and the
        # adapter suppresses this final, emitting the empty final instead
        _timed_transcript(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            "background chatter",
            speaker_id="S1",
            start_time=0.3,
            end_time=0.4,
        ),
    ]
    adapter = _diarizing_adapter(events, frames=4)
    frames = [_tone_frame(3000), _tone_frame(3000), _tone_frame(50), _tone_frame(50)]

    emitted = await _adapter_events(adapter, frames, count=3)

    assert [event.type for event in emitted] == [
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert emitted[0].alternatives[0].text == "hello"
    assert emitted[1].alternatives[0].text == "book a ticket"
    assert emitted[2].alternatives[0].text == ""  # the adapter's synthesized clear

    ar = _make_recognition()
    for event in emitted:
        await ar._on_stt_event(event)

    assert ar._audio_interim_transcript == "book a ticket"
    committed = await ar._commit_user_turn(
        audio_detached=False, transcript_timeout=0.01, stt_flush_duration=0.0
    )
    assert committed == "hello book a ticket"


async def test_commit_waits_for_primary_final_after_a_suppressed_background_final() -> None:
    """The empty final must not complete a commit that waits on the primary segment.

    The empty final closes the suppressed background segment, not the primary
    interim buffered before it. If it marked the commit complete, the commit
    would promote the interim to final and return; the primary final that
    arrives right after would then be dropped by the committed-turn gate. The
    commit has to keep waiting for the primary final (and fall back to the
    interim only after ``transcript_timeout``).
    """
    events = [
        _timed_transcript(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            "hello",
            speaker_id="S0",
            start_time=0.0,
            end_time=0.2,
        ),
        _timed_transcript(
            stt.SpeechEventType.INTERIM_TRANSCRIPT,
            "book a ticket",
            speaker_id="S0",
            start_time=0.2,
            end_time=0.3,
        ),
        _timed_transcript(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            "background chatter",
            speaker_id="S1",
            start_time=0.3,
            end_time=0.4,
        ),
    ]
    adapter = _diarizing_adapter(events, frames=4)
    frames = [_tone_frame(3000), _tone_frame(3000), _tone_frame(50), _tone_frame(50)]

    emitted = await _adapter_events(adapter, frames, count=3)
    assert emitted[1].alternatives[0].is_primary_speaker is True
    assert emitted[2].alternatives[0].text == ""  # the adapter's synthesized clear

    ar = _make_recognition()
    for event in emitted:
        await ar._on_stt_event(event)

    commit = ar._commit_user_turn(
        audio_detached=False, transcript_timeout=1.0, stt_flush_duration=0.0
    )
    await asyncio.sleep(0.1)
    assert not commit.done(), "commit returned before the primary final arrived"

    await ar._on_stt_event(_final_transcript("book a ticket tomorrow", speaker_id="S0"))
    committed = await asyncio.wait_for(commit, timeout=1.0)

    assert "tomorrow" in committed
    assert committed == "hello book a ticket tomorrow"


async def test_speakerless_interim_is_still_cleared_by_the_empty_final() -> None:
    """Control: the Deepgram-style case this fix targets is still cleared.

    Deepgram reports interims without a speaker id, so they bypass the adapter's
    suppression and reach recognition. The suppressed segment's final arrives as
    the empty final, and the buffered interim is that same segment's partial
    text: it must not stay in the user turn.
    """
    events = [
        _timed_transcript(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            "hello",
            speaker_id="S0",
            start_time=0.0,
            end_time=0.2,
        ),
        _timed_transcript(
            stt.SpeechEventType.INTERIM_TRANSCRIPT,
            "background chatter",
            speaker_id=None,
            start_time=0.2,
            end_time=0.3,
        ),
        _timed_transcript(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            "background chatter",
            speaker_id="S1",
            start_time=0.3,
            end_time=0.4,
        ),
    ]
    adapter = _diarizing_adapter(events, frames=4)
    frames = [_tone_frame(3000), _tone_frame(3000), _tone_frame(50), _tone_frame(50)]

    emitted = await _adapter_events(adapter, frames, count=3)
    assert [event.alternatives[0].text for event in emitted] == [
        "hello",
        "background chatter",
        "",
    ]

    ar = _make_recognition()
    for event in emitted:
        await ar._on_stt_event(event)

    assert ar._audio_interim_transcript == ""
    assert ar._current_transcript == "hello"
    committed = await ar._commit_user_turn(
        audio_detached=False, transcript_timeout=0.01, stt_flush_duration=0.0
    )
    assert committed == "hello"
