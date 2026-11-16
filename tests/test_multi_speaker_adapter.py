"""Verify adapter delegation and primary-speaker detection."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, utils
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.metrics import STTMetrics
from livekit.agents.stt import (
    STT,
    MultiSpeakerAdapter,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.stt.multi_speaker_adapter import (
    PrimarySpeakerDetectionOptions,
    _PrimarySpeakerDetector,
)
from livekit.agents.stt.stt import RecognitionUsage
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.keyterm_detection import KeytermDetector

from .test_transcription_delay_anchor import _make_recognition

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

_CONN = APIConnectOptions(max_retry=0, timeout=5.0)


class _RecordingSTT(STT):
    """Diarization-capable STT that records every framework hook it receives.

    ``diarization`` is required by MultiSpeakerAdapter; ``keyterms`` and ``chat_context``
    are both offered by shipped diarization plugins (deepgram, assemblyai).
    """

    def __init__(self) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=True,
                keyterms=True,
                chat_context=True,
            )
        )
        self.pushed_keyterms: list[list[str]] = []
        self.chat_items: list[ChatMessage] = []
        self.prewarmed = False

    @property
    def model(self) -> str:
        return "recording-model-v1"

    @property
    def provider(self) -> str:
        return "recording-provider"

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self.pushed_keyterms.append(list(keyterms))

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        if isinstance(ev.item, ChatMessage):
            self.chat_items.append(ev.item)

    def prewarm(self) -> None:
        self.prewarmed = True

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _CONN,
    ) -> SpeechEvent:
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id="req-1",
            alternatives=[SpeechData(language="en", text="hello")],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _CONN,
    ) -> RecognizeStream:
        return _RecordingStream(stt=self, conn_options=conn_options)


class _RecordingStream(RecognizeStream):
    """Emits one transcript plus the provider's usage report, as a real plugin stream does."""

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                request_id="req-1",
                alternatives=[SpeechData(language="en", text="hello", speaker_id="A")],
            )
        )
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.RECOGNITION_USAGE,
                request_id="req-1",
                recognition_usage=RecognitionUsage(audio_duration=1.0),
            )
        )


class _HiccupSTT(_RecordingSTT):
    """Its stream delivers a good transcript and then drops the connection, every attempt.

    ``max_retry=0`` on the inner stream mirrors a plugin that has already spent its own retry
    budget, so each hiccup propagates to the adapter's wrapper.
    """

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _CONN,
    ) -> RecognizeStream:
        return _HiccupStream(
            stt=self, conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
        )


class _HiccupStream(RecognizeStream):
    async def _run(self) -> None:
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                request_id="req-1",
                alternatives=[SpeechData(language="en", text="hello", speaker_id="A")],
            )
        )
        await asyncio.sleep(0)
        raise APIConnectionError("brief connection hiccup")


class _FakeSession(rtc.EventEmitter[Any]):
    pass


def _adapter() -> tuple[MultiSpeakerAdapter, _RecordingSTT]:
    wrapped = _RecordingSTT()
    return MultiSpeakerAdapter(stt=wrapped), wrapped


def _silence() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
    )


def _metrics(label: str) -> STTMetrics:
    return STTMetrics(
        request_id="r1",
        timestamp=0.0,
        duration=0.0,
        label=label,
        audio_duration=1.0,
        streamed=True,
    )


@pytest.mark.parametrize("timed_endpoint", [False, True])
async def test_mixed_speaker_turn_ends_after_suppressed_final(timed_endpoint: bool) -> None:
    primary = SpeechData(
        language="en", text="primary speech", speaker_id="A", start_time=0.0, end_time=0.2
    )
    background = SpeechData(
        language="en", text="background speech", speaker_id="B", start_time=0.2, end_time=0.4
    )
    events = [
        SpeechEvent(type=SpeechEventType.START_OF_SPEECH),
        SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[primary]),
        SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[background]),
        SpeechEvent(
            type=SpeechEventType.END_OF_SPEECH,
            alternatives=[background] if timed_endpoint else [],
        ),
    ]

    class _TurnStream(RecognizeStream):
        async def _run(self) -> None:
            async for _ in self._input_ch:
                pass
            for event in events:
                self._event_ch.send_nowait(event)

    class _TurnSTT(_RecordingSTT):
        def stream(self, *, language=NOT_GIVEN, conn_options=_CONN):
            return _TurnStream(stt=self, conn_options=conn_options)

    adapter = MultiSpeakerAdapter(
        stt=_TurnSTT(),
        suppress_background_speaker=True,
        primary_detection_options=PrimarySpeakerDetectionOptions(min_rms_samples=1),
    )
    stream = adapter.stream(conn_options=_CONN)
    recognition = _make_recognition(vad=None, input_started_at=100.0, mode="stt")
    recognition._eou_wait_not_committed = 0
    try:
        for amplitude in [3000, 3000, 50, 50]:
            stream.push_frame(
                rtc.AudioFrame(
                    data=amplitude.to_bytes(2, "little", signed=True) * 1600,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=1600,
                )
            )
        stream.end_input()
        forwarded = [event async for event in stream]
        finals = [
            e.alternatives[0].text for e in forwarded if e.type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals == ["primary speech", ""]

        for event in forwarded:
            await recognition._on_stt_event(event)

        assert recognition._audio_transcript == "primary speech"
        assert recognition._user_turn_committed
        assert not recognition._speaking
        recognition._run_eou_detection.assert_called_once()
    finally:
        await stream.aclose()
        await adapter.aclose()


def test_adapter_forwards_remapped_timestamps_without_a_second_remap() -> None:
    adapter, wrapped = _adapter()
    assert wrapped.requires_timestamp_translation is True
    assert adapter.requires_timestamp_translation is False


def test_keyterms_reach_the_wrapped_stt() -> None:
    adapter, wrapped = _adapter()
    # the adapter advertises keyterms=True (inherited), so the base warn-and-skip never runs:
    # a dropped push here is silent
    assert adapter.capabilities.keyterms

    adapter._update_session_keyterms(["LiveKit", "WebRTC"])
    assert wrapped.pushed_keyterms == [["LiveKit", "WebRTC"]]


async def test_keyterm_detector_reaches_the_wrapped_stt() -> None:
    """The real framework path: AgentActivity binds the detector to the session's STT."""
    adapter, wrapped = _adapter()
    detector = KeytermDetector(static_keyterms=["Acme"])

    detector.start(_FakeSession(), stt=adapter)  # type: ignore[arg-type]
    assert wrapped.pushed_keyterms == [["Acme"]]

    detector.set_static_keyterms(["Acme", "Cartesia"])  # mid-call update
    assert wrapped.pushed_keyterms[-1] == ["Acme", "Cartesia"]

    await detector.aclose()


def test_conversation_items_reach_the_wrapped_stt() -> None:
    adapter, wrapped = _adapter()
    assert adapter.capabilities.chat_context

    chat_ctx = ChatContext.empty()
    item = chat_ctx.add_message(role="assistant", content="your room is booked")
    adapter._push_conversation_item(ConversationItemAddedEvent(item=item))

    assert [i.text_content for i in wrapped.chat_items] == ["your room is booked"]


def test_prewarm_reaches_the_wrapped_stt() -> None:
    adapter, wrapped = _adapter()
    adapter.prewarm()
    assert wrapped.prewarmed


def test_model_and_provider_come_from_the_wrapped_stt() -> None:
    adapter, wrapped = _adapter()
    # "unknown" is the STT base default; reporting it would mislabel telemetry
    assert (adapter.model, adapter.provider) == ("recording-model-v1", "recording-provider")


async def test_metrics_are_forwarded_and_detached_on_aclose() -> None:
    adapter, wrapped = _adapter()
    received: list[STTMetrics] = []
    adapter.on("metrics_collected", received.append)

    metrics = _metrics(wrapped.label)
    wrapped.emit("metrics_collected", metrics)
    assert received == [metrics]

    await adapter.aclose()  # detaches, so a later emit is dropped
    wrapped.emit("metrics_collected", metrics)
    assert received == [metrics]


async def test_recognize_reports_usage_once() -> None:
    """Re-emitting the wrapped STT's metrics must not add a second measurement."""
    adapter, wrapped = _adapter()
    received: list[STTMetrics] = []
    adapter.on("metrics_collected", received.append)

    await adapter.recognize([_silence()], conn_options=_CONN)

    assert len(received) == 1, f"one recognition, {len(received)} metrics events"
    # the wrapped STT's own measurement is the accurate one: it carries its label, and its
    # model/provider rather than the adapter's
    assert received[0].label == wrapped.label


async def test_stream_reports_usage_once() -> None:
    adapter, wrapped = _adapter()
    received: list[STTMetrics] = []
    adapter.on("metrics_collected", received.append)

    stream = adapter.stream(conn_options=_CONN)
    stream.push_frame(_silence())
    stream.end_input()
    async for _ in stream:
        pass
    await stream.aclose()

    assert len(received) == 1, f"one usage report, {len(received)} metrics events"
    assert received[0].label == wrapped.label
    assert received[0].audio_duration == 1.0


async def test_successful_transcript_forgives_earlier_hiccups() -> None:
    """Suppressing the adapter's own metrics must not drop the retry-count reset.

    ``RecognizeStream._main_task`` gives up once ``_num_retries`` exceeds ``max_retry``, and
    the base metrics monitor resets it on every final transcript. Without that reset,
    unrelated brief failures accumulate over a long call until recognition dies for good.
    """
    inner = _HiccupSTT()
    adapter = MultiSpeakerAdapter(stt=inner)
    conn = APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=5.0)
    stream = adapter.stream(conn_options=conn)

    transcripts: list[str] = []
    fatal: list[BaseException] = []

    async def _read() -> None:
        try:
            async for ev in stream:
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives[0].text:
                    transcripts.append(ev.alternatives[0].text)
        except Exception as e:  # noqa: BLE001 - recorded, asserted on below
            fatal.append(e)

    task = asyncio.create_task(_read())
    stream.push_frame(_silence())
    while len(transcripts) < 5 and not fatal:
        await asyncio.sleep(0.01)

    assert not fatal, f"recognition died after {len(transcripts)} good transcripts: {fatal}"
    assert stream._num_retries == 0

    await utils.aio.cancel_and_wait(task)
    with contextlib.suppress(Exception):
        await stream.aclose()


# -- primary speaker detection --

_DETECTOR_RATE = 1000
_DETECTOR_FRAME = 100  # 100ms at 1000Hz, matches frame_size_ms below


def _speech_data(speaker_id: str | None, start_time: float, end_time: float) -> SpeechData:
    return SpeechData(
        language="en", text="hello", start_time=start_time, end_time=end_time, speaker_id=speaker_id
    )


def _detector(**kwargs: Any) -> _PrimarySpeakerDetector:
    opts = PrimarySpeakerDetectionOptions(frame_size_ms=100, min_rms_samples=1)
    return _PrimarySpeakerDetector(primary_detection_options=opts, **kwargs)


def _push_speech(
    detector: _PrimarySpeakerDetector, speaker_id: str | None, amplitude: int, start_time: float
) -> None:
    frame = rtc.AudioFrame(
        data=amplitude.to_bytes(2, "little", signed=True) * _DETECTOR_FRAME,
        sample_rate=_DETECTOR_RATE,
        num_channels=1,
        samples_per_channel=_DETECTOR_FRAME,
    )
    detector.push_audio(frame)
    detector.push_audio(frame)
    detector._update_primary_speaker(_speech_data(speaker_id, start_time, start_time + 0.2))


@pytest.mark.parametrize("speaker_id", ["A", "B"])
@pytest.mark.parametrize("suppress_background", [False, True])
def test_timed_endpoint_is_forwarded_without_formatting(
    speaker_id: str, suppress_background: bool
) -> None:
    detector = _PrimarySpeakerDetector(
        suppress_background_speaker=suppress_background,
        primary_detection_options=PrimarySpeakerDetectionOptions(min_rms_samples=1),
        primary_format="primary: {text}",
        background_format="background: {text}",
    )
    _push_speech(detector, "A", amplitude=3000, start_time=0.0)
    _push_speech(detector, "B", amplitude=50, start_time=0.2)

    final = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[_speech_data(speaker_id, 0.2, 0.4)],
    )
    updated_final = detector.on_stt_event(final)
    if speaker_id == "B" and suppress_background:
        assert updated_final is None
    else:
        assert updated_final is final
        prefix = "primary" if speaker_id == "A" else "background"
        assert final.alternatives[0].text == f"{prefix}: hello"

    final_text = final.alternatives[0].text
    endpoint = SpeechEvent(type=SpeechEventType.END_OF_SPEECH, alternatives=final.alternatives)

    assert detector.on_stt_event(endpoint) is endpoint
    assert endpoint.alternatives[0].end_time == 0.4
    assert final.alternatives[0].text == final_text


def test_unattributed_segment_keeps_primary_speaker() -> None:
    """A segment the diarizer could not attribute does not unseat the primary speaker."""
    detector = _detector()

    _push_speech(detector, "A", amplitude=3000, start_time=0.0)
    _push_speech(detector, None, amplitude=3000, start_time=0.2)
    assert detector._primary_speaker == "A"

    # a much quieter speaker must still lose the loudness comparison
    _push_speech(detector, "B", amplitude=50, start_time=0.4)
    assert detector._primary_speaker == "A"


def test_detect_primary_disabled_keeps_primary_unset() -> None:
    detector = _detector(detect_primary_speaker=False)
    _push_speech(detector, "A", amplitude=3000, start_time=0.0)
    assert detector._primary_speaker is None


class _DiarizedStream(RecognizeStream):
    """Reads its input; the test sends the provider's events."""

    async def _run(self) -> None:
        async for _ in self._input_ch:
            pass


class _DiarizedSTT(_RecordingSTT):
    def __init__(self) -> None:
        super().__init__()
        self.streams: list[_DiarizedStream] = []

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _CONN,
    ) -> RecognizeStream:
        self.streams.append(_DiarizedStream(stt=self, conn_options=conn_options))
        return self.streams[-1]


@pytest.mark.parametrize("gap", [0.0, 5.0])
@pytest.mark.parametrize("start_time_offset", [None, 5.0])
async def test_primary_speaker_follows_the_audio_across_input_gaps(
    gap: float, start_time_offset: float | None
) -> None:
    """Detect speakers across input gaps while preserving forwarded timestamps."""
    provider = _DiarizedSTT()
    adapter = MultiSpeakerAdapter(stt=provider, suppress_background_speaker=True)
    stream = adapter.stream(conn_options=_CONN)
    if start_time_offset is not None:
        stream.start_time_offset = start_time_offset
    # the timeline begins `start_time_offset` ago, or with the first audio
    timeline_started_at = time.time() - (start_time_offset or 0.0)
    captured_at: list[float] = []  # timeline time of the start of each second of audio

    async def push_second(amplitude: int) -> None:
        captured_at.append(time.time() - timeline_started_at)
        for _ in range(10):
            await asyncio.sleep(0.1)  # a frame is pushed when its capture ends
            stream.push_frame(
                rtc.AudioFrame(
                    data=amplitude.to_bytes(2, "little", signed=True) * 1600,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=1600,
                )
            )
        await asyncio.sleep(gap)

    async def recognize(speaker_id: str, second: int) -> SpeechData:
        """The provider's final for speech in ``second`` of the audio it received."""
        provider.streams[0]._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[_speech_data(speaker_id, second + 0.1, second + 0.8)],
                speech_end_time=second + 0.8,
            )
        )
        event = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert event.type == SpeechEventType.FINAL_TRANSCRIPT
        if event.alternatives[0].text:
            # the provider's stream placed the end of speech on the wall clock, once
            assert event.speech_end_time == pytest.approx(
                timeline_started_at + captured_at[second] + 0.8, rel=0, abs=1e-3
            )
        else:
            assert event.speech_end_time is None  # the final of a suppressed speaker
        return event.alternatives[0]

    try:
        # the provider receives 4s of audio: speaker A twice, a louder B, a quiet C
        for amplitude in (1000, 1000, 3000, 100):
            await push_second(amplitude)

        for second in (0, 1):
            first = await recognize("A", second)
            assert first.is_primary_speaker
            assert first.start_time == pytest.approx(captured_at[second] + 0.1, abs=1e-3)
            assert first.end_time == pytest.approx(captured_at[second] + 0.8, abs=1e-3)

        louder = await recognize("B", 2)
        assert louder.is_primary_speaker
        assert louder.text == "hello"
        assert louder.start_time == pytest.approx(captured_at[2] + 0.1, abs=1e-3)
        assert louder.end_time == pytest.approx(captured_at[2] + 0.8, abs=1e-3)
        assert captured_at[2] == pytest.approx((start_time_offset or 0.0) + 2 * (1 + gap), abs=1e-3)

        # a speaker that is not louder stays in the background, and is suppressed
        quiet = await recognize("C", 3)
        assert quiet.text == ""
        assert (await recognize("A", 1)).text == ""
    finally:
        await stream.aclose()
        await adapter.aclose()


@pytest.mark.parametrize("step", [3600.0, -3600.0])
async def test_wrapped_stream_keeps_the_session_start_across_a_wall_clock_step(
    step: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapped stream continues the adapter stream's timeline from the same session start."""
    provider = _DiarizedSTT()
    adapter = MultiSpeakerAdapter(stt=provider)
    stream = adapter.stream(conn_options=_CONN)
    stream.start_time_offset = 30.0
    session_start = stream._input_clock.session_start
    assert session_start is not None

    wall_clock = time.time
    monkeypatch.setattr(time, "time", lambda: wall_clock() + step)
    try:
        frame = rtc.AudioFrame(
            data=b"\x00\x00" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
        )
        for _ in range(10):
            await asyncio.sleep(frame.duration)  # a frame is pushed when its capture ends
            stream.push_frame(frame)
        wrapped = provider.streams[0]
        while wrapped._input_clock.dequeued_duration < 1.0 - 1e-6:
            await asyncio.sleep(0)

        assert wrapped._input_clock.session_start is session_start
        wrapped._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[_speech_data("A", 0.2, 0.8)],
            )
        )
        placed = (await asyncio.wait_for(stream.__anext__(), timeout=1.0)).alternatives[0]
        assert (placed.start_time, placed.end_time) == pytest.approx((30.2, 30.8))
    finally:
        await stream.aclose()
