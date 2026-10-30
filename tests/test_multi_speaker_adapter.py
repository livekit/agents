"""MultiSpeakerAdapter delegation contract.

The adapter wraps another STT and inherits its ``capabilities`` wholesale, so the framework
treats it as if it were the wrapped recognizer: it pushes keyterms and conversation items to
it, prewarms it, and reads ``model``/``provider`` off it for metrics. Every one of those has
to reach the wrapped STT, otherwise the capability the adapter advertises is a no-op.
"""

from __future__ import annotations

import array
import asyncio
import contextlib
import math
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
from livekit.agents.stt.stt import RecognitionUsage
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr, TimedString
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.keyterm_detection import KeytermDetector
from livekit.plugins.xai.stt import _words_to_speech_data

pytestmark = pytest.mark.unit

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
        self.received_conn_options: list[APIConnectOptions] = []

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
        self.received_conn_options.append(conn_options)
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


def _xai_speech_data(speaker: int, start: float, end: float, text: str = "hello") -> SpeechData:
    """Build the exact ``SpeechData`` shape the xAI plugin emits.

    ``_words_to_speech_data`` copies the provider's word clocks verbatim; it never adds
    ``RecognizeStream.start_time_offset``. This is the convention that broke the earlier
    "seed the child's offset" fix, so it is pinned here with the real function.
    """
    return _words_to_speech_data(
        [{"text": text, "start": start, "end": end, "speaker": speaker}],
        text,
        language="en",
        enable_diarization=True,
    )


class _ProviderClockSTT(_RecordingSTT):
    """Diarizing STT whose legs report on the provider's own clock.

    ``convention="aligned"`` mirrors plugins such as deepgram/assemblyai/aws, which add
    ``start_time_offset`` to their word clocks before emitting ``SpeechData``.
    ``convention="xai"`` mirrors the xAI plugin, which ignores that field.

    ``fail_first_leg_after`` makes the first leg die -- immediately at zero, or after
    consuming that many seconds of audio -- so the failure escapes to
    ``MultiSpeakerAdapterWrapper._run`` and the wrapper rebuilds the leg. ``max_retry=0``
    on the leg mirrors a plugin that has already spent its own retry budget.
    """

    def __init__(
        self,
        *,
        convention: str,
        fail_first_leg_after: float | None = None,
        audio_needed: float = 1.0,
    ) -> None:
        assert convention in ("aligned", "xai")
        super().__init__()
        self._convention = convention
        self._fail_first_leg_after = fail_first_leg_after
        self._audio_needed = audio_needed
        self.legs: list[_ProviderClockStream] = []

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _CONN,
    ) -> RecognizeStream:
        leg = _ProviderClockStream(
            stt=self,
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0),
            convention=self._convention,
            fail_after=self._fail_first_leg_after if not self.legs else None,
            audio_needed=self._audio_needed,
        )
        self.legs.append(leg)
        return leg


class _ProviderClockStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: _ProviderClockSTT,
        conn_options: APIConnectOptions,
        convention: str,
        fail_after: float | None,
        audio_needed: float,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._convention = convention
        self._fail_after = fail_after
        self._audio_needed = audio_needed
        self.offset_at_run: float | None = None

    async def _run(self) -> None:
        self.offset_at_run = self.start_time_offset
        if self._fail_after == 0.0:
            raise APIConnectionError("stream died")

        heard = 0.0
        async for item in self._input_ch:
            if not isinstance(item, rtc.AudioFrame):
                continue
            heard += item.duration
            if self._fail_after is not None and heard > self._fail_after:
                raise APIConnectionError("stream died")
            if heard < self._audio_needed:
                continue
            # two speakers over the same audio: the second is no louder, so it has to
            # stay background (and be suppressed when the adapter is asked to)
            self._emit(speaker=0, start=0.0, end=0.9)
            self._emit(speaker=1, start=0.0, end=0.9)
            return

    def _emit(self, *, speaker: int, start: float, end: float) -> None:
        if self._convention == "xai":
            sd = _xai_speech_data(speaker, start, end)
        else:
            offset = self.start_time_offset  # what an aligned plugin adds itself
            sd = SpeechData(
                language="en",
                text="hello",
                speaker_id=f"speaker-{speaker}",
                start_time=start + offset,
                end_time=end + offset,
                words=[
                    TimedString(
                        "hello",
                        start_time=start + offset,
                        end_time=end + offset,
                        start_time_offset=offset,
                    )
                ],
            )
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                request_id="req-provider-clock",
                alternatives=[sd],
            )
        )


class _ChildRetrySTT(_RecordingSTT):
    """Diarizing STT whose first leg drops after ``_FAIL_AFTER`` seconds of audio.

    The leg is created with whatever ``conn_options`` the adapter hands it: a nonzero
    ``max_retry`` lets the base ``RecognizeStream._main_task`` retry the same leg
    internally, while ``max_retry=0`` sends the failure back to the adapter wrapper, which
    builds a new leg instead.
    """

    _FAIL_AFTER = 0.6
    _SEGMENT_END = 1.2

    def __init__(self, *, convention: str) -> None:
        assert convention in ("aligned", "xai")
        super().__init__()
        self._convention = convention
        self.legs: list[_ChildRetryStream] = []

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = _CONN,
    ) -> RecognizeStream:
        leg = _ChildRetryStream(
            stt=self,
            conn_options=conn_options,
            convention=self._convention,
            fail_first=not self.legs,
        )
        self.received_conn_options.append(conn_options)
        self.legs.append(leg)
        return leg


class _ChildRetryStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: _ChildRetrySTT,
        conn_options: APIConnectOptions,
        convention: str,
        fail_first: bool,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._convention = convention
        self._fail_first = fail_first
        self.attempts = 0

    async def _run(self) -> None:
        self.attempts += 1
        heard = 0.0  # the provider's clock starts at zero on every attempt

        if self._fail_first and self.attempts == 1:
            async for item in self._input_ch:
                if not isinstance(item, rtc.AudioFrame):
                    continue
                heard += item.duration
                if heard >= _ChildRetrySTT._FAIL_AFTER:
                    raise APIConnectionError("provider session dropped")
            return

        async for item in self._input_ch:
            if not isinstance(item, rtc.AudioFrame):
                continue
            heard += item.duration
            if heard < _ChildRetrySTT._SEGMENT_END:
                continue
            # the quiet window first, the loud one second. on the 1.8 s of audio the
            # detector has forwarded these are [0.6, 1.1) and [1.3, 1.8): the audio the
            # dropped attempt did not consume, then the tone that follows it
            self._emit(speaker=0, start=0.0, end=0.5)
            self._emit(speaker=1, start=0.7, end=1.2)
            return

    def _emit(self, *, speaker: int, start: float, end: float) -> None:
        if self._convention == "xai":
            sd = _xai_speech_data(speaker, start, end)
        else:
            offset = self.start_time_offset  # what an aligned plugin adds itself
            sd = SpeechData(
                language="en",
                text="hello",
                speaker_id=f"speaker-{speaker}",
                start_time=start + offset,
                end_time=end + offset,
                words=[
                    TimedString(
                        "hello",
                        start_time=start + offset,
                        end_time=end + offset,
                        start_time_offset=offset,
                    )
                ],
            )
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                request_id="req-child-retry",
                alternatives=[sd],
            )
        )


async def _collect_finals(
    stream: RecognizeStream, finals: list[SpeechData], done: asyncio.Event, *, expected: int = 1
) -> None:
    async for ev in stream:
        if ev.type == SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
            finals.append(ev.alternatives[0])
            if len(finals) >= expected:
                done.set()


async def _push_audio(stream: RecognizeStream, frames: int) -> None:
    for _ in range(frames):
        stream.push_frame(_silence())
        await asyncio.sleep(0)


async def _wait_for_leg(stt: _ProviderClockSTT, index: int, *, timeout: float = 5.0) -> None:
    """Wait until leg ``index`` has started, failing instead of hanging if it never does."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while len(stt.legs) <= index or stt.legs[index].offset_at_run is None:
        if loop.time() > deadline:
            raise AssertionError(
                f"leg {index} did not start within {timeout}s (created: {len(stt.legs)})"
            )
        await asyncio.sleep(0.01)


async def _wait_for_failed_leg_retried(stt: _ChildRetrySTT, *, timeout: float = 5.0) -> None:
    """Wait until the dropped first attempt has a successor.

    When the adapter owns retries the successor is a second leg. A wrapped stream that was
    still allowed to retry internally would instead run a second attempt on the first leg.
    Both are accepted here so the test can push the follow-up audio at the right moment;
    the assertions then pin down which one actually happened.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(stt.legs) >= 2 or (stt.legs and stt.legs[0].attempts >= 2):
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"the failed attempt was not retried within {timeout}s")


class _FakeSession(rtc.EventEmitter[Any]):
    pass


def _adapter() -> tuple[MultiSpeakerAdapter, _RecordingSTT]:
    wrapped = _RecordingSTT()
    return MultiSpeakerAdapter(stt=wrapped), wrapped


def _silence() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
    )


def _tone() -> rtc.AudioFrame:
    """A 220 Hz tone frame of the same 10 ms duration as :func:`_silence`."""
    samples_per_channel = 160
    samples = array.array(
        "h",
        (int(6000 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(samples_per_channel)),
    )
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples_per_channel,
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


def test_wrapped_stt_exposed() -> None:
    adapter, wrapped = _adapter()
    assert adapter.wrapped_stt is wrapped


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
        await asyncio.sleep(0)

    assert not fatal, f"recognition died after {len(transcripts)} good transcripts: {fatal}"
    assert stream._num_retries == 0

    await utils.aio.cancel_and_wait(task)
    with contextlib.suppress(Exception):
        await stream.aclose()


@pytest.mark.parametrize("convention", ["aligned", "xai"])
async def test_nonzero_offset_keeps_primary_and_suppresses_background(convention: str) -> None:
    """Provider timestamps are leg-local, so the adapter must place them on the input timeline.

    ``agent.stt_node`` seeds the wrapper with the time already elapsed in the session.
    Whichever convention the wrapped plugin follows -- adding ``start_time_offset`` to its
    word clocks (deepgram/assemblyai/aws) or ignoring it (xAI) -- the segment must still be
    found in the detector's forwarded-audio buffer, so the primary speaker is selected and
    background speech is suppressed.
    """
    inner = _ProviderClockSTT(convention=convention)
    adapter = MultiSpeakerAdapter(
        stt=inner, suppress_background_speaker=True, primary_format="[P] {text}"
    )
    stream = adapter.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.start_time_offset = 30.0  # what stt_node seeds when recognition starts late

    finals: list[SpeechData] = []
    done = asyncio.Event()
    task = asyncio.create_task(_collect_finals(stream, finals, done, expected=2))
    try:
        await _push_audio(stream, 100)  # 1s; the reported segment covers the first 0.9s
        await asyncio.wait_for(done.wait(), 5.0)

        primary, background = finals[0], finals[1]
        assert primary.is_primary_speaker is True
        assert primary.text == "[P] hello"
        assert primary.start_time >= 30.0
        assert primary.end_time >= 30.9
        assert primary.words is not None
        assert all(w.start_time >= 30.0 for w in primary.words)
        if convention == "aligned":
            # plugins that set the per-word offset must keep it consistent with the
            # word times after the adapter shifts them
            assert all(w.start_time_offset >= 30.0 for w in primary.words)
        # the quieter speaker never becomes primary and its text is dropped entirely
        assert background.text == ""
    finally:
        await utils.aio.cancel_and_wait(task)
        with contextlib.suppress(Exception):
            await stream.aclose()


@pytest.mark.parametrize("convention", ["aligned", "xai"])
async def test_wrapper_retry_keeps_forwarded_timestamps_on_input_timeline(
    convention: str,
) -> None:
    """A wrapper-level retry must not restart the forwarded clock at zero.

    ``MultiSpeakerAdapterWrapper._run`` builds a fresh leg on every attempt, and the
    adapter has to carry the elapsed offset itself instead of requiring the plugin to
    apply ``start_time_offset`` (xAI does not). The retried leg is therefore created with
    an untouched offset, while its events still land on the input timeline and primary
    selection keeps working.
    """
    inner = _ProviderClockSTT(convention=convention, fail_first_leg_after=0.0)
    adapter = MultiSpeakerAdapter(stt=inner, primary_format="[P] {text}")
    stream = adapter.stream(
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0)
    )
    stream.start_time_offset = 30.0

    finals: list[SpeechData] = []
    done = asyncio.Event()
    task = asyncio.create_task(_collect_finals(stream, finals, done))
    try:
        await _wait_for_leg(inner, 1)
        # the adapter owns the timeline; it must not depend on plugins honoring the field
        # (every leg starts at zero, up to the base class's sub-millisecond bookkeeping)
        assert [leg.offset_at_run for leg in inner.legs] == [
            pytest.approx(0.0, abs=0.05),
            pytest.approx(0.0, abs=0.05),
        ]

        await _push_audio(stream, 100)
        await asyncio.wait_for(done.wait(), 5.0)

        primary = finals[0]
        assert primary.is_primary_speaker is True
        assert primary.text == "[P] hello"
        assert primary.start_time >= 30.0
        assert primary.end_time >= 30.9
    finally:
        await utils.aio.cancel_and_wait(task)
        with contextlib.suppress(Exception):
            await stream.aclose()


@pytest.mark.parametrize("convention", ["aligned", "xai"])
async def test_wrapper_retry_maps_audio_consumed_before_the_gap(convention: str) -> None:
    """Audio consumed before a reconnect still counts for the detector's RMS buffer.

    The first leg forwards half a second of audio and then dies. The retried leg starts
    its provider clock at zero, but on the detector's forwarded-audio clock it starts at
    the audio already consumed -- not at the wall-clock offset, which also covers the
    retry gap. Primary selection has to use the window that actually holds the segment.
    """
    inner = _ProviderClockSTT(convention=convention, fail_first_leg_after=0.45)
    adapter = MultiSpeakerAdapter(stt=inner, primary_format="[P] {text}")
    stream = adapter.stream(
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0)
    )
    stream.start_time_offset = 30.0

    finals: list[SpeechData] = []
    done = asyncio.Event()
    task = asyncio.create_task(_collect_finals(stream, finals, done))
    try:
        await _push_audio(stream, 50)  # 0.5s: consumed by the leg that then dies
        await _wait_for_leg(inner, 1)
        await _push_audio(stream, 100)  # the retried leg transcribes this second
        await asyncio.wait_for(done.wait(), 5.0)

        primary = finals[0]
        assert primary.is_primary_speaker is True
        assert primary.text == "[P] hello"
        assert primary.start_time >= 30.0
        assert primary.end_time >= 30.9
    finally:
        await utils.aio.cancel_and_wait(task)
        with contextlib.suppress(Exception):
            await stream.aclose()


@pytest.mark.parametrize("convention", ["aligned", "xai"])
async def test_zero_offset_leaves_provider_timestamps_untouched(convention: str) -> None:
    """With no elapsed offset the adapter must forward the provider's clock unchanged."""
    inner = _ProviderClockSTT(convention=convention)
    adapter = MultiSpeakerAdapter(stt=inner)
    stream = adapter.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))

    finals: list[SpeechData] = []
    done = asyncio.Event()
    task = asyncio.create_task(_collect_finals(stream, finals, done))
    try:
        await _push_audio(stream, 100)
        await asyncio.wait_for(done.wait(), 5.0)

        first = finals[0]
        # with no elapsed offset the provider clock is forwarded as-is; only the adapter's
        # own scheduling delay (before the first leg starts) may show up
        assert first.start_time == pytest.approx(0.0, abs=0.05)
        assert first.end_time == pytest.approx(0.9, abs=0.05)
        assert first.is_primary_speaker is True
        assert first.text == "hello"
        assert first.words is not None
        assert first.words[0].start_time == pytest.approx(0.0, abs=0.05)
        assert first.words[0].end_time == pytest.approx(0.9, abs=0.05)
    finally:
        await utils.aio.cancel_and_wait(task)
        with contextlib.suppress(Exception):
            await stream.aclose()


async def test_wrapped_stream_gets_no_retry_budget() -> None:
    """The wrapper owns retries, so the wrapped provider stream must not retry itself.

    A provider stream retrying internally advances its own ``start_time_offset`` while the
    adapter keeps the offsets it computed for the first attempt. Diarization plugins
    disagree about whether that offset is already folded into ``SpeechData``
    (deepgram/assemblyai/aws add it, xAI ignores it), so the adapter cannot translate both
    conventions after the fact. Everything else about the connect options still passes
    through unchanged.
    """
    inner = _RecordingSTT()
    adapter = MultiSpeakerAdapter(stt=inner)
    wrapper = adapter.stream(
        conn_options=APIConnectOptions(
            max_retry=2, retry_interval=0.5, timeout=7.0, liveness_timeout=11.0
        )
    )
    try:
        for _ in range(100):
            if inner.received_conn_options:
                break
            await asyncio.sleep(0)
        assert len(inner.received_conn_options) == 1, "the wrapped stream was never created"
        leg_conn = inner.received_conn_options[0]
        # the wrapper keeps the budget it was given and takes over the retries itself
        assert wrapper._conn_options.max_retry == 2
        assert leg_conn.max_retry == 0
        assert leg_conn.timeout == 7.0
        assert leg_conn.retry_interval == 0.5
        assert leg_conn.liveness_timeout == 11.0
    finally:
        with contextlib.suppress(Exception):
            await wrapper.aclose()


@pytest.mark.parametrize("convention", ["aligned", "xai"])
async def test_child_stream_retry_keeps_the_detector_on_its_audio(convention: str) -> None:
    """A provider-level retry must not leave the detector reading a stale clock.

    The first leg consumes 0.6 s of silence and then drops. The retried leg transcribes a
    quiet window followed by a loud one, so the louder speaker has to take over as primary
    (and, with suppression on, its text must survive). A wrapped stream that retried
    internally would move its provider clock under the adapter: xAI-shaped timestamps
    restart at zero while the detector's buffer has already advanced past the dropped
    audio, and aligned timestamps add an offset the adapter's fixed translation does not
    know about. Either way the loud window is read as silence and the takeover does not
    happen.
    """
    inner = _ChildRetrySTT(convention=convention)
    adapter = MultiSpeakerAdapter(
        stt=inner, suppress_background_speaker=True, primary_format="[P] {text}"
    )
    stream = adapter.stream(
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0)
    )
    stream.start_time_offset = 30.0  # what stt_node seeds in a running session

    finals: list[SpeechData] = []
    done = asyncio.Event()
    task = asyncio.create_task(_collect_finals(stream, finals, done, expected=2))
    try:
        for _ in range(60):  # 0.6 s of silence, consumed before the drop
            stream.push_frame(_silence())
            await asyncio.sleep(0)

        await _wait_for_failed_leg_retried(inner)

        # 1.2 s: [0, 0.6) silence, then [0.6, 1.2) tone. together with the audio above the
        # detector holds 1.8 s, and the two provider segments map onto [0.6, 1.1) silence
        # and [1.3, 1.8) tone
        for i in range(120):
            stream.push_frame(_silence() if i < 60 else _tone())
            await asyncio.sleep(0)

        await asyncio.wait_for(done.wait(), 5.0)

        quiet, loud = finals[0], finals[1]
        assert quiet.is_primary_speaker is True
        assert quiet.text == "[P] hello"
        # the loud segment must be looked up on the tone so that it beats the quiet
        # primary; with a stale clock it reads the silent window and is suppressed instead
        assert loud.is_primary_speaker is True
        assert loud.text == "[P] hello"
        # forwarded timestamps stay on the input timeline despite the retry
        assert quiet.start_time >= 30.0
        assert loud.start_time >= 30.0

        # the retry belongs to the wrapper: one attempt per leg, and no leg retries on its
        # own -- that is what keeps the provider clock fixed for a leg's whole lifetime
        assert [opts.max_retry for opts in inner.received_conn_options] == [0, 0]
        assert [leg.attempts for leg in inner.legs] == [1, 1]
    finally:
        await utils.aio.cancel_and_wait(task)
        with contextlib.suppress(Exception):
            await stream.aclose()
