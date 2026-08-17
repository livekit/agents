"""MultiSpeakerAdapter delegation contract.

The adapter wraps another STT and inherits its ``capabilities`` wholesale, so the framework
treats it as if it were the wrapped recognizer: it pushes keyterms and conversation items to
it, prewarms it, and reads ``model``/``provider`` off it for metrics. Every one of those has
to reach the wrapped STT, otherwise the capability the adapter advertises is a no-op.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.keyterm_detection import KeytermDetector

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
