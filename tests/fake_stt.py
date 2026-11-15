from __future__ import annotations

import asyncio
import contextlib
import copy
import time
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel

from livekit import rtc
from livekit.agents import NOT_GIVEN, LanguageCode, NotGivenOr, utils
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.stt.clock import _SessionStart
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.audio import AudioBuffer


class RecognizeSentinel:
    pass


class FakeUserSpeech(BaseModel):
    type: Literal["user_speech"] = "user_speech"
    start_time: float
    end_time: float
    transcript: str  # empty string fires VAD SOS/EOS only — no STT events
    stt_delay: float
    final: bool = True

    def speed_up(self, factor: float) -> FakeUserSpeech:
        obj = copy.deepcopy(self)
        obj.start_time /= factor
        obj.end_time /= factor
        obj.stt_delay /= factor
        return obj


class FakeSTT(STT):
    def __init__(
        self,
        *,
        fake_exception: Exception | None = None,
        fake_transcript: str | None = None,
        fake_timeout: float | None = None,
        fake_user_speeches: list[FakeUserSpeech] | None = None,
        fake_require_audio: bool = False,
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False),
        )

        self._fake_exception = fake_exception
        self._fake_transcript = fake_transcript
        self._fake_timeout = fake_timeout
        self._fake_require_audio = fake_require_audio

        if fake_user_speeches is not None:
            fake_user_speeches = sorted(fake_user_speeches, key=lambda x: x.start_time)
            for prev, next in zip(fake_user_speeches[:-1], fake_user_speeches[1:], strict=False):
                if prev.end_time > next.start_time:
                    raise ValueError("fake user speeches overlap")
        self._fake_user_speeches = fake_user_speeches
        self._done_fut = asyncio.Future[None]()

        self._recognize_ch = utils.aio.Chan[RecognizeSentinel]()
        self._stream_ch = utils.aio.Chan[FakeRecognizeStream]()

    def update_options(
        self,
        *,
        fake_exception: NotGivenOr[Exception | None] = NOT_GIVEN,
        fake_transcript: NotGivenOr[str | None] = NOT_GIVEN,
        fake_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(fake_exception):
            self._fake_exception = fake_exception

        if utils.is_given(fake_transcript):
            self._fake_transcript = fake_transcript

        if utils.is_given(fake_timeout):
            self._fake_timeout = fake_timeout

    @property
    def recognize_ch(self) -> utils.aio.ChanReceiver[RecognizeSentinel]:
        return self._recognize_ch

    @property
    def stream_ch(self) -> utils.aio.ChanReceiver[FakeRecognizeStream]:
        return self._stream_ch

    @property
    def fake_user_speeches(self) -> list[FakeUserSpeech]:
        return self._fake_user_speeches

    @property
    def fake_user_speeches_done(self) -> asyncio.Future[None]:
        return self._done_fut

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        if self._fake_timeout is not None:
            await asyncio.sleep(self._fake_timeout)

        if self._fake_exception is not None:
            raise self._fake_exception

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(text=self._fake_transcript or "", language=LanguageCode(language or ""))
            ],
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        self._recognize_ch.send_nowait(RecognizeSentinel())
        return await super().recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> FakeRecognizeStream:
        stream = FakeRecognizeStream(
            stt=self,
            conn_options=conn_options,
        )
        self._stream_ch.send_nowait(stream)
        return stream


class FakeRecognizeStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: FakeSTT,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options)
        self._attempt = 0
        self._start_fut = asyncio.Future[float]()
        self._stt: FakeSTT = stt

    @property
    def attempt(self) -> int:
        return self._attempt

    def send_fake_transcript(self, transcript: str, is_final: bool = True) -> None:
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT
                if is_final
                else SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[SpeechData(text=transcript, language=LanguageCode(""))],
            )
        )

    async def _run(self) -> None:
        self._attempt += 1
        assert isinstance(self._stt, FakeSTT)

        if self._stt._fake_timeout is not None:
            await asyncio.sleep(self._stt._fake_timeout)

        if self._stt._fake_require_audio:
            got_audio = False
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    got_audio = True
                elif isinstance(data, self._FlushSentinel) and got_audio:
                    if self._stt._fake_transcript is not None:
                        self.send_fake_transcript(self._stt._fake_transcript)
                    got_audio = False
        else:
            if self._stt._fake_transcript is not None:
                self.send_fake_transcript(self._stt._fake_transcript)

            await self._fake_user_speech_task()

            async for _ in self._input_ch:
                pass

        if self._stt._fake_exception is not None:
            raise self._stt._fake_exception

    async def _fake_user_speech_task(self) -> None:
        if not self._stt._fake_user_speeches:
            return

        # start from when the first frame is pushed
        await self._input_ch.recv()
        start_time = time.perf_counter()

        def curr_time() -> float:
            return time.perf_counter() - start_time

        for fake_speech in self._stt._fake_user_speeches:
            if not fake_speech.transcript:
                # empty transcript: VAD SOS/EOS fire but STT emits nothing — still
                # advance the clock past this speech so subsequent speeches (and the
                # done signal) are timed correctly relative to it.
                final_transcript_time = fake_speech.end_time + fake_speech.stt_delay
                if curr_time() < final_transcript_time:
                    await asyncio.sleep(final_transcript_time - curr_time())
                continue
            interim_transcript_time = fake_speech.end_time + fake_speech.stt_delay * 0.5
            if curr_time() < interim_transcript_time:
                await asyncio.sleep(interim_transcript_time - curr_time())
            self.send_fake_transcript(" ".join(fake_speech.transcript.split()[:2]), is_final=False)

            final_transcript_time = fake_speech.end_time + fake_speech.stt_delay
            if curr_time() < final_transcript_time:
                await asyncio.sleep(final_transcript_time - curr_time())
            if not fake_speech.final:
                continue
            self.send_fake_transcript(fake_speech.transcript, is_final=True)

        with contextlib.suppress(asyncio.InvalidStateError):
            self._stt._done_fut.set_result(None)


class _RecordingEventChan(RecognizeStream._EventChan):
    def __init__(self, stream: RecognizeStream) -> None:
        super().__init__(translate_event_timestamps=stream._translate_event_timestamps)
        self.events: list[SpeechEvent] = []

    def send_nowait(self, value: SpeechEvent) -> None:
        super().send_nowait(value)
        self.events.append(self._queue[-1])


class _DetachedStream(RecognizeStream):
    async def _run(self) -> None:
        raise NotImplementedError


@contextlib.contextmanager
def current_event_loop() -> Iterator[None]:
    """Provide a temporary loop when a synchronous test has no current loop.

    The test only constructs channels; it does not await them on this loop.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    else:
        yield


def attach_provider_audio_channels(
    stream: object, *, start_time_offset: float = 0.0, input_started_at: float | None = None
) -> list[SpeechEvent]:
    """Attach production channels and clocks to a stream built without ``__init__``.

    A stand-in object that is not a ``RecognizeStream`` gets the channels and the
    input clock of a detached stream. Return the translated events in enqueue order.
    """
    owner = stream if isinstance(stream, RecognizeStream) else object.__new__(_DetachedStream)
    with current_event_loop():
        owner._init_channels(translate_timestamps=True)
        event_ch = _RecordingEventChan(owner)
    owner._event_ch = event_ch
    owner._input_clock.set_session_timeline(start_time_offset)
    if input_started_at is not None:
        # move the wall-clock start only; the declared offset stays
        owner._input_clock._session_start = _SessionStart.from_wall_timestamp(input_started_at)
    if owner is not stream:
        stream._input_ch = owner._input_ch  # type: ignore[attr-defined]
        stream._event_ch = event_ch  # type: ignore[attr-defined]
        stream._input_clock = owner._input_clock  # type: ignore[attr-defined]
    return event_ch.events


def queued_events(stream: RecognizeStream) -> list[SpeechEvent]:
    """Return queued events that no consumer has read."""
    return list(stream._event_ch._queue)
