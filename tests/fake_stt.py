from __future__ import annotations

import asyncio
import copy
import time

from pydantic import BaseModel

from livekit.agents import NOT_GIVEN, NotGivenOr, utils
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.audio import AudioBuffer


class RecognizeSentinel:
    pass


class FakeUserSpeech(BaseModel):
    start_time: float
    end_time: float
    transcript: str
    stt_delay: float

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
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False),
        )

        self._fake_exception = fake_exception
        self._fake_transcript = fake_transcript
        self._fake_timeout = fake_timeout

        if fake_user_speeches is not None:
            fake_user_speeches = sorted(fake_user_speeches, key=lambda x: x.start_time)
            for prev, next in zip(fake_user_speeches[:-1], fake_user_speeches[1:]):
                if prev.end_time > next.start_time:
                    raise ValueError("fake user speeches overlap")
        self._fake_user_speeches = fake_user_speeches

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
            alternatives=[SpeechData(text=self._fake_transcript or "", language=language or "")],
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

    def send_fake_transcript(self, transcript: str) -> None:
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text=transcript, language="")],
            )
        )

    async def _run(self) -> None:
        self._attempt += 1
        assert isinstance(self._stt, FakeSTT)

        if self._stt._fake_timeout is not None:
            await asyncio.sleep(self._stt._fake_timeout)

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
        start_time = time.time()

        for fake_speech in self._stt._fake_user_speeches:
            curr_time = time.time() - start_time
            next_transcript_time = fake_speech.end_time + fake_speech.stt_delay
            if curr_time < next_transcript_time:
                await asyncio.sleep(next_transcript_time - curr_time)

            self.send_fake_transcript(fake_speech.transcript)
