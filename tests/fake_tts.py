from __future__ import annotations

import asyncio

from livekit import rtc
from livekit.agents import NOT_GIVEN, NotGivenOr, tts, utils
from livekit.agents.tts import (
    TTS,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


class FakeTTS(TTS):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        num_channels: int = 1,
        fake_timeout: float | None = None,
        fake_audio_duration: float | None = None,
        fake_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._fake_timeout = fake_timeout
        self._fake_audio_duration = fake_audio_duration
        self._fake_exception = fake_exception

        self._synthesize_ch = utils.aio.Chan[FakeChunkedStream]()
        self._stream_ch = utils.aio.Chan[FakeSynthesizeStream]()

    def update_options(
        self,
        *,
        fake_timeout: NotGivenOr[float | None] = NOT_GIVEN,
        fake_audio_duration: NotGivenOr[float | None] = NOT_GIVEN,
        fake_exception: NotGivenOr[Exception | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(fake_timeout):
            self._fake_timeout = fake_timeout

        if utils.is_given(fake_audio_duration):
            self._fake_audio_duration = fake_audio_duration

        if utils.is_given(fake_exception):
            self._fake_exception = fake_exception

    @property
    def synthesize_ch(self) -> utils.aio.ChanReceiver[FakeChunkedStream]:
        return self._synthesize_ch

    @property
    def stream_ch(self) -> utils.aio.ChanReceiver[FakeSynthesizeStream]:
        return self._stream_ch

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        stream = FakeChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._synthesize_ch.send_nowait(stream)
        return stream

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        stream = FakeSynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )
        self._stream_ch.send_nowait(stream)
        return stream


class FakeChunkedStream(ChunkedStream):
    def __init__(self, *, tts: FakeTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    async def _run(self, output_emitter: tts.SynthesizedAudioEmitter):
        self._attempt += 1

        assert isinstance(self._tts, FakeTTS)

        output_emitter.start(
            request_id=utils.shortuuid("fake_tts_"),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            format="audio/pcm",
        )

        if self._tts._fake_timeout is not None:
            await asyncio.sleep(self._tts._fake_timeout)

        if self._tts._fake_audio_duration is not None:
            pushed_samples = 0
            max_samples = (
                int(self._tts.sample_rate * self._tts._fake_audio_duration + 0.5)
                * self._tts.num_channels
            )
            while pushed_samples < max_samples:
                num_samples = min(self._tts.sample_rate // 100, max_samples - pushed_samples)
                output_emitter.push(b"\x00\x00" * num_samples)
                pushed_samples += num_samples

        if self._tts._fake_exception is not None:
            raise self._tts._fake_exception

        output_emitter.flush()


class FakeSynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    async def _run(self) -> None:
        self._attempt += 1

        assert isinstance(self._tts, FakeTTS)

        if self._tts._fake_timeout is not None:
            await asyncio.sleep(self._tts._fake_timeout)

        has_data = False
        async for data in self._input_ch:
            if isinstance(data, str):
                has_data = True
                continue
            elif isinstance(data, SynthesizeStream._FlushSentinel) and not has_data:
                continue

            has_data = False

            if self._tts._fake_audio_duration is None:
                continue

            request_id = utils.shortuuid("fake_tts_")
            segment_id = utils.shortuuid("fake_segment_")

            pushed_samples = 0
            max_samples = (
                int(self._tts.sample_rate * self._tts._fake_audio_duration + 0.5)
                * self._tts.num_channels
            )
            while pushed_samples < max_samples:
                num_samples = min(self._tts.sample_rate // 100, max_samples - pushed_samples)
                self._event_ch.send_nowait(
                    SynthesizedAudio(
                        request_id=request_id,
                        segment_id=segment_id,
                        is_final=(pushed_samples + num_samples >= max_samples),
                        frame=rtc.AudioFrame(
                            data=b"\x00\x00" * num_samples,
                            samples_per_channel=num_samples // self._tts.num_channels,
                            sample_rate=self._tts.sample_rate,
                            num_channels=self._tts.num_channels,
                        ),
                    )
                )
                pushed_samples += num_samples

        if self._tts._fake_exception is not None:
            raise self._tts._fake_exception
