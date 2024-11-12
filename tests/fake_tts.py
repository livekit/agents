import asyncio

from livekit import rtc
from livekit.agents.tts import (
    TTS,
    TTSCapabilities,
    ChunkedStream,
    SynthesizeStream,
    SynthesizedAudio,
)
from livekit.agents import utils, NotGivenOr, NOT_GIVEN
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


class FakeTTS(TTS):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        num_channels: int = 1,
        fake_connection_time: float | None = None,
        fake_audio_duration: float | None = None,
        fake_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._fake_connection_time = fake_connection_time
        self._fake_audio_duration = fake_audio_duration
        self._fake_exception = fake_exception

        self._synthesize_ch = utils.aio.Chan[FakeChunkedStream]()

    def update_options(
        self,
        *,
        fake_connection_time: NotGivenOr[float | None] = NOT_GIVEN,
        fake_audio_duration: NotGivenOr[float | None] = NOT_GIVEN,
        fake_exception: NotGivenOr[Exception | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(fake_connection_time):
            self._fake_connection_time = fake_connection_time

        if utils.is_given(fake_audio_duration):
            self._fake_audio_duration = fake_audio_duration

        if utils.is_given(fake_exception):
            self._fake_exception = fake_exception

    @property
    def synthesize_ch(self) -> utils.aio.ChanReceiver["FakeChunkedStream"]:
        return self._synthesize_ch

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "FakeChunkedStream":
        stream = FakeChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._synthesize_ch.send_nowait(stream)
        return stream

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        raise NotImplementedError("not implemented")


class FakeChunkedStream(ChunkedStream):
    def __init__(
        self, *, tts: FakeTTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _run(self) -> None:
        assert isinstance(self._tts, FakeTTS)

        request_id = utils.shortuuid("fake_tts_")

        if self._tts._fake_connection_time is not None:
            await asyncio.sleep(self._tts._fake_connection_time)

        if self._tts._fake_audio_duration is not None:
            pushed_samples = 0
            max_samples = (
                int(self._tts.sample_rate * self._tts._fake_audio_duration + 0.5)
                * self._tts.num_channels
            )
            while pushed_samples < max_samples:
                num_samples = min(
                    self._tts.sample_rate // 100, max_samples - pushed_samples
                )
                self._event_ch.send_nowait(
                    SynthesizedAudio(
                        request_id=request_id,
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
