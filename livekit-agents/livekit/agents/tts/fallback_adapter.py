import asyncio
from typing import Literal, Union, AsyncGenerator
from livekit import rtc
import dataclasses

from .tts import (
    TTS,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
)

from .. import utils
from ..utils import aio
from .._exceptions import APIConnectionError, APIError

from ..log import logger

from dataclasses import dataclass


# don't retry when using the fallback adapter
DEFAULT_FALLBACK_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class _TTSStatus:
    available: bool
    recovering_task: asyncio.Task | None
    resampler: rtc.AudioResampler | None


@dataclass
class AvailabilityChangedEvent:
    tts: TTS
    available: bool


class FallbackAdapter(
    TTS[Literal["tts_availability_changed"]],
):
    def __init__(
        self,
        tts: list[TTS],
        *,
        timeout: float = 10.0,
        no_fallback_after_audio_duration: float = 3.0,
        sample_rate: int | None = None,
    ) -> None:
        if len(tts) < 1:
            raise ValueError("At least one TTS instance must be provided.")

        if len(set(t.num_channels for t in tts)) != 1:
            raise ValueError("all TTS must have the same number of channels")

        if sample_rate is None:
            sample_rate = max(t.sample_rate for t in tts)

        num_channels = tts[0].num_channels

        super().__init__(
            capabilities=TTSCapabilities(
                streaming=all(t.capabilities.streaming for t in tts),
            ),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._wrapped_tts = tts
        self._timeout = timeout
        self._no_fallback_after_audio_duration = no_fallback_after_audio_duration

        self._status: list[_TTSStatus] = []
        for t in tts:
            resampler = None
            if sample_rate != t.sample_rate:
                logger.info(
                    f"resampling {t.label} from {t.sample_rate}Hz to {sample_rate}Hz"
                )
                resampler = rtc.AudioResampler(
                    input_rate=t.sample_rate, output_rate=sample_rate
                )

            self._status.append(
                _TTSStatus(available=True, recovering_task=None, resampler=resampler)
            )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> "FallbackChunkedStream":
        return FallbackChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> "FallbackSynthesizeStream":
        raise NotImplementedError("not implemented")

    async def aclose(self) -> None:
        pass


class FallbackChunkedStream(ChunkedStream):
    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _try_synthesize(
        self, *, tts: TTS, recovering: bool = False
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        assert isinstance(self._tts, FallbackAdapter)

        # disable retry of a specific TTS when using the fallback adapter
        req_conn_options = dataclasses.replace(self._conn_options, max_retry=0)

        try:
            audio_duration = 0.0
            async with tts.synthesize(
                self._input_text, conn_options=req_conn_options
            ) as stream:
                while True:
                    try:
                        audio = await asyncio.wait_for(
                            stream.__anext__(),
                            self._tts._timeout if audio_duration == 0.0 else None,
                        )

                        audio_duration += audio.frame.duration
                        yield audio
                    except StopAsyncIteration:
                        break

            if audio_duration == 0.0:
                raise APIConnectionError("no audio received")

        except asyncio.TimeoutError:
            if recovering:
                logger.warning(f"{tts.label} recovery timed out")
                raise

            logger.warning(f"{tts.label} timed out, switching to next TTS")
            raise
        except APIError as e:
            if recovering:
                logger.warning(f"{tts.label} recovery failed", exc_info=e)
                raise

            logger.warning(
                f"{tts.label} failed, switching to next TTS",
                exc_info=e,
            )
            raise
        except Exception:
            if recovering:
                logger.exception(f"{tts.label} recovery unexpected error")
                raise

            logger.exception(f"{tts.label} unexpected error, switching to next TTS")
            raise

    async def _run(self) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        all_failed = all(not tts_status.available for tts_status in self._tts._status)
        if all_failed:
            logger.warning("all TTSs are unavailable, retrying..")

        for i, tts in enumerate(self._tts._wrapped_tts):
            tts_status = self._tts._status[i]
            if tts_status.available or all_failed:
                audio_duration = 0.0
                try:
                    request_id: str | None = None
                    async for synthesized_audio in self._try_synthesize(
                        tts=tts, recovering=False
                    ):
                        audio_duration += synthesized_audio.frame.duration
                        request_id = synthesized_audio.request_id

                        if tts_status.resampler is not None:
                            for resampled_frame in tts_status.resampler.push(
                                synthesized_audio.frame
                            ):
                                self._event_ch.send_nowait(
                                    SynthesizedAudio(
                                        frame=resampled_frame,
                                        request_id=synthesized_audio.request_id,
                                    )
                                )
                        else:
                            self._event_ch.send_nowait(synthesized_audio)

                    if tts_status.resampler is not None and request_id is not None:
                        for resampled_frame in tts_status.resampler.flush():
                            self._event_ch.send_nowait(
                                SynthesizedAudio(
                                    frame=resampled_frame,
                                    request_id=request_id,
                                )
                            )

                    return
                except Exception:
                    if tts_status.available:
                        tts_status.available = False
                        self._tts.emit(
                            "tts_availability_changed",
                            AvailabilityChangedEvent(tts=tts, available=False),
                        )

                    if audio_duration >= self._tts._no_fallback_after_audio_duration:
                        logger.warning(
                            f"{tts.label} already synthesized {audio_duration}s of audio, ignoring fallback"
                        )
                        return

            if tts_status.recovering_task is None:

                async def _recover_tts_task(tts: TTS) -> None:
                    try:
                        async for _ in self._try_synthesize(tts=tts, recovering=True):
                            pass

                        tts_status.available = True
                        logger.info(f"tts.FallbackAdapter, {tts.label} recovered")
                        self._tts.emit(
                            "tts_availability_changed",
                            AvailabilityChangedEvent(tts=tts, available=True),
                        )
                    except Exception:
                        return

                tts_status.recovering_task = asyncio.create_task(_recover_tts_task(tts))

                def _on_done(_: asyncio.Task) -> None:
                    tts_status.recovering_task = None

                tts_status.recovering_task.add_done_callback(_on_done)

        raise APIConnectionError(
            "all TTSs failed (%s)" % [tts.label for tts in self._tts._wrapped_tts]
        )


class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(self, tts: TTS) -> None:
        super().__init__(tts)
