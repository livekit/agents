import asyncio
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import AsyncGenerator, Literal

from livekit import rtc

from .. import utils
from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..utils import aio
from .tts import (
    DEFAULT_API_CONNECT_OPTIONS,
    TTS,
    APIConnectOptions,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)

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
        return FallbackSynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        for tts_status in self._status:
            if tts_status.recovering_task is not None:
                await aio.gracefully_cancel(tts_status.recovering_task)


class FallbackChunkedStream(ChunkedStream):
    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _try_synthesize(
        self, *, tts: TTS, recovering: bool = False
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        assert isinstance(self._tts, FallbackAdapter)

        try:
            audio_duration = 0.0
            async with tts.synthesize(
                self._input_text,
                conn_options=dataclasses.replace(self._conn_options, max_retry=0),
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
                logger.warning(
                    f"{tts.label} recovery timed out", extra={"streamed": False}
                )
                raise

            logger.warning(
                f"{tts.label} timed out, switching to next TTS",
                extra={"streamed": False},
            )

            raise
        except APIError as e:
            if recovering:
                logger.warning(
                    f"{tts.label} recovery failed",
                    exc_info=e,
                    extra={"streamed": False},
                )
                raise

            logger.warning(
                f"{tts.label} failed, switching to next TTS",
                exc_info=e,
                extra={"streamed": False},
            )
            raise
        except Exception:
            if recovering:
                logger.exception(
                    f"{tts.label} recovery unexpected error", extra={"streamed": False}
                )
                raise

            logger.exception(
                f"{tts.label} unexpected error, switching to next TTS",
                extra={"streamed": False},
            )
            raise

    def _try_recovery(self, tts: TTS) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        tts_status = self._tts._status[self._tts._wrapped_tts.index(tts)]
        if tts_status.recovering_task is None or tts_status.recovering_task.done():

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

    async def _run(self) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        all_failed = all(not tts_status.available for tts_status in self._tts._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        for i, tts in enumerate(self._tts._wrapped_tts):
            tts_status = self._tts._status[i]
            if tts_status.available or all_failed:
                audio_duration = 0.0
                try:
                    request_id: str | None = None
                    resampler = tts_status.resampler
                    async for synthesized_audio in self._try_synthesize(
                        tts=tts, recovering=False
                    ):
                        audio_duration += synthesized_audio.frame.duration
                        request_id = synthesized_audio.request_id

                        if resampler is not None:
                            for rf in resampler.push(synthesized_audio.frame):
                                self._event_ch.send_nowait(
                                    SynthesizedAudio(
                                        frame=rf,
                                        request_id=synthesized_audio.request_id,
                                    )
                                )

                            continue

                        self._event_ch.send_nowait(synthesized_audio)

                    if resampler is not None and request_id is not None:
                        for rf in resampler.flush():
                            self._event_ch.send_nowait(
                                SynthesizedAudio(
                                    frame=rf,
                                    request_id=request_id,
                                )
                            )

                    return
                except Exception:  # exceptions already logged inside _try_synthesize
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

            self._try_recovery(tts)

        raise APIConnectionError(
            "all TTSs failed (%s)" % [tts.label for tts in self._tts._wrapped_tts]
        )


class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)

        self._total_segments: list[list[str]] = []
        self._fallback_pending_texts: list[list[str]] = []
        self._fallback_text: list[str] = []

    async def _try_synthesize(
        self,
        tts: TTS,
        input_ch: aio.ChanReceiver[str | SynthesizeStream._FlushSentinel],
        recovering: bool = False,
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        assert isinstance(self._tts, FallbackAdapter)

        req_conn_options = dataclasses.replace(self._conn_options, max_retry=0)
        stream = tts.stream(conn_options=req_conn_options)

        start_timeout_fut = asyncio.Future()

        async def _input_task() -> None:
            nonlocal start_timeout_fut

            try:
                async for data in input_ch:
                    if isinstance(data, str):
                        with contextlib.suppress(asyncio.InvalidStateError):
                            start_timeout_fut.set_result(None)

                        stream.push_text(data)
                    elif isinstance(data, self._FlushSentinel):
                        stream.flush()

            finally:
                stream.end_input()
                with contextlib.suppress(asyncio.InvalidStateError):
                    start_timeout_fut.set_result(None)

        input_task = asyncio.create_task(_input_task())

        try:
            audio_duration = 0.0

            await start_timeout_fut

            async with stream:
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

        except asyncio.TimeoutError:
            if recovering:
                logger.warning(
                    f"{tts.label} recovery timed out", extra={"streamed": True}
                )
                raise

            logger.warning(
                f"{tts.label} timed out, switching to next TTS",
                extra={"streamed": True},
            )
            raise
        except APIError as e:
            if recovering:
                logger.warning(
                    f"{tts.label} recovery failed", exc_info=e, extra={"streamed": True}
                )
                raise

            logger.warning(
                f"{tts.label} failed, switching to next TTS",
                exc_info=e,
                extra={"streamed": True},
            )
            raise
        except Exception:
            if recovering:
                logger.exception(
                    f"{tts.label} recovery unexpected error", extra={"streamed": True}
                )
                raise

            logger.exception(
                f"{tts.label} unexpected error, switching to next TTS",
                extra={"streamed": True},
            )
            raise
        finally:
            await utils.aio.gracefully_cancel(input_task)

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        all_failed = all(not tts_status.available for tts_status in self._tts._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        new_input_ch: aio.Chan[str | SynthesizeStream._FlushSentinel] | None = None

        async def _forward_input_task():
            nonlocal new_input_ch

            async for data in self._input_ch:
                if new_input_ch:
                    new_input_ch.send_nowait(data)

                if isinstance(data, str) and data:
                    self._fallback_text.append(data)

                elif isinstance(data, self._FlushSentinel) and self._fallback_text:
                    self._total_segments.append(self._fallback_text)
                    self._fallback_pending_texts.append(self._fallback_text)
                    self._fallback_text = []

            if new_input_ch:
                new_input_ch.close()

        input_task = asyncio.create_task(_forward_input_task())

        try:
            for i, tts in enumerate(self._tts._wrapped_tts):
                tts_status = self._tts._status[i]
                if tts_status.available or all_failed:
                    audio_duration = 0.0
                    try:
                        new_input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()

                        for text in self._fallback_pending_texts:
                            for t in text:
                                new_input_ch.send_nowait(t)

                            new_input_ch.send_nowait(self._FlushSentinel())

                        for t in self._fallback_text:
                            new_input_ch.send_nowait(t)

                        if self._input_ch.closed:
                            new_input_ch.close()

                        last_segment_id: str | None = None
                        resampler = tts_status.resampler

                        async for synthesized_audio in self._try_synthesize(
                            tts=tts, input_ch=new_input_ch, recovering=False
                        ):
                            audio_duration += synthesized_audio.frame.duration

                            if resampler is not None:
                                for resampled_frame in resampler.push(
                                    synthesized_audio.frame
                                ):
                                    self._event_ch.send_nowait(
                                        dataclasses.replace(
                                            synthesized_audio, frame=resampled_frame
                                        )
                                    )

                                if synthesized_audio.is_final:
                                    for resampled_frame in resampler.flush():
                                        self._event_ch.send_nowait(
                                            dataclasses.replace(
                                                synthesized_audio, frame=resampled_frame
                                            )
                                        )
                            else:
                                self._event_ch.send_nowait(synthesized_audio)

                            if (
                                synthesized_audio.is_final
                                or (
                                    last_segment_id is not None
                                    and synthesized_audio.segment_id != last_segment_id
                                )
                            ) and self._fallback_pending_texts:
                                audio_duration = 0.0

                            last_segment_id = synthesized_audio.segment_id

                        return
                    except (
                        Exception
                    ):  # exceptions already logged inside _try_synthesize
                        if tts_status.available:
                            tts_status.available = False
                            self._tts.emit(
                                "tts_availability_changed",
                                AvailabilityChangedEvent(tts=tts, available=False),
                            )

                        if (
                            audio_duration
                            >= self._tts._no_fallback_after_audio_duration
                            and self._fallback_pending_texts
                        ):
                            logger.warning(
                                f"{tts.label} already synthesized {audio_duration}s of audio, ignoring the current segment for the tts fallback"
                            )
                            return

                retry_segments: list[list[str]] = [self._fallback_text.copy()]
                if self._total_segments:
                    retry_segments.insert(0, self._total_segments[-1])

                self._try_recovery(tts, retry_segments)

            raise APIConnectionError(
                "all TTSs failed (%s)" % [tts.label for tts in self._tts._wrapped_tts]
            )
        finally:
            await utils.aio.gracefully_cancel(input_task)

    def _try_recovery(self, tts: TTS, segments: list[list[str]]) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        tts_status = self._tts._status[self._tts._wrapped_tts.index(tts)]
        if tts_status.recovering_task is None or tts_status.recovering_task.done():

            async def _recover_tts_task(tts: TTS) -> None:
                try:
                    input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
                    for segment in segments:
                        for t in segment:
                            input_ch.send_nowait(t)

                        input_ch.send_nowait(self._FlushSentinel())

                    input_ch.close()

                    async for _ in self._try_synthesize(
                        tts=tts, input_ch=input_ch, recovering=True
                    ):
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
