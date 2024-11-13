import asyncio
from typing import Literal, Union, AsyncGenerator
from livekit import rtc
import dataclasses
import contextlib

from pydantic.type_adapter import P

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
        return FallbackSynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )

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
                    # exceptions already logged inside _try_synthesize

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
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)

        self._fallback_pending_texts: list[str] = []
        self._fallback_text = ""

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
                    if isinstance(data, str) and data:
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
            last_segment_id: str | None = None
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

    async def _run(self) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        all_failed = all(not tts_status.available for tts_status in self._tts._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        async def _input_task():
            async for data in self._input_ch:
                if isinstance(data, str) and data:
                    self._fallback_text += data
                elif isinstance(data, self._FlushSentinel) and self._fallback_text:
                    self._fallback_pending_texts.append(self._fallback_text)
                    self._fallback_text = ""

        for i, tts in enumerate(self._tts._wrapped_tts):
            tts_status = self._tts._status[i]
            last_segment: str | None = None
            if tts_status.available or all_failed:
                audio_duration = 0.0
                try:
                    input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
                    for text in self._fallback_pending_texts:
                        # send the pending texts to the input channel
                        # (segments that were not fully synthesized from the previous failed TTS)
                        input_ch.send_nowait(text)
                        input_ch.send_nowait(self._FlushSentinel())

                    input_ch.send_nowait(self._fallback_text)
                    last_segment_id: str | None = None

                    async for synthesized_audio in self._try_synthesize(
                        tts=tts, input_ch=input_ch, recovering=False
                    ):
                        audio_duration += synthesized_audio.frame.duration

                        if tts_status.resampler is not None:
                            for resampled_frame in tts_status.resampler.push(
                                synthesized_audio.frame
                            ):
                                frame = dataclasses.replace(
                                    synthesized_audio, frame=resampled_frame
                                )
                                self._event_ch.send_nowait(frame)

                            if synthesized_audio.is_final:
                                for resampled_frame in tts_status.resampler.flush():
                                    frame = dataclasses.replace(
                                        synthesized_audio, frame=resampled_frame
                                    )
                                    self._event_ch.send_nowait(frame)
                        else:
                            self._event_ch.send_nowait(synthesized_audio)

                        if (
                            synthesized_audio.is_final
                            or (
                                last_segment_id is not None
                                and synthesized_audio.segment_id != last_segment_id
                            )
                        ) and self._fallback_pending_texts:
                            last_segment = self._fallback_pending_texts.pop(0)
                            audio_duration = 0.0

                        last_segment_id = synthesized_audio.segment_id

                    return
                except Exception:
                    # exceptions already logged inside _try_synthesize
                    if tts_status.available:
                        tts_status.available = False
                        self._tts.emit(
                            "tts_availability_changed",
                            AvailabilityChangedEvent(tts=tts, available=False),
                        )

                    if (
                        audio_duration >= self._tts._no_fallback_after_audio_duration
                        and self._fallback_pending_texts
                    ):
                        last_segment = self._fallback_pending_texts.pop(0)
                        logger.warning(
                            f"{tts.label} already synthesized {audio_duration}s of audio, ignoring the current segment for the tts fallback"
                        )
                        return

            if tts_status.recovering_task is None and last_segment is not None:

                async def _recover_tts_task(tts: TTS) -> None:
                    assert last_segment is not None

                    try:
                        input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
                        input_ch.send_nowait(last_segment)
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

                def _on_done(_: asyncio.Task) -> None:
                    tts_status.recovering_task = None

                tts_status.recovering_task.add_done_callback(_on_done)

        raise APIConnectionError(
            "all TTSs failed (%s)" % [tts.label for tts in self._tts._wrapped_tts]
        )
