from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Literal, Union

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
    """
    Manages multiple TTS instances, providing a fallback mechanism to ensure continuous TTS service.
    """

    def __init__(
        self,
        tts: list[TTS],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_tts: int = 1,  # only retry once by default
        retry_interval: float = 5,
        no_fallback_after_audio_duration: float | None = 3.0,
        sample_rate: int | None = None,
    ) -> None:
        """
        Initialize a FallbackAdapter that manages multiple TTS instances.

        Args:
            tts (list[TTS]): A list of TTS instances to use for fallback.
            attempt_timeout (float, optional): Timeout for each synthesis attempt in seconds. Defaults to 10.0.
            max_retry_per_tts (int, optional): Maximum number of retries per TTS instance. Defaults to 1.
            no_fallback_after_audio_duration (float | None, optional): Disables fallback after this duration of audio is synthesized. Defaults to 3.0.
                This is used to prevent unnaturally resaying the same text when the first TTS
                instance fails.
            sample_rate (int | None, optional): Desired sample rate for the synthesized audio. If None, uses the maximum sample rate among the TTS instances.

        Raises:
            ValueError: If less than one TTS instance is provided.
            ValueError: If TTS instances have different numbers of channels.
        """

        if len(tts) < 1:
            raise ValueError("at least one TTS instance must be provided.")

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

        self._tts_instances = tts
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_tts = max_retry_per_tts
        self._retry_interval = retry_interval
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
        self, *, tts: FallbackAdapter, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._fallback_adapter = tts

    async def _try_synthesize(
        self, *, tts: TTS, recovering: bool = False
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        try:
            audio_duration = 0.0
            async with tts.synthesize(
                self._input_text,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=self._fallback_adapter._max_retry_per_tts,
                    timeout=self._fallback_adapter._attempt_timeout,
                    retry_interval=self._fallback_adapter._retry_interval,
                ),
            ) as stream:
                while True:
                    try:
                        audio = await asyncio.wait_for(
                            stream.__anext__(),
                            self._fallback_adapter._attempt_timeout
                            if audio_duration == 0.0
                            else None,
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

        tts_status = self._tts._status[self._tts._tts_instances.index(tts)]
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

        start_time = time.time()

        all_failed = all(not tts_status.available for tts_status in self._tts._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        for i, tts in enumerate(self._tts._tts_instances):
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

                    if self._tts._no_fallback_after_audio_duration is not None:
                        if (
                            audio_duration
                            >= self._tts._no_fallback_after_audio_duration
                        ):
                            logger.warning(
                                f"{tts.label} already synthesized {audio_duration}s of audio, ignoring fallback"
                            )
                            return

            self._try_recovery(tts)

        raise APIConnectionError(
            "all TTSs failed (%s) after %s seconds"
            % (
                [tts.label for tts in self._tts._tts_instances],
                time.time() - start_time,
            )
        )


class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: FallbackAdapter,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._fallback_adapter = tts

        self._total_segments: list[list[str]] = []
        self._pending_segments_chunks: list[list[str]] = []
        self._current_segment_text: list[str] = []

    async def _try_synthesize(
        self,
        *,
        tts: TTS,
        input_ch: aio.ChanReceiver[str | SynthesizeStream._FlushSentinel],
        conn_options: APIConnectOptions,
        recovering: bool = False,
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        stream = tts.stream(conn_options=conn_options)
        input_sent_fut = asyncio.Future()  # type: ignore

        @utils.log_exceptions(logger=logger)
        async def _input_task() -> None:
            try:
                segment = ""
                async for data in input_ch:
                    if isinstance(data, str):
                        segment += data
                        stream.push_text(data)
                    elif isinstance(data, self._FlushSentinel):
                        # start the timeout on flush
                        if segment:
                            segment = ""
                            with contextlib.suppress(asyncio.InvalidStateError):
                                input_sent_fut.set_result(True)

                        stream.flush()
            finally:
                with contextlib.suppress(RuntimeError):
                    stream.end_input()

                with contextlib.suppress(asyncio.InvalidStateError):
                    input_sent_fut.set_result(False)

        input_task = asyncio.create_task(_input_task())
        next_audio_task: asyncio.Future[SynthesizedAudio] | None = None

        try:
            audio_duration = 0.0
            async with stream:
                while True:
                    if next_audio_task is None or next_audio_task.done():
                        next_audio_task = asyncio.ensure_future(stream.__anext__())

                    try:
                        if not input_sent_fut.done():
                            await asyncio.wait(
                                [input_sent_fut, next_audio_task],
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            if not next_audio_task.done():
                                continue

                            audio = next_audio_task.result()
                        else:
                            audio = await asyncio.wait_for(
                                next_audio_task, self._fallback_adapter._attempt_timeout
                            )

                        audio_duration += audio.frame.duration
                        if audio.is_final:
                            input_sent_fut = asyncio.Future()
                            audio_duration = 0.0

                        yield audio
                    except StopAsyncIteration:
                        break

            if (
                audio_duration == 0.0
                and input_sent_fut.done()
                and input_sent_fut.result()
            ):
                raise APIConnectionError("no audio received")

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
                    f"{tts.label} recovery unexpected error",
                    extra={"streamed": True},
                )
                raise

            logger.exception(
                f"{tts.label} unexpected error, switching to next TTS",
                extra={"streamed": True},
            )
            raise
        finally:
            if next_audio_task is not None:
                await utils.aio.gracefully_cancel(next_audio_task)

            await utils.aio.gracefully_cancel(input_task)

    async def _run(self) -> None:
        start_time = time.time()

        all_failed = all(
            not tts_status.available for tts_status in self._fallback_adapter._status
        )
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        new_input_ch: aio.Chan[str | SynthesizeStream._FlushSentinel] | None = None

        async def _forward_input_task():
            nonlocal new_input_ch

            async for data in self._input_ch:
                if new_input_ch:
                    new_input_ch.send_nowait(data)

                if isinstance(data, str) and data:
                    self._current_segment_text.append(data)

                elif (
                    isinstance(data, self._FlushSentinel) and self._current_segment_text
                ):
                    self._total_segments.append(self._current_segment_text)
                    self._pending_segments_chunks.append(self._current_segment_text)
                    self._current_segment_text = []

            if new_input_ch:
                new_input_ch.close()

        input_task = asyncio.create_task(_forward_input_task())

        try:
            for i, tts in enumerate(self._fallback_adapter._tts_instances):
                tts_status = self._fallback_adapter._status[i]
                if tts_status.available or all_failed:
                    audio_duration = 0.0
                    try:
                        new_input_ch = aio.Chan[
                            Union[str, SynthesizeStream._FlushSentinel]
                        ]()

                        for text in self._pending_segments_chunks:
                            for chunk in text:
                                new_input_ch.send_nowait(chunk)

                            new_input_ch.send_nowait(self._FlushSentinel())

                        for chunk in self._current_segment_text:
                            new_input_ch.send_nowait(chunk)

                        if input_task.done():
                            new_input_ch.close()

                        last_segment_id: str | None = None
                        resampler = tts_status.resampler

                        async for synthesized_audio in self._try_synthesize(
                            tts=tts,
                            input_ch=new_input_ch,
                            conn_options=dataclasses.replace(
                                self._conn_options,
                                max_retry=self._fallback_adapter._max_retry_per_tts,
                                timeout=self._fallback_adapter._attempt_timeout,
                                retry_interval=self._fallback_adapter._retry_interval,
                            ),
                            recovering=False,
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
                            ) and self._pending_segments_chunks:
                                audio_duration = 0.0
                                self._pending_segments_chunks.pop(0)

                            last_segment_id = synthesized_audio.segment_id

                        return
                    except Exception:
                        if tts_status.available:
                            tts_status.available = False
                            self._tts.emit(
                                "tts_availability_changed",
                                AvailabilityChangedEvent(tts=tts, available=False),
                            )

                        if (
                            self._fallback_adapter._no_fallback_after_audio_duration
                            is not None
                        ):
                            if (
                                audio_duration
                                >= self._fallback_adapter._no_fallback_after_audio_duration
                                and self._pending_segments_chunks
                            ):
                                logger.warning(
                                    f"{tts.label} already synthesized {audio_duration}s of audio, ignoring the current segment for the tts fallback"
                                )
                                return

                self._try_recovery(tts)

            raise APIConnectionError(
                "all TTSs failed (%s) after %s seconds"
                % (
                    [tts.label for tts in self._fallback_adapter._tts_instances],
                    time.time() - start_time,
                )
            )
        finally:
            await utils.aio.gracefully_cancel(input_task)

    def _try_recovery(self, tts: TTS) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        retry_segments = [self._current_segment_text.copy()]
        if self._total_segments:
            retry_segments.insert(0, self._total_segments[-1])

        tts_status = self._tts._status[self._tts._tts_instances.index(tts)]
        if tts_status.recovering_task is None or tts_status.recovering_task.done():

            async def _recover_tts_task(tts: TTS) -> None:
                try:
                    input_ch = aio.Chan[Union[str, SynthesizeStream._FlushSentinel]]()
                    for segment in retry_segments:
                        for t in segment:
                            input_ch.send_nowait(t)

                        input_ch.send_nowait(self._FlushSentinel())

                    input_ch.close()

                    async for _ in self._try_synthesize(
                        tts=tts,
                        input_ch=input_ch,
                        recovering=True,
                        conn_options=dataclasses.replace(
                            self._conn_options,
                            max_retry=0,
                            timeout=self._fallback_adapter._attempt_timeout,
                            retry_interval=self._fallback_adapter._retry_interval,
                        ),
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
