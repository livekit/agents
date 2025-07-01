from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Union

from livekit import rtc

from .. import utils
from .._exceptions import APIConnectionError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio
from .tts import (
    TTS,
    AudioEmitter,
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
    recovering_task: asyncio.Task[None] | None
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
        max_retry_per_tts: int = 2,
        sample_rate: int | None = None,
    ) -> None:
        """
        Initialize a FallbackAdapter that manages multiple TTS instances.

        Args:
            tts (list[TTS]): A list of TTS instances to use for fallback.
            max_retry_per_tts (int, optional): Maximum number of retries per TTS instance. Defaults to 2.
            sample_rate (int | None, optional): Desired sample rate for the synthesized audio. If None, uses the maximum sample rate among the TTS instances.

        Raises:
            ValueError: If less than one TTS instance is provided.
            ValueError: If TTS instances have different numbers of channels.
        """  # noqa: E501

        if len(tts) < 1:
            raise ValueError("at least one TTS instance must be provided.")

        if len({t.num_channels for t in tts}) != 1:
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
        self._max_retry_per_tts = max_retry_per_tts

        self._status: list[_TTSStatus] = []
        for t in tts:
            resampler = None
            if sample_rate != t.sample_rate:
                logger.info(f"resampling {t.label} from {t.sample_rate}Hz to {sample_rate}Hz")
                resampler = rtc.AudioResampler(input_rate=t.sample_rate, output_rate=sample_rate)

            self._status.append(
                _TTSStatus(available=True, recovering_task=None, resampler=resampler)
            )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS
    ) -> FallbackChunkedStream:
        return FallbackChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS
    ) -> FallbackSynthesizeStream:
        return FallbackSynthesizeStream(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        if self._tts_instances:
            self._tts_instances[0].prewarm()

    async def aclose(self) -> None:
        for tts_status in self._status:
            if tts_status.recovering_task is not None:
                await aio.cancel_and_wait(tts_status.recovering_task)


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
            async with tts.synthesize(
                self._input_text,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=self._fallback_adapter._max_retry_per_tts,
                    timeout=self._conn_options.timeout,
                    retry_interval=self._conn_options.retry_interval,
                ),
            ) as stream:
                async for audio in stream:
                    yield audio

        except Exception as e:
            if recovering:
                logger.warning(
                    f"{tts.label} recovery failed", extra={"streamed": False}, exc_info=e
                )
                raise

            logger.warning(
                f"{tts.label} error, switching to next TTS",
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
                except Exception:  # exceptions already logged inside _try_synthesize
                    return

            tts_status.recovering_task = asyncio.create_task(_recover_tts_task(tts))

    async def _run(self, output_emitter: AudioEmitter) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        start_time = time.time()

        all_failed = all(not tts_status.available for tts_status in self._tts._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )

        for i, tts in enumerate(self._tts._tts_instances):
            tts_status = self._tts._status[i]
            if tts_status.available or all_failed:
                try:
                    resampler = tts_status.resampler
                    async for synthesized_audio in self._try_synthesize(tts=tts, recovering=False):
                        if resampler is not None:
                            for rf in resampler.push(synthesized_audio.frame):
                                output_emitter.push(rf.data.tobytes())
                        else:
                            output_emitter.push(synthesized_audio.frame.data.tobytes())

                    if resampler is not None:
                        for rf in resampler.flush():
                            output_emitter.push(rf.data.tobytes())

                    return
                except Exception:  # exceptions already logged inside _try_synthesize
                    if tts_status.available:
                        tts_status.available = False
                        self._tts.emit(
                            "tts_availability_changed",
                            AvailabilityChangedEvent(tts=tts, available=False),
                        )

                    if output_emitter.pushed_duration() > 0.0:
                        logger.warning(
                            f"{tts.label} already synthesized of audio, ignoring fallback"
                        )
                        return

            self._try_recovery(tts)

        raise APIConnectionError(
            f"all TTSs failed ({[tts.label for tts in self._tts._tts_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
        )


class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(self, *, tts: FallbackAdapter, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._fallback_adapter = tts
        self._pushed_tokens: list[str] = []

    async def _try_synthesize(
        self,
        *,
        tts: TTS,
        input_ch: aio.ChanReceiver[str | SynthesizeStream._FlushSentinel],
        conn_options: APIConnectOptions,
        recovering: bool = False,
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        stream = tts.stream(conn_options=conn_options)

        @utils.log_exceptions(logger=logger)
        async def _forward_input_task() -> None:
            try:
                async for data in input_ch:
                    if isinstance(data, str):
                        stream.push_text(data)
                    elif isinstance(data, self._FlushSentinel):
                        stream.flush()
            finally:
                stream.end_input()

        input_task = asyncio.create_task(_forward_input_task())

        try:
            async with stream:
                async for audio in stream:
                    yield audio
        except Exception as e:
            if recovering:
                logger.warning(
                    f"{tts.label} recovery failed",
                    extra={"streamed": True},
                    exc_info=e,
                )
                raise

            logger.exception(
                f"{tts.label} error, switching to next TTS",
                extra={"streamed": True},
            )
            raise
        finally:
            await utils.aio.cancel_and_wait(input_task)

    async def _run(self, output_emitter: AudioEmitter) -> None:
        start_time = time.time()

        all_failed = all(not tts_status.available for tts_status in self._fallback_adapter._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        new_input_ch: aio.Chan[str | SynthesizeStream._FlushSentinel] | None = None
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._fallback_adapter.sample_rate,
            num_channels=self._fallback_adapter.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())

        async def _forward_input_task() -> None:
            nonlocal new_input_ch

            async for data in self._input_ch:
                if new_input_ch:
                    new_input_ch.send_nowait(data)

                if isinstance(data, str) and data:
                    self._pushed_tokens.append(data)

            if new_input_ch:
                new_input_ch.close()

        input_task = asyncio.create_task(_forward_input_task())

        try:
            for i, tts in enumerate(self._fallback_adapter._tts_instances):
                tts_status = self._fallback_adapter._status[i]
                if tts_status.available or all_failed:
                    try:
                        new_input_ch = aio.Chan[Union[str, SynthesizeStream._FlushSentinel]]()

                        for text in self._pushed_tokens:
                            new_input_ch.send_nowait(text)

                        if input_task.done():
                            new_input_ch.close()

                        resampler = tts_status.resampler
                        async for synthesized_audio in self._try_synthesize(
                            tts=tts,
                            input_ch=new_input_ch,
                            conn_options=dataclasses.replace(
                                self._conn_options,
                                max_retry=self._fallback_adapter._max_retry_per_tts,
                                timeout=self._conn_options.timeout,
                                retry_interval=self._conn_options.retry_interval,
                            ),
                            recovering=False,
                        ):
                            if resampler is not None:
                                for resampled_frame in resampler.push(synthesized_audio.frame):
                                    output_emitter.push(resampled_frame.data.tobytes())

                                if synthesized_audio.is_final:
                                    for resampled_frame in resampler.flush():
                                        output_emitter.push(resampled_frame.data.tobytes())
                            else:
                                output_emitter.push(synthesized_audio.frame.data.tobytes())

                        return
                    except Exception:
                        if tts_status.available:
                            tts_status.available = False
                            self._tts.emit(
                                "tts_availability_changed",
                                AvailabilityChangedEvent(tts=tts, available=False),
                            )

                        if output_emitter.pushed_duration() > 0.0:
                            logger.warning(
                                f"{tts.label} already synthesized of audio, ignoring the current segment for the tts fallback"  # noqa: E501
                            )
                            return

                self._try_recovery(tts)

            raise APIConnectionError(
                f"all TTSs failed ({[tts.label for tts in self._fallback_adapter._tts_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
            )
        finally:
            await utils.aio.cancel_and_wait(input_task)

    def _try_recovery(self, tts: TTS) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        retry_text = self._pushed_tokens.copy()
        if not retry_text:
            return

        tts_status = self._tts._status[self._tts._tts_instances.index(tts)]
        if tts_status.recovering_task is None or tts_status.recovering_task.done():

            async def _recover_tts_task(tts: TTS) -> None:
                try:
                    input_ch = aio.Chan[Union[str, SynthesizeStream._FlushSentinel]]()
                    for t in retry_text:
                        input_ch.send_nowait(t)

                    input_ch.close()

                    async for _ in self._try_synthesize(
                        tts=tts,
                        input_ch=input_ch,
                        recovering=True,
                        conn_options=dataclasses.replace(
                            self._conn_options,
                            max_retry=0,
                            timeout=self._conn_options.timeout,
                            retry_interval=self._conn_options.retry_interval,
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
