from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from dataclasses import dataclass
from typing import Literal

from livekit import rtc
from livekit.agents.utils.audio import AudioBuffer

from .. import utils
from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

# don't retry when using the fallback adapter
DEFAULT_FALLBACK_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class AvailabilityChangedEvent:
    stt: STT
    available: bool


@dataclass
class _STTStatus:
    available: bool
    recovering_synthesize_task: asyncio.Task | None
    recovering_stream_task: asyncio.Task | None


class FallbackAdapter(
    STT[Literal["stt_availability_changed"]],
):
    def __init__(
        self,
        stt: list[STT],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        if len(stt) < 1:
            raise ValueError("At least one STT instance must be provided.")

        super().__init__(
            capabilities=STTCapabilities(
                streaming=all(t.capabilities.streaming for t in stt),
                interim_results=all(t.capabilities.interim_results for t in stt),
            )
        )

        self._stt_instances = stt
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_stt = max_retry_per_stt
        self._retry_interval = retry_interval

        self._status: list[_STTStatus] = [
            _STTStatus(
                available=True,
                recovering_synthesize_task=None,
                recovering_stream_task=None,
            )
            for _ in self._stt_instances
        ]

    async def _try_recognize(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: str | None = None,
        conn_options: APIConnectOptions,
        recovering: bool = False,
    ) -> SpeechEvent:
        try:
            return await stt.recognize(
                buffer,
                language=language,
                conn_options=dataclasses.replace(
                    conn_options,
                    max_retry=self._max_retry_per_stt,
                    timeout=self._attempt_timeout,
                    retry_interval=self._retry_interval,
                ),
            )
        except asyncio.TimeoutError:
            if recovering:
                logger.warning(
                    f"{stt.label} recovery timed out", extra={"streamed": False}
                )
                raise

            logger.warning(
                f"{stt.label} timed out, switching to next STT",
                extra={"streamed": False},
            )

            raise
        except APIError as e:
            if recovering:
                logger.warning(
                    f"{stt.label} recovery failed",
                    exc_info=e,
                    extra={"streamed": False},
                )
                raise

            logger.warning(
                f"{stt.label} failed, switching to next STT",
                exc_info=e,
                extra={"streamed": False},
            )
            raise
        except Exception:
            if recovering:
                logger.exception(
                    f"{stt.label} recovery unexpected error", extra={"streamed": False}
                )
                raise

            logger.exception(
                f"{stt.label} unexpected error, switching to next STT",
                extra={"streamed": False},
            )
            raise

    def _try_recovery(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        stt_status = self._status[self._stt_instances.index(stt)]
        if (
            stt_status.recovering_synthesize_task is None
            or stt_status.recovering_synthesize_task.done()
        ):

            async def _recover_stt_task(stt: STT) -> None:
                try:
                    await self._try_recognize(
                        stt=stt,
                        buffer=buffer,
                        language=language,
                        conn_options=conn_options,
                        recovering=True,
                    )

                    stt_status.available = True
                    logger.info(f"{stt.label} recovered")
                    self.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=True),
                    )
                except Exception:
                    return

            stt_status.recovering_synthesize_task = asyncio.create_task(
                _recover_stt_task(stt)
            )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ):
        start_time = time.time()

        all_failed = all(not stt_status.available for stt_status in self._status)
        if all_failed:
            logger.error("all STTs are unavailable, retrying..")

        for i, stt in enumerate(self._stt_instances):
            stt_status = self._status[i]
            if stt_status.available or all_failed:
                try:
                    return await self._try_recognize(
                        stt=stt,
                        buffer=buffer,
                        language=language,
                        conn_options=conn_options,
                        recovering=False,
                    )
                except Exception:  # exceptions already logged inside _try_recognize
                    if stt_status.available:
                        stt_status.available = False
                        self.emit(
                            "stt_availability_changed",
                            AvailabilityChangedEvent(stt=stt, available=False),
                        )

            self._try_recovery(
                stt=stt, buffer=buffer, language=language, conn_options=conn_options
            )

        raise APIConnectionError(
            "all STTs failed (%s) after %s seconds"
            % (
                [stt.label for stt in self._stt_instances],
                time.time() - start_time,
            )
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await super().recognize(
            buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return FallbackRecognizeStream(
            stt=self, language=language, conn_options=conn_options
        )

    async def aclose(self) -> None:
        for stt_status in self._status:
            if stt_status.recovering_synthesize_task is not None:
                await aio.gracefully_cancel(stt_status.recovering_synthesize_task)

            if stt_status.recovering_stream_task is not None:
                await aio.gracefully_cancel(stt_status.recovering_stream_task)


class FallbackRecognizeStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: FallbackAdapter,
        language: str | None,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=None)
        self._language = language
        self._fallback_adapter = stt
        self._recovering_streams: list[RecognizeStream] = []

    async def _run(self) -> None:
        start_time = time.time()

        all_failed = all(
            not stt_status.available for stt_status in self._fallback_adapter._status
        )
        if all_failed:
            logger.error("all STTs are unavailable, retrying..")

        main_stream: RecognizeStream | None = None
        forward_input_task: asyncio.Task | None = None

        async def _forward_input_task() -> None:
            with contextlib.suppress(RuntimeError):  # stream might be closed
                async for data in self._input_ch:
                    for stream in self._recovering_streams:
                        if isinstance(data, rtc.AudioFrame):
                            stream.push_frame(data)
                        elif isinstance(data, self._FlushSentinel):
                            stream.flush()

                    if main_stream is not None:
                        if isinstance(data, rtc.AudioFrame):
                            main_stream.push_frame(data)
                        elif isinstance(data, self._FlushSentinel):
                            main_stream.flush()

                if main_stream is not None:
                    main_stream.end_input()

        for i, stt in enumerate(self._fallback_adapter._stt_instances):
            stt_status = self._fallback_adapter._status[i]
            if stt_status.available or all_failed:
                try:
                    main_stream = stt.stream(
                        language=self._language,
                        conn_options=dataclasses.replace(
                            self._conn_options,
                            max_retry=self._fallback_adapter._max_retry_per_stt,
                            timeout=self._fallback_adapter._attempt_timeout,
                            retry_interval=self._fallback_adapter._retry_interval,
                        ),
                    )

                    if forward_input_task is None or forward_input_task.done():
                        forward_input_task = asyncio.create_task(_forward_input_task())

                    try:
                        async with main_stream:
                            async for ev in main_stream:
                                self._event_ch.send_nowait(ev)

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"{stt.label} timed out, switching to next STT",
                            extra={"streamed": True},
                        )
                        raise
                    except APIError as e:
                        logger.warning(
                            f"{stt.label} failed, switching to next STT",
                            exc_info=e,
                            extra={"streamed": True},
                        )
                        raise
                    except Exception:
                        logger.exception(
                            f"{stt.label} unexpected error, switching to next STT",
                            extra={"streamed": True},
                        )
                        raise

                    return
                except Exception:
                    if stt_status.available:
                        stt_status.available = False
                        self._stt.emit(
                            "stt_availability_changed",
                            AvailabilityChangedEvent(stt=stt, available=False),
                        )

            self._try_recovery(stt)

        if forward_input_task is not None:
            await aio.gracefully_cancel(forward_input_task)

        await asyncio.gather(*[stream.aclose() for stream in self._recovering_streams])

        raise APIConnectionError(
            "all STTs failed (%s) after %s seconds"
            % (
                [stt.label for stt in self._fallback_adapter._stt_instances],
                time.time() - start_time,
            )
        )

    def _try_recovery(self, stt: STT) -> None:
        stt_status = self._fallback_adapter._status[
            self._fallback_adapter._stt_instances.index(stt)
        ]
        if (
            stt_status.recovering_stream_task is None
            or stt_status.recovering_stream_task.done()
        ):
            stream = stt.stream(
                language=self._language,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=0,
                    timeout=self._fallback_adapter._attempt_timeout,
                ),
            )
            self._recovering_streams.append(stream)

            async def _recover_stt_task() -> None:
                try:
                    nb_transcript = 0
                    async with stream:
                        async for ev in stream:
                            if ev.type in SpeechEventType.FINAL_TRANSCRIPT:
                                if not ev.alternatives or not ev.alternatives[0].text:
                                    continue

                                nb_transcript += 1
                                break

                    if nb_transcript == 0:
                        return

                    stt_status.available = True
                    logger.info(f"tts.FallbackAdapter, {stt.label} recovered")
                    self._fallback_adapter.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=True),
                    )

                except asyncio.TimeoutError:
                    logger.warning(
                        f"{stream._stt.label} recovery timed out",
                        extra={"streamed": True},
                    )
                except APIError as e:
                    logger.warning(
                        f"{stream._stt.label} recovery failed",
                        exc_info=e,
                        extra={"streamed": True},
                    )
                except Exception:
                    logger.exception(
                        f"{stream._stt.label} recovery unexpected error",
                        extra={"streamed": True},
                    )
                    raise

            stt_status.recovering_stream_task = task = asyncio.create_task(
                _recover_stt_task()
            )
            task.add_done_callback(lambda _: self._recovering_streams.remove(stream))
