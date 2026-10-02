from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc

from .. import utils
from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils.audio import AudioBuffer
from ..vad import VAD
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

if TYPE_CHECKING:
    from ..llm.chat_context import MetricsMetadata
    from ..voice.events import ConversationItemAddedEvent

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
    recovering_recognize_task: asyncio.Task[None] | None
    recovering_stream_task: asyncio.Task[None] | None
    waiting_streams: dict[FallbackRecognizeStream, None]
    recovery_closed: bool


class FallbackAdapter(
    STT[Literal["stt_availability_changed"]],
):
    """Agent Fallback Adapter for STT. Manages multiple STT instances with automatic fallback
    when the primary provider fails.
    """

    def __init__(
        self,
        stt: list[STT],
        *,
        vad: VAD | None = None,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        if len(stt) < 1:
            raise ValueError("At least one STT instance must be provided.")

        owned_stream_adapters: list[STT] = []
        non_streaming_stt = [t for t in stt if not t.capabilities.streaming]
        if non_streaming_stt:
            if vad is None:
                labels = ", ".join(t.label for t in non_streaming_stt)
                raise ValueError(
                    f"STTs do not support streaming: {labels}. "
                    "Provide a VAD to enable stt.StreamAdapter automatically "
                    "or wrap them with stt.StreamAdapter before using this adapter."
                )
            from ..stt import StreamAdapter

            adapted_stt: list[STT] = []
            for stt_instance in stt:
                if not stt_instance.capabilities.streaming:
                    stt_instance = StreamAdapter(stt=stt_instance, vad=vad)
                    owned_stream_adapters.append(stt_instance)
                adapted_stt.append(stt_instance)
            stt = adapted_stt

        # Use the primary STT's aligned_transcript if all providers support it, since
        # the SDK only checks truthiness, not the specific granularity.
        aligned_transcript: Literal["word", "chunk", False] = False
        if all(t.capabilities.aligned_transcript for t in stt):
            aligned_transcript = stt[0].capabilities.aligned_transcript

        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=all(t.capabilities.interim_results for t in stt),
                diarization=all(t.capabilities.diarization for t in stt),
                aligned_transcript=aligned_transcript,
                keyterms=any(t.capabilities.keyterms for t in stt),
                chat_context=any(t.capabilities.chat_context for t in stt),
            )
        )

        self._stt_instances = stt
        self._owned_stream_adapters = owned_stream_adapters
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_stt = max_retry_per_stt
        self._retry_interval = retry_interval

        self._status: list[_STTStatus] = [
            _STTStatus(
                available=True,
                recovering_recognize_task=None,
                recovering_stream_task=None,
                waiting_streams={},
                recovery_closed=False,
            )
            for _ in self._stt_instances
        ]

        # the instance that most recently served a request; used to label metrics & traces
        self._active_instance: STT = self._stt_instances[0]

        for stt_instance in self._stt_instances:
            stt_instance.on("metrics_collected", self._on_metrics_collected)
        self._recognize_metrics_needed = False  # don't emit metrics via fallback adapter
        self._closed = False

    def _next_instance(self) -> STT:
        """The instance the next request goes to first: the first one marked available, or
        the primary once all are down (they are then all retried, primary first). A failed
        instance's recovery task flips it back to available, so a recovered primary is
        reported again before it has served."""
        for instance, status in zip(self._stt_instances, self._status, strict=True):
            if status.available:
                return instance
        return self._stt_instances[0]

    @property
    def model(self) -> str:
        """The model of the instance that serves next (see :meth:`_next_instance`). Spans and
        metrics read this, so a failover shows the model expected to answer rather than the
        adapter; the instance that actually served is stamped per request by the stream."""
        return self._next_instance().model

    @property
    def provider(self) -> str:
        """The provider of the instance that serves next (see :attr:`model`)."""
        return self._next_instance().provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """Metadata of the instance that most recently served a request (the primary before any traffic)."""  # noqa: E501
        return self._active_instance.metrics_metadata

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        # forward to every underlying STT; unsupported ones warn-and-skip internally
        for stt_instance in self._stt_instances:
            stt_instance._update_session_keyterms(keyterms)

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        # forward to every underlying STT; unsupported ones warn-and-skip internally
        for stt_instance in self._stt_instances:
            stt_instance._push_conversation_item(ev)

    async def _try_recognize(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: NotGivenOr[str] = NOT_GIVEN,
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
            logger.warning(
                "STT recovery timed out"
                if recovering
                else "STT timed out, switching to next provider",
                extra={
                    "stt": stt.label,
                    "streamed": False,
                    "error_type": "TimeoutError",
                },
            )
            raise
        except APIError as e:
            logger.warning(
                "STT recovery failed" if recovering else "STT failed, switching to next provider",
                extra={
                    "stt": stt.label,
                    "streamed": False,
                    "error_type": type(e).__name__,
                },
            )
            raise
        except Exception as e:
            log = logger.debug if recovering else logger.warning
            log(
                "STT recovery failed" if recovering else "STT failed, switching to next provider",
                extra={
                    "stt": stt.label,
                    "streamed": False,
                    "error_type": type(e).__name__,
                },
            )
            raise

    def _try_recovery(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        stt_status = self._status[self._stt_instances.index(stt)]
        if self._closed or stt_status.recovery_closed:
            return
        if (
            stt_status.recovering_recognize_task is None
            or stt_status.recovering_recognize_task.done()
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

                    if self._closed or stt_status.recovery_closed:
                        return
                    stt_status.available = True
                    logger.info(f"{stt.label} recovered")
                    self.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=True),
                    )
                except Exception:
                    return

            task = asyncio.create_task(_recover_stt_task(stt))
            stt_status.recovering_recognize_task = task

            def _clear_recovery(done: asyncio.Task[None]) -> None:
                if stt_status.recovering_recognize_task is done:
                    stt_status.recovering_recognize_task = None

            task.add_done_callback(_clear_recovery)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        start_time = time.time()

        all_failed = all(not stt_status.available for stt_status in self._status)
        if all_failed:
            logger.error("all STTs are unavailable, retrying..")

        for i, stt in enumerate(self._stt_instances):
            stt_status = self._status[i]
            if stt_status.available or all_failed:
                try:
                    event = await self._try_recognize(
                        stt=stt,
                        buffer=buffer,
                        language=language,
                        conn_options=conn_options,
                        recovering=False,
                    )
                    self._active_instance = stt
                    return event
                except Exception:  # exceptions already logged inside _try_recognize
                    if stt_status.available:
                        stt_status.available = False
                        self.emit(
                            "stt_availability_changed",
                            AvailabilityChangedEvent(stt=stt, available=False),
                        )

            self._try_recovery(stt=stt, buffer=buffer, language=language, conn_options=conn_options)

        raise APIConnectionError(
            f"all STTs failed ({[stt.label for stt in self._stt_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await super().recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return FallbackRecognizeStream(stt=self, language=language, conn_options=conn_options)

    def prewarm(self) -> None:
        """Pre-warm the primary STT.

        Only the first instance is prewarmed; the remaining instances are not expected to
        serve traffic unless the primary fails.
        """
        if self._stt_instances:
            self._stt_instances[0].prewarm()

    async def aclose(self) -> None:
        was_closed = self._closed
        self._closed = True
        tasks: set[asyncio.Task[None]] = set()
        for stt_status in self._status:
            stt_status.recovery_closed = True
            stt_status.waiting_streams.clear()
            if stt_status.recovering_recognize_task is not None:
                tasks.add(stt_status.recovering_recognize_task)

            if stt_status.recovering_stream_task is not None:
                tasks.add(stt_status.recovering_stream_task)

        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=1.0)

        if not was_closed:
            for stt in self._stt_instances:
                stt.off("metrics_collected", self._on_metrics_collected)

            for stream_adapter in self._owned_stream_adapters:
                await stream_adapter.aclose()

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)


class FallbackRecognizeStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: FallbackAdapter,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=NOT_GIVEN)
        self._language = language
        self._fallback_adapter = stt
        self._recovering_streams: dict[RecognizeStream, asyncio.Task[None]] = {}
        self._input_ended = False
        self._closing = False
        self._run_generation = 0
        self._close_tasks: set[asyncio.Task[None]] = set()

    async def aclose(self) -> None:
        self._closing = True
        await super().aclose()

    def _close_stream(self, stream: RecognizeStream) -> None:
        task = asyncio.create_task(stream.aclose())
        self._close_tasks.add(task)

        def _close_done(done: asyncio.Task[None]) -> None:
            self._close_tasks.discard(done)
            if not done.cancelled():
                done.exception()
            child_task = stream._task
            if child_task.done() and not child_task.cancelled():
                child_task.exception()

        task.add_done_callback(_close_done)

    def _try_recovery(self, stt: STT) -> bool:
        generation = self._run_generation
        stt_status = self._fallback_adapter._status[
            self._fallback_adapter._stt_instances.index(stt)
        ]
        if self._closing or self._task.done() or stt_status.available or stt_status.recovery_closed:
            return False
        if (
            stt_status.recovering_stream_task is not None
            and not stt_status.recovering_stream_task.done()
        ):
            stt_status.waiting_streams[self] = None
            return False

        stt_status.waiting_streams.pop(self, None)
        try:
            stream = stt.stream(
                language=self._language,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=0,
                    timeout=self._fallback_adapter._attempt_timeout,
                    retry_interval=self._fallback_adapter._retry_interval,
                ),
            )
        except Exception as e:
            logger.warning(
                "STT recovery failed",
                extra={
                    "stt": stt.label,
                    "streamed": True,
                    "error_type": type(e).__name__,
                },
            )
            return False

        async def _recover_stt_task() -> None:
            try:
                async with stream:
                    async for ev in stream:
                        if (
                            self._closing
                            or generation != self._run_generation
                            or stt_status.recovery_closed
                        ):
                            return
                        if (
                            ev.type == SpeechEventType.FINAL_TRANSCRIPT
                            and ev.alternatives
                            and ev.alternatives[0].text
                        ):
                            if not stt_status.available:
                                stt_status.available = True
                                logger.info("stt.FallbackAdapter, %s recovered", stt.label)
                                self._fallback_adapter.emit(
                                    "stt_availability_changed",
                                    AvailabilityChangedEvent(stt=stt, available=True),
                                )
                            return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log = logger.warning if isinstance(e, APIError) else logger.debug
                log(
                    "STT recovery failed",
                    extra={
                        "stt": stt.label,
                        "streamed": True,
                        "error_type": type(e).__name__,
                    },
                )

        task = asyncio.create_task(_recover_stt_task())
        self._recovering_streams[stream] = task
        stt_status.recovering_stream_task = task

        def _recovery_done(done: asyncio.Task[None]) -> None:
            if self._recovering_streams.get(stream) is done:
                del self._recovering_streams[stream]
            if stt_status.recovering_stream_task is not done:
                return
            stt_status.recovering_stream_task = None
            if stt_status.available or stt_status.recovery_closed:
                stt_status.waiting_streams.clear()
                return
            while stt_status.waiting_streams:
                waiter = next(iter(stt_status.waiting_streams))
                stt_status.waiting_streams.pop(waiter, None)
                if waiter._try_recovery(stt):
                    break

        task.add_done_callback(_recovery_done)
        if self._input_ended:
            with contextlib.suppress(RuntimeError):
                stream.end_input()
        return True

    async def _run(self) -> None:
        self._run_generation += 1
        generation = self._run_generation
        start_time = time.time()

        all_failed = all(not stt_status.available for stt_status in self._fallback_adapter._status)
        if all_failed:
            logger.error("all STTs are unavailable, retrying..")

        main_stream: RecognizeStream | None = None

        async def _forward_input_task() -> None:
            try:
                async for data in self._input_ch:
                    for stream in list(self._recovering_streams):
                        try:
                            if isinstance(data, rtc.AudioFrame):
                                stream.push_frame(data)
                            elif isinstance(data, self._FlushSentinel):
                                stream.flush()
                        except Exception:
                            pass

                    current = main_stream
                    if current is None:
                        continue
                    try:
                        if isinstance(data, rtc.AudioFrame):
                            current.push_frame(data)
                        elif isinstance(data, self._FlushSentinel):
                            current.flush()
                    except Exception as e:
                        logger.debug(
                            "error forwarding input to main stream",
                            extra={
                                "stt": current._stt.label,
                                "streamed": True,
                                "error_type": type(e).__name__,
                            },
                        )
            finally:
                self._input_ended = True
                for end_target in [main_stream, *list(self._recovering_streams)]:
                    if end_target is not None:
                        with contextlib.suppress(RuntimeError):
                            end_target.end_input()

        forward_input_task = asyncio.create_task(_forward_input_task())

        try:
            for i, stt in enumerate(self._fallback_adapter._stt_instances):
                if self._closing:
                    return
                stt_status = self._fallback_adapter._status[i]
                if not stt_status.available and not all_failed:
                    self._try_recovery(stt)
                    continue

                try:
                    child = stt.stream(
                        language=self._language,
                        conn_options=dataclasses.replace(
                            self._conn_options,
                            max_retry=self._fallback_adapter._max_retry_per_stt,
                            timeout=self._fallback_adapter._attempt_timeout,
                            retry_interval=self._fallback_adapter._retry_interval,
                        ),
                    )
                    child.start_time_offset = self.start_time_offset + (time.time() - start_time)
                    main_stream = child
                    try:
                        if self._closing:
                            return
                        if self._input_ended:
                            with contextlib.suppress(RuntimeError):
                                child.end_input()
                        async for ev in child:
                            if self._closing or self._event_ch.closed:
                                return
                            self._fallback_adapter._active_instance = stt
                            self._event_ch.send_nowait(ev)
                    finally:
                        self._close_stream(child)

                    if self._closing or self._input_ended:
                        return
                    if stt_status.available:
                        stt_status.available = False
                        self._stt.emit(
                            "stt_availability_changed",
                            AvailabilityChangedEvent(stt=stt, available=False),
                        )
                        await asyncio.sleep(0)
                        if self._closing:
                            return
                    logger.warning(
                        "STT failed, switching to next provider",
                        extra={"stt": stt.label, "streamed": True},
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await asyncio.sleep(0)
                    if self._closing:
                        return
                    logger.warning(
                        "STT failed, switching to next provider",
                        extra={
                            "stt": stt.label,
                            "streamed": True,
                            "error_type": type(e).__name__,
                        },
                    )
                    if stt_status.available:
                        stt_status.available = False
                        self._stt.emit(
                            "stt_availability_changed",
                            AvailabilityChangedEvent(stt=stt, available=False),
                        )
                        await asyncio.sleep(0)
                        if self._closing:
                            return
                finally:
                    main_stream = None

                if self._closing:
                    return
                self._try_recovery(stt)

            if self._closing:
                return
            raise APIConnectionError(
                f"all STTs failed ({[stt.label for stt in self._fallback_adapter._stt_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
            )
        finally:
            if self._run_generation == generation:
                self._run_generation += 1
            if not self._input_ch.closed:
                self._input_ch.close()
            for stt_status in self._fallback_adapter._status:
                stt_status.waiting_streams.pop(self, None)
            recovery_tasks = list(self._recovering_streams.values())
            for task in recovery_tasks:
                task.cancel()
            for stream in list(self._recovering_streams):
                self._close_stream(stream)
            forward_input_task.cancel()
            tasks = {forward_input_task, *recovery_tasks, *self._close_tasks}
            if tasks:
                await asyncio.wait(tasks, timeout=1.0)

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        async for _ in event_aiter:
            pass
