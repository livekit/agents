from __future__ import annotations

import asyncio
import contextlib
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from livekit import rtc

from ... import utils
from ...language import LanguageCode
from ...log import logger
from ...utils import aio
from .detector import TurnDetectionEvent, TurnDetectorOptions

if TYPE_CHECKING:
    from .detector import AudioTurnDetector


MIN_SILENCE_DURATION_MS = 200


class _InferenceStatus(str, Enum):
    DEACTIVATED = "deactivated"
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    FLUSHED = "flushed"


class BaseAudioTurnDetectionStream(ABC):
    """Shared scaffolding for audio EOT streams.

    Owns the audio resampler, audio/event channels, status FSM, and the
    public `_TurnDetector` Protocol surface (`model`, `provider`,
    `predict_end_of_turn`, `unlikely_threshold`, `supports_language`).
    Subclasses implement transport-specific hooks for the actual
    inference dispatch.
    """

    @dataclass
    class _FlushSentinel:
        reason: str | None = None
        # Number of milliseconds of trailing audio to retain on flush. 0 clears
        # the entire buffer. Used to anchor turn isolation on VAD SOS while
        # preserving a small pre-roll covering VAD onset latency.
        keep_tail_ms: int = 0

    def __init__(
        self,
        *,
        detector: AudioTurnDetector,
        opts: TurnDetectorOptions,
    ) -> None:
        self._detector = detector
        self._opts = opts

        self._audio_input_sample_rate: int | None = None
        self._audio_input_num_channels: int | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._audio_ch = aio.Chan[rtc.AudioFrame | BaseAudioTurnDetectionStream._FlushSentinel]()
        self._event_ch = aio.Chan[TurnDetectionEvent]()

        self._status: _InferenceStatus = _InferenceStatus.DEACTIVATED
        self._active_request_id: str | None = None
        self._active_request_fut: asyncio.Future[float] | None = None
        self._active_window_min_client_created_at_ms: int | None = None

        self._tasks: set[asyncio.Task[None]] = set()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    # region: _TurnDetector Protocol proxies

    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.unlikely_threshold(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self._detector.supports_language(language)

    # endregion

    # region: state machine

    @property
    def is_active(self) -> bool:
        return self._status == _InferenceStatus.ACTIVE

    @property
    def is_inference_running(self) -> bool:
        return self._status in (_InferenceStatus.WARMING_UP, _InferenceStatus.ACTIVE)

    def warmup(self) -> asyncio.Future[float]:
        if not self.is_inference_running:
            self._warmup()
        if self._active_request_fut is None:
            raise RuntimeError("eot detection warmup failed, no request future")
        return self._active_request_fut

    def stop_warmup(self) -> None:
        if not self.is_inference_running:
            return
        self._status = _InferenceStatus.DEACTIVATED
        self._active_request_id = None
        self._active_window_min_client_created_at_ms = None
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
            self._active_request_fut = None

    def _warmup(self) -> None:
        if self._status == _InferenceStatus.WARMING_UP:
            return
        self._status = _InferenceStatus.WARMING_UP
        request_id = utils.shortuuid("turn_request_")
        self._active_request_id = request_id
        self._active_request_fut = asyncio.Future[float]()
        self._on_warmup_start(request_id)

    def _activate(self) -> None:
        if self._status == _InferenceStatus.ACTIVE:
            return
        self._status = _InferenceStatus.ACTIVE
        self._on_activate()

    def _deactivate(self) -> None:
        if self._status == _InferenceStatus.DEACTIVATED:
            return
        self._active_request_id = None
        self._active_window_min_client_created_at_ms = None
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
            self._active_request_fut = None
        self._status = _InferenceStatus.DEACTIVATED

    def set_active(self, active: bool, trigger: str | None = None) -> None:
        if not self._transport_ready():
            return

        if active:
            if not self.is_active:
                if self._status != _InferenceStatus.WARMING_UP:
                    logger.warning("eot detector not warmed up before activation")
                    self.warmup()
                logger.trace("turn detection activated", extra={"trigger": trigger})
                self._activate()
            return

        if not self.is_active:
            return

        logger.trace("turn detection deactivated", extra={"trigger": trigger})
        self._deactivate()
        self._on_inference_stop(reason=trigger)

    def flush(self, reason: str | None = None, *, keep_tail_ms: int = 0) -> None:
        if self._audio_ch.closed or self._status == _InferenceStatus.FLUSHED:
            return

        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(
            BaseAudioTurnDetectionStream._FlushSentinel(reason=reason, keep_tail_ms=keep_tail_ms)
        )
        logger.trace("turn detection audio flushed", extra={"reason": reason})
        self._deactivate()
        self._status = _InferenceStatus.FLUSHED
        self._on_inference_stop(reason=reason)

    # endregion

    # region: audio ingress

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._audio_ch.closed:
            return
        for resampled_frame in self._resample_audio_frame(frame):
            self._audio_ch.send_nowait(resampled_frame)

    def end_input(self) -> None:
        self.flush()
        self._audio_ch.close()

    def _resample_audio_frame(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if self._audio_input_sample_rate is None or self._audio_input_num_channels is None:
            self._audio_input_sample_rate = frame.sample_rate
            self._audio_input_num_channels = frame.num_channels
            if self._audio_input_sample_rate != self._opts.sample_rate:
                self._audio_resampler = rtc.AudioResampler(
                    input_rate=self._audio_input_sample_rate,
                    output_rate=self._opts.sample_rate,
                    num_channels=self._audio_input_num_channels,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )
        elif (
            frame.sample_rate != self._audio_input_sample_rate
            or frame.num_channels != self._audio_input_num_channels
        ):
            logger.error(
                "a frame with different audio format was already pushed",
                extra={
                    "sample_rate": frame.sample_rate,
                    "expected_sample_rate": self._audio_input_sample_rate,
                    "num_channels": frame.num_channels,
                    "expected_num_channels": self._audio_input_num_channels,
                },
            )
            return []

        if self._audio_resampler is None:
            return [frame]
        return self._audio_resampler.push(frame)

    def _flush_audio_resampler(self) -> list[rtc.AudioFrame]:
        frames = self._audio_resampler.flush() if self._audio_resampler is not None else []
        self._reset_audio_resampler()
        return frames

    def _reset_audio_resampler(self) -> None:
        self._audio_resampler = None
        self._audio_input_sample_rate = None
        self._audio_input_num_channels = None

    # endregion

    # region: results

    def _emit_prediction(self, probability: float) -> None:
        self._event_ch.send_nowait(
            TurnDetectionEvent(
                type="eot_prediction",
                last_speaking_time=time.time(),
                end_of_turn_probability=probability,
            )
        )

    async def predict_end_of_turn(self, *, timeout: float | None = None) -> float:
        """Run a warmup inference and wait for a prediction within `timeout`.

        Used purely as a timeout shim around the warmup/activate path:
        - time to first prediction
        - time to next prediction since last prediction
        """
        timeout = timeout if timeout is not None else 0.5
        fut: asyncio.Future[float] | None = None
        try:
            fut = self.warmup()
            self._activate()
            done, _ = await asyncio.wait([fut], timeout=timeout)
            if not done:
                raise asyncio.TimeoutError()
            return done.pop().result()
        except asyncio.TimeoutError:
            logger.warning(
                "eot prediction timed out, returning a default value",
                extra={
                    "timeout": timeout,
                    "request_id": self._active_request_id,
                    "default": 1.0,
                },
            )
            if fut is not None:
                with contextlib.suppress(asyncio.InvalidStateError):
                    fut.set_result(1.0)
            self._active_request_fut = None
            self._active_request_id = None
            self._active_window_min_client_created_at_ms = None
            # default to a positive prediction so min_endpointing_delay is used
            return 1.0

    # endregion

    # region: async iteration

    async def __anext__(self) -> TurnDetectionEvent:
        try:
            return await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc  # noqa: B904
            raise StopAsyncIteration from None

    def __aiter__(self) -> AsyncIterator[TurnDetectionEvent]:
        return self

    async def aclose(self) -> None:
        self.end_input()
        await aio.cancel_and_wait(self._task)
        await aio.cancel_and_wait(*self._tasks)
        if self._active_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(0.0)
        self._active_request_fut = None
        self._active_request_id = None
        self._active_window_min_client_created_at_ms = None
        self._status = _InferenceStatus.DEACTIVATED

    # endregion

    # region: main task scaffolding

    async def _main_task(self) -> None:
        await self._run_transport()

    async def _drain_audio_channel(self) -> None:
        """Helper subclasses can call from `_run_transport` to dispatch
        audio frames and flush sentinels through the per-event hooks."""
        async for item in self._audio_ch:
            if isinstance(item, BaseAudioTurnDetectionStream._FlushSentinel):
                await self._on_flush_sentinel(item)
            else:
                await self._on_audio_chunk(item)

    # endregion

    # region: subclass hooks

    def _transport_ready(self) -> bool:
        """Return False to short-circuit `set_active` (e.g. transport closed)."""
        return True

    @abstractmethod
    async def _run_transport(self) -> None:
        """Long-running main-task body. Cloud runs the WS retry+send/recv
        loop here; local drains the audio channel into a rolling buffer."""
        ...

    def _on_warmup_start(self, request_id: str) -> None:  # noqa: B027
        """Called when the FSM transitions to WARMING_UP. Cloud sends an
        `inference_start` over WS; local snapshots the buffer and spawns
        an in-process predict task."""

    def _on_inference_stop(self, *, reason: str | None) -> None:  # noqa: B027
        """Called after `set_active(False)` or `flush()`. Cloud sends an
        `inference_stop`; local does nothing (in-flight predicts are
        cheap and can complete)."""

    async def _on_audio_chunk(self, frame: rtc.AudioFrame) -> None:  # noqa: B027
        """Called per resampled frame drained from `_audio_ch`. Cloud
        sends `input_audio` over WS; local appends bytes to the rolling
        buffer with overflow trim."""

    async def _on_flush_sentinel(  # noqa: B027
        self, sentinel: BaseAudioTurnDetectionStream._FlushSentinel
    ) -> None:
        """Called when a flush sentinel is drained from `_audio_ch`.
        Cloud sends `session_flush`; local clears the rolling buffer."""

    def _on_activate(self) -> None:  # noqa: B027
        """Called on FSM transition to ACTIVE. Cloud no-op; local
        replays any prediction that completed during WARMING_UP."""

    # endregion
