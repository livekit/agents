"""Audio EOT detector base, the merged stream state machine (with built-in
cloud→local fallback), and the transport Protocol concrete backends satisfy.

Lives next to its transports rather than in ``voice/turn`` so the
fallback logic is a peer of the transports it switches between rather than a
template-method-across-packages.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from livekit import rtc

from ... import utils
from ...language import LanguageCode
from ...llm import ChatContext
from ...log import logger
from ...types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)
from ...utils import aio
from ...voice.turn import TurnDetectionEvent
from .languages import ThresholdOptions, TurnDetectorModels

DEFAULT_SAMPLE_RATE: int = 16000

MIN_SILENCE_DURATION_MS = 200
"""Minimum VAD silence the audio EOT detector needs before it will warm up an
inference window. Enforced against the caller-supplied VAD's
``min_silence_duration`` in ``AudioRecognition``."""


class _Status(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class TurnDetectorOptions:
    sample_rate: int
    thresholds: ThresholdOptions


@runtime_checkable
class _AudioTurnDetectionTransport(Protocol):
    async def run(self) -> None: ...

    def start_inference(self, request_id: str) -> None: ...
    def stop_inference(self, *, reason: str | None) -> None: ...
    def push_frame(self, frame: rtc.AudioFrame) -> None: ...
    def flush(self) -> None: ...

    def attach(self, stream: _AudioTurnDetectorStream) -> None: ...
    def detach(self) -> None: ...


class _AudioTurnDetector(rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(self, *, opts: TurnDetectorOptions) -> None:
        super().__init__()
        self._opts = opts

    @property
    def model(self) -> TurnDetectorModels:
        raise NotImplementedError

    @property
    def provider(self) -> str:
        return "livekit"

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _AudioTurnDetectorStream:
        raise NotImplementedError

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._opts.thresholds.lookup(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return self._opts.thresholds.supports(language)


class _AudioTurnDetectorStream:
    @dataclass
    class _FlushSentinel:
        reason: str | None = None

    def __init__(
        self,
        *,
        detector: _AudioTurnDetector,
        opts: TurnDetectorOptions,
        transport: _AudioTurnDetectionTransport,
        model: TurnDetectorModels = "turn-detector-mini",
    ) -> None:
        self._detector = detector
        self._opts = opts
        self._transport = transport
        self._transport.attach(self)

        self._model: TurnDetectorModels = model
        self._is_fallback = False
        self._warned_cloud_failure = False
        self._warned_local_failure = False
        self._transport_task: asyncio.Task[None] | None = None
        self._fallback_requested = False

        self._audio_input_sample_rate: int | None = None
        self._audio_input_num_channels: int | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._audio_ch = aio.Chan[rtc.AudioFrame | _AudioTurnDetectorStream._FlushSentinel]()

        self._status: _Status = _Status.IDLE
        self._preemptive_request_id: str | None = None
        self._preemptive_request_fut: asyncio.Future[float] | None = None
        self._last_prediction: TurnDetectionEvent | None = None
        self._last_language: LanguageCode | None = None
        self._user_turn_committed: bool = False
        self._late_predict_warned: bool = False

        self._tasks: set[asyncio.Task[None]] = set()
        self._task = asyncio.create_task(self._main_task())

    # region: detector proxies

    @property
    def model(self) -> TurnDetectorModels:
        # The stream owns its active model, so after a fallback this reports
        # "turn-detector-mini". The detector and stream share one mutable
        # ``ThresholdOptions``, and the cloud→local fallback it performs is
        # one-way and sticky: once degraded it never returns to cloud, so the
        # detector view stays consistent for the rest of its lifetime.
        return self._model

    @property
    def provider(self) -> str:
        return self._detector.provider

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._opts.thresholds.lookup(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return self._opts.thresholds.supports(language)

    def update_language(self, language: LanguageCode | None) -> None:
        self._last_language = language

    def _is_likely(self, probability: float) -> bool:
        threshold = self._opts.thresholds.lookup(self._last_language)
        return threshold is not None and probability >= threshold

    # endregion

    # region: state machine

    @property
    def is_active(self) -> bool:
        return self._status == _Status.ACTIVE

    @property
    def is_inference_running(self) -> bool:
        return self._preemptive_request_id is not None

    @property
    def preemptive_request_id(self) -> str | None:
        return self._preemptive_request_id

    def warmup(self) -> asyncio.Future[float]:
        if self._preemptive_request_id is None:
            request_id = utils.shortuuid("turn_request_")
            self._preemptive_request_id = request_id
            self._preemptive_request_fut = asyncio.Future[float]()
            # New inference window — drop any cached prediction from the
            # previous window so ``predict_end_of_turn`` won't return stale.
            self._last_prediction = None
            self._transport.start_inference(request_id)
        if self._preemptive_request_fut is None:
            raise RuntimeError("eot detection warmup failed, no request future")
        return self._preemptive_request_fut

    def activate(self, trigger: str | None = None) -> None:
        if self._status == _Status.ACTIVE:
            return
        if self._preemptive_request_id is None:
            logger.trace(
                "eot detector not warmed up before activation, likely due to overlapping speech"
            )
            self.warmup()
        self._status = _Status.ACTIVE
        # A prediction may have resolved during the preemptive warmup window,
        # before activation. We deliberately hold off acting on the threshold
        # until now: a confident EOT only commits once VAD confirms
        # end-of-speech (the trigger that calls ``activate``).
        if self._last_prediction is not None and self._is_likely(
            self._last_prediction.end_of_turn_probability
        ):
            self.deactivate(trigger="positive eou prediction")

    def deactivate(self, trigger: str | None = None) -> None:
        # Set the turn flag before any early returns: callers (VAD SOS, in
        # particular) rely on deactivate re-arming the stream even when there's
        # no in-flight inference to stop. ``flush`` overrides this back to True
        # after calling us to mark the turn as committed, blocking late
        # ``predict_end_of_turn`` calls until the next deactivate re-arms it.
        self._user_turn_committed = False
        if self._preemptive_request_id is None and self._status == _Status.IDLE:
            return
        self._preemptive_request_id = None
        if self._preemptive_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._preemptive_request_fut.set_result(0.0)
            self._preemptive_request_fut = None
        self._status = _Status.IDLE
        self._transport.stop_inference(reason=trigger)

    def flush(self, reason: str | None = None) -> None:
        # Idempotent: a second call sends another sentinel that transports
        # treat as a no-op (cloud: redundant session_flush; local: empty trim).
        if self._audio_ch.closed:
            return

        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(_AudioTurnDetectorStream._FlushSentinel(reason=reason))
        # Turn boundary — the cached prediction belongs to the turn we just
        # closed and must not leak into the next one.
        self._last_prediction = None
        self.deactivate(trigger=reason)
        # Commit the turn after deactivate so the flag override sticks; until
        # the next VAD SOS (which calls deactivate again) ``predict_end_of_turn``
        # short-circuits.
        self._user_turn_committed = True

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

    def _handle_prediction(
        self,
        request_id: str,
        probability: float,
        *,
        inference_duration: float | None = None,
        detection_delay: float | None = None,
    ) -> None:
        """Accept a prediction from a transport. Stream owns dedup (by
        request_id), future resolution, and the inline early-deactivate."""
        if request_id != self._preemptive_request_id:
            return
        if self._preemptive_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._preemptive_request_fut.set_result(probability)
        event = TurnDetectionEvent(
            type="eot_prediction",
            last_speaking_time=time.time(),
            end_of_turn_probability=probability,
            detection_delay=detection_delay,
            inference_duration=inference_duration,
        )
        self._last_prediction = event
        # Early-deactivate: stop inference as soon as a confident EOT lands so a
        # later intra-speech silence can warm up a fresh window. Only while
        # active — predictions during preemptive warmup are cached and
        # re-checked in ``activate()``. ``deactivate`` just sends a non-blocking
        # ``stop_inference``, so calling it inline from the transport's
        # prediction callback is safe (no reentrant await).
        if self.is_active and self._is_likely(probability):
            self.deactivate(trigger="positive eou prediction")

    @property
    def last_prediction(self) -> TurnDetectionEvent | None:
        """Most recent resolved prediction in the current inference window,
        or ``None`` if no prediction has arrived yet."""
        return self._last_prediction

    async def predict_end_of_turn(
        self,
        chat_ctx: ChatContext | None = None,
        *,
        timeout: float | None = None,
    ) -> float:
        """Run a warmup inference and wait for a prediction within `timeout`.

        Returns the cached prediction if one has already arrived for the
        current inference window. ``chat_ctx`` is accepted (and ignored) so
        the call site stays uniform with text-based ``_TurnDetector``
        implementations.
        """
        if self._last_prediction is not None:
            return self._last_prediction.end_of_turn_probability

        if self._user_turn_committed:
            if not self._late_predict_warned:
                self._late_predict_warned = True
                logger.warning(
                    "predict_end_of_turn called after the audio eot model already "
                    "committed the turn (likely a late stt final). consider raising "
                    "`min_delay` in the endpointing options to accommodate slow stt. "
                    "subsequent occurrences on this stream will log at debug level.",
                )
            else:
                logger.debug(
                    "stt transcript arrived after a turn commit, short-circuiting",
                )
            return 1.0

        timeout = timeout if timeout is not None else 0.5
        fut: asyncio.Future[float] | None = None
        try:
            fut = self.warmup()
            self.activate()
            done, _ = await asyncio.wait([fut], timeout=timeout)
            if not done:
                raise asyncio.TimeoutError()
            return done.pop().result()
        except asyncio.TimeoutError:
            # Contract on timeout: we couldn't tell within `timeout`, so assume
            # the turn is over. Resolve the future with 1.0 (so any concurrent
            # waiter sees the same value) and deactivate the inference window
            # (a stale prediction arriving later must not fire an event).
            logger.warning(
                "eot prediction timed out, returning a default value",
                extra={
                    "timeout": timeout,
                    "request_id": self._preemptive_request_id,
                    "default": 1.0,
                },
            )
            if fut is not None:
                with contextlib.suppress(asyncio.InvalidStateError):
                    fut.set_result(1.0)
            self.deactivate(trigger="predict_end_of_turn timeout")
            # Cloud predict timeout = transport failure; promote the mini model
            # for the rest of the session and let the next call retry on the new
            # transport.
            if self._model == "turn-detector":
                self._fall_back_to_local(reason=asyncio.TimeoutError("predict_end_of_turn"))
            # Positive default so min_endpointing_delay applies.
            return 1.0

    # endregion

    # region: teardown

    async def aclose(self) -> None:
        self._transport.detach()
        self.end_input()
        await aio.cancel_and_wait(self._task)
        await aio.cancel_and_wait(*self._tasks)
        if self._preemptive_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._preemptive_request_fut.set_result(0.0)
        self._preemptive_request_fut = None
        self._preemptive_request_id = None
        self._status = _Status.IDLE

    # endregion

    # region: main task + fallback

    async def _main_task(self) -> None:
        await self._run()

    async def _drain_audio_channel(self) -> None:
        async for item in self._audio_ch:
            if isinstance(item, _AudioTurnDetectorStream._FlushSentinel):
                self._transport.flush()
            else:
                self._transport.push_frame(item)

    async def _run(self) -> None:
        """Run the active transport, retrying on cloud failure by swapping in
        a local transport in-place. ``turn-detector-mini`` just runs the
        transport once and surfaces failures to the caller via
        ``_handle_prediction`` (default 1.0)."""
        while True:
            task = asyncio.create_task(self._transport.run())
            self._transport_task = task
            try:
                await task
                return
            except asyncio.CancelledError:
                # _fall_back_to_local sets _fallback_requested before cancelling
                # this child task; any other cancellation (e.g. aclose cancelling
                # the parent) leaves the flag unset and propagates.
                if self._fallback_requested:
                    self._fallback_requested = False
                    continue
                if not task.done():
                    await aio.cancel_and_wait(task)
                raise
            except Exception as e:  # noqa: BLE001 — any cloud error degrades to local
                if self._model == "turn-detector":
                    self._fall_back_to_local(reason=e)
                    continue
                self._on_local_failure(reason=e)
                return

    def _fall_back_to_local(self, *, reason: BaseException) -> None:
        # Lazy import: transports.py imports this module for the Protocol and
        # constants, so importing it at module load would cycle.
        from .transports import _LocalTransport

        if not self._warned_cloud_failure:
            logger.warning(
                "cloud turn detector failed (%s); falling back to local mini model",
                reason,
            )
            self._warned_cloud_failure = True

        self._emit_default_for_inflight()
        self._transport.detach()
        self._opts.thresholds._to_local_fallback()
        from .detector import AudioTurnDetector

        if isinstance(self._detector, AudioTurnDetector):
            self._detector._model = "turn-detector-mini"
        self._transport = _LocalTransport(opts=self._opts)
        self._transport.attach(self)
        self._model = "turn-detector-mini"
        self._is_fallback = True
        # If transport.run() is still in flight (e.g. predict timeout while
        # the cloud session was otherwise idle), signal+cancel so _run loops
        # onto the local transport. Without this the orphaned WS lingers until
        # the gateway closes it for inactivity and the ensuing error surfaces
        # as a misleading log.
        task = self._transport_task
        if task is not None and not task.done():
            self._fallback_requested = True
            task.cancel()

    def _on_local_failure(self, *, reason: BaseException) -> None:
        if not self._warned_local_failure:
            logger.warning(
                "local audio turn detector failed (%s); defaulting to 1.0 and retrying on next turn",
                reason,
            )
            self._warned_local_failure = True
        self._emit_default_for_inflight()

    def _emit_default_for_inflight(self) -> None:
        request_id = self._preemptive_request_id
        if request_id is not None:
            self._handle_prediction(request_id, 1.0)

    # endregion
