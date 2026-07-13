"""Audio EOT detector base, the per-window inference stream (with built-in
cloud→local fallback), and the transport Protocol concrete backends satisfy.

Lives next to its transports rather than in ``voice/turn`` so the
fallback logic is a peer of the transports it switches between rather than a
template-method-across-packages.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from livekit import rtc

from ... import utils
from ..._exceptions import APITimeoutError
from ...language import LanguageCode
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
"""Minimum VAD silence the audio EOT detector needs before it sends
an inference request. Enforced against the caller-supplied VAD's
``min_silence_duration`` in ``AudioRecognition``."""

DEFAULT_PREDICTION_TIMEOUT = 1.0


@dataclass
class TurnDetectorOptions:
    sample_rate: int
    thresholds: ThresholdOptions


@runtime_checkable
class _StreamingTurnDetectionTransport(Protocol):
    async def run(self) -> None: ...

    def run_inference(self, request_id: str) -> None: ...
    def push_frame(self, frame: rtc.AudioFrame) -> None: ...
    def flush(self) -> None: ...

    def attach(self, stream: _BaseStreamingTurnDetectorStream) -> None: ...
    def detach(self) -> None: ...


class _BaseStreamingTurnDetector(rtc.EventEmitter[Literal["metrics_collected"]]):
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
    ) -> _BaseStreamingTurnDetectorStream:
        raise NotImplementedError

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._opts.thresholds.lookup(language)

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return self._opts.thresholds.lookup_backchannel(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return self._opts.thresholds.supports(language)


class _BaseStreamingTurnDetectorStream:
    @dataclass
    class _FlushSentinel:
        reason: str | None = None

    def __init__(
        self,
        *,
        detector: _BaseStreamingTurnDetector,
        opts: TurnDetectorOptions,
        transport: _StreamingTurnDetectionTransport,
        model: TurnDetectorModels = "turn-detector-v1-mini",
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
        self._audio_ch = aio.Chan[
            rtc.AudioFrame | _BaseStreamingTurnDetectorStream._FlushSentinel
        ]()

        self._request_id: str | None = None
        self._request_fut: asyncio.Future[TurnDetectionEvent] | None = None

        self._tasks: set[asyncio.Task[None]] = set()
        self._task = asyncio.create_task(self._main_task())

    # region: detector proxies

    @property
    def model(self) -> TurnDetectorModels:
        # The stream owns its active model, so after a fallback this reports
        # "turn-detector-v1-mini". The detector and stream share one mutable
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

    @property
    def prediction_timeout(self) -> float:
        return DEFAULT_PREDICTION_TIMEOUT

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._opts.thresholds.lookup(language)

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return self._opts.thresholds.lookup_backchannel(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return self._opts.thresholds.supports(language)

    # endregion

    # region: inference requests

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        """Start a new inference request and return its future."""
        if self._audio_ch.closed:
            fut: asyncio.Future[TurnDetectionEvent] = asyncio.get_running_loop().create_future()
            fut.set_result(self._default_event(1.0))
            return fut

        self.cancel_inference()  # supersede any previous request
        self._request_id = utils.shortuuid("turn_request_")
        self._request_fut = asyncio.get_running_loop().create_future()
        self._transport.run_inference(self._request_id)
        return self._request_fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        """Close the current inference request (new speech, turn boundary,
        prediction timeout, mode change) and fall back if needed.
        """
        if self._request_id is not None:
            fut = self._request_fut
            self._request_id = None
            self._request_fut = None
            if fut is not None and not fut.done():
                fut.set_result(self._default_event(0.0))

        # trigger fallback immediately
        if timed_out and self._model == "turn-detector-v1":
            self._fall_back_to_local(reason=APITimeoutError("eot prediction timed out"))

    def flush(self, reason: str | None = None) -> None:
        # Idempotent: a second call sends another sentinel that transports
        # treat as a no-op (cloud: redundant session_flush; local: empty trim).
        if self._audio_ch.closed:
            return

        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(_BaseStreamingTurnDetectorStream._FlushSentinel(reason=reason))
        self.cancel_inference()

    @staticmethod
    def _default_event(probability: float) -> TurnDetectionEvent:
        return TurnDetectionEvent(
            type="eot_prediction",
            last_speaking_time=time.time(),
            end_of_turn_probability=probability,
        )

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

    def _resolve_prediction(
        self,
        request_id: str,
        probability: float,
        *,
        inference_duration: float | None = None,
        detection_delay: float | None = None,
        backchannel_probability: float | None = None,
    ) -> None:
        """Accept a prediction from a transport. Stale response is ignored."""
        if request_id != self._request_id:
            return
        fut = self._request_fut
        self._request_id = None
        self._request_fut = None
        if fut is not None and not fut.done():
            fut.set_result(
                TurnDetectionEvent(
                    type="eot_prediction",
                    last_speaking_time=time.time(),
                    end_of_turn_probability=probability,
                    detection_delay=detection_delay,
                    inference_duration=inference_duration,
                    backchannel_probability=backchannel_probability,
                )
            )

    # endregion

    # region: teardown

    async def aclose(self) -> None:
        self._transport.detach()
        self.end_input()  # the flush inside closes the in-flight request
        await aio.cancel_and_wait(self._task)
        await aio.cancel_and_wait(*self._tasks)
        self.cancel_inference()  # defensive, normally a no-op

    # endregion

    # region: main task + fallback

    async def _main_task(self) -> None:
        await self._run()

    async def _drain_audio_channel(self) -> None:
        async for item in self._audio_ch:
            if isinstance(item, _BaseStreamingTurnDetectorStream._FlushSentinel):
                self._transport.flush()
            else:
                self._transport.push_frame(item)

    async def _run(self) -> None:
        """Run the active transport, retrying on cloud failure by swapping in
        a local transport in-place. ``turn-detector-v1-mini`` just runs the
        transport once and surfaces failures to the caller via
        ``_resolve_prediction`` (default 1.0)."""
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
                if self._model == "turn-detector-v1":
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
        from .detector import TurnDetector

        if isinstance(self._detector, TurnDetector):
            self._detector._model = "turn-detector-v1-mini"
        self._transport = _LocalTransport(opts=self._opts)
        self._transport.attach(self)
        self._model = "turn-detector-v1-mini"
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
        # Positive default so any waiter commits after min_endpointing_delay.
        request_id = self._request_id
        if request_id is not None:
            self._resolve_prediction(request_id, 1.0)

    # endregion
