"""Audio EOT (end-of-turn) detector base, stream state machine, and the
transport Protocol that concrete cloud/local backends implement.

Concrete implementations live in ``livekit.agents.inference.eot``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
from .base import TurnDetectionEvent

DEFAULT_SAMPLE_RATE: int = 16000
MIN_SILENCE_DURATION_MS = 200


class _Status(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class TurnDetectorOptions:
    """Options shared by the audio EOT stream and every transport.

    Cloud-only transport concerns (base URL, credentials, conn options)
    live on a separate options dataclass owned by the cloud transport.
    """

    sample_rate: int
    thresholds: dict[str, float] = field(default_factory=dict)


class _AudioTurnDetector(rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(self, *, opts: TurnDetectorOptions) -> None:
        super().__init__()
        self._opts = opts
        self._streams: weakref.WeakSet[_AudioTurnDetectorStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
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
        # `language=None` means "unknown" — treat as English so callers without
        # a detected language still get a working threshold. Explicit unknown
        # codes (e.g. "vi") still miss the dict and return None, so
        # `supports_language` returns False for unsupported languages.
        lang_key = language.language if language is not None else "en"
        return self._opts.thresholds.get(lang_key)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self.unlikely_threshold(language) is not None

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


@runtime_checkable
class _AudioTurnDetectionTransport(Protocol):
    """Transport adapter for ``_AudioTurnDetectorStream`` — owns the I/O
    (WebSocket session, in-process predict, etc.). The stream calls these
    methods directly; transports report predictions back via
    ``stream._handle_prediction(request_id, probability, ...)``.
    """

    def bind(self, stream: _AudioTurnDetectorStream) -> None: ...
    async def run(self) -> None: ...
    def start_inference(self, request_id: str) -> None: ...
    async def push_frame(self, frame: rtc.AudioFrame) -> None: ...
    async def flush(self, sentinel: _AudioTurnDetectorStream._FlushSentinel) -> None: ...
    def stop_inference(self, *, reason: str | None) -> None: ...
    def detach(self) -> None: ...
    def transport_ready(self) -> bool: ...


class _AudioTurnDetectorStream:
    @dataclass
    class _FlushSentinel:
        reason: str | None = None
        # TODO: @chenghao-mou wire this in
        # Trailing audio (ms) to retain on flush — preserves VAD-onset pre-roll
        # so the next turn starts with a few frames of context.
        keep_tail_ms: int = 0

    def __init__(
        self,
        *,
        detector: _AudioTurnDetector,
        opts: TurnDetectorOptions,
        transport: _AudioTurnDetectionTransport,
    ) -> None:
        self._detector = detector
        self._opts = opts
        self._transport = transport
        self._transport.bind(self)

        self._audio_input_sample_rate: int | None = None
        self._audio_input_num_channels: int | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._audio_ch = aio.Chan[rtc.AudioFrame | _AudioTurnDetectorStream._FlushSentinel]()
        self._event_ch = aio.Chan[TurnDetectionEvent]()

        self._status: _Status = _Status.IDLE
        self._preemptive_request_id: str | None = None
        self._preemptive_request_fut: asyncio.Future[float] | None = None
        self._preemptive_prediction: TurnDetectionEvent | None = None
        # Latest resolved prediction for the current inference window. Cleared
        # when a new window starts (next warmup) or on commit (flush). Lets
        # ``predict_end_of_turn`` return immediately when a prediction has
        # already arrived via the event stream.
        self._last_prediction: TurnDetectionEvent | None = None
        # True between ``on_speech_started()`` and the next ``flush()`` —
        # i.e. a user turn is open and ``predict_end_of_turn`` should run.
        # When False, predict short-circuits to a positive default: there's
        # no fresh speech to evaluate (e.g. an STT final arriving after the
        # audio EOT model already committed the turn). Initialised True so
        # the first turn isn't gated before any flush has happened.
        self._user_turn_started: bool = True
        # Warn once per stream when ``predict_end_of_turn`` is called after
        # the audio EOT model has already committed the turn (common with
        # slow STT finals). Subsequent occurrences log at debug to avoid
        # flooding production logs with what is normal, expected behavior.
        self._late_predict_warned: bool = False

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
        # See `_AudioTurnDetector.unlikely_threshold`: None defaults to "en";
        # unsupported language codes return None.
        lang_key = language.language if language is not None else "en"
        return self._opts.thresholds.get(lang_key)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self.unlikely_threshold(language) is not None

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
        if not self._transport.transport_ready() or self._status == _Status.ACTIVE:
            return
        if self._preemptive_request_id is None:
            logger.trace(
                "eot detector not warmed up before activation, likely due to overlapping speech"
            )
            self.warmup()
        self._status = _Status.ACTIVE
        if self._preemptive_prediction is not None:
            held = self._preemptive_prediction
            self._preemptive_prediction = None
            self._emit_event(held)

    def deactivate(self, trigger: str | None = None) -> None:
        if not self._transport.transport_ready():
            return
        if self._preemptive_request_id is None and self._status == _Status.IDLE:
            return
        self._preemptive_request_id = None
        self._preemptive_prediction = None
        if self._preemptive_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._preemptive_request_fut.set_result(0.0)
            self._preemptive_request_fut = None
        self._status = _Status.IDLE
        self._transport.stop_inference(reason=trigger)

    def flush(self, reason: str | None = None, *, keep_tail_ms: int = 0) -> None:
        # Idempotent: a second call sends another sentinel that transports
        # treat as a no-op (cloud: redundant session_flush; local: empty trim).
        if self._audio_ch.closed:
            return

        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(
            _AudioTurnDetectorStream._FlushSentinel(reason=reason, keep_tail_ms=keep_tail_ms)
        )
        # Turn boundary — the cached prediction belongs to the turn we just
        # closed and must not leak into the next one.
        self._last_prediction = None
        # Close the user turn: until the next ``on_speech_started()`` signal,
        # ``predict_end_of_turn`` short-circuits.
        self._user_turn_started = False
        self.deactivate(trigger=reason)

    def on_speech_started(self) -> None:
        """Signal that a fresh user utterance has started. Opens the user
        turn so ``predict_end_of_turn`` runs normally again, and deactivates
        any in-flight inference for the now-stale prior window — the next
        VAD silence/end-of-speech will warm up and activate a fresh one."""
        self._user_turn_started = True
        self.deactivate(trigger="vad sos")

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

    def _emit_event(self, event: TurnDetectionEvent) -> None:
        self._event_ch.send_nowait(event)

    def _handle_prediction(
        self,
        request_id: str,
        probability: float,
        *,
        inference_duration: float | None = None,
        detection_delay: float | None = None,
    ) -> None:
        """Accept a prediction from a transport. Stream owns dedup (by
        request_id), future resolution, and active-vs-held event routing."""
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
        if self.is_active:
            self._emit_event(event)
        else:
            self._preemptive_prediction = event

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

        if not self._user_turn_started:
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
            self._on_predict_timeout()
            # Positive default so min_endpointing_delay applies.
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
        if self._preemptive_request_fut is not None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._preemptive_request_fut.set_result(0.0)
        self._preemptive_request_fut = None
        self._preemptive_request_id = None
        self._preemptive_prediction = None
        self._status = _Status.IDLE

    # endregion

    # region: main task scaffolding

    async def _main_task(self) -> None:
        await self._run()

    async def _drain_audio_channel(self) -> None:
        async for item in self._audio_ch:
            if isinstance(item, _AudioTurnDetectorStream._FlushSentinel):
                await self._transport.flush(item)
            else:
                await self._transport.push_frame(item)

    # endregion

    # region: subclass hooks

    async def _run(self) -> None:
        """Default: hand control to the transport. Subclasses override for
        cross-transport orchestration (e.g. cloud→local fallback)."""
        await self._transport.run()

    def _on_predict_timeout(self) -> None:  # noqa: B027
        """Genuine event: ``predict_end_of_turn`` timed out. Subclasses may
        override to react (e.g. promote local on cloud timeout)."""
        pass

    # endregion
