from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol

from typing_extensions import TypedDict

from livekit import rtc

from .. import utils
from ..language import LanguageCode
from ..llm import ChatContext
from ..log import logger
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import aio, is_given

DEFAULT_SAMPLE_RATE: int = 16000
MIN_SILENCE_DURATION_MS = 200


class _InferenceStatus(str, Enum):
    DEACTIVATED = "deactivated"
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    FLUSHED = "flushed"


@dataclass
class TurnDetectionEvent:
    type: Literal["eot_prediction"]
    end_of_turn_probability: float
    last_speaking_time: float
    detection_delay: float | None = None


@dataclass
class TurnDetectorOptions:
    sample_rate: int
    base_url: str
    api_key: str
    api_secret: str
    conn_options: APIConnectOptions
    # Materialized per-language thresholds keyed by base ISO code.
    thresholds: dict[str, float] = field(default_factory=dict)


def _normalize_user_threshold(
    value: NotGivenOr[float | dict[LanguageCode | str, float]],
) -> float | dict[str, float] | None:
    # Dict keys go through `LanguageCode(k).language` so "English"/"en"/"en-US"
    # all collapse to "en" — matches the table key shape used in lookups.
    if isinstance(value, dict):
        return {LanguageCode(k).language: float(v) for k, v in value.items()}
    if not is_given(value):
        return None
    return float(value)


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
        lang_key = language.language if language is not None else "en"
        return self._opts.thresholds.get(lang_key)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self.unlikely_threshold(language) is not None

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class _AudioTurnDetectorStream(ABC):
    @dataclass
    class _FlushSentinel:
        reason: str | None = None
        # Trailing audio (ms) to retain on flush — preserves VAD-onset pre-roll
        # so the next turn starts with a few frames of context.
        keep_tail_ms: int = 0

    def __init__(
        self,
        *,
        detector: _AudioTurnDetector,
        opts: TurnDetectorOptions,
    ) -> None:
        self._detector = detector
        self._opts = opts

        self._audio_input_sample_rate: int | None = None
        self._audio_input_num_channels: int | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._audio_ch = aio.Chan[rtc.AudioFrame | _AudioTurnDetectorStream._FlushSentinel]()
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
        # Read from the stream's own opts — on fallback the stream rebinds
        # `_opts` to a derived instance with the local-mode thresholds.
        lang_key = language.language if language is not None else "en"
        return self._opts.thresholds.get(lang_key)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self.unlikely_threshold(language) is not None

    # endregion

    # region: state machine

    @property
    def is_active(self) -> bool:
        return self._status == _InferenceStatus.ACTIVE

    @property
    def is_inference_running(self) -> bool:
        return self._status in (_InferenceStatus.WARMING_UP, _InferenceStatus.ACTIVE)

    @property
    def active_request_id(self) -> str | None:
        return self._active_request_id

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
                self._activate()
            return

        if not self.is_inference_running:
            return

        if self._status == _InferenceStatus.WARMING_UP:
            self.stop_warmup()
        else:
            self._deactivate()
        self._on_inference_stop(reason=trigger)

    def flush(self, reason: str | None = None, *, keep_tail_ms: int = 0) -> None:
        if self._audio_ch.closed or self._status == _InferenceStatus.FLUSHED:
            return

        for resampled_frame in self._flush_audio_resampler():
            self._audio_ch.send_nowait(resampled_frame)
        self._audio_ch.send_nowait(
            _AudioTurnDetectorStream._FlushSentinel(reason=reason, keep_tail_ms=keep_tail_ms)
        )
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

    def _emit_prediction(self, probability: float, *, detection_delay: float | None = None) -> None:
        self._event_ch.send_nowait(
            TurnDetectionEvent(
                type="eot_prediction",
                last_speaking_time=time.time(),
                end_of_turn_probability=probability,
                detection_delay=detection_delay,
            )
        )

    async def predict_end_of_turn(self, *, timeout: float | None = None) -> float:
        """Run a warmup inference and wait for a prediction within `timeout`."""
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
            # Reset FSM so next warmup() doesn't see a stale ACTIVE status.
            self._deactivate()
            self._on_inference_stop(reason="predict_end_of_turn timeout")
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
        async for item in self._audio_ch:
            if isinstance(item, _AudioTurnDetectorStream._FlushSentinel):
                await self._on_flush_sentinel(item)
            else:
                await self._on_audio_chunk(item)

    # endregion

    # region: subclass hooks

    def _transport_ready(self) -> bool:
        return True

    @abstractmethod
    async def _run_transport(self) -> None: ...

    def _on_warmup_start(self, request_id: str) -> None:  # noqa: B027
        pass

    def _on_inference_stop(self, *, reason: str | None) -> None:  # noqa: B027
        pass

    async def _on_audio_chunk(self, frame: rtc.AudioFrame) -> None:  # noqa: B027
        pass

    async def _on_flush_sentinel(  # noqa: B027
        self, sentinel: _AudioTurnDetectorStream._FlushSentinel
    ) -> None:
        pass

    def _on_activate(self) -> None:  # noqa: B027
        pass

    def _on_predict_timeout(self) -> None:  # noqa: B027
        pass

    # endregion


# ---------------------------------------------------------------------------
# Turn detection mode (session-level configuration)
# ---------------------------------------------------------------------------


class _TurnDetector(Protocol):
    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def supports_language(self, language: LanguageCode | None) -> bool: ...

    async def predict_end_of_turn(
        self, chat_ctx: ChatContext, *, timeout: float | None = None
    ) -> float: ...


TurnDetectionMode = (
    Literal["stt", "vad", "realtime_llm", "manual"] | _TurnDetector | _AudioTurnDetector
)
"""
The mode of turn detection to use.

- "stt": use speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manually manage the turn detection
- _TurnDetector: use the default mode with the provided turn detector

(default) If not provided, automatically choose the best mode based on
    available models (realtime_llm -> vad -> stt -> manual)
If the needed model (VAD, STT, or RealtimeModel) is not provided, fallback to the default mode.
"""


class EndpointingOptions(TypedDict, total=False):
    """Configuration for endpointing.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).
    """

    mode: Literal["fixed", "dynamic"]
    """Endpointing mode. ``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay. Defaults to ``"fixed"``."""
    min_delay: float
    """Minimum time (s) since last detected speech before declaring the
    user's turn complete. Defaults to ``0.5``."""
    max_delay: float
    """Maximum time (s) the agent waits before terminating the turn.
    Defaults to ``3.0``."""
    alpha: float
    """Exponential moving average coefficient for dynamic endpointing.
    The higher the value, the more weight is given to the history.
    Defaults to ``0.9``. Only applies when mode is ``dynamic``."""


_ENDPOINTING_DEFAULTS: EndpointingOptions = {
    "mode": "fixed",
    "min_delay": 0.5,
    "max_delay": 3.0,
    "alpha": 0.9,
}


class InterruptionOptions(TypedDict, total=False):
    """Configuration for interruption handling.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).

    ``mode`` absent means the session picks the best available strategy.
    """

    enabled: bool
    """Whether interruptions are enabled. Defaults to ``True``."""
    mode: Literal["adaptive", "vad"]
    """Interruption handling strategy. ``"adaptive"`` for ML-based
    detection, ``"vad"`` for simple voice-activity detection.
    Absent means auto-detect."""
    discard_audio_if_uninterruptible: bool
    """Drop buffered audio while the agent speaks and cannot be
    interrupted. Defaults to ``True``."""
    min_duration: float
    """Minimum speech length (s) to register as an interruption.
    Defaults to ``0.5``."""
    min_words: int
    """Minimum word count to consider an interruption (STT only).
    Defaults to ``0``."""
    resume_false_interruption: bool
    """Resume the agent's speech after a false interruption.
    Defaults to ``True``."""
    false_interruption_timeout: float | None
    """Seconds of silence after an interruption before it is
    classified as false. ``None`` disables. Defaults to ``2.0``."""
    backchannel_boundary: float | tuple[float, float] | None
    """Seconds to suppress adaptive interruption handling when the agent
    starts or stops speaking each turn to allow for easier turn correction.
    Use tuple to apply different values for start and end separately.
    ``None`` disables. Defaults to ``(1.0, 3.5)``. End value should be higher
    to account for STT transcript timestamp inaccuracy."""


_INTERRUPTION_DEFAULTS: InterruptionOptions = {
    "enabled": True,
    "discard_audio_if_uninterruptible": True,
    "min_duration": 0.5,
    "min_words": 0,
    "resume_false_interruption": True,
    "false_interruption_timeout": 2.0,
    "backchannel_boundary": (
        1.0,
        3.5,  # higher value for the end as STT timestamps aren't very reliable
    ),
}


class PreemptiveGenerationOptions(TypedDict, total=False):
    """Configuration for preemptive generation."""

    enabled: bool
    """Whether preemptive generation is enabled. Defaults to ``True``."""

    preemptive_tts: bool
    """Whether to also run TTS preemptively before the turn is confirmed.
    When ``False`` (default), only LLM runs preemptively; TTS starts once the
    turn is confirmed and the speech is scheduled."""

    max_speech_duration: float
    """Maximum user speech duration (s) for which preemptive generation
    is attempted. Beyond this threshold, preemptive generation is skipped
    since long utterances are more likely to change and users may expect
    slower responses. Defaults to ``10.0``."""

    max_retries: int
    """Maximum number of preemptive generation attempts per user turn.
    The counter resets when the turn completes. Defaults to ``3``."""


_PREEMPTIVE_GENERATION_DEFAULTS: PreemptiveGenerationOptions = {
    "enabled": True,
    "preemptive_tts": False,
    "max_speech_duration": 10.0,
    "max_retries": 3,
}


class TurnHandlingOptions(TypedDict, total=False):
    """Configuration for the turn handling system.

    Can be passed as a plain dict::

        AgentSession(
            turn_handling={
                "endpointing": {"min_delay": 0.3},
                "interruption": {"enabled": False},
                "preemptive_generation": {"preemptive_tts": True},
            },
        )

    All keys are optional and default to sensible values.
    """

    turn_detection: TurnDetectionMode | None
    """Strategy for deciding when the user has finished speaking.
    Absent means the session auto-selects."""
    endpointing: EndpointingOptions
    """Endpointing configuration. Defaults to ``{"min_delay": 0.5, "max_delay": 3.0}``."""
    interruption: InterruptionOptions
    """Interruption handling configuration. Use ``{"enabled": False}`` to disable."""
    preemptive_generation: PreemptiveGenerationOptions
    """Preemptive generation configuration. Use ``{"enabled": False}`` to disable."""


def _resolve_preemptive_generation(
    config: PreemptiveGenerationOptions | None = None,
) -> PreemptiveGenerationOptions:
    """Fill in defaults for missing keys."""
    if config is None:
        return PreemptiveGenerationOptions(**_PREEMPTIVE_GENERATION_DEFAULTS)
    return PreemptiveGenerationOptions(**{**_PREEMPTIVE_GENERATION_DEFAULTS, **config})


def _resolve_endpointing(config: EndpointingOptions | None = None) -> EndpointingOptions:
    """Fill in defaults for missing keys."""
    if config is None:
        return EndpointingOptions(**_ENDPOINTING_DEFAULTS)
    return EndpointingOptions(**{**_ENDPOINTING_DEFAULTS, **config})


def _resolve_interruption(
    config: InterruptionOptions | None = None,
) -> InterruptionOptions:
    """Fill in defaults for missing keys (``mode`` stays absent if not provided)."""
    if config is None:
        return InterruptionOptions(**_INTERRUPTION_DEFAULTS)
    return InterruptionOptions(**{**_INTERRUPTION_DEFAULTS, **config})


def _migrate_turn_handling(
    min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    discard_audio_if_uninterruptible: NotGivenOr[bool] = NOT_GIVEN,
    min_interruption_duration: NotGivenOr[float] = NOT_GIVEN,
    min_interruption_words: NotGivenOr[int] = NOT_GIVEN,
    allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    resume_false_interruption: NotGivenOr[bool] = NOT_GIVEN,
    agent_false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    preemptive_generation: NotGivenOr[bool] = NOT_GIVEN,
) -> TurnHandlingOptions:
    """Build a TurnHandlingOptions from deprecated keyword arguments."""
    if is_given(agent_false_interruption_timeout):
        false_interruption_timeout = agent_false_interruption_timeout

    result: TurnHandlingOptions = {}

    # endpointing — only include keys that were explicitly provided
    endpointing_opts: EndpointingOptions = {}
    if is_given(min_endpointing_delay):
        endpointing_opts["min_delay"] = min_endpointing_delay
    if is_given(max_endpointing_delay):
        endpointing_opts["max_delay"] = max_endpointing_delay
    if endpointing_opts:
        result["endpointing"] = endpointing_opts

    # interruption — only include keys that were explicitly provided
    interruption: InterruptionOptions = {}
    if allow_interruptions is False:
        interruption["enabled"] = False
    if is_given(discard_audio_if_uninterruptible):
        interruption["discard_audio_if_uninterruptible"] = discard_audio_if_uninterruptible
    if is_given(min_interruption_duration):
        interruption["min_duration"] = min_interruption_duration
    if is_given(min_interruption_words):
        interruption["min_words"] = min_interruption_words
    if is_given(false_interruption_timeout):
        interruption["false_interruption_timeout"] = false_interruption_timeout
    if is_given(resume_false_interruption):
        interruption["resume_false_interruption"] = resume_false_interruption
    if interruption:
        result["interruption"] = interruption

    if is_given(turn_detection):
        result["turn_detection"] = turn_detection

    if is_given(preemptive_generation):
        result["preemptive_generation"] = {"enabled": preemptive_generation}

    return result
