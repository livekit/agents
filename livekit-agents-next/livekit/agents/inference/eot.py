"""Audio end-of-turn detection: the core's decisions (``_core.EotSession``) and transports
(``_core.EotTransport``), cloud with a fallback to the local mini model."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from livekit import rtc
from livekit.local_inference import EOT

from .. import _core, utils
from ..language import LanguageCode
from ..log import logger
from ..metrics import EOTInferenceMetrics
from ..metrics.base import Metadata
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio, is_given
from ..voice.turn import TurnDetectionEvent
from ._utils import get_default_inference_url, get_inference_headers

__all__ = ["TurnDetector", "TurnDetectorStream", "TurnDetectorModels", "TurnDetectorVersions"]

TurnDetectorModels = Literal["turn-detector-v1", "turn-detector-v1-mini"]
TurnDetectorVersions = Literal["v1", "v1-mini"]

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_PREDICTION_TIMEOUT = 1.0

_WARNINGS = {
    "cloud_failed_falling_back": "cloud turn detector failed (%s); falling back to local mini model",
    "cloud_failed_degraded": "cloud turn detector failed (%s); local fallback is disabled, so turn "
    "detection is off for the rest of this stream and turns commit on the endpointing delay",
    "local_failed": "local audio turn detector failed (%s); defaulting to 1.0 and retrying on "
    "next turn",
    "local_prediction_failed": "local audio turn detector prediction failed (%s)",
    "transport_latency": "audio turn detector is behind (%s)",
    "unexpected_server_message": "unexpected message from the turn detector (%s)",
}

Overrides = float | dict[LanguageCode | str, float]


def _overrides(value: NotGivenOr[Overrides]) -> Any:
    if not is_given(value):
        return None
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else value


class TurnDetector(rtc.EventEmitter[Literal["metrics_collected"]]):
    """The end-of-turn model the session asks at each pause, on the user's audio."""

    def __init__(
        self,
        *,
        version: NotGivenOr[TurnDetectorVersions] = NOT_GIVEN,
        unlikely_threshold: NotGivenOr[Overrides] = NOT_GIVEN,
        backchannel_threshold: NotGivenOr[Overrides] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        local_fallback: bool = True,
        http_session: Any = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            version: ``"v1"`` (cloud) or ``"v1-mini"`` (local); by default cloud when hosted or
                in dev mode, else local.
            unlikely_threshold: Overrides of the server's end-of-turn thresholds, one value or
                one per language.
            backchannel_threshold: Overrides of the backchannel thresholds.
            base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            sample_rate: The rate the detector hears.
            local_fallback: Whether ``v1`` may degrade to the local ``v1-mini`` model when the
                gateway fails; without it turns commit on the endpointing delay instead.
            http_session: Unused; the core owns the connection.
            conn_options: The gateway's timeout and retries.
        """
        super().__init__()
        self._base_url = base_url if is_given(base_url) else get_default_inference_url()
        self._api_key = _core.inference_resolve_api_key(api_key if is_given(api_key) else None)
        self._api_secret = _core.inference_resolve_api_secret(
            api_secret if is_given(api_secret) else None
        )
        model, self._cloud, warnings = _core.eot_select(
            version if is_given(version) else None,
            utils.is_hosted(),
            utils.is_dev_mode(),
            self._base_url,
            self._api_key,
            self._api_secret,
            local_fallback,
        )
        for warning in warnings:
            logger.warning(warning)
        self._thresholds = _core.EotThresholdOptions(
            model, _overrides(unlikely_threshold), _overrides(backchannel_threshold)
        )
        self._sample_rate = sample_rate
        self._local_fallback = local_fallback
        self._conn_options = conn_options
        self._warn_overrides()

    @property
    def model(self) -> TurnDetectorModels:
        return self._thresholds.model  # type: ignore[no-any-return]

    @property
    def provider(self) -> str:
        return "livekit"

    def describe_options(self) -> dict[str, Any]:
        """What the session report shows: the model, where it runs and the user's overrides."""
        options: dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "sample_rate": self._sample_rate,
            "local_fallback": self._local_fallback,
        }
        if (overrides := self._thresholds.overrides) is not None:
            options["threshold_overrides"] = overrides
        if (overrides := self._thresholds.backchannel_overrides) is not None:
            options["backchannel_threshold_overrides"] = overrides
        return options

    def _warn_overrides(self) -> None:
        for name, value in (
            ("unlikely_threshold", self._thresholds.overrides),
            ("backchannel_threshold", self._thresholds.backchannel_overrides),
        ):
            if value is not None:
                logger.warning(
                    "a non-default turn detection threshold was provided (%s=%s); the server "
                    "provides calibrated defaults and overriding them may be suboptimal",
                    name,
                    value,
                )

    def update_options(
        self,
        *,
        unlikely_threshold: NotGivenOr[Overrides] = NOT_GIVEN,
        backchannel_threshold: NotGivenOr[Overrides] = NOT_GIVEN,
    ) -> None:
        if is_given(unlikely_threshold):
            self._thresholds.update_overrides(_overrides(unlikely_threshold))
        if is_given(backchannel_threshold):
            self._thresholds.update_backchannel_overrides(_overrides(backchannel_threshold))
        self._warn_overrides()

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._thresholds.lookup(language)  # type: ignore[no-any-return]

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return self._thresholds.lookup_backchannel(language)  # type: ignore[no-any-return]

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return self._thresholds.supports(language)  # type: ignore[no-any-return]

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> TurnDetectorStream:
        return TurnDetectorStream(self, conn_options=conn_options)

    def _transport(self, model: str, conn_options: APIConnectOptions) -> _core.EotTransport:
        if model == "turn-detector-v1":
            return _core.EotTransport.cloud(
                base_url=self._base_url,
                api_key=self._api_key,
                api_secret=self._api_secret,
                thresholds=self._thresholds,
                headers=lambda: list(get_inference_headers().items()),
                sample_rate=self._sample_rate,
                timeout=conn_options.timeout,
                max_retry=conn_options.max_retry,
            )
        return _core.EotTransport.local(EOT(), self._thresholds)


class TurnDetectorStream:
    """One stream of the user's audio, answering ``predict()`` at each pause."""

    def __init__(self, detector: TurnDetector, *, conn_options: APIConnectOptions) -> None:
        self._detector = detector
        self._conn_options = conn_options
        self._session = _core.EotSession(
            detector.model, detector._thresholds, local_fallback=detector._local_fallback
        )
        self._transport = detector._transport(detector.model, conn_options)
        self._waiters: dict[str, asyncio.Future[TurnDetectionEvent]] = {}
        self._resampler: rtc.AudioResampler | None = None
        self._input_format: tuple[int, int] | None = None
        self._degraded = False
        self._swapped = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    @property
    def model(self) -> str:
        return self._session.model  # type: ignore[no-any-return]

    @property
    def provider(self) -> str:
        return self._detector.provider

    @property
    def is_fallback(self) -> bool:
        return self._session.is_fallback  # type: ignore[no-any-return]

    @property
    def is_degraded(self) -> bool:
        """True once the cloud transport is gone with no local model to replace it."""
        return self._session.is_degraded  # type: ignore[no-any-return]

    @property
    def prediction_timeout(self) -> float:
        return DEFAULT_PREDICTION_TIMEOUT

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.unlikely_threshold(language)

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return await self._detector.backchannel_threshold(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return await self._detector.supports_language(language)

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        """Ask for a prediction over the audio so far; supersedes any in flight."""
        session_id = self._transport.session_id
        request_id = utils.shortuuid(f"{session_id}_eot_" if session_id else "turn_request_")
        fut = self._waiters[request_id] = asyncio.get_running_loop().create_future()
        self._apply(self._session.start(time.time(), request_id))
        return fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        self._apply(self._session.cancel(time.time(), timed_out=timed_out))

    def flush(self, reason: str | None = None) -> None:
        if self._resampler is not None:
            for frame in self._resampler.flush():
                self._push(frame)
        self._resampler, self._input_format = None, None
        self._apply(self._session.flush(time.time()))

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        rate = self._detector._sample_rate
        if self._input_format is None:
            self._input_format = (frame.sample_rate, frame.num_channels)
            if frame.sample_rate != rate:
                self._resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    rate,
                    num_channels=frame.num_channels,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )
        elif self._input_format != (frame.sample_rate, frame.num_channels):
            logger.error("a frame with a different audio format was already pushed")
            return
        for out in self._resampler.push(frame) if self._resampler else [frame]:
            self._push(out)

    def _push(self, frame: rtc.AudioFrame) -> None:
        if not self._degraded:
            self._transport.push_frame(frame)

    def end_input(self) -> None:
        self._apply(self._session.end_input(time.time()))
        self._transport.end_input()

    async def aclose(self) -> None:
        self._transport.detach()
        self.end_input()
        await aio.cancel_and_wait(self._task)
        self.cancel_inference()

    def _apply(self, actions: list[dict[str, Any]]) -> None:
        for action in actions:
            kind = action.pop("type")
            if kind == "resolve":
                if (fut := self._waiters.pop(action.pop("id"), None)) and not fut.done():
                    fut.set_result(TurnDetectionEvent(type="eot_prediction", **action))
            elif kind == "start_inference":
                self._transport.run_inference(action["id"])
            elif kind == "flush":
                self._transport.flush()
            elif kind == "swap_to_local":
                self._transport.detach()
                self._transport = self._detector._transport(self.model, self._conn_options)
                self._swapped.set()
            elif kind == "degrade":
                self._degraded = True
            elif kind == "warn":
                warning = _WARNINGS.get(action["warning"], action["warning"] + " (%s)")
                logger.warning(warning, action.get("reason"))

    async def _run(self) -> None:
        while not self._degraded:
            self._swapped.clear()
            transport = self._transport
            events = asyncio.create_task(self._events(transport))
            run = asyncio.ensure_future(transport.run())
            swapped = asyncio.create_task(self._swapped.wait())
            try:
                await asyncio.wait({run, swapped}, return_when=asyncio.FIRST_COMPLETED)
                if run.done() and (exc := run.exception()) is not None:
                    self._apply(self._session.transport_failed(time.time(), str(exc)))
                    if self._transport is transport:
                        return
                elif not swapped.done():
                    return
            finally:
                await aio.cancel_and_wait(events, run, swapped)

    async def _events(self, transport: _core.EotTransport) -> None:
        while (ev := await transport.recv_event()) is not None:
            kind = ev.pop("type")
            if kind == "prediction":
                self._apply(self._session.resolve(time.time(), ev.pop("id"), **ev))
            elif kind == "metrics":
                metadata = Metadata(
                    model_name=ev.pop("model_name"), model_provider=ev.pop("model_provider")
                )
                self._detector.emit(
                    "metrics_collected", EOTInferenceMetrics(**ev, metadata=metadata)
                )
            elif kind == "warn":
                self._apply([{"type": "warn", **ev}])
