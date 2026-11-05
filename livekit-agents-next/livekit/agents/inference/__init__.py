"""LiveKit Inference: the gateway's LLM, STT and TTS, its VAD, end-of-turn and interruption
detectors, and the lazily loaded avatar and realtime integrations."""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import JsonValue

from livekit import rtc
from livekit.agents._generated.interruption_errors import (
    InterruptionDetectionError as InterruptionDetectionError,
)

from .. import _bindings, _core
from .._generated import vad
from .._generated.inference_models import (
    InferenceClass as InferenceClass,
    TurnDetectorModels as TurnDetectorModels,
    TurnDetectorVersions as TurnDetectorVersions,
    VADModels as VADModels,
)
from .._generated.language import LanguageCode
from .._generated.overlap import OverlappingSpeechEvent as OverlappingSpeechEvent
from ..job import _server_setting
from ..log import logger
from ..metrics import EOTInferenceMetrics
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from ..utils import is_dev_mode, is_given, is_hosted, shortuuid
from ..version import __version__

if TYPE_CHECKING:
    from .avatar import AvatarSession, LemonSliceOptions
    from .realtime import RealtimeModel, RealtimeSession


HEADER_SESSION_ID = "X-LiveKit-Session-ID"
HEADER_INFERENCE_PROVIDER = "X-LiveKit-Inference-Provider"
HEADER_INFERENCE_PRIORITY = "X-LiveKit-Inference-Priority"


def get_default_inference_url() -> str:
    """``LIVEKIT_INFERENCE_URL``, else the staging or production gateway by the server's URL."""
    return _core.inference_default_url(
        os.environ.get("LIVEKIT_INFERENCE_URL"), _server_setting("LIVEKIT_URL")
    )


def extract_quota_usage(headers: Mapping[str, str]) -> dict[str, str]:
    """The gateway's quota headers as log fields; empty when it stamped none."""
    return dict(_core.inference_quota_usage(list(headers.items())))


def default_api_key(explicit: str | None) -> str:
    """``explicit``, else ``LIVEKIT_INFERENCE_API_KEY``, else the server's key; ``""`` when
    nothing resolves."""
    return _core.inference_resolve_api_key(
        explicit, os.environ.get("LIVEKIT_INFERENCE_API_KEY"), _server_setting("LIVEKIT_API_KEY")
    )


def default_api_secret(explicit: str | None) -> str:
    """The same ladder for the secret."""
    return _core.inference_resolve_api_secret(
        explicit,
        os.environ.get("LIVEKIT_INFERENCE_API_SECRET"),
        _server_setting("LIVEKIT_API_SECRET"),
    )


def resolve_credentials(api_key: str | None, api_secret: str | None) -> tuple[str, str]:
    """The inference key and secret (``default_api_key``, ``default_api_secret``); raises
    when either is missing."""
    key, secret = default_api_key(api_key), default_api_secret(api_secret)
    if not key:
        raise ValueError(
            "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
        )
    if not secret:
        raise ValueError(
            "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental "
            "variable"
        )
    return key, secret


@dataclass(frozen=True)
class Credentials:
    """Where a gateway client connects, and with what."""

    base_url: str
    api_key: str
    api_secret: str

    @classmethod
    def resolve(
        cls,
        base_url: NotGivenOr[str] | None,
        api_key: NotGivenOr[str] | None,
        api_secret: NotGivenOr[str] | None,
    ) -> Credentials:
        key, secret = resolve_credentials(
            api_key if is_given(api_key) else None, api_secret if is_given(api_secret) else None
        )
        url = base_url if is_given(base_url) and base_url else get_default_inference_url()
        return cls(base_url=url, api_key=key, api_secret=secret)


def get_inference_headers(*, inference_class: str | None = None) -> dict[str, str]:
    """The identification headers of an inference request: the SDK, and in a job its room, job,
    agent, worker token, the job's ``inference_headers`` and a fresh session id."""
    headers = {"User-Agent": f"LiveKit Agents/{__version__} (python {platform.python_version()})"}
    if inference_class:
        headers[HEADER_INFERENCE_PRIORITY] = inference_class
    try:
        from ..job import get_job_context

        ctx = get_job_context()
    except RuntimeError:
        return headers
    if isinstance(room_sid := ctx.job.room.sid, str) and room_sid:
        headers["X-LiveKit-Room-ID"] = room_sid
    if isinstance(job_id := ctx.job.id, str) and job_id:
        headers["X-LiveKit-Job-ID"] = job_id
    if worker_token := os.getenv("LIVEKIT_WORKER_TOKEN"):
        headers["X-LiveKit-Worker-Token"] = worker_token
    # the local participant exists only once the room is connected
    if ctx.room.isconnected() and isinstance(agent_sid := ctx.agent.sid, str) and agent_sid:
        headers["X-LiveKit-Agent-ID"] = agent_sid
    headers.update(ctx.inference_headers)
    headers[HEADER_SESSION_ID] = shortuuid("inference_")
    return headers


def _silero_model() -> Any:
    # the host's model, which the core asks for only when the build links no native one
    from livekit.local_inference import VAD

    return VAD()


class VAD(vad.VAD):
    """Silero voice activity detection, the detection machine in the core (``_core.InferenceVad``)
    and the model native to the core when the build links it, else ``livekit-local-inference``'s."""

    def __init__(
        self,
        *,
        model: VADModels = "silero",
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.25,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            model: The VAD model; only ``"silero"``.
            min_speech_duration: Speech needed before a start of speech.
            min_silence_duration: Silence needed before an end of speech.
            prefix_padding_duration: Audio kept before the start of speech.
            max_buffered_speech: The longest speech kept for the end-of-speech event.
            activation_threshold: The probability above which a window is speech.
            deactivation_threshold: The probability below which speech ends;
                ``activation_threshold - 0.15`` by default.
        """
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))
        if model != "silero":
            raise ValueError(f"Unknown VAD model: {model!r}. Supported: 'silero'.")
        if is_given(deactivation_threshold) and deactivation_threshold <= 0:
            raise ValueError("deactivation_threshold must be greater than 0")
        self._model = model
        self._opts = {
            "min_speech_duration": min_speech_duration,
            "min_silence_duration": min_silence_duration,
            "prefix_padding_duration": prefix_padding_duration,
            "max_buffered_speech": max_buffered_speech,
            "activation_threshold": activation_threshold,
            "deactivation_threshold": deactivation_threshold
            if is_given(deactivation_threshold)
            else max(activation_threshold - 0.15, 0.01),
        }
        self._native = _core.InferenceVad(_silero_model, **self._opts)

    def _core_leg(self) -> Any:
        """The core's own detector, unless a subclass replaces its stream."""
        return self if type(self).stream is not VAD.stream else self._native

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "livekit-local-inference"

    @property
    def min_silence_duration(self) -> float:
        return self._opts["min_silence_duration"]

    def stream(self) -> vad.VADStream:
        return _VADStream(self)

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update the options; running streams apply them before their next frame."""
        given = {
            "min_speech_duration": min_speech_duration,
            "min_silence_duration": min_silence_duration,
            "prefix_padding_duration": prefix_padding_duration,
            "max_buffered_speech": max_buffered_speech,
            "activation_threshold": activation_threshold,
            "deactivation_threshold": deactivation_threshold,
        }
        self._opts.update({k: v for k, v in given.items() if is_given(v)})
        self._native.update_options(**self._opts)


class _VADStream(vad.VADStream):
    """The native VAD's stream: frames and flushes in, its events out, through ``pump``."""

    def __init__(self, parent: VAD) -> None:
        self._native = parent._native.stream()
        super().__init__(parent)

    async def _main_task(self) -> None:
        await _bindings.pump(self._native, self.input, self._feed, self.send)

    def _feed(self, item: rtc.AudioFrame | vad.VADStream.Flush) -> None:
        if isinstance(item, self.Flush):
            self._native.flush()
        else:
            self._native.push(item)


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_PREDICTION_TIMEOUT = 1.0


Overrides = float | dict[LanguageCode | str, float]


def _overrides(value: NotGivenOr[Overrides]) -> float | dict[str, float] | None:
    if isinstance(value, NotGiven):
        return None
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else value


def _local_model() -> Any:
    from livekit.local_inference import EOT

    return EOT()


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
            conn_options: The gateway's timeout and retries.
        """
        super().__init__()
        self._base_url = base_url if is_given(base_url) else get_default_inference_url()
        self._api_key = default_api_key(api_key if is_given(api_key) else None)
        self._api_secret = default_api_secret(api_secret if is_given(api_secret) else None)
        model, _, warnings = _core.eot_select(
            version if is_given(version) else None,
            is_hosted(),
            is_dev_mode(),
            self._base_url,
            self._api_key,
            self._api_secret,
            local_fallback,
        )
        for warning in warnings:
            logger.warning(warning)
        self._sample_rate = sample_rate
        self._local_fallback = local_fallback
        self._detector = _core.InferenceEot(
            model,
            _overrides(unlikely_threshold),
            _overrides(backchannel_threshold),
            local_fallback=local_fallback,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            sample_rate=sample_rate,
            local_model=_local_model,
            on_metrics=self._emit_metrics,
        )
        self._conn_options = conn_options
        self._warn_overrides()

    @property
    def model(self) -> TurnDetectorModels:
        return self._detector.model

    @property
    def provider(self) -> str:
        return "livekit"

    def describe_options(self) -> dict[str, JsonValue]:
        """What the session report shows: the model, where it runs and the user's overrides."""
        options: dict[str, JsonValue] = {
            "model": self.model,
            "provider": self.provider,
            "sample_rate": self._sample_rate,
            "local_fallback": self._local_fallback,
        }
        if (overrides := self._detector.overrides) is not None:
            options["threshold_overrides"] = _json(overrides)
        if (overrides := self._detector.backchannel_overrides) is not None:
            options["backchannel_threshold_overrides"] = _json(overrides)
        return options

    def _warn_overrides(self) -> None:
        for name, value in (
            ("unlikely_threshold", self._detector.overrides),
            ("backchannel_threshold", self._detector.backchannel_overrides),
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
            self._detector.update_overrides(_overrides(unlikely_threshold))
        if is_given(backchannel_threshold):
            self._detector.update_backchannel_overrides(_overrides(backchannel_threshold))
        self._warn_overrides()

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._detector.lookup(language)

    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None:
        return self._detector.lookup_backchannel(language)

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return self._detector.supports(language)

    def stream(self, *, conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN) -> _core.EotStream:
        resolved = conn_options if is_given(conn_options) else self._conn_options
        return self._detector.stream(list(get_inference_headers().items()), resolved)

    def _emit_metrics(self, ev: EOTInferenceMetrics) -> None:
        self.emit("metrics_collected", ev)


TurnDetectorStream = _core.EotStream


def _json(overrides: float | dict[str, float]) -> JsonValue:
    return dict[str, JsonValue](overrides) if isinstance(overrides, dict) else overrides


class AdaptiveInterruptionDetector(_core.InterruptionDetector):
    """The gateway model that tells an interruption from a backchannel while the agent speaks.
    ``base_url``, ``api_key`` and ``api_secret`` default to the server's settings."""

    def __new__(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        **options: Any,
    ) -> AdaptiveInterruptionDetector:
        credentials = Credentials.resolve(base_url, api_key, api_secret)
        return super().__new__(
            cls,
            base_url=credentials.base_url,
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            **options,
        )


# The provider modules build on the gateway helpers and detectors above.
from .llm import (  # noqa: E402
    LLM,
    LLMModels,
    LLMStream,
    OpenAIRealtimeModels,
    RealtimeModels,
    XAIRealtimeModels,
)
from .stt import STT, STTModels  # noqa: E402
from .tts import TTS, TTSModels  # noqa: E402


def __getattr__(name: str) -> object:
    # lazy: avatar subclasses voice's AvatarSession and realtime the OpenAI realtime model,
    # and voice imports this package while it initializes
    if name in ("AvatarSession", "LemonSliceOptions"):
        from . import avatar

        return avatar.AvatarSession if name == "AvatarSession" else avatar.LemonSliceOptions
    if name in ("RealtimeModel", "RealtimeSession"):
        from . import realtime

        return realtime.RealtimeModel if name == "RealtimeModel" else realtime.RealtimeSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InferenceClass",
    "Credentials",
    "extract_quota_usage",
    "get_default_inference_url",
    "get_inference_headers",
    "resolve_credentials",
    "VAD",
    "VADModels",
    "TurnDetector",
    "TurnDetectorModels",
    "TurnDetectorVersions",
    "AdaptiveInterruptionDetector",
    "InterruptionDetectionError",
    "OverlappingSpeechEvent",
    "STT",
    "TTS",
    "LLM",
    "AvatarSession",
    "LemonSliceOptions",
    "LLMStream",
    "STTModels",
    "TTSModels",
    "LLMModels",
    "RealtimeModel",
    "RealtimeSession",
    "RealtimeModels",
    "OpenAIRealtimeModels",
    "XAIRealtimeModels",
]
