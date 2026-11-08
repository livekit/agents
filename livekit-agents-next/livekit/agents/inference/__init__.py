from __future__ import annotations

import platform
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import JsonValue

from livekit import rtc

from .. import _core
from .._generated import facades_inference as _facades, public_inference as _public
from .._generated.language import LanguageCode
from .._generated.public_inference import *  # noqa: F403
from ..job import get_job_context
from ..log import logger
from ..metrics import EOTInferenceMetrics
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from ..utils import is_dev_mode, is_given, is_hosted
from ..version import __version__

if TYPE_CHECKING:
    from .avatar import AvatarSession, LemonSliceOptions
    from .realtime import RealtimeModel, RealtimeSession


def get_default_inference_url() -> str:
    """``LIVEKIT_INFERENCE_URL``, else the staging or production gateway by the server's URL."""
    return _core.inference_url()


def extract_quota_usage(headers: Mapping[str, str]) -> dict[str, str]:
    """The gateway's quota headers as log fields; empty when it stamped none."""
    return dict(_core.inference_quota_usage(list(headers.items())))


def default_api_key(explicit: str | None) -> str:
    """``explicit``, else ``LIVEKIT_INFERENCE_API_KEY``, else the server's key; ``""`` when
    nothing resolves."""
    return _core.inference_api_key(explicit)


def default_api_secret(explicit: str | None) -> str:
    """The same ladder for the secret."""
    return _core.inference_api_secret(explicit)


def resolve_credentials(api_key: str | None, api_secret: str | None) -> tuple[str, str]:
    """The inference key and secret (``default_api_key``, ``default_api_secret``); raises
    when either is missing."""
    creds = _core.inference_credentials(None, api_key, api_secret)
    return creds.api_key, creds.api_secret


def _resolve_credentials(
    base_url: NotGivenOr[str] | None,
    api_key: NotGivenOr[str] | None,
    api_secret: NotGivenOr[str] | None,
) -> Credentials:
    given = [v if is_given(v) else None for v in (base_url, api_key, api_secret)]
    return _core.inference_credentials(*given)


_USER_AGENT = f"LiveKit Agents/{__version__} (python {platform.python_version()})"


def get_inference_headers(*, inference_class: str | None = None) -> dict[str, str]:
    """The identification headers of an inference request: the SDK, and in a job its room, job,
    agent, worker token, the job's ``inference_headers`` and a fresh session id."""
    ctx = get_job_context(required=False)
    if ctx is None:
        return dict(_core.inference_identity_headers(_USER_AGENT, inference_class))
    return dict(
        _core.inference_identity_headers(
            _USER_AGENT,
            inference_class,
            ctx.job.room.sid,
            ctx.job.id,
            # the local participant exists only once the room is connected
            ctx.agent.sid if ctx.room.isconnected() else None,
            list(ctx.inference_headers.items()),
        )
    )


class VAD(_facades.VAD):
    """Silero voice activity detection."""

    @property
    def provider(self) -> str:
        return "livekit-local-inference"


DEFAULT_SAMPLE_RATE = 16000


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
        selected = _core.eot_select(
            version if is_given(version) else None,
            is_hosted(),
            is_dev_mode(),
            self._base_url,
            self._api_key,
            self._api_secret,
            local_fallback,
        )
        for warning in selected.warnings:
            logger.warning(warning)
        self._sample_rate = sample_rate
        self._local_fallback = local_fallback
        self._detector = _core.InferenceEot(
            selected.model,
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

    def stream(
        self, *, conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN
    ) -> TurnDetectorStream:
        resolved = conn_options if is_given(conn_options) else self._conn_options
        return self._detector.stream(list(get_inference_headers().items()), resolved)

    def _emit_metrics(self, ev: EOTInferenceMetrics) -> None:
        self.emit("metrics_collected", ev)


def _json(overrides: float | dict[str, float]) -> JsonValue:
    return dict[str, JsonValue](overrides) if isinstance(overrides, dict) else overrides


# imported last: these modules use the helpers above
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
    "extract_quota_usage",
    "get_default_inference_url",
    "get_inference_headers",
    "resolve_credentials",
    "VAD",
    "TurnDetector",
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
__all__ += _public.__all__
