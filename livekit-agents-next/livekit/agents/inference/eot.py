"""Audio end-of-turn detection: the detector's options here, its stream the core's
(``_core.EotStream``), cloud with a fallback to the local mini model."""

from __future__ import annotations

from typing import Any, Literal

from livekit import rtc

from .. import _core, utils
from .._generated.language import LanguageCode
from ..llm._realtime import _native_connect_timeout
from ..log import logger
from ..metrics import EOTInferenceMetrics, Metadata
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from ._utils import get_default_inference_url, get_inference_headers

__all__ = ["TurnDetector", "TurnDetectorStream", "TurnDetectorModels", "TurnDetectorVersions"]

TurnDetectorModels = Literal["turn-detector-v1", "turn-detector-v1-mini"]
TurnDetectorVersions = Literal["v1", "v1-mini"]

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_PREDICTION_TIMEOUT = 1.0


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
    ) -> _core.EotStream:
        return _core.EotStream(
            self.model,
            self._thresholds,
            local_fallback=self._local_fallback,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            headers=lambda: list(get_inference_headers().items()),
            sample_rate=self._sample_rate,
            timeout=_native_connect_timeout(conn_options) if self._cloud else 0.0,
            max_retry=conn_options.max_retry,
            on_metrics=self._emit_metrics,
        )

    def _emit_metrics(self, ev: dict[str, Any]) -> None:
        metadata = Metadata(
            model_name=ev.pop("model_name"), model_provider=ev.pop("model_provider")
        )
        ev.pop("type", None)
        self.emit("metrics_collected", EOTInferenceMetrics(**ev, metadata=metadata))


TurnDetectorStream = _core.EotStream
