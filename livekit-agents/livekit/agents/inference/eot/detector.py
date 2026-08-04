"""Audio end-of-turn detector with cloud → local fallback."""

from __future__ import annotations

from dataclasses import replace

import aiohttp

from ... import utils
from ...language import LanguageCode
from ...log import logger
from ...types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ...utils import is_given
from .._utils import get_default_inference_url
from .base import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _BaseStreamingTurnDetector,
    _BaseStreamingTurnDetectorStream,
    _StreamingTurnDetectionTransport,
)
from .languages import ThresholdOptions, TurnDetectorModels, TurnDetectorVersions
from .transports import _CloudTransport, _CloudTransportOptions, _LocalTransport

__all__ = ["TurnDetector"]


class TurnDetector(_BaseStreamingTurnDetector):
    def __init__(
        self,
        *,
        version: NotGivenOr[TurnDetectorVersions] = NOT_GIVEN,
        unlikely_threshold: NotGivenOr[float | dict[LanguageCode | str, float]] = NOT_GIVEN,
        backchannel_threshold: NotGivenOr[float | dict[LanguageCode | str, float]] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        auto = not is_given(version)
        resolved_version: TurnDetectorVersions = (
            version
            if is_given(version)
            else ("v1" if (utils.is_hosted() or utils.is_dev_mode()) else "v1-mini")
        )
        resolved_model: TurnDetectorModels = (
            "turn-detector-v1" if resolved_version == "v1" else "turn-detector-v1-mini"
        )

        cloud_opts: _CloudTransportOptions | None = None

        if resolved_version == "v1":
            lk_base_url = utils.resolve_env_var(
                base_url,
                "LIVEKIT_INFERENCE_URL",
                default=get_default_inference_url(),
            )
            lk_api_key = utils.resolve_env_var(
                api_key, "LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_API_KEY", default=""
            )
            lk_api_secret = utils.resolve_env_var(
                api_secret,
                "LIVEKIT_INFERENCE_API_SECRET",
                "LIVEKIT_API_SECRET",
                default="",
            )
            missing: list[str] = []
            if not lk_base_url:
                missing.append("LIVEKIT_INFERENCE_URL")
            if not lk_api_key:
                missing.append("LIVEKIT_API_KEY")
            if not lk_api_secret:
                missing.append("LIVEKIT_API_SECRET")
            if missing:
                if auto:
                    logger.warning(
                        "LIVEKIT_INFERENCE_URL is set but %s missing; "
                        "falling back to the turn-detector-v1-mini model",
                        ", ".join(missing),
                    )
                    resolved_model = "turn-detector-v1-mini"
                else:
                    raise ValueError(
                        f"TurnDetector(version='v1') requires "
                        f"{', '.join(missing)} (env or constructor argument)."
                    )
            else:
                cloud_opts = _CloudTransportOptions(
                    base_url=lk_base_url,
                    api_key=lk_api_key,
                    api_secret=lk_api_secret,
                    conn_options=conn_options,
                )

        opts = TurnDetectorOptions(
            sample_rate=sample_rate,
            thresholds=ThresholdOptions(resolved_model, unlikely_threshold, backchannel_threshold),
        )
        super().__init__(opts=opts)

        self._model: TurnDetectorModels = resolved_model
        self._cloud_opts = cloud_opts
        self._http_session = http_session

        self._warn_threshold_override()

    @property
    def model(self) -> TurnDetectorModels:
        return self._model

    def _warn_threshold_override(self) -> None:
        thresholds = self._opts.thresholds
        if is_given(overrides := thresholds.overrides):
            logger.warning(
                "a non-default turn detection threshold was provided "
                "(unlikely_threshold=%s); the server provides calibrated defaults and "
                "overriding them may be suboptimal",
                overrides,
            )
        if is_given(bc_overrides := thresholds.backchannel_overrides):
            logger.warning(
                "a non-default backchannel threshold was provided "
                "(backchannel_threshold=%s); the server provides calibrated defaults and "
                "overriding them may be suboptimal",
                bc_overrides,
            )

    def update_options(
        self,
        *,
        unlikely_threshold: NotGivenOr[float | dict[LanguageCode | str, float]] = NOT_GIVEN,
        backchannel_threshold: NotGivenOr[float | dict[LanguageCode | str, float]] = NOT_GIVEN,
    ) -> None:
        if is_given(unlikely_threshold):
            self._opts.thresholds.update_overrides(unlikely_threshold)
        if is_given(backchannel_threshold):
            self._opts.thresholds.update_backchannel_overrides(backchannel_threshold)
        self._warn_threshold_override()

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _BaseStreamingTurnDetectorStream:
        cloud_opts = (
            replace(self._cloud_opts, conn_options=conn_options)
            if self._cloud_opts is not None
            else None
        )

        transport: _StreamingTurnDetectionTransport
        if self._model == "turn-detector-v1":
            assert cloud_opts is not None, "turn-detector-v1 requires cloud_opts"
            transport = _CloudTransport(
                detector=self,
                opts=self._opts,
                cloud_opts=cloud_opts,
                http_session=self._http_session,
            )
        else:
            transport = _LocalTransport(opts=self._opts)

        return _BaseStreamingTurnDetectorStream(
            detector=self,
            opts=self._opts,
            transport=transport,
            model=self._model,
        )
