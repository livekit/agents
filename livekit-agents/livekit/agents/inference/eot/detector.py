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
    _WIRE_MODEL,
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _AudioTurnDetectionTransport,
    _AudioTurnDetector,
    _AudioTurnDetectorStream,
    _Backend,
)
from .languages import materialize_thresholds
from .transports import _CloudTransport, _CloudTransportOptions, _LocalTransport

__all__ = ["AudioTurnDetector"]


class AudioTurnDetector(_AudioTurnDetector):
    """Audio end-of-turn detector. Auto-selects cloud when
    ``LIVEKIT_REMOTE_EOT_URL`` is set, local otherwise."""

    def __init__(
        self,
        *,
        backend: NotGivenOr[_Backend] = NOT_GIVEN,
        unlikely_threshold: NotGivenOr[float | dict[LanguageCode | str, float]] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        # auto = caller didn't pin a backend; missing cloud creds warn-and-fall-back
        # instead of raising.
        auto = not is_given(backend)
        resolved_backend: _Backend = (
            backend
            if is_given(backend)
            else ("cloud" if (utils.is_hosted() or utils.is_dev_mode()) else "local")
        )

        cloud_opts: _CloudTransportOptions | None = None

        if resolved_backend == "cloud":
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
                        "falling back to local backend",
                        ", ".join(missing),
                    )
                    resolved_backend = "local"
                else:
                    raise ValueError(
                        f"AudioTurnDetector(backend='cloud') requires "
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
            thresholds=materialize_thresholds(unlikely_threshold, resolved_backend),
        )
        super().__init__(opts=opts)

        self._backend: _Backend = resolved_backend
        self._cloud_opts = cloud_opts
        self._http_session = http_session

    @property
    def model(self) -> str:
        return _WIRE_MODEL[self._backend]

    @property
    def backend(self) -> _Backend:
        return self._backend

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _AudioTurnDetectorStream:
        # Per-stream override for conn_options (e.g. an override at call site)
        # is layered on top of the detector-level cloud options.
        cloud_opts = (
            replace(self._cloud_opts, conn_options=conn_options)
            if self._cloud_opts is not None
            else None
        )

        transport: _AudioTurnDetectionTransport
        if self._backend == "cloud":
            assert cloud_opts is not None, "cloud backend requires cloud_opts"
            transport = _CloudTransport(
                detector=self,
                opts=self._opts,
                cloud_opts=cloud_opts,
                http_session=self._http_session,
            )
        else:
            transport = _LocalTransport(opts=self._opts)

        return _AudioTurnDetectorStream(
            detector=self,
            opts=self._opts,
            transport=transport,
            backend=self._backend,
        )
