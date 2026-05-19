"""Unified audio EOT detector with cloud → local fallback."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import Literal

import aiohttp

from livekit.agents import Plugin, utils
from livekit.agents._exceptions import APIConnectionError, APIError, APITimeoutError
from livekit.agents.language import LanguageCode
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.turn import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _AudioTurnDetector,
    _AudioTurnDetectorStream,
)

from .languages import materialize_thresholds, rescale_for_local_fallback
from .log import logger
from .transports import (
    _AudioTurnDetectionTransport,
    _CloudTransport,
    _get_lib_load_error,
    _lib_available,
    _LocalTransport,
)
from .version import __version__

__all__ = ["AudioTurnDetector"]


# Wire-level model id sent to the gateway. Decoupled from the public `backend`
# kwarg so we don't leak gateway routing names into the API.
_Backend = Literal["cloud", "local"]
_WIRE_MODEL: dict[_Backend, str] = {
    "cloud": "eot-audio",
    "local": "eot-audio-mini",
}


def _resolve_backend(backend: NotGivenOr[_Backend]) -> tuple[_Backend, bool]:
    if is_given(backend):
        return backend, False
    if os.environ.get("LIVEKIT_REMOTE_EOT_URL"):
        return "cloud", True
    return "local", True


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
        resolved_backend, auto = _resolve_backend(backend)

        lk_base_url = ""
        lk_api_key = ""
        lk_api_secret = ""

        if resolved_backend == "cloud":
            lk_base_url = utils.resolve_env_var(base_url, "LIVEKIT_REMOTE_EOT_URL", default="")
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
                missing.append("LIVEKIT_REMOTE_EOT_URL")
            if not lk_api_key:
                missing.append("LIVEKIT_API_KEY")
            if not lk_api_secret:
                missing.append("LIVEKIT_API_SECRET")
            if missing:
                if auto:
                    logger.warning(
                        "LIVEKIT_REMOTE_EOT_URL is set but %s missing; "
                        "falling back to local backend",
                        ", ".join(missing),
                    )
                    resolved_backend = "local"
                else:
                    raise ValueError(
                        f"AudioTurnDetector(backend='cloud') requires "
                        f"{', '.join(missing)} (env or constructor argument)."
                    )

        if resolved_backend == "local" and not _lib_available():
            err = _get_lib_load_error() or RuntimeError("audio EOT native library not loaded")
            raise err

        opts = TurnDetectorOptions(
            sample_rate=sample_rate,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            conn_options=conn_options,
            thresholds=materialize_thresholds(unlikely_threshold, resolved_backend),
        )
        super().__init__(opts=opts)

        self._backend: _Backend = resolved_backend
        self._http_session = http_session

    @property
    def model(self) -> str:
        # Reports the backend chosen at construction. After a session-scoped
        # cloud→local fallback, the stream tracks the current backend on
        # ``stream.backend``; the detector view is intentionally stable.
        return _WIRE_MODEL[self._backend]

    @property
    def backend(self) -> _Backend:
        return self._backend

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _AudioTurnDetectorStream:
        stream = _AudioTurnDetectorStreamImpl(
            detector=self,
            opts=self._opts,
            backend=self._backend,
            http_session=self._http_session,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream


class _AudioTurnDetectorStreamImpl(_AudioTurnDetectorStream):
    def __init__(
        self,
        *,
        detector: AudioTurnDetector,
        opts: TurnDetectorOptions,
        backend: _Backend,
        http_session: aiohttp.ClientSession | None,
        conn_options: APIConnectOptions,
    ) -> None:
        self._backend: _Backend = backend
        self._http_session = http_session
        self._conn_options = conn_options
        self._is_fallback = False
        self._warned_cloud_failure = False
        self._warned_local_failure = False

        transport: _AudioTurnDetectionTransport
        if backend == "cloud":
            transport = _CloudTransport(
                detector=detector,
                opts=opts,
                http_session=http_session,
                conn_options=conn_options,
            )
        else:
            transport = _LocalTransport(opts=opts)

        super().__init__(detector=detector, opts=opts, transport=transport)

    @property
    def backend(self) -> _Backend:
        return self._backend

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    async def _run(self) -> None:
        if self._backend == "cloud":
            try:
                await self._transport.run()
            except (APIConnectionError, APITimeoutError, APIError) as e:
                self._fall_back_to_local(reason=e)
            except Exception as e:  # noqa: BLE001 — any cloud error degrades to local
                self._fall_back_to_local(reason=e)
        # After fallback (or if started local), drain via local; survive its raises.
        if self._backend == "local":
            try:
                await self._transport.run()
            except Exception as e:  # noqa: BLE001
                self._on_local_failure(reason=e)

    def _fall_back_to_local(self, *, reason: BaseException) -> None:
        if not _lib_available():
            # Lib missing — emit 1.0 and let cloud retry next turn.
            if not self._warned_cloud_failure:
                logger.warning(
                    "cloud audio EOT failed (%s) and local mini lib is unavailable; "
                    "defaulting to 1.0 for current and future failures",
                    reason,
                )
                self._warned_cloud_failure = True
            self._emit_default_for_inflight()
            return

        if not self._warned_cloud_failure:
            logger.warning(
                "cloud audio EOT failed (%s); falling back to local mini model",
                reason,
            )
            self._warned_cloud_failure = True

        self._emit_default_for_inflight()
        self._transport.detach()
        self._opts = replace(
            self._opts,
            thresholds=rescale_for_local_fallback(self._opts.thresholds),
        )
        self._transport = _LocalTransport(opts=self._opts)
        self._transport.bind(self)
        self._backend = "local"
        self._is_fallback = True

    def _on_local_failure(self, *, reason: BaseException) -> None:
        if not self._warned_local_failure:
            logger.warning(
                "local audio EOT mini failed (%s); defaulting to 1.0 and retrying on next turn",
                reason,
            )
            self._warned_local_failure = True
        self._emit_default_for_inflight()

    def _emit_default_for_inflight(self) -> None:
        request_id = self._preemptive_request_id
        if request_id is not None:
            self._handle_prediction(request_id, 1.0)

    def _on_predict_timeout(self) -> None:
        # Cloud predict timeout = transport failure; promote local for the session.
        if self._backend == "cloud":
            self._fall_back_to_local(reason=asyncio.TimeoutError("predict_end_of_turn"))

    async def aclose(self) -> None:
        self._transport.detach()
        await super().aclose()


class _AudioEotPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        pass


Plugin.register_plugin(_AudioEotPlugin())
