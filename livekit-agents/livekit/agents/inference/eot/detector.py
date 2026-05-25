"""Audio end-of-turn detector with cloud → local fallback."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import Literal

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
from ...utils.aio import cancel_and_wait
from ...voice.turn import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _AudioTurnDetector,
    _AudioTurnDetectorStream,
)
from .languages import materialize_thresholds, rescale_for_local_fallback
from .transports import (
    _AudioTurnDetectionTransport,
    _CloudTransport,
    _CloudTransportOptions,
    _LocalTransport,
)

__all__ = ["AudioTurnDetector"]


# Wire-level model id sent to the gateway. Decoupled from the public `backend`
# kwarg so we don't leak gateway routing names into the API.
_Backend = Literal["cloud", "local"]
_WIRE_MODEL: dict[_Backend, str] = {
    "cloud": "eot-audio",
    "local": "eot-audio-mini",
}


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
            else ("cloud" if os.environ.get("LIVEKIT_REMOTE_EOT_URL") else "local")
        )

        cloud_opts: _CloudTransportOptions | None = None

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
        stream = _AudioTurnDetectorStreamImpl(
            detector=self,
            opts=self._opts,
            cloud_opts=cloud_opts,
            backend=self._backend,
            http_session=self._http_session,
        )
        self._streams.add(stream)
        return stream


class _AudioTurnDetectorStreamImpl(_AudioTurnDetectorStream):
    def __init__(
        self,
        *,
        detector: AudioTurnDetector,
        opts: TurnDetectorOptions,
        cloud_opts: _CloudTransportOptions | None,
        backend: _Backend,
        http_session: aiohttp.ClientSession | None,
    ) -> None:
        self._backend: _Backend = backend
        self._cloud_opts = cloud_opts
        self._http_session = http_session
        self._is_fallback = False
        self._warned_cloud_failure = False
        self._warned_local_failure = False
        self._transport_task: asyncio.Task[None] | None = None
        self._fallback_cancel_pending = False

        transport: _AudioTurnDetectionTransport
        if backend == "cloud":
            assert cloud_opts is not None, "cloud backend requires cloud_opts"
            transport = _CloudTransport(
                detector=detector,
                opts=opts,
                cloud_opts=cloud_opts,
                http_session=http_session,
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
        while True:
            task = asyncio.create_task(self._transport.run())
            self._transport_task = task
            try:
                await task
                return
            except asyncio.CancelledError:
                # _fall_back_to_local cancels the cloud task to break out of
                # the dead session and continue on the new local transport.
                if self._fallback_cancel_pending:
                    self._fallback_cancel_pending = False
                    continue
                # External cancellation (e.g. aclose) — ensure the in-flight
                # transport task is also stopped, then propagate.
                if not task.done():
                    await cancel_and_wait(task)
                raise
            except Exception as e:  # noqa: BLE001 — any cloud error degrades to local
                if self._backend == "cloud":
                    self._fall_back_to_local(reason=e)
                    continue
                self._on_local_failure(reason=e)
                return

    def _fall_back_to_local(self, *, reason: BaseException) -> None:
        if not self._warned_cloud_failure:
            logger.warning(
                "cloud audio eot failed (%s); falling back to local mini model",
                reason,
            )
            self._warned_cloud_failure = True

        self._emit_default_for_inflight()
        self._transport.detach()
        rescaled = replace(
            self._opts,
            thresholds=rescale_for_local_fallback(self._opts.thresholds),
        )
        self._opts = rescaled
        self._transport = _LocalTransport(opts=self._opts)
        self._transport.bind(self)
        self._backend = "local"
        self._is_fallback = True
        # Propagate the swap onto the detector so its model/threshold/backend
        # views (read by metrics + by ``audio_recognition`` when consulting the
        # detector directly) reflect the active local backend instead of the
        # construction-time cloud one.
        detector = self._detector
        if isinstance(detector, AudioTurnDetector):
            detector._backend = "local"
            detector._opts = rescaled
        # If transport.run() is still in flight (e.g. predict timeout while
        # the cloud session was otherwise idle), cancel it so the run loop
        # tears down the dead cloud WS and restarts on the local transport.
        # Without this the orphaned WS lingers until the gateway closes it
        # for inactivity and the ensuing error surfaces as a misleading log.
        task = self._transport_task
        if task is not None and not task.done():
            self._fallback_cancel_pending = True
            task.cancel()

    def _on_local_failure(self, *, reason: BaseException) -> None:
        if not self._warned_local_failure:
            logger.warning(
                "local audio eot mini failed (%s); defaulting to 1.0 and retrying on next turn",
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
