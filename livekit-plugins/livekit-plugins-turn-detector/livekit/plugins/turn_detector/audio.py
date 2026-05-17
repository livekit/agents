"""Unified audio EOT detector with cloud → local fallback."""

from __future__ import annotations

import asyncio
import contextlib
import os
import weakref
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
    _normalize_user_threshold,
)

from .languages import CLOUD_LANGUAGES, LOCAL_LANGUAGES
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


def _materialize_thresholds(
    backend: _Backend,
    user_threshold: float | dict[str, float] | None,
) -> dict[str, float]:
    base = CLOUD_LANGUAGES if backend == "cloud" else LOCAL_LANGUAGES
    if user_threshold is None:
        return dict(base)
    if isinstance(user_threshold, dict):
        return {lang: user_threshold.get(lang, default) for lang, default in base.items()}
    return dict.fromkeys(base, user_threshold)


def _materialize_local_thresholds(cloud_thresholds: dict[str, float]) -> dict[str, float]:
    # Per-language rescale: local = local_default * (cloud / cloud_default).
    # Preserves the user's ratio across fallback.
    return {
        lang: LOCAL_LANGUAGES[lang] * (cloud_t / CLOUD_LANGUAGES[lang])
        for lang, cloud_t in cloud_thresholds.items()
        if lang in LOCAL_LANGUAGES and lang in CLOUD_LANGUAGES and CLOUD_LANGUAGES[lang] != 0
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

        normalized = _normalize_user_threshold(unlikely_threshold)
        opts = TurnDetectorOptions(
            sample_rate=sample_rate,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            conn_options=conn_options,
            thresholds=_materialize_thresholds(resolved_backend, normalized),
        )
        super().__init__(opts=opts)

        self._backend: _Backend = resolved_backend
        self._http_session = http_session
        self._stream_ref: weakref.ref[_AudioTurnDetectorStreamImpl] | None = None

    @property
    def model(self) -> str:
        # Effective backend, so post-fallback metrics report the real model.
        return _WIRE_MODEL[self._effective_backend()]

    @property
    def backend(self) -> _Backend:
        return self._effective_backend()

    def _effective_backend(self) -> _Backend:
        if self._stream_ref is not None:
            stream = self._stream_ref()
            if stream is not None:
                return stream.backend
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
        self._stream_ref = weakref.ref(stream)
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

        self._transport: _AudioTurnDetectionTransport
        if backend == "cloud":
            self._transport = _CloudTransport(
                detector=detector,
                opts=opts,
                http_session=http_session,
                conn_options=conn_options,
            )
        else:
            self._transport = _LocalTransport(opts=opts)

        # super().__init__ starts _main_task → _run_transport, so bind first.
        super().__init__(detector=detector, opts=opts)
        self._transport.bind(self)

    @property
    def backend(self) -> _Backend:
        return self._backend

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    # region: FSM hook dispatch

    def _transport_ready(self) -> bool:
        return self._transport.transport_ready()

    def _on_warmup_start(self, request_id: str) -> None:
        self._transport.on_warmup_start(request_id)

    def _on_inference_stop(self, *, reason: str | None) -> None:
        self._transport.on_inference_stop(reason=reason)

    def _on_activate(self) -> None:
        self._transport.on_activate()

    async def _on_audio_chunk(self, frame) -> None:  # type: ignore[no-untyped-def]
        await self._transport.on_audio_chunk(frame)

    async def _on_flush_sentinel(self, sentinel: _AudioTurnDetectorStream._FlushSentinel) -> None:
        await self._transport.on_flush_sentinel(sentinel)

    # endregion

    async def _run_transport(self) -> None:
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
        self._transport.close_nowait()
        self._opts = replace(
            self._opts,
            thresholds=_materialize_local_thresholds(self._opts.thresholds),
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
        if self._active_request_fut is not None and not self._active_request_fut.done():
            with contextlib.suppress(asyncio.InvalidStateError):
                self._active_request_fut.set_result(1.0)
        self._emit_prediction(1.0)

    def _on_predict_timeout(self) -> None:
        # Cloud predict timeout = transport failure; promote local for the session.
        if self._backend == "cloud":
            self._fall_back_to_local(reason=asyncio.TimeoutError("predict_end_of_turn"))

    async def aclose(self) -> None:
        self._transport.close_nowait()
        await super().aclose()


class _AudioEotPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        logger.info(
            "audio EOT model is bundled with the plugin (no download step). "
            "If the native lib is missing, run `make fetch-audio-eot`."
        )


Plugin.register_plugin(_AudioEotPlugin())
