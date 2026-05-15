"""Unified audio EOT detector.

One public class ``AudioTurnDetector`` selects between two backends:

- ``eot-audio-cloud`` — WebSocket transport to the LiveKit inference gateway
  (auto-selected when ``LIVEKIT_REMOTE_EOT_URL`` is set in the environment).
- ``eot-audio-mini`` — in-process ctypes inference using the bundled native lib.

When started in cloud mode, any transport failure or `predict_end_of_turn`
timeout swaps the session to local for the remainder of the stream lifetime.
A single warning per failure mode is logged per session. Local failures
emit the default 1.0 prediction and retry on the next turn (no permanent
disable).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import weakref
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
from livekit.agents.voice.turn import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _AudioTurnDetector,
    _AudioTurnDetectorStream,
)

from .languages import CLOUD_LANGUAGES, LOCAL_LANGUAGES
from .log import logger
from .transports import (
    AudioTurnDetectionTransport,
    _CloudTransport,
    _LocalTransport,
    lib_available,
    lib_load_error,
)
from .version import __version__

__all__ = ["AudioTurnDetector"]


_AudioBackend = Literal["eot-audio-cloud", "eot-audio-mini"]


def _resolve_mode(model: _AudioBackend | None) -> tuple[_AudioBackend, bool]:
    """Return (mode, auto). ``auto`` is True iff the caller did not specify."""
    if model is not None:
        return model, False
    if os.environ.get("LIVEKIT_REMOTE_EOT_URL"):
        return "eot-audio-cloud", True
    return "eot-audio-mini", True


class AudioTurnDetector(_AudioTurnDetector):
    """Audio end-of-turn detector.

    Parameters
    ----------
    model:
        Pin a backend explicitly. ``None`` (default) auto-selects:
        ``"eot-audio-cloud"`` when ``LIVEKIT_REMOTE_EOT_URL`` is set,
        ``"eot-audio-mini"`` otherwise.
    unlikely_threshold:
        Override the per-language threshold. Anchored on the cloud
        defaults (0.4 across languages); a custom value is scaled
        multiplicatively onto the local table when fallback engages, so
        a user-tuned operating point survives a cloud → local swap.
    base_url / api_key / api_secret:
        Cloud credentials. ``base_url`` defaults to
        ``LIVEKIT_REMOTE_EOT_URL``; ``api_key`` / ``api_secret`` default
        to ``LIVEKIT_INFERENCE_API_*`` then ``LIVEKIT_API_*``. Only used
        when the resolved backend is cloud.
    """

    def __init__(
        self,
        *,
        model: _AudioBackend | None = None,
        unlikely_threshold: float | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        resolved_model, auto = _resolve_mode(model)

        # Resolve cloud config eagerly so we can downgrade in auto-mode if the
        # creds aren't present. In explicit-cloud mode, missing creds raises.
        lk_base_url = utils.resolve_env_var(base_url, "LIVEKIT_REMOTE_EOT_URL", default="")
        lk_api_key = utils.resolve_env_var(
            api_key, "LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_API_KEY", default=""
        )
        lk_api_secret = utils.resolve_env_var(
            api_secret, "LIVEKIT_INFERENCE_API_SECRET", "LIVEKIT_API_SECRET", default=""
        )

        if resolved_model == "eot-audio-cloud":
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
                        "falling back to eot-audio-mini",
                        ", ".join(missing),
                    )
                    resolved_model = "eot-audio-mini"
                else:
                    raise ValueError(
                        f"AudioTurnDetector(model='eot-audio-cloud') requires "
                        f"{', '.join(missing)} (env or constructor argument)."
                    )

        if resolved_model == "eot-audio-mini" and not lib_available():
            # Only raise for explicit local — auto-fallback (env present, lib
            # missing) is implausible since the lib ships with the plugin, but
            # we raise loudly either way to surface the install problem.
            err = lib_load_error() or RuntimeError("audio EOT native library not loaded")
            raise err

        opts = TurnDetectorOptions(
            sample_rate=sample_rate,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            conn_options=conn_options,
        )
        super().__init__(opts=opts)

        self._mode: _AudioBackend = resolved_model
        self._user_threshold: float | None = unlikely_threshold
        self._http_session = http_session
        self._stream_ref: weakref.ref[_AudioTurnDetectorStreamImpl] | None = None

    @property
    def model(self) -> str:
        # Reflects the *effective* mode — flips after a fallback so metrics
        # and span attributes reflect the actual model serving the request.
        return self._effective_mode()

    def _effective_mode(self) -> _AudioBackend:
        if self._stream_ref is not None:
            stream = self._stream_ref()
            if stream is not None:
                return stream.mode
        return self._mode

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        lang_key = language.language if language is not None else "en"
        mode = self._effective_mode()
        table = LOCAL_LANGUAGES if mode == "eot-audio-mini" else CLOUD_LANGUAGES
        default = table.get(lang_key)
        if default is None:
            return None
        if self._user_threshold is None:
            return default
        # Multiplicative scaling: anchor on cloud_default. If the user picked
        # 0.5 on a 0.4 cloud default, the local fallback uses 0.3 * (0.5/0.4)
        # = 0.375 — same proportional perturbation from each model's default.
        cloud_default = CLOUD_LANGUAGES.get(lang_key, default)
        if cloud_default == 0:
            return default
        return default * (self._user_threshold / cloud_default)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _AudioTurnDetectorStream:
        stream = _AudioTurnDetectorStreamImpl(
            detector=self,
            opts=self._opts,
            mode=self._mode,
            http_session=self._http_session,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        self._stream_ref = weakref.ref(stream)
        return stream


class _AudioTurnDetectorStreamImpl(_AudioTurnDetectorStream):
    """Single concrete stream class that dispatches all FSM hooks to the
    currently-bound transport. On cloud failure (transport raise or predict
    timeout) the cloud transport is closed and replaced with a local one for
    the rest of the session.
    """

    def __init__(
        self,
        *,
        detector: AudioTurnDetector,
        opts: TurnDetectorOptions,
        mode: _AudioBackend,
        http_session: aiohttp.ClientSession | None,
        conn_options: APIConnectOptions,
    ) -> None:
        self._detector_typed = detector  # narrowed type so we can call cloud-only attrs
        self._mode: _AudioBackend = mode
        self._http_session = http_session
        self._conn_options = conn_options
        self._fell_back = False
        self._warned_cloud_failure = False
        self._warned_local_failure = False

        self._transport: AudioTurnDetectionTransport
        if mode == "eot-audio-cloud":
            self._transport = _CloudTransport(
                detector=detector,
                opts=opts,
                http_session=http_session,
                conn_options=conn_options,
            )
        else:
            self._transport = _LocalTransport(opts=opts)

        # super().__init__ kicks off `_main_task` which calls `_run_transport`,
        # so the transport must be bound before that runs.
        super().__init__(detector=detector, opts=opts)
        self._transport.bind(self)

    @property
    def mode(self) -> _AudioBackend:
        return self._mode

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
        if self._mode == "eot-audio-cloud":
            try:
                await self._transport.run()
            except (APIConnectionError, APITimeoutError, APIError) as e:
                self._fall_back_to_local(reason=e)
            except Exception as e:  # noqa: BLE001
                # Any unexpected cloud error is also a fallback trigger —
                # we'd rather degrade to local than tear down the session.
                self._fall_back_to_local(reason=e)
        # After fall-back (or if started local), drain via local; survive its raises.
        if self._mode == "eot-audio-mini":
            try:
                await self._transport.run()
            except Exception as e:  # noqa: BLE001
                self._on_local_failure(reason=e)

    def _fall_back_to_local(self, *, reason: BaseException) -> None:
        if not lib_available():
            # Lib missing — can't promote. Emit 1.0 for this request and let
            # the cloud transport try again on the next turn.
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
        self._transport = _LocalTransport(opts=self._opts)
        self._transport.bind(self)
        self._mode = "eot-audio-mini"
        self._fell_back = True

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
        # Treat a predict_end_of_turn timeout while in cloud mode as a
        # transport-level failure: emit was already 1.0 (by base FSM), now
        # promote local for the rest of the session.
        if self._mode == "eot-audio-cloud":
            self._fall_back_to_local(reason=asyncio.TimeoutError("predict_end_of_turn"))

    async def aclose(self) -> None:
        self._transport.close_nowait()
        await super().aclose()


class _AudioEotPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        # Weights are embedded in the native library, which ships inside
        # the wheel (or is dropped in place by `make fetch-audio-eot`).
        # Nothing to download at runtime.
        logger.info(
            "audio EOT model is bundled with the plugin (no download step). "
            "If the native lib is missing, run `make fetch-audio-eot`."
        )


Plugin.register_plugin(_AudioEotPlugin())
