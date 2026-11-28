from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Coroutine, Mapping
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.typing import Subprotocol

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import FaceModeAPI
from .errors import FaceModeAPIError, FaceModeError, FaceModeNotReadyError, FaceModeProtocolError
from .log import logger
from .models import SUPPORTED_INPUT_PROVIDERS, LiveKitRoom, SessionDetails

_MIN_INPUT_SAMPLE_RATE = 8_000
_MAX_INPUT_SAMPLE_RATE = 48_000
_INGESTION_READY_TIMEOUT_SECONDS = 240.0
_INGESTION_INITIAL_RETRY_SECONDS = 0.25
_INGESTION_MAX_RETRY_SECONDS = 2.0
_WS_OPEN_TIMEOUT_SECONDS = 240.0
_PROTOCOL_RESPONSE_TIMEOUT_SECONDS = 15.0
_RECONNECT_MAX_ATTEMPTS = 2
_RECONNECT_BACKOFF_SECONDS = 0.5
_PREFERRED_SAMPLE_RATES = frozenset(
    {8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000}
)
_DEFAULT_AVATAR_IDENTITY = "facemode-avatar"
_DEFAULT_AVATAR_NAME = "FaceMode Avatar"


class _FaceModeAudioOutput(AudioOutput):
    def __init__(self, owner: AvatarSession):
        super().__init__(
            label="FaceModeAudioOutput",
            next_in_chain=None,
            sample_rate=None,
            capabilities=AudioOutputCapabilities(pause=False),
        )
        self.owner = owner
        self._utterance_started = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await self.owner._drain_control_tasks()
        await super().capture_frame(frame)
        if not self._utterance_started:
            await self.owner._ensure_protocol_started(frame.sample_rate, frame.num_channels)
            await self.owner._send_start_utterance()
            self._utterance_started = True
        await self.owner._send_tts_audio(frame)

    def flush(self) -> None:
        super().flush()
        if self._utterance_started:
            self.owner._schedule_control(self._finish_utterance(), "facemode-end-utterance")
            self._utterance_started = False

    def clear_buffer(self) -> None:
        if self._utterance_started:
            self.owner._schedule_control(
                self.owner._send_cancel_utterance(), "facemode-cancel-utterance"
            )
        self._utterance_started = False

    async def _finish_utterance(self) -> None:
        await self.owner._send_end_utterance()


class AvatarSession(BaseAvatarSession):
    """A FaceMode avatar session for LiveKit Agents."""

    def __init__(
        self,
        *,
        avatar_id: str = "",
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        api_url: NotGivenOr[str | None] = NOT_GIVEN,
        input_provider: str | None = None,
        avatar_participant_identity: NotGivenOr[str | None] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a FaceMode avatar session.

        Args:
            avatar_id: FaceMode avatar ID to render.
            api_key: FaceMode API key. Defaults to `FACEMODE_API_KEY`.
            api_url: FaceMode API base URL. Defaults to `FACEMODE_API_URL` or
                the public FaceMode API.
            input_provider: Optional upstream speech input provider pinned for
                the session. One of deepgram, gemini, gnani, elevenlabs,
                openai, cartesia, sarvam, or custom.
            avatar_participant_identity: LiveKit participant identity for the
                avatar worker.
            avatar_participant_name: Display name for the avatar participant.
            conn_options: Timeout and retry options for FaceMode API requests.

        Raises:
            ValueError: If `input_provider` is not a supported provider.
            FaceModeAPIError: If no API key is provided.
        """
        super().__init__()
        if input_provider is not None and input_provider not in SUPPORTED_INPUT_PROVIDERS:
            raise ValueError(
                "input_provider must be one of: " + ", ".join(sorted(SUPPORTED_INPUT_PROVIDERS))
            )
        self._avatar_id = avatar_id
        self._input_provider = input_provider
        self._avatar_participant_identity = _resolve_optional_string(
            avatar_participant_identity, _DEFAULT_AVATAR_IDENTITY
        )
        self._avatar_participant_name = _resolve_optional_string(
            avatar_participant_name, _DEFAULT_AVATAR_NAME
        )
        self._api = FaceModeAPI(api_key=api_key, api_url=api_url, conn_options=conn_options)
        self._session: SessionDetails | None = None
        self._websocket: ClientConnection | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._control_tasks: set[asyncio.Task[None]] = set()
        self._protocol_started = asyncio.Event()
        self._protocol_response = asyncio.Event()
        self._avatar_ready = asyncio.Event()
        self._session_ended = asyncio.Event()
        self._protocol_error: FaceModeError | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._protocol_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._audio_sample_rate: int | None = None
        self._audio_channels: int | None = None
        self._session_log_marker: str | None = None
        self._sequence = 0
        self._started_once = False
        self._stopped = False

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "facemode"

    @property
    def session_id(self) -> str | None:
        """FaceMode session ID after `start()` succeeds, otherwise `None`."""
        return self._session.session_id if self._session else None

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str | None] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Start the FaceMode session and connect it to the LiveKit room.

        Args:
            agent_session: LiveKit agent session whose audio output should be
                routed through the avatar.
            room: Connected LiveKit room for the agent job.
            livekit_url: LiveKit server URL. Defaults to `LIVEKIT_URL`.
            livekit_api_key: LiveKit API key. Defaults to `LIVEKIT_API_KEY`.
            livekit_api_secret: LiveKit API secret. Defaults to
                `LIVEKIT_API_SECRET`.

        Raises:
            RuntimeError: If the same `AvatarSession` is started more than once.
            FaceModeError: If required credentials are missing or the FaceMode
                session cannot be established.
        """
        if self._started_once:
            raise RuntimeError("AvatarSession.start() called twice; create a new AvatarSession.")
        self._started_once = True

        await super().start(agent_session, room)

        try:
            livekit_url_value = _resolve_env_or_value(livekit_url, "LIVEKIT_URL")
            livekit_api_key_value = _resolve_env_or_value(livekit_api_key, "LIVEKIT_API_KEY")
            livekit_api_secret_value = _resolve_env_or_value(
                livekit_api_secret, "LIVEKIT_API_SECRET"
            )
            if not livekit_url_value or not livekit_api_key_value or not livekit_api_secret_value:
                raise FaceModeError(
                    "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                    "by arguments or environment variables"
                )

            worker_token = self._mint_worker_token(
                room=room,
                livekit_api_key=livekit_api_key_value,
                livekit_api_secret=livekit_api_secret_value,
            )
            livekit_room = LiveKitRoom(url=livekit_url_value, token=worker_token, name=room.name)

            self._session = await self._api.start_session(
                avatar_id=self._avatar_id,
                room=livekit_room,
                room_name=room.name,
                input_provider=self._input_provider,
            )
            self._session = await self._wait_for_ingestion(self._session)
            self._session_log_marker = _sanitize_session_marker(self._session.session_id)
            if (
                self._avatar_participant_identity == _DEFAULT_AVATAR_IDENTITY
                and self._session.avatar_participant_identity
            ):
                self._avatar_participant_identity = self._session.avatar_participant_identity
            self._websocket = await self._connect_websocket()
            self._receiver_task = asyncio.create_task(
                self._receive_events(), name="facemode-ws-events"
            )
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(), name="facemode-ws-keepalive"
            )
            tts = getattr(agent_session, "tts", None)
            if tts is not None:
                await self._ensure_protocol_started(int(tts.sample_rate), int(tts.num_channels))
            agent_session.output.replace_audio_tail(_FaceModeAudioOutput(self))
            logger.info("FaceMode avatar session started session=%s", self._session_log_marker)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self.aclose()
            raise
        except Exception as error:
            with contextlib.suppress(Exception):
                await self.aclose()
            if isinstance(error, (FaceModeError,)):
                raise
            raise FaceModeAPIError(
                f"Unable to start FaceMode avatar session: {error.__class__.__name__}"
            ) from error

    async def aclose(self) -> None:
        """End the FaceMode session and release LiveKit avatar resources."""
        if self._stopped:
            return
        self._stopped = True
        await self._cancel_task(self._keepalive_task)
        self._keepalive_task = None
        await self._cancel_task(self._reconnect_task)
        self._reconnect_task = None
        self._protocol_response.set()
        with contextlib.suppress(Exception):
            await self._drain_control_tasks()

        websocket = self._websocket
        if websocket is not None:
            if self._protocol_started.is_set():
                with contextlib.suppress(Exception):
                    await self._send_control("end_session", self._next_sequence())
                    await asyncio.wait_for(self._session_ended.wait(), timeout=3.0)
            with contextlib.suppress(Exception):
                await websocket.close()

        await self._cancel_task(self._receiver_task)
        self._receiver_task = None
        for task in tuple(self._control_tasks):
            task.cancel()
        if self._control_tasks:
            await asyncio.gather(*tuple(self._control_tasks), return_exceptions=True)
        self._control_tasks.clear()
        self._websocket = None
        with contextlib.suppress(Exception):
            await super().aclose()

    def _mint_worker_token(
        self,
        *,
        room: rtc.Room,
        livekit_api_key: str,
        livekit_api_secret: str,
    ) -> str:
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            local_participant_identity = job_ctx.local_participant_identity
        elif room.isconnected():
            local_participant_identity = room.local_participant.identity
        else:
            raise FaceModeError("failed to get local participant identity")

        if not local_participant_identity:
            raise FaceModeError("failed to get local participant identity")

        grants = api.VideoGrants(
            room_join=True,
            room=room.name,
            can_publish=True,
            can_subscribe=True,
        )
        return (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(grants)
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

    async def _wait_for_ingestion(self, initial_session: SessionDetails) -> SessionDetails:
        """Poll the session resource until its ingestion connection is usable.

        The create-session response usually arrives already ready, in which
        case no polling happens at all. Terminal worker states short-circuit
        before any retry delay.
        """
        session = initial_session
        self._raise_terminal_worker_state(session)
        if not session.ingestion.ready:
            deadline = asyncio.get_running_loop().time() + _INGESTION_READY_TIMEOUT_SECONDS
            delay = _INGESTION_INITIAL_RETRY_SECONDS
            while not session.ingestion.ready:
                if asyncio.get_running_loop().time() >= deadline:
                    raise FaceModeAPIError("Timed out waiting for FaceMode ingestion assignment")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, _INGESTION_MAX_RETRY_SECONDS)
                session = await self._api.get_session(session.session_id, fallback=session)
                self._raise_terminal_worker_state(session)

        if not session.ingestion.url or not session.ingestion.ws_token:
            raise FaceModeAPIError("FaceMode ingestion assignment is missing WebSocket credentials")
        return session

    @staticmethod
    def _raise_terminal_worker_state(session: SessionDetails) -> None:
        if session.worker_status in {"FAILED", "ENDED"}:
            raise FaceModeAPIError(
                f"FaceMode ingestion worker entered {session.worker_status.lower()} state"
            )

    async def _connect_websocket(self) -> ClientConnection:
        """Open the canonical ingestion socket with backend-issued headers intact."""
        if self._session is None:
            raise FaceModeProtocolError("FaceMode session is not initialized")
        ingestion = self._session.ingestion
        if not ingestion.ready or not ingestion.url or not ingestion.ws_token:
            raise FaceModeProtocolError("FaceMode ingestion assignment is not ready")
        return await websockets.connect(
            ingestion.url,
            subprotocols=[Subprotocol(f"facemode.{ingestion.ws_token}")],
            additional_headers=dict(ingestion.headers),
            max_size=2**20,
            ping_interval=None,
            compression=None,
            open_timeout=_WS_OPEN_TIMEOUT_SECONDS,
        )

    async def _request_reconnect_session(self) -> SessionDetails:
        """Refresh the session with a fresh one-time ingestion WebSocket token.

        The request body intentionally carries only the LiveKit room object; no
        prior WebSocket credential or extra secret is replayed.
        """
        if self._session is None:
            raise FaceModeAPIError("FaceMode session is not initialized")
        refreshed = await self._api.reconnect_session(
            self._session.session_id,
            room=self._session.room,
            fallback=self._session,
        )
        if (
            not refreshed.ingestion.ready
            or not refreshed.ingestion.url
            or not refreshed.ingestion.ws_token
        ):
            raise FaceModeAPIError("FaceMode reconnect response is missing WebSocket credentials")
        return refreshed

    def _reconnecting(self) -> bool:
        task = self._reconnect_task
        return task is not None and not task.done()

    def _start_reconnect(self) -> asyncio.Task[None] | None:
        """Single-flight starter shared by every transport-loss signal.

        Returns the in-flight task when one is already running, a fresh task
        when reconnect is allowed, or ``None`` when the session is stopped,
        ended, already in a fatal error state, or never negotiated the
        protocol in the first place.
        """
        task = self._reconnect_task
        if task is not None and not task.done():
            return task
        if (
            self._stopped
            or self._session_ended.is_set()
            or self._protocol_error is not None
            or not self._protocol_started.is_set()
        ):
            return None
        self._protocol_started.clear()
        self._protocol_response.clear()
        self._avatar_ready.clear()
        task = asyncio.create_task(self._reconnect(), name="facemode-ws-reconnect")
        self._reconnect_task = task
        return task

    async def _reconnect(self) -> None:
        """Open a fresh ingestion socket and renegotiate the canonical start.

        Bounded to ``_RECONNECT_MAX_ATTEMPTS`` attempts with linear backoff.
        Every attempt requests a new one-time wsToken; a consumed token is
        never replayed. Exhaustion is recorded as a typed FaceMode error on
        ``_protocol_error`` rather than raised through the receiver task.
        """
        last_error: FaceModeError | None = None
        self._protocol_started.clear()
        self._protocol_response.clear()
        for attempt in range(1, _RECONNECT_MAX_ATTEMPTS + 1):
            if self._stopped or self._session_ended.is_set():
                return
            pending: ClientConnection | None = None
            adopted = False
            try:
                self._protocol_error = None
                refreshed = await self._request_reconnect_session()
                self._session = refreshed
                pending = await self._connect_websocket()
                self._websocket = pending
                async with self._send_lock:
                    await pending.send(self._start_message())
                await self._await_reconnect_started(pending)
                adopted = True
                logger.info(
                    "FaceMode WebSocket reconnected session=%s attempt=%s",
                    self._session_log_marker,
                    attempt,
                )
                return
            except asyncio.CancelledError:
                raise
            except FaceModeError as error:
                last_error = error
            except Exception as error:
                last_error = FaceModeNotReadyError(
                    f"FaceMode reconnect failed: {error.__class__.__name__}"
                )
            finally:
                if pending is not None and not adopted:
                    try:
                        await pending.close()
                    finally:
                        if self._websocket is pending:
                            self._websocket = None
            if attempt < _RECONNECT_MAX_ATTEMPTS:
                await asyncio.sleep(_RECONNECT_BACKOFF_SECONDS * attempt)
        self._protocol_error = last_error or FaceModeNotReadyError(
            "FaceMode reconnect attempts exhausted"
        )
        self._protocol_response.set()
        logger.warning("FaceMode reconnect exhausted: %s", self._protocol_error)

    async def _await_reconnect_started(self, websocket: ClientConnection) -> None:
        """Read the fresh socket until the canonical ``started`` message lands."""
        iterator = websocket.__aiter__()
        while not self._protocol_started.is_set():
            self._raise_protocol_error()
            if self._stopped or self._session_ended.is_set():
                raise FaceModeNotReadyError("FaceMode session ended while reconnecting")
            try:
                raw = await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=_PROTOCOL_RESPONSE_TIMEOUT_SECONDS,
                )
            except StopAsyncIteration as error:
                raise FaceModeNotReadyError("FaceMode WebSocket closed during reconnect") from error
            except asyncio.TimeoutError as error:
                raise FaceModeNotReadyError(
                    "FaceMode did not acknowledge reconnect negotiation"
                ) from error
            self._dispatch_message(raw)

    async def _wait_for_send_ready(self) -> None:
        """Block outbound sends until renegotiation finishes or fails."""
        while not self._protocol_started.is_set():
            self._raise_protocol_error()
            if self._stopped:
                raise FaceModeProtocolError("FaceMode session is stopped")
            if self._session_ended.is_set():
                raise FaceModeProtocolError("FaceMode session has ended")
            await self._protocol_response.wait()

    async def _send_start_utterance(self) -> None:
        await self._send_control("start_utterance", self._next_sequence())

    async def _send_end_utterance(self) -> None:
        if self._protocol_started.is_set():
            await self._send_control("end_utterance", self._next_sequence())

    async def _send_cancel_utterance(self) -> None:
        if self._protocol_started.is_set():
            await self._send_control("cancel_utterance", self._next_sequence())

    async def _send_tts_audio(self, frame: rtc.AudioFrame) -> None:
        await self._ensure_protocol_started(frame.sample_rate, frame.num_channels)
        self._raise_protocol_error()
        if self._websocket is None:
            raise FaceModeProtocolError("FaceMode WebSocket is not connected")
        audio = frame.data.tobytes()
        alignment = 2 * frame.num_channels
        if len(audio) % alignment:
            raise FaceModeProtocolError("LiveKit TTS PCM payload is not channel aligned")
        async with self._send_lock:
            if not self._protocol_started.is_set():
                raise FaceModeProtocolError("FaceMode protocol is not ready")
            await self._websocket.send(audio)

    def _start_message(self) -> str:
        """Canonical ``start`` payload shared by negotiation and reconnect."""
        if self._session is None or self._audio_sample_rate is None or self._audio_channels is None:
            raise FaceModeProtocolError("FaceMode session audio format is not initialized")
        return json.dumps(
            {
                "type": "start",
                "session_id": self._session.session_id,
                "audio_encoding": "pcm_s16le",
                "sample_rate": int(self._audio_sample_rate),
                "channels": int(self._audio_channels),
                "avatar_id": self._avatar_id,
                "metadata": {"source": "livekit-agents"},
            },
            separators=(",", ":"),
        )

    async def _ensure_protocol_started(self, sample_rate: int, channels: int) -> None:
        async with self._protocol_lock:
            self._raise_protocol_error()
            if not self._protocol_started.is_set() and self._reconnecting():
                await self._wait_for_send_ready()
            if self._protocol_started.is_set():
                if self._audio_sample_rate != sample_rate or self._audio_channels != channels:
                    raise FaceModeProtocolError(
                        "LiveKit TTS audio format changed during the FaceMode session"
                    )
                return
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
                raise FaceModeProtocolError("LiveKit TTS sample rate must be an integer")
            if not _MIN_INPUT_SAMPLE_RATE <= sample_rate <= _MAX_INPUT_SAMPLE_RATE:
                raise FaceModeProtocolError(
                    f"LiveKit TTS sample rate must be between {_MIN_INPUT_SAMPLE_RATE} "
                    f"and {_MAX_INPUT_SAMPLE_RATE}: {sample_rate}"
                )
            if sample_rate not in _PREFERRED_SAMPLE_RATES:
                logger.info("Using uncommon LiveKit TTS sample rate: %s", sample_rate)
            if channels not in (1, 2):
                raise FaceModeProtocolError(f"Unsupported LiveKit TTS channel count: {channels}")
            if self._websocket is None or self._session is None:
                raise FaceModeProtocolError("FaceMode WebSocket is not connected")
            self._audio_sample_rate = sample_rate
            self._audio_channels = channels
            async with self._send_lock:
                await self._websocket.send(self._start_message())
            try:
                await asyncio.wait_for(
                    self._protocol_response.wait(), timeout=_PROTOCOL_RESPONSE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError as error:
                raise FaceModeProtocolError(
                    "FaceMode did not acknowledge protocol negotiation"
                ) from error
            self._raise_protocol_error()
            if not self._protocol_started.is_set():
                raise FaceModeProtocolError("FaceMode protocol negotiation did not start")

    async def _send_control(self, message_type: str, sequence: int) -> None:
        self._raise_protocol_error()
        if not self._protocol_started.is_set() and self._reconnecting():
            await self._wait_for_send_ready()
        if self._websocket is None:
            if self._stopped:
                return
            raise FaceModeProtocolError("FaceMode WebSocket is not connected")
        async with self._send_lock:
            if not self._protocol_started.is_set():
                raise FaceModeProtocolError("FaceMode protocol is not ready")
            await self._websocket.send(
                json.dumps({"type": message_type, "seq": sequence}, separators=(",", ":"))
            )

    async def _send_ping(self) -> None:
        self._raise_protocol_error()
        if not self._protocol_started.is_set() and self._reconnecting():
            await self._wait_for_send_ready()
        if self._websocket is None:
            raise FaceModeProtocolError("FaceMode WebSocket is not connected")
        async with self._send_lock:
            await self._websocket.send(
                json.dumps({"type": "ping", "ts": int(time.time() * 1000)}, separators=(",", ":"))
            )

    async def _keepalive_loop(self) -> None:
        try:
            await self._protocol_response.wait()
            if not self._protocol_started.is_set() or self._protocol_error is not None:
                return
            while not self._stopped:
                await asyncio.sleep(15.0)
                if self._websocket is None or self._stopped:
                    return
                try:
                    await self._send_ping()
                except Exception as error:
                    if not self._stopped:
                        protocol_error = FaceModeProtocolError(
                            f"FaceMode keepalive failed: {error.__class__.__name__}"
                        )
                        self._set_protocol_error(protocol_error)
                        logger.warning("%s", protocol_error)
                    return
        except asyncio.CancelledError:
            raise

    async def _receive_events(self) -> None:
        """Read inbound messages, transparently re-attaching after drops.

        A transport loss after a successful ``start`` triggers the
        single-flight reconnect path instead of a protocol error. Intentional
        close, session ``ended``, fatal protocol errors, and drops before the
        first negotiation keep the original fail-fast behavior.
        """
        try:
            while True:
                websocket = self._websocket
                if websocket is None:
                    return
                transport_error: Exception | None = None
                try:
                    async for raw in websocket:
                        try:
                            fatal = self._dispatch_message(raw)
                        except FaceModeProtocolError as error:
                            self._set_protocol_error(error)
                            logger.warning("%s", error)
                            fatal = True
                        if fatal:
                            break
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    transport_error = error
                if self._stopped or self._session_ended.is_set():
                    return
                if self._protocol_error is not None:
                    return
                task = self._start_reconnect()
                if task is None:
                    if transport_error is not None:
                        protocol_error = FaceModeProtocolError(
                            "FaceMode WebSocket receiver stopped: "
                            f"{transport_error.__class__.__name__}"
                            f"{_websocket_close_details(transport_error)}"
                        )
                    else:
                        protocol_error = FaceModeProtocolError(
                            "FaceMode WebSocket closed unexpectedly"
                            f"{_websocket_close_details(websocket)}"
                        )
                    self._set_protocol_error(protocol_error)
                    logger.warning("%s", protocol_error)
                    return
                await asyncio.wait({task})
                if (
                    self._stopped
                    or self._session_ended.is_set()
                    or self._protocol_error is not None
                    or self._websocket is websocket
                ):
                    return
        except asyncio.CancelledError:
            raise

    def _dispatch_message(self, raw: str | bytes) -> bool:
        """Handle one wire message; return True if the socket must be dropped."""
        if isinstance(raw, bytes):
            return False
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as error:
            raise FaceModeProtocolError("FaceMode WebSocket message was not JSON") from error
        if not isinstance(message, Mapping):
            raise FaceModeProtocolError("FaceMode WebSocket message was not an object")
        message_type = str(message.get("type", ""))
        if message_type == "started":
            self._handle_started(message)
        elif message_type == "audio_ready":
            self._avatar_ready.set()
        elif message_type == "utterance_ended":
            logger.debug(
                "FaceMode utterance ended seq=%s cycles=%s",
                message.get("seq"),
                message.get("cycles"),
            )
        elif message_type == "pong":
            logger.debug("FaceMode keepalive acknowledged")
        elif message_type == "session_ending":
            protocol_error = FaceModeProtocolError(
                f"FaceMode session is ending: {message.get('reason', 'unknown')}"
            )
            self._set_protocol_error(protocol_error)
            logger.warning("%s", protocol_error)
        elif message_type == "ended":
            self._session_ended.set()
            self._protocol_response.set()
        elif message_type == "error":
            protocol_error = FaceModeProtocolError(
                f"{message.get('code', 'INTERNAL_ERROR')}: {message.get('message', 'FaceMode error')}"
            )
            self._set_protocol_error(protocol_error)
            logger.error("FaceMode protocol error: %s", protocol_error)
            if bool(message.get("fatal", False)):
                return True
        else:
            logger.debug("FaceMode WebSocket unhandled message type: %s", message_type)
        return False

    def _handle_started(self, message: Mapping[str, Any]) -> None:
        if self._session is None:
            self._set_protocol_error(FaceModeProtocolError("FaceMode session is not initialized"))
            return
        if str(message.get("session_id", "")) != self._session.session_id:
            self._set_protocol_error(
                FaceModeProtocolError("FaceMode started response session ID did not match")
            )
            return
        if message.get("server_sample_rate") != 48_000 or message.get("server_channels") != 1:
            self._set_protocol_error(
                FaceModeProtocolError(
                    "FaceMode server reported an unsupported canonical audio format"
                )
            )
            return
        self._protocol_started.set()
        self._protocol_response.set()
        logger.info("FaceMode protocol started session=%s", self._session_log_marker)

    def _schedule_control(self, coroutine: Coroutine[Any, Any, None], name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self._control_tasks.add(task)
        task.add_done_callback(self._control_task_done)

    def _control_task_done(self, task: asyncio.Task[None]) -> None:
        self._control_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None or self._stopped:
            return
        if self._reconnecting() or self._start_reconnect() is not None:
            logger.warning("FaceMode control message failed during reconnect: %s", error)
            return
        protocol_error = (
            error
            if isinstance(error, FaceModeProtocolError)
            else FaceModeProtocolError(
                f"FaceMode control message failed: {error.__class__.__name__}"
            )
        )
        self._set_protocol_error(protocol_error)
        logger.warning("%s", protocol_error)

    async def _drain_control_tasks(self) -> None:
        tasks = tuple(self._control_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._raise_protocol_error()

    async def _cancel_task(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _set_protocol_error(self, error: FaceModeError) -> None:
        if self._protocol_error is None:
            self._protocol_error = error
        self._protocol_response.set()

    def _raise_protocol_error(self) -> None:
        if self._protocol_error is not None:
            raise self._protocol_error

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence


def _resolve_optional_string(value: NotGivenOr[str | None], default: str) -> str:
    if utils.is_given(value) and value:
        return value
    return default


def _resolve_env_or_value(value: NotGivenOr[str | None], env_name: str) -> str | None:
    if utils.is_given(value) and value is not None:
        return value
    return os.getenv(env_name)


def _sanitize_session_marker(session_id: str) -> str:
    return "".join(
        char if char.isascii() and (char.isalnum() or char in "_-") else "_" for char in session_id
    )[:96]


def _websocket_close_details(value: Any) -> str:
    if value is None:
        return " (code=unknown)"
    received = getattr(value, "rcvd", None)
    code = getattr(value, "close_code", None) or getattr(value, "code", None)
    reason = getattr(value, "close_reason", None) or getattr(value, "reason", None)
    if received is not None:
        code = code or getattr(received, "code", None)
        reason = reason or getattr(received, "reason", None)
    reason_text = str(reason or "").strip()[:160]
    suffix = f", reason={reason_text}" if reason_text else ""
    return f" (code={code if code is not None else 'unknown'}{suffix})"


__all__ = ["AvatarSession"]
