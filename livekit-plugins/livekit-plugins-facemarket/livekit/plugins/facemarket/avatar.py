from __future__ import annotations

import asyncio
import datetime as dt
import inspect
import json
import uuid
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any, Literal, Protocol, TypeAlias

from livekit import rtc
from livekit.agents import AgentSession
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.api import AccessToken, VideoGrants

from .api import FaceMarketAPI
from .exceptions import FaceMarketSessionError, SessionReadyTimeoutError
from .log import logger
from .schemas import SessionInfo, StartSessionRequest

DEFAULT_TOKEN_TTL = dt.timedelta(hours=4)
DEFAULT_READY_TIMEOUT = 30.0
RENDERER_IDENTITY = "avatar-renderer"
LEGACY_RENDERER_IDENTITY_PREFIX = "avatar-StreamId("
COORDINATOR_IDENTITY = "avatar-coordinator"
SUPPORTED_EVENTS = {
    "session_ready",
    "session_state_change",
    "idle_trigger",
    "session_closing",
    "error",
}

EVENT_CALLBACK = Callable[..., Any]

AGENT_SESSION_EVENT_NAME: TypeAlias = Literal[
    "user_state_changed",
    "agent_state_changed",
    "user_input_transcribed",
    "conversation_item_added",
    "agent_false_interruption",
    "overlapping_speech",
    "function_tools_executed",
    "metrics_collected",
    "session_usage_updated",
    "speech_created",
    "error",
    "close",
]


class _FaceMarketAPIClient(Protocol):
    async def start_session(self, request: StartSessionRequest) -> SessionInfo: ...

    async def stop_session(self, session_id: str) -> None: ...


def _state_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    candidate = getattr(value, "value", None)
    if isinstance(candidate, str):
        return candidate.lower()
    candidate = getattr(value, "name", None)
    if isinstance(candidate, str):
        return candidate.lower()
    return str(value).lower()


class AvatarSession:
    def __init__(
        self,
        avatar_id: str,
        platform_api_key: str,
        livekit_url: str,
        livekit_api_key: str,
        livekit_api_secret: str,
        api_client: _FaceMarketAPIClient | None = None,
    ) -> None:
        if not avatar_id:
            raise ValueError("avatar_id is required")
        if not platform_api_key:
            raise ValueError("platform_api_key is required")
        if not livekit_url:
            raise ValueError("livekit_url is required")
        if not livekit_api_key:
            raise ValueError("livekit_api_key is required")
        if not livekit_api_secret:
            raise ValueError("livekit_api_secret is required")

        self._avatar_id = avatar_id
        self._platform_api_key = platform_api_key
        self._livekit_url = livekit_url
        self._livekit_api_key = livekit_api_key
        self._livekit_api_secret = livekit_api_secret
        self._api = api_client or FaceMarketAPI(platform_api_key=platform_api_key)

        self._room: rtc.Room | None = None
        self._agent_session: AgentSession | None = None
        self._session_info: SessionInfo | None = None
        self._ready_event = asyncio.Event()
        self._callbacks: dict[str, list[EVENT_CALLBACK]] = defaultdict(list)
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._room_data_handler: Callable[..., Any] | None = None
        self._track_subscribed_handler: Callable[..., Any] | None = None
        self._agent_handlers: list[tuple[AGENT_SESSION_EVENT_NAME, Callable[..., Any]]] = []
        self._is_prompt_mode = False
        self._agent_is_speaking = False
        self._response_audio_active = False
        self._user_is_speaking = False
        self._ready_emitted = False
        self._stopping = False

    @property
    def session_id(self) -> str | None:
        if self._session_info is None:
            return None
        return self._session_info.session_id

    def on(self, event: str) -> Callable[[EVENT_CALLBACK], EVENT_CALLBACK]:
        if event not in SUPPORTED_EVENTS:
            raise ValueError(f"unsupported event: {event}")

        def decorator(callback: EVENT_CALLBACK) -> EVENT_CALLBACK:
            self._callbacks[event].append(callback)
            return callback

        return decorator

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        ready_timeout: float = DEFAULT_READY_TIMEOUT,
    ) -> SessionInfo:
        if self._session_info is not None:
            raise FaceMarketSessionError("AvatarSession is already started")

        self._room = room
        self._agent_session = agent_session
        self._ready_event = asyncio.Event()
        self._is_prompt_mode = False
        self._agent_is_speaking = False
        self._response_audio_active = False
        self._user_is_speaking = False
        self._ready_emitted = False
        self._stopping = False

        try:
            agent_identity = room.local_participant.identity
            room_name = room.name
        except Exception as exc:
            self._room = None
            self._agent_session = None
            raise FaceMarketSessionError(
                "room must already be connected before calling avatar.start()"
            ) from exc

        self._attach_room_handler(room)
        self._attach_track_handler(room)
        self._attach_agent_handlers(agent_session)

        try:
            renderer_token = self._make_participant_token(
                identity=RENDERER_IDENTITY,
                room_name=room_name,
                attributes={ATTRIBUTE_PUBLISH_ON_BEHALF: agent_identity},
            )
            coordinator_token = self._make_participant_token(
                identity=COORDINATOR_IDENTITY,
                room_name=room_name,
                hidden=True,
            )

            self._session_info = await self._start_remote_session(
                agent_identity=agent_identity,
                room_name=room_name,
                renderer_token=renderer_token,
                coordinator_token=coordinator_token,
            )

            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=ready_timeout)
            except TimeoutError as exc:
                session_id = self.session_id or "<unknown>"
                await self._safe_stop_remote_session()
                self._reset_runtime_state()
                raise SessionReadyTimeoutError(
                    f"FaceMarket session did not become ready in {ready_timeout:.1f}s "
                    f"(session_id={session_id})"
                ) from exc

            return self._session_info
        except Exception:
            if self._session_info is not None:
                await self._safe_stop_remote_session()
            self._reset_runtime_state()
            raise

    async def interrupt(self) -> None:
        if self._room is None:
            return
        await self._publish_event("control.interrupt")

    async def stop(self) -> None:
        if self._session_info is None:
            self._reset_runtime_state()
            return
        self._stopping = True
        try:
            await self._stop_remote_session(self._session_info.session_id)
        finally:
            await self._reset_runtime_state_async()

    async def _emit(self, event: str, *args: Any) -> None:
        callbacks = list(self._callbacks.get(event, ()))
        for callback in callbacks:
            try:
                result = callback(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("callback failed for event %s", event)

    def _create_bg_task(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)

        def _done(completed: asyncio.Task[Any]) -> None:
            self._bg_tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("background task failed")

        task.add_done_callback(_done)

    def _attach_room_handler(self, room: rtc.Room) -> None:
        def _on_data_received(packet: Any, *_: Any) -> None:
            self._create_bg_task(self._handle_data_packet(packet))

        self._room_data_handler = _on_data_received
        room.on("data_received", _on_data_received)

    def _attach_track_handler(self, room: rtc.Room) -> None:
        def _on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            participant_identity = getattr(participant, "identity", None)
            track_kind = getattr(track, "kind", None)
            publication_kind = getattr(publication, "kind", None)
            logger.info(
                "track subscribed participant=%s track_kind=%s publication_kind=%s",
                participant_identity,
                track_kind,
                publication_kind,
            )
            is_renderer = participant_identity == RENDERER_IDENTITY or (
                isinstance(participant_identity, str)
                and participant_identity.startswith(LEGACY_RENDERER_IDENTITY_PREFIX)
            )
            if not is_renderer:
                return
            if (
                track_kind != rtc.TrackKind.KIND_VIDEO
                and publication_kind != rtc.TrackKind.KIND_VIDEO
            ):
                return
            self._create_bg_task(self._mark_ready())

        self._track_subscribed_handler = _on_track_subscribed
        room.on("track_subscribed", _on_track_subscribed)

    async def _mark_ready(self) -> None:
        if self._ready_emitted:
            return
        self._ready_emitted = True
        self._ready_event.set()
        await self._emit("session_ready")

    async def _handle_data_packet(self, packet: Any) -> None:
        if self._stopping:
            return

        participant = getattr(packet, "participant", None)
        participant_identity = getattr(participant, "identity", None)
        if participant_identity not in (None, COORDINATOR_IDENTITY):
            return

        raw = getattr(packet, "data", b"")
        if not raw:
            return

        try:
            payload = json.loads(raw)
        except Exception:
            logger.debug("ignored non-json data packet")
            return

        event = payload.get("event")
        data = payload.get("data") or {}
        logger.info(
            "received FaceMarket data packet participant=%s event=%s data=%s",
            participant_identity,
            event,
            data,
        )

        if event == "session.state":
            await self._emit("session_state_change", data.get("state", ""))
            return

        if event == "session.closing":
            await self._emit("session_closing")
            return

        if event == "system.idleTrigger":
            self._is_prompt_mode = True
            await self._emit("idle_trigger")
            return

        if event == "session.stop":
            await self.stop()
            return

        if event == "error":
            await self._emit("error", data)

    def _attach_agent_handlers(self, agent_session: AgentSession) -> None:
        def _user_state_changed(event: Any) -> None:
            new_state = _state_name(getattr(event, "new_state", None))
            if new_state == "speaking" and not self._user_is_speaking:
                self._user_is_speaking = True
                self._create_bg_task(self._publish_event("input.voice.start"))
                return

            if self._user_is_speaking and new_state != "speaking":
                self._user_is_speaking = False
                self._create_bg_task(self._publish_event("input.voice.finish"))

        def _agent_state_changed(event: Any) -> None:
            new_state = _state_name(getattr(event, "new_state", None))
            if new_state == "speaking" and not self._agent_is_speaking:
                self._agent_is_speaking = True
                self._create_bg_task(
                    self._publish_response_audio_start(source="agent_state_changed")
                )
                return

            if self._agent_is_speaking and new_state != "speaking":
                self._agent_is_speaking = False
                self._create_bg_task(
                    self._publish_response_audio_finish(source="agent_state_changed")
                )

        def _speech_created(event: Any) -> None:
            speech_handle = getattr(event, "speech_handle", None)
            self._create_bg_task(self._bridge_speech_playout(speech_handle))

        def _user_input_transcribed(event: Any) -> None:
            transcript = getattr(event, "transcript", None) or getattr(event, "text", None)
            if not transcript:
                return

            message = {
                "text": transcript,
            }
            language = getattr(event, "language", None)
            if language:
                message["language"] = language

            event_name = (
                "input.asr.final"
                if bool(getattr(event, "is_final", False))
                else "input.asr.partial"
            )
            self._create_bg_task(self._publish_event(event_name, message))

        bindings: list[tuple[AGENT_SESSION_EVENT_NAME, Callable[..., Any]]] = [
            ("user_state_changed", _user_state_changed),
            ("agent_state_changed", _agent_state_changed),
            ("user_input_transcribed", _user_input_transcribed),
            ("speech_created", _speech_created),
        ]

        for event_name, handler in bindings:
            agent_session.on(event_name, handler)
            self._agent_handlers.append((event_name, handler))

    async def _bridge_speech_playout(self, speech_handle: Any) -> None:
        await self._publish_response_audio_start(source="speech_created")
        if speech_handle is not None and hasattr(speech_handle, "wait_for_playout"):
            try:
                await speech_handle.wait_for_playout()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("failed while waiting for speech playout")
        await self._publish_response_audio_finish(source="speech_created")

    async def _publish_response_audio_start(self, *, source: str) -> None:
        if self._response_audio_active:
            logger.info("FaceMarket response audio already active, skip start source=%s", source)
            return
        self._response_audio_active = True
        dc_event = "response.audio.promptStart" if self._is_prompt_mode else "response.audio.start"
        logger.info("FaceMarket response audio start source=%s event=%s", source, dc_event)
        await self._publish_event(dc_event)

    async def _publish_response_audio_finish(self, *, source: str) -> None:
        if not self._response_audio_active:
            logger.info("FaceMarket response audio is not active, skip finish source=%s", source)
            return
        dc_event = (
            "response.audio.promptFinish" if self._is_prompt_mode else "response.audio.finish"
        )
        self._response_audio_active = False
        self._is_prompt_mode = False
        logger.info("FaceMarket response audio finish source=%s event=%s", source, dc_event)
        await self._publish_event(dc_event)

    async def _publish_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        if self._room is None:
            return
        payload: dict[str, Any] = {
            "event": event,
            "requestId": str(uuid.uuid4()),
        }
        if data is not None:
            payload["data"] = data
        encoded = json.dumps(payload).encode("utf-8")
        logger.info("publishing FaceMarket data event=%s data=%s", event, data or {})
        await self._room.local_participant.publish_data(encoded, reliable=True)

    def _make_participant_token(
        self,
        *,
        identity: str,
        room_name: str,
        hidden: bool = False,
        attributes: dict[str, str] | None = None,
    ) -> str:
        token = AccessToken(self._livekit_api_key, self._livekit_api_secret)
        token = token.with_identity(identity)
        token = token.with_kind("agent")
        token = token.with_ttl(DEFAULT_TOKEN_TTL)
        token = token.with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                hidden=hidden,
            )
        )
        if attributes:
            token = token.with_attributes(attributes)
        return token.to_jwt()

    async def _start_remote_session(
        self,
        *,
        agent_identity: str,
        room_name: str,
        renderer_token: str,
        coordinator_token: str,
    ) -> SessionInfo:
        return await self._api.start_session(
            StartSessionRequest(
                avatar_id=self._avatar_id,
                agent_identity=agent_identity,
                livekit_url=self._livekit_url,
                room_name=room_name,
                renderer_token=renderer_token,
                coordinator_token=coordinator_token,
            )
        )

    async def _stop_remote_session(self, session_id: str) -> None:
        await self._api.stop_session(session_id)

    async def _safe_stop_remote_session(self) -> None:
        session_id = self.session_id
        if not session_id:
            return
        try:
            await self._stop_remote_session(session_id)
        except Exception:
            logger.exception("failed to stop FaceMarket session during cleanup")

    def _detach_handlers(self) -> None:
        if (
            self._room is not None
            and self._room_data_handler is not None
            and hasattr(self._room, "off")
        ):
            try:
                self._room.off("data_received", self._room_data_handler)
            except Exception:
                logger.debug("failed to detach room data handler", exc_info=True)
        self._room_data_handler = None

        if (
            self._room is not None
            and self._track_subscribed_handler is not None
            and hasattr(self._room, "off")
        ):
            try:
                self._room.off("track_subscribed", self._track_subscribed_handler)
            except Exception:
                logger.debug("failed to detach room track handler", exc_info=True)
        self._track_subscribed_handler = None

        if self._agent_session is not None and hasattr(self._agent_session, "off"):
            for event_name, handler in self._agent_handlers:
                try:
                    self._agent_session.off(event_name, handler)
                except Exception:
                    logger.debug("failed to detach agent handler %s", event_name, exc_info=True)
        self._agent_handlers.clear()

    def _reset_runtime_state(self) -> None:
        self._detach_handlers()
        self._room = None
        self._agent_session = None
        self._session_info = None
        self._ready_event = asyncio.Event()
        self._is_prompt_mode = False
        self._agent_is_speaking = False
        self._response_audio_active = False
        self._user_is_speaking = False
        self._ready_emitted = False
        self._stopping = False

    async def _reset_runtime_state_async(self) -> None:
        current = asyncio.current_task()
        pending_tasks = [task for task in self._bg_tasks if task is not current and not task.done()]
        self._reset_runtime_state()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
