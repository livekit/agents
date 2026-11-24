from __future__ import annotations

import base64
import os
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from pydantic import BaseModel, JsonValue
from typing_extensions import TypedDict

import livekit.rtc as rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    http,
)
from livekit.agents.utils import is_given
from livekit.agents.utils.media import AudioConverter
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarStart,
    AvatarWebSocket,
    ForwardingAvatarSession,
    Keepalive,
    WebSocketTarget,
)
from livekit.agents.voice.events import AgentStateChangedEvent

from .log import logger

if TYPE_CHECKING:
    from livekit.agents import VoiceSession

DEFAULT_API_URL = "https://api.liveavatar.com/v1/sessions"
SAMPLE_RATE = 24000
KEEP_ALIVE_INTERVAL = 60
# speech goes out in chunks: a short first one, then longer ones
_FIRST_CHUNK = 0.6
_CHUNK = 1.0

VideoQuality = Literal["very_high", "high", "medium", "low"]


class LiveAvatarException(Exception):
    """Exception for LiveAvatar errors"""


_DataT = TypeVar("_DataT")


class _Answer(BaseModel, Generic[_DataT]):
    data: _DataT


class _Token(BaseModel):
    session_id: str
    session_token: str


class _Started(BaseModel):
    ws_url: str


class _Stopped(BaseModel):
    code: int


_AnswerT = TypeVar("_AnswerT", bound=BaseModel)


class _ServerEvent(TypedDict, total=False):
    type: str
    event_id: str
    source_event_id: str | None
    state: str
    previous_state: str
    new_state: str
    error: JsonValue
    warning: JsonValue


def _keep_alive() -> dict[str, JsonValue]:
    return {"type": "session.keep_alive", "event_id": str(uuid.uuid4())}


class _Events(AvatarWebSocket[_ServerEvent, dict[str, JsonValue]]):
    """LiveAvatar's LITE-mode event socket: nothing goes before the session says it is
    connected, then a keep-alive at once and after each quiet ``KEEP_ALIVE_INTERVAL``."""

    awaits_ready = True

    def __init__(self, ws_url: str, handle: Callable[[_ServerEvent], None]) -> None:
        super().__init__("LiveAvatar")
        self._ws_url = ws_url
        self._handle = handle
        self._connected = False
        self.keepalive = Keepalive(every=KEEP_ALIVE_INTERVAL, frame=_keep_alive())

    async def connect(self, conn_options: APIConnectOptions) -> WebSocketTarget:
        return WebSocketTarget(self._ws_url)

    def on_message(self, message: _ServerEvent) -> None:
        if (
            not self._connected
            and message.get("type") == "session.state_updated"
            and message.get("state") == "connected"
        ):
            self._connected = True
            self.ready()
            self.send(_keep_alive())
        self._handle(message)


@dataclass
class _Utterance:
    event_id: str | None = None
    interrupt_id: str | None = None
    playback_position: float = 0.0
    started: bool = False
    finished: Literal["ended", "interrupted"] | None = None


class AvatarSession(ForwardingAvatarSession):
    """A LiveAvatar avatar session"""

    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        is_sandbox: NotGivenOr[bool] = NOT_GIVEN,
        video_quality: NotGivenOr[VideoQuality] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            avatar_participant_identity=avatar_participant_identity,
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        avatar_id = avatar_id if is_given(avatar_id) else os.getenv("LIVEAVATAR_AVATAR_ID", "")
        if not avatar_id:
            raise LiveAvatarException("avatar_id must be set")
        self._avatar_id = avatar_id
        api_key = api_key if is_given(api_key) else os.getenv("LIVEAVATAR_API_KEY", "")
        if not api_key:
            raise LiveAvatarException("api_key or LIVEAVATAR_API_KEY must be set")
        self._api_key = api_key
        self._api_url = api_url or DEFAULT_API_URL
        self._is_sandbox = is_sandbox if is_given(is_sandbox) else False
        self._video_quality = video_quality

        self._token: _Token | None = None
        self._event_socket: _Events | None = None
        self._agent_session_listened: VoiceSession | None = None
        self._audio_playing = False
        self._utterances: deque[_Utterance] = deque()
        self._current_utterance: _Utterance | None = None
        self._discard_chunk()

    @property
    def provider(self) -> str:
        return "liveavatar"

    async def create_session(self, ctx: AvatarStart) -> str:
        payload: dict[str, JsonValue] = {
            "mode": "LITE",
            "avatar_id": self._avatar_id,
            "is_sandbox": self._is_sandbox,
            "livekit_config": {
                "livekit_room": ctx.room.name,
                "livekit_url": ctx.livekit_url,
                "livekit_client_token": await ctx.livekit_token(),
            },
        }
        if is_given(self._video_quality):
            payload["video_quality"] = self._video_quality
        token = (
            await self._post("/token", payload, {"X-API-KEY": self._api_key}, _Answer[_Token])
        ).data
        started = await self._post(
            "/start",
            {"session_id": token.session_id},
            {"Authorization": f"Bearer {token.session_token}"},
            _Answer[_Started],
        )
        self._token = token
        self._event_socket = _Events(started.data.ws_url, self._handle_server_event)
        ctx.agent_session.on("agent_state_changed", self._on_agent_state_changed)
        self._agent_session_listened = ctx.agent_session
        logger.info(f"LiveAvatar session started: {token.session_id}")
        return token.session_id

    async def end_session(self, session_id: str) -> None:
        assert self._token is not None
        stopped = await self._post(
            "/stop",
            {"session_id": session_id, "reason": "USER_DISCONNECTED"},
            {"Authorization": f"Bearer {self._token.session_token}"},
            _Stopped,
        )
        if stopped.code <= 200:
            logger.info(f"LiveAvatar session stopped: {session_id}")

    async def _post(
        self,
        path: str,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        answer: type[_AnswerT],
    ) -> _AnswerT:
        async with http.http_session().post(
            self._api_url + path, headers=headers, json=payload, conn_options=self._conn_options
        ) as response:
            await response.raise_for_status()
            return answer.model_validate_json(await response.text())

    def send_event(self, msg: dict[str, JsonValue]) -> None:
        """Send a LITE-mode command once the session is connected; dropped before ``start()``
        created the session and after ``aclose()``."""
        if self._event_socket is not None:
            self._event_socket.send(msg)

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        if ev.new_state == "idle":
            self.send_event({"type": "agent.stop_listening", "event_id": str(uuid.uuid4())})

    async def aclose(self) -> None:
        if (listened := self._agent_session_listened) is not None:
            self._agent_session_listened = None
            listened.off("agent_state_changed", self._on_agent_state_changed)
        try:
            if self._event_socket is not None:
                await self._event_socket.aclose()
        finally:
            await super().aclose()

    async def receive(self) -> None:
        assert self._event_socket is not None
        try:
            await self._event_socket.run(self._conn_options)
        except APIError as e:
            # the sandbox ends its sessions by closing the socket after a minute
            if self._is_sandbox and e.kind == "connection":
                raise APIError(
                    "The LiveAvatar Sandbox connection surpassed the 1 minute limit", kind="quota"
                ) from e
            raise

    async def forward(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, rtc.AudioFrame):
            if self._current_utterance is None:
                self._current_utterance = _Utterance()
                self._utterances.append(self._current_utterance)
            self._audio_playing = True
            self._push_frames(self._converter.push(frame))
            return

        self._push_frames(self._converter.flush())
        self._flush_chunk()
        utterance, self._current_utterance = self._current_utterance, None
        self._first_chunk = True
        if utterance is None:
            return
        if utterance.event_id is None:
            utterance.finished = "ended"
            self._finish_utterances()
            return
        self.send_event({"type": "agent.speak_end", "event_id": str(uuid.uuid4())})
        self.send_event({"type": "agent.start_listening", "event_id": str(uuid.uuid4())})

    async def clear(self) -> None:
        self._discard_chunk()
        if self._audio_playing:
            interrupt_id = str(uuid.uuid4())
            for utterance in self._utterances:
                if utterance.finished is None and utterance.interrupt_id is None:
                    utterance.interrupt_id = interrupt_id
            self.send_event({"type": "agent.interrupt", "event_id": interrupt_id})
            self._current_utterance = None
        self._audio_playing = False

    def _discard_chunk(self) -> None:
        self._converter = AudioConverter(SAMPLE_RATE, 1)
        self._chunk: list[bytes] = []
        self._chunk_duration = 0.0
        self._first_chunk = True

    def _push_frames(self, frames: list[rtc.AudioFrame]) -> None:
        for frame in frames:
            if not frame.data:
                continue
            self._chunk.append(frame.data.tobytes())
            self._chunk_duration += frame.duration
            if self._chunk_duration >= (_FIRST_CHUNK if self._first_chunk else _CHUNK):
                self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self._chunk:
            return
        utterance = self._current_utterance
        assert utterance is not None
        if utterance.event_id is None:
            utterance.event_id = str(uuid.uuid4())
        self.send_event(
            {
                "type": "agent.speak",
                "event_id": utterance.event_id,
                "audio": base64.b64encode(b"".join(self._chunk)).decode("utf-8"),
            }
        )
        utterance.playback_position += self._chunk_duration
        self._chunk = []
        self._chunk_duration = 0.0
        self._first_chunk = False

    def _handle_server_event(self, event: _ServerEvent) -> None:
        event_type = event.get("type")
        if event_type == "session.state_updated":
            logger.debug(f"LiveAvatar session state: {event.get('state')}")
        elif event_type == "agent.speak_interrupted":
            self._handle_agent_speak_interrupted(event)
        elif event_type == "agent.speak_ended":
            self._handle_agent_speak_ended(event)
        elif event_type == "agent.speak_started":
            self._handle_agent_speak_started(event)
        elif event_type == "agent.state_updated":
            self._handle_agent_state_updated(event)
        elif event_type == "agent.audio_buffer_cleared":
            self._handle_agent_speak_interrupted(event)
        elif event_type == "agent.audio_buffer_appended":
            pass  # one ack per speak chunk, too frequent to log
        elif event_type == "agent.audio_buffer_committed":
            # command ack; playback follows speak_* / agent.state_updated
            logger.debug(f"LiveAvatar {event_type}")
        elif event_type == "error":
            logger.error(f"LiveAvatar error: {event.get('error', event)}")
        elif event_type == "warning":
            logger.warning(f"LiveAvatar warning: {event.get('warning', event)}")
        else:
            logger.debug(f"Unhandled LiveAvatar event: {event_type}")

    def _handle_agent_state_updated(self, event: _ServerEvent) -> None:
        new_state = event.get("new_state", event.get("state"))
        logger.debug(f"LiveAvatar agent state: {event.get('previous_state')} -> {new_state}")
        if new_state == "talking":
            self._handle_agent_speak_started(event)
        # Idle/listening transitions have no source_event_id in LITE mode. Only the
        # correlated speak_ended event can safely finish a queued utterance.

    def _utterance_for_event(self, event: _ServerEvent) -> _Utterance | None:
        source = event.get("source_event_id")
        if source is None:
            return None
        return next((item for item in self._utterances if item.event_id == source), None)

    def _finish_utterances(self) -> None:
        # A later response may arrive before an older acknowledgement. Preserve the
        # QueueAudioOutput's FIFO reports without allowing duplicates to advance it.
        while self._utterances and self._utterances[0].finished is not None:
            utterance = self._utterances.popleft()
            interrupted = utterance.finished == "interrupted"
            self.playback_finished(
                0.0 if interrupted else utterance.playback_position, interrupted=interrupted
            )
        self._audio_playing = any(
            item.finished is None and item.interrupt_id is None for item in self._utterances
        )

    def _handle_agent_speak_interrupted(self, event: _ServerEvent) -> None:
        if event.get("type") == "agent.audio_buffer_cleared":
            source = event.get("source_event_id")
            if source is None:
                return
            interrupted = [item for item in self._utterances if item.interrupt_id == source]
        else:
            utterance = self._utterance_for_event(event)
            interrupted = [utterance] if utterance is not None else []
        if not interrupted:
            return
        for utterance in interrupted:
            if utterance.finished is None:
                utterance.finished = "interrupted"
        self._finish_utterances()

    def _handle_agent_speak_ended(self, event: _ServerEvent) -> None:
        utterance = self._utterance_for_event(event)
        if utterance is None or utterance.finished is not None:
            return
        utterance.finished = "ended"
        self._finish_utterances()

    def _handle_agent_speak_started(self, event: _ServerEvent) -> None:
        utterance = self._utterance_for_event(event)
        if (
            utterance is None
            or utterance.started
            or utterance.finished is not None
            or utterance.interrupt_id is not None
        ):
            return
        utterance.started = True
        self.playback_started()
