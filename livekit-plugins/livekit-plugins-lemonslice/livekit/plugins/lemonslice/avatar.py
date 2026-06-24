from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from typing import Any

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
)
from livekit.agents.voice import room_io
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import LemonSliceAPI, LemonSliceException
from .meeting import (
    JoinMeetingResult,
    MeetingAudioInput,
    MeetingChatRelay,
    meeting_room_options,
    stream_meeting_relay,
)

SAMPLE_RATE = 16000
_MEETING_BROADCAST_IDENTITY = "lemonslice-meeting-broadcast"
_AVATAR_AGENT_IDENTITY = "lemonslice-avatar-agent"
_AVATAR_AGENT_NAME = "lemonslice-avatar-agent"


class AvatarSession(BaseAvatarSession):
    """A LemonSlice avatar session"""

    def __init__(
        self,
        *,
        agent_id: NotGivenOr[str] = NOT_GIVEN,
        agent_image_url: NotGivenOr[str] = NOT_GIVEN,
        agent_prompt: NotGivenOr[str] = NOT_GIVEN,
        agent_idle_prompt: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout: NotGivenOr[int] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._agent_id = agent_id
        self._agent_image_url = agent_image_url
        self._agent_prompt = agent_prompt
        self._agent_idle_prompt = agent_idle_prompt
        self._idle_timeout = idle_timeout
        self._api_url = api_url
        self._api_key = api_key
        self._http_session: aiohttp.ClientSession | None = None
        self._conn_options = conn_options
        self._extra_payload: NotGivenOr[dict[str, Any]] = kwargs if kwargs else NOT_GIVEN

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME

        self._session_id: str | None = None
        self._agent_session: AgentSession | None = None

        # Cached from start() for join_meeting() token minting.
        self._livekit_url: str | None = None
        self._livekit_api_key: str | None = None
        self._livekit_api_secret: str | None = None
        self._livekit_room: str | None = None

        # External meeting state (Zoom, Meet, Teams); set by join_meeting().
        self._meeting_bot_id: str | None = None
        self._meeting_chat: MeetingChatRelay | None = None
        self._meeting_relay_stop: asyncio.Event | None = None
        self._meeting_relay_task: asyncio.Task[None] | None = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "lemonslice"

    async def start(  # type: ignore[override]
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> str:
        await super().start(agent_session, room)
        self._agent_session = agent_session

        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise LemonSliceException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        self._livekit_url = livekit_url
        self._livekit_api_key = livekit_api_key
        self._livekit_api_secret = livekit_api_secret
        self._livekit_room = room.name

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity
        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar agent to publish audio and video on behalf of your local agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        # Rebind audio output BEFORE the slow upstream HTTP call so
        # subsequent generations are routed to the (about-to-arrive)
        # avatar identity immediately. wait_remote_track buffers
        # frames until the video track shows up, so nothing is lost
        # in the gap. replace_audio_tail keeps the TranscriptSynchronizer
        # / RecorderAudioOutput chain intact across hot swaps.
        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                clear_buffer_timeout=None,
                wait_playback_start=True,
            ),
        )

        async with LemonSliceAPI(
            api_url=self._api_url,
            api_key=self._api_key,
            conn_options=self._conn_options,
            session=self._http_session,
        ) as lemonslice_api:
            session_id = await lemonslice_api.start_agent_session(
                agent_id=self._agent_id,
                agent_image_url=self._agent_image_url,
                agent_prompt=self._agent_prompt,
                agent_idle_prompt=self._agent_idle_prompt,
                idle_timeout=self._idle_timeout,
                livekit_url=livekit_url,
                livekit_token=livekit_token,
                extra_payload=self._extra_payload,
            )

        self._session_id = session_id
        return session_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def _mint_broadcast_token(self) -> str:
        if not self._livekit_api_key or not self._livekit_api_secret or not self._livekit_room:
            raise LemonSliceException("call start() before join_meeting()")

        grants = api.VideoGrants(
            room_join=True,
            room=self._livekit_room,
            can_subscribe=True,
            can_publish=False,
            can_publish_data=False,
        )
        return (
            api.AccessToken(self._livekit_api_key, self._livekit_api_secret)
            .with_identity(_MEETING_BROADCAST_IDENTITY)
            .with_name(_MEETING_BROADCAST_IDENTITY)
            .with_grants(grants)
            .with_ttl(timedelta(hours=4))
            .to_jwt()
        )

    async def join_meeting(
        self,
        meeting_url: str,
        *,
        bot_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> JoinMeetingResult:
        """Send this avatar into an external video meeting (Zoom, Meet, or Teams).

        Call after :meth:`start` and before :meth:`room_options` / ``session.start``.
        """
        if not self._session_id or self._agent_session is None or not self._livekit_url:
            raise LemonSliceException("call start() before join_meeting()")

        broadcast_token = self._mint_broadcast_token()

        async with LemonSliceAPI(
            api_url=self._api_url,
            api_key=self._api_key,
            conn_options=self._conn_options,
            session=self._http_session,
        ) as lemonslice_api:
            result = await lemonslice_api.join_meeting(
                self._session_id,
                meeting_url=meeting_url,
                livekit_url=self._livekit_url,
                broadcast_token=broadcast_token,
                bot_name=bot_name,
            )

        meeting_audio = MeetingAudioInput()
        self._agent_session.input.audio = meeting_audio
        relay_stop = asyncio.Event()
        self._meeting_relay_stop = relay_stop

        chat_relay = MeetingChatRelay(
            self._agent_session,
            bot_name=bot_name,
        )
        chat_relay.start()

        self._meeting_relay_task = asyncio.create_task(
            stream_meeting_relay(
                result.agent_audio_websocket_url,
                meeting_audio.submit,
                chat_sink=chat_relay.submit_json,
                stop=relay_stop,
            )
        )
        self._meeting_chat = chat_relay
        self._meeting_bot_id = result.meeting_bot_id
        return result

    async def aclose(self) -> None:
        await self.leave_meeting()
        await super().aclose()

    async def leave_meeting(self) -> None:
        """Leave the external meeting and stop the audio/chat relay."""
        meeting_bot_id = self._meeting_bot_id
        session_id = self._session_id
        if not meeting_bot_id or not session_id:
            return

        self._meeting_bot_id = None

        async with LemonSliceAPI(
            api_url=self._api_url,
            api_key=self._api_key,
            conn_options=self._conn_options,
            session=self._http_session,
        ) as lemonslice_api:
            await lemonslice_api.leave_meeting(session_id, meeting_bot_id=meeting_bot_id)

        if self._meeting_relay_stop is not None:
            self._meeting_relay_stop.set()
        relay_task = self._meeting_relay_task
        if relay_task is not None:
            relay_task.cancel()
            await asyncio.gather(relay_task, return_exceptions=True)
            self._meeting_relay_task = None

        if self._meeting_chat is not None:
            await self._meeting_chat.aclose()
            self._meeting_chat = None

        self._meeting_relay_stop = None

    def room_options(self, **kwargs: Any) -> room_io.RoomOptions:
        """Room I/O options for ``AgentSession.start``.

        When :meth:`join_meeting` has been called, disables LiveKit room audio I/O
        (meeting audio is fed directly into STT).
        """
        if self._meeting_bot_id is not None:
            return meeting_room_options(**kwargs)
        return room_io.RoomOptions(**kwargs)
