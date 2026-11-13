from __future__ import annotations

import asyncio
import io
import json
import os
from dataclasses import replace
from datetime import timedelta
from typing import TypeVar

from PIL import Image
from pydantic import BaseModel, JsonValue

import livekit.rtc as rtc
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
    http,
    utils,
)
from livekit.agents.voice import room_io
from livekit.agents.voice.avatar import (
    AvatarSession as BaseAvatarSession,
    AvatarStart,
    DataStreamAudioOutput,
)

from .log import logger
from .meeting.audio import MeetingAudioInput, stream_meeting_relay
from .meeting.chat import MeetingChatRelay
from .meeting.room import JoinMeetingResult

DEFAULT_API_URL = "https://lemonslice.com/api/liveai/sessions"
_MEETING_BROADCAST_IDENTITY = "lemonslice-meeting-broadcast"


class LemonSliceException(Exception):
    """Exception for LemonSlice errors"""


class _Session(BaseModel):
    session_id: str


class _JoinMeeting(BaseModel):
    websocket_url: str
    meeting_bot_id: str


class _Empty(BaseModel):
    pass


_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


class AvatarSession(BaseAvatarSession):
    """A LemonSlice avatar session"""

    sample_rate = 16000

    def __init__(
        self,
        *,
        agent_id: NotGivenOr[str] = NOT_GIVEN,
        agent_image_url: NotGivenOr[str] = NOT_GIVEN,
        agent_image: NotGivenOr[Image.Image] = NOT_GIVEN,
        agent_prompt: NotGivenOr[str] = NOT_GIVEN,
        agent_idle_prompt: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout: NotGivenOr[int] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: JsonValue,
    ) -> None:
        """
        Args:
            agent_id: The ID of the LemonSlice agent to add to the session.
            agent_image_url: The URL of the image to use as the agent's avatar.
            agent_image: A PIL image to upload and use as the agent's avatar.
            agent_prompt: A prompt that subtly influences the avatar's movements and
                expressions while responding.
            agent_idle_prompt: The same, while idle.
            idle_timeout: The idle timeout, in seconds.
            api_url: The LemonSlice sessions API URL.
            api_key: Your LemonSlice API key; ``LEMONSLICE_API_KEY`` by default.
            avatar_participant_identity: The avatar's room identity.
            avatar_participant_name: The avatar's display name.
            livekit_url: The URL the avatar joins through; ``LIVEKIT_URL`` by default.
            livekit_api_key: Signs the avatar's token; ``LIVEKIT_API_KEY`` by default.
            livekit_api_secret: Signs the avatar's token; ``LIVEKIT_API_SECRET`` by default.
            conn_options: Retries and limits of the LemonSlice requests.
            **kwargs: Added to the session request.
        """
        super().__init__(
            avatar_participant_identity=avatar_participant_identity,
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        sources = [s for s in (agent_id, agent_image_url, agent_image) if utils.is_given(s)]
        if len(sources) != 1:
            raise LemonSliceException(
                "Exactly one of agent_id, agent_image_url or agent_image must be provided"
            )
        api_key = api_key if utils.is_given(api_key) else os.getenv("LEMONSLICE_API_KEY") or ""
        if not api_key:
            raise LemonSliceException("LEMONSLICE_API_KEY must be set")
        self._headers = {"X-API-Key": api_key}
        self._api_url = (api_url or DEFAULT_API_URL).rstrip("/")

        payload: dict[str, JsonValue] = {"transport_type": "livekit"}
        for key, text in (
            ("agent_id", agent_id),
            ("agent_image_url", agent_image_url),
            ("agent_prompt", agent_prompt),
            ("agent_idle_prompt", agent_idle_prompt),
        ):
            if utils.is_given(text):
                payload[key] = text
        if utils.is_given(idle_timeout):
            payload["idle_timeout"] = idle_timeout
        self._payload = payload | kwargs
        self._image: bytes | None = None
        if utils.is_given(agent_image):
            buffer = io.BytesIO()
            agent_image.save(buffer, format="PNG")
            self._image = buffer.getvalue()

        self._session_id: str | None = None
        self._start: AvatarStart | None = None

        # External meeting state (Zoom, Meet, Teams, Webex); set by join_meeting().
        self._meeting_bot_id: str | None = None
        self._meeting_chat: MeetingChatRelay | None = None
        self._meeting_relay_stop: asyncio.Event | None = None
        self._meeting_relay_task: asyncio.Task[None] | None = None
        self._meeting_tasks = utils.aio.TaskSet()

    @property
    def provider(self) -> str:
        return "lemonslice"

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def create_session(self, ctx: AvatarStart) -> str:
        self._start = ctx
        # replies started while LemonSlice creates the session already go to the avatar, whose
        # output holds them until it joins
        self._attach_audio_output(ctx.agent_session, self.audio_output(ctx))

        job = get_job_context(required=False)
        properties: dict[str, JsonValue] = {
            "livekit_url": ctx.livekit_url,
            "livekit_token": ctx.livekit_token(),
            "livekit_session_id": (job.job.room.sid if job else "") or await ctx.room.sid,
        }
        payload = {"properties": properties} | self._payload
        form: http.FormData | None = None
        if self._image is not None:
            form = http.FormData()
            form.add_field("payload", json.dumps(payload), content_type="application/json")
            form.add_field("image", self._image, filename="image.png", content_type="image/png")
        async with http.http_session().post(
            self._api_url,
            headers=self._headers,
            data=form,
            json=payload if form is None else None,
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            self._session_id = _Session.model_validate_json(await response.text()).session_id
        return self._session_id

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput:
        if isinstance(self._audio_output, DataStreamAudioOutput):
            return self._audio_output
        return DataStreamAudioOutput(
            room=ctx.room,
            destination_identity=self.avatar_identity,
            sample_rate=self.sample_rate,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            clear_buffer_timeout=None,
            wait_playback_start=True,
        )

    async def _post(
        self, path: str, payload: dict[str, JsonValue], answer: type[_ResponseT]
    ) -> _ResponseT:
        async def request() -> _ResponseT:
            async with http.http_session().post(
                f"{self._api_url}/{path}",
                headers=self._headers,
                json=payload,
                conn_options=self._conn_options,
            ) as response:
                await response.raise_for_status()
                return answer.model_validate_json(await response.text())

        return await http.retry_request(request, conn_options=self._conn_options)

    async def join_meeting(
        self,
        meeting_url: str,
        *,
        bot_name: NotGivenOr[str] = NOT_GIVEN,
        listen_to_meeting_chat: bool = True,
    ) -> JoinMeetingResult:
        """Send this avatar into an external video meeting.

        Supports Zoom, Google Meet, Microsoft Teams, and Webex. Call after start() and
        before room_options() / VoiceSession.start().

        Args:
            meeting_url: URL of the external meeting to join.
            bot_name: Display name for the bot in the meeting.
            listen_to_meeting_chat: When True, relay meeting chat into the agent
                session.

        Returns:
            JoinMeetingResult with relay WebSocket URL and meeting bot ID.
        """
        ctx = self._start
        if not self._session_id or ctx is None:
            raise LemonSliceException("call start() before join_meeting()")
        if self._meeting_bot_id is not None:
            raise LemonSliceException("already joined a meeting; call leave_meeting() first")

        broadcast_token = (
            ctx.access_token()
            .with_identity(_MEETING_BROADCAST_IDENTITY)
            .with_name(_MEETING_BROADCAST_IDENTITY)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=ctx.room.name,
                    can_subscribe=True,
                    can_publish=False,
                    can_publish_data=False,
                )
            )
            .with_ttl(timedelta(hours=4))
            .to_jwt()
        )
        payload: dict[str, JsonValue] = {
            "session_id": self._session_id,
            "meeting_url": meeting_url,
            "livekit_url": ctx.livekit_url,
            "broadcast_token": broadcast_token,
        }
        if bot_name:
            payload["bot_name"] = bot_name
        joined = await self._post(f"{self._session_id}/join-meeting", payload, _JoinMeeting)
        result = JoinMeetingResult(
            websocket_url=joined.websocket_url, meeting_bot_id=joined.meeting_bot_id
        )

        meeting_audio = MeetingAudioInput()
        ctx.agent_session.input.audio = meeting_audio
        relay_stop = asyncio.Event()
        self._meeting_relay_stop = relay_stop

        chat_sink = None
        if listen_to_meeting_chat:
            chat_relay = MeetingChatRelay(ctx.agent_session, bot_name=bot_name)
            chat_relay.start()
            self._meeting_chat = chat_relay
            chat_sink = chat_relay.submit_json

        self._meeting_relay_task = self._meeting_tasks.spawn(
            stream_meeting_relay(
                result.websocket_url,
                meeting_audio.submit,
                chat_sink=chat_sink,
                stop=relay_stop,
                conn_options=self._conn_options,
            )
        )
        self._meeting_bot_id = result.meeting_bot_id
        return result

    async def aclose(self) -> None:
        try:
            await self.leave_meeting()
        finally:
            await self._meeting_tasks.aclose()
            await super().aclose()

    async def leave_meeting(self) -> None:
        """Leave the external meeting and stop the audio and chat relay."""
        meeting_bot_id = self._meeting_bot_id
        session_id = self._session_id
        if not meeting_bot_id or not session_id:
            return

        try:
            await self._post(
                f"{session_id}/leave-meeting", {"meeting_bot_id": meeting_bot_id}, _Empty
            )
        except Exception:
            logger.warning("failed to leave meeting via LemonSlice API", exc_info=True)
        finally:
            self._meeting_bot_id = None

            if self._meeting_relay_stop is not None:
                self._meeting_relay_stop.set()
            relay_task = self._meeting_relay_task
            if relay_task is not None:
                await utils.aio.cancel_and_wait(relay_task)
                self._meeting_relay_task = None

            if self._meeting_chat is not None:
                await self._meeting_chat.aclose()
                self._meeting_chat = None

            self._meeting_relay_stop = None

    def room_options(self, options: room_io.RoomOptions | None = None) -> room_io.RoomOptions:
        """Return room I/O options for VoiceSession.start.

        When join_meeting() has been called, disables LiveKit room audio input.
        Meeting audio is fed directly into STT instead.

        Args:
            options: Room I/O configuration to apply to the current avatar mode.

        Returns:
            RoomOptions appropriate for the current avatar mode.
        """
        if options is None:
            options = room_io.RoomOptions()
        if self._meeting_bot_id is not None:
            return replace(options, audio_input=False, audio_output=False)
        return options
