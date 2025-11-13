from __future__ import annotations

import asyncio
import base64
import os
import uuid
from collections.abc import Iterator

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
)
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import HeyGenAPI, HeyGenException
from .log import logger

SAMPLE_RATE = 24000
_AVATAR_AGENT_IDENTITY = "heygen-avatar-agent"
_AVATAR_AGENT_NAME = "heygen-avatar-agent"


class AvatarSession:
    """A HeyGen avatar session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._avatar_id = avatar_id or os.getenv("HEYGEN_AVATAR_ID")
        self._api_key = api_key or os.getenv("HEYGEN_API_KEY")
        self._api_url = api_url
        if not self._avatar_id:
            raise HeyGenException("avatar_id or HEYGEN_AVATAR_ID must be set")
        if not self._api_key:
            raise HeyGenException("api_key or HEYGEN_API_KEY must be set")

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._main_atask = asyncio.Task | None
        self._session_data = None

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        self._agent_session = agent_session
        self._room = room
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise HeyGenException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set"
            )

        try:
            job_ctx = get_job_context()
            self._local_participant_identity = job_ctx.token_claims().identity
        except RuntimeError as e:
            if not room.isconnected():
                raise HeyGenException("failed to get local participant identity") from e
            self._local_participant_identity = room.local_participant.identity

        livekit_token = (
            api.AccessToken(
                api_key=livekit_api_key,
                api_secret=livekit_api_secret,
            )
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=self._room.name))
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: self._local_participant_identity})
            .to_jwt()
        )

        logger.debug("starting avatar session")

        heygen_quality = os.getenv("HEYGEN_STREAM_QUALITY", "high").strip().lower()
        heygen_encoding = os.getenv("HEYGEN_VIDEO_ENCODING", "H264").strip().upper()

        logger.info(
            f"Requesting HeyGen session with quality={heygen_quality}, encoding={heygen_encoding}"
        )
        self._api = HeyGenAPI(api_key=self._api_key)

        session_data = await self._api.create_streaming_session(
            livekit_url=livekit_url,
            livekit_token=livekit_token,
            room=self._room,
            avatar_id=self._avatar_id,
            quality=heygen_quality,
            video_encoding=heygen_encoding,
        )
        self._session_id = session_data["data"].get("session_id")
        self._realtime_url = session_data["data"].get("realtime_endpoint")

        logger.info(f"HeyGen session created: {self._session_id}")

        data = await self._api.start_streaming_session(self._session_id)

        logger.info("HeyGen streaming session started")

        self._agent_audio_track = list(self._room.local_participant.track_publications.values())[
            0
        ].track

        self._main_atask = asyncio.create_task(self._main_task(), name="AvatarSession._main_task")

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._audio_resampler:
            yield from self._audio_resampler.push(frame)

    async def _main_task(self) -> None:
        local_participant = self._room.local_participant
        track_perms = rtc.ParticipantTrackPermission(
            participant_identity=_AVATAR_AGENT_IDENTITY, allow_all=True
        )
        local_participant.set_track_subscription_permissions(
            allow_all_participants=False, participant_permissions=[track_perms]
        )

        if self._agent_audio_track is not None:
            agent_audio_stream = rtc.AudioStream.from_track(track=self._agent_audio_track)

        ws_conn = await self._api._ensure_http_session().ws_connect(url=self._realtime_url)

        # TODO check input rate dynamically
        self._audio_resampler = rtc.AudioResampler(
            input_rate=48000, output_rate=24000, num_channels=1
        )

        async def _send_task() -> None:
            event_id = str(uuid.uuid4())

            @self._agent_session.on("agent_state_changed")
            def on_agent_state_changed(ev):
                nonlocal event_id
                if ev.old_state == "speaking" and ev.new_state == "listening":
                    msg = {"type": "agent.speak_end", "event_id": event_id}
                    asyncio.create_task(ws_conn.send_json(data=msg))
                    event_id = str(uuid.uuid4())

            @self._agent_session.on("conversation_item_added")
            def on_conversation_item_added(ev):
                nonlocal event_id
                if (
                    self._agent_session.current_speech is not None
                    and self._agent_session.current_speech.interrupted
                ):
                    msg = {"type": "agent.interrupt", "event_id": event_id}
                    asyncio.create_task(ws_conn.send_json(data=msg))
                    event_id = str(uuid.uuid4())

            async for audio_event in agent_audio_stream:
                audio_frame = audio_event.frame

                if not any(audio_frame.data):
                    continue

                for resampled_frame in self._resample_audio(audio_event.frame):
                    data = resampled_frame.data.tobytes()
                    encoded_audio = base64.b64encode(data).decode("utf-8")

                    msg = {
                        "type": "agent.speak",
                        "event_id": event_id,
                        "audio": encoded_audio,
                    }

                    try:
                        await ws_conn.send_json(data=msg)
                    except Exception as e:
                        print(e)
                        break

        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIConnectionError(message="HeyGen connection closed unexpectedly.")

        tasks = [
            asyncio.create_task(_send_task(), name="_send_task"),
            asyncio.create_task(_recv_task(), name="_recv_task"),
        ]

        asyncio.gather(*tasks)
