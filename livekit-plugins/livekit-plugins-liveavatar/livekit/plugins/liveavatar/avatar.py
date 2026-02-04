from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

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
    utils,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import LiveAvatarAPI, LiveAvatarException
from .log import logger

SAMPLE_RATE = 24000
KEEP_ALIVE_INTERVAL = 60
_AVATAR_AGENT_IDENTITY = "liveavatar-avatar-agent"
_AVATAR_AGENT_NAME = "liveavatar-avatar-agent"


class AvatarSession:
    """A LiveAvatar avatar session"""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        is_sandbox: NotGivenOr[bool] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._avatar_id = avatar_id if is_given(avatar_id) else os.getenv("LIVEAVATAR_AVATAR_ID")
        self._session_id: str | None = None
        self._session_token: str | None = None
        resolved_api_key = api_key if is_given(api_key) else os.getenv("LIVEAVATAR_API_KEY", "")
        if is_given(api_url):
            self._api = LiveAvatarAPI(
                api_key=resolved_api_key,
                api_url=api_url,
                conn_options=conn_options,
            )
        else:
            self._api = LiveAvatarAPI(
                api_key=resolved_api_key,
                conn_options=conn_options,
            )
        self._is_sandbox = is_sandbox if is_given(is_sandbox) else False

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._tasks: set[asyncio.Task[Any]] = set()
        self._main_atask: asyncio.Task | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._session_data = None
        self._msg_ch = utils.aio.Chan[dict]()

        # State tracking: separate "streaming to avatar" from "avatar is speaking"
        self._audio_streaming = False  # True while sending audio frames to avatar
        self._avatar_speaking = False  # True while avatar is actually speaking (from server events)
        self._playback_position = 0.0

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
            raise LiveAvatarException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set"
            )

        try:
            job_ctx = get_job_context()
            self._local_participant_identity = job_ctx.token_claims().identity
        except RuntimeError as e:
            if not room.isconnected():
                raise LiveAvatarException("failed to get local participant identity") from e
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

        if not self._avatar_id:
            raise LiveAvatarException("avatar_id must be set")

        session_config_data = await self._api.create_streaming_session(
            livekit_url=livekit_url,
            livekit_token=livekit_token,
            room=self._room,
            avatar_id=self._avatar_id,
            is_sandbox=self._is_sandbox,
        )
        self._session_id = session_config_data["data"]["session_id"]
        self._session_token = session_config_data["data"]["session_token"]
        logger.info(f"LiveAvatar session created: {self._session_id}")

        assert self._session_id is not None
        assert self._session_token is not None
        session_start_data = await self._api.start_streaming_session(
            self._session_id, self._session_token
        )
        self._ws_url = session_start_data["data"]["ws_url"]
        logger.info("LiveAvatar streaming session started")

        @self._agent_session.on("agent_state_changed")
        def on_agent_state_changed(ev: Any) -> None:
            # Send listening state changes for avatar animations
            # Note: agent.speak_end is now sent on AudioSegmentEnd, not here
            if ev.new_state == "listening" and ev.old_state != "listening":
                self.send_event({"type": "agent.start_listening", "event_id": str(uuid.uuid4())})
            elif ev.new_state == "idle":
                self.send_event({"type": "agent.stop_listening", "event_id": str(uuid.uuid4())})

        @self._agent_session.on("close")
        def on_agent_session_close(ev: Any) -> None:
            self._msg_ch.close()

        self._audio_buffer = QueueAudioOutput(sample_rate=SAMPLE_RATE)
        await self._audio_buffer.start()
        self._audio_buffer.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]

        agent_session.output.audio = self._audio_buffer
        self._main_atask = asyncio.create_task(self._main_task(), name="AvatarSession._main_task")

    def _on_clear_buffer(self) -> None:
        """Handle buffer clear (interruption) using actual avatar speaking state."""

        @utils.log_exceptions(logger=logger)
        async def _handle_clear_buffer(avatar_speaking: bool, playback_position: float) -> None:
            if avatar_speaking:
                self._audio_buffer.notify_playback_finished(
                    playback_position=playback_position,
                    interrupted=True,
                )
                self.send_event({"type": "agent.interrupt", "event_id": str(uuid.uuid4())})
                self._playback_position = 0.0
                self._avatar_speaking = False

        # Capture both values at call time to avoid race conditions
        clear_buffer_task = asyncio.create_task(
            _handle_clear_buffer(self._avatar_speaking, self._playback_position)
        )
        self._tasks.add(clear_buffer_task)
        clear_buffer_task.add_done_callback(self._tasks.discard)
        self._audio_streaming = False

    def _on_server_event(self, event: dict) -> None:
        """Process incoming server events from LiveAvatar."""
        event_type = event.get("type")

        if event_type == "agent.speak_started":
            self._avatar_speaking = True
            logger.debug(f"Avatar started speaking (event_id: {event.get('event_id')})")

        elif event_type == "agent.speak_ended":
            if self._avatar_speaking:
                self._avatar_speaking = False
                self._audio_buffer.notify_playback_finished(
                    playback_position=self._playback_position,
                    interrupted=False,
                )
                logger.debug(
                    f"Avatar finished speaking (event_id: {event.get('event_id')}, "
                    f"position: {self._playback_position:.3f}s)"
                )
                self._playback_position = 0.0

        elif event_type == "session.state_updated":
            logger.debug(f"Session state: {event.get('state')}")

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._audio_resampler:
            if frame.sample_rate != self._audio_resampler._input_rate:
                self._audio_resampler = None

        if self._audio_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1
        ):
            self._audio_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=1,
            )

        if self._audio_resampler:
            yield from self._audio_resampler.push(frame)
        else:
            yield frame

    def send_event(self, msg: dict) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(msg)

    async def _main_task(self) -> None:
        ws_conn = await self._api._ensure_http_session().ws_connect(url=self._ws_url)
        closing = False
        ping_interval = utils.aio.interval(KEEP_ALIVE_INTERVAL)

        async def _forward_audio() -> None:
            async for audio_frame in self._audio_buffer:
                if isinstance(audio_frame, rtc.AudioFrame):
                    if not self._audio_streaming:
                        self._audio_streaming = True
                    for resampled_frame in self._resample_audio(audio_frame):
                        data = resampled_frame.data.tobytes()
                        encoded_audio = base64.b64encode(data).decode("utf-8")

                        msg = {
                            "type": "agent.speak",
                            "event_id": str(uuid.uuid4()),
                            "audio": encoded_audio,
                        }

                        self.send_event(msg)
                        self._playback_position += resampled_frame.duration

                elif isinstance(audio_frame, AudioSegmentEnd):
                    if self._audio_streaming:
                        self._audio_streaming = False
                        # Signal end of audio stream to avatar
                        self.send_event({"type": "agent.speak_end", "event_id": str(uuid.uuid4())})
                        # Note: Do NOT call notify_playback_finished here
                        # Wait for agent.speak_ended server event instead

        async def _keep_alive_task() -> None:
            try:
                while True:
                    await ping_interval.tick()
                    if closing:
                        break
                    msg = {
                        "type": "session.keep_alive",
                        "event_id": str(uuid.uuid4()),
                    }
                    self.send_event(msg)
            except asyncio.CancelledError:
                return

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing

            async for msg in self._msg_ch:
                try:
                    await ws_conn.send_json(data=msg)
                    ping_interval.reset()
                except Exception:
                    break
            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                        self._on_server_event(event)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse server event: {msg.data}")
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    raise APIConnectionError(message="LiveAvatar connection closed unexpectedly.")

        io_tasks = [
            asyncio.create_task(_forward_audio(), name="_forward_audio_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_keep_alive_task(), name="_keep_alive_task"),
        ]
        try:
            done, _ = await asyncio.wait(io_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*io_tasks)
            await utils.aio.cancel_and_wait(*self._tasks)
            try:
                if self._session_id and self._session_token:
                    data = await self._api.stop_streaming_session(
                        self._session_id, self._session_token
                    )
                    if data["code"] <= 200:
                        logger.info(f"LiveAvatar session stopped: {self._session_id}")
            except Exception as e:
                logger.warning(f"Failed to stop LiveAvatar session: {e}", exc_info=True)

            await self._audio_buffer.aclose()
            await ws_conn.close()
