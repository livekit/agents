from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import time
import uuid
from abc import ABC
from collections.abc import Iterator
from typing import Any, Literal, TypeVar, Union

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
from livekit.agents.metrics import UsageCollector, VideoAvatarMetrics, log_metrics
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import LiveAvatarAPI, LiveAvatarException
from .log import logger

TEvent = TypeVar("TEvent")

SAMPLE_RATE = 24000
KEEP_ALIVE_INTERVAL = 60
_AVATAR_AGENT_IDENTITY = "liveavatar-avatar-agent"
_AVATAR_AGENT_NAME = "liveavatar-avatar-agent"


class AvatarSession(ABC, rtc.EventEmitter[Union[Literal["metrics_collected", "error"], TEvent]]):
    """A LiveAvatar avatar session"""

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
        super().__init__()
        self._avatar_id = avatar_id or os.getenv("LIVEAVATAR_AVATAR_ID")
        self._session_id: str | None = None
        self._session_token: str | None = None
        self._api = LiveAvatarAPI(api_key=api_key, api_url=api_url, conn_options=conn_options)

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._tasks: set[asyncio.Task[Any]] = set()
        self._main_atask: asyncio.Task | None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._session_data = None
        self._msg_ch = utils.aio.Chan[dict]()
        self._audio_playing = False
        self._playback_position = 0.0
        self._pending = {}
        self._latency_pending = {}
        self._last_latency = None
        self._usage_collector = UsageCollector()

        self.on("metrics_collected", self._on_metrics_collected)

    def _on_metrics_collected(self, ev: VideoAvatarMetrics):
        m = ev
        log_metrics(m)
        self._usage_collector.collect(m)

    async def _video_frame_loop(self, video_stream: rtc.VideoStream):
        try:
            async for frame_event in video_stream:
                self._on_video_frame(frame_event.frame)
        except Exception as e:
            logger.warning(f"video frame loop ended: {e}")

    def _on_video_frame(self, frame, event_id=None):
        now = time.perf_counter()

        for eid, item in list(self._latency_pending.items()):
            if item.get("ws_received") and "video_received" not in item:
                item["video_received"] = now

                audio_sent = item["audio_sent"]
                ws_received = item["ws_received"]
                video_received = item["video_received"]

                full_latency = video_received - audio_sent
                server_latency = ws_received - audio_sent
                video_pipeline_latency = video_received - ws_received

                metrics = VideoAvatarMetrics(
                    event_id=eid,
                    timestamp=time.time(),
                    audio_sent_ts=audio_sent,
                    ws_received_ts=ws_received,
                    video_received_ts=video_received,
                    full_latency=full_latency,
                    server_latency=server_latency,
                    video_pipeline_latency=video_pipeline_latency,
                )

                self.emit("metrics_collected", metrics)
                self._latency_pending.pop(eid, None)
                break

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

        session_config_data = await self._api.create_streaming_session(
            livekit_url=livekit_url,
            livekit_token=livekit_token,
            room=self._room,
            avatar_id=self._avatar_id,
        )
        self._session_id = session_config_data["data"]["session_id"]
        self._session_token = session_config_data["data"]["session_token"]
        logger.info(f"LiveAvatar session created: {self._session_id}")

        session_start_data = await self._api.start_streaming_session(
            self._session_id, self._session_token
        )
        self._ws_url = session_start_data["data"]["ws_url"]
        logger.info("LiveAvatar streaming session started")

        @room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                video_stream = rtc.VideoStream(track)
                asyncio.create_task(self._video_frame_loop(video_stream))

        @self._agent_session.on("agent_state_changed")
        def on_agent_state_changed(ev):
            if ev.old_state == "speaking" and ev.new_state == "listening":
                self.send_event({"type": "agent.speak_end", "event_id": str(uuid.uuid4())})
                self.send_event({"type": "agent.start_listening", "event_id": str(uuid.uuid4())})
            if ev.new_state == "idle":
                self.send_event({"type": "agent.stop_listening", "event_id": str(uuid.uuid4())})

        @self._agent_session.on("close")
        def on_agent_session_close(ev):
            self._msg_ch.close()

        self._audio_buffer = QueueAudioOutput(sample_rate=SAMPLE_RATE)
        await self._audio_buffer.start()
        self._audio_buffer.on("clear_buffer", self._on_clear_buffer)

        agent_session.output.audio = self._audio_buffer
        self._main_atask = asyncio.create_task(self._main_task(), name="AvatarSession._main_task")

    def _on_clear_buffer(self) -> None:
        @utils.log_exceptions(logger=logger)
        async def _handle_clear_buffer(audio_playing: bool) -> None:
            if audio_playing:
                notify_task = self._audio_buffer.notify_playback_finished(
                    playback_position=self._playback_position,
                    interrupted=True,
                )
                self.send_event({"type": "agent.interrupt", "event_id": str(uuid.uuid4())})
                self._playback_position = 0.0
                if asyncio.iscoroutine(notify_task):
                    await notify_task

        clear_buffer_task = asyncio.create_task(_handle_clear_buffer(self._audio_playing))
        self._tasks.add(clear_buffer_task)
        clear_buffer_task.add_done_callback(self._tasks.discard)
        self._audio_playing = False

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
                    if not self._audio_playing:
                        self._audio_playing = True
                    for resampled_frame in self._resample_audio(audio_frame):
                        data = resampled_frame.data.tobytes()
                        encoded_audio = base64.b64encode(data).decode("utf-8")

                        msg = {
                            "type": "agent.speak",
                            "event_id": str(uuid.uuid4()),
                            "audio": encoded_audio,
                        }
                        send_ts = time.perf_counter()
                        msg_id = msg["event_id"]
                        # store structured entry so recv_task and video handler can share data
                        self._latency_pending[msg_id] = {
                            "audio_sent": send_ts,
                            "ws_received": None,
                            "created_at": time.perf_counter(),
                        }

                        self.send_event(msg)
                        self._playback_position += resampled_frame.duration

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
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    raise APIConnectionError(message="LiveAvatar connection closed unexpectedly.")
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = msg.json()

                        recv_ts = time.perf_counter()
                        event_id = data.get("event_id")

                        if event_id in self._latency_pending:
                            entry = self._latency_pending[event_id]
                            entry["ws_received"] = recv_ts

                            metrics = VideoAvatarMetrics(
                                event_id=event_id,
                                timestamp=time.time(),
                                audio_sent_ts=entry["audio_sent"],
                                ws_received_ts=recv_ts,
                                full_latency=-1,
                                server_latency=recv_ts - entry["audio_sent"],
                                video_pipeline_latency=-1,
                            )

                            self.emit("metrics_collected", metrics)

                        # fallback: if server didn't echo event_id, measure first pending entry
                        elif self._latency_pending:
                            # find earliest entry without ws_received
                            for _first_id, entry in list(self._latency_pending.items()):
                                if isinstance(entry, dict) and entry.get("ws_received") is None:
                                    entry["ws_received"] = recv_ts
                                    break

                    except Exception:
                        pass

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
