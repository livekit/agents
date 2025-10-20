# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable
from livekit.plugins.dataspike.schema_pb2 import (
    DeepfakeStreamingSchemaFrameRequest,
    DeepfakeStreamingSchemaFrameRequestFormat,
    DeepfakeStreamingSchemaResultEvent,
    DeepfakeStreamingSchemaResultEventType,
)

from livekit import rtc
from livekit.agents.utils.images import encode, EncodeOptions


class InputTrack:
    def __init__(
        self,
        *,
        track: rtc.Track,
        participant_identity: str,
        room: rtc.Room,
        burst_fps: float = 1,
        normal_fps: float = 0.2,
        state: DeepfakeStreamingSchemaResultEventType = DeepfakeStreamingSchemaResultEventType.CLEAR,
        quality: int = 75,
    ):
        self.track = track
        self.participant_identity = participant_identity
        self.room = room
        self.burst_fps = burst_fps
        self.normal_fps = normal_fps
        self.last_sampled_time: float | None = None
        self.state = state
        self.running = False

    def _skip_frame(self, now: float) -> bool:
        target_fps = (
            self.burst_fps
            if self.state == DeepfakeStreamingSchemaResultEventType.SUSPICIOUS
            else self.normal_fps
        )
        if target_fps == 0:
            return True

        min_frame_interval = 1.0 / target_fps

        if self.last_sampled_time is None:
            self.last_sampled_time = now
            return False

        if (now - self.last_sampled_time) >= min_frame_interval:
            self.last_sampled_time = now
            return False

        return True

    async def consume(
        self,
        q: asyncio.Queue[DeepfakeStreamingSchemaFrameRequest],
    ) -> None:
        self.running = True
        stream = rtc.VideoStream(track)
        async for frame in stream:
            now = time.time()

            if not self.running:
                break

            if self._skip_frame(now):
                continue

            image_bytes = encode(
                frame,
                EncodeOptions(
                    format="JPEG",
                    quality=self.quality,
                ),
            )
            event = DeepfakeStreamingSchemaFrameRequest(
                participant_id=self.participant_identity,
                track_id=self.track.sid,
                timestamp_ms=int(now * 1000),
                format=DeepfakeStreamingSchemaFrameRequestFormat.JPEG,
                data=image_bytes,
            )

            try:
                await q.put_nowait(event)
            except asyncio.QueueFull:
                # drop message silently, we do not want stale messages to block the queue
                pass

        await stream.aclose()

    def stop(self) -> None:
        self.running = False


class DataspikeDetector:
    MAX_QUEUE_SIZE = 64

    def __init__(
        self,
        *,
        api_key: str | None = None,
        conn_options: APIConnectOptions,
        burst_fps: float = 1,
        normal_fps: float = 0.2,
        quality: int = 75,
    ) -> None:

        self._ws_url = "wss://api.dataspike.io/api/v4/deepfake/stream"
        self._api_key = api_key or os.getenv("DATASPIKE_API_KEY")

        self._burst_fps = burst_fps
        self._normal_fps = normal_fps
        self._quality = quality

        self._conn_options = conn_options
        self._session: aiohttp.ClientSession | None = None
        self._agent_session: AgentSession | None = None
        self._room: rtc.Room | None = None
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )
        self._send_queue: asyncio.Queue[DeepfakeStreamingSchemaFrameRequest] = (
            asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        )
        self._input_tracks: list[InputTrack] = []

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._ws_url
        headers = {"Authorization": f"Bearer {self._api_key}"}
        return await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        self._agent_session = agent_session
        self._room = room

        await self._run_ws()

        for participant in self._room.remote_participants.values():
            video_tracks = [
                publication.track
                for publication in list(remote_participant.track_publications.values())
                if publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            self._add_track(participant, video_tracks[0])

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._add_track(participant, track)

        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._remove_track(track)

    def _add_track(self, participant: rtc.RemoteParticipant, track: rtc.Track) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return

        input_track = InputTrack(
            track=track,
            participant_identity=participant.identity,
            room=self._room,
            burst_fps=self._burst_fps,
            normal_fps=self._normal_fps,
            quality=self._quality,
        )
        self._input_tracks.append(input_track)

        asyncio.create_task(input_track.consume(self._send_queue))

    def _remove_track(self, track: rtc.Track) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return

        input_track = next(
            (t for t in self._input_tracks if t.track.sid == track.sid), None
        )
        if input_track:
            input_track.stop()
            self._input_tracks.remove(input_track)

    async def _run_ws(self) -> None:

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for event in self._send_queue:
                await ws.send_bytes(event.SerializeToString())

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Dataspike websocket connection closed unexpectedly"
                    )

                if msg.type != aiohttp.WSMsgType.BINARY:
                    logger.warning("unexpected Dataspike message type %s", msg.type)
                    continue

                resp = DeepfakeStreamingSchemaResultEvent()
                try:
                    resp = resp.parse(msg.data)
                except Exception as e:
                    logger.warning("failed to parse Dataspike message", exc_info=e)
                    continue

                if resp.type == DeepfakeStreamingSchemaResultEventType.SUSPICIOUS:
                    for input_track in self._input_tracks:
                        if input_track.track.sid == resp.track_id:
                            input_track.state = resp.type
                            break
                elif resp.type == DeepfakeStreamingSchemaResultEventType.CLEAR:
                    for input_track in self._input_tracks:
                        if input_track.track.sid == resp.track_id:
                            if (
                                input_track.state
                                == DeepfakeStreamingSchemaResultEventType.ALERT
                            ):
                                # TODO: send text to room chat
                                pass
                            input_track.state = resp.type
                            break
                elif resp.type == DeepfakeStreamingSchemaResultEventType.ALERT:
                    for input_track in self._input_tracks:
                        if input_track.track.sid == resp.track_id:
                            # TODO: send text to room chat
                            input_track.state = resp.type
                            break

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
