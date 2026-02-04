from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
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
OUTBOUND_QUEUE_MAXSIZE = 256
SEGMENT_END_GRACE_SECONDS = 2.0
WS_RECEIVE_TIMEOUT = 90  # Slightly longer than keep-alive to detect dead connections
SPEAK_END_EVENT_TYPE = "agent.speak_end"
_AVATAR_AGENT_IDENTITY = "liveavatar-avatar-agent"
_AVATAR_AGENT_NAME = "liveavatar-avatar-agent"


@dataclass
class _AudioSegment:
    """Tracks state for a single audio segment sent to the avatar."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    playback_position: float = 0.0
    speaking: bool = False  # True when server confirms avatar is speaking this segment
    task_id: str | None = None
    timeout_task: asyncio.Task[None] | None = None


class AvatarSession:
    """A LiveAvatar avatar session.

    Manages the connection to LiveAvatar's LITE mode API, handling audio streaming
    and avatar state synchronization. Each audio segment is tracked independently
    using event IDs to properly correlate with server events.
    """

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
        self._main_atask: asyncio.Task[None] | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._audio_resampler_input_rate: int | None = None
        self._msg_ch = utils.aio.Chan[dict[str, Any]](maxsize=OUTBOUND_QUEUE_MAXSIZE)
        self._closed = False

        self._state_lock = asyncio.Lock()
        # Connection state tracking - wait for "connected" before sending commands
        self._connected_event = asyncio.Event()
        self._close_event = asyncio.Event()
        self._last_send_at = time.monotonic()

        # Per-segment state tracking using event IDs for proper correlation
        # with server events (agent.speak_started, agent.speak_ended)
        self._current_segment: _AudioSegment | None = None  # Segment being streamed
        self._pending_segments: dict[str, _AudioSegment] = {}  # Segments awaiting playback
        self._task_id_to_event_id: dict[str, str] = {}
        self._desired_listening: bool | None = None
        self._sent_listening: bool | None = None

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Start the avatar session.

        Args:
            agent_session: The agent session to attach to
            room: The LiveKit room
            livekit_url: LiveKit server URL (defaults to LIVEKIT_URL env var)
            livekit_api_key: LiveKit API key (defaults to LIVEKIT_API_KEY env var)
            livekit_api_secret: LiveKit API secret (defaults to LIVEKIT_API_SECRET env var)
        """
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
            if ev.new_state == "listening":
                self._update_listening_state(True)
            elif ev.old_state == "listening":
                self._update_listening_state(False)

        @self._agent_session.on("close")
        def on_agent_session_close(ev: Any) -> None:
            task = asyncio.create_task(self.aclose())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._desired_listening = self._agent_session.agent_state == "listening"

        self._audio_buffer = QueueAudioOutput(sample_rate=SAMPLE_RATE)
        await self._audio_buffer.start()
        self._audio_buffer.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]

        agent_session.output.audio = self._audio_buffer
        self._main_atask = asyncio.create_task(self._main_task(), name="AvatarSession._main_task")

    async def aclose(self) -> None:
        """Close the avatar session and clean up resources."""
        if self._closed:
            return
        self._closed = True
        self._close_event.set()

        self._msg_ch.close()

        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

        await utils.aio.cancel_and_wait(*self._tasks)

    def _on_clear_buffer(self) -> None:
        """Handle buffer clear (interruption) by notifying all queued segments."""

        @utils.log_exceptions(logger=logger)
        async def _handle_clear_buffer() -> None:
            async with self._state_lock:
                segments = self._drain_segments()
                self._task_id_to_event_id.clear()

            if not segments:
                return

            self._notify_segments_finished(segments, interrupted=True)

            # Send interrupt command to avatar
            self._send_event({"type": "agent.interrupt", "event_id": str(uuid.uuid4())})

        clear_buffer_task = asyncio.create_task(_handle_clear_buffer())
        self._tasks.add(clear_buffer_task)
        clear_buffer_task.add_done_callback(self._tasks.discard)

    def _drain_segments(self) -> list[_AudioSegment]:
        """Collect and clear all known segments."""
        segments: list[_AudioSegment] = []

        if self._current_segment:
            segments.append(self._current_segment)
            self._current_segment = None

        if self._pending_segments:
            segments.extend(self._pending_segments.values())
            self._pending_segments.clear()

        for segment in segments:
            if segment.timeout_task:
                segment.timeout_task.cancel()
                segment.timeout_task = None

        return segments

    def _notify_segments_finished(
        self, segments: list[_AudioSegment], *, interrupted: bool
    ) -> None:
        """Notify the audio buffer that segments have finished."""
        for segment in segments:
            self._audio_buffer.notify_playback_finished(
                playback_position=segment.playback_position,
                interrupted=interrupted,
            )
            if interrupted:
                logger.debug(
                    f"Interrupted segment {segment.event_id} at position "
                    f"{segment.playback_position:.3f}s"
                )

    async def _flush_segments(self, *, interrupted: bool) -> None:
        async with self._state_lock:
            segments = self._drain_segments()
            self._task_id_to_event_id.clear()
        if segments:
            self._notify_segments_finished(segments, interrupted=interrupted)

    def _extract_event_ids(self, event: dict[str, Any]) -> tuple[str | None, str | None]:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            event_id = None

        task_id = None
        task_data = event.get("task")
        if isinstance(task_data, dict):
            task_value = task_data.get("id")
            if isinstance(task_value, str) and task_value:
                task_id = task_value

        return event_id, task_id

    async def _on_server_event(self, event: dict[str, Any]) -> None:
        """Process incoming server events from LiveAvatar."""
        event_type = event.get("type")

        if event_type == "session.state_updated":
            state = event.get("state") if isinstance(event.get("state"), str) else None
            self._set_connected_state(state)
            if state in ("closing", "closed"):
                await self._flush_segments(interrupted=True)
                if not self._msg_ch.closed:
                    self._msg_ch.close()
                logger.info(f"LiveAvatar session state: {state}")
            return

        event_id, task_id = self._extract_event_ids(event)

        if event_type == "agent.speak_started":
            async with self._state_lock:
                segment = self._find_segment(event_id, task_id)
                if segment:
                    if task_id:
                        self._record_task_id(segment, task_id)
                    segment.speaking = True
                    logger.debug(f"Avatar started speaking segment {segment.event_id}")
                else:
                    logger.warning(f"Received speak_started for unknown segment: {event_id}")
            return

        if event_type == "agent.speak_ended":
            segment: _AudioSegment | None
            async with self._state_lock:
                segment = self._find_segment(event_id, task_id)
                if segment and task_id:
                    self._record_task_id(segment, task_id)
                if segment:
                    self._remove_segment(segment)

            if segment:
                if not segment.speaking:
                    logger.debug(f"Segment {segment.event_id} ended without speak_started")
                self._audio_buffer.notify_playback_finished(
                    playback_position=segment.playback_position,
                    interrupted=False,
                )
                logger.debug(
                    f"Avatar finished speaking segment {segment.event_id} "
                    f"(position: {segment.playback_position:.3f}s)"
                )
            else:
                # Could be a segment that was already cleared by interruption
                logger.debug(f"Received speak_ended for unknown/cleared segment: {event_id}")
            return

    def _find_segment(self, event_id: str | None, task_id: str | None) -> _AudioSegment | None:
        """Find a segment by event_id or task_id."""
        if event_id:
            if self._current_segment and self._current_segment.event_id == event_id:
                return self._current_segment
            segment = self._pending_segments.get(event_id)
            if segment:
                return segment

        if task_id:
            mapped_event_id = self._task_id_to_event_id.get(task_id)
            if mapped_event_id:
                if self._current_segment and self._current_segment.event_id == mapped_event_id:
                    return self._current_segment
                segment = self._pending_segments.get(mapped_event_id)
                if segment:
                    return segment

            # Fallback if server uses task_id as event_id
            if self._current_segment and self._current_segment.event_id == task_id:
                return self._current_segment
            segment = self._pending_segments.get(task_id)
            if segment:
                return segment

        return None

    def _record_task_id(self, segment: _AudioSegment, task_id: str) -> None:
        if segment.task_id and segment.task_id != task_id:
            logger.warning(
                "Segment %s already bound to task_id %s, got %s",
                segment.event_id,
                segment.task_id,
                task_id,
            )
            return
        segment.task_id = task_id
        self._task_id_to_event_id[task_id] = segment.event_id

    def _remove_segment(self, segment: _AudioSegment) -> None:
        if self._current_segment is segment:
            self._current_segment = None
        else:
            self._pending_segments.pop(segment.event_id, None)

        if segment.task_id:
            self._task_id_to_event_id.pop(segment.task_id, None)

        if segment.timeout_task:
            segment.timeout_task.cancel()
            segment.timeout_task = None

    def _set_connected_state(self, state: str | None) -> None:
        logger.debug(f"Session state: {state}")
        if state == "connected":
            if not self._connected_event.is_set():
                self._connected_event.set()
                logger.info("LiveAvatar WebSocket connected")
            self._maybe_send_listening_state()
            return

        if self._connected_event.is_set():
            self._connected_event.clear()

    def _schedule_segment_timeout(self, segment: _AudioSegment) -> None:
        if segment.timeout_task:
            segment.timeout_task.cancel()
        segment.timeout_task = asyncio.create_task(
            self._segment_timeout(segment.event_id, segment.playback_position)
        )

    async def _segment_timeout(self, event_id: str, playback_position: float) -> None:
        try:
            await asyncio.sleep(playback_position + SEGMENT_END_GRACE_SECONDS)
        except asyncio.CancelledError:
            return

        async with self._state_lock:
            segment = self._pending_segments.pop(event_id, None)
            if segment and segment.task_id:
                self._task_id_to_event_id.pop(segment.task_id, None)

        if segment:
            self._audio_buffer.notify_playback_finished(
                playback_position=segment.playback_position,
                interrupted=False,
            )
            logger.warning(
                "Timed out waiting for speak_ended on segment %s; finalized playback at %.3fs",
                segment.event_id,
                segment.playback_position,
            )

    def _update_listening_state(self, listening: bool) -> None:
        self._desired_listening = listening
        self._maybe_send_listening_state()

    def _maybe_send_listening_state(self) -> None:
        if not self._connected_event.is_set() or self._desired_listening is None:
            return
        if self._sent_listening is None and not self._desired_listening:
            return
        if self._sent_listening == self._desired_listening:
            return

        msg_type = "agent.start_listening" if self._desired_listening else "agent.stop_listening"
        self._send_event({"type": msg_type, "event_id": str(uuid.uuid4())})
        self._sent_listening = self._desired_listening

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._audio_resampler and self._audio_resampler_input_rate is not None:
            if frame.sample_rate != self._audio_resampler_input_rate:
                self._audio_resampler = None
                self._audio_resampler_input_rate = None

        if self._audio_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1
        ):
            self._audio_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=1,
            )
            self._audio_resampler_input_rate = frame.sample_rate

        if self._audio_resampler:
            yield from self._audio_resampler.push(frame)
        else:
            yield frame

    def _send_event(self, msg: dict[str, Any]) -> None:
        """Queue an event to be sent to the avatar, only if connected."""
        # Only send if we're connected (per LiveAvatar docs requirement)
        if not self._connected_event.is_set():
            logger.debug(f"Dropping event {msg.get('type')} - not connected yet")
            return
        try:
            self._msg_ch.send_nowait(msg)
        except utils.aio.channel.ChanFull:
            logger.warning("Dropping outbound event %s: queue is full", msg.get("type"))
        except utils.aio.channel.ChanClosed:
            return

    async def _queue_event(self, msg: dict[str, Any]) -> None:
        """Queue an event, waiting for capacity if needed."""
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            await self._msg_ch.send(msg)

    async def _wait_until_connected(self) -> bool:
        if self._connected_event.is_set():
            return True

        connected_wait = asyncio.create_task(self._connected_event.wait())
        closed_wait = asyncio.create_task(self._close_event.wait())
        done, pending = await asyncio.wait(
            [connected_wait, closed_wait],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        return self._connected_event.is_set() and not self._close_event.is_set()

    async def _main_task(self) -> None:
        # Use shared http session from context instead of private API method
        http_session = utils.http_context.http_session()
        ws_conn = await http_session.ws_connect(url=self._ws_url)
        closing = False

        async def _forward_audio() -> None:
            # Wait for connection before forwarding audio
            try:
                connected = await asyncio.wait_for(self._wait_until_connected(), timeout=30.0)
                if not connected:
                    return
            except asyncio.TimeoutError:
                raise APIConnectionError("Timeout waiting for LiveAvatar connection") from None

            async for audio_frame in self._audio_buffer:
                if isinstance(audio_frame, rtc.AudioFrame):
                    for resampled_frame in self._resample_audio(audio_frame):
                        async with self._state_lock:
                            if self._current_segment is None:
                                self._current_segment = _AudioSegment()
                                logger.debug(
                                    "Started new audio segment: %s",
                                    self._current_segment.event_id,
                                )

                            segment = self._current_segment
                            segment.playback_position += resampled_frame.duration
                            event_id = segment.event_id

                        data = resampled_frame.data.tobytes()
                        encoded_audio = base64.b64encode(data).decode("utf-8")

                        # Use consistent event_id for all chunks in this segment
                        msg = {
                            "type": "agent.speak",
                            "event_id": event_id,
                            "audio": encoded_audio,
                        }

                        await self._queue_event(msg)

                elif isinstance(audio_frame, AudioSegmentEnd):
                    async with self._state_lock:
                        if self._current_segment is None:
                            continue
                        segment = self._current_segment
                        self._current_segment = None
                        self._pending_segments[segment.event_id] = segment
                        self._schedule_segment_timeout(segment)
                        event_id = segment.event_id
                        duration = segment.playback_position

                    # Signal end of audio stream with the same event_id
                    await self._queue_event(
                        {
                            "type": SPEAK_END_EVENT_TYPE,
                            "event_id": event_id,
                        }
                    )

                    logger.debug(
                        "Finished streaming segment %s (duration: %.3fs)",
                        event_id,
                        duration,
                    )

        async def _keep_alive_task() -> None:
            # Wait for connection before sending keep-alives
            await self._connected_event.wait()

            try:
                while True:
                    await asyncio.sleep(KEEP_ALIVE_INTERVAL)
                    if closing:
                        break
                    if time.monotonic() - self._last_send_at >= KEEP_ALIVE_INTERVAL:
                        msg = {
                            "type": "session.keep_alive",
                            "event_id": str(uuid.uuid4()),
                        }
                        self._send_event(msg)
            except asyncio.CancelledError:
                return

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing

            async for msg in self._msg_ch:
                try:
                    if not await self._wait_until_connected():
                        break
                    await ws_conn.send_json(data=msg)
                    self._last_send_at = time.monotonic()
                except Exception as e:
                    logger.error(f"Failed to send WebSocket message: {e}")
                    break
            closing = True
            self._close_event.set()
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                try:
                    msg = await asyncio.wait_for(ws_conn.receive(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("WebSocket receive timeout - connection may be dead")
                    raise APIConnectionError("LiveAvatar WebSocket receive timeout") from None

                # Handle connection close/error first (early exit pattern)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    self._close_event.set()
                    raise APIConnectionError(message="LiveAvatar connection closed unexpectedly.")

                if msg.type == aiohttp.WSMsgType.ERROR:
                    self._close_event.set()
                    raise APIConnectionError(f"WebSocket error: {ws_conn.exception()}")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    event = json.loads(msg.data)
                    await self._on_server_event(event)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse server event: {msg.data}")

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
            self._close_event.set()
            await utils.aio.cancel_and_wait(*io_tasks)
            await utils.aio.cancel_and_wait(*self._tasks)
            await self._flush_segments(interrupted=True)
            try:
                if self._session_id and self._session_token:
                    data = await self._api.stop_streaming_session(
                        self._session_id, self._session_token
                    )
                    if data["code"] < 300:
                        logger.info(f"LiveAvatar session stopped: {self._session_id}")
            except Exception as e:
                logger.warning(f"Failed to stop LiveAvatar session: {e}", exc_info=True)

            await self._audio_buffer.aclose()
            await ws_conn.close()
