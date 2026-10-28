from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import aiohttp
from pydantic import JsonValue, TypeAdapter
from typing_extensions import Required, TypedDict

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    NotGivenOr,
    VoiceSession,
    get_job_context,
    utils,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.avatar import (
    AvatarSession as BaseAvatarSession,
    QueueAudioOutput,
)
from livekit.agents.voice.events import AgentStateChangedEvent, CloseEvent
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import LiveAvatarAPI, LiveAvatarException
from .log import logger


class _ServerEvent(TypedDict, total=False):
    type: Required[str]
    event_id: str
    source_event_id: str | None
    state: str
    previous_state: str
    new_state: str
    error: JsonValue
    warning: JsonValue


_SERVER_EVENT = TypeAdapter(_ServerEvent)


@dataclass
class _Utterance:
    event_id: str | None = None
    interrupt_id: str | None = None
    playback_position: float = 0.0
    started: bool = False
    finished: Literal["ended", "interrupted"] | None = None


SAMPLE_RATE = 24000
KEEP_ALIVE_INTERVAL = 60
_AVATAR_AGENT_IDENTITY = "liveavatar-avatar-agent"
_AVATAR_AGENT_NAME = "liveavatar-avatar-agent"

VideoQuality = Literal["very_high", "high", "medium", "low"]


class AvatarSession(BaseAvatarSession):
    """A LiveAvatar avatar session"""

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
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__()
        self._avatar_id = avatar_id if is_given(avatar_id) else os.getenv("LIVEAVATAR_AVATAR_ID")
        self._conn_options = conn_options
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
        self._video_quality: VideoQuality | None = (
            video_quality if is_given(video_quality) else None
        )

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._main_atask: asyncio.Task[None] | None = None
        self._audio_resampler: rtc.AudioResampler | None = None
        self._resampler_input_rate: int | None = None
        self._session_data = None
        self._msg_ch = utils.aio.Chan[dict[str, JsonValue]]()
        self._audio_playing = False
        self._utterances: deque[_Utterance] = deque()
        self._current_utterance: _Utterance | None = None
        self._session_connected = asyncio.Event()
        self._chunk_interrupted = asyncio.Event()

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "liveavatar"

    async def start(
        self,
        agent_session: VoiceSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        await super().start(agent_session, room)
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
            video_quality=self._video_quality,
        )
        token = session_config_data.data
        self._session_id = token.session_id
        self._session_token = token.session_token
        logger.info(f"LiveAvatar session created: {self._session_id}")

        assert self._session_id is not None
        assert self._session_token is not None
        session_start_data = await self._api.start_streaming_session(
            self._session_id, self._session_token
        )
        self._ws_url = session_start_data.data.ws_url
        logger.info("LiveAvatar streaming session started")

        @self._agent_session.on("agent_state_changed")
        def on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
            if ev.new_state == "idle":
                self.send_event({"type": "agent.stop_listening", "event_id": str(uuid.uuid4())})

        @self._agent_session.on("close")
        def on_agent_session_close(ev: CloseEvent) -> None:
            self._msg_ch.close()

        self._audio_buffer = QueueAudioOutput(sample_rate=SAMPLE_RATE, wait_playback_start=True)
        await self._audio_buffer.start()
        self._audio_buffer.on("clear_buffer", self._on_clear_buffer)

        agent_session.output.replace_audio_tail(self._audio_buffer)
        self._main_atask = asyncio.create_task(self._main_task(), name="AvatarSession._main_task")

    def _on_clear_buffer(self) -> None:
        self._chunk_interrupted.set()
        if self._audio_playing:
            interrupt_id = str(uuid.uuid4())
            for utterance in self._utterances:
                if utterance.finished is None and utterance.interrupt_id is None:
                    utterance.interrupt_id = interrupt_id
            self.send_event({"type": "agent.interrupt", "event_id": interrupt_id})
            self._current_utterance = None
        self._audio_playing = False

    async def aclose(self) -> None:
        self._msg_ch.close()
        main = self._main_atask
        try:
            if main is not None:
                if not main.done() and not main.cancelling():
                    main.cancel()
                await asyncio.shield(asyncio.gather(main, return_exceptions=True))
                await self._audio_buffer.aclose()
                if self._main_atask is main:
                    self._main_atask = None
        finally:
            await super().aclose()

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._audio_resampler:
            if frame.sample_rate != self._resampler_input_rate:
                self._audio_resampler = None

        if self._audio_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1
        ):
            self._audio_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=1,
            )
            self._resampler_input_rate = frame.sample_rate

        if self._audio_resampler:
            yield from self._audio_resampler.push(frame)
        else:
            yield frame

    def send_event(self, msg: dict[str, JsonValue]) -> None:
        with contextlib.suppress(utils.aio.ChanClosed):
            self._msg_ch.send_nowait(msg)

    async def _main_task(self) -> None:
        ws_conn = await utils.http_context.ws_connect(
            self._api.http_session(), self._ws_url, conn_options=self._conn_options
        )
        closing = False
        ping_interval = utils.aio.interval(KEEP_ALIVE_INTERVAL)

        async def _forward_audio() -> None:
            await self._session_connected.wait()
            chunk_buf: list[bytes] = []
            chunk_duration = 0.0
            is_first_chunk = True
            first_chunk_threshold = 0.6  # 600ms
            subsequent_chunk_threshold = 1.0  # 1s

            def _flush_chunk() -> None:
                nonlocal chunk_buf, chunk_duration, is_first_chunk
                if not chunk_buf:
                    return
                combined = b"".join(chunk_buf)
                encoded_audio = base64.b64encode(combined).decode("utf-8")
                utterance = self._current_utterance
                assert utterance is not None
                if utterance.event_id is None:
                    utterance.event_id = str(uuid.uuid4())
                msg: dict[str, JsonValue] = {
                    "type": "agent.speak",
                    "event_id": utterance.event_id,
                    "audio": encoded_audio,
                }
                self.send_event(msg)
                utterance.playback_position += chunk_duration
                chunk_buf = []
                chunk_duration = 0.0
                is_first_chunk = False

            def _discard_chunk() -> None:
                nonlocal chunk_buf, chunk_duration, is_first_chunk
                chunk_buf = []
                chunk_duration = 0.0
                is_first_chunk = True

            async for audio_frame in self._audio_buffer:
                if self._chunk_interrupted.is_set():
                    self._chunk_interrupted.clear()
                    _discard_chunk()

                if isinstance(audio_frame, rtc.AudioFrame):
                    if self._current_utterance is None:
                        self._current_utterance = _Utterance()
                        self._utterances.append(self._current_utterance)
                    self._audio_playing = True
                    for resampled_frame in self._resample_audio(audio_frame):
                        if not resampled_frame.data:
                            continue
                        chunk_buf.append(resampled_frame.data.tobytes())
                        chunk_duration += resampled_frame.duration
                        threshold = (
                            first_chunk_threshold if is_first_chunk else subsequent_chunk_threshold
                        )
                        if chunk_duration >= threshold:
                            _flush_chunk()
                else:
                    _flush_chunk()
                    if self._current_utterance is None:
                        continue
                    utterance = self._current_utterance
                    self._current_utterance = None
                    is_first_chunk = True
                    if utterance.event_id is None:
                        utterance.finished = "ended"
                        self._finish_utterances()
                        continue
                    self.send_event({"type": "agent.speak_end", "event_id": str(uuid.uuid4())})
                    self.send_event(
                        {"type": "agent.start_listening", "event_id": str(uuid.uuid4())}
                    )

        async def _keep_alive_task() -> None:
            await self._session_connected.wait()
            try:
                while True:
                    await ping_interval.tick()
                    if closing:
                        break
                    msg: dict[str, JsonValue] = {
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
                    if self._is_sandbox:
                        logger.warning(
                            "The LiveAvatar Sandbox connection surpassed the 1 minute limit"
                        )
                        return
                    else:
                        raise APIConnectionError(
                            message="LiveAvatar connection closed unexpectedly."
                        )
                event = _SERVER_EVENT.validate_json(msg.data, strict=True)
                self._handle_server_event(event)

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
            try:
                if self._session_id and self._session_token:
                    data = await self._api.stop_streaming_session(
                        self._session_id, self._session_token
                    )
                    if data.code <= 200:
                        logger.info(f"LiveAvatar session stopped: {self._session_id}")
            except Exception as e:
                logger.warning(f"Failed to stop LiveAvatar session: {e}", exc_info=True)

            await self._audio_buffer.aclose()
            await ws_conn.close()

    def _handle_server_event(self, event: _ServerEvent) -> None:
        event_type = event.get("type")
        if event_type == "session.state_updated":
            state = event.get("state")
            logger.debug(f"LiveAvatar session state: {state}")
            if state == "connected":
                self._session_connected.set()
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
            self._audio_buffer.notify_playback_finished(
                playback_position=0.0 if interrupted else utterance.playback_position,
                interrupted=interrupted,
            )
        self._audio_playing = any(
            item.finished is None and item.interrupt_id is None for item in self._utterances
        )

    def _handle_agent_speak_interrupted(self, event: _ServerEvent) -> None:
        if event["type"] == "agent.audio_buffer_cleared":
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
        self._audio_buffer.notify_playback_started()
