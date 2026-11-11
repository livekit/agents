from __future__ import annotations

import asyncio
import contextlib
import json
import os
import struct
import time
from collections import deque
from dataclasses import dataclass
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
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger

DEFAULT_API_URL = "wss://api.talking-avatar.dev"
DEFAULT_AVATAR_ID = "ava4"
SAMPLE_RATE = 24000
AUDIO_TRACK_NAME = "avatar-audio"
# the avatar joins the room before it answers "ready", which takes a few seconds
READY_TIMEOUT = 20.0
# the avatar reports whole milliseconds per run of audio
_PLAYED_TOLERANCE = 0.05
# each audio message: play time (0: queue it after what is queued), sample rate, then PCM16 mono
_AUDIO_HEADER = struct.Struct("<dI")

_AVATAR_AGENT_IDENTITY = "talkingavatar-avatar-agent"
_AVATAR_AGENT_NAME = "talkingavatar-avatar-agent"

_CLOSE_REASONS = {
    1011: "the avatar's stream was lost",
    4001: "invalid or missing API key",
    4002: "out of credits, or the monthly spending limit is reached",
    4003: "the session reached its 3-hour limit",
}


class TalkingAvatarException(Exception):
    """Exception for Talking Avatar errors"""


class AvatarSession(BaseAvatarSession):
    """A Talking Avatar session: a photorealistic avatar that joins the room and speaks the agent's voice.

    The agent's audio goes to the avatar over the Talking Avatar realtime API. The avatar publishes its video
    and the voice, in sync with the lips, on behalf of the agent.
    """

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            avatar_id: An avatar ID from the Talking Avatar dashboard, such as ``ava4``. Defaults to the
                ``TALKING_AVATAR_AVATAR_ID`` environment variable, then ``ava4``.
            api_key: Your ``tav_`` API key. Defaults to the ``TALKING_AVATAR_API_KEY`` environment variable.
            api_url: The realtime API's URL. Defaults to ``TALKING_AVATAR_API_URL``, then
                ``wss://api.talking-avatar.dev``.
            avatar_participant_identity: The avatar participant's identity in the room.
            avatar_participant_name: The avatar participant's name in the room.
            conn_options: Retries and timeouts for connecting to the API.
        """
        super().__init__()
        self._avatar_id = avatar_id or os.getenv("TALKING_AVATAR_AVATAR_ID") or DEFAULT_AVATAR_ID
        self._api_url = api_url or os.getenv("TALKING_AVATAR_API_URL") or DEFAULT_API_URL
        self._api_key = api_key or os.getenv("TALKING_AVATAR_API_KEY")
        if not self._api_key:
            raise TalkingAvatarException(
                "The api_key must be set either by passing api_key to the client or "
                "by setting the TALKING_AVATAR_API_KEY environment variable"
            )

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME
        self._conn_options = conn_options

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._own_http_session: aiohttp.ClientSession | None = None
        self._audio_output: _AvatarAudioOutput | None = None
        self._send_ch = utils.aio.Chan[bytes | str]()
        self._tasks: list[asyncio.Task[Any]] = []
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "talkingavatar"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        await super().start(agent_session, room)

        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise TalkingAvatarException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            local_participant_identity = job_ctx.local_participant_identity
        elif room.isconnected():
            local_participant_identity = room.local_participant.identity
        else:
            raise TalkingAvatarException("connect to the room before starting the avatar")

        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            # publish only: the agent's audio reaches the avatar through the Talking Avatar API
            .with_grants(api.VideoGrants(room_join=True, room=room.name, can_subscribe=False))
            # allow the avatar to publish audio and video on behalf of your local agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        logger.debug("starting avatar session")
        self._ws = await self._connect(
            {
                "type": "start",
                "apiKey": self._api_key,
                "avatar": self._avatar_id,
                "livekitUrl": livekit_url,
                "token": livekit_token,
                "audio": {"publish": True, "trackName": AUDIO_TRACK_NAME},
                "sampleRate": SAMPLE_RATE,
            }
        )
        logger.info("Talking Avatar session started", extra={"avatar_id": self._avatar_id})

        self._audio_output = _AvatarAudioOutput(self._send_ch)
        agent_session.output.replace_audio_tail(self._audio_output)
        agent_session.on("close", self._on_agent_session_close)
        self._tasks = [
            asyncio.create_task(self._send_task(), name="talkingavatar_send"),
            asyncio.create_task(self._recv_task(), name="talkingavatar_recv"),
        ]

    async def aclose(self) -> None:
        """Ends the avatar session: the avatar leaves the room and billing stops."""
        self._closing = True
        self._send_ch.close()
        if self._ws is not None:
            await self._ws.close()  # code 1000
        await utils.aio.cancel_and_wait(*self._tasks)
        if self._own_http_session is not None:
            await self._own_http_session.close()
            self._own_http_session = None
        await super().aclose()

    def _on_agent_session_close(self, _: Any) -> None:
        if not self._closing and self._close_task is None:
            self._close_task = asyncio.create_task(self.aclose())

    def _http_session(self) -> aiohttp.ClientSession:
        try:
            return utils.http_context.http_session()
        except RuntimeError:  # outside a job (scripts, tests)
            if self._own_http_session is None:
                self._own_http_session = aiohttp.ClientSession()
            return self._own_http_session

    async def _connect(self, start_msg: dict[str, Any]) -> aiohttp.ClientWebSocketResponse:
        for attempt in range(self._conn_options.max_retry + 1):
            try:
                ws = await asyncio.wait_for(
                    self._http_session().ws_connect(self._api_url), self._conn_options.timeout
                )
                try:
                    await ws.send_str(json.dumps(start_msg))
                    await asyncio.wait_for(_wait_ready(ws), READY_TIMEOUT)
                except BaseException:
                    await ws.close()
                    raise
                return ws
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    "failed to start the Talking Avatar session",
                    extra={"attempt": attempt + 1, "error": str(e) or type(e).__name__},
                )
                if attempt >= self._conn_options.max_retry:
                    raise APIConnectionError(
                        "Failed to start the Talking Avatar session after all retries"
                    ) from e
                await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to start the Talking Avatar session after all retries")

    async def _send_task(self) -> None:
        assert self._ws is not None
        with contextlib.suppress(
            ConnectionError, aiohttp.ClientError
        ):  # closed: _recv_task reports it
            async for data in self._send_ch:
                if isinstance(data, bytes):
                    await self._ws.send_bytes(data)
                else:
                    await self._ws.send_str(data)

    async def _recv_task(self) -> None:
        assert self._ws is not None and self._audio_output is not None
        try:
            async for msg in self._ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(msg.data)
                if event.get("type") == "error":
                    logger.error(f"Talking Avatar error: {event.get('message')}")
                else:
                    self._audio_output._on_event(event)
        finally:
            self._audio_output._on_closed()
            if not self._closing:
                code = self._ws.close_code
                logger.error(
                    f"Talking Avatar session ended unexpectedly: {_CLOSE_REASONS.get(code or 0, code)}",
                    extra={"code": code},
                )


async def _wait_ready(ws: aiohttp.ClientWebSocketResponse) -> None:
    error: str | None = None
    async for msg in ws:
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        event = json.loads(msg.data)
        if event.get("type") == "ready":
            return
        if event.get("type") == "error":
            error = str(event.get("message"))

    code = ws.close_code
    if error or code in (4001, 4002):
        # refused (API key, credits, unknown avatar, LiveKit join failed...): a retry fails the same way
        raise TalkingAvatarException(error or _CLOSE_REASONS[code or 0])
    raise aiohttp.ClientError(f"connection closed before the avatar was ready ({code})")


@dataclass
class _Segment:
    pushed: float = 0.0  # seconds of audio sent
    heard: float = 0.0  # seconds the avatar reported as played
    flushed: bool = False
    started: bool = False
    clearing: bool = False


class _AvatarAudioOutput(AudioOutput):
    """Sends the agent's audio to the avatar, which plays it in the room in sync with the lips.

    The avatar reports playback per run of back-to-back audio: ``speech_started``, then ``speech_ended`` with
    the milliseconds played once its queue runs dry (also mid-segment, when the TTS falls behind), or ``cleared``
    after an interruption. Those milliseconds are credited to segments in order, and a segment finishes once
    it is flushed and fully heard.
    """

    def __init__(self, send_ch: utils.aio.Chan[bytes | str]) -> None:
        super().__init__(
            label="TalkingAvatar",
            capabilities=AudioOutputCapabilities(pause=False),
            sample_rate=SAMPLE_RATE,  # the agent session resamples to it
        )
        self._send_ch = send_ch
        self._segments: deque[_Segment] = deque()  # captured and not finished, oldest first
        self._current: _Segment | None = None  # being captured
        self._clears: deque[list[_Segment]] = deque()  # per clear sent, the segments it cuts
        self._closed = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._current is None:
            self._current = _Segment()
            self._segments.append(self._current)
        if self._closed or not frame.samples_per_channel:
            return
        self._current.pushed += frame.duration
        self._send(_AUDIO_HEADER.pack(0, frame.sample_rate) + frame.data.tobytes())

    def flush(self) -> None:
        super().flush()
        if self._current is None:
            return
        self._current.flushed = True
        self._current = None
        if self._closed:
            self._finish_all()
        else:
            self._finish_heard()

    def clear_buffer(self) -> None:
        self._current = None
        if self._closed:
            self._finish_all()
            return
        cut = [s for s in self._segments if not s.clearing]
        if not cut:  # nothing playing or queued (the session also clears after it has)
            return
        for s in cut:
            s.clearing = True
        self._clears.append(cut)
        self._send(json.dumps({"type": "clear"}))

    def _send(self, data: bytes | str) -> None:
        with contextlib.suppress(utils.aio.ChanClosed):
            self._send_ch.send_nowait(data)

    def _on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "speech_started":
            if self._segments and not self._segments[0].started:
                self._segments[0].started = True
                self.on_playback_started(created_at=time.time())
        elif kind == "speech_ended":
            self._credit(list(self._segments), event.get("playedMs", 0) / 1000)
            self._finish_heard()
        elif kind == "cleared":
            cut = self._clears.popleft() if self._clears else []
            self._credit(cut, event.get("playedMs", 0) / 1000)
            for seg in cut:
                self._finish(seg, interrupted=True)

    def _on_closed(self) -> None:
        self._closed = True
        self._finish_all()

    @staticmethod
    def _credit(segments: list[_Segment], played: float) -> None:
        for seg in segments:
            take = min(played, seg.pushed - seg.heard)
            if take > 0:
                seg.heard += take
                played -= take

    def _finish_heard(self) -> None:
        while self._segments:
            seg = self._segments[0]
            if seg.heard > 0 and not seg.started:
                # it played back to back with the segment before it, in the same run, so the
                # avatar sent no speech_started of its own: it started `heard` seconds before now
                seg.started = True
                self.on_playback_started(created_at=time.time() - seg.heard)
            if seg.clearing or not seg.flushed or seg.heard < seg.pushed - _PLAYED_TOLERANCE:
                return
            self._finish(seg, interrupted=False)

    def _finish_all(self) -> None:
        self._clears.clear()
        for seg in list(self._segments):
            if seg is not self._current:
                self._finish(seg, interrupted=True)

    def _finish(self, seg: _Segment, *, interrupted: bool) -> None:
        self._segments.remove(seg)
        position = seg.pushed if not interrupted else min(seg.heard, seg.pushed)
        self.on_playback_finished(playback_position=position, interrupted=interrupted)
