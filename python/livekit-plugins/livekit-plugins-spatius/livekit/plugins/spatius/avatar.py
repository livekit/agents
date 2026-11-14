# Copyright 2026 LiveKit, Inc.
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

import json
import math
import os
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TypedDict

from pydantic import TypeAdapter

import livekit.rtc as rtc
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.types import ErrorKind
from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarStart, ForwardingAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from . import _ogg_opus, _proto
from .log import logger
from .version import __version__

# "auto" resolves the ingress region via the Spatius bootstrap API, falling back
# to us-west if resolution fails; set region/SPATIUS_REGION to pin a concrete region
DEFAULT_REGION = "auto"
_FALLBACK_REGION = "us-west"
_BOOTSTRAP_URL = "https://global.spatialwalk.top/bootstrap"
_REGION_TTL = 300.0
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_OPUS_FRAME_DURATION_MS = 20
DEFAULT_OPUS_APPLICATION = "audio"
SUPPORTED_OPUS_SAMPLE_RATES = _ogg_opus.SAMPLE_RATES
DEFAULT_SESSION_TTL = timedelta(hours=1)
# a prefetched session token is reused until this long before it expires
_TOKEN_MARGIN = timedelta(minutes=1)

# the backend egress consumes this participant attribute to route the playback
# lifecycle RPCs back to the dispatched LiveKit Agents participant
_LIVEKIT_AGENT_IDENTITY_ATTRIBUTE = "livekit_agent_identity"
_RPC_PLAYBACK_STARTED = "lk.playback_started"
_RPC_PLAYBACK_FINISHED = "lk.playback_finished"

# what a server error code says failed: gRPC codes and Spatius' own
_ERROR_KINDS: dict[int, ErrorKind] = {
    3: "invalid_request",
    16: "auth",
    14: "server",
    4001: "quota",
    4002: "quota",
}


class AudioFormat(StrEnum):
    """The audio the avatar service receives."""

    PCM_S16LE = "pcm_s16le"
    """Raw mono 16-bit little-endian PCM."""
    OGG_OPUS = "ogg_opus"
    """One mono Ogg Opus stream per request, encoded by the plugin."""


_PROTO_FORMATS = {AudioFormat.PCM_S16LE: 0, AudioFormat.OGG_OPUS: 1}


class SpatiusException(Exception):
    """Exception raised for Spatius avatar integration errors."""


class _Token(TypedDict):
    sessionToken: str


class _Region(TypedDict):
    current: str


class _Bootstrap(TypedDict):
    region: _Region


_TOKEN = TypeAdapter(_Token)
_BOOTSTRAP = TypeAdapter(_Bootstrap)


@dataclass
class _CachedToken:
    token: str
    expire_at: datetime


# process-wide, so a prewarm() at process start serves the jobs that follow
_regions: dict[str, tuple[str, float]] = {}
_tokens: dict[tuple[str, str, str], _CachedToken] = {}


async def _endpoints(
    *, app_id: str, region: str, console_url: str, ingress_url: str, conn_options: APIConnectOptions
) -> tuple[str, str]:
    """The console and ingress URLs: the explicit ones, else those of the region. Both explicit
    need no region; one explicit composes the other from us-west."""
    if console_url and ingress_url:
        return console_url, ingress_url
    if console_url or ingress_url:
        region = _FALLBACK_REGION
    else:
        region = await _resolve_region(app_id, region, conn_options)
    domain = "spatialwalk.top" if region.startswith("cn-") else "spatius.ai"
    return (
        console_url or f"https://console.{region}.{domain}/v1/console",
        ingress_url or f"wss://api.{region}.{domain}/v2/driveningress",
    )


async def _resolve_region(app_id: str, region: str, conn_options: APIConnectOptions) -> str:
    """A concrete region: ``auto`` asks the bootstrap API, cached for a while; a failure falls
    back to the last region resolved, else us-west, and never raises."""
    if region != DEFAULT_REGION:
        return region
    cached = _regions.get(app_id)
    if cached is not None and time.monotonic() - cached[1] < _REGION_TTL:
        return cached[0]
    try:
        async with http.http_session().post(
            _BOOTSTRAP_URL,
            json={
                "app_id": app_id,
                "sdk_version": __version__,
                "region": DEFAULT_REGION,
                "platform": "python",
            },
            conn_options=conn_options,
        ) as response:
            await response.raise_for_status()
            current = _BOOTSTRAP.validate_json(await response.read())["region"]["current"]
        if not current:
            raise ValueError("the bootstrap answer names no region")
    except (APIError, ValueError) as e:
        fallback = cached[0] if cached is not None else _FALLBACK_REGION
        logger.warning(f"Spatius region resolution failed, using {fallback}: {e}")
        return fallback
    _regions[app_id] = (current, time.monotonic())
    return current


async def _session_token(
    *,
    api_key: str,
    app_id: str,
    console_url: str,
    conn_options: APIConnectOptions,
    prefetch: bool = False,
) -> str:
    """A session token for the console's credentials: one prefetched and still fresh, else a
    new one. A ``prefetch`` fetches one and keeps it for the sessions that follow."""
    key = (api_key, app_id, console_url)
    cached = _tokens.get(key)
    if not prefetch and cached is not None:
        if datetime.now(UTC) < cached.expire_at - _TOKEN_MARGIN:
            return cached.token
        del _tokens[key]
    expire_at = datetime.now(UTC) + DEFAULT_SESSION_TTL
    async with http.http_session().post(
        console_url.rstrip("/") + "/session-tokens",
        headers={"X-Api-Key": api_key},
        json={"expireAt": int(expire_at.timestamp())},
        conn_options=conn_options,
    ) as response:
        await response.raise_for_status()
        token = _TOKEN.validate_json(await response.read())["sessionToken"]
    if prefetch:
        _tokens[key] = _CachedToken(token, expire_at)
    return token


def _server_error(message: _proto.ServerError) -> APIError:
    return APIError(
        f"Spatius avatar session error (server code {message['code']})",
        kind=_ERROR_KINDS.get(message["code"], "server"),
        detail=message["message"] or None,
    )


class AvatarSession(ForwardingAvatarSession):
    """A Spatius avatar session.

    The LiveKit agent produces speech as usual. This plugin forwards the TTS audio
    to Spatius motion server, it joins the LiveKit room to publish synchronized
    avatar audio and motion data. The worker reports playback started and finished
    back to the agent over LiveKit RPC.

    By default the ingress region is resolved automatically through the Spatius
    bootstrap API (``region="auto"``). Pass ``region`` or set the ``SPATIUS_REGION``
    environment variable to pin a concrete region such as ``us-west``.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
        console_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        ingress_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout_seconds: int = 0,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat | str] = NOT_GIVEN,
        bitrate: int = 0,
        opus_frame_duration_ms: int = DEFAULT_OPUS_FRAME_DURATION_MS,
        opus_application: str = DEFAULT_OPUS_APPLICATION,
        extra_params: dict[str, str] | None = None,
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
        self._api_key = _require(api_key, "SPATIUS_API_KEY", "api_key")
        self._app_id = _require(app_id, "SPATIUS_APP_ID", "app_id")
        self._avatar_id = _require(avatar_id, "SPATIUS_AVATAR_ID", "avatar_id")
        self._region = str(_optional(region, "SPATIUS_REGION", DEFAULT_REGION)).strip()
        self._console_endpoint_url = str(
            _optional(console_endpoint_url, "SPATIUS_CONSOLE_ENDPOINT", "")
        )
        self._ingress_endpoint_url = str(
            _optional(ingress_endpoint_url, "SPATIUS_INGRESS_ENDPOINT", "")
        )

        if idle_timeout_seconds < 0:
            raise SpatiusException("idle_timeout_seconds must be greater than or equal to 0")
        if utils.is_given(sample_rate) and sample_rate <= 0:
            raise SpatiusException("sample_rate must be greater than 0")
        if bitrate < 0:
            raise SpatiusException("bitrate must be greater than or equal to 0")

        audio_format_value = _optional(audio_format, "SPATIUS_AUDIO_FORMAT", AudioFormat.OGG_OPUS)
        try:
            self._audio_format = AudioFormat(audio_format_value)
        except ValueError as e:
            raise SpatiusException(f"unsupported audio_format: {audio_format_value}") from e
        if self._audio_format == AudioFormat.OGG_OPUS:
            if opus_frame_duration_ms not in _ogg_opus.FRAME_DURATIONS_MS:
                raise SpatiusException(
                    f"unsupported opus_frame_duration_ms: {opus_frame_duration_ms}"
                )
            if opus_application not in _ogg_opus.APPLICATIONS:
                raise SpatiusException(f"unsupported opus_application: {opus_application}")

        self._idle_timeout_seconds = idle_timeout_seconds
        self._sample_rate = int(sample_rate) if utils.is_given(sample_rate) else None
        self._bitrate = bitrate
        self._extra_params = dict(extra_params or {})
        self._opus_frame_duration_ms = opus_frame_duration_ms
        self._opus_application = opus_application
        self._ws: http.WebSocket | None = None
        # the request being sent, its encoder, and the last request, which a clear interrupts
        self._request: str | None = None
        self._encoder: _ogg_opus.OggOpusEncoder | None = None
        self._last_request: str | None = None

    @property
    def provider(self) -> str:
        return "spatius"

    async def create_session(self, ctx: AvatarStart) -> str:
        sample_rate = self._sample_rate
        if sample_rate is None:
            tts = ctx.agent_session.tts
            sample_rate = tts.sample_rate if tts else DEFAULT_SAMPLE_RATE
        if (
            self._audio_format == AudioFormat.OGG_OPUS
            and sample_rate not in SUPPORTED_OPUS_SAMPLE_RATES
        ):
            supported = ", ".join(str(rate) for rate in sorted(SUPPORTED_OPUS_SAMPLE_RATES))
            raise APIError(
                f"Ogg Opus encoding supports sample rates: {supported} Hz; got {sample_rate} Hz",
                kind="invalid_request",
            )
        self.sample_rate = sample_rate

        console_url, ingress_url = await _endpoints(
            app_id=self._app_id,
            region=self._region,
            console_url=self._console_endpoint_url,
            ingress_url=self._ingress_endpoint_url,
            conn_options=ctx.conn_options,
        )
        token = await _session_token(
            api_key=self._api_key,
            app_id=self._app_id,
            console_url=console_url,
            conn_options=ctx.conn_options,
        )

        # the egress token joins the room as an agent, publishes the avatar tracks, and
        # publishes data so the worker can send the playback lifecycle RPCs
        egress_token = ctx.livekit_token(
            ttl=DEFAULT_SESSION_TTL,
            grants=api.VideoGrants(
                room_join=True,
                room=ctx.room.name,
                can_subscribe=False,
                can_publish_data=True,
                # the worker publishes exactly one avatar video and one audio track
                can_publish_sources=["camera", "microphone"],
            ),
        )
        scheme, _, rest = ingress_url.rstrip("/").partition("://")
        scheme = {"http": "ws", "https": "wss"}.get(scheme, scheme)
        ws = await http.http_session().ws_connect(
            f"{scheme}://{rest}/websocket",
            params={"id": self._avatar_id},
            headers={"X-App-ID": self._app_id, "X-Session-Key": token},
            conn_options=ctx.conn_options,
        )
        try:
            await ws.send_bytes(
                _proto.configure_session(
                    sample_rate=sample_rate,
                    bitrate=self._bitrate,
                    audio_format=_PROTO_FORMATS[self._audio_format],
                    extra_params=self._extra_params,
                    url=ctx.livekit_url,
                    api_token=egress_token,
                    room_name=ctx.room.name,
                    publisher_id=ctx.avatar_identity,
                    extra_attributes={
                        ATTRIBUTE_PUBLISH_ON_BEHALF: ctx.agent_identity,
                        _LIVEKIT_AGENT_IDENTITY_ATTRIBUTE: ctx.agent_identity,
                    },
                    idle_timeout=self._idle_timeout_seconds,
                )
            )
            confirmed = await self._read(ws, ctx.conn_options.read_timeout)
            if confirmed["type"] == "error":
                raise _server_error(confirmed)
            if not confirmed["connection_id"]:
                raise APIError("Spatius confirmed a session without its id", kind="protocol")
        except BaseException:
            await ws.close()
            raise
        self._ws, self._request, self._encoder, self._last_request = ws, None, None, None
        participant = ctx.room.local_participant
        participant.register_rpc_method(_RPC_PLAYBACK_STARTED, self._on_playback_started_rpc)
        participant.register_rpc_method(_RPC_PLAYBACK_FINISHED, self._on_playback_finished_rpc)
        return confirmed["connection_id"]

    async def end_session(self, session_id: str) -> None:
        ws, self._ws = self._ws, None
        if self._room is not None:
            self._room.local_participant.unregister_rpc_method(_RPC_PLAYBACK_STARTED)
            self._room.local_participant.unregister_rpc_method(_RPC_PLAYBACK_FINISHED)
        if ws is not None:
            await ws.close()

    async def receive(self) -> None:
        assert self._ws is not None
        while True:
            message = await self._read(self._ws, None)
            if message["type"] == "error":
                raise _server_error(message)

    async def _read(
        self, ws: http.WebSocket, timeout: float | None
    ) -> _proto.ConfirmSession | _proto.ServerError:
        """The next server message the plugin reads; a close fails by its code."""
        while True:
            msg = await ws.receive(timeout)
            if msg.type is http.WebSocketMessageType.CLOSE:
                raise ws.close_error("Spatius closed the avatar session")
            if msg.type is not http.WebSocketMessageType.BINARY:
                raise APIError("Spatius sent a text frame", kind="protocol", detail=msg.text)
            try:
                message = _proto.decode(msg.binary)
            except ValueError as e:
                raise APIError(
                    "Spatius sent a malformed message", kind="protocol", detail=str(e)
                ) from e
            if message is not None:
                return message

    async def forward(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        assert self._ws is not None
        end = isinstance(frame, AudioSegmentEnd)
        pcm = b"" if isinstance(frame, AudioSegmentEnd) else bytes(frame.data)
        if self._request is None:
            self._request = f"{datetime.now(UTC):%Y%m%d%H%M%S}_{secrets.token_urlsafe(9)}"
            self._last_request = self._request
        if self._audio_format == AudioFormat.OGG_OPUS:
            if self._encoder is None:
                self._encoder = _ogg_opus.OggOpusEncoder(
                    sample_rate=self.sample_rate,
                    bitrate=self._bitrate,
                    frame_duration_ms=self._opus_frame_duration_ms,
                    application=self._opus_application,
                )
            pcm = self._encoder.encode(pcm, end=end)
            if not pcm and not end:
                return
        await self._ws.send_bytes(_proto.audio_input(self._request, pcm, end=end))
        if end:
            self._request = self._encoder = None

    async def clear(self) -> None:
        if self._ws is None or self._last_request is None:
            return
        await self._ws.send_bytes(_proto.interrupt(self._last_request))
        self._request = self._encoder = None

    def _on_playback_started_rpc(self, data: rtc.RpcInvocationData) -> str:
        if not self._is_avatar_caller(data, "playback-started"):
            return "reject"
        self.playback_started()
        return "ok"

    def _on_playback_finished_rpc(self, data: rtc.RpcInvocationData) -> str:
        if not self._is_avatar_caller(data, "playback-finished"):
            return "reject"
        try:
            payload = json.loads(data.payload)
            playback_position = float(payload["playback_position"])
            interrupted = payload["interrupted"]
            if not isinstance(interrupted, bool) or not math.isfinite(playback_position):
                raise ValueError("invalid playback-finished payload")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "invalid Spatius playback-finished RPC payload",
                extra={"payload": data.payload, "error": str(e)},
            )
            return "reject"
        # every notification goes to the output, which does the 1:1 segment accounting and
        # ignores extras on its own
        self.playback_finished(playback_position, interrupted=interrupted)
        return "ok"

    def _is_avatar_caller(self, data: rtc.RpcInvocationData, event: str) -> bool:
        if data.caller_identity == self.avatar_identity:
            return True
        logger.warning(
            f"Spatius {event} RPC received from unexpected participant",
            extra={
                "caller_identity": data.caller_identity,
                "expected_identity": self.avatar_identity,
            },
        )
        return False


def _require(value: NotGivenOr[str], env_var: str, name: str) -> str:
    resolved = value if utils.is_given(value) else os.getenv(env_var)
    if not resolved:
        raise SpatiusException(
            f"{name} must be set either by passing it to AvatarSession or "
            f"by setting the {env_var} environment variable"
        )
    return str(resolved)


def _optional(value: NotGivenOr[str], env_var: str, default: str) -> str:
    return value if utils.is_given(value) else os.getenv(env_var, default)
