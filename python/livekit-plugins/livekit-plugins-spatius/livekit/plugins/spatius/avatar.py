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
from datetime import UTC, datetime, timedelta

import livekit.rtc as rtc
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    create_api_error_from_http,
    utils,
)
from livekit.agents.types import ErrorKind
from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarStart, ForwardingAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from spatius import (
    AudioFormat,
    AvatarSession as SpatiusSDKSession,
    LiveKitEgressConfig,
    OggOpusEncoderConfig,
    new_avatar_session,
)
from spatius.errors import AvatarSDKError, AvatarSDKErrorCode

from .log import logger

# "auto" resolves the ingress region via the Spatius bootstrap API, falling back
# to us-west if resolution fails; set region/SPATIUS_REGION to pin a concrete region
DEFAULT_REGION = "auto"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_AUDIO_FORMAT = AudioFormat.OGG_OPUS
DEFAULT_OPUS_FRAME_DURATION_MS = 20
DEFAULT_OPUS_APPLICATION = "audio"
SUPPORTED_OPUS_SAMPLE_RATES = {8000, 12000, 16000, 24000, 48000}
DEFAULT_SESSION_TTL = timedelta(hours=1)

# the backend egress consumes this participant attribute to route the playback
# lifecycle RPCs back to the dispatched LiveKit Agents participant
_LIVEKIT_AGENT_IDENTITY_ATTRIBUTE = "livekit_agent_identity"
_RPC_PLAYBACK_STARTED = "lk.playback_started"
_RPC_PLAYBACK_FINISHED = "lk.playback_finished"

_KINDS: dict[AvatarSDKErrorCode, ErrorKind] = {
    AvatarSDKErrorCode.sessionTokenExpired: "auth",
    AvatarSDKErrorCode.sessionTokenInvalid: "auth",
    AvatarSDKErrorCode.appIDUnrecognized: "auth",
    AvatarSDKErrorCode.appIDMismatch: "auth",
    AvatarSDKErrorCode.avatarNotFound: "invalid_request",
    AvatarSDKErrorCode.billingRequired: "quota",
    AvatarSDKErrorCode.creditsExhausted: "quota",
    AvatarSDKErrorCode.sessionDurationExceeded: "quota",
    AvatarSDKErrorCode.unsupportedSampleRate: "invalid_request",
    AvatarSDKErrorCode.invalidEgressConfig: "invalid_request",
    AvatarSDKErrorCode.invalidRequest: "invalid_request",
    AvatarSDKErrorCode.connectionFailed: "connection",
    AvatarSDKErrorCode.connectionClosed: "connection",
    AvatarSDKErrorCode.protocolError: "protocol",
}


def _api_error(e: AvatarSDKError) -> APIError:
    """The SDK's failure by its kind: an HTTP answer's status, else its code."""
    if e.http_status is not None:
        return create_api_error_from_http(
            e.message, status=e.http_status, body=e.raw_body, kind=_KINDS.get(e.code)
        )
    return APIError(e.message, kind=_KINDS.get(e.code, "server"), detail=e.server_detail)


class SpatiusException(Exception):
    """Exception raised for Spatius avatar integration errors."""


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
        self._region = str(_optional(region, "SPATIUS_REGION", DEFAULT_REGION))
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

        audio_format_value = _optional(audio_format, "SPATIUS_AUDIO_FORMAT", DEFAULT_AUDIO_FORMAT)
        try:
            self._audio_format = AudioFormat(audio_format_value)
        except ValueError as e:
            raise SpatiusException(f"unsupported audio_format: {audio_format_value}") from e

        self._idle_timeout_seconds = idle_timeout_seconds
        self._sample_rate = int(sample_rate) if utils.is_given(sample_rate) else None
        self._bitrate = bitrate
        self._extra_params = dict(extra_params or {})
        self._opus_encoder = (
            OggOpusEncoderConfig(
                frame_duration_ms=opus_frame_duration_ms,
                application=opus_application,
            )
            if self._audio_format == AudioFormat.OGG_OPUS
            else None
        )
        self._spatius_session: SpatiusSDKSession | None = None
        # the SDK interrupts the last request it was sent
        self._sent = False

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
        sdk = new_avatar_session(
            api_key=self._api_key,
            app_id=self._app_id,
            avatar_id=self._avatar_id,
            region=self._region,
            console_endpoint_url=self._console_endpoint_url,
            ingress_endpoint_url=self._ingress_endpoint_url,
            expire_at=datetime.now(UTC) + DEFAULT_SESSION_TTL,
            livekit_egress=LiveKitEgressConfig(
                url=ctx.livekit_url,
                api_token=egress_token,
                room_name=ctx.room.name,
                publisher_id=ctx.avatar_identity,
                extra_attributes={
                    ATTRIBUTE_PUBLISH_ON_BEHALF: ctx.agent_identity,
                    _LIVEKIT_AGENT_IDENTITY_ATTRIBUTE: ctx.agent_identity,
                },
                idle_timeout=self._idle_timeout_seconds,
            ),
            sample_rate=sample_rate,
            bitrate=self._bitrate,
            audio_format=self._audio_format,
            ogg_opus_encoder=self._opus_encoder,
            extra_params=self._extra_params,
            on_error=lambda e: logger.warning("Spatius avatar session error", exc_info=e),
        )
        try:
            await sdk.init()
            connection_id = await sdk.start()
        except AvatarSDKError as e:
            await sdk.close()
            raise _api_error(e) from e
        except BaseException:
            await sdk.close()
            raise
        self._spatius_session, self._sent = sdk, False
        participant = ctx.room.local_participant
        participant.register_rpc_method(_RPC_PLAYBACK_STARTED, self._on_playback_started_rpc)
        participant.register_rpc_method(_RPC_PLAYBACK_FINISHED, self._on_playback_finished_rpc)
        return connection_id

    async def end_session(self, session_id: str) -> None:
        sdk, self._spatius_session = self._spatius_session, None
        if self._room is not None:
            self._room.local_participant.unregister_rpc_method(_RPC_PLAYBACK_STARTED)
            self._room.local_participant.unregister_rpc_method(_RPC_PLAYBACK_FINISHED)
        if sdk is not None:
            await sdk.close()

    async def forward(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        assert self._spatius_session is not None
        try:
            if isinstance(frame, rtc.AudioFrame):
                await self._spatius_session.send_audio(audio=bytes(frame.data), end=False)
            else:
                await self._spatius_session.send_audio(audio=b"", end=True)
        except AvatarSDKError as e:
            raise _api_error(e) from e
        self._sent = True

    async def clear(self) -> None:
        if self._spatius_session is None or not self._sent:
            return
        try:
            await self._spatius_session.interrupt()
        except AvatarSDKError as e:
            raise _api_error(e) from e

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
