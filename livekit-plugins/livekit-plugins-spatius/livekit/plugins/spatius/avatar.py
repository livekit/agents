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

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarSession as BaseAvatarSession,
    QueueAudioOutput,
)
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from spatius import (
    AudioFormat,
    AvatarSession as SpatiusSDKSession,
    LiveKitEgressConfig,
    OggOpusEncoderConfig,
    new_avatar_session,
)

from .log import logger

DEFAULT_REGION = "us-west"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_AUDIO_FORMAT = AudioFormat.OGG_OPUS
DEFAULT_OPUS_FRAME_DURATION_MS = 20
DEFAULT_OPUS_APPLICATION = "audio"
SUPPORTED_OPUS_SAMPLE_RATES = {8000, 12000, 16000, 24000, 48000}
DEFAULT_SESSION_TTL = timedelta(hours=1)
LIVEKIT_AVATAR_PUBLISH_SOURCES = ["camera", "microphone"]
_AVATAR_AGENT_IDENTITY = "spatius-avatar-agent"
_AVATAR_AGENT_NAME = "spatius-avatar-agent"


class SpatiusException(Exception):
    """Exception raised for Spatius avatar integration errors."""


@dataclass
class _SegmentState:
    req_id: str
    pushed_duration: float = 0.0
    first_frame_at: float | None = None


class AvatarSession(BaseAvatarSession):
    """A Spatius avatar session.

    The LiveKit agent produces speech as usual. This plugin forwards the TTS audio
    to Spatius, and the Spatius avatar worker joins the LiveKit room to publish
    synchronized avatar audio/video.
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
    ) -> None:
        super().__init__()

        resolved_api_key = api_key if utils.is_given(api_key) else os.getenv("SPATIUS_API_KEY")
        if not resolved_api_key:
            raise SpatiusException(
                "api_key must be set either by passing it to AvatarSession or "
                "by setting the SPATIUS_API_KEY environment variable"
            )

        resolved_app_id = app_id if utils.is_given(app_id) else os.getenv("SPATIUS_APP_ID")
        if not resolved_app_id:
            raise SpatiusException(
                "app_id must be set either by passing it to AvatarSession or "
                "by setting the SPATIUS_APP_ID environment variable"
            )

        resolved_avatar_id = (
            avatar_id if utils.is_given(avatar_id) else os.getenv("SPATIUS_AVATAR_ID")
        )
        if not resolved_avatar_id:
            raise SpatiusException(
                "avatar_id must be set either by passing it to AvatarSession or "
                "by setting the SPATIUS_AVATAR_ID environment variable"
            )

        if idle_timeout_seconds < 0:
            raise SpatiusException("idle_timeout_seconds must be greater than or equal to 0")
        if utils.is_given(sample_rate) and sample_rate <= 0:
            raise SpatiusException("sample_rate must be greater than 0")
        if bitrate < 0:
            raise SpatiusException("bitrate must be greater than or equal to 0")

        try:
            audio_format_value = (
                audio_format
                if utils.is_given(audio_format)
                else os.getenv("SPATIUS_AUDIO_FORMAT", DEFAULT_AUDIO_FORMAT)
            )
            resolved_audio_format = AudioFormat(audio_format_value)
        except ValueError as e:
            raise SpatiusException(f"unsupported audio_format: {audio_format_value}") from e

        self._api_key = str(resolved_api_key)
        self._app_id = str(resolved_app_id)
        self._avatar_id = str(resolved_avatar_id)
        self._region = str(
            region if utils.is_given(region) else os.getenv("SPATIUS_REGION", DEFAULT_REGION)
        )
        self._console_endpoint_url = str(
            console_endpoint_url
            if utils.is_given(console_endpoint_url)
            else os.getenv("SPATIUS_CONSOLE_ENDPOINT", "")
        )
        self._ingress_endpoint_url = str(
            ingress_endpoint_url
            if utils.is_given(ingress_endpoint_url)
            else os.getenv("SPATIUS_INGRESS_ENDPOINT", "")
        )
        self._avatar_participant_identity = str(
            avatar_participant_identity
            if utils.is_given(avatar_participant_identity)
            else _AVATAR_AGENT_IDENTITY
        )
        self._avatar_participant_name = str(
            avatar_participant_name
            if utils.is_given(avatar_participant_name)
            else _AVATAR_AGENT_NAME
        )
        self._idle_timeout_seconds = idle_timeout_seconds
        self._sample_rate = sample_rate if utils.is_given(sample_rate) else None
        self._audio_format = resolved_audio_format
        self._bitrate = bitrate
        self._opus_encoder_config = (
            OggOpusEncoderConfig(
                frame_duration_ms=opus_frame_duration_ms,
                application=opus_application,
            )
            if resolved_audio_format == AudioFormat.OGG_OPUS
            else None
        )

        self._spatius_session: SpatiusSDKSession | None = None
        self._agent_session: AgentSession | None = None
        self._audio_buffer: QueueAudioOutput | None = None
        self._user_state_changed_handler_registered = False
        self._session_close_handler_registered = False
        self._interrupt_task: asyncio.Task[None] | None = None
        self._main_task: asyncio.Task | None = None
        self._initialized = False
        self._active_segment: _SegmentState | None = None
        self._segment_finalize_lock = asyncio.Lock()

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "spatius"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        livekit_room_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Start the Spatius avatar session and attach it to the agent output."""
        if self._initialized:
            logger.warning("Avatar session already initialized")
            return

        await super().start(agent_session, room)

        resolved_livekit_url = (
            livekit_url if utils.is_given(livekit_url) else os.getenv("LIVEKIT_URL")
        )
        resolved_livekit_api_key = (
            livekit_api_key if utils.is_given(livekit_api_key) else os.getenv("LIVEKIT_API_KEY")
        )
        resolved_livekit_api_secret = (
            livekit_api_secret
            if utils.is_given(livekit_api_secret)
            else os.getenv("LIVEKIT_API_SECRET")
        )
        if (
            not resolved_livekit_url
            or not resolved_livekit_api_key
            or not resolved_livekit_api_secret
        ):
            raise SpatiusException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set by arguments or environment variables"
            )

        agent_room_name = room.name
        room_name = str(livekit_room_name) if utils.is_given(livekit_room_name) else agent_room_name
        if not room_name:
            raise SpatiusException("livekit_room_name must not be empty")

        local_participant_identity = self._resolve_local_participant_identity(room)
        logger.debug(
            "starting Spatius avatar session",
            extra={
                "room": room_name,
                "agent_room": agent_room_name,
                "region": self._region,
            },
        )

        egress_attributes = {ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity}
        livekit_token = (
            api.AccessToken(
                api_key=str(resolved_livekit_api_key),
                api_secret=str(resolved_livekit_api_secret),
            )
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_ttl(DEFAULT_SESSION_TTL)
            .with_attributes(egress_attributes)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_subscribe=False,
                    can_publish_data=False,
                    can_publish_sources=LIVEKIT_AVATAR_PUBLISH_SOURCES,
                )
            )
            .to_jwt()
        )

        livekit_egress = LiveKitEgressConfig(
            url=str(resolved_livekit_url),
            api_token=livekit_token,
            room_name=room_name,
            publisher_id=self._avatar_participant_identity,
            extra_attributes=egress_attributes,
            idle_timeout=self._idle_timeout_seconds,
        )

        resolved_sample_rate = self._sample_rate
        if resolved_sample_rate is None:
            resolved_sample_rate = (
                agent_session.tts.sample_rate if agent_session.tts else DEFAULT_SAMPLE_RATE
            )
        if resolved_sample_rate <= 0:
            raise SpatiusException("sample_rate must be greater than 0")
        if (
            self._audio_format == AudioFormat.OGG_OPUS
            and resolved_sample_rate not in SUPPORTED_OPUS_SAMPLE_RATES
        ):
            raise SpatiusException(
                "Ogg Opus encoding supports sample rates: "
                + ", ".join(str(rate) for rate in sorted(SUPPORTED_OPUS_SAMPLE_RATES))
                + f" Hz; got {resolved_sample_rate} Hz"
            )

        self._agent_session = agent_session

        try:
            self._spatius_session = new_avatar_session(
                api_key=self._api_key,
                app_id=self._app_id,
                avatar_id=self._avatar_id,
                region=self._region,
                console_endpoint_url=self._console_endpoint_url,
                ingress_endpoint_url=self._ingress_endpoint_url,
                expire_at=datetime.now(timezone.utc) + DEFAULT_SESSION_TTL,
                livekit_egress=livekit_egress,
                sample_rate=resolved_sample_rate,
                bitrate=self._bitrate,
                audio_format=self._audio_format,
                ogg_opus_encoder=self._opus_encoder_config,
            )
            await self._spatius_session.init()
            await self._spatius_session.start()

            self._audio_buffer = QueueAudioOutput(sample_rate=resolved_sample_rate)
            await self._audio_buffer.start()
            self._audio_buffer.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]

            agent_session.output.replace_audio_tail(self._audio_buffer)
            self._main_task = asyncio.create_task(
                self._run_main_task(),
                name="spatius_avatar_audio_forwarder",
            )
            self._initialized = True

            agent_session.on("user_state_changed", self._on_user_state_changed)
            self._user_state_changed_handler_registered = True
            agent_session.on("close", self._on_session_close)
            self._session_close_handler_registered = True

        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception as e:
            logger.debug("Spatius avatar session startup failed", exc_info=True)
            await self.aclose()
            raise SpatiusException(
                self._build_start_error_message(
                    error=e,
                    room_name=room_name,
                    sample_rate=resolved_sample_rate,
                )
            ) from None

    def _build_start_error_message(
        self,
        *,
        error: Exception,
        room_name: str,
        sample_rate: int,
    ) -> str:
        return (
            "Failed to start Spatius avatar session. "
            "Check Spatius credentials, LiveKit room auth/token configuration, "
            "region/endpoint URLs, and outbound network access. "
            f"room={room_name}, avatar_id={self._avatar_id}, region={self._region}, "
            f"sample_rate={sample_rate}, audio_format={self._audio_format.value}. "
            f"Reason: {self._format_error_reason(error)}"
        )

    @staticmethod
    def _resolve_local_participant_identity(room: rtc.Room) -> str:
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            return job_ctx.local_participant_identity
        if room.isconnected():
            return room.local_participant.identity
        raise SpatiusException("failed to get local participant identity")

    @staticmethod
    def _format_error_reason(error: BaseException) -> str:
        root_error = error
        seen_errors: set[int] = set()

        while id(root_error) not in seen_errors:
            seen_errors.add(id(root_error))
            next_error = root_error.__cause__ or (
                None if root_error.__suppress_context__ else root_error.__context__
            )
            if next_error is None:
                break
            root_error = next_error

        message = str(root_error) or str(error)
        if message:
            return f"{type(root_error).__name__}: {message}"
        return type(root_error).__name__

    async def _run_main_task(self) -> None:
        if not self._audio_buffer or not self._spatius_session:
            return

        try:
            async for item in self._audio_buffer:
                if isinstance(item, rtc.AudioFrame):
                    await self._send_audio_frame(item)
                elif isinstance(item, AudioSegmentEnd):
                    if not await self._finalize_active_segment(source="segment_end"):
                        logger.debug("Avatar segment end received without an active request")
        except asyncio.CancelledError:
            logger.debug("Spatius avatar audio forwarder cancelled")
        except Exception as e:
            logger.error("Error in Spatius avatar audio forwarder", exc_info=e)

    async def _send_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._spatius_session:
            return

        req_id = await self._spatius_session.send_audio(audio=bytes(frame.data), end=False)
        segment = self._active_segment
        if segment is None:
            segment = _SegmentState(req_id=req_id, first_frame_at=time.time())
            self._active_segment = segment
            logger.debug("Spatius avatar first audio frame", extra={"request_id": req_id})
        elif segment.req_id != req_id:
            logger.warning(
                "Spatius request ID changed within LiveKit audio segment",
                extra={"previous_request_id": segment.req_id, "request_id": req_id},
            )
            segment.req_id = req_id

        segment.pushed_duration += frame.duration

    async def _finalize_active_segment(self, *, source: str) -> bool:
        if self._active_segment is None or not self._spatius_session:
            return False

        async with self._segment_finalize_lock:
            segment = self._active_segment
            if segment is None:
                return False

            req_id = await self._spatius_session.send_audio(audio=b"", end=True)
            if req_id != segment.req_id:
                logger.warning(
                    "Spatius request ID changed while finalizing audio segment",
                    extra={"previous_request_id": segment.req_id, "request_id": req_id},
                )
                segment.req_id = req_id

            self._active_segment = None
            self._notify_segment_finished(segment, interrupted=False, reason=source)
            return True

    def _notify_segment_finished(
        self, segment: _SegmentState, *, interrupted: bool, reason: str
    ) -> None:
        playback_position = (
            self._estimate_interrupted_playback_position(segment)
            if interrupted
            else segment.pushed_duration
        )

        if self._audio_buffer:
            self._audio_buffer.notify_playback_finished(
                playback_position=playback_position,
                interrupted=interrupted,
            )

        logger.debug(
            "Spatius avatar segment playback completed",
            extra={
                "request_id": segment.req_id,
                "reason": reason,
                "interrupted": interrupted,
                "playback_position": playback_position,
                "pushed_duration": segment.pushed_duration,
            },
        )

    @staticmethod
    def _estimate_interrupted_playback_position(segment: _SegmentState) -> float:
        if segment.first_frame_at is None:
            return 0.0

        elapsed = max(0.0, time.time() - segment.first_frame_at)
        return min(segment.pushed_duration, elapsed)

    def _complete_active_segment(self, *, interrupted: bool, reason: str) -> bool:
        segment = self._active_segment
        if segment is None:
            return False

        self._active_segment = None
        self._notify_segment_finished(segment, interrupted=interrupted, reason=reason)
        return True

    def _on_clear_buffer(self) -> None:
        self._schedule_interrupt()

    def _on_user_state_changed(self, ev: Any) -> None:
        if getattr(ev, "new_state", None) == "speaking":
            self._schedule_interrupt()

    def _on_session_close(self, _: Any) -> None:
        asyncio.create_task(self.aclose())

    def _schedule_interrupt(self) -> None:
        if not self._spatius_session:
            return
        if self._active_segment is None:
            return
        if self._interrupt_task and not self._interrupt_task.done():
            return

        self._interrupt_task = asyncio.create_task(self._handle_interrupt())
        self._interrupt_task.add_done_callback(self._on_interrupt_task_done)

    def _on_interrupt_task_done(self, task: asyncio.Task[None]) -> None:
        if self._interrupt_task is task:
            self._interrupt_task = None

    async def _handle_interrupt(self) -> None:
        if not self._spatius_session:
            return

        try:
            interrupted_id = await self._spatius_session.interrupt()

            async with self._segment_finalize_lock:
                self._complete_active_segment(interrupted=True, reason="interrupt")

            logger.debug("Spatius avatar interrupted", extra={"request_id": interrupted_id})
        except Exception as e:
            logger.warning("Failed to interrupt Spatius avatar", exc_info=e)

    async def aclose(self) -> None:
        if self._agent_session and self._user_state_changed_handler_registered:
            self._agent_session.off("user_state_changed", self._on_user_state_changed)
            self._user_state_changed_handler_registered = False
        if self._agent_session and self._session_close_handler_registered:
            self._agent_session.off("close", self._on_session_close)
            self._session_close_handler_registered = False

        if self._interrupt_task:
            await utils.aio.cancel_and_wait(self._interrupt_task)
            self._interrupt_task = None

        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None

        self._complete_active_segment(interrupted=True, reason="session_close")

        if self._audio_buffer:
            await self._audio_buffer.aclose()
            self._audio_buffer = None

        if self._spatius_session:
            try:
                await self._spatius_session.close()
                logger.debug("Spatius avatar session closed")
            except Exception as e:
                logger.warning("Error closing Spatius avatar session", exc_info=e)
            finally:
                self._spatius_session = None

        await super().aclose()

        self._initialized = False
        self._agent_session = None
