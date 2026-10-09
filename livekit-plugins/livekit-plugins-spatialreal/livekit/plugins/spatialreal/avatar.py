# Copyright 2026 SpatialReal.
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
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarSession as BaseAvatarSession
from livekit.agents.voice.io import AudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from spatialreal import (
    AvatarSession as AvatarkitSession,
    InterruptReason,
    LiveKitEgressConfig,
    PlaybackSignal,
    PlaybackState,
    PlaybackStateEvent,
    new_avatar_session,
)

from .log import logger
from .resumable_queue_io import ResumableQueueAudioOutput

__all__ = [
    "AvatarPlaybackStartedEvent",
    "AvatarProviderConnectionEvent",
    "AvatarSession",
    "SpatialRealException",
]

DEFAULT_AVATAR_PARTICIPANT_IDENTITY = "spatialreal-avatar"
DEFAULT_AVATAR_PARTICIPANT_NAME = "spatialreal-avatar"
DEFAULT_SAMPLE_RATE = 24000
MIN_COMPLETION_TIMEOUT_SECONDS = 3.0
LIVEKIT_ACTIVITY_COMPLETION_BUFFER_SECONDS = 1.0
LIVEKIT_ACTIVITY_START_WAIT_SECONDS = 8.0
LIVEKIT_ACTIVITY_MAX_OVERRUN_SECONDS = 8.0
LIVEKIT_ACTIVITY_RECHECK_SECONDS = 0.5
ACTIVE_SEGMENT_IDLE_END_SECONDS = 1.0
# int16 amplitude (~ -50 dBFS): decoded Opus "silence" is rarely exact zeros,
# so playback-start detection needs a small energy floor instead of any-nonzero
AVATAR_AUDIO_ACTIVITY_THRESHOLD = 100
# egress-mode frame retransmission (ALR) can deliver end=true more than once
# per req_id; remember recent completions so duplicates are dropped
COMPLETED_REQ_ID_HISTORY = 32
DEFAULT_RESUME_BUFFER_MAX_SECONDS = 180.0
DEFAULT_SESSION_TTL = timedelta(hours=1)
# TTL of the LiveKit room-join token the egress worker uses to (re)join the room.
# This is NOT the SpatialReal session lifetime — the egress worker keeps re-joining
# the SFU across the whole session, including after a provider-WS reconnect, so the
# token must outlive any realistic conversation. When it expires mid-session, egress
# rejoins with a stale token and the SFU returns 401, which the plugin surfaces as an
# unrecoverable provider failure and the avatar drops out. Default 24h; override with
# SPATIALREAL_LIVEKIT_TOKEN_TTL_SECONDS or the livekit_token_ttl constructor arg.
DEFAULT_LIVEKIT_TOKEN_TTL = timedelta(hours=24)
LIVEKIT_AVATAR_PUBLISH_SOURCES = ["camera", "microphone"]
PROVIDER_RECONNECT_DELAYS_SECONDS = (0.0, 0.5, 1.0)
PROVIDER_CONNECT_TIMEOUT_SECONDS = 15.0
PROVIDER_CLOSE_TIMEOUT_SECONDS = 5.0

# No endpoint lives here: the SpatialReal SDK resolves them from the deployment it is told to
# use, so a new region or a moved deployment never needs a new release of this plugin.


class SpatialRealException(Exception):
    """Exception raised for SpatialReal avatar integration errors."""


class _ProviderClearDuringSend(RuntimeError):
    """LiveKit cleared playback while a provider send was queued or active."""


@dataclass(frozen=True)
class AvatarPlaybackStartedEvent:
    """Observed playback start for one SpatialReal request."""

    request_id: str
    source: str
    observed_at: float


@dataclass(frozen=True)
class AvatarProviderConnectionEvent:
    """Provider connection state emitted without including conversation content."""

    state: Literal["connected", "recovering", "recovered", "failed"]
    reason: str
    attempt: int
    generation: int
    observed_at: float


@dataclass
class _BufferedAudioFrame:
    audio: bytes
    duration: float
    sample_rate: int
    num_channels: int


@dataclass
class _SegmentState:
    req_id: str
    pushed_duration: float = 0.0
    first_frame_at: float | None = None
    playback_started: bool = False
    playback_started_at: float | None = None
    playback_start_source: str | None = None
    provider_playback_completed: bool = False
    completion_timeout_task: asyncio.Task[None] | None = None
    audio_frames: deque[_BufferedAudioFrame] = field(default_factory=deque)
    buffered_duration: float = 0.0
    buffer_start_position: float = 0.0
    playback_position_offset: float = 0.0
    attempt_duration: float = 0.0
    input_finalized: bool = False
    provider_events_trusted: bool = True
    # set while the server holds playback (native pause); time held is not playback
    native_paused_at: float | None = None


class AvatarSession(BaseAvatarSession):
    """A SpatialReal avatar session.

    The LiveKit agent produces speech as usual. This plugin forwards the TTS audio
    to SpatialReal, and the SpatialReal avatar worker joins the LiveKit room to publish
    synchronized avatar audio/video.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        environment: NotGivenOr[str] = NOT_GIVEN,
        console_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        ingress_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout_seconds: int = 0,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        livekit_token_ttl_seconds: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        resolved_api_key = api_key if utils.is_given(api_key) else os.getenv("SPATIALREAL_API_KEY")
        if not resolved_api_key:
            raise SpatialRealException(
                "api_key must be set either by passing it to AvatarSession or "
                "by setting the SPATIALREAL_API_KEY environment variable"
            )

        resolved_app_id = app_id if utils.is_given(app_id) else os.getenv("SPATIALREAL_APP_ID")
        if not resolved_app_id:
            raise SpatialRealException(
                "app_id must be set either by passing it to AvatarSession or "
                "by setting the SPATIALREAL_APP_ID environment variable"
            )

        resolved_avatar_id = (
            avatar_id if utils.is_given(avatar_id) else os.getenv("SPATIALREAL_AVATAR_ID")
        )
        if not resolved_avatar_id:
            raise SpatialRealException(
                "avatar_id must be set either by passing it to AvatarSession or "
                "by setting the SPATIALREAL_AVATAR_ID environment variable"
            )

        if idle_timeout_seconds < 0:
            raise SpatialRealException("idle_timeout_seconds must be greater than or equal to 0")
        if utils.is_given(sample_rate) and sample_rate <= 0:
            raise SpatialRealException("sample_rate must be greater than 0")

        resolved_livekit_token_ttl = (
            int(livekit_token_ttl_seconds)
            if utils.is_given(livekit_token_ttl_seconds)
            else self._int_env(
                "SPATIALREAL_LIVEKIT_TOKEN_TTL_SECONDS",
                int(DEFAULT_LIVEKIT_TOKEN_TTL.total_seconds()),
            )
        )
        if resolved_livekit_token_ttl <= 0:
            raise SpatialRealException("livekit_token_ttl_seconds must be greater than 0")
        self._livekit_token_ttl = timedelta(seconds=resolved_livekit_token_ttl)

        self._api_key = str(resolved_api_key)
        self._app_id = str(resolved_app_id)
        self._avatar_id = str(resolved_avatar_id)
        # The SDK owns where a deployment lives: it resolves the endpoints from `environment`
        # (its own default when unset). The two URLs are overrides for a deployment SpatialReal
        # hands you an address for; empty means "whatever the environment resolves to".
        self._environment = str(
            environment
            if utils.is_given(environment)
            else os.getenv("SPATIALREAL_ENVIRONMENT") or ""
        )
        self._console_endpoint_url = str(
            console_endpoint_url
            if utils.is_given(console_endpoint_url)
            else os.getenv("SPATIALREAL_CONSOLE_ENDPOINT") or ""
        )
        self._ingress_endpoint_url = str(
            ingress_endpoint_url
            if utils.is_given(ingress_endpoint_url)
            else os.getenv("SPATIALREAL_INGRESS_ENDPOINT") or ""
        )
        self._avatar_participant_identity = str(
            avatar_participant_identity
            if utils.is_given(avatar_participant_identity)
            else DEFAULT_AVATAR_PARTICIPANT_IDENTITY
        )
        self._avatar_participant_name = str(
            avatar_participant_name
            if utils.is_given(avatar_participant_name)
            else DEFAULT_AVATAR_PARTICIPANT_NAME
        )
        self._idle_timeout_seconds = idle_timeout_seconds
        self._sample_rate = sample_rate if utils.is_given(sample_rate) else None

        self._avatarkit_session: AvatarkitSession | None = None
        self._agent_session: AgentSession | None = None
        self._room: rtc.Room | None = None
        self._audio_buffer: ResumableQueueAudioOutput | None = None
        self._original_audio_output: Any | None = None
        self._original_audio_tail: AudioOutput | None = None
        self._audio_output_attached = False
        self._main_task: asyncio.Task | None = None
        self._initialized = False
        self._segments: dict[str, _SegmentState] = {}
        self._pending_segment_ids: deque[str] = deque()
        self._early_provider_started_ids: set[str] = set()
        self._early_provider_completed_ids: set[str] = set()
        self._recently_completed_req_ids: deque[str] = deque(maxlen=COMPLETED_REQ_ID_HISTORY)
        self._active_req_id: str | None = None
        self._represented_participant_identity: str | None = None
        self._avatar_is_speaking = False
        self._avatar_audio_stream: rtc.AudioStream | None = None
        self._avatar_audio_monitor_task: asyncio.Task[None] | None = None
        self._active_segment_idle_end_task: asyncio.Task[None] | None = None
        self._segment_finalize_lock = asyncio.Lock()
        self._interrupt_lock = asyncio.Lock()
        self._provider_io_lock = asyncio.Lock()
        self._provider_reconnect_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._closing = False
        self._server_playback_control = False
        self._pause_requested = False
        self._pause_requested_at: float | None = None
        self._discard_requested = False
        self._paused_segment: _SegmentState | None = None
        self._drop_frames_until_segment_end = False
        self._provider_send_in_flight = False
        self._clear_during_provider_send = False
        self._clear_generation = 0
        self._provider_recovery_in_flight = False
        self._clear_during_provider_recovery = False
        self._provider_reconnect_task: asyncio.Task[bool] | None = None
        self._provider_connection_state = "disconnected"
        self._provider_generation = 0
        self._provider_session_generation = 0
        self._provider_expected_close_generations: set[int] = set()
        self._provider_recovery_pending_playback = False
        self._provider_terminal_failure = False
        self._livekit_egress: LiveKitEgressConfig | None = None
        self._resolved_sample_rate: int | None = None
        self._local_segment_sequence = 0
        self._resume_buffer_max_seconds = self._float_env(
            "SPATIALREAL_RESUME_BUFFER_MAX_SECONDS",
            DEFAULT_RESUME_BUFFER_MAX_SECONDS,
            minimum=10.0,
        )

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "spatialreal"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Start the SpatialReal avatar session and attach it to the agent output."""
        if self._initialized:
            logger.warning("Avatar session already initialized")
            return

        # Validate before super().start(): it registers listeners and a join task, and a
        # raise after that point would leave them behind with no session to clean them up.
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
            raise SpatialRealException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set by arguments or environment variables"
            )

        await super().start(agent_session, room)

        room_name = room.name
        local_participant_identity = self._resolve_local_participant_identity(room)
        self._represented_participant_identity = local_participant_identity
        logger.debug(
            "starting SpatialReal avatar session",
            extra={"room": room_name},
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
            .with_ttl(self._livekit_token_ttl)
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
            raise SpatialRealException("sample_rate must be greater than 0")

        self._agent_session = agent_session
        self._room = room
        self._livekit_egress = livekit_egress
        self._resolved_sample_rate = resolved_sample_rate
        self._original_audio_output = agent_session.output.audio
        self._original_audio_tail = self._resolve_audio_tail(agent_session.output.audio)

        try:
            last_provider_error: Exception | None = None
            for attempt, delay in enumerate(PROVIDER_RECONNECT_DELAYS_SECONDS, start=1):
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    async with self._provider_io_lock:
                        await self._connect_fresh_provider_session()
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_provider_error = error
                    logger.warning(
                        "SpatialReal provider startup attempt failed",
                        exc_info=error,
                        extra={"attempt": attempt},
                    )
            else:
                assert last_provider_error is not None
                raise last_provider_error

            self._provider_connection_state = "connected"
            self._emit_provider_connection_state(
                state="connected",
                reason="initial_start",
                attempt=attempt,
            )

            # wait_playback_start defers the framework's playback_started event
            # (transcript sync, first-frame metrics) until we observe the avatar
            # actually playing in the room, instead of firing when the first TTS
            # frame is merely handed to SpatialReal.
            self._audio_buffer = ResumableQueueAudioOutput(
                sample_rate=resolved_sample_rate,
                wait_playback_start=True,
            )
            await self._audio_buffer.start()
            self._audio_buffer.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]
            self._audio_buffer.on("pause", self._on_pause)  # type: ignore[arg-type]
            self._audio_buffer.on("resume", self._on_resume)  # type: ignore[arg-type]

            # Until the SpatialReal egress backend emits request-correlated
            # playback lifecycle events, playback start is observed client-side
            # from the avatar's published LiveKit audio track (primary) and the
            # room's active-speaker state (secondary).
            room.on("active_speakers_changed", self._on_active_speakers_changed)
            room.on("track_published", self._on_track_published)
            room.on("track_subscribed", self._on_track_subscribed)
            room.on("track_unsubscribed", self._on_track_unsubscribed)

            for participant in room.remote_participants.values():
                if not self._is_avatar_output_participant(participant):
                    continue
                for publication in participant.track_publications.values():
                    if publication.kind != rtc.TrackKind.KIND_AUDIO:
                        continue
                    publication.set_subscribed(True)
                    if publication.track is not None:
                        self._start_avatar_audio_monitor(publication.track)

            # keep output wrappers (TranscriptSynchronizer, recorder, ...) in
            # the chain: swap only the tail sink
            agent_session.output.replace_audio_tail(self._audio_buffer)
            self._audio_output_attached = True
            self._main_task = asyncio.create_task(
                self._run_main_task(),
                name="spatialreal_avatar_audio_forwarder",
            )
            self._initialized = True

            # Interruption is owned entirely by the framework. ``pause`` holds
            # a candidate interruption, ``resume`` restores a rejected one,
            # and ``clear_buffer`` confirms a true interruption. We do NOT
            # interrupt on ``user_state_changed`` -> "speaking" — that fires on
            # raw VAD, upstream of every framework gate, so coughs, throat
            # clears and single-word fragments would truncate the avatar even
            # when turn handling would not treat them as a real interruption.
            # This matches the framework's reference receiver
            # (livekit/agents/voice/avatar/_runner.py:_on_clear_buffer); none of
            # the official avatar plugins listen to user_state_changed.

            @agent_session.on("close")
            def _on_session_close(_: Any) -> None:
                self._spawn_background_task(self.aclose(), name="spatialreal_avatar_session_close")

        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception as e:
            logger.debug("SpatialReal avatar session startup failed", exc_info=True)
            await self.aclose()
            raise SpatialRealException(
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
            "Failed to start SpatialReal avatar session. "
            "Check SpatialReal credentials, LiveKit room auth/token configuration, "
            "endpoint URLs, and outbound network access. "
            f"room={room_name}, avatar_id={self._avatar_id}, "
            f"ingress_endpoint_url={self._ingress_endpoint_url or '(from environment)'}, "
            f"sample_rate={sample_rate}. Reason: {self._format_error_reason(error)}"
        )

    @staticmethod
    def _resolve_local_participant_identity(room: rtc.Room) -> str:
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            return job_ctx.local_participant_identity
        if room.isconnected():
            return room.local_participant.identity
        raise SpatialRealException("failed to get local participant identity")

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

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("%s=%r is invalid; using %d", name, raw, default)
            return default
        if value <= 0:
            logger.warning("%s=%d must be > 0; using %d", name, value, default)
            return default
        return value

    @staticmethod
    def _float_env(name: str, default: float, *, minimum: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("%s=%r is invalid; using %.2f", name, raw, default)
            return default
        if value < minimum:
            logger.warning(
                "%s=%.2f is below minimum %.2f; using %.2f",
                name,
                value,
                minimum,
                default,
            )
            return default
        return value

    def _spawn_background_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str,
    ) -> asyncio.Task[Any] | None:
        if self._closing:
            coro.close()
            return None
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @staticmethod
    def _resolve_audio_tail(sink: AudioOutput | None) -> AudioOutput | None:
        while sink is not None and sink.next_in_chain is not None:
            sink = sink.next_in_chain
        return sink

    async def _run_main_task(self) -> None:
        if not self._audio_buffer or not self._avatarkit_session:
            return

        try:
            async for item in self._audio_buffer:
                try:
                    if self._drop_frames_until_segment_end:
                        if isinstance(item, AudioSegmentEnd):
                            self._drop_frames_until_segment_end = False
                        continue
                    if isinstance(item, rtc.AudioFrame):
                        await self._send_audio_frame(item)
                    elif isinstance(item, AudioSegmentEnd):
                        if not await self._finalize_active_segment(source="segment_end"):
                            logger.debug("Avatar segment end received without an active request")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await self._recover_provider_forwarding_error(item=item, error=e)
        except asyncio.CancelledError:
            logger.debug("SpatialReal avatar audio forwarder cancelled")
        except Exception as e:
            logger.error("Error in SpatialReal avatar audio forwarder", exc_info=e)

    async def _recover_provider_forwarding_error(
        self,
        *,
        item: rtc.AudioFrame | AudioSegmentEnd,
        error: Exception,
    ) -> None:
        """Release the current framework playout and keep forwarding later turns."""
        logger.error(
            "SpatialReal provider audio forwarding failed; abandoning affected segment",
            exc_info=error,
        )
        clear_before_recovery = self._clear_during_provider_send
        self._clear_during_provider_send = False
        self._provider_recovery_in_flight = True
        self._clear_during_provider_recovery = False
        try:
            async with self._provider_io_lock:
                if not self._avatarkit_session:
                    return

                try:
                    await self._avatarkit_session.interrupt()
                except Exception:
                    logger.warning(
                        "Failed to stop SpatialReal provider after audio forwarding error",
                        exc_info=True,
                    )

                if (
                    isinstance(item, rtc.AudioFrame)
                    and not self._segments
                    and self._paused_segment is None
                ):
                    local_segment = self._new_local_segment()
                    self._segments[local_segment.req_id] = local_segment
                    self._append_audio_frame(
                        local_segment,
                        _BufferedAudioFrame(
                            audio=bytes(item.data),
                            duration=item.duration,
                            sample_rate=item.sample_rate,
                            num_channels=item.num_channels,
                        ),
                        sent_to_provider=False,
                    )

                self._cancel_active_segment_idle_end()
                self._complete_all_segments(
                    interrupted=True,
                    reason="provider_forwarding_error",
                )
                self._drop_frames_until_segment_end = (
                    isinstance(item, rtc.AudioFrame)
                    and not clear_before_recovery
                    and not self._clear_during_provider_recovery
                )
                self._active_req_id = None
                self._paused_segment = None
                self._pause_requested = False
                self._pause_requested_at = None
                self._discard_requested = False
        finally:
            self._provider_recovery_in_flight = False
            self._clear_during_provider_recovery = False

        recovered = await self._ensure_provider_connection(
            reason=f"audio_forwarding_error:{self._format_error_reason(error)}"
        )
        if not recovered:
            self._drop_frames_until_segment_end = True

    def _create_provider_session(self) -> AvatarkitSession:
        if self._livekit_egress is None or self._resolved_sample_rate is None:
            raise SpatialRealException("SpatialReal provider configuration is unavailable")

        self._provider_generation += 1
        generation = self._provider_generation
        provider = new_avatar_session(
            api_key=self._api_key,
            app_id=self._app_id,
            avatar_id=self._avatar_id,
            **self._endpoint_options(),
            expire_at=datetime.now(timezone.utc) + DEFAULT_SESSION_TTL,
            livekit_egress=self._livekit_egress,
            sample_rate=self._resolved_sample_rate,
            on_playback=self._on_playback_signal,
            on_playback_state=self._on_playback_state,
            on_error=lambda error: self._on_provider_error(generation, error),
            on_close=lambda: self._on_provider_close(generation),
        )
        self._provider_session_generation = generation
        return provider

    async def _connect_fresh_provider_session(self) -> AvatarkitSession:
        provider = self._create_provider_session()
        generation = self._provider_session_generation
        self._avatarkit_session = provider

        async def _start() -> None:
            await provider.init()
            await provider.start()

        try:
            await asyncio.wait_for(_start(), timeout=PROVIDER_CONNECT_TIMEOUT_SECONDS)
        except BaseException:
            await self._close_provider_session(provider, generation=generation)
            if self._avatarkit_session is provider:
                self._avatarkit_session = None
            raise

        # When the backend advertises server-side playback control we drive
        # false-interruption pause/resume natively (no interrupt + re-send).
        # Feature-detected per connection so a mixed fleet degrades gracefully.
        self._server_playback_control = "playback_control" in getattr(provider, "capabilities", ())
        return provider

    async def _close_provider_session(
        self,
        provider: AvatarkitSession,
        *,
        generation: int,
    ) -> None:
        self._provider_expected_close_generations.add(generation)
        try:
            await asyncio.wait_for(
                provider.close(),
                timeout=PROVIDER_CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "SpatialReal provider close did not complete cleanly",
                exc_info=error,
                extra={"generation": generation},
            )
        finally:
            self._provider_expected_close_generations.discard(generation)

    def _on_provider_error(self, generation: int, error: Exception) -> None:
        if generation != self._provider_session_generation or self._closing:
            return
        logger.error(
            "SpatialReal provider websocket reported an error",
            exc_info=error,
            extra={"generation": generation},
        )
        self._schedule_provider_recovery(
            reason=f"provider_error:{self._format_error_reason(error)}"
        )

    def _on_provider_close(self, generation: int) -> None:
        if (
            generation in self._provider_expected_close_generations
            or generation != self._provider_session_generation
            or self._closing
        ):
            return
        logger.warning(
            "SpatialReal provider websocket closed unexpectedly",
            extra={"generation": generation},
        )
        # a server-side hold dies with its connection; fall back to duration completion
        self._release_native_hold()
        # Audio already handed to the closed connection was never played out. Finish those
        # segments as interrupted — the same treatment a forwarding error gets — so the
        # framework is not told later that they played in full.
        self._complete_all_segments(interrupted=True, reason="provider_closed")
        self._schedule_provider_recovery(reason="provider_closed")

    def _schedule_provider_recovery(self, *, reason: str) -> asyncio.Task[bool] | None:
        if self._closing:
            return None
        task = self._provider_reconnect_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(
            self._recover_provider_connection(reason=reason),
            name="spatialreal_provider_reconnect",
        )
        self._provider_reconnect_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _ensure_provider_connection(self, *, reason: str) -> bool:
        task = self._schedule_provider_recovery(reason=reason)
        if task is None:
            return False
        return await asyncio.shield(task)

    async def _recover_provider_connection(self, *, reason: str) -> bool:
        async with self._provider_reconnect_lock:
            if self._closing:
                return False
            if self._provider_terminal_failure:
                return False

            self._provider_connection_state = "recovering"
            self._provider_recovery_pending_playback = False
            self._emit_provider_connection_state(
                state="recovering",
                reason=reason,
                attempt=0,
            )

            last_error: Exception | None = None
            for attempt, delay in enumerate(PROVIDER_RECONNECT_DELAYS_SECONDS, start=1):
                if delay > 0:
                    await asyncio.sleep(delay)
                if self._closing:
                    return False

                try:
                    async with self._provider_io_lock:
                        provider = self._avatarkit_session
                        if provider is not None:
                            generation = self._provider_session_generation
                            await self._close_provider_session(
                                provider,
                                generation=generation,
                            )
                            if self._avatarkit_session is provider:
                                self._avatarkit_session = None

                        # Always create a fresh AvatarKit session. Reusing the
                        # closed object also reuses its private request state,
                        # which can correlate a new WebSocket with the failed
                        # request from the old connection.
                        await self._connect_fresh_provider_session()

                    self._provider_connection_state = "connected"
                    self._provider_recovery_pending_playback = True
                    logger.info(
                        "SpatialReal provider websocket reconnected; awaiting observed avatar playback",
                        extra={
                            "attempt": attempt,
                            "generation": self._provider_session_generation,
                            "reason": reason,
                        },
                    )
                    self._emit_provider_connection_state(
                        state="connected",
                        reason="reconnected_awaiting_playback",
                        attempt=attempt,
                    )
                    return True
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_error = error
                    logger.warning(
                        "SpatialReal provider reconnect attempt failed",
                        exc_info=error,
                        extra={"attempt": attempt, "reason": reason},
                    )

            self._provider_terminal_failure = True
            self._provider_connection_state = "failed"
            failure_reason = (
                f"{reason}:{self._format_error_reason(last_error)}"
                if last_error is not None
                else reason
            )
            logger.error(
                "SpatialReal provider recovery exhausted; full room restart required",
                extra={"reason": failure_reason},
            )
            self._emit_provider_connection_state(
                state="failed",
                reason=failure_reason,
                attempt=len(PROVIDER_RECONNECT_DELAYS_SECONDS),
            )
            return False

    def _emit_provider_connection_state(
        self,
        *,
        state: Literal["connected", "recovering", "recovered", "failed"],
        reason: str,
        attempt: int,
    ) -> None:
        self.emit(
            "provider_connection_state_changed",
            AvatarProviderConnectionEvent(
                state=state,
                reason=reason,
                attempt=attempt,
                generation=self._provider_session_generation,
                observed_at=time.time(),
            ),
        )

    def _endpoint_options(self) -> dict[str, str]:
        """Only what the caller actually set; the SDK resolves the rest from the environment."""
        options: dict[str, str] = {}
        if self._environment:
            options["environment"] = self._environment
        if self._console_endpoint_url:
            options["console_endpoint_url"] = self._console_endpoint_url
        if self._ingress_endpoint_url:
            options["ingress_endpoint_url"] = self._ingress_endpoint_url
        return options

    async def disconnect_provider_for_test(self) -> None:
        """Close the ingress WebSocket so staging can verify provider recovery."""
        if self._closing or self._avatarkit_session is None:
            raise SpatialRealException("SpatialReal provider session is unavailable")
        await self._avatarkit_session.close()

    async def _send_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if self._provider_terminal_failure:
            raise SpatialRealException("SpatialReal provider recovery is exhausted")
        if not self._avatarkit_session:
            return

        buffered_frame = _BufferedAudioFrame(
            audio=bytes(frame.data),
            duration=frame.duration,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
        )

        clear_generation = self._clear_generation
        async with self._provider_io_lock:
            if clear_generation != self._clear_generation:
                return
            if self._discard_requested and not self._segments and self._paused_segment is None:
                self._discard_requested = False

            if self._pause_requested:
                segment = self._paused_segment
                if segment is None:
                    segment = self._new_local_segment()
                    self._paused_segment = segment
                self._append_audio_frame(segment, buffered_frame, sent_to_provider=False)
                return

            previous_req_id = self._active_req_id
            req_id = await self._send_provider_audio(
                audio=buffered_frame.audio,
                end=False,
                clear_generation=clear_generation,
            )
            provider_events_trusted = not self._reactivate_request_id(req_id)
            if previous_req_id and previous_req_id != req_id:
                logger.warning(
                    "Avatar request ID changed while streaming audio",
                    extra={"previous": previous_req_id, "current": req_id},
                )
                previous_segment = self._segments.get(previous_req_id)
                if previous_segment is not None:
                    self._mark_segment_waiting_for_completion(previous_segment)

            segment = self._segments.get(req_id)
            if segment is None:
                segment = _SegmentState(
                    req_id=req_id,
                    provider_events_trusted=provider_events_trusted,
                )
                self._segments[req_id] = segment
            else:
                segment.provider_events_trusted = (
                    segment.provider_events_trusted and provider_events_trusted
                )

            if not segment.provider_events_trusted:
                logger.warning(
                    "SpatialReal reused a request ID; provider lifecycle events are disabled for active attempt",
                    extra={"request_id": req_id},
                )

            self._consume_early_provider_events(segment)

            if segment.first_frame_at is None:
                segment.first_frame_at = time.time()
                logger.debug("SpatialReal avatar first audio frame", extra={"request_id": req_id})

            self._append_audio_frame(segment, buffered_frame, sent_to_provider=True)
            self._active_req_id = req_id
            self._schedule_active_segment_idle_end()

    async def _send_provider_audio(
        self,
        *,
        audio: bytes,
        end: bool,
        clear_generation: int,
    ) -> str:
        """Send provider audio only while its LiveKit clear generation is current."""
        if not self._avatarkit_session:
            raise RuntimeError("SpatialReal provider session is unavailable")
        if clear_generation != self._clear_generation:
            raise _ProviderClearDuringSend("LiveKit cleared playback before provider send")

        self._provider_send_in_flight = True
        self._clear_during_provider_send = False
        try:
            req_id = await self._avatarkit_session.send_audio(audio=audio, end=end)
        finally:
            self._provider_send_in_flight = False

        if self._clear_during_provider_send or clear_generation != self._clear_generation:
            raise _ProviderClearDuringSend("LiveKit cleared playback during provider send")
        return req_id

    def _append_audio_frame(
        self,
        segment: _SegmentState,
        frame: _BufferedAudioFrame,
        *,
        sent_to_provider: bool,
    ) -> None:
        segment.audio_frames.append(frame)
        segment.buffered_duration += frame.duration
        segment.pushed_duration += frame.duration
        if sent_to_provider:
            segment.attempt_duration += frame.duration

        while (
            segment.buffered_duration > self._resume_buffer_max_seconds
            and len(segment.audio_frames) > 1
        ):
            dropped = segment.audio_frames.popleft()
            segment.buffered_duration -= dropped.duration
            segment.buffer_start_position += dropped.duration

    @staticmethod
    def _remaining_audio_frames(
        segment: _SegmentState,
        playback_position: float,
    ) -> list[_BufferedAudioFrame]:
        remaining: list[_BufferedAudioFrame] = []
        skip = max(0.0, playback_position - segment.buffer_start_position)

        for frame in segment.audio_frames:
            if skip >= frame.duration:
                skip -= frame.duration
                continue

            if skip <= 0.0:
                remaining.append(frame)
                continue

            bytes_per_sample = max(1, frame.num_channels) * 2
            samples_to_skip = min(
                int(round(skip * frame.sample_rate)),
                len(frame.audio) // bytes_per_sample,
            )
            byte_offset = samples_to_skip * bytes_per_sample
            audio = frame.audio[byte_offset:]
            duration = len(audio) / (bytes_per_sample * frame.sample_rate)
            if audio and duration > 0.0:
                remaining.append(
                    _BufferedAudioFrame(
                        audio=audio,
                        duration=duration,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    )
                )
            skip = 0.0

        return remaining

    def _consume_early_provider_events(self, segment: _SegmentState) -> None:
        if not segment.provider_events_trusted:
            self._early_provider_started_ids.discard(segment.req_id)
            self._early_provider_completed_ids.discard(segment.req_id)
            return
        if segment.req_id in self._early_provider_started_ids:
            self._early_provider_started_ids.discard(segment.req_id)
            self._mark_playback_started(segment, source="spatialreal_transport_frame")
        if segment.req_id in self._early_provider_completed_ids:
            self._early_provider_completed_ids.discard(segment.req_id)
            segment.provider_playback_completed = True

    def _cancel_active_segment_idle_end(self) -> None:
        if self._active_segment_idle_end_task and not self._active_segment_idle_end_task.done():
            self._active_segment_idle_end_task.cancel()
        self._active_segment_idle_end_task = None

    def _schedule_active_segment_idle_end(self) -> None:
        active_req_id = self._active_req_id
        if active_req_id is None:
            return

        self._cancel_active_segment_idle_end()
        self._active_segment_idle_end_task = asyncio.create_task(
            self._wait_for_active_segment_idle_end(active_req_id, ACTIVE_SEGMENT_IDLE_END_SECONDS),
            name=f"spatialreal_idle_segment_end_{active_req_id}",
        )

    async def _wait_for_active_segment_idle_end(self, req_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        if self._active_req_id != req_id:
            return
        if req_id in self._pending_segment_ids:
            return
        if req_id not in self._segments:
            return
        try:
            finalized = await self._finalize_active_segment(source="idle_timeout")
        except Exception as error:
            await self._recover_provider_forwarding_error(
                item=AudioSegmentEnd(),
                error=error,
            )
            return

        if finalized:
            logger.warning(
                "Avatar segment end marker missing; forcing finalization",
                extra={"request_id": req_id, "idle_timeout": timeout},
            )

    async def _finalize_active_segment(self, *, source: str) -> bool:
        if not self._avatarkit_session:
            return False

        clear_generation = self._clear_generation
        async with self._provider_io_lock:
            if clear_generation != self._clear_generation:
                return False
            if self._pause_requested:
                if self._paused_segment is None:
                    self._paused_segment = self._new_local_segment()
                self._paused_segment.input_finalized = True
                logger.debug(
                    "SpatialReal avatar input finalized while playback paused",
                    extra={"source": source},
                )
                return True

            if self._active_req_id is None:
                return False

            async with self._segment_finalize_lock:
                active_req_id = self._active_req_id
                if active_req_id is None:
                    return False

                self._cancel_active_segment_idle_end()
                req_id = await self._send_provider_audio(
                    audio=b"",
                    end=True,
                    clear_generation=clear_generation,
                )
                finalized_events_trusted = not self._reactivate_request_id(req_id)
                if req_id != active_req_id:
                    logger.warning(
                        "Avatar request ID changed while finalizing segment",
                        extra={"expected": active_req_id, "actual": req_id, "source": source},
                    )

                self._active_req_id = None
                active_segment = self._segments.pop(active_req_id, None)
                segment = self._segments.get(req_id)

                if active_segment is None and segment is None:
                    return True
                if segment is None:
                    if active_segment is None:
                        return True
                    active_segment.req_id = req_id
                    segment = active_segment
                    self._segments[req_id] = segment
                elif active_segment is not None and segment is not active_segment:
                    self._merge_segment_state(segment, active_segment)

                segment.input_finalized = True
                segment.provider_events_trusted = (
                    segment.provider_events_trusted and finalized_events_trusted
                )
                self._consume_early_provider_events(segment)
                self._mark_segment_waiting_for_completion(segment)
                return True

    def _mark_segment_waiting_for_completion(self, segment: _SegmentState) -> None:
        if segment.req_id not in self._pending_segment_ids:
            self._pending_segment_ids.append(segment.req_id)

        # a provider completion that arrived before local finalization is
        # preserved (not dropped): apply it now
        if segment.provider_playback_completed:
            self._complete_segment(
                req_id=segment.req_id,
                interrupted=False,
                reason="provider_end_deferred",
            )
            return

        self._schedule_segment_completion(segment)

    def _schedule_segment_completion(self, segment: _SegmentState) -> None:
        if segment.completion_timeout_task and not segment.completion_timeout_task.done():
            segment.completion_timeout_task.cancel()

        timeout = self._compute_completion_timeout(segment)
        segment.completion_timeout_task = asyncio.create_task(
            self._wait_for_segment_completion_timeout(segment.req_id, timeout),
            name=f"spatialreal_segment_timeout_{segment.req_id}",
        )

    @staticmethod
    def _compute_completion_timeout(segment: _SegmentState) -> float:
        if segment.playback_started_at is not None:
            expected_playback_end = segment.playback_started_at + segment.attempt_duration
            remaining_playback = max(0.0, expected_playback_end - time.time())
            return max(
                MIN_COMPLETION_TIMEOUT_SECONDS,
                remaining_playback + LIVEKIT_ACTIVITY_COMPLETION_BUFFER_SECONDS,
            )

        if segment.first_frame_at is None:
            return MIN_COMPLETION_TIMEOUT_SECONDS

        expected_playback_end = segment.first_frame_at + segment.attempt_duration
        remaining_playback = max(0.0, expected_playback_end - time.time())
        return max(
            MIN_COMPLETION_TIMEOUT_SECONDS,
            remaining_playback + LIVEKIT_ACTIVITY_START_WAIT_SECONDS,
        )

    async def _wait_for_segment_completion_timeout(self, req_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        segment = self._segments.get(req_id)
        if segment is None:
            return

        # the server is holding playback; resume reschedules against the shifted start
        if segment.native_paused_at is not None:
            return

        # if the avatar is still audibly speaking, allow a bounded overrun
        # before declaring the segment finished
        if segment.playback_started_at is not None and self._avatar_is_speaking:
            expected_end = segment.playback_started_at + segment.attempt_duration
            if time.time() < expected_end + LIVEKIT_ACTIVITY_MAX_OVERRUN_SECONDS:
                segment.completion_timeout_task = asyncio.create_task(
                    self._wait_for_segment_completion_timeout(
                        req_id, LIVEKIT_ACTIVITY_RECHECK_SECONDS
                    ),
                    name=f"spatialreal_segment_activity_wait_{req_id}",
                )
                return

        playback_observed = segment.playback_started_at is not None
        reason = "livekit_activity_duration" if playback_observed else "timeout_no_playback_signal"
        if self._complete_segment(req_id=req_id, interrupted=False, reason=reason):
            if playback_observed:
                logger.debug(
                    "SpatialReal avatar completion derived from observed LiveKit playback",
                    extra={
                        "request_id": req_id,
                        "playback_start_source": segment.playback_start_source,
                        "pushed_duration": segment.pushed_duration,
                    },
                )
            else:
                logger.warning(
                    "Avatar segment received no provider or LiveKit playback signal; using duration fallback",
                    extra={"request_id": req_id, "timeout": timeout},
                )

    def _on_playback_state(self, event: PlaybackStateEvent) -> None:
        # Structured server-side playback state (egress playback_control). The
        # PLAYING/PAUSED/ENDED signals are already handled via on_playback
        # (ServerResponseAnimation); the one thing that path can't express is a
        # server-forced interrupt of a held pause — surface that here so the
        # framework's speech handle isn't left hanging on a segment the server
        # has abandoned.
        if event.state != PlaybackState.INTERRUPTED:
            return
        if event.reason != InterruptReason.PAUSE_TIMEOUT:
            # explicit / preempted interrupts already flow through the framework's
            # own clear_buffer path; nothing extra to do.
            return
        logger.warning(
            "SpatialReal avatar pause timed out on the server; finalizing segment as interrupted",
            extra={"request_id": event.req_id, "played_ms": event.played_ms},
        )
        self._discard_requested = True
        self._pause_requested = False
        self._spawn_background_task(
            self._handle_interrupt(),
            name="spatialreal_avatar_pause_timeout",
        )

    def _on_playback_signal(self, signal: PlaybackSignal) -> None:
        req_id = signal.req_id or None
        is_last = signal.end
        if req_id is not None and req_id in self._recently_completed_req_ids:
            logger.debug(
                "Ignoring duplicate provider event for completed request",
                extra={"request_id": req_id, "is_last": is_last},
            )
            return
        if req_id is not None:
            segment = self._segments.get(req_id)
            if segment is not None and not segment.provider_events_trusted:
                logger.debug(
                    "Ignoring ambiguous provider event for a reused request ID",
                    extra={"request_id": req_id, "is_last": is_last},
                )
                return
            if segment is None:
                self._early_provider_started_ids.add(req_id)
            else:
                self._mark_playback_started(segment, source="spatialreal_transport_frame")

            if not is_last:
                return

            if segment is None:
                self._early_provider_completed_ids.add(req_id)
                logger.debug(
                    "Preserving provider completion received before local segment creation",
                    extra={"request_id": req_id},
                )
                return

            segment.provider_playback_completed = True
            if req_id not in self._pending_segment_ids:
                logger.debug(
                    "Deferring provider completion until local segment finalization",
                    extra={"request_id": req_id},
                )
                return

            if not self._complete_segment(req_id=req_id, interrupted=False, reason="provider_end"):
                logger.debug("Completion event for unknown request", extra={"request_id": req_id})
            return

        if not is_last:
            return

        if self._pending_segment_ids:
            fallback_req_id = self._pending_segment_ids[0]
            if self._complete_segment(
                req_id=fallback_req_id,
                interrupted=False,
                reason="provider_end_fallback",
            ):
                logger.warning(
                    "Avatar completion event missing request ID; matched oldest pending segment",
                    extra={"request_id": fallback_req_id},
                )
            return

        if self._active_req_id is not None:
            active_segment = self._segments.get(self._active_req_id)
            if active_segment is not None:
                self._mark_playback_started(active_segment, source="spatialreal_transport_frame")
                active_segment.provider_playback_completed = True
                logger.warning(
                    "Avatar completion event missing request ID; deferred against active segment",
                    extra={"request_id": self._active_req_id},
                )

    def _is_avatar_output_participant(self, participant: rtc.Participant) -> bool:
        if participant.identity == self._avatar_participant_identity:
            return True

        represented_identity = self._represented_participant_identity
        if represented_identity is None:
            return False

        attributes = getattr(participant, "attributes", {}) or {}
        return attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF) == represented_identity

    def _on_active_speakers_changed(self, speakers: list[rtc.Participant]) -> None:
        avatar_is_speaking = any(
            self._is_avatar_output_participant(speaker) for speaker in speakers
        )
        if avatar_is_speaking == self._avatar_is_speaking:
            return

        self._avatar_is_speaking = avatar_is_speaking
        logger.debug(
            "SpatialReal avatar LiveKit speaking state changed",
            extra={
                "is_speaking": avatar_is_speaking,
                "active_speaker_identities": [speaker.identity for speaker in speakers],
            },
        )
        if not avatar_is_speaking or self._playback_observation_suspended():
            return

        segment = self._current_observable_segment()
        if segment is not None:
            self._mark_playback_started(segment, source="livekit_active_speaker")

    def _playback_observation_suspended(self) -> bool:
        # SpatialReal 1.7.1 correction: an egress tail that is draining while
        # paused is not the start of the resumed attempt.
        return self._pause_requested or self._paused_segment is not None

    def _on_track_published(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        # rooms configured without auto-subscribe never fire track_subscribed
        # unless we opt in explicitly
        if publication.kind == rtc.TrackKind.KIND_AUDIO and self._is_avatar_output_participant(
            participant
        ):
            publication.set_subscribed(True)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or not self._is_avatar_output_participant(
            participant
        ):
            return

        self._start_avatar_audio_monitor(track)
        logger.debug(
            "SpatialReal avatar LiveKit audio track subscribed",
            extra={
                "participant_identity": participant.identity,
                "track_sid": publication.sid,
            },
        )

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        _: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO and self._is_avatar_output_participant(
            participant
        ):
            self._stop_avatar_audio_monitor()

    def _start_avatar_audio_monitor(self, track: rtc.Track) -> None:
        if self._avatar_audio_monitor_task and not self._avatar_audio_monitor_task.done():
            return

        self._avatar_audio_stream = rtc.AudioStream(track, capacity=10)
        self._avatar_audio_monitor_task = asyncio.create_task(
            self._monitor_avatar_audio(),
            name="spatialreal_avatar_audio_monitor",
        )

    def _stop_avatar_audio_monitor(self) -> None:
        if self._avatar_audio_monitor_task and not self._avatar_audio_monitor_task.done():
            self._avatar_audio_monitor_task.cancel()
        self._avatar_audio_monitor_task = None

        if self._avatar_audio_stream is not None:
            self._spawn_background_task(
                self._avatar_audio_stream.aclose(),
                name="spatialreal_avatar_audio_stream_close",
            )
        self._avatar_audio_stream = None

    async def _monitor_avatar_audio(self) -> None:
        stream = self._avatar_audio_stream
        if stream is None:
            return

        try:
            async for event in stream:
                self._on_avatar_audio_frame(event.frame)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning("SpatialReal avatar audio monitor failed", exc_info=True)

    def _current_observable_segment(self) -> _SegmentState | None:
        if self._paused_segment is not None:
            return self._paused_segment
        if self._pending_segment_ids:
            segment = self._segments.get(self._pending_segment_ids[0])
            if segment is not None:
                return segment
        if self._active_req_id is not None:
            return self._segments.get(self._active_req_id)
        return None

    def _on_avatar_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if self._playback_observation_suspended():
            return
        segment = self._current_observable_segment()
        if segment is None or segment.playback_started:
            return

        if not self._frame_has_audible_audio(frame):
            return

        self._mark_playback_started(segment, source="livekit_avatar_audio_track")

    @staticmethod
    def _frame_has_audible_audio(frame: rtc.AudioFrame) -> bool:
        threshold = AVATAR_AUDIO_ACTIVITY_THRESHOLD
        for sample in frame.data:
            if sample >= threshold or sample <= -threshold:
                return True
        return False

    def _mark_playback_started(self, segment: _SegmentState, *, source: str) -> None:
        if segment.playback_started:
            return

        segment.playback_started = True
        segment.playback_started_at = time.time()
        segment.playback_start_source = source
        if self._audio_buffer:
            self._audio_buffer.notify_playback_started()
        if segment.req_id in self._pending_segment_ids:
            self._schedule_segment_completion(segment)
        logger.debug(
            "SpatialReal avatar playback started",
            extra={"request_id": segment.req_id, "source": source},
        )
        self.emit(
            "playback_started",
            AvatarPlaybackStartedEvent(
                request_id=segment.req_id,
                source=source,
                observed_at=segment.playback_started_at,
            ),
        )
        if self._provider_recovery_pending_playback:
            self._provider_recovery_pending_playback = False
            self._provider_connection_state = "recovered"
            logger.info(
                "SpatialReal provider recovery confirmed by observed avatar playback",
                extra={
                    "request_id": segment.req_id,
                    "source": source,
                    "generation": self._provider_session_generation,
                },
            )
            self._emit_provider_connection_state(
                state="recovered",
                reason=f"playback_observed:{source}",
                attempt=0,
            )

    def _complete_segment(self, *, req_id: str, interrupted: bool, reason: str) -> bool:
        segment = self._segments.pop(req_id, None)
        if segment is None:
            return False

        self._early_provider_started_ids.discard(req_id)
        self._early_provider_completed_ids.discard(req_id)
        self._recently_completed_req_ids.append(req_id)

        self._pending_segment_ids = deque(
            pending_req_id
            for pending_req_id in self._pending_segment_ids
            if pending_req_id != req_id
        )

        if segment.completion_timeout_task and not segment.completion_timeout_task.done():
            segment.completion_timeout_task.cancel()

        if self._active_req_id == req_id:
            self._active_req_id = None
            self._cancel_active_segment_idle_end()

        playback_position = (
            self._estimate_interrupted_playback_position(segment)
            if interrupted
            else segment.pushed_duration
        )

        if self._audio_buffer:
            # with wait_playback_start the framework only sees playback_started
            # when we notify it; guarantee started precedes finished even if no
            # playback signal was ever observed
            if not segment.playback_started:
                self._audio_buffer.notify_playback_started()
            self._audio_buffer.notify_playback_finished(
                playback_position=playback_position,
                interrupted=interrupted,
            )

        logger.debug(
            "SpatialReal avatar segment playback completed",
            extra={
                "request_id": req_id,
                "reason": reason,
                "interrupted": interrupted,
                "playback_position": playback_position,
                "pushed_duration": segment.pushed_duration,
                "playback_start_source": segment.playback_start_source,
                "provider_playback_completed": segment.provider_playback_completed,
            },
        )
        return True

    @staticmethod
    def _estimate_interrupted_playback_position(segment: _SegmentState) -> float:
        # Sending frames to SpatialReal does not mean the avatar has played
        # them. Before an observed playback-start signal, resume from the last
        # confirmed position so an early false interruption cannot skip audio.
        anchor = segment.playback_started_at
        if anchor is None:
            return min(segment.pushed_duration, segment.playback_position_offset)

        elapsed = max(0.0, time.time() - anchor)
        return min(segment.pushed_duration, segment.playback_position_offset + elapsed)

    def _complete_all_segments(self, *, interrupted: bool, reason: str) -> None:
        self._complete_paused_segment(interrupted=interrupted, reason=reason)
        for req_id in list(self._segments.keys()):
            self._complete_segment(req_id=req_id, interrupted=interrupted, reason=reason)

        self._active_req_id = None
        self._cancel_active_segment_idle_end()
        self._pending_segment_ids.clear()
        self._early_provider_started_ids.clear()
        self._early_provider_completed_ids.clear()
        self._avatar_is_speaking = False
        self._paused_segment = None
        self._pause_requested = False
        self._pause_requested_at = None
        self._discard_requested = False

    def _new_local_segment(self) -> _SegmentState:
        self._local_segment_sequence += 1
        return _SegmentState(req_id=f"local-pending-{self._local_segment_sequence}")

    @staticmethod
    def _merge_segment_state(target: _SegmentState, source: _SegmentState) -> None:
        """Preserve retained audio and lifecycle state across provider ID rollover."""
        target.pushed_duration = max(target.pushed_duration, source.pushed_duration)
        target.attempt_duration = max(target.attempt_duration, source.attempt_duration)
        target.playback_position_offset = max(
            target.playback_position_offset,
            source.playback_position_offset,
        )
        target.input_finalized = target.input_finalized or source.input_finalized
        target.provider_events_trusted = (
            target.provider_events_trusted and source.provider_events_trusted
        )

        if source.buffered_duration > target.buffered_duration:
            target.audio_frames = deque(source.audio_frames)
            target.buffered_duration = source.buffered_duration
            target.buffer_start_position = source.buffer_start_position

        if target.first_frame_at is None:
            target.first_frame_at = source.first_frame_at
        target.playback_started = target.playback_started or source.playback_started
        target.provider_playback_completed = (
            target.provider_playback_completed or source.provider_playback_completed
        )
        if target.playback_started_at is None:
            target.playback_started_at = source.playback_started_at
            target.playback_start_source = source.playback_start_source

    def _complete_paused_segment(self, *, interrupted: bool, reason: str) -> bool:
        segment = self._paused_segment
        if segment is None:
            return False

        self._paused_segment = None
        req_id = segment.req_id
        existing = self._segments.get(req_id)
        if existing is not None and existing is not segment:
            self._merge_segment_state(existing, segment)
            segment = existing
        else:
            self._segments[req_id] = segment
        return self._complete_segment(
            req_id=segment.req_id,
            interrupted=interrupted,
            reason=reason,
        )

    def _on_pause(self) -> None:
        if self._closing or self._drop_frames_until_segment_end:
            return
        self._discard_requested = False
        if self._server_playback_control:
            # Native path: the server holds its send cursor and keeps ingesting,
            # so we must NOT set _pause_requested (which diverts/retains frames)
            # — audio keeps flowing to the same request untouched.
            self._spawn_background_task(
                self._handle_pause_native(),
                name="spatialreal_avatar_pause_native",
            )
            return
        self._pause_requested = True
        self._pause_requested_at = time.time()
        self._spawn_background_task(
            self._handle_pause(),
            name="spatialreal_avatar_pause",
        )

    def _on_resume(self) -> None:
        if self._closing or self._drop_frames_until_segment_end:
            return
        self._pause_requested = False
        if self._discard_requested:
            logger.debug("Ignoring SpatialReal avatar resume after confirmed interruption")
            return
        if self._server_playback_control:
            self._spawn_background_task(
                self._handle_resume_native(),
                name="spatialreal_avatar_resume_native",
            )
            return
        self._spawn_background_task(
            self._handle_resume(),
            name="spatialreal_avatar_resume",
        )

    async def _handle_pause_native(self) -> None:
        """Server-side pause: hold the cursor, keep the segment and audio flow intact."""
        if not self._avatarkit_session or self._discard_requested:
            return
        async with self._provider_io_lock:
            if self._discard_requested or self._avatarkit_session is None:
                return
            try:
                req_id = await self._avatarkit_session.pause()
                logger.debug(
                    "SpatialReal avatar playback paused (server-side)", extra={"request_id": req_id}
                )
            except Exception as e:
                # Nothing is holding playback: fall back to the emulated path rather than
                # letting frames keep flowing through the framework's interruption hold.
                logger.warning(
                    "Failed to pause SpatialReal avatar playback (server-side); "
                    "falling back to local pause",
                    exc_info=e,
                )
                self._pause_requested = True
                self._pause_requested_at = time.time()
                self._spawn_background_task(
                    self._handle_pause(),
                    name="spatialreal_avatar_pause_fallback",
                )
                return
            segment = self._segments.get(req_id) or self._segments.get(self._active_req_id or "")
            if segment is not None and segment.native_paused_at is None:
                segment.native_paused_at = time.time()

    async def _handle_resume_native(self) -> None:
        """Server-side resume: continue the held cursor from where it stopped."""
        try:
            if not self._avatarkit_session or self._discard_requested:
                return
            async with self._provider_io_lock:
                if self._discard_requested or self._avatarkit_session is None:
                    return
                try:
                    req_id = await self._avatarkit_session.resume()
                    logger.debug(
                        "SpatialReal avatar playback resumed (server-side)",
                        extra={"request_id": req_id},
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to resume SpatialReal avatar playback (server-side)", exc_info=e
                    )
        finally:
            # released on every path (failed resume, resume during a provider
            # reconnect) so a segment can't be stranded; a hold the server keeps
            # anyway ends in its pause_timeout interrupt
            self._release_native_hold()

    def _release_native_hold(self) -> None:
        """Shift held segments' playback start by the time the server held them."""
        now = time.time()
        for segment in self._segments.values():
            if segment.native_paused_at is None:
                continue
            held = now - segment.native_paused_at
            segment.native_paused_at = None
            if segment.playback_started_at is not None:
                segment.playback_started_at += held
            elif segment.first_frame_at is not None:
                segment.first_frame_at += held
            if segment.req_id in self._pending_segment_ids:
                self._schedule_segment_completion(segment)

    def _on_clear_buffer(self) -> None:
        if self._closing:
            return
        self._clear_generation += 1
        if self._provider_send_in_flight:
            # The provider call still owns the I/O lock. Defer the confirmed
            # interruption until the send returns; if it fails, recovery uses
            # this same clear as the authoritative sentence boundary.
            self._clear_during_provider_send = True
            self._discard_requested = True
            self._pause_requested = False
            return
        if self._provider_recovery_in_flight:
            # Recovery already owns the provider interrupt. A concurrent clear
            # is the authoritative framework boundary even if the provider
            # round trip has not completed yet.
            self._clear_during_provider_recovery = True
            self._discard_requested = True
            self._pause_requested = False
            return
        if self._drop_frames_until_segment_end:
            # LiveKit flushes before clear_buffer on a confirmed
            # interruption. The clear can drain the queued AudioSegmentEnd,
            # so it is itself the authoritative boundary for the damaged
            # segment and must release drop mode for the next response.
            self._drop_frames_until_segment_end = False
            return
        self._discard_requested = True
        self._pause_requested = False
        self._spawn_background_task(self._handle_interrupt(), name="spatialreal_avatar_interrupt")

    async def _handle_pause(self) -> None:
        if not self._avatarkit_session or self._discard_requested:
            return

        async with self._provider_io_lock:
            if self._discard_requested or not self._pause_requested:
                return

            segment = self._current_observable_segment()
            if segment is None:
                logger.debug("SpatialReal avatar paused before provider audio started")
                return

            old_req_id = segment.req_id
            was_pending = old_req_id in self._pending_segment_ids
            try:
                interrupted_id = await self._avatarkit_session.interrupt()
            except Exception as e:
                # Keep the original provider request authoritative. The
                # framework may still emit resume/clear, but new frames must
                # not be diverted into an orphaned paused segment.
                self._pause_requested = False
                self._pause_requested_at = None
                logger.warning("Failed to pause SpatialReal avatar playback", exc_info=e)
                return

            segment.playback_position_offset = self._estimate_interrupted_playback_position(segment)
            segment.input_finalized = segment.input_finalized or was_pending
            if segment.completion_timeout_task and not segment.completion_timeout_task.done():
                segment.completion_timeout_task.cancel()
            self._pending_segment_ids = deque(
                req_id for req_id in self._pending_segment_ids if req_id != old_req_id
            )
            if self._active_req_id == old_req_id:
                self._active_req_id = None
            self._cancel_active_segment_idle_end()
            self._recently_completed_req_ids.append(old_req_id)
            if interrupted_id != old_req_id:
                self._recently_completed_req_ids.append(interrupted_id)

            segment.playback_started = False
            segment.playback_started_at = None
            segment.playback_start_source = None
            segment.provider_playback_completed = False
            segment.first_frame_at = None
            segment.attempt_duration = 0.0
            self._paused_segment = segment
            provider_pause_latency_ms = None
            if self._pause_requested_at is not None:
                provider_pause_latency_ms = max(
                    0.0,
                    (time.time() - self._pause_requested_at) * 1000.0,
                )

            logger.debug(
                "SpatialReal avatar playback paused",
                extra={
                    "request_id": old_req_id,
                    "provider_interrupt_id": interrupted_id,
                    "playback_position": segment.playback_position_offset,
                    "pushed_duration": segment.pushed_duration,
                    "input_finalized": segment.input_finalized,
                    "provider_pause_latency_ms": provider_pause_latency_ms,
                },
            )

    async def _handle_resume(self) -> None:
        if not self._avatarkit_session or self._discard_requested:
            return

        clear_generation = self._clear_generation
        async with self._provider_io_lock:
            segment = self._paused_segment
            if (
                segment is None
                or self._discard_requested
                or self._pause_requested
                or clear_generation != self._clear_generation
            ):
                return

            self._pause_requested_at = None

            old_req_id = segment.req_id
            replay_from = max(
                segment.playback_position_offset,
                segment.buffer_start_position,
            )
            if replay_from > segment.playback_position_offset:
                logger.warning(
                    "SpatialReal resume buffer no longer contains requested playback position",
                    extra={
                        "request_id": old_req_id,
                        "requested_position": segment.playback_position_offset,
                        "buffer_start_position": segment.buffer_start_position,
                    },
                )
                segment.playback_position_offset = replay_from

            remaining_frames = self._remaining_audio_frames(segment, replay_from)
            req_id: str | None = None
            attempt_duration = 0.0
            provider_events_trusted = True
            resumed_request_ids: set[str] = set()
            replay_transmission_started = False
            try:
                for frame in remaining_frames:
                    replay_transmission_started = True
                    current_req_id = await self._send_provider_audio(
                        audio=frame.audio,
                        end=False,
                        clear_generation=clear_generation,
                    )
                    provider_events_trusted = provider_events_trusted and (
                        current_req_id not in self._recently_completed_req_ids
                    )
                    resumed_request_ids.add(current_req_id)
                    if req_id is not None and current_req_id != req_id:
                        logger.warning(
                            "Avatar request ID changed while resuming audio",
                            extra={"previous": req_id, "current": current_req_id},
                        )
                    req_id = current_req_id
                    attempt_duration += frame.duration

                finalized_req_id: str | None = None
                if req_id is not None and segment.input_finalized:
                    finalized_req_id = await self._send_provider_audio(
                        audio=b"",
                        end=True,
                        clear_generation=clear_generation,
                    )
                    provider_events_trusted = provider_events_trusted and (
                        finalized_req_id not in self._recently_completed_req_ids
                    )
                    resumed_request_ids.add(finalized_req_id)
            except Exception as e:
                logger.warning("Failed to resume SpatialReal avatar playback", exc_info=e)
                if replay_transmission_started:
                    try:
                        await self._avatarkit_session.interrupt()
                    except Exception:
                        logger.warning(
                            "Failed to stop partial SpatialReal resume after provider error",
                            exc_info=True,
                        )
                self._pause_requested = False
                self._pause_requested_at = None
                self._complete_paused_segment(
                    interrupted=True,
                    reason="resume_provider_error",
                )
                return

            if req_id is None:
                if segment.input_finalized:
                    self._complete_paused_segment(
                        interrupted=False,
                        reason="resume_after_playback_complete",
                    )
                else:
                    # Everything buffered had already played, so there was nothing to
                    # resend — but the turn is still open. Put the segment back in play so
                    # the next frame continues it (or rolls it over) instead of leaving it
                    # stranded beside a fresh request.
                    self._segments[segment.req_id] = segment
                    self._active_req_id = segment.req_id
                    self._paused_segment = None
                return

            # The provider can reuse request IDs. A delayed completion from the
            # interrupted attempt is then indistinguishable from the resumed
            # attempt, so rely on observed avatar playback plus duration for
            # this attempt instead of trusting ambiguous provider frames.
            for resumed_request_id in resumed_request_ids:
                self._reactivate_request_id(resumed_request_id)

            self._segments.pop(old_req_id, None)
            segment.req_id = finalized_req_id or req_id
            segment.first_frame_at = time.time()
            segment.attempt_duration = attempt_duration
            segment.provider_events_trusted = provider_events_trusted
            if not provider_events_trusted:
                logger.warning(
                    "SpatialReal reused a request ID; provider lifecycle events are disabled for resumed attempt",
                    extra={"request_id": segment.req_id, "previous_request_id": old_req_id},
                )
            existing = self._segments.get(segment.req_id)
            if existing is not None and existing is not segment:
                self._merge_segment_state(segment, existing)
            self._segments[segment.req_id] = segment
            self._paused_segment = None
            self._consume_early_provider_events(segment)

            if segment.input_finalized:
                self._mark_segment_waiting_for_completion(segment)
            else:
                self._active_req_id = segment.req_id
                self._schedule_active_segment_idle_end()

            logger.debug(
                "SpatialReal avatar playback resumed",
                extra={
                    "request_id": segment.req_id,
                    "previous_request_id": old_req_id,
                    "replay_from": replay_from,
                    "remaining_duration": segment.attempt_duration,
                    "pushed_duration": segment.pushed_duration,
                    "input_finalized": segment.input_finalized,
                },
            )

    async def _handle_interrupt(self) -> None:
        if not self._avatarkit_session:
            return

        async with self._interrupt_lock:
            async with self._provider_io_lock:
                # Coalesce duplicate clear callbacks once the interrupted turn
                # has already been finalized.
                if (
                    not self._segments
                    and self._active_req_id is None
                    and self._paused_segment is None
                ):
                    logger.debug("Ignoring duplicate SpatialReal avatar interrupt")
                    return

                paused_segment = self._paused_segment
                interrupted_id = (
                    paused_segment.req_id if paused_segment is not None else self._active_req_id
                )
                if paused_segment is None:
                    try:
                        interrupted_id = await self._avatarkit_session.interrupt()
                    except Exception as e:
                        # Provider failure must not strand the framework speech
                        # handle. Finalize local lifecycle state below and let
                        # diagnostics expose the provider-control failure.
                        logger.warning("Failed to interrupt SpatialReal avatar", exc_info=e)

                async with self._segment_finalize_lock:
                    if paused_segment is not None:
                        self._complete_paused_segment(
                            interrupted=True,
                            reason="interrupt_paused",
                        )
                    elif interrupted_id is not None:
                        if (
                            not self._complete_segment(
                                req_id=interrupted_id,
                                interrupted=True,
                                reason="interrupt",
                            )
                            and self._active_req_id is not None
                        ):
                            self._complete_segment(
                                req_id=self._active_req_id,
                                interrupted=True,
                                reason="interrupt_fallback",
                            )

                    for req_id in list(self._segments.keys()):
                        self._complete_segment(
                            req_id=req_id,
                            interrupted=True,
                            reason="interrupt_remaining",
                        )

                self._paused_segment = None
                self._pause_requested_at = None
                self._discard_requested = False
                logger.debug("SpatialReal avatar interrupted", extra={"request_id": interrupted_id})

    def _reactivate_request_id(self, req_id: str) -> bool:
        """Remove a completion tombstone and report provider request-ID reuse."""
        if req_id not in self._recently_completed_req_ids:
            return False
        self._recently_completed_req_ids = deque(
            [
                completed_id
                for completed_id in self._recently_completed_req_ids
                if completed_id != req_id
            ],
            maxlen=COMPLETED_REQ_ID_HISTORY,
        )
        return True

    def _detach_audio_output(self) -> None:
        if not self._agent_session or not self._audio_output_attached or self._audio_buffer is None:
            return

        output = self._agent_session.output
        if output.audio is self._audio_buffer:
            # replace_audio_tail fell back to a whole-chain assignment (no
            # wrappers were present), so restore the whole chain
            output.audio = self._original_audio_output
        elif self._resolve_audio_tail(output.audio) is self._audio_buffer:
            # we were swapped in as the tail under wrapper(s); swap the
            # original tail back so the closed buffer doesn't linger in the
            # chain
            if self._original_audio_tail is not None:
                output.replace_audio_tail(self._original_audio_tail)

        self._audio_output_attached = False

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closing:
                return
            self._closing = True

            if self._room is not None:
                self._room.off("active_speakers_changed", self._on_active_speakers_changed)
                self._room.off("track_published", self._on_track_published)
                self._room.off("track_subscribed", self._on_track_subscribed)
                self._room.off("track_unsubscribed", self._on_track_unsubscribed)
                self._room = None

            if self._avatar_audio_monitor_task and not self._avatar_audio_monitor_task.done():
                self._avatar_audio_monitor_task.cancel()
                await asyncio.gather(
                    self._avatar_audio_monitor_task,
                    return_exceptions=True,
                )
            self._avatar_audio_monitor_task = None

            avatar_audio_stream = self._avatar_audio_stream
            self._avatar_audio_stream = None
            if avatar_audio_stream is not None:
                try:
                    await avatar_audio_stream.aclose()
                except Exception:
                    logger.warning(
                        "Failed to close SpatialReal avatar audio stream",
                        exc_info=True,
                    )

            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
                self._main_task = None

            current_task = asyncio.current_task()
            lifecycle_tasks = [
                task
                for task in self._background_tasks
                if task is not current_task and not task.done()
            ]
            for task in lifecycle_tasks:
                task.cancel()
            if lifecycle_tasks:
                await asyncio.gather(*lifecycle_tasks, return_exceptions=True)

            self._cancel_active_segment_idle_end()
            self._complete_all_segments(interrupted=True, reason="session_close")
            self._drop_frames_until_segment_end = False
            self._provider_send_in_flight = False
            self._clear_during_provider_send = False
            self._clear_generation = 0
            self._provider_recovery_in_flight = False
            self._clear_during_provider_recovery = False
            self._provider_reconnect_task = None
            self._provider_connection_state = "disconnected"
            self._provider_recovery_pending_playback = False
            self._provider_terminal_failure = False
            self._provider_expected_close_generations.clear()

            self._detach_audio_output()
            self._original_audio_output = None
            self._original_audio_tail = None
            self._represented_participant_identity = None

            if self._audio_buffer:
                await self._audio_buffer.aclose()
                self._audio_buffer = None

            if self._avatarkit_session:
                try:
                    await self._avatarkit_session.close()
                    logger.debug("SpatialReal avatar session closed")
                except Exception as e:
                    logger.warning("Error closing SpatialReal avatar session", exc_info=e)
                finally:
                    self._avatarkit_session = None

            # The base class owns what it registered in start(): removing the avatar
            # participant from the room, unregistering its listeners and cancelling the
            # join task. It reads _agent_session and _room, so it has to run before the
            # fields below are cleared.
            try:
                await super().aclose()
            except Exception as e:
                logger.warning("Error in base avatar session close", exc_info=e)

            self._initialized = False
            self._agent_session = None
            self._livekit_egress = None
            self._resolved_sample_rate = None
