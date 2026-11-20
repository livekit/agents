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
import contextlib
import os
import time
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from livekit import api, rtc
from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, get_job_context, utils
from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarSession as BaseAvatarSession
from livekit.agents.voice.io import AudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from spatialreal import (
    AvatarSession as SpatialRealSession,
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
# Lifetime of the SpatialReal session. The server closes the session when it expires,
# so it has to outlive the call; the server caps it at 24h, less five minutes for clock skew.
DEFAULT_SESSION_TTL = timedelta(hours=24) - timedelta(minutes=5)
# TTL of the room-join token the SpatialReal egress uses. The egress rejoins the room
# across the whole session, so the token has to outlive the call too.
DEFAULT_LIVEKIT_TOKEN_TTL = timedelta(hours=24)
LIVEKIT_AVATAR_PUBLISH_SOURCES = ["camera", "microphone"]
# server capabilities: it can hold playback (pause/resume) while it keeps ingesting audio,
# and it reports each request's playback state with its played position
PLAYBACK_CONTROL_CAPABILITY = "playback_control"
PLAYBACK_STATE_CAPABILITY = "playback_state"

PROVIDER_RECONNECT_DELAYS_SECONDS = (0.0, 0.5, 1.0)
PROVIDER_CONNECT_TIMEOUT_SECONDS = 15.0
PROVIDER_CLOSE_TIMEOUT_SECONDS = 5.0

# The server reports when a reply has played out. If that report never comes, the reply
# is finished locally once it should have played, plus this margin.
COMPLETION_MARGIN_SECONDS = 3.0
# allowance for a reply that hasn't started playing yet (inference and egress latency)
START_ALLOWANCE_SECONDS = 5.0

# No endpoint lives here: the SpatialReal SDK resolves them from the deployment it is told to
# use, so a new region or a moved deployment never needs a new release of this plugin.


def _egress_video_grants(room_name: str) -> api.VideoGrants:
    """Grants of the room-join token the SpatialReal egress uses to publish the avatar.

    The egress also publishes data, so the token must allow it.
    """
    return api.VideoGrants(
        room_join=True,
        room=room_name,
        can_subscribe=False,
        can_publish_data=True,
        can_publish_sources=LIVEKIT_AVATAR_PUBLISH_SOURCES,
    )


class SpatialRealException(Exception):
    """Exception raised for SpatialReal avatar integration errors."""


@dataclass(frozen=True)
class AvatarPlaybackStartedEvent:
    """The avatar started playing one reply."""

    request_id: str
    source: str
    observed_at: float


@dataclass(frozen=True)
class AvatarProviderConnectionEvent:
    """State of the connection to SpatialReal (no conversation content)."""

    state: Literal["connected", "recovering", "recovered", "failed"]
    reason: str
    attempt: int
    generation: int
    observed_at: float


@dataclass(eq=False)
class _Reply:
    """One framework playout segment, sent to SpatialReal as one request."""

    req_id: str | None = None
    pushed: float = 0.0
    input_done: bool = False
    started_at: float | None = None
    # the last played position the server reported (start counts as 0), when, and
    # whether playback was held then; the position runs on from it while playing
    played_at_report: float = 0.0
    reported_at: float | None = None
    held: bool = False
    finished: bool = False
    completion_task: asyncio.Task[None] | None = None

    def report(self, played: float, now: float, *, held: bool) -> None:
        self.played_at_report, self.reported_at, self.held = played, now, held

    def played(self, now: float) -> float:
        if self.reported_at is None:
            return 0.0
        played = self.played_at_report + (0.0 if self.held else now - self.reported_at)
        return min(self.pushed, max(0.0, played))


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
        self._api_key = self._required(api_key, "api_key", "SPATIALREAL_API_KEY")
        self._app_id = self._required(app_id, "app_id", "SPATIALREAL_APP_ID")
        self._avatar_id = self._required(avatar_id, "avatar_id", "SPATIALREAL_AVATAR_ID")

        if idle_timeout_seconds < 0:
            raise SpatialRealException("idle_timeout_seconds must be greater than or equal to 0")
        if utils.is_given(sample_rate) and sample_rate <= 0:
            raise SpatialRealException("sample_rate must be greater than 0")

        token_ttl = (
            int(livekit_token_ttl_seconds)
            if utils.is_given(livekit_token_ttl_seconds)
            else self._int_env(
                "SPATIALREAL_LIVEKIT_TOKEN_TTL_SECONDS",
                int(DEFAULT_LIVEKIT_TOKEN_TTL.total_seconds()),
            )
        )
        if token_ttl <= 0:
            raise SpatialRealException("livekit_token_ttl_seconds must be greater than 0")
        self._livekit_token_ttl = timedelta(seconds=token_ttl)

        # The SDK owns where a deployment lives: it resolves the endpoints from `environment`
        # (its own default when unset). The two URLs are overrides for a deployment SpatialReal
        # hands you an address for; empty means "whatever the environment resolves to".
        self._environment = self._optional(environment, "SPATIALREAL_ENVIRONMENT")
        self._console_endpoint_url = self._optional(
            console_endpoint_url, "SPATIALREAL_CONSOLE_ENDPOINT"
        )
        self._ingress_endpoint_url = self._optional(
            ingress_endpoint_url, "SPATIALREAL_INGRESS_ENDPOINT"
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

        self._agent_session: AgentSession | None = None
        self._room: rtc.Room | None = None
        self._audio_buffer: ResumableQueueAudioOutput | None = None
        self._original_audio_output: AudioOutput | None = None
        self._original_audio_tail: AudioOutput | None = None
        self._livekit_egress: LiveKitEgressConfig | None = None
        self._resolved_sample_rate: int | None = None
        self._main_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._initialized = False
        self._closing = False
        self._close_lock = asyncio.Lock()

        # provider connection
        self._provider: SpatialRealSession | None = None
        self._generation = 0
        self._expected_close_generations: set[int] = set()
        self._reconnect_task: asyncio.Task[bool] | None = None
        self._recovery_exhausted = False
        self._playback_control = False
        # serializes everything sent to the provider, so an interrupt can't overtake
        # the audio it is meant to stop (or the reverse)
        self._io_lock = asyncio.Lock()

        # replies: the one being fed (`_input`) and every one not yet played out
        self._input: _Reply | None = None
        self._replies: list[_Reply] = []
        self._drop_until_segment_end = False
        # bumped on every clear: audio taken from the queue before it is stale
        self._clear_generation = 0
        self._interrupt_pending = False
        # a send has the request open on the server before its id comes back
        self._send_in_flight = False
        # pause: what the framework asked for, and whether the server holds playback
        self._paused = False
        self._server_holding = False

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "spatialreal"

    # --- lifecycle ----------------------------------------------------------------------

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
        url = str(livekit_url if utils.is_given(livekit_url) else os.getenv("LIVEKIT_URL", ""))
        key = str(
            livekit_api_key if utils.is_given(livekit_api_key) else os.getenv("LIVEKIT_API_KEY", "")
        )
        secret = str(
            livekit_api_secret
            if utils.is_given(livekit_api_secret)
            else os.getenv("LIVEKIT_API_SECRET", "")
        )
        if not url or not key or not secret:
            raise SpatialRealException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set by arguments "
                "or environment variables"
            )

        await super().start(agent_session, room)

        room_name = room.name
        egress_attributes = {
            ATTRIBUTE_PUBLISH_ON_BEHALF: self._resolve_local_participant_identity(room)
        }
        livekit_token = (
            api.AccessToken(api_key=key, api_secret=secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_ttl(self._livekit_token_ttl)
            .with_attributes(egress_attributes)
            .with_grants(_egress_video_grants(room_name))
            .to_jwt()
        )
        self._livekit_egress = LiveKitEgressConfig(
            url=url,
            api_token=livekit_token,
            room_name=room_name,
            publisher_id=self._avatar_participant_identity,
            extra_attributes=egress_attributes,
            idle_timeout=self._idle_timeout_seconds,
        )
        sample_rate = self._sample_rate or (
            agent_session.tts.sample_rate if agent_session.tts else DEFAULT_SAMPLE_RATE
        )
        self._resolved_sample_rate = sample_rate
        self._agent_session = agent_session
        self._room = room
        self._original_audio_output = agent_session.output.audio
        self._original_audio_tail = self._resolve_audio_tail(agent_session.output.audio)
        logger.debug("starting SpatialReal avatar session", extra={"room": room_name})

        try:
            attempt = await self._connect_with_retries()
            self._emit_connection_state("connected", reason="initial_start", attempt=attempt)

            # wait_playback_start: the framework's playback_started (transcript sync,
            # first-frame metrics) fires when the server reports the avatar playing,
            # not when the first frame is handed over.
            self._audio_buffer = ResumableQueueAudioOutput(
                sample_rate=sample_rate,
                wait_playback_start=True,
                can_pause=self._playback_control,
            )
            await self._audio_buffer.start()
            self._audio_buffer.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]
            self._audio_buffer.on("pause", self._on_pause)  # type: ignore[arg-type]
            self._audio_buffer.on("resume", self._on_resume)  # type: ignore[arg-type]

            # keep output wrappers (TranscriptSynchronizer, recorder, ...) in the chain
            agent_session.output.replace_audio_tail(self._audio_buffer)
            self._main_task = asyncio.create_task(
                self._forward_audio(), name="spatialreal_avatar_audio_forwarder"
            )
            self._initialized = True

            # Interruption is owned by the framework: pause holds a candidate
            # interruption, resume restores a rejected one, clear_buffer confirms one.
            @agent_session.on("close")
            def _on_session_close(_: Any) -> None:
                self._spawn(self.aclose(), name="spatialreal_avatar_session_close")

        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception as e:
            logger.debug("SpatialReal avatar session startup failed", exc_info=True)
            await self.aclose()
            raise SpatialRealException(
                "Failed to start SpatialReal avatar session. Check SpatialReal credentials, "
                "LiveKit room auth/token configuration, endpoint URLs, and outbound network "
                f"access. room={room_name}, avatar_id={self._avatar_id}, "
                f"ingress_endpoint_url={self._ingress_endpoint_url or '(from environment)'}, "
                f"sample_rate={sample_rate}. Reason: {self._format_error_reason(e)}"
            ) from None

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closing:
                return
            self._closing = True

            current = asyncio.current_task()
            tasks = [t for t in self._background_tasks if t is not current and not t.done()]
            if self._main_task is not None:
                tasks.append(self._main_task)
                self._main_task = None
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            self._finish_all(reason="session_close")
            self._detach_audio_output()
            if self._audio_buffer is not None:
                await self._audio_buffer.aclose()
                self._audio_buffer = None

            provider, self._provider = self._provider, None
            if provider is not None:
                try:
                    await provider.close()
                    logger.debug("SpatialReal avatar session closed")
                except Exception as e:
                    logger.warning("Error closing SpatialReal avatar session", exc_info=e)

            # The base class removes the avatar participant from the room and unregisters
            # what it registered in start(). It reads _agent_session and _room, so it runs
            # before they are cleared.
            try:
                await super().aclose()
            except Exception as e:
                logger.warning("Error in base avatar session close", exc_info=e)

            self._initialized = False
            self._room = None
            self._agent_session = None
            self._original_audio_output = None
            self._original_audio_tail = None

    def _detach_audio_output(self) -> None:
        if self._agent_session is None or self._audio_buffer is None:
            return
        output = self._agent_session.output
        if output.audio is self._audio_buffer:
            # no wrappers were present, so the whole chain was replaced
            output.audio = self._original_audio_output
        elif (
            self._resolve_audio_tail(output.audio) is self._audio_buffer
            and self._original_audio_tail is not None
        ):
            output.replace_audio_tail(self._original_audio_tail)

    async def disconnect_provider_for_test(self) -> None:
        """Close the ingress WebSocket so a test can exercise provider recovery."""
        if self._closing or self._provider is None:
            raise SpatialRealException("SpatialReal provider session is unavailable")
        await self._provider.close()

    # --- provider connection ------------------------------------------------------------

    async def _connect(self) -> None:
        if self._livekit_egress is None or self._resolved_sample_rate is None:
            raise SpatialRealException("SpatialReal provider configuration is unavailable")
        self._generation += 1
        generation = self._generation
        provider = new_avatar_session(
            api_key=self._api_key,
            app_id=self._app_id,
            avatar_id=self._avatar_id,
            **self._endpoint_options(),
            expire_at=datetime.now(timezone.utc) + DEFAULT_SESSION_TTL,
            livekit_egress=self._livekit_egress,
            sample_rate=self._resolved_sample_rate,
            on_playback=lambda signal: self._on_playback_signal(signal, generation),
            on_playback_state=lambda event: self._on_playback_state(event, generation),
            on_error=lambda error: self._on_provider_lost(generation, "provider_error", error),
            on_close=lambda: self._on_provider_lost(generation, "provider_closed"),
        )

        async def _start() -> None:
            await provider.init()
            await provider.start()

        try:
            await asyncio.wait_for(_start(), timeout=PROVIDER_CONNECT_TIMEOUT_SECONDS)
        except BaseException:
            await self._close_quietly(provider, generation)
            raise
        # pause needs the played position too: time held isn't time played
        self._playback_control = {PLAYBACK_CONTROL_CAPABILITY, PLAYBACK_STATE_CAPABILITY} <= set(
            provider.capabilities
        )
        self._provider = provider

    async def _connect_with_retries(self) -> int:
        last_error: Exception | None = None
        for attempt, delay in enumerate(PROVIDER_RECONNECT_DELAYS_SECONDS, start=1):
            if delay > 0:
                await asyncio.sleep(delay)
            if self._closing:
                raise SpatialRealException("SpatialReal avatar session is closing")
            try:
                await self._connect()
                return attempt
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                logger.warning(
                    "SpatialReal provider connect attempt failed",
                    exc_info=error,
                    extra={"attempt": attempt},
                )
        assert last_error is not None
        raise last_error

    async def _close_quietly(self, provider: SpatialRealSession, generation: int) -> None:
        self._expected_close_generations.add(generation)
        try:
            await asyncio.wait_for(provider.close(), timeout=PROVIDER_CLOSE_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("SpatialReal provider close did not complete cleanly", exc_info=error)
        finally:
            self._expected_close_generations.discard(generation)

    def _on_provider_lost(
        self, generation: int, reason: str, error: Exception | None = None
    ) -> None:
        if (
            self._closing
            or generation != self._generation
            or generation in self._expected_close_generations
            or self._provider is None  # already being replaced
        ):
            return
        logger.warning(
            "SpatialReal provider connection lost",
            exc_info=error,
            extra={"generation": generation, "reason": reason},
        )
        # whatever the lost connection was playing won't play: finish it, and drop the
        # rest of a reply that is still being fed
        self._finish_all(reason=reason, drop_rest=True)
        provider, self._provider = self._provider, None
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(
                self._reconnect(provider, generation, reason),
                name="spatialreal_provider_reconnect",
            )
            self._background_tasks.add(self._reconnect_task)
            self._reconnect_task.add_done_callback(self._background_tasks.discard)

    async def _reconnect(
        self, lost: SpatialRealSession | None, generation: int, reason: str
    ) -> bool:
        self._emit_connection_state("recovering", reason=reason, attempt=0)
        if lost is not None:
            await self._close_quietly(lost, generation)
        try:
            attempt = await self._connect_with_retries()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._closing:
                return False
            self._recovery_exhausted = True
            logger.error(
                "SpatialReal provider recovery exhausted; the avatar stays silent",
                extra={"reason": f"{reason}:{self._format_error_reason(error)}"},
            )
            self._emit_connection_state("failed", reason=reason, attempt=0)
            return False
        logger.info("SpatialReal provider reconnected", extra={"attempt": attempt})
        self._emit_connection_state("recovered", reason=reason, attempt=attempt)
        return True

    async def _ready_provider(self) -> SpatialRealSession:
        """The live provider, waiting out a reconnect in flight."""
        task = self._reconnect_task
        if self._provider is None and task is not None and not task.done():
            await asyncio.shield(task)
        if self._provider is None:
            raise SpatialRealException("SpatialReal provider is unavailable")
        return self._provider

    def _emit_connection_state(
        self,
        state: Literal["connected", "recovering", "recovered", "failed"],
        *,
        reason: str,
        attempt: int,
    ) -> None:
        self.emit(
            "provider_connection_state_changed",
            AvatarProviderConnectionEvent(
                state=state,
                reason=reason,
                attempt=attempt,
                generation=self._generation,
                observed_at=time.time(),
            ),
        )

    # --- audio forwarding ---------------------------------------------------------------

    async def _forward_audio(self) -> None:
        assert self._audio_buffer is not None
        async for item in self._audio_buffer:
            clear_generation = self._clear_generation
            try:
                if isinstance(item, AudioSegmentEnd):
                    await self._end_input(clear_generation)
                else:
                    await self._send_frame(item, clear_generation)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._input_failed(item, error, clear_generation)

    @contextlib.asynccontextmanager
    async def _provider_io(self) -> AsyncIterator[SpatialRealSession]:
        """The live provider, held exclusively; waits out a reconnect in flight.

        A provider that fails while in use is reported as lost, under its own generation,
        so a stale reference can never tear down its replacement.
        """
        while True:
            provider = await self._ready_provider()
            generation = self._generation
            async with self._io_lock:
                if provider is not self._provider:
                    continue  # replaced while waiting for the lock
                await self._send_pending_interrupt(provider)
                try:
                    yield provider
                except Exception as error:
                    self._on_provider_lost(generation, "forwarding_error", error)
                    raise
                return

    async def _send_frame(self, frame: rtc.AudioFrame, clear_generation: int) -> None:
        if self._drop_until_segment_end:
            return
        if self._input is None:
            # the framework counts the reply from its first frame, so it gets a record
            # (and a finish) even if sending it fails
            self._input = _Reply()
            self._replies.append(self._input)
        reply = self._input
        async with self._provider_io() as provider:
            if clear_generation != self._clear_generation or reply.finished:
                return
            self._send_in_flight = True
            try:
                reply.req_id = await provider.send_audio(bytes(frame.data), end=False)
            finally:
                self._send_in_flight = False
            reply.pushed += frame.duration
            if self._paused and not self._server_holding and self._playback_control:
                # paused before this reply reached the server: hold it from the start
                await self._hold(provider)

    async def _end_input(self, clear_generation: int) -> None:
        if self._drop_until_segment_end:
            self._drop_until_segment_end = False
            return
        reply, self._input = self._input, None
        if reply is None or reply.finished:
            return
        if reply.req_id is None:
            self._finish(reply, interrupted=True, reason="no_audio_sent")
            return
        try:
            async with self._provider_io() as provider:
                if clear_generation != self._clear_generation or reply.finished:
                    return
                await provider.send_audio(b"", end=True)
                reply.input_done = True
        except Exception:
            self._finish(reply, interrupted=True, reason="forwarding_error")
            raise
        self._arm_completion_fallback(reply)

    def _input_failed(
        self, item: rtc.AudioFrame | AudioSegmentEnd, error: Exception, clear_generation: int
    ) -> None:
        if self._recovery_exhausted:
            logger.warning("SpatialReal provider is unavailable; finishing the reply")
        else:
            logger.error("SpatialReal provider audio forwarding failed", exc_info=error)
        if clear_generation != self._clear_generation:
            return  # the reply was interrupted meanwhile, and the clear finished it
        reply, self._input = self._input, None
        if reply is None:
            return  # already finished (a lost connection or a pause timeout drops its rest)
        self._finish(reply, interrupted=True, reason="forwarding_error")
        # the end marker of this reply is still to come: drop what is left of it
        self._drop_until_segment_end = isinstance(item, rtc.AudioFrame)

    async def _send_pending_interrupt(self, provider: SpatialRealSession) -> None:
        if not self._interrupt_pending:
            return
        self._interrupt_pending = False
        try:
            await provider.interrupt()
        except Exception as e:
            logger.warning("Failed to interrupt SpatialReal avatar", exc_info=e)

    # --- framework events ---------------------------------------------------------------

    def _on_clear_buffer(self) -> None:
        """A confirmed interruption: finish what is playing and stop the server."""
        if self._closing:
            return
        self._clear_generation += 1
        self._paused = False
        # LiveKit flushes before clearing, and the clear can drain the queued segment end,
        # so the clear is itself the boundary of the interrupted reply
        self._drop_until_segment_end = False
        sent = self._send_in_flight or any(r.req_id is not None for r in self._replies)
        self._finish_all(reason="interrupt")
        self._finish_drained_segments()
        if not sent:
            return
        self._interrupt_pending = True
        self._spawn(self._interrupt(), name="spatialreal_avatar_interrupt")

    def _finish_drained_segments(self) -> None:
        """Finish segments the clear drained from the queue before any frame was read.

        The framework counts a segment when it captures the first frame, so one cleared
        before it reached us has no record here, but still needs its finish.
        """
        buffer = self._audio_buffer
        if buffer is None:
            return
        for _ in range(buffer._pending_playback_count):
            buffer.notify_playback_started()
            buffer.notify_playback_finished(playback_position=0.0, interrupted=True)

    async def _interrupt(self) -> None:
        async with self._io_lock:
            provider = self._provider
            if provider is None:
                self._interrupt_pending = False  # a replacement connection has nothing to stop
                return
            await self._send_pending_interrupt(provider)
        logger.debug("SpatialReal avatar interrupted")

    def _on_pause(self) -> None:
        if self._closing:
            return
        self._paused = True
        self._spawn(self._apply_pause(), name="spatialreal_avatar_pause")

    def _on_resume(self) -> None:
        if self._closing:
            return
        self._paused = False
        self._spawn(self._apply_resume(), name="spatialreal_avatar_resume")

    async def _apply_pause(self) -> None:
        async with self._io_lock:
            provider = self._provider
            if provider is None:
                return
            await self._send_pending_interrupt(provider)
            in_flight = any(r.req_id is not None for r in self._replies)
            # with nothing sent yet the reply is held as soon as its first frame goes out
            if self._paused and not self._server_holding and in_flight:
                await self._hold(provider)

    async def _apply_resume(self) -> None:
        async with self._io_lock:
            provider = self._provider
            if provider is None or self._paused or not self._server_holding:
                return
            try:
                await provider.resume()
            except Exception as e:
                logger.warning("Failed to resume SpatialReal avatar playback", exc_info=e)
            self._set_held(False)
            logger.debug(
                "SpatialReal avatar playback resumed (server-side)",
                extra={"request_id": self._latest_req_id()},
            )

    async def _hold(self, provider: SpatialRealSession) -> None:
        try:
            await provider.pause()
        except Exception as e:
            logger.warning("Failed to pause SpatialReal avatar playback", exc_info=e)
            return
        self._set_held(True)
        logger.debug(
            "SpatialReal avatar playback paused (server-side)",
            extra={"request_id": self._latest_req_id()},
        )

    def _set_held(self, held: bool) -> None:
        # the server's reply refines the position; until then it runs on from here
        self._server_holding = held
        now = time.time()
        for reply in self._replies:
            if reply.reported_at is not None:
                reply.report(reply.played(now), now, held=held)

    # --- server events ------------------------------------------------------------------

    def _latest_req_id(self) -> str | None:
        return next((r.req_id for r in reversed(self._replies) if r.req_id), None)

    def _reply_for(self, req_id: str) -> _Reply | None:
        if not req_id:
            return None
        return next((r for r in self._replies if r.req_id == req_id), None)

    def _on_playback_signal(self, signal: PlaybackSignal, generation: int) -> None:
        if generation != self._generation:
            return
        # an unknown request already finished (the egress can repeat end=true) or
        # belongs to a reply that was interrupted
        reply = self._reply_for(signal.req_id)
        if reply is None:
            return
        self._mark_started(reply, source="spatialreal_transport_frame")
        if signal.end:
            self._finish(reply, interrupted=False, reason="provider_end")

    def _on_playback_state(self, event: PlaybackStateEvent, generation: int) -> None:
        # start and end arrive through the playback signal; this gives the played position,
        # and tells when the server gave up on a held pause
        if generation != self._generation:
            return
        reply = self._reply_for(event.req_id)
        if reply is None:
            return
        if event.state in (PlaybackState.PLAYING, PlaybackState.PAUSED):
            reply.report(
                event.played_ms / 1000, time.time(), held=event.state == PlaybackState.PAUSED
            )
            return
        if (
            event.state != PlaybackState.INTERRUPTED
            or event.reason != InterruptReason.PAUSE_TIMEOUT
        ):
            return
        logger.warning(
            "SpatialReal avatar pause timed out on the server; finishing the reply",
            extra={"request_id": event.req_id, "played_ms": event.played_ms},
        )
        reply.report(event.played_ms / 1000, time.time(), held=True)
        self._paused = False
        self._server_holding = False
        if reply is self._input:
            self._input = None
            self._drop_until_segment_end = True
            # the server dropped the request, but the SDK still has it open: close it, or
            # the next reply would be sent under the abandoned request id
            self._interrupt_pending = True
            self._spawn(self._interrupt(), name="spatialreal_avatar_pause_timeout")
        self._finish(reply, interrupted=True, reason="pause_timeout")

    # --- reply bookkeeping --------------------------------------------------------------

    def _mark_started(self, reply: _Reply, *, source: str) -> None:
        if reply.started_at is not None:
            return
        reply.started_at = time.time()
        if reply.reported_at is None:
            reply.report(0.0, reply.started_at, held=False)
        if self._audio_buffer is not None:
            self._audio_buffer.notify_playback_started()
        logger.debug(
            "SpatialReal avatar playback started",
            extra={"request_id": reply.req_id, "source": source},
        )
        self.emit(
            "playback_started",
            AvatarPlaybackStartedEvent(
                request_id=reply.req_id or "",
                source=source,
                observed_at=reply.started_at,
            ),
        )
        if reply.input_done:
            # the fallback deadline now counts from the real start
            self._arm_completion_fallback(reply)

    def _finish(self, reply: _Reply, *, interrupted: bool, reason: str) -> None:
        """Tell the framework one reply is done. Each reply is finished exactly once."""
        if reply.finished:
            return
        reply.finished = True
        position = reply.played(time.time()) if interrupted else reply.pushed
        if reply in self._replies:
            self._replies.remove(reply)
        if self._input is reply:
            self._input = None
        task = reply.completion_task
        reply.completion_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        if not self._replies:
            self._server_holding = False

        logger.debug(
            "SpatialReal avatar segment playback completed",
            extra={
                "request_id": reply.req_id,
                "reason": reason,
                "interrupted": interrupted,
                "playback_position": round(position, 3),
                "pushed_duration": round(reply.pushed, 3),
            },
        )
        if self._audio_buffer is None:
            return
        if reply.started_at is None:
            # the framework expects playback_started before playback_finished
            self._audio_buffer.notify_playback_started()
        self._audio_buffer.notify_playback_finished(
            playback_position=position, interrupted=interrupted
        )

    def _finish_all(self, *, reason: str, drop_rest: bool = False) -> None:
        if drop_rest and self._input is not None:
            self._drop_until_segment_end = True
        for reply in list(self._replies):
            self._finish(reply, interrupted=True, reason=reason)
        self._server_holding = False

    def _arm_completion_fallback(self, reply: _Reply) -> None:
        if reply.completion_task is not None:
            reply.completion_task.cancel()
        reply.completion_task = self._spawn(
            self._completion_fallback(reply), name="spatialreal_avatar_completion_fallback"
        )

    async def _completion_fallback(self, reply: _Reply) -> None:
        armed_at = time.time()
        while not reply.finished:
            now = time.time()
            if reply.held or self._server_holding:
                # time held by a pause isn't playback: wait it out
                await asyncio.sleep(COMPLETION_MARGIN_SECONDS)
                continue
            if reply.reported_at is None:
                deadline = armed_at + reply.pushed + START_ALLOWANCE_SECONDS
            else:
                # when the rest should have played out; each server report moves it, so it
                # is worked out again on every wake-up
                deadline = reply.reported_at + reply.pushed - reply.played_at_report
            deadline += COMPLETION_MARGIN_SECONDS
            if now < deadline:
                await asyncio.sleep(deadline - now)
                continue
            logger.warning(
                "SpatialReal reported no end of playback; finishing the reply",
                extra={"request_id": reply.req_id},
            )
            self._finish(reply, interrupted=False, reason="completion_fallback")

    # --- helpers ------------------------------------------------------------------------

    def _spawn(self, coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task[Any] | None:
        if self._closing:
            coro.close()
            return None
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _endpoint_options(self) -> dict[str, str]:
        options = {
            "environment": self._environment,
            "console_endpoint_url": self._console_endpoint_url,
            "ingress_endpoint_url": self._ingress_endpoint_url,
        }
        return {key: value for key, value in options.items() if value}

    @staticmethod
    def _required(value: NotGivenOr[str], name: str, env: str) -> str:
        resolved = value if utils.is_given(value) else os.getenv(env)
        if not resolved:
            raise SpatialRealException(
                f"{name} must be set either by passing it to AvatarSession or "
                f"by setting the {env} environment variable"
            )
        return str(resolved)

    @staticmethod
    def _optional(value: NotGivenOr[str], env: str) -> str:
        return str(value if utils.is_given(value) else os.getenv(env) or "")

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("%s=%r is invalid; using %d", name, raw, default)
            return default
        if value <= 0:
            logger.warning("%s=%d must be > 0; using %d", name, value, default)
            return default
        return value

    @staticmethod
    def _resolve_audio_tail(sink: AudioOutput | None) -> AudioOutput | None:
        while sink is not None and sink.next_in_chain is not None:
            sink = sink.next_in_chain
        return sink

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
        root: BaseException = error
        seen: set[int] = set()
        while id(root) not in seen:
            seen.add(id(root))
            cause = root.__cause__ or (None if root.__suppress_context__ else root.__context__)
            if cause is None:
                break
            root = cause
        message = str(root) or str(error)
        return f"{type(root).__name__}: {message}" if message else type(root).__name__
