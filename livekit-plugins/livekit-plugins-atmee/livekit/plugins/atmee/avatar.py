# Copyright 2026 Atmanity
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
from datetime import timedelta
from typing import Any, Literal

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import (
    AtmeeAPI,
    AtmeeException,
    AvatarSessionInfo,
    AvatarVersion,
    WaitFor,
    _check_avatar_version,
)
from .log import logger

# The rendering worker resamples nothing: it consumes 16 kHz mono PCM over the
# data stream, so the agent's audio tail is asked for exactly that.
SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "atmee-avatar-agent"
_AVATAR_AGENT_NAME = "atmee-avatar-agent"
# Ceiling of a session when none is given (also the most a session can bill
# if the agent dies without the worker ever reporting; server default).
DEFAULT_MAX_DURATION_SECONDS = 3600
# The avatar's token outlives the session ceiling by this much so a slow start
# never joins with an expired token.
_TOKEN_TTL_SLACK = timedelta(minutes=10)


class AvatarSession(BaseAvatarSession[Literal["avatar_disconnected"]]):
    """An Atmee avatar session: render an Atmee **v1 avatar** into your agent's room.

    v1 avatars are talking heads generated from a single portrait (see
    :meth:`AtmeeAPI.create_avatar`); this plugin targets them specifically.

    The avatar joins the room as its own participant (``atmee-avatar-agent``)
    publishing on behalf of your agent; your agent's audio output is replaced
    by a data stream to it, and the avatar publishes synced video and audio.
    Billing runs while the avatar is in the room and stops when your agent
    leaves, the room closes, or :meth:`aclose` runs (which also removes the
    avatar participant); a ceiling of ``max_duration_seconds`` bounds it.

    Emits ``avatar_disconnected`` (the ``rtc.RemoteParticipant``) if the
    avatar participant leaves the room while the session is still running,
    for example because its rendering worker died.
    """

    def __init__(
        self,
        *,
        avatar_id: str,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
        metadata: dict[str, Any] | None = None,
        wait_for: WaitFor = "initializing",
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        http_session: aiohttp.ClientSession | None = None,
        avatar_version: AvatarVersion = "v1",
    ) -> None:
        """
        Args:
            avatar_id: the Atmee avatar to render (``AtmeeAPI.create_avatar``).
            avatar_version: the avatar generation to render. ``"v1"`` (default)
                is a talking head generated from a single portrait, the only
                version the plugin renders today; ``"v2"``, Atmee's next
                generation, is not available through the plugin yet and
                raises ``ValueError`` right here.
            api_key: your ``sk_atmee_...`` key; defaults to ``ATMEE_API_KEY``.
            api_url: API base; defaults to ``ATMEE_API_URL`` or the Atmee cloud.
            avatar_participant_identity: identity the avatar joins with
                (``atmee-avatar-agent``). Set it when one room hosts several
                avatars.
            max_duration_seconds: hard ceiling of the session (default 3600).
            metadata: free-form JSON attached to the session on the Atmee side.
            wait_for: ``"initializing"`` (default) returns from :meth:`start`
                once the worker acknowledged the start; ``"avatar_joined"``
                blocks until the avatar is in the room.
            http_session: an aiohttp session to reuse; the job's shared one by
                default, or a private one outside a job (released by
                :meth:`aclose`).
        """
        super().__init__()
        if not avatar_id:
            raise AtmeeException("avatar_id is required")
        _check_avatar_version(avatar_version)
        self._avatar_id = avatar_id
        self._avatar_version: AvatarVersion = avatar_version
        self._max_duration_seconds = int(max_duration_seconds)
        self._metadata = metadata
        self._wait_for: WaitFor = wait_for
        self._conn_options = conn_options
        # Resolved lazily by AtmeeAPI: the job's shared session inside a
        # worker, a private one (closed by aclose) outside.
        self._api = AtmeeAPI(
            api_key=api_key,
            api_url=api_url,
            conn_options=conn_options,
            session=http_session,
        )

        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME

        self.session_id: str | None = None
        """The Atmee avatar session id once :meth:`start` returned."""
        self.session_info: AvatarSessionInfo | None = None
        """The full start response once :meth:`start` returned."""

        self._end_task: asyncio.Task[None] | None = None
        self._end_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._ended = False
        self._started = False
        self._render_requested = False
        self._room_for_events: rtc.Room | None = None
        self._agent_session_for_events: AgentSession[Any] | None = None
        self._aclose_task: asyncio.Task[None] | None = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "atmee"

    @property
    def avatar_version(self) -> AvatarVersion:
        """The avatar generation this session renders (``"v1"``)."""
        return self._avatar_version

    @property
    def api(self) -> AtmeeAPI:
        """The API client this session uses (same key and base URL)."""
        return self._api

    async def start(
        self,
        agent_session: AgentSession[Any],
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Start rendering the avatar into ``room`` for ``agent_session``.

        Mints a LiveKit access token for the avatar participant with YOUR
        project's credentials (``LIVEKIT_URL`` / ``LIVEKIT_API_KEY`` /
        ``LIVEKIT_API_SECRET`` or the arguments), asks Atmee to render into
        the room, and routes the agent's audio to the avatar. Call before
        ``agent_session.start``.

        One-shot: each call creates a separately billed render, so a second
        ``start()`` on the same instance raises. After a failed start, create
        a new ``AvatarSession``.
        """
        if self._started:
            raise AtmeeException(
                "AvatarSession.start() was already called; create a new AvatarSession "
                "for another render"
            )
        self._started = True
        await super().start(agent_session, room)
        try:
            await self._start_render(
                agent_session,
                room,
                livekit_url=livekit_url,
                livekit_api_key=livekit_api_key,
                livekit_api_secret=livekit_api_secret,
            )
        except BaseException:
            # Undo what super().start() set up (listeners, join task). Once the
            # render was requested its outcome is uncertain, so close fully:
            # end a session that may exist and remove the avatar participant.
            # Before that, no avatar of ours can be in the room, and removing
            # the participant could disconnect another session's avatar that
            # uses the same identity. Either way the instance stays spent.
            if self._render_requested:
                await self.aclose()
            else:
                await self._release_without_removing_participant()
            raise

    async def _release_without_removing_participant(self) -> None:
        async with self._close_lock:
            room = self._room
            if room is not None:
                room.off("connection_state_changed", self._on_connection_state_changed)
                # the base aclose() removes the avatar identity from a
                # connected room; with no room it only drops its listener
                # and cancels the join task
                self._room = None
            await super().aclose()
            await self._api.aclose()

    async def _start_render(
        self,
        agent_session: AgentSession[Any],
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str],
        livekit_api_key: NotGivenOr[str],
        livekit_api_secret: NotGivenOr[str],
    ) -> None:

        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise AtmeeException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context(required=False)
        local_participant_identity = (
            job_ctx.local_participant_identity
            if job_ctx is not None
            else room.local_participant.identity
        )
        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar to publish audio and video on behalf of your local agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .with_ttl(timedelta(seconds=self._max_duration_seconds) + _TOKEN_TTL_SLACK)
            .to_jwt()
        )

        logger.debug(
            "starting atmee avatar session",
            extra={
                "avatar_id": self._avatar_id,
                "avatar_version": self._avatar_version,
                "lk.pii.room": room.name,
            },
        )
        self._render_requested = True
        info = await self._api.create_avatar_session(
            self._avatar_id,
            livekit_url=livekit_url,
            livekit_token=livekit_token,
            agent_identity=local_participant_identity,
            max_duration_seconds=self._max_duration_seconds,
            metadata=self._metadata,
            wait_for=self._wait_for,
            avatar_version=self._avatar_version,
        )
        self.session_info = info
        self.session_id = info.session_id
        logger.info(
            "atmee avatar session started",
            extra={
                "session_id": info.session_id,
                "avatar_version": info.avatar_version,
                "status": info.status,
                "billing_mode": info.billing_mode,
                "max_duration_seconds": info.max_duration_seconds,
            },
        )

        self._room_for_events = room
        room.on("participant_disconnected", self._on_participant_disconnected)
        # The agent session can close without the job shutting down
        # (AgentSession.aclose(), an error): end the render then too, or it
        # would bill until max_duration_seconds.
        self._agent_session_for_events = agent_session
        agent_session.on("close", self._on_agent_session_close)

        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            ),
        )

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_participant_identity or self._ended:
            return
        # The avatar left while we are still running: its worker died or was
        # removed. The worker reports its own end to Atmee; ending here too
        # closes the gap for a worker that could not (idempotent).
        logger.warning(
            "atmee avatar participant left the room",
            extra={"session_id": self.session_id, "lk.pii.identity": participant.identity},
        )
        self.emit("avatar_disconnected", participant)
        self._schedule_end()

    def _on_agent_session_close(self, _: Any) -> None:
        if self._aclose_task is None:
            self._aclose_task = asyncio.create_task(self.aclose())

    def _schedule_end(self) -> None:
        if self._end_task is None or self._end_task.done():
            self._end_task = asyncio.create_task(self._end_session())

    async def _end_session(self) -> None:
        """Best-effort ``POST /v1/avatar_sessions/{id}/end``.

        Serialized, and marked done only once the API confirmed the end (the
        call is idempotent), so a failed or cancelled attempt can be retried
        by a later :meth:`aclose`.
        """
        async with self._end_lock:
            if self._ended or not self.session_id:
                return
            try:
                await self._api.end_avatar_session(self.session_id)
            except AtmeeException as e:
                if 400 <= e.status_code < 500 and e.status_code not in (408, 429):
                    # The session is unknown or already settled on the Atmee
                    # side: nothing left to end, retrying would not help.
                    self._ended = True
                logger.warning(
                    "failed to end atmee avatar session; the worker's own report or the "
                    "session ceiling will settle it",
                    extra={"session_id": self.session_id, "error": str(e)},
                )
                return
            except Exception as e:  # never let a billing hint break teardown
                logger.warning(
                    "failed to end atmee avatar session; the worker's own report or the "
                    "session ceiling will settle it",
                    extra={"session_id": self.session_id, "error": str(e)},
                )
                return
            self._ended = True
            logger.debug("atmee avatar session ended", extra={"session_id": self.session_id})

    async def aclose(self) -> None:
        """End the Atmee session and remove the avatar from the room.

        Registered as a job shutdown callback by :meth:`start`; call it
        yourself when running outside a job. Safe to call more than once and
        concurrently (the agent session's ``close`` event also triggers it):
        calls are serialized, so the HTTP session is never closed while
        another call is still ending the render.
        """
        async with self._close_lock:
            await self._aclose_locked()

    async def _aclose_locked(self) -> None:
        if self._room_for_events is not None:
            self._room_for_events.off("participant_disconnected", self._on_participant_disconnected)
            self._room_for_events = None
        if self._agent_session_for_events is not None:
            self._agent_session_for_events.off("close", self._on_agent_session_close)
            self._agent_session_for_events = None
        if self._end_task is not None and not self._end_task.done():
            try:
                await self._end_task
            except Exception:
                pass
        await self._end_session()
        await super().aclose()
        await self._api.aclose()
