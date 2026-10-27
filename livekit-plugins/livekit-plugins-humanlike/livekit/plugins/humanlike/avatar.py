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
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput

from .api import DEFAULT_API_URL, HumanlikeAPI, HumanlikeException
from .log import logger

DEFAULT_MODEL = "humanlike/homo-1"
SAMPLE_RATE = 24000
DEFAULT_START_TIMEOUT = 180.0
DEFAULT_JOIN_TIMEOUT = 60.0
ATTRIBUTE_RENDER = "humanlike.avatar.render"
"""Participant attribute on the agent that carries what a client-rendered model draws."""

_AVATAR_IDENTITY = "humanlike-avatar-agent"
_AVATAR_NAME = "Humanlike Avatar"
_TOKEN_TTL = timedelta(hours=6)
# A capped session (HOMO 1 stops at 570 s) is renewed this long before its cap.
_RENEW_BEFORE_S = 45.0
_MIN_RENEW_DELAY_S = 30.0


def _to_ws_url(url: str) -> str:
    """LiveKit Cloud injects an https:// URL on deployed agents; the avatar connects over ws(s)."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


class AvatarSession(BaseAvatarSession):
    """A Humanlike realtime avatar for a LiveKit voice agent.

    ``model`` picks the avatar model by its public id. Humanlike's own models are
    ``humanlike/homo-1`` (the default, rendered on a GPU and published as video) and
    ``humanlike/homo-0`` (rendered in the viewer's browser or app, see
    :data:`ATTRIBUTE_RENDER`). The same API key also starts the other vendors' models
    Humanlike resells, such as ``anam/cara-4``.

    Call ``start()`` before ``AgentSession.start()``. The API key falls back to the
    ``HUMANLIKE_API_KEY`` environment variable.
    """

    def __init__(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        image_url: NotGivenOr[str] = NOT_GIVEN,
        options: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        fallback: NotGivenOr[list[str]] = NOT_GIVEN,
        max_session_seconds: NotGivenOr[int] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        join_timeout: float = DEFAULT_JOIN_TIMEOUT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            model: Public model id, e.g. ``humanlike/homo-1`` (default) or ``anam/cara-4``.
            avatar_id: The avatar (face or persona) to use, for models that take one.
            image_url: A photo to animate, for models that take one.
            options: Model-specific options, checked by the API against the model's schema.
            fallback: Model ids to try, in order, if ``model`` cannot start.
            max_session_seconds: End the avatar after this long.
            api_url: Defaults to ``HUMANLIKE_API_URL`` or ``https://api.humanlike.co``.
            api_key: Defaults to ``HUMANLIKE_API_KEY``. Use a secret key, never a publishable one.
            avatar_participant_identity: The identity the avatar joins under. Must be unique
                per concurrent avatar in a room.
            avatar_participant_name: The display name the avatar joins under.
            start_timeout: How long the start request may take, cold GPU boot included.
            join_timeout: How long to wait for the avatar's video once the API answered.
            conn_options: Retry settings for API requests.
            http_session: An aiohttp session to use outside a job context.
        """
        super().__init__()
        key = api_key if utils.is_given(api_key) else os.getenv("HUMANLIKE_API_KEY")
        if not key:
            raise HumanlikeException("HUMANLIKE_API_KEY must be set, or pass api_key")
        if key.startswith("hk_pub_"):
            raise HumanlikeException(
                "a publishable key (hk_pub_...) cannot start avatars; use a secret key"
            )
        self._api_key = key
        self._api_url = (
            api_url if utils.is_given(api_url) else os.getenv("HUMANLIKE_API_URL")
        ) or DEFAULT_API_URL

        self._model = model if utils.is_given(model) else DEFAULT_MODEL
        self._avatar_id = avatar_id
        self._image_url = image_url
        self._options = options
        self._fallback = fallback
        self._max_session_seconds = max_session_seconds
        self._avatar_identity = (
            avatar_participant_identity
            if utils.is_given(avatar_participant_identity)
            else _AVATAR_IDENTITY
        )
        self._avatar_name = (
            avatar_participant_name if utils.is_given(avatar_participant_name) else _AVATAR_NAME
        )
        self._start_timeout = start_timeout
        self._join_timeout = join_timeout
        self._conn_options = conn_options
        self._http_session = http_session

        self._client: HumanlikeAPI | None = None
        self._session_id: str | None = None
        self._model_id: str | None = None
        self._render: dict[str, Any] | None = None
        self._livekit: tuple[str, str, str] | None = None
        self._avatar_sid: str | None = None
        self._renew_task: asyncio.Task[None] | None = None
        self._end_task: asyncio.Task[None] | None = None
        self._renewing = False
        self._closed = False

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def provider(self) -> str:
        return "humanlike"

    @property
    def session_id(self) -> str | None:
        """The Humanlike session id, once started. Quote it when contacting support."""
        return self._session_id

    @property
    def model(self) -> str | None:
        """The model that started. Differs from the one asked for when a fallback started."""
        return self._model_id

    @property
    def render(self) -> dict[str, Any] | None:
        """What a client-rendered model (``humanlike/homo-0``) draws; ``None`` for video models."""
        return self._render

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        url = livekit_url if utils.is_given(livekit_url) else os.getenv("LIVEKIT_URL")
        lk_key = (
            livekit_api_key if utils.is_given(livekit_api_key) else os.getenv("LIVEKIT_API_KEY")
        )
        lk_secret = (
            livekit_api_secret
            if utils.is_given(livekit_api_secret)
            else os.getenv("LIVEKIT_API_SECRET")
        )
        if not url or not lk_key or not lk_secret:
            raise HumanlikeException(
                "livekit_url, livekit_api_key and livekit_api_secret must be set "
                "by arguments or environment variables"
            )
        self._livekit = (_to_ws_url(url), lk_key, lk_secret)

        await super().start(agent_session, room)
        self._client = HumanlikeAPI(
            self._api_key,
            self._api_url,
            session=self._http_session or utils.http_context.http_session(),
            conn_options=self._conn_options,
        )

        try:
            logger.debug("starting avatar session", extra={"model": self._model})
            view = await self._client.create_session(
                self._session_request(room), timeout=self._start_timeout
            )
            self._session_id = view.get("id")
            self._model_id = view.get("model")
            audio_in = view.get("audio_in")

            if audio_in == "livekit-datastream":
                room.on("participant_connected", self._on_participant_connected)
                room.on("participant_disconnected", self._on_participant_disconnected)
                agent_session.output.replace_audio_tail(
                    DataStreamAudioOutput(
                        room,
                        destination_identity=self._avatar_identity,
                        sample_rate=int(view.get("sample_rate") or SAMPLE_RATE),
                        # hold the agent's speech until the avatar's video is up
                        wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                    )
                )
                await self.wait_for_join(timeout=self._join_timeout)
            elif audio_in == "none":
                # Rendered by the client: the agent keeps its own audio track and
                # publishes what to draw. No avatar participant joins.
                self._render = view.get("render") or {}
                await room.local_participant.set_attributes(
                    {ATTRIBUTE_RENDER: json.dumps(self._render, separators=(",", ":"))}
                )
            else:
                raise HumanlikeException(
                    f"model {self._model_id!r} needs audio input {audio_in!r}, which this "
                    "plugin version does not support; upgrade livekit-plugins-humanlike"
                )
        except asyncio.TimeoutError as e:
            await self.aclose()
            raise HumanlikeException(f"avatar did not join within {self._join_timeout:.0f}s") from e
        except BaseException:
            await self.aclose()
            raise

        expires_at = view.get("expires_at")
        if expires_at and self._render is None:
            self._renew_task = asyncio.create_task(self._renew_loop(expires_at))

    async def wait_for_join(self, *, timeout: float | None = 30.0) -> None:
        if self._render is not None:
            return  # client-rendered: no avatar participant joins
        await super().wait_for_join(timeout=timeout)

    def _agent_identity(self, room: rtc.Room) -> str:
        # Inside a job the room may still be connecting when start() runs, so the job
        # context is the reliable source of the agent's identity.
        job_ctx = get_job_context(required=False)
        identity = (
            job_ctx.local_participant_identity
            if job_ctx is not None
            else room.local_participant.identity
        )
        if not identity:
            raise HumanlikeException(
                "the room's local participant has no identity; connect the room first"
            )
        return identity

    def _transport(self, room: rtc.Room) -> dict[str, Any]:
        """Where the avatar joins. It gets a token for one participant in one room, never
        the LiveKit API secret."""
        assert self._livekit is not None
        url, lk_key, lk_secret = self._livekit
        agent_identity = self._agent_identity(room)
        token = (
            api.AccessToken(lk_key, lk_secret)
            .with_kind("agent")
            .with_identity(self._avatar_identity)
            .with_name(self._avatar_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room.name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            # the avatar publishes audio and video on behalf of the agent
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: agent_identity})
            .with_ttl(_TOKEN_TTL)
            .to_jwt()
        )
        return {
            "type": "livekit",
            "url": url,
            "room_name": room.name,
            "token": token,
            "identity": self._avatar_identity,
            "agent_identity": agent_identity,
        }

    def _session_request(self, room: rtc.Room) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self._model, "transport": self._transport(room)}
        avatar: dict[str, str] = {}
        if utils.is_given(self._avatar_id):
            avatar["id"] = self._avatar_id
        if utils.is_given(self._image_url):
            avatar["image_url"] = self._image_url
        if avatar:
            body["avatar"] = avatar
        if utils.is_given(self._options):
            body["options"] = self._options
        if utils.is_given(self._fallback):
            body["fallback"] = list(self._fallback)
        if utils.is_given(self._max_session_seconds):
            body["max_session_seconds"] = self._max_session_seconds
        return body

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._avatar_identity:
            self._avatar_sid = participant.sid

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity != self._avatar_identity or self._closed or self._renewing:
            return
        if self._avatar_sid is not None and participant.sid != self._avatar_sid:
            return  # the predecessor of a renewed avatar
        if self._room is not None and not self._room.isconnected():
            return  # the room is closing; aclose() ends the session
        logger.warning("avatar left the room unexpectedly", extra={"session_id": self._session_id})
        # stop billing now rather than at the session's cap
        if self._end_task is None:
            self._end_task = asyncio.create_task(self._end_session())

    async def _renew_loop(self, expires_at: str) -> None:
        """Start a successor before the session's cap. It joins with the same identity, so
        LiveKit replaces the old participant and the agent's audio keeps flowing; the
        predecessor is ended once the successor has joined."""
        assert self._client is not None and self._room is not None
        room = self._room
        while expires_at and not self._closed:
            ends = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            remaining = (ends - datetime.now(timezone.utc)).total_seconds()
            await asyncio.sleep(max(_MIN_RENEW_DELAY_S, remaining - _RENEW_BEFORE_S))
            previous = self._session_id
            if self._closed or previous is None or self._end_task is not None:
                return

            current = room.remote_participants.get(self._avatar_identity)
            joined = _ParticipantJoin(
                room, self._avatar_identity, replacing=current.sid if current else None
            )
            successor: dict[str, Any] = {}
            self._renewing = True
            try:
                successor = await self._client.renew_session(
                    previous, {"transport": self._transport(room)}, timeout=self._start_timeout
                )
                await joined.wait(self._join_timeout)
            except Exception:
                logger.warning(
                    "avatar renewal failed; the avatar stops at its cap",
                    extra={"session_id": previous},
                    exc_info=True,
                )
                if successor.get("id"):
                    await self._end_quietly(successor["id"])
                return
            finally:
                joined.close()
                self._renewing = False

            self._session_id = successor.get("id")
            expires_at = successor.get("expires_at") or ""
            logger.debug(
                "avatar session renewed",
                extra={"previous": previous, "session_id": self._session_id},
            )
            await self._end_quietly(previous)

    async def _end_quietly(self, session_id: str) -> None:
        if self._client is None:
            return
        try:
            await self._client.end_session(session_id)
        except Exception:
            logger.debug("ending avatar session failed", extra={"session_id": session_id})

    async def _end_session(self) -> None:
        session_id, self._session_id = self._session_id, None
        if session_id is not None:
            await self._end_quietly(session_id)

    async def aclose(self) -> None:
        """End the avatar, which stops its billing. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._renew_task is not None:
            await utils.aio.cancel_and_wait(self._renew_task)
            self._renew_task = None
        if self._end_task is not None:
            await asyncio.gather(self._end_task, return_exceptions=True)
            self._end_task = None
        if self._room is not None:
            self._room.off("participant_connected", self._on_participant_connected)
            self._room.off("participant_disconnected", self._on_participant_disconnected)
        await self._end_session()
        await super().aclose()


class _ParticipantJoin:
    """Resolves when ``identity`` joins as a different participant than ``replacing``.

    Armed before the successor is requested so a fast join is not missed. Waiting for
    the identity alone would return the old participant at once.
    """

    def __init__(self, room: rtc.Room, identity: str, *, replacing: str | None) -> None:
        self._room = room
        self._identity = identity
        self._replacing = replacing
        self._joined: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        room.on("participant_connected", self._on_connected)

    def _on_connected(self, participant: rtc.RemoteParticipant) -> None:
        if (
            participant.identity == self._identity
            and participant.sid != self._replacing
            and not self._joined.done()
        ):
            self._joined.set_result(None)

    async def wait(self, timeout: float) -> None:
        await asyncio.wait_for(asyncio.shield(self._joined), timeout)

    def close(self) -> None:
        self._room.off("participant_connected", self._on_connected)
        if not self._joined.done():
            self._joined.cancel()
