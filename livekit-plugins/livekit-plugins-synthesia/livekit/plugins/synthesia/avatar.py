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
import dataclasses
import enum
import json
import os
from typing import TypeGuard

from livekit import api, rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, AgentSession, get_job_context
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput

from .api import SynthesiaAPI
from .errors import ErrorType, SynthesiaError
from .log import logger
from .types import (
    AVATAR_IDENTITY,
    AVATAR_NAME,
    DEFAULT_API_URL,
    DEFAULT_JOIN_TIMEOUT,
    DEFAULT_SWAP_TIMEOUT,
    TOKEN_TTL,
    AvatarConfig,
    SessionConfig,
    StartSessionRequest,
)


def _present(value: str | None) -> TypeGuard[str]:
    return bool(value and value.strip())


def _to_ws_url(url: str) -> str:
    """Coerce a LiveKit URL to the ws(s):// scheme Synthesia requires; LiveKit
    Cloud injects https:// on deployed agents."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


class _State(enum.Enum):
    IDLE = "idle"
    STARTED = "started"
    CLOSED = "closed"


class AvatarSession(BaseAvatarSession):
    """A Synthesia interactive avatar for a LiveKit voice agent.

    Construct with up to five gallery ``avatar_ids`` and call ``start()`` before
    ``AgentSession.start()``. The first id is the active avatar and the rest are
    precomputed for a future mid-session swap. Credentials fall back to the
    ``SYNTHESIA_API_KEY`` and ``SYNTHESIA_API_URL`` environment variables.

    Logs when the room ends cleanly and when the avatar track drops
    unexpectedly mid-session.

    Pass ``avatar_participant_identity`` to override the LiveKit identity the
    avatar joins under. It must be unique per concurrent avatar in a room, since
    LiveKit evicts an existing participant when a second joins with the same
    identity. It defaults to ``synthesia-avatar-agent``.
    """

    def __init__(
        self,
        avatar_config: AvatarConfig,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        join_timeout: float = DEFAULT_JOIN_TIMEOUT,
        avatar_participant_identity: str | None = None,
        avatar_participant_name: str | None = None,
    ) -> None:
        super().__init__()

        if avatar_participant_identity is not None and not avatar_participant_identity.strip():
            raise SynthesiaError("avatar_participant_identity must be a non-empty string")
        if avatar_participant_name is not None and not avatar_participant_name.strip():
            raise SynthesiaError("avatar_participant_name must be a non-empty string")
        self._avatar_identity = avatar_participant_identity or AVATAR_IDENTITY
        self._avatar_name = avatar_participant_name or AVATAR_NAME

        key = api_key or os.environ.get("SYNTHESIA_API_KEY")
        if not key:
            raise SynthesiaError(
                "a Synthesia API key is required: pass api_key or set SYNTHESIA_API_KEY"
            )

        url = api_url or os.environ.get("SYNTHESIA_API_URL") or DEFAULT_API_URL

        self._config = SessionConfig(
            avatar_ids=tuple(avatar_config.avatar_ids),
            api_key=key,
            api_url=url,
            join_timeout=join_timeout,
        )
        self._state = _State.IDLE
        # Guards against re-entrant start() calls. Distinct from _state: while a
        # start is awaiting a prior teardown the lifecycle is still its old
        # state, so this cannot be folded into _state.
        self._starting = False
        self._close_done = asyncio.Event()
        self._session_id: str | None = None
        self._teardown_task: asyncio.Task[None] | None = None
        # The exact sink this session installed, tracked by identity rather than
        # type or chain position: AgentSession.start() can wrap it in
        # TranscriptSynchronizer/RecorderAudioOutput, and a caller may already
        # have their own DataStreamAudioOutput installed before start() runs.
        self._audio_output: DataStreamAudioOutput | None = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def provider(self) -> str:
        return "synthesia"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
    ) -> None:
        if self._starting:
            raise SynthesiaError("start() is already in progress")
        self._starting = True
        try:
            if self._teardown_task is not None:
                teardown = self._teardown_task
                await asyncio.shield(asyncio.gather(teardown, return_exceptions=True))
                if not teardown.cancelled() and teardown.exception() is not None:
                    raise SynthesiaError(
                        "the previous avatar session did not tear down cleanly; not restarting"
                    ) from teardown.exception()
            if self._state is _State.STARTED:
                return

            url = livekit_url or os.environ.get("LIVEKIT_URL")
            lk_key = livekit_api_key or os.environ.get("LIVEKIT_API_KEY")
            lk_secret = livekit_api_secret or os.environ.get("LIVEKIT_API_SECRET")
            # A blank key or secret still mints a syntactically valid token that
            # Synthesia accepts, since it cannot verify a signature made with the
            # developer's own secret. LiveKit only rejects it once the avatar
            # tries to join, so catch it here instead.
            if not _present(url) or not _present(lk_key) or not _present(lk_secret):
                raise SynthesiaError(
                    "LiveKit url, API key, and API secret are required: pass them or "
                    "set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET"
                )
            url = _to_ws_url(url)
            if not url.startswith(("ws://", "wss://")):
                raise SynthesiaError(f"livekit_url {url!r} is not a ws:// or wss:// URL")

            self._state = _State.IDLE
            self._close_done.clear()
            self._teardown_task = None

            try:
                await super().start(agent_session, room)
                token = self._mint_token(room=room, lk_key=lk_key, lk_secret=lk_secret)
                client = SynthesiaAPI(api_key=self._config.api_key, api_url=self._config.api_url)
                launch_conn = dataclasses.replace(
                    DEFAULT_API_CONNECT_OPTIONS,
                    timeout=self._config.join_timeout,
                    max_retry=0,
                )
                response = await client.start_session(
                    StartSessionRequest(
                        avatar_ids=list(self._config.avatar_ids),
                        livekit_url=url,
                        lk_token=token,
                    ),
                    conn_options=launch_conn,
                )
                self._session_id = response.session_id

                # TODO: confirm the avatar worker's expected audio sample rate and
                # pass sample_rate explicitly once it is verified against the real
                # worker.
                audio_output = DataStreamAudioOutput(
                    room,
                    destination_identity=self.avatar_identity,
                    wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                )
                # replace_audio_tail keeps any wrapper AgentSession.start() adds
                # later (TranscriptSynchronizer, RecorderAudioOutput) attached.
                agent_session.output.replace_audio_tail(audio_output)
                self._audio_output = audio_output

                await self.wait_for_join(timeout=self._config.join_timeout)
            except asyncio.TimeoutError as e:
                await self.aclose()
                await self._discard_partial_start()
                raise SynthesiaError(
                    f"avatar did not join within {self._config.join_timeout}s",
                    type=ErrorType.TIMEOUT,
                ) from e
            except BaseException:
                await self.aclose()
                await self._discard_partial_start()
                raise

            if self._state is _State.CLOSED:
                await self._close_done.wait()
                await self._discard_partial_start()
                raise SynthesiaError("avatar session was closed while starting")

            room.on("disconnected", self._on_room_disconnected)
            room.on("track_unpublished", self._on_track_unpublished)
            room.on("participant_disconnected", self._on_participant_disconnected)
            self._state = _State.STARTED
        finally:
            self._starting = False

    async def _discard_partial_start(self) -> None:
        # An aclose() that ran concurrently with start() cannot see the audio
        # output and session id that start() set after it completed, so a failed
        # start cleans them up itself, closing the audio output's background
        # tasks rather than just dropping the reference. Only the sink this
        # session installed is touched: it may now be wrapped by
        # AgentSession.start() (TranscriptSynchronizer, RecorderAudioOutput), and
        # the caller may have had their own DataStreamAudioOutput before this
        # session ever ran, so neither the current chain head nor its type says
        # what to close.
        audio_output, self._audio_output = self._audio_output, None
        if audio_output is not None and hasattr(audio_output, "aclose"):
            await audio_output.aclose()
        self._session_id = None

    async def swap_avatar(self, avatar_id: str, *, timeout: float = DEFAULT_SWAP_TIMEOUT) -> str:
        """Switch the rendered avatar mid-session.

        ``avatar_id`` must be one of the ids passed in ``AvatarConfig``, or
        ``"default"`` for the first id in that list. Returns the now-active
        avatar id once the swap has taken effect. Raises a ``SynthesiaError``
        of type ``UNKNOWN_AVATAR`` for an id that was not precomputed,
        ``CONNECTION`` if the worker cannot be reached, and no ``type`` if the
        worker rejects the swap.
        """
        if self._state is not _State.STARTED or self._ending() or self._room is None:
            raise SynthesiaError("swap_avatar() requires a started avatar session")
        if avatar_id != "default" and avatar_id not in self._config.avatar_ids:
            raise SynthesiaError(
                f"avatar {avatar_id!r} was not in initial list of avatar_ids",
                type=ErrorType.UNKNOWN_AVATAR,
            )

        try:
            raw = await self._room.local_participant.perform_rpc(
                destination_identity=self.avatar_identity,
                method="swapAvatar",
                payload=json.dumps({"avatar_id": avatar_id}),
                response_timeout=timeout,
            )
        except Exception as e:
            raise SynthesiaError(f"avatar swap RPC failed: {e}", type=ErrorType.CONNECTION) from e

        try:
            response = json.loads(raw)
        except ValueError as e:
            raise SynthesiaError("avatar swap returned a malformed response") from e
        avatar_id_result = response.get("avatar_id") if isinstance(response, dict) else None
        if (
            not isinstance(response, dict)
            or response.get("error")
            or not isinstance(avatar_id_result, str)
        ):
            detail = response.get("error") if isinstance(response, dict) else None
            raise SynthesiaError(f"avatar swap failed: {detail or raw}")
        return avatar_id_result

    def _mint_token(self, *, room: rtc.Room, lk_key: str, lk_secret: str) -> str:
        # Synthesia rejects a token whose publish-on-behalf attribute is empty,
        # since the worker has no agent to publish the avatar's audio for. Read
        # from the job context rather than room.local_participant: the room may
        # not have finished connecting yet when start() runs.
        agent_identity = get_job_context().local_participant_identity
        if not _present(agent_identity):
            raise SynthesiaError(
                "the room's local participant has no identity; connect the room "
                "before starting the avatar session"
            )

        token = (
            api.AccessToken(lk_key, lk_secret)
            .with_identity(self.avatar_identity)
            .with_name(self._avatar_name)
            .with_kind("agent")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room.name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: agent_identity})
            .with_ttl(TOKEN_TTL)
        )
        return token.to_jwt()

    def _on_room_disconnected(self, *args: object) -> None:
        if self._ending():
            return
        logger.info("avatar session ended")
        self._begin_teardown()

    def _on_track_unpublished(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            participant.identity == self.avatar_identity
            and publication.kind == rtc.TrackKind.KIND_VIDEO
        ):
            self._report_avatar_lost()

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self.avatar_identity:
            self._report_avatar_lost()

    def _report_avatar_lost(self) -> None:
        if self._ending():
            return
        # The avatar leaving while the room is still up is a worker crash. If the
        # room itself is going down, this is a clean end and _on_room_disconnected
        # reports it instead.
        if self._room is not None and not self._room.isconnected():
            return
        logger.warning("avatar left the room unexpectedly")
        self._begin_teardown()

    def _ending(self) -> bool:
        return self._state is _State.CLOSED or self._teardown_task is not None

    def _begin_teardown(self) -> None:
        if self._teardown_task is None and self._state is not _State.CLOSED:
            self._teardown_task = asyncio.create_task(self.aclose())
            self._teardown_task.add_done_callback(self._on_teardown_done)

    def _on_teardown_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("avatar teardown failed", exc_info=exc)

    async def aclose(self) -> None:
        if self._state is _State.CLOSED:
            await self._close_done.wait()
            return
        self._state = _State.CLOSED
        self._session_id = None

        try:
            # Only the sink this session installed is closed, by identity: it may
            # now be wrapped by AgentSession.start() (TranscriptSynchronizer,
            # RecorderAudioOutput), and the caller may have had their own
            # DataStreamAudioOutput before this session ever ran, so neither the
            # current chain head nor its type says what to close. The wrapper
            # chain itself is left in place; there is no public API to remove a
            # tail sink from it without supplying a replacement.
            audio_output, self._audio_output = self._audio_output, None
            if audio_output is not None and hasattr(audio_output, "aclose"):
                await audio_output.aclose()

            if self._room is not None:
                self._room.off("disconnected", self._on_room_disconnected)
                self._room.off("track_unpublished", self._on_track_unpublished)
                self._room.off("participant_disconnected", self._on_participant_disconnected)

            await super().aclose()
        finally:
            self._close_done.set()
