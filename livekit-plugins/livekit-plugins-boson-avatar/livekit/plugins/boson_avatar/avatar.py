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
from datetime import timedelta
from typing import Any

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
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .api import AvatarSessionInfo, BosonAvatarAPI
from .errors import BosonAvatarException
from .log import logger

SAMPLE_RATE = 24000
MAX_DURATION_SECONDS = 14_400
_AVATAR_AGENT_IDENTITY = "boson-avatar-agent"
_AVATAR_AGENT_NAME = "Boson Avatar"


class AvatarSession(BaseAvatarSession[Any]):
    """A provider-agnostic audio session for Boson Higgs Avatar rendering."""

    def __init__(
        self,
        *,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        width: NotGivenOr[int] = NOT_GIVEN,
        height: NotGivenOr[int] = NOT_GIVEN,
        max_duration_seconds: NotGivenOr[int] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Boson Avatar session.

        Args:
            avatar_id: Boson Avatar asset ID. Defaults to `BOSON_AVATAR_ID`.
            api_key: Boson API key. Defaults to `BOSON_API_KEY`.
            api_url: Boson Avatar API URL. Defaults to `BOSON_AVATAR_API_URL`
                or the production Boson API.
            width: Optional output width. Must be provided with `height`.
            height: Optional output height. Must be provided with `width`.
            max_duration_seconds: Optional maximum Avatar session duration.
            avatar_participant_identity: LiveKit identity used by the Avatar.
            avatar_participant_name: LiveKit display name used by the Avatar.
            conn_options: Timeout and retry options for Boson API requests.

        Raises:
            BosonAvatarException: If required configuration is missing or invalid.
        """
        super().__init__()
        resolved_avatar_id = _resolve_env_or_value(avatar_id, "BOSON_AVATAR_ID")
        if not resolved_avatar_id:
            raise BosonAvatarException(
                "avatar_id must be set by passing it to AvatarSession or setting "
                "the BOSON_AVATAR_ID environment variable"
            )
        self._avatar_id = resolved_avatar_id

        self._width = _resolve_optional_positive_int(width, "width")
        self._height = _resolve_optional_positive_int(height, "height")
        if (self._width is None) != (self._height is None):
            raise BosonAvatarException("width and height must be provided together")
        self._max_duration_seconds = _resolve_optional_positive_int(
            max_duration_seconds,
            "max_duration_seconds",
            maximum=MAX_DURATION_SECONDS,
        )

        self._avatar_identity = _resolve_optional_string(
            avatar_participant_identity, _AVATAR_AGENT_IDENTITY
        )
        self._avatar_name = _resolve_optional_string(avatar_participant_name, _AVATAR_AGENT_NAME)
        self._api = BosonAvatarAPI(
            api_key=api_key,
            api_url=api_url,
            conn_options=conn_options,
        )
        self._session_info: AvatarSessionInfo | None = None
        self._start_called = False
        self._closed = False
        self._close_requested = False
        self._lifecycle_lock = asyncio.Lock()
        self._create_task: asyncio.Task[AvatarSessionInfo] | None = None
        self._startup_cleanup_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._agent_close_task: asyncio.Task[None] | None = None
        self._tracked_agent_session: AgentSession[Any] | None = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def provider(self) -> str:
        return "boson"

    @property
    def session_id(self) -> str | None:
        """Boson Avatar session ID after `start()` succeeds, otherwise `None`."""
        return self._session_info.id if self._session_info is not None else None

    async def start(  # type: ignore[override]
        self,
        agent_session: AgentSession[Any],
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> str:
        """Start the Avatar participant and route generic AgentSession audio to it."""
        if self._start_called:
            raise RuntimeError("AvatarSession.start() called twice; create a new AvatarSession.")
        self._start_called = True

        livekit_url_value = _resolve_env_or_value(livekit_url, "LIVEKIT_URL")
        livekit_api_key_value = _resolve_env_or_value(livekit_api_key, "LIVEKIT_API_KEY")
        livekit_api_secret_value = _resolve_env_or_value(livekit_api_secret, "LIVEKIT_API_SECRET")
        if not livekit_url_value or not livekit_api_key_value or not livekit_api_secret_value:
            raise BosonAvatarException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        publisher_identity = _local_participant_identity(room)
        livekit_token = self._mint_avatar_token(
            room=room,
            publisher_identity=publisher_identity,
            livekit_api_key=livekit_api_key_value,
            livekit_api_secret=livekit_api_secret_value,
        )

        async with self._lifecycle_lock:
            if self._close_requested or self._closed:
                raise RuntimeError("AvatarSession was closed before start() completed")

            await super().start(agent_session, room)
            self._tracked_agent_session = agent_session
            agent_session.on("close", self._on_agent_session_close)
            started_info: AvatarSessionInfo | None = None
            try:
                self._create_task = asyncio.create_task(
                    self._api.start_session(
                        avatar_id=self._avatar_id,
                        livekit_url=livekit_url_value,
                        livekit_room=room.name,
                        livekit_token=livekit_token,
                        avatar_identity=self._avatar_identity,
                        publisher_identity=publisher_identity,
                        width=self._width,
                        height=self._height,
                        max_duration_seconds=self._max_duration_seconds,
                    ),
                    name="boson_avatar_create_session",
                )
                # The provider create is shielded so cancellation cannot discard a
                # successful response before its session ID can be compensated.
                started_info = await asyncio.shield(self._create_task)
                self._session_info = started_info

                if self._close_requested:
                    raise RuntimeError("AvatarSession was closed while start() was in progress")

                audio_output = DataStreamAudioOutput(
                    room=room,
                    destination_identity=self._avatar_identity,
                    sample_rate=SAMPLE_RATE,
                    # Higgs publishes audio before its first generated video frame.
                    # Waiting on audio avoids a first-frame dependency cycle while
                    # still buffering speech until the Avatar is ready.
                    wait_remote_track=rtc.TrackKind.KIND_AUDIO,
                )
                replace_audio_tail = getattr(agent_session.output, "replace_audio_tail", None)
                if callable(replace_audio_tail):
                    replace_audio_tail(audio_output)
                else:  # livekit-agents 1.5 compatibility for the experiment image
                    agent_session.output.audio = audio_output
            except BaseException:
                # Move all startup cleanup into an independent task. It owns both
                # the provider result and DELETE, so repeated caller cancellation
                # cannot interrupt compensation or base-session cleanup.
                self._startup_cleanup_task = asyncio.create_task(
                    self._cleanup_failed_start(
                        create_task=self._create_task if started_info is None else None,
                        session_info=started_info,
                    ),
                    name="boson_avatar_startup_cleanup",
                )
                self._startup_cleanup_task.add_done_callback(self._consume_startup_cleanup_result)
                try:
                    await asyncio.shield(self._startup_cleanup_task)
                except BaseException:
                    # The owned task remains alive and completes independently.
                    pass
                raise
            finally:
                if self._startup_cleanup_task is None:
                    self._create_task = None

        logger.debug(
            "boson avatar session started",
            extra={"session_id": started_info.id, "avatar_id": self._avatar_id},
        )
        return started_info.id

    async def aclose(self) -> None:
        """End the hosted Avatar session and remove its LiveKit participant."""
        # Set this before acquiring the lock so an in-flight start knows that
        # close won the race and compensates the provider session it just created.
        self._close_requested = True
        while True:
            owned_task: asyncio.Task[None] | None
            is_startup_cleanup = False
            async with self._lifecycle_lock:
                if self._startup_cleanup_task is not None:
                    owned_task = self._startup_cleanup_task
                    is_startup_cleanup = True
                elif self._shutdown_task is not None:
                    owned_task = self._shutdown_task
                elif self._closed and self._session_info is None:
                    return
                else:
                    self._shutdown_task = asyncio.create_task(
                        self._run_shutdown(), name="boson_avatar_shutdown"
                    )
                    self._shutdown_task.add_done_callback(self._consume_shutdown_result)
                    owned_task = self._shutdown_task

            await asyncio.shield(owned_task)
            if not is_startup_cleanup:
                return
            # Startup cleanup can retain a session ID when provider DELETE
            # exhausts its retries. Re-evaluate once so this close call can retry.

    async def _cleanup_failed_start(
        self,
        *,
        create_task: asyncio.Task[AvatarSessionInfo] | None,
        session_info: AvatarSessionInfo | None,
    ) -> None:
        try:
            if session_info is None and create_task is not None:
                try:
                    session_info = await create_task
                except Exception:
                    session_info = None
            if session_info is not None:
                self._session_info = session_info
                await self._compensate_start(session_info)
        finally:
            if create_task is not None and self._create_task is create_task:
                self._create_task = None
            self._detach_agent_close_listener()
            try:
                await super().aclose()
            finally:
                self._closed = True

    async def _run_shutdown(self) -> None:
        session_info = self._session_info
        if session_info is not None:
            try:
                await self._api.end_session(session_info.id)
            except Exception:  # noqa: BLE001 - a later aclose() can retry by ID
                logger.warning(
                    "failed to end boson avatar session",
                    extra={"session_id": session_info.id},
                    exc_info=True,
                )
            else:
                if self._session_info is session_info:
                    self._session_info = None

        if not self._closed:
            self._detach_agent_close_listener()
            try:
                await super().aclose()
            finally:
                self._closed = True

    def _consume_startup_cleanup_result(self, task: asyncio.Task[None]) -> None:
        if self._startup_cleanup_task is task:
            self._startup_cleanup_task = None
        self._consume_background_task_result(
            task, "failed to compensate cancelled boson avatar startup"
        )

    def _consume_shutdown_result(self, task: asyncio.Task[None]) -> None:
        if self._shutdown_task is task:
            self._shutdown_task = None
        self._consume_background_task_result(task, "failed to close boson avatar session")

    async def _compensate_start(self, session_info: AvatarSessionInfo) -> None:
        try:
            await self._api.end_session(session_info.id)
        except Exception:  # noqa: BLE001 - startup compensation is best-effort
            logger.warning(
                "failed to compensate boson avatar session after startup error",
                extra={"session_id": session_info.id},
                exc_info=True,
            )
        else:
            if self._session_info is session_info:
                self._session_info = None

    def _on_agent_session_close(self, _: Any) -> None:
        self._close_requested = True
        if (self._closed and self._session_info is None) or self._agent_close_task is not None:
            return
        self._agent_close_task = asyncio.create_task(
            self.aclose(), name="boson_avatar_agent_session_close"
        )
        self._agent_close_task.add_done_callback(self._consume_agent_close_result)

    def _consume_agent_close_result(self, task: asyncio.Task[None]) -> None:
        self._consume_background_task_result(
            task, "failed to close boson avatar after AgentSession closed"
        )

    @staticmethod
    def _consume_background_task_result(task: asyncio.Task[None], message: str) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                message,
                exc_info=(type(error), error, error.__traceback__),
            )

    def _detach_agent_close_listener(self) -> None:
        if self._tracked_agent_session is not None:
            self._tracked_agent_session.off("close", self._on_agent_session_close)
            self._tracked_agent_session = None

    def _mint_avatar_token(
        self,
        *,
        room: rtc.Room,
        publisher_identity: str,
        livekit_api_key: str,
        livekit_api_secret: str,
    ) -> str:
        grants = api.VideoGrants(
            room_join=True,
            room=room.name,
            can_subscribe=False,
            can_publish_data=True,
            can_publish_sources=["camera", "microphone"],
        )
        ttl = timedelta(
            seconds=(self._max_duration_seconds + 300)
            if self._max_duration_seconds is not None
            else 4 * 60 * 60
        )
        return (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_identity)
            .with_name(self._avatar_name)
            .with_grants(grants)
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: publisher_identity})
            .with_ttl(ttl)
            .to_jwt()
        )


def _local_participant_identity(room: rtc.Room) -> str:
    job_ctx = get_job_context(required=False)
    if job_ctx is not None and job_ctx.local_participant_identity:
        return job_ctx.local_participant_identity
    if room.isconnected() and room.local_participant.identity:
        return room.local_participant.identity
    raise BosonAvatarException("failed to get the local LiveKit participant identity")


def _resolve_env_or_value(value: NotGivenOr[str], env_name: str) -> str | None:
    if utils.is_given(value) and value:
        return str(value).strip() or None
    env_value = os.getenv(env_name)
    return env_value.strip() if env_value and env_value.strip() else None


def _resolve_optional_string(value: NotGivenOr[str], default: str) -> str:
    if utils.is_given(value) and value:
        return str(value).strip() or default
    return default


def _resolve_optional_positive_int(
    value: NotGivenOr[int],
    name: str,
    *,
    maximum: int | None = None,
) -> int | None:
    if not utils.is_given(value) or value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise BosonAvatarException(f"{name} must be a positive integer")
    if value <= 0:
        raise BosonAvatarException(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise BosonAvatarException(f"{name} must be between 1 and {maximum}")
    return value


__all__ = ["MAX_DURATION_SECONDS", "SAMPLE_RATE", "AvatarSession"]
