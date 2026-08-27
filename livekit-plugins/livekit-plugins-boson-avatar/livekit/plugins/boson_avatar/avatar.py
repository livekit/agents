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
_AVATAR_AGENT_IDENTITY = "boson-avatar-agent"
_AVATAR_AGENT_NAME = "Boson Avatar"


class AvatarSession(BaseAvatarSession[Any]):
    """A Boson Higgs Audio-Driven Avatar session for LiveKit Agents."""

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
            max_duration_seconds, "max_duration_seconds"
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
        """Start the Avatar participant and route AgentSession audio to it."""
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

        await super().start(agent_session, room)
        started_info: AvatarSessionInfo | None = None
        try:
            started_info = await self._api.start_session(
                avatar_id=self._avatar_id,
                livekit_url=livekit_url_value,
                livekit_room=room.name,
                livekit_token=livekit_token,
                avatar_identity=self._avatar_identity,
                publisher_identity=publisher_identity,
                width=self._width,
                height=self._height,
                max_duration_seconds=self._max_duration_seconds,
            )
            self._session_info = started_info

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
            if started_info is not None:
                try:
                    await self._api.end_session(started_info.id)
                except Exception:  # noqa: BLE001 - startup compensation is best-effort
                    logger.warning(
                        "failed to compensate boson avatar session after startup error",
                        extra={"session_id": started_info.id},
                        exc_info=True,
                    )
                self._session_info = None
            try:
                await super().aclose()
            finally:
                self._closed = True
            raise

        logger.debug(
            "boson avatar session started",
            extra={"session_id": started_info.id, "avatar_id": self._avatar_id},
        )
        return started_info.id

    async def aclose(self) -> None:
        """End the hosted Avatar session and remove its LiveKit participant."""
        if self._closed:
            return
        self._closed = True
        session_info = self._session_info
        self._session_info = None
        try:
            if session_info is not None:
                try:
                    await self._api.end_session(session_info.id)
                except Exception:  # noqa: BLE001 - shutdown cleanup is best-effort
                    logger.warning(
                        "failed to end boson avatar session",
                        extra={"session_id": session_info.id},
                        exc_info=True,
                    )
        finally:
            await super().aclose()

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
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
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


def _resolve_optional_positive_int(value: NotGivenOr[int], name: str) -> int | None:
    if not utils.is_given(value) or value is None:
        return None
    if isinstance(value, bool):
        raise BosonAvatarException(f"{name} must be a positive integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise BosonAvatarException(f"{name} must be a positive integer") from exc
    if resolved <= 0:
        raise BosonAvatarException(f"{name} must be a positive integer")
    return resolved


__all__ = ["SAMPLE_RATE", "AvatarSession"]
