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
import os
import uuid
from datetime import timedelta

from pydantic import JsonValue

import livekit.rtc as rtc
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import (
    AvatarSession as BaseAvatarSession,
    AvatarStart,
    DataStreamAudioOutput,
)

from .api import BosonAvatarAPI
from .errors import BosonAvatarException
from .log import logger

MAX_DURATION_SECONDS = 14_400
_SESSION_OBJECT = "avatar.livekit.session"
# the session statuses known not to stream yet or anymore: a created session in one is refused
_NOT_LIVE = frozenset({"pending", "terminated", "failed", "error"})
# the ones known to still run: a delete answered with one did not end the session; any other
# status (one Boson added since) reads as the usual case
_LIVE = frozenset({"active", "pending"})


class AvatarSession(BaseAvatarSession):
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
        idempotency_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Boson Avatar session.

        Args:
            avatar_id: Boson Avatar asset ID. Defaults to `BOSON_AVATAR_ID`.
            api_key: Boson API key. Defaults to `BOSON_API_KEY`.
            api_url: Avatar session service URL supplied by the application.
                Defaults to `BOSON_AVATAR_API_URL`; there is no hard-coded
                production endpoint.
            width: Optional output width. Must be provided with `height`.
            height: Optional output height. Must be provided with `width`.
            max_duration_seconds: Optional maximum Avatar session duration.
            avatar_participant_identity: LiveKit identity used by the Avatar.
                Must differ from the local agent participant identity.
            avatar_participant_name: LiveKit display name used by the Avatar.
            idempotency_key: Optional stable key for provider-session creation.
                Must be a UUID when provided. Inside an Agent job, the default
                is a UUID derived from the job ID and immutable Avatar session
                binding, so a redelivered job recovers the provider session.
            livekit_url: LiveKit server URL. Defaults to `LIVEKIT_URL`.
            livekit_api_key: LiveKit API key. Defaults to `LIVEKIT_API_KEY`.
            livekit_api_secret: LiveKit API secret. Defaults to `LIVEKIT_API_SECRET`.
            conn_options: Timeout and retry options for Boson API requests.

        Raises:
            BosonAvatarException: If required configuration is missing or invalid.
        """
        super().__init__(
            avatar_participant_identity=_resolve_optional_string(
                avatar_participant_identity, "boson-avatar-agent"
            ),
            avatar_participant_name=_resolve_optional_string(
                avatar_participant_name, "Boson Avatar"
            ),
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
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
        self._idempotency_key = _resolve_optional_idempotency_key(idempotency_key)
        self._api = BosonAvatarAPI(api_key=api_key, api_url=api_url, conn_options=conn_options)
        # one key per start, shared by its retries, so Boson allocates one session
        self._attempts_of: AvatarStart | None = None
        self._attempt_key = ""
        self._session_id: str | None = None

    @property
    def provider(self) -> str:
        return "boson"

    @property
    def session_id(self) -> str | None:
        """Boson Avatar session ID after `start()` succeeds, otherwise `None`."""
        return self._session_id

    async def create_session(self, ctx: AvatarStart) -> str:
        if ctx.avatar_identity == ctx.agent_identity:
            raise APIError(
                "avatar_participant_identity must differ from the local agent participant identity",
                kind="invalid_request",
            )
        if self._attempts_of is not ctx:
            self._attempts_of = ctx
            self._attempt_key = (
                self._idempotency_key
                or _livekit_job_idempotency_key(
                    livekit_url=ctx.livekit_url,
                    room_name=ctx.room.name,
                    avatar_id=self._avatar_id,
                    avatar_identity=ctx.avatar_identity,
                    publisher_identity=ctx.agent_identity,
                    width=self._width,
                    height=self._height,
                    max_duration_seconds=self._max_duration_seconds,
                )
                or str(uuid.uuid4())
            )
        grants = api.VideoGrants(
            room_join=True,
            room=ctx.room.name,
            can_subscribe=False,
            can_publish_data=True,
            can_publish_sources=["camera", "microphone"],
        )
        ttl = timedelta(
            seconds=self._max_duration_seconds + 300 if self._max_duration_seconds else 4 * 3600
        )
        body: dict[str, JsonValue] = {
            "avatar_id": self._avatar_id,
            "transport": {
                "type": "livekit",
                "url": ctx.livekit_url,
                "room_name": ctx.room.name,
                "participant_token": await ctx.livekit_token(ttl=ttl, grants=grants),
                "participant_identity": ctx.avatar_identity,
                "publisher_identity": ctx.agent_identity,
                "audio_source": "data_stream",
            },
        }
        if self._width is not None and self._height is not None:
            body["output"] = {"width": self._width, "height": self._height}
        if self._max_duration_seconds is not None:
            body["max_duration_seconds"] = self._max_duration_seconds

        _, data = await self._api.request(
            "POST",
            "/sessions",
            json=body,
            headers={"Idempotency-Key": self._attempt_key},
            success_statuses=frozenset({200, 201}),
        )
        session_id = data.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise APIError("Boson Avatar API response is missing a session id", kind="protocol")
        if (
            data.get("object") != _SESSION_OBJECT
            or data.get("status") in _NOT_LIVE
            or data.get("avatar_identity") != ctx.avatar_identity
        ):
            # the answer still names an allocated session: end it before failing
            try:
                await self.end_session(session_id)
            except Exception as e:
                logger.warning(
                    "failed to end an invalid boson avatar session",
                    extra={"error_type": type(e).__name__, "lk.pii.session_id": session_id},
                )
            raise APIError("Boson Avatar API returned an invalid active session", kind="protocol")
        self._session_id = session_id
        logger.debug(
            "boson avatar session started",
            extra={"lk.pii.session_id": session_id, "lk.pii.avatar_id": self._avatar_id},
        )
        return session_id

    async def end_session(self, session_id: str) -> None:
        """Idempotently stop the hosted Boson Avatar session."""
        status, data = await self._api.request(
            "DELETE",
            f"/sessions/{session_id}",
            allow_empty=True,
            success_statuses=frozenset({200, 204}),
        )
        if (status == 204 and data) or (
            status == 200
            and (
                data.get("id") != session_id
                or data.get("object") != _SESSION_OBJECT
                or data.get("status") in _LIVE
            )
        ):
            raise APIError(
                "Boson Avatar API returned an invalid terminated session", kind="protocol"
            )
        if self._session_id == session_id:
            self._session_id = None

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput:
        return DataStreamAudioOutput(
            room=ctx.room,
            destination_identity=self.avatar_identity,
            sample_rate=self.sample_rate,
            # Higgs publishes audio before its first generated video frame. Waiting on audio
            # avoids a first-frame dependency cycle while still buffering speech until the
            # Avatar is ready.
            wait_remote_track=rtc.TrackKind.KIND_AUDIO,
        )


def _livekit_job_idempotency_key(
    *,
    livekit_url: str,
    room_name: str,
    avatar_id: str,
    avatar_identity: str,
    publisher_identity: str,
    width: int | None,
    height: int | None,
    max_duration_seconds: int | None,
) -> str | None:
    job_ctx = get_job_context(required=False)
    if job_ctx is None:
        return None
    job_id = str(job_ctx.job.id).strip()
    if not job_id:
        return None
    intent = json.dumps(
        {
            "job_id": job_id,
            "livekit_url": livekit_url,
            "room_name": room_name,
            "avatar_id": avatar_id,
            "avatar_identity": avatar_identity,
            "publisher_identity": publisher_identity,
            "width": width,
            "height": height,
            "max_duration_seconds": max_duration_seconds,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"boson-avatar-livekit:{intent}"))


def _resolve_env_or_value(value: NotGivenOr[str], env_name: str) -> str | None:
    if utils.is_given(value) and value:
        return str(value).strip() or None
    env_value = os.getenv(env_name)
    return env_value.strip() if env_value and env_value.strip() else None


def _resolve_optional_string(value: NotGivenOr[str], default: str) -> str:
    if utils.is_given(value) and value:
        return str(value).strip() or default
    return default


def _resolve_optional_idempotency_key(value: object) -> str | None:
    if not utils.is_given(value) or value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BosonAvatarException("idempotency_key must be a UUID string")
    try:
        return str(uuid.UUID(value.strip()))
    except ValueError:
        raise BosonAvatarException("idempotency_key must be a UUID string") from None


def _resolve_optional_positive_int(
    value: object,
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
