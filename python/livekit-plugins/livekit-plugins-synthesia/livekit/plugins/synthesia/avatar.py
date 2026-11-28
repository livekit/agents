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

from pydantic import JsonValue, TypeAdapter

from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    NotGivenOr,
    http,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart

from .api import SESSION_PATH, parse_json, public_avatar_id, session_error
from .errors import ErrorType, SynthesiaError
from .types import (
    AVATAR_IDENTITY,
    AVATAR_NAME,
    DEFAULT_API_URL,
    DEFAULT_SWAP_TIMEOUT,
    TOKEN_TTL,
    AvatarConfig,
)

_JSON_VALUE = TypeAdapter[JsonValue](JsonValue)


def _to_ws_url(url: str) -> str:
    """Coerce a LiveKit URL to the ws(s):// scheme Synthesia requires; LiveKit
    Cloud injects https:// on deployed agents."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


class AvatarSession(BaseAvatarSession):
    """A Synthesia interactive avatar for a LiveKit voice agent.

    Construct with up to five gallery ``avatar_ids`` and call ``start()`` before
    ``VoiceSession.start()``. The first id is the active avatar and the rest are
    precomputed for a future mid-session swap. Credentials fall back to the
    ``SYNTHESIA_API_KEY`` and ``SYNTHESIA_API_URL`` environment variables.

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
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            avatar_participant_identity=avatar_participant_identity or AVATAR_IDENTITY,
            avatar_participant_name=avatar_participant_name or AVATAR_NAME,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        key = api_key or os.environ.get("SYNTHESIA_API_KEY")
        if not key:
            raise SynthesiaError(
                "a Synthesia API key is required: pass api_key or set SYNTHESIA_API_KEY"
            )
        self._api_key = key
        self._api_url = (api_url or os.environ.get("SYNTHESIA_API_URL") or DEFAULT_API_URL).rstrip(
            "/"
        )
        self._avatar_ids = tuple(avatar_config.avatar_ids)
        self._session_id: str | None = None

    @property
    def provider(self) -> str:
        return "synthesia"

    async def create_session(self, ctx: AvatarStart) -> str:
        livekit_url = _to_ws_url(ctx.livekit_url)
        if not livekit_url.startswith(("ws://", "wss://")):
            raise SynthesiaError(f"livekit_url {livekit_url!r} is not a ws:// or wss:// URL")
        grants = api.VideoGrants(
            room_join=True,
            room=ctx.room.name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        async with http.http_session().post(
            self._api_url + SESSION_PATH,
            headers={"Authorization": self._api_key},
            json={
                "avatarIds": [public_avatar_id(a) for a in self._avatar_ids],
                "livekitUrl": livekit_url,
                "livekitToken": await ctx.livekit_token(ttl=TOKEN_TTL, grants=grants),
            },
            conn_options=ctx.conn_options,
        ) as resp:
            if resp.status >= 500:
                await resp.raise_for_status()
            body = parse_json(await resp.read())
            if not resp.ok:
                raise session_error(resp, body)
        session_id = body.get("id") or body.get("session_id") if isinstance(body, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise APIError("Synthesia response did not contain a session id", kind="protocol")
        self._session_id = session_id
        return session_id

    async def swap_avatar(self, avatar_id: str, *, timeout: float = DEFAULT_SWAP_TIMEOUT) -> str:
        """Switch the rendered avatar mid-session.

        ``avatar_id`` must be one of the ids passed in ``AvatarConfig``, or
        ``"default"`` for the first id in that list. Returns the now-active
        avatar id once the swap has taken effect. Raises a ``SynthesiaError``
        of type ``UNKNOWN_AVATAR`` for an id that was not precomputed,
        ``CONNECTION`` if the worker cannot be reached, and no ``type`` if the
        worker rejects the swap.
        """
        if self._session_id is None or self._room is None:
            raise SynthesiaError("swap_avatar() requires a started avatar session")
        if avatar_id != "default" and avatar_id not in self._avatar_ids:
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
            response = _JSON_VALUE.validate_json(raw)
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

    async def aclose(self) -> None:
        self._session_id = None
        await super().aclose()
