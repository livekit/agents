from __future__ import annotations

import asyncio
import os

import aiohttp

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    AgentStateChangedEvent,
    APIConnectionError,
    APIStatusError,
    NotGivenOr,
    UserStateChangedEvent,
    get_job_context,
    utils,
)
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import (
    DataStreamAudioOutput,
)

from .log import logger

DEFAULT_API_URL = "https://api.avaluma.ai"
SAMPLE_RATE = 16000


class AvalumaException(Exception):
    """Exception for Avaluma errors"""


class AvatarSession:
    """An Avaluma avatar session"""

    def __init__(
        self,
        *,
        license_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        avatar_server_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        resolved_license_key = license_key or os.getenv("AVALUMA_LICENSE_KEY")
        if not resolved_license_key:
            raise AvalumaException(
                "license_key must be set either by passing license_key to the "
                "client or by setting the AVALUMA_LICENSE_KEY environment variable"
            )

        resolved_avatar_id = avatar_id or os.getenv("AVALUMA_AVATAR_ID")
        if not resolved_avatar_id:
            raise AvalumaException(
                "avatar_id must be set either by passing avatar_id to the client "
                "or by setting the AVALUMA_AVATAR_ID environment variable"
            )

        self._license_key = resolved_license_key
        self._avatar_id = resolved_avatar_id
        self._avatar_server_url = (
            avatar_server_url or os.getenv("AVALUMA_AVATAR_SERVER_URL") or DEFAULT_API_URL
        )
        self._conn_options = DEFAULT_API_CONNECT_OPTIONS

        # the avatar participant identity/name must match what the remote server
        # creates from the avatar_id, so they are not configurable
        self._avatar_participant_identity = f"avatar-{resolved_avatar_id}"
        self._avatar_participant_name = f"avatar-{resolved_avatar_id}"

        self._http_session = utils.http_context.http_session()
        self._room: rtc.Room | None = None
        self._session_id: str | None = None
        # keep strong references to in-flight RPC tasks so they are not
        # garbage-collected mid-execution (see asyncio.create_task docs)
        self._rpc_tasks: set[asyncio.Task] = set()

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise AvalumaException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        self._room = room

        # Get local participant identity
        try:
            job_ctx = get_job_context()
            local_participant_identity = job_ctx.token_claims().identity
        except RuntimeError as e:
            if not room.isconnected():
                raise AvalumaException("failed to get local participant identity") from e
            local_participant_identity = room.local_participant.identity

        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar agent to publish audio and video on behalf of your local agent
            .with_attributes(
                {
                    ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity,
                }
            )
            .to_jwt()
        )

        await self._request_remote_avatar_to_join(livekit_url, livekit_token)

        # Register turn taking event handlers
        self._register_turn_taking_events(agent_session)

        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            ),
        )

    async def _request_remote_avatar_to_join(self, livekit_url: str, livekit_token: str) -> None:
        # Prepare JSON data
        json_data = {
            "livekit_url": livekit_url,
            "livekit_token": livekit_token,
            "avaluma_key": self._license_key,
            "avaluma_avatar_id": self._avatar_id,
        }

        for i in range(self._conn_options.max_retry):
            try:
                async with self._http_session.post(
                    self._avatar_server_url + "/v1/livekit/start-avatar",
                    headers={
                        "Content-Type": "application/json",
                        "api-secret": self._license_key,
                    },
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Server returned an error",
                            status_code=response.status,
                            body=text,
                        )

                    # Try to get session_id from response
                    try:
                        response_data = await response.json()
                        self._session_id = response_data.get("session_id")
                        if self._session_id:
                            logger.debug(f"Remote avatar session started: {self._session_id}")
                    except Exception:
                        # Response might not be JSON, that's ok
                        pass

                    return

            except Exception as e:
                if isinstance(e, APIStatusError) and not e.retryable:
                    raise

                if isinstance(e, APIConnectionError):
                    logger.warning("failed to call avaluma avatar api", extra={"error": str(e)})
                else:
                    logger.exception("failed to call avaluma avatar api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to start Avaluma Avatar Session after all retries")

    def _register_turn_taking_events(self, session: AgentSession) -> None:
        # If the agent or user state changes it sends the new state to the avatar

        @session.on("user_state_changed")
        def on_user_state_changed(ev: UserStateChangedEvent) -> None:
            # States: ["speaking", "listening", "away"]
            self._send_state_rpc("user_state_changed", ev.new_state)

        @session.on("agent_state_changed")
        def on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
            # States: ["initializing", "idle", "listening", "thinking", "speaking"]
            self._send_state_rpc("agent_state_changed", ev.new_state)

    def _send_state_rpc(self, method: str, state: str) -> None:
        task = asyncio.create_task(self._perform_state_rpc(method, state))
        self._rpc_tasks.add(task)
        task.add_done_callback(self._rpc_tasks.discard)

    async def _perform_state_rpc(self, method: str, payload: str) -> None:
        if self._room is None:
            return
        try:
            await self._room.local_participant.perform_rpc(
                destination_identity=self._avatar_participant_identity,
                method=method,
                payload=payload,
            )
        except Exception:
            # The avatar may not have joined yet, or the RPC method may not be
            # registered on the avatar side; these failures are non-fatal.
            logger.debug("failed to send %s RPC to avatar", method, exc_info=True)
