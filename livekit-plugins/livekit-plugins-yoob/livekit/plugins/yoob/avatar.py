from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
    utils,
)

from .log import logger

DEFAULT_API_BASE = "https://app.yoob.com"
DEFAULT_CHARACTER = "astrid"

#: The agent's attributes that tell the user's device which character to show.
ATTRIBUTE_CHARACTER = "yoob.avatar.character"
ATTRIBUTE_VERSION = "yoob.avatar.version"
#: The RPC the user's device calls on the agent to open a Yoob avatar session.
RPC_SESSION = "yoob.avatar.session"

# LiveKit RPC answers are limited to 15 KiB.
_MAX_RPC_RESPONSE_BYTES = 15 * 1024


class YoobAvatarException(Exception):
    """The Yoob avatar could not be set up."""


class AvatarSession:
    """A Yoob character as the agent's face, rendered on the user's own device.

    Nothing renders on a server and the room carries no video: the user's app runs a Yoob SDK (web
    ``@yoob/avatar/livekit``, iOS ``YoobLiveKit``, Android ``yoob-avatar-livekit``), which plays the agent's normal
    audio track through the character so the face moves with the voice. This class is the agent's half: it tells the user's device which character to show (participant
    attributes) and opens the Yoob session the device needs to load and meter that character (an RPC the device
    calls), so the Yoob API key stays with the agent and the app needs no backend of its own for it.

    Avatar minutes are billed to the workspace of the Yoob API key, from the device's heartbeats.
    """

    def __init__(
        self,
        *,
        character: str = DEFAULT_CHARACTER,
        version: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_base: NotGivenOr[str] = NOT_GIVEN,
        max_sessions_per_minute: int = 6,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            character: The Yoob character to show, for example ``"astrid"`` or ``"luna-realistic"``.
                The characters are listed at https://app.yoob.com.
            version: Pin a character version. The newest one is used by default.
            api_key: Defaults to the ``YOOB_API_KEY`` environment variable. Keys come from https://app.yoob.com.
            api_base: Defaults to ``YOOB_API_BASE`` or https://app.yoob.com.
            max_sessions_per_minute: Sessions one participant may open per minute. The SDK opens one when the
                character loads and another when it resumes after a long pause, so a few are plenty.
        """
        key = api_key or os.environ.get("YOOB_API_KEY")
        if not key:
            raise YoobAvatarException("Yoob API key is required: pass api_key or set YOOB_API_KEY")
        if not character or not all(c.isalnum() or c == "-" for c in character):
            raise YoobAvatarException(f"invalid Yoob character id: {character!r}")
        self._api_key = key
        self._api_base = (api_base or os.environ.get("YOOB_API_BASE") or DEFAULT_API_BASE).rstrip(
            "/"
        )
        self._character = character
        self._version = version
        self._max_per_minute = max(1, max_sessions_per_minute)
        self._conn_options = conn_options
        self._room: rtc.Room | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._recent: defaultdict[str, deque[float]] = defaultdict(deque)
        self._loaded = asyncio.Event()

    @property
    def provider(self) -> str:
        return "yoob"

    @property
    def character(self) -> str:
        return self._character

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        """Announce the character and answer the user's device. Call it after the room is connected.

        ``agent_session`` is not changed: the agent keeps publishing its own audio track, which the user's device
        plays through the character.
        """
        if not room.isconnected():
            raise YoobAvatarException("connect the room first: await ctx.connect()")
        self._room = room
        attributes = {ATTRIBUTE_CHARACTER: self._character}
        if utils.is_given(self._version):
            attributes[ATTRIBUTE_VERSION] = self._version
        room.local_participant.register_rpc_method(RPC_SESSION, self._on_session_request)
        await room.local_participant.set_attributes(attributes)
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            job_ctx.add_shutdown_callback(self.aclose)
        logger.debug("yoob avatar ready", extra={"character": self._character})

    async def wait_for_join(self, *, timeout: float | None = 30.0) -> None:
        """Wait until a user's device has opened its Yoob session, that is, until the character is loading there.

        Raises ``asyncio.TimeoutError`` if no device asks within ``timeout`` seconds (``None`` waits forever).
        """
        await asyncio.wait_for(self._loaded.wait(), timeout=timeout)

    async def aclose(self) -> None:
        room, self._room = self._room, None
        if room is None:
            return
        room.local_participant.unregister_rpc_method(RPC_SESSION)
        if room.isconnected():
            try:
                await room.local_participant.set_attributes(
                    {ATTRIBUTE_CHARACTER: "", ATTRIBUTE_VERSION: ""}
                )
            except Exception:
                logger.debug("failed to clear the yoob avatar attributes", exc_info=True)

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    async def _on_session_request(self, data: rtc.RpcInvocationData) -> str:
        """Open a Yoob avatar session for the participant that asked, and answer with its credentials."""
        room = self._room
        caller = room.remote_participants.get(data.caller_identity) if room else None
        if caller is None or caller.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            # only the people in the call get sessions; agents and avatar workers never need one
            raise rtc.RpcError(rtc.RpcError.ErrorCode.APPLICATION_ERROR, "not allowed")

        now = time.monotonic()
        recent = self._recent[data.caller_identity]
        while recent and now - recent[0] > 60:
            recent.popleft()
        if len(recent) >= self._max_per_minute:
            raise rtc.RpcError(
                rtc.RpcError.ErrorCode.APPLICATION_ERROR, "too many Yoob sessions, try again later"
            )
        recent.append(now)

        status, body = await self._open_session()
        if status >= 400:
            logger.warning(
                "yoob avatar session refused",
                extra={"status": status, "character": self._character},
            )
            raise rtc.RpcError(
                rtc.RpcError.ErrorCode.APPLICATION_ERROR, "Yoob session refused", body[:1000]
            )
        if len(body.encode()) > _MAX_RPC_RESPONSE_BYTES:
            raise rtc.RpcError(
                rtc.RpcError.ErrorCode.APPLICATION_ERROR, "Yoob session too large for an RPC answer"
            )
        self._loaded.set()
        return body

    async def _open_session(self) -> tuple[int, str]:
        """POST /api/v1/avatar/sessions with the agent's key: (status, body). Retries network errors and 5xx."""
        session = self._ensure_http_session()
        payload: dict[str, object] = {"characters": [self._character]}
        attempts = self._conn_options.max_retry + 1
        for attempt in range(attempts):
            try:
                async with session.post(
                    f"{self._api_base}/api/v1/avatar/sessions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        sock_connect=self._conn_options.timeout,
                        sock_read=self._conn_options.timeout,
                    ),
                ) as response:
                    body = await response.text()
                    if response.status < 500 or attempt + 1 >= attempts:
                        if response.status < 400:
                            try:
                                json.loads(body)
                            except ValueError as e:
                                # the session may exist already: no retry, which could open a second one
                                raise rtc.RpcError(
                                    rtc.RpcError.ErrorCode.APPLICATION_ERROR,
                                    "Yoob returned an invalid session",
                                ) from e
                        return response.status, body
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt + 1 >= attempts:
                    logger.warning("failed to reach the yoob avatar api", exc_info=e)
                    raise rtc.RpcError(
                        rtc.RpcError.ErrorCode.APPLICATION_ERROR, "Yoob could not be reached"
                    ) from e
            await asyncio.sleep(self._conn_options.retry_interval)
        raise rtc.RpcError(rtc.RpcError.ErrorCode.APPLICATION_ERROR, "Yoob could not be reached")
