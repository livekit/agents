from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Any

import aiohttp
from PIL import Image

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    utils,
)

from .log import logger
from .meeting.room import JoinMeetingResult


class LemonSliceException(Exception):
    """Exception for LemonSlice errors"""


DEFAULT_API_URL = "https://lemonslice.com/api/liveai/sessions"


class LemonSliceAPI:
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initializes the LemonSliceAPI client.

        Args:
            api_key: Your LemonSlice API key. If not provided, it is read from
                    the LEMONSLICE_API_KEY environment variable.
            api_url: The base URL of the LemonSlice API.
            conn_options: Connection options for the aiohttp session.
            session: An optional existing aiohttp.ClientSession to use for requests.
        """
        ls_api_key = api_key if utils.is_given(api_key) else os.getenv("LEMONSLICE_API_KEY")
        if not ls_api_key:
            raise LemonSliceException("LEMONSLICE_API_KEY must be set")
        self._api_key = ls_api_key

        self._api_url = api_url or DEFAULT_API_URL
        self._conn_options = conn_options
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> LemonSliceAPI:
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def start_agent_session(
        self,
        *,
        livekit_url: str,
        livekit_token: str,
        livekit_session_id: str,
        agent_id: NotGivenOr[str] = NOT_GIVEN,
        agent_image_url: NotGivenOr[str] = NOT_GIVEN,
        agent_image: NotGivenOr[Image.Image] = NOT_GIVEN,
        agent_prompt: NotGivenOr[str] = NOT_GIVEN,
        agent_idle_prompt: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout: NotGivenOr[int] = NOT_GIVEN,
        extra_payload: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> str:
        """
        Initiates a new LemonSlice agent session.

        Args:
            livekit_url: The LiveKit Cloud server URL.
            livekit_token: The LiveKit access token for the agent.
            livekit_session_id: LiveKit room session ID (room SID).
            agent_id: The ID of the LemonSlice agent to add to the session.
            agent_image_url: The URL of the image to use as the agent's avatar.
            agent_image: A PIL image to upload and use as the agent's avatar. Sent to
                    LemonSlice as a multipart image upload.
            agent_prompt: A prompt that subtly influences the avatar's movements and expressions while responding.
            agent_idle_prompt: A prompt that subtly influences the avatar's movements and expressions while idle.
            idle_timeout: The idle timeout, in seconds.
            extra_payload: Additional payload to include in the request.

        Returns:
            The unique session ID for the LemonSlice agent session.
        """
        given_sources = [
            source for source in (agent_id, agent_image_url, agent_image) if utils.is_given(source)
        ]
        if len(given_sources) == 0:
            raise LemonSliceException("Missing one of agent_id, agent_image_url or agent_image")
        if len(given_sources) > 1:
            raise LemonSliceException(
                "Only one of agent_id, agent_image_url or agent_image can be provided"
            )

        payload: dict[str, Any] = {
            "transport_type": "livekit",
            "properties": {
                "livekit_url": livekit_url,
                "livekit_token": livekit_token,
                "livekit_session_id": livekit_session_id,
            },
        }

        image_bytes: bytes | None = None

        if utils.is_given(agent_id):
            payload["agent_id"] = agent_id
        if utils.is_given(agent_image_url):
            payload["agent_image_url"] = agent_image_url
        if utils.is_given(agent_prompt):
            payload["agent_prompt"] = agent_prompt
        if utils.is_given(agent_idle_prompt):
            payload["agent_idle_prompt"] = agent_idle_prompt
        if utils.is_given(idle_timeout):
            payload["idle_timeout"] = idle_timeout
        if utils.is_given(extra_payload):
            payload.update(extra_payload)
        if utils.is_given(agent_image):
            image_bytes = _encode_image(agent_image)

        response_data = await self._post(payload, image_bytes=image_bytes)
        session_id = response_data["session_id"]
        logger.debug(f"LemonSlice Session ID = {session_id}")
        return session_id  # type: ignore

    async def join_meeting(
        self,
        session_id: str,
        *,
        meeting_url: str,
        livekit_url: str,
        broadcast_token: str,
        bot_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> JoinMeetingResult:
        """Add an active avatar session to an external video meeting.

        Supports Zoom, Google Meet, Microsoft Teams, and Webex.

        Args:
            session_id: LemonSlice agent session ID.
            meeting_url: URL of the external meeting to join.
            livekit_url: LiveKit server URL for the agent room.
            broadcast_token: LiveKit token used to subscribe to avatar media.
            bot_name: Optional display name for the bot in the meeting.

        Returns:
            JoinMeetingResult with relay WebSocket URL and meeting bot ID.
        """
        payload: dict[str, Any] = {
            "session_id": session_id,
            "meeting_url": meeting_url,
            "livekit_url": livekit_url,
            "broadcast_token": broadcast_token,
        }
        if utils.is_given(bot_name) and bot_name:
            payload["bot_name"] = bot_name

        url = f"{self._api_url.rstrip('/')}/{session_id}/join-meeting"
        data = await self._post(payload, url=url)
        return JoinMeetingResult(
            websocket_url=str(data["websocket_url"]),
            meeting_bot_id=str(data["meeting_bot_id"]),
        )

    async def leave_meeting(
        self,
        session_id: str,
        *,
        meeting_bot_id: str,
    ) -> None:
        """Remove the avatar from an external meeting.

        Args:
            session_id: LemonSlice agent session ID.
            meeting_bot_id: Meeting bot ID returned by join_meeting().
        """
        url = f"{self._api_url.rstrip('/')}/{session_id}/leave-meeting"
        await self._post(
            {"meeting_bot_id": meeting_bot_id},
            url=url,
        )

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        url: str | None = None,
        image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """
        Make a POST request to the LemonSlice API with retry logic.

        Args:
            payload: JSON payload for the request.
            url: Optional URL override.
            image_bytes: Optional PNG-encoded image.

        Returns:
            Response data as a dictionary

        Raises:
            APIConnectionError: If the request fails after all retries
        """
        session = self._session or aiohttp.ClientSession()
        try:
            for i in range(self._conn_options.max_retry + 1):
                try:
                    headers = {"X-API-Key": self._api_key}
                    request_kwargs: dict[str, Any]
                    if image_bytes is not None:
                        form = aiohttp.FormData()
                        form.add_field(
                            "payload", json.dumps(payload), content_type="application/json"
                        )
                        # Upload the image using multipart
                        form.add_field(
                            "image",
                            image_bytes,
                            filename="image.png",
                            content_type="image/png",
                        )
                        request_kwargs = {"data": form}
                    else:
                        headers["Content-Type"] = "application/json"
                        request_kwargs = {"json": payload}

                    async with session.post(
                        url or self._api_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(
                            total=30.0,
                            sock_connect=self._conn_options.timeout,
                        ),
                        **request_kwargs,
                    ) as response:
                        if not response.ok:
                            text = await response.text()
                            raise APIStatusError(
                                "LemonSlice Server returned an error",
                                status_code=response.status,
                                body=text,
                            )
                        return await response.json()  # type: ignore
                except Exception as e:
                    if isinstance(e, APIStatusError):
                        logger.error(
                            "LemonSlice API returned an error",
                            extra={
                                "status_code": e.status_code,
                                "body": e.body,
                            },
                        )
                        if not e.retryable:
                            raise e
                    elif isinstance(e, APIConnectionError):
                        logger.warning("failed to call LemonSlice api", extra={"error": str(e)})
                    else:
                        logger.exception("failed to call lemonslice api")

                    if i < self._conn_options.max_retry:
                        await asyncio.sleep(self._conn_options._interval_for_retry(i))
        finally:
            if not self._session:  # if we created the session, we close it
                await session.close()

        raise APIConnectionError("Failed to call LemonSlice API after all retries")


def _encode_image(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes for a multipart upload."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
