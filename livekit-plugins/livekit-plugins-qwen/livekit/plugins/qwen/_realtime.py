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

"""Shared WebSocket plumbing for Model Studio's realtime STT and TTS endpoints.

Both endpoints take the model in the query string and the API key as a bearer token,
and both end with a ``session.finish`` / ``session.finished`` handshake. The STT and
TTS classes only translate events; connecting, tagging events with ids and bounding
the shutdown wait live here.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, utils

# How long to wait for `session.finished` once we have asked to finish. Model Studio
# drains any audio it still owes in this window, so it has to be generous; without a
# bound a wedged upstream stalls the session, since APIConnectOptions.timeout only
# covers the connect.
DEFAULT_FINISH_TIMEOUT = 10.0

# How long a cancelled stream may spend on a courtesy session.finish before dropping
# the socket. LiveKit tears live streams down with aclose(), which cancels the run
# before the normal finish path; Model Studio counts a socket closed without the
# handshake as a failed request. Short, because this sits on the session-teardown path.
TEARDOWN_FINISH_TIMEOUT = 2.0

# Silence is normal before the handshake starts (the user simply isn't talking), so the
# finish deadline can only be checked between receives.
_POLL_INTERVAL = 0.5

CLOSE_TYPES = (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING)


async def connect(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> aiohttp.ClientWebSocketResponse:
    """Open ``<base_url>?model=<model>`` with a bearer token, mapping failures to API errors."""
    try:
        return await asyncio.wait_for(
            session.ws_connect(
                f"{base_url}?model={model}",
                headers={"Authorization": f"Bearer {api_key}"},
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise APITimeoutError() from None
    except aiohttp.ClientResponseError as e:
        # A rejected upgrade (WSServerHandshakeError) carries RequestInfo, whose headers
        # hold the API key; drop the cause so it can never be logged.
        raise APIStatusError(
            message=e.message, status_code=e.status, request_id=None, body=None
        ) from None
    except Exception as e:
        raise APIConnectionError() from e


class RealtimeSocket:
    """A Model Studio realtime socket that cannot hang on shutdown."""

    def __init__(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        *,
        finish_timeout: float = DEFAULT_FINISH_TIMEOUT,
    ) -> None:
        self._ws = ws
        self._finish_timeout = finish_timeout
        self._deadline: float | None = None
        self._finish_sent = False

    async def send(self, event_type: str, **fields: Any) -> None:
        """Send one client event, tagged with a fresh ``event_id``."""
        await self._ws.send_json(
            {"event_id": utils.shortuuid("evt_"), "type": event_type, **fields}
        )

    async def finish(self) -> None:
        """Ask the server to wrap up, and start the clock on its reply."""
        self._finish_sent = True
        await self.send("session.finish")
        self._deadline = time.monotonic() + self._finish_timeout

    async def receive(self) -> aiohttp.WSMessage:
        """The next message, or ``APITimeoutError`` once the finish wait is over."""
        while True:
            try:
                return await self._ws.receive(timeout=_POLL_INTERVAL)
            except asyncio.TimeoutError:
                if self._deadline is not None and time.monotonic() > self._deadline:
                    raise APITimeoutError(
                        f"Model Studio did not answer session.finish within {self._finish_timeout}s"
                    ) from None

    async def close(self) -> None:
        await self._ws.close()

    async def close_with_finish(self) -> None:
        """Send ``session.finish`` if it has not gone out, then close without waiting.

        For teardown paths that sit on a latency-critical wait, such as a TTS barge-in,
        where the voice pipeline awaits the stream's ``aclose()`` before clearing the
        playout buffer. The server books the request as finished on receiving the event;
        waiting for its ``session.finished`` reply would only delay silencing the agent.
        Best effort: never raises, always closes.
        """
        try:
            if not self._finish_sent and not self._ws.closed:
                await self.finish()
        except Exception:
            pass
        finally:
            await self._ws.close()

    async def close_gracefully(self, timeout: float = TEARDOWN_FINISH_TIMEOUT) -> None:
        """Complete the finish handshake if it never ran, then close.

        For the cancellation path: the receive loop is already stopped, so this drains
        the socket itself until ``session.finished`` or the bound. Best effort: never
        raises, always closes.
        """
        try:
            if not self._finish_sent and not self._ws.closed:
                await asyncio.wait_for(self._finish_handshake(), timeout)
        except Exception:
            pass
        finally:
            await self._ws.close()

    async def _finish_handshake(self) -> None:
        await self.finish()
        while True:
            msg = await self._ws.receive()
            if msg.type in CLOSE_TYPES:
                return
            if (
                msg.type is aiohttp.WSMsgType.TEXT
                and json.loads(msg.data).get("type") == "session.finished"
            ):
                return


def status_error_from(event: dict[str, Any]) -> APIStatusError:
    """Model Studio reports failures as an ``error`` event, not an HTTP status.

    Only ``error.type`` says whether a retry can help: an ``invalid_request_error`` will
    fail the same way again, so it maps to 400 (which LiveKit never retries); anything
    else keeps the retryable default.
    """
    # A top-level or malformed error payload (no nested `error` object) still has to
    # produce an error, so fall back to the event itself. An empty `error` object is kept
    # as-is rather than being replaced wholesale by the event.
    nested = event.get("error")
    error: dict[str, Any] = nested if isinstance(nested, dict) else event
    message = error.get("message")
    error_type = error.get("type")
    return APIStatusError(
        message=message or "Model Studio returned an error event",
        status_code=400 if error_type == "invalid_request_error" else -1,
        request_id=event.get("event_id"),
        body=error,
    )
