"""HTTP server for AgentServer with health check and text endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import aiohttp
import aiohttp.typedefs
from aiohttp import web
from google.protobuf.json_format import MessageToDict, ParseDict

from livekit.protocol import agent
from livekit.protocol.agent_pb import agent_text

from . import utils
from ._exceptions import TextMessageError
from .log import logger
from .utils.http_server import HttpServer
from .version import __version__

if TYPE_CHECKING:
    from .worker import AgentServer

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


@web.middleware
async def _cors_middleware(
    request: web.Request, handler: aiohttp.typedefs.Handler
) -> web.StreamResponse:
    try:
        resp = await handler(request)
    except web.HTTPException as exc:
        for k, v in _CORS_HEADERS.items():
            exc.headers[k] = v
        raise
    for k, v in _CORS_HEADERS.items():
        resp.headers[k] = v
    return resp


class AgentHttpServer(HttpServer):
    """HTTP server that handles health checks, worker info, and text message endpoints."""

    def __init__(
        self,
        agent_server: AgentServer,
        *,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(host, port, loop)
        self._agent_server = agent_server

        self._app.middlewares.append(_cors_middleware)
        self._app.add_routes(
            [
                web.get("/", self._health_check),
                web.get("/worker", self._worker_info),
                web.options("/{path:.*}", self._handle_cors_preflight),
                web.post("/text", self._handle_text_request),
                web.post("/text/sessions/{session_id}", self._handle_text_request),
                web.post("/text/{endpoint}", self._handle_text_request),
                web.post("/text/{endpoint}/sessions/{session_id}", self._handle_text_request),
            ]
        )

    async def _handle_cors_preflight(self, _: web.Request) -> web.Response:
        return web.Response(headers={**_CORS_HEADERS, "Access-Control-Max-Age": "3600"})

    async def _health_check(self, _: web.Request) -> web.Response:
        """Health check endpoint - returns 200 if server is healthy."""
        if (
            self._agent_server._inference_executor is not None
            and not self._agent_server._inference_executor.is_alive()
        ):
            return web.Response(status=503, text="inference process not running")

        if self._agent_server._connection_failed:
            return web.Response(status=503, text="failed to connect to livekit")

        return web.Response(text="OK")

    async def _worker_info(self, _: web.Request) -> web.Response:
        """Worker info endpoint - returns worker metadata."""
        body = json.dumps(
            {
                "agent_name": self._agent_server._agent_name,
                "worker_type": agent.JobType.Name(self._agent_server._server_type.value),
                "worker_load": getattr(self._agent_server, "_worker_load", 0.0),
                "active_jobs": len(self._agent_server.active_jobs),
                "sdk_version": __version__,
                "project_type": "python",
            }
        )
        return web.Response(body=body, content_type="application/json")

    async def _handle_text_request(self, request: web.Request) -> web.StreamResponse:
        """Handle POST /text/{endpoint}[/sessions/{session_id}]

        Request body is a JSON object that is parsed as `agent_text.TextMessageRequest`:
        ```json
        {
            "text": "user message",
            "metadata": {"key": "value"},  // optional
            "session_state": {  // optional
                "version": 1,
                "snapshot": "base64...",
                "delta": "base64..."
            }
        }
        ```

        Response as streaming NDJSON of `agent_text.TextMessageResponse` events:
        ```json
        {
            "session_id": "...",
            "message_id": "...",
            "message|function_call|function_call_output|agent_handoff": {...},
            "complete": {
                "session_state": {...},
                "error": {
                    "message": "...",
                    "code": "..."
                }
            }
        }
        ```
        """
        logger.info(f"handling text request: {request.method} {request.path}")
        endpoint, text_request = await self._parse_text_request(request)

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                **_CORS_HEADERS,
            },
        )
        await response.prepare(request)

        completed = False

        async def _stream_message(msg: agent_text.TextMessageResponse) -> None:
            nonlocal completed

            msg_json = json.dumps(MessageToDict(msg, preserving_proto_field_name=True))
            if completed:
                logger.warning("received message after completion", extra={"message": msg_json})
                return

            await response.write(msg_json.encode() + b"\n")
            if msg.WhichOneof("event") == "complete":
                completed = True

        try:
            session_info = await self._agent_server._launch_text_job(endpoint, text_request)

            # stream response events
            async for ev in session_info.event_ch:
                await _stream_message(ev)

        except TextMessageError as e:
            logger.error(
                "error processing text request",
                extra={
                    "session_id": text_request.session_id,
                    "error": e.message,
                    "error_code": agent_text.TextMessageErrorCode.Name(e.code),
                },
            )
            if not completed:
                await _stream_message(
                    agent_text.TextMessageResponse(
                        session_id=text_request.session_id,
                        message_id=text_request.message_id,
                        complete=agent_text.TextMessageComplete(error=e.to_proto()),
                    )
                )
        except Exception:
            logger.exception(
                "unexpected error processing text request",
                extra={"session_id": text_request.session_id},
            )
            if not completed:
                await _stream_message(
                    agent_text.TextMessageResponse(
                        session_id=text_request.session_id,
                        message_id=text_request.message_id,
                        complete=agent_text.TextMessageComplete(
                            error=TextMessageError("internal error").to_proto()
                        ),
                    )
                )
        finally:
            await response.write_eof()

        return response

    async def _parse_text_request(
        self, request: web.Request
    ) -> tuple[str, agent_text.TextMessageRequest]:
        endpoint = request.match_info.get("endpoint", "")
        session_id = request.match_info.get("session_id")
        logger.info(f"parsing text request: {endpoint} {session_id}")

        if endpoint not in self._agent_server._text_handler_fncs:
            raise web.HTTPNotFound(
                reason=json.dumps({"error": f"Service '{endpoint}' not found"}),
                content_type="application/json",
            )
        try:
            body = await request.json()
        except Exception as e:
            raise web.HTTPBadRequest(
                reason=json.dumps({"error": f"Invalid JSON: {str(e)}"}),
                content_type="application/json",
            ) from e

        if "text" not in body:
            raise web.HTTPBadRequest(
                reason=json.dumps({"error": "Missing 'text' field"}),
                content_type="application/json",
            )

        if "message_id" not in body:
            body["message_id"] = utils.shortuuid("text_msg_")
        if "agent_name" not in body:
            body["agent_name"] = self._agent_server._agent_name
        body["session_id"] = session_id or utils.shortuuid("text_session_")

        try:
            text_request = ParseDict(
                body, agent_text.TextMessageRequest(), ignore_unknown_fields=True
            )
        except Exception as e:
            raise web.HTTPBadRequest(
                reason=json.dumps({"error": f"Invalid request body: {str(e)}"}),
                content_type="application/json",
            ) from e

        return endpoint, text_request


class AgentHttpClient:
    """Client to interact with AgentHttpServer API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: aiohttp.ClientTimeout | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            base_url: Base URL of the agent server (e.g., "http://localhost:8080")
            timeout: Optional timeout configuration for requests
        """
        self._base_url = base_url.rstrip("/")
        self._loop = loop or asyncio.get_event_loop()
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(loop=self._loop), timeout=timeout, loop=self._loop
        )

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    async def aclose(self) -> None:
        """Close the HTTP client session."""
        await self._session.close()

    async def __aenter__(self) -> AgentHttpClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    async def health_check(self) -> bool:
        """Check if the agent server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with self._session.get(f"{self._base_url}/") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def worker_info(self) -> dict[str, Any]:
        """Get worker information from the agent server.

        Returns:
            Dictionary containing worker metadata

        Raises:
            aiohttp.ClientError: If request fails
        """
        async with self._session.get(f"{self._base_url}/worker") as resp:
            resp.raise_for_status()
            return await resp.json()  # type: ignore

    async def send_text(
        self,
        text: str,
        *,
        endpoint: str = "",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_state: agent_text.AgentSessionState | None = None,
    ) -> AsyncIterator[agent_text.TextMessageResponse]:
        """Send a text message and stream responses as they arrive.

        Args:
            text: The text message to send
            endpoint: Optional endpoint name to route to specific handler
            session_id: Optional session ID to continue an existing session
            metadata: Optional metadata dictionary
            session_state: Optional session state to restore

        Yields:
            TextMessageResponse: Events as they arrive

        Raises:
            aiohttp.ClientError: If request fails
        """
        # build url path
        url = f"{self._base_url}/text"
        if endpoint:
            url = f"{url}/{endpoint}"
        if session_id:
            url = f"{url}/sessions/{session_id}"

        body: dict[str, Any] = {"text": text}
        if metadata:
            body["metadata"] = metadata
        if session_state:
            body["session_state"] = MessageToDict(session_state, preserving_proto_field_name=True)

        async with self._session.post(url, json=body) as resp:
            resp.raise_for_status()

            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse NDJSON line: {e}")
                    continue

                yield ParseDict(data, agent_text.TextMessageResponse())
