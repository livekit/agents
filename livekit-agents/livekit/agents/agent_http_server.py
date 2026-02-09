"""HTTP server for AgentServer with health check and text endpoints."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web

from livekit.protocol import agent

from . import llm, utils
from .job import TextMessageError
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
async def _cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
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

        Request body (JSON):
            {
                "text": "user message",
                "metadata": {"key": "value"},  // optional
                "session_state": {  // optional
                    "version": 1,
                    "snapshot": "base64...",
                    "delta": "base64..."
                }
            }

        Response (streaming NDJSON):
            1. Session acknowledgment: {"type": "ack", "session_id": "...", "message_id": "..."}
            2. Response events: {"type": "message|function_call|...", "data": {...}}
            3. Completion: {"type": "complete", "session_state": {...}, "error": "..."}
        """
        logger.info(f"handling text request: {request.method} {request.path}")
        text_request = await self._parse_text_request(request)

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

        try:
            session_info = await self._agent_server._launch_text_job(text_request)
            ack_data = {
                "type": "ack",
                "session_id": text_request.session_id,
                "message_id": text_request.message_id,
            }
            await response.write(json.dumps(ack_data).encode() + b"\n")

            # stream response events
            async for ev in session_info.event_ch:
                event_data = {
                    "type": ev.event_type,
                    "data": json.loads(ev.data) if ev.data else {},
                }
                await response.write(json.dumps(event_data).encode() + b"\n")

            # final response
            complete = await session_info.done_fut
            completion_data: dict[str, Any] = {"type": "complete"}

            if complete.error:
                completion_data["error"] = complete.error

            elif complete.session_state:
                completion_data["session_state"] = {
                    "version": complete.session_state.version,
                }
                if complete.session_state.snapshot:
                    completion_data["session_state"]["snapshot"] = base64.b64encode(
                        complete.session_state.snapshot
                    ).decode()
                if complete.session_state.delta:
                    completion_data["session_state"]["delta"] = base64.b64encode(
                        complete.session_state.delta
                    ).decode()

            await response.write(json.dumps(completion_data).encode() + b"\n")
        except TextMessageError as e:
            logger.error(
                "error processing text request",
                extra={"session_id": text_request.session_id, "error": str(e)},
            )
            with contextlib.suppress(Exception):
                error_data = {"type": "complete", "error": str(e)}
                await response.write(json.dumps(error_data).encode() + b"\n")
        except Exception:
            logger.exception(
                "unexpected error processing text request",
                extra={"session_id": text_request.session_id},
            )
            with contextlib.suppress(Exception):
                error_data = {
                    "type": "complete",
                    "error": str(TextMessageError("internal error")),
                }
                await response.write(json.dumps(error_data).encode() + b"\n")
        finally:
            await response.write_eof()

        return response

    async def _parse_text_request(self, request: web.Request) -> agent.TextMessageRequest:
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

        text = body.get("text")
        if text is None:
            raise web.HTTPBadRequest(
                reason=json.dumps({"error": "Missing 'text' field"}),
                content_type="application/json",
            )

        if not session_id:
            session_id = utils.shortuuid("text_session_")
        message_id = utils.shortuuid("text_msg_")

        # add "endpoint" to metadata for now
        metadata = body.get("metadata", {})
        if not isinstance(metadata, dict):
            raise web.HTTPBadRequest(
                reason=json.dumps({"error": "'metadata' must be a JSON object"}),
                content_type="application/json",
            )
        metadata["endpoint"] = endpoint
        metadata_str = json.dumps(metadata)

        # create TextMessageRequest proto
        text_request = agent.TextMessageRequest(
            message_id=message_id,
            session_id=session_id,
            agent_name=self._agent_server._agent_name,
            metadata=metadata_str,
            text=text,
        )
        if session_state_dict := body.get("session_state"):
            session_state = agent.AgentSessionState(version=session_state_dict.get("version", 0))
            if (snapshot := session_state_dict.get("snapshot")) is not None:
                session_state.snapshot = base64.b64decode(snapshot)
            if (delta := session_state_dict.get("delta")) is not None:
                session_state.delta = base64.b64decode(delta)
            text_request.session_state.CopyFrom(session_state)
        return text_request


@dataclass
class TextSessionStarted:
    """Acknowledgment response when starting a text session."""

    session_id: str
    message_id: str


@dataclass
class TextResponseEvent:
    """Event from the agent during text processing."""

    item: llm.ChatItem


@dataclass
class TextSessionComplete:
    """Completion response with final session state or error."""

    session_state: agent.AgentSessionState | None = None
    error: str | None = None


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

    async def send_text_stream(
        self,
        text: str,
        *,
        endpoint: str = "",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_state: agent.AgentSessionState | None = None,
    ) -> AsyncIterator[TextSessionStarted | TextResponseEvent | TextSessionComplete]:
        """Send a text message and stream responses as they arrive.

        Args:
            text: The text message to send
            endpoint: Optional endpoint name to route to specific handler
            session_id: Optional session ID to continue an existing session
            metadata: Optional metadata dictionary
            session_state: Optional session state to restore

        Yields:
            TextSessionAck: Initial acknowledgment
            TextResponseEvent: Events as they arrive
            TextSessionComplete: Final completion with session state or error

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
            body["session_state"] = {
                "version": session_state.version,
            }
            if session_state.HasField("snapshot"):
                body["session_state"]["snapshot"] = base64.b64encode(
                    session_state.snapshot
                ).decode()
            if session_state.HasField("delta"):
                body["session_state"]["delta"] = base64.b64encode(session_state.delta).decode()

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

                event_type = data.get("type")
                if event_type == "ack":
                    yield TextSessionStarted(
                        session_id=data["session_id"],
                        message_id=data["message_id"],
                    )
                elif event_type == "complete":
                    # parse completion
                    error = data.get("error")
                    session_state_data = data.get("session_state")

                    parsed_state: agent.AgentSessionState | None = None
                    if session_state_data:
                        parsed_state = agent.AgentSessionState(
                            version=session_state_data.get("version", 0)
                        )
                        if (snapshot := session_state_data.get("snapshot")) is not None:
                            parsed_state.snapshot = base64.b64decode(snapshot)
                        if (delta := session_state_data.get("delta")) is not None:
                            parsed_state.delta = base64.b64decode(delta)

                    yield TextSessionComplete(session_state=parsed_state, error=error)
                else:
                    data = data.get("data", {})
                    ev: (
                        llm.ChatMessage
                        | llm.FunctionCall
                        | llm.FunctionCallOutput
                        | llm.AgentHandoff
                    )
                    if event_type == "message":
                        ev = llm.ChatMessage.model_validate(data)
                    elif event_type == "function_call":
                        ev = llm.FunctionCall.model_validate(data)
                    elif event_type == "function_call_output":
                        ev = llm.FunctionCallOutput.model_validate(data)
                    elif event_type == "agent_handoff":
                        ev = llm.AgentHandoff.model_validate(data)
                    else:
                        logger.warning(f"unsupported event type: {event_type}")
                        continue

                    yield TextResponseEvent(item=ev)
