"""HTTP server for AgentServer with health check and text endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from livekit.protocol import agent

from . import utils
from .log import logger
from .utils.http_server import HttpServer
from .version import __version__

if TYPE_CHECKING:
    from .worker import AgentServer


class AgentHttpServer(HttpServer):
    """HTTP server that handles health checks, worker info, and text message endpoints."""

    def __init__(
        self,
        agent_server: AgentServer,
        *,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(host, port, loop)
        self._agent_server = agent_server

        self._app.add_routes(
            [
                web.get("/", self._health_check),
                web.get("/worker", self._worker_info),
                web.post("/text", self._handle_text_request),
                web.post("/text/sessions/{session_id}", self._handle_text_request),
                web.post("/text/{endpoint}", self._handle_text_request),
                web.post("/text/{endpoint}/sessions/{session_id}", self._handle_text_request),
            ]
        )

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

        text_request = await self._parse_text_request(request)

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
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
        except Exception as e:
            logger.exception(
                "error processing text request", extra={"session_id": text_request.session_id}
            )
            try:
                error_data = {"type": "complete", "error": str(e)}
                await response.write(json.dumps(error_data).encode() + b"\n")
            except Exception:
                pass
        finally:
            await response.write_eof()

        return response

    async def _parse_text_request(self, request: web.Request) -> agent.TextMessageRequest:
        endpoint = request.match_info.get("endpoint", "")
        session_id = request.match_info.get("session_id")

        if endpoint not in self._agent_server._text_handler_fncs:
            raise web.HTTPNotFound(
                body=json.dumps({"error": f"Service '{endpoint}' not found"}),
                content_type="application/json",
            )
        try:
            body = await request.json()
        except Exception as e:
            raise web.HTTPBadRequest(
                body=json.dumps({"error": f"Invalid JSON: {str(e)}"}),
                content_type="application/json",
            ) from e

        text = body.get("text")
        if text is None:
            raise web.HTTPBadRequest(
                body=json.dumps({"error": "Missing 'text' field"}),
                content_type="application/json",
            )

        if not session_id:
            session_id = utils.shortuuid("text_session_")
        message_id = utils.shortuuid("text_msg_")

        # add "endpoint" to metadata for now
        metadata = body.get("metadata", {})
        if not isinstance(metadata, dict):
            raise web.HTTPBadRequest(
                body=json.dumps({"error": "'metadata' must be a JSON object"}),
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
        if session_state := body.get("session_state"):
            text_request.session_state = agent.AgentSessionState(
                version=session_state.get("version", 0)
            )
            if (snapshot := session_state.get("snapshot")) is not None:
                text_request.session_state.snapshot = base64.b64decode(snapshot)
            if (delta := session_state.get("delta")) is not None:
                text_request.session_state.delta = base64.b64decode(delta)

        return text_request
