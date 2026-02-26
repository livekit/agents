"""HTTP server for AgentServer with health check and text endpoints."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, get_type_hints

import aiohttp
import aiohttp.typedefs
from aiohttp import web
from google.protobuf.json_format import MessageToDict, ParseDict

from livekit.protocol import agent
from livekit.protocol.agent_pb import agent_text

from .log import logger
from .types import NOT_GIVEN, NotGivenOr
from .utils import is_given
from .utils.http_server import HttpServer
from .version import __version__

if TYPE_CHECKING:
    from .worker import AgentServer


class _ParamSource(Enum):
    REQUEST = "request"
    AGENT_SERVER = "agent_server"
    VALUE = "value"


_MISSING: Any = object()


@dataclass
class _ParamSpec:
    name: str
    source: _ParamSource
    default: Any = _MISSING


@dataclass
class _HttpEndpointInfo:
    handler: Callable[..., Awaitable[Any]]
    path: str
    methods: list[str] = field(default_factory=lambda: ["GET"])
    headers: dict[str, str] | None = None


DEFAULT_HEADERS: dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class AgentHttpServer(HttpServer):
    """HTTP server that handles health checks, worker info, and text message endpoints."""

    RESERVED_PATHS: set[str] = {"/"}
    RESERVED_PATH_PREFIXES: tuple[str, ...] = ("/worker", "/text")

    def __init__(
        self,
        agent_server: AgentServer,
        *,
        host: str,
        port: int,
        default_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
    ) -> None:
        super().__init__(host, port)
        self._agent_server = agent_server
        self._default_headers = default_headers if is_given(default_headers) else DEFAULT_HEADERS

        if self._default_headers:
            headers = self._default_headers

            @web.middleware
            async def _headers_middleware(
                request: web.Request, handler: aiohttp.typedefs.Handler
            ) -> web.StreamResponse:
                try:
                    resp = await handler(request)
                except web.HTTPException as exc:
                    for k, v in headers.items():
                        exc.headers[k] = v
                    raise
                for k, v in headers.items():
                    resp.headers[k] = v
                return resp

            self._app.middlewares.append(_headers_middleware)

        self._app.add_routes(
            [
                web.get("/", self._health_check),
                web.get("/worker", self._worker_info),
            ]
        )

        if "Access-Control-Allow-Origin" in self._default_headers:

            async def _handle_cors_preflight(request: web.Request) -> web.Response:
                return web.Response(
                    headers={**self._default_headers, "Access-Control-Max-Age": "3600"}
                )

            self._app.router.add_route("OPTIONS", "/{path:.*}", _handle_cors_preflight)

        # text routes (reserved)
        from .text_job import _handle_text_request

        text_handler = self._wrap_user_handler(
            _handle_text_request, {"Content-Type": "application/x-ndjson"}
        )
        for path in [
            "/text",
            "/text/sessions/{session_id}",
            "/text/{endpoint}",
            "/text/{endpoint}/sessions/{session_id}",
        ]:
            self._app.router.add_route("POST", path, text_handler)

        # user-defined endpoints
        for ep in agent_server._http_endpoints:
            wrapped = self._wrap_user_handler(ep.handler, ep.headers)
            for method in ep.methods:
                self._app.router.add_route(method, ep.path, wrapped)

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

    def _wrap_user_handler(
        self,
        handler: Callable[..., Awaitable[Any]],
        endpoint_headers: dict[str, str] | None = None,
    ) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
        params = self._build_injection_plan(handler)

        async def _handler(request: web.Request) -> web.StreamResponse:
            kwargs = await self._resolve_kwargs(request, params)
            try:
                result = await handler(**kwargs)
            except web.HTTPException:
                raise
            except Exception:
                logger.exception(
                    "error in user http endpoint handler",
                    extra={"path": request.path},
                )
                return web.json_response({"error": "internal server error"}, status=500)

            try:
                return await self._result_to_response(result, request, endpoint_headers)
            except Exception:
                logger.exception(
                    "error converting handler return value",
                    extra={"path": request.path},
                )
                return web.json_response({"error": "internal server error"}, status=500)

        return _handler

    def _build_injection_plan(self, handler: Callable[..., Awaitable[Any]]) -> list[_ParamSpec]:
        """Introspect handler once at wrap time. Returns params."""
        try:
            # localns needed because AgentServer is TYPE_CHECKING-only import
            hints = get_type_hints(handler, localns={"AgentServer": type(self._agent_server)})
        except Exception:
            hints = {}

        agent_server_type = type(self._agent_server)
        sig = inspect.signature(handler)
        params: list[_ParamSpec] = []
        for name, param in sig.parameters.items():
            hint = hints.get(name, inspect.Parameter.empty)
            default = param.default if param.default is not inspect.Parameter.empty else _MISSING
            if hint is web.Request:
                params.append(_ParamSpec(name, _ParamSource.REQUEST))
            elif hint is not inspect.Parameter.empty and (
                hint is agent_server_type
                or (isinstance(hint, type) and issubclass(hint, agent_server_type))
            ):
                params.append(_ParamSpec(name, _ParamSource.AGENT_SERVER))
            else:
                params.append(_ParamSpec(name, _ParamSource.VALUE, default))

        return params

    async def _resolve_kwargs(
        self, request: web.Request, params: list[_ParamSpec]
    ) -> dict[str, Any]:
        """Resolve handler kwargs from request data. Called per request."""
        values: dict[str, Any] = {}
        if any(p.source is _ParamSource.VALUE for p in params):
            if request.method in {"POST", "PUT", "PATCH"}:
                try:
                    body = await request.json()
                except Exception as e:
                    raise web.HTTPBadRequest(
                        reason=json.dumps({"error": f"Invalid JSON: {str(e)}"}),
                        content_type="application/json",
                    ) from e
                values.update(body)
            else:
                values.update(request.query)

            # match_info contains path parameters, overrides others
            values.update(dict(request.match_info))

        kwargs: dict[str, Any] = {}
        for p in params:
            if p.source is _ParamSource.REQUEST:
                kwargs[p.name] = request
            elif p.source is _ParamSource.AGENT_SERVER:
                kwargs[p.name] = self._agent_server
            elif p.name in values:
                kwargs[p.name] = values[p.name]
            elif p.default is not _MISSING:
                kwargs[p.name] = p.default
            else:
                raise web.HTTPBadRequest(
                    reason=json.dumps({"error": f"Missing required parameter: '{p.name}'"}),
                    content_type="application/json",
                )
        return kwargs

    async def _result_to_response(
        self, result: Any, request: web.Request, endpoint_headers: dict[str, str] | None
    ) -> web.StreamResponse:
        """Convert handler return value to web.StreamResponse."""
        if isinstance(result, web.StreamResponse):
            return result

        # streaming: must include default_headers before prepare() since
        # the middleware runs after the handler returns — too late for
        # headers that were already sent via prepare().
        if isinstance(result, AsyncIterator):
            headers = dict(self._default_headers)
            if endpoint_headers:
                headers.update(endpoint_headers)
            headers.setdefault("Content-Type", "application/octet-stream")

            resp = web.StreamResponse(status=200, headers=headers)
            await resp.prepare(request)
            async for chunk in result:
                await resp.write(chunk.encode() if isinstance(chunk, str) else chunk)
            await resp.write_eof()
            return resp

        # non-streaming: middleware handles default_headers; only apply
        # endpoint-specific headers here.
        if isinstance(result, dict):
            resp = web.json_response(result)
        elif isinstance(result, str):
            resp = web.Response(text=result)
        elif isinstance(result, bytes):
            resp = web.Response(body=result)
        else:
            logger.error(
                "unsupported return type from http endpoint handler",
                extra={"path": request.path, "type": type(result).__name__},
            )
            return web.json_response({"error": "internal server error"}, status=500)

        if endpoint_headers:
            resp.headers.update(endpoint_headers)
        return resp


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
