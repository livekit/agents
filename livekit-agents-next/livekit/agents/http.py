"""``AgentServer.http``: the ASGI app behind an agent's HTTP endpoints, served on the endpoint data
plane.

Routes are served through the LiveKit server at ``/agents/{agent_name}/{deployment}/{path}``; no
local listener exposes them. The Rust worker (``_core.Worker``) accepts one stream per request,
parses the node's preamble and hands the stream here; ``_core.EndpointServer`` runs one ASGI call
over it: an HTTP/1.1 exchange, or a WebSocket after an upgrade.

The handler sees the preamble through ``request.state.lk_request`` (:class:`EndpointRequest`):
the request id, whether the caller holds the call permission, the manifest route that matched,
and whether the route is public (``configure_endpoint(path, public=True)``).
The project credential never reaches the worker: the node strips it and asserts ``authorized``
here; a thread grant reaches a public route as its bearer, for the handler to verify.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from starlette.applications import Starlette
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute
from starlette.types import ASGIApp

from . import _core
from ._core import (
    HSR_ABORT as HSR_ABORT,
    HSR_INTERNAL as HSR_INTERNAL,
    HSR_PROTOCOL as HSR_PROTOCOL,
    HSR_REFUSED as HSR_REFUSED,
    HSR_TIMEOUT as HSR_TIMEOUT,
)
from ._generated.worker_endpoint import (
    EndpointOptions as EndpointOptions,
    EndpointRequest as EndpointRequest,
)

_ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"]


class AgentHttp:
    """The routes an agent serves, as a Starlette application.

    ::

        @server.http.get("/token")
        async def token(request: Request) -> Response:
            return JSONResponse({"ok": True})

        @server.http.websocket("/echo")
        async def echo(ws: WebSocket) -> None:
            await ws.accept()
            ...

        server.http.mount("/api", fastapi_app)

    Handlers are Starlette endpoints: a request in, a response out. Any ASGI application mounts.
    """

    def __init__(self) -> None:
        self._routes: list[BaseRoute] = []
        self._app: ASGIApp | None = None
        self._options: dict[str, EndpointOptions] = {}
        self._server = _core.EndpointServer()

    # --- registration -------------------------------------------------------------------------

    def route(
        self, path: str, *, methods: Iterable[str], name: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_route(path, func, methods=list(methods), name=name)
            return func

        return decorator

    def get(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["GET"], **kw)

    def post(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["POST"], **kw)

    def put(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["PUT"], **kw)

    def patch(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["PATCH"], **kw)

    def delete(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["DELETE"], **kw)

    def websocket(
        self, path: str, *, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_websocket_route(path, func, name=name)
            return func

        return decorator

    def add_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        methods: list[str],
        name: str | None = None,
    ) -> None:
        self._add(Route(path, endpoint, methods=methods, name=name))

    def add_websocket_route(
        self, path: str, endpoint: Callable[..., Any], *, name: str | None = None
    ) -> None:
        self._add(WebSocketRoute(path, endpoint, name=name))

    def mount(self, path: str, app: ASGIApp, *, name: str | None = None) -> None:
        """Serve an ASGI application (FastAPI, another Starlette app, ...) under ``path``."""
        self._add(Mount(path, app=app, name=name))

    def configure_endpoint(self, path: str, *, public: bool = False) -> None:
        """Set routing policy for one route, by its registered path."""
        self._options[path] = EndpointOptions(public=public)

    def _add(self, route: BaseRoute) -> None:
        if self._app is not None:
            raise RuntimeError("routes cannot be added once the agent server is running")
        self._routes.append(route)

    @property
    def routes(self) -> list[BaseRoute]:
        return list(self._routes)

    # --- what the worker needs --------------------------------------------------------------

    @property
    def app(self) -> ASGIApp:
        """The ASGI application, built on first use; routes are fixed from then on."""
        if self._app is None:
            self._app = Starlette(routes=self._routes)
        return self._app

    def manifest(self) -> list[dict[str, Any]]:
        """The endpoint manifest the worker registers, in route order: ``{path, methods, kind,
        public}`` per route, as ``_core.Worker`` takes it.

        Methods travel exactly as Starlette routes them (a GET route also answers HEAD); a
        WebSocket route is a GET; a mount declares every method under ``{path:path}``.
        """
        manifest: list[dict[str, Any]] = []
        for route in self._routes:
            if isinstance(route, Route):
                path, methods = route.path, sorted(route.methods or ())
            elif isinstance(route, WebSocketRoute):
                path, methods = route.path, ["GET"]
            elif isinstance(route, Mount):
                path, methods = route.path.rstrip("/") + "/{path:path}", list(_ALL_METHODS)
            else:
                continue
            opts = self._options.get(route.path, EndpointOptions())
            manifest.append(
                {
                    "path": path,
                    "methods": methods,
                    "kind": "http",
                    "public": opts.public,
                }
            )
        return manifest

    async def serve(self, args: dict[str, Any]) -> None:
        """One ``open("endpoint", args)`` call from the worker: serve the stream to completion."""
        await serve_endpoint(args.pop("stream"), _request_from_args(args), self.app, self)

    async def close_websockets(self, code: int = 1001, reason: str = "worker draining") -> None:
        """Send every open WebSocket a close frame, so clients reconnect to another worker."""
        await self._server.close_websockets(code, reason)


def _request_from_args(args: dict[str, Any]) -> EndpointRequest:
    return EndpointRequest(
        request_id=args.get("request_id", ""),
        authorized=bool(args.get("authorized", False)),
        route=args.get("route", ""),
        kind=args.get("kind", "http"),
        client_addr=args.get("client_addr", ""),
        scheme=args.get("scheme") or "http",
        timeout_ms=int(args.get("timeout_ms") or 0),
    )


async def serve_endpoint(
    stream: Any, request: EndpointRequest, app: ASGIApp, registry: AgentHttp | None = None
) -> None:
    """Serve one exchange stream (an ``EndpointStream`` after its preamble, or any object with
    its five methods) with ``app``; ``registry`` keeps its WebSocket for draining."""
    server = registry._server if registry is not None else _core.EndpointServer()
    await server.serve(stream, app, request)
