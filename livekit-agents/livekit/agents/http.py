"""The HTTP server behind ``AgentServer.http``: its built-in routes and the uvicorn runner."""

from __future__ import annotations

import asyncio
import contextlib
import re
import socket
from collections.abc import Generator
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from google.protobuf.json_format import MessageToJson
from starlette.responses import Response

from livekit.protocol import agent, agent_worker

from .log import logger
from .version import __version__

if TYPE_CHECKING:
    from .worker import AgentServer

_URL_SAFE = re.compile(r"[A-Za-z0-9._~-]+")
"""What may name an endpoint, so no path or query can hide inside one."""


def agent_health(server: AgentServer) -> str | None:
    """Why the agent server cannot take jobs, or None when it can.

    This is what ``GET /`` reports, and what a health route of your own can build on.
    """
    if server._inference_executor and not server._inference_executor.is_alive():
        return "inference process not running"

    if server._connection_failed:
        return "failed to connect to livekit"

    return None


def _claimed(app: FastAPI, method: str, path: str) -> bool:
    return any(
        getattr(route, "path", None) == path and method in (getattr(route, "methods", None) or ())
        for route in app.routes
    )


def _proxied_endpoints(app: FastAPI) -> list[str]:
    """The first path segment of every route, which is what names it to the cloud.

    A route at ``/`` names nothing and stays local, and so does one whose first segment is
    a parameter.
    """
    endpoints: list[str] = []
    for route in app.routes:
        path: str = getattr(route, "path", "")
        first = path.lstrip("/").split("/", 1)[0]
        if not first or first in endpoints:
            continue
        if not _URL_SAFE.fullmatch(first):
            logger.warning(f"route {path!r} cannot be reached through the cloud")
            continue
        endpoints.append(first)
    return endpoints


def _register_builtin_routes(server: AgentServer) -> None:
    """Add the health and worker info routes to ``server.http``.

    Called after the user's decorators, so a health check of theirs keeps ``GET /``.
    """
    app = server.http

    async def health() -> Response:
        reason = agent_health(server)
        if reason is None:
            return Response("OK", media_type="text/plain")
        return Response(reason, status_code=503, media_type="text/plain")

    async def worker_info() -> Response:
        from .worker import WORKER_PROTOCOL_VERSION  # deferred: worker imports this module

        info = agent_worker.WorkerInfo(
            worker_type=agent.JobType.Name(server._server_type.value),
            agent_name=server._agent_name,
            active_jobs=len(server.active_jobs),
            sdk_version=__version__,
            worker_load=server._worker_load,
            protocol_version=WORKER_PROTOCOL_VERSION,
        )
        return Response(
            MessageToJson(info, preserving_proto_field_name=True),
            media_type="application/json",
        )

    if not _claimed(app, "GET", "/"):
        app.add_api_route("/", health, methods=["GET"])

    # a protocol contract, unlike the health check: the control plane and lk CLI read it
    if _claimed(app, "GET", "/worker"):
        raise ValueError(
            "'GET /worker' is reserved by the agent server and cannot be registered on server.http"
        )
    app.add_api_route("/worker", worker_info, methods=["GET"])


class _UvicornServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        # the CLI owns SIGINT and SIGTERM so it can drain jobs; uvicorn would replace them
        yield


class _HttpRunner:
    """Serves an ASGI app for the lifetime of the agent server."""

    def __init__(self, app: FastAPI, *, host: str, port: int) -> None:
        self._app = app
        self._host = host
        self._port = port
        self._server: _UvicornServer | None = None
        self._serve_task: asyncio.Task[None] | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_config=None,  # keep the logging the CLI already set up
            log_level="warning",  # uvicorn's own loggers; its banner duplicates ours
            access_log=False,
        )
        self._server = _UvicornServer(config)
        self._serve_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            if self._serve_task.done():
                self._serve_task.result()  # re-raise whatever stopped it
                raise RuntimeError("HTTP server stopped before it finished starting")
            await asyncio.sleep(0.01)

        # an empty host binds one socket per address family, each with its own port
        socks = [sock for server in self._server.servers for sock in server.sockets]
        if socks:
            chosen = next((s for s in socks if s.family == socket.AF_INET), socks[0])
            self._port = chosen.getsockname()[1]

    async def aclose(self) -> None:
        if self._server is None or self._serve_task is None:
            return

        self._server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await self._serve_task

        self._server = None
        self._serve_task = None
