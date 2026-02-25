from __future__ import annotations

import asyncio

from aiohttp import web


class HttpServer:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    @property
    def app(self) -> web.Application:
        return self._app

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        if self._port == 0:
            server = self._site._server
            if isinstance(server, asyncio.Server) and server.sockets:
                self._port = server.sockets[0].getsockname()[1]

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
