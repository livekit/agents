from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.web_runner import AppRunner, TCPSite


class HttpServer:
    def __init__(
        self, host: str, port: int, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._host = host
        self._port = port
        self._app = web.Application()
        self._lock = asyncio.Lock()
        self._runner: AppRunner | None = None
        self._site: TCPSite | None = None

    @property
    def app(self) -> web.Application:
        return self._app

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        async with self._lock:
            self._runner = AppRunner(self._app)
            await self._runner.setup()

            self._site = TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            if self._port == 0:
                for site in self._runner.sites:
                    if hasattr(site, "_server") and site._server.sockets:
                        self._port = site._server.sockets[0].getsockname()[1]
                        break

    async def aclose(self) -> None:
        async with self._lock:
            if self._site:
                await self._site.stop()
                self._site = None

            if self._runner:
                await self._runner.cleanup()
                self._runner = None
