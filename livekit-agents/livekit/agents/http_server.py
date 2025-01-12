from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web


class HttpServer:
    def __init__(
        self, host: str, port: int, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._host = host
        self._port = port
        self._app = web.Application(loop=self._loop)
        self._close_future = asyncio.Future[None](loop=self._loop)

    @property
    def app(self) -> web.Application:
        return self._app

    async def run(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        try:
            await self._close_future
        finally:
            await self._runner.cleanup()

    async def aclose(self) -> None:
        if not self._close_future.done():
            self._close_future.set_result(None)
