from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from aiohttp import web


async def health_check(_: Any):
    return web.Response(text="OK")


class HttpServer:
    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop) -> None:
        self._host, self._port, self._loop = host, port, loop
        self._app = web.Application(loop=self._loop)
        self._close_future = asyncio.Future[None](loop=self._loop)

    @property
    def app(self) -> web.Application:
        return self._app

    async def run(self) -> None:
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()

        try:
            await self._close_future
        finally:
            await runner.cleanup()

    async def aclose(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self._close_future.set_result(None)
