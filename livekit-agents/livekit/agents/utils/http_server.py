from __future__ import annotations

from aiohttp import web


class HttpServer:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None

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
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        if self._port == 0:
            address = self._runner.addresses
            if address:
                self._port = address[0][1]

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
