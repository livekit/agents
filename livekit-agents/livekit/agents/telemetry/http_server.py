from __future__ import annotations

import asyncio

import aiohttp.web_request

from .. import utils
from aiohttp import web

from prometheus_client import (
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
)


async def metrics(_request: aiohttp.web_request.Request):
    def _get_metrics():
        registry = CollectorRegistry(auto_describe=True)
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)

    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _get_metrics)
    return web.Response(
        body=data,
        headers={"Content-Type": CONTENT_TYPE_LATEST, "Content-Length": str(len(data))},
    )


class HttpServer(utils.http_server.HttpServer):
    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(host, port, loop)
        self._app.add_routes([web.get("/", metrics)])
