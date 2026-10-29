from typing import assert_type

from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.websockets import WebSocket

from livekit.agents.http import AgentRoutes


async def endpoint_signatures(http: AgentRoutes, request: Request, socket: WebSocket) -> None:
    @http.get("/hello", name="hello")
    async def hello(request: Request) -> PlainTextResponse:
        return PlainTextResponse(request.url.path)

    assert_type(await hello(request), PlainTextResponse)

    @http.websocket("/echo")
    async def echo(socket: WebSocket) -> None:
        await socket.accept()

    assert_type(await echo(socket), None)
