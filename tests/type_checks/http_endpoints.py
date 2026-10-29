from typing import assert_type

from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.websockets import WebSocket

from livekit.agents.http import AgentHttp


def endpoint_signatures(http: AgentHttp) -> None:
    @http.get("/hello", name="hello")
    async def hello(request: Request) -> PlainTextResponse:
        return PlainTextResponse(request.url.path)

    async def use_hello(request: Request) -> None:
        assert_type(await hello(request), PlainTextResponse)

    @http.websocket("/echo")
    async def echo(socket: WebSocket) -> None:
        await socket.accept()

    async def use_echo(socket: WebSocket) -> None:
        assert_type(await echo(socket), None)
