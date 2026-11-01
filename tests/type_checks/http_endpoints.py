from typing import assert_type

from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.websockets import WebSocket

from livekit.agents._generated.worker_endpoint import EndpointKind, HttpScheme
from livekit.agents._generated.worker_server import EndpointRoute
from livekit.agents.http import AgentRoutes, EndpointRequest


async def endpoint_signatures(http: AgentRoutes, request: Request, socket: WebSocket) -> None:
    @http.get("/hello", name="hello")
    async def hello(request: Request) -> PlainTextResponse:
        return PlainTextResponse(request.url.path)

    assert_type(await hello(request), PlainTextResponse)

    @http.websocket("/echo")
    async def echo(socket: WebSocket) -> None:
        await socket.accept()

    assert_type(await echo(socket), None)


def endpoint_metadata(http: AgentRoutes) -> None:
    routes = http.manifest()
    assert_type(routes, list[EndpointRoute])
    assert_type(routes[0].kind, EndpointKind)


def request_metadata(request: EndpointRequest) -> None:
    assert_type(request.kind, EndpointKind)
    assert_type(request.scheme, HttpScheme)
