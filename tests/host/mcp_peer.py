"""A real MCP SDK server with scripted tools, run in its own process over Streamable HTTP
(``serve_http``) or stdio (``stdio_command``).

Every request it handles is appended to ``journal.jsonl`` in its directory, which the test
reads. ``python mcp_peer.py stdio|http DIR [--port=] [--server=fastmcp] [--sse]`` runs it; over
HTTP it writes its URL to ``DIR/url`` and a line of its stdout once listening. The Rust lifecycle tests run it too.
``--server=fastmcp`` serves ``build_fastmcp``, realistic tools as a FastMCP app.

Tools, listed over two pages:

- ``echo(text)``: the text back.
- ``rich``: a text block and an image block.
- ``refuse``: an ``isError`` result, the failure the model is meant to read.
- ``empty``: a result without content.
- ``reject``: a JSON-RPC error (-32602, ``data={"field": "seat"}``).
- ``crash``: the handler raises with an internal detail.
- ``wait``: journals ``entered``, blocks until cancelled, then journals ``settled``.
- ``progress(steps, interval, gate)``: a progress message per step, then ``done``; with a
  ``gate``, journals ``holding`` after the first message and goes on once
  ``POST /release/<gate>`` (HTTP) opens it.
- ``meet(who, parties)``: answers ``met <who>`` once ``parties`` calls of it are in flight.
- ``slow(seconds)``: answers after ``seconds``, without progress.
- ``whoami``: the request's ``Authorization`` header.
- ``unicode``: a large non-ASCII text.
- ``environment``: ``PEER_FLAG`` from its environment and its working directory.
- ``exit``: the process exits mid-call (stdio).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import sys
from collections.abc import AsyncIterator
from enum import StrEnum
from pathlib import Path
from typing import Any

import anyio
from fastmcp import Context as FastMCPContext, FastMCP
from fastmcp.exceptions import ToolError as FastMCPToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools import ToolResult
from mcp import types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field

UNICODE = "Réservation confirmée ✈️ 東京 → Zürich, siège 12C. " * 2000
CRASH_DETAIL = "database password is hunter2"
"""What ``crash`` raises with; it must not reach the model."""

_OBJECT: dict[str, Any] = {"type": "object", "properties": {}}
PAGES: list[list[types.Tool]] = [
    [
        types.Tool(
            name="echo",
            description="Repeat a text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "What to repeat."}},
                "required": ["text"],
            },
        ),
        types.Tool(name="rich", description="A chart and its caption.", input_schema=_OBJECT),
        types.Tool(name="refuse", description="Refuse the booking.", input_schema=_OBJECT),
        types.Tool(name="empty", input_schema=_OBJECT),
        types.Tool(name="reject", input_schema=_OBJECT),
        types.Tool(name="crash", input_schema=_OBJECT),
    ],
    [
        types.Tool(name="wait", description="Never answers.", input_schema=_OBJECT),
        types.Tool(
            name="progress",
            input_schema={
                "type": "object",
                "properties": {
                    "steps": {"type": "integer"},
                    "interval": {"type": "number"},
                    "gate": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="meet",
            description="Answer once every party has called.",
            input_schema={
                "type": "object",
                "properties": {"who": {"type": "string"}, "parties": {"type": "integer"}},
            },
        ),
        types.Tool(
            name="slow",
            description="Answer after some seconds, silently.",
            input_schema={"type": "object", "properties": {"seconds": {"type": "number"}}},
        ),
        types.Tool(name="whoami", input_schema=_OBJECT),
        types.Tool(name="unicode", input_schema=_OBJECT),
        types.Tool(name="environment", input_schema=_OBJECT),
        types.Tool(name="exit", input_schema=_OBJECT),
    ],
]
NAMES = [tool.name for page in PAGES for tool in page]


class Journal:
    """The requests a peer handled, as JSON lines in ``directory``."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "journal.jsonl"

    def write(self, event: str, **fields: Any) -> None:
        line = json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\n"
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line)

    def entries(self, event: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line]
        return [e for e in entries if event is None or e["event"] == event]

    async def until(
        self, event: str, count: int = 1, *, timeout: float = 10
    ) -> list[dict[str, Any]]:
        """The ``event`` entries once there are ``count`` of them."""
        async with asyncio.timeout(timeout):
            while len(found := self.entries(event)) < count:
                await asyncio.sleep(0.01)
        return found


_gates: dict[str, anyio.Event] = {}


def _gate(name: str) -> anyio.Event:
    """The event ``POST /release/<name>`` sets, created by whichever side comes first."""
    if name not in _gates:
        _gates[name] = anyio.Event()
    return _gates[name]


def _text(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])


def build(journal: Journal) -> Server[Any]:
    met: list[str] = []

    async def list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        page = int(params.cursor) if params is not None and params.cursor else 0
        journal.write("list", page=page, **_request(ctx))
        more = page + 1 < len(PAGES)
        return types.ListToolsResult(tools=PAGES[page], next_cursor=str(page + 1) if more else None)

    async def call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        arguments = dict(params.arguments or {})
        journal.write("call", tool=params.name, arguments=arguments, **_request(ctx))
        match params.name:
            case "echo":
                return _text(str(arguments["text"]))
            case "rich":
                return types.CallToolResult(
                    content=[
                        types.TextContent(type="text", text="Weekly bookings"),
                        types.ImageContent(
                            type="image", data="iVBORw0KGgo=", mime_type="image/png"
                        ),
                    ]
                )
            case "refuse":
                return types.CallToolResult(
                    content=[
                        types.TextContent(type="text", text="That date is unavailable."),
                        types.TextContent(type="text", text="Try another day."),
                    ],
                    is_error=True,
                )
            case "empty":
                return types.CallToolResult(content=[])
            case "reject":
                raise MCPError(-32602, "invalid tool arguments", data={"field": "seat"})
            case "crash":
                raise RuntimeError(CRASH_DETAIL)
            case "wait":
                journal.write("entered", tool="wait")
                try:
                    await anyio.sleep_forever()
                finally:
                    journal.write("settled", tool="wait")
            case "progress":
                steps, interval = (
                    int(arguments.get("steps", 3)),
                    float(arguments.get("interval", 0)),
                )
                for step in range(1, steps + 1):
                    await anyio.sleep(interval)
                    await ctx.session.report_progress(step, steps, f"Step {step} of {steps}.")
                    if step == 1 and (gate := arguments.get("gate")):
                        journal.write("holding", tool="progress")
                        await _gate(str(gate)).wait()
                return _text("done")
            case "meet":
                met.append(str(arguments["who"]))
                if len(met) >= int(arguments["parties"]):
                    _gate("met").set()
                await _gate("met").wait()
                return _text(f"met {arguments['who']}")
            case "slow":
                await anyio.sleep(float(arguments["seconds"]))
                return _text(f"done after {arguments['seconds']}s")
            case "whoami":
                return _text(_request(ctx)["authorization"] or "anonymous")
            case "unicode":
                return _text(UNICODE)
            case "environment":
                flag = os.environ.get("PEER_FLAG", "")
                return _text(json.dumps({"flag": flag, "cwd": os.getcwd()}))
            case "exit":
                os._exit(3)
        raise MCPError(-32602, f"unknown tool {params.name}")

    return Server(
        "livekit-test-peer", version="1", on_list_tools=list_tools, on_call_tool=call_tool
    )


def _request(ctx: ServerRequestContext[Any]) -> dict[str, Any]:
    """Who sent a request: its HTTP session and credentials, or the stdio process."""
    headers = getattr(ctx.request, "headers", None)
    if headers is None:
        return {"connection": f"stdio-{os.getpid()}", "authorization": None, "headers": {}}
    return {
        "connection": headers.get("mcp-session-id"),
        "authorization": headers.get("authorization"),
        "headers": {name: value for name, value in headers.items() if name.startswith("x-")},
    }


class Cabin(StrEnum):
    ECONOMY = "economy"
    BUSINESS = "business"


class Passenger(BaseModel):
    name: str
    age: int | None = None


class FlightQuery(BaseModel):
    origin: str = Field(description="IATA code of the departure airport.")
    destination: str
    cabin: Cabin = Cabin.ECONOMY
    passengers: list[Passenger]


class Flight(BaseModel):
    number: str
    seats_left: int


class Confirmation(BaseModel):
    confirmed: bool


def build_fastmcp(journal: Journal) -> FastMCP:
    """The same journal behind a FastMCP app, as most real servers are built: Pydantic
    arguments, structured output, annotations, ``ToolError``, logging, elicitation, rich
    content, result metadata and tools enabled at runtime."""
    server = FastMCP("livekit-test-fastmcp")

    def journal_call(tool: str, **arguments: Any) -> None:
        headers = get_http_headers(include_all=True)
        journal.write(
            "call",
            tool=tool,
            arguments=arguments,
            connection=headers.get("mcp-session-id"),
            authorization=headers.get("authorization"),
        )

    @server.tool(title="Search flights")
    async def search_flights(query: FlightQuery) -> list[Flight]:
        """Find flights for the passengers."""
        journal_call("search_flights", query=query.model_dump(mode="json"))
        return [Flight(number=f"LK{len(query.passengers)}01", seats_left=4)]

    @server.tool(annotations={"destructiveHint": True, "readOnlyHint": False})
    async def cancel_booking(reference: str) -> str:
        """Cancel a booking."""
        journal_call("cancel_booking", reference=reference)
        if reference == "FLOWN":
            raise FastMCPToolError("Booking FLOWN has already been flown.")
        if reference == "BUG":
            raise KeyError(CRASH_DETAIL)
        return f"Cancelled {reference}."

    @server.tool
    async def quote(nights: int, ctx: FastMCPContext) -> str:
        """Quote a stay, logging and reporting progress as it goes."""
        journal_call("quote", nights=nights)
        await ctx.info("Pricing the stay.")
        for night in range(1, nights + 1):
            await ctx.report_progress(night, nights, f"Priced night {night} of {nights}.")
        return f"{nights} nights: {nights * 120} EUR"

    @server.tool
    async def confirm_seat(seat: str, ctx: FastMCPContext) -> str:
        """Ask the person to confirm a seat."""
        journal_call("confirm_seat", seat=seat)
        answer = await ctx.elicit(f"Confirm seat {seat}?", Confirmation)
        return f"Seat {seat}: {answer.action}"

    @server.tool
    async def itinerary() -> ToolResult:
        """The itinerary as a document, a link and an audio note, with result metadata."""
        journal_call("itinerary")
        return ToolResult(
            content=[
                types.EmbeddedResource(
                    type="resource",
                    resource=types.TextResourceContents(
                        uri="booking://ABC/itinerary", mime_type="text/plain", text="Paris -> Tokyo"
                    ),
                ),
                types.ResourceLink(
                    type="resource_link", uri="booking://ABC/receipt", name="receipt"
                ),
                types.AudioContent(type="audio", data="UklGRg==", mime_type="audio/wav"),
            ],
            meta={"trace": "t-1"},
        )

    @server.tool(tags={"upgrades"})
    def upgrade(reference: str) -> str:
        """Upgrade a booking."""
        journal_call("upgrade", reference=reference)
        return f"Upgraded {reference}."

    server.disable(tags={"upgrades"})

    @server.tool
    async def unlock_upgrades(ctx: FastMCPContext) -> str:
        """Enable the upgrade tool; the server tells the client its tools changed."""
        journal_call("unlock_upgrades")
        server.enable(tags={"upgrades"})
        await ctx.send_notification(types.ToolListChangedNotification())
        return "Upgrades unlocked."

    return server


def _listen(port: int = 0) -> socket.socket:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen()
    return sock


class HttpPeer:
    """The peer in its own process. ``stop()`` kills it, as a crash would; ``start()`` serves
    again on the same port, without the sessions it had."""

    def __init__(self, directory: Path, *, server: str, json_response: bool) -> None:
        self.journal = Journal(directory)
        self._directory = directory
        self._options = [f"--server={server}"] + ([] if json_response else ["--sse"])
        self._port = 0
        self._process: asyncio.subprocess.Process | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/mcp"

    async def start(self) -> None:
        (self._directory / "url").unlink(missing_ok=True)
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "http",
            str(self._directory),
            f"--port={self._port}",
            *self._options,
            stdout=asyncio.subprocess.PIPE,
        )
        assert self._process.stdout is not None
        async with asyncio.timeout(20):
            listening = await self._process.stdout.readline()  # its URL, once listening
        if not listening:
            raise RuntimeError(f"the MCP peer exited with {await self._process.wait()}")
        self._port = int(listening.decode().rsplit(":", 1)[1].split("/")[0])

    async def release(self, gate: str) -> None:
        """Opens ``gate``: a ``progress`` call holding on it answers."""
        reader, writer = await asyncio.open_connection("127.0.0.1", self._port)
        writer.write(f"POST /release/{gate} HTTP/1.1\r\nHost: peer\r\n\r\n".encode())
        await writer.drain()
        assert (await reader.readline()).startswith(b"HTTP/1.1 200")
        writer.close()
        await writer.wait_closed()

    async def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()


@contextlib.asynccontextmanager
async def serve_http(
    directory: Path, *, server: str = "lowlevel", json_response: bool = True
) -> AsyncIterator[HttpPeer]:
    """``server="fastmcp"`` serves ``build_fastmcp`` instead of the scripted tools."""
    peer = HttpPeer(directory, server=server, json_response=json_response)
    try:
        await peer.start()
        yield peer
    finally:
        await peer.stop()


def stdio_command(directory: Path) -> tuple[str, list[str]]:
    """The ``MCPServerStdio(command, args)`` that runs this peer."""
    return sys.executable, [str(Path(__file__).resolve()), "stdio", str(directory)]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stdio", "http"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--server", choices=["lowlevel", "fastmcp"], default="lowlevel")
    parser.add_argument("--sse", action="store_true", help="reply over SSE instead of JSON")
    options = parser.parse_args()
    journal = Journal(options.directory)
    (options.directory / "pid").write_text(str(os.getpid()))
    if options.mode == "stdio":
        from mcp.server.stdio import stdio_server

        async def stdio() -> None:
            server = build(journal)
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        anyio.run(stdio)
        return
    import uvicorn
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    async def release(request: Request) -> Response:
        _gate(request.path_params["gate"]).set()
        return Response()

    sock = _listen(options.port)
    if options.server == "lowlevel":
        app = build(journal).streamable_http_app(json_response=not options.sse)
        app.router.routes.insert(0, Route("/release/{gate}", release, methods=["POST"]))
    else:
        app = build_fastmcp(journal).http_app(path="/mcp", json_response=not options.sse)
    config = uvicorn.Config(app, log_level="critical")
    server = uvicorn.Server(config)
    # written once listening, so a reader of the URL can connect at once
    url = options.directory / "url"
    port = sock.getsockname()[1]

    async def serve() -> None:
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            await asyncio.sleep(0.01)
        url.with_suffix(".tmp").write_text(f"http://127.0.0.1:{port}/mcp")
        url.with_suffix(".tmp").rename(url)
        print(f"http://127.0.0.1:{port}/mcp", flush=True)
        await task

    asyncio.run(serve())


if __name__ == "__main__":
    _main()
