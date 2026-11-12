"""MCP toolsets against real MCP servers (``mcp_peer``: the SDK's low-level server and a FastMCP
app), through a ``TextSession`` turn and a ``VoiceSession`` reply: what the model is offered
and reads back, slow tools, transports, credentials, outages, timeouts and cancellation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from livekit.agents import Agent, RunCompleted, RunResult, TextSession, VoiceSession, _core
from livekit.agents.llm import ChatItem, RawFunctionTool, Toolset
from livekit.agents.llm.mcp import MCPServerHTTP, MCPServerStdio, MCPToolOptions, MCPToolset
from livekit.agents.voice.events import ToolCallUpdated, ToolExecutionUpdatedEvent
from tests_new.fakes import FakeAudioOutput, FakeLLM, FakeTranscriptionOutput, FakeTTS, tool_call

from .mcp_peer import CRASH_DETAIL, NAMES, UNICODE, HttpPeer, Journal, serve_http, stdio_command

INTERNAL = "An internal error occurred"


# --- Surfaces --------------------------------------------------------------------------------


@dataclass
class Reply:
    output: object
    history: list[ChatItem]

    def outputs(self) -> list[Output]:
        return [
            Output(item.name, item.is_error, item.output)
            for item in self.history
            if item.type == "function_call_output"
        ]


@dataclass
class Output:
    name: str
    is_error: bool
    text: str


async def text(agent: Agent[Any, Any], *messages: str, context: Any = None) -> Reply:
    """Each message a ``TextSession`` turn, continuing the last one's state."""
    state = None
    for message in messages:
        turn = await TextSession(agent=agent).run(message, state=state, context=context)
        state = turn.state
    return _reply(turn.result)


async def voice(agent: Agent[Any, Any], *messages: str, context: Any = None) -> Reply:
    """Each message a reply on one ``VoiceSession``."""
    async with VoiceSession(context=context) as session:
        session.output.transcription = FakeTranscriptionOutput()
        await session.start(agent)
        for message in messages:
            result = await session._run(user_input=message)
    return _reply(result)


def _reply(result: RunResult[Any]) -> Reply:
    output = result.output if isinstance(result, RunCompleted) else None
    return Reply(output, result.all_messages())


Surface = Callable[..., Awaitable[Reply]]
SURFACES = [pytest.param(text, id="text"), pytest.param(voice, id="voice")]


def agent_with(model: FakeLLM, *toolsets: Toolset) -> Agent[Any, Any]:
    return Agent(instructions="Help.", llm=model, tools=list(toolsets))


def calling(*calls: tuple[str, dict[str, Any]]) -> FakeLLM:
    """A model that calls the tools together in one step, then answers."""
    return FakeLLM([[tool_call(name, **arguments) for name, arguments in calls], "Done."])


def offered(model: FakeLLM) -> list[str]:
    return [tool.id for tool in model.tools[0]]


def schema(model: FakeLLM, name: str) -> dict[str, Any]:
    """The schema the model was offered for ``name``."""
    [tool] = [tool for tool in model.tools[0] if tool.id == name]
    assert isinstance(tool, RawFunctionTool)
    return dict(tool.info.raw_schema)


# --- Peers -----------------------------------------------------------------------------------


@pytest.fixture
async def peer(tmp_path: Path) -> AsyncIterator[HttpPeer]:
    async with serve_http(tmp_path / "peer") as served:
        yield served


@pytest.fixture
async def streaming_peer(tmp_path: Path) -> AsyncIterator[HttpPeer]:
    """Replies over SSE, the only way a Streamable HTTP server can send progress."""
    async with serve_http(tmp_path / "peer", json_response=False) as served:
        yield served


@pytest.fixture
async def fastmcp(tmp_path: Path) -> AsyncIterator[HttpPeer]:
    """Replies over SSE, FastMCP's default; a JSON reply cannot carry its notifications."""
    async with serve_http(tmp_path / "peer", server="fastmcp", json_response=False) as served:
        yield served


@pytest.fixture
async def silent_url() -> AsyncIterator[str]:
    """A server that accepts connections and never answers."""
    held: list[asyncio.StreamWriter] = []

    async def hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)

    listener = await asyncio.start_server(hold, "127.0.0.1", 0)
    yield f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}/mcp"
    for writer in held:
        writer.close()
    listener.close()
    await listener.wait_closed()


# --- Discovery -------------------------------------------------------------------------------


async def test_the_model_is_offered_every_page_of_tools_with_their_schemas(peer: HttpPeer) -> None:
    model = FakeLLM(["Hello."])
    await text(agent_with(model, MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))), "hi")

    assert offered(model) == NAMES
    echo = schema(model, "echo")
    assert echo == {
        "name": "echo",
        "description": "Repeat a text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "What to repeat."}},
            "required": ["text"],
        },
    }


async def test_a_tool_without_a_description_is_offered(peer: HttpPeer) -> None:
    """MCP makes ``description`` optional; one undescribed tool failed every run with
    ``failed to extract field FunctionToolInfo.spec``."""
    model = calling(("empty", {}))
    reply = await text(
        agent_with(model, MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))), "go"
    )

    empty = schema(model, "empty")
    assert empty["description"] == ""
    assert reply.output == "Done."


async def test_filtered_hides_tools_from_the_model(peer: HttpPeer) -> None:
    model = FakeLLM(["Hello."])
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))
    await text(agent_with(model, toolset.filtered(lambda tool: tool.id == "echo")), "hi")
    assert offered(model) == ["echo"]


async def test_a_fastmcp_pydantic_schema_reaches_the_model_and_its_arguments_the_server(
    fastmcp: HttpPeer,
) -> None:
    query = {
        "origin": "CDG",
        "destination": "HND",
        "cabin": "business",
        "passengers": [{"name": "Ada", "age": 36}, {"name": "Alan"}],
    }
    model = calling(("search_flights", {"query": query}))
    toolset = MCPToolset(id="flights", mcp_server=MCPServerHTTP(fastmcp.url))
    reply = await text(agent_with(model, toolset), "go")

    search = schema(model, "search_flights")
    cabin = search["parameters"]["properties"]["query"]["properties"]["cabin"]
    assert cabin["enum"] == ["economy", "business"]
    [call] = fastmcp.journal.entries("call")
    alan = {"name": "Alan", "age": None}
    assert call["arguments"] == {"query": {**query, "passengers": [query["passengers"][0], alan]}}
    [found] = reply.outputs()
    assert not found.is_error
    assert json.loads(found.text) == [{"number": "LK201", "seats_left": 4}]


# --- What the model reads back ---------------------------------------------------------------


@pytest.mark.parametrize("converse", SURFACES)
async def test_results_reach_the_model_as_mcp_intends(converse: Surface, peer: HttpPeer) -> None:
    """A result's content reaches the model; an ``isError`` result is the failure the server
    meant for it; a protocol error or a crashed handler stays an internal error."""
    model = calling(
        ("echo", {"text": "Seat 12C."}),
        ("rich", {}),
        ("refuse", {}),
        ("empty", {}),
        ("reject", {}),
        ("crash", {}),
        ("unicode", {}),
    )
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))
    reply = await converse(agent_with(model, toolset), "go")

    assert reply.output == "Done."
    by_tool = {output.name: output for output in reply.outputs()}
    assert (by_tool["echo"].is_error, by_tool["echo"].text) == (False, "Seat 12C.")
    assert json.loads(by_tool["rich"].text) == [
        {"type": "text", "text": "Weekly bookings"},
        {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
    ]
    assert (by_tool["refuse"].is_error, by_tool["refuse"].text) == (
        True,
        "That date is unavailable.\nTry another day.",
    )
    assert (by_tool["empty"].is_error, by_tool["empty"].text) == (
        True,
        "Tool 'empty' completed without producing a result",
    )
    assert (by_tool["reject"].is_error, by_tool["reject"].text) == (True, INTERNAL)
    assert (by_tool["crash"].is_error, by_tool["crash"].text) == (True, INTERNAL)
    assert by_tool["unicode"].text == UNICODE
    echoed = next(call for call in peer.journal.entries("call") if call["tool"] == "echo")
    assert echoed["arguments"] == {"text": "Seat 12C."}


@pytest.mark.parametrize("converse", SURFACES)
async def test_fastmcp_results_reach_the_model_as_mcp_intends(
    converse: Surface, fastmcp: HttpPeer
) -> None:
    """A FastMCP ``ToolError`` and an argument the server's schema rejects are for the model.
    An unexpected exception is an ``isError`` result too; FastMCP includes its message unless
    the server sets ``mask_error_details``, and the client cannot tell the two apart."""
    model = calling(
        ("cancel_booking", {"reference": "ABC"}),
        ("cancel_booking", {"reference": "FLOWN"}),
        ("cancel_booking", {"reference": "BUG"}),
        ("cancel_booking", {"reference": 12}),
        ("itinerary", {}),
    )
    toolset = MCPToolset(id="flights", mcp_server=MCPServerHTTP(fastmcp.url))
    reply = await converse(agent_with(model, toolset), "go")

    cancelled, flown, bug, invalid, itinerary = reply.outputs()
    assert (cancelled.is_error, cancelled.text) == (False, "Cancelled ABC.")
    assert (flown.is_error, flown.text) == (True, "Booking FLOWN has already been flown.")
    assert bug.is_error and CRASH_DETAIL in bug.text
    assert invalid.is_error and "reference" in invalid.text
    assert [block["type"] for block in json.loads(itinerary.text)] == [
        "resource",
        "resource_link",
        "audio",
    ]
    assert reply.output == "Done."


@pytest.mark.parametrize("converse", SURFACES)
async def test_an_elicitation_is_declined_and_the_call_completes(
    converse: Surface, fastmcp: HttpPeer
) -> None:
    """Elicitation is not supported: the client advertises no capability, and a server that
    asks anyway is answered ``decline`` at once instead of holding the call."""
    model = calling(("confirm_seat", {"seat": "12C"}))
    toolset = MCPToolset(id="flights", mcp_server=MCPServerHTTP(fastmcp.url))
    async with asyncio.timeout(5):
        reply = await converse(agent_with(model, toolset), "go")
    [confirm] = reply.outputs()
    assert (confirm.is_error, confirm.text) == (False, "Seat 12C: decline")
    assert reply.output == "Done."


async def test_tools_a_server_enables_are_offered_from_the_next_turn(fastmcp: HttpPeer) -> None:
    """FastMCP announces enabled tools with ``tools/list_changed``; the toolset lists again for
    the next turn instead of waiting for its idle connection to close."""
    toolset = MCPToolset(id="flights", mcp_server=MCPServerHTTP(fastmcp.url))
    unlocking = calling(("unlock_upgrades", {}))
    await text(agent_with(unlocking, toolset), "unlock upgrades")
    assert "upgrade" not in offered(unlocking)

    upgrading = calling(("upgrade", {"reference": "ABC"}))
    reply = await text(agent_with(upgrading, toolset), "upgrade ABC")
    assert "upgrade" in offered(upgrading)
    assert [output.text for output in reply.outputs()] == ["Upgraded ABC."]


@pytest.mark.parametrize("json_response", [True, False], ids=["json", "sse"])
async def test_both_streamable_http_reply_modes_answer(tmp_path: Path, json_response: bool) -> None:
    async with serve_http(tmp_path, json_response=json_response) as served:
        model = calling(("echo", {"text": "hi"}))
        toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(served.url))
        reply = await text(agent_with(model, toolset), "go")
    assert [output.text for output in reply.outputs()] == ["hi"]


# --- Slow tools ------------------------------------------------------------------------------


@pytest.mark.parametrize("converse", SURFACES)
async def test_slow_calls_in_one_step_run_concurrently(converse: Surface, peer: HttpPeer) -> None:
    model = calling(("slow", {"seconds": 1}), ("slow", {"seconds": 1.2}))
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))
    started = time.monotonic()
    reply = await converse(agent_with(model, toolset), "go")

    assert time.monotonic() - started < 2, "the second call waited for the first"
    assert [output.text for output in reply.outputs()] == ["done after 1s", "done after 1.2s"]
    assert reply.output == "Done."


@pytest.mark.parametrize("converse", SURFACES)
async def test_a_call_slower_than_its_timeout_fails_and_the_reply_goes_on(
    converse: Surface, peer: HttpPeer
) -> None:
    model = calling(("slow", {"seconds": 30}))
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url, request_timeout=0.5))
    started = time.monotonic()
    reply = await converse(agent_with(model, toolset), "go")

    assert time.monotonic() - started < 3
    [slow] = reply.outputs()
    assert (slow.is_error, slow.text) == (True, INTERNAL)
    assert reply.output == "Done."


async def test_a_tool_timeout_replaces_the_request_timeout(peer: HttpPeer) -> None:
    model = calling(("wait", {}))
    toolset = MCPToolset(
        id="peer",
        mcp_server=MCPServerHTTP(peer.url, request_timeout=None),
        tool_options={"wait": MCPToolOptions(timeout=0.5)},
    )
    async with asyncio.timeout(5):
        reply = await text(agent_with(model, toolset), "go")
    [waited] = reply.outputs()
    assert waited.is_error
    await peer.journal.until("settled", timeout=5)


async def test_progress_keeps_a_slow_call_alive_past_its_timeout(streaming_peer: HttpPeer) -> None:
    model = calling(("progress", {"steps": 4, "interval": 0.3}))
    toolset = MCPToolset(
        id="peer", mcp_server=MCPServerHTTP(streaming_peer.url, request_timeout=0.5)
    )
    reply = await text(agent_with(model, toolset), "go")
    [progress] = reply.outputs()
    assert (progress.is_error, progress.text) == (False, "done")


async def test_a_text_turn_waits_for_a_reporting_tool_to_finish(streaming_peer: HttpPeer) -> None:
    model = calling(("progress", {"steps": 3, "interval": 0.2}))
    toolset = MCPToolset(
        id="peer",
        mcp_server=MCPServerHTTP(streaming_peer.url),
        tool_options={"progress": MCPToolOptions(report_progress=True)},
    )
    reply = await text(agent_with(model, toolset), "go")
    assert [output.text for output in reply.outputs()] == ["done"]
    assert reply.output == "Done."


async def test_on_a_call_a_reporting_tool_speaks_its_progress_then_its_result(
    streaming_peer: HttpPeer,
) -> None:
    """The person hears the agent while a slow tool works: each progress message is a tool
    update, and the result reaches the model when the tool finishes."""
    model = FakeLLM(
        [tool_call("progress", steps=2, interval=0.3), "Working on it.", "Still on it.", "Done."]
    )
    toolset = MCPToolset(
        id="peer",
        mcp_server=MCPServerHTTP(streaming_peer.url),
        tool_options={"progress": MCPToolOptions(report_progress=True)},
    )
    updates: list[str] = []

    def on_update(event: ToolExecutionUpdatedEvent) -> None:
        if isinstance(event.update, ToolCallUpdated):
            updates.append(event.update.message)

    async with VoiceSession(llm=model) as session:
        session.output.transcription = FakeTranscriptionOutput()
        session.on("tool_execution_updated", on_update)
        await session.start(Agent(instructions="Help.", tools=[toolset]))
        started = time.monotonic()
        first = await session._run(user_input="quote my stay")
        answered = time.monotonic() - started
        async with asyncio.timeout(5):
            while not any(
                item.type == "function_call_output" and item.output == "done"
                for item in session.history.items
            ):
                await asyncio.sleep(0.02)

    assert answered < 0.6, "the first reply waited for the slow tool"
    assert _reply(first).output == "Working on it."
    assert updates == ["Step 1 of 2.", "Step 2 of 2."]


# --- stdio -----------------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_a_stdio_server_runs_with_its_env_and_cwd_and_is_reaped_on_close(
    tmp_path: Path,
) -> None:
    command, args = stdio_command(tmp_path)
    server = MCPServerStdio(command, args, env={"PEER_FLAG": "on"}, cwd=str(tmp_path))
    model = calling(("environment", {}))
    reply = await text(agent_with(model, MCPToolset(id="local", mcp_server=server)), "go")

    [environment] = reply.outputs()
    assert json.loads(environment.text) == {"flag": "on", "cwd": str(tmp_path.resolve())}
    pid = int((tmp_path / "pid").read_text())
    assert _alive(pid)
    await server.aclose()
    assert not _alive(pid)


@pytest.mark.parametrize("converse", SURFACES)
async def test_a_stdio_server_that_exits_mid_call_fails_that_call_and_restarts(
    converse: Surface, tmp_path: Path
) -> None:
    command, args = stdio_command(tmp_path)
    server = MCPServerStdio(command, args)
    model = FakeLLM([tool_call("exit"), tool_call("echo", text="still here"), "Done."])
    try:
        async with asyncio.timeout(10):
            reply = await converse(
                agent_with(model, MCPToolset(id="local", mcp_server=server)), "go"
            )
    finally:
        await server.aclose()

    exited, echoed = reply.outputs()
    assert (exited.is_error, exited.text) == (True, INTERNAL)
    assert (echoed.is_error, echoed.text) == (False, "still here")
    first, second = (call["pid"] for call in Journal(tmp_path).entries("call"))
    assert first != second


# --- Credentials -----------------------------------------------------------------------------


async def test_static_headers_and_token_reach_the_server(peer: HttpPeer) -> None:
    server = MCPServerHTTP(
        peer.url, headers={"X-Tenant": "acme", "Accept": "text/plain"}, auth="s3cret-token"
    )
    model = calling(("whoami", {}))
    reply = await text(agent_with(model, MCPToolset(id="peer", mcp_server=server)), "go")

    # a reserved header never replaces the transport's own: the call still works
    assert [output.text for output in reply.outputs()] == ["Bearer s3cret-token"]
    for request in peer.journal.entries():
        assert request["authorization"] == "Bearer s3cret-token"
        assert request["headers"] == {"x-tenant": "acme"}
    assert "s3cret-token" not in repr(server)


@dataclass
class Member:
    token: str


def per_member(url: str) -> MCPToolset:
    server = MCPServerHTTP(url, auth=lambda ctx: ctx.context.token)
    return MCPToolset(id="peer", mcp_server=server)


async def test_turns_with_different_tokens_never_share_a_connection(
    peer: HttpPeer, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    toolset = per_member(peer.url)

    async def whoami(token: str) -> str:
        reply = await text(
            agent_with(calling(("whoami", {})), toolset), "go", context=Member(token)
        )
        [output] = reply.outputs()
        return output.text

    assert list(await asyncio.gather(whoami("alice"), whoami("bob"), whoami("alice"))) == [
        "Bearer alice",
        "Bearer bob",
        "Bearer alice",
    ]
    connections: dict[str, set[str]] = {}
    for call in peer.journal.entries("call"):
        connections.setdefault(call["authorization"], set()).add(call["connection"])
    assert len(connections["Bearer alice"]) == 1, "equal credentials share one connection"
    assert connections["Bearer alice"].isdisjoint(connections["Bearer bob"])
    assert "alice" not in caplog.text and "bob" not in caplog.text


async def test_a_filtered_toolset_with_per_run_tokens_filters_each_run(peer: HttpPeer) -> None:
    only_whoami = per_member(peer.url).filtered(lambda tool: tool.id == "whoami")

    async def whoami(token: str) -> tuple[list[str], str]:
        model = calling(("whoami", {}))
        reply = await text(agent_with(model, only_whoami), "go", context=Member(token))
        [output] = reply.outputs()
        return offered(model), output.text

    assert list(await asyncio.gather(whoami("alice"), whoami("bob"))) == [
        (["whoami"], "Bearer alice"),
        (["whoami"], "Bearer bob"),
    ]


async def test_a_token_refreshed_during_a_call_is_used_for_the_next_request(
    peer: HttpPeer,
) -> None:
    member = Member("first")
    model = FakeLLM([tool_call("whoami"), "Done.", tool_call("whoami"), "Done again."])
    async with VoiceSession(context=member) as session:
        session.output.transcription = FakeTranscriptionOutput()
        await session.start(agent_with(model, per_member(peer.url)))
        await session._run(user_input="who am I")
        member.token = "refreshed"
        await session._run(user_input="and now")

    assert [call["authorization"] for call in peer.journal.entries("call")] == [
        "Bearer first",
        "Bearer refreshed",
    ]


# --- Outages ---------------------------------------------------------------------------------


@pytest.mark.parametrize("converse", SURFACES)
async def test_an_unreachable_server_leaves_the_turn_without_its_tools_until_it_answers(
    converse: Surface, peer: HttpPeer
) -> None:
    await peer.stop()
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))

    model = FakeLLM(["I can't check that right now."])
    reply = await converse(agent_with(model, toolset), "go")
    assert reply.output == "I can't check that right now."
    assert offered(model) == []

    await peer.start()
    async with asyncio.timeout(10):
        while True:
            model = FakeLLM(["Hello."])
            await converse(agent_with(model, toolset), "hi")
            if offered(model):
                break
            await asyncio.sleep(0.1)
    assert offered(model) == NAMES


@pytest.mark.parametrize("converse", SURFACES)
async def test_an_outage_after_listing_fails_calls_and_recovers(
    converse: Surface, peer: HttpPeer
) -> None:
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))
    await converse(agent_with(FakeLLM(["Hello."]), toolset), "hi")

    await peer.stop()
    model = calling(("echo", {"text": "down"}), ("echo", {"text": "still down"}))
    async with asyncio.timeout(5):
        reply = await converse(agent_with(model, toolset), "go")
    assert offered(model) == NAMES, "listed tools stay offered through the outage"
    assert [output.is_error for output in reply.outputs()] == [True, True]
    assert reply.output == "Done."

    await peer.start()
    async with asyncio.timeout(10):
        while True:
            reply = await converse(agent_with(calling(("echo", {"text": "back"})), toolset), "go")
            [output] = reply.outputs()
            if not output.is_error:
                break
            await asyncio.sleep(0.2)
    assert output.text == "back"


async def test_a_server_answering_garbage_leaves_the_turn_without_its_tools() -> None:
    async def garbage(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        body = b"this is not JSON-RPC"
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    listener = await asyncio.start_server(garbage, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}/mcp"
    try:
        model = FakeLLM(["Hello."])
        toolset = MCPToolset(id="broken", mcp_server=MCPServerHTTP(url, init_timeout=2))
        async with asyncio.timeout(10):
            reply = await text(agent_with(model, toolset), "go")
    finally:
        listener.close()
        await listener.wait_closed()
    assert reply.output == "Hello."
    assert offered(model) == []


async def test_init_timeout_bounds_setup_against_a_silent_server(silent_url: str) -> None:
    toolset = MCPToolset(id="silent", mcp_server=MCPServerHTTP(silent_url, init_timeout=1))
    started = time.monotonic()
    await toolset.setup()
    elapsed = time.monotonic() - started
    await toolset.aclose()
    assert toolset.tools == []
    assert elapsed < 2, f"setup took {elapsed:.1f}s with init_timeout=1"


async def test_a_cancelled_setup_is_cancelled(silent_url: str) -> None:
    toolset = MCPToolset(id="silent", mcp_server=MCPServerHTTP(silent_url, init_timeout=30))
    setup = asyncio.create_task(toolset.setup())
    await asyncio.sleep(0.2)
    setup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await setup


# --- Cancellation ----------------------------------------------------------------------------


async def test_a_cancelled_text_turn_cancels_its_call_on_the_server(peer: HttpPeer) -> None:
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url, request_timeout=None))
    turn = asyncio.ensure_future(text(agent_with(calling(("wait", {})), toolset), "go"))
    await peer.journal.until("entered")
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    await peer.journal.until("settled", timeout=5)


async def test_an_interruption_frees_the_conversation_and_the_watchdog_cancels_the_call(
    peer: HttpPeer,
) -> None:
    """A hung MCP server never holds the conversation: the person interrupts and the next turn
    is answered at once, while the call runs on until the 5s interruption watchdog cancels it
    on the server."""
    model = FakeLLM([tool_call("wait"), "Yes, I'm here."])
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url, request_timeout=None))
    session = VoiceSession(
        llm=model, vad=None, tts=FakeTTS(seconds=0.05), turn_handling={"turn_detection": None}
    )
    session.output.audio = FakeAudioOutput()
    async with session:
        await session.start(Agent(instructions="Help.", tools=[toolset]))
        speech = session.generate_reply(user_input="book it")
        await peer.journal.until("entered")
        async with asyncio.timeout(1):
            await session.interrupt()
            await speech
            later = await session._run(user_input="are you there?")
        await peer.journal.until("settled", timeout=6)
    assert speech.interrupted
    assert _reply(later).output == "Yes, I'm here."


# --- Sharing ---------------------------------------------------------------------------------


async def test_concurrent_turns_share_one_connection_and_one_listing(peer: HttpPeer) -> None:
    toolset = MCPToolset(id="peer", mcp_server=MCPServerHTTP(peer.url))

    async def echo(said: str) -> None:
        await text(agent_with(calling(("echo", {"text": said})), toolset), "go")

    await asyncio.gather(*(echo(f"call {n}") for n in range(5)))
    calls = peer.journal.entries("call")
    assert len(calls) == 5
    assert len({call["connection"] for call in calls}) == 1
    assert len(peer.journal.entries("list")) == 2, "listed once: two pages"


# --- Construction ----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 1e300])
@pytest.mark.parametrize("field", ["init_timeout", "read_timeout", "request_timeout"])
def test_an_invalid_timeout_fails_at_construction(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        MCPServerHTTP("http://127.0.0.1:1/mcp", **{field: bad})  # type: ignore[arg-type]
    if field != "read_timeout":
        with pytest.raises(ValueError, match=field):
            MCPServerStdio("never-launched", [], **{field: bad})  # type: ignore[arg-type]


def test_an_invalid_tool_timeout_fails_when_its_options_resolve() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _core.resolve_mcp_tool_options(MCPToolOptions(timeout=-1.0))
