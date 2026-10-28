"""Public A2A client and remote tools against an owned local HTTP/SSE peer."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest
import pytest_asyncio

from livekit.agents import APIConnectOptions, APITimeoutError
from livekit.agents.a2a import (
    AgentClient,
    DataPart,
    FilePart,
    Message,
    MessageMetadata,
    RemoteAgent,
    TextPart,
)

pytestmark = pytest.mark.unit

URI = "https://livekit.io/a2a/ext/agents/v1"
OPTIONS = APIConnectOptions(max_retry=0, read_timeout=2.0)


class Peer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, Any] | None]] = []
        self.scenario = "question"
        self.capable = True
        self.structured = False
        self.url = ""
        self.sockets: set[asyncio.StreamWriter] = set()
        self.tasks: set[asyncio.Task[None]] = set()
        self.release = asyncio.Event()
        self.canceled = asyncio.Event()

    async def accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        self.sockets.add(writer)
        try:
            await self.serve(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            self.sockets.discard(writer)
            self.tasks.discard(task)

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        method, _, _ = (await reader.readline()).decode().split()
        headers: dict[str, str] = {}
        while (line := await reader.readline()) != b"\r\n":
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        body = bytearray()
        if headers.get("transfer-encoding") == "chunked":
            while size := int((await reader.readline()).split(b";", 1)[0], 16):
                body.extend(await reader.readexactly(size))
                assert await reader.readexactly(2) == b"\r\n"
            assert await reader.readline() == b"\r\n"
        elif length := int(headers.get("content-length", "0")):
            body.extend(await reader.readexactly(length))
        rpc = json.loads(body) if body else None
        self.requests.append((method, headers, rpc))
        if method == "GET":
            if self.scenario == "stall-card":
                await self.release.wait()
                return
            extensions = []
            if self.capable:
                extension: dict[str, Any] = {"uri": URI}
                if self.structured:
                    extension["params"] = {
                        "toolName": "check_coverage",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"account": {"type": "integer"}},
                            "required": ["account"],
                            "additionalProperties": False,
                        },
                    }
                extensions.append(extension)
            card = {
                "name": "coverage_expert",
                "description": "Checks account coverage.",
                "version": "1",
                "supportedInterfaces": [
                    {
                        "url": self.url + "/rpc",
                        "protocolBinding": "JSONRPC",
                        "protocolVersion": "1.0",
                    }
                ],
                "capabilities": {"streaming": True, "extensions": extensions},
                "defaultInputModes": ["text/plain"],
                "defaultOutputModes": ["text/plain"],
                "skills": [],
            }
            await self.respond(writer, "application/json", json.dumps(card).encode())
            return
        assert rpc is not None
        params = rpc["params"]
        if rpc["method"] == "CancelTask":
            self.canceled.set()
            result = {
                "id": params["id"],
                "contextId": params["contextId"],
                "status": {"state": "TASK_STATE_CANCELED"},
            }
            await self.respond(writer, "application/json", self.envelope(rpc, result))
            return
        message = params.get("message", {})
        context = message.get("contextId", params.get("contextId", "remote-context"))
        task_id = message.get("taskId", params.get("id", "remote-task"))
        events = [
            {
                "task": {
                    "id": task_id,
                    "contextId": context,
                    "status": {"state": "TASK_STATE_SUBMITTED"},
                }
            }
        ]
        events.append(self.status(task_id, context, "WORKING", "Checking café"))
        if self.scenario in ("wait", "partial-stall"):
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            for event in events:
                data = b"data: " + self.envelope(rpc, event) + b"\n\n"
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            await writer.drain()
            await self.release.wait()
            return
        if self.scenario == "flood":
            events.extend(
                self.status(task_id, context, "WORKING", f"progress {i}") for i in range(40)
            )
        if self.scenario == "failed":
            events.append(self.status(task_id, context, "FAILED", "coverage unavailable"))
        elif self.scenario == "canceled":
            events.append(self.status(task_id, context, "CANCELED", "stopped"))
        elif self.scenario == "question" and not message.get("taskId"):
            events.append(self.status(task_id, context, "INPUT_REQUIRED", "Which account?"))
        else:
            events.append(
                {
                    "artifactUpdate": {
                        "taskId": task_id,
                        "contextId": context,
                        "artifact": {
                            "artifactId": "answer",
                            "name": "answer",
                            "parts": [{"text": "Confirmed "}, {"data": {"receipt": 42}}],
                        },
                    }
                }
            )
            final = self.status(task_id, context, "COMPLETED", "")
            final["statusUpdate"]["metadata"] = {URI + "/verbatim": "Confirmed "}
            events.append(final)
        wire = b"".join(b"data: " + self.envelope(rpc, event) + b"\n\n" for event in events)
        await self.respond(writer, "text/event-stream", wire)

    @staticmethod
    def envelope(rpc: dict[str, Any], result: object) -> bytes:
        return json.dumps(
            {"jsonrpc": "2.0", "id": rpc["id"], "result": result}, ensure_ascii=False
        ).encode()

    @staticmethod
    def status(task: str, context: str, state: str, text: str) -> dict[str, Any]:
        return {
            "statusUpdate": {
                "taskId": task,
                "contextId": context,
                "status": {
                    "state": "TASK_STATE_" + state,
                    "message": {
                        "messageId": "status-" + state,
                        "role": "ROLE_AGENT",
                        "parts": [{"text": text}],
                    },
                },
            }
        }

    @staticmethod
    async def respond(writer: asyncio.StreamWriter, kind: str, body: bytes) -> None:
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        )
        for offset in range(0, len(body), 7):
            writer.write(body[offset : offset + 7])
            await writer.drain()


@pytest_asyncio.fixture
async def peer() -> AsyncIterator[Peer]:
    peer = Peer()
    server = await asyncio.start_server(peer.accept, "127.0.0.1", 0)
    assert server.sockets
    peer.url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/expert"
    try:
        yield peer
    finally:
        server.close()
        await server.wait_closed()
        peer.release.set()
        for writer in tuple(peer.sockets):
            writer.close()
        await asyncio.gather(*tuple(peer.tasks))


@pytest.mark.asyncio
async def test_public_client_ack_continuation_and_full_artifacts(peer: Peer) -> None:
    client = AgentClient(peer.url, token="peer-token", conn_options=OPTIONS)
    try:
        first = await client.send("Check my coverage")
        assert (first.task_id, first.thread_id) == ("remote-task", "remote-context")
        updates = [update async for update in first]
        assert [update.state for update in updates] == ["submitted", "working", "input-required"]
        assert updates[1].text == "Checking café"
        reply = await client.send("account 42", task_id=first.task_id)
        final = [update async for update in reply][-1]
        assert final.state == "completed" and final.text == 'Confirmed {"receipt":42}'
        assert final.metadata["livekit/verbatim"] == "Confirmed "
        assert final.artifacts[0].parts[-1] == DataPart(data={"receipt": 42})
        assert sum(method == "GET" for method, _, _ in peer.requests) == 1
        sends = [rpc for _, _, rpc in peer.requests if rpc]
        assert sends[1]["params"]["message"]["taskId"] == first.task_id
        assert sends[1]["params"]["message"]["contextId"] == first.thread_id
        assert all(headers["a2a-version"] == "1.0" for _, headers, _ in peer.requests)
        assert all(
            headers["authorization"] == "Bearer peer-token" for _, headers, _ in peer.requests
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_generated_message_parts_and_typed_metadata(peer: Peer) -> None:
    peer.scenario = "complete"
    client = AgentClient(peer.url, conn_options=OPTIONS)
    try:
        reply = await client.send(
            Message(
                parts=[
                    TextPart(text="Check"),
                    DataPart(data={"account": 42}),
                    FilePart(url="https://example.test/coverage.pdf", media_type="application/pdf"),
                ],
                message_id="stable-id",
                metadata=MessageMetadata(channel="portal", application={"trace": "abc"}),
            )
        )
        assert [update async for update in reply][-1].state == "completed"
        message = peer.requests[-1][2]["params"]["message"]
        assert message["messageId"] == "stable-id"
        assert message["metadata"] == {URI + "/channel": "portal", "trace": "abc"}
        assert message["parts"] == [
            {"text": "Check"},
            {"data": {"account": 42}},
            {"url": "https://example.test/coverage.pdf", "mediaType": "application/pdf"},
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cancel_only_autoconnects_and_never_manufactures_completion(peer: Peer) -> None:
    peer.capable = False
    client = AgentClient(peer.url, conn_options=OPTIONS)
    try:
        canceled = await client.cancel(
            "pending", thread_id="their-context", reason="caller-canceled"
        )
        assert canceled.state == "canceled"
        rpc = peer.requests[-1][2]
        assert rpc["params"]["contextId"] == "their-context"
        assert rpc["params"]["metadata"] == {URI + "/reason": "caller-canceled"}
        await client.cancel("pending", thread_id="their-context")
        assert sum(method == "GET" for method, _, _ in peer.requests) == 1
        assert all(headers["a2a-version"] == "1.0" for _, headers, _ in peer.requests)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_discovery_timeout_keeps_its_phase(peer: Peer) -> None:
    peer.scenario = "stall-card"
    client = AgentClient(peer.url, conn_options=replace(OPTIONS, read_timeout=0.02))
    try:
        with pytest.raises(APITimeoutError) as error:
            await client.send("Check")
        assert error.value.phase == "read"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_timeout_after_progress_keeps_phase_without_replaying(peer: Peer) -> None:
    peer.scenario = "partial-stall"
    client = AgentClient(peer.url, conn_options=replace(OPTIONS, read_timeout=0.02, max_retry=2))
    try:
        reply = await client.send("Check")
        assert (await anext(reply)).state == "submitted"
        assert (await anext(reply)).state == "working"
        with pytest.raises(APITimeoutError) as error:
            await anext(reply)
        assert error.value.phase == "read"
        assert (
            sum(rpc is not None and rpc["method"].startswith("Send") for _, _, rpc in peer.requests)
            == 1
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_remote_card_defines_non_ask_name_and_typed_input(peer: Peer) -> None:
    peer.structured = True
    toolset = RemoteAgent(peer.url, conn_options=OPTIONS).as_tool()
    await toolset.setup()
    tool = toolset.tools[0]
    assert tool.info.name == "check_coverage"
    assert tool.info.raw_schema["description"] == "Checks account coverage."
    assert tool.info.raw_schema["parameters"]["properties"] == {"account": {"type": "integer"}}
    assert "request" not in tool.info.raw_schema["parameters"]["properties"]
    with pytest.raises(TypeError):
        RemoteAgent(peer.url).as_tool(needs_approval=True)


@pytest.mark.asyncio
async def test_remote_continuation_survives_reconstruction_and_isolates_conversations(
    peer: Peer,
) -> None:
    from livekit.agents import Agent
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLM, FakeLLMResponse

    def response(text: str, call_id: str) -> FakeLLMResponse:
        return FakeLLMResponse(
            input=text,
            content="",
            ttft=0,
            duration=0,
            tool_calls=[
                FunctionToolCall(
                    name="coverage_expert", arguments=json.dumps({"request": text}), call_id=call_id
                )
            ],
        )

    tool = RemoteAgent(peer.url, conn_options=OPTIONS).as_tool()
    desk = Agent(
        instructions="Call coverage_expert.",
        tools=[tool],
        llm=FakeLLM(
            fake_responses=[
                response("Check first", "first"),
                response("Check second", "second"),
                response("account 42", "answer"),
            ]
        ),
    )
    first = await desk.run("Check first")
    second = await desk.run("Check second")
    assert first.status == second.status == "input_required"
    first_call = next(item for item in first.all_messages() if item.type == "function_call")
    second_call = next(item for item in second.all_messages() if item.type == "function_call")
    assert (
        first_call.extra["lk.delegation.state"]["id"]["delegation_id"]
        != second_call.extra["lk.delegation.state"]["id"]["delegation_id"]
    )
    await desk.run("account 42", message_history=first.all_messages())
    messages = [
        rpc["params"]["message"]
        for _, _, rpc in peer.requests
        if rpc and "message" in rpc["params"]
    ]
    assert "contextId" not in messages[0] and "contextId" not in messages[1]
    assert messages[2]["contextId"] == "remote-context" and messages[2]["taskId"] == "remote-task"
    assert all(message["metadata"][URI + "/kind"] == "delegation" for message in messages)
    histories = [
        part["data"]["items"]
        for message in messages
        for part in message["parts"]
        if part.get("metadata", {}).get(URI + "/kind") == "chat_ctx"
    ]
    assert histories and all(
        not any(
            item.get("call_id") in {"first", "second", "answer"} and item["type"] == "function_call"
            for item in history
        )
        for history in histories[:2]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["failed", "canceled"])
async def test_remote_failure_and_cancellation_are_truthful_tool_outcomes(
    peer: Peer, state: str
) -> None:
    from livekit.agents import Agent
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLM, FakeLLMResponse

    peer.scenario = state
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="Check",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(
                        name="coverage_expert", arguments='{"request":"Check"}', call_id="check"
                    )
                ],
            )
        ]
    )
    agent = Agent(
        instructions="Check coverage.",
        llm=model,
        tools=[RemoteAgent(peer.url, conn_options=OPTIONS).as_tool()],
    )
    if state == "canceled":
        with pytest.raises(asyncio.CancelledError):
            await agent.run("Check")
    else:
        result = await agent.run("Check")
        output = next(item for item in result.all_messages() if item.type == "function_call_output")
        assert output.is_error and output.output == "coverage unavailable"


@pytest.mark.asyncio
async def test_closing_a_pending_public_reply_cancels_exact_remote_task(peer: Peer) -> None:
    peer.scenario = "wait"
    client = AgentClient(peer.url, conn_options=replace(OPTIONS, read_timeout=None))
    try:
        reply = await client.send("Check")
        await reply.aclose()
        await asyncio.wait_for(peer.canceled.wait(), timeout=2)
        cancel = next(rpc for _, _, rpc in peer.requests if rpc and rpc["method"] == "CancelTask")
        assert cancel["params"]["id"] == reply.task_id
        assert cancel["params"]["contextId"] == reply.thread_id
        assert cancel["params"]["metadata"] == {URI + "/reason": "caller-canceled"}
        assert [update async for update in reply] == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_malformed_message_is_refused_before_discovery(peer: Peer) -> None:
    client = AgentClient(peer.url, conn_options=OPTIONS)
    try:
        with pytest.raises(ValueError, match="content parts"):
            await client.send(Message(parts=[]))
        with pytest.raises(ValueError, match="exactly one"):
            await client.send(Message(parts=[FilePart()]))
        assert peer.requests == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_total_deadline_cancels_remote_work_behind_an_unread_progress_queue(
    peer: Peer,
) -> None:
    peer.scenario = "flood"
    client = AgentClient(
        peer.url, conn_options=replace(OPTIONS, total_timeout=0.1, read_timeout=None)
    )
    try:
        reply = await client.send("Check")
        await asyncio.wait_for(peer.canceled.wait(), timeout=1)
        with pytest.raises(APITimeoutError) as error:
            async for _ in reply:
                pass
        assert error.value.phase == "total"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_remote_tool_uses_only_its_configured_project_origin_for_default_credentials(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.agents.a2a import Credentials

    monkeypatch.setenv("LIVEKIT_API_KEY", "delegation-test")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "delegation-test-secret-which-is-long-enough")
    monkeypatch.setenv("LIVEKIT_URL", peer.url.replace("http://", "ws://").removesuffix("/expert"))
    tool = RemoteAgent(peer.url, conn_options=OPTIONS).as_tool()
    await tool.setup()
    authorization = peer.requests[-1][1]["authorization"]
    principal = Credentials(
        "delegation-test", "delegation-test-secret-which-is-long-enough"
    ).verify(authorization.removeprefix("Bearer "))
    assert principal.identity == "delegate" and principal.kind == "project"

    for project in ["wss://foreign.test", peer.url.replace("http://", "wss://")]:
        monkeypatch.setenv("LIVEKIT_URL", project)
        await RemoteAgent(peer.url, conn_options=OPTIONS).as_tool().setup()
        assert "authorization" not in peer.requests[-1][1]


@pytest.mark.parametrize("failed", [False, True])
def test_native_remote_result_crosses_the_voice_tool_output_adapter(failed: bool) -> None:
    from livekit.agents._generated.tool_report import (
        ToolOutcomeToolError,
        ToolOutcomeValue,
        ToolReport,
    )
    from livekit.agents.llm import FunctionCall
    from livekit.agents.llm.utils import make_function_call_output

    report = ToolReport(
        outcome=ToolOutcomeToolError(message="unavailable")
        if failed
        else ToolOutcomeValue(text="covered")
    )
    result = make_function_call_output(
        fnc_call=FunctionCall(call_id="call", name="coverage_expert", arguments="{}"),
        output=report,
        exception=None,
    )
    assert result.fnc_call_out.output == ("unavailable" if failed else "covered")
    assert result.fnc_call_out.is_error == failed
