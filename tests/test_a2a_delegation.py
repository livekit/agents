"""Public A2A client and remote tools against an owned local HTTP/SSE peer."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import replace
from typing import Any

import pytest

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

from .a2a_peer import URI, Peer

pytestmark = pytest.mark.unit

OPTIONS = APIConnectOptions(max_retry=0, read_timeout=2.0)


def _remote(peer: Peer) -> RemoteAgent:
    return RemoteAgent(
        peer.url,
        name="coverage_expert",
        description="Checks account coverage.",
        conn_options=OPTIONS,
    )


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
async def test_remote_tool_is_defined_in_code_and_fetches_nothing_at_setup(peer: Peer) -> None:
    toolset = _remote(peer).as_tool()
    await toolset.setup()
    tool = toolset.tools[0]
    assert tool.info.name == "coverage_expert"
    assert tool.info.raw_schema["description"] == "Checks account coverage."
    assert tool.info.raw_schema["parameters"]["required"] == ["request"]
    assert peer.requests == []
    with pytest.raises(ValueError):
        RemoteAgent(peer.url)
    with pytest.raises(TypeError):
        _remote(peer).as_tool(needs_approval=True)


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

    tool = _remote(peer).as_tool()
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
        tools=[_remote(peer).as_tool()],
    )
    if state == "canceled":
        with pytest.raises(asyncio.CancelledError):
            await agent.run("Check")
    else:
        result = await agent.run("Check")
        output = next(item for item in result.all_messages() if item.type == "function_call_output")
        assert output.is_error and output.output == "coverage unavailable"


# Catches: a cancelled delegation leaving its remote task running, or its client open.
@pytest.mark.asyncio
async def test_cancelling_a_delegating_run_cancels_the_remote_task(peer: Peer) -> None:
    from livekit.agents import Agent
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLM, FakeLLMResponse

    peer.scenario = "wait"
    call = FunctionToolCall(
        name="coverage_expert", arguments=json.dumps({"request": "Check"}), call_id="c"
    )
    agent = Agent(
        instructions="Call coverage_expert.",
        tools=[_remote(peer).as_tool()],
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(input="Check", content="", ttft=0, duration=0, tool_calls=[call])
            ]
        ),
    )
    run = asyncio.create_task(agent.run("Check"))
    async with asyncio.timeout(5):
        while not any(rpc and "message" in rpc["params"] for _, _, rpc in peer.requests):
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        await peer.canceled.wait()
    cancel = next(rpc for _, _, rpc in peer.requests if rpc and rpc["method"] == "CancelTask")
    assert cancel["params"]["id"] == "remote-task"
    peer.release.set()


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
    await _call(_remote(peer))
    authorization = next(h for _, h, rpc in peer.requests if rpc)["authorization"]
    principal = Credentials(
        "delegation-test", "delegation-test-secret-which-is-long-enough"
    ).verify(authorization.removeprefix("Bearer "))
    assert principal.identity == "delegate" and principal.kind == "project"

    for project in ["wss://foreign.test", peer.url.replace("http://", "wss://")]:
        monkeypatch.setenv("LIVEKIT_URL", project)
        peer.requests.clear()
        await _call(_remote(peer))
        assert all("authorization" not in headers for _, headers, _ in peer.requests)


async def _call(remote: RemoteAgent) -> None:
    """One delegation through ``remote``, as a text run's model would make it."""
    from livekit.agents import Agent
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLM, FakeLLMResponse

    call = FunctionToolCall(
        name="coverage_expert", arguments=json.dumps({"request": "Check"}), call_id="c"
    )
    agent = Agent(
        instructions="Call coverage_expert.",
        tools=[remote.as_tool()],
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(input="Check", content="", ttft=0, duration=0, tool_calls=[call])
            ]
        ),
    )
    await agent.run("Check")


# Catches: a RemoteAgent tool a call lists at setup not opened as a delegation (its frame built
# before setup knew no expert), so its result never reached the call before the session closed.
@pytest.mark.asyncio
async def test_a_remote_agent_on_a_call_delegates_and_its_result_reaches_the_call(
    peer: Peer,
) -> None:
    from livekit.agents import Agent, VoiceSession
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLM, FakeLLMResponse

    peer.scenario = "complete"
    call = FunctionToolCall(
        name="coverage_expert", arguments=json.dumps({"request": "Check coverage"}), call_id="c1"
    )
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="Check coverage", content="", ttft=0, duration=0, tool_calls=[call]
            )
        ]
    )
    tool = _remote(peer).as_tool()
    agent = Agent(instructions="Ask the expert.", name="desk", tools=[tool])
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(agent)
        await session._run(user_input="Check coverage")

        async def result() -> str:
            while True:
                for item in session.history.items:
                    if item.type == "function_call_output" and "Confirmed" in item.output:
                        return item.output
                await asyncio.sleep(0.01)

        assert "Confirmed" in await asyncio.wait_for(result(), 5)


# Catches: a console worker delegating to Cloud (503: nothing serves its endpoints there), or its
# loopback listener refusing a caller that has no credentials to present.
@pytest.mark.asyncio
async def test_a_console_worker_delegates_to_its_own_endpoint_over_loopback(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.agents import AgentHandler, AgentServer
    from livekit.agents.a2a import A2ASessionContext
    from livekit.agents.storage import SQLiteStorage

    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_WORKER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    received: list[str] = []
    back_office = AgentHandler(lane="backend")

    @back_office.a2a_session(endpoint="coverage_expert", description="Checks account coverage.")
    async def text(ctx: A2ASessionContext) -> str:
        received.append(ctx.message.text_content or "")
        return "covered"

    server = AgentServer(
        agent_name="clinic",
        storage=SQLiteStorage(directory=tmp_path),
        listeners={"health": {"port": 0}},
    )
    server.add(back_office)
    run = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        for _ in range(100):
            with contextlib.suppress(RuntimeError):
                url = text.delegation_url()
                break
            await asyncio.sleep(0.05)
        assert url.startswith("http://127.0.0.1:")
        assert url.endswith("/agents/clinic/default/coverage_expert")
        await _call(RemoteAgent(text, conn_options=OPTIONS))
        assert received == ["Check"]
    finally:
        await server.aclose()
        await run
