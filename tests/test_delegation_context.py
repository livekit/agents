"""What a delegation carries (docs/delegation.md): the person's message is on the caller's lane
before the delegation goes out, the delegation names that lane, a delegation to the caller's own
lane is refused, a tool's duplicate policy holds in text, and the acknowledgement can ride the
reply that makes the call."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from livekit.agents import Agent, AgentServer, RunContext, function_tool
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.agents.delegation.a2a import ACKNOWLEDGE_WITH_CALL
from livekit.agents.llm import FunctionToolCall
from livekit.agents.storage import AgentDBStorage
from livekit.agents.voice.tool_executor import UPDATE_TEMPLATE

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_a2a_server import (  # noqa: F401  (no_project_env: autouse)
    AGENT,
    API_KEY,
    API_SECRET,
    ENDPOINT,
    http_client,
    make_server,
    no_project_env,
    rpc,
    sse_results,
    state_of,
    wire,
)
from .test_live_inbox import told

pytestmark = [pytest.mark.unit]


async def test_the_expert_reads_the_persons_words_on_the_callers_lane(
    agentdb_workers: tuple[AgentDBStorage, AgentDBStorage],
) -> None:
    clinic_storage, expert_storage = agentdb_workers
    seen: dict[str, Any] = {}

    async def expert(ctx: A2ASessionContext[Any]) -> str:
        seen["caller_lane"] = ctx.caller_lane
        history = await ctx.thread.chat_history(ctx.caller_lane)
        said = [i.text_content for i in history.items if i.type == "message" and i.role == "user"]
        seen["said"] = said
        return "covered"

    expert_server = AgentServer(storage=expert_storage, agent_name="eligibility")
    expert_server.a2a_session(endpoint=ENDPOINT, description="Eligibility.")(expert)
    async with http_client(expert_server) as http:
        delegate = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=http)
        front = Agent(
            instructions="front desk",
            tools=[delegate],
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input="am I covered for a cleaning?",
                        content="",
                        ttft=0,
                        duration=0,
                        tool_calls=[
                            FunctionToolCall(
                                name="ask", arguments='{"request": "coverage"}', call_id="c1"
                            )
                        ],
                    ),
                    FakeLLMResponse(input="covered", content="You are.", ttft=0, duration=0),
                ]
            ),
        )

        async def desk(ctx: A2ASessionContext[Any]) -> Any:
            return (await front.run(ctx.message, message_history=ctx.message_history)).output

        clinic, _ = make_server(clinic_storage, desk)
        async with http_client(clinic) as clinic_http:
            response = await rpc(
                clinic_http,
                "message/stream",
                {"message": wire("am I covered for a cleaning?", "m1")},
            )
        await delegate.aclose()
    assert state_of(sse_results(response)[-1]["result"]) == "TASK_STATE_COMPLETED"
    assert seen["caller_lane"] == AGENT
    assert seen["said"] == ["am I covered for a cleaning?"], "the turn's message, before its commit"


async def test_a_delegation_to_the_callers_own_lane_is_refused(
    agentdb_storage: AgentDBStorage,
) -> None:
    ran: list[str] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        ran.append(ctx.message.text_content)
        return "ran"

    server, _ = make_server(agentdb_storage, handler)
    message = wire("loop", "m1", **{"livekit/kind": "delegation", "livekit/callerLane": AGENT})
    async with http_client(server) as http:
        r = await rpc(http, "message/send", {"message": message})
    assert "error" in r.json() and "own lane" in r.json()["error"]["message"]
    assert ran == []


async def test_a_duplicate_of_a_call_in_the_background_follows_the_tools_policy(
    agentdb_storage: AgentDBStorage,
) -> None:
    """Under ``live`` a call goes on in the background; a second request that calls the same
    tool while it runs is refused by ``on_duplicate="reject"``, with the running call named."""
    entered, release = asyncio.Event(), asyncio.Event()

    @function_tool(on_duplicate="reject")
    async def book(ctx: RunContext, day: str) -> str:
        """Book.

        Args:
            day: the day
        """
        await ctx.update("booking")
        entered.set()
        await release.wait()
        return f"booked {day}"

    desk = Agent(instructions="front desk", tools=[book])
    progress = UPDATE_TEMPLATE.format(function_name="book", call_id="b1", message="booking")
    refused: list[str] = []

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book monday",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="book", arguments='{"day": "monday"}', call_id="b1")
                ],
            ),
            FakeLLMResponse(input=progress, content="On it.", ttft=0, duration=0),
            FakeLLMResponse(
                input="book monday again",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="book", arguments='{"day": "monday"}', call_id="b2")
                ],
            ),
            FakeLLMResponse(input=told("b1"), content="Booked.", ttft=0, duration=0),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        result = await desk.run(ctx.message, message_history=ctx.message_history)
        refused.extend(
            str(i.output)
            for i in result.new_messages()
            if i.type == "function_call_output" and i.call_id == "b2"
        )
        return result.output

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:

        async def stream(text: str, message_id: str) -> list[dict[str, Any]]:
            response = await rpc(
                http, "message/stream", {"message": wire(text, message_id, context_id=thread_id)}
            )
            return [r["result"] for r in sse_results(response)]

        first = asyncio.create_task(stream("book monday", "m1"))
        await asyncio.wait_for(entered.wait(), 20)
        await asyncio.sleep(1.0)
        await asyncio.wait_for(stream("book monday again", "m2"), 20)
        release.set()
        await asyncio.wait_for(first, 30)

    assert refused and "already running" in refused[0]
    assert json.loads(refused[0].split("\n")[1])["call_id"] == "b1"


async def test_the_acknowledgement_can_ride_the_reply_that_makes_the_call(
    agentdb_storage: AgentDBStorage,
) -> None:
    async def expert(ctx: A2ASessionContext[Any]) -> str:
        return "ok"

    server = AgentServer(storage=agentdb_storage, agent_name="eligibility")
    server.a2a_session(endpoint=ENDPOINT, description="Eligibility.")(expert)
    async with http_client(server) as http:
        delegate = A2ADelegate(
            f"http://agents.test/{ENDPOINT}", httpx_client=http, announce="with_call"
        )
        await delegate.setup()
        (tool,) = delegate.tools
        await delegate.aclose()
    assert ACKNOWLEDGE_WITH_CALL in tool.info.description


async def test_a_delegation_to_its_own_worker_goes_over_the_loopback(
    agentdb_storage: AgentDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job's delegation to a door of its own worker: the worker's loopback listener serves it,
    found through ``LIVEKIT_AGENT_LOCAL_ROUTES``, with the project token the node would have
    checked, and the expert's progress streams as it is written, not when the answer ends."""
    from livekit.agents.http import LOCAL_ROUTES_ENV, serve_loopback

    release = asyncio.Event()

    async def expert(ctx: A2ASessionContext[Any]) -> str:
        await ctx.update("checking the calendar")
        await release.wait()
        return "booked"

    server = AgentServer(
        storage=agentdb_storage,
        agent_name=AGENT,
        api_key=API_KEY,
        api_secret=API_SECRET,
        ws_url="wss://project.livekit.test",
    )
    server.a2a_session(endpoint="backend", description="the back office", public=False)(expert)
    public = "https://project.livekit.test/agents/front-desk/default"
    listener, local = await serve_loopback(server.http.app)
    monkeypatch.setenv(LOCAL_ROUTES_ENV, f"{public}={local}")
    monkeypatch.setenv("LIVEKIT_URL", "wss://project.livekit.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", API_KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", API_SECRET)
    try:
        client = await A2ADelegate(f"{public}/backend")._connect()
        reply = await client.send("book monday")
        progress = await asyncio.wait_for(anext(aiter(reply)), 20)
        assert progress.state == "working" and progress.text == "checking the calendar"
        release.set()
        updates = [u async for u in reply]
    finally:
        listener.close()
    assert updates[-1].state == "completed" and updates[-1].text == "booked"


async def test_on_a_call_the_expert_reads_the_persons_words_too(
    agentdb_workers: tuple[AgentDBStorage, AgentDBStorage],
) -> None:
    """The call writes the person's message on its lane, under its lease, before the
    delegation goes out; the expert reads it while the call's turn is still running."""
    from livekit.agents import LiveSession
    from livekit.agents.storage.resolve import Thread

    clinic_storage, expert_storage = agentdb_workers
    seen: dict[str, Any] = {}

    async def expert(ctx: A2ASessionContext[Any]) -> str:
        history = await ctx.thread.chat_history(ctx.caller_lane)
        said = [i.text_content for i in history.items if i.type == "message" and i.role == "user"]
        seen["said"] = said
        return "covered"

    expert_server = AgentServer(storage=expert_storage, agent_name="eligibility")
    expert_server.a2a_session(endpoint=ENDPOINT, description="Eligibility.")(expert)
    thread_id = (await clinic_storage.threads.create_thread()).thread_id
    async with http_client(expert_server) as http:
        delegate = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=http)
        front = Agent(instructions="front desk", tools=[delegate])
        llm = FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="am I covered for a cleaning?",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(
                            name="ask", arguments='{"request": "coverage"}', call_id="c1"
                        )
                    ],
                ),
                FakeLLMResponse(input="covered", content="You are.", ttft=0, duration=0),
            ]
        )
        thread = Thread(clinic_storage, thread_id, agent_name=AGENT)
        async with LiveSession(llm=llm, thread=thread) as session:
            await session.start(front)
            await session.run(user_input="am I covered for a cleaning?")
        await delegate.aclose()
    assert seen["said"] == ["am I covered for a cleaning?"], "on the lane before the commit"
