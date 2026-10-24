"""End-to-end: provider (server-side) tool calls surface as a start/end lifecycle
on the AgentSession, parallel to `tool_execution_updated` for locally-run tools.

Drives a real AgentSession pipeline with a synthetic LLM stream that emits the
`provider_tool_call` event, and asserts the session re-emits
`provider_tool_execution_updated` — the exact contract a consumer (e.g. the
dashboard voice worker) subscribes to for its "tool is running" UX.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import TypeAdapter

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, ProviderToolCall
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice import (
    Agent,
    AgentEvent,
    AgentSession,
    ProviderToolCallEnded,
    ProviderToolCallStarted,
    ProviderToolExecutionUpdatedEvent,
)

pytestmark = pytest.mark.unit


class _ProviderToolLLM(llm.LLM):
    """Synthetic LLM that runs one or more provider tools, then answers with text."""

    def __init__(
        self,
        *,
        calls: list[tuple[str, str, str]],
        finish_tool: asyncio.Event | None = None,
    ) -> None:
        super().__init__()
        self._calls = calls  # (call_id, name, arguments)
        self._finish_tool = finish_tool

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> _ProviderToolStream:
        return _ProviderToolStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _ProviderToolStream(llm.LLMStream):
    def __init__(self, llm_v: _ProviderToolLLM, **kwargs: Any) -> None:
        super().__init__(llm_v, **kwargs)
        self._calls = llm_v._calls
        self._finish_tool = llm_v._finish_tool

    async def _run(self) -> None:
        for call_id, name, arguments in self._calls:
            # a provider tool begins running server-side (early, "is running" cue)...
            self.emit(
                "provider_tool_call",
                ProviderToolCall(phase="started", call_id=call_id, name=name, arguments=arguments),
            )
            if self._finish_tool is not None:
                await self._finish_tool.wait()
            # ...and finishes, with its result
            self.emit(
                "provider_tool_call",
                ProviderToolCall(
                    phase="done", call_id=call_id, name=name, arguments=arguments, result="ok"
                ),
            )
        # a short assistant answer so the turn completes normally
        self._event_ch.send_nowait(
            ChatChunk(id="msg", delta=ChoiceDelta(role="assistant", content="done"))
        )


async def _collect_updates(
    calls: list[tuple[str, str, str]],
    *,
    use_fallback: bool,
) -> list[ProviderToolCallStarted | ProviderToolCallEnded]:
    updates: list[ProviderToolCallStarted | ProviderToolCallEnded] = []
    model: llm.LLM = _ProviderToolLLM(calls=calls)
    if use_fallback:
        model = llm.FallbackAdapter([model])
    async with model, AgentSession(llm=model) as session:
        session.on("provider_tool_execution_updated", lambda ev: updates.append(ev.update))
        await session.start(Agent(instructions="You are a test agent."))
        await session.run(user_input="look it up")
    return updates


@pytest.mark.asyncio
@pytest.mark.parametrize("use_fallback", [False, True], ids=["direct", "fallback"])
async def test_provider_tool_lifecycle_emits_start_then_end(use_fallback: bool) -> None:
    updates = await _collect_updates(
        [("t1", "web_search", '{"q":"livekit"}')], use_fallback=use_fallback
    )

    assert len(updates) == 2
    started, ended = updates

    assert isinstance(started, ProviderToolCallStarted)
    assert started.call_id == "t1"
    assert started.name == "web_search"
    assert started.arguments == '{"q":"livekit"}'

    assert isinstance(ended, ProviderToolCallEnded)
    assert ended.call_id == "t1"
    assert ended.name == "web_search"
    assert ended.arguments == '{"q":"livekit"}'
    assert ended.result == "ok"
    assert ended.status == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_fallback", [False, True], ids=["direct", "fallback"])
async def test_multiple_provider_tools_tracked_in_order(use_fallback: bool) -> None:
    updates = await _collect_updates(
        [("t1", "web_search", "{}"), ("t2", "code_interpreter", "{}")],
        use_fallback=use_fallback,
    )

    # each tool gets its own start/end pair, in call order — the dashboard worker
    # relies on this to bracket a "thinking" cue per provider tool
    assert [(type(u).__name__, u.call_id) for u in updates] == [
        ("ProviderToolCallStarted", "t1"),
        ("ProviderToolCallEnded", "t1"),
        ("ProviderToolCallStarted", "t2"),
        ("ProviderToolCallEnded", "t2"),
    ]


@pytest.mark.parametrize("use_fallback", [False, True], ids=["direct", "fallback"])
async def test_started_event_arrives_before_tool_finishes(use_fallback: bool) -> None:
    finish_tool = asyncio.Event()
    updates: asyncio.Queue[ProviderToolExecutionUpdatedEvent] = asyncio.Queue()
    local_updates: list[object] = []
    model: llm.LLM = _ProviderToolLLM(calls=[("t1", "web_search", "{}")], finish_tool=finish_tool)
    if use_fallback:
        model = llm.FallbackAdapter([model])

    async with model, AgentSession(llm=model) as session:
        session.on("provider_tool_execution_updated", updates.put_nowait)
        session.on("tool_execution_updated", local_updates.append)
        session.on("function_tools_executed", local_updates.append)
        await session.start(Agent(instructions="You are a test agent."))
        run = session.run(user_input="look it up")

        try:
            started = await asyncio.wait_for(updates.get(), timeout=5.0)
            assert isinstance(started.update, ProviderToolCallStarted)
            assert updates.empty()
            assert not run.done()
        finally:
            finish_tool.set()

        await run
        ended = updates.get_nowait()
        assert isinstance(ended.update, ProviderToolCallEnded)
        assert ended.update.call_id == started.update.call_id
        assert updates.empty()
        assert local_updates == []


async def test_listeners_follow_model_swap_and_detach_on_close() -> None:
    old_model = _ProviderToolLLM(calls=[("old", "web_search", "{}")])
    new_model = _ProviderToolLLM(calls=[("new", "web_search", "{}")])
    updates: list[ProviderToolExecutionUpdatedEvent] = []
    agent = Agent(instructions="You are a test agent.")

    async with AgentSession(llm=old_model) as session:
        session.on("provider_tool_execution_updated", updates.append)
        await session.start(agent)
        await session.run(user_input="first lookup")
        agent.update_options(llm=new_model)
        old_model.emit(
            "provider_tool_call",
            ProviderToolCall(phase="started", call_id="stale", name="web_search"),
        )
        await session.run(user_input="second lookup")

    new_model.emit(
        "provider_tool_call",
        ProviderToolCall(phase="started", call_id="closed", name="web_search"),
    )
    assert [event.update.call_id for event in updates] == ["old", "old", "new", "new"]


@pytest.mark.parametrize(
    "update",
    [
        ProviderToolCallStarted(call_id="t1", name="web_search", arguments="{}"),
        ProviderToolCallEnded(call_id="t1", name="web_search", arguments="{}", result="ok"),
    ],
)
def test_provider_tool_event_round_trips_as_agent_event(
    update: ProviderToolCallStarted | ProviderToolCallEnded,
) -> None:
    event = ProviderToolExecutionUpdatedEvent(update=update)
    assert TypeAdapter(AgentEvent).validate_json(event.model_dump_json()) == event
