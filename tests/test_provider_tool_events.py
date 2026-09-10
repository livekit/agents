"""End-to-end: provider (server-side) tool calls surface as a start/end lifecycle
on the AgentSession, parallel to `tool_execution_updated` for locally-run tools.

Drives a real AgentSession pipeline with a synthetic LLM that emits the
`provider_tool_call` event on the LLM (the same EventEmitter seam as
`metrics_collected`), and asserts the session re-emits
`provider_tool_execution_updated` — the exact contract a consumer (e.g. the
dashboard voice worker) subscribes to for its "tool is running" UX.
"""

from __future__ import annotations

from typing import Any

import pytest

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, ProviderToolCall
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice import (
    Agent,
    AgentSession,
    ProviderToolCallEnded,
    ProviderToolCallStarted,
)

pytestmark = pytest.mark.unit


class _ProviderToolLLM(llm.LLM):
    """Synthetic LLM that runs one or more provider tools, then answers with text."""

    def __init__(self, *, calls: list[tuple[str, str, str]]) -> None:
        super().__init__()
        self._calls = calls  # (call_id, name, arguments)

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

    async def _run(self) -> None:
        for call_id, name, arguments in self._calls:
            # a provider tool begins running server-side (early, "is running" cue)...
            self._llm.emit(
                "provider_tool_call",
                ProviderToolCall(phase="started", call_id=call_id, name=name, arguments=arguments),
            )
            # ...and finishes, with its result
            self._llm.emit(
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
) -> list[ProviderToolCallStarted | ProviderToolCallEnded]:
    updates: list[ProviderToolCallStarted | ProviderToolCallEnded] = []
    async with AgentSession(llm=_ProviderToolLLM(calls=calls)) as session:
        session.on("provider_tool_execution_updated", lambda ev: updates.append(ev.update))
        await session.start(Agent(instructions="You are a test agent."))
        await session.run(user_input="look it up")
    return updates


@pytest.mark.asyncio
async def test_provider_tool_lifecycle_emits_start_then_end() -> None:
    updates = await _collect_updates([("t1", "web_search", '{"q":"livekit"}')])

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


@pytest.mark.asyncio
async def test_multiple_provider_tools_tracked_in_order() -> None:
    updates = await _collect_updates([("t1", "web_search", "{}"), ("t2", "code_interpreter", "{}")])

    # each tool gets its own start/end pair, in call order — the dashboard worker
    # relies on this to bracket a "thinking" cue per provider tool
    assert [(type(u).__name__, u.call_id) for u in updates] == [
        ("ProviderToolCallStarted", "t1"),
        ("ProviderToolCallEnded", "t1"),
        ("ProviderToolCallStarted", "t2"),
        ("ProviderToolCallEnded", "t2"),
    ]
