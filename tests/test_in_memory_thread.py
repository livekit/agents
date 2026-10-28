"""Direct invocations expose the ordinary thread API without configured storage."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from livekit.agents import Agent, RunContext, persisted
from livekit.agents.llm import FunctionToolCall
from livekit.agents.storage.resolve import Thread

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


@dataclass
class Preferences:
    title: str = persisted(default="")


async def test_direct_runs_have_independent_in_memory_threads() -> None:
    threads: list[Thread] = []
    agent = Agent(
        name="desk",
        instructions="Help",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="remember",
                    content="",
                    tool_calls=[FunctionToolCall(name="remember", arguments="{}", call_id="c1")],
                    ttft=0,
                    duration=0,
                ),
                FakeLLMResponse(input="memory ready", content="Remembered", ttft=0, duration=0),
            ]
        ),
    )

    @agent.tool
    async def remember(ctx: RunContext) -> str:
        """Keep a title for this conversation."""
        thread = ctx.thread
        threads.append(thread)
        assert (await thread.info()).attributes == {}
        await thread.update(attributes={"title": "A conversation"})
        assert (await thread.info()).attributes == {"title": "A conversation"}
        assert (await thread.chat_history()).items == []
        assert await thread.tasks() == []
        assert await thread.journal() == []
        preferences = Preferences()
        await ctx.store.load_context(preferences)
        preferences.title = "Saved in memory"
        await ctx.store.save_context(preferences)
        restored = Preferences()
        await ctx.store.load_context(restored)
        assert restored.title == "Saved in memory"
        return "memory ready"

    try:
        for _ in range(2):
            result = await agent.run("remember")
            assert result.status == "completed" and result.output == "Remembered"
        assert threads[0].id != threads[1].id
        assert (await threads[0].info()).attributes == {"title": "A conversation"}
    finally:
        for thread in threads:
            await thread.storage.aclose()
