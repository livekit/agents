"""Switching agents in a realtime call switches the realtime model each agent runs on."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, VoiceSession
from livekit.agents.llm import ChatContext
from tests.fakes import FakeLLM, FakeRealtime, until


def texts(history: ChatContext) -> list[str]:
    return [
        item.text_content or ""
        for item in history.items
        if item.type == "message" and item.role in ("user", "assistant")
    ]


def said(session: VoiceSession, text: str) -> int:
    return sum(
        item.type == "message" and item.role == "assistant" and item.text_content == text
        for item in session.history.items
    )


def entered(agent: Agent) -> asyncio.Event:
    event = asyncio.Event()

    @agent.on_enter
    async def arrived(session: VoiceSession) -> None:
        event.set()

    return event


async def switch(session: VoiceSession, agent: Agent, arrived: asyncio.Event) -> None:
    arrived.clear()
    session.update_agent(agent)
    await until(arrived.is_set, f"{agent.name} never took over")
    await session.wait_for_idle()


# Catches: an agent's own realtime model refused at the switch (or ignored, the old connection
# answering), the next model starting without the conversation, or a replaced connection left
# open.
async def test_each_agent_replies_on_its_own_realtime_model_with_the_conversation() -> None:
    one = FakeRealtime(lambda history, instructions: "From model one.")
    two = FakeRealtime(lambda history, instructions: "From model two.")
    desk = Agent(name="desk", instructions="Help.", llm=one)
    billing = Agent(name="billing", instructions="Bill.", llm=two)
    desk_entered, billing_entered = entered(desk), entered(billing)
    async with VoiceSession(vad=None) as session:
        await session.start(desk)
        await session.generate_reply(user_input="Hello")
        await until(lambda: said(session, "From model one.") == 1, "From model one. was not said")

        await switch(session, billing, billing_entered)
        assert one.sessions[0].closed, "the replaced connection stays open"
        await session.generate_reply(user_input="What do I owe?")
        await until(lambda: said(session, "From model two.") == 1, "From model two. was not said")
        [(history, _)] = two.requests
        assert texts(history)[:2] == ["Hello", "From model one."]

        await switch(session, desk, desk_entered)
        assert two.sessions[0].closed
        await session.generate_reply(user_input="Thanks")
        await until(lambda: said(session, "From model one.") == 2, "From model one. was not said")
        assert len(one.sessions) == 2, "the desk's model opened a new connection"
        history, _ = one.requests[-1]
        assert texts(history)[:4] == [
            "Hello",
            "From model one.",
            "What do I owe?",
            "From model two.",
        ]
        assert not one.sessions[1].closed
    assert one.sessions[1].closed


# Catches: switching to an agent on the connected model closing and reopening the connection.
async def test_an_agent_on_the_connected_model_keeps_the_connection() -> None:
    model = FakeRealtime(lambda history, instructions: "Same model.")
    desk = Agent(name="desk", instructions="Help.")
    billing = Agent(name="billing", instructions="Bill.", llm=model)
    billing_entered = entered(billing)
    async with VoiceSession(llm=model, vad=None) as session:
        await session.start(desk)
        await switch(session, billing, billing_entered)
        await session.generate_reply(user_input="Hello")
        await until(lambda: said(session, "Same model.") == 1, "Same model. was not said")
        assert len(model.sessions) == 1 and not model.sessions[0].closed


# Catches: a switch between a realtime model and a pipeline LLM accepted, leaving the session's
# audio pipeline shaped for the other kind of model.
async def test_a_switch_between_a_realtime_model_and_a_pipeline_llm_is_refused() -> None:
    model = FakeRealtime(lambda history, instructions: "Realtime.")
    desk = Agent(name="desk", instructions="Help.")
    pipeline = Agent(name="pipeline", instructions="Help.", llm=FakeLLM(["Pipeline."]))
    async with VoiceSession(llm=model, vad=None) as session:
        await session.start(desk)
        with pytest.raises(ValueError, match="kind of model"):
            session.update_agent(pipeline)
        assert session.current_agent is desk
        await session.generate_reply(user_input="Hello")
        await until(lambda: said(session, "Realtime.") == 1, "Realtime. was not said")
