"""Regression tests for https://github.com/livekit/agents/issues/3702

Completed tool calls/outputs must survive interruption, or the next inference
re-issues the call and duplicates side effects.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


class WeatherAgent(Agent):
    def __init__(self, *, tool_delay: float = 0.0) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self.tool_delay = tool_delay
        self.tool_executed = asyncio.Event()

    @function_tool
    async def get_weather(self, location: str) -> str:
        """
        Called when the user asks about the weather.

        Args:
            location: The location to get the weather for
        """
        if self.tool_delay > 0.0:
            await asyncio.sleep(self.tool_delay)
        self.tool_executed.set()
        return f"The weather in {location} is sunny today."


def _assert_weather_tool_preserved(agent: Agent, session: AgentSession) -> None:
    for label, items in (
        ("agent chat_ctx", agent.chat_ctx.items),
        ("session history", session.history.items),
    ):
        calls = [i for i in items if i.type == "function_call"]
        outs = [i for i in items if i.type == "function_call_output"]
        assert len(calls) == 1, f"{label}: the tool call must be preserved exactly once"
        assert calls[0].name == "get_weather"
        assert len(outs) == 1, f"{label}: the tool output must be preserved exactly once"
        assert outs[0].output == "The weather in Tokyo is sunny today."
        assert items.index(calls[0]) < items.index(outs[0])


def _weather_tool_turn(actions: FakeActions, *, tts_duration: float) -> None:
    actions.add_user_speech(0.5, 2.5, "What's the weather in Tokyo?")
    actions.add_llm(
        content="Let me check the weather for you.",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_tts(tts_duration)


async def test_tool_results_preserved_when_tool_reply_turn_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interruption lands right after the tool reply turn is scheduled: the reply
    returns at an early interruption gate, tool messages must already be committed."""
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=1.0)  # playout ~3.5s -> 4.5s
    # tool reply turn, interrupted before it generates anything
    actions.add_llm(
        content="It's sunny in Tokyo!",
        input="The weather in Tokyo is sunny today.",
        ttft=2.0,
        duration=2.5,
    )
    actions.add_tts(2.0)

    session = create_session(actions)
    agent = WeatherAgent()

    # the tool reply turn is the only speech scheduled with force=True;
    # interrupt synchronously right after that scheduling decision
    forced_schedules: list[SpeechHandle] = []
    orig_schedule = AgentActivity._schedule_speech

    def _interrupt_after_forced_schedule(
        self: AgentActivity, speech: SpeechHandle, priority: int, force: bool = False
    ) -> None:
        orig_schedule(self, speech, priority, force=force)
        if force:
            forced_schedules.append(speech)
            speech.interrupt()

    monkeypatch.setattr(AgentActivity, "_schedule_speech", _interrupt_after_forced_schedule)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert forced_schedules, "the tool reply turn was never scheduled; the test needs updating"
    _assert_weather_tool_preserved(agent, session)

    # the interrupted tool reply turn never spoke, its message must not appear
    assistant_texts = [
        i.text_content
        for i in agent.chat_ctx.items
        if i.type == "message" and i.role == "assistant"
    ]
    assert "It's sunny in Tokyo!" not in assistant_texts


async def test_tool_results_preserved_when_interrupted_during_playout() -> None:
    """Interruption lands while the agent is still speaking the tool turn."""
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=10.0)  # playout 3.5s -> 13.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # interrupts at 5.5s
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)

    session = create_session(actions)
    agent = WeatherAgent()  # the tool completes at ~3.4s, before the interruption

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    _assert_weather_tool_preserved(agent, session)


async def test_tool_results_preserved_when_tool_in_flight_at_interruption() -> None:
    """Interruption fires while the tool is still executing: in-flight tools run
    to completion and their results must be committed."""
    actions = FakeActions()
    _weather_tool_turn(actions, tts_duration=10.0)  # playout 3.5s -> 13.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # interrupts at 5.5s
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)

    session = create_session(actions)
    # the tool starts at ~3.4s and completes at ~7.4s, well after the interruption
    agent = WeatherAgent(tool_delay=4.0)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    _assert_weather_tool_preserved(agent, session)


async def test_handoff_tool_not_recorded_when_interrupted() -> None:
    """Handoffs aren't applied on interrupted speech, so their calls must not be
    recorded as completed — the LLM needs to retry them."""

    class TransferAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        @function_tool
        async def transfer_to_billing(self) -> Agent:
            """Transfer the user to the billing department."""
            return Agent(instructions="You are the billing agent.")

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "I have a billing question.")
    actions.add_llm(
        content="Transferring you to billing now.",
        tool_calls=[FunctionToolCall(name="transfer_to_billing", arguments="{}", call_id="1")],
    )
    actions.add_tts(10.0)  # playout 3.5s -> 13.5s
    actions.add_user_speech(5.0, 6.0, "Stop!", stt_delay=0.2)  # interrupts at 5.5s
    actions.add_llm(content="Okay, stopping.")
    actions.add_tts(1.0)

    session = create_session(actions)
    agent = TransferAgent()

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    for items in (agent.chat_ctx.items, session.history.items):
        assert not any(i.type in ("function_call", "function_call_output") for i in items)
