from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class ChattyAgent(Agent):
    """Speaks twice from on_enter: a deterministic say, then a generated reply."""

    def __init__(self) -> None:
        super().__init__(instructions="chatty agent")

    async def on_enter(self) -> None:
        await self.session.say("deterministic greeting")
        await self.session.generate_reply(instructions="generated_greeting")


class SilentAgent(Agent):
    """Produces no speech at all from on_enter."""

    def __init__(self) -> None:
        super().__init__(instructions="silent agent")


def _llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="generated_greeting", content="generated greeting", ttft=0.1, duration=0.1
            ),
        ]
    )


async def test_start_capture_run_records_every_on_enter_speech() -> None:
    """``start(capture_run=True)`` must keep the run open for the whole of on_enter.

    The run used to complete as soon as the first speech did, so a
    ``generate_reply`` issued after an awaited ``say`` never reached the
    RunResult (#4662).
    """
    async with AgentSession(llm=_llm()) as sess:
        result = await asyncio.wait_for(sess.start(ChattyAgent(), capture_run=True), timeout=10.0)

        result.expect.next_event().is_agent_handoff(new_agent_type=ChattyAgent)
        first = result.expect.next_event().is_message(role="assistant")
        assert first.event().item.text_content == "deterministic greeting"
        second = result.expect.next_event().is_message(role="assistant")
        assert second.event().item.text_content == "generated greeting"
        result.expect.no_more_events()


async def test_start_capture_run_completes_with_a_silent_on_enter() -> None:
    """An on_enter that produces no speech must still complete the run.

    Without a watched handle the RunResult never resolved, and
    ``start(capture_run=True)`` hung forever.
    """
    async with AgentSession(llm=_llm()) as sess:
        result = await asyncio.wait_for(sess.start(SilentAgent(), capture_run=True), timeout=5.0)

        result.expect.next_event().is_agent_handoff(new_agent_type=SilentAgent)
        result.expect.no_more_events()
