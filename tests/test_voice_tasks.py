"""A VoiceTask on a live session: it takes the floor, collects its answer over the person's
turns, and hands the typed result back to the tool that awaited it (docs/voice-tasks.md)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest

from livekit.agents import Agent, RunContext, VoiceSession
from livekit.agents.beta.workflows import GetEmailResult, GetEmailTask
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

ASK = "Ask for their email address, or confirm one they already gave."
READ_BACK = "Just to confirm, your email is theo@example.com, right?"


def _reply(said: str, content: str = "", *calls: tuple[str, str, dict]) -> FakeLLMResponse:
    tool_calls = [
        FunctionToolCall(name=name, arguments=json.dumps(args), call_id=call_id)
        for name, call_id, args in calls
    ]
    return FakeLLMResponse(input=said, content=content, ttft=0, duration=0, tool_calls=tool_calls)


async def _say(session: VoiceSession, text: str, until: Callable[[], bool]) -> None:
    """The person's turn, then the session's events until ``until`` holds: the turn's reply runs
    while the task holds the floor, so no speech of the turn's own marks its end."""
    changed = asyncio.Event()
    for event in ("conversation_item_added", "agent_state_changed"):
        session.on(event, lambda _: changed.set())
    session.generate_reply(user_input=text)
    async with asyncio.timeout(5):
        while not (until() and session.agent_state == "listening"):
            await changed.wait()
            changed.clear()


# Catches: a VoiceTask that cannot be driven on a call (a read-back never spoken without a voice,
# the person's turns cancelling the awaiting tool), runs its body before the person confirmed,
# or does not restore the agent that awaited it with the typed result.
async def test_get_email_task_confirms_the_address_and_returns_it_to_the_waiting_tool() -> None:
    address = {"email_address": "theo@example.com"}
    llm = FakeLLM(
        fake_responses=[
            _reply("change my email", "", ("change_receipt_email", "d1", {})),
            _reply(ASK, "What is your email?"),
            _reply("theo at example dot com", "", ("confirm_email_address", "c1", address)),
            _reply("yes", "", ("confirm_email_address", "c2", address)),
            _reply("email is theo@example.com", "Updated."),
        ]
    )
    desk = Agent(instructions="Help with receipts.")
    results: list[GetEmailResult] = []

    @desk.tool
    async def change_receipt_email(ctx: RunContext) -> str:
        """Change the receipt email."""
        results.append(await GetEmailTask().run())
        return f"email is {results[-1].email_address}"

    def said(text: str) -> Callable[[], bool]:
        return lambda: any(
            i.type == "message" and i.role == "assistant" and i.text_content == text
            for i in session.history.items
        )

    async with VoiceSession(llm=llm) as session:
        await session.start(desk)
        await _say(session, "change my email", said("What is your email?"))
        assert session.current_agent is not desk
        await _say(session, "theo at example dot com", said(READ_BACK))
        assert results == [], "the address was kept before the person confirmed it"
        await _say(session, "yes", said("Updated."))

        assert results == [GetEmailResult(email_address="theo@example.com")]
        assert session.current_agent is desk
        outputs = [i for i in session.history.items if i.type == "function_call_output"]
        assert outputs[-1].output == "email is theo@example.com"
