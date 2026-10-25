"""Progress of a call in the background, on a call (docs/delegation.md, Flow B): once the person
heard from the turn that made the call, a progress update goes into the model's context and asks
for no reply; with nothing said, the first update is spoken."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, LiveSession, RunContext, llm as llm_types
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.tool_executor import (
    REPLY_INSTRUCTIONS_AT_TAIL,
    UPDATE_TEMPLATE,
    _PendingUpdate,
)

from .fake_llm import FakeLLMResponse
from .test_agent_api import RecordingLLM

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("beside", ["On it.", ""], ids=["replied-beside", "silent"])
async def test_progress_asks_no_reply_once_the_person_heard_from_the_turn(beside: str) -> None:
    second_update = asyncio.Event()
    release = asyncio.Event()
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        await ctx.update("booking now")
        await second_update.wait()
        await ctx.update("still booking")
        await release.wait()
        return f"booked {day}"

    progress = UPDATE_TEMPLATE.format(function_name="book", call_id="b1", message="booking now")
    call = FunctionToolCall(name="book", arguments='{"day": "monday"}', call_id="b1")
    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book monday", content=beside, ttft=0, duration=0, tool_calls=[call]
            ),
            FakeLLMResponse(input=progress, content="Booking it now.", ttft=0, duration=0),
        ]
    )

    def said() -> list[str | None]:
        return [
            i.text_content
            for i in session.history.items
            if i.type == "message" and i.role == "assistant"
        ]

    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="book monday")
        second_update.set()
        await asyncio.sleep(0.5)
        heard = said()
        asked = [r.items[-1] for r in llm.requests]
        before_result = len(llm.requests)
        release.set()
        await asyncio.sleep(0.5)

    if beside:
        assert heard == ["On it."], "the progress was not answered again"
        assert not any(getattr(i, "output", None) == progress for i in asked)
    else:
        assert heard == ["Booking it now."], "nothing was said: the first progress is spoken"
    # the second progress is the model's context only
    assert before_result == (1 if beside else 2)
    outputs = [i.output for i in session.history.items if i.type == "function_call_output"]
    assert any("still booking" in o for o in outputs)


async def test_a_result_after_only_an_acknowledgement_and_the_persons_words_is_told() -> None:
    """The result landed after the acknowledgement, and the person spoke after it before the
    reply could go out: no reply had the result in front of it, so the model is told to tell
    it, not that it may have told it already."""
    desk = Agent(instructions="front desk")
    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input=REPLY_INSTRUCTIONS_AT_TAIL.format(call_ids=["b1_final"]),
                content="Monday is booked.",
                ttft=0,
                duration=0,
            )
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        activity = session._activity
        assert activity is not None
        call = llm_types.FunctionCall(call_id="b1", name="book", arguments='{"day": "monday"}')
        update = llm_types.FunctionCallOutput(
            call_id="b1", name="book", output="booking now", is_error=False
        )
        final = llm_types.FunctionCall(
            call_id="b1_final", name="book", arguments='{"day": "monday"}'
        )
        result = llm_types.FunctionCallOutput(
            call_id="b1_final", name="book", output="booked monday", is_error=False
        )
        chat_ctx = activity.chat_ctx.copy()
        chat_ctx.add_message(role="user", content="book monday")
        chat_ctx.add_message(role="assistant", content="On it.")
        chat_ctx.insert([call, update, final, result])
        chat_ctx.add_message(role="user", content="and is parking free?")
        await activity.update_chat_ctx(chat_ctx)
        executor = activity._tool_executor
        executor._pending_updates.append(
            _PendingUpdate(ctx=None, items=[final, result], target=activity)  # type: ignore[arg-type]
        )
        await executor._deliver_reply(session)
        for _ in range(50):
            if llm.requests:
                break
            await asyncio.sleep(0.05)

    told = [i.text_content for i in llm.requests[-1].items if i.type == "message"]
    assert told[-1] == REPLY_INSTRUCTIONS_AT_TAIL.format(call_ids=["b1_final"])


async def test_no_later_progress_asks_for_a_reply_once_the_turn_said_something() -> None:
    """The call's first progress got no words, then the turn said something for the call in a
    reply of its own: the call's later progress asks for no reply, and its result is told."""
    second_update = asyncio.Event()
    release = asyncio.Event()
    desk = Agent(instructions="front desk")

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str:
        """The back office.

        Args:
            request: the request
        """
        await ctx.update("Checking.")
        await second_update.wait()
        await ctx.update("Checking the calendar.")
        await release.wait()
        return "There are 10 slots."

    later = UPDATE_TEMPLATE.format(
        function_name="ask", call_id="a1_update_1", message="Checking the calendar."
    )
    call = FunctionToolCall(name="ask", arguments='{"request": "slots"}', call_id="a1")
    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(input="any slots?", content="", ttft=0, duration=0, tool_calls=[call]),
            FakeLLMResponse(input="ok", content="Sure, let me check.", ttft=0, duration=0),
            FakeLLMResponse(
                input=REPLY_INSTRUCTIONS_AT_TAIL.format(call_ids=["a1_update_1"]),
                content="Checking the calendar now.",
                ttft=0,
                duration=0,
            ),
            FakeLLMResponse(
                input=REPLY_INSTRUCTIONS_AT_TAIL.format(call_ids=["a1_final"]),
                content="We have 10 slots.",
                ttft=0,
                duration=0,
            ),
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="any slots?")
        await session.run(user_input="ok")
        second_update.set()
        await asyncio.sleep(0.5)
        release.set()
        for _ in range(50):
            if any(
                i.output == "There are 10 slots."
                for i in session.history.items
                if i.type == "function_call_output"
            ):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)

    said = [
        i.text_content
        for i in session.history.items
        if i.type == "message" and i.role == "assistant"
    ]
    assert said == ["Sure, let me check.", "We have 10 slots."]
    outputs = [i.output for i in session.history.items if i.type == "function_call_output"]
    assert any(later == o for o in outputs), "the later progress is in the model's context"
