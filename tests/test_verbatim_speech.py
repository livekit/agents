"""Verbatim text on a call: ``ctx.say`` and ``ctx.update`` share one ordered queue, read in what
the session speaks and what the model gets."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, LiveSession, RunContext, function_tool
from livekit.agents.llm import ChatContext, FunctionCall, ToolResult

from .a2a_driver import Reply, ScriptedLLM, call
from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_session import FakeActions, create_session, run_session
from .test_realtime_agent_state_during_tool import _generation

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

NOTICE = "Calls are recorded for quality."
CODE = "Your code is HTL-7A3K."


def last_text(*needles: str) -> Callable[[ChatContext], bool]:
    """A request whose last item, a message or a call's output, holds every needle."""

    def matches(ctx: ChatContext) -> bool:
        item = ctx.items[-1]
        text = item.output if item.type == "function_call_output" else item.text_content or ""
        return all(needle in text for needle in needles)

    return matches


async def call_desk(
    model: ScriptedLLM,
    book: Callable[..., Any],
    replies: list[tuple[str, float]],
    *,
    cut_in: tuple[float, float, str] | None = None,
    drain_delay: float = 20,
) -> list[str]:
    """The person says "Book me in" (and ``cut_in``) to a desk whose one tool is ``book``; what
    the session spoke, each of ``replies`` taking its seconds of speech."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Book me in")
    if cut_in is not None:
        actions.add_user_speech(*cut_in)
    for text, seconds in replies:
        actions.add_tts(seconds, input=text)

    session = create_session(actions)
    desk = Agent(instructions="desk", llm=model, tools=[function_tool(book)])
    await asyncio.wait_for(run_session(session, desk, drain_delay=drain_delay), timeout=60.0)
    return [
        item.text_content or ""
        for item in session.history.items
        if item.type == "message" and item.role == "assistant"
    ]


@pytest.mark.rule("R22", "R23")
async def test_says_and_updates_are_spoken_in_call_order_and_a_say_is_never_merged() -> None:
    """A say, an update, a say, then an update and the result: the first say is the turn's
    reply with no model step, and the rest, queued while the agent speaks, are delivered at
    idle in call order, each say as written and the last update with the result as one reply."""
    model = ScriptedLLM(
        Reply("Let me check.", [call("book", "1")], when=last_text("Book me in")),
        Reply("Found a slot.", when=last_text("1_update_1")),
        Reply("It is booked.", when=last_text("1_update_3", "1_final")),
    )

    async def book(ctx: RunContext) -> str:
        """Book."""
        await ctx.say(NOTICE)  # type: ignore[attr-defined]
        await ctx.update("found a slot")
        await ctx.say(CODE)  # type: ignore[attr-defined]
        await ctx.update("holding it")
        return "booked Tuesday"

    replies = ["Let me check.", NOTICE, "Found a slot.", CODE, "It is booked."]
    spoken = await call_desk(model, book, [(text, 1.0) for text in replies])

    assert spoken == replies
    # the says reached no model: no request carries one as an instruction to word
    assert not any(last_text(NOTICE)(r) or last_text(CODE)(r) for r in model.requests)
    assert len(model.requests) == 3


@pytest.mark.rule("R22", "R23")
async def test_a_say_cut_off_before_it_played_is_spoken_at_the_next_idle() -> None:
    """The person speaks over a reply with a say queued behind it: the reply is cut, the say
    is not lost, and it is spoken once the agent is idle again."""
    model = ScriptedLLM(
        Reply("Let me check.", [call("book", "1")], when=last_text("Book me in")),
        Reply("One moment.", when=last_text("checking")),
        Reply("Found a slot on Tuesday at nine in the morning.", when=last_text("1_update_1")),
        Reply("Sure.", when=last_text("Wait a second")),
    )

    async def book(ctx: RunContext) -> None:
        """Book."""
        await ctx.update("checking")
        await ctx.update("found a slot")
        await ctx.say(NOTICE)  # type: ignore[attr-defined]
        await asyncio.sleep(30.0)

    spoken = await call_desk(
        model,
        book,
        [
            ("Let me check.", 1.0),
            ("One moment.", 1.0),
            ("Found a slot on Tuesday at nine in the morning.", 6.0),
            ("Sure.", 1.0),
            (NOTICE, 1.0),
        ],
        cut_in=(8.0, 9.5, "Wait a second"),
        drain_delay=25,
    )

    assert spoken.count(NOTICE) == 1
    assert spoken.index("Sure.") < spoken.index(NOTICE)


@pytest.mark.rule("B45", "R61")
async def test_a_model_that_cannot_say_is_asked_to_speak_the_exact_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A realtime model with no TTS and no ``say()``: each verbatim result is a reply the model
    is asked for, with a warning naming its text, and the call does not fail."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_say=False))
    outputs: list[str] = []

    async def disclose(text: str) -> ToolResult:
        """Disclose."""
        return ToolResult(text, verbatim=True, reply_required=False)

    async def replies(n: int) -> list[asyncio.Future[Any]]:
        for _ in range(500):
            if len(model.active_session._reply_futs) >= n:
                break
            await asyncio.sleep(0.01)
        return model.active_session._reply_futs

    caplog.set_level(logging.WARNING)
    async with LiveSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="desk", tools=[function_tool(disclose)]))
        session.on(
            "function_tools_executed",
            lambda ev: outputs.extend(o.output for o in ev.function_call_outputs if o),
        )
        reply = session.generate_reply()
        calls = [
            FunctionCall(call_id=str(i), name="disclose", arguments=json.dumps({"text": text}))
            for i, text in enumerate((NOTICE, CODE))
        ]
        (await replies(1))[0].set_result(
            _generation(
                response_id="r", text="one moment", audio_duration=0.5, function_calls=calls
            )
        )
        for n, text in enumerate((NOTICE, CODE), start=2):
            asked = await replies(n)
            assert len(asked) == n, f"{text!r} was never asked for"
            asked[-1].set_result(_generation(response_id=text, text=text, audio_duration=0.5))
        await asyncio.wait_for(reply.wait_for_playout(), timeout=10)

    assert model.active_session.say_calls == []
    assert outputs and all("delivered" in o for o in outputs)
    warned = [r for r in caplog.records if "not guaranteed" in r.getMessage()]
    assert [getattr(r, "text", None) for r in warned] == [NOTICE, CODE]
