"""Exact text on a call: ``ctx.say`` and ``ctx.update`` share one ordered queue.

The first progress call of a tool releases it to the background; later ones queue in call order
and are delivered at idle, consecutive updates as one reply and each say as its own speech. A
say the person cut off before any of it played is spoken at the next idle. A model that cannot
speak text as written is asked to. Read in what the session speaks and what the model gets.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from livekit.agents import Agent, LiveSession, RunContext, function_tool
from livekit.agents.llm import ChatContext, FunctionCall, ToolResult
from livekit.agents.voice.tool_executor import UPDATE_TEMPLATE

from .a2a_driver import Reply, ScriptedLLM, call
from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities
from .fake_session import FakeActions, create_session, run_session
from .test_realtime_agent_state_during_tool import _generation

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

NOTICE = "Calls are recorded for quality."
NOT_YET = "ctx.say and the ordered queue: not built yet"
SESSION_TIMEOUT = 60.0


def last_text(*needles: str) -> Callable[[ChatContext], bool]:
    """A request whose last item, a message or a call's output, holds every needle."""

    def matches(ctx: ChatContext) -> bool:
        item = ctx.items[-1]
        if item.type == "function_call_output":
            text = item.output
        elif item.type == "message":
            text = item.text_content or ""
        else:
            return False
        return all(needle in text for needle in needles)

    return matches


def update_of(message: str) -> str:
    return UPDATE_TEMPLATE.format(function_name="book", call_id="1", message=message)


def spoken(session: LiveSession) -> list[str]:
    return [
        item.text_content or ""
        for item in session.history.items
        if item.type == "message" and item.role == "assistant"
    ]


@pytest.mark.rule("R22", "R23")
@pytest.mark.xfail(strict=True, reason=NOT_YET)
async def test_updates_and_says_are_spoken_in_call_order_and_a_say_is_never_merged() -> None:
    """An update, a say, then an update and the result, queued while the agent speaks: at idle
    the first update is one reply, the say is spoken as written, and the rest one reply."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Book me in")
    for text, seconds in (
        ("Let me check.", 1.0),
        ("One moment.", 4.0),
        ("Found a slot.", 1.0),
        (NOTICE, 1.0),
        ("It is booked.", 1.0),
    ):
        actions.add_tts(seconds, input=text)

    model = ScriptedLLM(
        Reply("Let me check.", [call("book", "1")], when=last_text("Book me in")),
        Reply("One moment.", when=last_text(update_of("checking"))),
        Reply("Found a slot.", when=last_text("1_update_1")),
        Reply("It is booked.", when=last_text("1_update_3", "1_final")),
    )

    class Desk(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="desk", llm=model)

        @function_tool
        async def book(self, ctx: RunContext) -> str:
            """Book."""
            await ctx.update("checking")
            await ctx.update("found a slot")
            await ctx.say(NOTICE)  # type: ignore[attr-defined]
            await ctx.update("holding it")
            return "booked Tuesday"

    session = create_session(actions)
    await asyncio.wait_for(run_session(session, Desk(), drain_delay=20), timeout=SESSION_TIMEOUT)

    assert spoken(session) == [
        "Let me check.",
        "One moment.",
        "Found a slot.",
        NOTICE,
        "It is booked.",
    ]
    # the say reached no model: no request carries it as an instruction to word
    assert not any(last_text(NOTICE)(r) for r in model.requests)


@pytest.mark.rule("R22", "R23")
@pytest.mark.xfail(strict=True, reason=NOT_YET)
async def test_a_first_say_is_the_turns_reply_with_no_model_step() -> None:
    """A tool whose first progress call is a say goes to the background, and the turn's reply
    is the said text, with no model step on it."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Book me in")
    actions.add_tts(1.0, input="Let me check.")
    actions.add_tts(1.0, input=NOTICE)

    model = ScriptedLLM(Reply("Let me check.", [call("book", "1")], when=last_text("Book me in")))
    finished = asyncio.Event()

    class Desk(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="desk", llm=model)

        @function_tool
        async def book(self, ctx: RunContext) -> None:
            """Book."""
            await ctx.say(NOTICE)  # type: ignore[attr-defined]
            await asyncio.sleep(5.0)
            finished.set()

    session = create_session(actions)
    await asyncio.wait_for(run_session(session, Desk(), drain_delay=20), timeout=SESSION_TIMEOUT)

    assert finished.is_set()
    assert spoken(session) == ["Let me check.", NOTICE]
    assert len(model.requests) == 1


@pytest.mark.rule("R22", "R23")
@pytest.mark.xfail(strict=True, reason=NOT_YET)
async def test_a_say_cut_off_before_it_played_is_spoken_at_the_next_idle() -> None:
    """The person speaks over a reply with a say queued behind it: the reply is cut, the say
    is not lost, and it is spoken once the agent is idle again."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Book me in")
    actions.add_user_speech(8.0, 9.5, "Wait a second")
    for text, seconds in (
        ("Let me check.", 1.0),
        ("One moment.", 1.0),
        ("Found a slot on Tuesday at nine in the morning.", 6.0),
        ("Sure.", 1.0),
        (NOTICE, 1.0),
    ):
        actions.add_tts(seconds, input=text)

    model = ScriptedLLM(
        Reply("Let me check.", [call("book", "1")], when=last_text("Book me in")),
        Reply("One moment.", when=last_text(update_of("checking"))),
        Reply("Found a slot on Tuesday at nine in the morning.", when=last_text("1_update_1")),
        Reply("Sure.", when=last_text("Wait a second")),
    )

    class Desk(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="desk", llm=model)

        @function_tool
        async def book(self, ctx: RunContext) -> None:
            """Book."""
            await ctx.update("checking")
            await ctx.update("found a slot")
            await ctx.say(NOTICE)  # type: ignore[attr-defined]
            await asyncio.sleep(30.0)

    session = create_session(actions)
    await asyncio.wait_for(run_session(session, Desk(), drain_delay=25), timeout=SESSION_TIMEOUT)

    said = spoken(session)
    assert said.count(NOTICE) == 1
    assert said.index("Sure.") < said.index(NOTICE)


class _Recording(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.instructions: list[str] = []

    def generate_reply(self, *, instructions: object = None, **kw: object):  # type: ignore[override]
        self.instructions.append(str(instructions or ""))
        return super().generate_reply(**kw)  # type: ignore[arg-type]


class _RecordingModel(FakeRealtimeModel):
    def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
        sess = _Recording(self)
        self.created_sessions.append(sess)
        return sess


@pytest.mark.rule("B45", "R61")
@pytest.mark.xfail(strict=True, reason=NOT_YET)
async def test_a_model_that_cannot_say_is_asked_to_speak_the_exact_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A realtime model with no TTS and no ``say()``: a verbatim result is spoken through a
    reply that asks for the exact text, with one warning, and the call does not fail."""
    from livekit.agents.llm.tool_context import SAY_EXACTLY  # type: ignore[attr-defined]

    model = _RecordingModel(capabilities=fake_capabilities(supports_say=False))
    outputs: list[str] = []

    class Desk(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="desk")

        @function_tool
        async def disclose(self) -> ToolResult:
            """Disclose."""
            return ToolResult(NOTICE, verbatim=True, reply_required=False)

    caplog.set_level(logging.WARNING)
    async with LiveSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Desk())
        session.on(
            "function_tools_executed",
            lambda ev: outputs.extend(o.output for o in ev.function_call_outputs if o),
        )

        reply = session.generate_reply()
        rt = model.active_session
        for _ in range(500):
            if rt._reply_futs:
                break
            await asyncio.sleep(0)
        rt._reply_futs[0].set_result(
            _generation(
                response_id="reply",
                text="one moment",
                audio_duration=0.5,
                function_calls=[FunctionCall(call_id="1", name="disclose", arguments="{}")],
            )
        )
        for _ in range(500):
            if len(rt._reply_futs) > 1:
                break
            await asyncio.sleep(0.01)
        assert len(rt._reply_futs) == 2, "the exact text was never asked for"
        rt._reply_futs[1].set_result(
            _generation(response_id="said", text=NOTICE, audio_duration=0.5)
        )
        await asyncio.wait_for(reply.wait_for_playout(), timeout=10)

    assert rt.instructions[-1].endswith(SAY_EXACTLY.format(text=NOTICE))
    assert rt.say_calls == []
    assert outputs and all("delivered" in o for o in outputs)
    warned = [r for r in caplog.records if "exact wording" in r.getMessage()]
    assert len(warned) == 1
