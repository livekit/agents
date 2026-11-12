"""Regression contracts for which tools a reply may call and how long a tool's interruption
protection lasts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from livekit.agents import Agent, RunContext, SessionState, VoiceSession, function_tool
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import ChatItem, ToolContext
from livekit.agents.storage import ContextStore, Storage, Thread, persisted
from tests_new.fakes import FakeAudioOutput, FakeLLM, FakeTTS, tool_call, until


async def test_an_on_enter_reply_cannot_call_a_toolset_tool_flagged_ignore_on_enter() -> None:
    """A toolset's IGNORE_ON_ENTER tool is withheld from the on_enter reply: calling it there
    is refused, and later turns go on. https://github.com/livekit/agents/issues/6354"""
    model = FakeLLM(
        [tool_call("end_call", call_id="greet-end"), "Hello, how can I help?", "Anything else?"]
    )
    agent = Agent(instructions="Help.", llm=model, tools=[EndCallTool(ignore_on_enter=True)])

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        await session.generate_reply(instructions="greet")

    async with VoiceSession(vad=None, turn_handling={"turn_detection": "manual"}) as session:
        await session.start(agent)
        await until(lambda: any(i.type == "message" for i in session.history.items), "no greeting")
        await session.generate_reply(user_input="hello")

    refused = next(i for i in session.history.items if i.type == "function_call_output")
    assert refused.call_id == "greet-end" and refused.is_error
    assert session.history.messages()[-1].text_content == "Anything else?"


async def test_disallowed_interruptions_hold_until_the_tool_body_completes() -> None:
    """A tool that disallowed interruptions keeps running through its intermediate speech and
    an interruption attempt, and later turns stay usable.
    https://github.com/livekit/agents/issues/4560"""
    model = FakeLLM([tool_call("transfer", call_id="t1"), "Transferred.", "Anything else?"])
    agent = Agent(instructions="Help.", llm=model)
    working, release = asyncio.Event(), asyncio.Event()
    outcome: list[str] = []

    @agent.tool
    async def transfer(ctx: RunContext) -> str:
        """Transfer the booking."""
        ctx.disallow_interruptions()
        session = ctx.live_session
        assert session is not None
        await session.say("Please wait.")
        working.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        outcome.append("completed")
        return "done"

    session = VoiceSession(
        vad=None, tts=FakeTTS(seconds=0.05), turn_handling={"turn_detection": None}
    )
    session.output.audio = FakeAudioOutput()
    async with session:
        await session.start(agent)
        speech = session.generate_reply(user_input="go")
        await working.wait()
        interrupting = session.interrupt()  # refused, or pending until the protected reply ends
        release.set()
        await speech
        await asyncio.gather(interrupting, return_exceptions=True)
        await session.generate_reply(user_input="next")

    assert outcome == ["completed"]
    assert not speech.interrupted
    assert session.history.messages()[-1].text_content == "Anything else?"


async def test_an_interruption_during_a_hung_tool_frees_the_conversation_at_once() -> None:
    """Interrupting a reply while its tool hangs releases the floor at once: the person's next
    turn is answered without waiting for the tool or the 5s interruption watchdog."""
    model = FakeLLM([tool_call("lookup", call_id="hung"), "Yes."])
    agent = Agent(instructions="Help.", llm=model)
    entered = asyncio.Event()

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the booking."""
        entered.set()
        await asyncio.Future[None]()
        return "unreachable"

    session = VoiceSession(
        vad=None, tts=FakeTTS(seconds=0.05), turn_handling={"turn_detection": None}
    )
    session.output.audio = FakeAudioOutput()
    async with session:
        await session.start(agent)
        speech = session.generate_reply(user_input="find my booking")
        await asyncio.wait_for(entered.wait(), 5)
        async with asyncio.timeout(1):
            await session.interrupt()
            await speech
            later = await session._run(user_input="are you there?")

    assert speech.interrupted
    assert later.status == "completed" and later.output == "Yes."


@dataclass
class _Desk:
    session: SessionState = persisted(default_factory=SessionState)


async def _call_store(storage: Storage, thread: Thread) -> ContextStore:
    """The call's store as a job binds it (``ctx.store``) for ``thread``'s desk partition."""
    activation = await storage.threads._core_context_store(thread.id, "desk")
    return ContextStore(activation=activation, thread=thread, scope_id="root:desk")


async def test_on_a_call_with_a_thread_an_interruption_frees_the_conversation_and_turns_commit_in_order(
    thread_storage: Storage,
) -> None:
    """On a call that saves to a thread, interrupting a reply while its tool runs still frees the
    conversation at once; only persistence waits. The tool's late result is saved with its own
    turn, before the next turn the person took meanwhile."""
    thread_id = (await thread_storage.threads.create_thread()).thread_id
    thread = Thread(thread_storage, thread_id, partition_key="desk")
    store = await _call_store(thread_storage, thread)
    desk = await store.load_context(_Desk())
    model = FakeLLM({"book it": tool_call("book", call_id="late"), "are you there?": "Yes."})
    agent = Agent(instructions="Help.", llm=model, name="desk")
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @agent.tool
    async def book(ctx: RunContext) -> str:
        """Book the room."""
        entered.set()
        await release.wait()
        finished.set()
        return "booked"

    session = VoiceSession(
        vad=None,
        tts=FakeTTS(seconds=0.05),
        turn_handling={"turn_detection": None},
        thread=thread,
        context=store.bind(desk),
    )
    session.output.audio = FakeAudioOutput()
    async with session:
        await session.start(agent)
        speech = session.generate_reply(user_input="book it")
        await asyncio.wait_for(entered.wait(), 5)
        async with asyncio.timeout(1):
            await session.interrupt()
            await speech
            later = await session._run(user_input="are you there?")
        release.set()
        await asyncio.wait_for(finished.wait(), 5)
        await until(
            lambda: any(i.type == "function_call_output" for i in session.history.items),
            "the late result never landed",
        )
        live = _described(session.history.items)

    assert speech.interrupted
    assert later.status == "completed" and later.output == "Yes."
    # The cut turn's call and late output stay together, before the turn taken meanwhile.
    in_order = [
        ("agent_handoff", "desk"),
        ("message", "book it"),
        ("function_call", "book"),
        ("function_call_output", "booked"),
        ("message", "are you there?"),
        ("message", "Yes."),
    ]
    assert live == in_order
    assert _described((await thread.chat_history()).items) == in_order


async def test_on_a_call_with_a_thread_the_turns_are_saved_in_the_order_they_happened(
    thread_storage: Storage,
) -> None:
    """The thread saves the call as it happened: the opening handoff first, then each turn's
    items together."""
    thread_id = (await thread_storage.threads.create_thread()).thread_id
    thread = Thread(thread_storage, thread_id, partition_key="desk")
    store = await _call_store(thread_storage, thread)
    desk = await store.load_context(_Desk())
    model = FakeLLM(
        {"book it": tool_call("book", call_id="now"), "booked": "Booked.", "are you there?": "Yes."}
    )
    agent = Agent(instructions="Help.", llm=model, name="desk")

    @agent.tool
    async def book(ctx: RunContext) -> str:
        """Book the room."""
        return "booked"

    session = VoiceSession(
        vad=None,
        tts=FakeTTS(seconds=0.05),
        turn_handling={"turn_detection": None},
        thread=thread,
        context=store.bind(desk),
    )
    session.output.audio = FakeAudioOutput()
    async with session:
        await session.start(agent)
        await session._run(user_input="book it")
        await session._run(user_input="are you there?")
        live = _described(session.history.items)

    in_order = [
        ("agent_handoff", "desk"),
        ("message", "book it"),
        ("function_call", "book"),
        ("function_call_output", "booked"),
        ("message", "Booked."),
        ("message", "are you there?"),
        ("message", "Yes."),
    ]
    assert live == in_order
    assert _described((await thread.chat_history()).items) == in_order


def _described(items: list[ChatItem]) -> list[tuple[str, str | None]]:
    """What a thread saves of a history, each item by its kind and what it says or names; the
    model's config notes are not saved."""
    described: list[tuple[str, str | None]] = []
    for item in items:
        if item.type == "message":
            described.append((item.type, item.text_content))
        elif item.type == "function_call":
            described.append((item.type, item.name))
        elif item.type == "function_call_output":
            described.append((item.type, item.output))
        elif item.type == "agent_handoff":
            described.append((item.type, item.new_agent_id))
    return described


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/5884 reproduces: tool order follows insertion",
)
@pytest.mark.parametrize("fmt", ["openai", "anthropic", "aws"])
def test_serialized_tools_do_not_depend_on_registration_order(fmt: str) -> None:
    """The same tool set serializes identically whatever order it was given in, so provider
    prompt caches keep hitting. https://github.com/livekit/agents/pull/5884"""

    async def book(ctx: RunContext) -> str:
        """Book a room."""
        return "booked"

    async def cancel(ctx: RunContext) -> str:
        """Cancel a booking."""
        return "cancelled"

    async def lookup(ctx: RunContext) -> str:
        """Look a booking up."""
        return "found"

    a, b, c = function_tool(book), function_tool(cancel), function_tool(lookup)

    shuffled = ToolContext([c, a, b]).parse_function_tools(fmt)  # type: ignore[call-overload]
    ordered = ToolContext([a, b, c]).parse_function_tools(fmt)  # type: ignore[call-overload]
    assert shuffled == ordered
