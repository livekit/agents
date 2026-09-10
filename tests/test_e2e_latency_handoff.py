from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import ChatMessage, FunctionToolCall

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

HANDOFF_CALL = FunctionToolCall(name="handoff", arguments="{}", call_id="call_1")


class Greeter(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="greeter")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet")


async def _messages(
    actions: FakeActions, agent: Agent, *, drain_delay: float = 2.0
) -> list[ChatMessage]:
    session = create_session(actions)
    events: list = []
    session.on("conversation_item_added", events.append)
    await run_session(session, agent, drain_delay=drain_delay)
    return [ev.item for ev in events if ev.item.type == "message"]


def _by_role(messages: list[ChatMessage], role: str) -> list[ChatMessage]:
    return [m for m in messages if m.role == role]


def _assert_answers(assistant: ChatMessage, user: ChatMessage) -> None:
    assert assistant.metrics["e2e_latency"] == pytest.approx(
        assistant.metrics["started_speaking_at"] - user.metrics["stopped_speaking_at"]
    )


async def test_handoff_reply_reports_e2e_latency() -> None:
    """A tool returning the next agent: its on_enter reply answers the user turn."""

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Greeter()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (user,) = _by_role(messages, "user")
    (greeting,) = _by_role(messages, "assistant")
    assert greeting.text_content == "hello from the greeter"
    _assert_answers(greeting, user)


async def test_update_agent_from_tool_reports_e2e_latency() -> None:
    """A tool calling update_agent itself is the same handoff."""

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(Greeter())

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (user,) = _by_role(messages, "user")
    (greeting,) = _by_role(messages, "assistant")
    _assert_answers(greeting, user)


async def test_speech_before_handoff_answers_the_turn() -> None:
    """The LLM speaks and hands off in one step: the speech answers, the greeting does not."""

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Greeter()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("let me transfer you", tool_calls=[HANDOFF_CALL])
    actions.add_tts(1.0)
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (user,) = _by_role(messages, "user")
    transfer, greeting = _by_role(messages, "assistant")
    assert transfer.text_content == "let me transfer you"
    _assert_answers(transfer, user)
    assert "e2e_latency" not in greeting.metrics


async def test_tool_reply_before_handoff_answers_the_turn() -> None:
    """A tool returning (agent, text): the tool reply answers, the greeting does not."""

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> tuple[Agent, str]:
            return Greeter(), "transferring"

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("transferring you now", input="transferring")
    actions.add_tts(1.0)
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (user,) = _by_role(messages, "user")
    tool_reply, greeting = _by_role(messages, "assistant")
    assert tool_reply.text_content == "transferring you now"
    _assert_answers(tool_reply, user)
    assert "e2e_latency" not in greeting.metrics


async def test_say_in_on_enter_answers_the_turn() -> None:
    """A say() greeting is the first thing the user hears; the reply after it is not."""

    class SayGreeter(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="say greeter")

        async def on_enter(self) -> None:
            await self.session.say("welcome")
            self.session.generate_reply(instructions="greet")

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return SayGreeter()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_tts(1.0, input="welcome")
    actions.add_llm("how can I help", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (user,) = _by_role(messages, "user")
    welcome, reply = _by_role(messages, "assistant")
    assert welcome.text_content == "welcome"
    _assert_answers(welcome, user)
    assert "e2e_latency" not in reply.metrics


async def test_concurrent_on_enter_speeches_answer_once() -> None:
    """say() and generate_reply() queued together from on_enter: only the first reports."""

    class DoubleGreeter(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="double greeter")

        async def on_enter(self) -> None:
            self.session.say("welcome")
            self.session.generate_reply(instructions="greet")

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return DoubleGreeter()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_tts(1.0, input="welcome")
    actions.add_llm("how can I help", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (user,) = _by_role(messages, "user")
    welcome, reply = _by_role(messages, "assistant")
    assert welcome.text_content == "welcome"
    _assert_answers(welcome, user)
    assert "e2e_latency" not in reply.metrics


async def test_unstored_say_in_on_enter_still_answers_the_turn() -> None:
    """say(add_to_chat_ctx=False) plays the answer; the reply after it reports nothing."""

    class QuietGreeter(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="quiet greeter")

        async def on_enter(self) -> None:
            await self.session.say("welcome", add_to_chat_ctx=False)
            self.session.generate_reply(instructions="greet")

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return QuietGreeter()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_tts(1.0, input="welcome")
    actions.add_llm("how can I help", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (reply,) = _by_role(messages, "assistant")
    assert reply.text_content == "how can I help"
    assert "e2e_latency" not in reply.metrics


async def test_nested_handoff_in_on_enter_reports_e2e_latency() -> None:
    """on_enter awaits an AgentTask: the task's own on_enter reply answers the user turn."""

    class AskName(AgentTask[None]):
        def __init__(self) -> None:
            super().__init__(instructions="ask name")

        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="ask_name")

        @function_tool
        async def record_name(self, ctx: RunContext, name: str) -> str:
            """Called when the user provides their name."""
            self.complete(None)
            return "recorded"

    class Survey(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="survey")

        async def on_enter(self) -> None:
            await AskName()

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Survey()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("what is your name?", input="ask_name")
    actions.add_tts(1.0)
    actions.add_user_speech(6.0, 7.0, "Bob")
    actions.add_llm(
        "",
        tool_calls=[
            FunctionToolCall(name="record_name", arguments='{"name": "Bob"}', call_id="c2")
        ],
    )

    messages = await _messages(actions, Router())
    go, _bob = _by_role(messages, "user")
    (question,) = _by_role(messages, "assistant")
    assert question.text_content == "what is your name?"
    _assert_answers(question, go)


async def test_task_spawned_in_on_enter_speaks_after_the_turn() -> None:
    """on_enter returned without speaking; a speech from a task it spawned is a new utterance."""

    class LateGreeter(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="late greeter")
            self._task: asyncio.Task[None] | None = None

        async def on_enter(self) -> None:
            async def _greet_later() -> None:
                await asyncio.sleep(0.5)
                self.session.generate_reply(instructions="greet")

            self._task = asyncio.create_task(_greet_later())

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return LateGreeter()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router())
    (greeting,) = _by_role(messages, "assistant")
    assert "e2e_latency" not in greeting.metrics


async def test_new_user_turn_supersedes_the_handoff_turn() -> None:
    """A user turn arriving before on_enter speaks is the one answered next; a late speech answers nothing."""

    class LateGreeter(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="late greeter")
            self._task: asyncio.Task[None] | None = None

        async def on_enter(self) -> None:
            async def _greet_later() -> None:
                await asyncio.sleep(4.0)
                self.session.generate_reply(instructions="greet")

            self._task = asyncio.create_task(_greet_later())

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return LateGreeter()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_user_speech(4.0, 4.5, "hello again")
    actions.add_llm("hi again")
    actions.add_tts(1.0)
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router(), drain_delay=4.0)
    _go, again = _by_role(messages, "user")
    reply, greeting = _by_role(messages, "assistant")
    assert reply.text_content == "hi again"
    _assert_answers(reply, again)
    assert greeting.text_content == "hello from the greeter"
    assert "e2e_latency" not in greeting.metrics


async def test_inline_task_in_tool_answers_each_turn() -> None:
    """user -> tool -> await task -> task asks -> user answers -> task done -> tool reply."""

    class AskEmail(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="ask email")

        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="ask_email")

        @function_tool
        async def record_email(self, ctx: RunContext, email: str) -> None:
            """Called when the user provides their email."""
            self.complete(email)

    class Booker(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="booker")

        @function_tool
        async def book(self, ctx: RunContext) -> str:
            """Book a room."""
            email = await AskEmail()
            return f"booked for {email}"

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "book a room")
    actions.add_llm("", tool_calls=[FunctionToolCall(name="book", arguments="{}", call_id="c1")])
    actions.add_llm("what is your email?", input="ask_email")
    actions.add_tts(1.0)
    actions.add_user_speech(6.0, 7.0, "a@b.com")
    actions.add_llm(
        "",
        tool_calls=[
            FunctionToolCall(name="record_email", arguments='{"email": "a@b.com"}', call_id="c2")
        ],
    )
    actions.add_llm("your room is booked", input="booked for a@b.com")
    actions.add_tts(1.0)

    messages = await _messages(actions, Booker(), drain_delay=4.0)
    book, email = _by_role(messages, "user")
    question, confirmation = _by_role(messages, "assistant")
    assert question.text_content == "what is your email?"
    _assert_answers(question, book)
    assert confirmation.text_content == "your room is booked"
    _assert_answers(confirmation, email)


def _handoff_later(session: AgentSession, delay: float) -> None:
    """A handoff nothing in the conversation caused, like a timer or an external event."""
    asyncio.get_event_loop().call_later(delay, session.update_agent, Greeter())


async def test_silent_tool_ends_the_turn() -> None:
    """A tool with no reply and no handoff ends the chain; a later unrelated handoff answers nothing."""

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> None:
            _handoff_later(self.session, 2.0)

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router(), drain_delay=4.0)
    (greeting,) = _by_role(messages, "assistant")
    assert "e2e_latency" not in greeting.metrics


async def test_silent_on_enter_ends_the_turn() -> None:
    """An on_enter that returns without speaking declines the turn; a later handoff answers nothing."""

    class Silent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="silent")

        async def on_enter(self) -> None:
            _handoff_later(self.session, 2.0)

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Silent()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router(), drain_delay=4.0)
    (greeting,) = _by_role(messages, "assistant")
    assert "e2e_latency" not in greeting.metrics


async def test_reply_after_awaited_task_in_on_enter_answers_the_last_turn() -> None:
    """on_enter awaits a task, then replies: the reply answers the turn the task left open."""

    class AskName(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="ask name")

        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="ask_name")

        @function_tool
        async def record_name(self, ctx: RunContext, name: str) -> None:
            """Called when the user provides their name."""
            self.complete(name)

    class Survey(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="survey")

        async def on_enter(self) -> None:
            name = await AskName()
            self.session.generate_reply(instructions=f"thank {name}")

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Survey()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("what is your name?", input="ask_name")
    actions.add_tts(1.0)
    actions.add_user_speech(6.0, 7.0, "Bob")
    actions.add_llm(
        "",
        tool_calls=[
            FunctionToolCall(name="record_name", arguments='{"name": "Bob"}', call_id="c2")
        ],
    )
    actions.add_llm("thanks Bob", input="thank Bob")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router(), drain_delay=4.0)
    go, bob = _by_role(messages, "user")
    question, thanks = _by_role(messages, "assistant")
    _assert_answers(question, go)
    assert thanks.text_content == "thanks Bob"
    _assert_answers(thanks, bob)


async def test_turn_committed_during_on_enter_survives_its_end() -> None:
    """The user speaks while on_enter still runs; that turn's tool-only handoff keeps its latency."""

    class Slow(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="slow")

        async def on_enter(self) -> None:
            await asyncio.sleep(3.0)

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Greeter()

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Slow()

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_user_speech(3.5, 4.0, "support please")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router(), drain_delay=4.0)
    _go, support = _by_role(messages, "user")
    (greeting,) = _by_role(messages, "assistant")
    _assert_answers(greeting, support)
