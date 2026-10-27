from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, RunContext, VoiceSession, function_tool
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
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(Greeter())

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
        async def handoff(self, ctx: RunContext) -> str:
            self.session.update_agent(Greeter())
            return "transferring"

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
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(SayGreeter())

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
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(DoubleGreeter())

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
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(QuietGreeter())

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
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(LateGreeter())

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
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(LateGreeter())

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


def _handoff_later(session: VoiceSession, delay: float) -> None:
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
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(Silent())

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm("", tool_calls=[HANDOFF_CALL])
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)

    messages = await _messages(actions, Router(), drain_delay=4.0)
    (greeting,) = _by_role(messages, "assistant")
    assert "e2e_latency" not in greeting.metrics


async def test_turn_committed_during_on_enter_survives_its_end() -> None:
    """The user speaks while on_enter still runs; that turn's tool-only handoff keeps its latency."""

    class Slow(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="slow")

        async def on_enter(self) -> None:
            await asyncio.sleep(3.0)

        @function_tool
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(Greeter())

    class Router(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="router")

        @function_tool
        async def handoff(self, ctx: RunContext) -> None:
            self.session.update_agent(Slow())

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
