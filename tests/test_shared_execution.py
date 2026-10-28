"""Shared execution vectors (C-21): each vector runs through the direct run, a TextSession turn
and a call's VoiceSession.run, and is asserted the same way on the invocation's events.

A vector is one scripted model conversation and one agent definition. Behavior that only a
call has (duplicate-call policies, the floor, on_enter, playback) is not here: those vectors
belong to the voice suite, and the text entrypoints neither run nor promise them."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from pydantic import Field

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    Agent,
    ModelRetry,
    RunContext,
    TextSession,
    TextSessionState,
    UnexpectedModelBehavior,
    VoiceSession,
)
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChatItem,
    ChoiceDelta,
    FunctionToolCall,
    LLMStream,
)
from livekit.agents.voice.run_result import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunAssert,
    RunEvent,
    VoiceRunResult,
)

pytestmark = pytest.mark.unit


@dataclass
class Reply:
    text: str = ""
    calls: list[FunctionToolCall] | None = None


def call(name: str, arguments: dict[str, object], call_id: str) -> Reply:
    return Reply(
        calls=[FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=call_id)]
    )


class Script(LLM[Any]):
    """Answers each model request with the next scripted reply and keeps what it was sent."""

    def __init__(self, *replies: Reply) -> None:
        super().__init__()
        self.replies = deque(replies)
        self.requests: list[ChatContext] = []

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:
        self.requests.append(chat_ctx.copy())
        if not self.replies:
            raise RuntimeError("unexpected model request")
        return _Stream(self, chat_ctx=chat_ctx, reply=self.replies.popleft())


class _Stream(LLMStream):
    def __init__(self, model: Script, *, chat_ctx: ChatContext, reply: Reply) -> None:
        super().__init__(
            model, chat_ctx=chat_ctx, tools=[], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )
        self._reply = reply

    async def _run(self) -> None:
        self.send(
            ChatChunk(
                id="reply",
                delta=ChoiceDelta(
                    role="assistant", content=self._reply.text, tool_calls=self._reply.calls or []
                ),
            )
        )


@dataclass
class Settled:
    output: object
    expect: RunAssert


def _events(items: list[ChatItem]) -> list[RunEvent]:
    events: list[RunEvent] = []
    for item in items:
        if item.type == "message" and item.role == "assistant":
            events.append(ChatMessageEvent(item=item))
        elif item.type == "function_call":
            events.append(FunctionCallEvent(item=item))
        elif item.type == "function_call_output":
            events.append(FunctionCallOutputEvent(item=item))
    return events


def _settled(output: object, items: list[ChatItem]) -> Settled:
    return Settled(output=output, expect=RunAssert(_events(items)))


async def _direct(agent: Agent[Any, Any], *messages: str) -> Settled:
    history: list[ChatItem] = []
    for message in messages:
        result = await agent.run(message, message_history=history)
        history = result.all_messages()
    assert result.status == "completed"
    return _settled(result.output, result.new_messages())


async def _text_session(agent: Agent[Any, Any], *messages: str) -> Settled:
    state: TextSessionState | None = None
    for message in messages:
        turn = await TextSession(agent=agent).run(message, state=state)
        state = turn.state
    assert turn.result.status == "completed"
    return _settled(turn.result.output, turn.result.new_messages())


async def _voice(agent: Agent[Any, Any], *messages: str) -> Settled:
    async with VoiceSession() as session:
        await session.start(agent)
        for message in messages:
            result: VoiceRunResult[Any] = await session.run(user_input=message)
    return _settled(result.output, result.new_messages())


Entrypoint = Callable[..., Awaitable[Settled]]

ENTRYPOINTS = [
    pytest.param(_direct, id="direct"),
    pytest.param(_text_session, id="text_session"),
    pytest.param(_voice, id="voice"),
]


@pytest.fixture(params=ENTRYPOINTS)
def run(request: pytest.FixtureRequest) -> Entrypoint:
    return request.param  # type: ignore[no-any-return]


async def test_a_plain_reply_is_the_output(run: Entrypoint) -> None:
    agent = Agent(name="desk", instructions="Help", llm=Script(Reply("Hello there.")))
    settled = await run(agent, "Hi")
    assert settled.output == "Hello there."
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()


async def test_tool_arguments_are_coerced_and_defaulted_before_the_body(run: Entrypoint) -> None:
    seen: list[tuple[int, str, str | None]] = []
    model = Script(call("book", {"nights": "2"}, "c1"), Reply("Booked two nights."))
    agent = Agent(name="desk", instructions="Help", llm=model)

    @agent.tool
    def book(
        ctx: RunContext,
        nights: Annotated[int, Field(ge=1)],
        room: str = "double",
        note: str | None = None,
    ) -> str:
        """Book a room.

        Args:
            nights: How many nights.
            room: The room type.
            note: A note for the front desk.
        """
        seen.append((nights, room, note))
        return f"booked {nights} {room}"

    settled = await run(agent, "Book two nights")
    assert seen == [(2, "double", None)]
    assert settled.output == "Booked two nights."
    settled.expect.next_event().is_function_call(name="book", arguments={"nights": "2"})
    settled.expect.next_event().is_function_call_output(output="booked 2 double", is_error=False)
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()


async def test_invalid_arguments_go_back_to_the_model_and_the_body_never_runs(
    run: Entrypoint,
) -> None:
    ran: list[int] = []
    model = Script(
        call("book", {"nights": 0}, "c1"),
        call("book", {"nights": 1}, "c2"),
        Reply("Booked one night."),
    )
    agent = Agent(name="desk", instructions="Help", llm=model)

    @agent.tool
    async def book(ctx: RunContext, nights: Annotated[int, Field(ge=1)]) -> str:
        """Book a room.

        Args:
            nights: How many nights.
        """
        ran.append(nights)
        return "booked"

    settled = await run(agent, "Book")
    assert ran == [1]
    assert settled.output == "Booked one night."
    settled.expect.next_event().is_function_call(name="book", arguments={"nights": 0})
    rejected = settled.expect.next_event().is_function_call_output(is_error=True).event()
    assert "nights" in rejected.item.output
    settled.expect.next_event().is_function_call(name="book", arguments={"nights": 1})
    settled.expect.next_event().is_function_call_output(output="booked", is_error=False)
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()


async def test_an_unknown_tool_is_reported_to_the_model(run: Entrypoint) -> None:
    model = Script(call("refund", {}, "c1"), Reply("I cannot refund."))
    agent = Agent(name="desk", instructions="Help", llm=model)
    settled = await run(agent, "Refund me")
    assert settled.output == "I cannot refund."
    settled.expect.next_event().is_function_call(name="refund")
    unknown = settled.expect.next_event().is_function_call_output(is_error=True).event()
    assert "refund" in unknown.item.output
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()


async def test_model_retry_from_a_tool_reaches_the_model_and_the_retry_succeeds(
    run: Entrypoint,
) -> None:
    model = Script(
        call("weather", {"city": "Paris, TX"}, "c1"),
        call("weather", {"city": "Paris"}, "c2"),
        Reply("Sunny in Paris."),
    )
    agent = Agent(name="desk", instructions="Help", llm=model)

    @agent.tool
    async def weather(ctx: RunContext, city: str) -> str:
        """The weather in a city.

        Args:
            city: The city.
        """
        if "," in city:
            raise ModelRetry("Give the city name alone.")
        return "sunny"

    settled = await run(agent, "Weather in Paris?")
    assert settled.output == "Sunny in Paris."
    settled.expect.next_event().is_function_call(name="weather", arguments={"city": "Paris, TX"})
    retry = settled.expect.next_event().is_function_call_output(is_error=True).event()
    assert "Give the city name alone." in retry.item.output
    settled.expect.next_event().is_function_call(name="weather", arguments={"city": "Paris"})
    settled.expect.next_event().is_function_call_output(output="sunny", is_error=False)
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()
    assert any(
        i.type == "function_call_output" and "Give the city name alone." in i.output
        for i in model.requests[1].items
    )


async def test_a_tool_that_keeps_asking_for_a_retry_exhausts_its_retries(run: Entrypoint) -> None:
    model = Script(*[call("weather", {"city": f"Paris {i}"}, f"c{i}") for i in range(4)])
    agent = Agent(name="desk", instructions="Help", llm=model)

    @agent.tool
    async def weather(ctx: RunContext, city: str) -> str:
        """The weather in a city.

        Args:
            city: The city.
        """
        raise ModelRetry("Give the city name alone.")

    with pytest.raises(UnexpectedModelBehavior, match="weather"):
        await run(agent, "Weather in Paris?")
    assert len(model.requests) == 2


async def test_arguments_that_never_validate_exhaust_the_tools_retries(run: Entrypoint) -> None:
    ran: list[int] = []
    model = Script(*[call("book", {"nights": 0}, f"c{i}") for i in range(4)])
    agent = Agent(name="desk", instructions="Help", llm=model)

    @agent.tool
    async def book(ctx: RunContext, nights: Annotated[int, Field(ge=1)]) -> str:
        """Book a room.

        Args:
            nights: How many nights.
        """
        ran.append(nights)
        return "booked"

    with pytest.raises(UnexpectedModelBehavior, match="book"):
        await run(agent, "Book")
    assert ran == []
    assert len(model.requests) == 2


async def test_the_next_turn_sees_the_previous_exchange(run: Entrypoint) -> None:
    model = Script(Reply("Hi Ada."), Reply("Your name is Ada."))
    agent = Agent(name="desk", instructions="Help", llm=model)
    settled = await run(agent, "My name is Ada.", "What is my name?")
    assert settled.output == "Your name is Ada."
    seen = [
        (i.role, i.text_content)
        for i in model.requests[1].items
        if i.type == "message" and i.role in ("user", "assistant")
    ]
    assert seen == [
        ("user", "My name is Ada."),
        ("assistant", "Hi Ada."),
        ("user", "What is my name?"),
    ]
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()
