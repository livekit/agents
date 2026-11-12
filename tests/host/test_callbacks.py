"""User Python the core calls: tool bodies and their arguments, output validators and output
functions, approval read-backs, hooks, session nodes and speech done callbacks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Annotated

import httpx
import pytest
from pydantic import AfterValidator, BaseModel, Field

from livekit import rtc
from livekit.agents import (
    Agent,
    Completed,
    InputRequired,
    ModelRetry,
    ModelSettings,
    RunContext,
    ToolResult,
    VoiceSession,
    VoiceTask,
    function_tool,
    mock_tools,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.evals import Judge, JudgeGroup, JudgmentResult
from livekit.agents.llm import LLM, ChatContext, ChatItem, Toolset
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
)
from tests.fakes import FakeAudioOutput, FakeLLM, FakeTTS, tool_call, until

from .a2a_door import message_send
from .conftest import A2AHandler

# -- tool bodies


@pytest.mark.parametrize(
    "arguments, note",
    [
        ({"nights": "2"}, None),
        ({"nights": "2", "note": None}, None),
        ({"nights": "2", "note": "window"}, "window"),
    ],
    ids=["omitted", "null", "provided"],
)
async def test_tool_arguments_are_coerced_and_defaulted_before_the_body(
    arguments: dict[str, str | None], note: str | None
) -> None:
    """The model's JSON arguments reach the body validated, coerced and defaulted."""
    seen: list[tuple[int, str, str | None]] = []
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("book", **arguments), "Booked."]))

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

    result = await agent.run("Book two nights")

    assert seen == [(2, "double", note)]
    result.expect.next_event().is_function_call(name="book", arguments=arguments)
    result.expect.next_event().is_function_call_output(output="booked 2 double", is_error=False)
    result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()


async def test_invalid_arguments_go_back_to_the_model_and_the_body_never_runs() -> None:
    """Arguments that fail validation are returned to the model as an error; only the
    corrected call runs the body."""
    ran: list[int] = []
    model = FakeLLM([tool_call("book", nights=0), tool_call("book", nights=1), "Booked."])
    agent = Agent(instructions="Help.", llm=model)

    @agent.tool
    async def book(ctx: RunContext, nights: Annotated[int, Field(ge=1)]) -> str:
        """Book a room."""
        ran.append(nights)
        return "booked"

    result = await agent.run("Book")

    assert ran == [1]
    result.expect.next_event().is_function_call(arguments={"nights": 0})
    rejected = result.expect.next_event().is_function_call_output(is_error=True).event()
    assert "nights" in rejected.item.output
    result.expect.next_event().is_function_call(arguments={"nights": 1})
    result.expect.next_event().is_function_call_output(output="booked", is_error=False)


async def test_an_annotated_validator_changes_the_value_the_body_receives() -> None:
    """A pydantic validator on a parameter runs before the body sees the value."""
    seen: list[str] = []
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("label", value="hello")]))

    @agent.tool
    async def label(
        ctx: RunContext, value: Annotated[str, AfterValidator(str.upper)]
    ) -> ToolResult:
        """Label an item."""
        seen.append(value)
        return ToolResult(value, reply_required=False)

    await agent.run("do it")

    assert seen == ["HELLO"]


async def test_concurrent_runs_keep_their_application_contexts_separate() -> None:
    """Two runs of one agent, interleaved inside their tools, each see only their own
    context."""
    model = FakeLLM(
        {
            "Hi Ada": tool_call("inspect"),
            "Hi Grace": tool_call("inspect"),
            "Ada": "Ada",
            "Grace": "Grace",
        }
    )
    agent = Agent(instructions="Help.", llm=model)
    entered: list[str] = []

    @agent.tool
    async def inspect(ctx: RunContext[dict[str, str]]) -> str:
        """Inspect this run's context."""
        entered.append(ctx.context["name"])
        await until(lambda: len(entered) == 2, "both runs entered the tool")
        return ctx.context["name"]

    ada, grace = await asyncio.gather(
        agent.run("Hi Ada", context={"name": "Ada"}),
        agent.run("Hi Grace", context={"name": "Grace"}),
    )

    assert ada.status == grace.status == "completed"
    assert (ada.output, grace.output) == ("Ada", "Grace")


async def test_a_toolsets_nested_application_result_reaches_the_model_as_json() -> None:
    """Dataclasses and pydantic models a toolset method returns are serialized as JSON."""

    class Seat(BaseModel):
        row: int

    @dataclass
    class Flight:
        number: str
        seats: list[Seat]

    class Flights(Toolset):
        @function_tool
        async def list_flights(self, ctx: RunContext) -> ToolResult:
            """List flights."""
            return ToolResult([Flight("SK204", [Seat(row=3)])], reply_required=False)

    agent = Agent(
        instructions="Help.",
        tools=[Flights(id="flights")],
        llm=FakeLLM([tool_call("list_flights")]),
    )

    result = await agent.run("do it")

    output = result.expect.next_event(type="function_call_output").event().item
    assert not output.is_error
    assert json.loads(output.output) == [{"number": "SK204", "seats": [{"row": 3}]}]


async def test_an_unserializable_tool_result_is_an_error_for_the_model() -> None:
    """A result that cannot be serialized reaches the model as an error output."""
    agent = Agent(instructions="Help.", llm=FakeLLM([tool_call("invalid"), "Sorry."]))

    @agent.tool
    async def invalid(ctx: RunContext) -> ToolResult:
        """Return an unsupported value."""
        return ToolResult(object(), reply_required=False)

    result = await agent.run("do it")

    result.expect.next_event().is_function_call(name="invalid")
    result.expect.next_event().is_function_call_output(is_error=True)


async def test_mock_tools_replace_a_tool_only_in_the_runs_started_inside_the_block() -> None:
    """`mock_tools` applies to runs started in its block; a concurrent run outside it calls the
    real tool."""
    seen: list[str] = []
    model = FakeLLM(
        {
            "weather?": tool_call("lookup", city="Paris"),
            "real: Paris": "real",
            "mock: Paris": "mocked",
        }
    )
    agent = Agent(instructions="Help.", llm=model)

    @agent.tool
    async def lookup(ctx: RunContext, city: str) -> str:
        """Look the weather up."""
        seen.append(city)
        return f"real: {city}"

    with mock_tools(agent, {"lookup": lambda city: f"mock: {city}"}):
        mocked = asyncio.create_task(agent.run("weather?"))
    real = await agent.run("weather?")

    mocked_result = await mocked
    assert mocked_result.status == "completed" and mocked_result.output == "mocked"
    assert real.status == "completed" and real.output == "real"
    assert seen == ["Paris"]


# -- outputs: validators, typed outputs and output functions


@dataclass
class Reservation:
    reference: str


async def test_an_output_validator_retries_then_replaces_the_accepted_output() -> None:
    """`ModelRetry` from a validator sends its message back to the model; the value the
    validator returns is the run's output."""
    model = FakeLLM(["unverified", "verified"])
    agent = Agent(instructions="Help.", llm=model)

    @agent.output_validator
    def check(ctx: RunContext, output: str) -> str:
        if output != "verified":
            raise ModelRetry("Verify the booking first.")
        return "Booking verified."

    result = await agent.run("Booking")

    assert result.status == "completed" and result.output == "Booking verified."
    retried = [item.text_content for item in model.requests[1].items if item.type == "message"]
    assert any("Verify the booking first." in (text or "") for text in retried)


async def test_a_typed_output_is_validated_and_retried_before_acceptance() -> None:
    """An output that fails its type's validation goes back to the model as an error."""

    class Booking(BaseModel):
        nights: Annotated[int, Field(ge=1)]

    model = FakeLLM([tool_call("final_result", nights=0), tool_call("final_result", nights="2")])

    result = await Agent(instructions="Help.", llm=model, output_type=Booking).run("Book")

    assert result.status == "completed" and result.output == Booking(nights=2)


async def test_an_output_function_resolves_the_value_with_the_runs_context() -> None:
    """An output function gets the run's context; its `ModelRetry` asks the model again."""

    async def resolve(ctx: RunContext[dict[str, str]], code: str) -> Reservation:
        if code not in ctx.context:
            raise ModelRetry(f"no booking {code}")
        return Reservation(reference=ctx.context[code])

    model = FakeLLM([tool_call("final_result", code="X"), tool_call("final_result", code="A1")])
    agent = Agent[dict[str, str], Reservation](instructions="Help.", output_type=resolve, llm=model)

    result = await agent.run("my booking", context={"A1": "BK-1"})

    assert result.status == "completed" and result.output == Reservation(reference="BK-1")


async def test_a_read_back_function_asks_for_a_missing_detail_then_reads_back() -> None:
    """A `needs_approval` function sees the call's arguments: it can ask for a detail first,
    then return the read-back the person approves."""

    def read_back(ctx: RunContext, date: str, email: str | None = None) -> str | InputRequired:
        if email is None:
            return InputRequired("Which email?", verbatim=True)
        return f"Book {date} to {email}?"

    model = FakeLLM(
        [tool_call("book", date="monday"), tool_call("book", date="monday", email="jo@example.com")]
    )
    desk = Agent(instructions="Help.", llm=model)

    @desk.tool(needs_approval=read_back)
    async def book(ctx: RunContext, date: str, email: str | None = None) -> str:
        """Book."""
        return "booked"

    missing = await desk.run("book monday")
    asked = await desk.run("jo@example.com", message_history=missing.all_messages())

    assert missing.status == "input_required" and missing.question == "Which email?"
    assert asked.status == "input_required" and asked.question == "Book monday to jo@example.com?"


async def test_a_python_judge_gets_the_conversation_reference_and_model() -> None:
    """A judge group hands each Python judge the very conversation, reference and model."""
    seen: list[tuple[ChatContext, ChatContext | None, LLM | None]] = []

    class Recorder(Judge):
        def __init__(self) -> None:
            super().__init__(name="recorder")

        async def evaluate(
            self,
            *,
            chat_ctx: ChatContext,
            reference: ChatContext | None = None,
            llm: LLM | None = None,
        ) -> JudgmentResult:
            seen.append((chat_ctx, reference, llm))
            return JudgmentResult(verdict="pass", reasoning="recorded")

    conversation = ChatContext()
    conversation.add_message(role="user", content="Book the 9am slot.")
    reference = ChatContext()
    reference.add_message(role="assistant", content="Booked for 10am.")
    model = FakeLLM()

    evaluation = await JudgeGroup(llm=model, judges=[Recorder()]).evaluate(
        conversation, reference=reference
    )

    assert evaluation.judgments["recorder"].passed
    [(chat_ctx, judged_reference, judge_model)] = seen
    assert chat_ctx is conversation and judged_reference is reference and judge_model is model


async def test_a_handlers_typed_output_is_the_tasks_data_artifact(
    a2a_door: Callable[[A2AHandler], AbstractAsyncContextManager[httpx.AsyncClient]],
) -> None:
    """A pydantic value an A2A handler returns is sent as the task's data part."""

    class Answer(BaseModel):
        reservation: str

    async def handler(ctx: A2ASessionContext) -> Answer:
        return Answer(reservation="R42")

    async with a2a_door(handler) as door:
        response = await door.post("/v1", json=message_send("book"))

    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["data"] == {"reservation": "R42"}


# -- a call's hooks, nodes and speech callbacks


async def test_instructions_functions_add_to_the_static_instructions() -> None:
    """The model's system message holds the static instructions and what `@instructions`
    functions return."""
    model = FakeLLM(["hi"])
    agent = Agent(instructions="static part", llm=model)

    @agent.instructions
    def dynamic(ctx: RunContext) -> str:
        return "dynamic part"

    await agent.run("hello")

    [system] = [i for i in model.requests[0].items if i.type == "message" and i.role == "system"]
    assert "static part" in (system.text_content or "")
    assert "dynamic part" in (system.text_content or "")


async def test_on_exit_receives_the_session_on_replacement_and_on_close() -> None:
    """`on_exit` runs with the session when another agent takes over and when the session
    closes."""
    first, second = Agent(instructions="first"), Agent(instructions="second")
    exited: list[tuple[str, VoiceSession]] = []

    @first.on_exit
    async def leave_first(session: VoiceSession) -> None:
        exited.append(("first", session))

    @second.on_exit
    async def leave_second(session: VoiceSession) -> None:
        exited.append(("second", session))

    async with VoiceSession(vad=None) as session:
        await session.start(first)
        session.update_agent(second)
        await until(lambda: exited, "the first agent never exited")

    assert exited == [("first", session), ("second", session)]


async def test_a_session_tts_node_runs_for_each_agent_holding_the_call() -> None:
    """A session's `tts_node` speaks for whichever agent holds the call."""
    spoken_for: list[str] = []
    session = VoiceSession(vad=None, tts=FakeTTS())
    session.output.audio = FakeAudioOutput()

    @session.tts_node
    def voice(
        ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        spoken_for.append(ctx.agent.name)
        return session.default.tts_node(ctx, text, model_settings)

    async with session:
        await session.start(Agent(instructions="Welcome.", name="reception"))
        await session.say("One moment.")
        handed_off = asyncio.Event()

        @session.on("conversation_item_added")
        def on_handoff(event: ConversationItemAddedEvent) -> None:
            if event.item.type == "agent_handoff":
                handed_off.set()

        session.update_agent(Agent(instructions="Book.", name="bookings"))
        await handed_off.wait()
        await session.say("Anything else?")

    assert spoken_for == ["reception", "bookings"]


async def test_a_done_callback_runs_after_the_speechs_own_events() -> None:
    """A speech's done callback runs once the message it added has been delivered."""
    session = VoiceSession(vad=None, tts=FakeTTS())
    session.output.audio = FakeAudioOutput()
    added: list[str | None] = []
    at_done: list[list[str | None]] = []

    @session.on("conversation_item_added")
    def on_message(event: ConversationItemAddedEvent) -> None:
        if event.item.type == "message":
            added.append(event.item.text_content)

    async with session:
        await session.start(Agent(instructions="Help."))
        speech = session.say("One moment.")
        speech.add_done_callback(lambda _: at_done.append(list(added)))
        await speech
        await until(lambda: at_done, "the done callback never ran")

    assert at_done == [["One moment."]]


def _texts(items: list[ChatItem]) -> list[str | None]:
    return [item.text_content for item in items if item.type == "message"]


async def test_an_event_callback_reads_the_history_as_of_its_event() -> None:
    """Inside a `conversation_item_added` callback, and in tasks it starts, `session.history`
    ends at that event's item, even after later items arrived."""
    session = VoiceSession(vad=None, tts=FakeTTS())
    session.output.audio = FakeAudioOutput()
    later = asyncio.Event()
    seen: list[list[str | None]] = []
    read_later: list[asyncio.Task[list[str | None]]] = []

    async def read_after_later_items() -> list[str | None]:
        await later.wait()
        return _texts(session.history.items)

    @session.on("conversation_item_added")
    def on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message":
            seen.append(_texts(session.history.items))
            read_later.append(asyncio.ensure_future(read_after_later_items()))

    async with session:
        await session.start(Agent(instructions="Greet."))
        await session.say("One moment.")
        await session.say("Anything else?")
        await until(lambda: len(seen) == 2, "both messages were delivered")
        later.set()
        read = [await task for task in read_later]

    assert seen == [["One moment."], ["One moment.", "Anything else?"]]
    assert read == seen


async def test_an_agent_state_callback_reads_the_state_it_announces() -> None:
    """Inside an `agent_state_changed` callback, `session.agent_state` is the event's new state."""
    session = VoiceSession(vad=None, llm=FakeLLM(["Hello."]), tts=FakeTTS())
    session.output.audio = FakeAudioOutput()
    seen: list[tuple[str, str]] = []

    @session.on("agent_state_changed")
    def on_state(ev: AgentStateChangedEvent) -> None:
        seen.append((ev.new_state, session.agent_state))

    async with session:
        await session.start(Agent(instructions="Greet."))
        await session.generate_reply(user_input="hi")

    assert "speaking" in [announced for announced, _ in seen]
    assert all(announced == read for announced, read in seen)


async def test_a_task_tool_reports_progress_then_returns_a_validated_completion() -> None:
    """A `VoiceTask` agent's tool can report progress and complete; its output validator
    shapes what the parent's tool receives."""
    release = asyncio.Event()
    collector = Agent(
        instructions="Collect the reference.",
        output_type=str,
        llm=FakeLLM([tool_call("collect", call_id="collect-1")]),
    )

    @collector.output_validator
    async def normalize(ctx: RunContext, value: str) -> str:
        return value.upper()

    @collector.tool
    async def collect(ctx: RunContext) -> Completed[str]:
        """Collect the reference."""
        await ctx.update("Collecting the reference.")
        await release.wait()
        return Completed("booking-1")

    front = Agent(instructions="Help.")
    received: list[str] = []

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Find the booking."""
        received.append(await VoiceTask(agent=collector).run())
        return received[-1]

    def progress(event: ToolExecutionUpdatedEvent) -> None:
        if isinstance(event.update, ToolCallUpdated) and event.update.call_id == "collect-1":
            release.set()

    async with VoiceSession(vad=None, llm=FakeLLM([tool_call("lookup"), "Okay."])) as session:
        session.on("tool_execution_updated", progress)
        await session.start(front)
        await session.generate_reply(user_input="find my booking")

        assert received == ["BOOKING-1"]
        assert session.current_agent is front
