"""Shared execution vectors (C-21): each vector runs through the direct run, a TextSession turn
and a call's VoiceSession.run, and is asserted the same way on the invocation's events.

A vector is one scripted model conversation and one agent definition. Behavior that only a
call has (duplicate-call policies, the floor, on_enter, playback) is not here: those vectors
belong to the voice suite, and the text entrypoints neither run nor promise them."""

from __future__ import annotations

import asyncio
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
    InputRequired,
    ModelRetry,
    RunContext,
    TextSession,
    TextSessionState,
    TextSessionTurn,
    ToolResult,
    UnexpectedModelBehavior,
    VoiceSession,
)
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChatItem,
    ChoiceDelta,
    FunctionCallOutput,
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
    status: str
    question: str | None
    history: list[ChatItem]


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


def _settled(
    output: object,
    items: list[ChatItem],
    status: str,
    question: str | None,
    history: list[ChatItem],
) -> Settled:
    return Settled(
        output=output,
        expect=RunAssert(_events(items)),
        status=status,
        question=question,
        history=history,
    )


async def _direct(agent: Agent[Any, Any], *messages: str, context: Any = None) -> Settled:
    history: list[ChatItem] = []
    for message in messages:
        result = await agent.run(message, message_history=history, context=context)
        history = result.all_messages()
    return _settled(
        result.output if result.status == "completed" else None,
        result.new_messages(),
        result.status,
        result.question if result.status == "input_required" else None,
        result.all_messages(),
    )


async def _text_session(agent: Agent[Any, Any], *messages: str, context: Any = None) -> Settled:
    state: TextSessionState | None = None
    for message in messages:
        turn = await TextSession(agent=agent).run(message, state=state, context=context)
        state = turn.state
    result = turn.result
    return _settled(
        result.output if result.status == "completed" else None,
        result.new_messages(),
        result.status,
        result.question if result.status == "input_required" else None,
        result.all_messages(),
    )


async def _voice(agent: Agent[Any, Any], *messages: str, context: Any = None) -> Settled:
    from .fake_io import FakeTextOutput

    async with VoiceSession(context=context) as session:
        session.output.transcription = FakeTextOutput()
        await session.start(agent)
        for message in messages:
            result: VoiceRunResult[Any] = await session.run(user_input=message)
        return _settled(
            result.output,
            result.new_messages(),
            result.status,
            result.question,
            result.all_messages(),
        )


Entrypoint = Callable[..., Awaitable[Settled]]

ENTRYPOINTS = [
    pytest.param(_direct, id="direct"),
    pytest.param(_text_session, id="text_session"),
    pytest.param(_voice, id="voice"),
]


@pytest.fixture(params=ENTRYPOINTS)
def run(request: pytest.FixtureRequest) -> Entrypoint:
    return request.param  # type: ignore[no-any-return]


# VoiceSession.run accepts output outside AgentRun: no output validators, no final_result, no
# output retries. These vectors flip once a call's reply composes the shared AgentRun.
VOICE_OUTPUT_GAP = pytest.mark.xfail(
    strict=True, reason="a call's reply does not run AgentRun's output acceptance yet"
)
OUTPUT_ENTRYPOINTS = [*ENTRYPOINTS[:2], pytest.param(_voice, id="voice", marks=VOICE_OUTPUT_GAP)]


async def test_a_plain_reply_is_the_output(run: Entrypoint) -> None:
    agent = Agent(name="desk", instructions="Help", llm=Script(Reply("Hello there.")))
    settled = await run(agent, "Hi")
    assert settled.output == "Hello there."
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()


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
    run: Entrypoint, arguments: dict[str, object], note: str | None
) -> None:
    seen: list[tuple[int, str, str | None]] = []
    model = Script(call("book", arguments, "c1"), Reply("Booked two nights."))
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
    assert seen == [(2, "double", note)]
    assert settled.output == "Booked two nights."
    settled.expect.next_event().is_function_call(name="book", arguments=arguments)
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


@pytest.mark.parametrize(
    "kind", ["tool_error", "internal", "invalid", "failed", "rejected", "completed"]
)
async def test_tool_outcomes_are_recorded_before_the_model_continues(
    run: Entrypoint, kind: str
) -> None:
    from livekit.agents import Completed, Failed, Rejected, ToolError

    model = Script(call("lookup", {}, "c1"), Reply("Handled."))
    agent = Agent(instructions="Help", llm=model)

    @agent.tool
    async def lookup(ctx: RunContext) -> Any:
        """Look up the booking."""
        if kind == "tool_error":
            raise ToolError("No booking")
        if kind == "internal":
            raise RuntimeError("private credentials")
        if kind == "invalid":
            return object()
        if kind == "failed":
            return Failed("No booking")
        if kind == "rejected":
            return Rejected("No booking")
        return Completed("Booking 42")

    settled = await run(agent, "Lookup")
    assert settled.output == "Handled."
    settled.expect.next_event().is_function_call(name="lookup")
    output = (
        settled.expect.next_event()
        .is_function_call_output(is_error=kind != "completed")
        .event()
        .item
    )
    expected = {
        "tool_error": "No booking",
        "internal": "An internal error occurred",
        "invalid": "the tool returned an invalid output",
        "failed": "No booking",
        "rejected": "No booking",
        "completed": "Booking 42",
    }
    assert output.output == expected[kind]
    assert any(
        i.type == "function_call_output" and i.output == output.output
        for i in model.requests[1].items
    )
    settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()


@pytest.mark.parametrize("verbatim", [False, True])
async def test_a_tool_can_end_the_turn_without_another_model_request(
    run: Entrypoint, verbatim: bool
) -> None:
    model = Script(call("receipt", {}, "c1"))
    agent = Agent(instructions="Help", llm=model)

    @agent.tool
    async def receipt(ctx: RunContext) -> ToolResult:
        """Return the receipt."""
        return ToolResult("Receipt 42", reply_required=False, verbatim=verbatim)

    settled = await run(agent, "Receipt")
    assert len(model.requests) == 1
    settled.expect.next_event().is_function_call(name="receipt")
    output = settled.expect.next_event().is_function_call_output(is_error=False).event().item
    assert "Receipt 42" in output.output
    if verbatim and run is not _voice:
        assert "Receipt 42" in str(settled.output)


async def test_stop_response_ends_the_turn_without_a_reply(run: Entrypoint) -> None:
    from livekit.agents import StopResponse

    model = Script(call("stop", {}, "c1"))
    agent = Agent(instructions="Help", llm=model)

    @agent.tool
    def stop(ctx: RunContext) -> str:
        """End this response."""
        raise StopResponse()

    settled = await run(agent, "Stop")
    assert settled.output is None
    assert len(model.requests) == 1
    settled.expect.next_event().is_function_call(name="stop")
    settled.expect.next_event().is_function_call_output(is_error=False)
    settled.expect.no_more_events()


async def test_nested_invocations_restore_the_parent_context(run: Entrypoint) -> None:
    child = Agent(
        instructions="Child", llm=Script(call("inspect", {}, "child"), Reply("Child done."))
    )
    parent = Agent(
        instructions="Parent", llm=Script(call("delegate", {}, "parent"), Reply("Parent done."))
    )
    seen: list[tuple[str, str]] = []

    @child.tool
    async def inspect(ctx: RunContext) -> str:
        """Inspect the child invocation."""
        seen.append(("child", ctx.function_call.call_id))
        return "inspected"

    @parent.tool
    async def delegate(ctx: RunContext) -> str:
        """Invoke the child."""
        seen.append(("before", ctx.function_call.call_id))
        result = await child.run("Inspect")
        assert result.status == "completed" and result.output == "Child done."
        seen.append(("after", ctx.function_call.call_id))
        return "delegated"

    settled = await run(parent, "Delegate")
    assert settled.output == "Parent done."
    assert seen == [("before", "parent"), ("child", "child"), ("after", "parent")]


async def test_concurrent_invocations_keep_their_application_contexts_separate(
    run: Entrypoint,
) -> None:
    class Model(Script):
        def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:
            self.requests.append(chat_ctx.copy())
            last = chat_ctx.items[-1]
            if isinstance(last, FunctionCallOutput):
                reply = Reply(last.output)
            else:
                reply = call("inspect", {}, last.id)
            return _Stream(self, chat_ctx=chat_ctx, reply=reply)

    agent = Agent(instructions="Help", llm=Model())
    entered = 0
    both = asyncio.Event()
    seen: list[tuple[str, str]] = []

    @agent.tool
    async def inspect(ctx: RunContext[dict[str, str]]) -> str:
        """Inspect this invocation's context."""
        nonlocal entered
        before = ctx.context["name"]
        entered += 1
        if entered == 2:
            both.set()
        await both.wait()
        seen.append((before, ctx.context["name"]))
        ctx.context["visited"] = before
        return before

    ada, grace = {"name": "Ada"}, {"name": "Grace"}
    first, second = await asyncio.gather(
        run(agent, "Hi Ada", context=ada), run(agent, "Hi Grace", context=grace)
    )
    assert first.output == "Ada" and second.output == "Grace"
    assert sorted(seen) == [("Ada", "Ada"), ("Grace", "Grace")]
    assert ada == {"name": "Ada", "visited": "Ada"}
    assert grace == {"name": "Grace", "visited": "Grace"}


async def test_approval_changes_require_another_answer_and_each_answer_is_consumed_once(
    run: Entrypoint,
) -> None:
    model = Script(
        call("cancel", {"date": "Monday"}, "first"),
        call("cancel", {"date": "Tuesday"}, "changed"),
        call("cancel", {"date": "Tuesday", "notify": True}, "repeat"),
        Reply("Canceled Tuesday."),
        call("cancel", {"date": "Tuesday"}, "again"),
    )
    agent = Agent(instructions="Help", llm=model)
    ran: list[str] = []

    @agent.tool(needs_approval=True)
    async def cancel(ctx: RunContext, date: str, notify: bool = True) -> str:
        """Cancel the appointment."""
        ran.append(date)
        return "canceled"

    settled = await run(agent, "Cancel Monday", "Yes", "Yes", "Cancel Tuesday again")
    assert ran == ["Tuesday"]
    assert settled.status == "input_required" and "Tuesday" in (settled.question or "")
    answers = {i.call_id: i.output for i in settled.history if isinstance(i, FunctionCallOutput)}
    assert answers["first"] == answers["changed"] == "Yes"
    calls = [i for i in model.requests[3].items if i.type == "function_call"]
    assert [(i.call_id, i.extra["lk.approval"]) for i in calls] == [
        ("first", "asked"),
        ("changed", "asked"),
        ("repeat", "granted"),
    ]


@pytest.mark.parametrize("verbatim", [False, True])
async def test_input_required_waits_for_the_person_and_the_next_turn_continues(
    run: Entrypoint, verbatim: bool
) -> None:
    model = Script(
        call("ask", {}, "question"),
        *([] if verbatim else [Reply("Which date would you like?")]),
        Reply("Tuesday it is."),
    )
    agent = Agent(instructions="Help", llm=model)

    @agent.tool
    async def ask(ctx: RunContext) -> InputRequired:
        """Ask for the date."""
        return InputRequired("Which date?", verbatim=verbatim)

    settled = await run(agent, "Book", "Tuesday")
    assert settled.status == "completed" and settled.output == "Tuesday it is."
    request = model.requests[-1]
    answered = next(i for i in request.items if i.type == "function_call_output")
    assert answered.output == "Tuesday"
    stored = next(i for i in settled.history if isinstance(i, FunctionCallOutput))
    assert stored.output == "Tuesday"
    assert len(model.requests) == (2 if verbatim else 3)


async def test_cancellation_before_launch_does_not_run_a_tool(run: Entrypoint) -> None:
    model = Script(call("effect", {}, "c1"))
    agent = Agent(instructions="Help", llm=model)
    ran: list[str] = []

    @agent.tool
    async def effect(ctx: RunContext) -> str:
        """Perform an effect."""
        ran.append("ran")
        return "done"

    entered, cleaned = asyncio.Event(), asyncio.Event()

    @agent.instructions
    async def prepare(ctx: RunContext) -> str:
        entered.set()
        try:
            await asyncio.Future[None]()
        finally:
            cleaned.set()
        return "unreachable"

    if run is _voice:
        async with VoiceSession() as session:
            await session.start(agent)
            result: VoiceRunResult[Any] = session.run(user_input="Act")
            try:
                await entered.wait()
            finally:
                await session.interrupt(force=True)
            await result
    else:
        invocation = asyncio.ensure_future(run(agent, "Act"))
        try:
            await entered.wait()
        finally:
            invocation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await invocation
    assert cleaned.is_set()
    assert ran == [] and model.requests == []


@pytest.mark.parametrize("late_result", [False, True])
async def test_cancellation_after_launch_settles_the_body_without_starting_a_followup(
    run: Entrypoint, late_result: bool
) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()
    model = Script(call("wait", {}, "c1"))
    agent = Agent(instructions="Help", llm=model)

    @agent.tool
    async def wait(ctx: RunContext) -> str:
        """Wait for the service."""
        entered.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            if late_result:
                return "late result"
            raise
        finally:
            cleaned.set()
        return "unreachable"

    invocation = asyncio.ensure_future(run(agent, "Wait"))
    try:
        await entered.wait()
    finally:
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
    assert cleaned.is_set()
    assert len(model.requests) == 1


async def test_text_reaches_the_consumer_before_the_model_ends(run: Entrypoint) -> None:
    from livekit import rtc
    from livekit.agents import tts

    from .fake_io import FakeAudioOutput

    eof = asyncio.Event()

    class GatedStream(_Stream):
        async def _run(self) -> None:
            await super()._run()
            await eof.wait()

    class Model(Script):
        def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:
            self.requests.append(chat_ctx.copy())
            return GatedStream(
                self,
                chat_ctx=chat_ctx,
                reply=Reply("Your booking is ready. Let me check the next step."),
            )

    model = Model()
    agent = Agent(instructions="Help", llm=model)
    if run is _voice:
        seen = asyncio.Event()

        class Audio(FakeAudioOutput):
            async def capture_frame(self, frame: rtc.AudioFrame) -> None:
                await super().capture_frame(frame)
                seen.set()

        class SpeechStream(tts.SynthesizeStream):
            async def _run(self, output_emitter: tts.AudioEmitter) -> None:
                output_emitter.initialize(
                    request_id="streaming",
                    sample_rate=24000,
                    num_channels=1,
                    mime_type="audio/pcm",
                    stream=True,
                )
                output_emitter.start_segment(segment_id="sentence")
                async for part in self.input:
                    if isinstance(part, str):
                        output_emitter.push(bytes(4800))
                output_emitter.end_segment()

        class Speech(tts.TTS):
            def __init__(self) -> None:
                super().__init__(
                    capabilities=tts.TTSCapabilities(streaming=True),
                    sample_rate=24000,
                    num_channels=1,
                )

            def stream(
                self, *, conn_options: Any = DEFAULT_API_CONNECT_OPTIONS
            ) -> tts.SynthesizeStream:
                return SpeechStream(tts=self, conn_options=conn_options)

            def synthesize(self, text: str, **kwargs: Any) -> tts.ChunkedStream:
                raise AssertionError("the voice reply should stream")

        async with VoiceSession(tts=Speech()) as session:
            session.output.audio = Audio()
            await session.start(agent)
            voice_result: VoiceRunResult[Any] = session.run(user_input="Booking")
            try:
                await seen.wait()
                assert not eof.is_set()
            finally:
                eof.set()
            assert (
                await voice_result
            ).output == "Your booking is ready. Let me check the next step."
    else:
        stream = (
            agent.run_stream("Booking")
            if run is _direct
            else TextSession(agent=agent).run_stream("Booking")
        )
        async with stream:
            chunks = stream.stream_text(delta=True)
            try:
                assert await anext(chunks) == "Your booking is ready. Let me check the next step."
                assert not eof.is_set()
            finally:
                eof.set()
            assert [chunk async for chunk in chunks] == []
            result = await stream.result()
            if isinstance(result, TextSessionTurn):
                result = result.result
            assert result.status == "completed"
            assert result.output == "Your booking is ready. Let me check the next step."
    assert len(model.requests) == 1


async def test_a_none_result_continues_in_text_and_asks_no_reply_on_a_call(run: Entrypoint) -> None:
    model = Script(call("note", {}, "c1"), *([] if run is _voice else [Reply("Noted.")]))
    agent = Agent(instructions="Help", llm=model)

    @agent.tool
    async def note(ctx: RunContext) -> None:
        """Note this request."""

    settled = await run(agent, "Note it")
    assert settled.output == (None if run is _voice else "Noted.")
    assert len(model.requests) == (1 if run is _voice else 2)
    settled.expect.next_event().is_function_call(name="note")
    settled.expect.next_event().is_function_call_output(is_error=False)
    if run is not _voice:
        settled.expect.next_event().is_message(role="assistant")
    settled.expect.no_more_events()


@pytest.mark.parametrize("run", OUTPUT_ENTRYPOINTS)
async def test_an_output_validator_retries_then_replaces_the_accepted_output(
    run: Entrypoint,
) -> None:
    model = Script(Reply("unverified"), Reply("verified"))
    agent = Agent(instructions="Help", llm=model)
    seen: list[str] = []

    @agent.output_validator
    def check(ctx: RunContext, output: str) -> str:
        seen.append(output)
        if output != "verified":
            raise ModelRetry("Verify the booking first.")
        return "Booking verified."

    settled = await run(agent, "Booking")
    assert seen == ["unverified", "verified"]
    assert settled.output == "Booking verified."
    assert any(
        i.type == "message" and "Verify the booking first." in (i.text_content or "")
        for i in model.requests[1].items
    )


@pytest.mark.parametrize("run", OUTPUT_ENTRYPOINTS)
async def test_an_output_that_keeps_failing_exhausts_only_its_output_retries(
    run: Entrypoint,
) -> None:
    model = Script(Reply("wrong"), Reply("still wrong"))
    agent = Agent(instructions="Help", llm=model)

    @agent.output_validator
    def check(ctx: RunContext, output: str) -> str:
        raise ModelRetry("Verify the booking first.")

    with pytest.raises(UnexpectedModelBehavior, match="output"):
        await run(agent, "Booking")
    assert len(model.requests) == 2


@pytest.mark.parametrize("run", OUTPUT_ENTRYPOINTS)
async def test_a_typed_output_is_validated_and_retried_before_acceptance(run: Entrypoint) -> None:
    from pydantic import BaseModel

    class Booking(BaseModel):
        nights: Annotated[int, Field(ge=1)]

    model = Script(
        call("final_result", {"nights": 0}, "bad"),
        call("final_result", {"nights": "2"}, "good"),
    )
    agent = Agent(instructions="Help", llm=model, output_type=Booking)
    settled = await run(agent, "Book two nights")
    assert settled.output == Booking(nights=2)
    assert len(model.requests) == 2
    assert any(
        i.type == "function_call_output" and i.is_error and "nights" in i.output
        for i in model.requests[1].items
    )


@pytest.mark.xfail(
    strict=True, reason="the transcription-only path does not deliver a verbatim question"
)
async def test_a_verbatim_voice_question_reaches_the_transcription_output() -> None:
    from .fake_io import FakeTextOutput

    agent = Agent(instructions="Help", llm=Script(call("ask", {}, "question")))

    @agent.tool
    async def ask(ctx: RunContext) -> InputRequired:
        """Ask for the date."""
        return InputRequired("Which date?", verbatim=True)

    output = FakeTextOutput()
    async with VoiceSession() as session:
        session.output.transcription = output
        await session.start(agent)
        result: VoiceRunResult[Any] = await session.run(user_input="Book")
        assert result.status == "input_required" and result.question == "Which date?"
        await session.wait_for_idle()
        assert "".join(output._messages) == "Which date?"
