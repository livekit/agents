"""The runner: one session answering requests, and which answer belongs to which request."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.a2a import TaskInput, TaskUpdate
from livekit.agents.a2a.runner import REQUEST_ID_KEY, RequestRun, SessionRunner
from livekit.agents.llm import ChatContext, FunctionCall, FunctionToolCall, ToolFlag
from livekit.agents.voice.tool_executor import _RunningTasks

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class _AnsweringLLM(FakeLLM):
    """A FakeLLM that answers anything it has no scripted reply for.

    A deferred tool reply is generated from a rendered instruction, so a strict map would
    return silence there and hide whether the reply landed at all.
    """

    def __init__(self, *, fake_responses: list[FakeLLMResponse], fallbacks: list[str]) -> None:
        super().__init__(fake_responses=fake_responses)
        self._fallbacks = list(fallbacks)

    def chat(self, *, chat_ctx: Any, **kwargs: Any) -> Any:
        stream = FakeLLMStream(
            self, chat_ctx=chat_ctx, tools=[], conn_options=kwargs["conn_options"]
        )
        index = stream._get_index_text()
        if index not in self.fake_response_map:
            content = self._fallbacks.pop(0) if self._fallbacks else "nothing left to say"
            self._fake_response_map[index] = FakeLLMResponse(
                input=index, content=content, ttft=0.0, duration=0.0
            )
        return super().chat(chat_ctx=chat_ctx, **kwargs)


def _tool_call(name: str, call_id: str, arguments: str = "{}") -> FunctionToolCall:
    return FunctionToolCall(type="function", name=name, arguments=arguments, call_id=call_id)


def _says(
    input: str, content: str, *, calls: list[FunctionToolCall] | None = None
) -> FakeLLMResponse:
    return FakeLLMResponse(
        input=input, content=content, ttft=0.0, duration=0.0, tool_calls=calls or []
    )


async def _serve(agent: Agent, *, llm: FakeLLM) -> tuple[AgentSession, SessionRunner]:
    session = AgentSession(llm=llm)
    await session.start(agent=agent)
    runner = SessionRunner(session)
    return session, runner


async def _collect(run: RequestRun, *, timeout: float = 30.0) -> list[TaskUpdate]:
    async def _read() -> list[TaskUpdate]:
        async with run:
            return [update async for update in run]

    return await asyncio.wait_for(_read(), timeout=timeout)


async def _close(session: AgentSession, runner: SessionRunner) -> None:
    await asyncio.wait_for(runner.aclose(), timeout=10.0)
    await asyncio.wait_for(session.aclose(), timeout=10.0)


def _texts(updates: list[TaskUpdate]) -> list[str]:
    return [u.text for u in updates if u.text]


async def test_a_request_is_answered_by_the_turn_it_opened() -> None:
    llm = _AnsweringLLM(
        fake_responses=[_says("what is the change fee", "The change fee is $75.")],
        fallbacks=[],
    )
    session, runner = await _serve(Agent(instructions="fare desk"), llm=llm)

    updates = await _collect(
        runner.submit(TaskInput(instruction="what is the change fee"), request_id="r1")
    )
    await _close(session, runner)

    assert [(u.state, u.text) for u in updates] == [("completed", "The change fee is $75.")]


async def test_each_item_travels_once() -> None:
    """generate_reply emits speech_created before returning, so its handle is an orphan
    first and the turn's own second; claiming it twice would double every item."""

    @function_tool
    async def check_fares(ctx: RunContext) -> str:
        """Check the fares."""
        return "fare is 240 USD"

    llm = _AnsweringLLM(
        fake_responses=[_says("what is the fare", "", calls=[_tool_call("check_fares", "cf1")])],
        fallbacks=["It is 240 USD."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[check_fares]), llm=llm)

    updates = await _collect(
        runner.submit(TaskInput(instruction="what is the fare"), request_id="r1")
    )
    await _close(session, runner)

    calls = [u.item for u in updates if u.item is not None and u.item.type == "function_call"]
    assert [item.call_id for item in calls] == ["cf1"]


async def test_a_tools_report_is_relayed_as_written_and_draws_no_reply() -> None:
    """ctx.update() releases the turn, so the answer comes from the return, not the report."""
    finished: list[str] = []

    @function_tool
    async def check_fares(ctx: RunContext) -> str:
        """Slow work that reports progress."""
        await ctx.update("checking the fare rules")
        await asyncio.sleep(2)
        finished.append("done")
        return "fare is 240 USD"

    llm = _AnsweringLLM(
        fake_responses=[_says("what is the fare", "", calls=[_tool_call("check_fares", "cf1")])],
        # only the return draws a reply; a reply to the report would take this line instead
        fallbacks=["It is 240 USD."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[check_fares]), llm=llm)

    updates = await _collect(
        runner.submit(TaskInput(instruction="what is the fare"), request_id="r1")
    )
    await _close(session, runner)

    assert finished == ["done"]
    assert "checking the fare rules" in _texts(updates)
    # the report was relayed, and the model was never asked to restate it
    assert "nothing left to say" not in _texts(updates)
    # the answer lands on the deferred reply speech, which must not count as open work
    # against itself and announce the answer as progress first
    assert [(u.state, u.text) for u in updates if u.text] == [
        ("working", "checking the fare rules"),
        ("completed", "It is 240 USD."),
    ]


async def test_a_person_s_turn_is_answered_by_the_expert_s_own_words() -> None:
    """A delegation relays the tool's words; a person's turn draws a line about them, and
    the report then travels as an item rather than being said twice."""

    @function_tool
    async def check_fares(ctx: RunContext) -> str:
        """Slow work that reports progress."""
        await ctx.update("checking the fare rules")
        return "fare is 240 USD"

    llm = _AnsweringLLM(
        fake_responses=[_says("what is the fare", "", calls=[_tool_call("check_fares", "cf1")])],
        fallbacks=["Looking that up.", "It is 240 USD."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[check_fares]), llm=llm)

    updates = await _collect(runner.submit(TaskInput(text="what is the fare"), request_id="r1"))
    await _close(session, runner)

    # the tool's own words stay in the item; what is said is the expert's line about them
    assert "checking the fare rules" not in _texts(updates)
    reports = [
        u for u in updates if u.item is not None and getattr(u.item, "update_of", None) == "cf1"
    ]
    assert len(reports) == 1 and reports[0].text == ""
    assert _texts(updates) == ["Looking that up.", "It is 240 USD."]


async def test_a_report_travels_as_a_call_naming_what_it_reports_for() -> None:
    @function_tool
    async def check_fares(ctx: RunContext) -> str:
        """Slow work that reports progress."""
        await ctx.update("checking the fare rules")
        return "fare is 240 USD"

    llm = _AnsweringLLM(
        fake_responses=[_says("what is the fare", "", calls=[_tool_call("check_fares", "cf1")])],
        fallbacks=["It is 240 USD."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[check_fares]), llm=llm)

    updates = await _collect(
        runner.submit(TaskInput(instruction="what is the fare"), request_id="r1")
    )
    await _close(session, runner)

    reports = [u for u in updates if u.text == "checking the fare rules"]
    assert len(reports) == 1
    item = reports[0].item
    assert item is not None and item.type == "function_call"
    assert item.update_of == "cf1"
    assert item.name == "check_fares"


async def test_a_silent_report_travels_as_an_item_with_no_text() -> None:
    """Inaudible, not invisible: a chat UI renders it and nobody on either side says it."""

    @function_tool
    async def check_fares(ctx: RunContext) -> str:
        """Slow work that reports where nobody is meant to hear it."""
        await ctx.update("checking the fare rules", silent=True)
        return "fare is 240 USD"

    llm = _AnsweringLLM(
        fake_responses=[_says("what is the fare", "", calls=[_tool_call("check_fares", "cf1")])],
        fallbacks=["It is 240 USD."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[check_fares]), llm=llm)

    updates = await _collect(
        runner.submit(TaskInput(instruction="what is the fare"), request_id="r1")
    )
    await _close(session, runner)

    reports = [
        u for u in updates if u.item is not None and getattr(u.item, "update_of", None) == "cf1"
    ]
    assert len(reports) == 1 and reports[0].text == ""
    assert _texts(updates) == ["It is 240 USD."]


async def test_a_line_the_expert_said_outright_is_said_as_written() -> None:
    @function_tool
    async def read_back(ctx: RunContext) -> str:
        """Read a code back to the caller."""
        ctx.session.say("Your confirmation code is AB12.")
        return "code read back"

    llm = _AnsweringLLM(
        fake_responses=[
            # a line of the model's own in the same turn as the call, so the two sources differ
            _says("read the code", "Let me read that back.", calls=[_tool_call("read_back", "rb1")])
        ],
        fallbacks=["All set."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[read_back]), llm=llm)

    updates = await _collect(runner.submit(TaskInput(instruction="read the code"), request_id="r1"))
    await _close(session, runner)

    by_text = {u.text: u.verbatim for u in updates if u.text}
    assert by_text["Your confirmation code is AB12."] is True
    # the model wrote this one, so the caller phrases it rather than reading it out
    assert by_text["Let me read that back."] is False


async def test_a_directive_rides_the_answer() -> None:
    @function_tool
    async def say_goodbye(ctx: RunContext) -> str:
        """Called when the caller is done."""
        request = ctx.request
        assert request is not None
        request.set_directive("end_session", reason="user_request")
        return "wrapped up"

    llm = _AnsweringLLM(
        fake_responses=[_says("that is all", "", calls=[_tool_call("say_goodbye", "sg1")])],
        fallbacks=["Thanks for calling."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[say_goodbye]), llm=llm)

    updates = await _collect(runner.submit(TaskInput(instruction="that is all"), request_id="r1"))
    await _close(session, runner)

    (answer,) = [u for u in updates if u.state == "completed"]
    assert answer.directive is not None
    assert (answer.directive.kind, answer.directive.reason) == ("end_session", "user_request")
    # advice only: nothing here closed the session
    assert all(u.directive is None for u in updates if u.state == "working")


async def test_a_tool_sees_the_callers_metadata() -> None:
    seen: list[dict[str, Any]] = []

    @function_tool
    async def look_up(ctx: RunContext) -> str:
        """Look something up."""
        request = ctx.request
        assert request is not None
        seen.append(dict(request.metadata))
        return "found it"

    llm = _AnsweringLLM(
        fake_responses=[_says("look it up", "", calls=[_tool_call("look_up", "lu1")])],
        fallbacks=["Found."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[look_up]), llm=llm)

    await _collect(
        runner.submit(
            TaskInput(instruction="look it up", metadata={"customer_id": "c-42"}),
            request_id="r1",
        )
    )
    await _close(session, runner)

    assert seen == [{"customer_id": "c-42"}]


async def test_one_reply_covering_two_requests_is_carried_by_the_newest() -> None:
    """The coalescer says one thing about several results. One request carries it; the
    earlier one is being superseded and ends with what it had already said."""
    released = asyncio.Event()
    resume = asyncio.Event()

    @function_tool
    async def slow_lookup(ctx: RunContext) -> str:
        """Slow work that reports and then keeps going."""
        await ctx.update("looking")
        released.set()
        await resume.wait()
        return "the first fact"

    @function_tool
    async def quick_lookup(ctx: RunContext) -> str:
        """Work that reports and returns."""
        await ctx.update("also looking")
        return "the second fact"

    llm = _AnsweringLLM(
        fake_responses=[
            _says("first", "", calls=[_tool_call("slow_lookup", "sl1")]),
            _says("second", "", calls=[_tool_call("quick_lookup", "ql1")]),
        ],
        fallbacks=["Both facts together."],
    )
    session, runner = await _serve(
        Agent(instructions="fare desk", tools=[slow_lookup, quick_lookup]), llm=llm
    )

    first = runner.submit(TaskInput(instruction="first"), request_id="r1")
    reading_first = asyncio.create_task(_collect(first))
    await asyncio.wait_for(released.wait(), timeout=10.0)

    second = runner.submit(TaskInput(instruction="second"), request_id="r2")
    reading_second = asyncio.create_task(_collect(second))
    resume.set()

    first_updates = await asyncio.wait_for(reading_first, timeout=30.0)
    second_updates = await asyncio.wait_for(reading_second, timeout=30.0)
    await _close(session, runner)

    # the merged reply is said once, by the request that is still live
    assert "Both facts together." in _texts(second_updates)
    assert "Both facts together." not in _texts(first_updates)
    # and the earlier request still ends, with what it had
    assert [u.state for u in first_updates if u.state != "working"]


async def test_a_released_tool_still_reads_its_own_request() -> None:
    """The supersede path: a tool that released the floor keeps running while the next
    request is already under way, and must not read the one that overtook it."""
    seen: list[dict[str, Any]] = []
    released = asyncio.Event()
    resume = asyncio.Event()

    @function_tool
    async def slow_lookup(ctx: RunContext) -> str:
        """Slow work that reports and then keeps going."""
        await ctx.update("looking")
        released.set()
        await resume.wait()
        request = ctx.request
        assert request is not None
        seen.append(dict(request.metadata))
        return "found it"

    llm = _AnsweringLLM(
        fake_responses=[
            _says("first", "", calls=[_tool_call("slow_lookup", "sl1")]),
            _says("second", "done second"),
        ],
        fallbacks=["done first."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[slow_lookup]), llm=llm)

    first = runner.submit(TaskInput(instruction="first", metadata={"n": 1}), request_id="r1")
    reading_first = asyncio.create_task(_collect(first))
    await asyncio.wait_for(released.wait(), timeout=10.0)

    # the second request starts while the first request's tool is still running, and is held
    # open so it is still the newest when that tool reads
    second = runner.submit(TaskInput(instruction="second", metadata={"n": 2}), request_id="r2")
    await asyncio.wait_for(second.__anext__(), timeout=10.0)

    resume.set()
    await asyncio.wait_for(reading_first, timeout=30.0)
    await asyncio.wait_for(second.aclose(), timeout=10.0)
    await _close(session, runner)

    # the tool belongs to the first request, whatever has happened since
    assert seen == [{"n": 1}]


async def test_an_ordinary_session_has_no_request_to_direct() -> None:
    """The same tool body runs in a session nobody is waiting on."""
    branch: list[str] = []

    @function_tool
    async def say_goodbye(ctx: RunContext) -> str:
        """Called when the caller is done."""
        if (request := ctx.request) is not None:
            request.set_directive("end_session")
            branch.append("directive")
        else:
            branch.append("close")
        return "wrapped up"

    llm = _AnsweringLLM(
        fake_responses=[_says("bye", "", calls=[_tool_call("say_goodbye", "sg1")])],
        fallbacks=["Bye."],
    )
    session = AgentSession(llm=llm)
    await session.start(agent=Agent(instructions="voice", tools=[say_goodbye]))
    session.generate_reply(user_input="bye")
    await asyncio.sleep(5)
    await asyncio.wait_for(session.aclose(), timeout=10.0)

    assert branch == ["close"]


async def test_the_conversation_is_merged_once_across_requests() -> None:
    """The caller sends what it holds, whole; the receiver takes the delta by item id."""
    llm = _AnsweringLLM(
        fake_responses=[_says("first", "one"), _says("second", "two")],
        fallbacks=[],
    )
    session, runner = await _serve(Agent(instructions="fare desk"), llm=llm)

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="change my Monday flight", id="m1")
    await _collect(
        runner.submit(TaskInput(instruction="first", chat_ctx=chat_ctx), request_id="r1")
    )

    chat_ctx.add_message(role="user", content="and the Tuesday one", id="m2")
    await _collect(
        runner.submit(TaskInput(instruction="second", chat_ctx=chat_ctx), request_id="r2")
    )

    items = session.current_agent.chat_ctx.items
    await _close(session, runner)

    # the caller's turns keep their own ids and roles rather than being quoted in a note
    caller_turns = [item.id for item in items if item.id in ("m1", "m2")]
    assert caller_turns == ["m1", "m2"]
    assert all(item.role == "user" for item in items if item.id in ("m1", "m2"))
    # what this session recorded itself is untouched, and nothing is merged twice
    assert len(items) == len({item.id for item in items})


async def test_what_the_caller_holds_arrives_as_conversation_not_plumbing() -> None:
    """A caller's calls, handoffs and instructions are not what was said."""
    llm = _AnsweringLLM(fake_responses=[_says("first", "one")], fallbacks=[])
    session, runner = await _serve(Agent(instructions="fare desk"), llm=llm)

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="change my Monday flight", id="m1")
    chat_ctx.insert(FunctionCall(id="fc1", call_id="c1", name="lk_agents_delegate", arguments="{}"))
    chat_ctx.add_message(role="system", content="you are a phone agent", id="s1")
    await _collect(
        runner.submit(TaskInput(instruction="first", chat_ctx=chat_ctx), request_id="r1")
    )

    ids = {item.id for item in session.current_agent.chat_ctx.items}
    await _close(session, runner)

    assert "m1" in ids
    assert "fc1" not in ids
    assert "s1" not in ids


async def test_cancelling_ends_the_request_with_what_it_had() -> None:
    """A cancelled request still declares a terminal state: a stream that ends without one
    reads as a failure, and the caller cannot tell the two apart."""

    @function_tool(flags={ToolFlag.CANCELLABLE})
    async def hold_seat(ctx: RunContext) -> str:
        """Hold a seat, slowly."""
        await ctx.update("holding the seat")
        await asyncio.sleep(60)
        return "held"

    llm = _AnsweringLLM(
        fake_responses=[_says("hold it", "", calls=[_tool_call("hold_seat", "hs1")])],
        fallbacks=["Holding."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[hold_seat]), llm=llm)

    run = runner.submit(TaskInput(instruction="hold it"), request_id="r1")
    updates: list[TaskUpdate] = []

    async def _read() -> None:
        async for update in run:
            updates.append(update)
            if update.text == "holding the seat":
                await run.cancel()

    await asyncio.wait_for(_read(), timeout=30.0)
    await asyncio.wait_for(run.aclose(), timeout=10.0)
    await _close(session, runner)

    assert updates[-1].state == "canceled"
    assert "hold_seat" in updates[-1].text


async def test_a_stop_reaches_a_call_the_executor_does_not_have_yet() -> None:
    """A call is in the turn's history before the executor registers it. A stop asked for in
    that window finds nothing to stop, so it has to land when the executor takes the call."""
    cancelled: list[str] = []

    @function_tool(flags={ToolFlag.CANCELLABLE})
    async def hold_seat(ctx: RunContext) -> str:
        """Hold a seat, slowly."""
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append("hold_seat")
            raise
        return "held"

    llm = _AnsweringLLM(
        fake_responses=[_says("hold it", "", calls=[_tool_call("hold_seat", "hs1")])],
        fallbacks=["Holding."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[hold_seat]), llm=llm)

    run = runner.submit(TaskInput(instruction="hold it"), request_id="r1")
    seen = run.on_item

    def stop_in_the_window(item: Any, handle: Any) -> None:
        seen(item, handle)
        if item.type == "function_call":
            # the speech is deliberately left alone: interrupting it stops the call too, and
            # what is under test is the stop only the call itself can take
            assert item.call_id not in _RunningTasks.get(session, {}), "not the window"
            run._stopping = True
            runner._spawn(run._cancel_tool_call(item.call_id))

    run.on_item = stop_in_the_window  # type: ignore[method-assign]

    async def _until_the_call_lands() -> None:
        async for update in run:
            if update.item is not None and update.item.type == "function_call":
                return

    await asyncio.wait_for(_until_the_call_lands(), timeout=10.0)
    await asyncio.sleep(2)
    # closing the run interrupts the speech, which stops the call too: the claim is that the
    # stop asked for in the window landed on its own
    stopped_in_the_window = list(cancelled)
    await asyncio.wait_for(run.aclose(), timeout=10.0)
    await _close(session, runner)

    assert stopped_in_the_window == ["hold_seat"]


async def test_closing_the_run_cancels_the_work_it_can_stop() -> None:
    cancelled: list[str] = []

    @function_tool(flags={ToolFlag.CANCELLABLE})
    async def hold_seat(ctx: RunContext) -> str:
        """Hold a seat, slowly."""
        await ctx.update("holding the seat")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append("hold_seat")
            raise
        return "held"

    llm = _AnsweringLLM(
        fake_responses=[_says("hold it", "", calls=[_tool_call("hold_seat", "hs1")])],
        fallbacks=["Holding."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[hold_seat]), llm=llm)

    run = runner.submit(TaskInput(instruction="hold it"), request_id="r1")

    async def _until_holding() -> None:
        # the turn records the call before the executor dispatches it, so waiting for the
        # call to appear would abandon before anything was running
        async for update in run:
            if update.text == "holding the seat":
                return

    await asyncio.wait_for(_until_holding(), timeout=10.0)
    await asyncio.wait_for(run.aclose(), timeout=10.0)
    await asyncio.sleep(1)
    # closing the session would cancel it too, so the claim is that closing the run did
    assert cancelled == ["hold_seat"]

    await _close(session, runner)


async def test_what_a_request_produced_is_stamped_with_it() -> None:
    @function_tool
    async def check_fares(ctx: RunContext) -> str:
        """Check the fares."""
        return "fare is 240 USD"

    llm = _AnsweringLLM(
        fake_responses=[_says("what is the fare", "", calls=[_tool_call("check_fares", "cf1")])],
        fallbacks=["It is 240 USD."],
    )
    session, runner = await _serve(Agent(instructions="fare desk", tools=[check_fares]), llm=llm)

    await _collect(runner.submit(TaskInput(instruction="what is the fare"), request_id="r1"))
    stamped = [
        item
        for item in session.current_agent.chat_ctx.items
        if getattr(item, "extra", {}).get(REQUEST_ID_KEY) == "r1"
    ]
    await _close(session, runner)

    assert stamped, "the request's own items carry its id"
    assert any(item.type == "function_call" for item in stamped)
