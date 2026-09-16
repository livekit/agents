"""The runner: one session answering requests, and which answer belongs to which request."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.a2a import TaskInput, TaskUpdate
from livekit.agents.a2a._runner import REQUEST_ID_KEY, RequestRun, SessionRunner
from livekit.agents.llm import ChatContext, FunctionToolCall, ToolFlag

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
    runner.attach()
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
    assert [(u.state, u.text) for u in updates if u.state != "working"] == [
        ("completed", "It is 240 USD.")
    ]


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
        request = ctx.session.request
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
        request = ctx.session.request
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


async def test_an_ordinary_session_has_no_request_to_direct() -> None:
    """The same tool body runs in a session nobody is waiting on."""
    branch: list[str] = []

    @function_tool
    async def say_goodbye(ctx: RunContext) -> str:
        """Called when the caller is done."""
        if (request := ctx.session.request) is not None:
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


async def test_the_conversation_is_shown_once_across_requests() -> None:
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

    notes = [
        item.text_content or ""
        for item in session.current_agent.chat_ctx.items
        if item.type == "message" and "since the last request" in (item.text_content or "")
    ]
    await _close(session, runner)

    assert len(notes) == 2
    assert "change my Monday flight" in notes[0]
    # the item already shown is not shown again
    assert "change my Monday flight" not in notes[1]
    assert "and the Tuesday one" in notes[1]


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
