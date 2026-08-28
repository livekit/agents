"""Delegation: the delegate contract, tool routing, and what running a delegate here buys.

A delegate answers a request, so most of what needs covering is that the framework hands it the
right request, delivers what it returns, and shares in-flight tool calls between delegations.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    DelegationUpdate,
    RunContext,
    function_tool,
)
from livekit.agents.llm import FunctionToolCall, ToolError, ToolFlag
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.delegation import (
    DELEGATE_TOOL_NAME,
    AgentDelegate,
    DelegationRequest,
    DelegationStream,
)

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@function_tool
async def send_dtmf(ctx: RunContext, digits: str) -> None:
    """Send DTMF digits."""


def _delegate_tools(activity: AgentActivity) -> list[Any]:
    return [
        t for t in activity.tools if getattr(t, "info", None) and t.info.name == DELEGATE_TOOL_NAME
    ]


def _names(tools: list[Any]) -> set[str]:
    return {t.info.name for t in tools if hasattr(t, "info")}


class _NoopDelegate(DelegationStream):
    async def _run(self) -> str:
        return "done"


# -- tool routing ------------------------------------------------------------------------------


def test_delegate_tool_absent_without_a_delegate() -> None:
    activity = AgentActivity(Agent(instructions="test", tools=[send_dtmf]), AgentSession())
    assert _names(activity.tools) == {"send_dtmf"}


def test_delegate_tool_joins_the_model_list() -> None:
    session = AgentSession(delegate=_NoopDelegate)
    activity = AgentActivity(Agent(instructions="test", tools=[send_dtmf]), session)
    # the conversation agent keeps its own tools; the delegate tool is added, not substituted
    assert _names(activity.tools) == {"send_dtmf", DELEGATE_TOOL_NAME}


def test_delegate_tool_identity_is_stable() -> None:
    """The model-visible schema must not change between turns, or the prompt cache is lost."""
    activity = AgentActivity(Agent(instructions="test"), AgentSession(delegate=_NoopDelegate))
    first = _delegate_tools(activity)
    second = _delegate_tools(activity)
    assert first[0] is second[0]


def test_agent_delegate_overrides_the_session() -> None:
    class Other(DelegationStream):
        async def _run(self) -> str:
            return "other"

    session = AgentSession(delegate=_NoopDelegate)
    assert AgentActivity(Agent(instructions="t"), session).delegate is _NoopDelegate
    assert AgentActivity(Agent(instructions="t", delegate=Other), session).delegate is Other


def test_agent_delegate_can_disable_the_session_one() -> None:
    session = AgentSession(delegate=_NoopDelegate)
    activity = AgentActivity(Agent(instructions="t", delegate=None), session)
    assert activity.delegate is None
    assert DELEGATE_TOOL_NAME not in _names(activity.tools)


# -- the contract ------------------------------------------------------------------------------


def _delegating_session(**kwargs: Any) -> tuple[AgentSession, Agent]:
    """A text-only session whose model delegates the first thing it is asked."""
    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="book a flight",
                    content="one sec",
                    ttft=0.0,
                    duration=0.0,
                    tool_calls=[
                        FunctionToolCall(
                            type="function",
                            name=DELEGATE_TOOL_NAME,
                            arguments='{"task": "book a flight to NYC on Monday"}',
                            call_id="call_1",
                        )
                    ],
                )
            ]
        ),
        **kwargs,
    )
    return session, Agent(instructions="you are the airline's voice")


async def _close(session: AgentSession) -> None:
    """Close under a deadline, so a stuck drain fails the test instead of hanging the run."""
    await asyncio.wait_for(session.aclose(), timeout=10.0)


async def _run_once(session: AgentSession, agent: Agent) -> None:
    await session.start(agent=agent)
    await asyncio.wait_for(session.run(user_input="book a flight"), timeout=10.0)
    await _close(session)


def _outputs(session: AgentSession) -> str:
    return " ".join(
        item.output for item in session.history.items if item.type == "function_call_output"
    )


def _answer(session: AgentSession) -> str:
    """What the `delegate` tool returned, which it records under `<call_id>_final`."""
    return " ".join(
        item.output
        for item in session.history.items
        if item.type == "function_call_output"
        and item.name == DELEGATE_TOOL_NAME
        and item.call_id.endswith("_final")
    )


def _errors(session: AgentSession) -> str:
    return " ".join(
        item.output
        for item in session.history.items
        if item.type == "function_call_output" and item.is_error
    )


@pytest.mark.asyncio
async def test_delegate_receives_the_request() -> None:
    seen: list[DelegationRequest] = []

    class Handler(DelegationStream):
        async def _run(self) -> str:
            seen.append(self.request)
            return "Booked, reference QX7."

    session, agent = _delegating_session(
        delegate=Handler, delegation_options={"metadata": {"customer_id": "c-42"}}
    )
    await _run_once(session, agent)

    assert len(seen) == 1
    req = seen[0]
    assert req.task == "book a flight to NYC on Monday"
    assert req.delegation_id == "call_1"
    assert req.metadata == {"customer_id": "c-42"}
    # the conversation so far, including what the user asked
    assert any(
        item.type == "message" and item.text_content == "book a flight"
        for item in req.chat_ctx.items
    )


@pytest.mark.asyncio
async def test_the_request_carries_no_tool_traffic_of_the_caller() -> None:
    """The delegate call is still in flight, so its output does not exist yet — passing it on
    leaves a dangling call that every request the expert makes warns about and drops."""
    seen: list[DelegationRequest] = []

    class Handler(DelegationStream):
        async def _run(self) -> str:
            seen.append(self.request)
            return "Booked, reference QX7."

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)

    assert len(seen) == 1
    kinds = {item.type for item in seen[0].chat_ctx.items}
    assert "function_call" not in kinds and "function_call_output" not in kinds
    # the conversation itself still travels
    assert any(item.type == "message" for item in seen[0].chat_ctx.items)


@pytest.mark.asyncio
async def test_a_returned_string_is_the_answer() -> None:
    """The whole contract for a delegate that does not stream, and for a text HTTP endpoint."""

    async def handler(req: DelegationRequest) -> str:
        return "Booked, reference QX7."

    session, agent = _delegating_session(delegate=handler)
    await _run_once(session, agent)
    assert "Booked, reference QX7." in _outputs(session)


@pytest.mark.asyncio
async def test_streamed_progress_arrives_before_the_answer() -> None:
    class Handler(DelegationStream):
        async def _run(self) -> str:
            self.send(DelegationUpdate("cancelling the earlier booking first"))
            return "Rebooked for Tuesday."

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)

    recorded = _outputs(session)
    assert recorded.index("cancelling the earlier booking first") < recorded.index(
        "Rebooked for Tuesday."
    )


@pytest.mark.asyncio
async def test_a_stream_that_declares_no_terminal_state_is_an_error() -> None:
    class Handler(DelegationStream):
        async def _run(self) -> None:
            self.send(DelegationUpdate("still looking"))

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)
    assert "without declaring a result" in _errors(session)


@pytest.mark.asyncio
async def test_delegate_error_surfaces_as_a_tool_error() -> None:
    class Handler(DelegationStream):
        async def _run(self) -> str:
            raise ToolError("the booking system is down")

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)
    assert "booking system is down" in _errors(session)


@pytest.mark.asyncio
async def test_a_declared_failure_surfaces_as_a_tool_error() -> None:
    class Handler(DelegationStream):
        async def _run(self) -> None:
            self.send(DelegationUpdate("checking the booking system"))
            self.send(DelegationUpdate("the booking system is down", state="failed"))

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)
    assert "booking system is down" in _errors(session)


@pytest.mark.asyncio
async def test_a_cancelled_delegation_still_answers() -> None:
    """Cancellation is a result: what happened, including side effects that already landed."""

    class Handler(DelegationStream):
        async def _run(self) -> None:
            self.send(
                DelegationUpdate("cancelled — the refund had already been issued", state="canceled")
            )

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)
    assert "the refund had already been issued" in _outputs(session)
    assert not _errors(session)


@pytest.mark.asyncio
async def test_input_required_reaches_the_conversation_as_an_error() -> None:
    """Not routed to the user in v0 — the conversation model is told what is missing."""

    class Handler(DelegationStream):
        async def _run(self) -> None:
            self.send(DelegationUpdate("which booking reference?", state="input-required"))

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)
    assert "which booking reference?" in _errors(session)


@pytest.mark.asyncio
async def test_a_terminal_state_ends_the_delegation() -> None:
    """The answer ends it: nothing said after is consumed, and closing cancels the work."""
    reached_the_end = False

    class Handler(DelegationStream):
        async def _run(self) -> None:
            nonlocal reached_the_end
            self.send(DelegationUpdate("Booked, reference QX7.", state="completed"))
            await asyncio.sleep(1.0)
            self.send(DelegationUpdate("and one more thing"))
            reached_the_end = True

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)

    assert "Booked, reference QX7." in _outputs(session)
    assert "and one more thing" not in _outputs(session)
    assert not reached_the_end


@pytest.mark.asyncio
async def test_silent_dispatch_asks_for_no_reply() -> None:
    """With `delegation_announce` off the dispatch note releases the turn without speaking.

    Pinned on the chat item rather than on what was voiced: the note is recorded either way,
    and `reply_required` is the flag that decides whether a turn is generated from it.
    """
    answered = asyncio.Event()

    class Handler(DelegationStream):
        async def _run(self) -> str:
            await answered.wait()
            return "Booked, reference QX7."

    session, agent = _delegating_session(delegate=Handler, delegation_options={"announce": False})
    await session.start(agent=agent)

    session.generate_reply(user_input="book a flight")
    await asyncio.sleep(0.3)

    dispatch = [
        item
        for item in session.history.items
        if item.type == "function_call_output" and item.name == DELEGATE_TOOL_NAME
    ]
    assert len(dispatch) == 1
    assert dispatch[0].reply_required is False
    # nothing else acknowledges, so the tool asks the model for a line in the same completion
    tool = next(t for t in _delegate_tools(session._activity))
    assert "In the same turn as the call" in tool.info.description

    answered.set()
    await asyncio.sleep(0.3)
    await _close(session)


@pytest.mark.asyncio
async def test_announce_asks_the_dispatch_note_for_a_reply() -> None:
    """`delegation_announce` moves the acknowledgement onto the dispatch note, for a model
    that will not reliably write its own line alongside the call."""
    answered = asyncio.Event()

    class Handler(DelegationStream):
        async def _run(self) -> str:
            await answered.wait()
            return "Booked, reference QX7."

    session, agent = _delegating_session(delegate=Handler, delegation_options={"announce": True})
    await session.start(agent=agent)

    session.generate_reply(user_input="book a flight")
    await asyncio.sleep(0.3)

    dispatch = [
        item
        for item in session.history.items
        if item.type == "function_call_output" and item.name == DELEGATE_TOOL_NAME
    ]
    assert len(dispatch) == 1
    assert dispatch[0].reply_required is True
    assert "Acknowledge" in dispatch[0].output
    # exactly one of the two acknowledges, so the tool no longer asks for a line of its own
    tool = next(iter(_delegate_tools(session._activity)))
    assert "In the same turn as the call" not in tool.info.description

    answered.set()
    await asyncio.sleep(0.3)
    await _close(session)


# -- nested sessions ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_session_started_inside_a_tool_is_nested() -> None:
    parents: list[Any] = []

    class Handler(DelegationStream):
        async def _run(self) -> str:
            nested = AgentSession()
            await nested.start(agent=Agent(instructions="expert"))
            parents.append(nested.parent)
            await _close(nested)
            return "done"

    session, agent = _delegating_session(delegate=Handler)
    await _run_once(session, agent)

    assert parents == [session]


@pytest.mark.asyncio
async def test_nesting_covers_every_task_a_session_spawns() -> None:
    """Not just tools: an activity marks itself on every task it spawns, so a node override
    and on_enter nest the same way a tool call does."""
    seen: dict[str, Any] = {}

    async def probe(where: str) -> None:
        nested = AgentSession()
        await nested.start(agent=Agent(instructions="probe"))
        seen[where] = nested.parent
        await _close(nested)

    @function_tool
    async def a_tool(ctx: RunContext) -> str:
        """A tool."""
        await probe("tool")
        return "ok"

    class Probing(Agent):
        def __init__(self) -> None:
            super().__init__(
                instructions="t",
                tools=[a_tool],
                llm=FakeLLM(
                    fake_responses=[
                        FakeLLMResponse(
                            input="go",
                            content="",
                            ttft=0.0,
                            duration=0.0,
                            tool_calls=[
                                FunctionToolCall(
                                    type="function", name="a_tool", arguments="{}", call_id="c1"
                                )
                            ],
                        ),
                        FakeLLMResponse(input="ok", content="done", ttft=0.0, duration=0.0),
                    ]
                ),
            )

        async def on_enter(self) -> None:
            await probe("on_enter")

        async def llm_node(self, chat_ctx, tools, model_settings):
            if "llm_node" not in seen:
                await probe("llm_node")
            return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    session = AgentSession()
    await session.start(agent=Probing())
    await asyncio.wait_for(session.run(user_input="go"), timeout=10.0)
    await asyncio.sleep(0.3)
    await _close(session)

    assert seen == {"on_enter": session, "llm_node": session, "tool": session}


@pytest.mark.asyncio
async def test_a_session_started_outside_a_tool_is_top_level() -> None:
    """Also the remote case: a delegate served over HTTP has no parent to share work with."""
    session = AgentSession()
    await session.start(agent=Agent(instructions="t"))
    assert session.parent is None
    await _close(session)


def test_parent_is_unknown_before_start() -> None:
    with pytest.raises(RuntimeError, match="not started"):
        _ = AgentSession().parent


# -- what the expert's run says ----------------------------------------------------------------


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


@pytest.mark.asyncio
async def test_a_delegated_tool_may_report_progress_and_still_answer() -> None:
    """`ctx.update()` releases the tool, so the expert's turn ends before the work does.

    The run has to outlive the turn, or the delegation answers from the dispatch note and the
    real result is delivered into a run nobody is reading.
    """
    finished: list[str] = []

    @function_tool
    async def check_fares(ctx: RunContext) -> str:
        """Slow work that reports progress."""
        await ctx.update("checking the fare rules")
        await asyncio.sleep(2)
        finished.append("done")
        return "fare is 240 USD"

    expert = _AnsweringLLM(
        fake_responses=[
            FakeLLMResponse(
                input="what is the fare",
                content="",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function", name="check_fares", arguments="{}", call_id="cf1"
                    )
                ],
            )
        ],
        # the report draws a reply of its own, then the return value draws the answer
        fallbacks=["still checking those rules", "It is 240 USD."],
    )
    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="how much",
                    content="one sec",
                    ttft=0.0,
                    duration=0.0,
                    tool_calls=[
                        FunctionToolCall(
                            type="function",
                            name=DELEGATE_TOOL_NAME,
                            arguments='{"task": "what is the fare"}',
                            call_id="d1",
                        )
                    ],
                )
            ]
        ),
        delegate=AgentDelegate(
            lambda: Agent(instructions="expert", llm=expert, tools=[check_fares])
        ),
    )
    await session.start(agent=Agent(instructions="voice"))
    session.generate_reply(user_input="how much")
    await asyncio.sleep(5)
    await _close(session)

    assert finished == ["done"]

    # the delegation's own answer, not one of the progress reports that preceded it
    assert _answer(session) == "It is 240 USD."
    # the report reached the conversation phrased by the expert, not as the raw report, which
    # carries the framing that model needs to read it
    assert "still checking those rules" in _outputs(session)
    assert "checking the fare rules" not in _outputs(session)


class _DetailTask(AgentTask):
    """A dialog that needs a further user turn before it completes."""

    def __init__(self) -> None:
        super().__init__(instructions="collect a detail")

    @function_tool
    async def finish(self, ctx: RunContext) -> str:
        """Complete the dialog."""
        self.complete("collected")
        return "dialog done"


@pytest.mark.asyncio
async def test_an_async_tool_can_hand_off_to_an_agent_task() -> None:
    """`ctx.update()` releases the tool, so the run watches its task to cover the real work.

    An AgentTask awaited inside that tool cannot finish without another user turn, and the run
    has to return for that turn to happen — so the task's handoff must release the watch, and
    take it back when the dialog completes.
    """

    @function_tool
    async def gather(ctx: RunContext) -> str:
        """Asks for a detail, releasing the turn first."""
        await ctx.update("one moment")
        detail = await _DetailTask()
        return f"gathered {detail}"

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="sure",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(type="function", name="gather", arguments="{}", call_id="g1")
                ],
            ),
            FakeLLMResponse(
                input="my detail",
                content="",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(type="function", name="finish", arguments="{}", call_id="f1")
                ],
            ),
        ]
    )
    session = AgentSession(llm=llm)
    await session.start(agent=Agent(instructions="root", tools=[gather]))

    # must return even though the tool is parked on a dialog waiting for the user
    await asyncio.wait_for(session.run(user_input="go"), timeout=10.0)
    assert isinstance(session.current_agent, _DetailTask)

    await asyncio.wait_for(session.run(user_input="my detail"), timeout=10.0)
    await asyncio.sleep(1)
    await _close(session)

    # the dialog handed back, and the tool that was waiting on it finished
    assert "gathered collected" in _outputs(session)


@pytest.mark.asyncio
async def test_only_the_last_word_is_the_answer() -> None:
    """A message with nothing outstanding still is not the answer if the expert keeps going.

    Two tool steps, each preceded by a message. Only the message the run ends on answers; the
    other announced the work that followed it.
    """

    @function_tool
    async def look_up(ctx: RunContext, what: str) -> str:
        """Look something up."""
        return f"{what} ok"

    expert = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="check both",
                content="checking the fare",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function", name="look_up", arguments='{"what": "fare"}', call_id="a"
                    )
                ],
            ),
            # nothing outstanding at this message, yet the expert calls again afterwards
            FakeLLMResponse(
                input="fare ok",
                content="now the baggage",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name="look_up",
                        arguments='{"what": "baggage"}',
                        call_id="b",
                    )
                ],
            ),
            FakeLLMResponse(input="baggage ok", content="All set.", ttft=0.0, duration=0.0),
        ]
    )
    session, agent = _delegating_session(
        delegate=AgentDelegate(lambda: Agent(instructions="expert", llm=expert, tools=[look_up]))
    )
    session._llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a flight",
                content="one sec",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name=DELEGATE_TOOL_NAME,
                        arguments='{"task": "check both"}',
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    await _run_once(session, agent)

    assert _answer(session) == "All set."
    # the message before the second step is not lost, it becomes progress
    assert "now the baggage" in _outputs(session)


@pytest.mark.asyncio
async def test_a_delegation_may_finish_with_nothing_to_add() -> None:
    """The expert works and concludes nothing, so the delegation answers with nothing."""

    @function_tool
    async def look_up(ctx: RunContext) -> str:
        """Look something up."""
        await asyncio.sleep(1)
        return "fare ok"

    expert = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="check the fare",
                content="",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(type="function", name="look_up", arguments="{}", call_id="a")
                ],
            ),
            # the expert concludes nothing after the tool returns
            FakeLLMResponse(input="fare ok", content="", ttft=0.0, duration=0.0),
        ]
    )
    session, agent = _delegating_session(
        delegate=AgentDelegate(lambda: Agent(instructions="expert", llm=expert, tools=[look_up]))
    )
    session._llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a flight",
                content="one sec",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name=DELEGATE_TOOL_NAME,
                        arguments='{"task": "check the fare"}',
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    await _run_once(session, agent)

    assert _answer(session) == ""


@pytest.mark.asyncio
async def test_a_line_a_tool_said_is_progress_not_an_answer() -> None:
    """A tool that says something and returns None draws no reply, so the expert concludes
    nothing. The line reached the conversation as progress; the delegation adds nothing."""

    @function_tool
    async def report_balance(ctx: RunContext) -> None:
        """Read the balance out directly."""
        ctx.session.say("your balance is 50 dollars")

    expert = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="what is the balance",
                content="Looking.",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function", name="report_balance", arguments="{}", call_id="a"
                    )
                ],
            )
        ]
    )
    session, agent = _delegating_session(
        delegate=AgentDelegate(
            lambda: Agent(instructions="expert", llm=expert, tools=[report_balance])
        )
    )
    session._llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a flight",
                content="one sec",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name=DELEGATE_TOOL_NAME,
                        arguments='{"task": "what is the balance"}',
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    await _run_once(session, agent)

    assert _answer(session) == ""
    # both lines still reached the conversation, as progress
    assert "your balance is 50 dollars" in _outputs(session)
    assert "Looking." in _outputs(session)


@pytest.mark.asyncio
async def test_an_async_tools_late_reply_is_the_answer() -> None:
    """An async tool concludes its step against a dispatch note, and its real result draws a
    reply afterwards, so that late reply answers and the conclusion it followed is progress."""

    @function_tool
    async def look_up(ctx: RunContext) -> str:
        """Look something up."""
        return "fare ok"

    @function_tool
    async def log_it(ctx: RunContext) -> str:
        """File the request, which takes a while and reports back when it lands."""
        await ctx.update("noted", silent=True)
        await asyncio.sleep(3)
        return "filed under 123"

    expert = _AnsweringLLM(
        fake_responses=[
            FakeLLMResponse(
                input="check the fare",
                content="checking",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(type="function", name="look_up", arguments="{}", call_id="a"),
                    FunctionToolCall(type="function", name="log_it", arguments="{}", call_id="b"),
                ],
            )
        ],
        fallbacks=["All set.", "Filed under 123."],
    )
    session, agent = _delegating_session(
        delegate=AgentDelegate(
            lambda: Agent(instructions="expert", llm=expert, tools=[look_up, log_it])
        )
    )
    session._llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a flight",
                content="one sec",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name=DELEGATE_TOOL_NAME,
                        arguments='{"task": "check the fare"}',
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    await session.start(agent=agent)
    await asyncio.wait_for(session.run(user_input="book a flight"), timeout=10.0)
    await asyncio.sleep(5)
    await _close(session)

    assert _answer(session) == "Filed under 123."
    # the displaced conclusion is not lost, it goes out as progress
    assert "All set." in _outputs(session)


@pytest.mark.asyncio
async def test_what_the_expert_says_on_enter_reaches_the_conversation() -> None:
    """`on_enter` speaks before the request is put, so its turn is captured by `start()`."""

    class Greeter(Agent):
        async def on_enter(self) -> None:
            self.session.say("pulling up your file")

    expert = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a flight to NYC on Monday",
                content="Booked, reference QX7.",
                ttft=0.0,
                duration=0.0,
            )
        ]
    )
    session, agent = _delegating_session(
        delegate=AgentDelegate(lambda: Greeter(instructions="expert", llm=expert))
    )
    await session.start(agent=agent)
    await asyncio.wait_for(session.run(user_input="book a flight"), timeout=10.0)
    await asyncio.sleep(5)
    await _close(session)

    assert "pulling up your file" in _outputs(session)
    assert _answer(session) == "Booked, reference QX7."


@pytest.mark.asyncio
async def test_a_tool_reports_in_its_own_words_by_saying_them() -> None:
    """`session.say()` is how a tool puts its own words out, and they reach the conversation
    while it is still working rather than waiting for the expert to conclude."""

    @function_tool
    async def look_up(ctx: RunContext) -> str:
        """Look something up, saying so while it runs."""
        ctx.session.say("one moment, pulling the fare")
        await asyncio.sleep(5)
        return "fare ok"

    expert = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="check the fare",
                content="",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(type="function", name="look_up", arguments="{}", call_id="a")
                ],
            ),
            FakeLLMResponse(input="fare ok", content="It is 240 USD.", ttft=0.0, duration=0.0),
        ]
    )
    session, agent = _delegating_session(
        delegate=AgentDelegate(lambda: Agent(instructions="expert", llm=expert, tools=[look_up]))
    )
    session._llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a flight",
                content="one sec",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name=DELEGATE_TOOL_NAME,
                        arguments='{"task": "check the fare"}',
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    await session.start(agent=agent)
    session.generate_reply(user_input="book a flight")

    # the tool is still working, and the line has already reached the conversation
    await asyncio.sleep(1)
    assert "one moment, pulling the fare" in _outputs(session)

    await asyncio.sleep(10)
    await _close(session)

    assert _answer(session) == "It is 240 USD."


@pytest.mark.asyncio
async def test_the_last_word_answers_while_background_work_runs() -> None:
    """An answer stands while another of the expert's tools is still working."""

    @function_tool
    async def look_up(ctx: RunContext) -> str:
        """Look something up."""
        return "fare ok"

    @function_tool
    async def log_it(ctx: RunContext) -> None:
        """File the request, which takes a while and reports nothing back."""
        await ctx.update("noted", silent=True)
        await asyncio.sleep(3)

    expert = _AnsweringLLM(
        fake_responses=[
            FakeLLMResponse(
                input="check the fare",
                content="checking",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(type="function", name="look_up", arguments="{}", call_id="a"),
                    FunctionToolCall(type="function", name="log_it", arguments="{}", call_id="b"),
                ],
            )
        ],
        fallbacks=["All set."],
    )
    session, agent = _delegating_session(
        delegate=AgentDelegate(
            lambda: Agent(instructions="expert", llm=expert, tools=[look_up, log_it])
        )
    )
    session._llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a flight",
                content="one sec",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name=DELEGATE_TOOL_NAME,
                        arguments='{"task": "check the fare"}',
                        call_id="call_1",
                    )
                ],
            )
        ]
    )
    await session.start(agent=agent)
    await asyncio.wait_for(session.run(user_input="book a flight"), timeout=10.0)
    await asyncio.sleep(5)
    await _close(session)

    assert _answer(session) == "All set."


# -- an Agent as the delegate -------------------------------------------------------------


_BOOKINGS: dict[str, list[str]] = {"started": [], "cancelled": [], "finished": []}


@function_tool(flags=ToolFlag.CANCELLABLE, on_duplicate="replace")
async def book_flight(ctx: RunContext, date: str) -> str:
    """Book a flight."""
    _BOOKINGS["started"].append(date)
    try:
        await asyncio.sleep(5)
    except asyncio.CancelledError:
        _BOOKINGS["cancelled"].append(date)
        raise
    _BOOKINGS["finished"].append(date)
    return f"booked {date}"


@pytest.fixture(autouse=True)
def _reset_bookings():
    for v in _BOOKINGS.values():
        v.clear()
    yield


def _expert_llm() -> FakeLLM:
    """Books whatever date the task names, then reports back."""
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book Monday",
                content="checking Monday",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name="book_flight",
                        arguments='{"date": "Monday"}',
                        call_id="bf_1",
                    )
                ],
            ),
            FakeLLMResponse(
                input="move it to Tuesday",
                content="",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    FunctionToolCall(
                        type="function",
                        name="book_flight",
                        arguments='{"date": "Tuesday"}',
                        call_id="bf_2",
                    )
                ],
            ),
            FakeLLMResponse(input="booked Tuesday", content="Rebooked.", ttft=0.0, duration=0.0),
        ]
    )


def _agent_delegate(expert: FakeLLM) -> AgentDelegate:
    return AgentDelegate(
        lambda: Agent(instructions="you handle flights", llm=expert, tools=[book_flight])
    )


def _two_turn_session(delegate: Any) -> tuple[AgentSession, Agent]:
    """A conversation model that delegates each of two user turns."""
    return (
        AgentSession(
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input="book me a flight",
                        content="one sec",
                        ttft=0.0,
                        duration=0.0,
                        tool_calls=[
                            FunctionToolCall(
                                type="function",
                                name=DELEGATE_TOOL_NAME,
                                arguments='{"task": "book Monday"}',
                                call_id="d_1",
                            )
                        ],
                    ),
                    FakeLLMResponse(
                        input="actually make it Tuesday",
                        content="sure",
                        ttft=0.0,
                        duration=0.0,
                        tool_calls=[
                            FunctionToolCall(
                                type="function",
                                name=DELEGATE_TOOL_NAME,
                                arguments='{"task": "move it to Tuesday"}',
                                call_id="d_2",
                            )
                        ],
                    ),
                ]
            ),
            delegate=delegate,
        ),
        Agent(instructions="you are the airline's voice"),
    )


@pytest.mark.asyncio
async def test_agent_delegate_runs_the_experts_tools() -> None:
    session, agent = _two_turn_session(_agent_delegate(_expert_llm()))
    await session.start(agent=agent)

    session.generate_reply(user_input="book me a flight")
    await asyncio.sleep(0.3)
    assert _BOOKINGS["started"] == ["Monday"]

    await asyncio.sleep(6)
    # the expert's tool traffic is work that happened, not something it said: only its words
    # cross back, so the raw tool output never reaches the conversation
    assert "booked Monday" not in _outputs(session)
    assert "checking Monday" in _outputs(session)

    await _close(session)


@pytest.mark.asyncio
async def test_agent_delegate_reports_text_alongside_a_tool_call_as_progress() -> None:
    """Held only until the tool call event, so it lands well before the booking finishes."""
    session, agent = _two_turn_session(_agent_delegate(_expert_llm()))
    await session.start(agent=agent)

    session.generate_reply(user_input="book me a flight")
    await asyncio.sleep(0.3)

    assert "checking Monday" in _outputs(session)
    assert _BOOKINGS["finished"] == []

    await _close(session)


@pytest.mark.asyncio
async def test_a_later_delegation_replaces_work_the_first_started() -> None:
    """The flight scenario: two nested sessions share one in-flight view, so `replace` reaches
    across them. This is the whole difference between running a delegate here and over HTTP."""
    session, agent = _two_turn_session(_agent_delegate(_expert_llm()))
    await session.start(agent=agent)

    session.generate_reply(user_input="book me a flight")
    await asyncio.sleep(1)
    assert _BOOKINGS["started"] == ["Monday"]
    assert _BOOKINGS["cancelled"] == []

    # the first booking is still in flight when the second delegation arrives
    session.generate_reply(user_input="actually make it Tuesday")
    await asyncio.sleep(2)

    assert _BOOKINGS["started"] == ["Monday", "Tuesday"]
    assert _BOOKINGS["cancelled"] == ["Monday"]

    await _close(session)


@pytest.mark.asyncio
async def test_the_experts_tool_traffic_stays_inside() -> None:
    session, agent = _two_turn_session(_agent_delegate(_expert_llm()))
    await session.start(agent=agent)

    session.generate_reply(user_input="book me a flight")
    await asyncio.sleep(0.3)
    await _close(session)

    # the conversation records the delegation, never the tool the expert reached for
    assert "book_flight" not in _outputs(session)
    names = {item.name for item in session.history.items if item.type == "function_call"}
    assert names == {DELEGATE_TOOL_NAME}
