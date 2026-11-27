from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Callable, Mapping
from typing import Any

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    ModelSettings,
    RunContext,
    function_tool,
    llm,
)
from livekit.agents.beta import Block, DecisionEvaluated, DecisionGuardrail, Steer
from livekit.agents.llm import FunctionToolCall, decisions
from livekit.agents.llm.decisions import DecisionResponse, ProbabilityResult
from livekit.agents.types import APIConnectOptions, FlushSentinel
from livekit.agents.voice.tool_executor import UPDATE_TEMPLATE

from .fake_session import FakeActions, create_session, run_session
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


class KeywordModel(decisions.DecisionModel):
    """Scores every decision 1.0 when the input contains ``keyword``, else 0.0."""

    def __init__(self, keyword: str) -> None:
        super().__init__(
            capabilities=decisions.DecisionCapabilities(decision_kinds=["probability"])
        )
        self._keyword = keyword
        self.inputs: list[str] = []

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, decisions.Decision],
        conn_options: APIConnectOptions,
        extra_headers: Mapping[str, str],
    ) -> DecisionResponse:
        self.inputs.append(input)
        value = 1.0 if self._keyword in input.lower() else 0.0
        return DecisionResponse(
            results={name: ProbabilityResult(value=value) for name in decisions}
        )


FLAG = {"flag": decisions.Probability("Flag this.")}


def _messages(events: list[ConversationItemAddedEvent]) -> list[tuple[str, str]]:
    return [
        (ev.item.role, ev.item.text_content or "") for ev in events if ev.item.type == "message"
    ]


def _guarded_session(
    actions: FakeActions,
    setup: Callable[[DecisionGuardrail], None],
    keyword: str,
    session_kwargs: dict[str, Any] | None = None,
) -> AgentSession:
    session = create_session(actions, extra_kwargs=session_kwargs)
    tts = session.tts
    assert isinstance(tts, FakeTTS)
    tts._fake_audio_duration = 0.5  # any text the test didn't script still plays

    guardrail = DecisionGuardrail(session, model=KeywordModel(keyword))
    setup(guardrail)
    return session


async def _run(
    actions: FakeActions, setup: Callable[[DecisionGuardrail], None], keyword: str
) -> list[tuple[str, str]]:
    session = _guarded_session(actions, setup, keyword)
    events: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", events.append)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="You are a helpful assistant.")),
        timeout=SESSION_TIMEOUT,
    )
    return _messages(events)


class BillingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You handle billing.")


class OrderAgent(Agent):
    """Records the tools that ran and what each LLM step was asked."""

    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self.tool_runs: list[tuple[str, Any]] = []
        self.tool_events: list[FunctionToolsExecutedEvent] = []
        # the last item the LLM saw, and the tool choice it was given
        self.llm_steps: list[tuple[str, Any]] = []

    @function_tool
    async def lookup_order(self, order_id: int = 0) -> str:
        """Look up an order."""
        self.tool_runs.append(("lookup_order", order_id))
        return "shipped"

    @function_tool
    async def start_refund(self, amount: int) -> str:
        """Refund an order."""
        self.tool_runs.append(("start_refund", amount))
        return "refunded"

    @function_tool
    async def log_complaint(self) -> None:
        """Record a complaint."""
        self.tool_runs.append(("log_complaint", None))

    @function_tool
    async def track_order(self, ctx: RunContext) -> str:
        """Track an order, reporting progress before the result."""
        await ctx.update("Tracking it down.")
        self.tool_runs.append(("track_order", None))
        return "shipped"

    @function_tool
    async def to_billing(self) -> Agent:
        """Hand off to billing."""
        self.tool_runs.append(("to_billing", None))
        return BillingAgent()

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        last = chat_ctx.items[-1]
        self.llm_steps.append(
            (
                last.role if last.type == "message" else last.type,
                model_settings.tool_choice or "auto",
            )
        )
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk


def _named(tool: str) -> llm.ToolChoice:
    return {"type": "function", "function": {"name": tool}}


def _history(session: AgentSession) -> list[tuple[str, str]]:
    summary: list[tuple[str, str]] = []
    for item in session.history.items:
        if item.type == "message":
            summary.append((item.role, item.text_content or ""))
        elif item.type == "function_call":
            summary.append(("call", f"{item.name}({item.arguments})"))
        elif item.type == "function_call_output":
            summary.append(("output", item.output))
        elif item.type == "agent_handoff" and item.old_agent_id is not None:
            summary.append(("handoff", item.new_agent_id))
    return summary


class NoToolChoiceAgent(OrderAgent):
    """An agent whose LLM may not call tools."""

    async def on_enter(self) -> None:
        self._get_activity_or_raise().update_options(tool_choice="none")


class RequiredToolChoiceAgent(OrderAgent):
    """An agent whose LLM has to call a tool."""

    async def on_enter(self) -> None:
        self._get_activity_or_raise().update_options(tool_choice="required")


async def _run_order_agent(
    actions: FakeActions,
    steer: Steer,
    *,
    on_error: Any = "allow",
    agent_type: type[OrderAgent] = OrderAgent,
    session_kwargs: dict[str, Any] | None = None,
) -> tuple[OrderAgent, list[tuple[str, str]]]:
    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_user_turn(decisions=FLAG, on_error=on_error)
        def before_reply(ev: DecisionEvaluated) -> Steer | None:
            return steer if ev.get("flag") else None

    session = _guarded_session(actions, setup, keyword="order", session_kwargs=session_kwargs)
    agent = agent_type()
    session.on("function_tools_executed", agent.tool_events.append)
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    return agent, _history(session)


async def test_user_turn_block_skips_the_reply_and_says_the_replacement() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "you are an idiot")
    actions.add_llm("This reply must never be generated.")

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_user_turn(decisions=FLAG)
        def before_reply(ev: DecisionEvaluated) -> Block | None:
            return Block(say="I'm going to end the call here.") if ev.get("flag") else None

    messages = await _run(actions, setup, keyword="idiot")
    assert messages == [
        ("user", "you are an idiot"),
        ("assistant", "I'm going to end the call here."),
    ]


async def test_user_turn_steer_adds_instructions_for_the_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "my order is still broken")
    actions.add_llm("Unsteered reply.")
    # the fake LLM answers by its last message, which is the steer's instructions
    actions.add_llm("What is still wrong?", input="The issue is not resolved.")

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_user_turn(decisions=FLAG)
        def before_reply(ev: DecisionEvaluated) -> Steer | None:
            return Steer(instructions="The issue is not resolved.") if ev.get("flag") else None

    messages = await _run(actions, setup, keyword="broken")
    assert messages == [
        ("user", "my order is still broken"),
        ("assistant", "What is still wrong?"),
    ]


async def test_agent_sentence_block_keeps_allowed_sentences_and_says_the_replacement() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "can I get my money back")
    actions.add_llm(
        "Thanks for asking about that order. You will get a full refund today. "
        "It should arrive within three days."
    )

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_agent_sentence(decisions=FLAG)
        def during_reply(ev: DecisionEvaluated) -> Block | None:
            return Block(say="Let me check with a colleague.") if ev.get("flag") else None

    messages = await _run(actions, setup, keyword="refund")
    assert messages[0] == ("user", "can I get my money back")
    assert messages[1][0] == "assistant"
    assert messages[1][1].strip() == "Thanks for asking about that order."
    assert messages[2] == ("assistant", "Let me check with a colleague.")
    assert len(messages) == 3


async def test_agent_turn_sees_each_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "hello")
    actions.add_llm("Sorry for the wait, how can I help?")
    seen: list[str] = []

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_agent_turn(decisions=FLAG)
        def after_reply(ev: DecisionEvaluated) -> None:
            seen.append(ev.input_text)

    await _run(actions, setup, keyword="sorry")
    assert seen == ["Sorry for the wait, how can I help?"]


async def test_agent_sentence_replacement_is_part_of_the_run_result() -> None:
    from livekit.agents import AgentSession

    from .fake_llm import FakeLLM, FakeLLMResponse

    # the replacement used to be said from a done callback that raced the run's own,
    # so it was missing from about half the run results: repeat to catch an ordering bug
    for _ in range(4):
        llm = FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="can I get my money back",
                    content="Thanks for asking about that order. You will get a full refund today.",
                    ttft=0.1,
                    duration=0.1,
                )
            ]
        )
        async with AgentSession(llm=llm) as session:
            guardrail = DecisionGuardrail(session, model=KeywordModel("refund"))

            @guardrail.on_agent_sentence(decisions=FLAG)
            def during_reply(ev: DecisionEvaluated) -> Block | None:
                return Block(say="Let me check with a colleague.") if ev.get("flag") else None

            await session.start(Agent(instructions="You are a helpful assistant."))
            result = await asyncio.wait_for(
                session.run(user_input="can I get my money back"), timeout=5.0
            )
            said = [
                (ev.item.text_content or "").strip()
                for ev in result.events
                if ev.type == "message" and ev.item.role == "assistant"
            ]
            assert said == [
                "Thanks for asking about that order.",
                "Let me check with a colleague.",
            ]


async def _run_typed(
    setup: Callable[[DecisionGuardrail], None], responses: list[Any], user_input: str
) -> list[str]:
    from livekit.agents import AgentSession

    from .fake_llm import FakeLLM

    async with AgentSession(llm=FakeLLM(fake_responses=responses)) as session:
        guardrail = DecisionGuardrail(session, model=KeywordModel("idiot"))
        setup(guardrail)
        await session.start(Agent(instructions="You are a helpful assistant."))
        result = await asyncio.wait_for(session.run(user_input=user_input), timeout=5.0)
        return [
            f"{ev.item.role}: {(ev.item.text_content or '').strip()}"
            for ev in result.events
            if ev.type == "message"
        ]


async def test_typed_user_turn_is_checked() -> None:
    from .fake_llm import FakeLLMResponse

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_user_turn(decisions=FLAG)
        def before_reply(ev: DecisionEvaluated) -> Block | None:
            return Block(say="Please keep it civil.") if ev.get("flag") else None

    responses = [
        FakeLLMResponse(input="you idiot", content="Never generated.", ttft=0.1, duration=0.1)
    ]
    # typed input (room chat, session.run) goes through generate_reply, not turn completion
    assert await _run_typed(setup, responses, "you idiot") == ["assistant: Please keep it civil."]


async def test_typed_user_turn_steer_reaches_the_reply() -> None:
    from .fake_llm import FakeLLMResponse

    def setup(guardrail: DecisionGuardrail) -> None:
        @guardrail.on_user_turn(decisions=FLAG)
        def before_reply(ev: DecisionEvaluated) -> Steer | None:
            return Steer(instructions="Stay calm.") if ev.get("flag") else None

    responses = [
        FakeLLMResponse(input="you idiot", content="Unsteered reply.", ttft=0.1, duration=0.1),
        # the fake LLM answers by its last message, which is the steer's instructions
        FakeLLMResponse(input="Stay calm.", content="Let's sort this out.", ttft=0.1, duration=0.1),
    ]
    assert await _run_typed(setup, responses, "you idiot") == ["assistant: Let's sort this out."]


async def test_typed_user_turn_steer_runs_tool_calls_and_forces_the_tool_choice() -> None:
    from .fake_llm import FakeLLM, FakeLLMResponse

    responses = [
        FakeLLMResponse(
            input="shipped",
            content="",
            ttft=0.1,
            duration=0.1,
            tool_calls=[
                FunctionToolCall(name="start_refund", arguments='{"amount": 5}', call_id="call_1")
            ],
        ),
        FakeLLMResponse(
            input="refunded", content="You will get 5 dollars back.", ttft=0.1, duration=0.1
        ),
    ]
    async with AgentSession(llm=FakeLLM(fake_responses=responses)) as session:
        guardrail = DecisionGuardrail(session, model=KeywordModel("order"))

        @guardrail.on_user_turn(decisions=FLAG)
        def before_reply(ev: DecisionEvaluated) -> Steer | None:
            return Steer(tool_calls="lookup_order", tool_choice="start_refund")

        agent = OrderAgent()
        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="refund my order"), timeout=5.0)

        # typed input replies through generate_reply: the steer's tools apply there too
        assert agent.tool_runs == [("lookup_order", 0), ("start_refund", 5)]
        assert agent.llm_steps == [
            ("function_call_output", _named("start_refund")),
            ("function_call_output", "auto"),
        ]
        assert _history(session)[-1] == ("assistant", "You will get 5 dollars back.")


async def test_user_turn_steer_runs_tool_calls_before_the_llm_replies() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "where is my order")
    # what a reply generated before the end of the turn would say: it must be discarded
    actions.add_llm("Unsteered reply.")
    actions.add_llm("It shipped.", input="shipped")

    # the fake LLM answers by the last tool output it sees
    agent, history = await _run_order_agent(
        actions, Steer(tool_calls={"log_complaint": {}, "lookup_order": {"order_id": 7}})
    )
    assert agent.tool_runs == [("log_complaint", None), ("lookup_order", 7)]
    assert history == [
        ("user", "where is my order"),
        ("call", "log_complaint({})"),
        ("call", 'lookup_order({"order_id": 7})'),
        ("output", ""),
        ("output", "shipped"),
        ("assistant", "It shipped."),
    ]
    # the first step is the discarded reply; after it the LLM only answered the tool results
    assert agent.llm_steps == [("user", "auto"), ("function_call_output", "auto")]
    # one event for the step, as for calls the LLM makes
    assert [[call.name for call in ev.function_calls] for ev in agent.tool_events] == [
        ["log_complaint", "lookup_order"]
    ]


async def test_user_turn_steer_tool_calls_run_when_the_llm_may_not_call_tools() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "where is my order")
    actions.add_llm("Unsteered reply.")
    actions.add_llm("It shipped.", input="shipped")

    agent, history = await _run_order_agent(
        actions, Steer(tool_calls="lookup_order"), agent_type=NoToolChoiceAgent
    )
    # the agent's tool choice limits its LLM, not the calls the handler asked for
    assert agent.tool_runs == [("lookup_order", 0)]
    assert history[-1] == ("assistant", "It shipped.")
    assert agent.llm_steps[-1] == ("function_call_output", "none")


async def test_user_turn_steer_tool_choice_applies_to_the_first_llm_step_only() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "where is my order")
    actions.add_llm(
        "", tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="call_1")]
    )
    actions.add_llm("It shipped.", input="shipped")

    agent, history = await _run_order_agent(actions, Steer(tool_choice="lookup_order"))
    assert agent.tool_runs == [("lookup_order", 0)]
    assert history[-1] == ("assistant", "It shipped.")
    # a reply generated before the end of the turn had no tool choice: it is not reused
    assert agent.llm_steps[-2:] == [
        ("user", _named("lookup_order")),
        ("function_call_output", "auto"),
    ]


async def test_user_turn_steer_tool_choice_applies_after_the_tool_calls() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "refund my order")
    actions.add_llm("Unsteered reply.")
    actions.add_llm(
        "",
        tool_calls=[
            FunctionToolCall(name="start_refund", arguments='{"amount": 5}', call_id="call_1")
        ],
        input="shipped",
    )
    actions.add_llm("You will get 5 dollars back.", input="refunded")

    agent, history = await _run_order_agent(
        actions, Steer(tool_calls="lookup_order", tool_choice="start_refund")
    )
    assert agent.tool_runs == [("lookup_order", 0), ("start_refund", 5)]
    assert history[-1] == ("assistant", "You will get 5 dollars back.")
    assert agent.llm_steps[-2:] == [
        ("function_call_output", _named("start_refund")),
        ("function_call_output", "auto"),
    ]


async def test_user_turn_steer_tool_choice_is_asked_for_after_tool_calls_without_a_result() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "refund my order")
    actions.add_llm("Unsteered reply.")
    actions.add_llm("You will get it back.", input="refunded")

    # log_complaint returns nothing, which on its own asks for no reply
    agent, history = await _run_order_agent(
        actions, Steer(tool_calls="log_complaint", tool_choice="start_refund")
    )
    assert agent.tool_runs == [("log_complaint", None)]
    assert agent.llm_steps[-1] == ("function_call_output", _named("start_refund"))


async def test_user_turn_steer_tool_call_without_a_result_gets_no_llm_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "this order is a mess")
    actions.add_llm("Unsteered reply.")

    agent, history = await _run_order_agent(actions, Steer(tool_calls="log_complaint"))
    assert agent.tool_runs == [("log_complaint", None)]
    assert history == [
        ("user", "this order is a mess"),
        ("call", "log_complaint({})"),
        ("output", ""),
    ]


async def test_user_turn_steer_tool_calls_without_a_result_ignore_the_agents_tool_choice() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "this order is a mess")
    actions.add_llm("Unsteered reply.")

    # the agent's "required" is for a step the LLM gets: it isn't a reason to give it one
    agent, history = await _run_order_agent(
        actions, Steer(tool_calls="log_complaint"), agent_type=RequiredToolChoiceAgent
    )
    assert history == [
        ("user", "this order is a mess"),
        ("call", "log_complaint({})"),
        ("output", ""),
    ]
    assert agent.llm_steps == [("user", "required")]


async def test_user_turn_steer_tool_calls_count_as_a_tool_step() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "refund my order")
    actions.add_llm("Unsteered reply.")
    actions.add_llm(
        "",
        tool_calls=[
            FunctionToolCall(name="start_refund", arguments='{"amount": 5}', call_id="call_1")
        ],
        input="shipped",
    )
    actions.add_llm("You will get 5 dollars back.", input="refunded")

    agent, history = await _run_order_agent(
        actions, Steer(tool_calls="lookup_order"), session_kwargs={"max_tool_steps": 1}
    )
    assert agent.tool_runs == [("lookup_order", 0), ("start_refund", 5)]
    # the LLM's own tool call is then over the limit, so its answer may not call another
    assert agent.llm_steps[-2:] == [
        ("function_call_output", "auto"),
        ("function_call_output", "none"),
    ]


async def test_user_turn_steer_tool_choice_outranks_a_tool_progress_report() -> None:
    report = UPDATE_TEMPLATE.format(
        function_name="track_order", call_id="", message="Tracking it down."
    )
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "refund my order")
    actions.add_llm("Unsteered reply.")
    actions.add_llm(
        "",
        tool_calls=[
            FunctionToolCall(name="start_refund", arguments='{"amount": 5}', call_id="call_1")
        ],
        input=report,
    )
    actions.add_llm("You will get 5 dollars back.", input="refunded")
    actions.add_llm("It shipped.", input="shipped")

    agent, _ = await _run_order_agent(
        actions, Steer(tool_calls="track_order", tool_choice="start_refund")
    )
    assert ("start_refund", 5) in agent.tool_runs
    # a report asks for an answer without tools, which would drop the steer's tool
    assert ("function_call_output", _named("start_refund")) in agent.llm_steps


NO_REPLY_WARNING = "no reply was generated after the direct tool calls"


@pytest.mark.parametrize(
    ("steer", "reply", "warned"),
    [
        # the instructions are for a reply that no tool asked for
        pytest.param(Steer("Apologize.", tool_calls="log_complaint"), [], True, id="unused"),
        # a handoff leaves the reply to the next agent, so the LLM never gets the tool
        pytest.param(
            Steer(tool_calls="to_billing", tool_choice="start_refund"), [], True, id="handoff"
        ),
        pytest.param(Steer(tool_calls="log_complaint"), [], False, id="no-instructions"),
        pytest.param(
            Steer("Apologize.", tool_calls="lookup_order"),
            [("assistant", "It shipped.")],
            False,
            id="tool-result",
        ),
    ],
)
async def test_user_turn_steer_the_reply_never_uses_is_warned_about(
    steer: Steer,
    reply: list[tuple[str, str]],
    warned: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "this order is a mess")
    actions.add_llm("Unsteered reply.")
    actions.add_llm("It shipped.", input="shipped")

    with caplog.at_level("WARNING", logger="livekit.agents"):
        _, history = await _run_order_agent(actions, steer)
    assert [item for item in history if item[0] == "assistant"] == reply
    warnings = [r for r in caplog.records if r.getMessage().startswith(NO_REPLY_WARNING)]
    assert len(warnings) == (1 if warned else 0)
    if warned:
        assert warnings[0].__dict__["tools"] == [steer.tool_calls]


async def test_user_turn_steer_tool_call_hands_off_without_an_llm_reply() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "I was charged twice for my order")
    actions.add_llm("Unsteered reply.")

    agent, history = await _run_order_agent(actions, Steer(tool_calls="to_billing"))
    assert agent.tool_runs == [("to_billing", None)]
    assert history == [
        ("user", "I was charged twice for my order"),
        ("call", "to_billing({})"),
        ("output", ""),
        ("handoff", "billing_agent"),
    ]


@pytest.mark.parametrize(
    "on_error, reply", [("allow", [("assistant", "Unsteered reply.")]), ("block", [])]
)
async def test_user_turn_steer_with_a_tool_the_agent_lacks_follows_on_error(
    on_error: Any, reply: list[tuple[str, str]]
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "where is my order")
    actions.add_llm("Unsteered reply.")

    agent, history = await _run_order_agent(
        actions, Steer(tool_calls=["lookup_order", "lookup_ordr"]), on_error=on_error
    )
    assert agent.tool_runs == []
    assert history == [("user", "where is my order"), *reply]
