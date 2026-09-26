from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from pydantic import TypeAdapter, ValidationError

from livekit.agents import Agent, AgentSession, APIError, decisions
from livekit.agents.decisions import DecisionResponse, ProbabilityResult
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.events import AgentEvent

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


class ControlledModel(decisions.DecisionModel):
    def __init__(self, *, ignore_cancellation: bool = False) -> None:
        super().__init__(capabilities=frozenset({"probability"}))
        self.calls: asyncio.Queue[
            tuple[ChatContext, Mapping[str, decisions.Decision], asyncio.Future[float]]
        ] = asyncio.Queue()
        self.ignore_cancellation = ignore_cancellation
        self.cancelled = asyncio.Event()

    async def _evaluate_impl(
        self,
        *,
        chat_ctx: ChatContext,
        decisions: Mapping[str, decisions.Decision],
        conn_options: APIConnectOptions,
    ) -> decisions.DecisionResponse:
        answer: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        await self.calls.put((chat_ctx, decisions, answer))
        try:
            value = await answer
        except asyncio.CancelledError:
            self.cancelled.set()
            if not self.ignore_cancellation:
                raise
            value = 0.99
        return DecisionResponse(
            results={name: ProbabilityResult(value=value) for name in decisions},
            input_tokens=12,
            output_tokens=2,
        )


def session_for(model: decisions.DecisionModel | None, **options: object) -> AgentSession:
    return AgentSession(
        vad=None,
        llm=FakeLLM(),
        turn_handling={"turn_detection": None},
        user_away_timeout=None,
        decision_model=model,
        decision_options=options,  # type: ignore[arg-type]
    )


class Receptionist(Agent):
    pass


def agent(name: str = "receptionist") -> Agent:
    return Receptionist(
        id=name,
        instructions="Help the caller.",
        decisions={"handoff": decisions.Probability("The caller requests a human.")},
    )


def add_user(session: AgentSession, text: str) -> ChatMessage:
    message = ChatMessage(role="user", content=[text])
    session._conversation_item_added(message)
    return message


async def next_call(model: ControlledModel):
    return await asyncio.wait_for(model.calls.get(), timeout=2)


@pytest.mark.parametrize("background", [False, True], ids=["on_demand_only", "with_background"])
async def test_on_demand_usage_is_collected_through_activity_transitions(background: bool) -> None:
    model = ControlledModel()
    session = session_for(model)
    collected = []
    session.on("session_usage_updated", collected.append)

    async def evaluate() -> None:
        assert session.decision_model is model
        request = asyncio.create_task(
            session.decision_model.evaluate(
                chat_ctx=ChatContext.empty(),
                decisions={"handoff": decisions.Probability("The caller requests a human.")},
            )
        )
        _, _, answer = await next_call(model)
        answer.set_result(0.75)
        await request

    def assert_usage(requests: int) -> None:
        assert len(session.usage.model_usage) == 1
        usage = session.usage.model_usage[0]
        assert usage.type == "decision_usage"
        assert usage.total_requests == requests
        assert usage.input_tokens == 12 * requests
        assert usage.output_tokens == 2 * requests
        assert len(collected) == requests
        assert collected[-1].usage.model_usage[0] == usage

    async with session:
        original = agent() if background else Receptionist(instructions="Use on-demand decisions.")
        await session.start(original)
        await evaluate()
        assert_usage(1)

        await session._update_activity(
            Receptionist(instructions="Temporary task."), previous_activity="pause"
        )
        await evaluate()
        assert_usage(2)

        await session._update_activity(original, new_activity="resume")
        await evaluate()
        assert_usage(3)

        session.update_agent(Receptionist(instructions="The next agent."))
        assert session._update_activity_atask is not None
        await session._update_activity_atask
        await evaluate()
        assert_usage(4)

    # The model can outlive the session. A later call must not update a closed session.
    await evaluate()
    assert_usage(4)


async def test_overlap_keeps_running_and_latest_pending_snapshot() -> None:
    model = ControlledModel()
    async with session_for(model) as session:
        events: asyncio.Queue[decisions.DecisionsCompletedEvent] = asyncio.Queue()
        session.on("decisions_completed", events.put_nowait)
        await session.start(agent())

        first = add_user(session, "Book a table.")
        first_context, _, first_answer = await next_call(model)
        add_user(session, "For two people.")
        third = add_user(session, "Tomorrow at seven.")
        assert model.calls.empty()
        assert [item.text_content for item in first_context.items] == ["Book a table."]

        first_answer.set_result(0.1)
        first_event = await asyncio.wait_for(events.get(), 2)
        assert first_event.source_message_id == first.id
        context, _, last_answer = await next_call(model)
        assert [item.text_content for item in context.items] == [
            "Book a table.",
            "For two people.",
            "Tomorrow at seven.",
        ]
        last_answer.set_result(0.2)
        last_event = await asyncio.wait_for(events.get(), 2)
        assert last_event.source_message_id == third.id
        assert first_event.activity_id == last_event.activity_id
        assert model.calls.empty()
        usage = session.usage.model_usage
        assert len(usage) == 1 and usage[0].type == "decision_usage"
        assert usage[0].input_tokens == 24
        assert usage[0].total_requests == 2
        restored = TypeAdapter(AgentEvent).validate_json(last_event.model_dump_json())
        assert restored == last_event


async def test_cadence_and_context_are_independent_and_snapshot_is_isolated() -> None:
    model = ControlledModel()
    async with session_for(model, turn_interval=2, max_context_turns=2) as session:
        await session.start(agent())
        add_user(session, "old")
        session._conversation_item_added(ChatMessage(role="assistant", content=["old reply"]))
        assert model.calls.empty()
        add_user(session, "second")
        _, _, answer = await next_call(model)
        third = add_user(session, "third")
        session._conversation_item_added(ChatMessage(role="assistant", content=["third reply"]))
        fourth = add_user(session, "fourth")
        # An ineligible fifth turn cannot move the pending request's cutoff.
        add_user(session, "fifth")
        third.content[:] = ["edited later"]
        fourth.content[:] = ["edited later too"]
        answer.set_result(0.1)
        context, _, pending_answer = await next_call(model)
        assert [item.text_content for item in context.items] == ["third", "third reply", "fourth"]
        pending_answer.set_result(0.2)


async def test_timeout_releases_pending_work_without_stopping_session() -> None:
    model = ControlledModel()
    async with session_for(model, timeout=0.05) as session:
        events: asyncio.Queue[decisions.DecisionsCompletedEvent] = asyncio.Queue()
        session.on("decisions_completed", events.put_nowait)
        await session.start(agent())
        add_user(session, "first")
        await next_call(model)
        latest = add_user(session, "second")
        _, _, answer = await next_call(model)
        assert model.cancelled.is_set()
        answer.set_result(0.8)
        event = await asyncio.wait_for(events.get(), 2)
        assert event.source_message_id == latest.id
        assert session.current_agent.id == "receptionist"


async def test_failed_request_does_not_lose_pending_work() -> None:
    model = ControlledModel()
    async with session_for(model) as session:
        events: asyncio.Queue[decisions.DecisionsCompletedEvent] = asyncio.Queue()
        session.on("decisions_completed", events.put_nowait)
        await session.start(agent())
        add_user(session, "first")
        _, _, answer = await next_call(model)
        latest = add_user(session, "second")
        answer.set_exception(APIError("invalid request", retryable=False))
        _, _, next_answer = await next_call(model)
        next_answer.set_result(0.7)
        assert (await asyncio.wait_for(events.get(), 2)).source_message_id == latest.id


async def test_handoff_suppresses_late_result_and_discards_pending_snapshot() -> None:
    model = ControlledModel(ignore_cancellation=True)
    async with session_for(model) as session:
        events: asyncio.Queue[decisions.DecisionsCompletedEvent] = asyncio.Queue()
        session.on("decisions_completed", events.put_nowait)
        await session.start(agent("first_agent"))
        add_user(session, "first")
        await next_call(model)
        add_user(session, "pending")
        session.update_agent(agent("second_agent"))
        assert session._update_activity_atask is not None
        await session._update_activity_atask
        assert events.empty()
        assert model.calls.empty()
        assert model.cancelled.is_set()
        latest = add_user(session, "new agent")
        _, _, answer = await next_call(model)
        answer.set_result(0.4)
        event = await asyncio.wait_for(events.get(), 2)
        assert event.agent_id == "second_agent"
        assert event.source_message_id == latest.id


async def test_close_cancels_background_work_without_emitting_late_result() -> None:
    model = ControlledModel(ignore_cancellation=True)
    events = []
    async with session_for(model) as session:
        session.on("decisions_completed", events.append)
        await session.start(agent())
        add_user(session, "first")
        await next_call(model)
        add_user(session, "pending")
    assert model.cancelled.is_set()
    assert events == []
    assert model.calls.empty()


async def test_pause_and_resume_restart_cadence_without_duplicate_listeners() -> None:
    model = ControlledModel()
    async with session_for(model, turn_interval=2) as session:
        original = agent()
        events: asyncio.Queue[decisions.DecisionsCompletedEvent] = asyncio.Queue()
        session.on("decisions_completed", events.put_nowait)
        await session.start(original)
        add_user(session, "first")
        add_user(session, "second")
        await next_call(model)
        await session._update_activity(
            Agent(instructions="Temporary task, without decisions."), previous_activity="pause"
        )
        assert model.cancelled.is_set()
        add_user(session, "temporary task")
        assert model.calls.empty()
        await session._update_activity(original, new_activity="resume")
        add_user(session, "resumed first")
        assert model.calls.empty()
        source = add_user(session, "resumed second")
        _, _, answer = await next_call(model)
        answer.set_result(0.3)
        event = await asyncio.wait_for(events.get(), 2)
        assert event.source_message_id == source.id
        assert events.empty()
        assert model.calls.empty()
        usage = session.usage.model_usage[0]
        assert usage.type == "decision_usage" and usage.total_requests == 1


async def test_session_reply_does_not_wait_for_decisions() -> None:
    model = ControlledModel()
    async with session_for(model) as session:
        await session.start(agent())
        await asyncio.wait_for(session.run(user_input="Hello"), 2)
        _, _, answer = await next_call(model)
        assert not answer.done()
        answer.set_result(0.1)


async def test_unsupported_kind_fails_before_provider_request() -> None:
    model = ControlledModel()
    with pytest.raises(ValueError, match="does not support"):
        await model.evaluate(
            chat_ctx=ChatContext.empty(),
            decisions={"intent": decisions.Choice("Intent?", options={"a": "A", "b": "B"})},
        )
    assert model.calls.empty()


@pytest.mark.parametrize(
    "unsupported_kind", [False, True], ids=["missing_model", "unsupported_score"]
)
async def test_invalid_decisions_fail_before_start_changes_session_state(
    unsupported_kind: bool,
) -> None:
    model = ControlledModel() if unsupported_kind else None
    invalid_agent = Receptionist(
        instructions="Invalid decisions.",
        decisions={
            "check": decisions.Score("Score?", levels=["Low", "High"])
            if unsupported_kind
            else decisions.Probability("True?")
        },
    )
    error = "does not support.*score" if unsupported_kind else "requires.*decision_model"
    async with session_for(model) as session:
        with pytest.raises(ValueError, match=error):
            await session.start(invalid_agent)
        assert session._agent is None
        assert session._activity is None
        assert session._started_at is None

        valid_agent = Receptionist(instructions="Help the caller.")
        await session.start(valid_agent)
        await asyncio.wait_for(session.run(user_input="Hello after the rejected start."), 2)
        assert session.current_agent is valid_agent
        assert any(
            item.type == "message" and item.text_content == "Hello after the rejected start."
            for item in session.history.items
        )


@pytest.mark.parametrize(
    "unsupported_kind", [False, True], ids=["missing_model", "unsupported_score"]
)
@pytest.mark.parametrize("transition", ["update_agent", "internal_activity_update"])
async def test_rejected_handoff_keeps_current_agent_accepting_turns(
    unsupported_kind: bool,
    transition: str,
) -> None:
    model = ControlledModel() if unsupported_kind else None
    original = Receptionist(instructions="Help the caller.")
    invalid_agent = Receptionist(
        instructions="Invalid decisions.",
        decisions={
            "check": decisions.Score("Score?", levels=["Low", "High"])
            if unsupported_kind
            else decisions.Probability("True?")
        },
    )
    error = "does not support.*score" if unsupported_kind else "requires.*decision_model"
    async with session_for(model) as session:
        await session.start(original)
        original_activity = session._activity
        with pytest.raises(ValueError, match=error):
            if transition == "update_agent":
                session.update_agent(invalid_agent)
            else:
                await session._update_activity(invalid_agent)
        assert session.current_agent is original
        assert session._activity is original_activity
        assert session._update_activity_atask is None
        await asyncio.wait_for(session.run(user_input="Still here after the rejected handoff."), 2)
        assert any(
            item.type == "message" and item.text_content == "Still here after the rejected handoff."
            for item in session.history.items
        )


@pytest.mark.parametrize(
    "options",
    [
        {"turn_interval": 0},
        {"max_context_turns": -1},
        {"timeout": 0},
        {"timeout": float("nan")},
    ],
)
def test_invalid_options_fail_at_construction(options) -> None:
    with pytest.raises(ValueError, match="decision_options"):
        session_for(ControlledModel(), **options)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_probabilities_must_be_finite_and_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        decisions.ProbabilityResult(value=value)
