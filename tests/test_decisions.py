from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import AsyncExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import TypeAdapter, ValidationError

from livekit.agents import Agent, AgentSession, APIError, APITimeoutError, decisions, utils
from livekit.agents.decisions import DecisionResponse, ProbabilityResult
from livekit.agents.llm import (
    AgentHandoff,
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    Toolset,
)
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
        include_context_events: bool,
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


@pytest.mark.parametrize("background", [False, True], ids=["on_demand", "background"])
@pytest.mark.parametrize("reuse_session_model", [False, True])
async def test_shared_model_usage_belongs_only_to_the_calling_session(
    background: bool,
    reuse_session_model: bool,
) -> None:
    model = ControlledModel()
    provider_metrics = []
    model.on("metrics_collected", provider_metrics.append)
    async with (
        session_for(model) as first,
        session_for(first.decision_model if reuse_session_model else model) as second,
        AsyncExitStack() as pending,
    ):
        first_metrics, second_metrics = [], []
        first.on("session_usage_updated", first_metrics.append)
        second.on("session_usage_updated", second_metrics.append)
        await first.start(agent("first"))
        await second.start(agent("second"))

        async def start_request(session):
            if background:
                add_user(session, "Please call me back.")
                task = session._activity._decision_runner._task
            else:
                task = asyncio.create_task(
                    session.decision_model.evaluate(
                        chat_ctx=ChatContext.empty(),
                        decisions={"handoff": decisions.Probability("The caller wants a human.")},
                    )
                )
            pending.push_async_callback(utils.aio.cancel_and_wait, task)
            _, _, answer = await next_call(model)
            return task, answer

        first_request, first_answer = await start_request(first)
        second_request, second_answer = await start_request(second)
        first_answer.set_result(0.9)
        await first_request
        assert first.usage.model_usage[0].input_tokens == 12
        assert second.usage.model_usage == []
        assert len(first_metrics) == 1
        assert second_metrics == []

        second_answer.set_result(0.8)
        await second_request
        assert second.usage.model_usage[0].input_tokens == 12
        assert first.usage.model_usage[0].input_tokens == 12
        assert len(provider_metrics) == 2
        assert len(second_metrics) == 1

        standalone = asyncio.create_task(
            model.evaluate(
                chat_ctx=ChatContext.empty(),
                decisions={"handoff": decisions.Probability("The caller wants a human.")},
            )
        )
        pending.push_async_callback(utils.aio.cancel_and_wait, standalone)
        _, _, answer = await next_call(model)
        answer.set_result(0.7)
        await standalone
        assert len(provider_metrics) == 3
        assert len(first_metrics) == len(second_metrics) == 1

        await first.aclose()
        request, answer = await start_request(second)
        answer.set_result(0.6)
        await request
        assert first.usage.model_usage[0].total_requests == 1
        assert second.usage.model_usage[0].total_requests == 2
        assert len(provider_metrics) == 4


@pytest.mark.parametrize("background", [False, True], ids=["on_demand_only", "with_background"])
async def test_on_demand_usage_tracks_session_lifetime(background: bool) -> None:
    model = ControlledModel()
    session = session_for(model)
    collected = []
    session.on("session_usage_updated", collected.append)

    async def evaluate() -> None:
        assert session.decision_model is not None
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

    await evaluate()
    assert_usage(4)


async def test_on_demand_usage_is_collected_while_next_agent_initializes() -> None:
    setup_started = asyncio.Event()
    finish_setup = asyncio.Event()

    class SlowTools(Toolset):
        async def setup(self):
            setup_started.set()
            await finish_setup.wait()
            return self

    model = ControlledModel()
    async with session_for(model) as session:
        collected = []
        session.on("session_usage_updated", collected.append)
        await session.start(Receptionist(instructions="First agent."))
        request = asyncio.create_task(
            session.decision_model.evaluate(
                chat_ctx=ChatContext.empty(),
                decisions={"check": decisions.Probability("True?")},
            )
        )
        _, _, answer = await next_call(model)
        session.update_agent(
            Receptionist(
                instructions="Next agent.",
                tools=[SlowTools(id="slow_tools")],
            )
        )
        try:
            await asyncio.wait_for(setup_started.wait(), 2)
            # The old activity has closed and the new one's models have not started.
            answer.set_result(0.5)
            await request
            assert len(session.usage.model_usage) == 1
            usage = session.usage.model_usage[0]
            assert usage.type == "decision_usage"
            assert (usage.total_requests, usage.input_tokens, usage.output_tokens) == (1, 12, 2)
            assert len(collected) == 1
        finally:
            finish_setup.set()
            assert session._update_activity_atask is not None
            await session._update_activity_atask
        assert len(collected) == 1


@pytest.mark.parametrize("error", [RuntimeError("startup failed"), asyncio.CancelledError()])
async def test_failed_start_does_not_collect_later_decision_usage(monkeypatch, error) -> None:
    model = ControlledModel()
    session = session_for(model)

    async def fail_start(old_task, agent):
        raise error

    monkeypatch.setattr(session, "_update_activity_task", fail_start)
    with pytest.raises(type(error)):
        # Keep startup's tracing context in its own task.
        await asyncio.create_task(session.start(Receptionist(instructions="First agent.")))
    request = asyncio.create_task(
        session.decision_model.evaluate(
            chat_ctx=ChatContext.empty(),
            decisions={"check": decisions.Probability("True?")},
        )
    )
    _, _, answer = await next_call(model)
    answer.set_result(0.5)
    await request
    assert session.usage.model_usage == []


@pytest.mark.parametrize("error", [RuntimeError("startup failed"), asyncio.CancelledError()])
@pytest.mark.parametrize("stage", ["activity_initialization", "session_host_start"])
@pytest.mark.parametrize("with_decisions", [False, True])
@pytest.mark.no_concurrent
async def test_failed_start_cleans_up_and_can_retry(
    monkeypatch,
    error,
    stage,
    with_decisions: bool,
) -> None:
    from livekit.agents.voice.agent_activity import AgentActivity

    model = ControlledModel()
    session = session_for(model if with_decisions else None)
    original = agent() if with_decisions else Receptionist(instructions="Help the caller.")

    async def fail_scheduling(self):
        raise error

    update_activity = session._update_activity_task

    async def install_failing_host(old_task, agent):
        await update_activity(old_task, agent)
        session._session_host = SimpleNamespace(start=AsyncMock(side_effect=error))

    events: asyncio.Queue[decisions.DecisionsCompletedEvent] = asyncio.Queue()
    session.on("decisions_completed", events.put_nowait)
    async with session:
        with monkeypatch.context() as patch:
            if stage == "activity_initialization":
                patch.setattr(AgentActivity, "_resume_scheduling_task", fail_scheduling)
            else:
                patch.setattr(session, "_update_activity_task", install_failing_host)
            with pytest.raises(type(error)):
                await session.start(original)
        assert not session._started
        await session.aclose()
        add_user(session, "Please call me back.")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(model.calls.get(), 0.02)
        assert events.empty()

        await session.start(original)
        await asyncio.wait_for(session.run(user_input="Hello after retrying."), 2)
        assert session.current_agent is original
        assert any(
            item.type == "message" and item.text_content == "Hello after retrying."
            for item in session.history.items
        )
        if with_decisions:
            _, _, answer = await next_call(model)
            answer.set_result(0.9)
            await asyncio.wait_for(events.get(), 2)
            assert events.empty() and model.calls.empty()
            usage = next(u for u in session.usage.model_usage if u.type == "decision_usage")
            assert usage.total_requests == 1


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
        add_user(session, "fifth")
        third.content[:] = ["edited later"]
        fourth.content[:] = ["edited later too"]
        answer.set_result(0.1)
        context, _, pending_answer = await next_call(model)
        assert [item.text_content for item in context.items] == ["third", "third reply", "fourth"]
        pending_answer.set_result(0.2)


@pytest.mark.parametrize("context_turns", [1, 3])
async def test_context_window_starts_with_user_and_ends_at_source_message(
    context_turns: int,
) -> None:
    model = ControlledModel()
    async with session_for(model, max_context_turns=context_turns) as session:
        session.history.add_message(role="assistant", content="Greeting", created_at=1.0)
        session.history.add_message(role="user", content="First turn", created_at=2.0)
        session.history.add_message(role="system", content="Instructions", created_at=2.5)
        session.history.add_message(role="assistant", content="Reply", created_at=3.0)
        session.history.add_message(role="assistant", content="After the source", created_at=5.0)
        await session.start(agent())

        session._conversation_item_added(
            ChatMessage(
                role="user",
                content=["Current turn"],
                created_at=4.0,
            )
        )
        context, _, answer = await next_call(model)
        expected = (
            ["Current turn"] if context_turns == 1 else ["First turn", "Reply", "Current turn"]
        )
        assert [item.text_content for item in context.items] == expected
        answer.set_result(0.5)


@pytest.mark.parametrize("include_context_events", [False, True])
async def test_background_context_events_respect_window_and_snapshot_isolation(
    include_context_events: bool,
) -> None:
    model = ControlledModel()
    async with session_for(
        model, max_context_turns=2, include_context_events=include_context_events
    ) as session:
        session.history.add_message(role="user", content="Old turn", created_at=1)
        session.history.insert(
            FunctionCall(call_id="old", name="old", arguments="{}", created_at=2)
        )
        session.history.add_message(role="user", content="Book a table", created_at=3)
        call = FunctionCall(call_id="new", name="book", arguments="{}", created_at=4)
        output = FunctionCallOutput(call_id="new", output="Full", is_error=False, created_at=5)
        handoff = AgentHandoff(old_agent_id="booking", new_agent_id="receptionist", created_at=6)
        session.history.insert([call, output, handoff])
        session.history.insert(
            ChatMessage(role="assistant", content=["I can offer"], interrupted=True, created_at=7)
        )
        await session.start(agent())
        session._conversation_item_added(
            ChatMessage(role="user", content=["Please call me back"], created_at=8)
        )
        context, _, answer = await next_call(model)
        call.arguments = '{"changed": true}'
        output.output = "Changed"
        handoff.new_agent_id = "changed"
        if include_context_events:
            assert [item.type for item in context.items] == [
                "message",
                "function_call",
                "function_call_output",
                "agent_handoff",
                "message",
                "message",
            ]
            assert context.items[1].arguments == "{}"
            assert context.items[2].output == "Full"
            assert context.items[3].new_agent_id == "receptionist"
            assert context.items[4].interrupted is True
        else:
            assert all(item.type == "message" for item in context.items)
        assert context.items[0].text_content == "Book a table"
        assert context.items[-1].text_content == "Please call me back"
        answer.set_result(0.5)


@pytest.mark.parametrize("failure", [None, "unknown decisions", "both a result and an error"])
@pytest.mark.parametrize("allow_partial", [False, True])
async def test_completed_batch_records_usage_once_even_when_rejected(
    monkeypatch,
    failure,
    allow_partial: bool,
) -> None:
    response = DecisionResponse(
        results={"handoff": ProbabilityResult(value=0.9)},
        model="resolved-model",
        provider="test-provider",
        input_tokens=50,
        output_tokens=5,
    )
    if failure == "unknown decisions":
        response.results["extra"] = ProbabilityResult(value=0.5)
    elif failure:
        response.errors["handoff"] = "invalid answer"
    model = ControlledModel()
    evaluate = AsyncMock(return_value=response)
    monkeypatch.setattr(model, "_evaluate_impl", evaluate)
    collected = []
    model.on("metrics_collected", collected.append)
    async with session_for(model) as session:
        await session.start(Receptionist(instructions="Help the caller."))
        expectation = pytest.raises(APIError, match=failure) if failure else nullcontext()
        with expectation:
            await session.decision_model.evaluate(
                chat_ctx=ChatContext.empty(),
                decisions={"handoff": decisions.Probability("Wants a human?")},
                allow_partial=allow_partial,
            )
        evaluate.assert_awaited_once()
        [metrics] = collected
        assert (metrics.input_tokens, metrics.output_tokens) == (50, 5)
        [usage] = session.usage.model_usage
        assert (usage.model, usage.provider) == ("resolved-model", "test-provider")
        assert (usage.input_tokens, usage.output_tokens, usage.total_requests) == (50, 5, 1)


@pytest.mark.parametrize("allow_partial", [False, True])
async def test_background_partial_results_and_model_identity(
    monkeypatch, allow_partial: bool
) -> None:
    model = ControlledModel()
    evaluate = AsyncMock(
        return_value=DecisionResponse(
            results={"handoff": ProbabilityResult(value=0.95)},
            errors={"tag": "invalid answer"},
            model="resolved-model",
            provider="test-provider",
            input_tokens=10,
            output_tokens=2,
        )
    )
    monkeypatch.setattr(model, "_evaluate_impl", evaluate)
    async with session_for(model, allow_partial=allow_partial) as session:
        events = []
        session.on("decisions_completed", events.append)
        await session.start(
            Agent(
                instructions="Help the caller.",
                decisions={
                    "handoff": decisions.Probability("Wants a human?"),
                    "tag": decisions.Probability("Wants a booking?"),
                },
            )
        )
        source = add_user(session, "Please call me back.")
        await session._activity._decision_runner._task
        assert len(events) == int(allow_partial)
        if allow_partial:
            [event] = events
            assert set(event.results) == {"handoff"}
            assert event.errors == {"tag": "invalid answer"}
            assert (event.model, event.provider) == ("resolved-model", "test-provider")
            assert event.source_message_id == source.id
            assert TypeAdapter(AgentEvent).validate_json(event.model_dump_json()) == event
        [usage] = session.usage.model_usage
        assert usage.input_tokens == 10
        assert usage.total_requests == 1


@pytest.mark.parametrize("max_retry", [0, 1])
async def test_request_timeout_preserves_retry_count_and_exception_cause(max_retry: int) -> None:
    model = ControlledModel()
    collected = []
    model.on("metrics_collected", collected.append)
    with pytest.raises(APITimeoutError) as error:
        await model.evaluate(
            chat_ctx=ChatContext.empty(),
            decisions={"check": decisions.Probability("True?")},
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0, timeout=0.01),
        )
    assert model.calls.qsize() == max_retry + 1
    assert isinstance(error.value.__cause__, asyncio.TimeoutError)
    assert collected == []


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
