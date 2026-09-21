from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import (
    FunctionCall,
    FunctionToolCall,
    LLMStream,
    Tool,
    ToolChoice,
    ToolError,
    ToolFlag,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.events import ToolCallEnded, ToolCallUpdated, ToolExecutionUpdatedEvent

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities
from .test_realtime_agent_state_during_tool import _generation

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _Trace:
    def __init__(self) -> None:
        self.order: list[str] = []
        self.started: dict[str, asyncio.Event] = {}
        self.updated: dict[str, asyncio.Event] = {}
        self.ended: dict[str, asyncio.Event] = {}
        self.terminals: dict[str, list[ToolCallEnded]] = {}

    def watch(self, session: AgentSession) -> None:
        session.on("tool_execution_updated", self._on_update)

    def event(self, mapping: dict[str, asyncio.Event], call_id: str) -> asyncio.Event:
        return mapping.setdefault(call_id, asyncio.Event())

    def _on_update(self, event: ToolExecutionUpdatedEvent) -> None:
        update = event.update
        if isinstance(update, ToolCallUpdated):
            self.event(self.updated, update.call_id).set()
        elif isinstance(update, ToolCallEnded):
            self.terminals.setdefault(update.call_id, []).append(update)
            self.event(self.ended, update.call_id).set()


class _ObservingFakeLLM(FakeLLM):
    def __init__(self, *, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.observed_inputs: list[str] = []
        self.pending_seen = asyncio.Event()

    def chat(
        self,
        *,
        chat_ctx: Any,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _ObservingFakeLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _ObservingFakeLLMStream(FakeLLMStream):
    _llm: _ObservingFakeLLM

    async def _run(self) -> None:
        input_text = self._get_index_text()
        self._llm.observed_inputs.append(input_text)
        if "pending" in input_text:
            self._llm.pending_seen.set()
        await super()._run()


class _DelayedBatchFakeLLM(_ObservingFakeLLM):
    def __init__(self, *, first_call: FunctionToolCall, second_call: FunctionToolCall) -> None:
        super().__init__(fake_responses=[])
        self.first_call = first_call
        self.second_call = second_call
        self.emit_second = asyncio.Event()
        self.close_stream = asyncio.Event()
        self.stream_eof = asyncio.Event()

    def chat(
        self,
        *,
        chat_ctx: Any,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _DelayedBatchFakeLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _DelayedBatchFakeLLMStream(FakeLLMStream):
    _llm: _DelayedBatchFakeLLM

    async def _run(self) -> None:
        if self._get_index_text() != "batch":
            return
        self._llm.observed_inputs.append("batch")
        self._send_chunk(tool_calls=[self._llm.first_call])
        await self._llm.emit_second.wait()
        self._send_chunk(tool_calls=[self._llm.second_call])
        await self._llm.close_stream.wait()
        self._llm.stream_eof.set()


class _RecordingRealtimeSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel, *, turn_detection_disabled: bool = False) -> None:
        super().__init__(model, turn_detection_disabled=turn_detection_disabled)
        self.progress_committed = asyncio.Event()
        self.meal_committed = asyncio.Event()
        self.reply_created = asyncio.Event()

    async def update_chat_ctx(self, chat_ctx: Any) -> None:
        await super().update_chat_ctx(chat_ctx)
        if any(
            item.type == "function_call_output"
            and item.call_id == "room"
            and "room reservation is pending" in str(item.output)
            for item in chat_ctx.items
        ):
            self.progress_committed.set()
        if any(
            item.type == "function_call_output"
            and item.call_id == "meal_final"
            and "meal saved" in str(item.output)
            for item in chat_ctx.items
        ):
            self.meal_committed.set()

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[Any]:
        future = super().generate_reply(
            instructions=instructions,
            tool_choice=tool_choice,
            tools=tools,
        )
        self.reply_created.set()
        return future


class _RecordingRealtimeModel(FakeRealtimeModel):
    def session(self, *, turn_detection_disabled: bool = False) -> _RecordingRealtimeSession:
        session = _RecordingRealtimeSession(
            self,
            turn_detection_disabled=turn_detection_disabled,
        )
        session.update_error = self.bring_up_error
        self.created_sessions.append(session)
        return session


async def _resolve_fake_realtime_replies(
    realtime_session: _RecordingRealtimeSession,
    stop: asyncio.Event,
) -> None:
    index = 1  # the initial response future is resolved by the test itself
    while not stop.is_set():
        while index < len(realtime_session._reply_futs):
            future = realtime_session._reply_futs[index]
            if not future.done():
                future.set_result(
                    _generation(
                        response_id=f"followup-{index}",
                        text="progress acknowledged",
                        audio_duration=0.01,
                    )
                )
            index += 1

        if stop.is_set():
            return
        realtime_session.reply_created.clear()
        stop_wait = asyncio.create_task(stop.wait())
        reply_wait = asyncio.create_task(realtime_session.reply_created.wait())
        done, pending = await asyncio.wait(
            {stop_wait, reply_wait}, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not done:
            raise asyncio.TimeoutError("fake realtime reply future was not created")
        if stop_wait in done:
            return


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=5)


def _await_chain(task: asyncio.Task[Any]) -> str:
    chain: list[str] = []
    awaitable: Any = task.get_coro()
    seen: set[int] = set()
    while awaitable is not None and id(awaitable) not in seen:
        seen.add(id(awaitable))
        name = getattr(awaitable, "__qualname__", type(awaitable).__name__)
        frame = getattr(awaitable, "cr_frame", None)
        if frame is not None:
            name = f"{name} ({frame.f_code.co_filename}:{frame.f_lineno})"
        chain.append(name)
        child = getattr(awaitable, "cr_await", None)
        if child is None:
            child = getattr(awaitable, "gi_yieldfrom", None)
        awaitable = child
    return " -> ".join(chain)


async def _close(session: AgentSession) -> None:
    try:
        await asyncio.wait_for(session.aclose(), timeout=5)
    except asyncio.TimeoutError:
        print("live tasks at natural session close timeout:")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and not task.done():
                print(f"  {task.get_name()}: {_await_chain(task)}")
        # Keep a failing lifecycle assertion from leaking a blocked speech/tool into
        # the next test. The timeout is re-raised after bounded best-effort cleanup.
        activity = session._activity
        if activity is not None:
            with contextlib.suppress(BaseException):
                await activity._tool_executor.cancel_all()
            speech = activity.current_speech
            if speech is not None:
                for task in speech._tasks:
                    task.cancel()
                speech._mark_done()
        dangling = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and (
                task.get_name().startswith("tool_dependency_")
                or task.get_name() in {"execute_tools_task", "tool_dependency_ready"}
            )
        ]
        for task in dangling:
            task.cancel()
        if dangling:
            await asyncio.gather(*dangling, return_exceptions=True)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(session.aclose(), timeout=2)
        raise


def _response(user_input: str, *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(
        input=user_input,
        content="",
        ttft=0,
        duration=0,
        tool_calls=list(calls),
    )


def _new_session(llm: FakeLLM, *, on_dependency_error: str = "skip") -> AgentSession:
    return AgentSession(
        llm=llm,
        stt=None,
        vad=None,
        tts=None,
        turn_handling={"turn_detection": None},
        tool_handling={"on_dependency_error": on_dependency_error},
    )


async def _start(
    agent: Agent,
    responses: list[FakeLLMResponse],
    *,
    on_dependency_error: str = "skip",
) -> AgentSession:
    session = _new_session(
        FakeLLM(fake_responses=responses), on_dependency_error=on_dependency_error
    )
    await session.start(agent)
    return session


@pytest.mark.asyncio
async def test_reverse_arrival_keeps_dependency_parked_until_terminal() -> None:
    trace = _Trace()
    room_started = asyncio.Event()
    release_room = asyncio.Event()
    meal_started = asyncio.Event()
    premature_meal = asyncio.Event()

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        trace.order.append("room:start")
        room_started.set()
        await release_room.wait()
        trace.order.append("room:end")
        return "room saved"

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        trace.order.append("meal:start")
        meal_started.set()
        if not trace.terminals.get("room"):
            premature_meal.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    session = await _start(
        agent,
        [
            _response(
                "book",
                FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
                FunctionToolCall(name="save_room", arguments="{}", call_id="room"),
            )
        ],
    )
    trace.watch(session)
    try:
        session.generate_reply(user_input="book")
        await _wait(room_started)
        release_room.set()
        await _wait(meal_started)
        assert not premature_meal.is_set()
        assert trace.order.index("room:end") < trace.order.index("meal:start")
        await _wait(trace.event(trace.ended, "meal"))
        assert [item.status for item in trace.terminals["meal"]] == ["done"]
    finally:
        release_room.set()
        await _close(session)


@pytest.mark.asyncio
async def test_progress_is_visible_before_dependency_terminal_and_does_not_admit_child() -> None:
    trace = _Trace()
    room_started = asyncio.Event()
    release_room = asyncio.Event()
    meal_started = asyncio.Event()
    premature_meal = asyncio.Event()

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        room_started.set()
        await ctx.update("room reservation is pending")
        await release_room.wait()
        return "room reservation completed"

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        if not trace.terminals.get("room"):
            premature_meal.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    llm = _ObservingFakeLLM(
        fake_responses=[
            _response(
                "book",
                FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
                FunctionToolCall(name="save_room", arguments="{}", call_id="room"),
            )
        ]
    )
    session = _new_session(llm)
    await session.start(agent)
    trace.watch(session)
    try:
        session.generate_reply(user_input="book")
        await _wait(room_started)
        await _wait(llm.pending_seen)
        outputs = [
            item
            for item in agent.chat_ctx.items
            if item.type == "function_call_output" and item.call_id == "room"
        ]
        assert outputs and "room reservation is pending" in outputs[-1].output

        release_room.set()
        await _wait(meal_started)
        assert not premature_meal.is_set()
        await _wait(trace.event(trace.ended, "room"))
        await _wait(trace.event(trace.ended, "meal"))
    finally:
        release_room.set()
        await _close(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy, should_run", [("skip", False), ("run", True)])
async def test_terminal_predecessor_failure_obeys_policy(policy: str, should_run: bool) -> None:
    trace = _Trace()
    meal_started = asyncio.Event()

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        raise ToolError("room provider rejected the reservation")

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    session = await _start(
        agent,
        [
            _response(
                "book",
                FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
                FunctionToolCall(name="save_room", arguments="{}", call_id="room"),
            )
        ],
        on_dependency_error=policy,
    )
    trace.watch(session)
    try:
        session.generate_reply(user_input="book")
        await _wait(trace.event(trace.ended, "room"))
        await _wait(trace.event(trace.ended, "meal"))
        assert meal_started.is_set() is should_run
        assert len(trace.terminals["room"]) == 1
        assert len(trace.terminals["meal"]) == 1
        assert trace.terminals["room"][0].status == "error"
        assert trace.terminals["meal"][0].status == ("done" if should_run else "error")
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_chain_waits_for_terminal_outcomes_and_repeated_names_wait_for_all() -> None:
    order: list[str] = []
    started = {name: asyncio.Event() for name in ("a", "b", "c")}
    release_a = asyncio.Event()
    release_a2_first = asyncio.Event()
    release_a2_second = asyncio.Event()
    started_a2_first = asyncio.Event()
    started_a2_second = asyncio.Event()
    premature_chain = asyncio.Event()
    premature_repeated = asyncio.Event()
    b_all_started = asyncio.Event()
    trace = _Trace()

    @function_tool(name="a")
    async def a(ctx: RunContext) -> str:
        order.append("a:start")
        started["a"].set()
        await release_a.wait()
        order.append("a:end")
        return "a done"

    @function_tool(name="b", after=("a",))
    async def b(ctx: RunContext) -> str:
        order.append("b:start")
        started["b"].set()
        if not trace.terminals.get("a"):
            premature_chain.set()
        return "b done"

    @function_tool(name="c", after=("b",))
    async def c(ctx: RunContext) -> str:
        order.append("c:start")
        started["c"].set()
        if not trace.terminals.get("b"):
            premature_chain.set()
        return "c done"

    @function_tool(name="a_second")
    async def a_second(ctx: RunContext, which: str) -> str:
        if which == "first":
            started_a2_first.set()
            await release_a2_first.wait()
        else:
            started_a2_second.set()
            await release_a2_second.wait()
        order.append(f"a2:{which}:end")
        return f"a2 {which} done"

    # The repeated-name assertion uses two calls to the same declared predecessor.
    @function_tool(name="b_all", after=("a_second",))
    async def b_all(ctx: RunContext) -> str:
        order.append("b_all:start")
        b_all_started.set()
        if len(trace.terminals.get("a2", [])) + len(trace.terminals.get("a3", [])) != 2:
            premature_repeated.set()
        return "b_all done"

    agent = Agent(instructions="graph", tools=[a, b, c, a_second, b_all])
    session = await _start(
        agent,
        [
            _response(
                "graph",
                FunctionToolCall(name="c", arguments="{}", call_id="c"),
                FunctionToolCall(name="b", arguments="{}", call_id="b"),
                FunctionToolCall(name="a", arguments="{}", call_id="a"),
                FunctionToolCall(name="b_all", arguments="{}", call_id="b_all"),
                FunctionToolCall(name="a_second", arguments='{"which":"first"}', call_id="a2"),
                FunctionToolCall(name="a_second", arguments='{"which":"second"}', call_id="a3"),
            )
        ],
    )
    trace.watch(session)
    try:
        session.generate_reply(user_input="graph")
        await _wait(started["a"])
        release_a.set()
        await _wait(started["b"])
        await _wait(started["c"])
        assert not premature_chain.is_set()
        assert order.index("a:end") < order.index("b:start") < order.index("c:start")
        await _wait(started_a2_first)
        await _wait(started_a2_second)
        assert not b_all_started.is_set()
        release_a2_first.set()
        await _wait(trace.event(trace.ended, "a2"))
        assert not b_all_started.is_set()
        release_a2_second.set()
        # b_all must wait for both same-name predecessors, not arrival order.
        await _wait(b_all_started)
        assert not premature_repeated.is_set()
        assert order.count("b_all:start") == 1
    finally:
        release_a.set()
        release_a2_first.set()
        release_a2_second.set()
        await _close(session)


@pytest.mark.asyncio
async def test_absent_registered_predecessor_is_not_a_cross_turn_wait() -> None:
    meal_started = asyncio.Event()

    @function_tool(name="never_called")
    async def never_called(ctx: RunContext) -> str:
        raise AssertionError("absent predecessor was invoked")

    @function_tool(name="save_meal", after=("never_called",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[never_called, save_meal])
    session = await _start(
        agent,
        [_response("book", FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"))],
    )
    try:
        session.generate_reply(user_input="book")
        await _wait(meal_started)
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_invalid_arguments_are_predecessor_failures() -> None:
    trace = _Trace()
    meal_started = asyncio.Event()

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext, room: str) -> str:
        return room

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    session = await _start(
        agent,
        [
            _response(
                "book",
                FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
                FunctionToolCall(name="save_room", arguments="{}", call_id="room-invalid"),
            )
        ],
    )
    trace.watch(session)
    try:
        session.generate_reply(user_input="book")
        await _wait(trace.event(trace.ended, "room-invalid"))
        await _wait(trace.event(trace.ended, "meal"))
        assert not meal_started.is_set()
        assert trace.terminals["meal"][0].status == "error"
        assert len(trace.terminals["meal"]) == 1
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_duplicate_rejection_is_a_predecessor_failure() -> None:
    trace = _Trace()
    meal_started = asyncio.Event()
    premature_meal = asyncio.Event()
    first_room_started = asyncio.Event()
    release_room = asyncio.Event()

    @function_tool(name="save_room", on_duplicate="reject")
    async def save_room(ctx: RunContext, room: str) -> str:
        first_room_started.set()
        await release_room.wait()
        return room

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        if not trace.terminals.get("room-1") or not trace.terminals.get("room-2"):
            premature_meal.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    session = await _start(
        agent,
        [
            _response(
                "book",
                FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
                FunctionToolCall(name="save_room", arguments='{"room":"A"}', call_id="room-1"),
                FunctionToolCall(name="save_room", arguments='{"room":"A"}', call_id="room-2"),
            )
        ],
    )
    trace.watch(session)
    try:
        session.generate_reply(user_input="book")
        await _wait(first_room_started)
        release_room.set()
        await _wait(trace.event(trace.ended, "room-1"))
        await _wait(trace.event(trace.ended, "room-2"))
        await _wait(trace.event(trace.ended, "meal"))
        assert not premature_meal.is_set()
        assert not meal_started.is_set()
        assert trace.terminals["room-2"][0].status == "error"
        assert trace.terminals["meal"][0].status == "error"
    finally:
        release_room.set()
        await _close(session)


@pytest.mark.asyncio
async def test_closing_with_queued_dependent_cancels_without_orphaning_terminal_results() -> None:
    trace = _Trace()
    room_started = asyncio.Event()
    release_room = asyncio.Event()
    room_cancelled = asyncio.Event()
    meal_started = asyncio.Event()

    @function_tool(name="save_room", flags=ToolFlag.CANCELLABLE)
    async def save_room(ctx: RunContext) -> str:
        room_started.set()
        try:
            await release_room.wait()
        except asyncio.CancelledError:
            room_cancelled.set()
            raise
        return "never"

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    session = await _start(
        agent,
        [
            _response(
                "book",
                FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
                FunctionToolCall(name="save_room", arguments="{}", call_id="room"),
            )
        ],
    )
    trace.watch(session)
    try:
        session.generate_reply(user_input="book")
        await _wait(room_started)
        try:
            await _close(session)
        except asyncio.TimeoutError:
            # Preserve the red lifecycle result while ensuring a broken close does
            # not leak this test's intentionally blocked tool into later tests.
            release_room.set()
            await _close(session)
            raise
        await _wait(room_cancelled)
        assert not meal_started.is_set()
        await _wait(trace.event(trace.ended, "room"))
        await _wait(trace.event(trace.ended, "meal"))
        assert len(trace.terminals["room"]) == 1
        assert trace.terminals["room"][0].status in ("cancelled", "error")
        assert len(trace.terminals["meal"]) == 1
        assert trace.terminals["meal"][0].status in ("cancelled", "error")
    finally:
        release_room.set()


@pytest.mark.asyncio
async def test_dependency_predecessor_can_await_agent_task_before_dependent_runs() -> None:
    meal_started = asyncio.Event()
    task_entered = asyncio.Event()

    class QuestionTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="question")

        async def on_enter(self) -> None:
            task_entered.set()
            self.session.generate_reply(instructions="question")

        @function_tool
        async def finish(self, ctx: RunContext) -> str:
            self.complete("answered")
            return "answered"

    @function_tool(name="start_question")
    async def start_question(ctx: RunContext) -> str:
        return await QuestionTask()

    @function_tool(name="save_meal", after=("start_question",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        return "meal saved"

    agent = Agent(instructions="root", tools=[start_question, save_meal])
    session = await _start(
        agent,
        [
            _response(
                "start",
                FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
                FunctionToolCall(name="start_question", arguments="{}", call_id="question"),
            ),
            _response("question"),
            _response("finish", FunctionToolCall(name="finish", arguments="{}", call_id="finish")),
            _response("answered"),
        ],
    )
    try:
        first_result = await asyncio.wait_for(session.run(user_input="start"), timeout=5)
        assert first_result is not None
        await _wait(task_entered)
        assert not meal_started.is_set()
        await asyncio.wait_for(session.run(user_input="finish"), timeout=5)
        await _wait(meal_started)
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_queued_dependent_can_handoff_without_artificial_progress_update() -> None:
    room_started = asyncio.Event()
    target_entered = asyncio.Event()

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    target = Target(instructions="target")

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        room_started.set()
        return "room saved"

    @function_tool(name="handoff", after=("save_room",))
    async def handoff(ctx: RunContext) -> Agent:
        # This must be admitted as a real handoff. It has no ctx.update(), so a
        # scheduler-generated pending output must not poison executor handoff rules.
        return target

    agent = Agent(instructions="root", tools=[save_room, handoff])
    session = await _start(
        agent,
        [
            _response(
                "switch",
                FunctionToolCall(name="handoff", arguments="{}", call_id="handoff"),
                FunctionToolCall(name="save_room", arguments="{}", call_id="room"),
            )
        ],
    )
    trace = _Trace()
    trace.watch(session)
    try:
        session.generate_reply(user_input="switch")
        await _wait(room_started)
        await _wait(target_entered)
        assert session.current_agent is target
        assert trace.terminals["handoff"][0].status == "done"
        assert not trace.terminals["handoff"][0].message
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_registered_dependency_mode_keeps_independent_roots_eager_until_stream_eof() -> None:
    fast_started = asyncio.Event()
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()

    @function_tool(name="fast_root")
    async def fast_root(ctx: RunContext) -> str:
        fast_started.set()
        return "fast done"

    @function_tool(name="slow_root")
    async def slow_root(ctx: RunContext) -> str:
        slow_started.set()
        await release_slow.wait()
        return "slow done"

    @function_tool(name="unused_dependent", after=("never_emitted",))
    async def unused_dependent(ctx: RunContext) -> str:
        raise AssertionError("an absent prerequisite must not invoke this tool")

    llm = _DelayedBatchFakeLLM(
        first_call=FunctionToolCall(name="fast_root", arguments="{}", call_id="fast"),
        second_call=FunctionToolCall(name="slow_root", arguments="{}", call_id="slow"),
    )
    agent = Agent(
        instructions="batch",
        tools=[fast_root, slow_root, unused_dependent],
    )
    session = _new_session(llm)
    await session.start(agent)
    try:
        speech = session.generate_reply(user_input="batch")
        await _wait(fast_started)
        assert not llm.emit_second.is_set()
        assert not speech.done(), "stream membership is not final before EOF"

        llm.emit_second.set()
        await _wait(slow_started)
        assert not llm.close_stream.is_set()
        assert not speech.done(), "slow root is still pending before terminal completion"

        llm.close_stream.set()
        await _wait(llm.stream_eof)
        assert not speech.done(), "EOF must not admit completion before slow root settles"

        release_slow.set()
        await asyncio.wait_for(speech, timeout=5)
    finally:
        release_slow.set()
        llm.emit_second.set()
        llm.close_stream.set()
        await _close(session)


@pytest.mark.asyncio
async def test_realtime_progress_dependency_waits_for_terminal_tool_result() -> None:
    harness_tasks = set(asyncio.all_tasks())
    model = _RecordingRealtimeModel(
        capabilities=fake_capabilities(auto_tool_reply_generation=False)
    )
    trace = _Trace()
    room_started = asyncio.Event()
    release_room = asyncio.Event()
    meal_started = asyncio.Event()
    premature_meal = asyncio.Event()
    stop_reply_resolver = asyncio.Event()

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        room_started.set()
        await ctx.update("room reservation is pending")
        await release_room.wait()
        return "room reservation completed"

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        meal_started.set()
        if not trace.terminals.get("room"):
            premature_meal.set()
        return "meal saved"

    agent = Agent(instructions="booking", tools=[save_room, save_meal])
    session = AgentSession(
        llm=model,
        tool_handling={"on_dependency_error": "skip"},
    )
    realtime_session: _RecordingRealtimeSession | None = None
    reply_resolver: asyncio.Task[None] | None = None
    body_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        session.output.audio = FakeAudioOutput()
        await session.start(agent)
        trace.watch(session)
        realtime_session = model.active_session
        assert realtime_session is not None
        reply = session.generate_reply()
        await _wait(realtime_session.reply_created)
        assert realtime_session._reply_futs
        realtime_session._reply_futs[0].set_result(
            _generation(
                response_id="dependency",
                text="checking the room",
                audio_duration=0.01,
                function_calls=[
                    FunctionCall(call_id="meal", name="save_meal", arguments="{}"),
                    FunctionCall(call_id="room", name="save_room", arguments="{}"),
                ],
            )
        )
        realtime_session.reply_created.clear()
        reply_resolver = asyncio.create_task(
            _resolve_fake_realtime_replies(realtime_session, stop_reply_resolver)
        )

        await _wait(room_started)
        await _wait(realtime_session.progress_committed)
        outputs = [
            item
            for item in realtime_session.chat_ctx.items
            if item.type == "function_call_output" and item.call_id == "room"
        ]
        assert outputs and "room reservation is pending" in outputs[-1].output
        assert not meal_started.is_set()

        release_room.set()
        await _wait(meal_started)
        assert not premature_meal.is_set()
        await _wait(trace.event(trace.ended, "room"))
        await _wait(trace.event(trace.ended, "meal"))
        await _wait(realtime_session.meal_committed)
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)
    except BaseException as exc:
        body_error = exc
    finally:
        release_room.set()
        try:
            await _close(session)
        except BaseException as exc:
            close_error = exc
        finally:
            stop_reply_resolver.set()
            if reply_resolver is not None and realtime_session is not None:
                realtime_session.reply_created.set()
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(reply_resolver, timeout=5)
            if close_error is None and body_error is None:
                leaked_tasks = [
                    task
                    for task in asyncio.all_tasks()
                    if task not in harness_tasks
                    and task is not asyncio.current_task()
                    and not task.done()
                ]
                assert not leaked_tasks, "realtime session leaked tasks: " + ", ".join(
                    f"{task.get_name()}: {_await_chain(task)}" for task in leaked_tasks
                )

    if body_error is not None:
        if close_error is not None:
            body_error.add_note(f"natural session close also failed: {close_error!r}")
        raise body_error
    if close_error is not None:
        raise close_error
