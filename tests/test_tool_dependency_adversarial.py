from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall, ToolContext, ToolFlag
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.voice.events import ToolCallEnded, ToolCallUpdated, ToolExecutionUpdatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse
from .tool_dependency_helpers import (
    DelayedBatchFakeLLM as _DelayedBatchFakeLLM,
    close as _close,
    response as _response,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


async def _start(
    agent: Agent,
    responses: list[FakeLLMResponse],
) -> AgentSession:
    session = AgentSession(
        llm=FakeLLM(fake_responses=responses),
    )
    await session.start(agent)
    return session


@pytest.mark.asyncio
async def test_replaced_prerequisite_finishes_before_dependent() -> None:
    first_started = asyncio.Event()
    replacement_started = asyncio.Event()
    dependent_started = asyncio.Event()
    dependent_terminal = asyncio.Event()
    terminal_events: list[ToolCallEnded] = []
    calls: list[str] = []

    @function_tool(name="prepare", flags=ToolFlag.CANCELLABLE, on_duplicate="replace")
    async def prepare(ctx: RunContext, version: str) -> str:
        calls.append(version)
        if version == "first":
            first_started.set()
            await asyncio.Event().wait()
        replacement_started.set()
        return version

    @function_tool(name="commit", after=("prepare",))
    async def commit(ctx: RunContext) -> str:
        dependent_started.set()
        return "committed"

    session = await _start(
        Agent(instructions="workflow", tools=[prepare, commit]),
        [
            _response(
                "replace",
                FunctionToolCall(name="commit", arguments="{}", call_id="commit-call"),
                FunctionToolCall(
                    name="prepare", arguments='{"version":"first"}', call_id="prepare-1"
                ),
                FunctionToolCall(
                    name="prepare", arguments='{"version":"second"}', call_id="prepare-2"
                ),
            )
        ],
    )

    def observe_terminal(event: ToolExecutionUpdatedEvent) -> None:
        if isinstance(event.update, ToolCallEnded):
            terminal_events.append(event.update)
            if event.update.call_id == "commit-call":
                dependent_terminal.set()

    session.on("tool_execution_updated", observe_terminal)
    try:
        session.generate_reply(user_input="replace")
        await asyncio.wait_for(first_started.wait(), timeout=5)
        await asyncio.wait_for(replacement_started.wait(), timeout=5)
        assert calls == ["first", "second"]
        await asyncio.wait_for(dependent_terminal.wait(), timeout=5)
        assert dependent_started.is_set()
        assert [event.call_id for event in terminal_events] == [
            "prepare-1",
            "prepare-2",
            "commit-call",
        ]
        dependent_event = next(event for event in terminal_events if event.call_id == "commit-call")
        assert dependent_event.id == "commit-call"
        assert dependent_event.status == "done"
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_duplicate_call_id_is_one_side_effect_and_one_explicit_failure() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    side_effects: list[str] = []

    @function_tool(name="write")
    async def write(ctx: RunContext, value: str) -> str:
        side_effects.append(value)
        started.set()
        await release.wait()
        return value

    @function_tool(name="after_write", after=("write",))
    async def after_write(ctx: RunContext) -> str:
        return "after"

    agent = Agent(instructions="writes", tools=[write, after_write])
    session = await _start(
        agent,
        [
            _response(
                "write",
                FunctionToolCall(name="after_write", arguments="{}", call_id="after"),
                FunctionToolCall(name="write", arguments='{"value":"one"}', call_id="same"),
                FunctionToolCall(name="write", arguments='{"value":"two"}', call_id="same"),
            )
        ],
    )
    outputs_committed = asyncio.Event()
    refusal_events: list[ToolCallEnded] = []
    history_insert = session.history.insert

    def observe_history(items: Any) -> None:
        history_insert(items)
        outputs = [
            item
            for item in session.history.items
            if item.type == "function_call_output" and item.name == "write"
        ]
        if any(item.output == "one" for item in outputs) and any(
            item.is_error and "duplicate function call id" in item.output for item in outputs
        ):
            outputs_committed.set()

    def observe_terminal(event: ToolExecutionUpdatedEvent) -> None:
        update = event.update
        if isinstance(update, ToolCallEnded) and "duplicate function call id" in str(
            update.message
        ):
            refusal_events.append(update)

    session.history.insert = observe_history
    session.on("tool_execution_updated", observe_terminal)
    try:
        session.generate_reply(user_input="write")
        await asyncio.wait_for(started.wait(), timeout=5)
        release.set()
        await asyncio.wait_for(outputs_committed.wait(), timeout=5)
        assert side_effects == ["one"]
        outputs = [
            item
            for item in agent.chat_ctx.items
            if item.type == "function_call_output" and item.name == "write"
        ]
        accepted = [item for item in outputs if not item.is_error]
        refusals = [item for item in outputs if item.is_error]
        assert len(accepted) == len(refusals) == 1
        assert accepted[0].call_id == "same"
        assert accepted[0].output == "one"
        assert "duplicate function call id" in refusals[0].output
        assert refusals[0].call_id != accepted[0].call_id
        assert len(refusal_events) == 1
        assert refusal_events[0].call_id == "same"
        assert refusal_events[0].id == refusals[0].call_id
    finally:
        release.set()
        await _close(session)


@pytest.mark.asyncio
async def test_malformed_prerequisite_settles_before_dependent() -> None:
    dependent_started = asyncio.Event()
    dependent_terminal = asyncio.Event()
    terminal_events: list[ToolCallEnded] = []

    @function_tool(name="prepare")
    async def prepare(ctx: RunContext, value: str) -> str:
        return value

    @function_tool(name="commit", after=("prepare",))
    async def commit(ctx: RunContext) -> str:
        dependent_started.set()
        return "committed"

    session = await _start(
        Agent(instructions="workflow", tools=[prepare, commit]),
        [
            _response(
                "malformed",
                FunctionToolCall(name="commit", arguments="{}", call_id="commit-call"),
                FunctionToolCall(name="prepare", arguments="not-json", call_id="prepare-call"),
            )
        ],
    )

    def observe_tool_update(event: ToolExecutionUpdatedEvent) -> None:
        update = event.update
        if isinstance(update, ToolCallEnded):
            terminal_events.append(update)
            if update.call_id == "commit-call":
                dependent_terminal.set()

    session.on("tool_execution_updated", observe_tool_update)
    try:
        session.generate_reply(user_input="malformed")
        await asyncio.wait_for(dependent_terminal.wait(), timeout=5)
        assert dependent_started.is_set()
        assert [event.call_id for event in terminal_events] == ["prepare-call", "commit-call"]
        dependent_terminals = [event for event in terminal_events if event.call_id == "commit-call"]
        assert len(dependent_terminals) == 1
        dependent_terminal_event = dependent_terminals[0]
        assert dependent_terminal_event.id == "commit-call"
        assert dependent_terminal_event.status == "done"

        prerequisite_terminals = [
            event for event in terminal_events if event.call_id == "prepare-call"
        ]
        assert len(prerequisite_terminals) == 1
        prerequisite_terminal_event = prerequisite_terminals[0]
        assert prerequisite_terminal_event.status == "error"
        assert prerequisite_terminal_event.message is not None
        assert "Error parsing arguments for `prepare`" in prerequisite_terminal_event.message
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_close_does_not_admit_pending_dependent() -> None:
    root_started = asyncio.Event()
    dependent_started = asyncio.Event()

    @function_tool(name="root", flags=ToolFlag.CANCELLABLE)
    async def root(ctx: RunContext) -> str:
        root_started.set()
        await asyncio.Event().wait()
        return "root"

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        dependent_started.set()
        return "dependent"

    session = await _start(
        Agent(instructions="close", tools=[root, dependent]),
        [
            _response(
                "close-run",
                FunctionToolCall(name="dependent", arguments="{}", call_id="dependent"),
                FunctionToolCall(name="root", arguments="{}", call_id="root"),
            )
        ],
    )
    try:
        session.generate_reply(user_input="close-run")
        await asyncio.wait_for(root_started.wait(), timeout=5)
        await asyncio.wait_for(session.aclose(), timeout=5)
        assert not dependent_started.is_set()
    finally:
        if not session._closing:
            await _close(session)


@pytest.mark.parametrize("with_progress", [False, True])
@pytest.mark.asyncio
async def test_pending_dependent_is_replaced_by_one_final_history_output(
    with_progress: bool,
) -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    dependent_progress_observed = asyncio.Event()
    release_dependent = asyncio.Event()
    dependent_final_committed = asyncio.Event()

    @function_tool(name="root")
    async def root(ctx: RunContext) -> str:
        root_started.set()
        await release_root.wait()
        return "root"

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        if with_progress:
            await ctx.update("dependent-progress")
            await release_dependent.wait()
        return "dependent-final"

    agent = Agent(instructions="history", tools=[root, dependent])
    session = await _start(
        agent,
        [
            _response(
                "history",
                FunctionToolCall(name="dependent", arguments="{}", call_id="dependent"),
                FunctionToolCall(name="root", arguments="{}", call_id="root"),
            )
        ],
    )

    history_insert = session.history.insert

    def observe_history_insert(items: Any) -> None:
        history_insert(items)
        if any(
            item.type == "function_call_output" and item.output == "dependent-final"
            for item in session.history.items
        ):
            dependent_final_committed.set()

    session.history.insert = observe_history_insert

    def observe_tool_update(event: ToolExecutionUpdatedEvent) -> None:
        if (
            isinstance(event.update, ToolCallUpdated)
            and event.update.call_id == "dependent"
            and event.update.message == "dependent-progress"
        ):
            dependent_progress_observed.set()

    session.on("tool_execution_updated", observe_tool_update)
    try:
        session.generate_reply(user_input="history")
        await asyncio.wait_for(root_started.wait(), timeout=5)
        release_root.set()
        if with_progress:
            await asyncio.wait_for(dependent_progress_observed.wait(), timeout=5)
            release_dependent.set()
        await asyncio.wait_for(dependent_final_committed.wait(), timeout=5)
        expected_ids = {"dependent_final"}
        if with_progress:
            expected_ids.add("dependent_update_0")

        for chat_ctx in (agent.chat_ctx, agent.chat_ctx.copy(tools=agent.tools)):
            outputs = [
                item
                for item in chat_ctx.items
                if item.type == "function_call_output"
                and item.call_id in {"dependent", *expected_ids}
            ]
            pending = [item for item in outputs if item.call_id == "dependent"]
            assert len(pending) == 1
            assert "pending prerequisite" in pending[0].output

            final_outputs = [item for item in outputs if item.call_id in expected_ids]
            assert {item.call_id for item in final_outputs} == expected_ids
            assert sum(item.output == "dependent-final" for item in final_outputs) == 1

            matching_calls = [
                item
                for item in chat_ctx.items
                if item.type == "function_call" and item.call_id in expected_ids
            ]
            assert {item.call_id for item in matching_calls} == expected_ids
            assert {item.call_id for item in matching_calls} == {
                item.call_id for item in final_outputs
            }
            relevant_ids = [item.id for item in (*matching_calls, *final_outputs)]
            assert len(relevant_ids) == len(set(relevant_ids))
    finally:
        release_root.set()
        release_dependent.set()
        await _close(session)


@pytest.mark.asyncio
async def test_permanent_handoff_abandons_queued_old_activity_dependent() -> None:
    target_entered = asyncio.Event()
    dependent_terminal = asyncio.Event()
    dependent_history_error = asyncio.Event()
    dependent_started = asyncio.Event()
    terminal_events: list[ToolCallEnded] = []

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    target = Target(instructions="target")

    @function_tool(name="handoff")
    async def handoff(ctx: RunContext) -> Agent:
        return target

    @function_tool(name="old_activity_dependent", after=("handoff",))
    async def old_activity_dependent(ctx: RunContext) -> str:
        dependent_started.set()
        raise AssertionError("old activity dependent must not run after permanent handoff")

    session = await _start(
        Agent(instructions="root", tools=[handoff, old_activity_dependent]),
        [
            _response(
                "handoff",
                FunctionToolCall(
                    name="old_activity_dependent", arguments="{}", call_id="dependent"
                ),
                FunctionToolCall(name="handoff", arguments="{}", call_id="handoff"),
            )
        ],
    )

    history_insert = session.history.insert

    def observe_history_insert(items: Any) -> None:
        history_insert(items)
        if any(
            item.type == "function_call_output"
            and item.call_id == "dependent_final"
            and item.is_error
            for item in session.history.items
        ):
            dependent_history_error.set()

    session.history.insert = observe_history_insert

    def on_tool_update(event: ToolExecutionUpdatedEvent) -> None:
        if isinstance(event.update, ToolCallEnded) and event.update.call_id == "dependent":
            terminal_events.append(event.update)
            if event.update.status in {"error", "cancelled"}:
                dependent_terminal.set()

    session.on("tool_execution_updated", on_tool_update)
    try:
        session.generate_reply(user_input="handoff")
        await asyncio.wait_for(target_entered.wait(), timeout=5)
        started_wait = asyncio.create_task(dependent_started.wait())
        terminal_wait = asyncio.create_task(dependent_terminal.wait())
        try:
            done, _ = await asyncio.wait(
                {started_wait, terminal_wait}, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            assert done, "queued dependent neither started nor reached a terminal error"
        finally:
            started_wait.cancel()
            terminal_wait.cancel()
            await asyncio.gather(started_wait, terminal_wait, return_exceptions=True)
        assert session.current_agent is target
        assert not dependent_started.is_set()
        await asyncio.wait_for(dependent_history_error.wait(), timeout=5)
        assert len(terminal_events) == 1
        assert terminal_events[0].status in {"error", "cancelled"}
        assert (
            sum(
                item.type == "function_call_output"
                and item.call_id == "dependent_final"
                and item.is_error
                for item in session.history.items
            )
            == 1
        )
    finally:
        await _close(session)


@pytest.mark.asyncio
async def test_dependency_gate_coordinates_session_and_activity_executors() -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    dependent_started = asyncio.Event()

    @function_tool(name="session_root")
    async def session_root(ctx: RunContext) -> str:
        root_started.set()
        await release_root.wait()
        return "root"

    session_tools = AsyncToolset(id="session", tools=[session_root])

    @function_tool(name="activity_dependent", after=("session_root",))
    async def activity_dependent(ctx: RunContext) -> str:
        dependent_started.set()
        return "dependent"

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                _response(
                    "cross-executor",
                    FunctionToolCall(
                        name="activity_dependent", arguments="{}", call_id="dependent"
                    ),
                    FunctionToolCall(name="session_root", arguments="{}", call_id="root"),
                )
            ]
        ),
        tools=[session_tools],
    )
    await session.start(Agent(instructions="workflow", tools=[activity_dependent]))
    try:
        session.generate_reply(user_input="cross-executor")
        await asyncio.wait_for(root_started.wait(), timeout=5)
        await asyncio.sleep(0)
        assert not dependent_started.is_set()
        release_root.set()
        await asyncio.wait_for(dependent_started.wait(), timeout=5)
    finally:
        release_root.set()
        await _close(session)


@pytest.mark.asyncio
async def test_handoff_abandons_activity_dependent_but_preserves_session_root() -> None:
    root_progress_observed = asyncio.Event()
    release_root = asyncio.Event()
    target_entered = asyncio.Event()
    dependent_started = asyncio.Event()
    dependent_terminal = asyncio.Event()
    root_final_committed = asyncio.Event()

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    target = Target(instructions="target")

    @function_tool(name="session_root", flags=ToolFlag.CANCELLABLE)
    async def session_root(ctx: RunContext) -> str:
        await ctx.update("session-root-progress")
        root_progress_observed.set()
        await release_root.wait()
        return "session-root-final"

    session_tools = AsyncToolset(id="session", tools=[session_root])

    @function_tool(name="activity_dependent", after=("session_root",))
    async def activity_dependent(ctx: RunContext) -> str:
        dependent_started.set()
        raise AssertionError("activity dependent must not run after old activity handoff")

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                _response(
                    "handoff-preserve",
                    FunctionToolCall(
                        name="activity_dependent", arguments="{}", call_id="dependent"
                    ),
                    FunctionToolCall(name="session_root", arguments="{}", call_id="root"),
                )
            ]
        ),
        tools=[session_tools],
    )
    await session.start(Agent(instructions="root", tools=[activity_dependent]))

    history_insert = session.history.insert

    def observe_history_insert(items: Any) -> None:
        history_insert(items)
        if any(
            item.type == "function_call_output"
            and item.call_id == "root_final"
            and item.output == "session-root-final"
            for item in session.history.items
        ):
            root_final_committed.set()

    session.history.insert = observe_history_insert

    def on_tool_update(event: ToolExecutionUpdatedEvent) -> None:
        update = event.update
        if isinstance(update, ToolCallUpdated) and update.call_id == "root":
            if update.message == "session-root-progress":
                root_progress_observed.set()
        elif isinstance(update, ToolCallEnded) and update.call_id == "dependent":
            if update.status in {"error", "cancelled"}:
                dependent_terminal.set()

    session.on("tool_execution_updated", on_tool_update)
    try:
        session.generate_reply(user_input="handoff-preserve")
        await asyncio.wait_for(root_progress_observed.wait(), timeout=5)
        session.update_agent(target)
        await asyncio.wait_for(target_entered.wait(), timeout=5)
        await asyncio.wait_for(dependent_terminal.wait(), timeout=5)
        assert not dependent_started.is_set()

        release_root.set()
        await asyncio.wait_for(root_final_committed.wait(), timeout=5)
        assert session.current_agent is target
        assert any(
            item.type == "function_call_output"
            and item.call_id == "root_final"
            and item.output == "session-root-final"
            for item in target.chat_ctx.items
        )
    finally:
        release_root.set()
        await _close(session)


@pytest.mark.asyncio
async def test_late_stream_call_after_handoff_is_terminally_rejected() -> None:
    handoff_terminal = asyncio.Event()
    target_entered = asyncio.Event()
    dependent_started = asyncio.Event()
    dependent_terminal = asyncio.Event()

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    target = Target(instructions="target")

    @function_tool(name="handoff")
    async def handoff(ctx: RunContext) -> Agent:
        return target

    @function_tool(name="late_dependent", after=("handoff",))
    async def late_dependent(ctx: RunContext) -> str:
        dependent_started.set()
        raise AssertionError("late dependent must not run after scheduler abandonment")

    llm = _DelayedBatchFakeLLM(
        first_call=FunctionToolCall(name="handoff", arguments="{}", call_id="handoff"),
        second_call=FunctionToolCall(
            name="late_dependent", arguments="{}", call_id="late-dependent"
        ),
    )
    session = AgentSession(
        llm=llm,
    )
    await session.start(Agent(instructions="root", tools=[handoff, late_dependent]))

    def on_tool_update(event: ToolExecutionUpdatedEvent) -> None:
        update = event.update
        if isinstance(update, ToolCallEnded) and update.call_id == "handoff":
            handoff_terminal.set()
        if isinstance(update, ToolCallEnded) and update.call_id == "late-dependent":
            if update.status in {"error", "cancelled"}:
                dependent_terminal.set()

    session.on("tool_execution_updated", on_tool_update)
    try:
        session.generate_reply(user_input="batch")
        await asyncio.wait_for(handoff_terminal.wait(), timeout=5)
        llm.emit_second.set()
        llm.close_stream.set()
        await asyncio.wait_for(llm.stream_eof.wait(), timeout=5)
        await asyncio.wait_for(target_entered.wait(), timeout=5)
        await asyncio.wait_for(dependent_terminal.wait(), timeout=5)
        assert not dependent_started.is_set()
    finally:
        llm.emit_second.set()
        llm.close_stream.set()
        await _close(session)


@pytest.mark.asyncio
async def test_cycle_rejection_does_not_strand_tool_dispatch() -> None:
    invoked: list[str] = []

    @function_tool(name="cycle_a", after=("cycle_b",))
    async def cycle_a(ctx: RunContext) -> str:
        invoked.append("a")
        return "a"

    @function_tool(name="cycle_b", after=("cycle_a",))
    async def cycle_b(ctx: RunContext) -> str:
        invoked.append("b")
        return "b"

    session = await _start(
        Agent(instructions="cycle", tools=[cycle_a, cycle_b]),
        [
            _response(
                "cycle",
                FunctionToolCall(name="cycle_a", arguments="{}", call_id="a"),
                FunctionToolCall(name="cycle_b", arguments="{}", call_id="b"),
            )
        ],
    )
    try:
        with pytest.raises(ValueError, match="cyclic tool dependency"):
            await asyncio.wait_for(session.run(user_input="cycle"), timeout=5)
        assert invoked == []
    finally:
        await _close(session)


def test_dependency_metadata_is_not_exposed_in_function_or_raw_schemas() -> None:
    @function_tool(name="root")
    async def root(ctx: RunContext) -> str:
        return "root"

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        return "dependent"

    @function_tool(
        name="raw_dependent",
        after=("root",),
        raw_schema={
            "name": "raw_dependent",
            "description": "raw dependent",
            "parameters": {"type": "object", "properties": {}},
        },
    )
    async def raw_dependent(ctx: RunContext, **kwargs: Any) -> str:
        return "raw"

    context = ToolContext([root, dependent, raw_dependent])
    assert dependent.info.after == ("root",)
    assert raw_dependent.info.after == ("root",)
    for schema in context.parse_function_tools("openai"):
        assert "after" not in str(schema)
    assert "after" not in str(context.parse_function_tools("google"))


def test_dependency_names_reject_a_single_string() -> None:
    with pytest.raises(TypeError, match="sequence of tool names"):

        @function_tool(after="root")  # type: ignore[arg-type]
        async def dependent(ctx: RunContext) -> str:
            return "dependent"
