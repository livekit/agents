"""Child application contexts are constructed and restored in their own scopes."""

from dataclasses import dataclass

import pytest

from livekit.agents import Agent, RunContext, TextSession, TextSessionState, persisted
from livekit.agents.storage.context_store import ContextStore
from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

from .test_text_session import Reply, Script, call

pytestmark = pytest.mark.unit


@dataclass
class Draft:
    service: object
    value: int = persisted(default=0)


async def test_child_contexts_restore_their_own_drafts_without_serializing_services() -> None:
    service = object()
    root_context = Draft(service, 99)
    constructed: list[Draft] = []
    scopes: list[str] = []

    def accepted(ctx: RunContext[Draft]) -> int:
        return ctx.context.value

    child = Agent(
        name="child",
        instructions="Collect a number.",
        output_type=accepted,
        llm=Script(
            Reply("Which number?"),
            call("final_result", {}, "finish-1"),
            call("final_result", {}, "finish-2"),
        ),
    )
    parent = Agent(
        name="parent",
        instructions="Ask the child twice.",
        tools=[child.as_tool(tool_name="child")],
        llm=Script(call("child", {}, "first"), call("child", {}, "second"), Reply("Done.")),
    )

    @child.instructions
    async def edit_draft(ctx: RunContext[Draft]) -> str:
        assert ctx.context.service is service
        ctx.context.value += 1
        await ctx.store.save_context(ctx.context)
        return "Continue."

    async def construct(agent_name: str, store: ContextStore) -> Draft:
        assert agent_name == child.name
        draft = Draft(service)
        await store.load_context(draft)
        constructed.append(draft)
        scopes.append(store._scope_id)
        return draft

    store = ContextStore()
    token = set_run_hooks(RunHooks(store=store))
    try:
        first = await TextSession(agent=parent, context_factory=construct).run(
            "start", context=root_context
        )
    finally:
        reset_run_hooks(token)
    assert first.result.status == "input_required"
    child_frame = first.state.frames[first.state.active_frame_id]
    assert child_frame.context_scope != "root"
    writes = [write.scope for write in store._scope_writes]
    restored = TextSessionState._from_json(first.state._to_json())
    store._close()
    store = ContextStore(scopes=writes)
    token = set_run_hooks(RunHooks(store=store))
    try:
        with pytest.raises(ValueError, match="unknown context scope"):
            await TextSession(agent=parent).run("two", state=restored, context=root_context)
        second = await TextSession(agent=parent, context_factory=construct).run(
            "two", state=restored, context=root_context
        )
    finally:
        reset_run_hooks(token)
        store._close()
    assert second.result.status == "completed"
    assert scopes[0] == scopes[1] and scopes[2] != scopes[0]
    assert [draft.value for draft in constructed] == [1, 2, 1]
    assert root_context.value == 99
    assert len({id(draft) for draft in constructed}) == 3


async def test_cancelling_child_context_construction_settles_before_running_the_model() -> None:
    import asyncio

    started = asyncio.Event()
    stopped = asyncio.Event()
    child_model = Script()
    child = Agent(name="child", instructions="Wait", output_type=str, llm=child_model)
    parent = Agent(
        name="parent",
        instructions="Delegate",
        tools=[child.as_tool(tool_name="child")],
        llm=Script(call("child", {}, "delegate")),
    )

    async def construct(agent_name: str, store: ContextStore) -> object:
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    async with TextSession(agent=parent, context_factory=construct).run_stream("start") as stream:
        consumer = asyncio.create_task(stream.result())
        await asyncio.wait_for(started.wait(), timeout=5)
        stream.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    assert stopped.is_set()
    assert child_model.requests == []


async def test_only_the_owner_closes_the_store_shared_with_child_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[ContextStore] = []
    original_close = ContextStore._core_release_hydration

    def close(store: ContextStore) -> None:
        assert store._activation.closed
        closed.append(store)
        original_close(store)

    monkeypatch.setattr(ContextStore, "_core_release_hydration", close)
    child = Agent(name="child", instructions="Finish", output_type=str, llm=Script(Reply("OK")))
    parent = Agent(
        name="parent",
        instructions="Delegate",
        tools=[child.as_tool(tool_name="child")],
        llm=Script(call("child", {}, "delegate"), Reply("Done")),
    )
    result = await TextSession(agent=parent).run("start")
    assert result.result.status == "completed"
    assert len(closed) == 1
