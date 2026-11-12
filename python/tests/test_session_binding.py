"""Explicit session bindings, detached reads and application object identity."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from livekit.agents import Agent, APIError, RunContext, SessionState, TextSession, persisted
from livekit.agents.storage import ContextStore, Storage, Thread

from .test_text_session import Reply, Script, call

pytestmark = pytest.mark.unit


@dataclass
class App:
    session: SessionState = persisted(default_factory=SessionState)
    count: int = persisted(default=0)


async def test_host_setup_failure_releases_the_native_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import livekit.agents.text_session as module

    store = ContextStore()
    app = await store.load_context(App())
    session = TextSession(agent=Agent(name="desk", instructions="Help", llm=Script()))

    def fail_setup(agent: object) -> list[object]:
        raise RuntimeError("host setup failed")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_agents", fail_setup)
        with pytest.raises(RuntimeError, match="host setup failed"):
            await session.run("start", context=store.bind(app))
    assert store._session_binding is None
    # A failed setup must not leave the native store claimed either.
    binding = store._bind_session(app)
    binding.close()


@pytest.mark.parametrize("fail", [False, True])
async def test_standalone_durable_binding_restores_and_saves_explicit_fields(
    thread_storage: Storage,
    fail: bool,
) -> None:
    info = await thread_storage.threads.create_thread()
    thread = Thread(thread_storage, info.thread_id, partition_key="desk")
    activation = await thread_storage.threads._core_context_store(thread.id, "desk")
    store = ContextStore(activation=activation, thread=thread, scope_id="root:desk")
    app = await store.load_context(App())
    model = Script(call("increment", {}, "c"), Reply("done", error=fail), Reply("again"))
    agent = Agent(name="desk", instructions="Help", llm=model)

    @agent.tool
    async def increment(ctx: RunContext[App]) -> str:
        """Save the application draft explicitly in this bound execution."""
        assert ctx.context is app
        app.count += 1
        await ctx.store.save_context(app)
        await ctx.store.flush()
        return "saved"

    if fail:
        with pytest.raises(APIError, match="scripted failure"):
            await TextSession(agent=agent).run("start", context=store.bind(app))
        first = await store.load_session()
    else:
        first = (await TextSession(agent=agent).run("start", context=store.bind(app))).state
    assert store._session_binding is None
    reloaded_activation = await thread_storage.threads._core_context_store(thread.id, "desk")
    reloaded = ContextStore(activation=reloaded_activation, thread=thread, scope_id="root:desk")
    restored = await reloaded.load_context(App())
    assert restored.count == (0 if fail else 1)
    assert restored.session.session_id == first.session_id
    second = await TextSession(agent=agent).run("continue", context=reloaded.bind(restored))
    assert second.state.session_id == first.session_id
    assert len(model.requests[-1].items) > len(model.requests[0].items)


async def test_binding_updates_the_original_app_but_detached_reads_do_not_follow() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    observed = app.session
    before = await store.load_session()
    declaration = store.bind(app)
    assert before is not app.session
    assert app.session.kind == "empty" and store._session_binding is None
    model = Script(Reply("first"), Reply("second"))
    session = TextSession(agent=Agent(name="desk", instructions="Help", llm=model))
    first = await session.run("hello", context=declaration)
    encoded = first.state._to_json()
    assert app.session is observed and observed.active_agent == "desk"
    assert store._session_binding is None
    second = await session.run("again", context=declaration)
    assert first.state._to_json() == encoded
    assert first.state.session_id == second.state.session_id
    assert len(model.requests[1].items) > len(model.requests[0].items)
    assert before.kind == "empty"
    after = await store.load_session()
    assert after is not app.session and after.session_id == app.session.session_id
    with pytest.raises(AttributeError):
        after.active_agent = "other"  # type: ignore[misc]


async def test_plain_context_does_not_advance_session_or_implicitly_load_storage() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    model = Script(Reply("one"), Reply("two"))
    session = TextSession(agent=Agent(name="desk", instructions="Help", llm=model))
    first = await session.run("same", context=app)
    second = await session.run("same", context=app)
    assert first.state.session_id != second.state.session_id
    assert app.session.kind == "empty" and (await store.load_session()).kind == "empty"
    assert len(model.requests[0].items) == len(model.requests[1].items)


async def test_bound_tools_receive_original_context_with_current_session_state() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    agent = Agent(
        name="desk", instructions="Help", llm=Script(call("inspect", {}, "c"), Reply("done"))
    )

    @agent.tool
    async def inspect(ctx: RunContext[App]) -> str:
        assert ctx.context is app
        assert app.session.active_agent == "desk"
        app.count += 1
        await ctx.store.save_context(app)
        await ctx.store.flush()
        return "seen"

    await TextSession(agent=agent).run("start", context=store.bind(app))
    assert app.count == 1
    # A conversation flush does not acknowledge ordinary application writes.
    assert store._scope_writes


async def test_unhydrated_and_overlapping_bindings_fail_before_execution() -> None:
    store = ContextStore()
    app = App()
    session = TextSession(agent=Agent(name="desk", instructions="Help", llm=Script()))
    with pytest.raises(ValueError, match="hydrate"):
        await session.run("hello", context=store.bind(app))
    await store.load_context(app)
    blocked = asyncio.Event()
    session = TextSession(
        agent=Agent(name="desk", instructions="Help", llm=Script(Reply(blocked=blocked)))
    )
    async with session.run_stream("hello", context=store.bind(app)):
        await asyncio.wait_for(blocked.wait(), 5)
        with pytest.raises(ValueError, match="active session binding"):
            await session.run("again", context=store.bind(app))
    assert store._session_binding is None


@pytest.mark.parametrize("with_context", [False, True])
async def test_binding_without_a_session_field_keeps_state_out_of_application_data(
    with_context: bool,
) -> None:
    store = ContextStore()
    app = object() if with_context else None
    model = Script(Reply("one"), Reply("two"))
    session = TextSession(agent=Agent(name="desk", instructions="Help", llm=model))
    first = await session.run("one", context=store.bind(app) if with_context else store.bind())
    second = await session.run("two", context=store.bind(app) if with_context else store.bind())
    assert first.state.session_id == second.state.session_id
    assert not hasattr(app, "session")


async def test_a_snapshot_with_an_unknown_tool_outcome_is_not_replayed() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    entered = asyncio.Event()
    agent = Agent(name="desk", instructions="Help", llm=Script(call("external", {}, "external")))

    @agent.tool
    async def external() -> str:
        """An external effect whose completion has not been recorded."""
        entered.set()
        await asyncio.Future()
        return "unreachable"

    async with TextSession(agent=agent).run_stream("start", context=store.bind(app)):
        await asyncio.wait_for(entered.wait(), 5)
        unknown = await store.load_session()
    with pytest.raises(ValueError, match="reconcile"):
        await TextSession(agent=agent).run("continue", state=unknown)
