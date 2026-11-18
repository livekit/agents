"""Generated state views expose conversation ownership without leaking private routing fields."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from livekit.agents import (
    Agent,
    APIError,
    Completed,
    DelegationOptions,
    InputRequired,
    RunContext,
    SessionState,
    TextSession,
    VoiceSession,
    VoiceTask,
    persisted,
)
from livekit.agents.llm import ChatContext
from livekit.agents.storage import ContextStore, SQLiteStorage, Thread
from livekit.agents.voice.events import CloseEvent, event_dict
from tests.fakes import FakeLLM, tool_call, until

pytestmark = pytest.mark.unit


@dataclass
class App:
    session: SessionState = persisted(default_factory=SessionState)


async def test_agent_views_separate_private_tool_history_from_delivered_dialogue() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    agent = Agent(
        name="desk", instructions="Help.", llm=FakeLLM([tool_call("lookup"), "Found it."])
    )

    @agent.tool
    async def lookup() -> str:
        """Look up a record."""
        return "private record"

    await TextSession(agent=agent).run("Find my booking.", context=store.bind(app))
    current, root = app.session.current_agent, app.session.root_agent
    assert current is not None and root is not None
    assert isinstance(current.chat_ctx, ChatContext)
    assert current.agent_name == root.agent_name == "desk"
    assert current.completed and current.output == "Found it." and current.error is None
    assert any(item.type == "function_call_output" for item in current.chat_ctx.items)
    assert [item.text_content for item in app.session.chat_history.items] == [
        "Find my booking.",
        "Found it.",
    ]
    assert current.calls == {}


async def test_a_childs_call_ref_and_pending_arguments_are_owned_by_its_parent() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    child = Agent(
        name="booking",
        instructions="Book.",
        output_type=str,
        llm=FakeLLM([tool_call("date", call_id="ask"), "reserved"]),
    )

    @child.tool
    async def date() -> InputRequired:
        """Ask for the date."""
        return InputRequired("Which date?", verbatim=True)

    root = Agent(
        name="desk",
        instructions="Help.",
        tools=[child.as_tool()],
        llm=FakeLLM([tool_call("booking", call_id="parent"), "done"]),
    )
    session = TextSession(
        agent=root,
        delegation=DelegationOptions(
            routing_model=FakeLLM(
                [
                    tool_call("route_input_response", candidates=[0], unclear=False),
                ]
            )
        ),
    )
    await session.run("Book a room.", context=store.bind(app))
    detached = app.session.snapshot()
    current, parent = app.session.current_agent, app.session.root_agent
    assert current is not None and parent is not None and current.origin is not None
    call = parent.calls["parent"]
    assert call.origin == current.origin
    assert call.parent_tool_name == "booking"
    assert call.validated_arguments == {}
    assert call.status == "running" and call.agent_id is not None
    assert current.agent_name == "booking" and parent.agent_name == "desk"
    assert next(iter(current.input_requests.values())).source_message.text_content == "Which date?"

    await session.run("Friday", context=store.bind(app))
    assert app.session.current_agent is not None
    assert app.session.current_agent.agent_name == "desk"
    assert app.session.current_agent.calls == {}
    assert detached.current_agent is not None
    assert detached.current_agent.agent_name == "booking"
    assert any(item.text_content == "Friday" for item in app.session.chat_history.items)


async def test_bound_agent_runs_and_text_sessions_continue_the_same_native_conversation() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    model = FakeLLM(["first", "second", "third"])
    agent = Agent(name="desk", instructions="Help.", llm=model)
    first = await agent.run("one", context=store.bind(app))
    snapshot = app.session.snapshot()
    second = await TextSession(agent=agent).run("two", context=store.bind(app))
    async with agent.run_stream("three", context=store.bind(app)) as stream:
        said = [text async for text in stream.stream_text(delta=True)]
        third = await stream.result()
    assert first.output == "first" and second.output == "second" and third.output == "third"
    assert "".join(said) == "third"
    assert snapshot.session_id == app.session.session_id
    assert len(model.requests) == 3
    assert any(
        item.type == "message" and item.text_content == "one" for item in model.requests[-1].items
    )
    with pytest.raises(ValueError, match="explicit history"):
        await agent.run("conflict", context=store.bind(app), message_history=[])


async def test_voice_restores_the_bound_agents_private_context_after_text() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    model = FakeLLM(["first", "second"])
    agent = Agent(name="desk", instructions="Help.", llm=model)
    await agent.run("Remember Alice.", context=store.bind(app))
    identity = app.session.session_id
    async with VoiceSession(
        context=store.bind(app), vad=None, turn_handling={"turn_detection": "manual"}
    ) as voice:
        await voice.start(agent)
        await voice.generate_reply(user_input="continue")
        await until(
            lambda: any(item.text_content == "second" for item in app.session.chat_history.items),
            "voice output did not reach the bound conversation",
        )
        assert app.session.session_id == identity
        assert any(
            item.type == "message" and item.text_content == "Remember Alice."
            for item in model.requests[-1].items
        )


async def test_a_failed_bound_run_retains_its_diagnostic_without_a_successful_output() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    agent = Agent(
        name="desk", instructions="Help.", llm=FakeLLM([RuntimeError("model unavailable")])
    )
    with pytest.raises(Exception, match="model unavailable"):
        await agent.run("hello", context=store.bind(app), retries=0)
    current = app.session.current_agent
    assert current is not None and current.error is not None
    assert "model unavailable" in current.error
    assert not current.completed and current.output is None


@pytest.mark.parametrize("start_with_child", [False, True])
async def test_voice_refuses_saved_local_agent_work_without_changing_it(
    start_with_child: bool,
) -> None:
    store = ContextStore()
    app = await store.load_context(App())
    child = Agent(
        name="booking",
        instructions="Book.",
        output_type=str,
        llm=FakeLLM([tool_call("date"), "reserved"]),
    )

    @child.tool
    async def date() -> InputRequired:
        """Ask for the date."""
        return InputRequired("Which date?", verbatim=True)

    root = Agent(
        name="desk",
        instructions="Help.",
        tools=[child.as_tool()],
        llm=FakeLLM([tool_call("booking"), "Done."]),
    )
    await TextSession(agent=root).run("Book", context=store.bind(app))
    before = app.session._to_json()
    async with VoiceSession(context=store.bind(app), vad=None) as voice:
        with pytest.raises(ValueError, match="local agent.*TextSession"):
            await voice.start(child if start_with_child else root)
    assert app.session._to_json() == before, "voice startup cannot reset a saved text continuation"


@pytest.mark.parametrize("root_question", [False, True])
async def test_voice_refuses_text_local_work_loaded_between_speeches(
    tmp_path: Path,
    root_question: bool,
) -> None:
    storage = SQLiteStorage(directory=tmp_path)
    thread_id = (await storage.threads.create_thread()).thread_id
    thread = Thread(storage, thread_id, partition_key="desk")
    activation = await storage.threads._core_context_store(thread_id, "desk")
    store = ContextStore(activation=activation, scope_id="root:desk", thread=thread)
    other_activation = await storage.threads._core_context_store(thread_id, "desk")
    text_store = ContextStore(activation=other_activation, scope_id="root:desk", thread=thread)
    child = Agent(
        name="booking", instructions="Book.", output_type=str, llm=FakeLLM([tool_call("date")])
    )

    @child.tool
    async def date() -> InputRequired:
        """Ask for the date."""
        return InputRequired("Which date?", verbatim=True)

    model = FakeLLM(["Hello.", tool_call("date" if root_question else "booking"), "Done."])
    root = Agent(name="desk", instructions="Help.", tools=[child.as_tool()], llm=model)
    if root_question:

        @root.tool(name="date")
        async def root_date() -> InputRequired:
            """Ask for the date without a local child."""
            return InputRequired("Which date?", verbatim=True)

    closes: list[CloseEvent] = []
    try:
        app = await store.load_context(App())
        async with VoiceSession(context=store.bind(app), vad=None) as voice:
            voice.on("close", closes.append)
            await voice.start(root)
            await voice.generate_reply(user_input="Hello")
            text_app = await text_store.load_context(App())
            result = await TextSession(agent=root).run("Book", context=text_store.bind(text_app))
            assert result.status == "input_required"
            before = text_app.session._to_json()
            await voice.generate_reply(user_input="Continue by voice")
            await until(lambda: closes, "voice did not refuse the saved text-local continuation")
            assert closes[0].reason.value == "error" and isinstance(closes[0].error, APIError)
            assert isinstance(event_dict(closes[0], json=True)["error"], str)
            assert len(model.requests) == 2, "voice must refuse before starting another model call"
        saved = await text_store.load_session()
        assert saved._to_json() == before, "voice cannot rewrite the text-local agent tree"
    finally:
        store._close()
        text_store._close()
        await storage.aclose()


@pytest.mark.parametrize("persist", [False, True])
async def test_voice_close_preserves_the_applications_context_store_for_a_text_turn(
    tmp_path: Path, persist: bool
) -> None:
    storage = SQLiteStorage(directory=tmp_path) if persist else None
    thread = None
    if storage is not None:
        thread_id = (await storage.threads.create_thread()).thread_id
        thread = Thread(storage, thread_id, partition_key="desk")
        activation = await storage.threads._core_context_store(thread_id, "desk")
        store = ContextStore(activation=activation, scope_id="root:desk", thread=thread)
    else:
        store = ContextStore()
    try:
        app = await store.load_context(App())
        model = FakeLLM(["Hello.", "Back on text."])
        agent = Agent(name="desk", instructions="Help.", llm=model)
        async with VoiceSession(context=store.bind(app), thread=thread, vad=None) as voice:
            await voice.start(agent)
            await voice.generate_reply(user_input="Hi")
        result = await agent.run("Continue", context=store.bind(app))
        assert result.output == "Back on text."
        assert any(item.text_content == "Hello." for item in app.session.chat_history.items)
        if storage is not None:
            saved = await store.load_session()
            assert [item.text_content for item in saved.chat_history.items] == [
                "Hi",
                "Hello.",
                "Continue",
                "Back on text.",
            ]
    finally:
        store._close()
        if storage is not None:
            await storage.aclose()


async def test_a_voice_handoff_does_not_replace_the_durable_root_agent() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    root_model = FakeLLM(["Back on text."])
    root = Agent(name="desk", instructions="Help.", llm=root_model)
    temporary = Agent(
        name="temporary", instructions="Help temporarily.", llm=FakeLLM(["Temporary reply."])
    )
    entered = asyncio.Event()

    @temporary.on_enter
    async def active(session: VoiceSession[App]) -> None:
        entered.set()

    async with VoiceSession(context=store.bind(app), vad=None) as voice:
        await voice.start(root)
        voice.update_agent(temporary)
        await until(entered.is_set, "the temporary voice role did not enter")
        await voice.generate_reply(user_input="Hello")
        assert app.session.root_agent is not None
        assert app.session.root_agent.agent_name == "desk", (
            "a live role is not a saved local delegation"
        )
    result = await root.run("Continue", context=store.bind(app))
    assert result.output == "Back on text."
    assert any(
        item.type == "message" and item.text_content == "Temporary reply."
        for item in root_model.requests[-1].items
    )


async def test_a_voice_task_does_not_replace_the_durable_root_agent() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    root = Agent(
        name="desk", instructions="Help.", llm=FakeLLM([tool_call("collect"), "Found it."])
    )
    collector = Agent(
        name="collector",
        instructions="Collect.",
        output_type=str,
        llm=FakeLLM([tool_call("finish")]),
    )
    roots_during_task: list[str] = []

    @collector.tool
    async def finish() -> Completed[str]:
        """Return the collected reference."""
        assert app.session.root_agent is not None
        roots_during_task.append(app.session.root_agent.agent_name)
        return Completed("ref-7")

    @root.tool
    async def collect(ctx: RunContext[App]) -> str:
        """Collect a reference during the live conversation."""
        return await VoiceTask(agent=collector).run()

    async with VoiceSession(context=store.bind(app), vad=None) as voice:
        await voice.start(root)
        await voice.generate_reply(user_input="Find the reference")
        assert app.session.root_agent is not None
        assert app.session.root_agent.agent_name == "desk"
        assert app.session.root_agent.calls == {} and len(app.session.agents) == 1
    assert roots_during_task == ["desk"]
