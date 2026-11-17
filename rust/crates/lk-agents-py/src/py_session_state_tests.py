"""Generated state views expose conversation ownership without leaking private routing fields."""

from dataclasses import dataclass

import pytest

from livekit.agents import Agent, InputRequired, SessionState, TextSession, VoiceSession, persisted
from livekit.agents.llm import ChatContext
from livekit.agents.storage import ContextStore
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
    session = TextSession(agent=root)
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
