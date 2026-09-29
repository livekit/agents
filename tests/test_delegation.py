"""The voice side: a conversation that hands work to an expert over A2A."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, DirectiveReceivedEvent
from livekit.agents.a2a import TaskInput, TaskUpdate
from livekit.agents.delegation import DELEGATE_TOOL_NAME, A2ADelegate, Delegate
from livekit.agents.llm import FunctionCall, FunctionToolCall

from .fake_llm import FakeLLM
from .test_a2a_runner import _AnsweringLLM, _says
from .test_a2a_server import _drain_sse_watcher, _serving

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _voice_llm(*, instruction: str = "what is the fare") -> FakeLLM:
    return _AnsweringLLM(
        fake_responses=[
            _says(
                "how much is it",
                "one sec",
                calls=[
                    FunctionToolCall(
                        type="function",
                        name=DELEGATE_TOOL_NAME,
                        arguments=f'{{"task": "{instruction}"}}',
                        call_id="d1",
                    )
                ],
            )
        ],
        fallbacks=["Sure, let me check.", "It is 240 USD."],
    )


class _Scripted(Delegate):
    """A delegate that answers with the updates it was handed."""

    def __init__(self, *updates: TaskUpdate) -> None:
        self._updates = list(updates)

    def submit(self, task_input: TaskInput) -> Any:
        return _ScriptedStream(list(self._updates))


class _ScriptedStream:
    def __init__(self, updates: list[TaskUpdate]) -> None:
        self._updates = updates

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> TaskUpdate:
        if not self._updates:
            raise StopAsyncIteration
        return self._updates.pop(0)

    async def cancel(self, reason: str = "") -> None:
        return None

    async def aclose(self) -> None:
        return None


def _outputs(session: AgentSession) -> list[str]:
    return [
        item.output
        for item in session.history.items
        if item.type == "function_call_output" and item.output
    ]


def _answer(session: AgentSession) -> str | None:
    """What the delegate call returned, recorded under ``<call_id>_final``."""
    return next(
        (
            item.output
            for item in session.history.items
            if item.type == "function_call_output" and item.call_id == "d1_final"
        ),
        None,
    )


@contextlib.asynccontextmanager
async def _conversation(llm: FakeLLM, **session_kwargs: Any) -> Any:
    async with _serving() as served:
        delegate = A2ADelegate(f"{served.base_url}/fare-desk")
        session = AgentSession(llm=llm, delegate=delegate, **session_kwargs)
        await session.start(agent=Agent(instructions="voice"))
        try:
            yield session, served
        finally:
            await asyncio.wait_for(session.aclose(), timeout=10.0)
            await _drain_sse_watcher()


async def test_the_delegate_tool_is_offered_when_a_delegate_is_in_force() -> None:
    async with _conversation(_voice_llm()) as (session, _):
        activity = session.current_agent._get_activity_or_raise()

        def _delegate_tool() -> Any:
            return next(
                t
                for t in activity.tools
                if getattr(t, "info", None) and t.info.name == DELEGATE_TOOL_NAME
            )

        assert _delegate_tool() is not None
        # built once, so the schema the model sees keeps its identity across turns
        assert _delegate_tool() is _delegate_tool()


async def test_a_session_without_a_delegate_offers_no_such_tool() -> None:
    session = AgentSession(llm=_voice_llm())
    await session.start(agent=Agent(instructions="voice"))
    try:
        activity = session.current_agent._get_activity_or_raise()
        names = {t.info.name for t in activity.tools if hasattr(t, "info")}
        assert DELEGATE_TOOL_NAME not in names
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_progress_is_relayed_and_the_answer_is_the_tools_return() -> None:
    async with _conversation(_voice_llm()) as (session, _):
        session.generate_reply(user_input="how much is it")
        await asyncio.sleep(5)

        # the expert's report reached the conversation as the tool wrote it
        assert "checking the fare rules" in " ".join(_outputs(session))
        assert _answer(session) == "It is 240 USD."


async def test_a_directive_is_raised_on_the_session_after_the_answer() -> None:
    events: list[DirectiveReceivedEvent] = []

    async with _conversation(_voice_llm(instruction="that is all")) as (session, _):
        session.on("directive_received", events.append)
        session.generate_reply(user_input="how much is it")
        await asyncio.sleep(5)

        assert [(e.kind, e.reason, e.call_id) for e in events] == [
            ("end_session", "user_request", "d1")
        ]
        # advice rides with an answer rather than replacing it, and closes nothing here
        assert _answer(session)
        assert session.history.items


async def test_the_conversation_is_sent_without_its_calls() -> None:
    """The expert gets what was said, not the plumbing that said it."""
    async with _conversation(_voice_llm()) as (session, served):
        session.generate_reply(user_input="how much is it")
        await asyncio.sleep(5)

    (expert,) = served.sessions
    items = expert.current_agent.chat_ctx.items
    said = [item.text_content or "" for item in items if item.type == "message"]
    assert any("how much is it" in text for text in said), "the caller's turn reached the expert"
    # the delegate call and its synthetic progress entries are not conversation
    assert not [item for item in items if getattr(item, "name", "") == DELEGATE_TOOL_NAME]


async def test_a_failing_expert_raises_a_tool_error() -> None:
    class _Failing(Delegate):
        def submit(self, task_input: TaskInput) -> Any:
            return _FailingStream()

    class _FailingStream:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            return TaskUpdate(state="failed", text="the fare service is unreachable")

        async def cancel(self, reason: str = "") -> None:
            return None

        async def aclose(self) -> None:
            return None

    session = AgentSession(llm=_voice_llm(), delegate=_Failing())
    await session.start(agent=Agent(instructions="voice"))
    try:
        session.generate_reply(user_input="how much is it")
        await asyncio.sleep(5)
        # the error reaches the model as the tool's result, so it can say something useful
        assert any("unreachable" in output for output in _outputs(session))
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_the_session_closes_the_delegate_it_owns() -> None:
    closed: list[str] = []

    async with _serving() as served:
        delegate = A2ADelegate(f"{served.base_url}/fare-desk")
        original = delegate.aclose

        async def _record() -> None:
            closed.append("delegate")
            await original()

        delegate.aclose = _record  # type: ignore[method-assign]
        session = AgentSession(llm=_voice_llm(), delegate=delegate)
        await session.start(agent=Agent(instructions="voice"))
        await asyncio.wait_for(session.aclose(), timeout=10.0)
        await _drain_sse_watcher()

    assert closed == ["delegate"]


async def test_an_agent_overrides_the_sessions_settings_key_by_key() -> None:
    """What the agent sets wins; what it leaves alone stays as the session had it."""

    class _Marker(Delegate):
        def submit(self, task_input: TaskInput) -> Any:
            raise AssertionError("not reached")

    session_delegate = _Marker()
    session = AgentSession(
        llm=_voice_llm(),
        delegate={"delegate": session_delegate, "announce": False, "metadata": {"c": "42"}},
    )
    await session.start(agent=Agent(instructions="voice", delegate={"announce": True}))
    try:
        activity = session.current_agent._get_activity_or_raise()
        assert activity._delegation["announce"] is True
        # the keys the agent said nothing about are still the session's
        assert activity._delegation["delegate"] is session_delegate
        assert activity._delegation["metadata"] == {"c": "42"}
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_an_agents_delegate_overrides_the_sessions() -> None:
    class _Marker(Delegate):
        def submit(self, task_input: TaskInput) -> Any:
            raise AssertionError("not reached")

    agent_delegate = _Marker()
    session_delegate = _Marker()
    session = AgentSession(llm=_voice_llm(), delegate=session_delegate)
    await session.start(agent=Agent(instructions="voice", delegate=agent_delegate))
    try:
        activity = session.current_agent._get_activity_or_raise()
        assert activity._delegation["delegate"] is agent_delegate
        assert session.delegate is session_delegate
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_a_delegation_that_ends_without_a_state_is_a_failure() -> None:
    """Rule 1: a stream that ends without a terminal status failed, and the conversation
    has to hear that rather than a stray StopAsyncIteration."""

    class _Silent(Delegate):
        def submit(self, task_input: TaskInput) -> Any:
            return _SilentStream()

    class _SilentStream:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

        async def cancel(self, reason: str = "") -> None:
            return None

        async def aclose(self) -> None:
            return None

    session = AgentSession(llm=_voice_llm(), delegate=_Silent())
    await session.start(agent=Agent(instructions="voice"))
    try:
        session.generate_reply(user_input="how much is it")
        await asyncio.sleep(5)
        assert any("without an answer" in output for output in _outputs(session))
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_a_verbatim_answer_is_said_as_written_and_draws_no_reply() -> None:
    """Rule 6: said once, as written. The model is given it to know, not to phrase."""
    answer = "Your confirmation code is AB12."
    delegate = _Scripted(TaskUpdate(state="completed", text=answer, verbatim=True))
    # announce off, so the dispatch note is silent and nothing but the answer is spoken
    session = AgentSession(llm=_voice_llm(), delegate={"delegate": delegate, "announce": False})
    await session.start(agent=Agent(instructions="voice"))
    try:
        session.generate_reply(user_input="how much is it")
        await asyncio.sleep(5)

        said = [
            item.text_content
            for item in session.history.items
            if item.type == "message" and item.role == "assistant"
        ]
        assert said == ["one sec", answer]
        # recorded for the model, and with no result entry the executor files no reply
        assert any(answer in output for output in _outputs(session))
        assert _answer(session) is None
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_an_update_carrying_only_an_item_is_not_relayed() -> None:
    """A silent report reaches the caller to be rendered, not to be said."""
    report = FunctionCall(call_id="cf1_update_0", name="check_fares", arguments="", update_of="cf1")
    delegate = _Scripted(
        TaskUpdate(item=report),
        TaskUpdate(state="completed", text="It is 240 USD."),
    )
    session = AgentSession(llm=_voice_llm(), delegate={"delegate": delegate, "announce": False})
    await session.start(agent=Agent(instructions="voice"))
    try:
        session.generate_reply(user_input="how much is it")
        await asyncio.sleep(5)

        assert _answer(session) == "It is 240 USD."
        # the dispatch note and the answer, and nothing the item-only update put there
        assert len(_outputs(session)) == 2
    finally:
        await asyncio.wait_for(session.aclose(), timeout=10.0)
