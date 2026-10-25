"""A failed temporary realtime configuration must not affect later replies."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import NOT_GIVEN, Agent, NotGivenOr, function_tool, llm, utils
from livekit.agents.utils import is_given
from livekit.plugins.rime._realtime_fallback import RealtimeModelFallbackAdapter
from livekit.plugins.rime._realtime_session import AgentSession
from livekit.plugins.rime._realtime_types import GenerationCreatedEvent

from .fake_rime_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = pytest.mark.unit


@function_tool
async def lookup() -> str:
    """Look up a result."""
    return "result"


class _FailingSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.armed = False
        self.setup_failure = "tools"
        self.restore_failure: str | None = None
        self.setup_error = llm.RealtimeError("temporary configuration failed")
        self.restore_error = llm.RealtimeError("configuration restore failed")
        self.received_choices: list[llm.ToolChoice | None] = []
        self.received_tools: list[list[str]] = []
        self.restore_started = asyncio.Event()
        self.release_restore: asyncio.Event | None = None

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        if self.armed:
            self.received_tools.append([tool.id for tool in tools])
            if not tools and self.setup_failure == "tools":
                await super().update_tools(tools)
                raise self.setup_error
            if tools:
                self.restore_started.set()
                if self.release_restore is not None:
                    await self.release_restore.wait()
                if self.restore_failure == "tools":
                    raise self.restore_error
        await super().update_tools(tools)

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if self.armed and is_given(tool_choice):
            self.received_choices.append(tool_choice)
            if tool_choice == "required" and self.setup_failure == "tool_choice":
                super().update_options(tool_choice=tool_choice)
                raise self.setup_error
            if tool_choice is None and self.restore_failure == "tool_choice":
                raise self.restore_error
        super().update_options(tool_choice=tool_choice)


class _FailingModel(FakeRealtimeModel):
    def __init__(self) -> None:
        super().__init__(capabilities=fake_capabilities(per_response_tool_choice=False))

    def session(self, *, turn_detection_disabled: bool = False) -> _FailingSession:
        session = _FailingSession(self)
        self.created_sessions.append(session)
        return session


async def _wait_until(predicate) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


async def _finish_reply(session: AgentSession, provider: FakeRealtimeSession) -> None:
    speech = session.generate_reply()
    await _wait_until(lambda: provider._reply_futs)
    messages = utils.aio.Chan[llm.MessageGeneration]()
    functions = utils.aio.Chan[llm.FunctionCall]()
    messages.close()
    functions.close()
    provider._reply_futs[-1].set_result(
        GenerationCreatedEvent(
            message_stream=messages, function_stream=functions, user_initiated=True
        )
    )
    await asyncio.wait_for(speech.wait_for_playout(), 3)
    assert speech.exception() is None


@pytest.mark.parametrize("setup_failure", ["tools", "tool_choice"])
async def test_failed_setup_restores_configuration_before_the_next_reply(setup_failure):
    model = _FailingModel()
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test", tools=[lookup]))
        provider = model.active_session
        provider.armed = True
        provider.setup_failure = setup_failure
        errors = []
        session.on("error", errors.append)

        speech = session.generate_reply(tool_choice="required", tools=[])
        await asyncio.wait_for(speech.wait_for_playout(), 3)
        assert speech.exception() is provider.setup_error
        await _wait_until(lambda: provider.tool_choice is None)
        assert list(provider.tools.function_tools) == ["lookup"]
        assert provider.generate_reply_calls == 0
        assert not errors
        await _finish_reply(session, provider)


@pytest.mark.parametrize("restore_failure", ["tools", "tool_choice"])
async def test_failed_restore_closes_the_uncertain_session(restore_failure):
    model = _FailingModel()
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test", tools=[lookup]))
        provider = model.active_session
        provider.armed = True
        provider.restore_failure = restore_failure
        errors = []
        closed = asyncio.Event()
        session.on("error", errors.append)
        session.on("close", lambda _: closed.set())

        speech = session.generate_reply(tool_choice="required", tools=[])
        await asyncio.wait_for(closed.wait(), 3)
        assert speech.exception() is provider.setup_error
        assert len(errors) == 1
        assert errors[0].error.error is provider.restore_error
        assert errors[0].error.recoverable is False
        assert provider.received_choices == ["required", None]
        assert provider.received_tools == [[], ["lookup"]]
        assert provider.closed


async def test_failed_setup_stays_pending_until_restore_is_acknowledged():
    model = _FailingModel()
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test", tools=[lookup]))
        provider = model.active_session
        provider.armed = True
        provider.release_restore = asyncio.Event()
        speech = session.generate_reply(tool_choice="required", tools=[])
        try:
            await asyncio.wait_for(provider.restore_started.wait(), 3)
            assert not speech.done()
            assert not provider.tools.function_tools
        finally:
            provider.release_restore.set()
        await asyncio.wait_for(speech.wait_for_playout(), 3)
        assert speech.exception() is provider.setup_error
        assert list(provider.tools.function_tools) == ["lookup"]


async def test_failed_restore_switches_to_a_clean_fallback_session():
    primary = _FailingModel()
    backup = FakeRealtimeModel(capabilities=fake_capabilities(per_response_tool_choice=False))
    fallback = RealtimeModelFallbackAdapter([primary, backup], regenerate_on_swap=False)
    async with AgentSession(llm=fallback) as session:
        await session.start(Agent(instructions="test", tools=[lookup]))
        provider = primary.active_session
        provider.armed = True
        provider.restore_failure = "tools"
        errors = []
        session.on("error", errors.append)

        speech = session.generate_reply(tool_choice="required", tools=[])
        await asyncio.wait_for(speech.wait_for_playout(), 3)
        await _wait_until(
            lambda: (
                backup.created_sessions and "lookup" in backup.active_session.tools.function_tools
            )
        )
        assert speech.exception() is provider.setup_error
        assert provider.closed
        assert errors[0].error.error is provider.restore_error
        assert errors[0].error.recoverable is True
        await _finish_reply(session, backup.active_session)
