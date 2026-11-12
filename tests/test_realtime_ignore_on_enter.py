"""IGNORE_ON_ENTER on a realtime model must not swap the session tools on every on_enter reply
when the model can't update tools mid-session (#6812).

Nova Sonic (and Gemini) declare ``mutable_tools=False`` and no per-response tools: hiding a
flagged tool there means restarting the provider session, which drops the reply in flight, so
every ``generate_reply`` after the first in ``on_enter`` went silent.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import NOT_GIVEN, Agent, AgentSession, NotGivenOr, utils
from livekit.agents.llm import (
    FunctionCall,
    GenerationCreatedEvent,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModelFallbackAdapter,
    ToolFlag,
    function_tool,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice
from livekit.agents.voice import SpeechHandle

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class _RecordingSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.tool_updates: list[set[str]] = []
        self.reply_tools: list[NotGivenOr[set[str]]] = []

    async def update_tools(self, tools: list[Tool]) -> None:
        self.tool_updates.append({t.info.name for t in tools})
        await super().update_tools(tools)

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        self.reply_tools.append(
            {t.info.name for t in tools} if utils.is_given(tools) else NOT_GIVEN
        )
        return super().generate_reply(
            instructions=instructions, tool_choice=tool_choice, tools=tools
        )


class _RecordingModel(FakeRealtimeModel):
    def session(self, *, turn_detection_disabled: bool = False) -> _RecordingSession:
        sess = _RecordingSession(self)
        self.created_sessions.append(sess)
        return sess


@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
async def flagged() -> str:
    """Only call this when the user asks for it."""
    return "ok"


@function_tool
async def visible() -> str:
    """Always available."""
    return "ok"


class _GreetingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test", tools=[flagged, visible])
        self.handles: list[SpeechHandle] = []
        self.entered = asyncio.Event()

    async def on_enter(self) -> None:
        self.handles.append(self.session.generate_reply(instructions="say hello"))
        self.handles.append(self.session.generate_reply(instructions="introduce yourself"))
        self.entered.set()


def _empty_generation() -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    message_ch.close()
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch, function_stream=function_ch, user_initiated=True
    )


async def _wait_for_reply(rt_session: _RecordingSession, index: int) -> None:
    # sleep with a real delay so virtual time can advance and the caller's timeout fires
    while len(rt_session._reply_futs) <= index:
        await asyncio.sleep(0.01)


async def _run_on_enter(
    capabilities: RealtimeCapabilities,
    *,
    fallback_capabilities: RealtimeCapabilities | None = None,
    nested: bool = False,
) -> _RecordingSession:
    model = _RecordingModel(capabilities=capabilities)
    llm: _RecordingModel | RealtimeModelFallbackAdapter = model
    if fallback_capabilities is not None:
        llm = RealtimeModelFallbackAdapter(
            [model, _RecordingModel(capabilities=fallback_capabilities)]
        )
        if nested:
            # the inner adapter is itself the primary of an outer adapter
            llm = RealtimeModelFallbackAdapter(
                [llm, _RecordingModel(capabilities=fallback_capabilities)]
            )
    agent = _GreetingAgent()

    async with AgentSession(llm=llm) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(agent)
        rt_session = model.created_sessions[-1]
        assert isinstance(rt_session, _RecordingSession)
        # drop the bring-up update, only the on_enter replies matter here
        rt_session.tool_updates.clear()

        await asyncio.wait_for(agent.entered.wait(), timeout=5)
        for i, handle in enumerate(agent.handles):
            await asyncio.wait_for(_wait_for_reply(rt_session, i), timeout=5)
            rt_session._reply_futs[i].set_result(_empty_generation())
            await asyncio.wait_for(handle.wait_for_playout(), timeout=5)

    assert rt_session.generate_reply_calls == 2
    return rt_session


async def test_immutable_tools_are_not_swapped_during_on_enter() -> None:
    rt_session = await _run_on_enter(
        fake_capabilities(mutable_tools=False, per_response_tool_choice=False)
    )

    # swapping tools here would restart the provider session and drop the pending reply
    assert rt_session.tool_updates == []
    assert rt_session.reply_tools == [NOT_GIVEN, NOT_GIVEN]
    assert set(rt_session.tools.function_tools) == {"flagged", "visible"}


async def test_mutable_tools_hide_flagged_tools_during_on_enter() -> None:
    rt_session = await _run_on_enter(
        fake_capabilities(mutable_tools=True, per_response_tool_choice=False)
    )

    # each on_enter reply hides the flagged tool, then restores the full set
    assert rt_session.tool_updates == [{"visible"}, {"flagged", "visible"}] * 2
    assert set(rt_session.tools.function_tools) == {"flagged", "visible"}


async def test_per_response_tools_hide_flagged_tools_during_on_enter() -> None:
    rt_session = await _run_on_enter(
        fake_capabilities(mutable_tools=False, per_response_tool_choice=True)
    )

    assert rt_session.tool_updates == []
    assert rt_session.reply_tools == [{"visible"}, {"visible"}]


async def test_fallback_adapter_uses_active_mutable_model() -> None:
    # the adapter ANDs capabilities across children; the active primary can still update tools
    rt_session = await _run_on_enter(
        fake_capabilities(mutable_tools=True, per_response_tool_choice=False),
        fallback_capabilities=fake_capabilities(
            mutable_tools=False, per_response_tool_choice=False
        ),
    )

    assert rt_session.tool_updates == [{"visible"}, {"flagged", "visible"}] * 2
    assert set(rt_session.tools.function_tools) == {"flagged", "visible"}


async def test_fallback_adapter_uses_active_per_response_model() -> None:
    rt_session = await _run_on_enter(
        fake_capabilities(mutable_tools=True, per_response_tool_choice=True),
        fallback_capabilities=fake_capabilities(
            mutable_tools=False, per_response_tool_choice=False
        ),
    )

    assert rt_session.tool_updates == []
    assert rt_session.reply_tools == [{"visible"}, {"visible"}]


async def test_nested_fallback_adapter_uses_active_mutable_model() -> None:
    # outer adapter -> inner adapter -> capable primary: caps must follow the active leaf
    rt_session = await _run_on_enter(
        fake_capabilities(mutable_tools=True, per_response_tool_choice=False),
        fallback_capabilities=fake_capabilities(
            mutable_tools=False, per_response_tool_choice=False
        ),
        nested=True,
    )

    assert rt_session.tool_updates == [{"visible"}, {"flagged", "visible"}] * 2
    assert set(rt_session.tools.function_tools) == {"flagged", "visible"}
