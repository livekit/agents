from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, function_tool, utils
from livekit.agents.llm import FunctionCall, GenerationCreatedEvent, MessageGeneration

from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _tool_generation(response_id: str, calls: list[FunctionCall]) -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    message_ch.close()
    function_ch = utils.aio.Chan[FunctionCall]()
    for call in calls:
        function_ch.send_nowait(call)
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=False,
        response_id=response_id,
    )


class _Dialog(AgentTask[str]):
    def __init__(self, name: str) -> None:
        super().__init__(instructions=f"collect the {name}")
        self._name = name

    async def on_enter(self) -> None:
        self.complete(self._name)


class _Parent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="book a room")
        self.outcomes: list[str] = []

    @function_tool
    async def open_name_dialog(self) -> str:
        """Collect the name."""
        self.outcomes.append(await _Dialog("name"))
        return "name captured"

    @function_tool
    async def open_email_dialog(self) -> str:
        """Collect the email."""
        self.outcomes.append(await _Dialog("email"))
        return "email captured"

    @function_tool
    async def open_phone_dialog(self) -> str:
        """Collect the phone."""
        self.outcomes.append(await _Dialog("phone"))
        return "phone captured"


@pytest.mark.parametrize("dialogs", [("name", "email"), ("name", "email", "phone")])
@pytest.mark.parametrize("separate_generations", [False, True])
async def test_parallel_inline_tasks_from_a_realtime_model_run_in_turn(
    separate_generations: bool, dialogs: tuple[str, ...]
) -> None:
    """A realtime model can deliver a turn's parallel tool calls each in a generation of its
    own, so every call runs under its own speech; their inline tasks still queue and all run."""
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    agent = _Parent()
    calls = [
        FunctionCall(call_id=f"call_{n}", name=f"open_{n}_dialog", arguments="{}") for n in dialogs
    ]
    session = AgentSession(llm=model)
    try:
        await session.start(agent)
        rt = model.active_session
        if separate_generations:
            for i, call in enumerate(calls):
                rt.emit("generation_created", _tool_generation(f"gen_{i}", [call]))
        else:
            rt.emit("generation_created", _tool_generation("gen", calls))
        for _ in range(500):
            if len(agent.outcomes) == len(dialogs):
                break
            await asyncio.sleep(0.01)
        assert sorted(agent.outcomes) == sorted(dialogs)
    finally:
        # a hung inline task leaves its function call unfinished and the close waiting on it
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(session.aclose(), timeout=10.0)
