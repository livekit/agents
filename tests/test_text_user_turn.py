"""Every text user turn runs on_user_turn_completed, as a spoken turn does: ``run()`` (which the
simulator drives through ``run_input`` on the session host) and the default text input callback."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, StopResponse
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.remote_session import RemoteSession, SessionHost
from livekit.agents.voice.room_io.types import TextInputEvent, _default_text_input_cb

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_run_input_errors import PairedTransport

pytestmark = pytest.mark.unit


class HookAgent(Agent):
    def __init__(self, *, edit_to: str | None = None, stop: bool = False) -> None:
        super().__init__(
            instructions="test agent",
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(input="hello there", content="unedited", ttft=0, duration=0),
                    FakeLLMResponse(input="edited", content="edited", ttft=0, duration=0),
                ]
            ),
        )
        self.turns: list[str] = []
        self._edit_to = edit_to
        self._stop = stop

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        self.turns.append(new_message.text_content or "")
        if self._stop:
            raise StopResponse()
        if self._edit_to is not None:
            new_message.content = [self._edit_to]


async def _run_input(agent: Agent, text: str) -> list[str]:
    host_transport, client_transport = PairedTransport.create_pair()
    session = AgentSession()
    host = SessionHost(host_transport)
    host.register_session(session)
    await session.start(agent=agent)
    await host.start()
    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.run(text, timeout=10.0)

    await client.aclose()
    await host.aclose()
    await session.aclose()
    return [item.message.content[0].text for item in resp.items if item.HasField("message")]


async def test_hook_edits_reach_the_llm() -> None:
    agent = HookAgent(edit_to="edited")
    replies = await _run_input(agent, "hello there")

    assert agent.turns == ["hello there"]
    assert replies == ["edited"]


async def test_stop_response_is_a_silent_turn() -> None:
    agent = HookAgent(stop=True)
    replies = await _run_input(agent, "hello there")

    assert agent.turns == ["hello there"]
    assert replies == []


async def test_default_text_input_runs_the_hook() -> None:
    agent = HookAgent(edit_to="edited")
    session = AgentSession()
    await session.start(agent=agent)
    try:
        await _default_text_input_cb(session, TextInputEvent(text="hello there"))
        await asyncio.wait_for(session.wait_for_idle(), timeout=10.0)
    finally:
        await session.aclose()

    assert agent.turns == ["hello there"]
    assert [m.text_content for m in agent.chat_ctx.messages() if m.role == "assistant"] == [
        "edited"
    ]


class NoLLMAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test agent")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        pass


async def test_a_reply_that_cannot_start_fails_the_run() -> None:
    session = AgentSession()
    await session.start(agent=NoLLMAgent())
    try:
        with pytest.raises(RuntimeError, match="without an LLM"):
            await asyncio.wait_for(session.run(user_input="hello there"), timeout=10.0)
    finally:
        await session.aclose()


class RecordingLLM(FakeLLM):
    def __init__(self, owner: str, asked: list[str]) -> None:
        super().__init__()
        self._owner = owner
        self._asked = asked

    def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        self._asked.append(self._owner)
        return super().chat(**kwargs)


class HandoffAgent(Agent):
    """Hands off from the hook. With ``wait``, the hook stays busy until the handoff lands;
    without it, the reply is decided while the transition is still pending."""

    def __init__(self, next_agent: Agent, asked: list[str], *, wait: bool) -> None:
        super().__init__(instructions="test agent", llm=RecordingLLM("old", asked))
        self._next_agent = next_agent
        self._wait = wait

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        self.session.update_agent(self._next_agent)
        if self._wait:
            await asyncio.sleep(0.5)


@pytest.mark.parametrize("wait", [True, False])
async def test_no_reply_after_the_hook_hands_off(wait: bool) -> None:
    # the turn's reply was built from the old agent's context: neither agent may answer it
    asked: list[str] = []
    next_agent = Agent(instructions="next agent", llm=RecordingLLM("next", asked))
    session = AgentSession()
    await session.start(agent=HandoffAgent(next_agent, asked, wait=wait))
    try:
        await asyncio.wait_for(session.run(user_input="hello there"), timeout=10.0)
    finally:
        await session.aclose()

    assert asked == []
    assert session.current_agent is next_agent
