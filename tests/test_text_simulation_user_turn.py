"""A text simulation runs on_user_turn_completed for each simulated user turn, as a live call
does. The simulator drives each turn through ``run_input`` on the session host."""

from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, StopResponse
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.remote_session import RemoteSession, SessionHost

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


@pytest.fixture
def text_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AgentSession, "_text_only", property(lambda self: True))


@pytest.mark.usefixtures("text_simulation")
async def test_hook_edits_reach_the_llm_in_a_text_simulation() -> None:
    agent = HookAgent(edit_to="edited")
    replies = await _run_input(agent, "hello there")

    assert agent.turns == ["hello there"]
    assert replies == ["edited"]


@pytest.mark.usefixtures("text_simulation")
async def test_stop_response_is_a_silent_turn_in_a_text_simulation() -> None:
    agent = HookAgent(stop=True)
    replies = await _run_input(agent, "hello there")

    assert agent.turns == ["hello there"]
    assert replies == []


async def test_run_input_outside_a_text_simulation_skips_the_hook() -> None:
    agent = HookAgent(edit_to="edited")
    replies = await _run_input(agent, "hello there")

    assert agent.turns == []
    assert replies == ["unedited"]
