"""A typed line sent over the session host (`lk console` text mode, text simulations)
stands in for a spoken turn, so it runs what a spoken turn runs: the final transcript
and `on_user_turn_completed` before the reply. An agent that hands work off from those
hooks then behaves the same typed as it does on a call."""

from __future__ import annotations

import pytest

from livekit.agents import llm
from livekit.agents.llm import StopResponse
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.events import UserInputTranscribedEvent
from livekit.agents.voice.remote_session import RemoteSession, SessionHost
from livekit.protocol.agent_pb import agent_session as agent_pb

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_run_input_errors import PairedTransport

pytestmark = pytest.mark.unit

CART_NOTE = "the order taker added: two calzones"


class OrderAgent(Agent):
    """Hands the order to a background taker from the user-turn hook, the way
    an agent does when a second model owns the cart."""

    def __init__(self, *, stop: bool = False) -> None:
        super().__init__(
            instructions="take the order",
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input=CART_NOTE, content="Two calzones. Anything else?", ttft=0, duration=0
                    )
                ]
            ),
        )
        self.turns: list[str] = []
        self._stop = stop

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.turns.append(new_message.text_content or "")
        if self._stop:
            raise StopResponse()
        turn_ctx.add_message(role="system", content=[CART_NOTE])


async def _run_typed(agent: OrderAgent, text: str) -> tuple[list, list[str]]:
    host_transport, client_transport = PairedTransport.create_pair()
    session = AgentSession()
    transcripts: list[str] = []

    @session.on("user_input_transcribed")
    def _on_transcript(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            transcripts.append(ev.transcript)

    host = SessionHost(host_transport)
    host.register_session(session)
    await session.start(agent=agent)
    await host.start()
    client = RemoteSession(client_transport)
    await client.start()
    try:
        resp = await client.run(text, timeout=10.0)
    finally:
        await client.aclose()
        await host.aclose()
        await session.aclose()
    return list(resp.items), transcripts


def _said(items: list) -> list[str]:
    return [
        "".join(c.text for c in i.message.content)
        for i in items
        if i.HasField("message") and i.message.role == agent_pb.ASSISTANT
    ]


@pytest.mark.asyncio
async def test_typed_turn_runs_the_user_turn_hooks() -> None:
    agent = OrderAgent()

    items, transcripts = await _run_typed(agent, "two calzones please")

    assert agent.turns == ["two calzones please"]
    assert transcripts == ["two calzones please"]
    # the reply was generated from the context the hook edited
    assert _said(items) == ["Two calzones. Anything else?"]
    # the turn is still the user's: their message lands in the history once
    history = [m for m in agent.chat_ctx.messages() if m.role == "user"]
    assert [m.text_content for m in history] == ["two calzones please"]


@pytest.mark.asyncio
async def test_typed_turn_stop_response_is_a_silent_turn() -> None:
    agent = OrderAgent(stop=True)

    items, _ = await _run_typed(agent, "two calzones please")

    assert agent.turns == ["two calzones please"]
    assert _said(items) == []
