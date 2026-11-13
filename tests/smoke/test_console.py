"""A console call: what the ``lk agent console`` CLI is sent."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, VoiceSession
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.room_io import RoomOptions
from livekit.protocol.agent_pb.agent_session import ASSISTANT, SessionRequest
from tests.fakes import FakeLLM, FakeTTS

from .conftest import ConsolePeer

pytestmark = pytest.mark.usefixtures("no_logged_tracebacks")


async def test_text_only_console_replies_without_audio(console: ConsolePeer) -> None:
    """A console call with audio output off and no TTS answers in text and plays nothing.
    https://github.com/livekit/agents/issues/5439"""
    session = VoiceSession(llm=FakeLLM({"hello": "You said: hello"}))
    await session.start(Agent(instructions="Answer."), room_options=RoomOptions(audio_output=False))

    await session.generate_reply(user_input="hello")
    await session.aclose()

    assert console.assistant_texts() == ["You said: hello"]
    assert "audio_output" not in console.kinds()


async def test_a_message_typed_in_the_console_is_answered_in_its_response(
    console: ConsolePeer,
) -> None:
    """Text the CLI sends as a ``run_input`` request runs a turn, and the request's response
    carries the agent's reply."""
    session = VoiceSession(
        llm=FakeLLM({"hello there": "You said: hello there"}), tts=FakeTTS(streaming=False)
    )
    await session.start(Agent(instructions="Answer."))
    typed = SessionRequest(
        request_id="typed", run_input=SessionRequest.RunInput(text="hello there")
    )

    response = await console.request(typed)
    await session.aclose()

    assert not response.error
    replies = [
        "".join(c.text for c in item.message.content)
        for item in response.run_input.items
        if item.WhichOneof("item") == "message" and item.message.role == ASSISTANT
    ]
    assert replies == ["You said: hello there"]


def greeting_agent() -> tuple[Agent, asyncio.Future[SpeechHandle]]:
    """An agent that greets on enter, and the greeting's speech once it is said."""
    agent = Agent(instructions="Answer.")
    greeting: asyncio.Future[SpeechHandle] = asyncio.get_running_loop().create_future()

    @agent.on_enter
    async def greet(session: VoiceSession) -> None:
        greeting.set_result(session.say("Welcome to the desk."))

    return agent, greeting


def said(session: VoiceSession) -> list[str | None]:
    return [item.text_content for item in session.history.items if item.type == "message"]


async def test_agent_initiated_messages_reach_the_console_once(console: ConsolePeer) -> None:
    """The on_enter greeting and a later say() reach the history and the CLI exactly once each.
    https://github.com/livekit/agents/issues/6577"""
    session = VoiceSession(tts=FakeTTS(streaming=False))
    agent, greeting = greeting_agent()
    await session.start(agent)
    await (await greeting)

    await session.say("Are you still there?")
    await session.aclose()

    assert said(session) == ["Welcome to the desk.", "Are you still there?"]
    assert console.assistant_texts() == ["Welcome to the desk.", "Are you still there?"]


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/6577 reproduces: on a console call without "
    "TTS, say() completes but its message never enters the history nor reaches the CLI",
)
async def test_agent_initiated_messages_reach_a_text_only_console_once(
    console: ConsolePeer,
) -> None:
    """Without a TTS, the greeting and a later say() reach the history and the CLI once each.
    https://github.com/livekit/agents/issues/6577"""
    session = VoiceSession()
    agent, greeting = greeting_agent()
    await session.start(agent, room_options=RoomOptions(audio_output=False))
    await (await greeting)

    await session.say("Are you still there?")
    await session.aclose()

    assert said(session) == ["Welcome to the desk.", "Are you still there?"]
    assert console.assistant_texts() == ["Welcome to the desk.", "Are you still there?"]
