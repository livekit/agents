from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_io import FakeAudioInput
from .fake_session import FakeActions, create_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_TRANSCRIPT = "the mailbox is full and cannot accept new messages at this time"


async def _start_session() -> tuple[AgentSession, Agent]:
    actions = FakeActions()
    actions.add_user_speech(0.05, 0.2, _TRANSCRIPT, stt_delay=0.3)
    session = create_session(
        actions,
        turn_handling={"turn_detection": "stt"},
        extra_kwargs={"session_close_transcript_timeout": 2.0},
    )
    agent = Agent(instructions="You are a helpful assistant.")
    await session.start(agent)

    audio_input = session.input.audio
    assert isinstance(audio_input, FakeAudioInput)
    audio_input.push(0.1)
    await asyncio.sleep(0.1)
    return session, agent


async def _close_session(session: AgentSession) -> None:
    audio_output = session.output.audio
    transcription_sync = (
        audio_output._synchronizer if isinstance(audio_output, _SyncedAudioOutput) else None
    )
    await session.aclose()
    if transcription_sync is not None:
        await transcription_sync.aclose()


def _user_transcripts(agent: Agent) -> list[str | None]:
    return [
        item.text_content
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role == "user"
    ]


@pytest.mark.asyncio
async def test_session_close_commits_trailing_transcript() -> None:
    session, agent = await _start_session()

    await _close_session(session)

    assert _user_transcripts(agent) == [_TRANSCRIPT]
