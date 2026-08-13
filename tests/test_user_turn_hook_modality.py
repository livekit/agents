"""on_user_turn_completed fires for a user turn regardless of how it arrived.

The hook is the documented seam for editing or dropping a user message before it
reaches the LLM, so a guard wired there has to hold for typed input the same way
it does for speech.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit.agents import Agent, AgentSession, StopResponse
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.room_io.types import TextInputEvent, _default_text_input_cb
from livekit.agents.voice.transcription.synchronizer import (
    TranscriptSynchronizer,
    _SyncedAudioOutput,
)

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class HookAgent(Agent):
    def __init__(self, *, stop: bool = False) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self.turns: list[str] = []
        self._stop = stop

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        self.turns.append(new_message.text_content or "")
        if self._stop:
            raise StopResponse()


async def _run_text_turn(session: AgentSession, agent: Agent, text: str) -> None:
    """run_session's shape, driven by a typed turn instead of the audio input."""
    transcription_sync: TranscriptSynchronizer | None = None
    if isinstance(session.output.audio, _SyncedAudioOutput):
        transcription_sync = session.output.audio._synchronizer

    await session.start(agent)
    await _default_text_input_cb(session, TextInputEvent(text=text))
    await asyncio.sleep(5.0)
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()

    if transcription_sync is not None:
        await transcription_sync.aclose()


async def test_hook_fires_for_spoken_turn() -> None:
    actions = FakeActions()
    actions.add_user_speech(start_time=0.0, end_time=1.0, transcript="hello there")
    actions.add_llm("hi!")
    actions.add_tts(audio_duration=0.5)

    agent = HookAgent()
    await asyncio.wait_for(run_session(create_session(actions), agent), timeout=60.0)

    assert agent.turns == ["hello there"]


async def test_hook_fires_for_text_turn() -> None:
    actions = FakeActions()
    actions.add_llm("hi!", input="hello there")
    actions.add_tts(audio_duration=0.5)

    agent = HookAgent()
    await _run_text_turn(create_session(actions, with_stt=False), agent, "hello there")

    assert agent.turns == ["hello there"]


async def test_hook_can_edit_the_chat_ctx_for_a_text_turn() -> None:
    class EditingAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="You are a helpful assistant.")

        async def on_user_turn_completed(
            self, turn_ctx: ChatContext, new_message: ChatMessage
        ) -> None:
            new_message.content = ["edited before the llm saw it"]

    actions = FakeActions()
    actions.add_llm("hi!", input="edited before the llm saw it")
    actions.add_tts(audio_duration=0.5)

    session = create_session(actions, with_stt=False)
    items: list[ChatMessage] = []
    session.on(
        "conversation_item_added",
        lambda ev: items.append(ev.item) if ev.item.type == "message" else None,
    )

    await _run_text_turn(session, EditingAgent(), "original")

    user_messages = [i for i in items if i.role == "user"]
    assert [m.text_content for m in user_messages] == ["edited before the llm saw it"]


async def test_stop_response_drops_a_text_turn() -> None:
    actions = FakeActions()
    actions.add_llm("hi!", input="hello there")
    actions.add_tts(audio_duration=0.5)

    session = create_session(actions, with_stt=False)
    replies: list[str] = []
    session.on(
        "conversation_item_added",
        lambda ev: (
            replies.append(ev.item.text_content or "")
            if ev.item.type == "message" and ev.item.role == "assistant"
            else None
        ),
    )

    agent = HookAgent(stop=True)
    await _run_text_turn(session, agent, "hello there")

    assert agent.turns == ["hello there"]
    assert replies == []
