"""The reconnection replay has to carry what the model already said.

A response that finishes writes each message's transcript into its mirror item in
`_handle_response_done`. A connection that drops mid-reply never delivers that event,
so without a flush the item stays empty, the replay drops it, and the replayed
conversation ends on the user's request with nothing answering it.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm import remote_chat_context
from livekit.agents.llm._realtime.openai import (
    RealtimeSession,
    _MessageGeneration,
    _ResponseGeneration,
)

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

_PARTIAL = "The capital of France is Pa"


def _session_mid_reply(transcript: str = _PARTIAL) -> tuple[RealtimeSession, str]:
    """A session whose model is part-way through a reply, as a dropped socket leaves it.

    The user's request is in the mirror, and so is the assistant item the response
    created -- with empty content, because only `response.done` fills it in.
    """
    session = RealtimeSession.__new__(RealtimeSession)
    session._remote_chat_ctx = remote_chat_context.RemoteChatContext()

    user = llm.ChatMessage(role="user", content=["What is the capital of France?"])
    session._remote_chat_ctx.insert(None, user)
    assistant = llm.ChatMessage(role="assistant", content=[])
    session._remote_chat_ctx.insert(user.id, assistant)

    session._current_generation = _ResponseGeneration(
        message_ch=utils.aio.Chan[llm.MessageGeneration](),
        function_ch=utils.aio.Chan[llm.FunctionCall](),
        messages={
            assistant.id: _MessageGeneration(
                message_id=assistant.id,
                text_ch=utils.aio.Chan[str](),
                audio_ch=utils.aio.Chan[rtc.AudioFrame](),
                modalities=asyncio.Future(),
                audio_transcript=transcript,
            )
        },
        _done_fut=asyncio.Future(),
        _created_timestamp=0.0,
    )
    return session, assistant.id


def _replayed(session: RealtimeSession) -> llm.ChatContext:
    """Exactly the context _reconnect replays."""
    return session.chat_ctx.copy(
        exclude_instructions=True,
        exclude_empty_message=True,
        exclude_handoff=True,
        exclude_config_update=True,
    )


def test_replay_keeps_what_the_agent_already_said() -> None:
    session, assistant_id = _session_mid_reply()

    session._flush_generation_transcripts()

    replayed = _replayed(session)
    assert [item.id for item in replayed.items] == [replayed.items[0].id, assistant_id]
    assert replayed.items[-1].role == "assistant"
    assert replayed.items[-1].content == [_PARTIAL]


def test_a_retried_reconnection_does_not_duplicate_the_transcript() -> None:
    """A reconnection whose sends fail restores the mirror and is retried.

    The transcript is still on the generation, so a second flush must be a no-op
    rather than appending the same text again.
    """
    session, assistant_id = _session_mid_reply()

    session._flush_generation_transcripts()
    session._flush_generation_transcripts()

    assert _replayed(session).items[-1].content == [_PARTIAL]
