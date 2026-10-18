"""A user turn that completes while the agent speaks uninterruptibly is kept, not dropped.

The reply is skipped on purpose: the speech cannot be interrupted, so nothing is generated
until it finishes. But what the user said must still land in the chat context (and be
announced as a conversation item), exactly like the ``skip_reply`` path does. Otherwise the
transcript loses the turn and the next reply is generated as if the user had never spoken.
"""

from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, ConversationItemAddedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_AGENT_LINE = "Your balance is eighty six thousand pesos, forty days overdue. Can you pay today?"
_USER_LINE = "I already paid that yesterday"


class _UninterruptibleGreetingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a collections agent.")

    async def on_enter(self) -> None:
        self.session.say(_AGENT_LINE, allow_interruptions=False)


async def test_user_turn_during_uninterruptible_speech_is_kept_in_chat_ctx() -> None:
    actions = FakeActions()
    # the line plays for 10s; the user speaks at 1s and their turn commits around 2s,
    # well before the line ends
    actions.add_tts(10.0, input=_AGENT_LINE)
    actions.add_user_speech(1.0, 1.5, _USER_LINE)

    # the user's audio is explicitly kept while the agent cannot be interrupted
    session: AgentSession = create_session(
        actions,
        turn_handling={"interruption": {"discard_audio_if_uninterruptible": False}},
    )
    agent = _UninterruptibleGreetingAgent()

    added_user_items: list[str | None] = []

    def _on_item_added(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "user":
            added_user_items.append(ev.item.text_content)

    session.on("conversation_item_added", _on_item_added)

    await run_session(session, agent, drain_delay=1.0)

    messages = [item for item in agent.chat_ctx.items if item.type == "message"]
    user_messages = [m for m in messages if m.role == "user"]
    assistant_messages = [m for m in messages if m.role == "assistant"]

    # the turn survived: in the chat context and announced to listeners
    assert [m.text_content for m in user_messages] == [_USER_LINE]
    assert added_user_items == [_USER_LINE]

    # the speech itself was untouched and no reply was generated for the skipped turn
    assert [m.text_content for m in assistant_messages] == [_AGENT_LINE]
    assert assistant_messages[0].interrupted is False
