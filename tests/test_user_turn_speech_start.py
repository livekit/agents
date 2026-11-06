"""``started_speaking_at`` of a user turn must point at the speech that was transcribed.

A VAD segment that never produces a transcript (a breath, a cough, line noise) opens a user
turn but cannot commit it. The next real utterance joins that turn, so the committed user
message used to report the noise onset as ``started_speaking_at``, sometimes tens of seconds
before the user spoke, and timestamp-sorted transcripts placed it ahead of agent speech that
actually came first.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import (
    Agent,
    AgentFalseInterruptionEvent,
    ConversationItemAddedEvent,
    TurnHandlingOptions,
)
from livekit.agents.llm import ChatMessage

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0
# FakeVAD back-dates START_OF_SPEECH to the speech onset; allow scheduling jitter
TOLERANCE = 0.3


class _Agent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


async def _run(
    actions: FakeActions,
    *,
    turn_handling: TurnHandlingOptions | None = None,
    can_pause_audio: bool = False,
    false_interruptions: list[AgentFalseInterruptionEvent] | None = None,
) -> tuple[float, list[ChatMessage]]:
    session = create_session(actions, turn_handling=turn_handling, can_pause_audio=can_pause_audio)
    messages: list[ChatMessage] = []

    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if isinstance(ev.item, ChatMessage):
            messages.append(ev.item)

    session.on("conversation_item_added", _on_item)
    if false_interruptions is not None:
        session.on("agent_false_interruption", false_interruptions.append)
    t_origin = await asyncio.wait_for(run_session(session, _Agent()), timeout=SESSION_TIMEOUT)
    return t_origin, messages


def _started_at(message: ChatMessage, t_origin: float) -> float:
    started = message.metrics.get("started_speaking_at")
    assert started is not None, f"no started_speaking_at on {message.text_content!r}"
    return started - t_origin


def _user_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [m for m in messages if m.role == "user"]


async def test_noise_while_agent_prepares_reply_does_not_backdate_next_turn() -> None:
    # noise lands while the agent is still generating, the agent speaks, and only then
    # does the user answer
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "I need to cancel a reservation.")  # EOU at ~3.0
    actions.add_llm("Sure, what is your phone number?", ttft=1.0, duration=1.2)
    actions.add_tts(2.0)  # agent audio ~4.2-6.2
    actions.add_user_speech(3.2, 3.4, "")  # noise during "thinking": VAD only, no transcript
    actions.add_user_speech(8.0, 10.0, "It's 555 0100.")  # EOU at ~10.5
    actions.add_llm("Thank you.", input="It's 555 0100.")
    actions.add_tts(1.0)

    t_origin, messages = await _run(actions)

    users = _user_messages(messages)
    assert [m.text_content for m in users] == [
        "I need to cancel a reservation.",
        "It's 555 0100.",
    ]
    assert _started_at(users[0], t_origin) == pytest.approx(0.5, abs=TOLERANCE)
    assert _started_at(users[1], t_origin) == pytest.approx(8.0, abs=TOLERANCE)

    # the user's answer is ordered after the agent's question it answers
    question = next(m for m in messages if m.text_content == "Sure, what is your phone number?")
    assert _started_at(question, t_origin) < _started_at(users[1], t_origin)


async def test_false_interruption_does_not_backdate_next_turn() -> None:
    # issue #7063: noise pauses the agent, the interruption is judged false and playout
    # resumes; the abandoned VAD segment must not become the start of the next user turn
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")  # EOU at ~3.0
    actions.add_llm("Once upon a time there was a hotel.", ttft=0.05, duration=0.05)
    actions.add_tts(6.0, ttfb=0.05, duration=0.05)  # agent audio from ~3.1
    actions.add_user_speech(4.0, 4.8, "")  # long enough to pause the agent, no transcript
    actions.add_user_speech(14.0, 15.5, "That was lovely, thank you.")
    actions.add_llm("You're welcome.", input="That was lovely, thank you.")
    actions.add_tts(1.0)

    false_interruptions: list[AgentFalseInterruptionEvent] = []
    t_origin, messages = await _run(
        actions, can_pause_audio=True, false_interruptions=false_interruptions
    )

    assert any(ev.resumed for ev in false_interruptions), "the noise must trigger a resume"
    users = _user_messages(messages)
    assert [m.text_content for m in users] == [
        "Tell me a story.",
        "That was lovely, thank you.",
    ]
    assert _started_at(users[1], t_origin) == pytest.approx(14.0, abs=TOLERANCE)


async def test_multi_segment_utterance_keeps_first_segment_start() -> None:
    # "Yes." <short pause> "The number is ..." is one utterance. The first segment's
    # transcript only lands once the second segment started, and the turn must still
    # start at the first segment.
    actions = FakeActions()
    actions.add_user_speech(1.0, 1.4, "Yes.", stt_delay=1.6)  # final at ~3.0
    actions.add_user_speech(2.1, 4.0, "The number is 555 0100.")
    actions.add_llm("Thanks.", input="Yes. The number is 555 0100.")
    actions.add_tts(1.0)

    # hold the turn open across the pause, as the turn detector does for an unfinished
    # sentence
    t_origin, messages = await _run(
        actions, turn_handling=TurnHandlingOptions(endpointing={"min_delay": 2.0})
    )

    users = _user_messages(messages)
    assert [m.text_content for m in users] == ["Yes. The number is 555 0100."]
    assert _started_at(users[0], t_origin) == pytest.approx(1.0, abs=TOLERANCE)


async def test_onset_shortly_before_speech_belongs_to_the_utterance() -> None:
    # a known limit: a VAD onset less than _UTTERANCE_MAX_PAUSE before the words (a breath,
    # a lip smack) can't be told apart from the speech and starts the utterance
    actions = FakeActions()
    actions.add_user_speech(1.0, 1.2, "")
    actions.add_user_speech(2.0, 3.5, "Good morning.")
    actions.add_llm("Good morning, how can I help?")
    actions.add_tts(1.0)

    t_origin, messages = await _run(actions)

    users = _user_messages(messages)
    assert [m.text_content for m in users] == ["Good morning."]
    assert _started_at(users[0], t_origin) == pytest.approx(1.0, abs=TOLERANCE)


async def test_later_utterance_in_the_same_turn_keeps_first_start() -> None:
    # only the turn's first transcript sets the start
    actions = FakeActions()
    actions.add_user_speech(1.0, 1.5, "Hello.")
    actions.add_user_speech(4.0, 5.0, "Is anyone there?")  # 2.5s pause, turn still held open
    actions.add_llm("Yes, how can I help?", input="Hello. Is anyone there?")
    actions.add_tts(1.0)

    t_origin, messages = await _run(
        actions, turn_handling=TurnHandlingOptions(endpointing={"min_delay": 4.0})
    )

    users = _user_messages(messages)
    assert [m.text_content for m in users] == ["Hello. Is anyone there?"]
    assert _started_at(users[0], t_origin) == pytest.approx(1.0, abs=TOLERANCE)


async def test_quick_next_turn_starts_at_its_own_speech() -> None:
    # the next turn begins less than _UTTERANCE_MAX_PAUSE after the previous one ended
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me about the spa.")  # EOU at ~3.0
    actions.add_llm("The spa has a pool, a sauna and treatment rooms.", ttft=0.05, duration=0.05)
    actions.add_tts(6.0, ttfb=0.05, duration=0.05)  # agent audio ~3.1-9.1
    actions.add_user_speech(4.0, 5.5, "Sorry, is it open today?")
    actions.add_llm("Yes, until eight.", input="Sorry, is it open today?")
    actions.add_tts(1.0)

    t_origin, messages = await _run(actions)

    users = _user_messages(messages)
    assert [m.text_content for m in users] == [
        "Tell me about the spa.",
        "Sorry, is it open today?",
    ]
    assert _started_at(users[1], t_origin) == pytest.approx(4.0, abs=TOLERANCE)
