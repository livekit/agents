from __future__ import annotations

import asyncio

from livekit.agents import Agent, UserTurnExceededEvent

from .fake_session import FakeActions, create_session, run_session

SESSION_TIMEOUT = 30


class _CapturingAgent(Agent):
    """Test agent that captures on_user_turn_exceeded events instead of responding."""

    def __init__(self) -> None:
        super().__init__(instructions="test agent")
        self.exceeded_events: list[UserTurnExceededEvent] = []

    async def on_user_turn_exceeded(self, ev: UserTurnExceededEvent) -> None:
        self.exceeded_events.append(ev)


async def test_word_limit_triggers() -> None:
    """When user speaks more words than max_words, on_user_turn_exceeded fires."""
    speed = 5.0
    agent = _CapturingAgent()

    actions = FakeActions()
    actions.add_user_speech(
        0.5, 3.0, "one two three four five six seven eight nine ten", stt_delay=0.2
    )
    actions.add_llm("OK", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"user_turn_limit": {"max_words": 5}},
    )

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(agent.exceeded_events) == 1
    assert agent.exceeded_events[0].accumulated_word_count >= 5


async def test_duration_limit_triggers() -> None:
    """When user speaks longer than max_duration, on_user_turn_exceeded fires."""
    speed = 5.0
    agent = _CapturingAgent()

    actions = FakeActions()
    actions.add_user_speech(0.5, 3.5, "hello world", stt_delay=0.2)
    actions.add_llm("OK", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"user_turn_limit": {"max_duration": 1.0 / speed}},
    )

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(agent.exceeded_events) == 1
    assert agent.exceeded_events[0].duration > 0


async def test_no_trigger_when_disabled() -> None:
    """No event fires when user_turn_limit is not configured."""
    speed = 5.0
    agent = _CapturingAgent()

    actions = FakeActions()
    actions.add_user_speech(
        0.5, 3.0, "one two three four five six seven eight nine ten", stt_delay=0.2
    )
    actions.add_llm("OK", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)

    session = create_session(actions, speed_factor=speed)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(agent.exceeded_events) == 0


async def test_reset_on_agent_speaking() -> None:
    """Counters reset when agent reaches speaking state.
    After reset, a second user turn under the limit should NOT trigger."""
    speed = 5.0
    agent = _CapturingAgent()

    # Turn 1: user says 3 words, agent replies successfully (reaches speaking)
    # Turn 2: user says 3 words — should NOT trigger with max_words=5
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "one two three", stt_delay=0.2)
    actions.add_llm("reply one", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)
    actions.add_user_speech(4.0, 5.0, "four five six", stt_delay=0.2)
    actions.add_llm("reply two", input="four five six", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, input="reply two", ttfb=0.1, duration=0.1)

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"user_turn_limit": {"max_words": 5}},
    )

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # Neither turn alone exceeds 5 words, and counters reset between them
    assert len(agent.exceeded_events) == 0


async def test_accumulation_across_interrupted_turns() -> None:
    """When a user turn completes and the user interrupts the agent before it speaks,
    the previous turn is committed to chat context and the exceeded event's
    accumulated_transcript equals (prior turn in chat ctx) + (current transcript).
    """
    speed = 5.0
    agent = _CapturingAgent()

    turn1 = "one two three"
    turn2 = "four five six"

    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, turn1, stt_delay=0.2)
    # Very slow LLM so the agent never reaches speaking before turn 2 interrupts
    actions.add_llm("reply one", ttft=5.0, duration=0.1)
    actions.add_tts(0.5, ttfb=0.5, duration=0.5)
    actions.add_user_speech(2.2, 3.2, turn2, stt_delay=0.2)
    actions.add_llm("reply two", input=turn2, ttft=5.0, duration=0.1)
    actions.add_tts(0.5, input="reply two", ttfb=0.5, duration=0.5)

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"user_turn_limit": {"max_words": 5}},
    )

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(agent.exceeded_events) == 1
    ev = agent.exceeded_events[0]

    # current (last) transcript is turn 2
    assert ev.transcript == turn2

    # turn 1 was committed to chat context via the normal on_user_turn_completed path
    user_texts = [
        item.text_content
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role == "user"
    ]
    assert turn1 in user_texts, f"turn1 not found in chat_ctx: {user_texts}"

    # accumulated_transcript = prior turn (in chat ctx) + last transcript
    assert ev.accumulated_transcript == f"{turn1} {turn2}"


async def test_callback_receives_correct_event_data() -> None:
    """The callback receives an event with correct transcript and word count."""
    speed = 5.0
    agent = _CapturingAgent()
    transcript = "alpha bravo charlie delta echo foxtrot"

    actions = FakeActions()
    actions.add_user_speech(0.5, 3.0, transcript, stt_delay=0.2)
    actions.add_llm("OK", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)

    session = create_session(
        actions,
        speed_factor=speed,
        turn_handling={"user_turn_limit": {"max_words": 3}},
    )

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(agent.exceeded_events) == 1
    ev = agent.exceeded_events[0]
    assert ev.accumulated_word_count >= 3
    assert ev.transcript == transcript
    assert ev.accumulated_transcript == transcript
    assert ev.duration >= 0
