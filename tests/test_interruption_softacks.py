from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent
from .fake_session import FakeActions, create_session, run_session


SESSION_TIMEOUT = 60.0


class SimpleAgent(Agent):
    """Agent that can handle user inputs and generate responses."""
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


@pytest.mark.asyncio
async def test_scenario_1_soft_acks_during_speaking() -> None:
    """
    Scenario 1: The Long Explanation
    Agent is reading a long paragraph. User says "Okay... yeah... uh-huh" while Agent is talking.
    Result: Agent audio does not break. It ignores the user input completely.
    
    PASS CONDITION: Only ONE thinking phase should occur (the initial response).
    If soft-acks caused interruptions, we'd see multiple thinking phases.
    """
    speed = 5.0
    actions = FakeActions()
    # Agent starts by processing a user request
    actions.add_user_speech(0.5, 1.0, "Tell me about history")
    # Agent generates a long response
    actions.add_llm("History is the study of past events. Let me explain more...", input="Tell me about history")
    actions.add_tts(8.0)  # Long TTS playback (8 seconds in fake-time) - covers all soft-acks during speaking

    # While agent is speaking, user interjects with soft-acks (should be ignored)
    actions.add_user_speech(1.5, 1.55, "okay")
    actions.add_user_speech(2.0, 2.05, "yeah")
    actions.add_user_speech(2.5, 2.55, "uh-huh")

    session = create_session(actions, speed_factor=speed)
    agent = SimpleAgent()

    agent_state_events = []
    def log_state_event(ev):
        print(f"TEST: Captured event: {ev.old_state} -> {ev.new_state}")
        agent_state_events.append(ev)
    session.on("agent_state_changed", log_state_event)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    # If soft-acks caused interruptions, we'd see multiple thinking phases
    transitions = [(ev.old_state, ev.new_state) for ev in agent_state_events]
    
    # Count thinking phases: should be exactly 1 (initial response)
    thinking_entries = [t for t in transitions if t[1] == "thinking"]
    assert len(thinking_entries) == 1, (
        f"Expected exactly 1 thinking phase (initial response only), "
        f"but got {len(thinking_entries)}. This means soft-acks triggered extra interrupts. "
        f"Transitions: {transitions}"
    )


@pytest.mark.asyncio
async def test_scenario_2_soft_acks_when_silent() -> None:
    """
    Scenario 2: The Passive Affirmation
    Agent asks "Are you ready?" and goes silent.
    User says "Yeah."
    Result: Agent processes "Yeah" as an answer and proceeds.
    
    PASS CONDITION: Agent should respond to "Yeah" when silent (2+ thinking phases).
    """
    speed = 5.0
    actions = FakeActions()
    
    # User initiates conversation
    actions.add_user_speech(0.5, 1.0, "Start")
    # Agent asks a question
    actions.add_llm("Are you ready to start?", input="Start")
    actions.add_tts(1.0)
    
    # User responds with soft-ack while agent is silent (after speaking finishes)
    actions.add_user_speech(4.0, 4.1, "Yeah")
    # Agent should process this as valid input and respond
    actions.add_llm("Great, let's begin!", input="Yeah")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=speed)
    agent = SimpleAgent()

    agent_state_events = []
    def log_state_event(ev):
        print(f"TEST: Captured event #{len(agent_state_events)}: {ev.old_state} -> {ev.new_state}")
        agent_state_events.append(ev)
    session.on("agent_state_changed", log_state_event)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    transitions = [(ev.old_state, ev.new_state) for ev in agent_state_events]
    print(f"\nTEST: Total events captured: {len(agent_state_events)}")
    print(f"TEST: All transitions: {transitions}")
    thinking_entries = [t for t in transitions if t[1] == "thinking"]
    print(f"TEST: Thinking transitions: {thinking_entries}")
    print(f"TEST: Number of thinking phases: {len(thinking_entries)}")
    assert len(thinking_entries) >= 2, (
        f"Soft-ack while agent silent should trigger a response. "
        f"Expected at least 2 thinking phases, got {len(thinking_entries)}. "
        f"Transitions: {transitions}"
    )


@pytest.mark.asyncio
async def test_scenario_3_strong_interrupt() -> None:
    """
    Scenario 3: The Correction
    Agent is counting "One, two, three..."
    User says "No stop."
    Result: Agent cuts off immediately.
    
    PASS CONDITION: Agent should transition to listening state (interrupt detected).
    """
    speed = 5.0
    actions = FakeActions()
    
    # Agent starts counting
    actions.add_llm("One, two, three, four, five...", input="count")
    actions.add_tts(4.0)  # Long TTS (4 seconds)

    # User interrupts with strong command during playback
    actions.add_user_speech(1.5, 1.55, "No stop")

    session = create_session(actions, speed_factor=speed)
    agent = SimpleAgent()

    agent_state_events = []
    session.on("agent_state_changed", agent_state_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # Strong interrupts should cause the agent to transition to listening
    # while it's still supposed to be speaking
    states = [ev.new_state for ev in agent_state_events]
    
    # After agent starts speaking, we should see a transition to listening
    # (representing the interrupt cutting off the speech)
    assert "listening" in states, (
        f"Strong interrupt ('No stop') should cause transition to listening state. "
        f"States: {states}"
    )


@pytest.mark.asyncio
async def test_scenario_4_mixed_input_with_strong_word() -> None:
    """
    Scenario 4: The Mixed Input
    Agent is speaking.
    User says "Yeah okay but wait."
    Result: Agent stops (because "but wait" is not in the ignore list).
    
    PASS CONDITION: Agent should interrupt when mixed input contains strong words.
    """
    speed = 5.0
    actions = FakeActions()
    
    # Agent starts speaking
    actions.add_llm("The answer is quite complex and requires explanation.", input="explain")
    actions.add_tts(3.0)

    # User says a mixed sentence containing a strong word
    actions.add_user_speech(1.5, 1.65, "Yeah okay but wait")

    session = create_session(actions, speed_factor=speed)
    agent = SimpleAgent()

    agent_state_events = []
    session.on("agent_state_changed", agent_state_events.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # Mixed input with strong word "wait" should cause interrupt
    states = [ev.new_state for ev in agent_state_events]
    
    # Should transition to listening (interrupt happened)
    assert "listening" in states, (
        f"Mixed input with strong word ('Yeah okay but wait') should interrupt. "
        f"States: {states}"
    )
