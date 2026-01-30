import asyncio

import pytest

from livekit.agents import llm
from livekit.agents.voice import Agent

from .fake_session import FakeActions, create_session, run_session
from .test_agent_session import check_timestamp


class AcknowledgmentAgent(Agent):
    def __init__(
        self,
        ack_text: str | None = None,
        ack_delay: float = 0.0,
        ack_timeout: float = 4.0,
        instructions: str = "You are a helpful assistant.",
    ):
        super().__init__(instructions=instructions)
        self.ack_text = ack_text
        self.ack_delay = ack_delay
        self.ack_timeout = ack_timeout

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.wait_for_acknowledgment(timeout=self.ack_timeout)
        if self.ack_text:

            async def _set_ack():
                if self.ack_delay > 0:
                    await asyncio.sleep(self.ack_delay)
                self.set_acknowledgment(self.ack_text)

            asyncio.create_task(_set_ack())


@pytest.mark.asyncio
async def test_normal_acknowledgment():
    speed = 2.0
    actions = FakeActions()
    actions.add_user_speech(0.0, 1.0, "hello")
    actions.add_llm("Hi!", ttft=2.0)  # Slow LLM
    actions.add_tts(0.5, input="OK")  # Acknowledgment audio
    actions.add_tts(0.5, input="Hi!")  # LLM response audio

    session = create_session(actions, speed_factor=speed)
    agent = AcknowledgmentAgent(ack_text="OK", ack_delay=0.1 / speed)

    conversation_events = []
    session.on("conversation_item_added", conversation_events.append)

    t_origin = await run_session(session, agent)

    # 1. User message
    # 2. Acknowledgment message
    # 3. LLM response message
    texts = [e.item.text_content for e in conversation_events]
    assert "hello" in texts
    assert "OK" in texts
    assert "Hi!" in texts

    # check timestamps
    # EOU at 1.0s. min_endpointing_delay 0.5. Starts at 1.5.
    # on_user_turn_completed finishes immediately. LLM starts at 1.5.
    # Acknowledgment "OK" starts at 1.5 + 0.1 = 1.6s
    # Playout of "OK" finishes at 1.6 + 0.2 (ttfb) + 0.5 (audio) = 2.3s
    # LLM response ready at 1.5 + 2.0 = 3.5s
    # Hi! finishes at 3.5 + 0.2 (ttfb) + 0.5 (audio) = 4.2s
    ok_ev = next(e for e in conversation_events if e.item.text_content == "OK")
    hi_ev = next(e for e in conversation_events if e.item.text_content == "Hi!")

    check_timestamp(ok_ev.created_at - t_origin, 2.3, speed_factor=speed)
    check_timestamp(hi_ev.created_at - t_origin, 4.2, speed_factor=speed)


@pytest.mark.asyncio
async def test_early_llm_skip_acknowledgment():
    speed = 2.0
    actions = FakeActions()
    actions.add_user_speech(0.0, 1.0, "hello")
    actions.add_llm("Hi!", ttft=0.1)  # Fast LLM
    actions.add_tts(0.5, input="OK")  # Acknowledgment audio (should be skipped)
    actions.add_tts(0.5, input="Hi!")  # LLM response audio

    session = create_session(actions, speed_factor=speed)
    agent = AcknowledgmentAgent(
        ack_text="OK", ack_delay=0.5 / speed
    )  # delay call to set_acknowledgment

    conversation_events = []
    session.on("conversation_item_added", conversation_events.append)

    await run_session(session, agent)

    # Acknowledgment should be skipped
    texts = [e.item.text_content for e in conversation_events]
    assert "hello" in texts
    assert "Hi!" in texts
    assert "OK" not in texts


@pytest.mark.asyncio
async def test_acknowledgment_timeout():
    speed = 2.0
    actions = FakeActions()
    actions.add_user_speech(0.0, 1.0, "hello")
    actions.add_llm("Hi!", ttft=2.0)
    actions.add_tts(0.5, input="Hi!")

    session = create_session(actions, speed_factor=speed)
    # ack_timeout is 0.5s, but we never call set_acknowledgment
    agent = AcknowledgmentAgent(ack_text=None, ack_timeout=0.5 / speed)

    conversation_events = []
    session.on("conversation_item_added", conversation_events.append)

    t_origin = await run_session(session, agent)

    texts = [e.item.text_content for e in conversation_events]
    assert "hello" in texts
    assert "Hi!" in texts
    assert "OK" not in texts

    hi_ev = next(e for e in conversation_events if e.item.text_content == "Hi!")
    # Starts at 1.5. Timeout at 2.0.
    # LLM starts at 1.5. Ready at 3.5.
    # Hi! finishes at 3.5 + 0.2 + 0.5 = 4.2.
    check_timestamp(hi_ev.created_at - t_origin, 4.2, speed_factor=speed)


@pytest.mark.asyncio
async def test_interruption_during_acknowledgment():
    speed = 2.0
    actions = FakeActions()
    actions.add_user_speech(0.0, 1.0, "hello")
    actions.add_llm("Hi!", ttft=5.0)  # Very slow LLM
    actions.add_tts(2.0, input="OK")  # Long acknowledgment

    # User interrupts at 2.0s (during "OK" playback)
    actions.add_user_speech(2.0, 2.5, "stop")
    actions.add_llm("Understood", input="stop", ttft=0.1)
    actions.add_tts(0.5, input="Understood")

    session = create_session(actions, speed_factor=speed)
    agent = AcknowledgmentAgent(ack_text="OK", ack_delay=0.1 / speed)

    conversation_events = []
    session.on("conversation_item_added", conversation_events.append)

    await run_session(session, agent)

    texts = [e.item.text_content for e in conversation_events]
    assert "hello" in texts
    assert "OK" in texts
    assert "stop" in texts
    assert "Understood" in texts

    ok_msg = next(e.item for e in conversation_events if e.item.text_content == "OK")
    assert ok_msg.interrupted

    # "Hi!" should NOT be in texts because it was cancelled by the new user turn
    assert "Hi!" not in texts
