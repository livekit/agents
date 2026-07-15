from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.voice import presets
from livekit.agents.voice.generation import (
    EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID,
    _strip_assistant_markup,
    remove_expressive_instructions,
    update_expressive_instructions,
)

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60


MARKED_UP = '<expression value="happy"/> Welcome back! [chuckling] Glad you called again.'


def test_strip_assistant_markup() -> None:
    ctx = ChatContext.empty()
    ctx.add_message(role="assistant", content=MARKED_UP)
    ctx.add_message(role="user", content='I typed <expression value="happy"/> literally')
    plain = ctx.add_message(role="assistant", content="No tags here.")
    plain_content = plain.content

    _strip_assistant_markup(ctx)

    assistant_texts = [
        item.text_content
        for item in ctx.items
        if item.type == "message" and item.role == "assistant"
    ]
    assert "<expression" not in (assistant_texts[0] or "")
    assert "[chuckling]" not in (assistant_texts[0] or "")
    assert "Welcome back!" in (assistant_texts[0] or "")
    assert "Glad you called again." in (assistant_texts[0] or "")

    # user content is never touched, tag-shaped or not
    user_text = next(
        item.text_content for item in ctx.items if item.type == "message" and item.role == "user"
    )
    assert '<expression value="happy"/>' in (user_text or "")

    # tag-free assistant content is left as-is (fast path)
    assert plain.content is plain_content


def test_update_options_expressive() -> None:
    session = AgentSession()
    assert session._expressive is False

    session.update_options(expressive={"speech_steering": presets.FORMAL})
    assert session._expressive == {"speech_steering": presets.FORMAL}

    # turn off
    session.update_options(expressive=False)
    assert session._expressive is False

    # turn back on with a different preset
    session.update_options(expressive={"speech_steering": presets.CASUAL})
    assert session._expressive == {"speech_steering": presets.CASUAL}

    # untouched when not given
    session.update_options()
    assert session._expressive == {"speech_steering": presets.CASUAL}


def test_update_and_remove_expressive_instructions() -> None:
    ctx = ChatContext.empty()
    update_expressive_instructions(ctx, text="markup guide v1")
    update_expressive_instructions(ctx, text="markup guide v2")

    guides = [item for item in ctx.items if item.id == EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID]
    assert len(guides) == 1, "re-injection must replace, not stack"
    assert guides[0].text_content == "markup guide v2"

    remove_expressive_instructions(ctx)
    assert all(item.id != EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID for item in ctx.items)


async def test_expressive_off_turn_scrubs_history() -> None:
    """A turn that runs with expressive off removes the injected markup guide and
    scrubs markup from past assistant turns."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)

    # FakeTTS has no markup dialect, so expressive resolves to off even though the
    # session asks for it — same situation as a handoff to a non-expressive TTS.
    session = create_session(actions)
    session._expressive = {"speech_steering": presets.FORMAL}

    # seed history as if previous turns ran expressive: marked-up assistant text +
    # the injected markup guide
    seeded = ChatContext.empty()
    seeded.add_message(role="assistant", content=MARKED_UP)
    update_expressive_instructions(seeded, text="Use the formatting tags below…")
    agent = Agent(instructions="You are a helpful assistant.", chat_ctx=seeded)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # the injected guide is gone
    assert all(item.id != EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID for item in agent.chat_ctx.items)

    assistant_texts = [
        item.text_content
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role == "assistant"
    ]
    assert assistant_texts, "expected assistant messages in history"
    for text in assistant_texts:
        assert "<expression" not in (text or "")
        assert "[chuckling]" not in (text or "")
    # the seeded message's visible text survives the scrub
    assert any("Welcome back!" in (t or "") for t in assistant_texts)
    # and the new reply went through normally
    assert any("I'm doing well" in (t or "") for t in assistant_texts)
