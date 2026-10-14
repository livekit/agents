from __future__ import annotations

import asyncio

import pytest
import scripted_callers
from scripted_callers import (
    disconnect_scripted_callers,
    is_scripted,
    load_scripted_questions,
    normalize,
)

from livekit.agents import AgentSession, ChatMessage, ConversationItemAddedEvent

pytestmark = pytest.mark.unit

REAL = "Can I get a large pepperoni for delivery?"


def test_the_bank_loads_normalized_lines() -> None:
    questions = load_scripted_questions()
    assert len(questions) == 40
    assert all(q == normalize(q) for q in questions)


def test_a_bank_question_matches_whatever_the_casing_and_punctuation() -> None:
    assert is_scripted("What is one plus one?")
    assert is_scripted(
        "What do you think a healthy lifestyle includes? "
        "Please limit your answers to about forty words."
    )


def test_real_callers_are_left_alone() -> None:
    assert not is_scripted("Hello?")
    assert not is_scripted(REAL)


def _said(session: AgentSession, text: str) -> None:
    item = ChatMessage(role="user", content=[text])
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=item))


async def _attach(monkeypatch: pytest.MonkeyPatch) -> tuple[AgentSession, list[str]]:
    session: AgentSession = AgentSession()
    session.interrupt = lambda **kwargs: None  # type: ignore[method-assign, assignment]
    reasons: list[str] = []

    async def hang_up(reason: str, page: object = None) -> None:
        reasons.append(reason)

    monkeypatch.setattr(scripted_callers, "hang_up", hang_up)
    disconnect_scripted_callers(session)
    return session, reasons


async def test_a_scripted_opening_ends_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    session, reasons = await _attach(monkeypatch)
    _said(session, "What is one plus one?")
    await asyncio.sleep(0)
    assert reasons == ["Scripted caller"]


async def test_only_the_first_turns_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    session, reasons = await _attach(monkeypatch)
    for _ in range(scripted_callers.MAX_TURNS_CHECKED):
        _said(session, REAL)
    _said(session, "What is one plus one?")
    await asyncio.sleep(0)
    assert reasons == []
