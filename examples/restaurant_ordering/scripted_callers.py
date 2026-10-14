"""Hang up on scripted callers.

A bot farm runs two fixed question banks against public LiveKit agents (an opinion bank
that ends "please limit your answers to about forty words", and a flat trivia bank); on the
homepage agent they were two thirds of all sessions. Their opening lines are known
(``scripted_questions.txt``, the same list as examples/homepage), so the check is an exact
match on the caller's first few turns, after normalizing away case and punctuation. On a
match the call ends, and the page shows why.

A turn here is the user message the session commits at end of turn, which carries the whole
utterance; final STT segments aren't used, since one utterance can arrive as several. Only
the first ``MAX_TURNS_CHECKED`` turns are checked: a real caller who's past their opening
is never cut off by this.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from call import hang_up
from page import Page

from livekit.agents import AgentSession, ChatMessage, ConversationItemAddedEvent

logger = logging.getLogger(__name__)

SCRIPTED_QUESTIONS_PATH = Path(__file__).with_name("scripted_questions.txt")
MAX_TURNS_CHECKED = 3

_NOT_ALNUM = re.compile(r"[^a-z0-9 ]")


def normalize(text: str) -> str:
    """Lowercase alphanumerics with single spaces, so STT punctuation and casing don't matter."""
    return " ".join(_NOT_ALNUM.sub("", text.lower()).split())


def load_scripted_questions(path: Path = SCRIPTED_QUESTIONS_PATH) -> frozenset[str]:
    """The normalized bank lines, one per non-blank, non-comment line of the file."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return frozenset(normalize(line) for line in lines if line and not line.startswith("#"))


SCRIPTED_QUESTIONS = load_scripted_questions()


def is_scripted(transcript: str, questions: frozenset[str] = SCRIPTED_QUESTIONS) -> bool:
    return normalize(transcript) in questions


def disconnect_scripted_callers(
    session: AgentSession[Any],
    page: Page | None = None,
    *,
    questions: frozenset[str] = SCRIPTED_QUESTIONS,
    max_turns: int = MAX_TURNS_CHECKED,
) -> None:
    checked = 0
    tasks: set[asyncio.Task[None]] = set()

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        nonlocal checked
        item = ev.item
        if not isinstance(item, ChatMessage) or item.role != "user" or checked >= max_turns:
            return
        checked += 1
        text = item.text_content or ""
        if not is_scripted(text, questions):
            return

        turn, checked = checked, max_turns
        logger.info(
            "disconnecting scripted caller",
            extra={"lk.pii.user_input": text, "caller_turn": turn},
        )
        try:
            session.interrupt(force=True)
        except RuntimeError:
            pass  # nothing to cut off; the call is ending regardless
        task = asyncio.create_task(hang_up("Scripted caller", page))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
