"""Disconnect scripted callers.

Two fixed question banks are run against the public homepage agent by
automated callers. Their opening lines are known (``scripted_questions.txt``),
so the check is an exact match on the caller's first few final transcripts,
after normalizing away case and punctuation. On a match the room is deleted,
which disconnects the caller and ends the job. Only the first
``MAX_TURNS_CHECKED`` caller turns are checked; the scripts open with a bank
question, and a real conversation that has got past its opening should never
be cut off by this.
"""

import logging
import re
from pathlib import Path

from livekit.agents import AgentSession, UserInputTranscribedEvent, get_job_context

logger = logging.getLogger("agent")

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
    session: AgentSession,
    *,
    questions: frozenset[str] = SCRIPTED_QUESTIONS,
    max_turns: int = MAX_TURNS_CHECKED,
) -> None:
    checked = 0

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        nonlocal checked
        if not ev.is_final or checked >= max_turns:
            return
        checked += 1
        if not is_scripted(ev.transcript, questions):
            return

        turn, checked = checked, max_turns
        logger.info(
            "disconnecting scripted caller",
            extra={"transcript": ev.transcript, "caller_turn": turn},
        )
        get_job_context().delete_room()
        try:
            session.interrupt(force=True)
        except RuntimeError:
            pass  # nothing to cut off; the room is going away regardless
