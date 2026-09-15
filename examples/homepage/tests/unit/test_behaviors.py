import behaviors.scripted_callers as scripted_callers
import pytest
from behaviors.frontend_attributes import frontend_attributes
from behaviors.scripted_callers import (
    SCRIPTED_QUESTIONS,
    disconnect_scripted_callers,
    is_scripted,
    load_scripted_questions,
    normalize,
)
from behaviors.user_away import CHECK_IN_INSTRUCTIONS, check_in_when_user_away

from livekit.agents import AgentSession, UserInputTranscribedEvent, UserStateChangedEvent

pytestmark = pytest.mark.unit


def test_frontend_attributes_carry_the_configured_tts_voice() -> None:
    assert frontend_attributes(tts_voice="Nate") == {"tts_voice": "Nate"}
    assert frontend_attributes(tts_voice=None) == {}


@pytest.mark.asyncio
async def test_user_away_checkin() -> None:
    session = AgentSession()
    check_in_when_user_away(session)

    replies = []
    session.generate_reply = lambda **kwargs: replies.append(kwargs)  # type: ignore[method-assign]

    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="away"),
    )
    assert replies == [
        {
            "instructions": CHECK_IN_INSTRUCTIONS,
            "allow_interruptions": True,
        }
    ]

    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="away", new_state="speaking"),
    )
    assert len(replies) == 1


def test_scripted_questions_file_loads_forty_normalized_lines() -> None:
    questions = load_scripted_questions()
    assert len(questions) == 40
    assert all(q == normalize(q) for q in questions)
    assert not any(q.startswith("#") for q in questions)


def test_is_scripted_ignores_case_and_punctuation() -> None:
    assert is_scripted("What is one plus one?")
    assert is_scripted("  what IS the speed of light.  ")
    assert is_scripted(
        "What do you think a healthy lifestyle includes? "
        "Please limit your answers to about forty words."
    )


def test_is_scripted_leaves_real_questions_alone() -> None:
    assert not is_scripted("Hello?")
    assert not is_scripted("Does LiveKit support SIP and what does it cost?")
    assert not is_scripted("How do I get started building an agent in Python?")
    assert not is_scripted("what is two plus two")


class _FakeJobContext:
    def __init__(self) -> None:
        self.deleted = 0

    def delete_room(self) -> None:
        self.deleted += 1


def _attach(monkeypatch: pytest.MonkeyPatch) -> tuple[AgentSession, list[int], _FakeJobContext]:
    session = AgentSession()
    interrupts: list[int] = []
    session.interrupt = lambda **kwargs: interrupts.append(1)  # type: ignore[method-assign]
    ctx = _FakeJobContext()
    monkeypatch.setattr(scripted_callers, "get_job_context", lambda: ctx)
    disconnect_scripted_callers(session)
    return session, interrupts, ctx


def _transcript(text: str, *, is_final: bool = True) -> UserInputTranscribedEvent:
    return UserInputTranscribedEvent(transcript=text, is_final=is_final)


@pytest.mark.asyncio
async def test_scripted_opening_line_deletes_the_room_once(monkeypatch: pytest.MonkeyPatch) -> None:
    session, interrupts, ctx = _attach(monkeypatch)
    scripted = next(iter(SCRIPTED_QUESTIONS))

    session.emit("user_input_transcribed", _transcript(scripted, is_final=False))
    assert ctx.deleted == 0

    session.emit("user_input_transcribed", _transcript(scripted))
    assert ctx.deleted == 1
    assert interrupts == [1]

    session.emit("user_input_transcribed", _transcript(scripted))
    assert ctx.deleted == 1


@pytest.mark.asyncio
async def test_real_caller_is_never_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    session, interrupts, ctx = _attach(monkeypatch)
    for line in ("Hello?", "I want to build a voice agent.", "Which STT should I use?"):
        session.emit("user_input_transcribed", _transcript(line))
    assert ctx.deleted == 0
    assert interrupts == []


@pytest.mark.asyncio
async def test_only_the_opening_turns_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _, ctx = _attach(monkeypatch)
    for line in ("Hello?", "I have a question.", "About pricing."):
        session.emit("user_input_transcribed", _transcript(line))

    session.emit("user_input_transcribed", _transcript("What is one plus one?"))
    assert ctx.deleted == 0
