from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import pytest

from livekit.agents import (
    DEFAULT_BACKCHANNEL_PHRASES,
    Agent,
    AgentSession,
    UserInputTranscribedEvent,
)
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.backchannel import is_backchannel_only
from livekit.agents.voice.io import PlaybackFinishedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


class EchoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


# region matcher


@pytest.mark.parametrize(
    "text",
    [
        "Okay.",
        "Mhm.",
        "OK, thank you!",
        "Yeah. Makes sense.",
        "Uh, sounds good.",
        "Got it, thanks.",
        # filler-only: with a low min_words a lone filler passes the
        # word-count gate, so the phrase filter must claim it
        "Uh.",
        "Um, hm.",
        # fillers must not be pre-stripped: that would eat the "uh" of
        # "uh huh" and the phrase could never match
        "Uh-huh.",
        "Mm-hmm.",
        "Uh-huh, yeah.",
    ],
)
def test_backchannel_only(text: str) -> None:
    assert is_backchannel_only(text, DEFAULT_BACKCHANNEL_PHRASES)


@pytest.mark.parametrize(
    "text",
    [
        "",  # no words yet: the min_duration/min_words gates own the decision
        "...",  # punctuation-only normalizes to nothing
        "Okay, but wait.",
        "No.",
        "Wait.",
        "Stop.",
        "Hello?",  # often "are you there?" — a real barge-in
        "Huh?",  # a "what?"-style barge-in; "uh huh" the backchannel matches
        "Uh-huh, but wait.",
        "So I need to know something.",
        "Yeah, actually my question is",
    ],
)
def test_not_backchannel(text: str) -> None:
    assert not is_backchannel_only(text, DEFAULT_BACKCHANNEL_PHRASES)


@pytest.mark.parametrize(
    "text",
    [
        "Thank",  # → "thank you"
        "Makes",  # → "makes sense"
        "Sounds",  # → "sounds good"
        "Okay, thank",  # matched phrase + prefix tail
        "Mm",  # filler AND prefix of "mm hmm" — either way, defer
    ],
)
def test_partial_defers_phrase_prefixes(text: str) -> None:
    """The interruption path judges live interims: a trailing prefix of a
    known phrase defers the cut (interims arrive word by word). The same text
    as a final is a complete utterance and must not match (except a lone
    filler, which is ignorable on finals too)."""
    assert is_backchannel_only(text, DEFAULT_BACKCHANNEL_PHRASES, partial=True)
    if text != "Mm":
        assert not is_backchannel_only(text, DEFAULT_BACKCHANNEL_PHRASES)


@pytest.mark.parametrize(
    "text",
    [
        "Thank god you called",  # prefix word, but the tail diverges → cut
        "No",  # real barge-in words are never prefixes — cut latency untouched
        "Wait",
        "So",
        "Makes no difference",
    ],
)
def test_partial_still_cuts_real_speech(text: str) -> None:
    assert not is_backchannel_only(text, DEFAULT_BACKCHANNEL_PHRASES, partial=True)


def test_custom_phrases() -> None:
    assert is_backchannel_only("de acuerdo", ["de acuerdo", "vale"])
    assert not is_backchannel_only("de acuerdo", DEFAULT_BACKCHANNEL_PHRASES)


def test_blank_phrase_entries_are_ignored() -> None:
    # a phrase that normalizes to nothing must not match-without-consuming
    # (it would hang the matching loop on any non-backchannel utterance)
    assert not is_backchannel_only("so my question is", ["", "...", "—", "okay"])
    assert is_backchannel_only("okay", ["", "...", "okay"])
    assert not is_backchannel_only("anything", ["", "..."])


def test_callable_filter() -> None:
    # a callable filter owns the classification entirely and receives the
    # text verbatim
    seen: list[str] = []

    def flt(text: str) -> bool:
        seen.append(text)
        return text == "Okay."

    assert is_backchannel_only("Okay.", flt)
    assert not is_backchannel_only("Okay, but wait.", flt)
    assert seen == ["Okay.", "Okay, but wait."]

    # partial is a phrase-matcher concept; the callable's verdict is used as-is
    assert is_backchannel_only("Okay.", flt, partial=True)

    # empty/punctuation-only text short-circuits to False without invoking
    # the callable: the min_duration/min_words gates own that decision
    assert not is_backchannel_only("", flt)
    assert not is_backchannel_only("...", flt)
    assert seen == ["Okay.", "Okay, but wait.", "Okay."]


# endregion

# region drop decision


def _make_recognition(
    *,
    audio_transcript: str = "",
    overlapped: bool = True,
    agent_speaking: bool = True,
    agent_state: str = "speaking",
    backchannel_filter: Sequence[str] | Callable[[str], bool] | None = DEFAULT_BACKCHANNEL_PHRASES,
    turn_detection_mode: str | None = "vad",
) -> AudioRecognition:
    ar = object.__new__(AudioRecognition)
    ar._session = SimpleNamespace(  # type: ignore[assignment]
        options=SimpleNamespace(interruption={"backchannel_filter": backchannel_filter}),
        agent_state=agent_state,
    )
    ar._speech_overlapped_agent = overlapped
    ar._agent_speaking = agent_speaking
    ar._audio_transcript = audio_transcript
    ar._turn_detection_mode = turn_detection_mode
    return ar


def test_drop_decision() -> None:
    # entirely-backchannel utterance over agent speech → dropped
    assert _make_recognition()._should_drop_backchannel_final("Okay, thank you.")

    # overlap sampled at onset only (final landed after agent went quiet) → still dropped
    assert _make_recognition(
        agent_speaking=False, agent_state="listening"
    )._should_drop_backchannel_final("Okay.")

    # overlap live at final only (utterance started right before the reply launched)
    assert _make_recognition(overlapped=False)._should_drop_backchannel_final("Okay.")

    # real words already accumulated for the turn: a trailing backchannel chunk
    # is part of the sentence ("Wait, is that right? ... Yes.") — never dropped
    assert not _make_recognition(
        audio_transcript="Wait, is that right?"
    )._should_drop_backchannel_final("Yes.")

    # agent idle and listening at both boundaries → a real answer, never dropped
    assert not _make_recognition(
        overlapped=False, agent_speaking=False, agent_state="listening"
    )._should_drop_backchannel_final("Okay.")

    # feature off
    assert not _make_recognition(backchannel_filter=None)._should_drop_backchannel_final("Okay.")

    # real speech never dropped
    assert not _make_recognition()._should_drop_backchannel_final("Okay, but wait.")

    # a callable filter drives the same decision
    flt = _make_recognition(backchannel_filter=lambda text: "vale" in text.lower())
    assert flt._should_drop_backchannel_final("Vale, vale.")
    assert not flt._should_drop_backchannel_final("Una pregunta.")

    # manual turn detection: commit_user_turn is an explicit app decision and
    # commits even with an empty transcript — dropping would resolve that
    # commit to "" and generate a reply to nothing (the min_words gates
    # exempt manual mode for the same reason)
    assert not _make_recognition(turn_detection_mode="manual")._should_drop_backchannel_final(
        "Okay."
    )


# endregion

# region session behavior


def _story_actions(overlap_text: str) -> FakeActions:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(10.0)  # playout starts at 3.5s, ends at 13.5s
    actions.add_user_speech(5.0, 6.0, overlap_text, stt_delay=0.2)
    return actions


def _collect_session_events(
    session: AgentSession,
) -> tuple[list[PlaybackFinishedEvent], list[UserInputTranscribedEvent]]:
    playback_finished_events: list[PlaybackFinishedEvent] = []
    user_transcription_events: list[UserInputTranscribedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)
    session.on("user_input_transcribed", user_transcription_events.append)
    return playback_finished_events, user_transcription_events


async def test_backchannel_over_agent_speech_is_dropped() -> None:
    # min_words=1 keeps the acoustic-only cut (empty transcript) from firing,
    # so the decision reaches the transcript and the phrase filter judges it
    session = create_session(
        _story_actions("Okay, thank you."),
        turn_handling={
            "interruption": {"min_words": 1, "backchannel_filter": DEFAULT_BACKCHANNEL_PHRASES}
        },
    )
    agent = EchoAgent()
    playback_finished_events, user_transcription_events = _collect_session_events(session)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # the agent talked straight through the acknowledgment
    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False
    assert playback_finished_events[0].playback_position == pytest.approx(10.0, abs=0.3)

    # and the acknowledgment never became a user turn
    user_messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert [m.text_content for m in user_messages] == ["Tell me a story."]
    # the acknowledgment still finalizes as a transcription event: live
    # captions publish under one segment id until an is_final flushes it, so
    # swallowing the final would leave the caption stuck open and the next
    # utterance would merge into the same segment
    final_acks = [
        ev
        for ev in user_transcription_events
        if ev.is_final and "thank" in (ev.transcript or "").lower()
    ]
    assert len(final_acks) == 1


async def test_callable_filter_drops_backchannel() -> None:
    # the callable form gives the user full control over classification;
    # here composing the built-in matcher (partial=True so live interims
    # with a trailing phrase prefix defer the cut instead of committing it)
    def acknowledgments_only(text: str) -> bool:
        return is_backchannel_only(text, DEFAULT_BACKCHANNEL_PHRASES, partial=True)

    session = create_session(
        _story_actions("Okay, thank you."),
        turn_handling={
            "interruption": {"min_words": 1, "backchannel_filter": acknowledgments_only}
        },
    )
    agent = EchoAgent()
    playback_finished_events, _ = _collect_session_events(session)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is False

    user_messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert [m.text_content for m in user_messages] == ["Tell me a story."]


async def test_backchannel_interrupts_without_phrase_filter() -> None:
    # same timeline without the filter: the acknowledgment cuts the agent
    session = create_session(
        _story_actions("Okay, thank you."),
        turn_handling={"interruption": {"min_words": 1}},
    )
    playback_finished_events, _ = _collect_session_events(session)

    await asyncio.wait_for(run_session(session, EchoAgent()), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) >= 1
    assert playback_finished_events[0].interrupted is True


async def test_real_barge_in_still_interrupts() -> None:
    session = create_session(
        _story_actions("Wait, I have a question."),
        turn_handling={
            "interruption": {"min_words": 1, "backchannel_filter": DEFAULT_BACKCHANNEL_PHRASES}
        },
    )
    agent = EchoAgent()
    playback_finished_events, _ = _collect_session_events(session)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert len(playback_finished_events) >= 1
    assert playback_finished_events[0].interrupted is True

    user_messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert any("question" in (m.text_content or "") for m in user_messages)


async def test_backchannel_while_listening_commits_turn() -> None:
    # a bare acknowledgment while the agent is idle is a real answer
    # (e.g. to "does that work for you?") and must commit normally
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("The end.")
    actions.add_tts(1.0)  # playout 3.5s → 4.5s
    actions.add_user_speech(6.0, 6.5, "Okay.")
    actions.add_llm("Glad you liked it.")
    actions.add_tts(1.0)

    session = create_session(
        actions,
        turn_handling={
            "interruption": {"min_words": 1, "backchannel_filter": DEFAULT_BACKCHANNEL_PHRASES}
        },
    )
    agent = EchoAgent()

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    user_messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert [m.text_content for m in user_messages] == ["Tell me a story.", "Okay."]


# endregion
