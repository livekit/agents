from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import (
    Agent,
    ConversationItemAddedEvent,
    LanguageCode,
    UserTranscriptionTimeoutEvent,
)
from livekit.agents.llm import ChatMessage
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import AudioRecognition, _pending_segment_text

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0
# The interim lands before VAD end of speech and the empty final after it, as providers do.
STT_DELAY = 0.8
OPT_IN = {"commit_interim_on_empty_final": True}


def _agent() -> Agent:
    return Agent(instructions="You are a helpful assistant.")


def _user_texts(events: list[ConversationItemAddedEvent]) -> list[str | None]:
    return [
        ev.item.text_content
        for ev in events
        if isinstance(ev.item, ChatMessage) and ev.item.role == "user"
    ]


async def test_empty_final_leaves_turn_open_by_default() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Pick up.", stt_delay=STT_DELAY, final_transcript="")
    actions.add_llm("Great, pickup it is.")
    actions.add_tts(1.0)

    session = create_session(actions)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)

    t_origin = await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    # only the session close after the drain commits the trailing interim
    user_items = [ev for ev in items if isinstance(ev.item, ChatMessage) and ev.item.role == "user"]
    assert all(ev.created_at - t_origin > 4.0 for ev in user_items)


async def test_empty_final_commits_buffered_interim() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Pick up.", stt_delay=STT_DELAY, final_transcript="")
    actions.add_llm("Great, pickup it is.")
    actions.add_tts(1.0)

    session = create_session(actions, extra_kwargs=OPT_IN)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)

    t_origin = await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert _user_texts(items) == ["Pick up."]
    # committed at end of turn, not by the session close that follows the drain
    user_item = next(
        ev for ev in items if isinstance(ev.item, ChatMessage) and ev.item.role == "user"
    )
    assert user_item.created_at - t_origin < 4.0


async def test_empty_final_promotes_cumulative_interim_over_chunked_preflight() -> None:
    # AssemblyAI's plugin sends the turn's words as the interim and only the words since the
    # last preflight as the preflight
    actions = FakeActions()
    actions.add_user_speech(
        0.5,
        1.5,
        "Pick up.",
        stt_delay=STT_DELAY,
        final_transcript="",
        preflight_transcript="up.",
    )
    actions.add_llm("Great, pickup it is.")
    actions.add_tts(1.0)

    session = create_session(actions, extra_kwargs=OPT_IN)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)

    t_origin = await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert _user_texts(items) == ["Pick up."]
    user_item = next(
        ev for ev in items if isinstance(ev.item, ChatMessage) and ev.item.role == "user"
    )
    assert user_item.created_at - t_origin < 4.0


@pytest.mark.parametrize(
    ("transcript", "preflight"),
    [
        pytest.param("", "Pick up.", id="no-interim"),
        pytest.param("Pick up please.", "Pick up please.", id="preflight-adds-words"),
    ],
)
async def test_empty_final_promotes_full_segment_preflight(transcript: str, preflight: str) -> None:
    # the fake interim is the first two words of `transcript`
    actions = FakeActions()
    actions.add_user_speech(
        0.5,
        1.5,
        transcript,
        stt_delay=STT_DELAY,
        final_transcript="",
        preflight_transcript=preflight,
    )
    actions.add_llm("Great, pickup it is.")
    actions.add_tts(1.0)

    session = create_session(actions, extra_kwargs=OPT_IN)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)

    t_origin = await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert _user_texts(items) == [preflight]
    user_item = next(
        ev for ev in items if isinstance(ev.item, ChatMessage) and ev.item.role == "user"
    )
    assert user_item.created_at - t_origin < 4.0


async def test_promoted_interim_cancels_transcription_timeout() -> None:
    # a long endpointing delay holds the commit past the timeout
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Pick up.", stt_delay=STT_DELAY, final_transcript="")
    actions.add_llm("Great, pickup it is.")
    actions.add_tts(1.0)

    session = create_session(
        actions,
        extra_kwargs={**OPT_IN, "transcription_timeout": 2.0},
        turn_handling={"endpointing": {"min_delay": 4.0, "max_delay": 6.0}},
    )
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent(), drain_delay=10), timeout=SESSION_TIMEOUT)

    assert events == []


async def test_empty_final_without_interim_does_not_commit() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "", stt_delay=STT_DELAY, final_transcript="")

    session = create_session(actions, extra_kwargs=OPT_IN)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert _user_texts(items) == []


@pytest.mark.parametrize(
    ("has_vad", "speech_start_time"),
    [
        pytest.param(True, None, id="vad-never-heard-speech"),
        pytest.param(False, 1.0, id="no-vad"),
    ],
)
def test_empty_final_keeps_interim_without_vad_speech(
    has_vad: bool, speech_start_time: float | None
) -> None:
    # without VAD speech in the turn, a retracted interim is more likely noise
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._hooks = MagicMock()
    ar._stt_pipeline = None
    ar._vad = MagicMock() if has_vad else None
    ar._speech_start_time = time.time() if speech_start_time is not None else None
    ar._last_speaking_time = None
    ar._turn_detection_mode = "vad" if has_vad else "stt"
    ar._last_language = None
    ar._final_transcript_received = asyncio.Event()
    ar._audio_transcript = ""
    ar._audio_interim_transcript = "uh"
    ar._last_interim_text = "uh"
    ar._last_preflight_text = ""
    ar._preflight_is_latest = False

    ar._process_stt_event(
        SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text="", language=LanguageCode(""))],
        )
    )

    assert ar._audio_transcript == ""
    ar._hooks.on_final_transcript.assert_not_called()


@pytest.mark.parametrize(
    ("interim", "preflight", "preflight_is_latest", "expected"),
    [
        pytest.param("Pick up", "up", True, "Pick up", id="chunked-preflight"),
        pytest.param("", "Pick up", True, "Pick up", id="preflight-without-interim"),
        pytest.param("Pick", "Pick up", True, "Pick up", id="preflight-adds-words"),
        pytest.param("Pick up please", "Pick up", False, "Pick up please", id="interim-grows"),
        pytest.param("Pick", "Pick up", False, "Pick", id="interim-retracts"),
        pytest.param("", "Pick up", False, "", id="interim-retracts-all"),
    ],
)
def test_pending_segment_text(
    interim: str, preflight: str, preflight_is_latest: bool, expected: str
) -> None:
    assert _pending_segment_text(interim, preflight, preflight_is_latest) == expected
