from __future__ import annotations

import pytest
from openai.types.realtime import ConversationItemInputAudioTranscriptionCompletedEvent

from livekit.agents import llm
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.plugins.xai.realtime.realtime_model import RealtimeSession

pytestmark = pytest.mark.unit


def _make_session() -> tuple[RealtimeSession, list[llm.InputTranscriptionCompleted]]:
    # build a session instance without going through __init__ (no network/model needed)
    session = RealtimeSession.__new__(RealtimeSession)
    session._remote_chat_ctx = RemoteChatContext()  # type: ignore[attr-defined]
    session._input_transcript_accumulators = {}  # type: ignore[attr-defined]

    emitted: list[llm.InputTranscriptionCompleted] = []
    session.emit = lambda name, ev: emitted.append(ev)  # type: ignore[method-assign,assignment]
    return session, emitted


def _completed_event(
    *, item_id: str, transcript: str, status: str | None
) -> ConversationItemInputAudioTranscriptionCompletedEvent:
    payload: dict = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "evt",
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }
    if status is not None:
        payload["status"] = status
    return ConversationItemInputAudioTranscriptionCompletedEvent.construct(**payload)


def test_in_progress_transcript_emitted_as_interim() -> None:
    session, emitted = _make_session()

    session._handle_conversion_item_input_audio_transcription_completed(
        _completed_event(item_id="item_1", transcript="what is", status="in_progress")
    )

    assert len(emitted) == 1
    assert emitted[0].item_id == "item_1"
    assert emitted[0].transcript == "what is"
    assert emitted[0].is_final is False


def test_multiple_in_progress_transcripts_never_final() -> None:
    session, emitted = _make_session()

    for partial in ["what is", "what is my", "what is my name"]:
        session._handle_conversion_item_input_audio_transcription_completed(
            _completed_event(item_id="item_1", transcript=partial, status="in_progress")
        )

    assert len(emitted) == 3
    assert all(ev.is_final is False for ev in emitted)
    assert [ev.transcript for ev in emitted] == [
        "what is",
        "what is my",
        "what is my name",
    ]


def test_completed_transcript_emitted_as_final() -> None:
    session, emitted = _make_session()

    session._handle_conversion_item_input_audio_transcription_completed(
        _completed_event(item_id="item_1", transcript="what is my name", status="completed")
    )

    assert len(emitted) == 1
    assert emitted[0].item_id == "item_1"
    assert emitted[0].transcript == "what is my name"
    assert emitted[0].is_final is True


def test_missing_status_defaults_to_final() -> None:
    # be defensive: an event without a `status` field is treated as final
    session, emitted = _make_session()

    session._handle_conversion_item_input_audio_transcription_completed(
        _completed_event(item_id="item_1", transcript="what is my name", status=None)
    )

    assert len(emitted) == 1
    assert emitted[0].is_final is True
