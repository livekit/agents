from __future__ import annotations

import pytest
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    RealtimeConversationItemUserMessage,
)

from livekit.agents import llm
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.plugins.xai.realtime.realtime_model import RealtimeSession

pytestmark = pytest.mark.unit


def _make_session() -> tuple[RealtimeSession, list[llm.InputTranscriptionCompleted]]:
    # build a session instance without going through __init__ (no network/model needed)
    session = RealtimeSession.__new__(RealtimeSession)
    session._remote_chat_ctx = RemoteChatContext()  # type: ignore[attr-defined]
    session._input_transcript_accumulators = {}  # type: ignore[attr-defined]
    session._item_create_future = {}  # type: ignore[attr-defined]

    emitted: list[llm.InputTranscriptionCompleted] = []
    session.emit = lambda name, ev: emitted.append(ev)  # type: ignore[method-assign,assignment]
    return session, emitted


def _item_added_event(*, item_id: str, previous_item_id: str | None) -> ConversationItemAdded:
    item = RealtimeConversationItemUserMessage.construct(
        id=item_id,
        type="message",
        role="user",
        content=[{"type": "input_text", "text": item_id}],
    )
    return ConversationItemAdded.construct(
        event_id="evt",
        type="conversation.item.added",
        item=item,
        previous_item_id=previous_item_id,
    )


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


def test_item_added_with_missing_anchor_self_heals_to_tail() -> None:
    # After an interruption's delete/truncate traffic desyncs the local chat context, a
    # server `conversation.item.added` can reference an item we no longer have. Anchoring
    # to a missing item made the base handler drop the item; instead we append at the tail.
    session, _ = _make_session()
    session._remote_chat_ctx.insert(None, llm.ChatMessage(id="a", role="user", content=["hi"]))
    session._remote_chat_ctx.insert("a", llm.ChatMessage(id="b", role="user", content=["there"]))

    session._handle_conversion_item_added(_item_added_event(item_id="c", previous_item_id="ghost"))

    # The item is kept (not dropped) and lands at the tail.
    assert session._remote_chat_ctx.get("c") is not None
    assert session._remote_chat_ctx._tail.item.id == "c"


def test_item_added_missing_anchor_does_not_cascade() -> None:
    # The real damage was the cascade: once one item is dropped, every later item that
    # anchors to it is dropped too. Self-healing keeps the chain intact.
    session, _ = _make_session()
    session._remote_chat_ctx.insert(None, llm.ChatMessage(id="a", role="user", content=["hi"]))

    session._handle_conversion_item_added(_item_added_event(item_id="c", previous_item_id="ghost"))
    # `d` anchors to `c`, which only exists because `c` self-healed above.
    session._handle_conversion_item_added(_item_added_event(item_id="d", previous_item_id="c"))

    assert session._remote_chat_ctx.get("c") is not None
    assert session._remote_chat_ctx.get("d") is not None
