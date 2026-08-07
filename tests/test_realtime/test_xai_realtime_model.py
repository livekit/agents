from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)

from livekit.agents import llm, utils
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.plugins.openai.realtime.realtime_model import (
    _DiscardedGeneration,
    _MessageGeneration,
)
from livekit.plugins.xai.realtime.realtime_model import RealtimeSession

pytestmark = pytest.mark.unit


def _make_session() -> tuple[RealtimeSession, list[llm.InputTranscriptionCompleted]]:
    # no network or model is needed, and a discarded generation skips the base bookkeeping
    session = RealtimeSession.__new__(RealtimeSession)
    session._remote_chat_ctx = RemoteChatContext()  # type: ignore[attr-defined]
    session._current_generation = _DiscardedGeneration()  # type: ignore[attr-defined]
    session._opts = SimpleNamespace(modalities=["audio", "text"])  # type: ignore[assignment]
    session._item_create_future = {}  # type: ignore[attr-defined]
    session._msg_ch = utils.aio.Chan()  # type: ignore[attr-defined]
    session._reset_input_turn_state()

    emitted: list[llm.InputTranscriptionCompleted] = []

    def _emit(name: str, ev: object) -> None:
        if name == "input_audio_transcription_completed":
            emitted.append(ev)  # type: ignore[arg-type]

    session.emit = _emit  # type: ignore[method-assign,assignment]
    return session, emitted


def _finals(emitted: list[llm.InputTranscriptionCompleted]) -> list[tuple[str, str]]:
    return [(ev.item_id, ev.transcript) for ev in emitted if ev.is_final]


def _transcript(session: RealtimeSession, item_id: str, text: str, *, status: str | None) -> None:
    payload: dict = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "evt",
        "item_id": item_id,
        "content_index": 0,
        "transcript": text,
    }
    if status is not None:
        payload["status"] = status
    session._handle_conversion_item_input_audio_transcription_completed(
        ConversationItemInputAudioTranscriptionCompletedEvent.construct(**payload)
    )


def _commit(session: RealtimeSession, item_id: str, text: str) -> None:
    """One audio-buffer commit, which re-sends the whole transcript of the item."""
    _transcript(session, item_id, text, status="completed")


def _speech_started(session: RealtimeSession, item_id: str) -> None:
    session._handle_input_audio_buffer_speech_started(
        InputAudioBufferSpeechStartedEvent.construct(
            type="input_audio_buffer.speech_started",
            event_id="evt",
            item_id=item_id,
            audio_start_ms=0,
        )
    )


def _agent_speaks(session: RealtimeSession) -> None:
    """The first audio chunk of a response, which ends the turn."""
    session._handle_response_audio_delta(
        ResponseAudioDeltaEvent.construct(
            type="response.output_audio.delta",
            event_id="evt",
            item_id="reply",
            response_id="resp",
            output_index=0,
            content_index=0,
            delta="",
        )
    )


def _mirror(session: RealtimeSession, *items: tuple[str, str, str]) -> None:
    """Fill the remote chat context with (item_id, role, text), in order."""
    previous: str | None = None
    for item_id, role, text in items:
        session._remote_chat_ctx.insert(
            previous, llm.ChatMessage(id=item_id, role=role, content=[text])
        )
        previous = item_id


def _item_added(session: RealtimeSession, item_id: str, *, after: str | None) -> None:
    session._handle_conversion_item_added(
        ConversationItemAdded.construct(
            type="conversation.item.added",
            event_id="evt",
            previous_item_id=after,
            item={
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
    )


def _response_created(session: RealtimeSession) -> None:
    session._handle_response_created(
        ResponseCreatedEvent.construct(
            type="response.created",
            event_id="evt",
            response={"id": "resp", "object": "realtime.response", "output": []},
        )
    )


def _reply_item_announced(session: RealtimeSession, item_id: str, *, after: str) -> None:
    """Announce the reply item of a response that can then be abandoned unspoken."""
    session._remote_chat_ctx.insert(
        after, llm.ChatMessage(id=item_id, role="assistant", content=[])
    )
    session._current_generation.messages[item_id] = _MessageGeneration(  # type: ignore[union-attr]
        message_id=item_id,
        text_ch=utils.aio.Chan(),
        audio_ch=utils.aio.Chan(),
        modalities=asyncio.Future(),
    )


# -- when a turn becomes final ----------------------------------------------------------------


def test_in_progress_transcripts_are_never_final() -> None:
    session, emitted = _make_session()

    for partial in ["what is", "what is my", "what is my name"]:
        _transcript(session, "item_1", partial, status="in_progress")

    assert _finals(emitted) == []
    assert [ev.transcript for ev in emitted] == ["what is", "what is my", "what is my name"]


def test_committed_transcript_is_final_once_the_agent_answers() -> None:
    session, emitted = _make_session()

    _commit(session, "item_1", "what is my name")
    assert _finals(emitted) == [], "a commit is not the end of the turn"

    _agent_speaks(session)
    assert _finals(emitted) == [("item_1", "what is my name")]


def test_missing_status_is_treated_as_committed() -> None:
    # an event without a `status` field is held like a committed one
    session, emitted = _make_session()

    _transcript(session, "item_1", "what is my name", status=None)
    _agent_speaks(session)

    assert _finals(emitted) == [("item_1", "what is my name")]


def test_text_only_response_ends_the_turn() -> None:
    session, emitted = _make_session()

    _commit(session, "item_1", "what is my name")
    session._handle_response_text_delta(
        ResponseTextDeltaEvent.construct(
            type="response.output_text.delta",
            event_id="evt",
            item_id="reply",
            response_id="resp",
            output_index=0,
            content_index=0,
            delta="your",
        )
    )

    assert _finals(emitted) == [("item_1", "what is my name")]


def test_pause_within_a_turn_yields_one_final() -> None:
    # the item outlives the pause and is re-transcribed from the start, "ones" to "once",
    # so there is no delta to send and only the last commit is the final (#6710)
    session, emitted = _make_session()

    _commit(session, "item_1", "Hello, how are you? Last ones.")
    _commit(session, "item_1", "Hello, how are you? Last once. I paid twice.")
    _agent_speaks(session)

    assert _finals(emitted) == [("item_1", "Hello, how are you? Last once. I paid twice.")]


def test_abandoned_response_does_not_end_the_turn() -> None:
    # a response the user talks over is dropped unspoken, so only a speaking one ends the turn
    session, emitted = _make_session()

    _commit(session, "item_1", "hello")
    _commit(session, "item_1", "hello and one more thing")
    assert _finals(emitted) == []

    _agent_speaks(session)
    assert _finals(emitted) == [("item_1", "hello and one more thing")]


def test_next_turn_finalizes_one_the_agent_never_answered() -> None:
    # the rescue when no response speaks: a different item id at speech start ends the turn
    session, emitted = _make_session()

    _commit(session, "item_1", "are you there")
    _speech_started(session, "item_1")
    assert _finals(emitted) == [], "the same item means the turn resumed"

    _speech_started(session, "item_2")
    assert _finals(emitted) == [("item_1", "are you there")]


def test_a_new_item_transcript_finalizes_the_previous_turn() -> None:
    session, emitted = _make_session()

    _commit(session, "item_1", "hello")
    _commit(session, "item_2", "are you there")

    assert _finals(emitted) == [("item_1", "hello")]


def test_reconnect_delivers_the_held_transcript() -> None:
    # item ids do not survive a reconnect, so deliver the turn before they go, and into the
    # mirror that is replayed to the new session
    session, emitted = _make_session()
    _mirror(session, ("item_1", "user", ""))

    _commit(session, "item_1", "are you still there")
    session._reset_input_turn_state()

    assert _finals(emitted) == [("item_1", "are you still there")]
    assert session._pending_transcription is None
    remote = session._remote_chat_ctx.get("item_1")
    assert remote is not None
    assert remote.item.content == ["are you still there"], "the turn is replayed without it"


def test_speech_onset_survives_a_resumed_turn() -> None:
    # the final reads the onset once, so a resumed segment must not overwrite it
    session, _ = _make_session()

    _speech_started(session, "item_1")
    first = session._input_speech_started_at["item_1"]
    _speech_started(session, "item_1")

    assert session._input_speech_started_at["item_1"] == first


# -- keeping the mirror and the reconcile honest -----------------------------------------------


def test_remote_item_holds_the_transcript_once() -> None:
    # the mirror is replayed on reconnect, so an append per commit resends the turn each time
    session, _ = _make_session()
    _mirror(session, ("item_1", "user", ""))

    _commit(session, "item_1", "Hello, how are you? Last ones.")
    _commit(session, "item_1", "Hello, how are you? Last once. I paid twice.")
    _agent_speaks(session)

    remote = session._remote_chat_ctx.get("item_1")
    assert remote is not None
    assert remote.item.content == ["Hello, how are you? Last once. I paid twice."]


def test_held_turn_is_not_deleted_from_the_server() -> None:
    # a held turn is missing from the agent's context, and a stale read deletes it
    session, _ = _make_session()
    _mirror(session, ("item_1", "user", ""), ("stale", "assistant", "dropped"))
    _commit(session, "item_1", "are you there")

    events = session._create_update_chat_ctx_events(llm.ChatContext.empty())

    deleted = {ev.item_id for ev in events if isinstance(ev, ConversationItemDeleteEvent)}
    assert "item_1" not in deleted
    assert "stale" in deleted, "an item the agent really dropped still goes"


def test_held_turn_keeps_the_reply_anchored_behind_it() -> None:
    # the reply item opens while the turn is held, so a stand-in at the end reorders them
    session, _ = _make_session()
    _mirror(
        session,
        ("earlier", "assistant", "hi"),
        ("item_1", "user", ""),
        ("reply", "assistant", "the full reply"),
    )
    _commit(session, "item_1", "are you there")

    agent_ctx = llm.ChatContext.empty()
    agent_ctx.items.append(llm.ChatMessage(id="earlier", role="assistant", content=["hi"]))
    agent_ctx.items.append(llm.ChatMessage(id="reply", role="assistant", content=["the trunc"]))
    events = session._create_update_chat_ctx_events(agent_ctx)

    created = {
        ev.item.id: ev.previous_item_id
        for ev in events
        if isinstance(ev, ConversationItemCreateEvent)
    }
    assert created == {"reply": "item_1"}, "the truncated reply must stay behind the user turn"


def test_an_item_anchored_to_an_unannounced_one_is_appended() -> None:
    # xAI anchors an item to a segment it dropped unannounced, so append rather than strand
    # everything behind it
    session, _ = _make_session()
    _mirror(session, ("item_1", "user", "hello"))

    _item_added(session, "item_2", after="never_announced")
    _item_added(session, "reply", after="item_2")

    assert [item.id for item in session._remote_chat_ctx.to_chat_ctx().items] == [
        "item_1",
        "item_2",
        "reply",
    ]


# -- dropping a response xAI abandoned ---------------------------------------------------------


async def test_interrupting_discards_a_response_that_never_spoke() -> None:
    # the next response.created lands after the commit, which a pause puts past the timeout
    session, _ = _make_session()
    _mirror(session, ("item_1", "user", "hello"))
    _response_created(session)
    _reply_item_announced(session, "phantom", after="item_1")
    generation = session._current_generation

    session.interrupt()

    assert generation._done_fut.done(), "the speech would wait out the timeout otherwise"
    # xAI drops such an item only sometimes, and anchors later ones to it either way
    assert session._remote_chat_ctx.get("phantom") is not None


async def test_a_speaking_response_is_left_alone() -> None:
    # a real barge-in goes through the normal interruption path, not the discard
    session, _ = _make_session()
    _mirror(session, ("item_1", "user", "hello"))
    _response_created(session)
    _reply_item_announced(session, "real", after="item_1")
    session._response_spoke = True  # what the first output delta sets
    generation = session._current_generation

    session.interrupt()

    assert not generation._done_fut.done()


async def test_a_silent_response_survives_the_user_speaking_over_it() -> None:
    # a speech start is no proof of a drop, and a discard there loses the reply still coming
    session, _ = _make_session()
    _mirror(session, ("item_1", "user", "hello"))
    _response_created(session)
    _reply_item_announced(session, "thinking", after="item_1")
    generation = session._current_generation

    _speech_started(session, "item_1")

    assert not generation._done_fut.done()
