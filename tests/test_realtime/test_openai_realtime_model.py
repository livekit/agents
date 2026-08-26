from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from openai.types.beta.realtime.session import TurnDetection as BetaTurnDetection
from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    RealtimeErrorEvent,
)
from openai.types.realtime.audio_transcription import AudioTranscription
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit.agents import llm
from livekit.agents._exceptions import APIError
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.agents.utils import is_given
from livekit.plugins.openai.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
    _is_fatal_error,
)

pytestmark = pytest.mark.unit


def test_update_options_only_propagates_given_turn_detection() -> None:
    # RealtimeModel.update_options must not force-sync turn_detection to sessions when the
    # caller didn't set it, else a session that opted out of server-side turn detection gets
    # server VAD re-enabled by an unrelated change (e.g. voice). An explicit value still
    # propagates, so callers can re-enable it. (PR #6495 review)
    class _StubSession:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def update_options(self, **kw: object) -> None:
            self.calls.append(kw)

    model = RealtimeModel(api_key="fake")
    stub = _StubSession()
    model._sessions.add(cast(RealtimeSession, stub))

    model.update_options(voice="verse")
    assert not is_given(stub.calls[-1]["turn_detection"])

    model.update_options(turn_detection=model._opts.turn_detection)
    assert is_given(stub.calls[-1]["turn_detection"])


def test_with_azure_preserves_can_disable_turn_detection() -> None:
    # with_azure fills in a default turn_detection, but the framework must still be allowed to
    # auto-disable server-side turn detection when the caller didn't configure it. An explicit
    # value is respected. (PR #6495 review)
    default_td = RealtimeModel.with_azure(
        azure_deployment="dep", api_key="fake", base_url="https://example.com/openai"
    )
    assert default_td.capabilities.turn_detection is True
    assert default_td.capabilities.can_disable_turn_detection is True

    explicit_off = RealtimeModel.with_azure(
        azure_deployment="dep",
        api_key="fake",
        base_url="https://example.com/openai",
        turn_detection=None,
    )
    assert explicit_off.capabilities.turn_detection is False
    assert explicit_off.capabilities.can_disable_turn_detection is False


def test_create_response_false_reports_client_side_turn_taking() -> None:
    # server VAD with create_response=False commits and transcribes the audio server-side but
    # leaves the reply to the client, so it must not count as server-side turn detection —
    # otherwise allow_interruptions=False is rejected (issue #6635)
    manual_reply = RealtimeModel(
        api_key="fake",
        turn_detection=ServerVad(type="server_vad", create_response=False),
    )
    assert manual_reply.capabilities.turn_detection is False

    auto_reply = RealtimeModel(
        api_key="fake",
        turn_detection=ServerVad(type="server_vad"),
    )
    assert auto_reply.capabilities.turn_detection is True

    assert RealtimeModel(api_key="fake").capabilities.turn_detection is True


def test_create_response_false_warns_when_the_server_still_interrupts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # interrupt_response is a separate switch, left on the server keeps cancelling its response
    # on user speech while the client believes it owns interruptions
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.openai"):
        RealtimeModel(
            api_key="fake",
            turn_detection=ServerVad(type="server_vad", create_response=False),
        )
    assert "pass interrupt_response=False" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.openai"):
        RealtimeModel(
            api_key="fake",
            turn_detection=ServerVad(
                type="server_vad", create_response=False, interrupt_response=False
            ),
        )
    assert caplog.text == ""


def test_update_options_keeps_derived_capabilities_in_sync() -> None:
    # a stale capability reports the server owning the turn after the caller handed turn
    # taking to the client, and AgentActivity then rejects allow_interruptions=False
    model = RealtimeModel(api_key="fake", turn_detection=ServerVad(type="server_vad"))
    assert model.capabilities.turn_detection is True
    assert model.capabilities.user_transcription is True

    model.update_options(
        turn_detection=ServerVad(
            type="server_vad", create_response=False, interrupt_response=False
        ),
        input_audio_transcription=None,
    )
    assert model.capabilities.turn_detection is False
    assert model.capabilities.user_transcription is False

    model.update_options(
        turn_detection=ServerVad(type="server_vad"),
        input_audio_transcription=AudioTranscription(model="whisper-1"),
    )
    assert model.capabilities.turn_detection is True
    assert model.capabilities.user_transcription is True


def test_update_options_leaves_derived_capabilities_alone_when_unset() -> None:
    # an unrelated update must not resync a model that opted out of server-side turn taking
    model = RealtimeModel(
        api_key="fake",
        turn_detection=ServerVad(
            type="server_vad", create_response=False, interrupt_response=False
        ),
    )
    model.update_options(voice="marin")
    assert model.capabilities.turn_detection is False


def _capabilities_session(model: RealtimeModel) -> RealtimeSession:
    # only the state update_options touches; a real session would open a websocket
    return cast(
        RealtimeSession,
        SimpleNamespace(
            _opts=replace(model._opts),
            _capabilities=replace(model.capabilities),
            send_event=lambda event: None,
            _wrap_session_update=lambda event_id, session: session,
        ),
    )


def test_session_update_options_keeps_capabilities_on_the_session() -> None:
    # turn detection is per session: one session handing turn taking to the client must not
    # change what the model, or any other session on it, reports
    model = RealtimeModel(api_key="fake", turn_detection=ServerVad(type="server_vad"))
    session = _capabilities_session(model)

    RealtimeSession.update_options(
        session,
        turn_detection=ServerVad(
            type="server_vad", create_response=False, interrupt_response=False
        ),
        input_audio_transcription=None,
    )

    assert session._capabilities.turn_detection is False
    assert session._capabilities.user_transcription is False
    assert model.capabilities.turn_detection is True
    assert model.capabilities.user_transcription is True


def test_legacy_turn_detection_keeps_interrupt_response() -> None:
    # the deprecated session.TurnDetection carries interrupt_response for server_vad too;
    # dropping it silently re-enabled server-side interruption
    model = RealtimeModel(
        api_key="fake",
        turn_detection=BetaTurnDetection(
            type="server_vad", create_response=False, interrupt_response=False
        ),
    )
    assert model._opts.turn_detection == ServerVad(
        type="server_vad", create_response=False, interrupt_response=False
    )


def test_update_chat_ctx_deletes_empty_remote_items() -> None:
    remote_ctx = RemoteChatContext()
    audio_item = llm.ChatMessage(id="audio_item", role="user", content=[])
    kept_item = llm.ChatMessage(id="assistant_item", role="assistant", content=["kept"])
    remote_ctx.insert(None, audio_item)
    remote_ctx.insert(audio_item.id, kept_item)

    session = cast(RealtimeSession, SimpleNamespace(_remote_chat_ctx=remote_ctx))
    events = RealtimeSession._create_update_chat_ctx_events(
        session,
        llm.ChatContext(items=[kept_item]),
    )

    delete_ids = [
        getattr(event, "item_id", None)
        for event in events
        if getattr(event, "type", None) == "conversation.item.delete"
    ]
    assert delete_ids == ["audio_item"]


# --------------------------------------------------------------------------- #
# fatal error classification: a fatal error must break the recv loop so that
# _main_task stops reconnecting (raised as APIError(retryable=False))
# --------------------------------------------------------------------------- #


def test_is_fatal_error_matches_known_codes() -> None:
    assert _is_fatal_error(SimpleNamespace(code="insufficient_quota"))
    assert _is_fatal_error(SimpleNamespace(code=None, type="invalid_api_key"))
    assert not _is_fatal_error(SimpleNamespace(code="server_error"))
    assert not _is_fatal_error(SimpleNamespace())
    assert not _is_fatal_error(None)


def _handle_error_session(
    capture: dict[str, object], *, turn_detection: ServerVad | None = None
) -> RealtimeSession:
    return cast(
        RealtimeSession,
        SimpleNamespace(
            _realtime_model=SimpleNamespace(_provider_label="openai"),
            _opts=SimpleNamespace(turn_detection=turn_detection),
            _chat_ctx_event_futures={},
            _response_created_futures={},
            _emit_error=lambda error, recoverable: capture.update(recoverable=recoverable),
        ),
    )


def test_handle_error_raises_on_fatal() -> None:
    # a fatal code is raised (not emitted here): the recv loop re-raises it so
    # _main_task emits it once with recoverable=False and stops reconnecting
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        error=SimpleNamespace(event_id=None, message="quota exceeded", code="insufficient_quota")
    )
    with pytest.raises(APIError) as exc_info:
        RealtimeSession._handle_error(session, event)
    assert exc_info.value.retryable is False
    assert captured == {}  # not emitted by the handler; _main_task owns the emit


def test_handle_error_emits_transient_as_recoverable() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        error=SimpleNamespace(event_id=None, message="server hiccup", code="server_error")
    )
    RealtimeSession._handle_error(session, event)
    assert captured["recoverable"] is True


def test_handle_error_ignores_cancellation_failed() -> None:
    captured: dict[str, object] = {}
    event = SimpleNamespace(
        error=SimpleNamespace(event_id=None, message="Cancellation failed: no response")
    )
    RealtimeSession._handle_error(_handle_error_session(captured), event)
    assert captured == {}  # early return, nothing emitted


def _empty_commit_event() -> SimpleNamespace:
    return SimpleNamespace(
        error=SimpleNamespace(
            event_id=None,
            message="Error committing input audio buffer: buffer too small.",
            code="input_audio_buffer_commit_empty",
        )
    )


def test_handle_error_ignores_empty_commit_with_server_vad() -> None:
    # our commit raced the server VAD's own; the server owns the buffer, nothing to report
    captured: dict[str, object] = {}
    session = _handle_error_session(captured, turn_detection=ServerVad(type="server_vad"))
    RealtimeSession._handle_error(session, _empty_commit_event())
    assert captured == {}


def test_handle_error_reports_empty_commit_without_server_vad() -> None:
    # nothing else commits the buffer, so an empty commit is a real bug worth surfacing
    captured: dict[str, object] = {}
    RealtimeSession._handle_error(_handle_error_session(captured), _empty_commit_event())
    assert captured["recoverable"] is True


def test_response_done_failed_fatal_raises() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        response=SimpleNamespace(
            id="resp_1",
            status="failed",
            status_details=SimpleNamespace(
                error=SimpleNamespace(type="insufficient_quota", code="insufficient_quota")
            ),
        )
    )
    with pytest.raises(APIError) as exc_info:
        RealtimeSession._handle_response_done_but_not_complete(session, event)
    assert exc_info.value.retryable is False
    assert captured == {}


def test_response_done_failed_transient_stays_recoverable() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        response=SimpleNamespace(
            id="resp_1",
            status="failed",
            status_details=SimpleNamespace(
                error=SimpleNamespace(type="invalid_request_error", code="rate_limit_exceeded")
            ),
        )
    )
    RealtimeSession._handle_response_done_but_not_complete(session, event)
    assert captured["recoverable"] is True


def _chat_ctx_update_session() -> RealtimeSession:
    """A session mid-update, with the delete and the create of one updated item in flight."""
    session = _handle_error_session({})
    session._item_delete_future = {"item_1": asyncio.Future()}
    session._item_create_future = {"item_1": asyncio.Future()}
    session._chat_ctx_event_futures = {
        "chat_ctx_delete_abc": session._item_delete_future["item_1"],
        "chat_ctx_create_abc": session._item_create_future["item_1"],
    }
    return session


def _rejection(event_id: str, code: str = "invalid_request_error") -> RealtimeErrorEvent:
    return RealtimeErrorEvent.construct(
        type="error",
        event_id=event_id,
        error={
            "message": "Item not found: item_1",
            "type": "invalid_request_error",
            "code": code,
            "event_id": event_id,
        },
    )


async def test_a_rejected_chat_ctx_event_releases_its_waiter() -> None:
    # a rejected delete gets an error instead of conversation.item.deleted, so nothing else
    # settles the future that update_chat_ctx awaits inside the speech
    session = _chat_ctx_update_session()
    waiter = session._item_delete_future["item_1"]

    RealtimeSession._handle_error(session, _rejection("chat_ctx_delete_abc"))

    assert waiter.done(), "update_chat_ctx would wait out its timeout, and the speech with it"
    assert isinstance(waiter.exception(), llm.RealtimeError)


async def test_a_rejected_waiter_is_not_settled_twice() -> None:
    # the waiter stays parked under its item id, where a late reply would set a result on it
    session = _chat_ctx_update_session()
    session._remote_chat_ctx = RemoteChatContext()
    session._input_transcript_accumulators = {}
    session._input_speech_started_at = {}
    waiter = session._item_delete_future["item_1"]
    RealtimeSession._handle_error(session, _rejection("chat_ctx_delete_abc"))

    RealtimeSession._handle_conversion_item_deleted(
        session,
        ConversationItemDeletedEvent.construct(
            type="conversation.item.deleted", event_id="evt", item_id="item_1"
        ),
    )

    assert isinstance(waiter.exception(), llm.RealtimeError)


async def test_an_error_outliving_its_update_is_still_reported() -> None:
    # an error that lands on a waiter the update already retired settles nothing, so it has to
    # go down the ordinary path rather than pass for a rejected item
    session = RealtimeSession.__new__(RealtimeSession)
    session._realtime_model = SimpleNamespace(_provider_label="openai", _label="openai")  # type: ignore[assignment]
    session._opts = SimpleNamespace(turn_detection=None)  # type: ignore[assignment]
    session._update_chat_ctx_lock = asyncio.Lock()
    session._remote_chat_ctx = RemoteChatContext()
    session._item_delete_future = {}
    session._item_create_future = {}
    session._chat_ctx_event_futures = {}
    session._response_created_futures = {}
    sent: list[ConversationItemCreateEvent] = []
    session.send_event = sent.append  # type: ignore[method-assign,assignment]
    errors: list[llm.RealtimeModelError] = []
    session.emit = lambda name, ev: errors.append(ev)  # type: ignore[method-assign,assignment]

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello", id="item_1")
    update = asyncio.create_task(session.update_chat_ctx(chat_ctx))
    await asyncio.sleep(0)  # let it send the create and start waiting on it
    assert (waiter := session._item_create_future.get("item_1")) is not None
    waiter.set_result(None)  # what conversation.item.added does
    await update

    session._handle_error(_rejection(sent[0].event_id))

    assert [e.recoverable for e in errors] == [True], "swallowed by a retired waiter"


async def test_a_duplicate_item_id_settles_its_waiter_as_a_no_op() -> None:
    # the item is already on the conversation, which is all the create asked for; failing the
    # waiter would report a rejection for an update that got what it wanted
    session = _chat_ctx_update_session()
    waiter = session._item_create_future["item_1"]

    RealtimeSession._handle_error(
        session, _rejection("chat_ctx_create_abc", code="item_create_duplicate_item_id")
    )

    assert waiter.done() and waiter.exception() is None


async def test_a_rejection_leaves_its_sibling_event_alone() -> None:
    # an updated item is deleted and created under one id, and the create can still land
    session = _chat_ctx_update_session()
    sibling = session._item_create_future["item_1"]

    RealtimeSession._handle_error(session, _rejection("chat_ctx_delete_abc"))

    assert not sibling.done()
    assert session._item_create_future["item_1"] is sibling


async def test_a_fatal_error_on_a_chat_ctx_event_still_ends_the_session() -> None:
    # the error names the event that drew it, and an exhausted quota must stop the session
    session = _chat_ctx_update_session()
    waiter = session._item_create_future["item_1"]

    with pytest.raises(APIError) as exc_info:
        RealtimeSession._handle_error(
            session, _rejection("chat_ctx_create_abc", code="insufficient_quota")
        )

    assert exc_info.value.retryable is False
    assert isinstance(waiter.exception(), llm.RealtimeError)


# --------------------------------------------------------------------------- #
# a response.create rejected before any response.created (the conversation already
# has an active response) must fail its generate_reply future immediately with the
# provider code, instead of orphaning it until the 10s timeout — while still emitting
# the error event.
# --------------------------------------------------------------------------- #


def _active_response_rejection(event_id: str) -> RealtimeErrorEvent:
    return RealtimeErrorEvent.construct(
        type="error",
        event_id=event_id,
        error={
            "message": "Conversation already has an active response",
            "type": "invalid_request_error",
            "code": "conversation_already_has_active_response",
            "event_id": event_id,
        },
    )


def test_active_response_rejection_fails_generate_reply_future_fast() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures = {"response_create_1": fut}

    RealtimeSession._handle_error(session, _active_response_rejection("response_create_1"))

    # settled immediately (no 10s timeout), with the typed error and provider code
    assert fut.done()
    err = fut.exception()
    assert isinstance(err, llm.RealtimeError)
    assert err.code == "conversation_already_has_active_response"
    # the future was consumed so nothing else touches it
    assert "response_create_1" not in session._response_created_futures
    # both surfaces: the error is still emitted as a recoverable "error" event
    assert captured["recoverable"] is True


def test_error_with_unknown_event_id_leaves_generate_reply_futures_untouched() -> None:
    # an error naming an event_id we aren't tracking must not disturb a pending future
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures = {"response_create_1": fut}

    RealtimeSession._handle_error(session, _active_response_rejection("response_create_other"))

    assert not fut.done()
    assert session._response_created_futures == {"response_create_1": fut}
    # still reported down the ordinary path
    assert captured["recoverable"] is True
