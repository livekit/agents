from __future__ import annotations

import asyncio
import types
from types import SimpleNamespace
from typing import cast

import pytest

from livekit.agents import APIConnectOptions, llm
from livekit.agents._exceptions import APIError
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.plugins.openai.realtime.realtime_model import RealtimeSession, _is_fatal_error

pytestmark = pytest.mark.unit


def _create_response_retry_session() -> RealtimeSession:
    session = RealtimeSession.__new__(RealtimeSession)
    session._opts = SimpleNamespace(conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    session._realtime_model = SimpleNamespace(
        _provider_label="openai",
        _label="openai",
        label="openai",
        model="gpt-realtime",
        provider="openai",
    )
    session._msg_ch = SimpleNamespace(closed=False)
    session._response_created_futures = {}
    session._response_create_params = {}
    session._response_retry_event_ids = set()
    session._discarded_event_ids = set()
    session._current_generation = None
    session._instructions = None
    session._sent_events = []
    session._errors = []
    session.send_event = types.MethodType(
        lambda self, event: self._sent_events.append(event),
        session,
    )
    session.emit = types.MethodType(
        lambda self, event_name, event: self._errors.append((event_name, event)),
        session,
    )
    return session


def _response_created(*, client_event_id: str, response_id: str = "resp_1") -> SimpleNamespace:
    return SimpleNamespace(
        response=SimpleNamespace(id=response_id, metadata={"client_event_id": client_event_id})
    )


def _response_failed(*, code: str = "rate_limit_exceeded") -> SimpleNamespace:
    return SimpleNamespace(
        response=SimpleNamespace(
            id="resp_1",
            status="failed",
            usage=None,
            status_details=SimpleNamespace(
                error=SimpleNamespace(type="server_error", code=code),
            ),
        )
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


def _handle_error_session(capture: dict[str, object]) -> RealtimeSession:
    return cast(
        RealtimeSession,
        SimpleNamespace(
            _realtime_model=SimpleNamespace(_provider_label="openai"),
            _emit_error=lambda error, recoverable: capture.update(recoverable=recoverable),
        ),
    )


def test_handle_error_raises_on_fatal() -> None:
    # a fatal code is raised (not emitted here): the recv loop re-raises it so
    # _main_task emits it once with recoverable=False and stops reconnecting
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(
        error=SimpleNamespace(message="quota exceeded", code="insufficient_quota")
    )
    with pytest.raises(APIError) as exc_info:
        RealtimeSession._handle_error(session, event)
    assert exc_info.value.retryable is False
    assert captured == {}  # not emitted by the handler; _main_task owns the emit


def test_handle_error_emits_transient_as_recoverable() -> None:
    captured: dict[str, object] = {}
    session = _handle_error_session(captured)
    event = SimpleNamespace(error=SimpleNamespace(message="server hiccup", code="server_error"))
    RealtimeSession._handle_error(session, event)
    assert captured["recoverable"] is True


def test_handle_error_ignores_cancellation_failed() -> None:
    captured: dict[str, object] = {}
    event = SimpleNamespace(error=SimpleNamespace(message="Cancellation failed: no response"))
    RealtimeSession._handle_error(_handle_error_session(captured), event)
    assert captured == {}  # early return, nothing emitted


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


async def test_response_done_failed_retries_before_output() -> None:
    session = _create_response_retry_session()

    fut = session.generate_reply(instructions="say hi")
    create_event = session._sent_events[-1]
    client_event_id = create_event.response.metadata["client_event_id"]
    session._handle_response_created(_response_created(client_event_id=client_event_id))

    generation = await fut
    assert generation.response_id == "resp_1"

    session._handle_response_done(_response_failed())
    await asyncio.sleep(0.15)

    assert len(session._sent_events) == 2
    retry_event = session._sent_events[-1]
    assert retry_event.type == "response.create"
    assert retry_event.event_id.startswith("response_retry_")
    assert retry_event.response.instructions == "say hi"
    created_events = [event for event in session._errors if event[0] == "generation_created"]
    session._handle_response_created(
        _response_created(client_event_id=retry_event.event_id, response_id="resp_2")
    )
    assert [
        event for event in session._errors if event[0] == "generation_created"
    ] == created_events
    assert retry_event.event_id not in session._response_retry_event_ids
    assert session._current_generation is not None
    assert session._current_generation.retry_count == 1
    error_event = next(event for event in session._errors if event[0] == "error")
    assert error_event[1].recoverable is True


async def test_response_done_failed_does_not_retry_after_output_started() -> None:
    session = _create_response_retry_session()

    fut = session.generate_reply(instructions="say hi")
    create_event = session._sent_events[-1]
    client_event_id = create_event.response.metadata["client_event_id"]
    session._handle_response_created(_response_created(client_event_id=client_event_id))
    await fut
    assert session._current_generation is not None
    session._current_generation.output_started = True

    session._handle_response_done(_response_failed())
    await asyncio.sleep(0.15)

    assert len(session._sent_events) == 1
    assert session._current_generation is None
