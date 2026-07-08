from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import cast

import pytest

from livekit.agents import llm, utils
from livekit.agents.llm.remote_chat_context import RemoteChatContext
from livekit.plugins.openai.realtime.realtime_model import RealtimeSession, _ResponseGeneration

pytestmark = pytest.mark.unit


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
# error classification (_handle_error): fatal -> non-recoverable
# --------------------------------------------------------------------------- #


def _handle_error_session(capture: dict[str, object]) -> RealtimeSession:
    return cast(
        RealtimeSession,
        SimpleNamespace(
            _realtime_model=SimpleNamespace(_provider_label="openai"),
            _emit_error=lambda error, recoverable: capture.update(recoverable=recoverable),
            _response_created_futures={},
        ),
    )


def test_handle_error_marks_fatal_non_recoverable() -> None:
    captured: dict[str, object] = {}
    event = SimpleNamespace(
        error=SimpleNamespace(message="quota exceeded", code="insufficient_quota")
    )
    RealtimeSession._handle_error(_handle_error_session(captured), event)
    assert captured["recoverable"] is False


def test_handle_error_marks_transient_recoverable() -> None:
    captured: dict[str, object] = {}
    event = SimpleNamespace(error=SimpleNamespace(message="server hiccup", code="server_error"))
    RealtimeSession._handle_error(_handle_error_session(captured), event)
    assert captured["recoverable"] is True


def test_handle_error_ignores_cancellation_failed() -> None:
    captured: dict[str, object] = {}
    event = SimpleNamespace(error=SimpleNamespace(message="Cancellation failed: no response"))
    RealtimeSession._handle_error(_handle_error_session(captured), event)
    assert captured == {}  # early return, nothing emitted


def test_handle_error_fails_pending_reply_future_on_fatal() -> None:
    # a fatal error must fail a pending generate_reply future (recoverable=False) so the
    # caller fails fast instead of waiting out the timeout
    fut: asyncio.Future[object] = asyncio.get_event_loop().create_future()
    session = cast(
        RealtimeSession,
        SimpleNamespace(
            _realtime_model=SimpleNamespace(_provider_label="openai"),
            _emit_error=lambda error, recoverable: None,
            _response_created_futures={"ev": fut},
        ),
    )
    event = SimpleNamespace(error=SimpleNamespace(message="quota", code="insufficient_quota"))
    RealtimeSession._handle_error(session, event)

    assert fut.done()
    exc = fut.exception()
    assert isinstance(exc, llm.RealtimeError)
    assert exc.recoverable is False


def test_handle_error_leaves_pending_future_on_transient() -> None:
    fut: asyncio.Future[object] = asyncio.get_event_loop().create_future()
    session = cast(
        RealtimeSession,
        SimpleNamespace(
            _realtime_model=SimpleNamespace(_provider_label="openai"),
            _emit_error=lambda error, recoverable: None,
            _response_created_futures={"ev": fut},
        ),
    )
    event = SimpleNamespace(error=SimpleNamespace(message="hiccup", code="server_error"))
    RealtimeSession._handle_error(session, event)

    assert not fut.done()  # transient error: let it time out / succeed normally
    fut.cancel()  # cleanup


# --------------------------------------------------------------------------- #
# cancel_and_wait: cancels the active response and waits for it to clear
# --------------------------------------------------------------------------- #


def _active_generation(done_fut: asyncio.Future[None]) -> _ResponseGeneration:
    return _ResponseGeneration(
        message_ch=utils.aio.Chan(),
        function_ch=utils.aio.Chan(),
        messages={},
        _created_timestamp=time.time(),
        _done_fut=done_fut,
    )


async def test_cancel_and_wait_noop_without_active_generation() -> None:
    sent: list[object] = []
    session = cast(
        RealtimeSession,
        SimpleNamespace(has_active_generation=False, send_event=sent.append),
    )
    await RealtimeSession.cancel_and_wait(session)
    assert sent == []


async def test_cancel_and_wait_sends_cancel_and_waits_for_done() -> None:
    done_fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    sent: list[object] = []
    session = cast(
        RealtimeSession,
        SimpleNamespace(
            has_active_generation=True,
            _current_generation=_active_generation(done_fut),
            send_event=sent.append,
        ),
    )
    task = asyncio.ensure_future(RealtimeSession.cancel_and_wait(session, timeout=1.0))
    await asyncio.sleep(0)
    assert len(sent) == 1  # response.cancel was sent
    assert not task.done()  # still waiting for the response to clear
    done_fut.set_result(None)
    await task


async def test_cancel_and_wait_times_out_gracefully() -> None:
    done_fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    session = cast(
        RealtimeSession,
        SimpleNamespace(
            has_active_generation=True,
            _current_generation=_active_generation(done_fut),
            send_event=lambda ev: None,
        ),
    )
    # returns without raising even though the response never clears
    await RealtimeSession.cancel_and_wait(session, timeout=0.05)
    done_fut.set_result(None)  # clean up the pending future
