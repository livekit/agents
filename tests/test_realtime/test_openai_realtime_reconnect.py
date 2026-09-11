from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import pytest

from livekit.agents import llm
from livekit.plugins.openai.realtime.realtime_model import RealtimeModel, RealtimeSession
from livekit.plugins.xai.realtime import RealtimeModel as XAIRealtimeModel

pytestmark = pytest.mark.unit


@pytest.fixture
def paused_realtime_main(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _paused_main(self: RealtimeSession) -> None:
        await self._msg_ch._close_ev.wait()

    monkeypatch.setattr(RealtimeSession, "_main_task", _paused_main)


async def test_reconnect_discards_queued_response_and_settles_future(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(api_key="key")
    session = model.session()
    session._msg_ch.recv_nowait()
    response_fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
    session._response_created_futures["response-event"] = response_fut
    session.send_event({"type": "response.create", "event_id": "response-event"})

    session._discard_pending_client_events()

    with pytest.raises(llm.RealtimeError, match="discarded due to session reconnection"):
        await response_fut
    assert "response-event" not in session._response_created_futures
    await session.aclose()


async def test_reconnect_discards_queued_chat_event_and_settles_future(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(api_key="key")
    session = model.session()
    session._msg_ch.recv_nowait()
    chat_fut: asyncio.Future[None] = asyncio.Future()
    session._chat_ctx_event_futures["chat-event"] = chat_fut
    session._item_create_future["message"] = chat_fut
    session.send_event(
        {
            "type": "conversation.item.create",
            "event_id": "chat-event",
            "item": {"type": "message", "id": "message"},
        }
    )

    session._discard_pending_client_events()

    with pytest.raises(llm.RealtimeError, match="discarded due to session reconnection"):
        await chat_fut
    assert "chat-event" not in session._chat_ctx_event_futures
    assert "message" not in session._item_create_future
    await session.aclose()


async def test_reconnect_settles_acknowledgements_from_dead_socket(
    paused_realtime_main: None,
) -> None:
    model = RealtimeModel(api_key="key")
    session = model.session()
    pending: list[asyncio.Future[Any]] = [
        asyncio.Future(),
        asyncio.Future(),
        asyncio.Future(),
    ]
    session._item_create_future["created"] = pending[0]
    session._item_delete_future["deleted"] = pending[1]
    session._response_created_futures["response"] = pending[2]

    session._discard_pending_client_events(settle_all_pending=True)

    for fut in pending:
        with pytest.raises(llm.RealtimeError, match="discarded due to session reconnection"):
            await fut
    assert not session._item_create_future
    assert not session._item_delete_future
    assert not session._response_created_futures
    await session.aclose()


async def test_xai_reconnect_cancels_connection_bound_say_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _paused_main(self: RealtimeSession) -> None:
        await self._msg_ch._close_ev.wait()

    monkeypatch.setattr(RealtimeSession, "_main_task", _paused_main)
    model = XAIRealtimeModel(api_key="key")
    session = model.session()
    blocker = asyncio.Event()
    say_task = asyncio.create_task(blocker.wait())
    session._say_tasks.add(say_task)
    session._pending_say_event_ids.append("stale-say")

    session._on_reconnect_connection_state_discarded()
    with suppress(asyncio.CancelledError):
        await say_task

    assert say_task.cancelled()
    assert not session._pending_say_event_ids
    await session.aclose()
