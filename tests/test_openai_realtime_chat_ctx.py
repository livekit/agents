import asyncio
import types

import pytest

from livekit.agents import llm
from livekit.agents.llm import remote_chat_context
from livekit.plugins.openai.realtime.realtime_model import RealtimeSession

pytestmark = pytest.mark.unit


def _create_session() -> RealtimeSession:
    session = RealtimeSession.__new__(RealtimeSession)
    session._update_chat_ctx_lock = asyncio.Lock()
    session._remote_chat_ctx = remote_chat_context.RemoteChatContext()
    session._item_delete_future = {}
    session._item_create_future = {}
    session._sent_events = []

    def send_event(self: RealtimeSession, event: object) -> None:
        self._sent_events.append(event)
        item = getattr(event, "item", None)
        if item is not None and item.id in self._item_create_future:
            self._item_create_future[item.id].set_result(None)

    session.send_event = types.MethodType(send_event, session)
    return session


async def test_update_chat_ctx_filters_new_empty_messages() -> None:
    session = _create_session()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content=[], id="empty-message")

    await session.update_chat_ctx(chat_ctx)

    assert session._sent_events == []


async def test_update_chat_ctx_keeps_existing_remote_empty_messages() -> None:
    session = _create_session()
    remote_message = llm.ChatMessage(role="user", content=[], id="remote-empty-message")
    session._remote_chat_ctx.insert(None, remote_message)

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.items.append(remote_message)

    await session.update_chat_ctx(chat_ctx)

    assert session._sent_events == []
