import asyncio
import types
from unittest.mock import Mock

import pytest

from livekit.agents import llm
from livekit.agents.llm import remote_chat_context
from livekit.plugins.openai.realtime import realtime_model
from livekit.plugins.openai.realtime.realtime_model import RealtimeSession

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


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


def test_truncate_deletes_item_when_no_audio_played() -> None:
    """Interrupting before any audio frame plays (audio_end_ms == 0) must not
    send a conversation.item.truncate (the Realtime API rejects it with
    "Only model output audio messages can be truncated"). The item is deleted
    instead so a generated-but-unplayed audio message is not left dangling in
    the remote chat ctx."""
    from openai.types.realtime import ConversationItemDeleteEvent

    session = _create_session()

    session.truncate(
        message_id="item_1",
        modalities=["audio", "text"],
        audio_end_ms=0,
    )

    assert len(session._sent_events) == 1
    event = session._sent_events[0]
    assert isinstance(event, ConversationItemDeleteEvent)
    assert event.item_id == "item_1"


def test_truncate_sends_event_when_audio_played() -> None:
    """When audio has actually played (audio_end_ms > 0) the truncate event is
    still sent for audio modalities."""
    from openai.types.realtime import ConversationItemTruncateEvent

    session = _create_session()

    session.truncate(
        message_id="item_1",
        modalities=["audio", "text"],
        audio_end_ms=500,
    )

    assert len(session._sent_events) == 1
    event = session._sent_events[0]
    assert isinstance(event, ConversationItemTruncateEvent)
    assert event.item_id == "item_1"
    assert event.audio_end_ms == 500


def test_response_done_handles_string_status_details(monkeypatch) -> None:
    session = _create_session()
    session._realtime_model = types.SimpleNamespace(_provider_label="xAI")
    event = types.SimpleNamespace(
        response=types.SimpleNamespace(
            id="resp_1",
            status="incomplete",
            status_details="incomplete",
        )
    )
    debug = Mock()
    monkeypatch.setattr(realtime_model.logger, "debug", debug)

    session._handle_response_done_but_not_complete(event)

    debug.assert_called_once()
    assert debug.call_args.args[3] == "incomplete"
    assert debug.call_args.args[4] is None
