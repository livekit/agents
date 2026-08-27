from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

# The ultravox plugin is not part of the uv workspace (see the root pyproject
# `[tool.uv.workspace].exclude`); skip cleanly when it is not installed.
pytest.importorskip("livekit.plugins.ultravox")

from livekit.plugins.ultravox.realtime.events import (  # noqa: E402
    ForcedAgentMessageEvent,
    UserTextMessageEvent,
)
from livekit.plugins.ultravox.realtime.realtime_model import RealtimeSession  # noqa: E402

pytestmark = pytest.mark.unit


def _make_session() -> tuple[RealtimeSession, list[object]]:
    """Build a session without __init__/websocket and capture outbound client events."""
    session = RealtimeSession.__new__(RealtimeSession)
    session._pending_generation_fut = None  # type: ignore[attr-defined]
    session._pending_generation_epoch = None  # type: ignore[attr-defined]
    session._current_generation = None  # type: ignore[attr-defined]
    session._say_tasks = set()  # type: ignore[attr-defined]

    sent: list[object] = []
    session._send_client_event = sent.append  # type: ignore[method-assign,assignment]
    return session, sent


async def test_say_str_enqueues_forced_agent_message() -> None:
    session, sent = _make_session()

    fut = session.say("hello world")

    assert not fut.done()
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, ForcedAgentMessageEvent)
    assert ev.content == "hello world"
    assert ev.uninterruptible is True
    # the future is armed as the pending generation
    assert session._pending_generation_fut is fut
    assert session._pending_generation_epoch is not None

    fut.cancel()


async def test_say_async_iterable_collects_and_tracks_task() -> None:
    session, sent = _make_session()

    async def chunks() -> AsyncIterator[str]:
        yield "foo "
        yield "bar"

    fut = session.say(chunks())

    # the collection task must be held with a strong ref so it is not GC'd
    assert len(session._say_tasks) == 1
    task = next(iter(session._say_tasks))

    await task
    # task cleans itself out of the set once complete
    assert session._say_tasks == set()

    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, ForcedAgentMessageEvent)
    assert ev.content == "foo bar"
    assert ev.uninterruptible is True

    fut.cancel()


async def test_cancelling_say_sends_barge_in() -> None:
    session, sent = _make_session()

    fut = session.say("interrupt me")
    assert isinstance(sent[0], ForcedAgentMessageEvent)

    fut.cancel()
    await asyncio.sleep(0)  # let the done-callback run

    barge_ins = [
        ev for ev in sent if isinstance(ev, UserTextMessageEvent) and ev.urgency == "immediate"
    ]
    assert len(barge_ins) == 1
    assert barge_ins[0].text == ""
    assert session._pending_generation_fut is None
    assert session._pending_generation_epoch is None


async def test_say_supersedes_pending_generate_reply_without_barge_in() -> None:
    session, sent = _make_session()

    reply_fut = session.generate_reply()
    assert session._pending_generation_fut is reply_fut

    say_fut = session.say("take over")
    await asyncio.sleep(0)  # let the superseded future's done-callback run

    # the prior generate_reply future was cancelled by say()
    assert reply_fut.cancelled()
    # say() is now the active pending generation
    assert session._pending_generation_fut is say_fut

    # the superseded future must NOT emit a barge-in (say cleared the slot first)
    barge_ins = [
        ev for ev in sent if isinstance(ev, UserTextMessageEvent) and ev.urgency == "immediate"
    ]
    assert barge_ins == []
    # a forced agent message was sent for the say()
    assert any(isinstance(ev, ForcedAgentMessageEvent) for ev in sent)

    say_fut.cancel()


def test_forced_agent_message_serialization() -> None:
    dumped = ForcedAgentMessageEvent(content="hi", uninterruptible=True).model_dump(
        by_alias=True, exclude_none=True, mode="json"
    )
    assert dumped == {
        "type": "forced_agent_message",
        "content": "hi",
        "uninterruptible": True,
    }

    # unset optionals are dropped
    defaults = ForcedAgentMessageEvent().model_dump(by_alias=True, exclude_none=True, mode="json")
    assert defaults == {"type": "forced_agent_message", "content": ""}


async def test_generate_reply_still_defers_via_user_text_message() -> None:
    session, sent = _make_session()

    fut = session.generate_reply()

    assert not fut.done()
    assert session._pending_generation_fut is fut
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, UserTextMessageEvent)
    assert ev.text == ""
    # non-deferred so Ultravox produces a reply (defer_response falsy)
    assert not ev.defer_response

    fut.cancel()
