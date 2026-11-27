from __future__ import annotations

from typing import Any

import pytest

from livekit.agents import llm
from livekit.plugins.aws.experimental.realtime import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit

_ASSISTANT_SPEC_START = {
    "event": {
        "contentStart": {
            "role": "ASSISTANT",
            "type": "TEXT",
            "contentId": "text-1",
            "additionalModelFields": '{"generationStage":"SPECULATIVE"}',
        }
    }
}


@pytest.fixture
async def session(monkeypatch: pytest.MonkeyPatch) -> Any:
    async def _no_streams(self: RealtimeSession, is_restart: bool = False) -> None:
        return None

    monkeypatch.setattr(RealtimeSession, "initialize_streams", _no_streams)
    model = RealtimeModel(region="us-east-1")
    sess = model.session()
    yield sess
    await sess.aclose()


def _collect_generations(sess: RealtimeSession) -> list[llm.GenerationCreatedEvent]:
    events: list[llm.GenerationCreatedEvent] = []
    sess.on("generation_created", events.append)
    return events


async def test_generate_reply_generation_is_user_initiated_and_emitted_once(
    session: RealtimeSession,
) -> None:
    events = _collect_generations(session)
    fut = session.generate_reply(instructions="Ask the caller for their first name.")

    # Nova starts replying: completionStart, then the turn tracker sees the speculative text
    session._create_response_generation()
    session._turn_tracker.feed(_ASSISTANT_SPEC_START)

    assert fut.done()
    reply_ev = fut.result()
    assert reply_ev.user_initiated
    # one event only, so the framework never starts a second speech on the same channels
    assert len(events) == 1
    assert events[0] is reply_ev


async def test_server_initiated_generation_is_emitted_once(session: RealtimeSession) -> None:
    events = _collect_generations(session)

    session._create_response_generation()
    session._turn_tracker.feed(_ASSISTANT_SPEC_START)
    session.emit_generation_event()

    assert len(events) == 1
    assert not events[0].user_initiated

    # the next response is a new generation and gets its own event
    session._close_current_generation()
    session._create_response_generation()
    assert len(events) == 2
    assert events[1].response_id != events[0].response_id


async def test_tool_call_generation_and_tool_reply_are_each_emitted_once(
    session: RealtimeSession,
) -> None:
    events = _collect_generations(session)

    # Nova's turn: speculative text, then a tool call (the turn tracker emits on both)
    await session._handle_event({"event": {"completionStart": {"completionId": "c1"}}})
    await session._handle_event(_ASSISTANT_SPEC_START)
    await session._handle_event(
        {"event": {"contentStart": {"type": "TOOL", "role": "TOOL", "contentId": "tool-1"}}}
    )
    await session._handle_event(
        {
            "event": {
                "toolUse": {
                    "toolUseId": "call-1",
                    "toolName": "lookup_weather",
                    "content": '{"city": "Seattle"}',
                }
            }
        }
    )

    assert len(events) == 1
    tool_gen = events[0]
    # the tool call reaches the single speech reading this generation, which then ends
    calls = [fnc async for fnc in tool_gen.function_stream]
    assert [(c.call_id, c.name) for c in calls] == [("call-1", "lookup_weather")]

    # Nova speaks the tool result as a new generation, announced once
    await session._handle_event(_ASSISTANT_SPEC_START)
    session.emit_generation_event()

    assert len(events) == 2
    assert not events[1].user_initiated
    assert events[1].response_id != tool_gen.response_id


async def _send_user_text(sess: RealtimeSession, text: str) -> None:
    chat_ctx = llm.ChatContext.empty()
    await sess.update_chat_ctx(chat_ctx)  # initial history sync
    chat_ctx.add_message(role="user", content=text)
    await sess.update_chat_ctx(chat_ctx)  # new user text, sent to Nova as interactive text


async def test_user_text_without_generate_reply_is_scheduled_by_the_framework(
    session: RealtimeSession,
) -> None:
    events = _collect_generations(session)
    await _send_user_text(session, "What's my balance?")
    assert session._pending_generation_fut is not None

    session._create_response_generation()
    session._turn_tracker.feed(_ASSISTANT_SPEC_START)

    # nothing awaits update_chat_ctx's future, so the reply must not be user_initiated,
    # or the framework would skip it and nobody would play it
    assert len(events) == 1
    assert not events[0].user_initiated
    assert session._pending_generation_fut is None
    # a later generate_reply() must not pick up the reply that is already playing
    assert not session.generate_reply().done()


async def test_generate_reply_with_user_input_is_user_initiated(session: RealtimeSession) -> None:
    events = _collect_generations(session)
    # the framework's generate_reply(user_input=...) flow: push the text, then generate_reply()
    await _send_user_text(session, "What's my balance?")
    fut = session.generate_reply()

    session._create_response_generation()
    session._turn_tracker.feed(_ASSISTANT_SPEC_START)

    assert fut.done()
    assert fut.result().user_initiated
    assert len(events) == 1
    assert events[0] is fut.result()
