"""
Tests for the chat-context push inside AgentActivity._realtime_reply_task.

The push before rt_session.generate_reply is best-effort: on a provider-reported
failure (e.g. the plugin's "update_chat_ctx timed out.", where the items were
already sent and only the ack timed out) the reply is still generated instead of
dropping the whole turn. Unexpected errors mark the SpeechHandle done with the
error (observable via SpeechHandle.exception()) without crashing the turn task.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from livekit.agents import llm
from livekit.agents.voice import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession

pytestmark = pytest.mark.unit


class _FakeActivity(SimpleNamespace):
    """The attribute surface _realtime_reply_task touches, around a FakeRealtimeSession."""

    def __init__(self, rt_session: FakeRealtimeSession) -> None:
        authorization_allowed = asyncio.Event()
        authorization_allowed.set()
        user_silence = asyncio.Event()
        user_silence.set()
        generation_calls: list[dict[str, Any]] = []

        async def _realtime_generation_task(**kwargs: Any) -> None:
            generation_calls.append(kwargs)

        super().__init__(
            _rt_session=rt_session,
            _authorization_allowed=authorization_allowed,
            _user_silence_event=user_silence,
            tools=[],
            _on_enter_ignored_tools=lambda tool_ctx: [],
            _tool_choice=None,
            _agent=SimpleNamespace(_chat_ctx=llm.ChatContext.empty()),
            _session=SimpleNamespace(
                _conversation_item_added=lambda msg: None,
                _update_agent_state=lambda state: None,
            ),
            _realtime_generation_task=_realtime_generation_task,
            generation_calls=generation_calls,
        )


def _run_reply_task(activity: _FakeActivity, speech_handle: SpeechHandle) -> asyncio.Task[None]:
    coro = AgentActivity._realtime_reply_task(
        cast(AgentActivity, activity),
        speech_handle=speech_handle,
        model_settings=ModelSettings(),
        user_input="hello",
    )
    return asyncio.create_task(coro)


async def _resolve_reply_future(rt_session: FakeRealtimeSession) -> None:
    while not rt_session._reply_futs:
        await asyncio.sleep(0)
    rt_session._reply_futs[-1].set_result(cast(llm.GenerationCreatedEvent, object()))


async def test_update_chat_ctx_success_generates_reply() -> None:
    rt_session = FakeRealtimeModel().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()

    task = _run_reply_task(activity, handle)
    await _resolve_reply_future(rt_session)
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    assert not handle.done()  # no error path marked the handle


async def test_update_chat_ctx_realtime_error_still_generates_reply() -> None:
    # the plugin raises RealtimeError("update_chat_ctx timed out.") after its ack
    # timeout; the items were already sent, so the turn must still produce a reply
    rt_session = FakeRealtimeModel().session()
    rt_session.update_error = llm.RealtimeError("update_chat_ctx timed out.")
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()

    task = _run_reply_task(activity, handle)
    await _resolve_reply_future(rt_session)
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    assert not handle.done()
    # the user message is still recorded locally
    assert any(
        item.type == "message" and item.role == "user" for item in activity._agent._chat_ctx.items
    )


async def test_update_chat_ctx_unexpected_error_fails_speech_handle() -> None:
    # a non-RealtimeError is unexpected: the turn is marked failed on the
    # SpeechHandle instead of crashing the task, and no reply is requested
    rt_session = FakeRealtimeModel().session()
    rt_session.update_error = RuntimeError("boom")
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()

    await _run_reply_task(activity, handle)

    assert rt_session.generate_reply_calls == 0
    assert activity.generation_calls == []
    assert handle.done()
    assert isinstance(handle.exception(), RuntimeError)
