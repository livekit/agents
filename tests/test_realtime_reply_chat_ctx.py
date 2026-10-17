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
import time
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
            # the fake model reports supports_overlapping_speech=False, so the caller gates
            _rt_overlapping_speech_enabled=False,
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


async def _resolve_reply_future(rt_session: FakeRealtimeSession, task: asyncio.Task[None]) -> None:
    deadline = time.monotonic() + 5.0
    while not rt_session._reply_futs:
        # a reply task that died never reaches generate_reply, so waiting on the future alone
        # would spin here forever; surface why it stopped instead
        if task.done():
            task.result()  # re-raises whatever killed it
            raise AssertionError("the reply task finished without calling generate_reply")
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for generate_reply")
        await asyncio.sleep(0)
    rt_session._reply_futs[-1].set_result(cast(llm.GenerationCreatedEvent, object()))


async def test_update_chat_ctx_success_generates_reply() -> None:
    rt_session = FakeRealtimeModel().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()

    task = _run_reply_task(activity, handle)
    await _resolve_reply_future(rt_session, task)
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
    await _resolve_reply_future(rt_session, task)
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


# --- when the user message becomes visible in the agent's own context --------


async def test_the_user_message_is_recorded_before_the_push() -> None:
    """Regression for https://github.com/livekit/agents/issues/7906.

    The push waits for the server to acknowledge the item. A context copied from
    this agent while that is in flight used to have no user message in it, so the
    next update built from that copy asked the server to delete an item it had
    just acknowledged, and the reply was generated without the user's turn.
    """
    rt_session = FakeRealtimeModel().session()
    activity = _FakeActivity(rt_session)
    during: list[list[str]] = []

    async def update(chat_ctx: llm.ChatContext) -> None:
        # exactly what a concurrent `agent.chat_ctx.copy()` would contain
        during.append([item.id for item in activity._agent._chat_ctx.items])

    rt_session.update_chat_ctx = update  # type: ignore[method-assign]
    handle = SpeechHandle.create()
    handle._authorize_generation()

    task = _run_reply_task(activity, handle)
    await _resolve_reply_future(rt_session, task)
    await task

    assert len(during) == 1
    assert len(during[0]) == 1, "the user message was not in the agent context during the push"


async def test_the_item_added_event_still_fires_after_the_push() -> None:
    """Recording the message early must not move the event that announces it."""
    rt_session = FakeRealtimeModel().session()
    activity = _FakeActivity(rt_session)
    order: list[str] = []

    async def update(chat_ctx: llm.ChatContext) -> None:
        order.append("push")

    rt_session.update_chat_ctx = update  # type: ignore[method-assign]
    activity._session = SimpleNamespace(
        _conversation_item_added=lambda msg: order.append("item_added"),
        _update_agent_state=lambda state: None,
    )
    handle = SpeechHandle.create()
    handle._authorize_generation()

    task = _run_reply_task(activity, handle)
    await _resolve_reply_future(rt_session, task)
    await task

    assert order == ["push", "item_added"]


async def test_an_unexpected_error_leaves_no_message_behind() -> None:
    """The turn is over and nothing was sent, so the history must not keep it.

    A RealtimeError is different: the items did reach the server and only the
    ack timed out, so that message stays (asserted above).
    """
    rt_session = FakeRealtimeModel().session()
    rt_session.update_error = RuntimeError("boom")
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()

    await _run_reply_task(activity, handle)

    assert handle.done()
    assert isinstance(handle.exception(), RuntimeError)
    assert [item.id for item in activity._agent._chat_ctx.items] == []
