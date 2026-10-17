"""
Tests for the chat-context push inside AgentActivity._realtime_reply_task.

The push before rt_session.generate_reply is best-effort: on a provider-reported
failure (e.g. the plugin's "update_chat_ctx timed out.", where the items were
already sent and only the ack timed out) the reply is still generated instead of
dropping the whole turn. Unexpected errors mark the SpeechHandle done with the
error (observable via SpeechHandle.exception()) without crashing the turn task.

The push also has to survive overlapping with another one: both diff their own
snapshot of the server conversation, so a snapshot taken before the other push
is acked turns the other push's items into deletes.
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
            _rt_chat_ctx_push_lock=asyncio.Lock(),
        )
        # the real push helper, so the lock it takes is part of what is under test
        self._extend_rt_chat_ctx = AgentActivity._extend_rt_chat_ctx.__get__(self)


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


class _MirroredRealtimeSession(FakeRealtimeSession):
    """A session whose `chat_ctx` is a mirror the server only updates on ack.

    The OpenAI plugin works this way: `update_chat_ctx` diffs the context it is handed
    against its mirror of the server conversation, sending creates for what is new and
    deletes for what is missing, and the mirror only catches up when the server acks.
    Items a caller's snapshot predates therefore come back out of the diff as deletes.
    """

    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.ack = asyncio.Event()
        self.deleted: list[str] = []
        # the plugin's own _update_chat_ctx_lock: it diffs, sends and waits for the ack
        # while holding it, so a push that overlaps one diffs the mirror the earlier
        # push left behind -- never the one it saw when it built its context
        self._update_lock = asyncio.Lock()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_lock:
            requested = {item.id for item in chat_ctx.items}
            self.deleted.extend(
                item.id for item in self._chat_ctx.items if item.id not in requested
            )
            await self.ack.wait()
            self._chat_ctx = chat_ctx


async def test_overlapping_pushes_keep_each_other_s_items() -> None:
    """Regression for https://github.com/livekit/agents/issues/7880.

    A function tool returns while another speech is queued, so its output push waits
    for its ack. A text user turn queued in the meantime used to snapshot the server
    conversation before that ack landed, and then send a delete for the
    `function_call_output` it could not see -- leaving the function call open with no
    output, so the tool reply and every later response were generated without the
    result and the model reports the operation as still running.
    """
    rt_session = _MirroredRealtimeSession(FakeRealtimeModel())
    activity = _FakeActivity(rt_session)

    fnc_output = llm.FunctionCallOutput(
        call_id="call_weather", name="get_weather", output="sunny", is_error=False
    )
    user_msg = llm.ChatMessage(role="user", content=["is it still raining?"])

    tool_push = asyncio.create_task(activity._extend_rt_chat_ctx([fnc_output]))
    await asyncio.sleep(0)  # the tool output is sent and waiting for its ack
    text_push = asyncio.create_task(activity._extend_rt_chat_ctx([user_msg]))
    await asyncio.sleep(0)  # the text turn must not snapshot behind the pending ack

    rt_session.ack.set()
    await asyncio.gather(tool_push, text_push)

    assert rt_session.deleted == []
    assert [item.id for item in rt_session.chat_ctx.items] == [fnc_output.id, user_msg.id]
