"""
Tests for the chat-context push inside AgentActivity._realtime_reply_task.

Provider-specific synchronization can reject a finalized item explicitly. A missing
acknowledgement is intentionally UNKNOWN and preserves the historical best-effort reply,
without relying on a provider chat-context mirror as a correctness gate.
"""

from __future__ import annotations

import asyncio
import gc
import weakref
from types import SimpleNamespace
from typing import Any, cast

import pytest

from livekit.agents import llm
from livekit.agents.llm.realtime import (
    _UserMessageSyncResult,
    _UserMessageSyncStatus,
)
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
            _realtime_chat_ctx_lock=asyncio.Lock(),
            _pending_realtime_user_message_ids=set(),
            _rt_user_activity_started=False,
            _clear_realtime_input=lambda: None,
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
        self._sync_realtime_user_message = AgentActivity._sync_realtime_user_message.__get__(
            self, _FakeActivity
        )
        self._commit_realtime_user_message = AgentActivity._commit_realtime_user_message.__get__(
            self, _FakeActivity
        )
        self._interrupt_created_realtime_generation_if_owned = (
            AgentActivity._interrupt_created_realtime_generation_if_owned.__get__(
                self, _FakeActivity
            )
        )


def _run_reply_task(
    activity: _FakeActivity,
    speech_handle: SpeechHandle,
    *,
    user_message: llm.ChatMessage | None = None,
) -> asyncio.Task[None]:
    coro = AgentActivity._realtime_reply_task(
        cast(AgentActivity, activity),
        speech_handle=speech_handle,
        model_settings=ModelSettings(),
        user_message=user_message or llm.ChatMessage(role="user", content=["hello"]),
    )
    return asyncio.create_task(coro)


async def _wait_for_reply_futures(
    rt_session: FakeRealtimeSession, reply_tasks: list[asyncio.Task[None]], *, count: int = 1
) -> None:
    async def _wait() -> None:
        while len(rt_session._reply_futs) < count:
            for task in reply_tasks:
                if task.done():
                    await task
                    raise AssertionError("reply task exited before creating its provider future")
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait(), timeout=1.0)


async def _resolve_reply_future(
    rt_session: FakeRealtimeSession, reply_task: asyncio.Task[None]
) -> None:
    await _wait_for_reply_futures(rt_session, [reply_task])
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


async def test_update_chat_ctx_preserves_finalized_message_identity_and_metrics() -> None:
    rt_session = FakeRealtimeModel().session()
    added: list[llm.ChatMessage] = []
    activity = _FakeActivity(rt_session)
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(
        role="user",
        content=["edited transcript"],
        transcript_confidence=0.87,
        metrics={"transcription_delay": 0.12},
    )

    task = _run_reply_task(activity, handle, user_message=message)
    await _resolve_reply_future(rt_session, task)
    await task

    provider_message = rt_session.chat_ctx.get_by_id(message.id)
    local_message = activity._agent._chat_ctx.get_by_id(message.id)
    assert provider_message is not None
    assert provider_message.raw_text_content == "edited transcript"
    assert provider_message.transcript_confidence == 0.87
    assert provider_message.metrics == {"transcription_delay": 0.12}
    assert local_message is not None
    assert [item.id for item in added] == [message.id]


async def test_update_chat_ctx_ack_error_after_provider_update_still_generates_reply() -> None:
    # the plugin raises RealtimeError("update_chat_ctx timed out.") after its ack
    # timeout; the items were already sent, so the turn must still produce a reply
    class _AckTimeoutRealtimeSession(FakeRealtimeSession):
        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            await super().update_chat_ctx(chat_ctx)
            raise llm.RealtimeError("update_chat_ctx timed out after provider update")

    model = FakeRealtimeModel()
    rt_session = _AckTimeoutRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()

    task = _run_reply_task(activity, handle)
    await _resolve_reply_future(rt_session, task)
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    assert not handle.done()
    assert any(item.type == "message" and item.role == "user" for item in rt_session.chat_ctx.items)
    assert any(
        item.type == "message" and item.role == "user" for item in activity._agent._chat_ctx.items
    )


async def test_ack_error_accepts_exact_text_when_provider_drops_local_metadata() -> None:
    class _MetadataStrippingRealtimeSession(FakeRealtimeSession):
        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            finalized = chat_ctx.messages()[-1]
            provider_copy = llm.ChatMessage(
                id=finalized.id,
                role=finalized.role,
                content=finalized.content.copy(),
            )
            self._chat_ctx = llm.ChatContext([provider_copy])
            raise llm.RealtimeError("acknowledgement timed out after provider conversion")

    model = FakeRealtimeModel()
    rt_session = _MetadataStrippingRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(
        role="user",
        content=["edited finalized transcript"],
        transcript_confidence=0.88,
        metrics={"transcription_delay": 0.15},
    )

    task = _run_reply_task(activity, handle, user_message=message)
    await _resolve_reply_future(rt_session, task)
    await task

    local_message = activity._agent._chat_ctx.get_by_id(message.id)
    assert local_message is message
    assert local_message.transcript_confidence == 0.88
    assert local_message.metrics == {"transcription_delay": 0.15}
    assert added == [message]
    assert rt_session.generate_reply_calls == 1


async def test_realtime_error_before_provider_ack_is_unknown_and_generates() -> None:
    rt_session = FakeRealtimeModel().session()
    rt_session.update_error = llm.RealtimeError("update_chat_ctx failed before provider update")
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(role="user", content=["unsynchronized turn"])

    task = _run_reply_task(activity, handle, user_message=message)
    await _resolve_reply_future(rt_session, task)
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    assert rt_session.chat_ctx.get_by_id(message.id) is None
    assert activity._agent._chat_ctx.get_by_id(message.id) is message
    assert added == [message]
    assert not handle.done()


async def test_default_sync_completion_does_not_require_a_provider_mirror_item() -> None:
    class _NonMirroringRealtimeSession(FakeRealtimeSession):
        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            del chat_ctx

    model = FakeRealtimeModel()
    rt_session = _NonMirroringRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(role="user", content=["provider normalizes this item"])

    task = _run_reply_task(activity, handle, user_message=message)
    await _resolve_reply_future(rt_session, task)
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    assert rt_session.chat_ctx.get_by_id(message.id) is None
    assert activity._agent._chat_ctx.get_by_id(message.id) is message
    assert added == [message]
    assert not handle.done()


async def test_explicit_provider_rejection_fails_without_generation() -> None:
    class _RejectingRealtimeSession(FakeRealtimeSession):
        async def _sync_user_message(
            self, chat_ctx: llm.ChatContext, message_id: str
        ) -> _UserMessageSyncResult:
            del chat_ctx, message_id
            error = llm.RealtimeError("provider rejected finalized item")
            return _UserMessageSyncResult(_UserMessageSyncStatus.REJECTED, error)

    model = FakeRealtimeModel()
    rt_session = _RejectingRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(role="user", content=["provider rejects this item"])

    await _run_reply_task(activity, handle, user_message=message)

    assert rt_session.generate_reply_calls == 0
    assert activity.generation_calls == []
    assert activity._agent._chat_ctx.get_by_id(message.id) is None
    assert added == []
    assert handle.done()
    assert isinstance(handle.exception(), llm.RealtimeError)


async def test_provider_echo_during_context_update_emits_finalized_item_once() -> None:
    activity: _FakeActivity

    class _EchoingRealtimeSession(FakeRealtimeSession):
        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            message = chat_ctx.messages()[-1]
            AgentActivity._on_remote_item_added(
                cast(AgentActivity, activity),
                llm.RemoteItemAddedEvent(item=message, previous_item_id=None),
            )
            await super().update_chat_ctx(chat_ctx)

    model = FakeRealtimeModel()
    rt_session = _EchoingRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(
        role="user",
        content=["provider echoes this finalized item before acknowledging it"],
    )

    task = _run_reply_task(activity, handle, user_message=message)
    await _resolve_reply_future(rt_session, task)
    await task

    provider_messages = rt_session.chat_ctx.messages()
    local_messages = activity._agent._chat_ctx.messages()
    assert provider_messages == [message]
    assert local_messages == [message]
    assert added == [message]
    assert activity._pending_realtime_user_message_ids == set()
    assert rt_session.generate_reply_calls == 1


async def test_unknown_sync_ignores_stale_provider_mirror_content() -> None:
    class _StaleContextRealtimeSession(FakeRealtimeSession):
        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            finalized = chat_ctx.messages()[-1]
            stale = finalized.model_copy(update={"content": ["stale pre-edit transcript"]})
            self._chat_ctx = llm.ChatContext([stale])
            raise llm.RealtimeError("update failed before finalized edit reached provider")

    model = FakeRealtimeModel()
    rt_session = _StaleContextRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(
        id="same-turn-id",
        role="user",
        content=["finalized edited transcript"],
    )

    task = _run_reply_task(activity, handle, user_message=message)
    await _resolve_reply_future(rt_session, task)
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    assert rt_session.chat_ctx.get_by_id(message.id) != message
    assert activity._agent._chat_ctx.get_by_id(message.id) is message
    assert added == [message]
    assert not handle.done()


async def test_reply_task_cancellation_settles_pending_provider_generation() -> None:
    rt_session = FakeRealtimeModel().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()
    task = _run_reply_task(activity, handle)

    await _wait_for_reply_futures(rt_session, [task])

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert rt_session._reply_futs[0].cancelled()


async def test_cancellation_retrieves_already_failed_generation_future() -> None:
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    outer_task: asyncio.Task[None] | None = None

    class _AlreadyFailedRealtimeSession(FakeRealtimeSession):
        def __init__(self, model: FakeRealtimeModel) -> None:
            super().__init__(model)
            self.failed_future_ref: (
                weakref.ReferenceType[asyncio.Future[llm.GenerationCreatedEvent]] | None
            ) = None

        def generate_reply(self, **kwargs: Any) -> asyncio.Future[llm.GenerationCreatedEvent]:
            del kwargs
            self.generate_reply_calls += 1
            fut = loop.create_future()
            fut.set_exception(llm.RealtimeError("already failed"))
            self.failed_future_ref = weakref.ref(fut)
            assert outer_task is not None
            # Cancel after the atomic inner task has produced the future, before the shielded
            # outer task can consume it.
            loop.call_soon(outer_task.cancel)
            return fut

    def _record_unhandled(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        unhandled.append(context)

    model = FakeRealtimeModel()
    rt_session = _AlreadyFailedRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()
    loop.set_exception_handler(_record_unhandled)
    try:
        outer_task = _run_reply_task(activity, handle)
        with pytest.raises(asyncio.CancelledError):
            await outer_task

        # Drop the session's only strong reference and force finalization so an unretrieved
        # exception is delivered deterministically through the event-loop handler.
        assert rt_session.failed_future_ref is not None
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
            if rt_session.failed_future_ref() is None:
                break
    finally:
        loop.set_exception_handler(previous_handler)

    assert rt_session.failed_future_ref() is None
    assert not [
        context
        for context in unhandled
        if context.get("message") == "Future exception was never retrieved"
    ]


async def test_text_cancellation_after_generation_creation_interrupts_owned_output() -> None:
    class _CreatedThenCancelledRealtimeSession(FakeRealtimeSession):
        activity: _FakeActivity
        reply_task: asyncio.Task[None]

        def generate_reply(self, **kwargs: Any) -> asyncio.Future[llm.GenerationCreatedEvent]:
            del kwargs
            self.generate_reply_calls += 1
            generation = cast(
                llm.GenerationCreatedEvent,
                SimpleNamespace(user_initiated=True),
            )
            future = asyncio.get_running_loop().create_future()
            future.set_result(generation)
            self.activity._active_realtime_generation = generation
            self.reply_task.cancel()
            return future

    model = FakeRealtimeModel()
    rt_session = _CreatedThenCancelledRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    rt_session.activity = activity
    handle = SpeechHandle.create()
    handle._authorize_generation()
    reply_task = _run_reply_task(activity, handle)
    rt_session.reply_task = reply_task

    with pytest.raises(asyncio.CancelledError):
        await reply_task

    assert rt_session.interrupted is True


async def test_handle_interruption_after_generation_creation_interrupts_owned_output() -> None:
    class _CreatedThenInterruptedRealtimeSession(FakeRealtimeSession):
        activity: _FakeActivity
        handle: SpeechHandle

        def generate_reply(self, **kwargs: Any) -> asyncio.Future[llm.GenerationCreatedEvent]:
            del kwargs
            self.generate_reply_calls += 1
            generation = cast(
                llm.GenerationCreatedEvent,
                SimpleNamespace(user_initiated=True),
            )
            future = asyncio.get_running_loop().create_future()
            future.set_result(generation)
            self.activity._active_realtime_generation = generation
            self.handle.interrupt()
            return future

    model = FakeRealtimeModel()
    rt_session = _CreatedThenInterruptedRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    rt_session.activity = activity
    handle = SpeechHandle.create()
    handle._authorize_generation()
    rt_session.handle = handle

    await _run_reply_task(activity, handle)

    assert rt_session.interrupted is True


async def test_cancellation_before_authorization_cleans_up_all_wait_tasks() -> None:
    rt_session = FakeRealtimeModel().session()
    activity = _FakeActivity(rt_session)
    activity._authorization_allowed.clear()
    activity._user_silence_event.clear()
    handle = SpeechHandle.create()
    baseline = set(asyncio.all_tasks())

    task = _run_reply_task(activity, handle)

    async def _wait_for_authorization_children() -> None:
        while (
            len([pending for pending in asyncio.all_tasks() - baseline if not pending.done()]) < 4
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait_for_authorization_children(), timeout=1.0)
    spawned = asyncio.all_tasks() - baseline
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert all(pending.done() for pending in spawned)
    assert rt_session.generate_reply_calls == 0


async def test_interrupted_reply_does_not_commit_unsent_message() -> None:
    rt_session = FakeRealtimeModel().session()
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle.interrupt()
    message = llm.ChatMessage(role="user", content=["first finalized turn"])

    task = _run_reply_task(activity, handle, user_message=message)
    await task
    handle._mark_done()

    assert rt_session.generate_reply_calls == 0
    assert rt_session.chat_ctx.get_by_id(message.id) is None
    assert activity._agent._chat_ctx.get_by_id(message.id) is None
    assert added == []


async def test_cancellation_during_context_update_finishes_sync_without_generation() -> None:
    class _BlockingRealtimeSession(FakeRealtimeSession):
        def __init__(self, model: FakeRealtimeModel) -> None:
            super().__init__(model)
            self.update_started = asyncio.Event()
            self.update_release = asyncio.Event()

        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            self.update_started.set()
            await self.update_release.wait()
            await super().update_chat_ctx(chat_ctx)

    model = FakeRealtimeModel()
    rt_session = _BlockingRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(role="user", content=["finish atomic sync"])
    task = _run_reply_task(activity, handle, user_message=message)
    await rt_session.update_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    rt_session.update_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert rt_session.generate_reply_calls == 1
    assert rt_session._reply_futs[0].cancelled()
    assert rt_session.chat_ctx.get_by_id(message.id) is not None
    assert activity._agent._chat_ctx.get_by_id(message.id) is not None
    assert [item.id for item in added] == [message.id]


async def test_concurrent_finalized_turns_serialize_provider_context_updates() -> None:
    class _SerialProbeSession(FakeRealtimeSession):
        def __init__(self, model: FakeRealtimeModel) -> None:
            super().__init__(model)
            self.update_calls = 0
            self.first_started = asyncio.Event()
            self.first_release = asyncio.Event()

        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            self.update_calls += 1
            if self.update_calls == 1:
                self.first_started.set()
                await self.first_release.wait()
            await super().update_chat_ctx(chat_ctx)

    model = FakeRealtimeModel()
    rt_session = _SerialProbeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    first_handle = SpeechHandle.create()
    first_handle._authorize_generation()
    second_handle = SpeechHandle.create()
    second_handle._authorize_generation()
    first_message = llm.ChatMessage(role="user", content=["first"])
    second_message = llm.ChatMessage(role="user", content=["second"])

    first_task = _run_reply_task(activity, first_handle, user_message=first_message)
    await rt_session.first_started.wait()
    second_task = _run_reply_task(activity, second_handle, user_message=second_message)
    await asyncio.sleep(0)
    assert rt_session.update_calls == 1

    rt_session.first_release.set()
    await _wait_for_reply_futures(rt_session, [first_task, second_task], count=2)
    for future in rt_session._reply_futs:
        future.set_result(cast(llm.GenerationCreatedEvent, object()))
    await asyncio.gather(first_task, second_task)

    assert [message.id for message in rt_session.chat_ctx.messages()] == [
        first_message.id,
        second_message.id,
    ]
    assert [message.id for message in activity._agent._chat_ctx.messages()] == [
        first_message.id,
        second_message.id,
    ]
    assert [message.id for message in added] == [first_message.id, second_message.id]


async def test_exact_local_message_is_committed_after_provider_sync_once() -> None:
    local_message_present_during_update = False

    class _EchoingRealtimeSession(FakeRealtimeSession):
        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            nonlocal local_message_present_during_update
            message = chat_ctx.messages()[-1]
            local_message_present_during_update = (
                activity._agent._chat_ctx.get_by_id(message.id) is message
            )
            await super().update_chat_ctx(chat_ctx)

    model = FakeRealtimeModel()
    rt_session = _EchoingRealtimeSession(model)
    activity = _FakeActivity(rt_session)
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(
        role="user",
        content=["exact finalized text"],
        transcript_confidence=0.93,
        metrics={"transcription_delay": 0.2},
    )

    task = _run_reply_task(activity, handle, user_message=message)
    await _resolve_reply_future(rt_session, task)
    await task

    assert local_message_present_during_update is False
    assert activity._agent._chat_ctx.get_by_id(message.id) is message
    assert [item.id for item in added] == [message.id]
    assert added[0].transcript_confidence == 0.93
    assert added[0].metrics == {"transcription_delay": 0.2}


async def test_update_chat_ctx_unexpected_error_propagates_and_fails_speech_handle() -> None:
    # A non-RealtimeError is an implementation failure, not an ambiguous provider ack.
    rt_session = FakeRealtimeModel().session()
    rt_session.update_error = RuntimeError("boom")
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()
    handle._authorize_generation()

    with pytest.raises(RuntimeError, match="boom"):
        await _run_reply_task(activity, handle)

    assert rt_session.generate_reply_calls == 0
    assert activity.generation_calls == []
    assert handle.done()
    assert isinstance(handle.exception(), RuntimeError)
