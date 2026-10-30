"""The say() context push must not race a concurrent tool-output push.

``session.say()`` writes its spoken message to the realtime provider by taking a
snapshot of the remote context and sending the whole context back. Every other
realtime context update is serialized by ``AgentActivity._rt_chat_ctx_lock``
(#7905) so a snapshot cannot be invalidated by an update that lands before the
write. The say() push added in #7947 takes its snapshot without that lock, so a
tool result applied while the say update is outstanding is deleted from the
provider's history by the older snapshot.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, ToolResult, function_tool, llm, utils

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = pytest.mark.unit

SAY_TEXT = "hold on"


class _WaitTrackingLock(asyncio.Lock):
    """An activity lock that reports when a second task has to wait on it."""

    def __init__(self) -> None:
        super().__init__()
        self.contended = asyncio.Event()

    async def acquire(self) -> bool:
        if self.locked():
            self.contended.set()
        return await super().acquire()


class _RacingSession(FakeRealtimeSession):
    """Apply whole-context updates the way a provider does: omit means delete."""

    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.say_push_started = asyncio.Event()
        self.release_say_ack = asyncio.Event()
        self.say_applied = asyncio.Event()
        self.tool_push_started = asyncio.Event()
        self.tool_applied = asyncio.Event()
        self.deleted_items: list[llm.ChatItem] = []
        self._provider_lock = asyncio.Lock()

    @staticmethod
    def _is_say_push(chat_ctx: llm.ChatContext) -> bool:
        return any(
            item.type == "message" and item.role == "assistant" and item.text_content == SAY_TEXT
            for item in chat_ctx.items
        )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        is_tool_push = any(item.type == "function_call_output" for item in chat_ctx.items)
        # A later tool-output push may include the spoken message in its own
        # snapshot; only the push that carries no tool output is the say push.
        is_say_push = not is_tool_push and self._is_say_push(chat_ctx)
        if is_say_push:
            self.say_push_started.set()
            await self.release_say_ack.wait()
        elif is_tool_push:
            self.tool_push_started.set()
        async with self._provider_lock:
            ids = {item.id for item in chat_ctx.items}
            self.deleted_items.extend(item for item in self._chat_ctx.items if item.id not in ids)
            self._chat_ctx = chat_ctx.copy()
        if is_say_push:
            self.say_applied.set()
        elif is_tool_push:
            self.tool_applied.set()


class _RacingModel(FakeRealtimeModel):
    def __init__(self) -> None:
        super().__init__(
            capabilities=fake_capabilities(
                audio_output=False,
                auto_tool_reply_generation=False,
                mutable_chat_context=True,
                supports_say=False,
            )
        )

    def session(self, *, turn_detection_disabled: bool = False) -> _RacingSession:
        session = _RacingSession(self)
        self.created_sessions.append(session)
        return session


def _generation(call: llm.FunctionCall) -> llm.GenerationCreatedEvent:
    messages = utils.aio.Chan[llm.MessageGeneration]()
    messages.close()
    calls = utils.aio.Chan[llm.FunctionCall]()
    calls.send_nowait(call)
    calls.close()
    return llm.GenerationCreatedEvent(
        message_stream=messages,
        function_stream=calls,
        user_initiated=True,
    )


async def test_say_push_keeps_concurrent_tool_output() -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> ToolResult:
            """Look up a result without requesting another model response."""
            tool_started.set()
            await release_tool.wait()
            return ToolResult("order 42 shipped", reply_required=False)

    model = _RacingModel()
    async with AgentSession(llm=model) as session:
        agent = ToolAgent()
        await session.start(agent)
        remote = model.created_sessions[-1]
        assert isinstance(remote, _RacingSession)
        lock = _WaitTrackingLock()
        session._activity._rt_chat_ctx_lock = lock

        # The say speech plays out and starts its context push; hold the ack so a
        # tool result can land between the snapshot and the write.
        say_handle = session.say(SAY_TEXT)
        await asyncio.wait_for(remote.say_push_started.wait(), timeout=5)

        # A tool turn runs while the say update is outstanding.
        tool_turn = session.generate_reply()
        while not remote._reply_futs:
            await asyncio.sleep(0)
        remote._reply_futs[-1].set_result(
            _generation(llm.FunctionCall(call_id="call_1", name="lookup", arguments="{}"))
        )
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        release_tool.set()
        # In the fixed code the tool push waits on the activity lock; in the
        # racy code it calls the provider immediately.
        tool_waiter = asyncio.ensure_future(remote.tool_push_started.wait())
        lock_waiter = asyncio.ensure_future(lock.contended.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait(
                    {tool_waiter, lock_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                ),
                timeout=5,
            )
        finally:
            tool_waiter.cancel()
            lock_waiter.cancel()

        remote.release_say_ack.set()
        await asyncio.wait_for(remote.say_applied.wait(), timeout=5)
        await asyncio.wait_for(remote.tool_push_started.wait(), timeout=5)
        await asyncio.wait_for(remote.tool_applied.wait(), timeout=5)
        await asyncio.wait_for(say_handle.wait_for_playout(), timeout=5)
        await asyncio.wait_for(tool_turn.wait_for_playout(), timeout=5)

        remote_outputs = [
            item for item in remote.chat_ctx.items if item.type == "function_call_output"
        ]
        remote_say = [
            item
            for item in remote.chat_ctx.items
            if item.type == "message" and item.text_content == SAY_TEXT
        ]
        assert len(remote_say) == 1
        assert [item.output for item in remote_outputs] == ["order 42 shipped"]
