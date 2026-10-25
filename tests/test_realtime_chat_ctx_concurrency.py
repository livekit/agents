from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, ToolResult, function_tool, llm, utils

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = pytest.mark.unit


class _DelayedSession(FakeRealtimeSession):
    """Mirror a provider that commits context updates only when acknowledged."""

    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.tool_push_started = asyncio.Event()
        self.release_tool_ack = asyncio.Event()
        self.deleted_items: list[llm.ChatItem] = []
        self._provider_lock = asyncio.Lock()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._provider_lock:
            if any(item.type == "function_call_output" for item in chat_ctx.items):
                self.tool_push_started.set()
                await self.release_tool_ack.wait()
            ids = {item.id for item in chat_ctx.items}
            self.deleted_items.extend(item for item in self._chat_ctx.items if item.id not in ids)
            self._chat_ctx = chat_ctx.copy()


class _DelayedModel(FakeRealtimeModel):
    def __init__(self) -> None:
        super().__init__(
            capabilities=fake_capabilities(audio_output=False, auto_tool_reply_generation=False)
        )

    def session(self, *, turn_detection_disabled: bool = False) -> _DelayedSession:
        session = _DelayedSession(self)
        self.created_sessions.append(session)
        return session


def _generation(call: llm.FunctionCall | None = None) -> llm.GenerationCreatedEvent:
    messages = utils.aio.Chan[llm.MessageGeneration]()
    calls = utils.aio.Chan[llm.FunctionCall]()
    messages.close()
    if call is not None:
        calls.send_nowait(call)
    calls.close()
    return llm.GenerationCreatedEvent(
        message_stream=messages, function_stream=calls, user_initiated=True
    )


async def _wait_for_reply(session: FakeRealtimeSession, count: int) -> None:
    async def wait() -> None:
        while len(session._reply_futs) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=5)


@pytest.mark.parametrize("interrupt", [False, True])
async def test_text_turn_preserves_pending_tool_output(interrupt: bool) -> None:
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

    model = _DelayedModel()
    async with AgentSession(llm=model) as session:
        agent = ToolAgent()
        await session.start(agent)
        remote = model.created_sessions[-1]
        assert isinstance(remote, _DelayedSession)
        try:
            tool_turn = session.generate_reply()
            await _wait_for_reply(remote, 1)
            call = llm.FunctionCall(call_id="call_1", name="lookup", arguments="{}")
            remote._chat_ctx.items.append(call)
            remote._reply_futs[0].set_result(_generation(call))
            await asyncio.wait_for(tool_started.wait(), timeout=5)
            if interrupt:
                tool_turn.interrupt()
            release_tool.set()
            await asyncio.wait_for(remote.tool_push_started.wait(), timeout=5)

            user_turn = session.generate_reply(user_input="What happened to my order?")
            authorized = asyncio.Event()
            wait_for_authorization = user_turn.wait_if_not_interrupted

            async def mark_authorized(futures: list[asyncio.Future[Any]]) -> None:
                await wait_for_authorization(futures)
                authorized.set()

            user_turn.wait_if_not_interrupted = mark_authorized
            # The producer continues to its next suspension after setting this event:
            # its context update is now queued while the tool acknowledgement is held.
            await asyncio.wait_for(authorized.wait(), timeout=5)
            remote.release_tool_ack.set()
            await _wait_for_reply(remote, 2)
            remote._reply_futs[1].set_result(_generation())
            await asyncio.wait_for(user_turn.wait_for_playout(), timeout=5)
            await asyncio.wait_for(tool_turn.wait_for_playout(), timeout=5)

            local_outputs = [i for i in agent.chat_ctx.items if i.type == "function_call_output"]
            remote_outputs = [i for i in remote.chat_ctx.items if i.type == "function_call_output"]
            assert len(local_outputs) == 1
            assert remote_outputs == local_outputs
            assert not any(i.type == "function_call_output" for i in remote.deleted_items)
            assert any(
                i.type == "message" and i.text_content == "What happened to my order?"
                for i in remote.chat_ctx.items
            )
        finally:
            release_tool.set()
            remote.release_tool_ack.set()
            for future in remote._reply_futs:
                if not future.done():
                    future.set_result(_generation())


async def test_explicit_context_replacement_can_remove_tool_outputs() -> None:
    model = _DelayedModel()
    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="test")
        await session.start(agent)
        remote = model.created_sessions[-1]
        assert isinstance(remote, _DelayedSession)
        remote.release_tool_ack.set()
        context = llm.ChatContext(
            [
                llm.FunctionCall(call_id="call_1", name="lookup", arguments="{}"),
                llm.FunctionCallOutput(call_id="call_1", output="done", is_error=False),
            ]
        )
        await agent.update_chat_ctx(context, exclude_invalid_function_calls=False)
        await agent.update_chat_ctx(llm.ChatContext.empty())
        assert remote.chat_ctx.items == []
        assert any(i.type == "function_call_output" for i in remote.deleted_items)


async def test_rejected_realtime_history_edit_preserves_agent_context(monkeypatch) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="Help")
        await session.start(agent)
        original = agent.chat_ctx.copy()
        target = original.copy()
        target.add_message(role="user", content="Rejected edit")

        async def reject(context: llm.ChatContext) -> None:
            raise llm.RealtimeError("history refused")

        monkeypatch.setattr(model.active_session, "update_chat_ctx", reject)
        with pytest.raises(llm.RealtimeError, match="history refused"):
            await agent.update_chat_ctx(target)
        assert agent.chat_ctx.is_equivalent(original)


async def test_realtime_history_edit_retains_transcript_received_during_ack(monkeypatch) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="Help")
        await session.start(agent)
        target = agent.chat_ctx.copy()
        target.add_message(role="user", content="Explicit edit")
        started = asyncio.Event()
        release = asyncio.Event()
        update = model.active_session.update_chat_ctx

        async def delayed(context: llm.ChatContext) -> None:
            started.set()
            await release.wait()
            await update(context)

        monkeypatch.setattr(model.active_session, "update_chat_ctx", delayed)
        edit = asyncio.create_task(agent.update_chat_ctx(target))
        try:
            await asyncio.wait_for(started.wait(), 3)
            model.active_session.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id="new-speech", transcript="New speech", is_final=True
                ),
            )
            release.set()
            await edit
            assert agent.chat_ctx.get_by_id("new-speech").text_content == "New speech"
            assert any(
                item.type == "message" and item.text_content == "Explicit edit"
                for item in agent.chat_ctx.items
            )
        finally:
            release.set()
            await asyncio.gather(edit, return_exceptions=True)


async def test_cancelled_history_edit_commits_after_acknowledgment(monkeypatch) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="Help")
        await session.start(agent)
        target = agent.chat_ctx.copy()
        target.add_message(role="user", content="Accepted edit")
        started = asyncio.Event()
        release = asyncio.Event()
        update = model.active_session.update_chat_ctx

        async def delayed(context: llm.ChatContext) -> None:
            started.set()
            await release.wait()
            await update(context)

        monkeypatch.setattr(model.active_session, "update_chat_ctx", delayed)
        edit = asyncio.create_task(agent.update_chat_ctx(target))
        try:
            await asyncio.wait_for(started.wait(), 3)
            edit.cancel()
            await asyncio.sleep(0)
            edit.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await edit
            assert agent.chat_ctx.is_equivalent(model.active_session.chat_ctx)
            assert any(
                item.type == "message" and item.text_content == "Accepted edit"
                for item in agent.chat_ctx.items
            )
        finally:
            release.set()
            await asyncio.gather(edit, return_exceptions=True)
