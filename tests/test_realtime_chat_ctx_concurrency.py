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
