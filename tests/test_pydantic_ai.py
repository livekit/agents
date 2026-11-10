"""Tests for the Pydantic AI LLMAdapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

from livekit.agents.llm import ChatChunk, ChatContext, FunctionCall, FunctionCallOutput
from livekit.plugins.pydantic_ai import LLMAdapter, chat_ctx_to_messages

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


async def collect(stream) -> tuple[list[str], list[ChatChunk]]:
    """Collect text deltas and every raw chunk from an LLMStream."""
    texts: list[str] = []
    chunks: list[ChatChunk] = []
    async for chunk in stream:
        chunks.append(chunk)
        if chunk.delta and chunk.delta.content:
            texts.append(chunk.delta.content)
    return texts, chunks


def user_ctx(text: str = "Hi") -> ChatContext:
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content=text)
    return chat_ctx


# --- streaming ---


async def test_streams_text_from_test_model():
    agent = Agent(TestModel(custom_output_text="Hello world from Pydantic AI"))
    adapter = LLMAdapter(agent)

    texts, _ = await collect(adapter.chat(chat_ctx=user_ctx()))

    assert len(texts) > 1  # TestModel streams word by word
    assert "".join(texts) == "Hello world from Pydantic AI"


async def test_streams_deltas_from_function_model():
    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "The "
        yield "capital "
        yield "is Paris."

    adapter = LLMAdapter(Agent(FunctionModel(stream_function=stream_fn)))

    texts, _ = await collect(adapter.chat(chat_ctx=user_ctx("Capital of France?")))

    assert texts == ["The ", "capital ", "is Paris."]


async def test_reports_usage_without_text():
    agent = Agent(TestModel(custom_output_text="ok"))
    adapter = LLMAdapter(agent)

    _, chunks = await collect(adapter.chat(chat_ctx=user_ctx()))

    usage_chunks = [c for c in chunks if c.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].delta is None
    assert usage_chunks[0].usage.prompt_tokens > 0
    assert usage_chunks[0].usage.completion_tokens > 0


async def test_tools_run_inside_pydantic_ai():
    """Tool calls are executed by Pydantic AI and never forwarded to LiveKit as tool_calls."""
    calls: list[str] = []

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name="lookup_order", json_args='{"order_id": "42"}')}
        else:
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            assert isinstance(last.parts[0], ToolReturnPart)
            yield f"Your order: {last.parts[0].content}"

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool_plain
    def lookup_order(order_id: str) -> str:
        calls.append(order_id)
        return f"order {order_id} ships tomorrow"

    adapter = LLMAdapter(agent)
    texts, chunks = await collect(adapter.chat(chat_ctx=user_ctx("Where is order 42?")))

    assert calls == ["42"]
    assert "".join(texts) == "Your order: order 42 ships tomorrow"
    assert all(not c.delta.tool_calls for c in chunks if c.delta)


async def test_history_and_deps_reach_the_model():
    seen: list[list[ModelMessage]] = []
    seen_instructions: list[str | None] = []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(messages)
        seen_instructions.append(info.instructions)
        yield "done"

    agent: Agent[dict[str, str], str] = Agent(
        FunctionModel(stream_function=stream_fn), deps_type=dict
    )

    @agent.instructions
    def add_name(ctx: RunContext[dict[str, str]]) -> str:
        return f"The caller is {ctx.deps['name']}."

    adapter = LLMAdapter(agent, deps={"name": "Ada"})

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Be brief.")
    chat_ctx.add_message(role="user", content="first question")
    chat_ctx.add_message(role="assistant", content="first answer")
    chat_ctx.add_message(role="user", content="second question")

    await collect(adapter.chat(chat_ctx=chat_ctx))

    assert len(seen) == 1
    messages = seen[0]
    # system + user, assistant, user
    assert [type(m) for m in messages] == [ModelRequest, ModelResponse, ModelRequest]
    first = messages[0]
    assert isinstance(first, ModelRequest)
    assert isinstance(first.parts[0], SystemPromptPart)
    assert first.parts[0].content == "Be brief."
    assert isinstance(first.parts[1], UserPromptPart)
    assert first.parts[1].content == "first question"
    assert seen_instructions[0] is not None
    assert "The caller is Ada." in seen_instructions[0]


async def test_trailing_system_message_is_sent_as_request():
    """generate_reply(instructions=...) leaves a system message last; the run must still start."""
    seen: list[list[ModelMessage]] = []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(messages)
        yield "Welcome!"

    adapter = LLMAdapter(Agent(FunctionModel(stream_function=stream_fn)))

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Greet the user warmly.")

    texts, _ = await collect(adapter.chat(chat_ctx=chat_ctx))

    assert texts == ["Welcome!"]
    last = seen[0][-1]
    assert isinstance(last, ModelRequest)
    assert isinstance(last.parts[-1], SystemPromptPart)


async def test_empty_chat_ctx_produces_nothing():
    adapter = LLMAdapter(Agent(TestModel(custom_output_text="never")))

    texts, chunks = await collect(adapter.chat(chat_ctx=ChatContext()))

    assert texts == []
    assert chunks == []


async def test_closing_the_stream_cancels_the_run():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "partial"
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    adapter = LLMAdapter(Agent(FunctionModel(stream_function=stream_fn)))
    stream = adapter.chat(chat_ctx=user_ctx())

    first = await stream.__anext__()
    assert first.delta and first.delta.content == "partial"
    await asyncio.wait_for(started.wait(), timeout=5)

    await asyncio.wait_for(stream.aclose(), timeout=5)
    await asyncio.wait_for(cancelled.wait(), timeout=5)


# --- model / provider labels ---


def test_model_and_provider_labels():
    adapter = LLMAdapter(Agent(TestModel()))
    assert adapter.provider == "Pydantic AI"
    assert adapter.model == "test"

    assert LLMAdapter(Agent()).model == "unknown"


# --- history conversion ---


def test_chat_ctx_to_messages_groups_and_maps_items():
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="sys")
    chat_ctx.add_message(role="user", content="hello")
    chat_ctx.add_message(role="assistant", content="let me check")
    chat_ctx.items.append(
        FunctionCall(call_id="c1", name="lookup", arguments='{"id": 1}'),
    )
    chat_ctx.items.append(
        FunctionCallOutput(call_id="c1", name="lookup", output="found", is_error=False),
    )
    chat_ctx.items.append(
        FunctionCallOutput(call_id="c2", name="lookup", output="boom", is_error=True),
    )
    chat_ctx.add_message(role="assistant", content="")  # no text: skipped
    chat_ctx.add_message(role="assistant", content="it is found")
    chat_ctx.add_message(role="user", content="thanks")

    messages = chat_ctx_to_messages(chat_ctx)

    assert [type(m) for m in messages] == [
        ModelRequest,
        ModelResponse,
        ModelRequest,
        ModelResponse,
        ModelRequest,
    ]

    req0, resp0, req1, resp1, req2 = messages
    assert isinstance(req0, ModelRequest) and isinstance(resp0, ModelResponse)
    assert isinstance(req1, ModelRequest) and isinstance(resp1, ModelResponse)
    assert isinstance(req2, ModelRequest)

    assert [type(p) for p in req0.parts] == [SystemPromptPart, UserPromptPart]

    assert [type(p) for p in resp0.parts] == [TextPart, ToolCallPart]
    call = resp0.parts[1]
    assert isinstance(call, ToolCallPart)
    assert (call.tool_name, call.args, call.tool_call_id) == ("lookup", '{"id": 1}', "c1")

    assert [type(p) for p in req1.parts] == [ToolReturnPart, ToolReturnPart]
    ret = req1.parts[0]
    assert isinstance(ret, ToolReturnPart)
    assert (ret.tool_name, ret.content, ret.tool_call_id) == ("lookup", "found", "c1")
    err = req1.parts[1]
    assert isinstance(err, ToolReturnPart)
    assert (err.tool_name, err.content, err.tool_call_id) == ("lookup", "boom", "c2")

    assert [type(p) for p in resp1.parts] == [TextPart]
    assert isinstance(resp1.parts[0], TextPart)
    assert resp1.parts[0].content == "it is found"

    assert [type(p) for p in req2.parts] == [UserPromptPart]


def test_chat_ctx_to_messages_trailing_user_message():
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="assistant", content="hi")
    chat_ctx.add_message(role="user", content="bye")

    messages = chat_ctx_to_messages(chat_ctx)

    assert [type(m) for m in messages] == [ModelResponse, ModelRequest]
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    assert isinstance(last.parts[0], UserPromptPart)
    assert last.parts[0].content == "bye"
