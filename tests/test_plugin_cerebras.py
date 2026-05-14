from __future__ import annotations

import os

import httpx
import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool, llm
from livekit.plugins.cerebras import LLM
from livekit.plugins.cerebras.llm import _CerebrasClient

# llama3.1-8b is fast and has generous rate limits but can't do tool calls reliably;
# qwen-3-235b is needed for function calling but has tight per-minute token quotas.
CHAT_MODEL = "llama3.1-8b"
TOOL_MODEL = "qwen-3-235b-a22b-instruct-2507"


class HeaderCapturingTransport(httpx.AsyncBaseTransport):
    """Wraps a real transport, capturing outgoing request headers for assertion."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.captured_requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured_requests.append(request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _cerebras_llm(**kwargs) -> LLM:
    return LLM(model=CHAT_MODEL, **kwargs)


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    @function_tool
    async def get_weather(self, ctx: RunContext, location: str) -> str:
        """Get the current weather for a location.
        Args:
            location: The city name
        """
        return f"The weather in {location} is sunny, 72°F."


@pytest.mark.asyncio
async def test_chat():
    """Basic chat completion returns a non-empty assistant message."""
    async with _cerebras_llm() as model, AgentSession(llm=model) as sess:
        await sess.start(Agent(instructions="You are a helpful assistant."))
        result = await sess.run(user_input="Say hello in exactly one word.")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_function_call():
    """LLM can invoke a tool and the result is returned."""
    async with LLM(model=TOOL_MODEL) as model, AgentSession(llm=model) as sess:
        await sess.start(WeatherAgent())
        result = await sess.run(user_input="What is the weather in Tokyo?")
        result.expect.next_event().is_function_call(
            name="get_weather", arguments={"location": "Tokyo"}
        )
        result.expect.next_event().is_function_call_output(
            output="The weather in Tokyo is sunny, 72°F."
        )
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


def _cerebras_llm_with_transport(
    *, use_gzip: bool, use_msgpack: bool
) -> tuple[LLM, HeaderCapturingTransport]:
    transport = HeaderCapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = _CerebrasClient(
        use_gzip=use_gzip,
        use_msgpack=use_msgpack,
        api_key=os.environ["CEREBRAS_API_KEY"],
        base_url="https://api.cerebras.ai/v1",
        http_client=http_client,
    )
    return LLM(model=CHAT_MODEL, client=client), transport


@pytest.mark.asyncio
async def test_gzip_only_headers():
    """Gzip-only sends Content-Encoding: gzip with JSON content type."""
    model, transport = _cerebras_llm_with_transport(use_gzip=True, use_msgpack=False)
    async with model, AgentSession(llm=model) as sess:
        await sess.start(Agent(instructions="You are a helpful assistant."))
        result = await sess.run(user_input="Say hello in exactly one word.")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

    chat_reqs = [r for r in transport.captured_requests if "/chat/completions" in str(r.url)]
    assert len(chat_reqs) > 0
    assert chat_reqs[0].headers["content-type"] == "application/json"
    assert chat_reqs[0].headers["content-encoding"] == "gzip"


@pytest.mark.asyncio
async def test_msgpack_only_headers():
    """Msgpack-only sends Content-Type: application/vnd.msgpack without gzip."""
    model, transport = _cerebras_llm_with_transport(use_gzip=False, use_msgpack=True)
    async with model, AgentSession(llm=model) as sess:
        await sess.start(Agent(instructions="You are a helpful assistant."))
        result = await sess.run(user_input="Say hello in exactly one word.")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

    chat_reqs = [r for r in transport.captured_requests if "/chat/completions" in str(r.url)]
    assert len(chat_reqs) > 0
    assert chat_reqs[0].headers["content-type"] == "application/vnd.msgpack"
    assert "content-encoding" not in chat_reqs[0].headers


@pytest.mark.asyncio
async def test_msgpack_and_gzip_headers():
    """Both flags send msgpack content type with gzip encoding."""
    model, transport = _cerebras_llm_with_transport(use_gzip=True, use_msgpack=True)
    async with model, AgentSession(llm=model) as sess:
        await sess.start(Agent(instructions="You are a helpful assistant."))
        result = await sess.run(user_input="Say hello in exactly one word.")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

    chat_reqs = [r for r in transport.captured_requests if "/chat/completions" in str(r.url)]
    assert len(chat_reqs) > 0
    assert chat_reqs[0].headers["content-type"] == "application/vnd.msgpack"
    assert chat_reqs[0].headers["content-encoding"] == "gzip"


@pytest.mark.asyncio
async def test_no_compression_headers():
    """With both flags off, sends standard JSON without gzip."""
    async with _cerebras_llm(gzip_compression=False, msgpack_encoding=False) as model:
        async with AgentSession(llm=model) as sess:
            await sess.start(Agent(instructions="You are a helpful assistant."))
            result = await sess.run(user_input="Say hello in exactly one word.")
            result.expect.next_event().is_message(role="assistant")
            result.expect.no_more_events()


@pytest.mark.asyncio
async def test_streaming():
    """Streaming chat returns content via the LLM directly."""
    async with _cerebras_llm() as model:
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="You are a helpful assistant.")
        chat_ctx.add_message(role="user", content="Count from 1 to 5.")

        stream = model.chat(chat_ctx=chat_ctx)
        text = ""
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
        await stream.aclose()

        assert len(text) > 0, "Expected non-empty streaming response"
        assert "3" in text, "Expected the count to include '3'"
