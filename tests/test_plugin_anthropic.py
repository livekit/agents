"""Unit tests for the Anthropic LLM plugin that do not require a real API key or network."""

from __future__ import annotations

from typing import Any

import anthropic
import httpx
import pytest

from livekit.agents import APIConnectOptions, llm
from livekit.plugins.anthropic.llm import LLMStream

# these tests are hermetic (fake client, no API key, no network), so they belong to the
# `unit` category that CI actually runs — no job in `.github/workflows` runs `--plugin`
pytestmark = pytest.mark.unit


def _make_llm(**kwargs):
    from livekit.plugins.anthropic import LLM

    return LLM(api_key="sk-ant-test", **kwargs)


class TestSDKRetries:
    def test_disabled_by_default(self) -> None:
        """The framework owns retries, so Anthropic SDK retries must default to disabled."""
        llm = _make_llm()
        assert llm._client.max_retries == 0


class TestHttpxTimeoutDefaults:
    def test_default_read_timeout_is_generous(self) -> None:
        """Default read timeout must accommodate adaptive-thinking pauses (≥30 s)."""
        llm = _make_llm()
        read = llm._client._client.timeout.read
        assert read >= 30.0, f"read timeout {read}s is too short for adaptive thinking"

    def test_default_connect_timeout_remains_tight(self) -> None:
        """Connect timeout should stay short so genuine connection failures surface fast."""
        llm = _make_llm()
        connect = llm._client._client.timeout.connect
        assert connect <= 10.0, f"connect timeout {connect}s is unexpectedly long"

    def test_default_timeout_is_split(self) -> None:
        """Default must be an httpx.Timeout object, not a flat scalar."""
        llm = _make_llm()
        t = llm._client._client.timeout
        assert isinstance(t, httpx.Timeout)
        assert t.read != t.connect, "read and connect timeouts should differ in the default"


class TestHttpxTimeoutCustom:
    def test_custom_timeout_honored(self) -> None:
        """A caller-supplied httpx.Timeout is passed through to the httpx client."""
        custom = httpx.Timeout(3.0, read=120.0)
        llm = _make_llm(timeout=custom)
        t = llm._client._client.timeout
        assert t.read == 120.0
        assert t.connect == 3.0

    def test_none_timeout_uses_default(self) -> None:
        """Passing timeout=None must fall back to the built-in default."""
        llm = _make_llm(timeout=None)
        assert llm._client._client.timeout.read >= 30.0

    def test_explicit_client_bypasses_timeout_param(self) -> None:
        """When a pre-built client= is supplied, timeout= is ignored (client wins)."""
        import anthropic

        tight_client = anthropic.AsyncClient(
            api_key="sk-ant-test",
            http_client=httpx.AsyncClient(timeout=httpx.Timeout(1.0)),
        )
        # timeout= argument should have no effect here
        llm = _make_llm(client=tight_client, timeout=httpx.Timeout(5.0, read=999.0))
        assert llm._client._client.timeout.read == 1.0


class _EmptyAnthropicStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _ScriptedAnthropicStream(_EmptyAnthropicStream):
    """Yields the given events, then raises — the shape of a stream dying mid-turn."""

    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        raise RuntimeError("stream died mid-turn")


class TestAnthropicStreamRetry:
    @pytest.mark.asyncio
    async def test_retry_creates_a_fresh_stream_awaitable(self) -> None:
        calls = 0

        async def failing_stream():
            raise RuntimeError("transient connect failure")

        async def empty_stream():
            return _EmptyAnthropicStream()

        def create_stream():
            nonlocal calls
            calls += 1
            return failing_stream() if calls == 1 else empty_stream()

        stream = LLMStream(
            _make_llm(),
            create_anthropic_stream=create_stream,
            chat_ctx=llm.ChatContext.empty(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )

        response = await stream.collect()

        assert calls == 2
        assert response.usage is not None


class TestPerAttemptState:
    """`_run` is re-entered on every retry with the same instance."""

    async def _collect_after_one_failed_attempt(
        self, first_attempt_events: list[Any]
    ) -> tuple[LLMStream, llm.CollectedResponse]:
        attempts = 0

        async def create_stream():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return _ScriptedAnthropicStream(first_attempt_events)
            return _EmptyAnthropicStream()

        stream = LLMStream(
            _make_llm(),
            create_anthropic_stream=create_stream,
            chat_ctx=llm.ChatContext.empty(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )
        response = await stream.collect()
        assert attempts == 2
        return stream, response

    @pytest.mark.asyncio
    async def test_chain_of_thought_latch_does_not_survive_a_retry(self) -> None:
        """Left set, it would swallow every text chunk of the successful attempt."""
        cot = anthropic.types.RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=anthropic.types.TextDelta(type="text_delta", text="<thinking>reasoning"),
        )

        stream, _ = await self._collect_after_one_failed_attempt([cot])

        assert stream._ignoring_cot is False

    @pytest.mark.asyncio
    async def test_half_read_tool_call_does_not_survive_a_retry(self) -> None:
        start = anthropic.types.RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=anthropic.types.ToolUseBlock(
                type="tool_use", id="toolu_1", name="lookup_order", input={}
            ),
        )

        stream, _ = await self._collect_after_one_failed_attempt([start])

        assert stream._tool_call_id is None
        assert stream._fnc_name is None
        assert stream._fnc_raw_arguments is None

    @pytest.mark.asyncio
    async def test_cache_token_counters_do_not_survive_a_retry(self) -> None:
        """`message_start` only reassigns them when non-zero, so a stale write is double-counted."""
        # input_tokens is left at 0 so prompt_tokens reflects the cache counters alone
        start = anthropic.types.RawMessageStartEvent(
            type="message_start",
            message=anthropic.types.Message(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-opus-5",
                content=[],
                usage=anthropic.types.Usage(
                    input_tokens=0,
                    output_tokens=0,
                    cache_creation_input_tokens=1000,
                    cache_read_input_tokens=500,
                ),
            ),
        )

        stream, response = await self._collect_after_one_failed_attempt([start])

        assert response.usage is not None
        assert response.usage.cache_creation_tokens == 0
        assert response.usage.cache_read_tokens == 0
        assert response.usage.prompt_cached_tokens == 0
        assert response.usage.prompt_tokens == 0
        assert stream._cache_creation_tokens == 0
        assert stream._cache_read_tokens == 0
