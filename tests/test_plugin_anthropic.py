"""Unit tests for the Anthropic LLM plugin that do not require a real API key or network."""

from __future__ import annotations

from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from livekit.agents import APIConnectOptions, llm
from livekit.plugins.anthropic.llm import LLMStream

pytestmark = pytest.mark.plugin("anthropic")


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


class _FakeAnthropicStream:
    """Replays a fixed list of stream events, like the SDK's message stream."""

    def __init__(self, events: list[Any], error: Exception | None = None) -> None:
        self._events = iter(events)
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            if self._error is not None:
                raise self._error from None
            raise StopAsyncIteration from None


def _text_block(*parts: str) -> list[Any]:
    """One text content block whose text arrives in the given deltas."""
    return [
        anthropic.types.RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=anthropic.types.TextBlock(type="text", text=""),
        ),
        *(
            anthropic.types.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=anthropic.types.TextDelta(type="text_delta", text=part),
            )
            for part in parts
        ),
        anthropic.types.RawContentBlockStopEvent(type="content_block_stop", index=0),
    ]


async def _reply_text(*attempts: list[Any] | _FakeAnthropicStream) -> str:
    """Runs one LLMStream; each attempt is one provider stream, used in turn by retries."""
    streams = iter(
        a if isinstance(a, _FakeAnthropicStream) else _FakeAnthropicStream(a) for a in attempts
    )

    async def open_stream():
        return next(streams)

    stream = LLMStream(
        _make_llm(),
        create_anthropic_stream=open_stream,
        chat_ctx=llm.ChatContext.empty(),
        tools=[],
        conn_options=APIConnectOptions(max_retry=len(attempts) - 1, retry_interval=0),
    )
    response = await stream.collect()
    return response.text


class TestThinkingTags:
    @pytest.mark.asyncio
    async def test_text_without_tags_is_unchanged(self) -> None:
        assert await _reply_text(_text_block("Booking ", "it now.")) == "Booking it now."

    @pytest.mark.asyncio
    async def test_thinking_opened_and_closed_in_one_delta(self) -> None:
        events = _text_block("<thinking>check the slot</thinking>", "Booking it now.")
        assert await _reply_text(events) == "Booking it now."

    @pytest.mark.asyncio
    async def test_closing_tag_split_across_deltas(self) -> None:
        events = _text_block("<thinking>check the slot", "</", "thinking>Booking it now.")
        assert await _reply_text(events) == "Booking it now."

    @pytest.mark.asyncio
    async def test_opening_tag_split_across_deltas(self) -> None:
        events = _text_block("<think", "ing>check the slot</thinking>", "Booking it now.")
        assert await _reply_text(events) == "Booking it now."

    @pytest.mark.asyncio
    async def test_text_held_as_a_possible_tag_is_released_at_block_end(self) -> None:
        assert await _reply_text(_text_block("Is 3 <")) == "Is 3 <"

    @pytest.mark.asyncio
    async def test_unclosed_thinking_does_not_silence_the_next_block(self) -> None:
        events = [*_text_block("<thinking>check the slot"), *_text_block("Booking it now.")]
        assert await _reply_text(events) == "Booking it now."

    @pytest.mark.asyncio
    async def test_a_retry_drops_text_held_by_the_failed_attempt(self) -> None:
        # the first attempt fails right after the start of a tag, before any chunk is sent
        failed = _FakeAnthropicStream(_text_block("<think")[:2], error=RuntimeError("reset"))
        assert await _reply_text(failed, _text_block("Hello.")) == "Hello."
