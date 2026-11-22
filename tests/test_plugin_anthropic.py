"""Unit tests for the Anthropic LLM plugin that do not require a real API key or network."""

from __future__ import annotations

import httpx2 as httpx
import pytest
from anthropic.types import (
    InputJSONDelta,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    TextDelta,
    ToolUseBlock,
)
from anthropic.types.raw_message_delta_event import Delta

from livekit.agents import APIConnectionError, APIConnectOptions, llm
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


class _FakeAnthropicStream:
    def __init__(self, events: list) -> None:
        self._events = iter(events)

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
            raise StopAsyncIteration from None


class TestAnthropicStreamRetry:
    @pytest.mark.asyncio
    async def test_retry_creates_a_fresh_stream_awaitable(self) -> None:
        calls = 0

        async def failing_stream():
            raise RuntimeError("transient connect failure")

        async def empty_stream():
            return _FakeAnthropicStream([RawMessageStopEvent(type="message_stop")])

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

    @pytest.mark.virtual_time
    @pytest.mark.parametrize(
        ("events", "expected_attempts", "expected_text"),
        [
            pytest.param([], 3, "", id="no-output"),
            pytest.param(
                [
                    RawContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=0,
                        delta=TextDelta(type="text_delta", text="Hello"),
                    )
                ],
                1,
                "Hello",
                id="after-text",
            ),
        ],
    )
    async def test_stream_ending_before_message_stop_raises(
        self, events: list, expected_attempts: int, expected_text: str
    ) -> None:
        """A body that ends without message_stop is truncated, not a complete response."""
        calls = 0

        async def truncated_stream():
            return _FakeAnthropicStream(events)

        def create_stream():
            nonlocal calls
            calls += 1
            return truncated_stream()

        stream = LLMStream(
            _make_llm(),
            create_anthropic_stream=create_stream,
            chat_ctx=llm.ChatContext.empty(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )

        text = ""
        with pytest.raises(APIConnectionError):
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content

        assert calls == expected_attempts
        assert text == expected_text

    @pytest.mark.asyncio
    async def test_stream_ending_after_stop_reason_is_complete(self) -> None:
        """A message_delta carrying stop_reason ends the message even if message_stop is lost."""
        events = [
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text="Hello"),
            ),
            RawMessageDeltaEvent(
                type="message_delta",
                delta=Delta(stop_reason="end_turn", stop_sequence=None),
                usage=MessageDeltaUsage(output_tokens=1),
            ),
        ]

        async def stream_without_message_stop():
            return _FakeAnthropicStream(events)

        stream = LLMStream(
            _make_llm(),
            create_anthropic_stream=stream_without_message_stop,
            chat_ctx=llm.ChatContext.empty(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0),
        )

        response = await stream.collect()

        assert response.text == "Hello"

    @pytest.mark.virtual_time
    async def test_retry_after_a_cut_tool_call_drops_its_partial_state(self) -> None:
        """A tool call cut short by a truncated attempt must not leak into the retry."""
        attempts = [
            [
                RawContentBlockStartEvent(
                    type="content_block_start",
                    index=0,
                    content_block=ToolUseBlock(type="tool_use", id="call_1", name="f", input={}),
                ),
                RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=InputJSONDelta(type="input_json_delta", partial_json='{"a":'),
                ),
            ],
            [
                RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=TextDelta(type="text_delta", text="Hi"),
                ),
                RawContentBlockStopEvent(type="content_block_stop", index=0),
                RawMessageStopEvent(type="message_stop"),
            ],
        ]

        async def next_stream():
            return _FakeAnthropicStream(attempts.pop(0))

        stream = LLMStream(
            _make_llm(),
            create_anthropic_stream=next_stream,
            chat_ctx=llm.ChatContext.empty(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )

        response = await stream.collect()

        assert response.text == "Hi"
        assert response.tool_calls == []
