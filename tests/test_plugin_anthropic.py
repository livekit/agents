"""Unit tests for the Anthropic LLM plugin that do not require a real API key or network."""

from __future__ import annotations

import httpx
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


class TestModelDisablesPrefill:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-6",
            "claude-fable-4-6",
            "claude-sonnet-4-7",
            "claude-opus-4-7",
            "claude-sonnet-4-8",
            "claude-opus-4-8",
            "claude-sonnet-4-9",
            "claude-opus-4-9",
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-haiku-5",
            "claude-fable-5",
            "claude-sonnet-5-1",
            "claude-sonnet-4-6-20260101",
        ],
    )
    def test_disables_prefill_newer_models(self, model: str) -> None:
        from livekit.plugins.anthropic.llm import _model_disables_prefill

        assert _model_disables_prefill(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "gpt-4o",
        ],
    )
    def test_allows_prefill_older_models(self, model: str) -> None:
        from livekit.plugins.anthropic.llm import _model_disables_prefill

        assert _model_disables_prefill(model) is False


class TestAnthropicChatContextFormatting:
    def test_trailing_assistant_injects_dummy_user_message(self) -> None:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="user", content="Hello")
        ctx.add_message(role="assistant", content="Hi there")

        messages, _ = ctx.to_provider_format(format="anthropic", inject_trailing_user_message=True)

        assert len(messages) == 3
        assert messages[-1] == {"role": "user", "content": [{"text": ".", "type": "text"}]}

    def test_trailing_assistant_with_tool_use_skips_dummy_user_message(self) -> None:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="user", content="Check weather")
        ctx.insert(
            llm.FunctionCall(call_id="call_1", name="get_weather", arguments='{"city": "SF"}')
        )

        messages, _ = ctx.to_provider_format(format="anthropic", inject_trailing_user_message=True)

        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"
        assert any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in messages[-1]["content"]
        )

    def test_trailing_assistant_with_text_and_tool_use_skips_dummy_user_message(self) -> None:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="user", content="Check weather")
        ctx.add_message(role="assistant", content="Looking up...")
        ctx.insert(
            llm.FunctionCall(call_id="call_1", name="get_weather", arguments='{"city": "SF"}')
        )

        messages, _ = ctx.to_provider_format(format="anthropic", inject_trailing_user_message=True)

        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"
        assert any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in messages[-1]["content"]
        )

    def test_trailing_assistant_inject_false_skips_dummy_user_message(self) -> None:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="user", content="Hello")
        ctx.add_message(role="assistant", content="Hi there")

        messages, _ = ctx.to_provider_format(format="anthropic", inject_trailing_user_message=False)

        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"
