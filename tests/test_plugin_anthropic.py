"""Unit tests for the Anthropic LLM plugin that do not require a real API key or network."""

from __future__ import annotations

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


def _ctx_with_user_message() -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hello")
    return ctx


async def _request_kwargs(anthropic_llm, **chat_kwargs) -> dict:
    """Run one chat() against a stubbed SDK client and return the kwargs it was called with."""
    from unittest.mock import AsyncMock

    create = AsyncMock(return_value=_EmptyAnthropicStream())
    anthropic_llm._client.messages.create = create

    await anthropic_llm.chat(chat_ctx=_ctx_with_user_message(), **chat_kwargs).collect()

    assert create.await_count == 1
    return create.call_args.kwargs


class TestThinkingDefaults:
    """Claude 5 models think by default. Inside the plugin's 1024-token default budget,
    Sonnet 5 and Haiku 5.5 spend it all thinking and return no text, so the plugin turns
    thinking off on those two unless the caller chooses otherwise."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-haiku-5-5"])
    async def test_thinking_disabled_on_models_that_reply_empty_by_default(
        self, model: str
    ) -> None:
        kwargs = await _request_kwargs(_make_llm(model=model))
        assert kwargs["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model",
        [
            # the default model, and the 4.x generation: no thinking unless requested
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            # reject {"type": "disabled"} with a 400, so the plugin must not send it
            "claude-sonnet-5-5",
            "claude-opus-5-5",
            # accepts "disabled" but answers on API defaults; left to the API
            "claude-opus-5",
            # unknown or future ids must get the API default, never a guessed value
            "claude-future-model",
        ],
    )
    async def test_no_thinking_field_on_other_models(self, model: str) -> None:
        kwargs = await _request_kwargs(_make_llm(model=model))
        assert "thinking" not in kwargs

    @pytest.mark.asyncio
    async def test_prefix_of_a_listed_model_does_not_match(self) -> None:
        """The table is keyed by exact id: ``claude-sonnet-5-5`` must not inherit the
        ``claude-sonnet-5`` default, which it would reject."""
        kwargs = await _request_kwargs(_make_llm(model="claude-sonnet-5-5"))
        assert "thinking" not in kwargs

    @pytest.mark.asyncio
    async def test_explicit_thinking_overrides_default(self) -> None:
        adaptive = {"type": "adaptive"}
        kwargs = await _request_kwargs(_make_llm(model="claude-sonnet-5", thinking=adaptive))
        assert kwargs["thinking"] == adaptive

    @pytest.mark.asyncio
    async def test_explicit_none_sends_no_thinking_field(self) -> None:
        kwargs = await _request_kwargs(_make_llm(model="claude-sonnet-5", thinking=None))
        assert "thinking" not in kwargs

    @pytest.mark.asyncio
    async def test_explicit_thinking_passes_through_on_models_without_default(self) -> None:
        between_tools = {"type": "between_tools"}
        kwargs = await _request_kwargs(_make_llm(model="claude-sonnet-5-5", thinking=between_tools))
        assert kwargs["thinking"] == between_tools

    @pytest.mark.asyncio
    async def test_extra_kwargs_thinking_wins_over_constructor_and_default(self) -> None:
        adaptive = {"type": "adaptive"}
        kwargs = await _request_kwargs(
            _make_llm(model="claude-sonnet-5", thinking={"type": "disabled"}),
            extra_kwargs={"thinking": adaptive},
        )
        assert kwargs["thinking"] == adaptive


class TestThinkingBudgetAndMaxTokens:
    """An enabled thinking budget is spent inside ``max_tokens``, and Anthropic rejects a
    request whose budget is not below ``max_tokens``. The plugin's 1024 default would
    collide with any valid budget (the minimum is 1024)."""

    @pytest.mark.asyncio
    async def test_default_max_tokens_leaves_room_above_the_budget(self) -> None:
        enabled = {"type": "enabled", "budget_tokens": 2048}
        kwargs = await _request_kwargs(_make_llm(model="claude-haiku-4-5", thinking=enabled))
        assert kwargs["thinking"] == enabled
        assert kwargs["max_tokens"] == 2048 + 1024

    @pytest.mark.asyncio
    async def test_default_max_tokens_unchanged_without_a_budget(self) -> None:
        kwargs = await _request_kwargs(_make_llm(model="claude-sonnet-5"))
        assert kwargs["max_tokens"] == 1024

        adaptive = {"type": "adaptive"}
        kwargs = await _request_kwargs(_make_llm(model="claude-sonnet-5-5", thinking=adaptive))
        assert kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_above_the_budget_is_kept(self) -> None:
        enabled = {"type": "enabled", "budget_tokens": 2048}
        kwargs = await _request_kwargs(
            _make_llm(model="claude-haiku-4-5", thinking=enabled, max_tokens=4096)
        )
        assert kwargs["max_tokens"] == 4096

    @pytest.mark.parametrize("max_tokens", [1024, 2048])
    def test_explicit_max_tokens_not_above_the_budget_is_rejected(self, max_tokens: int) -> None:
        with pytest.raises(ValueError, match="must be greater than thinking budget_tokens"):
            _make_llm(
                model="claude-haiku-4-5",
                thinking={"type": "enabled", "budget_tokens": 2048},
                max_tokens=max_tokens,
            )

    @pytest.mark.asyncio
    async def test_budget_from_extra_kwargs_also_raises_the_default(self) -> None:
        enabled = {"type": "enabled", "budget_tokens": 1024}
        kwargs = await _request_kwargs(
            _make_llm(model="claude-haiku-4-5"), extra_kwargs={"thinking": enabled}
        )
        assert kwargs["max_tokens"] == 2048


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
