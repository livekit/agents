"""Unit tests for the Anthropic LLM plugin that do not require a real API key or network."""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import httpx
import pytest

from livekit.agents import APIConnectOptions, function_tool, llm
from livekit.plugins.anthropic.llm import LLMStream
from livekit.plugins.anthropic.models import (
    _model_supports_prefill,
    _model_supports_sampling_params,
    _model_thinking_support,
    _model_version,
)

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


class _RecordingAnthropicClient:
    """Stand-in for anthropic.AsyncClient that records the request it was handed."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._base_url = httpx.URL("https://api.anthropic.com")
        outer = self

        class _Messages:
            async def create(self, **kwargs: Any) -> _EmptyAnthropicStream:
                outer.requests.append(kwargs)
                return _EmptyAnthropicStream()

        self.messages = _Messages()


async def _request_for(model: str, *, chat_ctx: llm.ChatContext | None = None, **kwargs: Any):
    """Run one chat() round-trip against a fake client and return the request payload."""
    client = _RecordingAnthropicClient()
    anthropic_llm = _make_llm(model=model, client=client, **kwargs)
    await anthropic_llm.chat(chat_ctx=chat_ctx or llm.ChatContext.empty()).collect()
    assert len(client.requests) == 1
    return client.requests[0]


class TestModelVersionParsing:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-4-6", (4, 6)),
            ("claude-sonnet-4-6", (4, 6)),
            ("claude-opus-4-8", (4, 8)),
            ("claude-opus-5", (5,)),
            ("claude-sonnet-5", (5,)),
            ("claude-fable-5", (5,)),
            ("claude-haiku-4-5", (4, 5)),
            # the trailing 8-digit segment is a release date, not a version
            ("claude-sonnet-4-20250514", (4,)),
            ("claude-opus-4-1-20250805", (4, 1)),
            ("claude-3-5-sonnet-20241022", (3, 5)),
            ("claude-3-haiku-20240307", (3,)),
            # provider prefixes and snapshot suffixes must not shift the version
            ("anthropic.claude-opus-5", (5,)),
            ("anthropic/claude-opus-4-6", (4, 6)),
            ("claude-opus-4-5@20251101", (4, 5)),  # Vertex snapshot
            ("claude-opus-5[1m]", (5,)),
            ("claude-opus-4-6-fast", (4, 6)),
            ("anthropic.claude-3-5-sonnet-20241022-v2:0", (3, 5)),  # legacy Bedrock ARN
            # a number in a proxy's own name is not the model's version
            ("gw-1-claude-opus-5", (5,)),
            # gateway / proxy aliases that hide the family carry no version at all
            ("my-claude-proxy", ()),
        ],
    )
    def test_version_is_read_from_the_id(self, model: str, expected: tuple[int, ...]) -> None:
        assert _model_version(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-opus-5",
            "anthropic/claude-opus-4-6",
            "claude-opus-5[1m]",
            "gw-1-claude-opus-5",
        ],
    )
    def test_prefixed_and_suffixed_ids_keep_the_4_6_guards(self, model: str) -> None:
        """A misparsed id silently re-enables prefilling — the 400 this guard exists for."""
        assert not _model_supports_prefill(model)


class TestPrefillGuard:
    """Regression guard for livekit/agents#4907 (400 on prefilled assistant messages)."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
        ],
    )
    def test_prefill_rejected_from_4_6_onwards(self, model: str) -> None:
        assert not _model_supports_prefill(model)

    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-3-5-sonnet-20241022",
            # an unrecognised id keeps the behaviour it had before the guard existed
            "my-claude-proxy",
        ],
    )
    def test_prefill_still_allowed_elsewhere(self, model: str) -> None:
        assert _model_supports_prefill(model)

    @pytest.mark.asyncio
    async def test_trailing_assistant_message_is_closed_with_a_user_turn(self) -> None:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="hello")
        chat_ctx.add_message(role="assistant", content="prefilled")

        request = await _request_for("claude-opus-5", chat_ctx=chat_ctx)
        assert request["messages"][-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_prefill_is_preserved_on_older_models(self) -> None:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="hello")
        chat_ctx.add_message(role="assistant", content="prefilled")

        request = await _request_for("claude-haiku-4-5", chat_ctx=chat_ctx)
        assert request["messages"][-1]["role"] == "assistant"


class TestSamplingParams:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
        ],
    )
    def test_rejected_from_4_7_onwards(self, model: str) -> None:
        assert not _model_supports_sampling_params(model)

    @pytest.mark.parametrize(
        "model",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5", "my-claude-proxy"],
    )
    def test_accepted_up_to_4_6(self, model: str) -> None:
        assert _model_supports_sampling_params(model)

    @pytest.mark.asyncio
    async def test_dropped_from_the_request_instead_of_failing_it(self) -> None:
        request = await _request_for("claude-opus-5", temperature=0.7, top_k=5)
        assert "temperature" not in request
        assert "top_k" not in request

    @pytest.mark.asyncio
    @pytest.mark.parametrize("param", ["temperature", "top_p", "top_k"])
    async def test_dropped_when_they_arrive_through_extra_kwargs(self, param: str) -> None:
        """`extra_kwargs` reaches the payload first; leaving it there still 400s."""
        client = _RecordingAnthropicClient()
        anthropic_llm = _make_llm(model="claude-opus-5", client=client)
        await anthropic_llm.chat(
            chat_ctx=llm.ChatContext.empty(), extra_kwargs={param: 0.7}
        ).collect()

        assert param not in client.requests[0]

    @pytest.mark.asyncio
    async def test_warns_about_the_ones_dropped_from_extra_kwargs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _RecordingAnthropicClient()
        anthropic_llm = _make_llm(model="claude-opus-5", client=client)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            await anthropic_llm.chat(
                chat_ctx=llm.ChatContext.empty(), extra_kwargs={"top_p": 0.9}
            ).collect()

        warnings = [
            r.getMessage() for r in caplog.records if "sampling parameters" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "top_p" in warnings[0]

    @pytest.mark.asyncio
    async def test_still_sent_to_models_that_accept_them(self) -> None:
        request = await _request_for("claude-sonnet-4-6", temperature=0.7, top_k=5)
        assert request["temperature"] == 0.7
        assert request["top_k"] == 5

    @pytest.mark.asyncio
    async def test_warns_once_per_instance(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _RecordingAnthropicClient()
        anthropic_llm = _make_llm(model="claude-opus-5", client=client, temperature=0.7)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            for _ in range(3):
                await anthropic_llm.chat(chat_ctx=llm.ChatContext.empty()).collect()

        warnings = [r for r in caplog.records if "sampling parameters" in r.getMessage()]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_no_warning_when_nothing_was_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            await _request_for("claude-opus-5")

        assert not [r for r in caplog.records if "sampling parameters" in r.getMessage()]


class TestThinking:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-sonnet-4-6", "configurable"),
            ("claude-opus-4-8", "configurable"),
            ("claude-opus-5", "configurable"),
            ("claude-sonnet-5", "configurable"),
            ("claude-fable-5", "always_on"),
            ("claude-mythos-5", "always_on"),
            ("claude-haiku-4-5", "unknown"),
            ("claude-3-5-sonnet-20241022", "unknown"),
            ("my-claude-proxy", "unknown"),
        ],
    )
    def test_support_is_derived_from_the_id(self, model: str, expected: str) -> None:
        assert _model_thinking_support(model) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-4-8", "claude-opus-5"])
    async def test_disabled_where_the_model_allows_it(self, model: str) -> None:
        request = await _request_for(model)
        assert request["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_omitted_where_it_cannot_be_disabled(self) -> None:
        """Claude Fable rejects {"type": "disabled"} with a 400, so it must not be sent."""
        request = await _request_for("claude-fable-5")
        assert "thinking" not in request

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["claude-haiku-4-5", "my-claude-proxy"])
    async def test_omitted_on_models_that_predate_the_parameter(self, model: str) -> None:
        request = await _request_for(model)
        assert "thinking" not in request

    @pytest.mark.asyncio
    async def test_extra_kwargs_can_opt_back_in(self) -> None:
        client = _RecordingAnthropicClient()
        anthropic_llm = _make_llm(model="claude-opus-5", client=client)
        await anthropic_llm.chat(
            chat_ctx=llm.ChatContext.empty(),
            extra_kwargs={"thinking": {"type": "adaptive"}},
        ).collect()

        assert client.requests[0]["thinking"] == {"type": "adaptive"}

    @pytest.mark.asyncio
    async def test_each_request_gets_its_own_thinking_dict(self) -> None:
        """The dict lands in the payload by reference; a shared one is process-wide state."""
        client = _RecordingAnthropicClient()
        anthropic_llm = _make_llm(model="claude-opus-5", client=client)
        for _ in range(2):
            await anthropic_llm.chat(chat_ctx=llm.ChatContext.empty()).collect()

        first, second = client.requests[0]["thinking"], client.requests[1]["thinking"]
        assert first is not second

        first["type"] = "adaptive"  # a caller (or a test) mutating one request
        assert (await _request_for("claude-opus-5"))["thinking"] == {"type": "disabled"}


def _one_tool() -> list[Any]:
    """A minimal, valid tool list for `chat(tools=...)`."""

    @function_tool
    async def lookup_order(order_id: str) -> str:
        """Look up the status of an order.

        Args:
            order_id: The identifier of the order.
        """
        return "shipped"

    return [lookup_order]


class TestThinkingWithTools:
    """Thinking blocks are never persisted, which breaks a turn carrying a tool call."""

    async def _chat(self, model: str, *, tools: list[Any] | None, times: int = 1, **kwargs: Any):
        client = _RecordingAnthropicClient()
        anthropic_llm = _make_llm(model=model, client=client)
        for _ in range(times):
            await anthropic_llm.chat(
                chat_ctx=llm.ChatContext.empty(), tools=tools, **kwargs
            ).collect()
        return client

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if "thinking blocks" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_warns_once_per_instance(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            # not in ChatModels, but `model` takes any string and the guard recognises
            # it from the id as a model that always thinks
            await self._chat("claude-fable-5", tools=_one_tool(), times=3)

        warnings = self._warnings(caplog)
        assert len(warnings) == 1
        assert "claude-fable-5" in warnings[0]

    @pytest.mark.asyncio
    async def test_warns_when_thinking_is_opted_back_in(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            await self._chat(
                "claude-opus-5",
                tools=_one_tool(),
                extra_kwargs={"thinking": {"type": "adaptive"}},
            )

        assert len(self._warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_silent_when_thinking_is_off(self, caplog: pytest.LogCaptureFixture) -> None:
        """The default on a configurable model: no reasoning, nothing to lose."""
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            client = await self._chat("claude-opus-5", tools=_one_tool())

        assert client.requests[0]["thinking"] == {"type": "disabled"}
        assert not self._warnings(caplog)

    @pytest.mark.asyncio
    async def test_silent_without_tools(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without a tool call in the turn, replaying it without thinking blocks is fine."""
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            await self._chat("claude-fable-5", tools=None)

        assert not self._warnings(caplog)


class TestMaxTokens:
    @pytest.mark.asyncio
    async def test_default_stays_small_when_thinking_is_off(self) -> None:
        request = await _request_for("claude-opus-5")
        assert request["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_default_grows_when_the_model_always_thinks(self) -> None:
        """max_tokens caps thinking and the reply together, so reasoning needs room."""
        request = await _request_for("claude-fable-5")
        assert request["max_tokens"] > 1024

    @pytest.mark.asyncio
    async def test_explicit_value_always_wins(self) -> None:
        request = await _request_for("claude-fable-5", max_tokens=256)
        assert request["max_tokens"] == 256


def _thinking_start_event() -> anthropic.types.RawContentBlockStartEvent:
    return anthropic.types.RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=anthropic.types.ThinkingBlock(type="thinking", thinking="", signature=""),
    )


class TestThinkingBlocksInStream:
    async def _stream(self, *, thinking_expected: bool = False) -> LLMStream:
        async def create_stream() -> _EmptyAnthropicStream:
            return _EmptyAnthropicStream()

        stream = LLMStream(
            _make_llm(model="claude-opus-5"),
            create_anthropic_stream=create_stream,
            chat_ctx=llm.ChatContext.empty(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0),
            thinking_expected=thinking_expected,
        )
        # drain the (empty) stream so no background task outlives the test
        await stream.collect()
        return stream

    @pytest.mark.asyncio
    async def test_thinking_text_never_reaches_the_caller(self) -> None:
        stream = await self._stream()
        delta = anthropic.types.RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=anthropic.types.ThinkingDelta(type="thinking_delta", thinking="let me think"),
        )

        assert stream._parse_event(_thinking_start_event()) is None
        assert stream._parse_event(delta) is None
        assert stream._thinking_blocks == 1
        assert stream._thinking_chars == len("let me think")

    @pytest.mark.asyncio
    async def test_signature_delta_is_ignored(self) -> None:
        stream = await self._stream()
        delta = anthropic.types.RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=anthropic.types.SignatureDelta(type="signature_delta", signature="abc"),
        )

        assert stream._parse_event(delta) is None

    @pytest.mark.asyncio
    async def test_redacted_thinking_is_counted(self) -> None:
        stream = await self._stream()
        start = anthropic.types.RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=anthropic.types.RedactedThinkingBlock(
                type="redacted_thinking", data="opaque"
            ),
        )

        assert stream._parse_event(start) is None
        assert stream._thinking_blocks == 1

    @pytest.mark.asyncio
    async def test_unexpected_thinking_is_reported_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stream = await self._stream(thinking_expected=False)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            stream._parse_event(_thinking_start_event())
            stream._parse_event(_thinking_start_event())

        assert len([r for r in caplog.records if "thinking block" in r.getMessage()]) == 1

    @pytest.mark.asyncio
    async def test_expected_thinking_is_not_warned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stream = await self._stream(thinking_expected=True)

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.anthropic"):
            stream._parse_event(_thinking_start_event())

        assert not [r for r in caplog.records if "thinking block" in r.getMessage()]


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
            _make_llm(model="claude-opus-5"),
            create_anthropic_stream=create_stream,
            chat_ctx=llm.ChatContext.empty(),
            tools=[],
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )
        response = await stream.collect()
        assert attempts == 2
        return stream, response

    @pytest.mark.asyncio
    async def test_thinking_counters_do_not_accumulate_across_retries(self) -> None:
        delta = anthropic.types.RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=anthropic.types.ThinkingDelta(type="thinking_delta", thinking="let me think"),
        )

        stream, _ = await self._collect_after_one_failed_attempt([_thinking_start_event(), delta])

        assert stream._thinking_blocks == 0
        assert stream._thinking_chars == 0

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
