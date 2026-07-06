from __future__ import annotations

import asyncio
import dataclasses

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.llm import (
    ChatContext,
    FunctionCallOutput,
    LLMStream,
    Tool,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.redaction import (
    REDACTION_FAILURE_MARKER,
    NoopRedactor,
    RedactionContext,
    RedactionOptions,
    RedactionResult,
    RedactionSink,
    RegexRedactor,
    redact_chat_ctx,
)

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

_CTX = RedactionContext(sink=RedactionSink.LLM, role="user")


@pytest.mark.asyncio
async def test_noop_redactor_returns_text_unchanged() -> None:
    text = "my card is 4242 4242 4242 4242"
    result = await NoopRedactor().redact(text, _CTX)
    assert result.text == text
    assert result.entities == []


@pytest.mark.asyncio
async def test_regex_redactor_redacts_luhn_valid_credit_card() -> None:
    result = await RegexRedactor().redact("my card is 4242 4242 4242 4242", _CTX)
    assert "4242" not in result.text
    assert "[CREDIT_CARD]" in result.text
    assert len(result.entities) == 1
    assert result.entities[0].type == "CREDIT_CARD"
    assert (result.entities[0].start, result.entities[0].end) == (11, 30)


@pytest.mark.asyncio
async def test_regex_redactor_ignores_luhn_invalid_number() -> None:
    text = "order number 4242 4242 4242 4243"
    result = await RegexRedactor().redact(text, _CTX)
    assert result.text == text
    assert result.entities == []


@pytest.mark.asyncio
async def test_regex_redactor_redacts_ssn() -> None:
    result = await RegexRedactor().redact("my ssn is 856-45-6789", _CTX)
    assert "856-45-6789" not in result.text
    assert "[US_SSN]" in result.text
    assert [e.type for e in result.entities] == ["US_SSN"]


@pytest.mark.asyncio
async def test_regex_redactor_ignores_invalid_ssn() -> None:
    text = "code 000-45-6789 and 856-00-6789 and 902-45-6789"
    result = await RegexRedactor().redact(text, _CTX)
    assert result.text == text
    assert result.entities == []


@pytest.mark.asyncio
async def test_regex_redactor_redacts_valid_iban() -> None:
    result = await RegexRedactor().redact("send to DE89370400440532013000 please", _CTX)
    assert "DE89370400440532013000" not in result.text
    assert "[IBAN]" in result.text
    assert [e.type for e in result.entities] == ["IBAN"]


@pytest.mark.asyncio
async def test_regex_redactor_ignores_invalid_iban() -> None:
    text = "ref DE00370400440532013000"
    result = await RegexRedactor().redact(text, _CTX)
    assert result.text == text
    assert result.entities == []


@pytest.mark.asyncio
async def test_regex_redactor_is_idempotent() -> None:
    redactor = RegexRedactor()
    once = await redactor.redact("card 4242 4242 4242 4242 ssn 856-45-6789", _CTX)
    twice = await redactor.redact(once.text, _CTX)
    assert twice.text == once.text
    assert twice.entities == []


@pytest.mark.asyncio
async def test_redacted_entities_carry_no_raw_values() -> None:
    result = await RegexRedactor().redact("my card is 4242 4242 4242 4242", _CTX)
    entity = result.entities[0]
    assert {f.name for f in dataclasses.fields(entity)} == {"type", "start", "end"}


class _RaisingRedactor:
    async def redact(self, text: str, ctx: RedactionContext) -> RedactionResult:
        raise RuntimeError("detector crashed")


class _HangingRedactor:
    async def redact(self, text: str, ctx: RedactionContext) -> RedactionResult:
        await asyncio.sleep(10)
        return RedactionResult(text=text)


@pytest.mark.asyncio
async def test_redact_chat_ctx_returns_redacted_copy_leaving_original_raw() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="my card is 4242 4242 4242 4242")
    opts = RedactionOptions(redactor=RegexRedactor())

    redacted = await redact_chat_ctx(chat_ctx, opts, sink=RedactionSink.LLM)

    assert redacted.items[0].text_content == "my card is [CREDIT_CARD]"
    # the original context (i.e. session history) must keep the raw value
    assert chat_ctx.items[0].text_content == "my card is 4242 4242 4242 4242"


@pytest.mark.asyncio
async def test_redact_chat_ctx_redacts_function_call_output() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.insert(
        FunctionCallOutput(
            name="lookup_account",
            call_id="call_1",
            output="account iban is DE89370400440532013000",
            is_error=False,
        )
    )
    opts = RedactionOptions(redactor=RegexRedactor())

    redacted = await redact_chat_ctx(chat_ctx, opts, sink=RedactionSink.LLM)

    assert redacted.items[0].output == "account iban is [IBAN]"
    assert chat_ctx.items[0].output == "account iban is DE89370400440532013000"


@pytest.mark.asyncio
async def test_redact_chat_ctx_fails_closed_on_redactor_error() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="my ssn is 856-45-6789")
    opts = RedactionOptions(redactor=_RaisingRedactor())

    redacted = await redact_chat_ctx(chat_ctx, opts, sink=RedactionSink.LLM)

    assert redacted.items[0].text_content == REDACTION_FAILURE_MARKER
    assert chat_ctx.items[0].text_content == "my ssn is 856-45-6789"


@pytest.mark.asyncio
async def test_redact_chat_ctx_fails_closed_on_timeout() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="my ssn is 856-45-6789")
    opts = RedactionOptions(redactor=_HangingRedactor(), timeout=0.01)

    redacted = await redact_chat_ctx(chat_ctx, opts, sink=RedactionSink.LLM)

    assert redacted.items[0].text_content == REDACTION_FAILURE_MARKER


_SENTINEL_TEXT = "my card is 4242 4242 4242 4242"
_REDACTED_TEXT = "my card is [CREDIT_CARD]"


class _CapturingFakeLLM(FakeLLM):
    """FakeLLM that records every message/tool-output text it receives."""

    def __init__(self, *, fake_responses: list[FakeLLMResponse] | None = None) -> None:
        super().__init__(fake_responses=fake_responses)
        self.seen_texts: list[str] = []

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict] = NOT_GIVEN,
    ) -> LLMStream:
        for item in chat_ctx.items:
            if item.type == "message" and item.text_content:
                self.seen_texts.append(item.text_content)
            elif item.type == "function_call_output":
                self.seen_texts.append(item.output)
        return super().chat(chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)


@pytest.mark.asyncio
async def test_llm_sink_redacts_chat_ctx_sent_to_llm() -> None:
    fake_llm = _CapturingFakeLLM(
        fake_responses=[
            FakeLLMResponse(input=_REDACTED_TEXT, content="ok", ttft=0.01, duration=0.02)
        ]
    )
    session = AgentSession(llm=fake_llm, redaction=RedactionOptions(redactor=RegexRedactor()))
    await session.start(Agent(instructions="test agent"))
    try:
        await session.generate_reply(user_input=_SENTINEL_TEXT)

        assert fake_llm.seen_texts
        assert all("4242" not in text for text in fake_llm.seen_texts)
        assert _REDACTED_TEXT in fake_llm.seen_texts
        # the in-memory history must keep the raw value
        assert any(
            item.type == "message" and item.text_content == _SENTINEL_TEXT
            for item in session.history.items
        )
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_llm_sink_covers_overridden_llm_node() -> None:
    seen_texts: list[str] = []

    class _OverridingAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test agent")

        async def llm_node(
            self,
            chat_ctx: ChatContext,
            tools: list[Tool],
            model_settings: ModelSettings,
        ):
            for item in chat_ctx.items:
                if item.type == "message" and item.text_content:
                    seen_texts.append(item.text_content)
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                yield chunk

    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input=_REDACTED_TEXT, content="ok", ttft=0.01, duration=0.02)
        ]
    )
    session = AgentSession(llm=fake_llm, redaction=RedactionOptions(redactor=RegexRedactor()))
    await session.start(_OverridingAgent())
    try:
        await session.generate_reply(user_input=_SENTINEL_TEXT)

        assert seen_texts
        assert all("4242" not in text for text in seen_texts)
        assert _REDACTED_TEXT in seen_texts
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_llm_sink_redacts_realtime_context_push() -> None:
    class _StubRealtimeSession:
        def __init__(self) -> None:
            self.seen_texts: list[str] = []

        async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
            for item in chat_ctx.items:
                if item.type == "message" and item.text_content:
                    self.seen_texts.append(item.text_content)

    session = AgentSession(llm=FakeLLM(), redaction=RedactionOptions(redactor=RegexRedactor()))
    agent = Agent(instructions="test agent")
    await session.start(agent)
    activity = session._activity
    assert activity is not None
    stub = _StubRealtimeSession()
    activity._rt_session = stub  # type: ignore[assignment]
    try:
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.add_message(role="user", content=_SENTINEL_TEXT)
        await activity.update_chat_ctx(chat_ctx)

        assert stub.seen_texts
        assert all("4242" not in text for text in stub.seen_texts)
        # the agent's stored context must keep the raw value
        assert any(
            item.type == "message" and item.text_content == _SENTINEL_TEXT
            for item in agent.chat_ctx.items
        )
    finally:
        activity._rt_session = None
        await session.aclose()


@pytest.mark.asyncio
async def test_no_redaction_configured_leaves_llm_input_raw() -> None:
    fake_llm = _CapturingFakeLLM(
        fake_responses=[
            FakeLLMResponse(input=_SENTINEL_TEXT, content="ok", ttft=0.01, duration=0.02)
        ]
    )
    session = AgentSession(llm=fake_llm)
    await session.start(Agent(instructions="test agent"))
    try:
        await session.generate_reply(user_input=_SENTINEL_TEXT)

        assert _SENTINEL_TEXT in fake_llm.seen_texts
    finally:
        await session.aclose()
