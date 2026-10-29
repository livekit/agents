from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import openai
import pytest

from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput, ImageContent
from livekit.agents.utils import is_given
from livekit.plugins.sarvam.llm.client import (
    LLM as SarvamLLM,
    SARVAM_LLM_BASE_URL_V1,
    SARVAM_LLM_BASE_URL_V2,
    USER_AGENT,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_async_client() -> openai.AsyncClient:
    return cast(openai.AsyncClient, MagicMock())


def _chat_ctx(text: str = "hello") -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content=text)
    return ctx


def _chat_ctx_with_image() -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(
        role="user",
        content=[
            "describe this image",
            ImageContent(image="data:image/png;base64,iVBORw0KGgo="),
        ],
    )
    return ctx


# ---------------------------------------------------------------------------
# Constructor — model validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported Sarvam model"):
        SarvamLLM(
            model="sarvam-unknown",
            api_key="sk_test",
            client=_fake_async_client(),
        )


# ---------------------------------------------------------------------------
# Auth and headers
# ---------------------------------------------------------------------------


def test_custom_headers_merged() -> None:
    llm = SarvamLLM(
        api_key="sk_test",
        client=_fake_async_client(),
        extra_headers={
            "api-subscription-key": "override_attempt",
            "User-Agent": "override_attempt",
            "X-Custom": "kept",
        },
    )
    headers = llm._opts.extra_headers
    # Sarvam headers override caller values on conflict
    assert headers["api-subscription-key"] == "sk_test"
    assert headers["User-Agent"] == USER_AGENT
    # Caller-provided non-conflicting headers are preserved
    assert headers["X-Custom"] == "kept"


# ---------------------------------------------------------------------------
# Extra body filtering
# ---------------------------------------------------------------------------


def test_filters_unsupported_extra_body_fields() -> None:
    llm = SarvamLLM(
        api_key="sk_test",
        client=_fake_async_client(),
        extra_body={
            "max_tokens": 64,
            "wiki_grounding": True,
            "service_tier": "flex",
            "unknown_field": "drop-me",
        },
    )
    assert llm._opts.extra_body == {
        "max_tokens": 64,
        "wiki_grounding": True,
    }


# ---------------------------------------------------------------------------
# Image rejection on non-vision models
# ---------------------------------------------------------------------------


def test_image_rejected_on_non_vision_model() -> None:
    llm = SarvamLLM(
        model="sarvam-105b",
        api_key="sk_test",
        client=_fake_async_client(),
    )
    with pytest.raises(ValueError, match="Image input is not supported"):
        llm.chat(chat_ctx=_chat_ctx_with_image())


def test_image_rejected_on_glm52() -> None:
    llm = SarvamLLM(
        model="glm5.2",
        api_key="sk_test",
        client=_fake_async_client(),
    )
    with pytest.raises(ValueError, match="Image input is not supported"):
        llm.chat(chat_ctx=_chat_ctx_with_image())


@pytest.mark.asyncio
async def test_image_accepted_on_vision_model() -> None:
    llm = SarvamLLM(
        model="gemma4",
        api_key="sk_test",
        client=_fake_async_client(),
    )
    # Should not raise
    stream = llm.chat(chat_ctx=_chat_ctx_with_image())
    assert stream is not None
    await stream.aclose()


@pytest.mark.asyncio
async def test_vision_validation_skips_non_dict_messages() -> None:
    """Non-message items (function calls, tool outputs) should not cause errors."""
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="hello")
    # Add a function call output (non-message item)
    from livekit.agents.llm import FunctionCallOutput

    ctx.insert(FunctionCallOutput(call_id="test_call", output="result", is_error=False))
    llm = SarvamLLM(
        model="sarvam-105b",
        api_key="sk_test",
        client=_fake_async_client(),
    )
    # Should not raise
    stream = llm.chat(chat_ctx=ctx)
    await stream.aclose()


# ---------------------------------------------------------------------------
# tool_choice validation
# ---------------------------------------------------------------------------


def test_tool_choice_without_tools_raises() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    with pytest.raises(ValueError, match="tool_choice requires a non-empty tools array"):
        llm.chat(chat_ctx=_chat_ctx(), tool_choice="required")


@pytest.mark.asyncio
async def test_tool_choice_none_without_tools_ok() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    # Should not raise
    stream = llm.chat(chat_ctx=_chat_ctx(), tool_choice="none")
    assert stream is not None
    await stream.aclose()


@pytest.mark.asyncio
async def test_tool_choice_auto_without_tools_ok() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    stream = llm.chat(chat_ctx=_chat_ctx(), tool_choice="auto")
    assert stream is not None
    await stream.aclose()


@pytest.mark.asyncio
async def test_tool_choice_with_tools_allowed() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    from livekit.agents.llm import function_tool

    @function_tool
    def get_weather(city: str) -> str:
        return "sunny"

    stream = llm.chat(
        chat_ctx=_chat_ctx("weather?"),
        tools=[get_weather],
        tool_choice="auto",
    )
    assert stream is not None
    await stream.aclose()


# ---------------------------------------------------------------------------
# Orphan tool history
# ---------------------------------------------------------------------------


def _chat_ctx_with_tool_call() -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="weather?")
    ctx.insert(
        [
            FunctionCall(call_id="call_1", name="get_weather", arguments='{"city": "Bengaluru"}'),
            FunctionCallOutput(
                call_id="call_1", name="get_weather", output="sunny", is_error=False
            ),
        ]
    )
    return ctx


async def _tool_messages(stream: Any) -> list[dict[str, Any]]:
    messages, _ = await stream._chat_ctx.to_provider_format(format="openai")
    return [m for m in messages if m.get("role") == "tool" or "tool_calls" in m]


@pytest.mark.asyncio
async def test_tool_history_stripped_when_turn_has_no_tools() -> None:
    """Sarvam 400s on tool messages with no `tools` array, poisoning the session."""
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    stream = llm.chat(chat_ctx=_chat_ctx_with_tool_call())
    assert await _tool_messages(stream) == []
    await stream.aclose()


@pytest.mark.asyncio
async def test_tool_history_kept_when_turn_has_tools() -> None:
    """A tool reply still sees its own result — only unmatchable history is dropped."""
    from livekit.agents.llm import function_tool

    @function_tool
    def get_weather(city: str) -> str:
        return "sunny"

    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    stream = llm.chat(
        chat_ctx=_chat_ctx_with_tool_call(),
        tools=[get_weather],
        tool_choice="none",
    )
    assert await _tool_messages(stream)
    await stream.aclose()


# ---------------------------------------------------------------------------
# Unsupported OpenAI fields stripped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_fields_stripped_from_stream() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    stream = llm.chat(chat_ctx=_chat_ctx())
    for field in ("stream_options", "max_completion_tokens", "service_tier"):
        assert field not in stream._extra_kwargs
    await stream.aclose()


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


def test_missing_api_key_raises() -> None:
    # Ensure no env var leaks in
    import os

    old = os.environ.pop("SARVAM_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="SARVAM_API_KEY is required"):
            SarvamLLM(client=_fake_async_client())
    finally:
        if old:
            os.environ["SARVAM_API_KEY"] = old


# ---------------------------------------------------------------------------
# sarvam-105b-conversations — endpoint routing and capability gating
# ---------------------------------------------------------------------------


def test_conversations_resolves_to_v1_endpoint() -> None:
    llm = SarvamLLM(model="sarvam-105b-conversations", api_key="sk_test")
    assert str(llm._client._base_url).rstrip("/") == SARVAM_LLM_BASE_URL_V1.rstrip("/")
    import asyncio

    asyncio.get_event_loop().run_until_complete(llm.aclose())


def test_v2_models_resolve_to_v2_endpoint() -> None:
    """Non-conversations models resolve to /v2."""
    for model in ["sarvam-105b", "gemma4", "glm5.2"]:
        llm = SarvamLLM(model=model, api_key="sk_test")
        assert str(llm._client._base_url).rstrip("/") == SARVAM_LLM_BASE_URL_V2.rstrip("/")
        import asyncio

        asyncio.get_event_loop().run_until_complete(llm.aclose())


def test_conversations_strips_reasoning_effort() -> None:
    """reasoning_effort is not supported on conversations — should not be set in opts."""
    llm = SarvamLLM(
        model="sarvam-105b-conversations",
        api_key="sk_test",
        client=_fake_async_client(),
        reasoning_effort="high",
    )
    assert not is_given(llm._opts.reasoning_effort)


@pytest.mark.asyncio
async def test_conversations_strips_reasoning_effort_from_stream() -> None:
    """reasoning_effort should be absent from built stream params for conversations."""
    llm = SarvamLLM(
        model="sarvam-105b-conversations",
        api_key="sk_test",
        client=_fake_async_client(),
    )
    stream = llm.chat(chat_ctx=_chat_ctx())
    assert "reasoning_effort" not in stream._extra_kwargs
    await stream.aclose()


def test_conversations_strips_wiki_grounding() -> None:
    """wiki_grounding is not supported on conversations — should not appear in extra_body."""
    llm = SarvamLLM(
        model="sarvam-105b-conversations",
        api_key="sk_test",
        client=_fake_async_client(),
        wiki_grounding=True,
    )
    if is_given(llm._opts.extra_body):
        assert "wiki_grounding" not in llm._opts.extra_body


def test_conversations_rejects_image_input() -> None:
    """Conversations model does not support vision — image raises ValueError."""
    llm = SarvamLLM(
        model="sarvam-105b-conversations",
        api_key="sk_test",
        client=_fake_async_client(),
    )
    with pytest.raises(ValueError, match="Image input is not supported"):
        llm.chat(chat_ctx=_chat_ctx_with_image())


@pytest.mark.asyncio
async def test_conversations_supports_tool_calling() -> None:
    """Conversations model supports tool calling — tools and tool_choice pass through."""
    llm = SarvamLLM(
        model="sarvam-105b-conversations",
        api_key="sk_test",
        client=_fake_async_client(),
    )
    from livekit.agents.llm import function_tool

    @function_tool
    def get_weather(city: str) -> str:
        return "sunny"

    stream = llm.chat(
        chat_ctx=_chat_ctx("weather?"),
        tools=[get_weather],
        tool_choice="auto",
    )
    assert stream is not None
    await stream.aclose()


def test_explicit_base_url_overrides_conversations_v1() -> None:
    """Explicit base_url wins even for conversations model."""
    custom = "https://custom.sarvam.ai/v1"
    llm = SarvamLLM(
        model="sarvam-105b-conversations",
        api_key="sk_test",
        base_url=custom,
    )
    assert str(llm._client._base_url).rstrip("/") == custom.rstrip("/")
    import asyncio

    asyncio.get_event_loop().run_until_complete(llm.aclose())
