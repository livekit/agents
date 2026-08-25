from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import openai
import pytest

from livekit.agents import APIStatusError
from livekit.agents.inference.llm import LLMStream as InferenceLLMStream
from livekit.agents.llm import ChatContext, ImageContent
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import is_given
from livekit.plugins.sarvam.llm.client import (
    LLM as SarvamLLM,
    SARVAM_LLM_BASE_URL_V1,
    SARVAM_LLM_BASE_URL_V2,
    USER_AGENT,
    _filter_extra_body,
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


def _has_not_given(params: dict[str, Any]) -> bool:
    """Recursively check if NOT_GIVEN sentinel leaked into params."""
    NOT_GIVEN_REPR = repr(NOT_GIVEN)
    if NOT_GIVEN_REPR in repr(params):
        return True
    for v in params.values():
        if isinstance(v, dict):
            if _has_not_given(v):
                return True
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and _has_not_given(item):
                    return True
        elif repr(v) == NOT_GIVEN_REPR:
            return True
    return False


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


def test_constructor_rejects_old_30b_model() -> None:
    with pytest.raises(ValueError, match="Unsupported Sarvam model"):
        SarvamLLM(
            model="sarvam-30b",
            api_key="sk_test",
            client=_fake_async_client(),
        )


@pytest.mark.parametrize("model", ["gemma4", "sarvam-105b", "sarvam-105b-conversations", "glm5.2"])
def test_constructor_accepts_each_supported_model(model: str) -> None:
    llm = SarvamLLM(model=model, api_key="sk_test", client=_fake_async_client())
    assert llm.model == model


def test_default_model_is_sarvam_105b() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    assert llm.model == "sarvam-105b"


# ---------------------------------------------------------------------------
# Auth and headers
# ---------------------------------------------------------------------------


def test_auth_headers_injected() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    headers = llm._opts.extra_headers
    assert headers["api-subscription-key"] == "sk_test"
    assert headers["User-Agent"] == USER_AGENT


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


def test_base_url_default() -> None:
    # Don't pass a mock client so a real AsyncClient is created with the default base_url
    llm = SarvamLLM(api_key="sk_test")
    assert str(llm._client._base_url).rstrip("/") == SARVAM_LLM_BASE_URL_V2.rstrip("/")
    import asyncio

    asyncio.get_event_loop().run_until_complete(llm.aclose())


def test_explicit_base_url_overrides_default() -> None:
    custom = "https://custom.sarvam.ai/v2"
    llm = SarvamLLM(api_key="sk_test", base_url=custom)
    assert str(llm._client._base_url).rstrip("/") == custom.rstrip("/")
    import asyncio

    asyncio.get_event_loop().run_until_complete(llm.aclose())


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


def test_filter_extra_body_function() -> None:
    result = _filter_extra_body(
        {
            "max_tokens": 100,
            "wiki_grounding": True,
            "service_tier": "flex",
            "unknown": "drop",
            "n": 2,
        }
    )
    assert result == {"max_tokens": 100, "wiki_grounding": True, "n": 2}


# ---------------------------------------------------------------------------
# Capability gating — wiki_grounding and reasoning_effort
# ---------------------------------------------------------------------------


def test_wiki_grounding_in_extra_body() -> None:
    llm = SarvamLLM(
        api_key="sk_test",
        client=_fake_async_client(),
        wiki_grounding=True,
    )
    assert llm._opts.extra_body["wiki_grounding"] is True


def test_reasoning_effort_set_for_supported_model() -> None:
    llm = SarvamLLM(
        model="sarvam-105b",
        api_key="sk_test",
        client=_fake_async_client(),
        reasoning_effort="high",
    )
    assert llm._opts.reasoning_effort == "high"


def test_reasoning_effort_set_for_gemma4() -> None:
    llm = SarvamLLM(
        model="gemma4",
        api_key="sk_test",
        client=_fake_async_client(),
        reasoning_effort="low",
    )
    assert llm._opts.reasoning_effort == "low"


def test_reasoning_effort_set_for_glm52() -> None:
    llm = SarvamLLM(
        model="glm5.2",
        api_key="sk_test",
        client=_fake_async_client(),
        reasoning_effort="medium",
    )
    assert llm._opts.reasoning_effort == "medium"


def test_reasoning_effort_not_given_by_default() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    assert not is_given(llm._opts.reasoning_effort)


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
# Core fields forwarded
# ---------------------------------------------------------------------------


def test_core_fields_forwarded_to_opts() -> None:
    llm = SarvamLLM(
        api_key="sk_test",
        client=_fake_async_client(),
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["END"],
        n=2,
        seed=42,
        frequency_penalty=0.5,
        presence_penalty=0.3,
    )
    assert llm._opts.temperature == 0.7
    assert llm._opts.top_p == 0.9
    assert llm._opts.extra_body["max_tokens"] == 512
    assert llm._opts.extra_body["stop"] == ["END"]
    assert llm._opts.extra_body["n"] == 2
    assert llm._opts.extra_body["seed"] == 42
    assert llm._opts.extra_body["frequency_penalty"] == 0.5
    assert llm._opts.extra_body["presence_penalty"] == 0.3


# ---------------------------------------------------------------------------
# Provider and model properties
# ---------------------------------------------------------------------------


def test_provider_is_sarvam() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    assert llm.provider == "Sarvam"


def test_model_property_returns_model() -> None:
    llm = SarvamLLM(model="gemma4", api_key="sk_test", client=_fake_async_client())
    assert llm.model == "gemma4"


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
# Error propagation (from existing test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sarvam_entry_points_share_native_status_error_behavior(monkeypatch) -> None:
    async def _raise_native_api_status_error(self) -> None:  # pragma: no cover - monkeypatched
        raise APIStatusError(
            "native provider error",
            status_code=422,
            request_id="req_test",
            body={"error": "bad request"},
            retryable=False,
        )

    monkeypatch.setattr(InferenceLLMStream, "_run", _raise_native_api_status_error)

    sarvam_stream = SarvamLLM(api_key="sk_test", client=_fake_async_client()).chat(
        chat_ctx=_chat_ctx()
    )
    streams = (sarvam_stream,)

    try:
        for stream in streams:
            with pytest.raises(APIStatusError) as excinfo:
                await stream._run()

            assert excinfo.value.message == "native provider error"
            assert excinfo.value.status_code == 422
            assert excinfo.value.body == {"error": "bad request"}
    finally:
        for stream in streams:
            await stream.aclose()


# ---------------------------------------------------------------------------
# No NOT_GIVEN sentinel leaks
# ---------------------------------------------------------------------------


def test_no_not_given_in_extra_body() -> None:
    llm = SarvamLLM(
        api_key="sk_test",
        client=_fake_async_client(),
        wiki_grounding=True,
        max_tokens=100,
    )
    if is_given(llm._opts.extra_body):
        assert not _has_not_given(llm._opts.extra_body)


def test_no_not_given_in_extra_headers() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    if is_given(llm._opts.extra_headers):
        assert not _has_not_given(llm._opts.extra_headers)


# ---------------------------------------------------------------------------
# Optional fields omitted when unset
# ---------------------------------------------------------------------------


def test_optional_fields_omitted_when_unset() -> None:
    llm = SarvamLLM(api_key="sk_test", client=_fake_async_client())
    # wiki_grounding should not be in extra_body when not set
    if is_given(llm._opts.extra_body):
        assert "wiki_grounding" not in llm._opts.extra_body
    # reasoning_effort should not be given when not set
    assert not is_given(llm._opts.reasoning_effort)


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
