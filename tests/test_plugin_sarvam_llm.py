from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import openai
import pytest

from livekit.agents import APIStatusError
from livekit.agents.inference.llm import LLMStream as InferenceLLMStream
from livekit.agents.llm import (
    ChatContext,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    function_tool,
)
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import is_given
from livekit.plugins.sarvam.llm.client import (
    LLM as SarvamLLM,
    SARVAM_LLM_BASE_URL_V1,
    SARVAM_LLM_BASE_URL_V2,
    USER_AGENT,
)

pytestmark = pytest.mark.unit

_CONVERSATIONS = "sarvam-105b-conversations"


def _llm(**kwargs: Any) -> SarvamLLM:
    return SarvamLLM(api_key="sk_test", client=cast(openai.AsyncClient, MagicMock()), **kwargs)


def _chat_ctx(text: str = "hello") -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content=text)
    return ctx


def _image_ctx() -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(
        role="user",
        content=["describe this image", ImageContent(image="data:image/png;base64,iVBORw0KGgo=")],
    )
    return ctx


@function_tool
def get_weather(city: str) -> str:
    return "sunny"


@pytest.mark.parametrize("model", ["sarvam-unknown", "sarvam-30b"])
def test_unsupported_model_is_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Sarvam model"):
        _llm(model=model)


@pytest.mark.parametrize("model", ["gemma4", "sarvam-105b", _CONVERSATIONS, "glm5.2"])
def test_supported_models_are_accepted(model: str) -> None:
    assert _llm(model=model).model == model
    assert _llm().model == "sarvam-105b"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SARVAM_API_KEY is required"):
        SarvamLLM(client=cast(openai.AsyncClient, MagicMock()))


def test_sarvam_headers_override_caller_headers() -> None:
    headers = _llm(
        extra_headers={"api-subscription-key": "x", "User-Agent": "x", "X-Custom": "kept"}
    )._opts.extra_headers

    assert headers["api-subscription-key"] == "sk_test"
    assert headers["User-Agent"] == USER_AGENT
    assert headers["X-Custom"] == "kept"


@pytest.mark.parametrize(
    ("kwargs", "base_url"),
    [
        ({}, SARVAM_LLM_BASE_URL_V2),
        ({"model": "gemma4"}, SARVAM_LLM_BASE_URL_V2),
        ({"model": "glm5.2"}, SARVAM_LLM_BASE_URL_V2),
        ({"model": _CONVERSATIONS}, SARVAM_LLM_BASE_URL_V1),
        ({"base_url": "https://custom.sarvam.ai/v2"}, "https://custom.sarvam.ai/v2"),
        (
            {"model": _CONVERSATIONS, "base_url": "https://custom.sarvam.ai/v1"},
            "https://custom.sarvam.ai/v1",
        ),
    ],
)
async def test_base_url_routing(kwargs: dict[str, Any], base_url: str) -> None:
    llm = SarvamLLM(api_key="sk_test", **kwargs)
    assert str(llm._client._base_url).rstrip("/") == base_url.rstrip("/")
    await llm.aclose()


def test_request_fields_are_forwarded_and_unsupported_ones_dropped() -> None:
    llm = _llm(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["END"],
        n=2,
        seed=42,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        wiki_grounding=True,
        extra_body={"service_tier": "flex", "unknown_field": "drop-me"},
    )

    assert (llm._opts.temperature, llm._opts.top_p) == (0.7, 0.9)
    assert llm._opts.extra_body == {
        "max_tokens": 512,
        "stop": ["END"],
        "n": 2,
        "seed": 42,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "wiki_grounding": True,
    }
    assert repr(NOT_GIVEN) not in repr(llm._opts.extra_body) + repr(llm._opts.extra_headers)


def test_optional_fields_omitted_when_unset() -> None:
    llm = _llm()
    assert not is_given(llm._opts.extra_body) or "wiki_grounding" not in llm._opts.extra_body
    assert not is_given(llm._opts.reasoning_effort)


@pytest.mark.parametrize("model", ["sarvam-105b", "gemma4", "glm5.2"])
def test_reasoning_effort_is_kept_on_reasoning_models(model: str) -> None:
    assert _llm(model=model, reasoning_effort="high")._opts.reasoning_effort == "high"


async def test_conversations_drops_unsupported_options() -> None:
    llm = _llm(model=_CONVERSATIONS, reasoning_effort="high", wiki_grounding=True)
    assert not is_given(llm._opts.reasoning_effort)
    assert not is_given(llm._opts.extra_body) or "wiki_grounding" not in llm._opts.extra_body

    stream = llm.chat(chat_ctx=_chat_ctx())
    for field in ("reasoning_effort", "stream_options", "max_completion_tokens", "service_tier"):
        assert field not in stream._extra_kwargs
    await stream.aclose()


@pytest.mark.parametrize("model", ["sarvam-105b", "glm5.2", _CONVERSATIONS])
def test_image_rejected_on_non_vision_model(model: str) -> None:
    with pytest.raises(ValueError, match="Image input is not supported"):
        _llm(model=model).chat(chat_ctx=_image_ctx())


async def test_vision_check_accepts_images_and_non_message_items() -> None:
    ctx = _image_ctx()
    ctx.insert(FunctionCallOutput(call_id="call", output="result", is_error=False))
    await _llm(model="gemma4").chat(chat_ctx=ctx).aclose()


def test_tool_choice_without_tools_raises() -> None:
    with pytest.raises(ValueError, match="tool_choice requires a non-empty tools array"):
        _llm().chat(chat_ctx=_chat_ctx(), tool_choice="required")


@pytest.mark.parametrize(
    ("model", "tools", "tool_choice"),
    [
        ("sarvam-105b", [], "none"),
        ("sarvam-105b", [], "auto"),
        ("sarvam-105b", [get_weather], "auto"),
        (_CONVERSATIONS, [get_weather], "auto"),
    ],
)
async def test_tool_choice_accepted(model: str, tools: list[Any], tool_choice: Any) -> None:
    await (
        _llm(model=model).chat(chat_ctx=_chat_ctx(), tools=tools, tool_choice=tool_choice).aclose()
    )


@pytest.mark.parametrize("with_tools", [False, True])
async def test_tool_history_kept_only_when_turn_has_tools(with_tools: bool) -> None:
    """Sarvam rejects tool messages sent without a `tools` array."""
    ctx = _chat_ctx("weather?")
    ctx.insert(
        [
            FunctionCall(call_id="call_1", name="get_weather", arguments='{"city": "Bengaluru"}'),
            FunctionCallOutput(
                call_id="call_1", name="get_weather", output="sunny", is_error=False
            ),
        ]
    )
    tools = [get_weather] if with_tools else []
    stream = _llm().chat(chat_ctx=ctx, tools=tools, tool_choice="none" if with_tools else NOT_GIVEN)

    messages, _ = stream._chat_ctx.to_provider_format(format="openai")
    has_tool_messages = any(m.get("role") == "tool" or "tool_calls" in m for m in messages)
    assert has_tool_messages is with_tools
    await stream.aclose()


async def test_native_status_error_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(self: object) -> None:
        raise APIStatusError("native", status_code=422, body={"error": "bad"}, retryable=False)

    monkeypatch.setattr(InferenceLLMStream, "_run", run)
    stream = _llm().chat(chat_ctx=_chat_ctx())
    try:
        with pytest.raises(APIStatusError) as excinfo:
            await stream._run()
    finally:
        await stream.aclose()

    assert (excinfo.value.status_code, excinfo.value.body) == (422, {"error": "bad"})
