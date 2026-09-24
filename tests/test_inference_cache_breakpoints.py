import json
from typing import Any

import httpx
import openai
import pytest

from livekit.agents import inference
from livekit.agents.inference.llm import supports_prompt_cache_breakpoints
from livekit.agents.llm import CacheBreakpoint, ChatContext

pytestmark = pytest.mark.unit

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
GATEWAY = "http://gateway.test/v1"

_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def test_rule_accepts_gpt_5_6_on_gateway():
    assert supports_prompt_cache_breakpoints("openai/gpt-5.6-luna") is True


def test_rule_accepts_bare_gpt_5_6():
    assert supports_prompt_cache_breakpoints("gpt-5.6") is True


def test_rule_accepts_newer_minor():
    assert supports_prompt_cache_breakpoints("gpt-5.7") is True


def test_rule_accepts_newer_major():
    assert supports_prompt_cache_breakpoints("gpt-6") is True


def test_rule_accepts_newer_major_variant():
    assert supports_prompt_cache_breakpoints("gpt-7-mini") is True


def test_rule_accepts_unrecognized_openai_name():
    assert supports_prompt_cache_breakpoints("openai/some-new-model") is True


def test_rule_is_case_insensitive():
    assert supports_prompt_cache_breakpoints("OpenAI/GPT-5.6-Luna") is True


def test_rule_rejects_gpt_4_1_on_gateway():
    assert supports_prompt_cache_breakpoints("openai/gpt-4.1") is False


def test_rule_rejects_bare_gpt_4_1():
    assert supports_prompt_cache_breakpoints("gpt-4.1") is False


def test_rule_rejects_gpt_5():
    assert supports_prompt_cache_breakpoints("gpt-5") is False


def test_rule_rejects_gpt_5_5_pro():
    assert supports_prompt_cache_breakpoints("gpt-5.5-pro") is False


def test_rule_rejects_versioned_chat_latest():
    assert supports_prompt_cache_breakpoints("gpt-5.1-chat-latest") is False


def test_rule_rejects_bare_chat_latest():
    assert supports_prompt_cache_breakpoints("openai/chat-latest") is False


def test_rule_accepts_versioned_chat_latest_from_5_6():
    # a chat-latest snapshot is gated by its version like any other name; OpenAI documents
    # breakpoints as supported on gpt-5.6 and later, and no 5.6+ snapshot exists to measure
    assert supports_prompt_cache_breakpoints("gpt-5.6-chat-latest") is True


def test_rule_rejects_chatgpt_alias():
    assert supports_prompt_cache_breakpoints("chatgpt-4o-latest") is False


def test_rule_rejects_gpt_4o():
    assert supports_prompt_cache_breakpoints("gpt-4o-mini") is False


def test_rule_rejects_o_series():
    assert supports_prompt_cache_breakpoints("o3") is False


def test_rule_rejects_gpt_oss():
    assert supports_prompt_cache_breakpoints("openai/gpt-oss-120b") is False


def test_rule_rejects_google_model():
    assert supports_prompt_cache_breakpoints("google/gemini-3.5-flash") is False


def test_rule_rejects_deepseek_model():
    assert supports_prompt_cache_breakpoints("deepseek-ai/deepseek-v3.2") is False


def _llm(model: str, **kwargs: Any) -> inference.LLM:
    return inference.LLM(model, base_url=GATEWAY, api_key="key", api_secret="secret" * 6, **kwargs)


def _ctx() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="system", content=[STATIC, CacheBreakpoint(), DYNAMIC])
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


async def _resolved(llm: inference.LLM) -> bool:
    stream = llm.chat(chat_ctx=_ctx())
    try:
        return stream._prompt_cache_breakpoints
    finally:
        await stream.aclose()
        await llm.aclose()


async def test_auto_enables_for_gpt_5_6():
    assert await _resolved(_llm("openai/gpt-5.6-luna")) is True


async def test_auto_disables_for_gpt_4_1():
    assert await _resolved(_llm("openai/gpt-4.1")) is False


async def test_auto_disables_for_google_model():
    assert await _resolved(_llm("google/gemini-3.5-flash")) is False


async def test_explicit_false_overrides_auto():
    assert await _resolved(_llm("openai/gpt-5.6-luna", prompt_cache_breakpoints=False)) is False


async def test_explicit_true_overrides_auto():
    assert await _resolved(_llm("openai/gpt-4.1", prompt_cache_breakpoints=True)) is True


async def test_update_options_model_reevaluates_auto():
    llm = _llm("openai/gpt-5.6-luna")
    llm.update_options(model="openai/gpt-4.1")

    assert await _resolved(llm) is False


async def test_update_options_setting_applies_to_next_chat():
    llm = _llm("openai/gpt-5.6-luna")
    llm.update_options(prompt_cache_breakpoints=False)

    assert await _resolved(llm) is False


async def _wire_body(llm: inference.LLM) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE)

    llm._client = openai.AsyncClient(
        api_key="token",
        base_url=GATEWAY,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    stream = llm.chat(chat_ctx=_ctx())
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await llm.aclose()
    return captured


async def test_wire_carries_breakpoint_when_enabled():
    body = await _wire_body(_llm("openai/gpt-5.6-luna"))

    assert body["messages"][0]["content"] == [
        {"type": "text", "text": STATIC, "prompt_cache_breakpoint": {"mode": "explicit"}},
        {"type": "text", "text": f"\n{DYNAMIC}"},
    ]


async def test_wire_omits_breakpoint_for_gpt_4_1():
    body = await _wire_body(_llm("openai/gpt-4.1"))

    assert body["messages"][0]["content"] == f"{STATIC}\n{DYNAMIC}"
    assert "prompt_cache_breakpoint" not in json.dumps(body)


async def test_wire_omits_breakpoint_for_google_model():
    body = await _wire_body(_llm("google/gemini-3.5-flash"))

    assert body["messages"][0]["content"] == f"{STATIC}\n{DYNAMIC}"
    assert "prompt_cache_breakpoint" not in json.dumps(body)
