import json
from typing import Any

import httpx
import openai
import pytest

from livekit.agents.llm import CacheBreakpoint, ChatContext
from livekit.plugins.openai import LLM

pytestmark = pytest.mark.unit

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
AZURE = {"azure_endpoint": "https://res.openai.azure.com", "api_version": "2025-04-01-preview"}

_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _ctx() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="system", content=[STATIC, CacheBreakpoint(), DYNAMIC])
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


async def _resolved(llm: LLM) -> bool:
    stream = llm.chat(chat_ctx=_ctx())
    try:
        return stream._prompt_cache_breakpoints
    finally:
        await stream.aclose()
        await llm.aclose()


async def test_auto_enables_for_gpt_5_6_on_openai():
    assert await _resolved(LLM(model="gpt-5.6", api_key="k")) is True


async def test_auto_disables_for_gpt_4_1_on_openai():
    assert await _resolved(LLM(model="gpt-4.1", api_key="k")) is False


async def test_auto_disables_for_openai_compatible_host():
    llm = LLM(model="gpt-5.6", api_key="k", base_url="https://openrouter.ai/api/v1")

    assert await _resolved(llm) is False


async def test_auto_disables_for_azure():
    assert await _resolved(LLM.with_azure(model="gpt-5.6", api_key="k", **AZURE)) is False


async def test_azure_opts_in_explicitly():
    llm = LLM.with_azure(model="gpt-5.6", api_key="k", prompt_cache_breakpoints=True, **AZURE)

    assert await _resolved(llm) is True


async def test_explicit_false_overrides_auto():
    assert (
        await _resolved(LLM(model="gpt-5.6", api_key="k", prompt_cache_breakpoints=False)) is False
    )


async def test_explicit_true_overrides_host_rule():
    llm = LLM(
        model="gpt-4.1",
        api_key="k",
        base_url="https://gw.example/v1",
        prompt_cache_breakpoints=True,
    )

    assert await _resolved(llm) is True


async def test_prompt_cache_options_reach_the_request():
    llm = LLM(model="gpt-5.6", api_key="k", prompt_cache_options={"ttl": "30m"})
    stream = llm.chat(chat_ctx=_ctx())
    try:
        assert stream._extra_kwargs["prompt_cache_options"] == {"ttl": "30m"}
    finally:
        await stream.aclose()
        await llm.aclose()


async def _wire_body(model: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE)

    client = openai.AsyncClient(
        api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    llm = LLM(model=model, client=client)
    stream = llm.chat(chat_ctx=_ctx())
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await llm.aclose()
        await client.close()
    return captured


async def test_wire_carries_breakpoint_for_gpt_5_6():
    body = await _wire_body("gpt-5.6")

    assert body["messages"][0]["content"] == [
        {"type": "text", "text": STATIC, "prompt_cache_breakpoint": {"mode": "explicit"}},
        {"type": "text", "text": f"\n{DYNAMIC}"},
    ]


async def test_wire_omits_breakpoint_for_gpt_4_1():
    body = await _wire_body("gpt-4.1")

    assert body["messages"][0]["content"] == f"{STATIC}\n{DYNAMIC}"
    assert "prompt_cache_breakpoint" not in json.dumps(body)
