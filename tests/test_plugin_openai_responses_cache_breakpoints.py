import json
from typing import Any

import httpx
import openai
import pytest

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.llm import CacheBreakpoint, ChatContext
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice.generation import mark_instructions_cache_boundary, update_instructions
from livekit.plugins.openai.responses import LLM

pytestmark = pytest.mark.unit

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
DOC = "Insurance card on file: Acme Health, member 1234."
BREAKPOINT = {"mode": "explicit"}


async def _resolved(**kwargs: Any) -> bool:
    llm = LLM(api_key="k", **kwargs)
    try:
        return llm._resolve_prompt_cache_breakpoints()
    finally:
        await llm.aclose()


async def test_auto_enables_for_gpt_5_6_on_openai_websocket():
    assert await _resolved(model="gpt-5.6-luna") is True


async def test_auto_enables_for_gpt_6_on_openai_http():
    assert await _resolved(model="gpt-6-sol", use_websocket=False) is True


async def test_auto_enables_with_an_explicit_port_on_websocket():
    assert await _resolved(model="gpt-5.6", base_url="wss://api.openai.com:443/v1/responses")


async def test_auto_enables_with_an_explicit_port_on_http():
    llm_kwargs = {"use_websocket": False, "base_url": "https://api.openai.com:443/v1"}

    assert await _resolved(model="gpt-5.6", **llm_kwargs) is True


async def test_auto_enables_with_a_trailing_dot_host():
    assert await _resolved(model="gpt-5.6", base_url="wss://api.openai.com./v1/responses")


async def test_auto_disables_for_gpt_4_1():
    assert await _resolved(model="gpt-4.1") is False


async def test_auto_disables_for_another_host():
    assert await _resolved(model="gpt-5.6", base_url="wss://gw.example/v1/responses") is False


async def test_explicit_false_overrides_auto():
    assert await _resolved(model="gpt-5.6", prompt_cache_breakpoints=False) is False


async def test_explicit_true_overrides_host_rule():
    kwargs = {"base_url": "wss://gw.example/v1/responses", "prompt_cache_breakpoints": True}

    assert await _resolved(model="gpt-4.1", **kwargs) is True


def _ctx() -> ChatContext:
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(STATIC, dynamic=DYNAMIC), add_if_missing=True
    )
    mark_instructions_cache_boundary(ctx)
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


async def _request_body(
    model: str, ctx: ChatContext | None = None, **kwargs: Any
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(500, json={"error": {"message": "stop here"}})

    client = openai.AsyncClient(
        api_key="k",
        base_url="https://api.openai.com/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm = LLM(model=model, client=client, use_websocket=False, **kwargs)
    stream = llm.chat(chat_ctx=ctx or _ctx(), conn_options=APIConnectOptions(max_retry=0))
    try:
        with pytest.raises(APIStatusError):
            async for _ in stream:
                pass
    finally:
        await stream.aclose()
        await llm.aclose()
        await client.close()
    return captured


def _tagged_parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        part
        for item in body["input"]
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if "prompt_cache_breakpoint" in part
    ]


async def test_wire_tags_the_instructions_for_gpt_5_6():
    body = await _request_body("gpt-5.6-luna")

    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": STATIC, "prompt_cache_breakpoint": BREAKPOINT}
    ]
    assert body["input"][1] == {"role": "system", "content": DYNAMIC}


async def test_wire_folds_dynamic_and_omits_breakpoints_for_gpt_4_1():
    body = await _request_body("gpt-4.1")

    assert body["input"][0] == {"role": "system", "content": f"{STATIC}\n{DYNAMIC}"}
    assert _tagged_parts(body) == []


async def test_wire_explicit_true_tags_gpt_4_1():
    body = await _request_body("gpt-4.1", prompt_cache_breakpoints=True)

    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": STATIC, "prompt_cache_breakpoint": BREAKPOINT}
    ]


async def test_wire_tags_a_caller_placed_breakpoint_in_a_user_message():
    ctx = _ctx()
    ctx.add_message(role="user", content=[DOC, CacheBreakpoint(), "Is my plan accepted?"])

    body = await _request_body("gpt-5.6-luna", ctx=ctx)

    assert body["input"][-1]["content"] == [
        {"type": "input_text", "text": DOC, "prompt_cache_breakpoint": BREAKPOINT},
        {"type": "input_text", "text": "\nIs my plan accepted?"},
    ]


async def test_wire_forwards_prompt_cache_options():
    body = await _request_body(
        "gpt-5.6-luna", prompt_cache_options={"mode": "implicit", "ttl": "30m"}
    )

    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}


async def test_wire_forwards_prompt_cache_options_even_with_breakpoints_off():
    # prompt_cache_options is the caller's own setting; it is not gated like breakpoints
    body = await _request_body(
        "gpt-5.6-luna",
        prompt_cache_breakpoints=False,
        prompt_cache_options={"mode": "implicit", "ttl": "30m"},
    )

    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert _tagged_parts(body) == []


async def test_wire_omits_prompt_cache_options_by_default():
    body = await _request_body("gpt-5.6-luna")

    assert "prompt_cache_options" not in body
