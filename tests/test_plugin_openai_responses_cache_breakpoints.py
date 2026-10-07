import json
from typing import Any

import httpx
import openai
import pytest

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.llm import ChatContext
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice.generation import mark_instructions_cache_boundary, update_instructions
from livekit.plugins.openai.responses import LLM

pytestmark = pytest.mark.unit

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
BREAKPOINT = {"mode": "explicit"}


def test_auto_enables_for_gpt_5_6_on_openai_websocket():
    assert LLM(model="gpt-5.6-luna", api_key="k")._resolve_prompt_cache_breakpoints() is True


def test_auto_enables_for_gpt_6_on_openai_http():
    llm = LLM(model="gpt-6-sol", api_key="k", use_websocket=False)

    assert llm._resolve_prompt_cache_breakpoints() is True


def test_auto_disables_for_gpt_4_1():
    assert LLM(model="gpt-4.1", api_key="k")._resolve_prompt_cache_breakpoints() is False


def test_auto_disables_for_another_host():
    llm = LLM(model="gpt-5.6", api_key="k", base_url="wss://gw.example/v1/responses")

    assert llm._resolve_prompt_cache_breakpoints() is False


def test_explicit_false_overrides_auto():
    llm = LLM(model="gpt-5.6", api_key="k", prompt_cache_breakpoints=False)

    assert llm._resolve_prompt_cache_breakpoints() is False


def test_explicit_true_overrides_host_rule():
    llm = LLM(
        model="gpt-4.1",
        api_key="k",
        base_url="wss://gw.example/v1/responses",
        prompt_cache_breakpoints=True,
    )

    assert llm._resolve_prompt_cache_breakpoints() is True


def _ctx() -> ChatContext:
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(STATIC, dynamic=DYNAMIC), add_if_missing=True
    )
    mark_instructions_cache_boundary(ctx)
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


async def _request_body(model: str, **kwargs: Any) -> dict[str, Any]:
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
    stream = llm.chat(chat_ctx=_ctx(), conn_options=APIConnectOptions(max_retry=0))
    try:
        with pytest.raises(APIStatusError):
            async for _ in stream:
                pass
    finally:
        await stream.aclose()
        await llm.aclose()
        await client.close()
    return captured


async def test_wire_tags_the_instructions_for_gpt_5_6():
    body = await _request_body("gpt-5.6-luna")

    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": STATIC, "prompt_cache_breakpoint": BREAKPOINT}
    ]
    assert body["input"][1] == {"role": "system", "content": DYNAMIC}


async def test_wire_folds_dynamic_and_omits_breakpoints_for_gpt_4_1():
    body = await _request_body("gpt-4.1")

    assert body["input"][0] == {"role": "system", "content": f"{STATIC}\n{DYNAMIC}"}
    assert "prompt_cache_breakpoint" not in json.dumps(body)


async def test_wire_forwards_prompt_cache_options():
    body = await _request_body(
        "gpt-5.6-luna", prompt_cache_options={"mode": "implicit", "ttl": "30m"}
    )

    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}


async def test_wire_omits_prompt_cache_options_by_default():
    body = await _request_body("gpt-5.6-luna")

    assert "prompt_cache_options" not in body
