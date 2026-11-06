"""The ``reasoning`` option of the chat-completions inference LLM.

It takes the same setting as ``inference.responses.LLM`` and sends its effort as the
chat-completions ``reasoning_effort`` field. Unset, nothing is sent: this LLM has no
default effort.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx2 as httpx
import openai
import pytest
from openai.types import Reasoning

from livekit.agents import inference
from livekit.agents.llm import ChatContext

pytestmark = pytest.mark.unit

_GATEWAY = "http://gateway.test/v1"
_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _llm(model: str = "openai/gpt-5.4", **kwargs: Any) -> inference.LLM:
    return inference.LLM(model, base_url=_GATEWAY, api_key="key", api_secret="secret" * 6, **kwargs)


def _ctx() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="user", content="hi")
    return ctx


async def _wire_body(llm: inference.LLM, **chat_kwargs: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE)

    await llm._client.close()
    llm._client = openai.AsyncClient(
        api_key="token",
        base_url=_GATEWAY,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    stream = llm.chat(chat_ctx=_ctx(), **chat_kwargs)
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await llm.aclose()
    return captured


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


async def test_no_reasoning_sends_no_reasoning_effort() -> None:
    body = await _wire_body(_llm())

    assert "reasoning_effort" not in body


async def test_no_reasoning_sends_no_default_for_reasoning_model() -> None:
    body = await _wire_body(_llm("openai/gpt-6-luna"))

    assert "reasoning_effort" not in body


async def test_reasoning_string_is_sent_as_reasoning_effort() -> None:
    body = await _wire_body(_llm(reasoning="low"))

    assert body["reasoning_effort"] == "low"


async def test_reasoning_never_sends_a_reasoning_field() -> None:
    body = await _wire_body(_llm(reasoning="low"))

    assert "reasoning" not in body


async def test_reasoning_dict_is_sent_as_reasoning_effort() -> None:
    body = await _wire_body(_llm(reasoning={"effort": "high"}))

    assert body["reasoning_effort"] == "high"


async def test_reasoning_object_is_sent_as_reasoning_effort() -> None:
    body = await _wire_body(_llm(reasoning=Reasoning(effort="medium")))

    assert body["reasoning_effort"] == "medium"


async def test_reasoning_dict_without_effort_sends_no_reasoning_effort() -> None:
    body = await _wire_body(_llm(reasoning={}))

    assert "reasoning_effort" not in body


def test_reasoning_with_summary_raises() -> None:
    with pytest.raises(ValueError, match="summary"):
        _llm(reasoning={"effort": "low", "summary": "auto"})


def test_reasoning_with_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        _llm(reasoning=Reasoning(effort="low", mode="pro"))


async def test_reasoning_overrides_extra_kwargs_reasoning_effort() -> None:
    body = await _wire_body(_llm(reasoning="high", extra_kwargs={"reasoning_effort": "low"}))

    assert body["reasoning_effort"] == "high"


async def test_extra_kwargs_reasoning_effort_still_sent_without_reasoning() -> None:
    body = await _wire_body(_llm(extra_kwargs={"reasoning_effort": "low"}))

    assert body["reasoning_effort"] == "low"


async def test_per_call_extra_kwargs_reasoning_effort_overrides_reasoning() -> None:
    body = await _wire_body(_llm(reasoning="low"), extra_kwargs={"reasoning_effort": "high"})

    assert body["reasoning_effort"] == "high"


def test_extra_kwargs_reasoning_raises() -> None:
    with pytest.raises(ValueError, match="reasoning="):
        _llm(extra_kwargs={"reasoning": {"effort": "low"}})


async def test_update_options_extra_kwargs_reasoning_raises() -> None:
    llm = _llm()

    with pytest.raises(ValueError, match="reasoning="):
        llm.update_options(extra_kwargs={"reasoning": {"effort": "low"}})
    await llm.aclose()


async def test_update_options_extra_kwargs_reasoning_changes_nothing() -> None:
    llm = _llm(extra_kwargs={"max_completion_tokens": 64})

    with pytest.raises(ValueError):
        llm.update_options(extra_kwargs={"reasoning": {"effort": "low"}})
    body = await _wire_body(llm)

    assert body["max_completion_tokens"] == 64


async def test_per_call_extra_kwargs_reasoning_raises() -> None:
    llm = _llm()

    with pytest.raises(ValueError, match="reasoning="):
        llm.chat(chat_ctx=_ctx(), extra_kwargs={"reasoning": {"effort": "low"}})
    await llm.aclose()


async def test_reasoning_conflicting_with_extra_kwargs_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    llm = _llm(reasoning="high", extra_kwargs={"reasoning_effort": "low"})
    await llm.aclose()

    assert any("reasoning_effort" in m for m in _warnings(caplog))


async def test_reasoning_matching_extra_kwargs_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    llm = _llm(reasoning="low", extra_kwargs={"reasoning_effort": "low"})
    await llm.aclose()

    assert _warnings(caplog) == []


async def test_update_options_reasoning_changes_effort() -> None:
    llm = _llm(reasoning="low")

    llm.update_options(reasoning="high")
    body = await _wire_body(llm)

    assert body["reasoning_effort"] == "high"


async def test_update_options_reasoning_keeps_extra_kwargs() -> None:
    llm = _llm(extra_kwargs={"max_completion_tokens": 64})

    llm.update_options(reasoning="high")
    body = await _wire_body(llm)

    assert body["max_completion_tokens"] == 64


async def test_update_options_reasoning_none_clears_effort() -> None:
    llm = _llm(reasoning="low")

    llm.update_options(reasoning=None)
    body = await _wire_body(llm)

    assert "reasoning_effort" not in body


async def test_update_options_without_reasoning_keeps_effort() -> None:
    llm = _llm(reasoning="low")

    llm.update_options(extra_kwargs={"max_completion_tokens": 64})
    body = await _wire_body(llm)

    assert body["reasoning_effort"] == "low"


async def test_update_options_reasoning_with_summary_raises() -> None:
    llm = _llm()

    with pytest.raises(ValueError, match="summary"):
        llm.update_options(reasoning={"effort": "low", "summary": "auto"})
    await llm.aclose()


async def test_effort_below_model_minimum_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    llm = _llm("openai/gpt-6.1-sol", reasoning="none")
    await llm.aclose()

    assert any("lowest" in m for m in _warnings(caplog))


async def test_effort_at_model_minimum_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    llm = _llm("openai/gpt-6.1-sol", reasoning="low")
    await llm.aclose()

    assert _warnings(caplog) == []


async def test_update_options_model_below_effort_minimum_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    llm = _llm("openai/gpt-6-luna", reasoning="none")

    llm.update_options(model="openai/gpt-6.1-sol")
    await llm.aclose()

    assert any("lowest" in m for m in _warnings(caplog))
