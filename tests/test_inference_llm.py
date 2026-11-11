"""Chat-completions inference LLM: stream parsing, usage, reasoning options and retries."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2 as httpx
import openai
import pytest
from openai.types import Reasoning
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

import livekit.agents.inference.llm as inference_llm
from livekit.agents import APIConnectionError, APIConnectOptions, APITimeoutError, inference, llm
from livekit.agents.inference._utils import HEADER_SESSION_ID
from livekit.agents.inference.llm import LLMStream
from livekit.agents.llm.utils import ThinkingTokenFilter
from livekit.agents.metrics import LLMMetrics
from livekit.plugins import openai as openai_plugin

pytestmark = pytest.mark.unit

# long enough to keep PyJWT's short-key warning out of the suite output
_SECRET = "f" * 32
_GATEWAY = "http://gateway.test/v1"


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chunk(delta: dict | None = None, *, finish_reason: str | None = None, **extra: Any) -> bytes:
    choices = (
        [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    return _sse(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": choices,
            **extra,
        }
    )


_DONE = b"data: [DONE]\n\n"
# the gateway stamps its deployment and billing tier onto a leading delta with no content
_METADATA_ONLY = _chunk(
    {
        "role": "assistant",
        "extra_content": {
            "livekit": {"inference_deployment": "d", "inference_tier_billed": "standard"}
        },
    }
)
_TEXT = _chunk({"role": "assistant", "content": "hello"})
_STOP = _chunk({}, finish_reason="stop")
_TOOL_NAME = _chunk(
    {
        "role": "assistant",
        "tool_calls": [
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup"}}
        ],
    }
)
_TOOL_ARGS = _chunk(
    {"role": "assistant", "tool_calls": [{"index": 0, "function": {"arguments": '{"q":"x"}'}}]}
)
_TOOL_DONE = _chunk({}, finish_reason="tool_calls")
_USAGE = _chunk(usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> openai.AsyncClient:
    return openai.AsyncClient(
        api_key=_SECRET,
        base_url=_GATEWAY,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _sse_response(content: bytes | httpx.AsyncByteStream) -> httpx.Response:
    headers = {"content-type": "text/event-stream"}
    if isinstance(content, bytes):
        return httpx.Response(200, headers=headers, content=content)
    return httpx.Response(200, headers=headers, stream=content)


def _ctx() -> llm.ChatContext:
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")
    return ctx


# ---------------------------------------------------------------------------
# tool-call delta parsing
# ---------------------------------------------------------------------------

CITIES = ["Paris", "London", "Rome"]


def _stream() -> LLMStream:
    stream = LLMStream.__new__(LLMStream)
    stream._tool_call_id = None
    stream._fnc_name = None
    stream._fnc_raw_arguments = None
    stream._tool_extra = None
    stream._tool_index = None
    return stream


def _call(index: int | None, id: str | None = None, name: str | None = None, args: str = ""):
    fnc = ChoiceDeltaToolCallFunction(name=name, arguments=args)
    if index is None:
        # gemini's OpenAI-compatible endpoint omits the index, and the SDK builds chunks unvalidated
        return ChoiceDeltaToolCall.model_construct(index=None, id=id, type="function", function=fnc)
    return ChoiceDeltaToolCall(index=index, id=id, type="function" if id else None, function=fnc)


def _city_call(i: int, city: str) -> ChoiceDeltaToolCall:
    return _call(i, f"call-{i}", "get_weather", json.dumps({"city": city}))


def _parse(*deltas: list[ChoiceDeltaToolCall], finish_reason: str = "tool_calls") -> list:
    choices = [Choice(index=0, delta=ChoiceDelta(tool_calls=d)) for d in deltas]
    choices.append(Choice(index=0, finish_reason=finish_reason, delta=ChoiceDelta()))
    stream = _stream()
    chunks = [stream._parse_choice("chunk", c, ThinkingTokenFilter()) for c in choices]
    return [
        call
        for chunk in chunks
        if chunk is not None and chunk.delta is not None
        for call in chunk.delta.tool_calls
    ]


def _cities(calls: list) -> list[str]:
    return [json.loads(call.arguments)["city"] for call in calls]


def test_parallel_calls_in_one_chunk_are_all_emitted() -> None:
    calls = _parse([_city_call(i, c) for i, c in enumerate(CITIES)])
    assert _cities(calls) == CITIES


@pytest.mark.parametrize("finish_reason", ["tool_calls", "stop"])
def test_parallel_calls_in_the_finishing_chunk_are_all_emitted(finish_reason: str) -> None:
    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        delta=ChoiceDelta(tool_calls=[_city_call(i, c) for i, c in enumerate(CITIES)]),
    )
    chunk = _stream()._parse_choice("chunk", choice, ThinkingTokenFilter())
    assert _cities(chunk.delta.tool_calls) == CITIES


def test_new_call_with_arguments_but_no_name_is_dropped() -> None:
    calls = _parse([_city_call(0, "Paris")], [_call(1, "call-1", args='{"city": ')])
    assert _cities(calls) == ["Paris"]


def test_arguments_before_name_are_kept() -> None:
    calls = _parse(
        [_call(0, "call-0", args='{"city":')],
        [_call(0, name="get_weather", args='"Paris"}')],
    )
    assert [(c.call_id, c.name, c.arguments) for c in calls] == [
        ("call-0", "get_weather", '{"city":"Paris"}')
    ]


def test_second_call_with_arguments_before_name_and_id() -> None:
    calls = _parse(
        [_city_call(0, "Paris")],
        [_call(1, args='{"city":')],
        [_call(1, "call-1", "get_weather", '"London"}')],
    )
    assert _cities(calls) == ["Paris", "London"]


def test_parallel_calls_without_index_are_split_by_id() -> None:
    deltas = [
        [_call(None, f"call-{i}", "get_weather", json.dumps({"city": c}))]
        for i, c in enumerate(CITIES)
    ]
    assert _cities(_parse(*deltas)) == CITIES


# ---------------------------------------------------------------------------
# usage reporting (shared by the openai plugin's chat-completions stream)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        # proxies may send null counts inside a present usage object; they default to 0
        (
            {"completion_tokens": None, "prompt_tokens": 7, "total_tokens": None},
            {
                "completion_tokens": 0,
                "prompt_tokens": 7,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cache_creation_tokens": 0,
            },
        ),
        # reasoning tokens are a subset of completion tokens, never an addition
        (
            {
                "completion_tokens": 40,
                "prompt_tokens": 7,
                "total_tokens": 47,
                "completion_tokens_details": {"reasoning_tokens": 32},
            },
            {"reasoning_tokens": 32, "completion_tokens": 40, "total_tokens": 47},
        ),
        # the cache write is a subset of prompt tokens, so totals stay untouched
        (
            {
                "completion_tokens": 40,
                "prompt_tokens": 2596,
                "total_tokens": 2636,
                "prompt_tokens_details": {"cached_tokens": 1024, "cache_write_tokens": 1536},
            },
            {
                "cache_creation_tokens": 1536,
                "prompt_cached_tokens": 1024,
                "prompt_tokens": 2596,
                "total_tokens": 2636,
            },
        ),
    ],
    ids=["null-fields", "reasoning", "cache-write"],
)
async def test_usage_is_reported(usage: dict, expected: dict) -> None:
    body = _chunk({"content": "ok", "role": "assistant"}) + _STOP + _chunk(usage=usage) + _DONE
    model = openai_plugin.LLM(model="m", client=_client(lambda _: _sse_response(body)))

    async with model.chat(chat_ctx=_ctx()) as stream:
        usages = [chunk.usage async for chunk in stream if chunk.usage is not None]
    await model.aclose()

    assert len(usages) == 1
    assert {k: getattr(usages[0], k) for k in expected} == expected


# ---------------------------------------------------------------------------
# reasoning option: sent as chat-completions reasoning_effort, nothing when unset
# ---------------------------------------------------------------------------


def _llm(model: str = "openai/gpt-5.4", **kwargs: Any) -> inference.LLM:
    return inference.LLM(model, base_url=_GATEWAY, api_key="key", api_secret=_SECRET, **kwargs)


async def _wire_body(model: inference.LLM, **chat_kwargs: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _sse_response(_TEXT + _STOP + _DONE)

    await model._client.close()
    model._client = _client(handler)
    async with model.chat(chat_ctx=_ctx(), **chat_kwargs) as stream:
        async for _ in stream:
            pass
    await model.aclose()
    return captured


@pytest.mark.parametrize(
    ("model_kwargs", "chat_kwargs", "effort"),
    [
        ({}, {}, None),
        ({"model": "openai/gpt-6-luna"}, {}, None),
        ({"reasoning": "low"}, {}, "low"),
        ({"reasoning": {"effort": "high"}}, {}, "high"),
        ({"reasoning": Reasoning(effort="medium")}, {}, "medium"),
        ({"reasoning": {}}, {}, None),
        ({"reasoning": "high", "extra_kwargs": {"reasoning_effort": "low"}}, {}, "high"),
        ({"extra_kwargs": {"reasoning_effort": "low"}}, {}, "low"),
        ({"reasoning": "low"}, {"extra_kwargs": {"reasoning_effort": "high"}}, "high"),
    ],
)
async def test_reasoning_effort_on_the_wire(
    model_kwargs: dict, chat_kwargs: dict, effort: str | None
) -> None:
    body = await _wire_body(_llm(**model_kwargs), **chat_kwargs)

    assert body.get("reasoning_effort") == effort
    assert "reasoning" not in body


@pytest.mark.parametrize(
    ("model_kwargs", "match"),
    [
        ({"reasoning": {"effort": "low", "summary": "auto"}}, "summary"),
        ({"reasoning": Reasoning(effort="low", mode="pro")}, "mode"),
        ({"extra_kwargs": {"reasoning": {"effort": "low"}}}, "reasoning="),
    ],
)
def test_unsupported_reasoning_settings_raise(model_kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _llm(**model_kwargs)


async def test_reasoning_in_update_or_per_call_extra_kwargs_raises_and_changes_nothing() -> None:
    model = _llm(extra_kwargs={"max_completion_tokens": 64})

    with pytest.raises(ValueError, match="reasoning="):
        model.update_options(extra_kwargs={"reasoning": {"effort": "low"}})
    with pytest.raises(ValueError, match="reasoning="):
        model.chat(chat_ctx=_ctx(), extra_kwargs={"reasoning": {"effort": "low"}})
    with pytest.raises(ValueError, match="summary"):
        model.update_options(reasoning={"effort": "low", "summary": "auto"})
    body = await _wire_body(model)

    assert body["max_completion_tokens"] == 64
    assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("model_kwargs", "update", "effort", "max_tokens"),
    [
        ({"reasoning": "low"}, {"reasoning": "high"}, "high", None),
        ({"extra_kwargs": {"max_completion_tokens": 64}}, {"reasoning": "high"}, "high", 64),
        ({"reasoning": "low"}, {"reasoning": None}, None, None),
        ({"reasoning": "low"}, {"extra_kwargs": {"max_completion_tokens": 64}}, "low", 64),
    ],
)
async def test_update_options_reasoning(
    model_kwargs: dict, update: dict, effort: str | None, max_tokens: int | None
) -> None:
    model = _llm(**model_kwargs)
    model.update_options(**update)
    body = await _wire_body(model)

    assert body.get("reasoning_effort") == effort
    assert body.get("max_completion_tokens") == max_tokens


@pytest.mark.parametrize(
    ("model_kwargs", "update", "warning"),
    [
        (
            {"reasoning": "high", "extra_kwargs": {"reasoning_effort": "low"}},
            None,
            "reasoning_effort",
        ),
        ({"reasoning": "low", "extra_kwargs": {"reasoning_effort": "low"}}, None, None),
        ({"model": "openai/gpt-6.1-sol", "reasoning": "none"}, None, "lowest"),
        ({"model": "openai/gpt-6.1-sol", "reasoning": "low"}, None, None),
        ({"model": "openai/gpt-6-luna", "reasoning": "none"}, "openai/gpt-6.1-sol", "lowest"),
    ],
)
async def test_reasoning_warnings(
    caplog: pytest.LogCaptureFixture, model_kwargs: dict, update: str | None, warning: str | None
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    model = _llm(**model_kwargs)
    if update:
        model.update_options(model=update)
    await model.aclose()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    if warning is None:
        assert warnings == []
    else:
        assert any(warning in m for m in warnings)


# ---------------------------------------------------------------------------
# retries: a stream that dies having emitted nothing visible must be retried
# ---------------------------------------------------------------------------


class _Stream(httpx.AsyncByteStream):
    """Yields the given SSE bytes, then either ends cleanly or stalls like a quiet provider."""

    def __init__(self, chunks: list[bytes], *, stall: bool = True) -> None:
        self._chunks = chunks
        self._stall = stall

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._stall:
            raise httpx.ReadTimeout("stalled mid-stream")
        yield _DONE


def _retry_llm(
    responder: Callable[[int], httpx.AsyncByteStream],
) -> tuple[inference.LLM, list[httpx.Request], list[LLMMetrics]]:
    attempts: list[httpx.Request] = []
    metrics: list[LLMMetrics] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return _sse_response(responder(len(attempts)))

    model = inference.LLM(model="google/gemma-4-31b-it", api_key=_SECRET, api_secret=_SECRET)
    model._client = _client(handler)
    model.on("metrics_collected", metrics.append)
    return model, attempts, metrics


async def _drain(model: inference.LLM, *, max_retry: int = 2) -> None:
    # aclose() awaits the metrics monitor, so metrics are settled once the block exits
    async with model.chat(
        chat_ctx=_ctx(),
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0, timeout=5.0),
    ) as stream:
        async for _ in stream:
            pass


async def _run_stalled(chunks: list[bytes], *, max_retry: int = 2):
    model, attempts, _ = _retry_llm(lambda _: _Stream(chunks))
    with pytest.raises(Exception) as exc_info:  # noqa: PT011
        await _drain(model, max_retry=max_retry)
    return exc_info.value, attempts


@pytest.mark.parametrize(
    ("chunks", "attempts_made"),
    [
        ([_METADATA_ONLY], 3),
        ([_METADATA_ONLY, _TOOL_NAME, _TOOL_ARGS], 3),
        ([_METADATA_ONLY, _TEXT], 1),
        ([_METADATA_ONLY, _TOOL_NAME, _TOOL_ARGS, _TOOL_DONE], 1),
    ],
    ids=["metadata-only", "partial-tool-args", "text", "completed-tool-call"],
)
async def test_retried_only_until_output_reaches_the_caller(
    chunks: list[bytes], attempts_made: int
) -> None:
    error, attempts = await _run_stalled(chunks)

    assert len(attempts) == attempts_made
    assert ("after 3 attempts" in str(error)) == (attempts_made == 3)


async def test_retries_get_fresh_sdk_session_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    session_ids = iter(("inference_first", "inference_second", "inference_third"))
    monkeypatch.setattr(
        inference_llm,
        "get_inference_headers",
        lambda **_kwargs: {HEADER_SESSION_ID: next(session_ids)},
    )
    model, attempts, _ = _retry_llm(lambda _: _Stream([_METADATA_ONLY]))
    model._opts.extra_kwargs["extra_headers"] = {HEADER_SESSION_ID: "caller_value"}

    with pytest.raises(APIConnectionError):
        await _drain(model)

    assert [r.headers[HEADER_SESSION_ID] for r in attempts] == [
        "inference_first",
        "inference_second",
        "inference_third",
    ]


async def test_stream_read_timeout_is_a_timeout_error() -> None:
    error, _ = await _run_stalled([_TEXT], max_retry=0)

    assert isinstance(error, APITimeoutError)
    assert "timed out" in str(error).lower()


async def test_ttft_spans_the_retry() -> None:
    """The clock runs until generation, not until the failed attempt's metadata."""
    model, attempts, metrics = _retry_llm(
        lambda attempt: (
            _Stream([_METADATA_ONLY], stall=True)
            if attempt == 1
            else _Stream([_METADATA_ONLY, _TEXT, _USAGE], stall=False)
        )
    )
    await _drain(model)

    assert len(attempts) == 2
    assert len(metrics) == 1
    # the first retry always waits APIConnectOptions._interval_for_retry(0) == 0.1s
    assert metrics[0].ttft > 0.09
    assert metrics[0].ttft <= metrics[0].duration


async def test_token_counts_survive_a_response_that_generates_nothing() -> None:
    model, attempts, metrics = _retry_llm(lambda _: _Stream([_METADATA_ONLY, _USAGE], stall=False))
    await _drain(model)

    assert len(attempts) == 1
    assert len(metrics) == 1
    assert metrics[0].ttft == -1
    assert metrics[0].completion_tokens == 7
    assert metrics[0].total_tokens == 18
