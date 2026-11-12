"""AWS Bedrock LLM: Converse requests and streamed replies through a local bedrock-runtime."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from smithy_aws_event_stream.events import EventMessage
from types_aiobotocore_bedrock_runtime.type_defs import (
    ContentBlockDeltaTypeDef,
    ConverseStreamOutputTypeDef,
    ConverseStreamRequestTypeDef,
    ReasoningContentBlockDeltaTypeDef,
    TokenUsageTypeDef,
)

from livekit.agents import (
    NOT_GIVEN,
    APIConnectOptions,
    APIError,
    APIStatusError,
    NotGivenOr,
    TokenUsage,
    llm,
)
from livekit.plugins import aws

OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0)
PONG: list[ConverseStreamOutputTypeDef] = [
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "pong"}}}
]


class BedrockPeer:
    """``POST /model/{id}/converse-stream``: records each request, streams ``events`` back.

    Each event is encoded the way Bedrock sends it, one event-stream frame per event; with
    ``error`` set, every request is rejected with that ``(type, status)`` instead.
    """

    def __init__(self) -> None:
        self.events: list[ConverseStreamOutputTypeDef] = PONG
        self.error: tuple[str, int] | None = None
        self.requests: list[ConverseStreamRequestTypeDef] = []
        self.connections: set[object] = set()

    async def converse(self, request: web.Request) -> web.Response:
        self.requests.append(await request.json())
        assert request.transport is not None
        self.connections.add(request.transport.get_extra_info("peername"))
        headers = {"x-amzn-RequestId": "request-1"}
        if self.error is not None:
            kind, status = self.error
            headers["x-amzn-ErrorType"] = kind
            return web.json_response({"message": kind}, status=status, headers=headers)
        frames = b"".join(
            EventMessage(
                headers={
                    ":message-type": "event",
                    ":event-type": next(iter(event)),
                    ":content-type": "application/json",
                },
                payload=json.dumps(next(iter(event.values()))).encode(),
            ).encode()
            for event in self.events
        )
        return web.Response(
            body=frames, content_type="application/vnd.amazon.eventstream", headers=headers
        )


@pytest.fixture
async def bedrock(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[BedrockPeer]:
    peer = BedrockPeer()
    app = web.Application()
    app.router.add_post("/model/{model}/converse-stream", peer.converse)
    async with TestServer(app) as server:
        for name, value in {
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_MAX_ATTEMPTS": "1",  # the plugin's own retries are the ones under test
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": str(server.make_url("")),
        }.items():
            monkeypatch.setenv(name, value)
        yield peer


async def reply(
    model: str,
    chat_ctx: llm.ChatContext | None = None,
    *,
    tools: list[llm.Tool] | None = None,
    tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
    conn_options: APIConnectOptions = OPTIONS,
    temperature: NotGivenOr[float] = NOT_GIVEN,
    top_p: NotGivenOr[float] = NOT_GIVEN,
) -> tuple[str, TokenUsage | None]:
    """The text and usage of one reply from ``model``."""
    provider = aws.LLM(model=model, temperature=temperature, top_p=top_p)
    try:
        async with provider.chat(
            chat_ctx=chat_ctx or hello(),
            tools=tools,
            tool_choice=tool_choice,
            conn_options=conn_options,
        ) as stream:
            result = await stream.collect()
            return result.text, result.usage
    finally:
        await provider.aclose()


def hello() -> llm.ChatContext:
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")
    return ctx


@llm.function_tool
async def get_weather(city: str) -> str:
    """Look up the weather."""
    return "sunny"


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-4-7",
        "us.openai.gpt-6-sol",
        "global.openai.gpt-6-sol",
        "us.openai.gpt-5.6-sol",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-luna",
    ],
)
async def test_sampling_params_are_dropped_for_models_that_reject_them(
    bedrock: BedrockPeer, model: str
) -> None:
    """Claude Opus 4.7 and OpenAI GPT-5.6/6 are sent no temperature or topP.

    https://github.com/livekit/agents/issues/6581 https://github.com/livekit/agents/issues/7413
    """
    await reply(model, temperature=0.5, top_p=0.9)

    config = bedrock.requests[0].get("inferenceConfig", {})
    assert "temperature" not in config
    assert "topP" not in config


async def test_dropped_sampling_params_warn_once_without_the_model_id(
    bedrock: BedrockPeer, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning is logged once per LLM, the model id only in its PII field.

    https://github.com/livekit/agents/pull/6947
    """
    provider = aws.LLM(model="us.anthropic.claude-opus-4-8", temperature=0.5)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            async with provider.chat(chat_ctx=hello()) as stream:
                await stream.collect()
    await provider.aclose()

    [warning] = [r for r in caplog.records if "does not support" in r.getMessage()]
    assert "claude-opus-4-8" not in warning.getMessage()
    assert warning.__dict__.get("lk.pii.model") == "us.anthropic.claude-opus-4-8"


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-5-5",
        "global.anthropic.claude-opus-5-5",
        "anthropic.claude-opus-5-5",
        "us.anthropic.claude-fable-5-1",
    ],
)
@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
    ids=["required", "named"],
)
async def test_forced_tool_choice_is_sent_as_auto_where_unsupported(
    bedrock: BedrockPeer, model: str, tool_choice: llm.ToolChoice
) -> None:
    """Models that reject forced tool use get ``auto``. https://github.com/livekit/agents/issues/7410"""
    await reply(model, tools=[get_weather], tool_choice=tool_choice)

    assert bedrock.requests[0]["toolConfig"]["toolChoice"] == {"auto": {}}


@pytest.mark.parametrize(
    ("kind", "status", "attempts"),
    [
        ("ValidationException", 400, 1),
        ("ThrottlingException", 429, 4),
        ("InternalServerException", 500, 4),
    ],
)
async def test_a_400_fails_fast_while_throttling_and_5xx_are_retried(
    bedrock: BedrockPeer, kind: str, status: int, attempts: int
) -> None:
    """https://github.com/livekit/agents/issues/7417"""
    bedrock.error = (kind, status)

    with pytest.raises(APIError) as raised:
        await reply("us.openai.gpt-6-sol", conn_options=APIConnectOptions(retry_interval=0))

    assert len(bedrock.requests) == attempts
    if status == 400:
        assert isinstance(raised.value, APIStatusError)
        assert raised.value.status_code == 400


@pytest.mark.parametrize("tool_choice", ["auto", "required", "none", NOT_GIVEN])
async def test_tool_history_is_sent_only_with_a_tool_config(
    bedrock: BedrockPeer, tool_choice: NotGivenOr[llm.ToolChoice]
) -> None:
    """toolUse/toolResult history is always sent with the tools it refers to.

    https://github.com/livekit/agents/issues/5589
    """
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="weather?")
    ctx.items.append(llm.FunctionCall(call_id="call-1", name="get_weather", arguments="{}"))
    ctx.items.append(
        llm.FunctionCallOutput(call_id="call-1", name="get_weather", output="sunny", is_error=False)
    )
    ctx.add_message(role="assistant", content="It is sunny.")
    ctx.add_message(role="user", content="thanks")

    text, _ = await reply(
        "amazon.nova-pro-v1:0",
        ctx,
        tools=[get_weather],
        tool_choice=tool_choice,
    )

    request = bedrock.requests[0]
    blocks = [block for message in request["messages"] for block in message["content"]]
    tool_blocks = [block for block in blocks if {"toolUse", "toolResult"} & block.keys()]
    assert text == "pong"
    assert [block["text"] for block in blocks if "text" in block] == [
        "weather?",
        "It is sunny.",
        "thanks",
    ]
    assert len(tool_blocks) == (2 if "toolConfig" in request else 0)
    assert "toolConfig" in request or tool_choice == "none"


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/6007 reproduces: a request ending "
    "with the assistant is sent to no-prefill Claude models",
)
@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-opus-4-7-v1:0",
    ],
)
async def test_no_prefill_claude_requests_end_with_a_user_turn(
    bedrock: BedrockPeer, model: str
) -> None:
    """Claude 4.5+ gets a request ending with a user turn, history intact.

    https://github.com/livekit/agents/issues/6007
    """
    ctx = hello()
    ctx.add_message(role="assistant", content="How can I help?")

    await reply(model, ctx)

    messages = bedrock.requests[0]["messages"]
    assert messages[:2] == [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"text": "How can I help?"}]},
    ]
    assert messages[-1]["role"] == "user"


@pytest.mark.parametrize("cached", [False, True])
async def test_usage_counts_cached_input(bedrock: BedrockPeer, cached: bool) -> None:
    """Input tokens include cache reads and writes, which are also reported apart.

    https://github.com/livekit/agents/pull/6663
    """
    sent: TokenUsageTypeDef = {"inputTokens": 10, "outputTokens": 7, "totalTokens": 17}
    if cached:
        sent.update({"cacheReadInputTokens": 40, "cacheWriteInputTokens": 20, "totalTokens": 77})
    bedrock.events = [{"metadata": {"usage": sent, "metrics": {"latencyMs": 1}}}]

    _, usage = await reply("amazon.nova-pro-v1:0")

    assert usage is not None
    assert usage.input_tokens == (70 if cached else 10)
    assert usage.output_tokens == 7
    assert usage.cache_read_tokens == (40 if cached else None)
    assert usage.cache_write_tokens == (20 if cached else None)


@pytest.mark.parametrize(
    "reasoning",
    [
        pytest.param({"text": "secret plan"}, id="reasoning-text"),
        pytest.param({"signature": "sig"}, id="reasoning-signature"),
        pytest.param({"redactedContent": base64.b64encode(b"rsn").decode()}, id="redacted"),
        pytest.param(
            None,
            id="nova-thinking-tags",
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/1599 reproduces: Nova "
                "<thinking> text reaches the speakable reply",
            ),
        ),
    ],
)
async def test_reasoning_never_reaches_the_speakable_reply(
    bedrock: BedrockPeer, reasoning: ReasoningContentBlockDeltaTypeDef | None
) -> None:
    """Only the answer text is spoken, never the model's reasoning.

    https://github.com/livekit/agents/issues/1599 https://github.com/livekit/agents/issues/7415
    """
    delta: ContentBlockDeltaTypeDef = (
        {"text": "<thinking>secret plan</thinking>"}
        if reasoning is None
        else {"reasoningContent": reasoning}
    )
    bedrock.events = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": delta}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "The answer is 42."}}},
    ]

    text, _ = await reply("amazon.nova-pro-v1:0")

    assert text == "The answer is 42."


@pytest.mark.parametrize(
    ("mime", "image"),
    [
        ("image/jpeg", b"\xff\xd8\xff\xe0jpeg"),
        ("image/png", b"\x89PNG\r\n\x1a\npng"),
        ("image/gif", b"GIF89apixels"),
        ("image/webp", b"RIFF\x00\x00\x00\x00WEBPVP8 "),
    ],
)
async def test_image_format_matches_its_bytes(
    bedrock: BedrockPeer, mime: str, image: bytes
) -> None:
    """An image is sent with the format of its bytes, which travel unchanged.

    https://github.com/livekit/agents/pull/6328
    """
    ctx = llm.ChatContext()
    url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    ctx.add_message(role="user", content=["what is this?", llm.ImageContent(image=url)])

    await reply("amazon.nova-pro-v1:0", ctx)

    sent = bedrock.requests[0]["messages"][0]["content"][1]["image"]
    assert sent["format"] == mime.removeprefix("image/")
    encoded = sent["source"]["bytes"]  # base64 text on the wire
    assert isinstance(encoded, str)
    assert base64.b64decode(encoded) == image


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/7419 reproduces: every reply opens "
    "and closes its own bedrock-runtime client",
)
async def test_consecutive_replies_share_one_bedrock_connection(bedrock: BedrockPeer) -> None:
    """Replies reuse one open bedrock-runtime client. https://github.com/livekit/agents/issues/7419"""
    provider = aws.LLM(model="amazon.nova-pro-v1:0")

    for _ in range(2):
        async with provider.chat(chat_ctx=hello()) as stream:
            await stream.collect()
    await provider.aclose()

    assert len(bedrock.connections) == 1
