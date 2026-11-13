"""AWS Bedrock LLM request shaping and error handling, without AWS access."""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from livekit.agents import APIConnectOptions, APIError, APIStatusError
from livekit.agents.llm import ChatContext, ToolChoice, function_tool
from livekit.plugins.aws import LLM as BedrockLLM
from livekit.plugins.aws.llm import LLMStream

pytestmark = pytest.mark.unit


_APP_PROFILE = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile"


@pytest.mark.parametrize(
    ("model", "kwargs", "sent"),
    [
        ("us.anthropic.claude-sonnet-4-6", {}, True),
        ("amazon.nova-2-lite-v1:0", {}, True),
        # gpt-oss accepts both, so the GPT-5.6/GPT-6 entries must not match it
        ("openai.gpt-oss-120b-1:0", {}, True),
        # these reject temperature/topP with a ValidationException
        ("us.anthropic.claude-opus-4-7", {}, False),
        ("anthropic.claude-opus-4-8", {}, False),
        ("eu.anthropic.claude-opus-4-7", {}, False),
        ("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-5", {}, False),
        ("us.openai.gpt-6-sol", {}, False),
        ("us.openai.gpt-6-luna", {}, False),
        ("global.openai.gpt-6-sol", {}, False),
        ("us.openai.gpt-5.6-sol", {}, False),
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-luna",
            {},
            False,
        ),
        # an application profile hides its model, so it is never guessed from the name
        (f"{_APP_PROFILE}/my-agent-llm", {}, True),
        (f"{_APP_PROFILE}/claude-opus-4-7-prod", {}, True),
        (f"{_APP_PROFILE}/my-agent-llm", {"supports_sampling_params": False}, False),
        ("us.anthropic.claude-opus-4-7", {"supports_sampling_params": True}, True),
    ],
)
async def test_sampling_params_sent_only_to_supporting_models(
    model: str, kwargs: dict, sent: bool
) -> None:
    instance = BedrockLLM(model=model, temperature=0.5, top_p=0.9, **kwargs)
    stream = instance.chat(chat_ctx=ChatContext())
    config = stream._opts["inferenceConfig"]
    await stream.aclose()
    assert config == ({"temperature": 0.5, "topP": 0.9} if sent else {})


async def test_sampling_params_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    # chat() runs once per turn; the warning must not repeat every turn
    with caplog.at_level("WARNING"):
        instance = BedrockLLM(model="us.anthropic.claude-opus-4-8", temperature=0.5)
        for _ in range(3):
            stream = instance.chat(chat_ctx=ChatContext())
            await stream.aclose()

    warnings = [r for r in caplog.records if "does not support" in r.message]
    assert len(warnings) == 1
    # the model ID may contain customer data (inference-profile ARNs), so it stays out of the text
    assert "claude-opus-4-8" not in warnings[0].getMessage()
    assert warnings[0].__dict__.get("lk.pii.model") == "us.anthropic.claude-opus-4-8"


def test_reasoning_deltas_skipped_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    # GPT-6 streams reasoning as redactedContent bytes, gpt-oss and Claude as text and signature
    for reasoning in ({"redactedContent": b"rsn_abc"}, {"text": "hmm"}, {"signature": "sig"}):
        chunk = {"contentBlockDelta": {"delta": {"reasoningContent": reasoning}}}
        with caplog.at_level("WARNING"):
            assert LLMStream._parse_chunk(object.__new__(LLMStream), "req-1", chunk) is None

    assert not [r for r in caplog.records if "unknown chunk type" in r.getMessage()]


@function_tool
async def get_weather(city: str) -> str:
    """Look up the weather."""
    return city


async def _tool_choice(model: str, tool_choice: ToolChoice) -> dict:
    instance = BedrockLLM(model=model)
    stream = instance.chat(chat_ctx=ChatContext(), tools=[get_weather], tool_choice=tool_choice)
    choice = stream._opts["toolConfig"]["toolChoice"]
    await stream.aclose()
    return choice


_NAMED: ToolChoice = {"type": "function", "function": {"name": "get_weather"}}


async def test_forced_tool_choice_sent_as_auto_for_opus_5_5() -> None:
    # Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 reject forced tool use with a ValidationException
    for model in (
        "us.anthropic.claude-opus-5-5",
        "global.anthropic.claude-opus-5-5",
        "anthropic.claude-opus-5-5",
        "us.anthropic.claude-sonnet-5-5",
        "global.anthropic.claude-sonnet-5-5",
        "us.anthropic.claude-fable-5-1",
    ):
        assert await _tool_choice(model, "required") == {"auto": {}}
        assert await _tool_choice(model, _NAMED) == {"auto": {}}


async def test_forced_tool_choice_kept_for_other_models() -> None:
    # Opus 5, Sonnet 5 and Fable 5 still accept any/tool; "claude-fable-5" must not match "-5-1"
    for model in (
        "us.anthropic.claude-opus-5",
        "us.anthropic.claude-sonnet-5",
        "us.anthropic.claude-fable-5",
        "us.anthropic.claude-sonnet-4-6",
    ):
        assert await _tool_choice(model, "required") == {"any": {}}
        assert await _tool_choice(model, _NAMED) == {"tool": {"name": "get_weather"}}


async def test_forced_tool_choice_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        instance = BedrockLLM(model="us.anthropic.claude-opus-5-5")
        for _ in range(3):
            stream = instance.chat(
                chat_ctx=ChatContext(), tools=[get_weather], tool_choice="required"
            )
            await stream.aclose()

    warnings = [r for r in caplog.records if "forced tool_choice" in r.message]
    assert len(warnings) == 1
    assert "claude-opus-5-5" not in warnings[0].getMessage()
    assert warnings[0].__dict__.get("lk.pii.model") == "us.anthropic.claude-opus-5-5"


async def _request_messages(**kwargs: object) -> list[dict]:
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content=["You are a helpful assistant."])
    chat_ctx.add_message(role="user", content=["What's the weather in Paris?"])
    stream = BedrockLLM(model="us.anthropic.claude-haiku-4-5", **kwargs).chat(
        chat_ctx=chat_ctx, tools=[get_weather]
    )
    messages = stream._opts["messages"]
    await stream.aclose()
    return messages


async def test_cache_messages_ends_the_conversation_with_a_cache_point() -> None:
    cache_point = {"cachePoint": {"type": "default"}}
    assert await _request_messages(cache_messages=True) == [
        {"role": "user", "content": [{"text": "What's the weather in Paris?"}, cache_point]}
    ]
    assert await _request_messages() == [
        {"role": "user", "content": [{"text": "What's the weather in Paris?"}]}
    ]


@pytest.mark.virtual_time
@pytest.mark.parametrize(
    ("code", "status", "attempts"),
    [
        ("ValidationException", 400, 1),
        ("ThrottlingException", 429, 4),
        ("InternalServerException", 500, 4),
    ],
)
async def test_bedrock_400_fails_fast_throttling_and_5xx_retried(
    monkeypatch: pytest.MonkeyPatch, code: str, status: int, attempts: int
) -> None:
    # a 400 fails the same way on every retry; throttling and 5xx are still retried
    calls = 0

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def converse_stream(self, **kwargs: object) -> dict:
            nonlocal calls
            calls += 1
            raise ClientError(
                {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
                "ConverseStream",
            )

    instance = BedrockLLM(model="us.openai.gpt-6-sol")
    monkeypatch.setattr(instance._session, "create_client", lambda *a, **kw: _Client())
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    stream = instance.chat(chat_ctx=chat_ctx, conn_options=APIConnectOptions(retry_interval=0))

    with pytest.raises(APIError) as exc_info:
        await stream.collect()

    assert calls == attempts
    if attempts == 1:
        assert isinstance(exc_info.value, APIStatusError)
        assert exc_info.value.status_code == 400
