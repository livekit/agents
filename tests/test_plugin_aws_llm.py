"""Hermetic unit tests for the AWS Bedrock LLM plugin (no AWS access needed)."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.llm import ChatContext, ToolChoice, function_tool
from livekit.plugins.aws import LLM as BedrockLLM

pytestmark = pytest.mark.unit


async def _inference_config(model: str, **kwargs: object) -> dict:
    instance = BedrockLLM(model=model, **kwargs)
    stream = instance.chat(chat_ctx=ChatContext())
    opts = stream._opts["inferenceConfig"]
    await stream.aclose()
    return opts


async def test_temperature_sent_for_supporting_models() -> None:
    config = await _inference_config("us.anthropic.claude-sonnet-4-6", temperature=0.5, top_p=0.9)

    assert config["temperature"] == 0.5
    assert config["topP"] == 0.9


async def test_temperature_omitted_for_opus_4_7(caplog: pytest.LogCaptureFixture) -> None:
    # Claude Opus 4.7 rejects `temperature` with a ValidationException
    # ("`temperature` is deprecated for this model").
    config = await _inference_config("us.anthropic.claude-opus-4-7", temperature=0.5, top_p=0.9)

    assert "temperature" not in config
    assert "topP" not in config


async def test_sampling_params_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    # chat() runs once per turn; the warning must not repeat every turn.
    with caplog.at_level("WARNING"):
        instance = BedrockLLM(model="us.anthropic.claude-opus-4-8", temperature=0.5)
        for _ in range(3):
            stream = instance.chat(chat_ctx=ChatContext())
            await stream.aclose()

    warnings = [r for r in caplog.records if "does not support" in r.message]
    assert len(warnings) == 1
    # the model ID may contain customer data (inference-profile ARNs) and must
    # stay out of the message body
    assert "claude-opus-4-8" not in warnings[0].getMessage()
    assert warnings[0].__dict__.get("lk.pii.model") == "us.anthropic.claude-opus-4-8"


async def test_explicit_override_for_opaque_inference_profiles() -> None:
    # An application inference-profile ARN can hide the underlying model name;
    # auto-detection deliberately never guesses for those (the profile name may
    # merely reference a model, or target an unrelated one), so sampling params
    # are sent by default and supports_sampling_params=False forces them out.
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-agent-llm"
    config = await _inference_config(arn, temperature=0.5, top_p=0.9)
    assert config["temperature"] == 0.5

    config = await _inference_config(
        arn, temperature=0.5, top_p=0.9, supports_sampling_params=False
    )
    assert "temperature" not in config
    assert "topP" not in config

    # an explicit True keeps them even for known-rejecting models
    config = await _inference_config(
        "us.anthropic.claude-opus-4-7",
        temperature=0.5,
        supports_sampling_params=True,
    )
    assert config["temperature"] == 0.5


async def test_application_profile_named_after_model_is_not_misclassified() -> None:
    # A supporting-model application profile that merely references a rejecting
    # model name must keep its explicitly configured sampling parameters.
    arn = (
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/claude-opus-4-7-prod"
    )
    config = await _inference_config(arn, temperature=0.5)
    assert config["temperature"] == 0.5


async def test_temperature_omitted_for_region_prefix_and_arn() -> None:
    for model in (
        "anthropic.claude-opus-4-8",
        "eu.anthropic.claude-opus-4-7",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-5",
    ):
        config = await _inference_config(model, temperature=0.5)
        assert "temperature" not in config


async def test_default_model_still_receives_temperature() -> None:
    config = await _inference_config("amazon.nova-2-lite-v1:0", temperature=0.7)
    assert config["temperature"] == 0.7


async def test_bedrock_client_is_reused_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = BedrockLLM(model="amazon.nova-2-lite-v1:0")
    created = 0
    entered = 0
    exited = 0
    calls = 0

    class FakeClient:
        async def converse_stream(self, **kwargs: object) -> dict:
            nonlocal calls
            calls += 1

            async def empty_stream():
                if False:
                    yield {}

            return {
                "ResponseMetadata": {"RequestId": "request-id", "HTTPStatusCode": 200},
                "stream": empty_stream(),
            }

    client = FakeClient()

    class ClientContext:
        async def __aenter__(self) -> FakeClient:
            nonlocal entered
            entered += 1
            return client

        async def __aexit__(self, *args: object) -> None:
            nonlocal exited
            exited += 1

    def create_client(*args: object, **kwargs: object) -> ClientContext:
        nonlocal created
        created += 1
        return ClientContext()

    monkeypatch.setattr(instance._session, "create_client", create_client)

    for _ in range(2):
        await instance.chat(chat_ctx=ChatContext()).collect()

    assert (created, entered, calls, exited) == (1, 1, 2, 0)

    await instance.aclose()

    assert exited == 1


async def test_bedrock_client_is_not_created_when_llm_is_closed() -> None:
    instance = BedrockLLM(model="amazon.nova-2-lite-v1:0")
    await instance.aclose()

    with pytest.raises(APIConnectionError, match="AWS Bedrock LLM is closed"):
        await instance.chat(chat_ctx=ChatContext()).collect()


async def test_concurrent_bedrock_turns_open_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = BedrockLLM(model="amazon.nova-2-lite-v1:0")
    created = 0
    calls = 0

    class FakeClient:
        async def converse_stream(self, **kwargs: object) -> dict:
            nonlocal calls
            calls += 1

            async def empty_stream():
                if False:
                    yield {}

            return {
                "ResponseMetadata": {"RequestId": "request-id", "HTTPStatusCode": 200},
                "stream": empty_stream(),
            }

    class ClientContext:
        async def __aenter__(self) -> FakeClient:
            await asyncio.sleep(0)
            return FakeClient()

        async def __aexit__(self, *args: object) -> None:
            pass

    def create_client(*args: object, **kwargs: object) -> ClientContext:
        nonlocal created
        created += 1
        return ClientContext()

    monkeypatch.setattr(instance._session, "create_client", create_client)

    await asyncio.gather(
        instance.chat(chat_ctx=ChatContext()).collect(),
        instance.chat(chat_ctx=ChatContext()).collect(),
    )

    assert (created, calls) == (1, 2)
    await instance.aclose()


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
    # Claude Opus 5.5 and Fable 5.1 reject forced tool use with a ValidationException
    # ('tool_choice: type "tool" and "any" are not supported for this model.').
    for model in (
        "us.anthropic.claude-opus-5-5",
        "global.anthropic.claude-opus-5-5",
        "anthropic.claude-opus-5-5",
        "us.anthropic.claude-fable-5-1",
    ):
        assert await _tool_choice(model, "required") == {"auto": {}}
        assert await _tool_choice(model, _NAMED) == {"auto": {}}


async def test_forced_tool_choice_kept_for_other_models() -> None:
    # Opus 5 and Fable 5 still accept any/tool; "claude-fable-5" must not match "-5-1".
    for model in (
        "us.anthropic.claude-opus-5",
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
