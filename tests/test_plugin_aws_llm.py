"""Hermetic unit tests for the AWS Bedrock LLM plugin (no AWS access needed)."""

from __future__ import annotations

import pytest

from livekit.agents.llm import ChatContext
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
