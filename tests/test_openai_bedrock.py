from __future__ import annotations

import openai
import pytest

from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM
from livekit.plugins.openai.responses import LLM as ResponsesLLM

pytestmark = pytest.mark.unit


# --- Chat Completions (gpt-oss) ---------------------------------------------


def test_chat_with_aws_bedrock_routes_gpt_oss_to_v1() -> None:
    bedrock = LLM.with_aws_bedrock(api_key="test-token", aws_region="us-west-2")

    assert bedrock.model == "openai.gpt-oss-120b"
    assert isinstance(bedrock._client, openai.AsyncBedrockOpenAI)
    assert bedrock.provider == "bedrock-mantle.us-west-2.api.aws"
    # gpt-oss is served on the mantle `/v1` path, NOT the SDK's default `/openai/v1`
    url = str(bedrock._client.base_url)
    assert ".api.aws/v1" in url and "/openai/v1" not in url
    assert bedrock._owns_client is True


def test_chat_with_aws_bedrock_accepts_explicit_base_url_and_model() -> None:
    bedrock = LLM.with_aws_bedrock(
        model="openai.gpt-oss-20b",
        api_key="test-token",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
    )

    assert bedrock.model == "openai.gpt-oss-20b"
    assert bedrock.provider == "bedrock-runtime.us-east-1.amazonaws.com"


def test_chat_with_aws_bedrock_rejects_conflicting_credentials() -> None:
    # api_key and bedrock_token_provider are mutually exclusive
    with pytest.raises(openai.OpenAIError):
        LLM.with_aws_bedrock(
            api_key="test-token",
            bedrock_token_provider=lambda: "another-token",
            aws_region="us-west-2",
        )


# --- Responses API (gpt-5.x, gpt-oss) ---------------------------------------


def test_responses_with_aws_bedrock_routes_gpt_5_5_to_openai_v1() -> None:
    bedrock = ResponsesLLM.with_aws_bedrock(api_key="test-token", aws_region="us-east-2")

    assert bedrock.model == "openai.gpt-5.5"
    assert isinstance(bedrock._client, openai.AsyncBedrockOpenAI)
    assert bedrock.provider == "bedrock-mantle.us-east-2.api.aws"
    # gpt-5.x is served on the mantle `/openai/v1` path
    assert ".api.aws/openai/v1" in str(bedrock._client.base_url)
    assert bedrock._owns_client is True
    # Bedrock has no OpenAI WebSocket transport; it must use the HTTP Responses path
    assert bedrock._opts.use_websocket is False


def test_responses_with_aws_bedrock_routes_gpt_oss_to_v1() -> None:
    bedrock = ResponsesLLM.with_aws_bedrock(
        model="openai.gpt-oss-120b",
        api_key="test-token",
        aws_region="us-west-2",
    )

    assert bedrock.model == "openai.gpt-oss-120b"
    url = str(bedrock._client.base_url)
    assert ".api.aws/v1" in url and "/openai/v1" not in url


def test_responses_with_aws_bedrock_applies_gpt_5_4_reasoning_default() -> None:
    # the `openai.` Bedrock prefix must still resolve the gpt-5.4 reasoning default,
    # matching `gpt-5.4` accessed directly via OpenAI (effort="none")
    bedrock = ResponsesLLM.with_aws_bedrock(
        model="openai.gpt-5.4", api_key="test-token", aws_region="us-east-2"
    )

    reasoning = bedrock._opts.reasoning
    assert is_given(reasoning) and reasoning.effort == "none"


def test_responses_with_aws_bedrock_gpt_5_5_has_no_reasoning_default() -> None:
    # gpt-5.5 is not in the reasoning-effort list (same as direct OpenAI usage)
    bedrock = ResponsesLLM.with_aws_bedrock(api_key="test-token", aws_region="us-east-2")

    assert bedrock.model == "openai.gpt-5.5"
    assert not is_given(bedrock._opts.reasoning)
