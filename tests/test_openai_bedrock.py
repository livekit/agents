from __future__ import annotations

import openai
import pytest

from livekit.plugins.openai import LLM
from livekit.plugins.openai.responses import LLM as ResponsesLLM

pytestmark = pytest.mark.unit


# --- Chat Completions (gpt-oss) ---------------------------------------------


def test_chat_with_aws_bedrock_resolves_regional_endpoint() -> None:
    bedrock = LLM.with_aws_bedrock(api_key="test-token", aws_region="us-west-2")

    assert bedrock.model == "openai.gpt-oss-120b"
    assert isinstance(bedrock._client, openai.AsyncBedrockOpenAI)
    # the client derives the regional Mantle endpoint from the region
    assert bedrock.provider == "bedrock-mantle.us-west-2.api.aws"
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


def test_responses_with_aws_bedrock_defaults_to_gpt_5_5() -> None:
    bedrock = ResponsesLLM.with_aws_bedrock(api_key="test-token", aws_region="us-east-2")

    assert bedrock.model == "openai.gpt-5.5"
    assert isinstance(bedrock._client, openai.AsyncBedrockOpenAI)
    assert bedrock.provider == "bedrock-mantle.us-east-2.api.aws"
    assert bedrock._owns_client is True
    # Bedrock has no OpenAI WebSocket transport; it must use the HTTP Responses path
    assert bedrock._opts.use_websocket is False


def test_responses_with_aws_bedrock_accepts_gpt_5_4() -> None:
    bedrock = ResponsesLLM.with_aws_bedrock(
        model="openai.gpt-5.4",
        api_key="test-token",
        aws_region="us-west-2",
    )

    assert bedrock.model == "openai.gpt-5.4"
    assert bedrock.provider == "bedrock-mantle.us-west-2.api.aws"
