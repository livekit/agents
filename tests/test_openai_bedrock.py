from __future__ import annotations

import openai
import pytest

from livekit.plugins.openai import LLM

pytestmark = pytest.mark.unit


def test_with_aws_bedrock_resolves_regional_endpoint() -> None:
    bedrock = LLM.with_aws_bedrock(api_key="test-token", aws_region="us-west-2")

    assert bedrock.model == "openai.gpt-oss-120b-1:0"
    assert isinstance(bedrock._client, openai.AsyncBedrockOpenAI)
    # the client derives the regional Mantle endpoint from the region
    assert bedrock.provider == "bedrock-mantle.us-west-2.api.aws"
    assert bedrock._owns_client is True


def test_with_aws_bedrock_accepts_explicit_base_url_and_model() -> None:
    bedrock = LLM.with_aws_bedrock(
        model="openai.gpt-oss-20b-1:0",
        api_key="test-token",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
    )

    assert bedrock.model == "openai.gpt-oss-20b-1:0"
    assert bedrock.provider == "bedrock-runtime.us-east-1.amazonaws.com"


def test_with_aws_bedrock_rejects_conflicting_credentials() -> None:
    # api_key and bedrock_token_provider are mutually exclusive
    with pytest.raises(openai.OpenAIError):
        LLM.with_aws_bedrock(
            api_key="test-token",
            bedrock_token_provider=lambda: "another-token",
            aws_region="us-west-2",
        )
