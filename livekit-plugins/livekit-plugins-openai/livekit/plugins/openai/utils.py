from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

AsyncAzureADTokenProvider = Callable[[], str | Awaitable[str]]

# Returns a fresh Amazon Bedrock bearer token for each request. Useful for
# short-lived credentials, e.g. ``aws_bedrock_token_generator.provide_token``.
AsyncBedrockTokenProvider = Callable[[], str | Awaitable[str]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


def resolve_bedrock_base_url(
    model: str, aws_region: str | None, base_url: str | None
) -> str | None:
    """Resolve the ``bedrock-mantle`` base URL for ``model``.

    On Bedrock's mantle endpoint the ``gpt-oss`` open-weight models are served on the
    ``/v1`` path, while the ``gpt-5.x`` models are served on ``/openai/v1``. The openai
    SDK's ``AsyncBedrockOpenAI`` only ever derives ``/openai/v1``, so resolve the ``/v1``
    URL for ``gpt-oss`` here. For every other case (explicit ``base_url``, an
    ``AWS_BEDROCK_BASE_URL`` override, a non-gpt-oss model, or an unresolved region) the
    value is returned unchanged so the SDK keeps its default behaviour.
    """
    if base_url is not None or not model.startswith("openai.gpt-oss"):
        return base_url
    if os.environ.get("AWS_BEDROCK_BASE_URL"):
        return base_url
    region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        return base_url
    return f"https://bedrock-mantle.{region}.api.aws/v1"


__all__ = [
    "get_base_url",
    "resolve_bedrock_base_url",
    "AsyncAzureADTokenProvider",
    "AsyncBedrockTokenProvider",
]
