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


__all__ = ["get_base_url", "AsyncAzureADTokenProvider", "AsyncBedrockTokenProvider"]
