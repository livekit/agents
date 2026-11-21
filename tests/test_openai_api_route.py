from __future__ import annotations

import pytest

from livekit.plugins import openai

pytestmark = pytest.mark.unit


async def test_api_route_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_ROUTE_API_KEY", "test-key")

    llm = openai.LLM.with_api_route()
    try:
        assert llm.model == "claude-sonnet-4-6"
        assert llm.provider == "global.api-route.com"
    finally:
        await llm.aclose()


async def test_api_route_explicit_configuration() -> None:
    llm = openai.LLM.with_api_route(
        model="deepseek-v4-pro",
        api_key="explicit-key",
        base_url="https://example.com/v1",
    )
    try:
        assert llm.model == "deepseek-v4-pro"
        assert llm.provider == "example.com"
    finally:
        await llm.aclose()


def test_api_route_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_ROUTE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API Route API key is required"):
        openai.LLM.with_api_route()
