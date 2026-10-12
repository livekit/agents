"""Hermetic unit tests for ``openai.LLM.with_azure`` request-parameter forwarding.

These construct the LLM object only and assert on its ``_opts`` — no network or Azure
credentials are required, so the module runs in the ``--unit`` gate.
"""

from __future__ import annotations

import pytest

from livekit.agents.llm import ChatContext
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import openai

pytestmark = pytest.mark.unit

# Dummy Azure connection details. ``AsyncAzureOpenAI`` validates that these are present at
# construction time but does not open a connection, so any non-empty values work here.
_AZURE_ENDPOINT = "https://example.openai.azure.com"
_API_VERSION = "2024-10-21"
_API_KEY = "test-key"


def test_with_azure_forwards_request_params() -> None:
    """Params shared with ``LLM.__init__`` must reach ``_opts`` instead of being dropped."""
    extra_body = {"data_sources": [{"type": "azure_search"}]}
    extra_headers = {"x-ms-custom": "1"}
    extra_query = {"foo": "bar"}
    metadata = {"team": "voice"}

    azure_llm = openai.LLM.with_azure(
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_API_VERSION,
        api_key=_API_KEY,
        azure_deployment="gpt-4o",
        store=True,
        metadata=metadata,
        prompt_cache_retention="24h",
        extra_body=extra_body,
        extra_headers=extra_headers,
        extra_query=extra_query,
    )

    opts = azure_llm._opts
    assert opts.store is True
    assert opts.metadata == metadata
    assert opts.prompt_cache_retention == "24h"
    assert opts.extra_body == extra_body
    assert opts.extra_headers == extra_headers
    assert opts.extra_query == extra_query


def test_with_azure_request_params_default_to_not_given() -> None:
    """When omitted, the forwarded params stay ``NOT_GIVEN`` so nothing extra is sent."""
    azure_llm = openai.LLM.with_azure(
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_API_VERSION,
        api_key=_API_KEY,
        azure_deployment="gpt-4o",
    )

    opts = azure_llm._opts
    assert opts.store is NOT_GIVEN
    assert opts.metadata is NOT_GIVEN
    assert opts.prompt_cache_retention is NOT_GIVEN
    assert opts.extra_body is NOT_GIVEN
    assert opts.extra_headers is NOT_GIVEN
    assert opts.extra_query is NOT_GIVEN


@pytest.mark.concurrent
async def test_store_is_forwarded_to_chat_request() -> None:
    """``store`` set on the LLM must actually reach the chat-completions request kwargs."""
    azure_llm = openai.LLM(api_key="test-key", store=True)
    stream = azure_llm.chat(chat_ctx=ChatContext.empty())
    try:
        assert stream._extra_kwargs.get("store") is True
    finally:
        await stream.aclose()


@pytest.mark.concurrent
async def test_store_absent_from_chat_request_when_unset() -> None:
    """When ``store`` is not set, it must not be injected into the request kwargs."""
    azure_llm = openai.LLM(api_key="test-key")
    stream = azure_llm.chat(chat_ctx=ChatContext.empty())
    try:
        assert "store" not in stream._extra_kwargs
    finally:
        await stream.aclose()
