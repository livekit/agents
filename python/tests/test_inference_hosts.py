"""The inference clients' generated facades: one native client per instance, built with it, and
the options it owns."""

from __future__ import annotations

import pytest

from livekit.agents import inference
from livekit.agents.inference.llm import ChatCompletionOptions, ProviderEndpoint

pytestmark = pytest.mark.unit

_CREDENTIALS = {"base_url": "https://gateway.invalid", "api_key": "key", "api_secret": "secret"}


def test_update_options_reaches_the_client_and_only_given_options_change() -> None:
    llm = inference.LLM(
        "openai/gpt-4.1-mini", extra_kwargs=ChatCompletionOptions(temperature=0.5), **_CREDENTIALS
    )
    llm.update_options(model="openai/gpt-4.1")
    assert llm.model == "openai/gpt-4.1"
    assert llm._native.extra_kwargs == {"temperature": 0.5}
    llm.update_options(extra_kwargs=ChatCompletionOptions(temperature=0.1))
    assert llm.model == "openai/gpt-4.1"
    assert llm._native.extra_kwargs == {"temperature": 0.1}


def test_a_provider_endpoint_needs_no_gateway_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.replace("LIVEKIT_", "LIVEKIT_INFERENCE_"), raising=False)
    endpoint = ProviderEndpoint(base_url="https://provider.invalid/v1", api_key="own")
    llm = inference.LLM("gpt-4.1", endpoint=endpoint)
    assert llm.model == "gpt-4.1"
