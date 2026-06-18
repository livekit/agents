"""Tests for Inception AI LLM plugin configuration and behavior."""

from __future__ import annotations

import pytest

from livekit.plugins.inception import LLM

# Let's write the test cleanly.
pytestmark = pytest.mark.plugin("inception")


def test_default_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
    llm = LLM()
    assert llm.model == "mercury-2"
    # AsyncClient stores the configured base URL on _base_url.
    assert str(llm._client.base_url).startswith("https://api.inceptionlabs.ai/v1")


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INCEPTION_API_KEY", raising=False)
    with pytest.raises(ValueError, match="INCEPTION_API_KEY"):
        LLM()


def test_provider_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
    assert (
        LLM().provider == "InceptionAI"
    )  # In our class we returned super().provider which is "OpenAI" or did we override it? Let's check!
    # Wait, did our LLM override provider?
    # In services.py of groq, LLM does NOT override provider. It inherits OpenAILLM. But wait! OpenAILLM.provider is "openai".
    # Wait, let's check what Groq LLM returns for provider, or what Cerebras LLM returns.
    # Let's check if OpenAILLM has a provider property we should override.
