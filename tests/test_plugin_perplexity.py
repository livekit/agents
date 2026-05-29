from __future__ import annotations

import pytest

from livekit.plugins.perplexity import LLM, __version__
from livekit.plugins.perplexity.llm import PERPLEXITY_BASE_URL


def test_default_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    llm = LLM()
    assert llm.model == "sonar-pro"
    assert PERPLEXITY_BASE_URL == "https://api.perplexity.ai"
    # AsyncClient stores the configured base URL on _base_url.
    assert str(llm._client.base_url).startswith("https://api.perplexity.ai")


def test_attribution_header_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    """X-Pplx-Integration must be attached on every outgoing chat request."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    llm = LLM()
    extra_headers = llm._opts.extra_headers
    assert extra_headers is not None
    assert extra_headers["X-Pplx-Integration"] == f"livekit-agents/{__version__}"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
        LLM()


def test_provider_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    assert LLM().provider == "Perplexity"
