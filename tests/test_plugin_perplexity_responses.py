from __future__ import annotations

import pytest

from livekit.plugins.perplexity import __version__, responses

pytestmark = pytest.mark.plugin("perplexity")


def test_attribution_header_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    llm = responses.LLM()
    assert llm._client.default_headers["X-Pplx-Integration"] == f"livekit-agents/{__version__}"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
        responses.LLM()
