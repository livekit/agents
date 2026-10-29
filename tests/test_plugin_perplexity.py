from __future__ import annotations

import pytest

from livekit.plugins.perplexity import LLM, __version__

pytestmark = pytest.mark.plugin("perplexity")


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
