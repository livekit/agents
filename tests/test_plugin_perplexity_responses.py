from __future__ import annotations

import pytest

from livekit.plugins.perplexity import __version__, responses

pytestmark = pytest.mark.plugin("perplexity")


def test_attribution_header_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    llm = responses.LLM()
    assert llm._client.default_headers["X-Pplx-Integration"] == f"livekit-agents/{__version__}"
