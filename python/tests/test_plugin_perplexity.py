from __future__ import annotations

import pytest

from livekit.plugins.perplexity import LLM, __version__

pytestmark = pytest.mark.plugin("perplexity")


def test_attribution_header_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    """X-Pplx-Integration must be attached on every outgoing chat request."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    llm = LLM()
    extra_headers = llm._extra_kwargs["extra_headers"]
    assert isinstance(extra_headers, dict)
    assert extra_headers["X-Pplx-Integration"] == f"livekit-agents/{__version__}"
