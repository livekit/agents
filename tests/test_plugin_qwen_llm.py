"""Tests for the Qwen LLM wrapper (an ``openai.LLM`` subclass pointed at Model Studio)."""

from __future__ import annotations

import pytest

from livekit.plugins.qwen import LLM
from livekit.plugins.qwen.models import COMPAT_BASE_URLS

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        LLM()


def test_reads_the_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")
    assert LLM()._client.api_key == "sk-env"


def test_provider_is_qwen_not_the_base_url_host() -> None:
    # openai.LLM reports the base_url host as its provider, which for Model Studio can be a
    # per-workspace domain; usage metrics want one stable label.
    llm = LLM(
        api_key="sk-x",
        base_url="https://ws-1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    )
    assert llm.provider == "Qwen"
    assert llm.model == "qwen-plus"


def test_region_selects_the_compatible_mode_endpoint() -> None:
    intl = str(LLM(api_key="sk-x")._client.base_url).rstrip("/")
    cn = str(LLM(api_key="sk-x", region="cn")._client.base_url).rstrip("/")
    assert intl == COMPAT_BASE_URLS["intl"]
    assert cn == COMPAT_BASE_URLS["cn"]


def test_thinking_is_pinned_off_by_default() -> None:
    assert LLM(api_key="sk-x")._opts.extra_body == {"enable_thinking": False}


def test_thinking_can_be_enabled_with_a_budget() -> None:
    llm = LLM(api_key="sk-x", enable_thinking=True, thinking_budget=512)
    assert llm._opts.extra_body == {"enable_thinking": True, "thinking_budget": 512}


def test_caller_extra_body_is_merged_over_the_thinking_settings() -> None:
    llm = LLM(api_key="sk-x", extra_body={"enable_search": True})
    assert llm._opts.extra_body == {"enable_thinking": False, "enable_search": True}
