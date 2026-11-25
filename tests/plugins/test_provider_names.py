from __future__ import annotations

import pytest

from livekit.plugins import anthropic, cerebras, groq, openai, perplexity, telnyx, xai

_OWN_URL = "https://llm.internal.example/v1"


@pytest.fixture(autouse=True)
def _no_endpoint_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


async def test_a_default_endpoint_reports_the_plugins_vendor() -> None:
    # Catches: the endpoint's host (api.openai.com) reported in place of the vendor.
    assert openai.LLM(api_key="k").provider == "openai"
    assert openai.responses.LLM(api_key="k").provider == "openai"
    assert openai.TTS(api_key="k").provider == "openai"
    assert openai.STT(api_key="k").provider == "openai"
    assert openai.realtime.RealtimeModel(api_key="k").provider == "openai"
    assert openai.realtime.GPTLiveModel(api_key="k").provider == "openai"
    assert anthropic.LLM(api_key="k").provider == "anthropic"


async def test_a_named_constructor_reports_its_vendor_as_that_vendors_plugin_spells_it() -> None:
    # Catches: a with_* constructor reporting its host or "unknown", or one vendor under two
    # names depending on the plugin that reaches it.
    assert openai.LLM.with_cerebras(api_key="k").provider == "cerebras"
    assert cerebras.LLM(api_key="k").provider == "cerebras"
    assert openai.LLM.with_x_ai(api_key="k").provider == xai.TTS(api_key="k").provider
    assert xai.responses.LLM(api_key="k").provider == xai.TTS(api_key="k").provider
    assert xai.realtime.RealtimeModel(api_key="k").provider == xai.TTS(api_key="k").provider
    assert openai.LLM.with_telnyx(api_key="k").provider == telnyx.TTS(api_key="k").provider
    assert openai.LLM.with_perplexity(api_key="k").provider == perplexity.LLM(api_key="k").provider
    assert groq.LLM(api_key="k").provider == groq.TTS(api_key="k").provider
    assert groq.STT(api_key="k").provider == groq.TTS(api_key="k").provider


async def test_azure_openai_reports_one_name_whatever_its_endpoint() -> None:
    # Catches: the Azure resource's host, a private name, reported as the provider.
    azure = {"azure_endpoint": "https://private-resource.openai.azure.com", "api_key": "k"}
    assert openai.LLM.with_azure(api_version="v", **azure).provider == "azure_openai"
    assert openai.TTS.with_azure(api_version="v", **azure).provider == "azure_openai"
    assert openai.STT.with_azure(api_version="v", **azure).provider == "azure_openai"
    realtime = openai.realtime.RealtimeModel.with_azure(azure_deployment="d", **azure)
    assert realtime.provider == "azure_openai"
    live = openai.realtime.GPTLiveModel.with_azure(
        azure_deployment="d", delegation="client", **azure
    )
    assert live.provider == "azure_openai"


async def test_an_endpoint_the_caller_gave_reports_unknown_never_its_host() -> None:
    # Catches: a self-hosted endpoint's host name reaching telemetry, or reported as the
    # plugin's vendor.
    assert openai.LLM(api_key="k", base_url=_OWN_URL).provider == "unknown"
    assert openai.responses.LLM(api_key="k", base_url=_OWN_URL).provider == "unknown"
    assert openai.TTS(api_key="k", base_url=_OWN_URL).provider == "unknown"
    assert openai.STT(api_key="k", base_url=_OWN_URL).provider == "unknown"
    assert openai.realtime.RealtimeModel(api_key="k", base_url=_OWN_URL).provider == "unknown"
    assert openai.realtime.GPTLiveModel(api_key="k", base_url=_OWN_URL).provider == "unknown"
    assert anthropic.LLM(api_key="k", base_url=_OWN_URL).provider == "unknown"
