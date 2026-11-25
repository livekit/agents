from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from livekit.plugins import anthropic, cerebras, groq, openai, perplexity, telnyx, xai

_OWN_URL = "https://llm.internal.example/v1"
_PYTHON = Path(__file__).parents[2] / "python"
_ID = re.compile(r"[a-z0-9_]+")


def _declared(path: Path) -> list[tuple[int, str]]:
    """Each name a file declares as a provider: what a ``provider`` property returns, and the
    vendor an OpenAI-compatible constructor names."""
    found: list[ast.expr] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "provider":
            found += [r.value for r in ast.walk(node) if isinstance(r, ast.Return) and r.value]
        elif isinstance(node, ast.keyword) and node.arg == "_provider":
            found.append(node.value)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ProviderEndpoint":
            found += [k.value for k in node.keywords if k.arg == "provider"]
    return [
        (name.lineno, name.value)
        for value in found
        for name in ast.walk(value)
        if isinstance(name, ast.Constant) and isinstance(name.value, str)
    ]


def test_every_declared_provider_is_a_lowercase_id() -> None:
    # Catches: a display name ("ElevenLabs", "Azure TTS", "d-id") declared as a provider, so one
    # vendor is spelled several ways across plugins and dashboards.
    sources = [
        *(_PYTHON / "livekit-plugins").glob("*/livekit/plugins/**/*.py"),
        *(_PYTHON / "livekit-agents/livekit/agents").rglob("*.py"),
    ]
    declared = [(path, line, name) for path in sources for line, name in _declared(path)]
    assert len(declared) > 100, "the scan no longer finds the plugins' providers"
    wrong = [
        f"{path.relative_to(_PYTHON)}:{line}: {name!r}"
        for path, line, name in declared
        if not _ID.fullmatch(name)
    ]
    assert not wrong, "a provider is a lowercase id (letters, digits, underscores):\n" + "\n".join(
        wrong
    )


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


async def test_one_vendor_has_one_id_whichever_plugin_reaches_it() -> None:
    # Catches: a with_* constructor reporting its host or "unknown", or one vendor under two
    # names depending on the plugin that reaches it.
    assert openai.LLM.with_cerebras(api_key="k").provider == "cerebras"
    assert cerebras.LLM(api_key="k").provider == "cerebras"
    for by_xai in (
        openai.LLM.with_x_ai(api_key="k"),
        xai.TTS(api_key="k"),
        xai.responses.LLM(api_key="k"),
        xai.realtime.RealtimeModel(api_key="k"),
    ):
        assert by_xai.provider == "xai"
    assert openai.LLM.with_fireworks(api_key="k").provider == "fireworksai"
    assert openai.LLM.with_telnyx(api_key="k").provider == telnyx.TTS(api_key="k").provider
    assert openai.LLM.with_perplexity(api_key="k").provider == "perplexity"
    assert perplexity.LLM(api_key="k").provider == "perplexity"
    for by_groq in (groq.LLM(api_key="k"), groq.STT(api_key="k"), groq.TTS(api_key="k")):
        assert by_groq.provider == "groq"


async def test_azure_openai_reports_one_name_whatever_its_endpoint() -> None:
    # Catches: the Azure resource's host, a private name, reported as the provider.
    endpoint = "https://private-resource.openai.azure.com"
    for model in (
        openai.LLM.with_azure(azure_endpoint=endpoint, api_key="k", api_version="v"),
        openai.TTS.with_azure(azure_endpoint=endpoint, api_key="k", api_version="v"),
        openai.STT.with_azure(azure_endpoint=endpoint, api_key="k", api_version="v"),
        openai.realtime.RealtimeModel.with_azure(
            azure_deployment="d", azure_endpoint=endpoint, api_key="k"
        ),
        openai.realtime.GPTLiveModel.with_azure(
            azure_deployment="d", azure_endpoint=endpoint, api_key="k", delegation="client"
        ),
    ):
        assert model.provider == "azure_openai"


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
