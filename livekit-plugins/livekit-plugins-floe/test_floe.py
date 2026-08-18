# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest

from livekit.agents import Plugin
from livekit.plugins import floe

pytestmark = pytest.mark.unit


def test_exports() -> None:
    assert floe.LLM is not None
    assert floe.FloeUsageReconciler is not None
    assert set(floe.__all__) == {"LLM", "FloeUsageReconciler", "__version__"}


def test_plugin_registered() -> None:
    names = [type(p).__name__ for p in Plugin.registered_plugins]
    assert "FloePlugin" in names


def test_byok_rejects_plaintext_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.setenv("FLOE_PROVIDER_KEY", "sk-test")
    with pytest.raises(ValueError):
        floe.LLM(base_url="http://evil.example.com/v1")


def test_byok_allows_https_and_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.setenv("FLOE_PROVIDER_KEY", "sk-test")
    floe.LLM(base_url="https://my-floe.example.com/v1")
    floe.LLM(base_url="http://localhost:8080/v1")


def test_keyless_rejects_plaintext_nonlocal(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even keyless sends the Floe API key, so the TLS guard applies here too.
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.delenv("FLOE_PROVIDER_KEY", raising=False)
    with pytest.raises(ValueError):
        floe.LLM(base_url="http://evil.example.com/v1")


def test_keyless_allows_https_and_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.delenv("FLOE_PROVIDER_KEY", raising=False)
    floe.LLM(base_url="https://my-floe.example.com/v1")
    floe.LLM(base_url="http://localhost:8080/v1")


def test_byok_rejects_plaintext_custom_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.setenv("FLOE_PROVIDER_KEY", "sk-test")
    import openai

    with pytest.raises(ValueError):
        floe.LLM(client=openai.AsyncClient(api_key="x", base_url="http://evil.example.com/v1"))


def test_byok_allows_https_custom_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.setenv("FLOE_PROVIDER_KEY", "sk-test")
    import openai

    floe.LLM(client=openai.AsyncClient(api_key="x", base_url="https://my-floe.example.com/v1"))


def test_reconciler_prices_only_floe_routed_usage() -> None:
    from livekit.agents.metrics import AgentSessionUsage, LLMModelUsage

    floe_use = LLMModelUsage(
        provider="floe", model="openai/gpt-4o", input_tokens=1000, output_tokens=500
    )
    other = LLMModelUsage(provider="openai", model="gpt-4o", input_tokens=9999, output_tokens=9999)

    rec = floe.FloeUsageReconciler()
    rec._latest = AgentSessionUsage(model_usage=[floe_use, other])
    report = rec.summary()

    # Only the Floe-routed entry is priced; the non-Floe one is ignored entirely.
    assert report.total_estimated_usd == pytest.approx(0.0075)
    assert len(report.per_model) == 1
    assert report.per_model[0].provider == "openai"  # display split from the id
    assert report.per_model[0].model == "gpt-4o"
    assert report.unpriced_models == []
