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
    assert set(floe.__all__) == {
        "LLM",
        "FloeUsageReconciler",
        "enable_cost_receipts",
        "__version__",
    }


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


def test_enable_cost_receipts_logs_only_floe_turns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    import types

    from livekit.agents.metrics import AgentSessionUsage, LLMModelUsage

    monkeypatch.delenv("FLOE_API_KEY", raising=False)  # keyless: no network budget read

    class _StubSession:
        def __init__(self) -> None:
            self._handlers: dict[str, object] = {}

        def on(self, name: str, fn: object) -> None:
            self._handlers[name] = fn

        def emit(self, name: str, ev: object) -> None:
            self._handlers[name](ev)  # type: ignore[operator]

    session = _StubSession()
    floe.enable_cost_receipts(session)  # type: ignore[arg-type]

    floe_use = LLMModelUsage(
        provider="floe", model="openai/gpt-4o", input_tokens=1000, output_tokens=500
    )
    non_floe = LLMModelUsage(
        provider="openai", model="gpt-4o", input_tokens=9999, output_tokens=9999
    )
    ev = types.SimpleNamespace(usage=AgentSessionUsage(model_usage=[floe_use, non_floe]))

    with caplog.at_level(logging.INFO, logger="livekit.plugins.floe"):
        session.emit("session_usage_updated", ev)

    receipts = [r.getMessage() for r in caplog.records if r.getMessage().startswith("floe · ")]
    assert receipts == ["floe · gpt-4o · $0.0075 est"]  # non-floe turn ignored


def _floe_event(cum_in: int, cum_out: int) -> object:
    import types

    from livekit.agents.metrics import AgentSessionUsage, LLMModelUsage

    usage = LLMModelUsage(
        provider="floe", model="openai/gpt-4o", input_tokens=cum_in, output_tokens=cum_out
    )
    return types.SimpleNamespace(usage=AgentSessionUsage(model_usage=[usage]))


def _capture_handler(session_holder: dict[str, object]) -> None:
    class _StubSession:
        def on(self, name: str, fn: object) -> None:
            session_holder[name] = fn

    floe.enable_cost_receipts(_StubSession())  # type: ignore[arg-type]


async def test_receipt_budget_read_is_offloaded_not_blocking(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import asyncio
    import logging

    from livekit.plugins.floe import receipt as receipt_mod

    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    calls = {"n": 0}

    def fake_remaining() -> float:
        calls["n"] += 1
        return 12.34

    monkeypatch.setattr(receipt_mod, "hosted_remaining_usd", fake_remaining)

    async def fake_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return fn(*args, **kwargs)  # run inline so the refresh completes deterministically

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    handlers: dict[str, object] = {}
    _capture_handler(handlers)
    handler = handlers["session_usage_updated"]

    with caplog.at_level(logging.INFO, logger="livekit.plugins.floe"):
        handler(_floe_event(1000, 500))  # type: ignore[operator]
        # the blocking read must NOT have run synchronously on the handler thread
        assert calls["n"] == 0
        turn1 = [r.getMessage() for r in caplog.records if r.getMessage().startswith("floe · ")]
        assert turn1 == ["floe · gpt-4o · $0.0075 est"]  # budget not yet cached

        for _ in range(10):  # let the scheduled background refresh run
            await asyncio.sleep(0)
            if calls["n"]:
                break
        assert calls["n"] == 1  # fetched exactly once, off the handler thread

        caplog.clear()
        handler(_floe_event(1800, 900))  # type: ignore[operator]
    turn2 = [r.getMessage() for r in caplog.records if r.getMessage().startswith("floe · ")]
    assert turn2 == ["floe · gpt-4o · $0.0060 est · left $12.34"]  # cached budget applied


async def test_receipt_budget_failure_drops_budget(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import asyncio
    import logging

    from livekit.plugins.floe import receipt as receipt_mod

    monkeypatch.setenv("FLOE_API_KEY", "floe_test")

    def boom() -> float:
        raise RuntimeError("network down")

    monkeypatch.setattr(receipt_mod, "hosted_remaining_usd", boom)

    async def fake_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    handlers: dict[str, object] = {}
    _capture_handler(handlers)
    handler = handlers["session_usage_updated"]

    with caplog.at_level(logging.INFO, logger="livekit.plugins.floe"):
        handler(_floe_event(1000, 500))  # type: ignore[operator]  # schedules a refresh that fails
        for _ in range(10):
            await asyncio.sleep(0)
        caplog.clear()
        handler(_floe_event(1800, 900))  # type: ignore[operator]
    turn2 = [r.getMessage() for r in caplog.records if r.getMessage().startswith("floe · ")]
    assert turn2 == ["floe · gpt-4o · $0.0060 est"]  # budget dropped on failure, cost still prints
