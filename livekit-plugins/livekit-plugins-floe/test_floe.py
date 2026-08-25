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
    assert floe.STT is not None
    assert floe.TTS is not None
    assert floe.FloeUsageReconciler is not None
    assert set(floe.__all__) == {
        "LLM",
        "STT",
        "TTS",
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

    def fake_remaining(api_key: object = None) -> float:
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

    def boom(api_key: object = None) -> float:
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


async def test_receipt_reads_budget_with_in_code_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from livekit.plugins.floe import receipt as receipt_mod

    monkeypatch.delenv("FLOE_API_KEY", raising=False)  # only the in-code key is available
    seen: dict[str, object] = {}

    def spy(api_key: object = None, **kwargs: object) -> float:
        seen["api_key"] = api_key
        return 5.0

    monkeypatch.setattr(receipt_mod, "hosted_remaining_usd", spy)

    async def fake_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    handlers: dict[str, object] = {}

    class _StubSession:
        def on(self, name: str, fn: object) -> None:
            handlers[name] = fn

    floe.enable_cost_receipts(_StubSession(), api_key="floe_incode")  # type: ignore[arg-type]

    handlers["session_usage_updated"](_floe_event(1000, 500))  # type: ignore[operator]
    for _ in range(10):
        await asyncio.sleep(0)
        if "api_key" in seen:
            break
    assert seen["api_key"] == "floe_incode"  # the in-code key was used, not the env


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #


def test_tts_base_url_swap_and_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.delenv("FLOE_PROVIDER_KEY", raising=False)
    t = floe.TTS()
    assert str(t._client.base_url).startswith("https://credit-api.floelabs.xyz/v1")
    assert t.provider == "floe"
    assert t.model == "openai/tts-1"


def test_tts_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        floe.TTS()


def test_tts_rejects_plaintext_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    with pytest.raises(ValueError):
        floe.TTS(base_url="http://evil.example.com/v1")


def test_tts_allows_https_and_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    floe.TTS(base_url="https://my-floe.example.com/v1")
    floe.TTS(base_url="http://localhost:8080/v1")


def test_tts_byok_sends_provider_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.setenv("FLOE_PROVIDER_KEY", "sk-test")
    t = floe.TTS()
    headers = {k.lower(): v for k, v in t._client.default_headers.items()}
    assert headers.get("x-floe-provider-key") == "sk-test"


def test_tts_task_id_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    monkeypatch.delenv("FLOE_PROVIDER_KEY", raising=False)
    t = floe.TTS(task_id="task-123")
    headers = {k.lower(): v for k, v in t._client.default_headers.items()}
    assert headers.get("x-floe-task-id") == "task-123"


def test_tts_format_selection_by_model_suffix() -> None:
    from livekit.plugins.floe.tts import _is_audio_stream_model

    assert _is_audio_stream_model("openai/tts-1") is True
    assert _is_audio_stream_model("openai/tts-1-hd") is True
    assert _is_audio_stream_model("openai/gpt-4o-mini-tts") is False


# --------------------------------------------------------------------------- #
# STT
# --------------------------------------------------------------------------- #


def test_stt_construct_and_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    s = floe.STT()
    assert s.model == "deepgram/nova-3"
    assert s.provider == "floe"
    assert s.capabilities.streaming is True
    assert s.capabilities.interim_results is True


def test_stt_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        floe.STT()


def test_stt_rejects_developer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Streaming STT is agent-scoped; a floe_live_ developer key is refused.
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        floe.STT(api_key="floe_live_deadbeef")


def test_stt_rejects_plaintext_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    with pytest.raises(ValueError):
        floe.STT(base_url="ws://evil.example.com/v1/audio/transcriptions/stream")


def test_stt_allows_wss_and_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    floe.STT(base_url="wss://my-floe.example.com/v1/audio/transcriptions/stream")
    floe.STT(base_url="ws://localhost:8080/v1/audio/transcriptions/stream")


def test_stt_rejects_out_of_range_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    with pytest.raises(ValueError):
        floe.STT(sample_rate=192000)


def test_stt_rejects_plaintext_batch_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The batch recognize() path sends the agent key + audio to batch_url, so a
    # non-loopback http:// batch_url must be refused (mirrors the ws guard).
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    with pytest.raises(ValueError):
        floe.STT(batch_url="http://evil.example.com/v1/audio/transcriptions")


def test_stt_allows_https_and_loopback_batch_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test")
    floe.STT(batch_url="https://my-floe.example.com/v1/audio/transcriptions")
    floe.STT(batch_url="http://localhost:8080/v1/audio/transcriptions")


def test_stt_auth_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_agentkey")
    s = floe.STT(task_id="t-9")
    headers = s._auth_headers()
    assert headers["Authorization"] == "Bearer floe_agentkey"
    assert headers["X-Floe-Task-Id"] == "t-9"


def test_stt_build_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.parse import parse_qs, urlparse

    from livekit.plugins.floe.stt import _build_ws_url

    url = _build_ws_url(
        "wss://credit-api.floelabs.xyz/v1/audio/transcriptions/stream",
        model="deepgram/nova-3",
        sample_rate=16000,
        language="en",
    )
    parsed = urlparse(url)
    assert parsed.scheme == "wss"
    q = parse_qs(parsed.query)
    assert q["model"] == ["deepgram/nova-3"]
    assert q["encoding"] == ["linear16"]
    assert q["sample_rate"] == ["16000"]
    assert q["language"] == ["en"]


def test_stt_speech_events_interim_then_final() -> None:
    from livekit.agents import stt as lk_stt
    from livekit.plugins.floe.stt import _speech_events

    # first non-empty interim -> START_OF_SPEECH + INTERIM
    events, speaking = _speech_events(
        {"type": "transcript", "text": "hel", "is_final": False, "speech_final": False},
        "en",
        False,
    )
    assert [e.type for e in events] == [
        lk_stt.SpeechEventType.START_OF_SPEECH,
        lk_stt.SpeechEventType.INTERIM_TRANSCRIPT,
    ]
    assert speaking is True
    assert events[1].alternatives[0].text == "hel"

    # final with speech_final -> FINAL + END_OF_SPEECH, speaking cleared
    events, speaking = _speech_events(
        {"type": "transcript", "text": "hello", "is_final": True, "speech_final": True},
        "en",
        True,
    )
    assert [e.type for e in events] == [
        lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
        lk_stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert speaking is False
    assert events[0].alternatives[0].text == "hello"


def test_stt_speech_events_empty_text_ignored() -> None:
    from livekit.plugins.floe.stt import _speech_events

    events, speaking = _speech_events(
        {"type": "transcript", "text": "", "is_final": True, "speech_final": True},
        "en",
        False,
    )
    assert events == []
    assert speaking is False


async def test_stt_batch_recognize_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit import rtc
    from livekit.agents import stt as lk_stt

    monkeypatch.setenv("FLOE_API_KEY", "floe_agentkey")
    s = floe.STT()

    captured: dict[str, object] = {}

    class _FakeResp:
        status = 200
        headers = {"X-Floe-Cost-USDC": "0.0012"}

        async def read(self) -> bytes:
            return b'{"text": "hello world"}'

        async def __aenter__(self) -> _FakeResp:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

    class _FakeSession:
        def post(self, url: str, *, data: object, headers: dict, timeout: object) -> _FakeResp:
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            return _FakeResp()

    monkeypatch.setattr(s, "_ensure_session", lambda: _FakeSession())

    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )
    event = await s.recognize([frame])

    assert captured["url"] == "https://credit-api.floelabs.xyz/v1/audio/transcriptions"
    assert captured["headers"]["Authorization"] == "Bearer floe_agentkey"  # type: ignore[index]
    assert event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello world"
