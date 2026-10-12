import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions
from livekit.agents.inference.stt import STT

pytestmark = pytest.mark.unit


def _make_stt(**kwargs):
    """Helper to create STT with required credentials."""
    defaults = {
        "model": "deepgram/nova-3",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": "https://example.livekit.cloud",
    }
    defaults.update(kwargs)
    return STT(**defaults)


class TestDefaultSampleRate:
    def test_openai_model_defaults_to_24khz(self):
        """OpenAI transcription sessions accept only 24 kHz PCM, so openai/* must not
        inherit the 16 kHz default the other inference providers expect."""
        assert _make_stt(model="openai/gpt-live-transcribe")._opts.sample_rate == 24000

    def test_openai_model_with_language_suffix_defaults_to_24khz(self):
        """The :language suffix is stripped before the rate is resolved."""
        assert _make_stt(model="openai/gpt-live-transcribe:en")._opts.sample_rate == 24000

    @pytest.mark.parametrize(
        "model",
        ["deepgram/nova-3", "cartesia/ink-whisper", "assemblyai/u3-rt-pro", "auto"],
    )
    def test_non_openai_models_keep_16khz(self, model):
        assert _make_stt(model=model)._opts.sample_rate == 16000

    def test_explicit_sample_rate_wins(self):
        assert (
            _make_stt(model="openai/gpt-live-transcribe", sample_rate=8000)._opts.sample_rate
            == 8000
        )


class TestSampleRateOnModelSwitch:
    """A default rate follows the model. A swap to or from openai/* must recompute the
    rate, or the gateway refuses the session with a 2004 before any audio arrives."""

    def test_switching_to_openai_re_resolves_the_default_rate(self):
        stt = _make_stt(model="deepgram/nova-3")
        assert stt._opts.sample_rate == 16000
        stt.update_options(model="openai/gpt-live-transcribe")
        assert stt._opts.sample_rate == 24000

    def test_switching_back_re_resolves_the_default_rate(self):
        stt = _make_stt(model="openai/gpt-live-transcribe")
        assert stt._opts.sample_rate == 24000
        stt.update_options(model="deepgram/nova-3")
        assert stt._opts.sample_rate == 16000

    def test_switching_to_openai_with_language_suffix_re_resolves_the_rate(self):
        stt = _make_stt(model="deepgram/nova-3")
        stt.update_options(model="openai/gpt-live-transcribe:en")
        assert stt._opts.sample_rate == 24000
        assert stt._opts.language == "en"

    def test_an_explicit_rate_survives_a_model_switch(self, caplog):
        stt = _make_stt(model="deepgram/nova-3", sample_rate=16000)
        with caplog.at_level(logging.WARNING):
            stt.update_options(model="openai/gpt-live-transcribe")
        assert stt._opts.sample_rate == 16000
        assert "takes audio only at 24000 Hz" in caplog.text

    def test_a_refused_rate_is_kept_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            stt = _make_stt(model="openai/gpt-live-transcribe", sample_rate=16000)
        assert stt._opts.sample_rate == 16000
        assert "takes audio only at 24000 Hz" in caplog.text

    def test_a_compatible_rate_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            _make_stt(model="openai/gpt-live-transcribe")
        assert "takes audio only" not in caplog.text


@contextlib.asynccontextmanager
async def _gateway(
    handler: Callable[[web.Request], Any],
) -> AsyncIterator[tuple[str, aiohttp.ClientSession]]:
    app = web.Application()
    app.router.add_get("/stt", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        yield str(server.make_url("")).rstrip("/"), session
    finally:
        await session.close()
        await server.close()


class _LiveGateway:
    """Records the sample_rate sent for each session and exposes an update hook."""

    def __init__(self) -> None:
        self.session_rates: list[int] = []
        self.created = asyncio.Event()
        self.updates: list[dict[str, Any]] = []

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                self.session_rates.append(int(event["settings"]["sample_rate"]))
                self.created.set()
            elif event["type"] == "session.update":
                self.updates.append(event.get("settings", {}))
        return ws


class TestLiveStreamSampleRate:
    """A stream's audio rate is fixed at creation. A live switch to a model that needs a
    different rate must be refused in place: the session keeps its running model rather
    than sending a session.create/session.update the gateway would reject with a 2004."""

    async def test_a_live_switch_to_openai_is_refused_when_the_rate_differs(self, caplog):
        gateway = _LiveGateway()
        async with _gateway(gateway.handler) as (base_url, session):
            stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                await asyncio.wait_for(gateway.created.wait(), timeout=1.0)
                with caplog.at_level(logging.WARNING):
                    stt.update_options(model="openai/gpt-live-transcribe")
                await asyncio.sleep(0)
                assert stream._opts.model == "deepgram/nova-3"
            finally:
                await stream.aclose()

        assert gateway.session_rates == [16000]
        assert not gateway.updates, "no session.update may be sent for a refused model"
        assert "takes audio only at 24000 Hz" in caplog.text

    async def test_a_live_switch_to_openai_is_applied_at_a_matching_rate(self):
        gateway = _LiveGateway()
        async with _gateway(gateway.handler) as (base_url, session):
            stt = _make_stt(
                model="deepgram/nova-3", sample_rate=24000, base_url=base_url, http_session=session
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                await asyncio.wait_for(gateway.created.wait(), timeout=1.0)
                stt.update_options(model="openai/gpt-live-transcribe")
                await asyncio.sleep(0.05)
                assert stream._opts.model == "openai/gpt-live-transcribe"
            finally:
                await stream.aclose()

        assert gateway.session_rates == [24000]
        assert gateway.updates == [{"model": "openai/gpt-live-transcribe"}]

    async def test_a_stream_created_after_the_switch_carries_the_new_rate(self):
        gateway = _LiveGateway()
        async with _gateway(gateway.handler) as (base_url, session):
            stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
            # a switch decided before the next turn's stream is created
            stt.update_options(model="openai/gpt-live-transcribe")
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                await asyncio.wait_for(gateway.created.wait(), timeout=1.0)
            finally:
                await stream.aclose()

        assert gateway.session_rates == [24000]
