"""`inference.STT` must end turns itself for OpenAI gpt-live-transcribe.

The model detects no turns: the gateway returns a final only for audio the client commits
with `session.finalize`. Without a VAD sending it at end of speech, a session gets interim
transcripts and no finals, so an AgentSession never answers. The model also takes PCM only
at 24 kHz, and the gateway refuses the 16 kHz default.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import livekit.agents.inference.vad as inference_vad
from livekit import rtc
from livekit.agents import NOT_GIVEN, APIConnectOptions
from livekit.agents.inference.stt import (
    STT,
    OpenAIOptions,
    _keyterms_extra_for_model,
)
from livekit.agents.stt import RecognizeStream, SpeechEvent, SpeechEventType

from .fake_stt import FakeUserSpeech
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _DefaultVAD(FakeVAD):
    """Stands in for the Silero VAD that inference.STT builds when none is passed."""


@pytest.fixture(autouse=True)
def _default_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_vad, "VAD", _DefaultVAD)


def _make_stt(**kwargs: Any) -> STT:
    defaults: dict[str, Any] = {
        "model": "openai/gpt-live-transcribe",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": "https://example.livekit.cloud",
    }
    defaults.update(kwargs)
    return STT(**defaults)


@pytest.mark.parametrize(
    "model", ["openai/gpt-live-transcribe", "openai/gpt-live-transcribe:en", "openai"]
)
def test_openai_gets_a_default_vad(model: str) -> None:
    assert isinstance(_make_stt(model=model)._vad, _DefaultVAD)


def test_openai_keeps_a_passed_vad() -> None:
    vad = FakeVAD()
    assert _make_stt(vad=vad)._vad is vad


@pytest.mark.parametrize(
    ("model", "gets_vad"),
    [
        ("speechmatics/enhanced", True),
        ("speechmatics/linden-1", False),
        ("deepgram/nova-3", False),
    ],
)
def test_other_models_keep_their_vad_gate(model: str, gets_vad: bool) -> None:
    assert isinstance(_make_stt(model=model)._vad, _DefaultVAD) is gets_vad


def test_vad_is_dropped_for_models_that_end_turns_server_side(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(model="deepgram/nova-3", vad=FakeVAD())

    assert stt._vad is None
    assert "`vad` will be ignored" in caplog.text


def test_switching_to_openai_adds_a_vad() -> None:
    stt = _make_stt(model="deepgram/nova-3")
    stt.update_options(model="openai/gpt-live-transcribe")
    assert isinstance(stt._vad, _DefaultVAD)


@pytest.mark.parametrize(
    "model", ["openai/gpt-live-transcribe", "openai/gpt-live-transcribe:en", "openai"]
)
def test_openai_defaults_to_24khz(model: str, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(model=model)

    assert stt._opts.sample_rate == 24000
    assert "takes audio only" not in caplog.text


def test_model_string_language_is_parsed_before_the_rate_is_chosen() -> None:
    stt = _make_stt(model="openai/gpt-live-transcribe:en")
    assert stt._opts.model == "openai/gpt-live-transcribe"
    assert stt._opts.language == "en"
    assert stt._opts.sample_rate == 24000


def test_other_models_keep_16khz() -> None:
    assert _make_stt(model="deepgram/nova-3")._opts.sample_rate == 16000


def test_a_refused_sample_rate_is_kept_with_a_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        stt = _make_stt(sample_rate=16000)

    assert stt._opts.sample_rate == 16000
    assert "takes audio only at 24000 Hz" in caplog.text


def test_switching_to_openai_re_resolves_the_default_rate() -> None:
    """A default rate follows the model, so streams created after the switch work."""
    stt = _make_stt(model="deepgram/nova-3")
    stt.update_options(model="openai/gpt-live-transcribe")
    assert stt._opts.sample_rate == 24000


def test_switching_back_re_resolves_the_default_rate() -> None:
    stt = _make_stt(model="openai/gpt-live-transcribe")
    stt.update_options(model="deepgram/nova-3")
    assert stt._opts.sample_rate == 16000
    assert stt._vad is None


def test_switching_with_an_explicit_rate_keeps_it_and_warns(caplog) -> None:
    stt = _make_stt(model="deepgram/nova-3", sample_rate=16000)
    with caplog.at_level(logging.WARNING):
        stt.update_options(model="openai/gpt-live-transcribe")

    assert stt._opts.sample_rate == 16000
    assert "takes audio only at 24000 Hz" in caplog.text


def test_keyterms_map_to_keywords() -> None:
    assert _make_stt().capabilities.keyterms is True
    assert _keyterms_extra_for_model(
        "openai/gpt-live-transcribe",
        extra_kwargs={"keywords": "LiveKit"},
        session_keyterms=["LiveKit", "Aura"],
    ) == {"keywords": ["LiveKit", "Aura"]}


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


def _silence(sample_rate: int, seconds: float) -> rtc.AudioFrame:
    samples = int(sample_rate * seconds)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _first_final(stream: RecognizeStream) -> SpeechEvent:
    async for event in stream:
        if event.type == SpeechEventType.FINAL_TRANSCRIPT:
            return event
    raise AssertionError("stream ended without a final transcript")


async def test_end_of_speech_finalizes_the_turn_while_input_is_open() -> None:
    """Input never ends here, so only a VAD-driven session.finalize can produce the final."""
    created: dict[str, Any] = {}
    audio_bytes = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal audio_bytes
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                created.update(event)
            elif event["type"] == "input_audio":
                audio_bytes += len(base64.b64decode(event["audio"]))
            elif event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hello", "language": "en"}
                )
                await ws.send_json({"type": "session.finalized"})
        return ws

    speech = FakeUserSpeech(start_time=0.0, end_time=0.05, transcript="hello", stt_delay=0.0)
    vad = FakeVAD(fake_user_speeches=[speech], min_speech_duration=0.01, min_silence_duration=0.05)
    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url=base_url, http_session=session, vad=vad)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            # room audio arrives at 48 kHz; the stream must resample it to 24 kHz
            stream.push_frame(_silence(48000, 0.1))
            final = await asyncio.wait_for(_first_final(stream), timeout=5.0)
        finally:
            await stream.aclose()

    assert final.alternatives[0].text == "hello"
    assert created["model"] == "openai/gpt-live-transcribe"
    assert created["settings"]["sample_rate"] == "24000"
    assert 0 < audio_bytes <= 24000 * 2 // 10, "more than 100 ms of 24 kHz PCM was sent"


async def _wait_until_stream_is_live(stream: RecognizeStream, *, timeout: float = 5.0) -> None:
    """Block until the stream's session is connected."""
    for _ in range(int(timeout / 0.02)):
        if stream._ws is not None and not stream._ws.closed:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("stream did not connect in time")


async def test_live_switch_off_client_endpointing_keeps_finalizing_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speechmatics detects no turns, so the client VAD finalizes them. Switching the
    parent to a server-endpointing model must not tear that VAD down while Speechmatics
    is still the running session, or later turns stop producing finals."""
    finals: list[str] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.finalize":
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hello", "language": "en"}
                )
                await ws.send_json({"type": "session.finalized"})
        return ws

    class _SpeechyVAD(FakeVAD):
        """The default VAD inference.STT builds, made to detect speech."""

        def __init__(self) -> None:
            super().__init__(
                fake_user_speeches=[
                    FakeUserSpeech(start_time=0.0, end_time=0.05, transcript="hello", stt_delay=0.0)
                ],
                min_speech_duration=0.01,
                min_silence_duration=0.05,
            )

    monkeypatch.setattr(inference_vad, "VAD", _SpeechyVAD)

    async with _gateway(handler) as (base_url, session):
        # 24 kHz so the Deepgram switch is not refused for its rate
        stt = _make_stt(
            model="speechmatics/enhanced",
            sample_rate=24000,
            base_url=base_url,
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))

        async def feed() -> None:
            while True:
                stream.push_frame(_silence(48000, 0.1))
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(feed())
        try:
            await _wait_until_stream_is_live(stream)
            assert stream._vad is not None

            # the parent moves on, but the live session stays on Speechmatics
            stt.update_options(model="deepgram/nova-3")
            final = await asyncio.wait_for(_first_final(stream), timeout=5.0)
            finals.append(final.alternatives[0].text)
        finally:
            feeder.cancel()
            await stream.aclose()

    assert finals == ["hello"], "a VAD-driven final must still reach the running session"
    assert stream._vad is not None, "the running session keeps its client endpointing"


async def test_live_switch_to_openai_keeps_client_finalization() -> None:
    """The gateway does not switch a live session's provider, so a stream on a
    server-endpointing model keeps that model and its server-side endpointing even after
    the parent moves to OpenAI: it must not wire a VAD it did not have."""
    created: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.create":
                created.update(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            assert stream._vad is None

            stt.update_options(model="openai/gpt-live-transcribe")
            await asyncio.sleep(0.05)
        finally:
            await stream.aclose()

    assert created["model"] == "deepgram/nova-3", "the session started on Deepgram"
    # the running session keeps Deepgram (and so no client endpointing), however the
    # parent and streams created later move on
    assert stream._vad is None
    assert stt._opts.model == "openai/gpt-live-transcribe"


async def test_live_switch_at_the_wrong_rate_keeps_the_session_and_warns(caplog) -> None:
    """The stream's audio rate is fixed at creation; a 16 kHz session must not be
    switched in-band to a model the gateway refuses at that rate, and its language from
    the model string must not leak onto the running session either."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    with caplog.at_level(logging.WARNING):
        async with _gateway(handler) as (base_url, session):
            stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
            try:
                await _wait_until_stream_is_live(stream)
                stream.push_frame(_silence(16000, 0.1))
                stt.update_options(model="openai/gpt-live-transcribe")
                await asyncio.sleep(0.1)  # let any (wrong) update flush
            finally:
                await stream.aclose()

    # the parent re-resolves its default, so streams created from now on work
    assert stt._opts.model == "openai/gpt-live-transcribe"
    assert stt._opts.sample_rate == 24000
    # the live session keeps the model (and rate) it was created with
    assert stream._opts.model == "deepgram/nova-3"
    assert "takes audio only at 24000 Hz" in caplog.text


async def test_a_model_suffix_language_reaches_the_stream_only_with_its_model() -> None:
    """A ``:language`` suffix belongs to its model: a stream that keeps an earlier model
    must not have the new model's language forced on it."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stream.push_frame(_silence(16000, 0.1))
            stt.update_options(model="openai/gpt-live-transcribe:fr")
            await asyncio.sleep(0.1)
        finally:
            await stream.aclose()

    assert stt._opts.language == "fr", "the parent's new model carries the language"
    assert stream._opts.language is NOT_GIVEN or stream._opts.language is None
    assert all("language" not in u["settings"] for u in updates), (
        "a language implied by the new model must not reach the session on the old one"
    )


async def test_an_explicit_language_still_reaches_a_live_stream() -> None:
    """A language passed alongside the model is the caller's, so a live stream gets it
    even when the model it is on does not change."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stt.update_options(model="deepgram/nova-3", language="fr")
            await asyncio.sleep(0.1)
        finally:
            await stream.aclose()

    assert any(u["settings"].get("language") == "fr" for u in updates)


async def test_live_switch_keeps_keyterms_on_the_running_model() -> None:
    """Framework keyterms must reach the session under the running model's key, not the
    model the parent moved to (deepgram ``keyterm`` rather than openai ``keywords``)."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stream.push_frame(_silence(16000, 0.1))
            # the parent moves to openai, but the live session stays on deepgram
            stt.update_options(model="openai/gpt-live-transcribe")
            stt._update_session_keyterms(["Acme"])
            await asyncio.sleep(0.1)
        finally:
            await stream.aclose()

    extras = [u["settings"]["extra"] for u in updates if "extra" in u["settings"]]
    assert not any("keywords" in e for e in extras), "openai key must not reach deepgram"
    assert {"keyterm": ["Acme"]} in extras, "deepgram must still get its keyterms"


async def test_a_cleared_keyterm_list_reaches_the_live_session() -> None:
    """A keyterm update must be formatted fresh: clearing the terms has to remove them
    from the running session, not resend an earlier list."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stt._update_session_keyterms(["Acme"])
            await asyncio.sleep(0.05)
            stt._update_session_keyterms([])
            await asyncio.sleep(0.05)
        finally:
            await stream.aclose()

    keyterm_updates = [
        u["settings"]["extra"]["keyterm"]
        for u in updates
        if "keyterm" in u["settings"].get("extra", {})
    ]
    assert keyterm_updates == [["Acme"], []], "the clear must reach the live session"


async def test_a_live_user_keyword_update_is_not_dropped_for_a_keyterm_model() -> None:
    """Deepgram's weighted ``keywords`` is a user option distinct from its framework
    ``keyterm``; a live update of it must reach the gateway, not be filtered out."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stt.update_options(extra={"keywords": [("LiveKit", 2.0)]})
            await asyncio.sleep(0.05)
        finally:
            await stream.aclose()

    assert any(
        u["settings"].get("extra", {}).get("keywords") == [["LiveKit", 2.0]] for u in updates
    ), "the provider's own keywords option must reach the live session"


def test_openai_options_are_accepted_by_the_typed_constructor() -> None:
    """`OpenAIOptions` is exposed through the overload, not only the implementation."""
    opts: OpenAIOptions = {"keywords": ["LiveKit"], "delay": "low"}
    stt = STT(
        model="openai/gpt-live-transcribe",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        extra_kwargs=opts,
    )
    assert stt._opts.extra_kwargs == opts


async def test_a_closed_vad_still_lets_input_end_finalize() -> None:
    """A VAD that ended on its own closes its stream; that must not abort the send task
    before it commits the audio with session.finalize, or the turn is lost."""
    message_types: list[str] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            message_types.append(event["type"])
            if event["type"] == "session.finalize":
                # only the client's commit can produce this final
                await ws.send_json(
                    {"type": "final_transcript", "transcript": "hello", "language": "en"}
                )
                await ws.send_json({"type": "session.finalized"})
        return ws

    # a VAD with nothing to report returns from its stream's main task at once, so its
    # stream is already closed by the time input ends
    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url=base_url, http_session=session, vad=FakeVAD())
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            await asyncio.sleep(0.05)  # let the VAD stream finish and close
            stream.end_input()
            final = await asyncio.wait_for(_first_final(stream), timeout=5.0)
        finally:
            await stream.aclose()

    assert "session.finalize" in message_types, "input ending must still commit the audio"
    assert final.alternatives[0].text == "hello"


async def test_an_update_during_speech_does_not_flush_deferred_keyterms() -> None:
    """Keyterms that change mid-utterance wait for a calm moment; an unrelated update
    (here a language) must not push them out early and make the gateway reconnect."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    def keyterm_updates() -> list[Any]:
        return [
            u["settings"]["extra"]["keyterm"]
            for u in updates
            if "keyterm" in u["settings"].get("extra", {})
        ]

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stream._speaking = True
            stt._update_session_keyterms(["Acme"])  # deferred to end of speech
            stt.update_options(language="fr")  # unrelated: must not flush them
            await asyncio.sleep(0.05)

            assert keyterm_updates() == [], "keyterms must wait for a calm moment"
            assert any(u["settings"].get("language") == "fr" for u in updates)

            stream._speaking = False
            stream._on_end_of_speech()  # the calm moment arrives
            await asyncio.sleep(0.05)
        finally:
            await stream.aclose()

    assert keyterm_updates() == [["Acme"]], "the deferred terms must still arrive"


async def test_a_direct_stream_suffix_language_is_not_applied_to_a_refused_model() -> None:
    """A ``:language`` suffix belongs to the model carrying it. A direct stream update
    naming a model this stream's rate refuses must not apply that model's language."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stream.push_frame(_silence(16000, 0.1))
            stream.update_options(model="openai/gpt-live-transcribe:fr")
            await asyncio.sleep(0.1)
        finally:
            await stream.aclose()

    assert stream._opts.model == "deepgram/nova-3"
    assert stream._opts.language is NOT_GIVEN or stream._opts.language is None
    assert all("language" not in u["settings"] for u in updates), (
        "a refused model must not impose its language on the running session"
    )


async def test_a_direct_stream_suffix_language_is_not_applied_across_providers() -> None:
    """A ``:language`` suffix belongs to the model carrying it. When a direct stream update
    names a model from another provider the live session stays on its running provider, so
    the new model's language must not be applied even though its rate is accepted."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        # 24 kHz so OpenAI's required rate matches; the session still runs Deepgram
        stt = _make_stt(
            model="deepgram/nova-3",
            sample_rate=24000,
            base_url=base_url,
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            stream.push_frame(_silence(24000, 0.1))
            stream.update_options(model="openai/gpt-live-transcribe:fr")
            await asyncio.sleep(0.1)
        finally:
            await stream.aclose()

    assert stream._opts.model == "deepgram/nova-3"
    assert stream._opts.language is NOT_GIVEN or stream._opts.language is None
    assert all("language" not in u["settings"] for u in updates), (
        "another provider's model must not impose its language on the running session"
    )


async def test_a_model_switch_does_not_send_the_new_providers_extra() -> None:
    """An extra written alongside a model switch is for that model. A live Deepgram stream
    must not be handed OpenAI's string ``keywords``, which it reads as weighted tuples."""
    updates: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                updates.append(event)
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(model="deepgram/nova-3", base_url=base_url, http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            await _wait_until_stream_is_live(stream)
            # 24 kHz only for the parent's default; the live session stays on Deepgram
            stt.update_options(model="openai/gpt-live-transcribe", extra={"keywords": ["Acme"]})
            await asyncio.sleep(0.1)
        finally:
            await stream.aclose()

    assert not any("keywords" in u["settings"].get("extra", {}) for u in updates), (
        "another provider's option must not reach the running session"
    )
    assert stt._opts.extra_kwargs == {"keywords": ["Acme"]}, "future streams still get it"
