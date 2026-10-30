"""Live Gnani STT checks against the real Vachana API. Skipped without ``GNANI_API_KEY``.

The speech under test is synthesised with Gnani TTS (a few short, billed requests), so no
audio fixtures are committed. Everything here is a handful of short utterances: these run
only when asked for.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt

pytestmark = [
    pytest.mark.plugin("gnani"),
    pytest.mark.skipif(not os.environ.get("GNANI_API_KEY"), reason="GNANI_API_KEY not set"),
]

SAMPLE_RATE = 16000
OPTS = APIConnectOptions(max_retry=0, timeout=30)
ENGLISH = "Please book a table for four people tomorrow evening."
HINDI = "नमस्ते, आप कैसे हैं? मैं आपकी मदद के लिए यहाँ हूँ।"


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


async def _speech(session, text: str, language: str) -> rtc.AudioFrame:
    """Synthesise ``text`` with Gnani TTS and return it as one 16 kHz mono frame."""
    from livekit.plugins.gnani import TTS

    tts = TTS(model="timbre-v3.0", language=language, sample_rate=SAMPLE_RATE)
    tts._session = session
    pcm = bytearray()
    async with tts.synthesize(text, conn_options=OPTS) as stream:
        async for ev in stream:
            pcm += bytes(ev.frame.data)
    return rtc.AudioFrame(
        data=bytes(pcm),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(pcm) // 2,
    )


@pytest.fixture
async def english(session) -> rtc.AudioFrame:
    return await _speech(session, ENGLISH, "en-IN")


@pytest.fixture
async def hindi(session) -> rtc.AudioFrame:
    return await _speech(session, HINDI, "hi-IN")


def _stt(session, **kwargs):
    from livekit.plugins.gnani import STT

    stt_ = STT(sample_rate=SAMPLE_RATE, **kwargs)
    stt_._session = session
    return stt_


def _silence(seconds: float) -> rtc.AudioFrame:
    n = int(SAMPLE_RATE * seconds)
    return rtc.AudioFrame(
        data=bytes(n * 2), sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=n
    )


async def _stream_transcripts(
    stt_, frame: rtc.AudioFrame, *, flush: bool = True, wait: float = 20.0, **kwargs
) -> list[stt.SpeechEvent]:
    """Push one utterance, wait for its final transcript, and return every event seen."""
    events: list[stt.SpeechEvent] = []
    stream = stt_.stream(conn_options=OPTS, **kwargs)
    try:
        stream.push_frame(frame)
        stream.push_frame(_silence(0.3))
        if flush:
            stream.flush()
        else:
            stream.end_input()

        async def read() -> None:
            async for ev in stream:
                events.append(ev)
                if ev.type == stt.SpeechEventType.END_OF_SPEECH:
                    return

        await asyncio.wait_for(read(), timeout=wait)
    finally:
        await stream.aclose()
    return events


def _final_text(events: list[stt.SpeechEvent]) -> str:
    return " ".join(
        ev.alternatives[0].text
        for ev in events
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives
    )


# -- REST ------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["prisma-v2.5", "prisma-v3.0"])
async def test_live_recognize_english(session, english, model):
    result = await _stt(session, model=model, language="en-IN", use_streaming=False).recognize(
        english, conn_options=OPTS
    )
    assert "table" in result.alternatives[0].text.lower()


async def test_live_recognize_reports_detected_language_on_v30(session, hindi):
    result = await _stt(
        session, model="prisma-v3.0", language="hi-IN", use_streaming=False
    ).recognize(hindi, conn_options=OPTS)
    assert result.alternatives[0].text.strip()
    assert str(result.alternatives[0].language) == "hi-IN"


async def test_live_recognize_accepts_a_language_list_on_v30(session, hindi):
    result = await _stt(
        session, model="prisma-v3.0", language=["en-IN", "hi-IN"], use_streaming=False
    ).recognize(hindi, conn_options=OPTS)
    assert result.alternatives[0].text.strip()
    assert str(result.alternatives[0].language) in {"hi-IN", "en-IN"}


@pytest.mark.parametrize("fmt", ["transcribe", "codemix", "transliterate"])
async def test_live_v30_formats_are_accepted(session, hindi, fmt):
    result = await _stt(
        session, model="prisma-v3.0", language="hi-IN", format=fmt, use_streaming=False
    ).recognize(hindi, conn_options=OPTS)
    assert result.alternatives[0].text.strip()


# -- WebSocket -------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["prisma-v2.5", "prisma-v3.0"])
async def test_live_stream_transcribes_one_utterance(session, english, model):
    events = await _stream_transcripts(_stt(session, model=model, language="en-IN"), english)
    assert "table" in _final_text(events).lower()
    types = [e.type for e in events]
    assert types[0] == stt.SpeechEventType.START_OF_SPEECH
    assert types[-1] == stt.SpeechEventType.END_OF_SPEECH


async def test_live_flush_commits_without_waiting_for_the_server_vad(session, english):
    """``commit`` closes the utterance immediately, which is what manual_flush relies on."""
    started = time.monotonic()
    events = await _stream_transcripts(
        _stt(session, model="prisma-v3.0", language="en-IN"), english, flush=True
    )
    assert _final_text(events)
    assert time.monotonic() - started < 10


async def test_live_end_input_still_delivers_the_last_utterance(session, english):
    events = await _stream_transcripts(
        _stt(session, model="prisma-v3.0", language="en-IN"), english, flush=False
    )
    assert "table" in _final_text(events).lower()


async def test_live_stream_with_indic_auto(session, hindi):
    events = await _stream_transcripts(
        _stt(session, model="prisma-v3.0", language="indic-auto"), hindi
    )
    final = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert final and final[0].alternatives[0].text.strip()
    assert str(final[0].alternatives[0].language) == "hi-IN"


async def test_live_stream_with_a_language_list(session, hindi):
    """The API rejects a comma list on the WebSocket; the plugin maps it to indic-auto."""
    events = await _stream_transcripts(
        _stt(session, model="prisma-v3.0", language=["en-IN", "hi-IN"]), hindi
    )
    assert _final_text(events).strip()


async def test_live_stream_with_keyterms_and_context(session, english):
    events = await _stream_transcripts(
        _stt(
            session,
            model="prisma-v3.0",
            language="en-IN",
            keyterms=["Gnani", "Vachana"],
            topic="restaurant reservations",
            previous_text="Hello, how can I help you?",
            speaker_names=["Customer"],
        ),
        english,
    )
    assert "table" in _final_text(events).lower()


async def test_live_keyterm_headers_are_applied_by_the_server(session):
    """The server echoes its effective config on ``update_settings``; ``bias_word_count``
    proves the plugin's keyterm header was understood (it is ignored on prisma-v2.5)."""
    import websockets

    from livekit.plugins.gnani import STT
    from livekit.plugins.gnani.stt import _build_ws_headers

    async def bias_count(model: str) -> int:
        opts = STT(
            model=model,
            language="en-IN",
            keyterms=["Gnani", "Vachana", "Prisma"],
            sample_rate=SAMPLE_RATE,
        )._opts
        async with websockets.connect(
            "wss://api.vachana.ai/stt/v3/stream", additional_headers=_build_ws_headers(opts)
        ) as ws:
            await asyncio.wait_for(ws.recv(), 10)
            await ws.send(json.dumps({"type": "update_settings", "format": "verbatim"}))
            for _ in range(3):
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if msg.get("type") == "settings_updated":
                    return int(msg["config"].get("bias_word_count") or 0)
        raise AssertionError("no settings_updated reply")

    assert await bias_count("prisma-v3.0") == 3


async def test_live_update_options_changes_the_running_session(session, english):
    stt_ = _stt(session, model="prisma-v3.0", language="en-IN")
    stream = stt_.stream(conn_options=OPTS)
    try:
        stt_.update_options(format="transcribe", keyterms=["Gnani"])
        stream.push_frame(english)
        stream.push_frame(_silence(0.3))
        stream.flush()
        seen = []

        async def read() -> None:
            async for ev in stream:
                seen.append(ev)
                if ev.type == stt.SpeechEventType.END_OF_SPEECH:
                    return

        await asyncio.wait_for(read(), timeout=20)
    finally:
        await stream.aclose()
    assert _final_text(seen)


async def test_live_idle_timeout_reconnects_without_surfacing_an_error(session, english):
    """The server ends an idle session; the plugin must reconnect and keep working."""
    stt_ = _stt(session, model="prisma-v3.0", language="en-IN", idle_timeout_sec=5)
    stream = stt_.stream(conn_options=OPTS)
    errors: list[stt.STTError] = []
    stt_.on("error", errors.append)
    try:
        await asyncio.sleep(9)  # past the 5 s idle expiry
        stream.push_frame(english)
        stream.push_frame(_silence(0.3))
        stream.flush()
        seen = []

        async def read() -> None:
            async for ev in stream:
                seen.append(ev)
                if ev.type == stt.SpeechEventType.END_OF_SPEECH:
                    return

        await asyncio.wait_for(read(), timeout=25)
    finally:
        await stream.aclose()
    assert "table" in _final_text(seen).lower()
    assert not errors


# -- Failure modes ---------------------------------------------------------------------


async def test_live_bad_api_key_is_reported_as_a_non_retryable_error(session, english):
    from livekit.plugins.gnani import STT

    stt_ = STT(api_key="definitely-not-valid", sample_rate=SAMPLE_RATE, language="en-IN")
    stt_._session = session
    errors: list[stt.STTError] = []
    stt_.on("error", errors.append)
    stream = stt_.stream(conn_options=OPTS)
    try:
        stream.push_frame(english)
        stream.flush()

        async def drain() -> None:
            async for _ in stream:
                pass

        with pytest.raises(APIStatusError) as exc_info:
            await asyncio.wait_for(drain(), timeout=20)
    finally:
        await stream.aclose()
    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False
    assert errors
    assert errors[0].recoverable is False
