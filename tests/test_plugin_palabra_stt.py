"""Tests for the Palabra STT plugin (built on the palabra-ai SDK).

Covers option handling plus streaming recognition and translation pairing.
The WS mock defined below stands in for the Palabra realtime STT endpoint.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from palabra_ai.client import REGIONS, Region

from livekit import rtc
from livekit.agents import stt as agents_stt
from livekit.plugins.palabra import STT

pytestmark = pytest.mark.plugin("palabra")


@pytest.fixture(autouse=True)
def _clean_palabra_env(monkeypatch):
    """Keep tests hermetic: ignore Palabra credentials/region from the dev machine env."""
    monkeypatch.delenv("PALABRA_API_KEY", raising=False)
    monkeypatch.delenv("PALABRA_REGION", raising=False)


@pytest.fixture
async def mock_palabra_stt_ws(monkeypatch):
    """Aiohttp WS server standing in for the Palabra realtime STT endpoint.

    Patches the SDK's "eu" region, so any `STT()` created in the test connects here.

    Yields a dict:
      - `url`: ws://127.0.0.1:<port>/asr/v1/speech-to-text/stream
      - `queries`: list[dict] of query params of each WS connect
      - `audio`: list[bytes] of binary audio frames the client sent
      - `send_queue`: asyncio.Queue — put dicts here to push as server JSON frames
    """
    queries: list[dict] = []
    audio: list[bytes] = []
    send_queue: asyncio.Queue = asyncio.Queue()

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        queries.append(dict(request.query))
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def pusher():
            while not ws.closed:
                try:
                    msg = await asyncio.wait_for(send_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                await ws.send_json(msg)

        pusher_task = asyncio.create_task(pusher())
        try:
            async for raw in ws:
                if raw.type == web.WSMsgType.BINARY:
                    audio.append(raw.data)
                elif raw.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
        finally:
            pusher_task.cancel()
        return ws

    app = web.Application()
    app.router.add_get("/asr/v1/speech-to-text/stream", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    url = f"ws://127.0.0.1:{port}/asr/v1/speech-to-text/stream"

    monkeypatch.setitem(REGIONS, "eu", Region(stt=url))

    yield {
        "url": url,
        "queries": queries,
        "audio": audio,
        "send_queue": send_queue,
    }
    await runner.cleanup()


SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE // 100  # 10 ms


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x01\x00" * FRAME_SAMPLES,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=FRAME_SAMPLES,
    )


def _transcription(
    text: str,
    *,
    is_eos: bool,
    language: str = "en",
    message_type: str = "transcription",
    transcription_id: str = "tr-1",
) -> dict:
    return {
        "message_type": message_type,
        "transcription_id": transcription_id,
        "language": language,
        "is_eos": is_eos,
        "segment": {"text": text, "start_time": 0.5, "end_time": 1.5},
        "delta": {"text": text, "start_time": 1.2, "end_time": 1.5},
    }


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.05)


# --- Configuration ---


def test_stt_requires_credentials():
    with pytest.raises(ValueError, match="api_key"):
        STT()


def test_stt_defaults():
    stt_ = STT(api_key="key")
    assert stt_._opts.language is None  # server auto-detect
    assert stt_._opts.translate_languages == []
    assert stt_._opts.filler_filter is None
    assert stt_._opts.sample_rate == SAMPLE_RATE
    assert stt_.capabilities.streaming is True
    assert stt_.capabilities.interim_results is True
    assert stt_.capabilities.offline_recognize is False
    assert stt_.provider == "Palabra"


def test_stt_translate_languages_normalization():
    assert STT(api_key="k", translate_languages="es")._opts.translate_languages == ["es"]
    assert STT(api_key="k", translate_languages=["es", "fr"])._opts.translate_languages == [
        "es",
        "fr",
    ]


def test_stt_update_options():
    stt_ = STT(api_key="key")
    stt_.update_options(language="en", translate_languages="de", filler_filter=False)
    assert stt_._opts.language == "en"
    assert stt_._opts.translate_languages == ["de"]
    assert stt_._opts.filler_filter is False


# --- Streaming recognition against the local WS mock ---


async def test_stt_stream_interim_and_final(mock_palabra_stt_ws):
    ws = mock_palabra_stt_ws
    stt_ = STT(api_key="test-key")
    stream = stt_.stream()

    for _ in range(20):  # 200 ms of audio -> two 100 ms binary chunks
        stream.push_frame(_frame())
    await _wait_for(lambda: len(ws["audio"]) >= 2)
    assert len(ws["audio"][0]) == SAMPLE_RATE // 10 * 2  # 100 ms of s16le mono

    query = ws["queries"][0]
    assert query["token"] == "test-key"
    assert query["format"] == "pcm_s16le"
    assert query["sample_rate"] == str(SAMPLE_RATE)

    ws["send_queue"].put_nowait(_transcription("Hello", is_eos=False))
    ws["send_queue"].put_nowait(_transcription("Hello world.", is_eos=True))

    events = []
    async for ev in stream:
        events.append(ev)
        if ev.type == agents_stt.SpeechEventType.END_OF_SPEECH:
            break

    types = [ev.type for ev in events]
    assert types == [
        agents_stt.SpeechEventType.START_OF_SPEECH,
        agents_stt.SpeechEventType.INTERIM_TRANSCRIPT,
        agents_stt.SpeechEventType.FINAL_TRANSCRIPT,
        agents_stt.SpeechEventType.END_OF_SPEECH,
    ]
    interim, final = events[1].alternatives[0], events[2].alternatives[0]
    assert interim.text == "Hello"
    assert final.text == "Hello world."
    assert final.language == "en"

    stream.end_input()
    await stream.aclose()
    await stt_.aclose()


async def test_stt_stream_translation(mock_palabra_stt_ws):
    ws = mock_palabra_stt_ws
    stt_ = STT(api_key="test-key", language="en", translate_languages=["es"])
    stream = stt_.stream()

    stream.push_frame(_frame())
    await _wait_for(lambda: len(ws["queries"]) == 1)
    assert ws["queries"][0]["translate_languages"] == "es"

    # Source-language transcript first, then the translation: only the translation is emitted.
    ws["send_queue"].put_nowait(_transcription("Good morning.", is_eos=True))
    ws["send_queue"].put_nowait(
        _transcription(
            "Buenos días.", is_eos=True, language="es", message_type="translated_transcription"
        )
    )

    events = []
    async for ev in stream:
        events.append(ev)
        if ev.type == agents_stt.SpeechEventType.END_OF_SPEECH:
            break

    finals = [ev for ev in events if ev.type == agents_stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    data = finals[0].alternatives[0]
    assert data.text == "Buenos días."
    assert data.language == "es"
    assert data.source_languages == ["en"]
    assert data.source_texts == ["Good morning."]
    assert data.target_texts == ["Buenos días."]

    stream.end_input()
    await stream.aclose()
    await stt_.aclose()
