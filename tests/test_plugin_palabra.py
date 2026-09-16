"""Tests for the Palabra STT and TTS plugins (built on the palabra-ai SDK).

Covers option handling, end-to-end streaming, the TTS interruption path, and
STT recognition/translation pairing. The WS mocks defined below stand in for
the Palabra realtime endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json

import pytest
from aiohttp import web
from palabra_ai.client import REGIONS, Region

from livekit import rtc
from livekit.agents import APIConnectOptions, APITimeoutError, stt as agents_stt
from livekit.plugins.palabra import STT, TTS

pytestmark = pytest.mark.plugin("palabra")


@pytest.fixture(autouse=True)
def _clean_palabra_env(monkeypatch):
    """Keep tests hermetic: ignore Palabra credentials/region from the dev machine env."""
    monkeypatch.delenv("PALABRA_API_KEY", raising=False)
    monkeypatch.delenv("PALABRA_REGION", raising=False)


@pytest.fixture
async def mock_palabra_ws(monkeypatch):
    """Aiohttp WS server standing in for the Palabra realtime TTS endpoint.

    Patches the SDK's "eu" region, so any `TTS()` created in the test connects here.

    Yields a dict:
      - `url`: ws://127.0.0.1:<port>/v1/text-to-speech/stream
      - `received`: list[dict] of decoded JSON frames the client sent
      - `send_queue`: asyncio.Queue — put dicts here to push as server frames
      - `close_event`: asyncio.Event — set to instruct the server to close the WS
    """
    received: list[dict] = []
    send_queue: asyncio.Queue = asyncio.Queue()
    close_event = asyncio.Event()

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        last_gen = {"id": ""}

        async def pusher():
            while not ws.closed:
                try:
                    msg = await asyncio.wait_for(send_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if close_event.is_set():
                        await ws.close()
                        return
                    continue
                # Like the real server, audio_chunk frames echo the client's generation_id:
                #   - a frame queued without an id waits for the next text frame
                #     and takes its id;
                #   - a frame queued with an explicit id goes out as-is (a stale generation).
                data = msg.get("data")
                if (
                    msg.get("message_type") == "audio_chunk"
                    and data is not None
                    and "generation_id" not in data
                ):
                    seen = last_gen["id"]
                    while last_gen["id"] == seen:
                        await asyncio.sleep(0.005)
                        if ws.closed or close_event.is_set():
                            return
                    data["generation_id"] = last_gen["id"]
                await ws.send_json(msg)

        pusher_task = asyncio.create_task(pusher())
        try:
            async for raw in ws:
                if raw.type == web.WSMsgType.TEXT:
                    frame = json.loads(raw.data)
                    received.append(frame)
                    if frame.get("type") == "text" and frame.get("generation_id"):
                        last_gen["id"] = frame["generation_id"]
                elif raw.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
        finally:
            pusher_task.cancel()
        return ws

    app = web.Application()
    app.router.add_get("/v1/text-to-speech/stream", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    url = f"ws://127.0.0.1:{port}/v1/text-to-speech/stream"

    monkeypatch.setitem(REGIONS, "eu", Region(tts=url))

    yield {
        "url": url,
        "received": received,
        "send_queue": send_queue,
        "close_event": close_event,
    }
    await runner.cleanup()


def _audio_chunk(pcm: bytes, *, last_chunk: bool, generation_id: str | None = None) -> dict:
    data: dict = {
        "audio": base64.b64encode(pcm).decode(),
        "size": len(pcm),
        "last_chunk": last_chunk,
    }
    if generation_id is not None:
        data["generation_id"] = generation_id
    return {"message_type": "audio_chunk", "data": data}


async def _collect(stream) -> bytes:
    out: list[bytes] = []
    async for ev in stream:
        if ev.frame is not None:
            out.append(bytes(ev.frame.data))
    return b"".join(out)


# --- TTS configuration ---


def test_tts_requires_credentials():
    with pytest.raises(ValueError, match="api_key"):
        TTS()


def test_tts_accepts_api_key_directly():
    tts = TTS(api_key="direct-key")
    assert tts._client.api_key == "direct-key"


def test_tts_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("PALABRA_API_KEY", "env-key")
    tts = TTS()
    assert tts._client.api_key == "env-key"


def test_tts_defaults():
    tts = TTS(api_key="key")
    assert tts._opts.language == "en"
    assert tts._opts.voice_id == "default_low"
    assert tts._opts.speed is None  # server default
    assert tts._opts.model == "auto"
    assert tts._opts.deaccent_strength == 1.0
    assert tts._client.region == "eu"


def test_tts_capabilities_and_audio():
    tts = TTS(api_key="key")
    assert tts.capabilities.streaming is True
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1


def test_tts_custom_sample_rate():
    tts = TTS(api_key="key", sample_rate=48000)
    assert tts.sample_rate == 48000


def test_tts_model_and_provider_properties():
    tts = TTS(api_key="key")
    assert tts.model == "auto"
    assert tts.provider == "Palabra"


def test_update_options():
    tts = TTS(api_key="key")
    tts.update_options(language="ru", voice_id="default_high", speed=0.7)
    assert tts._opts.language == "ru"
    assert tts._opts.voice_id == "default_high"
    assert tts._opts.speed == 0.7
    assert tts._opts_dirty is True  # session reconnects on next stream


# --- TTS end-to-end stream against the local WS mock ---


async def test_synthesize_stream_full_flow(mock_palabra_ws):
    ws = mock_palabra_ws

    tts = TTS(api_key="test-key")
    stream = tts.stream()
    stream.push_text("Hello world.")
    stream.end_input()

    pcm = b"\x11" * 4800
    ws["send_queue"].put_nowait(_audio_chunk(pcm, last_chunk=True))

    audio = await _collect(stream)

    assert pcm in audio
    init_frames = [m for m in ws["received"] if m.get("type") == "init"]
    assert len(init_frames) == 1
    assert init_frames[0]["language"] == "en"
    assert init_frames[0]["output"] == {"format": "pcm", "sample_rate": 24000}
    text_frames = [m for m in ws["received"] if m.get("type") == "text"]
    assert any(m.get("is_eos") for m in text_frames)

    await tts.aclose()


async def test_synthesize_one_shot(mock_palabra_ws):
    ws = mock_palabra_ws

    tts = TTS(api_key="test-key")
    chunked = tts.synthesize("Hello there.")

    pcm = b"\xcc" * 1000
    ws["send_queue"].put_nowait(_audio_chunk(pcm, last_chunk=True))

    audio = await _collect(chunked)

    assert pcm in audio
    text_frames = [m for m in ws["received"] if m.get("type") == "text"]
    assert text_frames and any(m.get("is_eos") for m in text_frames)
    await tts.aclose()


async def test_session_reused_across_turns(mock_palabra_ws):
    """Two turns on one TTS reuse the SDK session.

    The Palabra `init` frame is sent once (one WS connection) and both turns produce audio.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    async def _turn(text: str, pcm: bytes) -> bytes:
        ws["send_queue"].put_nowait(_audio_chunk(pcm, last_chunk=True))
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        return await _collect(stream)

    first = await _turn("First turn.", b"\x11" * 2000)
    second = await _turn("Second turn.", b"\x22" * 2000)

    assert b"\x11" * 2000 in first
    assert b"\x22" * 2000 in second  # 2nd turn on the reused session still works
    init_frames = [m for m in ws["received"] if m.get("type") == "init"]
    assert len(init_frames) == 1  # init sent once (session reused)

    await tts.aclose()


# --- TTS interruption and generation routing ---


async def test_interruption_flow(mock_palabra_ws):
    """Interruption end to end: cancel is sent, the session survives, stale audio is dropped.

    - closing the stream mid-generation sends `cancel` frames for its own generations;
    - aclose() returns quickly — the framework awaits it before the next reply starts;
    - the next turn synthesizes on the same session (no reconnect);
    - late chunks of the cancelled generation are filtered, not played.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    ws["send_queue"].put_nowait(_audio_chunk(b"\x44" * 2000, last_chunk=False))
    stream = tts.stream()
    stream.push_text("A long reply that will be interrupted.")
    stream.flush()

    async for ev in stream:  # wait for the first audio frame, then interrupt
        assert ev.frame is not None
        break

    loop = asyncio.get_running_loop()
    started = loop.time()
    await stream.aclose()
    assert loop.time() - started < 1.5

    for _ in range(100):
        if any(m.get("type") == "cancel" for m in ws["received"]):
            break
        await asyncio.sleep(0.01)
    cancels = [m for m in ws["received"] if m.get("type") == "cancel"]
    assert cancels
    sent_gens = {m["generation_id"] for m in ws["received"] if m.get("type") == "text"}
    assert all(m.get("generation_id") in sent_gens for m in cancels)
    cancelled_gen = cancels[0]["generation_id"]

    # Next turn on the same session; a late chunk of the cancelled generation arrives first.
    junk = b"\xde\xad" * 500
    real = b"\x88" * 2000
    ws["send_queue"].put_nowait(_audio_chunk(junk, last_chunk=True, generation_id=cancelled_gen))
    ws["send_queue"].put_nowait(_audio_chunk(real, last_chunk=True))

    stream2 = tts.stream()
    stream2.push_text("You said: hello.")
    stream2.end_input()
    audio = await asyncio.wait_for(_collect(stream2), timeout=8.0)

    assert real in audio
    assert junk not in audio  # stale generation was filtered, not played
    init_frames = [m for m in ws["received"] if m.get("type") == "init"]
    assert len(init_frames) == 1  # the interrupted session was reused, not replaced
    await tts.aclose()


async def test_overlapping_streams_get_their_own_audio(mock_palabra_ws):
    """Two live streams on one session receive their own audio in crossed order.

    The session reader routes frames by generation_id, so neither stream can
    consume or drop the other stream's chunks.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    s1 = tts.stream()
    s1.push_text("First reply.")
    s1.end_input()
    s2 = tts.stream()
    s2.push_text("Second reply.")
    s2.end_input()

    for _ in range(200):
        if len([m for m in ws["received"] if m.get("type") == "text"]) >= 2:
            break
        await asyncio.sleep(0.01)
    by_text = {m["text"]: m["generation_id"] for m in ws["received"] if m.get("type") == "text"}
    g1, g2 = by_text["First reply."], by_text["Second reply."]

    a1, a2 = b"\x21" * 2000, b"\x42" * 2000
    for pcm, gen in ((a2, g2), (a1, g1)):  # crossed: second stream's audio first
        ws["send_queue"].put_nowait(_audio_chunk(pcm, last_chunk=True, generation_id=gen))

    r1, r2 = await asyncio.wait_for(asyncio.gather(_collect(s1), _collect(s2)), timeout=10.0)
    assert a1 in r1
    assert a2 in r2
    await tts.aclose()


@pytest.mark.parametrize(
    "phase", ["no_audio_times_out", "partial_audio_completes", "slow_input_no_timeout"]
)
async def test_timeout_phases(mock_palabra_ws, phase):
    """The receive deadline applies only while sent generations are outstanding.

    - no audio at all for sent text -> APITimeoutError instead of ending empty;
    - partial audio without a terminal chunk -> delivered normally, not failed;
    - a pause in the LLM input longer than the timeout -> not a failure.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.5))
    pcm = b"\x55" * 2000

    if phase == "no_audio_times_out":
        stream.push_text("Hello world.")
        stream.end_input()
        with pytest.raises(APITimeoutError):
            await _collect(stream)
        await tts.aclose()
        return

    push_task: asyncio.Task | None = None
    if phase == "partial_audio_completes":
        ws["send_queue"].put_nowait(_audio_chunk(pcm, last_chunk=False))
        stream.push_text("Hello world.")
        stream.end_input()
    else:  # slow_input_no_timeout

        async def _late_push() -> None:
            await asyncio.sleep(1.2)  # longer than the timeout: the LLM is still producing
            stream.push_text("Hello world.")
            stream.end_input()

        ws["send_queue"].put_nowait(_audio_chunk(pcm, last_chunk=True))
        push_task = asyncio.create_task(_late_push())

    audio = await _collect(stream)
    if push_task is not None:
        await push_task

    assert pcm in audio
    await tts.aclose()


# --- STT fixtures and helpers ---


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


# --- STT configuration ---


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


# --- STT streaming recognition against the local WS mock ---


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


async def test_late_event_after_final_is_ignored(mock_palabra_stt_ws):
    """Events arriving after the FINAL for the same transcription_id are dropped.

    The server can re-send tail updates and extra translation targets for a
    finalized id; they must not restart the utterance or duplicate the transcript.
    """
    ws = mock_palabra_stt_ws
    stt_ = STT(api_key="test-key")
    stream = stt_.stream()
    stream.push_frame(_frame())
    await _wait_for(lambda: len(ws["queries"]) == 1)

    ws["send_queue"].put_nowait(_transcription("Hello world.", is_eos=True))
    ws["send_queue"].put_nowait(_transcription("Hello world!!", is_eos=True))  # duplicate final
    ws["send_queue"].put_nowait(_transcription("tail", is_eos=False))  # late interim

    events = []
    with contextlib.suppress(asyncio.TimeoutError):
        while True:
            events.append(await asyncio.wait_for(stream.__anext__(), timeout=1.0))

    types = [ev.type for ev in events]
    assert types.count(agents_stt.SpeechEventType.START_OF_SPEECH) == 1
    assert types.count(agents_stt.SpeechEventType.FINAL_TRANSCRIPT) == 1
    assert types.count(agents_stt.SpeechEventType.END_OF_SPEECH) == 1

    stream.end_input()
    await stream.aclose()
    await stt_.aclose()
