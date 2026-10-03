"""Offline HTTP and LiveKit audio lifecycle coverage for 60db."""

import asyncio
import base64
import io
import json
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIError, APIStatusError, APITimeoutError, tts, utils
from livekit.plugins import sixtydb
from livekit.plugins.sixtydb import tts as provider

pytestmark = pytest.mark.unit
PCM = b"\x01\x00" * 4800
OPTIONS = APIConnectOptions(max_retry=0, timeout=0.2)


def wav_bytes(rate: int = 24000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(PCM)
    return buffer.getvalue()


@asynccontextmanager
async def endpoint(handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_post("/tts-synthesize", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = runner.addresses[0][1]
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


async def collect(engine: sixtydb.TTS) -> bytes:
    async with engine.synthesize("Hello", conn_options=OPTIONS) as stream:
        events = [event async for event in stream]
    assert events[-1].is_final
    assert all(
        event.frame.sample_rate == 24000 and event.frame.num_channels == 1 for event in events
    )
    return b"".join(bytes(event.frame.data) for event in events)


@pytest.mark.parametrize("kind", ["wav", "pcm", "json", "ndjson", "envelope"])
async def test_http_audio_and_request_contract(kind: str):
    requests = []

    async def handler(request):
        requests.append(await request.json())
        assert request.headers["Authorization"] == "Bearer private-test-key"
        encoded = base64.b64encode(PCM).decode()
        if kind == "wav":
            return web.Response(body=wav_bytes(), content_type="audio/wav")
        if kind == "pcm":
            return web.Response(body=PCM, content_type="audio/pcm")
        if kind == "json":
            return web.json_response(
                {"success": True, "audio_base64": encoded, "sample_rate": 24000}
            )
        if kind == "envelope":
            encoded = base64.b64encode(json.dumps({"audioContent": encoded}).encode()).decode()
        body = json.dumps({"result": {"audioContent": encoded}}) + '\n{"type":"complete"}\n'
        return web.Response(text=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(
            voice_id="workspace-voice",
            api_key="private-test-key",
            http_session=session,
            base_url=url,
            speed=1.2,
        )
        assert await collect(engine) == PCM
        await engine.aclose()
        assert not session.closed
    assert requests == [
        {
            "text": "Hello",
            "voice_id": "workspace-voice",
            "speed": 1.2,
            "timestamp_type": "NONE",
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000},
        }
    ]


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
async def test_http_errors_do_not_expose_response_or_credentials(status: int):
    async def handler(request):
        return web.Response(status=status, text="private-test-key", headers={"Location": "/other"})

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(
            voice_id="voice", api_key="private-test-key", base_url=url, http_session=session
        )
        with pytest.raises(APIStatusError) as exc:
            await collect(engine)
        assert exc.value.status_code == status
        assert "private-test-key" not in str(exc.value)


@pytest.mark.parametrize(
    "body,content_type",
    [
        (b'{"success":false,"message":"private-test-key"}', "application/json"),
        (b'{"audio_base64":"bad!"}', "application/json"),
        (b'{"audio_base64":"AQ=="}', "application/json"),
        (b'{"audio_base64":"AQAAAg==","encoding":"mp3"}', "application/json"),
        (b'{"audio_base64":"AQAAAg==","sample_rate":16000}', "application/json"),
        (b'{"result":null}', "application/json"),
        (b'{"type":"complete"}', "application/x-ndjson"),
        (b"not-json", "application/json"),
        (b"ID3broken", "audio/pcm"),
        (b"RIFFbroken", "audio/wav"),
        (wav_bytes(rate=16000), "audio/wav"),
        (wav_bytes(channels=2), "audio/wav"),
        (b"<html>error</html>", "text/html"),
    ],
)
async def test_invalid_audio_is_not_emitted(body: bytes, content_type: str):
    async def handler(request):
        return web.Response(body=body, content_type=content_type)

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        with pytest.raises(APIError) as exc:
            await collect(engine)
        assert not exc.value.retryable
        assert "private-test-key" not in str(exc.value)


async def test_timeout():
    async def handler(request):
        await asyncio.sleep(0.4)
        return web.Response(body=PCM, content_type="audio/pcm")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        with pytest.raises(APITimeoutError):
            await collect(engine)


async def test_response_limit(monkeypatch):
    monkeypatch.setattr(provider, "_MAX_RESPONSE_BYTES", 100)

    async def handler(request):
        return web.Response(body=PCM, content_type="audio/pcm")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        with pytest.raises(APIError):
            await collect(engine)


async def test_snapshot_and_stream_adapter_separate_segments():
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.Response(body=PCM, content_type="audio/pcm")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="first", api_key="key", base_url=url, http_session=session)
        stream = engine.synthesize("Snapshot", conn_options=OPTIONS)
        engine.update_options(voice_id="second", speed=1.5)
        async with stream:
            assert [event async for event in stream]
        adapter = tts.StreamAdapter(tts=engine)
        for sentence in ("First sentence.", "Second sentence."):
            # LiveKit requires a fresh stream for each input segment.
            async with adapter.stream(conn_options=OPTIONS) as streaming:
                streaming.push_text(sentence)
                streaming.flush()
                streaming.end_input()
                events = [event async for event in streaming]
            assert events[-1].is_final
            # StreamAdapter flushes the tail before ending its segment, so the
            # framework emits a synthetic 10 ms final marker.
            assert bytes(events[-1].frame.data) == b"\0\0" * 240
            assert b"".join(bytes(event.frame.data) for event in events[:-1]) == PCM
        await adapter.aclose()
    assert requests[0]["voice_id"] == "first"
    assert requests[0]["speed"] == 1.0
    assert len(requests) == 3
    assert all(
        request["voice_id"] == "second" and request["speed"] == 1.5 for request in requests[1:]
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "ftp://example.com",
        "https://user:pass@example.com",
        "https://example.com?key=value",
        "https://example.com#fragment",
    ],
)
def test_unsafe_urls_rejected(url: str):
    with pytest.raises(ValueError):
        sixtydb.TTS(voice_id="voice", api_key="key", base_url=url)


@pytest.mark.parametrize("speed", [0.4, 2.1, float("nan"), float("inf"), True])
def test_invalid_speed(speed: float):
    with pytest.raises(ValueError):
        sixtydb.TTS(voice_id="voice", api_key="key", speed=speed)


def test_credentials_options_and_text(monkeypatch):
    monkeypatch.delenv("SIXTYDB_API_KEY", raising=False)
    with pytest.raises(ValueError):
        sixtydb.TTS(voice_id="voice")
    monkeypatch.setenv("SIXTYDB_API_KEY", "environment-key")
    engine = sixtydb.TTS(voice_id="voice")
    assert engine._api_key == "environment-key"
    assert not engine.capabilities.streaming
    assert engine.provider == "60db"
    for text in ("", "  "):
        with pytest.raises(ValueError):
            engine.synthesize(text)
    with pytest.raises(ValueError):
        engine.update_options(voice_id="changed", speed=3)
    assert engine._opts.voice_id == "voice"


@pytest.mark.parametrize(
    "header,value",
    [
        ("X-Sample-Rate", "16000"),
        ("X-Channels", "2"),
        ("X-Bit-Depth", "8"),
        ("X-Sample-Rate", "invalid"),
    ],
)
async def test_incompatible_response_headers(header, value):
    async def handler(request):
        return web.Response(body=PCM, content_type="audio/pcm", headers={header: value})

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        with pytest.raises(APIError):
            await collect(engine)


async def test_late_ndjson_error_emits_no_audio():
    async def handler(request):
        body = json.dumps({"result": {"audioContent": base64.b64encode(PCM).decode()}})
        body += '\n{"type":"error","message":"private-test-key"}\n'
        return web.Response(text=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        events = []
        with pytest.raises(APIError):
            async with engine.synthesize("Hello", conn_options=OPTIONS) as stream:
                async for event in stream:
                    events.append(event)
        assert events == []


async def test_cancellation_closes_pending_response():
    started = asyncio.Event()
    disconnected = asyncio.Event()

    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(PCM)
        started.set()
        for _ in range(100):
            if request.transport is None or request.transport.is_closing():
                disconnected.set()
                break
            await asyncio.sleep(0.01)
        return response

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        stream = engine.synthesize("Hello", conn_options=APIConnectOptions(max_retry=0, timeout=2))
        await asyncio.wait_for(started.wait(), 2)
        await stream.aclose()
        await asyncio.wait_for(disconnected.wait(), 2)
        assert not session.closed


@pytest.mark.parametrize("text", ["a" * 5001, ("word " * 2200) + "end", "a" * 5000])
async def test_long_text_preserves_input_and_audio_order(text):
    requests = []
    expected_audio = bytearray()

    async def handler(request):
        piece = (await request.json())["text"]
        requests.append(piece)
        audio = bytes([len(requests), 0]) * 480
        expected_audio.extend(audio)
        return web.Response(body=audio, content_type="audio/pcm")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        async with engine.synthesize(text, conn_options=OPTIONS) as stream:
            events = [event async for event in stream]
        await engine.aclose()
    assert "".join(requests) == text
    assert all(0 < len(piece) <= 5000 for piece in requests)
    assert len(requests) == (1 if len(text) <= 5000 else (3 if len(text) > 10000 else 2))
    assert b"".join(bytes(event.frame.data) for event in events) == bytes(expected_audio)
    assert events[-1].is_final


async def test_managed_session_reuse_across_contexts():
    async def handler(request):
        return web.Response(body=PCM, content_type="audio/pcm")

    async with endpoint(handler) as url:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url)
        async with utils.http_context.open():
            first = engine._ensure_session()
            assert await collect(engine) == PCM
        assert first.closed
        async with utils.http_context.open():
            second = engine._ensure_session()
            assert second is not first and not second.closed
            assert await collect(engine) == PCM
        assert second.closed
        await engine.aclose()


async def test_later_piece_error_emits_no_partial_audio():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            return web.Response(status=500)
        return web.Response(body=PCM, content_type="audio/pcm")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        events = []
        with pytest.raises(APIStatusError):
            async with engine.synthesize("a" * 5001, conn_options=OPTIONS) as stream:
                async for event in stream:
                    events.append(event)
        assert not events and calls == 2
        await engine.aclose()


@pytest.mark.parametrize("text", ["a" * 5000 + " ", "a" * 5000 + "\n\t", " " * 6000 + "spoken"])
async def test_splitter_never_sends_whitespace_only_request(text):
    requests = []

    async def handler(request):
        piece = (await request.json())["text"]
        requests.append(piece)
        if not piece.strip():
            return web.Response(status=400)
        return web.Response(body=PCM, content_type="audio/pcm")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        async with engine.synthesize(text, conn_options=OPTIONS) as stream:
            events = [event async for event in stream]
        await engine.aclose()
    assert requests and all(piece.strip() and len(piece) <= 5000 for piece in requests)
    assert "".join(requests).strip() == text.strip()
    assert b"".join(bytes(event.frame.data) for event in events) == PCM * len(requests)


async def test_ndjson_wav_chunks_preserve_all_audio():
    body = (
        b"\n".join(
            json.dumps({"audioContent": base64.b64encode(wav_bytes()).decode()}).encode()
            for _ in range(2)
        )
        + b'\n{"type":"complete"}\n'
    )

    async def handler(request):
        return web.Response(body=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        assert await collect(engine) == PCM * 2
        await engine.aclose()


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("pcm", [PCM, b"RIFF" + PCM], ids=["ordinary", "riff_samples"])
async def test_ndjson_split_wav_preserves_audio(pcm, split):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    wav_payload = buffer.getvalue()
    chunks = [wav_payload[:13], wav_payload[13:45], wav_payload[45:]] if split else [wav_payload]
    body = b"\n".join(
        json.dumps({"audioContent": base64.b64encode(chunk).decode()}).encode() for chunk in chunks
    )

    async def handler(request):
        return web.Response(body=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        assert await collect(engine) == pcm
        await engine.aclose()


@pytest.mark.parametrize(
    "chunks,expected",
    [
        ([wav_bytes(), PCM], PCM * 2),
        ([PCM, wav_bytes()], PCM * 2),
        ([b"\x01\x00", b"RIFF\x01\x00\x02\x00"], b"\x01\x00RIFF\x01\x00\x02\x00"),
    ],
    ids=["wav_first", "pcm_first", "riff_pcm"],
)
async def test_ndjson_mixed_audio_records(chunks, expected):
    body = b"\n".join(
        json.dumps({"audioContent": base64.b64encode(chunk).decode()}).encode() for chunk in chunks
    )

    async def handler(request):
        return web.Response(body=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        assert await collect(engine) == expected
        await engine.aclose()


@pytest.mark.parametrize("content_type", ["application/x-ndjson", "application/json"])
@pytest.mark.parametrize("encoding,signature", [("wav", b"NOPE"), ("pcm", b"WAVE")])
async def test_ndjson_declared_format_controls_decoding(encoding, signature, content_type):
    payload = b"RIFF\x08\x00\x00\x00" + signature + b"abcd"
    body = json.dumps(
        {"encoding": encoding, "audioContent": base64.b64encode(payload).decode()}
    ).encode()

    async def handler(request):
        return web.Response(body=body, content_type=content_type)

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        if encoding == "wav":
            with pytest.raises(APIError):
                await collect(engine)
        else:
            assert await collect(engine) == payload
        await engine.aclose()


async def test_labeled_pcm_followed_by_unlabeled_wav():
    body = b"\n".join(
        [
            json.dumps(
                {"encoding": "pcm", "audioContent": base64.b64encode(PCM).decode()}
            ).encode(),
            json.dumps({"audioContent": base64.b64encode(wav_bytes()).decode()}).encode(),
        ]
    )

    async def handler(request):
        return web.Response(body=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        assert await collect(engine) == PCM * 2
        await engine.aclose()


@pytest.mark.parametrize(
    "tail",
    [
        b"ID3\x00",
        b"RIFF\x00\x00\x00\x00WAVE",
        b"RIFF\x04\x00\x00\x00WAVE",
        b"RIFF\xff\xff\xff\xffWAVE",
    ],
)
async def test_unlabeled_pcm_continuation_preserves_signature_samples(tail):
    body = b"\n".join(
        [
            json.dumps(
                {"encoding": "pcm", "audioContent": base64.b64encode(PCM).decode()}
            ).encode(),
            json.dumps({"audioContent": base64.b64encode(tail).decode()}).encode(),
        ]
    )

    async def handler(request):
        return web.Response(body=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        assert await collect(engine) == PCM + tail
        await engine.aclose()


@pytest.mark.parametrize("content_type", ["application/json", "application/x-ndjson"])
@pytest.mark.parametrize("nested", [False, True])
async def test_wav_container_precedes_linear16_sample_encoding(content_type, nested):
    audio = base64.b64encode(wav_bytes()).decode()
    if nested:
        audio = base64.b64encode(
            json.dumps({"audio_encoding": "LINEAR16", "audioContent": audio}).encode()
        ).decode()
    body = json.dumps(
        {
            "output_format": "wav",
            "audio_config": {"audio_encoding": "LINEAR16"},
            "audioContent": audio,
        }
    ).encode()

    async def handler(request):
        return web.Response(body=body, content_type=content_type)

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        assert await collect(engine) == PCM
        await engine.aclose()


@pytest.mark.parametrize("invalid_kind", ["rate", "container", "frames", "descriptor"])
async def test_incompatible_unlabeled_wav_after_pcm_is_rejected(invalid_kind):
    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setparams(
            (1, 2, 16000 if invalid_kind == "rate" else 24000, 0, "NONE", "not compressed")
        )
        output.writeframes(PCM)
    payload = wav.getvalue()
    if invalid_kind in {"container", "frames"}:
        payload = payload[:-2]
    if invalid_kind == "descriptor":
        payload = payload[:20]
    if invalid_kind == "frames":
        payload = payload[:4] + (len(payload) - 8).to_bytes(4, "little") + payload[8:]
    body = b"\n".join(
        [
            json.dumps(
                {"encoding": "pcm", "audioContent": base64.b64encode(PCM).decode()}
            ).encode(),
            json.dumps({"audioContent": base64.b64encode(payload).decode()}).encode(),
        ]
    )

    async def handler(request):
        return web.Response(body=body, content_type="application/x-ndjson")

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        with pytest.raises(APIError):
            await collect(engine)
        await engine.aclose()


@pytest.mark.parametrize("content_type", ["audio/pcm", "audio/wav", "audio/x-wav"])
async def test_binary_content_type_controls_decoding(content_type):
    payload = b"RIFF\x04\x00\x00\x00WAVE" if content_type == "audio/pcm" else PCM

    async def handler(request):
        return web.Response(body=payload, content_type=content_type)

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        if content_type == "audio/pcm":
            assert await collect(engine) == payload
        else:
            with pytest.raises(APIError):
                await collect(engine)
        await engine.aclose()


@pytest.mark.parametrize("content_type", ["audio/pcm", "application/octet-stream"])
@pytest.mark.parametrize(
    "payload", [b"ID3\x00\x00\x00\x00\x00", b"OggS\x00\x00\x00\x00", b"fLaC\x00\x00\x00\x00"]
)
async def test_binary_pcm_declaration_preserves_signature_samples(content_type, payload):
    async def handler(request):
        return web.Response(body=payload, content_type=content_type)

    async with endpoint(handler) as url, aiohttp.ClientSession() as session:
        engine = sixtydb.TTS(voice_id="voice", api_key="key", base_url=url, http_session=session)
        if content_type == "audio/pcm":
            assert await collect(engine) == payload
        else:
            with pytest.raises(APIError):
                await collect(engine)
        await engine.aclose()
