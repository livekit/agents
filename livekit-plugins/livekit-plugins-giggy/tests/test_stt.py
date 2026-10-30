import asyncio
import io
import json
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.plugins import giggy

pytestmark = pytest.mark.unit
KEY = "offline-stt-key"
GENERATION = "00000000-0000-4000-8000-000000000001"
RESULT_URL = "https://00000000000000000000000000000000.r2.cloudflarestorage.com/transcript.json?signature=offline"


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "bad_ack",
        "bad_identity",
        "bad_attempt",
        "bad_artifact",
        "durable_pending",
        "cancel",
        "eof",
        "short_resume",
        "short_resume_eof",
    ],
)
async def test_native_streaming_websocket_contract(session, monkeypatch, case):
    from livekit.agents import vad
    from livekit.plugins.giggy import stt as implementation

    class Detection:
        def __init__(self):
            self.events, self.count = asyncio.Queue(), 0

        def push_frame(self, frame):
            self.count += 1
            resumed = case in {"short_resume", "short_resume_eof"}
            self.events.put_nowait(
                SimpleNamespace(
                    type=vad.VADEventType.INFERENCE_DONE,
                    frames=[frame],
                    raw_accumulated_silence=0 if self.count in (1, 3) else 0.032,
                )
            )
            if self.count == 1 or (resumed and self.count == 3):
                self.events.put_nowait(SimpleNamespace(type=vad.VADEventType.START_OF_SPEECH))
            elif case != "eof" and not (case == "short_resume_eof" and self.count == 4):
                self.events.put_nowait(
                    SimpleNamespace(
                        type=vad.VADEventType.END_OF_SPEECH,
                        silence_duration=0.1,
                        inference_duration=0,
                    )
                )

        def end_input(self):
            self.events.put_nowait(None)

        async def aclose(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            event = await self.events.get()
            if event is None:
                raise StopAsyncIteration
            return event

    class Detector:
        def stream(self):
            return Detection()

    received, controls = [], []
    disconnected = asyncio.Event()

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer " + KEY
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        start = await socket.receive_json()
        assert start["options"] == {"endpointing": "external"} and start["sampleRateHz"] == 16000
        await socket.send_json({"type": "generation_created", "generation_uuid": GENERATION})
        await socket.send_json(
            {"type": "ready", "generation_uuid": GENERATION, "attempt": 1, "sampleRateHz": 16000}
        )
        try:
            async for message in socket:
                if message.type == aiohttp.WSMsgType.BINARY:
                    received.append(message.data)
                    continue
                body = json.loads(message.data)
                controls.append(body)
                if body["type"] == "prepare":
                    assert (
                        body["decode_end_sample"] is None
                        or body["decode_end_sample"] <= sum(map(len, received)) // 2
                    )
                    continue
                assert body == {"type": "stop"}
                await socket.send_json(
                    {
                        "type": "input_complete",
                        "generation_uuid": GENERATION,
                        "sequence": len(received) + (1 if case == "bad_ack" else 0),
                    }
                )
                await socket.send_json(
                    {
                        "type": "transcription",
                        "generation_uuid": "00000000-0000-4000-8000-000000000099"
                        if case == "bad_identity"
                        else GENERATION,
                        "attempt": 2 if case == "bad_attempt" else 1,
                        "sequence": 0,
                        "replace": True,
                        "is_final": True,
                        "alternatives": [{"transcript": "Hello LiveKit."}],
                    }
                )
                await socket.send_json(
                    {
                        "type": "generation_updated",
                        "generation_uuid": GENERATION,
                        "status": "completed",
                    }
                )
                break
        finally:
            disconnected.set()
            await socket.close()
        return socket

    app = web.Application()
    app.router.add_get("/v1/audio/transcriptions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    monkeypatch.setattr(
        implementation, "GIGGY_ORIGIN", f"http://127.0.0.1:{runner.addresses[0][1]}"
    )
    metadata = Response(
        {
            "generation": {
                "generation_uuid": GENERATION,
                "status": "completed",
                "result": {"url": RESULT_URL},
            }
        },
        hold=case == "durable_pending",
    )
    session.get.side_effect = [
        metadata,
        Response(
            {
                "text": "wrong" if case == "bad_artifact" else "Hello LiveKit.",
                "audio_duration_ms": 600 if case.startswith("short_resume") else 520,
            }
        ),
    ]
    try:
        async with aiohttp.ClientSession() as client:
            monkeypatch.setattr(client, "get", session.get)
            provider = giggy.STT(api_key=KEY, vad=Detector(), http_session=client)
            stream = provider.stream(conn_options=APIConnectOptions(timeout=2, max_retry=5))
            stream.push_frame(audio(duration=0.2 if case.startswith("short_resume") else 0.5))
            if case == "cancel":
                async with asyncio.timeout(3):
                    while not received:
                        await asyncio.sleep(0.001)
                await stream.aclose()
                await asyncio.wait_for(disconnected.wait(), 3)
                assert not controls
            else:
                if case.startswith("short_resume"):
                    for duration in (0.1, 0.2, 0.1):
                        stream.push_frame(audio(duration=duration))
                else:
                    stream.push_frame(audio(duration=0.02))
                stream.end_input()
                events = []
                if case == "durable_pending":
                    async with asyncio.timeout(3):
                        async for event in stream:
                            events.append(event)
                            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                                break
                    await asyncio.wait_for(metadata.started.wait(), 3)
                    assert not metadata.closed
                elif case in {"bad_ack", "bad_identity", "bad_attempt", "bad_artifact"}:
                    with pytest.raises(APIConnectionError):
                        async with asyncio.timeout(3):
                            async for event in stream:
                                events.append(event)
                    assert any(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in events) is (
                        case == "bad_artifact"
                    )
                else:
                    async with asyncio.timeout(3):
                        events = [event async for event in stream]
                    assert [e.type for e in events] == [
                        stt.SpeechEventType.START_OF_SPEECH,
                        stt.SpeechEventType.END_OF_SPEECH,
                        *(
                            [stt.SpeechEventType.START_OF_SPEECH, stt.SpeechEventType.END_OF_SPEECH]
                            if case.startswith("short_resume")
                            else []
                        ),
                        stt.SpeechEventType.FINAL_TRANSCRIPT,
                        stt.SpeechEventType.RECOGNITION_USAGE,
                    ]
                durations = (0.2, 0.1, 0.2, 0.1) if case.startswith("short_resume") else (0.5, 0.02)
                assert b"".join(received) == b"".join(
                    bytes(audio(duration=d).data) for d in durations
                )
                assert any(c["type"] == "prepare" for c in controls)
                if case == "eof":
                    assert [c for c in controls if c["type"] == "prepare"][-1] == {
                        "type": "prepare",
                        "decode_end_sample": None,
                    }
                await stream.aclose()
            await provider.aclose()
            assert not client.closed
    finally:
        await runner.cleanup()


class Response:
    def __init__(self, body, *, status=200, hold=False, error=None):
        self.status = status
        self.headers = {"Content-Type": "application/json", "x-request-id": "stt-request"}
        self.body = json.dumps(body).encode()
        if isinstance(body, list):
            self.headers["Content-Type"] = "text/event-stream"
            self.body = b"".join(
                f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
                for event, payload in body
            )
        self.content = self
        self.hold = hold
        self.error = error
        self.started = asyncio.Event()
        self.closed = False

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def iter_chunked(self, size):
        self.started.set()
        if self.hold:
            await asyncio.Event().wait()
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


def transcript_events(text="Hello LiveKit.", *, state="completed", url=RESULT_URL):
    return [
        ("generation_created", {"generation_uuid": GENERATION}),
        (
            "transcript_chunk",
            {
                "generation_uuid": GENERATION,
                "attempt": 1,
                "sequence": 0,
                "text": text,
                "replace": True,
                "complete": True,
            },
        ),
        (
            "generation_updated",
            {
                "generation_uuid": GENERATION,
                "status": state,
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": state,
                    "result": {"url": url},
                },
            },
        ),
    ]


@pytest.fixture
def session(monkeypatch):
    import livekit.plugins.giggy.stt as giggy_stt

    monkeypatch.setattr(giggy_stt, "uuid4", lambda: GENERATION)
    client = MagicMock(spec=aiohttp.ClientSession)
    client.auth = None
    client.headers = {}
    client.cookie_jar = []
    client.uploads = []
    client.post.return_value = Response(transcript_events())

    def capture_upload(*args, **kwargs):
        source = kwargs["data"]
        client.uploads.append(source.read())
        source.seek(0)
        return client.post.return_value

    client.post.side_effect = capture_upload
    client.get.side_effect = [
        Response(
            {
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": "completed",
                    "result": {"url": RESULT_URL},
                }
            }
        ),
        Response({"text": "Hello LiveKit.", "languages": ["English"], "audio_duration_ms": 500}),
    ]
    return client


def audio(rate=16000, channels=1, duration=0.5):
    return rtc.AudioFrame(
        data=b"\x01\x02" * int(rate * duration) * channels,
        sample_rate=rate,
        num_channels=channels,
        samples_per_channel=int(rate * duration),
    )


def test_stt_exports_and_credentials(monkeypatch):
    assert hasattr(giggy, "STT")
    monkeypatch.delenv("GIGGY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="key"):
        giggy.STT()
    monkeypatch.setenv("GIGGY_API_KEY", KEY)
    provider = giggy.STT()
    assert provider.provider == "Giggy"
    assert provider.model == "giggyspeech-stt"
    assert provider.capabilities.streaming is True
    assert provider.capabilities.interim_results is True


@pytest.mark.parametrize("language", [None, "en-US", "English", "yue", "fil"])
@pytest.mark.parametrize("rate", [16000, 48000])
async def test_stt_native_request(session, language, rate, monkeypatch):
    import livekit.plugins.giggy.stt as giggy_stt

    monkeypatch.setattr(giggy_stt, "uuid4", lambda: GENERATION)
    provider = giggy.STT(api_key=KEY, language=language, http_session=session)
    options = APIConnectOptions(max_retry=5, timeout=7)
    result = await provider.recognize(audio(rate), conn_options=options)
    assert result.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert result.request_id == GENERATION
    assert result.alternatives[0].text == "Hello LiveKit."
    assert result.alternatives[0].language == "en"
    assert result.alternatives[0].source_languages == ["en"]
    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args == ("https://api.giggy.ai/v1/audio/transcriptions",)
    assert kwargs["params"]["mode"] == "streaming"
    assert kwargs["params"]["client_request_id"] == GENERATION
    assert kwargs["headers"] == {
        "xi-api-key": KEY,
        "Content-Type": "audio/wav",
        "Accept": "text/event-stream",
    }
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"].sock_read == 7
    with wave.open(io.BytesIO(session.uploads[0]), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert wav.getnframes() == 8000
    expected_language = {
        "en-US": "English",
        "English": "English",
        "yue": "Cantonese",
        "fil": "Filipino",
    }
    assert kwargs["params"].get("language") == expected_language.get(language)
    assert not session.get.call_args.kwargs.get("headers")
    assert session.get.call_args.args == (RESULT_URL,)
    assert options.max_retry == 5
    await provider.aclose()
    session.close.assert_not_called()


@pytest.mark.parametrize("language", ["unknown", "xx", " "])
def test_stt_language_rejected(language):
    with pytest.raises(ValueError, match="language"):
        giggy.STT(api_key=KEY, language=language)


@pytest.mark.parametrize("kwargs", [{"duration": 0.49}, {"channels": 2}])
async def test_stt_invalid_audio(session, kwargs):
    provider = giggy.STT(api_key=KEY, http_session=session)
    with pytest.raises(ValueError):
        await provider.recognize(audio(**kwargs))
    session.post.assert_not_called()


@pytest.mark.parametrize("status", [301, 401, 402, 429, 503])
async def test_stt_admission_error_no_retry(session, status):
    session.post.return_value = Response({"error": KEY}, status=status)
    with pytest.raises(APIStatusError) as caught:
        await giggy.STT(api_key=KEY, http_session=session).recognize(
            audio(), conn_options=APIConnectOptions(max_retry=5, retry_interval=0)
        )
    assert caught.value.status_code == status
    assert caught.value.retryable is False
    assert caught.value.body is None
    assert KEY not in str(caught.value)
    session.post.assert_called_once()


@pytest.mark.parametrize(
    "body",
    [
        {"text": 1},
        {"text": "hello", "audio_duration_ms": 499},
        {"text": "hello", "audio_duration_ms": True},
        {"text": "hello", "audio_duration_ms": 500, "language": 123},
    ],
)
async def test_stt_invalid_artifact(session, body):
    session.get.side_effect = [
        Response(
            {
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": "completed",
                    "result": {"url": RESULT_URL},
                }
            }
        ),
        Response(body),
    ]
    with pytest.raises(APIConnectionError) as caught:
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    assert caught.value.retryable is False


@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example/a",
        "https://user:pass@example.com/a",
        "//example.com/a",
        "https://127.0.0.1/a",
        "https://169.254.169.254/a",
        "https://localhost/a",
        "https://internal.example/a",
        "https://00000000000000000000000000000000.r2.cloudflarestorage.com.evil.example/a",
        "https://00000000000000000000000000000000.r2.cloudflarestorage.com:8443/a",
    ],
)
async def test_stt_unsafe_artifact_url(session, url):
    session.post.return_value = Response(transcript_events(url=url))
    session.get.side_effect = [
        Response(
            {
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": "completed",
                    "result": {"url": url},
                }
            }
        )
    ]
    with pytest.raises(APIConnectionError):
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    assert session.get.call_count == 1


async def test_stt_snapshots_and_language_override(session, monkeypatch):
    import livekit.plugins.giggy.stt as giggy_stt

    monkeypatch.setattr(giggy_stt, "uuid4", lambda: GENERATION)
    events = transcript_events("")
    partial = dict(events[1][1], text="revised", complete=False, sequence=0)
    events[1][1]["sequence"] = 1
    events.insert(1, ("transcript_chunk", partial))
    session.post.return_value = Response(events)
    session.get.side_effect = [
        Response(
            {
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": "completed",
                    "result": {"url": RESULT_URL},
                }
            }
        ),
        Response({"text": "", "languages": [], "audio_duration_ms": 500}),
    ]
    provider = giggy.STT(api_key=KEY, language="en", http_session=session)
    event = await provider.recognize(audio(), language="French")
    assert session.post.call_args.kwargs["params"]["language"] == "French"
    assert event.alternatives[0].text == ""
    assert event.alternatives[0].language == "fr"
    assert session.get.call_count == 2


@pytest.mark.parametrize("state", ["failed", "cancelled", "admission_failed", "unknown"])
async def test_stt_terminal_failure(session, state):
    session.post.return_value = Response(transcript_events(state=state))
    with pytest.raises(APIConnectionError):
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    session.get.assert_not_called()


async def test_stt_shutdown_cancels_operation(session, monkeypatch):
    import livekit.plugins.giggy.stt as giggy_stt

    monkeypatch.setattr(giggy_stt, "uuid4", lambda: GENERATION)
    held = Response(transcript_events(), hold=True)
    session.post.return_value = held
    provider = giggy.STT(api_key=KEY, http_session=session)
    task = asyncio.create_task(provider.recognize(audio()))
    await asyncio.wait_for(held.started.wait(), 2)
    await provider.aclose()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert held.closed
    session.post.assert_called_once()
    session.get.assert_not_called()
    await provider.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await provider.recognize(audio())
    session.close.assert_not_called()


async def test_stt_read_timeout_no_retry(session, monkeypatch):
    import livekit.plugins.giggy.stt as giggy_stt

    monkeypatch.setattr(giggy_stt, "uuid4", lambda: GENERATION)
    session.post.return_value = Response({}, error=asyncio.TimeoutError(KEY))
    with pytest.raises(APITimeoutError) as caught:
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    assert caught.value.retryable is False
    assert caught.value.__cause__ is None
    assert KEY not in str(caught.value)


@pytest.mark.parametrize("header", ["Authorization", "xi-api-key", "Cookie"])
async def test_stt_session_credentials_rejected(session, header):
    session.headers = {header: KEY}
    with pytest.raises(ValueError, match="credential"):
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    session.post.assert_not_called()


@pytest.mark.parametrize("case", ["success", "headers_timeout", "body_timeout", "wrong_attempt"])
async def test_stt_real_http(session, monkeypatch, case):
    import livekit.plugins.giggy.stt as giggy_stt

    requests = []

    async def handler(request):
        requests.append(
            {
                "headers": dict(request.headers),
                "query": dict(request.query),
                "audio": await request.read(),
            }
        )
        if case == "headers_timeout":
            await asyncio.sleep(0.2)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if case == "body_timeout":
            await asyncio.sleep(0.2)
            return response
        if request.transport is None or request.transport.is_closing():
            return response
        events = transcript_events()
        if case == "wrong_attempt":
            partial = dict(events[1][1], complete=False, sequence=0)
            events[1][1].update(attempt=2, sequence=1)
            events.insert(1, ("transcript_chunk", partial))
        wire = Response(events).body.replace(b"\n", b"\r\n")
        for index in range(0, len(wire), 17):
            await response.write(wire[index : index + 17])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        monkeypatch.setattr(giggy_stt, "GIGGY_ORIGIN", f"http://127.0.0.1:{port}")
        async with aiohttp.ClientSession() as client:
            # Only signed-artifact/status reads are mocked; admission, framing,
            # connection cleanup and read timeouts exercise actual aiohttp.
            monkeypatch.setattr(client, "get", session.get)
            provider = giggy.STT(api_key=KEY, http_session=client)
            options = APIConnectOptions(timeout=0.05 if "timeout" in case else 2, max_retry=5)
            if case == "success":
                event = await provider.recognize(audio(), conn_options=options)
                assert event.alternatives[0].text == "Hello LiveKit."
            else:
                expected = APITimeoutError if "timeout" in case else APIConnectionError
                with pytest.raises(expected) as caught:
                    await provider.recognize(audio(), conn_options=options)
                assert caught.value.retryable is False
                assert caught.value.__cause__ is None
            await provider.aclose()
            assert not client.closed
        assert len(requests) == 1
        assert requests[0]["headers"]["xi-api-key"] == KEY
        assert requests[0]["query"]["mode"] == "streaming"
        with wave.open(io.BytesIO(requests[0]["audio"]), "rb") as wav:
            assert wav.getnframes() == 8000
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "change", ["identity", "sequence", "missing_final", "missing_status", "truncated", "oversized"]
)
async def test_stt_rejects_incomplete_or_invalid_stream(session, change):
    events = transcript_events()
    if change == "identity":
        events[1][1]["generation_uuid"] = "another-generation"
    elif change == "sequence":
        events[1][1]["sequence"] = True
    elif change == "missing_final":
        events[1][1]["complete"] = False
    elif change == "missing_status":
        events.pop()
    response = Response(events)
    if change == "truncated":
        response.body = response.body[:-1]
    elif change == "oversized":
        response.body = b"data:" + b"x" * (1024 * 1024)
    session.post.return_value = response
    with pytest.raises(APIConnectionError) as caught:
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    assert caught.value.retryable is False
    assert response.closed
    session.post.assert_called_once()
    session.get.assert_not_called()


async def test_stt_overall_deadline_closes_stream(session):
    response = Response(transcript_events(), hold=True)
    session.post.return_value = response
    provider = giggy.STT(api_key=KEY, http_session=session, recognition_timeout=0.05)
    with pytest.raises(APITimeoutError) as caught:
        await provider.recognize(audio())
    assert not caught.value.retryable
    assert response.closed
    assert not provider._requests
    session.post.assert_called_once()


async def test_stt_cookie_session_rejected(session):
    session.cookie_jar = ["private-cookie"]
    with pytest.raises(ValueError, match="credential"):
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    session.post.assert_not_called()


async def test_stt_long_recording_without_merging(session, monkeypatch):
    frames = [audio()] * 61  # 30.5 seconds without allocating a long recording.
    merger = MagicMock(side_effect=AssertionError("Recordings must not be merged in memory"))
    monkeypatch.setattr(rtc, "combine_audio_frames", merger)
    session.get.side_effect = [
        Response(
            {
                "generation": {
                    "generation_uuid": GENERATION,
                    "status": "completed",
                    "result": {"url": RESULT_URL},
                }
            }
        ),
        Response({"text": "Hello LiveKit.", "languages": ["English"], "audio_duration_ms": 30500}),
    ]
    result = await giggy.STT(api_key=KEY, http_session=session).recognize(frames)
    assert result.alternatives[0].end_time == 30.5
    with wave.open(io.BytesIO(session.uploads[0]), "rb") as wav:
        assert wav.getnframes() == 488000
    merger.assert_not_called()


async def test_stt_recording_api_limit_checked_before_encoding(session, monkeypatch):
    frames = [audio()] * 86401  # 12 hours plus 0.5 seconds, using shared frame references.
    provider = giggy.STT(api_key=KEY, http_session=session)
    encoder = MagicMock()
    monkeypatch.setattr(provider, "_encode_wav", encoder, raising=False)
    with pytest.raises(ValueError, match="12 hours"):
        await provider.recognize(frames)
    encoder.assert_not_called()
    session.post.assert_not_called()
