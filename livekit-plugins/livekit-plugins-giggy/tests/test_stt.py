import asyncio
import io
import json
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from yarl import URL

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
        "short_resume_delayed",
        "short_end_quiet",
        "short_end_quiet_next",
    ],
)
async def test_native_streaming_websocket_contract(session, monkeypatch, case):
    from livekit.agents import vad

    class Detection:
        def __init__(self):
            self.events, self.count = asyncio.Queue(), 0

        def push_frame(self, frame):
            self.count += 1
            if case == "short_end_quiet" and self.count == 18:
                return  # A real partial window retained for clean EOF.
            if case == "short_resume_delayed" or case.startswith("short_end_quiet"):
                self.events.put_nowait(
                    SimpleNamespace(
                        type=vad.VADEventType.INFERENCE_DONE,
                        frames=[frame],
                        raw_accumulated_silence=0 if self.count <= 4 else 0.032,
                    )
                )
                if self.count == 1 or (case == "short_resume_delayed" and self.count == 17):
                    self.events.put_nowait(SimpleNamespace(type=vad.VADEventType.START_OF_SPEECH))
                elif self.count == 4 or (case == "short_resume_delayed" and self.count == 20):
                    self.events.put_nowait(
                        SimpleNamespace(
                            type=vad.VADEventType.END_OF_SPEECH,
                            silence_duration=0.1,
                            inference_duration=0,
                        )
                    )
                return
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
    first_audio, disconnected = asyncio.Event(), asyncio.Event()

    async def handler(headers, socket):
        assert headers["Authorization"] == "Bearer " + KEY
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
                    first_audio.set()
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

    servers = fake_websocket(session, handler)
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
                "audio_duration_ms": 640
                if case == "short_resume_delayed"
                else 554
                if case == "short_end_quiet"
                else 512
                if case == "short_end_quiet_next"
                else 600
                if case.startswith("short_resume")
                else 520,
            }
        ),
    ]
    provider = giggy.STT(api_key=KEY, vad=Detector(), http_session=session)
    stream = provider.stream(conn_options=APIConnectOptions(timeout=2, max_retry=5))
    durations = (
        (0.032,)
        * (20 if case == "short_resume_delayed" else 18 if case == "short_end_quiet_next" else 17)
        if (case == "short_resume_delayed" or case.startswith("short_end_quiet"))
        else (0.2, 0.1, 0.2, 0.1)
        if case.startswith("short_resume")
        else (0.5, 0.02)
    )
    if case == "short_end_quiet":
        durations += (0.01,)
    stream.push_frame(audio(duration=durations[0]))
    if case == "cancel":
        await asyncio.wait_for(first_audio.wait(), 3)
        await stream.aclose()
        await asyncio.wait_for(disconnected.wait(), 3)
        assert not controls
    else:
        for duration in durations[1:]:
            stream.push_frame(audio(duration=duration))
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
        if case == "short_end_quiet_next":
            durations = durations[:16]
        assert b"".join(received) == b"".join(bytes(audio(duration=d).data) for d in durations)
        assert any(c["type"] == "prepare" for c in controls)
        if case == "eof":
            assert [c for c in controls if c["type"] == "prepare"][-1] == {
                "type": "prepare",
                "decode_end_sample": None,
            }
        await stream.aclose()
    await provider.aclose()
    session.close.assert_not_called()
    await asyncio.wait_for(asyncio.gather(*servers), 3)


class FakeSocket:
    """One end of an in-memory WebSocket pair, standing in for aiohttp's client and server."""

    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue):
        self._inbox, self._outbox = inbox, outbox
        self.closed = False

    async def send_json(self, data):
        self._outbox.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(data)))

    async def send_bytes(self, data):
        self._outbox.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=bytes(data)))

    async def receive_json(self):
        return json.loads((await self._inbox.get()).data)

    async def close(self):
        if not self.closed:
            self.closed = True
            self._outbox.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._inbox.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def fake_websocket(session, handler):
    """Route ``session.ws_connect`` to ``handler(headers, server_socket)`` with no network."""
    servers: list[asyncio.Task] = []

    async def ws_connect(url, *, headers, **kwargs):
        assert url.endswith("/v1/audio/transcriptions")
        to_server, to_client = asyncio.Queue(), asyncio.Queue()
        servers.append(asyncio.create_task(handler(headers, FakeSocket(to_server, to_client))))
        return FakeSocket(to_client, to_server)

    session.ws_connect = ws_connect
    return servers


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
    client.cookie_jar = MagicMock()
    client.cookie_jar.filter_cookies.return_value = {}
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


@pytest.mark.parametrize("cookie_host", ["https://api.giggy.ai/", "https://other.example/"])
async def test_stt_cookie_session_rejected(session, cookie_host):
    """Only cookies that would be sent to Giggy block a shared session."""
    session.cookie_jar = aiohttp.CookieJar()
    session.cookie_jar.update_cookies({"sid": "caller-private"}, URL(cookie_host))
    provider = giggy.STT(api_key=KEY, http_session=session)
    if "giggy" in cookie_host:
        with pytest.raises(ValueError, match="credential"):
            await provider.recognize(audio())
        session.post.assert_not_called()
    else:
        event = await provider.recognize(audio())
        assert event.alternatives[0].text == "Hello LiveKit."


async def test_stt_artifact_cookie_rejected_before_download(session):
    """A cookie scoped to the signed result host must never be sent with the download."""
    session.cookie_jar = aiohttp.CookieJar()
    session.cookie_jar.update_cookies({"sid": "caller-private"}, URL(RESULT_URL))
    with pytest.raises(APIConnectionError):
        await giggy.STT(api_key=KEY, http_session=session).recognize(audio())
    session.post.assert_called_once()
    assert session.get.call_count == 1  # status lookup only; the artifact is never fetched


async def test_stream_drops_short_turn_and_keeps_listening(session):
    """Speech under 0.5 s is dropped without ending the stream, and later speech is still heard."""
    from livekit.agents import vad

    class Detection:
        """Report START on the first frame, every frame as speech, and END at input end."""

        def __init__(self):
            self.events, self.started = asyncio.Queue(), False

        def push_frame(self, frame):
            self.events.put_nowait(
                SimpleNamespace(
                    type=vad.VADEventType.INFERENCE_DONE, frames=[frame], raw_accumulated_silence=0
                )
            )
            if not self.started:
                self.started = True
                self.events.put_nowait(SimpleNamespace(type=vad.VADEventType.START_OF_SPEECH))

        def end_input(self):
            self.events.put_nowait(
                SimpleNamespace(
                    type=vad.VADEventType.END_OF_SPEECH, silence_duration=0.1, inference_duration=0
                )
            )
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

    async def handler(headers, socket):
        await socket.receive_json()
        await socket.send_json({"type": "generation_created", "generation_uuid": GENERATION})
        await socket.send_json(
            {"type": "ready", "generation_uuid": GENERATION, "attempt": 1, "sampleRateHz": 16000}
        )
        frames = 0
        async for message in socket:
            if message.type == aiohttp.WSMsgType.BINARY:
                frames += 1
                continue
            if json.loads(message.data)["type"] != "stop":
                continue
            for payload in (
                {"type": "input_complete", "sequence": frames},
                {
                    "type": "transcription",
                    "attempt": 1,
                    "sequence": 0,
                    "replace": True,
                    "is_final": True,
                    "alternatives": [{"transcript": "Hello LiveKit."}],
                },
                {"type": "generation_updated", "status": "completed"},
            ):
                await socket.send_json({"generation_uuid": GENERATION, **payload})
            break
        await socket.close()

    servers = fake_websocket(session, handler)
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
        Response({"text": "Hello LiveKit.", "audio_duration_ms": 600}),
    ]
    provider = giggy.STT(api_key=KEY, vad=Detector(), http_session=session)
    stream = provider.stream()
    stream.push_frame(audio(duration=0.2))  # a short blip, below Giggy's 0.5-second minimum
    stream.flush()
    stream.push_frame(audio(duration=0.5))
    stream.push_frame(audio(duration=0.1))
    stream.end_input()
    async with asyncio.timeout(3):
        events = [event async for event in stream]
    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    assert events[4].alternatives[0].text == "Hello LiveKit."
    assert stream._pending_bytes == 0
    await stream.aclose()
    await provider.aclose()
    await asyncio.wait_for(asyncio.gather(*servers), 3)


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
