import asyncio
import io
import json
import wave
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
    client.post.return_value = Response(transcript_events())
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
    assert provider.capabilities.streaming is False
    assert provider.capabilities.interim_results is False


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
    assert args == ("https://giggy.ai/v1/audio/transcriptions",)
    assert kwargs["params"]["mode"] == "streaming"
    assert kwargs["params"]["client_request_id"] == GENERATION
    assert kwargs["headers"] == {
        "xi-api-key": KEY,
        "Content-Type": "audio/wav",
        "Accept": "text/event-stream",
    }
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"].sock_read == 7
    with wave.open(io.BytesIO(kwargs["data"]), "rb") as wav:
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


async def test_stt_utterance_bound_before_merging(session, monkeypatch):
    frames = [audio()] * 61  # 30.5 seconds without allocating a long recording.
    merger = MagicMock()
    monkeypatch.setattr(rtc, "combine_audio_frames", merger)
    with pytest.raises(ValueError, match="30 seconds"):
        await giggy.STT(api_key=KEY, http_session=session).recognize(frames)
    merger.assert_not_called()
    session.post.assert_not_called()
