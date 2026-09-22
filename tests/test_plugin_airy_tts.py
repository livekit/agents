from __future__ import annotations

import asyncio
import struct
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError, tokenize, tts

pytestmark = pytest.mark.unit

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _pcm(num_samples: int) -> bytes:
    return struct.pack(
        f"<{num_samples}h", *(((index * 97) % 4000) - 2000 for index in range(num_samples))
    )


def _audio_headers(**overrides: str) -> dict[str, str]:
    headers = {
        "Content-Type": "audio/pcm",
        "X-Audio-Sample-Rate": "24000",
        "X-Audio-Channels": "1",
        "X-Audio-Sample-Format": "s16le",
        "X-Request-Id": "req_test",
    }
    headers.update(overrides)
    return headers


class _Server:
    def __init__(self, handler: Handler, *, raise_for_status: bool = False) -> None:
        self._handler = handler
        self._raise_for_status = raise_for_status

    async def __aenter__(self) -> _Server:
        app = web.Application()
        app.router.add_post("/v1/audio/speech/stream", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.session = aiohttp.ClientSession(raise_for_status=self._raise_for_status)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()


async def _collect(synth: Any, text: str = "hello", **options: Any) -> tuple[bytes, list[Any]]:
    connect_options = {"max_retry": 0, "timeout": 2.0, **options}
    conn_options = APIConnectOptions(**connect_options)
    stream = synth.synthesize(text, conn_options=conn_options)
    events = []
    try:
        async for event in stream:
            events.append(event)
    finally:
        await stream.aclose()
    return b"".join(event.frame.data.tobytes() for event in events), events


def test_requires_api_key() -> None:
    from livekit.plugins.airy import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Airy API key"):
            TTS(language="ko")


def test_explicit_api_key_takes_precedence() -> None:
    from livekit.plugins.airy import TTS

    with patch.dict("os.environ", {"AIRY_API_KEY": "env-key"}):
        synth = TTS(language="ko", api_key="argument-key")
    assert synth._opts.api_key == "argument-key"


def test_api_key_falls_back_to_environment() -> None:
    from livekit.plugins.airy import TTS

    with patch.dict("os.environ", {"AIRY_API_KEY": "env-key"}):
        synth = TTS(language="en")
    assert synth._opts.api_key == "env-key"


@pytest.mark.parametrize("api_key", ["", "   "])
def test_rejects_empty_explicit_api_key(api_key: str) -> None:
    from livekit.plugins.airy import TTS

    with patch.dict("os.environ", {"AIRY_API_KEY": "env-key"}):
        with pytest.raises(ValueError, match="Airy API key"):
            TTS(language="ko", api_key=api_key)


def test_language_is_required() -> None:
    from livekit.plugins.airy import TTS

    with pytest.raises(TypeError, match="language"):
        TTS(api_key="test-key")  # type: ignore[call-arg]


@pytest.mark.parametrize("language", ["", "ja", "KO"])
def test_rejects_invalid_language(language: str) -> None:
    from livekit.plugins.airy import TTS

    with pytest.raises(ValueError, match="language"):
        TTS(language=language, api_key="test-key")  # type: ignore[arg-type]


@pytest.mark.parametrize("style", ["", "happy", "NORMAL"])
def test_rejects_invalid_style(style: str) -> None:
    from livekit.plugins.airy import TTS

    with pytest.raises(ValueError, match="style"):
        TTS(language="ko", style=style, api_key="test-key")  # type: ignore[arg-type]


def test_capabilities_and_metadata() -> None:
    from livekit.plugins.airy import TTS

    synth = TTS(language="ko", api_key="test-key")
    assert synth.capabilities.streaming is False
    assert synth.capabilities.aligned_transcript is False
    assert synth.sample_rate == 24000
    assert synth.num_channels == 1
    assert synth.model == "airy-tts-v1"
    assert synth.provider == "Airy"


@pytest.mark.parametrize("text", ["", " \t\n", "a" * 1281])
def test_rejects_invalid_text_without_starting_request(text: str) -> None:
    from livekit.plugins.airy import TTS

    synth = TTS(language="ko", api_key="test-key")
    with pytest.raises(ValueError, match="text"):
        synth.synthesize(text)


def test_rejects_versioned_base_url() -> None:
    from livekit.plugins.airy import TTS

    with pytest.raises(ValueError, match="API root"):
        TTS(language="ko", api_key="test-key", base_url="https://api.airy.so/v1/")


async def test_request_contract_and_trailing_slash() -> None:
    from livekit.plugins.airy import TTS

    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["path"] = request.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = await request.json()
        return web.Response(body=_pcm(1200), headers=_audio_headers())

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            model="airy-tts-v1-custom",
            voice="voice-test",
            style="calm",
            api_key="test-key",
            base_url=server.base_url + "/",
            http_session=server.session,
        )
        audio, _ = await _collect(synth, "입력 내용 그대로")

    assert audio == _pcm(1200)
    assert captured == {
        "path": "/v1/audio/speech/stream",
        "authorization": "Bearer test-key",
        "content_type": "application/json",
        "body": {
            "input": "입력 내용 그대로",
            "language": "ko",
            "model": "airy-tts-v1-custom",
            "voice": "voice-test",
            "style": "calm",
        },
    }


@pytest.mark.parametrize("text", ["가", "가" * 1280])
async def test_accepts_unicode_text_boundaries(text: str) -> None:
    from livekit.plugins.airy import TTS

    received: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        received.append((await request.json())["input"])
        return web.Response(body=_pcm(480), headers=_audio_headers())

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        await _collect(synth, text)

    assert received == [text]


async def test_preserves_pcm_across_odd_http_chunks() -> None:
    from livekit.plugins.airy import TTS

    expected = _pcm(2500)

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers=_audio_headers())
        await response.prepare(request)
        offsets = (1, 4, 11, 512, 1801, len(expected))
        start = 0
        for end in offsets:
            await response.write(expected[start:end])
            start = end
        await response.write_eof()
        return response

    async with _Server(handler) as server:
        synth = TTS(
            language="en",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        audio, events = await _collect(synth)

    assert audio == expected
    assert events
    assert all(event.frame.sample_rate == 24000 for event in events)
    assert all(event.frame.num_channels == 1 for event in events)
    assert all(event.request_id == "req_test" for event in events)


async def test_accepts_normalized_content_type_and_missing_optional_headers() -> None:
    from livekit.plugins.airy import TTS

    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            body=_pcm(480),
            headers={"Content-Type": "Audio/PCM; format=s16le"},
        )

    async with _Server(handler) as server:
        synth = TTS(
            language="en",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        audio, events = await _collect(synth)

    assert audio == _pcm(480)
    assert events[0].request_id
    assert events[0].request_id != "req_test"


async def test_emits_first_frame_before_response_eof() -> None:
    from livekit.plugins.airy import TTS

    release = asyncio.Event()
    response_finished = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers=_audio_headers())
        await response.prepare(request)
        await response.write(_pcm(1200))
        await release.wait()
        await response.write(_pcm(480))
        await response.write_eof()
        response_finished.set()
        return response

    async with _Server(handler) as server:
        synth = TTS(
            language="en",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        stream = synth.synthesize("hello", conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            first = await asyncio.wait_for(anext(stream), timeout=1)
            assert first.frame.samples_per_channel > 0
            assert not response_finished.is_set()
            release.set()
            remaining = [event async for event in stream]
            assert remaining
        finally:
            release.set()
            await stream.aclose()


@pytest.mark.parametrize(
    ("headers", "field"),
    [
        ({"Content-Type": "application/json"}, "Content-Type"),
        ({"X-Audio-Sample-Rate": "16000"}, "X-Audio-Sample-Rate"),
        ({"X-Audio-Channels": "2"}, "X-Audio-Channels"),
        ({"X-Audio-Sample-Format": "f32le"}, "X-Audio-Sample-Format"),
    ],
)
async def test_rejects_unexpected_audio_format(headers: dict[str, str], field: str) -> None:
    from livekit.plugins.airy import TTS

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=_pcm(480), headers=_audio_headers(**headers))

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        with pytest.raises(APIStatusError) as raised:
            await _collect(synth)

    assert raised.value.status_code == 502
    assert raised.value.retryable is False
    assert raised.value.body == {"field": field, "received": next(iter(headers.values()))}


@pytest.mark.parametrize("body", [b"", b"\x01\x02\x03"])
async def test_rejects_empty_or_incomplete_pcm(body: bytes) -> None:
    from livekit.plugins.airy import TTS

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=body, headers=_audio_headers())

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        with pytest.raises(APIStatusError) as raised:
            await _collect(synth)

    assert raised.value.status_code == 502
    if body:
        assert raised.value.retryable is False


@pytest.mark.parametrize(
    ("status", "expected_requests"),
    [(400, 1), (401, 1), (402, 1), (403, 1), (404, 1), (429, 3), (500, 3)],
)
@pytest.mark.parametrize("raise_for_status", [False, True])
async def test_http_error_retry_policy(
    status: int, expected_requests: int, raise_for_status: bool
) -> None:
    from livekit.plugins.airy import TTS

    request_count = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal request_count
        request_count += 1
        return web.json_response(
            {
                "error": {
                    "type": "invalid_request_error",
                    "code": "test_error",
                    "message": "must not be retained",
                    "param": "input",
                },
                "request_id": "req_error",
            },
            status=status,
            headers={"Retry-After": "0.25"},
        )

    async with _Server(handler, raise_for_status=raise_for_status) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        stream = synth.synthesize(
            "hello",
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0, timeout=2),
        )
        try:
            with pytest.raises(APIStatusError) as raised:
                async for _ in stream:
                    pass
        finally:
            await stream.aclose()

    assert request_count == expected_requests
    assert raised.value.status_code == status
    assert raised.value.request_id == "req_error"
    assert "must not be retained" not in str(raised.value)
    assert "must not be retained" not in repr(raised.value)
    assert isinstance(raised.value.body, dict)
    assert raised.value.body["error_code"] == "test_error"
    if status == 429:
        assert raised.value.body["retry_after"] == "0.25"


@pytest.mark.parametrize(
    ("status", "expected_requests"),
    [(400, 1), (401, 1), (402, 1), (403, 1), (404, 1), (429, 3), (500, 3)],
)
@pytest.mark.parametrize("body_failure", ["timeout", "disconnect"])
async def test_http_status_survives_error_body_failure(
    status: int, expected_requests: int, body_failure: str
) -> None:
    from livekit.plugins.airy import TTS

    release = asyncio.Event()
    request_count = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal request_count
        request_count += 1
        response = web.StreamResponse(
            status=status,
            headers={
                "Content-Type": "application/json",
                "Content-Length": "1000",
                "X-Request-Id": "req_body_failure",
                "Retry-After": "0.25",
            },
        )
        await response.prepare(request)
        await response.write(b'{"error":')
        if body_failure == "disconnect":
            assert request.transport is not None
            request.transport.close()
        else:
            await release.wait()
        return response

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        try:
            with pytest.raises(APIStatusError) as raised:
                await _collect(synth, max_retry=2, retry_interval=0, timeout=0.05)
        finally:
            release.set()

    assert request_count == expected_requests
    assert raised.value.status_code == status
    assert raised.value.request_id == "req_body_failure"
    assert raised.value.retryable is (expected_requests > 1)
    assert isinstance(raised.value.body, dict)
    assert raised.value.body["status_code"] == status
    if status == 429:
        assert raised.value.body["retry_after"] == "0.25"


async def test_non_json_error_does_not_expose_response_body() -> None:
    from livekit.plugins.airy import TTS

    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=500, text="private upstream response")

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        with pytest.raises(APIStatusError) as raised:
            await _collect(synth)

    assert "private upstream response" not in str(raised.value)
    assert "private upstream response" not in repr(raised.value)


async def test_read_timeout_is_mapped() -> None:
    from livekit.plugins.airy import TTS

    release = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers=_audio_headers())
        await response.prepare(request)
        await release.wait()
        return response

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        try:
            with pytest.raises(APITimeoutError) as raised:
                await _collect(synth, timeout=0.05)
        finally:
            release.set()

    assert raised.value.retryable is True


async def test_connection_pool_wait_times_out() -> None:
    from livekit.plugins.airy import TTS

    release = asyncio.Event()
    request_count = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal request_count
        request_count += 1
        response = web.StreamResponse(headers=_audio_headers())
        await response.prepare(request)
        await release.wait()
        return response

    async with (
        _Server(handler) as server,
        aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session,
    ):
        try:
            # Hold the only connection while synthesis waits for a slot.
            async with session.post(f"{server.base_url}/v1/audio/speech/stream"):
                synth = TTS(
                    language="ko",
                    api_key="test-key",
                    base_url=server.base_url,
                    http_session=session,
                )
                with pytest.raises(APITimeoutError) as raised:
                    await asyncio.wait_for(_collect(synth, timeout=0.05), timeout=2)
        finally:
            release.set()

        assert not session.closed

    assert raised.value.retryable is True
    assert request_count == 1


async def test_timeout_after_audio_is_not_retried() -> None:
    from livekit.plugins.airy import TTS

    release = asyncio.Event()
    request_count = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal request_count
        request_count += 1
        response = web.StreamResponse(headers=_audio_headers())
        await response.prepare(request)
        await response.write(_pcm(1200))
        await release.wait()
        return response

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        stream = synth.synthesize(
            "hello",
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0, timeout=0.05),
        )
        events = []
        try:
            with pytest.raises(APITimeoutError) as raised:
                async for event in stream:
                    events.append(event)
        finally:
            release.set()
            await stream.aclose()

    assert events
    assert raised.value.retryable is False
    assert request_count == 1


async def test_stream_cancellation_releases_task_and_preserves_injected_session() -> None:
    from livekit.plugins.airy import TTS

    release = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers=_audio_headers())
        await response.prepare(request)
        await response.write(_pcm(1200))
        await release.wait()
        return response

    async with _Server(handler) as server:
        synth = TTS(
            language="ko",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        stream = synth.synthesize("hello")
        try:
            await asyncio.wait_for(anext(stream), timeout=1)
            await stream.aclose()
            assert stream.done
            await synth.aclose()
            assert not server.session.closed
        finally:
            release.set()
            await stream.aclose()


async def test_stream_adapter_synthesizes_multiple_sentences() -> None:
    from livekit.plugins.airy import TTS

    inputs: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        inputs.append((await request.json())["input"])
        return web.Response(body=_pcm(1200), headers=_audio_headers())

    async with _Server(handler) as server:
        synth = TTS(
            language="en",
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
        )
        adapter = tts.StreamAdapter(
            tts=synth,
            sentence_tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=1, stream_context_len=1, retain_format=True
            ),
        )
        stream = adapter.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        events = []
        try:
            stream.push_text("Hello. World.")
            stream.end_input()
            async for event in stream:
                events.append(event)
        finally:
            await stream.aclose()
            await adapter.aclose()

    assert inputs == ["Hello.", "World."]
    assert events
