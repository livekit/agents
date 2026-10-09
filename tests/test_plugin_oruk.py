from __future__ import annotations

import asyncio
import io
import wave
from email import policy
from email.parser import BytesParser
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, stt
from livekit.plugins.oruk import STT

pytestmark = pytest.mark.unit
OPTIONS = APIConnectOptions(max_retry=2, retry_interval=0, timeout=1)


def audio(rate=16000, channels=1, seconds=0.1):
    data = np.full((round(rate * seconds), channels), 1000, dtype=np.int16)
    return rtc.AudioFrame(data.tobytes(), rate, channels, len(data))


def parts(request):
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + request.headers["Content-Type"].encode() + b"\r\n\r\n" + request.content
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("rate,channels", [(16000, 1), (48000, 2), (12000, 1), (96000, 1)])
async def test_authenticated_upload_and_audio_conversion(rate, channels):
    def handler(request):
        assert request.url == "https://speech-api.oruk.ai/v1/audio/transcriptions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["X-Request-ID"]
        fields = parts(request)
        assert set(fields) == {"file", "model"}
        assert fields["model"] == b"oruk-spectra-2"
        with wave.open(io.BytesIO(fields["file"])) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            assert abs(wav.getnframes() / 16000 - 0.1) < 0.001
        return httpx.Response(
            200, json={"text": "Hallo Welt"}, headers={"X-Request-ID": "result-1"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plugin = STT(api_key="test-key", http_client=client)
        event = await plugin.recognize(audio(rate, channels), conn_options=OPTIONS)
        assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        assert event.request_id == "result-1"
        assert event.alternatives[0].text == "Hallo Welt"
        assert event.alternatives[0].language == ""
        assert not plugin.capabilities.streaming
        assert not plugin.capabilities.aligned_transcript
        await plugin.aclose()
        assert not client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 403, 409, 400])
async def test_permanent_errors_do_not_retry(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"code": "request_already_completed"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIStatusError) as caught:
            await STT(api_key="test", http_client=client).recognize(audio(), conn_options=OPTIONS)
        assert caught.value.status_code == status
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["network", "timeout", "server", "upload_busy", "model_busy"])
async def test_retries_preserve_audio_and_request_identity(failure):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "network":
                raise httpx.ReadError("connection lost", request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            return httpx.Response(
                500 if failure == "server" else 429,
                json={"error": {"code": failure}},
                headers={"Retry-After": "2"},
            )
        return httpx.Response(200, json={"text": "retry succeeded"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch("livekit.plugins.oruk.stt.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await STT(api_key="test", http_client=client).recognize(audio(), conn_options=OPTIONS)
            if failure in ("upload_busy", "model_busy"):
                sleep.assert_awaited_once_with(2)
        assert len(requests) == 2
        assert parts(requests[0])["file"] == parts(requests[1])["file"]
        assert (requests[0].headers["X-Request-ID"] == requests[1].headers["X-Request-ID"]) is (
            failure != "model_busy"
        )


@pytest.mark.asyncio
async def test_concurrent_utterances_have_distinct_ids():
    ids = []

    async def handler(request):
        ids.append(request.headers["X-Request-ID"])
        await asyncio.sleep(0)
        return httpx.Response(200, json={"text": "hello"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plugin = STT(api_key="test", http_client=client)
        await asyncio.gather(*(plugin.recognize(audio(), conn_options=OPTIONS) for _ in range(4)))
        assert len(set(ids)) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [{"text": None}, {"text": "hello", "language": 42}, []])
async def test_malformed_success_is_not_retried(result):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError, match="Invalid Oruk"):
            await STT(api_key="test", http_client=client).recognize(audio(), conn_options=OPTIONS)
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_rejects_unsupported_audio_and_language_before_network():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("network"))
    ) as client:
        plugin = STT(api_key="test", http_client=client)
        for seconds in (0.01, 60.01):
            with pytest.raises(ValueError, match="45 ms"):
                await plugin.recognize(audio(seconds=seconds))
        with pytest.raises(ValueError, match="language forcing"):
            await plugin.recognize(audio(), language="de")


def test_missing_key(monkeypatch):
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ORUK_API_KEY"):
        STT()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["network", "server", "model_busy"])
async def test_recovered_attempt_is_not_a_terminal_error(failure):
    requests = []
    errors = []
    metrics = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "network":
                raise httpx.ReadError("connection lost", request=request)
            return httpx.Response(
                429 if failure == "model_busy" else 500,
                json={"error": {"code": failure}},
            )
        return httpx.Response(200, json={"text": "recovered"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plugin = STT(api_key="test", http_client=client)
        plugin.on("error", errors.append)
        plugin.on("metrics_collected", metrics.append)
        with patch("livekit.plugins.oruk.stt.asyncio.sleep", new_callable=AsyncMock):
            event = await plugin.recognize(audio(), conn_options=OPTIONS)
        assert event.alternatives[0].text == "recovered"
        assert [error.recoverable for error in errors] == [True]
        assert len(metrics) == 1
        assert metrics[0].request_id == event.request_id
        assert metrics[0].audio_duration == pytest.approx(0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,max_retry", [(503, 2), (503, 0), (401, 2)])
async def test_only_exhausted_or_permanent_failure_emits_terminal_error(status, max_retry):
    requests = []
    errors = []
    metrics = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": "unavailable"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plugin = STT(api_key="test", http_client=client)
        plugin.on("error", errors.append)
        plugin.on("metrics_collected", metrics.append)
        with patch("livekit.plugins.oruk.stt.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(APIStatusError) as caught:
                await plugin.recognize(
                    audio(),
                    conn_options=APIConnectOptions(
                        max_retry=max_retry, retry_interval=0, timeout=1
                    ),
                )
        attempts = max_retry + 1 if status == 503 else 1
        assert len(requests) == attempts
        assert [error.recoverable for error in errors] == [True] * (attempts - 1) + [False]
        assert errors[-1].error is caught.value
        assert metrics == []
        assert len({request.headers["X-Request-ID"] for request in requests}) == 1


@pytest.mark.asyncio
async def test_cancelled_backoff_does_not_replay_audio_or_reuse_request_id():
    requests = []
    errors = []
    metrics = []
    backing_off = asyncio.Event()
    resume = asyncio.Event()

    async def backoff(_delay):
        backing_off.set()
        await resume.wait()

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"text": "new utterance"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        plugin = STT(api_key="test", http_client=client)
        plugin.on("error", errors.append)
        plugin.on("metrics_collected", metrics.append)
        with patch("livekit.plugins.oruk.stt.asyncio.sleep", side_effect=backoff):
            task = asyncio.create_task(plugin.recognize(audio(), conn_options=OPTIONS))
            try:
                await asyncio.wait_for(backing_off.wait(), timeout=1)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert len(requests) == 1
        assert [error.recoverable for error in errors] == [True]
        assert metrics == []
        event = await plugin.recognize(audio(), conn_options=OPTIONS)
        assert event.alternatives[0].text == "new utterance"
        assert len(requests) == 2
        assert requests[0].headers["X-Request-ID"] != requests[1].headers["X-Request-ID"]
        assert len(metrics) == 1
