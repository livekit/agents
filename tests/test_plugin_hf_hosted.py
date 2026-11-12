from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from huggingface_hub.errors import HfHubHTTPError

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError
from livekit.plugins import hf
from livekit.plugins.hf import _hosted, stt as hf_stt, tts as hf_tts

pytestmark = pytest.mark.unit

_NO_RETRY = APIConnectOptions(max_retry=0, timeout=5)


def _wav(seconds: float, sample_rate: int = 16000) -> bytes:
    samples = int(seconds * sample_rate)
    tone = (np.sin(np.linspace(0, 400, samples)) * 8000).astype(np.int16)
    frame = rtc.AudioFrame(
        data=tone.tobytes(), sample_rate=sample_rate, num_channels=1, samples_per_channel=samples
    )
    return frame.to_wav_bytes()


def _http_error(status: int) -> HfHubHTTPError:
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    return HfHubHTTPError(f"{status} error", response=response)


class _FakeClient:
    def __init__(self, opts: _hosted.HostedOptions, *, error: Exception | None = None) -> None:
        self.opts = opts
        self.error = error
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.closed = False

    async def automatic_speech_recognition(self, audio: str, **kwargs: Any) -> Any:
        # a .wav path, so every provider can tell the format; it's gone after the call
        assert audio.endswith(".wav")
        with open(audio, "rb") as f:
            self.calls.append(("asr", f.read(), kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=" habari yako ")

    async def text_to_speech(self, text: str, **kwargs: Any) -> bytes:
        self.calls.append(("tts", text, kwargs))
        if self.error is not None:
            raise self.error
        return _wav(1.0)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> list[_FakeClient]:
    clients: list[_FakeClient] = []

    def _make(opts: _hosted.HostedOptions) -> _FakeClient:
        clients.append(_FakeClient(opts))
        return clients[-1]

    monkeypatch.setattr(_hosted, "make_client", _make)

    def _no_local_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("hosted mode must not load a local model")

    monkeypatch.setattr(hf_stt, "_load_stt", _no_local_load)
    monkeypatch.setattr(hf_tts, "_load_tts", _no_local_load)
    return clients


def _frame(seconds: float) -> rtc.AudioFrame:
    samples = int(seconds * 48000)
    return rtc.AudioFrame(
        data=np.zeros(samples, dtype=np.int16).tobytes(),
        sample_rate=48000,
        num_channels=1,
        samples_per_channel=samples,
    )


async def test_stt_provider_sends_wav_and_language(fake_client: list[_FakeClient]) -> None:
    instance = hf.STT(model="openai/whisper-large-v3", provider="fal-ai", language="sw")
    instance.prewarm()  # nothing to load
    instance.load()

    event = await instance.recognize(_frame(0.5))

    assert event.alternatives[0].text == "habari yako"
    assert event.recognition_usage is not None
    assert event.recognition_usage.audio_duration == pytest.approx(0.5)
    kind, audio, kwargs = fake_client[0].calls[0]
    assert audio[:4] == b"RIFF"
    assert kwargs == {"model": "openai/whisper-large-v3", "extra_body": {"language": "sw"}}

    await instance.aclose()
    assert fake_client[0].closed


async def test_stt_hf_inference_and_endpoint(fake_client: list[_FakeClient]) -> None:
    on_hf = hf.STT(model="openai/whisper-large-v3-turbo", provider="hf-inference", language="en")
    endpoint = hf.STT(base_url="https://my-endpoint.test", extra_body={"language": "lug"})

    await on_hf.recognize(_frame(0.2))
    await endpoint.recognize(_frame(0.2))

    # hf-inference rejects unknown parameters, so the language is not added there
    assert fake_client[0].calls[0][2]["extra_body"] is None
    # a dedicated endpoint is the model; no model id is sent
    assert fake_client[1].calls[0][2] == {"model": None, "extra_body": {"language": "lug"}}
    assert fake_client[1].opts.base_url == "https://my-endpoint.test"


@pytest.mark.parametrize(("status", "retryable"), [(503, True), (429, True), (400, False)])
async def test_stt_http_errors_keep_retry_semantics(
    fake_client: list[_FakeClient], status: int, retryable: bool
) -> None:
    instance = hf.STT(provider="hf-inference")
    instance._client = _FakeClient(instance._hosted, error=_http_error(status))

    with pytest.raises(APIStatusError) as exc_info:
        await instance.recognize(_frame(0.2), conn_options=_NO_RETRY)
    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable


async def test_stt_timeout(fake_client: list[_FakeClient]) -> None:
    instance = hf.STT(provider="hf-inference")
    instance._client = _FakeClient(instance._hosted, error=TimeoutError())

    with pytest.raises(APITimeoutError):
        await instance.recognize(_frame(0.2), conn_options=_NO_RETRY)


async def test_tts_provider_decodes_clip(fake_client: list[_FakeClient]) -> None:
    instance = hf.TTS(model="hexgrad/Kokoro-82M", provider="fal-ai", voice="af_heart")
    instance.load()  # nothing to load

    audio = [a async for a in instance.synthesize("Hello there.")]

    pcm = np.concatenate([np.frombuffer(a.frame.data, dtype=np.int16) for a in audio])
    assert audio[0].frame.sample_rate == 24000
    assert abs(len(pcm) - 24000) < 1200  # one second at 16 kHz, resampled to 24 kHz
    kind, text, kwargs = fake_client[0].calls[0]
    assert text == "Hello there."
    assert kwargs == {"model": "hexgrad/Kokoro-82M", "extra_body": {"voice": "af_heart"}}


async def test_hosted_calls_get_more_than_the_connect_timeout(
    fake_client: list[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    timeouts: list[float] = []
    real_wait_for = hf_tts.asyncio.wait_for

    async def _wait_for(awaitable: Any, timeout: float) -> Any:
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(hf_tts.asyncio, "wait_for", _wait_for)
    instance = hf.TTS(model="hexgrad/Kokoro-82M", provider="fal-ai")

    _ = [a async for a in instance.synthesize("Hi.", conn_options=_NO_RETRY)]

    # synthesizing a whole clip on a queued provider can take longer than a connect timeout
    assert timeouts == [_hosted.MIN_REQUEST_TIMEOUT]


def test_sniff_audio_mime() -> None:
    assert _hosted.sniff_audio_mime(_wav(0.1)) == "audio/wav"
    assert _hosted.sniff_audio_mime(b"fLaC....") == "audio/flac"
    assert _hosted.sniff_audio_mime(b"ID3\x04....") == "audio/mpeg"
    assert _hosted.sniff_audio_mime(b"\xff\xfb\x90\x00") == "audio/mpeg"
    assert _hosted.sniff_audio_mime(b"OggS....") == "audio/ogg"
