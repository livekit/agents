from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from xml.etree import ElementTree

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, APIError, APIStatusError, APITimeoutError, tts
from livekit.plugins import microsoft_ai

from .microsoft_ai_fakes import FakeResponse, fake_session, wav_bytes

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

OPTIONS = APIConnectOptions(max_retry=0, timeout=0.5)
PCM = b"\x81\xff" * 3717
TTS_URL = "https://tts.example.invalid/cognitiveservices/v1?deployment=dummy"
VOICE = "en-US-Dummy:test-synthesizer"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Hermetic Microsoft AI tests must not make network requests")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    monkeypatch.delenv("MICROSOFT_AI_ENV_FILE", raising=False)
    for name in ("URL", "REGION", "MODEL", "API_KEY", "VOICE", "SAMPLE_RATE"):
        monkeypatch.delenv(f"MICROSOFT_AI_TTS_{name}", raising=False)


def provider(response: FakeResponse, **options) -> tuple[microsoft_ai.TTS, MagicMock]:
    session = fake_session()
    session.post.return_value = response
    return microsoft_ai.TTS(
        url=TTS_URL,
        model="test-synthesizer",
        voice=VOICE,
        sample_rate=24000,
        api_key="dummy-tts-key",
        http_session=session,
        **options,
    ), session


async def collect(stream: tts.ChunkedStream) -> list[tts.SynthesizedAudio]:
    async def run() -> list[tts.SynthesizedAudio]:
        return [event async for event in stream]

    return await asyncio.wait_for(run(), 3.0)


def ssml_voice(data: bytes) -> ElementTree.Element:
    root = ElementTree.fromstring(data)
    assert root.tag == "{http://www.w3.org/2001/10/synthesis}speak"
    voice = root.find("{http://www.w3.org/2001/10/synthesis}voice")
    assert voice is not None
    return voice


async def test_complete_wav_request_and_exact_audio_framing() -> None:
    response = FakeResponse(wav_bytes(PCM))
    instance, session = provider(response)
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        events = await collect(stream)
    assert b"".join(event.frame.data.tobytes() for event in events) == PCM
    assert all(
        event.frame.sample_rate == 24000 and event.frame.num_channels == 1 for event in events
    )
    assert sum(event.is_final for event in events) == 1 and events[-1].is_final
    assert len({event.request_id for event in events}) == 1
    assert not instance.capabilities.streaming
    assert not instance.capabilities.aligned_transcript
    assert instance.model == "test-synthesizer" and instance.provider == "Microsoft AI"
    args, kwargs = session.post.call_args
    assert args == (TTS_URL,)
    assert "json" not in kwargs
    voice = ssml_voice(kwargs["data"])
    assert voice.text == "Hello."
    assert voice.attrib["name"] == VOICE
    assert ElementTree.fromstring(kwargs["data"]).attrib == {
        "version": "1.0",
        "{http://www.w3.org/XML/1998/namespace}lang": "en-US",
    }
    assert kwargs["headers"]["Ocp-Apim-Subscription-Key"] == "dummy-tts-key"
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["headers"]["Accept"] == "audio/wav"
    assert kwargs["headers"]["Content-Type"] == "application/ssml+xml"
    assert kwargs["headers"]["X-Microsoft-OutputFormat"] == "riff-24khz-16bit-mono-pcm"
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"].total == 30
    assert kwargs["timeout"].connect == OPTIONS.timeout
    assert response.closed
    session.close.assert_not_awaited()


async def test_environment_configuration_and_custom_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "URL": TTS_URL,
        "MODEL": "environment-model",
        "VOICE": "en-US-Dummy:environment-model",
        "SAMPLE_RATE": "24000",
        "API_KEY": "must-not-be-used",
    }.items():
        monkeypatch.setenv(f"MICROSOFT_AI_TTS_{name}", value)
    session = fake_session()
    session.post.return_value = FakeResponse(wav_bytes(PCM))
    instance = microsoft_ai.TTS(headers={}, http_session=session)
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        await collect(stream)
    assert session.post.call_args.kwargs["headers"] == {
        "User-Agent": "LiveKit Agents",
        "Accept": "audio/wav",
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
    }
    assert instance.model == "environment-model"


def test_model_voice_rate_and_credentials_have_no_guessed_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-microsoft-ai")
    with pytest.raises(ValueError, match="SAMPLE_RATE"):
        microsoft_ai.TTS()
    with pytest.raises(ValueError, match="TTS_MODEL"):
        microsoft_ai.TTS(sample_rate=24000)
    with pytest.raises(ValueError, match="TTS_VOICE"):
        microsoft_ai.TTS(sample_rate=24000, model="test")
    with pytest.raises(ValueError, match="TTS_API_KEY"):
        microsoft_ai.TTS(sample_rate=24000, model="test", voice="en-US-Dummy:test", url=TTS_URL)
    assert not hasattr(microsoft_ai.TTS, "with_azure")


@pytest.mark.parametrize("status", [302, 400, 401, 403, 422, 429, 500, 503])
async def test_http_errors_are_explicit_and_do_not_echo_response_bodies(status: int) -> None:
    response = FakeResponse(b"do-not-log-this-dummy-secret", status=status)
    instance, session = provider(response)
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        with pytest.raises(APIStatusError) as caught:
            await collect(stream)
    assert caught.value.status_code == status
    assert caught.value.retryable == (status in (429, 500, 503))
    assert "do-not-log" not in str(caught.value)
    assert response.closed
    session.post.assert_called_once()


async def test_remote_499_is_not_mistaken_for_successful_local_cancellation() -> None:
    instance, _ = provider(FakeResponse(status=499))
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        with pytest.raises(APIError, match="499"):
            await collect(stream)


async def test_retry_after_rate_limit_emits_only_successful_audio() -> None:
    failed = FakeResponse(status=429)
    success = FakeResponse(wav_bytes(PCM))
    instance, session = provider(failed)
    session.post.side_effect = [failed, success]
    async with (
        instance,
        instance.synthesize(
            "Hello.", conn_options=APIConnectOptions(max_retry=1, timeout=0.5)
        ) as stream,
    ):
        events = await collect(stream)
    assert b"".join(event.frame.data.tobytes() for event in events) == PCM
    assert session.post.call_count == 2
    assert failed.closed and success.closed


async def test_interrupted_http_body_does_not_emit_stale_audio_on_retry() -> None:
    failed = FakeResponse(wav_bytes(PCM)[:200], error=aiohttp.ClientPayloadError("dummy"))
    success = FakeResponse(wav_bytes(PCM))
    instance, session = provider(failed)
    session.post.side_effect = [failed, success]
    async with (
        instance,
        instance.synthesize(
            "Hello.", conn_options=APIConnectOptions(max_retry=1, timeout=0.5)
        ) as stream,
    ):
        events = await collect(stream)
    assert b"".join(event.frame.data.tobytes() for event in events) == PCM
    assert len({event.request_id for event in events}) == 1
    assert failed.closed and success.closed


async def test_timeout_is_translated() -> None:
    response = FakeResponse(error=asyncio.TimeoutError("do-not-log-this-endpoint"))
    instance, _ = provider(response)
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        with pytest.raises(APITimeoutError) as caught:
            await collect(stream)
    assert "do-not-log" not in str(caught.value)
    assert response.closed


@pytest.mark.parametrize(
    "content_type", ["application/json", "audio/pcm", "text/event-stream", "audio/mpeg"]
)
async def test_unconfirmed_response_formats_are_not_guessed(content_type: str) -> None:
    instance, _ = provider(FakeResponse(wav_bytes(PCM), content_type=content_type))
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        with pytest.raises(APIError, match="WAV response") as caught:
            await collect(stream)
    assert not caught.value.retryable


@pytest.mark.parametrize(
    "data",
    [
        b"not a WAV",
        wav_bytes(b""),
        wav_bytes(PCM)[:-3],
        wav_bytes(PCM, sample_rate=16000),
        wav_bytes(PCM, channels=2),
        wav_bytes(PCM, width=1),
    ],
)
async def test_invalid_empty_truncated_or_wrong_format_wav_is_rejected(data: bytes) -> None:
    instance, _ = provider(FakeResponse(data))
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        with pytest.raises(APIError) as caught:
            await collect(stream)
    assert not caught.value.retryable


@pytest.mark.parametrize("declared_length", [None, 1000000])
async def test_response_size_bound_applies_to_content_length_and_chunked_bodies(
    declared_length: int | None,
) -> None:
    response = FakeResponse(wav_bytes(PCM), content_length=declared_length)
    instance, _ = provider(response, max_audio_bytes=128)
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        with pytest.raises(APIError, match="max_audio_bytes"):
            await collect(stream)
    assert response.closed


async def test_text_and_timeout_limits_fail_before_http() -> None:
    instance, session = provider(FakeResponse(), max_text_length=8)
    async with instance:
        with pytest.raises(ValueError, match="nonempty"):
            instance.synthesize(" ")
        with pytest.raises(ValueError, match="max_text_length"):
            instance.synthesize("a" * 9)
        with pytest.raises(ValueError, match="greater than zero"):
            instance.synthesize("hello", conn_options=APIConnectOptions(timeout=0))
        with pytest.raises(NotImplementedError, match="streaming"):
            instance.stream()
    session.post.assert_not_called()


async def test_cancellation_closes_http_and_emits_no_late_audio() -> None:
    response = FakeResponse(wav_bytes(PCM), gate=asyncio.Event())
    instance, session = provider(response)
    stream = instance.synthesize("Hello.", conn_options=OPTIONS)
    await asyncio.wait_for(response.content.started.wait(), 1.0)
    await stream.aclose()
    response.content.gate.set()
    assert await collect(stream) == []
    assert response.closed
    assert stream._synthesize_task.done() and stream._metrics_task.done()
    await instance.aclose()
    session.close.assert_not_awaited()


async def test_close_discards_already_buffered_audio() -> None:
    instance, _ = provider(FakeResponse(wav_bytes(PCM)))
    async with instance:
        stream = instance.synthesize("Hello.", conn_options=OPTIONS)
        assert (await asyncio.wait_for(stream.__anext__(), 1.0)).frame
        await stream.aclose()
        assert await collect(stream) == []


async def test_provider_close_cancels_all_requests_and_borrows_session() -> None:
    responses = [FakeResponse(wav_bytes(PCM), gate=asyncio.Event()) for _ in range(2)]
    instance, session = provider(responses[0])
    session.post.side_effect = responses
    streams = [instance.synthesize("Hello.", conn_options=OPTIONS) for _ in range(2)]
    await asyncio.wait_for(asyncio.gather(*(r.content.started.wait() for r in responses)), 1.0)
    await instance.aclose()
    assert all(response.closed for response in responses)
    for stream in streams:
        assert await collect(stream) == []
    session.close.assert_not_awaited()
    with pytest.raises(RuntimeError, match="closed"):
        instance.synthesize("Hello.")


async def test_owned_session_is_lazy_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = fake_session()
    session.post.return_value = FakeResponse(wav_bytes(PCM))
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    instance = microsoft_ai.TTS(
        url=TTS_URL, model="test", voice="en-US-Dummy:test", sample_rate=24000, headers={}
    )
    factory.assert_not_called()
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        await collect(stream)
    factory.assert_called_once()
    session.close.assert_awaited_once()


async def test_existing_sentence_stream_adapter_supports_incremental_text() -> None:
    instance, session = provider(FakeResponse(wav_bytes(PCM)))
    session.post.side_effect = [FakeResponse(wav_bytes(PCM)), FakeResponse(wav_bytes(PCM))]
    adapter = tts.StreamAdapter(tts=instance)
    async with instance, adapter, adapter.stream(conn_options=OPTIONS) as stream:
        stream.push_text("The first sentence is long enough to tokenize on its own. ")
        stream.push_text("The second sentence is also long enough to tokenize on its own.")
        stream.end_input()

        async def consume() -> list[tts.SynthesizedAudio]:
            return [event async for event in stream]

        events = await asyncio.wait_for(consume(), 2.0)
    assert [ssml_voice(call.kwargs["data"]).text for call in session.post.call_args_list] == [
        "The first sentence is long enough to tokenize on its own.",
        "The second sentence is also long enough to tokenize on its own.",
    ]
    assert sum(event.frame.samples_per_channel for event in events) >= len(PCM)
    assert adapter.capabilities.streaming and not instance.capabilities.streaming


@pytest.mark.parametrize("rate", [0, 500, 7999, 192001, True])
def test_unusable_emitter_sample_rates_are_rejected(rate: int) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        microsoft_ai.TTS(
            sample_rate=rate, model="test", voice="en-US-Dummy:test", url=TTS_URL, headers={}
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example.invalid/speech",
        "https://user:dummy@remote.example.invalid/speech",
        "https://remote.example.invalid/speech#fragment",
        "file:///speech",
    ],
)
def test_invalid_or_insecure_remote_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        microsoft_ai.TTS(
            sample_rate=24000, model="test", voice="en-US-Dummy:test", url=url, headers={}
        )


async def test_text_is_escaped_not_interpreted_as_ssml() -> None:
    text = '</voice><audio src="https://example.invalid/audio"/> & <voice>literal'
    instance, session = provider(FakeResponse(wav_bytes(PCM)))
    async with instance, instance.synthesize(text, conn_options=OPTIONS) as stream:
        await collect(stream)
    data = session.post.call_args.kwargs["data"]
    voice = ssml_voice(data)
    assert voice.text == text
    assert list(voice) == []
    assert len(list(ElementTree.fromstring(data).iter())) == 2
    assert b"&lt;audio" in data and b"&amp;" in data


async def test_voice_attribute_is_xml_escaped() -> None:
    voice_id = 'en-US-Dummy" /><audio src="https://example.invalid/a"/><voice name="x:test'
    session = fake_session()
    session.post.return_value = FakeResponse(wav_bytes(PCM))
    instance = microsoft_ai.TTS(
        url=TTS_URL,
        model="test",
        voice=voice_id,
        sample_rate=24000,
        headers={},
        http_session=session,
    )
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        await collect(stream)
    data = session.post.call_args.kwargs["data"]
    assert ssml_voice(data).attrib["name"] == voice_id
    assert len(list(ElementTree.fromstring(data).iter())) == 2


async def test_invalid_xml_controls_fail_before_network() -> None:
    instance, session = provider(FakeResponse())
    async with instance:
        with pytest.raises(ValueError, match="invalid in XML"):
            instance.synthesize("Hello\x00world.")
    session.post.assert_not_called()


@pytest.mark.parametrize(
    ("rate", "format"),
    [
        (8000, "riff-8khz-16bit-mono-pcm"),
        (22050, "riff-22050hz-16bit-mono-pcm"),
        (24000, "riff-24khz-16bit-mono-pcm"),
        (44100, "riff-44100hz-16bit-mono-pcm"),
        (48000, "riff-48khz-16bit-mono-pcm"),
    ],
)
async def test_documented_wav_output_formats(rate: int, format: str) -> None:
    session = fake_session()
    session.post.return_value = FakeResponse(wav_bytes(PCM, sample_rate=rate))
    instance = microsoft_ai.TTS(
        url=TTS_URL,
        model="test",
        voice="en-US-Dummy:test",
        sample_rate=rate,
        headers={},
        http_session=session,
    )
    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        events = await collect(stream)
    assert session.post.call_args.kwargs["headers"]["X-Microsoft-OutputFormat"] == format
    assert all(event.frame.sample_rate == rate for event in events)
    assert b"".join(event.frame.data.tobytes() for event in events) == PCM
