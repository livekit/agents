from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from xml.etree import ElementTree

import aiohttp
import pytest

from examples.microsoft import microsoft_ai_smoke as smoke, microsoft_ai_tts_room as room_example
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    JobContext,
    tts,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import microsoft_ai

from .fake_io import FakeAudioOutput
from .microsoft_ai_fakes import (
    DUMMY_CONFIG,
    FakeResponse,
    fake_session,
    no_http_session as no_http_session,
    no_network as no_network,
    wav_bytes,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

OPTIONS = APIConnectOptions(max_retry=0, timeout=0.5)
PCM = b"\x81\xff" * 3717
TTS_URL = "https://tts.example.invalid/cognitiveservices/v1?deployment=dummy"
VOICE = "en-US-Dummy:test-synthesizer"


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


@pytest.mark.usefixtures("no_http_session")
@pytest.mark.parametrize("voice", ["short-name", "en-US-Dummy:wrong-model", ":file-synthesizer"])
def test_voice_must_include_the_configured_model(tmp_path: Path, voice: str) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    with pytest.raises(ValueError, match="full voice ID"):
        microsoft_ai.TTS(env_file=path, voice=voice)


@pytest.mark.usefixtures("no_http_session")
def test_model_validation_is_case_insensitive_but_does_not_invent_aliases(tmp_path: Path) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    instance = microsoft_ai.TTS(
        env_file=path, model="mai-test-flash", voice="en-US-Dummy:MAI-Test-Flash"
    )
    assert instance.model == "mai-test-flash"
    with pytest.raises(ValueError, match="configured model"):
        microsoft_ai.TTS(
            env_file=path, model="mai-test-2.1-flash", voice="en-US-Dummy:MAI-Test-2-Flash"
        )


@pytest.mark.usefixtures("no_http_session")
@pytest.mark.parametrize("region", ["eastus2", "EastUS2"])
def test_region_constructs_the_documented_public_cloud_endpoint(region: str) -> None:
    instance = microsoft_ai.TTS(
        region=region, api_key="dummy", model="test", voice="en-US-Dummy:test", sample_rate=24000
    )
    assert instance._client.url == "https://eastus2.tts.speech.microsoft.com/cognitiveservices/v1"


@pytest.mark.usefixtures("no_http_session")
def test_region_from_file_and_constructor_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG.replace(
            "MICROSOFT_AI_TTS_URL=https://tts.example.invalid/cognitiveservices/v1",
            "MICROSOFT_AI_TTS_URL=\nMICROSOFT_AI_TTS_REGION=eastus2",
        ),
        encoding="utf-8",
    )
    assert microsoft_ai.TTS(env_file=path)._client.url == (
        "https://eastus2.tts.speech.microsoft.com/cognitiveservices/v1"
    )
    monkeypatch.setenv("MICROSOFT_AI_TTS_REGION", "westeurope")
    assert microsoft_ai.TTS(env_file=path)._client.url.startswith("https://westeurope.")
    assert microsoft_ai.TTS(env_file=path, region="eastus2")._client.url.startswith(
        "https://eastus2."
    )


@pytest.mark.usefixtures("no_http_session")
def test_configured_url_wins_over_region_and_is_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    instance = microsoft_ai.TTS(env_file=path, region="eastus2")
    assert instance._client.url == "https://tts.example.invalid/cognitiveservices/v1"
    monkeypatch.setenv("MICROSOFT_AI_TTS_URL", "https://override.example.invalid/custom?q=dummy")
    instance = microsoft_ai.TTS(env_file=path, region="eastus2")
    assert instance._client.url == "https://override.example.invalid/custom?q=dummy"
    instance = microsoft_ai.TTS(
        env_file=path, region="eastus2", url="https://argument.example.invalid/exact"
    )
    assert instance._client.url == "https://argument.example.invalid/exact"


@pytest.mark.usefixtures("no_http_session")
@pytest.mark.parametrize("region", ["east us 2", "eastus2.example.invalid/path", "../eastus2", ""])
def test_invalid_region_is_not_interpolated_into_a_host(region: str) -> None:
    with pytest.raises(ValueError):
        microsoft_ai.TTS(
            region=region,
            api_key="dummy",
            model="test",
            voice="en-US-Dummy:test",
            sample_rate=24000,
        )


def test_smoke_refuses_audio_dumping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LK_DUMP_TTS", "1")
    monkeypatch.setattr(sys, "argv", ["microsoft_ai_smoke.py", "--run-live", "--tts"])
    factory = MagicMock()
    monkeypatch.setattr(microsoft_ai, "TTS", factory)
    with pytest.raises(SystemExit) as caught:
        smoke.main()
    assert caught.value.code == 2
    factory.assert_not_called()


async def test_tts_smoke_sends_only_the_fixed_approved_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session = fake_session()
    response = FakeResponse(wav_bytes(b"\x01\x00" * 1001))
    session.post.return_value = response
    instance = microsoft_ai.TTS(
        url="https://tts.example.invalid/speech",
        model="test",
        voice="en-US-Dummy:test",
        sample_rate=24000,
        headers={},
        http_session=session,
    )
    async with instance:
        await smoke._check_tts(instance)
    session.post.assert_called_once()
    body = ElementTree.fromstring(session.post.call_args.kwargs["data"])
    assert next(iter(body)).text == smoke.TTS_TEXT
    assert "json" not in session.post.call_args.kwargs
    output = capsys.readouterr().out
    assert smoke.TTS_TEXT not in output
    assert "PCM16 mono frames at 24000 Hz" in output
    assert "audio_duration=" in output and "not model TTFA" in output
    assert response.closed
    assert smoke.CONNECT_OPTIONS.max_retry == 0
    assert smoke.MAX_DURATION == 5.0


def test_opted_in_cli_runs_only_selected_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LK_DUMP_TTS", raising=False)
    monkeypatch.setattr(sys, "argv", ["microsoft_ai_smoke.py", "--run-live", "--tts"])
    run = AsyncMock()
    monkeypatch.setattr(smoke, "_run", run)
    smoke.main()
    run.assert_awaited_once_with(pcm=None, expected=None, check_tts=True, env_file=None)


class _Handle:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.awaited = False

    def __await__(self):
        async def finish() -> None:
            self.awaited = True

        return finish().__await__()

    def exception(self) -> Exception | None:
        assert self.awaited
        return self.error


def _mock_example(monkeypatch: pytest.MonkeyPatch):
    speech = MagicMock(spec=microsoft_ai.TTS)
    speech.sample_rate = 24000
    speech.__aenter__.return_value = speech
    speech.__aexit__.return_value = False
    provider_factory = MagicMock(return_value=speech)
    monkeypatch.setattr(room_example.microsoft_ai, "TTS", provider_factory)
    monkeypatch.setattr(
        room_example.microsoft_ai,
        "STT",
        MagicMock(side_effect=AssertionError("TTS-only example must not construct STT")),
    )
    started = asyncio.Event()
    ready = asyncio.Event()
    handle = _Handle()
    session = MagicMock(spec=AgentSession)
    session.room_io = SimpleNamespace(wait_for_ready=AsyncMock(side_effect=ready.wait))
    session.say.return_value = handle

    async def start(**kwargs: object) -> None:
        started.set()

    session.start = AsyncMock(side_effect=start)
    session_factory = MagicMock(return_value=session)
    monkeypatch.setattr(room_example, "AgentSession", session_factory)
    context = MagicMock(spec=JobContext)
    context.room = MagicMock(spec=rtc.Room)
    return context, session, speech, provider_factory, session_factory, started, ready, handle


async def test_waits_for_subscription_and_says_once_without_input_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, factory, session_factory, started, ready, handle = _mock_example(
        monkeypatch
    )
    monkeypatch.setenv("MICROSOFT_AI_ENV_FILE", "selected-private-config.env")

    def shutdown(*, reason: str) -> None:
        assert reason == "TTS greeting complete"
        session.aclose.assert_awaited_once()
        speech.__aexit__.assert_awaited_once()

    context.shutdown.side_effect = shutdown
    task = asyncio.create_task(room_example.entrypoint(context))
    try:
        await asyncio.wait_for(started.wait(), 1)
        session.say.assert_not_called()
        ready.set()
        await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    factory.assert_called_once_with(env_file="selected-private-config.env")
    kwargs = session_factory.call_args.kwargs
    assert kwargs["tts"] is speech
    assert kwargs["vad"] is None
    assert kwargs["turn_handling"] == {"turn_detection": None}
    assert kwargs["user_away_timeout"] is None
    assert "stt" not in kwargs and "llm" not in kwargs
    assert kwargs["conn_options"].tts_conn_options.max_retry == 0
    start = session.start.call_args.kwargs
    assert start["room"] is context.room
    assert start["record"] is False and start["session_host"] is False
    options = start["room_options"]
    assert options.audio_input is False and options.video_input is False
    assert options.text_input is False and options.text_output is False
    assert options.audio_output.sample_rate == 24000
    session.say.assert_called_once_with(
        "Hello, this is a Microsoft AI voice test.",
        allow_interruptions=False,
        add_to_chat_ctx=False,
    )
    assert handle.awaited
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()
    context.shutdown.assert_called_once_with(reason="TTS greeting complete")


async def test_cancelling_before_room_ready_closes_without_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, _, _, started, _, _ = _mock_example(monkeypatch)
    task = asyncio.create_task(room_example.entrypoint(context))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    session.say.assert_not_called()
    context.shutdown.assert_not_called()
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_readiness_timeout_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, _, _, _, _, _ = _mock_example(monkeypatch)
    session.room_io.wait_for_ready.side_effect = asyncio.TimeoutError()
    with pytest.raises(asyncio.TimeoutError):
        await room_example.entrypoint(context)
    session.say.assert_not_called()
    context.shutdown.assert_not_called()
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_speech_handle_error_is_checked_and_resources_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, _, _, _, ready, handle = _mock_example(monkeypatch)
    error = APIStatusError("Microsoft AI TTS request failed", status_code=401)
    handle.error = error
    ready.set()
    with pytest.raises(APIStatusError) as caught:
        await room_example.entrypoint(context)
    assert caught.value is error
    assert handle.awaited
    context.shutdown.assert_not_called()
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_start_failure_closes_session_and_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    context, session, speech, _, _, _, _, _ = _mock_example(monkeypatch)
    session.start.side_effect = RuntimeError("room setup failed")
    with pytest.raises(RuntimeError, match="room setup failed"):
        await room_example.entrypoint(context)
    session.say.assert_not_called()
    context.shutdown.assert_not_called()
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_actual_agent_session_say_uses_only_tts_and_emits_audio() -> None:
    pcm = b"\x81\x01" * 1200
    response = FakeResponse(wav_bytes(pcm))
    http = fake_session()
    http.post.return_value = response
    speech = microsoft_ai.TTS(
        url="https://tts.example.invalid/cognitiveservices/v1",
        model="test-model",
        voice="en-US-Dummy:test-model",
        sample_rate=24000,
        headers={},
        http_session=http,
    )
    output = FakeAudioOutput(sample_rate=24000)
    capture = AsyncMock(wraps=output.capture_frame)
    output.capture_frame = capture
    async with speech:
        session = AgentSession(
            tts=speech,
            vad=None,
            turn_handling={"turn_detection": None},
            user_away_timeout=None,
            conn_options=SessionConnectOptions(
                tts_conn_options=APIConnectOptions(max_retry=0, timeout=0.5)
            ),
        )
        session.output.audio = output
        try:
            await session.start(
                Agent(instructions="Say the supplied text."), session_host=False, record=False
            )
            handle = session.say(
                room_example.GREETING, allow_interruptions=False, add_to_chat_ctx=False
            )
            await asyncio.wait_for(handle, 3)
            assert handle.exception() is None
            assert session.stt is None and session.llm is None and session.vad is None
        finally:
            await session.aclose()
    http.post.assert_called_once()
    root = ElementTree.fromstring(http.post.call_args.kwargs["data"])
    assert next(iter(root)).text == room_example.GREETING
    frames = [call.args[0] for call in capture.await_args_list]
    assert frames
    assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in frames)
    assert b"".join(frame.data.tobytes() for frame in frames).startswith(pcm)
    assert response.closed
