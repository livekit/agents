from __future__ import annotations

import asyncio
import json
import types
from typing import Any

import pytest

from livekit import rtc
from livekit.plugins.mistralai import STT

pytestmark = pytest.mark.plugin("mistralai")


def _frame(ms: int = 100, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _FakeTranscriptions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def complete_async(self, **kwargs: Any) -> object:
        self.requests.append(kwargs)
        return types.SimpleNamespace(text="hello", language=None, segments=[])


def _stt(transcriptions: _FakeTranscriptions, **kwargs: Any) -> STT:
    client = types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=transcriptions))
    kwargs.setdefault("model", "voxtral-mini-2602")  # the default model is a realtime one
    return STT(client=client, **kwargs)  # type: ignore[arg-type]


async def test_recognize_language_override_applies_to_one_call() -> None:
    transcriptions = _FakeTranscriptions()
    batch_stt = _stt(transcriptions, language="en")

    overridden = await batch_stt.recognize([_frame()], language="fr")
    default = await batch_stt.recognize([_frame()])

    assert [r["language"] for r in transcriptions.requests] == ["fr", "en"]
    assert overridden.alternatives[0].language == "fr"
    assert default.alternatives[0].language == "en"
    assert batch_stt._opts.language == "en"


async def test_recognize_language_override_keeps_auto_detection() -> None:
    # without a configured language the model detects it;
    # a one-off language must not stick for the calls after it
    transcriptions = _FakeTranscriptions()
    batch_stt = _stt(transcriptions)

    await batch_stt.recognize([_frame()], language="fr")
    await batch_stt.recognize([_frame()])

    first, second = transcriptions.requests
    assert first["language"] == "fr"
    assert first["timestamp_granularities"] == ["segment"]
    assert second["language"] is None
    assert second["timestamp_granularities"] == ["segment"]
    assert batch_stt._opts.language is None


@pytest.mark.parametrize("processing", [False, True])
async def test_provider_progress_resets_retries_without_transcript(processing):
    from livekit.plugins.mistralai import stt as mistral_stt

    message = (
        {"type": "transcription.language", "audio_language": "en"}
        if processing
        else {"type": "session.created", "session": {}}
    )

    class Websocket:
        async def __aiter__(self):
            yield json.dumps(message)

    class Connection:
        request_id = ""
        _websocket = Websocket()

    stream = mistral_stt.SpeechStream.__new__(mistral_stt.SpeechStream)
    stream._on_provider_progress = None
    stream._num_retries = 2
    stream._pending_flushes = 0
    from livekit.agents import APIConnectionError

    with pytest.raises(APIConnectionError, match="transcription.done"):
        await stream._recv_task(Connection())
    assert stream._num_retries == (0 if processing else 2)


async def _wait_for_event(events, event_type):
    async def wait():
        while not any(event.type == event_type for event in events):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 1)


async def test_voxtral3_turn_events_keep_the_stream_open(monkeypatch):
    from livekit.agents import stt
    from tests.stt_harness.discovery import provider_by_name
    from tests.stt_harness.drivers import make_audio_segments, open_stream, push_audio

    config = provider_by_name("mistralai_voxtral3")
    async with open_stream(config, monkeypatch) as (stream, collector, server, events, errors):
        first, tail = make_audio_segments(config)
        push_audio(stream, config, first)
        await _wait_for_event(events, stt.SpeechEventType.INTERIM_TRANSCRIPT)
        peer = server.connections[0]
        assert peer.protocol.session["turn_detection"] == {}
        peer.send({"type": "turn.eager_end", "time_s": 0.3})
        await _wait_for_event(events, stt.SpeechEventType.PREFLIGHT_TRANSCRIPT)
        events.clear()
        peer.send({"type": "turn.resume", "time_s": 0.35})
        await _wait_for_event(events, stt.SpeechEventType.INTERIM_TRANSCRIPT)
        peer.send({"type": "turn.end", "time_s": 0.4, "end_reason": "threshold"})
        await _wait_for_event(events, stt.SpeechEventType.FINAL_TRANSCRIPT)
        await _wait_for_event(events, stt.SpeechEventType.END_OF_SPEECH)
        assert not collector.done()
        push_audio(stream, config, tail)
        stream.end_input()
        await asyncio.wait_for(collector, 1)
        assert [
            event.alternatives[0].text
            for event in events
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ] == ["first", "tail"]
        assert not errors


async def test_voxtral3_diarization_reaches_public_events(monkeypatch):
    from livekit.agents import stt
    from tests.stt_harness.discovery import provider_by_name
    from tests.stt_harness.drivers import make_audio_segments, open_stream, push_audio

    config = provider_by_name("mistralai_voxtral3_diarized")
    async with open_stream(config, monkeypatch) as (stream, collector, server, events, errors):
        for segment in make_audio_segments(config):
            push_audio(stream, config, segment)
        stream.end_input()
        await asyncio.wait_for(collector, 1)
        assert server.connections[0].protocol.session["diarize"] is True
        assert "turn_detection" not in server.connections[0].protocol.session
        finals = [
            event.alternatives[0]
            for event in events
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals and all(result.speaker_id == "speaker_1" for result in finals)
        assert all(word.speaker_id == "speaker_1" for result in finals for word in result.words)
        assert not errors


@pytest.mark.parametrize("language", ["fr", "es"])
async def test_voxtral3_session_settings_and_stream_language(language, monkeypatch):
    from livekit.agents import APIConnectOptions
    from tests.stt_harness.discovery import provider_by_name
    from tests.stt_harness.drivers import collect_events, make_audio_segments, push_audio
    from tests.stt_harness.server import STTServer

    config = provider_by_name("mistralai_voxtral3")
    async with STTServer(config, segments=make_audio_segments(config)) as server:
        from mistralai.client import Mistral

        async with STT(
            api_key="test-key",
            client=Mistral(api_key="test-key", server_url=server.url),
            language="fr",
            context_bias=["LiveKit"],
            target_streaming_delay_ms=480,
        ) as instance:
            stream = instance.stream(
                language=language,
                conn_options=APIConnectOptions(max_retry=0, final_response_timeout=1),
            )
            collector = asyncio.create_task(collect_events(stream, []))
            try:
                for segment in make_audio_segments(config):
                    push_audio(stream, config, segment)
                stream.end_input()
                await asyncio.wait_for(collector, 2)
                settings = server.connections[0].protocol.session
                assert settings["language"] == language
                assert settings["context_bias"] == ["LiveKit"]
                assert settings["target_streaming_delay_ms"] == 480
            finally:
                await stream.aclose()
                await asyncio.gather(collector, return_exceptions=True)


@pytest.mark.parametrize(
    "event_type, expected_retries",
    [
        ("turn.start", 0),
        ("turn.eager_end", 0),
        ("turn.resume", 0),
        ("turn.end", 0),
        ("turn.unknown", 2),
    ],
)
async def test_voxtral3_turn_events_reset_retries(event_type, expected_retries):
    from livekit.plugins.mistralai.stt import SpeechStream

    stream = SpeechStream.__new__(SpeechStream)
    stream._on_provider_progress = None
    stream._num_retries = 2
    stream._speaking = False
    stream._audio_pending = False
    stream._audio_ended = False
    stream._event_ch = types.SimpleNamespace(send_nowait=lambda event: None)
    await stream._on_event(types.SimpleNamespace(), {"type": event_type})
    assert stream._num_retries == expected_retries


async def test_batch_voxtral3_diarization_preserves_speaker_labels():
    class DiarizedTranscriptions(_FakeTranscriptions):
        async def complete_async(self, **kwargs):
            self.requests.append(kwargs)
            return types.SimpleNamespace(
                text="hello",
                language="en",
                segments=[
                    types.SimpleNamespace(text="hello", start=0.0, end=0.1, speaker_id="speaker_1")
                ],
            )

    transcriptions = DiarizedTranscriptions()
    instance = _stt(transcriptions, model="voxtral-mini-transcribe-3", diarize=True)
    try:
        event = await instance.recognize([_frame()])
        assert transcriptions.requests[0]["diarize"] is True
        assert transcriptions.requests[0]["timestamp_granularities"] == ["segment"]
        result = event.alternatives[0]
        assert result.speaker_id == "speaker_1"
        assert result.words and result.words[0].speaker_id == "speaker_1"
    finally:
        await instance.aclose()


@pytest.mark.parametrize(
    "provider_name", ["mistralai", "mistralai_voxtral3", "mistralai_voxtral3_diarized"]
)
@pytest.mark.parametrize("completion", ["done", "close", "silent"])
async def test_flush_completion_cannot_replace_end_audio_completion(
    provider_name, completion, monkeypatch
):
    from dataclasses import replace

    from livekit.agents import APIConnectionError, APITimeoutError, stt
    from tests.stt_harness.discovery import provider_by_name
    from tests.stt_harness.drivers import make_audio_segments, open_stream, push_audio
    from tests.stt_harness.server import json_message

    config = provider_by_name(provider_name)

    class HoldEndAudio(config.protocol):
        async def receive(self, message):
            if json_message(message)["type"] == "input_audio.end":
                self.connection.end_input()
                return
            await super().receive(message)

    config = replace(config, protocol=HoldEndAudio)
    async with open_stream(config, monkeypatch) as (stream, collector, server, events, errors):
        for segment in make_audio_segments(config):
            push_audio(stream, config, segment)
        stream.end_input()
        await asyncio.wait_for(server.input_ended.wait(), 1)
        await _wait_for_event(events, stt.SpeechEventType.FINAL_TRANSCRIPT)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(collector), 0.05)
        peer = server.connections[0]
        if completion == "done":
            peer.send(peer.protocol.done_message(""))
            await asyncio.wait_for(collector, 1)
            assert not errors
        else:
            if completion == "close":
                await peer.ws.close()
            expected = APIConnectionError if completion == "close" else APITimeoutError
            with pytest.raises(expected):
                await asyncio.wait_for(collector, 1)
            assert errors and not errors[-1].recoverable
