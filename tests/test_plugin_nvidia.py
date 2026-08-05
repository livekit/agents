# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.nvidia import stt as nvidia_stt, tts as nvidia_tts

pytestmark = pytest.mark.unit


def _audio_frame(byte: int = 1) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes([byte, byte]) * 160,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


@pytest.mark.parametrize("method_name", ["synthesize", "synthesize_online"])
def test_installed_nvidia_speech_client_has_supported_tts_signature(
    method_name: str,
) -> None:
    method = getattr(nvidia_tts.riva.client.SpeechSynthesisService, method_name)
    parameters = inspect.signature(method).parameters

    legacy_options = {"audio_prompt_file", "quality"}
    current_options = {"zero_shot_audio_prompt_file", "zero_shot_quality"}
    assert legacy_options <= parameters.keys() or current_options <= parameters.keys()


def test_stt_reports_capabilities_for_selected_inference_mode() -> None:
    automatic = nvidia_stt.STT(api_key="test-key")
    streaming = nvidia_stt.STT(api_key="test-key", inference_mode="streaming")
    offline = nvidia_stt.STT(api_key="test-key", inference_mode="offline")

    assert automatic.capabilities.streaming is True
    assert automatic.capabilities.offline_recognize is True
    assert streaming.capabilities.streaming is True
    assert streaming.capabilities.offline_recognize is False
    assert streaming.capabilities.interim_results is True
    assert offline.capabilities.streaming is False
    assert offline.capabilities.offline_recognize is True
    assert offline.capabilities.interim_results is False

    with pytest.raises(ValueError, match="inference_mode='streaming'"):
        offline.stream()


async def test_streaming_stt_rejects_offline_recognize() -> None:
    service = nvidia_stt.STT(api_key="test-key", inference_mode="streaming")

    with pytest.raises(ValueError, match="inference_mode='offline'"):
        await service.recognize([_audio_frame()])


async def test_stt_auto_mode_preserves_offline_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_stt.STT(api_key="test-key")
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[SimpleNamespace(transcript="hello", confidence=0.9, words=[])]
            )
        ]
    )
    fake_service = SimpleNamespace(offline_recognize=lambda audio, config: response)
    monkeypatch.setattr(service, "_ensure_asr_service", lambda: fake_service)

    event = await service.recognize([_audio_frame()], conn_options=APIConnectOptions(max_retry=0))

    assert event.alternatives[0].text == "hello"


def test_stt_builds_nvidia_speech_recognition_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def recognition_config(**kwargs):
        captured["recognition"] = kwargs
        return SimpleNamespace(
            diarization_config=SimpleNamespace(
                CopyFrom=lambda config: captured.setdefault("diarization", config)
            )
        )

    monkeypatch.setattr(nvidia_stt.riva.client, "RecognitionConfig", recognition_config)
    monkeypatch.setattr(
        nvidia_stt.riva.client,
        "add_word_boosting_to_config",
        lambda config, words, score: captured.setdefault("word_boosting", (words, score)),
    )

    service = nvidia_stt.STT(
        api_key="test-key",
        model="parakeet-sortformer",
        enable_diarization=True,
        max_speaker_count=2,
        profanity_filter=True,
        verbatim_transcripts=True,
        boosted_lm_words=["LiveKit"],
        boosted_lm_score=7.0,
    )

    assert service.provider == "NVIDIA Speech"

    service._create_recognition_config(
        language=LanguageCode("en-US"),
        sample_rate=48000,
        audio_channel_count=2,
    )

    assert captured["recognition"]["model"] == "parakeet-sortformer"
    assert captured["recognition"]["max_alternatives"] == 1
    assert captured["recognition"]["profanity_filter"] is True
    assert captured["recognition"]["verbatim_transcripts"] is True
    assert captured["recognition"]["sample_rate_hertz"] == 48000
    assert captured["recognition"]["audio_channel_count"] == 2
    assert captured["word_boosting"] == (["LiveKit"], 7.0)
    assert captured["diarization"].enable_speaker_diarization is True
    assert captured["diarization"].max_speaker_count == 2


def test_stt_applies_streaming_endpointing_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    streaming_config = SimpleNamespace()

    monkeypatch.setattr(
        nvidia_stt.riva.client,
        "add_endpoint_parameters_to_config",
        lambda config, *args: captured.setdefault("endpointing", (config, args)),
    )
    monkeypatch.setattr(
        nvidia_stt.riva.client,
        "add_custom_configuration_to_config",
        lambda config, custom: captured.setdefault("custom", (config, custom)),
        raising=False,
    )

    service = nvidia_stt.STT(
        api_key="test-key",
        endpointing=nvidia_stt.EndpointingConfig(
            start_history=11,
            start_threshold=0.35,
            stop_history=21,
            stop_history_eou=31,
            stop_threshold=0.45,
            stop_threshold_eou=0.55,
        ),
        options={"custom_configuration": "enable_vad_endpointing:true"},
    )

    service._apply_streaming_config_extensions(streaming_config)

    assert captured["endpointing"] == (
        streaming_config,
        (11, 0.35, 21, 31, 0.45, 0.55),
    )
    assert captured["custom"] == (streaming_config, "enable_vad_endpointing:true")


def test_stt_rejects_unsupported_custom_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(
        nvidia_stt.riva.client,
        "add_custom_configuration_to_config",
        raising=False,
    )
    service = nvidia_stt.STT(
        api_key="test-key",
        options={"custom_configuration": "enable_vad_endpointing:true"},
    )

    with pytest.raises(ValueError, match="custom_configuration"):
        service._apply_streaming_config_extensions(SimpleNamespace())


def test_stt_endpointing_low_latency_preset() -> None:
    assert nvidia_stt._endpointing_values(nvidia_stt.EndpointingConfig()) == (
        -1,
        -1.0,
        500,
        240,
        -1.0,
        -1.0,
    )


def test_stt_normalizes_word_timestamps_and_speaker_id() -> None:
    words = [
        SimpleNamespace(word="hello", start_time=1000, end_time=1250, speaker_tag=1),
        SimpleNamespace(word="world", start_time=1.25, end_time=1.75, speaker_tag=1),
    ]
    alternative = SimpleNamespace(transcript="hello world", confidence=0.9, words=words)

    speech_data = nvidia_stt._convert_to_speech_data(
        alternative,
        language=LanguageCode("en-US"),
        start_time_offset=0.5,
        enable_diarization=True,
        is_final=True,
    )

    assert speech_data.start_time == 1.5
    assert speech_data.end_time == 2.25
    assert speech_data.words is not None
    assert speech_data.words[0].start_time == 1.5
    assert speech_data.words[0].end_time == 1.75
    assert speech_data.words[1].start_time == 1.75
    assert speech_data.words[1].end_time == 2.25
    assert speech_data.speaker_id == "S1"


def test_stt_normalizes_protobuf_duration_timestamps() -> None:
    value = SimpleNamespace(seconds=2, nanos=500_000_000)

    assert nvidia_stt._time_offset_seconds(value) == 2.5


def test_stt_offline_combines_consecutive_results_into_complete_hypotheses() -> None:
    def word(text: str, start: int, end: int, speaker_tag: int) -> SimpleNamespace:
        return SimpleNamespace(
            word=text,
            start_time=start,
            end_time=end,
            speaker_tag=speaker_tag,
        )

    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript="hello",
                        confidence=0.8,
                        words=[word("hello", 0, 500, 1)],
                    ),
                    SimpleNamespace(
                        transcript="yellow",
                        confidence=0.4,
                        words=[word("yellow", 0, 500, 2)],
                    ),
                ]
            ),
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript="world",
                        confidence=0.9,
                        words=[word("world", 500, 1000, 1)],
                    ),
                    SimpleNamespace(
                        transcript="word",
                        confidence=0.6,
                        words=[word("word", 500, 1000, 2)],
                    ),
                ]
            ),
        ]
    )

    event = nvidia_stt._response_to_speech_event(
        response,
        language=LanguageCode("en-US"),
        request_id="request-id",
        event_type=nvidia_stt.stt.SpeechEventType.FINAL_TRANSCRIPT,
        enable_diarization=True,
        is_final=True,
    )

    assert len(event.alternatives) == 2
    assert event.alternatives[0].text == "hello world"
    assert event.alternatives[0].confidence == pytest.approx(0.85)
    assert event.alternatives[0].speaker_id == "S1"
    assert event.alternatives[0].words is not None
    assert event.alternatives[0].words[-1].end_time == 1.0
    assert event.alternatives[1].text == "yellow word"


async def test_stt_flush_starts_a_new_backend_rpc_for_each_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_stt.STT(api_key="test-key")
    calls: list[list[bytes]] = []

    class FakeASRService:
        def streaming_response_generator(self, audio_generator, config):
            calls.append(list(audio_generator))
            return iter(())

    monkeypatch.setattr(service, "_ensure_asr_service", lambda: FakeASRService())
    monkeypatch.setattr(
        nvidia_stt.SpeechStream,
        "_create_streaming_config",
        lambda self: SimpleNamespace(),
    )

    stream = service.stream()
    stream.push_frame(_audio_frame(1))
    stream.flush()
    stream.push_frame(_audio_frame(2))
    stream.flush()
    stream.end_input()

    await asyncio.wait_for(stream._task, timeout=2.0)

    assert calls == [
        [_audio_frame(1).data.tobytes()],
        [_audio_frame(2).data.tobytes()],
    ]
    await stream.aclose()


async def test_stt_replays_segment_after_retryable_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_stt.STT(api_key="test-key")
    calls: list[list[bytes]] = []

    class FakeASRService:
        def streaming_response_generator(self, audio_generator, config):
            calls.append(list(audio_generator))
            if len(calls) == 1:
                raise APIConnectionError("temporary failure")
            return iter(())

    monkeypatch.setattr(service, "_ensure_asr_service", lambda: FakeASRService())
    monkeypatch.setattr(
        nvidia_stt.SpeechStream,
        "_create_streaming_config",
        lambda self: SimpleNamespace(),
    )

    stream = service.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    stream.push_frame(_audio_frame(3))
    stream.end_input()

    await asyncio.wait_for(stream._task, timeout=2.0)

    expected_audio = [_audio_frame(3).data.tobytes()]
    assert calls == [expected_audio, expected_audio]
    await stream.aclose()


async def test_stt_retry_exhaustion_does_not_trigger_audio_losing_outer_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_stt.STT(api_key="test-key")
    calls: list[list[bytes]] = []

    class FailingASRService:
        def streaming_response_generator(self, audio_generator, config):
            calls.append(list(audio_generator))
            raise APIConnectionError("temporary failure")

    monkeypatch.setattr(service, "_ensure_asr_service", lambda: FailingASRService())
    monkeypatch.setattr(
        nvidia_stt.SpeechStream,
        "_create_streaming_config",
        lambda self: SimpleNamespace(),
    )

    stream = service.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    stream.push_frame(_audio_frame(4))
    stream.end_input()

    with pytest.raises(APIConnectionError, match="failed after 2 attempts"):
        await asyncio.wait_for(stream._task, timeout=2.0)

    expected_audio = [_audio_frame(4).data.tobytes()]
    assert calls == [expected_audio, expected_audio]
    await stream.aclose()


async def test_stt_closes_speech_segment_when_backend_has_no_final_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_stt.STT(api_key="test-key")
    interim_response = SimpleNamespace(
        results=[
            SimpleNamespace(
                is_final=False,
                alternatives=[SimpleNamespace(transcript="hello", confidence=0.5, words=[])],
            )
        ]
    )

    class InterimOnlyASRService:
        def streaming_response_generator(self, audio_generator, config):
            list(audio_generator)
            return iter((interim_response,))

    monkeypatch.setattr(service, "_ensure_asr_service", lambda: InterimOnlyASRService())
    monkeypatch.setattr(
        nvidia_stt.SpeechStream,
        "_create_streaming_config",
        lambda self: SimpleNamespace(),
    )

    stream = service.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_frame(_audio_frame(5))
    stream.end_input()

    await asyncio.wait_for(stream._task, timeout=2.0)
    events = [event async for event in stream]

    assert [event.type for event in events] == [
        nvidia_stt.stt.SpeechEventType.START_OF_SPEECH,
        nvidia_stt.stt.SpeechEventType.INTERIM_TRANSCRIPT,
        nvidia_stt.stt.SpeechEventType.END_OF_SPEECH,
    ]
    await stream.aclose()


async def test_stt_cancellation_closes_backend_audio_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_stt.STT(api_key="test-key")
    started = threading.Event()
    finished = threading.Event()

    class BlockingASRService:
        def streaming_response_generator(self, audio_generator, config):
            started.set()
            list(audio_generator)
            finished.set()
            return iter(())

    monkeypatch.setattr(service, "_ensure_asr_service", lambda: BlockingASRService())
    monkeypatch.setattr(
        nvidia_stt.SpeechStream,
        "_create_streaming_config",
        lambda self: SimpleNamespace(),
    )

    stream = service.stream()
    stream.push_frame(_audio_frame(6))
    assert await asyncio.to_thread(started.wait, 1.0)

    await asyncio.wait_for(stream.aclose(), timeout=2.0)

    assert await asyncio.to_thread(finished.wait, 1.0)


def test_stt_model_listing_uses_proto_request_and_inference_mode() -> None:
    request: object | None = None

    class FakeStub:
        def GetRivaSpeechRecognitionConfig(self, value):
            nonlocal request
            request = value
            return SimpleNamespace(
                model_config=[
                    SimpleNamespace(
                        model_name="streaming-model",
                        parameters={"type": "online", "language_code": "en-US"},
                    ),
                    SimpleNamespace(
                        model_name="offline-model",
                        parameters={"type": "offline", "language_code": "en-US"},
                    ),
                ]
            )

    service = nvidia_stt.STT(api_key="test-key", inference_mode="offline")
    models = service.log_asr_models(SimpleNamespace(stub=FakeStub()))

    assert isinstance(request, nvidia_stt.RivaSpeechRecognitionConfigRequest)
    assert models == {"en-US": [{"model": ["offline-model"]}]}

    automatic = nvidia_stt.STT(api_key="test-key")
    assert automatic.log_asr_models(SimpleNamespace(stub=FakeStub())) == {
        "en-US": [
            {"model": ["streaming-model"]},
            {"model": ["offline-model"]},
        ]
    }


async def test_stt_stream_emits_transcript_events_from_nvidia_speech_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_stt.STT(api_key="test-key")
    monkeypatch.setattr(service, "_ensure_asr_service", lambda: SimpleNamespace())

    def fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=fake_create_task):
        stream = nvidia_stt.SpeechStream(
            stt=service,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            language="en-US",
        )

    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                is_final=True,
                alternatives=[SimpleNamespace(transcript="hello world", confidence=0.9)],
            )
        ]
    )

    stream._handle_response(response, event_loop=asyncio.get_running_loop())
    await asyncio.sleep(0)

    assert stream._event_ch.recv_nowait().type == nvidia_stt.stt.SpeechEventType.START_OF_SPEECH
    transcript_event = stream._event_ch.recv_nowait()
    assert transcript_event.type == nvidia_stt.stt.SpeechEventType.FINAL_TRANSCRIPT
    assert transcript_event.alternatives[0].text == "hello world"
    assert stream._event_ch.recv_nowait().type == nvidia_stt.stt.SpeechEventType.END_OF_SPEECH


def test_tts_passes_locked_nvidia_speech_synthesize_online_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    service = nvidia_tts.TTS(
        api_key="test-key",
        sample_rate=22050,
        audio_prompt_file="prompt.wav",
        quality=30,
        options={"encoding": 999},
    )

    assert service.provider == "NVIDIA Speech"

    class FakeSynthesisService:
        def synthesize_online(
            self,
            text,
            voice_name=None,
            language_code="en-US",
            encoding=nvidia_tts.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=44100,
            audio_prompt_file=None,
            audio_prompt_encoding=nvidia_tts.AudioEncoding.LINEAR_PCM,
            quality=20,
        ):
            captured["text"] = text
            captured["voice_name"] = voice_name
            captured["language_code"] = language_code
            captured["encoding"] = encoding
            captured["sample_rate_hz"] = sample_rate_hz
            captured["audio_prompt_file"] = audio_prompt_file
            captured["audio_prompt_encoding"] = audio_prompt_encoding
            captured["quality"] = quality
            return [SimpleNamespace(audio=b"abc")]

    monkeypatch.setattr(service, "_ensure_session", lambda: FakeSynthesisService())

    responses = list(service._synthesize(" hello ", service._opts))

    assert responses[0].audio == b"abc"
    assert captured["text"] == "hello"
    assert captured["voice_name"] == "Magpie-Multilingual.EN-US.Leo"
    assert captured["language_code"] == "en-US"
    assert captured["sample_rate_hz"] == 22050
    assert captured["encoding"] == nvidia_tts.AudioEncoding.LINEAR_PCM
    assert captured["audio_prompt_file"] == Path("prompt.wav")
    assert captured["quality"] == 30


def test_tts_translates_zero_shot_options_for_new_nvidia_speech_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    service = nvidia_tts.TTS(
        api_key="test-key",
        audio_prompt_file="prompt.wav",
        quality=30,
        options={"custom_dictionary": {"LiveKit": "live kit"}},
    )

    class FakeSynthesisService:
        def synthesize_online(
            self,
            text,
            voice_name=None,
            language_code="en-US",
            encoding=nvidia_tts.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=44100,
            zero_shot_audio_prompt_file=None,
            zero_shot_quality=20,
            custom_dictionary=None,
            custom_configuration=None,
        ):
            captured.update(locals())
            return [SimpleNamespace(audio=b"abc")]

    monkeypatch.setattr(service, "_ensure_session", lambda: FakeSynthesisService())

    list(service._synthesize("hello", service._opts))

    assert captured["zero_shot_audio_prompt_file"] == Path("prompt.wav")
    assert captured["zero_shot_quality"] == 30
    assert captured["custom_dictionary"] == {"LiveKit": "live kit"}


def test_tts_rejects_options_unsupported_by_installed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_tts.TTS(
        api_key="test-key",
        options={"custom_dictionary": {"LiveKit": "live kit"}},
    )

    class LegacySynthesisService:
        def synthesize_online(
            self,
            text,
            voice_name=None,
            language_code="en-US",
            encoding=nvidia_tts.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=44100,
        ):
            return []

    monkeypatch.setattr(service, "_ensure_session", lambda: LegacySynthesisService())

    with pytest.raises(ValueError, match="custom_dictionary"):
        service._synthesize("hello", service._opts)


def test_tts_uses_offline_rpc_for_batch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    service = nvidia_tts.TTS(
        api_key="test-key",
        inference_mode="offline",
        audio_prompt_file="prompt.wav",
        quality=30,
        options={"custom_configuration": {"pace": "fast"}},
    )

    class FakeSynthesisService:
        def synthesize(
            self,
            text,
            voice_name=None,
            language_code="en-US",
            encoding=nvidia_tts.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=22050,
            zero_shot_audio_prompt_file=None,
            zero_shot_quality=20,
            future=False,
            custom_configuration=None,
        ):
            captured.update(locals())
            return SimpleNamespace(audio=b"offline-audio")

    monkeypatch.setattr(service, "_ensure_session", lambda: FakeSynthesisService())

    responses = list(service._synthesize("hello", service._opts))

    assert responses[0].audio == b"offline-audio"
    assert captured["zero_shot_audio_prompt_file"] == Path("prompt.wav")
    assert captured["zero_shot_quality"] == 30
    assert captured["future"] is False
    assert captured["custom_configuration"] == {"pace": "fast"}


def test_tts_list_voices_splits_locales_and_deduplicates_voices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = nvidia_tts.TTS(api_key="test-key")
    model_config = SimpleNamespace(
        parameters={
            "language_code": "en-US, fr-FR, en-US",
            "voice_name": "Magpie-Multilingual",
            "subvoices": "Leo:Male,Sofia:Female",
        }
    )
    fake_service = SimpleNamespace(
        stub=SimpleNamespace(
            GetRivaSynthesisConfig=lambda request: SimpleNamespace(model_config=[model_config])
        )
    )
    monkeypatch.setattr(service, "_ensure_session", lambda: fake_service)

    assert service.list_voices() == {
        "en-US": {"voices": ["Magpie-Multilingual.Leo", "Magpie-Multilingual.Sofia"]},
        "fr-FR": {"voices": ["Magpie-Multilingual.Leo", "Magpie-Multilingual.Sofia"]},
    }


async def test_tts_stream_finishes_on_first_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    service = nvidia_tts.TTS(api_key="test-key")

    class FakeSynthesisService:
        def synthesize_online(
            self,
            text,
            voice_name=None,
            language_code="en-US",
            encoding=nvidia_tts.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=44100,
        ):
            return [SimpleNamespace(audio=b"\0\0" * 160)]

    monkeypatch.setattr(service, "_ensure_session", lambda: FakeSynthesisService())

    stream = service.stream()
    stream.push_text("Hello world.")
    stream.flush()

    await asyncio.wait_for(stream._task, timeout=2.0)
    await stream.aclose()


def test_nvidia_invalid_argument_is_non_retryable() -> None:
    error = TypeError("unexpected keyword argument")

    mapped = nvidia_tts._to_tts_api_error(error, operation="NVIDIA Speech TTS request")

    assert isinstance(mapped, APIStatusError)
    assert mapped.status_code == 400
    assert mapped.retryable is False


def test_nvidia_provider_cancellation_is_not_graceful_499() -> None:
    class CancelledRpcError(nvidia_tts.grpc.RpcError):
        def code(self):
            return nvidia_tts.grpc.StatusCode.CANCELLED

        def details(self):
            return "provider cancelled"

    mapped = nvidia_tts._to_tts_api_error(
        CancelledRpcError(), operation="NVIDIA Speech TTS request"
    )

    assert isinstance(mapped, APIConnectionError)
    assert mapped.retryable is True
