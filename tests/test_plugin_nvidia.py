# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import grpc
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions
from livekit.plugins.nvidia import stt as nvidia_stt, tts as nvidia_tts
from livekit.plugins.nvidia.utils import to_api_error

pytestmark = pytest.mark.unit


def _audio_frame(byte: int = 1) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes([byte, byte]) * 160,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


def _response(text="hello", *, final=True, words=(), confidence=0.9):
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                is_final=final,
                alternatives=[SimpleNamespace(transcript=text, confidence=confidence, words=words)],
            )
        ]
    )


def _mock_stt(monkeypatch, responses):
    channels = []

    def create_auth(**kwargs):
        channel = SimpleNamespace(closed=False)
        channel.close = lambda: setattr(channel, "closed", True)
        channels.append(channel)
        return SimpleNamespace(channel=channel)

    monkeypatch.setattr(nvidia_stt.auth, "create_riva_auth", create_auth)
    monkeypatch.setattr(
        nvidia_stt.riva.client,
        "ASRService",
        lambda auth: SimpleNamespace(streaming_response_generator=responses),
    )
    service = nvidia_stt.STT(api_key="test-key")
    service.test_channels = channels
    return service


class _RpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "provider failure"


@pytest.mark.parametrize(
    "mode,streaming,offline",
    [("auto", True, True), ("streaming", True, False), ("offline", False, True)],
)
async def test_stt_inference_modes(monkeypatch, mode, streaming, offline):
    service = nvidia_stt.STT(api_key="test-key", inference_mode=mode)
    assert service.capabilities.streaming == streaming
    assert service.capabilities.offline_recognize == offline
    assert service.capabilities.interim_results == streaming
    backend = SimpleNamespace(offline_recognize=lambda audio, config: _response())
    monkeypatch.setattr(service, "_ensure_asr_service", lambda: backend)
    if offline:
        event = await service.recognize([_audio_frame()])
        assert event.alternatives[0].text == "hello"
    else:
        with pytest.raises(ValueError, match="inference_mode='offline'"):
            await service.recognize([_audio_frame()])
    if not streaming:
        with pytest.raises(ValueError, match="inference_mode='streaming'"):
            service.stream()


def test_stt_recognition_and_endpointing_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        nvidia_stt.riva.client,
        "add_endpoint_parameters_to_config",
        lambda config, *args: captured.setdefault("endpointing", args),
    )
    monkeypatch.setattr(
        nvidia_stt.riva.client,
        "add_custom_configuration_to_config",
        lambda config, value: captured.setdefault("custom", value),
        raising=False,
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
        endpointing=nvidia_stt.EndpointingConfig(start_history=11),
        options={"custom_configuration": "enable_vad_endpointing:true"},
    )
    config = service._create_recognition_config(
        language=LanguageCode("en-US"), sample_rate=48000, audio_channel_count=2
    )
    assert (config.model, config.sample_rate_hertz, config.audio_channel_count) == (
        "parakeet-sortformer",
        48000,
        2,
    )
    assert config.profanity_filter and config.verbatim_transcripts
    assert config.diarization_config.max_speaker_count == 2
    assert config.speech_contexts[0].phrases == ["LiveKit"]
    assert config.speech_contexts[0].boost == 7.0
    service._apply_streaming_config_extensions(SimpleNamespace())
    assert captured == {
        "endpointing": (11, -1.0, 500, 240, -1.0, -1.0),
        "custom": "enable_vad_endpointing:true",
    }
    monkeypatch.delattr(nvidia_stt.riva.client, "add_custom_configuration_to_config")
    with pytest.raises(ValueError, match="custom_configuration"):
        service._apply_streaming_config_extensions(SimpleNamespace())


def test_offline_results_combine_hypotheses_and_zero_confidence():
    response = SimpleNamespace(
        results=[
            _response("hello", confidence=0.0).results[0],
            _response("world", confidence=0.9).results[0],
        ]
    )
    response.results[0].alternatives.append(SimpleNamespace(transcript="yellow", words=[]))
    response.results[1].alternatives.append(SimpleNamespace(transcript="word", words=[]))
    event = nvidia_stt._response_to_speech_event(
        response,
        language=LanguageCode("en-US"),
        request_id="test",
        event_type=nvidia_stt.stt.SpeechEventType.FINAL_TRANSCRIPT,
        enable_diarization=False,
        is_final=True,
    )
    assert [a.text for a in event.alternatives] == ["hello world", "yellow word"]
    assert event.alternatives[0].confidence == pytest.approx(0.45)


def _record_attempts(monkeypatch):
    attempts = []
    original_start = nvidia_stt.SpeechStream._start_recognition_attempt

    def start(self, config, loop, **kwargs):
        attempt = original_start(self, config, loop, **kwargs)
        attempts.append(attempt)
        return attempt

    monkeypatch.setattr(nvidia_stt.SpeechStream, "_start_recognition_attempt", start)
    return attempts


@pytest.mark.parametrize("retry", [False, True], ids=["flush", "retry"])
@pytest.mark.parametrize("final", [False, True], ids=["interim", "final"])
async def test_stt_rpc_timestamps_and_replay(monkeypatch, retry, final):
    calls = []

    def responses(audio, config):
        calls.append(list(audio))
        if retry and len(calls) == 1:
            raise APIConnectionError("dropped")
        yield _response(
            words=[SimpleNamespace(word="hello", start_time=100, end_time=500, speaker_tag=1)],
            final=final,
        )

    attempts = _record_attempts(monkeypatch)
    service = _mock_stt(monkeypatch, responses)
    stream = service.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    try:
        stream.start_time_offset = 7.25
        stream.push_frame(_audio_frame(1))
        if not retry:
            stream.flush()
            stream.push_frame(_audio_frame(2))
        stream.end_input()
        await asyncio.wait_for(stream._task, 2)
        transcripts = [e.alternatives[0] async for e in stream if e.alternatives]
        assert len(calls) == 2
        assert len(transcripts) == (1 if retry else 2)
        assert calls == [
            [_audio_frame(1).data.tobytes()],
            [_audio_frame(1 if retry else 2).data.tobytes()],
        ]
        offsets = [attempt.start_time_offset for attempt in attempts]
        if retry:
            assert offsets[0] == offsets[1]
        else:
            assert offsets[1] - offsets[0] == pytest.approx(0.01, abs=0.001)
        for speech, offset in zip(transcripts, offsets[-len(transcripts) :], strict=True):
            assert speech.words[0].start_time == pytest.approx(offset + 0.1)
            assert speech.words[0].end_time == pytest.approx(offset + 0.5)
    finally:
        await stream.aclose()
    assert all(channel.closed for channel in service.test_channels)


@pytest.mark.parametrize("termination", ["error", "clean"])
async def test_stt_reconnects_without_flush(monkeypatch, termination):
    calls = []
    restarted = threading.Event()

    def responses(audio, config):
        chunks = [next(audio)]
        calls.append(chunks)
        if len(calls) == 1:
            if termination == "error":
                raise APIConnectionError("dropped")
            return
        restarted.set()
        yield _response()
        chunks.extend(audio)

    stream = _mock_stt(monkeypatch, responses).stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    )
    try:
        stream.push_frame(_audio_frame(1))
        assert await asyncio.to_thread(restarted.wait, 1)
        stream.push_frame(_audio_frame(2))
        stream.end_input()
        await asyncio.wait_for(stream._task, 2)
        events = [e async for e in stream]
        assert [e.type for e in events] == [
            nvidia_stt.stt.SpeechEventType.START_OF_SPEECH,
            nvidia_stt.stt.SpeechEventType.FINAL_TRANSCRIPT,
            nvidia_stt.stt.SpeechEventType.END_OF_SPEECH,
        ]
        assert calls[1] == [_audio_frame(1).data.tobytes(), _audio_frame(2).data.tobytes()]
    finally:
        await stream.aclose()


async def test_stt_clean_end_after_final_continues_on_a_new_rpc(monkeypatch):
    calls = []
    first_done = threading.Event()

    def responses(audio, config):
        chunks = [next(audio)]
        calls.append(chunks)
        yield _response(text=f"part {len(calls)}")
        if len(calls) == 1:
            first_done.set()
            return
        chunks.extend(audio)

    attempts = _record_attempts(monkeypatch)
    stream = _mock_stt(monkeypatch, responses).stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        stream.push_frame(_audio_frame(1))
        assert await asyncio.to_thread(first_done.wait, 1)
        await asyncio.sleep(0.05)
        stream.push_frame(_audio_frame(2))
        stream.end_input()
        await asyncio.wait_for(stream._task, 2)
        texts = [e.alternatives[0].text async for e in stream if e.alternatives]
        assert texts == ["part 1", "part 2"]
        assert calls == [[_audio_frame(1).data.tobytes()], [_audio_frame(2).data.tobytes()]]
        assert attempts[1].start_time_offset == pytest.approx(attempts[0].start_time_offset + 0.01)
    finally:
        await stream.aclose()


@pytest.mark.parametrize("termination", ["error", "clean", "flush-error"])
async def test_stt_retry_budget(monkeypatch, termination):
    calls = []

    def responses(audio, config):
        calls.append(list(audio) if termination == "flush-error" else [next(audio)])
        if termination != "clean":
            raise APIConnectionError("dropped")
        return iter(())

    stream = _mock_stt(monkeypatch, responses).stream(
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0)
    )
    try:
        stream.push_frame(_audio_frame())
        if termination == "flush-error":
            stream.end_input()
        with pytest.raises(APIError):
            await asyncio.wait_for(stream._task, 2)
        assert len(calls) == 3
        # every attempt replays the audio that was not transcribed
        assert all(call == [_audio_frame().data.tobytes()] for call in calls)
    finally:
        await stream.aclose()


@pytest.mark.parametrize("termination", ["error", "clean", "flush-error"])
async def test_stt_replays_recent_audio_after_buffer_overflow(monkeypatch, termination):
    calls = []
    received = threading.Event()
    release = threading.Event()

    def responses(audio, config):
        chunks = []
        calls.append(chunks)
        if len(calls) == 1:
            for _ in range(3):
                chunks.append(next(audio))
            received.set()
            assert release.wait(2)
            if termination == "flush-error":
                chunks.extend(audio)
            if termination != "clean":
                raise APIConnectionError("dropped")
            return
        chunks.append(next(audio))
        yield _response(words=[SimpleNamespace(word="hello", start_time=0, end_time=10)])
        chunks.extend(audio)

    monkeypatch.setattr(nvidia_stt, "_MAX_REPLAY_AUDIO_SECONDS", 0.02)
    attempts = _record_attempts(monkeypatch)
    stream = _mock_stt(monkeypatch, responses).stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    )
    try:
        for byte in (1, 2, 3):
            stream.push_frame(_audio_frame(byte))
        assert await asyncio.to_thread(received.wait, 1)
        if termination == "flush-error":
            stream.end_input()
        release.set()
        if termination != "flush-error":
            event = await asyncio.wait_for(anext(stream), 2)
            assert event.type == nvidia_stt.stt.SpeechEventType.START_OF_SPEECH
            stream.end_input()
        await asyncio.wait_for(stream._task, 2)
        transcripts = [e async for e in stream if e.alternatives]
        assert calls[1] == [_audio_frame(2).data.tobytes(), _audio_frame(3).data.tobytes()]
        offsets = [attempt.start_time_offset for attempt in attempts]
        assert offsets[1] == pytest.approx(offsets[0] + 0.01)
        assert transcripts[0].alternatives[0].start_time == pytest.approx(offsets[1])
    finally:
        release.set()
        await stream.aclose()


async def test_stt_cancellation_stops_worker(monkeypatch):
    started, finished = threading.Event(), threading.Event()

    def responses(audio, config):
        started.set()
        list(audio)
        finished.set()
        return iter(())

    service = _mock_stt(monkeypatch, responses)
    stream = service.stream()
    stream.push_frame(_audio_frame())
    assert await asyncio.to_thread(started.wait, 1)
    await asyncio.wait_for(stream.aclose(), 2)
    assert await asyncio.to_thread(finished.wait, 1)
    assert all(channel.closed for channel in service.test_channels)


async def test_provider_result_resets_retries_without_transcript(monkeypatch):
    stream = _mock_stt(monkeypatch, lambda audio, config: iter(())).stream()
    try:
        stream._num_retries = 2
        attempt = SimpleNamespace(abandoned=False, start_time_offset=0.0)
        stream._handle_response(
            SimpleNamespace(
                results=[SimpleNamespace(alternatives=[SimpleNamespace(transcript="")])]
            ),
            attempt,
        )
        assert stream._num_retries == 0
    finally:
        await stream.aclose()


@pytest.mark.parametrize("is_final", [False, True])
@pytest.mark.parametrize("offset", [0.0, 7.25])
async def test_word_timestamps_use_seconds(monkeypatch, is_final, offset):
    def responses(audio, config):
        list(audio)
        yield SimpleNamespace(
            results=[
                SimpleNamespace(
                    is_final=is_final,
                    alternatives=[
                        SimpleNamespace(
                            transcript="hello world",
                            words=[
                                SimpleNamespace(word="hello", start_time=42120, end_time=42560),
                                SimpleNamespace(word="world", start_time=44240, end_time=44480),
                            ],
                        )
                    ],
                )
            ]
        )

    provider = _mock_stt(monkeypatch, responses)
    async with provider, provider.stream() as stream:
        stream.start_time_offset = offset
        stream.end_input()
        await asyncio.wait_for(stream._task, 1)
        event_type = (
            nvidia_stt.stt.SpeechEventType.FINAL_TRANSCRIPT
            if is_final
            else nvidia_stt.stt.SpeechEventType.INTERIM_TRANSCRIPT
        )
        transcripts = [event async for event in stream if event.type == event_type]

    assert len(transcripts) == 1
    speech = transcripts[0].alternatives[0]
    assert [word.start_time for word in speech.words] == pytest.approx(
        [42.12 + offset, 44.24 + offset]
    )
    assert [word.end_time for word in speech.words] == pytest.approx(
        [42.56 + offset, 44.48 + offset]
    )


async def test_stt_replay_buffer_rolls_and_resets_after_final():
    attempt = nvidia_stt._RecognitionAttempt(
        audio_queue=queue.Queue(),
        done_fut=asyncio.get_running_loop().create_future(),
        start_time_offset=2,
        bytes_per_second=2,
    )
    attempt.buffer_for_replay(b"1234", max_bytes=6)
    attempt.buffer_for_replay(b"5678", max_bytes=6)
    assert attempt.replay_snapshot() == ([b"34", b"5678"], 3, False)
    attempt.mark_final_transcript()
    assert attempt.replay_snapshot() == ([], 6, True)
    attempt.buffer_for_replay(b"abcdefgh", max_bytes=6)
    assert attempt.replay_snapshot() == ([b"cdefgh"], 7, False)


@pytest.mark.parametrize("api_key", [NOT_GIVEN, None, "", "explicit"])
def test_tts_api_key_fallback(monkeypatch, api_key):
    monkeypatch.setenv("NVIDIA_API_KEY", "environment-key")
    assert nvidia_tts.TTS(api_key=api_key).nvidia_api_key == (
        "explicit" if api_key == "explicit" else "environment-key"
    )
    monkeypatch.delenv("NVIDIA_API_KEY")
    if api_key != "explicit":
        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            nvidia_tts.TTS(api_key=api_key)
        assert not nvidia_tts.TTS(api_key=api_key, use_ssl=False).nvidia_api_key


@pytest.mark.parametrize(
    "voice,family",
    [("Magpie-Multilingual.EN-US.Leo", "Magpie-Multilingual"), ("Chatterbox", "Chatterbox")],
)
def test_tts_model_family(voice, family):
    assert nvidia_tts.TTS(api_key="test", voice=voice).model == family


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_tts_installed_client_signature(monkeypatch, mode):
    service = nvidia_tts.TTS(
        api_key="test",
        inference_mode=mode,
        audio_prompt_file="prompt.wav",
        quality=30,
        options={"encoding": 999},
    )
    method = "synthesize_online" if mode == "online" else "synthesize"
    original = getattr(nvidia_tts.riva.client.SpeechSynthesisService, method)
    stub = getattr(
        create_autospec(nvidia_tts.riva.client.SpeechSynthesisService, instance=True), method
    )
    response = SimpleNamespace(audio=b"audio")
    stub.return_value = [response] if mode == "online" else response
    monkeypatch.setattr(service, "_ensure_session", lambda: SimpleNamespace(**{method: stub}))
    assert list(service._synthesize(" hello ", service._opts))[0].audio == b"audio"
    args, kwargs = stub.call_args
    assert args == ("hello", "Magpie-Multilingual.EN-US.Leo", "en-US")
    assert kwargs["encoding"] == nvidia_tts.AudioEncoding.LINEAR_PCM
    parameters = inspect.signature(original).parameters
    prompt = (
        "zero_shot_audio_prompt_file"
        if "zero_shot_audio_prompt_file" in parameters
        else "audio_prompt_file"
    )
    quality = "zero_shot_quality" if "zero_shot_quality" in parameters else "quality"
    assert kwargs[prompt] == Path("prompt.wav") and kwargs[quality] == 30


@pytest.mark.parametrize("current", [False, True], ids=["legacy", "current"])
def test_tts_client_option_translation(monkeypatch, current):
    captured = {}

    def legacy(text, voice, language, *, sample_rate_hz, encoding, audio_prompt_file, quality):
        captured.update(prompt=audio_prompt_file, quality=quality)
        return iter(())

    def modern(
        text,
        voice,
        language,
        *,
        sample_rate_hz,
        encoding,
        zero_shot_audio_prompt_file,
        zero_shot_quality,
        custom_dictionary,
    ):
        captured.update(
            prompt=zero_shot_audio_prompt_file,
            quality=zero_shot_quality,
            dictionary=custom_dictionary,
        )
        return iter(())

    options = {"custom_dictionary": {"LiveKit": "live kit"}}
    service = nvidia_tts.TTS(
        api_key="test",
        audio_prompt_file="prompt.wav",
        quality=30,
        options=options if current else {},
    )
    monkeypatch.setattr(
        service,
        "_ensure_session",
        lambda: SimpleNamespace(synthesize_online=modern if current else legacy),
    )
    list(service._synthesize("hello", service._opts))
    assert captured["prompt"] == Path("prompt.wav") and captured["quality"] == 30
    if current:
        assert captured["dictionary"] == options["custom_dictionary"]
    else:
        service._opts.options = options
        with pytest.raises(ValueError, match="custom_dictionary"):
            service._synthesize("hello", service._opts)


@pytest.mark.parametrize("failure", [False, True], ids=["flush", "worker-error"])
async def test_tts_stream_completion_with_open_text_input(monkeypatch, failure):
    service = nvidia_tts.TTS(api_key="test")

    def synthesize(text, opts):
        if failure:
            raise TypeError("invalid synthesis option")
        return iter([SimpleNamespace(audio=b"\0\0" * 160)])

    monkeypatch.setattr(service, "_synthesize", synthesize)
    stream = service.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        stream.push_text(
            "This first sentence is long enough to synthesize. Another sentence follows. "
        )
        if failure:
            with pytest.raises(APIStatusError, match="invalid synthesis option"):
                await asyncio.wait_for(stream._task, 2)
        else:
            stream.flush()
            await asyncio.wait_for(stream._task, 2)
    finally:
        await stream.aclose()


async def test_tts_aborted_request_is_retried(monkeypatch):
    service = nvidia_tts.TTS(api_key="test")
    calls = []

    def synthesize(text, opts):
        calls.append(text)
        if len(calls) == 1:
            raise _RpcError(grpc.StatusCode.ABORTED)
        return iter([SimpleNamespace(audio=b"\0\0" * 160)])

    monkeypatch.setattr(service, "_synthesize", synthesize)
    async with service.synthesize(
        "hello", conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    ) as stream:
        assert [event async for event in stream]
    assert calls == ["hello", "hello"]


async def test_tts_cancellation_stops_waiting_worker(monkeypatch):
    started = threading.Event()
    finished = asyncio.Event()
    complete = nvidia_tts._complete_future

    def synthesize(text, opts):
        started.set()
        return iter(())

    def finish(fut, error):
        complete(fut, error)
        finished.set()

    service = nvidia_tts.TTS(api_key="test")
    monkeypatch.setattr(service, "_synthesize", synthesize)
    monkeypatch.setattr(nvidia_tts, "_complete_future", finish)
    stream = service.stream()
    try:
        stream.push_text(
            "This first sentence is long enough to synthesize. Another sentence follows. "
        )
        assert await asyncio.to_thread(started.wait, 1)
    finally:
        await asyncio.wait_for(stream.aclose(), 2)
    await asyncio.wait_for(finished.wait(), 1)


@pytest.mark.parametrize(
    "code,status,retryable",
    [
        (grpc.StatusCode.ABORTED, 503, True),
        (grpc.StatusCode.UNAVAILABLE, 503, True),
        (grpc.StatusCode.INVALID_ARGUMENT, 400, False),
        (grpc.StatusCode.UNAUTHENTICATED, 401, False),
        (grpc.StatusCode.CANCELLED, None, True),
        (grpc.StatusCode.DEADLINE_EXCEEDED, None, True),
    ],
)
def test_shared_speech_error_mapping(code, status, retryable):
    error = to_api_error(_RpcError(code), operation="test")
    assert error.retryable == retryable
    if status is not None:
        assert isinstance(error, APIStatusError) and error.status_code == status
    else:
        assert isinstance(
            error,
            APITimeoutError if code == grpc.StatusCode.DEADLINE_EXCEEDED else APIConnectionError,
        )
