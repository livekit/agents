# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import grpc
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, LanguageCode
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
    service = nvidia_stt.STT(api_key="test-key")
    monkeypatch.setattr(
        service,
        "_ensure_asr_service",
        lambda: SimpleNamespace(streaming_response_generator=responses),
    )
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


class _Clock:
    """Freeze the wall and monotonic clocks; each pushed frame advances them by its duration."""

    def __init__(self, monkeypatch) -> None:
        self.now = 1000.0
        monkeypatch.setattr(time, "time", lambda: self.now)
        monkeypatch.setattr(time, "perf_counter", lambda: self.now)

    def push(self, stream, byte: int) -> None:
        self.now += 0.01
        stream.push_frame(_audio_frame(byte))


def _word(start_ms: int, end_ms: int):
    return SimpleNamespace(word="hello", start_time=start_ms, end_time=end_ms, speaker_tag=1)


def _pcm(*frame_bytes: int) -> list[bytes]:
    return [_audio_frame(byte).data.tobytes() for byte in frame_bytes]


def _word_times(event) -> tuple[float, float]:
    word = event.alternatives[0].words[0]
    return word.start_time, word.end_time


@pytest.mark.no_concurrent
@pytest.mark.parametrize("final", [False, True], ids=["interim", "final"])
async def test_stt_rpc_timestamps_restart_at_flush(monkeypatch, final):
    """Each flushed RPC counts from its first frame, and the old RPC's audio stays there."""
    clock = _Clock(monkeypatch)
    calls = []
    first_rpc_closed, release = threading.Event(), threading.Event()

    def responses(audio, config):
        chunks = list(audio)
        calls.append(chunks)
        if len(calls) == 1:
            first_rpc_closed.set()
            assert release.wait(2)
        # the word is this RPC's last frame
        end_ms = len(chunks) * 10
        yield _response(words=[_word(end_ms - 10, end_ms)], final=final)

    stream = _mock_stt(monkeypatch, responses).stream()
    try:
        stream.start_time_offset = 7.0
        clock.push(stream, 1)  # session [7.00, 7.01]
        stream.flush()
        assert await asyncio.to_thread(first_rpc_closed.wait, 1)
        # the first RPC is closing: audio pushed now waits for the next one
        clock.now += 5.0
        clock.push(stream, 2)  # session [12.01, 12.02]
        clock.push(stream, 3)  # session [12.02, 12.03]
        stream.end_input()
        await asyncio.sleep(0)
        assert stream._input_ch.qsize() == 3  # two frames and the last flush
        release.set()
        await asyncio.wait_for(stream._task, 2)
        transcripts = [e async for e in stream if e.alternatives]
        assert calls == [_pcm(1), _pcm(2, 3)]
        # the first RPC's delayed response keeps its own clock
        assert [_word_times(e) for e in transcripts] == [
            pytest.approx((7.0, 7.01), abs=1e-6),
            pytest.approx((12.02, 12.03), abs=1e-6),
        ]
        speech = transcripts[1].alternatives[0]
        assert (speech.start_time, speech.end_time) == pytest.approx((12.02, 12.03), abs=1e-6)
    finally:
        release.set()
        await stream.aclose()


@pytest.mark.no_concurrent
@pytest.mark.parametrize("termination", ["error", "clean"])
async def test_stt_next_rpc_starts_at_unread_input_without_replay(monkeypatch, termination):
    """An RPC that ends mid-stream loses the audio it took; the next starts at unread input."""
    clock = _Clock(monkeypatch)
    calls = []
    sleeps = []
    took_audio, release = threading.Event(), threading.Event()
    backoff, resume = asyncio.Event(), asyncio.Event()

    def responses(audio, config):
        chunks = []
        calls.append(chunks)
        chunks.append(next(audio))
        if len(calls) == 1:
            chunks.append(next(audio))
            took_audio.set()
            assert release.wait(2)
            if termination == "error":
                raise APIConnectionError("dropped")
            yield _response("first", words=[_word(0, 10)])
            return
        yield _response("second", words=[_word(0, 10)])
        chunks.extend(audio)

    async def sleep(delay, result=None):
        sleeps.append(delay)
        backoff.set()
        await resume.wait()

    stream = _mock_stt(monkeypatch, responses).stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    )
    restarted = asyncio.Event()
    restart_connection_clock = stream._restart_connection_clock

    def restart():
        restart_connection_clock()
        restarted.set()

    monkeypatch.setattr(stream, "_restart_connection_clock", restart)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    try:
        stream.start_time_offset = 7.0
        clock.push(stream, 1)  # session [7.00, 7.01]
        clock.push(stream, 2)  # taken by the first RPC, never transcribed
        assert await asyncio.to_thread(took_audio.wait, 1)
        release.set()
        if termination == "error":
            await asyncio.wait_for(backoff.wait(), 1)
        else:
            await asyncio.wait_for(restarted.wait(), 1)
        clock.now += 5.0
        clock.push(stream, 3)  # session [12.02, 12.03]
        clock.push(stream, 4)
        stream.end_input()
        if termination == "error":
            # nothing reads the input until the retry starts
            assert stream._input_ch.qsize() == 3
            assert sleeps == [0.1]
            resume.set()
        await asyncio.wait_for(stream._task, 2)
        events = [e async for e in stream]
        transcripts = [e for e in events if e.alternatives]
        speech_event = nvidia_stt.stt.SpeechEventType
        turn = [
            speech_event.START_OF_SPEECH,
            speech_event.FINAL_TRANSCRIPT,
            speech_event.END_OF_SPEECH,
        ]
        assert [e.type for e in events] == turn * len(transcripts)
        assert calls == [_pcm(1, 2), _pcm(3, 4)]
        assert _word_times(transcripts[-1]) == pytest.approx((12.02, 12.03), abs=1e-6)
        if termination == "clean":
            assert [e.alternatives[0].text for e in transcripts] == ["first", "second"]
            assert _word_times(transcripts[0]) == pytest.approx((7.0, 7.01), abs=1e-6)
    finally:
        release.set()
        resume.set()
        await stream.aclose()


@pytest.mark.no_concurrent
async def test_stt_keeps_input_queued_while_joining_an_ended_rpc(monkeypatch):
    clock = _Clock(monkeypatch)
    calls = []
    joining, release = asyncio.Event(), asyncio.Event()
    join_attempt = nvidia_stt._join_attempt

    async def join(attempt):
        if not joining.is_set():
            joining.set()
            await release.wait()
        await join_attempt(attempt)

    def responses(audio, config):
        calls.append([next(audio)])
        yield _response(words=[_word(0, 10)])
        if len(calls) > 1:
            calls[-1].extend(audio)

    monkeypatch.setattr(nvidia_stt, "_join_attempt", join)
    stream = _mock_stt(monkeypatch, responses).stream()
    try:
        stream.start_time_offset = 7.0
        clock.push(stream, 1)
        await asyncio.wait_for(joining.wait(), 1)
        clock.now += 5.0
        clock.push(stream, 2)
        for _ in range(5):
            await asyncio.sleep(0)
        assert stream._input_ch.qsize() == 1
        assert stream._input_clock.dequeued_duration == pytest.approx(0.01)
        release.set()
        transcripts = []
        while len(transcripts) < 2:
            event = await asyncio.wait_for(anext(stream), 1)
            if event.alternatives:
                transcripts.append(event)
        assert calls == [_pcm(1), _pcm(2)]
        assert _word_times(transcripts[0]) == pytest.approx((7.0, 7.01), rel=0, abs=1e-6)
        assert _word_times(transcripts[1]) == pytest.approx((12.01, 12.02), rel=0, abs=1e-6)
    finally:
        release.set()
        await stream.aclose()


@pytest.mark.no_concurrent
@pytest.mark.parametrize("termination", ["error", "clean"])
async def test_stt_retry_budget_and_backoff(monkeypatch, termination):
    calls = []
    sleeps = []

    def responses(audio, config):
        calls.append([next(audio)])
        if termination == "error":
            raise APIConnectionError("dropped")
        return iter(())

    stream = _mock_stt(monkeypatch, responses).stream(
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0.25)
    )

    async def sleep(delay, result=None):
        sleeps.append(delay)
        # each retry needs new audio: the failed RPC's audio is not sent again
        stream.push_frame(_audio_frame(len(sleeps) + 1))

    monkeypatch.setattr(asyncio, "sleep", sleep)
    try:
        stream.push_frame(_audio_frame(1))
        with pytest.raises(APIConnectionError, match="after 2 attempts"):
            await asyncio.wait_for(stream._task, 2)
        assert calls == [_pcm(1), _pcm(2), _pcm(3)]
        assert sleeps == [0.1, 0.25]
    finally:
        await stream.aclose()


async def test_stt_failed_last_flush_is_not_retried(monkeypatch):
    """No input is left for a retry, so it would report an empty flush as success."""
    calls = []

    def responses(audio, config):
        calls.append(list(audio))
        raise APIConnectionError("dropped")

    stream = _mock_stt(monkeypatch, responses).stream(
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0)
    )
    try:
        stream.push_frame(_audio_frame(1))
        stream.end_input()
        with pytest.raises(APIConnectionError, match="dropped") as exc:
            await asyncio.wait_for(stream._task, 2)
        assert not exc.value.retryable
        assert calls == [_pcm(1)]
    finally:
        await stream.aclose()


async def test_stt_failed_rpc_closes_open_speech(monkeypatch):
    """The interrupted speech is not resumed, so its turn ends with the RPC."""

    def responses(audio, config):
        next(audio)
        yield _response("hel", final=False)
        raise APIConnectionError("dropped")

    stream = _mock_stt(monkeypatch, responses).stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        stream.push_frame(_audio_frame(1))
        events = []
        with pytest.raises(APIConnectionError, match="dropped"):
            async for event in stream:
                events.append(event.type)
        speech_event = nvidia_stt.stt.SpeechEventType
        assert events == [
            speech_event.START_OF_SPEECH,
            speech_event.INTERIM_TRANSCRIPT,
            speech_event.END_OF_SPEECH,
        ]
    finally:
        await stream.aclose()


async def test_stt_drops_responses_of_an_abandoned_rpc(monkeypatch):
    """A worker that outlives its RPC cannot emit on a later RPC's clock."""
    stream = _mock_stt(monkeypatch, lambda audio, config: iter(())).stream()
    attempt = nvidia_stt._RecognitionAttempt(
        audio_queue=queue.Queue(),
        done_fut=asyncio.get_running_loop().create_future(),
        abandoned=True,
    )
    try:
        stream._handle_response(attempt, _response())
        stream.end_input()
        await asyncio.wait_for(stream._task, 2)
        assert [e async for e in stream] == []
        assert not attempt.received_final_transcript
    finally:
        await stream.aclose()


async def test_stt_cancellation_stops_worker(monkeypatch):
    started, finished = threading.Event(), threading.Event()

    def responses(audio, config):
        started.set()
        list(audio)
        finished.set()
        return iter(())

    stream = _mock_stt(monkeypatch, responses).stream()
    stream.push_frame(_audio_frame())
    assert await asyncio.to_thread(started.wait, 1)
    await asyncio.wait_for(stream.aclose(), 2)
    assert await asyncio.to_thread(finished.wait, 1)


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
